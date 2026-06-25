#include <random>
#include <vector>
#include <iostream>
#include <cstddef>
#include <algorithm>

#include "CLI/CLI.hpp"
#include "eztimer/eztimer.hpp"

#ifndef USE_SINGLE
#define FLOAT double
#else
#define FLOAT float
#endif

template<int num_acc_, class Iterator1_, class Iterator2_>
FLOAT super_dot_product(const std::size_t len, Iterator1_ start1, Iterator2_ start2) {
    const std::size_t cycles = len / num_acc_;
    std::array<FLOAT, num_acc_> dots{};
    std::size_t c = 0;
    for (std::size_t c0 = 0; c0 < cycles; ++c0) {
        for (std::size_t i = 0; i < num_acc_; ++i) {
            dots[i] += *(start1 + (c + i)) * *(start2 + (c + i));
        }
        c += num_acc_;
    }
    FLOAT extras = 0;
    for (; c < len; ++c) {
        extras += *(start1 + c) * *(start2 + c);
    }
    return std::accumulate(dots.begin(), dots.end(), extras);
}

template<int num_acc_>
void blocked_mult(
    const std::size_t NR,
    const std::size_t NC,
    const std::vector<std::vector<FLOAT> >& matrix,
    const std::vector<FLOAT>& rhs,
    std::vector<FLOAT>& product,
    std::size_t block_size
) {
    // Assuming that we can safely fit around 1000 elements in the cache.
    // Technically, the cache size is in bytes and the "ideal" number of elements would depend on sizeof(FLOAT).
    // However, we'll just keep it simple here, given that this cache size (in bytes or elements) is just a guess anyway. 
    const std::size_t line_size = 1024 / block_size;
    std::size_t r = 0;
    while (r < NR) {
        const std::size_t rend = r + std::min(block_size, NR - r);
        std::size_t c = 0;
        while (c < NC) {
            const std::size_t cnum = std::min(line_size, NC - c);
            const std::size_t cend = c + cnum;

            for (auto rcopy = r; rcopy < rend; ++rcopy) {
                if constexpr(num_acc_ == 1) {
                    product[rcopy] = std::inner_product(
                        rhs.begin() + c,
                        rhs.begin() + cend,
                        matrix[rcopy].begin() + c,
                        product[rcopy]
                    );
                } else {
                    product[rcopy] += super_dot_product<num_acc_>(
                        cnum,
                        rhs.begin() + c,
                        matrix[rcopy].begin() + c
                    );
                }
            }

            c = cend;
        }
        r = rend;
    }
}

int main(int argc, char ** argv) {
    CLI::App app{"Dense row matrix x single vector performance tests"};
    std::size_t NR;
    app.add_option("-r,--row", NR, "Number of matrix rows")->default_val(10000);
    std::size_t NC;
    app.add_option("-c,--column", NC, "Number of matrix columns")->default_val(5000);
    int iterations;
    app.add_option("-i,--iter", iterations, "Number of iterations")->default_val(10);
    unsigned long long seed;
    app.add_option("-s,--seed", seed, "Seed for the simulated data")->default_val(69);
    CLI11_PARSE(app, argc, argv);

    // Simulating a row-major matrix.
    std::mt19937_64 rng(seed);
    std::vector<std::vector<FLOAT> > matrix(NR);
    std::normal_distribution<> dist;
    for (std::size_t r = 0; r < NR; ++r) {
        matrix[r].resize(NC);
        for (auto& x : matrix[r]) {
            x = dist(rng);
        }
    }

    // Simulating the RHS vector.
    std::vector<FLOAT> rhs(NC);
    for (auto& x : rhs) {
        x = dist(rng);
    }

    std::vector<std::function<FLOAT()> > funs;
    funs.reserve(6);
    std::vector<std::string> names;
    names.reserve(6);

    // Naive multiplication.
    std::vector<FLOAT> naive(NR);
    names.push_back("naive");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            naive[r] = std::inner_product(rhs.begin(), rhs.end(), matrix[r].begin(), 0.0);
        }
        return naive.front() + naive.back();
    });

    std::vector<FLOAT> naive_two_acc(NR);
    names.push_back("naive, two accumulators");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            naive_two_acc[r] = super_dot_product<2>(NC, rhs.begin(), matrix[r].begin());
        }
        return naive_two_acc.front() + naive_two_acc.back();
    });

    std::vector<FLOAT> naive_four_acc(NR);
    names.push_back("naive, four accumulators");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            naive_four_acc[r] = super_dot_product<4>(NC, rhs.begin(), matrix[r].begin());
        }
        return naive_four_acc.front() + naive_four_acc.back();
    });

    std::vector<FLOAT> naive_eight_acc(NR);
    names.push_back("naive, eight accumulators");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            naive_eight_acc[r] = super_dot_product<8>(NC, rhs.begin(), matrix[r].begin());
        }
        return naive_eight_acc.front() + naive_eight_acc.back();
    });

    // Blocked multiplication.
    std::vector<FLOAT> blocked_4(NR);
    names.push_back("blocked 4");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult<1>(NR, NC, matrix, rhs, blocked_4, 4);
        return blocked_4.front() + blocked_4.back();
    });

    std::vector<FLOAT> blocked_8(NR);
    names.push_back("blocked 8");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult<1>(NR, NC, matrix, rhs, blocked_8, 8);
        return blocked_8.front() + blocked_8.back();
    });

    std::vector<FLOAT> blocked_16(NR);
    names.push_back("blocked 16");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult<1>(NR, NC, matrix, rhs, blocked_16, 16);
        return blocked_16.front() + blocked_16.back();
    });

    std::vector<FLOAT> blocked_32(NR);
    names.push_back("blocked 32");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult<1>(NR, NC, matrix, rhs, blocked_32, 32);
        return blocked_32.front() + blocked_32.back();
    });

    // Blocked multiplication with four accumulators.
    std::vector<FLOAT> ma_blocked_4(NR);
    names.push_back("blocked 4, four accumulators");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult<4>(NR, NC, matrix, rhs, ma_blocked_4, 4);
        return ma_blocked_4.front() + ma_blocked_4.back();
    });

    std::vector<FLOAT> ma_blocked_8(NR);
    names.push_back("blocked 8, four accumulators");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult<4>(NR, NC, matrix, rhs, ma_blocked_8, 8);
        return ma_blocked_8.front() + ma_blocked_8.back();
    });

    std::vector<FLOAT> ma_blocked_16(NR);
    names.push_back("blocked 16, four accumulators");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult<4>(NR, NC, matrix, rhs, ma_blocked_16, 16);
        return ma_blocked_16.front() + ma_blocked_16.back();
    });

    std::vector<FLOAT> ma_blocked_32(NR);
    names.push_back("blocked 32, four accumulators");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult<4>(NR, NC, matrix, rhs, ma_blocked_32, 32);
        return ma_blocked_32.front() + ma_blocked_32.back();
    });

    // Performing the iterations.
    eztimer::Options opt;
    opt.iterations = iterations;
    opt.setup = [&]() -> void {
        std::fill(blocked_4.begin(), blocked_4.end(), 0);
        std::fill(blocked_8.begin(), blocked_8.end(), 0);
        std::fill(blocked_16.begin(), blocked_16.end(), 0);
        std::fill(blocked_32.begin(), blocked_32.end(), 0);
        std::fill(ma_blocked_4.begin(), ma_blocked_4.end(), 0);
        std::fill(ma_blocked_8.begin(), ma_blocked_8.end(), 0);
        std::fill(ma_blocked_16.begin(), ma_blocked_16.end(), 0);
        std::fill(ma_blocked_32.begin(), ma_blocked_32.end(), 0);
    };

    std::optional<FLOAT> expected;
    auto res = eztimer::time<FLOAT>(
        funs,
        [&](const FLOAT& res, std::size_t i) -> void {
            if (expected.has_value()) {
#ifndef USE_SINGLE
                if (std::abs(*expected - res) > 1e-8) {
#else
                if (std::abs(*expected - res) > 1e-2) {
#endif
                    throw std::runtime_error("oops that's not right");
                }
            } else {
                expected = res;
            }
        },
        opt
    );

    std::cout << "Results for " << NR << " x " << NC << std::endl;
    const std::size_t num_methods = names.size();
    for (std::size_t m = 0; m < num_methods; ++m) {
        auto name = names[m];
        if (name.size() < 40) {
            name.insert(name.end(), 40 - name.size(), ' ');
        }
        std::cout << name << ": " << res[m].mean.count() << " ± " << res[m].sd.count() / std::sqrt(res[m].times.size()) << std::endl;
    }

    return 0;
}
