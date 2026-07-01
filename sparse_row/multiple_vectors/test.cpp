#include <random>
#include <vector>
#include <iostream>
#include <cstddef>
#include <array>

#include "CLI/CLI.hpp"
#include "eztimer/eztimer.hpp"

#ifndef USE_SINGLE
#define FLOAT double
#else
#define FLOAT float
#endif

#include "dense_sparse_dot_product.h"

template<int num_acc_>
void blocked_mult(
    const std::size_t NR,
    const std::size_t NC,
    const std::vector<std::vector<FLOAT> >& mat_value,
    const std::vector<std::vector<int> >& mat_index,
    const std::size_t NRHS,
    const std::vector<std::vector<FLOAT> >& rhs,
    std::vector<std::vector<FLOAT> >& product,
    const std::size_t block_size 
) {
    std::size_t r = 0;
    while (r < NR) {
        const std::size_t rend = r + std::min(NR - r, block_size);
        for (std::size_t h = 0; h < NRHS; ++h) {
            const auto& rcol = rhs[h];
            for (std::size_t rcopy = r; rcopy < rend; ++rcopy) {
                const auto& mval = mat_value[rcopy];
                const auto& midx = mat_index[rcopy];
                product[h][rcopy] = dense_sparse_dot_product<num_acc_>(mat_value[rcopy], mat_index[rcopy], rcol);
            }
        }
        r = rend;
    }
}

int main(int argc, char ** argv) {
    CLI::App app{"Dense row matrix x sparse matrix performance tests"};
    std::size_t NR;
    app.add_option("-r,--row", NR, "Number of matrix rows")->default_val(10000);
    std::size_t NC;
    app.add_option("-c,--column", NC, "Number of matrix columns")->default_val(5000);
    std::size_t NRHS;
    app.add_option("-H,--rhs", NRHS, "Number of right-hand-side matrix columns")->default_val(100);
    int iterations;
    app.add_option("-i,--iter", iterations, "Number of iterations")->default_val(10);
    unsigned long long seed;
    app.add_option("-s,--seed", seed, "Seed for the simulated data")->default_val(69);
    CLI11_PARSE(app, argc, argv);

    // Simulating a row-major sparse matrix.
    std::mt19937_64 rng(seed);
    std::vector<std::vector<FLOAT> > mat_value(NR);
    std::vector<std::vector<int> > mat_index(NR);
    std::uniform_real_distribution<> udist;
    std::normal_distribution<> dist;
    for (std::size_t r = 0; r < NR; ++r) {
        for (std::size_t c = 0; c < NC; ++c) {
            if (udist(rng) <= 0.1) {
                mat_index[r].push_back(c);
                mat_value[r].push_back(dist(rng));
            }
        }
        mat_index[r].shrink_to_fit();
        mat_value[r].shrink_to_fit();
    }

    // Simulating the RHS vector.
    std::vector<std::vector<FLOAT> > rhs(NRHS);
    for (std::size_t h = 0; h < NRHS; ++h) {
        rhs[h].resize(NC);
        for (auto& x : rhs[h]) {
            x = dist(rng);
        }
    }

    auto preallocate_output = [&]() -> auto {
        std::vector<std::vector<FLOAT> > output(NRHS);
        for (std::size_t h = 0; h < NRHS; ++h) {
            output[h].resize(NR);
        }
        return output;
    };

    auto reset_output = [&](std::vector<std::vector<FLOAT> >& output) -> void {
        for (std::size_t h = 0; h < NRHS; ++h) {
            std::fill(output[h].begin(), output[h].end(), 0);
        }
    };

    std::vector<std::function<FLOAT()> > funs;
    std::vector<std::string> names;
    funs.reserve(20);
    names.reserve(20);

    // Naive multiplication.
    auto naive = preallocate_output();
    names.emplace_back("naive");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            for (std::size_t h = 0; h < NRHS; ++h) {
                naive[h][r] = dense_sparse_dot_product<1>(mat_value[r], mat_index[r], rhs[h]);
            }
        }
        return naive.front().front() + naive.front().back() + naive.back().front() + naive.back().back();
    });

    // Multiple accumulators.
    auto ma_2 = preallocate_output();
    names.emplace_back("2 accumulators");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            for (std::size_t h = 0; h < NRHS; ++h) {
                ma_2[h][r] = dense_sparse_dot_product<2>(mat_value[r], mat_index[r], rhs[h]);
            }
        }
        return ma_2.front().front() + ma_2.front().back() + ma_2.back().front() + ma_2.back().back();
    });

    auto ma_4 = preallocate_output();
    names.emplace_back("4 accumulators");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            for (std::size_t h = 0; h < NRHS; ++h) {
                ma_4[h][r] = dense_sparse_dot_product<4>(mat_value[r], mat_index[r], rhs[h]);
            }
        }
        return ma_4.front().front() + ma_4.front().back() + ma_4.back().front() + ma_4.back().back();
    });

    auto ma_8 = preallocate_output();
    names.emplace_back("8 accumulators");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            for (std::size_t h = 0; h < NRHS; ++h) {
                ma_8[h][r] = dense_sparse_dot_product<8>(mat_value[r], mat_index[r], rhs[h]);
            }
        }
        return ma_8.front().front() + ma_8.front().back() + ma_8.back().front() + ma_8.back().back();
    });

    // Blocked.
    auto blocked_4 = preallocate_output();
    names.emplace_back("blocked 4");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult<1>(NR, NC, mat_value, mat_index, NRHS, rhs, blocked_4, 4);
        return blocked_4.front().front() + blocked_4.front().back() + blocked_4.back().front() + blocked_4.back().back();
    });

    auto blocked_8 = preallocate_output();
    names.emplace_back("blocked 8");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult<1>(NR, NC, mat_value, mat_index, NRHS, rhs, blocked_8, 8);
        return blocked_8.front().front() + blocked_8.front().back() + blocked_8.back().front() + blocked_8.back().back();
    });

    auto blocked_16 = preallocate_output();
    names.emplace_back("blocked 16");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult<1>(NR, NC, mat_value, mat_index, NRHS, rhs, blocked_16, 16);
        return blocked_16.front().front() + blocked_16.front().back() + blocked_16.back().front() + blocked_16.back().back();
    });

    auto blocked_32 = preallocate_output();
    names.emplace_back("blocked 32");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult<1>(NR, NC, mat_value, mat_index, NRHS, rhs, blocked_32, 32);
        return blocked_32.front().front() + blocked_32.front().back() + blocked_32.back().front() + blocked_32.back().back();
    });

    // Blocked with multiple accumulators.
    auto ma_blocked_4 = preallocate_output();
    names.emplace_back("blocked 4, multiple accumulators");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult<4>(NR, NC, mat_value, mat_index, NRHS, rhs, ma_blocked_4, 4);
        return ma_blocked_4.front().front() + ma_blocked_4.front().back() + ma_blocked_4.back().front() + ma_blocked_4.back().back();
    });

    auto ma_blocked_8 = preallocate_output();
    names.emplace_back("blocked 8, multiple accumulators");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult<4>(NR, NC, mat_value, mat_index, NRHS, rhs, ma_blocked_8, 8);
        return ma_blocked_8.front().front() + ma_blocked_8.front().back() + ma_blocked_8.back().front() + ma_blocked_8.back().back();
    });

    auto ma_blocked_16 = preallocate_output();
    names.emplace_back("blocked 16, multiple accumulators");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult<4>(NR, NC, mat_value, mat_index, NRHS, rhs, ma_blocked_16, 16);
        return ma_blocked_16.front().front() + ma_blocked_16.front().back() + ma_blocked_16.back().front() + ma_blocked_16.back().back();
    });

    auto ma_blocked_32 = preallocate_output();
    names.emplace_back("blocked 32, multiple accumulators");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult<4>(NR, NC, mat_value, mat_index, NRHS, rhs, ma_blocked_32, 32);
        return ma_blocked_32.front().front() + ma_blocked_32.front().back() + ma_blocked_32.back().front() + ma_blocked_32.back().back();
    });

    // Performing the iterations.
    eztimer::Options opt;
    opt.iterations = iterations;
    opt.setup = [&]() -> void {
        reset_output(blocked_4);
        reset_output(blocked_8);
        reset_output(blocked_16);
        reset_output(blocked_32);
        reset_output(ma_blocked_4);
        reset_output(ma_blocked_8);
        reset_output(ma_blocked_16);
        reset_output(ma_blocked_32);
    };

    std::optional<FLOAT> expected;
    auto res = eztimer::time<FLOAT>(
        funs,
        [&](const FLOAT& res, std::size_t i) -> void {
            std::cout << res << "\t" << i << std::endl;
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

    std::cout << "Results for " << NR << " x " << NC << " x " << NRHS << std::endl;
    const std::size_t num_methods = names.size();
    for (std::size_t m = 0; m < num_methods; ++m) {
        auto name = names[m];
        if (name.size() < 50) {
            name.insert(name.end(), 50 - name.size(), ' ');
        }
        std::cout << name << ": " << res[m].mean.count() << " ± " << res[m].sd.count() / std::sqrt(res[m].times.size()) << std::endl;
    }

    return 0;
}
