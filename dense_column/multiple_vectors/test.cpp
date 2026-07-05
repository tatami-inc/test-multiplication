#include <random>
#include <vector>
#include <iostream>
#include <cstddef>

#include "CLI/CLI.hpp"
#include "eztimer/eztimer.hpp"

#ifndef USE_SINGLE
#define FLOAT double
#else
#define FLOAT float
#endif

void blocked_mult(
    const std::size_t NR,
    const std::size_t NC,
    const std::vector<std::vector<FLOAT> >& matrix,
    const std::size_t NRHS,
    const std::vector<std::vector<FLOAT> >& rhs,
    std::vector<std::vector<FLOAT> >& product,
    std::size_t block_size
) {
    // Assuming that we can safely fit around 1000 elements in the cache.
    // Technically, the cache size is in bytes and the "ideal" number of elements would depend on sizeof(FLOAT).
    // However, we'll just keep it simple here, given that this cache size (in bytes or elements) is just a guess anyway. 
    const std::size_t line_size = 1024 / block_size;
    std::size_t c = 0;
    while (c < NC) {
        const std::size_t cend = c + std::min(block_size, NC - c);
        std::size_t h = 0;
        while (h < NRHS) {
            const std::size_t hend = h + std::min(block_size, NRHS - h);
            std::size_t r = 0;
            while (r < NR) {
                const std::size_t rend = r + std::min(line_size, NR - r);

                for (auto ccopy = c; ccopy < cend; ++ccopy) {
                    const auto& col = matrix[ccopy];
                    for (auto hcopy = h; hcopy < hend; ++hcopy) {
                        const auto mult = rhs[hcopy][ccopy];
                        auto& out = product[hcopy];
                        for (auto rcopy = r; rcopy < rend; ++rcopy) {
                            out[rcopy] += mult * col[rcopy];
                        }
                    }
                }

                r = rend;
            }
            h = hend;
        }
        c = cend;
    }
}

int main(int argc, char ** argv) {
    CLI::App app{"Timings for dense column-major LHS, multiple vectors RHS"};
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

    // Simulating a column-major matrix.
    std::mt19937_64 rng(seed);
    std::vector<std::vector<FLOAT> > matrix(NC);
    std::normal_distribution<> dist;
    for (std::size_t c = 0; c < NC; ++c) {
        matrix[c].resize(NR);
        for (auto& x : matrix[c]) {
            x = dist(rng);
        }
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
    funs.reserve(5);
    std::vector<std::string> names;
    names.reserve(5);

    // Naive multiplication.
    auto naive = preallocate_output();
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t c = 0; c < NC; ++c) {
            const auto& col = matrix[c];
            for (std::size_t h = 0; h < NRHS; ++h) {
                auto& out = naive[h];
                const auto mult = rhs[h][c];
                for (std::size_t r = 0; r < NR; ++r) {
                    out[r] += mult * col[r];
                }
            }
        }
        return naive.front().front() + naive.front().back() + naive.back().front() + naive.back().back();
    });
    names.push_back("naive");

    // Blocked multiplication.
    auto blocked_4 = preallocate_output();
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult(NR, NC, matrix, NRHS, rhs, blocked_4, 4);
        return blocked_4.front().front() + blocked_4.front().back() + blocked_4.back().front() + blocked_4.back().back();
    });
    names.push_back("blocked (B = 4)");

    auto blocked_8 = preallocate_output();
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult(NR, NC, matrix, NRHS, rhs, blocked_8, 8);
        return blocked_8.front().front() + blocked_8.front().back() + blocked_8.back().front() + blocked_8.back().back();
    });
    names.push_back("blocked (B = 8)");

    auto blocked_16 = preallocate_output();
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult(NR, NC, matrix, NRHS, rhs, blocked_16, 16);
        return blocked_16.front().front() + blocked_16.front().back() + blocked_16.back().front() + blocked_16.back().back();
    });
    names.push_back("blocked (B = 16)");

    auto blocked_32 = preallocate_output();
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult(NR, NC, matrix, NRHS, rhs, blocked_32, 32);
        return blocked_32.front().front() + blocked_32.front().back() + blocked_32.back().front() + blocked_32.back().back();
    });
    names.push_back("blocked (B = 32)");

    // Performing the iterations.
    eztimer::Options opt;
    opt.iterations = iterations;
    opt.setup = [&]() -> void {
        reset_output(naive);
        reset_output(blocked_4);
        reset_output(blocked_8);
        reset_output(blocked_16);
        reset_output(blocked_32);
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
        if (name.size() < 40) {
            name.insert(name.end(), 40 - name.size(), ' ');
        }
        std::cout << name << ": " << res[m].mean.count() << " ± " << res[m].sd.count() / std::sqrt(res[m].times.size()) << std::endl;
    }

    return 0;
}
