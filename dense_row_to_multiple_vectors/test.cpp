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
    std::size_t r = 0;
    while (r < NR) {
        const std::size_t rend = r + std::min(block_size, NR - r);
        std::size_t h = 0;
        while (h < NRHS) {
            const std::size_t hend = h + std::min(block_size, NRHS - h);
            std::size_t c = 0;
            while (c < NC) {
                const std::size_t cend = c + std::min(block_size, NC - c);

                for (auto hcopy = h; hcopy < hend; ++hcopy) {
                    for (auto rcopy = r; rcopy < rend; ++rcopy) {
                        product[hcopy][rcopy] = std::inner_product(
                            rhs[hcopy].begin() + c,
                            rhs[hcopy].begin() + cend,
                            matrix[rcopy].begin() + c,
                            product[hcopy][rcopy]
                        );
                    }
                }

                c = cend;
            }
            h = hend;
        }
        r = rend;
    }
}

int main(int argc, char ** argv) {
    CLI::App app{"Log-normalization performance tests"};
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

    // Naive multiplication.
    auto naive = preallocate_output();
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            for (std::size_t h = 0; h < NRHS; ++h) {
                naive[h][r] = std::inner_product(rhs[h].begin(), rhs[h].end(), matrix[r].begin(), 0.0);
            }
        }
        return naive.front().front() + naive.front().back() + naive.back().front() + naive.back().back();
    });

    // Blocked multiplication.
    auto blocked_4 = preallocate_output();
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult(NR, NC, matrix, NRHS, rhs, blocked_4, 4);
        return blocked_4.front().front() + blocked_4.front().back() + blocked_4.back().front() + blocked_4.back().back();
    });

    auto blocked_8 = preallocate_output();
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult(NR, NC, matrix, NRHS, rhs, blocked_8, 8);
        return blocked_8.front().front() + blocked_8.front().back() + blocked_8.back().front() + blocked_8.back().back();
    });

    auto blocked_16 = preallocate_output();
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult(NR, NC, matrix, NRHS, rhs, blocked_16, 16);
        return blocked_16.front().front() + blocked_16.front().back() + blocked_16.back().front() + blocked_16.back().back();
    });

    auto blocked_32 = preallocate_output();
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult(NR, NC, matrix, NRHS, rhs, blocked_32, 32);
        return blocked_32.front().front() + blocked_32.front().back() + blocked_32.back().front() + blocked_32.back().back();
    });

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
                if (std::abs(*expected - res) > 1e-3) {
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
    std::cout << "naive:          " << res[0].mean.count() << " ± " << res[0].sd.count() / std::sqrt(res[0].times.size()) << std::endl;
    std::cout << "blocked (4):    " << res[1].mean.count() << " ± " << res[1].sd.count() / std::sqrt(res[1].times.size()) << std::endl;
    std::cout << "blocked (8):    " << res[2].mean.count() << " ± " << res[2].sd.count() / std::sqrt(res[2].times.size()) << std::endl;
    std::cout << "blocked (16):   " << res[3].mean.count() << " ± " << res[3].sd.count() / std::sqrt(res[3].times.size()) << std::endl;
    std::cout << "blocked (32):   " << res[4].mean.count() << " ± " << res[4].sd.count() / std::sqrt(res[4].times.size()) << std::endl;

    return 0;
}
