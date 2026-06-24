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
    const std::vector<FLOAT>& rhs,
    std::vector<FLOAT>& product,
    std::size_t block_size
) {
    std::size_t r = 0;
    while (r < NR) {
        const std::size_t rend = r + std::min(block_size, NR - r);
        std::size_t c = 0;
        while (c < NC) {
            const std::size_t cend = c + std::min(block_size, NC - c);

            for (auto rcopy = r; rcopy < rend; ++rcopy) {
                product[rcopy] = std::inner_product(
                    rhs.begin() + c,
                    rhs.begin() + cend,
                    matrix[rcopy].begin() + c,
                    product[rcopy]
                );
            }

            c = cend;
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
    funs.reserve(5);

    // Naive multiplication.
    std::vector<FLOAT> naive(NR);
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            naive[r] = std::inner_product(rhs.begin(), rhs.end(), matrix[r].begin(), 0.0);
        }
        return naive.front() + naive.back();
    });

    // Blocked multiplication.
    std::vector<FLOAT> blocked_4(NR);
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult(NR, NC, matrix, rhs, blocked_4, 4);
        return blocked_4.front() + blocked_4.back();
    });

    std::vector<FLOAT> blocked_8(NR);
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult(NR, NC, matrix, rhs, blocked_8, 8);
        return blocked_8.front() + blocked_8.back();
    });

    std::vector<FLOAT> blocked_16(NR);
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult(NR, NC, matrix, rhs, blocked_16, 16);
        return blocked_16.front() + blocked_16.back();
    });

    std::vector<FLOAT> blocked_32(NR);
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult(NR, NC, matrix, rhs, blocked_32, 32);
        return blocked_32.front() + blocked_32.back();
    });

    // Performing the iterations.
    eztimer::Options opt;
    opt.iterations = iterations;
    opt.setup = [&]() -> void {
        std::fill(naive.begin(), naive.end(), 0);
        std::fill(blocked_4.begin(), blocked_4.end(), 0);
        std::fill(blocked_8.begin(), blocked_8.end(), 0);
        std::fill(blocked_16.begin(), blocked_16.end(), 0);
        std::fill(blocked_32.begin(), blocked_32.end(), 0);
    };

    std::optional<FLOAT> expected;
    auto res = eztimer::time<FLOAT>(
        funs,
        [&](const FLOAT& res, std::size_t i) -> void {
            if (expected.has_value()) {
#ifndef USE_SINGLE
                if (std::abs(*expected - res) > 1e-8) {
#else
                if (std::abs(*expected - res) > 1e-4) {
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
