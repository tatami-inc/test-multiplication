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
    std::size_t c = 0;
    while (c < NC) {
        const std::size_t cend = c + std::min(block_size, NC - c);
        std::size_t r = 0;
        while (r < NR) {
            const std::size_t rend = r + std::min(line_size, NR - r);

            for (auto ccopy = c; ccopy < cend; ++ccopy) {
                const auto mult = rhs[ccopy];
                const auto& col = matrix[ccopy];
                for (auto rcopy = r; rcopy < rend; ++rcopy) {
                    product[rcopy] += mult * col[rcopy];
                }
            }

            r = rend;
        }
        c = cend;
    }
}

int main(int argc, char ** argv) {
    CLI::App app{"Dense column x single vector performance tests"};
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
    std::vector<std::vector<FLOAT> > matrix(NC);
    std::normal_distribution<> dist;
    for (std::size_t c = 0; c < NC; ++c) {
        matrix[c].resize(NR);
        for (auto& x : matrix[c]) {
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
        for (std::size_t c = 0; c < NC; ++c) {
            const auto mult = rhs[c];
            const auto& col = matrix[c];
            for (std::size_t r = 0; r < NR; ++r) {
                naive[r] += mult * col[r];
            }
        }
        return naive.front() + naive.back();
    });

    // Blocked multiplication.
    std::vector<FLOAT> blocked_4(NR);
    names.push_back("blocked 4");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult(NR, NC, matrix, rhs, blocked_4, 4);
        return blocked_4.front() + blocked_4.back();
    });

    std::vector<FLOAT> blocked_8(NR);
    names.push_back("blocked 8");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult(NR, NC, matrix, rhs, blocked_8, 8);
        return blocked_8.front() + blocked_8.back();
    });

    std::vector<FLOAT> blocked_16(NR);
    names.push_back("blocked 16");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult(NR, NC, matrix, rhs, blocked_16, 16);
        return blocked_16.front() + blocked_16.back();
    });

    std::vector<FLOAT> blocked_32(NR);
    names.push_back("blocked 32");
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
