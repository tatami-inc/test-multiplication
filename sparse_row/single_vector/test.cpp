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

int main(int argc, char ** argv) {
    CLI::App app{"Dense row matrix x sparse matrix performance tests"};
    std::size_t NR;
    app.add_option("-r,--row", NR, "Number of matrix rows")->default_val(10000);
    std::size_t NC;
    app.add_option("-c,--column", NC, "Number of matrix columns")->default_val(5000);
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
    std::vector<FLOAT> rhs(NC);
    for (std::size_t c = 0; c < NC; ++c) {
        rhs[c] = dist(rng);
    }

    std::vector<std::function<FLOAT()> > funs;
    std::vector<std::string> names;
    funs.reserve(20);
    names.reserve(20);

    // Naive multiplication.
    std::vector<FLOAT> naive(NR);
    names.emplace_back("naive");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            naive[r] = dense_sparse_dot_product<1>(mat_value[r], mat_index[r], rhs);
        }
        return naive.front() + naive.back();
    });

    // Multiple accumulators.
    std::vector<FLOAT> ma_2(NR);
    names.emplace_back("2 accumulators");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            ma_2[r] = dense_sparse_dot_product<2>(mat_value[r], mat_index[r], rhs);
        }
        return ma_2.front() + ma_2.back();
    });

    std::vector<FLOAT> ma_4(NR);
    names.emplace_back("4 accumulators");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            ma_4[r] = dense_sparse_dot_product<4>(mat_value[r], mat_index[r], rhs);
        }
        return ma_4.front() + ma_4.back();
    });

    std::vector<FLOAT> ma_8(NR);
    names.emplace_back("8 accumulators");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            ma_8[r] = dense_sparse_dot_product<4>(mat_value[r], mat_index[r], rhs);
        }
        return ma_8.front() + ma_8.back();
    });

    // Performing the iterations.
    eztimer::Options opt;
    opt.iterations = iterations;

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
        if (name.size() < 50) {
            name.insert(name.end(), 50 - name.size(), ' ');
        }
        std::cout << name << ": " << res[m].mean.count() << " ± " << res[m].sd.count() / std::sqrt(res[m].times.size()) << std::endl;
    }

    return 0;
}
