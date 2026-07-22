#include <vector>
#include <random>
#include <array>
#include <cstddef>
#include <iostream>
#include <optional>

#include "eztimer/eztimer.hpp"
#include "CLI/CLI.hpp"

#include "implementations.h"

int main(int argc, char ** argv) {
    CLI::App app{"Timings for dense row-major LHS, single vector RHS"};
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
    std::vector<std::vector<double> > matrix(NR);
    std::normal_distribution<> dist;
    for (std::size_t r = 0; r < NR; ++r) {
        matrix[r].resize(NC);
        for (auto& x : matrix[r]) {
            x = dist(rng);
        }
    }

    // Simulating the RHS vector.
    std::vector<double> rhs(NC);
    for (auto& x : rhs) {
        x = dist(rng);
    }

    std::vector<std::function<double()> > funs;
    funs.reserve(5);
    std::vector<std::string> names;
    names.reserve(5);

    // Setting up the outputs.
    std::vector<double> loop_res(NR);
    names.push_back("loop");
    funs.emplace_back([&]() -> double {
        for (std::size_t r = 0; r < NR; ++r) {
            loop_res[r] = loop(NC, matrix[r].data(), rhs.data());
        }
        return loop_res.front() + loop_res.back();
    });

    std::vector<double> manually_unrolled_res(NR);
    names.push_back("manually unrolled");
    funs.emplace_back([&]() -> double {
        for (std::size_t r = 0; r < NR; ++r) {
            manually_unrolled_res[r] = manually_unrolled(NC, matrix[r].data(), rhs.data());
        }
        return manually_unrolled_res.front() + manually_unrolled_res.back();
    });

    std::vector<double> peeled_res(NR);
    names.push_back("peeled");
    funs.emplace_back([&]() -> double {
        for (std::size_t r = 0; r < NR; ++r) {
            peeled_res[r] = peeled(NC, matrix[r].data(), rhs.data());
        }
        return peeled_res.front() + peeled_res.back();
    });

    std::vector<double> vectorized_epilogue_res(NR);
    names.push_back("vectorized epilogue");
    funs.emplace_back([&]() -> double {
        for (std::size_t r = 0; r < NR; ++r) {
            vectorized_epilogue_res[r] = vectorized_epilogue(NC, matrix[r].data(), rhs.data());
        }
        return vectorized_epilogue_res.front() + vectorized_epilogue_res.back();
    });

    std::vector<double> recursive_sum_res(NR);
    names.push_back("recursive sum");
    funs.emplace_back([&]() -> double {
        for (std::size_t r = 0; r < NR; ++r) {
            recursive_sum_res[r] = recursive_sum(NC, matrix[r].data(), rhs.data());
        }
        return recursive_sum_res.front() + recursive_sum_res.back();
    });

    // Performing the iterations.
    eztimer::Options opt;
    opt.iterations = iterations;

    std::optional<double> expected;
    auto res = eztimer::time<double>(
        funs,
        [&](const double& res, std::size_t i) -> void {
            if (expected.has_value()) {
                if (std::abs(*expected - res) > 1e-8) {
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
