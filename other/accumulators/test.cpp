#include <vector>
#include <random>
#include <array>
#include <cstddef>
#include <iostream>
#include <optional>

#include "eztimer/eztimer.hpp"
#include "CLI/CLI.hpp"

double loop(const std::size_t len, const std::vector<double>& left, const std::vector<double>& right);
double manually_unrolled(const std::size_t len, const std::vector<double>& left, const std::vector<double>& right);
double peeled(const std::size_t len, const std::vector<double>& left, const std::vector<double>& right);
double vectorized_epilogue(const std::size_t len, const std::vector<double>& left, const std::vector<double>& right);
double recursive_sum(const std::size_t len, const std::vector<double>& left, const std::vector<double>& right);
double combined(const std::size_t len, const std::vector<double>& left, const std::vector<double>& right);

int main(int argc, char** argv) {
    CLI::App app{"Timings for multiple accumulator implementations"};
    std::size_t len;
    app.add_option("-l,--length", len, "Length of the array")->default_val(20);
    int iterations;
    app.add_option("-i,--iter", iterations, "Number of iterations")->default_val(100000);
    unsigned long long seed;
    app.add_option("-s,--seed", seed, "Seed for the simulated data")->default_val(69);
    CLI11_PARSE(app, argc, argv);

    std::mt19937_64 rng(seed);
    std::vector<double> left(len), right(len);
    std::normal_distribution<> dist;
    for (std::size_t l = 0; l < len; ++l) {
        left[l] = dist(rng);
        right[l] = dist(rng);
    }

    std::vector<std::function<double()> > funs;
    funs.reserve(6);
    std::vector<std::string> names;
    names.reserve(6);

    names.push_back("loop");
    funs.emplace_back([&]() -> double {
        double res = 0;
        for (int i = 0; i < iterations; ++i) {
            res += loop(len, left, right);
        }
        return res;
    });

    names.push_back("manually unrolled");
    funs.emplace_back([&]() -> double {
        double res = 0;
        for (int i = 0; i < iterations; ++i) {
            res += loop(len, left, right);
        }
        return res;
    });

    names.push_back("peeled");
    funs.emplace_back([&]() -> double {
        double res = 0;
        for (int i = 0; i < iterations; ++i) { 
            res += peeled(len, left, right);
        }
        return res;
    });

    names.push_back("vectorized epilogue");
    funs.emplace_back([&]() -> double {
        double res = 0;
        for (int i = 0; i < iterations; ++i) { 
            res += vectorized_epilogue(len, left, right);
        }
        return res;
    });

    names.push_back("recursive sum");
    funs.emplace_back([&]() -> double {
        double res = 0;
        for (int i = 0; i < iterations; ++i) { 
            res += recursive_sum(len, left, right);
        }
        return res;
    });

    names.push_back("combined");
    funs.emplace_back([&]() -> double {
        double res = 0;
        for (int i = 0; i < iterations; ++i) { 
            res += combined(len, left, right);
        }
        return res;
    });

    eztimer::Options opt;
    opt.iterations = 10;
    std::optional<double> expected;
    auto res = eztimer::time<double>(
        funs,
        [&](double val, std::size_t i) -> void {
//            std::cout << val << "\t" << i << std::endl;
            if (expected.has_value()) {
                if (std::abs(val - *expected) > 1e-8) {
                    throw std::runtime_error("unexpected value for '" + names[i] + "'"); 
                }
            } else {
                expected = val;
            }
        },
        opt 
    );

    std::cout << "Results for a vector of length " << len << std::endl;
    const std::size_t num_methods = names.size();
    for (std::size_t m = 0; m < num_methods; ++m) {
        auto name = names[m];
        constexpr std::size_t pad_to = 50;
        if (name.size() < pad_to) {
            name.insert(name.end(), pad_to - name.size(), ' ');
        }
        std::cout << name << ": " << res[m].mean.count() << " ± " << res[m].sd.count() / std::sqrt(res[m].times.size()) << std::endl;
    }

    return 0;
}
