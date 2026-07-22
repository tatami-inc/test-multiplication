#include <random>
#include <vector>
#include <iostream>
#include <cstddef>

#include "CLI/CLI.hpp"
#include "eztimer/eztimer.hpp"

#include "implementations.h"

template<int mode_>
void blocked_mult(
    const std::size_t NR,
    const std::size_t NC,
    const std::vector<std::vector<double> >& matrix,
    const std::size_t NRHS,
    const std::vector<std::vector<double> >& rhs,
    std::vector<std::vector<double> >& product
) {
    constexpr std::size_t block_size = 16;
    constexpr std::size_t line_size = 64;

    std::size_t r = 0;
    while (r < NR) {
        const std::size_t rend = r + std::min(block_size, NR - r);
        std::size_t h = 0;
        while (h < NRHS) {
            const std::size_t hend = h + std::min(block_size, NRHS - h);
            std::size_t c = 0;
            while (c < NC) {
                const std::size_t cnum = std::min(line_size, NC - c);
                const std::size_t cend = c + cnum;

                for (auto hcopy = h; hcopy < hend; ++hcopy) {
                    auto& outcol = product[hcopy];
                    const auto& rightcol = rhs[hcopy];
                    for (auto rcopy = r; rcopy < rend; ++rcopy) {

                        if constexpr(mode_ == 0) {
                            outcol[rcopy] = outcol[rcopy] + loop(cnum, rightcol.data() + c, matrix[rcopy].data() + c);
                        } else if constexpr(mode_ == 1) {
                            outcol[rcopy] = outcol[rcopy] + manually_unrolled(cnum, rightcol.data() + c, matrix[rcopy].data() + c);
                        } else if constexpr(mode_ == 2) {
                            outcol[rcopy] = outcol[rcopy] + peeled(cnum, rightcol.data() + c, matrix[rcopy].data() + c);
                        } else if constexpr(mode_ == 3) {
                            outcol[rcopy] = outcol[rcopy] + vectorized_epilogue(cnum, rightcol.data() + c, matrix[rcopy].data() + c);
                        } else if constexpr(mode_ == 4) {
                            outcol[rcopy] = outcol[rcopy] + recursive_sum(cnum, rightcol.data() + c, matrix[rcopy].data() + c);
                        }
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
    CLI::App app{"Timings for dense row-major LHS, multiple vectors RHS"};
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
    std::vector<std::vector<double> > matrix(NR);
    std::normal_distribution<> dist;
    for (std::size_t r = 0; r < NR; ++r) {
        matrix[r].resize(NC);
        for (auto& x : matrix[r]) {
            x = dist(rng);
        }
    }

    // Simulating the RHS vector.
    std::vector<std::vector<double> > rhs(NRHS);
    for (std::size_t h = 0; h < NRHS; ++h) {
        rhs[h].resize(NC);
        for (auto& x : rhs[h]) {
            x = dist(rng);
        }
    }

    auto preallocate_output = [&]() -> auto {
        std::vector<std::vector<double> > output(NRHS);
        for (std::size_t h = 0; h < NRHS; ++h) {
            output[h].resize(NR);
        }
        return output;
    };

    auto reset_output = [&](std::vector<std::vector<double> >& output) -> void {
        for (std::size_t h = 0; h < NRHS; ++h) {
            std::fill(output[h].begin(), output[h].end(), 0);
        }
    };

    std::vector<std::function<double()> > funs;
    funs.reserve(5);
    std::vector<std::string> names;
    names.reserve(5);

    // Setting up the outputs.
    auto loop_res = preallocate_output();
    names.push_back("loop");
    funs.emplace_back([&]() -> double {
        blocked_mult<0>(NR, NC, matrix, NRHS, rhs, loop_res);
        return loop_res.front().front() + loop_res.back().back();
    });

    auto manually_unrolled_res = preallocate_output();
    names.push_back("manually unrolled");
    funs.emplace_back([&]() -> double {
        blocked_mult<1>(NR, NC, matrix, NRHS, rhs, manually_unrolled_res);
        return manually_unrolled_res.front().front() + manually_unrolled_res.back().back();
    });

    auto peeled_res = preallocate_output();
    names.push_back("peeled");
    funs.emplace_back([&]() -> double {
        blocked_mult<2>(NR, NC, matrix, NRHS, rhs, peeled_res);
        return peeled_res.front().front() + peeled_res.back().back();
    });

    auto vectorized_epilogue_res = preallocate_output();
    names.push_back("vectorized epilogue");
    funs.emplace_back([&]() -> double {
        blocked_mult<3>(NR, NC, matrix, NRHS, rhs, vectorized_epilogue_res);
        return vectorized_epilogue_res.front().front() + vectorized_epilogue_res.back().back();
    });

    auto recursive_sum_res = preallocate_output();
    names.push_back("recursive sum");
    funs.emplace_back([&]() -> double {
        blocked_mult<4>(NR, NC, matrix, NRHS, rhs, recursive_sum_res);
        return recursive_sum_res.front().front() + recursive_sum_res.back().back();
    });

    // Performing the iterations.
    eztimer::Options opt;
    opt.iterations = iterations;
    opt.setup = [&]() -> void {
        reset_output(loop_res);
        reset_output(manually_unrolled_res);
        reset_output(peeled_res);
        reset_output(vectorized_epilogue_res);
        reset_output(recursive_sum_res);
    };

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

