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

template<int num_acc_, class Values_, class Indices_, class Dense_>
FLOAT dense_sparse_dot_product(const Values_& values, const Indices_& indices, const Dense_& dense) {
    const std::size_t num_non_zeros = values.size();

    if constexpr(num_acc_ == 1) {
        FLOAT dot = 0;
        for (std::size_t i = 0; i < num_non_zeros; ++i) {
            dot += dense[indices[i]] * values[i];
        }
        return dot;

    } else {
        std::array<FLOAT, num_acc_> dots{};
        const std::size_t cycles = num_non_zeros / num_acc_;
        const std::size_t remainder = num_non_zeros % num_acc_;

        for (std::size_t c = 0; c < cycles; ++c) {
            for (std::size_t i = 0; i < num_acc_; ++i) {
                const auto idx = c * num_acc_ + i;
                dots[i] += dense[indices[idx]] * values[idx]; 
            }
        }

        FLOAT extras = 0;
        for (std::size_t i = 0; i < remainder; ++i) {
            const auto idx = cycles * num_acc_ + i;
            extras += dense[indices[idx]] * values[idx];
        }
        return std::accumulate(dots.begin(), dots.end(), extras);
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
    std::vector<std::vector<FLOAT> > rhs_value_by_col(NRHS);
    std::vector<std::vector<int> > rhs_index_by_col(NRHS);
    std::uniform_real_distribution<> udist;
    for (std::size_t h = 0; h < NRHS; ++h) {
        for (std::size_t c = 0; c < NC; ++c) {
            if (udist(rng) <= 0.1) {
                rhs_index_by_col[h].push_back(c);
                rhs_value_by_col[h].push_back(dist(rng));
            }
        }
        rhs_index_by_col[h].shrink_to_fit();
        rhs_value_by_col[h].shrink_to_fit();
    }

    std::vector<std::vector<FLOAT> > rhs_value_by_row(NC);
    std::vector<std::vector<int> > rhs_index_by_row(NC);
    for (std::size_t h = 0; h < NRHS; ++h) {
        const std::size_t nnz = rhs_value_by_col[h].size();
        for (std::size_t i = 0; i < nnz; ++i) {
            rhs_value_by_row[rhs_index_by_col[h][i]].push_back(rhs_value_by_col[h][i]);
            rhs_index_by_row[rhs_index_by_col[h][i]].push_back(h);
        }
    }
    for (std::size_t c = 0; c < NC; ++c) {
        rhs_index_by_row[c].shrink_to_fit();
        rhs_value_by_row[c].shrink_to_fit();
    }

    // Doing all this.
    auto preallocate_column_output = [&]() -> auto {
        std::vector<std::vector<FLOAT> > output(NRHS);
        for (std::size_t h = 0; h < NRHS; ++h) {
            output[h].resize(NR);
        }
        return output;
    };

    auto preallocate_row_output = [&]() -> auto {
        std::vector<std::vector<FLOAT> > output(NR);
        for (std::size_t r = 0; r < NR; ++r) {
            output[r].resize(NRHS);
        }
        return output;
    };

    auto reset_output = [&](std::vector<std::vector<FLOAT> >& output) -> void {
        for (auto& out : output) {
            std::fill(out.begin(), out.end(), 0);
        }
    };

    std::vector<std::function<FLOAT()> > funs;
    std::vector<std::string> names;
    funs.reserve(20);
    names.reserve(20);

    // Naive multiplication.
    auto naive_cr_co = preallocate_column_output();
    names.emplace_back("naive, column RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            for (std::size_t h = 0; h < NRHS; ++h) {
                naive_cr_co[h][r] = dense_sparse_dot_product<1>(rhs_value_by_col[h], rhs_index_by_col[h], matrix[r]);
            }
        }
        return naive_cr_co.front().front() + naive_cr_co.front().back() + naive_cr_co.back().front() + naive_cr_co.back().back();
    });

    auto naive_cr_ro = preallocate_row_output();
    names.emplace_back("naive, column RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            for (std::size_t h = 0; h < NRHS; ++h) {
                naive_cr_ro[r][h] = dense_sparse_dot_product<1>(rhs_value_by_col[h], rhs_index_by_col[h], matrix[r]);
            }
        }
        return naive_cr_ro.front().front() + naive_cr_ro.front().back() + naive_cr_ro.back().front() + naive_cr_ro.back().back();
    });

    auto naive_rr_co = preallocate_column_output();
    names.emplace_back("naive, row RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            for (std::size_t c = 0; c < NC; ++c) {
                const auto mult = matrix[r][c];
                const auto& values = rhs_value_by_row[c];
                const auto& indices = rhs_index_by_row[c];
                const std::size_t nnz = values.size();
                for (std::size_t x = 0; x < nnz; ++x) {
                    naive_rr_co[indices[x]][r] += values[x] * mult; // ugh, gross inner loop.
                }
            }
        }
        return naive_rr_co.front().front() + naive_rr_co.front().back() + naive_rr_co.back().front() + naive_rr_co.back().back();
    });

    auto naive_rr_ro = preallocate_row_output();
    names.emplace_back("naive, row RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            auto& out = naive_rr_ro[r];
            for (std::size_t c = 0; c < NC; ++c) {
                const auto mult = matrix[r][c];
                const auto& values = rhs_value_by_row[c];
                const auto& indices = rhs_index_by_row[c];
                const std::size_t nnz = values.size();
                for (std::size_t x = 0; x < nnz; ++x) {
                    out[indices[x]] += values[x] * mult;
                }
            }
        }
        return naive_rr_ro.front().front() + naive_rr_ro.front().back() + naive_rr_ro.back().front() + naive_rr_ro.back().back();
    });

    // Multiple accumulators.
    auto ma2_cr_co = preallocate_column_output();
    names.emplace_back("2 accumulators, column RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            for (std::size_t h = 0; h < NRHS; ++h) {
                ma2_cr_co[h][r] = dense_sparse_dot_product<2>(rhs_value_by_col[h], rhs_index_by_col[h], matrix[r]);
            }
        }
        return ma2_cr_co.front().front() + ma2_cr_co.front().back() + ma2_cr_co.back().front() + ma2_cr_co.back().back();
    });

    auto ma2_cr_ro = preallocate_row_output();
    names.emplace_back("2 accumulators, column RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            for (std::size_t h = 0; h < NRHS; ++h) {
                ma2_cr_ro[r][h] = dense_sparse_dot_product<2>(rhs_value_by_col[h], rhs_index_by_col[h], matrix[r]);
            }
        }
        return ma2_cr_ro.front().front() + ma2_cr_ro.front().back() + ma2_cr_ro.back().front() + ma2_cr_ro.back().back();
    });

    auto ma4_cr_co = preallocate_column_output();
    names.emplace_back("4 accumulators, column RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            for (std::size_t h = 0; h < NRHS; ++h) {
                ma4_cr_co[h][r] = dense_sparse_dot_product<4>(rhs_value_by_col[h], rhs_index_by_col[h], matrix[r]);
            }
        }
        return ma4_cr_co.front().front() + ma4_cr_co.front().back() + ma4_cr_co.back().front() + ma4_cr_co.back().back();
    });

    auto ma4_cr_ro = preallocate_row_output();
    names.emplace_back("4 accumulators, column RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            for (std::size_t h = 0; h < NRHS; ++h) {
                ma4_cr_ro[r][h] = dense_sparse_dot_product<4>(rhs_value_by_col[h], rhs_index_by_col[h], matrix[r]);
            }
        }
        return ma4_cr_ro.front().front() + ma4_cr_ro.front().back() + ma4_cr_ro.back().front() + ma4_cr_ro.back().back();
    });

    auto ma8_cr_co = preallocate_column_output();
    names.emplace_back("8 accumulators, column RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            for (std::size_t h = 0; h < NRHS; ++h) {
                ma8_cr_co[h][r] = dense_sparse_dot_product<8>(rhs_value_by_col[h], rhs_index_by_col[h], matrix[r]);
            }
        }
        return ma8_cr_co.front().front() + ma8_cr_co.front().back() + ma8_cr_co.back().front() + ma8_cr_co.back().back();
    });

    auto ma8_cr_ro = preallocate_row_output();
    names.emplace_back("8 accumulators, column RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            for (std::size_t h = 0; h < NRHS; ++h) {
                ma8_cr_ro[r][h] = dense_sparse_dot_product<8>(rhs_value_by_col[h], rhs_index_by_col[h], matrix[r]);
            }
        }
        return ma8_cr_ro.front().front() + ma8_cr_ro.front().back() + ma8_cr_ro.back().front() + ma8_cr_ro.back().back();
    });

    // Performing the iterations.
    eztimer::Options opt;
    opt.iterations = iterations;
    opt.setup = [&]() -> void {
        reset_output(naive_rr_co);
        reset_output(naive_rr_ro);
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
