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

template<int num_acc_, bool output_columnar_>
void best_mult_with_right_column(
    const std::size_t NR,
    const std::size_t NC,
    const std::vector<std::vector<FLOAT> >& mat_value,
    const std::vector<std::vector<int> >& mat_index,
    const std::size_t NRHS,
    const std::vector<std::vector<FLOAT> >& rhs_value_by_col,
    const std::vector<std::vector<int> >& rhs_index_by_col,
    std::vector<std::vector<FLOAT> >& product
) {
    std::size_t r = 0;
    std::vector<FLOAT> buffer(NC);
    for (std::size_t r = 0; r < NR; ++r) {
        const auto& mval = mat_value[r];
        const auto& midx = mat_index[r];
        const std::size_t mnnz = mval.size();
        for (std::size_t x = 0; x < mnnz; ++x) {
            buffer[midx[x]] = mval[x];
        }

        for (std::size_t h = 0; h < NRHS; ++h) {
            const auto& rval = rhs_value_by_col[h];
            const auto& ridx = rhs_index_by_col[h];
            if constexpr(output_columnar_) {
                product[h][r] = dense_sparse_dot_product<num_acc_>(rval, ridx, buffer);
            } else {
                product[r][h] = dense_sparse_dot_product<num_acc_>(rval, ridx, buffer);
            }
        }

        // Setting it back to zero so that the next row has an all-zero vector.
        // This is slightly faster than a big fill() when the input data is actually sparse.
        for (std::size_t x = 0; x < mnnz; ++x) {
            buffer[midx[x]] = 0;
        }
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

    // Simulating the RHS matrix.
    std::vector<std::vector<FLOAT> > rhs_value_by_col(NRHS);
    std::vector<std::vector<int> > rhs_index_by_col(NRHS);
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
    funs.reserve(5);
    names.reserve(5);

    // Naive for row-major.
    auto naive_rr_ro = preallocate_row_output();
    names.emplace_back("naive, row RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            const auto& mval = mat_value[r];
            const auto& midx = mat_index[r];
            const std::size_t mnnz = mval.size();
            auto& out = naive_rr_ro[r];

            for (std::size_t x = 0; x < mnnz; ++x) {
                const auto& rval = rhs_value_by_row[midx[x]];
                const auto& ridx = rhs_index_by_row[midx[x]];
                const auto mult = mval[x];
                const std::size_t rnnz = rval.size();
                for (std::size_t y = 0; y < rnnz; ++y) {
                    out[ridx[y]] += mult * rval[y]; 
                }
            }
        }
        return naive_rr_ro.front().front() + naive_rr_ro.front().back() + naive_rr_ro.back().front() + naive_rr_ro.back().back();
    });

    auto naive_rr_co = preallocate_column_output();
    names.emplace_back("naive, row RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            const auto& mval = mat_value[r];
            const auto& midx = mat_index[r];
            const std::size_t mnnz = mval.size();

            for (std::size_t x = 0; x < mnnz; ++x) {
                const auto& rval = rhs_value_by_row[midx[x]];
                const auto& ridx = rhs_index_by_row[midx[x]];
                const auto mult = mval[x];
                const std::size_t rnnz = rval.size();
                for (std::size_t y = 0; y < rnnz; ++y) {
                    naive_rr_co[ridx[y]][r] += mult * rval[y]; 
                }
            }
        }
        return naive_rr_co.front().front() + naive_rr_co.front().back() + naive_rr_co.back().front() + naive_rr_co.back().back();
    });

    // Best of column major.
    auto best_cr_co = preallocate_column_output();
    names.emplace_back("naive, column RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        best_mult_with_right_column<4, true>(NR, NC, mat_value, mat_index, NRHS, rhs_value_by_col, rhs_index_by_col, best_cr_co);
        return best_cr_co.front().front() + best_cr_co.front().back() + best_cr_co.back().front() + best_cr_co.back().back();
    });

    auto best_cr_ro = preallocate_row_output();
    names.emplace_back("naive, column RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        best_mult_with_right_column<4, false>(NR, NC, mat_value, mat_index, NRHS, rhs_value_by_col, rhs_index_by_col, best_cr_ro);
        return best_cr_ro.front().front() + best_cr_ro.front().back() + best_cr_ro.back().front() + best_cr_ro.back().back();
    });

    // Performing the iterations.
    eztimer::Options opt;
    opt.iterations = iterations;
    opt.setup = [&]() -> void {
        reset_output(naive_rr_ro);
        reset_output(naive_rr_co);
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
