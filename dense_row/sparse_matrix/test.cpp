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
void blocked_mult_with_right_column(
    const std::size_t NR,
    const std::size_t NC,
    const std::vector<std::vector<FLOAT> >& matrix,
    const std::size_t NRHS,
    const std::vector<std::vector<FLOAT> >& rhs_value_by_col,
    const std::vector<std::vector<int> >& rhs_index_by_col,
    std::vector<std::vector<FLOAT> >& product,
    std::size_t block_size
) {
    std::size_t r = 0;
    while (r < NR) {
        const std::size_t rend = r + std::min(block_size, NR - r);
        for (std::size_t h = 0; h < NRHS; ++h) {
            const auto& rval = rhs_value_by_col[h];
            const auto& ridx = rhs_index_by_col[h];
            const std::size_t nnz = rval.size();
            std::size_t x = 0;
            for (std::size_t rcopy = r; rcopy < rend; ++rcopy) {
                if constexpr(output_columnar_) {
                    product[h][rcopy] = dense_sparse_dot_product<num_acc_>(rval, ridx, matrix[rcopy]);
                } else {
                    product[rcopy][h] = dense_sparse_dot_product<num_acc_>(rval, ridx, matrix[rcopy]);
                }
            }
        }
        r = rend; 
    }
}

void blocked_mult_with_right_row_to_output_row(
    const std::size_t NR,
    const std::size_t NC,
    const std::vector<std::vector<FLOAT> >& matrix,
    const std::size_t NRHS,
    const std::vector<std::vector<FLOAT> >& rhs_value_by_row,
    const std::vector<std::vector<int> >& rhs_index_by_row,
    std::vector<std::vector<FLOAT> >& product,
    std::size_t block_size
) {
    std::size_t r = 0;
    while (r < NR) {
        const std::size_t rend = r + std::min(block_size, NR - r);
        for (std::size_t c = 0; c < NC; ++c) {
            const auto& rval = rhs_value_by_row[c];
            const auto& ridx = rhs_index_by_row[c];
            const std::size_t nnz = rval.size();
            for (std::size_t rcopy = r; rcopy < rend; ++rcopy) {
                const auto mult = matrix[rcopy][c];
                auto& out = product[rcopy];
                for (std::size_t x = 0; x < nnz; ++x) {
                    out[ridx[x]] += rval[x] * mult;
                }
            }
        }
        r = rend; 
    }
}

void blocked_mult_with_right_row_to_output_column(
    const std::size_t NR,
    const std::size_t NC,
    const std::vector<std::vector<FLOAT> >& matrix,
    const std::size_t NRHS,
    const std::vector<std::vector<FLOAT> >& rhs_value_by_row,
    const std::vector<std::vector<int> >& rhs_index_by_row,
    std::vector<std::vector<FLOAT> >& product,
    std::size_t block_size
) {
    std::size_t r = 0;
    std::vector<FLOAT> buffer(block_size);
    while (r < NR) {
        const std::size_t rend = r + std::min(block_size, NR - r);
        for (std::size_t c = 0; c < NC; ++c) {
            for (std::size_t rcopy = r; rcopy < rend; ++rcopy) {
                buffer[rcopy - r] = matrix[rcopy][c];
            }
            const auto& rval = rhs_value_by_row[c];
            const auto& ridx = rhs_index_by_row[c];
            const std::size_t nnz = rval.size();
            for (std::size_t x = 0; x < nnz; ++x) {
                auto& out = product[ridx[x]];
                const auto mult = rval[x];
                for (std::size_t rcopy = r; rcopy < rend; ++rcopy) {
                    out[rcopy] += buffer[rcopy - r] * mult;
                }
            }
        }
        r = rend; 
    }
}

int main(int argc, char ** argv) {
    CLI::App app{"Timings for dense row-major LHS, sparse matrix RHS"};
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
    funs.reserve(40);
    names.reserve(40);

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
        std::vector<FLOAT> buffer(NRHS);
        for (std::size_t r = 0; r < NR; ++r) {
            for (std::size_t c = 0; c < NC; ++c) {
                const auto mult = matrix[r][c];
                const auto& values = rhs_value_by_row[c];
                const auto& indices = rhs_index_by_row[c];
                const std::size_t nnz = values.size();
                for (std::size_t x = 0; x < nnz; ++x) { // a bit of an ugly loop here.
                    naive_rr_co[indices[x]][r] += values[x] * mult;
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

    // Blocked multiplication (column RHS, column output).
    auto blocked_cr_co_2 = preallocate_column_output();
    names.emplace_back("blocked (B = 2), column RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_column<4, true>(NR, NC, matrix, NRHS, rhs_value_by_col, rhs_index_by_col, blocked_cr_co_2, 2);
        return blocked_cr_co_2.front().front() + blocked_cr_co_2.front().back() + blocked_cr_co_2.back().front() + blocked_cr_co_2.back().back();
    });

    auto blocked_cr_co_4 = preallocate_column_output();
    names.emplace_back("blocked (B = 4), column RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_column<4, true>(NR, NC, matrix, NRHS, rhs_value_by_col, rhs_index_by_col, blocked_cr_co_4, 4);
        return blocked_cr_co_4.front().front() + blocked_cr_co_4.front().back() + blocked_cr_co_4.back().front() + blocked_cr_co_4.back().back();
    });

    auto blocked_cr_co_8 = preallocate_column_output();
    names.emplace_back("blocked (B = 8), column RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_column<4, true>(NR, NC, matrix, NRHS, rhs_value_by_col, rhs_index_by_col, blocked_cr_co_8, 8);
        return blocked_cr_co_8.front().front() + blocked_cr_co_8.front().back() + blocked_cr_co_8.back().front() + blocked_cr_co_8.back().back();
    });

    auto blocked_cr_co_16 = preallocate_column_output();
    names.emplace_back("blocked (B = 16), column RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_column<4, true>(NR, NC, matrix, NRHS, rhs_value_by_col, rhs_index_by_col, blocked_cr_co_16, 16);
        return blocked_cr_co_16.front().front() + blocked_cr_co_16.front().back() + blocked_cr_co_16.back().front() + blocked_cr_co_16.back().back();
    });

    // Blocked multiplication (column RHS, row output).
    auto blocked_cr_ro_2 = preallocate_row_output();
    names.emplace_back("blocked (B = 2), column RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_column<4, false>(NR, NC, matrix, NRHS, rhs_value_by_col, rhs_index_by_col, blocked_cr_ro_2, 2);
        return blocked_cr_ro_2.front().front() + blocked_cr_ro_2.front().back() + blocked_cr_ro_2.back().front() + blocked_cr_ro_2.back().back();
    });

    auto blocked_cr_ro_4 = preallocate_row_output();
    names.emplace_back("blocked (B = 4), column RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_column<4, false>(NR, NC, matrix, NRHS, rhs_value_by_col, rhs_index_by_col, blocked_cr_ro_4, 4);
        return blocked_cr_ro_4.front().front() + blocked_cr_ro_4.front().back() + blocked_cr_ro_4.back().front() + blocked_cr_ro_4.back().back();
    });

    auto blocked_cr_ro_8 = preallocate_row_output();
    names.emplace_back("blocked (B = 8), column RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_column<4, false>(NR, NC, matrix, NRHS, rhs_value_by_col, rhs_index_by_col, blocked_cr_ro_8, 8);
        return blocked_cr_ro_8.front().front() + blocked_cr_ro_8.front().back() + blocked_cr_ro_8.back().front() + blocked_cr_ro_8.back().back();
    });

    auto blocked_cr_ro_16 = preallocate_row_output();
    names.emplace_back("blocked (B = 16), column RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_column<4, false>(NR, NC, matrix, NRHS, rhs_value_by_col, rhs_index_by_col, blocked_cr_ro_16, 16);
        return blocked_cr_ro_16.front().front() + blocked_cr_ro_16.front().back() + blocked_cr_ro_16.back().front() + blocked_cr_ro_16.back().back();
    });

    // Blocked multiplication (row RHS, column output).
    auto blocked_rr_co_4 = preallocate_column_output();
    names.emplace_back("blocked (B = 4), row RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_column(NR, NC, matrix, NRHS, rhs_value_by_row, rhs_index_by_row, blocked_rr_co_4, 4);
        return blocked_rr_co_4.front().front() + blocked_rr_co_4.front().back() + blocked_rr_co_4.back().front() + blocked_rr_co_4.back().back();
    });

    auto blocked_rr_co_8 = preallocate_column_output();
    names.emplace_back("blocked (B = 8), row RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_column(NR, NC, matrix, NRHS, rhs_value_by_row, rhs_index_by_row, blocked_rr_co_8, 8);
        return blocked_rr_co_8.front().front() + blocked_rr_co_8.front().back() + blocked_rr_co_8.back().front() + blocked_rr_co_8.back().back();
    });

    auto blocked_rr_co_16 = preallocate_column_output();
    names.emplace_back("blocked (B = 16), row RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_column(NR, NC, matrix, NRHS, rhs_value_by_row, rhs_index_by_row, blocked_rr_co_16, 16);
        return blocked_rr_co_16.front().front() + blocked_rr_co_16.front().back() + blocked_rr_co_16.back().front() + blocked_rr_co_16.back().back();
    });

    auto blocked_rr_co_32 = preallocate_column_output();
    names.emplace_back("blocked (B = 32), row RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_column(NR, NC, matrix, NRHS, rhs_value_by_row, rhs_index_by_row, blocked_rr_co_32, 32);
        return blocked_rr_co_32.front().front() + blocked_rr_co_32.front().back() + blocked_rr_co_32.back().front() + blocked_rr_co_32.back().back();
    });


    // Blocked multiplication (row RHS, row output).
    auto blocked_rr_ro_2 = preallocate_row_output();
    names.emplace_back("blocked (B = 2), row RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_row(NR, NC, matrix, NRHS, rhs_value_by_row, rhs_index_by_row, blocked_rr_ro_2, 2);
        return blocked_rr_ro_2.front().front() + blocked_rr_ro_2.front().back() + blocked_rr_ro_2.back().front() + blocked_rr_ro_2.back().back();
    });

    auto blocked_rr_ro_4 = preallocate_row_output();
    names.emplace_back("blocked (B = 4), row RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_row(NR, NC, matrix, NRHS, rhs_value_by_row, rhs_index_by_row, blocked_rr_ro_4, 4);
        return blocked_rr_ro_4.front().front() + blocked_rr_ro_4.front().back() + blocked_rr_ro_4.back().front() + blocked_rr_ro_4.back().back();
    });

    auto blocked_rr_ro_8 = preallocate_row_output();
    names.emplace_back("blocked (B = 8), row RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_row(NR, NC, matrix, NRHS, rhs_value_by_row, rhs_index_by_row, blocked_rr_ro_8, 8);
        return blocked_rr_ro_8.front().front() + blocked_rr_ro_8.front().back() + blocked_rr_ro_8.back().front() + blocked_rr_ro_8.back().back();
    });

    auto blocked_rr_ro_16 = preallocate_row_output();
    names.emplace_back("blocked (B = 16), row RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_row(NR, NC, matrix, NRHS, rhs_value_by_row, rhs_index_by_row, blocked_rr_ro_16, 16);
        return blocked_rr_ro_16.front().front() + blocked_rr_ro_16.front().back() + blocked_rr_ro_16.back().front() + blocked_rr_ro_16.back().back();
    });

    // Performing the iterations.
    eztimer::Options opt;
    opt.iterations = iterations;
    opt.setup = [&]() -> void {
        reset_output(naive_rr_co);
        reset_output(naive_rr_ro);
        reset_output(blocked_cr_ro_2);
        reset_output(blocked_cr_ro_4);
        reset_output(blocked_cr_ro_8);
        reset_output(blocked_cr_ro_16);
        reset_output(blocked_cr_co_2);
        reset_output(blocked_cr_co_4);
        reset_output(blocked_cr_co_8);
        reset_output(blocked_cr_co_16);
        reset_output(blocked_rr_ro_2);
        reset_output(blocked_rr_ro_4);
        reset_output(blocked_rr_ro_8);
        reset_output(blocked_rr_ro_16);
        reset_output(blocked_rr_co_4);
        reset_output(blocked_rr_co_8);
        reset_output(blocked_rr_co_16);
        reset_output(blocked_rr_co_32);
    };

    std::optional<FLOAT> expected;
    auto res = eztimer::time<FLOAT>(
        funs,
        [&](const FLOAT& res, std::size_t i) -> void {
            //std::cout << res << "\t" << i << std::endl;
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
        constexpr std::size_t pad_to = 50;
        if (name.size() < pad_to) {
            name.insert(name.end(), pad_to - name.size(), ' ');
        }
        std::cout << name << ": " << res[m].mean.count() << " ± " << res[m].sd.count() / std::sqrt(res[m].times.size()) << std::endl;
    }

    return 0;
}
