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

void blocked_mult_right_column_to_output_column(
    const std::size_t NR,
    const std::size_t NC,
    const std::vector<std::vector<FLOAT> >& mat_value,
    const std::vector<std::vector<int> >& mat_index,
    const std::size_t NRHS,
    const std::vector<std::vector<FLOAT> >& rhs,
    std::vector<std::vector<FLOAT> >& product,
    const std::size_t block_size 
) {
    std::size_t c = 0;
    while (c < NC) {
        const std::size_t cend = c + std::min(NC - c, block_size);
        for (std::size_t h = 0; h < NRHS; ++h) {
            auto& out = product[h];
            const auto& rightcol = rhs[h];
            for (std::size_t ccopy = c; ccopy < cend; ++ccopy) {
                const auto& mval = mat_value[ccopy];
                const auto& midx = mat_index[ccopy];
                const std::size_t mnnz = mval.size();
                const auto mult = rightcol[ccopy];
                for (std::size_t x = 0; x < mnnz; ++x) {
                    out[midx[x]] += mult * mval[x];
                }
            }
        }
        c = cend;
    }
}

void blocked_mult_right_column_to_output_row(
    const std::size_t NR,
    const std::size_t NC,
    const std::vector<std::vector<FLOAT> >& mat_value,
    const std::vector<std::vector<int> >& mat_index,
    const std::size_t NRHS,
    const std::vector<std::vector<FLOAT> >& rhs,
    std::vector<std::vector<FLOAT> >& product,
    const std::size_t block_size 
) {
    const std::size_t line_size = 1024 / block_size;
    for (std::size_t c = 0; c < NC; ++c) {
        const auto& mval = mat_value[c];
        const auto& midx = mat_index[c];
        const std::size_t mnnz = mval.size();
        std::size_t x = 0;
        while (x < mnnz) {
            const std::size_t xend = x + std::min(mnnz - x, block_size);
            std::size_t h = 0;
            while (h < NRHS) {
                const std::size_t hend = h + std::min(NRHS - h, line_size);
                for (std::size_t xcopy = x; xcopy < xend; ++xcopy) {
                    auto& out = product[midx[xcopy]];
                    const auto mult = mval[xcopy];
                    for (std::size_t hcopy = h; hcopy < hend; ++hcopy) {
                        out[hcopy] += mult * rhs[hcopy][c];
                    }
                }
                h = hend;
            }
            x = xend;
        }
    }
}

void blocked_mult_right_row_to_output_column(
    const std::size_t NR,
    const std::size_t NC,
    const std::vector<std::vector<FLOAT> >& mat_value,
    const std::vector<std::vector<int> >& mat_index,
    const std::size_t NRHS,
    const std::vector<std::vector<FLOAT> >& rhs,
    std::vector<std::vector<FLOAT> >& product,
    const std::size_t block_size 
) {
    std::size_t c = 0;
    while (c < NC) {
        const std::size_t cend = c + std::min(NC - c, block_size);
        for (std::size_t h = 0; h < NRHS; ++h) {
            auto& out = product[h];
            for (std::size_t ccopy = c; ccopy < cend; ++ccopy) {
                const auto& mval = mat_value[ccopy];
                const auto& midx = mat_index[ccopy];
                const auto mult = rhs[ccopy][h];
                const std::size_t mnnz = mval.size();
                for (std::size_t x = 0; x < mnnz; ++x) {
                    out[midx[x]] += mult * mval[x];
                }
            }
        }
        c = cend;
    }
}

void blocked_mult_right_row_to_output_row(
    const std::size_t NR,
    const std::size_t NC,
    const std::vector<std::vector<FLOAT> >& mat_value,
    const std::vector<std::vector<int> >& mat_index,
    const std::size_t NRHS,
    const std::vector<std::vector<FLOAT> >& rhs,
    std::vector<std::vector<FLOAT> >& product,
    const std::size_t block_size 
) {
    const std::size_t line_size = 1024 / block_size;
    for (std::size_t c = 0; c < NC; ++c) {
        const auto& mval = mat_value[c];
        const auto& midx = mat_index[c];
        const std::size_t mnnz = mval.size();
        const auto& rightrow = rhs[c];
        std::size_t x = 0;
        while (x < mnnz) {
            const std::size_t xend = x + std::min(mnnz - x, block_size);
            std::size_t h = 0;
            while (h < NRHS) {
                const std::size_t hend = h + std::min(NRHS - h, line_size);
                for (std::size_t xcopy = x; xcopy < xend; ++xcopy) {
                    const auto mult = mval[xcopy];
                    auto& out = product[midx[xcopy]];
                    for (std::size_t hcopy = h; hcopy < hend; ++hcopy) {
                        out[hcopy] += mult * rightrow[hcopy];
                    }
                }
                h = hend;
            }
            x = xend; 
        }
    }
}

int main(int argc, char ** argv) {
    CLI::App app{"Sparse column matrix x dense vector performance tests"};
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

    // Simulating a column-major sparse matrix.
    std::mt19937_64 rng(seed);
    std::vector<std::vector<FLOAT> > mat_value(NC);
    std::vector<std::vector<int> > mat_index(NC);
    std::uniform_real_distribution<> udist;
    std::normal_distribution<> dist;
    for (std::size_t c = 0; c < NC; ++c) {
        for (std::size_t r = 0; r < NR; ++r) {
            if (udist(rng) <= 0.1) {
                mat_index[c].push_back(r);
                mat_value[c].push_back(dist(rng));
            }
        }
        mat_index[c].shrink_to_fit();
        mat_value[c].shrink_to_fit();
    }

    // Simulating the RHS vector.
    std::vector<std::vector<FLOAT> > rhs_by_col(NRHS);
    for (std::size_t h = 0; h < NRHS; ++h) {
        rhs_by_col[h].resize(NC);
        for (auto& x : rhs_by_col[h]) {
            x = dist(rng);
        }
    }

    std::vector<std::vector<FLOAT> > rhs_by_row(NC);
    for (std::size_t c = 0; c < NC; ++c) {
        rhs_by_row[c].reserve(NRHS);
    }
    for (std::size_t h = 0; h < NRHS; ++h) {
        for (std::size_t c = 0; c < NC; ++c) {
            rhs_by_row[c].push_back(rhs_by_col[h][c]);
        }
    }

    auto preallocate_row_output = [&]() -> auto {
        std::vector<std::vector<FLOAT> > output(NR);
        for (std::size_t r = 0; r < NR; ++r) {
            output[r].resize(NRHS);
        }
        return output;
    };

    auto preallocate_column_output = [&]() -> auto {
        std::vector<std::vector<FLOAT> > output(NRHS);
        for (std::size_t h = 0; h < NRHS; ++h) {
            output[h].resize(NR);
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

    // Naive multiplication for row-major RHS.
    auto naive_rr_ro = preallocate_row_output();
    names.emplace_back("naive, row-major RHS, row-major output");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t c = 0; c < NC; ++c) {
            const auto& mval = mat_value[c];
            const auto& midx = mat_index[c];
            const std::size_t mnnz = mval.size();
            const auto& rightrow = rhs_by_row[c];
            for (std::size_t x = 0; x < mnnz; ++x) {
                auto& out = naive_rr_ro[midx[x]];
                const auto mult = mval[x];
                for (std::size_t h = 0; h < NRHS; ++h) {
                    out[h] += mult * rightrow[h];
                }
            }
        }
        return naive_rr_ro.front().front() + naive_rr_ro.front().back() + naive_rr_ro.back().front() + naive_rr_ro.back().back();
    });

    auto naive_rr_co = preallocate_column_output();
    names.emplace_back("naive, row-major RHS, column-major output");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t c = 0; c < NC; ++c) {
            const auto& mval = mat_value[c];
            const auto& midx = mat_index[c];
            const std::size_t mnnz = mval.size();
            const auto& rightrow = rhs_by_row[c];
            for (std::size_t h = 0; h < NRHS; ++h) {
                const auto mult = rightrow[h];
                auto& out = naive_rr_co[h];
                for (std::size_t x = 0; x < mnnz; ++x) {
                    out[midx[x]] += mult * mval[x];
                }
            }
        }
        return naive_rr_co.front().front() + naive_rr_co.front().back() + naive_rr_co.back().front() + naive_rr_co.back().back();
    });

    // Naive multiplication for column-major RHS with row-major output.
    auto naive_cr_ro = preallocate_row_output();
    names.emplace_back("naive, column-major RHS, row-major output");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t c = 0; c < NC; ++c) {
            const auto& mval = mat_value[c];
            const auto& midx = mat_index[c];
            const std::size_t mnnz = mval.size();
            for (std::size_t x = 0; x < mnnz; ++x) {
                const auto mult = mval[x];
                auto& out = naive_cr_ro[midx[x]];
                for (std::size_t h = 0; h < NRHS; ++h) {
                    out[h] += mult * rhs_by_col[h][c];
                }
            }
        }
        return naive_cr_ro.front().front() + naive_cr_ro.front().back() + naive_cr_ro.back().front() + naive_cr_ro.back().back();
    });

    // Best of column-major RHS with column-major output.
    auto best_cr_co = preallocate_column_output();
    names.emplace_back("best, column-major RHS, column-major output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_right_column_to_output_column(NR, NC, mat_value, mat_index, NRHS, rhs_by_col, best_cr_co, 16);
        return best_cr_co.front().front() + best_cr_co.front().back() + best_cr_co.back().front() + best_cr_co.back().back();
    });

    // Blocked row-major RHS, row-major output.
    auto blocked_rr_ro_4 = preallocate_row_output();
    names.emplace_back("blocked (B = 4), row-major RHS, row-major output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_right_row_to_output_row(NR, NC, mat_value, mat_index, NRHS, rhs_by_row, blocked_rr_ro_4, 4);
        return blocked_rr_ro_4.front().front() + blocked_rr_ro_4.front().back() + blocked_rr_ro_4.back().front() + blocked_rr_ro_4.back().back();
    });
 
    auto blocked_rr_ro_8 = preallocate_row_output();
    names.emplace_back("blocked (B = 8), row-major RHS, row-major output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_right_row_to_output_row(NR, NC, mat_value, mat_index, NRHS, rhs_by_row, blocked_rr_ro_8, 8);
        return blocked_rr_ro_8.front().front() + blocked_rr_ro_8.front().back() + blocked_rr_ro_8.back().front() + blocked_rr_ro_8.back().back();
    });
 
    auto blocked_rr_ro_16 = preallocate_row_output();
    names.emplace_back("blocked (B = 16), row-major RHS, row-major output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_right_row_to_output_row(NR, NC, mat_value, mat_index, NRHS, rhs_by_row, blocked_rr_ro_16, 16);
        return blocked_rr_ro_16.front().front() + blocked_rr_ro_16.front().back() + blocked_rr_ro_16.back().front() + blocked_rr_ro_16.back().back();
    });
 
    auto blocked_rr_ro_32 = preallocate_row_output();
    names.emplace_back("blocked (B = 32), row-major RHS, row-major output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_right_row_to_output_row(NR, NC, mat_value, mat_index, NRHS, rhs_by_row, blocked_rr_ro_32, 32);
        return blocked_rr_ro_32.front().front() + blocked_rr_ro_32.front().back() + blocked_rr_ro_32.back().front() + blocked_rr_ro_32.back().back();
    });
 
    // Blocked row-major RHS, column-major output.
    auto blocked_rr_co_4 = preallocate_column_output();
    names.emplace_back("blocked (B = 4), row-major RHS, column-major output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_right_row_to_output_column(NR, NC, mat_value, mat_index, NRHS, rhs_by_row, blocked_rr_co_4, 4);
        return blocked_rr_co_4.front().front() + blocked_rr_co_4.front().back() + blocked_rr_co_4.back().front() + blocked_rr_co_4.back().back();
    });
 
    auto blocked_rr_co_8 = preallocate_column_output();
    names.emplace_back("blocked (B = 8), row-major RHS, column-major output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_right_row_to_output_column(NR, NC, mat_value, mat_index, NRHS, rhs_by_row, blocked_rr_co_8, 8);
        return blocked_rr_co_8.front().front() + blocked_rr_co_8.front().back() + blocked_rr_co_8.back().front() + blocked_rr_co_8.back().back();
    });
 
    auto blocked_rr_co_16 = preallocate_column_output();
    names.emplace_back("blocked (B = 16), row-major RHS, column-major output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_right_row_to_output_column(NR, NC, mat_value, mat_index, NRHS, rhs_by_row, blocked_rr_co_16, 16);
        return blocked_rr_co_16.front().front() + blocked_rr_co_16.front().back() + blocked_rr_co_16.back().front() + blocked_rr_co_16.back().back();
    });
 
    auto blocked_rr_co_32 = preallocate_column_output();
    names.emplace_back("blocked (B = 32), row-major RHS, column-major output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_right_row_to_output_column(NR, NC, mat_value, mat_index, NRHS, rhs_by_row, blocked_rr_co_32, 32);
        return blocked_rr_co_32.front().front() + blocked_rr_co_32.front().back() + blocked_rr_co_32.back().front() + blocked_rr_co_32.back().back();
    });
 
    // Blocked column-major RHS, row-major output.
    auto blocked_cr_ro_4 = preallocate_row_output();
    names.emplace_back("blocked (B = 4), column-major RHS, row-major output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_right_column_to_output_row(NR, NC, mat_value, mat_index, NRHS, rhs_by_col, blocked_cr_ro_4, 4);
        return blocked_cr_ro_4.front().front() + blocked_cr_ro_4.front().back() + blocked_cr_ro_4.back().front() + blocked_cr_ro_4.back().back();
    });
 
    auto blocked_cr_ro_8 = preallocate_row_output();
    names.emplace_back("blocked (B = 8), column-major RHS, row-major output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_right_column_to_output_row(NR, NC, mat_value, mat_index, NRHS, rhs_by_col, blocked_cr_ro_8, 8);
        return blocked_cr_ro_8.front().front() + blocked_cr_ro_8.front().back() + blocked_cr_ro_8.back().front() + blocked_cr_ro_8.back().back();
    });
 
    auto blocked_cr_ro_16 = preallocate_row_output();
    names.emplace_back("blocked (B = 16), column-major RHS, row-major output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_right_column_to_output_row(NR, NC, mat_value, mat_index, NRHS, rhs_by_col, blocked_cr_ro_16, 16);
        return blocked_cr_ro_16.front().front() + blocked_cr_ro_16.front().back() + blocked_cr_ro_16.back().front() + blocked_cr_ro_16.back().back();
    });
 
    auto blocked_cr_ro_32 = preallocate_row_output();
    names.emplace_back("blocked (B = 32), column-major RHS, row-major output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_right_column_to_output_row(NR, NC, mat_value, mat_index, NRHS, rhs_by_col, blocked_cr_ro_32, 32);
        return blocked_cr_ro_32.front().front() + blocked_cr_ro_32.front().back() + blocked_cr_ro_32.back().front() + blocked_cr_ro_32.back().back();
    });

    // Performing the iterations.
    eztimer::Options opt;
    opt.iterations = iterations;
    opt.setup = [&]() -> void {
        reset_output(naive_rr_co);
        reset_output(naive_rr_ro);
        reset_output(naive_cr_ro);
        reset_output(best_cr_co);
        reset_output(blocked_rr_co_4);
        reset_output(blocked_rr_co_8);
        reset_output(blocked_rr_co_16);
        reset_output(blocked_rr_co_32);
        reset_output(blocked_rr_ro_4);
        reset_output(blocked_rr_ro_8);
        reset_output(blocked_rr_ro_16);
        reset_output(blocked_rr_ro_32);
        reset_output(blocked_cr_ro_4);
        reset_output(blocked_cr_ro_8);
        reset_output(blocked_cr_ro_16);
        reset_output(blocked_cr_ro_32);
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
        constexpr std::size_t pad_to = 55;
        if (name.size() < pad_to) {
            name.insert(name.end(), pad_to - name.size(), ' ');
        }
        std::cout << name << ": " << res[m].mean.count() << " ± " << res[m].sd.count() / std::sqrt(res[m].times.size()) << std::endl;
    }

    return 0;
}
