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

#include "super_dot_product.h"

template<int num_acc_>
void blocked_mult_with_right_column_to_output_column(
    const std::size_t NR,
    const std::size_t NC,
    const std::vector<std::vector<FLOAT> >& matrix,
    const std::size_t NRHS,
    const std::vector<std::vector<FLOAT> >& rhs,
    std::vector<std::vector<FLOAT> >& product,
    std::size_t block_size
) {
    const std::size_t line_size = 1024 / block_size;
    std::size_t r = 0;
    while (r < NR) {
        const std::size_t rend = r + std::min(block_size, NR - r);
        std::size_t h = 0;
        while (h < NRHS) {
            const std::size_t hend = h + std::min(block_size, NRHS - h);
            std::size_t c = 0;
            while (c < NC) { // jump by block to go down the RHS columns as this is most contiguous.
                const std::size_t cnum = std::min(line_size, NC - c);
                const std::size_t cend = c + cnum;

                for (auto hcopy = h; hcopy < hend; ++hcopy) {
                    const auto& rightcol = rhs[hcopy];
                    auto& outcol = product[hcopy];
                    for (auto rcopy = r; rcopy < rend; ++rcopy) {
                        if constexpr(num_acc_ == 1) {
                            outcol[rcopy] = std::inner_product(
                                rightcol.begin() + c,
                                rightcol.begin() + cend,
                                matrix[rcopy].begin() + c,
                                outcol[rcopy]
                            );
                        } else {
                            product[hcopy][rcopy] = super_dot_product<num_acc_>(
                                cnum,
                                rightcol.begin() + c,
                                matrix[rcopy].begin() + c,
                                outcol[rcopy]
                            );
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

template<int num_acc_>
void blocked_mult_with_right_column_to_output_row(
    const std::size_t NR,
    const std::size_t NC,
    const std::vector<std::vector<FLOAT> >& matrix,
    const std::size_t NRHS,
    const std::vector<std::vector<FLOAT> >& rhs,
    std::vector<std::vector<FLOAT> >& product,
    std::size_t block_size
) {
    const std::size_t line_size = 1024 / block_size;
    std::size_t r = 0;
    while (r < NR) {
        const std::size_t rend = r + std::min(block_size, NR - r);
        std::size_t h = 0;
        while (h < NRHS) {
            const std::size_t hend = h + std::min(block_size, NRHS - h);
            std::size_t c = 0;
            while (c < NC) { // jump by block to go down the RHS columns as this is most contiguous.
                const std::size_t cnum = std::min(line_size, NC - c);
                const std::size_t cend = c + cnum;

                for (auto rcopy = r; rcopy < rend; ++rcopy) {
                    auto& outrow = product[rcopy];
                    const auto& matrow = matrix[rcopy];
                    for (auto hcopy = h; hcopy < hend; ++hcopy) {
                        const auto& rightcol = rhs[hcopy];
                        if constexpr(num_acc_ == 1) {
                            outrow[hcopy] = std::inner_product(
                                rightcol.begin() + c,
                                rightcol.begin() + cend,
                                matrix[rcopy].begin() + c,
                                outrow[hcopy]
                            );
                        } else {
                            outrow[hcopy] = super_dot_product<num_acc_>(
                                cnum,
                                rightcol.begin() + c,
                                matrow.begin() + c,
                                outrow[hcopy]
                            );
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

void blocked_mult_with_right_row_to_output_column(
    const std::size_t NR,
    const std::size_t NC,
    const std::vector<std::vector<FLOAT> >& matrix,
    const std::size_t NRHS,
    const std::vector<std::vector<FLOAT> >& rhs,
    std::vector<std::vector<FLOAT> >& product,
    std::size_t block_size
) {
    const std::size_t line_size = 1024 / block_size;
    std::size_t r = 0;
    std::vector<std::vector<FLOAT> > buffer(block_size);
    for (auto& b : buffer) {
        b.resize(NRHS);
    }

    while (r < NR) {
        const std::size_t rend = r + std::min(block_size, NR - r);
        std::size_t c = 0;
        while (c < NC) { 
            const std::size_t cend = c + std::min(block_size, NC - c);
            std::size_t h = 0;
            while (h < NRHS) {
                const std::size_t hend = h + std::min(line_size, NRHS - h);
                for (auto rcopy = r; rcopy < rend; ++rcopy) {
                    const auto& matrow = matrix[rcopy];
                    auto& out = buffer[rcopy - r];
                    for (auto ccopy = c; ccopy < cend; ++ccopy) {
                        const auto mult = matrow[ccopy];
                        const auto& right = rhs[ccopy];
                        for (std::size_t hcopy = h; hcopy < hend; ++hcopy) {
                            out[hcopy] += mult * right[hcopy];
                        }
                    }
                }
                h = hend;
            }
            c = cend;
        }

        // Perform a blocked transposition to the output columns.
        std::size_t h = 0;
        while (h < NRHS) {
            const std::size_t hend = h + std::min(line_size, NRHS - h);
            for (auto rcopy = r; rcopy < rend; ++rcopy) {
                auto& out = buffer[rcopy - r];
                for (std::size_t hcopy = h; hcopy < hend; ++hcopy) {
                    product[hcopy][rcopy] = out[hcopy];
                }
            }
            h = hend;
        }
        for (auto& b : buffer) {
            std::fill(b.begin(), b.end(), 0);
        }
        r = rend;
    }
}

void blocked_mult_with_right_row_to_output_row(
    const std::size_t NR,
    const std::size_t NC,
    const std::vector<std::vector<FLOAT> >& matrix,
    const std::size_t NRHS,
    const std::vector<std::vector<FLOAT> >& rhs,
    std::vector<std::vector<FLOAT> >& product,
    std::size_t block_size
) {
    const std::size_t line_size = 1024 / block_size; // see logic above.
    std::size_t r = 0;
    while (r < NR) {
        const std::size_t rend = r + std::min(block_size, NR - r);
        std::size_t c = 0;
        while (c < NC) { 
            const std::size_t cend = c + std::min(block_size, NC - c);
            std::size_t h = 0;
            while (h < NRHS) { // jump by block to go to the right of the RHS rows as this is most contiguous.
                const std::size_t hend = h + std::min(line_size, NRHS - h);

                for (auto rcopy = r; rcopy < rend; ++rcopy) {
                    const auto& matrow = matrix[rcopy];
                    auto& prod = product[rcopy];
                    for (auto ccopy = c; ccopy < cend; ++ccopy) {
                        const auto mult = matrow[ccopy];
                        const auto& right = rhs[ccopy];
                        for (auto hcopy = h; hcopy < hend; ++hcopy) {
                            prod[hcopy] += mult * right[hcopy];
                        }
                    }
                }

                h = hend;
            }
            c = cend;
        }
        r = rend;
    }
}

int main(int argc, char ** argv) {
    CLI::App app{"Timings for dense row LHS, dense matrix RHS"};
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
    auto naive_rr_co = preallocate_column_output();
    names.emplace_back("naive, row RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        std::vector<FLOAT> buffer(NRHS);
        for (std::size_t r = 0; r < NR; ++r) {
            for (std::size_t c = 0; c < NC; ++c) {
                const auto mult = matrix[r][c];
                const auto& right = rhs_by_row[c];
                for (std::size_t h = 0; h < NRHS; ++h) { // ugh, lots of non-continguous writes.
                    buffer[h] += right[h] * mult;
                }
            }
            for (std::size_t h = 0; h < NRHS; ++h) {
                naive_rr_co[h][r] = buffer[h];
                buffer[h] = 0;
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
                const auto& right = rhs_by_row[c];
                for (std::size_t h = 0; h < NRHS; ++h) {
                    out[h] += right[h] * mult;
                }
            }
        }
        return naive_rr_ro.front().front() + naive_rr_ro.front().back() + naive_rr_ro.back().front() + naive_rr_ro.back().back();
    });

    // Best choices for column RHS, based on the multiple_vectors results.
    auto best_cr_co = preallocate_column_output();
    names.emplace_back("best column RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_column_to_output_column<4>(NR, NC, matrix, NRHS, rhs_by_col, best_cr_co, 16);
        return best_cr_co.front().front() + best_cr_co.front().back() + best_cr_co.back().front() + best_cr_co.back().back();
    });

    auto best_cr_ro = preallocate_row_output();
    names.emplace_back("best column RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_column_to_output_row<4>(NR, NC, matrix, NRHS, rhs_by_col, best_cr_ro, 16);
        return best_cr_ro.front().front() + best_cr_ro.front().back() + best_cr_ro.back().front() + best_cr_ro.back().back();
    });

    // Blocked multiplication (row RHS, column output).
    auto blocked_rr_co_4 = preallocate_column_output();
    names.emplace_back("blocked (B = 4), row RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_column(NR, NC, matrix, NRHS, rhs_by_row, blocked_rr_co_4, 4);
        return blocked_rr_co_4.front().front() + blocked_rr_co_4.front().back() + blocked_rr_co_4.back().front() + blocked_rr_co_4.back().back();
    });

    auto blocked_rr_co_8 = preallocate_column_output();
    names.emplace_back("blocked (B = 8), row RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_column(NR, NC, matrix, NRHS, rhs_by_row, blocked_rr_co_8, 8);
        return blocked_rr_co_8.front().front() + blocked_rr_co_8.front().back() + blocked_rr_co_8.back().front() + blocked_rr_co_8.back().back();
    });

    auto blocked_rr_co_16 = preallocate_column_output();
    names.emplace_back("blocked (B = 16), row RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_column(NR, NC, matrix, NRHS, rhs_by_row, blocked_rr_co_16, 16);
        return blocked_rr_co_16.front().front() + blocked_rr_co_16.front().back() + blocked_rr_co_16.back().front() + blocked_rr_co_16.back().back();
    });

    auto blocked_rr_co_32 = preallocate_column_output();
    names.emplace_back("blocked (B = 32), row RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_column(NR, NC, matrix, NRHS, rhs_by_row, blocked_rr_co_32, 32);
        return blocked_rr_co_32.front().front() + blocked_rr_co_32.front().back() + blocked_rr_co_32.back().front() + blocked_rr_co_32.back().back();
    });

    // Blocked multiplication (row RHS, row output).
    auto blocked_rr_ro_4 = preallocate_row_output();
    names.emplace_back("blocked (B = 4), row RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_row(NR, NC, matrix, NRHS, rhs_by_row, blocked_rr_ro_4, 4);
        return blocked_rr_ro_4.front().front() + blocked_rr_ro_4.front().back() + blocked_rr_ro_4.back().front() + blocked_rr_ro_4.back().back();
    });

    auto blocked_rr_ro_8 = preallocate_row_output();
    names.emplace_back("blocked (B = 8), row RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_row(NR, NC, matrix, NRHS, rhs_by_row, blocked_rr_ro_8, 8);
        return blocked_rr_ro_8.front().front() + blocked_rr_ro_8.front().back() + blocked_rr_ro_8.back().front() + blocked_rr_ro_8.back().back();
    });

    auto blocked_rr_ro_16 = preallocate_row_output();
    names.emplace_back("blocked (B = 16), row RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_row(NR, NC, matrix, NRHS, rhs_by_row, blocked_rr_ro_16, 16);
        return blocked_rr_ro_16.front().front() + blocked_rr_ro_16.front().back() + blocked_rr_ro_16.back().front() + blocked_rr_ro_16.back().back();
    });

    auto blocked_rr_ro_32 = preallocate_row_output();
    names.emplace_back("blocked (B = 32), row RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_row(NR, NC, matrix, NRHS, rhs_by_row, blocked_rr_ro_32, 32);
        return blocked_rr_ro_32.front().front() + blocked_rr_ro_32.front().back() + blocked_rr_ro_32.back().front() + blocked_rr_ro_32.back().back();
    });

    // Performing the iterations.
    eztimer::Options opt;
    opt.iterations = iterations;
    opt.setup = [&]() -> void {
        reset_output(naive_rr_co);
        reset_output(naive_rr_ro);
        reset_output(best_cr_co);
        reset_output(best_cr_ro);
        reset_output(blocked_rr_co_4);
        reset_output(blocked_rr_co_8);
        reset_output(blocked_rr_co_16);
        reset_output(blocked_rr_co_32);
        reset_output(blocked_rr_ro_4);
        reset_output(blocked_rr_ro_8);
        reset_output(blocked_rr_ro_16);
        reset_output(blocked_rr_ro_32);
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
        constexpr std::size_t pad_to = 44;
        if (name.size() < pad_to) {
            name.insert(name.end(), pad_to - name.size(), ' ');
        }
        std::cout << name << ": " << res[m].mean.count() << " ± " << res[m].sd.count() / std::sqrt(res[m].times.size()) << std::endl;
    }

    return 0;
}
