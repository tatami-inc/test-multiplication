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

void blocked_mult_with_right_row_to_output_column(
    const std::size_t NR,
    const std::size_t NC,
    const std::vector<std::vector<FLOAT> >& matrix,
    const std::size_t NRHS,
    const std::vector<std::vector<FLOAT> >& rhs_value,
    const std::vector<std::vector<int> >& rhs_index,
    std::vector<std::vector<FLOAT> >& product,
    std::size_t line_size
) {
    for (std::size_t c = 0; c < NC; ++c) {
        const auto& matcol = matrix[c];
        const auto& values = rhs_value[c];
        const auto& indices = rhs_index[c];
        const std::size_t nnz = values.size();
        std::size_t r = 0;
        while (r < NR) { // jump by block to go down LHS columns as this is most contiguous.
            const std::size_t rend = r + std::min(line_size, NR - r);
            for (std::size_t x = 0; x < nnz; ++x) {
                auto& out = product[indices[x]];
                const auto mult = values[x];
                for (std::size_t rcopy = r; rcopy < rend; ++rcopy) {
                    out[rcopy] += matcol[rcopy] * mult;
                }
            }

            r = rend;
        }
    }
}

void blocked_mult_with_right_row_to_output_row(
    const std::size_t NR,
    const std::size_t NC,
    const std::vector<std::vector<FLOAT> >& matrix,
    const std::size_t NRHS,
    const std::vector<std::vector<FLOAT> >& rhs_value,
    const std::vector<std::vector<int> >& rhs_index,
    std::vector<std::vector<FLOAT> >& product,
    std::size_t line_size
) {
    for (std::size_t c = 0; c < NC; ++c) {
        const auto& matcol = matrix[c];
        const auto& values = rhs_value[c];
        const auto& indices = rhs_index[c];
        const std::size_t nnz = values.size();
        std::size_t x = 0;
        while (x < nnz) {
            const std::size_t xend = x + std::min(line_size, nnz - x);
            for (std::size_t r = 0; r < NR; ++r) {
                const auto mult = matcol[r];
                auto& out = product[r];
                for (std::size_t xcopy = x; xcopy < xend; ++xcopy) {
                    out[indices[xcopy]] += values[xcopy] * mult;
                }
            }
            x = xend;
        }
    }
}

int main(int argc, char ** argv) {
    CLI::App app{"Dense column matrix x sparse matrix performance tests"};
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
    std::vector<std::vector<FLOAT> > matrix(NC);
    std::normal_distribution<> dist;
    for (std::size_t c = 0; c < NC; ++c) {
        matrix[c].resize(NR);
        for (auto& x : matrix[c]) {
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
    auto naive_rr_co = preallocate_column_output();
    names.emplace_back("naive, row RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t c = 0; c < NC; ++c) {
            const auto& matcol = matrix[c];
            const auto& values = rhs_value_by_row[c];
            const auto& indices = rhs_index_by_row[c];
            const std::size_t nnz = values.size();
            for (std::size_t x = 0; x < nnz; ++x) {
                auto& out = naive_rr_co[indices[x]];
                const auto mult = values[x]; 
                for (std::size_t r = 0; r < NR; ++r) {
                    out[r] += matcol[r] * mult;
                }
            }
        }
        return naive_rr_co.front().front() + naive_rr_co.front().back() + naive_rr_co.back().front() + naive_rr_co.back().back();
    });

    auto naive_rr_ro = preallocate_row_output();
    names.emplace_back("naive, row RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t c = 0; c < NC; ++c) {
            const auto& matcol = matrix[c];
            const auto& values = rhs_value_by_row[c];
            const auto& indices = rhs_index_by_row[c];
            const std::size_t nnz = values.size();
            for (std::size_t r = 0; r < NR; ++r) {
                const auto mult = matcol[r];
                auto& out = naive_rr_ro[r];
                for (std::size_t x = 0; x < nnz; ++x) {
                    out[indices[x]] += values[x] * mult;
                }
            }
        }
        return naive_rr_ro.front().front() + naive_rr_ro.front().back() + naive_rr_ro.back().front() + naive_rr_ro.back().back();
    });

    // Blocked multiplication (row RHS, column output).
    auto blocked_rr_co_128 = preallocate_column_output();
    names.emplace_back("blocked (128), row RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_column(NR, NC, matrix, NRHS, rhs_value_by_row, rhs_index_by_row, blocked_rr_co_128, 128);
        return blocked_rr_co_128.front().front() + blocked_rr_co_128.front().back() + blocked_rr_co_128.back().front() + blocked_rr_co_128.back().back();
    });

    auto blocked_rr_co_256 = preallocate_column_output();
    names.emplace_back("blocked (256), row RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_column(NR, NC, matrix, NRHS, rhs_value_by_row, rhs_index_by_row, blocked_rr_co_256, 256);
        return blocked_rr_co_256.front().front() + blocked_rr_co_256.front().back() + blocked_rr_co_256.back().front() + blocked_rr_co_256.back().back();
    });

    auto blocked_rr_co_512 = preallocate_column_output();
    names.emplace_back("blocked (512), row RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_column(NR, NC, matrix, NRHS, rhs_value_by_row, rhs_index_by_row, blocked_rr_co_512, 512);
        return blocked_rr_co_512.front().front() + blocked_rr_co_512.front().back() + blocked_rr_co_512.back().front() + blocked_rr_co_512.back().back();
    });

    auto blocked_rr_co_1024 = preallocate_column_output();
    names.emplace_back("blocked (1024), row RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_column(NR, NC, matrix, NRHS, rhs_value_by_row, rhs_index_by_row, blocked_rr_co_1024, 1024);
        return blocked_rr_co_1024.front().front() + blocked_rr_co_1024.front().back() + blocked_rr_co_1024.back().front() + blocked_rr_co_1024.back().back();
    });

    // Blocked multiplication (row RHS, row output).
    auto blocked_rr_ro_128 = preallocate_row_output();
    names.emplace_back("blocked (128), row RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_row(NR, NC, matrix, NRHS, rhs_value_by_row, rhs_index_by_row, blocked_rr_ro_128, 128);
        return blocked_rr_ro_128.front().front() + blocked_rr_ro_128.front().back() + blocked_rr_ro_128.back().front() + blocked_rr_ro_128.back().back();
    });

    auto blocked_rr_ro_256 = preallocate_row_output();
    names.emplace_back("blocked (256), row RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_row(NR, NC, matrix, NRHS, rhs_value_by_row, rhs_index_by_row, blocked_rr_ro_256, 256);
        return blocked_rr_ro_256.front().front() + blocked_rr_ro_256.front().back() + blocked_rr_ro_256.back().front() + blocked_rr_ro_256.back().back();
    });

    auto blocked_rr_ro_512 = preallocate_row_output();
    names.emplace_back("blocked (512), row RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_row(NR, NC, matrix, NRHS, rhs_value_by_row, rhs_index_by_row, blocked_rr_ro_512, 512);
        return blocked_rr_ro_512.front().front() + blocked_rr_ro_512.front().back() + blocked_rr_ro_512.back().front() + blocked_rr_ro_512.back().back();
    });

    auto blocked_rr_ro_1024 = preallocate_row_output();
    names.emplace_back("blocked (1024), row RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_with_right_row_to_output_row(NR, NC, matrix, NRHS, rhs_value_by_row, rhs_index_by_row, blocked_rr_ro_1024, 1024);
        return blocked_rr_ro_1024.front().front() + blocked_rr_ro_1024.front().back() + blocked_rr_ro_1024.back().front() + blocked_rr_ro_1024.back().back();
    });

    // Performing the iterations.
    eztimer::Options opt;
    opt.iterations = iterations;
    opt.setup = [&]() -> void {
        reset_output(naive_rr_co);
        reset_output(naive_rr_ro);
        reset_output(blocked_rr_co_128);
        reset_output(blocked_rr_co_256);
        reset_output(blocked_rr_co_512);
        reset_output(blocked_rr_co_1024);
        reset_output(blocked_rr_ro_128);
        reset_output(blocked_rr_ro_256);
        reset_output(blocked_rr_ro_512);
        reset_output(blocked_rr_ro_1024);
    };

    std::optional<FLOAT> expected;
    auto res = eztimer::time<FLOAT>(
        funs,
        [&](const FLOAT& res, std::size_t i) -> void {
//            std::cout << res << "\t" << i << std::endl;
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
