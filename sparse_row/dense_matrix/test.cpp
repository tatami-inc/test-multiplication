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
void blocked_mult_right_column(
    const std::size_t NR,
    const std::size_t NC,
    const std::vector<std::vector<FLOAT> >& mat_value,
    const std::vector<std::vector<int> >& mat_index,
    const std::size_t NRHS,
    const std::vector<std::vector<FLOAT> >& rhs,
    std::vector<std::vector<FLOAT> >& product,
    const std::size_t block_size 
) {
    std::size_t r = 0;
    while (r < NR) {
        const std::size_t rend = r + std::min(NR - r, block_size);
        for (std::size_t h = 0; h < NRHS; ++h) {
            const auto& rcol = rhs[h];
            for (std::size_t rcopy = r; rcopy < rend; ++rcopy) {
                const auto& mval = mat_value[rcopy];
                const auto& midx = mat_index[rcopy];
                const auto val = dense_sparse_dot_product<num_acc_>(mat_value[rcopy], mat_index[rcopy], rcol);
                if constexpr(output_columnar_) {
                    product[h][rcopy] = val;
                } else {
                    product[rcopy][h] = val;
                }
            }
        }
        r = rend;
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
    for (std::size_t r = 0; r < NR; ++r) {
        const auto& mval = mat_value[r];
        const auto& midx = mat_index[r];
        const std::size_t nnz = mval.size();
        auto& out = product[r];
        std::size_t h = 0;
        while (h < NRHS) {
            const std::size_t hend = h + std::min(NRHS - h, block_size);
            for (std::size_t x = 0; x < nnz; ++x) {
                const auto mult = mval[x];
                const auto& rightrow = rhs[midx[x]];
                for (std::size_t hcopy = h; hcopy < hend; ++hcopy) {
                    out[hcopy] += mult * rightrow[hcopy];
                }
            }
            h = hend; 
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
    std::vector<FLOAT> buffer(NRHS);
    for (std::size_t r = 0; r < NR; ++r) {
        const auto& mval = mat_value[r];
        const auto& midx = mat_index[r];
        const std::size_t nnz = mval.size();
        std::size_t h = 0;
        while (h < NRHS) {
            const std::size_t hend = h + std::min(NRHS - h, block_size);
            for (std::size_t x = 0; x < nnz; ++x) {
                const auto mult = mval[x];
                const auto& rightrow = rhs[midx[x]];
                for (std::size_t hcopy = h; hcopy < hend; ++hcopy) {
                    buffer[hcopy] += mult * rightrow[hcopy];
                }
            }
            h = hend; 
        }
        for (std::size_t h = 0; h < NRHS; ++h) {
            product[h][r] = buffer[h];
            buffer[h] = 0;
        }
    }
}

int main(int argc, char ** argv) {
    CLI::App app{"Timings for sparse row-major LHS, dense matrix RHS"};
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
    funs.reserve(5);
    names.reserve(5);

    // Naive for row-major.
    auto naive_rr_ro = preallocate_row_output();
    names.emplace_back("naive, row RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t r = 0; r < NR; ++r) {
            const auto& mval = mat_value[r];
            const auto& midx = mat_index[r];
            const std::size_t nnz = mval.size();
            auto& out = naive_rr_ro[r];
            for (std::size_t x = 0; x < nnz; ++x) {
                const auto mult = mval[x];
                const auto& rightrow = rhs_by_row[midx[x]];
                for (std::size_t h = 0; h < NRHS; ++h) {
                    out[h] += mult * rightrow[h];
                }
            }
        }
        return naive_rr_ro.front().front() + naive_rr_ro.front().back() + naive_rr_ro.back().front() + naive_rr_ro.back().back();
    });

    auto naive_rr_co = preallocate_column_output();
    names.emplace_back("naive, row RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        std::vector<FLOAT> buffer(NRHS);
        for (std::size_t r = 0; r < NR; ++r) {
            const auto& mval = mat_value[r];
            const auto& midx = mat_index[r];
            const std::size_t nnz = mval.size();
            for (std::size_t x = 0; x < nnz; ++x) {
                const auto mult = mval[x];
                const auto& rightrow = rhs_by_row[midx[x]];
                for (std::size_t h = 0; h < NRHS; ++h) {
                    buffer[h] += mult * rightrow[h];
                }
            }
            for (std::size_t h = 0; h < NRHS; ++h) {
                naive_rr_co[h][r] = buffer[h];
                buffer[h] = 0;
            }
        }
        return naive_rr_co.front().front() + naive_rr_co.front().back() + naive_rr_co.back().front() + naive_rr_co.back().back();
    });

    // Best of column major.
    auto blocked_cr_co = preallocate_column_output();
    names.emplace_back("blocked, column RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_right_column<4, true>(NR, NC, mat_value, mat_index, NRHS, rhs_by_col, blocked_cr_co, 16);
        return blocked_cr_co.front().front() + blocked_cr_co.front().back() + blocked_cr_co.back().front() + blocked_cr_co.back().back();
    });

    auto blocked_cr_ro = preallocate_row_output();
    names.emplace_back("blocked, column RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_right_column<4, false>(NR, NC, mat_value, mat_index, NRHS, rhs_by_col, blocked_cr_ro, 16);
        return blocked_cr_ro.front().front() + blocked_cr_ro.front().back() + blocked_cr_ro.back().front() + blocked_cr_ro.back().back();
    });

    // Blocked.
    auto blocked_rr_ro_128 = preallocate_row_output();
    names.emplace_back("blocked (C = 128), row RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_right_row_to_output_row(NR, NC, mat_value, mat_index, NRHS, rhs_by_row, blocked_rr_ro_128, 128);
        return blocked_rr_ro_128.front().front() + blocked_rr_ro_128.front().back() + blocked_rr_ro_128.back().front() + blocked_rr_ro_128.back().back();
    });

    auto blocked_rr_ro_256 = preallocate_row_output();
    names.emplace_back("blocked (C = 256), row RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_right_row_to_output_row(NR, NC, mat_value, mat_index, NRHS, rhs_by_row, blocked_rr_ro_256, 256);
        return blocked_rr_ro_256.front().front() + blocked_rr_ro_256.front().back() + blocked_rr_ro_256.back().front() + blocked_rr_ro_256.back().back();
    });

    auto blocked_rr_ro_512 = preallocate_row_output();
    names.emplace_back("blocked (C = 512), row RHS, row output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_right_row_to_output_row(NR, NC, mat_value, mat_index, NRHS, rhs_by_row, blocked_rr_ro_512, 512);
        return blocked_rr_ro_512.front().front() + blocked_rr_ro_512.front().back() + blocked_rr_ro_512.back().front() + blocked_rr_ro_512.back().back();
    });

    auto blocked_rr_co_128 = preallocate_column_output();
    names.emplace_back("blocked (C = 128), row RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_right_row_to_output_column(NR, NC, mat_value, mat_index, NRHS, rhs_by_row, blocked_rr_co_128, 128);
        return blocked_rr_co_128.front().front() + blocked_rr_co_128.front().back() + blocked_rr_co_128.back().front() + blocked_rr_co_128.back().back();
    });

    auto blocked_rr_co_256 = preallocate_column_output();
    names.emplace_back("blocked (C = 256), row RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_right_row_to_output_column(NR, NC, mat_value, mat_index, NRHS, rhs_by_row, blocked_rr_co_256, 256);
        return blocked_rr_co_256.front().front() + blocked_rr_co_256.front().back() + blocked_rr_co_256.back().front() + blocked_rr_co_256.back().back();
    });

    auto blocked_rr_co_512 = preallocate_column_output();
    names.emplace_back("blocked (C = 512), row RHS, column output");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult_right_row_to_output_column(NR, NC, mat_value, mat_index, NRHS, rhs_by_row, blocked_rr_co_512, 512);
        return blocked_rr_co_512.front().front() + blocked_rr_co_512.front().back() + blocked_rr_co_512.back().front() + blocked_rr_co_512.back().back();
    });

    // Performing the iterations.
    eztimer::Options opt;
    opt.iterations = iterations;
    opt.setup = [&]() -> void {
        reset_output(naive_rr_ro);
        reset_output(naive_rr_co);
        reset_output(blocked_rr_co_128);
        reset_output(blocked_rr_co_256);
        reset_output(blocked_rr_co_512);
        reset_output(blocked_rr_ro_128);
        reset_output(blocked_rr_ro_256);
        reset_output(blocked_rr_ro_512);
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
