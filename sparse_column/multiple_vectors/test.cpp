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

void blocked_mult(
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

int main(int argc, char ** argv) {
    CLI::App app{"Timings for sparse column-major LHS, multiple vectors RHS"};
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
    std::vector<std::vector<FLOAT> > rhs(NRHS);
    for (std::size_t h = 0; h < NRHS; ++h) {
        rhs[h].resize(NC);
        for (auto& x : rhs[h]) {
            x = dist(rng);
        }
    }

    auto preallocate_output = [&]() -> auto {
        std::vector<std::vector<FLOAT> > output(NRHS);
        for (std::size_t h = 0; h < NRHS; ++h) {
            output[h].resize(NR);
        }
        return output;
    };

    auto reset_output = [&](std::vector<std::vector<FLOAT> >& output) -> void {
        for (std::size_t h = 0; h < NRHS; ++h) {
            std::fill(output[h].begin(), output[h].end(), 0);
        }
    };

    std::vector<std::function<FLOAT()> > funs;
    std::vector<std::string> names;
    funs.reserve(5);
    names.reserve(5);

    // Naive multiplication.
    auto naive = preallocate_output();
    names.emplace_back("naive");
    funs.emplace_back([&]() -> FLOAT {
        for (std::size_t c = 0; c < NC; ++c) {
            const auto& mval = mat_value[c];
            const auto& midx = mat_index[c];
            const std::size_t mnnz = mval.size();
            for (std::size_t h = 0; h < NRHS; ++h) {
                auto& out = naive[h];
                const auto mult = rhs[h][c];
                for (std::size_t x = 0; x < mnnz; ++x) {
                    out[midx[x]] += mult * mval[x];
                }
            }
        }
        return naive.front().front() + naive.front().back() + naive.back().front() + naive.back().back();
    });

    // Blocked.
    auto blocked_4 = preallocate_output();
    names.emplace_back("blocked (B = 4)");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult(NR, NC, mat_value, mat_index, NRHS, rhs, blocked_4, 4);
        return blocked_4.front().front() + blocked_4.front().back() + blocked_4.back().front() + blocked_4.back().back();
    });

    auto blocked_8 = preallocate_output();
    names.emplace_back("blocked (B = 8)");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult(NR, NC, mat_value, mat_index, NRHS, rhs, blocked_8, 8);
        return blocked_8.front().front() + blocked_8.front().back() + blocked_8.back().front() + blocked_8.back().back();
    });

    auto blocked_16 = preallocate_output();
    names.emplace_back("blocked (B = 16)");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult(NR, NC, mat_value, mat_index, NRHS, rhs, blocked_16, 16);
        return blocked_16.front().front() + blocked_16.front().back() + blocked_16.back().front() + blocked_16.back().back();
    });

    auto blocked_32 = preallocate_output();
    names.emplace_back("blocked (B = 32)");
    funs.emplace_back([&]() -> FLOAT {
        blocked_mult(NR, NC, mat_value, mat_index, NRHS, rhs, blocked_32, 32);
        return blocked_32.front().front() + blocked_32.front().back() + blocked_32.back().front() + blocked_32.back().back();
    });

    // Performing the iterations.
    eztimer::Options opt;
    opt.iterations = iterations;
    opt.setup = [&]() -> void {
        reset_output(naive);
        reset_output(blocked_4);
        reset_output(blocked_8);
        reset_output(blocked_16);
        reset_output(blocked_32);
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
