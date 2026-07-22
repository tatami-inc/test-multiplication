#ifndef IMPLEMENTATIONS_H
#define IMPLEMENTATIONS_H

#include <vector>
#include <array>
#include <cstddef>
#include <numeric>

template<std::size_t counter_ = 0, class Iterator1_, class Iterator2_>
void unrolled_dense_dot_product(const std::size_t idx, Iterator1_ start1, Iterator2_ start2, std::array<double, ACCUMULATORS>& dots) {
    dots[counter_] += static_cast<double>(*(start1 + idx + counter_)) * static_cast<double>(*(start2 + idx + counter_));
    if constexpr(counter_ + 1 < ACCUMULATORS) {
        unrolled_dense_dot_product<counter_ + 1>(idx, start1, start2, dots);
    }
}

template<std::size_t stride_>
double recursive_sum(std::array<double, stride_>& dots) {
    if constexpr(stride_ == 2) {
        return dots[0] + dots[1];
    } else {
        constexpr auto half_stride = stride_ / 2;
        // Increase potential for vectorization.
        std::array<double, half_stride> tmp;
        for (std::size_t s = 0; s < half_stride; ++s) {
            tmp[s] = dots[s] + dots[s + half_stride];
        }
        if constexpr(stride_ % 2 == 1) {
            return recursive_sum<stride_ / 2>(tmp) + dots[stride_ - 1];
        } else {
            return recursive_sum<stride_ / 2>(tmp);
        }
    }
}

inline double loop(const std::size_t len, const double* left, const double* right) {
    const std::size_t cycles = len / ACCUMULATORS;
    const std::size_t remainder = len % ACCUMULATORS;
    std::array<double, ACCUMULATORS> dots{};

    for (std::size_t c = 0; c < cycles; ++c) {
        for (std::size_t a = 0; a < ACCUMULATORS; ++a) {
            const auto idx = c * ACCUMULATORS + a;
            dots[a] += left[idx] * right[idx];
        }
    }

    double extra = 0;
    for (std::size_t i = 0; i < remainder; ++i) {
        const auto idx = cycles * ACCUMULATORS + i;
        extra += left[idx] * right[idx];
    }

    return std::accumulate(dots.begin(), dots.end(), extra);
}

inline double manually_unrolled(const std::size_t len, const double* left, const double* right) {
    const std::size_t cycles = len / ACCUMULATORS;
    const std::size_t remainder = len % ACCUMULATORS;
    std::array<double, ACCUMULATORS> dots{};

    for (std::size_t c = 0; c < cycles; ++c) {
        unrolled_dense_dot_product(c * ACCUMULATORS, left, right, dots);
    }

    double extra = 0;
    for (std::size_t i = 0; i < remainder; ++i) {
        const auto idx = cycles * ACCUMULATORS + i;
        extra += left[idx] * right[idx];
    }

    return std::accumulate(dots.begin(), dots.end(), extra);
}

inline double peeled(const std::size_t len, const double* left, const double* right) {
    if (len <= ACCUMULATORS) {
        // We're already checking, so we might as well provide a fast path.
        double extra = 0;
        for (std::size_t i = 0; i < len; ++i) {
            extra += left[i] * right[i];
        }
        return extra;
    }

    const std::size_t cycles = len / ACCUMULATORS;
    const std::size_t remainder = len % ACCUMULATORS;

    std::array<double, ACCUMULATORS> dots; // we'll be setting it, so no need to initialize.
    for (std::size_t a = 0; a < ACCUMULATORS; ++a) {
        dots[a] = left[a] * right[a];
    }

    for (std::size_t c = 1; c < cycles; ++c) {
        for (std::size_t a = 0; a < ACCUMULATORS; ++a) {
            const auto idx = c * ACCUMULATORS + a;
            dots[a] += left[idx] * right[idx];
        }
    }

    double extra = 0;
    for (std::size_t i = 0; i < remainder; ++i) {
        const auto idx = cycles * ACCUMULATORS + i;
        extra += left[idx] * right[idx];
    }

    return std::accumulate(dots.begin(), dots.end(), extra);
}

inline double vectorized_epilogue(const std::size_t len, const double* left, const double* right) {
    const std::size_t cycles = len / ACCUMULATORS;
    const std::size_t remainder = len % ACCUMULATORS;
    std::array<double, ACCUMULATORS> dots{};

    for (std::size_t c = 0; c < cycles; ++c) {
        for (std::size_t a = 0; a < ACCUMULATORS; ++a) {
            const auto idx = c * ACCUMULATORS + a;
            dots[a] += left[idx] * right[idx];
        }
    }

    for (std::size_t i = 0; i < remainder; ++i) {
        const auto idx = cycles * ACCUMULATORS + i;
        dots[i] += left[idx] * right[idx];
    }

    return std::accumulate(dots.begin(), dots.end(), 0.0);
}

inline double recursive_sum(const std::size_t len, const double* left, const double* right) {
    const std::size_t cycles = len / ACCUMULATORS;
    const std::size_t remainder = len % ACCUMULATORS;
    std::array<double, ACCUMULATORS> dots{};

    for (std::size_t c = 0; c < cycles; ++c) {
        for (std::size_t a = 0; a < ACCUMULATORS; ++a) {
            const auto idx = c * ACCUMULATORS + a;
            dots[a] += left[idx] * right[idx];
        }
    }

    double extra = 0;
    for (std::size_t i = 0; i < remainder; ++i) {
        const auto idx = cycles * ACCUMULATORS + i;
        extra += left[idx] * right[idx];
    }

    return extra + recursive_sum(dots);
}

#endif
