#ifndef DENSE_SPARSE_DOT_PRODUCT_H
#define DENSE_SPARSE_DOT_PRODUCT_H

#include <cstddef>
#include <array>
#include <numeric>

// Force unrolling to avoid relying on optimizer decisions.
// For example, older versions of GCC won't unroll at -O2, so we need to do it ourselves if we don't want a run-time nested loop.
// Fortunately, compilers will stil auto-vectorize a manually-unrolled loop, so this won't be a pessimisation in the long term.
template<std::size_t num_acc_, std::size_t counter_ = 0, class ValueIterator_, class IndexIterator_, class Dense_>
void manual_dense_sparse_dot_unroller(const std::size_t idx, ValueIterator_ vptr, IndexIterator_ iptr, const Dense_& dense, std::array<FLOAT, num_acc_>& dots) {
    dots[counter_] += dense[*(iptr + idx + counter_)] * *(vptr + idx + counter_);
    if constexpr(counter_ + 1 < num_acc_) {
        manual_dense_sparse_dot_unroller<num_acc_, counter_ + 1>(idx, vptr, iptr, dense, dots);
    }
}

template<int num_acc_, class ValueIterator_, class IndexIterator_, class Dense_>
FLOAT dense_sparse_dot_product(const std::size_t num_non_zeros, ValueIterator_ vptr, IndexIterator_ iptr, const Dense_& dense, FLOAT initial) {
    if constexpr(num_acc_ == 1) {
        FLOAT dot = initial;
        for (std::size_t i = 0; i < num_non_zeros; ++i) {
            dot += dense[*(iptr + i)] * *(vptr + i);
        }
        return dot;

    } else {
        std::array<FLOAT, num_acc_> dots{};
        const std::size_t cycles = num_non_zeros / num_acc_;
        const std::size_t remainder = num_non_zeros % num_acc_;

        for (std::size_t c = 0; c < cycles; ++c) {
            manual_dense_sparse_dot_unroller(c * num_acc_, vptr, iptr, dense, dots);
        }

        FLOAT extras = initial;
        for (std::size_t i = 0; i < remainder; ++i) {
            const auto idx = cycles * num_acc_ + i;
            extras += dense[*(iptr + idx)] * *(vptr + idx);
        }
        return std::accumulate(dots.begin(), dots.end(), extras);
    }
}

template<int num_acc_, class Values_, class Indices_, class Dense_>
FLOAT dense_sparse_dot_product(const Values_& values, const Indices_& indices, const Dense_& dense) {
    return dense_sparse_dot_product<num_acc_>(values.size(), values.begin(), indices.begin(), dense, 0);
}

#endif
