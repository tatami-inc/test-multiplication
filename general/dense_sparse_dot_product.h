#ifndef DENSE_SPARSE_DOT_PRODUCT_H
#define DENSE_SPARSE_DOT_PRODUCT_H

#include <cstddef>
#include <array>
#include <numeric>

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
            for (std::size_t i = 0; i < num_acc_; ++i) {
                const auto idx = c * num_acc_ + i;
                dots[i] += dense[*(iptr + idx)] * *(vptr + idx);
            }
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
