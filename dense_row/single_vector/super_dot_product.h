#ifndef SUPER_DOT_PRODUCT_H
#define SUPER_DOT_PRODUCT_H

#include <cstddef>
#include <numeric>

template<std::size_t num_acc_, class Iterator1_, class Iterator2_>
FLOAT super_dot_product(const std::size_t len, Iterator1_ start1, Iterator2_ start2, FLOAT initial) {
    const std::size_t cycles = len / num_acc_;
    const std::size_t remainder = len % num_acc_;
    std::array<FLOAT, num_acc_> dots{};

    // One might think to initialize the array with the first cycle of the dot products, like:
    // 
    // for (std::size_t i = 0; i < num_acc_; ++i) {
    //     dots[i] = *(start1 + i) * *(start2 + i);
    // }
    // counter = num_acc_;
    // 
    // and setting c = 1 below, which saves us one set of additions.
    // But when I test this out, this seems to be no better, and indeed, a slight pessimisation for len ~= 200.
    // I would guess that the time saved by skipping the addition is offset by the increased size of the program.

    for (std::size_t c = 0; c < cycles; ++c) {
        for (std::size_t i = 0; i < num_acc_; ++i) {
            const auto idx = c * num_acc_ + i;
            dots[i] += *(start1 + idx) * *(start2 + idx);
        }
    }

    for (std::size_t i = 0; i < remainder; ++i) {
        const auto idx = cycles * num_acc_ + i;
        initial += *(start1 + idx) * *(start2 + idx);
    }
    return std::accumulate(dots.begin(), dots.end(), initial);
}


#endif
