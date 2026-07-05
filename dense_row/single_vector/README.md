# Dense row-major LHS, single vector RHS

## Strategies

All strategies must iterate across consecutive rows of the LHS matrix in the outermost loop.
We will be loading rows on demand via the **tatami** interface, so the full matrix will not be available for random access.

### Naive

For each LHS row, we compute the dot product against the RHS vector.

### Naive with multiple accumulators

This is the same as the naive approach, except that the dot product is computed with multiple accumulators.
Check out the explanation in [`general/README.md`](../../general/README.md).

### Blocking

We consider a $B$-by-$C$ LHS submatrix, a length-$C$ RHS slice and a length-$B$ output slice.
In this set of submatrices/slices, each LHS column corresponds to an RHS entry, and each LHS row corresponds to an output entry.

For each row $i$ in the LHS submatrix, we compute its partial dot product with each column $j$ of the RHS submatrix and store it in the $i$-th entry of the output slice.
We repeat this for all valid sets of submatrices/slices until the full matrix product is computed.
When choosing the next set of submatrices/slices, the LHS columns are the fastest-changing dimension while the LHS rows are the slowest.

The idea is to keep the RHS vector in cache for each block of $B$ rows, such that we don't have to reload it for every single row.
We test a range of different values for the $B$ given a fixed value for $BC = 1024$, i.e., a thousand elements in the cache at once.
Check out [`general/README.md`](../../general/README.md) for more details.
(Technically, our RHS "submatrix" only has one column so we could reasonably increase $BC$ while still fitting in a typical L1 cache.
But for simplicity's sake, we will stick with the limit used in the other tests.)

### Blocking with multiple accumulators

The blocking approach can also use multiple accumulators in each of its partial dot products.
To keep things simple, we only consider the performance of blocking with 4 accumulators,
given that it's better than 2 and only slightly worse than 8 in the naive approach.

## Instructions

To build the test executable, use the usual CMake process:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Double-precision results

Trying out a range of matrix shapes.
All timings below are obtained on an Intel i7-8850H.

```console
$ ./build/test
Results for 10000 x 5000
naive                                   : 0.054 ± 8.22871e-05
naive, 2 accumulators                   : 0.0307047 ± 5.10532e-05
naive, 4 accumulators                   : 0.0268394 ± 5.75564e-05
naive, 8 accumulators                   : 0.026854 ± 0.000149206
blocked (B = 4)                         : 0.0413164 ± 0.000120829
blocked (B = 8)                         : 0.0307188 ± 5.32347e-05
blocked (B = 16)                        : 0.0308984 ± 0.000112529
blocked (B = 32)                        : 0.0338886 ± 0.000173706
blocked (B = 4), 4 accumulators         : 0.0279144 ± 0.000532351
blocked (B = 8), 4 accumulators         : 0.0302185 ± 5.8495e-05
blocked (B = 16), 4 accumulators        : 0.0270262 ± 0.000121249
blocked (B = 32), 4 accumulators        : 0.0283397 ± 0.000130035

$ ./build/test -r 200 -c 50000
Results for 200 x 50000
naive                                   : 0.0108439 ± 2.91234e-05
naive, 2 accumulators                   : 0.00646193 ± 0.000153859
naive, 4 accumulators                   : 0.00576936 ± 4.73473e-05
naive, 8 accumulators                   : 0.005692 ± 4.20082e-05
blocked (B = 4)                         : 0.0102212 ± 0.000452175
blocked (B = 8)                         : 0.00843814 ± 3.40908e-05
blocked (B = 16)                        : 0.00882655 ± 3.23727e-05
blocked (B = 32)                        : 0.00804491 ± 3.27883e-05
blocked (B = 4), 4 accumulators         : 0.0062029 ± 0.00022658
blocked (B = 8), 4 accumulators         : 0.00640366 ± 7.41181e-05
blocked (B = 16), 4 accumulators        : 0.00667484 ± 8.01836e-05
blocked (B = 32), 4 accumulators        : 0.00694721 ± 3.33449e-05

$ ./build/test -r 50000 -c 200
Results for 50000 x 200
naive                                   : 0.00948304 ± 2.47682e-05
naive, 2 accumulators                   : 0.00591148 ± 2.7944e-05
naive, 4 accumulators                   : 0.005392 ± 1.95814e-05
naive, 8 accumulators                   : 0.00546011 ± 4.72608e-05
blocked (B = 4)                         : 0.00963489 ± 2.50798e-05
blocked (B = 8)                         : 0.00894947 ± 4.34422e-05
blocked (B = 16)                        : 0.00952565 ± 0.000120571
blocked (B = 32)                        : 0.00992812 ± 0.00113073
blocked (B = 4), 4 accumulators         : 0.00569053 ± 0.00025808
blocked (B = 8), 4 accumulators         : 0.00639567 ± 4.23242e-05
blocked (B = 16), 4 accumulators        : 0.00691397 ± 0.000134848
blocked (B = 32), 4 accumulators        : 0.00672805 ± 3.04725e-05
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float
Results for 10000 x 5000
naive                                   : 0.0529079 ± 0.000193716
naive, 2 accumulators                   : 0.0287049 ± 0.00011162
naive, 4 accumulators                   : 0.0156929 ± 6.10475e-05
naive, 8 accumulators                   : 0.0134603 ± 3.0829e-05
blocked (B = 4)                         : 0.048711 ± 0.000202014
blocked (B = 8)                         : 0.0445642 ± 0.000270465
blocked (B = 16)                        : 0.0364819 ± 0.000112232
blocked (B = 32)                        : 0.0327344 ± 7.13401e-05
blocked (B = 4), 4 accumulators         : 0.0158104 ± 0.000188776
blocked (B = 8), 4 accumulators         : 0.0168891 ± 3.75229e-05
blocked (B = 16), 4 accumulators        : 0.0172761 ± 3.67064e-05
blocked (B = 32), 4 accumulators        : 0.015765 ± 4.56557e-05

$ ./build/test_float -r 200 -c 50000
Results for 200 x 50000
naive                                   : 0.0103902 ± 3.25234e-05
naive, 2 accumulators                   : 0.00568034 ± 2.82777e-05
naive, 4 accumulators                   : 0.00309989 ± 2.15332e-05
naive, 8 accumulators                   : 0.00273271 ± 2.35036e-05
blocked (B = 4)                         : 0.00947168 ± 8.29157e-05
blocked (B = 8)                         : 0.00857786 ± 3.61417e-05
blocked (B = 16)                        : 0.00692126 ± 2.12998e-05
blocked (B = 32)                        : 0.0066643 ± 2.27636e-05
blocked (B = 4), 4 accumulators         : 0.00324065 ± 2.83608e-05
blocked (B = 8), 4 accumulators         : 0.0032029 ± 4.86237e-05
blocked (B = 16), 4 accumulators        : 0.00349992 ± 3.00997e-05
blocked (B = 32), 4 accumulators        : 0.00360221 ± 2.82781e-05

$ ./build/test_float -r 50000 -c 200
Results for 50000 x 200
naive                                   : 0.00980517 ± 7.33511e-05
naive, 2 accumulators                   : 0.00507253 ± 1.50702e-05
naive, 4 accumulators                   : 0.00285418 ± 2.55216e-05
naive, 8 accumulators                   : 0.0027371 ± 7.65567e-06
blocked (B = 4)                         : 0.00844719 ± 2.2577e-05
blocked (B = 8)                         : 0.00836316 ± 1.53324e-05
blocked (B = 16)                        : 0.00720698 ± 2.26424e-05
blocked (B = 32)                        : 0.0073962 ± 1.9011e-05
blocked (B = 4), 4 accumulators         : 0.00289314 ± 2.29114e-05
blocked (B = 8), 4 accumulators         : 0.00287145 ± 2.24661e-05
blocked (B = 16), 4 accumulators        : 0.00329408 ± 2.32918e-05
blocked (B = 32), 4 accumulators        : 0.00369884 ± 3.25145e-05
```

## Conclusion

Use of multiple accumulators helps a lot, presumably because it improves parallelization of instructions within each CPU.

Blocking gives a moderate speed-up with $B = 16$.
This is probably due to improved cache efficiency, as the RHS vector does not need to be constantly reloaded per LHS row.
However, the naive approach with multiple accumulators is still faster that blocking, with or without multiple accumulators.
It seems that the lack of looping overhead outweighs the theoretical cache suboptimality.

This result is helpful as it means that we can get good performance without blocking.
Specifically, blocking is annoying as we need to allocate more space to load multiple rows from a **tatami** matrix.
This would force us to make judgement calls on the speed-memory trade-off.
Now, we can just load a single row and use the naive approach with multiple accumulators.
