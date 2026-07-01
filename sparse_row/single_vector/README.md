# Sparse row-major product with a single vector

## Strategies

All strategies must iterate across consecutive rows of the LHS matrix in the outermost loop.
We will be loading rows on demand via the **tatami** interface, so the full matrix will not be available for random access.

### Naive

For each LHS row, we compute the dot product against the RHS vector.

### Accumulators

This is the same as the naive approach, except that the dot product is computed with multiple accumulators.
The idea is to break dependency chains in the CPU's instruction pipeline by allowing multiple accumulations to occur in parallel.
It also provides some opportunities for auto-vectorization of the sum.

### Comments on blocking 

We don't test any blocking schemes here as it is difficult to consider a block of multiple sparse LHS rows, see [`general/README.md`](../../general/README.md) for comments.
There's only one RHS vector so there's no opportunity for blocking on the RHS either.

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
$ ./build/test -r 20000 -c 1000
Results for 20000 x 1000
naive                                             : 0.00230701 ± 1.29069e-05
2 accumulators                                    : 0.00180446 ± 2.35341e-06
4 accumulators                                    : 0.00171246 ± 7.20213e-06
8 accumulators                                    : 0.00171252 ± 7.68304e-06

$ ./build/test -r 5000 -c 10000
Results for 5000 x 10000
naive                                             : 0.00613053 ± 1.03335e-05
2 accumulators                                    : 0.00463167 ± 1.42148e-05
4 accumulators                                    : 0.00444495 ± 8.76551e-06
8 accumulators                                    : 0.00441638 ± 6.97368e-06

$ ./build/test -r 1000 -c 50000
Results for 1000 x 50000
naive                                             : 0.00683221 ± 1.23612e-05
2 accumulators                                    : 0.00658741 ± 8.13006e-06
4 accumulators                                    : 0.00623488 ± 9.91883e-06
8 accumulators                                    : 0.00624136 ± 1.04279e-05
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 20000 -c 1000
Results for 20000 x 1000
naive                                             : 0.0021056 ± 5.78579e-06
2 accumulators                                    : 0.00153783 ± 5.90938e-06
4 accumulators                                    : 0.00129449 ± 5.0177e-06
8 accumulators                                    : 0.00129928 ± 8.42645e-06

$ ./build/test_float -r 5000 -c 10000
Results for 5000 x 10000
naive                                             : 0.00575126 ± 0.000104191
2 accumulators                                    : 0.00377424 ± 2.49744e-05
4 accumulators                                    : 0.00345115 ± 2.39759e-05
8 accumulators                                    : 0.00343676 ± 3.81891e-05

$ ./build/test_float -r 1000 -c 50000
Results for 1000 x 50000
naive                                             : 0.00564083 ± 6.63435e-06
2 accumulators                                    : 0.00449082 ± 1.26535e-05
4 accumulators                                    : 0.00407817 ± 1.24714e-05
8 accumulators                                    : 0.00407757 ± 7.91102e-06
```

## Conclusion

Multiple accumulators provide a modest improvement to performance, consistent with conclusions from other projects ([IRLBA](https://github.com/libscran/irlba), and Eigen itself).
This is most apparent when the shared dimension extent is high, though it's also not too bad when the shared dimension is small.
It seems that 4 accumulators is typically the sweet spot.
