# Dense row-major product with multiple vectors

## Strategies

The naive approach involves computing dot products for each row of the LHS matrix and each of the multiple RHS vectors.

The blocked approach operates on $B$ rows of the LHS matrix and $B$ RHS vectors at a time.
For each combination of row/vector in this block, we compute the dot product of its first $C$ elements against the first $L$ elements of the RHS vector.
It repeatedly adds the dot product of the next $C$ elements until all columns of the LHS matrix have been traversed.
It proceeds to the next $B$ RHS vectors until all vectors have been traversed, and then onto the next $B$ rows until the entire LHS matrix is traversed.
We test a range of different values for the $B$ given a fixed value for $BC = 1024$, i.e., a thousand elements in the cache at once.
Even for 8-byte types like `double`, this should easily fit into a modern L1 cache.

Both the naive and blocked approaches can be augmented with the use of multiple accumulators for the dot product.
To keep things simple, we only consider the performance of blocking with 4 accumulators,
given that it's better than 2 but 8 does not provide much more benefit.

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
$ ./build/test -r 2000
Results for 2000 x 5000 x 100
naive                                   : 0.931083 ± 0.000271358
naive, two accumulators                 : 0.470815 ± 0.000189178
naive, four accumulators                : 0.270865 ± 7.64997e-05
naive, eight accumulators               : 0.28028 ± 0.000140144
blocked 4                               : 0.810837 ± 0.000181974
blocked 8                               : 0.653165 ± 0.000259827
blocked 16                              : 0.57531 ± 0.000122809
blocked 32                              : 0.460908 ± 0.000217027
blocked 4, four accumulators            : 0.235163 ± 0.000120853
blocked 8, four accumulators            : 0.224714 ± 0.000116363
blocked 16, four accumulators           : 0.218522 ± 0.000144172
blocked 32, four accumulators           : 0.232119 ± 0.000142567

$ ./build/test -r 200 -c 20000 -H 500
Results for 200 x 20000 x 500
naive                                   : 2.16632 ± 0.00181801
naive, two accumulators                 : 1.30129 ± 0.00205328
naive, four accumulators                : 1.1237 ± 0.00181062
naive, eight accumulators               : 1.11454 ± 0.00205761
blocked 4                               : 1.74719 ± 0.00153097
blocked 8                               : 1.39932 ± 0.00121046
blocked 16                              : 1.24309 ± 0.00878583
blocked 32                              : 1.06029 ± 0.00408846
blocked 4, four accumulators            : 0.659002 ± 0.00199716
blocked 8, four accumulators            : 0.5772 ± 0.00140272
blocked 16, four accumulators           : 0.545833 ± 0.000845653
blocked 32, four accumulators           : 0.629646 ± 0.00102681

$ ./build/test -r 200 -c 1000 -H 10000
Results for 200 x 1000 x 10000
naive                                   : 1.97432 ± 0.00788377
naive, two accumulators                 : 1.19039 ± 0.0121833
naive, four accumulators                : 1.01601 ± 0.00723093
naive, eight accumulators               : 1.01028 ± 0.00978623
blocked 4                               : 1.71058 ± 0.0104521
blocked 8                               : 1.33874 ± 0.00156847
blocked 16                              : 1.17088 ± 0.00263961
blocked 32                              : 0.935107 ± 0.00320983
blocked 4, four accumulators            : 0.634075 ± 0.010078
blocked 8, four accumulators            : 0.522427 ± 0.00580233
blocked 16, four accumulators           : 0.46607 ± 0.000651084
blocked 32, four accumulators           : 0.484209 ± 0.00317751
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 2000
Results for 2000 x 5000 x 100
naive                                   : 0.952828 ± 0.00559331
naive, two accumulators                 : 0.493037 ± 0.00702512
naive, four accumulators                : 0.246303 ± 0.00132168
naive, eight accumulators               : 0.145701 ± 0.00137546
blocked 4                               : 0.801249 ± 0.00520382
blocked 8                               : 0.624466 ± 0.000845246
blocked 16                              : 0.562254 ± 0.00169971
blocked 32                              : 0.431991 ± 0.00182126
blocked 4, four accumulators            : 0.199216 ± 0.00335436
blocked 8, four accumulators            : 0.189622 ± 0.00096756
blocked 16, four accumulators           : 0.170519 ± 0.000674691
blocked 32, four accumulators           : 0.173001 ± 0.000520807

$ ./build/test_float -r 200 -c 20000 -H 500
Results for 200 x 20000 x 500
naive                                   : 2.07496 ± 0.00101154
naive, two accumulators                 : 1.13732 ± 0.000960123
naive, four accumulators                : 0.650955 ± 0.00153222
naive, eight accumulators               : 0.541223 ± 0.00197016
blocked 4                               : 1.62648 ± 0.00107627
blocked 8                               : 1.26175 ± 0.000351013
blocked 16                              : 1.12787 ± 0.000914332
blocked 32                              : 0.867131 ± 0.000508647
blocked 4, four accumulators            : 0.432471 ± 0.00142198
blocked 8, four accumulators            : 0.401382 ± 0.000592332
blocked 16, four accumulators           : 0.351643 ± 0.000766802
blocked 32, four accumulators           : 0.35141 ± 0.00047543

$ ./build/test_float -r 200 -c 1000 -H 10000
Results for 200 x 1000 x 10000
naive                                   : 2.04689 ± 0.000611888
naive, two accumulators                 : 1.13471 ± 0.00070399
naive, four accumulators                : 0.6474 ± 0.00128234
naive, eight accumulators               : 0.555074 ± 0.000587176
blocked 4                               : 1.66804 ± 0.00128785
blocked 8                               : 1.27279 ± 0.000391161
blocked 16                              : 1.13222 ± 0.00144957
blocked 32                              : 0.865212 ± 0.0022863
blocked 4, four accumulators            : 0.485484 ± 0.000322224
blocked 8, four accumulators            : 0.424536 ± 0.000320396
blocked 16, four accumulators           : 0.363588 ± 0.000390151
blocked 32, four accumulators           : 0.355772 ± 0.000478127
```

## Conclusion

Both blocking and the use of multiple accumulators provide a performance boost, and moreso when combined.
Blocking improves cache efficiency as each RHS vector does not need to be constantly reloaded per LHS row; rather, each RHS vector is only reloaded once per $B$ LHS rows.
Multiple accumulators improve CPU throughput, which complements the improved cache behavior.

We see that $B = 16$ with 4 accumulators performs best (or nearly so) in all tested scenarios.
Presumably, this offers the best compromise between cache efficiency and looping overhead.
Using more or fewer accumulators doesn't seem to change much so we might as well go with 4 seems like a decent number.
