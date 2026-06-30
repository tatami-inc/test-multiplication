# Sparse row-major product with multiple vectors

## Strategies

All strategies must iterate across consecutive rows of the LHS matrix in the outermost loop.
We will be loading rows on demand via the **tatami** interface, so the full matrix will not be available for random access.

### Naive

For each LHS row, we iterate across the RHS vectors and compute the dot product against each vector.

### Accumulators

This is the same as the naive approach, except that the dot product is computed with multiple accumulators.
The idea is to break dependency chains in the CPU's instruction pipeline by allowing multiple accumulations to occur in parallel.
It also provides some opportunities for auto-vectorization of the sum.

### Blocking

For each LHS row, we consider a block of $B$ RHS columns.
For each RHS column, we compute a partial dot product using the first $C$ structural non-zeros of the LHS row.
We repeat this for all RHS columns in the block.
Then we proceed to the next $C$ structural non-zeros and repeat the calculation, accumulating dot products until all non-zeros are processed.
We repeat this entire process until all LHS rows are processed.
The idea is to keep parts of the LHS row in cache to avoid reloading it for each RHS column.
We try various values of $B$ and $C$ while keeping $BC = 1024$ to avoid dragging too much into cache.
Specifically, this avoids eviction of cache lines of RHS columns that might be accessed in the next block of $C$ non-zeros.

We do not try conventional blocking as we cannot easily process $B$ LHS rows at once.
This is not amenable to fixed-size blocking, which would require extra overhead to track the start/end of each block in each RHS row.
Alternatively, we could consider using variable-size blocks where we take a certain number of non-zero elements from multiple RHS rows;
however, if these elements have very different positions, the corresponding RHS submatrix may be very large and not fit into the cache.

We could consider using variable-size "blocks" of multiple LHS rows, which contain no more than a certain number of non-zero elements from each RHS row/column.
This would improve the efficiency of traversal along the sparse rows by making the overhead (mostly) proportional to the number of non-zero elements processed.
However, different LHS rows have different number of structural non-zeros, so towards the end of each row, we'd be wasting iterations on rows that have no more non-zeros.
Additionally, for each sparse "block" of this nature, we would still need to access the union of all indices of the structural non-zeros in the dense RHS vector.
The non-zero values can be arbitrarily distributed so we might end up using the entirety of the RHS vector. 

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
$ ./build/test -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive                                             : 0.23637 ± 0.000614496
2 accumulators                                    : 0.241283 ± 0.000466012
4 accumulators                                    : 0.218738 ± 0.000586264
8 accumulators                                    : 0.234722 ± 0.000645088
blocked 4                                         : 0.242075 ± 0.00287805
blocked 8                                         : 0.242098 ± 0.000513339
blocked 16                                        : 0.25107 ± 0.000769248
blocked 32                                        : 0.261351 ± 0.00108893
blocked 4, multiple accumulators                  : 0.224882 ± 0.000630429
blocked 8, multiple accumulators                  : 0.224451 ± 0.000582492
blocked 16, multiple accumulators                 : 0.241541 ± 0.000589569
blocked 32, multiple accumulators                 : 0.261741 ± 0.000832864

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive                                             : 0.231306 ± 0.00144139
2 accumulators                                    : 0.234472 ± 0.000600344
4 accumulators                                    : 0.208238 ± 0.000495223
8 accumulators                                    : 0.226078 ± 0.00056013
blocked 4                                         : 0.230757 ± 0.00302533
blocked 8                                         : 0.235452 ± 0.000593333
blocked 16                                        : 0.239243 ± 0.000556593
blocked 32                                        : 0.249662 ± 0.00177232
blocked 4, multiple accumulators                  : 0.212285 ± 0.00171598
blocked 8, multiple accumulators                  : 0.217252 ± 0.000436678
blocked 16, multiple accumulators                 : 0.226287 ± 0.000508018
blocked 32, multiple accumulators                 : 0.247716 ± 0.00063571

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive                                             : 0.310007 ± 0.00219874
2 accumulators                                    : 0.322858 ± 0.00106521
4 accumulators                                    : 0.33649 ± 0.000769059
8 accumulators                                    : 0.30889 ± 0.00331351
blocked 4                                         : 0.358821 ± 0.00370177
blocked 8                                         : 0.354294 ± 0.000694325
blocked 16                                        : 0.350167 ± 0.00245437
blocked 32                                        : 0.349114 ± 0.00061804
blocked 4, multiple accumulators                  : 0.403454 ± 0.000868485
blocked 8, multiple accumulators                  : 0.388477 ± 0.00226098
blocked 16, multiple accumulators                 : 0.385531 ± 0.000616976
blocked 32, multiple accumulators                 : 0.37896 ± 0.0029434

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive                                             : 0.0996442 ± 0.000221464
2 accumulators                                    : 0.0972717 ± 0.000165983
4 accumulators                                    : 0.0886371 ± 0.000285857
8 accumulators                                    : 0.0947561 ± 0.000176564
blocked 4                                         : 0.1101 ± 0.000578658
blocked 8                                         : 0.109172 ± 0.000192396
blocked 16                                        : 0.105653 ± 0.000204968
blocked 32                                        : 0.105195 ± 0.000262573
blocked 4, multiple accumulators                  : 0.0961397 ± 0.000114273
blocked 8, multiple accumulators                  : 0.0966584 ± 0.000211232
blocked 16, multiple accumulators                 : 0.0986054 ± 0.000222758
blocked 32, multiple accumulators                 : 0.105337 ± 0.000143673

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive                                             : 0.118089 ± 0.000370454
2 accumulators                                    : 0.122102 ± 0.00016739
4 accumulators                                    : 0.132069 ± 0.000345597
8 accumulators                                    : 0.128264 ± 0.000462715
blocked 4                                         : 0.186407 ± 0.00128172
blocked 8                                         : 0.181508 ± 0.000433705
blocked 16                                        : 0.179979 ± 0.000760631
blocked 32                                        : 0.178047 ± 0.000637205
blocked 4, multiple accumulators                  : 0.207981 ± 0.000384889
blocked 8, multiple accumulators                  : 0.201317 ± 0.000833191
blocked 16, multiple accumulators                 : 0.197726 ± 0.000592045
blocked 32, multiple accumulators                 : 0.198081 ± 0.000492861

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive                                             : 0.451036 ± 0.000599061
2 accumulators                                    : 0.459105 ± 0.000479968
4 accumulators                                    : 0.443242 ± 0.000570188
8 accumulators                                    : 0.45263 ± 0.000475962
blocked 4                                         : 0.450426 ± 0.00245717
blocked 8                                         : 0.456397 ± 0.000384654
blocked 16                                        : 0.463351 ± 0.000269874
blocked 32                                        : 0.475819 ± 0.00149735
blocked 4, multiple accumulators                  : 0.447573 ± 0.000247909
blocked 8, multiple accumulators                  : 0.450489 ± 0.000693172
blocked 16, multiple accumulators                 : 0.460471 ± 0.00038784
blocked 32, multiple accumulators                 : 0.476475 ± 0.000363314

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive                                             : 0.464672 ± 0.00253977
2 accumulators                                    : 0.473904 ± 0.00138776
4 accumulators                                    : 0.476267 ± 0.0154202
8 accumulators                                    : 0.488158 ± 0.010837
blocked 4                                         : 0.488684 ± 0.00540755
blocked 8                                         : 0.510444 ± 0.0126268
blocked 16                                        : 0.501943 ± 0.00489827
blocked 32                                        : 0.529142 ± 0.0155281
blocked 4, multiple accumulators                  : 0.475453 ± 0.00262923
blocked 8, multiple accumulators                  : 0.478854 ± 0.00437168
blocked 16, multiple accumulators                 : 0.50659 ± 0.0136623
blocked 32, multiple accumulators                 : 0.519172 ± 0.00928673

```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive                                             : 0.113255 ± 0.00245019
2 accumulators                                    : 0.0965071 ± 0.00192287
4 accumulators                                    : 0.0862503 ± 0.0011995
8 accumulators                                    : 0.10011 ± 0.00306974
blocked 4                                         : 0.120258 ± 0.00134813
blocked 8                                         : 0.126394 ± 0.00343799
blocked 16                                        : 0.115542 ± 0.00197923
blocked 32                                        : 0.107058 ± 0.000617219
blocked 4, multiple accumulators                  : 0.09212 ± 0.00201814
blocked 8, multiple accumulators                  : 0.092195 ± 0.00276769
blocked 16, multiple accumulators                 : 0.0946609 ± 0.00105223
blocked 32, multiple accumulators                 : 0.101811 ± 0.00212483

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive                                             : 0.11314 ± 0.00328362
2 accumulators                                    : 0.0833488 ± 0.00148131
4 accumulators                                    : 0.0741359 ± 0.00114331
8 accumulators                                    : 0.0876531 ± 0.00235068
blocked 4                                         : 0.110229 ± 0.00333482
blocked 8                                         : 0.106343 ± 0.00192468
blocked 16                                        : 0.102758 ± 0.00283745
blocked 32                                        : 0.0976403 ± 0.00165574
blocked 4, multiple accumulators                  : 0.0758526 ± 0.00161722
blocked 8, multiple accumulators                  : 0.0771823 ± 0.00252098
blocked 16, multiple accumulators                 : 0.0771074 ± 0.000999707
blocked 32, multiple accumulators                 : 0.0838433 ± 0.00271613

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive                                             : 0.151017 ± 0.00765341
2 accumulators                                    : 0.147772 ± 0.00621233
4 accumulators                                    : 0.170211 ± 0.0110143
8 accumulators                                    : 0.163039 ± 0.00863322
blocked 4                                         : 0.200259 ± 0.0126188
blocked 8                                         : 0.206509 ± 0.0139646
blocked 16                                        : 0.204192 ± 0.0126621
blocked 32                                        : 0.193379 ± 0.0123789
blocked 4, multiple accumulators                  : 0.206781 ± 0.0116979
blocked 8, multiple accumulators                  : 0.217248 ± 0.0140699
blocked 16, multiple accumulators                 : 0.210436 ± 0.0144416
blocked 32, multiple accumulators                 : 0.205276 ± 0.0120298

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive                                             : 0.0907883 ± 0.000285629
2 accumulators                                    : 0.0683808 ± 0.000297593
4 accumulators                                    : 0.0599125 ± 9.07062e-05
8 accumulators                                    : 0.0723509 ± 0.000269506
blocked 4                                         : 0.10073 ± 0.000432075
blocked 8                                         : 0.0998946 ± 0.000139762
blocked 16                                        : 0.0915959 ± 0.000233415
blocked 32                                        : 0.0874179 ± 0.000245697
blocked 4, multiple accumulators                  : 0.0648297 ± 0.000333561
blocked 8, multiple accumulators                  : 0.0646975 ± 0.000169349
blocked 16, multiple accumulators                 : 0.069478 ± 0.000282345
blocked 32, multiple accumulators                 : 0.0742206 ± 0.000106735

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive                                             : 0.0979337 ± 0.000655536
2 accumulators                                    : 0.0953958 ± 0.000689904
4 accumulators                                    : 0.100197 ± 0.000762258
8 accumulators                                    : 0.106707 ± 0.000579663
blocked 4                                         : 0.146073 ± 0.000767754
blocked 8                                         : 0.141515 ± 0.00055999
blocked 16                                        : 0.138871 ± 0.000687162
blocked 32                                        : 0.13709 ± 0.000528009
blocked 4, multiple accumulators                  : 0.152443 ± 0.000599039
blocked 8, multiple accumulators                  : 0.14419 ± 0.00048797
blocked 16, multiple accumulators                 : 0.141413 ± 0.000538184
blocked 32, multiple accumulators                 : 0.140452 ± 0.000505281

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive                                             : 0.241895 ± 0.00018675
2 accumulators                                    : 0.240363 ± 0.000107915
4 accumulators                                    : 0.23149 ± 0.000119007
8 accumulators                                    : 0.242545 ± 0.000596584
blocked 4                                         : 0.244143 ± 0.00152564
blocked 8                                         : 0.24965 ± 0.000101284
blocked 16                                        : 0.247069 ± 0.000160047
blocked 32                                        : 0.291034 ± 9.12188e-05
blocked 4, multiple accumulators                  : 0.235246 ± 0.000134351
blocked 8, multiple accumulators                  : 0.240964 ± 9.53703e-05
blocked 16, multiple accumulators                 : 0.242352 ± 0.000414107
blocked 32, multiple accumulators                 : 0.280857 ± 0.00014266

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive                                             : 0.272258 ± 0.0159429
2 accumulators                                    : 0.271219 ± 0.0155422
4 accumulators                                    : 0.251734 ± 0.00493011
8 accumulators                                    : 0.28693 ± 0.0202251
blocked 4                                         : 0.263908 ± 0.00274331
blocked 8                                         : 0.268552 ± 0.00282942
blocked 16                                        : 0.315856 ± 0.0194087
blocked 32                                        : 0.310676 ± 0.00282507
blocked 4, multiple accumulators                  : 0.257736 ± 0.00225431
blocked 8, multiple accumulators                  : 0.25846 ± 0.00249919
blocked 16, multiple accumulators                 : 0.287063 ± 0.0026964
blocked 32, multiple accumulators                 : 0.305351 ± 0.00379457
```

## Conclusion

The use of multiple accumulators usually provides a slight benefit over the naive calculation.
However, the improvement is much weaker and less consistent compared to the [single vector](../single_vector) case.
This is a bit perplexing; perhaps memory access becomes a greater bottleneck when each RHS vector needs to be reloaded into cache.
By comparison, much of the single RHS vector can be held in cache and re-used in the single vector case, which exposes the accumulation as the rate-limiting factor.

Blocking is mostly unhelpful here. 
Presumably the extra looping overhead outweighs any improvement in cache re-use.
