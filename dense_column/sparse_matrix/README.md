# Dense column-major product with sparse matrix

## Strategies

All strategies must iterate across consecutive columns of the LHS matrix in the outermost loop.
We will be loading columns on demand via the **tatami** interface, so the full matrix will not be available for random access.

### Column-major RHS

When dealing with a sparse matrix, we always want to traverse the structural non-zeros for one of the dimensions.
However, walking down the RHS column would require all the LHS columns to be in memory to compute the cumulative dot products.
We can't effectively block, either, because we can't easily estimate the boundaries of the block submatrices within a sparse matrix.

Which is not to say that this is impossible.
We can iterate across all of the RHS columns to construct each row as it is needed.
However, this is not very amenable to parallelization as each thread would need to search for the start and end of its block of rows in each column.
At that point, it is better to just construct a row-major representation for the RHS matrix in the first place.

Thus, for the rest of this document, we'll restrict our consideration to a row-major sparse RHS matrix. 

### Naive row-major RHS

The general idea is to compute an outer product between each LHS column and RHS row, and iteratively add the outer products together in the output matrix.

For column-major output, we can iterate across the structural non-zeros in each RHS row.
For each structural non-zero, we perform a vectorized multiply-add of the corresponding LHS column to the corresponding column of the output matrix.

For row-major output, we iterate down the LHS column and compute a sparse vector multiply-add of the corresponding RHS row to the corresponding row of the output matrix.
This will be less efficient than the column-major output as the writes will not be contiguous, but oh well.

### Blocked row-major RHS

For column output, we consider a block of $C$ elements of each LHS column when computing the outer product for each structural non-zero of an RHS row.
Once all structural non-zeros are processed, we move to the next block of $C$ elements and repeat the outer product calculation.
This is done until the entire LHS column is processed, and then we move onto the next column.
Our hope is to improve caching of the "active" part of each LHS column when it is repeatedly used to compute the outer product across each RHS row.
Otherwise, for LHS matrices with many rows, the start of the column would need to be loaded back into cache per non-zero element in the RHS row.

For row output, we follow the principles described in the "Blocking to cache the dense vector" section of [`general/README.md`](../../general.README.md).
We load a block of $B$ LHS columns.
Then we iterate over the LHS rows, where for each LHS row, we loop over $B$ RHS rows and perform a sparse matrix multiply-add to the corresponding output row.
This is repeated with the next set of $B$ LHS columns until the entire LHS matrix is traversed.
The hope is to improve caching of the dense output vector when $B$ sparse vectors are added to it.

More conventional blocking schemes are difficult to implement for sparse data, see [`general/README.md`](../../general/README.md) for discussion.

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
naive, row RHS, column output                     : 0.0382613 ± 0.00108262
naive, row RHS, row output                        : 0.27955 ± 0.0412399
blocked (128), row RHS, column output             : 0.0546305 ± 0.00281208
blocked (256), row RHS, column output             : 0.0611033 ± 0.00606461
blocked (512), row RHS, column output             : 0.0485472 ± 0.00381691
blocked (1024), row RHS, column output            : 0.0512083 ± 0.00544596
blocked (4), row RHS, row output                  : 0.131512 ± 0.0057078
blocked (8), row RHS, row output                  : 0.106446 ± 0.00354658
blocked (16), row RHS, row output                 : 0.0987169 ± 0.0053635
blocked (32), row RHS, row output                 : 0.0927441 ± 0.00601149

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row RHS, column output                     : 0.0517356 ± 0.00267864
naive, row RHS, row output                        : 0.275152 ± 0.00883477
blocked (128), row RHS, column output             : 0.0441317 ± 0.00234766
blocked (256), row RHS, column output             : 0.0495983 ± 0.00305395
blocked (512), row RHS, column output             : 0.0473876 ± 0.00344251
blocked (1024), row RHS, column output            : 0.0466181 ± 0.00185135
blocked (4), row RHS, row output                  : 0.14254 ± 0.00618068
blocked (8), row RHS, row output                  : 0.101491 ± 0.00317978
blocked (16), row RHS, row output                 : 0.0886108 ± 0.00207807
blocked (32), row RHS, row output                 : 0.0880822 ± 0.00136578

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row RHS, column output                     : 0.0954991 ± 0.00234279
naive, row RHS, row output                        : 0.730352 ± 0.0109643
blocked (128), row RHS, column output             : 0.104751 ± 0.00305071
blocked (256), row RHS, column output             : 0.104459 ± 0.00270958
blocked (512), row RHS, column output             : 0.103189 ± 0.0026147
blocked (1024), row RHS, column output            : 0.096452 ± 0.00145977
blocked (4), row RHS, row output                  : 0.292895 ± 0.00595696
blocked (8), row RHS, row output                  : 0.190024 ± 0.00378832
blocked (16), row RHS, row output                 : 0.133282 ± 0.00233815
blocked (32), row RHS, row output                 : 0.103843 ± 0.00136087

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row RHS, column output                     : 0.111341 ± 0.00499959
naive, row RHS, row output                        : 0.74231 ± 0.0267971
blocked (128), row RHS, column output             : 0.147807 ± 0.0178216
blocked (256), row RHS, column output             : 0.130947 ± 0.00614717
blocked (512), row RHS, column output             : 0.117188 ± 0.00275232
blocked (1024), row RHS, column output            : 0.110137 ± 0.00348756
blocked (4), row RHS, row output                  : 0.315526 ± 0.0079869
blocked (8), row RHS, row output                  : 0.201465 ± 0.00285079
blocked (16), row RHS, row output                 : 0.145581 ± 0.00116541
blocked (32), row RHS, row output                 : 0.11481 ± 0.000880796

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row RHS, column output                     : 0.0321749 ± 0.000166408
naive, row RHS, row output                        : 0.132961 ± 0.000127396
blocked (128), row RHS, column output             : 0.0361085 ± 8.70949e-05
blocked (256), row RHS, column output             : 0.0373698 ± 0.000129677
blocked (512), row RHS, column output             : 0.035544 ± 7.96733e-05
blocked (1024), row RHS, column output            : 0.0347656 ± 0.000160004
blocked (4), row RHS, row output                  : 0.0827407 ± 0.00017475
blocked (8), row RHS, row output                  : 0.0765929 ± 0.000136637
blocked (16), row RHS, row output                 : 0.0779572 ± 0.000226027
blocked (32), row RHS, row output                 : 0.0856737 ± 0.000261498

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, row RHS, column output                     : 0.0666157 ± 0.00197137
naive, row RHS, row output                        : 0.253484 ± 0.0152084
blocked (128), row RHS, column output             : 0.0749495 ± 0.00368563
blocked (256), row RHS, column output             : 0.0807689 ± 0.00561185
blocked (512), row RHS, column output             : 0.073617 ± 0.00263882
blocked (1024), row RHS, column output            : 0.073312 ± 0.00290372
blocked (4), row RHS, row output                  : 0.146789 ± 0.00491561
blocked (8), row RHS, row output                  : 0.110101 ± 0.00324162
blocked (16), row RHS, row output                 : 0.099211 ± 0.00228832
blocked (32), row RHS, row output                 : 0.0925284 ± 0.0023735

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, row RHS, column output                     : 0.0360064 ± 0.000115595
naive, row RHS, row output                        : 0.121573 ± 0.000343223
blocked (128), row RHS, column output             : 0.0396995 ± 0.000102005
blocked (256), row RHS, column output             : 0.0399962 ± 0.000129911
blocked (512), row RHS, column output             : 0.0401837 ± 0.000298805
blocked (1024), row RHS, column output            : 0.0399487 ± 0.000187784
blocked (4), row RHS, row output                  : 0.0836941 ± 0.000209797
blocked (8), row RHS, row output                  : 0.0746061 ± 0.00028528
blocked (16), row RHS, row output                 : 0.0694784 ± 0.000323153
blocked (32), row RHS, row output                 : 0.0675504 ± 0.000133718
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive, row RHS, column output                     : 0.0160336 ± 0.00119729
naive, row RHS, row output                        : 0.0880649 ± 0.00159922
blocked (128), row RHS, column output             : 0.0213458 ± 0.000171325
blocked (256), row RHS, column output             : 0.0191606 ± 0.000112949
blocked (512), row RHS, column output             : 0.0202057 ± 0.00162055
blocked (1024), row RHS, column output            : 0.0165851 ± 6.93575e-05
blocked (4), row RHS, row output                  : 0.0656131 ± 0.00042278
blocked (8), row RHS, row output                  : 0.0622599 ± 0.00141482
blocked (16), row RHS, row output                 : 0.0587727 ± 0.000396065
blocked (32), row RHS, row output                 : 0.0599779 ± 0.000578869

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row RHS, column output                     : 0.0170441 ± 7.63969e-05
naive, row RHS, row output                        : 0.0993793 ± 0.00178118
blocked (128), row RHS, column output             : 0.0182264 ± 5.99917e-05
blocked (256), row RHS, column output             : 0.019053 ± 0.000858089
blocked (512), row RHS, column output             : 0.0189546 ± 0.000554487
blocked (1024), row RHS, column output            : 0.0188844 ± 0.000798865
blocked (4), row RHS, row output                  : 0.0699941 ± 0.00237746
blocked (8), row RHS, row output                  : 0.0631289 ± 0.000172379
blocked (16), row RHS, row output                 : 0.0636844 ± 0.00123093
blocked (32), row RHS, row output                 : 0.0653346 ± 0.000221161

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row RHS, column output                     : 0.0417609 ± 0.000186815
naive, row RHS, row output                        : 0.402959 ± 0.0146882
blocked (128), row RHS, column output             : 0.0542227 ± 0.00154987
blocked (256), row RHS, column output             : 0.0487254 ± 0.00226327
blocked (512), row RHS, column output             : 0.0499641 ± 0.00371017
blocked (1024), row RHS, column output            : 0.0512505 ± 0.00260315
blocked (4), row RHS, row output                  : 0.13926 ± 0.00402431
blocked (8), row RHS, row output                  : 0.0960424 ± 0.000658447
blocked (16), row RHS, row output                 : 0.0823079 ± 0.00200762
blocked (32), row RHS, row output                 : 0.0701545 ± 0.000741408

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row RHS, column output                     : 0.0516368 ± 0.00165685
naive, row RHS, row output                        : 0.391667 ± 0.0154296
blocked (128), row RHS, column output             : 0.065984 ± 0.00563036
blocked (256), row RHS, column output             : 0.0570945 ± 0.000404293
blocked (512), row RHS, column output             : 0.0659909 ± 0.00552408
blocked (1024), row RHS, column output            : 0.0528243 ± 0.00186417
blocked (4), row RHS, row output                  : 0.157912 ± 0.00397713
blocked (8), row RHS, row output                  : 0.11135 ± 0.00225319
blocked (16), row RHS, row output                 : 0.0876062 ± 0.000625303
blocked (32), row RHS, row output                 : 0.0781889 ± 0.00267377

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row RHS, column output                     : 0.0161384 ± 0.000376463
naive, row RHS, row output                        : 0.0947564 ± 0.000336538
blocked (128), row RHS, column output             : 0.0191226 ± 0.000389925
blocked (256), row RHS, column output             : 0.0190129 ± 0.00056677
blocked (512), row RHS, column output             : 0.0183748 ± 0.00016008
blocked (1024), row RHS, column output            : 0.0179056 ± 0.000590553
blocked (4), row RHS, row output                  : 0.0680196 ± 0.000382714
blocked (8), row RHS, row output                  : 0.0668513 ± 0.000978892
blocked (16), row RHS, row output                 : 0.0667488 ± 0.000890778
blocked (32), row RHS, row output                 : 0.0736452 ± 0.000586625

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, row RHS, column output                     : 0.0280162 ± 0.00175273
naive, row RHS, row output                        : 0.0997651 ± 0.00234908
blocked (128), row RHS, column output             : 0.031345 ± 0.00165664
blocked (256), row RHS, column output             : 0.0331905 ± 0.00245808
blocked (512), row RHS, column output             : 0.0318022 ± 0.00177606
blocked (1024), row RHS, column output            : 0.0312771 ± 0.00130039
blocked (4), row RHS, row output                  : 0.0752692 ± 0.00269995
blocked (8), row RHS, row output                  : 0.0671601 ± 0.00173445
blocked (16), row RHS, row output                 : 0.064585 ± 0.00142504
blocked (32), row RHS, row output                 : 0.0630742 ± 0.000949918

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, row RHS, column output                     : 0.0199717 ± 0.000151419
naive, row RHS, row output                        : 0.0860514 ± 0.000420687
blocked (128), row RHS, column output             : 0.023898 ± 0.000216278
blocked (256), row RHS, column output             : 0.0242719 ± 0.000252552
blocked (512), row RHS, column output             : 0.0234015 ± 0.000301838
blocked (1024), row RHS, column output            : 0.0231707 ± 0.000252311
blocked (4), row RHS, row output                  : 0.063899 ± 0.000428684
blocked (8), row RHS, row output                  : 0.0603694 ± 0.000579407
blocked (16), row RHS, row output                 : 0.0584591 ± 0.000577096
blocked (32), row RHS, row output                 : 0.0578164 ± 0.000441026
```

## Conclusion

For column-major output, it seems like blocking doesn't help much.
Even with larger LHS matrices where the first column cannot fit into the cache, we don't see a big benefit from blocking: 

```console
$ ./build/test -r 500000 -c 20 -H 200
Results for 500000 x 20 x 200
naive, row RHS, column output                     : 0.193006 ± 0.00312997
naive, row RHS, row output                        : 1.36091 ± 0.0364098
blocked (128), row RHS, column output             : 0.216437 ± 0.00195136
blocked (256), row RHS, column output             : 0.212758 ± 0.00220637
blocked (512), row RHS, column output             : 0.188694 ± 0.00147777
blocked (1024), row RHS, column output            : 0.185769 ± 0.00205918
blocked (128), row RHS, row output                : 1.31491 ± 0.0142829
blocked (256), row RHS, row output                : 1.30407 ± 0.0116512
blocked (512), row RHS, row output                : 1.31817 ± 0.0188771
blocked (1024), row RHS, row output               : 1.32232 ± 0.0182209
```

I'd guess that the cache misses in the naive approach are either masked by some kind of prefetching, and/or they're offset by the extra looping overhead. 
This is a perfectly acceptable result as the naive approach is much easier to implement anyway.

For row-major output, blocking is much more helpful.
The writes might be non-contiguous but blocking does quite well at keeping the output vector in cache to give us near-column-major performance.
