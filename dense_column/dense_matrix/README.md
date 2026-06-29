# Dense column-major product with dense matrix

## Strategies

All strategies must iterate across consecutive columns of the LHS matrix in the outermost loop.
We will be loading columns on demand via the **tatami** interface, so the full matrix will not be available for random access.

### Naive row-major RHS

The general idea is to compute an outer product between each RHS column and LHS row, and iteratively add the outer products together in the output matrix.

For row-major output, we iterate across each element of a LHS column and we perform a vector multiply-add for the corresponding RHS row with the corresponding output row.
We repeat this for the next pair of LHS column/RHS row until both matrices are traversed.

For column-major output, we iterate across each element of an RHS row and we perform a vector multiply-add for the corresponding LHS column with the corresponding output column.
We repeat this for the next pair of LHS column/RHS row until both matrices are traversed.

### Column-major RHS

For row-major output, we iterate across the $i$-th elements of all RHS columns and we perform a vector multiply-add of the $i$-th LHS column with the (conceptual) $i$-th output column.
We repeat this for all $i$ until all columns of the LHS matrix are traversed.
The conceptual $i$-th output column consists of the $i$-th elements of all output rows, which is very non-contiguous.
Unfortunately, there's not much else we can do here as the output's layout does not align with that of the LHS or RHS matrices.

For column-major output, we don't bother testing the naive approach as we already did so in the [`multiple_vectors`](../multiple_vectors) tests.
So, for comparison's sake, we'll just use the best approach from that test suite, i.e., blocking with $B = 16$.

### Blocking

The blocked approach computes outer products from small blocks of the input matrices, i.e., submatrices of size $BC$.
The idea is to keep data into cache for faster re-use, e.g., when a LHS column is re-used to compute the outer product with multiple RHS row elements.

- For row-major RHS with row-major output, we consider $B$-by-$C$ blocks of the RHS matrix, i.e., $B$ rows and $C$ columns.
  Once all outer products are computed for one block, we move onto the next $C$ columns of the RHS matrix, to take advantage of contiguous access along the RHS rows;
  once those are exhausted, we move onto the next $B$ rows.
- For all other configurations, we consider $C$-by-$B$ blocks of the LHS matrix, i.e., $C$ rows and $B$ columns.
  Once all outer products are computed for one block, we move onto the next $C$ rows of the LHS matrix, to take advantage of fast contiguous access along the LHS columns;
  once those are exhausted, we move onto the next $B$ columns.

We test a range of different values for the $B$ given a fixed value for $BC = 1024$, i.e., a thousand elements in the cache at once.
(The actual number of elements in the cache is more like $2BC$ as we need to hold the output submatrix as well.)
Even for 8-byte types like `double`, this should easily fit into a modern L1 cache.
We keep $B$ relatively small so that we don't have to keep a large block of columns in memory, while $C$ is relatively large to reduce overhead of the vectorizable loops.

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
naive, column RHS, row output           : 1.13441 ± 0.0142815
naive, row RHS, column output           : 0.482782 ± 0.00965907
naive, row RHS, row output              : 0.413554 ± 0.00811345
best column RHS, column output          : 0.291878 ± 0.00182005
blocked (4), column RHS, row output     : 0.877836 ± 0.0146205
blocked (8), column RHS, row output     : 0.795359 ± 0.00988325
blocked (16), column RHS, row output    : 0.681348 ± 0.00500385
blocked (32), column RHS, row output    : 0.607631 ± 0.0032298
blocked (4), row RHS, column output     : 0.338906 ± 0.00502169
blocked (8), row RHS, column output     : 0.296554 ± 0.00357441
blocked (16), row RHS, column output    : 0.296506 ± 0.00269732
blocked (32), row RHS, column output    : 0.30921 ± 0.00118227
blocked (4), row RHS, row output        : 0.343566 ± 0.00601703
blocked (8), row RHS, row output        : 0.294053 ± 0.00408511
blocked (16), row RHS, row output       : 0.287878 ± 0.00164302
blocked (32), row RHS, row output       : 0.303954 ± 0.00192049

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, column RHS, row output           : 4.3968 ± 0.00160299
naive, row RHS, column output           : 0.491406 ± 0.000674049
naive, row RHS, row output              : 0.451247 ± 0.00112919
best column RHS, column output          : 0.298156 ± 0.00095969
blocked (4), column RHS, row output     : 0.91112 ± 0.00111479
blocked (8), column RHS, row output     : 0.744472 ± 0.000640708
blocked (16), column RHS, row output    : 0.647416 ± 0.000413389
blocked (32), column RHS, row output    : 0.633537 ± 0.00100368
blocked (4), row RHS, column output     : 0.356752 ± 0.000894244
blocked (8), row RHS, column output     : 0.30916 ± 0.000527273
blocked (16), row RHS, column output    : 0.299618 ± 0.00151049
blocked (32), row RHS, column output    : 0.338742 ± 0.000480785
blocked (4), row RHS, row output        : 0.348234 ± 0.00050319
blocked (8), row RHS, row output        : 0.309436 ± 0.000629013
blocked (16), row RHS, row output       : 0.306866 ± 0.000402418
blocked (32), row RHS, row output       : 0.322249 ± 0.000534545

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, column RHS, row output           : 7.27477 ± 0.00731364
naive, row RHS, column output           : 0.903054 ± 0.00285936
naive, row RHS, row output              : 0.914655 ± 0.00643375
best column RHS, column output          : 0.295331 ± 0.00121907
blocked (4), column RHS, row output     : 1.34919 ± 0.00415898
blocked (8), column RHS, row output     : 0.97151 ± 0.00351566
blocked (16), column RHS, row output    : 0.762647 ± 0.00212331
blocked (32), column RHS, row output    : 0.662207 ± 0.0012826
blocked (4), row RHS, column output     : 0.425688 ± 0.00211845
blocked (8), row RHS, column output     : 0.34354 ± 0.00164738
blocked (16), row RHS, column output    : 0.310432 ± 0.00133179
blocked (32), row RHS, column output    : 0.330044 ± 0.00242953
blocked (4), row RHS, row output        : 0.434773 ± 0.00229862
blocked (8), row RHS, row output        : 0.345932 ± 0.00136045
blocked (16), row RHS, row output       : 0.318876 ± 0.000914844
blocked (32), row RHS, row output       : 0.315933 ± 0.00147163

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, column RHS, row output           : 1.74201 ± 0.000793512
naive, row RHS, column output           : 0.915445 ± 0.00357926
naive, row RHS, row output              : 0.920203 ± 0.00527007
best column RHS, column output          : 0.307657 ± 0.00111088
blocked (4), column RHS, row output     : 1.16387 ± 0.00572065
blocked (8), column RHS, row output     : 0.918503 ± 0.00141678
blocked (16), column RHS, row output    : 0.706605 ± 0.00104709
blocked (32), column RHS, row output    : 0.619395 ± 0.00124481
blocked (4), row RHS, column output     : 0.44251 ± 0.00203739
blocked (8), row RHS, column output     : 0.352801 ± 0.0023404
blocked (16), row RHS, column output    : 0.325707 ± 0.00118968
blocked (32), row RHS, column output    : 0.323556 ± 0.000749992
blocked (4), row RHS, row output        : 0.427461 ± 0.00281007
blocked (8), row RHS, row output        : 0.343546 ± 0.00232564
blocked (16), row RHS, row output       : 0.311809 ± 0.00113776
blocked (32), row RHS, row output       : 0.328545 ± 0.00235786

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, column RHS, row output           : 1.23857 ± 0.00513853
naive, row RHS, column output           : 0.367147 ± 0.000581963
naive, row RHS, row output              : 0.280462 ± 0.00127505
best column RHS, column output          : 0.280919 ± 0.000724379
blocked (4), column RHS, row output     : 0.644198 ± 0.00107034
blocked (8), column RHS, row output     : 0.620688 ± 0.00219818
blocked (16), column RHS, row output    : 0.5851 ± 0.000980643
blocked (32), column RHS, row output    : 0.592414 ± 0.0012067
blocked (4), row RHS, column output     : 0.290509 ± 0.000968319
blocked (8), row RHS, column output     : 0.270288 ± 0.00078978
blocked (16), row RHS, column output    : 0.282283 ± 0.000431789
blocked (32), row RHS, column output    : 0.30639 ± 0.000840931
blocked (4), row RHS, row output        : 0.284595 ± 0.000387089
blocked (8), row RHS, row output        : 0.269536 ± 0.00116972
blocked (16), row RHS, row output       : 0.281493 ± 0.00102867
blocked (32), row RHS, row output       : 0.305659 ± 0.000540296

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, column RHS, row output           : 0.676409 ± 0.00084024
naive, row RHS, column output           : 0.506011 ± 0.000781143
naive, row RHS, row output              : 0.461296 ± 0.000932597
best column RHS, column output          : 0.357614 ± 0.000786245
blocked (4), column RHS, row output     : 1.00777 ± 0.000737685
blocked (8), column RHS, row output     : 0.872415 ± 0.000906405
blocked (16), column RHS, row output    : 0.687875 ± 0.000458845
blocked (32), column RHS, row output    : 0.632406 ± 0.000730452
blocked (4), row RHS, column output     : 0.350689 ± 0.000807457
blocked (8), row RHS, column output     : 0.312998 ± 0.000701366
blocked (16), row RHS, column output    : 0.309227 ± 0.00132428
blocked (32), row RHS, column output    : 0.331712 ± 0.00054492
blocked (4), row RHS, row output        : 0.354147 ± 0.00178743
blocked (8), row RHS, row output        : 0.308809 ± 0.000794632
blocked (16), row RHS, row output       : 0.298105 ± 0.000498122
blocked (32), row RHS, row output       : 0.332964 ± 0.00120509

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, column RHS, row output           : 0.795909 ± 0.000538227
naive, row RHS, column output           : 0.368147 ± 0.000487231
naive, row RHS, row output              : 0.269331 ± 0.000347969
best column RHS, column output          : 0.301601 ± 0.00111493
blocked (4), column RHS, row output     : 0.747089 ± 0.000644499
blocked (8), column RHS, row output     : 0.704806 ± 0.000314938
blocked (16), column RHS, row output    : 0.610971 ± 0.000763304
blocked (32), column RHS, row output    : 0.600497 ± 0.000436406
blocked (4), row RHS, column output     : 0.290552 ± 0.000428626
blocked (8), row RHS, column output     : 0.274262 ± 0.000416324
blocked (16), row RHS, column output    : 0.288748 ± 0.000932867
blocked (32), row RHS, column output    : 0.317865 ± 0.000484923
blocked (4), row RHS, row output        : 0.292428 ± 0.000910848
blocked (8), row RHS, row output        : 0.270287 ± 0.000449311
blocked (16), row RHS, row output       : 0.282814 ± 0.00042584
blocked (32), row RHS, row output       : 0.30232 ± 0.00033603
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive, column RHS, row output           : 0.963736 ± 0.000576581
naive, row RHS, column output           : 0.166031 ± 0.000273309
naive, row RHS, row output              : 0.163722 ± 0.00166516
best column RHS, column output          : 0.214189 ± 0.00022007
blocked (4), column RHS, row output     : 0.682211 ± 0.000564583
blocked (8), column RHS, row output     : 0.678471 ± 0.000705271
blocked (16), column RHS, row output    : 0.595868 ± 0.000493238
blocked (32), column RHS, row output    : 0.579692 ± 0.000426309
blocked (4), row RHS, column output     : 0.217065 ± 0.000450175
blocked (8), row RHS, column output     : 0.207799 ± 0.000254512
blocked (16), row RHS, column output    : 0.20995 ± 0.00101376
blocked (32), row RHS, column output    : 0.225899 ± 0.000182569
blocked (4), row RHS, row output        : 0.162004 ± 0.000317016
blocked (8), row RHS, row output        : 0.154773 ± 0.00038742
blocked (16), row RHS, row output       : 0.15882 ± 0.000431762
blocked (32), row RHS, row output       : 0.17433 ± 0.000195172

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, column RHS, row output           : 2.52125 ± 0.000378394
naive, row RHS, column output           : 0.180163 ± 0.00201368
naive, row RHS, row output              : 0.179596 ± 0.00185877
best column RHS, column output          : 0.214984 ± 0.000612781
blocked (4), column RHS, row output     : 0.696648 ± 0.000790819
blocked (8), column RHS, row output     : 0.632568 ± 0.000600508
blocked (16), column RHS, row output    : 0.585315 ± 0.000972708
blocked (32), column RHS, row output    : 0.584572 ± 0.000731091
blocked (4), row RHS, column output     : 0.21252 ± 0.00048769
blocked (8), row RHS, column output     : 0.210169 ± 0.000762364
blocked (16), row RHS, column output    : 0.213907 ± 0.000965798
blocked (32), row RHS, column output    : 0.228507 ± 0.000349805
blocked (4), row RHS, row output        : 0.170241 ± 0.000348013
blocked (8), row RHS, row output        : 0.16282 ± 0.00020965
blocked (16), row RHS, row output       : 0.173197 ± 0.000412202
blocked (32), row RHS, row output       : 0.19367 ± 0.000347993

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, column RHS, row output           : 6.13758 ± 0.0362565
naive, row RHS, column output           : 0.50498 ± 0.0235199
naive, row RHS, row output              : 0.46745 ± 0.011917
best column RHS, column output          : 0.217233 ± 0.00166648
blocked (4), column RHS, row output     : 1.12447 ± 0.0272236
blocked (8), column RHS, row output     : 0.856431 ± 0.010773
blocked (16), column RHS, row output    : 0.688557 ± 0.0127853
blocked (32), column RHS, row output    : 0.641299 ± 0.00564188
blocked (4), row RHS, column output     : 0.267305 ± 0.00623669
blocked (8), row RHS, column output     : 0.227367 ± 0.00171564
blocked (16), row RHS, column output    : 0.221772 ± 0.00192985
blocked (32), row RHS, column output    : 0.233593 ± 0.00208
blocked (4), row RHS, row output        : 0.236901 ± 0.00671538
blocked (8), row RHS, row output        : 0.18396 ± 0.00271136
blocked (16), row RHS, row output       : 0.174996 ± 0.00160596
blocked (32), row RHS, row output       : 0.180889 ± 0.000789805

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, column RHS, row output           : 2.3348 ± 0.00893755
naive, row RHS, column output           : 0.449654 ± 0.00987343
naive, row RHS, row output              : 0.469783 ± 0.0245931
best column RHS, column output          : 0.222666 ± 0.00133135
blocked (4), column RHS, row output     : 0.964053 ± 0.00447988
blocked (8), column RHS, row output     : 0.79543 ± 0.00457205
blocked (16), column RHS, row output    : 0.67078 ± 0.00446308
blocked (32), column RHS, row output    : 0.621979 ± 0.00801092
blocked (4), row RHS, column output     : 0.275685 ± 0.0044088
blocked (8), row RHS, column output     : 0.238295 ± 0.00244195
blocked (16), row RHS, column output    : 0.227592 ± 0.00152268
blocked (32), row RHS, column output    : 0.233234 ± 0.00131918
blocked (4), row RHS, row output        : 0.20891 ± 0.0035392
blocked (8), row RHS, row output        : 0.172989 ± 0.00237846
blocked (16), row RHS, row output       : 0.166083 ± 0.00198867
blocked (32), row RHS, row output       : 0.182609 ± 0.00188322

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, column RHS, row output           : 1.15688 ± 0.00893958
naive, row RHS, column output           : 0.142334 ± 0.00132755
naive, row RHS, row output              : 0.147077 ± 0.000988311
best column RHS, column output          : 0.210323 ± 0.00111615
blocked (4), column RHS, row output     : 0.62642 ± 0.0023885
blocked (8), column RHS, row output     : 0.622685 ± 0.00346946
blocked (16), column RHS, row output    : 0.577803 ± 0.00281446
blocked (32), column RHS, row output    : 0.585009 ± 0.00320889
blocked (4), row RHS, column output     : 0.203515 ± 0.00102458
blocked (8), row RHS, column output     : 0.201153 ± 0.000998046
blocked (16), row RHS, column output    : 0.208661 ± 0.00121826
blocked (32), row RHS, column output    : 0.228056 ± 0.00106348
blocked (4), row RHS, row output        : 0.149525 ± 0.000880282
blocked (8), row RHS, row output        : 0.149175 ± 0.000708493
blocked (16), row RHS, row output       : 0.163261 ± 0.000829699
blocked (32), row RHS, row output       : 0.191225 ± 0.00115918

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 1000 x 10000
naive, column RHS, row output           : 0.649565 ± 0.00171419
naive, row RHS, column output           : 0.165934 ± 0.00386514
naive, row RHS, row output              : 0.163011 ± 0.00172838
best column RHS, column output          : 0.250118 ± 0.00433078
blocked (4), column RHS, row output     : 0.761958 ± 0.00876049
blocked (8), column RHS, row output     : 0.713609 ± 0.00474977
blocked (16), column RHS, row output    : 0.623731 ± 0.00286865
blocked (32), column RHS, row output    : 0.615694 ± 0.00243091
blocked (4), row RHS, column output     : 0.214815 ± 0.0046918
blocked (8), row RHS, column output     : 0.211446 ± 0.00281718
blocked (16), row RHS, column output    : 0.223576 ± 0.00394143
blocked (32), row RHS, column output    : 0.245115 ± 0.00245451
blocked (4), row RHS, row output        : 0.14913 ± 0.0018137
blocked (8), row RHS, row output        : 0.149208 ± 0.00213545
blocked (16), row RHS, row output       : 0.155587 ± 0.000983201
blocked (32), row RHS, row output       : 0.173115 ± 0.000372145

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 10000 x 1000
naive, column RHS, row output           : 0.68952 ± 0.00696331
naive, row RHS, column output           : 0.14378 ± 0.000711818
naive, row RHS, row output              : 0.135594 ± 0.000471568
best column RHS, column output          : 0.219401 ± 0.00156973
blocked (4), column RHS, row output     : 0.714929 ± 0.00302813
blocked (8), column RHS, row output     : 0.687691 ± 0.00438928
blocked (16), column RHS, row output    : 0.583975 ± 0.00271579
blocked (32), column RHS, row output    : 0.592146 ± 0.00201493
blocked (4), row RHS, column output     : 0.202277 ± 0.000708553
blocked (8), row RHS, column output     : 0.203417 ± 0.00087828
blocked (16), row RHS, column output    : 0.214742 ± 0.00111279
blocked (32), row RHS, column output    : 0.240292 ± 0.000904051
blocked (4), row RHS, row output        : 0.143231 ± 0.000648054
blocked (8), row RHS, row output        : 0.143925 ± 0.000503733
blocked (16), row RHS, row output       : 0.154672 ± 0.0013664
blocked (32), row RHS, row output       : 0.173755 ± 0.000688976
```

## Conclusion

Blocking consistently provides some assistance across a range of matrix shapes.
The exception is that of single-precision floats where blocking is slightly worse than the naive approach, probably because the data is already small enough to fit into cache. 
A block size of 16 seems to perform the best, or nearly so, in a wide range of scenarios.

With blocking, most configurations can be used here without too much difference in performance.
The exception is that of column-major RHS with row-major output, which is relatively slow as the inner loop involves a lot of non-contiguous access.
If row-major output is desired, it may be better to coerce the RHS matrix into a row-major format prior to multiplication.
