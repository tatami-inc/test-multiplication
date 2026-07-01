# Dense row-major LHS, dense matrix RHS

## Strategies

All strategies must iterate across consecutive rows of the LHS matrix in the outermost loop.
We will be loading rows on demand via the **tatami** interface, so the full matrix will not be available for random access.

### Naive row-major RHS

The general idea is to accumulate the partial dot products by processing each RHS row.

If the output is row-major: for each LHS row, we iterate across the RHS rows.
For each RHS row, we perform a vector multiply-add to the output row using the corresponding element from the LHS row as the scaling factor.
Each multiply-add involves contiguous memory and is highly amenable to auto-vectorization.

If the output is column-major, the process is much the same except that the output is done with conceptual output rows.
The conceptual $i$-th output row consists of the $i$-th element from each output column, which is quite non-contiguous.
There's not much that can be done here as the output's layout does not align with that of the LHS or RHS matrices.

### Blocking, row-major RHS

We consider a $B$-by-$B$ LHS submatrix with $B$-by-$C$ RHS and output submatrices.
For each part of the RHS row in this block, we perform a vector multipy-add to the corresponding part of the output row with a scaling factor from the LHS block. 
We repeat this with the next $C$ columns of the RHS matrix, to take advantage of fast access to contiguous memory.
Once all RHS columns are traversed, we move onto the next $B$ RHS rows;
and once all RHS rows are traversed, we move onto the next $B$ LHS rows.

The hope is that these submatrices are small enough to keep in cache for fast access.
We test a range of different values for the $B$ given a fixed value for $BC = 1024$. 
Check out [`general/README.md`](../../general/README.md) for more details.

### Column-major RHS

For column-major output, we already tested the naive and blocked approaches in the [`multiple_vectors`](../multiple_vectors) tests.
So, for comparison's sake, we'll just use the best approach from that test suite, i.e., blocking with $B = 16$ and multiple accumulators.

For row-major output, we'll re-use the best approach for column-major output.
The change in the output layout should not have a major impact on performance as we don't iterate over the output in the innermost loop (i.e., the dot product).

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
naive, row RHS, column output           : 1.14181 ± 0.0090064
naive, row RHS, row output              : 0.406434 ± 0.00921344
best column RHS, column output          : 0.275376 ± 0.00331017
best column RHS, row output             : 0.279818 ± 0.00173978
blocked (4), row RHS, column output     : 0.74231 ± 0.0064378
blocked (8), row RHS, column output     : 0.720306 ± 0.00677513
blocked (16), row RHS, column output    : 0.633932 ± 0.00456445
blocked (32), row RHS, column output    : 0.60666 ± 0.00467762
blocked (4), row RHS, row output        : 0.32194 ± 0.00536093
blocked (8), row RHS, row output        : 0.299246 ± 0.00398991
blocked (16), row RHS, row output       : 0.295146 ± 0.00393415
blocked (32), row RHS, row output       : 0.307388 ± 0.0033064

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row RHS, column output           : 0.673274 ± 0.000972752
naive, row RHS, row output              : 0.2755 ± 0.00063009
best column RHS, column output          : 0.254778 ± 0.000730795
best column RHS, row output             : 0.260148 ± 0.00058367
blocked (4), row RHS, column output     : 0.670024 ± 0.00115721
blocked (8), row RHS, column output     : 0.657196 ± 0.000729419
blocked (16), row RHS, column output    : 0.582156 ± 0.000789015
blocked (32), row RHS, column output    : 0.588345 ± 0.000836385
blocked (4), row RHS, row output        : 0.278105 ± 0.000843222
blocked (8), row RHS, row output        : 0.265977 ± 0.00076025
blocked (16), row RHS, row output       : 0.277504 ± 0.00107574
blocked (32), row RHS, row output       : 0.301405 ± 0.00062487

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row RHS, column output           : 1.63602 ± 0.00221493
naive, row RHS, row output              : 0.270307 ± 0.000474084
best column RHS, column output          : 0.267482 ± 0.00130722
best column RHS, row output             : 0.263713 ± 0.0011129
blocked (4), row RHS, column output     : 0.794182 ± 0.00958376
blocked (8), row RHS, column output     : 0.70401 ± 0.00124635
blocked (16), row RHS, column output    : 0.619634 ± 0.00139458
blocked (32), row RHS, column output    : 0.597538 ± 0.0023083
blocked (4), row RHS, row output        : 0.278909 ± 0.000712068
blocked (8), row RHS, row output        : 0.265325 ± 0.000751598
blocked (16), row RHS, row output       : 0.275436 ± 0.000657342
blocked (32), row RHS, row output       : 0.298484 ± 0.00187375

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row RHS, column output           : 7.10912 ± 0.0177359
naive, row RHS, row output              : 0.398405 ± 0.00124572
best column RHS, column output          : 0.308833 ± 0.000854042
best column RHS, row output             : 0.273566 ± 0.00267607
blocked (4), row RHS, column output     : 1.06327 ± 0.00893533
blocked (8), row RHS, column output     : 0.793328 ± 0.00204802
blocked (16), row RHS, column output    : 0.683009 ± 0.005691
blocked (32), row RHS, column output    : 0.628944 ± 0.00111994
blocked (4), row RHS, row output        : 0.319499 ± 0.00426812
blocked (8), row RHS, row output        : 0.28645 ± 0.00244728
blocked (16), row RHS, row output       : 0.281681 ± 0.000324267
blocked (32), row RHS, row output       : 0.319069 ± 0.00186328

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row RHS, column output           : 0.766125 ± 0.00812122
naive, row RHS, row output              : 0.392673 ± 0.00524348
best column RHS, column output          : 0.271098 ± 0.00387039
best column RHS, row output             : 0.269938 ± 0.00182902
blocked (4), row RHS, column output     : 0.700244 ± 0.00332105
blocked (8), row RHS, column output     : 0.672409 ± 0.00361187
blocked (16), row RHS, column output    : 0.603421 ± 0.00482097
blocked (32), row RHS, column output    : 0.598411 ± 0.00208094
blocked (4), row RHS, row output        : 0.321106 ± 0.00560392
blocked (8), row RHS, row output        : 0.286587 ± 0.00182183
blocked (16), row RHS, row output       : 0.29807 ± 0.0049441
blocked (32), row RHS, row output       : 0.317887 ± 0.00480961

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, row RHS, column output           : 4.19342 ± 0.0135607
naive, row RHS, row output              : 0.556459 ± 0.00959393
best column RHS, column output          : 0.271378 ± 0.000598733
best column RHS, row output             : 0.278117 ± 0.00311665
blocked (4), row RHS, column output     : 0.835782 ± 0.0103813
blocked (8), row RHS, column output     : 0.68633 ± 0.008705
blocked (16), row RHS, column output    : 0.606487 ± 0.0099304
blocked (32), row RHS, column output    : 0.605305 ± 0.00325106
blocked (4), row RHS, row output        : 0.346979 ± 0.00383017
blocked (8), row RHS, row output        : 0.303923 ± 0.00786391
blocked (16), row RHS, row output       : 0.290099 ± 0.00426397
blocked (32), row RHS, row output       : 0.312249 ± 0.00144373

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, row RHS, column output           : 1.17892 ± 0.0116815
naive, row RHS, row output              : 0.53437 ± 0.00230292
best column RHS, column output          : 0.265228 ± 0.0012643
best column RHS, row output             : 0.271109 ± 0.00105825
blocked (4), row RHS, column output     : 0.685557 ± 0.00422191
blocked (8), row RHS, column output     : 0.64215 ± 0.00158618
blocked (16), row RHS, column output    : 0.59143 ± 0.00270529
blocked (32), row RHS, column output    : 0.588837 ± 0.00144323
blocked (4), row RHS, row output        : 0.352563 ± 0.00329588
blocked (8), row RHS, row output        : 0.299155 ± 0.000971689
blocked (16), row RHS, row output       : 0.293965 ± 0.000840501
blocked (32), row RHS, row output       : 0.304828 ± 0.000649014
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive, row RHS, column output           : 0.995373 ± 0.0106139
naive, row RHS, row output              : 0.153407 ± 0.00368134
best column RHS, column output          : 0.174032 ± 0.00211291
best column RHS, row output             : 0.175405 ± 0.00157908
blocked (4), row RHS, column output     : 0.678058 ± 0.00613704
blocked (8), row RHS, column output     : 0.6782 ± 0.00656981
blocked (16), row RHS, column output    : 0.595189 ± 0.00275884
blocked (32), row RHS, column output    : 0.583604 ± 0.003078
blocked (4), row RHS, row output        : 0.144068 ± 0.00288571
blocked (8), row RHS, row output        : 0.140968 ± 0.00148047
blocked (16), row RHS, row output       : 0.148043 ± 0.00134987
blocked (32), row RHS, row output       : 0.17845 ± 0.000883242

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row RHS, column output           : 0.653834 ± 0.00223527
naive, row RHS, row output              : 0.140248 ± 0.00065084
best column RHS, column output          : 0.168765 ± 0.000633969
best column RHS, row output             : 0.17088 ± 0.000631422
blocked (4), row RHS, column output     : 0.662567 ± 0.000945804
blocked (8), row RHS, column output     : 0.650173 ± 0.000681504
blocked (16), row RHS, column output    : 0.570393 ± 0.000765585
blocked (32), row RHS, column output    : 0.579928 ± 0.000820694
blocked (4), row RHS, row output        : 0.138988 ± 0.000422896
blocked (8), row RHS, row output        : 0.137067 ± 0.000587394
blocked (16), row RHS, row output       : 0.151987 ± 0.000138358
blocked (32), row RHS, row output       : 0.19608 ± 0.000626169

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row RHS, column output           : 2.39676 ± 0.00545126
naive, row RHS, row output              : 0.136354 ± 0.000229279
best column RHS, column output          : 0.173862 ± 0.000119673
best column RHS, row output             : 0.177191 ± 0.000151278
blocked (4), row RHS, column output     : 0.797226 ± 0.000349497
blocked (8), row RHS, column output     : 0.679625 ± 0.000431804
blocked (16), row RHS, column output    : 0.601449 ± 0.000205244
blocked (32), row RHS, column output    : 0.582744 ± 0.000723343
blocked (4), row RHS, row output        : 0.133372 ± 0.000125279
blocked (8), row RHS, row output        : 0.136798 ± 0.000251205
blocked (16), row RHS, row output       : 0.142908 ± 0.000293207
blocked (32), row RHS, row output       : 0.17702 ± 0.000108953

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row RHS, column output           : 6.16424 ± 0.0211997
naive, row RHS, row output              : 0.164002 ± 0.00312256
best column RHS, column output          : 0.202905 ± 0.00416451
best column RHS, row output             : 0.186214 ± 0.00466774
blocked (4), row RHS, column output     : 0.966442 ± 0.0120897
blocked (8), row RHS, column output     : 0.762323 ± 0.00635152
blocked (16), row RHS, column output    : 0.635044 ± 0.00829168
blocked (32), row RHS, column output    : 0.62992 ± 0.0119501
blocked (4), row RHS, row output        : 0.146768 ± 0.00388939
blocked (8), row RHS, row output        : 0.144303 ± 0.00136362
blocked (16), row RHS, row output       : 0.149362 ± 0.00233546
blocked (32), row RHS, row output       : 0.180375 ± 0.00142351

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row RHS, column output           : 0.65519 ± 0.00110044
naive, row RHS, row output              : 0.166344 ± 0.000537955
best column RHS, column output          : 0.173773 ± 0.000609548
best column RHS, row output             : 0.176758 ± 0.000710614
blocked (4), row RHS, column output     : 0.681058 ± 0.0027566
blocked (8), row RHS, column output     : 0.645614 ± 0.000882622
blocked (16), row RHS, column output    : 0.568711 ± 0.000612447
blocked (32), row RHS, column output    : 0.575892 ± 0.000544711
blocked (4), row RHS, row output        : 0.154663 ± 0.000634633
blocked (8), row RHS, row output        : 0.148788 ± 0.000422302
blocked (16), row RHS, row output       : 0.161081 ± 0.000561996
blocked (32), row RHS, row output       : 0.200616 ± 0.000543943

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 1000 x 10000
naive, row RHS, column output           : 2.38295 ± 0.0256462
naive, row RHS, row output              : 0.281731 ± 0.00402007
best column RHS, column output          : 0.183541 ± 0.00188656
best column RHS, row output             : 0.185959 ± 0.00154508
blocked (4), row RHS, column output     : 0.725439 ± 0.00693281
blocked (8), row RHS, column output     : 0.650509 ± 0.00374173
blocked (16), row RHS, column output    : 0.594884 ± 0.00295908
blocked (32), row RHS, column output    : 0.583377 ± 0.00205308
blocked (4), row RHS, row output        : 0.174357 ± 0.00235888
blocked (8), row RHS, row output        : 0.157964 ± 0.00149684
blocked (16), row RHS, row output       : 0.153222 ± 0.00103073
blocked (32), row RHS, row output       : 0.182571 ± 0.000714109

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 10000 x 1000
naive, row RHS, column output           : 1.13679 ± 0.0114791
naive, row RHS, row output              : 0.288454 ± 0.00632527
best column RHS, column output          : 0.174514 ± 0.000937055
best column RHS, row output             : 0.180243 ± 0.000899654
blocked (4), row RHS, column output     : 0.646429 ± 0.00442716
blocked (8), row RHS, column output     : 0.623653 ± 0.00100016
blocked (16), row RHS, column output    : 0.584295 ± 0.00139962
blocked (32), row RHS, column output    : 0.566716 ± 0.0016149
blocked (4), row RHS, row output        : 0.185374 ± 0.00279585
blocked (8), row RHS, row output        : 0.163912 ± 0.00209672
blocked (16), row RHS, row output       : 0.155613 ± 0.00111048
blocked (32), row RHS, row output       : 0.183313 ± 0.00107358
```

## Conclusion

Blocking provides some assistance here, depending on the exact shape of the matrices involved.
A block size of 8 seems to perform the best, or nearly so, in a wide range of scenarios.
At the very least, it doesn't do any harm.

With blocking, multiplication with a row-major RHS is comparable to that of a column-major RHS.
This is pretty nice as the latter would usually be considered the "optimal" way to do this multiplication. 
In fact, it surpasses column-major RHS multiplication for `float` types, maybe because of better auto-vectorization opportunities.
