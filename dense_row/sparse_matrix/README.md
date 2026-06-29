# Dense row-major product with sparse matrix

## Strategies

All strategies must iterate across consecutive columns of the LHS matrix in the outermost loop.
We will be loading columns on demand via the **tatami** interface, so the full matrix will not be available for random access.

### Naive column-major RHS

For each LHS row, we iterate over the RHS columns and compute a dense-sparse dot product for each one.
This can be trivially stored in both column- and row-major output;
the output layout should not have a major impact on performance as we don't iterate over the output in the innermost loop (i.e., the dot product).

### Accumulators, column-major RHS

This is the same as the naive approach, except that the sparse vector dot product is computed with multiple accumulators.
The idea is to break dependency chains in the CPU's instruction pipeline by allowing multiple accumulations to occur in parallel.
It also provides some opportunities for auto-vectorization of the sum, though this relies on some good gather instructions.

### Naive row-major RHS

The general idea is to accumulate the partial dot products by processing each RHS row.

If the output is row-major: for each LHS row, we iterate across the RHS rows.
For each RHS row, we perform a sparse vector multiply-add of the corresponding element from the LHS row with the relevant output row.
This involves only operating on the structural non-zeros of the sparse RHS row, so it is not contiguous, but at least data locality is preserved.

If the output is column-major, the process is much the same except that the output is done with conceptual output rows.
The conceptual $i$-th output row consists of the $i$-th element from each output column, which involves many scattered memory accesses.
There's not much that can be done here as the output's layout does not align with that of the LHS or RHS matrices.

### Blocking 

For column-major RHS, we can implement a blocking scheme to improve cache re-use of each RHS column across multiple LHS rows.
Specifically, for each RHS column, we consider a block of $C$ structural non-zeros.
We load in a block of $B$ LHS rows and we compute the partial dot product of the $C$ non-zeros with each of those rows.
The idea is to only reload this block of $C$ non-zeros per $B$ LHS rows rather than for each LHS row.
We repeat the calculation with the next block of $C$ non-zeros until the entire RHS column is processed, then we move onto the next column.
Once all RHS columns are processed, we consider the next block of $B$ rows.
For a valid comparison with the other strategies, we use 4 accumulators to compute the dot product, given that this is (near-)optimal in the non-blocked strategies.

For row-major RHS, we can use a similar blocking scheme to improve cache re-use of each RHS row across multiple LHS rows.
Specifically, for each RHS row, we consider a block of $C$ structural non-zeros.
We load in a block of $B$ LHS rows and we compute the sparse vector multiply-add for the $C$ non-zeros with the corresponding scaling factor from each LHS row.
The idea is to only reload this block of $C$ non-zeros per $B$ LHS rows rather than for each LHS row.
We repeat this with the next block of $C$ non-zeros until the entire RHS row is processed, then we move onto the next row.
Once all RHS rows are processed, we consider the next block of $B$ rows.

In both cases, we try various values of $B$ and we set $C = 256$.
We could actually set $C$ to be much larger as we are only concerned about keeping the structural non-zeros in the cache;
we don't have to worry about any $B$-by-$C$ submatrices here.
However, we do want to catch at least some looping overhead so we don't use $C$ that is so large that all non-zeros will be cached in all circumstances.

We do not consider conventional blocking with multiple RHS rows/columns as all of the innermost loops involve iteration over the sparse indices.
For fixed-size blocks, the blocking overhead would be comparable to the actual calculations within each block when the data is sparse.
In fact, the overhead would be larger than the calculations themselves if the block size is smaller than the inverse of the density.
Additionally, we can't easily determine where to restart the innermost loop for each new block of RHS rows/columns.
Doing so would require either a binary search or extra tracking of the positions of the last non-zero element for each sparse row/column in the preceding block.

In theory, we could use variable-size "blocks" of multiple RHS rows/columns, which contain no more than a certain number of non-zero elements from each RHS row/column.
This would improve the efficiency of traversal along the sparse row/column by making the overhead (mostly) proportional to the number of non-zero elements processed.
However, different RHS rows/columns have different number of structural non-zeros, so towards the end of each row/column, we'd be wasting iterations on rows/columns that have no more non-zeros.
Additionally, for each sparse "block" of this nature, we would still need to access the union of all indices of the structural non-zeros in the dense LHS/output matrix.
The non-zero values can be arbitrarily distributed so the union is not guaranteed to fit into a typical L1 cache, which defeats the purpose of blocking.

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
naive, column RHS, column output                  : 0.0892774 ± 0.00311636
naive, column RHS, row output                     : 0.0876773 ± 0.000536357
naive, row RHS, column output                     : 0.162109 ± 0.000221191
naive, row RHS, row output                        : 0.0637885 ± 5.88212e-05
2 accumulators, column RHS, column output         : 0.0597727 ± 8.61024e-05
2 accumulators, column RHS, row output            : 0.0591234 ± 6.20328e-05
4 accumulators, column RHS, column output         : 0.0501428 ± 0.00201441
4 accumulators, column RHS, row output            : 0.047122 ± 8.9244e-05
8 accumulators, column RHS, column output         : 0.0522704 ± 9.57777e-05
8 accumulators, column RHS, row output            : 0.0512333 ± 6.02626e-05
blocked (4), column RHS, column output            : 0.0503465 ± 0.00144052
blocked (8), column RHS, column output            : 0.055035 ± 8.52601e-05
blocked (16), column RHS, column output           : 0.05745 ± 5.1339e-05
blocked (32), column RHS, column output           : 0.068815 ± 0.000211931
blocked (4), column RHS, row output               : 0.0492887 ± 0.00229152
blocked (8), column RHS, row output               : 0.0552832 ± 0.00114932
blocked (16), column RHS, row output              : 0.0574875 ± 0.000102992
blocked (32), column RHS, row output              : 0.0742268 ± 7.66527e-05
blocked (4), row RHS, row output                  : 0.0610615 ± 0.000142981
blocked (8), row RHS, row output                  : 0.065272 ± 0.00152066
blocked (16), row RHS, row output                 : 0.0707286 ± 0.000181263
blocked (32), row RHS, row output                 : 0.0910446 ± 8.38578e-05

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, column RHS, column output                  : 0.0896739 ± 0.000141229
naive, column RHS, row output                     : 0.0905204 ± 8.89256e-05
naive, row RHS, column output                     : 0.15935 ± 0.000174946
naive, row RHS, row output                        : 0.100997 ± 0.000142555
2 accumulators, column RHS, column output         : 0.0617334 ± 6.61855e-05
2 accumulators, column RHS, row output            : 0.0612758 ± 0.000164586
4 accumulators, column RHS, column output         : 0.0477319 ± 5.803e-05
4 accumulators, column RHS, row output            : 0.0474101 ± 4.74352e-05
8 accumulators, column RHS, column output         : 0.0533701 ± 6.52493e-05
8 accumulators, column RHS, row output            : 0.0530003 ± 6.01431e-05
blocked (4), column RHS, column output            : 0.0528784 ± 0.000161211
blocked (8), column RHS, column output            : 0.058653 ± 4.64868e-05
blocked (16), column RHS, column output           : 0.0616251 ± 0.000113136
blocked (32), column RHS, column output           : 0.0715242 ± 3.74633e-05
blocked (4), column RHS, row output               : 0.0527134 ± 6.41559e-05
blocked (8), column RHS, row output               : 0.0602044 ± 4.12874e-05
blocked (16), column RHS, row output              : 0.0621268 ± 9.20006e-05
blocked (32), column RHS, row output              : 0.0740952 ± 0.000201506
blocked (4), row RHS, row output                  : 0.0855018 ± 6.41533e-05
blocked (8), row RHS, row output                  : 0.0758042 ± 9.65839e-05
blocked (16), row RHS, row output                 : 0.0763759 ± 0.000147177
blocked (32), row RHS, row output                 : 0.0736443 ± 9.33352e-05

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, column RHS, column output                  : 0.0950454 ± 0.000141788
naive, column RHS, row output                     : 0.102617 ± 0.000179523
naive, row RHS, column output                     : 0.181536 ± 8.65722e-05
naive, row RHS, row output                        : 0.0649275 ± 6.66381e-05
2 accumulators, column RHS, column output         : 0.103378 ± 8.82317e-05
2 accumulators, column RHS, row output            : 0.0841351 ± 0.000111589
4 accumulators, column RHS, column output         : 0.111306 ± 0.000128773
4 accumulators, column RHS, row output            : 0.0936444 ± 0.000100604
8 accumulators, column RHS, column output         : 0.128851 ± 7.36412e-05
8 accumulators, column RHS, row output            : 0.115579 ± 0.000107116
blocked (4), column RHS, column output            : 0.12352 ± 0.000155212
blocked (8), column RHS, column output            : 0.107909 ± 0.000156859
blocked (16), column RHS, column output           : 0.0978609 ± 0.000108351
blocked (32), column RHS, column output           : 0.097173 ± 0.000260662
blocked (4), column RHS, row output               : 0.106155 ± 0.000155506
blocked (8), column RHS, row output               : 0.101461 ± 7.27072e-05
blocked (16), column RHS, row output              : 0.0954051 ± 9.94933e-05
blocked (32), column RHS, row output              : 0.0975488 ± 7.19246e-05
blocked (4), row RHS, row output                  : 0.0680621 ± 0.000107587
blocked (8), row RHS, row output                  : 0.071647 ± 6.56509e-05
blocked (16), row RHS, row output                 : 0.0738261 ± 0.000108291
blocked (32), row RHS, row output                 : 0.0787044 ± 4.0355e-05

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, column RHS, column output                  : 0.104862 ± 0.00039149
naive, column RHS, row output                     : 0.113929 ± 4.59929e-05
naive, row RHS, column output                     : 0.714385 ± 0.000200138
naive, row RHS, row output                        : 0.0830595 ± 8.01918e-05
2 accumulators, column RHS, column output         : 0.135867 ± 0.000120349
2 accumulators, column RHS, row output            : 0.127569 ± 0.000169469
4 accumulators, column RHS, column output         : 0.139199 ± 0.000228435
4 accumulators, column RHS, row output            : 0.129963 ± 0.000331864
8 accumulators, column RHS, column output         : 0.157908 ± 0.000155329
8 accumulators, column RHS, row output            : 0.148905 ± 0.000126681
blocked (4), column RHS, column output            : 0.133962 ± 0.000368637
blocked (8), column RHS, column output            : 0.126775 ± 0.000417733
blocked (16), column RHS, column output           : 0.117856 ± 0.000205073
blocked (32), column RHS, column output           : 0.11381 ± 0.000337485
blocked (4), column RHS, row output               : 0.110793 ± 9.13275e-05
blocked (8), column RHS, row output               : 0.103157 ± 0.000385362
blocked (16), column RHS, row output              : 0.0952371 ± 0.000112142
blocked (32), column RHS, row output              : 0.0975245 ± 0.000537609
blocked (4), row RHS, row output                  : 0.102378 ± 8.32979e-05
blocked (8), row RHS, row output                  : 0.127291 ± 0.000257373
blocked (16), row RHS, row output                 : 0.128616 ± 6.66014e-05
blocked (32), row RHS, row output                 : 0.13087 ± 0.000102859

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, column RHS, column output                  : 0.0998887 ± 0.000620346
naive, column RHS, row output                     : 0.0995326 ± 0.000289097
naive, row RHS, column output                     : 0.201937 ± 0.000962524
naive, row RHS, row output                        : 0.128207 ± 0.000260258
2 accumulators, column RHS, column output         : 0.066282 ± 0.000436208
2 accumulators, column RHS, row output            : 0.0665921 ± 0.000546502
4 accumulators, column RHS, column output         : 0.060641 ± 7.65202e-05
4 accumulators, column RHS, row output            : 0.0609755 ± 0.000327009
8 accumulators, column RHS, column output         : 0.0611615 ± 0.000423257
8 accumulators, column RHS, row output            : 0.0619696 ± 0.000803974
blocked (4), column RHS, column output            : 0.0791777 ± 0.000635575
blocked (8), column RHS, column output            : 0.0901228 ± 0.000296881
blocked (16), column RHS, column output           : 0.0914351 ± 0.000636663
blocked (32), column RHS, column output           : 0.0920371 ± 8.76251e-05
blocked (4), column RHS, row output               : 0.0793424 ± 0.000641507
blocked (8), column RHS, row output               : 0.0906352 ± 9.42776e-05
blocked (16), column RHS, row output              : 0.0915081 ± 0.000717165
blocked (32), column RHS, row output              : 0.0933739 ± 0.000672125
blocked (4), row RHS, row output                  : 0.106279 ± 0.000625118
blocked (8), row RHS, row output                  : 0.0921787 ± 0.000582896
blocked (16), row RHS, row output                 : 0.0900692 ± 0.000527936
blocked (32), row RHS, row output                 : 0.0820354 ± 0.000598814

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, column RHS, column output                  : 0.112362 ± 0.000223281
naive, column RHS, row output                     : 0.112396 ± 0.00159201
naive, row RHS, column output                     : 0.392945 ± 0.00302875
naive, row RHS, row output                        : 0.10274 ± 0.00186278
2 accumulators, column RHS, column output         : 0.0895425 ± 0.000220995
2 accumulators, column RHS, row output            : 0.086574 ± 0.00139177
4 accumulators, column RHS, column output         : 0.0840668 ± 0.00166227
4 accumulators, column RHS, row output            : 0.0783171 ± 0.00147379
8 accumulators, column RHS, column output         : 0.0868273 ± 0.000202269
8 accumulators, column RHS, row output            : 0.0826047 ± 0.00150183
blocked (4), column RHS, column output            : 0.066548 ± 0.000136511
blocked (8), column RHS, column output            : 0.0661931 ± 0.000967279
blocked (16), column RHS, column output           : 0.0604965 ± 4.55506e-05
blocked (32), column RHS, column output           : 0.0719991 ± 0.000842371
blocked (4), column RHS, row output               : 0.0556914 ± 6.7125e-05
blocked (8), column RHS, row output               : 0.0590753 ± 0.000816474
blocked (16), column RHS, row output              : 0.0581154 ± 8.57675e-05
blocked (32), column RHS, row output              : 0.0746892 ± 4.1354e-05
blocked (4), row RHS, row output                  : 0.109445 ± 0.000171211
blocked (8), row RHS, row output                  : 0.123154 ± 0.000222105
blocked (16), row RHS, row output                 : 0.124078 ± 0.00127223
blocked (32), row RHS, row output                 : 0.122868 ± 0.00124057

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, column RHS, column output                  : 0.11908 ± 0.00156035
naive, column RHS, row output                     : 0.117684 ± 0.000627956
naive, row RHS, column output                     : 0.226216 ± 0.00310865
naive, row RHS, row output                        : 0.121474 ± 0.00211006
2 accumulators, column RHS, column output         : 0.0882659 ± 0.000217523
2 accumulators, column RHS, row output            : 0.088848 ± 0.00160652
4 accumulators, column RHS, column output         : 0.0831308 ± 0.000710365
4 accumulators, column RHS, row output            : 0.0823708 ± 0.00126313
8 accumulators, column RHS, column output         : 0.0837547 ± 0.000168437
8 accumulators, column RHS, row output            : 0.0839312 ± 0.00128118
blocked (4), column RHS, column output            : 0.0977332 ± 5.28102e-05
blocked (8), column RHS, column output            : 0.114487 ± 0.000832751
blocked (16), column RHS, column output           : 0.114263 ± 0.000212684
blocked (32), column RHS, column output           : 0.113349 ± 0.00029408
blocked (4), column RHS, row output               : 0.099031 ± 0.00085099
blocked (8), column RHS, row output               : 0.114383 ± 0.000267943
blocked (16), column RHS, row output              : 0.115843 ± 0.0010561
blocked (32), column RHS, row output              : 0.11449 ± 0.000857512
blocked (4), row RHS, row output                  : 0.0805472 ± 0.00237428
blocked (8), row RHS, row output                  : 0.0737721 ± 0.000235451
blocked (16), row RHS, row output                 : 0.0809959 ± 0.000739008
blocked (32), row RHS, row output                 : 0.0968116 ± 0.000818091
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive, column RHS, column output                  : 0.084794 ± 1.70524e-05
naive, column RHS, row output                     : 0.0860574 ± 0.000106904
naive, row RHS, column output                     : 0.156139 ± 0.000112304
naive, row RHS, row output                        : 0.0615171 ± 7.99304e-05
2 accumulators, column RHS, column output         : 0.0579636 ± 3.54169e-05
2 accumulators, column RHS, row output            : 0.057601 ± 1.92757e-05
4 accumulators, column RHS, column output         : 0.0419705 ± 3.56717e-05
4 accumulators, column RHS, row output            : 0.0417615 ± 5.52174e-05
8 accumulators, column RHS, column output         : 0.0442458 ± 6.00516e-05
8 accumulators, column RHS, row output            : 0.0438837 ± 1.75433e-05
blocked (4), column RHS, column output            : 0.0408319 ± 2.79126e-05
blocked (8), column RHS, column output            : 0.0414759 ± 3.57206e-05
blocked (16), column RHS, column output           : 0.0456595 ± 4.21842e-05
blocked (32), column RHS, column output           : 0.0471338 ± 2.96883e-05
blocked (4), column RHS, row output               : 0.0392126 ± 3.37548e-05
blocked (8), column RHS, row output               : 0.0416918 ± 4.99251e-05
blocked (16), column RHS, row output              : 0.0466943 ± 3.45911e-05
blocked (32), column RHS, row output              : 0.0490402 ± 3.34616e-05
blocked (4), row RHS, row output                  : 0.0798452 ± 0.000134379
blocked (8), row RHS, row output                  : 0.0787249 ± 9.72572e-05
blocked (16), row RHS, row output                 : 0.0779139 ± 7.1665e-05
blocked (32), row RHS, row output                 : 0.0775751 ± 7.46835e-05

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, column RHS, column output                  : 0.0846439 ± 9.06159e-05
naive, column RHS, row output                     : 0.0856898 ± 0.000152455
naive, row RHS, column output                     : 0.148788 ± 0.000350496
naive, row RHS, row output                        : 0.093241 ± 7.8556e-05
2 accumulators, column RHS, column output         : 0.057202 ± 7.76663e-05
2 accumulators, column RHS, row output            : 0.0568503 ± 6.9655e-05
4 accumulators, column RHS, column output         : 0.0390355 ± 9.36556e-05
4 accumulators, column RHS, row output            : 0.0388564 ± 3.87268e-05
8 accumulators, column RHS, column output         : 0.042899 ± 5.77994e-05
8 accumulators, column RHS, row output            : 0.0428841 ± 4.24061e-05
blocked (4), column RHS, column output            : 0.040598 ± 7.56921e-05
blocked (8), column RHS, column output            : 0.042261 ± 8.46275e-05
blocked (16), column RHS, column output           : 0.0472035 ± 6.50256e-05
blocked (32), column RHS, column output           : 0.0485726 ± 5.12884e-05
blocked (4), column RHS, row output               : 0.040102 ± 6.90963e-05
blocked (8), column RHS, row output               : 0.0426047 ± 0.000360339
blocked (16), column RHS, row output              : 0.0482694 ± 9.34152e-05
blocked (32), column RHS, row output              : 0.0495094 ± 8.14422e-05
blocked (4), row RHS, row output                  : 0.0968677 ± 0.000165809
blocked (8), row RHS, row output                  : 0.08676 ± 0.000137591
blocked (16), row RHS, row output                 : 0.0848575 ± 0.000163335
blocked (32), row RHS, row output                 : 0.0854549 ± 9.11942e-05

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, column RHS, column output                  : 0.0913859 ± 0.000114128
naive, column RHS, row output                     : 0.101728 ± 5.33351e-05
naive, row RHS, column output                     : 0.18883 ± 0.0001558
naive, row RHS, row output                        : 0.0568931 ± 0.000109212
2 accumulators, column RHS, column output         : 0.108405 ± 0.000597306
2 accumulators, column RHS, row output            : 0.0951006 ± 0.000144995
4 accumulators, column RHS, column output         : 0.0970135 ± 0.000105064
4 accumulators, column RHS, row output            : 0.081333 ± 0.000132641
8 accumulators, column RHS, column output         : 0.117602 ± 0.000233889
8 accumulators, column RHS, row output            : 0.107551 ± 9.46512e-05
blocked (4), column RHS, column output            : 0.105824 ± 0.00020663
blocked (8), column RHS, column output            : 0.0914136 ± 5.92165e-05
blocked (16), column RHS, column output           : 0.0815836 ± 0.000149629
blocked (32), column RHS, column output           : 0.0797173 ± 0.000117896
blocked (4), column RHS, row output               : 0.0951281 ± 0.00013393
blocked (8), column RHS, row output               : 0.0909134 ± 0.000110245
blocked (16), column RHS, row output              : 0.0845074 ± 8.30242e-05
blocked (32), column RHS, row output              : 0.0833955 ± 0.000122094
blocked (4), row RHS, row output                  : 0.0818575 ± 7.94732e-05
blocked (8), row RHS, row output                  : 0.0808141 ± 8.44689e-05
blocked (16), row RHS, row output                 : 0.0802487 ± 3.63879e-05
blocked (32), row RHS, row output                 : 0.0797902 ± 0.000143855

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, column RHS, column output                  : 0.100413 ± 4.02799e-05
naive, column RHS, row output                     : 0.11251 ± 3.74806e-05
naive, row RHS, column output                     : 0.615147 ± 0.000248069
naive, row RHS, row output                        : 0.0613567 ± 0.000312934
2 accumulators, column RHS, column output         : 0.137683 ± 0.000390171
2 accumulators, column RHS, row output            : 0.132346 ± 0.000357842
4 accumulators, column RHS, column output         : 0.132476 ± 0.000472351
4 accumulators, column RHS, row output            : 0.129515 ± 3.1431e-05
8 accumulators, column RHS, column output         : 0.145505 ± 6.7558e-05
8 accumulators, column RHS, row output            : 0.140108 ± 0.000351626
blocked (4), column RHS, column output            : 0.110259 ± 5.05944e-05
blocked (8), column RHS, column output            : 0.0975857 ± 0.000409976
blocked (16), column RHS, column output           : 0.0898299 ± 6.45457e-05
blocked (32), column RHS, column output           : 0.0897473 ± 6.72026e-05
blocked (4), column RHS, row output               : 0.0992287 ± 3.17899e-05
blocked (8), column RHS, row output               : 0.0906332 ± 0.000377806
blocked (16), column RHS, row output              : 0.0827582 ± 0.000386502
blocked (32), column RHS, row output              : 0.0828078 ± 0.000396234
blocked (4), row RHS, row output                  : 0.0782456 ± 4.91378e-05
blocked (8), row RHS, row output                  : 0.0830548 ± 3.48512e-05
blocked (16), row RHS, row output                 : 0.0891941 ± 5.12143e-05
blocked (32), row RHS, row output                 : 0.0906412 ± 6.46352e-05

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, column RHS, column output                  : 0.0958055 ± 8.25959e-05
naive, column RHS, row output                     : 0.0957094 ± 4.10407e-05
naive, row RHS, column output                     : 0.199924 ± 0.00063591
naive, row RHS, row output                        : 0.126355 ± 8.34428e-05
2 accumulators, column RHS, column output         : 0.0558933 ± 2.49886e-05
2 accumulators, column RHS, row output            : 0.0559309 ± 3.92536e-05
4 accumulators, column RHS, column output         : 0.0470359 ± 3.3328e-05
4 accumulators, column RHS, row output            : 0.0471921 ± 0.000170917
8 accumulators, column RHS, column output         : 0.0476225 ± 0.000153346
8 accumulators, column RHS, row output            : 0.0474701 ± 3.75329e-05
blocked (4), column RHS, column output            : 0.050799 ± 2.75573e-05
blocked (8), column RHS, column output            : 0.0597274 ± 5.85018e-05
blocked (16), column RHS, column output           : 0.0667309 ± 8.16709e-05
blocked (32), column RHS, column output           : 0.068235 ± 0.000130013
blocked (4), column RHS, row output               : 0.0507034 ± 2.46412e-05
blocked (8), column RHS, row output               : 0.0599317 ± 4.42333e-05
blocked (16), column RHS, row output              : 0.0674253 ± 8.53503e-05
blocked (32), column RHS, row output              : 0.0686396 ± 6.99214e-05
blocked (4), row RHS, row output                  : 0.112901 ± 0.000206972
blocked (8), row RHS, row output                  : 0.10058 ± 0.000194115
blocked (16), row RHS, row output                 : 0.0964759 ± 0.000129282
blocked (32), row RHS, row output                 : 0.0942743 ± 0.00017451

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, column RHS, column output                  : 0.10239 ± 9.29342e-05
naive, column RHS, row output                     : 0.102929 ± 0.000800297
naive, row RHS, column output                     : 0.190198 ± 0.003002
naive, row RHS, row output                        : 0.0992077 ± 0.000617553
2 accumulators, column RHS, column output         : 0.0637313 ± 6.49734e-05
2 accumulators, column RHS, row output            : 0.0634595 ± 0.000148899
4 accumulators, column RHS, column output         : 0.0563183 ± 0.00116798
4 accumulators, column RHS, row output            : 0.0544894 ± 0.000207483
8 accumulators, column RHS, column output         : 0.0559217 ± 0.000381063
8 accumulators, column RHS, row output            : 0.0559359 ± 0.00124064
blocked (4), column RHS, column output            : 0.0530792 ± 0.000179156
blocked (8), column RHS, column output            : 0.0572189 ± 0.00013893
blocked (16), column RHS, column output           : 0.0608619 ± 0.000175529
blocked (32), column RHS, column output           : 0.0608395 ± 0.000109755
blocked (4), column RHS, row output               : 0.0540602 ± 0.00117397
blocked (8), column RHS, row output               : 0.0575566 ± 8.41688e-05
blocked (16), column RHS, row output              : 0.0615489 ± 0.000298955
blocked (32), column RHS, row output              : 0.0615831 ± 0.000243932
blocked (4), row RHS, row output                  : 0.0929346 ± 0.00140628
blocked (8), row RHS, row output                  : 0.0856732 ± 0.000306817
blocked (16), row RHS, row output                 : 0.0826747 ± 8.3933e-05
blocked (32), row RHS, row output                 : 0.0817792 ± 0.000483472

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, column RHS, column output                  : 0.10167 ± 0.00245262
naive, column RHS, row output                     : 0.101627 ± 0.00256919
naive, row RHS, column output                     : 0.312681 ± 0.00206008
naive, row RHS, row output                        : 0.068918 ± 0.000197113
2 accumulators, column RHS, column output         : 0.0735944 ± 0.000176469
2 accumulators, column RHS, row output            : 0.068505 ± 0.000181076
4 accumulators, column RHS, column output         : 0.0621546 ± 0.00209164
4 accumulators, column RHS, row output            : 0.0552213 ± 0.000215034
8 accumulators, column RHS, column output         : 0.0627242 ± 0.00018245
8 accumulators, column RHS, row output            : 0.0574877 ± 0.000281717
blocked (4), column RHS, column output            : 0.0497174 ± 0.000420343
blocked (8), column RHS, column output            : 0.0450094 ± 7.15588e-05
blocked (16), column RHS, column output           : 0.0472924 ± 0.000105056
blocked (32), column RHS, column output           : 0.0474575 ± 0.000168646
blocked (4), column RHS, row output               : 0.0436621 ± 0.000184923
blocked (8), column RHS, row output               : 0.043019 ± 0.000122343
blocked (16), column RHS, row output              : 0.0471367 ± 0.000114752
blocked (32), column RHS, row output              : 0.0484018 ± 0.000181158
blocked (4), row RHS, row output                  : 0.0798827 ± 0.000232124
blocked (8), row RHS, row output                  : 0.0826787 ± 0.000231733
blocked (16), row RHS, row output                 : 0.0868478 ± 0.000109948
blocked (32), row RHS, row output                 : 0.0870259 ± 0.000132196
```

## Conclusion

Multiple accumulators often improve performance for multiplication with a column-major RHS matrix.
This is most apparent when the shared dimension extent is high, though it's also not too bad when the shared dimension is small.
It seems that 4 accumulators is typically the sweet spot, similar to our conclusions in the [`dense`](../single_vector) case.

Performance is pretty good for a row-major RHS matrix with row-major output.
I find this interesting because a sparse vector multiply-add is not easily vectorizable.
The values of the indices are not known to the compiler, so it can't just execute the instructions in parallel as it can't be sure that the indices don't overlap.

The behavior of blocking is interesting as it is sometimes helpful and sometimes detrimental:

- It's a pessimisation when the extent of the relevant dimension (i.e, RHS rows for row-major, or RHS columns for column-major) is large.
  This is not what we'd expect as blocking should mitigate cache misses when the RHS row/column has many structural non-zeros to be re-used across multiple LHS rows.
  The obvious culprit is looping overhead, though $C$ should be more than large enough to offset that.
  In fact, performance gets even worse with greater $B$, which is definitely unexpected as this should have improved the rate of re-use.
  I suspect the real cause is that cache lines from earlier LHS/output rows in the current block are being evicted by the time we get to the later rows.
  If these cache lines are "hanging off the edge" of the previous block, they could have provided fast access for the next block of $C$ non-zeros;
  this is no longer possible when they are evicted.
- It's an improvement when the extent of the relevant dimension is small.
  There is no looping overhead as $C$ is large enough to hold all structural non-zeros in memory for each RHS row/column.
  Thus, there is only optimal re-use of the cached RHS row/column across the current block of LHS rows.
  Indeed, if the dimension extent is small enough, even the current block of dense LHS/output rows might remain in cache for re-use by the next RHS row/column.

So it seems like the absence of blocking is actually more scalable as the dimension extent increases.
