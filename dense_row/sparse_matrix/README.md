# Dense row-major LHS, sparse matrix RHS

## Strategies

All strategies must iterate across consecutive columns of the LHS matrix in the outermost loop.
We will be loading columns on demand via the **tatami** interface, so the full matrix will not be available for random access.

### Naive column-major RHS

For each LHS row, we iterate over the RHS columns and compute a dense-sparse dot product for each one.
This can be trivially stored in both column- and row-major output;
the output layout should not have a major impact on performance as we don't iterate over the output in the innermost loop (i.e., the dot product).

### Naive column-major RHS with accumulators

This is the same as the naive approach, except that the sparse vector dot product is computed with multiple accumulators.
Check out the explanation in [`general/README.md`](../../general/README.md).

### Naive row-major RHS

If the output is row-major: for each LHS row $i$, we iterate across the RHS rows.
For each RHS row $h$, we perform a sparse vector multiply-add to the $i$-th output row, using the $h$-th element of the $i$-th LHS row as a scaling factor.
This involves only operating on the structural non-zeros of RHS row $h$, so it is not contiguous, but at least data locality is preserved for better caching.

If the output is column-major, the process is much the same except that we use a temporary buffer for the output row. 
Once all RHS rows have been processed, we transpose the temporary buffer's contents into the output columns.

### Blocking column-major RHS

We'll use the framework described in the "Blocking to cache many dense vectors" section in [`general/README.md`](../../general/README.md). 

We load in a block of $B$ LHS rows.
For each LHS block, we iterate over all RHS columns.
For each RHS column $h$, we compute its dot product with each LHS row $i$ in the block, storing the result in entry $(i, h)$ of the output matrix.
Once all RHS columns are processed, we repeat this process with the next block of $B$ LHS rows.

The idea is to keep the RHS column in cache, to re-use across LHS rows in the block; and to keep the block of LHS rows in cache, to re-use with the next RHS column. 
For a valid comparison with the other strategies, we use 4 accumulators to compute the dot product, given that this is (near-)optimal in the non-blocked strategies.

### Blocking row-major RHS, row-major output

We'll use the framework described in the "Blocking to cache many dense vectors" section in [`general/README.md`](../../general/README.md). 

We load in a block of $B$ LHS rows.
For each LHS block, we iterate over all RHS rows.
For each RHS row $h$, we perform a sparse vector multiply-add to the $i$-th output row, using the $h$-th element of the $i$-th LHS row as the scaling factor.
Once all RHS rows are processed, we repeat this process with the next block of $B$ LHS rows.

The idea is to only reload this block of $C$ non-zeros per $B$ LHS rows rather than for each LHS row.
Again, we try multiple values of $B$ with the constraint that $BC = 256$.

### Blocking row-major RHS, column-major output

We first load in a block of $B$ LHS rows.
For each LHS block, we iterate over all RHS rows.
For each RHS row $h$, we extract the corresponding column of the LHS row block into a contiguous buffer.
We then iterate over the structural non-zeros of the RHS row $h$.
For each non-zero at RHS column $j$ with value $x$, we perform a vector multiply-add of the column buffer to the corresponding block of the $j$-th output column,
using $x$ as the scaling factor.
Once all RHS rows are processed, we move onto the next block of $B$ LHS rows until the entire LHS matrix is traversed.

The idea is to encourage contiguous writes to the same output column while we have it in cache.
We try multiple values of $B$; in this case, we aren't really re-using anything, so there's no need to consider the size of the cache.

### More comments on blocking

The "Blocking to cache the dense vector" section in [`general/README.md`](../../general/README.md) suggests arranging the loops so that the dense vector is kept in cache. 
However, there is no need to explicitly do so here.
The naive approaches will automatically process multiple sparse vectors for a single dense vector,
be it the LHS row (for column-major RHS) or the output row (for row-major RHS).

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
naive, column RHS, column output                  : 0.0873692 ± 7.51068e-05
naive, column RHS, row output                     : 0.0857962 ± 0.000110618
naive, row RHS, column output                     : 0.0695115 ± 0.000400819
naive, row RHS, row output                        : 0.0637944 ± 4.76855e-05
2 accumulators, column RHS, column output         : 0.0584543 ± 4.91254e-05
2 accumulators, column RHS, row output            : 0.0583851 ± 6.23521e-05
4 accumulators, column RHS, column output         : 0.0503198 ± 6.86463e-05
4 accumulators, column RHS, row output            : 0.0485293 ± 8.40211e-05
8 accumulators, column RHS, column output         : 0.0522479 ± 3.20809e-05
8 accumulators, column RHS, row output            : 0.0515443 ± 0.000101093
blocked (B = 2), column RHS, column output        : 0.0485031 ± 5.57897e-05
blocked (B = 4), column RHS, column output        : 0.0464858 ± 8.5378e-05
blocked (B = 8), column RHS, column output        : 0.0524755 ± 9.25897e-05
blocked (B = 16), column RHS, column output       : 0.0550883 ± 2.35778e-05
blocked (B = 2), column RHS, row output           : 0.0471888 ± 4.02838e-05
blocked (B = 4), column RHS, row output           : 0.0462438 ± 7.05591e-05
blocked (B = 8), column RHS, row output           : 0.0531626 ± 7.89116e-05
blocked (B = 16), column RHS, row output          : 0.0564179 ± 8.37298e-05
blocked (B = 4), row RHS, column output           : 0.0806897 ± 0.000125204
blocked (B = 8), row RHS, column output           : 0.0609894 ± 3.25558e-05
blocked (B = 16), row RHS, column output          : 0.0516171 ± 7.3777e-05
blocked (B = 32), row RHS, column output          : 0.0466046 ± 6.07209e-05
blocked (B = 2), row RHS, row output              : 0.0652574 ± 0.000122541
blocked (B = 4), row RHS, row output              : 0.0612126 ± 8.92356e-05
blocked (B = 8), row RHS, row output              : 0.0639251 ± 7.37553e-05
blocked (B = 16), row RHS, row output             : 0.0725272 ± 0.000108604

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, column RHS, column output                  : 0.0913986 ± 4.76272e-05
naive, column RHS, row output                     : 0.0901282 ± 0.000109522
naive, row RHS, column output                     : 0.117405 ± 0.000114272
naive, row RHS, row output                        : 0.102074 ± 9.42283e-05
2 accumulators, column RHS, column output         : 0.0613567 ± 9.84973e-05
2 accumulators, column RHS, row output            : 0.06141 ± 9.80204e-05
4 accumulators, column RHS, column output         : 0.0507058 ± 6.5318e-05
4 accumulators, column RHS, row output            : 0.0495649 ± 5.39974e-05
8 accumulators, column RHS, column output         : 0.0542428 ± 2.21073e-05
8 accumulators, column RHS, row output            : 0.0540727 ± 7.05792e-05
blocked (B = 2), column RHS, column output        : 0.050592 ± 8.25806e-05
blocked (B = 4), column RHS, column output        : 0.050867 ± 3.7359e-05
blocked (B = 8), column RHS, column output        : 0.0570509 ± 7.95219e-05
blocked (B = 16), column RHS, column output       : 0.0614485 ± 0.000104855
blocked (B = 2), column RHS, row output           : 0.0497083 ± 5.89506e-05
blocked (B = 4), column RHS, row output           : 0.0509525 ± 8.39959e-05
blocked (B = 8), column RHS, row output           : 0.0588304 ± 5.0321e-05
blocked (B = 16), column RHS, row output          : 0.0628735 ± 9.46784e-05
blocked (B = 4), row RHS, column output           : 0.1051 ± 5.34998e-05
blocked (B = 8), row RHS, column output           : 0.0761438 ± 0.00010321
blocked (B = 16), row RHS, column output          : 0.0611524 ± 4.62595e-05
blocked (B = 32), row RHS, column output          : 0.0512131 ± 4.05509e-05
blocked (B = 2), row RHS, row output              : 0.091348 ± 6.18149e-05
blocked (B = 4), row RHS, row output              : 0.0824551 ± 0.000104913
blocked (B = 8), row RHS, row output              : 0.0772661 ± 4.03827e-05
blocked (B = 16), row RHS, row output             : 0.08007 ± 6.45426e-05

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, column RHS, column output                  : 0.105442 ± 0.000168996
naive, column RHS, row output                     : 0.0852126 ± 9.63015e-05
naive, row RHS, column output                     : 0.0972942 ± 0.000202491
naive, row RHS, row output                        : 0.0643745 ± 0.00014086
2 accumulators, column RHS, column output         : 0.0866667 ± 0.000105405
2 accumulators, column RHS, row output            : 0.071433 ± 0.000185318
4 accumulators, column RHS, column output         : 0.103931 ± 6.03469e-05
4 accumulators, column RHS, row output            : 0.0823945 ± 9.95051e-05
8 accumulators, column RHS, column output         : 0.115485 ± 0.000153338
8 accumulators, column RHS, row output            : 0.10148 ± 7.58166e-05
blocked (B = 2), column RHS, column output        : 0.0966616 ± 0.000105016
blocked (B = 4), column RHS, column output        : 0.0875272 ± 3.58035e-05
blocked (B = 8), column RHS, column output        : 0.0818119 ± 2.82844e-05
blocked (B = 16), column RHS, column output       : 0.0753522 ± 4.65274e-05
blocked (B = 2), column RHS, row output           : 0.0897046 ± 0.000114229
blocked (B = 4), column RHS, row output           : 0.0846553 ± 7.64904e-05
blocked (B = 8), column RHS, row output           : 0.0815095 ± 8.88054e-05
blocked (B = 16), column RHS, row output          : 0.0765556 ± 7.50312e-05
blocked (B = 4), row RHS, column output           : 0.0903791 ± 2.51968e-05
blocked (B = 8), row RHS, column output           : 0.0692032 ± 6.65255e-05
blocked (B = 16), row RHS, column output          : 0.0589841 ± 4.45552e-05
blocked (B = 32), row RHS, column output          : 0.0530657 ± 4.56018e-05
blocked (B = 2), row RHS, row output              : 0.0673976 ± 5.78481e-05
blocked (B = 4), row RHS, row output              : 0.0682064 ± 0.000114531
blocked (B = 8), row RHS, row output              : 0.0727871 ± 4.68891e-05
blocked (B = 16), row RHS, row output             : 0.0762442 ± 3.77146e-05

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, column RHS, column output                  : 0.118215 ± 0.00152666
naive, column RHS, row output                     : 0.0984299 ± 0.000497314
naive, row RHS, column output                     : 0.169385 ± 0.000551877
naive, row RHS, row output                        : 0.0851573 ± 0.00046942
2 accumulators, column RHS, column output         : 0.129382 ± 0.00125741
2 accumulators, column RHS, row output            : 0.127823 ± 0.000740719
4 accumulators, column RHS, column output         : 0.138871 ± 0.00097638
4 accumulators, column RHS, row output            : 0.135721 ± 0.000710946
8 accumulators, column RHS, column output         : 0.147449 ± 0.000857355
8 accumulators, column RHS, row output            : 0.143933 ± 0.000522945
blocked (B = 2), column RHS, column output        : 0.117773 ± 0.000416738
blocked (B = 4), column RHS, column output        : 0.0945322 ± 0.000318193
blocked (B = 8), column RHS, column output        : 0.0842965 ± 0.000268887
blocked (B = 16), column RHS, column output       : 0.077107 ± 0.000708967
blocked (B = 2), column RHS, row output           : 0.114876 ± 0.000664299
blocked (B = 4), column RHS, row output           : 0.0941116 ± 0.000879119
blocked (B = 8), column RHS, row output           : 0.0853344 ± 0.000476428
blocked (B = 16), column RHS, row output          : 0.0771585 ± 0.000211143
blocked (B = 4), row RHS, column output           : 0.193476 ± 0.00067028
blocked (B = 8), row RHS, column output           : 0.121785 ± 0.000487657
blocked (B = 16), row RHS, column output          : 0.0955974 ± 0.000613659
blocked (B = 32), row RHS, column output          : 0.0998127 ± 0.000473266
blocked (B = 2), row RHS, row output              : 0.0876428 ± 0.000395995
blocked (B = 4), row RHS, row output              : 0.111971 ± 0.000312478
blocked (B = 8), row RHS, row output              : 0.130961 ± 0.000400414
blocked (B = 16), row RHS, row output             : 0.132602 ± 0.000381407

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, column RHS, column output                  : 0.100079 ± 0.000311638
naive, column RHS, row output                     : 0.0994969 ± 0.000167176
naive, row RHS, column output                     : 0.136747 ± 0.000658118
naive, row RHS, row output                        : 0.130946 ± 0.000712052
2 accumulators, column RHS, column output         : 0.0665883 ± 0.000291635
2 accumulators, column RHS, row output            : 0.0671904 ± 0.000760865
4 accumulators, column RHS, column output         : 0.061908 ± 0.000294463
4 accumulators, column RHS, row output            : 0.0614538 ± 0.000268745
8 accumulators, column RHS, column output         : 0.0625759 ± 0.000344053
8 accumulators, column RHS, row output            : 0.06257 ± 0.000410822
blocked (B = 2), column RHS, column output        : 0.0698557 ± 0.000814765
blocked (B = 4), column RHS, column output        : 0.0896562 ± 0.0026045
blocked (B = 8), column RHS, column output        : 0.102664 ± 0.00405707
blocked (B = 16), column RHS, column output       : 0.104499 ± 0.00427162
blocked (B = 2), column RHS, row output           : 0.0697227 ± 0.000788852
blocked (B = 4), column RHS, row output           : 0.0881327 ± 0.00258728
blocked (B = 8), column RHS, row output           : 0.105183 ± 0.00398633
blocked (B = 16), column RHS, row output          : 0.104855 ± 0.00419334
blocked (B = 4), row RHS, column output           : 0.120537 ± 0.000312845
blocked (B = 8), row RHS, column output           : 0.0857137 ± 0.000214464
blocked (B = 16), row RHS, column output          : 0.0659317 ± 9.97571e-05
blocked (B = 32), row RHS, column output          : 0.0528977 ± 0.000109647
blocked (B = 2), row RHS, row output              : 0.115934 ± 0.000473222
blocked (B = 4), row RHS, row output              : 0.103166 ± 0.000778081
blocked (B = 8), row RHS, row output              : 0.0930493 ± 0.000329111
blocked (B = 16), row RHS, row output             : 0.0927408 ± 0.000322766

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, column RHS, column output                  : 0.117758 ± 0.000240973
naive, column RHS, row output                     : 0.11713 ± 0.000166933
naive, row RHS, column output                     : 0.124598 ± 0.000917639
naive, row RHS, row output                        : 0.119958 ± 0.000349361
2 accumulators, column RHS, column output         : 0.0866685 ± 0.000427295
2 accumulators, column RHS, row output            : 0.0860315 ± 0.000395347
4 accumulators, column RHS, column output         : 0.0822613 ± 0.000329794
4 accumulators, column RHS, row output            : 0.0813301 ± 0.000443122
8 accumulators, column RHS, column output         : 0.0831556 ± 0.000452233
8 accumulators, column RHS, row output            : 0.082409 ± 0.000305268
blocked (B = 2), column RHS, column output        : 0.0779798 ± 0.000493732
blocked (B = 4), column RHS, column output        : 0.0861449 ± 0.000180471
blocked (B = 8), column RHS, column output        : 0.0915139 ± 0.000112845
blocked (B = 16), column RHS, column output       : 0.0895066 ± 0.000146932
blocked (B = 2), column RHS, row output           : 0.0772367 ± 0.000124716
blocked (B = 4), column RHS, row output           : 0.0857558 ± 0.000141226
blocked (B = 8), column RHS, row output           : 0.091562 ± 0.000197188
blocked (B = 16), column RHS, row output          : 0.0896493 ± 0.000118936
blocked (B = 4), row RHS, column output           : 0.0952167 ± 0.000472923
blocked (B = 8), row RHS, column output           : 0.0688146 ± 0.000138027
blocked (B = 16), row RHS, column output          : 0.0577225 ± 0.000160209
blocked (B = 32), row RHS, column output          : 0.0505217 ± 0.000119569
blocked (B = 2), row RHS, row output              : 0.0965568 ± 0.000322744
blocked (B = 4), row RHS, row output              : 0.0786976 ± 0.000129639
blocked (B = 8), row RHS, row output              : 0.0747945 ± 0.00020499
blocked (B = 16), row RHS, row output             : 0.0767614 ± 0.000135452

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, column RHS, column output                  : 0.114902 ± 0.000389761
naive, column RHS, row output                     : 0.109462 ± 0.000231342
naive, row RHS, column output                     : 0.109822 ± 0.0002161
naive, row RHS, row output                        : 0.101128 ± 0.0001767
2 accumulators, column RHS, column output         : 0.0901926 ± 0.000263401
2 accumulators, column RHS, row output            : 0.0852878 ± 0.000169802
4 accumulators, column RHS, column output         : 0.0849361 ± 0.000243528
4 accumulators, column RHS, row output            : 0.0788872 ± 0.000192894
8 accumulators, column RHS, column output         : 0.0880149 ± 0.000191576
8 accumulators, column RHS, row output            : 0.082823 ± 0.000188203
blocked (B = 2), column RHS, column output        : 0.0661789 ± 0.000171248
blocked (B = 4), column RHS, column output        : 0.0564117 ± 0.00016457
blocked (B = 8), column RHS, column output        : 0.0573969 ± 0.000125819
blocked (B = 16), column RHS, column output       : 0.0575133 ± 4.14005e-05
blocked (B = 2), column RHS, row output           : 0.0635396 ± 0.000171372
blocked (B = 4), column RHS, row output           : 0.0554707 ± 0.000265454
blocked (B = 8), column RHS, row output           : 0.057617 ± 0.000346051
blocked (B = 16), column RHS, row output          : 0.058276 ± 5.4617e-05
blocked (B = 4), row RHS, column output           : 0.158633 ± 0.000110166
blocked (B = 8), row RHS, column output           : 0.108854 ± 0.000133456
blocked (B = 16), row RHS, column output          : 0.0844228 ± 0.000226959
blocked (B = 32), row RHS, column output          : 0.0764465 ± 0.000176745
blocked (B = 2), row RHS, row output              : 0.100578 ± 0.000342848
blocked (B = 4), row RHS, row output              : 0.119129 ± 0.000137414
blocked (B = 8), row RHS, row output              : 0.133154 ± 0.00016581
blocked (B = 16), row RHS, row output             : 0.132287 ± 0.000107928
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive, column RHS, column output                  : 0.0850025 ± 0.00011502
naive, column RHS, row output                     : 0.0860365 ± 0.000138674
naive, row RHS, column output                     : 0.0660852 ± 0.00012018
naive, row RHS, row output                        : 0.0617091 ± 7.20355e-05
2 accumulators, column RHS, column output         : 0.0560858 ± 8.49786e-05
2 accumulators, column RHS, row output            : 0.0545783 ± 7.82868e-05
4 accumulators, column RHS, column output         : 0.0407529 ± 8.19277e-05
4 accumulators, column RHS, row output            : 0.0397625 ± 6.111e-05
8 accumulators, column RHS, column output         : 0.0440277 ± 0.000109652
8 accumulators, column RHS, row output            : 0.0434018 ± 6.83023e-05
blocked (B = 2), column RHS, column output        : 0.0398408 ± 8.53921e-05
blocked (B = 4), column RHS, column output        : 0.037459 ± 8.81803e-05
blocked (B = 8), column RHS, column output        : 0.0384078 ± 5.39176e-05
blocked (B = 16), column RHS, column output       : 0.0433692 ± 4.28804e-05
blocked (B = 2), column RHS, row output           : 0.0395145 ± 7.24553e-05
blocked (B = 4), column RHS, row output           : 0.0372946 ± 1.90338e-05
blocked (B = 8), column RHS, row output           : 0.0389513 ± 6.29868e-05
blocked (B = 16), column RHS, row output          : 0.0437615 ± 7.06315e-05
blocked (B = 4), row RHS, column output           : 0.0730563 ± 0.000126544
blocked (B = 8), row RHS, column output           : 0.0426288 ± 7.81882e-05
blocked (B = 16), row RHS, column output          : 0.0313343 ± 4.98258e-05
blocked (B = 32), row RHS, column output          : 0.0267209 ± 3.69668e-05
blocked (B = 2), row RHS, row output              : 0.0613085 ± 9.86757e-05
blocked (B = 4), row RHS, row output              : 0.0587273 ± 0.000125084
blocked (B = 8), row RHS, row output              : 0.0574436 ± 0.00013585
blocked (B = 16), row RHS, row output             : 0.0584202 ± 0.000296514

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, column RHS, column output                  : 0.0867979 ± 0.000515005
naive, column RHS, row output                     : 0.0872555 ± 0.000547712
naive, row RHS, column output                     : 0.109416 ± 0.000626677
naive, row RHS, row output                        : 0.101348 ± 0.000536587
2 accumulators, column RHS, column output         : 0.0563021 ± 0.000231057
2 accumulators, column RHS, row output            : 0.0553798 ± 0.00042554
4 accumulators, column RHS, column output         : 0.0387202 ± 0.000274202
4 accumulators, column RHS, row output            : 0.0379262 ± 0.000174661
8 accumulators, column RHS, column output         : 0.0438044 ± 0.000365007
8 accumulators, column RHS, row output            : 0.043468 ± 0.00023953
blocked (B = 2), column RHS, column output        : 0.0393055 ± 0.000217837
blocked (B = 4), column RHS, column output        : 0.0392591 ± 0.000197745
blocked (B = 8), column RHS, column output        : 0.0413088 ± 0.000202588
blocked (B = 16), column RHS, column output       : 0.0462563 ± 0.000207925
blocked (B = 2), column RHS, row output           : 0.0392226 ± 0.000265
blocked (B = 4), column RHS, row output           : 0.0390716 ± 0.000210074
blocked (B = 8), column RHS, row output           : 0.041509 ± 0.000123023
blocked (B = 16), column RHS, row output          : 0.0471043 ± 0.000333802
blocked (B = 4), row RHS, column output           : 0.0925037 ± 0.000501291
blocked (B = 8), row RHS, column output           : 0.0560225 ± 0.00023885
blocked (B = 16), row RHS, column output          : 0.0403395 ± 0.000175473
blocked (B = 32), row RHS, column output          : 0.0332185 ± 0.00019261
blocked (B = 2), row RHS, row output              : 0.0885258 ± 0.00153246
blocked (B = 4), row RHS, row output              : 0.0791197 ± 0.000315025
blocked (B = 8), row RHS, row output              : 0.0719961 ± 0.000358684
blocked (B = 16), row RHS, row output             : 0.0710548 ± 0.000304099

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, column RHS, column output                  : 0.0921568 ± 0.000269039
naive, column RHS, row output                     : 0.104344 ± 0.000300952
naive, row RHS, column output                     : 0.0945517 ± 0.000352201
naive, row RHS, row output                        : 0.0597335 ± 0.000281556
2 accumulators, column RHS, column output         : 0.0896543 ± 0.000299991
2 accumulators, column RHS, row output            : 0.0747801 ± 0.000339035
4 accumulators, column RHS, column output         : 0.0855908 ± 0.000343185
4 accumulators, column RHS, row output            : 0.0707336 ± 0.000377737
8 accumulators, column RHS, column output         : 0.106836 ± 0.000312246
8 accumulators, column RHS, row output            : 0.0960319 ± 0.000504014
blocked (B = 2), column RHS, column output        : 0.0835625 ± 0.000229907
blocked (B = 4), column RHS, column output        : 0.0761653 ± 0.000221387
blocked (B = 8), column RHS, column output        : 0.0708359 ± 0.00026587
blocked (B = 16), column RHS, column output       : 0.0649473 ± 0.000236998
blocked (B = 2), column RHS, row output           : 0.0798481 ± 0.000236797
blocked (B = 4), column RHS, row output           : 0.0747648 ± 0.000318811
blocked (B = 8), column RHS, row output           : 0.0710054 ± 0.000308582
blocked (B = 16), column RHS, row output          : 0.0656064 ± 0.000145211
blocked (B = 4), row RHS, column output           : 0.0852335 ± 0.000314645
blocked (B = 8), row RHS, column output           : 0.0516073 ± 0.000362428
blocked (B = 16), row RHS, column output          : 0.0383989 ± 0.000158791
blocked (B = 32), row RHS, column output          : 0.0322123 ± 0.000190983
blocked (B = 2), row RHS, row output              : 0.0613232 ± 0.000283844
blocked (B = 4), row RHS, row output              : 0.0609991 ± 0.000219096
blocked (B = 8), row RHS, row output              : 0.060382 ± 0.000176984
blocked (B = 16), row RHS, row output             : 0.0613868 ± 7.25287e-05

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, column RHS, column output                  : 0.103345 ± 0.000477627
naive, column RHS, row output                     : 0.115544 ± 0.00049915
naive, row RHS, column output                     : 0.129928 ± 0.000174017
naive, row RHS, row output                        : 0.0629955 ± 0.000346052
2 accumulators, column RHS, column output         : 0.125859 ± 0.000191159
2 accumulators, column RHS, row output            : 0.110943 ± 0.000128082
4 accumulators, column RHS, column output         : 0.126888 ± 0.000246864
4 accumulators, column RHS, row output            : 0.116686 ± 0.00035034
8 accumulators, column RHS, column output         : 0.148888 ± 0.000450556
8 accumulators, column RHS, row output            : 0.139356 ± 0.000312398
blocked (B = 2), column RHS, column output        : 0.106415 ± 0.000121794
blocked (B = 4), column RHS, column output        : 0.0839377 ± 0.000197695
blocked (B = 8), column RHS, column output        : 0.0738644 ± 0.000192831
blocked (B = 16), column RHS, column output       : 0.0656307 ± 5.28497e-05
blocked (B = 2), column RHS, row output           : 0.106107 ± 0.000587185
blocked (B = 4), column RHS, row output           : 0.0841378 ± 9.35025e-05
blocked (B = 8), column RHS, row output           : 0.0743816 ± 0.000113504
blocked (B = 16), column RHS, row output          : 0.066102 ± 0.000156667
blocked (B = 4), row RHS, column output           : 0.168977 ± 0.000474876
blocked (B = 8), row RHS, column output           : 0.0937719 ± 0.000516722
blocked (B = 16), row RHS, column output          : 0.062021 ± 0.00015191
blocked (B = 32), row RHS, column output          : 0.0488383 ± 0.000227413
blocked (B = 2), row RHS, row output              : 0.0625575 ± 0.000162536
blocked (B = 4), row RHS, row output              : 0.0668979 ± 0.000184717
blocked (B = 8), row RHS, row output              : 0.082331 ± 0.000170795
blocked (B = 16), row RHS, row output             : 0.0918166 ± 0.000132207

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, column RHS, column output                  : 0.0985043 ± 5.01755e-05
naive, column RHS, row output                     : 0.0983872 ± 0.000182965
naive, row RHS, column output                     : 0.133827 ± 0.000326655
naive, row RHS, row output                        : 0.132776 ± 0.000291502
2 accumulators, column RHS, column output         : 0.0567701 ± 8.63815e-05
2 accumulators, column RHS, row output            : 0.0564912 ± 0.000151202
4 accumulators, column RHS, column output         : 0.0479288 ± 0.000147027
4 accumulators, column RHS, row output            : 0.0476956 ± 5.32254e-05
8 accumulators, column RHS, column output         : 0.0484862 ± 3.84998e-05
8 accumulators, column RHS, row output            : 0.0484692 ± 6.46982e-05
blocked (B = 2), column RHS, column output        : 0.0484091 ± 7.26901e-05
blocked (B = 4), column RHS, column output        : 0.050891 ± 3.65566e-05
blocked (B = 8), column RHS, column output        : 0.058013 ± 0.000132964
blocked (B = 16), column RHS, column output       : 0.0628469 ± 0.000107729
blocked (B = 2), column RHS, row output           : 0.0485097 ± 6.78753e-05
blocked (B = 4), column RHS, row output           : 0.0510764 ± 0.000107325
blocked (B = 8), column RHS, row output           : 0.0581036 ± 2.66985e-05
blocked (B = 16), column RHS, row output          : 0.0630262 ± 0.000114536
blocked (B = 4), row RHS, column output           : 0.107169 ± 0.000291483
blocked (B = 8), row RHS, column output           : 0.0658881 ± 0.000168029
blocked (B = 16), row RHS, column output          : 0.0451476 ± 0.000203707
blocked (B = 32), row RHS, column output          : 0.0347423 ± 0.000160729
blocked (B = 2), row RHS, row output              : 0.111699 ± 0.00026079
blocked (B = 4), row RHS, row output              : 0.0974254 ± 0.000196737
blocked (B = 8), row RHS, row output              : 0.0866176 ± 0.000120605
blocked (B = 16), row RHS, row output             : 0.0832053 ± 0.000445132

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, column RHS, column output                  : 0.107322 ± 0.00142699
naive, column RHS, row output                     : 0.107979 ± 0.0023564
naive, row RHS, column output                     : 0.107689 ± 0.000827483
naive, row RHS, row output                        : 0.103488 ± 0.000315125
2 accumulators, column RHS, column output         : 0.0738883 ± 0.00509946
2 accumulators, column RHS, row output            : 0.0722156 ± 0.00453789
4 accumulators, column RHS, column output         : 0.0618794 ± 0.0038898
4 accumulators, column RHS, row output            : 0.0600337 ± 0.0032464
8 accumulators, column RHS, column output         : 0.0631911 ± 0.00372305
8 accumulators, column RHS, row output            : 0.0585828 ± 0.00151411
blocked (B = 2), column RHS, column output        : 0.0546805 ± 0.000552653
blocked (B = 4), column RHS, column output        : 0.0549886 ± 0.000356375
blocked (B = 8), column RHS, column output        : 0.0619882 ± 0.000117139
blocked (B = 16), column RHS, column output       : 0.0672666 ± 0.000890313
blocked (B = 2), column RHS, row output           : 0.0540954 ± 0.000247613
blocked (B = 4), column RHS, row output           : 0.0572129 ± 0.00224875
blocked (B = 8), column RHS, row output           : 0.0624205 ± 0.000303419
blocked (B = 16), column RHS, row output          : 0.0672469 ± 0.000615879
blocked (B = 4), row RHS, column output           : 0.0865425 ± 0.00168684
blocked (B = 8), row RHS, column output           : 0.0556061 ± 0.00139586
blocked (B = 16), row RHS, column output          : 0.0388881 ± 0.000659253
blocked (B = 32), row RHS, column output          : 0.0314247 ± 9.30535e-05
blocked (B = 2), row RHS, row output              : 0.0900144 ± 0.00398816
blocked (B = 4), row RHS, row output              : 0.0785474 ± 0.00293748
blocked (B = 8), row RHS, row output              : 0.068009 ± 0.00185547
blocked (B = 16), row RHS, row output             : 0.0647013 ± 0.00033422

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, column RHS, column output                  : 0.105094 ± 0.00277082
naive, column RHS, row output                     : 0.10368 ± 0.00216131
naive, row RHS, column output                     : 0.0872493 ± 0.0027537
naive, row RHS, row output                        : 0.0735858 ± 0.00145879
2 accumulators, column RHS, column output         : 0.0780683 ± 0.00193755
2 accumulators, column RHS, row output            : 0.070972 ± 0.00178345
4 accumulators, column RHS, column output         : 0.063118 ± 0.000673068
4 accumulators, column RHS, row output            : 0.060428 ± 0.00348445
8 accumulators, column RHS, column output         : 0.073609 ± 0.00363637
8 accumulators, column RHS, row output            : 0.0630023 ± 0.0021434
blocked (B = 2), column RHS, column output        : 0.0522667 ± 0.00114213
blocked (B = 4), column RHS, column output        : 0.0467299 ± 0.00154741
blocked (B = 8), column RHS, column output        : 0.0433021 ± 0.000708268
blocked (B = 16), column RHS, column output       : 0.0459944 ± 0.000461968
blocked (B = 2), column RHS, row output           : 0.0512607 ± 0.0032257
blocked (B = 4), column RHS, row output           : 0.0435837 ± 0.000956208
blocked (B = 8), column RHS, row output           : 0.0438677 ± 0.00103072
blocked (B = 16), column RHS, row output          : 0.0463123 ± 0.000328841
blocked (B = 4), row RHS, column output           : 0.114634 ± 0.00177626
blocked (B = 8), row RHS, column output           : 0.068332 ± 0.000491192
blocked (B = 16), row RHS, column output          : 0.0495333 ± 0.000944705
blocked (B = 32), row RHS, column output          : 0.0387599 ± 0.000682002
blocked (B = 2), row RHS, row output              : 0.0689168 ± 0.000959863
blocked (B = 4), row RHS, row output              : 0.0715835 ± 0.000790334
blocked (B = 8), row RHS, row output              : 0.0846382 ± 0.000862388
blocked (B = 16), row RHS, row output             : 0.0914229 ± 0.000711777
```

## Conclusion

Multiple accumulators often improve performance for multiplication with a column-major RHS matrix.
This is most apparent when the shared dimension extent is high, though it's also not too bad when the shared dimension is small.
It seems that 4 accumulators is typically the sweet spot, similar to our conclusions in the [`dense`](../single_vector) case.

Performance is pretty good for a row-major RHS matrix with row-major output.
I find this interesting because a sparse vector multiply-add is not easily vectorizable.
The values of the indices are not known to the compiler, so it can't just execute the instructions in parallel as it can't be sure that the indices don't overlap.

Blocking is consistently helpful for a row-major RHS matrix with column-major output.
This is because it eliminates the need for non-contiguous writes in the innermost loop.
$B = 16$ is again the sweet spot that gets most of the benefits without requiring too much memory usage.

For the other configurations, blocking is sometimes helpful and sometimes detrimental.
In general, it's a pessimisation when the extent of the sparse vector is large but an improvement when it's short, and this difference is amplified by increasing $B$.
This is consistent with whether the block of dense vectors can be held in cache for quick re-use with the next sparse vector.
