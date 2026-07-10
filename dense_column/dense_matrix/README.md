# Dense column-major LHS, dense matrix RHS

## Strategies

All strategies must iterate across consecutive columns of the LHS matrix in the outermost loop.
We will be loading columns on demand via the **tatami** interface, so the full matrix will not be available for random access.

### Naive row-major RHS, row-major output

For each LHS column $i$, we iterate over the LHS rows.
For each LHS row $j$, we perform a vector multiply-add of RHS row $i$ to output row $j$, using the LHS entry $(j, i)$ as the scaling factor.
We repeat this for each LHS column until the entirety of the LHS matrix is traversed.

### Naive row-major RHS, column-major output

For each LHS column $i$, we iterate over the RHS columns.
For each RHS column $h$, we perform a vector multiply-add of LHS column $i$ to output row $h$, using the RHS entry $(i, h)$ as the scaling factor.
We repeat this for each LHS column until the entirety of the LHS matrix is traversed.

### Naive column-major RHS, row-major output

For each LHS column $i$, we iterate over the RHS columns.
For each RHS column $h$, we perform a vector multiply-add of LHS column $i$ to the conceptual output column $h$, using the RHS entry $(i, h)$ as the scaling factor.
(The conceptual column $h$ consists of the $h$-th elements of all output rows.)
We repeat this for each LHS column until the entirety of the LHS matrix is traversed.

### Best column-major RHS, column-major output

We don't bother testing this configuration as we already did so in the [`multiple_vectors`](../multiple_vectors) tests.
So, for comparison's sake, we'll just use the best approach from that test suite, i.e., blocking with $B = 16$.

### Blocking, row-major RHS, row-major output

We consider $B$-by-$C$ RHS and output submatrices and a $B$-by-$B$ LHS submatrix.
In this set of submatrices, each LHS row corresponds to an output row, each LHS column corresponds to an RHS row, and each RHS column corresponds to an output column.

For each row $i$ in the RHS submatrix, we loop over the rows of the output submatrix.
We perform a vector multiply-add of RHS row $i$ to output row $j$ using the LHS entry $(j, i)$ as a scaling factor.
We repeat this for all valid sets of submatrices until the full matrix product is computed.
When choosing the next set of submatrices, the RHS/output columns are the fastest-changing dimension while the LHS columns are the slowest.

The hope is that these submatrices are small enough to keep in cache for fast access.
We test a range of different values for the $B$ given a fixed value for $BC = 1024$. 
Check out [`general/README.md`](../../general/README.md) for more details.

### Blocking, row-major RHS, column-major output

We consider $C$-by-$B$ LHS and output submatrices and a $B$-by-$B$ RHS submatrix.
In this set of submatrices, each LHS row corresponds to an output row, each LHS column corresponds to an RHS row, and each RHS column corresponds to an output column.

For each column $i$ in the LHS submatrix, we loop over the columns of the RHS submatrix.
For each RHS column $j$, we perform a vector multiply-add of LHS column $i$ to output column $j$ using the RHS entry $(i, j)$ as the scaling factor.
We repeat this for all valid sets of submatrices until the full matrix product is computed.
When choosing the next set of submatrices, the LHS/output rows are the fastest-changing dimension while the LHS columns are the slowest.

The hope is that these submatrices are small enough to keep in cache for fast access.
We test a range of different values for the $B$ given a fixed value for $BC = 1024$. 
Check out [`general/README.md`](../../general/README.md) for more details.

### Blocking, column-major RHS, row-major output

The process is the same as for column-major output, except that we use temporary buffers to represent the output submatrix in column-major format.
After processing each set of submatrices, we transpose the contents of the temporary buffers into the actual row-major output submatrix.
We don't have to worry much about extra cache usage as none of the submatrices will be re-used anyway.

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
naive, column RHS, row output               : 1.47305 ± 0.00186042
naive, row RHS, column output               : 0.41812 ± 0.000832731
naive, row RHS, row output                  : 0.488289 ± 0.000844044
best column RHS, column output              : 0.328013 ± 0.000474933
blocked (B = 4), column RHS, row output     : 0.922632 ± 0.00171112
blocked (B = 8), column RHS, row output     : 0.50181 ± 0.000512732
blocked (B = 16), column RHS, row output    : 0.405638 ± 0.000819073
blocked (B = 32), column RHS, row output    : 0.366319 ± 0.000412469
blocked (B = 4), row RHS, column output     : 0.344078 ± 0.00169005
blocked (B = 8), row RHS, column output     : 0.298771 ± 0.00105317
blocked (B = 16), row RHS, column output    : 0.292209 ± 0.00163676
blocked (B = 32), row RHS, column output    : 0.305628 ± 0.000380234
blocked (B = 4), row RHS, row output        : 0.347127 ± 0.000941688
blocked (B = 8), row RHS, row output        : 0.299008 ± 0.000702733
blocked (B = 16), row RHS, row output       : 0.297619 ± 0.00060855
blocked (B = 32), row RHS, row output       : 0.312529 ± 0.000523905

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, column RHS, row output               : 4.28975 ± 0.00176246
naive, row RHS, column output               : 0.455761 ± 0.000638025
naive, row RHS, row output                  : 0.521329 ± 0.0197183
best column RHS, column output              : 0.328769 ± 0.000526815
blocked (B = 4), column RHS, row output     : 1.03711 ± 0.0191201
blocked (B = 8), column RHS, row output     : 0.523213 ± 0.00608177
blocked (B = 16), column RHS, row output    : 0.413082 ± 0.00539162
blocked (B = 32), column RHS, row output    : 0.396572 ± 0.00070595
blocked (B = 4), row RHS, column output     : 0.353863 ± 0.000247244
blocked (B = 8), row RHS, column output     : 0.307261 ± 0.000605712
blocked (B = 16), row RHS, column output    : 0.297433 ± 0.00178511
blocked (B = 32), row RHS, column output    : 0.334565 ± 0.00143407
blocked (B = 4), row RHS, row output        : 0.352142 ± 0.00181669
blocked (B = 8), row RHS, row output        : 0.310324 ± 0.000547291
blocked (B = 16), row RHS, row output       : 0.312273 ± 0.000937443
blocked (B = 32), row RHS, row output       : 0.334147 ± 0.00356132

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, column RHS, row output               : 7.87644 ± 0.0416099
naive, row RHS, column output               : 0.908197 ± 0.00634481
naive, row RHS, row output                  : 0.924365 ± 0.0106099
best column RHS, column output              : 0.324561 ± 0.00108076
blocked (B = 4), column RHS, row output     : 1.41708 ± 0.0108487
blocked (B = 8), column RHS, row output     : 0.697603 ± 0.00743189
blocked (B = 16), column RHS, row output    : 0.48414 ± 0.00261837
blocked (B = 32), column RHS, row output    : 0.422351 ± 0.00333167
blocked (B = 4), row RHS, column output     : 0.425982 ± 0.00196894
blocked (B = 8), row RHS, column output     : 0.344694 ± 0.00338937
blocked (B = 16), row RHS, column output    : 0.311265 ± 0.00324444
blocked (B = 32), row RHS, column output    : 0.324379 ± 0.00283337
blocked (B = 4), row RHS, row output        : 0.461001 ± 0.0154626
blocked (B = 8), row RHS, row output        : 0.357688 ± 0.00694092
blocked (B = 16), row RHS, row output       : 0.325481 ± 0.00120499
blocked (B = 32), row RHS, row output       : 0.32279 ± 0.0026045

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, column RHS, row output               : 2.85123 ± 0.041964
naive, row RHS, column output               : 1.00615 ± 0.0350284
naive, row RHS, row output                  : 1.00258 ± 0.0304963
best column RHS, column output              : 0.342986 ± 0.00379179
blocked (B = 4), column RHS, row output     : 1.22611 ± 0.0141398
blocked (B = 8), column RHS, row output     : 0.647358 ± 0.00666261
blocked (B = 16), column RHS, row output    : 0.481612 ± 0.00640789
blocked (B = 32), column RHS, row output    : 0.391632 ± 0.00389546
blocked (B = 4), row RHS, column output     : 0.454886 ± 0.00898936
blocked (B = 8), row RHS, column output     : 0.355271 ± 0.00341687
blocked (B = 16), row RHS, column output    : 0.32694 ± 0.00335341
blocked (B = 32), row RHS, column output    : 0.318836 ± 0.00191589
blocked (B = 4), row RHS, row output        : 0.447731 ± 0.011172
blocked (B = 8), row RHS, row output        : 0.352207 ± 0.00539213
blocked (B = 16), row RHS, row output       : 0.317959 ± 0.00436889
blocked (B = 32), row RHS, row output       : 0.339725 ± 0.00425858

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, column RHS, row output               : 1.12738 ± 0.00306687
naive, row RHS, column output               : 0.272523 ± 0.00124971
naive, row RHS, row output                  : 0.371773 ± 0.00173722
best column RHS, column output              : 0.321724 ± 0.00268743
blocked (B = 4), column RHS, row output     : 0.771329 ± 0.00170517
blocked (B = 8), column RHS, row output     : 0.406207 ± 0.000784353
blocked (B = 16), column RHS, row output    : 0.351187 ± 0.00160416
blocked (B = 32), column RHS, row output    : 0.361416 ± 0.00184883
blocked (B = 4), row RHS, column output     : 0.296675 ± 0.00120179
blocked (B = 8), row RHS, column output     : 0.27122 ± 0.000779817
blocked (B = 16), row RHS, column output    : 0.282518 ± 0.0013422
blocked (B = 32), row RHS, column output    : 0.304554 ± 0.00142507
blocked (B = 4), row RHS, row output        : 0.290582 ± 0.000860015
blocked (B = 8), row RHS, row output        : 0.275474 ± 0.00118818
blocked (B = 16), row RHS, row output       : 0.291504 ± 0.00246641
blocked (B = 32), row RHS, row output       : 0.321172 ± 0.00156506

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, column RHS, row output               : 1.54969 ± 0.000477187
naive, row RHS, column output               : 0.450131 ± 0.000821496
naive, row RHS, row output                  : 0.493203 ± 0.000863871
best column RHS, column output              : 0.396362 ± 0.000268831
blocked (B = 4), column RHS, row output     : 1.07748 ± 0.00203925
blocked (B = 8), column RHS, row output     : 0.602277 ± 0.000495543
blocked (B = 16), column RHS, row output    : 0.45895 ± 0.00064829
blocked (B = 32), column RHS, row output    : 0.414181 ± 0.000335233
blocked (B = 4), row RHS, column output     : 0.3465 ± 0.000552301
blocked (B = 8), row RHS, column output     : 0.308565 ± 0.000448907
blocked (B = 16), row RHS, column output    : 0.306407 ± 0.00162203
blocked (B = 32), row RHS, column output    : 0.325581 ± 0.00156716
blocked (B = 4), row RHS, row output        : 0.353559 ± 0.0005143
blocked (B = 8), row RHS, row output        : 0.307769 ± 0.000368832
blocked (B = 16), row RHS, row output       : 0.300771 ± 0.000477378
blocked (B = 32), row RHS, row output       : 0.337013 ± 0.000316884

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, column RHS, row output               : 0.929261 ± 0.000282354
naive, row RHS, column output               : 0.279212 ± 0.000285322
naive, row RHS, row output                  : 0.37222 ± 0.00340713
best column RHS, column output              : 0.341048 ± 0.000223823
blocked (B = 4), column RHS, row output     : 0.770927 ± 0.00104886
blocked (B = 8), column RHS, row output     : 0.424105 ± 0.000658064
blocked (B = 16), column RHS, row output    : 0.366185 ± 0.000320822
blocked (B = 32), column RHS, row output    : 0.376368 ± 0.000280648
blocked (B = 4), row RHS, column output     : 0.28768 ± 0.000158274
blocked (B = 8), row RHS, column output     : 0.27138 ± 0.000293129
blocked (B = 16), row RHS, column output    : 0.282798 ± 0.000715967
blocked (B = 32), row RHS, column output    : 0.314191 ± 0.00476613
blocked (B = 4), row RHS, row output        : 0.292042 ± 0.000308279
blocked (B = 8), row RHS, row output        : 0.271941 ± 0.00045367
blocked (B = 16), row RHS, row output       : 0.28598 ± 0.000303644
blocked (B = 32), row RHS, row output       : 0.309007 ± 0.000431224
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive, column RHS, row output               : 1.0157 ± 0.00158754
naive, row RHS, column output               : 0.153365 ± 0.00144387
naive, row RHS, row output                  : 0.152713 ± 0.00198375
best column RHS, column output              : 0.147651 ± 0.000834267
blocked (B = 4), column RHS, row output     : 0.36101 ± 0.00167382
blocked (B = 8), column RHS, row output     : 0.323572 ± 0.00133253
blocked (B = 16), column RHS, row output    : 0.211161 ± 0.000818147
blocked (B = 32), column RHS, row output    : 0.259641 ± 0.000460722
blocked (B = 4), row RHS, column output     : 0.152086 ± 0.00155686
blocked (B = 8), row RHS, column output     : 0.147735 ± 0.00124202
blocked (B = 16), row RHS, column output    : 0.154109 ± 0.00096228
blocked (B = 32), row RHS, column output    : 0.172369 ± 0.000304319
blocked (B = 4), row RHS, row output        : 0.208414 ± 0.00138518
blocked (B = 8), row RHS, row output        : 0.202081 ± 0.00115514
blocked (B = 16), row RHS, row output       : 0.207952 ± 0.000453807
blocked (B = 32), row RHS, row output       : 0.224797 ± 0.000531708

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, column RHS, row output               : 2.39184 ± 0.0103696
naive, row RHS, column output               : 0.174217 ± 0.0105206
naive, row RHS, row output                  : 0.202778 ± 0.0150019
best column RHS, column output              : 0.152701 ± 0.00291203
blocked (B = 4), column RHS, row output     : 0.433323 ± 0.00972169
blocked (B = 8), column RHS, row output     : 0.345045 ± 0.0034089
blocked (B = 16), column RHS, row output    : 0.223402 ± 0.00612893
blocked (B = 32), column RHS, row output    : 0.281385 ± 0.00657845
blocked (B = 4), row RHS, column output     : 0.163339 ± 0.00734139
blocked (B = 8), row RHS, column output     : 0.160735 ± 0.00402575
blocked (B = 16), row RHS, column output    : 0.156568 ± 0.00152585
blocked (B = 32), row RHS, column output    : 0.177524 ± 0.00189618
blocked (B = 4), row RHS, row output        : 0.239625 ± 0.012751
blocked (B = 8), row RHS, row output        : 0.21544 ± 0.00382236
blocked (B = 16), row RHS, row output       : 0.225931 ± 0.00392352
blocked (B = 32), row RHS, row output       : 0.250852 ± 0.00341828

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, column RHS, row output               : 6.50236 ± 0.0261801
naive, row RHS, column output               : 0.566572 ± 0.0452681
naive, row RHS, row output                  : 0.464018 ± 0.00583044
best column RHS, column output              : 0.15601 ± 0.00257174
blocked (B = 4), column RHS, row output     : 0.778919 ± 0.0163681
blocked (B = 8), column RHS, row output     : 0.496805 ± 0.00318736
blocked (B = 16), column RHS, row output    : 0.300718 ± 0.00361045
blocked (B = 32), column RHS, row output    : 0.310273 ± 0.00374698
blocked (B = 4), row RHS, column output     : 0.218805 ± 0.00812
blocked (B = 8), row RHS, column output     : 0.174959 ± 0.00138474
blocked (B = 16), row RHS, column output    : 0.179272 ± 0.00513871
blocked (B = 32), row RHS, column output    : 0.1799 ± 0.00103529
blocked (B = 4), row RHS, row output        : 0.27238 ± 0.00316135
blocked (B = 8), row RHS, row output        : 0.237093 ± 0.00185467
blocked (B = 16), row RHS, row output       : 0.228454 ± 0.00232018
blocked (B = 32), row RHS, row output       : 0.232624 ± 0.000780201

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, column RHS, row output               : 2.74387 ± 0.0140655
naive, row RHS, column output               : 0.487865 ± 0.0240614
naive, row RHS, row output                  : 0.484833 ± 0.0227596
best column RHS, column output              : 0.166825 ± 0.0031279
blocked (B = 4), column RHS, row output     : 0.612198 ± 0.00908737
blocked (B = 8), column RHS, row output     : 0.448808 ± 0.00337682
blocked (B = 16), column RHS, row output    : 0.276103 ± 0.00158075
blocked (B = 32), column RHS, row output    : 0.28657 ± 0.00160498
blocked (B = 4), row RHS, column output     : 0.230879 ± 0.00333025
blocked (B = 8), row RHS, column output     : 0.188359 ± 0.00420107
blocked (B = 16), row RHS, column output    : 0.173263 ± 0.00215808
blocked (B = 32), row RHS, column output    : 0.179651 ± 0.000802335
blocked (B = 4), row RHS, row output        : 0.255706 ± 0.00555581
blocked (B = 8), row RHS, row output        : 0.226974 ± 0.00179715
blocked (B = 16), row RHS, row output       : 0.222398 ± 0.00195965
blocked (B = 32), row RHS, row output       : 0.230515 ± 0.0011198

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, column RHS, row output               : 1.06932 ± 0.0035949
naive, row RHS, column output               : 0.137698 ± 0.00073253
naive, row RHS, row output                  : 0.159426 ± 0.000809064
best column RHS, column output              : 0.146246 ± 0.00062961
blocked (B = 4), column RHS, row output     : 0.348237 ± 0.00220518
blocked (B = 8), column RHS, row output     : 0.321209 ± 0.0017114
blocked (B = 16), column RHS, row output    : 0.202677 ± 0.000855002
blocked (B = 32), column RHS, row output    : 0.261894 ± 0.00195443
blocked (B = 4), row RHS, column output     : 0.145505 ± 0.000834064
blocked (B = 8), row RHS, column output     : 0.145586 ± 0.00116404
blocked (B = 16), row RHS, column output    : 0.155253 ± 0.00089087
blocked (B = 32), row RHS, column output    : 0.176417 ± 0.0010635
blocked (B = 4), row RHS, row output        : 0.203445 ± 0.000838945
blocked (B = 8), row RHS, row output        : 0.206316 ± 0.000975934
blocked (B = 16), row RHS, row output       : 0.217227 ± 0.00171116
blocked (B = 32), row RHS, row output       : 0.245533 ± 0.00122786

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 1000 x 10000
naive, column RHS, row output               : 0.94404 ± 0.00144073
naive, row RHS, column output               : 0.173555 ± 0.000274761
naive, row RHS, row output                  : 0.160905 ± 0.000554032
best column RHS, column output              : 0.18616 ± 0.000600004
blocked (B = 4), column RHS, row output     : 0.433718 ± 0.00120593
blocked (B = 8), column RHS, row output     : 0.368627 ± 0.000845659
blocked (B = 16), column RHS, row output    : 0.253728 ± 0.000839013
blocked (B = 32), column RHS, row output    : 0.305644 ± 0.000650835
blocked (B = 4), row RHS, column output     : 0.156166 ± 0.000329739
blocked (B = 8), row RHS, column output     : 0.153267 ± 0.000225677
blocked (B = 16), row RHS, column output    : 0.165943 ± 0.000748553
blocked (B = 32), row RHS, column output    : 0.188577 ± 0.000169508
blocked (B = 4), row RHS, row output        : 0.201419 ± 0.000291289
blocked (B = 8), row RHS, row output        : 0.201634 ± 0.000341862
blocked (B = 16), row RHS, row output       : 0.207779 ± 0.000212276
blocked (B = 32), row RHS, row output       : 0.2247 ± 0.000170642

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 10000 x 1000
naive, column RHS, row output               : 0.874618 ± 0.00342569
naive, row RHS, column output               : 0.158247 ± 0.000876315
naive, row RHS, row output                  : 0.137744 ± 0.000897382
best column RHS, column output              : 0.162221 ± 0.00153408
blocked (B = 4), column RHS, row output     : 0.377079 ± 0.00176536
blocked (B = 8), column RHS, row output     : 0.325261 ± 0.00246689
blocked (B = 16), column RHS, row output    : 0.217631 ± 0.00152331
blocked (B = 32), column RHS, row output    : 0.280709 ± 0.00237711
blocked (B = 4), row RHS, column output     : 0.146008 ± 0.000449356
blocked (B = 8), row RHS, column output     : 0.148314 ± 0.000920793
blocked (B = 16), row RHS, column output    : 0.161821 ± 0.000638363
blocked (B = 32), row RHS, column output    : 0.187144 ± 0.00069804
blocked (B = 4), row RHS, row output        : 0.202156 ± 0.00117493
blocked (B = 8), row RHS, row output        : 0.200524 ± 0.000910865
blocked (B = 16), row RHS, row output       : 0.209848 ± 0.00149024
blocked (B = 32), row RHS, row output       : 0.226941 ± 0.000575084
```

## Conclusion

Blocking consistently provides some assistance across a range of matrix shapes.
$B = 16$ seems to perform the best, or nearly so.
The exception is that of single-precision floats where blocking is sometimes slightly worse than the naive approach,
probably because the data is already small enough to fit into cache. 
