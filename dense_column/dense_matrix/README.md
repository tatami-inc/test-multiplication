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
For each RHS column $h$, we perform a vector multiply-add of LHS column $i$ to a temporary buffer, using the RHS entry $(i, h)$ as the scaling factor.
Once all RHS columns have been traversed, we transpose the temporary buffer into the $i$-th elements of all output rows.
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
naive, column RHS, row output               : 0.410307 ± 0.00191576
naive, row RHS, column output               : 0.406372 ± 0.00114939
naive, row RHS, row output                  : 0.470523 ± 0.00609405
best column RHS, column output              : 0.29349 ± 0.00119879
blocked (B = 4), column RHS, row output     : 0.675329 ± 0.00559642
blocked (B = 8), column RHS, row output     : 0.480915 ± 0.00258964
blocked (B = 16), column RHS, row output    : 0.39234 ± 0.000831853
blocked (B = 32), column RHS, row output    : 0.320877 ± 0.00111188
blocked (B = 4), row RHS, column output     : 0.324962 ± 0.00347364
blocked (B = 8), row RHS, column output     : 0.287703 ± 0.000942631
blocked (B = 16), row RHS, column output    : 0.286524 ± 0.00155926
blocked (B = 32), row RHS, column output    : 0.299362 ± 0.000753913
blocked (B = 4), row RHS, row output        : 0.322451 ± 0.00158962
blocked (B = 8), row RHS, row output        : 0.282095 ± 0.000987572
blocked (B = 16), row RHS, row output       : 0.275237 ± 0.000944592
blocked (B = 32), row RHS, row output       : 0.275452 ± 0.000824818

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, column RHS, row output               : 0.581248 ± 0.00587327
naive, row RHS, column output               : 0.448554 ± 0.0100034
naive, row RHS, row output                  : 0.474424 ± 0.00374164
best column RHS, column output              : 0.296688 ± 0.00240913
blocked (B = 4), column RHS, row output     : 0.759814 ± 0.00949711
blocked (B = 8), column RHS, row output     : 0.496396 ± 0.00243618
blocked (B = 16), column RHS, row output    : 0.393739 ± 0.00107682
blocked (B = 32), column RHS, row output    : 0.348449 ± 0.00108996
blocked (B = 4), row RHS, column output     : 0.350172 ± 0.0116082
blocked (B = 8), row RHS, column output     : 0.298529 ± 0.00176577
blocked (B = 16), row RHS, column output    : 0.292863 ± 0.00291783
blocked (B = 32), row RHS, column output    : 0.325062 ± 0.00169495
blocked (B = 4), row RHS, row output        : 0.34167 ± 0.0178835
blocked (B = 8), row RHS, row output        : 0.295482 ± 0.00345575
blocked (B = 16), row RHS, row output       : 0.283729 ± 0.000996208
blocked (B = 32), row RHS, row output       : 0.286637 ± 0.00100233

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, column RHS, row output               : 0.566558 ± 0.00669069
naive, row RHS, column output               : 0.852382 ± 0.00866346
naive, row RHS, row output                  : 0.849878 ± 0.00510134
best column RHS, column output              : 0.290523 ± 0.00195222
blocked (B = 4), column RHS, row output     : 1.12135 ± 0.00537975
blocked (B = 8), column RHS, row output     : 0.649558 ± 0.00257772
blocked (B = 16), column RHS, row output    : 0.460174 ± 0.00252584
blocked (B = 32), column RHS, row output    : 0.366802 ± 0.00126486
blocked (B = 4), row RHS, column output     : 0.398638 ± 0.00419721
blocked (B = 8), row RHS, column output     : 0.326846 ± 0.00334892
blocked (B = 16), row RHS, column output    : 0.29669 ± 0.00122849
blocked (B = 32), row RHS, column output    : 0.316613 ± 0.0033839
blocked (B = 4), row RHS, row output        : 0.402994 ± 0.00260867
blocked (B = 8), row RHS, row output        : 0.322833 ± 0.0019221
blocked (B = 16), row RHS, row output       : 0.296536 ± 0.00114037
blocked (B = 32), row RHS, row output       : 0.285117 ± 0.00351383

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, column RHS, row output               : 0.336743 ± 0.00257074
naive, row RHS, column output               : 0.925911 ± 0.0409507
naive, row RHS, row output                  : 0.86984 ± 0.0247857
best column RHS, column output              : 0.302581 ± 0.00349213
blocked (B = 4), column RHS, row output     : 0.938699 ± 0.0130053
blocked (B = 8), column RHS, row output     : 0.615405 ± 0.00803386
blocked (B = 16), column RHS, row output    : 0.456179 ± 0.0053236
blocked (B = 32), column RHS, row output    : 0.340407 ± 0.00222082
blocked (B = 4), row RHS, column output     : 0.419201 ± 0.00705063
blocked (B = 8), row RHS, column output     : 0.336619 ± 0.00441775
blocked (B = 16), row RHS, column output    : 0.312416 ± 0.00364209
blocked (B = 32), row RHS, column output    : 0.312488 ± 0.0029069
blocked (B = 4), row RHS, row output        : 0.403414 ± 0.00664981
blocked (B = 8), row RHS, row output        : 0.324409 ± 0.00439103
blocked (B = 16), row RHS, row output       : 0.288467 ± 0.00280063
blocked (B = 32), row RHS, row output       : 0.291026 ± 0.00275733

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, column RHS, row output               : 0.578929 ± 0.0274475
naive, row RHS, column output               : 0.270326 ± 0.00153092
naive, row RHS, row output                  : 0.37122 ± 0.00138711
best column RHS, column output              : 0.284871 ± 0.00179818
blocked (B = 4), column RHS, row output     : 0.524383 ± 0.00291819
blocked (B = 8), column RHS, row output     : 0.390411 ± 0.00175355
blocked (B = 16), column RHS, row output    : 0.345522 ± 0.00149836
blocked (B = 32), column RHS, row output    : 0.317916 ± 0.000884511
blocked (B = 4), row RHS, column output     : 0.28087 ± 0.00187482
blocked (B = 8), row RHS, column output     : 0.266912 ± 0.00131745
blocked (B = 16), row RHS, column output    : 0.279415 ± 0.00160809
blocked (B = 32), row RHS, column output    : 0.298226 ± 0.00140614
blocked (B = 4), row RHS, row output        : 0.273144 ± 0.00111346
blocked (B = 8), row RHS, row output        : 0.260562 ± 0.000896055
blocked (B = 16), row RHS, row output       : 0.265438 ± 0.0017776
blocked (B = 32), row RHS, row output       : 0.274143 ± 0.000962535

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, column RHS, row output               : 0.294175 ± 0.00121974
naive, row RHS, column output               : 0.41998 ± 0.0039185
naive, row RHS, row output                  : 0.47302 ± 0.00674844
best column RHS, column output              : 0.353218 ± 0.00886569
blocked (B = 4), column RHS, row output     : 0.853054 ± 0.0185053
blocked (B = 8), column RHS, row output     : 0.588072 ± 0.013757
blocked (B = 16), column RHS, row output    : 0.441898 ± 0.00302783
blocked (B = 32), column RHS, row output    : 0.363245 ± 0.00427451
blocked (B = 4), row RHS, column output     : 0.332625 ± 0.00567772
blocked (B = 8), row RHS, column output     : 0.295764 ± 0.00259124
blocked (B = 16), row RHS, column output    : 0.301499 ± 0.00351722
blocked (B = 32), row RHS, column output    : 0.318951 ± 0.00322775
blocked (B = 4), row RHS, row output        : 0.330751 ± 0.00274498
blocked (B = 8), row RHS, row output        : 0.287383 ± 0.00283725
blocked (B = 16), row RHS, row output       : 0.276266 ± 0.0016317
blocked (B = 32), row RHS, row output       : 0.296305 ± 0.00150881

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, column RHS, row output               : 0.438141 ± 0.0203612
naive, row RHS, column output               : 0.281523 ± 0.000581997
naive, row RHS, row output                  : 0.366984 ± 0.000669008
best column RHS, column output              : 0.298502 ± 0.00105497
blocked (B = 4), column RHS, row output     : 0.552425 ± 0.00143265
blocked (B = 8), column RHS, row output     : 0.417708 ± 0.00186106
blocked (B = 16), column RHS, row output    : 0.365181 ± 0.000721592
blocked (B = 32), column RHS, row output    : 0.33208 ± 0.00175645
blocked (B = 4), row RHS, column output     : 0.277755 ± 0.000560664
blocked (B = 8), row RHS, column output     : 0.267096 ± 0.000459854
blocked (B = 16), row RHS, column output    : 0.283745 ± 0.00279845
blocked (B = 32), row RHS, column output    : 0.304485 ± 0.000828012
blocked (B = 4), row RHS, row output        : 0.279693 ± 0.000884956
blocked (B = 8), row RHS, row output        : 0.260187 ± 0.00151313
blocked (B = 16), row RHS, row output       : 0.266004 ± 0.000609679
blocked (B = 32), row RHS, row output       : 0.271217 ± 0.000516993
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive, column RHS, row output               : 0.158457 ± 0.000251198
naive, row RHS, column output               : 0.152229 ± 0.00035095
naive, row RHS, row output                  : 0.14789 ± 0.000431385
best column RHS, column output              : 0.180997 ± 0.000211027
blocked (B = 4), column RHS, row output     : 0.366642 ± 0.00313485
blocked (B = 8), column RHS, row output     : 0.31631 ± 0.00205628
blocked (B = 16), column RHS, row output    : 0.27059 ± 0.000781797
blocked (B = 32), column RHS, row output    : 0.200234 ± 0.00185398
blocked (B = 4), row RHS, column output     : 0.153319 ± 0.00144075
blocked (B = 8), row RHS, column output     : 0.15488 ± 0.00019557
blocked (B = 16), row RHS, column output    : 0.175155 ± 0.000303103
blocked (B = 32), row RHS, column output    : 0.22155 ± 0.000186442
blocked (B = 4), row RHS, row output        : 0.203377 ± 0.000263293
blocked (B = 8), row RHS, row output        : 0.200467 ± 0.00144089
blocked (B = 16), row RHS, row output       : 0.207835 ± 0.00146026
blocked (B = 32), row RHS, row output       : 0.223724 ± 0.00033579

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, column RHS, row output               : 0.282901 ± 0.00215165
naive, row RHS, column output               : 0.158756 ± 0.000692978
naive, row RHS, row output                  : 0.159877 ± 0.000744569
best column RHS, column output              : 0.180956 ± 0.000382946
blocked (B = 4), column RHS, row output     : 0.393052 ± 0.00033383
blocked (B = 8), column RHS, row output     : 0.31316 ± 0.00026979
blocked (B = 16), column RHS, row output    : 0.273193 ± 0.000491376
blocked (B = 32), column RHS, row output    : 0.195441 ± 0.000631713
blocked (B = 4), row RHS, column output     : 0.150227 ± 0.000269683
blocked (B = 8), row RHS, column output     : 0.155091 ± 0.000224785
blocked (B = 16), row RHS, column output    : 0.1756 ± 0.000478942
blocked (B = 32), row RHS, column output    : 0.221844 ± 0.000200603
blocked (B = 4), row RHS, row output        : 0.206317 ± 0.000418823
blocked (B = 8), row RHS, row output        : 0.205346 ± 0.000414001
blocked (B = 16), row RHS, row output       : 0.220866 ± 0.00522823
blocked (B = 32), row RHS, row output       : 0.240313 ± 0.000282025

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, column RHS, row output               : 0.249133 ± 0.0077293
naive, row RHS, column output               : 0.420122 ± 0.0131537
naive, row RHS, row output                  : 0.446411 ± 0.0169348
best column RHS, column output              : 0.186466 ± 0.000910186
blocked (B = 4), column RHS, row output     : 0.721881 ± 0.00652394
blocked (B = 8), column RHS, row output     : 0.459545 ± 0.004486
blocked (B = 16), column RHS, row output    : 0.352649 ± 0.00586993
blocked (B = 32), column RHS, row output    : 0.234727 ± 0.00293422
blocked (B = 4), row RHS, column output     : 0.201086 ± 0.00316955
blocked (B = 8), row RHS, column output     : 0.176863 ± 0.00282368
blocked (B = 16), row RHS, column output    : 0.186594 ± 0.00234959
blocked (B = 32), row RHS, column output    : 0.228948 ± 0.00136429
blocked (B = 4), row RHS, row output        : 0.257162 ± 0.00254558
blocked (B = 8), row RHS, row output        : 0.230127 ± 0.0035107
blocked (B = 16), row RHS, row output       : 0.222251 ± 0.00221354
blocked (B = 32), row RHS, row output       : 0.231629 ± 0.00140024

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, column RHS, row output               : 0.177559 ± 0.00110483
naive, row RHS, column output               : 0.416479 ± 0.0141878
naive, row RHS, row output                  : 0.421845 ± 0.0154778
best column RHS, column output              : 0.19041 ± 0.000939366
blocked (B = 4), column RHS, row output     : 0.578441 ± 0.00629483
blocked (B = 8), column RHS, row output     : 0.416568 ± 0.00387155
blocked (B = 16), column RHS, row output    : 0.327622 ± 0.00311478
blocked (B = 32), column RHS, row output    : 0.209849 ± 0.00170332
blocked (B = 4), row RHS, column output     : 0.215332 ± 0.00293974
blocked (B = 8), row RHS, column output     : 0.183021 ± 0.00201577
blocked (B = 16), row RHS, column output    : 0.190563 ± 0.00193873
blocked (B = 32), row RHS, column output    : 0.226002 ± 0.000971328
blocked (B = 4), row RHS, row output        : 0.236518 ± 0.00399373
blocked (B = 8), row RHS, row output        : 0.215862 ± 0.00246057
blocked (B = 16), row RHS, row output       : 0.214567 ± 0.00198814
blocked (B = 32), row RHS, row output       : 0.226055 ± 0.000883115

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, column RHS, row output               : 0.273859 ± 0.00368255
naive, row RHS, column output               : 0.135061 ± 0.000494825
naive, row RHS, row output                  : 0.143627 ± 0.000742722
best column RHS, column output              : 0.180572 ± 0.000621043
blocked (B = 4), column RHS, row output     : 0.338962 ± 0.000934059
blocked (B = 8), column RHS, row output     : 0.305489 ± 0.000921134
blocked (B = 16), column RHS, row output    : 0.264923 ± 0.00044182
blocked (B = 32), column RHS, row output    : 0.191263 ± 0.000724482
blocked (B = 4), row RHS, column output     : 0.147583 ± 0.000584321
blocked (B = 8), row RHS, column output     : 0.154272 ± 0.000792774
blocked (B = 16), row RHS, column output    : 0.176633 ± 0.000708805
blocked (B = 32), row RHS, column output    : 0.222896 ± 0.000700362
blocked (B = 4), row RHS, row output        : 0.20427 ± 0.0010309
blocked (B = 8), row RHS, row output        : 0.20278 ± 0.000740042
blocked (B = 16), row RHS, row output       : 0.214053 ± 0.00149644
blocked (B = 32), row RHS, row output       : 0.239569 ± 0.00126342

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 1000 x 10000
naive, column RHS, row output               : 0.150576 ± 0.00077886
naive, row RHS, column output               : 0.172479 ± 0.00786846
naive, row RHS, row output                  : 0.161306 ± 0.00265935
best column RHS, column output              : 0.223798 ± 0.00706293
blocked (B = 4), column RHS, row output     : 0.456901 ± 0.0167431
blocked (B = 8), column RHS, row output     : 0.369446 ± 0.0126635
blocked (B = 16), column RHS, row output    : 0.313757 ± 0.00697819
blocked (B = 32), column RHS, row output    : 0.242936 ± 0.00628363
blocked (B = 4), row RHS, column output     : 0.173404 ± 0.00450005
blocked (B = 8), row RHS, column output     : 0.169651 ± 0.00255551
blocked (B = 16), row RHS, column output    : 0.19514 ± 0.001597
blocked (B = 32), row RHS, column output    : 0.238213 ± 0.00104053
blocked (B = 4), row RHS, row output        : 0.203339 ± 0.00193935
blocked (B = 8), row RHS, row output        : 0.205618 ± 0.00393141
blocked (B = 16), row RHS, row output       : 0.211022 ± 0.00248878
blocked (B = 32), row RHS, row output       : 0.22733 ± 0.00179976

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 10000 x 1000
naive, column RHS, row output               : 0.165364 ± 0.00299665
naive, row RHS, column output               : 0.142857 ± 0.000425823
naive, row RHS, row output                  : 0.136483 ± 0.000709431
best column RHS, column output              : 0.19207 ± 0.000694679
blocked (B = 4), column RHS, row output     : 0.387247 ± 0.00190064
blocked (B = 8), column RHS, row output     : 0.319253 ± 0.0009836
blocked (B = 16), column RHS, row output    : 0.279736 ± 0.00130105
blocked (B = 32), column RHS, row output    : 0.209214 ± 0.000938432
blocked (B = 4), row RHS, column output     : 0.162108 ± 0.000811677
blocked (B = 8), row RHS, column output     : 0.162116 ± 0.0014655
blocked (B = 16), row RHS, column output    : 0.191762 ± 0.000873852
blocked (B = 32), row RHS, column output    : 0.236172 ± 0.000665423
blocked (B = 4), row RHS, row output        : 0.200138 ± 0.000593057
blocked (B = 8), row RHS, row output        : 0.200841 ± 0.000662227
blocked (B = 16), row RHS, row output       : 0.209164 ± 0.00118012
blocked (B = 32), row RHS, row output       : 0.227638 ± 0.000723136
```

## Conclusion

For products involving a row-major RHS, blocking consistently provides some assistance across a range of matrix shapes.
$B = of 16 seems to perform the best, or nearly so.
The exception is that of single-precision floats where blocking is slightly worse than the naive approach, probably because the data is already small enough to fit into cache. 

For column-major RHS with row-major output, the benefits of blocking are less clear.
It is a pessimisation when the output matrix is large, presumably because the transposition of the contents of the temporary buffers is relatively time-consuming.
When it does provide a benefit, it usually only does so at $B = 32$, which is consistent with square blocks being more friendly to transposition. 
