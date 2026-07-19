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
naive, column RHS, row output               : 1.45273 ± 0.00732121
naive, row RHS, column output               : 0.398225 ± 0.00449492
naive, row RHS, row output                  : 0.455635 ± 0.00305141
best column RHS, column output              : 0.320249 ± 0.00153884
blocked (B = 4), column RHS, row output     : 0.663862 ± 0.00187173
blocked (B = 8), column RHS, row output     : 0.501172 ± 0.00162061
blocked (B = 16), column RHS, row output    : 0.359796 ± 0.00112321
blocked (B = 32), column RHS, row output    : 0.320612 ± 0.00117378
blocked (B = 4), row RHS, column output     : 0.327547 ± 0.0033609
blocked (B = 8), row RHS, column output     : 0.286044 ± 0.00151342
blocked (B = 16), row RHS, column output    : 0.288496 ± 0.00285113
blocked (B = 32), row RHS, column output    : 0.300841 ± 0.00148923
blocked (B = 4), row RHS, row output        : 0.327587 ± 0.00221231
blocked (B = 8), row RHS, row output        : 0.286357 ± 0.00130078
blocked (B = 16), row RHS, row output       : 0.291165 ± 0.000944402
blocked (B = 32), row RHS, row output       : 0.307752 ± 0.00097284

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, column RHS, row output               : 4.14571 ± 0.0134253
naive, row RHS, column output               : 0.431341 ± 0.00452887
naive, row RHS, row output                  : 0.480593 ± 0.0109085
best column RHS, column output              : 0.321941 ± 0.0013211
blocked (B = 4), column RHS, row output     : 0.759357 ± 0.0062807
blocked (B = 8), column RHS, row output     : 0.565355 ± 0.00508378
blocked (B = 16), column RHS, row output    : 0.390936 ± 0.00245866
blocked (B = 32), column RHS, row output    : 0.351794 ± 0.00236939
blocked (B = 4), row RHS, column output     : 0.339523 ± 0.00264102
blocked (B = 8), row RHS, column output     : 0.299875 ± 0.00481955
blocked (B = 16), row RHS, column output    : 0.290624 ± 0.00229921
blocked (B = 32), row RHS, column output    : 0.32741 ± 0.00267274
blocked (B = 4), row RHS, row output        : 0.338909 ± 0.00254423
blocked (B = 8), row RHS, row output        : 0.30935 ± 0.00764014
blocked (B = 16), row RHS, row output       : 0.303492 ± 0.00145859
blocked (B = 32), row RHS, row output       : 0.329232 ± 0.00157817

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, column RHS, row output               : 7.68748 ± 0.00671896
naive, row RHS, column output               : 0.888356 ± 0.0267381
naive, row RHS, row output                  : 0.860943 ± 0.00162817
best column RHS, column output              : 0.322405 ± 0.0013068
blocked (B = 4), column RHS, row output     : 0.856017 ± 0.00165751
blocked (B = 8), column RHS, row output     : 0.637772 ± 0.00357504
blocked (B = 16), column RHS, row output    : 0.447544 ± 0.00220057
blocked (B = 32), column RHS, row output    : 0.376013 ± 0.00176963
blocked (B = 4), row RHS, column output     : 0.408 ± 0.00183692
blocked (B = 8), row RHS, column output     : 0.332458 ± 0.00109808
blocked (B = 16), row RHS, column output    : 0.303773 ± 0.00119771
blocked (B = 32), row RHS, column output    : 0.322876 ± 0.00256126
blocked (B = 4), row RHS, row output        : 0.423296 ± 0.00543553
blocked (B = 8), row RHS, row output        : 0.340769 ± 0.00388921
blocked (B = 16), row RHS, row output       : 0.319736 ± 0.000605217
blocked (B = 32), row RHS, row output       : 0.319896 ± 0.000997355

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, column RHS, row output               : 2.61096 ± 0.0139305
naive, row RHS, column output               : 0.879282 ± 0.014015
naive, row RHS, row output                  : 0.858827 ± 0.00549373
best column RHS, column output              : 0.332325 ± 0.000945985
blocked (B = 4), column RHS, row output     : 0.772225 ± 0.00279831
blocked (B = 8), column RHS, row output     : 0.538847 ± 0.00169485
blocked (B = 16), column RHS, row output    : 0.386345 ± 0.00214072
blocked (B = 32), column RHS, row output    : 0.344609 ± 0.00125557
blocked (B = 4), row RHS, column output     : 0.426087 ± 0.00186087
blocked (B = 8), row RHS, column output     : 0.343825 ± 0.00434729
blocked (B = 16), row RHS, column output    : 0.314941 ± 0.00113642
blocked (B = 32), row RHS, column output    : 0.311787 ± 0.00124921
blocked (B = 4), row RHS, row output        : 0.414795 ± 0.00678709
blocked (B = 8), row RHS, row output        : 0.330458 ± 0.00200177
blocked (B = 16), row RHS, row output       : 0.308533 ± 0.00166729
blocked (B = 32), row RHS, row output       : 0.327117 ± 0.000907515

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, column RHS, row output               : 1.0293 ± 0.00213184
naive, row RHS, column output               : 0.268409 ± 0.000595341
naive, row RHS, row output                  : 0.366957 ± 0.00085621
best column RHS, column output              : 0.315232 ± 0.000663079
blocked (B = 4), column RHS, row output     : 0.631388 ± 0.00360067
blocked (B = 8), column RHS, row output     : 0.473318 ± 0.000891176
blocked (B = 16), column RHS, row output    : 0.351381 ± 0.00286577
blocked (B = 32), column RHS, row output    : 0.314292 ± 0.000671207
blocked (B = 4), row RHS, column output     : 0.289845 ± 0.00102944
blocked (B = 8), row RHS, column output     : 0.267203 ± 0.00079427
blocked (B = 16), row RHS, column output    : 0.278625 ± 0.00103747
blocked (B = 32), row RHS, column output    : 0.300679 ± 0.00134616
blocked (B = 4), row RHS, row output        : 0.287169 ± 0.000612963
blocked (B = 8), row RHS, row output        : 0.271991 ± 0.000690604
blocked (B = 16), row RHS, row output       : 0.286927 ± 0.000582179
blocked (B = 32), row RHS, row output       : 0.317383 ± 0.00151837

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, column RHS, row output               : 1.52441 ± 0.00301827
naive, row RHS, column output               : 0.436745 ± 0.00119376
naive, row RHS, row output                  : 0.470088 ± 0.00124914
best column RHS, column output              : 0.385662 ± 0.00064907
blocked (B = 4), column RHS, row output     : 0.708922 ± 0.000580805
blocked (B = 8), column RHS, row output     : 0.521583 ± 0.00101314
blocked (B = 16), column RHS, row output    : 0.413781 ± 0.00090715
blocked (B = 32), column RHS, row output    : 0.363602 ± 0.000752951
blocked (B = 4), row RHS, column output     : 0.337529 ± 0.000755736
blocked (B = 8), row RHS, column output     : 0.301175 ± 0.000377632
blocked (B = 16), row RHS, column output    : 0.300012 ± 0.00164578
blocked (B = 32), row RHS, column output    : 0.3195 ± 0.000315221
blocked (B = 4), row RHS, row output        : 0.346363 ± 0.00110421
blocked (B = 8), row RHS, row output        : 0.301577 ± 0.000701138
blocked (B = 16), row RHS, row output       : 0.295439 ± 0.00107547
blocked (B = 32), row RHS, row output       : 0.331982 ± 0.000344278

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, column RHS, row output               : 0.931103 ± 0.00529972
naive, row RHS, column output               : 0.278396 ± 0.00104971
naive, row RHS, row output                  : 0.366701 ± 0.000997275
best column RHS, column output              : 0.340848 ± 0.00507341
blocked (B = 4), column RHS, row output     : 0.569393 ± 0.00193775
blocked (B = 8), column RHS, row output     : 0.432194 ± 0.00158065
blocked (B = 16), column RHS, row output    : 0.358851 ± 0.00175108
blocked (B = 32), column RHS, row output    : 0.330127 ± 0.00285898
blocked (B = 4), row RHS, column output     : 0.287997 ± 0.00143572
blocked (B = 8), row RHS, column output     : 0.2708 ± 0.00116682
blocked (B = 16), row RHS, column output    : 0.282293 ± 0.00169818
blocked (B = 32), row RHS, column output    : 0.311181 ± 0.00301649
blocked (B = 4), row RHS, row output        : 0.288694 ± 0.000475708
blocked (B = 8), row RHS, row output        : 0.27274 ± 0.0023936
blocked (B = 16), row RHS, row output       : 0.283877 ± 0.00136947
blocked (B = 32), row RHS, row output       : 0.306756 ± 0.00204456
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive, column RHS, row output               : 1.01923 ± 0.0103671
naive, row RHS, column output               : 0.155242 ± 0.00272175
naive, row RHS, row output                  : 0.159559 ± 0.00545392
best column RHS, column output              : 0.153395 ± 0.00677771
blocked (B = 4), column RHS, row output     : 0.451281 ± 0.00211305
blocked (B = 8), column RHS, row output     : 0.34439 ± 0.00284314
blocked (B = 16), column RHS, row output    : 0.273424 ± 0.001565
blocked (B = 32), column RHS, row output    : 0.223396 ± 0.007564
blocked (B = 4), row RHS, column output     : 0.154397 ± 0.00364278
blocked (B = 8), row RHS, column output     : 0.148984 ± 0.00227346
blocked (B = 16), row RHS, column output    : 0.15487 ± 0.00142762
blocked (B = 32), row RHS, column output    : 0.174176 ± 0.00136586
blocked (B = 4), row RHS, row output        : 0.210537 ± 0.00380007
blocked (B = 8), row RHS, row output        : 0.20676 ± 0.00197997
blocked (B = 16), row RHS, row output       : 0.211188 ± 0.00274311
blocked (B = 32), row RHS, row output       : 0.224936 ± 0.000619632

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, column RHS, row output               : 2.37336 ± 0.00786314
naive, row RHS, column output               : 0.173458 ± 0.00532682
naive, row RHS, row output                  : 0.181423 ± 0.00220345
best column RHS, column output              : 0.147974 ± 0.00127322
blocked (B = 4), column RHS, row output     : 0.509329 ± 0.00134159
blocked (B = 8), column RHS, row output     : 0.358908 ± 0.00130145
blocked (B = 16), column RHS, row output    : 0.284703 ± 0.00165691
blocked (B = 32), column RHS, row output    : 0.21862 ± 0.00148582
blocked (B = 4), row RHS, column output     : 0.154676 ± 0.00210583
blocked (B = 8), row RHS, column output     : 0.153215 ± 0.00264672
blocked (B = 16), row RHS, column output    : 0.157413 ± 0.0017911
blocked (B = 32), row RHS, column output    : 0.173958 ± 0.000574441
blocked (B = 4), row RHS, row output        : 0.212042 ± 0.00151317
blocked (B = 8), row RHS, row output        : 0.209863 ± 0.00118273
blocked (B = 16), row RHS, row output       : 0.219015 ± 0.000669645
blocked (B = 32), row RHS, row output       : 0.244736 ± 0.000499352

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, column RHS, row output               : 6.49386 ± 0.0546823
naive, row RHS, column output               : 0.423402 ± 0.00870096
naive, row RHS, row output                  : 0.413125 ± 0.00387442
best column RHS, column output              : 0.153281 ± 0.00460117
blocked (B = 4), column RHS, row output     : 0.599591 ± 0.0026603
blocked (B = 8), column RHS, row output     : 0.413837 ± 0.00335387
blocked (B = 16), column RHS, row output    : 0.324818 ± 0.00326552
blocked (B = 32), column RHS, row output    : 0.250876 ± 0.00211611
blocked (B = 4), row RHS, column output     : 0.193489 ± 0.00196561
blocked (B = 8), row RHS, column output     : 0.168098 ± 0.00460521
blocked (B = 16), row RHS, column output    : 0.166301 ± 0.00433224
blocked (B = 32), row RHS, column output    : 0.175542 ± 0.000368348
blocked (B = 4), row RHS, row output        : 0.257367 ± 0.00450363
blocked (B = 8), row RHS, row output        : 0.224849 ± 0.00146233
blocked (B = 16), row RHS, row output       : 0.220422 ± 0.00107788
blocked (B = 32), row RHS, row output       : 0.228481 ± 0.000853322

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, column RHS, row output               : 2.43109 ± 0.00635729
naive, row RHS, column output               : 0.415843 ± 0.00372668
naive, row RHS, row output                  : 0.41644 ± 0.0018201
best column RHS, column output              : 0.160207 ± 0.00111074
blocked (B = 4), column RHS, row output     : 0.570479 ± 0.000544906
blocked (B = 8), column RHS, row output     : 0.3881 ± 0.000740582
blocked (B = 16), column RHS, row output    : 0.296531 ± 0.000998244
blocked (B = 32), column RHS, row output    : 0.232083 ± 0.000784959
blocked (B = 4), row RHS, column output     : 0.221307 ± 0.0030802
blocked (B = 8), row RHS, column output     : 0.178671 ± 0.000853978
blocked (B = 16), row RHS, column output    : 0.170335 ± 0.000800665
blocked (B = 32), row RHS, column output    : 0.177419 ± 0.000606834
blocked (B = 4), row RHS, row output        : 0.238298 ± 0.0024962
blocked (B = 8), row RHS, row output        : 0.21784 ± 0.00114123
blocked (B = 16), row RHS, row output       : 0.218812 ± 0.00109292
blocked (B = 32), row RHS, row output       : 0.228602 ± 0.000701208

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, column RHS, row output               : 1.0874 ± 0.00279658
naive, row RHS, column output               : 0.138251 ± 0.00016472
naive, row RHS, row output                  : 0.157237 ± 0.000197539
best column RHS, column output              : 0.146043 ± 0.000316751
blocked (B = 4), column RHS, row output     : 0.450433 ± 0.00273754
blocked (B = 8), column RHS, row output     : 0.334536 ± 0.0012717
blocked (B = 16), column RHS, row output    : 0.269116 ± 0.000224701
blocked (B = 32), column RHS, row output    : 0.210928 ± 0.000372771
blocked (B = 4), row RHS, column output     : 0.144714 ± 0.00016591
blocked (B = 8), row RHS, column output     : 0.143713 ± 0.000175682
blocked (B = 16), row RHS, column output    : 0.153667 ± 0.000482533
blocked (B = 32), row RHS, column output    : 0.173829 ± 0.000351575
blocked (B = 4), row RHS, row output        : 0.20228 ± 0.000340423
blocked (B = 8), row RHS, row output        : 0.203963 ± 0.000336515
blocked (B = 16), row RHS, row output       : 0.21573 ± 0.000486684
blocked (B = 32), row RHS, row output       : 0.242531 ± 0.000320918

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 1000 x 10000
naive, column RHS, row output               : 0.94846 ± 0.0130578
naive, row RHS, column output               : 0.179819 ± 0.0026006
naive, row RHS, row output                  : 0.16442 ± 0.00289727
best column RHS, column output              : 0.189784 ± 0.00319243
blocked (B = 4), column RHS, row output     : 0.437992 ± 0.00795455
blocked (B = 8), column RHS, row output     : 0.338961 ± 0.00158601
blocked (B = 16), column RHS, row output    : 0.306006 ± 0.00756264
blocked (B = 32), column RHS, row output    : 0.255092 ± 0.00279845
blocked (B = 4), row RHS, column output     : 0.159609 ± 0.00217756
blocked (B = 8), row RHS, column output     : 0.1557 ± 0.00170166
blocked (B = 16), row RHS, column output    : 0.170279 ± 0.00289596
blocked (B = 32), row RHS, column output    : 0.190713 ± 0.000953976
blocked (B = 4), row RHS, row output        : 0.204998 ± 0.00190574
blocked (B = 8), row RHS, row output        : 0.204762 ± 0.0015315
blocked (B = 16), row RHS, row output       : 0.210517 ± 0.00244249
blocked (B = 32), row RHS, row output       : 0.225615 ± 0.000499437

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 10000 x 1000
naive, column RHS, row output               : 0.868024 ± 0.000466371
naive, row RHS, column output               : 0.155736 ± 0.00016953
naive, row RHS, row output                  : 0.13456 ± 0.00011764
best column RHS, column output              : 0.158337 ± 0.000282541
blocked (B = 4), column RHS, row output     : 0.382892 ± 0.000114017
blocked (B = 8), column RHS, row output     : 0.313338 ± 0.000277091
blocked (B = 16), column RHS, row output    : 0.272843 ± 0.000219753
blocked (B = 32), column RHS, row output    : 0.223872 ± 0.000319385
blocked (B = 4), row RHS, column output     : 0.14493 ± 0.000111437
blocked (B = 8), row RHS, column output     : 0.145812 ± 0.000159816
blocked (B = 16), row RHS, column output    : 0.159917 ± 0.000217625
blocked (B = 32), row RHS, column output    : 0.185406 ± 0.000197754
blocked (B = 4), row RHS, row output        : 0.198089 ± 0.000100393
blocked (B = 8), row RHS, row output        : 0.197644 ± 0.00015382
blocked (B = 16), row RHS, row output       : 0.205887 ± 0.000267595
blocked (B = 32), row RHS, row output       : 0.22401 ± 0.000170356
```

## Conclusion

Blocking consistently provides some assistance across a range of matrix shapes.
$B = 16$ seems to perform the best, or nearly so.
The exception is that of single-precision floats where blocking is sometimes slightly worse than the naive approach,
probably because the data is already small enough to fit into cache. 
