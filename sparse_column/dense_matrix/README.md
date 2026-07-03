# Sparse column-major LHS, dense matrix RHS

## Strategies

All strategies must iterate across consecutive columns of the LHS matrix in the outermost loop.
We will be loading columns on demand via the **tatami** interface, so the full matrix will not be available for random access.

### Naive, row-major RHS, row-major output

For each column $i$ of the LHS matrix, we compute the outer product with the $i$-th RHS row.
This is done by iterating over the structural non-zeros of the LHS column;
for a non-zero at row $j$ and with value $x$, we perform a vector multiply-add of the $x$-th RHS row to the $j$-th output row using $x$ as the scaling factor.
Once all non-zeros have been processed, we repeat this with the next LHS column.

### Naive, row-major RHS, column-major output

For each column $i$ of the LHS matrix, we compute the outer product with the $i$-th RHS row.
This is done by iterating over the columns of the RHS matrix;
for RHS column $j$ with value $x$ in the $i$-th RHS row, we perform a sparse vector multiply-add of the $i$-th LHS column to the $j$-th output column using $x$ as the scaling factor.
Once all RHS columns have been processed, we repeat this with the next LHS column.

### Naive, column-major RHS, row-major output

For each column $i$ of the LHS matrix, we compute the outer product with the conceptual $i$-th RHS row.
(The conceptual row is not physically present and consists of the $i$-th entry of all RHS columns.)
This is done by iterating over the structural non-zeros of the LHS column;
for a non-zero at row $j$ and with value $x$, we perform a vector multiply-add of the conceptual $x$-th RHS row to the $j$-th output row using $x$ as the scaling factor.
Once all RHS columns have been processed, we repeat this with the next LHS column.

### Best, column-major RHS, column-major output 

Here, we'll just re-use the best strategy from the [multiple vectors test suite](../multiple_vectors).
This involves blocking with 16 LHS columns. 

### Blocked, row-major RHS, row-major output

For each LHS column $i$, we split up its structural non-zeros into blocks of length $B$.
Similarly, we split up the (conceptual) RHS columns into blocks of length $C$.
For each combination of LHS and RHS blocks, we loop over the structural non-zeros in the LHS block.
For a non-zero at row $j$ and value $x$, we perform a vector multiply-add of RHS row $i$ to the $j$-th output row using $x$ as the scaling factor.
However, this vector multiply-add is only done for the $C$ columns in the RHS block.
We then proceed to the next block of $C$ RHS columns until the entire RHS matrix is traversed;
and then the next LHS block until all non-zeros in column $Qi$ are traversed;
and then we move onto the next LHS column until the entire LHS matrix is traversed.

The aim is to keep a small block of the RHS row in cache so that it can be easily re-used across multiple LHS non-zeros.
We consider various choices of $B$ with the constraint that $BC = 1024$, to ensure that (i) the RHS row is small enough to keep in cache
and (ii) trailing cache lines from each output row are also kept in cache for re-use in the next block of RHS columns.

### Blocked, row-major RHS, column-major output

We follow the recommendation in the "Blocking to cache the dense vector" section of [`general/README.md`](../../general/README.md).
First, we load a block of $B$ LHS columns.
For each block, we loop over the (conceptual) RHS columns, and for each RHS column $h$, we loop again over our block of LHS columns.
For each LHS column $j$ in our block, we perform a sparse vector multiply-add to the $h$-th output column as described above, using the RHS matrix entry $(j, h)$ as the scaling factor.
Once all RHS columns are processed, we move onto the next block of $B$ LHS columns, and so on until all LHS columns are traversed.
The aim is to keep the most of the output column $h$ in cache so that it can be easily re-used with multiple sparse vectors.

### Blocked, column-major RHS, row-major output

We use the same strategy as done for the row-major RHS and row-major output,
except that each RHS row $i$ is conceptual, i.e., it consists of the $i$-th element of each output row.
Our aim is to keep the RHS row (or specifically, the non-contiguous cache lines that constitute it) in cache for use across multiple LHS non-zeros.

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
naive, row-major RHS, row-major output                 : 0.0665722 ± 0.00174171
naive, row-major RHS, column-major output              : 0.251917 ± 0.0014731
naive, column-major RHS, row-major output              : 0.141265 ± 0.00139908
best, column-major RHS, column-major output            : 0.102374 ± 0.000661015
blocked (B = 4), row-major RHS, row-major output       : 0.0684787 ± 0.000872654
blocked (B = 8), row-major RHS, row-major output       : 0.0640682 ± 0.00118458
blocked (B = 16), row-major RHS, row-major output      : 0.06162 ± 0.000705645
blocked (B = 32), row-major RHS, row-major output      : 0.0621415 ± 0.00123139
blocked (B = 4), row-major RHS, column-major output    : 0.152726 ± 0.00106493
blocked (B = 8), row-major RHS, column-major output    : 0.121334 ± 0.000871399
blocked (B = 16), row-major RHS, column-major output   : 0.100274 ± 0.00309979
blocked (B = 32), row-major RHS, column-major output   : 0.0901418 ± 0.000446042
blocked (B = 4), column-major RHS, row-major output    : 0.110867 ± 0.000572097
blocked (B = 8), column-major RHS, row-major output    : 0.115996 ± 0.00118236
blocked (B = 16), column-major RHS, row-major output   : 0.114392 ± 0.00101517
blocked (B = 32), column-major RHS, row-major output   : 0.113582 ± 0.00103819

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row-major RHS, row-major output                 : 0.038537 ± 0.000562522
naive, row-major RHS, column-major output              : 0.122217 ± 0.00103676
naive, column-major RHS, row-major output              : 0.102562 ± 0.00178353
best, column-major RHS, column-major output            : 0.0655811 ± 0.000852174
blocked (B = 4), row-major RHS, row-major output       : 0.0411788 ± 0.000530042
blocked (B = 8), row-major RHS, row-major output       : 0.0405495 ± 0.000687589
blocked (B = 16), row-major RHS, row-major output      : 0.0412371 ± 0.000547253
blocked (B = 32), row-major RHS, row-major output      : 0.0423855 ± 0.000676837
blocked (B = 4), row-major RHS, column-major output    : 0.0837464 ± 0.000780281
blocked (B = 8), row-major RHS, column-major output    : 0.0721916 ± 0.000820167
blocked (B = 16), row-major RHS, column-major output   : 0.0650779 ± 0.000847086
blocked (B = 32), row-major RHS, column-major output   : 0.064091 ± 0.000998798
blocked (B = 4), column-major RHS, row-major output    : 0.082391 ± 0.000775688
blocked (B = 8), column-major RHS, row-major output    : 0.0812985 ± 0.000675039
blocked (B = 16), column-major RHS, row-major output   : 0.0771589 ± 0.00106997
blocked (B = 32), column-major RHS, row-major output   : 0.0734509 ± 0.000670647

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row-major RHS, row-major output                 : 0.119136 ± 0.000449875
naive, row-major RHS, column-major output              : 0.72527 ± 0.00281953
naive, column-major RHS, row-major output              : 0.263213 ± 0.00419915
best, column-major RHS, column-major output            : 0.137163 ± 0.00211002
blocked (B = 4), row-major RHS, row-major output       : 0.123747 ± 0.00225814
blocked (B = 8), row-major RHS, row-major output       : 0.12403 ± 0.00456422
blocked (B = 16), row-major RHS, row-major output      : 0.125527 ± 0.000843992
blocked (B = 32), row-major RHS, row-major output      : 0.123396 ± 0.00310221
blocked (B = 4), row-major RHS, column-major output    : 0.299994 ± 0.00352528
blocked (B = 8), row-major RHS, column-major output    : 0.193555 ± 0.00134148
blocked (B = 16), row-major RHS, column-major output   : 0.137553 ± 0.00267942
blocked (B = 32), row-major RHS, column-major output   : 0.106945 ± 0.00178885
blocked (B = 4), column-major RHS, row-major output    : 0.163817 ± 0.00380205
blocked (B = 8), column-major RHS, row-major output    : 0.166875 ± 0.00418679
blocked (B = 16), column-major RHS, row-major output   : 0.150583 ± 0.000387478
blocked (B = 32), column-major RHS, row-major output   : 0.15883 ± 0.0025921

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row-major RHS, row-major output                 : 0.0897707 ± 0.00160131
naive, row-major RHS, column-major output              : 0.255147 ± 0.00289259
naive, column-major RHS, row-major output              : 0.148652 ± 0.000440243
best, column-major RHS, column-major output            : 0.117615 ± 0.000690613
blocked (B = 4), row-major RHS, row-major output       : 0.0940967 ± 0.000343607
blocked (B = 8), row-major RHS, row-major output       : 0.0933157 ± 0.000378354
blocked (B = 16), row-major RHS, row-major output      : 0.0890798 ± 0.000538612
blocked (B = 32), row-major RHS, row-major output      : 0.084171 ± 0.00172325
blocked (B = 4), row-major RHS, column-major output    : 0.164595 ± 0.000513955
blocked (B = 8), row-major RHS, column-major output    : 0.13644 ± 0.00187478
blocked (B = 16), row-major RHS, column-major output   : 0.115746 ± 0.0028892
blocked (B = 32), row-major RHS, column-major output   : 0.105428 ± 0.000844702
blocked (B = 4), column-major RHS, row-major output    : 0.132045 ± 0.00071617
blocked (B = 8), column-major RHS, row-major output    : 0.129932 ± 0.000419508
blocked (B = 16), column-major RHS, row-major output   : 0.124612 ± 0.00112466
blocked (B = 32), column-major RHS, row-major output   : 0.115004 ± 0.00055603

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row-major RHS, row-major output                 : 0.127368 ± 0.000426811
naive, row-major RHS, column-major output              : 0.722647 ± 0.00345616
naive, column-major RHS, row-major output              : 0.164688 ± 0.00134796
best, column-major RHS, column-major output            : 0.157399 ± 0.00152117
blocked (B = 4), row-major RHS, row-major output       : 0.12741 ± 0.00151903
blocked (B = 8), row-major RHS, row-major output       : 0.127267 ± 0.00296695
blocked (B = 16), row-major RHS, row-major output      : 0.130138 ± 0.00305484
blocked (B = 32), row-major RHS, row-major output      : 0.131306 ± 0.00293893
blocked (B = 4), row-major RHS, column-major output    : 0.323783 ± 0.00175102
blocked (B = 8), row-major RHS, column-major output    : 0.218776 ± 0.000619801
blocked (B = 16), row-major RHS, column-major output   : 0.155891 ± 0.00159137
blocked (B = 32), row-major RHS, column-major output   : 0.122389 ± 0.00111401
blocked (B = 4), column-major RHS, row-major output    : 0.13834 ± 0.00365486
blocked (B = 8), column-major RHS, row-major output    : 0.141045 ± 0.00277808
blocked (B = 16), column-major RHS, row-major output   : 0.145563 ± 0.000151128
blocked (B = 32), column-major RHS, row-major output   : 0.15188 ± 0.00141311

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, row-major RHS, row-major output                 : 0.0362007 ± 0.000775784
naive, row-major RHS, column-major output              : 0.129465 ± 0.00029837
naive, column-major RHS, row-major output              : 0.184367 ± 0.000754976
best, column-major RHS, column-major output            : 0.0836971 ± 0.000688374
blocked (B = 4), row-major RHS, row-major output       : 0.040858 ± 0.000714785
blocked (B = 8), row-major RHS, row-major output       : 0.0388966 ± 0.000602508
blocked (B = 16), row-major RHS, row-major output      : 0.0388599 ± 0.000573924
blocked (B = 32), row-major RHS, row-major output      : 0.0409941 ± 0.000657415
blocked (B = 4), row-major RHS, column-major output    : 0.0807026 ± 0.00038861
blocked (B = 8), row-major RHS, column-major output    : 0.0745221 ± 0.000522646
blocked (B = 16), row-major RHS, column-major output   : 0.0713979 ± 0.000836081
blocked (B = 32), row-major RHS, column-major output   : 0.0782839 ± 0.000724626
blocked (B = 4), column-major RHS, row-major output    : 0.116487 ± 0.000845947
blocked (B = 8), column-major RHS, row-major output    : 0.108211 ± 0.000613813
blocked (B = 16), column-major RHS, row-major output   : 0.0945683 ± 0.000682945
blocked (B = 32), column-major RHS, row-major output   : 0.0923403 ± 0.000803237

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, row-major RHS, row-major output                 : 0.0812748 ± 0.00173542
naive, row-major RHS, column-major output              : 0.28183 ± 0.00262408
naive, column-major RHS, row-major output              : 0.421143 ± 0.000788594
best, column-major RHS, column-major output            : 0.140511 ± 0.00210359
blocked (B = 4), row-major RHS, row-major output       : 0.0767819 ± 0.00101214
blocked (B = 8), row-major RHS, row-major output       : 0.0697371 ± 0.000999435
blocked (B = 16), row-major RHS, row-major output      : 0.0653986 ± 0.00158839
blocked (B = 32), row-major RHS, row-major output      : 0.0628018 ± 0.00104037
blocked (B = 4), row-major RHS, column-major output    : 0.159373 ± 0.00121773
blocked (B = 8), row-major RHS, column-major output    : 0.122837 ± 0.00210887
blocked (B = 16), row-major RHS, column-major output   : 0.10253 ± 0.0026536
blocked (B = 32), row-major RHS, column-major output   : 0.0947954 ± 0.000382723
blocked (B = 4), column-major RHS, row-major output    : 0.210662 ± 0.000391944
blocked (B = 8), column-major RHS, row-major output    : 0.184141 ± 0.00165337
blocked (B = 16), column-major RHS, row-major output   : 0.173173 ± 0.000904518
blocked (B = 32), column-major RHS, row-major output   : 0.175525 ± 0.000625881
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive, row-major RHS, row-major output                 : 0.0215769 ± 0.00186612
naive, row-major RHS, column-major output              : 0.112056 ± 0.000998005
naive, column-major RHS, row-major output              : 0.120179 ± 0.000583293
best, column-major RHS, column-major output            : 0.0729133 ± 0.00141703
blocked (B = 4), row-major RHS, row-major output       : 0.0196503 ± 0.000255432
blocked (B = 8), row-major RHS, row-major output       : 0.0214589 ± 0.00125603
blocked (B = 16), row-major RHS, row-major output      : 0.0227308 ± 0.00116625
blocked (B = 32), row-major RHS, row-major output      : 0.0248266 ± 0.00124296
blocked (B = 4), row-major RHS, column-major output    : 0.103357 ± 0.000430448
blocked (B = 8), row-major RHS, column-major output    : 0.100083 ± 0.00133865
blocked (B = 16), row-major RHS, column-major output   : 0.0902359 ± 0.00129251
blocked (B = 32), row-major RHS, column-major output   : 0.0905775 ± 0.000522998
blocked (B = 4), column-major RHS, row-major output    : 0.0842647 ± 0.0010063
blocked (B = 8), column-major RHS, row-major output    : 0.0842551 ± 0.000282744
blocked (B = 16), column-major RHS, row-major output   : 0.0807421 ± 0.00170373
blocked (B = 32), column-major RHS, row-major output   : 0.0746234 ± 0.000543759

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row-major RHS, row-major output                 : 0.020243 ± 0.00033356
naive, row-major RHS, column-major output              : 0.0847487 ± 0.000268609
naive, column-major RHS, row-major output              : 0.0933062 ± 0.000188799
best, column-major RHS, column-major output            : 0.0593022 ± 0.000404018
blocked (B = 4), row-major RHS, row-major output       : 0.0254439 ± 0.000239347
blocked (B = 8), row-major RHS, row-major output       : 0.0249559 ± 0.000298569
blocked (B = 16), row-major RHS, row-major output      : 0.0257285 ± 0.000291033
blocked (B = 32), row-major RHS, row-major output      : 0.0268436 ± 0.000507107
blocked (B = 4), row-major RHS, column-major output    : 0.0844003 ± 0.000170172
blocked (B = 8), row-major RHS, column-major output    : 0.0837036 ± 0.000457192
blocked (B = 16), row-major RHS, column-major output   : 0.0822149 ± 0.000527093
blocked (B = 32), row-major RHS, column-major output   : 0.0824488 ± 0.000326275
blocked (B = 4), column-major RHS, row-major output    : 0.0753752 ± 0.00017191
blocked (B = 8), column-major RHS, row-major output    : 0.0738113 ± 0.000291808
blocked (B = 16), column-major RHS, row-major output   : 0.0658937 ± 0.000145309
blocked (B = 32), column-major RHS, row-major output   : 0.0665613 ± 0.00025297

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row-major RHS, row-major output                 : 0.0704136 ± 0.000928505
naive, row-major RHS, column-major output              : 0.400843 ± 0.00310142
naive, column-major RHS, row-major output              : 0.221434 ± 0.000731723
best, column-major RHS, column-major output            : 0.0929543 ± 0.00155121
blocked (B = 4), row-major RHS, row-major output       : 0.069927 ± 0.00130478
blocked (B = 8), row-major RHS, row-major output       : 0.0692406 ± 0.0025243
blocked (B = 16), row-major RHS, row-major output      : 0.0709356 ± 0.00297036
blocked (B = 32), row-major RHS, row-major output      : 0.0715554 ± 0.0023344
blocked (B = 4), row-major RHS, column-major output    : 0.170485 ± 0.000564246
blocked (B = 8), row-major RHS, column-major output    : 0.133977 ± 0.000457391
blocked (B = 16), row-major RHS, column-major output   : 0.110824 ± 0.00144799
blocked (B = 32), row-major RHS, column-major output   : 0.0987574 ± 0.000758987
blocked (B = 4), column-major RHS, row-major output    : 0.120183 ± 0.00243447
blocked (B = 8), column-major RHS, row-major output    : 0.113247 ± 0.00196128
blocked (B = 16), column-major RHS, row-major output   : 0.105667 ± 0.000532014
blocked (B = 32), column-major RHS, row-major output   : 0.102157 ± 0.00103051

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row-major RHS, row-major output                 : 0.0359555 ± 0.00220563
naive, row-major RHS, column-major output              : 0.111688 ± 0.00124734
naive, column-major RHS, row-major output              : 0.113557 ± 0.00139435
best, column-major RHS, column-major output            : 0.0784485 ± 0.000882129
blocked (B = 4), row-major RHS, row-major output       : 0.0450757 ± 0.00224646
blocked (B = 8), row-major RHS, row-major output       : 0.0409924 ± 0.00206683
blocked (B = 16), row-major RHS, row-major output      : 0.0425688 ± 0.00170753
blocked (B = 32), row-major RHS, row-major output      : 0.0330475 ± 0.00137124
blocked (B = 4), row-major RHS, column-major output    : 0.101301 ± 0.00213208
blocked (B = 8), row-major RHS, column-major output    : 0.0936367 ± 0.0016181
blocked (B = 16), row-major RHS, column-major output   : 0.084192 ± 0.00160663
blocked (B = 32), row-major RHS, column-major output   : 0.0858286 ± 0.000728645
blocked (B = 4), column-major RHS, row-major output    : 0.0979021 ± 0.00126022
blocked (B = 8), column-major RHS, row-major output    : 0.0955799 ± 0.00139373
blocked (B = 16), column-major RHS, row-major output   : 0.0856177 ± 0.00131863
blocked (B = 32), column-major RHS, row-major output   : 0.0821498 ± 0.00120224

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row-major RHS, row-major output                 : 0.0772374 ± 0.00148101
naive, row-major RHS, column-major output              : 0.398814 ± 0.00322426
naive, column-major RHS, row-major output              : 0.13529 ± 0.000246512
best, column-major RHS, column-major output            : 0.100248 ± 0.00137225
blocked (B = 4), row-major RHS, row-major output       : 0.0798703 ± 0.000227116
blocked (B = 8), row-major RHS, row-major output       : 0.078175 ± 0.0024292
blocked (B = 16), row-major RHS, row-major output      : 0.0795119 ± 0.00234717
blocked (B = 32), row-major RHS, row-major output      : 0.0810603 ± 0.00262306
blocked (B = 4), row-major RHS, column-major output    : 0.183901 ± 0.000634923
blocked (B = 8), row-major RHS, column-major output    : 0.138196 ± 0.000349493
blocked (B = 16), row-major RHS, column-major output   : 0.111584 ± 0.00138816
blocked (B = 32), row-major RHS, column-major output   : 0.0976101 ± 0.00071154
blocked (B = 4), column-major RHS, row-major output    : 0.104086 ± 0.00250876
blocked (B = 8), column-major RHS, row-major output    : 0.101897 ± 0.00223566
blocked (B = 16), column-major RHS, row-major output   : 0.101881 ± 0.000176608
blocked (B = 32), column-major RHS, row-major output   : 0.101414 ± 0.000285217

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, row-major RHS, row-major output                 : 0.0185238 ± 0.000573356
naive, row-major RHS, column-major output              : 0.0906247 ± 0.000186556
naive, column-major RHS, row-major output              : 0.222626 ± 0.000364521
best, column-major RHS, column-major output            : 0.0701646 ± 0.000374643
blocked (B = 4), row-major RHS, row-major output       : 0.0197089 ± 0.000412994
blocked (B = 8), row-major RHS, row-major output       : 0.0197946 ± 0.000370812
blocked (B = 16), row-major RHS, row-major output      : 0.0216127 ± 0.000561171
blocked (B = 32), row-major RHS, row-major output      : 0.0245554 ± 0.000260438
blocked (B = 4), row-major RHS, column-major output    : 0.0858752 ± 0.000551694
blocked (B = 8), row-major RHS, column-major output    : 0.0854996 ± 0.00042135
blocked (B = 16), row-major RHS, column-major output   : 0.0841783 ± 0.000625899
blocked (B = 32), row-major RHS, column-major output   : 0.0869532 ± 0.000438749
blocked (B = 4), column-major RHS, row-major output    : 0.120477 ± 0.000430419
blocked (B = 8), column-major RHS, row-major output    : 0.103436 ± 0.000311699
blocked (B = 16), column-major RHS, row-major output   : 0.0879551 ± 0.000393947
blocked (B = 32), column-major RHS, row-major output   : 0.0891583 ± 0.000449989

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, row-major RHS, row-major output                 : 0.0269661 ± 0.00151931
naive, row-major RHS, column-major output              : 0.120543 ± 0.000990463
naive, column-major RHS, row-major output              : 0.364473 ± 0.000629537
best, column-major RHS, column-major output            : 0.105449 ± 0.00144336
blocked (B = 4), row-major RHS, row-major output       : 0.024169 ± 0.001063
blocked (B = 8), row-major RHS, row-major output       : 0.023496 ± 0.00104979
blocked (B = 16), row-major RHS, row-major output      : 0.0246885 ± 0.00131279
blocked (B = 32), row-major RHS, row-major output      : 0.0265542 ± 0.00109419
blocked (B = 4), row-major RHS, column-major output    : 0.104353 ± 0.000935941
blocked (B = 8), row-major RHS, column-major output    : 0.0974812 ± 0.00123738
blocked (B = 16), row-major RHS, column-major output   : 0.0913394 ± 0.00153525
blocked (B = 32), row-major RHS, column-major output   : 0.091404 ± 0.000392013
blocked (B = 4), column-major RHS, row-major output    : 0.170471 ± 0.000726524
blocked (B = 8), column-major RHS, row-major output    : 0.143395 ± 0.000687753
blocked (B = 16), column-major RHS, row-major output   : 0.120789 ± 0.000869485
blocked (B = 32), column-major RHS, row-major output   : 0.120867 ± 0.000916599
```

## Conclusion

Blocking is pretty helpful for row-major RHS with column-major output, similar to our conclusions for the column-major RHS with column-major output.
Seems like $B = 16$ is generally a good choice;
it provides almost all of the benefit without double the memory usage of $B = 32$.

Blocking is also pretty helpful for column-major RHS with row-major output.
Presumably it mitigates the penalty of the highly non-contiguous accesses. 

For row-major RHS with row-major output, blocking is less beneficial.
Presumably the looping overhead offsets any gains in cache performance.

