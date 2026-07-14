# Sparse column-major LHS, dense matrix RHS

## Strategies

All strategies must iterate across consecutive columns of the LHS matrix in the outermost loop.
We will be loading columns on demand via the **tatami** interface, so the full matrix will not be available for random access.

### Naive, row-major RHS, row-major output

For each column $i$ of the LHS matrix, we iterate over its structural non-zeros. 
For a non-zero at LHS row $j$ and with value $x$, we perform a vector multiply-add of the $i$-th RHS row to the $j$-th output row using $x$ as the scaling factor.
Once all non-zeros have been processed, we repeat this with the next LHS column.

### Naive, row-major RHS, column-major output

For each column $i$ of the LHS matrix, we iterate over the entries of the RHS row $i$.
For an RHS value of $x$ at $(i, j)$, we perform a sparse vector multiply-add of the $i$-th LHS column to the $j$-th output column using $x$ as the scaling factor.
Once all RHS columns have been processed, we repeat this with the next LHS column.

### Naive, column-major RHS, row-major output

We use the same process as that for row-major RHS with row-major output.
The only difference is that we form each RHS row $i$ by taking the $i$-th element of each RHS column and storing it in a temporary buffer.
This buffer is re-used for the vector multiply-add for each structural non-zero in the LHS column.
The aim is to spend some time and memory to enable contiguous access and auto-vectorizable operations.

### Best, column-major RHS, column-major output 

Here, we'll just re-use the best strategy from the [multiple vectors test suite](../multiple_vectors).
This involves blocking with 16 LHS columns. 

### Blocked, row-major RHS, row-major output

For each LHS column $i$, we split up its structural non-zeros into "primary" blocks of length $B$.
Similarly, we split up the (conceptual) RHS columns into "secondary" blocks of length $C$.

For each combination of LHS and RHS blocks, we loop over the structural non-zeros in the primary LHS block.
For a LHS non-zero at row $j$ and value $x$, we perform a vector multiply-add of the secondary block of RHS row $i$ to the corresponding part of the $j$-th output row,
using $x$ as the scaling factor.
We repeat this with the next secondary block until all RHS columns are processed;
and then the next primary block until all non-zeros in LHS column $i$ are traversed;
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

The aim is to keep the most of the output column $h$ in cache so that it can be easily re-used with multiple sparse LHS columns.

### Blocked, column-major RHS, row-major output

We use the same strategy as done for the row-major RHS and row-major output,
except that we store the $i$-th element of each RHS column in a temporary buffer for easy access during the vector multiply-add operations.

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
naive, row-major RHS, row-major output                 : 0.0506428 ± 0.00191627
naive, row-major RHS, column-major output              : 0.245216 ± 0.00530372
naive, column-major RHS, row-major output              : 0.0552567 ± 0.00224573
best, column-major RHS, column-major output            : 0.0896481 ± 0.00372179
blocked (B = 4), row-major RHS, row-major output       : 0.054644 ± 0.00307724
blocked (B = 8), row-major RHS, row-major output       : 0.0500918 ± 0.00258048
blocked (B = 16), row-major RHS, row-major output      : 0.0517701 ± 0.0034937
blocked (B = 32), row-major RHS, row-major output      : 0.0504843 ± 0.00311379
blocked (B = 4), row-major RHS, column-major output    : 0.152639 ± 0.00727195
blocked (B = 8), row-major RHS, column-major output    : 0.126757 ± 0.00464603
blocked (B = 16), row-major RHS, column-major output   : 0.108615 ± 0.0026288
blocked (B = 32), row-major RHS, column-major output   : 0.102496 ± 0.00230648
blocked (B = 4), column-major RHS, row-major output    : 0.0580745 ± 0.00370901
blocked (B = 8), column-major RHS, row-major output    : 0.0571572 ± 0.0028089
blocked (B = 16), column-major RHS, row-major output   : 0.0566407 ± 0.00369678
blocked (B = 32), column-major RHS, row-major output   : 0.0562941 ± 0.00294327

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row-major RHS, row-major output                 : 0.0371893 ± 0.000108211
naive, row-major RHS, column-major output              : 0.122387 ± 0.000300592
naive, column-major RHS, row-major output              : 0.0392773 ± 0.000143532
best, column-major RHS, column-major output            : 0.0627398 ± 8.45881e-05
blocked (B = 4), row-major RHS, row-major output       : 0.0394475 ± 9.21327e-05
blocked (B = 8), row-major RHS, row-major output       : 0.0385087 ± 9.68279e-05
blocked (B = 16), row-major RHS, row-major output      : 0.0382834 ± 8.60152e-05
blocked (B = 32), row-major RHS, row-major output      : 0.0387048 ± 0.000116949
blocked (B = 4), row-major RHS, column-major output    : 0.0941797 ± 0.000193769
blocked (B = 8), row-major RHS, column-major output    : 0.0871504 ± 0.000224899
blocked (B = 16), row-major RHS, column-major output   : 0.0833332 ± 0.000298124
blocked (B = 32), row-major RHS, column-major output   : 0.0819619 ± 0.000221423
blocked (B = 4), column-major RHS, row-major output    : 0.0422455 ± 0.000107153
blocked (B = 8), column-major RHS, row-major output    : 0.0414547 ± 8.42591e-05
blocked (B = 16), column-major RHS, row-major output   : 0.0425764 ± 0.000128806
blocked (B = 32), column-major RHS, row-major output   : 0.0437356 ± 0.000207876

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row-major RHS, row-major output                 : 0.111104 ± 0.00244494
naive, row-major RHS, column-major output              : 0.753135 ± 0.0160274
naive, column-major RHS, row-major output              : 0.112887 ± 0.00190844
best, column-major RHS, column-major output            : 0.13291 ± 0.00229232
blocked (B = 4), row-major RHS, row-major output       : 0.115772 ± 0.00335328
blocked (B = 8), row-major RHS, row-major output       : 0.115095 ± 0.0033124
blocked (B = 16), row-major RHS, row-major output      : 0.115622 ± 0.00301246
blocked (B = 32), row-major RHS, row-major output      : 0.117827 ± 0.00302002
blocked (B = 4), row-major RHS, column-major output    : 0.299396 ± 0.00269783
blocked (B = 8), row-major RHS, column-major output    : 0.202627 ± 0.00324228
blocked (B = 16), row-major RHS, column-major output   : 0.152886 ± 0.00593279
blocked (B = 32), row-major RHS, column-major output   : 0.121976 ± 0.00167922
blocked (B = 4), column-major RHS, row-major output    : 0.114462 ± 0.00347512
blocked (B = 8), column-major RHS, row-major output    : 0.117115 ± 0.00234855
blocked (B = 16), column-major RHS, row-major output   : 0.122102 ± 0.00228783
blocked (B = 32), column-major RHS, row-major output   : 0.125317 ± 0.00221356

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row-major RHS, row-major output                 : 0.0859326 ± 0.00301707
naive, row-major RHS, column-major output              : 0.26461 ± 0.00777034
naive, column-major RHS, row-major output              : 0.0877323 ± 0.00363418
best, column-major RHS, column-major output            : 0.11157 ± 0.0017395
blocked (B = 4), row-major RHS, row-major output       : 0.0859676 ± 0.00549923
blocked (B = 8), row-major RHS, row-major output       : 0.0890967 ± 0.0029214
blocked (B = 16), row-major RHS, row-major output      : 0.0854349 ± 0.002967
blocked (B = 32), row-major RHS, row-major output      : 0.0764259 ± 0.00363928
blocked (B = 4), row-major RHS, column-major output    : 0.166477 ± 0.00552266
blocked (B = 8), row-major RHS, column-major output    : 0.134853 ± 0.00354649
blocked (B = 16), row-major RHS, column-major output   : 0.114644 ± 0.00273092
blocked (B = 32), row-major RHS, column-major output   : 0.100492 ± 0.00204684
blocked (B = 4), column-major RHS, row-major output    : 0.0910484 ± 0.00317833
blocked (B = 8), column-major RHS, row-major output    : 0.0915682 ± 0.00467427
blocked (B = 16), column-major RHS, row-major output   : 0.0875484 ± 0.00424869
blocked (B = 32), column-major RHS, row-major output   : 0.084616 ± 0.00308323

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row-major RHS, row-major output                 : 0.118288 ± 0.0024561
naive, row-major RHS, column-major output              : 0.731442 ± 0.00534127
naive, column-major RHS, row-major output              : 0.122569 ± 0.00370225
best, column-major RHS, column-major output            : 0.160329 ± 0.00385127
blocked (B = 4), row-major RHS, row-major output       : 0.124784 ± 0.00409559
blocked (B = 8), row-major RHS, row-major output       : 0.12251 ± 0.00463445
blocked (B = 16), row-major RHS, row-major output      : 0.120467 ± 0.00277679
blocked (B = 32), row-major RHS, row-major output      : 0.12352 ± 0.0031433
blocked (B = 4), row-major RHS, column-major output    : 0.329781 ± 0.00454424
blocked (B = 8), row-major RHS, column-major output    : 0.217955 ± 0.00285719
blocked (B = 16), row-major RHS, column-major output   : 0.151595 ± 0.00167698
blocked (B = 32), row-major RHS, column-major output   : 0.123415 ± 0.00242886
blocked (B = 4), column-major RHS, row-major output    : 0.118548 ± 0.00501131
blocked (B = 8), column-major RHS, row-major output    : 0.13296 ± 0.00734529
blocked (B = 16), column-major RHS, row-major output   : 0.122561 ± 0.00371713
blocked (B = 32), column-major RHS, row-major output   : 0.128132 ± 0.00190888

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, row-major RHS, row-major output                 : 0.0343578 ± 0.000496936
naive, row-major RHS, column-major output              : 0.131063 ± 0.000153847
naive, column-major RHS, row-major output              : 0.0612915 ± 0.00112143
best, column-major RHS, column-major output            : 0.078938 ± 0.00127514
blocked (B = 4), row-major RHS, row-major output       : 0.037472 ± 0.000570946
blocked (B = 8), row-major RHS, row-major output       : 0.0358574 ± 0.000585805
blocked (B = 16), row-major RHS, row-major output      : 0.035663 ± 0.000591117
blocked (B = 32), row-major RHS, row-major output      : 0.0370484 ± 0.000240613
blocked (B = 4), row-major RHS, column-major output    : 0.0879087 ± 0.000722393
blocked (B = 8), row-major RHS, column-major output    : 0.0855604 ± 0.000388928
blocked (B = 16), row-major RHS, column-major output   : 0.0862122 ± 0.000736958
blocked (B = 32), row-major RHS, column-major output   : 0.0899577 ± 0.000490515
blocked (B = 4), column-major RHS, row-major output    : 0.0693713 ± 0.000718545
blocked (B = 8), column-major RHS, row-major output    : 0.0673782 ± 0.0010169
blocked (B = 16), column-major RHS, row-major output   : 0.0679769 ± 0.000849231
blocked (B = 32), column-major RHS, row-major output   : 0.0717588 ± 0.000758189

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, row-major RHS, row-major output                 : 0.0655276 ± 0.000814057
naive, row-major RHS, column-major output              : 0.296142 ± 0.00993514
naive, column-major RHS, row-major output              : 0.121389 ± 0.00554881
best, column-major RHS, column-major output            : 0.134423 ± 0.00268256
blocked (B = 4), row-major RHS, row-major output       : 0.0671243 ± 0.0037674
blocked (B = 8), row-major RHS, row-major output       : 0.0585596 ± 0.000409314
blocked (B = 16), row-major RHS, row-major output      : 0.0542371 ± 0.00101279
blocked (B = 32), row-major RHS, row-major output      : 0.0572452 ± 0.00248414
blocked (B = 4), row-major RHS, column-major output    : 0.167429 ± 0.015351
blocked (B = 8), row-major RHS, column-major output    : 0.124598 ± 0.0036812
blocked (B = 16), row-major RHS, column-major output   : 0.110167 ± 0.00326494
blocked (B = 32), row-major RHS, column-major output   : 0.103364 ± 0.00140262
blocked (B = 4), column-major RHS, row-major output    : 0.118222 ± 0.00214906
blocked (B = 8), column-major RHS, row-major output    : 0.113876 ± 0.0020014
blocked (B = 16), column-major RHS, row-major output   : 0.109581 ± 0.000462197
blocked (B = 32), column-major RHS, row-major output   : 0.119455 ± 0.00562755
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive, row-major RHS, row-major output                 : 0.0207932 ± 0.00144582
naive, row-major RHS, column-major output              : 0.111186 ± 0.00343564
naive, column-major RHS, row-major output              : 0.0277818 ± 0.00208183
best, column-major RHS, column-major output            : 0.0669164 ± 0.00132779
blocked (B = 4), row-major RHS, row-major output       : 0.022087 ± 0.00110779
blocked (B = 8), row-major RHS, row-major output       : 0.0246642 ± 0.00213764
blocked (B = 16), row-major RHS, row-major output      : 0.0214161 ± 0.00113825
blocked (B = 32), row-major RHS, row-major output      : 0.0230113 ± 0.000430264
blocked (B = 4), row-major RHS, column-major output    : 0.0886562 ± 0.00258601
blocked (B = 8), row-major RHS, column-major output    : 0.078057 ± 0.00166611
blocked (B = 16), row-major RHS, column-major output   : 0.0793955 ± 0.00234003
blocked (B = 32), row-major RHS, column-major output   : 0.0713616 ± 0.000987086
blocked (B = 4), column-major RHS, row-major output    : 0.0340994 ± 0.00228583
blocked (B = 8), column-major RHS, row-major output    : 0.0309715 ± 0.00175
blocked (B = 16), column-major RHS, row-major output   : 0.0317404 ± 0.00192145
blocked (B = 32), column-major RHS, row-major output   : 0.0309731 ± 0.00106828

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row-major RHS, row-major output                 : 0.020263 ± 0.000213527
naive, row-major RHS, column-major output              : 0.0822175 ± 0.000473845
naive, column-major RHS, row-major output              : 0.022453 ± 0.000235301
best, column-major RHS, column-major output            : 0.0589475 ± 0.000292317
blocked (B = 4), row-major RHS, row-major output       : 0.025439 ± 7.36832e-05
blocked (B = 8), row-major RHS, row-major output       : 0.0247845 ± 0.000258282
blocked (B = 16), row-major RHS, row-major output      : 0.0249107 ± 0.000170994
blocked (B = 32), row-major RHS, row-major output      : 0.0250315 ± 0.000260981
blocked (B = 4), row-major RHS, column-major output    : 0.0721551 ± 0.000458163
blocked (B = 8), row-major RHS, column-major output    : 0.0686044 ± 0.000429251
blocked (B = 16), row-major RHS, column-major output   : 0.0673038 ± 0.00025404
blocked (B = 32), row-major RHS, column-major output   : 0.065813 ± 0.000348522
blocked (B = 4), column-major RHS, row-major output    : 0.0278406 ± 0.000238878
blocked (B = 8), column-major RHS, row-major output    : 0.0276582 ± 0.000359069
blocked (B = 16), column-major RHS, row-major output   : 0.0291055 ± 0.000128314
blocked (B = 32), column-major RHS, row-major output   : 0.0312999 ± 0.000200495

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row-major RHS, row-major output                 : 0.0543056 ± 0.00232744
naive, row-major RHS, column-major output              : 0.404674 ± 0.0047218
naive, column-major RHS, row-major output              : 0.0616835 ± 0.00280921
best, column-major RHS, column-major output            : 0.086893 ± 0.00245774
blocked (B = 4), row-major RHS, row-major output       : 0.055439 ± 0.00259042
blocked (B = 8), row-major RHS, row-major output       : 0.0574482 ± 0.0024578
blocked (B = 16), row-major RHS, row-major output      : 0.0612349 ± 0.00363938
blocked (B = 32), row-major RHS, row-major output      : 0.0603289 ± 0.00217384
blocked (B = 4), row-major RHS, column-major output    : 0.150287 ± 0.00282936
blocked (B = 8), row-major RHS, column-major output    : 0.112582 ± 0.00204197
blocked (B = 16), row-major RHS, column-major output   : 0.0934552 ± 0.00230574
blocked (B = 32), row-major RHS, column-major output   : 0.0823472 ± 0.00185048
blocked (B = 4), column-major RHS, row-major output    : 0.0593996 ± 0.00262649
blocked (B = 8), column-major RHS, row-major output    : 0.0671585 ± 0.004576
blocked (B = 16), column-major RHS, row-major output   : 0.0663459 ± 0.00424975
blocked (B = 32), column-major RHS, row-major output   : 0.06665 ± 0.00259345

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row-major RHS, row-major output                 : 0.0321197 ± 0.00205971
naive, row-major RHS, column-major output              : 0.10652 ± 0.00128718
naive, column-major RHS, row-major output              : 0.0322124 ± 0.00155647
best, column-major RHS, column-major output            : 0.0729208 ± 0.00101015
blocked (B = 4), row-major RHS, row-major output       : 0.0404051 ± 0.00143723
blocked (B = 8), row-major RHS, row-major output       : 0.0401422 ± 0.00135828
blocked (B = 16), row-major RHS, row-major output      : 0.0367236 ± 0.00198503
blocked (B = 32), row-major RHS, row-major output      : 0.0314596 ± 0.00106379
blocked (B = 4), row-major RHS, column-major output    : 0.0842658 ± 0.00108877
blocked (B = 8), row-major RHS, column-major output    : 0.0788912 ± 0.00107863
blocked (B = 16), row-major RHS, column-major output   : 0.0746291 ± 0.00139911
blocked (B = 32), row-major RHS, column-major output   : 0.0720224 ± 0.0004093
blocked (B = 4), column-major RHS, row-major output    : 0.0407575 ± 0.0010402
blocked (B = 8), column-major RHS, row-major output    : 0.0385348 ± 0.00165831
blocked (B = 16), column-major RHS, row-major output   : 0.0372798 ± 0.00167836
blocked (B = 32), column-major RHS, row-major output   : 0.0341086 ± 0.000574094

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row-major RHS, row-major output                 : 0.0657691 ± 0.000580213
naive, row-major RHS, column-major output              : 0.405771 ± 0.0154172
naive, column-major RHS, row-major output              : 0.0683741 ± 0.00262391
best, column-major RHS, column-major output            : 0.0944974 ± 0.0010501
blocked (B = 4), row-major RHS, row-major output       : 0.066871 ± 0.000431579
blocked (B = 8), row-major RHS, row-major output       : 0.0682078 ± 0.00308193
blocked (B = 16), row-major RHS, row-major output      : 0.0692198 ± 0.00255085
blocked (B = 32), row-major RHS, row-major output      : 0.0702673 ± 0.00342375
blocked (B = 4), row-major RHS, column-major output    : 0.170697 ± 0.00317972
blocked (B = 8), row-major RHS, column-major output    : 0.12137 ± 0.000613375
blocked (B = 16), row-major RHS, column-major output   : 0.0974762 ± 0.00109713
blocked (B = 32), row-major RHS, column-major output   : 0.0839815 ± 0.000858015
blocked (B = 4), column-major RHS, row-major output    : 0.0668818 ± 0.00261843
blocked (B = 8), column-major RHS, row-major output    : 0.0680498 ± 0.0022165
blocked (B = 16), column-major RHS, row-major output   : 0.0703213 ± 0.00117885
blocked (B = 32), column-major RHS, row-major output   : 0.07181 ± 0.00225117

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, row-major RHS, row-major output                 : 0.0193615 ± 0.000563328
naive, row-major RHS, column-major output              : 0.0965565 ± 0.000347926
naive, column-major RHS, row-major output              : 0.0440515 ± 0.00025292
best, column-major RHS, column-major output            : 0.0689322 ± 0.000185331
blocked (B = 4), row-major RHS, row-major output       : 0.0201592 ± 0.000314401
blocked (B = 8), row-major RHS, row-major output       : 0.0193056 ± 0.000252811
blocked (B = 16), row-major RHS, row-major output      : 0.0211525 ± 0.00037886
blocked (B = 32), row-major RHS, row-major output      : 0.0229837 ± 0.000322576
blocked (B = 4), row-major RHS, column-major output    : 0.0751474 ± 0.000511412
blocked (B = 8), row-major RHS, column-major output    : 0.0724937 ± 0.000319374
blocked (B = 16), row-major RHS, column-major output   : 0.0716212 ± 0.000561058
blocked (B = 32), row-major RHS, column-major output   : 0.0730743 ± 0.000543915
blocked (B = 4), column-major RHS, row-major output    : 0.0518418 ± 0.00039927
blocked (B = 8), column-major RHS, row-major output    : 0.0523804 ± 0.000353551
blocked (B = 16), column-major RHS, row-major output   : 0.0539456 ± 0.000360892
blocked (B = 32), column-major RHS, row-major output   : 0.0560944 ± 0.0003361

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, row-major RHS, row-major output                 : 0.0259658 ± 0.00202331
naive, row-major RHS, column-major output              : 0.107563 ± 0.00232186
naive, column-major RHS, row-major output              : 0.0686498 ± 0.00196937
best, column-major RHS, column-major output            : 0.101279 ± 0.00630885
blocked (B = 4), row-major RHS, row-major output       : 0.0234907 ± 0.00100675
blocked (B = 8), row-major RHS, row-major output       : 0.022398 ± 0.00159402
blocked (B = 16), row-major RHS, row-major output      : 0.022856 ± 0.00116544
blocked (B = 32), row-major RHS, row-major output      : 0.0248837 ± 0.00139085
blocked (B = 4), row-major RHS, column-major output    : 0.0845227 ± 0.00194081
blocked (B = 8), row-major RHS, column-major output    : 0.0797735 ± 0.00246894
blocked (B = 16), row-major RHS, column-major output   : 0.0751193 ± 0.00176172
blocked (B = 32), row-major RHS, column-major output   : 0.0740239 ± 0.000911282
blocked (B = 4), column-major RHS, row-major output    : 0.073361 ± 0.00150693
blocked (B = 8), column-major RHS, row-major output    : 0.0718057 ± 0.00192456
blocked (B = 16), column-major RHS, row-major output   : 0.0766518 ± 0.00294518
blocked (B = 32), column-major RHS, row-major output   : 0.07598 ± 0.00118126
```

## Conclusion

Blocking is pretty helpful for row-major RHS with column-major output, similar to our conclusions for the column-major RHS with column-major output.
Seems like $B = 16$ is generally a good choice;
it provides almost all of the benefit without double the memory usage of $B = 32$.

For row-major output, blocking is less beneficial.
Presumably the looping overhead offsets any gains in cache performance.
