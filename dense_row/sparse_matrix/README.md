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

If the output is column-major, the process is much the same except that the output row is conceptual.
That is, a conceptual output row $i$ consists of the $i$-th values of all output columns.
We do not use a temporary buffer as the transfer of data from the temporary buffer to the output columns is a dense operation;
this may be more expensive than the cachen-unfriendly accesses to multiple output columns.

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

### Blocking row-major RHS, row-major output

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
naive, column RHS, column output                  : 0.0878962 ± 0.000324639
naive, column RHS, row output                     : 0.0881572 ± 0.000810743
naive, row RHS, column output                     : 0.153606 ± 0.00290407
naive, row RHS, row output                        : 0.065012 ± 0.000358812
2 accumulators, column RHS, column output         : 0.0596714 ± 0.000650157
2 accumulators, column RHS, row output            : 0.0571463 ± 0.000520297
4 accumulators, column RHS, column output         : 0.0486166 ± 0.000556656
4 accumulators, column RHS, row output            : 0.0464251 ± 0.000205512
8 accumulators, column RHS, column output         : 0.0528684 ± 0.000518945
8 accumulators, column RHS, row output            : 0.0519468 ± 0.00049468
blocked (B = 2), column RHS, column output        : 0.0473204 ± 0.00104315
blocked (B = 4), column RHS, column output        : 0.0455085 ± 0.000275765
blocked (B = 8), column RHS, column output        : 0.0523257 ± 0.000651195
blocked (B = 16), column RHS, column output       : 0.0577133 ± 0.00124297
blocked (B = 2), column RHS, row output           : 0.0456382 ± 6.28293e-05
blocked (B = 4), column RHS, row output           : 0.046021 ± 0.000851325
blocked (B = 8), column RHS, row output           : 0.0526119 ± 0.00035841
blocked (B = 16), column RHS, row output          : 0.0571435 ± 0.000553862
blocked (B = 4), row RHS, column output           : 0.0834659 ± 0.0025922
blocked (B = 8), row RHS, column output           : 0.0617852 ± 0.000389524
blocked (B = 16), row RHS, column output          : 0.0537889 ± 0.000751421
blocked (B = 32), row RHS, column output          : 0.0468745 ± 7.49321e-05
blocked (B = 2), row RHS, row output              : 0.0688432 ± 0.00206754
blocked (B = 4), row RHS, row output              : 0.0615878 ± 0.00103339
blocked (B = 8), row RHS, row output              : 0.063838 ± 0.000824228
blocked (B = 16), row RHS, row output             : 0.0730145 ± 0.00147694

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, column RHS, column output                  : 0.0948069 ± 0.00204315
naive, column RHS, row output                     : 0.0927566 ± 0.0012289
naive, row RHS, column output                     : 0.14566 ± 0.000846019
naive, row RHS, row output                        : 0.102035 ± 0.000534477
2 accumulators, column RHS, column output         : 0.0612898 ± 0.000463529
2 accumulators, column RHS, row output            : 0.0595399 ± 0.000487048
4 accumulators, column RHS, column output         : 0.048603 ± 0.000644526
4 accumulators, column RHS, row output            : 0.0471952 ± 0.000776097
8 accumulators, column RHS, column output         : 0.0544294 ± 0.00115983
8 accumulators, column RHS, row output            : 0.0543957 ± 0.00115063
blocked (B = 2), column RHS, column output        : 0.0479996 ± 0.000677928
blocked (B = 4), column RHS, column output        : 0.0510733 ± 0.000752213
blocked (B = 8), column RHS, column output        : 0.0577385 ± 0.000428673
blocked (B = 16), column RHS, column output       : 0.0606805 ± 0.000665472
blocked (B = 2), column RHS, row output           : 0.0480033 ± 0.000535116
blocked (B = 4), column RHS, row output           : 0.0519693 ± 0.00112301
blocked (B = 8), column RHS, row output           : 0.0589256 ± 0.000730411
blocked (B = 16), column RHS, row output          : 0.0626283 ± 0.000945612
blocked (B = 4), row RHS, column output           : 0.0930699 ± 0.000579988
blocked (B = 8), row RHS, column output           : 0.0703955 ± 0.000661792
blocked (B = 16), row RHS, column output          : 0.0588253 ± 0.000670634
blocked (B = 32), row RHS, column output          : 0.0514689 ± 0.000749348
blocked (B = 2), row RHS, row output              : 0.0838474 ± 0.000608089
blocked (B = 4), row RHS, row output              : 0.0796109 ± 0.00134379
blocked (B = 8), row RHS, row output              : 0.0716066 ± 0.000632828
blocked (B = 16), row RHS, row output             : 0.0749021 ± 0.000908967

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, column RHS, column output                  : 0.109946 ± 0.000735865
naive, column RHS, row output                     : 0.0875347 ± 0.000611004
naive, row RHS, column output                     : 0.177027 ± 0.00161501
naive, row RHS, row output                        : 0.0664627 ± 0.000709139
2 accumulators, column RHS, column output         : 0.0970532 ± 0.000982323
2 accumulators, column RHS, row output            : 0.079504 ± 0.000660645
4 accumulators, column RHS, column output         : 0.10199 ± 0.000823762
4 accumulators, column RHS, row output            : 0.082061 ± 0.000702401
8 accumulators, column RHS, column output         : 0.121531 ± 0.00183817
8 accumulators, column RHS, row output            : 0.105179 ± 0.000721412
blocked (B = 2), column RHS, column output        : 0.096622 ± 0.000929839
blocked (B = 4), column RHS, column output        : 0.0888532 ± 0.000758845
blocked (B = 8), column RHS, column output        : 0.0823445 ± 0.000698358
blocked (B = 16), column RHS, column output       : 0.0782719 ± 0.00123108
blocked (B = 2), column RHS, row output           : 0.0904822 ± 0.00102316
blocked (B = 4), column RHS, row output           : 0.0849091 ± 0.000648226
blocked (B = 8), column RHS, row output           : 0.0820203 ± 0.0005767
blocked (B = 16), column RHS, row output          : 0.0780949 ± 0.000479895
blocked (B = 4), row RHS, column output           : 0.0889112 ± 0.000959019
blocked (B = 8), row RHS, column output           : 0.0694219 ± 0.000775445
blocked (B = 16), row RHS, column output          : 0.0597469 ± 0.000602835
blocked (B = 32), row RHS, column output          : 0.0544123 ± 0.000805632
blocked (B = 2), row RHS, row output              : 0.0686331 ± 0.000785255
blocked (B = 4), row RHS, row output              : 0.0698478 ± 0.000865177
blocked (B = 8), row RHS, row output              : 0.0731732 ± 0.00082704
blocked (B = 16), row RHS, row output             : 0.0759668 ± 0.000618334

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, column RHS, column output                  : 0.120722 ± 0.00196983
naive, column RHS, row output                     : 0.0996909 ± 0.000981552
naive, row RHS, column output                     : 0.7957 ± 0.0132383
naive, row RHS, row output                        : 0.0858552 ± 0.00173779
2 accumulators, column RHS, column output         : 0.128142 ± 0.00229829
2 accumulators, column RHS, row output            : 0.113045 ± 0.000822001
4 accumulators, column RHS, column output         : 0.135153 ± 0.0020972
4 accumulators, column RHS, row output            : 0.117616 ± 0.00118846
8 accumulators, column RHS, column output         : 0.154784 ± 0.00135165
8 accumulators, column RHS, row output            : 0.147388 ± 0.000867336
blocked (B = 2), column RHS, column output        : 0.120502 ± 0.00618823
blocked (B = 4), column RHS, column output        : 0.0934729 ± 0.00120754
blocked (B = 8), column RHS, column output        : 0.0839814 ± 0.00111222
blocked (B = 16), column RHS, column output       : 0.0789731 ± 0.00290781
blocked (B = 2), column RHS, row output           : 0.112555 ± 0.000799508
blocked (B = 4), column RHS, row output           : 0.0921867 ± 0.000770479
blocked (B = 8), column RHS, row output           : 0.0842882 ± 0.000966198
blocked (B = 16), column RHS, row output          : 0.0763672 ± 0.000443098
blocked (B = 4), row RHS, column output           : 0.200511 ± 0.00822637
blocked (B = 8), row RHS, column output           : 0.129857 ± 0.00848038
blocked (B = 16), row RHS, column output          : 0.1007 ± 0.00487358
blocked (B = 32), row RHS, column output          : 0.101715 ± 0.0024552
blocked (B = 2), row RHS, row output              : 0.0837698 ± 0.00160519
blocked (B = 4), row RHS, row output              : 0.111519 ± 0.00133888
blocked (B = 8), row RHS, row output              : 0.132684 ± 0.000982688
blocked (B = 16), row RHS, row output             : 0.134107 ± 0.000932192

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, column RHS, column output                  : 0.101465 ± 0.000716218
naive, column RHS, row output                     : 0.101284 ± 0.000678351
naive, row RHS, column output                     : 0.178274 ± 0.00103491
naive, row RHS, row output                        : 0.132779 ± 0.00138293
2 accumulators, column RHS, column output         : 0.0676833 ± 0.00108071
2 accumulators, column RHS, row output            : 0.0679856 ± 0.00116596
4 accumulators, column RHS, column output         : 0.0642947 ± 0.00124476
4 accumulators, column RHS, row output            : 0.0630937 ± 0.0015401
8 accumulators, column RHS, column output         : 0.0621583 ± 0.000524084
8 accumulators, column RHS, row output            : 0.0644425 ± 0.00155881
blocked (B = 2), column RHS, column output        : 0.0705926 ± 0.0019203
blocked (B = 4), column RHS, column output        : 0.0855155 ± 0.000654267
blocked (B = 8), column RHS, column output        : 0.098875 ± 0.000803154
blocked (B = 16), column RHS, column output       : 0.101587 ± 0.00236132
blocked (B = 2), column RHS, row output           : 0.0700531 ± 0.00192769
blocked (B = 4), column RHS, row output           : 0.0868342 ± 0.00150989
blocked (B = 8), column RHS, row output           : 0.0997562 ± 0.00107749
blocked (B = 16), column RHS, row output          : 0.10273 ± 0.00156294
blocked (B = 4), row RHS, column output           : 0.116441 ± 0.0031435
blocked (B = 8), row RHS, column output           : 0.0844556 ± 0.00199657
blocked (B = 16), row RHS, column output          : 0.0637723 ± 0.000695132
blocked (B = 32), row RHS, column output          : 0.0526532 ± 0.000564773
blocked (B = 2), row RHS, row output              : 0.11168 ± 0.000661394
blocked (B = 4), row RHS, row output              : 0.10056 ± 0.00112758
blocked (B = 8), row RHS, row output              : 0.0920169 ± 0.00255467
blocked (B = 16), row RHS, row output             : 0.0879888 ± 0.000641472

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, column RHS, column output                  : 0.120817 ± 0.00290571
naive, column RHS, row output                     : 0.121032 ± 0.00188736
naive, row RHS, column output                     : 0.219273 ± 0.00707955
naive, row RHS, row output                        : 0.124384 ± 0.00225408
2 accumulators, column RHS, column output         : 0.0977286 ± 0.00365746
2 accumulators, column RHS, row output            : 0.0881466 ± 0.00170752
4 accumulators, column RHS, column output         : 0.0891254 ± 0.00315387
4 accumulators, column RHS, row output            : 0.0884202 ± 0.00337249
8 accumulators, column RHS, column output         : 0.090639 ± 0.0035948
8 accumulators, column RHS, row output            : 0.0892081 ± 0.00396238
blocked (B = 2), column RHS, column output        : 0.0839918 ± 0.00278455
blocked (B = 4), column RHS, column output        : 0.0896635 ± 0.00154538
blocked (B = 8), column RHS, column output        : 0.0935343 ± 0.000855449
blocked (B = 16), column RHS, column output       : 0.0917518 ± 0.000941979
blocked (B = 2), column RHS, row output           : 0.0834112 ± 0.00238374
blocked (B = 4), column RHS, row output           : 0.0886158 ± 0.00138965
blocked (B = 8), column RHS, row output           : 0.0924072 ± 0.000781431
blocked (B = 16), column RHS, row output          : 0.0929008 ± 0.00109351
blocked (B = 4), row RHS, column output           : 0.098158 ± 0.00250134
blocked (B = 8), row RHS, column output           : 0.0684725 ± 0.000638602
blocked (B = 16), row RHS, column output          : 0.0573312 ± 0.000722654
blocked (B = 32), row RHS, column output          : 0.0509606 ± 0.000623419
blocked (B = 2), row RHS, row output              : 0.0992463 ± 0.0024367
blocked (B = 4), row RHS, row output              : 0.0796003 ± 0.00146093
blocked (B = 8), row RHS, row output              : 0.075233 ± 0.000861898
blocked (B = 16), row RHS, row output             : 0.0835632 ± 0.000959446

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, column RHS, column output                  : 0.117501 ± 0.00149812
naive, column RHS, row output                     : 0.118636 ± 0.00798042
naive, row RHS, column output                     : 0.420457 ± 0.01394
naive, row RHS, row output                        : 0.103149 ± 0.000544505
2 accumulators, column RHS, column output         : 0.103868 ± 0.00935742
2 accumulators, column RHS, row output            : 0.0835087 ± 0.000418898
4 accumulators, column RHS, column output         : 0.0834135 ± 0.000569912
4 accumulators, column RHS, row output            : 0.0836616 ± 0.00654664
8 accumulators, column RHS, column output         : 0.098503 ± 0.00987126
8 accumulators, column RHS, row output            : 0.0838651 ± 0.00174494
blocked (B = 2), column RHS, column output        : 0.0646724 ± 0.00124711
blocked (B = 4), column RHS, column output        : 0.0547292 ± 0.000686996
blocked (B = 8), column RHS, column output        : 0.0582607 ± 0.00193014
blocked (B = 16), column RHS, column output       : 0.058896 ± 0.0011779
blocked (B = 2), column RHS, row output           : 0.0620762 ± 0.000572058
blocked (B = 4), column RHS, row output           : 0.0563956 ± 0.00279614
blocked (B = 8), column RHS, row output           : 0.056569 ± 0.000202527
blocked (B = 16), column RHS, row output          : 0.0586977 ± 0.000482509
blocked (B = 4), row RHS, column output           : 0.149451 ± 0.000766648
blocked (B = 8), row RHS, column output           : 0.103363 ± 0.000736214
blocked (B = 16), row RHS, column output          : 0.0812461 ± 0.000929646
blocked (B = 32), row RHS, column output          : 0.0756875 ± 0.000534859
blocked (B = 2), row RHS, row output              : 0.104942 ± 0.00519459
blocked (B = 4), row RHS, row output              : 0.118235 ± 0.000517746
blocked (B = 8), row RHS, row output              : 0.132969 ± 0.00172349
blocked (B = 16), row RHS, row output             : 0.129434 ± 0.000435351
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive, column RHS, column output                  : 0.0874761 ± 0.000826283
naive, column RHS, row output                     : 0.0872662 ± 0.000611679
naive, row RHS, column output                     : 0.137887 ± 0.000890043
naive, row RHS, row output                        : 0.0629186 ± 0.000597571
2 accumulators, column RHS, column output         : 0.0558275 ± 9.11151e-05
2 accumulators, column RHS, row output            : 0.0568908 ± 0.000389589
4 accumulators, column RHS, column output         : 0.0430918 ± 0.000351869
4 accumulators, column RHS, row output            : 0.0434039 ± 0.000365486
8 accumulators, column RHS, column output         : 0.0430729 ± 0.000312195
8 accumulators, column RHS, row output            : 0.0426086 ± 0.00014262
blocked (B = 2), column RHS, column output        : 0.0401271 ± 5.68176e-05
blocked (B = 4), column RHS, column output        : 0.0384243 ± 0.000555571
blocked (B = 8), column RHS, column output        : 0.0394067 ± 0.000543067
blocked (B = 16), column RHS, column output       : 0.0439723 ± 0.000182956
blocked (B = 2), column RHS, row output           : 0.0411475 ± 0.000123173
blocked (B = 4), column RHS, row output           : 0.0385671 ± 0.000441023
blocked (B = 8), column RHS, row output           : 0.0397183 ± 0.000363693
blocked (B = 16), column RHS, row output          : 0.044455 ± 0.000142503
blocked (B = 4), row RHS, column output           : 0.0670589 ± 0.00128168
blocked (B = 8), row RHS, column output           : 0.0429492 ± 0.000240883
blocked (B = 16), row RHS, column output          : 0.033055 ± 0.000609506
blocked (B = 32), row RHS, column output          : 0.0271798 ± 0.000103213
blocked (B = 2), row RHS, row output              : 0.060916 ± 0.000410721
blocked (B = 4), row RHS, row output              : 0.0584108 ± 0.000701943
blocked (B = 8), row RHS, row output              : 0.0577139 ± 0.000760101
blocked (B = 16), row RHS, row output             : 0.0578677 ± 0.000316478

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, column RHS, column output                  : 0.0857596 ± 9.33081e-05
naive, column RHS, row output                     : 0.0858909 ± 0.000131817
naive, row RHS, column output                     : 0.13396 ± 0.000105479
naive, row RHS, row output                        : 0.100756 ± 3.32669e-05
2 accumulators, column RHS, column output         : 0.0548645 ± 3.30878e-05
2 accumulators, column RHS, row output            : 0.0551683 ± 0.000162397
4 accumulators, column RHS, column output         : 0.039272 ± 6.24103e-05
4 accumulators, column RHS, row output            : 0.0399384 ± 8.6505e-05
8 accumulators, column RHS, column output         : 0.0418875 ± 6.52852e-05
8 accumulators, column RHS, row output            : 0.0419878 ± 9.6938e-05
blocked (B = 2), column RHS, column output        : 0.039116 ± 0.000167554
blocked (B = 4), column RHS, column output        : 0.0388157 ± 7.64176e-05
blocked (B = 8), column RHS, column output        : 0.040493 ± 3.98156e-05
blocked (B = 16), column RHS, column output       : 0.0453839 ± 2.15657e-05
blocked (B = 2), column RHS, row output           : 0.0392978 ± 3.80115e-05
blocked (B = 4), column RHS, row output           : 0.0387615 ± 8.15636e-05
blocked (B = 8), column RHS, row output           : 0.0411562 ± 2.0559e-05
blocked (B = 16), column RHS, row output          : 0.0456293 ± 5.46038e-05
blocked (B = 4), row RHS, column output           : 0.086951 ± 5.56997e-05
blocked (B = 8), row RHS, column output           : 0.0605789 ± 3.37464e-05
blocked (B = 16), row RHS, column output          : 0.0443298 ± 8.23804e-05
blocked (B = 32), row RHS, column output          : 0.036885 ± 3.89106e-05
blocked (B = 2), row RHS, row output              : 0.0783272 ± 7.27655e-05
blocked (B = 4), row RHS, row output              : 0.0720772 ± 0.000109543
blocked (B = 8), row RHS, row output              : 0.0648631 ± 5.40128e-05
blocked (B = 16), row RHS, row output             : 0.0641543 ± 8.07201e-05


$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, column RHS, column output                  : 0.0904429 ± 0.000109953
naive, column RHS, row output                     : 0.0958459 ± 0.000346578
naive, row RHS, column output                     : 0.169234 ± 9.46053e-05
naive, row RHS, row output                        : 0.0575207 ± 6.76014e-05
2 accumulators, column RHS, column output         : 0.0882594 ± 0.000146815
2 accumulators, column RHS, row output            : 0.0805351 ± 0.000187047
4 accumulators, column RHS, column output         : 0.0870306 ± 8.35311e-05
4 accumulators, column RHS, row output            : 0.0774202 ± 0.00015193
8 accumulators, column RHS, column output         : 0.105422 ± 4.32307e-05
8 accumulators, column RHS, row output            : 0.0945545 ± 5.6317e-05
blocked (B = 2), column RHS, column output        : 0.0845216 ± 5.89951e-05
blocked (B = 4), column RHS, column output        : 0.0769251 ± 5.76316e-05
blocked (B = 8), column RHS, column output        : 0.0708743 ± 7.31354e-05
blocked (B = 16), column RHS, column output       : 0.064435 ± 9.73394e-05
blocked (B = 2), column RHS, row output           : 0.0820184 ± 6.03442e-05
blocked (B = 4), column RHS, row output           : 0.0770186 ± 0.000114863
blocked (B = 8), column RHS, row output           : 0.0725927 ± 3.36216e-05
blocked (B = 16), column RHS, row output          : 0.0653072 ± 6.88316e-05
blocked (B = 4), row RHS, column output           : 0.0781293 ± 0.000102733
blocked (B = 8), row RHS, column output           : 0.049153 ± 1.96414e-05
blocked (B = 16), row RHS, column output          : 0.0375653 ± 2.24668e-05
blocked (B = 32), row RHS, column output          : 0.0314439 ± 4.96616e-05
blocked (B = 2), row RHS, row output              : 0.0588936 ± 0.00014004
blocked (B = 4), row RHS, row output              : 0.0589017 ± 4.66834e-05
blocked (B = 8), row RHS, row output              : 0.0576305 ± 8.78069e-05
blocked (B = 16), row RHS, row output             : 0.0599425 ± 5.74156e-05

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, column RHS, column output                  : 0.103402 ± 0.000591671
naive, column RHS, row output                     : 0.109042 ± 0.00034039
naive, row RHS, column output                     : 0.680291 ± 0.00256334
naive, row RHS, row output                        : 0.0624215 ± 0.000430773
2 accumulators, column RHS, column output         : 0.121977 ± 0.000737886
2 accumulators, column RHS, row output            : 0.124202 ± 0.000932276
4 accumulators, column RHS, column output         : 0.134648 ± 0.000844813
4 accumulators, column RHS, row output            : 0.134478 ± 0.00064767
8 accumulators, column RHS, column output         : 0.140461 ± 0.000923256
8 accumulators, column RHS, row output            : 0.142307 ± 0.00202228
blocked (B = 2), column RHS, column output        : 0.115073 ± 0.00422314
blocked (B = 4), column RHS, column output        : 0.087282 ± 0.000887459
blocked (B = 8), column RHS, column output        : 0.0743906 ± 0.000383931
blocked (B = 16), column RHS, column output       : 0.0673725 ± 0.00217689
blocked (B = 2), column RHS, row output           : 0.110205 ± 0.000490405
blocked (B = 4), column RHS, row output           : 0.0853586 ± 0.000288785
blocked (B = 8), column RHS, row output           : 0.0760766 ± 0.000523352
blocked (B = 16), column RHS, row output          : 0.0664922 ± 0.000684839
blocked (B = 4), row RHS, column output           : 0.156809 ± 0.00408777
blocked (B = 8), row RHS, column output           : 0.0958329 ± 0.00380633
blocked (B = 16), row RHS, column output          : 0.060527 ± 0.0010359
blocked (B = 32), row RHS, column output          : 0.0474259 ± 0.000982873
blocked (B = 2), row RHS, row output              : 0.0628672 ± 0.00164571
blocked (B = 4), row RHS, row output              : 0.0618025 ± 0.000721071
blocked (B = 8), row RHS, row output              : 0.0796419 ± 0.000701721
blocked (B = 16), row RHS, row output             : 0.0899807 ± 0.00023954

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, column RHS, column output                  : 0.0970194 ± 0.00102181
naive, column RHS, row output                     : 0.096757 ± 0.000775361
naive, row RHS, column output                     : 0.171078 ± 0.000282767
naive, row RHS, row output                        : 0.131811 ± 0.00102691
2 accumulators, column RHS, column output         : 0.0568552 ± 0.000793957
2 accumulators, column RHS, row output            : 0.0561863 ± 0.000185234
4 accumulators, column RHS, column output         : 0.0469599 ± 7.47189e-05
4 accumulators, column RHS, row output            : 0.0469413 ± 0.000210769
8 accumulators, column RHS, column output         : 0.0473048 ± 9.86116e-05
8 accumulators, column RHS, row output            : 0.0473119 ± 0.000160209
blocked (B = 2), column RHS, column output        : 0.048645 ± 0.000660848
blocked (B = 4), column RHS, column output        : 0.0502121 ± 0.000173857
blocked (B = 8), column RHS, column output        : 0.0579122 ± 0.00100614
blocked (B = 16), column RHS, column output       : 0.0620852 ± 0.000579696
blocked (B = 2), column RHS, row output           : 0.0475623 ± 3.49144e-05
blocked (B = 4), column RHS, row output           : 0.0502915 ± 0.000227254
blocked (B = 8), column RHS, row output           : 0.0579356 ± 0.000933086
blocked (B = 16), column RHS, row output          : 0.0628348 ± 0.00101077
blocked (B = 4), row RHS, column output           : 0.102879 ± 0.000212399
blocked (B = 8), row RHS, column output           : 0.0689095 ± 0.000569149
blocked (B = 16), row RHS, column output          : 0.0489801 ± 0.0012399
blocked (B = 32), row RHS, column output          : 0.0395315 ± 0.000853605
blocked (B = 2), row RHS, row output              : 0.103728 ± 0.000207474
blocked (B = 4), row RHS, row output              : 0.0916972 ± 0.000534467
blocked (B = 8), row RHS, row output              : 0.0795484 ± 0.00019749
blocked (B = 16), row RHS, row output             : 0.0760228 ± 0.000546352

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, column RHS, column output                  : 0.10333 ± 0.000356871
naive, column RHS, row output                     : 0.104481 ± 0.00147538
naive, row RHS, column output                     : 0.176817 ± 0.000685472
naive, row RHS, row output                        : 0.0993551 ± 0.000361996
2 accumulators, column RHS, column output         : 0.063726 ± 8.0622e-05
2 accumulators, column RHS, row output            : 0.0659513 ± 0.00221415
4 accumulators, column RHS, column output         : 0.0563938 ± 0.00137642
4 accumulators, column RHS, row output            : 0.0543498 ± 0.00037212
8 accumulators, column RHS, column output         : 0.0568389 ± 0.000945001
8 accumulators, column RHS, row output            : 0.0603894 ± 0.00419283
blocked (B = 2), column RHS, column output        : 0.0554934 ± 0.00358938
blocked (B = 4), column RHS, column output        : 0.0544192 ± 0.00159569
blocked (B = 8), column RHS, column output        : 0.0578591 ± 8.85873e-05
blocked (B = 16), column RHS, column output       : 0.0616368 ± 0.0008792
blocked (B = 2), column RHS, row output           : 0.0525777 ± 0.000566421
blocked (B = 4), column RHS, row output           : 0.0530893 ± 0.000260208
blocked (B = 8), column RHS, row output           : 0.0578554 ± 0.000117874
blocked (B = 16), column RHS, row output          : 0.0631518 ± 0.00115664
blocked (B = 4), row RHS, column output           : 0.0824344 ± 0.00432385
blocked (B = 8), row RHS, column output           : 0.0524671 ± 0.001873
blocked (B = 16), row RHS, column output          : 0.0369001 ± 0.000252406
blocked (B = 32), row RHS, column output          : 0.0312814 ± 0.000815846
blocked (B = 2), row RHS, row output              : 0.0834662 ± 0.00252013
blocked (B = 4), row RHS, row output              : 0.0706718 ± 0.000229354
blocked (B = 8), row RHS, row output              : 0.0663548 ± 0.00213634
blocked (B = 16), row RHS, row output             : 0.0634923 ± 0.00149477

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, column RHS, column output                  : 0.101929 ± 0.00112494
naive, column RHS, row output                     : 0.0968838 ± 9.89073e-05
naive, row RHS, column output                     : 0.312543 ± 0.00319355
naive, row RHS, row output                        : 0.069419 ± 0.000418902
2 accumulators, column RHS, column output         : 0.071648 ± 0.000115074
2 accumulators, column RHS, row output            : 0.0676612 ± 0.000162107
4 accumulators, column RHS, column output         : 0.0601939 ± 0.000166274
4 accumulators, column RHS, row output            : 0.0574829 ± 0.00170087
8 accumulators, column RHS, column output         : 0.0614036 ± 0.000148623
8 accumulators, column RHS, row output            : 0.0569126 ± 0.000145065
blocked (B = 2), column RHS, column output        : 0.0491185 ± 0.000110913
blocked (B = 4), column RHS, column output        : 0.0428849 ± 0.000135381
blocked (B = 8), column RHS, column output        : 0.0415226 ± 0.000207746
blocked (B = 16), column RHS, column output       : 0.0449605 ± 0.000190048
blocked (B = 2), column RHS, row output           : 0.0476799 ± 8.94829e-05
blocked (B = 4), column RHS, row output           : 0.0417056 ± 0.000154823
blocked (B = 8), column RHS, row output           : 0.0412175 ± 0.000138052
blocked (B = 16), column RHS, row output          : 0.0448308 ± 8.71383e-05
blocked (B = 4), row RHS, column output           : 0.105653 ± 0.00012674
blocked (B = 8), row RHS, column output           : 0.0644307 ± 9.74212e-05
blocked (B = 16), row RHS, column output          : 0.0457139 ± 0.000117872
blocked (B = 32), row RHS, column output          : 0.0350888 ± 9.0585e-05
blocked (B = 2), row RHS, row output              : 0.0656247 ± 0.000301986
blocked (B = 4), row RHS, row output              : 0.067279 ± 0.000105812
blocked (B = 8), row RHS, row output              : 0.0785529 ± 0.0002544
blocked (B = 16), row RHS, row output             : 0.0865626 ± 0.000245619
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


