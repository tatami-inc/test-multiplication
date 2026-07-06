# Sparse column-major LHS, sparse matrix RHS

## Strategies

All strategies must iterate across consecutive columns of the LHS matrix in the outermost loop.
We will be loading columns on demand via the **tatami** interface, so the full matrix will not be available for random access.

### Column-major RHS

When dealing with a sparse matrix, we always want to traverse the structural non-zeros for one of the dimensions.
However, walking down the RHS column would require all the LHS columns to be in memory to compute the cumulative dot products.
We can't effectively block, either, because we can't easily estimate the boundaries of the block submatrices within a sparse matrix.

Which is not to say that this is impossible.
We can iterate across all of the RHS columns to construct each row as it is needed.
However, this is not very amenable to parallelization as each thread would need to search for the start and end of its block of rows in each column.
At that point, it is better to just construct a row-major representation for the RHS matrix in the first place.

Thus, for the rest of this document, we'll restrict our consideration to a row-major sparse RHS matrix. 

### Naive row-major RHS

For row-major output, we consider each LHS column and RHS row $i$. 
We iterate over the non-zero elements of the LHS column;
for a non-zero at row $j$ with value $x$, we perform a sparse vector multiply-add of the RHS row to the output row $j$.
We repeat this process for each $i$ until both matrices are fully traversed.

For column-major output, we consider each LHS column and RHS row $i$. 
We iterate over the non-zero elements of the RHS row;
for a non-zero at column $j$ with value $x$, we perform a sparse vector multiply-add of the LHS column to the output column $j$.
We repeat this process for each $i$ until both matrices are fully traversed.

### Blocked row-major RHS

We follow the framework described in the "Blocking along non-zeros" section in [`general/README.md`](../../general/README.md).
Consider each LHS column and RHS row $i$. 
We split the structural non-zeros in the LHS column into "primary" blocks of size $B$, while we split the structural non-zeros in the RHS row into "secondary" blocks of size $C$. 

For row-major output, we iterate over the non-zero elements in each primary LHS block.
For a non-zero at LHS row $j$ with value $x$, we perform a sparse vector multiply-add of the secondary RHS block to the corresponding part of output row $j$.
Once all non-zeros in the current primary block are processed, we move onto the next secondary block of $C$ RHS non-zeros.
Once all non-zeros in the RHS row $i$ are processed, we move onto the next primary block in LHS column $i$, and so on until all LHS non-zeros are processed.
We repeat this process for each $i$ until both matrices are fully traversed.

For column-major output, we iterate over the non-zero elements in each RHS block.
For a non-zero at RHS column $j$ with value $x$, we perform a sparse vector multiply-add of the primary LHS block to output column $j$.
Once all non-zeros in the current secondary block are processed, we move onto the next primary block of $C$ LHS non-zeros.
Once all non-zeros in the LHS column $i$ are processed, we move onto the next secondary block in RHS row $i$, and so on until all RHS non-zeros are processed.
We repeat this process for each $i$ until both matrices are fully traversed.

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
naive, row-major RHS, row-major output                 : 0.0405192 ± 0.00313948
naive, row-major RHS, column-major output              : 0.0441518 ± 0.00339525
blocked (B = 2), row-major RHS, row-major output       : 0.039344 ± 0.00265603
blocked (B = 4), row-major RHS, row-major output       : 0.041196 ± 0.00370417
blocked (B = 8), row-major RHS, row-major output       : 0.0469882 ± 0.00321585
blocked (B = 16), row-major RHS, row-major output      : 0.0366927 ± 0.00315293
blocked (B = 2), row-major RHS, column-major output    : 0.0503542 ± 0.00507091
blocked (B = 4), row-major RHS, column-major output    : 0.0491628 ± 0.00406384
blocked (B = 8), row-major RHS, column-major output    : 0.0434314 ± 0.00344844
blocked (B = 16), row-major RHS, column-major output   : 0.0417621 ± 0.00280602

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row-major RHS, row-major output                 : 0.0168375 ± 0.000342175
naive, row-major RHS, column-major output              : 0.0136184 ± 0.000173615
blocked (B = 2), row-major RHS, row-major output       : 0.0168648 ± 0.000130614
blocked (B = 4), row-major RHS, row-major output       : 0.016968 ± 0.000369545
blocked (B = 8), row-major RHS, row-major output       : 0.0166619 ± 0.000205425
blocked (B = 16), row-major RHS, row-major output      : 0.017141 ± 0.000449898
blocked (B = 2), row-major RHS, column-major output    : 0.0145027 ± 0.000131034
blocked (B = 4), row-major RHS, column-major output    : 0.0135918 ± 0.000283606
blocked (B = 8), row-major RHS, column-major output    : 0.0131943 ± 0.000176436
blocked (B = 16), row-major RHS, column-major output   : 0.0137081 ± 0.000441721

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row-major RHS, row-major output                 : 0.0888594 ± 0.00278199
naive, row-major RHS, column-major output              : 0.102923 ± 0.00706601
blocked (B = 2), row-major RHS, row-major output       : 0.0901913 ± 0.00349716
blocked (B = 4), row-major RHS, row-major output       : 0.0853341 ± 0.00444571
blocked (B = 8), row-major RHS, row-major output       : 0.0898517 ± 0.00242385
blocked (B = 16), row-major RHS, row-major output      : 0.0998395 ± 0.00624027
blocked (B = 2), row-major RHS, column-major output    : 0.103777 ± 0.00406469
blocked (B = 4), row-major RHS, column-major output    : 0.0965769 ± 0.00262984
blocked (B = 8), row-major RHS, column-major output    : 0.101536 ± 0.00440318
blocked (B = 16), row-major RHS, column-major output   : 0.101461 ± 0.00649295

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row-major RHS, row-major output                 : 0.0505245 ± 0.000842248
naive, row-major RHS, column-major output              : 0.0347033 ± 0.0010465
blocked (B = 2), row-major RHS, row-major output       : 0.0527702 ± 0.00125355
blocked (B = 4), row-major RHS, row-major output       : 0.0496327 ± 0.000770769
blocked (B = 8), row-major RHS, row-major output       : 0.0510496 ± 0.00150301
blocked (B = 16), row-major RHS, row-major output      : 0.0495113 ± 0.00156928
blocked (B = 2), row-major RHS, column-major output    : 0.0343262 ± 0.00138931
blocked (B = 4), row-major RHS, column-major output    : 0.038708 ± 0.00198027
blocked (B = 8), row-major RHS, column-major output    : 0.0384619 ± 0.00224962
blocked (B = 16), row-major RHS, column-major output   : 0.0368596 ± 0.00190075

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row-major RHS, row-major output                 : 0.0943426 ± 0.00285678
naive, row-major RHS, column-major output              : 0.0902203 ± 0.00342234
blocked (B = 2), row-major RHS, row-major output       : 0.0998984 ± 0.00723238
blocked (B = 4), row-major RHS, row-major output       : 0.0903079 ± 0.00315797
blocked (B = 8), row-major RHS, row-major output       : 0.0962891 ± 0.00592812
blocked (B = 16), row-major RHS, row-major output      : 0.0960318 ± 0.00271388
blocked (B = 2), row-major RHS, column-major output    : 0.105143 ± 0.00572834
blocked (B = 4), row-major RHS, column-major output    : 0.106999 ± 0.00660132
blocked (B = 8), row-major RHS, column-major output    : 0.0961348 ± 0.00321408
blocked (B = 16), row-major RHS, column-major output   : 0.0949019 ± 0.0035055

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, row-major RHS, row-major output                 : 0.0150475 ± 0.000434244
naive, row-major RHS, column-major output              : 0.0170339 ± 0.000502326
blocked (B = 2), row-major RHS, row-major output       : 0.0147774 ± 0.000334677
blocked (B = 4), row-major RHS, row-major output       : 0.0147409 ± 0.000453347
blocked (B = 8), row-major RHS, row-major output       : 0.0142325 ± 0.00015287
blocked (B = 16), row-major RHS, row-major output      : 0.0141707 ± 0.00033391
blocked (B = 2), row-major RHS, column-major output    : 0.0186019 ± 0.000295858
blocked (B = 4), row-major RHS, column-major output    : 0.0167263 ± 0.000365569
blocked (B = 8), row-major RHS, column-major output    : 0.0171712 ± 0.000306486
blocked (B = 16), row-major RHS, column-major output   : 0.0169835 ± 0.000183257

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, row-major RHS, row-major output                 : 0.0428413 ± 0.00291154
naive, row-major RHS, column-major output              : 0.0582624 ± 0.00194524
blocked (B = 2), row-major RHS, row-major output       : 0.0493016 ± 0.002837
blocked (B = 4), row-major RHS, row-major output       : 0.0447901 ± 0.00352125
blocked (B = 8), row-major RHS, row-major output       : 0.0474824 ± 0.00220804
blocked (B = 16), row-major RHS, row-major output      : 0.0471554 ± 0.0025167
blocked (B = 2), row-major RHS, column-major output    : 0.0671984 ± 0.0035578
blocked (B = 4), row-major RHS, column-major output    : 0.0603722 ± 0.00213078
blocked (B = 8), row-major RHS, column-major output    : 0.0591185 ± 0.00267446
blocked (B = 16), row-major RHS, column-major output   : 0.0612271 ± 0.00233626
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive, row-major RHS, row-major output                 : 0.0152596 ± 0.0008932
naive, row-major RHS, column-major output              : 0.0161517 ± 0.00137696
blocked (B = 2), row-major RHS, row-major output       : 0.0165977 ± 0.00139456
blocked (B = 4), row-major RHS, row-major output       : 0.0167755 ± 0.00125329
blocked (B = 8), row-major RHS, row-major output       : 0.0181597 ± 0.00126639
blocked (B = 16), row-major RHS, row-major output      : 0.0141727 ± 0.00108137
blocked (B = 2), row-major RHS, column-major output    : 0.0241357 ± 0.00129091
blocked (B = 4), row-major RHS, column-major output    : 0.0182071 ± 0.000864617
blocked (B = 8), row-major RHS, column-major output    : 0.0152277 ± 0.00125581
blocked (B = 16), row-major RHS, column-major output   : 0.0141455 ± 0.00104695

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row-major RHS, row-major output                 : 0.013125 ± 0.000178279
naive, row-major RHS, column-major output              : 0.009742 ± 0.000294046
blocked (B = 2), row-major RHS, row-major output       : 0.0140704 ± 0.000202643
blocked (B = 4), row-major RHS, row-major output       : 0.0141598 ± 0.000313207
blocked (B = 8), row-major RHS, row-major output       : 0.0137937 ± 0.000245618
blocked (B = 16), row-major RHS, row-major output      : 0.013093 ± 0.000246407
blocked (B = 2), row-major RHS, column-major output    : 0.0121952 ± 0.000313591
blocked (B = 4), row-major RHS, column-major output    : 0.0112357 ± 0.000335456
blocked (B = 8), row-major RHS, column-major output    : 0.011122 ± 0.000412773
blocked (B = 16), row-major RHS, column-major output   : 0.0100223 ± 0.000198227

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row-major RHS, row-major output                 : 0.0650003 ± 0.00358477
naive, row-major RHS, column-major output              : 0.0698671 ± 0.00388485
blocked (B = 2), row-major RHS, row-major output       : 0.0647582 ± 0.00341096
blocked (B = 4), row-major RHS, row-major output       : 0.0628183 ± 0.00332158
blocked (B = 8), row-major RHS, row-major output       : 0.0664842 ± 0.00303748
blocked (B = 16), row-major RHS, row-major output      : 0.0674512 ± 0.00382289
blocked (B = 2), row-major RHS, column-major output    : 0.0873428 ± 0.00193853
blocked (B = 4), row-major RHS, column-major output    : 0.0805864 ± 0.00193787
blocked (B = 8), row-major RHS, column-major output    : 0.0743181 ± 0.00359962
blocked (B = 16), row-major RHS, column-major output   : 0.0711561 ± 0.00333334

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row-major RHS, row-major output                 : 0.023636 ± 0.00100385
naive, row-major RHS, column-major output              : 0.0157838 ± 0.00166218
blocked (B = 2), row-major RHS, row-major output       : 0.0310908 ± 0.00210912
blocked (B = 4), row-major RHS, row-major output       : 0.0262435 ± 0.00177497
blocked (B = 8), row-major RHS, row-major output       : 0.0255606 ± 0.00165331
blocked (B = 16), row-major RHS, row-major output      : 0.0248265 ± 0.00163419
blocked (B = 2), row-major RHS, column-major output    : 0.0178946 ± 0.00188931
blocked (B = 4), row-major RHS, column-major output    : 0.014352 ± 0.00113246
blocked (B = 8), row-major RHS, column-major output    : 0.0138033 ± 0.00108937
blocked (B = 16), row-major RHS, column-major output   : 0.0157569 ± 0.00112936

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row-major RHS, row-major output                 : 0.0612957 ± 0.00269437
naive, row-major RHS, column-major output              : 0.0549771 ± 0.0018628
blocked (B = 2), row-major RHS, row-major output       : 0.0627537 ± 0.00330228
blocked (B = 4), row-major RHS, row-major output       : 0.0575118 ± 0.00243982
blocked (B = 8), row-major RHS, row-major output       : 0.0612216 ± 7.40208e-05
blocked (B = 16), row-major RHS, row-major output      : 0.0619518 ± 0.0017691
blocked (B = 2), row-major RHS, column-major output    : 0.0728023 ± 0.00197931
blocked (B = 4), row-major RHS, column-major output    : 0.0617762 ± 0.00116792
blocked (B = 8), row-major RHS, column-major output    : 0.0581354 ± 4.34213e-05
blocked (B = 16), row-major RHS, column-major output   : 0.0623088 ± 0.00345042

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, row-major RHS, row-major output                 : 0.0116466 ± 0.000483123
naive, row-major RHS, column-major output              : 0.0172992 ± 0.000822806
blocked (B = 2), row-major RHS, row-major output       : 0.0133705 ± 0.00071129
blocked (B = 4), row-major RHS, row-major output       : 0.0134927 ± 0.000922227
blocked (B = 8), row-major RHS, row-major output       : 0.0136321 ± 0.000749432
blocked (B = 16), row-major RHS, row-major output      : 0.0129619 ± 0.000622445
blocked (B = 2), row-major RHS, column-major output    : 0.0202933 ± 0.00094646
blocked (B = 4), row-major RHS, column-major output    : 0.0174116 ± 0.000559471
blocked (B = 8), row-major RHS, column-major output    : 0.0161797 ± 0.000388238
blocked (B = 16), row-major RHS, column-major output   : 0.0171122 ± 0.000681321

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 1000 x 10000
naive, row-major RHS, row-major output                 : 0.0201811 ± 0.0013114
naive, row-major RHS, column-major output              : 0.0269652 ± 0.00173434
blocked (B = 2), row-major RHS, row-major output       : 0.0167842 ± 0.00132894
blocked (B = 4), row-major RHS, row-major output       : 0.0175099 ± 0.00142496
blocked (B = 8), row-major RHS, row-major output       : 0.0156303 ± 0.00111985
blocked (B = 16), row-major RHS, row-major output      : 0.0142567 ± 0.00113295
blocked (B = 2), row-major RHS, column-major output    : 0.0375725 ± 0.00199885
blocked (B = 4), row-major RHS, column-major output    : 0.0243696 ± 0.000910245
blocked (B = 8), row-major RHS, column-major output    : 0.0261992 ± 0.00187906
blocked (B = 16), row-major RHS, column-major output   : 0.029683 ± 0.00255992
```

## Conclusions

Blocking doesn't help all that much.
I'd guess that the bottleneck is the random access to the dense output vectors.
We don't have much opportunity to follow the advice in the "Blocking to cache the dense vector" section of [`general/README.md`](../../general/README.md).
In particular, we can't easily re-use the dense output vectors across multiple $i$, given that different LHS columns/RHS rows will have their non-zeros in different positions.
