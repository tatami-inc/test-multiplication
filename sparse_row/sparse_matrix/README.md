# Sparse row-major product with sparse matrix

## Strategies

All strategies must iterate across consecutive rows of the LHS matrix in the outermost loop.
We will be loading rows on demand via the **tatami** interface, so the full matrix will not be available for random access.

### Column-major RHS

We use exactly the same strategies as described for a [dense row-major product with a sparse matrix](../../dense_row/sparse_matrix).
This is because the dot product calculation for two sparse vectors is best done by expanding one of the vectors into a dense array,
as discussed in [`general/README.md`](../../general/README.md).
So, if the LHS matrix is sparse, we just load it as a dense array and compute a dense-sparse dot product with multiple accumulators.

### Naive row-major RHS

The general idea is to accumulate the partial dot products by processing each RHS row.

If the output is row-major: for each LHS row $i$, we iterate across its structural non-zeros.
For a structural non-zero at column $j$, we perform a sparse vector multiply-add of the $j$-th RHS row to output row $i$,
using the value at column $j$ from the $i$-th LHS row as the scaling factor.

If the output is column-major, the process is much the same except that the output is done with conceptual output rows.
The conceptual $i$-th output row consists of the $i$-th element from each output column.

There's not much opportunity for blocking as we already follow the advice described in "Blocking to cache the dense vector" in [`general/README.md`](../general/README.md).
Specifically, we re-use the single dense $i$-th output row over multiple sparse RHS rows.

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
naive, row RHS, row output                        : 0.00974691 ± 0.000546057
naive, row RHS, column output                     : 0.0243895 ± 0.000262058
naive, column RHS, column output                  : 0.0527461 ± 0.000703644
naive, column RHS, row output                     : 0.0504591 ± 0.00046531

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row RHS, row output                        : 0.0218466 ± 0.00023848
naive, row RHS, column output                     : 0.0379368 ± 0.000744777
naive, column RHS, column output                  : 0.0622704 ± 0.000652685
naive, column RHS, row output                     : 0.0628882 ± 0.000503966

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row RHS, row output                        : 0.0245719 ± 0.00167466
naive, row RHS, column output                     : 0.093078 ± 0.00137624
naive, column RHS, column output                  : 0.155701 ± 0.000697547
naive, column RHS, row output                     : 0.134717 ± 0.000559659

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row RHS, row output                        : 0.0189484 ± 0.000551797
naive, row RHS, column output                     : 0.0341232 ± 0.000349086
naive, column RHS, column output                  : 0.0508091 ± 0.00138805
naive, column RHS, row output                     : 0.0518458 ± 0.00165641

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row RHS, row output                        : 0.0185508 ± 0.00174834
naive, row RHS, column output                     : 0.0461541 ± 0.00226978
naive, column RHS, column output                  : 0.103137 ± 0.00107079
naive, column RHS, row output                     : 0.0863968 ± 0.00123122

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, row RHS, row output                        : 0.02784 ± 0.00195484
naive, row RHS, column output                     : 0.0502914 ± 0.00130914
naive, column RHS, column output                  : 0.1037 ± 0.00224712
naive, column RHS, row output                     : 0.101403 ± 0.00176143

$ ./build/test -r 100 -c 1000 -H 1000
Results for 100 x 1000 x 10000
naive, row RHS, row output                        : 0.0120568 ± 0.000798161
naive, row RHS, column output                     : 0.061703 ± 0.00126865
naive, column RHS, column output                  : 0.111474 ± 0.00108271
naive, column RHS, row output                     : 0.104188 ± 0.000666023
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive, row RHS, row output                        : 0.008098 ± 0.000266361
naive, row RHS, column output                     : 0.0227222 ± 0.000498579
naive, column RHS, column output                  : 0.0455685 ± 0.000578896
naive, column RHS, row output                     : 0.0438019 ± 0.000275878

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row RHS, row output                        : 0.0208201 ± 0.000252517
naive, row RHS, column output                     : 0.0327509 ± 0.000223691
naive, column RHS, column output                  : 0.0503243 ± 0.000609337
naive, column RHS, row output                     : 0.0495242 ± 0.000233614

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 1000 x 100 x 10000
naive, row RHS, row output                        : 0.0114681 ± 0.000641552
naive, row RHS, column output                     : 0.0748511 ± 0.00195879
naive, column RHS, column output                  : 0.144002 ± 0.00220149
naive, column RHS, row output                     : 0.129835 ± 0.00100537

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 1000 x 100
naive, row RHS, row output                        : 0.0170281 ± 0.000258704
naive, row RHS, column output                     : 0.0276715 ± 0.000250128
naive, column RHS, column output                  : 0.041074 ± 0.000335663
naive, column RHS, row output                     : 0.040046 ± 0.000285261

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 10000 x 100 x 1000
naive, row RHS, row output                        : 0.0106828 ± 0.000643659
naive, row RHS, column output                     : 0.0406703 ± 0.000473592
naive, column RHS, column output                  : 0.0956511 ± 0.00036083
naive, column RHS, row output                     : 0.0810072 ± 0.000241104

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 10000 x 1000
naive, row RHS, row output                        : 0.0224175 ± 0.00235243
naive, row RHS, column output                     : 0.042039 ± 0.00129258
naive, column RHS, column output                  : 0.0791434 ± 0.00122514
naive, column RHS, row output                     : 0.0777247 ± 0.00162919

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 1000 x 10000
naive, row RHS, row output                        : 0.00812157 ± 0.000684961
naive, row RHS, column output                     : 0.0500534 ± 0.00211645
naive, column RHS, column output                  : 0.0799424 ± 0.00403167
naive, column RHS, row output                     : 0.0682742 ± 0.00340126
```

## Conclusion

Row-major RHS products are quite fast, even when the output is column-major and writes are not to near-contiguous memory.

Column-major RHS products are slightly slower than one might expect as we need to spend time expanding the sparse data into the dense array for the dot product.
But it's not too bad.
