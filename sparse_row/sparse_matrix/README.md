# Sparse row-major LHS, sparse matrix RHS

## Strategies

All strategies must iterate across consecutive rows of the LHS matrix in the outermost loop.
We will be loading rows on demand via the **tatami** interface, so the full matrix will not be available for random access.

### Column-major RHS

We use exactly the same strategies as described for a [dense row-major product with a sparse matrix](../../dense_row/sparse_matrix).
This is because the dot product calculation for two sparse vectors is best done by expanding one of the vectors into a dense array,
as discussed in [`general/README.md`](../../general/README.md).
So, if the LHS matrix is sparse, we just load it as a dense array and compute a dense-sparse dot product with multiple accumulators.

### Naive row-major RHS

If the output is row-major: for each LHS row $i$, we iterate across its structural non-zeros.
For a structural non-zero at column $j$, we perform a sparse vector multiply-add of the $j$-th RHS row to output row $i$,
using the value at column $j$ from the $i$-th LHS row as the scaling factor.

If the output is column-major, the process is much the same except that the output is stored in a temporary buffer.
Once all non-zeros have been processed for an LHS row $i$, the buffer is transposed back into the $i$-th elements of each output column.

There's not much opportunity for blocking as we already follow the advice described in "Blocking to cache the dense vector" in [`general/README.md`](../general/README.md).
Specifically, we already re-use the single dense $i$-th output row over multiple sparse RHS rows.
We can't easily use the advice in the "Blocking to cache many dense vectors" section as using multiple output vectors would require considering multiple sparse LHS vectors,
and there's no guarantee that different sparse vectors have non-zeros in the same positions (to use the same RHS rows).

### Blocked row-major RHS

We consider a block of $B$ LHS rows.


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
naive, row RHS, row output                        : 0.0091949 ± 0.000320808
naive, row RHS, column output                     : 0.0141732 ± 0.000177574
best, column RHS, column output                   : 0.0518278 ± 0.000672238
best, column RHS, row output                      : 0.0497647 ± 0.000708259

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row RHS, row output                        : 0.0231038 ± 0.000779585
naive, row RHS, column output                     : 0.0237298 ± 0.000738317
best, column RHS, column output                   : 0.059413 ± 0.000593104
best, column RHS, row output                      : 0.0589458 ± 0.000615948

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row RHS, row output                        : 0.0221899 ± 0.000570881
naive, row RHS, column output                     : 0.105274 ± 0.000650406
best, column RHS, column output                   : 0.148366 ± 0.00145275
best, column RHS, row output                      : 0.130263 ± 0.000886095

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row RHS, row output                        : 0.0185764 ± 0.000379937
naive, row RHS, column output                     : 0.0209513 ± 0.000351586
best, column RHS, column output                   : 0.045692 ± 0.00046442
best, column RHS, row output                      : 0.043898 ± 0.000310554

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row RHS, row output                        : 0.0219146 ± 0.00101841
naive, row RHS, column output                     : 0.0446172 ± 0.000726846
best, column RHS, column output                   : 0.108007 ± 0.00114751
best, column RHS, row output                      : 0.0872965 ± 0.00109877

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, row RHS, row output                        : 0.027202 ± 0.00194448
naive, row RHS, column output                     : 0.0309349 ± 0.0019711
best, column RHS, column output                   : 0.106226 ± 0.00357625
best, column RHS, row output                      : 0.106918 ± 0.00162478

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, row RHS, row output                        : 0.0176258 ± 0.00245426
naive, row RHS, column output                     : 0.0266274 ± 0.00137205
best, column RHS, column output                   : 0.108291 ± 0.00351356
best, column RHS, row output                      : 0.097566 ± 0.00153462
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive, row RHS, row output                        : 0.00790758 ± 0.000166003
naive, row RHS, column output                     : 0.0126563 ± 0.000126303
best, column RHS, column output                   : 0.0435713 ± 0.00025489
best, column RHS, row output                      : 0.0416087 ± 0.000106816

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row RHS, row output                        : 0.0218208 ± 0.000374235
naive, row RHS, column output                     : 0.022381 ± 0.000349104
best, column RHS, column output                   : 0.0494421 ± 0.000197255
best, column RHS, row output                      : 0.0500609 ± 0.000482941

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row RHS, row output                        : 0.0111717 ± 0.000521235
naive, row RHS, column output                     : 0.0773264 ± 0.000891896
best, column RHS, column output                   : 0.13357 ± 0.00165109
best, column RHS, row output                      : 0.117397 ± 0.00118258

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row RHS, row output                        : 0.0167115 ± 0.000136954
naive, row RHS, column output                     : 0.0197596 ± 0.00015288
best, column RHS, column output                   : 0.0388678 ± 0.000362312
best, column RHS, row output                      : 0.0381919 ± 0.000256835

$ ./build/test_float -r 10000 -c 100 -H 1000
naive, row RHS, row output                        : 0.00994663 ± 0.000366776
naive, row RHS, column output                     : 0.0422773 ± 0.000505419
best, column RHS, column output                   : 0.0869793 ± 0.000614834
best, column RHS, row output                      : 0.0744111 ± 0.000744273

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, row RHS, row output                        : 0.0109078 ± 0.00147093
naive, row RHS, column output                     : 0.0192994 ± 0.000875545
best, column RHS, column output                   : 0.08689 ± 0.00390084
best, column RHS, row output                      : 0.0745847 ± 0.00299514

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, row RHS, row output                        : 0.0208793 ± 0.00242385
naive, row RHS, column output                     : 0.0229121 ± 0.00139054
best, column RHS, column output                   : 0.0827183 ± 0.00320654
best, column RHS, row output                      : 0.0830956 ± 0.00320684
```

## Conclusion

Row-major RHS products are quite fast, even when the output is column-major and transposition is required.

Column-major RHS products are slightly slower than one might expect as we need to spend time expanding the sparse data into the dense array for the dot product.
But it's not too bad.
