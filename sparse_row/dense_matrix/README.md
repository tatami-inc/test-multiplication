# Sparse row-major LHS, dense matrix RHS

## Strategies

All strategies must iterate across consecutive rows of the LHS matrix in the outermost loop.
We will be loading rows on demand via the **tatami** interface, so the full matrix will not be available for random access.

### Column-major RHS

Here, we'll just re-use the best strategy from the [multiple vectors test suite](../multiple_vectors).
This involves blocking with 16 LHS rows and 4 accumulators for the dot product.
We test both row- and column-major output though it shouldn't make much difference as we don't iterate over RHS columns in the innermost loop.

### Naive row-major RHS

For each LHS row $i$, we iterate over its structural non-zeros.
For each non-zero at column $j$ with value $x$, we perform a vector multiply-add of RHS row $j$ to the $i$-th output row using $x$ as the scaling factor.
This is most obviously implemented when the output is row-major, in which case the $l$-th output row is trivially defined.

If the output is column-major, we use a temporary buffer to represent the $i$-th output row.
Once we finish the iteration over the non-zeros of LHS row $i$, we transpose the temporary buffer to the $i$-th element of each of the output columns.

### Blocked row-major RHS, row-major output

This is much the same as the naive strategy, but we only consider a block of $C$ elements along each RHS row at any given time.
For this block, we iterate down the RHS rows and perform a vector multiply-add to the output row;
then we move onto the next $C$ elements and repeat, until all RHS columns have been processed.

The idea is to keep $C$ small enough to avoid eviction of the active part of the output row from cache. 
This is particularly relevant for column-major output where the cache will be populated with unused values in the same cache line.

Other than the above strategy, there is no obvious blocking strategy here.
We can't easily block on multiple LHS rows, for the reasons described in [`general/README.md`](../../general/README.md).
The advice in the "Blocking to cache the dense vector" section isn't really applicable either,
as our innermost loop consists of a dense vector multiply-add and we are already re-using the output vector for each LHS row.

### Blocked row-major RHS, column-major output

We load a block of $B$ LHS rows and allocate a temporary buffer of $B$ output rows.
For each LHS row $i$, we iterate over its structural non-zeros.
For each non-zero at column $j$ with value $x$, we perform a vector multiply-add of RHS row $j$ to the corresponding row output buffer, using $x$ as the scaling factor.
Once all rows in the block have been processed, we transpose the results from the temporary buffers to the column-major output.
We use a blocked transposition to improve the cache-friendliness of this step.

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
naive, row RHS, row output                        : 0.0531619 ± 0.00204687
naive, row RHS, column output                     : 0.0641245 ± 0.00614352
blocked, column RHS, column output                : 0.0856173 ± 0.00941316
blocked, column RHS, row output                   : 0.0858758 ± 0.00923166
blocked (C = 128), row RHS, row output            : 0.0582133 ± 0.00153615
blocked (C = 256), row RHS, row output            : 0.0686236 ± 0.00523097
blocked (C = 512), row RHS, row output            : 0.0584703 ± 0.00216074
blocked (B = 4), row RHS, column output           : 0.0643563 ± 0.00209143
blocked (B = 8), row RHS, column output           : 0.0638314 ± 0.00199189
blocked (B = 16), row RHS, column output          : 0.0661167 ± 0.00567413
blocked (B = 32), row RHS, column output          : 0.0603181 ± 0.00243208

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row RHS, row output                        : 0.0856713 ± 0.00152719
naive, row RHS, column output                     : 0.0898249 ± 0.00029347
blocked, column RHS, column output                : 0.0946824 ± 0.00215273
blocked, column RHS, row output                   : 0.096595 ± 0.000274054
blocked (C = 128), row RHS, row output            : 0.0902187 ± 0.00140039
blocked (C = 256), row RHS, row output            : 0.0900259 ± 0.00142414
blocked (C = 512), row RHS, row output            : 0.0911258 ± 9.03564e-05
blocked (B = 4), row RHS, column output           : 0.0887401 ± 0.000391289
blocked (B = 8), row RHS, column output           : 0.0888624 ± 0.000649195
blocked (B = 16), row RHS, column output          : 0.0866399 ± 0.000410527
blocked (B = 32), row RHS, column output          : 0.0862644 ± 0.000120005

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row RHS, row output                        : 0.069483 ± 0.000414982
naive, row RHS, column output                     : 0.139724 ± 0.000401318
blocked, column RHS, column output                : 0.103431 ± 0.00229154
blocked, column RHS, row output                   : 0.100808 ± 0.000774137
blocked (C = 128), row RHS, row output            : 0.0615221 ± 0.00119138
blocked (C = 256), row RHS, row output            : 0.0673438 ± 0.000364143
blocked (C = 512), row RHS, row output            : 0.0674208 ± 0.000992831
blocked (B = 4), row RHS, column output           : 0.092927 ± 0.0017434
blocked (B = 8), row RHS, column output           : 0.0943795 ± 0.00039187
blocked (B = 16), row RHS, column output          : 0.0930141 ± 7.37075e-05
blocked (B = 32), row RHS, column output          : 0.0910255 ± 0.00187214

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row RHS, row output                        : 0.0391782 ± 0.00023335
naive, row RHS, column output                     : 0.0413922 ± 0.000266983
blocked, column RHS, column output                : 0.0503172 ± 0.000492084
blocked, column RHS, row output                   : 0.0507439 ± 0.000191085
blocked (C = 128), row RHS, row output            : 0.0419907 ± 0.000288902
blocked (C = 256), row RHS, row output            : 0.0417604 ± 0.000241622
blocked (C = 512), row RHS, row output            : 0.0416756 ± 0.000265695
blocked (B = 4), row RHS, column output           : 0.0440057 ± 0.000269009
blocked (B = 8), row RHS, column output           : 0.0440412 ± 0.000276055
blocked (B = 16), row RHS, column output          : 0.0401689 ± 0.000188585
blocked (B = 32), row RHS, column output          : 0.0395667 ± 0.00017011

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row RHS, row output                        : 0.0423824 ± 0.000772636
naive, row RHS, column output                     : 0.0679759 ± 0.000506275
blocked, column RHS, column output                : 0.074905 ± 0.000927277
blocked, column RHS, row output                   : 0.0729296 ± 0.000439386
blocked (C = 128), row RHS, row output            : 0.0431688 ± 0.000668584
blocked (C = 256), row RHS, row output            : 0.0446172 ± 0.000769494
blocked (C = 512), row RHS, row output            : 0.0440136 ± 0.000976258
blocked (B = 4), row RHS, column output           : 0.0636664 ± 0.00106353
blocked (B = 8), row RHS, column output           : 0.0602809 ± 0.000995218
blocked (B = 16), row RHS, column output          : 0.0506073 ± 0.000946779
blocked (B = 32), row RHS, column output          : 0.0500628 ± 0.00127536

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, row RHS, row output                        : 0.0884082 ± 0.000292221
naive, row RHS, column output                     : 0.0906979 ± 0.000351552
blocked, column RHS, column output                : 0.112153 ± 0.00179048
blocked, column RHS, row output                   : 0.113663 ± 0.000106475
blocked (C = 128), row RHS, row output            : 0.102816 ± 0.000269955
blocked (C = 256), row RHS, row output            : 0.102888 ± 0.000300531
blocked (C = 512), row RHS, row output            : 0.0986944 ± 0.000366371
blocked (B = 4), row RHS, column output           : 0.0890737 ± 0.000150343
blocked (B = 8), row RHS, column output           : 0.0891317 ± 0.000196053
blocked (B = 16), row RHS, column output          : 0.0890965 ± 0.000686669
blocked (B = 32), row RHS, column output          : 0.0892334 ± 0.000415616

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, row RHS, row output                        : 0.0815722 ± 0.000223037
naive, row RHS, column output                     : 0.0889557 ± 0.000297555
blocked, column RHS, column output                : 0.101582 ± 0.00195426
blocked, column RHS, row output                   : 0.103175 ± 4.65001e-05
blocked (C = 128), row RHS, row output            : 0.0939818 ± 0.000244572
blocked (C = 256), row RHS, row output            : 0.0883892 ± 0.000229266
blocked (C = 512), row RHS, row output            : 0.0868249 ± 0.000226909
blocked (B = 4), row RHS, column output           : 0.0854133 ± 7.81529e-05
blocked (B = 8), row RHS, column output           : 0.0846133 ± 7.42737e-05
blocked (B = 16), row RHS, column output          : 0.0841916 ± 0.00010081
blocked (B = 32), row RHS, column output          : 0.0846137 ± 7.0485e-05
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive, row RHS, row output                        : 0.021626 ± 0.00126211
naive, row RHS, column output                     : 0.027625 ± 0.00119626
blocked, column RHS, column output                : 0.0521559 ± 0.00136639
blocked, column RHS, row output                   : 0.0544677 ± 0.00079664
blocked (C = 128), row RHS, row output            : 0.0293907 ± 0.00135793
blocked (C = 256), row RHS, row output            : 0.0233254 ± 0.00058938
blocked (C = 512), row RHS, row output            : 0.0261696 ± 0.00122895
blocked (B = 4), row RHS, column output           : 0.0275783 ± 0.00060588
blocked (B = 8), row RHS, column output           : 0.0250745 ± 0.000575899
blocked (B = 16), row RHS, column output          : 0.0285058 ± 0.00104804
blocked (B = 32), row RHS, column output          : 0.0232666 ± 0.000585074

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row RHS, row output                        : 0.0381041 ± 0.00147711
naive, row RHS, column output                     : 0.0410161 ± 0.00191833
blocked, column RHS, column output                : 0.0649515 ± 0.00116332
blocked, column RHS, row output                   : 0.0678684 ± 0.0013343
blocked (C = 128), row RHS, row output            : 0.0417984 ± 0.00172737
blocked (C = 256), row RHS, row output            : 0.0404723 ± 0.0013249
blocked (C = 512), row RHS, row output            : 0.0377359 ± 0.000654907
blocked (B = 4), row RHS, column output           : 0.0411086 ± 0.00245698
blocked (B = 8), row RHS, column output           : 0.0381917 ± 0.00143555
blocked (B = 16), row RHS, column output          : 0.040678 ± 0.00215948
blocked (B = 32), row RHS, column output          : 0.0398246 ± 0.00144654

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row RHS, row output                        : 0.0280585 ± 0.00102034
naive, row RHS, column output                     : 0.0871159 ± 0.00107565
blocked, column RHS, column output                : 0.0828377 ± 0.00197309
blocked, column RHS, row output                   : 0.0774736 ± 0.000834622
blocked (C = 128), row RHS, row output            : 0.0248281 ± 0.0016772
blocked (C = 256), row RHS, row output            : 0.0216012 ± 0.000369832
blocked (C = 512), row RHS, row output            : 0.028053 ± 0.00106399
blocked (B = 4), row RHS, column output           : 0.0538462 ± 0.00175104
blocked (B = 8), row RHS, column output           : 0.0526342 ± 0.00195228
blocked (B = 16), row RHS, column output          : 0.0502564 ± 0.00140831
blocked (B = 32), row RHS, column output          : 0.0457392 ± 0.00220482

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row RHS, row output                        : 0.022669 ± 0.000262077
naive, row RHS, column output                     : 0.0238907 ± 0.000116622
blocked, column RHS, column output                : 0.0454731 ± 0.000316511
blocked, column RHS, row output                   : 0.0456586 ± 9.98044e-05
blocked (C = 128), row RHS, row output            : 0.0240402 ± 0.000205071
blocked (C = 256), row RHS, row output            : 0.0237174 ± 7.85689e-05
blocked (C = 512), row RHS, row output            : 0.0239914 ± 0.000200272
blocked (B = 4), row RHS, column output           : 0.0235057 ± 5.86192e-05
blocked (B = 8), row RHS, column output           : 0.0225829 ± 0.000136151
blocked (B = 16), row RHS, column output          : 0.0230082 ± 5.9264e-05
blocked (B = 32), row RHS, column output          : 0.022401 ± 0.000110861

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row RHS, row output                        : 0.0193061 ± 0.000334876
naive, row RHS, column output                     : 0.0531932 ± 0.000339249
blocked, column RHS, column output                : 0.070624 ± 0.000452946
blocked, column RHS, row output                   : 0.0695536 ± 0.000270751
blocked (C = 128), row RHS, row output            : 0.0208621 ± 0.000514719
blocked (C = 256), row RHS, row output            : 0.0211473 ± 0.00084275
blocked (C = 512), row RHS, row output            : 0.0220455 ± 0.00046039
blocked (B = 4), row RHS, column output           : 0.0366048 ± 0.000555268
blocked (B = 8), row RHS, column output           : 0.0298142 ± 0.000534813
blocked (B = 16), row RHS, column output          : 0.0325311 ± 0.000666317
blocked (B = 32), row RHS, column output          : 0.0277245 ± 0.000270295

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, row RHS, row output                        : 0.0551785 ± 0.00141492
naive, row RHS, column output                     : 0.0586561 ± 0.000640402
blocked, column RHS, column output                : 0.0803403 ± 0.00160477
blocked, column RHS, row output                   : 0.0825825 ± 0.000748725
blocked (C = 128), row RHS, row output            : 0.0639381 ± 0.0017474
blocked (C = 256), row RHS, row output            : 0.0618487 ± 0.00168013
blocked (C = 512), row RHS, row output            : 0.0622367 ± 0.00167277
blocked (B = 4), row RHS, column output           : 0.0576401 ± 0.00136104
blocked (B = 8), row RHS, column output           : 0.0570247 ± 0.0012424
blocked (B = 16), row RHS, column output          : 0.0585327 ± 0.00130539
blocked (B = 32), row RHS, column output          : 0.0592498 ± 0.00109283

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, row RHS, row output                        : 0.045465 ± 0.00162198
naive, row RHS, column output                     : 0.0532988 ± 0.00136606
blocked, column RHS, column output                : 0.0717811 ± 0.00169182
blocked, column RHS, row output                   : 0.0712592 ± 0.00109731
blocked (C = 128), row RHS, row output            : 0.061824 ± 0.00124696
blocked (C = 256), row RHS, row output            : 0.0555167 ± 0.00126437
blocked (C = 512), row RHS, row output            : 0.0485487 ± 0.00126352
blocked (B = 4), row RHS, column output           : 0.050089 ± 0.00196203
blocked (B = 8), row RHS, column output           : 0.0478909 ± 0.00149774
blocked (B = 16), row RHS, column output          : 0.0469713 ± 0.00153467
blocked (B = 32), row RHS, column output          : 0.047363 ± 0.00177184
```

## Conclusion

Row-major RHS to row-major output is very efficient, even beating out the "obvious" product with a column-major RHS.
This is probably to be expected given that the innermost loop involves contiguous memory and can be easily vectorization.

Blocking for row-major RHS and row-major output does not help, probably because the loop restart overhead offsets any improvement in cache-friendliness.

Blocking for row-major RHS and column-major output is occasionally helpful, mostly for situations where there are many RHS columns.
