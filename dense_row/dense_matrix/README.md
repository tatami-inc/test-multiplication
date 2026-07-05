# Dense row-major LHS, dense matrix RHS

## Strategies

All strategies must iterate across consecutive rows of the LHS matrix in the outermost loop.
We will be loading rows on demand via the **tatami** interface, so the full matrix will not be available for random access.

### Naive row-major RHS

If the output is row-major: for each LHS row $i$, we iterate across the RHS rows.
For each RHS row $h$, we perform a vector multiply-add to the $i$-th output row using the $h$-th element from LHS row $i$ as the scaling factor.
Each multiply-add involves contiguous memory and should be highly amenable to auto-vectorization.

If the output is column-major, the process is much the same except that we use a temporary buffer to represent output row $i$.
The $h$-th element of this buffer will be assigned to the $i$-th element from the $h$-output column once all RHS rows have been traversed.

### Blocked row-major RHS, row-major output

We consider a $B$-by-$B$ LHS submatrix with $B$-by-$C$ RHS and output submatrices.
In this set of submatrices, each LHS column corresponds to an RHS row, each LHS row corresponds to an output row, and each RHS column corresponds to an output column.

For each combination of LHS row $i$ and RHS row $j$ in these submatrices,
we perform a vector multipy-add of the current block of the $j$-th RHS row to the corresponding block of the $i$-th output row,
using the LHS matrix entry at $(i, j)$ as the scaling factor.
We repeat this for all valid sets of submatrices until the full matrix product is computed.
When choosing the next set of submatrices, the RHS/output columns are the fastest-changing dimension while the LHS rows are the slowest.

The hope is that these submatrices are small enough to keep in cache for fast access.
We test a range of different values for the $B$ given a fixed value for $BC = 1024$. 
Check out [`general/README.md`](../../general/README.md) for more details.

### Blocked row-major RHS, column-major output

The process is the same as for row-major output, except that we use temporary buffers to represent each of the $B$ output rows in each block.
After processing each block of $B$ LHS rows, we transpose the contents of the temporary buffers into the output columns.
We don't have to worry much about extra cache usage as none of the submatrices will be re-used anyway.

### Column-major RHS

For column-major output, we already tested the naive and blocked approaches in the [`multiple_vectors`](../multiple_vectors) tests.
So, for comparison's sake, we'll just use the best approach from that test suite, i.e., blocking with $B = 16$ and multiple accumulators.

For row-major output, we'll re-use the best approach for column-major output.
The change in the output layout should not have a major impact on performance as we don't iterate over the output in the innermost loop to compute the dot product.

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
naive, row RHS, column output               : 0.437899 ± 0.00072442
naive, row RHS, row output                  : 0.398676 ± 0.00207614
best column RHS, column output              : 0.266676 ± 0.000439561
best column RHS, row output                 : 0.2659 ± 0.000369683
blocked (B = 4), row RHS, column output     : 0.450132 ± 0.00242193
blocked (B = 8), row RHS, column output     : 0.405208 ± 0.000833427
blocked (B = 16), row RHS, column output    : 0.29406 ± 0.000715164
blocked (B = 32), row RHS, column output    : 0.323554 ± 0.000767128
blocked (B = 4), row RHS, row output        : 0.336541 ± 0.000700647
blocked (B = 8), row RHS, row output        : 0.29607 ± 0.000499779
blocked (B = 16), row RHS, row output       : 0.295757 ± 0.000426207
blocked (B = 32), row RHS, row output       : 0.32041 ± 0.000849475

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row RHS, column output               : 0.288928 ± 0.00131725
naive, row RHS, row output                  : 0.270894 ± 0.000512001
best column RHS, column output              : 0.253628 ± 0.000604204
best column RHS, row output                 : 0.25306 ± 0.000329962
blocked (B = 4), row RHS, column output     : 0.398548 ± 0.000947658
blocked (B = 8), row RHS, column output     : 0.384668 ± 0.000572594
blocked (B = 16), row RHS, column output    : 0.278284 ± 0.000825687
blocked (B = 32), row RHS, column output    : 0.326618 ± 0.000671052
blocked (B = 4), row RHS, row output        : 0.285168 ± 0.000665342
blocked (B = 8), row RHS, row output        : 0.271641 ± 0.000709636
blocked (B = 16), row RHS, row output       : 0.285958 ± 0.000638006
blocked (B = 32), row RHS, row output       : 0.325352 ± 0.000573886

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row RHS, column output               : 0.333524 ± 0.00208053
naive, row RHS, row output                  : 0.27048 ± 0.00119442
best column RHS, column output              : 0.271346 ± 0.00481316
best column RHS, row output                 : 0.254035 ± 0.00224047
blocked (B = 4), row RHS, column output     : 0.426133 ± 0.00279311
blocked (B = 8), row RHS, column output     : 0.400765 ± 0.00233773
blocked (B = 16), row RHS, column output    : 0.297948 ± 0.00264039
blocked (B = 32), row RHS, column output    : 0.332458 ± 0.00191485
blocked (B = 4), row RHS, row output        : 0.292837 ± 0.00157564
blocked (B = 8), row RHS, row output        : 0.272919 ± 0.00155839
blocked (B = 16), row RHS, row output       : 0.286658 ± 0.00145764
blocked (B = 32), row RHS, row output       : 0.320406 ± 0.00280008

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row RHS, column output               : 0.593617 ± 0.0109575
naive, row RHS, row output                  : 0.427729 ± 0.0068705
best column RHS, column output              : 0.318785 ± 0.0031724
best column RHS, row output                 : 0.269399 ± 0.000759304
blocked (B = 4), row RHS, column output     : 0.491014 ± 0.00737877
blocked (B = 8), row RHS, column output     : 0.430509 ± 0.00551897
blocked (B = 16), row RHS, column output    : 0.314571 ± 0.00289053
blocked (B = 32), row RHS, column output    : 0.34862 ± 0.0025226
blocked (B = 4), row RHS, row output        : 0.347171 ± 0.00535732
blocked (B = 8), row RHS, row output        : 0.30281 ± 0.0032267
blocked (B = 16), row RHS, row output       : 0.299008 ± 0.00208
blocked (B = 32), row RHS, row output       : 0.346838 ± 0.00331583

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row RHS, column output               : 0.441089 ± 0.0208323
naive, row RHS, row output                  : 0.397438 ± 0.00255121
best column RHS, column output              : 0.2657 ± 0.00152234
best column RHS, row output                 : 0.267902 ± 0.00128961
blocked (B = 4), row RHS, column output     : 0.434921 ± 0.00229695
blocked (B = 8), row RHS, column output     : 0.406835 ± 0.00166458
blocked (B = 16), row RHS, column output    : 0.294183 ± 0.000818794
blocked (B = 32), row RHS, column output    : 0.339148 ± 0.00161019
blocked (B = 4), row RHS, row output        : 0.329323 ± 0.0016543
blocked (B = 8), row RHS, row output        : 0.299934 ± 0.00143177
blocked (B = 16), row RHS, row output       : 0.307632 ± 0.00301931
blocked (B = 32), row RHS, row output       : 0.336338 ± 0.00126129

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, row RHS, column output               : 0.631187 ± 0.0184838
naive, row RHS, row output                  : 0.631705 ± 0.0176237
best column RHS, column output              : 0.277432 ± 0.00220333
best column RHS, row output                 : 0.283335 ± 0.0034047
blocked (B = 4), row RHS, column output     : 0.479593 ± 0.00685639
blocked (B = 8), row RHS, column output     : 0.421318 ± 0.00363932
blocked (B = 16), row RHS, column output    : 0.30188 ± 0.00284947
blocked (B = 32), row RHS, column output    : 0.349334 ± 0.00415359
blocked (B = 4), row RHS, row output        : 0.37158 ± 0.00349224
blocked (B = 8), row RHS, row output        : 0.318844 ± 0.00270321
blocked (B = 16), row RHS, row output       : 0.307273 ± 0.00270318
blocked (B = 32), row RHS, row output       : 0.34038 ± 0.00335587

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, row RHS, column output               : 0.612913 ± 0.0150398
naive, row RHS, row output                  : 0.612567 ± 0.0166061
best column RHS, column output              : 0.269251 ± 0.00206825
best column RHS, row output                 : 0.278653 ± 0.00285842
blocked (B = 4), row RHS, column output     : 0.488235 ± 0.00527424
blocked (B = 8), row RHS, column output     : 0.431874 ± 0.00412719
blocked (B = 16), row RHS, column output    : 0.311128 ± 0.00223578
blocked (B = 32), row RHS, column output    : 0.335939 ± 0.00212429
blocked (B = 4), row RHS, row output        : 0.391184 ± 0.00738979
blocked (B = 8), row RHS, row output        : 0.320376 ± 0.00354205
blocked (B = 16), row RHS, row output       : 0.312036 ± 0.00184319
blocked (B = 32), row RHS, row output       : 0.333418 ± 0.00240631
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive, row RHS, column output               : 0.161638 ± 0.000284352
naive, row RHS, row output                  : 0.199396 ± 0.00339187
best column RHS, column output              : 0.166731 ± 0.000263575
best column RHS, row output                 : 0.163074 ± 0.000635099
blocked (B = 4), row RHS, column output     : 0.153444 ± 0.000836379
blocked (B = 8), row RHS, column output     : 0.15391 ± 0.000379985
blocked (B = 16), row RHS, column output    : 0.14853 ± 0.000210509
blocked (B = 32), row RHS, column output    : 0.161317 ± 0.000369077
blocked (B = 4), row RHS, row output        : 0.138477 ± 0.000265352
blocked (B = 8), row RHS, row output        : 0.134184 ± 0.000513172
blocked (B = 16), row RHS, row output       : 0.14394 ± 0.00048642
blocked (B = 32), row RHS, row output       : 0.177017 ± 0.000246293

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row RHS, column output               : 0.159465 ± 0.000140734
naive, row RHS, row output                  : 0.184221 ± 0.000282945
best column RHS, column output              : 0.164408 ± 0.000152838
best column RHS, row output                 : 0.162395 ± 0.000104752
blocked (B = 4), row RHS, column output     : 0.148381 ± 0.000242169
blocked (B = 8), row RHS, column output     : 0.145988 ± 0.000139037
blocked (B = 16), row RHS, column output    : 0.153823 ± 0.000129325
blocked (B = 32), row RHS, column output    : 0.173353 ± 0.000342607
blocked (B = 4), row RHS, row output        : 0.139348 ± 0.000168755
blocked (B = 8), row RHS, row output        : 0.13688 ± 0.000123741
blocked (B = 16), row RHS, row output       : 0.152909 ± 0.000108553
blocked (B = 32), row RHS, row output       : 0.195312 ± 8.55476e-05

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row RHS, column output               : 0.181743 ± 0.000148653
naive, row RHS, row output                  : 0.186719 ± 0.00016006
best column RHS, column output              : 0.173122 ± 0.000109129
best column RHS, row output                 : 0.16715 ± 0.000291087
blocked (B = 4), row RHS, column output     : 0.160559 ± 0.000304697
blocked (B = 8), row RHS, column output     : 0.160428 ± 9.12494e-05
blocked (B = 16), row RHS, column output    : 0.155 ± 0.000366798
blocked (B = 32), row RHS, column output    : 0.167786 ± 0.00053901
blocked (B = 4), row RHS, row output        : 0.134188 ± 9.19232e-05
blocked (B = 8), row RHS, row output        : 0.132066 ± 0.000120042
blocked (B = 16), row RHS, row output       : 0.143652 ± 0.000144683
blocked (B = 32), row RHS, row output       : 0.178331 ± 0.000802006

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row RHS, column output               : 0.242138 ± 0.000882997
naive, row RHS, row output                  : 0.192627 ± 0.000212773
best column RHS, column output              : 0.199194 ± 0.000773907
best column RHS, row output                 : 0.170762 ± 0.000526195
blocked (B = 4), row RHS, column output     : 0.172254 ± 0.00117453
blocked (B = 8), row RHS, column output     : 0.169629 ± 0.000207833
blocked (B = 16), row RHS, column output    : 0.160439 ± 0.000256924
blocked (B = 32), row RHS, column output    : 0.172166 ± 0.000216261
blocked (B = 4), row RHS, row output        : 0.139066 ± 0.000134722
blocked (B = 8), row RHS, row output        : 0.134669 ± 0.000273303
blocked (B = 16), row RHS, row output       : 0.144467 ± 0.00028222
blocked (B = 32), row RHS, row output       : 0.177556 ± 0.000148788

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row RHS, column output               : 0.170776 ± 0.00325711
naive, row RHS, row output                  : 0.196893 ± 0.00198153
best column RHS, column output              : 0.165782 ± 0.000640434
best column RHS, row output                 : 0.163658 ± 0.00077499
blocked (B = 4), row RHS, column output     : 0.154899 ± 0.00176995
blocked (B = 8), row RHS, column output     : 0.152188 ± 0.00168378
blocked (B = 16), row RHS, column output    : 0.158317 ± 0.00115569
blocked (B = 32), row RHS, column output    : 0.177954 ± 0.00102425
blocked (B = 4), row RHS, row output        : 0.14912 ± 0.00253945
blocked (B = 8), row RHS, row output        : 0.144368 ± 0.0016994
blocked (B = 16), row RHS, row output       : 0.157114 ± 0.000737085
blocked (B = 32), row RHS, row output       : 0.199716 ± 0.00148129

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 1000 x 10000
naive, row RHS, column output               : 0.324839 ± 0.010048
naive, row RHS, row output                  : 0.327677 ± 0.0180731
best column RHS, column output              : 0.184761 ± 0.00154214
best column RHS, row output                 : 0.177766 ± 0.00114094
blocked (B = 4), row RHS, column output     : 0.21122 ± 0.00597486
blocked (B = 8), row RHS, column output     : 0.182426 ± 0.00334437
blocked (B = 16), row RHS, column output    : 0.166479 ± 0.00203459
blocked (B = 32), row RHS, column output    : 0.175689 ± 0.00401508
blocked (B = 4), row RHS, row output        : 0.187399 ± 0.00327087
blocked (B = 8), row RHS, row output        : 0.158446 ± 0.00158088
blocked (B = 16), row RHS, row output       : 0.161613 ± 0.00496004
blocked (B = 32), row RHS, row output       : 0.186456 ± 0.00138066

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 10000 x 1000
naive, row RHS, column output               : 0.294548 ± 0.00499718
naive, row RHS, row output                  : 0.329479 ± 0.030412
best column RHS, column output              : 0.174634 ± 0.000931726
best column RHS, row output                 : 0.175255 ± 0.00131088
blocked (B = 4), row RHS, column output     : 0.204172 ± 0.00387211
blocked (B = 8), row RHS, column output     : 0.181869 ± 0.0021178
blocked (B = 16), row RHS, column output    : 0.163676 ± 0.00163132
blocked (B = 32), row RHS, column output    : 0.167195 ± 0.000733748
blocked (B = 4), row RHS, row output        : 0.194654 ± 0.00736274
blocked (B = 8), row RHS, row output        : 0.169845 ± 0.00621585
blocked (B = 16), row RHS, row output       : 0.159044 ± 0.000791696
blocked (B = 32), row RHS, row output       : 0.183318 ± 0.000611644
```

## Conclusion

For row-major RHS with row-major output, blocking provides a clear benefit over the naive implementation in some scenarios, and is at least no worse.
$B = 8$ seems to perform the best or nearly so.

For row-major RHS with column-major output, blocking is also beneficial but this is much more dependent on the block shape,
presumably because the transposition uses up a lot more cache lines when the matrices are very fat or thin.
$B = 16$ seems to perform the best here.

Products involving the column-major RHS are pretty fast in all scenarios.
It is the best for double-precision floats and still competitive for single-precision;
row-major RHS's speed in the latter is probably due to improved auto-vectorization for smaller types.
