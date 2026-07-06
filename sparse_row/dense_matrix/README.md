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

### Blocked row-major RHS

This is much the same as the naive row-major strategy, but we only consider a block of $C$ elements along each RHS row at any given time.
For this block, we iterate down the RHS rows and perform a vector multiply-add to the output row;
then we move onto the next $C$ elements and repeat, until all RHS columns have been processed.

The idea is to keep $C$ small enough to avoid eviction of the active part of the output row from cache. 
This is particularly relevant for column-major output where the cache will be populated with unused values in the same cache line.

Other than the above strategy, there is no obvious blocking strategy here.
We can't easily block on multiple LHS rows, for the reasons described in [`general/README.md`](../../general/README.md).
The advice in the "Blocking to cache the dense vector" section isn't really applicable either,
as our innermost loop consists of a dense vector multiply-add and we are already re-using the output vector for each LHS row.

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
naive, row RHS, row output                        : 0.0460588 ± 0.00330281
naive, row RHS, column output                     : 0.0540998 ± 0.00271173
blocked, column RHS, column output                : 0.0630759 ± 0.00185478
blocked, column RHS, row output                   : 0.064546 ± 0.00179411
blocked (C = 128), row RHS, row output            : 0.0525098 ± 0.00263784
blocked (C = 256), row RHS, row output            : 0.0521226 ± 0.00208019
blocked (C = 512), row RHS, row output            : 0.0484932 ± 0.00191745
blocked (C = 128), row RHS, column output         : 0.056303 ± 0.00160926
blocked (C = 256), row RHS, column output         : 0.0543689 ± 0.00174219
blocked (C = 512), row RHS, column output         : 0.0524003 ± 0.00201286

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row RHS, row output                        : 0.0735664 ± 0.000313009
naive, row RHS, column output                     : 0.0779421 ± 0.000550484
blocked, column RHS, column output                : 0.0835776 ± 0.001549
blocked, column RHS, row output                   : 0.0804031 ± 0.00107411
blocked (C = 128), row RHS, row output            : 0.081701 ± 0.00204057
blocked (C = 256), row RHS, row output            : 0.0799298 ± 0.00172603
blocked (C = 512), row RHS, row output            : 0.0810431 ± 0.00179551
blocked (C = 128), row RHS, column output         : 0.0775434 ± 0.00186665
blocked (C = 256), row RHS, column output         : 0.0758692 ± 0.000289546
blocked (C = 512), row RHS, column output         : 0.0772875 ± 0.00158213

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row RHS, row output                        : 0.0552198 ± 0.00119858
naive, row RHS, column output                     : 0.151166 ± 0.000879399
blocked, column RHS, column output                : 0.106721 ± 0.000905374
blocked, column RHS, row output                   : 0.0998967 ± 0.00109546
blocked (C = 128), row RHS, row output            : 0.0518616 ± 0.000225112
blocked (C = 256), row RHS, row output            : 0.0534019 ± 0.0010846
blocked (C = 512), row RHS, row output            : 0.0538481 ± 0.000964849
blocked (C = 128), row RHS, column output         : 0.14645 ± 0.000770905
blocked (C = 256), row RHS, column output         : 0.150693 ± 0.000441325
blocked (C = 512), row RHS, column output         : 0.148965 ± 0.000817317

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row RHS, row output                        : 0.0382865 ± 0.000131652
naive, row RHS, column output                     : 0.0409861 ± 8.40963e-05
blocked, column RHS, column output                : 0.0456873 ± 6.76628e-05
blocked, column RHS, row output                   : 0.0454683 ± 0.000184191
blocked (C = 128), row RHS, row output            : 0.0409425 ± 5.3359e-05
blocked (C = 256), row RHS, row output            : 0.0408791 ± 9.59056e-05
blocked (C = 512), row RHS, row output            : 0.0407881 ± 8.04983e-05
blocked (C = 128), row RHS, column output         : 0.0427848 ± 0.000104469
blocked (C = 256), row RHS, column output         : 0.0424956 ± 0.000107274
blocked (C = 512), row RHS, column output         : 0.0425717 ± 0.000136229

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row RHS, row output                        : 0.037236 ± 0.000441647
naive, row RHS, column output                     : 0.0700007 ± 0.000314508
blocked, column RHS, column output                : 0.0874915 ± 0.00017073
blocked, column RHS, row output                   : 0.0842602 ± 0.000368006
blocked (C = 128), row RHS, row output            : 0.03905 ± 0.000151329
blocked (C = 256), row RHS, row output            : 0.041021 ± 0.000581308
blocked (C = 512), row RHS, row output            : 0.0401235 ± 0.000481669
blocked (C = 128), row RHS, column output         : 0.0696954 ± 0.000353812
blocked (C = 256), row RHS, column output         : 0.0696242 ± 0.000140244
blocked (C = 512), row RHS, column output         : 0.0678285 ± 0.000365976

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, row RHS, row output                        : 0.0763929 ± 0.00127659
naive, row RHS, column output                     : 0.0861563 ± 0.00294199
blocked, column RHS, column output                : 0.104407 ± 0.00143473
blocked, column RHS, row output                   : 0.100668 ± 0.000941184
blocked (C = 128), row RHS, row output            : 0.108184 ± 0.0137536
blocked (C = 256), row RHS, row output            : 0.0910038 ± 0.00132061
blocked (C = 512), row RHS, row output            : 0.0881999 ± 0.00184591
blocked (C = 128), row RHS, column output         : 0.112797 ± 0.0157514
blocked (C = 256), row RHS, column output         : 0.122903 ± 0.0280496
blocked (C = 512), row RHS, column output         : 0.112684 ± 0.0222334

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, row RHS, row output                        : 0.068989 ± 0.000115923
naive, row RHS, column output                     : 0.0782779 ± 0.00138271
blocked, column RHS, column output                : 0.0928836 ± 0.000243174
blocked, column RHS, row output                   : 0.0949537 ± 0.00276308
blocked (C = 128), row RHS, row output            : 0.0821882 ± 0.000272044
blocked (C = 256), row RHS, row output            : 0.0777324 ± 0.00136665
blocked (C = 512), row RHS, row output            : 0.0750155 ± 0.000293431
blocked (C = 128), row RHS, column output         : 0.0946204 ± 0.00320736
blocked (C = 256), row RHS, column output         : 0.0859863 ± 0.00181124
blocked (C = 512), row RHS, column output         : 0.0837395 ± 0.000549932
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive, row RHS, row output                        : 0.02195 ± 0.00105427
naive, row RHS, column output                     : 0.0247275 ± 0.000944228
blocked, column RHS, column output                : 0.0444091 ± 0.00071597
blocked, column RHS, row output                   : 0.0431082 ± 0.000981927
blocked (C = 128), row RHS, row output            : 0.0252006 ± 0.00118032
blocked (C = 256), row RHS, row output            : 0.0232142 ± 0.000720437
blocked (C = 512), row RHS, row output            : 0.0248254 ± 0.0010744
blocked (C = 128), row RHS, column output         : 0.0433392 ± 0.000956829
blocked (C = 256), row RHS, column output         : 0.0418413 ± 0.000620812
blocked (C = 512), row RHS, column output         : 0.0322344 ± 0.000759028

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row RHS, row output                        : 0.034289 ± 0.000542321
naive, row RHS, column output                     : 0.034269 ± 0.000647106
blocked, column RHS, column output                : 0.0573675 ± 0.00185613
blocked, column RHS, row output                   : 0.0544813 ± 0.000664038
blocked (C = 128), row RHS, row output            : 0.0360739 ± 0.000549512
blocked (C = 256), row RHS, row output            : 0.0367187 ± 0.000409705
blocked (C = 512), row RHS, row output            : 0.0367051 ± 0.000780183
blocked (C = 128), row RHS, column output         : 0.0463962 ± 0.000935793
blocked (C = 256), row RHS, column output         : 0.0464278 ± 0.00121811
blocked (C = 512), row RHS, column output         : 0.0452083 ± 0.00237745

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row RHS, row output                        : 0.0247778 ± 0.000998121
naive, row RHS, column output                     : 0.0921905 ± 0.0012589
blocked, column RHS, column output                : 0.0813766 ± 0.00105047
blocked, column RHS, row output                   : 0.0731434 ± 0.000864366
blocked (C = 128), row RHS, row output            : 0.0234632 ± 0.00117256
blocked (C = 256), row RHS, row output            : 0.024544 ± 0.000936309
blocked (C = 512), row RHS, row output            : 0.0258157 ± 0.00119773
blocked (C = 128), row RHS, column output         : 0.10607 ± 0.00164087
blocked (C = 256), row RHS, column output         : 0.106958 ± 0.000860245
blocked (C = 512), row RHS, column output         : 0.0974364 ± 0.00107223

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row RHS, row output                        : 0.0224964 ± 0.000204959
naive, row RHS, column output                     : 0.024747 ± 0.000140967
blocked, column RHS, column output                : 0.0379048 ± 0.000266363
blocked, column RHS, row output                   : 0.0376875 ± 0.000290801
blocked (C = 128), row RHS, row output            : 0.0239417 ± 0.000110955
blocked (C = 256), row RHS, row output            : 0.0241115 ± 0.000258061
blocked (C = 512), row RHS, row output            : 0.0240214 ± 0.000217932
blocked (C = 128), row RHS, column output         : 0.0359655 ± 8.00151e-05
blocked (C = 256), row RHS, column output         : 0.0360672 ± 0.000155637
blocked (C = 512), row RHS, column output         : 0.0325091 ± 0.000123851

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row RHS, row output                        : 0.0189474 ± 0.000301625
naive, row RHS, column output                     : 0.050856 ± 0.000284762
blocked, column RHS, column output                : 0.0731365 ± 0.000504326
blocked, column RHS, row output                   : 0.0697358 ± 0.000285929
blocked (C = 128), row RHS, row output            : 0.0209172 ± 0.000517376
blocked (C = 256), row RHS, row output            : 0.0211328 ± 0.00077664
blocked (C = 512), row RHS, row output            : 0.0209946 ± 0.000490834
blocked (C = 128), row RHS, column output         : 0.0681931 ± 0.000236411
blocked (C = 256), row RHS, column output         : 0.0673098 ± 0.000418667
blocked (C = 512), row RHS, column output         : 0.0562273 ± 0.000499628

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, row RHS, row output                        : 0.0492799 ± 0.00150064
naive, row RHS, column output                     : 0.052661 ± 0.0021239
blocked, column RHS, column output                : 0.0729214 ± 0.00161577
blocked, column RHS, row output                   : 0.0721894 ± 0.00181488
blocked (C = 128), row RHS, row output            : 0.0620729 ± 0.0025592
blocked (C = 256), row RHS, row output            : 0.053281 ± 0.00141068
blocked (C = 512), row RHS, row output            : 0.0569848 ± 0.00257189
blocked (C = 128), row RHS, column output         : 0.0671772 ± 0.00182631
blocked (C = 256), row RHS, column output         : 0.0672233 ± 0.00260849
blocked (C = 512), row RHS, column output         : 0.0600476 ± 0.00186834

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, row RHS, row output                        : 0.0377793 ± 0.000434014
naive, row RHS, column output                     : 0.0455398 ± 0.000180272
blocked, column RHS, column output                : 0.0592305 ± 0.000280337
blocked, column RHS, row output                   : 0.0579889 ± 0.000724933
blocked (C = 128), row RHS, row output            : 0.0542895 ± 0.000607432
blocked (C = 256), row RHS, row output            : 0.0457816 ± 0.0010903
blocked (C = 512), row RHS, row output            : 0.0420522 ± 0.000224078
blocked (C = 128), row RHS, column output         : 0.0714634 ± 0.00194997
blocked (C = 256), row RHS, column output         : 0.0610694 ± 0.00158261
blocked (C = 512), row RHS, column output         : 0.0531562 ± 0.000802023
```

## Conclusion

Row-major RHS to row-major output is very efficient, even beating out the "obvious" product with a column-major RHS.
This is probably to be expected given that the innermost loop involves contiguous memory and can be easily vectorization.

Blocking for row-major RHS does not help, probably because the loop restart overhead offsets any improvement in cache-friendliness.
