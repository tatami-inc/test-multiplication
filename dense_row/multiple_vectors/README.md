# Dense row-major product with multiple vectors

## Strategies

The naive approach involves computing dot products for each row of the LHS matrix and each of the multiple RHS vectors.

The blocked approach operates on $B$ rows of the LHS matrix and $B$ RHS vectors at a time.
For each combination of row/vector in this block, we compute the dot product of its first $C$ elements against the first $L$ elements of the RHS vector.
It repeatedly adds the dot product of the next $C$ elements until all columns of the LHS matrix have been traversed.
It proceeds to the next $B$ RHS vectors until all vectors have been traversed, and then onto the next $B$ rows until the entire LHS matrix is traversed.
We test a range of different values for the $B$ given a fixed value for $BC = 1024$, i.e., a thousand elements in the cache at once.
(The actual number of elements in the cache is actually $2BC$, as we hold a block from each of the LHS and RHS.)
Even for 8-byte types like `double`, this should easily fit into a modern L1 cache. 

Both the naive and blocked approaches can be augmented with the use of multiple accumulators for the dot product.
To keep things simple, we only consider the performance of blocking with 4 accumulators,
given that it's better than 2 and only slightly worse than 8 in the [`single_vector`](../single_vector) tests.

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
naive                                   : 0.999227 ± 0.0141087
naive, two accumulators                 : 0.566553 ± 0.0129811
naive, four accumulators                : 0.372495 ± 0.008084
naive, eight accumulators               : 0.355078 ± 0.00707665
blocked 4                               : 0.863145 ± 0.0115869
blocked 8                               : 0.672147 ± 0.00572202
blocked 16                              : 0.592091 ± 0.00438817
blocked 32                              : 0.474943 ± 0.00460112
blocked 4, four accumulators            : 0.279718 ± 0.00523501
blocked 8, four accumulators            : 0.256541 ± 0.005901
blocked 16, four accumulators           : 0.235334 ± 0.00303531
blocked 32, four accumulators           : 0.244417 ± 0.00286227

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive                                   : 0.97106 ± 0.000153536
naive, two accumulators                 : 0.540406 ± 9.52561e-05
naive, four accumulators                : 0.366584 ± 0.000168314
naive, eight accumulators               : 0.367815 ± 0.000522695
blocked 4                               : 0.831113 ± 0.000369822
blocked 8                               : 0.664952 ± 5.09318e-05
blocked 16                              : 0.582669 ± 9.7218e-05
blocked 32                              : 0.482899 ± 0.000299185
blocked 4, four accumulators            : 0.272045 ± 0.000166493
blocked 8, four accumulators            : 0.243787 ± 9.03451e-05
blocked 16, four accumulators           : 0.227969 ± 7.04959e-05
blocked 32, four accumulators           : 0.256277 ± 0.000494328

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive                                   : 0.731411 ± 0.00602103
naive, two accumulators                 : 0.527717 ± 0.0209102
naive, four accumulators                : 0.454956 ± 0.0234653
naive, eight accumulators               : 0.446888 ± 0.0150071
blocked 4                               : 0.744077 ± 0.014651
blocked 8                               : 0.684 ± 0.0108072
blocked 16                              : 0.595532 ± 0.00338098
blocked 32                              : 0.494437 ± 0.0053735
blocked 4, four accumulators            : 0.366804 ± 0.0211281
blocked 8, four accumulators            : 0.308618 ± 0.0126645
blocked 16, four accumulators           : 0.26734 ± 0.00187592
blocked 32, four accumulators           : 0.272722 ± 0.00163813

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive                                   : 0.919471 ± 0.00534865
naive, two accumulators                 : 0.471014 ± 0.00270843
naive, four accumulators                : 0.262593 ± 0.00182105
naive, eight accumulators               : 0.236827 ± 0.000726063
blocked 4                               : 0.831859 ± 0.00620036
blocked 8                               : 0.664245 ± 0.00554793
blocked 16                              : 0.583336 ± 0.00232623
blocked 32                              : 0.464274 ± 0.00165186
blocked 4, four accumulators            : 0.239678 ± 0.00189999
blocked 8, four accumulators            : 0.230658 ± 0.00115849
blocked 16, four accumulators           : 0.224108 ± 0.000936242
blocked 32, four accumulators           : 0.236428 ± 0.00129169

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive                                   : 0.628319 ± 0.00361067
naive, two accumulators                 : 0.386971 ± 0.000821229
naive, four accumulators                : 0.295477 ± 0.00120307
naive, eight accumulators               : 0.29753 ± 0.00165816
blocked 4                               : 0.646213 ± 0.00441981
blocked 8                               : 0.62206 ± 0.00355441
blocked 16                              : 0.556047 ± 0.00235836
blocked 32                              : 0.473805 ± 0.00245191
blocked 4, four accumulators            : 0.251717 ± 0.000676378
blocked 8, four accumulators            : 0.241184 ± 0.0016104
blocked 16, four accumulators           : 0.237158 ± 0.000649683
blocked 32, four accumulators           : 0.260334 ± 0.00116961

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive                                   : 1.04539 ± 0.00628575
naive, two accumulators                 : 0.640405 ± 0.00772523
naive, four accumulators                : 0.541648 ± 0.00997151
naive, eight accumulators               : 0.544991 ± 0.0118673
blocked 4                               : 0.86593 ± 0.00910082
blocked 8                               : 0.682858 ± 0.00291522
blocked 16                              : 0.592273 ± 0.00208968
blocked 32                              : 0.486961 ± 0.00214323
blocked 4, four accumulators            : 0.314259 ± 0.00225389
blocked 8, four accumulators            : 0.263469 ± 0.00249615
blocked 16, four accumulators           : 0.237589 ± 0.00172976
blocked 32, four accumulators           : 0.267265 ± 0.00538772

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive                                   : 1.00755 ± 0.00323985
naive, two accumulators                 : 0.606277 ± 0.00224609
naive, four accumulators                : 0.525764 ± 0.00296464
naive, eight accumulators               : 0.517379 ± 0.00217134
blocked 4                               : 0.865631 ± 0.0042153
blocked 8                               : 0.677012 ± 0.00180266
blocked 16                              : 0.591617 ± 0.0014843
blocked 32                              : 0.470007 ± 0.000516619
blocked 4, four accumulators            : 0.324845 ± 0.00317286
blocked 8, four accumulators            : 0.270567 ± 0.00250231
blocked 16, four accumulators           : 0.24194 ± 0.00323833
blocked 32, four accumulators           : 0.246396 ± 0.00105775
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive                                   : 0.927111 ± 0.000177594
naive, two accumulators                 : 0.471429 ± 0.000122773
naive, four accumulators                : 0.237814 ± 0.000122445
naive, eight accumulators               : 0.148698 ± 0.00011969
blocked 4                               : 0.787381 ± 0.000297445
blocked 8                               : 0.616301 ± 0.000103417
blocked 16                              : 0.552611 ± 0.000227896
blocked 32                              : 0.425048 ± 0.000175531
blocked 4, four accumulators            : 0.193343 ± 0.000130933
blocked 8, four accumulators            : 0.185601 ± 0.000118423
blocked 16, four accumulators           : 0.167228 ± 5.56122e-05
blocked 32, four accumulators           : 0.16889 ± 9.75357e-05

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive                                   : 0.941254 ± 0.000694926
naive, two accumulators                 : 0.477819 ± 0.00125533
naive, four accumulators                : 0.257699 ± 0.0128375
naive, eight accumulators               : 0.14805 ± 0.0011838
blocked 4                               : 0.78962 ± 0.000990038
blocked 8                               : 0.620055 ± 0.00063895
blocked 16                              : 0.557479 ± 0.000382053
blocked 32                              : 0.428371 ± 0.000484753
blocked 4, four accumulators            : 0.196768 ± 0.00701761
blocked 8, four accumulators            : 0.186646 ± 0.000825933
blocked 16, four accumulators           : 0.167171 ± 0.000186653
blocked 32, four accumulators           : 0.172561 ± 0.00159016

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive                                   : 0.835945 ± 0.00918114
naive, two accumulators                 : 0.427878 ± 0.00568296
naive, four accumulators                : 0.241157 ± 0.00716525
naive, eight accumulators               : 0.196138 ± 0.00363902
blocked 4                               : 0.628396 ± 0.00418696
blocked 8                               : 0.614561 ± 0.00572022
blocked 16                              : 0.543498 ± 0.00368265
blocked 32                              : 0.454945 ± 0.00316938
blocked 4, four accumulators            : 0.199081 ± 0.00321556
blocked 8, four accumulators            : 0.184038 ± 0.00237273
blocked 16, four accumulators           : 0.19023 ± 0.00176979
blocked 32, four accumulators           : 0.201058 ± 0.00157715

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive                                   : 0.940208 ± 0.00324212
naive, two accumulators                 : 0.473242 ± 0.00156247
naive, four accumulators                : 0.232722 ± 0.000945434
naive, eight accumulators               : 0.135833 ± 0.000889309
blocked 4                               : 0.80048 ± 0.00356678
blocked 8                               : 0.625835 ± 0.00207349
blocked 16                              : 0.56202 ± 0.00222198
blocked 32                              : 0.43137 ± 0.00096261
blocked 4, four accumulators            : 0.191418 ± 0.0010049
blocked 8, four accumulators            : 0.185261 ± 0.000645282
blocked 16, four accumulators           : 0.170599 ± 0.000746051
blocked 32, four accumulators           : 0.171061 ± 0.00052631

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive                                   : 0.813977 ± 0.00335783
naive, two accumulators                 : 0.395884 ± 0.00125107
naive, four accumulators                : 0.213328 ± 0.00728414
naive, eight accumulators               : 0.16527 ± 0.000758403
blocked 4                               : 0.61067 ± 0.00234821
blocked 8                               : 0.598957 ± 0.0023902
blocked 16                              : 0.533296 ± 0.00323165
blocked 32                              : 0.444372 ± 0.00219162
blocked 4, four accumulators            : 0.172805 ± 0.000557134
blocked 8, four accumulators            : 0.168812 ± 0.000531372
blocked 16, four accumulators           : 0.17503 ± 0.00137613
blocked 32, four accumulators           : 0.186983 ± 0.000906087

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive                                   : 1.00632 ± 0.00291581
naive, two accumulators                 : 0.551342 ± 0.00236665
naive, four accumulators                : 0.316702 ± 0.00204051
naive, eight accumulators               : 0.272891 ± 0.00309747
blocked 4                               : 0.809332 ± 0.00371117
blocked 8                               : 0.632275 ± 0.00382744
blocked 16                              : 0.56589 ± 0.00269777
blocked 32                              : 0.432145 ± 0.00118256
blocked 4, four accumulators            : 0.216822 ± 0.00308176
blocked 8, four accumulators            : 0.198626 ± 0.00188497
blocked 16, four accumulators           : 0.173258 ± 0.000935007
blocked 32, four accumulators           : 0.1746 ± 0.00083553

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive                                   : 0.989677 ± 0.000514148
naive, two accumulators                 : 0.548796 ± 0.000398729
naive, four accumulators                : 0.317634 ± 0.000657407
naive, eight accumulators               : 0.277745 ± 0.000978992
blocked 4                               : 0.815797 ± 0.000888942
blocked 8                               : 0.632707 ± 0.00048781
blocked 16                              : 0.561392 ± 0.000613884
blocked 32                              : 0.430996 ± 0.000498649
blocked 4, four accumulators            : 0.234011 ± 0.000443195
blocked 8, four accumulators            : 0.208287 ± 0.000293056
blocked 16, four accumulators           : 0.180977 ± 0.000436692
blocked 32, four accumulators           : 0.175798 ± 0.000442611
```

## Conclusion

Both blocking and the use of multiple accumulators provide a performance boost, and moreso when combined.
Blocking improves cache efficiency as each RHS vector does not need to be constantly reloaded per LHS row; rather, each RHS vector is only reloaded once per $B$ LHS rows.
Multiple accumulators improve CPU throughput, which complements the improved cache behavior.

We see that $B = 16$ with multiple accumulators performs best (or nearly so) in all tested scenarios.
Presumably, this offers the best compromise between cache efficiency and looping overhead.

Blocking with more accumulators doesn't seem to improve performance greatly so we might as well go with 4.
Too many accumulators could interfere with instruction caching and increase register pressure.
