# Sparse row-major LHS, multiple vectors RHS

## Strategies

All strategies must iterate across consecutive rows of the LHS matrix in the outermost loop.
We will be loading rows on demand via the **tatami** interface, so the full matrix will not be available for random access.

### Naive

For each LHS row $i$, we iterate across the RHS vectors.
For each RHS vector $h$, we compute the dot product of LHS row $i$ against $h$ and store the result in the $i$-th entry of the $h$-th output vector.

### Accumulators

This is the same as the naive approach, except that the dot product is computed with multiple accumulators.
The idea is to break dependency chains in the CPU's instruction pipeline by allowing multiple accumulations to occur in parallel.
It also provides some opportunities for auto-vectorization of the sum.

### Blocking

We follow the recommendations mentioned in the "Blocking to cache the dense vector" in [`general/README.md`](../../general/README.md).

First, we load in a block of $B$ sparse LHS rows.
For each RHS vector $h$, we compute the dot product with each of the LHS rows $i$ in the block, storing the result in the $i$-th entry of the $h$-th output vector.
This is repeated for the next block of $B$ LHS rows until the entire LHS matrix is traversed.
The idea is to keep as much of each RHS vector in the cache as possible.

### Blocking with accumulators

The blocking approach can also use multiple accumulators in each of its partial dot products.
To keep things simple, we only consider the performance of blocking with 4 accumulators,
given that it's better than 2 and only slightly worse than 8 in the naive approach.

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
naive                                             : 0.239534 ± 0.0112955
2 accumulators                                    : 0.242318 ± 0.00665925
4 accumulators                                    : 0.212299 ± 0.00589962
8 accumulators                                    : 0.232038 ± 0.00691499
blocked (B = 4)                                   : 0.150769 ± 0.00388108
blocked (B = 8)                                   : 0.12777 ± 0.00514117
blocked (B = 16)                                  : 0.111567 ± 0.00291508
blocked (B = 32)                                  : 0.103708 ± 0.00195957
blocked (B = 4), 4 accumulators                   : 0.118643 ± 0.00326225
blocked (B = 8), 4 accumulators                   : 0.0921437 ± 0.00534881
blocked (B = 16), 4 accumulators                  : 0.07058 ± 0.00218664
blocked (B = 32), 4 accumulators                  : 0.056724 ± 0.000210763

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive                                             : 0.229367 ± 0.00360619
2 accumulators                                    : 0.252649 ± 0.00599014
4 accumulators                                    : 0.219583 ± 0.00516234
8 accumulators                                    : 0.240714 ± 0.00737905
blocked (B = 4)                                   : 0.150848 ± 0.00289895
blocked (B = 8)                                   : 0.134204 ± 0.00239428
blocked (B = 16)                                  : 0.12329 ± 0.00165561
blocked (B = 32)                                  : 0.114943 ± 0.00106567
blocked (B = 4), 4 accumulators                   : 0.130564 ± 0.00348676
blocked (B = 8), 4 accumulators                   : 0.104549 ± 0.00277402
blocked (B = 16), 4 accumulators                  : 0.0845042 ± 0.00223196
blocked (B = 32), 4 accumulators                  : 0.0743997 ± 0.00170618

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive                                             : 0.306568 ± 0.0061944
2 accumulators                                    : 0.321876 ± 0.00577633
4 accumulators                                    : 0.328871 ± 0.00507749
8 accumulators                                    : 0.313548 ± 0.00656646
blocked (B = 4)                                   : 0.155289 ± 0.00365538
blocked (B = 8)                                   : 0.112967 ± 0.00138985
blocked (B = 16)                                  : 0.0921518 ± 0.00154091
blocked (B = 32)                                  : 0.0776362 ± 0.00152533
blocked (B = 4), 4 accumulators                   : 0.179185 ± 0.00273608
blocked (B = 8), 4 accumulators                   : 0.128477 ± 0.00250349
blocked (B = 16), 4 accumulators                  : 0.105029 ± 0.00158726
blocked (B = 32), 4 accumulators                  : 0.0915734 ± 0.00127477

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive                                             : 0.0977307 ± 0.000121769
2 accumulators                                    : 0.0955005 ± 0.000191911
4 accumulators                                    : 0.0870518 ± 0.000159248
8 accumulators                                    : 0.0933729 ± 0.000201395
blocked (B = 4)                                   : 0.0905084 ± 0.00023115
blocked (B = 8)                                   : 0.0885866 ± 0.000139731
blocked (B = 16)                                  : 0.0875149 ± 0.000123734
blocked (B = 32)                                  : 0.0878445 ± 0.000433951
blocked (B = 4), 4 accumulators                   : 0.0578517 ± 6.52973e-05
blocked (B = 8), 4 accumulators                   : 0.0487341 ± 6.65614e-05
blocked (B = 16), 4 accumulators                  : 0.0448579 ± 0.000112011
blocked (B = 32), 4 accumulators                  : 0.0435579 ± 6.76307e-05

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive                                             : 0.126518 ± 0.000783699
2 accumulators                                    : 0.132081 ± 0.000565721
4 accumulators                                    : 0.142351 ± 0.00070075
8 accumulators                                    : 0.136792 ± 0.000680093
blocked (B = 4)                                   : 0.083104 ± 0.000872533
blocked (B = 8)                                   : 0.0740875 ± 0.000784814
blocked (B = 16)                                  : 0.0681843 ± 0.000702151
blocked (B = 32)                                  : 0.0671435 ± 0.00066624
blocked (B = 4), 4 accumulators                   : 0.104817 ± 0.00096392
blocked (B = 8), 4 accumulators                   : 0.0903113 ± 0.000925393
blocked (B = 16), 4 accumulators                  : 0.0837639 ± 0.000842475
blocked (B = 32), 4 accumulators                  : 0.0819698 ± 0.00109471

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive                                             : 0.451574 ± 0.000357918
2 accumulators                                    : 0.459364 ± 0.000464554
4 accumulators                                    : 0.4443 ± 0.000665183
8 accumulators                                    : 0.454165 ± 0.00048221
blocked (B = 4)                                   : 0.223284 ± 0.00233943
blocked (B = 8)                                   : 0.172737 ± 0.000279924
blocked (B = 16)                                  : 0.142522 ± 0.000265937
blocked (B = 32)                                  : 0.125646 ± 0.000322397
blocked (B = 4), 4 accumulators                   : 0.206332 ± 0.000254734
blocked (B = 8), 4 accumulators                   : 0.145171 ± 0.000358425
blocked (B = 16), 4 accumulators                  : 0.110232 ± 0.000206634
blocked (B = 32), 4 accumulators                  : 0.090316 ± 0.000126692

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive                                             : 0.499198 ± 0.0254992
2 accumulators                                    : 0.522714 ± 0.0389661
4 accumulators                                    : 0.46793 ± 0.00351446
8 accumulators                                    : 0.488409 ± 0.00523938
blocked (B = 4)                                   : 0.218625 ± 0.00328404
blocked (B = 8)                                   : 0.168208 ± 0.00306306
blocked (B = 16)                                  : 0.131736 ± 0.00206854
blocked (B = 32)                                  : 0.114602 ± 0.00234414
blocked (B = 4), 4 accumulators                   : 0.208699 ± 0.00437172
blocked (B = 8), 4 accumulators                   : 0.136617 ± 0.00388331
blocked (B = 16), 4 accumulators                  : 0.0976068 ± 0.00211879
blocked (B = 32), 4 accumulators                  : 0.0732898 ± 0.00211574
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive                                             : 0.121776 ± 0.0020113
2 accumulators                                    : 0.110728 ± 0.00357415
4 accumulators                                    : 0.106288 ± 0.00628164
8 accumulators                                    : 0.111073 ± 0.00278971
blocked (B = 4)                                   : 0.10661 ± 0.00627095
blocked (B = 8)                                   : 0.0999482 ± 0.00234612
blocked (B = 16)                                  : 0.0940516 ± 0.00135539
blocked (B = 32)                                  : 0.0922287 ± 0.00118436
blocked (B = 4), 4 accumulators                   : 0.0608438 ± 0.00173353
blocked (B = 8), 4 accumulators                   : 0.0517113 ± 0.00159384
blocked (B = 16), 4 accumulators                  : 0.0446547 ± 0.000392405
blocked (B = 32), 4 accumulators                  : 0.0417533 ± 0.000172627

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive                                             : 0.108703 ± 0.00238643
2 accumulators                                    : 0.0806034 ± 0.0015088
4 accumulators                                    : 0.0746706 ± 0.00236114
8 accumulators                                    : 0.0927483 ± 0.00812164
blocked (B = 4)                                   : 0.106986 ± 0.00231332
blocked (B = 8)                                   : 0.114939 ± 0.00467948
blocked (B = 16)                                  : 0.104701 ± 0.00111056
blocked (B = 32)                                  : 0.102177 ± 0.000971926
blocked (B = 4), 4 accumulators                   : 0.0619964 ± 0.00323678
blocked (B = 8), 4 accumulators                   : 0.0612671 ± 0.0018646
blocked (B = 16), 4 accumulators                  : 0.0608381 ± 0.00262758
blocked (B = 32), 4 accumulators                  : 0.0556463 ± 0.00133346

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive                                             : 0.128394 ± 0.00148198
2 accumulators                                    : 0.128729 ± 0.00460693
4 accumulators                                    : 0.129344 ± 0.00287177
8 accumulators                                    : 0.145972 ± 0.00532839
blocked (B = 4)                                   : 0.0911002 ± 0.00433142
blocked (B = 8)                                   : 0.080359 ± 0.00216076
blocked (B = 16)                                  : 0.0710156 ± 0.00106145
blocked (B = 32)                                  : 0.0657845 ± 0.000768431
blocked (B = 4), 4 accumulators                   : 0.100082 ± 0.0055679
blocked (B = 8), 4 accumulators                   : 0.0851881 ± 0.00275537
blocked (B = 16), 4 accumulators                  : 0.0784358 ± 0.00257811
blocked (B = 32), 4 accumulators                  : 0.0700711 ± 0.000548125

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive                                             : 0.0907024 ± 0.000231573
2 accumulators                                    : 0.0679661 ± 0.000264067
4 accumulators                                    : 0.0596752 ± 0.000213813
8 accumulators                                    : 0.0722238 ± 0.000298685
blocked (B = 4)                                   : 0.0904111 ± 0.000584034
blocked (B = 8)                                   : 0.0894397 ± 0.000490403
blocked (B = 16)                                  : 0.0881504 ± 0.000246554
blocked (B = 32)                                  : 0.0888669 ± 0.000370406
blocked (B = 4), 4 accumulators                   : 0.0438598 ± 0.000309629
blocked (B = 8), 4 accumulators                   : 0.0396818 ± 0.000303003
blocked (B = 16), 4 accumulators                  : 0.0379876 ± 0.000322006
blocked (B = 32), 4 accumulators                  : 0.0372349 ± 0.000102376

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive                                             : 0.0921966 ± 0.000305964
2 accumulators                                    : 0.0911264 ± 0.000535258
4 accumulators                                    : 0.0963545 ± 0.000218466
8 accumulators                                    : 0.103585 ± 0.000466893
blocked (B = 4)                                   : 0.0714229 ± 0.000450231
blocked (B = 8)                                   : 0.0650551 ± 0.000222925
blocked (B = 16)                                  : 0.062963 ± 0.000304407
blocked (B = 32)                                  : 0.0623845 ± 0.00030112
blocked (B = 4), 4 accumulators                   : 0.0796034 ± 0.000287016
blocked (B = 8), 4 accumulators                   : 0.0713973 ± 0.000504899
blocked (B = 16), 4 accumulators                  : 0.0675427 ± 0.000277836
blocked (B = 32), 4 accumulators                  : 0.0653929 ± 0.000119211

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive                                             : 0.255637 ± 0.00498401
2 accumulators                                    : 0.260414 ± 0.00599941
4 accumulators                                    : 0.246663 ± 0.00448366
8 accumulators                                    : 0.254853 ± 0.00701283
blocked (B = 4)                                   : 0.144581 ± 0.00142047
blocked (B = 8)                                   : 0.129018 ± 0.00295314
blocked (B = 16)                                  : 0.118068 ± 0.00286128
blocked (B = 32)                                  : 0.108598 ± 0.000597594
blocked (B = 4), 4 accumulators                   : 0.105714 ± 0.000498539
blocked (B = 8), 4 accumulators                   : 0.0868403 ± 0.00355302
blocked (B = 16), 4 accumulators                  : 0.0695696 ± 0.0011526
blocked (B = 32), 4 accumulators                  : 0.0609156 ± 0.000756398

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive                                             : 0.268343 ± 0.00334741
2 accumulators                                    : 0.292079 ± 0.0210781
4 accumulators                                    : 0.268486 ± 0.00618299
8 accumulators                                    : 0.26203 ± 0.00522957
blocked (B = 4)                                   : 0.153899 ± 0.00753411
blocked (B = 8)                                   : 0.12095 ± 0.00317249
blocked (B = 16)                                  : 0.105397 ± 0.00069364
blocked (B = 32)                                  : 0.100899 ± 0.00219659
blocked (B = 4), 4 accumulators                   : 0.113661 ± 0.00421226
blocked (B = 8), 4 accumulators                   : 0.0803986 ± 0.00191894
blocked (B = 16), 4 accumulators                  : 0.0646812 ± 0.00202839
blocked (B = 32), 4 accumulators                  : 0.0535294 ± 0.00147812
```

## Conclusion

The use of multiple accumulators usually provides a slight benefit over the naive calculation.
However, the improvement is much weaker and less consistent compared to the [single vector](../single_vector) case.
This is a bit perplexing; perhaps memory access becomes a greater bottleneck when each RHS vector needs to be reloaded into cache.
By comparison, much of the single RHS vector can be held in cache and re-used in the single vector case, which exposes the accumulation as the rate-limiting factor.

Blocking is mostly unhelpful here. 
Presumably the extra looping overhead outweighs any improvement in cache re-use.
