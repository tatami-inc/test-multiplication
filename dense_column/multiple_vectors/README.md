# Dense row-major product with multiple vectors

## Strategies

The naive approach involves computing the outer product for each column of the LHS matrix and each of the multiple RHS vectors,
and then adding them to the output matrix.

The blocked approach operates on $B$ columns of the LHS matrix and $B$ RHS vectors at a time.
For each combination of row/vector in this block, we compute the dot product of its first $C$ elements against the first $C$ elements of the RHS vector.
It repeatedly adds the dot product of the next $C$ elements until all columns of the LHS matrix have been traversed.
It proceeds to the next $B$ RHS vectors until all vectors have been traversed, and then onto the next $B$ columns until the entire LHS matrix is traversed.
We test a range of different values for the $B$ given a fixed value for $BC = 1024$, i.e., a thousand elements in the cache at once.
(The actual number of elements in the cache is actually $2BC$, as we hold a block from each of the LHS and RHS.)
Even for 8-byte types like `double`, this should easily fit into a modern L1 cache. 

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
naive                                   : 0.4296 ± 0.00350459
blocked 4                               : 0.353492 ± 0.00176296
blocked 8                               : 0.343134 ± 0.000685697
blocked 16                              : 0.293872 ± 0.000998699
blocked 32                              : 0.302393 ± 0.000333362

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive                                   : 0.27907 ± 0.000941406
blocked 4                               : 0.301322 ± 0.000563974
blocked 8                               : 0.31377 ± 0.000475146
blocked 16                              : 0.280048 ± 0.000276852
blocked 32                              : 0.300783 ± 0.000677204

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive                                   : 0.936886 ± 0.0037108
blocked 4                               : 0.44801 ± 0.00124831
blocked 8                               : 0.391335 ± 0.00064324
blocked 16                              : 0.32041 ± 0.000652238
blocked 32                              : 0.307007 ± 0.000948445

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive                                   : 0.454534 ± 0.00401893
blocked 4                               : 0.356708 ± 0.00139613
blocked 8                               : 0.34547 ± 0.000779635
blocked 16                              : 0.294481 ± 0.00039965
blocked 32                              : 0.328113 ± 0.000408572

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive                                   : 0.910564 ± 0.00333136
blocked 4                               : 0.432103 ± 0.00112454
blocked 8                               : 0.380378 ± 0.000844505
blocked 16                              : 0.307828 ± 0.000541428
blocked 32                              : 0.309883 ± 0.000960286

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive                                   : 0.430597 ± 0.000782266
blocked 4                               : 0.359917 ± 0.000576835
blocked 8                               : 0.31622 ± 0.00038035
blocked 16                              : 0.297492 ± 0.000494627
blocked 32                              : 0.313096 ± 0.000544973

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive                                   : 0.683692 ± 0.00433952
blocked 4                               : 0.471634 ± 0.000816473
blocked 8                               : 0.402539 ± 0.000694061
blocked 16                              : 0.363027 ± 0.000561223
blocked 32                              : 0.352103 ± 0.00133258
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive                                   : 0.179199 ± 0.00265633
blocked 4                               : 0.169975 ± 0.000302476
blocked 8                               : 0.166017 ± 0.000965121
blocked 16                              : 0.176199 ± 0.000482495
blocked 32                              : 0.202802 ± 0.000693281

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive                                   : 0.146119 ± 0.000425042
blocked 4                               : 0.154319 ± 0.000208239
blocked 8                               : 0.155796 ± 0.000182105
blocked 16                              : 0.171286 ± 0.000553479
blocked 32                              : 0.204363 ± 0.000571149

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive                                   : 0.459653 ± 0.00417505
blocked 4                               : 0.230051 ± 0.000930913
blocked 8                               : 0.191584 ± 0.000782923
blocked 16                              : 0.188382 ± 0.000337226
blocked 32                              : 0.206484 ± 0.000350733

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive                                   : 0.17462 ± 0.0027113
blocked 4                               : 0.162268 ± 0.00137978
blocked 8                               : 0.161566 ± 0.000939838
blocked 16                              : 0.175219 ± 0.000446092
blocked 32                              : 0.211444 ± 0.00484783

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive                                   : 0.447046 ± 0.00383157
blocked 4                               : 0.214538 ± 0.000650747
blocked 8                               : 0.183484 ± 0.000417645
blocked 16                              : 0.181432 ± 0.000893799
blocked 32                              : 0.205161 ± 0.000510374

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive                                   : 0.261018 ± 0.000413913
blocked 4                               : 0.202111 ± 0.000540232
blocked 8                               : 0.186943 ± 0.000262083
blocked 16                              : 0.196607 ± 0.000233292
blocked 32                              : 0.226883 ± 0.000472973

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive                                   : 0.315187 ± 0.0032727
blocked 4                               : 0.238317 ± 0.000605378
blocked 8                               : 0.219454 ± 0.000821138
blocked 16                              : 0.220106 ± 0.000792842
blocked 32                              : 0.248734 ± 0.000551626
```

## Conclusion

Blocking provides a clear improvement in many scenarios.
I assume that cache utilitization is more efficient as each RHS vector does not need to be constantly reloaded per LHS row;
rather, each RHS vector is only reloaded once per $B$ LHS rows.

We see that $B = 16$ with multiple accumulators performs best (or nearly so) in all tested scenarios.
Presumably, this offers the best compromise between cache efficiency and looping overhead.
