# Dense column-major LHS, multiple vectors RHS

## Strategies

All strategies must iterate across consecutive columns of the LHS matrix in the outermost loop.
We will be loading columns on demand via the **tatami** interface, so the full matrix will not be available for random access.

### Naive 

For each column $i$ of the LHS matrix, we compute the outer product with the $i$-th element of each of our multiple RHS vectors.
Specifically, we perform a vector multiply-add of LHS column $i$ to the $h$-th output vector, using the $i$-th entry of the $h$-th RHS vector as a scaling factor.
We repeat this for all $i$ until all columns of the LHS matrix are traversed.

### Blocking 

We consider $C$-by-$B$ LHS and output submatrices and a $B$-by-$B$ RHS submatrix.
In this set of submatrices, each LHS row corresponds to an output row, each LHS column corresponds to an RHS row, and each RHS column corresponds to an output column.

For each column $i$ in the LHS submatrix, we iterate over the block of RHS columns.
For each RHS column $h$, we perform a vector multiply-add of LHS column $i$ to output column $h$ using the RHS entry $(i, h)$ as a scaling factor.
We repeat this for all valid sets of submatrices until the full matrix product is computed.
When choosing the next set of submatrices, the RHS/output columns are the fastest-changing dimension while the LHS columns are the slowest.

The hope is that these submatrices are small enough to keep in cache for fast access.
We test a range of different values for the $B$ given a fixed value for $BC = 1024$. 
Check out [`general/README.md`](../../general/README.md) for more details.

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
blocked (B = 4)                         : 0.353492 ± 0.00176296
blocked (B = 8)                         : 0.343134 ± 0.000685697
blocked (B = 16)                        : 0.293872 ± 0.000998699
blocked (B = 32)                        : 0.302393 ± 0.000333362

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive                                   : 0.27907 ± 0.000941406
blocked (B = 4)                         : 0.301322 ± 0.000563974
blocked (B = 8)                         : 0.31377 ± 0.000475146
blocked (B = 16)                        : 0.280048 ± 0.000276852
blocked (B = 32)                        : 0.300783 ± 0.000677204

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive                                   : 0.936886 ± 0.0037108
blocked (B = 4)                         : 0.44801 ± 0.00124831
blocked (B = 8)                         : 0.391335 ± 0.00064324
blocked (B = 16)                        : 0.32041 ± 0.000652238
blocked (B = 32)                        : 0.307007 ± 0.000948445

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive                                   : 0.454534 ± 0.00401893
blocked (B = 4)                         : 0.356708 ± 0.00139613
blocked (B = 8)                         : 0.34547 ± 0.000779635
blocked (B = 16)                        : 0.294481 ± 0.00039965
blocked (B = 32)                        : 0.328113 ± 0.000408572

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive                                   : 0.910564 ± 0.00333136
blocked (B = 4)                         : 0.432103 ± 0.00112454
blocked (B = 8)                         : 0.380378 ± 0.000844505
blocked (B = 16)                        : 0.307828 ± 0.000541428
blocked (B = 32)                        : 0.309883 ± 0.000960286

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive                                   : 0.430597 ± 0.000782266
blocked (B = 4)                         : 0.359917 ± 0.000576835
blocked (B = 8)                         : 0.31622 ± 0.00038035
blocked (B = 16)                        : 0.297492 ± 0.000494627
blocked (B = 32)                        : 0.313096 ± 0.000544973

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive                                   : 0.683692 ± 0.00433952
blocked (B = 4)                         : 0.471634 ± 0.000816473
blocked (B = 8)                         : 0.402539 ± 0.000694061
blocked (B = 16)                        : 0.363027 ± 0.000561223
blocked (B = 32)                        : 0.352103 ± 0.00133258
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive                                   : 0.179199 ± 0.00265633
blocked (B = 4)                         : 0.169975 ± 0.000302476
blocked (B = 8)                         : 0.166017 ± 0.000965121
blocked (B = 16)                        : 0.176199 ± 0.000482495
blocked (B = 32)                        : 0.202802 ± 0.000693281

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive                                   : 0.146119 ± 0.000425042
blocked (B = 4)                         : 0.154319 ± 0.000208239
blocked (B = 8)                         : 0.155796 ± 0.000182105
blocked (B = 16)                        : 0.171286 ± 0.000553479
blocked (B = 32)                        : 0.204363 ± 0.000571149

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive                                   : 0.459653 ± 0.00417505
blocked (B = 4)                         : 0.230051 ± 0.000930913
blocked (B = 8)                         : 0.191584 ± 0.000782923
blocked (B = 16)                        : 0.188382 ± 0.000337226
blocked (B = 32)                        : 0.206484 ± 0.000350733

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive                                   : 0.17462 ± 0.0027113
blocked (B = 4)                         : 0.162268 ± 0.00137978
blocked (B = 8)                         : 0.161566 ± 0.000939838
blocked (B = 16)                        : 0.175219 ± 0.000446092
blocked (B = 32)                        : 0.211444 ± 0.00484783

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive                                   : 0.447046 ± 0.00383157
blocked (B = 4)                         : 0.214538 ± 0.000650747
blocked (B = 8)                         : 0.183484 ± 0.000417645
blocked (B = 16)                        : 0.181432 ± 0.000893799
blocked (B = 32)                        : 0.205161 ± 0.000510374

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive                                   : 0.261018 ± 0.000413913
blocked (B = 4)                         : 0.202111 ± 0.000540232
blocked (B = 8)                         : 0.186943 ± 0.000262083
blocked (B = 16)                        : 0.196607 ± 0.000233292
blocked (B = 32)                        : 0.226883 ± 0.000472973

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive                                   : 0.315187 ± 0.0032727
blocked (B = 4)                         : 0.238317 ± 0.000605378
blocked (B = 8)                         : 0.219454 ± 0.000821138
blocked (B = 16)                        : 0.220106 ± 0.000792842
blocked (B = 32)                        : 0.248734 ± 0.000551626
```

## Conclusion

Blocking provides a clear improvement in many scenarios.
I assume that cache utilitization is more efficient as each RHS vector does not need to be constantly reloaded per LHS row;
rather, each RHS vector is only reloaded once per $B$ LHS rows.

We see that $B = 16$ with multiple accumulators performs best (or nearly so) in all tested scenarios.
Presumably, this offers the best compromise between cache efficiency and looping overhead.
