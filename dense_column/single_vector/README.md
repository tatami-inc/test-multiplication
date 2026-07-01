# Dense column-major matrix, single vector product

## Overview 

All strategies must iterate across consecutive columns of the LHS matrix in the outermost loop.
We will be loading columns on demand via the **tatami** interface, so the full matrix will not be available for random access.

### Naive 

The naive approach involves taking each column of the LHS matrix, scaling it by the corresponding RHS value, and then adding it to the output vector.
This vector multiply-add involves contiguous memory accesses and is very amenable to auto-vectorization.

### Blocking

We consider a $C$-by-$B$ LHS submatrix and a $B$-by-1 RHS submatrix (i.e., just a part of the RHS vector).
For each RHS column in this block, we perform a vector multiply-add with the output vector, using the corresponding LHS value as a scaling factor.
We repeat this for the next $C$ elements until all LHS rows are traversed.
Then, we proceed to the next block of $B$ LHS columns until all columns have been traversed.

The idea is to keep the parts of the RHS vector in cache for each block of $B$ columns, such that we don't have to reload it for every single row.
We test a range of different values for the $B$ given a fixed value for $BC = 1024$, i.e., a thousand elements in the cache at once.
(Technically, our RHS "submatrix" only has one column so we could reasonably increase $BC$ while still fitting in a typical L1 cache.
But for simplicity's sake, we will stick with the limit used in the other tests.)
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
$ ./build/test
Results for 10000 x 5000
naive                                   : 0.0478973 ± 0.00283251
blocked 4                               : 0.0481584 ± 0.00193468
blocked 8                               : 0.0451068 ± 0.00175856
blocked 16                              : 0.0477027 ± 0.00172159
blocked 32                              : 0.0465743 ± 0.00202951

$ ./build/test -r 200 -c 50000
Results for 200 x 50000
naive                                   : 0.00750066 ± 0.000780113
blocked 4                               : 0.00942407 ± 0.00116771
blocked 8                               : 0.0133052 ± 0.000989436
blocked 16                              : 0.0098911 ± 0.00049167
blocked 32                              : 0.013592 ± 0.00105466

$ ./build/test -r 50000 -c 200
Results for 50000 x 200
naive                                   : 0.0084623 ± 0.00127567
blocked 4                               : 0.0101603 ± 0.00161221
blocked 8                               : 0.0108176 ± 0.00152984
blocked 16                              : 0.012082 ± 0.00145271
blocked 32                              : 0.013397 ± 0.00120539
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float
Results for 10000 x 5000
naive                                   : 0.0230941 ± 0.00189858
blocked 4                               : 0.0264096 ± 0.000971959
blocked 8                               : 0.0253525 ± 0.00154191
blocked 16                              : 0.0250269 ± 0.00123717
blocked 32                              : 0.0261859 ± 0.00100457

$ ./build/test_float -r 200 -c 50000
Results for 200 x 50000
naive                                   : 0.00548461 ± 0.00102968
blocked 4                               : 0.00681155 ± 0.00103824
blocked 8                               : 0.00548724 ± 0.000560026
blocked 16                              : 0.00602658 ± 0.000704428
blocked 32                              : 0.0052365 ± 0.000575123

$ ./build/test_float -r 50000 -c 200
Results for 50000 x 200
naive                                   : 0.00414278 ± 0.000494245
blocked 4                               : 0.00742659 ± 0.00109427
blocked 8                               : 0.00584028 ± 0.000868928
blocked 16                              : 0.00517399 ± 0.000652772
blocked 32                              : 0.00532506 ± 0.00072966
```

## Conclusion

Blocking doesn't really help much here.
Any cache benefits seem to be more than offset by the extra looping overhead.
