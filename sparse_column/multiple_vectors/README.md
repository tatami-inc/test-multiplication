# Sparse column-major LHS, multiple vectors RHS

## Strategies

All strategies must iterate across consecutive columns of the LHS matrix in the outermost loop.
We will be loading columns on demand via the **tatami** interface, so the full matrix will not be available for random access.

### Naive 

For each column $i$ of the LHS matrix, we compute the outer product with the $i$-th element of each of multiple RHS vectors.
Specifically, we loop over the RHS vectors, and for RHS vector $h$, we perform a sparse vector multiply-add to the $h$-th output vector,
using the $i$-th element of the $h$-th RHS vector as the scaling factor.
We repeat this for all $i$ until all columns of the LHS matrix are traversed.

### Blocking 

We follow the recommendation in the "Blocking to cache the dense vector" section of [`general/README.md`](../../general/README.md).
First, we load a block of $B$ LHS columns.
For each block, we loop over the RHS vectors, and for each RHS vector $h$, we loop again over our block of LHS columns.
For each LHS column in our block, we perform a sparse vector multiply-add to the $h$-th output vector as described above.
Once all RHS vectors are processed, we move onto the next block of $B$ columns, and so on until all columns are traversed.
The aim is to keep the most of the dense vector in cache so that it can be easily re-used with multiple sparse vectors.

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
naive                                             : 0.273452 ± 0.00784225
blocked 4                                         : 0.137967 ± 0.00490301
blocked 8                                         : 0.107883 ± 0.00319729
blocked 16                                        : 0.0864737 ± 0.00359922
blocked 32                                        : 0.0722515 ± 0.00222882

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive                                             : 0.138121 ± 0.00343921
blocked 4                                         : 0.0844311 ± 0.00235734
blocked 8                                         : 0.0730783 ± 0.00216751
blocked 16                                        : 0.067306 ± 0.00214921
blocked 32                                        : 0.0658963 ± 0.00192802

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive                                             : 0.763477 ± 0.00948505
blocked 4                                         : 0.30524 ± 0.00366934
blocked 8                                         : 0.205768 ± 0.00921586
blocked 16                                        : 0.141361 ± 0.00429071
blocked 32                                        : 0.102201 ± 0.00187041

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive                                             : 0.256672 ± 0.00719586
blocked 4                                         : 0.16798 ± 0.00543757
blocked 8                                         : 0.133577 ± 0.00530372
blocked 16                                        : 0.115845 ± 0.00287374
blocked 32                                        : 0.104216 ± 0.0026363

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive                                             : 0.739376 ± 0.00633632
blocked 4                                         : 0.330525 ± 0.00492284
blocked 8                                         : 0.221951 ± 0.00216609
blocked 16                                        : 0.160642 ± 0.00201894
blocked 32                                        : 0.124209 ± 0.00150974

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive                                             : 0.203631 ± 0.00118195
blocked 4                                         : 0.102761 ± 0.000203452
blocked 8                                         : 0.0868192 ± 0.000519846
blocked 16                                        : 0.0847582 ± 0.000118274
blocked 32                                        : 0.0839406 ± 0.000243594

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive                                             : 0.389436 ± 0.00298933
blocked 4                                         : 0.239421 ± 0.00250821
blocked 8                                         : 0.181484 ± 0.000800922
blocked 16                                        : 0.140982 ± 0.000679225
blocked 32                                        : 0.112465 ± 0.000347825
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive                                             : 0.12393 ± 0.00506827
blocked 4                                         : 0.0906377 ± 0.00154389
blocked 8                                         : 0.0857654 ± 0.00133574
blocked 16                                        : 0.0755464 ± 0.0011454
blocked 32                                        : 0.0713837 ± 0.000257978

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive                                             : 0.0857965 ± 0.000762112
blocked 4                                         : 0.0652648 ± 0.00049677
blocked 8                                         : 0.0612928 ± 0.000704686
blocked 16                                        : 0.0592272 ± 0.000481518
blocked 32                                        : 0.0591727 ± 0.000602528

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive                                             : 0.421019 ± 0.00467478
blocked 4                                         : 0.159768 ± 0.00411405
blocked 8                                         : 0.115853 ± 0.00285133
blocked 16                                        : 0.0901528 ± 0.00169497
blocked 32                                        : 0.0780474 ± 0.0012184

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive                                             : 0.113475 ± 0.00218876
blocked 4                                         : 0.0895261 ± 0.00272559
blocked 8                                         : 0.0851966 ± 0.00164004
blocked 16                                        : 0.0821211 ± 0.000256602
blocked 32                                        : 0.0761431 ± 0.000602769

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive                                             : 0.41871 ± 0.00636905
blocked 4                                         : 0.175173 ± 0.00569336
blocked 8                                         : 0.125085 ± 0.00194848
blocked 16                                        : 0.100826 ± 0.00262
blocked 32                                        : 0.0829158 ± 0.00138909

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive                                             : 0.148069 ± 0.00170518
blocked 4                                         : 0.0896907 ± 0.000580572
blocked 8                                         : 0.0753749 ± 0.000861014
blocked 16                                        : 0.0693922 ± 0.000615171
blocked 32                                        : 0.0711292 ± 0.0012345

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive                                             : 0.189382 ± 0.00637699
blocked 4                                         : 0.140249 ± 0.00835579
blocked 8                                         : 0.113396 ± 0.00540009
blocked 16                                        : 0.0993958 ± 0.00447489
blocked 32                                        : 0.0908611 ± 0.00333428
```

## Conclusion

Blocking is pretty helpful, we should do it.
Seems like $B = 16$ is generally a good choice;
it provides almost all of the benefit without double the memory usage of $B = 32$.
