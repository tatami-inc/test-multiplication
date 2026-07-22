# Implementing multiple accumulators

Here we test various implementations for multiple accumulators.

- `loop`: just loops across the accumulators, trusting the compiler to unroll as needed.
- `manually unroll`: manually unrolls the accumulator loop via templates.
- `peeled`: peels off the first loop iteration to avoid an unnecessary addition to the all-zero `dots`. 
- `vectorized epilogue`: add to the `dots` in the epilogue loop, which is most amenable to vectorization.
- `recursive sum`: use a recursive algorithm to sum the accumulators.

## Instructions

To build the test executables, use the usual CMake process:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This compiles with various choices for the number of accumulators (4, 8 and 16).

## Single vector

We examine the effect on a dense row LHS with a single vector RHS and no blocking.
We'll use a small number of LHS columns so that we can observe some differences between the implementations;
otherwise, the runtime is dominated by the common main loop.

```console
$ ./build/single4 -r 1000000 -c 50
Results for 1000000 x 50
loop                                    : 0.0291835 ± 0.000241569
manually unrolled                       : 0.0291729 ± 7.39586e-05
peeled                                  : 0.029454 ± 0.000125082
vectorized epilogue                     : 0.0295726 ± 0.000104773
recursive sum                           : 0.0290335 ± 9.75734e-05

$ ./build/single4 -r 100000 -c 99
Results for 100000 x 99
loop                                    : 0.00539249 ± 2.94704e-05
manually unrolled                       : 0.00542473 ± 7.87283e-05
peeled                                  : 0.00534832 ± 2.57154e-05
vectorized epilogue                     : 0.00542633 ± 2.72348e-05
recursive sum                           : 0.00532815 ± 2.62941e-05

$ ./build/single8 -r 1000000 -c 50
Results for 1000000 x 50
loop                                    : 0.0309609 ± 0.000710464
manually unrolled                       : 0.0296608 ± 0.000135104
peeled                                  : 0.030441 ± 0.000121915
vectorized epilogue                     : 0.03031 ± 3.01516e-05
recursive sum                           : 0.0292679 ± 8.22193e-05

$ ./build/single8 -r 100000 -c 99
Results for 100000 x 99
loop                                    : 0.00549083 ± 0.000106359
manually unrolled                       : 0.00543404 ± 3.98142e-05
peeled                                  : 0.00544832 ± 3.60815e-05
vectorized epilogue                     : 0.00553122 ± 3.35395e-05
recursive sum                           : 0.00536924 ± 4.08273e-05

$ ./build/single16 -r 1000000 -c 50
Results for 1000000 x 50
loop                                    : 0.0312693 ± 0.000104015
manually unrolled                       : 0.0311502 ± 8.94648e-05
peeled                                  : 0.0315175 ± 0.000141862
vectorized epilogue                     : 0.0370732 ± 0.00023939
recursive sum                           : 0.0300897 ± 6.07423e-05

$ ./build/single16 -r 100000 -c 99
Results for 100000 x 99
loop                                    : 0.00552894 ± 2.4821e-05
manually unrolled                       : 0.00554947 ± 3.29621e-05
peeled                                  : 0.0057172 ± 7.61668e-05
vectorized epilogue                     : 0.0059612 ± 1.66815e-05
recursive sum                           : 0.00539817 ± 3.20029e-05
```

## Blocked 

We examine the effect on a dense row LHS with a dense column RHS and blocking with 16-by-64 submatrices.
Again, we'll use a small number of LHS columns so that we can observe some differences between the implementations.

```console
$ ./build/blocked4 -r 10000 -c 50 -H 1000
Results for 10000 x 50 x 1000
loop                                    : 0.125771 ± 0.000385737
manually unrolled                       : 0.126086 ± 0.000397907
peeled                                  : 0.131595 ± 0.00172411
vectorized epilogue                     : 0.144718 ± 0.00207583
recursive sum                           : 0.125943 ± 0.00112817

$ ./build/blocked4 -r 10000 -c 99 -H 1000
Results for 10000 x 99 x 1000
loop                                    : 0.22556 ± 0.00123396
manually unrolled                       : 0.225075 ± 0.00097169
peeled                                  : 0.237933 ± 0.000444414
vectorized epilogue                     : 0.242627 ± 0.000613019
recursive sum                           : 0.22381 ± 0.00105779

$ ./build/blocked8 -r 10000 -c 50 -H 1000
Results for 10000 x 50 x 1000
loop                                    : 0.13565 ± 0.000445148
manually unrolled                       : 0.135042 ± 0.0003263
peeled                                  : 0.14746 ± 0.000420035
vectorized epilogue                     : 0.14798 ± 0.00102321
recursive sum                           : 0.114978 ± 0.000221034

$ ./build/blocked8 -r 10000 -c 99 -H 1000
Results for 10000 x 99 x 1000
loop                                    : 0.249081 ± 0.000596361
manually unrolled                       : 0.249857 ± 0.00109592
peeled                                  : 0.283001 ± 0.0012302
vectorized epilogue                     : 0.269705 ± 0.000650635
recursive sum                           : 0.213554 ± 0.000485496

$ ./build/blocked16 -r 10000 -c 50 -H 1000
Results for 10000 x 50 x 1000
loop                                    : 0.179841 ± 0.0021922
manually unrolled                       : 0.180437 ± 0.0043776
peeled                                  : 0.190283 ± 0.00387688
vectorized epilogue                     : 0.273451 ± 0.00180517
recursive sum                           : 0.117808 ± 0.00039584

$ ./build/blocked16 -r 10000 -c 99 -H 1000
Results for 10000 x 99 x 1000
loop                                    : 0.339109 ± 0.000107241
manually unrolled                       : 0.334759 ± 0.000397949
peeled                                  : 0.368455 ± 0.000124964
vectorized epilogue                     : 0.463111 ± 0.000280311
recursive sum                           : 0.226652 ± 0.000179248
```

## Conclusions

The recursive sum is the most consistent improvement.
For lower number of accumulators, it is not worse than `loop`, while it is much better for higher numbers of accumulators.

Manual unrolling very little no effect, probably because GCC unrolls the loop in `loop` at `-O3` anyway.
Some inspection on Godbolt suggests that the unrolled loop does a few more (unnecessary) stores. 

Peeling doesn't help much and is a pessimization with blocking, probably because of the extra conditional introducing a pipeline stall. 

Vectorized epilogue is pretty bad.
This is partially because the remainder is pretty small in all tested cases, but having a large remainder doesn't help either:

```console
$ ./build/blocked16 -r 10000 -c 95 -H 1000
Results for 10000 x 95 x 1000
loop                                    : 0.386967 ± 0.000174965
manually unrolled                       : 0.3809 ± 0.000590987
peeled                                  : 0.388781 ± 0.000879457
vectorized epilogue                     : 0.46018 ± 0.000804861
recursive sum                           : 0.26792 ± 0.000539519
```
I think the real reason is because the vectorized epilogue introduces a dependency on the accumulator array,
such that it cannot be executed out of order with the end of the main loop.
