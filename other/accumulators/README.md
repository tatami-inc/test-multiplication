# Implementing multiple accumulators

Here we test various implementations for multiple accumulators.

- `loop`: just loops across the accumulators, trusting the compiler to unroll as needed.
- `manually unroll`: manually unrolls the accumulator loop via templates.
- `peeled`: peels off the first loop iteration to avoid an unnecessary addition to the all-zero `dots`. 
- `vectorized epilogue`: add to the `dots` in the epilogue loop, which is most amenable to vectorization.
- `recursive sum`: use a recursive algorithm to sum the accumulators.
- `combined`: combines `peeled`, `recursive sum`, and (for 16+ accumulators) `vectorized epilogue`. 

## Instructions

To build the test executable, use the usual CMake process:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This compiles with various choices for the number of accumulators (4, 8 and 16).

## Vectorized epilogue

Sometimes it helps:

```console
$ ./build/test16 -l 29
Results for a vector of length 29
loop                                              : 0.00250919 ± 1.25749e-05
manually unrolled                                 : 0.00251855 ± 1.97719e-05
peeled                                            : 0.00185423 ± 7.48596e-06
vectorized epilogue                               : 0.00201737 ± 1.45924e-05
recursive sum                                     : 0.00198312 ± 1.43347e-05
combined                                          : 0.00148308 ± 5.75849e-06
```

and other times it doesn't:

```console
$ ./build/test4 -l 9
Results for a vector of length 9
loop                                              : 0.000513285 ± 1.06217e-06
manually unrolled                                 : 0.000529902 ± 8.3442e-06
peeled                                            : 0.000479797 ± 2.09554e-06
vectorized epilogue                               : 0.000618349 ± 3.05021e-05
recursive sum                                     : 0.00042117 ± 7.3637e-06
combined                                          : 0.000556544 ± 4.23583e-06
```

The problem is that the epilogue is now dependent on `dots` so has to wait for the main loop to finish.
This stalls the pipeline and counteracts any benefits from vector instructions.
By comparison, a naive accumulation doesn't have to wait. 

## Peeling

Peeling does pretty well compared to `loop` in many situations, for example:

```console
$ ./build/test4 -l 15
Results for a vector of length 15
loop                                              : 0.000679681 ± 9.34198e-06
manually unrolled                                 : 0.000683238 ± 6.70825e-06
peeled                                            : 0.000585089 ± 1.03187e-05
vectorized epilogue                               : 0.000704479 ± 4.85825e-06
recursive sum                                     : 0.000507646 ± 5.79391e-06
combined                                          : 0.000603108 ± 1.86221e-06
```

And it's a lot faster when the length is less than the number of accumulators, as we baked in a short-cut for that scenario:

```console
$ ./build/test16 -l 8
Results for a vector of length 8
loop                                              : 0.00181886 ± 3.61663e-06
manually unrolled                                 : 0.00183083 ± 6.72697e-06
peeled                                            : 0.000547975 ± 7.09111e-06
vectorized epilogue                               : 0.00150043 ± 5.38343e-06
recursive sum                                     : 0.00135885 ± 1.41946e-05
combined                                          : 0.000584397 ± 7.70532e-06
```

However, sometimes it doesn't help:

```console
$ ./build/test8 -l 128
Results for a vector of length 128
loop                                              : 0.00260234 ± 1.57836e-05
manually unrolled                                 : 0.00263361 ± 3.08048e-05
peeled                                            : 0.00292591 ± 5.75354e-06
vectorized epilogue                               : 0.00306841 ± 4.78548e-05
recursive sum                                     : 0.0023583 ± 1.06776e-05
combined                                          : 0.00283428 ± 3.89244e-05
```

This seems to be due to the extra conditional.
In practice, if we're computing a dot product in a tight loop and the function can be inlined, a sufficiently smart compiler (e.g., GCC 16) can hoist the conditional out of the loop. 

## Recursive sum

Offers a speed-up over `loop` when there are many accumulators:

```console
$ ./build/test16 -l 30
Results for a vector of length 30
loop                                              : 0.00259188 ± 1.13688e-05
manually unrolled                                 : 0.00255921 ± 5.77791e-06
peeled                                            : 0.00191791 ± 4.62762e-06
vectorized epilogue                               : 0.00201344 ± 5.60702e-06
recursive sum                                     : 0.00198596 ± 1.62073e-06
combined                                          : 0.000981628 ± 7.28822e-07
```

Even does well when there aren't that many accumulators:

```console
$ ./build/test4 -l 30
Results for a vector of length 30
loop                                              : 0.00093533 ± 1.1387e-05
manually unrolled                                 : 0.000965973 ± 9.61781e-06
peeled                                            : 0.000797491 ± 1.13914e-05
vectorized epilogue                               : 0.000968414 ± 6.42487e-06
recursive sum                                     : 0.000736849 ± 8.47261e-06
combined                                          : 0.000854695 ± 1.22741e-05
```

## Manual unrolling

This should be exactly the same as `loop`, as GCC will already unroll fixed-size loops at `-O3`.
Manual unrolling is nicer for older compilers (e.g., GCC <12) that don't do this at `-O2`.
That said, we would prefer to avoid manual unrolling as the compiler is usually in a better position to determine how much unrolling should be done,
e.g., to avoid register spills and to keep the program size small.

## Combined

For large numbers of accumulators, all of the changes synergize appropriately:

```console
$ ./build/test16 -l 18
Results for a vector of length 18
loop                                              : 0.00186555 ± 2.06845e-05
manually unrolled                                 : 0.00183516 ± 9.30086e-06
peeled                                            : 0.00125216 ± 1.40248e-05
vectorized epilogue                               : 0.00185857 ± 1.36576e-05
recursive sum                                     : 0.00206842 ± 8.43279e-06
combined                                          : 0.000735653 ± 9.24555e-06
```

For small numbers of accumulators, peeling and recursive summation don't synergize when they're combined:

```console
$ ./build/test4 -l 22
Results for a vector of length 22
loop                                              : 0.00082722 ± 4.16689e-05
manually unrolled                                 : 0.000805229 ± 4.12291e-06
peeled                                            : 0.000709305 ± 2.18848e-05
vectorized epilogue                               : 0.000880836 ± 4.88305e-05
recursive sum                                     : 0.000616033 ± 2.16033e-06
combined                                          : 0.000719391 ± 7.6331e-06
```

## Conclusions

What a mess.
Only the recursive sum consistently improves performance over the naive `loop`.
Peeling is generally okay but fights with recursive sum so I'd prefer the latter if I have to choose between them.
Perhaps it doesn't really matter: once we get to larger vectors, it's pretty much all the same as the main loop dominates the runtime.
