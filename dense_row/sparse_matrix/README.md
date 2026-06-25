# Dense row-major product with sparse matrix

## Strategies

For column-major RHS matrices, we can compute a dense-sparse dot product for each LHS row and RHS column.
The dot product itself can be augmented with multiple accumulators.

For row-major RHS matrices, we perform a sparse vector multiply-add for each RHS row.
This is best done with row-major output as both inputs and outputs are contiguous. 
However, we also test with column-major output as well.

We do not consider blocking as all of the innermost loops involve iteration over the sparse indices.
We can't easily block as we can't quickly determine where to even restart the innermost loop across blocks.
Moreover, the blocking overhead can quickly become larger than the processing itself if the block size is smaller than the inverse of the density.

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
naive, column RHS, column output                  : 0.0934547 ± 0.000716936
naive, column RHS, row output                     : 0.0929627 ± 0.000757053
naive, row RHS, column output                     : 0.174599 ± 0.000978767
naive, row RHS, row output                        : 0.0705209 ± 0.000632345
2 accumulators, column RHS, column output         : 0.063427 ± 0.000855962
2 accumulators, column RHS, row output            : 0.0642508 ± 0.000413885
4 accumulators, column RHS, column output         : 0.0541848 ± 0.000732155
4 accumulators, column RHS, row output            : 0.0551527 ± 0.00061382
8 accumulators, column RHS, column output         : 0.0593225 ± 0.00056431
8 accumulators, column RHS, row output            : 0.0570361 ± 0.000760488

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, column RHS, column output                  : 0.0946339 ± 0.000680404
naive, column RHS, row output                     : 0.0940679 ± 0.000604609
naive, row RHS, column output                     : 0.161636 ± 0.000586725
naive, row RHS, row output                        : 0.103092 ± 0.000823804
2 accumulators, column RHS, column output         : 0.0651331 ± 0.000865175
2 accumulators, column RHS, row output            : 0.0648855 ± 0.000736219
4 accumulators, column RHS, column output         : 0.0531022 ± 0.000879875
4 accumulators, column RHS, row output            : 0.0520125 ± 0.000614923
G8 accumulators, column RHS, column output         : 0.0588147 ± 0.000862452
8 accumulators, column RHS, row output            : 0.0581021 ± 0.00094392

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, column RHS, column output                  : 0.101261 ± 7.46859e-05
naive, column RHS, row output                     : 0.0968312 ± 8.34582e-05
naive, row RHS, column output                     : 0.188293 ± 0.000289527
naive, row RHS, row output                        : 0.0743674 ± 0.00108413
2 accumulators, column RHS, column output         : 0.0978074 ± 0.000195286
2 accumulators, column RHS, row output            : 0.0878531 ± 0.000117484
4 accumulators, column RHS, column output         : 0.105414 ± 0.000553097
4 accumulators, column RHS, row output            : 0.0898624 ± 0.000228346
8 accumulators, column RHS, column output         : 0.124156 ± 0.000272754
8 accumulators, column RHS, row output            : 0.112338 ± 0.000355444

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, column RHS, column output                  : 0.112785 ± 0.00140121
naive, column RHS, row output                     : 0.104724 ± 0.00102307
naive, row RHS, column output                     : 0.741669 ± 0.00239059
naive, row RHS, row output                        : 0.0883935 ± 0.00106461
2 accumulators, column RHS, column output         : 0.125945 ± 0.00131723
2 accumulators, column RHS, row output            : 0.120139 ± 0.00106404
4 accumulators, column RHS, column output         : 0.138755 ± 0.00182263
4 accumulators, column RHS, row output            : 0.139531 ± 0.0010243
8 accumulators, column RHS, column output         : 0.15028 ± 0.00140967
8 accumulators, column RHS, row output            : 0.137161 ± 0.00108532

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, column RHS, column output                  : 0.104232 ± 0.000200744
naive, column RHS, row output                     : 0.103492 ± 0.000141981
naive, row RHS, column output                     : 0.205353 ± 0.000527921
naive, row RHS, row output                        : 0.132618 ± 0.000521526
2 accumulators, column RHS, column output         : 0.0709193 ± 0.00047707
2 accumulators, column RHS, row output            : 0.0702366 ± 7.35311e-05
4 accumulators, column RHS, column output         : 0.0647699 ± 0.000113237
4 accumulators, column RHS, row output            : 0.064574 ± 6.36324e-05
8 accumulators, column RHS, column output         : 0.0659902 ± 0.000896201
8 accumulators, column RHS, row output            : 0.0658045 ± 0.000264537

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, column RHS, column output                  : 0.138065 ± 0.0014605
naive, column RHS, row output                     : 0.133237 ± 0.000246403
naive, row RHS, column output                     : 0.446974 ± 0.000398633
naive, row RHS, row output                        : 0.127045 ± 0.00280774
2 accumulators, column RHS, column output         : 0.117418 ± 0.000248166
2 accumulators, column RHS, row output            : 0.110274 ± 0.000290913
4 accumulators, column RHS, column output         : 0.113361 ± 0.000564379
4 accumulators, column RHS, row output            : 0.105511 ± 0.000261701
8 accumulators, column RHS, column output         : 0.115033 ± 0.000379372
8 accumulators, column RHS, row output            : 0.107665 ± 0.00039051

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, column RHS, column output                  : 0.140303 ± 0.000436447
naive, column RHS, row output                     : 0.139152 ± 0.00015534
naive, row RHS, column output                     : 0.244229 ± 0.00142908
naive, row RHS, row output                        : 0.142843 ± 0.00306017
2 accumulators, column RHS, column output         : 0.112049 ± 0.00147586
2 accumulators, column RHS, row output            : 0.112554 ± 0.000341425
4 accumulators, column RHS, column output         : 0.108641 ± 0.000376725
4 accumulators, column RHS, row output            : 0.10764 ± 0.000262866
8 accumulators, column RHS, column output         : 0.10888 ± 0.000161158
8 accumulators, column RHS, row output            : 0.108562 ± 0.000465231
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive, column RHS, column output                  : 0.0902739 ± 0.000316414
naive, column RHS, row output                     : 0.0900369 ± 0.000346563
naive, row RHS, column output                     : 0.159087 ± 0.000559325
naive, row RHS, row output                        : 0.0651068 ± 0.000432855
2 accumulators, column RHS, column output         : 0.0588163 ± 0.000398278
2 accumulators, column RHS, row output            : 0.0593066 ± 0.000345315
4 accumulators, column RHS, column output         : 0.044343 ± 0.00022404
4 accumulators, column RHS, row output            : 0.0451157 ± 0.000226042
8 accumulators, column RHS, column output         : 0.0487443 ± 0.000291171
8 accumulators, column RHS, row output            : 0.048636 ± 0.000322536

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, column RHS, column output                  : 0.0915429 ± 0.000550296
naive, column RHS, row output                     : 0.0903958 ± 0.000397949
naive, row RHS, column output                     : 0.150774 ± 0.000299917
naive, row RHS, row output                        : 0.0979478 ± 0.000550618
2 accumulators, column RHS, column output         : 0.0599505 ± 0.000659373
2 accumulators, column RHS, row output            : 0.0611979 ± 0.000433588
4 accumulators, column RHS, column output         : 0.04475 ± 0.000488252
4 accumulators, column RHS, row output            : 0.0444712 ± 0.000437299
8 accumulators, column RHS, column output         : 0.0499935 ± 0.000572451
8 accumulators, column RHS, row output            : 0.0483996 ± 0.000660756

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, column RHS, column output                  : 0.0933667 ± 0.000211262
naive, column RHS, row output                     : 0.095992 ± 0.000234407
naive, row RHS, column output                     : 0.201218 ± 0.000538952
naive, row RHS, row output                        : 0.0636484 ± 0.000858093
2 accumulators, column RHS, column output         : 0.0912578 ± 0.000178137
2 accumulators, column RHS, row output            : 0.0809723 ± 0.000197359
4 accumulators, column RHS, column output         : 0.0843962 ± 0.000285103
4 accumulators, column RHS, row output            : 0.0767456 ± 0.000259279
8 accumulators, column RHS, column output         : 0.111667 ± 0.000183149
8 accumulators, column RHS, row output            : 0.10441 ± 0.000243525

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, column RHS, column output                  : 0.116623 ± 0.000368672
naive, column RHS, row output                     : 0.110891 ± 0.000565092
naive, row RHS, column output                     : 0.648945 ± 0.00454479
naive, row RHS, row output                        : 0.0681991 ± 0.000914975
2 accumulators, column RHS, column output         : 0.130582 ± 0.00140014
2 accumulators, column RHS, row output            : 0.117974 ± 0.000828349
4 accumulators, column RHS, column output         : 0.131704 ± 0.000670924
4 accumulators, column RHS, row output            : 0.132131 ± 0.000524022
8 accumulators, column RHS, column output         : 0.161075 ± 0.00138207
8 accumulators, column RHS, row output            : 0.153662 ± 0.000753242

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, column RHS, column output                  : 0.1031 ± 0.000822693
naive, column RHS, row output                     : 0.103051 ± 0.00112901
naive, row RHS, column output                     : 0.201844 ± 0.00266197
naive, row RHS, row output                        : 0.138472 ± 0.00226794
2 accumulators, column RHS, column output         : 0.06137 ± 0.00079693
2 accumulators, column RHS, row output            : 0.0627078 ± 0.0009225
4 accumulators, column RHS, column output         : 0.0527329 ± 0.000670073
4 accumulators, column RHS, row output            : 0.0539449 ± 0.000806118
8 accumulators, column RHS, column output         : 0.0547635 ± 0.00109101
8 accumulators, column RHS, row output            : 0.0536587 ± 0.000765172

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, column RHS, column output                  : 0.112017 ± 0.000437391
naive, column RHS, row output                     : 0.110576 ± 0.000914948
naive, row RHS, column output                     : 0.211954 ± 0.00462376
naive, row RHS, row output                        : 0.112381 ± 0.00150865
2 accumulators, column RHS, column output         : 0.0778409 ± 0.00141706
2 accumulators, column RHS, row output            : 0.0751575 ± 0.00189123
4 accumulators, column RHS, column output         : 0.0722609 ± 0.00323624
4 accumulators, column RHS, row output            : 0.0695678 ± 0.00159968
8 accumulators, column RHS, column output         : 0.0702651 ± 0.00147379
8 accumulators, column RHS, row output            : 0.066853 ± 0.000836725

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, column RHS, column output                  : 0.117194 ± 0.00411092
naive, column RHS, row output                     : 0.113119 ± 0.00234069
naive, row RHS, column output                     : 0.370282 ± 0.00539727
naive, row RHS, row output                        : 0.0880584 ± 0.00347757
2 accumulators, column RHS, column output         : 0.0930302 ± 0.00237894
2 accumulators, column RHS, row output            : 0.0866952 ± 0.00240987
4 accumulators, column RHS, column output         : 0.0823754 ± 0.00251068
4 accumulators, column RHS, row output            : 0.073523 ± 0.00309813
8 accumulators, column RHS, column output         : 0.0834597 ± 0.00251206
8 accumulators, column RHS, row output            : 0.0773625 ± 0.00299932
```

## Conclusion

Multiple accumulators often improve performance for multiplication with a column-major RHS matrix.
This is most apparent when the shared dimension extent is high, though it's also not too bad when the shared dimension is small.
It seems that 4 accumulators is typically the sweet spot, similar to our conclusions in the [`dense`](../single_vector) case.

Performance is pretty good for a row-major RHS matrix with row-major output.
I find this interesting because a sparse vector multiply-add is not easily vectorizable.
The values of the indices are not known to the compiler, so it can't just execute the instructions in parallel as it can't be sure that the indices don't overlap.
