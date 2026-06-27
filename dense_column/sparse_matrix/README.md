# Dense column-major product with sparse matrix

## Strategies

For a column-major sparse RHS matrix, I just don't see any way to do this multiplication efficiently.
When dealing with a sparse matrix, we always want to traverse the structural non-zeros for one of the dimensions.
However, walking down the RHS column would require all the LHS columns to be in memory to compute the cumulative dot products.
We can't effectively block, either, because we can't easily estimate the boundaries of the blocking submatrices within a sparse matrix.

So, we'll restrict ourselves to a row-major sparse RHS matrix in this performance test.
The general idea is to compute an outer product between each RHS column and LHS row, and iteratively add the outer products together in the output matrix.
For column-major output, we can iterate across the structural non-zeros in each RHS row and perform a vectorized multiply-add of the corresponding LHS column.
For row-major output, we iterate down the LHS column and compute a sparse vector multiply-add with the corresponding RHS row.

We try out some small-scale blocking for column-major output, whereby we only consider $C$ elements of each LHS column when computing the outer product.
This aims to improve caching of the LHS column when it is repeatedly used to compute the outer product across non-zero elements of each RHS row.
Otherwise, for LHS matrices with many rows, the start of the column would need to be loaded back into cache per non-zero element.
Obviously, this blocking strategy comes at the cost of more looping overhead. 

More general-purpose blocking schemes are difficult as we can't define fixed-size submatrices easily in the presence of sparse data.

- As we are forced to iterate across each row of the RHS matrix in its entirey, the submatrix can be of unbounded size and is not guaranteed to fit into cache.
  This defeats the whole purpose of blocking if accessing one part of the submatrix causes another part to be evicted.
- Even if we did consider a block of RHS rows, each row would likely have its structural non-zeros at different positions.
  This means that we would probably not even be re-using the same output columns for different RHS rows.

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
naive, row RHS, column output                     : 0.0673965 ± 0.0017896
naive, row RHS, row output                        : 0.249862 ± 0.00267117
blocked (128), row RHS, column output             : 0.0697039 ± 0.00340042
blocked (256), row RHS, column output             : 0.0754084 ± 0.000981832
blocked (512), row RHS, column output             : 0.0728159 ± 0.0018075
blocked (1024), row RHS, column output            : 0.0710095 ± 0.000755745

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row RHS, column output                     : 0.0790625 ± 0.000897332
naive, row RHS, row output                        : 0.271194 ± 0.00300747
blocked (128), row RHS, column output             : 0.0672854 ± 0.00356284
blocked (256), row RHS, column output             : 0.0727121 ± 0.00133243
blocked (512), row RHS, column output             : 0.0750375 ± 0.00147722
blocked (1024), row RHS, column output            : 0.0763331 ± 0.00189665

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row RHS, column output                     : 0.114739 ± 0.00210393
naive, row RHS, row output                        : 0.684722 ± 0.000975781
blocked (128), row RHS, column output             : 0.121555 ± 0.00338566
blocked (256), row RHS, column output             : 0.118232 ± 0.00216201
blocked (512), row RHS, column output             : 0.12104 ± 0.000979505
blocked (1024), row RHS, column output            : 0.121029 ± 0.000551061

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row RHS, column output                     : 0.123472 ± 0.00123613
naive, row RHS, row output                        : 0.681982 ± 0.00254695
blocked (128), row RHS, column output             : 0.135834 ± 0.00401781
blocked (256), row RHS, column output             : 0.134912 ± 0.0019188
blocked (512), row RHS, column output             : 0.132769 ± 0.00123002
blocked (1024), row RHS, column output            : 0.12531 ± 0.000531403

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row RHS, column output                     : 0.0371322 ± 0.000283184
naive, row RHS, row output                        : 0.138154 ± 0.000259616
blocked (128), row RHS, column output             : 0.0392189 ± 0.000805308
blocked (256), row RHS, column output             : 0.0420455 ± 0.000982291
blocked (512), row RHS, column output             : 0.0421799 ± 0.000811446
blocked (1024), row RHS, column output            : 0.0413134 ± 0.000951588

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, row RHS, column output                     : 0.093105 ± 0.00065584
naive, row RHS, row output                        : 0.250352 ± 0.0022071
blocked (128), row RHS, column output             : 0.0938343 ± 0.00346417
blocked (256), row RHS, column output             : 0.0963734 ± 0.00037792
blocked (512), row RHS, column output             : 0.0951747 ± 0.000134878
blocked (1024), row RHS, column output            : 0.0959375 ± 0.000714929

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, row RHS, column output                     : 0.0396952 ± 0.000292445
naive, row RHS, row output                        : 0.129358 ± 0.000241423
blocked (128), row RHS, column output             : 0.0434884 ± 0.000489585
blocked (256), row RHS, column output             : 0.0436682 ± 0.000168456
blocked (512), row RHS, column output             : 0.0439241 ± 0.0003182
blocked (1024), row RHS, column output            : 0.0442645 ± 0.000182771
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive, row RHS, column output                     : 0.0261063 ± 0.00198227
naive, row RHS, row output                        : 0.106675 ± 0.0014779
blocked (128), row RHS, column output             : 0.0316461 ± 0.00186792
blocked (256), row RHS, column output             : 0.0293535 ± 0.00106234
blocked (512), row RHS, column output             : 0.0305591 ± 0.00144373
blocked (1024), row RHS, column output            : 0.0260433 ± 0.00127117

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row RHS, column output                     : 0.0286557 ± 0.00133492
naive, row RHS, row output                        : 0.124886 ± 0.00174963
blocked (128), row RHS, column output             : 0.0280641 ± 0.00184736
blocked (256), row RHS, column output             : 0.026682 ± 0.00168951
blocked (512), row RHS, column output             : 0.0308996 ± 0.0015138
blocked (1024), row RHS, column output            : 0.0278827 ± 0.00148759

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row RHS, column output                     : 0.0682345 ± 0.00119874
naive, row RHS, row output                        : 0.382477 ± 0.00262165
blocked (128), row RHS, column output             : 0.0768757 ± 0.00307084
blocked (256), row RHS, column output             : 0.0709875 ± 0.00116747
blocked (512), row RHS, column output             : 0.0721702 ± 0.00149434
blocked (1024), row RHS, column output            : 0.0721757 ± 0.00139581

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row RHS, column output                     : 0.0784319 ± 0.000494473
naive, row RHS, row output                        : 0.380763 ± 0.00279911
blocked (128), row RHS, column output             : 0.0815592 ± 0.00330609
blocked (256), row RHS, column output             : 0.0830454 ± 0.00101703
blocked (512), row RHS, column output             : 0.083835 ± 0.00117161
blocked (1024), row RHS, column output            : 0.0799912 ± 0.000732749

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row RHS, column output                     : 0.0198588 ± 0.000839737
naive, row RHS, row output                        : 0.103285 ± 0.000124911
blocked (128), row RHS, column output             : 0.0212838 ± 0.000590712
blocked (256), row RHS, column output             : 0.0209693 ± 0.00048333
blocked (512), row RHS, column output             : 0.0219511 ± 0.000473371
blocked (1024), row RHS, column output            : 0.0199762 ± 0.000572923

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 1000 x 10000
naive, row RHS, column output                     : 0.0376723 ± 0.00239248
naive, row RHS, row output                        : 0.112698 ± 0.00188724
blocked (128), row RHS, column output             : 0.0474164 ± 0.00235916
blocked (256), row RHS, column output             : 0.041559 ± 0.00165679
blocked (512), row RHS, column output             : 0.0486375 ± 0.00200621
blocked (1024), row RHS, column output            : 0.0435867 ± 0.00176446

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 10000 x 1000
naive, row RHS, column output                     : 0.0224365 ± 0.000242222
naive, row RHS, row output                        : 0.0953896 ± 0.000297585
blocked (128), row RHS, column output             : 0.0270338 ± 0.000372836
blocked (256), row RHS, column output             : 0.0261562 ± 0.000254687
blocked (512), row RHS, column output             : 0.0263166 ± 0.00027765
blocked (1024), row RHS, column output            : 0.025532 ± 0.000306452
```

## Conclusion

It seems like blocking doesn't help much.
Even with more extreme dimensions where the first column must exceed the cache, we don't see a big benefit from blocking:

```console
$ ./build/test -r 500000 -c 20 -H 200
Results for 500000 x 20 x 200
naive, row RHS, column output                     : 0.238632 ± 0.00927429
naive, row RHS, row output                        : 1.47265 ± 0.0287914
blocked (128), row RHS, column output             : 0.254016 ± 0.00984475
blocked (256), row RHS, column output             : 0.249547 ± 0.00860834
blocked (512), row RHS, column output             : 0.223707 ± 0.00621101
blocked (1024), row RHS, column output            : 0.23182 ± 0.00787642
```

I'd guess that the cache misses in the naive approach are either masked by some kind of prefetching, and/or they're offset by the extra looping overhead. 

Row-major output is much slower as the writes are non-contiguous, but there's not much that we can do there if row output is requested.
