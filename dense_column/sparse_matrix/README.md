# Dense column-major product with sparse matrix

## Strategies

All strategies must iterate across consecutive columns of the LHS matrix in the outermost loop.
We will be loading columns on demand via the **tatami** interface, so the full matrix will not be available for random access.

### Column-major RHS

When dealing with a sparse matrix, we always want to traverse the structural non-zeros for one of the dimensions.
However, walking down the RHS column would require all the LHS columns to be in memory to compute the cumulative dot products.
We can't effectively block, either, because we can't easily estimate the boundaries of the block submatrices within a sparse matrix.

Which is not to say that this is impossible.
We can iterate across all of the RHS columns to construct each row as it is needed.
However, this is not very amenable to parallelization as each thread would need to search for the start and end of its block of rows in each column.
At that point, it is better to just construct a row-major representation for the RHS matrix in the first place.

Thus, for the rest of this document, we'll restrict our consideration to a row-major sparse RHS matrix. 

### Naive row-major RHS

The general idea is to compute an outer product between each LHS column and RHS row, and iteratively add the outer products together in the output matrix.

For column-major output, we can iterate across the structural non-zeros in each RHS row.
For each structural non-zero, we perform a vectorized multiply-add of the corresponding LHS column to the corresponding column of the output matrix.

For row-major output, we iterate down the LHS column and compute a sparse vector multiply-add of the corresponding RHS row to the corresponding row of the output matrix.
This will be less efficient than the column-major output as the writes will not be contiguous, but oh well.

## Blocked row-major RHS

For column output, we consider a block of $C$ elements of each LHS column when computing the outer product for each structural non-zero of an RHS row.
Once all structural non-zeros are processed, we move to the next block of $C$ elements and repeat the outer product calculation.
This is done until the entire LHS column is processed, and then we move onto the next column.
Our hope is to improve caching of the "active" part of each LHS column when it is repeatedly used to compute the outer product across each RHS row.
Otherwise, for LHS matrices with many rows, the start of the column would need to be loaded back into cache per non-zero element.

For row output, we consider a block of $C$ structural non-zero elements of each RHS row when computing the outer product for each element of an LHS column.
Once all column elements are processed, we move to the next block of $C$ structural non-zeros and repeat the outer product calculation.
This is done until the entire RHS row is processed, and then we move onto the next row.
Again, the hope is to improve caching of the active non-zero elements for each RHS row.

More general-purpose blocking schemes are difficult as we can't define fixed-size submatrices easily in the presence of sparse data.

- As we are forced to iterate across each row of the RHS matrix in its entirey, the submatrix can be of unbounded size and is not guaranteed to fit into cache.
  This defeats the whole purpose of blocking if accessing one part of the submatrix causes another part to be evicted from cache.
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
naive, row RHS, column output                     : 0.0356972 ± 0.000374893
naive, row RHS, row output                        : 0.218077 ± 0.00695667
blocked (128), row RHS, column output             : 0.0543426 ± 0.00752661
blocked (256), row RHS, column output             : 0.0436372 ± 0.0011435
blocked (512), row RHS, column output             : 0.040556 ± 0.000988374
blocked (1024), row RHS, column output            : 0.0388785 ± 0.000363062
blocked (128), row RHS, row output                : 0.211781 ± 0.00154023
blocked (256), row RHS, row output                : 0.23902 ± 0.0238299
blocked (512), row RHS, row output                : 0.203794 ± 0.000842167
blocked (1024), row RHS, row output               : 0.205347 ± 0.00126108

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row RHS, column output                     : 0.0429419 ± 5.50559e-05
naive, row RHS, row output                        : 0.225763 ± 0.00156107
blocked (128), row RHS, column output             : 0.0379663 ± 8.07015e-05
blocked (256), row RHS, column output             : 0.0409306 ± 3.38256e-05
blocked (512), row RHS, column output             : 0.0407225 ± 3.5139e-05
blocked (1024), row RHS, column output            : 0.04085 ± 0.000131732
blocked (128), row RHS, row output                : 0.222398 ± 0.000141892
blocked (256), row RHS, row output                : 0.224421 ± 0.000248181
blocked (512), row RHS, row output                : 0.226588 ± 0.000299493
blocked (1024), row RHS, row output               : 0.225024 ± 0.000225303

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row RHS, column output                     : 0.0826236 ± 0.000129477
naive, row RHS, row output                        : 0.647292 ± 0.00030174
blocked (128), row RHS, column output             : 0.0918274 ± 0.000229209
blocked (256), row RHS, column output             : 0.0903443 ± 3.29796e-05
blocked (512), row RHS, column output             : 0.0890556 ± 7.33839e-05
blocked (1024), row RHS, column output            : 0.0871623 ± 7.39543e-05
blocked (128), row RHS, row output                : 0.648411 ± 0.000821032
blocked (256), row RHS, row output                : 0.644456 ± 0.000199821
blocked (512), row RHS, row output                : 0.64333 ± 0.000160644
blocked (1024), row RHS, row output               : 0.64293 ± 0.000220818

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row RHS, column output                     : 0.0898975 ± 7.16354e-05
naive, row RHS, row output                        : 0.6456 ± 0.000306622
blocked (128), row RHS, column output             : 0.106199 ± 0.000197261
blocked (256), row RHS, column output             : 0.107575 ± 0.00021641
blocked (512), row RHS, column output             : 0.0998377 ± 8.43866e-05
blocked (1024), row RHS, column output            : 0.0912486 ± 6.12763e-05
blocked (128), row RHS, row output                : 0.674806 ± 0.0012003
blocked (256), row RHS, row output                : 0.652902 ± 0.000194016
blocked (512), row RHS, row output                : 0.646549 ± 0.00105537
blocked (1024), row RHS, row output               : 0.639416 ± 0.000285466

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row RHS, column output                     : 0.0293141 ± 3.64269e-05
naive, row RHS, row output                        : 0.123904 ± 6.27186e-05
blocked (128), row RHS, column output             : 0.0330869 ± 4.00872e-05
blocked (256), row RHS, column output             : 0.0345428 ± 7.91711e-05
blocked (512), row RHS, column output             : 0.0327772 ± 5.08264e-05
blocked (1024), row RHS, column output            : 0.0318902 ± 3.36624e-05
blocked (128), row RHS, row output                : 0.12308 ± 2.49843e-05
blocked (256), row RHS, row output                : 0.12271 ± 8.1004e-05
blocked (512), row RHS, row output                : 0.123246 ± 3.71537e-05
blocked (1024), row RHS, row output               : 0.123218 ± 4.4521e-05

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, row RHS, column output                     : 0.0574062 ± 3.96908e-05
naive, row RHS, row output                        : 0.206826 ± 0.00026353
blocked (128), row RHS, column output             : 0.0621617 ± 5.10567e-05
blocked (256), row RHS, column output             : 0.062438 ± 0.000139902
blocked (512), row RHS, column output             : 0.0610275 ± 9.28634e-05
blocked (1024), row RHS, column output            : 0.0617146 ± 0.000148137
blocked (128), row RHS, row output                : 0.21272 ± 8.79476e-05
blocked (256), row RHS, row output                : 0.20171 ± 0.000313028
blocked (512), row RHS, row output                : 0.206764 ± 8.11526e-05
blocked (1024), row RHS, row output               : 0.204386 ± 9.84298e-05

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, row RHS, column output                     : 0.0345843 ± 6.88292e-05
naive, row RHS, row output                        : 0.117546 ± 3.49324e-05
blocked (128), row RHS, column output             : 0.038077 ± 2.09119e-05
blocked (256), row RHS, column output             : 0.0376885 ± 2.161e-05
blocked (512), row RHS, column output             : 0.0382012 ± 2.24354e-05
blocked (1024), row RHS, column output            : 0.0380293 ± 1.96443e-05
blocked (128), row RHS, row output                : 0.116883 ± 9.54328e-05
blocked (256), row RHS, row output                : 0.117679 ± 3.0667e-05
blocked (512), row RHS, row output                : 0.117885 ± 0.000186638
blocked (1024), row RHS, row output               : 0.117089 ± 3.39686e-05
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive, row RHS, column output                     : 0.014517 ± 2.70395e-05
naive, row RHS, row output                        : 0.0860351 ± 0.000124818
blocked (128), row RHS, column output             : 0.020889 ± 1.59357e-05
blocked (256), row RHS, column output             : 0.0215458 ± 3.99729e-05
blocked (512), row RHS, column output             : 0.0203722 ± 3.51145e-05
blocked (1024), row RHS, column output            : 0.0192068 ± 1.13609e-05
blocked (128), row RHS, row output                : 0.0858326 ± 3.85258e-05
blocked (256), row RHS, row output                : 0.0842049 ± 0.000138161
blocked (512), row RHS, row output                : 0.0856964 ± 3.14999e-05
blocked (1024), row RHS, row output               : 0.0846422 ± 8.53149e-05

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row RHS, column output                     : 0.0162398 ± 1.57706e-05
naive, row RHS, row output                        : 0.0963365 ± 0.000376648
blocked (128), row RHS, column output             : 0.0203757 ± 1.96133e-05
blocked (256), row RHS, column output             : 0.021985 ± 2.51778e-05
blocked (512), row RHS, column output             : 0.0206915 ± 5.2287e-05
blocked (1024), row RHS, column output            : 0.0199033 ± 4.66126e-05
blocked (128), row RHS, row output                : 0.0944467 ± 5.46561e-05
blocked (256), row RHS, row output                : 0.0938153 ± 4.11847e-05
blocked (512), row RHS, row output                : 0.0943724 ± 0.000105578
blocked (1024), row RHS, row output               : 0.0933367 ± 9.7814e-05

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row RHS, column output                     : 0.0388172 ± 9.22274e-05
naive, row RHS, row output                        : 0.34107 ± 0.0014162
blocked (128), row RHS, column output             : 0.0483873 ± 4.9841e-05
blocked (256), row RHS, column output             : 0.0418941 ± 3.1504e-05
blocked (512), row RHS, column output             : 0.0413151 ± 4.00638e-05
blocked (1024), row RHS, column output            : 0.0411247 ± 6.53267e-05
blocked (128), row RHS, row output                : 0.339791 ± 0.000544971
blocked (256), row RHS, row output                : 0.339255 ± 0.000370551
blocked (512), row RHS, row output                : 0.339528 ± 0.000413606
blocked (1024), row RHS, row output               : 0.339887 ± 0.000463765

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row RHS, column output                     : 0.0461412 ± 5.74351e-05
naive, row RHS, row output                        : 0.33818 ± 0.00123946
blocked (128), row RHS, column output             : 0.0521783 ± 3.27125e-05
blocked (256), row RHS, column output             : 0.0538597 ± 2.78899e-05
blocked (512), row RHS, column output             : 0.0531086 ± 3.42765e-05
blocked (1024), row RHS, column output            : 0.0470909 ± 4.63054e-05
blocked (128), row RHS, row output                : 0.411105 ± 0.000301979
blocked (256), row RHS, row output                : 0.362838 ± 0.000179276
blocked (512), row RHS, row output                : 0.346857 ± 0.000136312
blocked (1024), row RHS, row output               : 0.338077 ± 0.000345245

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row RHS, column output                     : 0.0146906 ± 1.47588e-05
naive, row RHS, row output                        : 0.0905756 ± 7.97093e-05
blocked (128), row RHS, column output             : 0.0209401 ± 3.81626e-05
blocked (256), row RHS, column output             : 0.0220464 ± 2.20336e-05
blocked (512), row RHS, column output             : 0.020711 ± 2.71918e-05
blocked (1024), row RHS, column output            : 0.0197321 ± 2.59937e-05
blocked (128), row RHS, row output                : 0.0841872 ± 5.83278e-05
blocked (256), row RHS, row output                : 0.0848358 ± 0.000101014
blocked (512), row RHS, row output                : 0.0830473 ± 6.6351e-05
blocked (1024), row RHS, row output               : 0.0834315 ± 0.000143334

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 1000 x 10000
naive, row RHS, column output                     : 0.0231085 ± 0.000120917
naive, row RHS, row output                        : 0.0887664 ± 5.19691e-05
blocked (128), row RHS, column output             : 0.0277539 ± 9.6769e-05
blocked (256), row RHS, column output             : 0.0277768 ± 8.96387e-05
blocked (512), row RHS, column output             : 0.0281257 ± 4.79516e-05
blocked (1024), row RHS, column output            : 0.0277472 ± 4.14784e-05
blocked (128), row RHS, row output                : 0.0913391 ± 0.000138699
blocked (256), row RHS, row output                : 0.0891319 ± 3.67025e-05
blocked (512), row RHS, row output                : 0.0904195 ± 7.33683e-05
blocked (1024), row RHS, row output               : 0.0885603 ± 0.000453622

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 10000 x 1000
naive, row RHS, column output                     : 0.0190737 ± 0.000106386
naive, row RHS, row output                        : 0.0849029 ± 0.000296306
blocked (128), row RHS, column output             : 0.0230551 ± 2.05512e-05
blocked (256), row RHS, column output             : 0.0228095 ± 1.63241e-05
blocked (512), row RHS, column output             : 0.0224252 ± 1.69779e-05
blocked (1024), row RHS, column output            : 0.0228206 ± 1.73203e-05
blocked (128), row RHS, row output                : 0.0775114 ± 4.97781e-05
blocked (256), row RHS, row output                : 0.0791293 ± 3.62393e-05
blocked (512), row RHS, row output                : 0.0759522 ± 3.64658e-05
blocked (1024), row RHS, row output               : 0.0762739 ± 4.07913e-05
```

## Conclusion

It seems like blocking doesn't help much.
Even with larger LHS matrices where the first column cannot fit into the cache, we don't see a big benefit from blocking with column-major output: 

```console
$ ./build/test -r 500000 -c 20 -H 200
Results for 500000 x 20 x 200
naive, row RHS, column output                     : 0.193006 ± 0.00312997
naive, row RHS, row output                        : 1.36091 ± 0.0364098
blocked (128), row RHS, column output             : 0.216437 ± 0.00195136
blocked (256), row RHS, column output             : 0.212758 ± 0.00220637
blocked (512), row RHS, column output             : 0.188694 ± 0.00147777
blocked (1024), row RHS, column output            : 0.185769 ± 0.00205918
blocked (128), row RHS, row output                : 1.31491 ± 0.0142829
blocked (256), row RHS, row output                : 1.30407 ± 0.0116512
blocked (512), row RHS, row output                : 1.31817 ± 0.0188771
blocked (1024), row RHS, row output               : 1.32232 ± 0.0182209
```

I'd guess that the cache misses in the naive approach are either masked by some kind of prefetching, and/or they're offset by the extra looping overhead. 
This is a perfectly acceptable result as the naive approach is much easier to implement anyway.

Row-major output is much slower as the writes are non-contiguous, but there's not much that we can do there if row output is requested.
