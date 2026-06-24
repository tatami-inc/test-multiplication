# Dense row-major versus a single vector

## Strategies

The naive approach involves computing dot-products for each row of the LHS matrix and the RHS vector.

The blocked approach takes $B$ rows of the LHS matrix and, for each row, computes the dot product of its first $B$ elements against the first $B$ elements of the RHS vector.
It repeatedly adds the dot product of the next $B$ elements until all columns of the LHS matrix have been traversed.
Then, it proceeds to the next $B$ rows until all rows have been traversed.
We test a range of different values for $B$.

## Double-precision results

Trying out a range of matrix shapes.
All timings below are obtained on an Intel i7-8850H.

```bash
$ ./build/dense_row/single_vector/drsv_test
Results for 10000 x 5000
naive:          0.0542285 ± 3.34228e-05
blocked (4):    0.042366 ± 0.000252097
blocked (8):    0.030795 ± 0.000146667
blocked (16):   0.0304501 ± 0.000135064
blocked (32):   0.0335762 ± 0.000152055

$ ./build/dense_row/single_vector/drsv_test -r 200 -c 50000
Results for 200 x 50000
naive:          0.0107306 ± 3.23375e-05
blocked (4):    0.0085998 ± 0.00018445
blocked (8):    0.0060053 ± 3.045e-05
blocked (16):   0.00613385 ± 0.000209699
blocked (32):   0.00641108 ± 3.29051e-05

$ ./build/dense_row/single_vector/drsv_test -r 50000 -c 200
Results for 50000 x 200
naive:          0.0092874 ± 1.72382e-05
blocked (4):    0.0135834 ± 7.5533e-05
blocked (8):    0.0072575 ± 0.000183823
blocked (16):   0.00678745 ± 2.94841e-05
blocked (32):   0.00854125 ± 2.97028e-05
```

## Single-precision results

Again, but with single-precision floats.

```bash
$ ./build/dense_row/single_vector/drsv_test_float
Results for 10000 x 5000
naive:          0.0526109 ± 0.000441491
blocked (4):    0.0355388 ± 0.000175101
blocked (8):    0.0235676 ± 0.000436915
blocked (16):   0.0227525 ± 0.000917256
blocked (32):   0.027698 ± 0.00082551

$ ./build/dense_row/single_vector/drsv_test -r 200 -c 50000
Results for 200 x 50000
naive:          0.0107807 ± 8.1561e-05
blocked (4):    0.00840016 ± 6.52994e-05
blocked (8):    0.00599754 ± 4.35428e-05
blocked (16):   0.005919 ± 1.70279e-05
blocked (32):   0.00641248 ± 1.25911e-05

$ ./build/dense_row/single_vector/drsv_test -r 50000 -c 200
Results for 50000 x 200
naive:          0.00931981 ± 1.72624e-05
blocked (4):    0.0135165 ± 7.46251e-05
blocked (8):    0.00725454 ± 0.000170445
blocked (16):   0.00685953 ± 2.52907e-05
blocked (32):   0.00854268 ± 3.86938e-05
Results for 50000 x 200
```

## Conclusion

Blocking gives a moderate speed-up with a block size of 16 elements.
This is attributable to the improvement in cache efficiency as the RHS vector does not need to be constantly reloaded per LHS row.
Rather, the RHS vector is only reloaded once per 16 LHS rows.

I'd speculate that $B = 16$ is the best compromise between cache efficiency and looping overhead, at least on the tested CPU.
Block sizes that are too small will have high overhead from the stop/starting across blocks,
while block sizes that are too large will not fit into cache.
