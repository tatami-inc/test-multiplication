# Dense row-major product with multiple vectors

## Strategies

The naive approach involves computing dot-products for each row of the LHS matrix and each of the multiple RHS vectors.

The blocked approach takes $B$ rows of the LHS matrix and, for each row, computes the dot product of its first $B$ elements against the first $B$ elements of the first $B$ RHS vectors.
It repeatedly adds the dot product of the next $B$ elements until all columns of the LHS matrix have been traversed.
Then, it repeats this process with the next $B$ RHS vectors, until all vectors have been traversed;
and then finally, it repeats this with the next $B$ rows, until all rows have been traversed.
We test a range of different values for $B$.

## Double-precision results

Trying out a range of matrix shapes.
All timings below are obtained on an Intel i7-8850H.

```
$ ./build/dense_row_to_multiple_vectors/drmv_test
Results for 10000 x 5000
naive:          4.8346 ± 0.0122438
blocked (4):    3.31543 ± 0.00283382
blocked (8):    2.32661 ± 0.00918316
blocked (16):   2.1229 ± 0.00216885
blocked (32):   2.40224 ± 0.00374654

$ ./build/dense_row_to_multiple_vectors/drmv_test -r 200 -c 20000
Results for 200 x 20000
naive:          0.451686 ± 0.00191212
blocked (4):    0.294111 ± 0.000508774
blocked (8):    0.222715 ± 0.000725062
blocked (16):   0.212835 ± 0.000223328
blocked (32):   0.221206 ± 0.000287933

$ ./build/dense_row_to_multiple_vectors/drmv_test -r 100000 -c 100
Results for 100000 x 100
naive:          0.611737 ± 0.000796687
blocked (4):    0.686432 ± 0.000785403
blocked (8):    0.495957 ± 0.00790312
blocked (16):   0.452568 ± 0.000626675
blocked (32):   0.485393 ± 0.000507892
```

## Single-precision results

Again, but with single-precision floats.

```
$ ./build/dense_row_to_multiple_vectors/drmv_test_float
Results for 10000 x 5000
naive:          4.77999 ± 0.00509535
blocked (4):    3.27212 ± 0.00309624
blocked (8):    1.94642 ± 0.00118942
blocked (16):   1.78595 ± 0.00360239
blocked (32):   2.17263 ± 0.00154612

$ ./build/dense_row_to_multiple_vectors/drmv_test_float -r 200 -c 20000
Results for 200 x 20000
naive:          0.41868 ± 0.0013014
blocked (4):    0.275022 ± 0.000288659
blocked (8):    0.164147 ± 0.000526079
blocked (16):   0.149895 ± 0.000141083
blocked (32):   0.17962 ± 0.000174575

$ ./build/dense_row_to_multiple_vectors/drmv_test_float -r 100000 -c 100
Results for 100000 x 100
naive:          0.82286 ± 0.000627209
blocked (4):    0.670731 ± 0.00110194
blocked (8):    0.414939 ± 0.000474328
blocked (16):   0.376814 ± 0.000391669
blocked (32):   0.447951 ± 0.000766568
```

## Conclusion

Similar to the [single-vector case](../dense_row_to_single_vector), blocking gives a moderate speed-up with a block size of 16 elements.
This is attributable to the improvement in cache efficiency as each RHS vector does not need to be constantly reloaded per LHS row.
Rather, each RHS vector is only reloaded once per 16 LHS rows.

I'd speculate that $B = 16$ is the best compromise between cache efficiency and looping overhead, at least on the tested CPU.
Block sizes that are too small will have high overhead from the stop/starting across blocks,
while block sizes that are too large will not fit into cache.
