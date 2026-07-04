# Performance testing for matrix multiplication

This repository contains performance testing for different matrix multiplication strategies, 
used to inform the implementation of the various `multiply()` overloads in [**tatami_mult**](https://github.com/tatami-inc/tatami_mult).
Check out some of the general principles in [`general/README.md`](general/README.md`),
as well as each subdirectory's `README.md` for specific commentary on each multiplication strategy:

- [Dense row-major LHS, single vector RHS](dense_row/single_vector)
- [Dense row-major LHS, multiple vectors RHS](dense_row/multiple_vectors)
- [Dense row-major LHS, dense matrix RHS](dense_row/dense_matrix)
- [Dense row-major LHS, dense matrix RHS](dense_row/sparse_matrix)
- [Dense column-major LHS, single vector RHS](dense_column/single_vector)
- [Dense column-major LHS, multiple vectors RHS](dense_column/multiple_vectors)
- [Dense column-major LHS, dense matrix RHS](dense_column/dense_matrix)
- [Dense column-major LHS, dense matrix RHS](dense_column/sparse_matrix)
- [Sparse row-major LHS, single vector RHS](sparse_row/single_vector)
- [Sparse row-major LHS, multiple vectors RHS](sparse_row/multiple_vectors)
- [Sparse row-major LHS, dense matrix RHS](sparse_row/sparse_matrix)
- [Sparse row-major LHS, dense matrix RHS](sparse_row/sparse_matrix)
- [Sparse column-major LHS, single vector RHS](sparse_column/single_vector)
- [Sparse column-major LHS, multiple vectors RHS](sparse_column/multiple_vectors)
- [Sparse column-major LHS, dense matrix RHS](sparse_column/sparse_matrix)
- [Sparse column-major LHS, dense matrix RHS](sparse_column/sparse_matrix)
