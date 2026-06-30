# General comments

## Multiple accumulators

When computing a sum (typically for a dot product), we can use multiple accumulators.
That is, instead of doing:

```cpp
double dot = 0;
for (std::size_t i = 0; i < N; ++i) {
    dot += left[i] * right[i];
}
```

We could instead do something like:

```cpp
double dot1 = 0, dot2 = 0;
if (N > 1) { // protect against wraparound.
    for (std::size_t i = 0; i < N - 1; i += 2) {
        dot1 += left[i] * right[i];
        dot2 += left[i + 1] * right[i + 1];
    }
}
if (N % 2 == 1) {
    dot1 += left[N- 1] * right[N - 1];
}
double dot = dot1 + dot2;
```

Or for any choice of number of accumulators, if we trust the compiler to unroll the inner loop:

```cpp
std::array<double, M> xs;
const std::size_t nloops = N / M;
const std::size_t remainder = N % M;

// We could, in theory, skip an unnecessary addition for the first outer iteration,
// by setting 'xs[j] = left[j] * right[j]' and then starting our loop at 'i = 1'.
// However, this seems to be a pessimisation, probably because we duplicate many
// of the instructions just to skip an addition step that is already very cheap.
for (std::size_t i = 0; i < nloops; ++i) {
    for (std::size_t j = 0; j < M; ++j) {
        const auto index = i * M + j;
        xs[j] += left[index] * right[index];
    }
}

double extras = 0;
for (std::size_t j = 0; j < remainder; ++j) {
    const auto index = nloops * M + j;
    extras += left[index] * right[index];
}
double dot = std::accumulate(xs.begin(), xs.end(), extras);
```

The idea is to break dependency chains in the CPU's instruction pipeline by allowing multiple accumulations to occur in parallel.
It also provides some opportunities for auto-vectorization of the sum in the innermost loop.
However, we don't want to use too many accumulators as this could cause register spills and inflate the program size (or even prevent unrolling of the innermost loop).

## Blocking

### Dense matrices

For the product of two dense matrices, we can use a blocking approach where we compute the product of two fixed-size submatrices.
We use small submatrices so that they can be stored in L1 cache for fast re-use of rows/columns. 
For example, when computing the product of a row-major RHS and column-major LHS, each RHS row is re-used to compute the dot product for each LHS column.
If the extent of the shared dimension is large enough to trigger cache evictions,
each RHS row or LHS column (depending on the iteration pattern) would need to be reloaded from memory on every use.
With blocking, we can reuse cached parts of multiple RHS rows with cached parts of multiple LHS columns to compute partial dot products;
we repeat this across blocks and then aggregate the results to obtain the full matrix product.

The exact blocking strategy depends on the layout of the the RHS, LHS and output matrices,
but we generally expect to operate on two $B$-by-$C$ (or $C$-by-$B$) matrices and one $B$-by-$B$ matrix:

- $C$ is the extent of the fastest-changing dimension and should be the larger number to reduce the looping overhead.
  The actual number of elements held in cache will usually be slightly larger than $C$ due to the cache lines exceeding the submatrix boundaries;
  this is fine as the content in those lines will be used when processing the next block that contains the next $C$ elements of the fastest-changing dimension. 
- $B$ determines the amount of cache re-use.
  In our example of a product of a row-major RHS and column-major LHS, each (part of a) LHS row only needs to be reloaded into cache once every $B$ RHS columns,
  and each (part of a) RHS column only needs to be reloaded into cache once every $B$ LHS columns.
  Technically, we could split $B$ into $B_1$ for one submatrix and $B_2$ for the other, but let's keep things simple here.

Roughly speaking, $2BC + B^2$ is the number of elements to be held in cache at any given time, plus some extra space based on the granularity of the cache lines.
For a given cache size, a larger $B$ will improve cache re-use but increase the overhead from loop restarts due to a lower $C$.
A larger $B$ also increases memory usage as more dimension elements that need to be realized by **tatami**.

Some research suggests that we can expect to have at least 32 kb of L1 cache per core on modern Intel, AMD and Apple ARM processors.
If we're working with double-precision types, requiring $BC = 1024$ and enforcing $B \leq C$ will use 16-24 kb, which should easily fit into L1.
We could of course be more exact but it is better than $B$ and $C be multiples of 2,
as this provides a chance to exploit existing data alignment and vectorization. 
Indeed, blocking can be combined with multiple accumulators, in which case $C$ should be a multiple of the number of accumulators to avoid entry into the epilogue loop.

### Sparse matrices

If either of the input matrices is sparse (following the usual compressed sparse format from `tatami::SparseRange`), blocking is much more difficult.
It is most efficient to iterate over the structural non-zeros but the distribution of structural non-zeros is not predictable. 
This greatly complicates the handling of blocks that consist of multiple rows/columns of a sparse matrix.

To illustrate, let us consider blocking with submatrices of fixed dimensions as described above for the dense matrices.
Finding the boundaries of each submatrix in the sparse matrix is already difficult,
requiring either a binary search (ugh!) or extra tracking of the positions of the last non-zero element for each row/column in the preceding block.
We would also need to pay the cost of looping even if there are no structural non-zeros for a row/column in a block.
Indeed, the looping overhead would be larger than the actual product computation if the block size is smaller than the inverse of the density.

In theory, we could develop a more exotic blocking scheme that uses the same number of non-zero elements from each of $B$ sparse rows/columns.
This would improve the efficiency of traversal along the sparse row/column by making the overhead (mostly) proportional to the number of non-zero elements processed.
However, the product calculation involves accessing an accompanying dense matrix at the positions of the structural non-zeros,
and if there is no similarity between those positions across sparse rows/columns, then there's not much opportunity for re-use of cached values from the dense matrix.
Different RHS rows/columns will also have different number of structural non-zeros,
so towards the end of each row/column, we'd be wasting time by looping over on rows/columns that have no more non-zeros.

We can avoid these problems by only considering one sparse row/column at a time in our blocking schemes.
This reduces the potential for cache re-use as we need to reload parts of the accompanying dense matrix for each sparse row/column, but so be it.
The exact scheme will depend on the layout, e.g., we could use a block of $C$ non-zeros from a single row/column or a $B$-by-$C$ block of the accompanying dense matrix.
