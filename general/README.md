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

## Sparse dot products

The calculation of a dot product between a sparse and dense vector is fairly straightforward.
We just iterate over the structural non-zero elements of the sparse vector, retrieve the value of the dense vector at each position, and add its product with the non-zero's value.
This can be used with multiple accumulators for improved pipelining -
and possibly auto-vectorization as well, though this relies on the availability of efficient gather instructions.

```cpp
// Two accumulators, similar to what Eigen does.
double dot1 = 0, dot2 = 0;
if (nnz > 1) { // protect against wraparound.
    for (std::size_t i = 0; i < nnz - 1; i += 2) {
        dot1 += dense[index[i]] * value[i];
        dot2 += dense[index[i + 1]] * value[i + 1];
    }
}
if (N % 2 == 1) {
    dot1 += dense[index[N- 1]] * value[N - 1];
}
double dot = dot1 + dot2;
```

For sparse-sparse dot products, it is probably not worth it to attempt an interleaved traversal.
Check out [some other performance tests](https://github.com/tatami-inc/test-sparse_indexed_extraction)
where we conclude that it is better to just expand one of the sparse vectors into a dense array and use the other one to do a look-up.
With this approach, the problem just collapses to that of a sparse-dense dot product as described above.

For simplicity, we will assume that there are no IEEE special values in the dense vector.
These greatly complicate matters as the product of special values with zero does not equal zero.
If we're expecting to deal with special values, it's probably best to just handle the sparse matrix as a dense matrix...
though to be honest, I don't see a practical use for the faithful propagation of special values in a matrix product.

## Blocking with dense matrices

For the product of two dense matrices, we can use a blocking approach where we compute the product of two fixed-size submatrices.
We use small submatrices so that they can be stored in L1 cache for fast re-use of rows/columns. 
For example, when computing the product of a row-major LHS and column-major RHS, each LHS row is re-used to compute the dot product for each RHS column.
If the extent of the shared dimension is large enough to trigger cache evictions,
each LHS row or RHS column (depending on the iteration pattern) would need to be reloaded from memory on every use.
With blocking, we can reuse cached parts of multiple LHS rows with cached parts of multiple RHS columns to compute partial dot products;
we repeat this across all pairs of submatrices and then aggregate the results to obtain the full matrix product.

The exact blocking strategy depends on the layout of the the RHS, LHS and output matrices,
but we generally expect to operate on two $B$-by-$C$ (or $C$-by-$B$) matrices and one $B$-by-$B$ matrix:

- $B$ determines the amount of cache re-use.
  In our example of a product of a row-major LHS and column-major RHS, each (part of a) LHS row only needs to be reloaded into cache once every $B$ RHS columns,
  compared to a naive approach where each LHS row may need to be reloaded into memory for each RHS column.
  Conversely, each (part of) an RHS column only needs to be loaded once every $B$ LHS rows.
  We could even split $B$ into $B_1$ for one submatrix and $B_2$ for the other, but let's keep things simple here.
- $C$ is the extent of the fastest-changing dimension and should be the larger number to reduce the looping overhead.
  The effective extent will usually be slightly larger than $C$ due to cache lines exceeding the submatrix boundaries.
  This is fine as the rest of those cache lines will be used when processing the submatrix that contains the next $C$ elements of the fastest-changing dimension. 

In this framework, $2BC + B^2$ is the number of elements to be held in cache at any given time, plus some extra space based on the granularity of the cache lines.
For a given cache size, a larger $B$ will improve cache re-use but increase the overhead from loop restarts due to a lower $C$.
A larger $B$ also increases memory usage as more dimension elements need to be realized into main memory by **tatami**;
and might put some more pressure on higher cache levels if the cache line sizes are different, 
though x86 seems to use 64 bytes for [all levels of the cache hierarchy](https://stackoverflow.com/questions/23024574/how-much-data-is-loaded-in-to-the-l2-and-l3-caches).

Some research suggests that we can expect to have at least 32 kb of L1 cache per core on modern Intel, AMD and Apple ARM processors.
If we're working with double-precision types, requiring $BC = 1024$ and enforcing $B \leq C$ will use 16-24 kb, which should easily fit into L1.
We could of course be more exact but it is better that $B$ and $C be multiples of 2,
as this provides a chance to exploit existing data alignment and vectorization. 
Indeed, blocking can be combined with multiple accumulators, in which case $C$ should be a multiple of the number of accumulators to avoid entry into the epilogue loop.

## Blocking with sparse matrices

### Challenges

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

### Blocking along non-zeros 

We can avoid the problems described above by only considering one sparse row/column at a time in our blocking schemes.
This reduces the potential for cache re-use as we need to reload parts of the accompanying dense matrix for each sparse row/column.

To illustrate, consider the most common case of a multiplication involving dense-sparse dot products,
i.e., the product of a dense row-major LHS matrix with a sparse column-major RHS matrix. 
For each RHS column, we consider a block of $C$ structural non-zeros.
We load in a block of $B$ LHS rows and we compute the partial dot product of those $C$ non-zeros with each LHS row.
We repeat the calculation with the next block of $C$ non-zeros until the entire RHS column is processed, then we move onto the next RHS column.
Once all RHS columns are processed, we consider the next block of $B$ LHS rows.
The idea is to only reload each block of $C$ non-zeros per $B$ LHS rows rather than for each LHS row.

The actual cache usage of this scheme is tricky to calculate.
In theory, the cache would hold $C$ non-zero values, $C$ non-zero indices, $B$ partial dot products, and at least $C$ values from one of the LHS rows.
If cache eviction uses an LRU policy, we will evict the used part of each LHS row while keeping the $C$ non-zeros in memory for re-use with the next LHS row;
thus, we won't need to cache parts from all $B$ rows at once, given that each part will only be used once for the current block of $C$ non-zeros.
However, as the $C$ LHS values might be non-contiguous, the actual cache usage of the $C$ LHS values will be higher than $C$ due to the granularity of cache lines.
If we assume double-precision data, 64-byte cache lines and one "useful" LHS value per cache line, the actual LHS usage might be up to $8C$, bringing us up to a total of $10C + B$.
So, setting $C = 256$ would fill the cache with a rough upper bound of ~2560 double-precision values, which should still fit comfortably in a >32kb L1 cache.

In practice, this approach does not seem to help all that much.
The structural non-zeros are already contiguous in memory so reloading them is pretty cheap,
compared to the (much more expensive) random access to the accompanying dense vector.

### Blocking to cache the dense vector

We have observed that the most expensive part of sparse matrix multiplication is the random accesses of the accompanying dense vector.
Each access might require a fetch from main memory if the position of the structural non-zero belongs in a new cache line.
To avoid this, we aim to process a block of $B$ sparse vectors for each dense vector.
For example, we might load a block of sparse LHS rows and compute the dot product for each sparse row with a single dense RHS column.

Here, we assume that each sparse vector in the block contains structural non-zeros at positions that lie within a cache line of the previous sparse vectors.
This means that we can re-use parts of the lone dense vector that have already been cached in operations with prior sparse vectors.
We also assume that the cache hierarchy is large enough to hold the entire dense vector,
or at least the parts corresponding to the structural non-zeros of all sparse vectors in the block.

The choice of $B$ is mainly a trade-off between speed and memory usage.
A larger $B$ improves re-use of the dense vector with more sparse vectors but requires realization of the sparse vectors into main memory by **tatami**.
Increasing $B$ also has an indirect effect on cache usage by forcing more of the dense vector to be loaded into cache, possibly causing the eviction of other values.
However, this is hard to predict as it depends on the sparsity and how much (near-)overlap of positions are present between sparse vectors in the block.

### Blocking to cache many dense vectors

Alternatively, if the dense vectors are small, we might consider loading a block of $B$ dense vectors into cache at once.
For each sparse vector, we iterate over the dense vectors in this block, computing dot products or performing a sparse vector multiply-add.
We then repeat this with a new sparse vector but re-using the same block of dense vectors.

The assumption is that the entire dense block is small enough to be completely held in the cache for fast re-use across sparse vectors.
The cache is also assumed to be large enough to store the sparse vector's contents for re-use across multiple dense vectors.
By caching both the dense and sparse vectors, this strategy is more effective than just caching one dense vector, but only when the relevant dimension extent is small.
