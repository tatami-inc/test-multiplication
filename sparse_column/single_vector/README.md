# Sparse column-major LHS, single vector RHS

This is a stub, provided for consistency.
As far as I can tell, there is only one sensible way to compute this product.
For each LHS column $i$, we perform a sparse vector multiply-add to the output vector using the $i$-th entry of the RHS vector as the scaling factor.
We then repeat this process for each $i$ until all LHS columns have been traversed.

There's very little opportunity for blocking here.
We can't easily operate on fixed-size submatrices involving the LHS columns, as discussed in [`general/README.md`](../../general/README.md);
and we already follow the recommendations in the "Blocking to cache the dense vector" section, as we are always writing to the same output vector.
