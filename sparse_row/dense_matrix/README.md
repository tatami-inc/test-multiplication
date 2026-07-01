# Sparse row-major product with multiple vectors

## Strategies

All strategies must iterate across consecutive rows of the LHS matrix in the outermost loop.
We will be loading rows on demand via the **tatami** interface, so the full matrix will not be available for random access.

### Column-major RHS

Here, we'll just re-use the best strategy from the [multiple vectors test suite](../multiple_vectors).
This involves blocking with 16 LHS rows and 4 accumulators for the dot product.
We test both row- and column-major output though it shouldn't make much difference as we don't iterate over RHS columns in the innermost loop.

### Naive row-major RHS

For each LHS row $l$, we iterate over its structural non-zeros.
For each column $k$ that contains a structural non-zero, we find the $k$-th RHS row and perform a vector multiply-add to the $l$-th output row,
using the non-zero value at $k$ as the scaling factor.
This is most obviously implemented when the output is row-major, in which case the $l$-th output row is trivially defined.
If the output is column-major, the $l$-th output row consists of the $l$-th element of each output column, which is quite non-contiguous;
this is largely unavoidable as the output doesn't align with any of the inputs.

### Blocked row-major RHS

This is much the same as the naive row-major strategy, but we only consider a block of $C$ elements along each RHS row at any given time.
For this block, we iterate down the RHS rows and perform a vector multiply-add to the output row;
then we move onto the next $C$ elements and repeat, until all RHS columns have been processed.

The idea is to keep $C$ small enough to avoid eviction of the active part of the output row from cache. 
This is particularly relevant for column-major output where the cache will be populated with unused values in the same cache line.

Other than the above strategy, there is no obvious blocking strategy here.
We can't easily block on multiple LHS rows, for the reasons described in [`general/README.md`](../../general/README.md).
The advice in the "Blocking to cache the dense vector" section isn't really applicable either, as our innermost loop consists of a dense vector multiply-add.

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
naive, row RHS, row output                        : 0.0580534 ± 0.000854938
naive, row RHS, column output                     : 0.127761 ± 0.000916853
blocked, column RHS, column output                : 0.0797759 ± 0.00101346
blocked, column RHS, row output                   : 0.0767074 ± 0.00229993
blocked 128, row RHS, row output                  : 0.0667878 ± 0.00136288
blocked 256, row RHS, row output                  : 0.0666751 ± 0.00064996
blocked 512, row RHS, row output                  : 0.0642879 ± 0.000362093
blocked 128, row RHS, column output               : 0.104537 ± 0.000659076
blocked 256, row RHS, column output               : 0.10154 ± 0.000841877
blocked 512, row RHS, column output               : 0.126328 ± 0.00126055

$ ./build/test -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row RHS, row output                        : 0.0840911 ± 0.000102788
naive, row RHS, column output                     : 0.126346 ± 0.00141991
blocked, column RHS, column output                : 0.0916553 ± 0.000338912
blocked, column RHS, row output                   : 0.0886739 ± 0.00211163
blocked 128, row RHS, row output                  : 0.0895941 ± 0.000139302
blocked 256, row RHS, row output                  : 0.0899717 ± 0.000289162
blocked 512, row RHS, row output                  : 0.0901458 ± 0.000358412
blocked 128, row RHS, column output               : 0.127407 ± 0.00042076
blocked 256, row RHS, column output               : 0.126854 ± 0.00155358
blocked 512, row RHS, column output               : 0.126947 ± 0.000411998

$ ./build/test -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row RHS, row output                        : 0.0693684 ± 0.00207116
naive, row RHS, column output                     : 0.739155 ± 0.00642194
blocked, column RHS, column output                : 0.11949 ± 0.00304828
blocked, column RHS, row output                   : 0.112032 ± 0.00382869
blocked 128, row RHS, row output                  : 0.0643858 ± 0.000462529
blocked 256, row RHS, row output                  : 0.0702421 ± 0.00208502
blocked 512, row RHS, row output                  : 0.0660661 ± 0.00134617
blocked 128, row RHS, column output               : 0.181473 ± 0.00311114
blocked 256, row RHS, column output               : 0.174896 ± 0.00306468
blocked 512, row RHS, column output               : 0.191844 ± 0.00753395

$ ./build/test -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row RHS, row output                        : 0.0396517 ± 0.000405834
naive, row RHS, column output                     : 0.0859285 ± 0.000403978
blocked, column RHS, column output                : 0.0479314 ± 0.000453107
blocked, column RHS, row output                   : 0.0473317 ± 0.000552319
blocked 128, row RHS, row output                  : 0.0423786 ± 0.000472069
blocked 256, row RHS, row output                  : 0.0425152 ± 0.000530565
blocked 512, row RHS, row output                  : 0.042891 ± 0.000421039
blocked 128, row RHS, column output               : 0.0849356 ± 0.000497282
blocked 256, row RHS, column output               : 0.0859169 ± 0.000403623
blocked 512, row RHS, column output               : 0.0858828 ± 0.000324581

$ ./build/test -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row RHS, row output                        : 0.0387713 ± 0.00118478
naive, row RHS, column output                     : 0.186487 ± 0.00329607
blocked, column RHS, column output                : 0.0930907 ± 0.00205479
blocked, column RHS, row output                   : 0.085156 ± 0.000761565
blocked 128, row RHS, row output                  : 0.0394711 ± 0.000744241
blocked 256, row RHS, row output                  : 0.0421912 ± 0.0012504
blocked 512, row RHS, row output                  : 0.04179 ± 0.00157544
blocked 128, row RHS, column output               : 0.104791 ± 0.00144385
blocked 256, row RHS, column output               : 0.10701 ± 0.00381514
blocked 512, row RHS, column output               : 0.12851 ± 0.00227159

$ ./build/test -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, row RHS, row output                        : 0.0833161 ± 0.00363464
naive, row RHS, column output                     : 0.150734 ± 0.00490139
blocked, column RHS, column output                : 0.108687 ± 0.0036227
blocked, column RHS, row output                   : 0.10362 ± 0.0024361
blocked 128, row RHS, row output                  : 0.0968997 ± 0.00250798
blocked 256, row RHS, row output                  : 0.0964904 ± 0.00307168
blocked 512, row RHS, row output                  : 0.0962094 ± 0.00401001
blocked 128, row RHS, column output               : 0.146913 ± 0.00355968
blocked 256, row RHS, column output               : 0.125998 ± 0.00301961
blocked 512, row RHS, column output               : 0.149201 ± 0.00267329

$ ./build/test -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, row RHS, row output                        : 0.0684087 ± 0.00303765
naive, row RHS, column output                     : 0.44381 ± 0.00309237
blocked, column RHS, column output                : 0.0917395 ± 0.00266562
blocked, column RHS, row output                   : 0.0869302 ± 0.00333048
blocked 128, row RHS, row output                  : 0.0777837 ± 0.00321725
blocked 256, row RHS, row output                  : 0.0781021 ± 0.00476133
blocked 512, row RHS, row output                  : 0.0792641 ± 0.00538608
blocked 128, row RHS, column output               : 0.11353 ± 0.00377774
blocked 256, row RHS, column output               : 0.110871 ± 0.00648472
blocked 512, row RHS, column output               : 0.149648 ± 0.0041393
```

## Single-precision results

Again, but with single-precision floats.

```console
$ ./build/test_float -r 1000 -c 1000 -H 1000
Results for 1000 x 1000 x 1000
naive, row RHS, row output                        : 0.0199549 ± 0.00118622
naive, row RHS, column output                     : 0.109614 ± 0.00187057
blocked, column RHS, column output                : 0.0488601 ± 0.000989886
blocked, column RHS, row output                   : 0.0462167 ± 0.00161444
blocked 128, row RHS, row output                  : 0.0314127 ± 0.00184375
blocked 256, row RHS, row output                  : 0.023861 ± 0.00104315
blocked 512, row RHS, row output                  : 0.0245345 ± 0.00143879
blocked 128, row RHS, column output               : 0.0861077 ± 0.0036111
blocked 256, row RHS, column output               : 0.0820253 ± 0.0033982
blocked 512, row RHS, column output               : 0.110495 ± 0.00496927

$ ./build/test_float -r 1000 -c 10000 -H 100
Results for 1000 x 10000 x 100
naive, row RHS, row output                        : 0.0345673 ± 0.00084894
naive, row RHS, column output                     : 0.093332 ± 0.002001
blocked, column RHS, column output                : 0.0576968 ± 0.0016289
blocked, column RHS, row output                   : 0.0557325 ± 0.000279281
blocked 128, row RHS, row output                  : 0.0401331 ± 0.000563717
blocked 256, row RHS, row output                  : 0.0367982 ± 0.000407448
blocked 512, row RHS, row output                  : 0.0377725 ± 0.00083929
blocked 128, row RHS, column output               : 0.0932589 ± 0.000819586
blocked 256, row RHS, column output               : 0.0915103 ± 0.000516813
blocked 512, row RHS, column output               : 0.0943374 ± 0.00337136

$ ./build/test_float -r 1000 -c 100 -H 10000
Results for 1000 x 100 x 10000
naive, row RHS, row output                        : 0.0258554 ± 0.00158974
naive, row RHS, column output                     : 0.631647 ± 0.00236086
blocked, column RHS, column output                : 0.0828117 ± 0.00192209
blocked, column RHS, row output                   : 0.0752683 ± 0.00148127
blocked 128, row RHS, row output                  : 0.0224248 ± 0.0013015
blocked 256, row RHS, row output                  : 0.0268823 ± 0.00163215
blocked 512, row RHS, row output                  : 0.0251725 ± 0.000854037
blocked 128, row RHS, column output               : 0.137031 ± 0.00175863
blocked 256, row RHS, column output               : 0.139397 ± 0.00345552
blocked 512, row RHS, column output               : 0.156128 ± 0.0014896

$ ./build/test_float -r 10000 -c 1000 -H 100
Results for 10000 x 1000 x 100
naive, row RHS, row output                        : 0.0230611 ± 0.000318291
naive, row RHS, column output                     : 0.0789659 ± 0.000369125
blocked, column RHS, column output                : 0.0394367 ± 0.000372288
blocked, column RHS, row output                   : 0.0380489 ± 0.000288291
blocked 128, row RHS, row output                  : 0.0301117 ± 0.000280637
blocked 256, row RHS, row output                  : 0.0246379 ± 0.000437824
blocked 512, row RHS, row output                  : 0.0249032 ± 0.000439102
blocked 128, row RHS, column output               : 0.0797317 ± 0.00038737
blocked 256, row RHS, column output               : 0.0790595 ± 0.000491343
blocked 512, row RHS, column output               : 0.0788373 ± 0.00034367

$ ./build/test_float -r 10000 -c 100 -H 1000
Results for 10000 x 100 x 1000
naive, row RHS, row output                        : 0.0187499 ± 0.000411621
naive, row RHS, column output                     : 0.238609 ± 0.00169203
blocked, column RHS, column output                : 0.0731233 ± 0.000508713
blocked, column RHS, row output                   : 0.0708886 ± 0.000494666
blocked 128, row RHS, row output                  : 0.0208946 ± 0.00045291
blocked 256, row RHS, row output                  : 0.0212928 ± 0.000656271
blocked 512, row RHS, row output                  : 0.0208948 ± 0.00064648
blocked 128, row RHS, column output               : 0.0964484 ± 0.00138662
blocked 256, row RHS, column output               : 0.100247 ± 0.000921871
blocked 512, row RHS, column output               : 0.132663 ± 0.000382331

$ ./build/test_float -r 100 -c 10000 -H 1000
Results for 100 x 10000 x 1000
naive, row RHS, row output                        : 0.0451977 ± 0.00160425
naive, row RHS, column output                     : 0.129111 ± 0.00108321
blocked, column RHS, column output                : 0.067338 ± 0.000614795
blocked, column RHS, row output                   : 0.0659567 ± 0.00117653
blocked 128, row RHS, row output                  : 0.0537058 ± 0.000802451
blocked 256, row RHS, row output                  : 0.052093 ± 0.00103617
blocked 512, row RHS, row output                  : 0.0527055 ± 0.000493052
blocked 128, row RHS, column output               : 0.115166 ± 0.0020727
blocked 256, row RHS, column output               : 0.100671 ± 0.000636266
blocked 512, row RHS, column output               : 0.122969 ± 0.00106657

$ ./build/test_float -r 100 -c 1000 -H 10000
Results for 100 x 1000 x 10000
naive, row RHS, row output                        : 0.0372867 ± 0.00054338
naive, row RHS, column output                     : 0.245643 ± 0.00249169
blocked, column RHS, column output                : 0.0611744 ± 0.000322518
blocked, column RHS, row output                   : 0.0593245 ± 0.00112578
blocked 128, row RHS, row output                  : 0.0542446 ± 0.000606041
blocked 256, row RHS, row output                  : 0.0454118 ± 0.000392657
blocked 512, row RHS, row output                  : 0.0431327 ± 0.00156864
blocked 128, row RHS, column output               : 0.0981634 ± 0.00153266
blocked 256, row RHS, column output               : 0.0905086 ± 0.000526096
blocked 512, row RHS, column output               : 0.125176 ± 0.000540498
```

## Conclusion

Row-major RHS to row-major output is very efficient, even beating out the "obvious" product with a column-major RHS.
This is probably to be expected given that the innermost loop involves contiguous memory and can be easily vectorization.

Blocking in row-major RHS to row-major output does not help, probably because the loop restart overhead offsets any improvement in cache-friendliness.

Blocking in row-major RHS to column-major output can be very helpful.
Presumably this mitigates a lot of the cache evictions from scattered accesses to separate columns.
However, it's still the worst-performing choice out of all the configurations.
