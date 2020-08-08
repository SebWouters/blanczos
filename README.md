blanczos: block lanczos for gf(2)
=================================

This project contains an implementation of Peter Montgomery (1995), **A block Lanczos algorithm for finding dependencies over GF(2)**, Advances in Cryptology - Eurocrypt'95, pages 106-120 [[doi](https://doi.org/10.1007/3-540-49264-X_9)] [[pdf](https://link.springer.com/content/pdf/10.1007%2F3-540-49264-X_9.pdf)]. The sparse matrix is internally stored in a cache-optimized block format, with linear size of the blocks dependent on the matrix density. Each block is stored in compressed sparse row format.

C and C++ usage:

    uint32_t Nsol = blanczos(const uint32_t * B, const uint64_t N, const uint32_t Nrow, const uint32_t Ncol, uint64_t * result);

For Python usage, see python/test.py

Input:
* `B` with size `2*N`, containing the indices of `N` non-zero elements
* `B[2*i]` = row index of element `i`; with `B[2*i] < Nrow`
* `B[2*i+1]` = col index of element `i`; with `B[2*i+1] < Ncol`
* `Ncol >= Nrow + 64`

Output:
* `Nsol` is the number of nullspace vectors (at most 64)
* the lower `Nsol` bits of `result` (size `Ncol`) contain `Nsol` nullspace vectors of size `Ncol`

Copyright
=========

Copyright (c) 2020, Sebastian Wouters

All rights reserved.

blanczos is licensed under the BSD 3-Clause License. A copy of the License can be found in the file LICENSE in the root folder of this project.

