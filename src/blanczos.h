/*
    blanczos: block lanczos for gf(2)

    Copyright (c) 2020, Sebastian Wouters
    All rights reserved.

    This file is part of blanczos, licensed under the BSD 3-Clause License.
    A copy of the License can be found in the file LICENSE in the root
    folder of this project.
*/

#pragma once

#include <stdint.h>

/*
    B has size 2*N and contains the indices of N non-zero elements:
        - B[2*i]   = row index of element i; with B[2*i]   < Nrow
        - B[2*i+1] = col index of element i; with B[2*i+1] < Ncol
    Note that Ncol >= Nrow + 64 is expected.
    The nullspace is stored in result (size Ncol).
    The return value is the number of validly set (lower) bits in result.
*/
extern "C" uint32_t blanczos(const uint32_t * B, const uint64_t N, const uint32_t Nrow, const uint32_t Ncol, uint64_t * result);

