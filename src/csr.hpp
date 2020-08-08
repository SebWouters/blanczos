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
#include <iostream>
#include <algorithm>
#include <vector>

namespace __blanczos_matrix
{

class csr
{

public:

    /*
        B has size 2*N and contains the indices of N non-zero elements:
            - B[2*i]   = row index of element i; with B[2*i]   < Nrow
            - B[2*i+1] = col index of element i; with B[2*i+1] < Ncol
    */
    csr(const uint32_t * B, const uint64_t N, const uint32_t Nrow, const uint32_t Ncol)
        : Nrows(Nrow), Ncols(Ncol)
    {
        __singletons__(B, N);
        __build__(B, N);
    }

    ~csr(){}

    void multiply_regular(std::vector<uint64_t> & out, const std::vector<uint64_t> & in) const
    {
        for (uint32_t krow = 0U; krow < Krows; ++krow){ out[krow] = 0U; }
        for (uint32_t krow = 0U; krow < Krows; ++krow)
            for (uint64_t idx = jumps[krow]; idx < jumps[krow + 1U]; ++idx)
                out[krow] ^= in[columns[idx]];
    }

    void multiply_transpose(std::vector<uint64_t> & out, const std::vector<uint64_t> & in) const
    {
        for (uint32_t kcol = 0U; kcol < Kcols; ++kcol){ out[kcol] = 0U; }
        for (uint32_t krow = 0U; krow < Krows; ++krow)
            for (uint64_t idx = jumps[krow]; idx < jumps[krow + 1U]; ++idx)
                out[columns[idx]] ^= in[krow];
    }

    uint32_t Nrow() const { return Nrows; }
    uint32_t Ncol() const { return Ncols; }
    uint32_t Krow() const { return Krows; }
    uint32_t Kcol() const { return Kcols; }

private:

    // Original dimensions
    const uint32_t Nrows;
    const uint32_t Ncols;

    // Contributing dimensions
    uint32_t Krows;
    uint32_t Kcols;
    std::vector<uint32_t> rowmap;
    std::vector<uint32_t> colmap;

    // Compressed sparse row (CSR): columns[jumps[row] : jumps[row + 1]] correspond to row
    std::vector<uint64_t> jumps;
    std::vector<uint32_t> columns;

    void __singletons__(const uint32_t * B, const uint64_t N)
    {
        rowmap = std::vector<uint32_t>(Nrows, 0U);
        colmap = std::vector<uint32_t>(Ncols, 0U);

        for (uint64_t elem = 0U; elem < N; ++elem){ ++rowmap[B[elem << 1U]]; }
        for (uint64_t elem = 0U; elem < N; ++elem)
            if (rowmap[B[(elem << 1U)]] < 2U)
                colmap[B[(elem << 1U) + 1U]] = UINT32_MAX;

        Krows = 0U; for (uint32_t nrow = 0U; nrow < Nrows; ++nrow){ rowmap[nrow] = rowmap[nrow] >= 2U ? Krows++ : UINT32_MAX; }
        Kcols = 0U; for (uint32_t ncol = 0U; ncol < Ncols; ++ncol){ colmap[ncol] = colmap[ncol] == 0U ? Kcols++ : UINT32_MAX; }
    }

    void __build__(const uint32_t * B, const uint64_t N)
    {
                              jumps = std::vector<uint64_t>(Krows + 1U, 0U);
        std::vector<uint32_t> temp  = std::vector<uint32_t>(Krows,      0U);

        for (uint64_t elem = 0U; elem < N; ++elem)
        {
            const uint32_t & krow = rowmap[B[(elem << 1U)]];
            const uint32_t & kcol = colmap[B[(elem << 1U) + 1U]];
            if ((krow != UINT32_MAX) && (kcol != UINT32_MAX))
                ++jumps[krow + 1U];
        }

        for (uint32_t krow = 0U; krow < Krows; ++krow){ jumps[krow + 1U] += jumps[krow]; }

        columns = std::vector<uint32_t>(jumps[Krows], 0U);
        for (uint64_t elem = 0U; elem < N; ++elem)
        {
            const uint32_t & krow = rowmap[B[(elem << 1U)]];
            const uint32_t & kcol = colmap[B[(elem << 1U) + 1U]];
            if ((krow != UINT32_MAX) && (kcol != UINT32_MAX))
                columns[jumps[krow] + temp[krow]++] = kcol;
        }

        for (uint32_t krow = 0U; krow < Krows; ++krow)
        {
            std::sort(&columns[0U] + jumps[krow], &columns[0U] + jumps[krow + 1U]);
            for (uint64_t elem = jumps[krow]; elem + 1U < jumps[krow + 1U]; ++elem)
                if (columns[elem] == columns[elem + 1U])
                {
                    std::cerr << "Two non-zero elements in B have identical indices." << std::endl;
                    exit(19);
                }
        }
    }

}; // End of class csr

} // End of namespace blanczos

