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
#include <math.h>
#include <iostream>
#include <algorithm>
#include <thread>
#include <vector>

namespace __blanczos_matrix
{

class sparse
{

public:

    /*
        B has size 2*N and contains the indices of N non-zero elements:
            - B[2*i]   = row index of element i; with B[2*i]   < Nrow
            - B[2*i+1] = col index of element i; with B[2*i+1] < Ncol
        cache is an upper bound for the memory per block
    */
    sparse(const uint32_t * B, const uint64_t N, const uint32_t Nrow, const uint32_t Ncol, const double cache)
        : Nbits(__bits__(N / (1.0 * Nrow * Ncol), cache)), Nrows(Nrow), Ncols(Ncol)
    {
        __singletons__(B, N);
        __build__(B, N);
    }

    ~sparse(){}

    // out (size Krows) = matrix * in (size Kcols)
    void multiply_regular(std::vector<uint64_t> & out, const std::vector<uint64_t> & in, const uint32_t nthreads) const
    {
        for (uint64_t & elem : out)
            elem = 0U;

        uint32_t batchsize = Brows / nthreads + (Brows % nthreads == 0U ? 0U : 1U);
        std::vector<std::thread> workers;
        workers.reserve(nthreads);
        for (uint32_t id = 0U; id < nthreads; ++id)
            workers.emplace_back([id, batchsize, this, &out, &in]()
            {
                const uint32_t begin = id * batchsize;
                const uint32_t end   = std::min(begin + batchsize, Brows);
                for (uint32_t brow = begin; brow < end; ++brow)
                    for (uint32_t bcol = 0U; bcol < Bcols; ++bcol)
                    {
                        const uint64_t start = jumps[brow + Brows * bcol];
                        const uint64_t stop  = jumps[brow + Brows * bcol + 1U];
                              uint64_t * left  = &out[brow << Nbits];
                        const uint64_t * right =  &in[bcol << Nbits];
                        for (uint64_t elem = start; elem < stop; ++elem)
                            left[indices[elem].rrow] ^= right[indices[elem].rcol];
                    }
            });
        for (std::thread& worker : workers)
            worker.join();
    }

    // out (size Kcols) = transpose(matrix) * in (size Krows)
    void multiply_transpose(std::vector<uint64_t> & out, const std::vector<uint64_t> & in, const uint32_t nthreads) const
    {
        for (uint64_t & elem : out)
            elem = 0U;

        uint32_t batchsize = Bcols / nthreads + (Bcols % nthreads == 0U ? 0U : 1U);
        std::vector<std::thread> workers;
        workers.reserve(nthreads);
        for (uint32_t id = 0U; id < nthreads; ++id)
            workers.emplace_back([id, batchsize, this, &out, &in]()
            {
                const uint32_t begin = id * batchsize;
                const uint32_t end   = std::min(begin + batchsize, Bcols);
                for (uint32_t bcol = begin; bcol < end; ++bcol)
                    for (uint32_t brow = 0U; brow < Brows; ++brow)
                    {
                        const uint64_t start = jumps[brow + Brows * bcol];
                        const uint64_t stop  = jumps[brow + Brows * bcol + 1U];
                              uint64_t * left  = &out[bcol << Nbits];
                        const uint64_t * right =  &in[brow << Nbits];
                        for (uint64_t elem = start; elem < stop; ++elem)
                            left[indices[elem].rcol] ^= right[indices[elem].rrow];
                    }
            });
        for (std::thread& worker : workers)
            worker.join();
    }

    uint32_t Nrow() const { return Nrows; }
    uint32_t Ncol() const { return Ncols; }
    uint32_t Krow() const { return Krows; }
    uint32_t Kcol() const { return Kcols; }
    uint32_t Nbit() const { return Nbits; }
    uint32_t kcol(const uint32_t ncol) const { return colmap[ncol]; }

private:

    struct relative
    {
        uint16_t rrow;
        uint16_t rcol;
    };

    // Block SIZE = 2^Nbits; and original dimensions
    const uint32_t Nbits;
    const uint32_t Nrows;
    const uint32_t Ncols;

    // Contributing dimensions
    uint32_t Krows;
    uint32_t Kcols;
    std::vector<uint32_t> rowmap; // size Nrows, rowmap[nrow] = krow < Krows if contributing and UINT32_MAX if non-contributing
    std::vector<uint32_t> colmap; // size Ncols, colmap[ncol] = kcol < Kcols if contributing and UINT32_MAX if non-contributing

    /*
        Blocks of SIZE x SIZE, with SIZE = 2^Nbits
            - start = jumps[brow + Brows * bcol]
            - stop  = jumps[brow + Brows * bcol + 1U]
            - start <= elem < stop:
                - krow = (brow << Nbits) + indices[elem].rrow
                - kcol = (bcol << Nbits) + indices[elem].rcol
    */
    uint32_t Brows;
    uint32_t Bcols;
    std::vector<uint64_t> jumps;   // size Brows * Bcols + 1U
    std::vector<relative> indices; // size jumps[Brows * Bcols]

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
        Brows = (Krows >> Nbits) + ((Krows & ((1U << Nbits) - 1U)) == 0U ? 0U : 1U);
        Bcols = (Kcols >> Nbits) + ((Kcols & ((1U << Nbits) - 1U)) == 0U ? 0U : 1U);
        const uint32_t Bsize = Brows * Bcols;

                              jumps = std::vector<uint64_t>(Bsize + 1U, 0U);
        std::vector<uint32_t> temp  = std::vector<uint32_t>(Bsize,      0U);

        for (uint64_t elem = 0U; elem < N; ++elem)
        {
            const uint32_t & krow = rowmap[B[(elem << 1U)]];
            const uint32_t & kcol = colmap[B[(elem << 1U) + 1U]];
            if ((krow != UINT32_MAX) && (kcol != UINT32_MAX))
                ++jumps[(krow >> Nbits) + Brows * (kcol >> Nbits) + 1U];
        }

        for (uint32_t blk = 0U; blk < Bsize; ++blk){ jumps[blk + 1U] += jumps[blk]; }

        indices = std::vector<relative>(jumps[Bsize], { 0U, 0U });
        for (uint64_t elem = 0U; elem < N; ++elem)
        {
            const uint32_t & krow = rowmap[B[(elem << 1U)]];
            const uint32_t & kcol = colmap[B[(elem << 1U) + 1U]];
            if ((krow != UINT32_MAX) && (kcol != UINT32_MAX))
            {
                const uint32_t blk = (krow >> Nbits) + Brows * (kcol >> Nbits);
                indices[jumps[blk] + temp[blk]++] = { static_cast<uint16_t>(krow & ((1U << Nbits) - 1U)),
                                                      static_cast<uint16_t>(kcol & ((1U << Nbits) - 1U)) };
            }
        }

        for (uint32_t blk = 0U; blk < Bsize; ++blk)
        {
            std::sort(&indices[0U] + jumps[blk], &indices[0U] + jumps[blk + 1U], [](const relative& left, const relative& right)
                {
                    return left.rcol < right.rcol ? true : (left.rcol > right.rcol ? false : (left.rrow < right.rrow));
                }
            );
            for (uint64_t elem = jumps[blk]; elem + 1U < jumps[blk + 1U]; ++elem)
                if ((indices[elem].rrow == indices[elem + 1U].rrow) && (indices[elem].rcol == indices[elem + 1U].rcol))
                {
                    std::cerr << "Two non-zero elements in B have identical indices." << std::endl;
                    exit(23);
                }
        }
    }

    /*
        Blocks of SIZE x SIZE, with SIZE = 2^Nbits:
        cache = 2 * SIZE * sizeof(uint64_t) + density * SIZE * SIZE * sizeof(relative)
    */
    static uint32_t __bits__(const double density, const double cache)
    {
        const double size = (sqrt(sizeof(uint64_t) * sizeof(uint64_t) + density * sizeof(relative) * cache) - sizeof(uint64_t)) / (density * sizeof(relative));
        const double nbit = log2(size);
        if (nbit > 16.0)
            return 16U;
        else if (nbit < 8.0)
            return 8U;
        else
            return static_cast<uint32_t>(nbit);
    }

}; // End of class sparse

} // End of namespace blanczos

