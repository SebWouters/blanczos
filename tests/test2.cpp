/*
    blanczos: block lanczos for gf(2)

    Copyright (c) 2020, Sebastian Wouters
    All rights reserved.

    This file is part of blanczos, licensed under the BSD 3-Clause License.
    A copy of the License can be found in the file LICENSE in the root
    folder of this project.
*/

#include "blanczos.h"

#include <stdint.h>
#include <random>
#include <chrono>
#include <vector>
#include <iostream>

int main()
{
    constexpr double   dens  = 0.01;
    constexpr uint32_t Nrow  = 1U << 12U;
    constexpr uint32_t Ncol  = Nrow + 64U;

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint32_t> dis_row(0U, Nrow - 1U);
    std::normal_distribution<double> dis_normal(dens * Nrow, 0.2 * dens * Nrow);

    std::vector<uint32_t> sparse;
    for (uint32_t col = 0U; col < Ncol; ++col)
    {
        const uint32_t Nsamples = lrint(dis_normal(gen));
        const uint64_t Nbefore  = sparse.size();
        for (uint32_t elem = 0U; elem < Nsamples; ++elem)
        {
            while (sparse.size() == Nbefore + 2U * elem)
            {
                const uint32_t row = dis_row(gen);
                bool exists = false;
                for (uint64_t old = Nbefore; old < sparse.size(); old += 2U)
                    if (sparse[old] == row)
                    {
                        exists = true;
                        break;
                    }
                if (!exists)
                {
                    sparse.push_back(row);
                    sparse.push_back(col);
                }
            }
        }
    }

    std::vector<uint64_t> result = std::vector<uint64_t>(Ncol, 0U);
    /*uint32_t Nsol =*/ blanczos(&sparse[0U], sparse.size() >> 1U, Nrow, Ncol, &result[0U]);
    
    return 0;
}

