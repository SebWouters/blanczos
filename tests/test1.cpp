/*
    blanczos: block lanczos for gf(2)

    Copyright (c) 2020, Sebastian Wouters
    All rights reserved.

    This file is part of blanczos, licensed under the BSD 3-Clause License.
    A copy of the License can be found in the file LICENSE in the root
    folder of this project.
*/

#include "csr.hpp"
#include "sparse.hpp"

#include <stdint.h>
#include <random>
#include <chrono>
#include <vector>
#include <iostream>

int main()
{
    constexpr uint32_t nthreads = 2U;
    constexpr double   cache    = 512.0 * 1024.0;

    constexpr double   dens  = 0.00005;
    constexpr uint32_t Nrow  = 1U << 18U;
    constexpr uint32_t Ncol  = Nrow + 64U;
    constexpr uint32_t Niter = 10U;

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint32_t> dis_row(0U, Nrow - 1U);
    std::normal_distribution<double> dis_normal(dens * Nrow, 0.2 * dens * Nrow);
    std::uniform_int_distribution<uint64_t> dis_vec(0U, UINT64_MAX);

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

    __blanczos_matrix::csr * mx1 = new __blanczos_matrix::csr(&sparse[0U], sparse.size() >> 1U, Nrow, Ncol);

    const uint32_t Krow = mx1->Krow();
    const uint32_t Kcol = mx1->Kcol();

    std::vector<uint64_t> work = std::vector<uint64_t>(Krow, 0U);
    std::vector<uint64_t> V1   = std::vector<uint64_t>(Kcol, 0U);
    std::vector<uint64_t> V2   = std::vector<uint64_t>(Kcol, 0U);
    for (uint32_t col = 0U; col < Kcol; ++col)
        V1[col] = V2[col] = dis_vec(gen);

    std::cout << "Sparse (CSR): keeping " << mx1->Krow() << " of the " << Nrow << " rows." << std::endl;
    std::cout << "Sparse (CSR): keeping " << mx1->Kcol() << " of the " << Ncol << " cols." << std::endl;

    auto start = std::chrono::system_clock::now();
    for (uint32_t iter = 0U; iter < Niter; ++iter)
    {
        mx1->multiply_regular(work, V1);
        mx1->multiply_transpose(V1, work);
    }
    auto stop = std::chrono::system_clock::now();
    uint64_t time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

    std::cout << "Sparse (CSR): average time per (B^T * B) multiplication = " << (1e-9 * time_ns) / Niter << " seconds." << std::endl;
    delete mx1;

    __blanczos_matrix::sparse * mx2 = new __blanczos_matrix::sparse(&sparse[0U], sparse.size() >> 1U, Nrow, Ncol, cache);

    std::cout << "Sparse (BLK): Keeping " << mx2->Krow() << " of the " << Nrow << " rows." << std::endl;
    std::cout << "Sparse (BLK): Keeping " << mx2->Kcol() << " of the " << Ncol << " cols." << std::endl;
    std::cout << "Sparse (BLK): Using blocks of size 2^" << mx2->Nbit() << " x 2^" << mx2->Nbit() << "." << std::endl;

    size_t timeREG = 0U;
    size_t timeTRA = 0U;

    start = std::chrono::system_clock::now();
    for (uint32_t iter = 0U; iter < Niter; ++iter)
    {
        auto start1 = std::chrono::system_clock::now();
        mx2->multiply_regular(work, V2, nthreads);
        auto mid1   = std::chrono::system_clock::now();
        mx2->multiply_transpose(V2, work, nthreads);
        auto end1   = std::chrono::system_clock::now();
        timeREG += std::chrono::duration_cast<std::chrono::nanoseconds>(mid1 - start1).count();
        timeTRA += std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - mid1).count();
    }
    stop = std::chrono::system_clock::now();
    time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

    std::cout << "Sparse (BLK): average time per (B^T * B) multiplication = " << (1e-9 * time_ns) / Niter << " seconds." << std::endl;
    std::cout << "Sparse (BLK): average time per B         multiplication = " << (1e-9 * timeREG) / Niter << " seconds." << std::endl;
    std::cout << "Sparse (BLK): average time per B^T       multiplication = " << (1e-9 * timeTRA) / Niter << " seconds." << std::endl;
    delete mx2;
    
    for (uint32_t col = 0U; col < Kcol; ++col)
        if (V1[col] != V2[col])
        {
            std::cerr << "Difference in solution." << std::endl;
            return 255;
        }

    std::cout << "The same result is obtained." << std::endl;
    return 0;
}

