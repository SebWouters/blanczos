/*
    blanczos: block lanczos for gf(2)

    Copyright (c) 2020, Sebastian Wouters
    All rights reserved.

    This file is part of blanczos, licensed under the BSD 3-Clause License.
    A copy of the License can be found in the file LICENSE in the root
    folder of this project.
*/

#include "sparse.hpp"

#include <random>
#include <iostream>
#include <vector>
#include <array>
#include <thread>

namespace
{


using vec64 = std::vector<uint64_t>;
using arr64 = std::array<uint64_t, 64U>;
constexpr uint64_t one64 = static_cast<uint64_t>(1U);


// result[i].bit[j] = XOR_k left[k].bit[i] * right[k].bit[j]
inline void dot(const vec64 & left, const vec64 & right, arr64 & result)
{
    for (uint64_t & elem : result){ elem = 0U; }

    for (uint32_t m = 0U; m < 64U; ++m)
    {
        uint64_t value = 0U;
        for (size_t k = 0U; k < left.size(); ++k)
            value ^= (left[k] << m) & right[k];
        for (uint32_t i = 0U; i < 64U - m; ++i)
            result[i] |= value & (one64 << (i + m));
    }
    for (uint32_t m = 1U; m < 64U; ++m)
    {
        uint64_t value = 0U;
        for (size_t k = 0U; k < left.size(); ++k)
            value ^= (left[k] >> m) & right[k];
        for (uint32_t i = 0U; i < 64U - m; ++i)
            result[i + m] |= value & (one64 << i);
    }
}


template <class type64> // type64 = arr64 or vec64
inline bool nonzero(const type64 & test)
{
    for (const uint64_t & elem : test)
        if (elem)
            return true;
    return false;
}


template <class type64> // type64 = arr64 or vec64
inline bool nonzero(const type64 & test, const uint64_t mask)
{
    for (const uint64_t & elem : test)
        if (elem & mask)
            return true;
    return false;
}


// result[i].bit[j] = XOR_k left[i].bit[k] * right[k].bit[j]
template <class type64> // type64 = arr64 or vec64
inline void mul(const type64 & left, const arr64 & right, type64 & result)
{
    for (uint32_t idx = 0U; idx < left.size(); ++idx)
    {
        result[idx]
            = (( left[idx]         & one64) * right[ 0U])
            ^ (((left[idx] >>  1U) & one64) * right[ 1U])
            ^ (((left[idx] >>  2U) & one64) * right[ 2U])
            ^ (((left[idx] >>  3U) & one64) * right[ 3U])
            ^ (((left[idx] >>  4U) & one64) * right[ 4U])
            ^ (((left[idx] >>  5U) & one64) * right[ 5U])
            ^ (((left[idx] >>  6U) & one64) * right[ 6U])
            ^ (((left[idx] >>  7U) & one64) * right[ 7U])
            ^ (((left[idx] >>  8U) & one64) * right[ 8U])
            ^ (((left[idx] >>  9U) & one64) * right[ 9U])
            ^ (((left[idx] >> 10U) & one64) * right[10U])
            ^ (((left[idx] >> 11U) & one64) * right[11U])
            ^ (((left[idx] >> 12U) & one64) * right[12U])
            ^ (((left[idx] >> 13U) & one64) * right[13U])
            ^ (((left[idx] >> 14U) & one64) * right[14U])
            ^ (((left[idx] >> 15U) & one64) * right[15U])
            ^ (((left[idx] >> 16U) & one64) * right[16U])
            ^ (((left[idx] >> 17U) & one64) * right[17U])
            ^ (((left[idx] >> 18U) & one64) * right[18U])
            ^ (((left[idx] >> 19U) & one64) * right[19U])
            ^ (((left[idx] >> 20U) & one64) * right[20U])
            ^ (((left[idx] >> 21U) & one64) * right[21U])
            ^ (((left[idx] >> 22U) & one64) * right[22U])
            ^ (((left[idx] >> 23U) & one64) * right[23U])
            ^ (((left[idx] >> 24U) & one64) * right[24U])
            ^ (((left[idx] >> 25U) & one64) * right[25U])
            ^ (((left[idx] >> 26U) & one64) * right[26U])
            ^ (((left[idx] >> 27U) & one64) * right[27U])
            ^ (((left[idx] >> 28U) & one64) * right[28U])
            ^ (((left[idx] >> 29U) & one64) * right[29U])
            ^ (((left[idx] >> 30U) & one64) * right[30U])
            ^ (((left[idx] >> 31U) & one64) * right[31U])
            ^ (((left[idx] >> 32U) & one64) * right[32U])
            ^ (((left[idx] >> 33U) & one64) * right[33U])
            ^ (((left[idx] >> 34U) & one64) * right[34U])
            ^ (((left[idx] >> 35U) & one64) * right[35U])
            ^ (((left[idx] >> 36U) & one64) * right[36U])
            ^ (((left[idx] >> 37U) & one64) * right[37U])
            ^ (((left[idx] >> 38U) & one64) * right[38U])
            ^ (((left[idx] >> 39U) & one64) * right[39U])
            ^ (((left[idx] >> 40U) & one64) * right[40U])
            ^ (((left[idx] >> 41U) & one64) * right[41U])
            ^ (((left[idx] >> 42U) & one64) * right[42U])
            ^ (((left[idx] >> 43U) & one64) * right[43U])
            ^ (((left[idx] >> 44U) & one64) * right[44U])
            ^ (((left[idx] >> 45U) & one64) * right[45U])
            ^ (((left[idx] >> 46U) & one64) * right[46U])
            ^ (((left[idx] >> 47U) & one64) * right[47U])
            ^ (((left[idx] >> 48U) & one64) * right[48U])
            ^ (((left[idx] >> 49U) & one64) * right[49U])
            ^ (((left[idx] >> 50U) & one64) * right[50U])
            ^ (((left[idx] >> 51U) & one64) * right[51U])
            ^ (((left[idx] >> 52U) & one64) * right[52U])
            ^ (((left[idx] >> 53U) & one64) * right[53U])
            ^ (((left[idx] >> 54U) & one64) * right[54U])
            ^ (((left[idx] >> 55U) & one64) * right[55U])
            ^ (((left[idx] >> 56U) & one64) * right[56U])
            ^ (((left[idx] >> 57U) & one64) * right[57U])
            ^ (((left[idx] >> 58U) & one64) * right[58U])
            ^ (((left[idx] >> 59U) & one64) * right[59U])
            ^ (((left[idx] >> 60U) & one64) * right[60U])
            ^ (((left[idx] >> 61U) & one64) * right[61U])
            ^ (((left[idx] >> 62U) & one64) * right[62U])
            ^ (((left[idx] >> 63U) & one64) * right[63U]);
    }
}


template <class type64>
inline void rox(const type64 & left, const type64 & right, type64 & result)
{
    for (size_t idx = 0U; idx < left.size(); ++idx)
        result[idx] = left[idx] ^ right[idx];
}


// mx[i].bit[j] += delta_ij
inline void plus1(arr64 & mx)
{
    for (uint32_t idx = 0U; idx < 64U; ++idx)
        mx[idx] ^= one64 << idx;
}


/*
    Montgomery (1995), page 116, figure 1: pseudocode for selecting S[i] and W[i].
        - input: T and mask (mask.bit[j] designates if j is present in S[i-1])
        - output: right (Winv[i]) and return (return.bit[j] designates if j is present in S[i])
*/
inline uint64_t invert(const arr64 & T, const uint64_t mask, std::array<uint32_t, 64U> & c, arr64 & left, arr64 & right)
{
    // Non-used bits in mask are given precedence in c
    uint32_t head = 0U;
    uint32_t tail = 63U;
    for (uint32_t bit = 0U; bit < 64U; ++bit)
        c[(mask >> bit) & one64 ? tail-- : head++] = bit;
    uint64_t newmask = 0U;

    std::copy(T.cbegin(), T.cend(), left.begin());
    for (uint32_t i = 0U; i < 64U; ++i)
        right[i] = one64 << i;

    for (uint32_t j = 0U; j < 64U; ++j)
    {
        const uint32_t & cj = c[j];
        uint32_t k = j;
        while ((k < 64U) && (((left[c[k]] >> cj) & one64) == 0U))
            ++k;
        if (k != 64U) // Found a pivot element
        {
            if (k != j) // Swap entire row
            {
                std::swap( left[cj],  left[c[k]]);
                std::swap(right[cj], right[c[k]]);
            }
            newmask |= one64 << cj; // S.append(cj)
            // Ensure work[cj, cj] is only non-zero element on col cj
            for (uint32_t row = 0U; row < 64U; ++row)
                if ((row != cj) && ((left[row] >> cj) & one64))
                {
                    left [row] ^= left [cj];
                    right[row] ^= right[cj];
                }
        }
        else // No pivot element found
        {
            uint32_t k = j;
            while ((k < 64U) && (((right[c[k]] >> cj) & one64) == 0U))
                ++k;
            if (k != j) // Swap entire row
            {
                std::swap( left[cj],  left[c[k]]);
                std::swap(right[cj], right[c[k]]);
            }
            // Ensure work[cj, cj + 64U] is only non-zero element on col cj + N
            for (uint32_t row = 0U; row < 64U; ++row)
                if ((row != cj) && ((right[row] >> cj) & one64))
                {
                    left [row] ^= left [cj];
                    right[row] ^= right[cj];
                }
            left [cj] = 0U;
            right[cj] = 0U;
        }
    }

    for (uint32_t i = 0U; i < 64U; ++i)
        if (((newmask >> i) & one64) == 0U)
            right[i] = 0U;
    for (uint64_t & elem : right)
        elem &= newmask;

    return newmask;
}


/*
    Gaussian elimination on the 128 vectors in B x { vec1, vec2 } to find linear combinations in { vec1, vec2 } which lie in the nullspace of B.
    The return value contains up to 64 nullspace vectors (vec64); and the number of retrieved nullspace vectors (uint32_t).
*/
inline std::pair<uint32_t, vec64> elimination(const vec64 & Bvec1, const vec64 & Bvec2, const vec64 & vec1, const vec64 & vec2)
{
    // B has size Zrows x Zcols
    const     uint64_t Zrows = Bvec1.size();
    const     uint64_t Zcols =  vec1.size();
    const     uint64_t Zsize = std::max(Zrows, Zcols);
    constexpr uint32_t Zvecs = 128U;

    // Place B x { vec1, vec2 } in vectors; make an identity matrix solution
    std::vector<uint8_t> vectors  = std::vector<uint8_t>(Zsize * Zvecs, 0U);
    std::vector<uint8_t> solution = std::vector<uint8_t>(Zvecs * Zvecs, 0U);
    for (uint32_t vec = 0U; vec < Zvecs; ++vec)
    {
        for (uint32_t row = 0U; row < Zrows; ++row)
            vectors[row + Zrows * vec] = (vec < 64U ? (Bvec1[row] >> vec) : (Bvec2[row] >> (vec - 64U))) & one64;
        solution[vec + Zvecs * vec] = 1U;
    }

    // Make vectors(row, sol) lower triangular; keep track of linear combinations in solution(vec, sol)
    uint32_t solindex = 0U;
    for (uint32_t row = 0U; row < Zrows; ++row)
    {
        uint32_t search = solindex;
        while ((search < Zvecs) && (vectors[row + Zrows * search] == 0U))
            ++search;
        if (search != Zvecs)
        {
            if (search != solindex)
            {
                for (uint32_t idx = row; idx < Zrows; ++idx)
                    std::swap(vectors[idx + Zrows * solindex], vectors[idx + Zrows * search]);
                for (uint32_t vec = 0U; vec < Zvecs; ++vec)
                    std::swap(solution[vec + Zvecs * solindex], solution[vec + Zvecs * search]);
            }
            for (uint32_t sol = solindex + 1U; sol < Zvecs; ++sol)
                if (vectors[row + Zrows * sol])
                {
                    for (uint32_t idx = row; idx < Zrows; ++idx)
                        vectors[idx + Zrows * sol] ^= vectors[idx + Zrows * solindex];
                    for (uint32_t vec = 0U; vec < Zvecs; ++vec)
                        solution[vec + Zvecs * sol] ^= solution[vec + Zvecs * solindex];
                }
            ++solindex;
        }
    }

    for (uint8_t & elem : vectors)
        elem = 0U;

    // Construct the linear combination of { vec1, vec2 } in the nullspace of B
    for (uint32_t sol = solindex; sol < Zvecs; ++sol)
        for (uint32_t vec = 0U; vec < Zvecs; ++vec)
            if (solution[vec + Zvecs * sol])
            {
                for (uint32_t col = 0U; col < Zcols; ++col)
                    vectors[col + Zcols * sol] ^= (vec < 64U ? (vec1[col] >> vec) : (vec2[col] >> (vec - 64U))) & one64;
            }

    // Place up to 64 non-zero vectors in the nullspace in result
    uint32_t Nsols  = 0U;
    uint32_t number = 0U;
    vec64 result = vec64(Zcols, 0U);
    for (uint32_t vec = 0U; vec < Zvecs; ++vec)
    {
        bool use = false;
        for (uint32_t col = 0U; col < Zcols; ++col)
            if (vectors[col + Zcols * vec])
            {
                use = Nsols < 64U;
                ++number;
                break;
            }
        if (use)
        {
            for (uint32_t col = 0U; col < Zcols; ++col)
                result[col] |= static_cast<uint64_t>(vectors[col + Zcols * vec]) << Nsols;
            ++Nsols;
        }
    }

    std::cout << "Returning " << Nsols << " of " << number << " non-zero nullspace vectors retrieved within " << Zvecs - solindex << " candidates." << std::endl;
    return { Nsols, result };
}


inline uint32_t runner(const __blanczos_matrix::sparse & matrix, vec64 & X, uint64_t * result, const uint32_t nthreads)
{
    const uint32_t Ncol = matrix.Ncol();
    const uint32_t Krow = matrix.Krow();
    const uint32_t Kcol = matrix.Kcol();

    if (X.size() != Kcol)
    {
        std::cerr << "X should have size matrix.Kcol()." << std::endl;
        exit(11);
    }

    // Quoted variables refer to the notation of Montgomery (1995).
    // On incoming,  X contains "Y"   (section 7).
    // On iterating, X contains "X-Y" (section 7).
    vec64 Q     = vec64(Kcol, 0U); // Q   contains "V[0] = A Y" and does NOT change on iterating.
    vec64 V0    = vec64(Kcol, 0U); // V0  contains "V[i]"
    vec64 V1    = vec64(Kcol, 0U); // V1  contains "V[i-1]"
    vec64 V2    = vec64(Kcol, 0U); // V2  contains "V[i-2]"
    vec64 AV0   = vec64(Kcol, 0U); // AV0 contains "A V[i]"
    vec64 N     = vec64(Kcol, 0U); // Not needed beyond while loop, contains "V[i+1]" in loop.
    vec64 workV = vec64(Kcol, 0U);
    vec64 work1 = vec64(Krow, 0U); // !! ROW
    vec64 work2 = vec64(Krow, 0U); // !! ROW

    arr64 T0  = {}; // T0  contains "V[i]^T A V[i]"
    arr64 T1  = {}; // T1  contains "V[i-1]^T A V[i-1]"
    arr64 W0i = {}; // W0i contains "Winv[i]"
    arr64 W1i = {}; // W1i contains "Winv[i-1]"
    arr64 W2i = {}; // W2i contains "Winv[i-2]"
    arr64 G   = {}; // G   contains "G[i] = V[i-1]^T A^2 V[i-1] S[i-1] S[i-1]^T + V[i-1]^T A V[i-1]" (custom helper variable)
    arr64 F   = {}; // Not needed beyond while loop, contains "F[i+1]" in loop (equation (19)).
    arr64 E   = {}; // Not needed beyond while loop, contains "E[i+1]" in loop (equation (19)).
    arr64 D   = {}; // Not needed beyond while loop, contains "D[i+1]" in loop (equation (19)).

    std::array<uint32_t, 64U> temp = {}; // !! 32 bit

    matrix.multiply_regular(work1, X, nthreads);
    matrix.multiply_transpose(Q, work1, nthreads);   // Q   = "V[0]" = "A Y"
    std::copy(Q.cbegin(), Q.cend(), V0.begin());     // V0  = "V[i]" (with i=0)
    matrix.multiply_regular(work1, V0, nthreads);
    matrix.multiply_transpose(AV0, work1, nthreads); // AV0 = "A V[i]" (with i=0)
    dot(V0, AV0, T0);                                // T0  = "V[i]^T A V[i]" (with i=0)

    uint32_t iter = 0U;
    uint64_t mask = UINT64_MAX; // mask defines indices in S[i] S[i]^T (with i=0), which is initially full rank

    while (nonzero(T0))
    {
        std::cout << "Lanczos iteration " << iter << std::endl;

        // Page 112, just before equation (15): "We achieve this by requiring that all vectors in V[j+1] be used either in W[j+1] or in W[j+2]."
        // Page 115, last sentence 2nd paragraph: "Afterwards I check whether all nonzero columns of V[i+1] were chosen in S[i] and/or S[i-1]."
        uint64_t mask2 = mask; // indices in S[i-1]
        mask = invert(T0, mask, temp, D, W0i);
        mask2 = ~(mask | mask2); // Indices not in S[i] and/or S[i-1]
        if (mask2)
        {
            if (nonzero(V1, mask2))
                return UINT32_MAX; // Fail
            else
                std::cout << "Not all indices in S[i] U S[i-1], but V[i-1] zero for those indices." << std::endl;
        }

        // X[i+1] = X[i] + V[i] Winv[i] V[i]^T V[0] (equation (20)).
        dot(V0, Q, D);
        mul(W0i, D, E);
        mul(V0, E, N);
        rox(X, N, X);

        // F[i+1] = -Winv[i-2] (I - V[i-1]^T A V[i-1] Winv[i-1]) (V[i-1]^T A^2 V[i-1] S[i-1] S[i-1]^T + V[i-1]^T A V[i-1]) S[i] S[i]^T (equation (19))
        //        =  Winv[i-2] (I + T[i-1] Winv[i-1]) G[i] S[i] S[i]^T (GF(2) specific)
        mul(T1, W1i, D);
        plus1(D);
        mul(W2i, D, E);
        mul(E, G, F);
        for (uint64_t & elem : F){ elem &= mask; }

        // E[i+1] = -Winv[i-1] V[i]^T A V[i] S[i] S[i]^T (equation (19))
        //        =  Winv[i-1] T[i] S[i] S[i]^T (GF(2) specific)
        mul(W1i, T0, E);
        for (uint64_t & elem : E){ elem &= mask; }

        // G[i+1] = V[i]^T A^2 V[i] S[i] S[i]^T + V[i]^T A V[i] (custom helper variable)
        //        = AV[i]^T AV[i] S[i] S[i]^T + T[i]
        dot(AV0, AV0, D);
        for (uint64_t & elem : D){ elem &= mask; }
        rox(D, T0, G);

        // D[i+1] = I - Winv[i] (V[i]^T A^2 V[i] S[i] S[i]^T + V[i]^T A V[i]) (equation (19))
        //        = I + Winv[i] G[i+1] (GF(2) specific)
        mul(W0i, G, D);
        plus1(D);

        // V[i+1] = AV[i] S[i] S[i]^T + V[i] D[i+1] + V[i-1] E[i+1] + V[i-2] F[i+1] (equation (18))
        for (uint64_t & elem : AV0){ elem &= mask; }
        mul(V0, D, workV);
        rox(AV0, workV, N);
        mul(V1, E, workV);
        rox(N, workV, N);
        mul(V2, F, workV);
        rox(N, workV, N);

        // Rotate
        std::swap(V2, V1);
        std::swap(V1, V0);
        std::swap(V0, N);
        matrix.multiply_regular(work1, V0, nthreads);
        matrix.multiply_transpose(AV0, work1, nthreads);
        std::swap(W2i, W1i);
        std::swap(W1i, W0i);
        std::swap(T1, T0);
        dot(V0, AV0, T0);
        ++iter;
    }

    // The nullspace of B is part of the nullspace of A = B^T B, but not necessarily vice versa.
    // Make a reduced matrix B x (X, V0) = (work1, work2) of size Krow x 128
    matrix.multiply_regular(work1,  X, nthreads);
    matrix.multiply_regular(work2, V0, nthreads);
    // Perform Gaussian elimination on (work1, work2) to find linear combination in nullspace: solution[kcol].bit[sol]
    const std::pair<uint32_t, vec64> solution = elimination(work1, work2, X, V0);
    matrix.multiply_regular(work1, solution.second, nthreads);
    if (nonzero(work1))
    {
        std::cerr << "The nullspace is incorrect." << std::endl;
        exit(13);
    }
    for (uint32_t ncol = 0U; ncol < Ncol; ++ncol)
    {
        const uint32_t kcol = matrix.kcol(ncol);
        result[ncol] = kcol == UINT32_MAX ? 0U : solution.second[kcol];
    }

    return solution.first;
}


} // End of anonymous namespace


extern "C" uint32_t blanczos(const uint32_t * B, const uint64_t N, const uint32_t Nrow, const uint32_t Ncol, uint64_t * result)
{
    const uint32_t nthreads = std::max(1U, std::thread::hardware_concurrency());
    constexpr double cache = 512.0 * 1024.0; // Blocks capped at 512 KB.

    // if (Ncol < Nrow + 64U)
    // {
    //     std::cerr << "Ncol >= Nrow + 64 is expected." << std::endl;
    //     exit(17);
    // }

    const __blanczos_matrix::sparse matrix = __blanczos_matrix::sparse(B, N, Nrow, Ncol, cache);

    uint32_t Nsol = UINT32_MAX;
    while (Nsol == UINT32_MAX)
    {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<uint64_t> dis(0U, UINT64_MAX);

        vec64 X = vec64(matrix.Kcol(), 0U);
        for (uint64_t & elem : X)
            elem = dis(gen);

        Nsol = runner(matrix, X, result, nthreads);
        if (Nsol == UINT32_MAX)
            std::cout << "Need to restart Lanczos." << std::endl;
    }
    return Nsol;
}


