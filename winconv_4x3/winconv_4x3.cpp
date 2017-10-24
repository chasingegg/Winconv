#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <assert.h>

#include <immintrin.h>
#include <mkl.h>

#include <winconv.hpp>
#include <sys/time.h>
using namespace std;

static void get_tiles_4x3_16t(int x, int y, int nrows, const float *dataSrc,
                              float *dataDst, int *counter)
{
    const int coter = *counter;
    //cout << "get tiles " << coter << endl;
    __m512 bufA[36];
    __m512 bufB, bufC, bufD, bufE, bufF, bufG, bufH, bufI;
    __m512i idx0 = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16,
                                    14, 12, 10, 8,  6,  4,  2,  0);
    __m512i idx1 = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17,
                                    15, 13, 11, 9,  7,  5,  3,  1);

    /* 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
       16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
       32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
       48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 */

    /* 0  2  4  6  8  10 12 14 16 18 20 22 24 26 28 30
       1  3  5  7  9  11 13 15 17 19 21 23 25 27 29 31
       32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62
       33 35 37 39 41 43 45 47 49 51 53 55 57 59 61 63 */

    /* 0  4  8  12 16 20 24 28 32 36 40 44 48 52 56 60    1, 3 permute0
       1  5  9  13 17 21 25 29 33 37 41 45 49 53 57 61    2, 4 permute0
       2  6  10 14 18 22 26 30 34 38 42 46 50 54 58 62    1, 3 permute1
       3  7  11 15 19 23 27 31 35 39 43 47 51 55 59 63    2, 4 permute1 */

    /* 0, 1, 2, 3, 4, 5 */
    bufB = _mm512_load_ps(dataSrc + (x+0) * nrows + y);
    bufC = _mm512_load_ps(dataSrc + (x+0) * nrows + y + 16);
    bufD = _mm512_load_ps(dataSrc + (x+0) * nrows + y + 32);
    bufE = _mm512_load_ps(dataSrc + (x+0) * nrows + y + 48);
    bufF = _mm512_permutex2var_ps(bufB, idx0, bufC);
    bufG = _mm512_permutex2var_ps(bufB, idx1, bufC);
    bufH = _mm512_permutex2var_ps(bufD, idx0, bufE);
    bufI = _mm512_permutex2var_ps(bufD, idx1, bufE);
    bufA[0] = _mm512_permutex2var_ps(bufF, idx0, bufH);
    bufA[1] = _mm512_permutex2var_ps(bufG, idx0, bufI);
    bufA[2] = _mm512_permutex2var_ps(bufF, idx1, bufH);
    bufA[3] = _mm512_permutex2var_ps(bufG, idx1, bufI);

    bufB = _mm512_load_ps(dataSrc + (x+0) * nrows + y + 2);
    bufC = _mm512_load_ps(dataSrc + (x+0) * nrows + y + 18);
    bufD = _mm512_load_ps(dataSrc + (x+0) * nrows + y + 34);
    bufE = _mm512_load_ps(dataSrc + (x+0) * nrows + y + 50);
    bufF = _mm512_permutex2var_ps(bufB, idx0, bufC);
    bufG = _mm512_permutex2var_ps(bufB, idx1, bufC);
    bufH = _mm512_permutex2var_ps(bufD, idx0, bufE);
    bufI = _mm512_permutex2var_ps(bufD, idx1, bufE);
    bufA[4] = _mm512_permutex2var_ps(bufF, idx1, bufH);
    bufA[5] = _mm512_permutex2var_ps(bufG, idx1, bufI);
    /* 6, 7, 8, 9, 10, 11 */
    bufB = _mm512_load_ps(dataSrc + (x+1) * nrows + y);
    bufC = _mm512_load_ps(dataSrc + (x+1) * nrows + y + 16);
    bufD = _mm512_load_ps(dataSrc + (x+1) * nrows + y + 32);
    bufE = _mm512_load_ps(dataSrc + (x+1) * nrows + y + 48);
    bufF = _mm512_permutex2var_ps(bufB, idx0, bufC);
    bufG = _mm512_permutex2var_ps(bufB, idx1, bufC);
    bufH = _mm512_permutex2var_ps(bufD, idx0, bufE);
    bufI = _mm512_permutex2var_ps(bufD, idx1, bufE);
    bufA[6] = _mm512_permutex2var_ps(bufF, idx0, bufH);
    bufA[7] = _mm512_permutex2var_ps(bufG, idx0, bufI);
    bufA[8] = _mm512_permutex2var_ps(bufF, idx1, bufH);
    bufA[9] = _mm512_permutex2var_ps(bufG, idx1, bufI);

    bufB = _mm512_load_ps(dataSrc + (x+1) * nrows + y + 2);
    bufC = _mm512_load_ps(dataSrc + (x+1) * nrows + y + 18);
    bufD = _mm512_load_ps(dataSrc + (x+1) * nrows + y + 34);
    bufE = _mm512_load_ps(dataSrc + (x+1) * nrows + y + 50);
    bufF = _mm512_permutex2var_ps(bufB, idx0, bufC);
    bufG = _mm512_permutex2var_ps(bufB, idx1, bufC);
    bufH = _mm512_permutex2var_ps(bufD, idx0, bufE);
    bufI = _mm512_permutex2var_ps(bufD, idx1, bufE);
    bufA[10] = _mm512_permutex2var_ps(bufF, idx1, bufH);
    bufA[11] = _mm512_permutex2var_ps(bufG, idx1, bufI);
    /* 12, 13, 14, 15, 16, 17 */
    bufB = _mm512_load_ps(dataSrc + (x+2) * nrows + y);
    bufC = _mm512_load_ps(dataSrc + (x+2) * nrows + y + 16);
    bufD = _mm512_load_ps(dataSrc + (x+2) * nrows + y + 32);
    bufE = _mm512_load_ps(dataSrc + (x+2) * nrows + y + 48);
    bufF = _mm512_permutex2var_ps(bufB, idx0, bufC);
    bufG = _mm512_permutex2var_ps(bufB, idx1, bufC);
    bufH = _mm512_permutex2var_ps(bufD, idx0, bufE);
    bufI = _mm512_permutex2var_ps(bufD, idx1, bufE);
    bufA[12] = _mm512_permutex2var_ps(bufF, idx0, bufH);
    bufA[13] = _mm512_permutex2var_ps(bufG, idx0, bufI);
    bufA[14] = _mm512_permutex2var_ps(bufF, idx1, bufH);
    bufA[15] = _mm512_permutex2var_ps(bufG, idx1, bufI);

    bufB = _mm512_load_ps(dataSrc + (x+2) * nrows + y + 2);
    bufC = _mm512_load_ps(dataSrc + (x+2) * nrows + y + 18);
    bufD = _mm512_load_ps(dataSrc + (x+2) * nrows + y + 34);
    bufE = _mm512_load_ps(dataSrc + (x+2) * nrows + y + 50);
    bufF = _mm512_permutex2var_ps(bufB, idx0, bufC);
    bufG = _mm512_permutex2var_ps(bufB, idx1, bufC);
    bufH = _mm512_permutex2var_ps(bufD, idx0, bufE);
    bufI = _mm512_permutex2var_ps(bufD, idx1, bufE);
    bufA[16] = _mm512_permutex2var_ps(bufF, idx1, bufH);
    bufA[17] = _mm512_permutex2var_ps(bufG, idx1, bufI);
    /* 18, 19, 20, 21, 22, 23 */
    bufB = _mm512_load_ps(dataSrc + (x+3) * nrows + y);
    bufC = _mm512_load_ps(dataSrc + (x+3) * nrows + y + 16);
    bufD = _mm512_load_ps(dataSrc + (x+3) * nrows + y + 32);
    bufE = _mm512_load_ps(dataSrc + (x+3) * nrows + y + 48);
    bufF = _mm512_permutex2var_ps(bufB, idx0, bufC);
    bufG = _mm512_permutex2var_ps(bufB, idx1, bufC);
    bufH = _mm512_permutex2var_ps(bufD, idx0, bufE);
    bufI = _mm512_permutex2var_ps(bufD, idx1, bufE);
    bufA[18] = _mm512_permutex2var_ps(bufF, idx0, bufH);
    bufA[19] = _mm512_permutex2var_ps(bufG, idx0, bufI);
    bufA[20] = _mm512_permutex2var_ps(bufF, idx1, bufH);
    bufA[21] = _mm512_permutex2var_ps(bufG, idx1, bufI);

    bufB = _mm512_load_ps(dataSrc + (x+3) * nrows + y + 2);
    bufC = _mm512_load_ps(dataSrc + (x+3) * nrows + y + 18);
    bufD = _mm512_load_ps(dataSrc + (x+3) * nrows + y + 34);
    bufE = _mm512_load_ps(dataSrc + (x+3) * nrows + y + 50);
    bufF = _mm512_permutex2var_ps(bufB, idx0, bufC);
    bufG = _mm512_permutex2var_ps(bufB, idx1, bufC);
    bufH = _mm512_permutex2var_ps(bufD, idx0, bufE);
    bufI = _mm512_permutex2var_ps(bufD, idx1, bufE);
    bufA[22] = _mm512_permutex2var_ps(bufF, idx1, bufH);
    bufA[23] = _mm512_permutex2var_ps(bufG, idx1, bufI);
    /* 24, 25, 26, 27, 28, 29 */
    bufB = _mm512_load_ps(dataSrc + (x+4) * nrows + y);
    bufC = _mm512_load_ps(dataSrc + (x+4) * nrows + y + 16);
    bufD = _mm512_load_ps(dataSrc + (x+4) * nrows + y + 32);
    bufE = _mm512_load_ps(dataSrc + (x+4) * nrows + y + 48);
    bufF = _mm512_permutex2var_ps(bufB, idx0, bufC);
    bufG = _mm512_permutex2var_ps(bufB, idx1, bufC);
    bufH = _mm512_permutex2var_ps(bufD, idx0, bufE);
    bufI = _mm512_permutex2var_ps(bufD, idx1, bufE);
    bufA[24] = _mm512_permutex2var_ps(bufF, idx0, bufH);
    bufA[25] = _mm512_permutex2var_ps(bufG, idx0, bufI);
    bufA[26] = _mm512_permutex2var_ps(bufF, idx1, bufH);
    bufA[27] = _mm512_permutex2var_ps(bufG, idx1, bufI);

    bufB = _mm512_load_ps(dataSrc + (x+4) * nrows + y + 2);
    bufC = _mm512_load_ps(dataSrc + (x+4) * nrows + y + 18);
    bufD = _mm512_load_ps(dataSrc + (x+4) * nrows + y + 34);
    bufE = _mm512_load_ps(dataSrc + (x+4) * nrows + y + 50);
    bufF = _mm512_permutex2var_ps(bufB, idx0, bufC);
    bufG = _mm512_permutex2var_ps(bufB, idx1, bufC);
    bufH = _mm512_permutex2var_ps(bufD, idx0, bufE);
    bufI = _mm512_permutex2var_ps(bufD, idx1, bufE);
    bufA[28] = _mm512_permutex2var_ps(bufF, idx1, bufH);
    bufA[29] = _mm512_permutex2var_ps(bufG, idx1, bufI);
    /* 30, 31, 32, 33, 34, 35 */
    bufB = _mm512_load_ps(dataSrc + (x+5) * nrows + y);
    bufC = _mm512_load_ps(dataSrc + (x+5) * nrows + y + 16);
    bufD = _mm512_load_ps(dataSrc + (x+5) * nrows + y + 32);
    bufE = _mm512_load_ps(dataSrc + (x+5) * nrows + y + 48);
    bufF = _mm512_permutex2var_ps(bufB, idx0, bufC);
    bufG = _mm512_permutex2var_ps(bufB, idx1, bufC);
    bufH = _mm512_permutex2var_ps(bufD, idx0, bufE);
    bufI = _mm512_permutex2var_ps(bufD, idx1, bufE);
    bufA[30] = _mm512_permutex2var_ps(bufF, idx0, bufH);
    bufA[31] = _mm512_permutex2var_ps(bufG, idx0, bufI);
    bufA[32] = _mm512_permutex2var_ps(bufF, idx1, bufH);
    bufA[33] = _mm512_permutex2var_ps(bufG, idx1, bufI);

    bufB = _mm512_load_ps(dataSrc + (x+5) * nrows + y + 2);
    bufC = _mm512_load_ps(dataSrc + (x+5) * nrows + y + 18);
    bufD = _mm512_load_ps(dataSrc + (x+5) * nrows + y + 34);
    bufE = _mm512_load_ps(dataSrc + (x+5) * nrows + y + 50);
    bufF = _mm512_permutex2var_ps(bufB, idx0, bufC);
    bufG = _mm512_permutex2var_ps(bufB, idx1, bufC);
    bufH = _mm512_permutex2var_ps(bufD, idx0, bufE);
    bufI = _mm512_permutex2var_ps(bufD, idx1, bufE);
    bufA[34] = _mm512_permutex2var_ps(bufF, idx1, bufH);
    bufA[35] = _mm512_permutex2var_ps(bufG, idx1, bufI);


    __m512 bufTemp[36];
    __m512 m0 = _mm512_setzero_ps();
    __m512 m1 = _mm512_set1_ps(1.0f);
    __m512 m2 = _mm512_set1_ps(2.0f);
    __m512 m4 = _mm512_set1_ps(4.0f);
    __m512 m5 = _mm512_set1_ps(5.0f);

    bufTemp[0] = _mm512_mul_ps(m4, bufA[0]);
    bufTemp[1] = _mm512_mul_ps(m4, bufA[1]);
    bufTemp[2] = _mm512_mul_ps(m4, bufA[2]);
    bufTemp[3] = _mm512_mul_ps(m4, bufA[3]);
    bufTemp[4] = _mm512_mul_ps(m4, bufA[4]);
    bufTemp[5] = _mm512_mul_ps(m4, bufA[5]);
    bufTemp[0] = _mm512_fnmadd_ps(m5, bufA[12], bufTemp[0]);
    bufTemp[1] = _mm512_fnmadd_ps(m5, bufA[13], bufTemp[1]);
    bufTemp[2] = _mm512_fnmadd_ps(m5, bufA[14], bufTemp[2]);
    bufTemp[3] = _mm512_fnmadd_ps(m5, bufA[15], bufTemp[3]);
    bufTemp[4] = _mm512_fnmadd_ps(m5, bufA[16], bufTemp[4]);
    bufTemp[5] = _mm512_fnmadd_ps(m5, bufA[17], bufTemp[5]);
    bufTemp[0] = _mm512_add_ps(bufA[24], bufTemp[0]);
    bufTemp[1] = _mm512_add_ps(bufA[25], bufTemp[1]);
    bufTemp[2] = _mm512_add_ps(bufA[26], bufTemp[2]);
    bufTemp[3] = _mm512_add_ps(bufA[27], bufTemp[3]);
    bufTemp[4] = _mm512_add_ps(bufA[28], bufTemp[4]);
    bufTemp[5] = _mm512_add_ps(bufA[29], bufTemp[5]);

    bufTemp[6] = _mm512_fnmadd_ps(m4, bufA[6], m0);
    bufTemp[7] = _mm512_fnmadd_ps(m4, bufA[7], m0);
    bufTemp[8] = _mm512_fnmadd_ps(m4, bufA[8], m0);
    bufTemp[9] = _mm512_fnmadd_ps(m4, bufA[9], m0);
    bufTemp[10] = _mm512_fnmadd_ps(m4, bufA[10], m0);
    bufTemp[11] = _mm512_fnmadd_ps(m4, bufA[11], m0);
    bufTemp[6] = _mm512_fnmadd_ps(m4, bufA[12], bufTemp[6]);
    bufTemp[7] = _mm512_fnmadd_ps(m4, bufA[13], bufTemp[7]);
    bufTemp[8] = _mm512_fnmadd_ps(m4, bufA[14], bufTemp[8]);
    bufTemp[9] = _mm512_fnmadd_ps(m4, bufA[15], bufTemp[9]);
    bufTemp[10] = _mm512_fnmadd_ps(m4, bufA[16], bufTemp[10]);
    bufTemp[11] = _mm512_fnmadd_ps(m4, bufA[17], bufTemp[11]);
    bufTemp[6] = _mm512_add_ps(bufA[18], bufTemp[6]);
    bufTemp[7] = _mm512_add_ps(bufA[19], bufTemp[7]);
    bufTemp[8] = _mm512_add_ps(bufA[20], bufTemp[8]);
    bufTemp[9] = _mm512_add_ps(bufA[21], bufTemp[9]);
    bufTemp[10] = _mm512_add_ps(bufA[22], bufTemp[10]);
    bufTemp[11] = _mm512_add_ps(bufA[23], bufTemp[11]);
    bufTemp[6] = _mm512_add_ps(bufA[24], bufTemp[6]);
    bufTemp[7] = _mm512_add_ps(bufA[25], bufTemp[7]);
    bufTemp[8] = _mm512_add_ps(bufA[26], bufTemp[8]);
    bufTemp[9] = _mm512_add_ps(bufA[27], bufTemp[9]);
    bufTemp[10] = _mm512_add_ps(bufA[28], bufTemp[10]);
    bufTemp[11] = _mm512_add_ps(bufA[29], bufTemp[11]);

    bufTemp[12] = _mm512_mul_ps(m4, bufA[6]);
    bufTemp[13] = _mm512_mul_ps(m4, bufA[7]);
    bufTemp[14] = _mm512_mul_ps(m4, bufA[8]);
    bufTemp[15] = _mm512_mul_ps(m4, bufA[9]);
    bufTemp[16] = _mm512_mul_ps(m4, bufA[10]);
    bufTemp[17] = _mm512_mul_ps(m4, bufA[11]);
    bufTemp[12] = _mm512_fnmadd_ps(m4, bufA[12], bufTemp[12]);
    bufTemp[13] = _mm512_fnmadd_ps(m4, bufA[13], bufTemp[13]);
    bufTemp[14] = _mm512_fnmadd_ps(m4, bufA[14], bufTemp[14]);
    bufTemp[15] = _mm512_fnmadd_ps(m4, bufA[15], bufTemp[15]);
    bufTemp[16] = _mm512_fnmadd_ps(m4, bufA[16], bufTemp[16]);
    bufTemp[17] = _mm512_fnmadd_ps(m4, bufA[17], bufTemp[17]);
    bufTemp[12] = _mm512_sub_ps(bufTemp[12], bufA[18]);
    bufTemp[13] = _mm512_sub_ps(bufTemp[13], bufA[19]);
    bufTemp[14] = _mm512_sub_ps(bufTemp[14], bufA[20]);
    bufTemp[15] = _mm512_sub_ps(bufTemp[15], bufA[21]);
    bufTemp[16] = _mm512_sub_ps(bufTemp[16], bufA[22]);
    bufTemp[17] = _mm512_sub_ps(bufTemp[17], bufA[23]);
    bufTemp[12] = _mm512_add_ps(bufTemp[12], bufA[24]);
    bufTemp[13] = _mm512_add_ps(bufTemp[13], bufA[25]);
    bufTemp[14] = _mm512_add_ps(bufTemp[14], bufA[26]);
    bufTemp[15] = _mm512_add_ps(bufTemp[15], bufA[27]);
    bufTemp[16] = _mm512_add_ps(bufTemp[16], bufA[28]);
    bufTemp[17] = _mm512_add_ps(bufTemp[17], bufA[29]);

    bufTemp[18] = _mm512_fnmadd_ps(m2, bufA[6], m0);
    bufTemp[19] = _mm512_fnmadd_ps(m2, bufA[7], m0);
    bufTemp[20] = _mm512_fnmadd_ps(m2, bufA[8], m0);
    bufTemp[21] = _mm512_fnmadd_ps(m2, bufA[9], m0);
    bufTemp[22] = _mm512_fnmadd_ps(m2, bufA[10], m0);
    bufTemp[23] = _mm512_fnmadd_ps(m2, bufA[11], m0);
    bufTemp[18] = _mm512_sub_ps(bufTemp[18], bufA[12]);
    bufTemp[19] = _mm512_sub_ps(bufTemp[19], bufA[13]);
    bufTemp[20] = _mm512_sub_ps(bufTemp[20], bufA[14]);
    bufTemp[21] = _mm512_sub_ps(bufTemp[21], bufA[15]);
    bufTemp[22] = _mm512_sub_ps(bufTemp[22], bufA[16]);
    bufTemp[23] = _mm512_sub_ps(bufTemp[23], bufA[17]);
    bufTemp[18] = _mm512_fmadd_ps(m2, bufA[18], bufTemp[18]);
    bufTemp[19] = _mm512_fmadd_ps(m2, bufA[19], bufTemp[19]);
    bufTemp[20] = _mm512_fmadd_ps(m2, bufA[20], bufTemp[20]);
    bufTemp[21] = _mm512_fmadd_ps(m2, bufA[21], bufTemp[21]);
    bufTemp[22] = _mm512_fmadd_ps(m2, bufA[22], bufTemp[22]);
    bufTemp[23] = _mm512_fmadd_ps(m2, bufA[23], bufTemp[23]);
    bufTemp[18] = _mm512_add_ps(bufTemp[18], bufA[24]);
    bufTemp[19] = _mm512_add_ps(bufTemp[19], bufA[25]);
    bufTemp[20] = _mm512_add_ps(bufTemp[20], bufA[26]);
    bufTemp[21] = _mm512_add_ps(bufTemp[21], bufA[27]);
    bufTemp[22] = _mm512_add_ps(bufTemp[22], bufA[28]);
    bufTemp[23] = _mm512_add_ps(bufTemp[23], bufA[29]);

    bufTemp[24] = _mm512_mul_ps(m2, bufA[6]);
    bufTemp[25] = _mm512_mul_ps(m2, bufA[7]);
    bufTemp[26] = _mm512_mul_ps(m2, bufA[8]);
    bufTemp[27] = _mm512_mul_ps(m2, bufA[9]);
    bufTemp[28] = _mm512_mul_ps(m2, bufA[10]);
    bufTemp[29] = _mm512_mul_ps(m2, bufA[11]);
    bufTemp[24] = _mm512_sub_ps(bufTemp[24], bufA[12]);
    bufTemp[25] = _mm512_sub_ps(bufTemp[25], bufA[13]);
    bufTemp[26] = _mm512_sub_ps(bufTemp[26], bufA[14]);
    bufTemp[27] = _mm512_sub_ps(bufTemp[27], bufA[15]);
    bufTemp[28] = _mm512_sub_ps(bufTemp[28], bufA[16]);
    bufTemp[29] = _mm512_sub_ps(bufTemp[29], bufA[17]);
    bufTemp[24] = _mm512_fnmadd_ps(m2, bufA[18], bufTemp[24]);
    bufTemp[25] = _mm512_fnmadd_ps(m2, bufA[19], bufTemp[25]);
    bufTemp[26] = _mm512_fnmadd_ps(m2, bufA[20], bufTemp[26]);
    bufTemp[27] = _mm512_fnmadd_ps(m2, bufA[21], bufTemp[27]);
    bufTemp[28] = _mm512_fnmadd_ps(m2, bufA[22], bufTemp[28]);
    bufTemp[29] = _mm512_fnmadd_ps(m2, bufA[23], bufTemp[29]);
    bufTemp[24] = _mm512_add_ps(bufTemp[24], bufA[24]);
    bufTemp[25] = _mm512_add_ps(bufTemp[25], bufA[25]);
    bufTemp[26] = _mm512_add_ps(bufTemp[26], bufA[26]);
    bufTemp[27] = _mm512_add_ps(bufTemp[27], bufA[27]);
    bufTemp[28] = _mm512_add_ps(bufTemp[28], bufA[28]);
    bufTemp[29] = _mm512_add_ps(bufTemp[29], bufA[29]);

    bufTemp[30] = _mm512_mul_ps(m4, bufA[6]);
    bufTemp[31] = _mm512_mul_ps(m4, bufA[7]);
    bufTemp[32] = _mm512_mul_ps(m4, bufA[8]);
    bufTemp[33] = _mm512_mul_ps(m4, bufA[9]);
    bufTemp[34] = _mm512_mul_ps(m4, bufA[10]);
    bufTemp[35] = _mm512_mul_ps(m4, bufA[11]);
    bufTemp[30] = _mm512_fnmadd_ps(m5, bufA[18], bufTemp[30]);
    bufTemp[31] = _mm512_fnmadd_ps(m5, bufA[19], bufTemp[31]);
    bufTemp[32] = _mm512_fnmadd_ps(m5, bufA[20], bufTemp[32]);
    bufTemp[33] = _mm512_fnmadd_ps(m5, bufA[21], bufTemp[33]);
    bufTemp[34] = _mm512_fnmadd_ps(m5, bufA[22], bufTemp[34]);
    bufTemp[35] = _mm512_fnmadd_ps(m5, bufA[23], bufTemp[35]);
    bufTemp[30] = _mm512_add_ps(bufTemp[30], bufA[30]);
    bufTemp[31] = _mm512_add_ps(bufTemp[31], bufA[31]);
    bufTemp[32] = _mm512_add_ps(bufTemp[32], bufA[32]);
    bufTemp[33] = _mm512_add_ps(bufTemp[33], bufA[33]);
    bufTemp[34] = _mm512_add_ps(bufTemp[34], bufA[34]);
    bufTemp[35] = _mm512_add_ps(bufTemp[35], bufA[35]);

    /* 4  0  0  0  0  0
       0 -4  4 -2  2  4
      -5 -4 -4 -1 -1  0
       0  1 -1  2 -2 -5
       1  1  1  1  1  0
       0  0  0  0  0  1 */

    bufB = _mm512_mul_ps(bufTemp[0], m4);
    bufB = _mm512_fnmadd_ps(m5, bufTemp[2], bufB);
    bufB = _mm512_add_ps(bufB, bufTemp[4]);
    _mm512_store_ps(dataDst + 0 * ISTRIDE + coter, bufB);

    bufC = _mm512_fnmadd_ps(m4, bufTemp[1], m0);
    bufC = _mm512_fnmadd_ps(m4, bufTemp[2], bufC);
    bufC = _mm512_add_ps(bufTemp[3], bufC);
    bufC = _mm512_add_ps(bufTemp[4], bufC);
    _mm512_store_ps(dataDst + 1 * ISTRIDE + coter, bufC);

    bufD = _mm512_mul_ps(m4, bufTemp[1]);
    bufD = _mm512_fnmadd_ps(m4, bufTemp[2], bufD);
    bufD = _mm512_sub_ps(bufD, bufTemp[3]);
    bufD = _mm512_add_ps(bufD, bufTemp[4]);
    _mm512_store_ps(dataDst + 2 * ISTRIDE + coter, bufD);

    bufE = _mm512_fnmadd_ps(m2, bufTemp[1], m0);
    bufE = _mm512_sub_ps(bufE, bufTemp[2]);
    bufE = _mm512_fmadd_ps(m2, bufTemp[3], bufE);
    bufE = _mm512_add_ps(bufE, bufTemp[4]);
    _mm512_store_ps(dataDst + 3 * ISTRIDE + coter, bufE);

    bufF = _mm512_mul_ps(m2, bufTemp[1]);
    bufF = _mm512_sub_ps(bufF, bufTemp[2]);
    bufF = _mm512_fnmadd_ps(m2, bufTemp[3], bufF);
    bufF = _mm512_add_ps(bufF, bufTemp[4]);
    _mm512_store_ps(dataDst + 4 * ISTRIDE + coter, bufF);

    bufG = _mm512_mul_ps(m4, bufTemp[1]);
    bufG = _mm512_fnmadd_ps(m5, bufTemp[3], bufG);
    bufG = _mm512_add_ps(bufG, bufTemp[5]);
    _mm512_store_ps(dataDst + 5 * ISTRIDE + coter, bufG);

    // -------------------------------------------------------
    bufB = _mm512_mul_ps(bufTemp[6], m4);
    bufB = _mm512_fnmadd_ps(m5, bufTemp[8], bufB);
    bufB = _mm512_add_ps(bufB, bufTemp[10]);
    _mm512_store_ps(dataDst + 6 * ISTRIDE + coter, bufB);

    bufC = _mm512_fnmadd_ps(m4, bufTemp[7], m0);
    bufC = _mm512_fnmadd_ps(m4, bufTemp[8], bufC);
    bufC = _mm512_add_ps(bufTemp[9], bufC);
    bufC = _mm512_add_ps(bufTemp[10], bufC);
    _mm512_store_ps(dataDst + 7 * ISTRIDE + coter, bufC);

    bufD = _mm512_mul_ps(m4, bufTemp[7]);
    bufD = _mm512_fnmadd_ps(m4, bufTemp[8], bufD);
    bufD = _mm512_sub_ps(bufD, bufTemp[9]);
    bufD = _mm512_add_ps(bufD, bufTemp[10]);
    _mm512_store_ps(dataDst + 8 * ISTRIDE + coter, bufD);

    bufE = _mm512_fnmadd_ps(m2, bufTemp[7], m0);
    bufE = _mm512_sub_ps(bufE, bufTemp[8]);
    bufE = _mm512_fmadd_ps(m2, bufTemp[9], bufE);
    bufE = _mm512_add_ps(bufE, bufTemp[10]);
    _mm512_store_ps(dataDst + 9 * ISTRIDE + coter, bufE);

    bufF = _mm512_mul_ps(m2, bufTemp[7]);
    bufF = _mm512_sub_ps(bufF, bufTemp[8]);
    bufF = _mm512_fnmadd_ps(m2, bufTemp[9], bufF);
    bufF = _mm512_add_ps(bufF, bufTemp[10]);
    _mm512_store_ps(dataDst + 10 * ISTRIDE + coter, bufF);

    bufG = _mm512_mul_ps(m4, bufTemp[7]);
    bufG = _mm512_fnmadd_ps(m5, bufTemp[9], bufG);
    bufG = _mm512_add_ps(bufG, bufTemp[11]);
    _mm512_store_ps(dataDst + 11 * ISTRIDE + coter, bufG);

    // ------------------------------------
    bufB = _mm512_mul_ps(bufTemp[12], m4);
    bufB = _mm512_fnmadd_ps(m5, bufTemp[14], bufB);
    bufB = _mm512_add_ps(bufB, bufTemp[16]);
    _mm512_store_ps(dataDst + 12 * ISTRIDE + coter, bufB);

    bufC = _mm512_fnmadd_ps(m4, bufTemp[13], m0);
    bufC = _mm512_fnmadd_ps(m4, bufTemp[14], bufC);
    bufC = _mm512_add_ps(bufTemp[15], bufC);
    bufC = _mm512_add_ps(bufTemp[16], bufC);
    _mm512_store_ps(dataDst + 13 * ISTRIDE + coter, bufC);

    bufD = _mm512_mul_ps(m4, bufTemp[13]);
    bufD = _mm512_fnmadd_ps(m4, bufTemp[14], bufD);
    bufD = _mm512_sub_ps(bufD, bufTemp[15]);
    bufD = _mm512_add_ps(bufD, bufTemp[16]);
    _mm512_store_ps(dataDst + 14 * ISTRIDE + coter, bufD);

    bufE = _mm512_fnmadd_ps(m2, bufTemp[13], m0);
    bufE = _mm512_sub_ps(bufE, bufTemp[14]);
    bufE = _mm512_fmadd_ps(m2, bufTemp[15], bufE);
    bufE = _mm512_add_ps(bufE, bufTemp[16]);
    _mm512_store_ps(dataDst + 15 * ISTRIDE + coter, bufE);

    bufF = _mm512_mul_ps(m2, bufTemp[13]);
    bufF = _mm512_sub_ps(bufF, bufTemp[14]);
    bufF = _mm512_fnmadd_ps(m2, bufTemp[15], bufF);
    bufF = _mm512_add_ps(bufF, bufTemp[16]);
    _mm512_store_ps(dataDst + 16 * ISTRIDE + coter, bufF);

    bufG = _mm512_mul_ps(m4, bufTemp[13]);
    bufG = _mm512_fnmadd_ps(m5, bufTemp[15], bufG);
    bufG = _mm512_add_ps(bufG, bufTemp[17]);
    _mm512_store_ps(dataDst + 17 * ISTRIDE + coter, bufG);

    // --------------------------------------------
    bufB = _mm512_mul_ps(bufTemp[18], m4);
    bufB = _mm512_fnmadd_ps(m5, bufTemp[20], bufB);
    bufB = _mm512_add_ps(bufB, bufTemp[22]);
    _mm512_store_ps(dataDst + 18 * ISTRIDE + coter, bufB);

    bufC = _mm512_fnmadd_ps(m4, bufTemp[19], m0);
    bufC = _mm512_fnmadd_ps(m4, bufTemp[20], bufC);
    bufC = _mm512_add_ps(bufTemp[21], bufC);
    bufC = _mm512_add_ps(bufTemp[22], bufC);
    _mm512_store_ps(dataDst + 19 * ISTRIDE + coter, bufC);

    bufD = _mm512_mul_ps(m4, bufTemp[19]);
    bufD = _mm512_fnmadd_ps(m4, bufTemp[20], bufD);
    bufD = _mm512_sub_ps(bufD, bufTemp[21]);
    bufD = _mm512_add_ps(bufD, bufTemp[22]);
    _mm512_store_ps(dataDst + 20 * ISTRIDE + coter, bufD);

    bufE = _mm512_fnmadd_ps(m2, bufTemp[19], m0);
    bufE = _mm512_sub_ps(bufE, bufTemp[20]);
    bufE = _mm512_fmadd_ps(m2, bufTemp[21], bufE);
    bufE = _mm512_add_ps(bufE, bufTemp[22]);
    _mm512_store_ps(dataDst + 21 * ISTRIDE + coter, bufE);

    bufF = _mm512_mul_ps(m2, bufTemp[19]);
    bufF = _mm512_sub_ps(bufF, bufTemp[20]);
    bufF = _mm512_fnmadd_ps(m2, bufTemp[21], bufF);
    bufF = _mm512_add_ps(bufF, bufTemp[22]);
    _mm512_store_ps(dataDst + 22 * ISTRIDE + coter, bufF);

    bufG = _mm512_mul_ps(m4, bufTemp[19]);
    bufG = _mm512_fnmadd_ps(m5, bufTemp[21], bufG);
    bufG = _mm512_add_ps(bufG, bufTemp[23]);
    _mm512_store_ps(dataDst + 23 * ISTRIDE + coter, bufG);

    // --------------------------------------------
    bufB = _mm512_mul_ps(bufTemp[24], m4);
    bufB = _mm512_fnmadd_ps(m5, bufTemp[26], bufB);
    bufB = _mm512_add_ps(bufB, bufTemp[28]);
    _mm512_store_ps(dataDst + 24 * ISTRIDE + coter, bufB);

    bufC = _mm512_fnmadd_ps(m4, bufTemp[25], m0);
    bufC = _mm512_fnmadd_ps(m4, bufTemp[26], bufC);
    bufC = _mm512_add_ps(bufTemp[27], bufC);
    bufC = _mm512_add_ps(bufTemp[28], bufC);
    _mm512_store_ps(dataDst + 25 * ISTRIDE + coter, bufC);

    bufD = _mm512_mul_ps(m4, bufTemp[25]);
    bufD = _mm512_fnmadd_ps(m4, bufTemp[26], bufD);
    bufD = _mm512_sub_ps(bufD, bufTemp[27]);
    bufD = _mm512_add_ps(bufD, bufTemp[28]);
    _mm512_store_ps(dataDst + 26 * ISTRIDE + coter, bufD);

    bufE = _mm512_fnmadd_ps(m2, bufTemp[25], m0);
    bufE = _mm512_sub_ps(bufE, bufTemp[26]);
    bufE = _mm512_fmadd_ps(m2, bufTemp[27], bufE);
    bufE = _mm512_add_ps(bufE, bufTemp[28]);
    _mm512_store_ps(dataDst + 27 * ISTRIDE + coter, bufE);

    bufF = _mm512_mul_ps(m2, bufTemp[25]);
    bufF = _mm512_sub_ps(bufF, bufTemp[26]);
    bufF = _mm512_fnmadd_ps(m2, bufTemp[27], bufF);
    bufF = _mm512_add_ps(bufF, bufTemp[28]);
    _mm512_store_ps(dataDst + 28 * ISTRIDE + coter, bufF);

    bufG = _mm512_mul_ps(m4, bufTemp[25]);
    bufG = _mm512_fnmadd_ps(m5, bufTemp[27], bufG);
    bufG = _mm512_add_ps(bufG, bufTemp[29]);
    _mm512_store_ps(dataDst + 29 * ISTRIDE + coter, bufG);

    // ----------------------------------------
    bufB = _mm512_mul_ps(bufTemp[30], m4);
    bufB = _mm512_fnmadd_ps(m5, bufTemp[32], bufB);
    bufB = _mm512_add_ps(bufB, bufTemp[34]);
    _mm512_store_ps(dataDst + 30 * ISTRIDE + coter, bufB);

    bufC = _mm512_fnmadd_ps(m4, bufTemp[31], m0);
    bufC = _mm512_fnmadd_ps(m4, bufTemp[32], bufC);
    bufC = _mm512_add_ps(bufTemp[33], bufC);
    bufC = _mm512_add_ps(bufTemp[34], bufC);
    _mm512_store_ps(dataDst + 31 * ISTRIDE + coter, bufC);

    bufD = _mm512_mul_ps(m4, bufTemp[31]);
    bufD = _mm512_fnmadd_ps(m4, bufTemp[32], bufD);
    bufD = _mm512_sub_ps(bufD, bufTemp[33]);
    bufD = _mm512_add_ps(bufD, bufTemp[34]);
    _mm512_store_ps(dataDst + 32 * ISTRIDE + coter, bufD);

    bufE = _mm512_fnmadd_ps(m2, bufTemp[31], m0);
    bufE = _mm512_sub_ps(bufE, bufTemp[32]);
    bufE = _mm512_fmadd_ps(m2, bufTemp[33], bufE);
    bufE = _mm512_add_ps(bufE, bufTemp[34]);
    _mm512_store_ps(dataDst + 33 * ISTRIDE + coter, bufE);

    bufF = _mm512_mul_ps(m2, bufTemp[31]);
    bufF = _mm512_sub_ps(bufF, bufTemp[32]);
    bufF = _mm512_fnmadd_ps(m2, bufTemp[33], bufF);
    bufF = _mm512_add_ps(bufF, bufTemp[34]);
    _mm512_store_ps(dataDst + 34 * ISTRIDE + coter, bufF);

    bufG = _mm512_mul_ps(m4, bufTemp[31]);
    bufG = _mm512_fnmadd_ps(m5, bufTemp[33], bufG);
    bufG = _mm512_add_ps(bufG, bufTemp[35]);
    _mm512_store_ps(dataDst + 35 * ISTRIDE + coter, bufG);

    *counter += 16;
}

static inline void pad_get_tiles(int x, int y, int lenX, int lenY, int nrows, const float *dataSrc,
                                 float *temp, float *dataDst, int *counter) {
    if (2 == lenX || 2 == lenY) return;
    int i, j;
    for (i = 0; i < lenX; ++i) {
        for (j = 0; j < lenY; ++j) {
            temp[i * 66 + j] = dataSrc[(x + i) * nrows + y + j];
        }
        for (; j < 66; ++j) {
            temp[i * 66 + j] = 0;
        }
        //memset(temp + i * 66 + j, 1, (66 - j) * sizeof(float));
    }
    /*if (i < 6) {
        memset(temp + i * 66, 0, 66 * (6 - i) * sizeof(float));
    }*/
    for (; i < 6; ++i) {
        for (j = 0; j < 66; ++j) {
            temp[i * 66 + j] = 0;
        }
    }

    get_tiles_4x3_16t(0, 0, 66, temp, dataDst, counter);
}

static inline void get_tiles_4x3_1t(int x, int y, int nrows, const float *dataSrc,
                                    float *dataDst, int *counter) {
    int coter = *counter;
    float temp[36] __attribute__((aligned(64)));

    temp[0] = dataSrc[(x + 0) * nrows + y + 0];
    temp[1] = dataSrc[(x + 0) * nrows + y + 1];
    temp[2] = dataSrc[(x + 0) * nrows + y + 2];
    temp[3] = dataSrc[(x + 0) * nrows + y + 3];
    temp[4] = dataSrc[(x + 0) * nrows + y + 4];
    temp[5] = dataSrc[(x + 0) * nrows + y + 5];
    temp[6] = dataSrc[(x + 1) * nrows + y + 0];
    temp[7] = dataSrc[(x + 1) * nrows + y + 1];
    temp[8] = dataSrc[(x + 1) * nrows + y + 2];
    temp[9] = dataSrc[(x + 1) * nrows + y + 3];
    temp[10] = dataSrc[(x + 1) * nrows + y + 4];
    temp[11] = dataSrc[(x + 1) * nrows + y + 5];
    temp[12] = dataSrc[(x + 2) * nrows + y + 0];
    temp[13] = dataSrc[(x + 2) * nrows + y + 1];
    temp[14] = dataSrc[(x + 2) * nrows + y + 2];
    temp[15] = dataSrc[(x + 2) * nrows + y + 3];
    temp[16] = dataSrc[(x + 2) * nrows + y + 4];
    temp[17] = dataSrc[(x + 2) * nrows + y + 5];
    temp[18] = dataSrc[(x + 3) * nrows + y + 0];
    temp[19] = dataSrc[(x + 3) * nrows + y + 1];
    temp[20] = dataSrc[(x + 3) * nrows + y + 2];
    temp[21] = dataSrc[(x + 3) * nrows + y + 3];
    temp[22] = dataSrc[(x + 3) * nrows + y + 4];
    temp[23] = dataSrc[(x + 3) * nrows + y + 5];
    temp[24] = dataSrc[(x + 4) * nrows + y + 0];
    temp[25] = dataSrc[(x + 4) * nrows + y + 1];
    temp[26] = dataSrc[(x + 4) * nrows + y + 2];
    temp[27] = dataSrc[(x + 4) * nrows + y + 3];
    temp[28] = dataSrc[(x + 4) * nrows + y + 4];
    temp[29] = dataSrc[(x + 4) * nrows + y + 5];
    temp[30] = dataSrc[(x + 5) * nrows + y + 0];
    temp[31] = dataSrc[(x + 5) * nrows + y + 1];
    temp[32] = dataSrc[(x + 5) * nrows + y + 2];
    temp[33] = dataSrc[(x + 5) * nrows + y + 3];
    temp[34] = dataSrc[(x + 5) * nrows + y + 4];
    temp[35] = dataSrc[(x + 5) * nrows + y + 5];

    float temp2[36]__attribute__((aligned(64)));
    temp2[0] = 4 * temp[0] - 5 * temp[12] + temp[24];
    temp2[1] = 4 * temp[1] - 5 * temp[13] + temp[25];
    temp2[2] = 4 * temp[2] - 5 * temp[14] + temp[26];
    temp2[3] = 4 * temp[3] - 5 * temp[15] + temp[27];
    temp2[4] = 4 * temp[4] - 5 * temp[16] + temp[28];
    temp2[5] = 4 * temp[5] - 5 * temp[17] + temp[29];
    temp2[6] = -4 * temp[6] - 4 * temp[12] + temp[18] + temp[24];
    temp2[7] = -4 * temp[7] - 4 * temp[13] + temp[19] + temp[25];
    temp2[8] = -4 * temp[8] - 4 * temp[14] + temp[20] + temp[26];
    temp2[9] = -4 * temp[9] - 4 * temp[15] + temp[21] + temp[27];
    temp2[10] = -4 * temp[10] - 4 * temp[16] + temp[22] + temp[28];
    temp2[11] = -4 * temp[11] - 4 * temp[17] + temp[23] + temp[29];
    temp2[12] = 4 * temp[6] - 4 * temp[12] - temp[18] + temp[24];
    temp2[13] = 4 * temp[7] - 4 * temp[13] - temp[19] + temp[25];
    temp2[14] = 4 * temp[8] - 4 * temp[14] - temp[20] + temp[26];
    temp2[15] = 4 * temp[9] - 4 * temp[15] - temp[21] + temp[27];
    temp2[16] = 4 * temp[10] - 4 * temp[16] - temp[22] + temp[28];
    temp2[17] = 4 * temp[11] - 4 * temp[17] - temp[23] + temp[29];
    temp2[18] = -2 * temp[6] - temp[12] + 2 * temp[18] + temp[24];
    temp2[19] = -2 * temp[7] - temp[13] + 2 * temp[19] + temp[25];
    temp2[20] = -2 * temp[8] - temp[14] + 2 * temp[20] + temp[26];
    temp2[21] = -2 * temp[9] - temp[15] + 2 * temp[21] + temp[27];
    temp2[22] = -2 * temp[10] - temp[16] + 2 * temp[22] + temp[28];
    temp2[23] = -2 * temp[11] - temp[17] + 2 * temp[23] + temp[29];
    temp2[24] = 2 * temp[6] - temp[12] - 2 * temp[18] + temp[24];
    temp2[25] = 2 * temp[7] - temp[13] - 2 * temp[19] + temp[25];
    temp2[26] = 2 * temp[8] - temp[14] - 2 * temp[20] + temp[26];
    temp2[27] = 2 * temp[9] - temp[15] - 2 * temp[21] + temp[27];
    temp2[28] = 2 * temp[10] - temp[16] - 2 * temp[22] + temp[28];
    temp2[29] = 2 * temp[11] - temp[17] - 2 * temp[23] + temp[29];
    temp2[30] = 4 * temp[6] - 5 * temp[18] + temp[30];
    temp2[31] = 4 * temp[7] - 5 * temp[19] + temp[31];
    temp2[32] = 4 * temp[8] - 5 * temp[20] + temp[32];
    temp2[33] = 4 * temp[9] - 5 * temp[21] + temp[33];
    temp2[34] = 4 * temp[10] - 5 * temp[22] + temp[34];
    temp2[35] = 4 * temp[11] - 5 * temp[23] + temp[35];

    dataDst[0 * ISTRIDE + coter] = temp2[0] * 4 - temp2[2] * 5 + temp2[4];
    dataDst[1 * ISTRIDE + coter] = -temp2[1] * 4 - temp2[2] * 4 + temp2[3] + temp2[4];
    dataDst[2 * ISTRIDE + coter] = temp2[1] * 4 - temp2[2] * 4 - temp2[3] + temp2[4];
    dataDst[3 * ISTRIDE + coter] = -temp2[1] * 2 - temp2[2] + temp2[3] * 2 + temp2[4];
    dataDst[4 * ISTRIDE + coter] = temp2[1] * 2 - temp2[2] - temp2[3] * 2 + temp2[4];
    dataDst[5 * ISTRIDE + coter] = temp2[1] * 4 - temp2[3] * 5 + temp2[5];
    dataDst[6 * ISTRIDE + coter] = temp2[6] * 4 - temp2[8] * 5 + temp2[10];
    dataDst[7 * ISTRIDE + coter] = -temp2[7] * 4 - temp2[8] * 4 + temp2[9] + temp2[10];
    dataDst[8 * ISTRIDE + coter] = temp2[7] * 4 - temp2[8] * 4 - temp2[9] + temp2[10];
    dataDst[9 * ISTRIDE + coter] = -temp2[7] * 2 - temp2[8] + temp2[9] * 2 + temp2[10];
    dataDst[10 * ISTRIDE + coter] = temp2[7] * 2 - temp2[8] - temp2[9] * 2 + temp2[10];
    dataDst[11 * ISTRIDE + coter] = temp2[7] * 4 - temp2[9] * 5 + temp2[11];
    dataDst[12 * ISTRIDE + coter] = temp2[12] * 4 - temp2[14] * 5 + temp2[16];
    dataDst[13 * ISTRIDE + coter] = -temp2[13] * 4 - temp2[14] * 4 + temp2[15] + temp2[16];
    dataDst[14 * ISTRIDE + coter] = temp2[13] * 4 - temp2[14] * 4 - temp2[15] + temp2[16];
    dataDst[15 * ISTRIDE + coter] = -temp2[13] * 2 - temp2[14] + temp2[15] * 2 + temp2[16];
    dataDst[16 * ISTRIDE + coter] = temp2[13] * 2 - temp2[14] - temp2[15] * 2 + temp2[16];
    dataDst[17 * ISTRIDE + coter] = temp2[13] * 4 - temp2[15] * 5 + temp2[17];
    dataDst[18 * ISTRIDE + coter] = temp2[18] * 4 - temp2[20] * 5 + temp2[22];
    dataDst[19 * ISTRIDE + coter] = -temp2[19] * 4 - temp2[20] * 4 + temp2[21] + temp2[22];
    dataDst[20 * ISTRIDE + coter] = temp2[19] * 4 - temp2[20] * 4 - temp2[21] + temp2[22];
    dataDst[21 * ISTRIDE + coter] = -temp2[19] * 2 - temp2[20] + temp2[21] * 2 + temp2[22];
    dataDst[22 * ISTRIDE + coter] = temp2[19] * 2 - temp2[20] - temp2[21] * 2 + temp2[22];
    dataDst[23 * ISTRIDE + coter] = temp2[19] * 4 - temp2[21] * 5 + temp2[23];
    dataDst[24 * ISTRIDE + coter] = temp2[24] * 4 - temp2[26] * 5 + temp2[28];
    dataDst[25 * ISTRIDE + coter] = -temp2[25] * 4 - temp2[26] * 4 + temp2[27] + temp2[28];
    dataDst[26 * ISTRIDE + coter] = temp2[25] * 4 - temp2[26] * 4 - temp2[27] + temp2[28];
    dataDst[27 * ISTRIDE + coter] = -temp2[25] * 2 - temp2[26] + temp2[27] * 2 + temp2[28];
    dataDst[28 * ISTRIDE + coter] = temp2[25] * 2 - temp2[26] - temp2[27] * 2 + temp2[28];
    dataDst[29 * ISTRIDE + coter] = temp2[25] * 4 - temp2[27] * 5 + temp2[29];
    dataDst[30 * ISTRIDE + coter] = temp2[30] * 4 - temp2[32] * 5 + temp2[34];
    dataDst[31 * ISTRIDE + coter] = -temp2[31] * 4 - temp2[32] * 4 + temp2[33] + temp2[34];
    dataDst[32 * ISTRIDE + coter] = temp2[31] * 4 - temp2[32] * 4 - temp2[33] + temp2[34];
    dataDst[33 * ISTRIDE + coter] = -temp2[31] * 2 - temp2[32] + temp2[33] * 2 + temp2[34];
    dataDst[34 * ISTRIDE + coter] = temp2[31] * 2 - temp2[32] - temp2[33] * 2 + temp2[34];
    dataDst[35 * ISTRIDE + coter] = temp2[31] * 4 - temp2[33] * 5 + temp2[35];

    (*counter)++;
}

static void filter_transform_4x3(const float* restrict filter, const int C, const int K, float* restrict out) {
    int m, n, x;
    const float *F;
    const float r4 = 1.0 / 4;
    const float r6 = 1.0 / 6;
    const float r12 = 1.0 / 12;
    const float r24 = 1.0 / 24;

#pragma omp parallel for collapse(2) private(m, n, x, F)
#pragma simd
    for (m = 0; m < K; ++m) {
        for (n = 0; n < C; ++n) {
            float c1[18] __attribute__((aligned(64)));
            F = filter + n * 3 * 3 + m * 3 * 3 * C;
            c1[0] = r4 * F[0];
            c1[1] = r4 * F[1];
            c1[2] = r4 * F[2];
            c1[3] = -r6 * (F[0] + F[3] + F[6]);
            c1[4] = -r6 * (F[1] + F[4] + F[7]);
            c1[5] = -r6 * (F[2] + F[5] + F[8]);
            c1[6] = -r6 * (F[0] - F[3] + F[6]);
            c1[7] = -r6 * (F[1] - F[4] + F[7]);
            c1[8] = -r6 * (F[2] - F[5] + F[8]);
            c1[9] =  r24 * F[0] + r12 * F[3] + r6 * F[6];
            c1[10] = r24 * F[1] + r12 * F[4] + r6 * F[7];
            c1[11] = r24 * F[2] + r12 * F[5] + r6 * F[8];
            c1[12] = r24 * F[0] - r12 * F[3] + r6 * F[6];
            c1[13] = r24 * F[1] - r12 * F[4] + r6 * F[7];
            c1[14] = r24 * F[2] - r12 * F[5] + r6 * F[8];
            c1[15] = F[6];
            c1[16] = F[7];
            c1[17] = F[8];

            float c2[36] __attribute__((aligned(64)));
            c2[0] = r4 * c1[0];
            c2[1] = -r6 * (c1[0] + c1[1] + c1[2]);
            c2[2] = -r6 * (c1[0] - c1[1] + c1[2]);
            c2[3] = r24 * c1[0] + r12 * c1[1] + r6 * c1[2];
            c2[4] = r24 * c1[0] - r12 * c1[1] + r6 * c1[2];
            c2[5] = c1[2];

            c2[6] = r4 * c1[3];
            c2[7] = -r6 * (c1[3] + c1[4] + c1[5]);
            c2[8] = -r6 * (c1[3] - c1[4] + c1[5]);
            c2[9] = r24 * c1[3] + r12 * c1[4] + r6 * c1[5];
            c2[10] = r24 * c1[3] - r12 * c1[4] + r6 * c1[5];
            c2[11] = c1[5];

            c2[12] = r4 * c1[6];
            c2[13] = -r6 * (c1[6] + c1[7] + c1[8]);
            c2[14] = -r6 * (c1[6] - c1[7] + c1[8]);
            c2[15] = r24 * c1[6] + r12 * c1[7] + r6 * c1[8];
            c2[16] = r24 * c1[6] - r12 * c1[7] + r6 * c1[8];
            c2[17] = c1[8];

            c2[18] = r4 * c1[9];
            c2[19] = -r6 * (c1[9] + c1[10] + c1[11]);
            c2[20] = -r6 * (c1[9] - c1[10] + c1[11]);
            c2[21] = r24 * c1[9] + r12 * c1[10] + r6 * c1[11];
            c2[22] = r24 * c1[9] - r12 * c1[10] + r6 * c1[11];
            c2[23] = c1[11];

            c2[24] = r4 * c1[12];
            c2[25] = -r6 * (c1[12] + c1[13] + c1[14]);
            c2[26] = -r6 * (c1[12] - c1[13] + c1[14]);
            c2[27] = r24 * c1[12] + r12 * c1[13] + r6 * c1[14];
            c2[28] = r24 * c1[12] - r12 * c1[13] + r6 * c1[14];
            c2[29] = c1[14];

            c2[30] = r4 * c1[15];
            c2[31] = -r6 * (c1[15] + c1[16] + c1[17]);
            c2[32] = -r6 * (c1[15] - c1[16] + c1[17]);
            c2[33] = r24 * c1[15] + r12 * c1[16] + r6 * c1[17];
            c2[34] = r24 * c1[15] - r12 * c1[16] + r6 * c1[17];
            c2[35] = c1[17];

#pragma unroll(9)
            for (x = 0; x < 36; ++x) {
                out[x * FSTRIDE + m * C + n] = c2[x];
            }
        }
    }
}

static void out_transform_4x3_16t(int x, int y, int nrows,
                                  const float* dataSrc, float* dataDst,
                                  int *counter) {
    int coter = *counter;
    float c1[384] __attribute__((aligned(64)));
    __m512 bufA[36], bufB, bufC, bufD, bufE, bufF, bufG, bufH, bufI;
    __m512 bufTemp[24];

    __m512i idx0 = _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4,
                                    19, 3, 18, 2, 17, 1, 16, 0);
    __m512i idx1 = _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12,
                                    27, 11, 26, 10, 25, 9, 24, 8);

    /* 0  4  8  12 16 20 24 28 32 36 40 44 48 52 56 60
       1  5  9  13 17 21 25 29 33 37 41 45 49 53 57 61
       2  6  10 14 18 22 26 30 34 38 42 46 50 54 58 62
       3  7  11 15 19 23 27 31 35 39 43 47 51 55 59 63 */

    /* 0  2  4  6  8  10 12 14 16 18 20 22 24 26 28 30
       1  3  5  7  9  11 13 15 17 19 21 23 25 27 29 31
       32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62
       33 35 37 39 41 43 45 47 49 51 53 55 57 59 61 63 */

    /* 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
       16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
       32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
       48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 */

    bufA[0] = _mm512_load_ps(dataSrc + 0 * OSTRIDE + coter);
    bufA[1] = _mm512_load_ps(dataSrc + 1 * OSTRIDE + coter);
    bufA[2] = _mm512_load_ps(dataSrc + 2 * OSTRIDE + coter);
    bufA[3] = _mm512_load_ps(dataSrc + 3 * OSTRIDE + coter);
    bufA[4] = _mm512_load_ps(dataSrc + 4 * OSTRIDE + coter);
    bufA[5] = _mm512_load_ps(dataSrc + 5 * OSTRIDE + coter);
    bufA[6] = _mm512_load_ps(dataSrc + 6 * OSTRIDE + coter);
    bufA[7] = _mm512_load_ps(dataSrc + 7 * OSTRIDE + coter);
    bufA[8] = _mm512_load_ps(dataSrc + 8 * OSTRIDE + coter);
    bufA[9] = _mm512_load_ps(dataSrc + 9 * OSTRIDE + coter);
    bufA[10] = _mm512_load_ps(dataSrc + 10 * OSTRIDE + coter);
    bufA[11] = _mm512_load_ps(dataSrc + 11 * OSTRIDE + coter);
    bufA[12] = _mm512_load_ps(dataSrc + 12 * OSTRIDE + coter);
    bufA[13] = _mm512_load_ps(dataSrc + 13 * OSTRIDE + coter);
    bufA[14] = _mm512_load_ps(dataSrc + 14 * OSTRIDE + coter);
    bufA[15] = _mm512_load_ps(dataSrc + 15 * OSTRIDE + coter);
    bufA[16] = _mm512_load_ps(dataSrc + 16 * OSTRIDE + coter);
    bufA[17] = _mm512_load_ps(dataSrc + 17 * OSTRIDE + coter);
    bufA[18] = _mm512_load_ps(dataSrc + 18 * OSTRIDE + coter);
    bufA[19] = _mm512_load_ps(dataSrc + 19 * OSTRIDE + coter);
    bufA[20] = _mm512_load_ps(dataSrc + 20 * OSTRIDE + coter);
    bufA[21] = _mm512_load_ps(dataSrc + 21 * OSTRIDE + coter);
    bufA[22] = _mm512_load_ps(dataSrc + 22 * OSTRIDE + coter);
    bufA[23] = _mm512_load_ps(dataSrc + 23 * OSTRIDE + coter);
    bufA[24] = _mm512_load_ps(dataSrc + 24 * OSTRIDE + coter);
    bufA[25] = _mm512_load_ps(dataSrc + 25 * OSTRIDE + coter);
    bufA[26] = _mm512_load_ps(dataSrc + 26 * OSTRIDE + coter);
    bufA[27] = _mm512_load_ps(dataSrc + 27 * OSTRIDE + coter);
    bufA[28] = _mm512_load_ps(dataSrc + 28 * OSTRIDE + coter);
    bufA[29] = _mm512_load_ps(dataSrc + 29 * OSTRIDE + coter);
    bufA[30] = _mm512_load_ps(dataSrc + 30 * OSTRIDE + coter);
    bufA[31] = _mm512_load_ps(dataSrc + 31 * OSTRIDE + coter);
    bufA[32] = _mm512_load_ps(dataSrc + 32 * OSTRIDE + coter);
    bufA[33] = _mm512_load_ps(dataSrc + 33 * OSTRIDE + coter);
    bufA[34] = _mm512_load_ps(dataSrc + 34 * OSTRIDE + coter);
    bufA[35] = _mm512_load_ps(dataSrc + 35 * OSTRIDE + coter);

    __m512 m2 = _mm512_set1_ps(2);
    __m512 m4 = _mm512_set1_ps(4);
    __m512 m8 = _mm512_set1_ps(8);

    bufTemp[0] = _mm512_add_ps(bufA[0], bufA[6]);
    bufTemp[1] = _mm512_add_ps(bufA[1], bufA[7]);
    bufTemp[2] = _mm512_add_ps(bufA[2], bufA[8]);
    bufTemp[3] = _mm512_add_ps(bufA[3], bufA[9]);
    bufTemp[4] = _mm512_add_ps(bufA[4], bufA[10]);
    bufTemp[5] = _mm512_add_ps(bufA[5], bufA[11]);
    bufTemp[0] = _mm512_add_ps(bufTemp[0], bufA[12]);
    bufTemp[1] = _mm512_add_ps(bufTemp[1], bufA[13]);
    bufTemp[2] = _mm512_add_ps(bufTemp[2], bufA[14]);
    bufTemp[3] = _mm512_add_ps(bufTemp[3], bufA[15]);
    bufTemp[4] = _mm512_add_ps(bufTemp[4], bufA[16]);
    bufTemp[5] = _mm512_add_ps(bufTemp[5], bufA[17]);
    bufTemp[0] = _mm512_add_ps(bufTemp[0], bufA[18]);
    bufTemp[1] = _mm512_add_ps(bufTemp[1], bufA[19]);
    bufTemp[2] = _mm512_add_ps(bufTemp[2], bufA[20]);
    bufTemp[3] = _mm512_add_ps(bufTemp[3], bufA[21]);
    bufTemp[4] = _mm512_add_ps(bufTemp[4], bufA[22]);
    bufTemp[5] = _mm512_add_ps(bufTemp[5], bufA[23]);
    bufTemp[0] = _mm512_add_ps(bufTemp[0], bufA[24]);
    bufTemp[1] = _mm512_add_ps(bufTemp[1], bufA[25]);
    bufTemp[2] = _mm512_add_ps(bufTemp[2], bufA[26]);
    bufTemp[3] = _mm512_add_ps(bufTemp[3], bufA[27]);
    bufTemp[4] = _mm512_add_ps(bufTemp[4], bufA[28]);
    bufTemp[5] = _mm512_add_ps(bufTemp[5], bufA[29]);

    bufTemp[6] = _mm512_sub_ps(bufA[6], bufA[12]);
    bufTemp[7] = _mm512_sub_ps(bufA[7], bufA[13]);
    bufTemp[8] = _mm512_sub_ps(bufA[8], bufA[14]);
    bufTemp[9] = _mm512_sub_ps(bufA[9], bufA[15]);
    bufTemp[10] = _mm512_sub_ps(bufA[10], bufA[16]);
    bufTemp[11] = _mm512_sub_ps(bufA[11], bufA[17]);
    bufTemp[6] = _mm512_fmadd_ps(bufA[18], m2, bufTemp[6]);
    bufTemp[7] = _mm512_fmadd_ps(bufA[19], m2, bufTemp[7]);
    bufTemp[8] = _mm512_fmadd_ps(bufA[20], m2, bufTemp[8]);
    bufTemp[9] = _mm512_fmadd_ps(bufA[21], m2, bufTemp[9]);
    bufTemp[10] = _mm512_fmadd_ps(bufA[22], m2, bufTemp[10]);
    bufTemp[11] = _mm512_fmadd_ps(bufA[23], m2, bufTemp[11]);
    bufTemp[6] = _mm512_fnmadd_ps(bufA[24], m2, bufTemp[6]);
    bufTemp[7] = _mm512_fnmadd_ps(bufA[25], m2, bufTemp[7]);
    bufTemp[8] = _mm512_fnmadd_ps(bufA[26], m2, bufTemp[8]);
    bufTemp[9] = _mm512_fnmadd_ps(bufA[27], m2, bufTemp[9]);
    bufTemp[10] = _mm512_fnmadd_ps(bufA[28], m2, bufTemp[10]);
    bufTemp[11] = _mm512_fnmadd_ps(bufA[29], m2, bufTemp[11]);

    bufTemp[12] = _mm512_add_ps(bufA[6], bufA[12]);
    bufTemp[13] = _mm512_add_ps(bufA[7], bufA[13]);
    bufTemp[14] = _mm512_add_ps(bufA[8], bufA[14]);
    bufTemp[15] = _mm512_add_ps(bufA[9], bufA[15]);
    bufTemp[16] = _mm512_add_ps(bufA[10], bufA[16]);
    bufTemp[17] = _mm512_add_ps(bufA[11], bufA[17]);
    bufTemp[12] = _mm512_fmadd_ps(m4, bufA[18], bufTemp[12]);
    bufTemp[13] = _mm512_fmadd_ps(m4, bufA[19], bufTemp[13]);
    bufTemp[14] = _mm512_fmadd_ps(m4, bufA[20], bufTemp[14]);
    bufTemp[15] = _mm512_fmadd_ps(m4, bufA[21], bufTemp[15]);
    bufTemp[16] = _mm512_fmadd_ps(m4, bufA[22], bufTemp[16]);
    bufTemp[17] = _mm512_fmadd_ps(m4, bufA[23], bufTemp[17]);
    bufTemp[12] = _mm512_fmadd_ps(m4, bufA[24], bufTemp[12]);
    bufTemp[13] = _mm512_fmadd_ps(m4, bufA[25], bufTemp[13]);
    bufTemp[14] = _mm512_fmadd_ps(m4, bufA[26], bufTemp[14]);
    bufTemp[15] = _mm512_fmadd_ps(m4, bufA[27], bufTemp[15]);
    bufTemp[16] = _mm512_fmadd_ps(m4, bufA[28], bufTemp[16]);
    bufTemp[17] = _mm512_fmadd_ps(m4, bufA[29], bufTemp[17]);

    bufTemp[18] = _mm512_sub_ps(bufA[6], bufA[12]);
    bufTemp[19] = _mm512_sub_ps(bufA[7], bufA[13]);
    bufTemp[20] = _mm512_sub_ps(bufA[8], bufA[14]);
    bufTemp[21] = _mm512_sub_ps(bufA[9], bufA[15]);
    bufTemp[22] = _mm512_sub_ps(bufA[10], bufA[16]);
    bufTemp[23] = _mm512_sub_ps(bufA[11], bufA[17]);
    bufTemp[18] = _mm512_fmadd_ps(m8, bufA[18], bufTemp[18]);
    bufTemp[19] = _mm512_fmadd_ps(m8, bufA[19], bufTemp[19]);
    bufTemp[20] = _mm512_fmadd_ps(m8, bufA[20], bufTemp[20]);
    bufTemp[21] = _mm512_fmadd_ps(m8, bufA[21], bufTemp[21]);
    bufTemp[22] = _mm512_fmadd_ps(m8, bufA[22], bufTemp[22]);
    bufTemp[23] = _mm512_fmadd_ps(m8, bufA[23], bufTemp[23]);
    bufTemp[18] = _mm512_fnmadd_ps(m8, bufA[24], bufTemp[18]);
    bufTemp[19] = _mm512_fnmadd_ps(m8, bufA[25], bufTemp[19]);
    bufTemp[20] = _mm512_fnmadd_ps(m8, bufA[26], bufTemp[20]);
    bufTemp[21] = _mm512_fnmadd_ps(m8, bufA[27], bufTemp[21]);
    bufTemp[22] = _mm512_fnmadd_ps(m8, bufA[28], bufTemp[22]);
    bufTemp[23] = _mm512_fnmadd_ps(m8, bufA[29], bufTemp[23]);
    bufTemp[18] = _mm512_add_ps(bufA[30], bufTemp[18]);
    bufTemp[19] = _mm512_add_ps(bufA[31], bufTemp[19]);
    bufTemp[20] = _mm512_add_ps(bufA[32], bufTemp[20]);
    bufTemp[21] = _mm512_add_ps(bufA[33], bufTemp[21]);
    bufTemp[22] = _mm512_add_ps(bufA[34], bufTemp[22]);
    bufTemp[23] = _mm512_add_ps(bufA[35], bufTemp[23]);

    bufB = _mm512_add_ps(bufTemp[0], bufTemp[1]);
    bufB = _mm512_add_ps(bufB, bufTemp[2]);
    bufB = _mm512_add_ps(bufB, bufTemp[3]);
    bufB = _mm512_add_ps(bufB, bufTemp[4]);

    bufC = _mm512_sub_ps(bufTemp[1], bufTemp[2]);
    bufC = _mm512_fmadd_ps(m2, bufTemp[3], bufC);
    bufC = _mm512_fnmadd_ps(m2, bufTemp[4], bufC);

    bufD = _mm512_add_ps(bufTemp[1], bufTemp[2]);
    bufD = _mm512_fmadd_ps(m4, bufTemp[3], bufD);
    bufD = _mm512_fmadd_ps(m4, bufTemp[4], bufD);

    bufE = _mm512_sub_ps(bufTemp[1], bufTemp[2]);
    bufE = _mm512_fmadd_ps(m8, bufTemp[3], bufE);
    bufE = _mm512_fnmadd_ps(m8, bufTemp[4], bufE);
    bufE = _mm512_add_ps(bufTemp[5], bufE);

    bufF = _mm512_permutex2var_ps(bufB, idx0, bufD);
    bufG = _mm512_permutex2var_ps(bufC, idx0, bufE);
    bufH = _mm512_permutex2var_ps(bufB, idx1, bufD);
    bufI = _mm512_permutex2var_ps(bufC, idx1, bufE);

    bufB = _mm512_permutex2var_ps(bufF, idx0, bufG);
    bufC = _mm512_permutex2var_ps(bufF, idx1, bufG);
    bufD = _mm512_permutex2var_ps(bufH, idx0, bufI);
    bufE = _mm512_permutex2var_ps(bufH, idx1, bufI);

    _mm512_store_ps(dataDst + (x + 0) * nrows + y + 0, bufB);
    _mm512_store_ps(dataDst + (x + 0) * nrows + y + 16, bufC);
    _mm512_store_ps(dataDst + (x + 0) * nrows + y + 32, bufD);
    _mm512_store_ps(dataDst + (x + 0) * nrows + y + 48, bufE);

    bufB = _mm512_add_ps(bufTemp[6], bufTemp[7]);
    bufB = _mm512_add_ps(bufB, bufTemp[8]);
    bufB = _mm512_add_ps(bufB, bufTemp[9]);
    bufB = _mm512_add_ps(bufB, bufTemp[10]);

    bufC = _mm512_sub_ps(bufTemp[7], bufTemp[8]);
    bufC = _mm512_fmadd_ps(m2, bufTemp[9], bufC);
    bufC = _mm512_fnmadd_ps(m2, bufTemp[10], bufC);

    bufD = _mm512_add_ps(bufTemp[7], bufTemp[8]);
    bufD = _mm512_fmadd_ps(m4, bufTemp[9], bufD);
    bufD = _mm512_fmadd_ps(m4, bufTemp[10], bufD);

    bufE = _mm512_sub_ps(bufTemp[7], bufTemp[8]);
    bufE = _mm512_fmadd_ps(m8, bufTemp[9], bufE);
    bufE = _mm512_fnmadd_ps(m8, bufTemp[10], bufE);
    bufE = _mm512_add_ps(bufTemp[11], bufE);

    bufF = _mm512_permutex2var_ps(bufB, idx0, bufD);
    bufG = _mm512_permutex2var_ps(bufC, idx0, bufE);
    bufH = _mm512_permutex2var_ps(bufB, idx1, bufD);
    bufI = _mm512_permutex2var_ps(bufC, idx1, bufE);

    bufB = _mm512_permutex2var_ps(bufF, idx0, bufG);
    bufC = _mm512_permutex2var_ps(bufF, idx1, bufG);
    bufD = _mm512_permutex2var_ps(bufH, idx0, bufI);
    bufE = _mm512_permutex2var_ps(bufH, idx1, bufI);

    _mm512_store_ps(dataDst + (x + 1) * nrows + y + 0, bufB);
    _mm512_store_ps(dataDst + (x + 1) * nrows + y + 16, bufC);
    _mm512_store_ps(dataDst + (x + 1) * nrows + y + 32, bufD);
    _mm512_store_ps(dataDst + (x + 1) * nrows + y + 48, bufE);

    bufB = _mm512_add_ps(bufTemp[12], bufTemp[13]);
    bufB = _mm512_add_ps(bufB, bufTemp[14]);
    bufB = _mm512_add_ps(bufB, bufTemp[15]);
    bufB = _mm512_add_ps(bufB, bufTemp[16]);

    bufC = _mm512_sub_ps(bufTemp[13], bufTemp[14]);
    bufC = _mm512_fmadd_ps(m2, bufTemp[15], bufC);
    bufC = _mm512_fnmadd_ps(m2, bufTemp[16], bufC);

    bufD = _mm512_add_ps(bufTemp[13], bufTemp[14]);
    bufD = _mm512_fmadd_ps(m4, bufTemp[15], bufD);
    bufD = _mm512_fmadd_ps(m4, bufTemp[16], bufD);

    bufE = _mm512_sub_ps(bufTemp[13], bufTemp[14]);
    bufE = _mm512_fmadd_ps(m8, bufTemp[15], bufE);
    bufE = _mm512_fnmadd_ps(m8, bufTemp[16], bufE);
    bufE = _mm512_add_ps(bufTemp[17], bufE);

    bufF = _mm512_permutex2var_ps(bufB, idx0, bufD);
    bufG = _mm512_permutex2var_ps(bufC, idx0, bufE);
    bufH = _mm512_permutex2var_ps(bufB, idx1, bufD);
    bufI = _mm512_permutex2var_ps(bufC, idx1, bufE);

    bufB = _mm512_permutex2var_ps(bufF, idx0, bufG);
    bufC = _mm512_permutex2var_ps(bufF, idx1, bufG);
    bufD = _mm512_permutex2var_ps(bufH, idx0, bufI);
    bufE = _mm512_permutex2var_ps(bufH, idx1, bufI);

    _mm512_store_ps(dataDst + (x + 2) * nrows + y + 0, bufB);
    _mm512_store_ps(dataDst + (x + 2) * nrows + y + 16, bufC);
    _mm512_store_ps(dataDst + (x + 2) * nrows + y + 32, bufD);
    _mm512_store_ps(dataDst + (x + 2) * nrows + y + 48, bufE);

    bufB = _mm512_add_ps(bufTemp[18], bufTemp[19]);
    bufB = _mm512_add_ps(bufB, bufTemp[20]);
    bufB = _mm512_add_ps(bufB, bufTemp[21]);
    bufB = _mm512_add_ps(bufB, bufTemp[22]);

    bufC = _mm512_sub_ps(bufTemp[19], bufTemp[20]);
    bufC = _mm512_fmadd_ps(m2, bufTemp[21], bufC);
    bufC = _mm512_fnmadd_ps(m2, bufTemp[22], bufC);

    bufD = _mm512_add_ps(bufTemp[19], bufTemp[20]);
    bufD = _mm512_fmadd_ps(m4, bufTemp[21], bufD);
    bufD = _mm512_fmadd_ps(m4, bufTemp[22], bufD);

    bufE = _mm512_sub_ps(bufTemp[19], bufTemp[20]);
    bufE = _mm512_fmadd_ps(m8, bufTemp[21], bufE);
    bufE = _mm512_fnmadd_ps(m8, bufTemp[22], bufE);
    bufE = _mm512_add_ps(bufTemp[23], bufE);

    bufF = _mm512_permutex2var_ps(bufB, idx0, bufD);
    bufG = _mm512_permutex2var_ps(bufC, idx0, bufE);
    bufH = _mm512_permutex2var_ps(bufB, idx1, bufD);
    bufI = _mm512_permutex2var_ps(bufC, idx1, bufE);

    bufB = _mm512_permutex2var_ps(bufF, idx0, bufG);
    bufC = _mm512_permutex2var_ps(bufF, idx1, bufG);
    bufD = _mm512_permutex2var_ps(bufH, idx0, bufI);
    bufE = _mm512_permutex2var_ps(bufH, idx1, bufI);

    _mm512_store_ps(dataDst + (x + 3) * nrows + y + 0, bufB);
    _mm512_store_ps(dataDst + (x + 3) * nrows + y + 16, bufC);
    _mm512_store_ps(dataDst + (x + 3) * nrows + y + 32, bufD);
    _mm512_store_ps(dataDst + (x + 3) * nrows + y + 48, bufE);

    *counter += 16;
}

static inline void pad_out_transform(int x, int y, int lenX, int lenY, int nrows, const float *dataSrc,
                                     float *temp, float *dataDst, int *counter) {
    if (0 == lenX || 0 == lenY) {
        return;
    }
    out_transform_4x3_16t(0, 0, 64, dataSrc, temp, counter);
    /*for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 64; ++j) {
            cout << temp[i * 64 + j] << ' ';
        }
        cout << endl;
    }*/
    for (int i = 0; i < lenX; ++i) {
        for (int j = 0; j < lenY; ++j) {
            dataDst[(x + i) * nrows + y + j] = temp[i * 64 + j];
        }
    }
}

static inline void out_transform_4x3_1t(int x, int y, int nrows, const float *dataSrc,
                                        float *dataDst, int *counter) {
    int coter = *counter;
    float c1[36]__attribute__((aligned(64)));
    c1[0] = dataSrc[0 * OSTRIDE + coter];
    c1[1] = dataSrc[1 * OSTRIDE + coter];
    c1[2] = dataSrc[2 * OSTRIDE + coter];
    c1[3] = dataSrc[3 * OSTRIDE + coter];
    c1[4] = dataSrc[4 * OSTRIDE + coter];
    c1[5] = dataSrc[5 * OSTRIDE + coter];
    c1[6] = dataSrc[6 * OSTRIDE + coter];
    c1[7] = dataSrc[7 * OSTRIDE + coter];
    c1[8] = dataSrc[8 * OSTRIDE + coter];
    c1[9] = dataSrc[9 * OSTRIDE + coter];
    c1[10] = dataSrc[10 * OSTRIDE + coter];
    c1[11] = dataSrc[11 * OSTRIDE + coter];
    c1[12] = dataSrc[12 * OSTRIDE + coter];
    c1[13] = dataSrc[13 * OSTRIDE + coter];
    c1[14] = dataSrc[14 * OSTRIDE + coter];
    c1[15] = dataSrc[15 * OSTRIDE + coter];
    c1[16] = dataSrc[16 * OSTRIDE + coter];
    c1[17] = dataSrc[17 * OSTRIDE + coter];
    c1[18] = dataSrc[18 * OSTRIDE + coter];
    c1[19] = dataSrc[19 * OSTRIDE + coter];
    c1[20] = dataSrc[20 * OSTRIDE + coter];
    c1[21] = dataSrc[21 * OSTRIDE + coter];
    c1[22] = dataSrc[22 * OSTRIDE + coter];
    c1[23] = dataSrc[23 * OSTRIDE + coter];
    c1[24] = dataSrc[24 * OSTRIDE + coter];
    c1[25] = dataSrc[25 * OSTRIDE + coter];
    c1[26] = dataSrc[26 * OSTRIDE + coter];
    c1[27] = dataSrc[27 * OSTRIDE + coter];
    c1[28] = dataSrc[28 * OSTRIDE + coter];
    c1[29] = dataSrc[29 * OSTRIDE + coter];
    c1[30] = dataSrc[30 * OSTRIDE + coter];
    c1[31] = dataSrc[31 * OSTRIDE + coter];
    c1[32] = dataSrc[32 * OSTRIDE + coter];
    c1[33] = dataSrc[33 * OSTRIDE + coter];
    c1[34] = dataSrc[34 * OSTRIDE + coter];
    c1[35] = dataSrc[35 * OSTRIDE + coter];

    float temp[24]__attribute__((aligned(64)));
    temp[0] = c1[0] + c1[6] + c1[12] + c1[18] + c1[24];
    temp[1] = c1[1] + c1[7] + c1[13] + c1[19] + c1[25];
    temp[2] = c1[2] + c1[8] + c1[14] + c1[20] + c1[26];
    temp[3] = c1[3] + c1[9] + c1[15] + c1[21] + c1[27];
    temp[4] = c1[4] + c1[10] + c1[16] + c1[22] + c1[28];
    temp[5] = c1[5] + c1[11] + c1[17] + c1[23] + c1[29];
    temp[6] = c1[6] - c1[12] + 2 * c1[18] - 2 * c1[24];
    temp[7] = c1[7] - c1[13] + 2 * c1[19] - 2 * c1[25];
    temp[8] = c1[8] - c1[14] + 2 * c1[20] - 2 * c1[26];
    temp[9] = c1[9] - c1[15] + 2 * c1[21] - 2 * c1[27];
    temp[10] = c1[10] - c1[16] + 2 * c1[22] - 2 * c1[28];
    temp[11] = c1[11] - c1[17] + 2 * c1[23] - 2 * c1[29];
    temp[12] = c1[6] + c1[12] + 4 * c1[18] + 4 * c1[24];
    temp[13] = c1[7] + c1[13] + 4 * c1[19] + 4 * c1[25];
    temp[14] = c1[8] + c1[14] + 4 * c1[20] + 4 * c1[26];
    temp[15] = c1[9] + c1[15] + 4 * c1[21] + 4 * c1[27];
    temp[16] = c1[10] + c1[16] + 4 * c1[22] + 4 * c1[28];
    temp[17] = c1[11] + c1[17] + 4 * c1[23] + 4 * c1[29];
    temp[18] = c1[6] - c1[12] + 8 * c1[18] - 8 * c1[24] + c1[30];
    temp[19] = c1[7] - c1[13] + 8 * c1[19] - 8 * c1[25] + c1[31];
    temp[20] = c1[8] - c1[14] + 8 * c1[20] - 8 * c1[26] + c1[32];
    temp[21] = c1[9] - c1[15] + 8 * c1[21] - 8 * c1[27] + c1[33];
    temp[22] = c1[10] - c1[16] + 8 * c1[22] - 8 * c1[28] + c1[34];
    temp[23] = c1[11] - c1[17] + 8 * c1[23] - 8 * c1[29] + c1[35];

    dataDst[(x + 0) * nrows + y] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4];
    dataDst[(x + 0) * nrows + y + 1] = temp[1] - temp[2] + 2 * temp[3] - 2 * temp[4];
    dataDst[(x + 0) * nrows + y + 2] = temp[1] + temp[2] + 4 * temp[3] + 4 * temp[4];
    dataDst[(x + 0) * nrows + y + 3] = temp[1] - temp[2] + 8 * temp[3] - 8 * temp[4] + temp[5];
    dataDst[(x + 1) * nrows + y] = temp[6] + temp[7] + temp[8] + temp[9] + temp[10];
    dataDst[(x + 1) * nrows + y + 1] = temp[7] - temp[8] + 2 * temp[9] - 2 * temp[10];
    dataDst[(x + 1) * nrows + y + 2] = temp[7] + temp[8] + 4 * temp[9] + 4 * temp[10];
    dataDst[(x + 1) * nrows + y + 3] = temp[7] - temp[8] + 8 * temp[9] - 8 * temp[10] + temp[11];
    dataDst[(x + 2) * nrows + y] = temp[12] + temp[13] + temp[14] + temp[15] + temp[16];
    dataDst[(x + 2) * nrows + y + 1] = temp[13] - temp[14] + 2 * temp[15] - 2 * temp[16];
    dataDst[(x + 2) * nrows + y + 2] = temp[13] + temp[14] + 4 * temp[15] + 4 * temp[16];
    dataDst[(x + 2) * nrows + y + 3] = temp[13] - temp[14] + 8 * temp[15] - 8 * temp[16] + temp[17];
    dataDst[(x + 3) * nrows + y] = temp[18] + temp[19] + temp[20] + temp[21] + temp[22];
    dataDst[(x + 3) * nrows + y + 1] = temp[19] - temp[20] + 2 * temp[21] - 2 * temp[22];
    dataDst[(x + 3) * nrows + y + 2] = temp[19] + temp[20] + 4 * temp[21] + 4 * temp[22];
    dataDst[(x + 3) * nrows + y + 3] = temp[19] - temp[20] + 8 * temp[21] - 8 * temp[22] + temp[23];

    (*counter)++;
}

static void get_tiles_4x3(const float* restrict image, const int ldi, const int irows, const int icols,
                          const int sizeI, const int C, float* restrict otile, const int N, const int ntiles, const int M) {
    int outHeight = irows - 2;
    int outWidth = icols - 2;
    int fullOutHeight = outHeight / 4 * 4;
    int fullOutWidth = outWidth / 64 * 64;
   
    //cout << "get tiles " << ntiles << ' ' << N * C << endl;
    #pragma omp parallel for 
    for (int t = 0; t < N * C; ++t) {
        int i, j;

        const int t1 = t / (C * M);
        const int t2 = (t % (C * M)) / M;
        const int t3 = t % M;

        const float *data = image + (t1 * M * C + t3 * C + t2) * sizeI;
        int tile_count = t * ntiles;

        const int num16t = (icols - 2) / 64 * 64;

        float temp[6 * 66]__attribute__((aligned(64)));
        for (i = 0; i < fullOutHeight; i += 4) {
            for (j = 0; j < fullOutWidth; j += 64) {
                get_tiles_4x3_16t(i, j, ldi, data, otile, &tile_count);
            }
            pad_get_tiles(i, j, 6, outWidth - fullOutWidth + 2, ldi, data, temp, otile, &tile_count);
        }
        for (j = 0; j < fullOutWidth; j += 64) {
            pad_get_tiles(i, j, outHeight - fullOutHeight + 2, 66, ldi, data, temp, otile, &tile_count);
        }
        pad_get_tiles(i, j, outHeight - fullOutHeight + 2, outWidth - fullOutWidth + 2, ldi, data, temp, otile, &tile_count);
    }

    /*for (int i = 0; i < 36; ++i) {
        for (int j = 0; j < 32; ++j) {
            cout << otile[i * ISTRIDE + j] << ' ';
        }
        cout << endl;
    }*/
        /*        for (i = 0; i < irows - 4; i += 4) {
            for (j = 0; j < num16t; j += 64) {
                get_tiles_4x3_16t(i, j, ldi, data, otile, &tile_count);
            }
#pragma simd
            for (; j < (icols - 4); j += 4) {
                get_tiles_4x3_1t(i, j, ldi, data, otile, &tile_count);
            }

            }*/

}

static void batched_gemm_4x3(const float* image, const int irows, const int icols, const float* filter, const int frows, const int fcols, float* restrict out, const int batch) {
    int t, i;
    const char trans = 'n';
    const float alpha = 1.0;
    const float beta = 0.0;
    const int ldi = irows;
    const int ldf = frows;
    const int ldo = irows;

    //cout << "batched_gemm " << 36 * batch << ' ' << ISTRIDE << ' ' << OSTRIDE << ' ' << irows << ' ' << fcols << ' ' << icols << endl;
#pragma omp parallel for collapse(2) private(t, i)
    for (i = 0; i < 36; ++i) {
        for (t = 0; t < batch; ++t) {
            const float* im = image + i * ISTRIDE + t * irows * icols;
            const float* fi = filter + i * FSTRIDE;
            float *ot = out + i * OSTRIDE + t * irows * fcols;

            sgemm(&trans, &trans, &irows, &fcols, &icols, &alpha, im, &ldi, fi, &ldf, &beta, ot, &ldo);
        }
    }

    /*for (int i = 0; i < 36; ++i) {
        for (int j = 0; j < 16; ++j) {
            cout << out[i * OSTRIDE + j] << ' ';
        }
        cout << endl;
    }*/
}

static void out_transform_4x3(const float* restrict d, const int K, const int ntiles, float* restrict out, const int ldo, const int oH, const int oW, const int N, const int M) {
    int t;
    int sizeO = oH * oW;
    const int OHP = oH / 4 * 4;
    const int OWP = oW / 4 * 4;

    //cout << "out transform " << N * K << endl;
#pragma omp parallel for private(t)
    for (t = 0; t < N * K; ++t) {
        int i, j;

        const int t1 = t / (K * M);
        const int t2 = (t % (K * M)) / M;
        const int t3 = t % M;

        float *data = out + (t1 * M * K + t3 * K + t2) * sizeO;
        int tile_offset = t * ntiles;
        const int num16t = oW / 64 * 64;
        float temp[4 * 64]__attribute__((aligned(64)));
        for (i = 0; i < OHP; i += 4) {
            for (j = 0; j < num16t; j += 64) {
                out_transform_4x3_16t(i, j, ldo, d, data, &tile_offset);
            }
            pad_out_transform(i, j, 4, oW - j, ldo, d, temp, data, &tile_offset);
        }
        for (j = 0; j < num16t; j += 64) {
            pad_out_transform(i, j, oH - i, 64, ldo, d, temp, data, &tile_offset);
        }
        pad_out_transform(i, j, oH - i, oW - j, ldo, d, temp, data, &tile_offset);
    }
            /*            #pragma simd
            for (; j < OWP; j += 4) {
                out_transform_4x3_1t(i, j, ldo, d, data, &tile_offset);
                }*/

}

void winconv_2x3(const int bblock, const int M, float* restrict image, const int irows, const int icols,
                 const int C, float* restrict filter, const int K, const int batch,
                 float* restrict out) {
    const int outHeight = irows - 2;
    const int outWidth = icols - 2;
    const int sizeI = irows * icols;
    const int tiles = (outHeight) * 0.25 * (outWidth) * 0.25;
    const int padHeight = (outHeight + 3) / 4 * 4;
    const int padWidth = (outWidth + 63) / 64 * 64;
    const int padTiles = padHeight / 4 * padWidth / 4;
    float *b_image;
    float *b_out;
    const int b_batchSize = 64;


    filter_transform_4x3(filter, C, K, t_filter);

    switch(bblock) {
    case BATCH_TOGETHER:
        timeval begin, end;
        double elapse_time;

        int temp1 = ISTRIDE;
        int temp2 = OSTRIDE;

        //ISTRIDE = batch * padTiles * C + 128;
        //OSTRIDE = batch * padTiles * K + 128;
        //gettimeofday(&begin, NULL);
        get_tiles_4x3(image, icols, irows, icols, sizeI, C, t_image, batch, padTiles, M);
        //gettimeofday(&end, NULL);
        //elapse_time = (end.tv_sec - begin.tv_sec) * 1e3 + (end.tv_usec - begin.tv_usec) * 1e-3;
        //cout << "get tiles time     = " << elapse_time << endl;

        //gettimeofday(&begin, NULL);
        batched_gemm_4x3(t_image, M * padTiles, C, t_filter, C, K, c_out, batch / M);
        //gettimeofday(&end, NULL);
        //elapse_time = (end.tv_sec - begin.tv_sec) * 1e3 + (end.tv_usec - begin.tv_usec) * 1e-3;
        //cout << "gemm time          = " << elapse_time << endl;

       
        //gettimeofday(&begin, NULL);
        out_transform_4x3(c_out, K, padTiles, out, outWidth, outHeight, outWidth, batch, M);
        //gettimeofday(&end, NULL);
        //elapse_time = (end.tv_sec - begin.tv_sec) * 1e3 + (end.tv_usec - begin.tv_usec) * 1e-3;
        //cout << "out_transform time = " << elapse_time << endl << endl;
        //ISTRIDE = temp1;
        //OSTRIDE = temp2;
        break;
    case BATCH_BLOCK:
        //ISTRIDE = batch * padTiles * C + 128;
        //OSTRIDE = batch * padTiles * K + 128;
        for (int i = 0; i < batch; i += b_batchSize) {
            b_image = image + i * C * irows * icols;
            b_out = out + i * K * outHeight * outWidth;
            get_tiles_4x3(b_image, icols, irows, icols, sizeI, C, t_image, b_batchSize, padTiles, M);
            batched_gemm_4x3(t_image, M * padTiles, C, t_filter, C, K, c_out, b_batchSize / M);
            out_transform_4x3(c_out, K, padTiles, b_out, outWidth, outHeight, outWidth, b_batchSize, M);
        }
        break;
    }
}

/*int main() {
    float *dataSrc = (float*) _mm_malloc(6 * 66 * sizeof(float), 64);
    float *dataDst = (float*) _mm_malloc(36 * 16 * sizeof(float), 64);
    int counter = 0;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 66; ++j) {
            dataSrc[i * 66 + j] = i * 66 + j;
        }
    }
    get_tiles_4x3_1t(0, 0, 66, dataSrc, dataDst, &counter);
    for (int i = 0; i < 36; ++i) {
        cout << dataDst[i * ISTRIDE] << endl;
    }

    float *filterSrc = (float*) _mm_malloc(9 * sizeof(float), 64);
    float *filterDst = (float*) _mm_malloc(36 * sizeof(float), 64);
    for (int i = 0; i < 9; ++i) {
        filterSrc[i] = i + 1;
    }
    filter_transform_4x3(filterSrc, 1, 1, filterDst);

    float *out = (float*) _mm_malloc(36 * 16 * sizeof(float), 64);
    float *outDst = (float*) _mm_malloc(4 * 64 * sizeof(float), 64);

    int t, i;
    const char trans = 'n';
    const float alpha = 1.0;
    const float beta = 0.0;

    int irows = 16;
    int frows = 1;
    int fcols = 1;
    int icols = 1;
    const int ldi = irows;
    const int ldf = frows;
    const int ldo = irows;

    for (i = 0; i < 36; ++i) {
        const float* im = dataDst + i * ISTRIDE;
        const float* fi = filterDst + i * FSTRIDE;
        float *ot = out + i * OSTRIDE;

        sgemm(&trans, &trans, &irows, &fcols, &icols, &alpha, im,
              &ldi, fi, &ldf, &beta, ot, &ldo);
    }

    int tile_offset = 0;
    out_transform_4x3_16t(0, 0, ldo, out, outDst, &tile_offset);

    cout << outDst[0] << ' ' << outDst[1] << ' ' << outDst[16] << endl;

    _mm_free(out);
    _mm_free(outDst);
    _mm_free(filterSrc);
    _mm_free(filterDst);
    _mm_free(dataSrc);
    _mm_free(dataDst);
    return 0;
}
*/
