#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <assert.h>

#include <mkl.h>
#include <immintrin.h>

#include <winconv.hpp>

/* Input data transform part with 16 tiles. */
static void get_tiles_2x3_16t(int x, int y, int nrows,  const float *dataSrc,
		float *dataDst, int *counter)
{
	const int coter = *counter;
	__m512 bufA[16], bufB, bufC, bufD;
	__m512i idx0 = _mm512_set_epi32(30,28,26,24,22,20,18,16,14,12,10, 8,6,4,2,0);
	__m512i idx1 = _mm512_set_epi32(31,29,27,25,23,21,19,17,15,13,11, 9,7,5,3,1);

	/* 0, 1, 2, 3 */
	bufB = _mm512_load_ps(&dataSrc[(x+0)*nrows + (y+0)]);
	bufC = _mm512_load_ps(&dataSrc[(x+0)*nrows + (y+16)]);
	bufA[ 0] = _mm512_permutex2var_ps(bufB, idx0, bufC);
	bufA[ 1] = _mm512_permutex2var_ps(bufB, idx1, bufC);

	bufB = _mm512_load_ps(&dataSrc[(x+0)*nrows + (y+2)]);
	bufC = _mm512_load_ps(&dataSrc[(x+0)*nrows + (y+18)]);
	bufA[ 2] = _mm512_permutex2var_ps(bufB, idx0, bufC);
	bufA[ 3] = _mm512_permutex2var_ps(bufB, idx1, bufC);

	/* 4, 5, 6, 7 */
	bufB = _mm512_load_ps(&dataSrc[(x+1)*nrows + (y+0)]);
	bufC = _mm512_load_ps(&dataSrc[(x+1)*nrows + (y+16)]);
	bufA[ 4] = _mm512_permutex2var_ps(bufB, idx0, bufC);
	bufA[ 5] = _mm512_permutex2var_ps(bufB, idx1, bufC);

	bufB = _mm512_load_ps(&dataSrc[(x+1)*nrows + (y+2)]);
	bufC = _mm512_load_ps(&dataSrc[(x+1)*nrows + (y+18)]);
	bufA[ 6] = _mm512_permutex2var_ps(bufB, idx0, bufC);
	bufA[ 7] = _mm512_permutex2var_ps(bufB, idx1, bufC);

	/* 8, 9, 10, 11 */	
	bufB = _mm512_load_ps(&dataSrc[(x+2)*nrows + (y+0)]);
	bufC = _mm512_load_ps(&dataSrc[(x+2)*nrows + (y+16)]);
	bufA[ 8] = _mm512_permutex2var_ps(bufB, idx0, bufC);
	bufA[ 9] = _mm512_permutex2var_ps(bufB, idx1, bufC);

	bufB = _mm512_load_ps(&dataSrc[(x+2)*nrows + (y+2)]);
	bufC = _mm512_load_ps(&dataSrc[(x+2)*nrows + (y+18)]);
	bufA[10] = _mm512_permutex2var_ps(bufB, idx0, bufC);
	bufA[11] = _mm512_permutex2var_ps(bufB, idx1, bufC);

	/* 12, 13, 14, 15 */
	bufB = _mm512_load_ps(&dataSrc[(x+3)*nrows + (y+0)]);
	bufC = _mm512_load_ps(&dataSrc[(x+3)*nrows + (y+16)]);
	bufA[12] = _mm512_permutex2var_ps(bufB, idx0, bufC);
	bufA[13] = _mm512_permutex2var_ps(bufB, idx1, bufC);

	bufB = _mm512_load_ps(&dataSrc[(x+3)*nrows + (y+2)]);
	bufC = _mm512_load_ps(&dataSrc[(x+3)*nrows + (y+18)]);
	bufA[14] = _mm512_permutex2var_ps(bufB, idx0, bufC);
	bufA[15] = _mm512_permutex2var_ps(bufB, idx1, bufC);
	
	/* 0 */
	bufB = _mm512_sub_ps(bufA[ 0], bufA[ 8]);
	bufC = _mm512_sub_ps(bufA[ 2], bufA[10]);
	bufD = _mm512_sub_ps(bufB, bufC);
	_mm512_store_ps(&dataDst[ 0*ISTRIDE + coter], bufD);
	
	/* 1 */
	bufB = _mm512_sub_ps(bufA[ 1], bufA[ 9]);
	bufD = _mm512_add_ps(bufB, bufC);
	_mm512_store_ps(&dataDst[ 1*ISTRIDE + coter], bufD);
	
	/* 2 */
	bufC = _mm512_sub_ps(bufA[ 2], bufA[10]);
	bufD = _mm512_sub_ps(bufC, bufB);
	_mm512_store_ps(&dataDst[ 2*ISTRIDE + coter], bufD);
	
	/* 3 */
	bufC = _mm512_sub_ps(bufA[ 3], bufA[11]);
	bufD = _mm512_sub_ps(bufB, bufC);
	_mm512_store_ps(&dataDst[ 3*ISTRIDE + coter], bufD);
	
	/* 4 */
	bufB = _mm512_add_ps(bufA[ 4], bufA[ 8]);
	bufC = _mm512_add_ps(bufA[ 6], bufA[10]);
	bufD = _mm512_sub_ps(bufB, bufC);
	_mm512_store_ps(&dataDst[ 4*ISTRIDE + coter], bufD);
	
	/* 5 */
	bufB = _mm512_add_ps(bufA[ 5], bufA[ 9]);
	bufD = _mm512_add_ps(bufB, bufC);
	_mm512_store_ps(&dataDst[ 5*ISTRIDE + coter], bufD);
	
	/* 6 */
	bufB = _mm512_add_ps(bufA[ 5], bufA[ 9]);
	bufD = _mm512_sub_ps(bufC, bufB);
	_mm512_store_ps(&dataDst[ 6*ISTRIDE + coter], bufD);
	
	/* 7 */
	bufC = _mm512_add_ps(bufA[ 7], bufA[11]);
	bufD = _mm512_sub_ps(bufB, bufC);
	_mm512_store_ps(&dataDst[ 7*ISTRIDE + coter], bufD);
	
	/* 8 */
	bufB = _mm512_sub_ps(bufA[ 8], bufA[ 4]);
	bufC = _mm512_sub_ps(bufA[10], bufA[ 6]);
	bufD = _mm512_sub_ps(bufB, bufC);
	_mm512_store_ps(&dataDst[ 8*ISTRIDE + coter], bufD);
	
	/* 9 */
	bufB = _mm512_sub_ps(bufA[ 9], bufA[ 5]);
	bufD = _mm512_add_ps(bufB, bufC);
	_mm512_store_ps(&dataDst[ 9*ISTRIDE + coter], bufD);
	
	/* 10 */
	bufD = _mm512_sub_ps(bufC, bufB);
	_mm512_store_ps(&dataDst[10*ISTRIDE + coter], bufD);
	
	/* 11 */
	bufC = _mm512_sub_ps(bufA[11], bufA[ 7]);
	bufD = _mm512_sub_ps(bufB, bufC);
	_mm512_store_ps(&dataDst[11*ISTRIDE + coter], bufD);
	
	/* 12 */
	bufB = _mm512_sub_ps(bufA[ 4], bufA[12]);
	bufC = _mm512_sub_ps(bufA[ 6], bufA[14]);
	bufD = _mm512_sub_ps(bufB, bufC);
	_mm512_store_ps(&dataDst[12*ISTRIDE + coter], bufD);
	
	/* 13 */
	bufB = _mm512_sub_ps(bufA[ 5], bufA[13]);
	bufD = _mm512_add_ps(bufB, bufC);
	_mm512_store_ps(&dataDst[13*ISTRIDE + coter], bufD);
	
	/* 14 */
	bufD = _mm512_sub_ps(bufC, bufB);
	_mm512_store_ps(&dataDst[14*ISTRIDE + coter], bufD);
	
	/* 15 */
	bufC = _mm512_sub_ps(bufA[ 7], bufA[15]);
	bufD = _mm512_sub_ps(bufB, bufC);
	_mm512_store_ps(&dataDst[15*ISTRIDE + coter], bufD);
	
	*counter += 16; 
}

/* Input data transform part with 1 tiles. */
static inline void get_tiles_2x3_1t(int x, int y, int nrows, const float *dataSrc,
		float *dataDst, int *counter)
{
	int coter = *counter;
	float tmp[16] __attribute__((aligned(64))); 
	
	tmp[ 0] = dataSrc[(x+0)*nrows + y+0];
	tmp[ 1] = dataSrc[(x+0)*nrows + y+1];
	tmp[ 2] = dataSrc[(x+0)*nrows + y+2];
	tmp[ 3] = dataSrc[(x+0)*nrows + y+3];

	tmp[ 4] = dataSrc[(x+1)*nrows + y+0];
	tmp[ 5] = dataSrc[(x+1)*nrows + y+1];
	tmp[ 6] = dataSrc[(x+1)*nrows + y+2];
	tmp[ 7] = dataSrc[(x+1)*nrows + y+3];
	
	tmp[ 8] = dataSrc[(x+2)*nrows + y+0];
	tmp[ 9] = dataSrc[(x+2)*nrows + y+1];
	tmp[10] = dataSrc[(x+2)*nrows + y+2];
	tmp[11] = dataSrc[(x+2)*nrows + y+3];

	tmp[12] = dataSrc[(x+3)*nrows + y+0];
	tmp[13] = dataSrc[(x+3)*nrows + y+1];
	tmp[14] = dataSrc[(x+3)*nrows + y+2];
	tmp[15] = dataSrc[(x+3)*nrows + y+3];
				
	// The tranformation manually simplified
	dataDst[coter+ 0*ISTRIDE] =(tmp[0] - tmp[8 ]) - (tmp[2 ]- tmp[10]);   
	dataDst[coter+ 1*ISTRIDE] =(tmp[1] - tmp[9 ]) + (tmp[2 ]- tmp[10]); 
	dataDst[coter+ 2*ISTRIDE] =(tmp[2] - tmp[10]) - (tmp[1 ]- tmp[9 ]); 
	dataDst[coter+ 3*ISTRIDE] =(tmp[1] - tmp[9 ]) - (tmp[3 ]- tmp[11]); 
	dataDst[coter+ 4*ISTRIDE] =(tmp[4] + tmp[8 ]) - (tmp[6 ]+ tmp[10]); 
	dataDst[coter+ 5*ISTRIDE] =(tmp[5] + tmp[9 ]) + (tmp[6 ]+ tmp[10]); 
	dataDst[coter+ 6*ISTRIDE] =(tmp[6] + tmp[10]) - (tmp[5 ]+ tmp[9 ]); 
	dataDst[coter+ 7*ISTRIDE] =(tmp[5] + tmp[9 ]) - (tmp[7 ]+ tmp[11]); 
	dataDst[coter+ 8*ISTRIDE] =(tmp[8] - tmp[4 ]) - (tmp[10]- tmp[6 ]); 
	dataDst[coter+ 9*ISTRIDE] =(tmp[9] - tmp[5 ]) + (tmp[10]- tmp[6 ]); 
	dataDst[coter+10*ISTRIDE] =(tmp[10]- tmp[6 ]) - (tmp[9 ]- tmp[5 ]); 
	dataDst[coter+11*ISTRIDE] =(tmp[9] - tmp[5 ]) - (tmp[11]- tmp[7 ]); 
	dataDst[coter+12*ISTRIDE] =(tmp[4] - tmp[12]) - (tmp[6 ]- tmp[14]); 
	dataDst[coter+13*ISTRIDE] =(tmp[5] - tmp[13]) + (tmp[6 ]- tmp[14]); 
	dataDst[coter+14*ISTRIDE] =(tmp[6] - tmp[14]) - (tmp[5 ]- tmp[13]); 
	dataDst[coter+15*ISTRIDE] =(tmp[5] - tmp[13]) - (tmp[7 ]- tmp[15]); 

	(*counter)++; 

}

// INTERNAL FUNCTION : FORM MATRIX A from input data, also includes transformation F(2,3)
static void get_tiles_2x3(const float* restrict image, const int ldi, const int irows, const int icols,
		const int sizeI, const int C, float* restrict otile, const int N, const int ntiles, const int M)
{
   
    int t, u;

	#pragma omp parallel for 
	for(t = 0; t < N*C; t++){
		int i, j; 

		const int t1 = t/(C*M);
		const int t2 = (t%(C*M))/M;
		const int t3 = t%M;

		//const float* data = image+t*sizeI; 
		const float *data = image + (t1*M*C + t3*C + t2)*sizeI;
		int tile_count = t*ntiles;
		const int num16t = (icols-2)/32*32;
	
		// work on one image plane at a time, irrespective of the order
		for(i = 0; i < irows-2; i += 2){
			/* 16 tiles together */
			for(j = 0; j < num16t; j += 32){
				get_tiles_2x3_16t(i, j, ldi, data, otile, &tile_count);
			}

			/* 1 tile together */
			#pragma simd
			for(j = num16t; j < (icols-2); j += 2){
				get_tiles_2x3_1t(i, j, ldi, data, otile, &tile_count);
			}
		}
	}
}

// INTERNAL FUNCTION: FORM MATRIX B, also includes filter transform F(2,3)
static void filter_transform_2x3(const float* restrict filter, const int C, const int K, float* restrict out)
{

    int m, n, x; 
    const float *F;

    #pragma omp parallel for collapse(2) private(m, n, x, F)
	#pragma simd
    for(m = 0; m < K; m++){
        for(n = 0; n < C; n++){
            float c1[16] __attribute__((aligned(64))); 
            F = filter+n*3*3 + m*3*3*C; 

            // work on in 3x3 plane at a time
            // The tranformation manually simplified
            c1[0]  = F[0]; 
            c1[1]  = (F[0]+F[2]+F[1])*0.5f; 
            c1[2]  = (F[0]+F[2]-F[1])*0.5f; 
            c1[3]  = F[2]; 
            c1[4]  = (F[0]+F[6]+F[3])*0.5f; 
            c1[5]  = ((F[0]+F[6]+F[3])+(F[2]+F[8]+F[5])+(F[1]+F[7]+F[4]))*0.25f; 
            c1[6]  = ((F[0]+F[6]+F[3])+(F[2]+F[8]+F[5])-(F[1]+F[7]+F[4]))*0.25f; 
            c1[7]  = (F[2]+F[8]+F[5])*0.5f; 
            c1[8]  = (F[0]+F[6]-F[3])*0.5f; 
            c1[9]  = ((F[0]+F[6]-F[3])+(F[2]+F[8]-F[5])+(F[1]+F[7]-F[4]))*0.25f; 
            c1[10] = ((F[0]+F[6]-F[3])+(F[2]+F[8]-F[5])-(F[1]+F[7]-F[4]))*0.25f; 
            c1[11] = (F[2]+F[8]-F[5])*0.5f; 
            c1[12] = F[6]; 
            c1[13] = (F[6]+F[8]+F[7])*0.5f; 
            c1[14] = (F[6]+F[8]-F[7])*0.5f; 
            c1[15] = F[8]; 

            // scatter
            #pragma unroll(16)
            for(x = 0; x < 16; x++){
                out[x*FSTRIDE+m*C+n] = c1[x]; 
            }
        }
    }
}

// INTERNAL FUNCTION F(2,3)
// GEMM specific to Ist layer of VGG with (M, N, K) = (12544, 64, 3)
// MKL performs bad
static void gemm_ker(int m, int n, int k, const float* a, const int lda, const float* b, const int ldb, float* c, const int ldc)
{

    const int BLK = 16; 
    int x, xx, y, z, i; 

    for(z = 0; z < n; z++){
        for(x = 0; x < m; x += BLK){
            float p[BLK] __attribute__((aligned(64))); 
            p[0:BLK] = 0.0f; 
            #pragma unroll(3)
            for(y = 0; y < 3; y++){
                #pragma vector aligned
                for(i = 0; i < BLK; i++){
                    p[i] += a[x+i+y*lda]*b[y+z*ldb]; 
                }
            }
            c[x+z*ldc:BLK] = p[0:BLK]; 
        }
    }

}


// INTERNAL FUNCTION F(2,3)
// C = A*B with beta = 0.0f and alpha = 1.0f
// Number of gemm calls is 16*BATCH 
static void batched_gemm_2x3(const float* image, const int irows, const int icols,
		const float* filter, const int frows, const int fcols, float* restrict  out, const int batch)
{

    int t, i; 
    const char trans ='n'; 
    const float alpha = 1.0; 
    const float beta =  0.0; 
	const int ldi = irows;
	const int ldf = frows;
	const int ldo = irows;
    
    #pragma omp parallel for num_threads(68) collapse(2) private(t, i)
    for(i = 0; i < 16; i++){
        for(t = 0; t < batch; t++){
            const float* im = image+i*ISTRIDE+t*irows*icols; 
            const float* fi = filter+i*FSTRIDE; 
            float* ot = out+i*OSTRIDE+t*irows*fcols; 
			sgemm(&trans, &trans, &irows, &fcols, &icols, &alpha, im, &ldi, fi, &ldf, &beta, ot, &ldo); 
        }
    }

} 
 
/* Output data transform part with 16 tiles. */
static void out_transform_2x3_16t(int x, int y, int nrows,  const float *dataSrc,
		float *dataDst, int *counter)
{
    int coter = *counter;
    float c1[256] __attribute__((aligned(64))); 
    __m512 bufA[16], bufB, bufC, bufD, bufE;
    __m512i idx0 = _mm512_set_epi32(23, 7,22, 6,21, 5,20, 4,19, 3,18, 2,17, 1,16, 0);
    __m512i idx1 = _mm512_set_epi32(31,15,30,14,29,13,28,12,27,11,26,10,25, 9,24, 8);

    // gather the 16 elements form C to form a tile
    c1[  0:16] = dataSrc[coter +  0*OSTRIDE:16]; 
    c1[ 16:16] = dataSrc[coter +  1*OSTRIDE:16]; 
    c1[ 32:16] = dataSrc[coter +  2*OSTRIDE:16]; 
    c1[ 48:16] = dataSrc[coter +  3*OSTRIDE:16]; 
    c1[ 64:16] = dataSrc[coter +  4*OSTRIDE:16]; 
    c1[ 80:16] = dataSrc[coter +  5*OSTRIDE:16]; 
    c1[ 96:16] = dataSrc[coter +  6*OSTRIDE:16]; 
    c1[112:16] = dataSrc[coter +  7*OSTRIDE:16]; 
    c1[128:16] = dataSrc[coter +  8*OSTRIDE:16]; 
    c1[144:16] = dataSrc[coter +  9*OSTRIDE:16]; 
    c1[160:16] = dataSrc[coter + 10*OSTRIDE:16]; 
    c1[176:16] = dataSrc[coter + 11*OSTRIDE:16]; 
    c1[192:16] = dataSrc[coter + 12*OSTRIDE:16]; 
    c1[208:16] = dataSrc[coter + 13*OSTRIDE:16]; 
    c1[224:16] = dataSrc[coter + 14*OSTRIDE:16]; 
    c1[240:16] = dataSrc[coter + 15*OSTRIDE:16]; 

    /* Register store the source data */
    bufA[ 0] = _mm512_load_ps(c1+  0);
    bufA[ 1] = _mm512_load_ps(c1+ 16);
    bufA[ 2] = _mm512_load_ps(c1+ 32);
    bufA[ 3] = _mm512_load_ps(c1+ 48);
    bufA[ 4] = _mm512_load_ps(c1+ 64);
    bufA[ 5] = _mm512_load_ps(c1+ 80);
    bufA[ 6] = _mm512_load_ps(c1+ 96);
    bufA[ 7] = _mm512_load_ps(c1+112);
    bufA[ 8] = _mm512_load_ps(c1+128);
    bufA[ 9] = _mm512_load_ps(c1+144);
    bufA[10] = _mm512_load_ps(c1+160);
    bufA[11] = _mm512_load_ps(c1+176);
    bufA[12] = _mm512_load_ps(c1+192);
    bufA[13] = _mm512_load_ps(c1+208);
    bufA[14] = _mm512_load_ps(c1+224);
    bufA[15] = _mm512_load_ps(c1+240);
	
	
    /* Compute the media result */
    bufB = _mm512_add_ps(bufA[ 0], bufA[ 1]);
    bufB = _mm512_add_ps(bufB, bufA[ 2]);
    bufB = _mm512_add_ps(bufB, bufA[ 4]);
    bufB = _mm512_add_ps(bufB, bufA[ 5]);
    bufB = _mm512_add_ps(bufB, bufA[ 6]);
    bufB = _mm512_add_ps(bufB, bufA[ 8]);
    bufB = _mm512_add_ps(bufB, bufA[ 9]);
    bufB = _mm512_add_ps(bufB, bufA[10]);

    bufC = _mm512_sub_ps(bufA[ 1], bufA[ 2]);
    bufC = _mm512_sub_ps(bufC, bufA[ 3]);
    bufC = _mm512_add_ps(bufC, bufA[ 5]);
    bufC = _mm512_sub_ps(bufC, bufA[ 6]);
    bufC = _mm512_sub_ps(bufC, bufA[ 7]);
    bufC = _mm512_add_ps(bufC, bufA[ 9]);
    bufC = _mm512_sub_ps(bufC, bufA[10]);
    bufC = _mm512_sub_ps(bufC, bufA[11]);

    bufD = _mm512_add_ps(bufA[ 4], bufA[ 5]);
    bufD = _mm512_add_ps(bufD, bufA[ 6]);
    bufD = _mm512_sub_ps(bufD, bufA[ 8]);
    bufD = _mm512_sub_ps(bufD, bufA[ 9]);
    bufD = _mm512_sub_ps(bufD, bufA[10]);
    bufD = _mm512_sub_ps(bufD, bufA[12]);
    bufD = _mm512_sub_ps(bufD, bufA[13]);
    bufD = _mm512_sub_ps(bufD, bufA[14]);

    bufE = _mm512_sub_ps(bufA[ 5], bufA[ 6]);
    bufE = _mm512_sub_ps(bufE, bufA[ 7]);
    bufE = _mm512_sub_ps(bufE, bufA[ 9]);
    bufE = _mm512_add_ps(bufE, bufA[10]);
    bufE = _mm512_add_ps(bufE, bufA[11]);
    bufE = _mm512_sub_ps(bufE, bufA[13]);
    bufE = _mm512_add_ps(bufE, bufA[14]);
    bufE = _mm512_add_ps(bufE, bufA[15]);

    /* Store the finally output data */
    bufA[0] = _mm512_permutex2var_ps(bufB, idx0, bufC);
    bufA[1] = _mm512_permutex2var_ps(bufB, idx1, bufC);
    bufA[2] = _mm512_permutex2var_ps(bufD, idx0, bufE);
    bufA[3] = _mm512_permutex2var_ps(bufD, idx1, bufE);
    
    _mm512_store_ps(&dataDst[(x+0)*nrows + (y+ 0)], bufA[0]);
    _mm512_store_ps(&dataDst[(x+0)*nrows + (y+16)], bufA[1]);
    _mm512_store_ps(&dataDst[(x+1)*nrows + (y+ 0)], bufA[2]);
    _mm512_store_ps(&dataDst[(x+1)*nrows + (y+16)], bufA[3]);
    
    (*counter) += 16; 
}

/* Output data transform part with 1 tile. */
static inline void out_transform_2x3_1t(int x, int y, int nrows,  const float *dataSrc,
		float *dataDst, int *counter)
{
    int coter = *counter;
    float c1[16] __attribute__((aligned(64)));
    float temp[8] __attribute__((aligned(64))); 

    // gather the 16 elements form C to form a tile
    c1[0 ] = dataSrc[coter+0 *OSTRIDE]; 
    c1[1 ] = dataSrc[coter+1 *OSTRIDE]; 
    c1[2 ] = dataSrc[coter+2 *OSTRIDE]; 
    c1[3 ] = dataSrc[coter+3 *OSTRIDE]; 
    c1[4 ] = dataSrc[coter+4 *OSTRIDE]; 
    c1[5 ] = dataSrc[coter+5 *OSTRIDE]; 
    c1[6 ] = dataSrc[coter+6 *OSTRIDE]; 
    c1[7 ] = dataSrc[coter+7 *OSTRIDE]; 
    c1[8 ] = dataSrc[coter+8 *OSTRIDE]; 
    c1[9 ] = dataSrc[coter+9 *OSTRIDE]; 
    c1[10] = dataSrc[coter+10*OSTRIDE]; 
    c1[11] = dataSrc[coter+11*OSTRIDE]; 
    c1[12] = dataSrc[coter+12*OSTRIDE]; 
    c1[13] = dataSrc[coter+13*OSTRIDE]; 
    c1[14] = dataSrc[coter+14*OSTRIDE]; 
    c1[15] = dataSrc[coter+15*OSTRIDE]; 

    
    // The tranformation manually simplified
    temp[0] = c1[0]+c1[1]+ c1[2]; 
    temp[1] = c1[1]-c1[2]- c1[3]; 
    temp[2] = c1[4]+c1[5]+ c1[6]; 
    temp[3] = c1[5]-c1[6]- c1[7]; 
    temp[4] = c1[8]+c1[9]+ c1[10]; 
    temp[5] = c1[9]-c1[10]- c1[11]; 
    temp[6] = c1[12]+c1[13]+ c1[14]; 
    temp[7] = c1[13]-c1[14]- c1[15]; 

    dataDst[(x+0)*nrows+y]   = temp[0]+temp[2]+temp[4]; 
    dataDst[(x+0)*nrows+y+1] = temp[1]+temp[3]+temp[5]; 
    dataDst[(x+1)*nrows+y]   = temp[2]-temp[4]-temp[6]; 
    dataDst[(x+1)*nrows+y+1] = temp[3]-temp[5]-temp[7]; 
    
    (*counter)++; 
}

// INTERNAL FUNCTION F(2,3)
// Transform matrix multiplication output
static void out_transform_2x3(const float* restrict d, const int K, const int ntiles,
		float* restrict out, const int ldo,const int oH, const  int oW, const int N, const int M)
{
    
    int t; 
    int sizeO = oH*oW;
    
    #pragma omp parallel for
    for(t = 0; t < N*K; t++){
        int i, j;    
        
	const int t1 = t/(K*M);
	const int t2 = (t%(K*M))/M;
	const int t3 = t%M;

	float *data = out + (t1*M*K + t3*K + t2)*sizeO;
        int tile_offset = t*ntiles; 
        const int num16t = oW/32*32;

        // work on one output plane at a time, irrespective of the order
        for(i = 0; i < oH; i += 2){
	    /* 16 tiles together */
            for(j = 0; j < num16t; j += 32){
		out_transform_2x3_16t(i, j, ldo, d, data, &tile_offset);
            }

	    /* 1 tile together */
	    #pragma simd
            for(j = num16t; j < oW; j += 2){
		out_transform_2x3_1t(i, j, ldo, d, data, &tile_offset);
            }
        }
    }
}

// User API for winograd F(2,3)
void winconv_2x3(const int bblock, const int M, float* restrict image, const int irows, const int icols, 
		const int C, float* restrict filter, const int K, const int batch,
		float* restrict out)
{

    const int outHeight = irows-2; 
    const int outWidth = icols-2; 
    const int sizeI = irows*icols; 
    const int tiles = (outHeight)*0.5*(outWidth)*0.5; 
	
    float *b_image;
    float *b_out;
    const int b_batchSize = 32;
        
    if(batch%b_batchSize != 0){
	printf("Error: Batch can't be divided by %d!\n", b_batchSize);
	exit(0);
    }

    filter_transform_2x3(filter, C, K, t_filter);
    switch(bblock){
	case BATCH_TOGETHER:
	    get_tiles_2x3(image, icols, irows, icols, sizeI, C, t_image, batch, tiles, M); 
	    batched_gemm_2x3(t_image, M*tiles, C, t_filter, C, K, c_out, batch/M); 
	    out_transform_2x3(c_out, K, tiles, out, outWidth, outHeight, outWidth, batch, M); 
	    break;
	case BATCH_BLOCK:
	    for(int i = 0; i < batch; i += b_batchSize){
		b_image = image + i*C*irows*icols;
		b_out = out + i*K*outHeight*outWidth;
		get_tiles_2x3(b_image, icols, irows, icols, sizeI, C, t_image, b_batchSize, tiles, M); 
		batched_gemm_2x3(t_image, M*tiles, C, t_filter, C, K, c_out, b_batchSize/M); 
		out_transform_2x3(c_out, K, tiles, b_out, outWidth, outHeight, outWidth, b_batchSize, M); 
	    }
	    break;
	default:
	    printf("Error: You need to decide wether to divide block for batch!\n");
	    break;
    }
}
