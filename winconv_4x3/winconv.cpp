#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mkl.h>
#include <string.h>
#include <assert.h>
#include <winconv.hpp>
using namespace std;
float* t_filter;    
float* t_image;    
float* c_out;    

#if 1
long ISTRIDE = (MAX_BATCH)*(MAX_IMAGE_CHANNELS+18)*(MAX_TILES+13); 
long FSTRIDE = (MAX_FILTER_CHANNELS+1)*(MAX_FILTERS+1); 
long OSTRIDE = ISTRIDE; 
#endif

// setup scratch memory used in the algorithm 
void winconv_init_lib()
{
    int ret;

    t_filter = (float *)mkl_malloc(36*FSTRIDE*sizeof(float), 64);
    //std::cout << 36 * FSTRIDE * sizeof(float) << std::endl;
    assert(t_filter != NULL); 
    t_image = (float *)mkl_malloc(36*ISTRIDE*sizeof(float), 64);
    //std::cout << 36 * ISTRIDE * sizeof(float) << std::endl;
    assert(t_image != NULL); 
    c_out = (float *)mkl_malloc(36*OSTRIDE*sizeof(float), 64);
    //std::cout << 36 * OSTRIDE * sizeof(float) << std::endl;
    assert(c_out != NULL); 
}

// free up 
void winconv_free_lib()
{
    mkl_free(t_filter);    
    mkl_free(t_image);    
    mkl_free(c_out);    
}

/* Make number not 4096 aligned. */
inline void no4k_aligned(long *num)
{
    long flag = *num;
	
    if(flag%4096 == 0)
        (*num) += 128;
}

/* Compute stride for input, filter and output. */
void compute_max_stride(const int lyn, const int N,
		const int *C, const int *H, const int *W, const int *K)
{
    int tmp;
    long istride, fstride, ostride;
    istride = fstride = ostride = 0;

    for(int i = 0; i < lyn; i++){
        int htile = (H[i] + 1) / 4; // outH = H[i] - 2; (outH + 3) / 4;
        int wtile = (W[i] + 1) / 4;
        tmp = N * htile * wtile * C[i];
	if(tmp > istride) istride = tmp;

	tmp = C[i] * K[i];
	if(tmp > fstride) fstride = tmp;

	tmp = N * htile * wtile * K[i];
	if(tmp > ostride) ostride = tmp;
    }

    no4k_aligned(&istride);
    no4k_aligned(&fstride);
    no4k_aligned(&ostride);

    ISTRIDE = istride;
    FSTRIDE = fstride;
    OSTRIDE = ostride;
}

/* Decide to how to divide block for batch. */
void decide_batch_block(const int lyn, const int N,
		const int *C, const int *H, const int *W, const int *K,
		int *bblock2x3)
{
    float m_used;

    /* F(2,3) */
    for(int i = 0; i < lyn; i++){
        int htile = (H[i] + 1) / 4;
        int wtile = (W[i] + 1) / 4;
	m_used  = 1.0f*(N*htile*wtile*C[i] + C[i]*K[i] + N*htile*wtile*K[i])/1024/1024/1024*36;
	m_used += 1.0f*(N*C[i]*H[i]*W[i] + K[i]*C[i]*3*3 + N*K[i]*(H[i]-2)*(W[i]-2))/1024/1024/1024;
	m_used *= 4;

	if(m_used <= BB_MEM_BOUND)
	    bblock2x3[i] = BATCH_TOGETHER;
	else
	    bblock2x3[i] = BATCH_BLOCK;
	    //bblock2x3[i] = BATCH_TOGETHER;
    }

}


void winconv(const int bblock2x3, const int M2x3, 
		float *image, const int irows, const int icols, const int C,
		float *filter, const int K, const int N, float *out)
{
    winconv_2x3(bblock2x3, M2x3, image, irows, icols, C, filter, K, N, out);
}
