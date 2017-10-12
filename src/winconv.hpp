#ifndef _FALCON_HPP_
#define _FALCON_HPP_

#include <mkl.h>
#include <omp.h>

// The below parameters are required to generate the scratch pad memory
// It is required to reserve enough memory to store data for all the sizes you will be working on
// For example : by default, the below parameters are set for the test network 
#define MAX_BATCH           128
#define MAX_IMAGE_CHANNELS  64
#define MAX_IROWS           1024
#define MAX_FILTER_CHANNELS 512
#define MAX_FILTERS         2048

#define BB_MEM_BOUND		4
#define BATCH_TOGETHER		0
#define BATCH_BLOCK			1

#define F2X3				2

/* STRIDE is the max batch*ntile*channel for input
 * FSTRIDE2X3 is the max C*K for filter
 * OSTRIDE2X3 is the max batch*ntile*nfilter
**/
const long MAX_TILES = (MAX_IROWS-2)*(MAX_IROWS-2)*0.25; 
#if 0
const long ISTRIDE = (MAX_BATCH)*(MAX_IMAGE_CHANNELS+18)*(MAX_TILES+13); 
const long FSTRIDE = (MAX_FILTER_CHANNELS+1)*(MAX_FILTERS+1); 
const long OSTRIDE = ISTRIDE; 
#else
extern long ISTRIDE; 
extern long FSTRIDE;
extern long OSTRIDE;
#endif
extern float* t_filter;    
extern float* t_image;    
extern float* c_out;    

void winconv(const int bblock2x3,  const int M2x3, 
		  float* image, const int irows, const int icols, 
	      const int C, float* filter, const int K, const int N, 
	      float* out);
		  
void winconv_2x3(const int bblock, const int M, float* image, const int irows, const int icols, 
	      const int C, float* filter, const int K, const int N, 
	      float* out);
		  

inline void no4k_aligned(long *);
void compute_max_stride(const int, const int, const int *, const int *, const int *, const int *);

void decide_batch_block(const int, const int, const int *, const int *, const int *, const int *, int *);
		  
void winconv_init_lib();
void winconv_free_lib();

// IMAGE LAYOUT : Image is a 4D data structure, image[N][C][H][W], where H=W=irows.
//                W is the inner most dimension with unit stride. Image data structure is stored in a linear
//                array I[N*channels*irows*irows].

// FILTER LAYOUT: Filter is a 4D data structure, filter[K][C][R][S], where R=S=3. S is the inner most dimension
//                with unit stride. Filter data structure is stored in a linear array F[K*C*3*3].

// OUTPUT LAYOUT: Ouput of convolution is a 4D data structure, out[N][K][oH][oW], where oH=oW=(irows-2).
//                oW is the inner most dimension with unit stride. output data structure is stored in a linear
//                array O[N*K*oH*oW].


// M      -> the merge factor
// image  -> pointer to I array 
// irows  -> is height or width of a square image
// C      -> number of image Channels
// Filter -> pointer to F array 
// K      -> number of filters
// N      -> batch size
// out    -> pointer to O array


// The Merge factor provides flexibility in the way the input data layout is used. 
// if M=1           -->  NCHW
// else if M=N      -->  CNHW
// else (1 < M < N) -->  (N/M)C(M*HW) 

#endif
