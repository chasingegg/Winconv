#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include <string.h>
#include <assert.h>

#include <mkl.h>

#include "winconv.hpp"
#include <time.h>
#include <stdint.h>
#include <sys/time.h>
#include <unistd.h>
using namespace std;
#define icrt_time_sec() ({ struct timeval tp; gettimeofday(&tp, 0); tp.tv_sec + tp. tv_usec * 1.e-6; })

#define CYCLE_NUM		100

#define F_2X3			1
//#define F_3X3			2
//#define F_HYBRID		3

int counter = 0;

int myDirectConv(float *in, float *kn, float *out,
		const int N, const int C, const int H, const int W, const int K)
{
	int inpos, knpos, outpos;

	int dimIn[4]  = {N, C, H, W};
	int dimKn[4]  = {K, C, 3, 3};
	int dimOut[4] = {N, K, H-2, W-2};

	int ingap[3] = {dimIn[1]*dimIn[2]*dimIn[3], dimIn[2]*dimIn[3], dimIn[3]};
	int kngap[3] = {dimKn[1]*dimKn[2]*dimKn[3], dimKn[2]*dimKn[3], dimKn[3]};
	int outgap[3] = {dimOut[1]*dimOut[2]*dimOut[3], dimOut[2]*dimOut[3], dimOut[3]};

	#pragma omp parallel for private(inpos, knpos, outpos)
	for(int inn = 0; inn < dimIn[0]; inn++)
		for(int knn = 0; knn < dimKn[0]; knn++)
			for(int inc = 0; inc < dimIn[1]; inc++){
				for(int outh = 0; outh < dimOut[2]; outh++)
					for(int outw = 0; outw < dimOut[3]; outw++){
						outpos = inn*outgap[0] + knn*outgap[1] + outh*outgap[2] + outw;
							for(int knh = 0; knh < dimKn[2]; knh++)
								for(int knw = 0; knw < dimKn[3]; knw++){
									inpos = inn*ingap[0] + inc*ingap[1] + (outh+knh)*ingap[2] + (outw+knw);
									//knpos = knn*kngap[0] + inc*kngap[1] + 8 - (knh*kngap[2] + knw);
									knpos = knn*kngap[0] + inc*kngap[1] + knh*kngap[2] + knw;
									out[outpos] += in[inpos] * kn[knpos];
								}
					}
			}

	return 0;
}

void winograd_conv(const int bblock2x3, const int M2x3,
		int irows, int icols,int C, int K, const int batch,
		long* total_flops, double* total_time, const int verify){
	counter++;

    long i, j, n;
    const int outHeight = irows-2;
    const int outWidth = icols-2;
    const int sizeI = irows*icols;
    const int sizeF = 3*3;
    const int sizeO = outHeight*outWidth;
    const int tiles = (outHeight)*0.5*(outWidth)*0.5;

    int mpiRank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    int num = (batch + mpiSize - 1) / mpiSize;
    int batchBegin, batchEnd;
    batchBegin = mpiRank * num;
    batchEnd = (mpiRank + 1) * num;
    if (mpiRank == mpiSize - 1) {
        batchEnd= batch;
    }
    int numBatch = batchEnd - batchBegin;
    int realBatch = numBatch;
    numBatch = (numBatch + M2x3 - 1) / M2x3 * M2x3;

    float* image, *filter, *out;
    if (mpiRank == 0) {
	image = (float *)mkl_malloc(batch*C*sizeI*sizeof(float), 64);
    } else {
        image = (float *)mkl_malloc(numBatch*C*sizeI*sizeof(float), 64);
    }
    assert(image != NULL);
	filter = (float *)mkl_malloc(K*C*sizeF*sizeof(float), 64);
    assert(filter != NULL);
    if (mpiRank == 0) {
	out = (float *)mkl_malloc(batch*K*sizeO*sizeof(float), 64);
    } else {
        out = (float *)mkl_malloc(numBatch*K*sizeO*sizeof(float), 64);
    }
    assert(out != NULL);

    //initialize image in parallel
    if (mpiRank == 0) {
        #pragma omp parallel for private(i)
        for (i = 0; i < batch*C*sizeI; i++)
            image[i] = (float)(i%11);
    } else {
        #pragma omp parallel for private(i)
        for(i = batchBegin*C*sizeI; i < batchEnd*C*sizeI; i++)
            image[i-batchBegin*C*sizeI] = (float)(i%11);
            //image[i] = rand()%5;
    }

    //initialize filter in parallel
    #pragma omp parallel for private(i)
    for(i = 0; i < K*C*sizeF; i++)
        filter[i] = (float)(i%7);
        //filter[i] = rand()%3;

    double timer;
    double timer_acc = 0.0f;

    double minTime = 1e32;
    int num_threads;

    double stime, etime;
    /* First Time */
    winconv_2x3(bblock2x3, M2x3, image, irows, icols, C, filter, K, numBatch, out);

    stime = icrt_time_sec();
    for(i = 0; i < CYCLE_NUM; i++){
        winconv_2x3(bblock2x3, M2x3, image, irows, icols, C, filter, K, numBatch, out);
    }
    etime = icrt_time_sec();

    timer_acc = etime - stime;

    if (mpiRank == 0) {
        MPI_Request req[4];
        for (int i = 1; i < mpiSize; ++i) {
            int batchBegin = i * num;
            int recvNumBatch = num;
            if (i == mpiSize - 1) {
                recvNumBatch = batch - batchBegin;
            }
            MPI_Irecv(out + batchBegin * K * sizeO, recvNumBatch*K*sizeO, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &(req[i]));
        }
        for (int i = 1; i < mpiSize; ++i) {
            MPI_Status sta;
            MPI_Wait(&(req[i]), &sta);
        }
    } else {
        MPI_Request req;
        MPI_Isend(out, numBatch*K*sizeO, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &req);
    }

    timer = timer_acc/CYCLE_NUM;
    long nflops = batch*K*C*(irows-2)*(icols-2)*3*3*2;
    double gflops = (double) nflops*1.0e-9/timer;
    *total_flops += nflops;
    *total_time += timer;

    if (mpiRank == 0) {
    if(verify){
	float* vout = (float *)malloc(batch*K*sizeO*sizeof(float));
	memset(vout, 0, batch*K*sizeO*sizeof(float));

	myDirectConv(image, filter, vout, batch, C, irows, icols, K);
	printf("CONV[%-2d], N-C-H-W-K-(Merge2x3-Block2x3) = %-3d %-3d %-3d %-3d %-3d (%-3d %-2d) : ",
			counter, batch, C, irows, icols, K, M2x3, bblock2x3 );
	for(n = 0; n < batch*sizeO*K; n++){
            if(fabs((out[n] - vout[n])/vout[n]) > 1e-4){
                printf("Output Error!!! winogradConv[%d] = %f || directConv[%d] = %f \n", n, out[n], n, vout[n]);
                break;
            }
        }
	if(n == batch*sizeO*K)
 	    printf("Output  True!!!\n");
	    free(vout);
    }
    else{
        printf("CONV[%d]:\tEFFECTIVE GFLOPS is %7.2f \tGFlops \tand timing is \t%f  ms \n", counter, gflops, timer*1000);
    }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    mkl_free(image);
    mkl_free(filter);
    mkl_free(out);
}

int main(int argc, char** argv){

    if(argc < 2){
        printf("Enter the running mode\n");
        printf("Example: ./test  0 or ./test  1\n");
    //    exit(-1);
    }

    MPI_Init(&argc, &argv);

    int i, j;
    double timer;

    int verify = 0; //fix me
    if(argc>1)
	verify = atoi(argv[1]);

    //const int max_tiles = 224*224*0.25;
    const long max_tiles = MAX_TILES;

    const int layer_num = 18; //1 40 1024 32 64     64 20 512 64 64
    const int C_array[18] = {    1,    32,  64,  64,  128, 128, 256, 256, 512, 512,  3,   32,  64,  64,  128, 96,  128, 128};
    const int IH_array[18] = {   40,   20,  20,  10,  10,  6,   6,   6,   6,   6,    100, 100, 50,  50,  20,  12,  12,  8};
    const int IW_array[18] = {   1024, 512, 512, 512, 512, 512, 512, 512, 512, 1024, 100, 100, 50,  50,  26,  12,  12,  8};
    const int K_array[18] = {    32,   64,  64,  128, 128, 256, 256, 512, 512, 2048, 32,  64,  64,  128, 96,  192, 256, 512};
    const int Batch_array[18] = {64,   64,  64,  64,  64,  64,  64,  64,  64,  64,   128, 128, 128, 128, 128, 128, 128, 128};
    int merge_array2x3[18] = {   1,    1,   1,   1,   2,   4,   4,   4,   4,   2,    1,   4,   1,   4,   1,   1,   1,   2};
    int b_block2x3[18];

    int t;
    double total_time;
    long total_flops;
    int batch = 64;
   /* for(t = 0; t < layer_num; t++){
        if(batch<Batch_array[t])
        batch=Batch_array[t];
    }*/

    compute_max_stride(layer_num, batch, C_array, IH_array, IW_array, K_array);
    decide_batch_block(layer_num, batch, C_array, IH_array, IW_array, K_array, b_block2x3);
    winconv_init_lib();

    /* Compute Convoltuion Using F(2,3) Winograd */
    total_time = 0.0f;
    total_flops = 0;

    int mpiRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    if (mpiRank == 0) {
    if(verify)
		printf("========== Verify Winograd Conv correctness for F(2,3)  ==========\n");
    else
		printf("========== Test WINOGRAD CONV Performance for F(2,3) ==========\n");
    }
    for(t = 0; t < layer_num; t++){
        int irows = IH_array[t];
        int icols = IW_array[t];
        int C = C_array[t];
        int K = K_array[t];
        int batch = Batch_array[t];

        b_block2x3[t] = BATCH_TOGETHER;
        winograd_conv(b_block2x3[t], merge_array2x3[t],
                      irows, icols, C, K, batch, &total_flops, &total_time, verify);
    }
    if (mpiRank == 0) {
    if(!verify){
        printf("WINOGRAD: OVERALL EFFECTIVE GFLOPS is %.2f GFLops and Timing is %.4f ms \n", (double)total_flops*1.0e-9/total_time, total_time*1000);
    }
    printf("\n *******************************************************************\n\n");
    }

    winconv_free_lib();

    return 0;
}
