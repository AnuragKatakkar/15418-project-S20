#include <stdio.h>
#include <mma.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <cuda_fp16.h>

#include "CycleTimer.h"

#define NUM_NEURONS 4096
#define NUM_IMAGES 60000
#define NUM_LAYERS 64

using namespace nvcuda;

#define MATRIX_M 60000
#define MATRIX_N 4096
#define MATRIX_K 4096

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Code for this kernel extended from reference implementation found in
// this NVIDIA blog : https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/
__global__ void tensorMatMulKernel(half *a, half *b, float *c, 
                             int M, int N, int K) 
{

    // printf("CORE LAUNCHED\n");
    // Leading dimensions. Packed with no transpositions.
    int lda = M;
    int ldb = K;
    int ldc = M;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over the K-dimension
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;
        
        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
    
            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    
    // Load in current value of c, scale by beta, and add to result scaled by alpha
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    
    if (cRow < M && cCol < N) {
        wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);
        //Add Bias
        for(int i=0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = acc_frag.x[i] - 0.35;

            //ReLU
            if (c_frag.x[i] < 0) {
                c_frag.x[i] = 0;
            } else if (c_frag.x[i] > 32) {
                c_frag.x[i] = 32;
            }
        }
        // Store the output
        wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
    }
}

__global__ void computeTruthKernel(float * output, short * truth) {
    int index = blockIdx.x;
    float sum = 0;
    for (int i = 0; i < NUM_NEURONS ; i++) {
        if (output[blockIdx.x + NUM_IMAGES * i] != 0){
            sum += output[blockIdx.x + NUM_IMAGES * i];
        }
    }
    if (sum > 0) {
        truth[index] = 1;
    } else {
        truth[index] = 0;
    }
}

void tensorMatMul(float *images, float **weights, float *results, short *truth) {
    printf("Called Tensor MatMul\n");

    half * matrix_a;
    half * matrix_b;
    float * c_wmma;
    short* device_truth;


    cudaMalloc((void**) &matrix_a, MATRIX_M * MATRIX_K * sizeof(half));
    cudaMalloc((void**) &matrix_b, MATRIX_K * MATRIX_N * sizeof(half));
    cudaMalloc((void**) &c_wmma, MATRIX_M * MATRIX_N * sizeof(float));

    printf("Casting to fp16\n");
    //Copy fp32 weights into fp16 
    half ** wts_fp16 = new half* [NUM_LAYERS];
    for (int i = 0; i < NUM_LAYERS ; i++){
        wts_fp16[i] = new half[NUM_NEURONS * NUM_NEURONS];
    }
    for (int k = 0 ; k < NUM_LAYERS ; k ++ ) {
        for (int i = 0; i < NUM_NEURONS * NUM_NEURONS ; i++){
            wts_fp16[k][i] = __float2half(weights[k][i]);
        }
    }

    //Copy fp32 images in fp16 format
    half *images_fp16 = new half[NUM_NEURONS * NUM_IMAGES];
    for (int j = 0; j < NUM_NEURONS * NUM_IMAGES; j++) {
        images_fp16[j] = __float2half(images[j]);
    }

    // First: using WMMA
    dim3 gridDim;
    dim3 blockDim;
    
    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);


    double time = 0.0f;
    for (int i = 0; i < NUM_LAYERS ; i++){
        double kernelStart = CycleTimer::currentSeconds();
        // if (i == 0) {
            cudaMemcpy(matrix_a, images_fp16, MATRIX_M * MATRIX_K * sizeof(half), cudaMemcpyHostToDevice);
        // }
            
        cudaMemcpy(matrix_b, wts_fp16[i], MATRIX_K * MATRIX_N * sizeof(half), cudaMemcpyHostToDevice);
        
        printf("Launching Kernel : %d\n", i);
            
        tensorMatMulKernel<<<gridDim, blockDim>>>(matrix_a, matrix_b, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K);
        cudaDeviceSynchronize();
        printf("Kernel returned\n");
        
        cudaMemcpy(results, c_wmma,  NUM_IMAGES * NUM_NEURONS * sizeof(float), cudaMemcpyDeviceToHost);
        double kernelEnd = CycleTimer::currentSeconds();
        time += (kernelEnd - kernelStart);
        for (int j = 0; j < NUM_NEURONS * NUM_IMAGES; j++) {
            images_fp16[j] = __float2half(results[j]);
        }
    }
    printf("Feed Forward Kernel took %.4f s\n", time);

    cudaMemcpy(results, c_wmma,  NUM_IMAGES * NUM_NEURONS * sizeof(float), cudaMemcpyDeviceToHost);

    //Compute Truth Categories
    cudaMalloc((void**) &device_truth, NUM_IMAGES * sizeof(short));
    printf("Computing rowsum\n");

    double truthKernelStart = CycleTimer::currentSeconds();
    computeTruthKernel<<<NUM_IMAGES, 1>>>(c_wmma, device_truth);
    double truthKernelEnd = CycleTimer::currentSeconds();
    printf("Time Taken to Copmute Truth Values : %.4f ms\n", 1000.f * (truthKernelEnd - truthKernelStart));
    cudaMemcpy(truth, device_truth, NUM_IMAGES * sizeof(short), cudaMemcpyDeviceToHost);

}