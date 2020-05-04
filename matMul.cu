#include <stdio.h>
#include <mma.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

#define NUM_NEURONS 1024
#define NUM_IMAGES 30000 
#define NUM_LAYERS 120


__global__ void simpleMatMulKernel(float * input, float * weights, float * results) {
    int index = blockDim.x*blockIdx.x + threadIdx.x;

    for (int j = 0 ; j < NUM_NEURONS/1024 ; j++) {
        index = index + j*1024;

        //Add Bias
        results[index] = -0.3;

        for(int i = 0; i < NUM_NEURONS ; i ++) {
            if (input[blockIdx.x*NUM_NEURONS + i] == 0 || weights[i*NUM_NEURONS + threadIdx.x + j*1024] == 0) {
                continue;
            }
            results[index] += input[blockIdx.x*NUM_NEURONS + i]*weights[i*NUM_NEURONS + threadIdx.x + j*1024];
        }
    
        //ReLU
        if (results[index] < 0) {
            results[index] = 0;
        } else if (results[index] > 32) {
            results[index] = 32;
        }
    }
    // printf("%d\n", index);
    return;
}

__global__ void computeTruthKernel(float * output, short * truth) {
    int index = blockIdx.x;
    float sum = 0;
    for (int i = 0; i < NUM_NEURONS ; i++) {
        if (output[blockIdx.x*NUM_NEURONS + i] != 0){
            sum += output[blockIdx.x*NUM_NEURONS + i];
        }
    }
    if (sum > 0) {
        truth[index] = 1;
    } else {
        truth[index] = 0;
    }

}

void simplemutMulCuda(float *images, float **weights, float *results, short *truth) {

    float* device_inputs;
    float* device_wts;
    float* device_results;
    short* device_truth;

    cudaMalloc((void**) &device_inputs, NUM_IMAGES * NUM_NEURONS * sizeof(float));
    cudaMalloc((void**) &device_wts, NUM_NEURONS * NUM_NEURONS * sizeof(float));
    cudaMalloc((void**) &device_results, NUM_IMAGES * NUM_NEURONS * sizeof(float));

    dim3 blockDim(1024, 1);
    dim3 gridDim(NUM_IMAGES);

    double kernelStart = CycleTimer::currentSeconds();
    for (int i = 0; i < NUM_LAYERS ; i++){
        if (i == 0) {
            cudaMemcpy(device_inputs, images, NUM_IMAGES * NUM_NEURONS * sizeof(float), cudaMemcpyHostToDevice);
        }

        cudaMemcpy(device_wts, weights[i], NUM_NEURONS * NUM_NEURONS * sizeof(float), cudaMemcpyHostToDevice);
    
        printf("Launching Kernel : %d\n", i);
    
        simpleMatMulKernel<<<gridDim, blockDim>>>(device_inputs, device_wts, device_results);
        cudaDeviceSynchronize();

        cudaMemcpy(device_inputs, device_results,  NUM_IMAGES * NUM_NEURONS * sizeof(float), cudaMemcpyHostToHost);
    }
    double kernelEnd = CycleTimer::currentSeconds();
    printf("Feed Forward Kernel took %.4f s\n", (kernelEnd - kernelStart));
    cudaMemcpy(results, device_results,  NUM_IMAGES * NUM_NEURONS * sizeof(float), cudaMemcpyDeviceToHost);

    //Free Memory on Device
    cudaFree(device_inputs);
    cudaFree(device_wts);

    //Compute Truth Categories
    cudaMalloc((void**) &device_truth, NUM_IMAGES * sizeof(short));
    printf("Computing rowsum\n");

    double truthKernelStart = CycleTimer::currentSeconds();
    computeTruthKernel<<<NUM_IMAGES, 1>>>(device_results, device_truth);
    double truthKernelEnd = CycleTimer::currentSeconds();
    printf("Time Taken to Copmute Truth Values : %.4f ms\n", 1000.f * (truthKernelEnd - truthKernelStart));
    cudaMemcpy(truth, device_truth, NUM_IMAGES * sizeof(short), cudaMemcpyDeviceToHost);

    cudaFree(device_results);
    cudaFree(device_truth);
}
