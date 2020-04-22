#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#define NUM_NEURONS 1024
#define NUM_IMAGES 60000
#define NUM_LAYERS 1
__global__ void simpleMatMulKernel(short * image, float * weights, float * results) {
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    for(int i = 0; i < 1024 ; i ++) {
        results[index] += image[i]*weights[i + NUM_NEURONS*index];
    }
    // printf("%d\n", index);
    return;
}

void simplemutMulCuda(short *image, float weights[][1024], float *results) {

    short* device_image;
    float* device_wts;
    float* device_results;

    cudaMalloc((void**) &device_image, NUM_NEURONS * sizeof(short));
    cudaMalloc((void**) &device_wts, NUM_LAYERS * NUM_NEURONS * NUM_NEURONS * sizeof(float));
    cudaMalloc((void**) &device_results, NUM_NEURONS * sizeof(float));
    
    // dim3 blockDim(512, 1);
    // dim3 gridDim(2);

    cudaMemcpy(device_image, image, NUM_NEURONS * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(device_wts, weights, NUM_LAYERS * NUM_NEURONS * NUM_NEURONS * sizeof(float), cudaMemcpyHostToDevice);

    simpleMatMulKernel<<<32, 32>>>(device_image, device_wts, device_results);

    cudaMemcpy(results, device_results,  NUM_NEURONS * sizeof(float), cudaMemcpyDeviceToHost);

}