#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
#include "mpiMatMul.h"

__global__ void matMulKernel(float *input, float *weights, int num_cols, bool *truth, bool mark_truth)
{
    int img_idx = blockIdx.x;
    int col_start = threadIdx.x * num_cols;

    __shared__ float image[NUM_NEURONS];
    __shared__ float results[NUM_NEURONS];
    __shared__ bool isPositive;

    isPositive = false;

    // Using all threads, lets load in the correct row to reduce global memory accesses. 
    for(int col_idx = col_start; col_idx < col_start + num_cols; col_idx++)
    {
        image[col_idx] = input[(img_idx * NUM_NEURONS) + col_idx];
    }

    __syncthreads();

    // Now, we'll calculate values for our assigned columns.
    for(int col_idx = col_start; col_idx < col_start + num_cols; col_idx++)
    {
        float sum = BIAS;
        for(int i = 0; i < NUM_NEURONS; i++)
        {
            // Weight matrix is transposed. 
            if (image[i] != 0)
            {
                sum += image[i] * weights[(col_idx * NUM_NEURONS) + i];
            }
        }

        if (sum <= RELU_MIN)
        {
            results[col_idx] = RELU_MIN;
        }
        else if (sum > RELU_MAX)
        {
            results[col_idx] = RELU_MAX;
            isPositive = true;
        }
        else
        {
            results[col_idx] = sum;
            isPositive = true;
        }
    }

    // Block until all results have been calculated.
    __syncthreads();

    if(mark_truth)
    {
        if (threadIdx.x == 0)
        {
            truth[img_idx] = isPositive;
        }
    }
    else
    {
        // Copy data back into input.
        for(int col_idx = col_start; col_idx < col_start + num_cols; col_idx++)
        {
            input[(img_idx * NUM_NEURONS) + col_idx] = results[col_idx];
        }
    }
}

double sparseMatMulCuda(float *images, float**weights, bool *truth, int img_start_idx, int img_end_idx)
{
    int img_chunk_size = img_end_idx - img_start_idx;
    int col_per_thread = NUM_NEURONS / THREADS_PER_BLOCK;

    dim3 numThreadsPerBlock(THREADS_PER_BLOCK);
    dim3 numBlocks(img_chunk_size);

    float* device_inputs;
    float* device_wts;
    bool* device_truth;

    cudaMalloc((void**) &device_inputs, img_chunk_size * NUM_NEURONS * sizeof(float));
    cudaMalloc((void**) &device_wts, NUM_NEURONS * NUM_NEURONS * sizeof(float));
    cudaMalloc((void**) &device_truth, img_chunk_size * sizeof(bool));

    // Copy our images into device memory.
    cudaMemcpy(device_inputs, images, img_chunk_size * NUM_NEURONS * sizeof(float), cudaMemcpyHostToDevice);

    double kernelStart = CycleTimer::currentSeconds();

    for (int layer_idx = 0; layer_idx < NUM_LAYERS; layer_idx++)
    {
        // We assume that our image/results are already in device_input.
        // Copy in our current weight layer.
        cudaMemcpy(device_wts, weights[layer_idx], NUM_NEURONS * NUM_NEURONS * sizeof(float), cudaMemcpyHostToDevice);

        // Launch the kernel. This will move data as needed.
        bool calculateTruth = layer_idx + 1 == NUM_LAYERS;
        matMulKernel<<<numBlocks, numThreadsPerBlock>>>(device_inputs, device_wts, col_per_thread, device_truth, calculateTruth);

        cudaDeviceSynchronize();
    }

    double kernelEnd = CycleTimer::currentSeconds();
    printf("Inference took %.4f s\n", (kernelEnd - kernelStart));

    cudaMemcpy(truth, device_truth, img_chunk_size * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(device_inputs);
    cudaFree(device_wts);
    cudaFree(device_truth);

    return (kernelEnd - kernelStart);
}