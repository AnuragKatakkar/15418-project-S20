#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
#include "mpiMatMul.h"

__global__ void matMulKernel(float *input, int* rows, float *weights, int num_cols, bool *truth, bool mark_truth)
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
            int row = rows[(col_idx * NUM_NEURONS) + i];
            
            // We've reached the end of weight values for this column. 
            if (row == -1)
            {
                break;
            }
            else if (image[row] > 0)
            {
                sum += image[row] * weights[(col_idx * NUM_NEURONS) + i];
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

double sparseMatMulCuda(float *images, int** weights_rows, float** weights_vals, bool *truth, int img_start_idx, int img_end_idx)
{
    int img_chunk_size = img_end_idx - img_start_idx;
    int col_per_thread = NUM_NEURONS / THREADS_PER_BLOCK;

    dim3 numThreadsPerBlock(THREADS_PER_BLOCK);
    dim3 numBlocks(img_chunk_size);

    float* device_inputs;
    int* device_wts_rows;
    float* device_wts_vals;
    bool* device_truth;

    cudaMalloc((void**) &device_inputs, img_chunk_size * NUM_NEURONS * sizeof(float));
    cudaMalloc((void**) &device_wts_rows, NUM_NEURONS * NUM_NEURONS * sizeof(int));
    cudaMalloc((void**) &device_wts_vals, NUM_NEURONS * NUM_NEURONS * sizeof(float));
    cudaMalloc((void**) &device_truth, img_chunk_size * sizeof(bool));

    // Copy our images into device memory.
    cudaMemcpy(device_inputs, images, img_chunk_size * NUM_NEURONS * sizeof(float), cudaMemcpyHostToDevice);

    double kernelStart = CycleTimer::currentSeconds();

    for (int layer_idx = 0; layer_idx < NUM_LAYERS; layer_idx++)
    {
        // We assume that our image/results are already in device_input.
        // Copy in our current weight layer values and rows.
        cudaMemcpy(device_wts_rows, weights_rows[layer_idx], NUM_NEURONS * NUM_NEURONS * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(device_wts_vals, weights_vals[layer_idx], NUM_NEURONS * NUM_NEURONS * sizeof(float), cudaMemcpyHostToDevice);

        // Launch the kernel. This will move data as needed.
        bool calculateTruth = layer_idx + 1 == NUM_LAYERS;
        matMulKernel<<<numBlocks, numThreadsPerBlock>>>(device_inputs, device_wts_rows, device_wts_vals, col_per_thread, device_truth, calculateTruth);

        cudaDeviceSynchronize();
    }

    double kernelEnd = CycleTimer::currentSeconds();
    printf("Inference took %.4f s\n", (kernelEnd - kernelStart));

    cudaMemcpy(truth, device_truth, img_chunk_size * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(device_inputs);
    cudaFree(device_wts_rows);
    cudaFree(device_wts_vals);
    cudaFree(device_truth);

    return (kernelEnd - kernelStart);
}