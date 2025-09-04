%%writefile svg.cu

#include <stdio.h>
#include <stdlib.h>

#define FILTER_WIDTH 3
#define TILE_WIDTH 16   // Block size (TILE_WIDTH x TILE_WIDTH)
#define IMAGE_SIZE 1024 // Assume a square image (1024x1024)

// Example 3x3 edge-detection filter (Sobel)
__constant__ float filter[FILTER_WIDTH * FILTER_WIDTH] = {
    -1, -1, -1,
    -1,  8, -1,
    -1, -1, -1
};

// Global Memory Only Kernel
__global__ void convolutionGlobal(float *input, float *output, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= width) return;

    float sum = 0.0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int nx = x + j;
            int ny = y + i;
            if (nx >= 0 && nx < width && ny >= 0 && ny < width) {
                sum += input[ny * width + nx] * filter[(i + 1) * 3 + (j + 1)];
            }
        }
    }
    output[y * width + x] = sum;
}

// Shared Memory Optimized Kernel
__global__ void convolutionShared(float *input, float *output, int width) {
    __shared__ float tile[TILE_WIDTH + 2][TILE_WIDTH + 2];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x + 1, ty = threadIdx.y + 1;

    // Load shared memory (including halo)
    if (x < width && y < width) {
        tile[ty][tx] = input[y * width + x];

        if (threadIdx.x == 0 && x > 0)
            tile[ty][0] = input[y * width + x - 1];
        if (threadIdx.x == blockDim.x - 1 && x < width - 1)
            tile[ty][tx + 1] = input[y * width + x + 1];
        if (threadIdx.y == 0 && y > 0)
            tile[0][tx] = input[(y - 1) * width + x];
        if (threadIdx.y == blockDim.y - 1 && y < width - 1)
            tile[ty + 1][tx] = input[(y + 1) * width + x];
    }

    __syncthreads();

    // Compute convolution
    if (x < width && y < width) {
        float sum = 0.0;
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
                sum += tile[ty + i][tx + j] * filter[(i + 1) * 3 + (j + 1)];

        output[y * width + x] = sum;
    }
}

int main() {
    int imageSize = IMAGE_SIZE * IMAGE_SIZE * sizeof(float);
    float *h_input, *h_outputGlobal, *h_outputShared;
    float *d_input, *d_outputGlobal, *d_outputShared;

    // Allocate host memory
    h_input = (float*)malloc(imageSize);
    h_outputGlobal = (float*)malloc(imageSize);
    h_outputShared = (float*)malloc(imageSize);

    // Initialize random image
    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++)
        h_input[i] = rand() % 256;

    // Allocate device memory
    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_outputGlobal, imageSize);
    cudaMalloc(&d_outputShared, imageSize);

    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((IMAGE_SIZE + TILE_WIDTH - 1) / TILE_WIDTH, (IMAGE_SIZE + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    float timeGlobal, timeShared;

    // Measure Global Memory Kernel Time
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    convolutionGlobal<<>>(d_input, d_outputGlobal, IMAGE_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeGlobal, start, stop);

    // Measure Shared Memory Kernel Time
    cudaEventRecord(start);
    convolutionShared<<>>(d_input, d_outputShared, IMAGE_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeShared, start, stop);

    // Copy results back
    cudaMemcpy(h_outputGlobal, d_outputGlobal, imageSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_outputShared, d_outputShared, imageSize, cudaMemcpyDeviceToHost);

    // Print execution times
    printf("Global Memory Kernel Time: %.3f ms\n", timeGlobal);
    printf("Shared Memory Kernel Time: %.3f ms\n", timeShared);
    printf("Speedup: %.2fx\n", timeGlobal / timeShared);

    // Cleanup
    free(h_input);
    free(h_outputGlobal);
    free(h_outputShared);
    cudaFree(d_input);
    cudaFree(d_outputGlobal);
    cudaFree(d_outputShared);

    return 0;
}
--------------------------------------------------------------------------------------------------
!nvcc -arch=sm_75 svg.cu -o svg
!./svg
