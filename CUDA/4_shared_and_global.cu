%%writefile svg.cu
#include <stdio.h>
#include <cuda.h>

#define N 1024           // Size of arrays
#define BLOCK_SIZE 256   // Threads per block

// =========================================================
//  Kernel using ONLY GLOBAL MEMORY
// =========================================================
__global__ void addGlobal(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];  // Directly read from global memory
    }
}

// =========================================================
//  Kernel using SHARED MEMORY
// =========================================================
__global__ void addShared(const float *A, const float *B, float *C, int n) {
    __shared__ float sA[BLOCK_SIZE];  // Shared memory array
    __shared__ float sB[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Load from global memory into shared memory
        sA[threadIdx.x] = A[idx];
        sB[threadIdx.x] = B[idx];
    }

    __syncthreads();  // Make sure all threads have loaded their data

    if (idx < n) {
        // Compute using shared memory (faster!)
        C[idx] = sA[threadIdx.x] + sB[threadIdx.x];
    }
}

// =========================================================
// MAIN FUNCTION
// =========================================================
int main() {
    int size = N * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // Allocate host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Initialize input arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    float timeGlobal, timeShared;

    // ---------------- GLOBAL MEMORY ----------------
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    addGlobal<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeGlobal, start, stop);

    // ---------------- SHARED MEMORY ----------------
    cudaEventRecord(start);

    addShared<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeShared, start, stop);

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print results
    printf("Global Memory Kernel Time: %.3f ms\n", timeGlobal);
    printf("Shared Memory Kernel Time: %.3f ms\n", timeShared);
    printf("Speedup: %.2fx\n", timeGlobal / timeShared);

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
