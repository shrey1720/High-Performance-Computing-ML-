%%writefile mm-cpu.cu

#include <stdio.h>
#include <stdlib.h>

int N = 1500;  // Matrix size

// Function to initialize matrices A and B
void setData(int *A, int *B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (j + i) % 100;
            B[i * N + j] = (j + i) % 100;
        }
    }
}

// Matrix multiplication on CPU
void matMulCPU(int *A, int *B, int *C, int N) {
    for (int i = 0; i < N; i++) {         // Row of A
        for (int j = 0; j < N; j++) {     // Column of B
            int sum = 0;
            for (int k = 0; k < N; k++) { // Dot product
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    int size = N * N * sizeof(int);

    // Allocate memory
    int *A = (int *)malloc(size);
    int *B = (int *)malloc(size);
    int *C = (int *)malloc(size);

    // Initialize matrices
    setData(A, B, N);

    // Perform matrix multiplication
    matMulCPU(A, B, C, N);

    // Print a small part of the result matrix
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            printf("%d ", C[i * N + j]);
        }
        printf("\n");
    }

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
-----------------------------------------------------------------------
!nvcc -arch=sm_75 mm-cpu.cu
%time !./a.out
------------------------------------------------------------------------
%%writefile mm-gpu.cu

#include <stdio.h>
#include <stdlib.h>

int N = 1500;  // Matrix size

// Kernel function: Each block computes one row of matrix C
__global__ void matMulKernel(int *A, int *B, int *C, int N) {
    int row = blockIdx.x; // Each block corresponds to one row

    if (row < N) {
        for (int j = 0; j < N; j++) { // J represents the column index
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + j];
            }
            C[row * N + j] = sum;
        }
    }
}

// Function to initialize matrices A and B
void setData(int *A, int *B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (j + i) % 100;
            B[i * N + j] = (j + i) % 100;
        }
    }
}

int main() {
    int size = N * N * sizeof(int);

    // Allocate host memory
    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);
    int *h_C = (int *)malloc(size);

    // Initialize input matrices
    setData(h_A, h_B, N);

    // Allocate device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel: N blocks, 1 thread per block
    matMulKernel<<<N, 1>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print a small part of the result matrix for verification
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            printf("%d ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
-----------------------------------------------------------------------
!nvcc -arch=sm_75 mm-gpu.cu
%time !./a.out
