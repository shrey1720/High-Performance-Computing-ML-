%%writefile logistic_serial.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 5000     // artificially increase dataset size
#define EPOCHS 20000
#define LR 0.1

// Sigmoid
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

// Train logistic regression
void train(double X[][2], int y[], double weights[], int n, int epochs, double lr) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double grad_w0 = 0.0, grad_w1 = 0.0, grad_b = 0.0;

        for (int i = 0; i < n; i++) {
            double z = weights[0]*X[i][0] + weights[1]*X[i][1] + weights[2];
            double pred = sigmoid(z);
            double error = y[i] - pred;

            grad_w0 += error * X[i][0];
            grad_w1 += error * X[i][1];
            grad_b  += error;

            // 🔥 Artificial delay to slow CPU down
            for (volatile int k = 0; k < 200; k++);
        }
        weights[0] += lr * grad_w0 / n;
        weights[1] += lr * grad_w1 / n;
        weights[2] += lr * grad_b / n;
    }
}

int main() {
    // Generate synthetic data (linearly separable)
    double (*X)[2] = malloc(N * sizeof *X);
    int *y = malloc(N * sizeof *y);
    for (int i = 0; i < N; i++) {
        X[i][0] = (double)(rand() % 100) / 100.0;
        X[i][1] = (double)(rand() % 100) / 100.0;
        y[i] = (X[i][0] + X[i][1] > 1.0) ? 1 : 0;  // simple separable condition
    }

    double weights[3] = {0.0, 0.0, 0.0};

    clock_t start = clock();
    train(X, y, weights, N, EPOCHS, LR);
    clock_t end = clock();

    printf("Serial Logistic Regression Results:\n");
    printf("Weights: w0=%.4f, w1=%.4f, bias=%.4f\n", weights[0], weights[1], weights[2]);
    printf("Training Time (serial): %.3f sec\n", (double)(end - start)/CLOCKS_PER_SEC);

    free(X);
    free(y);
    return 0;
}
-------------------------------------------------------------------------------------------------------
!gcc logistic_serial.c -o logistic_serial -lm
!./logistic_serial
-------------------------------------------------------------------------------------------------------
%%writefile logistic_cuda.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define N 5000
#define EPOCHS 20000
#define LR 0.1

__device__ float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

__global__ void compute_gradients(float *X, int *y, float *weights, float *grad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float z = weights[0]*X[2*i] + weights[1]*X[2*i+1] + weights[2];
        float pred = sigmoid(z);
        float error = y[i] - pred;

        atomicAdd(&grad[0], error * X[2*i]);   // grad w0
        atomicAdd(&grad[1], error * X[2*i+1]); // grad w1
        atomicAdd(&grad[2], error);            // grad bias
    }
}

int main() {
    float *h_X = (float*)malloc(2*N*sizeof(float));
    int *h_y = (int*)malloc(N*sizeof(int));
    float h_weights[3] = {0, 0, 0};

    // Generate synthetic dataset
    for (int i = 0; i < N; i++) {
        h_X[2*i]   = (float)(rand() % 100) / 100.0;
        h_X[2*i+1] = (float)(rand() % 100) / 100.0;
        h_y[i] = (h_X[2*i] + h_X[2*i+1] > 1.0f) ? 1 : 0;
    }

    float *d_X, *d_grad, *d_weights;
    int *d_y;
    cudaMalloc((void**)&d_X, 2*N*sizeof(float));
    cudaMalloc((void**)&d_y, N*sizeof(int));
    cudaMalloc((void**)&d_weights, 3*sizeof(float));
    cudaMalloc((void**)&d_grad, 3*sizeof(float));

    cudaMemcpy(d_X, h_X, 2*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, 3*sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float zero[3] = {0,0,0};
        cudaMemcpy(d_grad, zero, 3*sizeof(float), cudaMemcpyHostToDevice);

        compute_gradients<<<blocks, threadsPerBlock>>>(d_X, d_y, d_weights, d_grad, N);

        float h_grad[3];
        cudaMemcpy(h_grad, d_grad, 3*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_weights, d_weights, 3*sizeof(float), cudaMemcpyDeviceToHost);

        h_weights[0] += LR * h_grad[0] / N;
        h_weights[1] += LR * h_grad[1] / N;
        h_weights[2] += LR * h_grad[2] / N;

        cudaMemcpy(d_weights, h_weights, 3*sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_weights, d_weights, 3*sizeof(float), cudaMemcpyDeviceToHost);

    printf("CUDA Logistic Regression Results:\n");
    printf("Weights: w0=%.4f, w1=%.4f, bias=%.4f\n", h_weights[0], h_weights[1], h_weights[2]);
    printf("Training Time (parallel CUDA): %.3f ms\n", milliseconds);

    cudaFree(d_X); cudaFree(d_y); cudaFree(d_weights); cudaFree(d_grad);
    free(h_X); free(h_y);
    return 0;
}
----------------------------------------------------------------------------------------------------------
!nvcc logistic_cuda.cu -o logistic_cuda
!./logistic_cuda
----------------------------------------------------------------------------------------------------------
