%%writefile logistic_serial.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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
        }
        weights[0] += lr * grad_w0 / n;
        weights[1] += lr * grad_w1 / n;
        weights[2] += lr * grad_b / n;
    }
}

int main() {
    // Dataset (AND gate)
    double X[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    int y[4] = {0,0,0,1};
    double weights[3] = {0.0, 0.0, 0.0};

    int epochs = 10000;
    double lr = 0.1;

    clock_t start = clock();
    train(X, y, weights, 4, epochs, lr);
    clock_t end = clock();

    printf("Serial Logistic Regression Results:\\n");
    printf("Weights: w0=%.4f, w1=%.4f, bias=%.4f\\n", weights[0], weights[1], weights[2]);
    printf("Training Time (serial): %.6f sec\\n\\n", (double)(end - start)/CLOCKS_PER_SEC);

    for (int i = 0; i < 4; i++) {
        double z = weights[0]*X[i][0] + weights[1]*X[i][1] + weights[2];
        double pred = sigmoid(z);
        printf("Input: (%.0f, %.0f) -> Prediction: %.4f\\n", X[i][0], X[i][1], pred);
    }
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

#define N 4
#define EPOCHS 10000
#define LR 0.1

__device__ double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

__global__ void compute_gradients(double *X, int *y, double *weights, double *grad, int n) {
    int i = threadIdx.x;
    if (i < n) {
        double z = weights[0]*X[2*i] + weights[1]*X[2*i+1] + weights[2];
        double pred = sigmoid(z);
        double error = y[i] - pred;

        atomicAdd(&grad[0], error * X[2*i]);   // grad w0
        atomicAdd(&grad[1], error * X[2*i+1]); // grad w1
        atomicAdd(&grad[2], error);            // grad bias
    }
}

int main() {
    double h_X[2*N] = {0,0, 0,1, 1,0, 1,1};
    int h_y[N] = {0,0,0,1};
    double h_weights[3] = {0,0,0};

    double *d_X, *d_grad, *d_weights;
    int *d_y;
    cudaMalloc((void**)&d_X, 2*N*sizeof(double));
    cudaMalloc((void**)&d_y, N*sizeof(int));
    cudaMalloc((void**)&d_weights, 3*sizeof(double));
    cudaMalloc((void**)&d_grad, 3*sizeof(double));

    cudaMemcpy(d_X, h_X, 2*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, 3*sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double zero[3] = {0,0,0};
        cudaMemcpy(d_grad, zero, 3*sizeof(double), cudaMemcpyHostToDevice);

        compute_gradients<<<1, N>>>(d_X, d_y, d_weights, d_grad, N);

        double h_grad[3];
        cudaMemcpy(h_grad, d_grad, 3*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_weights, d_weights, 3*sizeof(double), cudaMemcpyDeviceToHost);

        h_weights[0] += LR * h_grad[0] / N;
        h_weights[1] += LR * h_grad[1] / N;
        h_weights[2] += LR * h_grad[2] / N;

        cudaMemcpy(d_weights, h_weights, 3*sizeof(double), cudaMemcpyHostToDevice);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_weights, d_weights, 3*sizeof(double), cudaMemcpyDeviceToHost);

    printf("CUDA Logistic Regression Results:\\n");
    printf("Weights: w0=%.4f, w1=%.4f, bias=%.4f\\n", h_weights[0], h_weights[1], h_weights[2]);
    printf("Training Time (parallel CUDA): %.6f ms\\n\\n", milliseconds);

    for (int i = 0; i < N; i++) {
        double z = h_weights[0]*h_X[2*i] + h_weights[1]*h_X[2*i+1] + h_weights[2];
        double pred = 1.0 / (1.0 + exp(-z));
        printf("Input: (%.0f, %.0f) -> Prediction: %.4f\\n", h_X[2*i], h_X[2*i+1], pred);
    }

    cudaFree(d_X); cudaFree(d_y); cudaFree(d_weights); cudaFree(d_grad);
    return 0;
}
----------------------------------------------------------------------------------------------------------
!nvcc logistic_cuda.cu -o logistic_cuda
!./logistic_cuda
----------------------------------------------------------------------------------------------------------
