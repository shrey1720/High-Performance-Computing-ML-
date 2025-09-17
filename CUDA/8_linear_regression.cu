%%writefile linear_serial.c
#include <stdio.h>
#include <stdlib.h>

#define N 100      // smaller dataset
#define EPOCHS 5000
#define LR 0.05    // learning rate

// Train linear regression
void train(double X[][2], double y[], double w[], int n) {
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double dw0 = 0, dw1 = 0, db = 0;

        for (int i = 0; i < n; i++) {
            double y_pred = w[0]*X[i][0] + w[1]*X[i][1] + w[2];
            double error = y[i] - y_pred;

            dw0 += -2 * error * X[i][0];
            dw1 += -2 * error * X[i][1];
            db  += -2 * error;
        }

        w[0] -= LR * dw0 / n;
        w[1] -= LR * dw1 / n;
        w[2] -= LR * db  / n;
    }
}

int main() {
    double X[N][2], y[N];
    for (int i = 0; i < N; i++) {
        X[i][0] = (double)(rand() % 100) / 100.0;
        X[i][1] = (double)(rand() % 100) / 100.0;
        y[i] = 3*X[i][0] + 2*X[i][1] + 5; // perfect linear data
    }

    double w[3] = {0, 0, 0}; // w0, w1, b
    train(X, y, w, N);

    printf("Learned Weights:\n");
    printf("w0=%.3f, w1=%.3f, bias=%.3f\n", w[0], w[1], w[2]);
    return 0;
}

-------------------------------------------------------------------------------------------------------
!gcc linear_serial.c -o linear_serial -lm
!./linear_serial
-------------------------------------------------------------------------------------------------------
%%writefile linear_cuda.cu
#include <stdio.h>
#include <stdlib.h>

#define N 100
#define EPOCHS 5000
#define LR 0.05
#define THREADS 128

__global__ void compute_gradients(double *X, double *y, double *w, double *grad, int n) {
    __shared__ double partial_dw0[THREADS];
    __shared__ double partial_dw1[THREADS];
    __shared__ double partial_db[THREADS];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    double dw0 = 0, dw1 = 0, db = 0;

    if (idx < n) {
        double x0 = X[2*idx];
        double x1 = X[2*idx + 1];
        double y_pred = w[0]*x0 + w[1]*x1 + w[2];
        double error = y[idx] - y_pred;

        dw0 = -2 * error * x0;
        dw1 = -2 * error * x1;
        db  = -2 * error;
    }

    partial_dw0[tid] = dw0;
    partial_dw1[tid] = dw1;
    partial_db[tid]  = db;
    __syncthreads();

    // Parallel reduction (sum within block)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_dw0[tid] += partial_dw0[tid + stride];
            partial_dw1[tid] += partial_dw1[tid + stride];
            partial_db[tid]  += partial_db[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&grad[0], partial_dw0[0]);
        atomicAdd(&grad[1], partial_dw1[0]);
        atomicAdd(&grad[2], partial_db[0]);
    }
}

int main() {
    double h_X[N][2], h_y[N];
    double h_w[3] = {0, 0, 0};

    for (int i = 0; i < N; i++) {
        h_X[i][0] = (double)(rand() % 100) / 100.0;
        h_X[i][1] = (double)(rand() % 100) / 100.0;
        h_y[i] = 3*h_X[i][0] + 2*h_X[i][1] + 5;
    }

    double *d_X, *d_y, *d_w, *d_grad;
    cudaMalloc(&d_X, N * 2 * sizeof(double));
    cudaMalloc(&d_y, N * sizeof(double));
    cudaMalloc(&d_w, 3 * sizeof(double));
    cudaMalloc(&d_grad, 3 * sizeof(double));

    cudaMemcpy(d_X, h_X, N * 2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, 3 * sizeof(double), cudaMemcpyHostToDevice);

    int blocks = (N + THREADS - 1) / THREADS;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double h_grad[3] = {0, 0, 0};
        cudaMemcpy(d_grad, h_grad, 3 * sizeof(double), cudaMemcpyHostToDevice);

        compute_gradients<<<blocks, THREADS>>>(d_X, d_y, d_w, d_grad, N);
        cudaDeviceSynchronize();

        cudaMemcpy(h_grad, d_grad, 3 * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_w, d_w, 3 * sizeof(double), cudaMemcpyDeviceToHost);

        h_w[0] -= LR * h_grad[0] / N;
        h_w[1] -= LR * h_grad[1] / N;
        h_w[2] -= LR * h_grad[2] / N;

        cudaMemcpy(d_w, h_w, 3 * sizeof(double), cudaMemcpyHostToDevice);
    }

    printf("Learned Weights:\n");
    printf("w0=%.3f, w1=%.3f, bias=%.3f\n", h_w[0], h_w[1], h_w[2]);

    cudaFree(d_X); cudaFree(d_y); cudaFree(d_w); cudaFree(d_grad);
    return 0;
}

----------------------------------------------------------------------------------------------------------
!nvcc linear_cuda.cu -o linear_cuda
!./linear_cuda
----------------------------------------------------------------------------------------------------------
