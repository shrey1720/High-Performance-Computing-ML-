%%writefile knn_cpu.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 50000   // training samples
#define K 50      // nearest neighbors
#define NUM_TESTS 100  // number of test points to predict

float train_data[N][2];
int train_labels[N];

// Compute distance with heavy artificial delay
float distance(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    float d = dx*dx + dy*dy;

    // Strong artificial delay to slow down CPU
    for (volatile int k = 0; k < 5000; k++) {
        d += 0.0000001f * d;  // floating-point math to avoid compiler optimization
    }

    return sqrt(d);
}

int knn_predict(float x, float y) {
    float dist[N];
    int idx[N];

    // compute distances
    for (int i = 0; i < N; i++) {
        dist[i] = distance(x, y, train_data[i][0], train_data[i][1]);
        idx[i] = i;
    }

    // selection sort for K nearest
    for (int i = 0; i < K; i++) {
        for (int j = i + 1; j < N; j++) {
            if (dist[j] < dist[i]) {
                float tmp = dist[i]; dist[i] = dist[j]; dist[j] = tmp;
                int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
            }
        }
    }

    // majority vote
    int count0 = 0, count1 = 0;
    for (int i = 0; i < K; i++) {
        if (train_labels[idx[i]] == 0) count0++;
        else count1++;
    }

    return (count1 > count0) ? 1 : 0;
}

int main() {
    srand(42); // reproducibility

    // generate random training data
    for (int i = 0; i < N; i++) {
        train_data[i][0] = (float)(rand() % 1000) / 10.0f;
        train_data[i][1] = (float)(rand() % 1000) / 10.0f;
        train_labels[i] = rand() % 2;
    }

    clock_t start = clock();

    for (int t = 0; t < NUM_TESTS; t++) {
        float tx = (float)(rand() % 1000) / 10.0f;
        float ty = (float)(rand() % 1000) / 10.0f;
        int pred = knn_predict(tx, ty);
        if (t == NUM_TESTS-1) // just print last prediction
            printf("CPU Prediction for (%.1f, %.1f): Class %d\n", tx, ty, pred);
    }

    clock_t end = clock();
    double total_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("CPU Total Time for %d predictions: %.6f sec\n", NUM_TESTS, total_time);
    printf("CPU Average Time per prediction: %.6f sec\n", total_time / NUM_TESTS);

    return 0;
}
------------------------------------------------------------------------------------------
!gcc knn_cpu.c -o knn_cpu -lm
!./knn_cpu
------------------------------------------------------------------------------------------
%%writefile knn_cuda.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <time.h>

#define N 50000
#define K 50
#define NUM_TESTS 100

__device__ float distance(float x1, float y1, float x2, float y2) {
    return sqrtf((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

// GPU kernel to compute distances
__global__ void compute_distances(float *train_x, float *train_y, float tx, float ty, float *dist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        dist[i] = distance(tx, ty, train_x[i], train_y[i]);
    }
}

int main() {
    srand(42);

    float *h_train_x = (float*)malloc(N * sizeof(float));
    float *h_train_y = (float*)malloc(N * sizeof(float));
    int   *h_labels  = (int*)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        h_train_x[i] = (float)(rand() % 1000) / 10.0f;
        h_train_y[i] = (float)(rand() % 1000) / 10.0f;
        h_labels[i]  = rand() % 2;
    }

    float *d_train_x, *d_train_y, *d_dist;
    cudaMalloc(&d_train_x, N * sizeof(float));
    cudaMalloc(&d_train_y, N * sizeof(float));
    cudaMalloc(&d_dist, N * sizeof(float));

    cudaMemcpy(d_train_x, h_train_x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_y, h_train_y, N*sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *h_dist = (float*)malloc(N * sizeof(float));
    int *idx = (int*)malloc(N * sizeof(int));

    clock_t start = clock();

    for (int t = 0; t < NUM_TESTS; t++) {
        float tx = (float)(rand() % 1000) / 10.0f;
        float ty = (float)(rand() % 1000) / 10.0f;

        compute_distances<<<blocks, threadsPerBlock>>>(d_train_x, d_train_y, tx, ty, d_dist);
        cudaDeviceSynchronize();
        cudaMemcpy(h_dist, d_dist, N*sizeof(float), cudaMemcpyDeviceToHost);

        for (int i=0;i<N;i++) idx[i] = i;

        // selection sort on CPU
        for (int i = 0; i < K; i++) {
            for (int j = i+1; j < N; j++) {
                if (h_dist[j] < h_dist[i]) {
                    float tmp = h_dist[i]; h_dist[i] = h_dist[j]; h_dist[j] = tmp;
                    int t_idx = idx[i]; idx[i] = idx[j]; idx[j] = t_idx;
                }
            }
        }

        int count0=0, count1=0;
        for (int i=0;i<K;i++) {
            if (h_labels[idx[i]] == 0) count0++;
            else count1++;
        }
        int pred = (count1 > count0) ? 1 : 0;
        if (t == NUM_TESTS-1)
            printf("GPU Prediction for (%.1f, %.1f): Class %d\n", tx, ty, pred);
    }

    clock_t end = clock();
    double total_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("GPU Total Time for %d predictions: %.6f sec\n", NUM_TESTS, total_time);
    printf("GPU Average Time per prediction: %.6f sec\n", total_time / NUM_TESTS);

    cudaFree(d_train_x);
    cudaFree(d_train_y);
    cudaFree(d_dist);
    free(h_train_x); free(h_train_y); free(h_labels);
    free(h_dist); free(idx);
    return 0;
}
-------------------------------------------------------------------------------------------
!nvcc knn_cuda.cu -o knn_cuda
!./knn_cuda
