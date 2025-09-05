%%writefile knn_cpu.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 8   // number of training samples
#define K 3   // number of nearest neighbors

// training data: 2D points + labels (0 or 1)
float train_data[N][2] = {
    {1.0, 2.0}, {2.0, 3.0}, {3.0, 3.0}, {6.0, 5.0},
    {7.0, 7.0}, {8.0, 6.0}, {1.5, 1.8}, {6.5, 5.5}
};
int train_labels[N] = {0, 0, 0, 1, 1, 1, 0, 1};

// compute distance
float distance(float x1, float y1, float x2, float y2) {
    return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

// simple KNN
int knn_predict(float x, float y) {
    float dist[N];
    int idx[N];

    // compute distances
    for (int i = 0; i < N; i++) {
        dist[i] = distance(x, y, train_data[i][0], train_data[i][1]);
        idx[i] = i;
    }

    // simple selection sort for K nearest
    for (int i = 0; i < K; i++) {
        for (int j = i+1; j < N; j++) {
            if (dist[j] < dist[i]) {
                float tmp = dist[i]; dist[i] = dist[j]; dist[j] = tmp;
                int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
            }
        }
    }

    // majority voting
    int count0 = 0, count1 = 0;
    for (int i = 0; i < K; i++) {
        if (train_labels[idx[i]] == 0) count0++;
        else count1++;
    }

    return (count1 > count0) ? 1 : 0;
}

int main() {
    float test[2] = {5.0, 5.0};
    int pred = knn_predict(test[0], test[1]);
    printf("CPU Prediction for (%.1f, %.1f): Class %d\n", test[0], test[1], pred);
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

#define N 8   // training samples
#define K 3   // nearest neighbors

__device__ float distance(float x1, float y1, float x2, float y2) {
    return sqrtf((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

// GPU kernel to compute distances
__global__ void compute_distances(float *train_x, float *train_y, float tx, float ty, float *dist) {
    int i = threadIdx.x;
    if (i < N) {
        dist[i] = distance(tx, ty, train_x[i], train_y[i]);
    }
}

int main() {
    float h_train_x[N] = {1.0,2.0,3.0,6.0,7.0,8.0,1.5,6.5};
    float h_train_y[N] = {2.0,3.0,3.0,5.0,7.0,6.0,1.8,5.5};
    int   h_labels[N]  = {0,0,0,1,1,1,0,1};

    float tx = 5.0, ty = 5.0;  // test point

    // allocate memory
    float *d_train_x, *d_train_y, *d_dist;
    cudaMalloc(&d_train_x, N * sizeof(float));
    cudaMalloc(&d_train_y, N * sizeof(float));
    cudaMalloc(&d_dist, N * sizeof(float));

    cudaMemcpy(d_train_x, h_train_x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_y, h_train_y, N*sizeof(float), cudaMemcpyHostToDevice);

    // launch kernel
    compute_distances<<<1, N>>>(d_train_x, d_train_y, tx, ty, d_dist);

    float h_dist[N];
    cudaMemcpy(h_dist, d_dist, N*sizeof(float), cudaMemcpyDeviceToHost);

    // now on CPU: sort + majority vote
    int idx[N];
    for (int i=0;i<N;i++) idx[i] = i;

    for (int i = 0; i < K; i++) {
        for (int j = i+1; j < N; j++) {
            if (h_dist[j] < h_dist[i]) {
                float tmp = h_dist[i]; h_dist[i] = h_dist[j]; h_dist[j] = tmp;
                int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
            }
        }
    }

    int count0=0, count1=0;
    for (int i=0;i<K;i++) {
        if (h_labels[idx[i]] == 0) count0++;
        else count1++;
    }
    int pred = (count1 > count0) ? 1 : 0;

    printf("GPU Prediction for (%.1f, %.1f): Class %d\n", tx, ty, pred);

    cudaFree(d_train_x);
    cudaFree(d_train_y);
    cudaFree(d_dist);
    return 0;
}
-------------------------------------------------------------------------------------------
!nvcc knn_cuda.cu -o knn_cuda
!./knn_cuda
