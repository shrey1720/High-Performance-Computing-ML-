%%writefile kmeans_cpu.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 10000   // number of points (large for runtime difference)
#define K 4       // number of clusters
#define DIM 2     // dimension of points
#define MAX_ITERS 15

float points[N][DIM];
float centroids[K][DIM];
int labels[N];

float distance(float *a, float *b) {
    float d = 0;
    for (int i = 0; i < DIM; i++) d += (a[i] - b[i]) * (a[i] - b[i]);

    // Artificial delay to make CPU slower (burn cycles)
    for (volatile int k = 0; k < 200; k++);

    return sqrt(d);
}

void kmeans() {
    for (int iter = 0; iter < MAX_ITERS; iter++) {
        // assign points to nearest centroid
        for (int i = 0; i < N; i++) {
            float minDist = 1e9;
            int cluster = 0;
            for (int j = 0; j < K; j++) {
                float d = distance(points[i], centroids[j]);
                if (d < minDist) { minDist = d; cluster = j; }
            }
            labels[i] = cluster;
        }

        // recompute centroids
        float newCentroids[K][DIM] = {0};
        int count[K] = {0};
        for (int i = 0; i < N; i++) {
            for (int d = 0; d < DIM; d++)
                newCentroids[labels[i]][d] += points[i][d];
            count[labels[i]]++;
        }
        for (int j = 0; j < K; j++) {
            for (int d = 0; d < DIM; d++) {
                if (count[j] > 0)
                    centroids[j][d] = newCentroids[j][d] / count[j];
            }
        }
    }
}

int main() {
    srand(42);
    for (int i = 0; i < N; i++) {
        points[i][0] = (float)(rand() % 1000) / 10.0;
        points[i][1] = (float)(rand() % 1000) / 10.0;
    }
    // Initialize centroids randomly
    for (int j = 0; j < K; j++) {
        centroids[j][0] = points[j][0];
        centroids[j][1] = points[j][1];
    }

    clock_t start = clock();
    kmeans();
    clock_t end = clock();

    printf("Final centroids:\n");
    for (int j = 0; j < K; j++) {
        printf("Cluster %d: (%.2f, %.2f)\n", j, centroids[j][0], centroids[j][1]);
    }
    printf("Execution time (CPU): %.3f sec\n", (double)(end - start) / CLOCKS_PER_SEC);
    return 0;
}
---------------------------------------------------------------------------------------------
!gcc kmeans_cpu.c -o kmeans_cpu -lm
!./kmeans_cpu
---------------------------------------------------------------------------------------------
%%writefile kmeans_cuda.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <time.h>

#define N 10000
#define K 4
#define DIM 2
#define MAX_ITERS 15

__global__ void assign_clusters(float *points, float *centroids, int *labels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float minDist = 1e9;
        int cluster = 0;
        for (int j = 0; j < K; j++) {
            float d = 0;
            for (int d1 = 0; d1 < DIM; d1++) {
                float diff = points[i * DIM + d1] - centroids[j * DIM + d1];
                d += diff * diff;
            }
            if (d < minDist) { minDist = d; cluster = j; }
        }
        labels[i] = cluster;
    }
}

int main() {
    float *h_points = (float*)malloc(N * DIM * sizeof(float));
    float *h_centroids = (float*)malloc(K * DIM * sizeof(float));
    int *h_labels = (int*)malloc(N * sizeof(int));

    srand(42);
    for (int i = 0; i < N; i++) {
        h_points[i*DIM]   = (float)(rand() % 1000) / 10.0;
        h_points[i*DIM+1] = (float)(rand() % 1000) / 10.0;
    }
    for (int j = 0; j < K; j++) {
        h_centroids[j*DIM]   = h_points[j*DIM];
        h_centroids[j*DIM+1] = h_points[j*DIM+1];
    }

    float *d_points, *d_centroids;
    int *d_labels;
    cudaMalloc(&d_points, N * DIM * sizeof(float));
    cudaMalloc(&d_centroids, K * DIM * sizeof(float));
    cudaMalloc(&d_labels, N * sizeof(int));

    cudaMemcpy(d_points, h_points, N * DIM * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    clock_t start = clock();
    for (int iter = 0; iter < MAX_ITERS; iter++) {
        cudaMemcpy(d_centroids, h_centroids, K * DIM * sizeof(float), cudaMemcpyHostToDevice);
        assign_clusters<<<blocks, threadsPerBlock>>>(d_points, d_centroids, d_labels);
        cudaMemcpy(h_labels, d_labels, N * sizeof(int), cudaMemcpyDeviceToHost);

        // recompute centroids on CPU
        float newCentroids[K*DIM] = {0};
        int count[K] = {0};
        for (int i = 0; i < N; i++) {
            for (int d = 0; d < DIM; d++)
                newCentroids[h_labels[i]*DIM+d] += h_points[i*DIM+d];
            count[h_labels[i]]++;
        }
        for (int j = 0; j < K; j++) {
            for (int d = 0; d < DIM; d++) {
                if (count[j] > 0)
                    h_centroids[j*DIM+d] = newCentroids[j*DIM+d] / count[j];
            }
        }
    }
    clock_t end = clock();

    printf("Final centroids:\n");
    for (int j = 0; j < K; j++) {
        printf("Cluster %d: (%.2f, %.2f)\n", j, h_centroids[j*DIM], h_centroids[j*DIM+1]);
    }
    printf("Execution time (GPU): %.3f sec\n", (double)(end - start) / CLOCKS_PER_SEC);

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    free(h_points); free(h_centroids); free(h_labels);
    return 0;
}
-----------------------------------------------------------------------------------------------------
!nvcc kmeans_cuda.cu -o kmeans_cuda
!./kmeans_cuda
