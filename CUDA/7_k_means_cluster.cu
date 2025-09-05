%%writefile kmeans_cpu.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 10    // number of points
#define K 2     // number of clusters
#define DIM 2   // dimension of points
#define MAX_ITERS 10

float points[N][DIM] = {
    {1.0,2.0},{1.5,1.8},{5.0,8.0},{8.0,8.0},{1.0,0.6},
    {9.0,11.0},{8.0,2.0},{10.0,2.0},{9.0,3.0},{8.0,9.0}
};

// initial centroids
float centroids[K][DIM] = {{1.0,2.0},{5.0,8.0}};
int labels[N];

float distance(float *a, float *b) {
    float d=0;
    for(int i=0;i<DIM;i++) d += (a[i]-b[i])*(a[i]-b[i]);
    return sqrt(d);
}

void kmeans() {
    for(int iter=0; iter<MAX_ITERS; iter++) {
        // assign points to nearest centroid
        for(int i=0;i<N;i++) {
            float minDist = 1e9;
            int cluster = 0;
            for(int j=0;j<K;j++) {
                float d = distance(points[i], centroids[j]);
                if(d < minDist) { minDist = d; cluster = j; }
            }
            labels[i] = cluster;
        }

        // recompute centroids
        float newCentroids[K][DIM] = {0};
        int count[K] = {0};
        for(int i=0;i<N;i++) {
            for(int d=0;d<DIM;d++)
                newCentroids[labels[i]][d] += points[i][d];
            count[labels[i]]++;
        }
        for(int j=0;j<K;j++) {
            for(int d=0;d<DIM;d++) {
                if(count[j]>0)
                    centroids[j][d] = newCentroids[j][d]/count[j];
            }
        }
    }
}

int main() {
    kmeans();
    printf("Final centroids:\n");
    for(int j=0;j<K;j++) {
        printf("Cluster %d: (%.2f, %.2f)\n", j, centroids[j][0], centroids[j][1]);
    }
    printf("\nPoint assignments:\n");
    for(int i=0;i<N;i++) {
        printf("(%.1f, %.1f) -> Cluster %d\n", points[i][0], points[i][1], labels[i]);
    }
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

#define N 10    // number of points
#define K 2     // number of clusters
#define DIM 2   // dimension
#define MAX_ITERS 10

__device__ float distance(float *a, float *b) {
    float d=0;
    for(int i=0;i<DIM;i++) d += (a[i]-b[i])*(a[i]-b[i]);
    return sqrtf(d);
}

// kernel: assign each point to nearest centroid
__global__ void assign_clusters(float *points, float *centroids, int *labels) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<N) {
        float minDist = 1e9;
        int cluster = 0;
        for(int j=0;j<K;j++) {
            float d=0;
            for(int d1=0; d1<DIM; d1++) {
                float diff = points[i*DIM+d1]-centroids[j*DIM+d1];
                d += diff*diff;
            }
            if(d < minDist) { minDist = d; cluster = j; }
        }
        labels[i] = cluster;
    }
}

int main() {
    float h_points[N*DIM] = {
        1.0,2.0, 1.5,1.8, 5.0,8.0, 8.0,8.0, 1.0,0.6,
        9.0,11.0, 8.0,2.0, 10.0,2.0, 9.0,3.0, 8.0,9.0
    };
    float h_centroids[K*DIM] = {1.0,2.0, 5.0,8.0};
    int h_labels[N];

    float *d_points, *d_centroids;
    int *d_labels;
    cudaMalloc(&d_points, N*DIM*sizeof(float));
    cudaMalloc(&d_centroids, K*DIM*sizeof(float));
    cudaMalloc(&d_labels, N*sizeof(int));

    cudaMemcpy(d_points, h_points, N*DIM*sizeof(float), cudaMemcpyHostToDevice);

    for(int iter=0; iter<MAX_ITERS; iter++) {
        cudaMemcpy(d_centroids, h_centroids, K*DIM*sizeof(float), cudaMemcpyHostToDevice);
        assign_clusters<<<1,N>>>(d_points, d_centroids, d_labels);
        cudaMemcpy(h_labels, d_labels, N*sizeof(int), cudaMemcpyDeviceToHost);

        // recompute centroids on CPU (simpler for teaching)
        float newCentroids[K*DIM] = {0};
        int count[K] = {0};
        for(int i=0;i<N;i++) {
            for(int d=0;d<DIM;d++)
                newCentroids[h_labels[i]*DIM+d] += h_points[i*DIM+d];
            count[h_labels[i]]++;
        }
        for(int j=0;j<K;j++) {
            for(int d=0;d<DIM;d++) {
                if(count[j]>0)
                    h_centroids[j*DIM+d] = newCentroids[j*DIM+d]/count[j];
            }
        }
    }

    printf("Final centroids:\n");
    for(int j=0;j<K;j++) {
        printf("Cluster %d: (%.2f, %.2f)\n", j, h_centroids[j*DIM], h_centroids[j*DIM+1]);
    }
    printf("\nPoint assignments:\n");
    for(int i=0;i<N;i++) {
        printf("(%.1f, %.1f) -> Cluster %d\n", h_points[i*DIM], h_points[i*DIM+1], h_labels[i]);
    }

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    return 0;
}
-----------------------------------------------------------------------------------------------------
!nvcc kmeans_cuda.cu -o kmeans_cuda
!./kmeans_cuda
