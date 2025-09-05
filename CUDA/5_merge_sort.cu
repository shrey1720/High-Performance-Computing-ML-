%%writefile cpu_merge_sort.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to merge two halves
void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    int *L = (int *)malloc(n1 * sizeof(int));
    int *R = (int *)malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    free(L);
    free(R);
}

// Recursive Merge Sort
void mergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

int main() {
    int n = 1 << 20; // 1M elements
    int *arr = (int *)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++)
        arr[i] = rand() % 10000;

    clock_t start = clock();
    mergeSort(arr, 0, n - 1);
    clock_t end = clock();

    printf("CPU Merge Sort Time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    free(arr);
    return 0;
}
---------------------------------------------------------------------------------------------------
!gcc cpu_merge_sort.c -o cpu_sort
!./cpu_sort
---------------------------------------------------------------------------------------------------
%%writefile cuda_merge_sort.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// GPU kernel for merging
__global__ void mergeKernel(int *arr, int *temp, int width, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * (2 * width);
    if (start >= n) return;

    int mid = min(start + width, n);
    int end = min(start + 2 * width, n);

    int i = start, j = mid, k = start;
    while (i < mid && j < end) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }
    while (i < mid) temp[k++] = arr[i++];
    while (j < end) temp[k++] = arr[j++];
}

void mergeSortGPU(int *arr, int n) {
    int *d_arr, *d_temp;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMalloc((void**)&d_temp, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize;

    for (int width = 1; width < n; width *= 2) {
        gridSize = (n + (2 * width * blockSize) - 1) / (2 * width * blockSize);
        mergeKernel<<<gridSize, blockSize>>>(d_arr, d_temp, width, n);
        cudaDeviceSynchronize();

        int *temp = d_arr;
        d_arr = d_temp;
        d_temp = temp;
    }

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_temp);
}

int main() {
    int n = 1 << 20; // 1M elements
    int *arr = (int *)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++)
        arr[i] = rand() % 10000;

    float gpuTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mergeSortGPU(arr, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    printf("GPU Merge Sort Time: %f ms\n", gpuTime);

    free(arr);
    return 0;
}
------------------------------------------------------------------------------------------------
!nvcc cuda_merge_sort.cu -o gpu_sort
!./gpu_sort
