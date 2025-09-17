#include <stdio.h>
#include <omp.h>

#define ARR_SIZE 10
#define THREADS 5

void main() {
    int j, tid;

    printf("=== Static Scheduling (chunk size = 2) ===\n");
    #pragma omp parallel private(tid) num_threads(THREADS)
    {
        tid = omp_get_thread_num();

        #pragma omp for schedule(static, 2)
        for (j = 0; j < ARR_SIZE; j++) {
            printf("Thread %d handles iteration %d\n", tid, j);
        }
    }

    printf("\n=== Dynamic Scheduling (chunk size = 2) ===\n");
    #pragma omp parallel private(tid) num_threads(THREADS)
    {
        tid = omp_get_thread_num();

        #pragma omp for schedule(dynamic, 2)
        for (j = 0; j < ARR_SIZE; j++) {
            printf("Thread %d handles iteration %d\n", tid, j);
        }
    }

    return;
}
