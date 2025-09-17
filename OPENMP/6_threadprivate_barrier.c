#include <omp.h>
#include <stdio.h>

void main() {
    int numt;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        if (tid == 0)
        {
            for (int j = 0; j < 10000000; j++);
            numt = omp_get_num_threads();
        }

        #pragma omp barrier
        printf("Hello World from thread %d of %d.\n", tid, numt);
    }

    return;
}
