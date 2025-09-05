#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 100000
#define ALPHA 0.01
#define ITERATIONS 10000

void generate_data(double *X, double *Y) {
    for (int i = 0; i < N; i++) {
        X[i] = (double)i / N;
        Y[i] = 2 * X[i] + 3 ;
    }
}

void parallel_linear_regression(double *X, double *Y, double *w, double *b, int thread_count) {
    for (int iter = 0; iter < ITERATIONS; iter++) {
        double dw = 0, db = 0;
        #pragma omp parallel num_threads(thread_count)
        {
            double dw_private = 0, db_private = 0;
            #pragma omp for nowait
            for (int i = 0; i < N; i++) {
                double y_pred = (*w) * X[i] + (*b);
                double error = Y[i] - y_pred;
                dw_private += (-2.0 * X[i] * error) / N;
                db_private += (-2.0 * error) / N;
            }
            #pragma omp critical
            {
                dw += dw_private;
                db += db_private;
            }
        }
        *w -= ALPHA * dw;
        *b -= ALPHA * db;
    }
}

int main() {
    double *X = (double *)malloc(N * sizeof(double));
    double *Y = (double *)malloc(N * sizeof(double));
    generate_data(X, Y);
    double w_seq = 0.0, b_seq = 0.0;
    double w_par = 0.0, b_par = 0.0;
    clock_t start_seq = clock();
    parallel_linear_regression(X, Y, &w_seq, &b_seq, 1);
    clock_t end_seq = clock();
    double time_seq = (double)(end_seq - start_seq) / CLOCKS_PER_SEC;
    double start_par = omp_get_wtime();
    parallel_linear_regression(X, Y, &w_par, &b_par, 4);
    double end_par = omp_get_wtime();
    double time_par = end_par - start_par;
    printf("\nSequential Model Results:\n");
    printf("w = %f, b = %f\n", w_seq, b_seq);
    printf("Execution Time: %f seconds\n", time_seq);
    printf("\nParallel Model Results:\n");
    printf("w = %f, b = %f\n", w_par, b_par);
    printf("Execution Time: %f seconds\n", time_par);
    printf("\nSpeedup: %f times\n", time_seq / time_par);
    free(X);
    free(Y);
}
