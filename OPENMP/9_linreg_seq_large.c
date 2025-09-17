#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100000
#define ALPHA 0.01
#define ITERATIONS 10000

void generate_data(double *X, double *Y) {
    for (int i = 0; i < N; i++) {
        X[i] = (double)i / N;
        Y[i] = 2 * X[i] + 3 ;
    }
}

void linear_regression(double *X, double *Y, double *m, double *b) {
    for (int iter = 0; iter < ITERATIONS; iter++) {
        double dm = 0, db = 0;
        for (int i = 0; i < N; i++) {
            double y_pred = (*m) * X[i] + (*b);
            double error = Y[i] - y_pred;
            dm += (-2.0 * X[i] * error) / N;
            db += (-2.0 * error) / N;
        }
        *m -= ALPHA * dm;
        *b -= ALPHA * db;
    }
}

int main() {
    double *X = (double *)malloc(N * sizeof(double));
    double *Y = (double *)malloc(N * sizeof(double));
    generate_data(X, Y);
    double m = 0.0, b = 0.0;
    clock_t start = clock();
    linear_regression(X, Y, &m, &b);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Final values after training:\n");
    printf("m = %f, b = %f\n", m, b);
    printf("Execution Time: %f seconds\n", time_spent);
    free(X);
    free(Y);
    return 0;
}
