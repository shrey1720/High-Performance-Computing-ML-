#include <stdio.h>
#include <stdlib.h>

#define N 5
#define ALPHA 0.01
#define ITERATIONS 10000

void linear_regression(double X[], double Y[], double *w, double *b) {
    for (int iter = 0; iter < ITERATIONS; iter++) {
        double dw = 0, db = 0;
        for (int i = 0; i < N; i++) {
            double y_pred = (*w) * X[i] + (*b);
            double error = Y[i] - y_pred;
            dw += -2.0 * X[i] * error;
            db += -2.0 * error;
        }
        dw /= N;
        db /= N;
        *w -= ALPHA * dw;
        *b -= ALPHA * db;
    }
}

void main() {
    double X[N] = {1, 2, 3, 4, 5};
    double Y[N] = {2, 4, 6, 8, 10};
    double w = 0.0, b = 0.0;
    linear_regression(X, Y, &w, &b);
    printf("Final values after training:\n");
    printf("w = %f, b = %f\n", w, b);
}
