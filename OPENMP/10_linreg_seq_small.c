#include <stdio.h>
#include <stdlib.h>

#define N 5
#define ALPHA 0.01
#define ITERATIONS 10000

void linear_regression(double X[], double Y[], double *m, double *b) {
    for (int iter = 0; iter < ITERATIONS; iter++) {
        double dm = 0, db = 0;
        for (int i = 0; i < N; i++) {
            double y_pred = (*m) * X[i] + (*b);
            double error = Y[i] - y_pred;
            dm += -2.0 * X[i] * error;
            db += -2.0 * error;
        }
        dm /= N;
        db /= N;
        *m -= ALPHA * dm;
        *b -= ALPHA * db;
    }
}

int main() {
    double X[N] = {1, 2, 3, 4, 5};
    double Y[N] = {2, 4, 6, 8, 10};
    double m = 0.0, b = 0.0;
    linear_regression(X, Y, &m, &b);
    printf("Final values after training:\n");
    printf("m = %f, b = %f\n", m, b);

    return 0;
}
