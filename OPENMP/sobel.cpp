#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <omp.h>

using namespace cv;
using namespace std;

void apply_sobel_serial(unsigned char* input, unsigned char* output, int rows, int cols) {
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            int sumX = 0, sumY = 0;
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    int pixel = input[(i + ki) * cols + (j + kj)];
                    sumX += pixel * Gx[ki + 1][kj + 1];
                    sumY += pixel * Gy[ki + 1][kj + 1];
                }
            }
            int magnitude = sqrt(sumX * sumX + sumY * sumY);
            output[i * cols + j] = (magnitude > 255) ? 255 : magnitude;
        }
    }
}

void apply_sobel_openmp(unsigned char* input, unsigned char* output, int rows, int cols) {
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            int sumX = 0, sumY = 0;
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    int pixel = input[(i + ki) * cols + (j + kj)];
                    sumX += pixel * Gx[ki + 1][kj + 1];
                    sumY += pixel * Gy[ki + 1][kj + 1];
                }
            }
            int magnitude = sqrt(sumX * sumX + sumY * sumY);
            output[i * cols + j] = (magnitude > 255) ? 255 : magnitude;
        }
    }
}

int main() {
    Mat gray_img = imread("image.jpg", IMREAD_GRAYSCALE);
    if (gray_img.empty()) {
        cout << "Error: Could not open image." << endl;
        return -1;
    }
    int rows = gray_img.rows;
    int cols = gray_img.cols;
    unsigned char* input = gray_img.data;
    Mat output_serial(rows, cols, CV_8UC1);
    double start_serial = omp_get_wtime();
    apply_sobel_serial(input, output_serial.data, rows, cols);
    double end_serial = omp_get_wtime();
    double time_serial = (end_serial - start_serial) * 1000;
    imwrite("sobel_serial.jpg", output_serial);
    Mat output_omp(rows, cols, CV_8UC1);
    double start_omp = omp_get_wtime();
    apply_sobel_openmp(input, output_omp.data, rows, cols);
    double end_omp = omp_get_wtime();
    double time_omp = (end_omp - start_omp) * 1000;
    imwrite("sobel_openmp.jpg", output_omp);
    cout << "Serial Execution Time  : " << time_serial << " ms" << endl;
    cout << "OpenMP Execution Time  : " << time_omp << " ms" << endl;
    cout << "Speedup (Serial / OpenMP) : " << (time_serial / time_omp) << "x" << endl;
    return 0;
}
