# OpenMP Teaching Repository

This repository contains a curated set of OpenMP examples with clear explanations, build and run instructions, and a JSON index (`lessons_openmp.json`) suitable for lesson delivery.

## Structure

- `lessons_openmp.json`: Structured metadata and code for the lessons
- `examples/openmp/`: Standalone source files for each example
- `LICENSE`: MIT license

## Prerequisites

- GCC or Clang with OpenMP support
  - Linux/macOS: `gcc`/`clang` with `-fopenmp`
  - Windows (MinGW-w64): `gcc` with `-fopenmp`
  - Windows (MSVC): `cl /openmp`
- Optional for Sobel example (C++): OpenCV 4 installed and discoverable via `pkg-config` or MSVC include/lib paths

## Build & Run (GCC/MinGW examples)

Examples below assume you are in the `examples/openmp` directory.

- Hello minimal:
  - `gcc -fopenmp hello_minimal.c -o hello_minimal && ./hello_minimal`
- Hello with IDs:
  - `gcc -fopenmp hello_tids.c -o hello_tids && ./hello_tids`
- Private/Shared:
  - `gcc -fopenmp hello_private_shared.c -o hello_private_shared && ./hello_private_shared`
- Worksharing for loop:
  - `gcc -fopenmp for_schedule.c -o for_schedule && ./for_schedule`
- Critical vs Atomic:
  - `gcc -fopenmp critical_vs_atomic.c -o critical_vs_atomic && ./critical_vs_atomic`
- Linear Regression (sequential small):
  - `gcc linreg_seq_small.c -o linreg_seq_small && ./linreg_seq_small`
- Linear Regression (sequential large):
  - `gcc linreg_seq_large.c -o linreg_seq_large && ./linreg_seq_large`
- Linear Regression (OpenMP):
  - `gcc -fopenmp linreg_openmp.c -o linreg_openmp && ./linreg_openmp`

## Build Sobel (C++)

- Linux/macOS (pkg-config):
  - `g++ -fopenmp sobel.cpp -o sobel \`pkg-config --cflags --libs opencv4\``
- Windows (MSVC):
  - `cl /openmp sobel.cpp /I<opencv_include> /link /LIBPATH:<opencv_lib> opencv_world4xx.lib`

## Notes

- Output ordering of printed lines in parallel examples is non-deterministic.
- For reductions, prefer `reduction(+:sum)` over `critical`/`atomic` when possible.
- Adjust the number of threads via `OMP_NUM_THREADS` environment variable or runtime API.
