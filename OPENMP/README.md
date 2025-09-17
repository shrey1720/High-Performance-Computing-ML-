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

Run Command :
```bash 
    gcc -fopenmp filename.c -o filename && ./filename && ./filename
```

- Hello minimal :
```bash 
  - `gcc -fopenmp 1_hello_minimal.c -o hello_minimal && ./hello_minimal`
```
- Hello with IDs:
```bash 
  - `gcc -fopenmp 2_hello_tids.c -o hello_tids && ./hello_tids`
```
- Race Condition :
```bash 
  - `gcc -fopenmp 3_hello_shared_race.c -o hello_shared && ./hello_shared`
```
- Private/Shared:
```bash 
  - `gcc -fopenmp 4_hello_private_shared.c -o hello_private_shared && ./hello_private_shared`
```
- Threadprivate:
```bash 
  - `gcc -fopenmp 5_threadprivate.c -o 5_threadprivate && ./5_threadprivate`
```
- Barrier:
```bash 
  - `gcc -fopenmp 6_threadprivate_barrier.c -o 6_threadprivate_barrier && ./6_threadprivate_barrier`
```
- Critical vs Atomic:
```bash 
  - `gcc -fopenmp 7_critical_vs_atomic.c -o critical_vs_atomic && ./critical_vs_atomic`   
```
- Worksharing for loop:
```bash 
  - `gcc -fopenmp 8_for_schedule.c -o for_schedule && ./for_schedule`
```
- Linear Regression (sequential small):
```bash 
  - `gcc 9_linreg_seq_small.c -o linreg_seq_small && ./linreg_seq_small`
```
- Linear Regression (sequential large):
```bash 
  - `gcc 10_linreg_seq_large.c -o linreg_seq_large && ./linreg_seq_large`
```
- Linear Regression (OpenMP):
```bash 
  - `gcc -fopenmp 11_linreg_openmp.c -o linreg_openmp && ./linreg_openmp`
``` 
## Build Sobel (C++)

- Linux/macOS (pkg-config):
  - `g++ -fopenmp sobel.cpp -o sobel \`pkg-config --cflags --libs opencv4\``
- Windows (MSVC):
  - `cl /openmp sobel.cpp /I<opencv_include> /link /LIBPATH:<opencv_lib> opencv_world4xx.lib`

## Notes

- Output ordering of printed lines in parallel examples is non-deterministic.
- For reductions, prefer `reduction(+:sum)` over `critical`/`atomic` when possible.
- Adjust the number of threads via `OMP_NUM_THREADS` environment variable or runtime API.
