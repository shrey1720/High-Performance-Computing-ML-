%%writefile addtion_on_device.cu

#include
__global__ void add(int *a, int *b, int *c) {
		*c = *a + *b;
}
int main(void) {
		int a, b, c;	            // host copies of a, b, c
		int *d_a, *d_b, *d_c;	     // device copies of a, b, c
		int size = sizeof(int);

		// Allocate space for device copies of a, b, c
		cudaMalloc((void **)&d_a, size);
		cudaMalloc((void **)&d_b, size);
		cudaMalloc((void **)&d_c, size);

		// Setup input values
		a = 2;
		b = 7;
		// Copy inputs to device
		cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

		// Launch add() kernel on GPU
		add<<<1,1>>>(d_a, d_b, d_c);

		// Copy result back to host
		cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

		// Cleanup
		cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    printf("Value of c after device addtion is %d",c);
		return 0;
	}
-------------------------------------------------------------
!nvcc -arch=sm_75 addtion_on_device.cu -o aod
!./aod
-------------------------------------------------------------
%%writefile P_AOD.cu

#include
#define N 512
__global__ void add(int *a, int *b, int *c) {
		c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}
int main(void) {
	int *a, *b, *c;		// host copies of a, b, c
	int *d_a, *d_b, *d_c;	// device copies of a, b, c
	int size = N * sizeof(int);

	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size);
	b = (int *)malloc(size);
	c = (int *)malloc(size);
  int counter;
  for(counter = 0; counter < N; counter++){
    a[counter] = counter;
    b[counter] = counter*2;
  }
        // Copy inputs to device
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

        // Launch add() kernel on GPU with N blocks
        add<<>>(d_a, d_b, d_c);

        // Copy result back to host
        cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

        // Cleanup
        free(a); free(b); free(c);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        for(counter = 0; counter < 10; counter++){
          printf("\nValue of c[%d] = %d",counter,c[counter]);
          b[counter] = counter*2;
        }
        return 0;
  }
----------------------------------------------------------------
!nvcc -arch=sm_75 P_AOD.cu -o p_aod
!./p_aod
