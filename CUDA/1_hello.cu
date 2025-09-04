!nvidia-smi
-------------------------------------------------------------------------
%%writefile hello.cu

#include
__global__ void hello(void)
{
    printf("GPU: Hello!\n");
}
int main(int argc,char **argv)
{
    printf("CPU: Hello!\n");
    hello<<<2,10>>>();
    cudaDeviceReset();
    return 0;
}
--------------------------------------------------------------------------
!nvcc -arch=sm_75 hello.cu -o hello
!./hello
