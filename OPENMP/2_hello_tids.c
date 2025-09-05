#include <omp.h>
#include <stdio.h>
int main()
{
#pragma omp parallel
{
int numt = omp_get_num_threads() ;
int tid = omp_get_thread_num() ;
printf( "Hello World from thread %d of %d.\n", tid, numt ) ;
}
}
