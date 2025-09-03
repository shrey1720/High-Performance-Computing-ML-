#include <omp.h>
#include <stdio.h>
void main()
{
int numt, tid;
#pragma omp parallel
{
numt = omp_get_num_threads();
tid = omp_get_thread_num();
for(int i=0;i<10000000;i++);
printf( "Hello World from thread %d of %d.\n", tid, numt );
}
}
