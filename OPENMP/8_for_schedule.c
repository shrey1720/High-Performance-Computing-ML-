#include<stdio.h>
#include<omp.h>
#define ARR_SIZE 10
int main()
{
int j, tid ;
int a[ARR_SIZE] ;
#pragma omp parallel private( tid)
{
tid = omp_get_thread_num() ;
// schedule(static, 2) or schedule(dynamic, 2)
#pragma omp for
for( j = 0 ; j < ARR_SIZE ; j++ )
{
a[j] = 1 ;
printf( "Thread %d, iteration %d\n", tid, j ) ;
}
}
}
