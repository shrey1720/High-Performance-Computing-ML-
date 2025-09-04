#include <omp.h>
#include <stdio.h>

int tid ;

#pragma omp threadprivate( tid )

void main()
{
int numt ;

#pragma omp parallel default( shared )
{
tid = omp_get_thread_num() ;
if ( tid == 0 )
{
for(int j = 0 ; j < 10000000 ; j++ ) ;
numt = omp_get_num_threads() ;
}
}

#pragma omp parallel default( shared )
printf( "Hello World from thread %d of %d.\n", tid, numt ) ;

}
