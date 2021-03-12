/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef WIN64
#include <winsock.h>			// defines struct timeval -- go figure?
#include <sys/timeb.h>
#endif							// WIN64

#include <cufft.h>
#include <cufftXt.h>

#include <helper_cuda.h>

#ifndef DESTROYFUNC
#define DESTROYFUNC destroy_sub
#endif

// ===  Test SPIRAL ===

void setup_spiral_test()
{
    checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    // printf("Running SPIRAL Hello World CUDA example...\n");

    INITFUNC();
}

void teardown_spiral_test()
{
    DESTROYFUNC();
}

void test_spiral(double* in, double* out)
{

    FUNC(out, in);
    checkCudaErrors(cudaGetLastError());
}


int main(int argc, char** argv)
{
	cufftDoubleReal  *out, *in;
	cufftDoubleReal  *dev_out, *dev_in;
	cudaEvent_t      begin, end;

#ifdef WIN64
    struct timeb     start, finish;
    ftime(&start);
#else
	struct timespec  start, finish;
	clock_gettime(CLOCK_MONOTONIC, &start);
#endif					 // WIN64

	// In many case ROWS & COLUMNS are equal; however, when they are not it is
	// important to use the correct one when allocating memory for the in/out
	// buffers.  The *input* buffer should be dimensioned by COLUMNS, while the
	// *output* buffer should be dimensioned by ROWS
	
	cudaEventCreate ( &begin );
	cudaEventCreate ( &end );
	cudaMallocHost  ( &in,      sizeof(cufftDoubleReal) * COLUMNS );
	cudaMallocHost  ( &out,     sizeof(cufftDoubleReal) * ROWS );
	cudaMalloc      ( &dev_in,  sizeof(cufftDoubleReal) * COLUMNS );
	cudaMalloc      ( &dev_out, sizeof(cufftDoubleReal) * ROWS );

	for (int i = 0; i < /* ROWS */ COLUMNS; i++)
		in[i] = i;

	setup_spiral_test();
	cudaMemcpy ( dev_in, in,   sizeof(cufftDoubleReal) * COLUMNS, cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());
	
	checkCudaErrors( cudaEventRecord(begin) );
	int iters = 1 * 100 ;
	for (int i = 0; i < iters; i++ ) {
		test_spiral(dev_in, dev_out);
	}
	checkCudaErrors( cudaEventRecord(end) );
	cudaDeviceSynchronize();
	cudaMemcpy ( out, dev_out, sizeof(cufftDoubleReal) * ROWS, cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaGetLastError());

	teardown_spiral_test();

	float milli = 0.0;
	checkCudaErrors ( cudaEventElapsedTime ( &milli, begin, end ) );
	printf("%f;\t\t##  SPIRAL GPU kernel execution [ms], averaged over %d iterations ##PICKME##\n", milli / iters, iters );

#ifdef WIN64
    ftime(&finish);
    double elapsed = (1000.0 * (finish.time - start.time)) + (finish.millitm - start.millitm);
	printf("%f;\t\t##  Timing test elapsed time [ms], completed %d iterations ##PICKME##\n", elapsed, iters);
#else
	// time for non-Windows systems
	clock_gettime(CLOCK_MONOTONIC, &finish);
	double elapsed = ( ( (double)finish.tv_sec * 1e9 + (double)finish.tv_nsec) -
					   ( (double)start.tv_sec  * 1e9 + (double)start.tv_nsec ) );
	printf("%f;\t\t##  Timing test elapsed time [ms], completed %d iterations ##PICKME##\n", elapsed * 1e-6 / iters, iters );
#endif // WIN64

	fflush(stdout);
	
	cudaFreeHost  ( in );
	cudaFreeHost  ( out );
	cudaFree      ( dev_in );
	cudaFree      ( dev_out );
	
	return EXIT_SUCCESS;
}
						 
