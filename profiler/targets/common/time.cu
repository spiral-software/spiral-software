/*
 *  Copyright (c) 2018-2023, Carnegie Mellon University
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

#include "common_macros.h"

#ifndef DESTROYFUNC
#define DESTROYFUNC destroy_sub
#endif

// ===  Test SPIRAL ===

void setup_spiral_test()
{
    DEVICE_CHECK_ERROR ( DEVICE_SET_CACHE_CONFIG ( DEVICE_CACHE_PREFER_SHARED ) );
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
	DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );
}


int main(int argc, char** argv)
{
	DEVICE_FFT_DOUBLEREAL  *out, *in;
	DEVICE_FFT_DOUBLEREAL  *dev_out, *dev_in;
	DEVICE_EVENT_T          begin, end;

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
	
	DEVICE_EVENT_CREATE ( &begin );
	DEVICE_EVENT_CREATE ( &end );
	in =  (DEVICE_FFT_DOUBLEREAL*) calloc(sizeof(DEVICE_FFT_DOUBLEREAL), COLUMNS );
	out = (DEVICE_FFT_DOUBLEREAL*) calloc(sizeof(DEVICE_FFT_DOUBLEREAL), ROWS );
	DEVICE_MALLOC      ( &dev_in,  sizeof(DEVICE_FFT_DOUBLEREAL) * COLUMNS );
	DEVICE_MALLOC      ( &dev_out, sizeof(DEVICE_FFT_DOUBLEREAL) * ROWS );

	for (int i = 0; i < /* ROWS */ COLUMNS; i++)
		in[i] = i;

	setup_spiral_test();
	DEVICE_MEM_COPY ( dev_in, in,   sizeof(DEVICE_FFT_DOUBLEREAL) * COLUMNS, MEM_COPY_HOST_TO_DEVICE);
	DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );
	
	DEVICE_CHECK_ERROR ( DEVICE_EVENT_RECORD ( begin ) );
	int iters = 1 * 100 ;
	for (int i = 0; i < iters; i++ ) {
		test_spiral(dev_in, dev_out);
	}
	DEVICE_CHECK_ERROR ( DEVICE_EVENT_RECORD ( end ) );
	DEVICE_SYNCHRONIZE();
	DEVICE_MEM_COPY ( out, dev_out, sizeof(DEVICE_FFT_DOUBLEREAL) * ROWS, MEM_COPY_DEVICE_TO_HOST);
	DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );

	teardown_spiral_test();

	float milli = 0.0;
	DEVICE_CHECK_ERROR ( DEVICE_EVENT_ELAPSED_TIME ( &milli, begin, end ) );
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
	
	free  ( in );
	free  ( out );
	DEVICE_FREE      ( dev_in );
	DEVICE_FREE      ( dev_out );
	
	return EXIT_SUCCESS;
}
						 
