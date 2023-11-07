/*
 *  Copyright (c) 2018-2023, Carnegie Mellon University
 *  See LICENSE for details
 */
/***************************************************************************
 * SPL Matrix                                                              *
 *                                                                         *
 * Computes matrix that corresponds to SPL generated routine               *
 ***************************************************************************/

#include <limits.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "common_macros.h"

#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

#ifndef ROWS
#error ROWS must be defined
#endif
#ifndef COLUMNS
#error COLUMNS must be defined
#endif

#ifndef NZERO
#define NZERO (1.0/(double)-INFINITY)
#endif

DEVICE_FFT_DOUBLEREAL  *Input, *Output;
DEVICE_FFT_DOUBLEREAL  *dev_in, *dev_out;

void initialize(int argc, char **argv) {

	// In many case ROWS & COLUMNS are equal; however, when they are not it is
	// important to use the correct one when allocating memory for the in/out
	// buffers.  The *input* buffer should be dimensioned by COLUMNS, while the
	// *output* buffer should be dimensioned by ROWS
	
	Input =  (DEVICE_FFT_DOUBLEREAL*) calloc(sizeof(DEVICE_FFT_DOUBLEREAL), COLUMNS );
	Output = (DEVICE_FFT_DOUBLEREAL*) calloc(sizeof(DEVICE_FFT_DOUBLEREAL), ROWS );

	DEVICE_MALLOC     ( &dev_in,  sizeof(DEVICE_FFT_DOUBLEREAL) * COLUMNS );
	DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );
	DEVICE_MALLOC     ( &dev_out, sizeof(DEVICE_FFT_DOUBLEREAL) * ROWS );
	DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );

	INITFUNC();
}

void finalize() {
	free (Output);
	free (Input);
	DEVICE_FREE     (dev_out);
	DEVICE_FREE     (dev_in);
}

void compute_vector()
{
	int indx;
	double nzero = NZERO;
	printf("[ ");

	DEVICE_MEM_COPY ( dev_in, Input, sizeof(DEVICE_FFT_DOUBLEREAL) * COLUMNS, MEM_COPY_HOST_TO_DEVICE);
	DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );
	
	// set dev_out to negative zero to catch holes transform
	for (indx = 0; indx < ROWS; indx++) {
		Output[indx] = nzero;
	}
	DEVICE_MEM_COPY(dev_out, Output, sizeof(DEVICE_FFT_DOUBLEREAL) * ROWS, MEM_COPY_HOST_TO_DEVICE);
	DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );
		
	// set Output to -Inf to catch incomplete copies
	for (indx = 0; indx < ROWS; indx++) {
		Output[indx] = (double)-INFINITY;
	}

	FUNC(dev_out, dev_in);
	DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );
	DEVICE_SYNCHRONIZE();

	DEVICE_MEM_COPY ( Output, dev_out, sizeof(DEVICE_FFT_DOUBLEREAL) * ROWS, MEM_COPY_DEVICE_TO_HOST);
	DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );

	for (indx = 0; indx < ROWS; indx++) {
		if (indx != 0) {
			if ((indx % 10) == 0) {
				printf("\n");
			}
			printf(", ");
		}
		printf("FloatString(\"%.18g\")", Output[indx]);
	}
	printf("];\n");
}



int main(int argc, char** argv) {

	initialize(argc, argv);

	int tlen = sizeof(testvector) / sizeof(testvector[0]);
	
	for (int i = 0; i < MIN(tlen, COLUMNS); i++) {
		Input[i] = (DEVICE_FFT_DOUBLEREAL)testvector[i];
	}
	
	compute_vector();
	finalize();
	return EXIT_SUCCESS;
}
