/*
 *  Copyright (c) 2018-2025, Carnegie Mellon University
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
#define NZERO (1.0/(DATATYPE)-INFINITY)
#endif

DATATYPE  *Input, *Output;
DATATYPE  *dev_in, *dev_out;

void initialize(int argc, char **argv) {

	// In many case ROWS & COLUMNS are equal; however, when they are not it is
	// important to use the correct one when allocating memory for the in/out
	// buffers.  The *input* buffer should be dimensioned by COLUMNS, while the
	// *output* buffer should be dimensioned by ROWS
	
	Input =  (DATATYPE*) calloc(sizeof(DATATYPE), COLUMNS );
	Output = (DATATYPE*) calloc(sizeof(DATATYPE), ROWS );

	DEVICE_MALLOC     ( &dev_in,  sizeof(DATATYPE) * COLUMNS );
	DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );
	DEVICE_MALLOC     ( &dev_out, sizeof(DATATYPE) * ROWS );
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
	DATATYPE nzero = NZERO;
	printf("[ ");

	DEVICE_MEM_COPY ( dev_in, Input, sizeof(DATATYPE) * COLUMNS, MEM_COPY_HOST_TO_DEVICE);
	DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );
	
	// set dev_out to negative zero to catch holes transform
	for (indx = 0; indx < ROWS; indx++) {
		Output[indx] = nzero;
	}
	DEVICE_MEM_COPY(dev_out, Output, sizeof(DATATYPE) * ROWS, MEM_COPY_HOST_TO_DEVICE);
	DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );
		
	// set Output to -Inf to catch incomplete copies
	for (indx = 0; indx < ROWS; indx++) {
		Output[indx] = (DATATYPE)-INFINITY;
	}

	FUNC(dev_out, dev_in);
	DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );
	DEVICE_SYNCHRONIZE();

	DEVICE_MEM_COPY ( Output, dev_out, sizeof(DATATYPE) * ROWS, MEM_COPY_DEVICE_TO_HOST);
	DEVICE_CHECK_ERROR ( DEVICE_GET_LAST_ERROR () );

	for (indx = 0; indx < ROWS; indx++) {
		if (indx != 0) {
			if ((indx % 10) == 0) {
				printf("\n");
			}
			printf(", ");
		}
		printf(DATAFORMATSTRING, Output[indx]);
	}
	printf("];\n");
}



int main(int argc, char** argv) {

	initialize(argc, argv);

	int tlen = sizeof(testvector) / sizeof(testvector[0]);
	
	for (int i = 0; i < MIN(tlen, COLUMNS); i++) {
		Input[i] = (DATATYPE)testvector[i];
	}
	
	compute_vector();
	finalize();
	return EXIT_SUCCESS;
}
