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
	Input =  (DATATYPE*) calloc(sizeof(DATATYPE), COLUMNS );
	Output = (DATATYPE*) calloc(sizeof(DATATYPE), ROWS );

	DEVICE_MALLOC     ( &dev_in,  sizeof(DATATYPE) * COLUMNS );
	DEVICE_MALLOC     ( &dev_out, sizeof(DATATYPE) * ROWS );

	INITFUNC();
}

void finalize() {
	free (Output);
	free (Input);
	DEVICE_FREE     (dev_out);
	DEVICE_FREE     (dev_in);
}

void set_value_in_vector(DATATYPE *arr, int elem)
{
	// Zero array and put '1' in the location indicated by element
	int idx;
	for (idx = 0; idx < COLUMNS; idx++)
		arr[idx] = (idx == elem) ? (DATATYPE)1.0 : (DATATYPE)0.0;

	return;
}

void compute_matrix()
{
	int x, y, indx, counter;
	int start_col, end_col, start_row, end_row;
	DATATYPE nzero = NZERO;

#ifdef CMATRIX_UPPER_ROW
	start_row = CMATRIX_UPPER_ROW - 1;
#else
	start_row = 0;
#endif
#ifdef CMATRIX_UPPER_COL
	start_col = CMATRIX_UPPER_COL - 1;
#else
	start_col = 0;
#endif	
#ifdef CMATRIX_LOWER_ROW
	end_row = CMATRIX_LOWER_ROW;
#else
	end_row = ROWS;
#endif
#ifdef CMATRIX_LOWER_COL
	end_col = CMATRIX_LOWER_COL;
#else
	end_col = COLUMNS;
#endif	
	
	printf("[ ");
	for (x = start_col; x < end_col; x++) {
		set_value_in_vector(Input, x);
		DEVICE_MEM_COPY (dev_in, Input, sizeof(DATATYPE) * COLUMNS, MEM_COPY_HOST_TO_DEVICE);
		
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
		DEVICE_MEM_COPY ( Output, dev_out, sizeof(DATATYPE) * ROWS, MEM_COPY_DEVICE_TO_HOST);
		
		if (x != start_col) {
			printf(",\n  [ ");
		}
		else {
			printf("[ ");
		}
		counter = 0;
		for (y = start_row; y < end_row; y++) {
			if (counter != 0) {
				if ((counter % 10) == 0) {
					printf("\n");
				}
				printf(", ");
			}
			printf(DATAFORMATSTRING, Output[y]);
			counter++;
		}
		printf(" ]");
	}
	printf("\n];\n");
}


int main(int argc, char** argv) {
	initialize(argc, argv);
	compute_matrix();
	finalize();
	return EXIT_SUCCESS;
}
