/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
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

#include "sys.h"
#include "conf.h"
#include "vector.h"
#include "opt_macros.h"
#include "xmalloc.h"
#include "vector_def.h" /* data_type */

#ifndef ROWS
#error ROWS must be defined
#endif
#ifndef COLUMNS
#error COLUMNS must be defined
#endif

#ifndef NZERO
#define NZERO (1.0/(double)-INFINITY)
#endif

vector_t * Input;
vector_t * Output;


void initialize(int argc, char **argv) {
	scalar_type_t *t = scalar_find_type(DATATYPE);

	Output = vector_create_zero(t, ROWS);
	Input = vector_create_zero(t, COLUMNS);

	INITFUNC();
}

void finalize() {
	vector_destroy(Output);
	vector_destroy(Input);
}

void compute_matrix()
{
	scalar_type_t *t = scalar_find_type(DATATYPE);
	DATATYPE_NO_QUOTES nz = NZERO;
	int x, y, counter;
	int start_col, end_col, start_row, end_row;
	
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
	
	for (x = 0; x < ROWS; x++) {
		SET(t, NTH(Output, x), &nz);
	}
	printf("[ ");
	for (x = start_col; x < end_col; x++) {
		vector_basis(Input, x);
		FUNC(Output->data, Input->data);
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
			t->fprint_gap(t, stdout, NTH(Output, y));
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
