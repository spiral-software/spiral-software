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

	int x, y;
	printf("[ ");
	for (x = 0; x < COLUMNS; x++) {
		vector_basis(Input, x);
		FUNC(Output->data, Input->data);
		if (x != 0) {
			printf(",\n  [ ");
		}
		else {
			printf("[ ");
		}
		for (y = 0; y < ROWS; y++) {
			if (y != 0) {
				printf(", ");
			}
			t->fprint_gap(t, stdout, NTH(Output, y));
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
