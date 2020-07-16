/*
 *  Copyright (c) 2018-2020, Carnegie Mellon University
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

#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

#ifndef ROWS
#error ROWS must be defined
#endif
#ifndef COLUMNS
#error COLUMNS must be defined
#endif

extern void INITFUNC(void);
extern void FUNC(void *Y, void *X);

vector_t * Input;
vector_t * Output;


void initialize(int argc, char **argv) {
	scalar_type_t *t = scalar_find_type(DATATYPE);

	Output = vector_create_zero(t, ROWS);
	Input  = vector_create_zero(t, ROWS);

	INITFUNC();
}

void finalize() {
	vector_destroy(Output);
	vector_destroy(Input);
}

void compute_vector(scalar_type_t *t)
{
	int x;
	printf("[ ");
	FUNC(Output->data, Input->data);
	for (x = 0; x < ROWS; x++) {
		if (x != 0) {
			printf(", ");
			}
		t->fprint_gap(t, stdout, NTH(Output, x));
	}
	printf("];\n");
}



int main(int argc, char** argv) {
	initialize(argc, argv);
	
	scalar_type_t *t = scalar_find_type(DATATYPE);
	int tlen = sizeof(testvector) / sizeof(testvector[0]);
	
	for (int i = 0; i < MIN(tlen, ROWS); i++) {
        SET(t, NTH(Input, i), &testvector[i]);
	}
	
	compute_vector(t);
	finalize();
	return EXIT_SUCCESS;
}
