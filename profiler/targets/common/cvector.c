/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */


#include <limits.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

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

#ifndef DATATYPE
#define DATATYPE double
#endif

#ifndef DATAFORMATSTRING
#define DATAFORMATSTRING "FloatString(\"%.18g\")"
#endif

#ifndef RUN_FUNC
#define RUN_FUNC FUNC(Output, Input)
#endif


DATATYPE * Input;
DATATYPE * Output;

void compute_vector()
{
	int x;
	DATATYPE nz = NZERO;
	for (x = 0; x < ROWS; x++) {
		Output[x] = nz;
	}
	printf("[ ");
	RUN_FUNC;
	for (x = 0; x < ROWS; x++) {
		if (x != 0) {
			if ((x % 10) == 0) {
				printf("\n");
			}
			printf(", ");
		}
		printf(DATAFORMATSTRING, Output[x]);
	}
	printf("];\n");
}



int main(int argc, char** argv) {
	Input  = (DATATYPE *) calloc(sizeof(DATATYPE), COLUMNS );
	Output = (DATATYPE *) calloc(sizeof(DATATYPE), ROWS );

    INITFUNC();
	
	int tlen = sizeof(testvector) / sizeof(testvector[0]);
	
	for (int i = 0; i < MIN(tlen, COLUMNS); i++) {
        Input[i] = testvector[i];
	}
	
	compute_vector();
	
	free(Input);
	free(Output);
	
	return EXIT_SUCCESS;
}
