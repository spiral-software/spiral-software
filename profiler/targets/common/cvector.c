/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */


#include <limits.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
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

#ifndef BLOCKSIZE
#define BLOCKSIZE 64
#endif

#ifdef _WIN32
#define MEMALIGN(blksz, memsz) _aligned_malloc((memsz), (blksz))
#else
#define MEMALIGN(blksz, memsz) memalign((blksz), (memsz))
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
	Input  = (DATATYPE *) MEMALIGN(BLOCKSIZE, sizeof(DATATYPE) * COLUMNS);
	Output = (DATATYPE *) MEMALIGN(BLOCKSIZE, sizeof(DATATYPE) * ROWS);

    INITFUNC();
	
	int tlen = sizeof(testvector) / sizeof(testvector[0]);
	
	for (int i = 0; i < COLUMNS; i++) {
        Input[i] = 0;
	}
	for (int i = 0; i < MIN(tlen, COLUMNS); i++) {
        Input[i] = testvector[i];
	}
	
	compute_vector();
	
	return EXIT_SUCCESS;
}
