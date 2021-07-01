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


void compute_matrix()
{
	DATATYPE nz = NZERO;
	int x, y, i, counter;
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
	
	
	printf("[ ");
	for (x = start_col; x < end_col; x++) {
		for(i = 0; i < COLUMNS; i++) {
			Input[i] = 0;
		}
		Input[x] = 1;
		for (i = 0; i < ROWS; i++) {
			Output[i] = nz;
		}
		RUN_FUNC;;
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
	Input  = (DATATYPE *) calloc(sizeof(DATATYPE), COLUMNS );
	Output = (DATATYPE *) calloc(sizeof(DATATYPE), ROWS );

    INITFUNC();
	
	compute_matrix();
	
	free(Input);
	free(Output);
	
	return EXIT_SUCCESS;
}
