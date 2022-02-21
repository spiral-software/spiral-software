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

#include <cufft.h>
#include <cufftXt.h>

#include <helper_cuda.h>

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

cufftDoubleReal  *Input, *Output;
cufftDoubleReal  *dev_in, *dev_out;


void initialize(int argc, char **argv) {

	// In many case ROWS & COLUMNS are equal; however, when they are not it is
	// important to use the correct one when allocating memory for the in/out
	// buffers.  The *input* buffer should be dimensioned by COLUMNS, while the
	// *output* buffer should be dimensioned by ROWS
	
	Input =  (cufftDoubleReal*) calloc(sizeof(cufftDoubleReal), COLUMNS );
	Output = (cufftDoubleReal*) calloc(sizeof(cufftDoubleReal), ROWS );

	cudaMalloc     ( &dev_in,  sizeof(cufftDoubleReal) * COLUMNS );
	checkCudaErrors(cudaGetLastError());
	cudaMalloc     ( &dev_out, sizeof(cufftDoubleReal) * ROWS );
	checkCudaErrors(cudaGetLastError());

	INITFUNC();
}

void finalize() {
	free (Output);
	free (Input);
	cudaFree     (dev_out);
	cudaFree     (dev_in);
}

void compute_vector()
{
	int indx;
	double nzero = NZERO;
	printf("[ ");

	cudaMemcpy ( dev_in, Input, sizeof(cufftDoubleReal) * COLUMNS, cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());
	
	// set dev_out to negative zero to catch holes transform
	for (indx = 0; indx < ROWS; indx++) {
		Output[indx] = nzero;
	}
	cudaMemcpy(dev_out, Output, sizeof(cufftDoubleReal) * ROWS, cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());
		
	// set Output to -Inf to catch incomplete copies
	for (indx = 0; indx < ROWS; indx++) {
		Output[indx] = (double)-INFINITY;
	}

	FUNC(dev_out, dev_in);
	checkCudaErrors(cudaGetLastError());
	cudaDeviceSynchronize();

	cudaMemcpy ( Output, dev_out, sizeof(cufftDoubleReal) * ROWS, cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaGetLastError());

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
		Input[i] = (cufftDoubleReal)testvector[i];
	}
	
	compute_vector();
	finalize();
	return EXIT_SUCCESS;
}
