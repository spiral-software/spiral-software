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

cufftDoubleReal  *Input, *Output;
cufftDoubleReal  *dev_in, *dev_out;


void initialize(int argc, char **argv) {

	// In many case ROWS & COLUMNS are equal; however, when they are not it is
	// important to use the correct one when allocating memory for the in/out
	// buffers.  The *input* buffer should be dimensioned by COLUMNS, while the
	// *output* buffer should be dimensioned by ROWS
	
	cudaMallocHost ( &Input,  sizeof(cufftDoubleReal) * COLUMNS );
	checkCudaErrors(cudaGetLastError());
	cudaMallocHost ( &Output, sizeof(cufftDoubleReal) * ROWS );
	checkCudaErrors(cudaGetLastError());

	cudaMalloc     ( &dev_in,  sizeof(cufftDoubleReal) * COLUMNS );
	checkCudaErrors(cudaGetLastError());
	cudaMalloc     ( &dev_out, sizeof(cufftDoubleReal) * ROWS );
	checkCudaErrors(cudaGetLastError());

	INITFUNC();
}

void finalize() {
	cudaFreeHost (Output);
	cudaFreeHost (Input);
	cudaFree     (dev_out);
	cudaFree     (dev_in);
}

void compute_vector()
{
	int indx;
	printf("[ ");

	cudaMemcpy ( dev_in, Input, sizeof(cufftDoubleReal) * COLUMNS, cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());

	FUNC(dev_out, dev_in);
	checkCudaErrors(cudaGetLastError());
	cudaDeviceSynchronize();

	cudaMemcpy ( Output, dev_out, sizeof(cufftDoubleReal) * ROWS, cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaGetLastError());

	for (indx = 0; indx < ROWS; indx++) {
		if (indx != 0) {
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
