/*
 **  Copyright (c) 2018-2026, Carnegie Mellon University
 **  See LICENSE for details
 */

#include <stdio.h>
#include <cuda_runtime.h>               // Standard CUDA header

int main ( int argc, char* argv[] )
{
	int deviceCount = 0;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);

	if (err != cudaSuccess) {
		fprintf(stderr, "looking for CUDA devices: %s\n", cudaGetErrorString(err));
		return (-1);
	}

	if (deviceCount == 0) {
		printf("No GPU devices found, exit abnormally\n");
		return (-1);
	}
	
	int device;
	for (device = 0; device < deviceCount; device++) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		printf("Device %d has compute capability %d.%d.\n",
			   device, deviceProp.major, deviceProp.minor);
	}
	return 0;
}
