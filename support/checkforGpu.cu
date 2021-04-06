/*
 **  Copyright (c) 2018-2021, Carnegie Mellon University
 **  See LICENSE for details
 */

#include <helper_cuda.h>

int main ( int argc, char* argv[] )
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	printLastCudaError("looking for CUDA devices: "); //cudaGetLastError()

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
