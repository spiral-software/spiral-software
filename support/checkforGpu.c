/*
 **  Copyright (c) 2018-2021, Carnegie Mellon University
 **  See LICENSE for details
 */

#include <stdio.h>

int main ( int argc, char* argv[] )
{
	// dummy program to report no CUDA devices -- built when GPU / nvcc is not available
	printf("No GPU devices available, exit abnormally\n");
	return (-1);
}
