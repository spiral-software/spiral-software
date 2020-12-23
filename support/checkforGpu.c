#include <stdio.h>

int main ( int argc, char* argv[] )
{
	// dummy program to report no CUDA devices -- built when GPU / nvcc is not available
	printf("No GPU devices available, exit abnormally\n");
	return (-1);
}
