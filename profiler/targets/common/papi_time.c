/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */

#include <papi.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>


#ifndef DATATYPE
#define DATATYPE double
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


// minimum number of total usecs for the main timing loop
#define REQUIRED_USECS         1E5

// minimum number of times to call measured function in main timing loop
#define MIN_NUM_RUNS 1
#define MAX_NUM_RUNS 100000

// number of times to run the main timing loop
#define NUM_MAIN_LOOPS 2

DATATYPE * Input;
DATATYPE * Output;


double perform_timing() {
	long_long start_usecs, end_usecs;
	long_long count, min_count;
	int run, i;
	int num_runs = MIN_NUM_RUNS;

	// make sure function runs enough times to get a minimum total usec count
	while (num_runs <= MAX_NUM_RUNS)
	{
		start_usecs = PAPI_get_real_usec();
		for (run = 0; run < num_runs; ++run) {
			RUN_FUNC;
		}
		end_usecs = PAPI_get_real_usec();

		if ((end_usecs - start_usecs) >= REQUIRED_USECS) break;

		num_runs *= 10;
	}

	/* start of measurement */

	min_count = 0;
	for (i = 0; i < NUM_MAIN_LOOPS; i++)
	{
		start_usecs = PAPI_get_real_usec();
		for (run = 0; run < num_runs; ++run) {
			RUN_FUNC;
		}
		end_usecs = PAPI_get_real_usec();

		count = end_usecs - start_usecs;

		if ((min_count <= 0) || (count < min_count)) {
			min_count = count;
		}
	}
	return ((double)min_count / num_runs);
}


int main(int argc, char** argv) {
	double usecs;

	Input  = (DATATYPE *) MEMALIGN(BLOCKSIZE, sizeof(DATATYPE) * COLUMNS);
	Output = (DATATYPE *) MEMALIGN(BLOCKSIZE, sizeof(DATATYPE) * ROWS);

    INITFUNC();

	usecs = perform_timing();

	printf("%e;\n", usecs);

	return EXIT_SUCCESS;
}
