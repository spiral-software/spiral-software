/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */

#include <limits.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include "rdtsc.h"


// minimum number of total CPU cycles for the main timing loop
#define REQUIRED_CYCLES		1E8

// minimum number of times to call measured function in main timing loop
#define MIN_NUM_RUNS 100

// number of times to run the main timing loop
#define NUM_MAIN_LOOPS 5

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


DATATYPE * Input;
DATATYPE * Output;


double perform_timing() {
	tsc_counter a,b;
	int run, i;
	double tm;
	myInt64 count, total_cycles;
	int num_runs = MIN_NUM_RUNS;

	/* warm up, according to Intel manual */
	CPUID(); RDTSC(a); CPUID(); RDTSC(b);
	/* warm up */
	CPUID(); RDTSC(a); CPUID(); RDTSC(b);
	/* warm up */
	CPUID(); RDTSC(a); CPUID(); RDTSC(b);

	// make sure function runs enough times to get a minimum total cycle count
	while (1)
	{
		RDTSC(a);
		for (run = 0; run < num_runs; ++run) {
			RUN_FUNC;
		}
		RDTSC(b);
		tm = (double)COUNTER_DIFF_SIMPLE(b, a);

		if (tm >= REQUIRED_CYCLES) break;

		num_runs *= 2;
	}

	/* start of measurement */
	
	total_cycles = 0;
	for (i = 0; i < NUM_MAIN_LOOPS; i++)
	{
		RDTSC(a);
		for (run = 0; run < num_runs; ++run) {
			RUN_FUNC;
		}
		RDTSC(b);

		count = COUNTER_DIFF_SIMPLE(b, a);
		if ((total_cycles <= 0) || (count < total_cycles)) {
			total_cycles = count;
		}
	}
	return ((double) total_cycles / num_runs);
}


int main(int argc, char** argv) {
	double cycles;

	Input  = (DATATYPE *) MEMALIGN(BLOCKSIZE, sizeof(DATATYPE) * COLUMNS);
	Output = (DATATYPE *) MEMALIGN(BLOCKSIZE, sizeof(DATATYPE) * ROWS);

    INITFUNC();

	cycles = perform_timing();

	printf("%e;\n", cycles);

	return EXIT_SUCCESS;
}
