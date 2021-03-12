/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */

#include <limits.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "rdtsc.h"
#include "vector.h"
#include "errcodes.h"
#include "opt_macros.h"

#if (ALLOCATE_MEMORY)
#define RUN_FUNC FUNC(out, in)
#else
#define RUN_FUNC FUNC()
#endif


// minimum number of total cycles for the main timing loop
#define REQUIRED_CYCLES		1E8

// minimum number of times to call measured function in main timing loop
#define MIN_NUM_RUNS 100

// number of times to run the main timing loop
#define NUM_MAIN_LOOPS 5


vector_t * Input;
vector_t * Output;
vector_t * OutputCopy;



void initialize(int argc, char **argv) {
    unsigned long rows, cols, page, outsz;
    scalar_type_t *typ;

    sys_set_progname(argv[0]);
    srand(time(0));

    typ = scalar_find_type(DATATYPE);
	if (typ == NULL) {
		sys_fatal(EXIT_CMDLINE, "Error: datatype " DATATYPE " not found");
	}

    page = PAGESIZE;
      
#if (ALLOCATE_MEMORY)
    rows = ROWS;
    cols = COLUMNS;
#else
    rows = 1;
    cols = 1;
#endif
    outsz = 2*rows + 1 + (page / typ->size);
    Output     = vector_create_random(typ, outsz);
    OutputCopy = vector_create       (typ, outsz);
    vector_copy(OutputCopy, Output); 
    Input  = vector_create_zero  (typ, cols + 1 + (page / typ->size));

    INITFUNC();
}


double perform_timing() {
	tsc_counter a,b;
	int run, i;
	double tm;
	myInt64 count, min_count;
	int num_runs = MIN_NUM_RUNS;

#if (ALLOCATE_MEMORY) 
	static void *in, *out;
	out = Output->data;
	in = Input->data;
#endif

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
	
	min_count = 0;
	for (i = 0; i < NUM_MAIN_LOOPS; i++)
	{
		RDTSC(a);
		for (run = 0; run < num_runs; ++run) {
			RUN_FUNC;
		}
		RDTSC(b);

		count = COUNTER_DIFF_SIMPLE(b, a);
		if ((min_count <= 0) || (count < min_count)) {
			min_count = count;
		}
	}
	return ((double) min_count / num_runs);
}


int main(int argc, char** argv) {
	double cycles;
#if (ALLOCATE_MEMORY) 
	void *out, *in;
#endif

	initialize(argc, argv);

#if (ALLOCATE_MEMORY) 
	out = Output->data;
	in = Input->data;
	Output->data = out;
#endif

	cycles = perform_timing();

	printf("%e;\n", cycles);

	return EXIT_SUCCESS;
}
