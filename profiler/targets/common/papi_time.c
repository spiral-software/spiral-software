/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */

#include <papi.h>
#include <stdlib.h>
#include <stdio.h>
#include "vector.h"
#include "errcodes.h"
#include "opt_macros.h"

#if (ALLOCATE_MEMORY)
#define RUN_FUNC FUNC(out, in)
#else
#define RUN_FUNC FUNC()
#endif


// minimum number of total usecs for the main timing loop
#define REQUIRED_USECS         1E5

// minimum number of times to call measured function in main timing loop
#define MIN_NUM_RUNS 1
#define MAX_NUM_RUNS 100000

// number of times to run the main timing loop
#define NUM_MAIN_LOOPS 2


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
	outsz = 2 * rows + 1 + (page / typ->size);
	Output = vector_create_random(typ, outsz);
	OutputCopy = vector_create(typ, outsz);
	vector_copy(OutputCopy, Output);
	Input = vector_create_zero(typ, cols + 1 + (page / typ->size));

	INITFUNC();
}

double perform_timing() {
	long_long start_usecs, end_usecs;
	long_long count, min_count;
	int run, i;
	int num_runs = MIN_NUM_RUNS;

#if (ALLOCATE_MEMORY)
	static void *in, *out;
	out = Output->data;
	in = Input->data;
#endif

	// make sure function runs enough times to get a minimum total usec count
	while (num_runs <= MAX_NUM_RUNS)
	{
		start_usecs = PAPI_get_real_usec();
		for (run = 0; run < num_runs; ++run) {
			RUN_FUNC;
		}
		end_usecs = PAPI_get_real_usec();
		//fprintf(stderr, "Runs: %d, usecs: %lld\n", num_runs,
		//	end_usecs - start_usecs);

		if ((end_usecs - start_usecs) >= REQUIRED_USECS) break;

		num_runs *= 10;
	}

	/* start of measurement */

	min_count = 0;
	for (i = 0; i < NUM_MAIN_LOOPS; i++)
	{
	  //fprintf(stderr, "\nOUTER LOOP %d\n", i+1);

		start_usecs = PAPI_get_real_usec();
		for (run = 0; run < num_runs; ++run) {
			RUN_FUNC;
		}
		end_usecs = PAPI_get_real_usec();

		//fprintf(stderr, "start %lld, end %lld\n", start_usecs, end_usecs);

		count = end_usecs - start_usecs;

		//fprintf(stderr, "count %lld\n", count);

		if ((min_count <= 0) || (count < min_count)) {
			min_count = count;
		}

		//fprintf(stderr, "min_count %lld, num_runs %d\n", min_count, num_runs);


	}
	return ((double)min_count / num_runs);
}


int main(int argc, char** argv) {
	double usecs;
#if (ALLOCATE_MEMORY)
	void *out, *in;
#endif

	initialize(argc, argv);

#if (ALLOCATE_MEMORY)
	out = Output->data;
	in = Input->data;
	Output->data = out;
#endif

	usecs = perform_timing();

	printf("%e;\n", usecs);

	return EXIT_SUCCESS;
}
