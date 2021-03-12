/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */

#include <stdlib.h>

#ifndef MALLOC

#define MM_ALIGN_LEN 64

#ifdef _MSC_VER
#include <malloc.h>
#define _MALLOC(a) _aligned_malloc(a, MM_ALIGN_LEN)
#define _FREE(a)   _aligned_free(a)
#else
#if defined(_WIN32) || defined(_WIN64)
#define _MALLOC(a) _mm_malloc(a, MM_ALIGN_LEN)
#define _FREE(a)   _mm_free(a)
#else
#include <malloc.h>
#define _MALLOC(a) memalign(MM_ALIGN_LEN, a)
#define _FREE(a)   free(a)
#endif
#endif

#define MALLOC(a) calloc_spiral(a)

__inline void *calloc_spiral(int size) {
	char * ptr;
	int i;

	ptr = (char *)_MALLOC(size);
	for (i=0; i<size; i++)
		ptr[i] = 0x00;

	return (void *) ptr;
}

#endif
