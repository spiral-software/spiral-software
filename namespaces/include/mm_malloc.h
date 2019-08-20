
#include <stdlib.h>

#ifndef MALLOC

#ifdef _MSC_VER
#include <malloc.h>
#define _MALLOC(a) _aligned_malloc(a, 16)
#else
#if defined(_WIN32) || defined(_WIN64)
#define _MALLOC(a) _mm_malloc(a, 16)
#else
#include <malloc.h>
#define _MALLOC(a) memalign(16, a)
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
