/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */
 
#ifndef XMALLOC_H
#define XMALLOC_H
#include <stddef.h> /* size_t */
#define MEM_ALIGNMENT 128

#ifdef SWIG
typedef unsigned long size_t
#endif

#ifndef SWIG
/** Defines what allocation functions will be invoked by xmalloc/xfree/.. */
typedef struct xmem_allocator {
    void* (*malloc) (size_t);
    void* (*realloc) (void*, size_t);
    void  (*free) (void*);
} xmem_allocator_t;
#endif

/** Sets memory allocator to be used by xmalloc/xfree/.. family */
void xset_mem_allocator(xmem_allocator_t *);

/** Returns the current memory allocator used by xmalloc/xfree/.. family */
xmem_allocator_t * xget_mem_allocator(void);


/** Allocates memory with malloc, if allocation fails - aborts the program */
void	*xmalloc(size_t);
/** Allocates memory with realloc, if allocations fails - aborts the program */
void	*xrealloc(void *, size_t);
/** Must be used with xmalloc */
void     xfree(void *);
/** Duplicates the string, if memory allocation fails - aborts the program */
char 	*xstrdup(const char *);

/** Allocate an aligned block of memory  with alignment = MEM_ALIGNMENT*/
void    *xaligned_malloc(size_t);
/** Allocate an aligned block of memory  with specified alignment */
void    *xaligned_malloc2(size_t, int alignment);
/** Deallocate memory allocated with xaligned_malloc */
void     xaligned_free(void *);

#endif				/* XMALLOC_H */

/* -*- Emacs -*- 
Local Variables:
c-basic-offset:4
fill-column:75
End: 
*/
