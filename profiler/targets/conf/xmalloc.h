/* Modified by Yevgen Voronenko (ysv22@drexel.edu). 
               Franz Franchetti (franz.franchetti@tuwien.ac.at) 
  $Id: xmalloc.h 8438 2009-06-02 14:02:14Z yvoronen $ */
/*
 * Author: Tatu Ylonen <ylo@cs.hut.fi>
 * Copyright (c) 1995 Tatu Ylonen <ylo@cs.hut.fi>, Espoo, Finland
 *                    All rights reserved
 * Created: Mon Mar 20 22:09:17 1995 ylo
 *
 * Versions of malloc and friends that check their results, and never return
 * failure (they call fatal if they encounter an error).
 *
 * As far as I am concerned, the code I have written for this software
 * can be used freely for any purpose.  Any derived versions of this
 * software must be clearly marked as such, and if the derived work is
 * incompatible with the protocol description in the RFC file, it must be
 * called by a name other than "ssh" or "Secure Shell".
 */

/* RCSID("$OpenBSD: xmalloc.h,v 1.7 2001/06/26 17:27:25 markus Exp $"); */

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
