/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */
 
#include <stdlib.h>
#include <string.h>
#if !defined(__APPLE__)
#include <malloc.h> /* memalign() */
#endif
#include <assert.h>
#include "xmalloc.h"
#include "exceptions.h"

static xmem_allocator_t x_default_allocator = { malloc, realloc, free };
static xmem_allocator_t * x_allocator = &x_default_allocator;

/** Sets memory allocator to be used by xmalloc/xfree/.. family */
void xset_mem_allocator(xmem_allocator_t *allocator) 
{
    assert(allocator!=NULL);
/*	FIXME!!!! Franz: "sorry, but bget does break on Windows. Who knows what the error really is?!" */
#if !defined(__WIN32__) || !defined(WIN32)
	x_allocator = allocator;
#endif
}

/** Returns the current memory allocator used by xmalloc/xfree/.. family */
xmem_allocator_t * xget_mem_allocator(void) {
    return x_allocator;
}

void * xmalloc(size_t size)
{
	void *ptr;

	if (size == 0)
		Throw exc(ERR_MEM_ALLOC_ZERO);

	ptr = (void*) x_allocator->malloc(size);

	if (ptr == NULL)
		Throw exc(ERR_MEM_ALLOC, size);

	return ptr;
}


void *
xrealloc(void *ptr, size_t new_size)
{
	void *new_ptr;

	if (new_size == 0)
	        Throw exc(ERR_MEM_REALLOC_ZERO);
	if (ptr == NULL)
		new_ptr = (void*) xmalloc(new_size);
	else
		new_ptr = (void*) x_allocator->realloc(ptr, new_size);
	if (new_ptr == NULL)
		Throw exc(ERR_MEM_REALLOC, new_size);
	return new_ptr;
}

void
xfree(void *ptr)
{
	if (ptr == NULL)
		Throw exc(ERR_MEM_FREE_NULL);

	x_allocator->free(ptr);
}

size_t xmalloc_strlcpy(char *dst, const char *src, size_t siz);

char *
xstrdup(const char *str)
{
	size_t len;
	char *cp;

    if(str == NULL)
        return NULL;

	len = strlen(str) + 1;
	cp = (char*) xmalloc(len);
	xmalloc_strlcpy(cp, str, len);
	return cp;
}

/* The following is a portion of a file strlcpy.
   Modified by Yevgen Voronenko (ysv22@drexel.edu).

   To remove the need for detection of whether strlcpy is
   available, I renamed this function to xmalloc_strlcpy.
   xmalloc() famil of functions uses this, no matter whether
   strlcpy is available in the C library 
*/

/*      $OpenBSD: strlcpy.c,v 1.5 2001/05/13 15:40:16 deraadt Exp $     */

/*
 * Copyright (c) 1998 Todd C. Miller <Todd.Miller@courtesan.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL
 * THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Copy src to string dst of size siz.  At most siz-1 characters
 * will be copied.  Always NUL terminates (unless siz == 0).
 * Returns strlen(src); if retval >= siz, truncation occurred.
 */
size_t xmalloc_strlcpy(char *dst, const char *src, size_t siz) {
        register char *d = dst;
        register const char *s = src;
        register size_t n = siz;

        /* Copy as many bytes as will fit */
        if (n != 0 && --n != 0) {
                do {
                        if ((*d++ = *s++) == 0)
                                break;
                } while (--n != 0);
        }

        /* Not enough room in dst, add NUL and traverse rest of src */
        if (n == 0) {
                if (siz != 0)
                        *d = '\0';              /* NUL-terminate dst */
                while (*s++)
                        ;
        }

        return(s - src - 1);    /* count does not include NUL */
}


/* Allocate an aligned block of memory  with alignment = MEM_ALIGNMENT*/
void * xaligned_malloc(size_t size) {
        return xaligned_malloc2(size, MEM_ALIGNMENT);
}

/* Allocate an aligned block of memory  with specified alignment */
void    *xaligned_malloc2(size_t size, int alignment) {
	void *ptr;
	if (size == 0)
		Throw exc(ERR_MEM_ALLOC_ZERO);
#if defined(__WIN32__) || defined(WIN32)
	ptr = (void*) _mm_malloc(size, alignment);
#elif defined(__APPLE__)
    ptr = (void*) malloc(size);
#else
	ptr = (void*) memalign(alignment, size);
#endif
	if (ptr == NULL)
		Throw exc(ERR_MEM_ALLOC, size);

	return ptr;
}


void xaligned_free(void *ptr)
{
	if (ptr == NULL)
		Throw exc(ERR_MEM_FREE_NULL);
#if defined(__WIN32__) || defined(WIN32)
	_mm_free(ptr);
#else
	free(ptr);
#endif
}

/* -*- Emacs -*- 
Local Variables:
c-basic-offset:8
fill-column:75
End: 
*/
