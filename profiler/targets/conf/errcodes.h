/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */
 
#ifndef ERRCODES_H_INCLUDED
#define ERRCODES_H_INCLUDED

enum {
    EXIT_FILE_NOT_FOUND, 
    EXIT_CANT_WRITE_FILE,
    EXIT_CANT_CREATE_TMPFILE,
    EXIT_SPL_COMPILE_FAILED,
    EXIT_TARGET_COMPILE_FAILED,
    EXIT_LINK_FAILED,
    EXIT_BAD_TARGET,
    EXIT_BAD_DIMENSION,
    EXIT_CMDLINE,
    EXIT_MEM  /**< Exit code used when memory can't be allocated (xmalloc/xrealloc/xstrdup) */
};

#endif

/* -*- Emacs -*- 
Local Variables:
c-basic-offset:4
fill-column:75
End: 
*/
