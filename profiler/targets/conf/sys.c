/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */
 
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include "sys.h"

void exit_abort(int code);

static int    SYS_VERBOSE = 0;
static char * SYS_PROGNAME = 0;

#ifndef DEBUG
static void (*SYS_EXIT_FUNC)(int)  = exit;
#else
/* Using abort() allows gdb to catch the point where it happened */
static void (*SYS_EXIT_FUNC)(int)  = exit_abort; 
#endif

void exit_abort(int code) {
    abort();
}

EXPORT void sys_set_verbose(int verbose) {
    SYS_VERBOSE = verbose;
}

int sys_get_verbose() {
    return SYS_VERBOSE; 
}

/* Sets the name of the running program (argv[0]) [appears in output of sys_err/sys_fatal] 
    If progname is NULL, sys_err/sys_fatal won't print the program name. */
EXPORT void   sys_set_progname(char * progname) {
    SYS_PROGNAME = progname;
}

/* Returns the name of the running program set with sys_set_progname() */
char  *sys_get_progname() {
    return SYS_PROGNAME; 
}

/* Sets the exit function to be called in sys_fatal */
void   sys_set_exit_func( void (*exit_func)(int) ) {
    SYS_EXIT_FUNC = exit_func;
}

/* Returns the exit function */
void (*sys_get_exit_func())(int) {
    return SYS_EXIT_FUNC;
}

/* if SYS_VERBOSE is set to 1 print the message */
EXPORT void sys_msg(const char *msg, ...) {
    va_list ap;
    if(sys_get_verbose()) {
	if(sys_get_progname()) fprintf(stderr, "%s: ", sys_get_progname());
	va_start(ap, msg);
	vfprintf(stderr, msg, ap);
	va_end(ap);
    }
    fflush(stderr);
}

/* if SYS_VERBOSE is set to 2 print the message */
void sys_msg2(const char *msg, ...) {
    va_list ap;
    if(sys_get_verbose()==2) {
	if(sys_get_progname() && msg && msg[strlen(msg)-1]=='\n') 
	    fprintf(stderr, "%s: ", sys_get_progname());
	va_start(ap, msg);
	vfprintf(stderr, msg, ap);
	va_end(ap);
    }
    fflush(stderr);
}

/* prints err_msg to stderr */
void sys_stderr(const char *err_msg, ...) {
    va_list ap;
    va_start(ap, err_msg);
    vfprintf(stderr, err_msg, ap);
    va_end(ap);
}

/* prints err_msg to stderr, printing progname in front of err_msg */
void sys_err(const char *err_msg, ...) {
    va_list ap;
    if(sys_get_progname()) fprintf(stderr, "%s: ", sys_get_progname());
    va_start(ap, err_msg);
    vfprintf(stderr, err_msg, ap);
    va_end(ap);
}

/* prints err_msg to stderr, and exits with specified code */
EXPORT void sys_fatal(int exit_code, const char *err_msg, ...) {
    va_list ap;
    if(sys_get_progname()) fprintf(stderr, "%s: ", sys_get_progname());
    va_start(ap, err_msg);
    vfprintf(stderr, err_msg, ap);
    va_end(ap);
    SYS_EXIT_FUNC(exit_code);
}

/* if DEBUG is set, print the message to stderr */
void sys_debug(const char *msg, ...) {
#ifdef DEBUG
    va_list ap;
    va_start(ap, msg);
    if(sys_get_progname()) fprintf(stderr, "%s: ", sys_get_progname());
    vfprintf(stderr, msg, ap);
    va_end(ap);
    fflush(stderr);
#endif
}

int sys_exists(const char *fname) {
    FILE * f = fopen(fname, "r");
    if(!f) return 0;
    fclose(f);
    return 1;
}

int sys_exists_m(const char *fname, const char *mode) {
    FILE * f = fopen(fname, mode);
    if(!f) return 0;
    fclose(f);
    return 1;
}

/* -*- Emacs -*- 
Local Variables:
c-basic-offset:4
fill-column:75
End: 
*/
