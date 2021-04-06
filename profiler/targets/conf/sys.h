/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */
 
#ifndef IO_H_INCLUDED
#define IO_H_INCLUDED

#define EXPORT

/** Determines sys_msg() behavior, 1: print to stderr, 0: do nothing */
EXPORT void   sys_set_verbose(int verbose);

/** Returns a value previously set with sys_set_verbose() */
int    sys_get_verbose(void);

/** Sets the name of the running program (argv[0]) [appears in output
 *  of sys_err/sys_fatal].  If progname is NULL, sys_err/sys_fatal
 *  won't print the program name. 
 */
EXPORT void   sys_set_progname(char * progname);

/** Returns the name of the running program set with sys_set_progname() */
char  *sys_get_progname(void);

/** Sets the exit function to be called in sys_fatal */
void   sys_set_exit_func( void (*exit_func)(int) );

/** Returns the exit function */
void (*sys_get_exit_func(void))(int);

/** Prints to stdout if sys_set_verbose(1) was called, otherwise nothing is done */
EXPORT void   sys_msg(const char *msg, ...);

/** Prints to stdout if sys_set_verbose(2) was called, otherwise nothing is done */
void   sys_msg2(const char *msg, ...);

/** Print to stderr */
void   sys_stderr(const char *err_msg, ...);

/** Print to stderr, printing progname in front of message */
void   sys_err(const char *err_msg, ...);

/** Print to stderr, then exit with specified exit code */
EXPORT void   sys_fatal(int exit_code, const char *err_msg, ...);

/** if DEBUG preprocessor symbol is defined, then print msg to stderr */
void   sys_debug(const char *msg, ...);

/** Returns 1 if fname exists and is readable, 0 otherwise */
int    sys_exists(const char *fname);

/** Returns 1 if fname exists and is openable with in spec. mode, 0 otherwise */
int    sys_exists_m(const char *fname, const char *mode);

#endif

/* -*- Emacs -*- 
Local Variables:
c-basic-offset:4
fill-column:75
End: 
*/
