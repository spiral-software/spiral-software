/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */
 
/*
  This file defines some useful macros and functions for 
  argc/argv command line option processing.
*/

#ifndef OPT_MACROS_H_INCLUDED
#define OPT_MACROS_H_INCLUDED
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sys.h"

extern char * usage(void);

#define MAX(a,b) ((a)>(b)?(a):(b))

/** Discard leftmost command like argument */
#define SHIFT() { argv++; \
                  argc--; \
                  if(argc < 0) sys_fatal(EXIT_CMDLINE, "test: internal error in parse_options"); \
                }
 
/** Returns 1 if st starts with '-', 0 otherwise */
#define HAS_MINUS(st) (strlen(st)>0 && (st)[0]=='-')

#define GET_ARG_FUNC(var,func) {  var=0; if(argc<2) \
                            sys_fatal(EXIT_CMDLINE, "'%s' option requires a parameter\n", argv[0]);\
                        else { \
                            var = argv[1]; \
                            func(); \
                        }\
                     }

/** Assigns next argument (string) to 'var', var must be of type char *.
    If no next argument is available error is reported via exit_with_error */
#define GET_ARG(var) GET_ARG_FUNC(var,SHIFT)

#define GET_ARG_EAT(var) GET_ARG_FUNC(var,EAT_OPT)



/** Converts next command line argument (string) to integer, and assigns value
    to var, var must be of type int. If no next argument is available or it
    cannot be converted - an error is reported via fatal */
#define GET_INT_ARG(var) { char *tmp=0; \
                           GET_ARG(tmp); \
                           cmdline_parse_int(var,tmp); }

/** Returns 1 if the next command line argument is str */
#define OPT(str) (strcmp(argv[0], (str))==0)

#define ARG_TO_CONFIG(config_key) { char * tmp; GET_ARG(tmp); \
                   config_update_val(config_key, SRC_CMDLINE, str_val(tmp)); }
#define ARG_TO_CONFIG_EAT(config_key) { char * tmp; GET_ARG_FUNC(tmp,EAT_OPT); \
                   config_update_val(config_key, SRC_CMDLINE, str_val(tmp)); }

#define EAT_OPT() \
    { argc = eat_option(argc,argv,0); }

/*#define cmdline_parse_int(intvar,st) {\
    char **endptr = (char **) malloc(sizeof(char*)); \
    intvar = strtol(st,endptr,10); \
    if(*st=='\0' || **endptr!='\0') sys_fatal(EXIT_CMDLINE, usage()); \
    free(endptr); \
}*/

#define cmdline_parse_int(intvar,st) {\
    char **endptr = (char **) malloc(sizeof(char*)); \
    intvar = strtol(st,endptr,10); \
    free(endptr); \
}


#endif

/* -*- Emacs -*- 
Local Variables:
c-basic-offset:4
fill-column:75
End: 
*/
