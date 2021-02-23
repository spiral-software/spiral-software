/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */
 
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "exceptions.h"
#include "conf.h"
#include "sys.h"

/* The implementation of exception functions does not use dynamic
 * memory allocation functions xmalloc/xfree. This allows to handle
 * memory allocation errors occurring in xmalloc/xfree using the same
 * exceptions interface as other errors.
 */

/* Declaring as an array allows to avoid creating a separate pointer variable */
struct extended_exception_context the_extended_exception_context[1];

#define EXCEPTIONS_C
#include "exceptions_msg.h"
#undef EXCEPTIONS_C

static exc_msg_template_t * exc_find_msg_template (exc_type_t type) {
    int i;
    if(type==ERR_ASSERTION_FAILED) 
	return & EXC_ASSERTION_TEMPLATE;

    for(i=0; i < sizeof(EXC_MSG_TEMPLATES)/sizeof(exc_msg_template_t); ++i)
	if(type==EXC_MSG_TEMPLATES[i].type)
	    return & EXC_MSG_TEMPLATES[i];

    Throw exc(ERR_ASSERTION_FAILED, "'type' must be a known type");
}

/* Initializes the_extended_exception_context->msg */
exc_type_t exc ( exc_type_t type , ... ) { 
    static int msg_len = sizeof(the_extended_exception_context->msg) / sizeof(char);
    exc_msg_template_t * templ = exc_find_msg_template(type);
    int num_printed = 0;
    va_list ap;
    va_start(ap, type);
#if WIN32
    num_printed = _vsnprintf(the_extended_exception_context->msg, msg_len, templ->msg, ap);
#else
    num_printed = vsnprintf(the_extended_exception_context->msg, msg_len, templ->msg, ap);
#endif

    va_end(ap);
    if(num_printed==-1 || num_printed >= msg_len)
	sys_debug("Exception message was clipped");
    return type;
}

char *exc_err_msg ( ) {
    return the_extended_exception_context->msg;
}

/* This function prints 'progname::  ' in front of every line of str */
static void _exc_show_with_progname(char * progname, char *str) {
    char * pos = str;
    while(pos!=NULL) {
	char * newpos;
	sys_stderr("%s::  ", progname);
	newpos = strchr(pos, '\n');
	if(newpos==NULL)
	    sys_stderr(pos); 
	else {
	    ++newpos;
	    fwrite(pos, newpos-pos, 1, stderr);
	}
	pos = newpos;
    }
}

/* Prints out the information stored in the_extended_exception_context */
void exc_show ( ) {
    char * progname = (char*) (sys_get_progname() ? sys_get_progname() : "");
    sys_stderr("%s: ", progname);
    sys_stderr(EXC_ORIGIN_TEMPLATE, the_exception_context->file, 
	       the_exception_context->func, 
	       the_exception_context->line);
    sys_stderr("\n");
    _exc_show_with_progname(progname, the_extended_exception_context->msg);
    sys_stderr("\n");
}

/* 
Local Variables:
c-basic-offset:4
End:
*/
