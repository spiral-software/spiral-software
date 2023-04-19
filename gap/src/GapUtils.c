/*
 * GapUtils.c
 * Utility functions for GAP. 
 *
 * Conventions:  
 *    public functions will be prefixed with Gu, and camel-case name, e.g., GuMakeMessage()
 *    private (static) functions will be prefixed with gu_, and underscore separated name,
 *	  e.g., gu_vmake_message()
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#if defined(_WIN32) || defined(_WIN64) || defined(WINDOWS)
#include <direct.h>
#else
#include <unistd.h>
#endif
#include "conf.h"
#include "GapUtils.h"

// for GAP to C functions
#include "system.h"
#include "memmgr.h" 
#include "integer.h"
#include "args.h"
#include "eval.h"


struct gu_msg_utils {
	int msg_verbose;
	char *msg_progname;
	void (*sys_exit_func)(int);
};

static struct gu_msg_utils gu_utils = { 0, (char *)0, (void *)0 };

void GuSysSetVerbose(int code)
{
	gu_utils.msg_verbose = code;
	return;
}

int GuSysGetVerbose()
{
	return gu_utils.msg_verbose;
}

/* 
 * Set the name of the running program (argv[0]) [appears in output of gu_sys_err/gu_sys_fatal] 
 * If progname is NULL, gu_sys_err/gu_sys_fatal won't print the program name.
 */

void GuSysSetProgname(char *progname)
{
    gu_utils.msg_progname = progname;
	return;
}

char *GuSysGetProgname()
{
    return gu_utils.msg_progname; 
}


static void gu_exit_abort(int code)
{
    abort();
}

/* Sets the exit function to be called in gu_sys_fatal */
void GuSysSetExitFunc(void (*exit_func)(int) )
{
    gu_utils.sys_exit_func = exit_func;
}




/*
 * Print to a string.  The correct amount of memory for result is automatically allocated.
 */


static char *gu_make_message(const char *fmt, va_list ap) 
{
    int n;
    char *p;
	va_list ap2;					/* vsnprintf() mangles the arg list, make a copy */

	va_copy(ap2, ap);				/* save a copy for 2nd vsnprintf() call */
	/* figure out the size needed for output buffer. */
	n = vsnprintf(NULL, 0, fmt, ap);
	if (n < 0) 
		return NULL;					/* error */
	
	/* allocate a buffer (n + 1) bytes -- add 1 for NULL terminator  */
	n++;
	if ((p = (char *)malloc(n)) == NULL)
		return NULL;

	n = vsnprintf(p, n, fmt, ap2);	/* pass virgin copy of arglist  */
	//  printf("gu_make_message: malloc() ptr = %p, copied string = \"%s\"\n", p, p);
	return p;
}


char *GuMakeMessage(const char *fmt, ...) {
    char *result;
    va_list ap;
    va_start(ap, fmt);
    result = gu_make_message(fmt, ap);
    va_end(ap);
    return result; 
}



/* prints err_msg to stderr, printing progname in front of err_msg */
static void gu_sys_err(const char *err_msg, ...)
{
    va_list ap;
    if(GuSysGetProgname()) 
		fprintf(stderr, "%s: ", GuSysGetProgname());

    va_start(ap, err_msg);
    vfprintf(stderr, err_msg, ap);
    va_end(ap);
}

/* prints err_msg to stderr, and exits with specified code */
static void gu_sys_fatal(int exit_code, const char *err_msg, ...)
{
    va_list ap;
    if (GuSysGetProgname())
		fprintf(stderr, "%s: ", GuSysGetProgname());
	
    va_start(ap, err_msg);
    vfprintf(stderr, err_msg, ap);
    va_end(ap);
    gu_utils.sys_exit_func(exit_code);
}

/* public function to print an [fatal] error message and exit with the specified code */
void GuFatalMsgExit(int exit_code, const char *err_msg, ...)
{
    va_list ap;
    va_start(ap, err_msg);
	gu_sys_err("Spiral fatal error, exiting\n");
	gu_sys_err(err_msg, ap);
	exit(exit_code);
}

/* if DEBUG is set, print the message to stderr */
void GuSysDebug(const char *msg, ...)
{
#ifdef DEBUG
    va_list ap;
    va_start(ap, msg);
    if(GuSysGetProgname())
		fprintf(stderr, "%s: ", GuSysGetProgname());
	
    vfprintf(stderr, msg, ap);
    va_end(ap);
    fflush(stderr);
#endif
}

/* prints err_msg to stderr */
void GuSysStderr(const char *err_msg, ...)
{
    va_list ap;
    va_start(ap, err_msg);
    vfprintf(stderr, err_msg, ap);
    va_end(ap);
}



/* Throw / Catch exception handling */
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
exc_type_t exc ( exc_type_t type , ... )
{ 
    static int msg_len = sizeof(the_extended_exception_context->msg) / sizeof(char);
    exc_msg_template_t * templ = exc_find_msg_template(type);
    int num_printed = 0;
    va_list ap;
    va_start(ap, type);
    num_printed = vsnprintf(the_extended_exception_context->msg,
							msg_len, templ->msg, ap);
    va_end(ap);
    if(num_printed==-1 || num_printed >= msg_len)
		GuSysDebug("Exception message was clipped");
    return type;
}


char *exc_err_msg ( )
{
    return the_extended_exception_context->msg;
}


/* This function prints 'progname::  ' in front of every line of str */
static void _exc_show_with_progname(char * progname, char *str)
{
    char * pos = str;
    while(pos!=NULL) {
		char * newpos;
		GuSysStderr("%s::  ", progname);
		newpos = strchr(pos, '\n');
		if(newpos==NULL)
			GuSysStderr(pos); 
		else {
			++newpos;
			fwrite(pos, newpos-pos, 1, stderr);
		}
		pos = newpos;
    }
}


/* Prints out the information stored in the_extended_exception_context */
void exc_show ( ) 
{
    char * progname = GuSysGetProgname() ? GuSysGetProgname() : "";
    GuSysStderr("%s: ", progname);
    GuSysStderr(EXC_ORIGIN_TEMPLATE, the_exception_context->file, 
			   the_exception_context->func, 
			   the_exception_context->line);
    GuSysStderr("\n");
    _exc_show_with_progname(progname, the_extended_exception_context->msg);
    GuSysStderr("\n");
}


int sys_exists(const char *fname) {
    FILE * f = fopen(fname, "r");
    if(!f) return 0;
    fclose(f);
    return 1;
}


int sys_rm(const char * name) {
    if(sys_exists(name)) {
		return remove(name);
    }
    else {
		return 0;
    }
}


char *PathSep()
{
#ifdef _WIN32
    return "\\";
#else
    return "/";
#endif
}


Bag FunFileExists(Bag argv) {
    char* usage = "sys_exists (const char *fname)";
    int  argc = GET_SIZE_BAG(argv) / SIZE_HD;

    int  _result;
    char* _arg0;

    if ((argc < 2) || (argc > 2)) {
        return Error(usage, 0, 0);
    }
    
    _arg0 = (char*)HdToString(ELM_ARGLIST(argv, 1),
            "<fname> must be a String.\nUsage: %s", (Int)usage, 0);

    _result = (int)sys_exists(_arg0);

    return INT_TO_HD(_result);
}


extern  Bag Error (char *msg, Int arg1, Int arg2);

Bag FunSysRm(Bag argv) {
    char* usage = "sys_rm (const char *name)";
    int  argc = GET_SIZE_BAG(argv) / SIZE_HD;

    int  _result;
    char* _arg0;

    if ((argc < 2) || (argc > 2)) {
        return Error(usage, 0, 0);
    }

    _arg0 = (char*)HdToString(ELM_ARGLIST(argv, 1),
            "<name> must be a String.\nUsage: %s", (Int)usage, 0);

    _result = (int)sys_rm(_arg0);
  
    return INT_TO_HD(_result);
}


Bag FunPathSep(Bag hdCall) {
    char* sep = PathSep();

    return StringToHd(sep);
}


void     Init_GAP_Utils(void) {

    InstIntFunc("FileExists", FunFileExists);
    InstIntFunc("sys_rm", FunSysRm);
    InstIntFunc("PathSep", FunPathSep);
}



