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

/* Returns the exit function */
void (*GuSysGetExitFunc())(int)
{
	if (gu_utils.sys_exit_func == (void *)0) {
#ifndef DEBUG
		gu_utils.sys_exit_func = (void *)exit;
#else
		gu_utils.sys_exit_func = (void *)gu_exit_abort;
#endif
	}
	return gu_utils.sys_exit_func;
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


/* Message printing utilities */

/* Print a message - depending on verbosity level */
static void gu_sys_msg(int level, const char *msg, ...)
{
    va_list ap;
    if (GuSysGetVerbose() >= level) {
		if (GuSysGetProgname())
			fprintf(stderr, "%s: ", GuSysGetProgname());
		
		va_start(ap, msg);
		vfprintf(stderr, msg, ap);
		va_end(ap);
		fflush(stderr);
    }
	return;
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


/** Create a name for a temporary file. Storage is automatically
 * allocated. So the result has to be xfree'd, to prevent a leak 
 *
 * This is a tricky function to implement. On modern UNIX systems
 * mkstemp is available, which is ideal. However, on Win32 and older
 * Unices, there is no mkstemp. In these cases we use tmpnam() which
 * works differently, and is more dangerous. 
 */

char *GuSysTmpname(char *dir, char *pathsep, char *template) 
{
#ifdef WIN32
    char *name;

	/* in Windows, _tempnam should do the same as mkstemp */
    name = _tempnam(dir, template);

    if(name == NULL)
        gu_sys_fatal(EXIT_CMDLINE, "TEMP name generation failed!");

	return name;
#else
#ifdef HAVE_MKSTEMP
    int fd;
    char *full_template = GuMakeMessage("%s%s%s", dir, pathsep, template);
    fd = mkstemp(full_template);
    if(fd==-1) {
		gu_sys_err("template='%s'\n", full_template);
		free(full_template);
		Throw exc(ERR_IO_TMPNAME);
		return NULL; /* never reached */
    }
    else {
		/* mkstemp actually creates an empty file; remove it, we only want the name */
		gu_sys_msg(1, "Unlinking %s\n", full_template);
		unlink(full_template); 
		close(fd); 
		return full_template;
    }
#else
    char * result = malloc(L_tmpnam);
    char * retval;
    retval = tmpnam(result);
    if(retval==NULL) {
		free(result);
		Throw exc(ERR_IO_TMPNAME);
		return NULL; /* never reached */
    }
    else {
		return result;
    }
#endif
#endif
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


/* Return the value of  an environment variable or else return an empty string */
config_val_t *config_demand_val(char * name)
{
    char* env = NULL;
    config_val_t* ret = (config_val_t *)malloc(sizeof(config_val_t));
	// char *dmsg = GuMakeMessage("***   Message: # bytes allocated = %d, ptr = %p, value = %s\n", sizeof(config_val_t), (char *)ret, (char *)ret);

	if (ret == (config_val_t *)NULL)
		return ret;
	
    if (strcmp(name, "tmp_dir") == 0) {
        env = getenv("SPIRAL_CONFIG_TMP_DIR");
    } else if( strcmp(name, "spiral_dir") == 0) {
        env = getenv("SPIRAL_CONFIG_SPIRAL_DIR");
    } else if( strcmp(name, "exec_dir") == 0) {
        env = getenv("SPIRAL_CONFIG_EXEC_DIR");
    } else if( strcmp(name, "path_sep") == 0) {
        env = getenv("SPIRAL_CONFIG_PATH_SEP");
    } else if( strcmp(name, "gap_lib_dir") == 0) {
        env = getenv("SPIRAL_GAP_LIB_DIR");
    } else
        return NULL;			// L-COMM: the function ConfHasVal in sys_conf.g
								// requires NULL return to report missing key-value pair.

    if (env == NULL) {
        env = "";
    }
    ret->type = VAL_STR;
    ret->strval = env;
    return ret;
}

config_val_t * config_get_val(char * name) { 
    return config_demand_val(name);
}

config_val_t * config_get_val_profile(config_profile_t * profile, char * name) {
    return config_demand_val(name);
}

config_val_t * config_valid_val(char * name) { 
    return config_demand_val(name);
}

config_val_t * config_valid_val_profile(config_profile_t * profile, char * name) {
    return config_demand_val(name);
}

/** Same as config_valid_val, but returns a string value, or "" 
    if value is not valid or type!=VAL_STR */
char * config_valid_strval(char * name) {
    config_val_t * temp = config_demand_val(name);
    return temp->strval;
}

char * config_valid_strval_profile(config_profile_t * profile, char *name) {
    return config_valid_strval(name);
}

config_val_t * config_demand_val_profile(config_profile_t * profile, char * name) {
    return config_demand_val(name);
}

/*
 * Check that a named file exists, abort if it does not
 */

void GuSysCheckExists(const char * fname)
{
    FILE *f = fopen(fname, "r");
	if (!f)
		Throw exc(ERR_IO_FILE_READ, fname);

	fclose(f);
	return;
}

/*
 * Set an environment variable to value
 */

int GuSysSetenv(char *var, char *value, int i)
{
#ifdef _WIN32
    char *assignment = GuMakeMessage("%s=%s", var, value);
    if(putenv(assignment) != 0) gu_sys_fatal(0xff, "_putenv: failed");
    free(assignment);
    return 0;
#else
    return setenv(var, value, i);
#endif
}

/*
 * Required for InstIntFunc() handling
 */

static char *quotify(const char *str) {
    char *buf = (char*) malloc(strlen(str) + 3);
    sprintf(buf, "%c%s%c", QUOTIFY_CHAR, str, QUOTIFY_CHAR);
    return buf;
}

static char *quotify_static(char *str) {
    static char *buf = NULL;
    if(buf!=NULL)
	free(buf);
    buf = quotify(str);
    return buf;
}

/* Under Windows we need to quotify the command (Franz, why?) */
#ifdef QUOTIFY_SHELL_COMMAND 
char *command_quotify_static(char * command) { return quotify_static(command); }
#else
char *command_quotify_static(char * command) { return command; }
#endif

char *file_quotify_static(char * fname) { 
    if(strchr(fname, ' ')!=NULL) return quotify_static(fname); 
    else return fname; 
}

int sys_exists(const char *fname) {
    FILE * f = fopen(fname, "r");
    if(!f) return 0;
    fclose(f);
    return 1;
}

int sys_mkdir(const char * name) {
    char * cmd_mkdir = config_demand_val("cmd_mkdir")->strval;
    char * command;
    int result;
#ifdef WIN32
    /*
	 * windows 'mkdir' fails if directory exists
	 * and fopen() cannot be used to check if directory exists
	 */
    char * cwd = (char*) getcwd(0,0);	/* get current directory */
    result = chdir(name);	/* see if 'name' already exists by chdir'ing */
    chdir(cwd);				/* go back to where we started */
    free(cwd); 
    if(result!=-1) {
		gu_sys_msg(1, "'%s' exists\n", name);
		return 0;
    }
#endif
    /* put directory name in quotes to handle names with spaces */
    command = GuMakeMessage("%s \"%s\"\n", cmd_mkdir, name);
    gu_sys_msg(1, command);
    result = system(command);
    free(command);
    return result;
}

int sys_rm(const char * name) {
    if(sys_exists(name)) {
		gu_sys_msg(2, "removing '%s'\n", name);
		return remove(name);
    }
    else {
		gu_sys_msg(2, "removing '%s' - file does not exist\n", name);
		return 0;
    }
}

void sys_check_exists(const char * fname) {
    if(!sys_exists(fname))
	Throw exc(ERR_IO_FILE_READ, fname);
}



