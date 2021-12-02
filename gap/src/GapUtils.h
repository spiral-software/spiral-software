#ifndef GAP_UTILS_H_INCLUDED
#define GAP_UTILS_H_INCLUDED

/*
 * GapUtils.h
 * Utility functions for GAP. 
 */

#include <stdarg.h>

/** Print to a string. Memory for result is automatically allocated. */
char *GuMakeMessage(const char *fmt, ...);

/*
 * Utilities to set/get: message verbosity level, program name, exit function
 */

void GuSysSetVerbose(int code);
int GuSysGetVerbose();
void GuSysSetProgname(char *progname);
char *GuSysGetProgname();
void GuSysSetExitFunc(void (*exit_func)(int) );

void GuFatalMsgExit(int exit_code, const char *err_msg, ...);


/* Utilities to print messages */

/** Print to stderr */
void GuSysStderr(const char *err_msg, ...);
void GuSysDebug(const char *msg, ...);


/* Exceptions support is built on-top of CExcept library by by Adam
 * M. Costello and Cosmin Truta (cexcept@sourceforge.net).  
 *
 * See exceptions_def.h for the complete code of CExcept.  
 */

/*  #include "conf.h"  */
#define EXCEPTIONS_H
#include "exceptions_def.h"
#undef EXCEPTIONS_H

/** Exception type.  
 *
 * This is the type of object that gets copied from the exception thrower
 * to the exception catcher. In Catch(e) expressions 'e' must be of this
 * type.
 * 
 * Each of the exception types defined here (except ERR_ASSERTION_FAILED)
 * need an error message template in EXC_MSG_TEMPLATES. If the error
 * message template can't be located, an assertion exception will be
 * raised, which has a separate template - EXC_ASSERT_TEMPLATE.
 * 
 * These values should not be thrown directly, instead exc() function must
 * be used to instantiate an exception. It will initialize the error
 * message using the template, and performing '%' expansions. For example,
 * a template for ERR_ALLOC_OUT_OF_MEMORY has one long integer expansion
 * thus:
 *   
 *   Throw exc(ERR_ALLOC_OUT_OF_MEMORY, (long)alloc_bytes);
 *
 * Creating a separate exception type for each exception is strongly
 * encouraged, however if you don't want to create one, you can use
 * ERR_OTHER, error message for which has one string expansion:
 * 
 *   Throw exc(ERR_OTHER, "your error message");
 * */

typedef enum exc_type {
    ERR_OTHER,
    ERR_GAP, 
    ERR_ASSERTION_FAILED
} exc_type_t;

/** Initializes an exception of type 'type'. This function needs to be
 * called with Throw. It sets the error message in the
 * extended_exception_context by and perfoming '%' expansions on error
 * message template using supplied variadic arguments. Example:
 * 
 *   Throw exc(ERR_ALLOC_OUT_OF_MEMORY, (long)alloc_bytes); 
 */
exc_type_t exc ( exc_type_t type , ... );


/** Prints the information about the last caught exception to stderr  */
void exc_show ( );

/** Returns the error message string of the last caught exception  */
char *exc_err_msg ( );

/** Defines an exception type for use by CExcept header  */
define_exception_type(exc_type_t);

/** Alternative to C assert macro, throws ERR_ASSERTION_FAILED if cond == 0 */
#define ASSERT(cond) if(!(cond)) {Throw exc(ERR_ASSERTION_FAILED, #cond);} else ;

/** Extension to CExcept's exception_context type. An instance will be
 *  known to both thrower and catcher.
 *
 * This particular extension allows dynamically allocated messages (created
 * from message templates). Since only one exception is active at any time,
 * msg field is automatically deallocated on each exc() call.
 *
 */
extern struct extended_exception_context {
    char msg[2000];
    struct exception_context context[1];
} the_extended_exception_context[1];

/** CExcept required portion of the exception context 
 */
#define the_exception_context (the_extended_exception_context->context)

/** Exception error message template.
 *
 * These templates associate an exception type with error messages. Since
 * in most cases error messages are not exactly constant (for example "Can
 * not read file 'non_constant_part' "), msg field allows '%' expansions to
 * be used just like in printf. These expansions are expanded by exc()
 * call, and the result is written to the_extended_exception_context->msg
 *
 */
typedef struct exc_msg_template { 
    exc_type_t type; 
    char * msg; 
    int num_msg_args; 
} exc_msg_template_t;

/** Appended in front of any error message. %s %s %d expansions are
 *  expanded as (file, func, line), which denote the source location of the
 *  exception origin.
 */
extern char *             EXC_ORIGIN_TEMPLATE;

/** Array of error message templates to be searched to create an error
 *  message when an exception occurs 
 */
extern exc_msg_template_t EXC_MSG_TEMPLATES[];

/** Error message template for EXC_ASSERTION_FAILED exceptions. It is
 *  separate, if the errorcmessage template can't be locatedin
 *  EXC_MSG_TEMPLATES, an assertion exception will be raised.
 */
extern exc_msg_template_t EXC_ASSERTION_TEMPLATE;


/* Program exit codes */
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


/** Structures and types supporting get profile functions **/



/** key_name -> semantic meaning association. 
    Used for the table of keys. The value of each key
    will be parsed according to its semantic meaning. 
    See config_atom/semantic_t declaration for possible values */
typedef struct config_key {
    char *name;
    int semantic;
    char *comment;
    void *semantic_arg; /* optional, used for SEM_ENUM */
} config_key_t;

/** default configuration keys */
extern config_key_t conf_keys[]; 

typedef struct config_profile {
    char * name;
    int type;
    struct config_profile * next;
    struct config_profile * parent;
} config_profile_t;

typedef enum config_val_type { 
    VAL_INT, 
    VAL_STR, 
    VAL_FLOAT 
} config_val_type_t;

/** Single value */
typedef struct config_val {
    config_val_type_t type;
    int intval;
    char * strval;
    double floatval;
} config_val_t;


#endif					// GAP_UTILS_H_INCLUDED

