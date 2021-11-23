#ifndef EXCEPTIONS_C
#error "Do NOT include this file directly. Include 'exceptions.h' instead."
#else

/* 
 * This file defines exception error message templates.
 * exc_msg_template_t format: 
 *
 *    { exception type, "template", number of % substitutions }
 * 
 * '$' keyword expansions are also used in message templates: 
 *    $cmd -> sys_get_progname()
 */

char * EXC_ORIGIN_TEMPLATE = "Exception at %s: %s(): %d (file:func:line)";

exc_msg_template_t EXC_ASSERTION_TEMPLATE = { 
    ERR_ASSERTION_FAILED, "$cmd: Assertion '%s' failed", 1 
};

exc_msg_template_t EXC_MSG_TEMPLATES[] = 
{ 
    { ERR_IO_FILE_READ,  "'%s' can not be opened for reading", 1 }, 
    { ERR_OTHER, "%s", 1 },
    { ERR_GAP, "", 0}
};

#endif
