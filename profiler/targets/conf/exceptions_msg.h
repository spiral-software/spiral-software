/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */
 
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
    { ERR_IO_FILE_WRITE, "'%s' can not be opened for writing", 1 }, 
    { ERR_IO_TMPDIR_CREATE, "Can't create temporary directory '%s'", 1 },
    { ERR_IO_TMPNAME, "Can't create a temporary file name using template '%s'", 1 },
    { ERR_MEM_ALLOC, "Out of memory (allocating %lu bytes)", 1 },
    { ERR_MEM_ALLOC_ZERO, "Tried to allocate 0 bytes", 0 },
    { ERR_MEM_REALLOC, "Out of memory (reallocating to %lu bytes)", 1 },
    { ERR_MEM_REALLOC_ZERO, "Tried to reallocate to 0 bytes", 0 },
    { ERR_MEM_FREE_NULL, "Tried to deallocate a NULL pointer", 0 }, 
    
    { ERR_SYS_SPL_COMPILE, "SPL compilation of '%s' using '%s' failed", 2 },
    { ERR_SYS_TARGET_COMPILE, "Target compilation of '%s' using '%s' failed", 2 },
    { ERR_SYS_LINK, "Linking using '%s' failed", 1 },
    { ERR_SYS_DIMENSION, "Bad dimension", 0}, 		
    { ERR_SYS_DIMENSION_DET, "Could not determine dimensions of '%s', specify manually", 1 }, 

    { ERR_INVALID_KEYWORD, "Invalid keyword '%s' in %s", 2 },
    { ERR_CMDLINE,       "Bad command line", 0 }, 

    { ERR_CONFIG_PROFILE, "Bad configuration profile selected: '%s'\n"
                          "Use 'spiral_config -profiles' to see defined profiles", 1 }, 
    { ERR_CONFIG_FILE, "Errors found in global configuration file '%s'", 1 },
    { ERR_CONFIG_ENV,  "Environment: %d bad variables", 1 },
    { ERR_CONFIG_UNINIT_EXPAND, "Value of uninitialized key '%s' requested", 1 },
    { ERR_CONFIG_DEMAND_NOT_VALID, "Key '%s' has invalid value", 1 },
    { ERR_CONFIG_DEMAND_NOT_INIT,  "Key '%s' was not initialized", 1 },

    { ERR_OTHER, "%s", 1 },
    { ERR_GAP, "", 0}
};

#endif
