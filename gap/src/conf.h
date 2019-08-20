/* $Id: conf.h.in 1294 2004-04-26 17:13:29Z uyvorone $ */

#ifndef CONF_H_INCLUDED
#define CONF_H_INCLUDED

#ifdef WIN32
#undef HAVE_GETTIMEOFDAY
#undef HAVE_GETUID
#undef HAVE_ISATTY
#undef HAVE_UNISTD_H
#undef HAVE_MKSTEMP
#else
#define HAVE_GETTIMEOFDAY 1
#define HAVE_GETUID 1
#define HAVE_ISATTY 1
#define HAVE_UNISTD_H 1
#define HAVE_MKSTEMP 1
#endif

#ifdef WIN32
#define QUOTIFY_SHELL_COMMAND 1
#define QUOTIFY_CHAR '\"'
#ifndef _MSC_VER
#define vsnprintf _vsnprintf
#endif
#define getcwd _getcwd
#define __func__ __FUNCTION__
#else
/* #undef QUOTIFY_SHELL_COMMAND */
#define QUOTIFY_CHAR '\''
#endif

#define DEFAULT_CONFIG_FILE "spiral.conf"

#endif /* CONF_H_INCLUDED */

