/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */

#include <stdarg.h>

#ifndef NULL
#define NULL ((void*)0)
#endif

/** Returns 1 if character c is considered whitespace */
int is_whitespace(char c);

/** Remove all whitespace from the end (r=right) of the string 
 *  Directly modifies the string, by inserting \0 at the cutoff point. */
char * chopr(char *s);

/** Remove newline at end (r=right) of the string.
 *  Directly modifies the string, by inserting \0 at the cutoff point. */
char * chopr_newline(char *s);

/** Returns string starting at the first non-whitespace character of s. */
char * chopl(char *s);

/** Return next token in configuration line */
char * strdelim(char **s);

/** Return next token in a string treating quotified entries as one token,
 *  and returning quotified tokens with quotes stripped */
char * strdelim_q(char **s);

/** Convert string to uppercase (in-place) */
char * strupper(char *s);

/** Concatenate two strings, memory for result is allocated. */
char * append_str(char * target, char * add);

/** Concatenate two strings with space inbetween, memory for result is allocated. */
char * append_str_spc(char * target, char * add);

/** Concatenates two strings, result is stored in *targetp, which will be
 *  xfree'd beforehand */
void append_to(char ** targetp, char * add);

/** Concatenates two strings with space inbetween, result is stored in
 *  *targetp, which will be xfree'd beforehand */
void append_to_spc(char ** targetp, char * add);

/** Print to a string. Memory for result is automatically allocated. */
char * make_message(const char *fmt, ...);

/** Print to a string. Memory for result is automatically allocated. */
char * vmake_message(const char *fmt, va_list ap);

/** Concatenates count strings in string array separated by delimitor and
 *  returns the resulting string. */
char * flatten_string_array(char** strings, int count, char *delimitor);

/** Compares two string ignoring case, returns 0 if strings are equal, 1 if
 *  st1 is greater, -1 if st2 is greater */
int xstrcasecmp(const char *st1, const char *st2);

/** Expands references in a string using the expand_ref_func.  References
 * are identifiers that start with delimitor. Every such identifier will be
 * substituted by the value of expand_ref_func(id) where id is the
 * identifier, stripped of the delimitor. Original string is untouched, but
 * the new string for result is constructed 
 */
char * expand_refs(char delimitor, 
		   char * (*expand_ref_func)(char*, void*), 
		   void * func_params,
		   char * identifier_chars,
		   char * st );

/** Generates a unique temporary file name from template. Template must be
 * a simple filename (without directory paths), last six characters of
 * template must be XXXXXX, and these are replaced with a string that makes
 * the filename unique. 
 *
 * None of the parameters will be modified. The result will be allocated
 * with xmalloc(). It has to be deallocated by the callee with xfree().
 *
 * Note: on older Unices and Win32 tmpnam() is used so that template
 * parameter will be effectively ignored. 
 */
char * sys_tmpname (char *dir, char *pathsep, char *templat);

/* ===================================================================
 * ======================= QUOTIFICATION ============================= 
 * ===================================================================
 */

/* There are two things that need to be quotified in some cases: files
 * for use in shell commands and shell commands themselves. The
 * following functions perform the required quotification when it is needed.
 *
 * xxx_quotify functions allocate storage for the result with xmalloc(), so
 * that result has to be freed with xfree().
 *
 * xxx_quotify_static functions allocate storage for the result, but
 * deallocate it on the next call. They are useful when the result is used
 * only once, for instance in a make_message() call.
 * 
 * xxx_quotify_realloc functions xfree() the argument and assign newly
 * allocated storage to the argument. They can be treated as being
 * in-place, since they perform allocation/deallocation transparently. If
 * given a pointer to a NULL string, xfree() will not be called.  */

/** Quotify by allocating a new string */
char *quotify(const char *str);

/** Same as quotify, but returns a pointer valid only until second call to
 * quotify_static, when it will be automatically freed */
char *quotify_static(char *str);

/** In place quotification. xfree *str, then allocate new storage */
char *quotify_realloc(char **str);


/* ========================================================================
 * The need for the quotification in the following functions is determined
 * by the QUOTIFY_xxx macros in conf.h
 * ========================================================================
 */

/** If needed, quotify a file name using quotify(),
 *  otherwise return a copy */
char *file_quotify(const char * fname);

/** If needed, quotify a file name using quotify_static(), 
 *  otherwise return unmodified fname */
char *file_quotify_static(char *fname);

/** If needed, quotify a file name using quotify_realloc(),
 *  otherwise return unmodified *fnamep */
char *file_quotify_realloc(char ** fnamep);


/** Same as file_quotify but for a shell command */
char *command_quotify(const char * command);

/** Same as file_quotify_static but for a shell command */
char *command_quotify_static(char * command);

/** Same as file_quotify_realloc but for a shell command */
char *command_quotify_realloc(char ** commandp);

/* -*- Emacs -*- 
Local Variables:
c-basic-offset:4
fill-column:75
End: 
*/
