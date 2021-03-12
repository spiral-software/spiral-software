/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */
 
#include <string.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h> /* unlink(), close() */
#endif 
#include "misc.h"
#include "xmalloc.h"
#include "conf.h"
#include "sys.h"
#include "exceptions.h"
#include "errcodes.h"

#ifdef __CYGWIN__
extern int mkstemp(char *template);
#endif

/** Remove newline at end (r=right) of the string.
    Directly modifies the string, by inserting \0 at the cutoff point.*/
char * chopr_newline(char *s) {
    char *t = s;
    while (*t) {
	if(*t == '\n' || *t == '\r') {
	    *t = '\0';
	    return s;
		}
	t++;
    }
    return s;
}

/** Characters considered whitespace */
#define WHITESPACE " \t\r\n"

/** Returns 1 if character c is considered whitespace */
int is_whitespace(char c) {
    return c!='\0' && strchr(WHITESPACE, c)!=NULL;
}

/** Remove all whitespace from the end (r=right) of the string 
    Directly modifies the string, by inserting \0 at the cutoff point.*/
char * chopr(char *s) {
    int i;
    if(s==0) return s;
    i = strlen(s)-1;
    while(is_whitespace(s[i]) && i>=0)
	--i;
    s[i+1] = '\0';
    return s;
}

/* Returns string starting at the first non-whitespace character of s. */
char * chopl(char *s) {
    char * retval = s;
    while(is_whitespace(*retval))
	++retval;
    return retval;
}

/** Return next token in configuration line */
char * strdelim(char **s) {
    char *old;
    int wspace = 0;

    if (*s == NULL)
	return NULL;
    
    old = *s;
    
    *s = strpbrk(*s, WHITESPACE "=");
    if (*s == NULL)
	return (old);

    /* Allow only one '=' to be skipped */
    if (*s[0] == '=')
	wspace = 1;
    *s[0] = '\0';
    
    *s += strspn(*s + 1, WHITESPACE) + 1;
    if (*s[0] == '=' && !wspace)
	*s += strspn(*s + 1, WHITESPACE) + 1;
    
    return (old);
}

/** Return next token in a string treating quotified entries as one token,
 * and returning quotified tokens with quotes stripped 
 */
char * strdelim_q(char **s) {
    char *retval;
    if (*s == NULL) return NULL;
    /* ignore leading whitespace if any */
    *s += strspn(*s, WHITESPACE);

    /* if first character is a quote */
    if(*s[0]=='"') {
	/* don't include first quote in returned token */
	retval = *s + 1; 
	/* look for other quote */
	*s = strpbrk(*s+1, "\"");
    }
    else {
	retval = *s;
	/* look for first whitespace character */
	*s = strpbrk(*s, WHITESPACE);
    }

    /* if quote/whitespace is not found then there is only one token */
    if (*s == NULL)
	return retval;
    else {
	/* set the cut point at the found character */
	*s[0] = '\0';
	*s += strspn(*s + 1, WHITESPACE) + 1;
	return retval;
    }
}

/** Convert string to uppercase (in-place) */
char * strupper(char *s) {
    char *p = s;
    if(s==NULL) return s;
    while(*p!='\0') {
	*p = (unsigned char) toupper(*p);
	p++;
    }
    return s;
}
	
/** Compares two string ignoring case, returns 0 if strings are equal, 
    1 if st1 is greater, -1 if st2 is greater */
int xstrcasecmp(const char *st1, const char *st2) {
    int retval;
    char *upper1 = strupper( xstrdup(st1) );
    char *upper2 = strupper( xstrdup(st2) );
    retval = strcmp(upper1, upper2);
    xfree(upper1);
    xfree(upper2);
    return retval;
}

/** Concatenates two strings, memory for result is allocated. */
char * append_str(char * target, char * add) {
    char * tmp;
    int len;
    if(target==NULL)
	return xstrdup(add);
    else {
	len = (strlen(target) + strlen(add)) + 1;
	tmp = (char*) xmalloc(sizeof(char) * len);
	sprintf(tmp, "%s%s", target, add);
	return tmp;
    }
}

/** Concatenates two strings with space inbetween, memory for result is allocated. */
char * append_str_spc(char * target, char * add) {
    char * tmp;
    int len;
    if(target==NULL)
	return xstrdup(add);
    else {
	len = (strlen(target) + strlen(add)) + 2;
	tmp = (char*) xmalloc(sizeof(char) * len);
	sprintf(tmp, "%s %s", target, add);
	return tmp;
    }
}

/** Concatenates two strings, result is stored in target,
    Target will be xfree'd (if non-NULL) before hand */
void append_to(char ** targetp, char * add) {
    char * tmp = append_str(*targetp, add);
    if(*targetp!=NULL) 
	xfree(*targetp);
    *targetp = tmp;    
}

/** Concatenates two strings, with space inbetween. Result is stored in target,
    Target will be xfree'd (if non-NULL) before hand */
void append_to_spc(char ** targetp, char * add) {
    char * tmp = append_str_spc(*targetp, add);
    if(*targetp!=NULL) 
	xfree(*targetp);
    *targetp = tmp;    
}

/** Print to a string. Memory for result is automatically allocated. */
char * vmake_message(const char *fmt, va_list ap) {
    /* Guess we need no more than 100 bytes. */
    /* VIENNA: 1000 are better */
    int n, size = 1000;
    char *p;

    if ((p = (char*)xmalloc (size)) == NULL)
	return NULL;
    while (1) {
	/* Try to print in the allocated space. */
#ifdef WIN32
	n = _vsnprintf (p, size, fmt, ap);
#else
	n = vsnprintf (p, size, fmt, ap);
#endif

	/* If that worked, return the string. */
	if (n > -1 && n < size)
	    return p;
	/* Else try again with more space. */
	if (n > -1)    /* glibc 2.1 */
	    size = n+1; /* precisely what is needed */
	else           /* glibc 2.0 */
	    size *= 2;  /* twice the old size */
	if ((p = (char*)xrealloc (p, size)) == NULL)
	    return NULL;
    }
}

char * make_message(const char *fmt, ...) {
    char * result;
    va_list ap;
    va_start(ap, fmt);
    result = vmake_message(fmt, ap);
    va_end(ap);
    return result; 
}

char * flatten_string_array(char** strings, int count, char *delimitor) {
    if(count > 0) {
	int len = 0, i, ofs;
	char * result;
	for(i = 0; i < count; i++) 
	    len += strlen(strings[i])+1;
	
	result = (char*) xmalloc( sizeof(char) * len + 1 );
	ofs = 0;
	for(i = 0; i < count; i++) {
	    if(i!=0) ofs += strlen(strings[i-1]) + 1;
	    sprintf(result + ofs, "%s%s", strings[i], delimitor);
	}
	return result;
    }
    else return "";
}

static char * _expand_ref_at_pos(
	 char * (*expand_ref_func)(char*, void*), /* expansion function (char*, void*) -> (char*) */
	 void * func_params,
	 char * identifier_chars,  /* characters that are part of reference identifier */
	 char * str, 
	 int pos) {

    int len = strlen(str);
    int endpos;
    /*printf(" pos=%d", pos);*/
    assert(pos >= 0 && pos < len && "This is a bug in expand_refs()" != 0);
    /* when $ is in last position, reference name is missing => nothing to do */
    if(pos==len-1) {
	char * result = xstrdup(str);
	result[len - 1] = '\0';
	return result;
    }
    else {
	char * ev_val;
	char * ev_name;
	int  ev_name_len;
	char * result, *tmp;

	pos = pos+1;
	endpos = pos+1; /* one position past $ */
	while( endpos<len && 
	       (isalpha(str[endpos]) || isdigit(str[endpos]) || strchr(identifier_chars, str[endpos])!=NULL))
	    ++endpos;

	ev_name_len = endpos - pos;
	ev_name = (char*) xmalloc(sizeof(char) * (ev_name_len+1)); /* length of name + 1 (for \0 char) */
	strncpy(ev_name, (char*) str + pos, ev_name_len);
	ev_name[ev_name_len] = '\0';
	/*printf(" ev_name='%s'\n", ev_name);*/

	ev_val = expand_ref_func(ev_name, func_params);

	tmp = xstrdup(str);
	tmp[pos-1] = '\0';

	result = make_message("%s%s%s", tmp, ev_val, (char*) tmp + endpos);

	xfree(tmp);
	xfree(ev_name);
	return result;
    }
}

char * expand_refs(char delimitor, /* reference identifier start delimitor */
		   char* (*expand_ref_func)(char*,void*), /* expansion function (char*,void*) -> (char*) */
		   void* func_params, /* void* parameter to expand_ref_func */
		   char* identifier_chars, /* characters that are part of reference identifier */
		   char* st) {
	char * posptr = 0;
	char * tmp = 0;
	char * expanded;
	tmp = xstrdup(st);
	posptr = strchr(tmp, delimitor);
	/*printf("posptr=%d", posptr);*/
	while(posptr!=NULL) {
	    int pos = posptr - tmp;
	    expanded = _expand_ref_at_pos(expand_ref_func, 
					  func_params,
					  identifier_chars,
					  tmp,
					  pos);
	    xfree(tmp);
	    tmp = expanded;
	    /* if the expanded reference starts with delimitor, we don't
	       want to catch it again */
	    posptr = strchr(tmp+pos+1, delimitor);
	}
	return tmp;
}

/** Create a name for a temporary file. Storage is automatically
 * allocated. So the result has to be xfree'd, to prevent a leak 
 *
 * This is a tricky function to implement. On modern UNIX systems
 * mkstemp is available, which is ideal. However, on Win32 and older
 * Unices, there is no mkstemp. In these cases we use tmpnam() which
 * works differently, and is more dangerous. 
 */

char * sys_tmpname (char *dir, char *pathsep, char *mytemplate) 
{

#ifdef WIN32 /* in Windows, _tempnam should do the same as mkstemp */
    char *name;

	if(!(name = _tempnam(dir, mytemplate)))
        sys_fatal(EXIT_CMDLINE, "Tempname generation failed!");

    return name;
#else
#ifdef HAVE_MKSTEMP
    int fd;
    char * full_template = make_message("%s%s%s", dir, pathsep, mytemplate);
    fd = mkstemp(full_template);
    if(fd==-1) 
	{
		sys_err("mytemplate='%s'\n", full_template);
		xfree(full_template);
	
		Throw exc(ERR_IO_TMPNAME);
	
		return NULL; /* never reached */
    }
    else {
	/* FIXME: unlinking the result looks bad */
	/* we are required to generate the name, but mkstemp actually      */
	/* creates an empty file                                           */
	/*#if ! defined(__CYGWIN32__), on older versions of cygwin, we actually need this,
	  but I don't remember which versions are these. Older cygwins have buggy mkstemp */
	sys_msg("Unlinking %s\n", full_template);
	unlink(full_template); 
	close(fd); 
	/*#endif*/
	return full_template;
    }
#else
    char * result = xmalloc(L_tmpnam);
    char * retval;
    retval = tmpnam(result);
    if(retval==NULL) {
	xfree(result);
	Throw exc(ERR_IO_TMPNAME);
	return NULL; /* never reached */
    }
    else {
	return result;
    }
#endif
#endif
}


/* File names should always be quotified inside shell commands, if
   they contain spaces. */

char *file_quotify(const char * fname) { 
    if(strchr(fname, ' ')!=NULL) return quotify(fname); 
    else return xstrdup(fname); /* so that callee always has to xfree the result */ 
}
char *file_quotify_static(char * fname) { 
    if(strchr(fname, ' ')!=NULL) return quotify_static(fname); 
    else return fname; 
}
char *file_quotify_realloc(char ** fnamep) { 
    if(strchr(*fnamep, ' ')!=NULL) return quotify_realloc(fnamep); 
    else return *fnamep;
}

/* Under Windows we need to quotify the command (Franz, why?) */
#ifdef QUOTIFY_SHELL_COMMAND 
char *command_quotify(const char * command) { return quotify(command); }
char *command_quotify_static(char * command) { return quotify_static(command); }
char *command_quotify_realloc(char ** commandp) { return quotify_realloc(commandp); }
#else
char *command_quotify(const char * command) { return xstrdup(command); }
char *command_quotify_static(char * command) { return command; }
char *command_quotify_realloc(char ** commandp) { return *commandp; }
#endif


char *quotify(const char *str) {
    char *buf = (char*) xmalloc(strlen(str) + 3);
    sprintf(buf, "%c%s%c", QUOTIFY_CHAR, str, QUOTIFY_CHAR);
    return buf;
}

char *quotify_static(char *str) {
    static char *buf = NULL;
    if(buf!=NULL)
	xfree(buf);
    buf = quotify(str);
    return buf;
}
char *quotify_realloc(char **str){
    if(*str==NULL) {
	*str = make_message("%c%c", QUOTIFY_CHAR, QUOTIFY_CHAR);
    }
    else {
	char * buf = quotify(*str);
	xfree(*str);
	*str = buf;
    }
    return *str;
}

/* -*- Emacs -*- 
Local Variables:
c-basic-offset:4
fill-column:75
End: 
*/
