/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */
 
#include <string.h>

#include "errcodes.h"
#include "vector_def.h"
#include "sys.h"

/**
 * Type records definitions for primitive REAL types 
 */

PRIMITIVE_REAL_PACKAGE(long double, long_double,
		       "%Lg", "FloatString(\"%.22Lg\")", 
		       -50 + 100*((long double)rand() / RAND_MAX));

PRIMITIVE_REAL_PACKAGE(double,      double,
		       "%g", "FloatString(\"%.18g\")",
		       -50 + 100*((double)rand() / RAND_MAX));

PRIMITIVE_REAL_PACKAGE(float,       float,
		       "%g", "FloatString(\"%.9g\")",
		       -50 + 100*((float)rand() / RAND_MAX));

/**
 * Type records definitions for primitive COMPLEX types 
 */
/* Since "cplx" types represent complex numbers as 2 adjacent values
   in array, functions taking them as arguments will have same
   signatures as functions with real arguments of the same type. The
   typedefs in stub header make this transparent. */

PRIMITIVE_CPLX_PACKAGE(long double, long_double,
		       "(%Lg, %Lg)", "Complex(FloatString(\"%.22Lg\"), FloatString(\"%.22Lg\"))",
		       -50 + 100*((long double)rand() / RAND_MAX), sqrt);

PRIMITIVE_CPLX_PACKAGE(double,      double,
		       "(%g, %g)", "Complex(FloatString(\"%.18g\"), FloatString(\"%.18g\"))",
		       -50 + 100*((double)rand() / RAND_MAX), sqrt);

PRIMITIVE_CPLX_PACKAGE(float,       float,
		       "(%g, %g)", "Complex(FloatString(\"%.9g\"), FloatString(\"%.9g\"))",
		       -50 + 100*((float)rand() / RAND_MAX), sqrt);

/* vector definitions for fixed point real types */

PRIMITIVE_FPREAL_PACKAGE(int, int,
                         "%g", "FloatString(\"%.22g\")", 
			 -50 + 100*((double)rand() / RAND_MAX));

PRIMITIVE_FPREAL_PACKAGE(short, short,
                         "%g", "FloatString(\"%.22g\")", 
			 -50 + 100*((double)rand() / RAND_MAX));

PRIMITIVE_FPREAL_PACKAGE(char, char,
                         "%g", "FloatString(\"%.22g\")", 
			 -50 + 100*((double)rand() / RAND_MAX));

/* vector definitions for fixed point complex types */

PRIMITIVE_FPCPLX_PACKAGE(int, int,
			 "(%g, %g)", "Complex(FloatString(\"%.9g\"), FloatString(\"%.9g\"))",
			 -50 + 100*((double)rand() / RAND_MAX), sqrt);

PRIMITIVE_FPCPLX_PACKAGE(short, short,
			 "(%g, %g)", "Complex(FloatString(\"%.9g\"), FloatString(\"%.9g\"))",
			 -50 + 100*((double)rand() / RAND_MAX), sqrt);

PRIMITIVE_FPCPLX_PACKAGE(char, char,
			 "(%g, %g)", "Complex(FloatString(\"%.9g\"), FloatString(\"%.9g\"))",
			 -5 + 5*((double)rand() / RAND_MAX), sqrt);

#ifdef IPP
PRIMITIVE_PLACEHOLDER(Ipp16, Ipp16,
			"", "",
			0, sqrt);

PRIMITIVE_PLACEHOLDER(Ipp32, Ipp32,
		      "", "",
		      0, sqrt);
#endif


extern scalar_type_t __char_record;
extern scalar_type_t __short_record;
extern scalar_type_t __int_record;
extern scalar_type_t __char_cplx_record;
extern scalar_type_t __short_cplx_record;
extern scalar_type_t __int_cplx_record;

scalar_type_t * SCALAR_TYPES[] = {
    &__long_double_record,
    &__double_record,
    &__float_record,

    &__long_double_cplx_record,
    &__double_cplx_record,
    &__float_cplx_record,
    
#ifdef IPP
    &__Ipp16_cplx_record,  /* ONLY FOR ARM */
    &__Ipp32_cplx_record,  /* ONLY FOR ARM */
#endif
    &__int_record,
    &__int_cplx_record,
    &__short_record,
    &__short_cplx_record,
    &__char_record,
    &__char_cplx_record,

    (scalar_type_t*) NULL
};

extern int sys_setenv(char *key, char *value, int i);

EXPORT scalar_type_t * scalar_find_type(char * name) 
{
#define TMPBUFLEN 100
	scalar_type_t ** typ = SCALAR_TYPES;
	char *fp_str;
	int fracbits = -1;
	char tmpbuf[TMPBUFLEN];

	if(strlen(name) >= TMPBUFLEN) sys_fatal(EXIT_CMDLINE, "name too long!");

	/* check if fixed point */
	fp_str = strstr(name, "fp");
	if(fp_str)
	{
		/* pull out the fractional bits */
		fracbits = atoi(fp_str+2);
		strncpy(tmpbuf, name, fp_str - name + 2);
		tmpbuf[fp_str - name + 2] = 0;
	}
	else
		strcpy(tmpbuf, name);

	/* traverse typ array and find the right one. */
	for( ; *typ != NULL && strcmp((*typ)->name, tmpbuf); ++typ);

	/* fixup name and add type info */
	if(*typ && fracbits != -1)
	{
		(*typ)->fracbits = fracbits;
		(*typ)->zerobits = 8*((*typ)->size) - fracbits;
	}

	return *typ;
}


/* -*- Emacs -*- 
Local Variables:
c-basic-offset:4
fill-column:75
End: 
*/
