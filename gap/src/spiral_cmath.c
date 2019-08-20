/* C Math Library Interface */

#include        <math.h>
#include        <stdlib.h>
#include        <stdio.h>               /* sscanf */
#include	<string.h>

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "scanner.h"             /* reading of tokens and printing  */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */
#include        "objects.h"
#include		"string4.h"
#include        "integer4.h"

#include        "list.h"                /* generic list package            */
#include        "plist.h"               /* 'LEN_PLIST', 'SET_LEN_PLIST',.. */
#include        "range.h"               /* 'LEN_RANGE', 'LOW_RANGE', ..    */
#include        "set.h"                 /* 'IsSet', 'SetList'              */

#include        "ieee754.h"      /* union ieee754_double */
#include		"GapUtils.h"


/****************************************************************************
**
*F  IsIntPair(<hdList>) . . . . . . test whether an object is an integer pair
**
**  'IsIntPair' returns 1 if the list <hdList> is a list with 2 elements, 
**  and each of these 2 is an integer.
*/
Int            IsIntPair (Bag hdList)
{
    /* hdList must be a list with 2 elements that are integers*/
    if ( !IS_LIST(hdList) ||
	 LEN_LIST(hdList) != 2 ||
	 /* exponent has to be a normal integer */
	 GET_TYPE_BAG(ELMF_LIST(hdList,2)) != T_INT )
	return 0;    
    else {
	Bag hdMantissa = ELMF_LIST(hdList, 1);
	int is_immediate = (GET_TYPE_BAG(hdMantissa) == T_INT);
	int is_long = (GET_TYPE_BAG(hdMantissa) == T_INTPOS || GET_TYPE_BAG(hdMantissa) == T_INTNEG);
	/* hdMantMax = 0x1FFFF.... = 2^52 (implicit leading one) + 
	                           = 2^52 - 1 (52-bits mantissa) =
                                   = 2^53 - 1
	   hdMantMaxPlus1 = hdMantMax + 1 = 2^53, this is needed there is
	   only have LtInt, which is "less then", and no "less-or-equal" */
	Bag hdMantMaxPlus1 = 
			 PROD(INT_TO_HD(1<<27), INT_TO_HD(1<<26));

	if(is_immediate) return 1;
	else if(is_long) {
	    if(GET_TYPE_BAG(hdMantissa)==T_INTPOS)
		return LtInt(hdMantissa, hdMantMaxPlus1) == HdTrue;
	    else
		return LtInt(PROD(hdMantMaxPlus1, INT_TO_HD(-1)),
			     hdMantissa) == HdTrue;
	}
	else return 0;
    }    
}


/****************************************************************************
**
*F  FunIsIntPair(<hdCall>)  . . . . test whether an object is an integer pair
**
**  'FunIsIntPair' implements the internal function 'IsIntPair'.
**
**  'IsIntPair( <obj> )'
**
**  'IsIntPair' returns true  if the object <obj> is a list with 2  elements, 
**  and each of these 2 is an integer.
*/
Bag       FunIsIntPair (Bag hdCall)
{
    Bag           hdList;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: IsIntPair( <obj> )",0,0);
    hdList = EVAL( PTR_BAG(hdCall)[1] );
    if ( hdList == HdVoid ) {
        return Error("IsIntPair: function must return a value",0,0);
    }
    /* let 'IsPair' do the work                                           */
    return IsIntPair( hdList ) ? HdTrue : HdFalse;
}

/****************************************************************************
**
*F  DoubleToIntPair(<hdResult>, <double>)  . . . .  convert double to IntPair
**
**  'DoubleToIntPair' converts  the  double precision  floating-point  number
**  to a   mantissa,  exponent pair,  stored  in  IntPair  handle. Result  is
**  constructed by directly modifying hdResult,  which is  assumed  to be  an
**  IntPair.
**  
*/
void            DoubleToIntPair (Bag hdResult, double d)
{
    union ieee754_double dbl;
    Int mantissa0;
    UInt mantissa1;
    Int exponent;

    Bag hdMantissa;
    Bag hdExponent;

    dbl.d = d;

    /* GAP can accept only 29-bit signed integers as immediate */
    mantissa1 = 0x0FFFFFFF & dbl.ieee.mantissa1; /* low 28 bits (unsigned) */

    /* the remaining 4 bits of mantissa1 go here, also the leading 1 is    */
    /* implicit, so it has to be added                                     */
    mantissa0 = ((0xF0000000 & dbl.ieee.mantissa1) >> 28) + 
	        (dbl.ieee.mantissa0 << 4) +
	        (1 << 24);

    exponent  = dbl.ieee.exponent - IEEE754_DOUBLE_BIAS;

    /* signed 2^28 is 30 bits, so we have to multiply by 2^27 and then by 2*/
    /* to get it                                                           */
    hdMantissa = INT_TO_HD(mantissa0);
    hdMantissa = PROD( PROD(hdMantissa, INT_TO_HD(1<<27)), INT_TO_HD(2) );
    hdMantissa = SUM( hdMantissa, INT_TO_HD(mantissa1) );
    if(dbl.ieee.negative) 
	hdMantissa = PROD( hdMantissa, INT_TO_HD(-1) );
        
    hdExponent = INT_TO_HD(-52+exponent);

    SET_ELM_PLIST(hdResult, 1, hdMantissa);
    SET_ELM_PLIST(hdResult, 2, hdExponent);
}

/****************************************************************************
**
*F  IntPairToDouble(<hdList>) . . . . . . . . . . . convert IntPair to double
**
**  'IntPairToDouble' converts the (mantissa, exponent) pair  in hdList to  a 
**  double precision floating point number.
**
**  No error checking is done, so hdList must be guaranteed to be a list of 2
**  integers of the correct size.
*/
double          IntPairToDouble(Bag hdList)
{
    Int exp = HD_TO_INT(ELMF_LIST(hdList,2));
    Bag hdMantissa = ELMF_LIST(hdList,1);
    if(GET_TYPE_BAG(hdMantissa)==T_INT) 
	return HD_TO_INT(hdMantissa) * pow(2, (double)(exp));
    else {
	double sign = +1.0;
	UInt mantissa0;
	UInt mantissa1;
	Bag hdAbsMantissa = hdMantissa;

	if(GET_TYPE_BAG(hdMantissa)==T_INTNEG) 
	    sign = -1.0;
	if(sign < 0) 
	    hdAbsMantissa = PROD(hdMantissa, INT_TO_HD(-1));

	/* most significant mantissa part */
	mantissa0 = HD_TO_INT(QuoInt(hdAbsMantissa, INT_TO_HD(1<<26)));
	/* least significant mantissa part */
	mantissa1 = HD_TO_INT(RemInt(hdAbsMantissa, INT_TO_HD(1<<26)));

	return sign * ((double)mantissa0 * pow(2,26) + (double)mantissa1) * pow(2, (double)exp);
    }	    
}

/****************************************************************************
**
*F  FunFloatStringIntPair(<hdCall>)  . . . . . .  convert IntPair to a string
**
**  'FunIsIntPair' implements the internal function 'FloatStringIntPair'.
**
**  'FloatStringIntPair( <format_string>, <int_pair> )'
**
**  'FloatStringIntPair' converts a given IntPair to a double precision floa-
**  ting  point number, and returns a  formatted string  created with sprintf
**  called with a format string and a double. 
**
**  <format_string> should not contain more than one %.
**  Example: FloatStringIntPair("%f", [1,0]) returns "1.000000",
**  since 1 * 2^0 = 0, and %f defaults to 6 digits of precision.
*/
Bag       FunFloatStringIntPair (Bag hdCall)
{
    Bag           hdIntPair, hdFmtString, hdResult;
    char * result;
    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )
        return Error("usage: FloatStringIntPair( <fmt_string>, <int_pair> )",
		     0,0);
    hdFmtString = EVAL( PTR_BAG(hdCall)[1] );
    hdIntPair   = EVAL( PTR_BAG(hdCall)[2] );
    if ( ! IsIntPair(hdIntPair) )
        return Error("usage: FloatStringIntPair( <int_pair> )",0,0);
    /*alloc*/ result = GuMakeMessage((char*) PTR_BAG(hdFmtString), 
				    IntPairToDouble(hdIntPair));
    hdResult = NewBag( T_STRING, strlen(result)+1 );
    SyStrncat( (char*)PTR_BAG(hdResult), (char*)result, strlen(result)+1);
    free(result);
    return hdResult;
}

/****************************************************************************
**
*F  IntPairString( <str> ) . . . . . . .  convert a string to IntPair (float)
**
** This  function  parses  the string and tries to convert it to  a  floating 
** point number. If it succeeds it returns an  IntPair  representation of the
** result, if it fails an Error is raised.
**
*/
Bag       FunIntPairString (Bag hdCall)
{
    char * usage = "usage: IntPairString( <str> )";
    Bag hdStr;
    char * str;
    int n;
    double dbl=0;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error(usage, 0,0);

    hdStr = EVAL( PTR_BAG(hdCall)[1] );
    if( GET_TYPE_BAG(hdStr) != T_STRING ) return Error(usage,0,0);

    /* parse the string */
    str = (char*) PTR_BAG(hdStr);
    n = sscanf(str, "%le", &dbl);
    if(n!=1) 
	return Error("Can't parse '%s' to a float", (Int)str, 0);
    else {
	Bag result = NewBag(T_LIST,  SIZE_PLEN_PLIST(2));
	SET_LEN_PLIST(result, 2);
	DoubleToIntPair(result, dbl);
	return result;
    }
}
     


/****************************************************************************
**
*F  Func1IntPair(<name>, <fPtr>, <hdCall>) . .  C function with 1 arg wrapper
**
**  'Func1IntPair' is a  wrapper for C  math functions  with 1 argument, that 
**  allows them to be called from within GAP. 
** 
**  'Func1IntPair' expects  one IntPair in  <hdCall>, and reports an error if 
**  there isn't one. <name> is the  internal  function name to be included in
**  the error message. 
**
**  If <hdCall> is valid,  then function pointed to by <fPtr> is invoked with
**  the double (obtained by converting an IntPair), and its result is conver-
**  ted back to an IntPair and returned as Bag.
**
*/
Bag       Func1IntPair (char *name, double (*fPtr) (double), Bag hdCall)
{
    Bag           hdList;
    Bag           hdResult;
    double result;
    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD)
        return Error("usage: Func1IntPair( <int_pair> )",0,0);
    hdList = EVAL( PTR_BAG(hdCall)[1] );
    if ( hdList == HdVoid )
        return Error("IsIntPair: function must return a value",0,0);
    if ( ! IsIntPair(hdList) )
        return Error("usage: Func1IntPair( <int_pair> )",0,0);

    /* calculate the desired function and return the result as (mantissa, exp) pair */
    result = fPtr(IntPairToDouble(hdList));
    hdResult = Copy(hdList);
    DoubleToIntPair(hdResult, result);
    return hdResult;
}

/****************************************************************************
**
*F  Func2IntPair(<name>, <fPtr>, <hdCall>) . . C function with 2 args wrapper
**
**  'Func2IntPair' is a wrapper for C  math functions  with 2 arguments, that 
**  allows them to be called from within GAP. 
** 
**  'Func2IntPair' expects TWO IntPairs in  <hdCall>, and reports an error if
**  <hdCall> is invalid. <name> is the  internal function name to be included
**  in the error message. 
**
**  If <hdCall> is valid,  then function pointed to by <fPtr> is invoked with
**  two double parameters (obtained by converting an IntPair), and its result
**  is converted back to an IntPair and returned as Bag.
**
*/
Bag       Func2IntPair (char *name, double (*fPtr) (double, double), Bag hdCall)
{
    Bag           hdIntPair1;
    Bag           hdIntPair2;
    Bag           hdResult;
    double result;
    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD)
        return Error("usage: Func2IntPair( <int_pair>, <int_pair> )",0,0);
    hdIntPair1 = EVAL( PTR_BAG(hdCall)[1] );
    hdIntPair2 = EVAL( PTR_BAG(hdCall)[2] );
    if ( hdIntPair1 == HdVoid || hdIntPair2 == HdVoid) 
        return Error("IsIntPair: function must return a value",0,0);
    if ( ! IsIntPair(hdIntPair1) || ! IsIntPair(hdIntPair2) )
        return Error("usage: Func2IntPair( <int_pair>, <int_pair> )",0,0);

    /* calculate the desired function and return the result as an IntPair  */
    result = fPtr(IntPairToDouble(hdIntPair1), IntPairToDouble(hdIntPair2));
    hdResult = Copy(hdIntPair1);
    DoubleToIntPair(hdResult, result);
    return hdResult;
}

/* This macro expands to a function definition FunCMath_name, that 
   implements GAP internal function CMath_name. It is a wrapper around
   C math function 'name' with one double argument, and double return
   value */
#define FUNC1(name) \
Bag  FunCMath_##name (Bag hdCall) { \
    return Func1IntPair("", &name, hdCall); \
}

/* Same as FUNC1 but for C functions with 2 double arguments. */
#define FUNC2(name) \
Bag  FunCMath_##name (Bag hdCall) { \
    return Func2IntPair("", &name, hdCall); \
}

/* Define internal GAP function wrappers */
FUNC1(exp); /* exponential of xdouble */
FUNC1(log);  /* natural logarithm of x */
FUNC1(log10) /* base-10 logarithm of x */
FUNC2(pow);  /* x raised to power y*/
FUNC1(sqrt); /* square root of x */
FUNC1(ceil); /* smallest integer not less than x */
FUNC1(floor); /* largest integer not greater than x */
FUNC1(fabs); /* absolute value of x */
FUNC2(fmod); /* if y non-zero, floating-point remainder of x/y, with same sign as x; 
		if y zero, result is implementation-defined */
FUNC1(sin);  /* sine of x */
FUNC1(cos);  /* cosine of x */
FUNC1(tan);  /* tangent of x */
FUNC1(asin); /* arc-sine of x */
FUNC1(acos); /* arc-cosine of x */
FUNC1(atan); /* arc-tangent of x */
FUNC2(atan2); /* arc-tangent of y/x */
FUNC1(sinh); /* hyperbolic sine of x */
FUNC1(cosh); /* hyperbolic cosine of x */
FUNC1(tanh); /* hyperbolic tangent of x */

/* Arithmetic */
double fadd(double x, double y) { return x+y; }
double fsub(double x, double y) { return x-y; }
double fmul(double x, double y) { return x*y; }
double fdiv(double x, double y) { return x/y; }
FUNC2(fadd);
FUNC2(fsub);
FUNC2(fmul);
FUNC2(fdiv);

/* Unimplemented math calls: 
double ldexp(double x, int n);   // x times 2 to the power n 
double frexp(double x, int* exp); // if x non-zero, returns value, 
   with absolute value in interval [1/2, 1), and assigns to *exp integer 
   such that product of return value and 2 raised to the power *exp equals
   x; if x zero, both return value and *exp are zero
double modf(double x, double* ip); // returns fractional part and assigns
   to *ip integral part of x, both with same sign as x 
*/

/****************************************************************************
**
*F  InitSPIRAL_CMath() . . . . . . . . . initializes C Math library interface
**
**  'InitSPIRA_CMath' initializes C Math library interface used by SPIRAL
*/
void InitSPIRAL_CMath(void) {
    InstIntFunc( "IsIntPair",          FunIsIntPair);

    InstIntFunc( "CMath_exp",          FunCMath_exp);
    InstIntFunc( "IntPairString",      FunIntPairString );
    InstIntFunc( "FloatStringIntPair", FunFloatStringIntPair);
    InstIntFunc( "CMath_log",          FunCMath_log);
    InstIntFunc( "CMath_log10",        FunCMath_log10);
    InstIntFunc( "CMath_pow",          FunCMath_pow);
    InstIntFunc( "CMath_sqrt",         FunCMath_sqrt);
    InstIntFunc( "CMath_ceil",         FunCMath_ceil);
    InstIntFunc( "CMath_floor",        FunCMath_floor);
    InstIntFunc( "CMath_fabs",         FunCMath_fabs);
    InstIntFunc( "CMath_fmod",         FunCMath_fmod);
    InstIntFunc( "CMath_sin",          FunCMath_sin);
    InstIntFunc( "CMath_cos",          FunCMath_cos);
    InstIntFunc( "CMath_tan",          FunCMath_tan);
    InstIntFunc( "CMath_asin",         FunCMath_asin);
    InstIntFunc( "CMath_acos",         FunCMath_acos);
    InstIntFunc( "CMath_atan",         FunCMath_atan);
    InstIntFunc( "CMath_atan2",        FunCMath_atan2);
    InstIntFunc( "CMath_sinh",         FunCMath_sinh);
    InstIntFunc( "CMath_cosh",         FunCMath_cosh);
    InstIntFunc( "CMath_tanh",         FunCMath_tanh);

    InstIntFunc( "CMath_add",          FunCMath_fadd);
    InstIntFunc( "CMath_sub",          FunCMath_fsub);
    InstIntFunc( "CMath_mul",          FunCMath_fmul);
    InstIntFunc( "CMath_div",          FunCMath_fdiv);
}
