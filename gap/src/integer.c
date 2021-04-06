/****************************************************************************
**
*A  integer.c                   GAP source                   Martin Schoenert
**                                                           & Alice Niemeyer
**                                                           & Werner  Nickel
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file implements the  functions  handling  arbitrary  size  integers.
**
**  There are three integer types in GAP: 'T_INT', 'T_INTPOS' and 'T_INTNEG'.
**  Each integer has a unique representation, e.g., an integer  that  can  be
**  represented as 'T_INT' is never  represented as 'T_INTPOS' or 'T_INTNEG'.
**
**  'T_INT' is the type of those integers small enough to fit into  29  bits.
**  Therefor the value range of this small integers is: $-2^{28}...2^{28}-1$.
**  This range contains about 99\% of all integers that usually occur in GAP.
**  (I just made up this number, obviously it depends on the application  :-)
**  Only these small integers can be used as index expression into sequences.
**
**  'T_INTPOS' and 'T_INTPOS' are the types of positive  respective  negative
**  integer values  that  can  not  be  represented  by  immediate  integers.
**
*/

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic memory manager          */
#include        "scanner.h"             /* reading of tokens and printing  */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* declaration part of the package */
#include        "gstring.h"
#include        "objects.h"
#include		"string4.h"
#include        "integer4.h"            /* low-level integer implementation
					   (from GAP4)                     */

/****************************************************************************
**
*F  INT_TO_HD( <INT> )  . . .  convert a small integer to an immediate handle
**
**  'INT_TO_HD' converts the integer <INT> which should be  small  enough  to
**  fit into 29  bits,  into  the  corresponding  immediate  integer  handle.
**
**  Applying this to an  integer  outside  $-2^{28}...2^{28}-1$  gives random
**  results.
**
**  'INT_TO_HD' is defined in the declaration file of the package as follows:
**
**  #define INT_TO_HD(INT)  ((Bag) (((long)INT << 2) + T_INT))
*/


/****************************************************************************
**
*F  HD_TO_INT( <HD> ) . . . .  convert an immediate handle to a small integer
**
**  'HD_TO_INT' converts the handle <HD> which should be an immediate integer
**  handle into the value of the integer constant represented by this handle.
**
**  Applying this to a non immediate integer  handle  gives  random  results.
**
**  'HD_TO_INT' is defined in the declaration file of the package as follows:
**
**  #define HD_TO_INT(HD)   (((long)HD) >> 2)
*/

#define IS_INT(x) (GET_TYPE_BAG(x) == T_INT || \
		   GET_TYPE_BAG(x) == T_INTPOS || \
		   GET_TYPE_BAG(x) == T_INTNEG)

/****************************************************************************
**
*F  FunIsInt( <hdCall> )  . . . . . . . . . . . . . internal function 'IsInt'
**
**  'FunIsInt' implements the internal function 'IsInt'.
**
**  'IsInt( <obj> )'
**
**  'IsInt'  returns 'true' if the object <obj> is  an  integer  and  'false'
**  otherwise.  May cause an error if <obj> is an unbound variable.
*/
Bag       FunIsInt (Bag hdCall)
{
    Bag           hdObj;

    /* evaluate and check the argument                                     */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: IsInt( <obj> )",0,0);
    hdObj = EVAL( PTR_BAG(hdCall)[1] );
    if ( hdObj == HdVoid )
        return Error("IsInt: function must return a value",0,0);

    /* return 'true' if <obj> is an integer and 'false' otherwise          */
    if (IS_INT(hdObj))
        return HdTrue;
    else
        return HdFalse;
}

/****************************************************************************
**
*F  FunQuoInt( <hdCall> ) . . . . . . . . . . . .  internal function 'QuoInt'
**
**  'FunQuo' implements the internal function 'QuoInt'.
**
**  'QuoInt( <i>, <k> )'
**
**  'Quo' returns the  integer part of the quotient  of its integer operands.
**  If <i>  and <k> are  positive 'Quo( <i>,  <k> )' is  the largest positive
**  integer <q>  such that '<q> * <k>  \<= <i>'.  If  <i> or  <k> or both are
**  negative we define 'Abs( Quo(<i>,<k>) ) = Quo( Abs(<i>), Abs(<k>) )'  and
**  'Sign( Quo(<i>,<k>) ) = Sign(<i>) * Sign(<k>)'.  Dividing by 0  causes an
**  error.  'Rem' (see "Rem") can be used to compute the remainder.
*/
Bag       FunQuo (Bag hdCall)
{
    register Bag  hdL, hdR;       /* left and right operand          */

    /* check the number and types of arguments                             */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )
        return Error("usage: QuoInt( <int>, <int> )",0,0);
    hdL = EVAL( PTR_BAG(hdCall)[1] );
    if ( !IS_INT(hdL) )
        return Error("usage: QuoInt( <int>, <int> )",0,0);
    hdR = EVAL( PTR_BAG(hdCall)[2] );
    if ( !IS_INT(hdR) )
        return Error("usage: QuoInt( <int>, <int> )",0,0);

    /* return the quotient                                                 */
    return QuoInt( hdL, hdR );
}

/****************************************************************************
**
*F  FunRemInt( <hdCall> ) . . . . . . . . . . . .  internal function 'RemInt'
**
**  'FunRem' implements the internal function 'RemInt'.
**
**  'RemInt( <i>, <k> )'
**
**  'Rem' returns the remainder of its two integer operands,  i.e., if <k> is
**  not equal to zero 'Rem( <i>, <k> ) = <i> - <k> *  Quo( <i>, <k> )'.  Note
**  that the rules given  for 'Quo' (see "Quo") imply  that 'Rem( <i>, <k> )'
**  has the same sign as <i> and its absolute value is strictly less than the
**  absolute value of <k>.  Dividing by 0 causes an error.
*/
Bag       FunRem (Bag hdCall)
{
    register Bag  hdL, hdR;       /* left and right operand          */

    /* check the number and types of arguments                             */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )
        return Error("usage: RemInt( <int>, <int> )",0,0);
    hdL = EVAL( PTR_BAG(hdCall)[1] );
    if ( !IS_INT(hdL) ) return Error("usage: RemInt( <int>, <int> )",0,0);
    hdR = EVAL( PTR_BAG(hdCall)[2] );
    if ( !IS_INT(hdR) ) return Error("usage: RemInt( <int>, <int> )",0,0);

    /* return the remainder                                                */
    return RemInt( hdL, hdR );
}


/****************************************************************************
**
*F  FunGcdInt( <hdCall> ) . . . . . . . . . . . .  internal function 'GcdInt'
**
**  'FunGcd' implements the internal function 'GcdInt'.
**
**  'GcdInt( <i>, <k> )'
**
**  'Gcd'  returns the greatest common divisor   of the two  integers <m> and
**  <n>, i.e.,  the  greatest integer that  divides  both <m>  and  <n>.  The
**  greatest common divisor is never negative, even if the arguments are.  We
**  define $gcd( m, 0 ) = gcd( 0, m ) = abs( m )$ and $gcd( 0, 0 ) = 0$.
*/
Bag       FunGcdInt (Bag hdCall)
{
    Bag  hdL, hdR;                /* left and right operand          */

    /* check the number and types of arguments                             */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )
        return Error("usage: GcdInt( <int>, <int> )",0,0);
    hdL = EVAL( PTR_BAG(hdCall)[1] );
    if ( !IS_INT(hdL) ) return Error("usage: GcdInt( <int>, <int> )",0,0);
    hdR = EVAL( PTR_BAG(hdCall)[2] );
    if ( !IS_INT(hdR) ) return Error("usage: GcdInt( <int>, <int> )",0,0);

    /* return the gcd                                                      */
    return GcdInt( hdL, hdR );
}

/****************************************************************************
**
*F FunLog2Int( <int> )  . . . . . . . . . . . . . internal function 'Log2Int'
**
** Implements internal function 'Log2Int'.
**
** 'Log2Int' returns number of bits of integer - 1
*/
Bag       FunLog2Int ( Bag hdCall )
{
    char * usage = "usage: Log2Int( <integer> )";
    Bag hdInteger;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hdInteger = EVAL( PTR_BAG(hdCall)[1] );
    if( ! IS_INT(hdInteger) ) return Error(usage,0,0);

    return Log2Int(hdInteger);
}

/****************************************************************************
**
*F FunStringInt( <int> ) . . . . . . . . . . .  internal function 'StringInt'
**
** Implements internal function 'StringInt'.
**
** 'StringInt' returns a string representing the integer <int>
*/
Bag       FunStringInt ( Bag hdCall )
{
    char * usage = "usage: StringInt( <integer> )";
    Bag hdInteger;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hdInteger = EVAL( PTR_BAG(hdCall)[1] );
    if( ! IS_INT(hdInteger) ) return Error(usage,0,0);

    return StringInt(hdInteger);
}

/****************************************************************************
**
*F FunHexStringInt( <int> )  . . . . . . . . internal function 'HexStringInt'
**
** Implements internal function 'HexStringInt'.
**
** 'HexStringInt' returns a hex string representing the integer <int>
*/
Bag       FunHexStringInt ( Bag hdCall )
{
    char * usage = "usage: HexStringInt( <integer> )";
    Bag hdInteger;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hdInteger = EVAL( PTR_BAG(hdCall)[1] );
    if( ! IS_INT(hdInteger) ) return Error(usage,0,0);

    return HexStringInt(hdInteger);
}

/****************************************************************************
**
*F FunIntHexString( <string> ) . . . . . . . internal function 'IntHexString'
**
** Implements internal function 'IntHexString'.
**
** `FuncHexStringInt'  takes  string in  hexadecimal notation and constructs
** the  corresponding integer.  Leading '-' is allowed, and a..f or A..F for 
** digits 10-15.
*/
Bag       FunIntHexString ( Bag hdCall )
{
    char * usage = "usage: IntHexString( <string> )";
    Bag hdString;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hdString = EVAL( PTR_BAG(hdCall)[1] );
    if( ! IsString(hdString) ) return Error(usage,0,0);

    return IntHexString(hdString);
}

Obj       FunBinShl ( Obj hdCall ) {
    Obj  hdL, hdR;       /* left and right operand          */
    char * usage = "usage: BinShl( <int>, <int> )";
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD ) return Error(usage,0,0);
    hdL = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hdL) != T_INT /*!IS_INT(hdL)*/ ) return Error(usage,0,0);
    hdR = EVAL( PTR_BAG(hdCall)[2] );
    if ( GET_TYPE_BAG(hdR) != T_INT /*!IS_INT(hdR)*/ ) return Error(usage,0,0);
    return INT_TO_HD( HD_TO_INT(hdL) << HD_TO_INT(hdR) );
}
Obj       FunBinShr ( Obj hdCall ) {
    Obj  hdL, hdR;       /* left and right operand          */
    char * usage = "usage: BinShr( <int>, <int> )";
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD ) return Error(usage,0,0);
    hdL = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hdL) != T_INT /*!IS_INT(hdL)*/ ) return Error(usage,0,0);
    hdR = EVAL( PTR_BAG(hdCall)[2] );
    if ( GET_TYPE_BAG(hdR) != T_INT /*!IS_INT(hdR)*/ ) return Error(usage,0,0);
    return INT_TO_HD( HD_TO_INT(hdL) >> HD_TO_INT(hdR) );
}

/* returns the parity of the 32 bit number passed in */
Obj       FunBinParity ( Obj hdCall ) {
	Obj  hd;
	unsigned int i;
	char * usage = "usage: BinParity(<int>)";

	if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage,0,0);

	hd = EVAL( PTR_BAG(hdCall)[1] );

	if ( GET_TYPE_BAG(hd) != T_INT ) return Error(usage,0,0);

	i = HD_TO_INT(hd);
	i ^= i >> 16;
	i ^= i >> 8;
	i ^= i >> 4;
	i &= 0xf;
	i = (0x6996 >> i) & 1;

	return INT_TO_HD(i);
}

Obj       FunBinAnd ( Obj hdCall ) {
    Obj  hdL, hdR;       /* left and right operand          */
    char * usage = "usage: BinAnd( <int>, <int> )";
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD ) return Error(usage,0,0);
    hdL = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hdL) != T_INT /*!IS_INT(hdL)*/ ) return Error(usage,0,0);
    hdR = EVAL( PTR_BAG(hdCall)[2] );
    if ( GET_TYPE_BAG(hdR) != T_INT /*!IS_INT(hdR)*/ ) return Error(usage,0,0);
    return INT_TO_HD( HD_TO_INT(hdL) & HD_TO_INT(hdR) );
}
Obj       FunBinOr ( Obj hdCall ) {
    Obj  hdL, hdR;       /* left and right operand          */
    char * usage = "usage: BinOr( <int>, <int> )";
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD ) return Error(usage,0,0);
    hdL = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hdL) != T_INT /*!IS_INT(hdL)*/ ) return Error(usage,0,0);
    hdR = EVAL( PTR_BAG(hdCall)[2] );
    if ( GET_TYPE_BAG(hdR) != T_INT /*!IS_INT(hdR)*/ ) return Error(usage,0,0);
    return INT_TO_HD( HD_TO_INT(hdL) | HD_TO_INT(hdR) );
}
Obj       FunBinXor ( Obj hdCall ) {
    Obj  hdL, hdR;       /* left and right operand          */
    char * usage = "usage: BinXor( <int>, <int> )";
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD ) return Error(usage,0,0);
    hdL = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hdL) != T_INT /*!IS_INT(hdL)*/ ) return Error(usage,0,0);
    hdR = EVAL( PTR_BAG(hdCall)[2] );
    if ( GET_TYPE_BAG(hdR) != T_INT /*!IS_INT(hdR)*/ ) return Error(usage,0,0);
    return INT_TO_HD( HD_TO_INT(hdL) ^ HD_TO_INT(hdR) );
}
Obj       FunBinNot ( Obj hdCall ) {
    Obj  hd; 
    char * usage = "usage: BinNot( <int> )";
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage,0,0);
    hd = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hd) != T_INT /*!IS_INT(hd)*/ ) return Error(usage,0,0);
    return INT_TO_HD( ~ HD_TO_INT(hd) );
}

/****************************************************************************
**
*F  InitInt() . . . . . . . . . . . . . . . . initializes the integer package
**
**  'InitInt' initializes the arbitrary size integer package.
*/
void            InitInt (void)
{
    /* Install operator handlers and such                                  */
    InitIntImplementation();

    /* Install the internal functions                                      */
    InstIntFunc( "IsInt",         FunIsInt        );
    InstIntFunc( "QuoInt",        FunQuo          );
    InstIntFunc( "RemInt",        FunRem          );
    InstIntFunc( "GcdInt",        FunGcdInt       );

    InstIntFunc( "HexStringInt",  FunHexStringInt );
    InstIntFunc( "IntHexString",  FunIntHexString );
    InstIntFunc( "Log2Int",       FunLog2Int      );

    InstIntFunc( "StringInt",     FunStringInt    );

    InstIntFunc( "BinShr",        FunBinShr     );
    InstIntFunc( "BinShl",        FunBinShl     );
    InstIntFunc( "BinParity",     FunBinParity  );
    InstIntFunc( "BinAnd",        FunBinAnd     );
    InstIntFunc( "BinOr",         FunBinOr      );
    InstIntFunc( "BinXor",        FunBinXor     );
    InstIntFunc( "BinNot",        FunBinNot     );
}



