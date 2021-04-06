/****************************************************************************
**
*A  unknown.c                   GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This  file implementes the  arithmetic  for unknown values,  unknowns for
**  short.  Unknowns  are written as 'Unknown(<n>)'  where  <n> is an integer
**  that distuingishes different unknowns.  Every unknown stands for a fixed,
**  well defined, but unknown  scalar value,  i.e., an  unknown  integer,  an
**  unknown rational, or an unknown cyclotomic.
**
**  Being unknown is a contagious property.  That is  to say  that the result
**  of  a scalar operation involving an  unknown is   also  unknown, with the
**  exception of multiplication by 0,  which  is  0.  Every scalar  operation
**  involving an  unknown operand is  a  new  unknown, with  the exception of
**  addition of 0 or multiplication by 1, which is the old unknown.
**
**  Note that infinity is not regarded as a well defined scalar value.   Thus
**  an unknown never stands for infinity.  Therefor division by 0 still gives
**  an  error, not an unknown.  Also  division by an  unknown gives an error,
**  because the unknown could stand for 0.
**
*/

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage management      */
#include        "scanner.h"             /* reading of tokens and printing  */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */

#include        "unknown.h"             /* declaration part of the package */


/****************************************************************************
**
*V  LargestUnknown  . . . . . . . .  largest used index for an unknown, local
**
**  'LargestUnknown' is the largest <n> that is used in  any  'Unknown(<n>)'.
**  This is used in 'NewUnknown' which increments this value  when  asked  to
**  make a new unknown.
*/
Int            LargestUnknown;


/****************************************************************************
**
*F  EvUnknown( <hdUnd> )  . . . . . . . . . . . . . . . . evaluate an unknown
**
**  'EvUnknown' returns the value of the unknown <hdUnd>.  Since unknowns are
**  constants and thus selfevaluating this simply returns <hdUnd>.
*/
Bag       EvUnknown (Bag hdUnk)
{
    return hdUnk;
}


/****************************************************************************
**
*F  NewUnknown()  . . . . . . . . . . . . . . . . .  make a new unknown value
**
**  'NewUnknown' returns a new unknown value, i.e., 'Unknown(<n>)' where  <n>
**  is an integer previously not used.
*/
Bag       NewUnknown (void)
{
    Bag           hdUnk;

    LargestUnknown++;
    hdUnk = NewBag( T_UNKNOWN, sizeof(Int) );
    ((Int*)PTR_BAG(hdUnk))[0] = LargestUnknown;
    return hdUnk;
}


/****************************************************************************
**
*F  SumUnknown( <hdL>, <hdR> )  . . . . . . . . . . . . . sum of two unknowns
**
**  'SumUnknown' returns  the  sum  of  the  two  unknowns <hdL>  and  <hdR>.
**  Either operand may also be a known scalar value.
**
**  Is called from the 'Sum' binop, so both operands are already evaluated.
*/
Bag       SumUnknown (Bag hdL, Bag hdR)
{
    if ( hdL == INT_TO_HD(0) )
        return hdR;
    else if ( hdR == INT_TO_HD(0) )
        return hdL;
    else
        return NewUnknown();
}


/****************************************************************************
**
*F  DiffUnknown( <hdL>, <hdR> ) . . . . . . . . .  difference of two unknowns
**
**  'DiffUnknown' returns the difference of the two unknowns <hdL> and <hdR>.
**  Either operand may also be a known scalar value.
**
**  Is called from the 'Diff' binop, so both operands are already evaluated.
*/
Bag       DiffUnknown (Bag hdL, Bag hdR)
{
    if ( hdR == INT_TO_HD(0) )
        return hdL;
    else if ( GET_TYPE_BAG(hdL) == T_UNKNOWN && GET_TYPE_BAG(hdR) == T_UNKNOWN
           && ((Int*)PTR_BAG(hdL))[0] == ((Int*)PTR_BAG(hdR))[0] )
        return INT_TO_HD(0);
    else
        return NewUnknown();
}


/****************************************************************************
**
*F  ProdUnknown( <hdL>, <hdR> ) . . . . . . . . . . . product of two unknowns
**
**  'ProdUnknown' returns the product of the two  unknowns  <hd>  and  <hdR>.
**  Either operand may also be a known scalar value.
**
**  Is called from the 'Prod' binop, so both operands are already evaluated.
*/
Bag       ProdUnknown (Bag hdL, Bag hdR)
{
    if ( hdL == INT_TO_HD(0) || hdR == INT_TO_HD(0) )
        return INT_TO_HD(0);
    else if ( hdL == INT_TO_HD(1) )
        return hdR;
    else if ( hdR == INT_TO_HD(1) )
        return hdL;
    else
        return NewUnknown();
}


/****************************************************************************
**
*F  QuoUnknown( <hdL>, <hdR> )  . . . . . . . . . .  quotient of two unknowns
**
**  'QuoUnknown' returns the quotient of the unknown  <hdL>  and  the  scalar
**  <hdR>.  <hdR> must not be zero, and must not be an unknown,  because  the
**  unknown could stand for zero.
**
**  Is called from the 'Quo' binop, so both operands are already evaluated.
*/
Bag       QuoUnknown (Bag hdL, Bag hdR)
{
    if ( hdR == INT_TO_HD(0) )
        return Error("divisor must be nonzero", 0, 0);
    else if ( GET_TYPE_BAG(hdR) == T_UNKNOWN )
        return Error("divisor must no be unknown (could be zero)", 0, 0);
    else if ( hdR == INT_TO_HD(1) )
        return hdL;
    else
        return NewUnknown();
}


/****************************************************************************
**
*F  PowUnknown( <hdL>, <hdR> )  . . . . . . . . . . . . . power of an unknown
**
**  'PowUnknown' returns the unknown <hdL> raised to the integer power <hdR>.
**  If <hdR> is 0, the result is the integer 1.  If <hdR> must  not  be  less
**  than 0, because <hdL> could stand for 0.
**
**  Is called from the 'Pow' binop, so both operands are already evaluted.
*/
Bag       PowUnknown (Bag hdL, Bag hdR)
{
    if ( hdR == INT_TO_HD(0) )
        return INT_TO_HD(1);
    else if ( HD_TO_INT(hdR) < 0 )
        return Error("divisor must not be unknown (could be zero)", 0, 0);
    else
        return NewUnknown();
}


/****************************************************************************
**
*F  EqUnknown( <hdL>, <hdR> ) . . . . . .  . . test if two unknowns are equal
**
**  'EqUnknown' returns 'true' if the two unknowns <hdL> and <hdR>  are equal
**  and 'false' otherwise.
**
**  Note that 'EqUnknown' assumes that two unknowns with  different  <n>  are
**  different.  I dont like this at all.
**
**  Is called from 'EvEq' binop, so both operands are already evaluated.
*/
Bag       EqUnknown (Bag hdL, Bag hdR)
{
    if ( ((Int*)PTR_BAG(hdL))[0] == ((Int*)PTR_BAG(hdR))[0] )
        return HdTrue;
    else
        return HdFalse;
}


/****************************************************************************
**
*F  LtUnknown( <hdL>, <hdR> ) . . .  test if one unknown is less than another
**
**  'LtUnknown' returns 'true' if the unknown <hdL> is less than the  unknown
**  <hdR>  are equal and 'false' otherwise.
**
**  Note that 'LtUnknown' assumes that two unknowns with  different  <n>  are
**  different.  I dont like this at all.
**
**  Is called from 'EvLt' binop, so both operands are already evaluated.
*/
Bag       LtUnknown (Bag hdL, Bag hdR)
{
    if ( ((Int*)PTR_BAG(hdL))[0] < ((Int*)PTR_BAG(hdR))[0] )
        return HdTrue;
    else
        return HdFalse;
}


/****************************************************************************
**
*F  PrUnknown( <hdUnk> )  . . . . . . . . . . . . . . . . .  print an unknown
**
**  'PrUnknown' prints the unknown <hdUnk> in the form 'Unknown(<n>)'.
*/
void            PrUnknown (Bag hdUnk)
{
    Pr("%>Unknown(%d)%<",((Int*)PTR_BAG(hdUnk))[0], 0);
}


/****************************************************************************
**
*F  FunUnknown( <hdCall> )  . . . . . . . . . . . . . . . . create an unknown
**
**  'FunUnknown' implements the internal function 'Unknown'.
**
**  'Unknown()'\\
**  'Unknown(<n>)'
**
**  In the first form 'Unknown' returns a new unknown 'Unknown(<n>)'  with  a
**  <n> that was not previously used.
**
**  In the second form 'Unknown' returns the unknown 'Unknown(<n>)'.
*/
Bag       FunUnknown (Bag hdCall)
{
    Bag           hdUnk;
    Int                n;

    /* check the arguments                                                 */
    if ( GET_SIZE_BAG(hdCall) != SIZE_HD && GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
        return Error("usage: Unknown() or Unknown(<n>)", 0, 0);

    /* get and check <n>                                                   */
    if ( GET_SIZE_BAG(hdCall) == 2 * SIZE_HD ) {
        hdUnk = EVAL( PTR_BAG(hdCall)[1] );
        if ( GET_TYPE_BAG(hdUnk) != T_INT || HD_TO_INT(hdUnk) <= 0 )
            return Error("Unknown: <n> must be a positive integer", 0, 0);
        n = HD_TO_INT(hdUnk);
        if ( LargestUnknown < n )  LargestUnknown = n;
    }
    else {
        LargestUnknown++;
        n = LargestUnknown;
    }

    /* create and return the new unknown                                   */
    hdUnk = NewBag( T_UNKNOWN, sizeof(Int) );
    ((Int*)PTR_BAG(hdUnk))[0] = n;
    return hdUnk;
}


/****************************************************************************
**
*F  FunIsUnknown( <hdCall> )  . . . . . . . .  test if an object is a unknown
**
**  'FunIsUnknown' implements the internal function 'IsUnknown'.
**
**  'IsUnknown( <obj> )'
**
**  'IsUnknown' returns 'true' if the object <obj> is an unknown and  'false'
**  otherwise.  Will cause an error if <obj> is an unbound variable.
*/
Bag       FunIsUnknown (Bag hdCall)
{
    Bag           hdObj;          /* handle of the object            */

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: IsUnknown( <obj> )", 0, 0);
    hdObj = EVAL( PTR_BAG(hdCall)[1] );
    if ( hdObj == HdVoid )
        return Error("IsUnknown: function must return a value", 0, 0);

    /* return 'true' if <obj> is an unknown and 'false' otherwise          */
    if ( GET_TYPE_BAG(hdObj) == T_UNKNOWN )
        return HdTrue;
    else
        return HdFalse;
}


/****************************************************************************
**
*F  InitUnknown() . . . . . . . . . . . . . .  initialize the unknown package
**
**  'InitUnknown' initializes the unknown package.
*/
void            InitUnknown (void)
{
    unsigned int        type;

    /* install the evaluation and printing function                        */
    InstEvFunc( T_UNKNOWN, EvUnknown );
    InstPrFunc( T_UNKNOWN, PrUnknown );

    /* install the binary operations                                       */
    for ( type = T_INT; type <= T_UNKNOWN; type++ ) {
        TabSum[  type ][ T_UNKNOWN ] = SumUnknown;
        TabSum[  T_UNKNOWN ][ type ] = SumUnknown;
        TabDiff[ type ][ T_UNKNOWN ] = DiffUnknown;
        TabDiff[ T_UNKNOWN ][ type ] = DiffUnknown;
        TabProd[ type ][ T_UNKNOWN ] = ProdUnknown;
        TabProd[ T_UNKNOWN ][ type ] = ProdUnknown;
        TabQuo[  type ][ T_UNKNOWN ] = QuoUnknown;
        TabQuo[  T_UNKNOWN ][ type ] = QuoUnknown;
    }
    TabPow[ T_UNKNOWN ][ T_INT ] = PowUnknown;
    TabEq[ T_UNKNOWN ][ T_UNKNOWN ] = EqUnknown;
    TabLt[ T_UNKNOWN ][ T_UNKNOWN ] = LtUnknown;

    /* and finally install the internal functions                          */
    InstIntFunc( "Unknown",   FunUnknown   );
    InstIntFunc( "IsUnknown", FunIsUnknown );
}



