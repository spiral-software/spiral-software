/****************************************************************************
**
*A  rational.c                  GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file contains  the  functions  for  the  artithmetic  of  rationals.
**
**  Rationals  are  the union of  integers  and fractions.   A fraction  is a
**  quotient of two integers where the denominator does not evenly divide the
**  numerator.  If in the description of a function we  use the term rational
**  this  implies  that the  function is also   capable of handling integers,
**  though its  function would usually  be performed   by  a routine  in  the
**  integer package.  We will use the  term fraction to  stress the fact that
**  something must not be an integer.
**
**  A  fraction is represented  as a pair of  two integers.  The first is the
**  numerator and the second  is  the  denominator.  This representation   is
**  always reduced, i.e., numerator and denominator  are relative prime.  The
**  denominator is always  positive and  greater than 1.    If it were 1  the
**  fraction would be an integer and would be represented as  integer.  Since
**  the denominator is always positive the numerator carries the sign of  the
**  fraction.
**
**  It  is very easy to  see  that for every  fraction   there is one  unique
**  reduced representation.  Because of   this comparisons of  fractions  are
**  quite easy,  we just compare  numerator  and denominator.  Also numerator
**  and denominator are as small as possible,  reducing the effort to compute
**  with them.   Of course  computing  the reduced  representation comes at a
**  cost.   After every arithmetic operation we  have to compute the greatest
**  common divisor of numerator and denominator, and divide them by the gcd.
**
**  Effort  has been made to improve  efficiency by avoiding unneccessary gcd
**  computations.  Also if  possible this  package will compute  two gcds  of
**  smaller integers instead of one gcd of larger integers.
**
**  However no effort has  been made to write special  code for the case that
**  some of the  integers are small integers   (i.e., less than  2^28).  This
**  would reduce the overhead  introduced by the  calls to the functions like
**  'SumInt', 'ProdInt' or 'GcdInt'.
**
*/

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage management      */
#include        "scanner.h"             /* reading of tokens and printing  */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */
#include        "integer4.h"

#include        "rational.h"            /* declaration part of the package */


/****************************************************************************
**
*F  EvRat( <hdRat> )  . . . . . . . . . . . . . . . . . . evaluate a rational
**
**  'EvRat' returns the value of the rational <hdRat>.  Because rationals are
**  constants and thus selfevaluating this just returns <hdRat>.
*/
Bag       EvRat (Bag hdRat)
{
    return hdRat;
}


/****************************************************************************
**
*F  SumRat( <hdL>, <hdR> )  . . . . . . . . . . . . . .  sum of two rationals
**
**  'SumRat'  returns the   sum of two  rationals  <hdL>  and <hdR>.   Either
**  operand may also be an integer.  The sum is reduced.
**
**  Is called from the 'Sum' binop, so both operands are already evaluated.
*/
Bag       SumRat (Bag hdL, Bag hdR)
{
    Bag           numL, denL;     /* numerator and denominator left  */
    Bag           numR, denR;     /* numerator and denominator right */
    Bag           hdG1, hdG2;     /* gcd of denominators             */
    Bag           numS, denS;     /* numerator and denominator sum   */
    Bag           hdS;            /* sum                             */

    /* get numerator and denominator of the operands                       */
    if ( GET_TYPE_BAG(hdL) == T_RAT ) { numL = PTR_BAG(hdL)[0];  denL = PTR_BAG(hdL)[1];  }
    else {                      numL = hdL;          denL = INT_TO_HD(1); }
    if ( GET_TYPE_BAG(hdR) == T_RAT ) { numR = PTR_BAG(hdR)[0];  denR = PTR_BAG(hdR)[1];  }
    else {                      numR = hdR;          denR = INT_TO_HD(1); }

    /* find the gcd of the denominators                                    */
    hdG1 = GcdInt( denL, denR );

    /* nothing can cancel if the gcd is 1                                  */
    if ( hdG1 == INT_TO_HD(1) ) {
        numS = SumInt( ProdInt( numL, denR ), ProdInt( numR, denL ) );
        denS = ProdInt( denL, denR );
    }

    /* a little bit more difficult otherwise                               */
    else {
        numS = SumInt( ProdInt( numL, QuoInt( denR, hdG1 ) ),
                       ProdInt( numR, QuoInt( denL, hdG1 ) ) );
        hdG2 = GcdInt( numS, hdG1 );
        numS = QuoInt( numS, hdG2 );
        denS = ProdInt( QuoInt( denL, hdG1 ), QuoInt( denR, hdG2 ) );
    }

    /* make the fraction or, if possible, the integer                      */
    if ( denS != INT_TO_HD(1) ) {
        hdS  = NewBag( T_RAT, 2 * SIZE_HD );
        SET_BAG(hdS, 0,  numS );
        SET_BAG(hdS, 1,  denS );
    }
    else {
        hdS = numS;
    }

    /* return the result                                                   */
    return hdS;
}


/****************************************************************************
**
*F  DiffRat( <hdL>, <hdR> ) . . . . . . . . . . . difference of two rationals
**
**  'DiffRat' returns the  difference  of  two  rationals  <hdL>  and  <hdR>.
**  Either operand may also be an integer.  The difference is reduced.
**
**  Is called from the 'Diff' binop, so both operands are already evaluated.
*/
Bag       DiffRat (Bag hdL, Bag hdR)
{
    Bag           numL, denL;     /* numerator and denominator left  */
    Bag           numR, denR;     /* numerator and denominator right */
    Bag           hdG1, hdG2;     /* gcd of denominators             */
    Bag           numD, denD;     /* numerator and denominator diff  */
    Bag           hdD;            /* diff                            */

    /* get numerator and denominator of the operands                       */
    if ( GET_TYPE_BAG(hdL) == T_RAT ) { numL = PTR_BAG(hdL)[0];  denL = PTR_BAG(hdL)[1];  }
    else {                      numL = hdL;          denL = INT_TO_HD(1); }
    if ( GET_TYPE_BAG(hdR) == T_RAT ) { numR = PTR_BAG(hdR)[0];  denR = PTR_BAG(hdR)[1];  }
    else {                      numR = hdR;          denR = INT_TO_HD(1); }

    /* find the gcd of the denominators                                    */
    hdG1 = GcdInt( denL, denR );

    /* nothing can cancel if the gcd is 1                                  */
    if ( hdG1 == INT_TO_HD(1) ) {
        numD = DiffInt( ProdInt( numL, denR ), ProdInt( numR, denL ) );
        denD = ProdInt( denL, denR );
    }

    /* a little bit more difficult otherwise                               */
    else {
        numD = DiffInt( ProdInt( numL, QuoInt( denR, hdG1 ) ),
                        ProdInt( numR, QuoInt( denL, hdG1 ) ) );
        hdG2 = GcdInt( numD, hdG1 );
        numD = QuoInt( numD, hdG2 );
        denD = ProdInt( QuoInt( denL, hdG1 ), QuoInt( denR, hdG2 ) );
    }

    /* make the fraction or, if possible, the integer                      */
    if ( denD != INT_TO_HD(1) ) {
        hdD  = NewBag( T_RAT, 2 * SIZE_HD );
        SET_BAG(hdD, 0,  numD );
        SET_BAG(hdD, 1,  denD );
    }
    else {
        hdD = numD;
    }

    /* return the result                                                   */
    return hdD;
}


/****************************************************************************
**
*F  ProdRat( <hdL>, <hdR> ) . . . . . . . . . . . .  product of two rationals
**
**  'ProdRat' returns the  product of two rationals <hdL> and  <hdR>.  Either
**  operand may also be an integer.  The product is reduced.
**
**  Is called from the 'Prod' binop, so both operands are already evaluated.
*/
Bag       ProdRat (Bag hdL, Bag hdR)
{
    Bag           numL, denL;     /* numerator and denominator left  */
    Bag           numR, denR;     /* numerator and denominator right */
    Bag           hdG1, hdG2;     /* gcd of denominators             */
    Bag           numP, denP;     /* numerator and denominator prod  */
    Bag           hdP;            /* prod                            */

    /* get numerator and denominator of the operands                       */
    if ( GET_TYPE_BAG(hdL) == T_RAT ) { numL = PTR_BAG(hdL)[0];  denL = PTR_BAG(hdL)[1];  }
    else {                      numL = hdL;          denL = INT_TO_HD(1); }
    if ( GET_TYPE_BAG(hdR) == T_RAT ) { numR = PTR_BAG(hdR)[0];  denR = PTR_BAG(hdR)[1];  }
    else {                      numR = hdR;          denR = INT_TO_HD(1); }

    /* find the gcds                                                       */
    hdG1 = GcdInt( numL, denR );
    hdG2 = GcdInt( numR, denL );

    /* nothing can cancel if the gcds are 1                                */
    if ( hdG1 == INT_TO_HD(1) && hdG2 == INT_TO_HD(1) ) {
        numP = ProdInt( numL, numR );
        denP = ProdInt( denL, denR );
    }

    /* a little bit more difficult otherwise                               */
    else {
        numP = ProdInt( QuoInt( numL, hdG1 ), QuoInt( numR, hdG2 ) );
        denP = ProdInt( QuoInt( denL, hdG2 ), QuoInt( denR, hdG1 ) );
    }

    /* make the fraction or, if possible, the integer                      */
    if ( denP != INT_TO_HD(1) ) {
        hdP = NewBag( T_RAT, 2 * SIZE_HD );
        SET_BAG(hdP, 0,  numP );
        SET_BAG(hdP, 1,  denP );
    }
    else {
        hdP = numP;
    }

    /* return the result                                                   */
    return hdP;
}


/****************************************************************************
**
*F  QuoRat( <hdL>, <hdR> )  . . . . . . . . . . . . quotient of two rationals
**
**  'QuoRat'  returns the quotient of two rationals <hdL> and  <hdR>.  Either
**  operand may also be an integer.  The quotient is reduced.
**
**  Is called from the 'Quo' binop, so both operands are already evaluated.
*/
Bag       QuoRat (Bag hdL, Bag hdR)
{
    Bag           numL, denL;     /* numerator and denominator left  */
    Bag           numR, denR;     /* numerator and denominator right */
    Bag           hdG1, hdG2;     /* gcd of denominators             */
    Bag           numQ, denQ;     /* numerator and denominator Qrod  */
    Bag           hdQ;            /* Qrod                            */

    /* get numerator and denominator of the operands                       */
    if ( GET_TYPE_BAG(hdL) == T_RAT ) { numL = PTR_BAG(hdL)[0];  denL = PTR_BAG(hdL)[1];  }
    else {                      numL = hdL;          denL = INT_TO_HD(1); }
    if ( GET_TYPE_BAG(hdR) == T_RAT ) { numR = PTR_BAG(hdR)[0];  denR = PTR_BAG(hdR)[1];  }
    else {                      numR = hdR;          denR = INT_TO_HD(1); }

    /* division by zero is an error                                        */
    if ( numR == INT_TO_HD(0) )
        return Error("divisor must not be zero",0,0);

    /* we multiply the left numerator with the right denominator           */
    /* so the right denominator should carry the sign of the right operand */
    if ( (GET_TYPE_BAG(numR)==T_INT && HD_TO_INT(numR)<0) || GET_TYPE_BAG(numR)==T_INTNEG ) {
        numR = ProdInt( INT_TO_HD(-1), numR );
        denR = ProdInt( INT_TO_HD(-1), denR );
    }

    /* find the gcds                                                       */
    hdG1 = GcdInt( numL, numR );
    hdG2 = GcdInt( denR, denL );

    /* nothing can cancel if the gcds are 1                                */
    if ( hdG1 == INT_TO_HD(1) && hdG2 == INT_TO_HD(1) ) {
        numQ = ProdInt( numL, denR );
        denQ = ProdInt( denL, numR );
    }

    /* a little bit more difficult otherwise                               */
    else {
        numQ = ProdInt( QuoInt( numL, hdG1 ), QuoInt( denR, hdG2 ) );
        denQ = ProdInt( QuoInt( denL, hdG2 ), QuoInt( numR, hdG1 ) );
    }

    /* make the fraction or, if possible, the integer                      */
    if ( denQ != INT_TO_HD(1) ) {
        hdQ = NewBag( T_RAT, 2 * SIZE_HD );
        SET_BAG(hdQ, 0,  numQ );
        SET_BAG(hdQ, 1,  denQ );
    }
    else {
        hdQ = numQ;
    }

    /* return the result                                                   */
    return hdQ;
}


/****************************************************************************
**
*F  ModRat( <hdL>, <hdL> )  . . . . . . . . remainder of fraction mod integer
**
**  'ModRat' returns the remainder  of the fraction  <hdL> modulo the integer
**  <hdR>.  The remainder is always an integer.
**
**  '<r>  / <s> mod  <n>' yields  the remainder of   the fraction '<r> / <s>'
**  modulo the integer '<n>'.
**
**  The  modular  remainder of  $r  / s$  mod $n$  is defined  as  a $l$ from
**  $0..n-1$ such that $r = l s$ mod $n$.  As a special  case $1 / s$ mod $n$
**  is the modular inverse of $s$ modulo $n$.
**
**  Note  that the remainder will  not exist if $s$  is not relative prime to
**  $n$.  However note that $4 / 6$  mod $32$ does  exist (and is $22$), even
**  though $6$ is not invertable modulo $32$, because the $2$ cancels.
**
**  Another possible  definition of $r/s$ mod $n$  would be  a rational $t/s$
**  such that $0 \<= t/s \< n$ and $r/s - t/s$ is a multiple of $n$.  This is
**  rarely needed while computing modular inverses is very useful.
**
**  Is called from the 'Mod' binop, so both operands are already evaluated.
*/
Bag       ModRat (Bag hdL, Bag hdR)
{
    Bag           hdA, hdAL, hdB, hdBL, hdH, hdHL, hdQ;

    /* make the integer positive                                           */
    if ( (GET_TYPE_BAG(hdR)==T_INT && HD_TO_INT(hdR)<0) || GET_TYPE_BAG(hdR)==T_INTNEG )
        hdR = ProdInt( INT_TO_HD(-1), hdR );

    /* invert the denominator with Euclids algorithm                       */
    hdA = hdR;          hdAL = INT_TO_HD(0);
    hdB = PTR_BAG(hdL)[1];  hdBL = INT_TO_HD(1);
    while ( hdB != INT_TO_HD(0) ) {
        hdQ  = QuoInt( hdA, hdB );
        hdH  = hdB;  hdHL = hdBL;
        hdB  = DiffInt( hdA,  ProdInt( hdQ, hdB  ) );
        hdBL = DiffInt( hdAL, ProdInt( hdQ, hdBL ) );
        hdA  = hdH;  hdAL = hdHL;
    }

    /* check whether the denominator really was invertable mod <hdR>       */
    if ( hdA != INT_TO_HD(1) )
        return Error("RatOps: denominator must be invertable",0,0);

    /* return the remainder                                                */
    return ModInt( ProdInt( PTR_BAG(hdL)[0], hdAL ), hdR );
}


/****************************************************************************
**
*F  PowRat( <hdL>, <hdR> )  . . . . . .  raise a rational to an integer power
**
**  'PowRat' raises the rational <hdL> to the  power  given  by  the  integer
**  <hdR>.  The power is reduced.
**
**  Is called from the 'Pow' binop, so both operands are already evaluated.
*/
Bag       PowRat (Bag hdL, Bag hdR)
{
    Bag           numP, denP;     /* numerator and denominator power */
    Bag           hdP;            /* power                           */

    /* raise numerator and denominator seperately                          */
    numP = PowInt( PTR_BAG(hdL)[0], hdR );
    denP = PowInt( PTR_BAG(hdL)[1], hdR );

    /* if <hdR> == 0 return 1                                              */
    if ( hdR == INT_TO_HD(0) ) {
        hdP = INT_TO_HD(1);
    }

    /* if <hdR> == 1 return <hdL>                                          */
    else if ( hdR == INT_TO_HD(1) ) {
        hdP = hdL;
    }

    /* if <hdR> is positive raise numberator and denominator seperately    */
    else if ( (GET_TYPE_BAG(hdR)==T_INT&&0<HD_TO_INT(hdR)) || GET_TYPE_BAG(hdR)==T_INTPOS ) {
        numP = PowInt( PTR_BAG(hdL)[0], hdR );
        denP = PowInt( PTR_BAG(hdL)[1], hdR );
        hdP = NewBag( T_RAT, 2 * SIZE_HD );
        SET_BAG(hdP, 0,  numP );
        SET_BAG(hdP, 1,  denP );
    }

    /* if <hdR> is negative and numerator is 1 just power the denominator  */
    else if ( PTR_BAG(hdL)[0] == INT_TO_HD(1) ) {
        hdP = PowInt( PTR_BAG(hdL)[1], ProdInt(INT_TO_HD(-1),hdR) );
    }

    /* if <hdR> is negative and numerator is -1 return (-1)^r * num(l)     */
    else if ( PTR_BAG(hdL)[0] == INT_TO_HD(-1) ) {
        hdP = ProdInt( PowInt( PTR_BAG(hdL)[0], ProdInt(INT_TO_HD(-1),hdR) ),
                       PowInt( PTR_BAG(hdL)[1], ProdInt(INT_TO_HD(-1),hdR) ) );
    }

    /* if <hdR> is negative do both powers, take care of the sign          */
    else {
        numP = PowInt( PTR_BAG(hdL)[1], ProdInt( INT_TO_HD(-1), hdR ) );
        denP = PowInt( PTR_BAG(hdL)[0], ProdInt( INT_TO_HD(-1), hdR ) );
        hdP  = NewBag( T_RAT, 2 * SIZE_HD );
        if ( (GET_TYPE_BAG(denP) == T_INT && 0 < HD_TO_INT(denP))
          || GET_TYPE_BAG(denP) == T_INTPOS ) {
            SET_BAG(hdP, 0,  numP );
            SET_BAG(hdP, 1,  denP );
        }
        else {
            SET_BAG(hdP, 0,  ProdInt( INT_TO_HD(-1), numP ) );
            SET_BAG(hdP, 1,  ProdInt( INT_TO_HD(-1), denP ) );
        }
    }

    /* return the result                                                   */
    return hdP;
}


/****************************************************************************
**
*F  EqRat( <hdL>, <hdR> ) . . . . . . . . . . . . . . test if <ratL> = <ratR>
**
**  'EqRat' returns 'true' if the two rationals <ratL> and <ratR>  are  equal
**  and 'false' otherwise.
**
**  Is called from 'EvEq' binop, so both operands are already evaluated.
*/
Bag       EqRat (Bag hdL, Bag hdR)
{
    Bag           numL, denL;     /* numerator and denominator left  */
    Bag           numR, denR;     /* numerator and denominator right */

    /* get numerator and denominator of the operands                       */
    if ( GET_TYPE_BAG(hdL) == T_RAT ) { numL = PTR_BAG(hdL)[0];  denL = PTR_BAG(hdL)[1];  }
    else {                      numL = hdL;          denL = INT_TO_HD(1); }
    if ( GET_TYPE_BAG(hdR) == T_RAT ) { numR = PTR_BAG(hdR)[0];  denR = PTR_BAG(hdR)[1];  }
    else {                      numR = hdR;          denR = INT_TO_HD(1); }

    /* two rationals are equal if numerators and denominators are equal    */
    if ( EqInt(numL,numR) == HdTrue && EqInt(denL,denR) == HdTrue )
        return HdTrue;
    else
        return HdFalse;
}


/****************************************************************************
**
*F  LtRat( <hdL>, <hdR> ) . . . . . . . . . . . . . . test if <ratL> < <ratR>
**
**  'LtRat' returns 'true'  if  the  rational  <ratL>  is  smaller  than  the
**  rational <ratR> and 'false' otherwise.  Either operand may be an integer.
**
**  Is called from 'EvLt' binop, so both operands are already evaluated.
*/
Bag       LtRat (Bag hdL, Bag hdR)
{
    Bag           numL, denL;     /* numerator and denominator left  */
    Bag           numR, denR;     /* numerator and denominator right */

    /* get numerator and denominator of the operands                       */
    if ( GET_TYPE_BAG(hdL) == T_RAT ) { numL = PTR_BAG(hdL)[0];  denL = PTR_BAG(hdL)[1];  }
    else {                      numL = hdL;          denL = INT_TO_HD(1); }
    if ( GET_TYPE_BAG(hdR) == T_RAT ) { numR = PTR_BAG(hdR)[0];  denR = PTR_BAG(hdR)[1];  }
    else {                      numR = hdR;          denR = INT_TO_HD(1); }

    /* a / b < c / d <=> a d < c b                                         */
    return LtInt( ProdInt( numL, denR ), ProdInt( numR, denL ) );
}


/****************************************************************************
**
*F  PrRat( <hdRat> )  . . . . . . . . . . . . . . . . . . .  print a rational
**
**  'PrRat' prints a rational in the standard form:
**
**      <numerator> / <denominator>
*/
void            PrRat (Bag hdRat)
{
    Pr("%>",0,0);
    Print( PTR_BAG(hdRat)[0] );
    Pr("%</%>",0,0);
    Print( PTR_BAG(hdRat)[1] );
    Pr("%<",0,0);
}


/****************************************************************************
**
*F  FunIsRat( <hdCall> )  . . . . . . . . .  internal function IsRat( <obj> )
**
**  'IsRat'  returns 'true' if the object <obj> is  a  rational  and  'false'
**  otherwise.  May cause an error if <obj> is an unbound variable.
*/
Bag       FunIsRat (Bag hdCall)
{
    Bag           hdObj;

    /* evaluate and check the argument                                     */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: IsRat( <obj> )",0,0);
    hdObj = EVAL( PTR_BAG(hdCall)[1] );
    if ( hdObj == HdVoid )
        return Error("IsRat: function must return a value",0,0);

    /* return 'true' if <obj> is a rational and 'false' otherwise          */
    if ( GET_TYPE_BAG(hdObj) == T_RAT    || GET_TYPE_BAG(hdObj) == T_INT
      || GET_TYPE_BAG(hdObj) == T_INTPOS || GET_TYPE_BAG(hdObj) == T_INTNEG )
        return HdTrue;
    else
        return HdFalse;
}


/****************************************************************************
**
*F  FunNumerator( <hdCall> )  . . . . . . . . . . internal function Numerator
**
**  'Numerator' returns the numerator of the rational argument.
*/
Bag       FunNumerator (Bag hdCall)
{
    Bag           hdRat;

    /* evaluate and check the argument                                     */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: Numerator( <rat> )",0,0);
    hdRat = EVAL( PTR_BAG(hdCall)[1] );
    if ( hdRat == HdVoid )
        return Error("Numerator: function must return a value",0,0);
    if ( GET_TYPE_BAG(hdRat) != T_RAT    && GET_TYPE_BAG(hdRat) != T_INT
      && GET_TYPE_BAG(hdRat) != T_INTPOS && GET_TYPE_BAG(hdRat) != T_INTNEG )
        return Error("usage: Numerator( <rat> )",0,0);

    /* return the numerator                                                */
    if ( GET_TYPE_BAG(hdRat) == T_RAT )
        return PTR_BAG(hdRat)[0];
    else
        return hdRat;
}


/****************************************************************************
**
*F  FunDenominator( <hdCall> )  . . . . . . . . internal function Denominator
**
**  'Denominator' returns the denominator of the rational argument.
*/
Bag       FunDenominator (Bag hdCall)
{
    Bag           hdRat;

    /* evaluate and check the argument                                     */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: Denominator( <rat> )",0,0);
    hdRat = EVAL( PTR_BAG(hdCall)[1] );
    if ( hdRat == HdVoid )
        return Error("Denominator: function must return a value",0,0);
    if ( GET_TYPE_BAG(hdRat) != T_RAT    && GET_TYPE_BAG(hdRat) != T_INT
      && GET_TYPE_BAG(hdRat) != T_INTPOS && GET_TYPE_BAG(hdRat) != T_INTNEG )
        return Error("usage: Denominator( <rat> )",0,0);

    /* return the denominator                                              */
    if ( GET_TYPE_BAG(hdRat) == T_RAT )
        return PTR_BAG(hdRat)[1];
    else
        return INT_TO_HD(1);
}


/****************************************************************************
**
*F  InitRat() . . . . . . . . . . . . . . . . initialize the rational package
**
**  'InitRat' initializes the rational package.
*/
void            InitRat (void)
{
    InstEvFunc( T_RAT, EvRat );
    InstPrFunc( T_RAT, PrRat );

    TabSum[  T_RAT    ][ T_RAT    ] = SumRat;
    TabSum[  T_INT    ][ T_RAT    ] = SumRat;
    TabSum[  T_INTPOS ][ T_RAT    ] = SumRat;
    TabSum[  T_INTNEG ][ T_RAT    ] = SumRat;
    TabSum[  T_RAT    ][ T_INT    ] = SumRat;
    TabSum[  T_RAT    ][ T_INTPOS ] = SumRat;
    TabSum[  T_RAT    ][ T_INTNEG ] = SumRat;

    TabDiff[ T_RAT    ][ T_RAT    ] = DiffRat;
    TabDiff[ T_INT    ][ T_RAT    ] = DiffRat;
    TabDiff[ T_INTPOS ][ T_RAT    ] = DiffRat;
    TabDiff[ T_INTNEG ][ T_RAT    ] = DiffRat;
    TabDiff[ T_RAT    ][ T_INT    ] = DiffRat;
    TabDiff[ T_RAT    ][ T_INTPOS ] = DiffRat;
    TabDiff[ T_RAT    ][ T_INTNEG ] = DiffRat;

    TabProd[ T_RAT    ][ T_RAT    ] = ProdRat;
    TabProd[ T_INT    ][ T_RAT    ] = ProdRat;
    TabProd[ T_INTPOS ][ T_RAT    ] = ProdRat;
    TabProd[ T_INTNEG ][ T_RAT    ] = ProdRat;
    TabProd[ T_RAT    ][ T_INT    ] = ProdRat;
    TabProd[ T_RAT    ][ T_INTPOS ] = ProdRat;
    TabProd[ T_RAT    ][ T_INTNEG ] = ProdRat;

    TabQuo[  T_INT    ][ T_INT    ] = QuoRat;
    TabQuo[  T_INT    ][ T_INTPOS ] = QuoRat;
    TabQuo[  T_INT    ][ T_INTNEG ] = QuoRat;
    TabQuo[  T_INTPOS ][ T_INT    ] = QuoRat;
    TabQuo[  T_INTPOS ][ T_INTPOS ] = QuoRat;
    TabQuo[  T_INTPOS ][ T_INTNEG ] = QuoRat;
    TabQuo[  T_INTNEG ][ T_INT    ] = QuoRat;
    TabQuo[  T_INTNEG ][ T_INTPOS ] = QuoRat;
    TabQuo[  T_INTNEG ][ T_INTNEG ] = QuoRat;

    TabQuo[  T_RAT    ][ T_RAT    ] = QuoRat;
    TabQuo[  T_INT    ][ T_RAT    ] = QuoRat;
    TabQuo[  T_INTPOS ][ T_RAT    ] = QuoRat;
    TabQuo[  T_INTNEG ][ T_RAT    ] = QuoRat;
    TabQuo[  T_RAT    ][ T_INT    ] = QuoRat;
    TabQuo[  T_RAT    ][ T_INTPOS ] = QuoRat;
    TabQuo[  T_RAT    ][ T_INTNEG ] = QuoRat;

    TabMod[  T_RAT    ][ T_INT    ] = ModRat;
    TabMod[  T_RAT    ][ T_INTPOS ] = ModRat;
    TabMod[  T_RAT    ][ T_INTNEG ] = ModRat;

    TabPow[  T_RAT    ][ T_INT    ] = PowRat;
    TabPow[  T_RAT    ][ T_INTPOS ] = PowRat;
    TabPow[  T_RAT    ][ T_INTNEG ] = PowRat;

    TabEq[   T_RAT    ][ T_RAT    ] = EqRat;

    TabLt[   T_RAT    ][ T_RAT    ] = LtRat;
    TabLt[   T_INT    ][ T_RAT    ] = LtRat;
    TabLt[   T_INTPOS ][ T_RAT    ] = LtRat;
    TabLt[   T_INTNEG ][ T_RAT    ] = LtRat;
    TabLt[   T_RAT    ][ T_INT    ] = LtRat;
    TabLt[   T_RAT    ][ T_INTPOS ] = LtRat;
    TabLt[   T_RAT    ][ T_INTNEG ] = LtRat;

    InstIntFunc( "IsRat",       FunIsRat       );
    InstIntFunc( "Numerator",   FunNumerator   );
    InstIntFunc( "Denominator", FunDenominator );
}



