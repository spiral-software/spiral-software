/****************************************************************************
**
*A  eval.c                      GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2020, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2020) by the GAP Group (www.gap-system.org).
**
**  This file contains the main evaluation functions.
**
*/

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "scanner.h"             /* reading of tokens and printing  */
#include        "idents.h"              /* symbol table managment          */
#include        "integer.h"             /* 'InitInt', large integers       */

#include        "rational.h"            /* 'InitRat'                       */
#include        "cyclotom.h"            /* 'InitCyc'                       */
#include        "unknown.h"             /* 'InitUnknown'                   */
#include        "finfield.h"            /* 'InitFF'                        */
#include        "polynom.h"             /* 'InitPolynom'                   */

#include        "permutat.h"            /* 'InitPermutat'                  */
#include        "word.h"                /* 'InitWord'                      */
#include        "costab.h"              /* 'InitCostab'                    */
#include        "tietze.h"              /* 'InitTietze'                    */
#include        "aggroup.h"             /* 'InitAg'                        */
#include        "pcpresen.h"            /* 'InitPcPres'                    */

#include        "list.h"                /* 'InitList', generic list funcs  */
#include        "plist.h"               /* 'InitPlist', 'LEN_PLIST', ..    */
#include        "set.h"                 /* 'InitSet'                       */
#include        "vector.h"              /* 'InitVector'                    */
#include        "vecffe.h"              /* 'InitVecFFE'                    */
#include        "eval.h"                /* definition part of this package */
#include        "blister.h"             /* 'InitBlist'                     */
#include        "range.h"               /* 'InitRange'                     */
#include        "gstring.h"              /* 'InitString', 'IsString'        */

#include        "record.h"              /* 'InitRec'                       */
#include        "statemen.h"            /* 'InitStat'                      */
#include        "function.h"            /* 'InitFunc'                      */
#include        "coding.h"              /* 'InitCoding'                    */

#include        "spiral.h"              /* Try/Catch                       */
/* #include        "flags.h" */			// defns & decls moved to memmgr.h
#include        "tables.h"
#include        "comments.h"            /* DocumentValue                   */
#include        "namespaces.h"
#include        "args.h"
#include		"GapUtils.h"
#include        "debug.h"
#include        "objects.h"

/****************************************************************************
**
*V  HdVoid  . . . . . . . . . . . . . . . . . . . . .  handle of the void bag
**
**  'HdVoid' is the handle of the void back, which is returned by procedures,
**  i.e., functions that when viewed at the  GAP  level do not return a value
**  at all.  This plays a role similar to  '*the-non-printing-object*'  which
**  exists in some lisp systems.
*/
Obj       HdVoid;


/****************************************************************************
**
*V  HdReturn  . . . . . . . . . . . . . . . . . . .  handle of the return bag
**
**  'HdReturn' is the handle of the bag where 'EvReturn' puts the value of a
**  'return' statement.  This bag is then passed through all  the  statement
**  execution functions all the  way  back  to  'EvFunccall'.  For  'return'
**  statements without an expression 'EvReturn' puts 'HdVoid' into this bag.
*/
Obj       HdReturn;


/****************************************************************************
**
*V  HdTrue  . . . . . . . . . . . . . . . . . . . . .  handle of the true bag
*V  HdFalse   . . . . . . . . . . . . . . . . . . . . handle of the false bag
**
**  'HdTrue' is the handle of the unique bag that represents the value 'true'
**  and 'HdFalse' is likewise the unique handle of the  bag  that  represents
**  the value 'HdFalse'.
*/
Obj       HdTrue,  HdFalse;


/****************************************************************************
**
**  EvalStack  . . . . . . . . . . . . . . . . . . . . . . . .  EVAL stack
**  EvalStackTop . . . . . . . . . . . . . . . . . . . . . . .  stack pointer
**
**  EvalStack is a plaine array that contains full EVAL stack.
**  EvalStackTop - index of the last plased element.
*/

Bag         EvalStack[EVAL_STACK_COUNT+EVAL_STACK_CUSHION];
UInt        EvalStackTop = 0;

/****************************************************************************
**
*F  EVAL( <hd> )  . . . . . . . . . . . . . . . . . . . .  evaluate an object
**
**  'EVAL' evaluates the bag <hd>  by  calling  the  corresponding  function.
**
**  It is defined in the definition file of this package as followings:
**
#define EVAL(hd)        ((long)(hd)&T_INT ? (hd) : (* EvTab[GET_TYPE_BAG(hd)])((hd)))
*/


/****************************************************************************
**
*V  EvTab[<type>] . . . . . . . . evaluation function for bags of type <type>
**
**  Is the main dispatching table that contains for every type a  pointer  to
**  the function that should be executed if a bag  of  that  type  is  found.
*/
Obj       (* EvTab[ T_ILLEGAL ]) ( Obj hd );


/****************************************************************************
**
*F  CantEval( <hd> )  . . . . . . . . . . . . illegal bag evaluation function
**
**  Is called if a illegal bag should be evaluated, it  generates  an  error.
**  If this is actually ever executed in GAP it  indicates  serious  trouble,
**  for  example  that  the  type  field  of  a  bag  has  been  overwritten.
*/
Obj       CantEval (Obj hd)
{
    return Error("Panic: can't eval bag of type %d",(Int)GET_TYPE_BAG(hd),0);
}


/****************************************************************************
**
*F  NoEval( <hd> )  . . . . . . . . . . . . . . evaluation by returning self
**
*/
Obj       NoEval (Obj hd)
{
    return hd;
}

/****************************************************************************
**
*F  Concat( <hdConcat> )  . . . . . . . . . . . evaluate a list concatenation
**
*/
Bag       Concat (Bag hdConcat) {
    Bag           hdL,  hdR,  hdRes;
    hdL = EVAL( PTR_BAG(hdConcat)[0] );
    hdR = EVAL( PTR_BAG(hdConcat)[1] );
    hdRes = ShallowCopy(hdL);
    Append(hdRes, hdR);
    return hdRes;
}

/****************************************************************************
**
*F  Sum( <hdSum> )  . . . . . . . . . . . . . . . . . . . . .  evaluate a sum
*F  SUM(<hdL>,<hdR>)  . . . . . . . . . . . . . . . . . . . .  evaluate a sum
*V  TabSum[<typeL>][<typeR>]  . . . . . . . . . . table of addition functions
*F  CantSum(<hdL>,<hdR>)  . . . . . . . . . . . . . . . . . . . undefined sum
**
**  'Sum' returns the sum of the two objects '<hdSum>[0]'  and  '<hdSum>[1]'.
**  'Sum' is called from 'EVAL' to eval bags of type 'T_SUM'.
**
**  'Sum' evaluates the operands and then calls the 'SUM' macro.
**
**  'SUM' finds the types of the two operands and uses  them  to  index  into
**  the table 'TabSum' of addition functions.
**
**  At places where performance really matters one should  copy  the  special
**  code from 'Sum' which checks for the addition of two  immediate  integers
**  and computes their sum without calling 'SUM'.
**
**  'SUM' is defined in the header file of this package as follows:
**
#define SUM(hdL,hdR)    ((*TabSum[GET_TYPE_BAG(hdL)][GET_TYPE_BAG(hdR)])((hdL),(hdR)))
*/
Obj       (*TabSum[EV_TAB_SIZE][EV_TAB_SIZE]) ( Obj, Obj );

Obj       Sum (Obj hd)
{
    Obj           hdL,  hdR;
    Int                result;

    hdL = EVAL( PTR_BAG(hd)[0] );  hdR = EVAL( PTR_BAG(hd)[1] );

    /* add two small integers with a small sum                             */
    /* add and compare top two bits to check that no overflow occured      */
    if ( (Int)hdL & (Int)hdR & T_INT ) {
        result = (Int)hdL + (Int)hdR - T_INT;
        if ( ((result << 1) >> 1) == result )
            return (Obj)result;
    }

    return SUM( hdL, hdR );
}

Obj       CantSum (Obj hdL, Obj hdR)
{
    return Error("operations: sum of %s and %s is not defined",
                 (Int)NameType[GET_TYPE_BAG(hdL)], (Int)NameType[GET_TYPE_BAG(hdR)] );
}


/****************************************************************************
**
*F  Diff( <hdDiff> )  . . . . . . . . . . . . . . . . . evaluate a difference
*F  DIFF(<hdL>,<hdR>)   . . . . . . . . . . . . . . . . evaluate a difference
*V  TabDiff[<typeL>][<typeR>] . . . . . . . .  table of subtraction functions
*F  CantDiff(<hdL>,<hdR>) . . . . . . . . . . . . . . .  undefined difference
**
**  'Diff' returns  the  difference of  the  two  objects  '<hdDiff>[0]'  and
**  '<hdDiff>[1]'.  'Diff'  is  called from  'EVAL'  to  eval  bags  of  type
**  'T_DIFF'.
**
**  'Diff' evaluates the operands and then calls the 'DIFF' macro.
**
**  'DIFF' finds the types of the two operands and uses them  to  index  into
**  the table 'TabDiff' of subtraction functions.
**
**  At places where performance really matters one should  copy  the  special
**  code from 'Diff'  which  checks for  the  subtraction  of  two  immediate
**  integers and computes their difference without calling 'DIFF'.
**
**  'DIFF' is defined in the header file of this package as follows:
**
#define DIFF(hdL,hdR)   ((*TabDiff[GET_TYPE_BAG(hdL)][GET_TYPE_BAG(hdR)])((hdL),(hdR)))
*/
Obj       (*TabDiff[EV_TAB_SIZE][EV_TAB_SIZE]) ( Obj, Obj );

Obj       Diff (Obj hd)
{
    Obj           hdL,  hdR;
    Int                result;

    hdL = EVAL( PTR_BAG(hd)[0] );  hdR = EVAL( PTR_BAG(hd)[1] );

    /* subtract two small integers with a small difference                 */
    /* sub and compare top two bits to check that no overflow occured      */
    if ( (Int)hdL & (Int)hdR & T_INT ) {
        result = (Int)hdL - (Int)hdR;
        if ( ((result << 1) >> 1) == result )
            return (Obj)(result + T_INT);
    }

    return DIFF(hdL,hdR);
}

Obj       CantDiff (Obj hdL, Obj hdR)
{
    return Error("operations difference of %s and %s is not defined",
                 (Int)NameType[GET_TYPE_BAG(hdL)], (Int)NameType[GET_TYPE_BAG(hdR)] );
}

/****************************************************************************
**
*F  Prod( <hdProd> )  . . . . . . . . . . . . . . . . . .  evaluate a product
*F  PROD(<hdL>,<hdR>)   . . . . . . . . . . . . . . . . .  evaluate a product
*V  TabProd[<typeL>][<typeR>] . . . . . . . table of multiplication functions
*F  CantProd(<hdL>,<hdR>) . . . . . . . . . . . . . . . . . undefined product
**
**  'Prod'  returns   the  product  of  the  two  objects  '<hdProd>[0]'  and
**  '<hdProd>[1]'.  'Prod'  is  called from  'EVAL'  to  eval  bags  of  type
**  'T_PROD'.
**
**  'Prod' evaluates the operands and then calls the 'PROD' macro.
**
**  'PROD' finds the types of the two operands and uses them  to  index  into
**  the table 'TabProd' of multiplication functions.
**
**  At places where performance really matters one should  copy  the  special
**  code from 'Prod'  which  checks for  the  subtraction  of  two  immediate
**  integers and computes their product without calling 'PROD'.
**
**  'PROD' is defined in the header file of this package as follows:
**
#define PROD(hdL,hdR)   ((*TabProd[GET_TYPE_BAG(hdL)][GET_TYPE_BAG(hdR)])((hdL),(hdR)))
*/
Obj       (*TabProd[EV_TAB_SIZE][EV_TAB_SIZE]) ( Obj, Obj );

Obj       Prod (Obj hd)
{
    Obj           hdL,  hdR;
    Obj           hdResult;

    hdL = EVAL( PTR_BAG(hd)[0] );
	hdR = EVAL( PTR_BAG(hd)[1] );

#if _DEBUG
    ObjType typeL = GET_TYPE_BAG(hdL);
    ObjType typeR = GET_TYPE_BAG(hdR);
    Obj(*prodfunc)(Obj, Obj) = TabProd[typeL][typeR];
    hdResult = prodfunc(hdL, hdR);
#else
    hdResult = PROD( hdL, hdR );
#endif
	return hdResult;
}

Obj       CantProd (Obj hdL, Obj hdR)
{
    return Error("operations: product of %s and %s is not defined",
				 (Int)InfoBags[GET_TYPE_BAG(hdL)].name, (Int)InfoBags[GET_TYPE_BAG(hdR)].name );
}


/****************************************************************************
**
*F  Quo( <hdQuo> )  . . . . . . . . . . . . . . . . . . . evaluate a quotient
*F  QUO(<hdL>,<hdR>)  . . . . . . . . . . . . . . . . . . evaluate a quotient
*V  TabQuo[<typeL>][<typeR>]  . . . . . . . . . . table of division functions
*F  CantQuo(<hdL>,<hdR>)  . . . . . . . . . . . . . . . .  undefined quotient
**
**  'Quo'  returns   the   quotient  of  the  two  objects  '<hdQuo>[0]'  and
**  '<hdQuo>[1]'.  'Quo' is called from 'EVAL' to eval bags of type 'T_QUO'.
**
**  'Quo' evaluates the operands and then calls the 'QUO' macro.
**
**  'QUO' finds the types of the two operands and uses  them  to  index  into
**  the table 'TabQuo' of division functions.
**
**  'QUO' is defined in the header file of this package as follows:
**
#define QUO(hdL,hdR)    ((*TabQuo[GET_TYPE_BAG(hdL)][GET_TYPE_BAG(hdR)])((hdL),(hdR)))
*/
Obj       (*TabQuo[EV_TAB_SIZE][EV_TAB_SIZE]) ( Obj, Obj );

Obj       Quo (Obj hd)
{
    Obj           hdL,  hdR;

    hdL = EVAL( PTR_BAG(hd)[0] );  hdR = EVAL( PTR_BAG(hd)[1] );

    return QUO( hdL, hdR );
}

Obj       CantQuo (Obj hdL, Obj hdR)
{
    return Error("operations: quotient of %s and %s is not defined",
                 (Int)NameType[GET_TYPE_BAG(hdL)], (Int)NameType[GET_TYPE_BAG(hdR)] );
}


/****************************************************************************
**
*F  Mod( <hdMod> )  . . . . . . . . . . . . . . . . . .  evaluate a remainder
*F  MOD(<hdL>,<hdR>)  . . . . . . . . . . . . . . . . .  evaluate a remainder
*V  TabMod[<typeL>][<typeR>]  . . . . . . . . . . table of division functions
*F  CantMod(<hdL>,<hdR>)  . . . . . . . . . . . . . . . . undefined remainder
**
**  'Mod' returns   the  remainder   of  the  two  objects  '<hdMod>[0]'  and
**  '<hdMod>[1]'.  'Mod' is called from 'EVAL' to eval bags of type 'T_MOD'.
**
**  'Mod' evaluates the operands and then calls the 'MOD' macro.
**
**  'MOD' finds the types of the two operands and uses  them  to  index  into
**  the table 'TabMod' of remainder functions.
**
**  'MOD' is defined in the header file of this package as follows:
**
#define MOD(hdL,hdR)    ((*TabMod[GET_TYPE_BAG(hdL)][GET_TYPE_BAG(hdR)])((hdL),(hdR)))
*/
Obj       (*TabMod[EV_TAB_SIZE][EV_TAB_SIZE]) ( Obj, Obj );

Obj       Mod (Obj hd)
{
    Obj           hdL,  hdR;

    hdL = EVAL( PTR_BAG(hd)[0] );  hdR = EVAL( PTR_BAG(hd)[1] );

    return MOD( hdL, hdR );
}

Obj       CantMod (Obj hdL, Obj hdR)
{
    return Error("operations: remainder of %s and %s is not defined",
                 (Int)NameType[GET_TYPE_BAG(hdL)], (Int)NameType[GET_TYPE_BAG(hdR)] );
}


/****************************************************************************
**
*F  Pow( <hdPow> )  . . . . . . . . . . . . . . . . . . . .  evaluate a power
*F  POW(<hdL>,<hdR>)  . . . . . . . . . . . . . . . . . . .  evaluate a power
*V  TabPow[<typeL>][<typeR>]  . . . . . . . table of exponentiation functions
*F  CantPow(<hdL>,<hdR>)  . . . . . . . . . . . . . . . . . . undefined power
**
**  'Pow' returns the power of the two objects '<hdPow>[0]' and '<hdPow>[1]'.
**  'Pow' is called from 'EVAL' to eval bags of type 'T_POW'.
**
**  'Pow' evaluates the operands and then calls the 'POW' macro.
**
**  'POW' finds the types of the two operands and uses  them  to  index  into
**  the table 'TabPow' of powering functions.
**
**  'POW' is defined in the header file of this package as follows:
**
#define POW(hdL,hdR)    ((*TabPow[GET_TYPE_BAG(hdL)][GET_TYPE_BAG(hdR)])((hdL),(hdR)))
*/
Obj       (*TabPow[EV_TAB_SIZE][EV_TAB_SIZE]) ( Obj, Obj );

Obj       Pow (Obj hd)
{
    Obj           hdL,  hdR;
    Obj           hdResult;

    hdL = EVAL( PTR_BAG(hd)[0] );
	hdR = EVAL( PTR_BAG(hd)[1] );

    hdResult = POW( hdL, hdR );
	return hdResult;
	
}

Obj       CantPow (Obj hdL, Obj hdR)
{
    return Error("operations: power of %s and %s is not defined",
                 (Int)NameType[GET_TYPE_BAG(hdL)], (Int)NameType[GET_TYPE_BAG(hdR)] );
}


/****************************************************************************
**
*F  FunComm( <hdCall> ) . . . . . . . . . . . . . . . . evaluate a commutator
**
**  'FunComm' implements the internal function 'Comm'.
**
**  'Comm( <expr1>, <expr2> )'
**
**  'Comm' returns the commutator of  the  two  group  elements  <expr1>  and
**  <expr2>, i.e., '<expr1>^-1 * <expr2>^-1 * <expr1> * <expr2>'.
**
**  This is a hack to replace the commutator operator until I have fixed  the
**  parser to read something like '(a & b)'
*/
Obj       (*TabComm[EV_TAB_SIZE][EV_TAB_SIZE]) ( Obj, Obj );

Obj       IntComm (Obj hdCall)
{
    Obj       hdL, hdR;

    /* check the arguments                                                 */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )
        return Error("usage: Comm( <expr>, <expr> )",0,0);

    /* evaluate the arguments and jump through the function table          */
    hdL = EVAL( PTR_BAG(hdCall)[1] );  hdR = EVAL( PTR_BAG(hdCall)[2] );
    return (* TabComm[ GET_TYPE_BAG(hdL) ][ GET_TYPE_BAG(hdR) ]) ( hdL, hdR );
}

Obj       CantComm (Obj hdL, Obj hdR)
{
    return Error("operations: commutator of %s and %s is not defined",
                 (Int)NameType[GET_TYPE_BAG(hdL)], (Int)NameType[GET_TYPE_BAG(hdR)] );
}


/****************************************************************************
**
*F  FunLeftQuotient( <hdCall> ) . . . . . . . . . .  evaluate a left quotient
**
**  'FunLeftQuotient' implements the internal function 'LeftQuotient'.
**
**  'LeftQuotient( <expr1>, <expr2> )'
**
**  'LeftQuotient'  returns  the  left  quotient  of  the  two group elements
**  <expr1> and <expr2>, i.e., '<expr1>^-1 * <expr2>'.
*/
Obj       FunLeftQuotient (Obj hdCall)
{
    Obj       hdL, hdR;

    /** check the arguments ************************************************/
    if ( GET_SIZE_BAG( hdCall ) != 3 * SIZE_HD )
        return Error( "usage: LeftQuotient( <expr>, <expr> )", 0, 0 );

    /** evaluate the arguments and jump through the function table *********/
    hdL = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdR = EVAL( PTR_BAG( hdCall )[ 2 ] );
    return ( * TabMod[ GET_TYPE_BAG(hdL) ][ GET_TYPE_BAG(hdR) ] ) ( hdL, hdR );
}


/****************************************************************************
**
*F  Eq( <hdEq> )  . . . . . . . . . . . . . . . . .  test if <objL> =  <objR>
*F  EQ(<hdL>,<hdR>) . . . . . . . . . . . . . . . .  test if <objL> =  <objR>
*V  TabEq[<typeL>][<typeR>] . . . . . . . . . . table of comparison functions
**
**  'Eq' returns 'HdTrue' if the object '<hdEq>[0]' is equal  to  the  object
**  '<hdEq>[1]' and 'HdFalse'  otherwise.  'Eq'  is  called  from  'EVAL'  to
**  evaluate bags of type 'T_EQ'.
**
**  'Eq' evaluates the operands and then calls the 'EQ' macro.
**
**  'EQ' finds the types of the two operands and  uses  them  to  index  into
**  the table 'TabEq' of comparison functions.
**
**  At places where performance really matters one should  copy  the  special
**  code from 'Eq'  which  checks for the comparison  of  immediate  integers
**  and computes the result without calling 'EQ'.
**
**  'EQ' is defined in the header file of this package as follows:
**
#define EQ(hdL,hdR)     ((*TabEq[GET_TYPE_BAG(hdL)][GET_TYPE_BAG(hdR)])((hdL),(hdR)))
*/
Obj       (*TabEq[EV_TAB_SIZE][EV_TAB_SIZE]) ( Obj, Obj );

Obj       Eq (Obj hd)
{
    Obj           hdL,  hdR;

    hdL = EVAL( PTR_BAG(hd)[0] );  hdR = EVAL( PTR_BAG(hd)[1] );

    /* if the handles are equal the objects certainly will be equal too    */
    if ( hdL == hdR )
        return HdTrue;

    /* Special code to compare two immediate integers.                     */
    if ( ((Int)hdL & (Int)hdR & T_INT) ) {
        if ( HD_TO_INT(hdL) == HD_TO_INT(hdR) )  return HdTrue;
        else                                     return HdFalse;
    }

    return EQ( hdL, hdR );
}


/****************************************************************************
**
*F  Lt( <hdLt> )  . . . . . . . . . . . . . . . . .  test if <objL> <  <objR>
*F  LT(<hdL>,<hdR>) . . . . . . . . . . . . . . . .  test if <objL> <  <objR>
*V  TabLt[<typeL>][<typeR>] , . . . . . . . . . table of comparison functions
**
**  'Lt'  returns 'HdTrue' if  the object '<hdLt>[0]' is less than the object
**  '<hdLt>[1]' and  'HdFalse'  otherwise.  'Lt'  is  called from  'EVAL'  to
**  evaluate bags of type 'T_LT'.
**
**  'Lt' evaluates the operands and then calls the 'LT' macro.
**
**  'LT' finds the types of the two operands and  uses  them  to  index  into
**  the table 'TabLt' of comparison functions.
**
**  At places where performance really matters one should  copy  the  special
**  code  from 'Lt' which  checks for the comparison  of  immediate  integers
**  and computes the result without calling 'LT'.
**
**  'LT' is defined in the header file of this package as follows:
**
#define LT(hdL,hdR)     ((*TabLt[GET_TYPE_BAG(hdL)][GET_TYPE_BAG(hdR)])((hdL),(hdR)))
*/
Obj       (*TabLt[EV_TAB_SIZE][EV_TAB_SIZE]) ( Obj, Obj );

Obj       Lt (Obj hd)
{
    Obj           hdL,  hdR;

    hdL = EVAL( PTR_BAG(hd)[0] );  hdR = EVAL( PTR_BAG(hd)[1] );

    /* if the handles are equal the objects certainly will be equal too    */
    if ( hdL == hdR )
        return HdFalse;

    /* Special code to compare two immediate integers.                     */
    if ( ((Int)hdL & (Int)hdR & T_INT) ) {
        if ( HD_TO_INT(hdL) < HD_TO_INT(hdR) )  return HdTrue;
        else                                    return HdFalse;
    }

    return LT( hdL, hdR );
}


/****************************************************************************
**
*F  Ne( <hdNe> )  . . . . . . . . . . . . . . . . .  test if <objL> <> <objR>
**
**  'Ne'  return 'HdTrue' if  the object <objL> is not equal  to  the  object
**  <objR>.  'Ne' is called from 'EVAL' to evaluate bags of type 'T_NE'.
**
**  'Ne' is simply implemented as 'not <objL> = <objR>'.
*/
Obj       Ne (Obj hd)
{
    Obj           hdL,  hdR;

    hdL = EVAL( PTR_BAG(hd)[0] );  hdR = EVAL( PTR_BAG(hd)[1] );

    /* if the handles are equal the objects certainly will be equal too    */
    if ( hdL == hdR )
        return HdFalse;

    /* Special code to compare two immediate integers.                     */
    if ( ((Int)hdL & (Int)hdR & T_INT) ) {
        if ( HD_TO_INT(hdL) != HD_TO_INT(hdR) )  return HdTrue;
        else                                     return HdFalse;
    }

    /* compute 'not <objL> = <objR>' and return it                         */
    if ( EQ(hdL,hdR) == HdTrue )  hdL = HdFalse;
    else                          hdL = HdTrue;
    return hdL;
}


/****************************************************************************
**
*F  Le( <hdLe> )  . . . . . . . . . . . . . . . . .  test if <objL> <= <objR>
**
**  'Le' returns 'HdTrue' if the object <objL>  is  less  than  or  equal  to
**  the object <objR> and 'HdFalse' otherwise.  'Le' is  called  from  'EVAL'
**  to evaluate bags of type 'T_LE'.
**
**  'Le' is simply implemented as 'not <objR> < <objL>'.
*/
Obj       Le (Obj hd)
{
    Obj           hdL,  hdR;

    hdL = EVAL( PTR_BAG(hd)[0] );  hdR = EVAL( PTR_BAG(hd)[1] );

    /* if the handles are equal the objects certainly will be equal too    */
    if ( hdL == hdR )
        return HdTrue;

    /* Special code to compare two immediate integers.                     */
    if ( ((Int)hdL & (Int)hdR & T_INT) ) {
        if ( HD_TO_INT(hdL) <= HD_TO_INT(hdR) )  return HdTrue;
        else                                     return HdFalse;
    }

    /* compute 'not <objR> < <objL>' and return it                         */
    if ( LT( hdR, hdL ) == HdTrue )  hdL = HdFalse;
    else                             hdL = HdTrue;
    return hdL;
}


/****************************************************************************
**
*F  Gt( <hdLe> )  . . . . . . . . . . . . . . . . .  test if <objL> >  <objR>
**
**  'Gt' returns 'HdTrue' if the object <objL>  is greater  than  the  object
**  <objR> and 'HdFalse' otherwise.  'Gt' is called from 'EVAL'  to  evaluate
**  bags of type 'T_GT'.
**
**  'Gt' is simply implemented as '<objR> < <objL>'.
*/
Obj       Gt (Obj hd)
{
    Obj    hdL,  hdR;

    hdL = EVAL( PTR_BAG(hd)[0] );  hdR = EVAL( PTR_BAG(hd)[1] );

    /* if the handles are equal the objects certainly will be equal too    */
    if ( hdL == hdR )
        return HdFalse;

    /* Special code to compare two immediate integers.                     */
    if ( ((Int)hdL & (Int)hdR & T_INT) ) {
        if ( HD_TO_INT(hdL) >  HD_TO_INT(hdR) )  return HdTrue;
        else                                     return HdFalse;
    }

    return LT( hdR, hdL );
}


/****************************************************************************
**
*F  Ge( <hdLe> )  . . . . . . . . . . . . . . . . .  test if <objL> >= <objR>
**
**  'Ge' returns 'HdTrue' if the object  <objL>  is  greater  or  equal  than
**  the object <objR> and 'HdFalse' otherwise.  'Le' is  called  from  'EVAL'
**  to evaluate bags of type 'T_GE'.
**
**  'Ge' is simply implemented as 'not <objL> < <objR>'.
*/
Obj       Ge (Obj hd)
{
    Obj           hdL,  hdR;

    hdL = EVAL( PTR_BAG(hd)[0] );  hdR = EVAL( PTR_BAG(hd)[1] );

    /* if the handles are equal the objects certainly will be equal too    */
    if ( hdL == hdR )
        return HdTrue;

    /* Special code to compare two immediate integers.                     */
    if ( ((Int)hdL & (Int)hdR & T_INT) ) {
        if ( HD_TO_INT(hdL) >= HD_TO_INT(hdR) )  return HdTrue;
        else                                     return HdFalse;
    }

    /* compute 'not <objL> < <objR>' and return it                         */
    if ( LT( hdL, hdR ) == HdTrue )  hdL = HdFalse;
    else                             hdL = HdTrue;
    return hdL;
}


/****************************************************************************
**
*F  IsTrue( <hdL>, <hdR> )  . . . . . . . .  default function for comparisons
**
**  'IsTrue' always returns  'HdTrue'  no  matter  what  the  arguments  are.
**  Is is used for those comparisons where already the types of the  operands
**  determines the outcome.  E.g., it is  used  above  the  diagonal  of  the
**  'TabLt' table.
*/
/*ARGSUSED*/
Obj       IsTrue (Obj hdL, Obj hdR)
{
    return HdTrue;
}


/****************************************************************************
**
*F  IsFalse( <hdL>, <hdR> ) . . . . . . . .  default function for comparisons
**
**  'IsFalse' always returns 'HdFalse' no  matter  what  the  arguments  are.
**  Is is used for those comparisons where already the types of the  operands
**  determines the outcome.  E.g., it is  used  below  the  diagonal  of  the
**  'TabLt' table.
*/
/*ARGSUSED*/
Obj       IsFalse (Obj hdL, Obj hdR)
{
    return HdFalse;
}


/****************************************************************************
**
*F  ProtectVar( <hdVar> ) . . . forbid variable overwriting from GAP session
*F  UnprotectVar( <hdVar> ) . . . . . . . . . .removes ProtectVar protection
*/
void       ProtectVar(Obj hdVar) {
    SET_FLAG_BAG(hdVar, BF_PROTECT_VAR);
}

void       UnprotectVar(Obj hdVar) {
    CLEAR_FLAG_BAG(hdVar, BF_PROTECT_VAR);
}

/****************************************************************************
**
*F  FunProtectVar( <hdVar> ) . . . . . . . . . internal function ProtectVar()
**
**  ProtectVar(var) forbids variable overwriting from GAP session.
**/
Obj        FunProtectVar(Obj hdCall) {
    Obj hdVar;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error("Usage: ProtectVar(<variable>)", 0, 0);
    hdVar = PTR_BAG(hdCall)[1];
    if ( GET_TYPE_BAG(hdVar) != T_VAR && GET_TYPE_BAG(hdVar) != T_VARAUTO )
        return Error("Usage: ProtectVar(<variable>)", 0, 0);
    ProtectVar(PTR_BAG(hdCall)[1]);
    return HdVoid;
}

Obj        FunProtectRec(Obj hdCall) {
    Obj hdRec;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error("Usage: ProtectRec(<variable>)", 0, 0);
    hdRec = EVAL(PTR_BAG(hdCall)[1]);
    if ( GET_TYPE_BAG(hdRec) != T_REC )
        return Error("Usage: ProtectRec(<record>)", 0, 0);
    ProtectVar(hdRec);
    return HdVoid;
}

Obj        FunProtectNamespace(Obj hdCall) {
    Obj hdNS;
    UInt size, i;

        if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
                return Error("Usage: ProtectNamespace(<variable>)", 0, 0);

    hdNS = EVAL(PTR_BAG(hdCall)[1]);

    if ( GET_TYPE_BAG(hdNS) != T_NAMESPACE )
                return Error("Usage: ProtectNamespace(<namespace>)", 0, 0);

    size = TableSize(hdNS);
    for ( i = 0; i < size; ++i ) {
        Obj var = PTR_BAG(hdNS)[i];
        if (var != 0) {
        	ProtectVar(var);
        	var = EVAL(var);
        	if(var != 0){
        		if((GET_TYPE_BAG(var)==T_FUNCTION)||(GET_TYPE_BAG(var)==T_METHOD)) ProtectVar(var);
        	}
        }
    }
    return HdVoid;
}

Obj        Fun_UnprotectVar(Obj hdCall) {
    Obj hdVar = PTR_BAG(hdCall)[1];
    if ( GET_TYPE_BAG(hdVar) != T_VAR && GET_TYPE_BAG(hdVar) != T_VARAUTO )
        return Error("Usage: UnprotectVar(<variable>)", 0, 0);
    UnprotectVar(PTR_BAG(hdCall)[1]);
    return HdVoid;
}
/****************************************************************************
**
*F  EvVar( <hdVar> )  . . . . . . . . . . . . . . . . . . evaluate a variable
**
**  'EvVar' returns the value  of  the  variable  with  the  handle  <hdVar>.
**
**  The value is the only subobject <hdVar>.  If this has the handle '0' then
**  no value has been assigned to the variable and  an  error  is  generated.
*/
Obj       EvVar (Obj hdVar)
{
    if ( PTR_BAG(hdVar)[0] == 0 )
        return Error("Variable: '%s' must have a value",
                     (Int)(PTR_BAG(hdVar)+OFS_IDENT), 0 );
    if ( GET_FLAG_BAG(hdVar, BF_VAR_AUTOEVAL) )
        return EVAL(VAR_VALUE(hdVar));
    else
        return VAR_VALUE(hdVar);

}


/****************************************************************************
**
*F  EvVarAuto( <hdVar> )  . . . . . . . . . . . . . eval an autoread variable
*/
Obj       EvVarAuto (Obj hdVar)
{
    Obj           ignore;

    /* evaluate the value cell, unless it is already a constant            */
    if ( T_VAR <= GET_TYPE_BAG( PTR_BAG(hdVar)[0] ) ) {
        ignore = EVAL( PTR_BAG(hdVar)[0] );
        if ( T_VAR <= GET_TYPE_BAG( PTR_BAG(hdVar)[0] ) ) {
            return Error("AUTO: '%s' must be defined by the evaluation",
                         (Int)(PTR_BAG(hdVar)+OFS_IDENT), 0 );
        }
    }

    /* convert the autoread variable to a normal one                       */
    Retype( hdVar, T_VAR );

    /* return the value                                                    */
    return PTR_BAG(hdVar)[0];
}


/****************************************************************************
**
*F  EvVarAss( <hdAss> ) . . . . . . . . . . . . . . . . execute an assignment
**
**  'EvVarAss' assigns the value of '<hdAss>[1]' to the variable '<hdAss>[0]'
**  and returns the value so that it can be printed in the ReadEvalPrintLoop.
**
**  'EvVarAss' is called from 'EVAL' for bags of type 'T_VARASS'.
*/
Obj       EvVarAss (Obj hdAss)
{
    Obj           hdVal;
    Obj           hdVar;

    hdVar = PTR_BAG(hdAss)[0];
    if ( GET_FLAG_BAG(hdVar, BF_PROTECT_VAR) ) {
        return Error("Assignment: variable %g is write-protected", (Int)hdVar, 0);
    }
    else {
        hdVal = EVAL( PTR_BAG(hdAss)[1] );
        if ( hdVal == HdVoid )
            return Error("Assignment: function must return a value",0,0);
    }
    SET_VAR_VALUE( hdVar, hdVal);
    DocumentValue( hdVal, PTR_BAG(hdAss)[2] );
    return hdVal;
}


/****************************************************************************
**
*F  EvBool( <hdBool> )  . . . . . . . . . . . . . .  evaluate a boolean value
**
**  'EvBool' returns the value of the boolean value <hdBool>.  Since  boolean
**  values are constants and thus selfevaluating it just returns <hdBool>.
*/
Obj       EvBool (Obj hdBool)
{
    return hdBool;
}


/****************************************************************************
**
*F  EvNot( <hdBool> ) . . . . . . . . . . . . . . . .  negate a boolean value
**
**  'EvNot' returns the boolean negation of the boolean value <hdBool>, i.e.,
**  it returns 'HdTrue' if <hdBool> is 'HdFalse' and vica versa.
*/
Obj       EvNot (Obj hdBool)
{
    /* evaluate the operand                                                */
    hdBool = EVAL( PTR_BAG(hdBool)[0] );

    /* check that it is 'true' or 'false' and return the negation          */
    if ( hdBool == HdTrue )
        return HdFalse;
    else if ( hdBool == HdFalse )
        return HdTrue;
    else
        return Error("not: <expr> must evaluate to 'true' or 'false'",0,0);
}


/****************************************************************************
**
*F  EvAnd( <hdAnd> )  . . . . . . . . . . .  evaluate a boolean and operation
**
**  'EvAnd' returns the logical and  of  the  two  operand  '<hdAnd>[0]'  and
**  '<hdAnd>[1]' which must be boolean values.
**
**  If '<hdAnd>[0]' is already  'false'  'EvAnd'  returns  'HdFalse'  without
**  evaluating '<hdAnd>[1]'.  This allows constructs like
**
**      if index <= max  and list[index] = 0  then ... fi;
*/
Obj       EvAnd (Obj hd)
{
    Obj           hd1;

    /* evaluate and check the left operand                                 */
    hd1 = EVAL( PTR_BAG(hd)[0] );
    if ( hd1 == HdFalse )
        return HdFalse;
    else if ( hd1 != HdTrue )
        return Error("and: <expr> must evaluate to 'true' or 'false'",0,0);

    /* evaluate and check the right operand                                */
    hd1 = EVAL( PTR_BAG(hd)[1] );
    if ( hd1 == HdFalse )
        return HdFalse;
    else if ( hd1 != HdTrue )
        return Error("and: <expr> must evaluate to 'true' or 'false'",0,0);

    return HdTrue;
}


/****************************************************************************
**
*F  EvIs( <hdIs> )  . . . . . . . . . . .  evaluate a boolean 'is' operation
**
**  'EvIs' evaliates boolean expression '_ObjId(<hdAnd>[0])=<hdAnd>[1]'
**
*/
Obj       EvIs (Obj hd)
{
    Obj hdL   = 0;
	Obj hdR   = 0;
    Obj hdRes = 0;
	Obj hdEQ  = 0;

    /* evaluate and check the left operand                                 */
    hdL = _ObjId(EVAL( PTR_BAG(hd)[0] ));
    hdR = EVAL( PTR_BAG(hd)[1] );
	if (!IS_LIST(hdR)) {
        hdEQ  = NewBag(T_EQ, 2*SIZE_HD);
	    SET_BAG(hdEQ, 0,  hdL );
	    SET_BAG(hdEQ, 1,  hdR );
        hdRes = EVAL(hdEQ);
	} else {
        /* search the element                                              */
        Int pos = POS_LIST( hdR, hdL, 0 );
        /* return the position                                             */
        hdRes = (pos != 0) ? HdTrue : HdFalse;
	}

	return hdRes;
}

/****************************************************************************
**
*F  EvOr( <hdOr> )  . . . . . . . . . . . . . evaluate a boolean or operation
**
**  'EvOr' returns the  logical  or  of  the  two  operands  '<hdOr>[0]'  and
**  '<hdOr>[1]' which must be boolean values.
**
**  If '<hdOr>[0]' is already 'true' 'EvOr' returns 'true' without evaluating
**  '<hdOr>[1]'.  This allows constructs like
**
**      if index > max  or list[index] = 0  then ... fi;
*/
Obj       EvOr (Obj hd)
{
    Obj           hd1;

    /* evaluate and check the left operand                                 */
    hd1 = EVAL( PTR_BAG(hd)[0] );
    if ( hd1 == HdTrue )
        return HdTrue;
    else if ( hd1 != HdFalse )
        return Error("or: <expr> must evaluate to 'true' or 'false'",0,0);

    /* evaluate and check the right operand                                */
    hd1 = EVAL( PTR_BAG(hd)[1] );
    if ( hd1 == HdTrue )
        return HdTrue;
    else if ( hd1 != HdFalse )
        return Error("or: <expr> must evaluate to 'true' or 'false'",0,0);

    return HdFalse;
}


/****************************************************************************
**
*F  EqBool( <hdL>, <hdR> )  . . . . . . . . . . .  test if <boolL> =  <boolR>
**
**  'EqBool' returns 'HdTrue' if the  two  boolean  values  <hdL>  and  <hdR>
**  are equal, and 'HdFalse' otherwise.
*/
Obj       EqBool (Obj hdL, Obj hdR)
{
    if ( hdL == hdR )  return HdTrue;
    else               return HdFalse;
}


/****************************************************************************
**
*F  EqPtr( <hdL>, <hdR> )  . . . . . . . . . . . . . . . .. compares pointers
**
**  'EqPtr' provides an ordering function for arbitrary objects by comparing
**  masterpointers
*/
Obj       EqPtr (Obj hdL, Obj hdR)
{
    if ( hdL == hdR )  return HdTrue;
    else              return HdFalse;
}


/****************************************************************************
**
*F  LtBool( <hdL>, <hdR> )  . . . . . . . . . . .  test if <boolL> <  <boolR>
**
**  'LtBool' return 'HdTrue' if  the  boolean value <hdL> is  less  than  the
**  boolean value <hdR> and 'HdFalse' otherwise.
*/
Obj       LtBool (Obj hdL, Obj hdR)
{
    if ( hdL == HdTrue && hdR == HdFalse )  return HdTrue;
    else                                    return HdFalse;
}

/****************************************************************************
**
*F  LtPtr( <hdL>, <hdR> )  . . . . . . . . . . . . . . . .. compares pointers
**
**  'LtPtr' provides an ordering function for arbitrary objects by comparing
**  masterpointers
*/
Obj       LtPtr (Obj hdL, Obj hdR)
{
    if ( hdL < hdR )  return HdTrue;
    else              return HdFalse;
}

/****************************************************************************
**
*F  PrBool( <hdBool> )  . . . . . . . . . . . . . . . . print a boolean value
**
**  'PrBool' prints the boolean value <hdBool>.
*/
void            PrBool (Obj hd)
{
    if ( hd == HdTrue )  Pr("true",0,0);
    else                 Pr("false",0,0);
}


/****************************************************************************
**
*F  FunIsBool( <hdCall> ) . . . . . . . . . internal function IsBool( <obj> )
**
**  'IsBool' returns 'true' if the object <obj>  is  a  boolean  and  'false'
**  otherwise.  May cause an error if <obj> is an unbound variable.
*/
Obj       FunIsBool (Obj hdCall)
{
    Obj           hdObj;

    /* evaluate and check the argument                                     */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: IsBool( <obj> )",0,0);
    hdObj = EVAL( PTR_BAG(hdCall)[1] );
    if ( hdObj == HdVoid )
        return Error("IsBool: function must return a value",0,0);

    /* return 'true' if <obj> is a boolean and 'false' otherwise           */
    if ( GET_TYPE_BAG(hdObj) == T_BOOL )
        return HdTrue;
    else
        return HdFalse;
}

/****************************************************************************
**
*F  ShallowCopy( <hdOld> )  . . . . . . . .  make a shallow copy of an object
*F  FullShallowCopy( <hdOld> )  . . . . make a full shallow copy of an object
**
**  'ShallowCopy' makes a copy of the object  <obj>.  If <obj> is not a  list
**  or a record, 'ShallowCopy' simply returns <obj>, since those objects  can
**  never be modified there is no way to distinguish the original object from
**  any copy, so we might as well not copy it.  If < obj>  is  a  list  or  a
**  record 'ShallowCopy' makes a copy of this object,  but does not copy  the
**  subobjects.
**
**  'FullShallowCopy' is identical to ShallowCopy,  except  that  ShallowCopy
**  does not copy objects with type > T_VARAUTO, while FullShalloCopy does.
**/
Obj       ShallowCopy (Obj hdOld)
{
    Obj           * ptOld;        /* pointer to the old object       */
    Obj           hdNew;          /* handle of the new object        */
    Obj           * ptNew;        /* pointer to the new object       */
    UInt       i;              /* loop variable                   */

    /* for mutable objects copy the bag                                    */
    if ( IS_MUTABLE(hdOld)  && ! GET_FLAG_BAG(hdOld, BF_NO_COPY)) {
        hdNew = NewBag( GET_TYPE_BAG(hdOld), GET_SIZE_BAG(hdOld) );
        ptOld = PTR_BAG(hdOld);
        ptNew = PTR_BAG(hdNew);
        for ( i = (GET_SIZE_BAG(hdOld)+SIZE_HD-1)/SIZE_HD; 0 < i; i-- )
            *ptNew++ = *ptOld++;

        /* special handling of T_DELAY */
        if ( GET_TYPE_BAG(hdNew) == T_DELAY ) {
            Obj hd = ShallowCopy(PTR_BAG(hdNew)[0]);
            SET_BAG(hdNew, 0,  hd );
        }
    }

    /* otherwise return the original object                                */
    else {
        hdNew = hdOld;
    }

    return hdNew;
}

Obj       FullShallowCopy (Obj hdOld)
{
    Obj hdResult;
    DoFullCopy = 1;
    hdResult = ShallowCopy(hdOld);
    DoFullCopy = 0;
    return hdResult;
}

/****************************************************************************
**
*F  FunShallowCopy( <hdCall> )  . . . . . .  make a shallow copy of an object
**
**  'FunShallowCopy' implements the internal functin 'ShallowCopy( <obj> )'.
**
**  'ShallowCopy' makes a copy of the object  <obj>.  If <obj> is not a  list
**  or a record, 'ShallowCopy' simply returns <obj>, since those objects  can
**  never be modified there is no way to distinguish the original object from
**  any copy, so we might as well not copy it.  If < obj>  is  a  list  or  a
**  record 'ShallowCopy' makes a copy of this object,  but does not copy  the
**  subobjects.
*/

Obj       FunShallowCopy (Obj hdCall)
{
    Obj           hdOld;          /* handle of the old object        */
    /* check the argument                                                  */
    if ( GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
        return Error("usage: ShallowCopy( <obj> )",0,0);
    /* evaluate the argument                                               */
    hdOld = EVAL( PTR_BAG(hdCall)[1] );
    return FullShallowCopy(hdOld);
}


/****************************************************************************
**
*F  Copy( <hdObj> ) . . . . . . . . . . . . . make a normal copy of an object
*F  CopyFunc( <hdObj> ) . . . . . . . . . . . . . . make a copy of a function
*F  FullCopy( <hdObj> ) . . . . . . . . . . . . make a full copy of an object
**
**  'Copy' makes a copy of the  object <hdObj>.  If <obj>  is not a list or a
**  record, 'Copy' simply  returns  <obj>, since those  objects can  never be
**  modified there  is no way  to  distinguish the  original object  from any
**  copy, so we might as  well not copy  it.  If <obj>  is a list or a record
**  'Copy' makes a copy of this object,  and calls itself recursively to copy
**  the subobjects.
**
**  'FullCopy' is identical to Copy, except that Copy() does not copy objects
**  with type > T_VARAUTO, while FullCopy does.
*/

Obj       FullCopy (Obj hdOld)
{
    Obj hdResult;
    DoFullCopy = 1;
    hdResult = Copy(hdOld);
    DoFullCopy = 0;
    return hdResult;
}

Obj       CopyShadow (Obj hdOld)
{
    Obj           hdNew;          /* shadow of <hdOld>               */
    Obj           hdTmp;          /* shadow of element of <hdOld>    */
    UInt       n;              /* number of handles of <hdOld>    */
    UInt       i;              /* loop variable                   */

    if ( GET_FLAG_BAG(hdOld, BF_COPY) )
        return GET_COPY_BAG(hdOld);
    else if ( GET_FLAG_BAG(hdOld, BF_NO_COPY) )
        return hdOld;

    /* make a shadow of the old bag                                        */
    hdNew = NewBag( GET_TYPE_BAG(hdOld), GET_SIZE_BAG(hdOld) );
    SET_FLAG_BAG(hdOld, BF_COPY);
    SET_COPY_BAG(hdOld, hdNew);

    /* and make recursively shadows of the subobjects                      */
    n = NrHandles( GET_TYPE_BAG(hdOld), GET_SIZE_BAG(hdOld) );
    for ( i = n; 0 < i; i-- ) {
        hdTmp = PTR_BAG(hdOld)[i-1];
        if ( hdTmp != 0 && IS_MUTABLE(hdTmp) ) {
            hdTmp = CopyShadow( hdTmp );
            SET_BAG(hdNew, i-1,  hdTmp );
        }
        else SET_BAG(hdNew, i-1,  0 );
    }

    SET_FLAG_BAG(hdNew, GET_FLAGS_BAG(hdOld) & (BF_ENVIRONMENT | BF_ENV_VAR /* | BF_WEAKREFS */ ));

    /* return the shadow                                                   */
    return hdNew;
}

void            CopyCopy (Obj hdOld, Obj hdNew)
                                        /* old bag                         */
                                        /* shadow of <hdOld>               */
{
    UInt       n;              /* number of handles of <hdOld>    */
    UInt       i;              /* loop variable                   */

    if ( ! GET_FLAG_BAG(hdOld, BF_COPY) )
        return;
    CLEAR_FLAG_BAG(hdOld, BF_COPY);

    /* copy the data area                                                  */
    n = NrHandles( GET_TYPE_BAG(hdOld), GET_SIZE_BAG(hdOld) );
    for ( i = (GET_SIZE_BAG(hdOld)+SIZE_HD-1)/SIZE_HD; n < i; i-- ) {
        ((Bag*)(PTR_BAG(hdNew)))[i-1] = PTR_BAG(hdOld)[i-1];
    }

    /* copy the handles area                                               */
    for(i = n; 0 < i; i--) {
        if(PTR_BAG(hdOld)[i-1] != 0 && PTR_BAG(hdNew)[i-1] != 0)
            CopyCopy(PTR_BAG(hdOld)[i-1], PTR_BAG(hdNew)[i-1]);
        else
            SET_BAG(hdNew, i-1,  PTR_BAG(hdOld)[i-1] );
    }

}

Obj       Copy (Obj hdOld)
{
    Obj           hdNew;          /* copy of <hdOld>                 */

    /* copy mutable objects                                                */
    if ( IS_MUTABLE(hdOld) && ! GET_FLAG_BAG(hdOld, BF_NO_COPY) ) {
        hdNew = CopyShadow( hdOld );
        CopyCopy( hdOld, hdNew );
    }
    /* for other objects simply return the object                          */
    else {
        hdNew = hdOld;
    }

    return hdNew;
}

Obj       CopyFunc (Obj hdOld)
{
    Obj           hdNew;          /* copy of <hdOld>                 */
    Int                type;

    type = GET_TYPE_BAG(hdOld);
    if ( type != T_FUNCTION && type != T_METHOD)
        Error("CopyFunc() expects a function or method", 0, 0);

    DoFullCopy = 1;
    hdNew = CopyShadow( hdOld );
    CopyCopy( hdOld, hdNew );
    RecursiveClearFlagMutable(hdOld, BF_COPY);
    DoFullCopy = 0;

    return hdNew;
}


/****************************************************************************
**
*F  FunCopy( <hdCall> ) . . . . . . . . . . . . . .  make a copy of an object
*F  FunCopyFunc( <hdCall> ) . . . . . . . . . . . . make a copy of a function
**
**  'FunCopy' implements the internal function 'Copy( <obj> )'.
**  'FunCopyFunc' implements the internal function 'CopyFunc( <obj> )'.
**
**  'Copy' makes a copy of the  object <hdObj>.  If <obj>  is not a list or a
**  record, 'Copy' simply  returns  <obj>, since those  objects can  never be
**  modified there  is no way  to  distinguish the  original object  from any
**  copy, so we might as  well not copy  it.  If <obj>  is a list or a record
**  'Copy' makes a copy of this object,  and calls itself recursively to copy
**  the subobjects.
**
**  'CopyFunc' copies a function, it is necessary when function bags are to be
**  modified from within the interpreter (using Child/SetChild functions).
*/
Obj       FunCopy (Obj hdCall)
{
    /* check the argument                                                  */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: Copy( <obj> )",0,0);

    /* return a copy of the object                                         */
    return Copy( EVAL( PTR_BAG(hdCall)[1] ) );
}

Obj       FunCopyFunc (Obj hdCall)
{
    char * usage = "usage: CopyFunc( <func> )";
    /* check the argument                                                  */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage, 0, 0);
    return CopyFunc( EVAL( PTR_BAG(hdCall)[1] ) );
}

/****************************************************************************
**
*F  FunIsBound( <hdCall> )  . . . .  test if a variable has an assigned value
**
**  'FunIsBound' implements the internal function 'IsBound( <expr> )'.
**
*/
Obj       FunIsBound (Obj hdCall)
{
    Obj           hd, hdList, hdInd, hdRec, hdNam, Result;

    /* check the argument                                                  */
    if ( GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
        return Error("usage: IsBound( <obj> )",0,0);
    hd = PTR_BAG(hdCall)[1];
    if ( GET_TYPE_BAG(hd) != T_VAR     && GET_TYPE_BAG(hd) != T_VARAUTO
      && GET_TYPE_BAG(hd) != T_LISTELM && GET_TYPE_BAG(hd) != T_RECELM )
        return Error("IsBound: <obj> must be a variable",0,0);

    /* easiest case first                                                  */
    if ( GET_TYPE_BAG(hd) == T_VAR ) {
        if ( PTR_BAG(hd)[0] != 0 )
            Result = HdTrue;
        else
            Result = HdFalse;
    }

    /* an variable that autoreads a file is considered bound               */
    else if ( GET_TYPE_BAG(hd) == T_VARAUTO ) {
        Result = HdTrue;
    }

    /* is a list element bound                                             */
    else if ( GET_TYPE_BAG(hd) == T_LISTELM ) {
        hdList = EVAL( PTR_BAG(hd)[0] );
        if ( ! IS_LIST(hdList) )
            return Error("IsBound: <list> must be a list",0,0);
        hdInd = EVAL( PTR_BAG(hd)[1] );
        if ( GET_TYPE_BAG(hdInd) != T_INT || HD_TO_INT(hdInd) <= 0 )
            return Error("IsBound: <index> must be positive int",0,0);
        if ( HD_TO_INT(hdInd) <= LEN_LIST(hdList)
          && ELMF_LIST(hdList,HD_TO_INT(hdInd)) != 0 )
            Result = HdTrue;
        else
            Result = HdFalse;
    }

    /* is a record element bound                                           */
    else {
        hdRec = EVAL( PTR_BAG(hd)[0] );
        if ( GET_TYPE_BAG(hdRec) != T_REC && GET_TYPE_BAG(hdRec) != T_NAMESPACE)
            return Error("IsBound: <record> must be a record",0,0);

        hdNam = RecnameObj( PTR_BAG(hd)[1] );
        if ( GET_TYPE_BAG(hdRec) == T_REC ) {
            Obj tmp = 0;
            if ( FindRecnameRec(hdRec, hdNam, &tmp) == 0 ) Result = HdFalse;
            else Result = HdTrue;
        }
        else {
            UInt k = TableLookup(hdRec, RECNAM_NAME(hdNam), OFS_IDENT);
            Obj hdIdent = PTR_BAG(hdRec)[k];
            if ( hdIdent != 0 && VAR_VALUE(hdIdent)!=0) Result = HdTrue;
            else Result = HdFalse;
        }
    }

    return Result;
}


/****************************************************************************
**
*F  FunUnbind( <hdCall> ) . . . . . . . . . . . . . . . . unassign a variable
**
**  'FunUnbind' implements the internal function 'Unbind( <expr> )'.
*/
Obj       FunUnbind (Obj hdCall)
{
    Obj           hd, hdList, hdInd, hdRec, hdNam;
    UInt       i;              /* loop variable                   */

    /* check the argument                                                  */
    if ( GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
        return Error("usage: Unbind( <obj> )",0,0);
    hd = PTR_BAG(hdCall)[1];
    if ( GET_TYPE_BAG(hd) != T_VAR     && GET_TYPE_BAG(hd) != T_VARAUTO
      && GET_TYPE_BAG(hd) != T_LISTELM && GET_TYPE_BAG(hd) != T_RECELM )
        return Error("Unbind: <obj> must be a variable",0,0);

    /* easiest case first                                                  */
    if ( GET_TYPE_BAG(hd) == T_VAR ) {
        SET_BAG(hd, 0,  0 );
    }

    /* an variable that autoreads a file is considered bound               */
    else if ( GET_TYPE_BAG(hd) == T_VARAUTO ) {
        Retype( hd, T_VAR );
        SET_BAG(hd, 0,  0 );
    }

    /* is a list element bound                                             */
    else if ( GET_TYPE_BAG(hd) == T_LISTELM ) {
        hdList = EVAL( PTR_BAG(hd)[0] );
        if ( ! IS_LIST( hdList ) )
            return Error("Unbind: <list> must be a list",0,0);
        PLAIN_LIST( hdList );
        Retype( hdList, T_LIST );
        hdInd = EVAL( PTR_BAG(hd)[1] );
        if ( GET_TYPE_BAG(hdInd) != T_INT || HD_TO_INT(hdInd) <= 0 )
            return Error("Unbind: <index> must be positive int",0,0);
        i = HD_TO_INT(hdInd);
        if ( i < LEN_PLIST(hdList) ) {
            SET_ELM_PLIST( hdList, i, 0 );
        }
        else if ( i == LEN_PLIST( hdList ) ) {
            SET_ELM_PLIST( hdList, i, 0 );
            while ( 0 < i && ELM_PLIST( hdList, i ) == 0 )
                i--;
            SET_LEN_PLIST( hdList, i );
        }
    }

    /* is a record element bound                                           */
    else {
        hdRec = EVAL( PTR_BAG(hd)[0] );
        if ( GET_TYPE_BAG(hdRec) != T_REC && GET_TYPE_BAG(hdRec) != T_NAMESPACE )
            return Error("Unbind: <record> must be a record",0,0);
        hdNam = RecnameObj( PTR_BAG(hd)[1] );

        if ( GET_TYPE_BAG(hdRec) == T_REC ) {
            for ( i = 0; i < GET_SIZE_BAG(hdRec)/(2*SIZE_HD); i++ ) {
                if ( PTR_BAG(hdRec)[2*i] == hdNam )
                    break;
            }
            if ( i < GET_SIZE_BAG(hdRec)/(2*SIZE_HD) ) {
                while ( i < GET_SIZE_BAG(hdRec)/(2*SIZE_HD)-1 ) {
                    SET_BAG(hdRec, 2*i,  PTR_BAG(hdRec)[2*i+2] );
                    SET_BAG(hdRec, 2*i+1,  PTR_BAG(hdRec)[2*i+3] );
                    i++;
                }
                Resize( hdRec, GET_SIZE_BAG(hdRec)-2*SIZE_HD );
            }
        }
        /* table (namespace) element */
        else {
            UInt k = TableLookup(hdRec, RECNAM_NAME(hdNam), OFS_IDENT);
            if(PTR_BAG(hdRec)[k] != 0)
                SET_VAR_VALUE(PTR_BAG(hdRec)[k], 0);
        }
    }
    return HdVoid;
}


/****************************************************************************
**
*V  PrTab[<type>] . . . . . . .  printing function for objects of type <type>
**
**  is the main dispatching table that contains for every type a  pointer  to
**  the function that should be executed if a bag  of  that  type  is  found.
*/
void            (* PrTab[ T_ILLEGAL ] ) ( Obj hd );


/****************************************************************************
**
*F  Print( <hd> ) . . . . . . . . . . . . . . . . . . . . . . print an object
**
**  'Print'  prints  the  object  with  handle  <hd>.  It dispatches   to the
**  appropriate function stored in 'PrTab[GET_TYPE_BAG(<hd>)]'.
*/
Obj       HdTildePr;

void            Print (Obj hd)
{
    UInt       len;            /* hdObj[1..<len>] are a path from */
    Obj           hdObj[256];     /* '~' to <hd>, where hdObj[<i>+1] */
    UInt       index[256];     /* is PTR_BAG(hdObj[<i>])[index[<i>]]  */
    Obj           cur;            /* current object along that path  */
    UInt       i;              /* loop variable                   */
    exc_type_t          e;

    /* check for interrupts                                                */
    if ( SyIsIntr() ) {
        Pr( "%c", (Int)'\03', 0 );
        /*N 19-Jun-90 martin do something about the current indent         */
        DbgBreak("user interrupt while printing", 0, 0);
    }

    if ( hd == 0 )
        Pr("_null_", 0, 0);

    else if ( ! IS_BAG(hd) && ! IS_INTOBJ(hd) )
        Pr("_invalid_%d_", (Int)hd, 0);

    /* print new objects                                                   */
    else if ( GET_TYPE_BAG(hd) == T_INT || ! GET_FLAG_BAG(hd, BF_PRINT) ) {
        Try {
            /* assign the current object to '~' if this is it              */
            if ( PTR_BAG(HdTildePr)[0] == 0 )
                SET_BAG(HdTildePr, 0,  hd );

            /* mark objects for '~...' detection                           */
            if ( (T_LIST <= GET_TYPE_BAG(hd) && GET_TYPE_BAG(hd) < T_VAR)
                 || GET_TYPE_BAG(hd) == T_PERM16 || GET_TYPE_BAG(hd) == T_PERM32 )
                SET_FLAG_BAG(hd, BF_PRINT);

            /* dispatch to the appropriate method                          */
            (* PrTab[ GET_TYPE_BAG(hd) ] ) (hd);

            /* unmark object again                                         */
            if ( (T_LIST <= GET_TYPE_BAG(hd) && GET_TYPE_BAG(hd) < T_VAR)
                 || GET_TYPE_BAG(hd) == T_PERM16 || GET_TYPE_BAG(hd) == T_PERM32 )
                CLEAR_FLAG_BAG(hd, BF_PRINT);

            /* unassign '~' again                                          */
            if ( hd == PTR_BAG(HdTildePr)[0] )
                SET_BAG(HdTildePr, 0,  0 );
        }
        Catch(e) {
            /* Cleanup, this ensures that CTRL-C interrupt will be Ok      */

            /* unmark object again                                         */
            if ( (T_LIST <= GET_TYPE_BAG(hd) && GET_TYPE_BAG(hd) < T_VAR)
                 || GET_TYPE_BAG(hd) == T_PERM16 || GET_TYPE_BAG(hd) == T_PERM32 )
                CLEAR_FLAG_BAG(hd, BF_PRINT);
            /* unassign '~' again                                          */
            if ( hd == PTR_BAG(HdTildePr)[0] )
                SET_BAG(HdTildePr, 0,  0 );

            Throw(e);
        }
    }

    /* handle common subobject                                             */
    else {

        /* find the subobject in the object again by a backtrack search    */
        len = 0;
        hdObj[0] = HdTildePr;
        index[0] = 0;
        cur = PTR_BAG( hdObj[len] )[ index[len] ];
        while ( hd != cur ) {
            for ( i = 0; i <= len && hdObj[i] != cur; i++ ) ;
            if ( cur != 0
              && (GET_TYPE_BAG(cur)==T_LIST || GET_TYPE_BAG(cur)==T_SET || GET_TYPE_BAG(cur)==T_REC)
              && GET_SIZE_BAG(cur) != 0
              && len < i ) {
                len++;
                hdObj[len] = cur;
                index[len] = 0;
                cur = PTR_BAG( hdObj[len] )[ index[len] ];
            }
            else if ( index[len] < GET_SIZE_BAG(hdObj[len])/SIZE_HD-1 ) {
                index[len]++;
                cur = PTR_BAG( hdObj[len] )[ index[len] ];
            }
            else {
                if ( len != 0 )  len--;
                cur = 0;
            }
        }

        /* print the path just found                                       */
        for ( i = 0; i <= len; i++ ) {
            if ( GET_TYPE_BAG(hdObj[i]) == T_VAR )
                Pr("~",0,0);
            else if ( GET_TYPE_BAG(hdObj[i])==T_LIST || GET_TYPE_BAG(hdObj[i])==T_SET )
                Pr("[%d]",index[i],0);
            else
                Pr(".%s",(Int)PTR_BAG(PTR_BAG(hdObj[i])[index[i]-1]),0);
        }

    }

}


/****************************************************************************
**
*F  CantPrint( <hd> ) . . . . . . . . . . . . . illegal bag printing function
**
**  Is called if a illegal bag should be  printed,  it  generates  an  error.
**  If this is actually ever executed in GAP it  indicates  serious  trouble,
**  for  example  that  the  type  field  of  a  bag  has  been  overwritten.
*/
void            CantPrint (Obj hd)
{
    Error("Panic: can't print bag of type %d",(Int)GET_TYPE_BAG(hd),0);
}

void            PrintBagType (Obj hd)
{
    Pr("_bag_%d_", GET_TYPE_BAG(hd), 0);
}

/****************************************************************************
**
*F  PrVar( <hdVar> )  . . . . . . . . . . . . . . . . . . .  print a variable
**
**  'PrVar' prints  the variable <hdVar>, or precisly  the identifier of that
**  variable.
*/
void            PrVar (Obj hdVar)
{
    char * name = VAR_NAME(hdVar);
    PrVarName(name);
}

/****************************************************************************
**
*F  PrVarName( <string> )  . . prints identifier, escaping special characters
**
*/
void            PrVarName (char *name)
{
    if ( !strcmp(name,"and")      || !strcmp(name,"do")
      || !strcmp(name,"elif")     || !strcmp(name,"else")
      || !strcmp(name,"end")      || !strcmp(name,"fi")
      || !strcmp(name,"for")      || !strcmp(name,"function")
      || !strcmp(name,"if")       || !strcmp(name,"in")
      || !strcmp(name,"local")    || !strcmp(name,"mod")
      || !strcmp(name,"not")      || !strcmp(name,"od")
      || !strcmp(name,"or")       || !strcmp(name,"repeat")
      || !strcmp(name,"return")   || !strcmp(name,"then")
      || !strcmp(name,"until")    || !strcmp(name,"while")
      || !strcmp(name,"quit") ) {
        Pr("\\",0,0);
    }

    /* print the name                                                      */
    for (   ; *name != '\0'; name++ ) {
        if ( IsAlpha(*name) || IsDigit(*name) || *name == '_' || *name == '@')
            Pr("%c",(Int)(*name),0);
        else
            Pr("\\%c",(Int)(*name),0);
    }
}

/****************************************************************************
**
*F  PrVarAss( <hdAss> ) . . . . . . . . . . . . . . . . . print an assignment
**
**  'PrVarAss' prints an assignment to a variable: '<Var> := <Expr>;'
**
**  Linebreaks are preffered before the ':='.
*/
void            PrVarAss (Obj hdAss)
{
    Pr("%2>",0,0);
    Print(PTR_BAG(hdAss)[0]);
    Pr("%< %>:= ",0,0);
    Print(PTR_BAG(hdAss)[1]);
    Pr("%2<",0,0);
}


/****************************************************************************
**
*V  prPrec  . . . . . . . . . . . . . . . . . . . . current preceedence level
**
**  This variable contains the current preceedence  level,  i.e.  an  integer
**  that indicates the binding  power  of  the  currently  printed  operator.
**  If one of the operands is an operation that has lower binding power it is
**  printed in parenthesis.  If the right operand has the same binding  power
**  it is put in parenthesis,  since all the operations are left associative.
**  Preceedence: 12: ^; 10: mod,/,*; 8: -,+; 6: in,=; 4: not; 2: and,or.
**  This sometimes puts in superflous  parenthesis:  2 * f( (3 + 4) ),  since
**  it doesn't know that a  function  call  adds  automatically  parenthesis.
*/
Int            prPrec;


/****************************************************************************
**
*F  PrNot( <hdNot> )  . . . . . . . . . . . . .  print a boolean not operator
**
**  'PrNot' print a not operation in the following form: 'not <expr>'.
*/
void            PrNot (Obj hdNot)
{
    Int                oldPrec;

    oldPrec = prPrec;  prPrec = 4;
    Pr("not%> ",0,0);  Print( PTR_BAG(hdNot)[0] );  Pr("%<",0,0);
    prPrec = oldPrec;
}


/****************************************************************************
**
*F  PrBinop( <hdOp> ) . . . . . . . . . . . . . . .  prints a binary operator
**
**  This prints any of the binary operator using  prPrec  for parenthesising.
*/
void            PrBinop (Obj hdOp)
{
    Int                oldPrec;
    char                * op;

    oldPrec = prPrec;

    switch ( GET_TYPE_BAG(hdOp) ) {
    case T_AND:    op = "and";  prPrec = 2;   break;
    case T_OR:     op = "or";   prPrec = 2;   break;
    case T_EQ:     op = "=";    prPrec = 6;   break;
    case T_LT:     op = "<";    prPrec = 6;   break;
    case T_GT:     op = ">";    prPrec = 6;   break;
    case T_NE:     op = "<>";   prPrec = 6;   break;
    case T_LE:     op = "<=";   prPrec = 6;   break;
    case T_GE:     op = ">=";   prPrec = 6;   break;
    case T_IN:     op = "in";   prPrec = 6;   break;
	case T_IS:     op = "_is";  prPrec = 6;   break;
    case T_CONCAT: op = "::";   prPrec = 8;   break;
    case T_SUM:    op = "+";    prPrec = 8;   break;
    case T_DIFF:   op = "-";    prPrec = 8;   break;
    case T_PROD:   op = "*";    prPrec = 10;  break;
    case T_QUO:    op = "/";    prPrec = 10;  break;
    case T_MOD:    op = "mod";  prPrec = 10;  break;
    case T_POW:    op = "^";    prPrec = 12;  break;
    default:       op = "<bogus-operator>";   break;
    }

    if ( oldPrec > prPrec )  Pr("%>(%>",0,0);
    else                     Pr("%2>",0,0);
    if ( GET_TYPE_BAG(hdOp) == T_POW
      && ((GET_TYPE_BAG(PTR_BAG(hdOp)[0]) == T_INT && HD_TO_INT(PTR_BAG(hdOp)[0]) < 0)
        || GET_TYPE_BAG(PTR_BAG(hdOp)[0]) == T_INTNEG) )
        Pr("(",0,0);
    Print( PTR_BAG(hdOp)[0] );
    if ( GET_TYPE_BAG(hdOp) == T_POW
      && ((GET_TYPE_BAG(PTR_BAG(hdOp)[0]) == T_INT && HD_TO_INT(PTR_BAG(hdOp)[0]) < 0)
        || GET_TYPE_BAG(PTR_BAG(hdOp)[0]) == T_INTNEG) )
        Pr(")",0,0);
    Pr("%2< %2>%s%> %<",(Int)op,0);
    ++prPrec;
    Print( PTR_BAG(hdOp)[1] );
    --prPrec;
    if ( oldPrec > prPrec )  Pr("%2<)",0,0);
    else                     Pr("%2<",0,0);
    prPrec = oldPrec;
}


/****************************************************************************
**
*F  PrComm( <hdComm> )  . . . . . . . . . . . . . . . . .  print a commutator
**
**  This prints a commutator.
*/
void            PrComm (Obj hd)
{
    Pr("%>Comm(%> ",0,0);
    Print(PTR_BAG(hd)[0]);
    Pr("%<,%>",0,0);
    Print(PTR_BAG(hd)[1]);
    Pr("%2<)",0,0);
}


/****************************************************************************
**
*F  InstEvFunc( <type>, <func> ) . . . . . . .  install a evaluation function
**
**  Installs the function  <func> as evaluation function for bags of  <type>.
*/
void            InstEvFunc (unsigned int type, Bag (*func) (Bag))
{
    EvTab[ type ] = func;
}


/****************************************************************************
**
*F  InstBinOp( <tab>, <typeL>, <typeR>, <func> )  .  install binary operation
**
**  Installs the function  <func>  as  evaluation  function  for  the  binary
**  operation with the table <tab> for operands of type  <typeL> and <typeR>.
*/
void            InstBinOp (Bag  (* table [EV_TAB_SIZE][EV_TAB_SIZE]) (), unsigned int leftType, unsigned int rightType, Obj (*func) (/* ??? */))
{
    table[ leftType ][ rightType ] = func;
}


/****************************************************************************
**
*F  InstPrFunc( <type>, <func> )  . . . . . . . . install a printing function
**
**  Installs the function <func> as printing function  for  bags  of  <type>.
*/
void            InstPrFunc (unsigned int type, void (*func) (Bag))
{
    PrTab[ type ] = func;
}


/****************************************************************************
**
*F  InstVar( <name>, <hdVal> )  . . . . . . . . . . . . . installs a variable
**
**  Installs the value <hdVal> ar value of the new variable with name <name>.
*/
void            InstVar (char *name, Obj hdVal)
{
    Obj           hdVar;

    hdVar = FindIdentWr( name );
    if ( PTR_BAG(hdVar)[0] != 0 )
        Error("Panic: symbol clash %s during initialization",(Int)name,0);
    SET_BAG(hdVar, 0,  hdVal );
}

#define MAX_INT_FUNCS   768

/* NOTE: IntFuncs would really be better as a search tree */

typedef struct {
    char*       name;
    PtrIntFunc  func;
} IntFuncs_t;

static IntFuncs_t   IntFuncs[MAX_INT_FUNCS];
static UInt         IntFuncsCount = 0;

/****************************************************************************
**
*F  FindIntFunc( <name> ) . . . . searching for install an internal function
**
**  Searching for installed internal function with the  name  <name>.
**  Returns pointer to function if found, 0 otherwise.
*/

PtrIntFunc     FindIntFunc(char* name)
{
    UInt i;
    for (i=0; i<IntFuncsCount; i++) {
        if (strcmp(IntFuncs[i].name, name)==0) {
            return IntFuncs[i].func;
        }
    }
    return 0;
}

/****************************************************************************
**
*F  InstIntFunc( <name>, <func> ) . . . . . . .  install an internal function
**
**  Installs the function <func> as internal function with the  name  <name>.
*/
Bag            InstIntFunc (char *name, PtrIntFunc func)
{
    Obj           hdDef,  hdVar;

    hdDef = NewBag( T_FUNCINT, sizeof(PtrIntFunc) + strlen(name) + 1 );
    * (PtrIntFunc*) PTR_BAG(hdDef) = func;

    if (FindIntFunc(name) != 0)
        Error("Panic: symbol clash %s during initialization",(Int)name,0);

    hdVar = FindIdentWr( name );
    if ( PTR_BAG(hdVar)[0] != 0 )
        Error("Panic: symbol clash %s during initialization",(Int)name,0);
    SET_BAG(hdVar, 0,  hdDef );

    if (IntFuncsCount>=MAX_INT_FUNCS) {
        Error("Panic: no more room for internal functions. Increase MAX_INT_FUNCS in eval.c to get more space.",0,0);
    } else {
        IntFuncs[IntFuncsCount].name = name;
        IntFuncs[IntFuncsCount].func = func;
        IntFuncsCount++;
    }

    ProtectVar(hdVar);
    /* this enables us to print internal functions */
    strncat( (char*)PTR_BAG(hdDef) + sizeof(PtrIntFunc), name, strlen(name));
    return hdDef;
}


/****************************************************************************
**
*F  FunCantCopy( <hdCall> ) . . . . . . . . . .  internal function 'CantCopy'
**
**  CantCopy(<obj>) marks object with BF_NO_COPY, so that the object is never
**  copied with Copy(). (Although ShallowCopy may still be used)
*/
Obj       FunCantCopy (Obj hdCall)
{
    Obj           hd = 0;
    /* evaluate and check the argument                                     */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: CantCopy( <obj> )",0,0);
    hd = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hd) != T_INT )
        SET_FLAG_BAG(hd, BF_NO_COPY);
    return hd;
}

/****************************************************************************
**
*F  FunMayCopy( <hdCall> ) . . . . . . . . . .  internal function 'MayCopy'
**
**  MayCopy(<obj>) removes BF_NO_COPY flag, so that the object can be copied
**  again with Copy(),
*/
Obj       FunMayCopy (Obj hdCall)
{
    Obj           hd = 0;
    /* evaluate and check the argument                                     */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: MayCopy( <obj> )",0,0);
    hd = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hd) != T_INT )
        CLEAR_FLAG_BAG(hd, BF_NO_COPY);
    return hd;
}


Obj EvMakeLet( Obj hd ) {
    UInt size = TableNumEnt(hd) - 1, lenres = 0;
    UInt i, evaluated = 0;
    Obj bindings = NewList((int)(size+1));
    Obj res = 0;
    exc_type_t e = 0;
    SET_BAG(bindings, 1,  hd );
    EVAL_STACK_PUSH(bindings);
    Try {
        /* evaluate all variables and set their values */
        for ( i = 0; i < size; ++i ) {
            Obj var = PTR_BAG(hd)[i];
            Obj uneval = PTR_BAG(var)[1];
            Obj oldbinding = VAR_VALUE(var);
            Obj eval;
            SET_BAG(bindings, i+2,  oldbinding );
            eval = EVAL(uneval);
            SET_VAR_VALUE(var, eval);
            evaluated++;
        }
        /* evaluate all statements, return result of last */
        res = PTR_BAG(PTR_BAG(hd)[size])[1];
        lenres = LEN_PLIST(res);
        for(i = 1; i < lenres; ++i)
            EVAL(PTR_BAG(res)[i]);
        if ( lenres == 0 ) res = HdVoid;
        else res = EVAL(PTR_BAG(res)[lenres]);
    } Catch(e) {
        // restore what we already did
        for ( i = 0; i < evaluated; ++i ) {
            Obj var = PTR_BAG(hd)[i];
            Obj oldbinding = PTR_BAG(bindings)[i+2];
            SET_VAR_VALUE(var, oldbinding);
        }
        Throw(e);
    }
    /* restore original variable bindings */
    for ( i = 0; i < size; ++i ) {
        Obj var = PTR_BAG(hd)[i];
        Obj oldbinding = PTR_BAG(bindings)[i+2];
        SET_VAR_VALUE(var, oldbinding);
    }
    EVAL_STACK_POP;
    return res;
}

void PrMakeLet( Obj hd ) {
    UInt size = TableNumEnt(hd) - 1, lenres;
    UInt i;
    Obj res;
    Pr("let(%2>", 0, 0);
    /* evaluate all variables and set their values */
    for ( i = 0; i < size; ++i ) {
        Obj var = PTR_BAG(hd)[i];
        Obj uneval = PTR_BAG(var)[1];
        Pr("%g := %2>%g%2<,\n", (Int)var, (Int)uneval);
    }

    res = PTR_BAG(PTR_BAG(hd)[size])[1];
    lenres = LEN_PLIST(res);
    /* expressions separated by a comma */
    for(i = 1; i <= lenres-1; ++i) {
        Pr("%2>%g%2<, ", (Int) PTR_BAG(res)[i], 0);
    }
    /* and the final expression without trailing comma */
    if (lenres >= 1) {
        Pr("%2>%g%2< ", (Int) PTR_BAG(res)[lenres], 0);
    }
    Pr("%2<)\n", 0, 0);
}


/****************************************************************************
**
*F  FunInherited( <hdCall> ) . . . . . . . . . .  internal function 'CantCopy'
**
**  CantCopy(<obj>) marks object with BF_NO_COPY, so that the object is never
**  copied with Copy(). (Although ShallowCopy may still be used)
*/
Obj       FunInherited (Obj hdCall)
{
    Obj     hd = 0;
    Obj     hdRecElm = 0;
    Obj     hdSelf = 0;
    Obj     hdMeth = 0;
    Obj*    ptRec = 0;
    /* find method on the call stack and count how many times Inherited() was called */
    Int     top = EvalStackTop;
    Int     inherited = 0, i = 0;
    while (top>0) {
        if (GET_TYPE_BAG(EvalStack[top]) == T_EXEC && GET_TYPE_BAG(PTR_BAG(EvalStack[top])[2]) == T_METHOD) {
            if (GET_FLAG_BAG(PTR_BAG(EvalStack[top])[3], BF_INHERITED_CALL)) {
                /* found call to Inherited() */
                inherited++;
            } else { /* found T_EXEC with T_METHOD */
                /* get self pointer from hd call and method defenition */
                hdRecElm = PTR_BAG(PTR_BAG(EvalStack[top])[3])[0]; /* first argument of hdCall is T_RECELM */
                break;
            }
        }
        top--;
    }
    if (top<=0)
        return Error("Inherited: method not found", 0, 0);
    if (GET_TYPE_BAG(hdRecElm) != T_RECELM)
        return Error("Inherited: ambiguity, method called in non-standard way.", 0, 0);

    /* find next inherited method */
    inherited++;

    hdSelf = PTR_BAG(hdRecElm)[0];
    if (GET_TYPE_BAG(hdSelf) != T_REC)
        return Error("Inherited: hd is not a record", (Int)hdSelf, 0);
    hd = PTR_BAG(hdRecElm)[1];
    if (GET_TYPE_BAG(hd) != T_RECNAM)
        return Error("Inherited: %g is not a field name", (Int)hd, 0);

    ptRec = FindRecnameRecWithDepth(hdSelf, hd, &hd, &inherited);
    if (ptRec==0)
        return HdVoid;

    hdMeth = EVAL(ptRec[1]);
    if (GET_TYPE_BAG(hdMeth) != T_METHOD)
        return Error("Inherited: %g is not a method", (Int)hdMeth, 0);

    hd = NewBag(T_FUNCCALL, GET_SIZE_BAG(hdCall) + SIZE_HD);
    SET_BAG(hd, 0, hdMeth);
    SET_BAG(hd, 1, hdSelf);
    for( i=1; i < GET_SIZE_BAG(hdCall)/SIZE_HD; ++i) {
        SET_BAG(hd, i+1, PTR_BAG(hdCall)[i]);
    }
    SET_FLAG_BAG(hd, BF_METHCALL);
    SET_FLAG_BAG(hd, BF_INHERITED_CALL);
    return EVAL(hd);
}

/****************************************************************************
**
*F  InitEval  . . . . . . . . . . . . . initialize the evaluator main package
**
**  This is called relative lately during the initialization from  InitGap().
*/
void            InitEval (void)
{
    unsigned int        type,  typeL,  typeR;

    /* clear the tables for the evaluation dispatching                     */
    for ( type = 0; type < T_ILLEGAL; ++type ) {
        EvTab[type] = NoEval;
        PrTab[type] = PrintBagType;
    }

    for ( typeL = 0; typeL < EV_TAB_SIZE; ++typeL ) {
        for ( typeR = 0; typeR < EV_TAB_SIZE; ++typeR ) {
            TabSum[typeL][typeR]  = CantSum;
            TabDiff[typeL][typeR] = CantDiff;
            TabProd[typeL][typeR] = CantProd;
            TabQuo[typeL][typeR]  = CantQuo;
            TabMod[typeL][typeR]  = CantMod;
            TabPow[typeL][typeR]  = CantPow;
            TabComm[typeL][typeR] = CantComm;
        }
    }
    for ( typeL = 0; typeL < EV_TAB_SIZE; ++typeL ) {
        for ( typeR = 0; typeR <= typeL; ++typeR ) {
            TabEq[typeL][typeR] = EqPtr;
            TabLt[typeL][typeR] = LtPtr;
        }
        for ( typeR = typeL+1; typeR < EV_TAB_SIZE; ++typeR ) {
            TabEq[typeL][typeR] = EqPtr;
            TabLt[typeL][typeR] = LtPtr;
        }
    }

    /* install the evaluators main evaluation functions                    */
    InstEvFunc( T_CONCAT,   Concat   );
    InstEvFunc( T_SUM,      Sum      );
    InstEvFunc( T_DIFF,     Diff     );
    InstEvFunc( T_PROD,     Prod     );
    InstEvFunc( T_QUO,      Quo      );
    InstEvFunc( T_MOD,      Mod      );
    InstEvFunc( T_POW,      Pow      );
    /*N hack to replace commutator operator until I fix the parser         */
    /*N InstEvFunc( T_COMM,     Comm     );                                */
    PGAP("gap");
    InstIntFunc( "Comm",  IntComm  );
    InstIntFunc( "LeftQuotient",  FunLeftQuotient );
    PEND("gap");
    InstEvFunc( T_EQ,       Eq       );
    InstEvFunc( T_LT,       Lt       );
    InstEvFunc( T_LE,       Le       );
    InstEvFunc( T_NE,       Ne       );
    InstEvFunc( T_GT,       Gt       );
    InstEvFunc( T_GE,       Ge       );

    /* install the main printing functions.                                */
    InstPrFunc( T_SUM,      PrBinop    );
    InstPrFunc( T_DIFF,     PrBinop    );
    InstPrFunc( T_PROD,     PrBinop    );
    InstPrFunc( T_QUO,      PrBinop    );
    InstPrFunc( T_MOD,      PrBinop    );
    InstPrFunc( T_POW,      PrBinop    );
    InstPrFunc( T_COMM,     PrComm     );
    InstPrFunc( T_EQ,       PrBinop    );
    InstPrFunc( T_LT,       PrBinop    );
    InstPrFunc( T_GT,       PrBinop    );
    InstPrFunc( T_NE,       PrBinop    );
    InstPrFunc( T_LE,       PrBinop    );
    InstPrFunc( T_GE,       PrBinop    );
    InstPrFunc( T_IN,       PrBinop    );

    /* variables and assignments                                           */
    InstEvFunc( T_VAR,      EvVar      );
    InstEvFunc( T_VARAUTO,  EvVarAuto  );
    InstEvFunc( T_VARASS,   EvVarAss   );
    InstPrFunc( T_VAR,      PrVar      );
    InstPrFunc( T_VARAUTO,  PrVar      );
    InstPrFunc( T_VARASS,   PrVarAss   );
    InstPrFunc( T_MULTIASS, PrVarAss   );

    InstEvFunc( T_MAKELET,   EvMakeLet   );
    InstPrFunc( T_MAKELET,   PrMakeLet   );

    /* void bag                                                            */
    HdVoid  = NewBag( T_VOID, NUM_TO_UINT(0) );

    /* boolean operations                                                  */
    PGAP("bool");
    HdTrue  = NewBag(T_BOOL,NUM_TO_UINT(0));  InstVar( "true",  HdTrue  );
    HdFalse = NewBag(T_BOOL,NUM_TO_UINT(0));  InstVar( "false", HdFalse );
    InstEvFunc( T_BOOL,     EvBool     );
    InstEvFunc( T_NOT,      EvNot      );
    InstEvFunc( T_AND,      EvAnd      );
    InstEvFunc( T_OR,       EvOr       );
	InstEvFunc( T_IS,       EvIs       );
    InstPrFunc( T_BOOL,     PrBool     );
    InstPrFunc( T_NOT,      PrNot      );
    InstPrFunc( T_AND,      PrBinop    );
    InstPrFunc( T_OR,       PrBinop    );
    InstPrFunc( T_CONCAT,   PrBinop    );
	InstPrFunc( T_IS,       PrBinop    );

    TabEq[ T_BOOL ][ T_BOOL ] = EqBool;
    TabLt[ T_BOOL ][ T_BOOL ] = LtBool;
    InstIntFunc( "IsBool",      FunIsBool      );
    PEND();

    /* install main evaluator internal functions.                          */
    PGAP("eval");
    InstIntFunc( "CantCopy",    FunCantCopy );
    InstIntFunc( "MayCopy",     FunMayCopy );
    InstIntFunc( "ShallowCopy", FunShallowCopy );
    InstIntFunc( "Copy",        FunCopy        );
    InstIntFunc( "CopyFunc",    FunCopyFunc    );
    InstIntFunc( "IsBound",     FunIsBound     );
    InstIntFunc( "Unbind",      FunUnbind      );
    InstIntFunc( "ProtectVar",  FunProtectVar      );
    InstIntFunc( "ProtectRec",  FunProtectRec      );
    InstIntFunc( "ProtectNamespace",  FunProtectNamespace      );
    InstIntFunc( "_UnprotectVar",  Fun_UnprotectVar      );
    InstIntFunc( "Inherited",       FunInherited);
    PEND();

    /* install the printing tilde                                          */
    HdTildePr = FindIdent( "~~" );

    /* initialize the evaluators subpackages                               */
    PGAP("int");     InitInt(); PEND();      /* init integer package            */
    PGAP("rat");     InitRat(); PEND();      /* init rational package           */
    PGAP("cyc");     InitCyc(); PEND();      /* init cyclotomic integer package */
    PGAP("unknown"); InitUnknown(); PEND();  /* init unknown package            */
    PGAP("ff");      InitFF(); PEND();       /* init finite field package       */
    PGAP("polynom"); InitPolynom(); PEND();  /* init polynomial package         */
    PGAP("perm");    InitPermutat(); PEND(); /* init permutation package        */
    PGAP("word");    InitWord(); PEND();     /* init word package               */
    PGAP("costab");  InitCosTab(); PEND();   /* init coset table package        */
    PGAP("tietze");  InitTietze(); PEND();   /* init tietze package             */
    PGAP("ag");      InitAg(); PEND();       /* init soluable group package     */
    PGAP("pcpres");  InitPcPres(); PEND();   /* init polycyclic pres            */
    PGAP("list");    InitList(); PEND();     /* init list package               */
                     InitPlist();            /* init plain list package         */
    PGAP("set");     InitSet(); PEND();      /* init set package                */
    PGAP("vector");  InitVector(); PEND();   /* init vector package             */
    PGAP("vecffe");  InitVecFFE(); PEND();   /* init finite fld vector package  */
    PGAP("blist");   InitBlist(); PEND();    /* init boolean list package       */
    PGAP("range");   InitRange(); PEND();    /* init range package              */
    PGAP("string");  InitString(); PEND();   /* init string package             */
    PGAP("rec");     InitRec(); PEND();      /* init record package             */
                     InitStat();             /* init statment package           */
    PGAP("func");    InitFunc(); PEND();     /* init function package           */
    PGAP("coding");  InitCoding(); PEND();   /* init coding package             */

    /* initialization of further evaluation packages goes here !!!         */

    EvalStackTop = 0;
}


/****************************************************************************
**
*E  Emacs . . . . . . . . . . . . . . . . . . . . . . . local emacs variables
**
**  Local Variables:
**  outline-regexp:     "*A\\|*F\\|*V\\|*T\\|*E"
**  fill-column:        73
**  fill-prefix:        "**  "
**  c-basic-offset:     4
**  End:
*/



