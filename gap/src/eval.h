/****************************************************************************
**
*A  eval.h                      GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2020, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2020) by the GAP Group (www.gap-system.org).
**
**  This file declares the main evaluation functions.
**
*/
#include <assert.h>

#define EV_TAB_SIZE T_NAMESPACE+1

#define CHK(hdL, hdR, eval_tab_access) \
                       (assert(GET_TYPE_BAG(hdL) < EV_TAB_SIZE && \
		    	       GET_TYPE_BAG(hdR) < EV_TAB_SIZE), \
                	(eval_tab_access))

/*N where should I place the declaration of 'Error' ?                      */
extern  Bag       Error ( char * msg, Int arg1, Int arg2 );


/****************************************************************************
**
*V  HdVoid  . . . . . . . . . . . . . . . . . . . . .  handle of the void bag
**
**  'HdVoid' is the handle of the void back, which is returned by procedures,
**  i.e., functions that when viewed at the  GAP  level do not return a value
**  at all.  This plays a role similar to  '*the-non-printing-object*'  which
**  exists in some lisp systems.
*/
extern  Bag       HdVoid;


/****************************************************************************
**
*V  HdReturn  . . . . . . . . . . . . . . . . . . .  handle of the return bag
**
**  'HdReturn' is the handle of the bag where 'EvReturn' puts the value of a
**  'return' statement.  This bag is then passed through all  the  statement
**  execution functions all the  way  back  to  'EvFunccall'.  For  'return'
**  statements without an expression 'EvReturn' puts 'HdVoid' into this bag.
*/
extern  Bag       HdReturn;


/****************************************************************************
**
*V  HdTrue  . . . . . . . . . . . . . . . . . . . . .  handle of the true bag
*V  HdFalse   . . . . . . . . . . . . . . . . . . . . handle of the false bag
**
**  'HdTrue' is the handle of the unique bag that represents the value 'true'
**  and 'HdFalse' is likewise the unique handle of the  bag  that  represents
**  the value 'HdFalse'.
*/
extern  Bag       HdTrue,  HdFalse;

/****************************************************************************
**
**  HdEvalStack  . . . . . . . . . . . . . . . . . . . . . . .  EVAL stack
**  EvalStackTop . . . . . . . . . . . . . . . . . . . . . . .  stack pointer
**
**  HdEvalStack contains full EVAL stack. Note that this stack is not
**  accounted during garbage collection cycle. 
**  EvalStackTop - index of the last plased element.
*/

#define EVAL_STACK_COUNT	65536
/* EVAL_STACK_CUSHION - just some extra space. Stack overflow check is in the 
** EvFunccall only and only EVAL_STACK_COUNT checked. So to make gap safer 
** buffer has some extra space
*/
#define EVAL_STACK_CUSHION	1024
extern Bag         EvalStack[EVAL_STACK_COUNT+EVAL_STACK_CUSHION];
extern UInt        EvalStackTop;

#define EVAL_STACK_PUSH(hd)     EvalStack[++EvalStackTop] = hd
#define EVAL_STACK_POP          EvalStackTop--

/****************************************************************************
**
*V  EvTab[<type>] . . . . . . . . evaluation function for bags of type <type>
**
**  Is the main dispatching table that contains for every type a  pointer  to
**  the function that should be executed if a bag  of  that  type  is  found.
*/
extern  Bag       (* EvTab[ T_ILLEGAL ]) ( Bag hd );


/****************************************************************************
**
*F  EVAL( <hd> )  . . . . . . . . . . . . . . . . . . . .  evaluate an object
**
**  'EVAL' evaluates the bag <hd>  by  calling  the  corresponding  function.
**
**  It is defined in the definition file of this package as followings:
*/


static inline Bag    EVAL(Bag hd) {
    if ((Int)(hd)&T_INT)
        return hd;
    else {        
        Bag   Result = 0;
        EVAL_STACK_PUSH(hd);
#ifdef _DEBUG
        ObjType bagtype = GET_TYPE_BAG(hd);
        Obj(*func) (Obj hd);
        func = EvTab[bagtype];
        Result = func(hd);
#else
		Result = (* EvTab[GET_TYPE_BAG( hd )])(hd);
#endif
        EVAL_STACK_POP;
        return Result;
    }
}

/****************************************************************************
**
*F  CantEval( <hd> )  . . . . . . . . . . . . illegal bag evaluation function
**
**  Is called if a illegal bag should be evaluated, it  generates  an  error.
**  If this is actually ever executed in GAP it  indicates  serious  trouble,
**  for  example  that  the  type  field  of  a  bag  has  been  overwritten.
*/
extern  Bag       CantEval ( Bag hd );


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
*/
#define SUM(hdL,hdR)    CHK(hdL, hdR, (*TabSum[GET_TYPE_BAG(hdL)][GET_TYPE_BAG(hdR)])((hdL),(hdR)))

extern  Bag       (*TabSum[EV_TAB_SIZE][EV_TAB_SIZE]) ( Bag, Bag );

extern  Bag       Sum ( Bag hd );

extern  Bag       CantSum ( Bag, Bag );


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
*/
#define DIFF(hdL,hdR)   CHK(hdL, hdR, (*TabDiff[GET_TYPE_BAG(hdL)][GET_TYPE_BAG(hdR)])((hdL),(hdR)))

extern  Bag       (*TabDiff[EV_TAB_SIZE][EV_TAB_SIZE]) ( Bag, Bag );

extern  Bag       Diff ( Bag hd );

extern  Bag       CantDiff ( Bag, Bag );


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
*/
#define PROD(hdL,hdR)   CHK(hdL, hdR, (*TabProd[GET_TYPE_BAG(hdL)][GET_TYPE_BAG(hdR)])((hdL),(hdR)))

extern  Bag       (*TabProd[EV_TAB_SIZE][EV_TAB_SIZE]) ( Bag, Bag );

extern  Bag       Prod ( Bag hd );

extern  Bag       CantProd ( Bag, Bag );


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
**  At places where performance really matters one should  copy  the  special
**  code  from  'Quo' which checks for the division of two immediate integers
**  and computes their quotient without calling 'QUO'.
**
**  'QUO' is defined in the header file of this package as follows:
*/
#define QUO(hdL,hdR)    CHK(hdL, hdR, (*TabQuo[GET_TYPE_BAG(hdL)][GET_TYPE_BAG(hdR)])((hdL),(hdR)))

extern  Bag       (*TabQuo[EV_TAB_SIZE][EV_TAB_SIZE]) ( Bag, Bag );

extern  Bag       Quo ( Bag hd );

extern  Bag       CantQuo ( Bag, Bag );


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
**  At places where performance really matters one should  copy  the  special
**  code from 'Mod' which checks for the division  of  two immediate integers
**  and computes their remainder without calling 'MOD'.
**
**  'MOD' is defined in the header file of this package as follows:
*/
#define MOD(hdL,hdR)    CHK(hdL, hdR, (*TabMod[GET_TYPE_BAG(hdL)][GET_TYPE_BAG(hdR)])((hdL),(hdR)))

extern  Bag       (*TabMod[EV_TAB_SIZE][EV_TAB_SIZE]) ( Bag, Bag );

extern  Bag       Mod ( Bag hd );

extern  Bag       CantMod ( Bag, Bag );


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
*/
#define POW(hdL,hdR)    CHK(hdL, hdR, (*TabPow[GET_TYPE_BAG(hdL)][GET_TYPE_BAG(hdR)])((hdL),(hdR)))

extern  Bag       (*TabPow[EV_TAB_SIZE][EV_TAB_SIZE]) ( Bag, Bag );

extern  Bag       Pow ( Bag hd );

extern  Bag       CantPow ( Bag, Bag );


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
extern  Bag       (*TabComm[EV_TAB_SIZE][EV_TAB_SIZE]) ( Bag, Bag );

extern  Bag       IntComm ( Bag hdCall );


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
*/
#define EQ(hdL,hdR)     CHK(hdL, hdR, (*TabEq[GET_TYPE_BAG(hdL)][GET_TYPE_BAG(hdR)])((hdL),(hdR)))

extern  Bag       (*TabEq[EV_TAB_SIZE][EV_TAB_SIZE]) ( Bag, Bag );

extern  Bag       Eq ( Bag hd );


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
*/
#define LT(hdL,hdR)     CHK(hdL, hdR, (*TabLt[GET_TYPE_BAG(hdL)][GET_TYPE_BAG(hdR)])((hdL),(hdR)))

extern  Bag       (*TabLt[EV_TAB_SIZE][EV_TAB_SIZE]) ( Bag, Bag );

extern  Bag       Lt ( Bag hd );


/****************************************************************************
**
*F  Ne( <hdNe> )  . . . . . . . . . . . . . . . . .  test if <objL> <> <objR>
**
**  'Ne'  return 'HdTrue' if  the object <objL> is not equal  to  the  object
**  <objR>.  'Ne' is called from 'EVAL' to evaluate bags of type 'T_NE'.
**
**  'Ne' is simply implemented as 'not <objL> = <objR>'.
*/
extern  Bag       Ne ( Bag hd );


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
extern  Bag       Le ( Bag hd );


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
extern  Bag       Gt ( Bag hd );


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
extern  Bag       Ge ( Bag hd );


/****************************************************************************
**
*F  IsTrue( <hdL>, <hdR> )  . . . . . . . .  default function for comparisons
**
**  'IsTrue' always returns  'HdTrue'  no  matter  what  the  arguments  are.
**  Is is used for those comparisons where already the types of the  operands
**  determines the outcome.  E.g., it is  used  above  the  diagonal  of  the
**  'TabLt' table.
*/
extern  Bag       IsTrue ( Bag hdL, Bag hdR );


/****************************************************************************
**
*F  IsFalse( <hdL>, <hdR> ) . . . . . . . .  default function for comparisons
**
**  'IsFalse' always returns 'HdFalse' no  matter  what  the  arguments  are.
**  Is is used for those comparisons where already the types of the  operands
**  determines the outcome.  E.g., it is  used  below  the  diagonal  of  the
**  'TabLt' table.
*/
extern  Bag       IsFalse ( Bag hdL, Bag hdR );


/****************************************************************************
**
*F  EvVar( <hdVar> )  . . . . . . . . . . . . . . . . . . evaluate a variable
**
**  'EvVar' returns the value  of  the  variable  with  the  handle  <hdVar>.
**
**  The value is the only subobject <hdVar>.  If this has the handle '0' then
**  no value has been assigned to the variable and  an  error  is  generated.
*/
extern  Bag       EvVar ( Bag hdVar );


/****************************************************************************
**
*F  EvVarAuto( <hdVar> )  . . . . . . . . . . . . . eval an autoread variable
*/
extern  Bag       EvVarAuto ( Bag hdVar );


/****************************************************************************
**
*F  EvVarAss( <hdAss> ) . . . . . . . . . . . . . . . . execute an assignment
**
**  'EvVarAss' assigns the value of '<hdAss>[1]' to the variable '<hdAss>[0]'
**  and returns the value so that it can be printed in the ReadEvalPrintLoop.
**
**  'EvVarAss' is called from 'EVAL' for bags of type 'T_VARASS'.
*/
extern  Bag       EvVarAss ( Bag hdAss );


/****************************************************************************
**
*F  EvBool( <hdBool> )  . . . . . . . . . . . . . .  evaluate a boolean value
**
**  'EvBool' returns the value of the boolean value <hdBool>.  Since  boolean
**  values are constants and thus selfevaluating it just returns <hdBool>.
*/
extern  Bag       EvBool ( Bag hdBool );


/****************************************************************************
**
*F  EvNot( <hdBool> ) . . . . . . . . . . . . . . . .  negate a boolean value
**
**  'EvNot' returns the boolean negation of the boolean value <hdBool>, i.e.,
**  it returns 'HdTrue' if <hdBool> is 'HdFalse' and vica versa.
*/
extern  Bag       EvNot ( Bag hdBool );


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
extern  Bag       EvAnd ( Bag hd );

/****************************************************************************
**
*F  EvIs( <hdIs> )  . . . . . . . . . . .  evaluate a boolean 'is' operation
**
**  'EvIs' evaliates boolean expression '_ObjId(<hdAnd>[0])=<hdAnd>[1]'
**
*/
extern  Bag       EvIs ( Bag hd );

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
extern  Bag       EvOr ( Bag hd );


/****************************************************************************
**
*F  EqBool( <hdL>, <hdR> )  . . . . . . . . . . .  test if <boolL> =  <boolR>
**
**  'EqBool' returns 'HdTrue' if the  two  boolean  values  <hdL>  and  <hdR>
**  are equal, and 'HdFalse' otherwise.
*/
extern  Bag       EqBool ( Bag hdL, Bag hdR );


/****************************************************************************
**
*F  LtBool( <hdL>, <hdR> )  . . . . . . . . . . .  test if <boolL> <  <boolR>
**
**  'LtBool' return 'HdTrue' if  the  boolean value <hdL> is  less  than  the
**  boolean value <hdR> and 'HdFalse' otherwise.
*/
extern  Bag       LtBool ( Bag hdL, Bag hdR );


/****************************************************************************
**
*F  PrBool( <hdBool> )  . . . . . . . . . . . . . . . . print a boolean value
**
**  'PrBool' prints the boolean value <hdBool>.
*/
extern  void            PrBool ( Bag hd );


/****************************************************************************
**
*F  FunIsBool( <hdCall> ) . . . . . . . . . internal function IsBool( <obj> )
**
**  'IsBool' returns 'true' if the object <obj>  is  a  boolean  and  'false'
**  otherwise.  May cause an error if <obj> is an unbound variable.
*/
extern  Bag       FunIsBool ( Bag hdCall );


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
extern  Bag       ShallowCopy ( Bag hdOld );
extern  Bag       FullShallowCopy ( Bag hdOld );
  

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
extern  Bag       FunShallowCopy ( Bag hdCall );


/****************************************************************************
**
*F  Copy( <hdObj> ) . . . . . . . . . . . . . make a normal copy of an object
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
**  with T_VARAUTO < type < T_RECNAM, while FullCopy does.
*/
extern  Bag       Copy ( Bag hdOld );
extern  Bag       FullCopy ( Bag hdOld );


/****************************************************************************
**
*F  FunCopy( <hdCall> ) . . . . . . . . . . . . . .  make a copy of an object
**
**  'FunCopy' implements the internal function 'Copy( <obj> )'.
**
**  'Copy' makes a copy of the  object <hdObj>.  If <obj>  is not a list or a
**  record, 'Copy' simply  returns  <obj>, since those  objects can  never be
**  modified there  is no way  to  distinguish the  original object  from any
**  copy, so we might as  well not copy  it.  If <obj>  is a list or a record
**  'Copy' makes a copy of this object,  and calls itself recursively to copy
**  the subobjects.
*/
extern  Bag       FunCopy ( Bag hdCall );


/****************************************************************************
**
*F  FunIsBound( <hdCall> )  . . . .  test if a variable has an assigned value
**
**  'FunIsBound' implements the internal function 'IsBound( <expr> )'.
**
*/
extern  Bag       FunIsBound ( Bag hdCall );


/****************************************************************************
**
*V  PrTab[<type>] . . . . . . .  printing function for objects of type <type>
**
**  is the main dispatching table that contains for every type a  pointer  to
**  the function that should be executed if a bag  of  that  type  is  found.
*/
extern  void            (* PrTab[ T_ILLEGAL ] ) ( Bag hd );


/****************************************************************************
**
*F  Print( <hd> ) . . . . . . . . . . . . . . . . . . . . . . print an object
**
**  'Print'  prints  the  object  with  handle  <hd>.  It dispatches   to the
**  appropriate function stored in 'PrTab[GET_TYPE_BAG(<hd>)]'.
*/
extern  void            Print ( Bag hd );


/****************************************************************************
**
*F  CantPrint( <hd> ) . . . . . . . . . . . . . illegal bag printing function
**
**  Is called if a illegal bag should be  printed,  it  generates  an  error.
**  If this is actually ever executed in GAP it  indicates  serious  trouble,
**  for  example  that  the  type  field  of  a  bag  has  been  overwritten.
*/
extern  void            CantPrint ( Bag hd );


/****************************************************************************
**
*F  PrVarName( <string> )  . . prints identifier, escaping special characters
**
*/
extern  void            PrVarName ( char * name );
  
/****************************************************************************
**
*F  PrVar( <hdVar> )  . . . . . . . . . . . . . . . . . . .  print a variable
**
**  'PrVar' prints  the variable <hdVar>, or precisly  the identifier of that
**  variable.
*/
extern  void            PrVar ( Bag hdVar );


/****************************************************************************
**
*F  PrVarAss( <hdAss> ) . . . . . . . . . . . . . . . . . print an assignment
**
**  'PrVarAss' prints an assignment to a variable: '<Var> := <Expr>;'
**
**  Linebreaks are preffered before the ':='.
*/
extern  void            PrVarAss ( Bag hdAss );


/****************************************************************************
**
*F  PrNot( <hdNot> )  . . . . . . . . . . . . .  print a boolean not operator
**
**  'PrNot' print a not operation in the following form: 'not <expr>'.
*/
extern  void            PrNot ( Bag hdNot );


/****************************************************************************
**
*F  PrBinop( <hdOp> ) . . . . . . . . . . . . . . .  prints a binary operator
**
**  This prints any of the binary operator using  prPrec  for parenthesising.
*/
extern  void            PrBinop ( Bag hdOp );


/****************************************************************************
**
*F  PrComm( <hdComm> )  . . . . . . . . . . . . . . . . .  print a commutator
**
**  This prints a commutator.
*/
extern  void            PrComm ( Bag hd );


/****************************************************************************
**
*F  InstEvFunc( <type>, <func> ) . . . . . . .  install a evaluation function
**
**  Installs the function  <func> as evaluation function for bags of  <type>.
*/
extern  void            InstEvFunc ( unsigned int     type,
                                     Bag (* func) (Bag) );


/****************************************************************************
**
*F  InstBinOp( <tab>, <typeL>, <typeR>, <func> )  .  install binary operation
**
**  Installs the function  <func>  as  evaluation  function  for  the  binary
**  operation with the table <tab> for operands of type  <typeL> and <typeR>.
*/
extern  void            InstBinOp ( Bag  (* table [EV_TAB_SIZE][EV_TAB_SIZE]) (),
                                      unsigned int      leftType,
                                      unsigned int      rightType,
                                      Bag         (* func) () );


/****************************************************************************
**
*F  InstPrFunc( <type>, <func> )  . . . . . . . . install a printing function
**
**  Installs the function <func> as printing function  for  bags  of  <type>.
*/
extern  void            InstPrFunc ( unsigned int type, void (* func)(Bag) );


/****************************************************************************
**
*F  InstVar( <name>, <hdVal> )  . . . . . . . . . . . . . installs a variable
**
**  Installs the value <hdVal> ar value of the new variable with name <name>.
*/
extern  void            InstVar ( char * name, Bag hdVal );

/* Pointer to a GAP internal function                                      */
typedef Bag (*PtrIntFunc) (Bag);

/****************************************************************************
**
*F  FindIntFunc( <name> ) . . . . searching for install an internal function
**
**  Searching for installed internal function with the  name  <name>.
**  Returns pointer to function if found, 0 otherwise.
*/

extern PtrIntFunc  FindIntFunc(char* name);

/****************************************************************************
**
*F  InstIntFunc( <name>, <func> ) . . . . . . .  install an internal function
**
**  Installs the function <func> as internal function with the  name  <name>.
*/
extern  Bag            InstIntFunc ( char* name, PtrIntFunc func );


/****************************************************************************
**
*F  InitEval  . . . . . . . . . . . . . initialize the evaluator main package
**
**  This is called relative lately during the initialization from  InitGap().
*/
extern  void            InitEval ( void );



