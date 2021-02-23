/****************************************************************************
**
*A  unknown.h                   GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This  file  defines  the  arithmetic  for unknown  values,  unknowns  for
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


/****************************************************************************
**
*F  EvUnknown( <hdUnd> )  . . . . . . . . . . . . . . . . evaluate an unknown
**
**  'EvUnknown' returns the value of the unknown <hdUnd>.  Since unknowns are
**  constants and thus selfevaluating this simply returns <hdUnd>.
*/
extern  Bag       EvUnknown ( Bag hdUnk );


/****************************************************************************
**
*F  SumUnknown( <hdL>, <hdR> )  . . . . . . . . . . . . . sum of two unknowns
**
**  'SumUnknown' returns  the  sum  of  the  two  unknowns <hdL>  and  <hdR>.
**  Either operand may also be a known scalar value.
**
**  Is called from the 'Sum' binop, so both operands are already evaluated.
*/
extern  Bag       SumUnknown ( Bag hdL, Bag hdR );


/****************************************************************************
**
*F  DiffUnknown( <hdL>, <hdR> ) . . . . . . . . .  difference of two unknowns
**
**  'DiffUnknown' returns the difference of the two unknowns <hdL> and <hdR>.
**  Either operand may also be a known scalar value.
**
**  Is called from the 'Diff' binop, so both operands are already evaluated.
*/
extern  Bag       DiffUnknown ( Bag hdL, Bag hdR );


/****************************************************************************
**
*F  ProdUnknown( <hdL>, <hdR> ) . . . . . . . . . . . product of two unknowns
**
**  'ProdUnknown' returns the product of the two  unknowns  <hd>  and  <hdR>.
**  Either operand may also be a known scalar value.
**
**  Is called from the 'Prod' binop, so both operands are already evaluated.
*/
extern  Bag       ProdUnknown ( Bag hdL, Bag hdR );


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
extern  Bag       QuoUnknown ( Bag hdL, Bag hdR );


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
extern  Bag       PowUnknown ( Bag hdL, Bag hdR );


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
extern  Bag       EqUnknown ( Bag hdL, Bag hdR );


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
extern  Bag       LtUnknown ( Bag hdL, Bag hdR );


/****************************************************************************
**
*F  PrUnknown( <hdUnk> )  . . . . . . . . . . . . . . . . .  print an unknown
**
**  'PrUnknown' prints the unknown <hdUnk> in the form 'Unknown(<n>)'.
*/
extern  void            PrUnknown ( Bag hdUnk );


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
extern  Bag       FunUnknown ( Bag hdCall );


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
extern  Bag       FunIsUnknown ( Bag hdCall );


/****************************************************************************
**
*F  InitUnknown() . . . . . . . . . . . . . .  initialize the unknown package
**
**  'InitUnknown' initializes the unknown package.
*/
extern void             InitUnknown ( void );



