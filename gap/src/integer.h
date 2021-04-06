/****************************************************************************
**
*A  integer.h                   GAP source                   Martin Schoenert
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
**  This file  declares  the  functions  handling  arbitrary  size  integers.
**
**  There are three integer types in GAP: 'T_INT', 'T_INTPOS' and 'T_INTNEG'.
**  Each integer has a unique representation, e.g., an integer  that  can  be
**  represented as 'T_INT' is never  represented as 'T_INTPOS' or 'T_INTNEG'.
**
**  Implementation is taken from GAP4. For details see 'integer4.h'
**
*/

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
*/
#define INT_TO_HD(INT)  ((Bag) (((Int)(INT) << 2) + T_INT))


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
*/
#define HD_TO_INT(HD)   (((Int)HD) >> 2)


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
Bag       FunIsInt ( Bag hdCall );


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
Bag       FunQuo ( Bag hdCall );


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
Bag       FunRem ( Bag hdCall );


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
Bag       FunGcdInt ( Bag hdCall );


/****************************************************************************
**
*F  InitInt() . . . . . . . . . . . . . . . . initializes the integer package
**
**  'InitInt' initializes the arbitrary size integer package.
*/
void            InitInt ( void );
