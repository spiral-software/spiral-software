/****************************************************************************
**
*A  permutat.h                  GAP source                   Martin Schoenert
**                                                           & Alice Niemeyer
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This  file  defines  the permutation type,  its operations and functions.
**
**  Mathematically a permutation is a bijective mapping  of a finite set onto
**  itself.  In \GAP\ this subset must always be of the form [ 1, 2, .., N ],
**  where N is at most $2^16$.
**
*/


/****************************************************************************
**
*F  EvPerm( <hdPerm> )  . . . . . . . . . . . evaluate a permutation constant
**
**  'EvPerm'  returns  the value  of    the permutation  <hdPerm>.    Because
**  permutations   are constants and  thus  selfevaluating  this just returns
**  <hdPerm>.
*/
extern  Bag       EvPerm ( Bag hdPerm );


/****************************************************************************
**
*F  EvMakeperm( <hdPerm> )  . . . . . . . . . evaluate a variable permutation
**
**  Evaluates the variable permutation <hdPerm> to  a  constant  permutation.
**  Variable permutations are neccessary because a  permutation  may  contain
**  variables and other stuff whose value is  unknown  until  runtime.  If  a
**  permutation contains no variables it will  be  converted  while  reading.
**
**  This code is a little bit  tricky  in  order  to  avoid  Resize()ing  the
**  permutation bag too often,  which would make this function terribly slow.
*/
extern  Bag       EvMakeperm ( Bag hdPerm );


/****************************************************************************
**
*F  ProdPerm( <hdL>, <hdR> )  . . . . . . . . . . . . product of permutations
**
**  'ProdPerm' returns the product of the two permutations <hdL> and <hdR>.
**
**  Is called from the 'Prod' binop, so both operands are already evaluated.
**
**  This is a little bit tuned but should be sufficiently easy to understand.
*/
extern  Bag       ProdPerm ( Bag hdL, Bag hdR );



/****************************************************************************
**
*F  QuoPerm( <hdL>, <hdR> ) . . . . . . . . . . . .  quotient of permutations
**
**  'QuoPerm' returns the quotient of the permutations <hdL> and <hdR>, i.e.,
**  the product '<hdL>\*<hdR>\^-1'.
**
**  Is called from the 'Quo' binop, so both operands are already evaluated.
**
**  Unfortunatly this can not be done in <degree> steps, we need 2 * <degree>
**  steps.
*/
extern  Bag       QuoPerm ( Bag hdL, Bag hdR );


/****************************************************************************
**
*F  ModPerm( <hdL>, <hdR> ) . . . . . . . . . . left quotient of permutations
**
**  'ModPerm'  returns the  left quotient of  the  two permutations <hdL> and
**  <hdR>, i.e., the value of '<hdL>\^-1*<hdR>', which sometimes comes handy.
**
**  Is called from the 'Mod' binop, so both operands are already evaluated.
**
**  This can be done as fast as a single multiplication or inversion.
*/
extern  Bag       ModPerm ( Bag hdL, Bag hdR );


/****************************************************************************
**
*F  PowPI( <hdL>, <hdR> ) . . . . . . . . . .  integer power of a permutation
**
**  'PowPI' returns the <hdR>-th power of  the permutation <hdL>.  <hdR> must
**  be a small integer.
**
**  Is called from the 'Pow' binop, so both operands are already evaluated.
**
**  This repeatedly applies the permutation <hdR> to all points  which  seems
**  to be faster than binary powering, and does not need  temporary  storage.
*/
extern  Bag       PowPI ( Bag hdL, Bag hdR );


/****************************************************************************
**
*F  PowI <hdL>, <hdR> ) . . . . . . image of an integer under a permutation
**
**  'PowIP' returns  the  image  of  the positive   integer <hdL>  under  the
**  permutation  <hdR>.  If <hdL> is  larger than the degree   of <hdR> it is
**  a fixpoint of the permutation and thus simply returned.
**
**  Is called from the 'Pow' binop, so both operands are already evaluated.
*/
extern  Bag       PowIP ( Bag hdL, Bag hdR );


/****************************************************************************
**
*F  QuoI <hdL>, <hdR> ) . . . .  preimage of an integer under a permutation
**
**  'QuoIP' returns the   preimage of the  preimage integer   <hdL> under the
**  permutation <hdR>.  If <hdL> is larger than  the degree of  <hdR> is is a
**  fixpoint, and thus simply returned.
**
**  Is called from the 'Quo' binop, so both operands are already evaluated.
**
**  There are basically two ways to find the preimage.  One is to run through
**  <hdR>  and  look  for <hdL>.  The index where it's found is the preimage.
**  The other is to  find  the image of  <hdL> under <hdR>, the image of that
**  point and so on, until we come  back to  <hdL>.  The  last point  is  the
**  preimage of <hdL>.  This is faster because the cycles are  usually short.
*/
extern  Bag       QuoIP ( Bag hdL, Bag hdR );


/****************************************************************************
**
*F  PowP <hdL>, <hdR> ) . . . . . . . . . . . . conjugation of permutations
**
**  'PowPP' returns the conjugation of the  two permutations <hdL> and <hdR>,
**  that s defined as the following product '<hdR>\^-1 \*\ <hdL> \*\ <hdR>'.
**
**  Is called from the 'Pow' binop, so both operands are already evaluated.
*/
extern  Bag       PowPP ( Bag hdL, Bag hdR );


/****************************************************************************
**
*F  CommPerm( <hdL>, <hdR> )  . . . . . . . .  commutator of two permutations
**
**  'CommPerm' returns the  commutator  of  the  two permutations  <hdL>  and
**  <hdR>, that is defined as '<hd>\^-1 \*\ <hdR>\^-1 \*\ <hdL> \*\ <hdR>'.
**
**  Is called from the 'Comm' binop, so both operands are already evaluated.
*/
extern  Bag       CommPerm ( Bag hdL, Bag hdR );


/****************************************************************************
**
*F  EqPerm( <hdL>, <hdR> )  . . . . . . .  test if two permutations are equal
**
**  'EqPerm' returns 'true' if the two permutations <hdL> and <hdR> are equal
**  and 'false' otherwise.
**
**  Is called from the 'Eq' binop, so both operands are already evaluated.
**
**  Two permutations may be equal, even if the two sequences do not have  the
**  same length, if  the  larger  permutation  fixes  the  exceeding  points.
*/
extern  Bag       EqPerm ( Bag hdL, Bag hdR );


/****************************************************************************
**
*F  LtPerm( <hdL>, <hdR> )  . test if one permutation is smaller than another
**
**  'LtPerm' returns  'true' if the permutation <hdL>  is strictly  less than
**  the permutation  <hdR>.  Permutations are  ordered lexicographically with
**  respect to the images of 1,2,.., etc.
**
**  Is called from the 'Lt' binop, so both operands are already evaluated.
*/
extern  Bag       LtPerm ( Bag hdL, Bag hdR );


/****************************************************************************
**
*F  PrPerm( <hdPerm> )  . . . . . . . . . . . . . . . . . print a permutation
**
**  'PrPerm' prints the permutation <hdPerm> in the usual cycle notation.  It
**  uses the degree to print all points with same width, which  looks  nicer.
**  Linebreaks are prefered most after cycles and  next  most  after  commas.
*/
extern  void            PrPerm ( Bag hdPerm );


/****************************************************************************
**
*F  PrMakeperm( <hdPerm> )  . . . . . . . . . .  print a variable permutation
**
**  'PrMakeperm' prints the variable permutation <hdPerm>  in the usual cycle
**  notation.
**
**  Linebreaks are prefered most after cycles and  next  most  after  commas.
*/
extern  void            PrMakeperm ( Bag hdPerm );


/****************************************************************************
**
*F  FunIsPerm( <hdCall> ) . . . . . . . .  test if an object is a permutation
**
**  'FunIsPerm' implements the internal function 'IsPerm'.
**
**  'IsPerm( <obj> )'
**
**  'IsPerm' returns 'true' if the object <obj> is a permutation and  'false'
**  otherwise.  Will signal an error if <obj> is an unbound variable.
*/
extern  Bag       FunIsPerm ( Bag hdCall );


/****************************************************************************
**
*F  FunPermList( <hdCall> ) . . . . . . . . . convert a list to a permutation
**
**  'FunPermList' implements the internal function 'PermList'
**
**  'PermList( <list> )'
**
**  Converts the list <list> into a  permutation,  which  is  then  returned.
**
**  'FunPermList' simply copies the list pointwise into  a  permutation  bag.
**  It also does some checks to make sure that the  list  is  a  permutation.
*/
extern  Bag       FunPermList ( Bag hdCall );


/****************************************************************************
**
*F  FunSupportPerm( <hdCall> )  . . . . . . . . . .  support of a permutation
**
**  'FunSupportPerm' implements the internal function 'SupportPerm'.
**
**  'SupportPerm( <perm> )'
**
**  'SupportPerm' returns the largest  positive integer that  is moved by the
**  permutation <perm>.
*/
extern  Bag       FunSupportPerm ( Bag hdCall );


/****************************************************************************
**
*F  FuncCycleLengthPerm( <hdCall> ) . . length of a cycle under a permutation
**
**  'FunCycleLength' implements the internal function 'CycleLengthPerm'.
**
**  'CycleLengthPerm( <perm>, <point> )'
**
**  'CycleLengthPerm' returns the length of the cycle of  <point>, which must
**  be a positive integer, under the permutation <perm>.
**
**  Note that the order of the arguments to this function has been  reversed.
*/
extern  Bag       FunCycleLengthPerm ( Bag hdCall );


/****************************************************************************
**
*F  FunCyclePerm( <hdCall> )  . . . . . . . . . . . .  cycle of a permutation
*
**  'FunCyclePerm' implements the internal function 'CyclePerm'.
**
**  'CyclePerm( <perm>, <point> )'
**
**  'CyclePerm' returns the cycle of <point>, which must  be a small positive
**  integer, under the permutation <perm> as a list.
*/
extern  Bag       FunCyclePerm ( Bag hdCall );


/****************************************************************************
**
*F  FunOrderPerm( <hdCall> )  . . . . . . . . . . . .  order of a permutation
**
**  'FunOrderPerm' implements the internal function 'OrderPerm'.
**
**  'OrderPerm( <perm> )'
**
**  'OrderPerm' returns the  order  of  the  permutation  <perm>,  i.e.,  the
**  smallest positive integer <n> such that '<perm>\^<n>' is the identity.
**
**  Since the largest element in S(65536) has oder greater than  10^382  this
**  computation may easily overflow.  So we have to use  arbitrary precision.
*/
extern  Bag       FunOrderPerm ( Bag hdCall );


/****************************************************************************
**
*F  FunSignPerm( <hdCall> ) . . . . . . . . . . . . . . sign of a permutation
**
**  'FunSignPerm' implements the internal function 'SignPerm'.
**
**  'SignPerm( <perm> )'
**
**  'SignPerm' returns the sign of the permutation <perm>.  The sign is +1 if
**  <perm> is the product of an *even* number of transpositions,  and  -1  if
**  <perm> is the product of an *odd*  number  of  transpositions.  The  sign
**  is a homomorphism from the symmetric group onto the multiplicative  group
**  $\{ +1, -1 \}$, the kernel of which is the alternating group.
*/
extern  Bag       FunSignPerm ( Bag hdCall );


/****************************************************************************
**
*F  FunSmallestGeneratorPerm( <hdCall> )   smallest generator of cyclic group
**
**  'FunSmallestGeneratorPerm' implements the internal function
**  'SmallestGeneratorPerm'.
**
**  'SmallestGeneratorPerm( <perm> )'
**
**  'SmallestGeneratorPerm' returns the   smallest generator  of  the  cyclic
**  group generated by the  permutation  <perm>.  That  is   the result is  a
**  permutation that generates the same  cyclic group as  <perm> and is  with
**  respect  to the lexicographical order  defined  by '\<' the smallest such
**  permutation.
*/
extern  Bag       FunSmallestGeneratorPerm ( Bag hdCall );


/****************************************************************************
**
*F  OnTuplesPerm( <hdTup>, <hdPrm> )  . . . .  operations on tuples of points
**
**  'OnTuplesPerm'  returns  the  image  of  the  tuple  <hdTup>   under  the
**  permutation <hdPrm>.  It is called from 'FunOnTuples'.
*/
extern  Bag       OnTuplesPerm ( Bag hdTup, Bag hdPrm );


/****************************************************************************
**
*F  OnSetsPerm( <hdSet>, <hdPrm> ) . . . . . . . operations on sets of points
**
**  'OnSetsPerm'  returns  the  image  of  the  tuple  <hdSet>   under  the
**  permutation <hdPrm>.  It is called from 'FunOnSets'.
*/
extern  Bag       OnSetsPerm ( Bag hdSet, Bag hdPrm );


/****************************************************************************
**
*F  InitPermutat()  . . . . . . . . . . . initializes the permutation package
**
**  Is  called  during  the  initialization  to  initialize  the  permutation
**  package.
*/
extern  void            InitPermutat ( void );



