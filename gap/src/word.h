/****************************************************************************
**
*A  word.h                      GAP source                   Martin Schoenert
**                                                             & Frank Celler
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file describes the interface of the module for computing with words.
**
*/


/****************************************************************************
**
*T  TypSword  . . . . . . . . . . . . . . . . . . . generator/exponenten list
*V  SIZE_SWORD  . . . . . . . . . . . . . . . . . . . . .  size of 'TypSword'
*D  MAX_SWORD_NR  . . . . . . . . . . . . . . . . .  maximal generator number
**
**  A sparse word <sword> of type 'T_SWORD' has the following structure:
**
**      +-------------+------+-----+-----+-----+-----+----+
**      |   hdList    |  i_1 | e_1 | ... | i_r | e_r | -1 |
**      +-------------+------+-----+-----+-----+-----+----+
**      | Handle area |           data area               |
**      +-------------+-----------------------------------+
**
**  <hdList> is the handle  of  a list  containing the abstract generators of
**  <sword>.  Let this list contain "g_1", ..., "g_n". Then <sword> describes
**  the element
**
**              g_{i_1} ^ e_1 * ... * g_{i_r} ^ e_r,
**
**  where all i_j <> 0. Both "i_j" and "e_j" are of type 'TypSword'.
*/
typedef     short           TypSword;

#define     SIZE_SWORD      ( (UInt) sizeof( TypSword ) )
#define     MAX_SWORD_NR    32768


/****************************************************************************
**
*F  SwordWord( <list>, <word> ) . . . . . . . .  convert/copy word into sword
*F  WordSword( <sword> )  . . . . . . . . . . .  convert/copy sword into word
*F  SwordSword( <list>, <sword> ) . . . . . . . . . . .  copy/convert <sword>
*/
extern Bag    SwordWord ( Bag, Bag );
extern Bag    WordSword ( Bag );
extern Bag    SwordSword ( Bag, Bag );
    

/****************************************************************************
**
*F  Words( <hdStr>, <n> ) . . . . . . . . . . . . . . . . . create <n> swords
*/
extern Bag    Words ( Bag, Int );


/****************************************************************************
**
*F  EvWord( <hdWord> )  . . . . . . . . . . . . . . . . . . . evaluate a word
**
**  This function evaluates a word in abstract generators, since  this  words
**  are constants nothing happens.
*/
extern Bag    EvWord ( Bag );


/****************************************************************************
**
*F  ProdWord( <hdL>, <hdR> )  . . . . . . . . . . . .  eval <wordL> * <wordR>
**
**  This function multplies the two words <hdL> and <hdR>. Since the function
**  is called from evalutor both operands are already evaluated.
*/
extern Bag    ProdWord ( Bag, Bag );


/****************************************************************************
**
*F  QuoWord( <hdL>, <hdR> ) . . . . . . . . . . . eval <wordL> * <wordR> ^ -1
*/
extern Bag    QuoWord ( Bag, Bag );


/****************************************************************************
**
*F  ModWord( <hdL>, <hdR> ) . . . . . . . . . . . eval <wordL> ^ -1 * <wordR>
*/
extern Bag    ModWord ( Bag, Bag );


/****************************************************************************
**
*F  PowWI( <hdL>, <hdR> ) . . . . . . . . . . . . . . . eval <wordL> ^ <intR>
**
**  'PowWI' is  called to evaluate the exponentiation of a word by a integer.
**  It is  called from  th evaluator so both  operands are already evaluated.
*N  This function should be rewritten, it can be faster, but for the moment..
*/
extern Bag    PowWI ( Bag, Bag );


/****************************************************************************
**
*F  PowWW( <hdL>, <hdR> ) . . . . . . . . . . . . . .  eval <wordL> ^ <wordR>
**
**  PowWW() is called to evaluate  the  conjugation  of  two  word  operands.
**  It is called from the evaluator so both operands are  already  evaluated.
*N  This function should be rewritten, it should not call 'ProdWord'.
*/
extern Bag    PowWW ( Bag, Bag );


/****************************************************************************
**
*F  CommWord( <hdL>, <hdR> )  . . . . . . . . . eval comm( <wordL>, <wordR> )
**
**  'CommWord' is  called to evaluate the commutator of  two  word  operands.
**  It is called from the evaluator so both operands are already evaluated.
*/
extern Bag    CommWord ( Bag, Bag );


/****************************************************************************
**
*F  EqWord( <hdL>, <hdR> )  . . . . . . . . . . . .test if <wordL>  = <wordR>
**
**  'EqWord'  is called to  compare  the  two  word  operands  for  equality.
**  It is called from the evaluator so both operands are  already  evaluated.
**  Two speed up the comparism we first check that they have the  same  size.
*/
extern Bag    EqWord ( Bag, Bag );


/****************************************************************************
**
*F  LtWord( <hdL>, <hdR> )  . . . . . . . . . . .  test if <wordL>  < <wordR>
**
**  'LtWord'  is called to test if the left operand is less than  the  right.
**  One word is considered smaller than another if  it  is  shorter,  or,  if
**  both are of equal length if it is first in  the lexicographical ordering.
**  Thus id<a<a^-1<b<b^-1<a^2<a*b<a*b^-1<a^-2<a^-1*b<a^-1*b^-1<b*a<b*a^-1 ...
**  This length-lexical ordering is a well ordering, ie there are no infinite
**  decreasing sequences, and translation invariant on  the  free monoid, ie.
**  if u,v,w are words an  u < v  then  u * w < v * w  if we  don't cancel in
**  between. It is called from the evaluator so  both  operands  are  already
**  evaluated.
*/
extern Bag    LtWord ( Bag, Bag );


/****************************************************************************
**
*F  PrWord( <hdWord> ) . . . . . . . . . . . . . . . . . . . . . print a word
**
**  The function PrWord() prints a word, the empty word is printed as IdWord.
**  A word is printed as a^-5 * b^10.
*/
extern void         PrWord ( Bag );


/****************************************************************************
**
*F  InitWord()  . . . . . . . . . . . . . . . . . . .  initialize word module
**
**  Is called during the initialization of GAP to initialize the word module.
*/
extern void         InitWord ( void );


/****************************************************************************
**
*V  HdIdWord  . . . . . . . . . . . . . . . . . . . . . . . . . identity word
*/
extern Bag    HdIdWord;
