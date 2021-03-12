/****************************************************************************
**
*A  aggroup.h                   GAP source                    Thomas Bischops
*A                                                             & Frank Celler
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
*/


/****************************************************************************
**
*F  EqAg( <hdL>, <hdR> )  . . . . . . . . . .  tests if two agwords are equal
*/
extern Bag    EqAg ( Bag, Bag );


/****************************************************************************
**
*F  LtAg( <hdL>, <hdR> )  . . . . . . . . . . . . . .  tests if <hdL> < <hdR>
*/
extern Bag    LtAg ( Bag, Bag );


/****************************************************************************
**
*F  EvAg( <hdAgWord> )  . . . . .  evaluates a normed word in a soluble group
*/
extern Bag    EvAg ( Bag );


/****************************************************************************
**
*F  ProdAg( <hdL>, <hdR> )  . . . . . . . . . . . . . evaluates <hdL> * <hdR>
*/
extern Bag    ProdAg ( Bag, Bag );


/****************************************************************************
**
*F  PowAgI( <hdL>, <hdR> )  . . . . . . . . . . . . . evaluates <hdL> ^ <hdR>
*/
extern Bag    PowAgI ( Bag, Bag );


/****************************************************************************
**
*F  QuoAg( <hdL>, <hdR> ) . . . . . . . . . . . . . . evaluates <hdL> / <hdR>
*/
extern Bag    QuoAg ( Bag, Bag );


/****************************************************************************
**
*F  ModAg( <hdL>, <hdR> ) . . . . . . . . . . . . . evaluates <hdL> mod <hdR>
*/
extern Bag    ModAg ( Bag, Bag );


/****************************************************************************
**
*F  PowAgAg( <hdL>, <hdR> ) . . . . . . . . . . . . . evaluates <hdL> ^ <hdR>
*/
extern Bag    PowAgAg ( Bag, Bag );


/****************************************************************************
**
*F  CommAg( <hdL>, <hdR> )  . . . . . evaluates the commutator of two agwords
*/
extern Bag    CommAg ( Bag, Bag );


/****************************************************************************
**
*F  FactorAgGroup( <hdG>, <n> ) . . . . . .  factor group of the group of <g>
*F  FunFactorAgGroup( <hdCall> )  . . . . . . . . .  internal 'FactorAgGroup'
*/
extern Bag    FactorAgGroup ( Bag, Int );
extern Bag    FunFactorAgGroup ( Bag );


/****************************************************************************
**
*V  HdRnDepthAgWord . . . . . . . . . . . . . . 'DepthAgWord' record name bag
*F  FunDepthAgWord( <hdCall> )  . . . . . . . . . . .  internal 'DepthAgWord'
*/
extern Bag    FunDepthAgWord ( Bag );


/****************************************************************************
**
*V  HdRnCentralWeightAgWord . . . . . . 'CentralWeightAgWord' record name bag
*F  FunCentralWeightAgWord( <hdCall>  ) . . .  internal 'CentralWeightAgWord'
*/
extern Bag       FunCentralWeightAgWord ( Bag );


/****************************************************************************
**
*V  HdRnLeadingExponentAgWord . . . . 'LeadingExponentAgWord' record name bag
*F  FunLeadingExponentAgWord( <hdCall> )  .  internal 'LeadingExponentAgWord'
*/
extern Bag       FunLeadingExp ( Bag );


/****************************************************************************
**
*F  FunIsAgWord( <hdCall> ) . . . . . . . . . .  internal function 'IsAgWord'
*/
extern Bag    FunIsAgWord ( Bag );


/****************************************************************************
**
*V  HdRnSumAgWord . . . . . . . . . . . handle of 'SumAgWord' record name bag
*F  SumAgWord( <P>, <v>, <w> )  . . . . . . . . . . sum of <v> and <w> in <P>
*F  FunSumAgWord( <hdCall> )  . . . . . . . . . . . . .  internal 'SumAgWord'
*/
extern Bag    HdRnSumAgWord;
extern Bag    SumAgWord ( Bag, Bag, Bag );
extern Bag    FunSumAgWord ( Bag );
    

/****************************************************************************
**
*V  HdRnDifferenceAgWord  . . . . . . . .  'DifferenceAgWord' record name bag
*F  DifferenceAgWord( <P>, <v>, <w> ) . . .  difference of <v> and <w> in <P>
*F  FunDifferenceAgWord( <hdCall> ) . . . . . . . internal 'DifferenceAgWord'
*/
extern Bag    HdRnDifferenceAgWord;
extern Bag    DifferenceAgWord ( Bag, Bag, Bag );
extern Bag    FunDifferenceAgWord ( Bag );
    

/****************************************************************************
**
*V  HdRnExponentsAgWord . . . . . . . . . . 'ExponentsAgWord' record name bag
*F  FFExponentsAgWord( <g>, <s>, <e>, <z> ) . . . . conversion into ff-vector
*F  IntExponentsAgWord( <g>, <s>, <e> ) . . . . .  conversion into int-vector
*F  FunExponentsAgWord( <hdCall> )  . . . . . . .  internal 'ExponentsAgWord'
*/
extern Bag    HdRnExponentsAgWord;
extern Bag    FFExponentsAgWord  (Bag, Int, Int, Bag);
extern Bag    IntExponentsAgWord (Bag, Int, Int);
extern Bag    FunExponentsAgWord (Bag);


/****************************************************************************
**
*V  HdIdAgWord  . . . . . . . . . . . . . . . . . . . . . .  general identity
*F  InitAg()  . . . . . . . . . . . . . . . initializes the collection module
*/
extern Bag    HdIdAgWord;
extern void         InitAg ( void );
