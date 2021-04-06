/****************************************************************************
**
*A  costab.h                    GAP source                   Martin Schoenert
*A                                                           & Volkmar Felsch
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file defines the functions for computing with coset tables.
**
*/


/****************************************************************************
**
*F  CompressDeductionList( )  . . .  removes unused items from deduction list
**
**  'CompressDeductionList'  tries to find and delete  deduction list entries
**  which are not used any more.
**
**  'dedgen',  'dedcos',  'dedfst',  'dedlst',  'dedSize'  and 'hdTable'  are
**  assumed to be known as static variables.
*/
extern  void            CompressDeductionList ( void );


/****************************************************************************
**
*F  FunApplyRel( <hdCall> ) . . . . . . .  apply a relator to a coset in a TC
**
**  'FunApplyRel' implements the internal function 'ApplyRel'.
**
**  'ApplyRel( <app>, <rel> )'
**
**  'ApplyRel'  applies the relator  <rel>  to the  application  list  <app>.
**  ...more about ApplyRel...
*/
extern  Bag       FunApplyRel ( Bag hdCall );


/****************************************************************************
**
*F  HandleCoinc(<cos1>,<cos2>) . . . . . . . . .  handle coincidences in a TC
**
**  'HandleCoinc'  is a subroutine of  'FunMakeConsequences'  and handles the
**  coincidence  cos2 = cos1.
*/
extern  void            HandleCoinc ( UInt, UInt );


/*****************************************************************************
**
*F  FunMakeConsequences(<hdCall>) . . find consequences of a coset definition
*/
extern  Bag       FunMakeConsequences ( Bag hdCall );


/****************************************************************************
**
*F  FunStandardizeTable(<hdCall>)  . . . . . . . .  standardize a coset table
*/
extern  Bag       FunStandardizeTable ( Bag hdCall );


/****************************************************************************
**
*F  InitializeCosetFactorWord( )  . . . . . .  initialize a coset factor word
**
**  'InitializeCosetFactorWord'  initializes  a word  in  which  a new  coset
**  factor is to be built up.
**
**  'treeType', 'hdTree2',  and  'treeWordLength'  are assumed to be known as
**  static variables.
*/
extern  void            InitializeCosetFactorWord ( void );


/****************************************************************************
**
*F  AddCosetFactor( <hdfactor> ) . . . . . . . . . . . add a coset rep factor
**
**  'AddCosetFactor' adds a factor to a coset representative word by changing
**  its exponent appropriately.
**
**  'treeType', 'hdWordValue',  and  'hdExponent'  are assumed to be known as
**  static variables, and 'treeType' is assumed to be 1.
**
**  Warning: 'factor' is not checked for being zero.
*/
extern  void            AddCosetFactor ( Bag );


/****************************************************************************
**
*F  AddCosetFactor2( <factor> ) . add a factor to a coset representative word
**
**  'AddCosetFactor2'  adds  a  factor  to a  coset  representative word  and
**  extends the tree appropriately, if necessary.
**
**  'treeType', 'wordList', and 'wordSize'  are assumed to be known as static
**  variables, and 'treeType' is assumed to be either 0 or 2,
**
**  Warning: 'factor' is not checked for being zero.
*/
extern  void            AddCosetFactor2 ( Int );


/****************************************************************************
**
*F  SubtractCosetFactor( <hdfactor> )  . . . . .  subtract a coset rep factor
**
**  'SubtractCosetFactor' subtracts a factor from a coset representative word
**  by changing its exponent appropriately.
**
**  'treeType', 'hdWordValue',  and  'hdExponent'  are assumed to be known as
**  static variables, and 'treeType' is assumed to be 1.
**
**  Warning: 'factor' is not checked for being zero.
*/
extern  void            SubtractCosetFactor ( Bag );


/****************************************************************************
**
*F  FunApplyRel2( <hdCall> ) . . . .  apply a relator to a coset rep in a CRT
**
**  'FunApplyRel2' implements the internal function 'ApplyRel2'.
**
**  'ApplyRel2( <app>, <rel>, <nums> )'
**
**  'ApplyRel2'  applies  the relator  <rel>  to a  coset representative  and
**  returns the corresponding factors in "word"
**  ...more about ApplyRel2...
*/
extern  Bag       FunApplyRel2 ( Bag hdCall );


/****************************************************************************
**
*F  FunCopyRel( <hdCall> ) . . . . . . . . . . . . . . . .  copy of a relator
**
**  'FunCopyRel'  returns a copy of the given  RRS relator  such that the bag
**  of the copy does not exceed the minimal required size.
*/
extern  Bag       FunCopyRel ( Bag hdCall );


/****************************************************************************
**
*F  FunMakeCanonical( <hdCall> ) . . . . . . . . . . make a relator canonical
**
**  'FunMakeCanonical'  is a subroutine of the  Reduced Reidemeister-Schreier
**  routines.  It replaces the given relator by its canonical representative.
**  It does not return anything.
*/
extern  Bag       FunMakeCanonical ( Bag hdCall );


/****************************************************************************
**
*F  FunTreeEntry( <hdCall> ) . . . .  returns a tree entry for the given word
**
**  'FunTreeEntry'  determines a  tree entry  which represents the given word
**  in the current generators,  if it finds any,  or it defines a  new proper
**  tree entry, and then returns it.
*/
extern  Bag       FunTreeEntry ( Bag hdCall );


/****************************************************************************
**
*F  TreeEntryC( )  . . . . . . . . . . .  returns a tree entry for a rep word
**
**  'TreeEntryC'  determines a tree entry  which represents the word given in
**  'wordList', if it finds any, or it defines a  new proper tree entry,  and
**  then returns it.
**
**  Warning:  It is assumed,  but not checked,  that the given word is freely
**  reduced  and that it does  not contain zeros,  and that the  tree type is
**  either 0 or 2.
**
**  'wordList'  is assumed to be known as static variable.
**
*/
extern  Int            TreeEntryC ( void );


/****************************************************************************
**
*F  HandleCoinc2(<cos1>,<cos2>,<hdfactor>)  . . handle coincidences in an MTC
**
**  'HandleCoinc2'  is a subroutine of 'FunMakeConsequences2' and handles the
**  coincidence  cos2 = factor * cos1.
*/
extern  void            HandleCoinc2 ( Int, Int, Bag );


/****************************************************************************
**
*F  FunMakeConsequences2(<hdCall>) .  find consequences of a coset definition
*/
extern  Bag       FunMakeConsequences2 ( Bag hdCall );


/****************************************************************************
**
*F  FunStandardizeTable2(<hdCall>) . . . . . . . . . standardize augmented CT
**
**  'FunStandardizeTable2' standardizes an augmented coset table.
*/
extern  Bag       FunStandardizeTable2 ( Bag hdCall );


/****************************************************************************
**
*F  FunAddAbelianRelator( <hdCall> ) . . . . . . internal 'AddAbelianRelator'
**
**  'FunAddAbelianRelator' implements 'AddAbelianRelator( <rels>, <number> )'
*/
extern  Bag       FunAddAbelianRelator ( Bag hdCall );


/****************************************************************************
**
*F  InitCostab()  . . . . . . . . . . . .  initialize the coset table package
**
**  'InitCostab' initializes the coset table package.
*/
extern  void            InitCosTab ( void );



