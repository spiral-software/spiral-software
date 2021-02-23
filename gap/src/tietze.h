/****************************************************************************
**
*A  tietze.h                    GAP source                     Volkmar Felsch
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file defines the functions for computing with finite presentations.
**
*/


/****************************************************************************
**
*F  TzRelExponent1( <relator> ) . . . . find the exponent of a Tietze relator
**
**  'TzRelExponent1'  determines the exponent of the given relator, i. e. the
**  maximal integer n such that the relator can be expresses as an nth power.
*/
extern  Bag       TzRelExponent1 ( Bag hdRel );


/****************************************************************************
**
*F  FunTzRelator( <hdCall> ) . . . . . . . convert a word to a Tietze relator
**
**  'FunTzRelator'  converts a  word in the group generators  to a Tietze re-
**  lator,  i.e. a free and cyclically reduced word in the Tietze generators.
**  It returns 'HdFalse" if it cannot convert the given word.
*/
extern  Bag       FunTzRelator ( Bag hdCall );


/****************************************************************************
**
*F  FunTzWord( <hdCall> ) . . . . . . . .  convert a Tietze relator to a word
**
**  'FunTzWord'  converts a Tietze relator to a word in the group generators.
*/
extern  Bag       FunTzWord ( Bag hdCall );


/****************************************************************************
**
*F  FunTzSortC(<hdCall>)  . . . . . . . . . . . . sort the relators by length
*/
extern  Bag       FunTzSortC ( Bag hdCall );


/****************************************************************************
**
*F  FunTzRenumberGens(<hdCall>)  . . . . . . . renumber the Tietze generators
*/
extern  Bag       FunTzRenumberGens ( Bag hdCall );


/****************************************************************************
**
*F  FunTzReplaceGens(<hdCall>)  . . . replace Tietze generators by other ones
*/
extern  Bag       FunTzReplaceGens ( Bag hdCall );


/****************************************************************************
**
*F  FunTzSubstituteGen(<hdCall>)  . . replace a Tietze generator by some word
*/
extern  Bag       FunTzSubstituteGen ( Bag hdCall );


/****************************************************************************
**
*F  FunTzOccurrences( <hdCall> ) . . . . . . occurrences of Tietze generators
**
**  'FunTzOccurrences' implements the internal function 'TzOccurrences'.
*/
extern  Bag       FunTzOccurrences ( Bag hdCall );


/****************************************************************************
**
*F  FunTzOccurrencesPairs(<hdCall>) occurrences of pairs of Tietze generators
*/
extern  Bag       FunTzOccurrencesPairs ( Bag hdCall );


/****************************************************************************
**
*F  FunTzSearchC(<hdCall>) . . . . .  find subword matches in Tietze relators
**
**  'FunTzSearchC' implements the internal function 'TzSearchC'.
*/
extern  Bag       FunTzSearchC ( Bag hdCall );


/****************************************************************************
**
*F  InitTietze()  . . . . . . . . . . . . . . . . . initialize tietze package
**
**  'InitTietze' initializes the Tietze package.
*/
extern  void            InitTietze ( void );



