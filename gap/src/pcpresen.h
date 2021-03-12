/****************************************************************************
**
*A  pcpresen.h                  GAP source                       Frank Celler
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  The file implements the functions handling finite polycyclic presentation
**  and extends the aggroupmodul implemented in "aggroup.c" and "agcollec.c".
**  Polycyclic presentations are aggroups which can change their presentation
**  but as consequence the elements  cannot  be  multiplied.  Arithmetic  ops
**  can only be performed if the words and the  presentation are  given.  The
**  functions  'ProductPcp',  'QuotientPcp',  'LeftQuotientPcp' and 'CommPcp'
**  implement  the   arithmetic   operations.  'DifferencePcp'  and  'SumPcp'
**  manipulate swords directly without calling a collector.  The presentation
**  itself can be  modified  via  '(Define|Add)(Comm|Power)Pcp',  'ShrinkPcp'
**  'ExtendCentralPcp'. Sometimes collector dependend details can be changed.
**  One expamle is 'DefineCentralWeightsPcp'.
**
**  This is a preliminary implementation used to support the PQ and SQ. Until
**  now no "pcp" with single-collector can be initialised.  But  I  hope  the
**  that the combinatorial "pcp" are no longer preliminary.
**
*/

/*--------------------------------------------------------------------------\
|                         Compilation control flags                         |
\--------------------------------------------------------------------------*/


/****************************************************************************
**
*V  PCP_DEBUG . . . . . . . . . . . . . . . . .  install some debug functions
*/
#ifndef     PCP_DEBUG
#   define  PCP_DEBUG       0
#endif


/*--------------------------------------------------------------------------\
|                                 Prototypes                                |
\--------------------------------------------------------------------------*/

/****************************************************************************
**
*T  boolean . . . . . . . . . . . . . . . . . . . . . . . . . . .  TRUE/FALSE
*/
#ifndef boolean
#define boolean         int
#endif


/****************************************************************************
**
*F  IsNormedPcp( <p>, <*v> )  . . . . . . . . . . . . . . . . is <v> normed ?
*/
extern boolean      IsNormedPcp ( Bag, Bag* );


/****************************************************************************
**
*F  InitPcPres( void )  . . . . . . . . . initialize polycyclic presentations
*/
extern void         InitPcPres ( void );
