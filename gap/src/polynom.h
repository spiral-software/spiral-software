/****************************************************************************
**
*A  polynom.h                    GAP source                      Frank Celler
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
*F  UnifiedFieldVecFFE( <hdL>, <hdR> )	. . . unify fields of <hdL> and <hdR>
**
**  Convert two finite field vectors into finite field vectors over  the same
**  finite field.  Signal an error if this conversion fails.
*/
extern Bag UnifiedFieldVecFFE (
    Bag, Bag );


/****************************************************************************
**
*F  FunShiftedCoeffs( <hdCall> )  . . . . . internal function 'ShiftedCoeffs'
**
**  'FunShiftedCoeffs' implements 'ShiftedCoeffs( <l>, <n> )'
*/
extern Bag (*TabShiftedCoeffs[T_VAR]) ( Bag, Int );
extern Bag FunShiftedCoeffs    ( Bag );
extern Bag CantShiftedCoeffs   ( Bag, Int );
extern Bag ShiftedCoeffsListx  ( Bag, Int );
extern Bag ShiftedCoeffsVecFFE ( Bag, Int );


/****************************************************************************
**
*F  FunNormalizeCoeffs( <hdCall> )  . . . internal function 'NormalizeCoeffs'
**
**  'FunNormalizeCoeffs' implements 'NormalizeCoeffs( <c> )'
*/
extern Bag (*TabNormalizeCoeffs[T_VAR]) ( Bag );
extern Bag FunNormalizeCoeffs    ( Bag );
extern Bag CantNormalizeCoeffs   ( Bag );
extern Bag NormalizeCoeffsVecFFE ( Bag );
extern Bag NormalizeCoeffsListx  ( Bag );


/****************************************************************************
**
*F  FunShrinkCoeffs( <hdCall> )  . . . . . . internal function 'ShrinkCoeffs'
**
**  'FunShrinkCoeffs' implements 'ShrinkCoeffs( <c> )'
*/
extern void (*TabShrinkCoeffs[T_VAR]) ( Bag );

extern Bag FunShrinkCoeffs ( Bag );
extern void CantShrinkCoeffs   ( Bag );
extern void ShrinkCoeffsVecFFE ( Bag );
extern void ShrinkCoeffsListx  ( Bag );


/****************************************************************************
**
*F  ADD_COEFFS( <hdL>, <hdR>, <hdM> ) . . . . . <hdL>+<hdM>*<hdR> into <hdL>
*/
#define ADD_COEFFS( hdL, hdR, hdM ) \
    (TabAddCoeffs[XType(hdL)][XType(hdR)]( hdL, hdR, hdM ))

extern void (*TabAddCoeffs[T_VAR][T_VAR]) (
    Bag, Bag, Bag );

extern void CantAddCoeffs         ( Bag, Bag, Bag );
extern void AddCoeffsListxListx   ( Bag, Bag, Bag );
extern void AddCoeffsVecFFEVecFFE ( Bag, Bag, Bag );
extern void AddCoeffsListxVecFFE  ( Bag, Bag, Bag );


/****************************************************************************
**
*F  FunAddCoeffs( <hdCall> )  . . . . . . . . . internal function 'AddCoeffs'
**
**  'FunAddCoeffs' implements 'AddCoeffs( <l>, <r> )'
*/
extern Bag FunAddCoeffs ( Bag );


/****************************************************************************
**
*F  FunSumCoeffs( <hdCall> )  . . . . . . . . . internal function 'SumCoeffs'
**
**  'FunSumCoeffs' implements 'SumCoeffs( <l>, <r> )'
*/
extern Bag FunSumCoeffs ( Bag );


/****************************************************************************
**
*F  MULTIPLY_COEFFS( <hdP>, <hdL>, <l>, <hdR>, <r> )   <hdL>*<hdR> into <hdP>
*/
#define MULTIPLY_COEFFS(hdP,hdL,l,hdR,r) \
    (TabMultiplyCoeffs[XType(hdL)][XType(hdR)](hdP,hdL,l,hdR,r))

extern Int (*TabMultiplyCoeffs[T_VAR][T_VAR]) (
    Bag, Bag, Int, Bag, Int );

extern Int CantMultiplyCoeffs (
    Bag, Bag, Int, Bag, Int );

extern Int MultiplyCoeffsListxListx (
    Bag, Bag, Int, Bag, Int );

extern Int MultiplyCoeffsVecFFEVecFFE (
    Bag, Bag, Int, Bag, Int );


/****************************************************************************
**
*F  FunProductCoeffs( <hdCall> )  . . . . . internal function 'ProductCoeffs'
**
**  'FunProductCoeffs' implements 'ProductCoeffs( <l>, <r> )'
*/
extern Bag (*TabProductCoeffs[T_VAR][T_VAR]) (
    Bag, Bag );

extern Bag FunProductCoeffs ( Bag );
extern Bag CantProductCoeffs         ( Bag, Bag );
extern Bag ProductCoeffsListxListx   ( Bag, Bag );
extern Bag ProductCoeffsVecFFEVecFFE ( Bag, Bag );


/****************************************************************************
**
*F  FunProductCoeffsMod( <hdCall> ) . .  internal function 'ProductCoeffsMod'
**
**  'FunProductCoeffsMod' implements 'ProductCoeffsMod( <l>, <r>, <p> )'
*/
extern Bag (*TabProductCoeffsMod[T_VAR][T_VAR]) (
    Bag, Bag, Bag );

extern Bag FunProductCoeffsMod ( Bag );

extern Bag CantProductCoeffsMod (
    Bag, Bag, Bag );

extern Bag ProductCoeffsModListxListx (
    Bag, Bag, Bag );


/****************************************************************************
**
*F  REDUCE_COEFFS( <hdL>, <l>, <hdR>, <r> ) . . . . . . reduce <hdL> by <hdR>
*/
#define REDUCE_COEFFS( hdL, l, hdR, r ) \
    (TabReduceCoeffs[XType(hdL)][XType(hdR)]( hdL, l, hdR, r ))

extern Int (*TabReduceCoeffs[T_VAR][T_VAR]) (
    Bag, Int, Bag, Int );

extern Int CantReduceCoeffs (
    Bag, Int, Bag, Int );

extern Int ReduceCoeffsListxListx (
    Bag, Int, Bag, Int );

extern Int ReduceCoeffsVecFFEVecFFE (
    Bag, Int, Bag, Int );


/****************************************************************************
**
*F  FunReduceCoeffs( <hdCall> ) . . . . . .  internal function 'ReduceCoeffs'
**
**  'FunReduceCoeffs' implements 'ReduceCoeffs( <l>, <r> )'
*/
extern Bag FunReduceCoeffs ( Bag );


/****************************************************************************
**
*F  FunRemainderCoeffs( <hdCall> )  . . . internal function 'RemainderCoeffs'
**
**  'FunRemainderCoeffs' implements 'RemainderCoeffs( <l>, <r> )'
*/
extern Bag FunRemainderCoeffs ( Bag );


/****************************************************************************
**
*F  REDUCE_COEFFS_MOD( <hdL>, <l>, <hdR>, <r>, <hdN> )  reduce <hdL> by <hdR>
*/
#define REDUCE_COEFFS_MOD( hdL, l, hdR, r ) \
    (TabReduceCoeffsMod[XType(hdL)][XType(hdR)]( hdL, l, hdR, r, hdN ))

extern Int (*TabReduceCoeffsMod[T_VAR][T_VAR]) (
    Bag, Int, Bag, Int, Bag );

extern Int CantReduceCoeffsMod (
    Bag, Int, Bag, Int, Bag );

extern Int ReduceCoeffsModListxListx (
    Bag, Int, Bag, Int, Bag );

extern Int ReduceCoeffsModListx (
    Bag, Int, Bag, Int, Bag );


/****************************************************************************
**
*F  FunReduceCoeffsMod( <hdCall> )  . . . internal function 'ReduceCoeffsMod'
**
**  'FunReduceCoeffsMod' implements 'ReduceCoeffsMod( <l>, <r>, <p> )'
*/
extern Bag FunReduceCoeffsMod ( Bag );


/****************************************************************************
**
*F  FunPowerModCoeffs( <hdCall> ) . . . .  internal function 'PowerModCoeffs'
**
**  'FunPowerModCoeffs' implements 'PowerModCoeffs( <g>, <n>, <r> )'
*/
extern Bag (*TabPowerModCoeffsInt[T_VAR][T_VAR]) (
    Bag, Bag, Bag );

extern Bag (*TabPowerModCoeffsLInt[T_VAR][T_VAR]) (
    Bag, Bag, Bag );

extern Bag FunPowerModCoeffs ( Bag );
extern Bag PowerModListxIntListx    (Bag,Bag,Bag);
extern Bag PowerModVecFFEIntVecFFE  (Bag,Bag,Bag);
extern Bag PowerModListxLIntListx   (Bag,Bag,Bag);
extern Bag PowerModVecFFELIntVecFFE (Bag,Bag,Bag);
extern Bag CantPowerModCoeffs       (Bag,Bag,Bag);


/****************************************************************************
**
*F  InitPolynom() . . . . . . . . . . . . . .  initialize the polynom package
*/
extern void InitPolynom ( void );
