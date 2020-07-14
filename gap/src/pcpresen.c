/****************************************************************************
**
*A  pcpresen.c                  GAP source                       Frank Celler
**
**
*Y  Copyright (C) 2018-2020, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2020) by the GAP Group (www.gap-system.org).
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

#include        "system.h"          /** system dependent functions        **/
#include        "memmgr.h"          /** dynamic storage manager           **/
#include        "scanner.h"         /** reading of tokens and printing    **/
#include        "eval.h"            /** evaluator main dispatcher         **/
#include        "integer.h"         /** arbitrary size integers           **/
#include        "idents.h"          /** 'FindRecname' is here             **/
#include        "list.h"            /** 'IsList' is here                  **/
#include        "plist.h"           /* plain list package                  */
#include        "word.h"            /** swords live here                  **/
#include        "aggroup.h"         /** solvable groups                   **/
#include        "agcollec.h"        /** solvable groups, private defs     **/

#include        "pcpresen.h"        /** presentation stuff                **/


/*--------------------------------------------------------------------------\
|                         polycyclic presentations                          |
\--------------------------------------------------------------------------*/


/****************************************************************************
**
*F  FunPcp( <hdCall> )  . . . . . . . . . . . . . . . . . . .  internal 'Pcp'
**
**  'FunPcp' implementes 'Pcp( <str>, <n>, <p>, <collector> )'
**
**  'Pcp' initializes a presentation of an elementary abelian <p>-group with
**  <n>-generators and collector <collector>.
*/
Bag       FunPcp (Bag hdCall)
{
    Bag       hdLst,  hdGrp,  hdN,  hdP,  hdStr,  hdCol,  hdSwrds;
    Int            p,  n,  i;
    char            * usage = "usage: Pcp( <str>, <n>, <p>, <collector> )";

    /** Evaluate and check the arguments. **********************************/
    if ( GET_SIZE_BAG( hdCall ) != 5 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdStr = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdN   = EVAL( PTR_BAG( hdCall )[ 2 ] );
    hdP   = EVAL( PTR_BAG( hdCall )[ 3 ] );
    hdCol = EVAL( PTR_BAG( hdCall )[ 4 ] );
    if (    GET_TYPE_BAG( hdStr ) != T_STRING
         || GET_TYPE_BAG( hdN   ) != T_INT
         || GET_TYPE_BAG( hdP   ) != T_INT
         || GET_TYPE_BAG( hdCol ) != T_STRING )
    {
        return Error( usage, 0, 0 );
    }

    /** Check <n> against 'MAX_SWORD_GEN', <p> must be a prime. ************/
    n = HD_TO_INT( hdN );
    p = HD_TO_INT( hdP );
    if ( n < 1 )
        return Error( "Pcp: <n> must be positive", 0, 0 );
    if ( n > MAX_SWORD_NR )
        return Error( "Pcp: <n> must be less then %d",
                      (Int) MAX_SWORD_NR, 0 );
    for ( i = 2;  i < p;  i++ )
        if ( p % i == 0 )
            return Error( "Pcp: <p> must be a prime", 0, 0 );

    /** Allocate the internal group-bag to store the group-informations. ***/
    hdGrp = BlankAgGroup();
    SET_BAG( hdGrp, NR_NUMBER_OF_GENS, INT_TO_HD( n ));
    hdLst = NewBag( T_AGEXP, SIZE_EXP * n );
    SET_BAG( hdGrp, NR_SAVE_EXPONENTS, hdLst);
    hdLst = NewBag( T_AGEXP, SIZE_EXP * n );
    SET_BAG( hdGrp, NR_COLLECT_EXPONENTS, hdLst);
    hdLst = NewBag( T_AGEXP, SIZE_EXP * n );
    SET_BAG( hdGrp, NR_COLLECT_EXPONENTS_2, hdLst);
    ClearCollectExponents( hdGrp );
    SetGeneratorsAgGroup( hdGrp );

    hdLst = NewBag( T_LIST, ( n + 1 ) * SIZE_HD );
    SET_BAG( hdLst ,  0 ,  INT_TO_HD( n ) );
    SET_BAG( hdGrp, NR_POWERS, hdLst);
    for ( i = n;  i > 0;  i-- )
        SET_BAG( hdLst ,  i ,  HD_IDENTITY( hdGrp ) );

    hdLst = NewBag( T_LIST, ( n * ( n - 1 ) / 2 + 1 ) * SIZE_HD );
    SET_BAG( hdLst ,  0 ,  INT_TO_HD( n * ( n - 1 ) / 2 ) );
    SET_BAG( hdGrp, NR_COMMUTATORS, hdLst);
    for ( i = n * ( n - 1 ) / 2;  i > 0;  i-- )
        SET_BAG( hdLst ,  i ,  HD_IDENTITY( hdGrp ) );

    hdLst = NewBag( T_INTPOS, n * sizeof( Int ) );
    SET_BAG( hdGrp, NR_INDICES, hdLst);
    for ( i = n - 1;  i >= 0;  i-- )
        SET_INDICES( hdGrp, i, p);

    /** Set collector and collector depended entries. **********************/
    for ( i = 0;  i <= COMBI_COLLECTOR;  i++ )
      if ( ! strcmp( Collectors[ i ].name, (char*) PTR_BAG( hdCol ) ) )
          break;
    if ( i > COMBI_COLLECTOR )
        return Error("Pcp: unknown collector \"%s\"", (Int)PTR_BAG(hdCol), 0);
    SET_BAG( hdGrp, NR_COLLECTOR, INT_TO_HD( i ));
    if ( i == COMBI_COLLECTOR || i == COMBI2_COLLECTOR )
    {
        SetCWeightsAgGroup( hdGrp, HdVoid );
        if ( p == 2 )
            SET_BAG( hdGrp, NR_COLLECTOR, INT_TO_HD( COMBI2_COLLECTOR ));
        else
            SET_BAG( hdGrp, NR_COLLECTOR, INT_TO_HD( COMBI_COLLECTOR ));
    }
    else if ( i == LEE_COLLECTOR )
    {
        SetCWeightsAgGroup( hdGrp, HdVoid );
        SET_BAG( hdGrp, NR_COLLECTOR, INT_TO_HD( LEE_COLLECTOR ));
    }
    else
        return Error( "Pcp: not ready yet", 0, 0 );

    /** Retype the generators and identity. ********************************/
    for ( i = n - 1;  i >= 0;  i-- )
        Retype( GENERATORS( hdGrp )[ i ], T_SWORD );
    Retype( HD_IDENTITY( hdGrp ), T_SWORD );

    /** Construct <n> new abstract generators. *****************************/
    hdSwrds = Words( hdStr, n );
    SET_BAG( hdGrp, NR_WORDS, PTR_BAG( PTR_BAG( hdSwrds )[ 1 ] )[0]);
    SetStacksAgGroup( hdGrp );
    Retype( hdGrp, T_AGGRP );

    hdLst = NewBag( T_PCPRES, SIZE_HD );
    SET_BAG( hdLst ,  0 ,  hdGrp );
    return hdLst;
}


/****************************************************************************
**
*F  FunAgPcp( <P> ) . . . . . . . . . . . . . . . . . . . .  internal 'AgPcp'
*/
Bag       FunAgPcp (Bag hdCall)
{
    Bag           hdP, hdTmp;
    extern Bag    GapAgGroup ( Bag );
    char                * usage = "usage: AgPcp( <P> )";

    /** Check and evaluate arguments ***************************************/
    if ( GET_SIZE_BAG( hdCall ) != 2 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    if ( GET_TYPE_BAG( hdP ) != T_PCPRES )
        return Error( usage, 0, 0 );
    hdP = *PTR_BAG( hdP );

    /** Use 'FactorAgGroup' in order to copy presenation. ******************/
    hdTmp = FactorAgGroup( hdP, NUMBER_OF_GENS( hdP ) );
    return GapAgGroup( hdTmp );
}


/****************************************************************************
**
*F  FunGeneratorsPcp( <P> ) . . . . . . . . . . . .  internal 'GeneratorsPcp'
**
**  'FunGeneratorsPcp' implements 'GeneratorsPcp( <P> )'
**
**  'GeneratorsPcp'  returns  the list of generators of <P>.  Note that we do
**  return a copy of that list.
*/
Bag       FunGeneratorsPcp (Bag hdCall)
{
    Bag       hdP;
    char            * usage = "usage: GeneratorsPcp( <P> )";

    /** Check and evaluate arguments ***************************************/
    if ( GET_SIZE_BAG( hdCall ) != 2 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    if ( GET_TYPE_BAG( hdP ) != T_PCPRES )
        return Error( usage, 0, 0 );

    /** Return a copy of the generators. ***********************************/
    return Copy( HD_GENERATORS( *PTR_BAG( hdP ) ) );
}


/****************************************************************************
**
*F  FunExtendCentralPcp( <hdCall> ) . . . . . . . internal 'ExtendCentralPcp'
**
**  'FunExtendCentralPcp' implements 'ExtendCentralPcp( <P>, <L>, <p> )'
**
**  Extend the presentation <P> central by the given generators <L> which are
**  of order <p>.
*/
Bag       FunExtendCentralPcp (Bag hdCall)
{
    Bag       * ptL,  hdL,  hdP,  hdN,  hdA,  hdI,  hdTmp;
    Int            i,  j,  old,  new,  len,  p;
    char            * usage = "usage: ExtendCentralPcp( <P>, <L>, <p> )";

    /** Evaluate and check the arguments. **********************************/
    if ( GET_SIZE_BAG( hdCall ) != 4 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdL = EVAL( PTR_BAG( hdCall )[ 2 ] );
    hdN = EVAL( PTR_BAG( hdCall )[ 3 ] );
    if ( GET_TYPE_BAG(hdP) != T_PCPRES || GET_TYPE_BAG(hdL) != T_LIST || GET_TYPE_BAG(hdN) != T_INT )
        return Error( usage, 0, 0 );
    hdP = *PTR_BAG( hdP );

    /** Check the new generators. ******************************************/
    p   = HD_TO_INT( hdN );
    old = NUMBER_OF_GENS( hdP );
    len = LEN_LIST( hdL );
    new = old + len;
    hdL = Copy( hdL );
    for ( i = len;  i > 0;  i-- )
    {
        hdTmp = ELM_PLIST( hdL, i );
        if ( GET_TYPE_BAG( hdTmp ) != T_STRING )
            return Error( usage, 0, 0 );
        hdA = NewBag( T_AGEN, SIZE_HD + GET_SIZE_BAG( hdTmp ) + 1 );
        *(char*)( PTR_BAG( hdA ) + 1 ) = '\0';
        strncat( (char*)( PTR_BAG( hdA ) + 1 ), "+", 1 );
        strncat( (char*)( PTR_BAG( hdA ) + 1 ),
                   (char*)( PTR_BAG( hdTmp ) ),
                   GET_SIZE_BAG( hdTmp ) - 1 );
        hdI = NewBag( T_AGEN, SIZE_HD + GET_SIZE_BAG( hdTmp ) + 1 );
        *(char*)( PTR_BAG( hdI ) + 1 ) = '\0';
        strncat( (char*)( PTR_BAG( hdI ) + 1 ), "-", 1 );
        strncat( (char*)( PTR_BAG( hdI ) + 1 ),
                   (char*)( PTR_BAG( hdTmp ) ),
                   GET_SIZE_BAG( hdTmp ) - 1 );
        SET_BAG( hdA ,  0 ,  hdI );
        SET_BAG( hdI ,  0 ,  hdA );
        SET_ELM_PLIST( hdL, i, hdA );
    }

    /** Collector depend check *********************************************/
    switch ( COLLECTOR( hdP ) )
    {
        case LEE_COLLECTOR:
        case COMBI_COLLECTOR:
        case COMBI2_COLLECTOR:
            if ( p != INDICES( hdP )[ 0 ] )
                return Error( "can only extend with prime %d",
                              INDICES( hdP )[ 0 ], 0 );
            break;
        default:
            return Error( "ExtendCentralPcp: not ready!", 0, 0 );
            /* break; */
    }

    /** Extend the <GENERATORS>. *******************************************/
    SET_BAG( hdP, NR_NUMBER_OF_GENS, INT_TO_HD( new ));
    SetGeneratorsAgGroup( hdP );
    for ( i = new - 1;  i >= 0;  i-- )
        Retype( GENERATORS( hdP )[ i ], T_SWORD );
    Retype( HD_IDENTITY( hdP ), T_SWORD );

    /** Resize <SAVE_EXPONENTS> and <COLLECT_EXPONENTS> ********************/
    Resize( HD_SAVE_EXPONENTS( hdP ),      new * SIZE_EXP );
    Resize( HD_COLLECT_EXPONENTS( hdP ),   new * SIZE_EXP );
    Resize( HD_COLLECT_EXPONENTS_2( hdP ), new * SIZE_EXP );

    /** Resize <POWERS> and append the new trivial rhs *********************/
    Resize( HD_POWERS( hdP ), ( new + 1 ) * SIZE_HD );
    ptL = POWERS( hdP );
    ptL[ -1 ] = INT_TO_HD( new );
    for ( i = old;  i < new;  i++ )
        ptL[ i ] = HD_IDENTITY( hdP );
    //  CHANGED_BAG(HD_POWERS( hdP ));
    /** Resize <COMMUTATORS> and append the new trivial rhs ****************/
    Resize( HD_COMMUTATORS( hdP ), ( new*(new-1)/2 + 1 ) * SIZE_HD );
    ptL = COMMUTATORS( hdP );
    ptL[ -1 ] = INT_TO_HD( new * ( new - 1 ) / 2 );
    for ( i = old * (old-1) / 2;  i < new * (new-1) / 2;  i++ )
        ptL[ i ] = HD_IDENTITY( hdP );
    //  CHANGED_BAG(HD_COMMUTATORS( hdP ));
    /** Resize <INDICES> and add the indices of the new generators. ********/
    Resize( HD_INDICES( hdP ), new * sizeof( Int ) );
    for ( i = old;  i < new;  i++ )
        SET_INDICES( hdP, i, p);

    /** Resize <WORDS> and add the new generators. *************************/
    Resize( HD_WORDS( hdP ), ( new + 1 ) * SIZE_HD );
    ptL = WORDS( hdP );
    ptL[ -1 ] = INT_TO_HD( new );
    for ( i = old;  i < new;  i++ )
        ptL[ i ] = ELM_PLIST( hdL, i + 1 - old );
    //  CHANGED_BAG(HD_WORDS( hdP ));
    /** Collector depend part **********************************************/
    switch ( COLLECTOR( hdP ) )
    {
        case LEE_COLLECTOR:
        case COMBI_COLLECTOR:
        case COMBI2_COLLECTOR:
            Resize( HD_CWEIGHTS( hdP ), new * sizeof( Int ) );
            j = CWEIGHTS( hdP )[ old - 1 ] + 1;
            for ( i = old;  i < new;  i++ )
                CWEIGHTS( hdP )[ i ] = j;
            len = GET_SIZE_BAG( HD_CSERIES( hdP ) ) / sizeof( Int );
            Resize( HD_CSERIES( hdP ), ( len + 1 ) * sizeof( Int ) );
            CSERIES( hdP )[ len ] = new - 1;
            CSERIES( hdP )[ 0 ] = CSERIES( hdP )[ 0 ] + 1;
            break;
        default:
            return Error( "ExtendCentralPcp: not ready!", 0, 0 );
    }
    return HdVoid;
}


/****************************************************************************
**
*F  FunCentralWeightsPcp( <hdCall> )  . . . . . . internal 'CentralWeightPcp'
**
**  'FunCentralWeightsPcp' implements 'CentralWeightsPcp( <P> )'
*/
Bag       FunCentralWeightsPcp (Bag hdCall)
{
    Bag       hdVec,  hdP;
    Bag       * ptVec;
    Int            i,  n,  * ptWgt;
    char            * usage = "usage: CentralWeightsPcp( <P> )";

    /** Evaluate and check the arguments. **********************************/
    if ( GET_SIZE_BAG( hdCall ) != 2 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    if ( GET_TYPE_BAG( hdP ) != T_PCPRES )
        return Error( usage, 0, 0 );
    hdP = *PTR_BAG( hdP );

    /** We must have the combinatorial collector. **************************/
    switch( COLLECTOR( hdP ) )
    {
        case LEE_COLLECTOR:
        case COMBI_COLLECTOR:
        case COMBI2_COLLECTOR:
            break;
        default:
            return Error( "combinatorial collector not installed", 0, 0 );
            /* break; */
    }

    /** Get central weights. ***********************************************/
    n = NUMBER_OF_GENS( hdP );
    hdVec = NewBag( T_LIST, ( n + 1 ) * SIZE_HD );
    ptVec = PTR_BAG( hdVec ) + 1;
    ptVec[ -1 ] = INT_TO_HD( n );
    ptWgt = CWEIGHTS( hdP );
    for ( i = n - 1;  i >= 0;  i-- )
        ptVec[ i ] = INT_TO_HD( ptWgt[ i ] );

    return hdVec;
}


/****************************************************************************
**
*F  FunDefineCentralWeightsPcp( <hdCall> )  . . . .  'DefineCentralWeightPcp'
**
**  'Fun...' implements 'DefineCentralWeightsPcp( <P>, <W> )'
*/
Bag       FunDefineCentralWeightsPcp (Bag hdCall)
{
    Bag       hdVec,  hdP;
    Bag       * ptVec;
    Int            i,  n;
    char            * usage = "DefineCentralWeightsPcp( <P>, <W> )";

    /** Evaluate and check the arguments. **********************************/
    if ( GET_SIZE_BAG( hdCall ) != 3 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP   = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdVec = EVAL( PTR_BAG( hdCall )[ 2 ] );
    if ( GET_TYPE_BAG( hdP ) != T_PCPRES || ! IsList( hdVec ) )
        return Error( usage, 0, 0 );
    hdP = *PTR_BAG( hdP );

    /** We must have the combinatorial collector. **************************/
    switch( COLLECTOR( hdP ) )
    {
        case LEE_COLLECTOR:
        case COMBI_COLLECTOR:
        case COMBI2_COLLECTOR:
            break;
        default:
            return Error( "combinatorial collector not installed", 0, 0 );
            /* break; */
    }

    /** Set central weights. ***********************************************/
    n = NUMBER_OF_GENS( hdP );
    if ( LEN_LIST( hdVec ) > n )
        return Error( "presentation has only %d generators", n, 0 );
    ptVec = PTR_BAG( hdVec ) + 1;
    for ( i = LEN_LIST( hdVec ) - 1;  i >= 0;  i-- )
        if ( GET_TYPE_BAG( ptVec[ i ] ) != T_INT )
            return Error( usage, 0, 0 );
    if ( LEN_LIST( hdVec ) != n )
    {
        hdVec = Copy( hdVec );
        i = LEN_LIST( hdVec ) + 1;
        Resize( hdVec, ( n + 1 ) * SIZE_HD );
        SET_BAG( hdVec ,  0 ,  INT_TO_HD( n ) );
        ptVec = PTR_BAG( hdVec );
        if ( i == 1 )
        {
            ptVec[ i++ ] = INT_TO_HD( 1 );
        }
        for ( ;  i <= n;  i++ )
            ptVec[ i ] = ptVec[ i - 1 ];
    }
    SetCWeightsAgGroup( hdP, hdVec );
    return HdVoid;
}


/****************************************************************************
**
*F  FunDefineCommPcp( <hdCall> )  . . . . . . . . . . . . . . 'DefineCommPcp'
**
**  'FunDefineCommPcp' implements 'DefineCommPcp( <P>, <i>, <j>, <w> )'
*/
Bag       FunDefineCommPcp (Bag hdCall)
{
    Bag       hdW,  hdP,  hdI,  hdJ;
    Int            len,  i,  j,  * ptWgt;
    char            * usage = "DefineCommPcp( <P>, <i>, <j>, <w> )";

    /** Evaluate and check the arguments. **********************************/
    if ( GET_SIZE_BAG( hdCall ) != 5 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdI = EVAL( PTR_BAG( hdCall )[ 2 ] );
    hdJ = EVAL( PTR_BAG( hdCall )[ 3 ] );
    hdW = EVAL( PTR_BAG( hdCall )[ 4 ] );
    if ( GET_TYPE_BAG(hdP) != T_PCPRES || GET_TYPE_BAG(hdI) != T_INT || GET_TYPE_BAG(hdJ) != T_INT )
        return Error( usage, 0, 0 );
    hdP = *PTR_BAG( hdP );
    len = NUMBER_OF_GENS( hdP );
    i   = HD_TO_INT( hdI ) - 1;
    j   = HD_TO_INT( hdJ ) - 1;
    if ( i < 0 || j < 0 )
        return Error( "generator number must be positive", 0, 0 );
    if ( i >= len || j >= len )
        return Error( "presenation has only %d generators", len, 0 );
    if ( i <= j )
        return Error( "<i> must be greater than <j>", 0, 0 );

    /** We must have the combinatorial collector. **************************/
    switch( COLLECTOR( hdP ) )
    {
        case LEE_COLLECTOR:
        case COMBI_COLLECTOR:
        case COMBI2_COLLECTOR:
            break;
        default:
            return Error( "combinatorial collector not installed", 0, 0 );
            /* break; */
    }
    if ( GET_TYPE_BAG( hdW ) != T_SWORD && GET_TYPE_BAG( hdW ) != T_WORD )
        return Error( usage, 0, 0 );
    if ( ! IsNormedPcp( hdP, &hdW ) )
        return Error( "<w> must be a normed word of <P>", 0, 0 );

    /** Check central weights and set commutator. **************************/
    if ( ISID_AW( hdW ) )
        SET_COMMUTATORS( hdP, IND( i, j ), HD_IDENTITY( hdP ));
    else
    {
        ptWgt = CWEIGHTS( hdP );
        if ( ptWgt[ i ] + ptWgt[ j ] > ptWgt[ PTR_AW( hdW )[ 0 ] ] )
            return Error( "central weights do not add", 0, 0 );
        SET_COMMUTATORS( hdP, IND( i, j ), hdW);
    }
    return HdVoid;
}


/****************************************************************************
**
*F  FunAddCommPcp( <hdCall> ) . . . . . . . . . . . . . . . . .  'AddCommPcp'
**
**  'FunAddCommPcp' implements 'AddCommPcp( <P>, <i>, <j>, <w> )'
*/
Bag       FunAddCommPcp (Bag hdCall)
{
    Bag       hdW,  hdP,  hdI,  hdJ;
    Int            len,  i,  j,  * ptWgt;
    char            * usage = "AddCommPcp( <P>, <i>, <j>, <w> )";

    /** Evaluate and check the arguments. **********************************/
    if ( GET_SIZE_BAG( hdCall ) != 5 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdI = EVAL( PTR_BAG( hdCall )[ 2 ] );
    hdJ = EVAL( PTR_BAG( hdCall )[ 3 ] );
    hdW = EVAL( PTR_BAG( hdCall )[ 4 ] );
    if ( GET_TYPE_BAG(hdP) != T_PCPRES || GET_TYPE_BAG(hdI) != T_INT || GET_TYPE_BAG(hdJ) != T_INT )
        return Error( usage, 0, 0 );
    hdP = *PTR_BAG( hdP );
    len = NUMBER_OF_GENS( hdP );
    i   = HD_TO_INT( hdI ) - 1;
    j   = HD_TO_INT( hdJ ) - 1;
    if ( i < 0 || j < 0 )
        return Error( "generator number must be positive", 0, 0 );
    if ( i >= len || j >= len )
        return Error( "presenation has only %d generators", len, 0 );
    if ( i <= j )
        return Error( "<i> must be greater than <j>", 0, 0 );

    /** We must have the combinatorial collector. **************************/
    switch( COLLECTOR( hdP ) )
    {
        case LEE_COLLECTOR:
        case COMBI_COLLECTOR:
        case COMBI2_COLLECTOR:
            break;
        default:
            return Error( "combinatorial collector not installed", 0, 0 );
            /* break; */
    }
    if ( GET_TYPE_BAG( hdW ) != T_SWORD && GET_TYPE_BAG( hdW ) != T_WORD )
        return Error( usage, 0, 0 );
    if ( ! IsNormedPcp( hdP, &hdW ) )
        return Error( "<w> must be a normed word of <P>", 0, 0 );

    /** Check central weights and set commutator. **************************/
    if ( ! ISID_AW( hdW ) )
    {
        hdW   = SumAgWord( hdP, hdW, COMMUTATORS( hdP )[ IND( i, j ) ] );
        ptWgt = CWEIGHTS( hdP );
        if ( ptWgt[ i ] + ptWgt[ j ] > ptWgt[ PTR_AW( hdW )[ 0 ] ] )
            return Error( "central weights do not add", 0, 0 );
        SET_COMMUTATORS( hdP, IND( i, j ), hdW);
        Retype( hdW, T_SWORD );
    }
    return HdVoid;
}


/****************************************************************************
**
*F  FunSubtractCommPcp( <hdCall> )  . . . . . . . . . . . . 'SubtractCommPcp'
**
**  'FunSubtractCommPcp' implements 'SubtractCommPcp( <P>, <i>, <j>, <w> )'
*/
Bag       FunSubtractCommPcp (Bag hdCall)
{
    Bag       hdW,  hdP,  hdI,  hdJ;
    Int            len,  i,  j,  * ptWgt;
    char            * usage = "SubtractCommPcp( <P>, <i>, <j>, <w> )";

    /** Evaluate and check the arguments. **********************************/
    if ( GET_SIZE_BAG( hdCall ) != 5 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdI = EVAL( PTR_BAG( hdCall )[ 2 ] );
    hdJ = EVAL( PTR_BAG( hdCall )[ 3 ] );
    hdW = EVAL( PTR_BAG( hdCall )[ 4 ] );
    if ( GET_TYPE_BAG(hdP) != T_PCPRES || GET_TYPE_BAG(hdI) != T_INT || GET_TYPE_BAG(hdJ) != T_INT )
        return Error( usage, 0, 0 );
    hdP = *PTR_BAG( hdP );
    len = NUMBER_OF_GENS( hdP );
    i   = HD_TO_INT( hdI ) - 1;
    j   = HD_TO_INT( hdJ ) - 1;
    if ( i < 0 || j < 0 )
        return Error( "generator number must be positive", 0, 0 );
    if ( i >= len || j >= len )
        return Error( "presenation has only %d generators", len, 0 );
    if ( i <= j )
        return Error( "<i> must be greater than <j>", 0, 0 );

    /** We must have the combinatorial collector. **************************/
    switch( COLLECTOR( hdP ) )
    {
        case LEE_COLLECTOR:
        case COMBI_COLLECTOR:
        case COMBI2_COLLECTOR:
            break;
        default:
            return Error( "combinatorial collector not installed", 0, 0 );
            /* break; */
    }
    if ( GET_TYPE_BAG( hdW ) != T_SWORD && GET_TYPE_BAG( hdW ) != T_WORD )
        return Error( usage, 0, 0 );
    if ( ! IsNormedPcp( hdP, &hdW ) )
        return Error( "<w> must be a normed word of <P>", 0, 0 );

    /** Check central weights and set commutator. **************************/
    if ( ! ISID_AW( hdW ) )
    {
        hdW   = DifferenceAgWord( hdP, COMMUTATORS( hdP )[ IND(i,j) ], hdW );
        ptWgt = CWEIGHTS( hdP );
        if ( ptWgt[ i ] + ptWgt[ j ] > ptWgt[ PTR_AW( hdW )[ 0 ] ] )
            return Error( "central weights do not add", 0, 0 );
        SET_COMMUTATORS( hdP, IND( i, j ), hdW);
        Retype( hdW, T_SWORD );
    }
    return HdVoid;
}


/****************************************************************************
**
*F  FunDefinePowerPcp( <hdCall> ) . . . . . . . . . . . . .  'DefinePowerPcp'
**
**  'FunDefinePowerPcp' implements 'DefinePowerPcp( <P>, <i>, <w> )'
*/
Bag       FunDefinePowerPcp (Bag hdCall)
{
    Bag       hdW,  hdP,  hdI;
    Int            len,  i,  * ptWgt;
    char            * usage = "DefinePowerPcp( <P>, <i>, <w> )";

    /** Evaluate and check the arguments. **********************************/
    if ( GET_SIZE_BAG( hdCall ) != 4 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdI = EVAL( PTR_BAG( hdCall )[ 2 ] );
    hdW = EVAL( PTR_BAG( hdCall )[ 3 ] );
    if ( GET_TYPE_BAG( hdP ) != T_PCPRES || GET_TYPE_BAG( hdI ) != T_INT )
        return Error( usage, 0, 0 );
    hdP = *PTR_BAG( hdP );
    len = NUMBER_OF_GENS( hdP );
    i   = HD_TO_INT( hdI ) - 1;
    if ( i < 0 )
        return Error( "generator number must be positive", 0, 0 );
    if ( i >= len )
        return Error( "presenation has only %d generators", len, 0 );
    if ( GET_TYPE_BAG( hdW ) != T_SWORD && GET_TYPE_BAG( hdW ) != T_WORD )
        return Error( usage, 0, 0 );
    if ( ! IsNormedPcp( hdP, &hdW ) )
        return Error( "<w> must be a normed word of <P>", 0, 0 );

    /** Check collector depend conditions. *********************************/
    if ( ISID_AW( hdW ) )
        SET_POWERS( hdP, i, HD_IDENTITY( hdP ));
    else
    {
        switch( COLLECTOR( hdP ) )
        {
            case LEE_COLLECTOR:
            case COMBI_COLLECTOR:
            case COMBI2_COLLECTOR:
                ptWgt = CWEIGHTS( hdP );
                if ( ptWgt[ i ] >= ptWgt[ PTR_AW( hdW )[ 0 ] ] )
                    return Error( "central weight does not grow", 0, 0 );
                break;
            case SINGLE_COLLECTOR:
                if ( i >= PTR_AW( hdW )[ 0 ] )
                    return Error( "depth does not grow", 0, 0 );
                break;
        }
        SET_POWERS( hdP, i, hdW);
    }
    return HdVoid;
}


/****************************************************************************
**
*F  FunAddPowerPcp( <hdCall> )  . . . . . . . . . . . . . . . . 'AddPowerPcp'
**
**  'FunAddPowerPcp' implements 'AddPowerPcp( <P>, <i>, <w> )'
*/
Bag       FunAddPowerPcp (Bag hdCall)
{
    Bag       hdW,  hdP,  hdI;
    Int            len,  i,  * ptWgt;
    char            * usage = "AddPowerPcp( <P>, <i>, <w> )";

    /** Evaluate and check the arguments. **********************************/
    if ( GET_SIZE_BAG( hdCall ) != 4 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdI = EVAL( PTR_BAG( hdCall )[ 2 ] );
    hdW = EVAL( PTR_BAG( hdCall )[ 3 ] );
    if ( GET_TYPE_BAG( hdP ) != T_PCPRES || GET_TYPE_BAG( hdI ) != T_INT )
        return Error( usage, 0, 0 );
    hdP = *PTR_BAG( hdP );
    len = NUMBER_OF_GENS( hdP );
    i   = HD_TO_INT( hdI ) - 1;
    if ( i < 0 )
        return Error( "generator number must be positive", 0, 0 );
    if ( i >= len )
        return Error( "presenation has only %d generators", len, 0 );
    if ( GET_TYPE_BAG( hdW ) != T_SWORD && GET_TYPE_BAG( hdW ) != T_WORD )
        return Error( usage, 0, 0 );
    if ( ! IsNormedPcp( hdP, &hdW ) )
        return Error( "<w> must be a normed word of <P>", 0, 0 );

    /** Check collector depend conditions. *********************************/
    if ( ! ISID_AW( hdW ) )
    {
        hdW = SumAgWord( hdP, hdW, POWERS( hdP )[ i ] );
        switch( COLLECTOR( hdP ) )
        {
            case LEE_COLLECTOR:
            case COMBI_COLLECTOR:
            case COMBI2_COLLECTOR:
                ptWgt = CWEIGHTS( hdP );
                if ( ptWgt[ i ] >= ptWgt[ PTR_AW( hdW )[ 0 ] ] )
                    return Error( "central weight does not grow", 0, 0 );
                break;
            case SINGLE_COLLECTOR:
                if ( i >= PTR_AW( hdW )[ 0 ] )
                    return Error( "depth does not grow", 0, 0 );
                break;
        }
        Retype( hdW, T_SWORD );
        SET_POWERS( hdP, i, hdW);
    }
    return HdVoid;
}


/****************************************************************************
**
*F  FunSubtractPowerPcp( <hdCall> ) . . . . . . . . . . .  'SubtractPowerPcp'
**
**  'FunSubtractPowerPcp' implements 'SubtractPowerPcp( <P>, <i>, <w> )'
*/
Bag       FunSubtractPowerPcp (Bag hdCall)
{
    Bag       hdW,  hdP,  hdI;
    Int            len,  i,  * ptWgt;
    char            * usage = "SubtractPowerPcp( <P>, <i>, <w> )";

    /** Evaluate and check the arguments. **********************************/
    if ( GET_SIZE_BAG( hdCall ) != 4 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdI = EVAL( PTR_BAG( hdCall )[ 2 ] );
    hdW = EVAL( PTR_BAG( hdCall )[ 3 ] );
    if ( GET_TYPE_BAG( hdP ) != T_PCPRES || GET_TYPE_BAG( hdI ) != T_INT )
        return Error( usage, 0, 0 );
    hdP = *PTR_BAG( hdP );
    len = NUMBER_OF_GENS( hdP );
    i   = HD_TO_INT( hdI ) - 1;
    if ( i < 0 )
        return Error( "generator number must be positive", 0, 0 );
    if ( i >= len )
        return Error( "presenation has only %d generators", len, 0 );
    if ( GET_TYPE_BAG( hdW ) != T_SWORD && GET_TYPE_BAG( hdW ) != T_WORD )
        return Error( usage, 0, 0 );
    if ( ! IsNormedPcp( hdP, &hdW ) )
        return Error( "<w> must be a normed word of <P>", 0, 0 );

    /** Check collector depend conditions. *********************************/
    if ( ! ISID_AW( hdW ) )
    {
        hdW = DifferenceAgWord( hdP, POWERS( hdP )[ i ], hdW );
        switch( COLLECTOR( hdP ) )
        {
            case LEE_COLLECTOR:
            case COMBI_COLLECTOR:
            case COMBI2_COLLECTOR:
                ptWgt = CWEIGHTS( hdP );
                if ( ptWgt[ i ] >= ptWgt[ PTR_AW( hdW )[ 0 ] ] )
                    return Error( "central weight does not grow", 0, 0 );
                break;
            case SINGLE_COLLECTOR:
                if ( i >= PTR_AW( hdW )[ 0 ] )
                    return Error( "depth does not grow", 0, 0 );
                break;
        }
        Retype( hdW, T_SWORD );
        SET_POWERS( hdP, i, hdW);
    }
    return HdVoid;
}


/****************************************************************************
**
*F  ShrinkSwords( <hdP>, <hdList>, <hdMap> )  . . . . . . . . . . . . . local
*/
void            ShrinkSwords(Bag hdP, Bag hdL, Bag hdM)
{
    Bag       hdG,  hdT;
    TypSword        * ptG,  * ptH;
    Int            i,  j,  new,  * ptM;

    for ( i = LEN_LIST( hdL );  i > 0;  i-- )
    {
        ptM = (Int*) PTR_BAG( hdM );
        hdG = ELM_PLIST( hdL, i );

        /** Get the number of nontrivial entries ***************************/
        ptG = PTR_AW( hdG );
        if ( *ptG == -1 )
        {
            SET_ELM_PLIST( hdL, i, HD_IDENTITY( hdP ) );
            continue;
        }
        new = 0;
        while ( *ptG != -1 )
        {
            if ( ptM[ ptG[0] ] != -1 )
                new++;
            ptG += 2;
        }

        /** Copy the agword,  remap the generator numbers. *****************/
        hdT = NewBag( T_SWORD, SIZE_HD + ( 2 * new + 1 ) * SIZE_SWORD );
        SET_BAG( hdT, 0, hdP);
        SET_ELM_PLIST( hdL, i, hdT );
        ptH = PTR_AW( hdT );
        ptG = PTR_AW( hdG );
        ptM = (Int*) PTR_BAG( hdM );
        while ( *ptG != -1 )
        {
            j = ptM[ ptG[0] ];
            if ( j != -1 )
            {
                ptH[0] = j;
                ptH[1] = ptG[1];
                ptH   += 2;
            }
            ptG += 2;
        }
        ptH[0] = -1;
    }
}


/****************************************************************************
**
*F  FunShrinkPcp( <hdCall> )  . . . . . . . . . . . . .  internal 'ShrinkPcp'
**
**  'FunShrinkPcp' implements 'ShrinkPcp( <P>, <L> )'
**
**  'ShrinkPcp' removes the generators given in <L> from <P>.  As  this would
**  change every existing sword with this presentation,  we  must remove this
**  presentation from every sword.
*/
Bag       FunShrinkPcp (Bag hdCall)
{
    Bag       hdP,  hdC,  hdL,  hdT,  hdG;
    Bag       * ptT,  * ptG,  * ptC;
    Int            * piT,  * piG,  * ptL;
    Int            i,  j,  i0,  j0,  new,  old,  len;
    char            * usage = "usage: ShrinkPcp( <P>, <L> )";

    /** Evaluate and check the arguments ***********************************/
    if ( GET_SIZE_BAG( hdCall ) != 3 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdC = EVAL( PTR_BAG( hdCall )[ 2 ] );
    if ( GET_TYPE_BAG( hdP ) != T_PCPRES || ! IsVector( hdC ) )
        return Error( usage, 0, 0 );
    if ( LEN_LIST( hdC ) == 0 )
        return HdVoid;

    i = COLLECTOR( *PTR_BAG( hdP ) );
    if (i != COMBI2_COLLECTOR && i != COMBI_COLLECTOR && i != LEE_COLLECTOR)
        return Error( "ShrinkPcp: not ready!", 0, 0 );

    /** Check list of generator numbers. ***********************************/
    len = LEN_LIST( hdC );
    old = NUMBER_OF_GENS( *PTR_BAG( hdP ) );
    hdL = NewBag( T_STRING, old * sizeof( Int ) );
    ptC = PTR_BAG( hdC ) + 1;
    ptL = (Int*) PTR_BAG( hdL );
    if ( old <= len )
        return Error( "cannot delete every generators of <P>", 0, 0 );
    for ( i = 0;  i < len;  i++ )
    {
        j = HD_TO_INT( ptC[ i ] ) - 1;
        if ( GET_TYPE_BAG( ptC[ i ] ) != T_INT || 0 > j || j >= old )
            return Error( "illegal generator number %d", j + 1, 0 );
        if ( ptL[ j ] != 0 )
            return Error( "duplicate generator number %d", j, 0 );
        ptL[ j ] = -1;
    }
    for ( i = 0, j = 0;  i < old;  i++ )
    {
        if ( ptL[ i ] == 0 )
            ptL[ i ] = j++;
    }

    /** Now reset all old swords.  Change old presentation into genlist. ***/
    hdG = *PTR_BAG( hdP );
    hdT = NewBag( T_AGGRP, GET_SIZE_BAG( hdG ) );
    ptG = PTR_BAG( hdG );
    ptT = PTR_BAG( hdT );
    for ( i = GET_SIZE_BAG( hdG ) / SIZE_HD - 1;  i >= 0;  i-- )
        *ptT++ = *ptG++;
    SET_BAG( hdP, 0, hdT);
    Resize( hdG, GET_SIZE_BAG( HD_WORDS( hdT ) ) );
    Retype( hdG, GET_TYPE_BAG( HD_WORDS( hdT ) ) );
    ptG = PTR_BAG( hdG );
    ptT = PTR_BAG( HD_WORDS( hdT ) );
    for ( i = GET_SIZE_BAG( hdG ) / SIZE_HD - 1;  i >= 0;  i-- )
        *ptG++ = *ptT++;
    hdP = hdT;

    /** Construct the new generators ***************************************/
    new = old - len;
    SET_BAG( hdP, NR_NUMBER_OF_GENS, INT_TO_HD( new ));
    SetGeneratorsAgGroup( hdP );
    for ( i = new - 1;  i >= 0;  i-- )
        Retype( GENERATORS( hdP )[ i ], T_SWORD );
    Retype( HD_IDENTITY( hdP ), T_SWORD );

    /** Shrink the abstract generators. ************************************/
    hdT = NewBag( T_LIST, ( new + 1 ) * SIZE_HD );
    ptT = PTR_BAG( hdT ) + 1;
    ptG = WORDS( hdP );
    ptL = (Int*) PTR_BAG( hdL );
    ptT[ -1 ] = INT_TO_HD( new );
    for ( i = old - 1;  i >= 0;  i-- )
    {
        j = ptL[ i ];
        if ( j != -1 )
            ptT[ j ] = ptG[ i ];
    }
    SET_BAG( hdP, NR_WORDS, hdT);

    /** Shrink the indices. ************************************************/
    hdT = NewBag( T_INTPOS, new * sizeof( Int ) );
    piT = (Int*) PTR_BAG( hdT );
    piG = INDICES( hdP );
    ptL = (Int*) PTR_BAG( hdL );
    for ( i = old - 1;  i >= 0;  i-- )
    {
        j = ptL[ i ];
        if ( j != -1 )
            piT[ j ] = piG[ i ];
    }
    SET_BAG( hdP, NR_INDICES, hdT);

    /** Shrink the powers,  do not use an addtional list. ******************/
    ptT = POWERS( hdP );
    ptG = POWERS( hdP );
    ptL = (Int*) PTR_BAG( hdL );
    ptT[ -1 ] = INT_TO_HD( new );
    for ( i = 0;  i < old;  i++ )
    {
        j = ptL[ i ];
        if ( j != -1 )
            ptT[ j ] = ptG[ i ];
    }
    Resize( HD_POWERS( hdP ), ( new + 1 ) * SIZE_HD );

    /** Shrink the commutators,  without an additional list. ***************/
    ptT = COMMUTATORS( hdP );
    ptG = COMMUTATORS( hdP );
    ptL = (Int*) PTR_BAG( hdL );
    ptT[ -1 ] = INT_TO_HD( new * ( new - 1 ) / 2 );
    for ( i = 1;  i < old;  i++ )
    {
        i0 = ptL[ i ];
        if ( i0 != -1 )
        {
            for ( j = 0;  j < i;  j++ )
            {
                j0 = ptL[ j ];
                if ( j0 != -1 )
                    ptT[ IND( i0, j0 ) ] = ptG[ IND( i, j ) ];
            }
        }
    }
    Resize( HD_COMMUTATORS( hdP ), ( new * (new - 1) / 2 + 1 ) * SIZE_HD );

    /** Shrink and renumber swords themselves. *****************************/
    ShrinkSwords( hdP, HD_POWERS( hdP ),      hdL );
    ShrinkSwords( hdP, HD_COMMUTATORS( hdP ), hdL );

    /** Shrink save and collect exponent vectors. **************************/
    Resize( HD_SAVE_EXPONENTS( hdP ),      new * SIZE_EXP );
    Resize( HD_COLLECT_EXPONENTS( hdP ),   new * SIZE_EXP );
    Resize( HD_COLLECT_EXPONENTS_2( hdP ), new * SIZE_EXP );

    /** Collector depend shrinking. ****************************************/
    switch ( COLLECTOR( hdP ) )
    {
        case LEE_COLLECTOR:
        case COMBI_COLLECTOR:
        case COMBI2_COLLECTOR:
            hdT = NewBag( T_INTPOS, new * sizeof( Int ) );
            piT = (Int*) PTR_BAG( hdT );
            piG = CWEIGHTS( hdP );
            ptL = (Int*) PTR_BAG( hdL );
            for ( i = old - 1;  i >= 0;  i-- )
            {
                j = ptL[ i ];
                if ( j != -1 )
                    piT[ j ] = piG[ i ];
            }
            SET_BAG( hdP, NR_CWEIGHTS, hdT);
            hdT = NewBag( T_INTPOS, ( new + 1 ) * sizeof( Int ) );
            piT = (Int*) PTR_BAG( hdT );
            piG = CWEIGHTS( hdP );
            piT[ 0 ] = 1;
            for ( i = 0;  i < new;  i++ )
            {
                if ( piG[ i ] > piT[ 0 ] )
                {
                    piT[ piT[ 0 ] ] = i - 1;
                    piT[ 0 ]++;
                }
            }
            piT[ piT[ 0 ] ] = i - 1;
            Resize( hdT, ( piT[ 0 ] + 1 ) * sizeof( Int ) );
            SET_BAG( hdP, NR_CSERIES, hdT);
            break;
        default:
            return Error( "ShrinkPcp: not ready!", 0, 0 );
    }

    return HdVoid;
}


/****************************************************************************
**
*F  FunTriangleIndex( <hdCall> )  . . . . . . . . .  internal 'TriangleIndex'
**
**  'FunTriangleIndex' implements 'TriangleIndex( <i>, <j> )'
**
**  'TriangleIndex'  exports the macro  'IND' used to address a commuator  in
**  the aggroup record field <COMMUTATOR>.
*/
Bag       FunTriangleIndex(Bag hdCall)
{
    Bag       hdI, hdJ;
    Int            i,   j;
    char            * usage = "usage: TriangleIndex( <i>, <j> )";

    /** Evaluate and check the arguments ***********************************/
    if ( GET_SIZE_BAG( hdCall ) != 3 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdI = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdJ = EVAL( PTR_BAG( hdCall )[ 2 ] );
    if ( GET_TYPE_BAG( hdI ) != T_INT || GET_TYPE_BAG( hdJ ) != T_INT )
        return Error( usage, 0, 0 );

    /** return *************************************************************/
    i = HD_TO_INT( hdI );
    j = HD_TO_INT( hdJ );
    return INT_TO_HD( ( i - 1 ) * ( i - 2 ) / 2 + j );
}


/*--------------------------------------------------------------------------\
|                           collector operations                            |
\--------------------------------------------------------------------------*/


/****************************************************************************
**
*F  NormalWordPcp( <P>, <g> ) . . . . . . . . . . . . . . . .  collected word
**
**  Return either the collected word or 'HdFalse'.
*/
Bag   NormalWordPcp (Bag hdP, Bag hdG)
{
    Bag           hdQ,  hdR;
    Bag           * ptW;
    TypSword            * ptG;
    Int                i;

    if ( GET_TYPE_BAG( hdG ) == T_SWORD )
    {
        hdQ = *PTR_BAG( hdG );
        if ( hdQ == hdP )
            return hdG;
        if ( GET_TYPE_BAG( hdQ ) == T_AGGRP )
            hdQ = HD_WORDS( hdQ );
        hdG = SwordSword( HD_WORDS( hdP ), hdG );
    }
    else if ( hdG == HdIdWord )
        return HD_IDENTITY( hdP );
    else
        hdG = SwordWord( HD_WORDS( hdP ), hdG );
    if ( hdG == HdFalse )
        return hdG;

    /*N One should watch for long runs and invert some all at once. ********/
    ptW = GENERATORS( hdP );
    ptG = PTR_AW( hdG );
    hdR = HD_IDENTITY( hdP );

    /** Run through the word and collect. **********************************/
    i = 0;
    while ( *ptG != -1 )
    {
        hdR = ProdAg( hdR, PowAgI( ptW[ ptG[0] ], INT_TO_HD( ptG[1] ) ) );
        i  += 2;
        ptG = &( PTR_AW( hdG )[ i ] );
        ptW = GENERATORS( hdP );
    }
    Retype( hdR, T_SWORD );

    return hdR;
}


/****************************************************************************
**
*F  FunNormalWordPcp( <hdCall> )  . . . . . . . . .  internal 'NormalWordPcp'
**
**  'FunNormalWordPcp' implements 'NormalWordPcp( <P>, <g> )'
*/
Bag       FunNormalWordPcp (Bag hdCall)
{
    Bag       hdG,  hdP;
    char            * usage = "NormalWordPcp( <P>, <g> )";

    /** Evaluate and check the arguments ***********************************/
    if ( GET_SIZE_BAG( hdCall ) != 3 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdG = EVAL( PTR_BAG( hdCall )[ 2 ] );
    if ( GET_TYPE_BAG(hdP)!=T_PCPRES || (GET_TYPE_BAG(hdG)!=T_WORD && GET_TYPE_BAG(hdG)!=T_SWORD) )
        return Error( usage, 0, 0 );

    hdP = *PTR_BAG( hdP );
    hdG = NormalWordPcp( hdP, hdG );
    if ( hdG == HdFalse )
        return Error( "<g> must be an element of <P>", 0, 0 );
    return hdG;
}


/****************************************************************************
**
*F  FunProductPcp( <hdCall> ) . . . . . . . . . . . . . internal 'ProductPcp'
**
**  'FunProductPcp' implements 'ProductPcp( <P>, <a>, <b> )'
**
**  'ProductPcp' returns the product of the two swords <a>  and <b> using the
**  presentation <P>.
*/
Bag       FunProductPcp (Bag hdCall)
{
    Bag           hdP,  hdA,  hdB,  hdR;
    Int                len,  a,  b;
    char                * usage = "usage: ProductPcp( <P>, <a>, <b> )";

    /** Evaluate and check the arguments ***********************************/
    if ( GET_SIZE_BAG( hdCall ) != 4 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    if ( GET_TYPE_BAG( hdP ) != T_PCPRES )
        return Error( usage, 0, 0 );
    hdP = *PTR_BAG( hdP );

    hdA = EVAL( PTR_BAG( hdCall )[ 2 ] );
    hdB = EVAL( PTR_BAG( hdCall )[ 3 ] );
    len = NUMBER_OF_GENS( hdP );

    /** Convert <a> and <b> into words of <P>. *****************************/
    if ( GET_TYPE_BAG( hdA ) == T_INT )
    {
        a = HD_TO_INT( hdA ) - 1;
        if ( a < 0 )
            return Error( "generator number must be positive", 0, 0 );
        if ( a >= len )
            return Error("presentation has only %d generators", len, 0);
        hdA = GENERATORS( hdP )[ a ];
    }
    else if ( GET_TYPE_BAG( hdA ) == T_WORD || GET_TYPE_BAG( hdA ) == T_SWORD )
    {
        hdA = NormalWordPcp( hdP, hdA );
        if ( hdA == HdFalse )
            return Error( "<a> must be a word of <P>", 0, 0 );
    }
    else
        return Error( usage, 0, 0 );
    if ( GET_TYPE_BAG( hdB ) == T_INT )
    {
        b = HD_TO_INT( hdB ) - 1;
        if ( b < 0 )
            return Error( "generator number must be positive", 0, 0 );
        if ( b >= len )
            return Error("presentation has only %d generators", len, 0);
        hdB = GENERATORS( hdP )[ b ];
    }
    else if ( GET_TYPE_BAG( hdB ) == T_WORD || GET_TYPE_BAG( hdB ) == T_SWORD )
    {
        hdB = NormalWordPcp( hdP, hdB );
        if ( hdB == HdFalse )
            return Error( "<b> must be a word of <P>", 0, 0 );
    }
    else
        return Error( usage, 0, 0 );

    hdR = ProdAg( hdA, hdB );
    Retype( hdR, T_SWORD );
    return hdR;
}


/****************************************************************************
**
*F  FunLeftQuotienPcp( <hdCall> ) . . . . . . . .  internal 'LeftQuotientPcp'
**
**  'FunLeftQuotientPcp' implements 'LeftQuotientPcp( <P>, <a>, <b> )'
**
**  'LeftQuotientPcp' returns the  left quotient  of the two swords <a>  and
**  <b> using the presentation <P>.
*/
Bag       FunLeftQuotientPcp (Bag hdCall)
{
    Bag           hdP,  hdA,  hdB,  hdR;
    Int                len,  a,  b;
    char                * usage = "usage: LeftQuotientPcp( <P>, <a>, <b> )";

    /** Evaluate and check the arguments ***********************************/
    if ( GET_SIZE_BAG( hdCall ) != 4 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    if ( GET_TYPE_BAG( hdP ) != T_PCPRES )
        return Error( usage, 0, 0 );
    hdP = *PTR_BAG( hdP );

    hdA = EVAL( PTR_BAG( hdCall )[ 2 ] );
    hdB = EVAL( PTR_BAG( hdCall )[ 3 ] );
    len = NUMBER_OF_GENS( hdP );

    /** Convert <a> and <b> into words of <P>. *****************************/
    if ( GET_TYPE_BAG( hdA ) == T_INT )
    {
        a = HD_TO_INT( hdA ) - 1;
        if ( a < 0 )
            return Error( "generator number must be positive", 0, 0 );
        if ( a >= len )
            return Error("presentation has only %d generators", len, 0);
        hdA = GENERATORS( hdP )[ a ];
    }
    else if ( GET_TYPE_BAG( hdA ) == T_WORD || GET_TYPE_BAG( hdA ) == T_SWORD )
    {
        hdA = NormalWordPcp( hdP, hdA );
        if ( hdA == HdFalse )
            return Error( "<a> must be a word of <P>", 0, 0 );
    }
    else
        return Error( usage, 0, 0 );
    if ( GET_TYPE_BAG( hdB ) == T_INT )
    {
        b = HD_TO_INT( hdB ) - 1;
        if ( b < 0 )
            return Error( "generator number must be positive", 0, 0 );
        if ( b >= len )
            return Error("presentation has only %d generators", len, 0);
        hdB = GENERATORS( hdP )[ b ];
    }
    else if ( GET_TYPE_BAG( hdB ) == T_WORD || GET_TYPE_BAG( hdB ) == T_SWORD )
    {
        hdB = NormalWordPcp( hdP, hdB );
        if ( hdB == HdFalse )
            return Error( "<b> must be a word of <P>", 0, 0 );
    }
    else
        return Error( usage, 0, 0 );

    hdR = ModAg( hdA, hdB );
    Retype( hdR, T_SWORD );
    return hdR;
}


/****************************************************************************
**
*F  FunQuotientPcp( <hdCall> )  . . . . . . . . . . .  internal 'QuotientPcp'
**
**  'FunQuotientPcp' implements 'QuotientPcp( <P>, <a>, <b> )'
**
**  'QuotientPcp' returns the quotient of the two swords  <a>  and <b>  using
**  the presentation <P>.
*/
Bag       FunQuotientPcp (Bag hdCall)
{
    Bag           hdP,  hdA,  hdB,  hdR;
    Int                len,  a,  b;
    char                * usage = "usage: QuotientPcp( <P>, <a>, <b> )";

    /** Evaluate and check the arguments ***********************************/
    if ( GET_SIZE_BAG( hdCall ) != 4 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    if ( GET_TYPE_BAG( hdP ) != T_PCPRES )
        return Error( usage, 0, 0 );
    hdP = *PTR_BAG( hdP );

    hdA = EVAL( PTR_BAG( hdCall )[ 2 ] );
    hdB = EVAL( PTR_BAG( hdCall )[ 3 ] );
    len = NUMBER_OF_GENS( hdP );

    /** Convert <a> and <b> into words of <P>. *****************************/
    if ( GET_TYPE_BAG( hdA ) == T_INT )
    {
        a = HD_TO_INT( hdA ) - 1;
        if ( a < 0 )
            return Error( "generator number must be positive", 0, 0 );
        if ( a >= len )
            return Error("presentation has only %d generators", len, 0);
        hdA = GENERATORS( hdP )[ a ];
    }
    else if ( GET_TYPE_BAG( hdA ) == T_WORD || GET_TYPE_BAG( hdA ) == T_SWORD )
    {
        hdA = NormalWordPcp( hdP, hdA );
        if ( hdA == HdFalse )
            return Error( "<a> must be a word of <P>", 0, 0 );
    }
    else
        return Error( usage, 0, 0 );
    if ( GET_TYPE_BAG( hdB ) == T_INT )
    {
        b = HD_TO_INT( hdB ) - 1;
        if ( b < 0 )
            return Error( "generator number must be positive", 0, 0 );
        if ( b >= len )
            return Error("presentation has only %d generators", len, 0);
        hdB = GENERATORS( hdP )[ b ];
    }
    else if ( GET_TYPE_BAG( hdB ) == T_WORD || GET_TYPE_BAG( hdB ) == T_SWORD )
    {
        hdB = NormalWordPcp( hdP, hdB );
        if ( hdB == HdFalse )
            return Error( "<b> must be a word of <P>", 0, 0 );
    }
    else
        return Error( usage, 0, 0 );

    hdR = QuoAg( hdA, hdB );
    Retype( hdR, T_SWORD );
    return hdR;
}


/****************************************************************************
**
*F  FunCommPcp( <hdCall> )  . . . . . . . . . . . . . . .  internal 'CommPcp'
**
**  'FunCommPcp' implements 'CommPcp( <P>, <a>, <b> )'
**
**  'CommPcp' returns the  commutator  of  the two  swords  <a> and <b> using
**  the presentation <P>.
**
**  Note that if the combinatorial collector is installed,  we  can  use  the
**  presentation in order to compute the commutator of two generators.
*/
Bag       FunCommPcp (Bag hdCall)
{
    Bag           hdP,  hdA,  hdB,  hdR;
    Int                col,  a,  b,  nrA,  nrB,  len;
    char                * usage = "usage: CommPcp( <P>, <a>, <b> )";

    /** Evaluate and check the arguments ***********************************/
    if ( GET_SIZE_BAG( hdCall ) != 4 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    if ( GET_TYPE_BAG( hdP ) != T_PCPRES )
        return Error( usage, 0, 0 );
    hdP = *PTR_BAG( hdP );

    hdA = EVAL( PTR_BAG( hdCall )[ 2 ] );
    hdB = EVAL( PTR_BAG( hdCall )[ 3 ] );
    col = COLLECTOR( hdP );
    len = NUMBER_OF_GENS( hdP );

    /** If combinatorial collector and integers, returns lhs of relation. **/
    if ( col==COMBI_COLLECTOR || col==COMBI2_COLLECTOR || col==LEE_COLLECTOR)
    {
        if ( GET_TYPE_BAG( hdA ) == T_INT && GET_TYPE_BAG( hdB ) == T_INT )
        {
            a = HD_TO_INT( hdA ) - 1;
            b = HD_TO_INT( hdB ) - 1;
            if ( a < 0 || b < 0 )
                return Error( "generator number must be positive", 0, 0 );
            if ( a >= len || b >= len )
                return Error("presentation has only %d generators", len, 0);
            if ( a == b )
                return HD_IDENTITY( hdP );
            else if ( a > b )
                return COMMUTATORS( hdP )[ IND( a, b ) ];
            else
            {
                hdR = COMMUTATORS( hdP )[ IND( b, a ) ];
                hdR = PowAgI( hdR, INT_TO_HD( -1 ) );
                Retype( hdR, T_SWORD );
                return hdR;
            }
        }
    }

    /** Convert <a> and <b> into words of <P>. *****************************/
    if ( GET_TYPE_BAG( hdA ) == T_INT )
    {
        a = HD_TO_INT( hdA ) - 1;
        if ( a < 0 )
            return Error( "generator number must be positive", 0, 0 );
        if ( a >= len )
            return Error("presentation has only %d generators", len, 0);
        hdA = GENERATORS( hdP )[ a ];
    }
    else if ( GET_TYPE_BAG( hdA ) == T_WORD || GET_TYPE_BAG( hdA ) == T_SWORD )
    {
        hdA = NormalWordPcp( hdP, hdA );
        if ( hdA == HdFalse )
            return Error( "<a> must be a word of <P>", 0, 0 );
    }
    else
        return Error( usage, 0, 0 );
    if ( GET_TYPE_BAG( hdB ) == T_INT )
    {
        b = HD_TO_INT( hdB ) - 1;
        if ( b < 0 )
            return Error( "generator number must be positive", 0, 0 );
        if ( b >= len )
            return Error("presentation has only %d generators", len, 0);
        hdB = GENERATORS( hdP )[ b ];
    }
    else if ( GET_TYPE_BAG( hdB ) == T_WORD || GET_TYPE_BAG( hdB ) == T_SWORD )
    {
        hdB = NormalWordPcp( hdP, hdB );
        if ( hdB == HdFalse )
            return Error( "<b> must be a word of <P>", 0, 0 );
    }
    else
        return Error( usage, 0, 0 );

    /** If <a>/<b> are gens and we have a combi-coll, use relations. *******/
    nrA = LEN_AW( hdA );
    if ( nrA == 0 )
        return HD_IDENTITY( hdP );
    nrB = LEN_AW( hdB );
    if ( nrB == 0 )
        return HD_IDENTITY( hdP );

    if (  (    col == COMBI_COLLECTOR
            || col == COMBI2_COLLECTOR
            || col == LEE_COLLECTOR )
         && nrA == 1
         && nrB == 1
         && PTR_AW( hdA )[ 1 ] == 1
         && PTR_AW( hdB )[ 1 ] == 1 )
    {
        a = PTR_AW( hdA )[ 0 ];
        b = PTR_AW( hdB )[ 0 ];
        if ( a == b )
            return HD_IDENTITY( hdP );
        if ( a > b )
            return COMMUTATORS( hdP )[ IND( a, b ) ];
        else
        {
            hdR = COMMUTATORS( hdP )[ IND( b, a ) ];
            hdR = PowAgI( hdR, INT_TO_HD( -1 ) );
            Retype( hdR, T_SWORD );
            return hdR;
        }
    }

    /** Solve the equation  <hdB> * <hdA> * x = <hdA> * <hdB>. *************/
    hdR = AgSolution2( hdB, hdA, hdA, hdB );
    Retype( hdR, T_SWORD );
    return hdR;
}


/****************************************************************************
**
*F  FunConjugatePcp( <hdCall> ) . . . . . . . . . . . internal 'ConjugatePcp'
**
**  'FunConjugatePcp' implements 'ConjugatePcp( <P>, <a>, <b>)'
**
**  'ConjugatePcp' returns the conjugate <a>^<b> using the presentation <P>.
**
**  Note that if the single collector is installed,  we  use the presentation
**  in order to compute the conjugate of two generators.
*/
Bag       FunConjugatePcp (Bag hdCall)
{
    Bag           hdP,  hdA,  hdB,  hdR;
    Int                col,  a,  b,  nrA,  nrB, len;
    char                * usage = "usage: ConjugatePcp( <P>, <a>, <b> )";

    /** Evaluate and check the arguments ***********************************/
    if ( GET_SIZE_BAG( hdCall ) != 4 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    if ( GET_TYPE_BAG( hdP ) != T_PCPRES )
        return Error( usage, 0, 0 );
    hdP = *PTR_BAG( hdP );

    hdA = EVAL( PTR_BAG( hdCall )[ 2 ] );
    hdB = EVAL( PTR_BAG( hdCall )[ 3 ] );
    col = COLLECTOR( hdP );
    len = NUMBER_OF_GENS( hdP );

    /** If combinatorial collector and integers, returns lhs of relation. **/
    if ( col == SINGLE_COLLECTOR )
    {
        if ( GET_TYPE_BAG( hdA ) == T_INT && GET_TYPE_BAG( hdB ) == T_INT )
        {
            a = HD_TO_INT( hdA ) - 1;
            b = HD_TO_INT( hdB ) - 1;
            if ( a < 0 || b < 0 )
                return Error( "generator number must be positive", 0, 0 );
            if ( a >= len || b >= len )
                return Error("presentation has only %d generators", len, 0);
            if ( a == b )
                return GENERATORS( hdP )[ a ];
            else if ( a > b )
                return CONJUGATES( hdP )[ IND( a, b ) ];
            else
            {
                hdA = GENERATORS( hdP )[ a ];
                hdB = GENERATORS( hdP )[ b ];
                hdR = AgSolution2( hdB, HD_IDENTITY( hdP ), hdA, hdB );
                Retype( hdR, T_SWORD );
                return hdR;
            }
        }
    }

    /** Convert <a> and <b> into words of <P>. *****************************/
    if ( GET_TYPE_BAG( hdA ) == T_INT )
    {
        a = HD_TO_INT( hdA ) - 1;
        if ( a < 0 )
            return Error( "generator number must be positive", 0, 0 );
        if ( a >= len )
            return Error("presentation has only %d generators", len, 0);
        hdA = GENERATORS( hdP )[ a ];
    }
    else if ( GET_TYPE_BAG( hdA ) == T_WORD || GET_TYPE_BAG( hdA ) == T_SWORD )
    {
        hdA = NormalWordPcp( hdP, hdA );
        if ( hdA == HdFalse )
            return Error( "<a> must be a word of <P>", 0, 0 );
    }
    else
        return Error( usage, 0, 0 );
    if ( GET_TYPE_BAG( hdB ) == T_INT )
    {
        b = HD_TO_INT( hdB ) - 1;
        if ( b < 0 )
            return Error( "generator number must be positive", 0, 0 );
        if ( b >= len )
            return Error("presentation has only %d generators", len, 0);
        hdB = GENERATORS( hdP )[ b ];
    }
    else if ( GET_TYPE_BAG( hdB ) == T_WORD || GET_TYPE_BAG( hdB ) == T_SWORD )
    {
        hdB = NormalWordPcp( hdP, hdB );
        if ( hdB == HdFalse )
            return Error( "<b> must be a word of <P>", 0, 0 );
    }
    else
        return Error( usage, 0, 0 );

    /** If <a>/<b> are gens and we have a single-coll, use relations. ******/
    nrA = LEN_AW( hdA );
    nrB = LEN_AW( hdB );
    if ( nrA == 0 || nrB == 0 )
        return hdA;

    if (    col == SINGLE_COLLECTOR
         && nrA == 1
         && nrB == 1
         && PTR_AW( hdA )[ 1 ] == 1
         && PTR_AW( hdB )[ 1 ] == 1 )
    {
        a = PTR_AW( hdA )[ 0 ];
        b = PTR_AW( hdB )[ 0 ];
        if ( a == b )
            return HD_IDENTITY( hdP );
        if ( a > b )
            return CONJUGATES( hdP )[ IND( a, b ) ];
        else
        {
            hdR = AgSolution2( hdB, HD_IDENTITY( hdP ), hdA, hdB );
            Retype( hdR, T_SWORD );
            return hdR;
        }
    }

    /** Solve the equation  <hdB> * x = <hdA> * <hdB>. *********************/
    hdR = AgSolution2( hdB, HD_IDENTITY( hdP ), hdA, hdB );
    Retype( hdR, T_SWORD );
    return hdR;
}


/****************************************************************************
**
*F  FunPowerPcp( <hdCall> ) . . . . . . . . . . . . . . . internal 'PowerPcp'
**
**  'FunPowerPcp' implements 'PowerPcp( <P>, <g>, <n> )'
**
**  'PowerPcp' returns the <n>.th power of <g>.  If  <n> is omitted the index
**  of <g> is assumed.  If <g> is an integer,  the <g>.th generator is taken.
*/
Bag       FunPowerPcp (Bag hdCall)
{
    Bag           hdP,  hdG, hdN, hdR;
    Int                n;
    char                * usage = "usage: PowerPcp( <P>, <g>, <n> )";

    /** Evaluate and check the arguments ***********************************/
    if ( GET_SIZE_BAG( hdCall ) < 3 * SIZE_HD || GET_SIZE_BAG( hdCall ) > 4 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    if ( GET_TYPE_BAG( hdP ) != T_PCPRES )
        return Error( usage, 0, 0 );
    hdP = *PTR_BAG( hdP );
    hdG = EVAL( PTR_BAG( hdCall )[ 2 ] );

    /** Convert <g> into a word of <P>. ************************************/
    if ( GET_TYPE_BAG( hdG ) == T_INT )
    {
        n = HD_TO_INT( hdG );
        if ( n < 1 )
            return Error( "generator number must be positive", 0, 0 );
        if ( n > NUMBER_OF_GENS( hdP ) )
            return Error( "presenation has only %d generators",
                          NUMBER_OF_GENS( hdP ), 0 );
        if ( GET_SIZE_BAG( hdCall ) == 3 * SIZE_HD )
            return POWERS( hdP )[ n - 1 ];
        hdG = GENERATORS( hdP )[ n - 1 ];
    }
    else if ( GET_TYPE_BAG( hdG ) == T_WORD || GET_TYPE_BAG( hdG ) == T_SWORD )
    {
        hdG = NormalWordPcp( hdP, hdG );
        if ( hdG == HdFalse )
            return Error( "<g> must be a word of <P>", 0, 0 );
    }
    else
        return Error( usage, 0, 0 );
    if ( ISID_AW( hdG ) )
        return hdG;

    /** Get the power to with <g> must be raised. **************************/
    if ( GET_SIZE_BAG( hdCall ) == 3 * SIZE_HD )
        hdN = INT_TO_HD( INDICES( hdP )[ *PTR_AW( hdG ) ] );
    else
        hdN = EVAL( PTR_BAG( hdCall )[ 3 ] );
    if ( GET_TYPE_BAG( hdN ) != T_INT )
        return Error( usage, 0, 0 );

    /** Collect the power,  this may return an agword. *********************/
    hdR = PowAgI( hdG, hdN );
    Retype( hdR, T_SWORD );

    return hdR;
}


/*--------------------------------------------------------------------------\
|                         non-collector operations                          |
\--------------------------------------------------------------------------*/


/****************************************************************************
**
*F  IsNormedPcp( <p>, <*v> )  . . . . . . . . . . . . . . . . is <v> normed ?
**
**  'IsNormedPcp' returns  'true' iff <v> is normed with respect to  <P>.  If
**  it is normed but is represented as  word not  as sword,  a bag containing
**  the representation as sword is created.
*/
boolean     IsNormedPcp (Bag hdP, Bag *hdV)
{
    Bag       hdQ;
    TypSword        * ptV,  lst;
    Int            * ptI;

    if ( GET_TYPE_BAG( *hdV ) == T_SWORD )
    {
        hdQ = *PTR_BAG( *hdV );
        if ( hdQ == hdP )
            return TRUE;
        if ( GET_TYPE_BAG( hdQ ) == T_AGGRP )
            hdQ = HD_WORDS( hdQ );
        *hdV = SwordSword( HD_WORDS( hdP ), *hdV );
    }
    else if ( *hdV == HdIdWord )
    {
        *hdV = HD_IDENTITY( hdP );
        return TRUE;
    }
    else
        *hdV = SwordWord( HD_WORDS( hdP ), *hdV );
    if ( *hdV == HdFalse )
        return FALSE;
    ptV = PTR_AW( *hdV );
    ptI = INDICES( hdP );
    lst = -1;
    while ( *ptV != -1 )
    {
        if ( ptV[ 0 ] <= lst )
            return FALSE;
        if ( ptV[ 1 ] < 0 || ptV[ 1 ] >= ptI[ ptV[ 0 ] ] )
            return FALSE;
        lst  = ptV[ 0 ];
        ptV += 2;
    }
    SET_BAG( *hdV, 0, hdP);

    return TRUE;
}


/****************************************************************************
**
*F  FunSumPcp( <hdCall> ) . . . . . . . . . . . . . . . . . internal 'SumPcp'
**
**  'FunSumPcp' implements 'SumPcp( <P>, <v>, <w> )'
*/
Bag       FunSumPcp (Bag hdCall)
{
    Bag       hdP,  hdV,  hdW;
    char            * usage = "usage: SumPcp( <P>, <v>, <w> )";

    /** Evaluate and check the arguments ***********************************/
    if ( GET_SIZE_BAG( hdCall ) != 4 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdV = EVAL( PTR_BAG( hdCall )[ 2 ] );
    hdW = EVAL( PTR_BAG( hdCall )[ 3 ] );
    if (      GET_TYPE_BAG( hdP ) != T_PCPRES
         || ( GET_TYPE_BAG( hdV ) != T_SWORD && GET_TYPE_BAG( hdV ) != T_WORD )
         || ( GET_TYPE_BAG( hdW ) != T_SWORD && GET_TYPE_BAG( hdW ) != T_WORD ) )
    {
        return Error( usage, 0, 0 );
    }
    hdP = *PTR_BAG( hdP );

    /** <v> and <w> must be normed elements of <P>. ************************/
    if ( ! IsNormedPcp( hdP, &hdV ) )
        return Error( "SumPcp: <v> must be a normed word of <P>", 0, 0 );
    if ( ! IsNormedPcp( hdP, &hdW ) )
        return Error( "SumPcp: <w> must be a normed word of <P>", 0, 0 );

    hdV = SumAgWord( hdP, hdV, hdW );
    Retype( hdV, T_SWORD );
    return hdV;
}


/****************************************************************************
**
*F  FunDifferencePcp( <hdCall> )  . . . . . . . . .  internal 'DifferencePcp'
**
**  'FunDifferencePcp' implements 'DifferencePcp( <P>, <v>, <w> )'
*/
Bag       FunDifferencePcp (Bag hdCall)
{
    Bag       hdP,  hdV,  hdW;
    char            * usage = "usage: DifferencePcp( <P>, <v>, <w> )";

    /** Evaluate and check the arguments ***********************************/
    if ( GET_SIZE_BAG( hdCall ) != 4 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdV = EVAL( PTR_BAG( hdCall )[ 2 ] );
    hdW = EVAL( PTR_BAG( hdCall )[ 3 ] );
    if (      GET_TYPE_BAG( hdP ) != T_PCPRES
         || ( GET_TYPE_BAG( hdV ) != T_SWORD && GET_TYPE_BAG( hdV ) != T_WORD )
         || ( GET_TYPE_BAG( hdW ) != T_SWORD && GET_TYPE_BAG( hdW ) != T_WORD ) )
    {
        return Error( usage, 0, 0 );
    }
    hdP = *PTR_BAG( hdP );

    /** <v> and <w> must be normed elements of <P>. ************************/
    if ( ! IsNormedPcp( hdP, &hdV ) )
       return Error("DifferencePcp: <v> must be a normed word of <P>",0,0);
    if ( ! IsNormedPcp( hdP, &hdW ) )
       return Error("DifferencePcp: <w> must be a normed word of <P>",0,0);

    hdV = DifferenceAgWord( hdP, hdV, hdW );
    Retype( hdV, T_SWORD );
    return hdV;
}


/****************************************************************************
**
*F  FunExponentPcp( <hdCall> )  . . . . . . . . . . .  internal 'ExponentPcp'
*/
Bag       FunExponentPcp (Bag hdCall)
{
    Bag       hdP,  hdG,  hdI;
    TypSword        * ptG,  * ptE;
    Int            i;
    char            * usage = "usage: ExponentPcp( <P>, <g>, <i> )";

    /** Evaluate and check the arguments. **********************************/
    if ( GET_SIZE_BAG( hdCall ) != 4 * SIZE_HD  )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdG = EVAL( PTR_BAG( hdCall )[ 2 ] );
    hdI = EVAL( PTR_BAG( hdCall )[ 3 ] );
    if ( GET_TYPE_BAG( hdP ) != T_PCPRES || GET_TYPE_BAG( hdI ) != T_INT )
        return Error( usage, 0, 0 );
    hdP = *PTR_BAG( hdP );
    i   = HD_TO_INT( hdI ) - 1;

    if ( *PTR_BAG( hdG ) != hdP && ! IsNormedPcp( hdP, &hdG ) )
        return Error( "<g> must be a normed word of <P>", 0, 0 );
    if ( i < 0 )
        return Error( "generator number must be positive", 0, 0 );
    if ( i >= NUMBER_OF_GENS( hdP ) )
        return Error( "presentation <P> has only %d generators",
                      NUMBER_OF_GENS( hdP ), 0 );

    /** Run through the sparse exponent vector  and search for <i>,  skip **/
    /** the last entry, which is an end mark.                             **/
    ptG = PTR_AW( hdG );
    ptE = ptG + 2 * LEN_AW( hdG );
    while ( ptG < ptE )
    {
        if ( ptG[0] == i )
            return INT_TO_HD( ptG[1] );
        else if ( ptG[0] > i )
            return INT_TO_HD( 0 );
        ptG += 2;
    }
    return INT_TO_HD( 0 );
}


/****************************************************************************
**
*F  FunExponentsPcp( <hdCall> )  . . . . . . . . . .  internal 'ExponentsPcp'
**
**  'FunExponentsPcp' implements 'ExponentsPcp( <P>, <v> )'
*/
Bag       FunExponentsPcp (Bag hdCall)
{
    Bag       hdP,  hdV;
    char            * usage = "usage: ExponentsPcp( <P>, <v> )";

    /** Evaluate and check the arguments ***********************************/
    if ( GET_SIZE_BAG( hdCall ) != 3 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdV = EVAL( PTR_BAG( hdCall )[ 2 ] );
    if (      GET_TYPE_BAG( hdP ) != T_PCPRES
         || ( GET_TYPE_BAG( hdV ) != T_SWORD && GET_TYPE_BAG( hdV ) != T_WORD ) )
    {
        return Error( usage, 0, 0 );
    }
    hdP = *PTR_BAG( hdP );

    /** <v> and <w> must be normed elements of <P>. ************************/
    if ( ! IsNormedPcp( hdP, &hdV ) )
       return Error("ExponentsPcp: <v> must be a normed word of <P>",0,0);

    return IntExponentsAgWord( hdV, 1, NUMBER_OF_GENS( hdP ) );
}


/****************************************************************************
**
*F  FunDepthPcp( <hdCall> ) . . . . . . . . . . . . . . . internal 'DepthPcp'
**
**  'FunDepthPcp' implements 'DepthPcp( <g> )'
*/
Bag       FunDepthPcp (Bag hdCall)
{
    Bag       hdWrd,  hdP;
    char            * usage = "usage: DepthPcp( <P>, <g> )";

    /** Evaluate and check the arguments. **********************************/
    if ( GET_SIZE_BAG( hdCall ) != 3 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP   = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdWrd = EVAL( PTR_BAG( hdCall )[ 2 ] );
    if ((GET_TYPE_BAG(hdWrd)!=T_WORD && GET_TYPE_BAG(hdWrd)!=T_SWORD) || GET_TYPE_BAG(hdP)!=T_PCPRES)
        return Error( usage, 0, 0 );
    hdP = *PTR_BAG( hdP );

    if ( ! IsNormedPcp( hdP, &hdWrd ) )
       return Error("DepthPcp: <g> must be a normed word of <P>",0,0);
    return INT_TO_HD( ( *( PTR_AW( hdWrd ) ) + 1 ) );
}


/****************************************************************************
**
*F  FunTailDepthPcp( <hdCall> ) . . . . . . . . . . . internal 'TailDepthPcp'
**
**  'FunTailDepthPcp' implements 'TailDepthPcp( <g> )'
*/
Bag       FunTailDepthPcp (Bag hdCall)
{
    Bag       hdWrd,  hdP;
    TypSword        * ptWrd;
    char            * usage = "usage: TailDepthPcp( <P>, <g> )";

    /** Evaluate and check the arguments. **********************************/
    if ( GET_SIZE_BAG( hdCall ) != 3 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP   = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdWrd = EVAL( PTR_BAG( hdCall )[ 2 ] );
    if ((GET_TYPE_BAG(hdWrd)!=T_WORD && GET_TYPE_BAG(hdWrd)!=T_SWORD) || GET_TYPE_BAG(hdP)!=T_PCPRES)
        return Error( usage, 0, 0 );
    hdP = *PTR_BAG( hdP );

    if ( ! IsNormedPcp( hdP, &hdWrd ) )
       return Error("TailDepthPcp: <g> must be a normed word of <P>",0,0);
    if ( ISID_AW( hdWrd ) )
        return INT_TO_HD( 0 );
    ptWrd = (TypSword*)( (char*) PTR_BAG( hdWrd ) + GET_SIZE_BAG( hdWrd ) );
    return INT_TO_HD( ( ptWrd[ -3 ] + 1 ) );
}


#if 0
/****************************************************************************
**
*F  FunAgWordExponents( <hdCall> )  . . . . . . . .internal 'AgWordExponents'
**
**  'FunAgWordExponents' implements
**              'AgWordExponents( <pcpres>, <list> )'
**              'AgWordExponents( <pcpres>, <list>, <start> )'
**
*/
Bag       FunAgWordExponents ( hdCall )
    Bag       hdCall;
{
    Bag       hdW, hdL, hdP, hdI;
    TypSword        * ptW;
    Int            * ptIndices, nonTrivial, i, exp, start;

    /** Evaluate and check the arguments ***********************************/
    if ( GET_SIZE_BAG( hdCall ) < 3 * SIZE_HD || GET_SIZE_BAG( hdCall ) > 4 * SIZE_HD )
    {
        return Error( "usage: AgWordExponents( <pcpres>, <list> )", 0, 0 );
    }
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdL = EVAL( PTR_BAG( hdCall )[ 2 ] );
    if ( GET_SIZE_BAG( hdCall ) == 4 * SIZE_HD )
    {
        hdI = EVAL( PTR_BAG( hdCall )[ 3 ] );
        if ( GET_TYPE_BAG( hdI ) != T_INT )
        {
          return Error("usage: AgWordExponents( <pcpres>, <list>, <start> )",
                       0, 0 );
        }
        start = HD_TO_INT( hdI ) - 1;
    }
    else
        start = 0;
    if ( GET_TYPE_BAG( hdP ) != T_PCPRES
         || ! IsVector( hdL )
         || ( LEN_LIST( hdL ) > 0 && GET_TYPE_BAG( ELM_PLIST(hdL,0) ) != T_INT ) )
    {
        return Error( "usage: AgWordExponents( <pcpres>, <list> )", 0, 0 );
    }

    /** enough but not too many generators ? *******************************/
    if ( start < 0 )
    {
        return Error( "AgWordExponents: <start> must be positive", 0, 0 );
    }
    if ( start + LEN_LIST( hdL ) > NUMBER_OF_GENS( hdP ) )
    {
        return Error( "AgWordExponents: too many generators", 0, 0 );
    }

    /** count nontrivial generators ****************************************/
    nonTrivial = 0;
    ptIndices = INDICES( hdP );
    for ( i = LEN_LIST( hdL ); i > 0; i-- )
        if ( HD_TO_INT( ELM_PLIST( hdL, i ) ) % ptIndices[ i+start-1 ] != 0 )
            nonTrivial++;

    /** Copy generators ****************************************************/
    hdW = NewBag( T_AGWORD, SIZE_HD + ( 2 * nonTrivial + 1 ) * SIZE_GEN );
    *PTR_BAG( hdW ) = hdP;
    ptW = PTR_AW( hdW ) + 2 * nonTrivial;
    *ptW-- = -1;
    ptIndices = INDICES( hdP );
    for ( i = LEN_LIST( hdL ); i > 0; i-- )
    {
        exp = HD_TO_INT( ELM_PLIST( hdL, i ) ) % ptIndices[ i+start-1 ];
        if ( exp != 0 )
        {
            *ptW-- = exp;
            *ptW-- = i + start - 1;
        }
    }
    return hdW;
}
#endif


/****************************************************************************
**
*F  FunBaseReducedPcp( <hdCall> ) . . . . . . . . . internal 'BaseReducedPcp'
**
**  'FunBaseReducedPcp' implements 'BaseReducedPcp( <P>, <B>, <v> )'
**
*/
Bag       FunBaseReducedPcp (Bag hdCall)
{
    Bag       hdV,  hdW,  hdL,  hdP;
    TypSword        * ptW,  * ptV;
    TypExp          * ptR;
    Int            * ptIdx,  lenL,  i,  exp;
    char            * usage = "usage: BaseReducedPcp( <P>, <B>, <v> )";

    /** Evaluate and check the arguments ***********************************/
    if ( GET_SIZE_BAG( hdCall ) != 4 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdL = EVAL( PTR_BAG( hdCall )[ 2 ] );
    hdW = EVAL( PTR_BAG( hdCall )[ 3 ] );
    if (    ! IsList( hdL )
         || ( GET_TYPE_BAG( hdW ) != T_WORD && GET_TYPE_BAG( hdW ) != T_SWORD )
         ||   GET_TYPE_BAG( hdP ) != T_PCPRES )
    {
        return Error( usage, 0, 0 );
    }
    hdP = *PTR_BAG( hdP );
    if ( ! IsNormedPcp( hdP, &hdW ) )
        return Error( "<v> must be a normed word of <P>", 0, 0 );

    lenL = LEN_LIST( hdL );
    if ( lenL > NUMBER_OF_GENS( hdP ) )
        return Error( "<P> has only %d generators, but <B> has length %d",
                      NUMBER_OF_GENS( hdP ), lenL );

    /** catch some trivial cases *******************************************/
    ptW = PTR_AW( hdW );
    if ( *ptW == -1 || *ptW >= lenL )
        return hdW;

    /** convert into exponent vector ***************************************/
    SetCollectExponents( hdW );
    ptIdx = INDICES( hdP );

    /** run through the exponents ******************************************/
    ptR = COLLECT_EXPONENTS( hdP );
    for ( i = *ptW;  i < lenL;  i++ )
    {
        exp = ptR[ i ];
        if ( exp != 0 )
        {
            hdV = ELM_PLIST( hdL, i + 1 );
            if ( hdV == 0 )
                continue;
            if ( *PTR_BAG( hdV ) != hdP )
            {
                if ( ! IsNormedPcp( hdP, &hdV ) )
                {
                    ClearCollectExponents( hdP );
                    return Error( "element %d must be a normed word of <P>",
                                  i + 1, 0 );
                }
                else
                {
                    SET_ELM_PLIST( hdL, i + 1, hdV );
                    ptIdx = INDICES( hdP );
                    ptR   = COLLECT_EXPONENTS( hdP );
                }
            }

            /** Check the depth and leading exponent of the element. *******/
            ptV = PTR_AW( hdV );
            if ( *ptV != i )
            {
                ClearCollectExponents( hdP );
                return Error( "depth of %d. base element must be %d",
                              i + 1, i + 1 );
            }
            if ( *( ptV + 1 ) != 1 )
            {
                ClearCollectExponents( hdP );
                return Error( "leading exponent of %d. element must be 1",
                              i + 1, 0 );
            }

            /** now reduce *************************************************/
            while ( *ptV != -1 )
            {
                ptR[*ptV] = (ptR[*ptV] - exp*(ptV[1])) % ptIdx[*ptV];
                if ( ptR[ *ptV ] < 0 )
                    ptR[ *ptV ] += ptIdx[ *ptV ];
                ptV += 2;
            }
        }
    }

    /** convert exponent vector back into an sword *************************/
    hdV = AgWordAgExp( HD_COLLECT_EXPONENTS( hdP ), hdP );
    Retype( hdV, T_SWORD );
    return hdV;
}


/****************************************************************************
**
*F  FunTailReducedPcp( <hdCall> ) . . . . . . . . . internal 'TailReducedPcp'
**
**  'FunTailReducedPcp' implements 'TailReducedPcp( <P>, <B>, <v> )'
**
*/
Bag       FunTailReducedPcp (Bag hdCall)
{
    Bag       hdV, hdW, hdL, hdP;
    TypSword        * ptW, * ptV;
    TypExp          * ptR;
    Int            * ptIdx, lenL, i, exp, lenV;
    char            * usage = "TailReducedPcp( <P>, <B>, <v> )";

    /** Evaluate and check the arguments ***********************************/
    if ( GET_SIZE_BAG( hdCall ) != 4 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdL = EVAL( PTR_BAG( hdCall )[ 2 ] );
    hdW = EVAL( PTR_BAG( hdCall )[ 3 ] );
    if (    ! IsList( hdL )
         || ( GET_TYPE_BAG( hdW ) != T_WORD && GET_TYPE_BAG( hdW ) != T_SWORD )
         ||   GET_TYPE_BAG( hdP ) != T_PCPRES )
    {
        return Error( usage, 0, 0 );
    }
    hdP = *PTR_BAG( hdP );
    if ( ! IsNormedPcp( hdP, &hdW ) )
        return Error( "<v> must be a normed word of <P>", 0, 0 );

    lenL = LEN_LIST( hdL );
    if ( lenL > NUMBER_OF_GENS( hdP ) )
        return Error( "<P> has only %d generators, but <B> has length %d",
                      NUMBER_OF_GENS( hdP ), lenL );

    /** catch some trivial cases *******************************************/
    ptW = PTR_AW( hdW );
    if ( *ptW == -1 )
        return hdW;

    /** convert into exponent vector ***************************************/
    SetCollectExponents( hdW );
    ptIdx = INDICES( hdP );

    /** run through the exponents ******************************************/
    ptR = COLLECT_EXPONENTS( hdP );
    for ( i = lenL - 1; i >= 0; i-- )
    {
        exp = ptR[ i ];
        if ( exp != 0 )
        {
            hdV = ELM_PLIST( hdL, i + 1 );
            if ( hdV == 0 )
                continue;
            if ( *PTR_BAG( hdV ) != hdP )
            {
                if ( ! IsNormedPcp( hdP, &hdV ) )
                {
                    ClearCollectExponents( hdP );
                    return Error( "element %d must be a normed word of <P>",
                                  i + 1, 0 );
                }
                else
                {
                    SET_ELM_PLIST( hdL, i + 1, hdV );
                    ptIdx = INDICES( hdP );
                    ptR   = COLLECT_EXPONENTS( hdP );
                }
            }

            /** Check the depth and trailing exponent of the element. ******/
            ptV  = PTR_AW( hdV );
            lenV = LEN_AW( hdV );
            if ( ptV[ 2 * lenV - 2 ] != i )
            {
                ClearCollectExponents( hdP );
                return Error( "tail depth of %d. base element must be %d",
                              i + 1, i + 1 );
            }
            if ( ptV[ 2 * lenV - 1 ] != 1 )
            {
                ClearCollectExponents( hdP );
                return Error( "trailing exponent of %d. element must be 1",
                              i + 1, 0 );
            }

            /** now reduce *************************************************/
            while ( *ptV != -1 )
            {
                ptR[*ptV] = (ptR[*ptV] - exp*(ptV[1])) % ptIdx[*ptV];
                if ( ptR[ *ptV ] < 0 )
                    ptR[ *ptV ] += ptIdx[ *ptV ];
                ptV += 2;
            }
        }
    }

    /** convert exponent vector back into an agword ************************/
    hdV = AgWordAgExp( HD_COLLECT_EXPONENTS( hdP ), hdP );
    Retype( hdV, T_SWORD );
    return hdV;
}


/*--------------------------------------------------------------------------\
|                              debug functions                              |
\--------------------------------------------------------------------------*/


#if PCP_DEBUG

/****************************************************************************
**
*F  FunPowersPcp( <P> ) . . . . . . . . . . . . . . . .  internal 'PowersPcp'
**
**  'FunPowersPcPres' implements 'PowersPcp( <P> )'
**
**  'PowersPcp' returns  the list of right hand sides power-relations  of <P>
**  Note that we do  NOT return a copy of that list,  so every change made to
**  the list is also made in the presentation.
**
**  This is for debug only !!!
*/
Bag       FunPowersPcp (Bag hdCall )
{
    Bag       hdP;
    char            * usage = "usage: PowersPcp( <P> )";

    /** Check and evaluate arguments ***************************************/
    if ( GET_SIZE_BAG( hdCall ) != 2 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    if ( GET_TYPE_BAG( hdP ) != T_PCPRES )
        return Error( usage, 0, 0 );

    /** Retype *************************************************************/
    return HD_POWERS( *PTR_BAG( hdP ) );
}


/****************************************************************************
**
*F  FunCommutatorsPcp( <P> )  . . . . . . . . . . . internal 'CommutatorsPcp'
**
**  'FunCommutatorsPcp' implements 'CommutatorsPcp( <P> )'
**
**  'CommutatorsPcp' returns the list of right hand sides  power-relations of
**  <P>.  Note that we do  NOT  return a copy of that list,  so every  change
**  made to the list is also made in the presentation.
**
**  This is for debug only !!!
*/
Bag       FunCommutatorsPcp (Bag hdCall )
{
    Bag       hdP;
    char            * usage = "usage: CommutatorsPcp( <P> )";

    /** Check and evaluate arguments ***************************************/
    if ( GET_SIZE_BAG( hdCall ) != 2 * SIZE_HD )
        return Error( usage, 0, 0 );
    hdP = EVAL( PTR_BAG( hdCall )[ 1 ] );
    if ( GET_TYPE_BAG( hdP ) != T_PCPRES )
        return Error( usage, 0, 0 );

    /** Retype *************************************************************/
    return HD_COMMUTATORS( *PTR_BAG( hdP ) );
}

#endif /* PCP_DEBUG */


/*--------------------------------------------------------------------------\
|                               print function                              |
\--------------------------------------------------------------------------*/


/****************************************************************************
**
*F  PrPcPres( <P> ) . . . . . . . . . . . . . print a polycyclic presentation
**
*N  Can the presentation be printed such that it could be read in again?
*/
void        PrPcPres(Bag hdP)
{
    Pr( "<Pcp: %d generators, %s collector>",
        (Int) NUMBER_OF_GENS( *PTR_BAG( hdP ) ),
        (Int) Collectors[ COLLECTOR( *PTR_BAG( hdP ) ) ].name );
}


/*--------------------------------------------------------------------------\
|                      install polycyclic presentations                     |
\--------------------------------------------------------------------------*/


/****************************************************************************
**
*F  InitPcPres( void )  . . . . . . . . . initialize polycyclic presentations
*/
void        InitPcPres (void)
{
    /** Arithmetic functions using a collector *****************************/
    InstIntFunc( "NormalWordPcp",           FunNormalWordPcp            );
    InstIntFunc( "PowerPcp",                FunPowerPcp                 );
    InstIntFunc( "CommPcp",                 FunCommPcp                  );
    InstIntFunc( "ConjugatePcp",            FunConjugatePcp             );
    InstIntFunc( "ProductPcp",              FunProductPcp               );
    InstIntFunc( "QuotientPcp",             FunQuotientPcp              );
    InstIntFunc( "LeftQuotientPcp",         FunLeftQuotientPcp          );

    /** Functions handling swords without using a collector ****************/
    InstIntFunc( "SumPcp",                  FunSumPcp                   );
    InstIntFunc( "DifferencePcp",           FunDifferencePcp            );
    InstIntFunc( "ExponentPcp",             FunExponentPcp              );
    InstIntFunc( "ExponentsPcp",            FunExponentsPcp             );
    InstIntFunc( "DepthPcp",                FunDepthPcp                 );
    InstIntFunc( "TailDepthPcp",            FunTailDepthPcp             );
    InstIntFunc( "BaseReducedPcp",          FunBaseReducedPcp           );
    InstIntFunc( "TailReducedPcp",          FunTailReducedPcp           );

    /** Install the print function for T_PCPRES ****************************/
    InstPrFunc( T_PCPRES,                   PrPcPres                    );

    /** Various functions **************************************************/
    InstIntFunc( "GeneratorsPcp",           FunGeneratorsPcp            );
    InstIntFunc( "CentralWeightsPcp",       FunCentralWeightsPcp        );
    InstIntFunc( "DefineCentralWeightsPcp", FunDefineCentralWeightsPcp  );
    InstIntFunc( "DefineCommPcp",           FunDefineCommPcp            );
    InstIntFunc( "AddCommPcp",              FunAddCommPcp               );
    InstIntFunc( "SubtractCommPcp",         FunSubtractCommPcp          );
    InstIntFunc( "DefinePowerPcp",          FunDefinePowerPcp           );
    InstIntFunc( "AddPowerPcp",             FunAddPowerPcp              );
    InstIntFunc( "SubtractPowerPcp",        FunSubtractPowerPcp         );
    InstIntFunc( "Pcp",                     FunPcp                      );
    InstIntFunc( "TriangleIndex",           FunTriangleIndex            );
    InstIntFunc( "ExtendCentralPcp",        FunExtendCentralPcp         );
    InstIntFunc( "ShrinkPcp",               FunShrinkPcp                );
    InstIntFunc( "AgPcp",                   FunAgPcp                    );

    /** Debug function *****************************************************/
#if PCP_DEBUG
        InstIntFunc( "PowersPcp",           FunPowersPcp                );
        InstIntFunc( "CommutatorsPcp",      FunCommutatorsPcp           );
#endif
}
