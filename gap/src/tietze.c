/****************************************************************************
**
*A  tietze.c                    GAP source                     Volkmar Felsch
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file contains the functions for computing with finite presentations.
**
*/

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */
#include        "list.h"                /* 'LEN_LIST' macro                */
#include        "agcollec.h"            /* 'HD_WORDS' macro for T_SWORDS   */
#include        "word.h"                /* words                           */

#include        "tietze.h"              /* declaration part of the package */


/****************************************************************************
**
**  Definig some constants for the Tietze routines.
*/
#define    TZ_NUMGENS        1
#define    TZ_NUMRELS        2
#define    TZ_TOTAL          3
#define    TZ_GENERATORS     4
#define    TZ_INVERSES       5
#define    TZ_RELATORS       6
#define    TZ_LENGTHS        7
#define    TZ_FLAGS          8
#define    TZ_LENGTHTIETZE  20


/****************************************************************************
**
*F  TzRelExponent1( <relator> ) . . . . find the exponent of a Tietze relator
**
**  'TzRelExponent1'  determines the exponent of the given relator, i. e. the
**  maximal integer n such that the relator can be expresses as an nth power.
*/
Bag       TzRelExponent1 (Bag hdRel)
{
    Bag                 * ptRel;        /* pointer to the Tietze relator   */
    Int                leng;           /* length of the given relator     */
    Int                exp;            /* exponent of the relator         */
    Int                i, j, ij;       /* loop variables                  */

    /*  Initialize some variables                                          */
    ptRel = (Bag*)PTR_BAG( hdRel ) + 1;
    leng = HD_TO_INT( ptRel[-1] );
    exp = 1;

    /*  Run through the relator and check for repitions                    */
    for ( i = 1; i <= leng/2; i++ ) {
        if ( leng % i == 0 ) {
            for ( j = 0, ij = i ; j < leng; j++, ij = ( ij + 1 ) % leng ) {
                if ( ptRel[j] != ptRel[ij] )  break;
            }
            if ( j == leng ) {
                exp = leng / i;
                break;  /*  loop over i  */
            }
        }
    }

    return INT_TO_HD( exp );
}


/****************************************************************************
**
*F  FunTzRelator( <hdCall> ) . . . . . . . convert a word to a Tietze relator
**
**  'FunTzRelator'  converts a  word in the group generators  to a Tietze re-
**  lator,  i.e. a free and cyclically reduced word in the Tietze generators.
**  It returns 'HdFalse" if it cannot convert the given word.
*/
Bag       FunTzRelator (Bag hdCall)
{
    Bag           hdTie;          /* handle of the Tietze stack      */
    Bag           * ptTie;        /* pointer to the Tietze stack     */
    Bag           hdGens;         /* handle of the Tietze gens list  */
    Bag           * ptGens;       /* pointer to this list            */
    Bag           hdInvs;         /* handle of inverses list         */
    Bag           * ptInvs;       /* pointer to this list            */
    Bag           hdWrd;          /* handle of the given word        */
    Bag           * ptWrd;        /* pointer to the given word       */
    Bag           hdRel;          /* handle of the Tietze relator    */
    Bag           * ptRel;        /* pointer to the Tietze relator   */
    Bag           * pt1, * pt2;   /* pointers to the Tietze relator  */
    Bag           hdFac, hdGen;   /* handles of generators           */
    Int                numgens;        /* number of Tietze generators     */
    Int                leng;           /* length of the given word        */
    Int                reduced;        /* length of reduced relator       */
    Int                i, j;           /* generator numbers               */

    /* Get and check arguments                                             */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error( "usage: TzRelator(<Tietze stack>,<word>)",  0,  0 );

    hdTie = EVAL( PTR_BAG(hdCall)[1] );
    hdWrd = EVAL( PTR_BAG(hdCall)[2] );

    /*  Check the first argument (Tietze stack)                            */
    if (GET_TYPE_BAG(hdTie) != T_LIST || HD_TO_INT(PTR_BAG(hdTie)[0]) != TZ_LENGTHTIETZE)
        return Error( "invalid <Tietze stack>",  0,  0 );
    ptTie = PTR_BAG( hdTie );

    /*  Get and check the Tietze generators list                           */
    hdGens = ptTie[TZ_GENERATORS];
    numgens = HD_TO_INT(ptTie[TZ_NUMGENS]);
    if ( hdGens == 0
      || (GET_TYPE_BAG(hdGens) != T_LIST && GET_TYPE_BAG(hdGens) != T_SET)
      || HD_TO_INT(PTR_BAG(hdGens)[0]) != numgens )
        return Error( "invalid Tietze generators list",  0,  0 );
    ptGens = PTR_BAG( hdGens );
    for ( i = 1; i <= numgens; i++ ) {
        if ( GET_TYPE_BAG(ptGens[i]) == T_SWORD ) ptGens[i] = WordSword( ptGens[i] );
    }

    /*  Get and check the Tietze inverses list                             */
    hdInvs = ptTie[TZ_INVERSES];
    if ( hdInvs == 0
      || (GET_TYPE_BAG(hdInvs) != T_LIST && GET_TYPE_BAG(hdInvs) != T_VECTOR)
      || HD_TO_INT(PTR_BAG(hdInvs)[0]) != 2 * numgens + 1 )
        return Error( "invalid Tietze inverses list",  0,  0 );

    /*  Check the second argument (word to be converted)                   */
    if ( GET_TYPE_BAG(hdWrd) == T_SWORD ) hdWrd = WordSword( hdWrd );
    if ( GET_TYPE_BAG(hdWrd) != T_WORD )
        return Error( "TzRelator: <word> must be a word",  0,  0 );
    leng = GET_SIZE_BAG( hdWrd ) / SIZE_HD;

    /*  Allocate a bag for the Tietze relator                              */
    hdRel = NewBag( T_LIST, (leng + 1) * SIZE_HD );
    ptGens = PTR_BAG( hdGens );
    ptInvs = PTR_BAG( hdInvs ) + numgens + 1;
    ptWrd = PTR_BAG( hdWrd );
    ptRel = PTR_BAG( hdRel );
    pt2 = ptRel;

    /*  Run through the word, identify each factor in the list of          */
    /*  generators, and convert it.                                        */
    for ( j = 0; j < leng; j++ ) {
        if ( GET_TYPE_BAG(ptWrd[j]) == T_SWORD ) ptWrd[j] = WordSword( ptWrd[j] );
        hdFac = ptWrd[j];
        for ( i = 1; i <= numgens; i++ ) {
            hdGen = PTR_BAG( ptGens[i] )[0];
            if ( hdFac == hdGen || hdFac == *PTR_BAG( hdGen ) )
                 break;
        }
        if ( i > numgens )
            return HdFalse;

        if ( hdFac != hdGen ) i = HD_TO_INT( ptInvs[i] );
        if ( pt2 > ptRel && *pt2 == ptInvs[i] )
            --pt2;
        else
            *++pt2 = INT_TO_HD( i );
    }

    /*  Now cyclically reduce the resulting relator                        */
    pt1 = ++ptRel;
    while ( pt1 < pt2 && *pt1 == ptInvs[HD_TO_INT(*pt2)] ) {
        ++pt1;  --pt2;
    }
    if ( ptRel < pt1 ) {
        while ( pt1 <= pt2 )   *ptRel++ = *pt1++;
        pt2 = --ptRel;
    }

    /*  Resize the resulting relator, if necessary, and return it          */
    reduced = pt2 - PTR_BAG( hdRel );
    if ( reduced < leng ) {
        Resize( hdRel, ( reduced + 1 ) * SIZE_HD );
        leng = reduced;
    }

    SET_BAG( hdRel , 0,  INT_TO_HD( leng ) );
    return hdRel;
}


/****************************************************************************
**
*F  FunTzWord( <hdCall> ) . . . . . . . .  convert a Tietze relator to a word
**
**  'FunTzWord'  converts a Tietze relator to a word in the group generators.
*/
Bag       FunTzWord (Bag hdCall)
{
    Bag           hdTie;          /* handle of the Tietze stack      */
    Bag           * ptTie;        /* pointer to the Tietze stack     */
    Bag           hdGens;         /* handle of the Tietze gens list  */
    Bag           * ptGens;       /* pointer to this list            */
    Bag           hdRel;          /* handle of the Tietze relator    */
    Bag           * ptRel;        /* pointer to the Tietze relator   */
    Bag           hdWrd;          /* handle of the new word          */
    Bag           * ptWrd;        /* pointer to the new word         */
    Int                numgens;        /* number of Tietze generators     */
    Int                leng;           /* length of the Tietze relator    */
    Int                i, iabs, j;     /* integer variables               */

    /* Get and check arguments                                             */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error( "usage: TzWord(<Tietze stack>,<Tietze relator>)",
             0,  0 );

    hdTie = EVAL( PTR_BAG(hdCall)[1] );
    hdRel = EVAL( PTR_BAG(hdCall)[2] );

    /*  Check the first argument (Tietze stack)                            */
    if (GET_TYPE_BAG(hdTie) != T_LIST || HD_TO_INT(PTR_BAG(hdTie)[0]) != TZ_LENGTHTIETZE)
        return Error( "invalid <Tietze stack>",  0,  0 );
    ptTie = PTR_BAG( hdTie );

    /*  Get and check the Tietze generators                                */
    hdGens = ptTie[TZ_GENERATORS];
    numgens = HD_TO_INT(ptTie[TZ_NUMGENS]);
    if ( hdGens == 0
      || (GET_TYPE_BAG(hdGens) != T_LIST && GET_TYPE_BAG(hdGens) != T_SET)
      || HD_TO_INT(PTR_BAG(hdGens)[0]) != numgens )
        return Error( "invalid Tietze generators list",  0,  0 );

    /*  Check the second argument (Tietze relator to be converted)         */
    if ( GET_TYPE_BAG(hdRel) != T_LIST && GET_TYPE_BAG(hdRel) != T_SET && GET_TYPE_BAG(hdRel) !=
        T_VECTOR ) return Error( "invalid <Tietze relator>",  0,  0 );
    ptRel = PTR_BAG( hdRel );
    leng = HD_TO_INT(ptRel[0]);

    /*  Allocate a bag for the new word                                    */
    hdWrd = NewBag( T_WORD, leng * SIZE_HD );
    ptGens = PTR_BAG( hdGens );
    ptRel = PTR_BAG( hdRel );
    ptWrd = PTR_BAG( hdWrd );

    /*  Now loop over all factors and convert them                         */
    for ( j = 0; j < leng; j++ ) {
        i = HD_TO_INT( *++ptRel );
	if ( i < -numgens || numgens < i || !i )
            return Error( "invalid <Tietze relator> entry [%d] = %d", j, i );
        iabs = ( i > 0 ) ? i : -i;
        if ( GET_TYPE_BAG(ptGens[iabs]) == T_SWORD )
            ptGens[iabs] = WordSword( ptGens[iabs] );
        if ( GET_TYPE_BAG(ptGens[iabs]) != T_WORD || GET_SIZE_BAG(ptGens[iabs]) != SIZE_HD )
            return Error( "invalid Tietze generator [%d]", iabs,  0 );
        *ptWrd++ = ( i > 0 ) ?
            PTR_BAG( ptGens[iabs] )[0] : *PTR_BAG( PTR_BAG( ptGens[iabs] )[0] );
    }

    return hdWrd;
}


/****************************************************************************
**
*F  FunTzSortC(<hdCall>)  . . . . . . . . . . . . sort the relators by length
*/
Bag       FunTzSortC (Bag hdCall)
{
    Bag           hdTie;          /* handle of the Tietze stack      */
    Bag           * ptTie;        /* pointer to the Tietze stack     */
    Bag           hdRels;         /* handle of the relators list     */
    Bag           * ptRels;       /* pointer to this list            */
    Bag           hdLens;         /* handle of the lengths list      */
    Bag           * ptLens;       /* pointer to this list            */
    Bag           hdFlags;        /* handle of the flags list        */
    Bag           * ptFlags;      /* pointer to this list            */
    UInt       numrels;        /* number of Tietze relators       */
    UInt       i, h, k;        /* loop variables                  */
    Bag           hdrel, hdlen;   /* handles of list entries         */
    Bag           hdflag;         /* handle of a list entry          */

    /* Get and check arguments                                             */
    if ( GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
        return Error( "usage: TzSortC(<Tietze stack>)",  0,  0 );

    /*  Check the Tietze stack                                             */
    hdTie = EVAL( PTR_BAG(hdCall)[1] );
    if (GET_TYPE_BAG(hdTie) != T_LIST || HD_TO_INT(PTR_BAG(hdTie)[0]) != TZ_LENGTHTIETZE)
        return Error( "invalid <Tietze stack>",  0,  0 );
    ptTie = PTR_BAG( hdTie );

    /*  Get and check the Tietze relators list                             */
    hdRels = ptTie[TZ_RELATORS];
    numrels = HD_TO_INT(ptTie[TZ_NUMRELS]);
    if ( hdRels == 0
      || (GET_TYPE_BAG(hdRels) != T_LIST && GET_TYPE_BAG(hdRels) != T_SET)
      || HD_TO_INT(PTR_BAG(hdRels)[0]) != numrels )
        return Error( "invalid Tietze relators list",  0,  0 );
    ptRels = PTR_BAG( hdRels );

    /*  Get and check the Tietze lengths list                              */
    hdLens = ptTie[TZ_LENGTHS];
    if ( hdLens == 0
      || (GET_TYPE_BAG(hdLens) != T_VECTOR && GET_TYPE_BAG(hdLens) != T_LIST)
      || HD_TO_INT(PTR_BAG(hdLens)[0]) != numrels )
        return Error( "invalid Tietze lengths list",  0,  0 );
    ptLens = PTR_BAG( hdLens );

    /*  Get and check the Tietze flags list                                */
    hdFlags = ptTie[TZ_FLAGS];
    if ( hdFlags == 0
      || (GET_TYPE_BAG(hdFlags) != T_VECTOR && GET_TYPE_BAG(hdFlags) != T_LIST)
      || HD_TO_INT(PTR_BAG(hdFlags)[0]) != numrels )
        return Error( "invalid Tietze flags list",  0,  0 );
    ptFlags = PTR_BAG( hdFlags );

    /* Check list <lengths> to contain the relator lengths                 */
    for ( i = 1; i <= numrels; i++ ) {
        if ( ptRels[i] == 0
          || (GET_TYPE_BAG(ptRels[i]) != T_LIST && GET_TYPE_BAG(ptRels[i]) != T_SET
           && GET_TYPE_BAG(ptRels[i]) != T_VECTOR)
          || HD_TO_INT(ptLens[i]) != HD_TO_INT(PTR_BAG(ptRels[i])[0]) )
            return Error( "inconsistent Tietze lengths list",  0,  0 );
    }

    h = 1;
    while ( 9 * h + 4 < numrels ) h = 3 * h + 1;
    while ( 0 < h ) {
        for ( i = h + 1; i <= numrels; i++ ) {
            hdrel = ptRels[i];  hdlen = ptLens[i];  hdflag = ptFlags[i];
            k = i;
            if ( HD_TO_INT(hdlen) ) {
                while ( h < k
                    && ( !HD_TO_INT(ptLens[k-h])
                       || hdlen < ptLens[k-h]
                       || (hdlen == ptLens[k-h] && hdflag > ptFlags[k-h]))) {
                    ptRels[k] = ptRels[k-h];
                    ptLens[k] = ptLens[k-h];
                    ptFlags[k] = ptFlags[k-h];
                    k = k - h;
                }
            }
            ptRels[k] = hdrel;  ptLens[k] = hdlen;  ptFlags[k] = hdflag;
        }
        h = h / 3;
    }
    for ( i = numrels; i > 0; i-- ) {
        if ( HD_TO_INT(ptLens[i]) )
            break;
    }
    if ( i < numrels ) {
        ptRels[0] = ptLens[0] = ptFlags[0] = ptTie[TZ_NUMRELS] =
            INT_TO_HD ( i );
        Resize( hdRels, ( i + 1 ) * SIZE_HD );
        Resize( hdLens, ( i + 1 ) * SIZE_HD );
        Resize( hdFlags, ( i + 1 ) * SIZE_HD );
    }

    return HdVoid;
}


/****************************************************************************
**
*F  FunTzRenumberGens(<hdCall>)  . . . . . . . renumber the Tietze generators
*/
Bag       FunTzRenumberGens (Bag hdCall)
{
    Bag           hdTie;          /* handle of the Tietze stack      */
    Bag           * ptTie;        /* pointer to this stack           */
    Bag           hdRels;         /* handle of the relators list     */
    Bag           * ptRels;       /* pointer to this list            */
    Bag           hdInvs;         /* handle of the inverses list     */
    Bag           * ptInvs;       /* pointer to this list            */
    Bag           * ptRel;        /* pointer to the ith relator      */
    Int                numgens;        /* number of Tietze generators     */
    Int                numrels;        /* number of Tietze relators       */
    Int                old;            /* generator or inverse            */
    Int                leng;           /* relator length                  */
    Int                i, j;           /* loop variables                  */

    /*  Get and check arguments                                            */
    if ( GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
        return Error( "usage: TzRenumberGens(<Tietze stack>)",  0,  0 );

    /*  Check the Tietze stack                                             */
    hdTie = EVAL( PTR_BAG(hdCall)[1] );
    if (GET_TYPE_BAG(hdTie) != T_LIST || HD_TO_INT(PTR_BAG(hdTie)[0]) != TZ_LENGTHTIETZE)
        return Error( "invalid <Tietze stack>",  0,  0 );
    ptTie = PTR_BAG( hdTie );

    /*  Get and check the Tietze relators list                             */
    hdRels = ptTie[TZ_RELATORS];
    numrels = HD_TO_INT(ptTie[TZ_NUMRELS]);
    if ( hdRels == 0
      || (GET_TYPE_BAG(hdRels) != T_LIST && GET_TYPE_BAG(hdRels) != T_SET)
      || HD_TO_INT(PTR_BAG(hdRels)[0]) != numrels )
        return Error( "invalid Tietze relators list",  0,  0 );
    ptRels = PTR_BAG( hdRels );

    /*  Get and check the Tietze inverses list                             */
    hdInvs = ptTie[TZ_INVERSES];
    numgens = HD_TO_INT(ptTie[TZ_NUMGENS]);
    if ( hdInvs == 0
      || (GET_TYPE_BAG(hdInvs) != T_LIST && GET_TYPE_BAG(hdInvs) != T_VECTOR)
      || HD_TO_INT(PTR_BAG(hdInvs)[0]) != 2 * numgens + 1 )
        return Error( "invalid Tietze inverses list",  0,  0 );
    ptInvs = PTR_BAG( hdInvs ) + numgens + 1;

    /*  Loop over all relators and replace the occurring generators        */
    for ( i = 1; i <= numrels; i++ ) {
        ptRel = PTR_BAG( ptRels[i] );
        leng = HD_TO_INT( ptRel[0] );
        /* Run through the relator and replace the occurring generators    */
        for ( j = 1; j <= leng; j++ ) {
            old = HD_TO_INT( ptRel[j] );
            if ( old < -numgens || numgens < old || old == 0 )
                return Error( "gen no. %d in rel no. %d out of range", j,i );
            ptRel[j] = ptInvs[-old];
        }
    }

    return HdVoid;
}


/****************************************************************************
**
*F  FunTzReplaceGens(<hdCall>)  . . . replace Tietze generators by other ones
*/
Bag       FunTzReplaceGens (Bag hdCall)
{
    Bag           hdTie;          /* handle of the Tietze stack      */
    Bag           * ptTie;        /* pointer to this stack           */
    Bag           hdRels;         /* handle of the relators list     */
    Bag           * ptRels;       /* pointer to this list            */
    Bag           hdLens;         /* handle of the lengths list      */
    Bag           * ptLens;       /* pointer to this list            */
    Bag           hdFlags;        /* handle of the flags list        */
    Bag           * ptFlags;      /* pointer to this list            */
    Bag           hdInvs;         /* handle of the inverses list     */
    Bag           * ptInvs;       /* pointer to this list            */
    Bag           hdRel;          /* handle of a relator             */
    Bag           * ptRel;        /* pointer to this relator         */
    Bag           * pt1, * pt2;   /* pointers to a relator           */
    Int                numgens;        /* number of Tietze generators     */
    Int                numrels;        /* number of Tietze relators       */
    Int                total;          /* total length of relators        */
    Int                old, new;       /* generators or inverses          */
    Int                leng, reduced;  /* relator lengths                 */
    Int                altered;        /* flag                            */
    Int                i, j;           /* loop variables                  */

    /*  Get and check arguments                                            */
    if ( GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
        return Error( "usage: TzReplaceGens(<Tietze stack>)",  0,  0 );

    /*  Check the Tietze stack                                             */
    hdTie = EVAL( PTR_BAG(hdCall)[1] );
    if (GET_TYPE_BAG(hdTie) != T_LIST || HD_TO_INT(PTR_BAG(hdTie)[0]) != TZ_LENGTHTIETZE)
        return Error( "invalid <Tietze stack>",  0,  0 );
    ptTie = PTR_BAG( hdTie );

    /*  Get and check the Tietze relators list                             */
    hdRels = ptTie[TZ_RELATORS];
    numrels = HD_TO_INT(ptTie[TZ_NUMRELS]);
    if ( hdRels == 0
      || (GET_TYPE_BAG(hdRels) != T_LIST && GET_TYPE_BAG(hdRels) != T_SET)
      || HD_TO_INT(PTR_BAG(hdRels)[0]) != numrels )
        return Error( "invalid Tietze relators list",  0,  0 );
    ptRels = PTR_BAG( hdRels );

    /*  Get and check the Tietze lengths list                              */
    hdLens = ptTie[TZ_LENGTHS];
    if ( hdLens == 0
      || (GET_TYPE_BAG(hdLens) != T_VECTOR && GET_TYPE_BAG(hdLens) != T_LIST)
      || HD_TO_INT(PTR_BAG(hdLens)[0]) != numrels )
        return Error( "invalid Tietze lengths list",  0,  0 );
    ptLens = PTR_BAG( hdLens );

    /*  Check list <lengths> to contain the relator lengths                */
    total = 0;
    for ( i = 1; i <= numrels; i++ ) {
        if ( ptRels[i] == 0
          || (GET_TYPE_BAG(ptRels[i]) != T_LIST && GET_TYPE_BAG(ptRels[i]) != T_SET
           && GET_TYPE_BAG(ptRels[i]) != T_VECTOR)
          || HD_TO_INT(ptLens[i]) != HD_TO_INT(PTR_BAG(ptRels[i])[0]) )
            return Error( "inconsistent Tietze lengths list entry [%d]",
               i,  0 );
        total += HD_TO_INT(ptLens[i]);
    }
    if ( total != HD_TO_INT(ptTie[TZ_TOTAL]) )
        return Error( "inconsistent total length",  0,  0 );

    /*  Get and check the Tietze flags list                                */
    hdFlags = ptTie[TZ_FLAGS];
    if ( hdFlags == 0
      || (GET_TYPE_BAG(hdFlags) != T_VECTOR && GET_TYPE_BAG(hdFlags) != T_LIST)
      || HD_TO_INT(PTR_BAG(hdFlags)[0]) != numrels )
        return Error( "invalid Tietze flags list",  0,  0 );
    ptFlags = PTR_BAG( hdFlags );

    /*  Get and check the Tietze inverses list                             */
    hdInvs = ptTie[TZ_INVERSES];
    numgens = HD_TO_INT(ptTie[TZ_NUMGENS]);
    if ( hdInvs == 0
      || (GET_TYPE_BAG(hdInvs) != T_LIST && GET_TYPE_BAG(hdInvs) != T_VECTOR)
      || HD_TO_INT(PTR_BAG(hdInvs)[0]) != 2 * numgens + 1 )
        return Error( "invalid Tietze inverses list",  0,  0 );
    ptInvs = PTR_BAG( hdInvs ) + numgens + 1;

    /*  Loop over all relators                                             */
    for ( i = 1; i <= numrels; i++ ) {
        hdRel = ptRels[i];
        pt2 = ptRel = PTR_BAG( hdRel );
        leng = HD_TO_INT( ptLens[i] );
        altered = 0;

        /* Don't change a sqare relator defining a valid involution        */
        if ( HD_TO_INT( ptFlags[i] ) == 3 && leng == 2 &&
            ptRel[1] == ptInvs[-HD_TO_INT(ptRel[1])] ) {
            continue;  /*  loop over i  */
        }

        /* Run through the relator and replace the occurring generators    */
        for ( j = 1; j <= leng; j++ ) {
            old = HD_TO_INT( ptRel[j] );
            if ( old < -numgens || numgens < old || old == 0 )
                return Error( "gen no. %d in rel no. %d out of range", j,i );
            new = HD_TO_INT( ptInvs[-old] );
            if ( !new ) {
                altered = 1;
                continue;  /*  loop over j  */
            }

            if ( pt2 > ptRel && *pt2 == ptInvs[new] ) {
                altered = 1;
                --pt2;
            }
            else {
                if ( new != old )  { altered = 1; }
                *++pt2 = INT_TO_HD( new );
            }
        }

        if ( !altered ) { continue; }  /*  loop over i  */

        /*  Now cyclically reduce the relator                              */
        pt1 = ++ptRel;
        while ( pt1 < pt2 && *pt1 == ptInvs[HD_TO_INT(*pt2)] ) {
            ++pt1;  --pt2;
        }
        if ( ptRel < pt1 ) {
            while ( pt1 <= pt2 )  { *ptRel++ = *pt1++; }
            pt2 = --ptRel;
        }

        /*  Resize the resulting relator, if necessary                     */
        ptRel = PTR_BAG( hdRel );
        reduced = pt2 - ptRel;
        if ( reduced < leng ) {
            ptRel[0] = INT_TO_HD( reduced );
            ptLens[i] = INT_TO_HD( reduced );
            total = total - leng + reduced;
            Resize( hdRel, ( reduced + 1 ) * SIZE_HD );
            ptRels = PTR_BAG( hdRels );
            ptLens = PTR_BAG( hdLens );
            ptFlags = PTR_BAG( hdFlags );
            ptInvs = PTR_BAG( hdInvs ) + numgens + 1;
        }

        /*  Redefine the corresponding search flag                         */
        SET_BAG( hdFlags , i,  INT_TO_HD( 1 ) );
    }

    ptTie = PTR_BAG( hdTie );
    ptTie[TZ_TOTAL] = INT_TO_HD( total );

    return HdVoid;
}


/****************************************************************************
**
*F  FunTzSubstituteGen(<hdCall>)  . . replace a Tietze generator by some word
*/
Bag       FunTzSubstituteGen (Bag hdCall)
{
    Bag           hdTie;          /* handle of the Tietze stack      */
    Bag           * ptTie;        /* pointer to this stack           */
    Bag           hdRels;         /* handle of the relators list     */
    Bag           * ptRels;       /* pointer to this list            */
    Bag           hdLens;         /* handle of the lengths list      */
    Bag           * ptLens;       /* pointer to this list            */
    Bag           hdFlags;        /* handle of the flags list        */
    Bag           hdInvs;         /* handle of the inverses list     */
    Bag           * ptInvs;       /* pointer to this list            */
    Bag           hdWrd;          /* handle of the replacing word    */
    Bag           * ptWrd;        /* pointer to this word            */
    Bag           hdIwrd;         /* handle of the inverse word      */
    Bag           * ptIwrd;       /* pointer to this word            */
    Bag           hdNew;          /* handle of a modified relator    */
    Bag           * ptNew;        /* pointer to this relator         */
    Bag           hdRel;          /* handle of a relator             */
    Bag           * ptRel;        /* pointer to this relator         */
    Bag           * pt1, * pt2;   /* pointers to a relator           */
    Bag           * pt3;          /* pointer to a relator            */
    Int                numgens;        /* number of Tietze generators     */
    Int                numrels;        /* number of Tietze relators       */
    Int                total;          /* total length of relators        */
    Int                given;          /* given generator and inverse     */
    Int                gen, ginv;      /* given generator and inverse     */
    Int                next;           /* generator or inverse            */
    Int                leng, newleng;  /* relator lengths                 */
    Int                wleng;          /* length of the replacing word    */
    Int                occ;            /* number of occurrences           */
    Int                i, j;           /* loop variables                  */

    /*  Get and check arguments                                            */
    if ( GET_SIZE_BAG(hdCall) != 4*SIZE_HD ) return Error( 
        "usage: TzSubstituteGen(<Tietze stack>,<gen no.>,<Tietze word>)",
         0,  0 );

    hdTie = EVAL( PTR_BAG(hdCall)[1] );
    given = HD_TO_INT( EVAL( PTR_BAG(hdCall)[2] ) );
    hdWrd = EVAL( PTR_BAG(hdCall)[3] );

    /*  Check the first argument (Tietze stack)                            */
    if (GET_TYPE_BAG(hdTie) != T_LIST || HD_TO_INT(PTR_BAG(hdTie)[0]) != TZ_LENGTHTIETZE)
        return Error( "invalid <Tietze stack>",  0,  0 );
    ptTie = PTR_BAG( hdTie );

    /*  Get and check the Tietze relators list                             */
    hdRels = ptTie[TZ_RELATORS];
    numrels = HD_TO_INT(ptTie[TZ_NUMRELS]);
    if ( hdRels == 0
      || (GET_TYPE_BAG(hdRels) != T_LIST && GET_TYPE_BAG(hdRels) != T_SET)
      || HD_TO_INT(PTR_BAG(hdRels)[0]) != numrels )
        return Error( "invalid Tietze relators list",  0,  0 );
    ptRels = PTR_BAG( hdRels );

    /*  Get and check the Tietze lengths list                              */
    hdLens = ptTie[TZ_LENGTHS];
    if ( hdLens == 0
      || (GET_TYPE_BAG(hdLens) != T_VECTOR && GET_TYPE_BAG(hdLens) != T_LIST)
      || HD_TO_INT(PTR_BAG(hdLens)[0]) != numrels )
        return Error( "invalid Tietze lengths list",  0,  0 );
    ptLens = PTR_BAG( hdLens );

    /*  Get and check the Tietze flags list                                */
    hdFlags = ptTie[TZ_FLAGS];
    if ( hdFlags == 0
      || (GET_TYPE_BAG(hdFlags) != T_VECTOR && GET_TYPE_BAG(hdFlags) != T_LIST)
      || HD_TO_INT(PTR_BAG(hdFlags)[0]) != numrels )
        return Error( "invalid Tietze flags list",  0,  0 );

    /*  Get and check the inverses list                                    */
    hdInvs = ptTie[TZ_INVERSES];
    numgens = HD_TO_INT(ptTie[TZ_NUMGENS]);
    if ( hdInvs == 0
      || (GET_TYPE_BAG(hdInvs) != T_LIST && GET_TYPE_BAG(hdInvs) != T_VECTOR)
      || HD_TO_INT(PTR_BAG(hdInvs)[0]) != 2 * numgens + 1 )
        return Error( "invalid Tietze inverses list",  0,  0 );
    ptInvs = PTR_BAG( hdInvs ) + numgens + 1;

    /*  Check the second argument (generator number)                       */
    gen = ( given > 0 ) ? given : -given;
    if ( gen <= 0 || numgens < gen )
        return Error( "generator number %d out of range", gen,  0 );
    ginv = HD_TO_INT(ptInvs[gen]);

    /*  Check the third argument (replacing word)                          */
    if ( GET_TYPE_BAG(hdWrd) != T_LIST && GET_TYPE_BAG(hdWrd) != T_VECTOR )
        return Error( "invalid replacing word",  0,  0 );
    ptWrd = PTR_BAG( hdWrd );
    wleng = HD_TO_INT( ptWrd[0] );
    for ( i = 1; i <= wleng; i++ ) {
        next = HD_TO_INT( ptWrd[i] );
        if ( next < -numgens || next == 0 || next > numgens )
            return Error( "entry [%d] of <Tietze word> out of range", i, 0 );
    }

    /*  Check list <lengths> to contain the relator lengths                */
    total = 0;
    for ( i = 1; i <= numrels; i++ ) {
        if ( ptRels[i] == 0
          || (GET_TYPE_BAG(ptRels[i]) != T_LIST && GET_TYPE_BAG(ptRels[i]) != T_SET
           && GET_TYPE_BAG(ptRels[i]) != T_VECTOR)
          || HD_TO_INT(ptLens[i]) != HD_TO_INT(PTR_BAG(ptRels[i])[0]) )
            return Error( "inconsistent Tietze lengths list",  0,  0 );
        total += HD_TO_INT(ptLens[i]);
    }
    if ( total != HD_TO_INT(ptTie[TZ_TOTAL]) )
        return Error( "inconsistent total length",  0,  0 );

    /*  Allocate a bag for the inverse of the replacing word               */
    hdIwrd = NewBag( T_LIST, (wleng + 1) * SIZE_HD );
    ptRels = PTR_BAG( hdRels );
    ptLens = PTR_BAG( hdLens );
    ptInvs = PTR_BAG( hdInvs ) + numgens + 1;
    ptWrd = PTR_BAG( hdWrd );
    ptIwrd = PTR_BAG( hdIwrd );

    /*  Invert the replacing word                                          */
    ptIwrd[0] = INT_TO_HD( wleng );
    pt1 = ptWrd;  pt2 = ptIwrd + wleng;
    while ( pt2 > ptIwrd )
        *pt2-- = ptInvs[HD_TO_INT(*++pt1)];
    if ( given < 0 ) {
        hdNew = hdWrd;  hdWrd = hdIwrd;  hdIwrd = hdNew;
        ptWrd = PTR_BAG( hdWrd );  ptIwrd = PTR_BAG( hdIwrd );
    }

    /*  Loop over all relators                                             */
    for ( i = 1; i <= numrels; i++ ) {
        hdRel = ptRels[i];
        ptRel = PTR_BAG( hdRel );
        leng = HD_TO_INT( ptLens[i] );
        if ( leng == 0 )  { continue; }

        /* Run through the relator and count the occurrences of gen        */
        occ = 0;
        for ( j = 1; j <= leng; j++ ) {
            next = HD_TO_INT( ptRel[j] );
            if ( next < -numgens || numgens < next )
                return Error( "gen no. %d in rel no. %d out of range", j,i );
            if (next == gen || next == ginv ) ++occ;
        }
        if ( occ == 0 )  { continue; }

        /*  Allocate a bag for the modified Tietze relator                 */
        hdNew = NewBag( T_LIST, (leng + occ * (wleng - 1) + 1) * SIZE_HD );
        ptLens = PTR_BAG( hdLens );
        ptInvs = PTR_BAG( hdInvs ) + numgens + 1;
        ptWrd = PTR_BAG( hdWrd );
        ptIwrd = PTR_BAG( hdIwrd );
        ptRel = PTR_BAG( hdRel );
        pt2 = ptNew = PTR_BAG( hdNew );

        /*  Now run again through the relator and modify it                */
        for ( j = 1; j <= leng; j++ ) {
            next = HD_TO_INT( ptRel[j] );
            if ( next == gen || next == -gen ) {
                pt1 = ( next > 0 ) ? ptWrd : ptIwrd;
                pt3 = pt1 + wleng;
                while ( pt1 < pt3 ) {
                    ++pt1;
                    if ( pt2 > ptNew && *pt2 == ptInvs[HD_TO_INT(*pt1)] )
                        --pt2;
                    else
                        *++pt2 = *pt1;
                }
            }
            else {
                if ( pt2 > ptNew && *pt2 == ptInvs[next] )
                    --pt2;
                else
                    *++pt2 = INT_TO_HD( next );
            }
        }

        /*  Now cyclically reduce the relator                              */
        pt1 = ++ptNew;
        while ( pt1 < pt2 && *pt1 == ptInvs[HD_TO_INT(*pt2)] ) {
            ++pt1;  --pt2;
        }
        if ( ptNew < pt1 ) {
            while ( pt1 <= pt2 )   *ptNew++ = *pt1++;
            pt2 = --ptNew;
        }

        /*  Resize and save the resulting relator                          */
        ptNew = PTR_BAG( hdNew );
        newleng = pt2 - ptNew;
        ptNew[0] = INT_TO_HD( newleng );
        ptLens[i] = INT_TO_HD( newleng );
        total = total - leng + newleng;
        Resize( hdNew, ( newleng + 1 ) * SIZE_HD );
        ptRels = PTR_BAG( hdRels );
        ptLens = PTR_BAG( hdLens );
        ptRels[i] = hdNew;
        SET_BAG( hdFlags , i,  INT_TO_HD( 1 ) );
    }

    ptTie = PTR_BAG( hdTie );
    ptTie[TZ_TOTAL] = INT_TO_HD( total );

    return HdVoid;
}


/****************************************************************************
**
*F  FunTzOccurrences( <hdCall> ) . . . . . . occurrences of Tietze generators
**
**  'FunTzOccurrences' implements the internal function 'TzOccurrences'.
*/
Bag       FunTzOccurrences (Bag hdCall)
{
    Bag           hdTie;          /* handle of the Tietze stack      */
    Bag           * ptTie;        /* pointer to the Tietze stack     */
    Bag           hdRels;         /* handle of the relators list     */
    Bag           * ptRels;       /* pointer to this list            */
    Bag           hdRes;          /* handle of the result            */
    Bag           hdCnts;         /* list of the counts              */
    Bag           * ptCnts;       /* pointer to the counts list      */
    Bag           hdMins;         /* list of minimal occurence list  */
    Bag           * ptMins;       /* pointer to the minimals list    */
    Bag           hdLens;         /* list of lengths of those        */
    Bag           * ptLens;       /* pointer to the lengths list     */
    Bag           hdRel;          /* handle of a relator             */
    Bag           * ptRel;        /* pointer to this relator         */
    Bag           hdAux;          /* auxiliary list                  */
    Int                * ptAux;        /* pointer to the lengths list     */
    Int                numgens;        /* number of Tietze generators     */
    Int                numrels;        /* number of Tietze relators       */
    Int                leng;           /* length of a relator             */
    Int                num, next;      /* generators or inverses          */
    Int                i, k;           /* loop variables                  */
    Int                c;              /* count of one generator          */
    Int                nr;             /* number of occurrences           */
    Int                nr1;            /* nr of occurrences in one word   */
    Int                nrm;            /* minimal value of 'nr1'          */
    Int                min;            /* word that has this minimum      */

    /* Get and check arguments                                             */
    if ( GET_SIZE_BAG(hdCall) != 2*SIZE_HD && GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error("usage: TzOccurrences( <Tietze stack> [, <gen no.> ] )",
             0,  0 );

    /*  Check the first argument (Tietze stack)                            */
    hdTie = EVAL( PTR_BAG(hdCall)[1] );
    if (GET_TYPE_BAG(hdTie) != T_LIST || HD_TO_INT(PTR_BAG(hdTie)[0]) != TZ_LENGTHTIETZE)
        return Error( "invalid <Tietze stack>",  0,  0 );
    ptTie = PTR_BAG( hdTie );

    /*  Get and check the Tietze relators list                             */
    numgens = HD_TO_INT(ptTie[TZ_NUMGENS]);
    numrels = HD_TO_INT(ptTie[TZ_NUMRELS]);
    hdRels = ptTie[TZ_RELATORS];
    if ( hdRels == 0
      || (GET_TYPE_BAG(hdRels) != T_LIST && GET_TYPE_BAG(hdRels) != T_SET)
      || HD_TO_INT(PTR_BAG(hdRels)[0]) != numrels )
        return Error( "invalid Tietze relators list",  0,  0 );
    ptRels = PTR_BAG( hdRels );

    /*  Get and check the given generator number                           */
    if ( GET_SIZE_BAG(hdCall) == 3*SIZE_HD ) {
        num = HD_TO_INT( EVAL( PTR_BAG(hdCall)[2] ) );
        if ( num <= 0 || numgens < num )
            return Error( "given generator number out of range",  0,  0 );
        numgens = 1;
    }
    else  num = numgens;

    /* the pointers may have been corrupted by the EVAL */
    ptTie = PTR_BAG( hdTie );
    hdRels = ptTie[TZ_RELATORS];

    /* allocate the result lists                                           */
    hdCnts = NewBag( T_LIST, SIZE_HD + numgens * SIZE_HD );
    SET_BAG(hdCnts, 0,  INT_TO_HD( numgens ) );
    for ( k = 1; k <= numgens; k++ )
        SET_BAG(hdCnts, k,  INT_TO_HD( 0 ) );
    hdMins = NewBag( T_LIST, SIZE_HD + numgens * SIZE_HD );
    SET_BAG(hdMins, 0,  INT_TO_HD( 0 ) );
    hdLens = NewBag( T_LIST, SIZE_HD + numgens * SIZE_HD );
    SET_BAG(hdLens, 0,  INT_TO_HD( 0 ) );
    hdRes = NewBag( T_LIST, SIZE_HD + 3 * SIZE_HD );
    SET_BAG(hdRes, 0,  INT_TO_HD( 3 ) );
    SET_BAG(hdRes, 1,  hdCnts );
    SET_BAG(hdRes, 2,  hdMins );
    SET_BAG(hdRes, 3,  hdLens );

    /* allocate an auxiliary list                                          */
    ptAux = 0;
    if ( numgens > 1 ) {
        hdAux = NewBag( T_STRING, ( numgens + 1 ) * sizeof( Int ) );
        ptAux = (Int*)PTR_BAG(hdAux);
        ptAux[0] = numgens;
        for ( k = 1; k <= numgens; k++ )
            ptAux[k] = 0;

    }

    /* now we can safely grab pointers                                     */
    ptRels = PTR_BAG(hdRels);
    ptCnts = PTR_BAG(hdCnts);
    ptLens = PTR_BAG(hdLens);
    ptMins = PTR_BAG(hdMins);

    /* handle special case of single generator in generator list           */
    if ( numgens == 1 ) {

        /* initialize the counters                                         */
        nr = 0;  nrm = 0;  min = 0;

        /* loop over all relators                                          */
        for ( i = 1; i <= numrels; i++ ) {
            hdRel = ptRels[i];
            if ( hdRel == 0
              || (GET_TYPE_BAG(hdRel) != T_LIST && GET_TYPE_BAG(hdRel) != T_SET
               && GET_TYPE_BAG(hdRel) != T_VECTOR) )
                return Error("invalid entry [%d] in Tietze relators list",
                             i, 0);
            ptRel = PTR_BAG(hdRel);
            leng = HD_TO_INT( ptRel[0] );

            /* loop over the letters of the relator                        */
            nr1 = 0;
            for ( k = 1; k <= leng; k++ ) {
                next = HD_TO_INT( ptRel[k] );
                if ( next == num || next == -num )  { nr1++; }
            }

            /* check whether the number of occurrences of num is less than */
            /* in the preceding relators                                   */
            nr += nr1;
            if ( nrm == 0
              || (0 < nr1 && nr1 < nrm)
              || (nr1 == nrm && GET_SIZE_BAG(hdRel) < GET_SIZE_BAG(ptRels[min])) ) {
                nrm = nr1;  min = i;
            }
        }

        /* put the information into the result bags                        */
        ptCnts[1] = INT_TO_HD( nr );
        if ( nr != 0 ) {
            ptCnts[1] = INT_TO_HD( nr );
            ptLens[0] = INT_TO_HD( 1 );  ptLens[1] = INT_TO_HD( nrm );
            ptMins[0] = INT_TO_HD( 1 );  ptMins[1] = INT_TO_HD( min );
        }
    }

    /* handle general case of all Tietze generators                        */
    else {

        /* loop over all relators                                          */
        for ( i = 1; i <= numrels; i++ ) {
            hdRel = ptRels[i];
            if ( hdRel == 0
              || (GET_TYPE_BAG(hdRel) != T_LIST && GET_TYPE_BAG(hdRel) != T_SET
               && GET_TYPE_BAG(hdRel) != T_VECTOR) )
                return Error("invalid entry [%d] in Tietze relators list",
                             i, 0);
            ptRel = PTR_BAG(hdRel);
            leng = HD_TO_INT( ptRel[0] );

            /* loop over the letters of the relator                        */
            for ( k = 1; k <= leng; k++ ) {
                next = HD_TO_INT( ptRel[k] );
                if ( next < 0 ) next = -next;
                if ( next == 0 || numgens < next ) return Error(
                    "invalid entry [%d][%d] in Tietze relators list",i,k );
                (ptAux[next])++;
            }

            /* loop over the generators, collecting the counts             */
            for ( k = 1; k <= numgens; k++ ) {
                c = ptAux[k];
                if ( !c )  continue;
                ptAux[k] = 0;
                ptCnts[k] = INT_TO_HD( HD_TO_INT( ptCnts[k] ) + c );
                if ( (0 < c && ptLens[k] == 0)
                  || (0 < c && c < HD_TO_INT(ptLens[k]))
                  || (0 < c && c == HD_TO_INT(ptLens[k])
                    && GET_SIZE_BAG(hdRel) < GET_SIZE_BAG(ptRels[HD_TO_INT(ptMins[k])])) ) {
                    ptLens[k] = INT_TO_HD( c );
                    ptMins[k] = INT_TO_HD( i );
                }
            }
        }

        /* find the correct length of the minimals and lengths lists       */
        k = numgens;
        while ( ptMins[k] == 0 )  k--;
        SET_BAG(hdMins, 0,  INT_TO_HD( k ) );
        SET_BAG(hdLens, 0,  INT_TO_HD( k ) );
    }

    /* return the result                                                   */
    return hdRes;
}


/****************************************************************************
**
*F  FunTzOccurrencesPairs(<hdCall>) occurrences of pairs of Tietze generators
*/
Bag       FunTzOccurrencesPairs (Bag hdCall)
{
    Bag           hdTie;          /* handle of the Tietze stack      */
    Bag           * ptTie;        /* pointer to the Tietze stack     */
    Bag           hdRels;         /* handle of the relators list     */
    Bag           * ptRels;       /* pointer to this list            */
    Bag           hdInvs;         /* handle of the inverses list     */
    Bag           * ptInvs;       /* pointer to this list            */
    Bag           hdRes;          /* handle of the resulting list    */
    Bag           * ptRes;        /* pointer to this list            */
    Bag           hdRel;          /* handle of a relator             */
    Bag           * ptRel;        /* pointer to this relator         */
    Bag           hdNum;          /* handle of generator number      */
    Bag           hdInv;          /* handle of inverse gen number    */
    Int                num, i, ii;     /* generator numbers               */
    Int                numgens;        /* number of Tietze generators     */
    Int                numrels;        /* number of Tietze relators       */
    Int                length;         /* length of the current relator   */
    Int                j1, j2, r;      /* loop variables                  */

    /* Get and check arguments                                             */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD && GET_SIZE_BAG(hdCall) != 4*SIZE_HD ) return
        Error(
            "usage: TzOccurrencesPairs( <Tietze stack>, <gen> [, <list> ] )",
             0,  0 );

    hdTie = EVAL( PTR_BAG(hdCall)[1] );
    hdNum = EVAL( PTR_BAG(hdCall)[2] );

    /*  Check the first argument (Tietze stack)                            */
    if (GET_TYPE_BAG(hdTie) != T_LIST || HD_TO_INT(PTR_BAG(hdTie)[0]) != TZ_LENGTHTIETZE)
        return Error( "invalid <Tietze stack>",  0,  0 );
    ptTie = PTR_BAG( hdTie );

    /*  Get and check the Tietze relators list                             */
    hdRels = ptTie[TZ_RELATORS];
    numrels = HD_TO_INT(ptTie[TZ_NUMRELS]);
    if ( hdRels == 0
      || (GET_TYPE_BAG(hdRels) != T_LIST && GET_TYPE_BAG(hdRels) != T_SET)
      || HD_TO_INT(PTR_BAG(hdRels)[0]) != numrels )
        return Error( "invalid Tietze relators list",  0,  0 );

    /*  Get and check the Tietze inverses list                             */
    hdInvs = ptTie[TZ_INVERSES];
    numgens = HD_TO_INT(ptTie[TZ_NUMGENS]);
    if ( hdInvs == 0
      || (GET_TYPE_BAG(hdInvs) != T_LIST && GET_TYPE_BAG(hdInvs) != T_VECTOR)
      || HD_TO_INT(PTR_BAG(hdInvs)[0]) != 2 * numgens + 1 )
        return Error( "invalid Tietze inverses list",  0,  0 );

    /*  Get and check the Tietze generator number                          */
    if ( GET_TYPE_BAG(hdNum) != T_INT )
        return Error( "<gen> must be a Tietze generator number",  0,  0 );
    num = HD_TO_INT( hdNum );
    if ( num <= 0 || num > numgens )
        return Error( "given generator number is out of range",  0,  0 );

    /*  Get and check the list for the results, if specified               */
    if ( GET_SIZE_BAG(hdCall) == 3* SIZE_HD ) {
        hdRes = NewBag( T_LIST, SIZE_HD + 4 * numgens * SIZE_HD );
        SET_BAG( hdRes , 0,  INT_TO_HD( 4 * numgens ) );
    }
    else {
        hdRes = EVAL( PTR_BAG(hdCall)[3] );
        if ( hdRes == 0
          || (GET_TYPE_BAG(hdRes) != T_LIST && GET_TYPE_BAG(hdRes) != T_VECTOR)
          || HD_TO_INT(PTR_BAG(hdRes)[0]) != 4 * numgens )
            return Error(
                "<list> must be a list of length 4 * number of generators",
                          0,  0 );
    }

    /* the EVAL may have corrupted the pointer */
    ptTie = PTR_BAG( hdTie );
    hdRels = ptTie[TZ_RELATORS];
    hdInvs = ptTie[TZ_INVERSES];

    /*  return, if num = numgens                                           */
    if ( num == numgens )  { return hdRes; }

    /*  get pointers to the involved lists                                 */
    ptRels = PTR_BAG( hdRels );
    ptInvs = PTR_BAG( hdInvs ) + numgens + 1;
    ptRes = PTR_BAG( hdRes );

    /* get the handle of the inverse of the given generator                */
    hdInv = ptInvs[num];

    /* ptRes[i]           counts the occurrences of gen * gen[i]           */
    /* ptRes[numgens+i]   counts the occurrences of gen * gen[i]^-1        */
    /* ptRes[2*numgens+i] counts the occurrences of gen^-1 * gen[i]        */
    /* ptRes[3*numgens+i] counts the occurrences of gen^-1 * gen[i]^-1     */

    /* initialize the counters                                             */
    for ( i = 1; i <= 4 * numgens; i++ ) {
        ptRes[i] = INT_TO_HD( 0 );
    }

    /* loop over the Tietze relators                                       */
    for ( r = 1; r <= numrels; r++ ) {
        hdRel = ptRels[r];
        if ( hdRel == 0
          || (GET_TYPE_BAG(hdRel) != T_LIST && GET_TYPE_BAG(hdRel) != T_SET
           && GET_TYPE_BAG(hdRel) != T_VECTOR) )
            return Error( "invalid Tietze relator [%d]",  0,  0 );
        ptRel = PTR_BAG( hdRel ) + 1;

        /* skip the current relator if its length is less than 2           */
        length = HD_TO_INT( ptRel[-1] );
        if ( length < 2 )  { continue; }

        /* loop over the current relator and investigate the pairs         */
        /* ( ptRel[j1], ptRel[j2] )                                        */
        j1 = ( length - 1 ) % length;
        for ( j2 = 0; j2 < length; j1 = j2, j2++ ) {

            /* count any "forward" pair  gen * gen[i],  gen * gen[i]^-1,   */
            /* gen^-1 * gen[i],  or  gen^-1 * gen[i]^-1  ( with num < i )  */
            if ( ptRel[j1] == hdNum || ptRel[j1] == hdInv ) {
                i = HD_TO_INT( ptRel[j2] );
                if ( -num <= i && i <= num )  { continue; }
	        if ( i < - numgens || numgens < i ) return Error(
                    "invalid entry %d in <Tietze relator> [%d]", i, r );
                if ( i < 0 ) i = numgens - i;
                if ( ptRel[j1] != hdNum ) i = i + 2 * numgens;
                ptRes[i] = INT_TO_HD( HD_TO_INT( ptRes[i] ) + 1 );
            }

            /* count any "backward" pair  gen[i]^-1 * gen^-1,              */
            /* gen[i] * gen^-1,  gen[i]^-1 * gen,  or  gen[i] * gen        */
            /* ( with num < i )  which is not covered by a forward pair    */
            else if ( ptRel[j2] == hdNum || ptRel[j2] == hdInv ) {
                i = HD_TO_INT( ptRel[j1] );
                if ( -num <= i && i <= num )  { continue; }
	        if ( i < - numgens || numgens < i ) return Error(
                    "invalid entry %d in <Tietze relator> [%d]", i, r );
                ii = HD_TO_INT( ptInvs[i] );
                if ( !( (hdNum == hdInv
                        && ptRel[(j2+1)%length] == INT_TO_HD(ii))
                     || (i == ii
                        && ptInvs[HD_TO_INT(ptRel[(j1+length-1)%length])] 
                           == ptRel[j2]) ) ) {
                    if ( ii < 0 ) ii = numgens - ii;
                    if ( ptRel[j2] != hdInv ) ii = ii + 2 * numgens;
                    ptRes[ii] = INT_TO_HD( HD_TO_INT( ptRes[ii] ) + 1 );
                }
            }
        }
    }

    return hdRes;
}


/****************************************************************************
**
*F  FunTzSearchC(<hdCall>) . . . . .  find subword matches in Tietze relators
**
**  'FunTzSearchC' implements the internal function 'TzSearchC'.
*/

#define TZKEYSZ 8192

Bag       FunTzSearchC (Bag hdCall)
{
    Bag           hdTie;          /* handle of the Tietze stack      */
    Bag           * ptTie;        /* pointer to this stack           */
    Bag           hdRels;         /* handle of the relators list     */
    Bag           * ptRels;       /* pointer to this list            */
    Bag           hdLens;         /* handle of the lengths list      */
    Bag           * ptLens;       /* pointer to this list            */
    Bag           hdInvs;         /* handle of the inverses list     */
    Bag           * ptInvs;       /* pointer to this list            */
    Bag           hdFlags;        /* handle of the flags list        */
    Bag           * ptFlags;      /* pointer to this list            */
    Bag           hdWrd;          /* handle of the given relator     */
    Bag           hdl;            /* handle of current list relator  */
    Bag           hdw;            /* handle of a relator             */
    Bag           hdPos1;         /* handle of the second argument   */
    Bag           hdPos2;         /* handle of the third argument    */
    Bag           hdEqu;          /* handle of the fifth argument    */
    char                keys1[TZKEYSZ], keys2[TZKEYSZ], keys3[TZKEYSZ];
                                        /* hash table of key values        */
    UInt       inv;            /* inverse for key computation     */
    UInt       key;            /* key value of subword            */
    Int                numgens;        /* number of Tietze generators     */
    Int                numrels;        /* number of Tietze relators       */
    Int                total;          /* total length of relators        */
    Bag           * ptr;          /* pointer to a relator            */
    Bag           * v,  * w;      /* pointers into relators          */
    Bag           * ptx,  * pty;  /* pointers into relators          */
    Int                i1, j1;         /* loop variables                  */
    Int                i2, j2;         /* loop variables                  */
    Int                i3, j3;         /* loop variables                  */
    Int                len1;           /* relator length                  */
    Int                lmin, lmax;     /* bound for relator lengths       */
    Int                pos1, pos2;     /* position of the given relator   */
    Int                xmax;           /* position of the given relator   */
    Int                newflag, flag1; /* Tietze relator flags            */
    Int                xflag, yflag;   /* Tietze relator flags            */
    Int                xlen, xlen1;    /* length of the given relator     */
    Int                mlen;           /* length of the wanted match      */
    Int                ylen, ylen1;    /* length of the current relator   */
    Int                newlen;         /* length of a new relator         */
    Int                n, m;           /* subword lengths                 */
    Int                count;          /* number of altered relators      */
    Int                i, j, jj, x, y; /* loop variables                  */
    Int                lasty;          /* flag                            */
    Int                altered;        /* flag                            */
    Int                equal;          /* flag                            */

    /* Get and check arguments                                             */
    if ( GET_SIZE_BAG(hdCall) != 4*SIZE_HD && GET_SIZE_BAG(hdCall) != 5*SIZE_HD )
        return Error(
            "usage: TzSearchC( <Tietze stack>, <pos1>, <pos1> [,<equal>] )",
             0,  0 );

    hdTie = EVAL( PTR_BAG(hdCall)[1] );
    hdPos1 = EVAL( PTR_BAG(hdCall)[2] );
    hdPos2 = EVAL( PTR_BAG(hdCall)[3] );
    hdEqu = (GET_SIZE_BAG(hdCall)==4*SIZE_HD) ? HdFalse : EVAL(PTR_BAG(hdCall)[4]);

    /*  Check the first argument (Tietze stack)                            */
    if (GET_TYPE_BAG(hdTie) != T_LIST || HD_TO_INT(PTR_BAG(hdTie)[0]) != TZ_LENGTHTIETZE)
        return Error( "invalid <Tietze stack>",  0,  0 );
    ptTie = PTR_BAG( hdTie );

    /*  Get and check the Tietze relators list                             */
    hdRels = ptTie[TZ_RELATORS];
    numrels = HD_TO_INT(ptTie[TZ_NUMRELS]);
    if ( hdRels == 0
      || (GET_TYPE_BAG(hdRels) != T_LIST && GET_TYPE_BAG(hdRels) != T_SET)
      || HD_TO_INT(PTR_BAG(hdRels)[0]) != numrels )
        return Error( "invalid Tietze relators list",  0,  0 );
    ptRels = PTR_BAG( hdRels );

    /*  Get and check the Tietze lengths list                              */
    hdLens = ptTie[TZ_LENGTHS];
    if ( hdLens == 0
      || (GET_TYPE_BAG(hdLens) != T_VECTOR && GET_TYPE_BAG(hdLens) != T_LIST)
      || HD_TO_INT(PTR_BAG(hdLens)[0]) != numrels )
        return Error( "invalid Tietze lengths list",  0,  0 );
    ptLens = PTR_BAG( hdLens );

    /*  Get and check the Tietze flags list                                */
    hdFlags = ptTie[TZ_FLAGS];
    if ( hdFlags == 0
      || (GET_TYPE_BAG(hdFlags) != T_VECTOR && GET_TYPE_BAG(hdFlags) != T_LIST)
      || HD_TO_INT(PTR_BAG(hdFlags)[0]) != numrels )
        return Error( "invalid Tietze flags list",  0,  0 );
    ptFlags = PTR_BAG( hdFlags );

    /*  Check list <lengths> to contain the relator lengths                */
    total = 0;
    for ( i = 1; i <= numrels; i++ ) {
        if ( ptRels[i] == 0
          || (GET_TYPE_BAG(ptRels[i]) != T_LIST && GET_TYPE_BAG(ptRels[i]) != T_SET
           && GET_TYPE_BAG(ptRels[i]) != T_VECTOR)
          || HD_TO_INT(ptLens[i]) != HD_TO_INT(PTR_BAG(ptRels[i])[0]) )
            return Error( "inconsistent Tietze lengths list entry [%d]",
                         i,  0 );
        total += HD_TO_INT(ptLens[i]);
    }
    if ( total != HD_TO_INT(ptTie[TZ_TOTAL]) )
        return Error( "inconsistent total length",  0,  0 );

    /*  Get and check the Tietze inverses list                             */
    hdInvs = ptTie[TZ_INVERSES];
    numgens = HD_TO_INT(ptTie[TZ_NUMGENS]);
    if ( hdInvs == 0
      || (GET_TYPE_BAG(hdInvs) != T_LIST && GET_TYPE_BAG(hdInvs) != T_VECTOR)
      || HD_TO_INT(PTR_BAG(hdInvs)[0]) != 2 * numgens + 1 )
        return Error( "invalid Tietze inverses list",  0,  0 );
    ptInvs = PTR_BAG( hdInvs ) + numgens + 1;

    /*  Check the second argument                                          */
    if ( GET_TYPE_BAG(hdPos1) != T_INT )
       return Error( "<position1> must be a posititve int",  0 , 0 );
    pos1 = HD_TO_INT( hdPos1 );
    if ( pos1 > numrels )
       return Error( "<position1> not in range: %d", pos1,  0 );

    /*  Check the third argument                                           */
    if ( GET_TYPE_BAG(hdPos2) != T_INT )
       return Error( "<position2> must be a posititve int",  0,  0 );
    pos2 = HD_TO_INT( hdPos2 );
    if ( pos2 > numrels )
       return Error( "<position2> not in range: %d", pos2,  0 );

    /*  Check the fourth argument                                          */
    if ( GET_TYPE_BAG(hdEqu) != T_BOOL )
       return Error( "<equal> must be false or true",  0,  0 );
    equal = ( hdEqu == HdTrue );

    /*  Skip relators of inconvenient lengths or with inconvenient flags,  */
    /*  and return if the remaining range is empty                         */
    while ( pos1 <= pos2
        && (HD_TO_INT( ptLens[pos1] ) < 2
         || HD_TO_INT( ptFlags[pos1] ) > 1
         || (equal && ( HD_TO_INT( ptLens[pos1] ) < 4
                     || HD_TO_INT( ptLens[pos1] ) % 2 == 1 ) ) ) )
        pos1++;
    if ( pos1 > pos2 || pos1 == numrels )  { return INT_TO_HD( 0 ); }

    /*  Get the range of compatible relator lengths                        */
    len1 = HD_TO_INT( ptLens[pos1] );
    lmin = len1 - ( len1 % 2 );
    lmax = ( equal ) ? lmin : lmin + 1;

    /*  Initialize some variables                                          */
    newflag = ( equal ) ? 1 : 2;
    count = 0;
    lasty = 0;
    xmax = pos1 - 1;
    flag1 = HD_TO_INT( ptFlags[pos1] );

    /* Compute the length of the wanted match and the corresponding        */
    /* inverse factor                                                      */
    mlen = equal ? ( lmin + 1 ) / 2 : lmin / 2 + 1;
    inv = 1;
    for ( i = 1; i <= mlen; i++ )
       inv = 109109 * inv;

    /* Initialize the hash table                                           */
    for ( i = 0; i < TZKEYSZ; i++ )
       keys1[i] = keys2[i] = keys3[i] = 0;

    /* Loop over the Tietze relators, starting at position pos1            */
    for ( y = pos1; y < numrels; ) {
       hdWrd = ptRels[y];
       ylen = HD_TO_INT( ptLens[y] );
       yflag = HD_TO_INT( ptFlags[y] );
       if ( y <= pos2 && lmin <= ylen && ylen <= lmax && yflag <= 1 ) {

          /* Add the key values of the current relator to the hash table   */
          ptr = PTR_BAG(hdWrd);
          key = 0;
          for ( i = 0, w = ptr+1; i < mlen; i++, w++ )
             key = 109109 * key + ( (UInt)*w >> 2 );
          for ( i = 0, v = ptr+1, w = v+mlen; i < ylen; i++, v++, w++ ) {
             keys1[ key & 8191 ] = 1;
             keys2[ (key >> 11) & 8191 ] |= (1 << ((key >> 8) & 7));
             keys3[ (key >> 19) & 8191 ] |= (1 << ((key >> 16) & 7));
             if ( i == ylen-mlen )  w = ptr+1;
             key = 109109 * key - inv * ( (UInt)*v >> 2 )
                 + ( (UInt)*w >> 2 );
          }
          key = 0;
          for ( i = 0, w = ptr+ylen; i < mlen; i++, w-- ) {
             key = 109109 * key
                 + ( (UInt) ptInvs[HD_TO_INT(*w)] >> 2 );
          }
          for ( i = 0, v = ptr+ylen, w = v-mlen; i < ylen; i++, v--, w-- ) {
             keys1[ key & 8191 ] = 1;
             keys2[ (key >> 11) & 8191 ] |= (1 << ((key >> 8) & 7));
             keys3[ (key >> 19) & 8191 ] |= (1 << ((key >> 16) & 7));
             if ( i == ylen-mlen )  w = ptr+ylen;
             key = 109109 * key
                 - inv * ( (UInt) ptInvs[HD_TO_INT(*v)] >> 2 )
                 + ( (UInt) ptInvs[HD_TO_INT(*w)] >> 2 );
          }
          if ( len1 > ylen ) len1 = ylen;
          if ( flag1 < yflag ) flag1 = yflag;
          xmax = y;
       }

       /* Move to next relator                                             */
       y++;

       /*  Initialize some variables                                       */
       hdl = ptRels[y];
       ylen = HD_TO_INT( ptLens[y] );
       yflag = HD_TO_INT( ptFlags[y] );
       ylen1 = ylen - 1;
       altered = 0;

       /* Loop to the next relator, if the current relator is too short    */
       if ( y > lasty
         && (ylen < len1 || yflag > 1 || (!equal && !(yflag + flag1)) ) )
          continue;  /*  loop over y                                       */
       lasty = y;

       /* Compute the key values of the current relator                    */
       ptr = PTR_BAG(hdl);
       key = 0;
       for ( j = 0, w = ptr+1; j < mlen; j++, w++ )
          key = 109109 * key + ( (UInt)*w >> 2 );
       for ( j = 0; j < ylen; j++ ) {

          /* Check for key match in the tables                             */
          if ( keys1[ key & 8191 ]
             && (keys2[ (key >> 11) & 8191 ] & (1 << ((key >> 8) & 7)))
             && (keys3[ (key >> 19) & 8191 ] & (1 << ((key >> 16) & 7))) ){

             /* Loop over the (relevant) given relators                    */
             for ( x = pos1; x <= xmax; x++ ) {

                hdw = ptRels[x];
                xlen = HD_TO_INT( ptLens[x] );
                xflag = HD_TO_INT( ptFlags[x] );
                if ( xlen < len1 || xlen > lmax || xlen > ylen
                  || xflag > 1 || (!equal && !( xflag + yflag )) )
                   continue;  /*  loop over x                              */

                xlen1 = xlen - 1;
                ptx = PTR_BAG(hdw) + 1;
                pty = PTR_BAG(hdl) + 1;

                /* Loop over all possible positions in the given relator   */
                for ( i = 0; i < xlen; i++ ) {

                   /* Search forward for a match of length at least mlen   */
                   i2 = i;  j2 = j;
                   for ( n = 0; n < xlen; n++,
                      i2 = (i2 == xlen1) ? 0 : i2 + 1,
                      j2 = (j2 == ylen1) ? 0 : j2 + 1 ) {
                      if ( ptx[i2] != pty[j2] )  break;  /*  loop over n   */
                   }
                   if ( n < mlen )  continue;  /*  loop over i             */

                   /* Search backward to find the whole match              */
                   i1 = (i == 0) ? xlen1 : i - 1;
                   j1 = (j == 0) ? ylen1 : j - 1;
                   for ( ; n < xlen; n++,
                      i1 = (i1 == 0) ? xlen1 : i1 - 1,
                      j1 = (j1 == 0) ? ylen1 : j1 - 1 ) {
                      if ( ptx[i1] != pty[j1] )  break;  /*  loop over n   */
                   }

                   /* Replace a matching substring of equal length         */
                   if ( n == xlen - n ) {
                      j2 = j;
                      for ( m = 0; m < n; m++,
                         i1 = (i1 == 0) ? xlen1 : i1 - 1,
                         j2 = (j2 == ylen1) ? 0 : j2 + 1 ) {
                         pty[j2] = ptInvs[HD_TO_INT(ptx[i1])];
                      }

                      /* Now replace all exact occurrences of this string  */
                      /* in the current word (not relator)                 */
                      i3 = (i + n) % xlen;

                      for ( jj = 0; jj <= ylen - n; jj++ ) {
                         i2 = i;  j2 = jj;
                         for ( m = 0; m < n; m++,
                            i2 = (i2 == xlen1) ? 0 : i2 + 1,
                            j2 = (j2 == ylen1) ? 0 : j2 + 1 ) {
                            if ( ptx[i2] != pty[j2] )
                                break;  /*  loop over m                    */
                         }
                         if ( m < n )  continue;  /*  loop over jj         */

                         i1 = (i == 0) ? xlen1 : i - 1;
                         if ( ptx[i1] == pty[(jj + ylen1) % ylen] ||
                            ptx[i3] == pty[(jj + n) % ylen] )
                            continue;  /*  loop over jj                    */

                         j2 = jj;
                         for ( m = 0; m < n; m++,
                            i1 = (i1 == 0) ? xlen1 : i1 - 1,
                            j2 = (j2 == ylen1) ? 0 : j2 + 1 ) {
                            pty[j2] = ptInvs[HD_TO_INT(ptx[i1])];
                         }

                         jj = -1;
                      }

                      ptFlags[y] = INT_TO_HD( newflag );
                      altered = 1;
                      ++count;
                      break;  /*  loop over i                              */
                   }

                   m = ylen - n;  n = xlen - n;

                   /* Find all canceling factors                           */
                   if ( n == 0 ) {
                      for ( ; 1 < m; m -= 2,
                         j1 = (j1 == 0) ? ylen1 : j1 - 1,
                         j2 = (j2 == ylen1) ? 0 : j2 + 1 ) {
                         if ( pty[j1] != ptInvs[HD_TO_INT(pty[j2])] )
                            break;  /*  loop over m                        */
                      }
                   }

                   /* Create the modified relator and save it              */
                   newlen = m + n;
                   if ( j2 > 0 ) {
                      if ( j2 <= j1 )  { jj = 0;  j3 = j1;  j1 = m - 1; }
                      else  { jj = j1 + n + 1;  j3 = ylen - 1; }
                      for ( ; j2 <= j3; ) {
                         pty[jj++] = pty[j2++];
                      }
                   }
                   for ( ; n > 0; n--, i1 = (i1 == 0) ? xlen1 : i1 - 1 ) {
                      pty[++j1] = ptInvs[HD_TO_INT(ptx[i1])];
                   }
                   pty[-1] = INT_TO_HD( newlen );
                   ptLens[y] = INT_TO_HD( newlen );
                   total = total - ylen + newlen;
                   ptFlags[y] = INT_TO_HD( newflag );

                   /* Reduce the bag size                                  */
                   Resize( hdl, (newlen + 1) * SIZE_HD );
                   ptRels = PTR_BAG( hdRels );
                   ptLens = PTR_BAG( hdLens );
                   ptFlags = PTR_BAG( hdFlags);
                   ptInvs = PTR_BAG( hdInvs ) + numgens + 1;

                   altered = 1;
                   ++count;
                   --y;
                   break;  /*  loop over i                                 */
                }

                if ( altered ) break;  /*  loop over x                     */

                /* Now try the inverse of the given relator                */
                for ( i = 0; i < xlen; i++ ) {

                   /* Search forward for a match of length at least mlen   */
                   i2 = xlen1 - i;  j2 = j;
                   for ( n = 0; n < xlen; n++,
                      i2 = (i2 == 0) ? xlen1 : i2 - 1,
                      j2 = (j2 == ylen1) ? 0 : j2 + 1 ) {
                      if ( ptInvs[HD_TO_INT(ptx[i2])] != pty[j2] )
                         break;  /*  loop over n                           */
                   }
                   if ( n < mlen )  continue;  /*  loop over i             */

                   /* Search backward to find the whole match              */
                   i1 = (i == 0) ? 0 : xlen - i;
                   j1 = (j == 0) ? ylen1 : j - 1;
                   for ( ; n < xlen; n++,
                      i1 = (i1 == xlen1) ? 0 : i1 + 1,
                      j1 = (j1 == 0) ? ylen1 : j1 - 1 ) {
                      if ( ptInvs[HD_TO_INT(ptx[i1])] != pty[j1] )
                         break;  /*  loop over n                           */
                   }

                   /* Replace a matching substring of equal length         */
                   if ( n == xlen - n ) {
                      j2 = j;
                      for ( m = 0; m < n; m++,
                         i1 = (i1 == xlen1) ? 0 : i1 + 1,
                         j2 = (j2 == ylen1) ? 0 : j2 + 1 ) {
                         pty[j2] = ptx[i1];
                      }

                      ptFlags[y] = INT_TO_HD( newflag );
                      altered = 1;
                      ++count;
                      break;  /*  loop over i                              */
                   }

                   m = ylen - n;  n = xlen - n;

                   /* Find all canceling factors                           */
                   if ( n == 0 ) {
                      for ( ; 1 < m; m -= 2,
                         j1 = (j1 == 0) ? ylen1 : j1 - 1,
                         j2 = (j2 == ylen1) ? 0 : j2 + 1 ) {
                         if ( pty[j1] != ptInvs[HD_TO_INT(pty[j2])] )
                            break;  /*  loop over m                        */
                      }
                   }

                   /* Create the modified relator and save it              */
                   newlen = m + n;
                   if ( j2 > 0 ) {
                      if ( j2 <= j1 )  { jj = 0;  j3 = j1;  j1 = m - 1; }
                      else  { jj = j1 + n + 1;  j3 = ylen - 1; }
                      for ( ; j2 <= j3; ) {
                         pty[jj++] = pty[j2++];
                      }
                   }
                   for ( ; n > 0; n--, i1 = (i1 == xlen1) ? 0 : i1 + 1 ) {
                      pty[++j1] = ptx[i1];
                   }
                   pty[-1] = INT_TO_HD( newlen );
                   ptLens[y] = INT_TO_HD( newlen );
                   total = total - ylen + newlen;
                   ptFlags[y] = INT_TO_HD( newflag );

                   /* Reduce the bag size                                  */
                   Resize( hdl, (newlen + 1) * SIZE_HD );
                   ptRels = PTR_BAG( hdRels );
                   ptLens = PTR_BAG( hdLens );
                   ptFlags = PTR_BAG( hdFlags);
                   ptInvs = PTR_BAG( hdInvs ) + numgens + 1;

                   altered = 1;
                   ++count;
                   --y;
                   break;  /*  loop over i                                 */
                }

                if ( altered ) break;  /*  loop over x                     */
             }
          }

          if ( altered ) break;  /*  loop over j                           */

          v = ptr + 1 + j;  w = ptr + 1 + ( j + mlen ) % ylen;
          key = 109109 * key - inv * ( (UInt)*v >> 2 )
              + ( (UInt)*w >> 2 );
       }
    }

    SET_BAG( hdTie , TZ_TOTAL,  INT_TO_HD( total ) );

    /* Return the number of altered relators                               */
    return INT_TO_HD( count );
}


/****************************************************************************
**
*F  InitTietze()  . . . . . . . . . . . . . . . . . initialize tietze package
**
**  'InitTietze' initializes the Tietze package.
*/
void            InitTietze (void)
{
    InstIntFunc( "TzRelator",           FunTzRelator          );
    InstIntFunc( "TzWord",              FunTzWord             );
    InstIntFunc( "TzSortC",             FunTzSortC            );
    InstIntFunc( "TzRenumberGens",      FunTzRenumberGens     );
    InstIntFunc( "TzReplaceGens",       FunTzReplaceGens      );
    InstIntFunc( "TzSubstituteGen",     FunTzSubstituteGen    );
    InstIntFunc( "TzOccurrences",       FunTzOccurrences      );
    InstIntFunc( "TzOccurrencesPairs",  FunTzOccurrencesPairs );
    InstIntFunc( "TzSearchC",           FunTzSearchC          );
}


/****************************************************************************
**
*E  Emacs . . . . . . . . . . . . . . . . . . . . . . . local emacs variables
**
**  Local Variables:
**  outline-regexp:     "*F\\|*V\\|*T\\|*E"
**  fill-column:        73
**  fill-prefix:        "**  "
**  End:
*/



