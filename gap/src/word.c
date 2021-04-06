/****************************************************************************
**
*A  word.c                      GAP source                   Martin Schoenert
**                                                             & Frank Celler
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file contains the code for computing with words and  abstract  gens.
**
*/

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "scanner.h"             /* reading of tokens and printing  */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */
#include        "list.h"                /* 'LEN_LIST' macro                */
#include        "gstring.h"              /* 'IsString' test                 */
#include        "agcollec.h"            /* 'HD_WORDS' macro for T_SWORDS   */

#include        "word.h"                /* declaration part of the package */


/****************************************************************************
**
*F  SwordWord( <list>, <word> ) . . . . . . . .  convert/copy word into sword
**
**  'SwordWord'  returns  either  a  sword  representing  <word> in <list> or
**  'HdFalse' if <word> could not convert.
*/
Bag       SwordWord (Bag hdLst, Bag hdWrd)
{
    Bag       hdSwrd,  * ptLst,  * ptEnd,  * ptG;
    TypSword        * ptSwrd;
    Int            len,  lnSwrd,  i,  exp;
    
    len = GET_SIZE_BAG( hdWrd ) / SIZE_HD;
    hdSwrd = NewBag( T_SWORD, SIZE_HD + 2 * SIZE_SWORD * len + SIZE_SWORD );
    ptEnd  = PTR_BAG( hdWrd ) + len - 1;
    ptLst  = PTR_BAG( hdLst ) + 1;
    len    = LEN_LIST( hdLst );

    /*N Run through the word and convert, this is a very stupid algorithm **/
    /*N but for the moment,  it should be fast enough.                    **/
    SET_BAG( hdSwrd , 0,  hdLst );
    ptSwrd = (TypSword*)( PTR_BAG( hdSwrd ) + 1 );
    lnSwrd = 0;
    for ( ptG = PTR_BAG( hdWrd ); ptG <= ptEnd; ptG++ )
    {
        for ( i = len - 1; i >= 0; i-- )
            if ( ptLst[i] == *ptG || ptLst[i] == *PTR_BAG( *ptG ) )
                break;
        if ( i < 0 )
            return HdFalse;
        exp = 1;
        while ( ptG < ptEnd && ptG[0] == ptG[1] )
        {
            ptG++;
            exp++;
        }
        if ( ptLst[ i ] == *PTR_BAG( *ptG ) )
            exp = -exp;
        *ptSwrd++ = i;
        *ptSwrd++ = exp;
	if ( MAX_SWORD_NR <= exp )
	    return Error( "SwordWord: exponent overflow",  0,  0 );
        lnSwrd++;
    }

    /** Append endmarker, reduce to correct size and return. ***************/
    *ptSwrd = -1;
    Resize( hdSwrd, SIZE_HD + ( 2 * lnSwrd + 1 ) * SIZE_SWORD );
    return hdSwrd;
}


/****************************************************************************
**
*F  WordSword( <sword> )  . . . . . . . . . . .  convert/copy sword into word
**
**  Return the representation of a T_SWORD <sword> as word of type T_WORD.
*/
Bag       WordSword (Bag hdSwrd)
{
    Bag       hdWrd, * ptWrd, * ptLst, hdAgn;
    TypSword        * ptSwrd;
    Int            len, i;

    /** Count number of generators and copy them. **************************/
    len    = 0;
    ptSwrd = (TypSword*)( PTR_BAG( hdSwrd ) + 1 );
    while ( *ptSwrd != -1 )
    {
        len    += ( ptSwrd[1] < 0 ) ? ( -ptSwrd[1] ) : ptSwrd[1];
        ptSwrd += 2;
    }
    hdWrd  = NewBag( T_WORD, len * SIZE_HD );
    ptWrd  = PTR_BAG( hdWrd );
    ptSwrd = (TypSword*)( PTR_BAG( hdSwrd ) + 1 );

    /** Catch sword with a polycyclic presentation. ************************/
    if ( GET_TYPE_BAG( *PTR_BAG( hdSwrd ) ) == T_AGGRP )
        ptLst = PTR_BAG( HD_WORDS( *PTR_BAG( hdSwrd ) ) ) + 1;
    else
        ptLst = PTR_BAG( *PTR_BAG( hdSwrd ) ) + 1;

    while ( *ptSwrd != -1 )
    {
        if ( ptSwrd[1] > 0 )
        {
            hdAgn = ptLst[ ptSwrd[ 0 ] ];
            len   = ptSwrd[ 1 ];
        }
        else
        {
            hdAgn = *PTR_BAG( ptLst[ ptSwrd[ 0 ] ] );
            len   = -ptSwrd[ 1 ];
        }
        for ( i = len; i > 0; i-- )
            *ptWrd++ = hdAgn;
        ptSwrd += 2;
    }
    return hdWrd;
}


/****************************************************************************
**
*F  SwordSword( <list>, <sword> ) . . . . . . . . . . .  copy/convert <sword>
**
**  Convert word  <sword>  of type T_SWORD into a T_SWORD with generator list
**  <list>.  Return 'HdFalse' if comverting process failed.
*/
Bag       SwordSword (Bag hdLst, Bag hdSwrd)
{
    if ( *PTR_BAG( hdSwrd ) == hdLst )
        return hdSwrd;

    /*N This is the most stupid way, but for the moment ... ****************/
    return SwordWord( hdLst, WordSword( hdSwrd ) );
}


/****************************************************************************
**
*F  EvWord( <hdWord> )  . . . . . . . . . . . . . . . . . . . evaluate a word
**
**  This function evaluates a word in abstract generators, since  this  words
**  are constants nothing happens.
*/
Bag       EvWord (Bag hdWord)
{
    return hdWord;
}


/****************************************************************************
**
*F  ProdWord( <hdL>, <hdR> )  . . . . . . . . . . . .  eval <wordL> * <wordR>
**
**  This function multplies the two words <hdL> and <hdR>. Since the function
**  is called from evalutor both operands are already evaluated.
*/
Bag       ProdWord (Bag hdL, Bag hdR)
{
    Int            lnL,  lnR,  lnRR, e;
    Bag       * ptL,  * ptR,  * ptRes;
    Bag       hdRes, hdLstL, hdLstR;
    TypSword        * gtL,  * gtR,  * gtRes;

    /** Catch trivial words and swords *************************************/
    if ( GET_TYPE_BAG( hdL ) == T_WORD && GET_SIZE_BAG( hdL ) == 0 )
        return hdR;
    if ( GET_TYPE_BAG( hdL ) == T_SWORD && GET_SIZE_BAG( hdL ) == SIZE_HD + SIZE_SWORD )
        return hdR;
    if ( GET_TYPE_BAG( hdR ) == T_WORD && GET_SIZE_BAG( hdR ) == 0 )
        return hdL;
    if ( GET_TYPE_BAG( hdR ) == T_SWORD && GET_SIZE_BAG( hdR ) == SIZE_HD + SIZE_SWORD )
        return hdL;

    /** Dispatch to different multiplication routines. *********************/
    if ( GET_TYPE_BAG( hdL ) == T_WORD || GET_TYPE_BAG( hdR ) == T_WORD )
    {
        /** Convert swords into words. *************************************/
        if ( GET_TYPE_BAG( hdL ) == T_SWORD )
            hdL = WordSword( hdL );
        if ( GET_TYPE_BAG( hdR ) == T_SWORD )
            hdR = WordSword( hdR );

        /** <hdL>  and  <hdR> are both string words,  set up pointers and **/
        /** counters, and find the part that cancels.                     **/
        lnL = GET_SIZE_BAG( hdL ) / SIZE_HD;  ptL = PTR_BAG( hdL ) + lnL - 1;
        lnR = GET_SIZE_BAG( hdR ) / SIZE_HD;  ptR = PTR_BAG( hdR );
        while ( 0 < lnL && 0 < lnR && *ptL == *PTR_BAG( *ptR ) )
        {
            --lnL;  --ptL;
            --lnR;  ++ptR;
        }
        if ( lnL + lnR == 0 )
            return HdIdWord;

        /** Allocate bag for the result and recompute pointer. *************/
        hdRes = NewBag( T_WORD, ( lnL + lnR ) * SIZE_HD );
        ptRes = PTR_BAG( hdRes );
        ptL   = PTR_BAG( hdL );
        ptR   = PTR_BAG( hdR ) + GET_SIZE_BAG( hdR ) / SIZE_HD - lnR;

        /** Copy both parts into <hdRes> and return. ***********************/
        while ( 0 < lnL )  { *ptRes++ = *ptL++;  --lnL; }
        while ( 0 < lnR )  { *ptRes++ = *ptR++;  --lnR; }
        return hdRes;
    }
    
    /** Now both operands are swords, but do the have the same genlist? ****/
    hdLstL = PTR_BAG( hdL )[ 0 ];
    hdLstR = PTR_BAG( hdR )[ 0 ];
    if ( GET_TYPE_BAG( hdLstL ) == T_AGGRP )
        hdLstL = HD_WORDS( hdLstL );
    if ( GET_TYPE_BAG( hdLstR ) == T_AGGRP )
        hdLstR = HD_WORDS( hdLstR );
    if ( hdLstL != hdLstR )
        return ProdWord( WordSword( hdL ), WordSword( hdR ) );
    
    /** Set up pointers and counters, find part that cancels ***************/
           lnL  = ( GET_SIZE_BAG( hdL ) - SIZE_HD - SIZE_SWORD ) / (2 * SIZE_SWORD);
    lnRR = lnR  = ( GET_SIZE_BAG( hdR ) - SIZE_HD - SIZE_SWORD ) / (2 * SIZE_SWORD);
    gtL = (TypSword*)( PTR_BAG( hdL ) + 1 ) + 2 * ( lnL - 1 );
    gtR = (TypSword*)( PTR_BAG( hdR ) + 1 );
    while ( 0 < lnL && 0 < lnR && gtL[0]==gtR[0] && gtL[1] == -gtR[1] )
    {
        --lnL;  gtL -= 2;
        --lnR;  gtR += 2;
    }
    if ( lnL + lnR == 0 )
        return HdIdWord;
    
    /** Allocate bag for the result, recompute pointers. *******************/
    if ( 0 < lnL && 0 < lnR && gtL[ 0 ] == gtR[ 0 ] )
        hdRes = NewBag( T_SWORD, SIZE_HD + (2*(lnL+lnR) - 1) * SIZE_SWORD );
    else
        hdRes = NewBag( T_SWORD, SIZE_HD + (2*(lnL+lnR) + 1) * SIZE_SWORD );
    gtRes = (TypSword*)( PTR_BAG( hdRes ) + 1 );
    gtL   = (TypSword*)( PTR_BAG( hdL ) + 1 );
    gtR   = (TypSword*)( PTR_BAG( hdR ) + 1 ) + 2 * ( lnRR - lnR );

    /** Copy both parts into <hdRes>, add endmarker and return. ************/
    while ( 1 < lnL )
    {
        *gtRes++ = *gtL++; 
        *gtRes++ = *gtL++;
        --lnL;
    }
    if ( 0 < lnL )
    {
        if ( 0 < lnR && gtL[0] == gtR[0] )
        {
            --lnR;
            ++gtR;
            *gtRes++ = *gtL++;
            e = *gtL++ + *gtR++;
            if ( ((e << 16) >> 16) != e )
                return Error( "Words: integer overflow",  0,  0 );
            *gtRes++ = e;
        }
        else
        {
            *gtRes++ = *gtL++; 
            *gtRes++ = *gtL++;
        }
    }
    while ( 0 < lnR )
    {
        *gtRes++ = *gtR++;
        *gtRes++ = *gtR++;
        --lnR;
    }

    SET_BAG( hdRes ,  0 ,  hdLstL );
    *gtRes = -1;

    return hdRes;
}


/****************************************************************************
**
*F  QuoWord( <hdL>, <hdR> ) . . . . . . . . . . . eval <wordL> * <wordR> ^ -1
**
**  This function divides the two words <hdL> and  <hdR>.  Since the function
**  is called from evalutor both operands are already evaluated.
*/
Bag       QuoWord (Bag hdL, Bag hdR)
{
    Int            lnL,  lnR,  e;
    Bag       * ptL,  * ptR,  * ptRes;
    Bag       hdRes, hdLstL, hdLstR;
    TypSword        * gtL,  * gtR,  * gtRes;

    /** Catch trivial words and swords *************************************/
    if ( GET_TYPE_BAG( hdL ) == T_WORD && GET_SIZE_BAG( hdL ) == 0 )
        return PowWI( hdR, INT_TO_HD( -1 ) );
    if ( GET_TYPE_BAG( hdL ) == T_SWORD && GET_SIZE_BAG( hdL ) == SIZE_HD + SIZE_SWORD )
        return PowWI( hdR, INT_TO_HD( -1 ) );
    if ( GET_TYPE_BAG( hdR ) == T_WORD && GET_SIZE_BAG( hdR ) == 0 )
        return hdL;
    if ( GET_TYPE_BAG( hdR ) == T_SWORD && GET_SIZE_BAG( hdR ) == SIZE_HD + SIZE_SWORD )
        return hdL;

    /** Dispatch to different multiplication routines. *********************/
    if ( GET_TYPE_BAG( hdL ) == T_WORD || GET_TYPE_BAG( hdR ) == T_WORD )
    {
        /** Convert swords into words. *************************************/
        if ( GET_TYPE_BAG( hdL ) == T_SWORD )
            hdL = WordSword( hdL );
        if ( GET_TYPE_BAG( hdR ) == T_SWORD )
            hdR = WordSword( hdR );

        /** <hdL>  and  <hdR> are both string words,  set up pointers and **/
        /** counters, and find the part that cancels.                     **/
        lnL = GET_SIZE_BAG( hdL ) / SIZE_HD;  ptL = PTR_BAG( hdL ) + lnL - 1;
        lnR = GET_SIZE_BAG( hdR ) / SIZE_HD;  ptR = PTR_BAG( hdR ) + lnR - 1;
        while ( 0 < lnL && 0 < lnR && *ptL == *ptR )
        {
            --lnL;  --ptL;
            --lnR;  --ptR;
        }
        if ( lnL + lnR == 0 )
            return HdIdWord;

        /** Allocate bag for the result and recompute pointer. *************/
        hdRes = NewBag( T_WORD, ( lnL + lnR ) * SIZE_HD );
        ptRes = PTR_BAG( hdRes );
        ptL   = PTR_BAG( hdL );
        ptR   = PTR_BAG( hdR ) + lnR - 1;

        /** Copy both parts into <hdRes> and return. ***********************/
        while ( 0 < lnL )  { *ptRes++ = *ptL++;          --lnL; }
        while ( 0 < lnR )  { *ptRes++ = *PTR_BAG( *ptR-- );  --lnR; }
        return hdRes;
    }
    
    /** Now both operands are swords, but do the have the same genlist? ****/
    hdLstL = PTR_BAG( hdL )[ 0 ];
    hdLstR = PTR_BAG( hdR )[ 0 ];
    if ( GET_TYPE_BAG( hdLstL ) == T_AGGRP )
        hdLstL = HD_WORDS( hdLstL );
    if ( GET_TYPE_BAG( hdLstR ) == T_AGGRP )
        hdLstR = HD_WORDS( hdLstR );
    if ( hdLstL != hdLstR )
        return QuoWord( WordSword( hdL ), WordSword( hdR ) );
    
    /** Set up pointers and counters, find part that cancels ***************/
    lnL  = ( GET_SIZE_BAG( hdL ) - SIZE_HD - SIZE_SWORD ) / ( 2 * SIZE_SWORD );
    lnR  = ( GET_SIZE_BAG( hdR ) - SIZE_HD - SIZE_SWORD ) / ( 2 * SIZE_SWORD );
    gtL = (TypSword*)( PTR_BAG( hdL ) + 1 ) + 2 * ( lnL - 1 );
    gtR = (TypSword*)( PTR_BAG( hdR ) + 1 ) + 2 * ( lnR - 1 );
    while ( 0 < lnL && 0 < lnR && gtL[0]==gtR[0] && gtL[1] == gtR[1] )
    {
        --lnL;  gtL -= 2;
        --lnR;  gtR -= 2;
    }
    if ( lnL + lnR == 0 )
        return HdIdWord;
    
    /** Allocate bag for the result, recompute pointers. *******************/
    if ( 0 < lnL && 0 < lnR && gtL[ 0 ] == gtR[ 0 ] )
        hdRes = NewBag( T_SWORD, SIZE_HD + (2*(lnL+lnR) - 1) * SIZE_SWORD );
    else
        hdRes = NewBag( T_SWORD, SIZE_HD + (2*(lnL+lnR) + 1) * SIZE_SWORD );
    gtRes = (TypSword*)( PTR_BAG( hdRes ) + 1 );
    gtL   = (TypSword*)( PTR_BAG( hdL ) + 1 );
    gtR   = (TypSword*)( PTR_BAG( hdR ) + 1 ) + 2 * ( lnR - 1 );

    /** Copy both parts into <hdRes>, add endmarker and return. ************/
    while ( 1 < lnL )
    {
        *gtRes++ = *gtL++;
        *gtRes++ = *gtL++;
        --lnL;
    }
    if ( 0 < lnL )
    {
        if ( 0 < lnR && gtL[0] == gtR[0] )
        {
            *gtRes++ = *gtL++;
            e = *gtL++ - gtR[1];
            if ( ((e << 16) >> 16) != e )
                return Error( "Words: integer overflow",  0,  0 );
            *gtRes++ = e;
            --lnR;
            gtR -= 2;
        }
        else
        {
            *gtRes++ = *gtL++; 
            *gtRes++ = *gtL++;
        }
    }
    while ( 0 < lnR )
    {
        *gtRes++ = gtR[0];
        *gtRes++ = -gtR[1];
        --lnR;
        gtR -= 2;
    }

    SET_BAG( hdRes ,  0 ,  hdLstL );
    *gtRes = -1;

    return hdRes;
}


/****************************************************************************
**
*F  ModWord( <hdL>, <hdR> ) . . . . . . . . . . . eval <wordL> ^ -1 * <wordR>
**
**  This function  left divides  the two words  <hdL>  and  <hdR>.  Since the
**  function is called from evalutor both operands are already evaluated.
*/
Bag       ModWord (Bag hdL, Bag hdR)
{
    Int            lnL,  lnR,  lnLL,  lnRR,  e;
    Bag       * ptL,  * ptR,  * ptRes;
    Bag       hdRes, hdLstL, hdLstR;
    TypSword        * gtL,  * gtR,  * gtRes;

    /** Catch trivial words and swords *************************************/
    if ( GET_TYPE_BAG( hdL ) == T_WORD && GET_SIZE_BAG( hdL ) == 0 )
        return hdR;
    if ( GET_TYPE_BAG( hdL ) == T_SWORD && GET_SIZE_BAG( hdL ) == SIZE_HD + SIZE_SWORD )
        return hdR;
    if ( GET_TYPE_BAG( hdR ) == T_WORD && GET_SIZE_BAG( hdR ) == 0 )
        return PowWI( hdL, INT_TO_HD( -1 ) );
    if ( GET_TYPE_BAG( hdR ) == T_SWORD && GET_SIZE_BAG( hdR ) == SIZE_HD + SIZE_SWORD )
        return PowWI( hdL, INT_TO_HD( -1 ) );

    /** Dispatch to different multiplication routines. *********************/
    if ( GET_TYPE_BAG( hdL ) == T_WORD || GET_TYPE_BAG( hdR ) == T_WORD )
    {
        /** Convert swords into words. *************************************/
        if ( GET_TYPE_BAG( hdL ) == T_SWORD )
            hdL = WordSword( hdL );
        if ( GET_TYPE_BAG( hdR ) == T_SWORD )
            hdR = WordSword( hdR );

        /** <hdL>  and  <hdR> are both string words,  set up pointers and **/
        /** counters, and find the part that cancels.                     **/
        lnL = GET_SIZE_BAG( hdL ) / SIZE_HD;  ptL = PTR_BAG( hdL );
        lnR = GET_SIZE_BAG( hdR ) / SIZE_HD;  ptR = PTR_BAG( hdR );
        while ( 0 < lnL && 0 < lnR && *ptL == *ptR )
        {
            --lnL;  ++ptL;
            --lnR;  ++ptR;
        }
        if ( lnL + lnR == 0 )
            return HdIdWord;

        /** Allocate bag for the result and recompute pointer. *************/
        hdRes = NewBag( T_WORD, ( lnL + lnR ) * SIZE_HD );
        ptRes = PTR_BAG( hdRes );
        ptL   = PTR_BAG( hdL ) + GET_SIZE_BAG( hdL ) / SIZE_HD - 1;
        ptR   = PTR_BAG( hdR ) + GET_SIZE_BAG( hdR ) / SIZE_HD - lnR;

        /** Copy both parts into <hdRes> and return. ***********************/
        while ( 0 < lnL )  { *ptRes++ = *PTR_BAG( *ptL-- );  --lnL; }
        while ( 0 < lnR )  { *ptRes++ = *ptR++;          --lnR; }
        return hdRes;
    }
    
    /** Now both operands are swords, but do the have the same genlist? ****/
    hdLstL = PTR_BAG( hdL )[ 0 ];
    hdLstR = PTR_BAG( hdR )[ 0 ];
    if ( GET_TYPE_BAG( hdLstL ) == T_AGGRP )
        hdLstL = HD_WORDS( hdLstL );
    if ( GET_TYPE_BAG( hdLstR ) == T_AGGRP )
        hdLstR = HD_WORDS( hdLstR );
    if ( hdLstL != hdLstR )
        return ModWord( WordSword( hdL ), WordSword( hdR ) );
    
    /** Set up pointers and counters, find part that cancels ***************/
    lnLL = lnL  = ( GET_SIZE_BAG( hdL ) - SIZE_HD - SIZE_SWORD ) / (2 * SIZE_SWORD);
    lnRR = lnR  = ( GET_SIZE_BAG( hdR ) - SIZE_HD - SIZE_SWORD ) / (2 * SIZE_SWORD);
    gtL = (TypSword*)( PTR_BAG( hdL ) + 1 );
    gtR = (TypSword*)( PTR_BAG( hdR ) + 1 );
    while ( 0 < lnL && 0 < lnR && gtL[0]==gtR[0] && gtL[1] == gtR[1] )
    {
        --lnL;  gtL += 2;
        --lnR;  gtR += 2;
    }
    if ( lnL + lnR == 0 )
        return HdIdWord;
    
    /** Allocate bag for the result, recompute pointers. *******************/
    if ( 0 < lnL && 0 < lnR && gtL[ 0 ] == gtR[ 0 ] )
        hdRes = NewBag( T_SWORD, SIZE_HD + (2*(lnL+lnR) - 1) * SIZE_SWORD );
    else
        hdRes = NewBag( T_SWORD, SIZE_HD + (2*(lnL+lnR) + 1) * SIZE_SWORD );
    gtRes = (TypSword*)( PTR_BAG( hdRes ) + 1 );
    gtL   = (TypSword*)( PTR_BAG( hdL ) + 1 ) + 2 * ( lnLL - 1 );
    gtR   = (TypSword*)( PTR_BAG( hdR ) + 1 ) + 2 * ( lnRR - lnR );

    /** Copy both parts into <hdRes>, add endmarker and return. ************/
    while ( 1 < lnL )
    {
        *gtRes++ = gtL[0];
        *gtRes++ = -gtL[1];
        gtL -= 2;
        --lnL;
    }
    if ( 0 < lnL )
    {
        if ( 0 < lnR && gtL[0] == gtR[0] )
        {
            --lnR;
            ++gtR;
            *gtRes++ = gtL[0];
            e = -gtL[1] + *gtR++;
            if ( ((e << 16) >> 16) != e )
                return Error( "Words: integer overflow",  0,  0 );
            *gtRes++ = e;
        }
        else
        {
            *gtRes++ = gtL[0];
            *gtRes++ = -gtL[1];
        }
    }
    while ( 0 < lnR )
    {
        *gtRes++ = *gtR++;
        *gtRes++ = *gtR++;
        --lnR;
    }

    SET_BAG( hdRes ,  0 ,  hdLstL );
    *gtRes = -1;

    return hdRes;
}


/****************************************************************************
**
*F  PowWI( <hdL>, <hdR> ) . . . . . . . . . . . . . . . eval <wordL> ^ <intR>
**
**  'PowWI' is  called to evaluate the exponentiation of a word by a integer.
**  It is  called from  th evaluator so both  operands are already evaluated.
*N  This function should be rewritten, it can be faster, but for the moment..
*/
Bag       PowWI (Bag hdL, Bag hdR)
{
    Bag       hdRes,  hdLst;
    Bag       * ptL,  * ptRes;
    TypSword        * gtL,  * gtR;
    Int            exp;

    /** Catch the trivial cases, trivial word and trivial exponent *********/
    exp = HD_TO_INT( hdR );
    if ( exp == 0 )
        return HdIdWord;
    if ( GET_TYPE_BAG( hdL ) == T_WORD && GET_SIZE_BAG( hdL ) == 0 )
        return HdIdWord;
    if ( GET_TYPE_BAG( hdL ) == T_SWORD && GET_SIZE_BAG( hdL ) == SIZE_HD + SIZE_SWORD )
        return HdIdWord;

    /** If neccessary invert the left operand. *****************************/
    if ( exp < 0 )
    {
        if ( GET_TYPE_BAG( hdL ) == T_WORD )
        {
            hdRes = NewBag( T_WORD, GET_SIZE_BAG( hdL ) );
            ptRes = PTR_BAG( hdRes );
            ptL   = PTR_BAG( hdL ) + GET_SIZE_BAG( hdL ) / SIZE_HD - 1;
            while ( ptL >= PTR_BAG(hdL) )  *ptRes++ = *PTR_BAG( *ptL-- );
        }
        else
        {
            hdRes = NewBag( T_SWORD, GET_SIZE_BAG( hdL ) );
            hdLst = PTR_BAG( hdL )[0];
            if ( GET_TYPE_BAG( hdLst ) == T_AGGRP )
                hdLst = HD_WORDS( PTR_BAG( hdL )[0] );
            SET_BAG( hdRes , 0,  hdLst );
            gtL  = (TypSword*)( PTR_BAG( hdL ) + 1 );
            gtR  = (TypSword*)( (char*) PTR_BAG( hdRes ) + GET_SIZE_BAG(hdRes) ) - 1;
            *gtR = -1;
            gtR -= 2;
            while ( *gtL != -1 )
            {
                gtR[0] = *gtL++;
                gtR[1] = -*gtL++;
                gtR   -= 2;
            }
        }
        hdL = hdRes;
        exp = - exp;
    }

    /** Raise the word to the power using the russian peasent method. ******/
    if ( exp == 1 )
    {
        hdRes = hdL;
    }
    else
    {
        hdRes = HdIdWord;
        while ( exp > 0 )
        {
            if ( exp % 2 == 1 )
            {
                hdRes = ProdWord( hdRes, hdL );
                exp   = exp - 1;
            }
            else
            {
                hdL = ProdWord( hdL, hdL );
                exp = exp / 2;
            }
        }
    }

    /** Return the result. *************************************************/
    return hdRes;
}


/****************************************************************************
**
*F  PowWW( <hdL>, <hdR> ) . . . . . . . . . . . . . .  eval <wordL> ^ <wordR>
**
**  PowWW() is called to evaluate  the  conjugation  of  two  word  operands.
**  It is called from the evaluator so both operands are  already  evaluated.
*N  This function should be rewritten, it should not call 'ProdWord'.
*/
Bag       PowWW (Bag hdL, Bag hdR)
{
    if ( GET_TYPE_BAG( hdL ) == T_WORD && GET_TYPE_BAG( hdR ) == T_SWORD )
        hdR = WordSword( hdR );
    if ( GET_TYPE_BAG( hdL ) == T_SWORD && GET_TYPE_BAG( hdR ) == T_WORD )
        hdL = WordSword( hdL );

    return ProdWord( PowWI( hdR, INT_TO_HD( -1 ) ), ProdWord( hdL, hdR ) );
}


/****************************************************************************
**
*F  CommWord( <hdL>, <hdR> )  . . . . . . . . . eval comm( <wordL>, <wordR> )
**
**  'CommWord' is  called to evaluate the commutator of  two  word  operands.
**  It is called from the evaluator so both operands are already evaluated.
*/
Bag       CommWord (Bag hdL, Bag hdR)
{
    if ( GET_TYPE_BAG( hdL ) == T_WORD && GET_TYPE_BAG( hdR ) == T_SWORD )
        hdR = WordSword( hdR );
    if ( GET_TYPE_BAG( hdL ) == T_SWORD && GET_TYPE_BAG( hdR ) == T_WORD )
        hdL = WordSword( hdL );

    return ProdWord( PowWI( hdL, INT_TO_HD( -1 ) ),
                     ProdWord( PowWI( hdR, INT_TO_HD( -1 ) ),
                               ProdWord( hdL, hdR ) ) );
}


/****************************************************************************
**
*F  EqWord( <hdL>, <hdR> )  . . . . . . . . . . . .test if <wordL>  = <wordR>
**
**  'EqWord'  is called to  compare  the  two  word  operands  for  equality.
**  It is called from the evaluator so both operands are  already  evaluated.
**  Two speed up the comparism we first check that they have the same size.
**
**  Special care must be taken, if one argument is a sword because we are not
**  allowed to call 'NewBag' for converting a sword into a word.
*/
Bag       EqWord (Bag hdL, Bag hdR)
{
    Bag       * ptL,  * ptR,  * ptEnd,  hdLstL,  hdLstR, hdTmp;
    TypSword        * gtL,  * gtR;
    Int            i, j;

    if ( GET_TYPE_BAG( hdL ) == T_WORD && GET_TYPE_BAG( hdR ) == T_WORD )
    {
        if ( GET_SIZE_BAG( hdL ) != GET_SIZE_BAG( hdR ) )
            return HdFalse;
        ptL = PTR_BAG( hdL );
        ptR = PTR_BAG( hdR );
        for ( i = GET_SIZE_BAG( hdL ) / SIZE_HD; i > 0; --i, ++ptL, ++ptR )
            if ( *ptL != *ptR )
                return HdFalse;
        return HdTrue;
    }
    else if ( GET_TYPE_BAG( hdL ) == T_SWORD && GET_TYPE_BAG( hdR ) == T_SWORD )
    {
        if ( GET_SIZE_BAG( hdL ) != GET_SIZE_BAG( hdR ) )
            return HdFalse;
        hdLstL = PTR_BAG( hdL )[ 0 ];
        hdLstR = PTR_BAG( hdR )[ 0 ];
        if ( GET_TYPE_BAG( hdLstL ) == T_AGGRP )
            hdLstL = HD_WORDS( hdLstL );
        if ( GET_TYPE_BAG( hdLstR ) == T_AGGRP )
            hdLstR = HD_WORDS( hdLstR );
        if ( hdLstL == hdLstR )
        {
            gtL = (TypSword*)( PTR_BAG( hdL ) + 1 );
            gtR = (TypSword*)( PTR_BAG( hdR ) + 1 );
            while ( *gtL != -1 && gtL[0] == gtR[0] && gtL[1] == gtR[1] )
            {
                gtL += 2;
                gtR += 2;
            }
            return ( *gtL == -1 && *gtR == -1 ) ? HdTrue : HdFalse;
        }
        else
        {
            ptL  = PTR_BAG( hdLstL ) + 1;
            ptR  = PTR_BAG( hdLstR ) + 1;
            gtL = (TypSword*)( PTR_BAG( hdL ) + 1 );
            gtR = (TypSword*)( PTR_BAG( hdR ) + 1 );

            while ( *gtL != -1
                    && *gtR != -1 
                    && ( ( ptL[ gtL[0] ] == ptR[ gtR[0] ]
                           && gtL[1] == gtR[1] )
                      || ( ptL[ gtL[0] ] == *PTR_BAG( ptR[ gtR[0] ] )
                           && gtL[1] != -gtR[1]) ) )
            {
                gtL += 2;
                gtR += 2;
            }
            return ( *gtL == -1 && *gtR == -1 ) ? HdTrue : HdFalse;
        }
    }
    else
    {
        if ( GET_TYPE_BAG( hdL ) == T_WORD )
        {
            hdTmp = hdL;
            hdL   = hdR;
            hdR   = hdTmp;
        }
        hdLstL = PTR_BAG( hdL )[ 0 ];
        if ( GET_TYPE_BAG( hdLstL ) == T_AGGRP )
            hdLstL = HD_WORDS( hdLstL );
        ptL   = PTR_BAG( hdLstL ) + 1;
        gtL  = (TypSword*)( PTR_BAG( hdL ) + 1 );
        ptR   = PTR_BAG( hdR );
        ptEnd = (Bag*)( (char*) ptR + GET_SIZE_BAG( hdR ) );
        while ( *gtL != -1 && ptR < ptEnd )
        {
            if ( *ptR == ptL[ gtL[0] ] )
            {
                if ( gtL[1] < 0 )
                    return HdFalse;
                hdTmp = ptL[ gtL[0] ];
                for ( j = gtL[1]; j > 0; j--, ptR++ )
                    if ( ptR == ptEnd || *ptR != hdTmp )
                        return HdFalse;
                gtL += 2;
            }
            else if ( *ptR == *PTR_BAG( ptL[ gtL[0] ] ) )
            {
                if ( gtL[1] > 0 )
                    return HdFalse;
                hdTmp = *PTR_BAG( ptL[ gtL[0] ] );
                for ( j = -gtL[1]; j > 0; j--, ptR++ )
                    if ( ptR == ptEnd || *ptR != hdTmp )
                        return HdFalse;
                gtL += 2;
            }
            else
                return HdFalse;
        }
        return ( *gtL == -1 && ptR == ptEnd ) ? HdTrue : HdFalse;
    }
}


/****************************************************************************
**
*F  LtAgen( <hdL>, <hdR> )  . . . . . . . . . . . . test if <agenL> < <agenR>
*F  LtWord( <hdL>, <hdR> )  . . . . . . . . . . . . test if <wordL> < <wordR>
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
**
**  Special care must be taken, if one argument is a sword because we are not
**  allowed to call 'NewBag' for converting a sword into a word.
*/
Bag       LtAgen (Bag hdL, Bag hdR)
{
    Int            c;

    if ( hdL == hdR )
        return HdFalse;
    else
    {
        /** If <hdL> == <hdR>^-1, inverse is greater. **********************/
        if ( *PTR_BAG( hdL ) == hdR )
        {
            if ( *( (char*) ( PTR_BAG( hdL ) + 1 ) ) == '-' )
                return HdFalse;
            else
                return HdTrue;
        }

        /** Compare the names of the generators. ***************************/
        c = strcmp( (char*)(PTR_BAG(hdL)+1)+1, (char*)(PTR_BAG(hdR)+1)+1 );
        if ( c < 0 )
            return HdTrue;
        else if ( c > 0 )
            return HdFalse;
        else
        {

            /** Two different generators with equal names. *****************/
            if ( *( (char*) ( PTR_BAG( hdL ) + 1 ) ) == '-' )
                hdL = *PTR_BAG( hdL );
            if ( *( (char*) ( PTR_BAG( hdR ) + 1 ) ) == '-' )
                hdR = *PTR_BAG( hdR );
            return ( hdL < hdR ) ? HdTrue : HdFalse;
        }
    }
}

Bag       LtWord (Bag hdL, Bag hdR)
{
    Bag       * ptL,  * ptR,  hdLstL,  hdLstR, hdTmp;
    TypSword          * gtL,  * gtR;
    Int            i,  j,  lnL,  lnR;

    if ( GET_TYPE_BAG( hdL ) == T_WORD && GET_TYPE_BAG( hdR ) == T_WORD )
    {
        if ( GET_SIZE_BAG( hdL ) < GET_SIZE_BAG( hdR ) )  return HdTrue;
        if ( GET_SIZE_BAG( hdL ) > GET_SIZE_BAG( hdR ) )  return HdFalse;

        ptL = PTR_BAG( hdL );
        ptR = PTR_BAG( hdR );
        for ( i = GET_SIZE_BAG( hdL ) / SIZE_HD; i > 0; --i, ++ptL, ++ptR )
            if ( *ptL != *ptR )
                return LtAgen( *ptL, *ptR );
        return HdFalse;
    }
    else if ( GET_TYPE_BAG( hdL ) == T_SWORD && GET_TYPE_BAG( hdR ) == T_SWORD )
    {
        gtL = (TypSword*)( PTR_BAG( hdL ) + 1 );
        lnL  = 0;
        while ( *gtL != -1 )
        {
            lnL  += ( gtL[1] < 0 ) ? -gtL[1] : gtL[1];
            gtL += 2;
        }
        gtR = (TypSword*)( PTR_BAG( hdR ) + 1 );
        lnR  = 0;
        while ( *gtR != -1 )
        {
            lnR  += ( gtR[1] < 0 ) ? -gtR[1] : gtR[1];
            gtR += 2;
        }
        if ( lnL != lnR )
            return ( lnL < lnR ) ? HdTrue : HdFalse;

        hdLstL = PTR_BAG( hdL )[ 0 ];
        hdLstR = PTR_BAG( hdR )[ 0 ];
        if ( GET_TYPE_BAG( hdLstL ) == T_AGGRP )
            hdLstL = HD_WORDS( hdLstL );
        if ( GET_TYPE_BAG( hdLstR ) == T_AGGRP )
            hdLstR = HD_WORDS( hdLstR );
        ptL  = PTR_BAG( hdLstL ) + 1;
        ptR  = PTR_BAG( hdLstR ) + 1;
        gtL = (TypSword*)( PTR_BAG( hdL ) + 1 );
        gtR = (TypSword*)( PTR_BAG( hdR ) + 1 );
        if ( hdLstL == hdLstR )
        {
            while ( *gtL != -1 && gtL[0] == gtR[0] && gtL[1] == gtR[1] )
            {
                gtL += 2;
                gtR += 2;
            }
        }
        else
        {
            while ( *gtL != -1
                    && ( ( ptL[ gtL[0] ] == ptR[ gtR[0] ]
                           && gtL[1] == gtR[1] )
                      || ( ptL[ gtL[0] ] == *PTR_BAG( ptR[ gtR[0] ] )
                           && gtL[1] != -gtR[1]) ) )
            {
                gtL += 2;
                gtR += 2;
            }
        }
        if ( *gtL == -1 )
            return HdFalse;

        hdL = ( gtL[1] < 0 ) ? *PTR_BAG( ptL[ gtL[0] ] ) : ptL[ gtL[0] ];
        hdR = ( gtR[1] < 0 ) ? *PTR_BAG( ptR[ gtR[0] ] ) : ptR[ gtR[0] ];
        if ( hdL != hdR )
            return LtAgen( hdL, hdR );
        lnL = ( gtL[1] < 0 ) ? -gtL[1] : gtL[1];
        lnR = ( gtR[1] < 0 ) ? -gtR[1] : gtR[1];
        if ( lnL < lnR )
        {
            gtL += 2;
            hdL  = ( gtL[1] < 0 ) ? *PTR_BAG( ptL[ gtL[0] ] ) : ptL[ gtL[0] ];
        }
        else
        {
            gtR += 2;
            hdR  = ( gtR[1] < 0 ) ? *PTR_BAG( ptR[ gtR[0] ] ) : ptR[ gtR[0] ];
        }
        return LtAgen( hdL, hdR );
    }
    else if ( GET_TYPE_BAG( hdL ) == T_SWORD && GET_TYPE_BAG( hdR ) == T_WORD )
    {
        gtL = (TypSword*)( PTR_BAG( hdL ) + 1 );
        lnL  = 0;
        while ( *gtL != -1 )
        {
            lnL  += ( gtL[1] < 0 ) ? -gtL[1] : gtL[1];
            gtL += 2;
        }
        lnR = GET_SIZE_BAG( hdR ) / SIZE_HD;
        if ( lnL != lnR )
            return ( lnL < lnR ) ? HdTrue : HdFalse;

        hdLstL = PTR_BAG( hdL )[ 0 ];
        if ( GET_TYPE_BAG( hdLstL ) == T_AGGRP )
            hdLstL = HD_WORDS( hdLstL );
        ptL   = PTR_BAG( hdLstL ) + 1;
        gtL  = (TypSword*)( PTR_BAG( hdL ) + 1 );
        ptR   = PTR_BAG( hdR );
        while ( *gtL != -1 )
        {
            if ( *ptR == ptL[ gtL[0] ] )
            {
                if ( gtL[1] < 0 )
                    return LtAgen( *PTR_BAG( ptL[ gtL[0] ] ), *ptR );
                hdTmp = ptL[ gtL[0] ];
                for ( j = gtL[1]; j > 0; j--, ptR++ )
                    if ( *ptR != hdTmp )
                        return LtAgen( hdTmp, *ptR );
                gtL += 2;
            }
            else if ( *ptR == *PTR_BAG( ptL[ gtL[0] ] ) )
            {
                if ( gtL[1] > 0 )
                    return LtAgen( ptL[ gtL[0] ], *ptR );
                hdTmp = *PTR_BAG( ptL[ gtL[0] ] );
                for ( j = -gtL[1]; j > 0; j--, ptR++ )
                    if ( *ptR != hdTmp )
                        return LtAgen( hdTmp, *ptR );
                gtL += 2;
            }
            else if ( gtL[1] > 0 )
                return LtAgen( ptL[ gtL[0] ], *ptR );
            else if ( gtL[1] < 0 )
                return LtAgen( *PTR_BAG( ptL[ gtL[0] ] ), *ptR );
        }
        return HdFalse;
    }
    else
    {
        if ( EqWord( hdL, hdR ) == HdTrue )
            return HdFalse;
        else
            return ( LtWord( hdR, hdL ) == HdTrue ) ? HdFalse : HdTrue;
    }
}


/****************************************************************************
**
*F  PrSword( <sword> )  . . . . . . . . . . . . . . . . . . . . print a sword
**
**  'PrSword' prints a sparse word in generators/exponent form. The empty word
**  is printed as "IdAgWord".
*/
void        PrSword (Bag hdWrd)
{
    Bag       * ptLst;
    TypSword        * ptWrd;

    ptWrd = (TypSword*)( PTR_BAG( hdWrd ) + 1 );
    if ( ptWrd[ 0 ] == -1 )
    {
        Pr( "IdWord",  0,  0 );
    }
    else
    {
        /** Catch sword with a polycyclic presentation. ********************/
        if ( GET_TYPE_BAG( *PTR_BAG( hdWrd ) ) == T_AGGRP )
            ptLst = PTR_BAG( HD_WORDS( *PTR_BAG( hdWrd ) ) ) + 1;
        else
            ptLst = PTR_BAG( *PTR_BAG( hdWrd ) ) + 1;

        if ( ptWrd[ 1 ] == 1 )
            Pr( "%s", (Int)((char*)(PTR_BAG(ptLst[ptWrd[0]])+1)+1),  0 );
        else
            Pr( "%s^%d",(Int)((char*)(PTR_BAG(ptLst[ptWrd[0]])+1)+1),ptWrd[1] );
        ptWrd += 2;
        while ( ptWrd[ 0 ] != -1 )
        {
            if ( ptWrd[ 1 ] != 1 )
                Pr( "*%s^%d",
                    (Int)((char*)(PTR_BAG(ptLst[ ptWrd[0] ])+1)+1),
                    ptWrd[ 1 ] );
            else
                Pr( "*%s", (Int)((char*)(PTR_BAG(ptLst[ ptWrd[0] ])+1)+1),  0 );
            ptWrd += 2;
        }
    }
}


/****************************************************************************
**
*F  PrWord( <word> )  . . . . . . . . . . . . . . . . . . . . .  print a word
**
**  'PrWord' prints a word, the empty word is printed as "IdWord".  All other
**  words are printed  in  generators/exponent  form,  ie,  "a^-1*a^-1*b"  is
**  printed as "a^-2 * b".
*/
void            PrWord (Bag hdWrd)
{
    Int            nr, i, exp;

    nr = GET_SIZE_BAG( hdWrd ) / SIZE_HD;
    if ( nr == 0 )
    {
        Pr( "IdWord",  0,  0 );
    }
    else
    {
        i = 0;
        while ( i < nr )
        {
            if ( PTR_BAG( hdWrd )[ i ] == 0 )
               Pr( "~",  0,  0 );
            else
            {
               exp = 1;
               while ( i < nr-1 && PTR_BAG( hdWrd )[i] == PTR_BAG( hdWrd )[i+1] )
               {
                   i++;
                   exp++;
               }
               if ( *( (char*) ( PTR_BAG( PTR_BAG( hdWrd )[ i ] ) + 1 ) ) == '-' )
                   exp *= -1;
               if ( exp == 1 )
                   Pr( "%s",
                       (Int) ( (char*)( PTR_BAG( PTR_BAG( hdWrd )[i] ) + 1 ) + 1 ),
                        0 );
               else
                   Pr( "%s^%d",
                       (Int) ( (char*)( PTR_BAG( PTR_BAG( hdWrd )[i] ) + 1 ) + 1 ),
                       (Int) exp );
            }
            if ( i != nr - 1 )
                Pr( "*",  0,  0 );
            i++;
        }
    }
}


/****************************************************************************
**
*F  FunAbstractGenerator( <hdCall> )  . . . . . internal 'AbstractGenerators'
**
**  'FunAbstractGenerator' implements 'AbstractGenerator( <str> )'
**
**  The internal   function   'AbstractGenerator'  creates   a  new  abstract
**  generator.  This  new generator  is  printed using  the <str>  passed  as
**  argument to 'Word'.
*/
Bag       FunAbstractGenerator (Bag hdCall)
{
    Bag       hdStr,  hdWrd,  hdAgn,  hdInv;

    /** Evalute and check the arguments. ***********************************/
    if ( GET_SIZE_BAG( hdCall ) != 2 * SIZE_HD )
        return Error( "usage: AbstractGenerator( <str> )",  0,  0 );
    hdStr = EVAL( PTR_BAG( hdCall )[ 1 ] );
    if ( GET_TYPE_BAG( hdStr ) != T_STRING && ! IsString( hdStr ) )
        return Error( "usage: AbstractGenerator( <str> )",  0,  0 );

    /** Create the two abstract generators <string> and <string>^-1.  The **/
    /** generator will get the name "+<string>",the invers generator will **/
    /** get the name "-<string>". This will be  used in  order  to  print **/
    /** a word as a^-1 * b^2. See 'PrWord' for details.                   **/
    hdAgn = NewBag( T_AGEN, SIZE_HD + GET_SIZE_BAG( hdStr ) + 1 );
    *(char*)( PTR_BAG( hdAgn ) + 1 ) = '\0';
    strncat( (char*) ( PTR_BAG( hdAgn ) + 1 ), "+", 1 );
    strncat( (char*) ( PTR_BAG( hdAgn ) + 1 ),
               (char*) PTR_BAG( hdStr ),
               GET_SIZE_BAG( hdStr ) - 1 );
    hdInv = NewBag( T_AGEN, SIZE_HD + GET_SIZE_BAG( hdStr ) + 1 );
    *(char*)( PTR_BAG( hdInv ) + 1 ) = '\0';
    strncat( (char*) ( PTR_BAG( hdInv ) + 1 ), "-", 1 );
    strncat( (char*) ( PTR_BAG( hdInv ) + 1 ),
               (char*) PTR_BAG( hdStr ),
               GET_SIZE_BAG( hdStr ) - 1 );

    /** The handle of an abstract generator will point to its invers. ******/
    SET_BAG( hdAgn ,  0 ,  hdInv );
    SET_BAG( hdInv ,  0 ,  hdAgn );

    /** Return the one generator word. *************************************/
    hdWrd             = NewBag( T_WORD, SIZE_HD );
    SET_BAG( hdWrd ,  0 ,  hdAgn );

    return hdWrd;
}


/****************************************************************************
**
*F  Words( <hdStr>, <n> ) . . . . . . . . . . . . . . . . . create <n> swords
*F  FunAbstractGenerators( <hdCall> ) . . . . . internal 'AbstractGenerators'
**
**  'FunAbstractGenerators' implements 'AbstractGenerators( <str>, <n> )'
*/
Bag       Words (Bag hdStr, Int n)
{
    Bag       hdLst,  hdAgn,  hdInv,  hdTmp,  hdWrds;
    Bag       * ptTmp;
    Int            i,  j;
    char            str[ 6 ],  * p;

    /** Make a list of <n> swords, create as many abstarct generators. *****/
    hdLst = NewBag( T_LIST, ( n + 1 ) * SIZE_HD );
    SET_BAG( hdLst ,  0 ,  INT_TO_HD( n ) );
    str[ 5 ] = '\0';
    for ( i = 1; i <= n; i++ )
    {
        p = str + 5;
        j = i;
        while ( j > 0 )
        {
            if ( p < str )
                return Error( "Words: integer-string overflow",  0,  0 );
            *--p = j % 10 + '0';
            j   /= 10;
        }
        j = strlen( p );
        hdAgn = NewBag( T_AGEN, SIZE_HD + GET_SIZE_BAG( hdStr ) + j + 1 );
        *(char*)( PTR_BAG( hdAgn ) + 1 ) = '\0';
        strncat( (char*) ( PTR_BAG( hdAgn ) + 1 ), "+", 1 );
        strncat( (char*) ( PTR_BAG( hdAgn ) + 1 ),
                   (char*) PTR_BAG( hdStr ),
                   GET_SIZE_BAG( hdStr ) - 1 );
        strncat( (char*) ( PTR_BAG( hdAgn ) + 1 ), p, j );
        hdInv = NewBag( T_AGEN, SIZE_HD + GET_SIZE_BAG( hdStr ) + j + 1 );
        *(char*)( PTR_BAG( hdInv ) + 1 ) = '\0';
        strncat( (char*) ( PTR_BAG( hdInv ) + 1 ), "-", 1 );
        strncat( (char*) ( PTR_BAG( hdInv ) + 1 ),
                   (char*) PTR_BAG( hdStr ),
                   GET_SIZE_BAG( hdStr ) - 1 );
        strncat( (char*) ( PTR_BAG( hdInv ) + 1 ), p, j );
        SET_BAG( hdAgn ,  0 ,  hdInv );
        SET_BAG( hdInv ,  0 ,  hdAgn );
        SET_BAG( hdLst ,  i ,  hdAgn );
    }
    hdWrds = NewBag( T_LIST, ( n + 1 ) * SIZE_HD );
    SET_BAG( hdWrds ,  0 ,  INT_TO_HD( n ) );
    for ( i = 1; i <= n; i++ )
    {
        hdTmp = NewBag( T_SWORD, SIZE_HD + 3 * SIZE_SWORD );
        ptTmp = PTR_BAG( hdTmp ) + 1;
        ptTmp[ -1 ] = hdLst;
        ( (TypSword*) ptTmp )[ 0 ] = i - 1;
        ( (TypSword*) ptTmp )[ 1 ] = 1;
        ( (TypSword*) ptTmp )[ 2 ] = -1;

        SET_BAG( hdWrds ,  i ,  hdTmp );
    }

    return hdWrds;
}
    
Bag       FunAbstractGenerators (Bag hdCall)
{
    Bag       hdStr, hdN;
    Int            n;

    if ( GET_SIZE_BAG( hdCall ) != 3 * SIZE_HD )
        return Error( "usage: AbstractGenerators( <str>, <n> )",  0,  0 );
    hdStr = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdN   = EVAL( PTR_BAG( hdCall )[ 2 ] );
    if ( (GET_TYPE_BAG( hdStr ) != T_STRING && ! IsString( hdStr ))
      || GET_TYPE_BAG( hdN ) != T_INT )
        return Error( "usage: AbstractGenerators( <str>, <n> )",  0,  0 );
    n = HD_TO_INT( hdN );
    if ( n <= 0 )
        return Error( "number of words <n> must be positive",  0,  0 );
    if ( n > MAX_SWORD_NR )
        return Error( "number of words <n> must be less than %d",
                      MAX_SWORD_NR - 1,  0 );

    return Words( hdStr, n );
}


/****************************************************************************
**
*F  FunLenWord( <hdCall> ) . . . . . internal function 'LengthWord( <word> )'
**
**  The internal function  'LengthWord'  computes the length of <word>. Since
**  words of T_WORD are stored in fully expanded form this is simply the
**  size, while we must count T_SWORD. 
*/
Bag       FunLenWord (Bag hdCall)
{
    Bag       hdWord;
    TypSword        * ptSwrd;
    Int            len;

    /** Evaluate and check the arguments. **********************************/
    if ( GET_SIZE_BAG( hdCall ) != 2 * SIZE_HD )
        return Error( "usage: LengthWord( <word> )",  0,  0 );
    hdWord = EVAL( PTR_BAG( hdCall )[ 1 ] );
    if ( GET_TYPE_BAG( hdWord ) == T_WORD )
        return INT_TO_HD( GET_SIZE_BAG( hdWord ) / SIZE_HD );
    else if ( GET_TYPE_BAG( hdWord ) == T_SWORD )
    {
        len = 0;
        ptSwrd = (TypSword*)( PTR_BAG( hdWord ) + 1 );
        while ( ptSwrd[0] != -1 )
        {
            len    += ( ptSwrd[1] < 0 ) ? -ptSwrd[1] : ptSwrd[1];
            ptSwrd += 2;
        }
        return INT_TO_HD( len );
    }
    else
        return Error( "usage: LengthWord( <word> )",  0,  0 );

}


/****************************************************************************
**
*F  FunSubword( <hdCall> )  . . . . . . .  internal function 'Subword( ... )'
**
**  The internal function Subword( <word>, <from>, <to> ) is  used to get the
**  subword of <word> starting at <from> and ending at <to>. Indexing is done
**  with origin 1. The new word is returned.
*/
Bag   FunSubword (Bag hdCall)
{
    Bag       hdWord, hdFrom, hdTo, hdRes;
    Int            i, toVal, fromVal;

    /** Evaluate and check the arguments.                                 **/
    if ( GET_SIZE_BAG( hdCall ) != 4 * SIZE_HD )
        return Error( "usage: Subword( <word>, <from>, <to> )",  0,  0 );
    hdWord  = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdFrom  = EVAL( PTR_BAG( hdCall )[ 2 ] );
    hdTo    = EVAL( PTR_BAG( hdCall )[ 3 ] );
    if ( GET_TYPE_BAG( hdWord ) == T_SWORD )
        hdWord = WordSword( hdWord );
    if ( GET_TYPE_BAG( hdWord )    != T_WORD
         || GET_TYPE_BAG( hdFrom ) != T_INT
         || GET_TYPE_BAG( hdTo )   != T_INT )
    {
        return Error( "usage: Subword( <word>, <from>, <to> )",  0,  0 );
    }
    fromVal = HD_TO_INT( hdFrom );
    if ( fromVal <= 0 || GET_SIZE_BAG( hdWord ) / SIZE_HD < fromVal )
        return Error( "Subword: illegal <from> value",  0,  0 );
    toVal = HD_TO_INT( hdTo );
    if ( toVal < fromVal || GET_SIZE_BAG( hdWord )/SIZE_HD < toVal )
        return Error( "Subword: illegal <to> value",  0,  0 );

    /** Allocate space for the result.                                    **/
    hdRes = NewBag( T_WORD, ( toVal - fromVal + 1 ) * SIZE_HD );
    for ( i = fromVal; i <= toVal; ++i )
        SET_BAG( hdRes ,  i - fromVal ,  PTR_BAG( hdWord )[ i - 1 ] );

    return hdRes;
}


/****************************************************************************
**
*F  FunSubs( <hdCall> ) . . . . . . internal function 'SubsitutedWord( ... )'
**
**  The  internal   function  'SubstitutedWord( <word>, <from>, <to>, <by> )'
**  replaces  the subword of <word> starting at position <from> and ending at
**  position  <to> by the word <by>. In other words
**        SubsitutedWord( <word>, <from>, <to>, <by>)
**  is:
**        Subword( <word>, 1, <from> - 1 ) *
**        <by> *
**        Subword( <word>, <to> + 1, length(<word>) ).
**
**  Indexing is done with origin 1. The new word is returned.
*/
Bag       FunSubs(Bag hdCall)
{
    register Bag      hdWord,  hdFrom,  hdTo,  hdBy;
             Bag      hdRes;
    register Bag      * ptWord,  * ptRes;
             Int           szRes,  i;
             Int           fromVal, toVal;

    /** Evaluate and check the arguments.                                 **/
    if ( GET_SIZE_BAG(hdCall) != 5 * SIZE_HD )
        return Error(
           "usage: SubstitutedWord( <word>, <from>, <to>, <by> )",  0, 0 );
    hdWord = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdFrom = EVAL( PTR_BAG( hdCall )[ 2 ] );
    hdTo   = EVAL( PTR_BAG( hdCall )[ 3 ] );
    hdBy   = EVAL( PTR_BAG( hdCall )[ 4 ] );
    if ( GET_TYPE_BAG( hdWord ) == T_SWORD )
        hdWord = WordSword( hdWord );
    if ( GET_TYPE_BAG( hdBy ) == T_SWORD )
        hdBy = WordSword( hdBy );
    if ( GET_TYPE_BAG( hdWord )    != T_WORD
         || GET_TYPE_BAG( hdFrom ) != T_INT
         || GET_TYPE_BAG( hdTo )   != T_INT
         || GET_TYPE_BAG( hdBy )   != T_WORD )
    {
        return Error(
           "usage: SubstitutedWord( <word>, <from>, <to>, <by> )",  0, 0 );
    }
    fromVal = HD_TO_INT( hdFrom );
    toVal   = HD_TO_INT( hdTo );
    if ( fromVal <= 0  || GET_SIZE_BAG(hdWord)/SIZE_HD < fromVal )
        return Error( "SubstitutedWord: illegal <from> value",  0,  0 );
    if ( toVal < fromVal || GET_SIZE_BAG(hdWord)/SIZE_HD < toVal )
        return Error( "SubstitutedWord: illegal <to> value",  0,  0 );
    szRes  = GET_SIZE_BAG( hdWord ) + GET_SIZE_BAG( hdBy ) - SIZE_HD * (toVal - fromVal + 1);
    hdRes  = NewBag( T_WORD, szRes );
    ptWord = PTR_BAG( hdWord );
    ptRes  = PTR_BAG( hdRes );
    for ( i = fromVal; i > 1; --i ) {
        *ptRes++ = *ptWord++;
    }
    ptWord = PTR_BAG( hdBy );
    ptRes--;
    while ( PTR_BAG( hdRes ) <= ptRes &&
            ptWord < PTR_BAG( hdBy ) + ( GET_SIZE_BAG( hdBy ) / SIZE_HD ) &&
            *PTR_BAG( *ptWord ) == *ptRes )
    {
        ptWord++;
        *ptRes-- = 0;
        szRes    = szRes - 2 * SIZE_HD;
    }
    ptRes++;
    while ( ptWord < PTR_BAG( hdBy ) + ( GET_SIZE_BAG( hdBy ) / SIZE_HD ) )
        *ptRes++ = *ptWord++;
        ptWord   = PTR_BAG( hdWord ) + toVal;
        ptRes--;
        while ( PTR_BAG( hdRes ) <= ptRes &&
            ptWord < PTR_BAG( hdWord ) + ( GET_SIZE_BAG( hdWord ) / SIZE_HD ) &&
            PTR_BAG( *ptWord )[ 0 ] == *ptRes )
        {
            ptWord++;
            *ptRes-- = 0;
            szRes    = szRes - 2 * SIZE_HD;
        }
        ptRes++;
        while ( ptWord < PTR_BAG( hdWord ) + ( GET_SIZE_BAG( hdWord ) / SIZE_HD ) )
            *ptRes++ = *ptWord++;
        Resize( hdRes, szRes );
        return hdRes;
}


/****************************************************************************
**
*F  FunPosWord( <hdCall> )  . . . . . internal function 'PositionWord( ... )'
**
**  This  is  the  internal  function 'PositionWord( <word>, <sub>, <from> )'
**  called to find the first subword of  <word>  matching  <sub> starting  at
**  position <from>.  'PositionWord'  returns  the  index  of  the  position.
**  Indexing is  done with origin 1. Thus
**       PositionWord( <word>, <sub>, <from> )
**  is the smallest integer <ind> larger than <from> such that
**       Subword( <word>, <ind>, <ind> + LengthWord( <sub> ) - 1 ) = <sub>.
**
**  If  no  match  of  the  word   <sub>  is  found in  <word>  at all  0  is
**  returned. For example  'PositionWord( a^4*b*a^4*b*a^4, a^2*b, 4 )'  is 8,
**  and 'PositionWord( a^4, b, 1 )' is 0.
**
**  If the optional parameter <from> is omitted, 1 is assumed.
**
**  This function might use a more  clever  string  matching  algorithm, like
**  Boyer-Moore or Knuth-Morrison-Pratt but since the alphabet and the <word>
**  are likely to be small it is not clear what could  be  gained  from that.
*/
Bag       FunPosWord (Bag hdCall)
{
             Bag      hdWord,  hdSub,  hdFrom;
    register Bag      * ptWord,  * ptSub;
    register Int           i,  j, fromVal, endVal;

    /** Evaluate and check arguments.                                     **/
    if ( GET_SIZE_BAG( hdCall ) != 4 * SIZE_HD && GET_SIZE_BAG( hdCall ) != 3 * SIZE_HD )
      return Error("usage: PositionWord( <word>, <sub>[, <from>] )",  0, 0 );
    hdWord = EVAL( PTR_BAG( hdCall )[ 1 ] );
    if ( GET_TYPE_BAG( hdWord ) == T_SWORD )
        hdWord = WordSword( hdWord );
    if ( GET_TYPE_BAG( hdWord ) != T_WORD )
      return Error("usage: PositionWord( <word>, <sub>[, <from>] )",  0, 0 );
    hdSub = EVAL( PTR_BAG( hdCall )[ 2 ] );
    if ( GET_TYPE_BAG( hdSub ) != T_WORD )
      return Error("usage: PositionWord( <word>, <sub>[, <from>] )",  0, 0 );
    if ( GET_SIZE_BAG( hdCall ) == 4 * SIZE_HD ) {
        hdFrom = EVAL( PTR_BAG(hdCall)[ 3 ] );
        if ( GET_TYPE_BAG( hdFrom ) != T_INT )
            return Error(
                "usage: PositionWord( <word>, <sub>[, <from>] )",  0,  0 );
        fromVal = HD_TO_INT( hdFrom );
    } else
        fromVal = 1;
    if ( fromVal < 1 )
        return Error( "PositionWord: illegal <from> value",  0,  0 );

    /** Loop from <from> to the last possible index.                      **/
    endVal = ( (Int) ( GET_SIZE_BAG(hdWord)-GET_SIZE_BAG(hdSub) ) / (Int) SIZE_HD + 1 );
    for ( i = fromVal; i <= endVal; ++i ) {

        /** Test for match.                                               **/
        ptWord = PTR_BAG( hdWord ) + i - 1;
        ptSub  = PTR_BAG( hdSub );
        for ( j = 0; j < GET_SIZE_BAG( hdSub ) / SIZE_HD; ++j ) {
            if ( *ptSub++ != *ptWord++ )
                break;
        }

        /** We have found a match, return index                           **/
        if ( j == GET_SIZE_BAG( hdSub ) / SIZE_HD ) {
            return INT_TO_HD( i );
        }

    }

    /** We haven't found the substring, return 0 to indicate failure.     **/
    return HdFalse;
}


/****************************************************************************
**
*F  FunIsWord( <hdCall> ) . . . . . . . . internal function 'IsWord( <obj> )'
**
**  'IsWord'  returns  'true'  if the object <obj> is a word in abstarct gens
**  and 'false' otherwise.
**
**  May cause an error if <obj> is an unbound variable.
*/
Bag       FunIsWord (Bag hdCall)
{
    Bag       hdObj;

    /** Evaluate and check the argument. ***********************************/
    if ( GET_SIZE_BAG( hdCall ) != 2 * SIZE_HD )
        return Error( "usage: IsWord( <obj> )",  0,  0 );
    hdObj = EVAL( PTR_BAG(hdCall)[1] );
    if ( hdObj == HdVoid )
        return Error( "IsWord: function must return a value",  0,  0 );

    /** Return 'true' if <obj> is a word and 'false' otherwise. ************/
    if ( GET_TYPE_BAG( hdObj ) == T_WORD || GET_TYPE_BAG( hdObj ) == T_SWORD )
        return HdTrue;
    else
        return HdFalse;
}


/****************************************************************************
**
*F  FunEliminated( <hdCall> ) . . . internal function 'EliminatedWord( ... )'
**
**  This is  the  internal  function  'EliminatedWord( <word>, <gen>, <by> )'
**  called  to replace all occurrences of a generator <gen>  in  <word>  with
**  the word <by>.
**
**  This is faster  than  using  'MappedWord'  with  just  one  new  abstract
**  generator.
*/
Bag       FunEliminated(Bag hdCall)
{
    Bag       hdWord,  * ptWord,  hdGen,  hdInv,  hdBy,  * ptBy;
    Bag       hdRes,  * ptRes;
    Int            szRes,  i;

    /** Check and evaluate the arguments.                                 **/
    if ( GET_SIZE_BAG( hdCall ) != 4 * SIZE_HD )
        return Error( "usage: EliminatedWord( <word>, <gen>, <by> )",
                 0,  0 );
    hdWord = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdGen  = EVAL( PTR_BAG( hdCall )[ 2 ] );
    hdBy   = EVAL( PTR_BAG( hdCall )[ 3 ] );
    if ( GET_TYPE_BAG( hdWord ) == T_SWORD )
        hdWord = WordSword( hdWord );
    if ( GET_TYPE_BAG( hdGen ) == T_SWORD )
        hdGen = WordSword( hdGen );
    if ( GET_TYPE_BAG( hdBy ) == T_SWORD )
        hdBy = WordSword( hdBy );
    if ( GET_TYPE_BAG( hdWord )    != T_WORD
         || GET_TYPE_BAG( hdGen )  != T_WORD
         || GET_SIZE_BAG( hdGen )  != SIZE_HD
         || GET_TYPE_BAG( hdBy )   != T_WORD )
    {
        return Error( "usage: EliminatedWord( <word>, <gen>, <by> )",
                 0,  0 );
    }
    hdGen = PTR_BAG( hdGen )[ 0 ];
    hdInv = PTR_BAG( hdGen )[ 0 ];

    /** Compute a bound for the results size assuming nothing cancels.    **/
    ptWord = PTR_BAG( hdWord );
    szRes  = GET_SIZE_BAG( hdWord );
    for ( i = 0; i < GET_SIZE_BAG( hdWord ) / SIZE_HD; ++i ) {
        if ( ptWord[ i ] == hdGen || ptWord[ i ] == hdInv )
            szRes = szRes + GET_SIZE_BAG(hdBy) - SIZE_HD;
    }

    /** Allocate the bag for the result and set up pointer.               **/
    hdRes  = NewBag( T_WORD, szRes );
    ptRes  = PTR_BAG( hdRes );
    ptWord = PTR_BAG( hdWord );

    while ( ptWord < PTR_BAG( hdWord ) + GET_SIZE_BAG( hdWord ) / SIZE_HD ) {
        if ( *ptWord == hdGen ) {

            /** Insert <by>.                                              **/
            ptBy = PTR_BAG( hdBy );
            while ( ptBy < PTR_BAG( hdBy ) + GET_SIZE_BAG( hdBy ) / SIZE_HD ) {
                if ( ptRes > PTR_BAG( hdRes ) &&
                     ptRes[-1] == PTR_BAG(ptBy[0])[0] )
                {
                    ptRes--;
                    szRes = szRes - 2 * SIZE_HD;
                } else {
                    *ptRes = *ptBy;
                    ptRes++;
                }
                ptBy++;
            }
        } else if ( *ptWord == hdInv ) {

            /** Insert the inverse of <by> now.                           **/
            ptBy = PTR_BAG( hdBy ) + GET_SIZE_BAG( hdBy ) / SIZE_HD - 1;
            while ( ptBy >= PTR_BAG( hdBy ) ) {
                if ( ptRes > PTR_BAG( hdRes ) && ptRes[-1] == *ptBy ) {
                    ptRes--;
                    szRes = szRes - 2 * SIZE_HD;
                } else {
                    *ptRes = PTR_BAG( ptBy[ 0 ] )[ 0 ];
                    ptRes++;
                }
                ptBy--;
            }
        } else {

            /** Check if this generator cancel in the result.             **/
            if ( ptRes > PTR_BAG(hdRes) &&
                 ptRes[-1] == PTR_BAG(ptWord[0])[0] )
            {
                ptRes--;
                szRes = szRes - 2 * SIZE_HD;
            } else {
                *ptRes = *ptWord;
                ptRes++;
            }
        }
        ptWord++;
    }

    /** Make the result have the right size and return it.                **/
    Resize( hdRes, szRes );
    return hdRes;
}


/****************************************************************************
**
*F  FunExpsum( <hdCall> ) . . . .  internal function 'ExponentSumWord( ... )'
**
**  This is the internal function  'ExponentSumWord( <word>, <gen> )'  called
**  to compute the sum of the exponents of all occurrences of  the  generator
**  <gen> in <word>.
*/
Bag       FunExpsum (Bag hdCall)
{
    Bag       hdWord,  * ptWord,  hdGen,  hdInv;
    Int            expsum,  i;

    /** Evaluate and check the arguments. **********************************/
    if ( GET_SIZE_BAG( hdCall ) != 3 * SIZE_HD )
        return Error( "usage: ExponentSumWord( <word>, <gen> )",  0,  0 );
    hdWord = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdGen  = EVAL( PTR_BAG( hdCall )[ 2 ] );

    /** Convert swords into words. *****************************************/
    if ( GET_TYPE_BAG( hdWord ) == T_SWORD )
        hdWord = WordSword( hdWord );
    if ( GET_TYPE_BAG( hdGen ) == T_SWORD )
        hdGen = WordSword( hdGen );

    if ( GET_TYPE_BAG( hdWord )    != T_WORD
         || GET_TYPE_BAG( hdGen )  != T_WORD
         || GET_SIZE_BAG( hdGen )  != SIZE_HD )
    {
        return Error( "usage: ExponentSumWord( <w>, <g> )",  0,  0 );
    }

    /*N This can be done for swords without converting into words, but .. **/
    hdGen  = PTR_BAG( hdGen )[ 0 ];
    hdInv  = PTR_BAG( hdGen )[ 0 ];
    expsum = 0;
    ptWord = PTR_BAG( hdWord );
    for ( i = 0; i < GET_SIZE_BAG( hdWord ) / SIZE_HD; ++i )
    {
        if ( ptWord[ i ] == hdGen )  expsum++;
        if ( ptWord[ i ] == hdInv )  expsum--;
    }
    return INT_TO_HD( expsum );
}


/****************************************************************************
**
*F  FunMappedWord( <hdCall> ) . . . . . internal function 'MappedWord( ... )'
**
**  ...something about the function...
*/
Bag       FunMappedWord (Bag hdCall)
{
    Int            i,  k,  exp;
    Bag       * ptOld,  hdGenOld = 0,  hdGen,  hdTmp2 = 0;
    Bag       hdWord,  hdOld,  hdNew,  hdNewWord,  hdTmp;
    Int            lenOld,  lenNew,  lenWord;
    char            * usage = "usage: MappedWord( <word>, <old>, <new> )";

    /** Evaluate and check the arguments. **********************************/
    if ( GET_SIZE_BAG( hdCall ) != 4 * SIZE_HD )
        return Error( usage,  0,  0 );
    hdWord = EVAL( PTR_BAG( hdCall )[ 1 ] );
    hdOld  = EVAL( PTR_BAG( hdCall )[ 2 ] );
    hdNew  = EVAL( PTR_BAG( hdCall )[ 3 ] );
    if ( GET_TYPE_BAG( hdWord ) == T_SWORD )
        hdWord = WordSword( hdWord );
    if ( GET_TYPE_BAG(hdWord) != T_WORD || ! IsList(hdOld) || ! IsList(hdNew) )
        return Error( usage,  0,  0 );

    /** Generators must be one generator words *****************************/
    lenOld = LEN_LIST( hdOld );
    lenNew = LEN_LIST( hdNew );
    if ( lenNew != lenOld )
        return Error( "needs lists of equal length",  0,  0 );
    if ( lenNew < 1 )
        return Error( "needs at least one generator",  0,  0 );
    for ( i = 1; i <= lenNew; i++ )
    {
        /*N  This is stupid, but for the moment ... ************************/
        hdTmp = PTR_BAG( hdOld )[ i ];
        if ( GET_TYPE_BAG( hdTmp ) == T_SWORD )
        {
            hdTmp = WordSword( hdTmp );
            SET_BAG( hdOld ,  i ,  hdTmp );
        }
        if ( GET_TYPE_BAG( hdTmp ) != T_WORD || GET_SIZE_BAG( hdTmp ) != SIZE_HD )
            return Error( "needs words of length 1",  0,  0 );
    }              

    /** Run through the word, use POW and PROD in order  to  replace  the **/
    /** abstract generators.                                              **/
    hdTmp     = PTR_BAG( hdNew )[ 1 ];
    hdNewWord = POW( hdTmp, INT_TO_HD( 0 ) );
    i         = 0;
    lenWord   = GET_SIZE_BAG( hdWord ) / SIZE_HD;
    while ( i < lenWord )
    {
        ptOld = PTR_BAG( hdOld );
        hdGen = PTR_BAG( hdWord )[ i ];

        /** Search the abstract generator <hdGen> in the list <ptOld>. *****/
        for ( k = lenNew;  0 < k;  k-- )
        {
            hdGenOld = PTR_BAG( ptOld[ k ] )[ 0 ];
            if ( hdGen == hdGenOld || hdGen == *PTR_BAG( hdGenOld ) )
                break;
        }

        /** We have found the generator, now get the exponent. *************/
        exp = 1;
        while ( i < lenWord - 1 && hdGen == PTR_BAG( hdWord )[ i + 1 ] )
        {
            i++;
            exp++;
        }

        /** Add the factor to the new word. ********************************/
        if ( k < 1 )
        {

            /** This is really a hack, but for the moment... ***************/
            if ( hdTmp2 == 0 || hdTmp2 == hdNewWord )
                hdTmp2 = NewBag( T_WORD, SIZE_HD );
            SET_BAG( hdTmp2 , 0,  hdGen );
            if ( exp == 1 )
                hdNewWord = PROD( hdNewWord, hdTmp2 );
            else
            {
                hdTmp = POW( hdTmp2, INT_TO_HD( exp ) );
                hdNewWord = PROD( hdNewWord, hdTmp );
            }
        }
        else
        {
            if ( hdGen != hdGenOld )
                exp *= -1;
            if ( exp == 1 )
                hdNewWord = PROD( hdNewWord, PTR_BAG( hdNew )[ k ] );
            else
            {
                hdTmp = POW( PTR_BAG( hdNew )[ k ], INT_TO_HD( exp ) );
                hdNewWord = PROD( hdNewWord, hdTmp );
            }
        }
        i++;
    }
    return hdNewWord;
}


/****************************************************************************
**
*V  HdIdWord  . . . . . . . . . . . . . . . . . . . . . trivial abstract word
*F  InitWord()  . . . . . . . . . . . . . . . . . . .  initialize word module
**
**  Is called during the initialization of GAP to initialize the word module.
*/
Bag       HdIdWord;

void            InitWord (void)
{
    Int            typeL, typeR;

    InstEvFunc( T_WORD,  EvWord  );
    InstEvFunc( T_SWORD, EvWord  );
    InstPrFunc( T_WORD,  PrWord  );
    InstPrFunc( T_SWORD, PrSword );

    for ( typeL = T_WORD; typeL <= T_SWORD; typeL++ )
    {
        for ( typeR = T_WORD; typeR <= T_SWORD; typeR++ )
        {
            TabProd[ typeL ][ typeR ] = ProdWord;
            TabQuo [ typeL ][ typeR ] = QuoWord;
            TabMod [ typeL ][ typeR ] = ModWord;
            TabPow [ typeL ][ typeR ] = PowWW;
            TabComm[ typeL ][ typeR ] = CommWord;
            TabEq  [ typeL ][ typeR ] = EqWord;
            TabLt  [ typeL ][ typeR ] = LtWord;
        }
        TabPow[ typeL ][ T_INT ] = PowWI;
    }

    InstIntFunc( "AbstractGenerator",   FunAbstractGenerator  );
    InstIntFunc( "AbstractGenerators",  FunAbstractGenerators );
    InstIntFunc( "LengthWord",          FunLenWord            );
    InstIntFunc( "Subword",             FunSubword            );
    InstIntFunc( "SubstitutedWord",     FunSubs               );
    InstIntFunc( "PositionWord",        FunPosWord            );
    InstIntFunc( "IsWord",              FunIsWord             );
    InstIntFunc( "ExponentSumWord",     FunExpsum             );
    InstIntFunc( "MappedWord",          FunMappedWord         );
    InstIntFunc( "EliminatedWord",      FunEliminated         );

    HdIdWord = NewBag( T_WORD, 0 );
    InstVar( "IdWord", HdIdWord );

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



