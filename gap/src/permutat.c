/****************************************************************************
**
*A  permutat.c                  GAP source                   Martin Schoenert
**                                                           & Alice Niemeyer
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file implements the permutation type,  its operations and functions.
**
**  Mathematically a permutation is a bijective mapping  of a finite set onto
**  itself.  In \GAP\ this subset must always be of the form [ 1, 2, .., N ],
**  where N is at most $2^16$.
**
**  Internally a permutation  is viewed as a mapping  of [ 0,  1,  .., N-1 ],
**  because in C indexing of  arrays is done  with the origin  0 instad of 1.
**  A permutation is represented by a bag of type 'T_PERM' of the form
**
**      +-------+-------+-------+-------+- - - -+-------+-------+
**      | image | image | image | image |       | image | image |
**      | of  0 | of  1 | of  2 | of  3 |       | of N-2| of N-1|
**      +-------+-------+-------+-------+- - - -+-------+-------+
**
**  The entries of the bag are of type  'UShort' (defined in 'system.h' as an
**  at least 16 bit   wide unsigned integer  type).   The first entry is  the
**  image of 0, the second is the image of 1, and so  on.  Thus, the entry at
**  C index <i> is the image of <i>, if we view the permutation as mapping of
**  [ 0, 1, 2, .., N-1 ] as described above.
**
**  Permutations are never  shortened.  For  example, if  the product  of two
**  permutations of degree 100 is the identity, it  is nevertheless stored as
**  array of length 100, in  which the <i>-th  entry is of course simply <i>.
**  Testing whether a product has trailing  fixpoints would be pretty costly,
**  and permutations of equal degree can be handled by the functions faster.
**
**
*N  13-Jan-91 martin should add 'CyclesPerm', 'CycleLengthsPerm'
*/

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "scanner.h"             /* reading of tokens and printing  */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */
#include        "integer4.h"
#include        "list.h"                /* list package                    */

#include        "permutat.h"            /* declaration part of the package */


/****************************************************************************
**
*T  TypPoint16  . . . . . . . . . . . . . .  operation domain of permutations
*T  TypPoint32  . . . . . . . . . . . . . .  operation domain of permutations
**
**  This is the type of points upon which permutations  are  acting  in \GAP.
**  It is defined as 1...65536, actually 0...65535 due to the index shifting.
*/
typedef unsigned short  TypPoint16;
#ifndef SYS_IS_64_BIT
typedef unsigned long   TypPoint32;
#else
typedef unsigned int   TypPoint32;
#endif

/****************************************************************************
**
*F  IMAGE(<I>,<PT>,<DG>)  . . . . . .  image of <I> under <PT> of degree <DG>
**
**  'IMAGE'  returns the  image of the   point <I> under  the permutation  of
**  degree <DG> pointed to  by <PT>.   If the  point  <I> is greater  than or
**  equal to <DG> the image is <I> itself.
**
**  'IMAGE' is  implemented as a macro so  do not use  it with arguments that
**  have side effects.
*/
#define IMAGE(I,PT,DG)  (((I) < (DG)) ? (PT)[(I)] : (I))


/****************************************************************************
**
*V  HdPerm  . . . . . . . handle of the buffer bag of the permutation package
**
**  'HdPerm' is the handle of  a bag of   type 'T_PERM', which is created  at
**  initialization time of this  package.  Functions in  this package can use
**  this bag for   whatever purpose they want.    They have to  make sure  of
**  course that it is large enough.
*/
Bag               HdPerm;


/****************************************************************************
**
*F  EvPerm( <hdPerm> )  . . . . . . . . . . . evaluate a permutation constant
**
**  'EvPerm'  returns  the value  of    the permutation  <hdPerm>.    Because
**  permutations   are constants and  thus  selfevaluating  this just returns
**  <hdPerm>.
*/
Bag       EvPerm (Bag hdPerm)
{
    return hdPerm;
}


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
Bag       EvMakeperm (Bag hdPerm)
{
    Bag           hdRes;          /* handle of the result            */
    Bag           hdCyc;          /* handle of one cycle of hdPerm   */
    Bag           hd;             /* temporary handle                */
    UInt       psize;          /* physical size of permutation    */
    UInt       lsize;          /* logical size of permutation     */
    UInt       nsize;          /* new size if resizing            */
    TypPoint32          first;          /* first point in a cycle          */
    TypPoint32          last;           /* last point seen in a cycle      */
    TypPoint32          curr;           /* current point seen in a cycle   */
    Int                i, j, k;        /* loop variables                  */

    /* create the perm bag, the sum of the cycle length is a good estimate */
    /* for the neccessary size of the perm bag, except for (1,10000).      */
    psize = 0;
    for ( i = 0; i < GET_SIZE_BAG(hdPerm)/SIZE_HD; i++ )
        psize += GET_SIZE_BAG( PTR_BAG(hdPerm)[i] ) / SIZE_HD;
    if ( psize <= 65536 ) {
        hdRes = NewBag(T_PERM16,(UInt)(psize*sizeof(TypPoint16)));
        for ( i = 0; i < psize; i++ )
            ((TypPoint16*)PTR_BAG(hdRes))[i] = i;
    }
    else {
        hdRes = NewBag(T_PERM32,(UInt)(psize*sizeof(TypPoint32)));
        for ( i = 0; i < psize; i++ )
            ((TypPoint32*)PTR_BAG(hdRes))[i] = i;
    }
    lsize = 0;

    /* loop over all cycles                                                */
    for ( i = 0; i < GET_SIZE_BAG(hdPerm)/SIZE_HD; i++ ) {
        hdCyc = PTR_BAG(hdPerm)[i];

        /* loop through this cycle                                         */
        first = 0;
        last = 0;
        for ( j = 0; j < GET_SIZE_BAG(hdCyc)/SIZE_HD; j++ ) {

            /* evaluate and check this entry                               */
            hd = EVAL( PTR_BAG(hdCyc)[j] );
            if ( GET_TYPE_BAG(hd) != T_INT || HD_TO_INT(hd) <= 0 )
                return Error("Perm: <point> must be an positive int",0,0);
            curr = HD_TO_INT(hd)-1;

            /* if neccessary resize the permutation bag                    */
            if ( psize < curr+1 ) {
                if ( psize+256 < curr+1 )       nsize = curr+1;
                else if ( psize+256 <= 65536 )  nsize = psize + 256;
                else if ( curr+1 <= 65536 )     nsize = 65536;
                else                            nsize = psize + 1024;
                if ( psize <= 65536 && 65536 < nsize ) {
                    Retype(hdRes,T_PERM32);
                    Resize(hdRes,(UInt)(nsize*sizeof(TypPoint32)));
                    for ( k = psize-1; 0 <= k; k-- ) {
                        ((TypPoint32*)PTR_BAG(hdRes))[k]
                            = ((TypPoint16*)PTR_BAG(hdRes))[k];
                    }
                    for ( ; psize < nsize; psize++ )
                        ((TypPoint32*)PTR_BAG(hdRes))[psize] = psize;
                }
                else if ( psize <= 65536 ) {
                    Resize(hdRes,(UInt)(nsize*sizeof(TypPoint16)));
                    for ( ; psize < nsize; psize++ )
                        ((TypPoint16*)PTR_BAG(hdRes))[psize] = psize;
                }
                else {
                    Resize(hdRes,(UInt)(nsize*sizeof(TypPoint32)));
                    for ( ; psize < nsize; psize++ )
                        ((TypPoint32*)PTR_BAG(hdRes))[psize] = psize;
                }
            }
            if ( lsize < curr+1 ) {
                lsize = curr+1;
            }

            /* make sure we haven't seen this point before                 */
            if ( (j != 0 && last == curr)
              || (psize <= 65536 && ((TypPoint16*)PTR_BAG(hdRes))[curr]!=curr)
              || (65536 < psize && ((TypPoint32*)PTR_BAG(hdRes))[curr]!=curr) ) {
                return Error("Perm: cycles must be disjoint",0,0);
            }

            /* unless this is the first, enter prev point at this position */
            if ( j == 0 )
                first = curr;
            else if ( psize <= 65536 )
                ((TypPoint16*)PTR_BAG(hdRes))[last] = curr;
            else
                ((TypPoint32*)PTR_BAG(hdRes))[last] = curr;

            /* the current point is the next last point                    */
            last = curr;

        }

        /* enter the last point in the cycle                               */
        if ( psize <= 65536 )
            ((TypPoint16*)PTR_BAG(hdRes))[last] = first;
        else
            ((TypPoint32*)PTR_BAG(hdRes))[last] = first;

    }

    /* shorten the result and return it                                    */
    if ( psize <= 65536 )
        Resize( hdRes, (UInt)(lsize * sizeof(TypPoint16)) );
    else
        Resize( hdRes, (UInt)(lsize * sizeof(TypPoint32)) );
    return hdRes;
}


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
Bag       ProdPP (Bag hdL, Bag hdR)
{
    Bag           hdP;            /* handle of the product (result)  */
    UInt       degP;           /* degree of the product           */
    TypPoint16          * ptP;          /* pointer to the product          */
    UInt       degL;           /* degree of the left operand      */
    TypPoint16          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint16          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* compute the size of the result and allocate a bag                   */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint16);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint16);
    degP = degL < degR ? degR : degL;
    hdP  = NewBag( T_PERM16, (UInt)(degP * sizeof(TypPoint16)) );

    /* set up the pointers                                                 */
    ptL = (TypPoint16*)PTR_BAG(hdL);
    ptR = (TypPoint16*)PTR_BAG(hdR);
    ptP = (TypPoint16*)PTR_BAG(hdP);

    /* if the left (inner) permutation has smaller degree, it is very easy */
    if ( degL <= degR ) {
        for ( p = 0; p < degL; p++ )
            *(ptP++) = ptR[ *(ptL++) ];
        for ( p = degL; p < degR; p++ )
            *(ptP++) = ptR[ p ];
    }

    /* otherwise we have to use the macro 'IMAGE'                          */
    else {
        for ( p = 0; p < degL; p++ )
            *(ptP++) = IMAGE( ptL[ p ], ptR, degR );
    }

    /* return the result                                                   */
    return hdP;
}


Bag       ProdPQ (Bag hdL, Bag hdR)
{
    Bag           hdP;            /* handle of the product (result)  */
    UInt       degP;           /* degree of the product           */
    TypPoint32          * ptP;          /* pointer to the product          */
    UInt       degL;           /* degree of the left operand      */
    TypPoint16          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint32          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* compute the size of the result and allocate a bag                   */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint16);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint32);
    degP = degL < degR ? degR : degL;
    hdP  = NewBag( T_PERM32, (UInt)(degP * sizeof(TypPoint32)) );

    /* set up the pointers                                                 */
    ptL = (TypPoint16*)PTR_BAG(hdL);
    ptR = (TypPoint32*)PTR_BAG(hdR);
    ptP = (TypPoint32*)PTR_BAG(hdP);

    /* if the left (inner) permutation has smaller degree, it is very easy */
    if ( degL <= degR ) {
        for ( p = 0; p < degL; p++ )
            *(ptP++) = ptR[ *(ptL++) ];
        for ( p = degL; p < degR; p++ )
            *(ptP++) = ptR[ p ];
    }

    /* otherwise we have to use the macro 'IMAGE'                          */
    else {
        for ( p = 0; p < degL; p++ )
            *(ptP++) = IMAGE( ptL[ p ], ptR, degR );
    }

    /* return the result                                                   */
    return hdP;
}


Bag       ProdQP (Bag hdL, Bag hdR)
{
    Bag           hdP;            /* handle of the product (result)  */
    UInt       degP;           /* degree of the product           */
    TypPoint32          * ptP;          /* pointer to the product          */
    UInt       degL;           /* degree of the left operand      */
    TypPoint32          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint16          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* compute the size of the result and allocate a bag                   */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint32);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint16);
    degP = degL < degR ? degR : degL;
    hdP  = NewBag( T_PERM32, (UInt)(degP * sizeof(TypPoint32)) );

    /* set up the pointers                                                 */
    ptL = (TypPoint32*)PTR_BAG(hdL);
    ptR = (TypPoint16*)PTR_BAG(hdR);
    ptP = (TypPoint32*)PTR_BAG(hdP);

    /* if the left (inner) permutation has smaller degree, it is very easy */
    if ( degL <= degR ) {
        for ( p = 0; p < degL; p++ )
            *(ptP++) = ptR[ *(ptL++) ];
        for ( p = degL; p < degR; p++ )
            *(ptP++) = ptR[ p ];
    }

    /* otherwise we have to use the macro 'IMAGE'                          */
    else {
        for ( p = 0; p < degL; p++ )
            *(ptP++) = IMAGE( ptL[ p ], ptR, degR );
    }

    /* return the result                                                   */
    return hdP;
}


Bag       ProdQQ (Bag hdL, Bag hdR)
{
    Bag           hdP;            /* handle of the product (result)  */
    UInt       degP;           /* degree of the product           */
    TypPoint32          * ptP;          /* pointer to the product          */
    UInt       degL;           /* degree of the left operand      */
    TypPoint32          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint32          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* compute the size of the result and allocate a bag                   */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint32);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint32);
    degP = degL < degR ? degR : degL;
    hdP  = NewBag( T_PERM32, (UInt)(degP * sizeof(TypPoint32)) );

    /* set up the pointers                                                 */
    ptL = (TypPoint32*)PTR_BAG(hdL);
    ptR = (TypPoint32*)PTR_BAG(hdR);
    ptP = (TypPoint32*)PTR_BAG(hdP);

    /* if the left (inner) permutation has smaller degree, it is very easy */
    if ( degL <= degR ) {
        for ( p = 0; p < degL; p++ )
            *(ptP++) = ptR[ *(ptL++) ];
        for ( p = degL; p < degR; p++ )
            *(ptP++) = ptR[ p ];
    }

    /* otherwise we have to use the macro 'IMAGE'                          */
    else {
        for ( p = 0; p < degL; p++ )
            *(ptP++) = IMAGE( ptL[ p ], ptR, degR );
    }

    /* return the result                                                   */
    return hdP;
}


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
Bag       QuoPP (Bag hdL, Bag hdR)
{
    Bag           hdQ;            /* handle of the quotient (result) */
    UInt       degQ;           /* degree of the quotient          */
    TypPoint16          * ptQ;          /* pointer to the quotient         */
    UInt       degL;           /* degree of the left operand      */
    TypPoint16          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint16          * ptR;          /* pointer to the right operand    */
    TypPoint16          * ptI;          /* pointer to the inverse          */
    UInt       p;              /* loop variable                   */

    /* compute the size of the result and allocate a bag                   */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint16);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint16);
    degQ = degL < degR ? degR : degL;
    hdQ  = NewBag( T_PERM16, (UInt)(degQ * sizeof(TypPoint16)) );

    /* make sure that the buffer bag is large enough to hold the inverse   */
    if ( GET_SIZE_BAG(HdPerm) < GET_SIZE_BAG(hdR) )  Resize( HdPerm, GET_SIZE_BAG(hdR) );

    /* invert the right permutation into the buffer bag                    */
    ptI = (TypPoint16*)PTR_BAG(HdPerm);
    ptR = (TypPoint16*)PTR_BAG(hdR);
    for ( p = 0; p < degR; p++ )
        ptI[ *ptR++ ] = p;

    /* multiply the left permutation with the inverse                      */
    ptL = (TypPoint16*)PTR_BAG(hdL);
    ptI = (TypPoint16*)PTR_BAG(HdPerm);
    ptQ = (TypPoint16*)PTR_BAG(hdQ);
    if ( degL <= degR ) {
        for ( p = 0; p < degL; p++ )
            *(ptQ++) = ptI[ *(ptL++) ];
        for ( p = degL; p < degR; p++ )
            *(ptQ++) = ptI[ p ];
    }
    else {
        for ( p = 0; p < degL; p++ )
            *(ptQ++) = IMAGE( ptL[ p ], ptI, degR );
    }

    /* make the buffer bag clean again                                     */
    ptI = (TypPoint16*)PTR_BAG(HdPerm);
    for ( p = 0; p < degR; p++ )
        ptI[ p ] = 0;

    /* return the result                                                   */
    return hdQ;
}

Bag       QuoPQ (Bag hdL, Bag hdR)
{
    Bag           hdQ;            /* handle of the quotient (result) */
    UInt       degQ;           /* degree of the quotient          */
    TypPoint32          * ptQ;          /* pointer to the quotient         */
    UInt       degL;           /* degree of the left operand      */
    TypPoint16          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint32          * ptR;          /* pointer to the right operand    */
    TypPoint32          * ptI;          /* pointer to the inverse          */
    UInt       p;              /* loop variable                   */

    /* compute the size of the result and allocate a bag                   */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint16);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint32);
    degQ = degL < degR ? degR : degL;
    hdQ  = NewBag( T_PERM32, (UInt)(degQ * sizeof(TypPoint32)) );

    /* make sure that the buffer bag is large enough to hold the inverse   */
    if ( GET_SIZE_BAG(HdPerm) < GET_SIZE_BAG(hdR) )  Resize( HdPerm, GET_SIZE_BAG(hdR) );

    /* invert the right permutation into the buffer bag                    */
    ptI = (TypPoint32*)PTR_BAG(HdPerm);
    ptR = (TypPoint32*)PTR_BAG(hdR);
    for ( p = 0; p < degR; p++ )
        ptI[ *ptR++ ] = p;

    /* multiply the left permutation with the inverse                      */
    ptL = (TypPoint16*)PTR_BAG(hdL);
    ptI = (TypPoint32*)PTR_BAG(HdPerm);
    ptQ = (TypPoint32*)PTR_BAG(hdQ);
    if ( degL <= degR ) {
        for ( p = 0; p < degL; p++ )
            *(ptQ++) = ptI[ *(ptL++) ];
        for ( p = degL; p < degR; p++ )
            *(ptQ++) = ptI[ p ];
    }
    else {
        for ( p = 0; p < degL; p++ )
            *(ptQ++) = IMAGE( ptL[ p ], ptI, degR );
    }

    /* make the buffer bag clean again                                     */
    ptI = (TypPoint32*)PTR_BAG(HdPerm);
    for ( p = 0; p < degR; p++ )
        ptI[ p ] = 0;

    /* return the result                                                   */
    return hdQ;
}

Bag       QuoQP (Bag hdL, Bag hdR)
{
    Bag           hdQ;            /* handle of the quotient (result) */
    UInt       degQ;           /* degree of the quotient          */
    TypPoint32          * ptQ;          /* pointer to the quotient         */
    UInt       degL;           /* degree of the left operand      */
    TypPoint32          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint16          * ptR;          /* pointer to the right operand    */
    TypPoint16          * ptI;          /* pointer to the inverse          */
    UInt       p;              /* loop variable                   */

    /* compute the size of the result and allocate a bag                   */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint32);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint16);
    degQ = degL < degR ? degR : degL;
    hdQ  = NewBag( T_PERM32, (UInt)(degQ * sizeof(TypPoint32)) );

    /* make sure that the buffer bag is large enough to hold the inverse   */
    if ( GET_SIZE_BAG(HdPerm) < GET_SIZE_BAG(hdR) )  Resize( HdPerm, GET_SIZE_BAG(hdR) );

    /* invert the right permutation into the buffer bag                    */
    ptI = (TypPoint16*)PTR_BAG(HdPerm);
    ptR = (TypPoint16*)PTR_BAG(hdR);
    for ( p = 0; p < degR; p++ )
        ptI[ *ptR++ ] = p;

    /* multiply the left permutation with the inverse                      */
    ptL = (TypPoint32*)PTR_BAG(hdL);
    ptI = (TypPoint16*)PTR_BAG(HdPerm);
    ptQ = (TypPoint32*)PTR_BAG(hdQ);
    if ( degL <= degR ) {
        for ( p = 0; p < degL; p++ )
            *(ptQ++) = ptI[ *(ptL++) ];
        for ( p = degL; p < degR; p++ )
            *(ptQ++) = ptI[ p ];
    }
    else {
        for ( p = 0; p < degL; p++ )
            *(ptQ++) = IMAGE( ptL[ p ], ptI, degR );
    }

    /* make the buffer bag clean again                                     */
    ptI = (TypPoint16*)PTR_BAG(HdPerm);
    for ( p = 0; p < degR; p++ )
        ptI[ p ] = 0;

    /* return the result                                                   */
    return hdQ;
}

Bag       QuoQQ (Bag hdL, Bag hdR)
{
    Bag           hdQ;            /* handle of the quotient (result) */
    UInt       degQ;           /* degree of the quotient          */
    TypPoint32          * ptQ;          /* pointer to the quotient         */
    UInt       degL;           /* degree of the left operand      */
    TypPoint32          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint32          * ptR;          /* pointer to the right operand    */
    TypPoint32          * ptI;          /* pointer to the inverse          */
    UInt       p;              /* loop variable                   */

    /* compute the size of the result and allocate a bag                   */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint32);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint32);
    degQ = degL < degR ? degR : degL;
    hdQ  = NewBag( T_PERM32, (UInt)(degQ * sizeof(TypPoint32)) );

    /* make sure that the buffer bag is large enough to hold the inverse   */
    if ( GET_SIZE_BAG(HdPerm) < GET_SIZE_BAG(hdR) )  Resize( HdPerm, GET_SIZE_BAG(hdR) );

    /* invert the right permutation into the buffer bag                    */
    ptI = (TypPoint32*)PTR_BAG(HdPerm);
    ptR = (TypPoint32*)PTR_BAG(hdR);
    for ( p = 0; p < degR; p++ )
        ptI[ *ptR++ ] = p;

    /* multiply the left permutation with the inverse                      */
    ptL = (TypPoint32*)PTR_BAG(hdL);
    ptI = (TypPoint32*)PTR_BAG(HdPerm);
    ptQ = (TypPoint32*)PTR_BAG(hdQ);
    if ( degL <= degR ) {
        for ( p = 0; p < degL; p++ )
            *(ptQ++) = ptI[ *(ptL++) ];
        for ( p = degL; p < degR; p++ )
            *(ptQ++) = ptI[ p ];
    }
    else {
        for ( p = 0; p < degL; p++ )
            *(ptQ++) = IMAGE( ptL[ p ], ptI, degR );
    }

    /* make the buffer bag clean again                                     */
    ptI = (TypPoint32*)PTR_BAG(HdPerm);
    for ( p = 0; p < degR; p++ )
        ptI[ p ] = 0;

    /* return the result                                                   */
    return hdQ;
}


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
Bag       ModPP (Bag hdL, Bag hdR)
{
    Bag           hdM;            /* handle of the quotient (result) */
    UInt       degM;           /* degree of the quotient          */
    TypPoint16          * ptM;          /* pointer to the quotient         */
    UInt       degL;           /* degree of the left operand      */
    TypPoint16          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint16          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* compute the size of the result and allocate a bag                   */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint16);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint16);
    degM = degL < degR ? degR : degL;
    hdM = NewBag( T_PERM16, (UInt)(degM * sizeof(TypPoint16)) );

    /* set up the pointers                                                 */
    ptL = (TypPoint16*)PTR_BAG(hdL);
    ptR = (TypPoint16*)PTR_BAG(hdR);
    ptM = (TypPoint16*)PTR_BAG(hdM);

    /* its one thing if the left (inner) permutation is smaller            */
    if ( degL <= degR ) {
        for ( p = 0; p < degL; p++ )
            ptM[ *(ptL++) ] = *(ptR++);
        for ( p = degL; p < degR; p++ )
            ptM[ p ] = *(ptR++);
    }

    /* and another if the right (outer) permutation is smaller             */
    else {
        for ( p = 0; p < degR; p++ )
            ptM[ *(ptL++) ] = *(ptR++);
        for ( p = degR; p < degL; p++ )
            ptM[ *(ptL++) ] = p;
    }

    /* return the result                                                   */
    return hdM;
}

Bag       ModPQ (Bag hdL, Bag hdR)
{
    Bag           hdM;            /* handle of the quotient (result) */
    UInt       degM;           /* degree of the quotient          */
    TypPoint32          * ptM;          /* pointer to the quotient         */
    UInt       degL;           /* degree of the left operand      */
    TypPoint16          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint32          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* compute the size of the result and allocate a bag                   */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint16);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint32);
    degM = degL < degR ? degR : degL;
    hdM = NewBag( T_PERM32, (UInt)(degM * sizeof(TypPoint32)) );

    /* set up the pointers                                                 */
    ptL = (TypPoint16*)PTR_BAG(hdL);
    ptR = (TypPoint32*)PTR_BAG(hdR);
    ptM = (TypPoint32*)PTR_BAG(hdM);

    /* its one thing if the left (inner) permutation is smaller            */
    if ( degL <= degR ) {
        for ( p = 0; p < degL; p++ )
            ptM[ *(ptL++) ] = *(ptR++);
        for ( p = degL; p < degR; p++ )
            ptM[ p ] = *(ptR++);
    }

    /* and another if the right (outer) permutation is smaller             */
    else {
        for ( p = 0; p < degR; p++ )
            ptM[ *(ptL++) ] = *(ptR++);
        for ( p = degR; p < degL; p++ )
            ptM[ *(ptL++) ] = p;
    }

    /* return the result                                                   */
    return hdM;
}

Bag       ModQP (Bag hdL, Bag hdR)
{
    Bag           hdM;            /* handle of the quotient (result) */
    UInt       degM;           /* degree of the quotient          */
    TypPoint32          * ptM;          /* pointer to the quotient         */
    UInt       degL;           /* degree of the left operand      */
    TypPoint32          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint16          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* compute the size of the result and allocate a bag                   */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint32);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint16);
    degM = degL < degR ? degR : degL;
    hdM = NewBag( T_PERM32, (UInt)(degM * sizeof(TypPoint32)) );

    /* set up the pointers                                                 */
    ptL = (TypPoint32*)PTR_BAG(hdL);
    ptR = (TypPoint16*)PTR_BAG(hdR);
    ptM = (TypPoint32*)PTR_BAG(hdM);

    /* its one thing if the left (inner) permutation is smaller            */
    if ( degL <= degR ) {
        for ( p = 0; p < degL; p++ )
            ptM[ *(ptL++) ] = *(ptR++);
        for ( p = degL; p < degR; p++ )
            ptM[ p ] = *(ptR++);
    }

    /* and another if the right (outer) permutation is smaller             */
    else {
        for ( p = 0; p < degR; p++ )
            ptM[ *(ptL++) ] = *(ptR++);
        for ( p = degR; p < degL; p++ )
            ptM[ *(ptL++) ] = p;
    }

    /* return the result                                                   */
    return hdM;
}

Bag       ModQQ (Bag hdL, Bag hdR)
{
    Bag           hdM;            /* handle of the quotient (result) */
    UInt       degM;           /* degree of the quotient          */
    TypPoint32          * ptM;          /* pointer to the quotient         */
    UInt       degL;           /* degree of the left operand      */
    TypPoint32          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint32          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* compute the size of the result and allocate a bag                   */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint32);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint32);
    degM = degL < degR ? degR : degL;
    hdM = NewBag( T_PERM32, (UInt)(degM * sizeof(TypPoint32)) );

    /* set up the pointers                                                 */
    ptL = (TypPoint32*)PTR_BAG(hdL);
    ptR = (TypPoint32*)PTR_BAG(hdR);
    ptM = (TypPoint32*)PTR_BAG(hdM);

    /* its one thing if the left (inner) permutation is smaller            */
    if ( degL <= degR ) {
        for ( p = 0; p < degL; p++ )
            ptM[ *(ptL++) ] = *(ptR++);
        for ( p = degL; p < degR; p++ )
            ptM[ p ] = *(ptR++);
    }

    /* and another if the right (outer) permutation is smaller             */
    else {
        for ( p = 0; p < degR; p++ )
            ptM[ *(ptL++) ] = *(ptR++);
        for ( p = degR; p < degL; p++ )
            ptM[ *(ptL++) ] = p;
    }

    /* return the result                                                   */
    return hdM;
}


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
Bag       PowPI (Bag hdL, Bag hdR)
{
    Bag           hdP;            /* handle of the power (result)    */
    TypPoint16          * ptP;          /* pointer to the power            */
    TypPoint16          * ptL;          /* pointer to the permutation      */
    TypPoint16          * ptKnown;      /* pointer to temporary bag        */
    UInt       deg;            /* degree of the permutation       */
    Int                exp,  e;        /* exponent (right operand)        */
    UInt       len;            /* length of cycle (result)        */
    UInt       p,  q,  r;      /* loop variables                  */

    /* get the operands and allocate a result bag                          */
    deg = GET_SIZE_BAG(hdL) / sizeof(TypPoint16);
    hdP = NewBag( T_PERM16, (UInt)(deg * sizeof(TypPoint16)) );

    /* compute the power by repeated mapping for small positive exponents  */
    if ( GET_TYPE_BAG(hdR)==T_INT && 0<=HD_TO_INT(hdR) && HD_TO_INT(hdR)<8 ) {

        /* get pointer to the permutation and the power                    */
        exp = HD_TO_INT(hdR);
        ptL = (TypPoint16*)PTR_BAG(hdL);
        ptP = (TypPoint16*)PTR_BAG(hdP);

        /* loop over the points of the permutation                         */
        for ( p = 0; p < deg; p++ ) {
            q = p;
            for ( e = 0; e < exp; e++ )
                q = ptL[q];
            ptP[p] = q;
        }

    }

    /* compute the power by raising the cycles individually for large exps */
    else if ( GET_TYPE_BAG(hdR)==T_INT && 8 <= HD_TO_INT(hdR) ) {

        /* make sure that the buffer bag is large enough                   */
        if ( GET_SIZE_BAG(HdPerm) < GET_SIZE_BAG(hdL) )  Resize( HdPerm, GET_SIZE_BAG(hdL) );
        ptKnown = (TypPoint16*)PTR_BAG(HdPerm);

        /* get pointer to the permutation and the power                    */
        exp = HD_TO_INT(hdR);
        ptL = (TypPoint16*)PTR_BAG(hdL);
        ptP = (TypPoint16*)PTR_BAG(hdP);

        /* loop over all cycles                                            */
        for ( p = 0; p < deg; p++ ) {

            /* if we haven't looked at this cycle so far                   */
            if ( ptKnown[p] == 0 ) {

                /* find the length of this cycle                           */
                len = 1;
                for ( q = ptL[p]; q != p; q = ptL[q] ) {
                    len++;  ptKnown[q] = 1;
                }

                /* raise this cycle to the power <exp> mod <len>           */
                r = p;
                for ( e = 0; e < exp % len; e++ )
                    r = ptL[r];
                ptP[p] = r;
                r = ptL[r];
                for ( q = ptL[p]; q != p; q = ptL[q] ) {
                    ptP[q] = r;
                    r = ptL[r];
                }

            }

        }

        /* clear the buffer bag again                                      */
        for ( p = 0; p < GET_SIZE_BAG(hdL)/sizeof(TypPoint16); p++ )
            ptKnown[p] = 0;

    }

    /* compute the power by raising the cycles individually for large exps */
    else if ( GET_TYPE_BAG(hdR)==T_INTPOS ) {

        /* make sure that the buffer bag is large enough                   */
        if ( GET_SIZE_BAG(HdPerm) < GET_SIZE_BAG(hdL) )  Resize( HdPerm, GET_SIZE_BAG(hdL) );
        ptKnown = (TypPoint16*)PTR_BAG(HdPerm);

        /* get pointer to the permutation and the power                    */
        ptL = (TypPoint16*)PTR_BAG(hdL);
        ptP = (TypPoint16*)PTR_BAG(hdP);

        /* loop over all cycles                                            */
        for ( p = 0; p < deg; p++ ) {

            /* if we haven't looked at this cycle so far                   */
            if ( ptKnown[p] == 0 ) {

                /* find the length of this cycle                           */
                len = 1;
                for ( q = ptL[p]; q != p; q = ptL[q] ) {
                    len++;  ptKnown[q] = 1;
                }

                /* raise this cycle to the power <exp> mod <len>           */
                r = p;
                exp = HD_TO_INT( ModInt( hdR, INT_TO_HD(len) ) );
                for ( e = 0; e < exp; e++ )
                    r = ptL[r];
                ptP[p] = r;
                r = ptL[r];
                for ( q = ptL[p]; q != p; q = ptL[q] ) {
                    ptP[q] = r;
                    r = ptL[r];
                }

            }

        }

        /* clear the buffer bag again                                      */
        for ( p = 0; p < GET_SIZE_BAG(hdL)/sizeof(TypPoint16); p++ )
            ptKnown[p] = 0;

    }

    /* special case for inverting permutations                             */
    else if ( GET_TYPE_BAG(hdR)==T_INT && HD_TO_INT(hdR) == -1 ) {

        /* get pointer to the permutation and the power                    */
        ptL = (TypPoint16*)PTR_BAG(hdL);
        ptP = (TypPoint16*)PTR_BAG(hdP);

        /* invert the permutation                                          */
        for ( p = 0; p < deg; p++ )
            ptP[ *(ptL++) ] = p;

    }

    /* compute the power by repeated mapping for small negative exponents  */
    else if ( GET_TYPE_BAG(hdR)==T_INT && -8<HD_TO_INT(hdR) && HD_TO_INT(hdR)<0 ) {

        /* get pointer to the permutation and the power                    */
        exp = -HD_TO_INT(hdR);
        ptL = (TypPoint16*)PTR_BAG(hdL);
        ptP = (TypPoint16*)PTR_BAG(hdP);

        /* loop over the points                                            */
        for ( p = 0; p < deg; p++ ) {
            q = p;
            for ( e = 0; e < exp; e++ )
                q = ptL[q];
            ptP[q] = p;
        }

    }

    /* compute the power by raising the cycles individually for large exps */
    else if ( GET_TYPE_BAG(hdR)==T_INT && HD_TO_INT(hdR) <= -8 ) {

        /* make sure that the buffer bag is large enough                   */
        if ( GET_SIZE_BAG(HdPerm) < GET_SIZE_BAG(hdL) )  Resize( HdPerm, GET_SIZE_BAG(hdL) );
        ptKnown = (TypPoint16*)PTR_BAG(HdPerm);

        /* get pointer to the permutation and the power                    */
        exp = -HD_TO_INT(hdR);
        ptL = (TypPoint16*)PTR_BAG(hdL);
        ptP = (TypPoint16*)PTR_BAG(hdP);

        /* loop over all cycles                                            */
        for ( p = 0; p < deg; p++ ) {

            /* if we haven't looked at this cycle so far                   */
            if ( ptKnown[p] == 0 ) {

                /* find the length of this cycle                           */
                len = 1;
                for ( q = ptL[p]; q != p; q = ptL[q] ) {
                    len++;  ptKnown[q] = 1;
                }

                /* raise this cycle to the power <exp> mod <len>           */
                r = p;
                for ( e = 0; e < exp % len; e++ )
                    r = ptL[r];
                ptP[r] = p;
                r = ptL[r];
                for ( q = ptL[p]; q != p; q = ptL[q] ) {
                    ptP[r] = q;
                    r = ptL[r];
                }

            }

        }

        /* clear the buffer bag again                                      */
        for ( p = 0; p < GET_SIZE_BAG(hdL)/sizeof(TypPoint16); p++ )
            ptKnown[p] = 0;

    }

    /* compute the power by raising the cycles individually for large exps */
    else if ( GET_TYPE_BAG(hdR)==T_INTNEG ) {

        /* make sure that the buffer bag is large enough                   */
        if ( GET_SIZE_BAG(HdPerm) < GET_SIZE_BAG(hdL) )  Resize( HdPerm, GET_SIZE_BAG(hdL) );
        ptKnown = (TypPoint16*)PTR_BAG(HdPerm);

        /* get pointer to the permutation and the power                    */
        hdR = ProdInt( INT_TO_HD(-1), hdR );
        ptL = (TypPoint16*)PTR_BAG(hdL);
        ptP = (TypPoint16*)PTR_BAG(hdP);

        /* loop over all cycles                                            */
        for ( p = 0; p < deg; p++ ) {

            /* if we haven't looked at this cycle so far                   */
            if ( ptKnown[p] == 0 ) {

                /* find the length of this cycle                           */
                len = 1;
                for ( q = ptL[p]; q != p; q = ptL[q] ) {
                    len++;  ptKnown[q] = 1;
                }

                /* raise this cycle to the power <exp> mod <len>           */
                r = p;
                exp = HD_TO_INT( ModInt( hdR, INT_TO_HD(len) ) );
                for ( e = 0; e < exp % len; e++ )
                    r = ptL[r];
                ptP[r] = p;
                r = ptL[r];
                for ( q = ptL[p]; q != p; q = ptL[q] ) {
                    ptP[r] = q;
                    r = ptL[r];
                }

            }

        }

        /* clear the buffer bag again                                      */
        for ( p = 0; p < GET_SIZE_BAG(hdL)/sizeof(TypPoint16); p++ )
            ptKnown[p] = 0;

    }

    /* return the result                                                   */
    return hdP;
}

Bag       PowQI (Bag hdL, Bag hdR)
{
    Bag           hdP;            /* handle of the power (result)    */
    TypPoint32          * ptP;          /* pointer to the power            */
    TypPoint32          * ptL;          /* pointer to the permutation      */
    TypPoint32          * ptKnown;      /* pointer to temporary bag        */
    UInt       deg;            /* degree of the permutation       */
    Int                exp,  e;        /* exponent (right operand)        */
    UInt       len;            /* length of cycle (result)        */
    UInt       p,  q,  r;      /* loop variables                  */

    /* get the operands and allocate a result bag                          */
    deg = GET_SIZE_BAG(hdL) / sizeof(TypPoint32);
    hdP = NewBag( T_PERM32, (UInt)(deg * sizeof(TypPoint32)) );

    /* compute the power by repeated mapping for small positive exponents  */
    if ( GET_TYPE_BAG(hdR)==T_INT && 0<=HD_TO_INT(hdR) && HD_TO_INT(hdR)<8 ) {

        /* get pointer to the permutation and the power                    */
        exp = HD_TO_INT(hdR);
        ptL = (TypPoint32*)PTR_BAG(hdL);
        ptP = (TypPoint32*)PTR_BAG(hdP);

        /* loop over the points of the permutation                         */
        for ( p = 0; p < deg; p++ ) {
            q = p;
            for ( e = 0; e < exp; e++ )
                q = ptL[q];
            ptP[p] = q;
        }

    }

    /* compute the power by raising the cycles individually for large exps */
    else if ( GET_TYPE_BAG(hdR)==T_INT && 8 <= HD_TO_INT(hdR) ) {

        /* make sure that the buffer bag is large enough                   */
        if ( GET_SIZE_BAG(HdPerm) < GET_SIZE_BAG(hdL) )  Resize( HdPerm, GET_SIZE_BAG(hdL) );
        ptKnown = (TypPoint32*)PTR_BAG(HdPerm);

        /* get pointer to the permutation and the power                    */
        exp = HD_TO_INT(hdR);
        ptL = (TypPoint32*)PTR_BAG(hdL);
        ptP = (TypPoint32*)PTR_BAG(hdP);

        /* loop over all cycles                                            */
        for ( p = 0; p < deg; p++ ) {

            /* if we haven't looked at this cycle so far                   */
            if ( ptKnown[p] == 0 ) {

                /* find the length of this cycle                           */
                len = 1;
                for ( q = ptL[p]; q != p; q = ptL[q] ) {
                    len++;  ptKnown[q] = 1;
                }

                /* raise this cycle to the power <exp> mod <len>           */
                r = p;
                for ( e = 0; e < exp % len; e++ )
                    r = ptL[r];
                ptP[p] = r;
                r = ptL[r];
                for ( q = ptL[p]; q != p; q = ptL[q] ) {
                    ptP[q] = r;
                    r = ptL[r];
                }

            }

        }

        /* clear the buffer bag again                                      */
        for ( p = 0; p < GET_SIZE_BAG(hdL)/sizeof(TypPoint32); p++ )
            ptKnown[p] = 0;

    }

    /* compute the power by raising the cycles individually for large exps */
    else if ( GET_TYPE_BAG(hdR)==T_INTPOS ) {

        /* make sure that the buffer bag is large enough                   */
        if ( GET_SIZE_BAG(HdPerm) < GET_SIZE_BAG(hdL) )  Resize( HdPerm, GET_SIZE_BAG(hdL) );
        ptKnown = (TypPoint32*)PTR_BAG(HdPerm);

        /* get pointer to the permutation and the power                    */
        ptL = (TypPoint32*)PTR_BAG(hdL);
        ptP = (TypPoint32*)PTR_BAG(hdP);

        /* loop over all cycles                                            */
        for ( p = 0; p < deg; p++ ) {

            /* if we haven't looked at this cycle so far                   */
            if ( ptKnown[p] == 0 ) {

                /* find the length of this cycle                           */
                len = 1;
                for ( q = ptL[p]; q != p; q = ptL[q] ) {
                    len++;  ptKnown[q] = 1;
                }

                /* raise this cycle to the power <exp> mod <len>           */
                r = p;
                exp = HD_TO_INT( ModInt( hdR, INT_TO_HD(len) ) );
                for ( e = 0; e < exp; e++ )
                    r = ptL[r];
                ptP[p] = r;
                r = ptL[r];
                for ( q = ptL[p]; q != p; q = ptL[q] ) {
                    ptP[q] = r;
                    r = ptL[r];
                }

            }

        }

        /* clear the buffer bag again                                      */
        for ( p = 0; p < GET_SIZE_BAG(hdL)/sizeof(TypPoint32); p++ )
            ptKnown[p] = 0;

    }

    /* special case for inverting permutations                             */
    else if ( GET_TYPE_BAG(hdR)==T_INT && HD_TO_INT(hdR) == -1 ) {

        /* get pointer to the permutation and the power                    */
        ptL = (TypPoint32*)PTR_BAG(hdL);
        ptP = (TypPoint32*)PTR_BAG(hdP);

        /* invert the permutation                                          */
        for ( p = 0; p < deg; p++ )
            ptP[ *(ptL++) ] = p;

    }

    /* compute the power by repeated mapping for small negative exponents  */
    else if ( GET_TYPE_BAG(hdR)==T_INT && -8<HD_TO_INT(hdR) && HD_TO_INT(hdR)<0 ) {

        /* get pointer to the permutation and the power                    */
        exp = -HD_TO_INT(hdR);
        ptL = (TypPoint32*)PTR_BAG(hdL);
        ptP = (TypPoint32*)PTR_BAG(hdP);

        /* loop over the points                                            */
        for ( p = 0; p < deg; p++ ) {
            q = p;
            for ( e = 0; e < exp; e++ )
                q = ptL[q];
            ptP[q] = p;
        }

    }

    /* compute the power by raising the cycles individually for large exps */
    else if ( GET_TYPE_BAG(hdR)==T_INT && HD_TO_INT(hdR) <= -8 ) {

        /* make sure that the buffer bag is large enough                   */
        if ( GET_SIZE_BAG(HdPerm) < GET_SIZE_BAG(hdL) )  Resize( HdPerm, GET_SIZE_BAG(hdL) );
        ptKnown = (TypPoint32*)PTR_BAG(HdPerm);

        /* get pointer to the permutation and the power                    */
        exp = -HD_TO_INT(hdR);
        ptL = (TypPoint32*)PTR_BAG(hdL);
        ptP = (TypPoint32*)PTR_BAG(hdP);

        /* loop over all cycles                                            */
        for ( p = 0; p < deg; p++ ) {

            /* if we haven't looked at this cycle so far                   */
            if ( ptKnown[p] == 0 ) {

                /* find the length of this cycle                           */
                len = 1;
                for ( q = ptL[p]; q != p; q = ptL[q] ) {
                    len++;  ptKnown[q] = 1;
                }

                /* raise this cycle to the power <exp> mod <len>           */
                r = p;
                for ( e = 0; e < exp % len; e++ )
                    r = ptL[r];
                ptP[r] = p;
                r = ptL[r];
                for ( q = ptL[p]; q != p; q = ptL[q] ) {
                    ptP[r] = q;
                    r = ptL[r];
                }

            }

        }

        /* clear the buffer bag again                                      */
        for ( p = 0; p < GET_SIZE_BAG(hdL)/sizeof(TypPoint32); p++ )
            ptKnown[p] = 0;

    }

    /* compute the power by raising the cycles individually for large exps */
    else if ( GET_TYPE_BAG(hdR)==T_INTNEG ) {

        /* make sure that the buffer bag is large enough                   */
        if ( GET_SIZE_BAG(HdPerm) < GET_SIZE_BAG(hdL) )  Resize( HdPerm, GET_SIZE_BAG(hdL) );
        ptKnown = (TypPoint32*)PTR_BAG(HdPerm);

        /* get pointer to the permutation and the power                    */
        hdR = ProdInt( INT_TO_HD(-1), hdR );
        ptL = (TypPoint32*)PTR_BAG(hdL);
        ptP = (TypPoint32*)PTR_BAG(hdP);

        /* loop over all cycles                                            */
        for ( p = 0; p < deg; p++ ) {

            /* if we haven't looked at this cycle so far                   */
            if ( ptKnown[p] == 0 ) {

                /* find the length of this cycle                           */
                len = 1;
                for ( q = ptL[p]; q != p; q = ptL[q] ) {
                    len++;  ptKnown[q] = 1;
                }

                /* raise this cycle to the power <exp> mod <len>           */
                r = p;
                exp = HD_TO_INT( ModInt( hdR, INT_TO_HD(len) ) );
                for ( e = 0; e < exp % len; e++ )
                    r = ptL[r];
                ptP[r] = p;
                r = ptL[r];
                for ( q = ptL[p]; q != p; q = ptL[q] ) {
                    ptP[r] = q;
                    r = ptL[r];
                }

            }

        }

        /* clear the buffer bag again                                      */
        for ( p = 0; p < GET_SIZE_BAG(hdL)/sizeof(TypPoint32); p++ )
            ptKnown[p] = 0;

    }

    /* return the result                                                   */
    return hdP;
}


/****************************************************************************
**
*F  PowIP( <hdL>, <hdR> ) . . . . . . image of an integer under a permutation
**
**  'PowIP' returns  the  image  of  the positive   integer <hdL>  under  the
**  permutation  <hdR>.  If <hdL> is  larger than the degree   of <hdR> it is
**  a fixpoint of the permutation and thus simply returned.
**
**  Is called from the 'Pow' binop, so both operands are already evaluated.
*/
Bag       PowIP (Bag hdL, Bag hdR)
{
    Int                img;            /* image (result)                  */

    /* large positive integers (> 2^28-1) are fixed by any permutation     */
    if ( GET_TYPE_BAG(hdL) == T_INTPOS )
        return hdL;

    /* permutations do not act on negative integers                        */
    img = HD_TO_INT( hdL );
    if ( img <= 0 )
        return Error("Perm Op: point must be positive (%d)",img,0);

    /* compute the image                                                   */
    if ( img <= GET_SIZE_BAG(hdR)/sizeof(TypPoint16) ) {
        img = ((TypPoint16*)PTR_BAG(hdR))[img-1] + 1;
    }

    /* return it                                                           */
    return INT_TO_HD(img);
}

Bag       PowIQ (Bag hdL, Bag hdR)
{
    Int                img;            /* image (result)                  */

    /* large positive integers (> 2^28-1) are fixed by any permutation     */
    if ( GET_TYPE_BAG(hdL) == T_INTPOS )
        return hdL;

    /* permutations do not act on negative integers                        */
    img = HD_TO_INT( hdL );
    if ( img <= 0 )
        return Error("Perm Op: point must be positive (%d)",img,0);

    /* compute the image                                                   */
    if ( img <= GET_SIZE_BAG(hdR)/sizeof(TypPoint32) ) {
        img = ((TypPoint32*)PTR_BAG(hdR))[img-1] + 1;
    }

    /* return it                                                           */
    return INT_TO_HD(img);
}


/****************************************************************************
**
*F  QuoIP( <hdL>, <hdR> ) . . . .  preimage of an integer under a permutation
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
Bag       QuoIP (Bag hdL, Bag hdR)
{
    Int                pre;            /* preimage (result)               */
    Int                img;            /* image (left operand)            */
    TypPoint16          * ptR;          /* pointer to the permutation      */

    /* large positive integers (> 2^28-1) are fixed by any permutation     */
    if ( GET_TYPE_BAG(hdL) == T_INTPOS )
        return hdL;

    /* permutations do not act on negative integers                        */
    img = HD_TO_INT(hdL);
    if ( img <= 0 )
        return Error("PermOps: %d must be positive",HD_TO_INT(hdL),0);

    /* compute the preimage                                                */
    pre = img;
    ptR = (TypPoint16*)PTR_BAG(hdR);
    if ( img <= GET_SIZE_BAG(hdR)/sizeof(TypPoint16) ) {
        while ( ptR[ pre-1 ] != img-1 )
            pre = ptR[ pre-1 ] + 1;
    }

    /* return it                                                           */
    return INT_TO_HD(pre);
}

Bag       QuoIQ (Bag hdL, Bag hdR)
{
    Int                pre;            /* preimage (result)               */
    Int                img;            /* image (left operand)            */
    TypPoint32          * ptR;          /* pointer to the permutation      */

    /* large positive integers (> 2^28-1) are fixed by any permutation     */
    if ( GET_TYPE_BAG(hdL) == T_INTPOS )
        return hdL;

    /* permutations do not act on negative integers                        */
    img = HD_TO_INT(hdL);
    if ( img <= 0 )
        return Error("PermOps: %d must be positive",HD_TO_INT(hdL),0);

    /* compute the preimage                                                */
    pre = img;
    ptR = (TypPoint32*)PTR_BAG(hdR);
    if ( img <= GET_SIZE_BAG(hdR)/sizeof(TypPoint32) ) {
        while ( ptR[ pre-1 ] != img-1 )
            pre = ptR[ pre-1 ] + 1;
    }

    /* return it                                                           */
    return INT_TO_HD(pre);
}


/****************************************************************************
**
*F  PowPP( <hdL>, <hdR> ) . . . . . . . . . . . . conjugation of permutations
**
**  'PowPP' returns the conjugation of the  two permutations <hdL> and <hdR>,
**  that s defined as the following product '<hdR>\^-1 \*\ <hdL> \*\ <hdR>'.
**
**  Is called from the 'Pow' binop, so both operands are already evaluated.
*/
Bag       PowPP (Bag hdL, Bag hdR)
{
    Bag           hdC;            /* handle of the conjugation (res) */
    UInt       degC;           /* degree of the conjugation       */
    TypPoint16          * ptC;          /* pointer to the conjugation      */
    UInt       degL;           /* degree of the left operand      */
    TypPoint16          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint16          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* compute the size of the result and allocate a bag                   */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint16);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint16);
    degC = degL < degR ? degR : degL;
    hdC = NewBag( T_PERM16, (UInt)(degC * sizeof(TypPoint16)) );

    /* set up the pointers                                                 */
    ptL = (TypPoint16*)PTR_BAG(hdL);
    ptR = (TypPoint16*)PTR_BAG(hdR);
    ptC = (TypPoint16*)PTR_BAG(hdC);

    /* its faster if the both permutations have the same size              */
    if ( degL == degR ) {
        for ( p = 0; p < degC; p++ )
            ptC[ ptR[p] ] = ptR[ ptL[p] ];
    }

    /* otherwise we have to use the macro 'IMAGE' three times              */
    else {
        for ( p = 0; p < degC; p++ )
            ptC[ IMAGE(p,ptR,degR) ] = IMAGE( IMAGE(p,ptL,degL), ptR, degR );
    }

    /* return the result                                                   */
    return hdC;
}

Bag       PowPQ (Bag hdL, Bag hdR)
{
    Bag           hdC;            /* handle of the conjugation (res) */
    UInt       degC;           /* degree of the conjugation       */
    TypPoint32          * ptC;          /* pointer to the conjugation      */
    UInt       degL;           /* degree of the left operand      */
    TypPoint16          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint32          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* compute the size of the result and allocate a bag                   */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint16);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint32);
    degC = degL < degR ? degR : degL;
    hdC = NewBag( T_PERM32, (UInt)(degC * sizeof(TypPoint32)) );

    /* set up the pointers                                                 */
    ptL = (TypPoint16*)PTR_BAG(hdL);
    ptR = (TypPoint32*)PTR_BAG(hdR);
    ptC = (TypPoint32*)PTR_BAG(hdC);

    /* its faster if the both permutations have the same size              */
    if ( degL == degR ) {
        for ( p = 0; p < degC; p++ )
            ptC[ ptR[p] ] = ptR[ ptL[p] ];
    }

    /* otherwise we have to use the macro 'IMAGE' three times              */
    else {
        for ( p = 0; p < degC; p++ )
            ptC[ IMAGE(p,ptR,degR) ] = IMAGE( IMAGE(p,ptL,degL), ptR, degR );
    }

    /* return the result                                                   */
    return hdC;
}

Bag       PowQP (Bag hdL, Bag hdR)
{
    Bag           hdC;            /* handle of the conjugation (res) */
    UInt       degC;           /* degree of the conjugation       */
    TypPoint32          * ptC;          /* pointer to the conjugation      */
    UInt       degL;           /* degree of the left operand      */
    TypPoint32          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint16          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* compute the size of the result and allocate a bag                   */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint32);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint16);
    degC = degL < degR ? degR : degL;
    hdC = NewBag( T_PERM32, (UInt)(degC * sizeof(TypPoint32)) );

    /* set up the pointers                                                 */
    ptL = (TypPoint32*)PTR_BAG(hdL);
    ptR = (TypPoint16*)PTR_BAG(hdR);
    ptC = (TypPoint32*)PTR_BAG(hdC);

    /* its faster if the both permutations have the same size              */
    if ( degL == degR ) {
        for ( p = 0; p < degC; p++ )
            ptC[ ptR[p] ] = ptR[ ptL[p] ];
    }

    /* otherwise we have to use the macro 'IMAGE' three times              */
    else {
        for ( p = 0; p < degC; p++ )
            ptC[ IMAGE(p,ptR,degR) ] = IMAGE( IMAGE(p,ptL,degL), ptR, degR );
    }

    /* return the result                                                   */
    return hdC;
}

Bag       PowQQ (Bag hdL, Bag hdR)
{
    Bag           hdC;            /* handle of the conjugation (res) */
    UInt       degC;           /* degree of the conjugation       */
    TypPoint32          * ptC;          /* pointer to the conjugation      */
    UInt       degL;           /* degree of the left operand      */
    TypPoint32          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint32          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* compute the size of the result and allocate a bag                   */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint32);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint32);
    degC = degL < degR ? degR : degL;
    hdC = NewBag( T_PERM32, (UInt)(degC * sizeof(TypPoint32)) );

    /* set up the pointers                                                 */
    ptL = (TypPoint32*)PTR_BAG(hdL);
    ptR = (TypPoint32*)PTR_BAG(hdR);
    ptC = (TypPoint32*)PTR_BAG(hdC);

    /* its faster if the both permutations have the same size              */
    if ( degL == degR ) {
        for ( p = 0; p < degC; p++ )
            ptC[ ptR[p] ] = ptR[ ptL[p] ];
    }

    /* otherwise we have to use the macro 'IMAGE' three times              */
    else {
        for ( p = 0; p < degC; p++ )
            ptC[ IMAGE(p,ptR,degR) ] = IMAGE( IMAGE(p,ptL,degL), ptR, degR );
    }

    /* return the result                                                   */
    return hdC;
}


/****************************************************************************
**
*F  CommPerm( <hdL>, <hdR> )  . . . . . . . .  commutator of two permutations
**
**  'CommPerm' returns the  commutator  of  the  two permutations  <hdL>  and
**  <hdR>, that is defined as '<hd>\^-1 \*\ <hdR>\^-1 \*\ <hdL> \*\ <hdR>'.
**
**  Is called from the 'Comm' binop, so both operands are already evaluated.
*/
Bag       CommPP (Bag hdL, Bag hdR)
{
    Bag           hdC;            /* handle of the commutator  (res) */
    UInt       degC;           /* degree of the commutator        */
    TypPoint16          * ptC;          /* pointer to the commutator       */
    UInt       degL;           /* degree of the left operand      */
    TypPoint16          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint16          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* compute the size of the result and allocate a bag                   */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint16);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint16);
    degC = degL < degR ? degR : degL;
    hdC = NewBag( T_PERM16, (UInt)(degC * sizeof(TypPoint16)) );

    /* set up the pointers                                                 */
    ptL = (TypPoint16*)PTR_BAG(hdL);
    ptR = (TypPoint16*)PTR_BAG(hdR);
    ptC = (TypPoint16*)PTR_BAG(hdC);

    /* its faster if the both permutations have the same size              */
    if ( degL == degR ) {
        for ( p = 0; p < degC; p++ )
            ptC[ ptL[ ptR[ p ] ] ] = ptR[ ptL[ p ] ];
    }

    /* otherwise we have to use the macro 'IMAGE' four times               */
    else {
        for ( p = 0; p < degC; p++ )
            ptC[ IMAGE( IMAGE(p,ptR,degR), ptL, degL ) ]
               = IMAGE( IMAGE(p,ptL,degL), ptR, degR );
    }

    /* return the result                                                   */
    return hdC;
}

Bag       CommPQ (Bag hdL, Bag hdR)
{
    Bag           hdC;            /* handle of the commutator  (res) */
    UInt       degC;           /* degree of the commutator        */
    TypPoint32          * ptC;          /* pointer to the commutator       */
    UInt       degL;           /* degree of the left operand      */
    TypPoint16          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint32          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* compute the size of the result and allocate a bag                   */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint16);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint32);
    degC = degL < degR ? degR : degL;
    hdC = NewBag( T_PERM32, (UInt)(degC * sizeof(TypPoint32)) );

    /* set up the pointers                                                 */
    ptL = (TypPoint16*)PTR_BAG(hdL);
    ptR = (TypPoint32*)PTR_BAG(hdR);
    ptC = (TypPoint32*)PTR_BAG(hdC);

    /* its faster if the both permutations have the same size              */
    if ( degL == degR ) {
        for ( p = 0; p < degC; p++ )
            ptC[ ptL[ ptR[ p ] ] ] = ptR[ ptL[ p ] ];
    }

    /* otherwise we have to use the macro 'IMAGE' four times               */
    else {
        for ( p = 0; p < degC; p++ )
            ptC[ IMAGE( IMAGE(p,ptR,degR), ptL, degL ) ]
               = IMAGE( IMAGE(p,ptL,degL), ptR, degR );
    }

    /* return the result                                                   */
    return hdC;
}

Bag       CommQP (Bag hdL, Bag hdR)
{
    Bag           hdC;            /* handle of the commutator  (res) */
    UInt       degC;           /* degree of the commutator        */
    TypPoint32          * ptC;          /* pointer to the commutator       */
    UInt       degL;           /* degree of the left operand      */
    TypPoint32          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint16          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* compute the size of the result and allocate a bag                   */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint32);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint16);
    degC = degL < degR ? degR : degL;
    hdC = NewBag( T_PERM32, (UInt)(degC * sizeof(TypPoint32)) );

    /* set up the pointers                                                 */
    ptL = (TypPoint32*)PTR_BAG(hdL);
    ptR = (TypPoint16*)PTR_BAG(hdR);
    ptC = (TypPoint32*)PTR_BAG(hdC);

    /* its faster if the both permutations have the same size              */
    if ( degL == degR ) {
        for ( p = 0; p < degC; p++ )
            ptC[ ptL[ ptR[ p ] ] ] = ptR[ ptL[ p ] ];
    }

    /* otherwise we have to use the macro 'IMAGE' four times               */
    else {
        for ( p = 0; p < degC; p++ )
            ptC[ IMAGE( IMAGE(p,ptR,degR), ptL, degL ) ]
               = IMAGE( IMAGE(p,ptL,degL), ptR, degR );
    }

    /* return the result                                                   */
    return hdC;
}

Bag       CommQQ (Bag hdL, Bag hdR)
{
    Bag           hdC;            /* handle of the commutator  (res) */
    UInt       degC;           /* degree of the commutator        */
    TypPoint32          * ptC;          /* pointer to the commutator       */
    UInt       degL;           /* degree of the left operand      */
    TypPoint32          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint32          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* compute the size of the result and allocate a bag                   */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint32);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint32);
    degC = degL < degR ? degR : degL;
    hdC = NewBag( T_PERM32, (UInt)(degC * sizeof(TypPoint32)) );

    /* set up the pointers                                                 */
    ptL = (TypPoint32*)PTR_BAG(hdL);
    ptR = (TypPoint32*)PTR_BAG(hdR);
    ptC = (TypPoint32*)PTR_BAG(hdC);

    /* its faster if the both permutations have the same size              */
    if ( degL == degR ) {
        for ( p = 0; p < degC; p++ )
            ptC[ ptL[ ptR[ p ] ] ] = ptR[ ptL[ p ] ];
    }

    /* otherwise we have to use the macro 'IMAGE' four times               */
    else {
        for ( p = 0; p < degC; p++ )
            ptC[ IMAGE( IMAGE(p,ptR,degR), ptL, degL ) ]
               = IMAGE( IMAGE(p,ptL,degL), ptR, degR );
    }

    /* return the result                                                   */
    return hdC;
}


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
Bag       EqPP (Bag hdL, Bag hdR)
{
    UInt       degL;           /* degree of the left operand      */
    TypPoint16          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint16          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* get the degrees                                                     */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint16);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint16);

    /* set up the pointers                                                 */
    ptL = (TypPoint16*)PTR_BAG(hdL);
    ptR = (TypPoint16*)PTR_BAG(hdR);

    /* search for a difference and return HdFalse if you find one          */
    if ( degL <= degR ) {
        for ( p = 0; p < degL; p++ )
            if ( *(ptL++) != *(ptR++) )
                return HdFalse;
        for ( p = degL; p < degR; p++ )
            if (        p != *(ptR++) )
                return HdFalse;
    }
    else {
        for ( p = 0; p < degR; p++ )
            if ( *(ptL++) != *(ptR++) )
                return HdFalse;
        for ( p = degR; p < degL; p++ )
            if ( *(ptL++) !=        p )
                return HdFalse;
    }

    /* otherwise they must be equal                                        */
    return HdTrue;
}

Bag       EqPQ (Bag hdL, Bag hdR)
{
    UInt       degL;           /* degree of the left operand      */
    TypPoint16          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint32          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* get the degrees                                                     */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint16);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint32);

    /* set up the pointers                                                 */
    ptL = (TypPoint16*)PTR_BAG(hdL);
    ptR = (TypPoint32*)PTR_BAG(hdR);

    /* search for a difference and return HdFalse if you find one          */
    if ( degL <= degR ) {
        for ( p = 0; p < degL; p++ )
            if ( *(ptL++) != *(ptR++) )
                return HdFalse;
        for ( p = degL; p < degR; p++ )
            if (        p != *(ptR++) )
                return HdFalse;
    }
    else {
        for ( p = 0; p < degR; p++ )
            if ( *(ptL++) != *(ptR++) )
                return HdFalse;
        for ( p = degR; p < degL; p++ )
            if ( *(ptL++) !=        p )
                return HdFalse;
    }

    /* otherwise they must be equal                                        */
    return HdTrue;
}

Bag       EqQP (Bag hdL, Bag hdR)
{
    UInt       degL;           /* degree of the left operand      */
    TypPoint32          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint16          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* get the degrees                                                     */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint32);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint16);

    /* set up the pointers                                                 */
    ptL = (TypPoint32*)PTR_BAG(hdL);
    ptR = (TypPoint16*)PTR_BAG(hdR);

    /* search for a difference and return HdFalse if you find one          */
    if ( degL <= degR ) {
        for ( p = 0; p < degL; p++ )
            if ( *(ptL++) != *(ptR++) )
                return HdFalse;
        for ( p = degL; p < degR; p++ )
            if (        p != *(ptR++) )
                return HdFalse;
    }
    else {
        for ( p = 0; p < degR; p++ )
            if ( *(ptL++) != *(ptR++) )
                return HdFalse;
        for ( p = degR; p < degL; p++ )
            if ( *(ptL++) !=        p )
                return HdFalse;
    }

    /* otherwise they must be equal                                        */
    return HdTrue;
}

Bag       EqQQ (Bag hdL, Bag hdR)
{
    UInt       degL;           /* degree of the left operand      */
    TypPoint32          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint32          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* get the degrees                                                     */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint32);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint32);

    /* set up the pointers                                                 */
    ptL = (TypPoint32*)PTR_BAG(hdL);
    ptR = (TypPoint32*)PTR_BAG(hdR);

    /* search for a difference and return HdFalse if you find one          */
    if ( degL <= degR ) {
        for ( p = 0; p < degL; p++ )
            if ( *(ptL++) != *(ptR++) )
                return HdFalse;
        for ( p = degL; p < degR; p++ )
            if (        p != *(ptR++) )
                return HdFalse;
    }
    else {
        for ( p = 0; p < degR; p++ )
            if ( *(ptL++) != *(ptR++) )
                return HdFalse;
        for ( p = degR; p < degL; p++ )
            if ( *(ptL++) !=        p )
                return HdFalse;
    }

    /* otherwise they must be equal                                        */
    return HdTrue;
}


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
Bag       LtPP (Bag hdL, Bag hdR)
{
    UInt       degL;           /* degree of the left operand      */
    TypPoint16          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint16          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* get the degrees of the permutations                                 */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint16);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint16);

    /* set up the pointers                                                 */
    ptL = (TypPoint16*)PTR_BAG(hdL);
    ptR = (TypPoint16*)PTR_BAG(hdR);

    /* search for a difference and return if you find one                  */
    if ( degL <= degR ) {
        for ( p = 0; p < degL; p++ )
	    if ( *(ptL++) != *(ptR++) ) {
                if ( *(--ptL) < *(--ptR) )  return HdTrue ;
                else                        return HdFalse;
	    }
        for ( p = degL; p < degR; p++ )
	    if (        p != *(ptR++) ) {
                if (        p < *(--ptR) )  return HdTrue ;
                else                        return HdFalse;
	    }
    }
    else {
        for ( p = 0; p < degR; p++ )
            if ( *(ptL++) != *(ptR++) ) {
                if ( *(--ptL) < *(--ptR) )  return HdTrue ;
                else                        return HdFalse;
	    }
        for ( p = degR; p < degL; p++ )
            if ( *(ptL++) != p ) {
                if ( *(--ptL) <        p )  return HdTrue ;
                else                        return HdFalse;
	    }
    }

    /* otherwise they must be equal                                        */
    return HdFalse;
}

Bag       LtPQ (Bag hdL, Bag hdR)
{
    UInt       degL;           /* degree of the left operand      */
    TypPoint16          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint32          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* get the degrees of the permutations                                 */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint16);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint32);

    /* set up the pointers                                                 */
    ptL = (TypPoint16*)PTR_BAG(hdL);
    ptR = (TypPoint32*)PTR_BAG(hdR);

    /* search for a difference and return if you find one                  */
    if ( degL <= degR ) {
        for ( p = 0; p < degL; p++ )
            if ( *(ptL++) != *(ptR++) ) {
                if ( *(--ptL) < *(--ptR) )  return HdTrue ;
                else                        return HdFalse;
	    }
        for ( p = degL; p < degR; p++ )
            if (        p != *(ptR++) ) {
                if (        p < *(--ptR) )  return HdTrue ;
                else                        return HdFalse;
	    }
    }
    else {
        for ( p = 0; p < degR; p++ )
            if ( *(ptL++) != *(ptR++) ) {
                if ( *(--ptL) < *(--ptR) )  return HdTrue ;
                else                        return HdFalse;
	    }
        for ( p = degR; p < degL; p++ )
            if ( *(ptL++) != p ) {
                if ( *(--ptL) <        p )  return HdTrue ;
                else                        return HdFalse;
	    }
    }

    /* otherwise they must be equal                                        */
    return HdFalse;
}

Bag       LtQP (Bag hdL, Bag hdR)
{
    UInt       degL;           /* degree of the left operand      */
    TypPoint32          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint16          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* get the degrees of the permutations                                 */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint32);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint16);

    /* set up the pointers                                                 */
    ptL = (TypPoint32*)PTR_BAG(hdL);
    ptR = (TypPoint16*)PTR_BAG(hdR);

    /* search for a difference and return if you find one                  */
    if ( degL <= degR ) {
        for ( p = 0; p < degL; p++ )
            if ( *(ptL++) != *(ptR++) ) {
                if ( *(--ptL) < *(--ptR) )  return HdTrue ;
                else                        return HdFalse;
	    }
        for ( p = degL; p < degR; p++ )
            if (        p != *(ptR++) ) {
                if (        p < *(--ptR) )  return HdTrue ;
                else                        return HdFalse;
	    }
    }
    else {
        for ( p = 0; p < degR; p++ )
            if ( *(ptL++) != *(ptR++) ) {
                if ( *(--ptL) < *(--ptR) )  return HdTrue ;
                else                        return HdFalse;
	    }
        for ( p = degR; p < degL; p++ )
            if ( *(ptL++) != p ) {
                if ( *(--ptL) <        p )  return HdTrue ;
                else                        return HdFalse;
	    }
    }

    /* otherwise they must be equal                                        */
    return HdFalse;
}

Bag       LtQQ (Bag hdL, Bag hdR)
{
    UInt       degL;           /* degree of the left operand      */
    TypPoint32          * ptL;          /* pointer to the left operand     */
    UInt       degR;           /* degree of the right operand     */
    TypPoint32          * ptR;          /* pointer to the right operand    */
    UInt       p;              /* loop variable                   */

    /* get the degrees of the permutations                                 */
    degL = GET_SIZE_BAG(hdL) / sizeof(TypPoint32);
    degR = GET_SIZE_BAG(hdR) / sizeof(TypPoint32);

    /* set up the pointers                                                 */
    ptL = (TypPoint32*)PTR_BAG(hdL);
    ptR = (TypPoint32*)PTR_BAG(hdR);

    /* search for a difference and return if you find one                  */
    if ( degL <= degR ) {
        for ( p = 0; p < degL; p++ )
            if ( *(ptL++) != *(ptR++) ) {
                if ( *(--ptL) < *(--ptR) )  return HdTrue ;
                else                        return HdFalse;
	    }
        for ( p = degL; p < degR; p++ ) 
            if (        p != *(ptR++) ) {
                if (        p < *(--ptR) )  return HdTrue ;
                else                        return HdFalse;
	    }
    }
    else {
        for ( p = 0; p < degR; p++ )
            if ( *(ptL++) != *(ptR++) ) {
                if ( *(--ptL) < *(--ptR) )  return HdTrue ;
                else                        return HdFalse;
	    }
        for ( p = degR; p < degL; p++ )
            if ( *(ptL++) != p ) {
                if ( *(--ptL) <        p )  return HdTrue ;
                else                        return HdFalse;
	    }
    }

    /* otherwise they must be equal                                        */
    return HdFalse;
}


/****************************************************************************
**
*F  PrPerm( <hdPerm> )  . . . . . . . . . . . . . . . . . print a permutation
**
**  'PrPerm' prints the permutation <hdPerm> in the usual cycle notation.  It
**  uses the degree to print all points with same width, which  looks  nicer.
**  Linebreaks are prefered most after cycles and  next  most  after  commas.
**
**  It does not remember which points have already  been  printed.  To  avoid
**  printing a cycle twice each is printed with the smallest  element  first.
**  This may in the worst case, for (1,2,..,n), take n^2/2 steps, but is fast
**  enough to keep a terminal at 9600 baud busy for all but the extrem cases.
**  This is done, because it is forbidden to create new bags during printing.
*/
void            PrPermP (Bag hdPerm)
{
    UInt       degPerm;        /* degree of the permutation       */
    TypPoint16          * ptPerm;       /* pointer to the permutation      */
    UInt       p,  q;          /* loop variables                  */
    short               isId;           /* permutation is the identity?    */
    char                * fmt1, * fmt2; /* common formats to print points  */

    /* set up the formats used, so all points are printed with equal width */
    degPerm = GET_SIZE_BAG(hdPerm) / sizeof(TypPoint16);
    if      ( degPerm <    10 ) { fmt1 = "%>(%>%1d%<"; fmt2 = ",%>%1d%<"; }
    else if ( degPerm <   100 ) { fmt1 = "%>(%>%2d%<"; fmt2 = ",%>%2d%<"; }
    else if ( degPerm <  1000 ) { fmt1 = "%>(%>%3d%<"; fmt2 = ",%>%3d%<"; }
    else if ( degPerm < 10000 ) { fmt1 = "%>(%>%4d%<"; fmt2 = ",%>%4d%<"; }
    else                        { fmt1 = "%>(%>%5d%<"; fmt2 = ",%>%5d%<"; }

    /* run through all points                                              */
    isId = 1;
    ptPerm = (TypPoint16*)PTR_BAG(hdPerm);
    for ( p = 0; p < degPerm; p++ ) {

        /* find the smallest element in this cycle                         */
        q = ptPerm[p];
        while ( p < q )  q = ptPerm[q];

        /* if the smallest is the one we started with lets print the cycle */
        if ( p == q && ptPerm[p] != p ) {
            isId = 0;
            Pr(fmt1,(Int)(p+1),0);
            for ( q = ptPerm[p]; q != p; q = ptPerm[q] )
                Pr(fmt2,(Int)(q+1),0);
            Pr("%<)",0,0);
        }

    }

    /* special case for the identity                                       */
    if ( isId )  Pr("()",0,0);
}

void            PrPermQ (Bag hdPerm)
{
    UInt       degPerm;        /* degree of the permutation       */
    TypPoint32          * ptPerm;       /* pointer to the permutation      */
    UInt       p,  q;          /* loop variables                  */
    short               isId;           /* permutation is the identity?    */
    char                * fmt1, * fmt2; /* common formats to print points  */

    /* set up the formats used, so all points are printed with equal width */
    degPerm = GET_SIZE_BAG(hdPerm) / sizeof(TypPoint32);
    if      ( degPerm <    10 ) { fmt1 = "%>(%>%1d%<"; fmt2 = ",%>%1d%<"; }
    else if ( degPerm <   100 ) { fmt1 = "%>(%>%2d%<"; fmt2 = ",%>%2d%<"; }
    else if ( degPerm <  1000 ) { fmt1 = "%>(%>%3d%<"; fmt2 = ",%>%3d%<"; }
    else if ( degPerm < 10000 ) { fmt1 = "%>(%>%4d%<"; fmt2 = ",%>%4d%<"; }
    else                        { fmt1 = "%>(%>%5d%<"; fmt2 = ",%>%5d%<"; }

    /* run through all points                                              */
    isId = 1;
    ptPerm = (TypPoint32*)PTR_BAG(hdPerm);
    for ( p = 0; p < degPerm; p++ ) {

        /* find the smallest element in this cycle                         */
        q = ptPerm[p];
        while ( p < q )  q = ptPerm[q];

        /* if the smallest is the one we started with lets print the cycle */
        if ( p == q && ptPerm[p] != p ) {
            isId = 0;
            Pr(fmt1,(Int)(p+1),0);
            for ( q = ptPerm[p]; q != p; q = ptPerm[q] )
                Pr(fmt2,(Int)(q+1),0);
            Pr("%<)",0,0);
        }

    }

    /* special case for the identity                                       */
    if ( isId )  Pr("()",0,0);
}


/****************************************************************************
**
*F  PrMakeperm( <hdPerm> )  . . . . . . . . . .  print a variable permutation
**
**  'PrMakeperm' prints the variable permutation <hdPerm>  in the usual cycle
**  notation.
**
**  Linebreaks are prefered most after cycles and  next  most  after  commas.
*/
void            PrMakeperm (Bag hdPerm)
{
    Bag           hdCyc;          /* handle of one cycle             */
    UInt       i,  k;          /* loop variables                  */

    /* print all cycles                                                    */
    for ( i = 0; i < GET_SIZE_BAG(hdPerm)/SIZE_HD; i++ ) {
        Pr("%>(",0,0);

        /* print all elements of that cycle                                */
        hdCyc = PTR_BAG(hdPerm)[i];
        for ( k = 0; k < GET_SIZE_BAG(hdCyc)/SIZE_HD; k++ ) {
            Pr("%>",0,0);
            Print( PTR_BAG(hdCyc)[k] );
            Pr("%<",0,0);
            if ( k < GET_SIZE_BAG(hdCyc)/SIZE_HD-1 )  Pr(",",0,0);
        }

        Pr("%<)",0,0);
    }
}


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
Bag       FunIsPerm (Bag hdCall)
{
    Bag           hdObj;          /* handle of the object            */

    /* evaluate and check the argument                                     */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: IsPerm( <obj> )",0,0);
    hdObj = EVAL( PTR_BAG(hdCall)[1] );
    if ( hdObj == HdVoid )
        return Error("IsPerm: function must return a value",0,0);

    /* return 'true' if <obj> is a permutation and 'false' otherwise       */
    if ( GET_TYPE_BAG(hdObj) == T_PERM16 || GET_TYPE_BAG(hdObj) == T_PERM32 )
        return HdTrue;
    else
        return HdFalse;
}


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
Bag       FunPermList (Bag hdCall)
{
    Bag           hdPerm;         /* handle of the permutation       */
    TypPoint16          * ptPerm16;     /* pointer to the permutation      */
    TypPoint32          * ptPerm32;     /* pointer to the permutation      */
    UInt       degPerm;        /* degree of the permutation       */
    Bag           hdList;         /* handle of the list (argument)   */
    Bag           * ptList;       /* pointer to the list             */
    TypPoint16          * ptTmp16;      /* pointer to the buffer bag       */
    TypPoint32          * ptTmp32;      /* pointer to the buffer bag       */
    Int                i,  k;          /* loop variables                  */

    /* evaluate and check the arguments                                    */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: PermList( <list> )",0,0);
    hdList = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IS_LIST( hdList ) )
        return Error("usage: PermList( <list> )",0,0);
    PLAIN_LIST( hdList );

    /* handle small permutations                                           */
    if ( LEN_LIST( hdList ) <= 65536 ) {

        degPerm = LEN_LIST( hdList );

        /* make sure that the global buffer bag is large enough for checkin*/
        if ( GET_SIZE_BAG(HdPerm) < degPerm * sizeof(TypPoint16) )
            Resize( HdPerm, degPerm * sizeof(TypPoint16) );

        /* allocate the bag for the permutation and get pointer            */
        hdPerm   = NewBag( T_PERM16, degPerm * sizeof(TypPoint16) );
        ptPerm16 = (TypPoint16*)PTR_BAG(hdPerm);
        ptList   = PTR_BAG(hdList);
        ptTmp16  = (TypPoint16*)PTR_BAG(HdPerm);

        /* run through all entries of the list                             */
        for ( i = 1; i <= degPerm; i++ ) {

            /* get the <i>th entry of the list                             */
            if ( ptList[i] == 0 ) {
             for ( i = 1; i <= degPerm; i++ )  ptTmp16[i-1] = 0;
             return Error("PermList: <list>[%d] must be defined",(Int)i,0);
            }
            if ( GET_TYPE_BAG(ptList[i]) != T_INT ) {
             for ( i = 1; i <= degPerm; i++ )  ptTmp16[i-1] = 0;
             return Error("PermList: <list>[%d] must be integer",(Int)i,0);
            }
            k = HD_TO_INT(ptList[i]);
            if ( k <= 0 || degPerm < k ) {
                for ( i = 1; i <= degPerm; i++ )  ptTmp16[i-1] = 0;
                return Error("PermList: <list>[%d] must lie in [1..%d]",
                             (Int)i, (Int)degPerm );
            }

            /* make sure we haven't seen this entry yet                     */
            if ( ptTmp16[k-1] != 0 ) {
                for ( i = 1; i <= degPerm; i++ )  ptTmp16[i-1] = 0;
                return Error("PermList: <point> %d must occur only once",
                             (Int)k, 0 );
            }
            ptTmp16[k-1] = 1;

            /* and finally copy it into the permutation                    */
            ptPerm16[i-1] = k-1;
        }

        /* make the buffer bag clean again                                 */
        for ( i = 1; i <= degPerm; i++ )
            ptTmp16[i-1] = 0;

    }

    /* handle large permutations                                           */
    else {

        degPerm = LEN_LIST( hdList );

        /* make sure that the global buffer bag is large enough for checkin*/
        if ( GET_SIZE_BAG(HdPerm) < degPerm * sizeof(TypPoint32) )
            Resize( HdPerm, degPerm * sizeof(TypPoint32) );

        /* allocate the bag for the permutation and get pointer            */
        hdPerm   = NewBag( T_PERM32, degPerm * sizeof(TypPoint32) );
        ptPerm32 = (TypPoint32*)PTR_BAG(hdPerm);
        ptList   = PTR_BAG(hdList);
        ptTmp32  = (TypPoint32*)PTR_BAG(HdPerm);

        /* run through all entries of the list                             */
        for ( i = 1; i <= degPerm; i++ ) {

            /* get the <i>th entry of the list                             */
            if ( ptList[i] == 0 ) {
             for ( i = 1; i <= degPerm; i++ )  ptTmp32[i-1] = 0;
             return Error("PermList: <list>[%d] must be defined",(Int)i,0);
            }
            if ( GET_TYPE_BAG(ptList[i]) != T_INT ) {
             for ( i = 1; i <= degPerm; i++ )  ptTmp32[i-1] = 0;
             return Error("PermList: <list>[%d] must be integer",(Int)i,0);
            }
            k = HD_TO_INT(ptList[i]);
            if ( k <= 0 || degPerm < k ) {
                for ( i = 1; i <= degPerm; i++ )  ptTmp32[i-1] = 0;
                return Error("PermList: <list>[%d] must lie in [1..%d]",
                             (Int)i, (Int)degPerm );
            }

            /* make sure we haven't seen this entry yet                     */
            if ( ptTmp32[k-1] != 0 ) {
                for ( i = 1; i <= degPerm; i++ )  ptTmp32[i-1] = 0;
                return Error("PermList: <point> %d must occur only once",
                             (Int)k, 0 );
            }
            ptTmp32[k-1] = 1;

            /* and finally copy it into the permutation                    */
            ptPerm32[i-1] = k-1;
        }

        /* make the buffer bag clean again                                 */
        for ( i = 1; i <= degPerm; i++ )
            ptTmp32[i-1] = 0;

    }

    /* return the permutation                                              */
    return hdPerm;
}


/****************************************************************************
**
*F  FunLargestMovedPointPerm( <hdCall> ) largest point moved by a permutation
**
**  'FunLargestMovedPointPerm' implements the internal function
**  'LargestMovedPointPerm'.
**
**  'LargestMovedPointPerm( <perm> )'
**
**  'LargestMovedPointPerm' returns  the  largest  positive  integer that  is
**  moved by the permutation <perm>.
**
**  This is easy, except that permutations may  contain  trailing  fixpoints.
*/
Bag       FunLargestMovedPointPerm (Bag hdCall)
{
    UInt       sup;            /* support (result)                */
    Bag           hdPerm;         /* handle of the permutation       */
    TypPoint16          * ptPerm16;     /* pointer to the permutation      */
    TypPoint32          * ptPerm32;     /* pointer to the permutation      */

    /* check the argument                                                  */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: LargestMovedPointPerm( <perm> )",0,0);
    hdPerm = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hdPerm) != T_PERM16 && GET_TYPE_BAG(hdPerm) != T_PERM32 )
        return Error("usage: LargestMovedPointPerm( <perm> )",0,0);

    /* handle small permutations                                           */
    if ( GET_TYPE_BAG(hdPerm) == T_PERM16 ) {

        /* find the largest moved point                                    */
        ptPerm16 = (TypPoint16*)PTR_BAG(hdPerm);
        for ( sup = GET_SIZE_BAG(hdPerm)/sizeof(TypPoint16); 1 <= sup; sup-- ) {
            if ( ptPerm16[sup-1] != sup-1 )
                break;
        }

    }

    /* handle large permutations                                           */
    else {

        /* find the largest moved point                                    */
        ptPerm32 = (TypPoint32*)PTR_BAG(hdPerm);
        for ( sup = GET_SIZE_BAG(hdPerm)/sizeof(TypPoint32); 1 <= sup; sup-- ) {
            if ( ptPerm32[sup-1] != sup-1 )
                break;
        }

    }

    /* check for identity                                                  */
    if ( sup == 0 )
      return Error("LargestMovedPointPerm: <perm> must not be the identity",
                   0,0);

    /* return it                                                           */
    return INT_TO_HD( sup );
}


/****************************************************************************
**
*F  FuncCycleLengthPermInt( <hdCall> )  length of a cycle under a permutation
**
**  'FunCycleLengthInt' implements the internal function 'CycleLengthPermInt'
**
**  'CycleLengthPermInt( <perm>, <point> )'
**
**  'CycleLengthPermInt' returns the length of the cycle  of  <point>,  which
**  must be a positive integer, under the permutation <perm>.
**
**  Note that the order of the arguments to this function has been  reversed.
*/
Bag       FunCycleLengthPermInt (Bag hdCall)
{
    Bag           hdPerm;         /* handle of the permutation       */
    TypPoint16          * ptPerm16;     /* pointer to the permutation      */
    TypPoint32          * ptPerm32;     /* pointer to the permutation      */
    UInt       deg;            /* degree of the permutation       */
    Bag           hdPnt;          /* handle of the point             */
    UInt       pnt;            /* value of the point              */
    UInt       len;            /* length of cycle (result)        */
    UInt       p;              /* loop variable                   */

    /* evaluate and check the arguments                                    */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )
        return Error("usage: CycleLengthPermInt( <perm>, <point> )",0,0);
    hdPerm = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hdPerm) != T_PERM16 && GET_TYPE_BAG(hdPerm) != T_PERM32 )
      return Error("CycleLengthPermInt: <perm> must be a permutation",0,0);
    hdPnt = EVAL( PTR_BAG(hdCall)[2] );
    if ( GET_TYPE_BAG(hdPnt) != T_INT || HD_TO_INT(hdPnt) <= 0 )
      return Error("CycleLengthPermInt: <point> must be an integer",0,0);

    /* handle small permutations                                           */
    if ( GET_TYPE_BAG(hdPerm) == T_PERM16 ) {

        /* get pointer to the permutation, the degree, and the point       */
        ptPerm16 = (TypPoint16*)PTR_BAG(hdPerm);
        deg = GET_SIZE_BAG(hdPerm)/sizeof(TypPoint16);
        pnt = HD_TO_INT(hdPnt)-1;

        /* now compute the length by looping over the cycle                */
        len = 1;
        if ( pnt < deg ) {
            for ( p = ptPerm16[pnt]; p != pnt; p = ptPerm16[p] )
                len++;
        }

    }

    /* handle large permutations                                           */
    else {

        /* get pointer to the permutation, the degree, and the point       */
        ptPerm32 = (TypPoint32*)PTR_BAG(hdPerm);
        deg = GET_SIZE_BAG(hdPerm)/sizeof(TypPoint32);
        pnt = HD_TO_INT(hdPnt)-1;

        /* now compute the length by looping over the cycle                */
        len = 1;
        if ( pnt < deg ) {
            for ( p = ptPerm32[pnt]; p != pnt; p = ptPerm32[p] )
                len++;
        }

    }

    /* return the length                                                   */
    return INT_TO_HD(len);
}


/****************************************************************************
**
*F  FunCyclePermInt( <hdCall> ) . . . . . . . . . . .  cycle of a permutation
*
**  'FunCyclePermInt' implements the internal function 'CyclePermInt'.
**
**  'CyclePermInt( <perm>, <point> )'
**
**  'CyclePermInt' returns the cycle of <point>, which  must  be  a  positive
**  integer, under the permutation <perm> as a list.
*/
Bag       FunCyclePermInt (Bag hdCall)
{
    Bag           hdList;         /* handle of the list (result)     */
    Bag           * ptList;       /* pointer to the list             */
    Bag           hdPerm;         /* handle of the permutation       */
    TypPoint16          * ptPerm16;     /* pointer to the permutation      */
    TypPoint32          * ptPerm32;     /* pointer to the permutation      */
    UInt       deg;            /* degree of the permutation       */
    Bag           hdPnt;          /* handle of the point             */
    UInt       pnt;            /* value of the point              */
    UInt       len;            /* length of the cycle             */
    UInt       p;              /* loop variable                   */

    /* evaluate and check the arguments                                    */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )
        return Error("usage: CyclePermInt( <perm>, <point> )",0,0);
    hdPerm = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hdPerm) != T_PERM16 && GET_TYPE_BAG(hdPerm) != T_PERM32 )
        return Error("CyclePermInt: <perm> must be a permutation",0,0);
    hdPnt = EVAL( PTR_BAG(hdCall)[2] );
    if ( GET_TYPE_BAG(hdPnt) != T_INT || HD_TO_INT(hdPnt) <= 0 )
        return Error("CyclePermInt: <point> must be an integer",0,0);

    /* handle small permutations                                           */
    if ( GET_TYPE_BAG(hdPerm) == T_PERM16 ) {

        /* get pointer to the permutation, the degree, and the point       */
        ptPerm16 = (TypPoint16*)PTR_BAG(hdPerm);
        deg = GET_SIZE_BAG(hdPerm)/sizeof(TypPoint16);
        pnt = HD_TO_INT(hdPnt)-1;

        /* now compute the length by looping over the cycle                */
        len = 1;
        if ( pnt < deg ) {
            for ( p = ptPerm16[pnt]; p != pnt; p = ptPerm16[p] )
                len++;
        }

        /* allocate the list                                               */
        hdList = NewBag( T_LIST, SIZE_HD + len*SIZE_HD );
        SET_BAG(hdList, 0,  INT_TO_HD( len ) );
        ptList = PTR_BAG(hdList);
        ptPerm16 = (TypPoint16*)PTR_BAG(hdPerm);

        /* copy the points into the list                                   */
        len = 1;
        ptList[len++] = INT_TO_HD( pnt+1 );
        if ( pnt < deg ) {
            for ( p = ptPerm16[pnt]; p != pnt; p = ptPerm16[p] )
                ptList[len++] = INT_TO_HD( p+1 );
        }

    }

    /* handle large permutations                                           */
    else {

        /* get pointer to the permutation, the degree, and the point       */
        ptPerm32 = (TypPoint32*)PTR_BAG(hdPerm);
        deg = GET_SIZE_BAG(hdPerm)/sizeof(TypPoint32);
        pnt = HD_TO_INT(hdPnt)-1;

        /* now compute the length by looping over the cycle                */
        len = 1;
        if ( pnt < deg ) {
            for ( p = ptPerm32[pnt]; p != pnt; p = ptPerm32[p] )
                len++;
        }

        /* allocate the list                                               */
        hdList = NewBag( T_LIST, SIZE_HD + len*SIZE_HD );
        SET_BAG(hdList, 0,  INT_TO_HD( len ) );
        ptList = PTR_BAG(hdList);
        ptPerm32 = (TypPoint32*)PTR_BAG(hdPerm);

        /* copy the points into the list                                   */
        len = 1;
        ptList[len++] = INT_TO_HD( pnt+1 );
        if ( pnt < deg ) {
            for ( p = ptPerm32[pnt]; p != pnt; p = ptPerm32[p] )
                ptList[len++] = INT_TO_HD( p+1 );
        }

    }

    /* return the list                                                     */
    return hdList;
}


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
Bag       FunOrderPerm (Bag hdCall)
{
    Bag           hdPerm;         /* handle of the permutation       */
    TypPoint16          * ptPerm16;     /* pointer to the permutation      */
    TypPoint32          * ptPerm32;     /* pointer to the permutation      */
    Bag           ord;            /* order (result), may be huge     */
    TypPoint16          * ptKnown16;    /* pointer to temporary bag        */
    TypPoint32          * ptKnown32;    /* pointer to temporary bag        */
    UInt       len;            /* length of one cycle             */
    UInt       gcd,  s,  t;    /* gcd( len, ord ), temporaries    */
    UInt       p,  q;          /* loop variables                  */

    /* check arguments and extract permutation                             */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: OrderPerm( <perm> )",0,0);
    hdPerm = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hdPerm) != T_PERM16 && GET_TYPE_BAG(hdPerm) != T_PERM32 )
        return Error("OrderPerm: <perm> must be a permutation",0,0);

    /* make sure that the buffer bag is large enough                       */
    if ( GET_SIZE_BAG(HdPerm) < GET_SIZE_BAG(hdPerm) )  Resize( HdPerm, GET_SIZE_BAG(hdPerm) );

    /* handle small permutations                                           */
    if ( GET_TYPE_BAG(hdPerm) == T_PERM16 ) {

        /* get the pointer to the bags                                     */
        ptPerm16  = (TypPoint16*)PTR_BAG(hdPerm);
        ptKnown16 = (TypPoint16*)PTR_BAG(HdPerm);

        /* start with order 1                                              */
        ord = INT_TO_HD(1);

        /* loop over all cycles                                            */
        for ( p = 0; p < GET_SIZE_BAG(hdPerm)/sizeof(TypPoint16); p++ ) {

            /* if we haven't looked at this cycle so far                   */
            if ( ptKnown16[p] == 0 && ptPerm16[p] != p ) {

                /* find the length of this cycle                           */
                len = 1;
                for ( q = ptPerm16[p]; q != p; q = ptPerm16[q] ) {
                    len++;  ptKnown16[q] = 1;
                }

                /* compute the gcd with the previously order ord           */
                /* Note that since len is single precision, ord % len is to*/
                gcd = len;  s = HD_TO_INT( ModInt( ord, INT_TO_HD(len) ) );
                while ( s != 0 ) {
                    t = s;  s = gcd % s;  gcd = t;
                }
                ord = ProdInt( ord, INT_TO_HD( len / gcd ) );

            }

        }

        /* clear the buffer bag again                                      */
        for ( p = 0; p < GET_SIZE_BAG(hdPerm)/sizeof(TypPoint16); p++ )
            ptKnown16[p] = 0;

    }

    /* handle larger permutations                                          */
    else {

        /* get the pointer to the bags                                     */
        ptPerm32  = (TypPoint32*)PTR_BAG(hdPerm);
        ptKnown32 = (TypPoint32*)PTR_BAG(HdPerm);

        /* start with order 1                                              */
        ord = INT_TO_HD(1);

        /* loop over all cycles                                            */
        for ( p = 0; p < GET_SIZE_BAG(hdPerm)/sizeof(TypPoint32); p++ ) {

            /* if we haven't looked at this cycle so far                   */
            if ( ptKnown32[p] == 0 && ptPerm32[p] != p ) {

                /* find the length of this cycle                           */
                len = 1;
                for ( q = ptPerm32[p]; q != p; q = ptPerm32[q] ) {
                    len++;  ptKnown32[q] = 1;
                }

                /* compute the gcd with the previously order ord           */
                /* Note that since len is single precision, ord % len is to*/
                gcd = len;  s = HD_TO_INT( ModInt( ord, INT_TO_HD(len) ) );
                while ( s != 0 ) {
                    t = s;  s = gcd % s;  gcd = t;
                }
                ord = ProdInt( ord, INT_TO_HD( len / gcd ) );

            }

        }

        /* clear the buffer bag again                                      */
        for ( p = 0; p < GET_SIZE_BAG(hdPerm)/sizeof(TypPoint32); p++ )
            ptKnown32[p] = 0;

    }

    /* return the order                                                    */
    return ord;
}


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
Bag       FunSignPerm (Bag hdCall)
{
    Bag           hdPerm;         /* handle of the permutation       */
    TypPoint16          * ptPerm16;     /* pointer to the permutation      */
    TypPoint32          * ptPerm32;     /* pointer to the permutation      */
    Int                sign;           /* sign (result)                   */
    TypPoint16          * ptKnown16;    /* pointer to temporary bag        */
    TypPoint32          * ptKnown32;    /* pointer to temporary bag        */
    UInt       len;            /* length of one cycle             */
    UInt       p,  q;          /* loop variables                  */

    /* check arguments and extract permutation                             */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: SignPerm( <perm> )",0,0);
    hdPerm = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hdPerm) != T_PERM16 && GET_TYPE_BAG(hdPerm) != T_PERM32 )
        return Error("SignPerm: <perm> must be a permutation",0,0);

    /* make sure that the buffer bag is large enough                       */
    if ( GET_SIZE_BAG(HdPerm) < GET_SIZE_BAG(hdPerm) )  Resize( HdPerm, GET_SIZE_BAG(hdPerm) );

    /* handle small permutations                                           */
    if ( GET_TYPE_BAG(hdPerm) == T_PERM16 ) {

        /* get the pointer to the bags                                     */
        ptPerm16  = (TypPoint16*)PTR_BAG(hdPerm);
        ptKnown16 = (TypPoint16*)PTR_BAG(HdPerm);

        /* start with sign  1                                              */
        sign = 1;

        /* loop over all cycles                                            */
        for ( p = 0; p < GET_SIZE_BAG(hdPerm)/sizeof(TypPoint16); p++ ) {

            /* if we haven't looked at this cycle so far                   */
            if ( ptKnown16[p] == 0 && ptPerm16[p] != p ) {

                /* find the length of this cycle                           */
                len = 1;
                for ( q = ptPerm16[p]; q != p; q = ptPerm16[q] ) {
                    len++;  ptKnown16[q] = 1;
                }

                /* if the length is even invert the sign                   */
                if ( len % 2 == 0 )
                    sign = -sign;

            }

        }

        /* clear the buffer bag again                                      */
        for ( p = 0; p < GET_SIZE_BAG(hdPerm)/sizeof(TypPoint16); p++ )
            ptKnown16[p] = 0;

    }

    /* handle large permutations                                           */
    else {

        /* get the pointer to the bags                                     */
        ptPerm32  = (TypPoint32*)PTR_BAG(hdPerm);
        ptKnown32 = (TypPoint32*)PTR_BAG(HdPerm);

        /* start with sign  1                                              */
        sign = 1;

        /* loop over all cycles                                            */
        for ( p = 0; p < GET_SIZE_BAG(hdPerm)/sizeof(TypPoint32); p++ ) {

            /* if we haven't looked at this cycle so far                   */
            if ( ptKnown32[p] == 0 && ptPerm32[p] != p ) {

                /* find the length of this cycle                           */
                len = 1;
                for ( q = ptPerm32[p]; q != p; q = ptPerm32[q] ) {
                    len++;  ptKnown32[q] = 1;
                }

                /* if the length is even invert the sign                   */
                if ( len % 2 == 0 )
                    sign = -sign;

            }

        }

        /* clear the buffer bag again                                      */
        for ( p = 0; p < GET_SIZE_BAG(hdPerm)/sizeof(TypPoint32); p++ )
            ptKnown32[p] = 0;

    }

    /* return the sign                                                     */
    return INT_TO_HD( sign );
}


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
Bag       FunSmallestGeneratorPerm (Bag hdCall)
{
    Bag           hdSmall;        /* handle of the smallest gen      */
    TypPoint16          * ptSmall16;    /* pointer to the smallest gen     */
    TypPoint32          * ptSmall32;    /* pointer to the smallest gen     */
    Bag           hdPerm;         /* handle of the permutation       */
    TypPoint16          * ptPerm16;     /* pointer to the permutation      */
    TypPoint32          * ptPerm32;     /* pointer to the permutation      */
    TypPoint16          * ptKnown16;    /* pointer to temporary bag        */
    TypPoint32          * ptKnown32;    /* pointer to temporary bag        */
    Bag           ord;            /* order, may be huge              */
    Bag           pow;            /* power, may also be huge         */
    UInt       len;            /* length of one cycle             */
    UInt       gcd,  s,  t;    /* gcd( len, ord ), temporaries    */
    UInt       min;            /* minimal element in a cycle      */
    UInt       p,  q;          /* loop variables                  */
    UInt       l, n, x, gcd2;  /* loop variable                   */

    /* check arguments and extract permutation                             */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: SmallestGeneratorPerm( <perm> )",0,0);
    hdPerm = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hdPerm) != T_PERM16 && GET_TYPE_BAG(hdPerm) != T_PERM32 )
        return Error("SmallestGeneratorPerm: <perm> must be a permutation",
                     0,0);

    /* make sure that the buffer bag is large enough                       */
    if ( GET_SIZE_BAG(HdPerm) < GET_SIZE_BAG(hdPerm) )  Resize( HdPerm, GET_SIZE_BAG(hdPerm) );

    /* handle small permutations                                           */
    if ( GET_TYPE_BAG(hdPerm) == T_PERM16 ) {

        /* allocate the result bag                                         */
        hdSmall = NewBag( T_PERM16, (UInt)GET_SIZE_BAG(hdPerm) );

        /* get the pointer to the bags                                     */
        ptPerm16   = (TypPoint16*)PTR_BAG(hdPerm);
        ptKnown16  = (TypPoint16*)PTR_BAG(HdPerm);
        ptSmall16  = (TypPoint16*)PTR_BAG(hdSmall);

        /* we only know that we must raise <perm> to a power = 0 mod 1     */
        ord = INT_TO_HD(1);  pow = INT_TO_HD(0);

        /* loop over all cycles                                            */
        for ( p = 0; p < GET_SIZE_BAG(hdPerm)/sizeof(TypPoint16); p++ ) {

            /* if we haven't looked at this cycle so far                   */
            if ( ptKnown16[p] == 0 ) {

                /* find the length of this cycle                           */
                len = 1;
                for ( q = ptPerm16[p]; q != p; q = ptPerm16[q] ) {
                    len++;  ptKnown16[q] = 1;
                }

                /* compute the gcd with the previously order ord           */
                /* Note that since len is single precision, ord % len is to*/
                gcd = len;  s = HD_TO_INT( ModInt( ord, INT_TO_HD(len) ) );
                while ( s != 0 ) {
                    t = s;  s = gcd % s;  gcd = t;
                }

                /* we must raise the cycle into a power = pow mod gcd      */
                x = HD_TO_INT( ModInt( pow, INT_TO_HD( gcd ) ) );

                /* find the smallest element in the cycle at such a positio*/
                min = GET_SIZE_BAG(hdPerm)/sizeof(TypPoint16)-1;
                n = 0;
                for ( q = p, l = 0; l < len; l++ ) {
                    gcd2 = len;  s = l;
                    while ( s != 0 ) { t = s; s = gcd2 % s; gcd2 = t; }
                    if ( l % gcd == x && gcd2 == 1 && q <= min ) {
                        min = q;
                        n = l;
                    }
                    q = ptPerm16[q];
                }

                /* raise the cycle to that power and put it in the result  */
                ptSmall16[p] = min;
                for ( q = ptPerm16[p]; q != p; q = ptPerm16[q] ) {
                    min = ptPerm16[min];  ptSmall16[q] = min;
                }

                /* compute the new order and the new power                 */
                while ( HD_TO_INT( ModInt( pow, INT_TO_HD(len) ) ) != n )
                    pow = SumInt( pow, ord );
                ord = ProdInt( ord, INT_TO_HD( len / gcd ) );

            }

        }

        /* clear the buffer bag again                                      */
        for ( p = 0; p < GET_SIZE_BAG(hdPerm)/sizeof(TypPoint16); p++ )
            ptKnown16[p] = 0;

    }

    /* handle large permutations                                           */
    else {

        /* allocate the result bag                                         */
        hdSmall = NewBag( T_PERM32, (UInt)GET_SIZE_BAG(hdPerm) );

        /* get the pointer to the bags                                     */
        ptPerm32   = (TypPoint32*)PTR_BAG(hdPerm);
        ptKnown32  = (TypPoint32*)PTR_BAG(HdPerm);
        ptSmall32  = (TypPoint32*)PTR_BAG(hdSmall);

        /* we only know that we must raise <perm> to a power = 0 mod 1     */
        ord = INT_TO_HD(1);  pow = INT_TO_HD(0);

        /* loop over all cycles                                            */
        for ( p = 0; p < GET_SIZE_BAG(hdPerm)/sizeof(TypPoint32); p++ ) {

            /* if we haven't looked at this cycle so far                   */
            if ( ptKnown32[p] == 0 ) {

                /* find the length of this cycle                           */
                len = 1;
                for ( q = ptPerm32[p]; q != p; q = ptPerm32[q] ) {
                    len++;  ptKnown32[q] = 1;
                }

                /* compute the gcd with the previously order ord           */
                /* Note that since len is single precision, ord % len is to*/
                gcd = len;  s = HD_TO_INT( ModInt( ord, INT_TO_HD(len) ) );
                while ( s != 0 ) {
                    t = s;  s = gcd % s;  gcd = t;
                }

                /* we must raise the cycle into a power = pow mod gcd      */
                x = HD_TO_INT( ModInt( pow, INT_TO_HD( gcd ) ) );

                /* find the smallest element in the cycle at such a positio*/
                min = GET_SIZE_BAG(hdPerm)/sizeof(TypPoint32)-1;
                n = 0;
                for ( q = p, l = 0; l < len; l++ ) {
                    gcd2 = len;  s = l;
                    while ( s != 0 ) { t = s; s = gcd2 % s; gcd2 = t; }
                    if ( l % gcd == x && gcd2 == 1 && q <= min ) {
                        min = q;
                        n = l;
                    }
                    q = ptPerm32[q];
                }

                /* raise the cycle to that power and put it in the result  */
                ptSmall32[p] = min;
                for ( q = ptPerm32[p]; q != p; q = ptPerm32[q] ) {
                    min = ptPerm32[min];  ptSmall32[q] = min;
                }

                /* compute the new order and the new power                 */
                while ( HD_TO_INT( ModInt( pow, INT_TO_HD(len) ) ) != n )
                    pow = SumInt( pow, ord );
                ord = ProdInt( ord, INT_TO_HD( len / gcd ) );

            }

        }

        /* clear the buffer bag again                                      */
        for ( p = 0; p < GET_SIZE_BAG(hdPerm)/sizeof(TypPoint32); p++ )
            ptKnown32[p] = 0;

    }

    /* return the smallest generator                                       */
    return hdSmall;
}


/****************************************************************************
**
*F  OnTuplesPerm( <hdTup>, <hdPrm> )  . . . .  operations on tuples of points
**
**  'OnTuplesPerm'  returns  the  image  of  the  tuple  <hdTup>   under  the
**  permutation <hdPrm>.  It is called from 'FunOnTuples'.
*/
Bag       OnTuplesPerm (Bag hdTup, Bag hdPrm)
{
    Bag           hdRes;          /* handle of the image, result     */
    Bag           * ptRes;        /* pointer to the result           */
    Bag           * ptTup;        /* pointer to the tuple            */
    TypPoint16          * ptPrm16;      /* pointer to the permutation      */
    TypPoint32          * ptPrm32;      /* pointer to the permutation      */
    Bag           hdTmp;          /* temporary handle                */
    UInt       lmp;            /* largest moved point             */
    UInt       i, k;           /* loop variables                  */

    /* make a bag for the result and initialize pointers                   */
    hdRes = NewBag( T_LIST, SIZE_HD + LEN_LIST(hdTup)*SIZE_HD );
    SET_BAG(hdRes, 0,  PTR_BAG(hdTup)[0] );

    /* handle small permutations                                           */
    if ( GET_TYPE_BAG(hdPrm) == T_PERM16 ) {

        /* get the pointer                                                 */
        ptTup = PTR_BAG(hdTup) + LEN_LIST(hdTup);
        ptRes = PTR_BAG(hdRes) + LEN_LIST(hdTup);
        ptPrm16 = (TypPoint16*)PTR_BAG(hdPrm);
        lmp = GET_SIZE_BAG(hdPrm) / sizeof(TypPoint16);

        /* loop over the entries of the tuple                              */
        for ( i = LEN_LIST(hdTup); 1 <= i; i--, ptTup--, ptRes-- ) {
            if ( GET_TYPE_BAG( *ptTup ) == T_INT ) {
                k = HD_TO_INT( *ptTup );
                if ( k <= 0 )
                    hdTmp = Error("Perm Op: point must be positive (%d)",k,0);
                else if ( k <= lmp )
                    hdTmp = INT_TO_HD( ptPrm16[k-1] + 1 );
                else
                    hdTmp = INT_TO_HD( k );
                *ptRes = hdTmp;
            }
            else {
                hdTmp = POW( *ptTup, hdPrm );
                ptTup = PTR_BAG(hdTup) + i;
                ptRes = PTR_BAG(hdRes) + i;
                ptPrm16 = (TypPoint16*)PTR_BAG(hdPrm);
                *ptRes = hdTmp;
            }
        }

    }

    /* handle large permutations                                           */
    else {

        /* get the pointer                                                 */
        ptTup = PTR_BAG(hdTup) + LEN_LIST(hdTup);
        ptRes = PTR_BAG(hdRes) + LEN_LIST(hdTup);
        ptPrm32 = (TypPoint32*)PTR_BAG(hdPrm);
        lmp = GET_SIZE_BAG(hdPrm) / sizeof(TypPoint32);

        /* loop over the entries of the tuple                              */
        for ( i = LEN_LIST(hdTup); 1 <= i; i--, ptTup--, ptRes-- ) {
            if ( GET_TYPE_BAG( *ptTup ) == T_INT ) {
                k = HD_TO_INT( *ptTup );
                if ( k <= 0 )
                    hdTmp = Error("Perm Op: point must be positive (%d)",k,0);
                else if ( k <= lmp )
                    hdTmp = INT_TO_HD( ptPrm32[k-1] + 1 );
                else
                    hdTmp = INT_TO_HD( k );
                *ptRes = hdTmp;
            }
            else {
                hdTmp = POW( *ptTup, hdPrm );
                ptTup = PTR_BAG(hdTup) + i;
                ptRes = PTR_BAG(hdRes) + i;
                ptPrm32 = (TypPoint32*)PTR_BAG(hdPrm);
                *ptRes = hdTmp;
            }
        }

    }

    /* return the result                                                   */
    return hdRes;
}


/****************************************************************************
**
*F  OnSetsPerm( <hdSet>, <hdPrm> ) . . . . . . . operations on sets of points
**
**  'OnSetsPerm'  returns  the  image  of  the  tuple  <hdSet>   under  the
**  permutation <hdPrm>.  It is called from 'FunOnSets'.
*/
Bag       OnSetsPerm (Bag hdSet, Bag hdPrm)
{
    Bag           hdRes;          /* handle of the image, result     */
    Bag           * ptRes;        /* pointer to the result           */
    Bag           * ptTup;        /* pointer to the tuple            */
    TypPoint16          * ptPrm16;      /* pointer to the permutation      */
    TypPoint32          * ptPrm32;      /* pointer to the permutation      */
    Bag           hdTmp;          /* temporary handle                */
    UInt       lmp;            /* largest moved point             */
    UInt       isint;          /* <set> only holds integers       */
    UInt       len;            /* logical length of the list      */
    UInt       h;              /* gap width in the shellsort      */
    UInt       i, k;           /* loop variables                  */

    /* make a bag for the result and initialize pointers                   */
    hdRes = NewBag( T_LIST, SIZE_HD + LEN_LIST(hdSet)*SIZE_HD );
    SET_BAG(hdRes, 0,  PTR_BAG(hdSet)[0] );

    /* handle small permutations                                           */
    if ( GET_TYPE_BAG(hdPrm) == T_PERM16 ) {

        /* get the pointer                                                 */
        ptTup = PTR_BAG(hdSet) + LEN_LIST(hdSet);
        ptRes = PTR_BAG(hdRes) + LEN_LIST(hdSet);
        ptPrm16 = (TypPoint16*)PTR_BAG(hdPrm);
        lmp = GET_SIZE_BAG(hdPrm) / sizeof(TypPoint16);

        /* loop over the entries of the tuple                              */
        isint = 1;
        for ( i = LEN_LIST(hdSet); 1 <= i; i--, ptTup--, ptRes-- ) {
            if ( GET_TYPE_BAG( *ptTup ) == T_INT ) {
                k = HD_TO_INT( *ptTup );
                if ( k <= 0 )
                    hdTmp = Error("Perm Op: point must be positive (%d)",k,0);
                else if ( k <= lmp )
                    hdTmp = INT_TO_HD( ptPrm16[k-1] + 1 );
                else
                    hdTmp = INT_TO_HD( k );
                *ptRes = hdTmp;
            }
            else {
                isint = 0;
                hdTmp = POW( *ptTup, hdPrm );
                ptTup = PTR_BAG(hdSet) + i;
                ptRes = PTR_BAG(hdRes) + i;
                ptPrm16 = (TypPoint16*)PTR_BAG(hdPrm);
                *ptRes = hdTmp;
            }
        }

    }

    /* handle large permutations                                           */
    else {

        /* get the pointer                                                 */
        ptTup = PTR_BAG(hdSet) + LEN_LIST(hdSet);
        ptRes = PTR_BAG(hdRes) + LEN_LIST(hdSet);
        ptPrm32 = (TypPoint32*)PTR_BAG(hdPrm);
        lmp = GET_SIZE_BAG(hdPrm) / sizeof(TypPoint32);

        /* loop over the entries of the tuple                              */
        isint = 1;
        for ( i = LEN_LIST(hdSet); 1 <= i; i--, ptTup--, ptRes-- ) {
            if ( GET_TYPE_BAG( *ptTup ) == T_INT ) {
                k = HD_TO_INT( *ptTup );
                if ( k <= 0 )
                    hdTmp = Error("Perm Op: point must be positive (%d)",k,0);
                else if ( k <= lmp )
                    hdTmp = INT_TO_HD( ptPrm32[k-1] + 1 );
                else
                    hdTmp = INT_TO_HD( k );
                *ptRes = hdTmp;
            }
            else {
                isint = 0;
                hdTmp = POW( *ptTup, hdPrm );
                ptTup = PTR_BAG(hdSet) + i;
                ptRes = PTR_BAG(hdRes) + i;
                ptPrm32 = (TypPoint32*)PTR_BAG(hdPrm);
                *ptRes = hdTmp;
            }
        }

    }

    /* special case if the result only holds integers                      */
    if ( isint ) {

        /* sort the set with a shellsort                                   */
        len = LEN_LIST(hdRes);
        h = 1;  while ( 9*h + 4 < len )  h = 3*h + 1;
        while ( 0 < h ) {
            for ( i = h+1; i <= len; i++ ) {
                hdTmp = PTR_BAG(hdRes)[i];  k = i;
                while ( h < k && ((Int)hdTmp < (Int)(PTR_BAG(hdRes)[k-h])) ) {
                    SET_BAG(hdRes, k,  PTR_BAG(hdRes)[k-h] );
                    k -= h;
                }
                SET_BAG(hdRes, k,  hdTmp );
            }
            h = h / 3;
        }
	Retype( hdRes, T_SET );
    }

    /* general case                                                        */
    else {

        /* sort the set with a shellsort                                   */
        len = LEN_LIST(hdRes);
        h = 1;  while ( 9*h + 4 < len )  h = 3*h + 1;
        while ( 0 < h ) {
            for ( i = h+1; i <= len; i++ ) {
                hdTmp = PTR_BAG(hdRes)[i];  k = i;
                while ( h < k && LT( hdTmp, PTR_BAG(hdRes)[k-h] ) == HdTrue ) {
                    SET_BAG(hdRes, k,  PTR_BAG(hdRes)[k-h] );
                    k -= h;
                }
                SET_BAG(hdRes, k,  hdTmp );
            }
            h = h / 3;
        }

        /* remove duplicates, shrink bag if possible                       */
        if ( 0 < len ) {
            hdTmp = PTR_BAG(hdRes)[1];  k = 1;
            for ( i = 2; i <= len; i++ ) {
                if ( EQ( hdTmp, PTR_BAG(hdRes)[i] ) != HdTrue ) {
                    k++;
                    hdTmp = PTR_BAG(hdRes)[i];
                    SET_BAG(hdRes, k,  hdTmp );
                }
            }
            if ( k < len ) {
                Resize( hdRes, SIZE_HD+k*SIZE_HD );
                SET_BAG(hdRes, 0,  INT_TO_HD(k) );
            }
        }

    }

    /* return the result                                                   */
    return hdRes;
}


/****************************************************************************
**
*F  InitPermutat()  . . . . . . . . . . . initializes the permutation package
**
**  Is  called  during  the  initialization  to  initialize  the  permutation
**  package.
*/
void            InitPermutat (void)
{
    /* install the evaluation and printing functions                       */
    InstEvFunc( T_PERM16,   EvPerm     );
    InstEvFunc( T_PERM32,   EvPerm     );
    InstEvFunc( T_MAKEPERM, EvMakeperm );
    InstPrFunc( T_PERM16,   PrPermP    );
    InstPrFunc( T_PERM32,   PrPermQ    );
    InstPrFunc( T_MAKEPERM, PrMakeperm );

    /* install the binary operations                                       */
    TabProd[ T_PERM16 ][ T_PERM16 ] = ProdPP;
    TabProd[ T_PERM16 ][ T_PERM32 ] = ProdPQ;
    TabProd[ T_PERM32 ][ T_PERM16 ] = ProdQP;
    TabProd[ T_PERM32 ][ T_PERM32 ] = ProdQQ;
    TabQuo[  T_PERM16 ][ T_PERM16 ] = QuoPP;
    TabQuo[  T_PERM16 ][ T_PERM32 ] = QuoPQ;
    TabQuo[  T_PERM32 ][ T_PERM16 ] = QuoQP;
    TabQuo[  T_PERM32 ][ T_PERM32 ] = QuoQQ;
    TabMod[  T_PERM16 ][ T_PERM16 ] = ModPP;
    TabMod[  T_PERM16 ][ T_PERM32 ] = ModPQ;
    TabMod[  T_PERM32 ][ T_PERM16 ] = ModQP;
    TabMod[  T_PERM32 ][ T_PERM32 ] = ModQQ;
    TabPow[  T_PERM16 ][ T_INT    ] = PowPI;
    TabPow[  T_PERM16 ][ T_INTPOS ] = PowPI;
    TabPow[  T_PERM16 ][ T_INTNEG ] = PowPI;
    TabPow[  T_PERM32 ][ T_INT    ] = PowQI;
    TabPow[  T_PERM32 ][ T_INTPOS ] = PowQI;
    TabPow[  T_PERM32 ][ T_INTNEG ] = PowQI;
    TabPow[  T_INT    ][ T_PERM16 ] = PowIP;
    TabPow[  T_INTPOS ][ T_PERM16 ] = PowIP;
    TabPow[  T_INT    ][ T_PERM32 ] = PowIQ;
    TabPow[  T_INTPOS ][ T_PERM32 ] = PowIQ;
    TabQuo[  T_INT    ][ T_PERM16 ] = QuoIP;
    TabQuo[  T_INTPOS ][ T_PERM16 ] = QuoIP;
    TabQuo[  T_INT    ][ T_PERM32 ] = QuoIQ;
    TabQuo[  T_INTPOS ][ T_PERM32 ] = QuoIQ;
    TabPow[  T_PERM16 ][ T_PERM16 ] = PowPP;
    TabPow[  T_PERM16 ][ T_PERM32 ] = PowPQ;
    TabPow[  T_PERM32 ][ T_PERM16 ] = PowQP;
    TabPow[  T_PERM32 ][ T_PERM32 ] = PowQQ;
    TabComm[ T_PERM16 ][ T_PERM16 ] = CommPP;
    TabComm[ T_PERM16 ][ T_PERM32 ] = CommPQ;
    TabComm[ T_PERM32 ][ T_PERM16 ] = CommQP;
    TabComm[ T_PERM32 ][ T_PERM32 ] = CommQQ;
    TabEq[   T_PERM16 ][ T_PERM16 ] = EqPP;
    TabEq[   T_PERM16 ][ T_PERM32 ] = EqPQ;
    TabEq[   T_PERM32 ][ T_PERM16 ] = EqQP;
    TabEq[   T_PERM32 ][ T_PERM32 ] = EqQQ;
    TabLt[   T_PERM16 ][ T_PERM16 ] = LtPP;
    TabLt[   T_PERM16 ][ T_PERM32 ] = LtPQ;
    TabLt[   T_PERM32 ][ T_PERM16 ] = LtQP;
    TabLt[   T_PERM32 ][ T_PERM32 ] = LtQQ;

    /* install the internal functions                                      */
    InstIntFunc( "IsPerm",                FunIsPerm                );
    InstIntFunc( "PermList",              FunPermList              );
    InstIntFunc( "LargestMovedPointPerm", FunLargestMovedPointPerm );
    InstIntFunc( "CycleLengthPermInt",    FunCycleLengthPermInt    );
    InstIntFunc( "CyclePermInt",          FunCyclePermInt          );
    /*N  13-Jan-91 martin should add 'CycleLengthsPerm', 'CyclesPerm'      */
    /*N InstIntFunc( "CycleLengthsPerm",      FunCycleLengthsPerm );       */
    /*N InstIntFunc( "CyclesPerm",            FunCyclesPerm       );       */
    InstIntFunc( "OrderPerm",             FunOrderPerm             );
    InstIntFunc( "SignPerm",              FunSignPerm              );
    InstIntFunc( "SmallestGeneratorPerm", FunSmallestGeneratorPerm );

    /* make the buffer bag                                                 */
    HdPerm = NewBag( T_PERM16, (UInt)1000*sizeof(TypPoint16) );
}
