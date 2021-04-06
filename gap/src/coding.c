/****************************************************************************
**
*A  coding.c                    GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file contains support functions for coding theory.
**
*/

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "scanner.h"             /* reading of tokens and printing  */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */
#include        "integer4.h" 

#include        "list.h"                /* generic list package            */
#include        "plist.h"               /* 'LEN_PLIST', 'SET_LEN_PLIST',.. */
#include        "blister.h"             /* 'LEN_BLIST', 'SET_LEN_BLIST',.. */

#include        "finfield.h"            /* 'TypFFE', 'SUM_FF', 'PROD_FF'.. */
#include        "vecffe.h"              /* 'CharVecFFE', 'DegreeVecFFE',.. */

#include        "coding.h"              /* declaration part of the package */


/****************************************************************************
**
*F  RootPrimePower(<q>) . . . . . . return the smallest root of a prime power
**
**  'RootPrimePower' returns the smallest  root  of the positive  prime power
**  <q>.  If <q> is not a positive prime power, 'RootPrimePower' returns 0.
*/
UInt   RootPrimePower (Int q)
{
    UInt       p;
    if ( q < 2 )  return 0;
    if ( q % 2 == 0 )  p = 2;
    else for ( p = 3; q % p != 0; p += 2 ) ;
    while ( q % p == 0 )  q /= p;
    if ( q != 1 )  return 0;
    return p;
}


/****************************************************************************
**
*F  ConvMatFFE(<mat>,<q>) . . . . .  convert a matrix into a particular field
**
**  'ConvMatFFE' converts the finite field matrix <mat> into 'GF(<q>)', i.e.,
**  all rows  will be finite field  vectors represented over  this field.  If
**  <mat> is not a finite  field matrix,  or if  not all  its entries lie  in
**  'GF(<q>)', 'ConvMatFFE' return 0, otherwise 1.
*/
UInt   ConvMatFFE (Bag mat, UInt q)
{
    UInt       i;              /* loop variable                   */

    /* check that <mat> is a matrix                                        */
    if ( ! IsXTypeMatFFE(mat) )
        return 0;

    /* force each row into GF(<q>)                                         */
    for ( i = 1; i <= LEN_LIST(mat); i++ )
        if ( ! ConvVecFFE( ELM_LIST( mat, i ), q ) )
            return 0;

    /* indicate success                                                    */
    return 1;
}


/****************************************************************************
**
*F  ConvVecFFE(<vec>,<q>) . . . . .  convert a vector into a particular field
**
**  'ConvVecFFE' converts the finite field vector <vec> into 'GF(<q>)', i.e.,
**  so that it is  represented over  this field.  If  <vec>  is not a  finite
**  field vector, or  if not all its entries  lie in  'GF(<q>)', 'ConvVecFFE'
**  return 0, otherwise 1.
*/
UInt   ConvVecFFE (Bag vec, UInt q)
{
    UInt       p;              /* characteristic                  */
    UInt       d;              /* degree of <vec>                 */
    UInt       q1;             /* size of field of <vec>          */
    TypFFE *            v;              /* pointer into <vec>              */
    TypFFE              t;              /* temporary                       */
    UInt       i;              /* loop variable                   */

    /* check that <vec> is a vector                                        */
    if ( ! IsXTypeVecFFE(vec) )
        return 0;

    /* return immediately if <vec> is already written over GF(<q>)         */
    if ( SIZE_FF( FLD_VECFFE(vec) ) == q )
        return 1;

    /* check that the current field of <vec> is a subfield of GF(<q>)      */
    p = CharVecFFE(vec);
    d = DegreeVecFFE(vec);
    q1 = 1;
    for ( i = 1; i <= d; i++ )
        q1 *= p;
    if ( q % p != 0 || (q-1) % (q1-1) != 0 )
        return 0;

    /* change the field to GF(<q>)                                         */
    q1 = SIZE_FF( FLD_VECFFE(vec) );
    SET_FLD_VECFFE( vec, FLD_FFE( RootFiniteField( q ) ) );

    /* rewrite the vector                                                  */
    v = (TypFFE*)(PTR_BAG(vec) + 1);
    for ( i = 1; i <= LEN_VECFFE(vec); i++, v++ ) {
        t = *v;
        if ( t != 0 )
            *v = (t-1) * (q-1) / (q1-1) + 1;
    }

    /* indicate success                                                    */
    return 1;
}


/****************************************************************************
**
*F  BlistsMatFF2(<mat>) . . . . . make a blist matrix for a matrix over GF(2)
**
**  'BlistsMatFF2' returns a new  list <mat2>, whose  rows are boolean  list,
**  such that '<mat2>[<i>][<k>] := <mat>[<i>][<k>] =  Z(2)', where <mat> is a
**  matrix over GF(2).
*/
Bag       BlistsMatFF2 (Bag mat)
{
    Bag           mat2;           /* new matrix, result              */
    Bag           row2;           /* row of <mat2>                   */
    UInt       i;              /* loop variable                   */

    /* create the new matrix                                               */
    mat2 = NewBag( T_LIST, SIZE_PLEN_PLIST( LEN_LIST( mat ) ) );
    SET_LEN_PLIST( mat2, LEN_LIST( mat ) );

    /* convert each row into a blist                                       */
    for ( i = 1; i <= LEN_LIST( mat ); i++ ) {
        row2 = BlistVecFF2( ELM_LIST( mat, i ) );
        SET_ELM_PLIST( mat2, i, row2 );
    }

    /* return the matrix                                                   */
    return mat2;
}


/****************************************************************************
**
*F  BlistVecFF2(<vec>)  . . . . . . . .  make a blist for a vector over GF(2)
**
**  'BlistVecFF2' returns a  new boolean list  <vec2>, such that '<vec2>[<i>]
**  := <vec>[<i>] = Z(2)', where <vec> is a vector over GF(2).
*/
Bag       BlistVecFF2 (Bag vec)
{
    Bag           vec2;           /* boolean list, result            */
    UInt       len;            /* length of vector                */
    TypFFE *            v;              /* pointer into <vec>              */
    UInt *     v2;             /* pointer into <vec2>             */
    UInt       blk;            /* one block of <vec2>             */
    UInt       bit;            /* one bit of <blk>                */
    UInt       i;              /* loop variable                   */

    /* create the boolean list                                             */
    len = LEN_VECFFE(vec);
    vec2 = NewBag( T_BLIST, SIZE_PLEN_BLIST( len ) );
    SET_LEN_BLIST( vec2, len );

    /* loop over the entries of the vector and set the corresponding bits  */
    v = (TypFFE*)(PTR_BAG(vec) + 1);
    v2 = (UInt *)(PTR_BAG(vec2) + 1);
    blk = 0;
    bit = 1;
    for ( i = 1; i <= len; i++, v++ ) {
        if ( *v != 0 )
            blk |= bit;
        bit <<= 1;
        if ( bit == 0 || i == len ) {
            *v2++ = blk;
            blk = 0;
            bit = 1;
        }
    }

    /* return the boolean list                                             */
    return vec2;
}


/****************************************************************************
**
*F  VecFF2Blist(<vec>)  . . . . . . . .  make a vector over GF(2) for a blist
**
**  'VecFF2Blist' returns a new vector over GF(2) <vec2>, such that
**  'if <vec>[<i>] then <vec2>[<i>] := Z(2); else <vec2>[<i>] := 0*Z(2); fi;'
**  where <vec> is a boolean list.
*/
Bag       VecFF2Blist (Bag vec)
{
    Bag           vec2;           /* vector over GF(2), result       */
    UInt       len;            /* length of blist                 */
    TypFFE *            v2;             /* pointer into <vec2>             */
    UInt       i;              /* loop variable                   */

    /* create the vector over GF(2)                                        */
    len = LEN_BLIST( vec );
    vec2 = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( len ) );
    SET_FLD_FFE( vec2, FLD_FFE( RootFiniteField( 2 ) ) );

    /* loop over the entries of the boolean list and assign accordingly    */
    v2 = (TypFFE*)(PTR_BAG(vec2) + 1);
    for ( i = 1; i <= len; i++, v2++ ) {
        if ( ELM_BLIST( vec, i ) == HdTrue )
            *v2 = 1;
    }

    /* return the vector over GF(2)                                        */
    return vec2;
}


/****************************************************************************
**
*F  WeightVecFFE(<vec>) . . . . . . . . . . . weight of a finite field vector
**
**  'WeightVecFFE' returns the weight of the finite field vector <vec>, i.e.,
**  the number of nonzero entries.
*/
UInt   WeightVecFFE (Bag vec)
{
    UInt       n;              /* weight, result                  */
    TypFFE *            v;              /* pointer into <vec>              */
    UInt       i;              /* loop variable                   */

    /* count the number of nonzero entries                                 */
    n = 0;
    v = (TypFFE*)(PTR_BAG(vec) + 1);
    for ( i = 1; i <= LEN_VECFFE( vec ); i++, v++ )
        n += (*v != 0);

    /* return the weight                                                   */
    return n;
}


/****************************************************************************
**
*F  FunDistanceVecFFE(<hdCall>) . . . . . compute distance between to vectors
**
**  'FunDistanceVecFFE' implements the internal function 'DistanceVecFFE'.
**
**  'DistanceVecFFE(<vec1>,<vec2>)'
**
**  'DistanceVecFFE' returns the distance  between the two vectors <vec1> and
**  <vec2>, which must have the same length  and whose elements must lie in a
**  common field.   The  distance is the  number  of places  where <vec1> and
**  <vec2> differ.
*/
Bag       DistanceVecFFE (Bag vec1, Bag vec2)
{
    UInt       n;              /* distance, result                */
    UInt       p;              /* characteristic                  */
    UInt       d;              /* degree of common field          */
    UInt       q;              /* size of common field            */
    UInt       d1;             /* degree of field of <vec1>       */
    UInt       d2;             /* degree of field of <vec2>       */
    TypFFE *            v1;             /* pointer into <vec1>             */
    TypFFE *            v2;             /* pointer into <vec2>             */
    UInt       k;              /* loop variable                   */

    /* check the arguments                                                 */
    if ( ! IsXTypeVecFFE(vec1) )
        return Error("DistancesVecFFE: %s",
            (Int)"<vec1> must be a finite field vector",0);
    if ( ! IsXTypeVecFFE(vec2) )
        return Error("DistancesVecFFE: %s",
            (Int)"<vec2> must be a finite field vector",0);
    if ( LEN_VECFFE(vec1) != LEN_VECFFE(vec2) )
        return Error("DistancesVecFFE: %s",
            (Int)"<vec1> and <vec2> must have the same length",0);

    /* convert the two vectors into a common field                         */
    if ( FLD_VECFFE(vec1) != FLD_VECFFE(vec2) ) {
        p = CharVecFFE( vec1 );
        if ( SIZE_FF( FLD_VECFFE(vec2) ) % p != 0 )
            return Error("DistancesVecFFE: %s",
                (Int)"<vec1> and <vec2> must lie in a common field",0);
        d1 = DegreeVecFFE( vec1 );
        d2 = DegreeVecFFE( vec2 );
        for ( d = 1, q = p; d % d1 != 0 || d % d2 != 0; d++ )  q *= p;
        if ( (  2 <= p && 17 <= d) || (  3 <= p && 11 <= d)
          || (  5 <= p &&  7 <= d) || (  7 <= p &&  6 <= d)
          || ( 11 <= p &&  5 <= d) || ( 17 <= p &&  4 <= d)
          || ( 41 <= p &&  3 <= d) || (257 <= p &&  2 <= d) )
            return Error("DistancesVecFFE: %s",
                (Int)"<vec1> and <vec2> must lie in a common field",0);
        ConvVecFFE( vec1, q );
        ConvVecFFE( vec2, q );
    }

    /* compute the distance                                                */
    n = 0;
    v1 = (TypFFE*)(PTR_BAG(vec1) + 1);
    v2 = (TypFFE*)(PTR_BAG(vec2) + 1);
    for ( k = 1; k <= LEN_VECFFE(vec1); k++, v1++, v2++ )
        n += (*v1 != *v2);

    /* return the distance                                                 */
    return INT_TO_HD(n);
}

Bag       FunDistanceVecFFE (Bag hdCall)
{
    /* check the number of arguments                                       */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error("number of arguments must be 2 not %d",
                     (Int)(GET_SIZE_BAG(hdCall)/SIZE_HD - 1), 0 );

    /* call the real internal function                                     */
    return DistanceVecFFE(
                        EVAL( PTR_BAG(hdCall)[1] ), EVAL( PTR_BAG(hdCall)[2] ) );
}


/****************************************************************************
**
*F  FunDistancesDistributionVecFFEsVecFFE(<hdCall>) .  distances distribution
*F  . . . . . . . . . . . . . . . . . . . . . . . . . .  for a nonlinear code
**
**  'FunDistancesDistributionVecFFEsVecFFE' implements the  internal function
**  'DistancesDistributionVecFFEsVecFFE'.
**
**  'DistancesDistributionVecFFEsVecFFE(<vecs>,<vec>)'
**
**  'DistancesDistributionVecFFEsVecFFE'  returns  the distances distribution
**  of the vector <vec> to the vectors in the  list <vecs>.  All vectors must
**  have the same length, and all  elements must lie in a  common field.  The
**  distances distribution is  a list <d>  of length 'Length(<vec>)+1',  such
**  that the value '<d>[<i>]'  is the number of  vectors in <vecs>  that have
**  distance '<i>+1' to <vec>.
*/
Bag       DistancesDistributionVecFFEsVecFFE (Bag vecs, Bag vec)
{
    Bag           dst;            /* distances distribution, result  */
    UInt       n;              /* distance of one row of <vecs>   */
    UInt       p;              /* characteristic                  */
    UInt       d;              /* degree of common field          */
    UInt       q;              /* size of common field            */
    UInt       d1;             /* degree of field of <vecs>       */
    UInt       d2;             /* degree of field of <vec>        */
    TypFFE *            v1;             /* pointer into one row of <vecs>  */
    TypFFE *            v2;             /* pointer into <vec>              */
    Bag           cnt;            /* current count                   */
    UInt       i, k;           /* loop variable                   */

    /* check the arguments                                                 */
    if ( ! IsXTypeMatFFE(vecs) )
        return Error("DistancesDistributionVecFFEsVecFFE: %s",
            (Int)"<vecs> must be a finite field matrix",0);
    if ( ! IsXTypeVecFFE(vec) )
        return Error("DistancesDistributionVecFFEsVecFFE: %s",
            (Int)"<vec> must be a finite field vector",0);
    if ( LEN_VECFFE( ELM_LIST(vecs,1) ) != LEN_VECFFE(vec) )
        return Error("DistancesDistributionVecFFEsVecFFE: %s",
            (Int)"<vecs>[1] and <vec> must have the same length",0);

    /* convert the two vectors into a common field                         */
    if ( FLD_VECFFE( ELM_LIST(vecs,1) ) != FLD_VECFFE(vec) ) {
        p = CharVecFFE( ELM_LIST(vecs,1) );
        if ( SIZE_FF( FLD_VECFFE(vec) ) % p != 0 )
            return Error("DistancesDistributionVecFFEsVecFFE: %s",
                (Int)"<vecs> and <vec> must lie in a common field",0);
        d1 = DegreeMatFFE( vecs );
        d2 = DegreeVecFFE( vec );
        for ( d = 1, q = p; d % d1 != 0 || d % d2 != 0; d++ )  q *= p;
        if ( (  2 <= p && 17 <= d) || (  3 <= p && 11 <= d)
          || (  5 <= p &&  7 <= d) || (  7 <= p &&  6 <= d)
          || ( 11 <= p &&  5 <= d) || ( 17 <= p &&  4 <= d)
          || ( 41 <= p &&  3 <= d) || (257 <= p &&  2 <= d) )
            return Error("DistancesDistributionVecFFEsVecFFE: %s",
                (Int)"<vecs> and <vec> must lie in a common field",0);
        ConvMatFFE( vecs, q );
        ConvVecFFE( vec, q );
    }

    /* create the distance distribution list                               */
    dst = NewBag( T_LIST, SIZE_PLEN_PLIST( LEN_VECFFE(vec) + 1 ) );
    SET_LEN_PLIST( dst, LEN_VECFFE(vec) + 1 );
    for ( i = 1; i <= LEN_VECFFE(vec) + 1; i++ )
        SET_ELM_PLIST( dst, i, INT_TO_HD(0) );

    /* compute the distances distribution                                  */
    for ( i = 1; i <= LEN_LIST(vecs); i++ ) {

        /* compute the distance of <vecs>[<i>] to <vec>                    */
        n = 0;
        v1 = (TypFFE*)(PTR_BAG( ELM_LIST(vecs,i) ) + 1);
        v2 = (TypFFE*)(PTR_BAG(vec) + 1);
        for ( k = 1; k <= LEN_VECFFE(vec); k++, v1++, v2++ )
            n += (*v1 != *v2);

        /* enter the distance in the distribution                          */
        /* note that this is bound by Length(<vecs>) so it cannot overflow */
        cnt = ELM_PLIST( dst, n+1 );
        cnt = (Bag)((Int)cnt + (Int)INT_TO_HD(1) - T_INT);
        SET_ELM_PLIST( dst, n+1, cnt );

    }

    /* return the distances distribution                                   */
    return dst;
}

Bag       FunDistancesDistributionVecFFEsVecFFE (Bag hdCall)
{
    /* check the number of arguments                                       */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error("number of arguments must be 2 not %d",
                     (Int)(GET_SIZE_BAG(hdCall)/SIZE_HD - 1), 0 );

    /* call the real internal function                                     */
    return DistancesDistributionVecFFEsVecFFE(
                        EVAL( PTR_BAG(hdCall)[1] ), EVAL( PTR_BAG(hdCall)[2] ) );
}


/****************************************************************************
**
*F  FunDistancesDistributionMatFFEVecFFE(<hdCall>)  .  distances distribution
*F  . . . . . . . . . . . . . . . . . . . . . . . . . . . . for a linear code
**
**  'FunDistancesDistributionMatFFEVecFFE'   implements the internal function
**  'DistancesDistributionMatFFEVecFFE'.
**
**  'DistancesDistributionMatFFEVecFFE(<mat>,<q>,<vec>)'
**
**  'DistancesDistributionMatFFEVecFFE' returns the distances distribution of
**  the vector <vec> to the vectors in the vector space generated by the rows
**  of the matrix <mat>  over the field  GF(<q>).  The length  of the rows of
**  <mat> and the length of <vec> must be equal, and all elements must lie in
**  GF(<q>).  The rows of <mat>  must be linearly independent.  The distances
**  distribution  is a list <d>  of  length 'Length(<vec>)+1', such that  the
**  value '<d>[<i>]' is the  number of vectors  in the vector space generated
**  by the rows of <mat> that have distance '<i>+1' to <vec>.
*/
void            DDM2V2 (Bag mat, Bag vec, Bag sum, UInt l, Bag dst)
{
    UInt       nrb;            /* number of blocks of <vec>       */
    UInt *     r;              /* pointer into row of <mat>       */
    UInt *     s;              /* pointer into <sum>              */
    UInt *     v;              /* pointer into <vec>              */
    UInt       n;              /* distance of <sum>               */
    UInt       b;              /* block of <vec> - <sum>          */
    Bag           cnt;            /* current count                   */
    UInt       k;              /* loop variable                   */

    /* initialize                                                          */
    nrb = (LEN_BLIST(vec) + BIPEB-1) / BIPEB;

    /* if we only have to consider one more summand                        */
    if ( l == LEN_LIST(mat)-1 ) {

        /* compute the distance of <sum>                                   */
        n = 0;
        s = (UInt *)(PTR_BAG( sum ) + 1);
        v = (UInt *)(PTR_BAG( vec ) + 1);
        for ( k = 1; k <= nrb; k++, s++, v++ ) {
            b = *s ^ *v;
            b = (b & 0x55555555) + ((b >> 1) & 0x55555555);
            b = (b & 0x33333333) + ((b >> 2) & 0x33333333);
            b = (b + (b >>  4)) & 0x0f0f0f0f;
            b = (b + (b >>  8));
            b = (b + (b >> 16)) & 0x000000ff;
            n += b;
        }

        /* enter the distance in the distribution                          */
        cnt = ELM_PLIST( dst, n+1 );
        cnt = (Bag)((Int)cnt + (Int)INT_TO_HD(1) - T_INT);
        if ( ((Int)cnt & 3) != T_INT
          || (((Int)cnt << 1) >> 1) != (Int)cnt )
            cnt = SumInt( cnt, INT_TO_HD(1) );
        SET_ELM_PLIST( dst, n+1, cnt );

        /* compute the distance of <sum> + <mat>[<l>+1]                    */
        n = 0;
        r = (UInt *)(PTR_BAG( ELM_LIST( mat, l+1 ) ) + 1);
        s = (UInt *)(PTR_BAG( sum ) + 1);
        v = (UInt *)(PTR_BAG( vec ) + 1);
        for ( k = 1; k <= nrb; k++, r++, s++, v++ ) {
            b = (*r ^ *s) ^ *v;
            b = (b & 0x55555555) + ((b >> 1) & 0x55555555);
            b = (b & 0x33333333) + ((b >> 2) & 0x33333333);
            b = (b + (b >>  4)) & 0x0f0f0f0f;
            b = (b + (b >>  8));
            b = (b + (b >> 16)) & 0x000000ff;
            n += b;
        }

        /* enter the distance in the distribution                          */
        cnt = ELM_PLIST( dst, n+1 );
        cnt = (Bag)((Int)cnt + (Int)INT_TO_HD(1) - T_INT);
        if ( ((Int)cnt & 3) != T_INT
          || (((Int)cnt << 1) >> 1) != (Int)cnt )
            cnt = SumInt( cnt, INT_TO_HD(1) );
        SET_ELM_PLIST( dst, n+1, cnt );

    }

    /* otherwise add all further summands and recurse                      */
    else {

        /* first recurse with <sum>                                        */
        DDM2V2( mat, vec, sum, l+1, dst );

        /* add <mat>[<l>+1] to <sum>                                       */
        r = (UInt *)(PTR_BAG( ELM_LIST( mat, l+1 ) ) + 1);
        s = (UInt *)(PTR_BAG( sum ) + 1);
        for ( k = 1; k <= nrb; k++, r++, s++ )
            *s ^= *r;

        /* recurse                                                         */
        DDM2V2( mat, vec, sum, l+1, dst );

        /* subtract <mat>[<l>+1] from <sum> to correct                     */
        r = (UInt *)(PTR_BAG( ELM_LIST( mat, l+1 ) ) + 1);
        s = (UInt *)(PTR_BAG( sum ) + 1);
        for ( k = 1; k <= nrb; k++, r++, s++ )
            *s ^= *r;

    }

}

void            DDMFVF (Bag mat, UInt q, Bag vec, Bag sum, UInt l, UInt lim, Bag dst)
{
    Bag           cnt;            /* current count                   */
    UInt       len;            /* length of <vec>                 */
    TypFFE *            fld;            /* successor table for field       */
    TypFFE *            r;              /* pointer into row of <mat>       */
    TypFFE *            s;              /* pointer into <sum>              */
    TypFFE *            v;              /* pointer into <vec>              */
    TypFFE              z;              /* factor to multiply row with     */
    UInt       n;              /* distance of <sum>               */
    TypFFE              t;              /* temporary                       */
    UInt       k;              /* loop variables                  */

    /* initialize                                                          */
    len = LEN_VECFFE(vec);
    fld = SUCC_FF( FLD_VECFFE(vec) );

    /* if we only have to consider one more summand                        */
    if ( l == LEN_LIST(mat)-1 ) {

        /* compute the distance of <sum>                                   */
        n = 0;
        s = (TypFFE*)(PTR_BAG( sum ) + 1);
        v = (TypFFE*)(PTR_BAG( vec ) + 1);
        for ( k = 1; k <= len; k++, s++, v++ ) {
            n += (*s != *v);
        }

        /* enter the distance in the distribution                          */
        cnt = ELM_LIST( dst, n+1 );
        cnt = (Bag)((Int)cnt + (Int)INT_TO_HD(1) - T_INT);
        if ( ((Int)cnt & 3) != T_INT
          || (((Int)cnt << 1) >> 1) != (Int)cnt )
            cnt = SumInt( cnt, INT_TO_HD(1) );
        SET_ELM_PLIST( dst, n+1, cnt );

        /* loop over all nonzero elements of the field                     */
        for ( z = 1; z <= lim; z++ ) {

            /* compute the distance of <sum> + <z> * <mat>[<l>+1]          */
            n = 0;
            r = (TypFFE*)(PTR_BAG( ELM_LIST( mat, l+1 ) ) + 1);
            s = (TypFFE*)(PTR_BAG( sum ) + 1);
            v = (TypFFE*)(PTR_BAG( vec ) + 1);
            for ( k = 1; k <= len; k++, r++, s++, v++ ) {
                t = PROD_FF( z, *r, fld );
                n += (SUM_FF( *s, t, fld ) != *v);
            }

            /* enter the distance in the distribution                      */
            cnt = ELM_LIST( dst, n+1 );
            cnt = (Bag)((Int)cnt + (Int)INT_TO_HD(1) - T_INT);
            if ( ((Int)cnt & 3) != T_INT
              || (((Int)cnt << 1) >> 1) != (Int)cnt )
                cnt = SumInt( cnt, INT_TO_HD(1) );
            SET_ELM_PLIST( dst, n+1, cnt );

        }

    }

    /* otherwise add all further summands and recurse                      */
    else {

        /* first recurse with <sum>                                        */
        DDMFVF( mat, q, vec, sum, l+1, lim, dst );

        /* loop over all nonzero elements of the field                     */
        for ( z = 1; z <= lim; z++ ) {

            /* add <z> * <mat>[<l>+1] to <sum>                             */
            r = (TypFFE*)(PTR_BAG( ELM_LIST( mat, l+1 ) ) + 1);
            s = (TypFFE*)(PTR_BAG( sum ) + 1);
            for ( k = 1; k <= len; k++, r++, s++ ) {
                t = PROD_FF( z, *r, fld );
                *s = SUM_FF( *s, t, fld );
            }

            /* recurse                                                     */
            DDMFVF( mat, q, vec, sum, l+1, q-1, dst );

            /* subtract <z> * <mat>[<l>+1] from <sum> to correct           */
            z = NEG_FF( z, fld );
            r = (TypFFE*)(PTR_BAG( ELM_LIST( mat, l+1 ) ) + 1);
            s = (TypFFE*)(PTR_BAG( sum ) + 1);
            for ( k = 1; k <= len; k++, r++, s++ ) {
                t = PROD_FF( z, *r, fld );
                *s = SUM_FF( *s, t, fld );
            }
            z = NEG_FF( z, fld );

        }

    }

}

Bag       DistancesDistributionMatFFEVecFFE (Bag mat, Bag q, Bag vec)
{
    Bag           dst;            /* distribution, result            */
    UInt       p;              /* characteristic                  */
    Bag           sum;            /* temporary for the combinations  */
    UInt       i;              /* loop variable                   */

    /* check the arguments                                                 */
    if ( GET_TYPE_BAG(q) != T_INT || ! (p = RootPrimePower( HD_TO_INT(q) )) )
        return Error("DistancesDistributionMatFFEVecFFE: %s",
            (Int)"<q> must be a positive prime power",0);
    if ( ! ConvMatFFE( mat, HD_TO_INT(q) ) )
        return Error("DistancesDistributionMatFFEVecFFE: %s",
            (Int)"<mat> must be a matrix over GF(<q>)",0);
    if ( ! ConvVecFFE( vec, HD_TO_INT(q) ) )
        return Error("DistancesDistributionMatFFEVecFFE: %s",
            (Int)"<vec> must be a vector over GF(<q>)",0);
    if ( LEN_VECFFE( ELM_LIST(mat,1) ) != LEN_VECFFE(vec) )
        return Error("DistancesDistributionMatFFEVecFFE: %s",
            (Int)"<mat>[<1>] and <vec> must have the same length",0);

    /* create the distance distribution list                               */
    dst = NewBag( T_LIST, SIZE_PLEN_PLIST( LEN_VECFFE(vec) + 1 ) );
    SET_LEN_PLIST( dst, LEN_VECFFE(vec) + 1 );
    for ( i = 1; i <= LEN_VECFFE(vec) + 1; i++ )
        SET_ELM_PLIST( dst, i, INT_TO_HD(0) );

    /* if <q> == 2, convert <mat> and <vec> to blists                      */
    if ( HD_TO_INT(q) == 2 ) {
        sum = NewBag( T_BLIST, SIZE_PLEN_BLIST( LEN_VECFFE(vec) ) );
        SET_LEN_BLIST( sum, LEN_VECFFE(vec) );
        DDM2V2( BlistsMatFF2(mat), BlistVecFF2(vec), sum, 0, dst );
    }

    /* if <vec> == 0, we need only enumerate reps for the 1-dim subspaces  */
    else if ( WeightVecFFE( vec ) == 0 ) {
        sum = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( LEN_VECFFE(vec) ) );
        SET_FLD_VECFFE( sum, FLD_VECFFE(vec) );
        DDMFVF( mat, HD_TO_INT(q), vec, sum, 0, 1, dst );
        dst = PROD( dst, INT_TO_HD( HD_TO_INT(q) - 1 ) );
        ASS_LIST( dst, 1, INT_TO_HD(1) );
    }

    /* otherwise, enumerate all vectors                                    */
    else {
        sum = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( LEN_VECFFE(vec) ) );
        SET_FLD_VECFFE( sum, FLD_VECFFE(vec) );
        DDMFVF( mat, HD_TO_INT(q), vec, sum, 0, HD_TO_INT(q)-1, dst );
    }

    /* return the distribution                                             */
    return dst;
}

Bag       FunDistancesDistributionMatFFEVecFFE (Bag hdCall)
{
    /* check the number of arguments                                       */
    if ( GET_SIZE_BAG(hdCall) != 4*SIZE_HD )
        return Error("number of arguments must be 3 not %d",
                     (Int)(GET_SIZE_BAG(hdCall)/SIZE_HD - 1), 0 );

    /* call the real internal function                                     */
    return DistancesDistributionMatFFEVecFFE(
                        EVAL( PTR_BAG(hdCall)[1] ), EVAL( PTR_BAG(hdCall)[2] ),
                        EVAL( PTR_BAG(hdCall)[3] ) );
}


/****************************************************************************
**
*F  FunAClosestVectorCombinationsMatFFEVecFFE(<hdCall>)  . . a closest vector
*F  . . . . . . . .  to a vector from linear combinations of rows of a matrix
**
**  'FunAClosestVectorCombinationsMatFFEVecFFE'   implements     the internal
**  function 'AClosestVectorCombinationsMatFFEVecFFE'.
**
**  'AClosestVectorCombinationsMatFFEVecFFE(<mat>,<q>,<vec>,<l>,<stop>)'
**
**  'AClosestVectorCombinationsMatFFEVecFFE'   returns  a  vector   from  the
**  vectors in  the vector space generated  by the rows  of the  matrix <mat>
**  over  the field GF(<q>) that can  be written as  combinations of <l> rows
**  that is closest to the vector <vec>.  The length of the rows of <mat> and
**  the length of <vec> must be equal, and all  elements must lie in GF(<q>).
**  The rows of <mat> must be linearly independent.  If  it finds a vector of
**  distance  less than or  equal  to  <stop>,  which must  be a  nonnegative
**  integer, then it stops immediately and returns this vector.
*/
void            CVCM2V2 (Bag mat, Bag vec, UInt l, UInt stop, Bag sum, UInt i, UInt *pmin, Bag best)
{
    UInt       nrb;            /* number of blocks of <vec>       */
    UInt *     r;              /* pointer into row of <mat>       */
    UInt *     s;              /* pointer into <sum>              */
    UInt *     v;              /* pointer into <vec>              */
    UInt       n;              /* distance of <sum>               */
    UInt       b;              /* block of <sum> - <vec>          */
    UInt       j, k;           /* loop variables                  */

    /* initialize                                                          */
    nrb = (LEN_BLIST(vec) + BIPEB-1) / BIPEB;

    /* if we only have to add one more vector                              */
    if ( l == 1 ) {

        /* loop over all possible last summands                            */
        for ( j = i+1; j <= LEN_LIST(mat); j++ ) {

            /* compute the sum of <sum> and <mat>[<j>] and its distance    */
            n = 0;
            r = (UInt *)(PTR_BAG( ELM_LIST( mat, j ) ) + 1);
            s = (UInt *)(PTR_BAG( sum ) + 1);
            v = (UInt *)(PTR_BAG( vec ) + 1);
            for ( k = 1; k <= nrb; k++, r++, s++, v++ ) {
                b = (*r ^ *s) ^ *v;
                b = (b & 0x55555555) + ((b >> 1) & 0x55555555);
                b = (b & 0x33333333) + ((b >> 2) & 0x33333333);
                b = (b + (b >>  4)) & 0x0f0f0f0f;
                b = (b + (b >>  8));
                b = (b + (b >> 16)) & 0x000000ff;
                n += b;
            }

            /* if we found a closer vector update the minimum              */
            if ( n < *pmin ) {
                r = (UInt *)(PTR_BAG( ELM_LIST( mat, j ) ) + 1);
                s = (UInt *)(PTR_BAG( sum ) + 1);
                v = (UInt *)(PTR_BAG( best ) + 1);
                for ( k = 1; k <= nrb; k++, r++, s++, v++ )
                    *v = *r ^ *s;
                *pmin = n;
            }

            /* test if we are done                                         */
            if ( *pmin <= stop )  return;

        }

    }

    /* otherwise add all possible further summands and recurse             */
    else {

        /* loop over all possible further summands                         */
        for ( j = i+1; j+(l-1) <= LEN_LIST(mat); j++ ) {

            /* add <mat>[<j>] to <sum>                                     */
            r = (UInt *)(PTR_BAG( ELM_LIST( mat, j ) ) + 1);
            s = (UInt *)(PTR_BAG( sum ) + 1);
            for ( k = 1; k <= nrb; k++, r++, s++ )
                *s ^= *r;

            /* recurse                                                     */
            CVCM2V2( mat, vec, l-1, stop, sum, j, pmin, best );

            /* test if we are done                                         */
            if ( *pmin <= stop )  return;

            /* subtract <mat>[<j>] again from <sum> to correct             */
            r = (UInt *)(PTR_BAG( ELM_LIST( mat, j ) ) + 1);
            s = (UInt *)(PTR_BAG( sum ) + 1);
            for ( k = 1; k <= nrb; k++, r++, s++ )
                *s ^= *r;

        }

    }

}

void            CVCMFVF (Bag mat, UInt q, Bag vec, UInt l, UInt stop, Bag sum, UInt i, UInt lim, UInt *pmin, Bag best)
{
    UInt       len;            /* length of <vec>                 */
    TypFFE *            fld;            /* successor table for field       */
    TypFFE *            r;              /* pointer into row of <mat>       */
    TypFFE *            s;              /* pointer into <sum>              */
    TypFFE *            v;              /* pointer into <vec>              */
    UInt       n;              /* distance of <sum>               */
    TypFFE              z;              /* factor to multiply row with     */
    TypFFE              t;              /* temporary                       */
    UInt       j, k;           /* loop variables                  */

    /* initialize                                                          */
    len = LEN_VECFFE(vec);
    fld = SUCC_FF( FLD_VECFFE(vec) );

    /* if we only have to add one more vector                              */
    if ( l == 1 ) {

        /* loop over all possible last summands                            */
        for ( j = i+1; j <= LEN_LIST(mat); j++ ) {

            /* loop over all nonzero elements of the field                 */
            for ( z = 1; z <= lim; z++ ) {

                /* compute the distance of <sum> + <z> * <mat>[<l>+1]      */
                n = 0;
                r = (TypFFE*)(PTR_BAG( ELM_LIST( mat, j ) ) + 1);
                s = (TypFFE*)(PTR_BAG( sum ) + 1);
                v = (TypFFE*)(PTR_BAG( vec ) + 1);
                for ( k = 1; k <= len; k++, r++, s++, v++ ) {
                    t = PROD_FF( z, *r, fld );
                    n += (SUM_FF( *s, t, fld ) != *v);
                }

                /* if we found a closer vector update the minimum          */
                if ( n < *pmin ) {
                    r = (TypFFE*)(PTR_BAG( ELM_LIST( mat, j ) ) + 1);
                    s = (TypFFE*)(PTR_BAG( sum ) + 1);
                    v = (TypFFE*)(PTR_BAG( best ) + 1);
                    for ( k = 1; k <= len; k++, r++, s++, v++ ) {
                        t = PROD_FF( z, *r, fld );
                        *v = SUM_FF( *s, t, fld );
                    }
                    *pmin = n;
                }

                /* test if we are done                                     */
                if ( *pmin <= stop )  return;

            }

        }

    }

    /* otherwise add all possible further summands and recurse             */
    else {

        /* loop over all possible further summands                         */
        for ( j = i+1; j+(l-1) <= LEN_LIST(mat); j++ ) {

            /* loop over all nonzero elements of the field                 */
            for ( z = 1; z <= lim; z++ ) {

                /* add <z> * <mat>[<l>+1] to <sum>                         */
                r = (TypFFE*)(PTR_BAG( ELM_LIST( mat, j ) ) + 1);
                s = (TypFFE*)(PTR_BAG( sum ) + 1);
                for ( k = 1; k <= len; k++, r++, s++ ) {
                    t = PROD_FF( z, *r, fld );
                    *s = SUM_FF( *s, t, fld );
                }

                /* recurse                                                 */
                CVCMFVF( mat, q, vec, l-1, stop, sum, j, q-1, pmin, best );

                /* test if we are done                                     */
                if ( *pmin <= stop )  return;

                /* subtract <z> * <mat>[<l>+1] from <sum> to correct       */
                z = NEG_FF( z, fld );
                r = (TypFFE*)(PTR_BAG( ELM_LIST( mat, j ) ) + 1);
                s = (TypFFE*)(PTR_BAG( sum ) + 1);
                for ( k = 1; k <= len; k++, r++, s++ ) {
                    t = PROD_FF( z, *r, fld );
                    *s = SUM_FF( *s, t, fld );
                }
                z = NEG_FF( z, fld );

            }

        }

    }

}

Bag       AClosestVectorCombinationsMatFFEVecFFE (Bag mat, Bag q, Bag vec, Bag l, Bag stop)
{
    Bag           best;           /* closest vector, result          */
    UInt       min;            /* minimum distance                */
    UInt       p;              /* characteristic                  */
    Bag           sum;            /* temporary for the combination   */

    /* check the arguments                                                 */
    if ( GET_TYPE_BAG(q) != T_INT || ! (p = RootPrimePower( HD_TO_INT(q) )) )
        return Error("AClosestVectorCombinationsMatFFEVecFFE: %s",
            (Int)"<q> must be a positive prime power",0);
    if ( ! ConvMatFFE( mat, HD_TO_INT(q) ) )
        return Error("AClosestVectorCombinationsMatFFEVecFFE: %s",
            (Int)"<mat> must be a matrix over GF(<q>)",0);
    if ( ! ConvVecFFE( vec, HD_TO_INT(q) ) )
        return Error("AClosestVectorCombinationsMatFFEVecFFE: %s",
            (Int)"<vec> must be a vector over GF(<q>)",0);
    if ( LEN_VECFFE( ELM_LIST(mat,1) ) != LEN_VECFFE(vec) )
        return Error("AClosestVectorCombinationsMatFFEVecFFE: %s",
            (Int)"<mat>[<1>] and <vec> must have the same length",0);
    if ( GET_TYPE_BAG(l) != T_INT || HD_TO_INT(l)<0 || LEN_LIST(mat)<HD_TO_INT(l) )
        return Error("AClosestVectorCombinationsMatFFEVecFFE: %s",
            (Int)"<l> must be an integer between 0 and Length(<mat>)",0);
    if ( GET_TYPE_BAG(stop) != T_INT || HD_TO_INT(stop)<0 )
        return Error("AClosestVectorCombinationsMatFFEVecFFE: %s",
            (Int)"<stop> must be a nonnegative integer",0);

    /* if <l> == 0, return the weight of <vec>                             */
    if ( HD_TO_INT(l) == 0 ) {
        best = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( LEN_VECFFE(vec) ) );
        SET_FLD_VECFFE( best, FLD_VECFFE(vec) );
    }

    /* if <q> == 2, convert <mat> and <vec> to blists                      */
    else if ( HD_TO_INT(q) == 2 ) {
        sum = NewBag( T_BLIST, SIZE_PLEN_BLIST( LEN_LIST(vec) ) );
        SET_LEN_BLIST( sum, LEN_LIST(vec) );
        best = NewBag( T_BLIST, SIZE_PLEN_BLIST( LEN_LIST(vec) ) );
        SET_LEN_BLIST( best, LEN_LIST(vec) );
        min = LEN_LIST(vec) + 1;
        CVCM2V2( BlistsMatFF2(mat), BlistVecFF2(vec),
                 HD_TO_INT(l), HD_TO_INT(stop), sum, 0, &min, best );
        best = VecFF2Blist(best);
    }

    /* if <vec> == 0, we need only enumerate reps for the 1-dim subspaces  */
    else if ( WeightVecFFE( vec ) == 0 ) {
        sum = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( LEN_VECFFE(vec) ) );
        SET_FLD_VECFFE( sum, FLD_VECFFE(vec) );
        best = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( LEN_VECFFE(vec) ) );
        SET_FLD_VECFFE( best, FLD_VECFFE(vec) );
        min = LEN_LIST(vec) + 1;
        CVCMFVF( mat, HD_TO_INT(q), vec, HD_TO_INT(l),
                 HD_TO_INT(stop), sum, 0, 1, &min, best );
    }

    /* otherwise, enumerate all vectors                                    */
    else {
        sum = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( LEN_VECFFE(vec) ) );
        SET_FLD_VECFFE( sum, FLD_VECFFE(vec) );
        best = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( LEN_VECFFE(vec) ) );
        SET_FLD_VECFFE( best, FLD_VECFFE(vec) );
        min = LEN_LIST(vec) + 1;
        CVCMFVF( mat, HD_TO_INT(q), vec, HD_TO_INT(l),
                 HD_TO_INT(stop), sum, 0, HD_TO_INT(q)-1, &min, best );
    }

    /* return the minimum distance                                         */
    return best;
}

Bag       FunAClosestVectorCombinationsMatFFEVecFFE (Bag hdCall)
{
    /* check the number of arguments                                       */
    if ( GET_SIZE_BAG(hdCall) != 6*SIZE_HD )
        return Error("number of arguments must be 5 not %d",
                     (Int)(GET_SIZE_BAG(hdCall)/SIZE_HD - 1), 0 );

    /* call the real internal function                                     */
    return AClosestVectorCombinationsMatFFEVecFFE(
                        EVAL( PTR_BAG(hdCall)[1] ), EVAL( PTR_BAG(hdCall)[2] ),
                        EVAL( PTR_BAG(hdCall)[3] ), EVAL( PTR_BAG(hdCall)[4] ),
                        EVAL( PTR_BAG(hdCall)[5] ) );
}


/****************************************************************************
**
*F  FunCosetLeadersMatFFE(<hdCall>) . . . . . coset leaders for a linear code
**
**  'FunCosetLeadersMatFFE' implements the internal function
**  'CosetLeadersMatFFE'.
**
**  'CosetLeadersMatFFE(<mat>,<q>)'
**
**  'CosetLeadersMatFFE' returns a list  of representatives of minimal weight
**  for the cosets of the vector space generated by the columns of <mat> over
**  the field GF(<q>).  All rows of <mat> must have the  same length, and all
**  elements must lie   in  GF(<q>).  The   rows of  <mat> must be   linearly
**  independent.
*/
UInt   CLM2 (Bag cls, UInt rem, UInt syn, Bag vec, Bag tam, UInt l, UInt i)
{
    UInt       len;            /* length of <vec>                 */
    TypFFE *            v;              /* pointer into <vec>              */
    Bag           cpy;            /* copy of <vec>                   */
    TypFFE *            c;              /* pointer into <cpy>              */
    UInt       n;              /* syndrome index                  */
    UInt       j, k;           /* loop variables                  */

    /* initialize                                                          */
    len = LEN_VECFFE(vec);

    /* if we have only one more 1 to add                                   */
    if ( l == 1 ) {

        /* loop over all possible position for this 1                      */
        for ( j = i+1; j <= len; j++ ) {

            /* set the 1 and compute the syndrome index                    */
            ((TypFFE*)(PTR_BAG(vec) + 1))[j-1] = 1;
            n = syn ^ (UInt)PTR_BAG(tam)[j];

            /* if no vector with this syndrome is known                    */
            if ( PTR_BAG(cls)[n+1] == 0 ) {

                /* copy the vector and add to the table                    */
                cpy = NewBag( T_VECFFE, SIZE_PLEN_VECFFE(len) );
                SET_FLD_VECFFE( cpy, FLD_VECFFE(vec) );
                v = (TypFFE*)(PTR_BAG(vec) + 1);
                c = (TypFFE*)(PTR_BAG(cpy) + 1);
                for ( k = 1; k <= len; k++, v++, c++ )
                    *c = *v;
                SET_BAG(cls, n+1,  cpy );
                rem--;
                if ( rem == 0 )  break;

            }

            /* remove the 1 again                                          */
            ((TypFFE*)(PTR_BAG(vec) + 1))[j-1] = 0;

        }

    }

    /* otherwise recurse                                                   */
    else {

        /* loop over all possible positions for one more 1                 */
        for ( j = i+1; j+(l-1) <= len; j++ ) {

            /* set the 1 and compute the syndrome                          */
            ((TypFFE*)(PTR_BAG(vec) + 1))[j-1] = 1;
            syn ^= (UInt)PTR_BAG(tam)[j];

            /* set the 1                                                   */
            ((TypFFE*)(PTR_BAG(vec) + 1))[j-1] = 1;

            /* recurse, stop if the table is full                          */
            rem = CLM2( cls, rem, syn, vec, tam, l-1, j );
            if ( rem == 0 )  break;

            /* remove the 1 again and correct syndrome                     */
            ((TypFFE*)(PTR_BAG(vec) + 1))[j-1] = 0;
            syn ^= (UInt)PTR_BAG(tam)[j];

        }

    }

    /* return the number of coset leaders that must still be found         */
    return rem;
}

UInt   CLMF (Bag cls, UInt rem, Bag syn, UInt q, Bag vec, Bag tam, UInt l, UInt i, UInt lim)
{
    UInt       len;            /* length of <vec>                 */
    UInt       len2;           /* length of <syn>                 */
    TypFFE *            fld;            /* successor table of field        */
    TypFFE *            v;              /* pointer into <vec>              */
    Bag           cpy;            /* copy of <vec>                   */
    TypFFE *            c;              /* pointer into <cpy>              */
    TypFFE *            r;              /* pointer into a row of <tam>     */
    TypFFE *            s;              /* pointer into <syn>              */
    UInt       n;              /* syndrome index                  */
    TypFFE              t;              /* temporary                       */
    TypFFE              z;              /* factor to multiply row with     */
    TypFFE              y;              /* factor to multiply <vec> with   */
    UInt       j, k;           /* loop variables                  */

    /* initialize                                                          */
    len = LEN_VECFFE(vec);
    len2 = LEN_VECFFE(syn);
    fld = SUCC_FF( FLD_VECFFE(vec) );

    /* if we have only one more non-zero elements to add                   */
    if ( l == 1 ) {

        /* loop over all possible position for this non-zero element       */
        for ( j = i+1; j <= len; j++ ) {

            /* loop over all possible non-zero elements                    */
            for ( z = 1; z <= lim; z++ ) {

                /* set the non-zero element and compute the syndrome index */
                ((TypFFE*)(PTR_BAG(vec) + 1))[j-1] = z;
                n = 0;
                r = (TypFFE*)(PTR_BAG( PTR_BAG(tam)[j] ) + 1);
                s = (TypFFE*)(PTR_BAG( syn ) + 1);
                for ( k = 1; k <= len2; k++, r++, s++ ) {
                    t = PROD_FF( z, *r, fld );
                    n = n * q + SUM_FF( *s, t, fld );
                }

                /* if no vector with this syndrome is known                */
                if ( PTR_BAG(cls)[n+1] == 0 ) {

                    /* copy all non-zero multiplies and add to the table   */
                    for ( y = 1; y <= q-1; y++ ) {

                        /* compute <y> * <vec>                             */
                        cpy = NewBag( T_VECFFE, SIZE_PLEN_VECFFE(len) );
                        SET_FLD_VECFFE( cpy, FLD_VECFFE(vec) );
                        v = (TypFFE*)(PTR_BAG(vec) + 1);
                        c = (TypFFE*)(PTR_BAG(cpy) + 1);
                        for ( k = 1; k <= len; k++, v++, c++ )
                            *c = PROD_FF( y, *v, fld );

                        /* compute the corresponding syndrom index         */
                        n = 0;
                        r = (TypFFE*)(PTR_BAG( PTR_BAG(tam)[j] ) + 1);
                        s = (TypFFE*)(PTR_BAG( syn ) + 1);
                        for ( k = 1; k <= len2; k++, r++, s++ ) {
                            t = PROD_FF( z, *r, fld );
                            t = SUM_FF( *s, t, fld );
                            n = n * q + PROD_FF( y, t, fld );
                        }

                        /* add the new coset leader to the table           */
                        SET_BAG(cls, n+1,  cpy );
                        rem--;
                        if ( rem == 0 )  break;

                    }

                }

                /* remove the non-zero element                             */
                ((TypFFE*)(PTR_BAG(vec) + 1))[j-1] = 0;

            }

        }

    }

    /* otherwise recurse                                                   */
    else {

        /* loop over all possible positions for one more 1                 */
        for ( j = i+1; j+(l-1) <= len; j++ ) {

            /* loop over all possible non-zero elements                    */
            for ( z = 1; z <= lim; z++ ) {

                /* set the non-zero element and compute the syndrome       */
                ((TypFFE*)(PTR_BAG(vec) + 1))[j-1] = z;
                r = (TypFFE*)(PTR_BAG( PTR_BAG(tam)[j] ) + 1);
                s = (TypFFE*)(PTR_BAG( syn ) + 1);
                for ( k = 1; k <= len2; k++, r++, s++ ) {
                    t = PROD_FF( z, *r, fld );
                    *s = SUM_FF( *s, t, fld );
                }

                /* recurse, stop if the table is full                      */
                rem = CLMF( cls, rem, syn, q, vec, tam, l-1, j, q-1 );
                if ( rem == 0 )  break;

                /* remove the non-zero element again and correct syndrome  */
                ((TypFFE*)(PTR_BAG(vec) + 1))[j-1] = 0;
                z = NEG_FF( z, fld );
                r = (TypFFE*)(PTR_BAG( PTR_BAG(tam)[j] ) + 1);
                s = (TypFFE*)(PTR_BAG( syn ) + 1);
                for ( k = 1; k <= len2; k++, r++, s++ ) {
                    t = PROD_FF( z, *r, fld );
                    *s = SUM_FF( *s, t, fld );
                }
                z = NEG_FF( z, fld );

            }

        }

    }

    /* return the number of coset leaders that must still be found         */
    return rem;
}

Bag       CosetLeaderMatFFE (Bag mat, Bag q)
{
    Bag           cls;            /* coset leaders, result           */
    UInt       rem;            /* number of coset leaders         */
    UInt       p;              /* characteristic                  */
    UInt       len;            /* length of <mat>                 */
    Bag           tam;            /* transposed of <mat>             */
    Bag           vec;            /* temporary                       */
    Bag           syn;            /* temporary                       */
    UInt       i, k, l;        /* loop variables                  */

    /* check the arguments                                                 */
    if ( GET_TYPE_BAG(q) != T_INT || ! (p = RootPrimePower( HD_TO_INT(q) )) )
        return Error("CosetLeaderMatFFE: %s",
            (Int)"<q> must be a positive prime power",0);
    if ( ! ConvMatFFE( mat, HD_TO_INT(q) ) )
        return Error("CosetLeaderMatFFE: %s",
            (Int)"<mat> must be a matrix over GF(<q>)",0);

    /* check that there is a hope to finish                                */
    len = LEN_LIST(mat);
    if ( (         2 <= HD_TO_INT(q) && 31 <= len)
      || (         3 <= HD_TO_INT(q) && 19 <= len)
      || (         4 <= HD_TO_INT(q) && 16 <= len)
      || (         5 <= HD_TO_INT(q) && 13 <= len)
      || (         8 <= HD_TO_INT(q) && 11 <= len)
      || (         9 <= HD_TO_INT(q) && 10 <= len)
      || (        13 <= HD_TO_INT(q) &&  9 <= len)
      || (        19 <= HD_TO_INT(q) &&  8 <= len)
      || (        32 <= HD_TO_INT(q) &&  7 <= len)
      || (        64 <= HD_TO_INT(q) &&  6 <= len)
      || (       181 <= HD_TO_INT(q) &&  5 <= len)
      || (      1024 <= HD_TO_INT(q) &&  4 <= len)
      || (     32768 <= HD_TO_INT(q) &&  3 <= len)
      || (1073741824 <= HD_TO_INT(q) &&  2 <= len) ) {
        return Error("CosetLeaderMatFFE: %s",
            (Int)"sorry, no hope to finish",0);
    }

    /* create the coset leaders table, enter trivial coset leader          */
    rem = 1;
    for ( i = 1; i <= len; i++ )
        rem *= HD_TO_INT(q);
    cls = NewBag( T_LIST, SIZE_PLEN_PLIST( rem ) );
    SET_LEN_PLIST( cls, rem );
    vec = NewBag( T_VECFFE, GET_SIZE_BAG(PTR_BAG(mat)[1]) );
    SET_FLD_VECFFE( vec, FLD_VECFFE(PTR_BAG(mat)[1]) );
    SET_BAG(cls, 1,  vec );
    rem--;

    /* if <q> == 2, transpose <mat> to integers (very short blists ;-)     */
    if ( HD_TO_INT(q) == 2 ) {

        /* transpose <mat> to integers                                     */
        tam = NewBag( T_BLIST, SIZE_PLEN_PLIST(LEN_VECFFE(PTR_BAG(mat)[1])) );
        SET_LEN_BLIST( tam, LEN_VECFFE(PTR_BAG(mat)[1]) );
        for ( i = 1; i <= LEN_PLIST( tam ); i++ ) {
            l = 0;
            for ( k = 1; k <= len; k++ ) {
                l = (l << 1)
                  | ((TypFFE*)(PTR_BAG( PTR_BAG(mat)[k] ) + 1))[i-1];
            }
            *(UInt*)&(PTR_BAG(tam)[i]) = (UInt)l;
        }

        /* create a vector                                                 */
        vec = NewBag( T_VECFFE, GET_SIZE_BAG(PTR_BAG(mat)[1]) );
        SET_FLD_VECFFE( vec, FLD_VECFFE(PTR_BAG(mat)[1]) );

        /* enumerate all vectors starting with low weight ones             */
        for ( l = 1; l <= len && rem != 0; l++ )
            rem = CLM2( cls, rem, 0, vec, tam, l, 0 );

    }

    /* otherwise, we have to enumerate reps for the 1-dim subspaces        */
    else {

        /* transpose <mat>                                                 */
        tam = NewBag( T_LIST, SIZE_PLEN_PLIST(LEN_VECFFE(PTR_BAG(mat)[1])) );
        SET_LEN_PLIST( tam, LEN_VECFFE(PTR_BAG(mat)[1]) );
        for ( i = 1; i <= LEN_PLIST( tam ); i++ ) {
            syn = NewBag( T_VECFFE, SIZE_PLEN_VECFFE(len) );
            SET_FLD_VECFFE( syn, FLD_VECFFE(PTR_BAG(mat)[1]) );
            for ( k = 1; k <= len; k++ ) {
                ((TypFFE*)(PTR_BAG(syn)+1))[k-1]
                    = ((TypFFE*)(PTR_BAG( PTR_BAG(mat)[k] ) + 1))[i-1];
            }
            SET_BAG(tam, i,  syn );
        }

        /* create a vector and a syndrome                                  */
        vec = NewBag( T_VECFFE, GET_SIZE_BAG(PTR_BAG(mat)[1]) );
        SET_FLD_VECFFE( vec, FLD_VECFFE(PTR_BAG(mat)[1]) );
        syn = NewBag( T_VECFFE, SIZE_PLEN_VECFFE(len) );
        SET_FLD_VECFFE( syn, FLD_VECFFE(PTR_BAG(mat)[1]) );

        /* enumerate all vectors starting with low weight ones             */
        for ( l = 1; l <= len && rem != 0; l++ )
            rem = CLMF( cls, rem, syn, HD_TO_INT(q), vec, tam, l, 0, 1 );

    }

    /* check that everything is ok                                         */
    if ( rem != 0 ) {
        return Error("CosetLeaderMatFFE: %s",
           (Int)"the rows of <mat> must be linearly independent",0);
    }

    /* return the coset leaders list                                       */
    return cls;
}

Bag       FunCosetLeadersMatFFE (Bag hdCall)
{
    /* check the number of arguments                                       */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error("number of arguments must be 2 not %d",
                     (Int)(GET_SIZE_BAG(hdCall)/SIZE_HD - 1), 0 );

    /* call the real internal function                                     */
    return CosetLeaderMatFFE(
                        EVAL( PTR_BAG(hdCall)[1] ), EVAL( PTR_BAG(hdCall)[2] ) );
}


/****************************************************************************
**
*F  InitCoding()  . . . . . . . . . . . . .  initialize coding theory package
**
**  'InitCoding' initializes the coding theory package.
*/
void            InitCoding (void)
{
    InstIntFunc( "DistanceVecFFE",
                 FunDistanceVecFFE );
    InstIntFunc( "DistancesDistributionVecFFEsVecFFE",
                 FunDistancesDistributionVecFFEsVecFFE );
    InstIntFunc( "DistancesDistributionMatFFEVecFFE",
                 FunDistancesDistributionMatFFEVecFFE );
    InstIntFunc( "AClosestVectorCombinationsMatFFEVecFFE",
                 FunAClosestVectorCombinationsMatFFEVecFFE );
    InstIntFunc( "CosetLeadersMatFFE",
                 FunCosetLeadersMatFFE );
}



