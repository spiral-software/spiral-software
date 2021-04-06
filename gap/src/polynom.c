/****************************************************************************
**
*A  polynom.c                    GAP source                      Frank Celler
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
*/
#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "scanner.h"             /* reading of tokens and printing  */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */
#include        "finfield.h"            /* finite field package            */
#include        "list.h"                /* generic list package            */
#include        "plist.h"               /* plain list package              */
#include        "vector.h"              /* vector package                  */
#include        "vecffe.h"              /* finite field vector package     */
#include        "objects.h"
#include		"string4.h"
#include        "integer4.h"            /* TypDigit                        */

#include        "polynom.h"             /* polynomial package              */


/****************************************************************************
**
*F  UnifiedFieldVecFFE( <hdL>, <hdR> )	. . . unify fields of <hdL> and <hdR>
**
**  Convert two finite field vectors into finite field vectors over  the same
**  finite field.  Signal an error if this conversion fails.
*/
Bag UnifiedFieldVecFFE (Bag hdL, Bag hdR)
             	        	        /* first finite field vector       */
                                        /* second finite field vector      */
{
    Bag           hdFld;          /* handle of the field             */
    UInt       p;              /* characteristic                  */
    UInt       q;              /* size of common finite field     */
    UInt       q1;             /* size of field of row            */
    UInt       dl;             /* degree of <hdL>                 */
    UInt       dr;             /* degree of <hdR>                 */
    Bag           hdElm;          /* one row of the list             */
    TypFFE              v;              /* value of one element            */
    UInt       i, k;           /* loop variables                  */

    /* if <hdL> and <hdR> have already the same field return               */
    if ( FLD_VECFFE(hdL) == FLD_VECFFE(hdR) )
	return FLD_VECFFE(hdL);

    /* check the we know a common superfield of <hdL> and <hdR>            */
    p = CharVecFFE(hdL);
    if ( p != CharVecFFE(hdR) )
	return Error( "vectors have different characteristic", 0, 0 );

    /* compute degree of superfield                                        */
    dl = DegreeVecFFE(hdL);
    dr = DegreeVecFFE(hdR);
    for ( k = dl;  dl % dr != 0;  dl += k )  ;

    /* make sure we can handle this field                                  */
    if ( (  2 <= p && 17 <= dl ) || (   3 <= p && 11 <= dl )
      || (  5 <= p &&  7 <= dl ) || (   7 <= p &&  6 <= dl )
      || ( 11 <= p &&  5 <= dl ) || (  17 <= p &&  4 <= dl )
      || ( 41 <= p &&  3 <= dl ) || ( 257 <= p &&  2 <= dl ) )
    {
	return Error( "common superfield is too large", 0, 0 );
    }

    /* get a field that contains all elements                              */
    for ( q = 1, k = 1;  k <= dl;  k++ )  q *= p;
    if ( (SIZE_FF(FLD_VECFFE(hdL))-1) % (q-1) == 0 )
	hdFld = FLD_VECFFE(hdL);
    else if ( (SIZE_FF(FLD_VECFFE(hdR))-1) % (q-1) == 0 )
	hdFld = FLD_VECFFE(hdR);
    else
	hdFld = FLD_FFE( RootFiniteField( q ) );
    q = SIZE_FF(hdFld);

    /* convert <hdL> and <hdR>                                             */
    for ( i = 0;  i <= 1;  i++ )
    {
	hdElm = ( i == 0 ) ? hdL : hdR;
	if ( FLD_VECFFE(hdElm) != hdFld )
	{
	    q1 = SIZE_FF( FLD_VECFFE(hdElm) );
	    for ( k = LEN_VECFFE(hdElm);  0 < k;  k-- )
	    {
		v = VAL_VECFFE( hdElm, k );
		SET_VAL_VECFFE( hdElm, k, v==0 ? v : (v-1)*(q-1)/(q1-1)+1 );
	    }
	    SET_FLD_VECFFE( hdElm, hdFld );
	}
    }

    /* and return the common                                               */
    return FLD_VECFFE(hdL);
}


/****************************************************************************
**
*F  FunShiftedCoeffs( <hdCall> )  . . . . . internal function 'ShiftedCoeffs'
**
**  'FunShiftedCoeffs' implements 'ShiftedCoeffs( <l>, <n> )'
*/
Bag (*TabShiftedCoeffs[T_VAR]) ( Bag, Int );

Bag FunShiftedCoeffs (Bag hdCall)
{
    Bag       	hdC;		/* coeffs list                     */
    Bag       	hdN;		/* number of shifts                */

    /* check arguments                                                     */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )
        return Error( "usage: ShiftedCoeffs( <c>, <nr> )", 0, 0 );
    hdC = EVAL( PTR_BAG(hdCall)[1] );
    hdN = EVAL( PTR_BAG(hdCall)[2] );
    if ( GET_TYPE_BAG(hdN) != T_INT )
	return Error( "<nr> must be an integer", 0, 0 );

    /* jump through the table                                              */
    return TabShiftedCoeffs[XType(hdC)]( hdC, HD_TO_INT(hdN) );
}

Bag CantShiftedCoeffs (Bag hdList, Int n)
{
    return Error( "<list> must be a vector", 0, 0 );
}

Bag ShiftedCoeffsListx (Bag hdC, Int n)
{
    Bag		hdS;		/* result                          */
    Bag		hdZero;		/* zero element                    */
    Int                l;		/* length of <hdC>                 */
    Int                i;

    /* compute the length of <hdC>                                         */
    l = LEN_LIST(hdC);
    if ( l == 0 )
	hdS = Copy(hdC);

    /* if <n> is negative shrink <hdC>                                     */
    else if ( n < 0 ) {

	/* if <l> is not bigger than <n>,  <hdC> will shrink to nothing    */
	if ( l <= -n ) {
	    hdS = NewBag( T_LIST, SIZE_PLEN_PLIST(0) );
	    SET_LEN_PLIST( hdS, 0 );
	    return hdS;
	}

	/* allocate a new vector and copy the entries                      */
	hdS = NewBag( T_LIST, SIZE_PLEN_PLIST(l+n) );
	SET_LEN_PLIST( hdS, l+n );
	for ( i = l;  -n < i;  i-- )
	    SET_ELM_PLIST( hdS, i+n, ELM_PLIST(hdC,i) );
    }

    /* if <n> is positive, copy entries and leading zero                   */
    else if ( 0 < n ) {
	hdS = NewBag( T_LIST, SIZE_PLEN_PLIST(l+n) );
	SET_LEN_PLIST( hdS, l+n );
	for ( i = l;  0 < i;  i-- )
	    SET_ELM_PLIST( hdS, i+n, ELM_PLIST(hdC,i) );
	hdZero = PROD( ELML_LIST(hdC,1), INT_TO_HD(0) );
	for ( i = n;  0 < i;  i-- )
	    SET_ELM_PLIST( hdS, i, hdZero );
    }

    /* if <n> is zero, return a copy of <hdC>                              */
    else
	hdS = Copy(hdC);

    /* return <hdS>                                                        */
    return hdS;
}

Bag ShiftedCoeffsVecFFE (Bag hdC, Int n)
{
    Bag		hdS;		/* result                          */
    TypFFE		tmp;		/* temporary                       */
    Int                l;		/* length of <hdC>                 */
    Int                i;

    /* compute the length of <hdC>                                         */
    l = LEN_VECFFE(hdC);

    /* if <n> is negative shrink <hdL>                                     */
    if ( n < 0 ) {

	/* if <l> is not bigger than <n>, <hdL> will shrink to nothing     */
	if ( l <= -n ) {
	    hdS = NewBag( T_LIST, SIZE_PLEN_PLIST(0) );
	    SET_LEN_PLIST( hdS, 0 );
	    return hdS;
	}

	/* allocate a new vector and copy the entries                      */
	hdS = NewBag( T_VECFFE, SIZE_PLEN_VECFFE(l+n) );
	SET_FLD_VECFFE( hdS, FLD_VECFFE(hdC) );
	for ( i = l;  -n < i;  i-- ) {
	    tmp = VAL_VECFFE( hdC, i );
	    SET_VAL_VECFFE( hdS, i+n, tmp );
	}
    }

    /* if <n> is positive, copy <hdC> and leading zeros                    */
    else if ( 0 < n )
    {
	hdS = NewBag( T_VECFFE, SIZE_PLEN_VECFFE(l+n) );
	SET_FLD_VECFFE( hdS, FLD_VECFFE(hdC) );
	for ( i = l;  0 < i;  i-- ) {
	    tmp = VAL_VECFFE( hdC, i );
	    SET_VAL_VECFFE( hdS, i+n, tmp );
	}
	for ( i = n;  0 < i;  i-- )
	    SET_VAL_VECFFE( hdS, i, 0 );
    }

    /* if <n> is zero, copy <hdL> and shrink if necessary                  */
    else
	hdS = Copy(hdC);
    return hdS;
}


/****************************************************************************
**
*F  FunNormalizeCoeffs( <hdCall> )  . . . internal function 'NormalizeCoeffs'
**
**  'FunNormalizeCoeffs' implements 'NormalizeCoeffs( <c> )'
*/
Bag (*TabNormalizeCoeffs[T_VAR]) ( Bag );

Bag FunNormalizeCoeffs (Bag hdCall)
{
    Bag       hdC;

    /* check arguments                                                     */
    if ( GET_SIZE_BAG( hdCall ) != 2 * SIZE_HD )
        return Error( "usage: NormalizeCoeffs( <c> )", 0, 0 );
    hdC = EVAL( PTR_BAG(hdCall)[1] );

    /* jump through the table                                              */
    return TabNormalizeCoeffs[XType(hdC)]( hdC );
}

Bag CantNormalizeCoeffs (Bag hdList)
{
    return Error( "<list> must be a vector", 0, 0 );
}

Bag NormalizeCoeffsVecFFE (Bag hdC)
{
    TypFFE		tmp;            /* temporary                       */
    Int		len;		/* length of <hdC>                 */
    Int		l1;		/* first non zero entry            */
    Int		l2;		/* last non zero entry             */
    Int		i;		/* loop                            */

    /* check for leading zeros                                             */
    len = LEN_VECFFE(hdC);
    for ( l1 = 1;  l1 <= len;  l1++ )
	if ( VAL_VECFFE( hdC, l1 ) != 0 )
	    break;

    /* check for trailing zero                                             */
    for ( l2 = len;  l1 < l2;  l2-- )
	if ( VAL_VECFFE( hdC, l2 ) != 0 )
	    break;

    /* if it is trivial, return empty list                                 */
    if ( l2 < l1 ) {
	Retype( hdC, T_LIST );
	Resize( hdC, SIZE_PLEN_PLIST(0) );
	SET_LEN_PLIST( hdC, 0 );
	return INT_TO_HD(0);
    }

    /* remove leadinf zeros                                                */
    if ( 1 < l1 )
	for ( i = l1;  i <= l2;  i++ ) {
	    tmp = VAL_VECFFE( hdC, i );
	    SET_VAL_VECFFE( hdC, i-l1+1, tmp );
	}

    /* shrink vector                                                       */
    Resize( hdC, SIZE_PLEN_VECFFE(l2-l1+1) );

    /* return the number of removed trailing zeros                         */
    return INT_TO_HD(l1-1);
}

Bag NormalizeCoeffsListx (Bag hdC)
{
    Bag		hdZero;         /* temporary                       */
    Bag		hdTmp;          /* temporary                       */
    Int		len;		/* length of <hdC>                 */
    Int		l1;		/* first non zero entry            */
    Int		l2;		/* last non zero entry             */
    Int		i;		/* loop                            */

    /* construct zero                                                      */
    if ( LEN_LIST(hdC) == 0 )
	return INT_TO_HD(0);
    hdZero = PROD( INT_TO_HD(0), ELM_LIST(hdC,1) );

    /* check for leading zeros                                             */
    len = LEN_LIST(hdC);
    for ( l1 = 1;  l1 <= len;  l1++ ) {
	hdTmp = ELML_LIST(hdC,l1);
	if ( EQ( hdTmp, hdZero ) == HdFalse )
	    break;
    }

    /* check for trailing zero                                             */
    for ( l2 = len;  l1 < l2;  l2-- ) {
	hdTmp = ELML_LIST(hdC,l2);
	if ( EQ( hdTmp, hdZero ) == HdFalse )
	    break;
    }

    /* if it is trivial, return empty list                                 */
    if ( l2 < l1 ) {
	Retype( hdC, T_LIST );
	Resize( hdC, SIZE_PLEN_PLIST(0) );
	SET_LEN_PLIST( hdC, 0 );
	return INT_TO_HD(0);
    }

    /* remove leading zeros                                                */
    if ( 1 < l1 )
	for ( i = l1;  i <= l2;  i++ ) {
	    hdTmp = ELML_LIST( hdC, i );
	    SET_ELM_PLIST( hdC, i-l1+1, hdTmp );
	}

    /* shrink vector                                                       */
    Resize( hdC, SIZE_PLEN_PLIST(l2-l1+1) );
    SET_LEN_PLIST( hdC, l2-l1+1 );

    /* return the number of removed trailing zeros                         */
    return INT_TO_HD(l1-1);
}


/****************************************************************************
**
*F  FunShrinkCoeffs( <hdCall> )  . . . . . . internal function 'ShrinkCoeffs'
**
**  'FunShrinkCoeffs' implements 'ShrinkCoeffs( <c> )'
*/
void (*TabShrinkCoeffs[T_VAR]) ( Bag );

Bag FunShrinkCoeffs (Bag hdCall)
{
    Bag       hdC;

    /* check arguments                                                     */
    if ( GET_SIZE_BAG( hdCall ) != 2 * SIZE_HD )
        return Error( "usage: ShrinkCoeffs( <c> )", 0, 0 );
    hdC = EVAL( PTR_BAG(hdCall)[1] );

    /* jump through the table                                              */
    TabShrinkCoeffs[XType(hdC)]( hdC );
    return HdVoid;
}

void CantShrinkCoeffs (Bag hdList)
{
    Error( "<list> must be a vector", 0, 0 );
}

void ShrinkCoeffsVecFFE (Bag hdC)
{
    Int		len;		/* length of <hdC>                 */
    Int		i;		/* loop                            */

    /* check for trailing zero                                             */
    len = LEN_VECFFE(hdC);
    for ( i = len;  0 < i;  i-- )
	if ( VAL_VECFFE( hdC, i ) != 0 )
	    break;

    /* if it is trivial, return empty list                                 */
    if ( 0 == i ) {
	Retype( hdC, T_LIST );
	Resize( hdC, SIZE_PLEN_PLIST(0) );
	SET_LEN_PLIST( hdC, 0 );
	return;
    }

    /* shrink vector                                                       */
    Resize( hdC, SIZE_PLEN_VECFFE(i) );
}

void ShrinkCoeffsListx (Bag hdC)
{
    Bag		hdZero;         /* temporary                       */
    Bag		hdTmp;          /* temporary                       */
    Int		len;		/* length of <hdC>                 */
    Int		i;		/* loop                            */

    /* construct zero                                                      */
    if ( LEN_LIST(hdC) == 0 )
	return;
    hdZero = PROD( INT_TO_HD(0), ELM_LIST(hdC,1) );

    /* check for trailing zero                                             */
    len = LEN_LIST(hdC);
    for ( i = len;  0 < i;  i-- ) {
	hdTmp = ELML_LIST(hdC,i);
	if ( EQ( hdTmp, hdZero ) == HdFalse )
	    break;
    }

    /* if it is trivial, return empty list                                 */
    if ( 0 == i ) {
	Retype( hdC, T_LIST );
	Resize( hdC, SIZE_PLEN_PLIST(0) );
	SET_LEN_PLIST( hdC, 0 );
	return;
    }

    /* shrink vector                                                       */
    Resize( hdC, SIZE_PLEN_PLIST(i) );
    SET_LEN_PLIST( hdC, i );
}


/****************************************************************************
**
*F  ADD_COEFFS( <hdL>, <hdR>, <hdM> ) . . . . . <hdL>+<hdM>*<hdR> into <hdL>
**
#define ADD_COEFFS( hdL, hdR, hdM ) \
    (TabAddCoeffs[XType(hdL)][XType(hdR)]( hdL, hdR, hdM ))
*/
void (*TabAddCoeffs[T_VAR][T_VAR]) ( Bag, Bag, Bag );

void CantAddCoeffs (Bag hdL, Bag hdR, Bag hdM)
{
    Error("<l> and <r> must be vectors over a common field", 0, 0);
}

void AddCoeffsListxListx (Bag hdL, Bag hdR, Bag hdM)
{
    Bag           hdLL;       /* one element of <hdL>                */
    Bag           hdRR;       /* one element of <hdR>                */
    Bag           hdAA;       /* sum of <hdLL> and <hdRR>            */
    Int                l;          /* length of <hdL>                     */
    Int                r;          /* length of <hdR>                     */
    Int                m;          /* minimum of <l> and <r>              */
    Int                i;          /* loop variables                      */

    /* compute the length of <hdL> and <hdR>                               */
    l = LEN_LIST(hdL);
    r = LEN_LIST(hdR);

    /* if <l> is less than <r>, enlarge <hdL>                              */
    if ( l < r ) {
	hdLL = PROD( ELML_LIST(hdR,1), INT_TO_HD(0) );
	for ( i = l+1;  i <= r;  i++ )
            ASS_LIST( hdL, i, hdLL );
	l = r;
    }

    /* add <hdR> to <hdL>                                                  */
    m = ( l < r ) ? l : r;
    if ( hdM == INT_TO_HD(1) ) {
	for ( i = 1;  i <= m;  i++ ) {
	    hdLL = ELM_LIST( hdL, i );
	    hdRR = ELM_LIST( hdR, i );
	    hdAA = SUM( hdLL, hdRR );
	    ASS_LIST( hdL, i, hdAA );
	}
	for ( i = m+1;  i <= r;  i++ ) {
	    hdRR = ELM_LIST( hdR, i );
	    ASS_LIST( hdL, i, hdRR );
	}
    }
    else {
	for ( i = 1;  i <= m;  i++ ) {
	    hdLL = ELM_LIST( hdL, i );
	    hdRR = ELM_LIST( hdR, i );
	    hdRR = PROD( hdRR, hdM );
	    hdAA = SUM( hdLL, hdRR );
	    ASS_LIST( hdL, i, hdAA );
	}
	for ( i = m+1;  i <= r;  i++ ) {
	    hdRR = ELM_LIST( hdR, i );
	    hdRR = PROD( hdRR, hdM );
	    ASS_LIST( hdL, i, hdRR );
	}
    }
}

void AddCoeffsVecFFEVecFFE (Bag hdL, Bag hdR, Bag hdM)
{
    Int            l;          /* degree plus one of left polynomial      */
    Int            r;          /* degree plus one of right polynomial     */
    TypFFE          m;          /* multiple of <hdR> to add                */
    TypFFE        * ptL;        /* coeffs vectors of left polynomial       */
    TypFFE        * ptR;        /* coeffs vectors of right polynomial      */
    TypFFE        * ptE;        /* end of coeffs vector of sum             */
    TypFFE        * f;          /* finite field                            */
    TypFFE          t;          /* temp for FFEs                           */

    /* if <hdM> is a finite field element check the field                  */
    if ( GET_TYPE_BAG(hdM) == T_FFE ) {
	if ( UnifiedFieldVecFFE(hdL,hdR) == FLD_FFE(hdM) )
	    m = VAL_FFE(hdM);
	else {
	    hdR = PROD( hdR, hdM );
	    m   = 1;
	}
    }

    /* otherwise replace <hdR> by the <hdM> multiple                       */
    else {
	if ( INT_TO_HD(1) != hdM ) {
	    hdR = PROD( hdR, hdM );
	    hdM = INT_TO_HD(1);
	    TabAddCoeffs[XType(hdL)][XType(hdR)]( hdL, hdR, hdM );
	    return;
	}
	else
	    m = 1;
    }

    /* if <m> is trivial there is nothing to add                           */
    if ( m == 0 )
	return;

    /* if <l> is less than <r> enlarge <hdL>                               */
    l = LEN_VECFFE(hdL);
    r = LEN_VECFFE(hdR);
    if ( l < r ) {
	Resize( hdL, SIZE_PLEN_VECFFE(r) );
	SET_VAL_VECFFE( hdL, l+1, 0 );
	l = r;
    }

    /* get the common finite field                                         */
    f = (TypFFE*) PTR_BAG( UnifiedFieldVecFFE( hdL, hdR ) );

    /* set up all the pointers                                             */
    ptL = (TypFFE*)( PTR_BAG(hdL) + 1 );
    ptR = (TypFFE*)( PTR_BAG(hdR) + 1 );
    ptE = ptR + r;

    /* add <hdR> to <hdL>                                                  */
    if ( m == 1 ) {
	while ( ptR < ptE ) {
	    *ptL = SUM_FF( *ptL, *ptR, f );
	    ptL++;
	    ptR++;
	}
    }
    else {
	while ( ptR < ptE ) {
	    t = PROD_FF( *ptR, m, f );
	    *ptL = SUM_FF( *ptL, t, f );
	    ptL++;
	    ptR++;
	}
    }
}

void AddCoeffsListxVecFFE (Bag hdL, Bag hdR, Bag hdM)
{
    /* catch the special case that <hdL> is empty                          */
    if ( LEN_LIST(hdL) == 0 ) {
	Retype( hdL, T_VECFFE );
	Resize( hdL, SIZE_PLEN_VECFFE(1) );
	SET_FLD_VECFFE( hdL, FLD_VECFFE(hdR) );
	SET_VAL_VECFFE( hdL, 1, 0 );

	/* use 'AddCoeffsVecFFEVecFFE'                                     */
	AddCoeffsVecFFEVecFFE( hdL, hdR, hdM );
    }

    /* use 'AddCoeffsListxListx'                                           */
    else
	AddCoeffsListxListx( hdL, hdR, hdM );
}


/****************************************************************************
**
*F  FunAddCoeffs( <hdCall> )  . . . . . . . . . internal function 'AddCoeffs'
**
**  'FunAddCoeffs' implements 'AddCoeffs( <l>, <r> )'
*/
Bag FunAddCoeffs (Bag hdCall)
{
    Bag       	hdL;
    Bag       	hdR;
    Bag       	hdM;

    /* check arguments                                                     */
    if ( 4 * SIZE_HD < GET_SIZE_BAG(hdCall) || GET_SIZE_BAG(hdCall) < 3 * SIZE_HD )
        return Error( "usage: AddCoeffs( <l>, <r> )", 0, 0 );
    hdL = EVAL( PTR_BAG(hdCall)[1] );
    hdR = EVAL( PTR_BAG(hdCall)[2] );
    if ( 4 * SIZE_HD == GET_SIZE_BAG(hdCall) )
	hdM = EVAL( PTR_BAG(hdCall)[3] );
    else
	hdM = INT_TO_HD(1);

    /* jump through the table                                              */
    TabAddCoeffs[XType(hdL)][XType(hdR)]( hdL, hdR, hdM );
    return HdVoid;
}


/****************************************************************************
**
*F  FunSumCoeffs( <hdCall> )  . . . . . . . . . internal function 'SumCoeffs'
**
**  'FunSumCoeffs' implements 'SumCoeffs( <l>, <r> )'
*/
Bag FunSumCoeffs (Bag hdCall)
{
    Bag       	hdL;
    Bag       	hdR;
    Bag       	hdM;

    /* check arguments                                                     */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )
        return Error( "usage: SumCoeffs( <l>, <r> )", 0, 0 );
    hdL = Copy( EVAL( PTR_BAG(hdCall)[1] ) );
    hdR = EVAL( PTR_BAG(hdCall)[2] );
    hdM = INT_TO_HD(1);

    /* jump through the table                                              */
    TabAddCoeffs[XType(hdL)][XType(hdR)]( hdL, hdR, hdM );

    /* and return the result stored in <hdL>                               */
    return hdL;
}


/****************************************************************************
**
*F  MULTIPLY_COEFFS( <hdP>, <hdL>, <l>, <hdR>, <r> )   <hdL>*<hdR> into <hdP>
**
#define MULTIPLY_COEFFS(hdP,hdL,l,hdR,r) \
    (TabMultiplyCoeffs[XType(hdL)][XType(hdR)](hdP,hdL,l,hdR,r))
*/
Int (*TabMultiplyCoeffs[T_VAR][T_VAR]) (
    Bag, Bag, Int, Bag, Int );

Int CantMultiplyCoeffs (Bag hdP, Bag hdL, Int l, Bag hdR, Int r)
             	    	    	/* space for result of <hdL> * <hdR>       */
                    	        /* left polynomial coeffs                  */
                    	        /* right polynomial coeffs                 */
                    	        /* degree plus one of left polynomial      */
                    	        /* degree plus one of right polynomial     */
{
    Error( "<l> and <r> must be vectors over a common field", 0, 0);
    return 0;
}

Int MultiplyCoeffsListxListx (Bag hdP, Bag hdL, Int l, Bag hdR, Int r)
             	    	    	/* space for result of <hdL> * <hdR>       */
                    	        /* left polynomial coeffs                  */
                    	        /* right polynomial coeffs                 */
                    	        /* degree plus one of left polynomial      */
                    	        /* degree plus one of right polynomial     */
{
    Bag       	hdLL;   /* one element of <hdL>                    */
    Bag       	hdRR;   /* one element of <hdR>                    */
    Bag       	hdPP;   /* one element of <hdP>                    */
    Bag       	hdTT;   /* temp element                            */
    Int            	i,u,k;  /* loop variables                          */

    /* catch the trivial case                                              */
    if ( l == 0 || r == 0 )
	return 0;

    /* fold the product                                                    */
    for ( i = l+r;  1 < i;  i-- )
    {
	hdPP = INT_TO_HD(0);
	u = ( i-1 < l ) ? i-1 : l;
	for ( k = ( i-r < 1 ) ? 1 : i-r;  k <= u;  k++ )
	{
	    hdLL = ELML_LIST( hdL, k   );
	    hdRR = ELMR_LIST( hdR, i-k );
	    hdTT = PROD( hdLL, hdRR );
	    hdPP = SUM( hdPP, hdTT );
	}
	ASS_LIST( hdP, i-1, hdPP );
    }
    return r+l-1;
}

Int MultiplyCoeffsVecFFEVecFFE (Bag hdP, Bag hdL, Int l, Bag hdR, Int r)
             	    	    	/* space for result of <hdL> * <hdR>       */
                    	        /* left polynomial coeffs                  */
                    	        /* right polynomial coeffs                 */
                    	        /* degree plus one of left polynomial      */
                    	        /* degree plus one of right polynomial     */
{
    TypFFE *            ptL;    /* coeffs vectors of left polynomial       */
    TypFFE * 		ptLL;   /* coeffs vectors of left polynomial       */
    TypFFE * 		ptEnd;  /* end of <ptL>                            */
    TypFFE * 		ptR;    /* coeffs vectors of right polynomial      */
    TypFFE * 		ptP;    /* coeffs vectors of product               */
    TypFFE * 		ptPP;   /* coeffs vectors of product               */
    TypFFE * 		f;      /* finite field                            */
    TypFFE              t;      /* temp for FFEs                           */
    Int                i;      /* loop variable                           */

    /* if the one of the polynomials is trivial, return the zero poly      */
    if ( l == 0 || r == 0 )
	return 0;

    /* get a common field of <hdL> and <hdR>                               */
    f = (TypFFE*) PTR_BAG( UnifiedFieldVecFFE( hdL, hdR ) );

    /* set common field in <hdP>                                           */
    SET_FLD_VECFFE( hdP, FLD_VECFFE(hdL) );

    /* chose larger vector as multiplicator                                */
    if ( l < r ) {
	ptL = (TypFFE*)( PTR_BAG(hdL) + 1 );
	ptR = (TypFFE*)( PTR_BAG(hdR) + 1 );
    }
    else {
	ptR = (TypFFE*)( PTR_BAG(hdL) + 1 );
	ptL = (TypFFE*)( PTR_BAG(hdR) + 1 );
	i = l;  l = r;  r = i;
    }

    /* clear <hdP>                                                         */
    ptP = (TypFFE*)( PTR_BAG(hdP) + 1 );
    for ( ptPP = ptP, ptEnd = ptP + (l+r-1);  ptPP < ptEnd;  ptPP++ )
	*ptPP = 0;

    /* add <hdL> to <hdP> multiplied by the elements of <hdR>              */
    ptEnd = ptL + l;
    for ( i = 0;  i < r;  i++, ptR++, ptP++ ) {
	if ( *ptR != 0 ) {
	    for ( ptLL = ptL, ptPP = ptP;  ptLL < ptEnd;  ptLL++, ptPP++ )
		if ( *ptLL != 0 ) {
		    t = PROD_FF( *ptLL, *ptR, f );
		    *ptPP = SUM_FF( *ptPP, t, f );
		}
	}
    }
    return l+r-1;
}


/****************************************************************************
**
*F  FunProductCoeffs( <hdCall> )  . . . . . internal function 'ProductCoeffs'
**
**  'FunProductCoeffs' implements 'ProductCoeffs( <l>, <r> )'
*/
Bag (*TabProductCoeffs[T_VAR][T_VAR]) ( Bag, Bag );

Bag FunProductCoeffs (Bag hdCall)
{
    Bag       	hdL;
    Bag       	hdR;

    /* check arguments                                                     */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )
        return Error( "usage: ProductCoeffs( <l>, <r> )", 0, 0 );
    hdL = EVAL( PTR_BAG(hdCall)[1] );
    hdR = EVAL( PTR_BAG(hdCall)[2] );

    /* jump through the table                                              */
    return TabProductCoeffs[XType(hdL)][XType(hdR)]( hdL, hdR );
}

Bag CantProductCoeffs (Bag hdL, Bag hdR)
{
    return Error("<l> and <r> must be vectors over a common field", 0, 0);
}

Bag ProductCoeffsListxListx (Bag hdL, Bag hdR)
{
    Bag		hdP;		/* result                          */
    Int		l;		/* length of <hdL>                 */
    Int		r;		/* length of <hdR>                 */

    /* get the length of <hdL> and hd <hdR>                                */
    l = LEN_LIST(hdL);
    r = LEN_LIST(hdR);

    /* if <hdL> or <hdR> is trivial,  return the trivial polynomial        */
    if ( l == 0 || r == 0 ) {
	hdP = NewBag( T_LIST, SIZE_PLEN_PLIST(0) );
	SET_LEN_PLIST( hdP, 0 );
	return hdP;
    }

    /* create a bag for the result                                         */
    hdP = NewBag( T_LIST, SIZE_PLEN_PLIST(l+r-1) );
    SET_LEN_PLIST( hdP, l+r-1 );

    /* fold the product                                                    */
    TabMultiplyCoeffs[T_LISTX][T_LISTX]( hdP, hdL, l, hdR, r );

    /* and return the result                                               */
    return hdP;
}

Bag ProductCoeffsVecFFEVecFFE (Bag hdL, Bag hdR)
{
    Bag		hdP;		/* result                          */
    Int		l;		/* length of <hdL>                 */
    Int		r;		/* length of <hdR>                 */

    /* get the length of <hdL> and hd <hdR>                                */
    l = LEN_VECFFE(hdL);
    r = LEN_VECFFE(hdR);

    /* if <hdL> or <hdR> is trivial,  return the trivial polynomial        */
    if ( l == 0 || r == 0 ) {
	hdP = NewBag( T_LIST, SIZE_PLEN_PLIST(0) );
	SET_LEN_PLIST( hdP, 0 );
	return hdP;
    }

    /* create a bag for the result                                         */
    hdP = NewBag( T_VECFFE, SIZE_PLEN_VECFFE(l+r-1) );

    /* fold the product                                                    */
    TabMultiplyCoeffs[T_VECFFE][T_VECFFE]( hdP, hdL, l, hdR, r );
    SET_FLD_VECFFE( hdP, FLD_VECFFE(hdL) );

    /* and return the result                                               */
    return hdP;
}


/****************************************************************************
**
*F  FunProductCoeffsMod( <hdCall> ) . .  internal function 'ProductCoeffsMod'
**
**  'FunProductCoeffsMod' implements 'ProductCoeffsMod( <l>, <r>, <p> )'
*/
Bag (*TabProductCoeffsMod[T_VAR][T_VAR]) (
    Bag, Bag, Bag );

Bag FunProductCoeffsMod (Bag hdCall)
{
    Bag       	hdL;
    Bag       	hdR;
    Bag       	hdN;

    /* check arguments                                                     */
    if ( GET_SIZE_BAG(hdCall) != 4 * SIZE_HD )
        return Error( "usage: ProductCoeffsMod( <l>, <r>, <p> )", 0, 0 );
    hdL = EVAL( PTR_BAG(hdCall)[1] );
    hdR = EVAL( PTR_BAG(hdCall)[2] );
    hdN = EVAL( PTR_BAG(hdCall)[3] );

    /* jump through the table                                              */
    return TabProductCoeffsMod[XType(hdL)][XType(hdR)]( hdL, hdR, hdN );
}

Bag CantProductCoeffsMod (Bag hdL, Bag hdR, Bag hdN)
{
    return Error("<l> and <r> must be vectors over a common field", 0, 0);
}

Bag ProductCoeffsModListxListx (Bag hdL, Bag hdR, Bag hdN)
{
    Bag		hdP;		/* result                          */
    Bag           hdLL;           /* one element of <hdL>            */
    Bag           hdRR;           /* one element of <hdR>            */
    Bag           hdPP;           /* one element of <hdP>            */
    Bag           hdTT;           /* temp element                    */
    Bag           hdQ;            /* <hdP> / 2                       */
    Int                i,  u,  k;      /* loop variables                  */
    Int		l;		/* length of <hdL>                 */
    Int		r;		/* length of <hdR>                 */

    /* get the length of <hdL> and hd <hdR>                                */
    l = LEN_LIST(hdL);
    r = LEN_LIST(hdR);

    /* if <hdL> or <hdR> is trivial,  return the trivial polynomial        */
    if ( l == 0 || r == 0 || hdN == INT_TO_HD(0) ) {
	hdP = NewBag( T_LIST, SIZE_PLEN_PLIST(0) );
	SET_LEN_PLIST( hdP, 0 );
	return hdP;
    }

    /* create a bag for the result                                         */
    hdP = NewBag( T_LIST, SIZE_PLEN_PLIST(l+r-1) );
    SET_LEN_PLIST( hdP, l+r-1 );

    /* if <hdP> is negative reduce into <hdP>/2 and <hdP>                  */
    if ( LT( hdN, INT_TO_HD(0) ) == HdTrue )
    {
	hdN = DIFF( INT_TO_HD(0), hdN );
	hdQ = SUM( hdN, INT_TO_HD(1) );
	hdQ = QuoInt( hdQ, INT_TO_HD(2) );
	hdQ = DIFF( hdQ, INT_TO_HD(1) );
    }
    else
	hdQ = 0;

    /* fold the product                                                    */
    for ( i = l+r;  1 < i;  i-- )
    {
	hdPP = INT_TO_HD(0);
	u = ( i-1 < l ) ? i-1 : l;
	for ( k = ( i-r < 1 ) ? 1 : i-r;  k <= u;  k++ )
	{
	    hdLL = ELML_LIST( hdL, k   );
	    hdRR = ELMR_LIST( hdR, i-k );
	    hdTT = PROD( hdLL, hdRR );
	    hdPP = SUM( hdPP, hdTT );
	}
	hdPP = MOD( hdPP, hdN );
	if ( hdQ != 0 && LT( hdQ, hdPP ) == HdTrue )
	    hdPP = DIFF( hdPP, hdN );
	SET_ELM_PLIST( hdP, i-1, hdPP );
    }

    /* and return the result                                               */
    return hdP;
}


/****************************************************************************
**
*F  REDUCE_COEFFS( <hdL>, <l>, <hdR>, <r> ) . . . . . . reduce <hdL> by <hdR>
**
#define REDUCE_COEFFS( hdL, l, hdR, r ) \
    (TabReduceCoeffs[XType(hdL)][XType(hdR)]( hdL, l, hdR, r ))
*/
Int (*TabReduceCoeffs[T_VAR][T_VAR]) (
    Bag, Int, Bag, Int );

Int CantReduceCoeffs (Bag hdL, Int l, Bag hdR, Int r)
                    	        /* left polynomial coeffs                  */
                    	        /* degree plus one of left polynomial      */
                    	        /* right polynomial coeffs                 */
                    	        /* degree plus one of right polynomial     */
{
    Error( "<l> and <r> must be vectors over a common field", 0, 0);
    return 0;
}

Int ReduceCoeffsListxListx (Bag hdL, Int l, Bag hdR, Int r)
                    	        /* left polynomial coeffs                  */
                    	        /* degree plus one of left polynomial      */
                    	        /* right polynomial coeffs                 */
                    	        /* degree plus one of right polynomial     */
{
    Bag           hdLL;   /* one element of <hdL>                    */
    Bag           hdRR;   /* one element of <hdR>                    */
    Bag           hdCC;   /* temp element                            */
    Bag           hdTT;   /* temp element                            */
    Int                i,  k;  /* loop variables                          */

    /* <r> must be none zero                                               */
    if ( r == 0 ) {
	Error( "<r> must be non zero", 0, 0 );
	return 0;
    }
    hdCC = PROD( ELML_LIST( hdR, 1 ), INT_TO_HD(0) );
    for ( ;  0 < r;  r-- ) {
	hdTT = ELMR_LIST( hdR, r );
	if ( EQ( hdCC, hdTT ) == HdFalse )
	    break;
    }
    if ( r == 0 ) {
	Error( "<r> must be non zero", 0, 0 );
	return 0;
    }

    /* if <l> is trivial return                                            */
    if ( l == 0 )
	return 0;

    /* compute the remainder                                               */
    for ( i = l-r;  0 <= i;  i-- )
    {
	hdLL = ELMR_LIST( hdL, i+r );
	if ( hdLL != INT_TO_HD(0) ) {
	    hdRR = ELML_LIST( hdR, r );
	    hdCC = QUO( hdLL, hdRR );
	    for ( k = r;  0 < k;  k-- )
	    {
		hdRR = ELML_LIST( hdR, k );
		hdTT = PROD( hdCC, hdRR );
		hdLL = ELMR_LIST( hdL, i+k );
		hdLL = DIFF( hdLL, hdTT );
		ASS_LIST( hdL, i+k, hdLL );
	    }
	}
    }
    return ( l < r ) ? l : r-1;
}

Int ReduceCoeffsVecFFEVecFFE (Bag hdL, Int l, Bag hdR, Int r)
                    	        /* left polynomial coeffs                  */
                    	        /* degree plus one of left polynomial      */
                    	        /* right polynomial coeffs                 */
                    	        /* degree plus one of right polynomial     */
{
    TypFFE *            ptL;    /* coeffs vectors of left polynomial       */
    TypFFE *            ptLL;   /* coeffs vectors of left polynomial       */
    TypFFE *            ptR;    /* coeffs vectors of right polynomial      */
    TypFFE *            ptRR;   /* coeffs vectors of right polynomial      */
    TypFFE *            f;      /* the finite field                        */
    TypFFE              o;      /* size of <f> minus one                   */
    TypFFE              c,q,t;  /* temps for FFEs                          */
    register Int       i,k;    /* loop variables                          */

    /* <r> must be none zero                                               */
    ptR = (TypFFE*)( PTR_BAG(hdR) + 1 ) + (r-1);
    while ( 0 < r && *ptR == 0 ) {
	r--;  ptR--;
    }
    if ( r == 0 )
	Error( "<r> must be non zero", 0, 0 );

    /* if <l> is trivial return                                            */
    if ( l == 0 )
	return 0;

    /* get a common field                                                  */
    f = (TypFFE*) PTR_BAG( UnifiedFieldVecFFE( hdL, hdR ) );
    o = *f;

    /* set up all the pointers                                             */
    ptL = (TypFFE*)( PTR_BAG(hdL) + 1 );
    ptR = (TypFFE*)( PTR_BAG(hdR) + 1 ) + (r-1);

    /* get the leading coefficient of <hdR> * -1, <c> is never zero        */
    c = ( o%2 == 1 ? (*ptR) : ( (*ptR) <= o/2 ? (*ptR)+o/2 : (*ptR)-o/2 ) );

    /* compute the remainder                                               */
    for ( i = l-r;  0 <= i;  i-- ) {

	/* if <ptLL> already ends with a zero, continue                    */
	ptLL = ptL + (i-1+r);
        if ( *ptLL == 0 )
	    continue;

        /* compute quotient of leading coefficients                        */
	q = ( c <= (*ptLL) ? (*ptLL)-c+1 : o-c+1+(*ptLL) );

        /* reduce <ptLL> by <q> * <ptRR>                                   */
	for ( k = r, ptRR = ptR;  0 < k;  k--, ptLL--, ptRR-- ) {
	    if ( *ptRR == 0 )
		continue;
	    t = ( q-1 <= o-(*ptRR) ) ? q-1+(*ptRR) : q-1-(o-(*ptRR));
	    if ( *ptLL == 0 )
		*ptLL = t;
	    else if ( *ptLL <= t )
		*ptLL = PROD_FF( *ptLL, f[t-(*ptLL)+1], f );
	    else
		*ptLL = PROD_FF( t, f[(*ptLL)-t+1], f );
	}
    }
    return ( l < r ) ? l : r-1;
}


/****************************************************************************
**
*F  FunReduceCoeffs( <hdCall> ) . . . . . .  internal function 'ReduceCoeffs'
**
**  'FunReduceCoeffs' implements 'ReduceCoeffs( <l>, <r> )'
*/
Bag FunReduceCoeffs (Bag hdCall)
{
    Bag       	hdL;
    Int                l;
    Bag           hdR;
    Int                r;

    /* check arguments                                                     */
    if ( GET_SIZE_BAG( hdCall ) != 3 * SIZE_HD )
        return Error( "usage: ReduceCoeffs( <l>, <r> )", 0, 0 );
    hdL = EVAL( PTR_BAG(hdCall)[1] );
    hdR = EVAL( PTR_BAG(hdCall)[2] );
    l   = LEN_LIST(hdL);
    r   = LEN_LIST(hdR);

    /* jump through the table                                              */
    TabReduceCoeffs[XType(hdL)][XType(hdR)]( hdL, l, hdR, r );
    return HdVoid;
}


/****************************************************************************
**
*F  FunRemainderCoeffs( <hdCall> )  . . . internal function 'RemainderCoeffs'
**
**  'FunRemainderCoeffs' implements 'RemainderCoeffs( <l>, <r> )'
*/
Bag FunRemainderCoeffs (Bag hdCall)
{
    Bag       	hdL;
    Int                l;
    Bag           hdR;
    Int                r;

    /* check arguments                                                     */
    if ( GET_SIZE_BAG( hdCall ) != 3 * SIZE_HD )
        return Error( "usage: ReduceCoeffs( <l>, <r> )", 0, 0 );
    hdL = EVAL( PTR_BAG(hdCall)[1] );
    hdR = EVAL( PTR_BAG(hdCall)[2] );
    l   = LEN_LIST(hdL);
    r   = LEN_LIST(hdR);

    /* jump through the table                                              */
    hdL = Copy(hdL);
    TabReduceCoeffs[XType(hdL)][XType(hdR)]( hdL, l, hdR, r );
    return hdL;
}


/****************************************************************************
**
*F  REDUCE_COEFFS_MOD( <hdL>, <l>, <hdR>, <r>, <hdN> )  reduce <hdL> by <hdR>
**
#define REDUCE_COEFFS_MOD( hdL, l, hdR, r ) \
    (TabReduceCoeffsMod[XType(hdL)][XType(hdR)]( hdL, l, hdR, r, hdN ))
*/
Int (*TabReduceCoeffsMod[T_VAR][T_VAR]) (
    Bag, Int, Bag, Int, Bag );

Int CantReduceCoeffsMod (Bag hdL, Int l, Bag hdR, Int r, Bag hdP)
                    	        /* left polynomial coeffs                  */
                    	        /* degree plus one of left polynomial      */
                    	        /* right polynomial coeffs                 */
                    	        /* degree plus one of right polynomial     */
                    	        /* modulo                                  */
{
    Error( "<l> and <r> must be vectors over a common field", 0, 0);
    return 0;
}

Int ReduceCoeffsModListxListx (Bag hdL, Int l, Bag hdR, Int r, Bag hdP)
                    	        /* left polynomial coeffs                  */
                    	        /* degree plus one of left polynomial      */
                    	        /* right polynomial coeffs                 */
                    	        /* degree plus one of right polynomial     */
                    	        /* modulo                                  */
{
    Bag       hdLL;       /* one element of <hdL>                    */
    Bag       hdRR;       /* one element of <hdR>                    */
    Bag       hdCC;       /* temp element                            */
    Bag       hdTT;       /* temp element                            */
    Bag       hdQ;        /* <hdP> / 2                               */
    Int            i,  k;      /* loop variables                          */

    /* <r> must be none zero                                               */
    if ( r == 0 ) {
	Error( "<r> must be non zero", 0, 0 );
	return 0;
    }
    hdCC = PROD( ELML_LIST( hdR, 1 ), INT_TO_HD(0) );
    for ( ;  0 < r;  r-- ) {
	hdTT = ELMR_LIST( hdR, r );
	if ( EQ( hdCC, hdTT ) == HdFalse )
	    break;
    }
    if ( r == 0 ) {
	Error( "<r> must be non zero", 0, 0 );
	return 0;
    }

    /* catch the trivial cases (<l> or <hdP> equal zero)                   */
    if ( l == 0 )
	return 0;
    if ( EQ( hdP, INT_TO_HD(0) ) == HdTrue ) {
	for ( i = l;  0 < i;  i-- )
	    ASS_LIST( hdL, i, INT_TO_HD(0) );
	return 0;
    }

    /* if <hdP> is negative reduce into <hdP>/2 and <hdP>                  */
    if ( LT( hdP, INT_TO_HD(0) ) == HdTrue ) {
	hdP = DIFF( INT_TO_HD(0), hdP );
	hdQ = SUM( hdP, INT_TO_HD(1) );
	hdQ = QuoInt( hdQ, INT_TO_HD(2) );
	hdQ = DIFF( hdQ, INT_TO_HD(1) );
    }
    else
	hdQ = 0;

    /* if <r> is zero or too big reduce modulo <hP>                        */
    if ( l < r ) {
	for ( i = l;  0 < i;  i-- ) {
	    hdTT = MOD( ELMF_LIST(hdL,i), hdP );
	    if ( hdQ != 0 && LT( hdQ, hdTT ) == HdTrue )
		hdTT = DIFF( hdTT, hdP );
	    ASS_LIST( hdL, i, hdTT );
	}
	r = l+1;
    }

    /* otherwise, compute the remainder                                    */
    else
	for ( i = l-r;  0 <= i;  i-- ) {
	    hdLL = ELM_LIST( hdL, i+r );
	    if ( hdLL != INT_TO_HD(0) ) {
		hdRR = ELM_LIST( hdR, r );
		hdCC = QUO( hdLL, hdRR );
		hdCC = MOD( hdCC, hdP );
		for ( k = r;  0 < k;  k-- ) {
		    hdRR = ELM_LIST( hdR, k );
		    hdTT = PROD( hdCC, hdRR );
		    hdLL = ELM_LIST( hdL, i+k );
		    hdLL = DIFF( hdLL, hdTT );
		    hdLL = MOD( hdLL, hdP );
		    if ( hdQ != 0 && LT( hdQ, hdLL ) == HdTrue )
			hdLL = DIFF( hdLL, hdP );
		    SET_ELM_PLIST( hdL, i+k, hdLL );
		}
	    }
	}

    /* and return the new length                                           */
    return r-1;
}

Int ReduceCoeffsModListx (Bag hdL, Int l, Bag hdR, Int r, Bag hdP)
                    	        /* left polynomial coeffs                  */
                    	        /* degree plus one of left polynomial      */
                    	        /* void                                    */
                    	        /* zero                                    */
                    	        /* modulo                                  */
{
    Bag       hdTT;       /* temp element                            */
    Bag       hdQ;        /* <hdP> / 2                               */
    Int            i;          /* loop variables                          */

    /* catch the trivial cases (<l> or <hdP> equal zero)                   */
    if ( l == 0 )
	return 0;
    if ( EQ( hdP, INT_TO_HD(0) ) == HdTrue ) {
	for ( i = l;  0 < i;  i-- )
	    ASS_LIST( hdL, i, INT_TO_HD(0) );
	return 0;
    }

    /* if <hdP> is negative reduce into <hdP>/2 and <hdP>                  */
    if ( LT( hdP, INT_TO_HD(0) ) == HdTrue ) {
	hdP = DIFF( INT_TO_HD(0), hdP );
	hdQ = SUM( hdP, INT_TO_HD(1) );
	hdQ = QuoInt( hdQ, INT_TO_HD(2) );
	hdQ = DIFF( hdQ, INT_TO_HD(1) );
    }
    else
	hdQ = 0;

    /* reduce modulo <hP>                                                  */
    for ( i = l;  0 < i;  i-- ) {
	hdTT = MOD( ELMF_LIST(hdL,i), hdP );
	if ( hdQ != 0 && LT( hdQ, hdTT ) == HdTrue )
	    hdTT = DIFF( hdTT, hdP );
	ASS_LIST( hdL, i, hdTT );
    }
    return l;
}


/****************************************************************************
**
*F  FunReduceCoeffsMod( <hdCall> )  . . . internal function 'ReduceCoeffsMod'
**
**  'FunReduceCoeffsMod' implements 'ReduceCoeffsMod( <l>, <r>, <p> )'
*/
Bag FunReduceCoeffsMod (Bag hdCall)
{
    Bag       	hdL;
    Int                l;
    Bag           hdR;
    Int                r;
    Bag           hdN;

    /* check arguments                                                     */
    if ( 4*SIZE_HD < GET_SIZE_BAG(hdCall) || GET_SIZE_BAG(hdCall) < 3*SIZE_HD )
        return Error( "usage: ReduceCoeffsMod( <l>, <r>, <p> )", 0, 0 );
    hdL = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_SIZE_BAG(hdCall) == 4*SIZE_HD ) {
	hdR = EVAL( PTR_BAG(hdCall)[2] );
	hdN = EVAL( PTR_BAG(hdCall)[3] );
	r   = LEN_LIST(hdR);
    }
    else {
	hdR = HdVoid;
	r   = 0;
	hdN = EVAL( PTR_BAG(hdCall)[2] );
    }
    l   = LEN_LIST(hdL);

    /* jump through the table                                              */
    TabReduceCoeffsMod[XType(hdL)][XType(hdR)]( hdL, l, hdR, r, hdN );
    return HdVoid;
}


/****************************************************************************
**
*F  FunPowerModCoeffs( <hdCall> ) . . . .  internal function 'PowerModCoeffs'
**
**  'FunPowerModCoeffs' implements 'PowerModCoeffs( <g>, <n>, <r> )'
*/
Bag (*TabPowerModCoeffsInt[T_VAR][T_VAR]) (
    Bag, Bag, Bag );

Bag (*TabPowerModCoeffsLInt[T_VAR][T_VAR]) (
    Bag, Bag, Bag );

Bag FunPowerModCoeffs (Bag hdCall)
{
    Bag       	hdG;
    Bag       	hdE;
    Bag       	hdR;

    /* check arguments                                                     */
    if ( GET_SIZE_BAG( hdCall ) != 4 * SIZE_HD )
        return Error( "usage: PowerModCoeffs( <g>, <exp>, <r> )", 0, 0 );
    hdG = EVAL( PTR_BAG(hdCall)[1] );
    hdE = EVAL( PTR_BAG(hdCall)[2] );
    hdR = EVAL( PTR_BAG(hdCall)[3] );
    if ( GET_TYPE_BAG(hdE)!=T_INTPOS && ( GET_TYPE_BAG(hdE)!=T_INT || HD_TO_INT(hdE)<1 ) )
	return Error( "<exp> must be a positive integer", 0, 0 );

    /* jump through the table                                              */
    if ( GET_TYPE_BAG(hdE) == T_INT )
	return TabPowerModCoeffsInt [XType(hdG)][XType(hdR)](hdG, hdE, hdR);
    else
	return TabPowerModCoeffsLInt[XType(hdG)][XType(hdR)](hdG, hdE, hdR);
}

Bag PowerModListxIntListx (Bag hdG, Bag hdE, Bag hdR)
                    	        /* polynomial coeffs                       */
                    	        /* exponent                                */
                    	        /* modulus                                 */
{
    Bag	        hdP;    /* result                                  */
    Bag           hdR1;   /* temporary storage for multiplication    */
    Bag           hdR2;   /* temporary storage for multiplication    */
    Int                g;      /* length of <hdG>                         */
    Int                r;      /* length of <hdR>                         */
    Int                p;      /* length of <hdP>                         */
    UInt       exp;    /* value of <hdE>                          */
    UInt       i;      /* loop variable                           */

    /* <hdR> must be none zero                                             */
    r = LEN_LIST(hdR);
    if ( r == 0 ) {
	Error( "<r> must be non zero", 0, 0 );
	return 0;
    }
    hdR1 = PROD( ELML_LIST( hdR, 1 ), INT_TO_HD(0) );
    for ( ;  0 < r;  r-- ) {
	hdR2 = ELMR_LIST( hdR, r );
	if ( EQ( hdR1, hdR2 ) == HdFalse )
	    break;
    }
    if ( r == 0 ) {
	Error( "<r> must be non zero", 0, 0 );
	return 0;
    }

    /* copy <hdG> and reduce it modulo <hdR>                               */
    hdG = Copy(hdG);
    g   = REDUCE_COEFFS( hdG, LEN_LIST(hdG), hdR, r );

    /* if <hdG> is already trivial,  return                                */
    for ( ;  0 < g;  g-- ) {
	hdR2 = ELMR_LIST( hdG, g );
	if ( EQ( hdR1, hdR2 ) == HdFalse )
	    break;
    }
    if ( g == 0 ) {
	hdG = NewBag( T_LIST, SIZE_PLEN_PLIST(0) );
	SET_LEN_PLIST( hdG, 0 );
	return hdG;
    }

    /* allocate storage for multiplication                                 */
    hdR1 = Copy(hdG);
    hdR2 = NewBag( T_LIST, SIZE_PLEN_PLIST(2*r) );

    /* we need at most two times the <r>                                   */
    PLAIN_LIST(hdG);
    PLAIN_LIST(hdR1);
    Resize( hdR1, SIZE_PLEN_PLIST(2*r) );
    SET_LEN_PLIST( hdR1, 2*r );

    /* compute the power using left to right repeated squaring             */
    i   = 1 << 31;
    p   = 0;
    hdP = 0;
    exp = HD_TO_INT(hdE);
    while ( 1 < i ) {
	if ( hdP ) {

	    /* use table directly in order to avoid changing the types     */
	    p = TabMultiplyCoeffs[T_LISTX][T_LISTX]( hdR2, hdP, p, hdP, p );
	    p = TabReduceCoeffs[T_LISTX][T_LISTX]( hdR2, p, hdR, r );
	    hdR1 = hdP;  hdP = hdR2;  hdR2 = hdR1;
	}
	i = i / 2;
	if ( i <= exp ) {
	    if ( hdP ) {

		/* use table directly in order to avoid changing the types */
		p = TabMultiplyCoeffs[T_LISTX][T_LISTX](hdR2,hdP,p,hdG,g);
		p = TabReduceCoeffs[T_LISTX][T_LISTX](hdR2,p,hdR,r);
		hdR1 = hdP;  hdP = hdR2;  hdR2 = hdR1;
	    }
	    else {
		hdP = hdR1;
		p = g;
	    }
	    exp = exp - i;
	}
    }

    /* resize result and return                                            */
    SET_LEN_PLIST( hdP, p );
    return hdP;
}

Bag PowerModVecFFEIntVecFFE (Bag hdG, Bag hdE, Bag hdR)
                    	        /* polynomial coeffs                       */
                    	        /* exponent                                */
                    	        /* modulus                                 */
{
    Bag	        hdP;    /* result                                  */
    Bag           hdR1;   /* temporary storage for multiplication    */
    Bag           hdR2;   /* temporary storage for multiplication    */
    Int                g;      /* length of <hdG>                         */
    Int                r;      /* length of <hdR>                         */
    Int                p;      /* length of <hdP>                         */
    UInt       exp;    /* value of <hdE>                          */
    UInt       i;      /* loop variable                           */

    /* <hdR> must be none zero                                             */
    r = LEN_VECFFE(hdR);
    if ( r == 0 ) {
	Error( "<r> must be non zero", 0, 0 );
	return 0;
    }
    for ( ;  0 < r;  r-- ) {
	if ( VAL_VECFFE( hdR, r ) != 0 )
	    break;
    }
    if ( r == 0 ) {
	Error( "<r> must be non zero", 0, 0 );
	return 0;
    }

    /* copy <hdG> and reduce it modulo <hdR>                               */
    hdG = Copy(hdG);
    g   = REDUCE_COEFFS( hdG, LEN_LIST(hdG), hdR, r );

    /* if <hdG> is already trivial,  return                                */
    for ( ;  0 < g;  g-- ) {
	if ( VAL_VECFFE( hdG, g ) != 0 )
	    break;
    }
    if ( g == 0 ) {
	hdG = NewBag( T_LIST, SIZE_PLEN_PLIST(0) );
	SET_LEN_PLIST( hdG, 0 );
	return hdG;
    }

    /* allocate storage for multiplication                                 */
    hdR1 = Copy(hdG);
    hdR2 = NewBag( T_VECFFE, SIZE_PLEN_VECFFE(2*r) );
    SET_FLD_VECFFE( hdR2, FLD_VECFFE(hdR1) );

    /* we need at most two times the <r>                                   */
    Resize( hdR1, SIZE_PLEN_VECFFE(2*r) );

    /* compute the power using left to right repeated squaring             */
    i   = 1 << 31;
    p   = 0;
    hdP = 0;
    exp = HD_TO_INT(hdE);
    while ( 1 < i ) {
	if ( hdP ) {

	    /* use table directly in order to avoid changing the types     */
	    p = TabMultiplyCoeffs[T_VECFFE][T_VECFFE](hdR2,hdP,p,hdP,p);
	    p = TabReduceCoeffs[T_VECFFE][T_VECFFE]( hdR2, p, hdR, r );
	    hdR1 = hdP;  hdP = hdR2;  hdR2 = hdR1;
	}
	i = i / 2;
	if ( i <= exp ) {
	    if ( hdP ) {

		/* use table directly in order to avoid changing the types */
		p = TabMultiplyCoeffs[T_VECFFE][T_VECFFE](hdR2,hdP,p,hdG,g);
		p = TabReduceCoeffs[T_VECFFE][T_VECFFE](hdR2,p,hdR,r);
		hdR1 = hdP;  hdP = hdR2;  hdR2 = hdR1;
	    }
	    else {
		hdP = hdR1;
		p = g;
	    }
	    exp = exp - i;
	}
    }

    /* resize result and return                                            */
    SET_LEN_VECFFE( hdP, p );
    return hdP;
}

Bag PowerModListxLIntListx (Bag hdG, Bag hdE, Bag hdR)
                    	        /* polynomial coeffs                       */
                    	        /* exponent                                */
                    	        /* modulus polynomial coeffs               */
{
    Bag	        hdP;    /* result                                  */
    Bag           hdR1;   /* temporary storage for multiplication    */
    Bag           hdR2;   /* temporary storage for multiplication    */
    TypDigit            e;      /* one digit of <hdE>                      */
    Int                g;      /* length of <hdG>                         */
    Int                r;      /* length of <hdR>                         */
    Int                p;      /* length of <hdP>                         */
    UInt       i;      /* loop variable                           */
    Int                l;      /* loop variable                           */

    /* <hdR> must be none zero                                             */
    r = LEN_LIST(hdR);
    if ( r == 0 ) {
	Error( "<r> must be non zero", 0, 0 );
	return 0;
    }
    hdR1 = PROD( ELML_LIST( hdR, 1 ), INT_TO_HD(0) );
    for ( ;  0 < r;  r-- ) {
	hdR2 = ELMR_LIST( hdR, r );
	if ( EQ( hdR1, hdR2 ) == HdFalse )
	    break;
    }
    if ( r == 0 ) {
	Error( "<r> must be non zero", 0, 0 );
	return 0;
    }

    /* copy <hdG> and reduce it modulo <hdR>                               */
    hdG = Copy(hdG);
    g   = REDUCE_COEFFS( hdG, LEN_LIST(hdG), hdR, r );

    /* if <hdG> is already trivial,  return                                */
    for ( ;  0 < g;  g-- ) {
	hdR2 = ELMR_LIST( hdG, g );
	if ( EQ( hdR1, hdR2 ) == HdFalse )
	    break;
    }
    if ( g == 0 ) {
	hdG = NewBag( T_LIST, SIZE_PLEN_PLIST(0) );
	SET_LEN_PLIST( hdG, 0 );
	return hdG;
    }

    /* allocate storage for multiplication                                 */
    hdR1 = Copy(hdG);
    hdR2 = NewBag( T_LIST, SIZE_PLEN_PLIST(2*r) );

    /* we need at most two times the <r>                                   */
    PLAIN_LIST(hdR1);
    PLAIN_LIST(hdG);
    Resize( hdR1, SIZE_PLEN_PLIST(2*r) );
    SET_LEN_PLIST( hdR1, 2*r );

    /* compute the power using left to right repeated squaring             */
    hdP = 0;
    p   = 0;
    for ( l = GET_SIZE_BAG(hdE)/sizeof(TypDigit)-1;  0 <= l;  l-- ) {
	i = NUM_TO_UINT(1) << (8*sizeof(TypDigit));
	e = ((TypDigit*) PTR_BAG(hdE))[l];
	while ( 1 < i ) {
	    if ( hdP ) {

		/* use table directly in order to avoid changing the types */
		p = TabMultiplyCoeffs[T_LISTX][T_LISTX](hdR2,hdP,p,hdP,p);
		p = TabReduceCoeffs[T_LISTX][T_LISTX]( hdR2, p, hdR, r );
		hdR1 = hdP;  hdP = hdR2;  hdR2 = hdR1;
	    }
	    i = i / 2;
	    if ( i <= e ) {
		if ( hdP ) {
		    /* use table directly                                  */
                   p = TabMultiplyCoeffs[T_LISTX][T_LISTX](hdR2,hdP,p,hdG,g);
		   p = TabReduceCoeffs[T_LISTX][T_LISTX]( hdR2, p, hdR, r );
		   hdR1 = hdP;  hdP = hdR2;  hdR2 = hdR1;
		}
		else {
		    hdP = hdR1;
		    p = g;
		}
		e = e - i;
	    }
	}
    }

    /* resize result and return                                            */
    SET_LEN_PLIST( hdP, p );
    return hdP;
}

Bag PowerModVecFFELIntVecFFE (Bag hdG, Bag hdE, Bag hdR)
                    	        /* polynomial coeffs                       */
                    	        /* exponent                                */
                    	        /* modulus                                 */
{
    Bag	        hdP;    /* result                                  */
    Bag           hdR1;   /* temporary storage for multiplication    */
    Bag           hdR2;   /* temporary storage for multiplication    */
    TypDigit            e;      /* one digit of <hdE>                      */
    Int                g;      /* length of <hdG>                         */
    Int                r;      /* length of <hdR>                         */
    Int                p;      /* length of <hdP>                         */
    UInt       i;      /* loop variable                           */
    Int                l;      /* loop variable                           */

    /* <hdR> must be none zero                                             */
    r = LEN_VECFFE(hdR);
    if ( r == 0 ) {
	Error( "<r> must be non zero", 0, 0 );
	return 0;
    }
    for ( ;  0 < r;  r-- ) {
	if ( VAL_VECFFE( hdR, r ) != 0 )
	    break;
    }
    if ( r == 0 ) {
	Error( "<r> must be non zero", 0, 0 );
	return 0;
    }

    /* copy <hdG> and reduce it modulo <hdR>                               */
    hdG = Copy(hdG);
    g   = REDUCE_COEFFS( hdG, LEN_LIST(hdG), hdR, r );

    /* if <hdG> is already trivial,  return                                */
    for ( ;  0 < g;  g-- ) {
	if ( VAL_VECFFE( hdG, g ) != 0 )
	    break;
    }
    if ( g == 0 ) {
	hdG = NewBag( T_LIST, SIZE_PLEN_PLIST(0) );
	SET_LEN_PLIST( hdG, 0 );
	return hdG;
    }

    /* allocate storage for multiplication                                 */
    hdR1 = Copy(hdG);
    hdR2 = NewBag( T_VECFFE, SIZE_PLEN_VECFFE(2*r) );
    SET_FLD_VECFFE( hdR2, FLD_VECFFE(hdR1) );

    /* we need at most two times the <r>                                   */
    Resize( hdR1, SIZE_PLEN_VECFFE(2*r) );

    /* compute the power using left to right repeated squaring             */
    hdP = 0;
    p   = 0;
    for ( l = GET_SIZE_BAG(hdE)/sizeof(TypDigit)-1;  0 <= l;  l-- ) {
	i = NUM_TO_UINT(1) << (8*sizeof(TypDigit));
	e = ((TypDigit*) PTR_BAG(hdE))[l];
	while ( 1 < i ) {
	    if ( hdP ) {

		/* use table directly in order to avoid changing the types */
		p = TabMultiplyCoeffs[T_VECFFE][T_VECFFE](hdR2,hdP,p,hdP,p);
		p = TabReduceCoeffs[T_VECFFE][T_VECFFE]( hdR2, p, hdR, r );
		hdR1 = hdP;  hdP = hdR2;  hdR2 = hdR1;
	    }
	    i = i / 2;
	    if ( i <= e ) {
		if ( hdP ) {
		 /* use table directly                                     */
                 p = TabMultiplyCoeffs[T_VECFFE][T_VECFFE](hdR2,hdP,p,hdG,g);
	         p = TabReduceCoeffs[T_VECFFE][T_VECFFE]( hdR2, p, hdR, r );
	         hdR1 = hdP;  hdP = hdR2;  hdR2 = hdR1;
		}
		else {
		    hdP = hdR1;
		    p = g;
		}
		e = e - i;
	    }
	}
    }

    /* resize result and return                                            */
    SET_LEN_VECFFE( hdP, p );
    return hdP;
}

Bag CantPowerModCoeffs (Bag hdG, Bag hdE, Bag hdR)
                    	        /* polynomial coeffs                       */
                    	        /* exponent                                */
                    	        /* modulus polynomial coeffs               */
{
    Error( "<g> and <r> must be vectors over a common field", 0, 0);
    return 0;
}



/****************************************************************************
**
*F  InitPolynom() . . . . . . . . . . . . . .  initialize the polynom package
*/
void InitPolynom (void)
{
    Int            type1, type2;    	/* loop variables                  */

    /* install tables for gap functions                                    */
    for ( type1 = T_VOID;  type1 < T_VAR;  type1++ ) {
        TabNormalizeCoeffs[type1] = CantNormalizeCoeffs;
        TabShrinkCoeffs   [type1] = CantShrinkCoeffs;
        TabShiftedCoeffs  [type1] = CantShiftedCoeffs;
	for ( type2 = T_VOID;  type2 < T_VAR;  type2++ ) {
	    TabAddCoeffs         [type1][type2] = CantAddCoeffs;
	    TabMultiplyCoeffs    [type1][type2] = CantMultiplyCoeffs;
	    TabProductCoeffs     [type1][type2] = CantProductCoeffs;
	    TabProductCoeffsMod  [type1][type2] = CantProductCoeffsMod;
	    TabReduceCoeffs      [type1][type2] = CantReduceCoeffs;
	    TabReduceCoeffsMod   [type1][type2] = CantReduceCoeffsMod;
	    TabPowerModCoeffsInt [type1][type2] = CantPowerModCoeffs;
	    TabPowerModCoeffsLInt[type1][type2] = CantPowerModCoeffs;
	}
    }

    TabNormalizeCoeffs[T_LISTX ] = NormalizeCoeffsListx;
    TabNormalizeCoeffs[T_VECTOR] = NormalizeCoeffsListx;
    TabNormalizeCoeffs[T_VECFFE] = NormalizeCoeffsVecFFE;

    TabShrinkCoeffs[T_LISTX ] = ShrinkCoeffsListx;
    TabShrinkCoeffs[T_VECTOR] = ShrinkCoeffsListx;
    TabShrinkCoeffs[T_VECFFE] = ShrinkCoeffsVecFFE;

    TabShiftedCoeffs  [T_LISTX ] = ShiftedCoeffsListx;
    TabShiftedCoeffs  [T_VECTOR] = ShiftedCoeffsListx;
    TabShiftedCoeffs  [T_VECFFE] = ShiftedCoeffsVecFFE;

    TabAddCoeffs[T_LISTX ][T_LISTX ] = AddCoeffsListxListx;
    TabAddCoeffs[T_LISTX ][T_VECTOR] = AddCoeffsListxListx;
    TabAddCoeffs[T_LISTX ][T_VECFFE] = AddCoeffsListxVecFFE;
    TabAddCoeffs[T_VECTOR][T_LISTX ] = AddCoeffsListxListx;
    TabAddCoeffs[T_VECTOR][T_VECTOR] = AddCoeffsListxListx;
    TabAddCoeffs[T_VECTOR][T_VECFFE] = AddCoeffsListxListx;
    TabAddCoeffs[T_VECFFE][T_LISTX ] = AddCoeffsListxListx;
    TabAddCoeffs[T_VECFFE][T_VECTOR] = AddCoeffsListxListx;
    TabAddCoeffs[T_VECFFE][T_VECFFE] = AddCoeffsVecFFEVecFFE;

    TabMultiplyCoeffs[T_LISTX ][T_LISTX ] = MultiplyCoeffsListxListx;
    TabMultiplyCoeffs[T_LISTX ][T_VECTOR] = MultiplyCoeffsListxListx;
    TabMultiplyCoeffs[T_LISTX ][T_VECFFE] = MultiplyCoeffsListxListx;
    TabMultiplyCoeffs[T_VECTOR][T_LISTX ] = MultiplyCoeffsListxListx;
    TabMultiplyCoeffs[T_VECTOR][T_VECTOR] = MultiplyCoeffsListxListx;
    TabMultiplyCoeffs[T_VECTOR][T_VECFFE] = MultiplyCoeffsListxListx;
    TabMultiplyCoeffs[T_VECFFE][T_LISTX ] = MultiplyCoeffsListxListx;
    TabMultiplyCoeffs[T_VECFFE][T_VECTOR] = MultiplyCoeffsListxListx;
    TabMultiplyCoeffs[T_VECFFE][T_VECFFE] = MultiplyCoeffsVecFFEVecFFE;

    TabProductCoeffs[T_LISTX ][T_LISTX ] = ProductCoeffsListxListx;
    TabProductCoeffs[T_LISTX ][T_VECTOR] = ProductCoeffsListxListx;
    TabProductCoeffs[T_LISTX ][T_VECFFE] = ProductCoeffsListxListx;
    TabProductCoeffs[T_VECTOR][T_LISTX ] = ProductCoeffsListxListx;
    TabProductCoeffs[T_VECTOR][T_VECTOR] = ProductCoeffsListxListx;
    TabProductCoeffs[T_VECTOR][T_VECFFE] = ProductCoeffsListxListx;
    TabProductCoeffs[T_VECFFE][T_LISTX ] = ProductCoeffsListxListx;
    TabProductCoeffs[T_VECFFE][T_VECTOR] = ProductCoeffsListxListx;
    TabProductCoeffs[T_VECFFE][T_VECFFE] = ProductCoeffsVecFFEVecFFE;

    TabProductCoeffsMod[T_LISTX ][T_LISTX ] = ProductCoeffsModListxListx;
    TabProductCoeffsMod[T_LISTX ][T_VECTOR] = ProductCoeffsModListxListx;
    TabProductCoeffsMod[T_LISTX ][T_VECFFE] = ProductCoeffsModListxListx;
    TabProductCoeffsMod[T_VECTOR][T_LISTX ] = ProductCoeffsModListxListx;
    TabProductCoeffsMod[T_VECTOR][T_VECTOR] = ProductCoeffsModListxListx;
    TabProductCoeffsMod[T_VECTOR][T_VECFFE] = ProductCoeffsModListxListx;
    TabProductCoeffsMod[T_VECFFE][T_LISTX ] = ProductCoeffsModListxListx;
    TabProductCoeffsMod[T_VECFFE][T_VECTOR] = ProductCoeffsModListxListx;
    TabProductCoeffsMod[T_VECFFE][T_VECFFE] = ProductCoeffsModListxListx;

    TabReduceCoeffs[T_LISTX ][T_LISTX ] = ReduceCoeffsListxListx;
    TabReduceCoeffs[T_LISTX ][T_VECTOR] = ReduceCoeffsListxListx;
    TabReduceCoeffs[T_LISTX ][T_VECFFE] = ReduceCoeffsListxListx;
    TabReduceCoeffs[T_VECTOR][T_LISTX ] = ReduceCoeffsListxListx;
    TabReduceCoeffs[T_VECTOR][T_VECTOR] = ReduceCoeffsListxListx;
    TabReduceCoeffs[T_VECTOR][T_VECFFE] = ReduceCoeffsListxListx;
    TabReduceCoeffs[T_VECFFE][T_LISTX ] = ReduceCoeffsListxListx;
    TabReduceCoeffs[T_VECFFE][T_VECTOR] = ReduceCoeffsListxListx;
    TabReduceCoeffs[T_VECFFE][T_VECFFE] = ReduceCoeffsVecFFEVecFFE;

    TabReduceCoeffsMod[T_LISTX ][T_LISTX ] = ReduceCoeffsModListxListx;
    TabReduceCoeffsMod[T_LISTX ][T_VECTOR] = ReduceCoeffsModListxListx;
    TabReduceCoeffsMod[T_LISTX ][T_VECFFE] = ReduceCoeffsModListxListx;
    TabReduceCoeffsMod[T_LISTX ][T_VOID  ] = ReduceCoeffsModListx;
    TabReduceCoeffsMod[T_VECTOR][T_LISTX ] = ReduceCoeffsModListxListx;
    TabReduceCoeffsMod[T_VECTOR][T_VECTOR] = ReduceCoeffsModListxListx;
    TabReduceCoeffsMod[T_VECTOR][T_VECFFE] = ReduceCoeffsModListxListx;
    TabReduceCoeffsMod[T_VECTOR][T_VOID  ] = ReduceCoeffsModListx;
    TabReduceCoeffsMod[T_VECFFE][T_LISTX ] = ReduceCoeffsModListxListx;
    TabReduceCoeffsMod[T_VECFFE][T_VECTOR] = ReduceCoeffsModListxListx;
    TabReduceCoeffsMod[T_VECFFE][T_VECFFE] = ReduceCoeffsModListxListx;
    TabReduceCoeffsMod[T_VECFFE][T_VOID  ] = ReduceCoeffsModListx;

    TabPowerModCoeffsInt[T_LISTX ][T_LISTX ] = PowerModListxIntListx;
    TabPowerModCoeffsInt[T_LISTX ][T_VECTOR] = PowerModListxIntListx;
    TabPowerModCoeffsInt[T_LISTX ][T_VECFFE] = PowerModListxIntListx;
    TabPowerModCoeffsInt[T_VECTOR][T_LISTX ] = PowerModListxIntListx;
    TabPowerModCoeffsInt[T_VECTOR][T_VECTOR] = PowerModListxIntListx;
    TabPowerModCoeffsInt[T_VECTOR][T_VECFFE] = PowerModListxIntListx;
    TabPowerModCoeffsInt[T_VECFFE][T_LISTX ] = PowerModListxIntListx;
    TabPowerModCoeffsInt[T_VECFFE][T_VECTOR] = PowerModListxIntListx;
    TabPowerModCoeffsInt[T_VECFFE][T_VECFFE] = PowerModVecFFEIntVecFFE;

    TabPowerModCoeffsLInt[T_LISTX ][T_LISTX ] = PowerModListxLIntListx;
    TabPowerModCoeffsLInt[T_LISTX ][T_VECTOR] = PowerModListxLIntListx;
    TabPowerModCoeffsLInt[T_LISTX ][T_VECFFE] = PowerModListxLIntListx;
    TabPowerModCoeffsLInt[T_VECTOR][T_LISTX ] = PowerModListxLIntListx;
    TabPowerModCoeffsLInt[T_VECTOR][T_VECTOR] = PowerModListxLIntListx;
    TabPowerModCoeffsLInt[T_VECTOR][T_VECFFE] = PowerModListxLIntListx;
    TabPowerModCoeffsLInt[T_VECFFE][T_LISTX ] = PowerModListxLIntListx;
    TabPowerModCoeffsLInt[T_VECFFE][T_VECTOR] = PowerModListxLIntListx;
    TabPowerModCoeffsLInt[T_VECFFE][T_VECFFE] = PowerModVecFFELIntVecFFE;

    /* install the internal functions                                      */
    InstIntFunc( "ShiftedCoeffs",   	FunShiftedCoeffs    );
    InstIntFunc( "NormalizeCoeffs",     FunNormalizeCoeffs  );
    InstIntFunc( "ShrinkCoeffs",        FunShrinkCoeffs  );
    InstIntFunc( "AddCoeffs",       	FunAddCoeffs 	    );
    InstIntFunc( "SumCoeffs",       	FunSumCoeffs 	    );
    InstIntFunc( "ProductCoeffs",   	FunProductCoeffs    );
    InstIntFunc( "ProductCoeffsMod",	FunProductCoeffsMod );
    InstIntFunc( "ReduceCoeffs",    	FunReduceCoeffs     );
    InstIntFunc( "RemainderCoeffs", 	FunRemainderCoeffs  );
    InstIntFunc( "ReduceCoeffsMod", 	FunReduceCoeffsMod  );
    InstIntFunc( "PowerModCoeffs",  	FunPowerModCoeffs   );
}
