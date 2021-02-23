/****************************************************************************
**
*A  finfield.c                  GAP source                      Werner Nickel
**                                                         & Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file contains the functions to compute with  finite fields elements.
**
**  Finite fields  are an  important   domain in computational   group theory
**  because the classical matrix groups are defined over those finite fields.
**  In GAP we support finite fields with  up to 65536 elements, larger fields
**  can be realized as polynomial domains over smaller fields.
**
**  Elements in finite fields are represented as  bags with two entries.  The
**  first is the handle of  the finite field bag, its  format is shown below.
**  The other, called value,  determines which element in this field this is.
**
**  If the  value is 0,  then the element is the  zero from the finite field.
**  Otherwise the integer is the logarithm of this  element with respect to a
**  fixed generator of the multiplicative group of the finite field plus one.
**  In the following desriptions we denote this generator always with $z$, it
**  is an element of order $o-1$, where $o$ is the order of the finite field.
**  Thus 1 corresponds to $z^{1-1} = z^0 = 1$, i.e., the  one from the field.
**  Likewise 2 corresponds to $z^{2-1} = z^1 = z$, i.e., the root itself.
**
**  This representation  makes multiplication very easy,  we only have to add
**  the values and subtract 1 , because  $z^{a-1} * z^{b-1} = z^{(a+b-1)-1}$.
**  Addition is reduced to * by the formula $z^a +  z^b = z^b * (z^{a-b}+1)$.
**  This makes it neccessary to know the successor $z^a + 1$ of every value.
**
**  The finite  field  bag contains the successor   for every nonzero  value,
**  i.e., '(TypFFE*)PTR_BAG(hdField)[a]' is   the successor of the   element 'a',
**  i.e, it is the logarithm of $z^{a-1} + 1$.  This list is  usually  called
**  the Zech-Logarithm table.  The zeroth entry in the finite  field  bag  is
**  the order of the finite field minus one.
**
*/

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "scanner.h"             /* reading of tokens and printing  */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */
#include        "list.h"                /* generic lists package           */
#include        "plist.h"               /* plain list package              */

#include        "finfield.h"            /* declaration part of the package */


/****************************************************************************
**
*T  TypFFE  . . . . . . . . . . . type of the values of finite field elements
**
**  'TypFFE' is the type used to store the values of finite  field  elements,
**  i.e., the logarithm of the element with respect to  the  root  plus  one.
**
**  Finite fields are restricted to contain at most 65536 elements, so we can
**  put the logarithm of any finite field element into an 'unsigned short'.
**
**  It is possible to change this to 'unsigned long'  to  allow  fields  with
**  more than than 65536 elements.  All the below macros and have been  coded
**  in such a way that they work without problems.  The exception is 'POW_FF'
**  which will only work if the product of integers of type 'TypFFE' does not
**  cause an overflow.  And of course the successor table stored for a finite
**  field will become quite large for fields with more than 65536 elements.
**
**  'TypFFE' is defined in the declaration file of this package as follows:
**
typedef unsigned short  TypFFE;
*/


/****************************************************************************
**
*F  SUCC_FF(<hdFF>) . . . . . . . . . . . . . successor table of finite field
**
**  'SUCC_FF' returns a pointer to the successor table  of  the  finite field
**  <hdFF>.
**
**  Note that 'SUCC_FF' is a macro, so do not call  it  with  arguments  that
**  sideeffects.
**
**  'SUCC_FF' is defined in the declaration part of this package as follows:
**
#define SUCC_FF(FF)             ((TypFFE*)PTR_BAG(FF))
*/


/****************************************************************************
**
*F  SIZE_FF(<hdFF>) . . . . . . . . . . . . . . . . .  size of a finite field
**
**  'SIZE_FF' returns the size of the finite field <hdFF>.
**
**  Note that 'SIZE_FF' is a macro, so do not call  it  with  arguments  that
**  have sideeffects.
**
**  'SIZE_FF' is defined in the declaration part of this package as follows:
**
#define SIZE_FF(FF)             (*SUCC_FF(FF)+1)
*/
/*N 1993/02/01 martin this should go away, but for the moment lets keep it */
#define ORD_FF(FFE)             (SIZE_FF(FLD_FFE(FFE)))


/****************************************************************************
**
*F  FLD_FFE(<hdFFE>)  . . . . . . . . . . . . field of a finite field element
**
**  'FLD_FFE' returns the handle of the finite field over  which  the  finite
**  field element <hdFFE> is represented.
**
**  Note that 'FLD_FFE' is a macro, so do not call  it  with  arguments  that
**  have sideeffects.
**
**  'FLD_FFE' is defined in the declaration part of this package as follows:
**
#define FLD_FFE(FFE)            (PTR_BAG(FFE)[0])
*/


/****************************************************************************
**
*F  SET_FLD_FFE(<hdFFE>,<hdFF>) . . . set the field of a finite field element
**
**  'SET_FLD_FFE' sets the field of  the  finite  field  element  <hdFFE>  to
**  <hdFF>.
**
**  Note that 'SET_FLD_FFE' is a macro, so do not call it with arguments that
**  have sideeffects.
**
**  'SET_FLD_FFE' is defined in the  declaration  part  of  this  package  as
**  follows:
**
#define SET_FLD_FFE(FFE,FF)     (FLD_FFE(FFE)=(FF))
*/


/****************************************************************************
**
*F  VAL_FFE(<hdFFE>)  . . . . . . . . . . . . value of a finite field element
**
**  'VAL_FFE' returns the value of the finite field element <hdFFE>.  That is
**  if <hdFFE> is $0_F$,'VAL_FFE' returns 0;  if <hdFFE> is $1_F$,  'VAL_FFE'
**  returns 1; if <hdFFE> is the primitive generator $z$,  'VAL_FFE'  returns
**  2; and otherwise if <hdFFE> is $z^i$, 'VAL_FFE' returns '<i>+1'.
**
**  Note that 'VAL_FFE' is a macro, so do not call  it  with  arguments  that
**  have sideeffects.
**
**  'VAL_FFE' is defined in the declaration part of this package as follows:
**
#define VAL_FFE(FFE)            (*(TypFFE*)(PTR_BAG(FFE)+1))
*/


/****************************************************************************
**
*F  SET_VAL_FFE(<hdFFE>,<val>)  . . . set the value of a finite field element
**
**  'SET_VAL_FFE' sets the value of the finite field element <hdFFE>  to  the
**  value <val>.  Note that no checking is performed, whether <val>  lies  in
**  the allowed range.
**
**  Note that 'SET_VAL_FFE' is a macro, so do not call it with arguments that
**  have sideeffects.
**
**  'SET_VAL_FFE' is defined in the  declaration  part  of  this  package  as
**  follows:
**
#define SET_VAL_FFE(FFE,VAL)    (VAL_FFE(FFE)=(VAL))
*/


/****************************************************************************
**
*F  SUM_FF(<a>,<b>,<f>) . . . . . . . . . . . . .  sum of finite field values
**
**  'SUM_FF' returns the sum of the two  finite  field  values  <a>  and  <b>
**  from the finite field pointed to by the pointer <f>.
**
**  Note that  'SUM_FF'  may only be used if  the operands are represented in
**  the same finite field.  If you want to add two elements where one lies in
**  a subfield of the other use 'SumFFE'.
**
**  Use 'SUM_FF' only with arguments that are  variables  or  array elements,
**  because it is a macro and arguments with sideeffects will behave strange,
**  and because it is a complex macro so most C compilers will  be  upset  by
**  complex arguments.  In particular, do not use 'SUM_FF(a,NEG_FF(b,f),f)'.
**
**  If either operand is 0, the sum is just the other operand.
**  If $a <= b$ we have
**  $a + b ~ z^{a-1}+z^{b-1} = z^{a-1} * (z^{(b-1)-(a-1)}+1) ~ a * f[b-a+1]$,
**  otherwise we have
**  $a + b ~ z^{b-1}+z^{a-1} = z^{b-1} * (z^{(a-1)-(b-1)}+1) ~ b * f[a-b+1]$.
**
**  'SUM_FF' is defined in the declaration part of this package as follows:
**
#define SUM_FF(a,b,f)   ( (a)==0 || (b)==0 ? (a)+(b) :\
                          ( (a)<=(b) ? PROD_FF(a,(f)[(b)-(a)+1],f) :\
                                       PROD_FF(b,(f)[(a)-(b)+1],f) ) )
*/


/****************************************************************************
**
*F  NEG_FF(<a>,<f>) . . . . . . . . . . . . .  negative of finite field value
**
**  'NEG_FF' returns the negative of the finite  field  value  <a>  from  the
**  finite field pointed to by the pointer <f>.
**
**  Use 'NEG_FF' only with arguments that are  variables  or  array elements,
**  because it is a macro and arguments with sideeffects will behave strange,
**  and because it is a complex macro so most C compilers will  be  upset  by
**  complex arguments.  In particular, do not use 'NEG_FF(PROD_FF(a,b,f),f)'.
**
**  If the characteristic is 2, every element is its  own  additive  inverse.
**  Otherwise note that $z^{o-1} = 1 = -1^2$ so $z^{(o-1)/2} = 1^{1/2} = -1$.
**  If $a <= (o-1)/2$ we have
**  $-a ~ -1 * z^{a-1} = z^{(o-1)/2} * z^{a-1} = z^{a+(o-1)/2-1} ~ a+(o-1)/2$
**  otherwise we have
**  $-a ~ -1 * z^{a-1} = z^{a+(o-1)/2-1} = z^{a+(o-1)/2-1-(o-1)} ~ a-(o-1)/2$
**
**  'NEG_FF' is defined in the declaration part of this package as follows:
**
#define NEG_FF(a,f)     ( (a)==0 ? 0 :\
                          ( *(f)%2==1 ? a :\
                            ( (a)<=*(f)/2 ? (a)+*(f)/2 : (a)-*(f)/2 ) ) )
*/


/****************************************************************************
**
*F  PROD_FF(<a>,<b>,<f>)  . . . . . . . . . . . product of finite field value
**
**  'PROD_FF' returns the product of the two finite field values <a> and  <b>
**  from the finite field pointed to by the pointer <f>.
**
**  Note that  'PROD_FF'  may only be used if the operands are represented in
**  the same finite field.  If you want to multiply two  elements  where  one
**  lies in a subfield of the other use 'ProdFFE'.
**
**  Use 'PROD_FF' only with arguments that are variables  or  array elements,
**  because it is a macro and arguments with sideeffects will behave strange,
**  and because it is a complex macro so most C compilers will  be  upset  by
**  complex arguments.  In particular, do not use 'NEG_FF(PROD_FF(a,b,f),f)'.
**
**  If one of the values is 0 the product is 0.
**  If $a+b <= o$ we have $a * b ~ z^{a-1} * z^{b-1} = z^{(a+b-1)-1} ~ a+b-1$
**  otherwise   we   have $a * b ~ z^{(a+b-2)-(o-1)} = z^{(a+b-o)-1} ~ a+b-o$
**
**  'PROD_FF' is defined in the declaration part of this package as follows:
**
#define PROD_FF(a,b,f)  ( (a)==0 || (b)==0 ? 0 :\
                          ( (a)-1<=*(f)-(b) ? (a)-1+(b) : (a)-1-(*(f)-(b)) ))
*/


/****************************************************************************
**
*F  QUO_FF(<a>,<b>,<f>) . . . . . . . . . . . quotient of finite field values
**
**  'QUO_FF' returns the quotient of the two finite field values <a> and  <b>
**  from the finite field pointed to by the pointer <f>.
**
**  Note that  'QUO_FF'  may only be used if the operands are represented  in
**  the same finite field.  If you want to  divide  two  elements  where  one
**  lies in a subfield of the other use 'QuoFFE'.
**
**  Use 'QUO_FF' only with arguments that are  variables  or  array elements,
**  because it is a macro and arguments with sideeffects will behave strange,
**  and because it is a complex macro so most C compilers will  be  upset  by
**  complex arguments.  In particular, do not use 'NEG_FF(PROD_FF(a,b,f),f)'.
**
**  A division by 0 is an error,  and dividing 0 by a nonzero value gives  0.
**  If $0 <= a-b$ we have  $a / b ~ z^{a-1} / z^{b-1} = z^{a-b+1-1} ~ a-b+1$,
**  otherwise   we   have  $a / b ~ z^{a-b+1-1}  =  z^{a-b+(o-1)}   ~ a-b+o$.
**
**  'QUO_FF' is defined in the declaration part of this package as follows:
**
#define QUO_FF(a,b,f)   ( (a)==0 ? 0 :\
                          ( (b)<=(a) ? (a)-(b)+1 : *(f)-(b)+1+(a) ) )
*/


/****************************************************************************
**
*F  POW_FF(<a>,<n>,<f>) . . . . . . . . . . . . power of a finite field value
**
**  'POW_FF' returns the <n>th power of the finite field value <a>  from  the
**  the finite field pointed to by the pointer <f>.
**
**  Note that 'POW_FF' may only be used if the right operand is an integer in
**  the range $0..order(f)-1$.
**
**  Finally 'POW_FF' may only be used if the product of two integers  of  the
**  size of 'TypFFE' does not cause an overflow, i.e.  only  if  'TypFFE'  is
**  'unsigned short'.
**
**  Note that 'POW_FF' is a macro, so do not  call  it  with  arguments  that
**  have sideeffects.  For optimal performance put the operands in  registers
**  before calling 'POW_FF'.
**
**  If the finite field element is 0 the power is also 0, otherwise  we  have
**  $a^n ~ (z^{a-1})^n = z^{(a-1)*n} = z^{(a-1)*n % (o-1)} ~ (a-1)*n % (o-1)$
**
**  'POW_FF' is defined in the declaration part of this package as follows:
**
#define POW_FF(a,n,f)   ( (n)==0 ? 1 :\
                          ( (a)==0 ? 0 : (((a)-1) * (n)) % *(f) + 1 ) )
*/


/****************************************************************************
**
*V  HdFields . . . . . . . . . . . . . . . . table of finite field generators
**
**  'HdFields' is a list of generators of finite fields.  'Z' stores in  this
**  list generators for all finite fields constructed  so  far,  so  that  it
**  need not construct a finite field twice.
*/
Bag       HdFields;


/****************************************************************************
**
*V  Pols  . . . . . . . . . . .  list of Conway polynomials for finite fields
**
**  'Pols' is a list of  Conway  polynomials  for  finite  fields.  The  even
**  entries are the proper prime powers, odd entries  are  the  corresponding
**  conway polynomials.
*/
UInt   Pols [] = {
       4, 1+2,
       8, 1+2,
      16, 1+2,
      32, 1  +4,
      64, 1+2  +8+16,
     128, 1+2,
     256, 1  +4+8+16,
     512, 1      +16,
    1024, 1+2+4+8   +32+64,
    2048, 1  +4,
    4096, 1+2  +8   +32+64+128,
    8192, 1+2  +8+16,
   16384, 1    +8   +32   +128,
   32768, 1  +4  +16+32,
   65536, 1  +4+8   +32,
       9,  2 +2*3,
      27,  1 +2*3,
      81,  2           +2*27,
     243,  1 +2*3,
     729,  2 +2*3 +1*9       +2*81,
    2187,  1      +2*9,
    6561,  2 +2*3 +2*9       +1*81 +2*243,
   19683,  1 +1*3 +2*9 +2*27,
   59049,  2 +1*3            +2*81 +2*243 +2*729,
      25,  2 +4*5,
     125,  3 +3*5,
     625,  2 +4*5 +4*25,
    3125,  3 +4*5,
   15625,  2      +1*25 +4*125 +1*625,
      49,  3 +6*7,
     343,  4      +6*49,
    2401,  3 +4*7 +5*49,
   16807,  4 +1*7,
     121,  2 + 7*11,
    1331,  9 + 2*11,
   14641,  2 +10*11 +8*121,
     169,  2 +12*13,
    2197, 11 + 2*13,
   28561,  2 +12*13 +3*169,
     289,  3 +16*17,
    4913, 14 + 1*17,
     361,  2 +18*19,
    6859, 17 + 4*19,
     529,  5 +21*23,
   12167, 18 + 2*23,
     841,  2 +24*29,
   24389, 27 + 2*29,
     961,  3 +29*31,
   29791, 28 + 1*31,
    1369,  2 +33*37,
   50653, 35 + 6*37,
    1681,  6 + 38* 41,
    1849,  3 + 42* 43,
    2209,  5 + 45* 47,
    2809,  2 + 49* 53,
    3481,  2 + 58* 59,
    3721,  2 + 60* 61,
    4489,  2 + 63* 67,
    5041,  7 + 69* 71,
    5329,  5 + 70* 73,
    6241,  3 + 78* 79,
    6889,  2 + 82* 83,
    7921,  3 + 82* 89,
    9409,  5 + 96* 97,
   10201,  2 + 97*101,
   10609,  5 +102*103,
   11449,  2 +103*107,
   11881,  6 +108*109,
   12769,  3 +101*113,
   16129,  3 +126*127,
   17161,  2 +127*131,
   18769,  3 +131*137,
   19321,  2 +138*139,
   22201,  2 +145*149,
   22801,  6 +149*151,
   24649,  5 +152*157,
   26569,  2 +159*163,
   27889,  5 +166*167,
   29929,  2 +169*173,
   32041,  2 +172*179,
   32761,  2 +177*181,
   36481, 19 +190*191,
   37249,  5 +192*193,
   38809,  2 +192*197,
   39601,  3 +193*199,
   44521,  2 +207*211,
   49729,  3 +221*223,
   51529,  2 +220*227,
   52441,  6 +228*229,
   54289,  3 +232*233,
   57121,  7 +237*239,
   58081,  7 +238*241,
   63001,  6 +242*251,
};


/****************************************************************************
**
*F  RootFiniteField( <q> )  . .  construct the finite field with <q> elements
**
**  'RootFiniteField' returns the handle of the  primitive root of the finite
**  field with <q> elements.  If <q> is not  a power of  a prime or is larger
**  than $2^{16}$  'RootFiniteField' returns the handle 0.    If the field is
**  already constructed,   i.e., is  in  the list   'HdFields' it  is  simply
**  returned.  Otherwise 'RootFiniteField' constructs  this finite  field and
**  remembers it in 'HdFields'.
*/
Bag       RootFiniteField (UInt q)
{
    Bag           hdZ;            /* handle of the primitive root    */
    Bag           hdFF;           /* handle of the finite field bag  */
    TypFFE              * ff;           /* pointer to the finite field bag */
    Bag           hdInd;          /* handle of the temp index bag    */
    TypFFE              * ind;          /* pointer to the temp index bag   */
    UInt       p;              /* characteristic of the field     */
    UInt       poly;           /* Conway polynomial of extension  */
    UInt       i, l, f, n, e;  /* loop variables                  */

    /* check the prime power                                               */
    if ( q <= 1 || 65536 < q )
        return 0;

    /* search through the finite field table                               */
    for ( i = 0;  i < GET_SIZE_BAG(HdFields)/SIZE_HD;  ++i ) {
        if ( ORD_FF( PTR_BAG(HdFields)[i] ) == q )
            return PTR_BAG(HdFields)[i];
    }

    /* compute the prime and check that <q> is a prime power               */
    if ( q % 2 == 0 )  p = 2;
    else for ( p = 3; q % p != 0; p += 2 ) ;
    i = q;
    while ( i % p == 0 )  { i = i / p; }
    if ( i != 1 )
        return 0;

    /* allocate a bag for the finite field and one for a temporary         */
    hdFF  = NewBag( T_FF, q * sizeof(TypFFE) );
    hdInd = NewBag( T_FF, q * sizeof(TypFFE) );
    ff    = SUCC_FF( hdFF );
    ind   = SUCC_FF( hdInd );

    /* if q is a prime find the smallest primitive root $e$, use $x - e$   */
    /*N 14-Feb-90 martin this is likely to explode if 'TypFFE' is 'ulong'  */
    /*N 14-Feb-90 martin there are few dumber ways to find primitive roots */
    if ( q == p ) {
        for ( e = 1, i = 1; i != p-1; ++e ) {
            for ( f = e, i = 1; f != 1; ++i )
                f = (f * e) % p;
        }
        poly = p-(e-1);
    }

    /* otherwise look up the polynomial used to construct this field       */
    else {
        for ( i = 0; Pols[i] != q; i += 2 ) ;
        poly = Pols[i+1];
    }

    /* construct 'ind' such that 'e = x^(ind[e]-1) % poly' for every e     */
    /*N 14-Feb-90 martin this is likely to explode if 'TypFFE' is 'ulong'  */
    ind[ 0 ] = 0;
    for ( e = 1, n = 0; n < q-1; ++n ) {
        ind[ e ] = n + 1;
        /* e =p*e mod poly =x*e mod poly =x*x^n mod poly =x^{n+1} mod poly */
        if ( p != 2 ) {
            f = p * (e % (q/p));  l = ((p-1) * (e / (q/p))) % p;  e = 0;
            for ( i = 1; i < q; i *= p )
                e = e + i * ((f/i + l * (poly/i)) % p);
        }
        else {
            if ( 2*e & q )  e = 2*e ^ poly ^ q;
            else            e = 2*e;
        }
    }

    /* construct 'ff' such that 'x^(n-1) + 1 = x^(ff[n]-1)' for every n    */
    ff[ 0 ] = q-1;
    for ( e = 1, f = p-1; e < q; ++e ) {
        if ( e < f ) {
            ff[ ind[e] ] = ind[ e+1 ];
        }
        else {
            ff[ ind[e] ] = ind[ e+1-p ];
            f += p;
        }
    }

    /* create the new generator                                            */
    hdZ = NewBag( T_FFE, SIZE_HD + sizeof(TypFFE) );
    SET_FLD_FFE( hdZ, hdFF );
    if ( q == 2 ) SET_VAL_FFE(hdZ,1);
    else          SET_VAL_FFE(hdZ,2);

    /* enter it into the fields bag                                        */
    Resize( HdFields, GET_SIZE_BAG(HdFields) + SIZE_HD );
    SET_BAG(HdFields,  GET_SIZE_BAG(HdFields)/SIZE_HD - 1 ,  hdZ );

    /* return the primitive root of the finite field                       */
    return hdZ;
}


/****************************************************************************
**
*F  CommonFF( <hdL>, <hdR> )  . . . . . find a field containing both elements
**
**  'CommonFF'  returns  the handle of the   bag of  the  smallest field that
**  contains the two finite  field elements <hdL> and <hdR>.   If this is not
**  possible, either  because <hdL>  and   <hdR> lie in fields  of  different
**  characteristic, or  because the  smallest field  has more   than $2^{16}$
**  elements, then 'CommonFF'   returns the handle  0.   Note that it  may be
**  neccessary for 'CommonFF'  to create  a  new finite field, triggering   a
**  garbage collection.
*/
Bag       CommonFF (Bag hdL, Bag hdR)
{
    Bag           hdZ;            /* root of the finite field        */
    UInt       q;              /* order  of the common field      */
    UInt       p;              /* char   of the common field      */
    UInt       d;              /* degree of the common field      */
    UInt       v;              /* value of an element             */
    UInt       ql, qr;         /* order of the minimal fields     */
    UInt       dl, dr;         /* degrees of the minimal fields   */

    /* get and check the characteristic                                    */
    q = ORD_FF(hdL);
    if ( q % 2 == 0 )  p = 2;
    else for ( p = 3; q % p != 0; p++ ) ;
    if ( ORD_FF(hdR) % p != 0 )
        return 0;

    /* get the order of the minimal field in which the left operand lies   */
    v = VAL_FFE(hdL);
    if ( v == 0 )  return PTR_BAG(hdR)[0];
    q = ORD_FF(hdL);
    ql = p;
    dl = 1;
    while ( (q-1) % (ql-1) != 0 || (v-1) % ((q-1)/(ql-1)) != 0 ) {
        ql *= p;
        dl += 1;
    }

    /* get the order of the minimal field in which the right operand lies  */
    v = VAL_FFE(hdR);
    if ( v == 0 )  return PTR_BAG(hdL)[0];
    q = ORD_FF(hdR);
    qr = p;
    dr = 1;
    while ( (q-1) % (qr-1) != 0 || (v-1) % ((q-1)/(qr-1)) != 0 ) {
        qr *= p;
        dr += 1;
    }

    /* get the degree of the smallest common superfield                    */
    q = ql;
    d = dl;
    while ( d % dr != 0 ) {
        q *= ql;
        d += dl;
    }
    if ( (  2 <= p && 17 <= d) || (  3 <= p && 11 <= d)
      || (  5 <= p &&  7 <= d) || (  7 <= p &&  6 <= d)
      || ( 11 <= p &&  5 <= d) || ( 17 <= p &&  4 <= d)
      || ( 41 <= p &&  3 <= d) || (257 <= p &&  2 <= d) )
        return (Bag)1;

    /* call 'RootFiniteField' to construct this field                      */
    hdZ = RootFiniteField( q );
    if ( hdZ == 0 )
        return 0;
    return FLD_FFE(hdZ);
}


/****************************************************************************
**
*F  EvFFE( <hdFFE> )  . . . . . . . . . . . . evaluate a finite field element
**
**  'EvFFE' returns  the value of  the finite field  element  <hdFFE>.  Since
**  finite field elements  are constants  and  thus selfevaluating this  just
**  returns <hdFFE>.
*/
Bag       EvFFE (Bag hdFFE)
{
    return hdFFE;
}


/****************************************************************************
**
*F  SumFFE( <hdL>, <hdR> )  . . . . . . . . . .  sum of finite field elements
**
**  'SumFFE' returns the sum of the  two  finite  field  elements  <hdL>  and
**  <hdR>.  The sum is represented over the field  over  which  the  operands
**  are represented, even if it lies in a much smaller field.
**
**  If one of the operands  is an integer  it is converted  into the field of
**  the  other  operand  before  the  addition.  If  one of the  elements  is
**  represented over a subfield of the field over which  the other element is
**  represented it is lifted into the larger field before the addition.
**
**  Is called from the 'Sum' binop, so both operands are already  evaluated.
**
**  'SumFFE' just does the conversions mentioned above  and  then  calls  the
**  macro 'SUM_FF' to do the actual addition.
*/
Bag       SumFFE (Bag hdL, Bag hdR)
{
    Bag           hdX;            /* handle of the sum               */
    TypFFE              x;              /* value of the sum                */
    Bag           hdFF;           /* handle of  the finite field bag */
    TypFFE              * field;        /* pointer to the finite field bag */
    TypFFE              l;              /* value of the left operand       */
    TypFFE              r;              /* value of the right operand      */

    /* sum of an integer and a finite field element                        */
    if ( GET_TYPE_BAG(hdL) == T_INT ) {
        hdFF = FLD_FFE(hdR);
        field = SUCC_FF( hdFF );
        r = (HD_TO_INT(hdL) % (Int)ORD_FF(hdR) + ORD_FF(hdR)) % ORD_FF(hdR);
        if ( r == 0 )  l = 0;
        else for ( l = 1; 1 < r; --r )  l = (l == 0 ? 1 : field[l]);
        r = VAL_FFE(hdR);
    }

    /* sum of a finite field element and an integer                        */
    else if ( GET_TYPE_BAG(hdR) == T_INT ) {
        hdFF = FLD_FFE(hdL);
        field = SUCC_FF( hdFF );
        l = (HD_TO_INT(hdR) % (Int)ORD_FF(hdL) + ORD_FF(hdL)) % ORD_FF(hdL);
        if ( l == 0 )  r = 0;
        else for ( r = 1; 1 < l; --l )  r = (r == 0 ? 1 : field[r]);
        l = VAL_FFE(hdL);
    }

    /* sum of two finite field element in the same finite field            */
    else if ( FLD_FFE(hdL) == FLD_FFE(hdR) ) {
        hdFF = FLD_FFE(hdL);
        field = SUCC_FF( hdFF );
        l = VAL_FFE(hdL);
        r = VAL_FFE(hdR);
    }

    /* sum of a finite field element and an element from a subfield        */
    else if ( ORD_FF(hdL)%ORD_FF(hdR)==0 && ORD_FF(hdL)%(ORD_FF(hdR)-1)<=1 ){
        hdFF = FLD_FFE(hdL);
        field = SUCC_FF( hdFF );
        l = VAL_FFE(hdL);
        if ( VAL_FFE(hdR) == 0 )  r = 0;
        else r = (ORD_FF(hdL)-1) / (ORD_FF(hdR)-1) * (VAL_FFE(hdR)-1) + 1;
    }

    /* sum of a finite field element and an element from a superfield      */
    else if ( ORD_FF(hdR)%ORD_FF(hdL)==0 && ORD_FF(hdR)%(ORD_FF(hdL)-1)<=1 ){
        hdFF = FLD_FFE(hdR);
        field = SUCC_FF( hdFF );
        if ( VAL_FFE(hdL) == 0 )  l = 0;
        else l = (ORD_FF(hdR)-1) / (ORD_FF(hdL)-1) * (VAL_FFE(hdL)-1) + 1;
        r = VAL_FFE(hdR);
    }

    /* else try to find a common finite field                              */
    else {
        hdFF = CommonFF( hdL, hdR );
        if ( hdFF == 0 )
            return Error(
                "Finite field +: operands must have the same characteristic",
                         0,0);
        else if ( hdFF == (Bag)1 )
            return Error(
                "Finite field +: smallest common superfield to large",
                         0,0);
        field = SUCC_FF( hdFF );
        if ( VAL_FFE(hdL) == 0 )  l = 0;
        else  l = (VAL_FFE(hdL)-1) * field[0] / (ORD_FF(hdL)-1) + 1;
        if ( VAL_FFE(hdR) == 0 )  r = 0;
        else  r = (VAL_FFE(hdR)-1) * field[0] / (ORD_FF(hdR)-1) + 1;
    }

    /* compute the sum                                                     */
    x             = SUM_FF( l, r, field );
    hdX           = NewBag( T_FFE, SIZE_HD + sizeof(TypFFE) );
    SET_FLD_FFE( hdX, hdFF );
    SET_VAL_FFE( hdX, x );
    return  hdX;
}


/****************************************************************************
**
*F  DiffFFE( <hdL>, <hdR> ) . . . . . . . difference of finite field elements
**
**  'DiffFFE' returns the difference of the two finite field  elements  <hdL>
**  and <hdR>.  The difference is represented over the field over  which  the
**  operands are represented, even if it lies in a much smaller field.
**
**  If one of the operands  is an integer  it is converted  into the field of
**  the other  operand before the subtraction.  If one  of  the  elements  is
**  represented over a subfield of the field over which  the other element is
**  represented it is lifted into the larger field before the subtraction.
**
**  Is called from the 'Diff' binop, so both operands are already  evaluated.
**
**  'DiffFFE' just does the conversions mentioned above and  then  calls  the
**  macros 'NEG_FF' and 'SUM_FF' to do the actual subtraction.
*/
Bag       DiffFFE (Bag hdL, Bag hdR)
{
    Bag           hdX;            /* handle of the difference        */
    TypFFE              x;              /* value of the difference         */
    Bag           hdFF;           /* handle of  the finite field bag */
    TypFFE              * field;        /* pointer to the finite field bag */
    TypFFE              l;              /* value of the left operand       */
    TypFFE              r;              /* value of the right operand      */

    /* difference of an integer and a finite field element                 */
    if ( GET_TYPE_BAG(hdL) == T_INT ) {
        hdFF = FLD_FFE(hdR);
        field = SUCC_FF( hdFF );
        r = (HD_TO_INT(hdL) % (Int)ORD_FF(hdR) + ORD_FF(hdR)) % ORD_FF(hdR);
        if ( r == 0 )  l = 0;
        else for ( l = 1; 1 < r; --r )  l = (l == 0 ? 1 : field[l]);
        r = VAL_FFE(hdR);
    }

    /* difference of a finite field element and an integer                 */
    else if ( GET_TYPE_BAG(hdR) == T_INT ) {
        hdFF = FLD_FFE(hdL);
        field = SUCC_FF( hdFF );
        l = (HD_TO_INT(hdR) % (Int)ORD_FF(hdL) + ORD_FF(hdL)) % ORD_FF(hdL);
        if ( l == 0 )  r = 0;
        else for ( r = 1; 1 < l; --l )  r = (r == 0 ? 1 : field[r]);
        l = VAL_FFE(hdL);
    }

    /* difference of two finite field element in the same finite field     */
    else if ( FLD_FFE(hdL) == FLD_FFE(hdR) ) {
        hdFF = FLD_FFE(hdL);
        field = SUCC_FF( hdFF );
        l = VAL_FFE(hdL);
        r = VAL_FFE(hdR);
    }

    /* difference of a finite field element and an element from a subfield */
    else if ( ORD_FF(hdL)%ORD_FF(hdR)==0 && ORD_FF(hdL)%(ORD_FF(hdR)-1)<=1 ){
        hdFF = FLD_FFE(hdL);
        field = SUCC_FF( hdFF );
        l = VAL_FFE(hdL);
        if ( VAL_FFE(hdR) == 0 )  r = 0;
        else r = (ORD_FF(hdL)-1) / (ORD_FF(hdR)-1) * (VAL_FFE(hdR)-1) + 1;
    }

    /* difference of a finite field element and an element from a superfld */
    else if ( ORD_FF(hdR)%ORD_FF(hdL)==0 && ORD_FF(hdR)%(ORD_FF(hdL)-1)<=1 ){
        hdFF = FLD_FFE(hdR);
        field = SUCC_FF( hdFF );
        if ( VAL_FFE(hdL) == 0 )  l = 0;
        else l = (ORD_FF(hdR)-1) / (ORD_FF(hdL)-1) * (VAL_FFE(hdL)-1) + 1;
        r = VAL_FFE(hdR);
    }

    /* else try to find a common finite field                              */
    else {
        hdFF = CommonFF( hdL, hdR );
        if ( hdFF == 0 )
            return Error(
                "Finite field -: operands must have the same characteristic",
                         0,0);
        else if ( hdFF == (Bag)1 )
            return Error(
                "Finite field -: smallest common superfield to large",
                         0,0);
        field = SUCC_FF( hdFF );
        if ( VAL_FFE(hdL) == 0 )  l = 0;
        else  l = (VAL_FFE(hdL)-1) * field[0] / (ORD_FF(hdL)-1) + 1;
        if ( VAL_FFE(hdR) == 0 )  r = 0;
        else  r = (VAL_FFE(hdR)-1) * field[0] / (ORD_FF(hdR)-1) + 1;
    }

    /* compute the difference                                              */
    x             = NEG_FF( r, field );
    x             = SUM_FF( l, x, field );
    hdX           = NewBag( T_FFE, SIZE_HD + sizeof(TypFFE) );
    SET_FLD_FFE( hdX, hdFF );
    SET_VAL_FFE( hdX, x );
    return  hdX;
}


/****************************************************************************
**
*F  ProdFFE( <hdL>, <hdR> ) . . . . . . . .  product of finite field elements
**
**  'ProdFFE' returns the product of the two finite field elements <hdL>  and
**  <hdR>.  The product is represented over the field over which the operands
**  are represented, even if it lies in a much smaller field.
**
**  If one of the operands  is an integer  it is converted  into the field of
**  the other  operand before the multiplication.  If  one of the elements is
**  represented over a subfield of the field over which  the other element is
**  represented it is lifted into the larger field before the multiplication.
**
**  Is called from the 'Prod' binop, so both operands are already  evaluated.
**
**  'ProdFFE' just does the conversions mentioned above and  then  calls  the
**  macro 'PROD_FF' to do the actual multiplication.
*/
Bag       ProdFFE (Bag hdL, Bag hdR)
{
    Bag           hdX;            /* handle of the product           */
    TypFFE              x;              /* value of the product            */
    Bag           hdFF;           /* handle of  the finite field bag */
    TypFFE              * field;        /* pointer to the finite field bag */
    TypFFE              l;              /* value of the left operand       */
    TypFFE              r;              /* value of the right operand      */

    /* product of an integer and a finite field element                    */
    if ( GET_TYPE_BAG(hdL) == T_INT ) {
        hdFF = FLD_FFE(hdR);
        field = SUCC_FF( hdFF );
        r = (HD_TO_INT(hdL) % (Int)ORD_FF(hdR) + ORD_FF(hdR)) % ORD_FF(hdR);
        if ( r == 0 )  l = 0;
        else for ( l = 1; 1 < r; --r )  l = (l == 0 ? 1 : field[l]);
        r = VAL_FFE(hdR);
    }

    /* product of a finite field element and an integer                    */
    else if ( GET_TYPE_BAG(hdR) == T_INT ) {
        hdFF = FLD_FFE(hdL);
        field = SUCC_FF( hdFF );
        l = (HD_TO_INT(hdR) % (Int)ORD_FF(hdL) + ORD_FF(hdL)) % ORD_FF(hdL);
        if ( l == 0 )  r = 0;
        else for ( r = 1; 1 < l; --l )  r = (r == 0 ? 1 : field[r]);
        l = VAL_FFE(hdL);
    }

    /* product of two finite field element in the same finite field        */
    else if ( FLD_FFE(hdL) == FLD_FFE(hdR) ) {
        hdFF = FLD_FFE(hdL);
        field = SUCC_FF( hdFF );
        l = VAL_FFE(hdL);
        r = VAL_FFE(hdR);
    }

    /* product of a finite field element and an element from a subfield    */
    else if ( ORD_FF(hdL)%ORD_FF(hdR)==0 && ORD_FF(hdL)%(ORD_FF(hdR)-1)<=1 ){
        hdFF = FLD_FFE(hdL);
        field = SUCC_FF( hdFF );
        l = VAL_FFE(hdL);
        if ( VAL_FFE(hdR) == 0 )  r = 0;
        else r = (ORD_FF(hdL)-1) / (ORD_FF(hdR)-1) * (VAL_FFE(hdR)-1) + 1;
    }

    /* product of a finite field element and an element from a superfield  */
    else if ( ORD_FF(hdR)%ORD_FF(hdL)==0 && ORD_FF(hdR)%(ORD_FF(hdL)-1)<=1 ){
        hdFF = FLD_FFE(hdR);
        field = SUCC_FF( hdFF );
        if ( VAL_FFE(hdL) == 0 )  l = 0;
        else l = (ORD_FF(hdR)-1) / (ORD_FF(hdL)-1) * (VAL_FFE(hdL)-1) + 1;
        r = VAL_FFE(hdR);
    }

    /* else try to find a common finite field                              */
    else {
        hdFF = CommonFF( hdL, hdR );
        if ( hdFF == 0 )
            return Error(
                "Finite field *: operands must have the same characteristic",
                         0,0);
        else if ( hdFF == (Bag)1 )
            return Error(
                "Finite field *: smallest common superfield to large",
                         0,0);
        field = SUCC_FF( hdFF );
        if ( VAL_FFE(hdL) == 0 )  l = 0;
        else  l = (VAL_FFE(hdL)-1) * field[0] / (ORD_FF(hdL)-1) + 1;
        if ( VAL_FFE(hdR) == 0 )  r = 0;
        else  r = (VAL_FFE(hdR)-1) * field[0] / (ORD_FF(hdR)-1) + 1;
    }

    /* compute the product                                                 */
    x             = PROD_FF( l, r, field );
    hdX           = NewBag( T_FFE, SIZE_HD + sizeof(TypFFE) );
    SET_FLD_FFE( hdX, hdFF );
    SET_VAL_FFE( hdX, x );
    return  hdX;
}


/****************************************************************************
**
*F  QuoFFE( <hdL>, <hdR> ) . . . . . . . .  quotient of finite field elements
**
**  'QuoFFE' returns the quotient of the two finite field elements <hdL>  and
**  <hdR>. The quotient is represented over the field over which the operands
**  are represented, even if it lies in a much smaller field.
**
**  If one of the operands  is an integer  it is converted  into the field of
**  the other  operand before  the  division.  If  one  of  the  elements  is
**  represented over a subfield of the field over which  the other element is
**  represented it is lifted into the larger field before the division.
**
**  Is called from the 'Quo' binop, so both operands are already  evaluated.
**
**  'QuoFFE' just does the conversions mentioned above and  then  calls  the
**  macro 'QUO_FF' to do the actual division.
*/
Bag       QuoFFE (Bag hdL, Bag hdR)
{
    Bag           hdX;            /* handle of the quotient          */
    TypFFE              x;              /* value of the quotient           */
    Bag           hdFF;           /* handle of  the finite field bag */
    TypFFE              * field;        /* pointer to the finite field bag */
    TypFFE              l;              /* value of the left operand       */
    TypFFE              r;              /* value of the right operand      */

    /* quotient of an integer and a finite field element                   */
    if ( GET_TYPE_BAG(hdL) == T_INT ) {
        hdFF = FLD_FFE(hdR);
        field = SUCC_FF( hdFF );
        r = (HD_TO_INT(hdL) % (Int)ORD_FF(hdR) + ORD_FF(hdR)) % ORD_FF(hdR);
        if ( r == 0 )  l = 0;
        else for ( l = 1; 1 < r; --r )  l = (l == 0 ? 1 : field[l]);
        r = VAL_FFE(hdR);
    }

    /* quotient of a finite field element and an integer                   */
    else if ( GET_TYPE_BAG(hdR) == T_INT ) {
        hdFF = FLD_FFE(hdL);
        field = SUCC_FF( hdFF );
        l = (HD_TO_INT(hdR) % (Int)ORD_FF(hdL) + ORD_FF(hdL)) % ORD_FF(hdL);
        if ( l == 0 )  r = 0;
        else for ( r = 1; 1 < l; --l )  r = (r == 0 ? 1 : field[r]);
        l = VAL_FFE(hdL);
    }

    /* quotient of two finite field element in the same finite field       */
    else if ( FLD_FFE(hdL) == FLD_FFE(hdR) ) {
        hdFF = FLD_FFE(hdL);
        field = SUCC_FF( hdFF );
        l = VAL_FFE(hdL);
        r = VAL_FFE(hdR);
    }

    /* quotient of a finite field element and an element from a subfield   */
    else if ( ORD_FF(hdL)%ORD_FF(hdR)==0 && ORD_FF(hdL)%(ORD_FF(hdR)-1)<=1 ){
        hdFF = FLD_FFE(hdL);
        field = SUCC_FF( hdFF );
        l = VAL_FFE(hdL);
        if ( VAL_FFE(hdR) == 0 )  r = 0;
        else r = (ORD_FF(hdL)-1) / (ORD_FF(hdR)-1) * (VAL_FFE(hdR)-1) + 1;
    }

    /* quotient of a finite field element and an element from a superfield */
    else if ( ORD_FF(hdR)%ORD_FF(hdL)==0 && ORD_FF(hdR)%(ORD_FF(hdL)-1)<=1 ){
        hdFF = FLD_FFE(hdR);
        field = SUCC_FF( hdFF );
        if ( VAL_FFE(hdL) == 0 )  l = 0;
        else l = (ORD_FF(hdR)-1) / (ORD_FF(hdL)-1) * (VAL_FFE(hdL)-1) + 1;
        r = VAL_FFE(hdR);
    }

    /* else try to find a common finite field                              */
    else {
        hdFF = CommonFF( hdL, hdR );
        if ( hdFF == 0 )
            return Error(
                "Finite field /: operands must have the same characteristic",
                         0,0);
        else if ( hdFF == (Bag)1 )
            return Error(
                "Finite field /: smallest common superfield to large",
                         0,0);
        field = SUCC_FF( hdFF );
        if ( VAL_FFE(hdL) == 0 )  l = 0;
        else  l = (VAL_FFE(hdL)-1) * field[0] / (ORD_FF(hdL)-1) + 1;
        if ( VAL_FFE(hdR) == 0 )  r = 0;
        else  r = (VAL_FFE(hdR)-1) * field[0] / (ORD_FF(hdR)-1) + 1;
    }

    /* compute the quotient                                                */
    if ( r == 0 ) return Error("divisor must be nonzero",0,0);
    x             = QUO_FF( l, r, field );
    hdX           = NewBag( T_FFE, SIZE_HD + sizeof(TypFFE) );
    SET_FLD_FFE( hdX, hdFF );
    SET_VAL_FFE( hdX, x );
    return  hdX;
}


/****************************************************************************
**
*F  PowFFE( <hdL>, <hdR> )  . . . . . . . . . power of a finite field element
**
**  'PowFFE' returns the power of the finite  field  element  <hdL>  and  the
**  integer <hdR>.  The power is represented over the field  over  which  the
**  left operand is represented, even if it lies in a much smaller field.
**
**  Is called from the 'Pow' binop, so both operands are already  evaluated.
**
**  'PowFFE' just does the conversions mentioned above and  then  calls  the
**  macro 'POW_FF' to do the actual exponentiation.
*/
Bag       PowFFE (Bag hdL, Bag hdR)
{
    Bag           hdX;            /* handle of the power             */
    TypFFE              x;              /* value of the power              */
    Bag           hdFF;           /* handle of  the finite field bag */
    TypFFE              * field;        /* pointer to the finite field bag */
    TypFFE              l;              /* value of the left operand       */
    Int                r;              /* value of the right operand      */

    /* get the finite field and the value                                  */
    hdFF = FLD_FFE(hdL);
    field = SUCC_FF( hdFF );
    l = VAL_FFE(hdL);

    /* get the integer exponent                                            */
    r = HD_TO_INT( hdR );

    /* if r is negative invert l before proceeding                         */
    if ( r < 0 ) {
        if ( l == 0 )  return Error("divisor must be nonzero",0,0);
        l = QUO_FF( 1, l, field );
        r = -r;
    }

    /* compute the power                                                   */
    x             = POW_FF( l, r, field );
    hdX           = NewBag( T_FFE, SIZE_HD + sizeof(TypFFE) );
    SET_FLD_FFE( hdX, hdFF );
    SET_VAL_FFE( hdX, x );
    return hdX;
}


/****************************************************************************
**
*F  EqFFE( <hdL>, <hdR> ) . . . . . . test if finite field elements are equal
**
**  'EqFFE' returns 'HdTrue' if the two finite field elements <hdL> and <hdR>
**  are equal and 'HdFalse' othwise.
**
**  Is called from the 'Eq' binop, so both operands are already  evaluated.
**
**  This is complicated because it must account  for the following situation.
**  Suppose 'a' is 'Z(3)', 'b' is 'Z(3^2)^4' and  finally 'c' is 'Z(3^3)^13'.
**  Mathematically 'a' is equal to 'b', so we  want 'a =  b' to be 'true' and
**  since 'a' is represented over a  subfield of 'b'  this is no big problem.
**  Again  'a' is equal to  'c', and again we want  'a = c'  to be 'true' and
**  again this is no problem since 'a' is represented over a subfield of 'c'.
**  Since '=' ought  to be transitive we also  want 'b = c'  to be 'true' and
**  this is a problem, because they are represented over incompatible fields.
*/
Bag       EqFFE (Bag hdL, Bag hdR)
{
    TypFFE              l,  r;          /* values                          */
    Int                fl, fr;         /* order of the representing field */
    Int                pl, pr;         /* characteristic of the fields    */
    Int                ml, mr;         /* order of the minimal fields     */

    /* get the values and the fields over which they are represented       */
    l  = VAL_FFE(hdL);
    r  = VAL_FFE(hdR);
    fl = ORD_FF(hdL);
    fr = ORD_FF(hdR);

    /* if one is zero the other must be zero too                           */
    if ( l == 0 || r == 0 ) {
        if ( l == 0 && r == 0 )  return HdTrue;
        else                     return HdFalse;
    }

    /* if the are represented over the same finite field its easy          */
    if ( fl == fr ) {
        if ( l == r )  return HdTrue;
        else           return HdFalse;
    }

    /* get the characteristics                                             */
    if ( fl % 2 == 0 )  pl = 2;
    else for ( pl = 3; fl % pl != 0; pl += 2 ) ;
    if ( fr % 2 == 0 )  pr = 2;
    else for ( pr = 3; fr % pr != 0; pr += 2 ) ;

    /* if they lie in fields of different characteristics the are different*/
    if ( pl != pr ) {
        return HdFalse;
    }

    /* now find the order of the minimal fields in which l and r lie       */
    ml = pl;
    while ( (fl-1) % (ml-1) != 0 || (l-1) % ((fl-1)/(ml-1)) != 0 )  ml *= pl;
    mr = pr;
    while ( (fr-1) % (mr-1) != 0 || (r-1) % ((fr-1)/(mr-1)) != 0 )  mr *= pr;

    /* if they are different l and r are different                         */
    if ( ml != mr ) {
        return HdFalse;
    }

    /* otherwise compare both elements in the minimal field                */
    if ( (l-1) / ((fl-1)/(ml-1)) == (r-1) / ((fr-1)/(mr-1)) )  return HdTrue;
    else                                                      return HdFalse;
}


/****************************************************************************
**
*F  LtFFE( <hdL>, <hdR> ) . . . . .  test if finite field elements is smaller
**
**  'LtFFE' returns 'HdTrue' if the finite field element  <hdL>  is  strictly
**  less than the finite field element <hdR> and 'HdFalse' otherwise.
**
**  Is called from the 'Lt' binop, so both operands are already  evaluated.
*/
Bag       LtFFE (Bag hdL, Bag hdR)
{
    TypFFE              l,  r;          /* values                          */
    Int                fl, fr;         /* order of the representing field */
    Int                pl, pr;         /* characteristic of the fields    */
    Int                ml, mr;         /* order of the minimal fields     */

    /* get the values and the fields over which they are represented       */
    l  = VAL_FFE(hdL);
    r  = VAL_FFE(hdR);
    fl = ORD_FF(hdL);
    fr = ORD_FF(hdR);

    /* zero is smaller than any other value from a finite field            */
    if ( l == 0 || r == 0 ) {
        if ( l == 0 && r != 0 )  return HdTrue;
        else                     return HdFalse;
    }

    /* get the characteristics                                             */
    if ( fl % 2 == 0 )  pl = 2;
    else for ( pl = 3; fl % pl != 0; pl += 2 ) ;
    if ( fr % 2 == 0 )  pr = 2;
    else for ( pr = 3; fr % pr != 0; pr += 2 ) ;

    /* if they lie in fields of different characteristics the are different*/
    if ( pl != pr ) {
        if ( pl < pr )  return HdTrue;
        else            return HdFalse;
    }

    /* now find the order of the minimal fields in which l and r lie       */
    ml = pl;
    while ( (fl-1) % (ml-1) != 0 || (l-1) % ((fl-1)/(ml-1)) != 0 )  ml *= pl;
    mr = pr;
    while ( (fr-1) % (mr-1) != 0 || (r-1) % ((fr-1)/(mr-1)) != 0 )  mr *= pr;

    /* if they are different l and r are different                         */
    if ( ml != mr ) {
        if ( ml < mr )  return HdTrue;
        else            return HdFalse;
    }

    /* otherwise compare both elements in the minimal field                */
    if ( (l-1) / ((fl-1)/(ml-1)) < (r-1) / ((fr-1)/(mr-1)) )  return HdTrue;
    else                                                      return HdFalse;
}


/****************************************************************************
**
*F  PrFFE( <hdFFE> )  . . . . . . . . . . . . .  print a finite field element
**
**  'PrFFE' prints the finite field element <hdFFE>.
*/
void            PrFFE (Bag hdFFE)
{
    PrFF( FLD_FFE(hdFFE), VAL_FFE(hdFFE) );
}


/****************************************************************************
**
*F  PrFF( <hdField>, <value> )  . . . . . . . . .  print a finite field value
**
**  'PrFF' prints the value <value> from the finite field <hdField>.
**
**  This procedure is called by the 'PrVector' printing procedure, which  can
**  not call 'PrFFE' because it would have to create  finite  field  elements
**  to do so and calling 'NewBag' from a printing procedure is forbidden.
*/
void            PrFF (Bag hdField, unsigned int value)
{
    UInt       o;              /* order of the finite field       */
    UInt       p;              /* characteristic of finite field  */
    UInt       m;              /* order of minimal finite field   */
    UInt       d;              /* degree of minimal finite field  */

    /* get the characteristic, order of the minimal field and the degree   */
    o = SIZE_FF( hdField );
    if ( o % 2 == 0 )  p = 2;
    else for ( p = 3; o % p != 0; p += 2 ) ;

    /* print the zero                                                      */
    if ( value == 0 ) {
        Pr("%>0*Z(%>%d%2<)",(Int)p,0);
    }

    /* print a nonzero element as power of the primitive root              */
    else {

        /* find the degree of the minimal field in that the element lies   */
        d = 1;  m = p;
        while ( (o-1) % (m-1) != 0 || (value-1) % ((o-1)/(m-1)) != 0 ) {
            d++;  m *= p;
        }
        value = (value-1) / ((o-1)/(m-1)) + 1;

        /* print the element                                               */
        Pr("%>Z(%>%d%<",(Int)p,0);
        if ( d == 1 )  Pr("%<)",0,0);
        else  Pr("^%>%d%2<)",(Int)d,0);
        if ( value != 2 )  Pr("^%>%d%<",(Int)value-1,0);

    }

}


/****************************************************************************
**
*F  FunIsFFE( <hdCall> )  . . . . . . . . . .  test for finite field elements
**
**  'FunIsFFE' implements the internal function 'IsFFE( <obj> )'.
**
**  'IsFFE' returns  'true' if its argument  <obj> is a finite  field element
**  and 'false' otherwise.   'IsFFE' will cause  an  error if  called with an
**  unbound variable.
*/
Bag       FunIsFFE (Bag hdCall)
{
    Bag           hdObj;

    /* check the arguments                                                 */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: IsFFE( <obj> )",0,0);

    /* evaluate and check the object                                       */
    hdObj = EVAL( PTR_BAG(hdCall)[1] );
    if ( hdObj == HdVoid )
        return Error("IsFFE: function must return a value",0,0);

    /* return 'true' if <obj> is a finite field element                    */
    if ( GET_TYPE_BAG(hdObj) == T_FFE )
        return HdTrue;
    else
        return HdFalse;
}


/****************************************************************************
**
*F  CharFFE(<hdFFE>)  . . . . . . .  characteristic of a finite field element
**
**  'CharFFE' returns the characteristic of the field  in  which  the  finite
**  field element <hdFFE> lies.
*/
Int            CharFFE (Bag hdFFE)
{
    UInt       p;              /* characteristic, result          */
    UInt       q;              /* size of the finite field        */

    /* get the size of the finite field                                    */
    q = SIZE_FF( FLD_FFE( hdFFE ) );

    /* simply find the smallest prime that divides the size                */
    if ( q % 2 == 0 ) {
        p = 2;
    }
    else {
        for ( p = 3; q % p != 0; p += 2 )
            ;
    }

    /* return the result                                                   */
    return p;
}


/****************************************************************************
**
*F  DegreeFFE(<hdFFE>)  . . . . . . . . . .  degree of a finite field element
**
**  'DegreeFFE' returns the degree of the smallest finite field in which  the
**  finite field element <hdFFE> lies.
*/
Int            DegreeFFE (Bag hdFFE)
{
    UInt       d;              /* degree, result                  */
    TypFFE              v;              /* value of the ffe                */
    UInt       p;              /* characteristic, result          */
    UInt       q;              /* size of the finite field        */
    UInt       r;              /* size of subfields               */

    /* get the value of the finite field element                           */
    v = VAL_FFE( hdFFE );

    /* get the size of the finite field                                    */
    q = SIZE_FF( FLD_FFE( hdFFE ) );

    /* simply find the smallest prime that divides the size                */
    if ( q % 2 == 0 ) {
        p = 2;
    }
    else {
        for ( p = 3; q % p != 0; p += 2 )
            ;
    }

    /* get the degree of the smallest field that contains the element      */
    r = p;
    d = 1;
    if ( v != 0 ) {
        while ( (q-1)%(r-1) != 0 || (v-1)%((q-1)/(r-1)) != 0 ) {
            r *= p;
            d += 1;
        }
    }

    /* return the result                                                   */
    return d;
}


/****************************************************************************
**
*F  FunLogFFE( <hdCall> ) . . . . . . .  logarithm of a finite field constant
**
**  'FunLogFFE' implements the internal function 'LogFFE( <z> )'.
**
**  If called with one argument 'LogFFE' returns the logarithm of the  finite
**  field element <z> with respect to the generator of the finite field  over
**  which <z> is represented.  If called with two arguments 'LogFFE'  returns
**  the logarithm of the finite field element <z> with respect to the  second
**  argument <r> which must lie in the same field like <z>.
**
**  If <z> is 0 'LogFFE' causes an error.
*/
Bag       FunLogFFE (Bag hdCall)
{
    Bag           hdZ;            /* handle of the first argument    */
    Bag           hdR;            /* handle of the second argument   */
    Bag           hdFF;           /* handle of common finite field   */
    TypFFE              * field;        /* pointer to common finite field  */
    Int                i,  k,  o,  a,  b,  t;

    /* check the arguments                                                 */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD && GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )
        return Error("usage: LogFFE( <z> ) or LogFFE( <z>, <r> )",0,0);

    /* called with one argument                                            */
    if ( GET_SIZE_BAG(hdCall) == 2 * SIZE_HD ) {
        hdZ = EVAL( PTR_BAG(hdCall)[1] );
        if ( GET_TYPE_BAG(hdZ) != T_FFE )
            return Error("LogFFE: <z> must be a finite field element",0,0);
        if ( VAL_FFE(hdZ)==0 )
            return Error("LogFFE: <z> must be nonzero",0,0);
        i = VAL_FFE(hdZ) - 1;
        k = 1;
        o = ORD_FF(hdZ) - 1;
    }

    /* called with two elements                                            */
    else {
        hdZ = EVAL( PTR_BAG(hdCall)[1] );
        if ( GET_TYPE_BAG(hdZ) != T_FFE )
            return Error("LogFFE: <z> must be a finite field element",0,0);
        hdR = EVAL( PTR_BAG(hdCall)[2] );
        if ( GET_TYPE_BAG(hdR) != T_FFE )
            return Error("LogFFE: <r> must be a finite field element",0,0);

        /* which lie over the same field                                   */
        if ( FLD_FFE(hdZ) == FLD_FFE(hdR) ) {
            if ( VAL_FFE(hdZ) == 0 )
                return Error("LogFFE: <z> must be nonzero",0,0);
            i = VAL_FFE(hdZ) - 1;
            if ( VAL_FFE(hdR) == 0 )
                return Error("LogFFE: <r> must be nonzero",0,0);
            k = VAL_FFE(hdR) - 1;
            o = ORD_FF(hdZ) - 1;
        }

        /* where <z> lies in a subfield                                    */
        else if (ORD_FF(hdR)%ORD_FF(hdZ)==0&&ORD_FF(hdR)%(ORD_FF(hdZ)-1)<=1){
            if ( VAL_FFE(hdZ) == 0 )
                return Error("LogFFE: <z> must be nonzero",0,0);
            i = (ORD_FF(hdR)-1) / (ORD_FF(hdZ)-1) * (VAL_FFE(hdZ)-1);
            if ( VAL_FFE(hdR) == 0 )
                return Error("LogFFE: <r> must be nonzero",0,0);
            k = VAL_FFE(hdR) - 1;
            o = ORD_FF(hdR) - 1;
        }

        /* where <r> lies in a subfield                                    */
        else if (ORD_FF(hdZ)%ORD_FF(hdR)==0&&ORD_FF(hdZ)%(ORD_FF(hdR)-1)<=1){
            if ( VAL_FFE(hdZ) == 0 )
                return Error("LogFFE: <z> must be nonzero",0,0);
            i = VAL_FFE(hdZ) - 1;
            if ( VAL_FFE(hdR) == 0 )
                return Error("LogFFE: <r> must be nonzero",0,0);
            k = (ORD_FF(hdZ)-1) / (ORD_FF(hdR)-1) * (VAL_FFE(hdR)-1);
            o = ORD_FF(hdZ) - 1;
        }

        /* otherwise try to find a common field                            */
        else {
            hdFF = CommonFF( hdZ, hdR );
            if ( hdFF == 0 )
                return Error(
                    "LogFFE: operands must have the same characteristic",
                             0,0);
            else if ( hdFF == (Bag)1 )
                return Error(
                    "LogFFE: smallest common superfield to large",
                             0,0);
            field = SUCC_FF( hdFF );
            if ( VAL_FFE(hdZ) == 0 )
                return Error("LogFFE: <z> must be nonzero",0,0);
            i = (VAL_FFE(hdZ)-1) * field[0] / (ORD_FF(hdZ)-1);
            if ( VAL_FFE(hdR) == 0 )
                return Error("LogFFE: <r> must be nonzero",0,0);
            k = (VAL_FFE(hdR)-1) * field[0] / (ORD_FF(hdR)-1);
            o = field[0];
        }

    }

    /* <i>=log(<z>), <k>=log(<r>), <o>=ord(<gf>)-1, solve <k>*<l>=<i>%<o>  */
    /*N 14-Feb-90 martin this is likely to explode if 'TypFFE' is 'ulong'  */
    a = 1;  b = 0;
    while ( o != 0 ) {
        t = b;  b = a - (k/o) * b;  a = t;
        t = o;  o = k - (k/o) * o;  k = t;
    }
    if ( i % k != 0 )
        return Error("LogFFE: <z> must be a power of <r>",0,0);

    /* return the logarithm                                                */
    return INT_TO_HD( i / k * a );
}


/****************************************************************************
**
*F  FunIntFFE( <hdCall> ) . . .  convert a finite field element to an integer
**
**  'FunIntFFE' implements the internal function 'IntFFE( <z> )'.
**
**  'IntFFE'  returns  the integer  that  corresponds  to  the  finite  field
**  element <z>, which must of course be  an element  of a prime field, i.e.,
**  the smallest integer <i> such that '<i> * <z>^0 = <z>'.
*/

/*V HdIntFFEs */
/*V HdLastIntFFE */
Bag HdIntFFEs;
Bag HdLastIntFFE;

Bag ConvTabIntFFE (Int q)
{
    Int                i;              /* loop variable                   */
    Bag           hdZ;            /* handle of the element           */
    TypFFE              y;              /* loop variable                   */
    TypFFE *            field;          /* log table                       */

    /* is this the same field as last time                                 */
    if ( HdLastIntFFE != 0 && LEN_LIST(HdLastIntFFE) == q )
	return HdLastIntFFE;

    /* check if the table is already stored in <HdIntFFEs>                 */
    for ( i = LEN_LIST(HdIntFFEs);  0 < i;  i-- ) {
	HdLastIntFFE = ELM_LIST( HdIntFFEs, i );
	if ( LEN_LIST(HdLastIntFFE) == q )
	    break;
    }
    if ( 0 < i )
	return HdLastIntFFE;

    /* create a new conversion table                                       */
    HdLastIntFFE = NewBag( T_LIST, SIZE_PLEN_PLIST(q) );
    hdZ   = RootFiniteField(q);
    field = SUCC_FF(FLD_FFE(hdZ));
    SET_LEN_PLIST( HdLastIntFFE, q );
    SET_ELM_PLIST( HdLastIntFFE, 1, INT_TO_HD(0) );
    for ( i = 1, y = 1;  y != 0;  y = field[y], i++ ) {
	SET_ELM_PLIST( HdLastIntFFE, y+1, INT_TO_HD(i) );
    }
    ASS_LIST( HdIntFFEs, LEN_LIST(HdIntFFEs)+1, HdLastIntFFE );

    /* and return the table                                                */
    return HdLastIntFFE;
}

Bag FunIntFFE (Bag hdCall)
{
    Bag           hdZ;            /* handle of the element           */
    Bag           ff;             /* finite field of <z>             */
    Int                q;              /* number of field elements        */
    Bag           tab;            /* conversion table                */
    Bag           hdRes;          /* the result                      */

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: IntFFE( <z> )", 0, 0 );
    hdZ = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hdZ) != T_FFE )
        return Error("IntFFE: <z> must be a finite field element",0,0);
    ff = FLD_FFE(hdZ);
    q  = SIZE_FF(ff);

    /* create the conversion table                                         */
    tab = ConvTabIntFFE(q);
 
    /* find the integer                                                    */
    hdRes = ELM_PLIST( tab, VAL_FFE(hdZ)+1 );
    if ( hdRes == 0 )
	return Error("IntFFE: <z> must lie in the prime field",0,0);
    else
	return hdRes;
}


/****************************************************************************
**
*F  FunZ( <hdCall> )  . . . . . . . .  return the generator of a finite field
**
**  'FunZ' implements the internal function 'Z( <q> )'.
**
**  'Z' returns the generators of the finite field  with  <q>  elements.  <q>
**  must be a positive prime power.
*/
Bag       FunZ (Bag hdCall)
{
    Bag           hdZ;            /* handle of the primitive root    */

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
        return Error("usage: Z( <q> )",0,0);
    hdZ = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hdZ) != T_INT || HD_TO_INT(hdZ) <= 1 )
        return Error("Z: <q> must be a prime power in [2..65536]",0,0);

    /* get and return the root of the finite field                         */
    hdZ = RootFiniteField( HD_TO_INT(hdZ) );
    if ( hdZ == 0 )
        return Error("Z: <q> must be a prime power in [2..65536]",0,0);
    return hdZ;
}


/****************************************************************************
**
*F  InitFF()  . . . . . . . . . . . . . . . . initialize finite field package
**
**  'InitFF' initializes the finite field package.
*/
void            InitFF (void)
{
    /* install the evaluation function                                     */
    InstEvFunc( T_FFE, EvFFE );
    InstPrFunc( T_FFE, PrFFE );

    /* install the binary operators                                        */
    TabSum[  T_FFE ][ T_FFE ] = SumFFE;
    TabSum[  T_INT ][ T_FFE ] = SumFFE;
    TabSum[  T_FFE ][ T_INT ] = SumFFE;
    TabDiff[ T_FFE ][ T_FFE ] = DiffFFE;
    TabDiff[ T_INT ][ T_FFE ] = DiffFFE;
    TabDiff[ T_FFE ][ T_INT ] = DiffFFE;
    TabProd[ T_FFE ][ T_FFE ] = ProdFFE;
    TabProd[ T_INT ][ T_FFE ] = ProdFFE;
    TabProd[ T_FFE ][ T_INT ] = ProdFFE;
    TabQuo[  T_FFE ][ T_FFE ] = QuoFFE;
    TabQuo[  T_INT ][ T_FFE ] = QuoFFE;
    TabQuo[  T_FFE ][ T_INT ] = QuoFFE;
    TabPow[  T_FFE ][ T_INT ] = PowFFE;
    TabEq[   T_FFE ][ T_FFE ] = EqFFE;
    TabLt[   T_FFE ][ T_FFE ] = LtFFE;

    /* create the fields and integer conversion bags                       */
    HdFields  = NewBag( T_LIST, 0 );
    HdIntFFEs = NewBag( T_LIST, SIZE_PLEN_PLIST(0) );
    SET_LEN_PLIST( HdIntFFEs, 0 );

    /* install the internal functions                                      */
    InstIntFunc( "IsFFE",   FunIsFFE   );
    InstIntFunc( "LogFFE",  FunLogFFE  );
    InstIntFunc( "IntFFE",  FunIntFFE  );
    InstIntFunc( "Z",       FunZ       );
}
