/****************************************************************************
**
*A  vector.c                    GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2020, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2020) by the GAP Group (www.gap-system.org).
**
**  This file contains the functions  that mainly  operate  on vectors  whose
**  elements are integers, rationals, or elements from cyclotomic fields.  As
**  vectors are special lists many things are done in the list package.
**
**  A *vector* is a list that has no holes,  and whose elements all come from
**  a common field.  For the full definition of vectors see chapter "Vectors"
**  in  the {\GAP} manual.   Read also about "More   about Vectors" about the
**  vector flag and the compact representation of vectors over finite fields.
**
**  A list  that  is  known  to be a vector is represented  by  a bag of type
**  'T_VECTOR', which  has  exactely the same representation  as bags of type
**  'T_LIST'.  As a matter of fact the functions in this  file do not  really
**  know   how   this   representation   looks,   they   use    the    macros
**  'SIZE_PLEN_PLIST',   'PLEN_SIZE_PLIST',   'LEN_PLIST',   'SET_LEN_PLIST',
**  'ELM_PLIST', and 'SET_ELM_PLIST' exported by the plain list package.
**
**  Note  that  a list  represented by  a  bag  of type 'T_LIST',  'T_SET' or
**  'T_RANGE' might still be a vector over the rationals  or cyclotomics.  It
**  is just that the kernel does not known this.
**
**  This  package only consists  of  the  functions 'LenVector', 'ElmVector',
**  'ElmsVector',   AssVector',   'AsssVector',  'PosVector',  'PlainVector',
**  'IsDenseVector',  'IsPossVector', 'EqVector', and  'LtVector'.  They  are
**  the  functions  required by  the  generic  lists  package.   Using  these
**  functions the  other  parts of  the  {\GAP} kernel can access and  modify
**  vectors without actually being aware that they are dealing with a vector.
**
*/

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "scanner.h"             /* reading of tokens and printing  */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */

#include        "list.h"                /* generic list package            */
#include        "plist.h"               /* 'LEN_PLIST', 'SET_LEN_PLIST',.. */
#include        "range.h"               /* 'LEN_RANGE', 'LOW_RANGE', ..    */
#include        "objects.h"
#include		"string4.h"
#include        "integer4.h"            /* TypDigit                        */

#include        "vector.h"              /* declaration part of the package */


/****************************************************************************
**
*F  LenVector(<hdList>) . . . . . . . . . . . . . . . . .  length of a vector
**
**  'LenVector' returns the length of the vector <hdList> as a C integer.
**
**  'LenVector' is the function in 'TabLenList' for vectors.
*/
Int            LenVector (Bag            hdList )
{
    return LEN_PLIST( hdList );
}


/****************************************************************************
**
*F  ElmVector(<hdList>,<pos>) . . . . . . . . . select an element of a vector
**
**  'ElmVector' selects the element at position <pos> of the vector <hdList>.
**  It is the responsibility of the caller to ensure that <pos> is a positive
**  integer.  An  error is signalled if <pos>  is  larger than the  length of
**  <hdList>.
**
**  'ElmfVector' does  the same thing than 'ElmList', but need not check that
**  <pos>  is less than  or  equal to the  length of  <hdList>, this  is  the
**  responsibility of the caller.
**
**  'ElmVector' is the function in 'TabElmList' for vectors.  'ElmfVector' is
**  the  function  in  'TabElmfList', 'TabElmlList',  and  'TabElmrList'  for
**  vectors.
*/
Bag       ElmVector (Bag            hdList, Int pos )
{
    Bag           hdElm;          /* the selected element, result    */

    /* check the position                                                  */
    if ( LEN_PLIST( hdList ) < pos ) {
        return Error(
          "List Element: <list>[%d] must have a value",
                     pos,  0 );
    }

    /* select and check the element                                        */
    hdElm = ELM_PLIST( hdList, pos );

    /* return the element                                                  */
    return hdElm;
}

Bag       ElmfVector (Bag hdList, Int pos)
{
    /* select and return the element                                       */
    return ELM_PLIST( hdList, pos );
}


/****************************************************************************
**
*F  ElmsVector(<hdList>,<hdPoss>) . . . . . .  select a sublist from a vector
**
**  'ElmsVector' returns a new list containing the elements  at the  position
**  given  in  the  list  <hdPoss>  from  the  vector  <hdList>.  It  is  the
**  responsibility  of  the  caller  to  ensure that  <hdPoss> is  dense  and
**  contains only positive integers.   An error is signalled if an element of
**  <hdPoss> is larger than the length of <hdList>.
**
**  'ElmsVector' is the function in 'TabElmsList' for vectors.
*/
Bag       ElmsVector (Bag hdList, Bag hdPoss)
{
    Bag           hdElms;         /* selected sublist, result        */
    Int                lenList;        /* length of <list>                */
    Bag           hdElm;          /* one element from <list>         */
    Int                lenPoss;        /* length of <positions>           */
    Int                pos;            /* <position> as integer           */
    Int                inc;            /* increment in a range            */
    Int                i;              /* loop variable                   */

    /* general code                                                        */
    if ( GET_TYPE_BAG(hdPoss) != T_RANGE ) {

        /* get the length of <list>                                        */
        lenList = LEN_PLIST( hdList );

        /* get the length of <positions>                                   */
        lenPoss = LEN_LIST( hdPoss );

        /* make the result list                                            */
        hdElms = NewBag( T_VECTOR, SIZE_PLEN_PLIST( lenPoss ) );
        SET_LEN_PLIST( hdElms, lenPoss );

        /* loop over the entries of <positions> and select                 */
        for ( i = 1; i <= lenPoss; i++ ) {

            /* get <position>                                              */
            pos = HD_TO_INT( ELMF_LIST( hdPoss, i ) );
            if ( lenList < pos ) {
                return Error(
                  "List Elements: <list>[%d] must have a value",
                             pos,  0 );
            }

            /* select the element                                          */
            hdElm = ELM_PLIST( hdList, pos );

            /* assign the element into <elms>                              */
            SET_ELM_PLIST( hdElms, i, hdElm );

        }

    }

    /* special code for ranges                                             */
    else {

        /* get the length of <list>                                        */
        lenList = LEN_PLIST( hdList );

        /* get the length of <positions>, the first elements, and the inc. */
        lenPoss = LEN_RANGE( hdPoss );
        pos = LOW_RANGE( hdPoss );
        inc = INC_RANGE( hdPoss );

        /* check that no <position> is larger than 'LEN_LIST(<list>)'      */
        if ( lenList < pos ) {
            return Error(
              "List Elements: <list>[%d] must have a value",
                         pos,  0 );
        }
        if ( lenList < pos + (lenPoss-1) * inc ) {
            return Error(
              "List Elements: <list>[%d] must have a value",
                         pos + (lenPoss-1) * inc,  0 );
        }

        /* make the result list                                            */
        hdElms = NewBag( T_VECTOR, SIZE_PLEN_PLIST( lenPoss ) );
        SET_LEN_PLIST( hdElms, lenPoss );

        /* loop over the entries of <positions> and select                 */
        for ( i = 1; i <= lenPoss; i++, pos += inc ) {

            /* select the element                                          */
            hdElm = ELM_PLIST( hdList, pos );

            /* assign the element to <elms>                                */
            SET_ELM_PLIST( hdElms, i, hdElm );

        }

    }

    /* return the result                                                   */
    return hdElms;
}


/****************************************************************************
**
*F  AssVector(<hdList>,<pos>,<hdVal>) . . . . . . . . . .  assign to a vector
**
**  'AssVector' assigns the  value  <hdVal>  to the  vector  <hdList>  at the
**  position <pos>.   It is  the responsibility of the  caller to ensure that
**  <pos> is positive, and that <hdVal> is not 'HdVoid'.
**
**  If the position is larger  then the length of the vector <list>, the list
**  is automatically  extended.  To avoid  making this too often, the  bag of
**  the list is extended by at least '<length>/8 +  4' handles.  Thus in  the
**  loop
**
**      l := [];  for i in [1..1024]  do l[i] := i^2;  od;
**
**  the list 'l' is extended only 32 times not 1024 times.
**
**  'AssVector' is the function in 'TabAssList' for vectors.
**
**  'AssVector' simply converts  the  vector into a plain list, and then does
**  the same  stuff as  'AssPlist'.   This is because  a  vector is  not very
**  likely to stay a vector after the assignment.
*/
Bag       AssVector (Bag hdList, Int pos, Bag hdVal)
{
    Int                plen;           /* physical length of <list>       */

    /* assignment of a scalar within the bound                             */
    if ( T_INT <= GET_TYPE_BAG(hdVal) && GET_TYPE_BAG(hdVal) <= T_UNKNOWN
      && pos <= LEN_PLIST(hdList) ) {
        SET_ELM_PLIST( hdList, pos, hdVal );
    }

    /* assignment of a scalar immediately after the end                    */
    else if ( T_INT <= GET_TYPE_BAG(hdVal) && GET_TYPE_BAG(hdVal) <= T_UNKNOWN
           && pos == LEN_PLIST(hdList)+1 ) {
        if ( PLEN_SIZE_PLIST( GET_SIZE_BAG(hdList) ) < pos )
            Resize( hdList, SIZE_PLEN_PLIST( (pos-1) + (pos-1)/8 + 4 ) );
        SET_LEN_PLIST( hdList, pos );
        SET_ELM_PLIST( hdList, pos, hdVal );
    }

    /* otherwise convert to plain list                                     */
    else {
        Retype( hdList, T_LIST );
        if ( LEN_PLIST( hdList ) < pos ) {
            plen = PLEN_SIZE_PLIST( GET_SIZE_BAG(hdList) );
            if ( plen + plen/8 + 4 < pos )
                Resize( hdList, SIZE_PLEN_PLIST( pos ) );
            else if ( plen < pos )
                Resize( hdList, SIZE_PLEN_PLIST( plen + plen/8 + 4 ) );
            SET_LEN_PLIST( hdList, pos );
        }
        SET_ELM_PLIST( hdList, pos, hdVal );
    }

    /* return the assigned value                                           */
    return hdVal;
}


/****************************************************************************
**
*F  AsssVector(<hdList>,<hdPoss>,<hdVals>)assign several elements to a vector
**
**  'AsssVector' assignes the values from the  list <hdVals> at the positions
**  given  in  the  list  <hdPoss>  to   the  vector  <hdList>.   It  is  the
**  responsibility  of the  caller to  ensure  that  <hdPoss>  is  dense  and
**  contains only positive integers, that <hdPoss> and <hdVals> have the same
**  length, and that <hdVals> is dense.
**
**  'AsssVector' is the function in 'TabAsssList' for vectors.
**
**  'AsssVector' simply converts the vector to a plain list and then does the
**  same stuff as 'AsssPlist'.  This is because a vector  is not  very likely
**  to stay a vector after the assignment.
*/
Bag       AsssVector (Bag hdList, Bag hdPoss, Bag hdVals)
{
    /* convert <list> to a plain list                                      */
    Retype( hdList, T_LIST );

    /* and delegate                                                        */
    return ASSS_LIST( hdList, hdPoss, hdVals );
}


/****************************************************************************
**
*F  PosVector(<hdList>,<hdVal>,<start>) .  position of an element in a vector
**
**  'PosVector' returns  the  position of the  value  <hdVal>  in the  vector
**  <hdList> after the first position <start> as  a C integer.  0 is returned
**  if <hdVal> is not in the list.
**
**  'PosVector' is the function in 'TabPosList' for vectors.
*/
Int            PosVector (Bag hdList, Bag hdVal, Int start)
{
    Int                lenList;        /* length of <list>                */
    Bag           hdElm;          /* one element of <list>           */
    Int                i;              /* loop variable                   */

    /* get the length of <list>                                            */
    lenList = LEN_PLIST( hdList );

    /* loop over all entries in <list>                                     */
    for ( i = start+1; i <= lenList; i++ ) {

        /* select one element from <list>                                  */
        hdElm = ELM_PLIST( hdList, i );

        /* compare with <val>                                              */
        if ( hdElm == hdVal || EQ( hdElm, hdVal ) == HdTrue )
            break;

    }

    /* return the position (0 if <val> was not found)                      */
    return (lenList < i ? 0 : i);
}


/****************************************************************************
**
*F  PlainVector(<hdList>) . . . . . . . . .  convert a vector to a plain list
**
**  'PlainVector'  converts the vector  <hdList> to  a plain list.   Not much
**  work.
**
**  'PlainVector' is the function in 'TabPlainList' for vectors.
*/
void            PlainVector (Bag hdList)
{
    return;
}


/****************************************************************************
**
*F  IsDenseVector(<hdList>) . . . . . .  dense list test function for vectors
**
**  'IsDenseVector' returns 1, since every vector is dense.
**
**  'IsDenseVector' is the function in 'TabIsDenseList' for vectors.
*/
Int            IsDenseVector (Bag hdList)
{
    return 1;
}


/****************************************************************************
**
*F  IsPossVector(<hdList>)  . . . .  positions list test function for vectors
**
**  'IsPossVector'  returns  1  if  the  vector  <hdList>  is  a  dense  list
**  containing only positive integers, and 0 otherwise.
**
**  'IsPossVector' is the function in 'TabIsPossList' for vectors.
*/
Int            IsPossVector (Bag hdList)
{
    Int                lenList;        /* length of <list>                */
    Bag           hdElm;          /* one element of <list>           */
    Int                i;              /* loop variable                   */

    /* get the length of the variable                                      */
    lenList = LEN_PLIST( hdList );

    /* loop over the entries of the list                                   */
    for ( i = 1; i <= lenList; i++ ) {
        hdElm = ELM_PLIST( hdList, i );
        if ( hdElm==0 || GET_TYPE_BAG(hdElm) != T_INT || HD_TO_INT(hdElm) <= 0 )
            return 0;
    }

    /* no problems found                                                   */
    return 1;
}


/****************************************************************************
**
*F  IsXTypeVector(<hdList>) . . . . . . . . . . .  test if a list is a vector
**
**  'IsXTypeVector' returns  1  if  the  list  <hdList>  is a  vector  and  0
**  otherwise.   As  a  sideeffect  the  type  of  the  list  is  changed  to
**  'T_VECTOR'.
**
**  'IsXTypeVector' is the function in 'TabIsXTypeList' for vectors.
*/
Int            IsXTypeVector (Bag hdList)
{
    Int                isVector;       /* result                          */
    UInt       len;            /* length of the list              */
    Bag           hdElm;          /* one element of the list         */
    UInt       i;              /* loop variable                   */

    /* if we already know that the list is a vector, very good             */
    if ( GET_TYPE_BAG( hdList ) == T_VECTOR ) {
        return 1;
    }

    /* a range is a vector, but we have to convert it                      */
    /*N 1993/01/30 martin finds it nasty that vector knows about ranges    */
    else if ( GET_TYPE_BAG(hdList) == T_RANGE ) {
        PLAIN_LIST( hdList );
        Retype( hdList, T_VECTOR );
        isVector = 1;
    }

    /* only a list or a set can be vector                                  */
    else if ( GET_TYPE_BAG(hdList) != T_LIST && GET_TYPE_BAG(hdList) != T_SET ) {
        isVector = 0;
    }

    /* if the list is empty it is not a vector                             */
    else if ( LEN_PLIST( hdList ) == 0 || ELM_PLIST( hdList, 1 ) == 0 ) {
        isVector = 0;
    }

    /* if the first entry is a scalar or a record, it might be a vector    */
    else if ( GET_TYPE_BAG( ELM_PLIST( hdList, 1 ) ) <= T_UNKNOWN 
	   || GET_TYPE_BAG( ELM_PLIST( hdList, 1 ) ) == T_REC ) {

        /* loop over the entries                                           */
        len = LEN_PLIST( hdList );
        for ( i = 2; i <= len; i++ ) {
            hdElm = ELM_PLIST( hdList, i );
            if ( hdElm == 0 || (T_UNKNOWN < GET_TYPE_BAG(hdElm) && GET_TYPE_BAG(hdElm)!=T_REC))
	        break;
        }

        /* if <hdList> is a vector, change its type to 'T_VECTOR'          */
        isVector = (len < i) ? 1 : 0;
        if ( len < i ) {
            Retype( hdList, T_VECTOR );
        }
    }

    /* otherwise the list is certainly not a vector                        */
    else {
        isVector = 0;
    }

    /* return the result                                                   */
    return isVector;
}


/****************************************************************************
**
*F  IsXTypeMatrix(<hdList>) . . . . . . . . . . .  test if a list is a matrix
**
**  'IsXTypeMatrix'  returns  1  if  the  list <hdList>  is  a  matrix and  0
**  otherwise.   As  a  sideeffect  the  type  of  the  rows  is  changed  to
**  'T_VECTOR'.
**
**  'IsXTypeMatrix' is the function in 'TabIsXTypeList' for matrices.
*/
Int            IsXTypeMatrix (Bag hdList)
{
    Int                isMatrix;       /* result                          */
    UInt       cols;           /* length of the rows              */
    UInt       len;            /* length of the list              */
    Bag           hdElm;          /* one element of the list         */
    UInt       i;              /* loop variable                   */

    /* only lists or sets could possibly be matrices                       */
    if ( GET_TYPE_BAG(hdList) != T_LIST && GET_TYPE_BAG(hdList) != T_SET ) {
        isMatrix = 0;
    }

    /* if the list is empty it is not a matrix                             */
    else if ( LEN_PLIST( hdList ) == 0 || ELM_PLIST( hdList, 1 ) == 0 ) {
        isMatrix = 0;
    }

    /* if the first entry is a vector of scalars, try that                 */
    else if ( IsXTypeVector( ELM_PLIST( hdList, 1 ) ) == 1 ) {

        /* remember the length of the row                                  */
        cols = LEN_PLIST( ELM_PLIST( hdList, 1 ) );

        /* loop over the entries                                           */
        len = LEN_PLIST( hdList );
        for ( i = 2; i <= len; i++ ) {
            hdElm = ELM_PLIST( hdList, i );
            if ( hdElm == 0
              || IsXTypeVector( hdElm ) != 1
              || LEN_PLIST( hdElm ) != cols ) {
                break;
            }
        }

        /* no representation change neccessary                             */
        isMatrix = (len < i) ? 1 : 0;

    }

    /* otherwise the list is certainly not a matrix                        */
    else {
        isMatrix = 0;
    }

    /* return the result                                                   */
    return isMatrix;
}


/****************************************************************************
**
*F  EqVector(<hdL>,<hdR>) . . . . . . . . . . . test if two vectors are equal
**
**  'EqVector'  returns 'true' if  the two vectors <hdL> and  <hdR> are equal
**  and 'false' otherwise.
**
**  Is called from the 'EQ' binop so both operands are already evaluated.
*/
Bag       EqVector (Bag hdL, Bag hdR)
{
    UInt       lenL;           /* length of the left operand      */
    UInt       lenR;           /* length of the right operand     */
    Bag           hdElmL;         /* element of the left operand     */
    Bag           hdElmR;         /* element of the right operand    */
    UInt       i;              /* loop variable                   */

    /* get the lengths of the lists and compare them                       */
    lenL = LEN_PLIST( hdL );
    lenR = LEN_PLIST( hdR );
    if ( lenL != lenR ) {
        return HdFalse;
    }

    /* loop over the elements and compare them                             */
    for ( i = 1; i <= lenL; i++ ) {
        hdElmL = ELM_PLIST( hdL, i );
        hdElmR = ELM_PLIST( hdR, i );
        if ( hdElmL != hdElmR && EQ( hdElmL, hdElmR ) == HdFalse ) {
            return HdFalse;
        }
    }

    /* no differences found, the lists are equal                           */
    return HdTrue;
}


/****************************************************************************
**
*F  LtVector(<hdL>,<hdR>) . . . . . . . . . . . test if two vectors are equal
**
**  'LtList' returns 'true' if the vector <hdL> is less than the vector <hdR>
**  and 'false' otherwise.
**
**  Is called from the 'LT' binop so both operands are already evaluated.
*/
Bag       LtVector (Bag hdL, Bag hdR)
{
    UInt       lenL;           /* length of the left operand      */
    UInt       lenR;           /* length of the right operand     */
    Bag           hdElmL;         /* element of the left operand     */
    Bag           hdElmR;         /* element of the right operand    */
    UInt       i;              /* loop variable                   */

    /* get the lengths of the lists and compare them                       */
    lenL = LEN_PLIST( hdL );
    lenR = LEN_PLIST( hdR );

    /* loop over the elements and compare them                             */
    for ( i = 1; i <= lenL && i <= lenR; i++ ) {
        hdElmL = ELM_PLIST( hdL, i );
        hdElmR = ELM_PLIST( hdR, i );
        if ( hdElmL != hdElmR && EQ( hdElmL, hdElmR ) == HdFalse ) {
            return LT( hdElmL, hdElmR );
        }
    }

    /* reached the end of at least one list                                */
    return (lenL < lenR) ? HdTrue : HdFalse;
}


/****************************************************************************
**
*F  SumIntVector(<hdL>,<hdR>) . . . . . . . .  sum of an integer and a vector
**
**  'SumIntVector' returns the sum of the integer <hdL> and the vector <hdR>.
**  The  sum  is  a list,  where each  entry  is  the  sum of <hdL>  and  the
**  corresponding entry of <hdR>.
**
**  'SumIntVector' is an improved version  of  'SumSclList', which  does  not
**  call 'SUM' if the operands are immediate integers.
*/
Bag       SumIntVector (Bag hdL, Bag hdR)
{
    Bag           hdS;            /* handle of the sum               */
    Bag           hdSS;           /* one element of sum list         */
    Bag *         ptR;            /* pointer into the right operand  */
    Bag           hdRR;           /* one element of right operand    */
    UInt       len;            /* length                          */
    UInt       isVec;          /* is the result a vector          */
    UInt       i;              /* loop variable                   */

    /* make the result list                                                */
    EnterKernel();
    len = LEN_PLIST( hdR );
    hdS = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
    SET_LEN_PLIST( hdS, len );
    isVec = 1;

    /* loop over the entries and add                                       */
    ptR = PTR_BAG( hdR );
    for ( i = 1; i <= len; i++ ) {
        hdRR = ptR[i];
        hdSS = (Bag)((Int)hdL + (Int)hdRR - T_INT);
        if ( (((Int)hdSS) & 3) != T_INT
          || ((((Int)hdSS)<<1)>>1) != ((Int)hdSS) ) {
            hdSS = SUM( hdL, hdRR );
            ptR  = PTR_BAG( hdR );
            isVec = isVec && GET_TYPE_BAG(hdSS) <= T_UNKNOWN;
        }
        SET_BAG(hdS, i, hdSS);
    }

    /* return the result                                                   */
    if ( isVec )  Retype( hdS, T_VECTOR );
    ExitKernel( hdS );
    return hdS;
}


/****************************************************************************
**
*F  SumVectorInt(<hdL>,<hdR>) . . . . . . . .  sum of a vector and an integer
**
**  'SumVectorInt' returns the sum of the vector <hdL> and the integer <hdR>.
**  The  sum  is  a list,  where each  entry  is  the  sum of <hdR>  and  the
**  corresponding entry of <hdL>.
**
**  'SumVectorInt' is an improved version  of  'SumListScl', which  does  not
**  call 'SUM' if the operands are immediate integers.
*/
Bag       SumVectorInt (Bag hdL, Bag hdR)
{
    Bag           hdS;            /* handle of the sum               */
    Bag           hdSS;           /* one element of sum list         */
    Bag *         ptL;            /* pointer into the left operand   */
    Bag           hdLL;           /* one element of left operand     */
    UInt       len;            /* length                          */
    UInt       isVec;          /* is the result a vector          */
    UInt       i;              /* loop variable                   */

    /* make the result list                                                */
    EnterKernel();
    len = LEN_PLIST( hdL );
    hdS = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
    SET_LEN_PLIST( hdS, len );
    isVec = 1;

    /* loop over the entries and add                                       */
    ptL = PTR_BAG( hdL );
    for ( i = 1; i <= len; i++ ) {
        hdLL = ptL[i];
        hdSS = (Bag)((Int)hdLL + (Int)hdR - T_INT);
        if ( (((Int)hdSS) & 3) != T_INT
          || ((((Int)hdSS)<<1)>>1) != ((Int)hdSS) ) {
            hdSS = SUM( hdLL, hdR );
            ptL  = PTR_BAG( hdL );
            isVec = isVec && GET_TYPE_BAG(hdSS) <= T_UNKNOWN;
        }
        SET_BAG(hdS, i, hdSS);
    }

    /* return the result                                                   */
    if ( isVec )  Retype( hdS, T_VECTOR );
    ExitKernel( hdS );
    return hdS;
}


/****************************************************************************
**
*F  SumVectorVector(<hdL>,<hdR>)  . . . . . . . . . . . .  sum of two vectors
**
**  'SumVectorVector'  returns the sum  of the two  vectors <hdL>  and <hdR>.
**  The sum is  a new list, where each entry is the  sum of the corresponding
**  entries of <hdL> and <hdR>.
**
**  'SumVectorVector' is an improved version of 'SumListList', which does not
**  call 'SUM' if the operands are immediate integers.
*/
Bag       SumVectorVector (Bag hdL, Bag hdR)
{
    Bag           hdS;            /* handle of the sum               */
    Bag           hdSS;           /* one element of sum list         */
    Bag *         ptL;            /* pointer into the left operand   */
    Bag           hdLL;           /* one element of left operand     */
    Bag *         ptR;            /* pointer into the right operand  */
    Bag           hdRR;           /* one element of right operand    */
    UInt       len;            /* length                          */
    UInt       isVec;          /* is the result a vector          */
    UInt       i;              /* loop variable                   */

    /* make the result list                                                */
    EnterKernel();
    len = LEN_PLIST( hdL );
    if ( len != LEN_PLIST( hdR ) ) {
        return Error(
          "Vector +: vectors must have the same length",
                      0,  0 );
    }
    hdS = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
    SET_LEN_PLIST( hdS, len );
    isVec = 1;

    /* loop over the entries and add                                       */
    ptL = PTR_BAG( hdL );
    ptR = PTR_BAG( hdR );
    for ( i = 1; i <= len; i++ ) {
        hdLL = ptL[i];
        hdRR = ptR[i];
        hdSS = (Bag)((Int)hdLL + (Int)hdRR - T_INT);
        if ( (((Int)hdSS) & 3) != T_INT
          || ((((Int)hdSS)<<1)>>1) != ((Int)hdSS) ) {
            hdSS = SUM( hdLL, hdRR );
            ptL  = PTR_BAG( hdL );
            ptR  = PTR_BAG( hdR );
            isVec = isVec && GET_TYPE_BAG(hdSS) <= T_UNKNOWN;
        }
        SET_BAG(hdS, i, hdSS);
    }

    /* return the result                                                   */
    if ( isVec )  Retype( hdS, T_VECTOR );
    ExitKernel( hdS );
    return hdS;
}


/****************************************************************************
**
*F  DiffIntVector(<hdL>,<hdR>)  . . . . difference of an integer and a vector
**
**  'DiffIntVector'  returns  the  difference of  the  integer <hdL> and  the
**  vector  <hdR>.   The difference  is  a list,  where  each  entry  is  the
**  difference of <hdL> and the corresponding entry of <hdR>.
**
**  'DiffIntVector'  is an  improved version of 'DiffSclList', which does not
**  call 'DIFF' if the operands are immediate integers.
*/
Bag       DiffIntVector (Bag hdL, Bag hdR)
{
    Bag           hdD;            /* handle of the difference        */
    Bag           hdDD;           /* one element of difference list  */
    Bag *         ptR;            /* pointer into the right operand  */
    Bag           hdRR;           /* one element of right operand    */
    UInt       len;            /* length                          */
    UInt       isVec;          /* is the result a vector          */
    UInt       i;              /* loop variable                   */

    /* make the result list                                                */
    EnterKernel();
    len = LEN_PLIST( hdR );
    hdD = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
    SET_LEN_PLIST( hdD, len );
    isVec = 1;

    /* loop over the entries and subtract                                  */
    ptR = PTR_BAG( hdR );
    for ( i = 1; i <= len; i++ ) {
        hdRR = ptR[i];
        hdDD = (Bag)((Int)hdL - (Int)hdRR + T_INT);
        if ( (((Int)hdDD) & 3) != T_INT || (((Int)hdL) & 3) != T_INT
          || ((((Int)hdDD)<<1)>>1) != ((Int)hdDD) ) {
            hdDD = DIFF( hdL, hdRR );
            ptR  = PTR_BAG( hdR );
            isVec = isVec && GET_TYPE_BAG(hdDD) <= T_UNKNOWN;
        }
        SET_BAG(hdD, i, hdDD);
    }

    /* return the result                                                   */
    if ( isVec )  Retype( hdD, T_VECTOR );
    ExitKernel( hdD );
    return hdD;
}


/****************************************************************************
**
*F  DiffVectorInt(<hdL>,<hdR>)  . . . . difference of a vector and an integer
**
**  'DiffVectorInt'  returns  the difference  of the  vector  <hdL>  and  the
**  integer <hdR>.   The difference  is  a  list,  where  each  entry is  the
**  difference of <hdR> and the corresponding entry of <hdL>.
**
**  'DiffVectorInt' is  an improved version of 'DiffListScl',  which does not
**  call 'DIFF' if the operands are immediate integers.
*/
Bag       DiffVectorInt (Bag hdL, Bag hdR)
{
    Bag           hdD;            /* handle of the difference        */
    Bag           hdDD;           /* one element of difference list  */
    Bag *         ptL;            /* pointer into the left operand   */
    Bag           hdLL;           /* one element of left operand     */
    UInt       len;            /* length                          */
    UInt       isVec;          /* is the result a vector          */
    UInt       i;              /* loop variable                   */

    /* make the result list                                                */
    EnterKernel();
    len = LEN_PLIST( hdL );
    hdD = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
    SET_LEN_PLIST( hdD, len );
    isVec = 1;

    /* loop over the entries and add                                       */
    ptL = PTR_BAG( hdL );
    for ( i = 1; i <= len; i++ ) {
        hdLL = ptL[i];
        hdDD = (Bag)((Int)hdLL - (Int)hdR + T_INT);
        if ( (((Int)hdDD) & 3) != T_INT || (((Int)hdLL) & 3) != T_INT
          || ((((Int)hdDD)<<1)>>1) != ((Int)hdDD) ) {
            hdDD = DIFF( hdLL, hdR );
            ptL  = PTR_BAG( hdL );
            isVec = isVec && GET_TYPE_BAG(hdDD) <= T_UNKNOWN;
        }
        SET_BAG(hdD, i, hdDD);
    }

    /* return the result                                                   */
    if ( isVec )  Retype( hdD, T_VECTOR );
    ExitKernel( hdD );
    return hdD;
}


/****************************************************************************
**
*F  DiffVectorVector(<hdL>,<hdR>) . . . . . . . . . difference of two vectors
**
**  'DiffVectorVector' returns the difference of the  two vectors  <hdL>  and
**  <hdR>.  The difference is a  new list, where each entry is the difference
**  of the corresponding entries of <hdL> and <hdR>.
**
**  'DiffVectorVector' is an improved  version of  'DiffListList', which does
**  not call 'DIFF' if the operands are immediate integers.
*/
Bag       DiffVectorVector (Bag hdL, Bag hdR)
{
    Bag           hdD;            /* handle of the difference        */
    Bag           hdDD;           /* one element of difference list  */
    Bag *         ptL;            /* pointer into the left operand   */
    Bag           hdLL;           /* one element of left operand     */
    Bag *         ptR;            /* pointer into the right operand  */
    Bag           hdRR;           /* one element of right operand    */
    UInt       len;            /* length                          */
    UInt       isVec;          /* is the result a vector          */
    UInt       i;              /* loop variable                   */

    /* make the result list                                                */
    EnterKernel();
    len = LEN_PLIST( hdL );
    if ( len != LEN_PLIST( hdR ) ) {
        return Error(
          "Vector -: vectors must have the same length",
                      0,  0 );
    }
    hdD = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
    SET_LEN_PLIST( hdD, len );
    isVec = 1;

    /* loop over the entries and add                                       */
    ptL = PTR_BAG( hdL );
    ptR = PTR_BAG( hdR );
    for ( i = 1; i <= len; i++ ) {
        hdLL = ptL[i];
        hdRR = ptR[i];
        hdDD = (Bag)((Int)hdLL - (Int)hdRR + T_INT);
        if ( (((Int)hdDD) & 3) != T_INT || (((Int)hdLL) & 3) != T_INT
          || ((((Int)hdDD)<<1)>>1) != ((Int)hdDD) ) {
            hdDD = DIFF( hdLL, hdRR );
            ptL  = PTR_BAG( hdL );
            ptR  = PTR_BAG( hdR );
            isVec = isVec && GET_TYPE_BAG(hdDD) <= T_UNKNOWN;
        }
        SET_BAG(hdD, i, hdDD);
    }

    /* return the result                                                   */
    if ( isVec )  Retype( hdD, T_VECTOR );
    ExitKernel( hdD );
    return hdD;
}


/****************************************************************************
**
*F  ProdIntVector(<hdL>,<hdR>)  . . . . .  product of an integer and a vector
**
**  'ProdIntVector' returns the product of  the integer <hdL> and  the vector
**  <hdR>.  The product is the list, where each entry is the product of <hdL>
**  and the corresponding entry of <hdR>.
**
**  'ProdIntVector'  is an  improved version of 'ProdSclList', which does not
**  call 'PROD' if the operands are immediate integers.
*/
Bag       ProdIntVector (Bag hdL, Bag hdR)
{
    Bag           hdP;            /* handle of the product           */
    Bag           hdPP;           /* one element of product list     */
    Bag *         ptR;            /* pointer into the right operand  */
    Bag           hdRR;           /* one element of right operand    */
    UInt       len;            /* length                          */
    UInt       isVec;          /* is the result a vector          */
    UInt       i;              /* loop variable                   */

    /* make the result list                                                */
    EnterKernel();
    len = LEN_PLIST( hdR );
    hdP = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
    SET_LEN_PLIST( hdP, len );
    isVec = 1;

    /* loop over the entries and multiply                                  */
    ptR = PTR_BAG( hdR );
    for ( i = 1; i <= len; i++ ) {
        hdRR = ptR[i];
        hdPP = (Bag)(((Int)hdL-T_INT) * ((Int)hdRR>>1));
        if ( ((Int)hdRR & 3) != T_INT
          || (((Int)hdRR >> 1) != 0
           && (Int)hdPP / ((Int)hdRR>>1) != ((Int)hdL-T_INT)) ) {
            hdPP = PROD( hdL, hdRR );
            ptR  = PTR_BAG( hdR );
            isVec = isVec && GET_TYPE_BAG(hdPP) <= T_UNKNOWN;
        }
        else {
            hdPP = (Bag)(((Int)hdPP>>1) + T_INT);
        }
        SET_BAG(hdP, i, hdPP);
    }

    /* return the result                                                   */
    if ( isVec )  Retype( hdP, T_VECTOR );
    ExitKernel( hdP );
    return hdP;
}


/****************************************************************************
**
*F  ProdVectorInt(<hdL>,<hdR>)  . . . . .  product of a scalar and an integer
**
**  'ProdVectorInt' returns the product of  the integer <hdR>  and the vector
**  <hdL>.  The product is the list, where each entry is the product of <hdR>
**  and the corresponding entry of <hdL>.
**
**  'ProdVectorInt'  is an  improved version of 'ProdSclList', which does not
**  call 'PROD' if the operands are immediate integers.
*/
Bag       ProdVectorInt (Bag hdL, Bag hdR)
{
    Bag           hdP;            /* handle of the product           */
    Bag           hdPP;           /* one element of product list     */
    Bag *         ptL;            /* pointer into the left operand   */
    Bag           hdLL;           /* one element of left operand     */
    UInt       len;            /* length                          */
    UInt       isVec;          /* is the result a vector          */
    UInt       i;              /* loop variable                   */

    /* make the result list                                                */
    EnterKernel();
    len = LEN_PLIST( hdL );
    hdP = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
    SET_LEN_PLIST( hdP, len );
    isVec = 1;

    /* loop over the entries and multiply                                  */
    ptL = PTR_BAG( hdL );
    for ( i = 1; i <= len; i++ ) {
        hdLL = ptL[i];
        hdPP = (Bag)(((Int)hdLL-T_INT) * ((Int)hdR>>1));
        if ( ((Int)hdLL & 3) != T_INT
          || (((Int)hdR>>1) != 0
           && (Int)hdPP / ((Int)hdR>>1) != ((Int)hdLL-T_INT)) ) {
            hdPP = PROD( hdLL, hdR );
            ptL  = PTR_BAG( hdL );
            isVec = isVec && GET_TYPE_BAG(hdPP) <= T_UNKNOWN;
        }
        else {
            hdPP = (Bag)(((Int)hdPP>>1) + T_INT);
        }
        SET_BAG(hdP, i, hdPP);
    }

    /* return the result                                                   */
    if ( isVec )  Retype( hdP, T_VECTOR );
    ExitKernel( hdP );
    return hdP;
}


/****************************************************************************
**
*F  ProdVectorVector(<hdL>,<hdR>) . . . . . . . . . .  product of two vectors
**
**  'ProdVectorVector'  returns the  product  of the  two  vectors <hdL>  and
**  <hdR>.   The  product  is  the  sum of the products of the  corresponding
**  entries of the two lists.
**
**  'ProdVectorVector' is an improved version  of 'ProdListList',  which does
**  not call 'PROD' if the operands are immediate integers.
*/
Bag       ProdVectorVector (Bag hdL, Bag hdR)
{
    Bag           hdP;            /* handle of the product           */
    Bag           hdPP;           /* one summand of product          */
    Bag           hdSS;           /* temporary for sum               */
    Bag *         ptL;            /* pointer into the left operand   */
    Bag           hdLL;           /* one element of left operand     */
    Bag *         ptR;            /* pointer into the right operand  */
    Bag           hdRR;           /* one element of right operand    */
    UInt       len;            /* length                          */
    UInt       i;              /* loop variable                   */

    /* check that the lengths agree                                        */
    EnterKernel();
    len = LEN_PLIST( hdL );
    if ( len != LEN_PLIST( hdR ) ) {
        return Error(
          "Vector *: vectors must have the same length",
                      0,  0 );
    }

    /* loop over the entries and multiply                                  */
    ptL = PTR_BAG( hdL );
    ptR = PTR_BAG( hdR );
    hdLL = ptL[1];
    hdRR = ptR[1];
    hdPP = (Bag)(((Int)hdLL-T_INT) * ((Int)hdRR>>1));
    if ( ((Int)hdLL & 3) != T_INT || ((Int)hdRR & 3) != T_INT
      || (((Int)hdRR>>1) != 0
       && (Int)hdPP / ((Int)hdRR>>1) != ((Int)hdLL-T_INT)) ) {
        hdPP = PROD( hdLL, hdRR );
        ptL  = PTR_BAG( hdL );
        ptR  = PTR_BAG( hdR );
    }
    else {
        hdPP = (Bag)(((Int)hdPP >> 1) + T_INT);
    }
    hdP = hdPP;
    for ( i = 2; i <= len; i++ ) {
        hdLL = ptL[i];
        hdRR = ptR[i];
        hdPP = (Bag)(((Int)hdLL-T_INT) * ((Int)hdRR>>1));
        if ( ((Int)hdLL & 3) != T_INT || ((Int)hdRR & 3) != T_INT
          || (((Int)hdRR>>1) != 0
           && (Int)hdPP / ((Int)hdRR>>1) != ((Int)hdLL-T_INT)) ) {
            hdPP = PROD( hdLL, hdRR );
            ptL  = PTR_BAG( hdL );
            ptR  = PTR_BAG( hdR );
        }
        else {
            hdPP = (Bag)(((Int)hdPP>>1) + T_INT);
        }
        hdSS = (Bag)((Int)hdP + (Int)hdPP - T_INT);
        if ( (((Int)hdSS) & 3) != T_INT
          || ((((Int)hdSS)<<1)>>1) != ((Int)hdSS) ) {
            hdSS = SUM( hdP, hdPP );
            ptL  = PTR_BAG( hdL );
            ptR  = PTR_BAG( hdR );
        }
        hdP = hdSS;
    }

    /* return the result                                                   */
    ExitKernel( hdP );
    return hdP;
}


/****************************************************************************
**
*F  ProdVectorMatrix(<hdL>,<hdR>) . . . . .  product of a vector and a matrix
**
**  'ProdVectorMatrix' returns the product of the vector <hdL> and the matrix
**  <hdR>.  The product is the sum of the  rows  of <hdR>, each multiplied by
**  the corresponding entry of <hdL>.
**
**  'ProdVectorMatrix'  is an improved version of 'ProdListList',  which does
**  not  call 'PROD' and  also accummulates  the sum into  one  fixed  vector
**  instead of allocating a new for each product and sum.
*/
Bag       ProdVectorMatrix (Bag hdL, Bag hdR)
{
    Bag           hdP;            /* handle of the product           */
    Bag           hdPP;           /* one summand of product          */
    Bag           hdSS;           /* temporary for sum               */
    Bag           hdQQ;           /* another temporary               */
    Bag           hdLL;           /* one element of left operand     */
    Bag           hdRR;           /* one element of right operand    */
    Bag *         ptRR;           /* pointer into the right operand  */
    Bag           hdRRR;          /* one element from a row          */
    UInt       len;            /* length                          */
    UInt       col;            /* length of the rows              */
    UInt       isVec;          /* is the result a vector          */
    UInt       i, k;           /* loop variables                  */

    /* check the lengths                                                   */
    len = LEN_PLIST( hdL );
    col = LEN_PLIST( ELM_PLIST( hdR, 1 ) );
    if ( len != LEN_PLIST( hdR ) )
        return Error("Vector *: vectors must have the same length", 0, 0);

    /* make the result list by multiplying the first entries               */
    hdP = PROD( ELM_PLIST( hdL, 1 ), ELM_PLIST( hdR, 1 ) );
    isVec = (GET_TYPE_BAG(hdP) == T_VECTOR);

    /* loop over the other entries and multiply                            */
    for ( i = 2; i <= len; i++ ) {
        EnterKernel();
        hdLL = ELM_PLIST( hdL, i );
        hdRR = ELM_PLIST( hdR, i );
        ptRR = PTR_BAG( hdRR );
        if ( hdLL == INT_TO_HD(1) ) {
            for ( k = 1; k <= col; k++ ) {
                hdRRR = ptRR[k];
                hdPP = PTR_BAG(hdP)[k];
                hdSS = (Bag)((Int)hdPP + (Int)hdRRR - T_INT);
                if ( (((Int)hdSS) & 3) != T_INT
                  || ((((Int)hdSS)<<1)>>1) != ((Int)hdSS) ) {
                    hdSS = SUM( hdPP, hdRRR );
                    ptRR = PTR_BAG( hdRR );
                    isVec = isVec && GET_TYPE_BAG(hdSS) <= T_UNKNOWN;
                }
                SET_BAG(hdP, k, hdSS);
            }
        }
        else if ( hdLL == INT_TO_HD(-1) ) {
            for ( k = 1; k <= col; k++ ) {
                hdRRR = ptRR[k];
                hdPP = PTR_BAG(hdP)[k];
                hdSS = (Bag)((Int)hdPP - (Int)hdRRR + T_INT);
                if ( (((Int)hdSS) & 3) != T_INT
                  || (((Int)hdPP) & 3) != T_INT
                  || ((((Int)hdSS)<<1)>>1) != ((Int)hdSS) ) {
                    hdSS = DIFF( hdPP, hdRRR );
                    ptRR = PTR_BAG( hdRR );
                    isVec = isVec && GET_TYPE_BAG(hdSS) <= T_UNKNOWN;
                }
                SET_BAG(hdP, k, hdSS);
            }
        }
        else if ( hdLL != INT_TO_HD(0) ) {
            for ( k = 1; k <= col; k++ ) {
                hdRRR = ptRR[k];
                hdPP = (Bag)(((Int)hdLL-T_INT) * ((Int)hdRRR>>1));
                if ( ((Int)hdLL & 3) != T_INT || ((Int)hdRRR & 3) != T_INT
                  || (((Int)hdRRR>>1) != 0
                   && (Int)hdPP / ((Int)hdRRR>>1) != ((Int)hdLL-T_INT))) {
                    hdPP = PROD( hdLL, hdRRR );
                    ptRR = PTR_BAG( hdRR );
                }
                else {
                    hdPP = (Bag)(((Int)hdPP>>1) + T_INT);
                }
                hdQQ = PTR_BAG(hdP)[k];
                hdSS = (Bag)((Int)hdQQ + (Int)hdPP - T_INT);
                if ( (((Int)hdSS) & 3) != T_INT
                  || ((((Int)hdSS)<<1)>>1) != ((Int)hdSS) ) {
                    hdSS = SUM( hdQQ, hdPP );
                    ptRR = PTR_BAG( hdRR );
                    isVec = isVec && GET_TYPE_BAG(hdSS) <= T_UNKNOWN;
                }
                SET_BAG(hdP, k, hdSS);
            }
        }
        ExitKernel( (Bag)0 );
    }

    /* return the result                                                   */
    if ( isVec )  Retype( hdP, T_VECTOR );
    return hdP;
}


/****************************************************************************
**
*F  PowMatrixInt(<hdL>,<hdR>) . . . . . . .  power of a matrix and an integer
**
**  'PowMatrixInt' returns the <hdR>-th power of the matrix <hdL>, which must
**  be a square matrix of course.
**
**  Note that  this  function also  does the  inversion  of matrices when the
**  exponent is negative.
*/
Bag       PowMatrixInt (Bag hdL, Bag hdR)
{
    Bag           hdP = 0;        /* power, result                   */
    Bag           hdPP;           /* one row of the power            */
    Bag           hdQQ;           /* another row of the power        */
    Bag           hdPPP;          /* one element of the row          */
    Bag           hdLL;           /* one row of left operand         */
    Bag           hdOne;          /* handle of the 1                 */
    Bag           hdZero;         /* handle of the 0                 */
    Bag           hdTmp;          /* temporary handle                */
    UInt       len;            /* length (and width) of matrix    */
    Int                e;              /* exponent                        */
    UInt       i, k, l;        /* loop variables                  */

    /* check that the operand is a *square* matrix                         */
    len = LEN_PLIST( hdL );
    if ( len != LEN_PLIST( ELM_PLIST( hdL, 1 ) ) ) {
        return Error(
          "Matrix operations ^: <mat> must be square",
                      0, 0);
    }
    hdOne = POW( ELM_PLIST( ELM_PLIST( hdL, 1 ), 1 ), INT_TO_HD(0) );
    hdZero = DIFF( hdOne, hdOne );

    /* if the right operand is zero, make the identity matrix              */
    if ( GET_TYPE_BAG(hdR) == T_INT && hdR == INT_TO_HD(0) ) {
        hdP = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
        SET_LEN_PLIST( hdP, len );
        for ( i = 1; i <= len; i++ ) {
            hdPP = NewBag( T_VECTOR, SIZE_PLEN_PLIST( len ) );
            SET_LEN_PLIST( hdPP, len );
            SET_ELM_PLIST( hdP, i, hdPP );
        }
        for ( i = 1; i <= len; i++ ) {
            hdPP = ELM_PLIST( hdP, i );
            for ( k = 1; k <= len; k++ )
                SET_ELM_PLIST( hdPP, k, hdZero );
            SET_ELM_PLIST( hdPP, i, hdOne );
        }
    }

    /* if the right operand is one, make a copy                            */
    if ( GET_TYPE_BAG(hdR) == T_INT && hdR == INT_TO_HD(1) ) {
        hdP = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
        SET_LEN_PLIST( hdP, len );
        for ( i = 1; i <= len; i++ ) {
            hdPP = NewBag( T_VECTOR, SIZE_PLEN_PLIST( len ) );
            SET_LEN_PLIST( hdPP, len );
            SET_ELM_PLIST( hdP, i, hdPP );
        }
        for ( i = 1; i <= len; i++ ) {
            hdPP = ELM_PLIST( hdP, i );
            hdLL = ELM_PLIST( hdL, i );
            for ( k = 1; k <= len; k++ )
                SET_ELM_PLIST( hdPP, k, ELM_PLIST( hdLL, k )  );
        }
    }

    /* if the right operand is negative, invert the matrix                 */
    if ( (GET_TYPE_BAG(hdR) == T_INT && HD_TO_INT(hdR) < 0)
      || (GET_TYPE_BAG(hdR) == T_INTNEG) ) {

        /* make a matrix of the form $ ( Id_<len> | <mat> ) $              */
        hdP = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
        SET_LEN_PLIST( hdP, len );
        for ( i = 1; i <= len; i++ ) {
            hdPP = NewBag( T_VECTOR, SIZE_PLEN_PLIST( 2 * len ) );
            SET_LEN_PLIST( hdPP, len );
            SET_ELM_PLIST( hdP, i, hdPP );
        }
        for ( i = 1; i <= len; i++ ) {
            hdPP = ELM_PLIST( hdP, i );
            for ( k = 1; k <= len; k++ )
                SET_ELM_PLIST( hdPP, k, hdZero );
            SET_ELM_PLIST( hdPP, i, hdOne );
        }
        for ( i = 1; i <= len; i++ ) {
            hdPP = ELM_PLIST( hdP, i );
            hdLL = ELM_PLIST( hdL, i );
            for ( k = 1; k <= len; k++ )
                SET_ELM_PLIST( hdPP, k + len, ELM_PLIST( hdLL, k )  );
        }

        /* make row operations to reach form $ ( <inv> | Id_<len> ) $      */
        /* loop over the columns of <mat>                                  */
        for ( k = len+1; k <= 2*len; k++ ) {
            EnterKernel();

            /* find a nonzero entry in this column                         */
            for ( i = k-len;
                  i <= len
               && (ELM_PLIST( ELM_PLIST(hdP,i), k ) == hdZero
                || EQ( ELM_PLIST( ELM_PLIST(hdP,i), k ), hdZero ) == HdTrue);
                  i++ )
                ;
            if ( len < i )
                return Error("Matrix: <mat> must be invertible", 0, 0);

            /* make the row the <k>-th row and normalize it                */
            hdPP = ELM_PLIST( hdP, i );
            SET_ELM_PLIST( hdP, i, ELM_PLIST( hdP, k-len ) );
            SET_ELM_PLIST( hdP, k-len, hdPP );
            hdPPP = POW( ELM_PLIST( hdPP, k ), INT_TO_HD(-1) );
            for ( l = 1; l <= 2*len; l++ ) {
                hdTmp = PROD( hdPPP, ELM_PLIST( hdPP, l ) );
                SET_ELM_PLIST( hdPP, l, hdTmp );
            }

            /* clear all entries in this column                            */
            for ( i = 1; i <= len; i++ ) {
                hdQQ = ELM_PLIST( hdP, i );
                hdPPP = DIFF( hdZero, ELM_PLIST( hdQQ, k ) );
                if ( i != k-len
                  && hdPPP != hdZero
                  && EQ( hdPPP, hdZero ) == HdFalse ) {
                    for ( l = 1; l <= 2*len; l++ ) {
                        hdTmp = PROD( hdPPP, ELM_PLIST( hdPP, l ) );
                        hdTmp = SUM( ELM_PLIST( hdQQ, l ), hdTmp );
                        SET_ELM_PLIST( hdQQ, l, hdTmp );
                    }
                }
            }

            ExitKernel( (Bag)0 );
        }

        /* throw away the right halves of each row                         */
        for ( i = 1; i <= len; i++ ) {
            Resize( ELM_PLIST( hdP, i ), SIZE_PLEN_PLIST( len ) );
            SET_LEN_PLIST( ELM_PLIST( hdP, i ), len );
        }

        /* assign back to left, invert exponent (only if immediate)        */
        hdL = hdP;
        if ( GET_TYPE_BAG(hdR) == T_INT )  hdR = INT_TO_HD( -HD_TO_INT(hdR) );

    }

    /* repeated squaring with an immediate integer                         */
    /* the loop invariant is: <res> = <p>^<k> * <l>^<e>, <e> < <k>         */
    /* <p> = 0 means that <p> is the identity matrix                       */
    if ( GET_TYPE_BAG(hdR) == T_INT && 1 < HD_TO_INT(hdR) ) {
        hdP = 0;
        k = NUM_TO_UINT(1) << 31;
        e = HD_TO_INT(hdR);
        while ( 1 < k ) {
            hdP = (hdP == 0 ? hdP : ProdListScl( hdP, hdP ));
            k = k / 2;
            if ( k <= e ) {
                hdP = (hdP == 0 ? hdL : ProdListScl( hdP, hdL ));
                e = e - k;
            }
        }
    }

    /* repeated squaring with a large integer                              */
    if ( GET_TYPE_BAG(hdR) != T_INT ) {
        hdP = 0;
        for ( i = GET_SIZE_BAG(hdR)/sizeof(TypDigit); 0 < i; i-- ) {
            k = NUM_TO_UINT(1) << (8*sizeof(TypDigit));
            e = ((TypDigit*) PTR_BAG(hdR))[i-1];
            while ( 1 < k ) {
                hdP = (hdP == 0 ? hdP : ProdListScl( hdP, hdP ));
                k = k / 2;
                if ( k <= e ) {
                    hdP = (hdP == 0 ? hdL : ProdListScl( hdP, hdL ));
                    e = e - k;
                }
            }
        }
    }

    /* return the result                                                   */
    return hdP;
}


/****************************************************************************
**
*F  InitVector()  . . . . . . . . . . . . . . . . . initialize vector package
**
**  'InitVector' initializes the vector package.
*/
void            InitVector (void)
{
    UInt       type;           /* loop variable                   */

    /* install the list functions in the tables                            */
    TabIsList     [T_VECTOR] = 2;
    TabIsList     [T_MATRIX] = 3;
    TabLenList    [T_VECTOR] = LenVector;
    TabElmList    [T_VECTOR] = ElmVector;
    TabElmfList   [T_VECTOR] = ElmfVector;
    TabElmlList   [T_VECTOR] = ElmfVector;
    TabElmrList   [T_VECTOR] = ElmfVector;
    TabElmsList   [T_VECTOR] = ElmsVector;
    TabAssList    [T_VECTOR] = AssVector;
    TabAsssList   [T_VECTOR] = AsssVector;
    TabPosList    [T_VECTOR] = PosVector;
    TabPlainList  [T_VECTOR] = PlainVector;
    TabIsDenseList[T_VECTOR] = IsDenseVector;
    TabIsPossList [T_VECTOR] = IsPossVector;
    TabIsXTypeList[T_VECTOR] = IsXTypeVector;
    TabIsXTypeList[T_MATRIX] = IsXTypeMatrix;

    /* install the default evaluation functions                            */
    EvTab[T_VECTOR] = EvList;
    PrTab[T_VECTOR] = PrList;

    /* install the comparision functions                                   */
    TabEq[T_VECTOR][T_VECTOR] = EqVector;
    TabLt[T_VECTOR][T_VECTOR] = LtVector;

    /* install the binary operations                                       */
    for ( type = T_INT; type <= T_FFE; type++ ) {
        TabSum [type    ][T_VECTOR] = SumSclList;
        TabSum [T_VECTOR][type    ] = SumListScl;
        TabSum [type    ][T_MATRIX] = SumSclList;
        TabSum [T_MATRIX][type    ] = SumListScl;
    }
    TabSum [T_INT   ][T_VECTOR] = SumIntVector;
    TabSum [T_VECTOR][T_INT   ] = SumVectorInt;
    TabSum [T_VECTOR][T_VECTOR] = SumVectorVector;
    TabSum [T_MATRIX][T_MATRIX] = SumListList;

    for ( type = T_INT; type <= T_FFE; type++ ) {
        TabDiff[type    ][T_VECTOR] = DiffSclList;
        TabDiff[T_VECTOR][type    ] = DiffListScl;
        TabDiff[type    ][T_MATRIX] = DiffSclList;
        TabDiff[T_MATRIX][type    ] = DiffListScl;
    }
    TabDiff[T_INT   ][T_VECTOR] = DiffIntVector;
    TabDiff[T_VECTOR][T_INT   ] = DiffVectorInt;
    TabDiff[T_VECTOR][T_VECTOR] = DiffVectorVector;
    TabDiff[T_MATRIX][T_MATRIX] = DiffListList;

    for ( type = T_INT; type <= T_FFE; type++ ) {
        TabProd[type    ][T_VECTOR] = ProdSclList;
        TabProd[T_VECTOR][type    ] = ProdListScl;
        TabProd[type    ][T_MATRIX] = ProdSclList;
        TabProd[T_MATRIX][type    ] = ProdListScl;
    }
    TabProd[T_INT   ][T_VECTOR] = ProdIntVector;
    TabProd[T_VECTOR][T_INT   ] = ProdVectorInt;
    TabProd[T_VECTOR][T_VECTOR] = ProdVectorVector;
    TabProd[T_VECTOR][T_MATRIX] = ProdVectorMatrix;
    TabProd[T_MATRIX][T_VECTOR] = ProdListScl;
    TabProd[T_MATRIX][T_MATRIX] = ProdListScl;
    TabProd[T_VECTOR][T_LISTX ] = ProdListList;
    TabProd[T_MATRIX][T_LISTX ] = ProdSclList;
    TabProd[T_LISTX ][T_MATRIX] = ProdListScl;

    for ( type = T_INT; type <= T_FFE; type++ ) {
        TabQuo [T_VECTOR][type    ] = QuoLists;
        TabQuo [type    ][T_MATRIX] = QuoLists;
        TabQuo [T_MATRIX][type    ] = QuoLists;
    }
    TabQuo [T_VECTOR][T_MATRIX] = QuoLists;
    TabQuo [T_MATRIX][T_MATRIX] = QuoLists;
    TabQuo [T_LISTX ][T_MATRIX] = QuoLists;

    for ( type = T_INT; type <= T_FFE; type++ ) {
        TabMod [type    ][T_VECTOR] = ModLists;
        TabMod [type    ][T_MATRIX] = ModLists;
        TabMod [T_MATRIX][type    ] = ModLists;
    }
    TabMod [T_MATRIX][T_VECTOR] = ModLists;
    TabMod [T_MATRIX][T_MATRIX] = ModLists;
    TabMod [T_MATRIX][T_LISTX ] = ModLists;

    TabPow [T_MATRIX][T_INT   ] = PowMatrixInt;
    TabPow [T_MATRIX][T_INTPOS] = PowMatrixInt;
    TabPow [T_MATRIX][T_INTNEG] = PowMatrixInt;
    TabPow [T_VECTOR][T_MATRIX] = ProdVectorMatrix;
    TabPow [T_MATRIX][T_MATRIX] = PowLists;

    TabComm[T_MATRIX][T_MATRIX] = CommLists;
}
