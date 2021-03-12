/****************************************************************************
**
*A  blister.c                   GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This  file contains the functions  that mainly operate  on boolean lists.
**  Because boolean lists are  just a special case  of lists many  things are
**  done in the list package.
**
**  A *boolean list* is a list that has no holes and contains only 'true' and
**  'false'.  For  the full definition of  boolean list  see chapter "Boolean
**  Lists" in the {\GAP} Manual.  Read  also the section "More  about Boolean
**  Lists" about the different internal representations of such lists.
**
**  A list that is known to be a boolean list is represented by a bag of type
**  'T_BLIST', which has the following format:
**
**      +-------+-------+-------+-------+- - - -+-------+
**      |logical| block | block | block |       | last  |
**      |length |   0   |   1   |   2   |       | block |
**      +-------+-------+-------+-------+- - - -+-------+
**             /         \
**        .---'           `-----------.
**       /                             \
**      +---+---+---+---+- - - -+---+---+
**      |bit|bit|bit|bit|       |bit|bit|
**      | 0 | 1 | 2 | 3 |       |n-1| n |
**      +---+---+---+---+- - - -+---+---+
**
**  The  first  entry is  the logical  length of the list,  represented as  a
**  {\GAP} immediate integer.  The other entries are blocks, represented as C
**  unsigned  long integer.   Each  block corresponds  to  <n>  (usually  32)
**  elements of the list.  The <j>-th bit (the bit corresponding to '2\^<j>')
**  in  the <i>-th block  is 1 if  the element  '<list>[BIPEB*<i>+<j>+1]'  it
**  'true'  and '0' if  it  is 'false'.  If the logical length of the boolean
**  list is not a multiple of BIPEB the  last block will contain unused bits,
**  which are then zero.
**
**  Note that a list represented by a  bag of type 'T_LIST'  might still be a
**  boolean list.  It is just that the kernel does not known this.
**
**  This package consists of three parts.
**
**  The  first  part  consists  of  the  macros  'BIPEB',  'SIZE_PLEN_BLIST',
**  'PLEN_SIZE_BLIST',   'LEN_BLIST',   'SET_LEN_BLIST',   'ELM_BLIST',   and
**  'SET_ELM_BLIST'.   They  determine the  representation of boolean  lists.
**  The  rest  of the {\GAP} kernel  uses those macros  to access and  modify
**  boolean lists.
**
**  The  second  part  consists  of  the  functions  'LenBlist',  'ElmBlist',
**  'ElmsBlist',   'AssBlist',    'AsssBlist',   'PosBlist',    'PlainBlist',
**  'IsDenseBlist',  'IsPossBlist', 'EqBlist', and  'LtBlist'.  They  are the
**  functions required by the  generic lists  package.  Using these functions
**  the other parts of  the {\GAP} kernel can access and modify boolean lists
**  without actually being aware that they are dealing with a boolean list.
**
**  The  third  part  consists  of  the  functions  'IsBlist',  'FunIsBlist',
**  'FunBlistList',   'FunListBlist',   'FunSizeBlist',   'FunIsSubsetBlist',
**  'FunUniteBlist',  'FunIntersectBlist',   and  'FunSubtractBlist'.   These
**  functions make it possible to make  boolean lists, either by converting a
**  list to a boolean list,  or  by computing the characteristic boolean list
**  of  a sublist, or  by computing the union, intersection or difference  of
**  two boolean lists.
**
*N  1992/12/16 martin should have 'LtBlist'
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
#include        "set.h"                 /* 'IsSet', 'SetList'              */

#include        "blister.h"             /* declaration part of the package */


/****************************************************************************
**
*V  BIPEB . . . . . . . . . . . . . . . . . . . . . . . . . .  bits per block
**
**  'BIPEB' is the number of bits per block, usually 32.
**
**  'BIPEB' is defined in the declaration part of this package as follows:
**
#define BIPEB                           (sizeof(unsigned long) * 8L)
*/


/****************************************************************************
**
*F  PLEN_SIZE_BLIST(<size>) . .  physical length from size for a boolean list
**
**  'PLEN_SIZE_BLIST'  computes  the  physical  length  (e.g.  the  number of
**  elements that could be stored  in a list) from the <size> (as reported by
**  'GET_SIZE_BAG') for a boolean list.
**
**  Note that 'PLEN_SIZE_BLIST' is a macro, so  do not call it with arguments
**  that have sideeffects.
**
**  'PLEN_SIZE_BLIST'  is defined in the declaration  part of this package as
**  follows:
**
#define PLEN_SIZE_BLIST(GET_SIZE_BAG)           ((((GET_SIZE_BAG)-SIZE_HD)/SIZE_HD) * BIPEB)
*/


/****************************************************************************
**
*F  SIZE_PLEN_BLIST(<plen>)size for a boolean list with given physical length
**
**  'SIZE_PLEN_BLIST' returns  the size  that a boolean list  with  room  for
**  <plen> elements must at least have.
**
**  Note that 'SIZE_PLEN_BLIST' is a macro, so do not call it with  arguments
**  that have sideeffects.
**
**  'SIZE_PLEN_BLIST' is  defined  in the declaration part of this package as
**  follows:
**
#define SIZE_PLEN_BLIST(PLEN)        (SIZE_HD+((PLEN)+BIPEB-1)/BIPEB*SIZE_HD)
*/


/****************************************************************************
**
*F  LEN_BLIST(<hdList>) . . . . . . . . . . . . . .  length of a boolean list
**
**  'LEN_BLIST' returns the logical length of the boolean list <hdBlist>,  as
**  a C integer.
**
**  Note that 'LEN_BLIST' is a macro, so do not call it  with  arguments that
**  have sideeffects.
**
**  'LEN_BLIST' is defined in the declaration part of the package as follows:
**
#define LEN_BLIST(LIST)                 (HD_TO_INT(PTR_BAG(LIST)[0]))
*/


/****************************************************************************
**
*F  SET_LEN_BLIST(<hdList>,<len>) . . . . .  set the length of a boolean list
**
**  'SET_LEN_BLIST' sets the length of the boolean list <hdList> to the value
**  <len>, which must be a positive C integer.
**
**  Note that 'SET_LEN_BLIST' is a macro, so do  not  call it with  arguments
**  that have sideeffects.
**
**  'SET_LEN_BLIST' is  defined in the declaration part of  this  package  as
**  follows:
**
#define SET_LEN_BLIST(LIST,LEN)         (PTR_BAG(LIST)[0] = INT_TO_HD(LEN))
*/


/****************************************************************************
**
*F  ELM_BLIST(<hdList>,<pos>) . . . . . . . . . . . element of a boolean list
**
**  'ELM_BLIST'  return the  <pos>-th element of the  boolean  list <hdList>,
**  which is either 'true' or 'false'.  <pos> must be a positive integer less
**  than or equal to the length of <hdList>.
**
**  Note that 'ELM_BLIST' is a macro, so do not call it  with arguments  that
**  have sideeffects.
**
**  'ELM_BLIST' is defined in the declaration part of the package as follows:
**
#define ELM_BLIST(LIST,POS)             \
  (((unsigned long*)(PTR_BAG(LIST)+1))[((POS)-1)/BIPEB]&(1L<<((POS)-1)%BIPEB) ? \
   HdTrue : HdFalse)
*/


/****************************************************************************
**
*F  SET_ELM_BLIST(<hdList>,<pos>,<val>) . .  set an element of a boolean list
**
**  'SET_ELM_BLIST' sets  the element  at position <pos>  in the boolean list
**  <hdList> to the  value <val>.  <pos> must be a positive integer less than
**  or equal to the  length of <hdList>.  <val>  must be either  'HdTrue'  or
**  'HdFalse'.
**
**  Note that  'SET_ELM_BLIST' is  a macro, so do not  call it with arguments
**  that have sideeffects.
**
**  'SET_ELM_BLIST' is defined in  the  declaration  part of this  package as
**  follows:
**
#define SET_ELM_BLIST(LIST,POS,VAL)     \
 ((VAL) == HdTrue ?                     \
  (((unsigned long*)(PTR_BAG(LIST)+1))[((POS)-1)/BIPEB]|=(1L<<((POS)-1)%BIPEB)):\
  (((unsigned long*)(PTR_BAG(LIST)+1))[((POS)-1)/BIPEB]&=~(1L<<((POS)-1)%BIPEB)))
*/


/****************************************************************************
**
*F  LenBlist(<hdList>)  . . . . . . . . . . . . . .  length of a boolean list
**
**  'LenBlist' returns  the length  of  the  boolean  list  <hdList>  as a  C
**  integer.
**
**  'LenBlist' is the function in 'TabLenList' for boolean lists.
*/
Int            LenBlist (Bag hdList)
{
    return LEN_BLIST( hdList );
}


/****************************************************************************
**
*F  ElmBlist(<hdList>,<pos>)  . . . . . . select an element of a boolean list
**
**  'ElmBlist'  selects  the  element at position <pos> of  the  boolean list
**  <hdList>.  It is the responsibility of the caller to ensure that <pos> is
**  a  positive integer.  An error is signalled if  <pos> is  larger than the
**  length of <hdList>.
**
**  'ElmfBlist' does  the same thing than 'ElmBlist', but need not check that
**  <pos>  is  less than or equal  to  the  length of <hdList>, this  is  the
**  responsibility of the caller.
**
**  'ElmBlist'  is  the   function   in  'TabElmBlist'  for   boolean  lists.
**  'ElmfBlist'  is  the  function  in  'TabElmfBlist',  'TabElmlBlist',  and
**  'TabElmrBlist' for boolean lists.
*/
Bag       ElmBlist (Bag hdList, Int pos)
{

    /* check the position                                                  */
    if ( LEN_BLIST( hdList ) < pos ) {
        return Error(
          "List Element: <list>[%d] must have a value",
                     pos, 0 );
    }

    /* select and return the element                                       */
    return ELM_BLIST( hdList, pos );
}

Bag       ElmfBlist (Bag hdList, Int pos)
{
    /* select and return the element                                       */
    return ELM_BLIST( hdList, pos );
}


/****************************************************************************
**
*F  ElmsBlist(<hdList>,<hdPoss>)  . . .  select a sublist from a boolean list
**
**  'ElmsBlist'  returns a new list containing the elements at  the positions
**  given  in the list  <hdPoss> from  the boolean  list <hdList>.  It is the
**  responsibility  of the  caller  to  ensure  that  <hdPoss>  is  dense and
**  contains only positive integers.  An error is signalled  if an element of
**  <hdPoss> is larger than the length of <hdList>.
**
**  'ElmsBlist' is the function in 'TabElmsList' for boolean lists.
*/
Bag       ElmsBlist (Bag hdList, Bag hdPoss)
{
    Bag           hdElms;         /* selected sublist, result        */
    Int                lenList;        /* length of <list>                */
    Bag           hdElm;          /* one element from <list>         */
    Int                lenPoss;        /* length of <positions>           */
    Int                pos;            /* <position> as integer           */
    Int                inc;            /* increment in a range            */
    UInt       block;          /* one block of <elms>             */
    UInt       bit;            /* one bit of a block              */
    Int                i;              /* loop variable                   */

    /* general code                                                        */
    if ( GET_TYPE_BAG(hdPoss) != T_RANGE ) {

        /* get the length of <list>                                        */
        lenList = LEN_BLIST( hdList );

        /* get the length of <positions>                                   */
        lenPoss = LEN_LIST( hdPoss );

        /* make the result list                                            */
        hdElms = NewBag( T_BLIST, SIZE_PLEN_BLIST( lenPoss ) );
        SET_LEN_BLIST( hdElms, lenPoss );

        /* loop over the entries of <positions> and select                 */
        block = 0;  bit = 1;
        for ( i = 1; i <= lenPoss; i++ ) {

            /* get <position>                                              */
            pos = HD_TO_INT( ELMF_LIST( hdPoss, i ) );
            if ( lenList < pos ) {
                return Error(
                  "List Elements: <list>[%d] must have a value",
                             pos, 0 );
            }

            /* select the element                                          */
            hdElm = ELM_BLIST( hdList, pos );

            /* assign the element into <elms>                              */
            if ( hdElm == HdTrue )
                block |= bit;
            bit <<= 1;
            if ( bit == 0 || i == lenPoss ) {
                ((UInt *)(PTR_BAG(hdElms)+1))[(i-1)/BIPEB] = block;
                block = 0;
                bit = 1;
            }

        }

    }

    /* special code for ranges                                             */
    /*N 1992/12/15 martin special code for ranges with increment 1         */
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
                         pos, 0 );
        }
        if ( lenList < pos + (lenPoss-1) * inc ) {
            return Error(
              "List Elements: <list>[%d] must have a value",
                         pos + (lenPoss-1) * inc, 0 );
        }

        /* make the result list                                            */
        hdElms = NewBag( T_BLIST, SIZE_PLEN_BLIST( lenPoss ) );
        SET_LEN_BLIST( hdElms, lenPoss );

        /* loop over the entries of <positions> and select                 */
        block = 0;  bit = 1;
        for ( i = 1; i <= lenPoss; i++, pos += inc ) {

            /* select the element                                          */
            hdElm = ELM_BLIST( hdList, pos );

            /* assign the element to <elms>                                */
            if ( hdElm == HdTrue )
                block |= bit;
            bit <<= 1;
            if ( bit == 0 || i == lenPoss ) {
                ((UInt *)(PTR_BAG(hdElms)+1))[(i-1)/BIPEB] = block;
                block = 0;
                bit = 1;
            }

        }

    }

    /* return the result                                                   */
    return hdElms;
}


/****************************************************************************
**
*F  AssBlist(<hdList>,<pos>,<hdVal>)  . . . . . . .  assign to a boolean list
**
**  'AssBlist' assigns the  value <hdVal> to the boolean list <hdList> at the
**  position <pos>.  It  is the  responsibility of the caller  to ensure that
**  <pos> is positive, and that <hdVal> is not 'HdVoid'.
**
**  'AssBlist' is the function in 'TabAssList' for boolean lists.
**
**  If <pos>  is less than or equal to the logical length of the boolean list
**  and <hdVal> is 'true' or 'false' the assignment  is  done by  setting the
**  corresponding bit.  If <pos> is one more  than the  logical length of the
**  boolean list  the  assignment  is done  by resizing  the boolean  list if
**  necessary,  setting the corresponding  bit  and incrementing the  logical
**  length  by one.  Otherwise the  boolean list is  converted to an ordinary
**  list and the assignment is performed the ordinary way.
*/
Bag       AssBlist (Bag hdList, Int pos, Bag hdVal)
{
    Int                plen;           /* physical length of <list>       */

    /* if <pos> is less than the logical length and <elm> is 'true'        */
    if      ( pos <= LEN_BLIST(hdList) && hdVal == HdTrue ) {
        SET_ELM_BLIST( hdList, pos, HdTrue );
    }

    /* if <i> is less than the logical length and <elm> is 'false'         */
    else if ( pos <= LEN_BLIST(hdList) && hdVal == HdFalse ) {
        SET_ELM_BLIST( hdList, pos, HdFalse );
    }

    /* if <i> is one more than the logical length and <elm> is 'true'      */
    else if ( pos == LEN_BLIST(hdList)+1 && hdVal == HdTrue ) {
        if ( GET_SIZE_BAG(hdList) < SIZE_PLEN_BLIST(pos) )
            Resize( hdList, SIZE_PLEN_BLIST(pos) );
        SET_LEN_BLIST( hdList, pos );
        SET_ELM_BLIST( hdList, pos, HdTrue );
    }

    /* if <i> is one more than the logical length and <elm> is 'true'      */
    else if ( pos == LEN_BLIST(hdList)+1 && hdVal == HdFalse ) {
        if ( GET_SIZE_BAG(hdList) < SIZE_PLEN_BLIST(pos) )
            Resize( hdList, SIZE_PLEN_BLIST(pos) );
        SET_LEN_BLIST( hdList, pos );
        SET_ELM_BLIST( hdList, pos, HdFalse );
    }

    /* otherwise convert to ordinary list and assign as in 'AssList'       */
    else {
        PLAIN_LIST( hdList );
        Retype( hdList, T_LIST );
        if ( LEN_PLIST(hdList) < pos ) {
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
*F  AsssBlist(<hdList>,<hdPoss>,<hdVals>)  assign several elements to a blist
**
**  'AsssBlist' assignes the values from  the list <hdVals>  at the positions
**  given in  the  list  <hdPoss> to the boolean list <hdList>.   It  is  the
**  responsibility  of  the  caller to  ensure  that  <hdPoss>  is  dense and
**  contains only positive integers, that <hdPoss> and <hdVals> have the same
**  length, and that <hdVals> is dense.
**
**  'AsssBlist' is the function in 'TabAsssList' for boolean lists.
**
**  'AsssBlist' simply  converts  the boolean list to a  plain  list and then
**  does the same  stuff as  'AsssPlist'.   This  is because a boolean is not
**  very likely to stay a boolean list after the assignment.
*/
Bag       AsssBlist (Bag hdList, Bag hdPoss, Bag hdVals)
{
    /* convert <list> to a plain list                                      */
    PLAIN_LIST( hdList );
    Retype( hdList, T_LIST );

    /* and delegate                                                        */
    return ASSS_LIST( hdList, hdPoss, hdVals );
}


/****************************************************************************
**
*F  PosBlist(<hdList>,<hdVal>,<start>) . position of an elm in a boolean list
**
**  'PosBlist'  returns  the  position  of  the first occurence of  the value
**  <hdVal>, which  may be an object of arbitrary  type, in the  boolean list
**  <hdList> after <start> as  a  C  integer.   If <hdVal> does not  occur in
**  <hdList> after <start>, then 0 is returned.
**
**  'PosBlist' is the function in 'TabPosList' for boolean lists.
*/
Int            PosBlist (Bag hdBlist, Bag hdVal, Int start)
{
    Int                k;              /* position, result                */
    Int                len;            /* logical length of the list      */
    UInt       * ptBlist;      /* pointer to the blocks           */
    Int                i,  j;          /* loop variables                  */

    len = LEN_BLIST(hdBlist);

    /* look just beyond end                                                */
    if ( len == start ) {
        k = 0;
    }

    /* look for 'true'                                                     */
    else if ( hdVal == HdTrue ) {
        ptBlist = (UInt *)(PTR_BAG(hdBlist)+1);
        if ( ptBlist[start/BIPEB] >> (start%BIPEB) != 0 ) {
            i = start/BIPEB;
            for ( j=start%BIPEB; j<BIPEB && (ptBlist[i]&(NUM_TO_UINT(1)<<j))==0; j++ )
                ;
        }
        else {
            for ( i=start/BIPEB+1; i<(len-1)/BIPEB && ptBlist[i]==0; i++ )
                ;
            for ( j=0; j<BIPEB && (ptBlist[i]&(NUM_TO_UINT(1)<<j))==0; j++ )
                ;
        }
        k = (BIPEB*i+j+1 <= len ? BIPEB*i+j+1 : 0);
    }

    /* look for 'false'                                                    */
    else if ( hdVal == HdFalse ) {
        ptBlist = (UInt *)(PTR_BAG(hdBlist)+1);
        if ( ~ptBlist[start/BIPEB] >> (start%BIPEB) != 0 ) {
            i = start/BIPEB;
            for ( j=start%BIPEB; j<BIPEB && (~ptBlist[i]&(NUM_TO_UINT(1)<<j))==0; j++ )
                ;
        }
        else {
            for ( i=start/BIPEB+1; i<(len-1)/BIPEB && ~ptBlist[i]==0; i++ )
                ;
            for ( j=0; j<BIPEB && (~ptBlist[i]&(NUM_TO_UINT(1)<<j))==0; j++ )
                ;
        }
        k = (BIPEB*i+j+1 <= len ? BIPEB*i+j+1 : 0);
    }

    /* look for something else                                             */
    else {
        k = 0;
    }

    /* return the position                                                 */
    return k;
}


/****************************************************************************
**
*F  PlainBlist(<hdList>)  . . .  convert a boolean list into an ordinary list
**
**  'PlainBlist' converts the boolean list <hdList> to a plain list.
**
**  'PlainBlist' is the function in 'TabPlainList' for boolean lists.
*/
void            PlainBlist (Bag hdList)
{
    Int                lenList;        /* length of <list>                */
    Int                i;              /* loop variable                   */

    /* resize the list and retype it, in this order                        */
    lenList = LEN_BLIST( hdList );
    Resize( hdList, SIZE_PLEN_PLIST( lenList ) );
    Retype( hdList, T_LIST );
    SET_LEN_PLIST( hdList, lenList );

    /* replace the bits by 'HdTrue' or 'HdFalse' as the case may be        */
    /* this must of course be done from the end of the list backwards      */
    for ( i = lenList; 0 < i; i-- )
        SET_ELM_PLIST( hdList, i, ELM_BLIST( hdList, i ) );

}


/****************************************************************************
**
*F  IsDenseBlist(<hdList>)  . . .  dense list test function for boolean lists
**
**  'IsDenseBlist' returns 1, since boolean lists are always dense.
**
**  'IsDenseBlist' is the function in 'TabIsDenseBlist' for boolean lists.
*/
Int            IsDenseBlist (Bag hdList)
{
    return 1;
}


/****************************************************************************
**
*F  IsPossBlist(<hdList>) . .  positions list test function for boolean lists
**
**  'IsPossBlist' returns  1 if  <hdList> is  empty, and 0 otherwise, since a
**  boolean list is a positions list if and only if it is empty.
*/
Int            IsPossBlist (Bag hdList)
{
    return LEN_BLIST(hdList) == 0;
}


/****************************************************************************
**
*F  EqBlist(<hdL>,<hdR>)  . . . . . . . . test if two boolean lists are equal
**
**  'EqBlist'  returns 'true'  if  the two boolean lists <hdL> and  <hdR> are
**  equal and 'false' otherwise.
**
**  Is called from the 'EQ' binop so both  operands  are  already  evaluated.
*/
Bag       EqBlist (Bag hdL, Bag hdR)
{
    Int                lenL;           /* length of the left operand      */
    Int                lenR;           /* length of the right operand     */
    UInt       * ptL;          /* pointer to the left operand     */
    UInt       * ptR;          /* pointer to the right operand    */
    UInt       i;              /* loop variable                   */

    /* get the lengths of the lists and compare them                       */
    lenL = LEN_BLIST( hdL );
    lenR = LEN_BLIST( hdR );
    if ( lenL != lenR ) {
        return HdFalse;
    }

    /* test for equality blockwise                                         */
    ptL = (UInt *)(PTR_BAG(hdL)+1);
    ptR = (UInt *)(PTR_BAG(hdR)+1);
    for ( i = (lenL+BIPEB-1)/BIPEB; 0 < i; i-- ) {
        if ( *ptL++ != *ptR++ )
            return HdFalse;
    }

    /* no differences found, the lists are equal                           */
    return HdTrue;
}


/****************************************************************************
**
*F  IsBlist(<hdList>) . . . . . . . . . test whether a list is a boolean list
**
**  'IsBlist' returns 1 if the list  <hdList> is a boolean list, i.e., a list
**  that has no holes and contains only 'true'  and 'false', and 0 otherwise.
**  As a sideeffect 'IsBlist' changes the  representation  of  boolean  lists
**  into the compact representation of type 'T_BLIST' described above.
*/
Int            IsBlist (Bag hdList)
{
    UInt       isBlist;        /* result of the test              */
    UInt       len;            /* logical length of the list      */
    UInt       block;          /* one block of the boolean list   */
    UInt       bit;            /* one bit of a block              */
    UInt       i;              /* loop variable                   */

    /* if <hdList> is known to be a boolean list, it is very easy          */
    if ( GET_TYPE_BAG(hdList) == T_BLIST ) {
        isBlist = 1;
    }

    /* if <hdList> is not a list, its not a boolean list (convert to list) */
    else if ( ! IS_LIST( hdList ) ) {
        isBlist = 0;
    }

    /* otherwise test if there are holes and if all elements are boolean   */
    else {

        /* test that all elements are bound and either 'true' or 'false'   */
        len = LEN_LIST( hdList );
        for ( i = 1; i <= len; i++ ) {
            if ( ELMF_LIST( hdList, i ) == 0
              || (ELMF_LIST( hdList, i ) != HdTrue
               && ELMF_LIST( hdList, i ) != HdFalse) ) {
                break;
            }
        }

        /* if <hdList> is a boolean list, change its representation        */
        isBlist = (len < i);
        if ( isBlist ) {
            block = 0;
            bit = 1;
            for ( i = 1; i <= len; i++ ) {
                if ( ELMF_LIST( hdList, i ) == HdTrue )
                    block |= bit;
                bit = bit << 1;
                if ( bit == 0 || i == len ) {
                    ((UInt *)(PTR_BAG(hdList)+1))[(i-1)/BIPEB] = block;
                    block = 0;
                    bit = 1;
                }
            }
            Retype( hdList, T_BLIST );
            Resize( hdList, SIZE_PLEN_BLIST( len ) );
            SET_LEN_BLIST( hdList, len );
        }

    }

    /* return the result                                                   */
    return isBlist;
}


/****************************************************************************
**
*F  FunIsBlist(<hdCall>)  . . . . . . . . test if an object is a boolean list
**
**  'FunIsBlist' implements the internal function 'IsBlist'.
**
**  'IsBlist( <obj> )'
**
**  'IsBlist' returns 'true' if the  object  <obj>  is  a  boolean  list  and
**  'false' otherwise.  An object is a boolean list if it is a lists  without
**  holes containing only 'true' and 'false'.  Will cause an  error if  <obj>
**  <obj> is an unbound variable.
*/
Bag       FunIsBlist (Bag hdCall)
{
    Bag           hdObj;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: IsBlist( <obj> )",0,0);
    hdObj = EVAL( PTR_BAG(hdCall)[1] );
    if ( hdObj == HdVoid )
        return Error("IsBlist: function must return a value",0,0);

    /* let 'IsBlist' do the work                                           */
    return IsBlist( hdObj ) ? HdTrue : HdFalse;
}


/****************************************************************************
**
*F  FunBlistList(<hdCall>)  . . . . . . .  make a boolean list from a sublist
**
**  'FunBlistList' implements the internal function 'BlistList'.
**
**  'BlistList( <list>, <sub> )'
**
**  'BlistList'  creates a boolean  list   that describes the  list <sub>  as
**  sublist of the  list <list>.  The  result is a  new boolean list <blist>,
**  which has the same  length as <list>, such  that '<blist>[<i>]' is 'true'
**  if '<list>[<i>]' is an element of <sub> and 'false' otherwise.
**
**  'BlistList' is most effective if <list> is a set, but can be used with an
**  arbitrary list that has no holes.
*/
Bag       FunBlistList (Bag hdCall)
{
    Bag           hdBlist;        /* handle of the result            */
    UInt       * ptBlist;      /* pointer to the boolean list     */
    UInt       block;          /* one block of boolean list       */
    UInt       bit;            /* one bit of block                */
    Bag           hdList;         /* handle of the first argument    */
    UInt       lnList;         /* logical length of the list      */
    Bag           hdSub;          /* handle of the second argument   */
    Bag           * ptSub;        /* pointer to the sublist          */
    UInt       lnSub;          /* logical length of sublist       */
    UInt       i, j, k, l;     /* loop variables                  */
    Int                s, t;           /* elements of a range             */

    /* get and check the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error("usage: BlistList( <list>, <sub> )",0,0);
    hdList = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IS_LIST(hdList) )
        return Error("BlistList: <list> must be a list",0,0);
    hdSub = EVAL( PTR_BAG(hdCall)[2] );
    if ( ! IS_LIST(hdSub) )
        return Error("BlistList: <sub> must be a list",0,0);

    /* for a range as subset of a range, it is extremly easy               */
    if ( GET_TYPE_BAG(hdList) == T_RANGE && GET_TYPE_BAG(hdSub) == T_RANGE ) {

        /* allocate the boolean list and get pointer                       */
        lnList  = LEN_RANGE( hdList );
        lnSub   = LEN_RANGE( hdSub );
        hdBlist = NewBag( T_BLIST, SIZE_PLEN_BLIST( lnList ) );
        SET_BAG(hdBlist, 0,  INT_TO_HD(lnList) );
        ptBlist = (UInt *)(PTR_BAG(hdBlist)+1);

        /* get the bounds of the subset with respect to the boolean list   */
        s = HD_TO_INT( ELM_RANGE( hdList, 1 ) );
        t = HD_TO_INT( ELM_RANGE( hdSub, 1 ) );
        if ( s <= t )  i = t - s + 1;
        else           i = 1;
        if ( i + lnSub - 1 <= lnList )  j = i + lnSub - 1;
        else                            j = lnList;

        /* set the corresponding entries to 'true'                         */
        for ( k = i; k <= j && (k-1)%BIPEB != 0; k++ )
            ptBlist[(k-1)/BIPEB] |= (NUM_TO_UINT(1) << (k-1)%BIPEB);
        for ( ; k+BIPEB <= j; k += BIPEB )
            ptBlist[(k-1)/BIPEB] = ~NUM_TO_UINT(0);
        for ( ; k <= j; k++ )
            ptBlist[(k-1)/BIPEB] |= (NUM_TO_UINT(1) << (k-1)%BIPEB);

    }

    /* for a list as subset of a range, we need basically no search        */
    else if ( GET_TYPE_BAG(hdList) == T_RANGE
          && (GET_TYPE_BAG(hdSub) == T_LIST || GET_TYPE_BAG(hdSub) == T_SET) ) {

        /* allocate the boolean list and get pointer                       */
        lnList  = LEN_RANGE( hdList );
        lnSub   = LEN_LIST( hdSub );
        hdBlist = NewBag( T_BLIST, SIZE_PLEN_BLIST( lnList ) );
        SET_BAG(hdBlist, 0,  INT_TO_HD(lnList) );
        ptBlist = (UInt *)(PTR_BAG(hdBlist)+1);
        ptSub = PTR_BAG(hdSub);

        /* loop over <sub> and set the corresponding entries to 'true'     */
        s = HD_TO_INT( ELM_RANGE( hdList, 1 ) );
        for ( l = 1; l <= LEN_LIST(hdSub); l++ ) {
            if ( ptSub[l] != 0 ) {

                /* if <sub>[<l>] is an integer it is very easy             */
                if ( GET_TYPE_BAG( ptSub[l] ) == T_INT ) {
                    t = HD_TO_INT( ptSub[l] ) - s + 1;
                    if ( 0 < t && t <= lnList )
                        ptBlist[(t-1)/BIPEB] |= (NUM_TO_UINT(1) << (t-1)%BIPEB);
                }

                /* otherwise it may be a record, let 'PosRange' handle it  */
                else {
                    k = PosRange( hdList, ptSub[l], 0 );
                    if ( k != 0 )
                        ptBlist[(k-1)/BIPEB] |= (NUM_TO_UINT(1) << (k-1)%BIPEB);
                }

            }
        }

    }

    /* if <list> is a set we have two possibilities                        */
    else if ( IsSet( hdList ) ) {

        /* get the length of <list> and its logarithm                      */
        lnList = LEN_PLIST( hdList );
        for ( i = lnList, l = 0; i != 0; i >>= 1, l++ ) ;
        if ( GET_TYPE_BAG(hdSub) != T_LIST && GET_TYPE_BAG(hdSub) != T_SET )
            IsList( hdSub );
        lnSub = LEN_LIST( hdSub );

        /* if <sub> is small, we loop over <sub> and use binary search     */
        if ( l * lnSub < 2 * lnList ) {

            /* allocate the boolean list and get pointer                   */
            hdBlist = NewBag( T_BLIST, SIZE_PLEN_BLIST( lnList ) );
            SET_BAG(hdBlist, 0,  INT_TO_HD(lnList) );

            /* run over the elements of <sub> and search for the elements  */
            for ( l = 1; l <= LEN_LIST(hdSub); l++ ) {
                if ( PTR_BAG(hdSub)[l] != 0 ) {

                    /* perform the binary search to find the position      */
                    i = 0;  k = lnList+1;
                    while ( i+1 < k ) {
                        j = (i + k) / 2;
                        if ( LT( PTR_BAG(hdList)[j], PTR_BAG(hdSub)[l] ) == HdTrue )
                            i = j;
                        else
                            k = j;
                    }

                    /* set bit if <sub>[<l>] was found at position k       */
                    if ( k <= lnList
                      && EQ( PTR_BAG(hdList)[k], PTR_BAG(hdSub)[l] ) == HdTrue )
                        ((UInt *)(PTR_BAG(hdBlist)+1))[(k-1)/BIPEB]
                            |= (NUM_TO_UINT(1) << (k-1)%BIPEB);
                }
            }

        }

        /* if <sub> is large, run over both list in parallel               */
        else {

            /* turn the <sub> into a set for faster searching              */
            if ( ! IsSet( hdSub ) )  hdSub = SetList( hdSub );

            /* allocate the boolean list and get pointer                   */
            hdBlist = NewBag( T_BLIST, SIZE_PLEN_BLIST( lnList ) );
            SET_BAG(hdBlist, 0,  INT_TO_HD(lnList) );

            /* run over the elements of <list>                             */
            k = 1;
            block = 0;
            bit   = 1;
            for ( l = 1; l <= lnList; l++ ) {

                /* test if <list>[<l>] is in <sub>                         */
                while ( k <= lnSub
                     && LT(PTR_BAG(hdSub)[k],PTR_BAG(hdList)[l]) == HdTrue )
                    k++;

                /* if <list>[<k>] is in <sub> set the current bit in block */
                if ( k <= lnSub
                  && EQ(PTR_BAG(hdSub)[k],PTR_BAG(hdList)[l]) == HdTrue ) {
                    block |= bit;
                    k++;
                }

                /* if block is full add it to boolean list and start next  */
                bit = bit << 1;
                if ( bit == 0 || l == lnList ) {
                    ((UInt *)(PTR_BAG(hdBlist)+1))[(l-1)/BIPEB] = block;
                    block = 0;
                    bit   = 1;
                }

            }
        }

    }

    /* if <list> is not a set, we have to use brute force                  */
    else {

        /* convert left argument to an ordinary list, ignore return value  */
        i = IsList( hdList );

        /* turn <sub> into a set for faster searching                      */
        if ( ! IsSet( hdSub ) )  hdSub = SetList( hdSub );

        /* allocate the boolean list and get pointer                       */
        lnList  = LEN_LIST( hdList );
        lnSub   = LEN_PLIST( hdSub );
        hdBlist = NewBag( T_BLIST, SIZE_PLEN_BLIST( lnList ) );
        SET_BAG(hdBlist, 0,  INT_TO_HD(lnList) );

        /* run over the elements of <list>                                 */
        k = 1;
        block = 0;
        bit   = 1;
        for ( l = 1; l <= lnList; l++ ) {

            /* test if <list>[<l>] is in <sub>                             */
            if ( l == 1 || LT(PTR_BAG(hdList)[l-1],PTR_BAG(hdList)[l]) == HdTrue ) {
                while ( k <= lnSub
                     && LT(PTR_BAG(hdSub)[k],PTR_BAG(hdList)[l]) == HdTrue )
                    k++;
            }
            else {
                i = 0;  k = LEN_PLIST(hdSub) + 1;
                while ( i+1 < k ) {
                    j = (i + k) / 2;
                    if ( LT( PTR_BAG(hdSub)[j], PTR_BAG(hdList)[l] ) == HdTrue )
                        i = j;
                    else
                        k = j;
                }
            }

            /* if <list>[<k>] is in <sub> set the current bit in the block */
            if ( k <= lnSub
              && EQ( PTR_BAG(hdSub)[k], PTR_BAG(hdList)[l] ) == HdTrue ) {
                block |= bit;
                k++;
            }

            /* if block is full add it to the boolean list and start next  */
            bit = bit << 1;
            if ( bit == 0 || l == lnList ) {
                ((UInt *)(PTR_BAG(hdBlist)+1))[(l-1)/BIPEB] = block;
                block = 0;
                bit   = 1;
            }

        }

    }

    /* return the boolean list                                             */
    return hdBlist;
}


/****************************************************************************
**
*F  FunListBlist(<hdCall>)  . . . . . . .  make a sublist from a boolean list
**
**  'FunListBlist' implements the internal function 'ListBlist'.
**
**  'ListBlist( <list>, <blist> )'
**
**  'ListBlist' returns the  sublist of the  elements of the list  <list> for
**  which the boolean list   <blist>, which must   have  the same  length  as
**  <list>, contains 'true'.  The order of the elements in the result is  the
**  same as in <list>.
**
*N  1992/12/15 martin this depends on 'BIPEB' being 32
*/
Bag       FunListBlist (Bag hdCall)
{
    Bag           hdSub;          /* handle of the result            */
    Bag           hdList;         /* handle of the first argument    */
    UInt       len;            /* logical length of the list      */
    Bag           hdBlist;        /* handle of the second argument   */
    UInt       * ptBlist;      /* pointer to blist                */
    UInt       nrb;            /* number of blocks in blist       */
    UInt       m;              /* number of bits in a block       */
    UInt       n;              /* number of bits in blist         */
    UInt       i;              /* loop variable                   */

    /* get and check the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error("usage: ListBlist( <list>, <blist> )",0,0);

    /* get and check the first argument                                    */
    hdList = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IS_LIST( hdList ) )
        return Error("ListBlist: <list> must be a list",0,0);

    /* get and check the second argument                                   */
    hdBlist = EVAL( PTR_BAG(hdCall)[2] );
    if ( ! IsBlist( hdBlist ) )
        return Error("ListBlist: <blist> must be a boolean list",0,0);
    if ( LEN_PLIST( hdList ) != LEN_BLIST( hdBlist ) )
        return Error("ListBlist: <list>, <blist> must have same size",0,0);

    /* handle the case where the two arguments are Identical 
       in this case we cannot have the first arg a plain list and the second
       a Blist simultaneously, so make a temporary copy
       The code is stolen from FunShallowCopy.
       Hopefully this is rare */
    
    if (hdList == hdBlist)
      {
	Bag *ptOld, *ptNew;
        hdList = NewBag( GET_TYPE_BAG(hdBlist), GET_SIZE_BAG(hdBlist) );
        ptOld = PTR_BAG(hdBlist);
        ptNew = PTR_BAG(hdList);
        for ( i = (GET_SIZE_BAG(hdBlist)+SIZE_HD-1)/SIZE_HD; 0 < i; i-- )
            *ptNew++ = *ptOld++;
      }

    PLAIN_LIST( hdList );
    
    /* compute the number of 'true'-s just as in 'FunSizeBlist'            */
    nrb = (LEN_BLIST(hdBlist)+BIPEB-1)/BIPEB;
    ptBlist = (UInt *)(PTR_BAG(hdBlist)+1);
    n = 0;
    for ( i = 1; i <= nrb; i++, ptBlist++ ) {
        m = *ptBlist;
        m = (m & 0x55555555) + ((m >> 1) & 0x55555555);
        m = (m & 0x33333333) + ((m >> 2) & 0x33333333);
        m = (m + (m >>  4)) & 0x0f0f0f0f;
        m = (m + (m >>  8));
        m = (m + (m >> 16)) & 0x000000ff;
        n += m;
    }

    /* make the sublist (we now know its size exactely)                    */
    hdSub = NewBag( GET_TYPE_BAG(hdList), SIZE_PLEN_PLIST( n ) );
    SET_LEN_PLIST( hdSub, n );

    /* loop over the boolean list and stuff elements into <sub>            */
    len = LEN_LIST( hdList );
    n = 1;
    for ( i = 1; i <= len; i++ ) {
        if ( ELM_BLIST( hdBlist, i ) == HdTrue ) {
            SET_ELM_PLIST( hdSub, n, ELMF_LIST( hdList, i ) );
            n++;
        }
    }

    /* return the sublist                                                  */
    return hdSub;
}


/****************************************************************************
**
*F  FunSizeBlist(<hdCall>)  . . .  number of 'true' entries in a boolean list
**
**  'FunSizeBlist' implements the internal function 'SizeBlist'
**
**  'SizeBlist( <blist> )'
**
**  'SizeBlist' returns the  number of entries  of the boolean  list  <blist>
**  that are 'true'.
**
**  The sequence to compute the  number of bits  in a block is quite  clever.
**  The idea is that after the <i>-th instruction each subblock of $2^i$ bits
**  holds the number   of bits of this   subblock in the original block  <m>.
**  This is illustrated in the example below for a block of with 8 bits:
**
**       // a b c d e f g h
**      m = (m & 0x55)       +  ((m >> 1) & 0x55);
**       // . b . d . f . h  +  . a . c . e . g   =  a+b c+d e+f g+h
**      m = (m & 0x33)       +  ((m >> 2) & 0x33);
**       // . . c+d . . g+h  +  . . a+b . . e+f   =  a+b+c+d e+f+g+h
**      m = (m & 0x0f)       +  ((m >> 4) & 0x0f);
**       // . . . . e+f+g+h  +  . . . . a+b+c+d   =  a+b+c+d+e+f+g+h
**
**  In the actual  code  some unnecessary mask  have  been removed, improving
**  performance quite a bit,  because masks are 32  bit immediate values  for
**  which most RISC  processors need two  instructions to load them.  Talking
**  about performance.  The code is  close to optimal,  it should compile  to
**  only about  22 MIPS  or SPARC instructions.   Dividing the  block into  4
**  bytes and looking up the number of bits  of a byte in a  table may be 10%
**  faster, but only if the table lives in the data cache.
*/
Bag       FunSizeBlist (Bag hdCall)
{
    Bag           hdBlist;        /* handle of the argument          */
    UInt       * ptBlist;      /* pointer to blist                */
    UInt       nrb;            /* number of blocks in blist       */
    UInt       m;              /* number of bits in a block       */
    UInt       n;              /* number of bits in blist         */
    UInt       i;              /* loop variable                   */

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
        return Error("usage: SizeBlist( <blist> )",0,0);
    hdBlist = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hdBlist) != T_BLIST && ! IsBlist(hdBlist) )
        return Error("SizeBlist: <blist> must be a boolean list",0,0);

    /* get the number of blocks and a pointer                              */
    nrb = (LEN_BLIST(hdBlist)+BIPEB-1)/BIPEB;
    ptBlist = (UInt *)(PTR_BAG(hdBlist)+1);

    /* loop over the blocks, adding the number of bits of each one         */
    n = 0;
    for ( i = 1; i <= nrb; i++, ptBlist++ ) {
        m = *ptBlist;
        m = (m & 0x55555555) + ((m >> 1) & 0x55555555);
        m = (m & 0x33333333) + ((m >> 2) & 0x33333333);
        m = (m + (m >>  4)) & 0x0f0f0f0f;
        m = (m + (m >>  8));
        m = (m + (m >> 16)) & 0x000000ff;
        n += m;
    }

    /* return the number of bits                                           */
    return INT_TO_HD( n );
}


/****************************************************************************
**
*F  FunIsSubsetBlist(<hdCall>)  . test if a boolean list is subset of another
**
**  'FunIsSubsetBlist' implements the internal function 'IsSubsetBlist'.
**
**  'IsSubsetBlist( <blist1>, <blist2> )'
**
**  'IsSubsetBlist' returns 'true' if  the boolean list <blist2> is  a subset
**  of the boolean list <list1>, which must have equal length.  <blist2> is a
**  subset if <blist1> if '<blist2>[<i>] >= <blist1>[<i>]' for all <i>.
*/
Bag       FunIsSubsetBlist (Bag hdCall)
{
    Bag           hdBlist1;       /* handle of the first argument    */
    Bag           hdBlist2;       /* handle of the second argument   */
    UInt       * ptBlist1;     /* pointer to the first argument   */
    UInt       * ptBlist2;     /* pointer to the second argument  */
    UInt       i;              /* loop variable                   */

    /* get and check the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
      return Error("usage: IsSubsetBlist( <blist1>, <blist2> )",0,0);
    hdBlist1 = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IsBlist( hdBlist1 ) )
      return Error("IsSubsetBlist: <blist1> must be a boolean list",0,0);
    hdBlist2 = EVAL( PTR_BAG(hdCall)[2] );
    if ( ! IsBlist( hdBlist2 ) )
      return Error("IsSubsetBlist: <blist2> must be a boolean list",0,0);
    if ( LEN_BLIST(hdBlist1) != LEN_BLIST(hdBlist2) )
      return Error("IsSubsetBlist: lists must have equal length",0,0);

    /* test for subset property blockwise                                  */
    ptBlist1 = (UInt *)(PTR_BAG(hdBlist1)+1);
    ptBlist2 = (UInt *)(PTR_BAG(hdBlist2)+1);
    for ( i = (LEN_BLIST(hdBlist1)+BIPEB-1)/BIPEB; 0 < i; i-- ) {
        if ( *ptBlist1 != (*ptBlist1 | *ptBlist2) )
            break;
        ptBlist1++;  ptBlist2++;
    }

    /* if no counterexample was found, <blist2> is a subset of <blist1>    */
    return (i == 0) ? HdTrue : HdFalse;
}


/****************************************************************************
**
*F  FunUniteBlist(<hdCall>) . . . . . . . unite one boolean list with another
**
**  'FunUniteBlist' implements the internal function 'UniteBlist'.
**
**  'UniteBlist( <blist1>, <blist2> )'
**
**  'UniteBlist'  unites  the  boolean list  <blist1>  with  the boolean list
**  <blist2>,  which  must  have the   same  length.  This  is  equivalent to
**  assigning '<blist1>[<i>] := <blist1>[<i>] or <blist2>[<i>]' for all <i>.
*/
Bag       FunUniteBlist (Bag hdCall)
{
    Bag           hdBlist1;       /* handle of the first argument    */
    Bag           hdBlist2;       /* handle of the second argument   */
    UInt       * ptBlist1;     /* pointer to the first argument   */
    UInt       * ptBlist2;     /* pointer to the second argument  */
    UInt       i;              /* loop variable                   */

    /* get and check the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error("usage: UniteBlist( <blist1>, <blist2> )",0,0);
    hdBlist1 = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IsBlist( hdBlist1 ) )
        return Error("UniteBlist: <blist1> must be a boolean list",0,0);
    hdBlist2 = EVAL( PTR_BAG(hdCall)[2] );
    if ( ! IsBlist( hdBlist2 ) )
        return Error("UniteBlist: <blist2> must be a boolean list",0,0);
    if ( LEN_BLIST(hdBlist1) != LEN_BLIST(hdBlist2) )
        return Error("UniteBlist: lists must have equal length",0,0);

    /* compute the union by *or*-ing blockwise                             */
    ptBlist1 = (UInt *)(PTR_BAG(hdBlist1)+1);
    ptBlist2 = (UInt *)(PTR_BAG(hdBlist2)+1);
    for ( i = (LEN_BLIST(hdBlist1)+BIPEB-1)/BIPEB; 0 < i; i-- )
        *ptBlist1++ |= *ptBlist2++;

    /* return nothing, this function is a procedure                        */
    return HdVoid;
}


/****************************************************************************
**
*F  FunIntersectBlist(<hdCall>) . . . intersect one boolean list with another
**
**  'FunIntersectBlist' implements the function 'IntersectBlist'.
**
**  'IntersectBlist( <blist1>, <blist2> )'
**
**  'IntersectBlist' intersects the boolean list   <blist1> with the  boolean
**  list <blist2>, which must  have the same  length.  This is equivalent  to
**  assigning '<blist1>[<i>] := <blist1>[<i>] and <blist2>[<i>]' for all <i>.
*/
Bag       FunIntersectBlist (Bag hdCall)
{
    Bag           hdBlist1;       /* handle of the first argument    */
    Bag           hdBlist2;       /* handle of the second argument   */
    UInt       * ptBlist1;     /* pointer to the first argument   */
    UInt       * ptBlist2;     /* pointer to the second argument  */
    UInt       i;              /* loop variable                   */

    /* get and check the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error("usage: IntersectBlist( <blist1>, <blist2> )",0,0);
    hdBlist1 = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IsBlist( hdBlist1 ) )
       return Error("IntersectBlist: <blist1> must be a boolean list",0,0);
    hdBlist2 = EVAL( PTR_BAG(hdCall)[2] );
    if ( ! IsBlist( hdBlist2 ) )
       return Error("IntersectBlist: <blist2> must be a boolean list",0,0);
    if ( LEN_BLIST(hdBlist1) != LEN_BLIST(hdBlist2) )
        return Error("IntersectBlist: lists must have equal length",0,0);

    /* compute the intersection by *and*-ing blockwise                     */
    ptBlist1 = (UInt *)(PTR_BAG(hdBlist1)+1);
    ptBlist2 = (UInt *)(PTR_BAG(hdBlist2)+1);
    for ( i = (LEN_BLIST(hdBlist1)+BIPEB-1)/BIPEB; 0 < i; i-- )
        *ptBlist1++ &= *ptBlist2++;

    /* return nothing, this function is a procedure                        */
    return HdVoid;
}


/****************************************************************************
**
*F  FunSubtractBlist(<hdCall>)  . . .  subtract one boolean list from another
**
**  'FunSubtractBlist' implements the internal function 'SubtractBlist'.
**
**  'SubtractBlist( <blist1>, <blist2> )'
**
**  'SubtractBlist' subtracts the boolean list <blist2> from the boolean list
**  <blist1>, which must have the same  length.  This is equivalent assigning
**  '<blist1>[<i>] := <blist1>[<i>] and not <blist2>[<i>]' for all <i>.
*/
Bag       FunSubtractBlist (Bag hdCall)
{
    Bag           hdBlist1;       /* handle of the first argument    */
    Bag           hdBlist2;       /* handle of the second argument   */
    UInt       * ptBlist1;     /* pointer to the first argument   */
    UInt       * ptBlist2;     /* pointer to the second argument  */
    UInt       i;              /* loop variable                   */

    /* get and check the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error("usage: SubtractBlist( <blist1>, <blist2> )",0,0);
    hdBlist1 = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IsBlist( hdBlist1 ) )
        return Error("SubtractBlist: <blist1> must be a boolean list",0,0);
    hdBlist2 = EVAL( PTR_BAG(hdCall)[2] );
    if ( ! IsBlist( hdBlist2 ) )
        return Error("SubtractBlist: <blist2> must be a boolean list",0,0);
    if ( LEN_BLIST(hdBlist1) != LEN_BLIST(hdBlist2) )
        return Error("SubtractBlist: lists must have equal length",0,0);

    /* compute the difference by operating blockwise                       */
    ptBlist1 = (UInt *)(PTR_BAG(hdBlist1)+1);
    ptBlist2 = (UInt *)(PTR_BAG(hdBlist2)+1);
    for ( i = (LEN_BLIST(hdBlist1)+BIPEB-1)/BIPEB; 0 < i; i-- )
        *ptBlist1++ &= ~ *ptBlist2++;

    /* return nothing, this function is a procedure                        */
    return HdVoid;
}


/****************************************************************************
**
*F  FunDistanceBlist(<hdCall>)  . . . . . . . . distance of two boolean lists
**
**  'FunDistanceBlist' implements the internal function 'DistanceBlist'.
**
**  'DistanceBlist( <blist1>, <blist2> )'
**
**  'DistanceBlist' computes the distance of two boolean list.  The  distance
**  is the number of position in which the two boolean list differ.
*/
Bag       FunDistanceBlist (Bag hdCall)
{
    Bag           hdBlist1;       /* handle of the first argument    */
    Bag           hdBlist2;       /* handle of the second argument   */
    UInt       * ptBlist1;     /* pointer to the first argument   */
    UInt       * ptBlist2;     /* pointer to the second argument  */
    UInt       m;              /* number of bits in a block       */
    UInt       n;              /* number of bits in blist         */
    UInt       i;              /* loop variable                   */

    /* get and check the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error("usage: DistanceBlist( <blist1>, <blist2> )",0,0);
    hdBlist1 = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IsBlist( hdBlist1 ) )
        return Error("DistanceBlist: <blist1> must be a boolean list",0,0);
    hdBlist2 = EVAL( PTR_BAG(hdCall)[2] );
    if ( ! IsBlist( hdBlist2 ) )
        return Error("DistanceBlist: <blist2> must be a boolean list",0,0);
    if ( LEN_BLIST(hdBlist1) != LEN_BLIST(hdBlist2) )
        return Error("DistanceBlist: lists must have equal length",0,0);

    /* compute the distance by operating blockwise                         */
    ptBlist1 = (UInt *)(PTR_BAG(hdBlist1)+1);
    ptBlist2 = (UInt *)(PTR_BAG(hdBlist2)+1);
    n = 0;
    for ( i = (LEN_BLIST(hdBlist1)+BIPEB-1)/BIPEB; 0 < i; i-- ) {
        m = (*ptBlist1++) ^ (*ptBlist2++);
        m = (m & 0x55555555) + ((m >> 1) & 0x55555555);
        m = (m & 0x33333333) + ((m >> 2) & 0x33333333);
        m = (m + (m >>  4)) & 0x0f0f0f0f;
        m = (m + (m >>  8));
        m = (m + (m >> 16)) & 0x000000ff;
        n += m;
    }
    return INT_TO_HD(n);
}


/****************************************************************************
**
*F  InitBlist() . . . . . . . . . . . . . initialize the boolean list package
**
**  'InitBlist' initializes the boolean list package.
*/
void            InitBlist (void)
{

    /* install the list functions in the tables                            */
    TabIsList[T_BLIST]      = 1;
    TabLenList[T_BLIST]     = LenBlist;
    TabElmList[T_BLIST]     = ElmBlist;
    TabElmfList[T_BLIST]    = ElmfBlist;
    TabElmlList[T_BLIST]    = ElmfBlist;
    TabElmrList[T_BLIST]    = ElmfBlist;
    TabElmsList[T_BLIST]    = ElmsBlist;
    TabAssList[T_BLIST]     = AssBlist;
    TabAsssList[T_BLIST]    = AsssBlist;
    TabPosList[T_BLIST]     = PosBlist;
    TabPlainList[T_BLIST]   = PlainBlist;
    TabIsDenseList[T_BLIST] = IsDenseBlist;
    TabIsPossList[T_BLIST]  = IsPossBlist;
    EvTab[T_BLIST]          = EvList;
    PrTab[T_BLIST]          = PrList;
    TabEq[T_BLIST][T_BLIST] = EqBlist;
    TabLt[T_BLIST][T_BLIST] = LtList;

    /* install the internal functions                                      */
    InstIntFunc( "IsBlist",        FunIsBlist        );
    InstIntFunc( "BlistList",      FunBlistList      );
    InstIntFunc( "ListBlist",      FunListBlist      );
    InstIntFunc( "SizeBlist",      FunSizeBlist      );
    InstIntFunc( "IsSubsetBlist",  FunIsSubsetBlist  );
    InstIntFunc( "IntersectBlist", FunIntersectBlist );
    InstIntFunc( "UniteBlist",     FunUniteBlist     );
    InstIntFunc( "SubtractBlist",  FunSubtractBlist  );
    InstIntFunc( "DistanceBlist",  FunDistanceBlist  );

}
