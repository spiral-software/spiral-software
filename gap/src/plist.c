/****************************************************************************
**
*A  plist.c                     GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This  file contains  the  functions  that  deal  with  plain lists.   The
**  interface  between the various list  packages and the rest of GAP  are in
**  "lists.c".
**
**  A plain list  is represented  by  a bag of type  'T_LIST', which  has the
**  following format:
**
**      +-------+-------+-------+- - - -+-------+-------+- - - -+-------+
**      |logical| first |second |       |last   |handle |       |handle |
**      |length |element|element|       |element|   0   |       |   0   |
**      +-------+-------+-------+- - - -+-------+-------+- - - -+-------+
**
**  The first  handle  is  the logical   length  of  the list stored   as GAP
**  immediate integer.  The second handle is the  handle of the first element
**  of the list.  The third handle is the handle of the second element of the
**  list, and so on.  If the  physical  length of a list  is greater than the
**  logical, there will be unused entries at the  end of the list, comtaining
**  handle 0.  The physical length might be greater than the logical, because
**  the physical size  of a list is  increased by at least   12.5\%, to avoid
**  doing this too often.
**
**  All the  other packages of GAP may rely  on this representation  of plain
**  lists.  Nevertheless  the  probably ought to use  the macros 'LEN_PLIST',
**  'SET_LEN_PLIST', 'ELM_PLIST', and 'SET_ELM_PLIST'.
**
**  This package consists of three parts.
**
**  The    first   part    consists   of    the   macros   'SIZE_PLEN_PLIST',
**  'PLEN_SIZE_PLIST',   'LEN_PLIST',   'SET_LEN_PLIST',   'ELM_PLIST',   and
**  'SET_ELM_PLIST'.   They  determine  the representation  of  plain  lists.
**  Everything else  in this file in the rest of the {\GAP} kernel uses those
**  macros to access and modify plain lists.
**
**  The  second  part  consists  of  the  functions  'LenPlist',  'ElmPlist',
**  'ElmsPlist',   'AssPlist',    'AsssPlist',    'PosPlist',   'PlainPlist',
**  'IsDensePlist', 'IsPossPlist',  'EqPlist', and 'LtPlist'.   They are  the
**  functions required by the generic lists  package.   Using these functions
**  the other  parts of the {\GAP} kernel can  access  and modify plain lists
**  without actually being aware that they are dealing with a plain list.
**
**  The third part consists of the functions 'MakeList', 'EvMakeList'.  These
**  function make it possible to make plain  lists by evaluating a plain list
**  literal.
**
*/

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "scanner.h"             /* reading of tokens and printing  */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */

#include        "list.h"                /* generic list package            */
#include        "range.h"               /* 'LEN_RANGE', 'LOW_RANGE', ..    */
#include        "record.h"              /* 'HdTilde', 'MakeRec'            */

#include        "plist.h"               /* declaration part of the package */


/****************************************************************************
**
*F  PLEN_SIZE_PLIST(<size>) . . .  physical length from size for a plain list
**
**  'PLEN_SIZE_PLIST'  computes  the  physical  length  (e.g.  the  number of
**  elements that could be stored  in a list) from the <size> (as reported by
**  'GET_SIZE_BAG') for a plain list.
**
**  Note that 'PLEN_SIZE_PLIST' is a macro, so  do not call it with arguments
**  that have sideeffects.
**
**  'PLEN_SIZE_PLIST'  is defined in the declaration  part of this package as
**  follows:
**
#define PLEN_SIZE_PLIST(GET_SIZE_BAG)           (((GET_SIZE_BAG) - SIZE_HD) / SIZE_HD)
*/


/****************************************************************************
**
*F  SIZE_PLEN_PLIST(<plen>)  size for a plain list with given physical length
**
**  'SIZE_PLEN_PLIST' returns the size that a plain list with room for <plen>
**  elements must at least have.
**
**  Note that 'SIZE_PLEN_PLIST' is a macro, so do not call it with  arguments
**  that have sideeffects.
**
**  'SIZE_PLEN_PLIST' is  defined  in the declaration part of this package as
**  follows:
**
#define SIZE_PLEN_PLIST(PLEN)           (SIZE_HD + (PLEN) * SIZE_HD)
*/


/****************************************************************************
**
*F  LEN_PLIST(<hdList>) . . . . . . . . . . . . . . .  length of a plain list
**
**  'LEN_PLIST' returns   the logical length  of   the list  <hdList> as  a C
**  integer.   The length is stored  as GAP immediate  integer as the zeroeth
**  handle.
**
**  Note that 'LEN_PLIST' is a  macro, so do  not call it with arguments that
**  have sideeffects.
**
**  'LEN_PLIST'  is  defined  in  the declaration  part  of  this  package as
**  follows:
**
#define LEN_PLIST(LIST)                 (HD_TO_INT(PTR_BAG(LIST)[0]))
*/


/****************************************************************************
**
*F  SET_LEN_PLIST(<hdList>,<len>) . . . . . .  set the length of a plain list
**
**  'SET_LEN_PLIST' sets the length of the plain list <hdList> to <len>.  The
**  length is stored as GAP immediate integer as the zeroeth handle.
**
**  Note  that 'SET_LEN_PLIST'  is a macro, so do not call it with  arguments
**  that have sideeffects.
**
**  'SET_LEN_PLIST'  is defined in the declaration  part of  this package  as
**  follows:
**
#define SET_LEN_PLIST(LIST,LEN)         (PTR_BAG(LIST)[0] = INT_TO_HD(LEN))
*/


/****************************************************************************
**
*F  ELM_PLIST(<hdList>,<pos>) . . . . . . . . . . . . element of a plain list
**
**  'ELM_PLIST' return the <pos>-th element of the list <hdList>.  <pos> must
**  be a positive integer less than or equal to the length of <hdList>.
**
**  Note that  'ELM_PLIST' is a macro, so do  not call it with arguments that
**  have sideeffects.
**
**  'ELM_PLIST'  is  defined  in  the  declaration part  of  this  package as
**  follows:
**
#define ELM_PLIST(LIST,POS)             (PTR_BAG(LIST)[POS])
*/


/****************************************************************************
**
*F  SET_ELM_PLIST(<hdList>,<pos>,<hdVal>) . assign an element to a plain list
**
**  'SET_ELM_PLIST'  assigns the value <hdVal> to the plain list <hdList>  at
**  the position <pos>.  <pos> must be a positive integer  less than or equal
**  to the length of <hdList>.
**
**  Note that 'SET_ELM_PLIST' is a  macro, so do not  call it  with arguments
**  that have sideeffects.
**
**  'SET_ELM_PLIST' is defined  in the  declaration part  of this  package as
**  follows:
**
#define SET_ELM_PLIST(LIST,POS,VAL)     (PTR_BAG(LIST)[POS] = (VAL))
*/


/****************************************************************************
**
*F  LenPlist(<hdList>)  . . . . . . . . . . . . . . .  length of a plain list
**
**  'LenPlist' returns the length of the plain list <hdList> as a C integer.
**
**  'LenPlist' is the function in 'TabLenList' for plain lists.
*/
Int            LenPlist (Bag hdList)
{
    return LEN_PLIST( hdList );
}


/****************************************************************************
**
*F  ElmPlist(<hdList>,<pos>)  . . . . . . . select an element of a plain list
**
**  'ElmPlist' selects  the  element at  position  <pos> of  the  plain  list
**  <hdList>.  It is the responsibility of the caller to ensure that <pos> is
**  a positive integer.   An error is signalled if <pos> is  larger than  the
**  length of <hdList> or if <hdList> has no  assigned value at the  position
**  <pos>.
**
**  'ElmfPlist' does the  same thing than 'ElmList', but need not check  that
**  <pos>  is  less  than or  equal to the  length  of <hdList>, this  is the
**  responsibility  of the  caller.   Also  it  returns 0 if  <hdList> has no
**  assigned value at the position <pos>.
**
**  'ElmPlist' is the function in 'TabElmList'  for plain lists.  'ElmfPlist'
**  is the function  in  'TabElmfList', 'TabElmlList', and  'TabElmrList' for
**  plain lists.
*/
Bag       ElmPlist (Bag hdList, Int pos)
{
    Bag           hdElm;          /* the selected element, result    */

    /* check the position                                                  */
    if ( LEN_PLIST( hdList ) < pos ) {
        return Error(
          "List Element: <list>[%d] must have a value",
                     pos, 0 );
    }

    /* select and check the element                                        */
    hdElm = ELM_PLIST( hdList, pos );
    if ( hdElm == 0 ) {
        return Error(
          "List Element: <list>[%d] must have a value",
                     pos, 0 );
    }

    /* return the element                                                  */
    return hdElm;
}

Bag       ElmfPlist (Bag hdList, Int pos)
{
    /* select and return the element                                       */
    return ELM_PLIST( hdList, pos );
}


/****************************************************************************
**
*F  ElmsPlist(<hdList>,<hdPoss>)  . . . .  select a sublist from a plain list
**
**  'ElmsPlist'  returns a new list containing the elements at  the  position
**  given  in the list  <hdPoss> from  the  plain  list <hdList>.   It is the
**  responsibility  of  the  caller  to  ensure  that <hdPoss>  is  dense and
**  contains only positive integers.   An error is signalled if <hdList>  has
**  no assigned value at any of the positions in  <hdPoss>, or  if an element
**  of <hdPoss> is larger than the length of <hdList>.
**
**  'ElmsPlist' is the function in 'TabElmsList' for plain lists.
*/
Bag       ElmsPlist (Bag hdList, Bag hdPoss)
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
        hdElms = NewBag( T_LIST, SIZE_PLEN_PLIST( lenPoss ) );
        SET_LEN_PLIST( hdElms, lenPoss );

        /* loop over the entries of <positions> and select                 */
        for ( i = 1; i <= lenPoss; i++ ) {

            /* get <position>                                              */
            pos = HD_TO_INT( ELMF_LIST( hdPoss, i ) );
            if ( lenList < pos ) {
                return Error(
                  "List Elements: <list>[%d] must have a value",
                             pos, 0 );
            }

            /* select the element                                          */
            hdElm = ELM_PLIST( hdList, pos );
            if ( hdElm == 0 ) {
                return Error(
                  "List Elements: <list>[%d] must have a value",
                             pos, 0 );
            }

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
                         pos, 0 );
        }
        if ( lenList < pos + (lenPoss-1) * inc ) {
            return Error(
              "List Elements: <list>[%d] must have a value",
                         pos + (lenPoss-1) * inc, 0 );
        }

        /* make the result list                                            */
        hdElms = NewBag( T_LIST, SIZE_PLEN_PLIST( lenPoss ) );
        SET_LEN_PLIST( hdElms, lenPoss );

        /* loop over the entries of <positions> and select                 */
        for ( i = 1; i <= lenPoss; i++, pos += inc ) {

            /* select the element                                          */
            hdElm = ELM_PLIST( hdList, pos );
            if ( hdElm == 0 ) {
                return Error(
                  "List Elements: <list>[%d] must have a value",
                             pos, 0 );
            }

            /* assign the element to <elms>                                */
            SET_ELM_PLIST( hdElms, i, hdElm );

        }

    }

    /* return the result                                                   */
    return hdElms;
}


/****************************************************************************
**
*F  AssPlist(<hdList>,<pos>,<hdVal>)  . . . . . . . .  assign to a plain list
**
**  'AssPlist' assigns  the value <hdVal> to the plain list  <hdList> at  the
**  position <pos>.  It is the  responsibility of the caller to  ensure  that
**  <pos> is positive, and that <hdVal> is not 'HdVoid'.
**
**  If the position is larger then the length of the list <list>, the list is
**  automatically  extended.  To avoid  making this too often, the bag of the
**  list is extended by at least '<length>/8 + 4' handles.  Thus in the loop
**
**      l := [];  for i in [1..1024]  do l[i] := i^2;  od;
**
**  the list 'l' is extended only 32 times not 1024 times.
**
**  'AssPlist' is the function in 'TabAssList' for plain lists.
*/
Bag       AssPlist (Bag hdList, Int pos, Bag hdVal)
{
    Int                plen;           /* physical length of <list>       */

    /* resize the list if necessary                                        */
    if ( LEN_PLIST( hdList ) < pos ) {
        plen = PLEN_SIZE_PLIST( GET_SIZE_BAG( hdList ) );
        if ( plen + plen/8 + 4 < pos )
            Resize( hdList, SIZE_PLEN_PLIST( pos ) );
        else if ( plen < pos )
            Resize( hdList, SIZE_PLEN_PLIST( plen + plen/8 + 4 ) );
        SET_LEN_PLIST( hdList, pos );
    }

    /* now perform the assignment and return the assigned value            */
    SET_ELM_PLIST( hdList, pos, hdVal );
    return hdVal;
}


/****************************************************************************
**
*F  AsssPlist(<hdList>,<hdPoss>,<hdVals>) . assign several elements to a list
**
**  'AsssPlist' assignes  the  values from the list <hdVals> at the positions
**  given  in  the  list   <hdPoss>   to  the   list  <hdList>.   It  is  the
**  responsibility of  the  caller  to ensure  that  <hdPoss>  is  dense  and
**  contains only positive integers, that <hdPoss> and <hdVals> have the same
**  length, and that <hdVals> is dense.
**
**  'AsssPlist' is the function in 'TabAsssList' for plain lists.
*/
Bag       AsssPlist (Bag hdList, Bag hdPoss, Bag hdVals)
{
    Int                lenPoss;        /* length of <positions>           */
    Int                pos;            /* <position> as integer           */
    Int                max;            /* larger position                 */
    Int                inc;            /* increment in a range            */
    Bag           hdVal;          /* one element from <vals>         */
    Int                plen;           /* physical size of <list>         */
    Int                i;              /* loop variable                   */

    /* general code                                                        */
    if ( GET_TYPE_BAG(hdPoss) != T_RANGE ) {

        /* get the length of <positions>                                   */
        lenPoss = LEN_LIST( hdPoss );

        /* find the largest entry in <positions>                           */
        max = LEN_PLIST( hdList );
        for ( i = 1; i <= lenPoss; i++ ) {
            if ( max < HD_TO_INT( ELMF_LIST( hdPoss, i ) ) )
                max = HD_TO_INT( ELMF_LIST( hdPoss, i ) );
        }

        /* resize the list if necessary                                    */
        if ( LEN_PLIST(hdList) < max ) {
            plen = PLEN_SIZE_PLIST( GET_SIZE_BAG(hdList) );
            if ( plen + plen/8 + 4 < max )
                Resize( hdList, SIZE_PLEN_PLIST( max ) );
            else if ( plen < max )
                Resize( hdList, SIZE_PLEN_PLIST( plen + plen/8 + 4 ) );
            SET_LEN_PLIST( hdList, max );
        }

        /* loop over the entries of <positions> and select                 */
        for ( i = 1; i <= lenPoss; i++ ) {

            /* get <position>                                              */
            pos = HD_TO_INT( ELMF_LIST( hdPoss, i ) );

            /* select the element                                          */
            hdVal = ELMF_LIST( hdVals, i );

            /* assign the element into <elms>                              */
            SET_ELM_PLIST( hdList, pos, hdVal );

        }

    }

    /* special code for ranges                                             */
    else {

        /* get the length of <positions>                                   */
        lenPoss = LEN_RANGE( hdPoss );
        pos = LOW_RANGE( hdPoss );
        inc = INC_RANGE( hdPoss );

        /* find the largest entry in <positions>                           */
        max = LEN_PLIST( hdList );
        if ( max < pos )
            max = pos;
        if ( max < pos + (lenPoss-1) * inc )
            max = pos + (lenPoss-1) * inc;

        /* resize the list if necessary                                    */
        if ( LEN_PLIST(hdList) < max ) {
            plen = PLEN_SIZE_PLIST( GET_SIZE_BAG(hdList) );
            if ( plen + plen/8 + 4 < max )
                Resize( hdList, SIZE_PLEN_PLIST( max ) );
            else if ( plen < max )
                Resize( hdList, SIZE_PLEN_PLIST( plen + plen/8 + 4 ) );
            SET_LEN_PLIST( hdList, max );
        }

        /* loop over the entries of <positions> and select                 */
        for ( i = 1; i <= lenPoss; i++, pos += inc ) {

            /* select the element                                          */
            hdVal = ELMF_LIST( hdVals, i );

            /* assign the element to <elms>                                */
            SET_ELM_PLIST( hdList, pos, hdVal );

        }

    }

    /* return the result                                                   */
    return hdVals;
}


/****************************************************************************
**
*F  PosPlist(<hdList>,<hdVal>,<start>)  . .  position of an element in a list
**
**  'PosPlist'  returns the position of the value <hdVal> in  the plain  list
**  <hdList> after the first  position <start> as a C integer.  0 is returned
**  if <hdVal> is not in the list.
**
**  'PosPlist' is the function in 'TabPosList' for plain lists.
*/
Int            PosPlist (Bag hdList, Bag hdVal, Int start)
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
        if ( hdElm != 0 && (hdElm == hdVal || EQ( hdElm, hdVal ) == HdTrue) )
            break;

    }

    /* return the position (0 if <val> was not found)                      */
    return (lenList < i ? 0 : i);
}


/****************************************************************************
**
*F  PlainPlist(<hdList>)  . . . . . . .  convert a plain list to a plain list
**
**  'PlainPlist' converts the plain list <hdList> to a plain list.  Not  much
**  work.
**
**  'PlainPlist' is the function in 'TabPlainList' for plain lists.
*/
void            PlainPlist (Bag hdList)
{
    return;
}


/****************************************************************************
**
*F  IsDensePlist(<hdList>)  . . . .  dense list test function for plain lists
**
**  'IsDensePlist' returns 1 if the plain list <hdList> is a dense list and 0
**  otherwise.
**
**  'IsDensePlist' is the function in 'TabIsDenseList' for plain lists.
*/
Int            IsDensePlist (Bag hdList)
{
    Int                lenList;        /* length of <list>                */
    Int                i;              /* loop variable                   */

    /* get the length of the list                                          */
    lenList = LEN_PLIST( hdList );

    /* loop over the entries of the list                                   */
    for ( i = 1; i <= lenList; i++ ) {
        if ( ELM_PLIST( hdList, i ) == 0 )
            return 0;
    }

    /* no hole found                                                       */
    return 1;
}


/****************************************************************************
**
*F  IsPossPlist(<hdList>) . . .  positions list test function for plain lists
**
**  'IsPossPlist' returns  1 if  the plain  list  <hdList>  is  a  dense list
**  containing only positive integers, and 0 otherwise.
**
**  'IsPossPlist' is the function in 'TabIsPossList' for plain lists.
*/
Int            IsPossPlist (Bag hdList)
{
    Int                lenList;        /* length of <list>                */
    Bag           hdElm;          /* one element of <list>           */
    Int                i;              /* loop variable                   */

    /* get the length of the variable                                      */
    lenList = LEN_PLIST( hdList );

    /* loop over the entries of the list                                   */
    for ( i = 1; i <= lenList; i++ ) {
        hdElm = ELM_PLIST( hdList, i );
        if ( hdElm == 0 || GET_TYPE_BAG(hdElm) != T_INT || HD_TO_INT(hdElm) <= 0 )
            return 0;
    }

    /* no problems found                                                   */
    return 1;
}


/****************************************************************************
**
*F  EqPlist(<hdL>,<hdR>) . . . . . . . . .  test if two plain lists are equal
**
**  'EqList' returns 'true' if the two plain lists <hdL> and <hdR> are  equal
**  and 'false' otherwise.
**
**  Is called from the 'EQ' binop so both  operands  are  already  evaluated.
*/
Bag       EqPlist (Bag hdL, Bag hdR)
{
    Int                lenL;           /* length of the left operand      */
    Int                lenR;           /* length of the right operand     */
    Bag           hdElmL;         /* element of the left operand     */
    Bag           hdElmR;         /* element of the right operand    */
    Int                i;              /* loop variable                   */

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
        if ( hdElmL == 0 && hdElmR != 0 ) {
            return HdFalse;
        }
        else if ( hdElmR == 0 && hdElmL != 0 ) {
            return HdFalse;
        }
        else if ( hdElmL != hdElmR && EQ( hdElmL, hdElmR ) == HdFalse ) {
            return HdFalse;
        }
    }

    /* no differences found, the lists are equal                           */
    return HdTrue;
}


/****************************************************************************
**
*F  LtPlist(<hdL>,<hdR>)  . . . . . . . . . test if two plain lists are equal
**
**  'LtList' returns 'true' if  the  plain list <hdL>  is less than the plain
**  list <hdR> and 'false' otherwise.
**
**  Is called from the 'LT' binop so both operands are already evaluated.
*/
Bag       LtPlist (Bag hdL, Bag hdR)
{
    Int                lenL;           /* length of the left operand      */
    Int                lenR;           /* length of the right operand     */
    Bag           hdElmL;         /* element of the left operand     */
    Bag           hdElmR;         /* element of the right operand    */
    Int                i;              /* loop variable                   */

    /* get the lengths of the lists and compare them                       */
    lenL = LEN_PLIST( hdL );
    lenR = LEN_PLIST( hdR );

    /* loop over the elements and compare them                             */
    for ( i = 1; i <= lenL && i <= lenR; i++ ) {
        hdElmL = ELM_PLIST( hdL, i );
        hdElmR = ELM_PLIST( hdR, i );
        if ( hdElmL == 0 && hdElmR != 0 ) {
            return HdTrue;
        }
        else if ( hdElmR == 0 && hdElmL != 0 ) {
            return HdFalse;
        }
        else if ( hdElmL != hdElmR && EQ( hdElmL, hdElmR ) == HdFalse ) {
            return LT( hdElmL, hdElmR );
        }
    }

    /* reached the end of at least one list                                */
    return (lenL < lenR) ? HdTrue : HdFalse;
}


/****************************************************************************
**
*F  EvMakeList( <hdLiteral> )  . . . evaluate list literal to a list constant
**
**  'EvMakeList'   evaluates  the  literal, i.e.,   not  yet evaluated,  list
**  <hdLiteral> to a constant list.
**
**  'EvMakeList' just calls 'MakeList' telling it that  the result  goes into
**  the variable '~'.  Thus expressions  in the variable   list can refer  to
**  this variable and its subobjects to create objects that are not trees.
*/
Bag       EvMakeList (Bag hdLiteral)
{
    Bag           hdList;         /* handle of the result            */

    /* top level literal, make the list into '~'                           */
    if ( PTR_BAG(HdTilde)[0] == 0 ) {
        hdList = MakeList( HdTilde, 0, hdLiteral );
        SET_BAG(HdTilde, 0,  0 );
    }

    /* not top level, do not write the result somewhere                    */
    else {
        hdList = MakeList( 0, 0, hdLiteral );
    }

    /* return the result                                                   */
    return hdList;
}


/****************************************************************************
**
*F  MakeList(<hdDst>,<ind>,<hdLiteral>) . evaluate list literal to a constant
**
**  'MakeRec' evaluates the  list  literal <hdLiteral>  to a constant one and
**  puts  the result  into the bag  <hdDst>  at position <ind>.    <hdDst> is
**  either the variable '~', a list, or a record.
**
**  Because of literals like 'rec( a := rec( b := 1, c  := ~.a.b ) )' we must
**  enter the handle  of the result  into the superobject  before we begin to
**  evaluate the list literal.
**
**  A list literal  is  very much like a   list, except that  instead of  the
**  elements  it contains  the     expressions, which  yield  the   elements.
**  Evaluating  a  list   literal  thus  means  looping over the  components,
**  evaluating the expressions.
*/
Bag       MakeList (Bag hdDst, Int ind, Bag hdLiteral)
{
    Bag           hdList;         /* handle of the result            */
    Int                len;            /* logical length of the list      */
    Bag           hd;             /* temporary handle                */
    Int                i;              /* loop variable                   */

    /* get length from the makelist, thanks 'RdList' for putting it there  */
    len = LEN_PLIST( hdLiteral );

    /* make a list of the right size                                       */
    hdList = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
    SET_LEN_PLIST( hdList, len );
    if ( hdDst != 0 )  SET_BAG(hdDst, ind,  hdList );

    /* evaluate all entries and put them in the list                       */
    for ( i = 1; i <= len; i++ ) {
        if ( ELM_PLIST( hdLiteral, i ) != 0 ) {
            if ( GET_TYPE_BAG( ELM_PLIST( hdLiteral, i ) ) == T_MAKELIST ) {
                MakeList( hdList, i, ELM_PLIST( hdLiteral, i ) );
            }
            else if ( GET_TYPE_BAG( PTR_BAG(hdLiteral)[i] ) == T_MAKEREC ) {
                MakeRec( hdList, i, ELM_PLIST( hdLiteral, i ) );
            }
            else {
                hd = EVAL( ELM_PLIST( hdLiteral, i ) );
                while ( hd == HdVoid ) {
                    hd = Error(
                      "List: function must return a value",
                               0,0);
                }
                SET_ELM_PLIST( hdList, i, hd );
            }
        }
    }

    /* return the list                                                     */
    return hdList;
}


/****************************************************************************
**
*F  PrMakeList(<hdMake>)  . . . . . . . . . . . . . . .  print a list literal
**
**  'PrMakeList' prints the list literal <hdMake>.
*/
void            PrMakeList (Bag hdMake)
{
    Int                lenList;        /* logical length of <list>        */
    Bag           hdElm;          /* one element from <list>         */
    Int                i;              /* loop variable                   */

    /* get the logical length of the list                                  */
    lenList = LEN_PLIST( hdMake );

    /* loop over the entries                                               */
    Pr("%2>[ %2>",0,0);
    for ( i = 1;  i <= lenList;  i++ ) {
        hdElm = ELM_PLIST( hdMake, i );
        if ( hdElm != 0 ) {
            if ( 1 < i )  Pr("%<,%< %2>",0,0);
            Print( hdElm );
        }
        else {
            if ( 1 < i )  Pr("%2<,%2>",0,0);
        }
    }
    Pr(" %4<]",0,0);
}


/****************************************************************************
**
*F  InitPlist() . . . . . . . . . . . . . . . . . initialize the list package
**
**  Is called during  the  initialization  to  initialize  the  list package.
*/
void            InitPlist (void)
{

    /* install the list functions in the tables                            */
    TabIsList[T_LIST]       = 1;
    TabLenList[T_LIST]      = LenPlist;
    TabElmList[T_LIST]      = ElmPlist;
    TabElmfList[T_LIST]     = ElmfPlist;
    TabElmlList[T_LIST]     = ElmfPlist;
    TabElmrList[T_LIST]     = ElmfPlist;
    TabElmsList[T_LIST]     = ElmsPlist;
    TabAssList[T_LIST]      = AssPlist;
    TabAsssList[T_LIST]     = AsssPlist;
    TabPosList[T_LIST]      = PosPlist;
    TabPlainList[T_LIST]    = PlainPlist;
    TabIsDenseList[T_LIST]  = IsDensePlist;
    TabIsPossList[T_LIST]   = IsPossPlist;
    EvTab[T_LIST]           = EvList;
    PrTab[T_LIST]           = PrList;
    TabEq[T_LIST][T_LIST]   = EqPlist;
    TabLt[T_LIST][T_LIST]   = LtPlist;

    /* functions for list literals                                         */
    EvTab[T_MAKELIST]       = EvMakeList;
    PrTab[T_MAKELIST]       = PrMakeList;

}


/****************************************************************************
**
*E  Emacs . . . . . . . . . . . . . . . . . . . . . . . local emacs variables
**
**  Local Variables:
**  outline-regexp:     "*A\\|*F\\|*V\\|*T\\|*E"
**  fill-column:        73
**  fill-prefix:        "**  "
**  End:
*/



