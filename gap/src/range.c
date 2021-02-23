/****************************************************************************
**
*A  range.c                     GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file contains the functions that mainly deal with ranges.  As ranges
**  are a special case of lists many things are done in the list package.
**
**  A *range* is  a list without  holes  consisting  of consecutive integers.
**  For the full definition of ranges see chapter "Ranges" in the GAP Manual.
**  Read  also   "More about Ranges"  about  the different  representation of
**  ranges.
**
**  A list that is  known to be  a  range is  represented  by a  bag of  type
**  'T_RANGE', which has the following format:
**
**      +-------+-------+-------+
**      |logical| first | incr- |
**      | length|element| ement |
**      +-------+-------+-------+
**
**  The first entry is the handle of the logical length.  The second entry is
**  the first element of the range.  The last  entry  is the  increment.  All
**  three are represented as immediate GAP integers.
**
**  The element at position <pos> is thus simply <first> + (<pos>-1) * <inc>.
**
**  Note  that  a list  represented by a   bag of type   'T_LIST', 'T_SET' or
**  'T_VECTOR' might still  be a range.  It is  just that the kernel does not
**  know this.
**
**  This package consists of three parts.
**
**  The  first  part  consists  of the  macros  'LEN_RANGE', 'SET_LEN_RANGE',
**  'LOW_RANGE',   'SET_FIRST_RANGE',   'INC_RANGE',   'SET_INC_RANGE',   and
**  'ELM_RANGE'.   They determine the  representation of  ranges.  Everything
**  else in this file and the rest of the {\GAP} kernel  uses those macros to
**  access and modify ranges.
**
**  The  second  part  consists  of  the  functions  'LenRange',  'ElmRange',
**  'ElmsRange',    'AssRange',   'AsssRange',   'PosRange',    'PlainRange',
**  'IsDenseRange', 'IsPossRange', 'PrRange', 'EqRange', and 'LtRange'.  They
**  are the functions required by the  generic  lists package.   Using  these
**  functions  the other parts  of the {\GAP}  kernel  can  access or  modify
**  ranges without actually being aware that they are dealing with a range.
**
**  The  third part consists  of the  functions 'EvMakeRange', 'PrMakeRange',
**  'IsRange',  and 'FunIsRange'.  These functions make it possible  to  make
**  ranges, either by  evaluating  a range  literal, or by converting another
**  list to a range.
**
*/

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "scanner.h"             /* reading of tokens and printing  */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */

#include        "list.h"                /* generic list package            */
#include        "plist.h"               /* 'LEN_PLIST', 'SET_LEN_PLIST',.. */

#include        "range.h"               /* declaration part of the package */


/****************************************************************************
**
*F  SIZE_PLEN_RANGE(<plen>) . . . . . . size from physical length for a range
**
**  'SIZE_PLEN_RANGE' returns the size that the bag for a range with room for
**  <plen> elements must have.
**
**  Note that 'SIZE_PLEN_RANGE' is a macro, so do not call it with  arguments
**  that have sideeffects.
**
**  'SIZE_PLEN_RANGE'  is defined in the declaration  part of this package as
**  follows:
**
#define SIZE_PLEN_RANGE(PLEN)           (3 * SIZE_HD)
*/


/****************************************************************************
**
*F  LEN_RANGE(<hdRange>)  . . . . . . . . . . . . . . . . . length of a range
**
**  'LEN_RANGE' returns the logical length  of  the  range <hdRange>,  as a C
**  integer.
**
**  Note that 'LEN_RANGE' is a macro, so do not call it with  arguments  that
**  have sideeffects.
**
**  'LEN_RANGE' is defined in the declaration part of the package as follows:
**
#define LEN_RANGE(LIST)                 HD_TO_INT(PTR_BAG(LIST)[0])
*/


/****************************************************************************
**
*F  SET_LEN_RANGE(<hdRange>,<len>)  . . . . . . . . set the length of a range
**
**  'SET_LEN_RANGE'  sets  the  length  of the range <hdRange>  to  the value
**  <len>, which must be a C integer larger than 1.
**
**  Note that 'SET_LEN_RANGE' is a macro,  so  do not  call it with arguments
**  that have sideeffects.
**
**  'SET_LEN_RANGE' is  defined in the  declaration part of  this  package as
**  follows:
**
#define SET_LEN_RANGE(LIST,LEN)         (PTR_BAG(LIST)[0] = INT_TO_HD(LEN))
*/


/****************************************************************************
**
*F  LOW_RANGE(<hdRange>)  . . . . . . . . . . . . .  first element of a range
**
**  'LOW_RANGE'  returns the  first element of  the  range  <hdRange>  as a C
**  integer.
**
**  Note that 'LOW_RANGE' is a macro,  so do not call it  with arguments that
**  have sideeffects.
**
**  'LOW_RANGE' is  defined  in  the  declaration  part  of this  package  as
**  follows:
**
#define LOW_RANGE(LIST)                 HD_TO_INT(PTR_BAG(LIST)[1])
*/


/****************************************************************************
**
*F  SET_LOW_RANGE(<hdRange>,<low>)  . . . .  set the first element of a range
**
**  'SET_LOW_RANGE' sets the  first  element  of  the range <hdRange>  to the
**  value <low>, which must be a C integer.
**
**  Note  that 'SET_LOW_RANGE' is a macro, so do not call  it with  arguments
**  that have sideeffects.
**
**  'SET_LOW_RANGE' is defined  in  the  declaration  part of this package as
**  follows:
**
#define SET_LOW_RANGE(LIST,LOW)         (PTR_BAG(LIST)[1] = INT_TO_HD(LOW))
*/


/****************************************************************************
**
*F  INC_RANCE(<hdRange>)  . . . . . . . . . . . . . . .  increment of a range
**
**  'INC_RANGE' returns the increment of the range <hdRange> as a C integer.
**
**  Note that 'INC_RANGE' is a macros,  so do not call it with arguments that
**  have sideeffects.
**
**  'INC_RANGE'  is  defined  in  the declaration  part  of  this  package as
**  follows:
**
#define INC_RANGE(LIST)                 HD_TO_INT(PTR_BAG(LIST)[2])
*/


/****************************************************************************
**
*F  SET_INC_RANGE(<hdRange>,<inc>)  . . . . . .  set the increment of a range
**
**  'SET_INC_RANGE'  sets the  increment of the range <hdRange> to  the value
**  <inc>, which must be a C integer.
**
**  Note that  'SET_INC_RANGE' is a macro,  so do  not call it with arguments
**  that have sideeffects.
**
**  'SET_INC_RANGE'  is defined in  the declaration part  of this  package as
**  follows:
**
#define SET_INC_RANGE(LIST,INC)         (PTR_BAG(LIST)[2] = INT_TO_HD(INC))
*/


/****************************************************************************
**
*F  ELM_RANGE(<hdRange>,<i>)  . . . . . . . . . . . . . .  element of a range
**
**  'ELM_RANGE' return the <i>-th element of the  range <hdRange>.  <i>  must
**  be a positive integer less than or equal to the length of <hdRange>.
**
**  Note that 'ELM_RANGE' is a macro, so do not call it with  arguments  that
**  have sideeffects.
**
**  'ELM_RANGE' is defined in the declaration part of the package as follows:
**
#define ELM_RANGE(L,POS)        INT_TO_HD(LOW_RANGE(L)+(POS-1)*INC_RANGE(L))
*/


/****************************************************************************
**
*F  LenRange(<hdList>)  . . . . . . . . . . . . . . . . . . length of a range
**
**  'LenRange' returns the length of the range <hdList> as a C integer.
**
**  'LenRange' is the function in 'TabLenList' for ranges.
*/
Int            LenRange (Bag hdList)
{
    return LEN_RANGE( hdList );
}


/****************************************************************************
**
*F  ElmRange(<hdList>,<pos>)  . . . . . . . . .  select an element of a range
**
**  'ElmRange' selects the element at position <pos>  of the range  <hdList>.
**  It is the responsibility of the caller to ensure that <pos> is a positive
**  integer.  An error is  signaller if <pos>  is larger than  the length  of
**  <hdList>.
**
**  'ElmfRange' does  the same thing than 'ElmRange', but need not check that
**  <pos> is  less than or  equal to  the  length of  <hdList>,  this is  the
**  responsibility of the caller.
**
**  'ElmRange' is the  function in  'TabElmList' for ranges.   'ElmfRange' is
**  the  function in  'TabElmfList',  'TabElmlList',  and  'TabElmrList'  for
**  ranges.
*/
Bag       ElmRange (Bag hdList, Int pos)
{
    /* check the position                                                  */
    if ( LEN_RANGE( hdList ) < pos ) {
        return Error(
          "List Element: <list>[%d] must have a value",
                     pos, 0 );
    }

    /* return the selected element                                         */
    return ELM_RANGE( hdList, pos );
}

Bag       ElmfRange (Bag hdList, Int pos)
{
    return ELM_RANGE( hdList, pos );
}


/****************************************************************************
**
*F  ElmsRange(<hdList>,<hdPoss>)  . . . . . . . select a sublist from a range
**
**  'ElmsRange' returns a new list containing  the elements  at the positions
**  given  in  the  list  <hdPoss>  from  the  range  <hdList>.   It  is  the
**  responsibility of  the  caller  to ensure  that  <hdPoss>  is  dense  and
**  contains only  positive integers.  An error is signalled if an element of
**  <hdPoss> is larger than the length of <hdList>.
**
**  'ElmsRange' is the function in 'TabElmsList' for ranges.
*/
Bag       ElmsRange (Bag hdList, Bag hdPoss)
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
        lenList = LEN_RANGE( hdList );

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
            hdElm = ELM_RANGE( hdList, pos );

            /* assign the element into <elms>                              */
            SET_ELM_PLIST( hdElms, i, hdElm );

        }

    }

    /* special code for ranges                                             */
    else {

        /* get the length of <list>                                        */
        lenList = LEN_RANGE( hdList );

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

        /* make the result range                                           */
        hdElms = NewBag( T_RANGE, SIZE_PLEN_RANGE( lenPoss ) );
        SET_LEN_RANGE( hdElms, lenPoss );
        SET_LOW_RANGE( hdElms, HD_TO_INT( ELM_RANGE( hdList, pos ) ) );
        SET_INC_RANGE( hdElms, inc * INC_RANGE( hdList ) );

    }

    /* return the result                                                   */
    return hdElms;
}


/****************************************************************************
**
*F  AssRange(<hdList>,<pos>,<hdVal>)  . . . . . . . . . . . assign to a range
**
**  'AssRange'  assigns  the value  <hdVal> to  the  range  <hdList>  at  the
**  position <pos>.  It is the  responsibility  of the caller to ensure  that
**  <pos> is positive, and that <hdVal> is not 'HdVoid'.
**
**  'AssRange' is the function in 'TabAssList' for ranges.
**
**  'AssRange' simply converts the range into a plain list, and then does the
**  same stuff as 'AssPlist'.   This is because a range is not very likely to
**  stay a range after the assignment.
*/
Bag       AssRange (Bag hdList, Int pos, Bag hdVal)
{
    Int                plen;           /* physical length of <list>       */

    /* convert the range into a plain list                                 */
    PLAIN_LIST( hdList );
    Retype( hdList, T_LIST );

    /* resize the list if necessary                                        */
    if ( LEN_PLIST( hdList ) < pos ) {
        plen = PLEN_SIZE_PLIST( GET_SIZE_BAG(hdList) );
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
*F  AsssRange(<hdList>,<hdPoss>,<hdVals>)  assign several elements to a range
**
**  'AsssRange'  assignes the values  from the list <hdVals> at the positions
**  given   in   the  list   <hdPoss>  to  the  range <hdList>.   It  is  the
**  responsibility  of the caller  to  ensure  that  <hdPoss>  is  dense  and
**  contains only positive integers, that <hdPoss> and <hdVals> have the same
**  length, and that <hdVals> is dense.
**
**  'AsssRange' is the function in 'TabAsssList' for ranges.
**
**  'AsssRange' simply converts the range to a plain  list  and then does the
**  same stuff as 'AsssPlist'.  This is because a range is not very likely to
**  stay a range after the assignment.
*/
Bag       AsssRange (Bag hdList, Bag hdPoss, Bag hdVals)
{
    /* convert <list> to a plain list                                      */
    PLAIN_LIST( hdList );
    Retype( hdList, T_LIST );

    /* and delegate                                                        */
    return ASSS_LIST( hdList, hdPoss, hdVals );
}


/****************************************************************************
**
*F  PosRange(<hdRange>,<hdVal>,<start>) . . position of an element in a range
**
**  'PosRange' returns the  position  of  the  value  <hdVal>  in  the  range
**  <hdList> after the first position <start> as a  C integer.  0 is returned
**  if <hdVal> is not in the list.
**
**  'PosRange' is the function in 'TabPosList' for ranges.
*/
Int            PosRange (Bag hdList, Bag hdVal, Int start)
{
    Int                k;              /* position, result                */
    Int                lenList;        /* length of <list>                */
    Int                low;            /* first element of <list>         */
    Int                inc;            /* increment of <list>             */
    Int                val;            /* numerical value of <val>        */

    /* get the length, the first element, and the increment of <list>      */
    lenList = LEN_RANGE(hdList);
    low     = LOW_RANGE(hdList);
    inc     = INC_RANGE(hdList);

    /* look just beyond the end                                            */
    if ( start == lenList ) {
        k = 0;
    }

    /* look for an integer                                                 */
    else if ( GET_TYPE_BAG(hdVal) == T_INT ) {
        val = HD_TO_INT(hdVal);
        if ( 0 < inc
          && low + start * inc <= val && val <= low + (lenList-1) * inc
          && (val - low) % inc == 0 ) {
            k = (val - low) / inc + 1;
        }
        else if ( inc < 0
          && low + (lenList-1) * inc <= val && val <= low + start * inc
          && (val - low) % inc == 0 ) {
            k = (val - low) / inc + 1;
        }
        else {
            k = 0;
        }
    }

    /* for a record compare every entry                                    */
    else if ( GET_TYPE_BAG(hdVal) == T_REC ) {
        for ( k = start+1; k <= lenList; k++ ) {
            if ( EQ( INT_TO_HD( low + (k-1) * inc ), hdVal ) == HdTrue )
                break;
        }
        if ( lenList < k ) {
            k = 0;
        }
    }

    /* otherwise it can not be an element of the range                     */
    else {
        k = 0;
    }

    /* return the position                                                 */
    return k;
}


/****************************************************************************
**
*F  PlainRange(<hdList>)  . . . . . . . . . . convert a range to a plain list
**
**  'PlainRange' converts the range <hdList> to a plain list.
**
**  'PlainRange' is the function in 'TabPlainList' for ranges.
*/
void            PlainRange (Bag hdList)
{
    Int                lenList;        /* length of <list>                */
    Int                low;            /* first element of <list>         */
    Int                inc;            /* increment of <list>             */
    Int                i;              /* loop variable                   */

    /* get the length, the first element, and the increment of <list>      */
    lenList = LEN_RANGE( hdList );
    low     = LOW_RANGE( hdList );
    inc     = INC_RANGE( hdList );

    /* change the type of the list, and allocate enough space              */
    Retype( hdList, T_LIST );
    Resize( hdList, SIZE_PLEN_PLIST( lenList ) );
    SET_LEN_PLIST( hdList, lenList );

    /* enter the values in <list>                                          */
    for ( i = 1; i <= lenList; i++ ) {
        SET_ELM_PLIST( hdList, i, INT_TO_HD( low + (i-1) * inc ) );
    }

}


/****************************************************************************
**
*F  IsDenseRange(<hdList>)  . . . . . . . dense list test function for ranges
**
**  'IsDenseRange' returns 1, since ranges are always dense.
**
**  'IsDenseRange' is the function in 'TabIsDenseList' for ranges.
*/
Int            IsDenseRange (Bag hdList)
{
    return 1;
}


/****************************************************************************
**
*F  IsPossRange(<hdList>) . . . . . . positions list test function for ranges
**
**  'IsPossRange' returns 1 if  the range <hdList> is a dense list containing
**  only positive integers, and 0 otherwise.
**
**  'IsPossRange' is the function in 'TabIsPossList' for ranges.
*/
Int            IsPossRange (Bag hdList)
{
    /* test if the first element is positive                               */
    if ( LOW_RANGE( hdList ) <= 0 )
        return 0;

    /* test if the last element is positive                                */
    if ( HD_TO_INT( ELM_RANGE( hdList, LEN_RANGE(hdList) ) ) <= 0 )
        return 0;

    /* otherwise <list> is a positions list                                */
    return 1;
}


/****************************************************************************
**
*F  PrRange(<hdRange>)  . . . . . . . . . . . . . . . . . . . . print a range
**
**  'PrRange' prints the range <hdRange>.
**
**  'PrRange' handles bags of type 'T_RANGE' and 'T_MAKERANGE'.
*/
void            PrRange (Bag hdRange)
{
    Pr( "%2>[ %2>%d",
        LOW_RANGE(hdRange), 0 );
    if ( INC_RANGE(hdRange) != 1 )
        Pr( "%<,%< %2>%d",
            LOW_RANGE(hdRange)+INC_RANGE(hdRange), 0 );
    Pr( "%2< .. %2>%d%4< ]",
        LOW_RANGE(hdRange)+(LEN_RANGE(hdRange)-1)*INC_RANGE(hdRange),0);
}


/****************************************************************************
**
*F  EqRange(<hdL>,<hdR>)  . . . . . . . . . . .  test if two ranges are equal
**
**  'EqRange' returns 'true' if the two  ranges <hdL> and <hdR> are equal and
**  'false' otherwise.
**
**  Is  called from the 'EQ' binop  so both  operands are  already evaluated.
*/
Bag       EqRange (Bag hdL, Bag hdR)
{
    if ( LEN_RANGE(hdL) == LEN_RANGE(hdR)
      && LOW_RANGE(hdL) == LOW_RANGE(hdR)
      && INC_RANGE(hdL) == INC_RANGE(hdR) ) {
        return HdTrue;
    }
    else {
        return HdFalse;
    }
}


/****************************************************************************
**
*F  LtRange(<hdL>,<hdR>)  . . . . . . . . . . .  test if two ranges are equal
**
**  'LtRange' returns 'true' if the range  <hdL> is less than the range <hdR>
**  and 'false' otherwise.
**
**  Is called from the 'LT' binop so both operands are already evaluated.
*/
Bag       LtRange (Bag hdL, Bag hdR)
{
    /* first compare the first elements                                    */
    if ( LOW_RANGE(hdL) < LOW_RANGE(hdR) )
        return HdTrue;
    else if ( LOW_RANGE(hdR) < LOW_RANGE(hdL) )
        return HdFalse;

    /* next compare the increments (or the second elements)                */
    if ( INC_RANGE(hdL) < INC_RANGE(hdR) )
        return HdTrue;
    else if ( INC_RANGE(hdR) < INC_RANGE(hdL) )
        return HdFalse;

    /* finally compare the lengths                                         */
    if ( LEN_RANGE(hdL) < LEN_RANGE(hdR) )
        return HdTrue;
    else if ( LEN_RANGE(hdR) < LEN_RANGE(hdL) )
        return HdFalse;

    /* the two ranges are equal                                            */
    return HdFalse;
}


/****************************************************************************
**
*F  EvMakeRange(<hdMake>) . . .  convert a variable range into a constant one
**
**  'EvMakeRange' turns the literal  range  <hdMake>  into  a  constant  one.
*/
Bag       EvMakeRange (Bag hdMake)
{
    Bag           hdRange;        /* handle of the result            */
    Bag           hdL;            /* handle of the first element     */
    Int                low;            /* low value                       */
    Bag           hdH;            /* handle of the last element      */
    Int                high;           /* high value                      */
    Int                inc;            /* increment                       */

    /* evaluate the low value                                              */
    hdL = EVAL( PTR_BAG(hdMake)[0] );
    if ( GET_TYPE_BAG(hdL) != T_INT )
        return Error("Range: <low> must be an integer",0,0);
    low = HD_TO_INT( hdL );

    /* evaluate the second value (if present)                              */
    if ( GET_SIZE_BAG( hdMake ) == 3 * SIZE_HD ) {
        hdH = EVAL( PTR_BAG(hdMake)[1] );
        if ( GET_TYPE_BAG(hdH) != T_INT )
            return Error("Range: <second> must be an integer",0,0);
        if ( HD_TO_INT( hdH ) == low )
            return Error("Range: <second> must not be equal to <low>",0,0);
        inc = HD_TO_INT( hdH ) - low;
    }
    else {
        inc = 1;
    }

    /* evaluate the high value                                             */
    hdH = EVAL( PTR_BAG(hdMake)[GET_SIZE_BAG(hdMake)/SIZE_HD-1] );
    if ( GET_TYPE_BAG( hdH ) != T_INT )
        return Error("Range: <high> must be an integer",0,0);
    high = HD_TO_INT( hdH );

    /* check that <high>-<low> is divisable by <inc>                       */
    if ( (high - low) % inc != 0 )
        return Error("Range: <high>-<low> must be divisible by <inc>",0,0);

    /* if <low> is larger than <high> the range is empty                   */
    if ( (0 < inc && high < low) || (inc < 0 && low < high) ) {
        hdRange = NewBag( T_LIST, SIZE_PLEN_PLIST( 0 ) );
        SET_LEN_PLIST( hdRange, 0 );
    }

    /* if <low> is equal to <high> the range is a singleton list           */
    else if ( low == high ) {
        hdRange = NewBag( T_LIST, SIZE_PLEN_PLIST( 1 ) );
        SET_LEN_PLIST( hdRange, 1 );
        SET_ELM_PLIST( hdRange, 1, INT_TO_HD( low ) );
    }

    /* else make the range                                                 */
    else {
        hdRange = NewBag( T_RANGE, SIZE_PLEN_RANGE( (high-low) / inc + 1 ) );
        SET_LEN_RANGE( hdRange, (high-low) / inc + 1 );
        SET_LOW_RANGE( hdRange, low );
        SET_INC_RANGE( hdRange, inc );
    }

    /* return the range                                                    */
    return hdRange;
}


/****************************************************************************
**
*F  PrMakeRange(<hdMake>) . . . . . . . . . . . . . . . print a range literal
**
**  'PrMakeRange' prints the range literal  <hdMake> in the form '[  <low> ..
**  <high> ]'.
*/
void            PrMakeRange (Bag hdMake)
{
    if ( GET_SIZE_BAG( hdMake ) == 2 * SIZE_HD ) {
        Pr("%2>[ %2>",0,0);    Print( PTR_BAG(hdMake)[0] );
        Pr("%2< .. %2>",0,0);  Print( PTR_BAG(hdMake)[1] );
        Pr(" %4<]",0,0);
    }
    else {
        Pr("%2>[ %2>",0,0);    Print( PTR_BAG(hdMake)[0] );
        Pr("%<,%< %2>",0,0);   Print( PTR_BAG(hdMake)[1] );
        Pr("%2< .. %2>",0,0);  Print( PTR_BAG(hdMake)[2] );
        Pr(" %4<]",0,0);
    }
}


/****************************************************************************
**
*F  IsRange(<hdList>) . . . . . . . . . . . . . . . test if a list is a range
**
**  'IsRange' returns 1 if the list with the handle <hdList> is a range and 0
**  otherwise.  As a  sideeffect 'IsRange' converts proper ranges represented
**  the ordinary way to the compact representation.
*/
Int            IsRange (Bag hdList)
{
    Int                isRange;        /* result of the test              */
    Int                len;            /* logical length of list          */
    Int                low;            /* value of first element of range */
    Int                inc;            /* increment                       */
    Int                i;              /* loop variable                   */

    /* if <hdList> is represented as a range, it is of course a range      */
    if ( GET_TYPE_BAG(hdList) == T_RANGE ) {
        isRange = 1;
    }

    /* if <hdList> is not a list, it is not a range                        */
    else if ( ! IS_LIST( hdList ) ) {
        isRange = 0;
    }

    /* if <hdList> is the empty list, it is a range by definition          */
    else if ( LEN_LIST(hdList) == 0 ) {
        isRange = 1;
    }

    /* if <hdList> is a list with just one integer, it is also a range     */
    else if ( LEN_LIST(hdList)==1 && GET_TYPE_BAG(ELMF_LIST(hdList,1))==T_INT ) {
        isRange = 1;
    }

    /* if the first element is not an integer, it is not a range           */
    else if ( ELMF_LIST(hdList,1)==0 || GET_TYPE_BAG(ELMF_LIST(hdList,1))!=T_INT ) {
        isRange = 0;
    }

    /* if the second element is not an integer, it is not a range          */
    else if ( ELMF_LIST(hdList,2)==0 || GET_TYPE_BAG(ELMF_LIST(hdList,2))!=T_INT ) {
        isRange = 0;
    }

    /* if the first and the second element are equal it is also not a range*/
    else if ( ELMF_LIST(hdList,1) == ELMF_LIST(hdList,2) ) {
        isRange = 0;
    }

    /* otherwise, test if the elements are consecutive integers            */
    else {

        /* get the logical length of the list                              */
        len = LEN_LIST(hdList);
        low = HD_TO_INT( ELMF_LIST( hdList, 1 ) );
        inc = HD_TO_INT( ELMF_LIST( hdList, 2 ) ) - low;

        /* test all entries against the first one                          */
        for ( i = 3;  i <= len;  i++ ) {
            if ( ELMF_LIST(hdList,i) != INT_TO_HD( low + (i-1) * inc ) )
                break;
        }

        /* if <hdList> is a range, convert to the compact representation   */
        isRange = (len < i);
        if ( isRange && 2 < len ) {
            Retype( hdList, T_RANGE );
            Resize( hdList, SIZE_PLEN_RANGE( len ) );
            SET_LEN_RANGE( hdList, len );
            SET_LOW_RANGE( hdList, low );
            SET_INC_RANGE( hdList, inc );
        }

    }

    /* return the result of the test                                       */
    return isRange;
}


/****************************************************************************
**
*F  FunIsRange(<hdCall>)  . . . . . . . . . . . . . . . . .  test for a range
**
**  'FunIsRange' implements the internal function 'IsRange'.
**
**  'IsRange( <obj> )'
**
**  'IsRange' returns 'true' if <obj>, which may be an object of any type, is
**  a range and 'false' otherwise.  A range is a list without holes such that
**  the elements are  consecutive integers.  Will cause an  error if <obj> is
**  an unassigned variable.
*/
Bag       FunIsRange (Bag hdCall)
{
    Bag           hdObj;          /* handle of the argument          */

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: IsRange( <obj> )",0,0);
    hdObj = EVAL( PTR_BAG(hdCall)[1] );
    if ( hdObj == HdVoid )
        return Error("IsRange: function must return a value",0,0);

    /* let 'IsRange' do the work for lists                                 */
    return IsRange(hdObj) ? HdTrue : HdFalse;
}


/****************************************************************************
**
*F  InitRange() . . . . . . . . . . . . . . . .  initialize the range package
**
**  'InitRange' initializes the range package.
*/
void            InitRange (void)
{

    /* install the list functions in the tables                            */
    TabIsList[T_RANGE]      = 1;
    TabLenList[T_RANGE]     = LenRange;
    TabElmList[T_RANGE]     = ElmRange;
    TabElmfList[T_RANGE]    = ElmfRange;
    TabElmlList[T_RANGE]    = ElmfRange;
    TabElmrList[T_RANGE]    = ElmfRange;
    TabElmsList[T_RANGE]    = ElmsRange;
    TabAssList[T_RANGE]     = AssRange;
    TabAsssList[T_RANGE]    = AsssRange;
    TabPosList[T_RANGE]     = PosRange;
    TabPlainList[T_RANGE]   = PlainRange;
    TabIsDenseList[T_RANGE] = IsDenseRange;
    TabIsPossList[T_RANGE]  = IsPossRange;
    EvTab[T_RANGE]          = EvList;
    PrTab[T_RANGE]          = PrRange;
    TabLt[T_RANGE][T_RANGE] = LtRange;

    /* install the functions to make a range                               */
    EvTab[T_MAKERANGE]      = EvMakeRange;
    PrTab[T_MAKERANGE]      = PrMakeRange;

    /* install the internal function                                       */
    InstIntFunc( "IsRange", FunIsRange );

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



