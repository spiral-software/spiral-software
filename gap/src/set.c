/****************************************************************************
**
*A  set.c                       GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file  contains  the functions that  mainly operate  on  proper sets.
**  As sets are special lists many things are done in the list package.
**
**  A *proper set* is a list that has no holes, no duplicates, and is sorted.
**  For the full definition  of sets see chapter "Sets" in the {\GAP} Manual.
**  Read also section "More about Sets" about the internal flag for sets.
**
**  A list that is known to be a set is represented by a bag of type 'T_SET',
**  which has exactely the  same representation as bags of type 'T_LIST'.  As
**  a  matter of fact the functions in this  file do not really know how this
**  representation   looks,    they   use   the   macros   'SIZE_PLEN_PLIST',
**  'PLEN_SIZE_PLIST',   'LEN_PLIST',   'SET_LEN_PLIST',   'ELM_PLIST',   and
**  'SET_ELM_PLIST' exported by the plain list package.
**
**  Note that  a list represented by a  bag of  type  'T_LIST', 'T_VECTOR' or
**  'T_VECFFE'  might still be a  set.  It is just  that  the kernel does not
**  known this.
**
**  This package consists of two parts.
**
**  The first part consists  of the functions 'LenSet', 'ElmSet',  'ElmsSet',
**  'AssSet',  'AsssSet',  'PosSet',  'PlainSet',  'IsDenseSet', 'IsPossSet',
**  'EqSet',  and  'LtSet'.   They are  the functions required by the generic
**  lists  package.  Using  these  functions the  other  parts of  the {\GAP}
**  kernel can access and modify sets without  actually being aware that they
**  are dealing with a set.
**
**  The second  part consists of the  functions 'SetList', 'FunSet', 'IsSet',
**  'FunIsSet',     'FunIsEqualSet',      'FunIsSubsetSet',      'FunAddSet',
**  'FunRemoveSet', 'FunUniteSet', 'FunIntersectSet',  and  'FunSubtractSet'.
**  These functions  make it possible  to  make sets, either by converting  a
**  list to a set, or  by computing the union, intersection, or difference of
**  two sets.
**
*/

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "scanner.h"             /* reading of tokens and printing  */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */

#include        "list.h"                /* generic lists package           */
#include        "plist.h"               /* 'LEN_PLIST', 'SET_LEN_PLIST',.. */
#include        "range.h"               /* 'LEN_RANGE', 'LOW_RANGE', ..    */

#include        "set.h"                 /* declaration part of the package */


/****************************************************************************
**
*F  LenSet(<hdList>)  . . . . . . . . . . . . . . . . . . . . length of a set
**
**  'LenSet' returns the length of the set <hdList> as a C integer.
**
**  'LenSet' is the function in 'TabLenList' for sets.
*/
Int            LenSet (Bag hdList)
{
    return LEN_PLIST( hdList );
}


/****************************************************************************
**
*F  ElmSet(<hdList>,<pos>)  . . . . . . . . . . .  select an element of a set
**
**  'ElmSet'  selects the element at position <pos> of the  set <hdList>.  It
**  is the responsibility of the caller to  ensure  that <pos> is a  positive
**  integer.  An error  is signalled if <pos> is  larger than the  length  of
**  <hdList>.
**
**  'ElmfSet'  does the same thing than  'ElmList', but need  not check  that
**  <pos>  is less than or equal to the  length  of  <hdList>,  this  is  the
**  responsibility of the caller.
**
**  'ElmSet' is  the  function  in 'TabElmList'  for sets.   'ElmfSet' is the
**  function in 'TabElmfList', 'TabElmlList', and 'TabElmrList' for sets.
*/
Bag       ElmSet (Bag hdList, Int pos)
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

    /* return the element                                                  */
    return hdElm;
}

Bag       ElmfSet (Bag hdList, Int pos)
{
    /* select and return the element                                       */
    return ELM_PLIST( hdList, pos );
}


/****************************************************************************
**
*F  ElmsSet(<hdList>,<hdPoss>)  . . . . . . . . . select a sublist from a set
**
**  'ElmsSet'  returns  a  new  list containing the elements at  the position
**  given  in   the  list  <hdPoss>  from  the  set  <hdList>.   It  is   the
**  responsibility  of  the caller  to  ensure  that  <hdPoss>  is dense  and
**  contains only positive integers.  An error is signalled if an element  of
**  <hdPoss> is larger than the length of <hdList>.
**
**  'ElmsSet' is the function in 'TabElmsList' for sets.
*/
Bag       ElmsSet (Bag hdList, Bag hdPoss)
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
        if ( GET_TYPE_BAG(hdPoss) == T_SET )
            hdElms = NewBag( T_SET, SIZE_PLEN_PLIST( lenPoss ) );
        else
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
        if ( 0 < inc )
            hdElms = NewBag( T_SET, SIZE_PLEN_PLIST( lenPoss ) );
        else
            hdElms = NewBag( T_LIST, SIZE_PLEN_PLIST( lenPoss ) );
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
*F  AssSet(<hdList>,<pos>,<hdVal>)  . . . . . . . . . . . . . assign to a set
**
**  'AssSet' assigns  the  value <hdVal> to the set <hdList> at the  position
**  <pos>.   It is the responsibility of  the caller  to ensure that <pos> is
**  positive, and that <hdVal> is not 'HdVoid'.
**
**  If the position is larger then the length of the list <list>, the list is
**  automatically extended.   To avoid making this too often, the bag  of the
**  list is extended by at least '<length>/8 + 4' handles.  Thus in the loop
**
**      l := [];  for i in [1..1024]  do l[i] := i^2;  od;
**
**  the list 'l' is extended only 32 times not 1024 times.
**
**  'AssSet' is the function in 'TabAssList' for sets.
**
**  'AssSet'  simply converts the set into  a plain list,  and then does  the
**  same stuff as 'AssPlist'.  This  is because a set is  not very  likely to
**  stay a set after the assignment.
*/
Bag       AssSet (Bag hdList, Int pos, Bag hdVal)
{
    Int                plen;           /* physical length of <list>       */

    /* get the logical length of <list>                                    */
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
*F  AsssSet(<hdList>,<hdPoss>,<hdVals>) . .  assign several elements to a set
**
**  'AsssSet' assignes  the values from the  list  <hdVals>  at the positions
**  given in the list <hdPoss> to the set <hdList>.  It is the responsibility
**  of the caller to ensure that <hdPoss> is dense and contains only positive
**  integers,  that <hdPoss>  and  <hdVals>  have  the same  length, and that
**  <hdVals> is dense.
**
**  'AsssSet' is the function in 'TabAsssList' for plain lists.
**
**  'AsssSet' simply converts the  set to a plain list and then does the same
**  stuff as 'AsssPlist'.  This is because a set is not very likely to stay a
**  set after the assignment.
*/
Bag       AsssSet (Bag hdList, Bag hdPoss, Bag hdVals)
{
    /* convert <list> to a plain list                                      */
    Retype( hdList, T_LIST );

    /* and delegate                                                        */
    return ASSS_LIST( hdList, hdPoss, hdVals );
}


/****************************************************************************
**
*F  PosSet(<hdList>,<hdVal>,<start>)  . . . . position of an element in a set
**
**  'PosSet' returns  the  position of the value <hdVal>  in the set <hdList>
**  after  the  first  position  <start> as  a C integer.  0  is returned  if
**  <hdVal> is not in the list.
**
**  'PosSet' is the function in 'TabPosList' for plain lists.
*/
Int            PosSet (Bag hdList, Bag hdVal, Int start)
{
    UInt       lenList;        /* logical length of the set       */
    UInt       i, j, k;        /* loop variables                  */

    /* get a pointer to the set and the logical length of the set          */
    lenList = LEN_PLIST( hdList );

    /* perform the binary search to find the position                      */
    i = start;  k = lenList + 1;
    while ( i+1 < k ) {                 /* set[i] < elm && elm <= set[k]   */
        j = (i + k) / 2;                /* i < j < k                       */
        if ( LT( ELM_PLIST(hdList,j), hdVal ) == HdTrue )  i = j;
        else                                               k = j;
    }

    /* test if the element was found at position k                         */
    if ( lenList < k || EQ( ELM_PLIST(hdList,k), hdVal ) != HdTrue )
        k = 0;

    /* return the position                                                 */
    return k;
}


/****************************************************************************
**
*F  PlainSet(<hdList>)  . . . . . . . . . . . . convert a set to a plain list
**
**  'PlainSet' converts the set <hdList> to a plain list.  Not much work.
**
**  'PlainSet' is the function in 'TabPlainList' for sets.
*/
void            PlainSet (Bag hdList)
{
    return;
}


/****************************************************************************
**
*F  IsDenseSet(<hdList>)  . . . . . . . . . dense list test function for sets
**
**  'IsDenseSet' returns 1, since every set is dense.
**
**  'IsDenseSet' is the function in 'TabIsDenseList' for sets.
*/
Int            IsDenseSet (Bag hdList)
{
    return 1;
}


/****************************************************************************
**
*F  IsPossSet(<hdList>) . . . . . . . . positions list test function for sets
**
**  'IsPossSet' returns 1 if the set <hdList> is a dense list containing only
**  positive integers, and 0 otherwise.
**
**  'IsPossSet' is the function in 'TabIsPossList' for sets.
*/
Int            IsPossSet (Bag hdList)
{
    Int                lenList;        /* length of <list>                */
    Bag           hdElm;          /* one element of <list>           */

    /* get the length of the variable                                      */
    lenList = LEN_PLIST( hdList );
    if ( lenList == 0 )
        return 1;

    /* test the first element                                              */
    hdElm = ELM_PLIST( hdList, 1 );
    if ( GET_TYPE_BAG(hdElm) != T_INT || HD_TO_INT(hdElm) <= 0 )
        return 0;

    /* test the last element                                               */
    hdElm = ELM_PLIST( hdList, lenList );
    if ( GET_TYPE_BAG(hdElm) != T_INT )
        return 0;

    /* no problems found                                                   */
    return 1;
}


/****************************************************************************
**
*F  EqSet(<hdL>,<hdR>)  . . . . . . . . . . . . .  test if two sets are equal
**
**  'EqList' returns  'true' if  the two sets <hdL> and  <hdR>  are equal and
**  'false' otherwise.
**
**  Is called from the 'EQ' binop so both  operands  are  already  evaluated.
*/
Bag       EqSet (Bag hdL, Bag hdR)
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
        if ( hdElmL != hdElmR && EQ( hdElmL, hdElmR ) == HdFalse ) {
            return HdFalse;
        }
    }

    /* no differences found, the lists are equal                           */
    return HdTrue;
}


/****************************************************************************
**
*F  LtSet(<hdL>,<hdR>)  . . . . . . . . . . . . .  test if two sets are equal
**
**  'LtSet'  returns 'true' if the  set <hdL> is less than the set <hdR>  and
**  'false' otherwise.
**
**  Is called from the 'LT' binop so both operands are already evaluated.
*/
Bag       LtSet (Bag hdL, Bag hdR)
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
        if ( hdElmL != hdElmR && EQ( hdElmL, hdElmR ) == HdFalse ) {
            return LT( hdElmL, hdElmR );
        }
    }

    /* reached the end of at least one list                                */
    return (lenL < lenR) ? HdTrue : HdFalse;
}


/****************************************************************************
**
*F  SetList(<hdList>) . . . . . . . . . . . . . . . .  make a set from a list
**
**  'SetList' returns the handle of a new set that contains the  elements  of
**  <hdList>.  Note that 'SetList' returns a new list even  if  <hdList>  was
**  already a set.  In this case 'SetList' is equal to 'ShallowCopy'.
**
**  'SetList' makes a copy of the list <hdList>, removes the holes, sorts the
**  copy and finally removes duplicates, which must appear next to each other
**  now that the copy is sorted.
*/
Bag       SetList (Bag hdList)
{
    Bag           hdSet;          /* handle of the result set        */
    Bag           hdElm;          /* one element of <list>           */
    Int                lenSet;         /* length of <set>                 */
    Int                lenList;        /* length of <list>                */
    Int                mutable;        /* the elements are mutable        */
    Int                h;              /* gap width in the shellsort      */
    Int                i, k;           /* loop variables                  */

    /* make a dense copy                                                   */
    lenList = LEN_LIST( hdList );
    hdSet = NewBag( T_SET, SIZE_PLEN_PLIST( lenList ) );
    lenSet = 0;
    mutable = 0;
    for ( i = 1; i <= lenList; i++ ) {
        hdElm = ELMF_LIST( hdList, i );
        if ( hdElm != 0 ) {
            lenSet += 1;
            mutable = mutable || (T_LIST <= GET_TYPE_BAG(hdElm));
            SET_ELM_PLIST( hdSet, lenSet, hdElm );
        }
    }

    /* sort the set with a shellsort                                       */
    h = 1;  while ( 9*h + 4 < lenSet )  h = 3*h + 1;
    while ( 0 < h ) {
        for ( i = h+1; i <= lenSet; i++ ) {
            hdElm = ELM_PLIST( hdSet, i );  k = i;
            while ( h < k && LT( hdElm, ELM_PLIST(hdSet,k-h) ) == HdTrue ) {
                SET_ELM_PLIST( hdSet, k, ELM_PLIST(hdSet,k-h) );
                k -= h;
            }
            SET_ELM_PLIST( hdSet, k, hdElm );
        }
        h = h / 3;
    }

    /* remove duplicates                                                   */
    if ( 0 < lenSet ) {
        hdElm = ELM_PLIST( hdSet, 1 );
        k = 1;
        for ( i = 2; i <= lenSet; i++ ) {
            if ( EQ( hdElm, ELM_PLIST( hdSet, i ) ) != HdTrue ) {
                k += 1;
                hdElm = ELM_PLIST( hdSet, i );
                SET_ELM_PLIST( hdSet, k, hdElm );
            }
        }
        if ( k < lenSet )
            lenSet = k;
    }

    /* resize the bag if possible                                          */
    if ( mutable )
        Retype( hdSet, T_LIST );
    if ( lenSet < lenList )
        Resize( hdSet, SIZE_PLEN_PLIST( lenSet ) );
    SET_LEN_PLIST( hdSet, lenSet );

    /* return set                                                          */
    return hdSet;
}


/****************************************************************************
**
*F  FunSet(<hdCall>)  . . . . . . . . . . . . . . . .  make a set from a list
**
**  'FunSet' implements the internal function 'Set'.
**
**  'Set( <list> )'
**
**  'Set' returns a  new proper  set, which  is represented as  a sorted list
**  without holes or duplicates, containing the elements of the list <list>.
**
**  'Set' returns a new list even if the list <list> is already a proper set,
**  in this  case  it is   equivalent to  'ShallowCopy' (see  "ShallowCopy").
*/
Bag       FunSet (Bag hdCall)
{
    Bag           hdSet;          /* handle of the result            */
    Bag           hdList;         /* handle of the argument          */
    Int                lenList;        /* length of <list>                */
    Int                i;              /* loop variable                   */

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: Set( <obj> )",0,0);
    hdList = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IS_LIST( hdList ) ) {
        return Error(
          "Set: <list> must be a list",
                     0, 0 );
    }
    
    /* if <list> is a set just shallow copy it                             */
    if ( IsSet( hdList ) ) {
        lenList = LEN_PLIST(hdList);
        hdSet = NewBag( GET_TYPE_BAG(hdList), SIZE_PLEN_PLIST( lenList ) );
        SET_LEN_PLIST( hdSet, lenList );
        for ( i = 1; i <= lenList; i++ ) {
            SET_ELM_PLIST( hdSet, i, ELM_PLIST( hdList, i ) );
        }
    }

    /* otherwise let 'SetList' do the work                                 */
    else {
        hdSet = SetList( hdList );
    }

    /* return the set                                                      */
    return hdSet;
}


/****************************************************************************
**
*F  IsSet(<hdList>) . . . . . . . . . . . . . . . . . test if a list is a set
**
**  'IsSet' returns 1  if the list <hdList> is  a proper set and 0 otherwise.
**  A proper set is a list that has  no holes,  no duplicates, and is sorted.
**  As a sideeffect 'IsSet' changes the type of proper sets to 'T_SET'.
**
**  A typical call in the set functions looks like this:                   \\
**  |    if ( ! IsSet(hdList) )  hdList = SetList(hdList); |               \\
**  This tests if  'hdList' is a proper  set.   If it is,   then the type  is
**  changed to 'T_SET'.  If it is not then 'SetList' is called to make a copy
**  of 'hdList', remove the holes, sort  the copy, and remove the duplicates.
*/
Int            IsSet (Bag hdList)
{
    Int                isSet;          /* result                          */
    Int                lenList;        /* length of <list>                */
    Bag           hdElm1, hdElm2; /* two elements of <list>          */
    Int                mutable;        /* are the elements mutable        */
    Int                i;              /* loop variable                   */

    /* if <list> is not a list, it certainly is not a set                  */
    if ( ! IS_LIST( hdList ) ) {
        isSet = 0;
    }

    /* if <list> is already a set, very good                               */
    else if ( GET_TYPE_BAG(hdList) == T_SET ) {
        isSet = 1;
    }

    /* if <list> is a range, it is a set if the increment is positive      */
    else if ( GET_TYPE_BAG(hdList) == T_RANGE && 0 < INC_RANGE(hdList) ) {
        PLAIN_LIST( hdList );
        Retype( hdList, T_SET );
        isSet = 1;
    }
    else if ( GET_TYPE_BAG(hdList) == T_RANGE ) {
        isSet = 0;
    }

    /* if <list> is empty, it is a set                                     */
    else if ( LEN_LIST(hdList) == 0 ) {
        PLAIN_LIST( hdList );
        Retype( hdList, T_SET );
        isSet = 1;
    }

    /* if <list> has a hole at the first position, it is not a set         */
    else if ( ELMF_LIST( hdList, 1 ) == 0 ) {
        isSet = 0;
    }

    /* otherwise convert to a plain list, and compare                      */
    else {
        PLAIN_LIST( hdList );
        lenList = LEN_PLIST( hdList );
        hdElm1 = ELM_PLIST( hdList, 1 );
        mutable = T_LIST <= GET_TYPE_BAG(hdElm1);
        for ( i = 2; i <= lenList; i++ ) {
            hdElm2 = ELM_PLIST( hdList, i );
            if ( hdElm2 == 0 || LT( hdElm1, hdElm2 ) != HdTrue )
                break;
            mutable = mutable || (T_LIST <= GET_TYPE_BAG(hdElm2));
            hdElm1 = hdElm2;
        }
        isSet = (lenList < i);
        if ( isSet && ! mutable )  Retype( hdList, T_SET );
    }

    /* return the result                                                   */
    return isSet;
}


/****************************************************************************
**
*F  FunIsSet(<hdCall>)  . . . . . . . . . . . . .  test if an object is a set
**
**  'FunIsSet' implements the internal function 'IsSet'.
**
**  'IsSet( <obj> )'
**
**  'IsSet'  returns   'true' if the   object   <obj> is  a set  and  'false'
**  otherwise.  An object is a  set if it is a  sorted lists without holes or
**  duplicates.  Will cause an  error if evaluation of   <obj> is an  unbound
**  variable.
*/
Bag       FunIsSet (Bag hdCall)
{
    Bag           hdObj;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: IsSet( <obj> )",0,0);
    hdObj = EVAL( PTR_BAG(hdCall)[1] );
    if ( hdObj == HdVoid )
        return Error("IsSet: function must return a value",0,0);

    /* let 'IsSet' do the work                                             */
    return IsSet( hdObj ) ? HdTrue : HdFalse;
}


/****************************************************************************
**
*F  FunIsEqualSet(<hdCall>) . . . . .   test if a two lists are equal as sets
**
**  'FunIsEqualSet' implements the internal function 'IsEqualSet'.
**
**  'IsEqualSet( <set1>, <set2> )'
**
**  'IsEqualSet' returns  'true' if the   two lists <list1>  and <list2>  are
**  equal *when viewed as sets*, and  'false' otherwise.  <list1> and <list2>
**  are equal if  every element of <list1> is  also an element of <list2> and
**  if every element of <list2> is also an element of <list1>.
*/
Bag       FunIsEqualSet (Bag hdCall)
{
    Bag           hdSet1;         /* handle  of the left  set        */
    Bag           hdSet2;         /* handle  of the right set        */
    UInt       l1;             /* length  of the left  set        */
    UInt       l2;             /* length  of the right set        */
    UInt       i;              /* loop variable                   */

    /* get and check the arguments, convert to sets if necessary           */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )
        return Error("usage: IsEqualSet( <set1>, <set2> )",0,0);
    hdSet1 = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IS_LIST(hdSet1) )
        return Error("IsEqualSet: <set1> must be a list",0,0);
    hdSet2 = EVAL( PTR_BAG(hdCall)[2] );
    if ( ! IS_LIST(hdSet2) )
        return Error("IsEqualSet: <set2> must be a list",0,0);
    if ( ! IsSet( hdSet1 ) )  hdSet1 = SetList( hdSet1 );
    if ( ! IsSet( hdSet2 ) )  hdSet2 = SetList( hdSet2 );

    /* get and compare the logical lengths and get the pointer             */
    l1 = LEN_PLIST( hdSet1 );
    l2 = LEN_PLIST( hdSet2 );
    if ( l1 != l2 )  return HdFalse;

    /* now compare the two sets componentwise                              */
    for ( i = 1; i <= l1; i++ ) {
        if ( ELM_PLIST(hdSet1,i) != ELM_PLIST(hdSet2,i)
          && EQ( ELM_PLIST(hdSet1,i), ELM_PLIST(hdSet2,i) ) != HdTrue )
            break;
    }

    /* return 'true' if all elements are equal                             */
    return (i == l1+1) ? HdTrue : HdFalse;
}


/****************************************************************************
**
*F  FunIsSubsetSet(<hdCall>)  . . .  test if a set is a subset of another set
**
**  'FunIsSubsetSet' implements the internal function 'IsSubsetSet'.
**
**  'IsSubsetSet( <set1>, <set2> )'
**
**  'IsSubsetSet' returns 'true' if the  set <set2> is a   subset of the  set
**  <set1>, that is if every element of <set2>  is also an element of <set1>.
**  Either argument  may also be  a list that  is not a  proper set, in which
**  case 'IsSubsetSet' silently applies 'Set' (see "Set") to it first.
*/
Bag       FunIsSubsetSet (Bag hdCall)
{
    Bag           hdSet1;         /* handle of  the left  set        */
    Bag           hdSet2;         /* handle of  the right set        */
    UInt       l1;             /* length of  the left  set        */
    UInt       l2;             /* length of  the right set        */
    UInt       i1;             /* index into the left  set        */
    UInt       i2;             /* index into the right set        */
    UInt       i, j, k;        /* loop variables                  */

    /* get and check the arguments, convert to sets if necessary           */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )
        return Error("usage: IsSubsetSet( <set1>, <set2> )",0,0);
    hdSet1 = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IS_LIST(hdSet1) )
        return Error("IsSubsetSet: <set1> must be a list",0,0);
    hdSet2 = EVAL( PTR_BAG(hdCall)[2] );
    if ( ! IS_LIST(hdSet2) )
        return Error("IsSubsetSet: <set2> must be a list",0,0);
    if ( ! IsSet( hdSet1 ) )  hdSet1 = SetList( hdSet1 );

    /* special case if the second argument is a set                        */
    if ( IsSet( hdSet2 ) ) {

        /* get the logical lengths and get the pointer                     */
        l1 = LEN_PLIST( hdSet1 );
        l2 = LEN_PLIST( hdSet2 );
        i1 = 1;
        i2 = 1;

        /* now compare the two sets                                        */
        while ( i2 <= l2 && l2 + i1 <= l1 + i2 ) {
            if ( ELM_PLIST(hdSet1,i1) == ELM_PLIST(hdSet2,i2)
              || EQ(ELM_PLIST(hdSet1,i1),ELM_PLIST(hdSet2,i2)) == HdTrue ) {
                i1++;  i2++;
            }
            else if (LT(ELM_PLIST(hdSet1,i1),ELM_PLIST(hdSet2,i2))==HdTrue) {
                i1++;
            }
            else {
                break;
            }
        }

    }

    /* general case                                                        */
    else {

        /* first convert the other argument into a proper list             */
        PLAIN_LIST( hdSet2 );

        /* get the logical lengths                                         */
        l1 = LEN_PLIST( hdSet1 );
        l2 = LEN_PLIST( hdSet2 );

        /* loop over the second list and look for every element            */
        for ( i2 = 1; i2 <= l2; i2++ ) {

            /* ignore holes                                                */
            if ( ELM_PLIST(hdSet2,i2) == 0 )
                continue;

            /* perform the binary search to find the position              */
            i = 0;  k = l1+1;
            while ( i+1 < k ) {
                j = (i + k) / 2;
                if ( LT(ELM_PLIST(hdSet1,j),ELM_PLIST(hdSet2,i2)) == HdTrue )
                    i = j;
                else
                    k = j;
            }

            /* test if the element was found at position k                 */
            if ( l1 < k
              || EQ(ELM_PLIST(hdSet1,k),ELM_PLIST(hdSet2,i2)) != HdTrue ) {
                break;
            }

        }

    }

    /* return 'true' if every element of <set2> appeared in <set1>         */
    return (i2 == l2 + 1) ? HdTrue : HdFalse;
}


/****************************************************************************
**
*F  FunAddSet(<hdCall>) . . . . . . . . . . . . . . . add an element to a set
**
**  'FunAddSet' implements the internal function 'AddSet'.
**
**  'AddSet( <set>, <obj> )'
**
**  'AddSet' adds <obj>, which may be an object  of an arbitrary type, to the
**  set <set>, which must be a proper set.  If <obj> is already an element of
**  the set <set>, then <set> is not changed.  Otherwise <obj> is inserted at
**  the correct position such that <set> is again a set afterwards.
**
**  'AddSet' does not return  anything, it is only  called for the sideeffect
**  of changing <set>.
*/
Bag       FunAddSet (Bag hdCall)
{
    Bag           hdSet;          /* handle of the set               */
    Bag           hdObj;          /* handle of the object            */
    UInt       len;            /* logical length of the list      */
    UInt       i,  j,  k;      /* loop variables                  */

    /* get and check the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )
        return Error("usage: AddSet( <set>, <obj> )",0,0);
    hdSet = EVAL( PTR_BAG(hdCall)[1] );
    hdObj = EVAL( PTR_BAG(hdCall)[2] );
    if ( ! IsSet( hdSet ) )
        return Error("AddSet: <set> must be a proper set",0,0);
    if ( hdObj == HdVoid )
        return Error("AddSet: <obj> function must return a value",0,0);

    /* perform the binary search to find the position                      */
    len   = LEN_PLIST( hdSet );
    i = 0;  k = len+1;
    while ( i+1 < k ) {                 /* set[i] < elm && elm <= set[k]   */
        j = (i + k) / 2;                /* i < j < k                       */
        if ( LT( ELM_PLIST(hdSet,j), hdObj ) == HdTrue )  i = j;
        else                                              k = j;
    }

    /* add the element to the set if it is not already there               */
    if ( len < k || EQ( ELM_PLIST(hdSet,k), hdObj ) != HdTrue ) {
        if ( GET_SIZE_BAG(hdSet) < SIZE_PLEN_PLIST( len+1 ) )
            Resize( hdSet, SIZE_PLEN_PLIST( len + len/8 + 4 ) );
        SET_LEN_PLIST( hdSet, len+1 );
        for ( i = len+1; k < i; i-- )
            SET_ELM_PLIST( hdSet, i, ELM_PLIST(hdSet,i-1) );
        SET_ELM_PLIST( hdSet, k, hdObj );
    	if ( GET_TYPE_BAG(hdSet) == T_SET && T_LIST <= GET_TYPE_BAG(hdObj) )
    	    Retype( hdSet, T_LIST );
    }

    /* return nothing, this function is a procedure                        */
    return HdVoid;
}


/****************************************************************************
**
*F  FunRemoveSet(<hdCall>)  . . . . . . . . . .  remove an element from a set
**
**  'FunRemoveSet' implements the internal function 'RemoveSet'.
**
**  'RemoveSet( <set>, <obj> )'
**
**  'RemoveSet' removes <obj>, which may be an object of arbitrary type, from
**  the set <set>, which must be a  proper set.  If  <obj> is in  <set> it is
**  removed and all  entries of <set>  are shifted one position leftwards, so
**  that <set> has no  holes.  If <obj>  is not in  <set>, then <set>  is not
**  changed.  No error is raised in this case.
**
**  'RemoveSet'   does   not return anything,  it   is  only called  for  the
**  sideeffect of changing <set>.
*/
Bag       FunRemoveSet (Bag hdCall)
{
    Bag           hdSet = 0;          /* handle of the set               */
    Bag           hdObj = 0;          /* handle of the object            */
    UInt       len = 0;            /* logical length of the list      */
    UInt       i = 0,  j = 0,  k = 0;      /* loop variables                  */

    /* get and check the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )
        return Error("usage: RemoveSet( <set>, <obj> )",0,0);
    hdSet = EVAL( PTR_BAG(hdCall)[1] );
    hdObj = EVAL( PTR_BAG(hdCall)[2] );
    if ( ! IsSet( hdSet ) )
        return Error("RemoveSet: <set> must be a proper set",0,0);
    if ( hdObj == HdVoid )
        return Error("RemoveSet: <obj> function must return a value",0,0);

    /* perform the binary search to find the position                      */
    len   = LEN_PLIST( hdSet );
    i = 0;  k = len+1;
    while ( i+1 < k ) {                 /* set[i] < elm && elm <= set[k]   */
        j = (i + k) / 2;                /* i < j < k                       */
        if ( LT( ELM_PLIST(hdSet,j), hdObj ) == HdTrue )  i = j;
        else                                              k = j;
    }

    /* remove the element from the set if it is there                      */
    if ( k <= len && EQ( ELM_PLIST(hdSet,k), hdObj ) == HdTrue ) {
        for ( i = k; i < len; i++ )
            SET_ELM_PLIST( hdSet, i, ELM_PLIST(hdSet,i+1) );
        SET_ELM_PLIST( hdSet, len, 0 );
        SET_LEN_PLIST( hdSet, len-1 );
    }

    /* return nothing, this function is a procedure                        */
    return HdVoid;
}


/****************************************************************************
**
*F  FunUniteSet(<hdCall>) . . . . . . . . . . . .  unite one set with another
*V  HdUnion . . . . . . . . . . . . . . . . . . . buffer for the union, local
**
**  'FunUniteSet' implements the internal function 'UniteSet'.
**
**  'UniteSet( <set1>, <set2> )'
**
**  'UniteSet' changes the set <set1> so that it becomes the  union of <set1>
**  and <set2>.  The union is the set of those elements  that are elements of
**  either set.  So 'UniteSet'  adds (see  "AddSet")  all elements to  <set1>
**  that are in <set2>.  <set2> may be a list that  is  not  a proper set, in
**  which case 'Set' is silently applied to it.
**
**  'FunUniteSet' merges <set1> and <set2> into a buffer that is allocated at
**  initialization time.
**
**  'HdUnion' is the handle of the global bag that serves  as  temporary  bag
**  for the union.  It is created in 'InitSet' and is resized when necessary.
*/
Bag       HdUnion;

Bag       FunUniteSet (Bag hdCall)
{
    Bag           hdSet1;         /* handle  of left  set            */
    Bag           hdSet2;         /* handle  of right set            */
    UInt       l1;             /* length  of left  set            */
    UInt       l2;             /* length  of right set            */
    UInt       lr;             /* length  of result set           */
    UInt       i1;             /* index into left  set            */
    UInt       i2;             /* index into right set            */
    UInt                plen;           /* physical length of 'HdUnion'    */

    /* get and check the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error("usage: UniteSet( <set1>, <set2> )",0,0);
    hdSet1 = EVAL( PTR_BAG(hdCall)[1] );
    hdSet2 = EVAL( PTR_BAG(hdCall)[2] );
    if ( ! IsSet( hdSet1 ) )
        return Error("UniteSet: <set1> must be a set",0,0);
    if ( ! IS_LIST(hdSet2) )
        return Error("UniteSet: <set2> must be a list",0,0);
    if ( ! IsSet( hdSet2 ) )  hdSet2 = SetList( hdSet2 );

    /* get the logical lengths and the pointer                             */
    l1 = LEN_PLIST( hdSet1 );
    l2 = LEN_PLIST( hdSet2 );
    if ( GET_SIZE_BAG(HdUnion) < SIZE_PLEN_PLIST( l1+l2 ) ) {
        plen = PLEN_SIZE_PLIST( GET_SIZE_BAG(HdUnion) );
        if ( plen + plen/8 + 4 < l1 + l2 )
            Resize( HdUnion, SIZE_PLEN_PLIST( l1+l2 ) );
        else
            Resize( HdUnion, SIZE_PLEN_PLIST( plen + plen/8 + 4 ) );
    }
    lr = 0;
    i1 = 1;
    i2 = 1;

    /* now merge the two sets into the union                               */
    while ( i1 <= l1 || i2 <= l2 ) {
        if ( i1 <= l1 && i2 <= l2
            && (ELM_PLIST(hdSet1,i1) == ELM_PLIST(hdSet2,i2)
              || EQ(ELM_PLIST(hdSet1,i1),ELM_PLIST(hdSet2,i2)) == HdTrue) ) {
            SET_ELM_PLIST( HdUnion, lr+1, ELM_PLIST(hdSet1,i1) );
            lr++; i1++;  i2++;
        }
        else if ( i2 == l2 + 1
               || (i1<=l1
                && LT(ELM_PLIST(hdSet1,i1),ELM_PLIST(hdSet2,i2))==HdTrue) ) {
            SET_ELM_PLIST( HdUnion, lr+1, ELM_PLIST(hdSet1,i1) );
            lr++; i1++;
        }
        else {
            SET_ELM_PLIST( HdUnion, lr+1, ELM_PLIST(hdSet2,i2) );
            lr++; i2++;
        }
    }

    /* resize the result and copy back from the union                      */
    if ( GET_SIZE_BAG(hdSet1) < SIZE_PLEN_PLIST( lr ) ) {
        plen = PLEN_SIZE_PLIST( GET_SIZE_BAG(hdSet1) );
        if ( plen + plen/8 + 4 < lr )
            Resize( hdSet1, SIZE_PLEN_PLIST( lr ) );
        else
            Resize( hdSet1, SIZE_PLEN_PLIST( plen + plen/8 + 4 ) );
    }
    for ( i1 = 1; i1 <= lr; i1++ ) {
        SET_ELM_PLIST( hdSet1, i1, ELM_PLIST( HdUnion, i1 ) );
        SET_ELM_PLIST( HdUnion, i1, 0 );
    }
    SET_LEN_PLIST( hdSet1, lr );

    /* return nothing, this function is a procedure                        */
    return HdVoid;
}


/****************************************************************************
**
*F  FunIntersectSet(<hdCall>) . . . . . . . .  intersect one set with another
**
**  'FunIntersectSet' implements the internal function 'IntersectSet'.
**
**  'IntersectSet( <set1>, <set2> )'
**
**  'IntersectSet' changes the set <set1> so that it becomes the intersection
**  of <set1> and <set2>.  The intersection is the set of those elements that
**  are  elements in both sets.   So 'IntersectSet' removes (see "RemoveSet")
**  all elements from <set1> that are not  in  <set2>.  <set2> may be a  list
**  that is not a proper set, in which case 'Set' is silently applied to it.
*/
Bag       FunIntersectSet (Bag hdCall)
{
    Bag           hdSet1;         /* handle  of left  set            */
    Bag           hdSet2;         /* handle  of right set            */
    UInt       l1;             /* length  of left  set            */
    UInt       l2;             /* length  of right set            */
    UInt       lr;             /* length  of result set           */
    UInt       i1;             /* index into left  set            */
    UInt       i2;             /* index into right set            */

    /* get and check the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error("usage: IntersectSet( <set1>, <set2> )",0,0);
    hdSet1 = EVAL( PTR_BAG(hdCall)[1] );
    hdSet2 = EVAL( PTR_BAG(hdCall)[2] );
    if ( ! IsSet( hdSet1 ) )
        return Error("IntersectSet: <set1> must be a set",0,0);
    if ( ! IS_LIST(hdSet2) )
        return Error("IntersectSet: <set2> must be a list",0,0);
    if ( ! IsSet( hdSet2 ) )  hdSet2 = SetList( hdSet2 );

    /* get the logical lengths and the pointer                             */
    l1 = LEN_PLIST( hdSet1 );
    l2 = LEN_PLIST( hdSet2 );
    lr = 0;
    i1 = 1;
    i2 = 1;

    /* now merge the two sets into the intersection                        */
    while ( i1 <= l1 && i2 <= l2 ) {
        if ( ELM_PLIST(hdSet1,i1) == ELM_PLIST(hdSet2,i2)
          || EQ( ELM_PLIST(hdSet1,i1), ELM_PLIST(hdSet2,i2) ) == HdTrue ) {
            SET_ELM_PLIST( hdSet1, lr+1, ELM_PLIST(hdSet1,i1) );
            lr++; i1++;  i2++;
        }
        else if ( LT(ELM_PLIST(hdSet1,i1),ELM_PLIST(hdSet2,i2)) == HdTrue ) {
            i1++;
        }
        else {
            i2++;
        }
    }

    /* resize the result or clear the rest of the bag                      */
    SET_LEN_PLIST( hdSet1, lr );
    if ( SIZE_PLEN_PLIST( lr + lr/8 + 4 ) < GET_SIZE_BAG(hdSet1) ) {
        Resize( hdSet1, SIZE_PLEN_PLIST( lr ) );
    }
    else {
        while ( lr < GET_SIZE_BAG(hdSet1)/SIZE_HD-1 ) {
            SET_ELM_PLIST( hdSet1, lr+1, 0 );
            lr++;
        }
    }

    /* return nothing, this function is a procedure                        */
    return HdVoid;
}


/****************************************************************************
**
*F  FunSubstractSet(<hdCall>) . . . . . . . . . subtract one set from another
**
**  'FunSubtractSet' implements the internal function 'SubstractSet'.
**
**  'SubstractSet( <set1>, <set2> )'
**
**  'SubstractSet' changes the  set <set1> so  that it becomes the difference
**  of <set1> and <set2>.  The difference is the set of the elements that are
**  in <set1> but not in <set2>.  So 'SubtractSet' removes  (see "RemoveSet")
**  all elements from <set1> that are in <set2>.   <set2> may  be a list that
**  is not a proper set, in which case 'Set' is silently applied to it.
*/
Bag       FunSubtractSet (Bag hdCall)
{
    Bag           hdSet1;         /* handle  of left  set            */
    Bag           hdSet2;         /* handle  of right set            */
    UInt       l1;             /* length  of left  set            */
    UInt       l2;             /* length  of right set            */
    UInt       lr;             /* length  of result set           */
    UInt       i1;             /* index into left  set            */
    UInt       i2;             /* index into right set            */

    /* get and check the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error("usage: SubtractSet( <set1>, <set2> )",0,0);
    hdSet1 = EVAL( PTR_BAG(hdCall)[1] );
    hdSet2 = EVAL( PTR_BAG(hdCall)[2] );
    if ( ! IsSet( hdSet1 ) )
        return Error("SubtractSet: <set1> must be a set",0,0);
    if ( ! IS_LIST(hdSet2) )
        return Error("SubtractSet: <set2> must be a list",0,0);
    if ( ! IsSet( hdSet2 ) )  hdSet2 = SetList( hdSet2 );

    /* get the logical lengths and the pointer                             */
    l1 = LEN_PLIST( hdSet1 );
    l2 = LEN_PLIST( hdSet2 );
    lr = 0;
    i1 = 1;
    i2 = 1;

    /* now merge the two sets into the difference                          */
    while ( i1 <= l1 ) {
        if ( i2 <= l2
          && (ELM_PLIST(hdSet1,i1) == ELM_PLIST(hdSet2,i2)
           || EQ( ELM_PLIST(hdSet1,i1), ELM_PLIST(hdSet2,i2) ) == HdTrue) ) {
            i1++;  i2++;
        }
        else if ( i2 == l2+1
               || LT(ELM_PLIST(hdSet1,i1),ELM_PLIST(hdSet2,i2)) == HdTrue ) {
            SET_ELM_PLIST( hdSet1, lr+1, ELM_PLIST(hdSet1,i1) );
            lr++; i1++;
        }
        else {
            i2++;
        }
    }

    /* resize the result or clear the rest of the bag                      */
    SET_LEN_PLIST( hdSet1, lr );
    if ( SIZE_PLEN_PLIST( lr + lr/8 + 4 ) < GET_SIZE_BAG(hdSet1) ) {
        Resize( hdSet1, SIZE_PLEN_PLIST( lr ) );
    }
    else {
        while ( lr < GET_SIZE_BAG(hdSet1)/SIZE_HD-1 ) {
            SET_ELM_PLIST( hdSet1, lr+1, 0 );
            lr++;
        }
    }

    /* return nothing, this function is a procedure                        */
    return HdVoid;
}


void PrSet(Obj s) { Pr("Set(%3>",0,0); PrList(s); Pr("%3<)", 0, 0); }

/****************************************************************************
**
*F  InitSet() . . . . . . . . . . . . . . . . . .  initialize the set package
**
**  'InitSet' initializes the set package.
*/
void            InitSet (void)
{

    /* install the list functions in the tables                            */
    TabIsList[T_SET]      = 1;
    TabLenList[T_SET]     = LenSet;
    TabElmList[T_SET]     = ElmSet;
    TabElmfList[T_SET]    = ElmfSet;
    TabElmlList[T_SET]    = ElmfSet;
    TabElmrList[T_SET]    = ElmfSet;
    TabElmsList[T_SET]    = ElmsSet;
    TabAssList[T_SET]     = AssSet;
    TabAsssList[T_SET]    = AsssSet;
    TabPosList[T_SET]     = PosSet;
    TabPlainList[T_SET]   = PlainSet;
    TabIsDenseList[T_SET] = IsDenseSet;
    TabIsPossList[T_SET]  = IsPossSet;
    EvTab[T_SET]          = EvList;
    PrTab[T_SET]          = PrSet;
    TabEq[T_SET][T_SET]   = EqSet;
    TabLt[T_SET][T_SET]   = LtSet;

    /* install internal functions                                          */
    InstIntFunc( "Set",          FunSet          );
    InstIntFunc( "IsSet",        FunIsSet        );
    InstIntFunc( "IsEqualSet",   FunIsEqualSet   );
    InstIntFunc( "IsSubsetSet",  FunIsSubsetSet  );
    InstIntFunc( "AddSet",       FunAddSet       );
    InstIntFunc( "RemoveSet",    FunRemoveSet    );
    InstIntFunc( "UniteSet",     FunUniteSet     );
    InstIntFunc( "IntersectSet", FunIntersectSet );
    InstIntFunc( "SubtractSet",  FunSubtractSet  );

    /* create the temporary union bag                                      */
    HdUnion = NewBag( T_SET, SIZE_PLEN_PLIST( 1024 ) );
    SET_LEN_PLIST( HdUnion, 0 );

}
