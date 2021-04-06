/****************************************************************************
**
*A  list.c                      GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file defines  the  functions and  macros  that  make it  possible to
**  access all various types of lists in a homogenous way.
**
**  This package serves  two purposes.  First it provides a uniform interface
**  to the  functions that access  lists  and their elements  for  the  other
**  packages in  the  GAP  kernel.  For example, 'EvFor'  can  loop  over the
**  elements in a  list with the macros 'LEN_LIST' and 'ELM_LIST' independent
**  of the type  of  the list.  Second it implements  the functions that make
**  the  list  functionality  available  to  the  GAP  evaluator,  e.g.,   it
**  implements 'EvElmList' and 'EvIn'.
**
**  This package uses plain  lists (of  type 'T_LIST') and assumes that it is
**  o.k.  to stuff  objects of any type into plain lists.  It uses the macros
**  'LEN_PLIST', 'SET_LEN_PLIST',  'ELM_PLIST', and 'SET_ELM_PLIST'  exported
**  by the plain list package to access and modify plain lists.
**
**  Also  it assumes that  ranges  (of type 'T_RANGE') are dense,  consist of
**  small integers only, that the first  element is  the smallest element and
**  the last element is the  largest, or vice versa, and that the differences
**  between adjacent  elements in  a range are all equal.  It uses the macros
**  'LEN_RANGE',  'LOW_RANGE', and 'INC_RANGE' exported by  the range package
**  (and thus does not rely on a specific representation of ranges).
**
*/

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "scanner.h"             /* reading of tokens and printing  */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */

#include        "plist.h"               /* 'LEN_PLIST', 'SET_LEN_PLIST',.. */

#include        "set.h"                 /* 'IsSet'                         */
#include        "permutat.h"            /* 'OnTuplesPerm', 'OnSetsPerm'    */
#include        "record.h"              /* 'InRec'                         */

#include        "list.h"                /* declaration part of the package */
#include        "namespaces.h"
#include        "idents.h"
#include        "gstring.h"             /* IsString */
#include        "args.h"                /* NewList */

/****************************************************************************
**
*F  IS_LIST(<hdList>) . . . . . . . . . . . . . . . . . . is an object a list
*V  TabIsList[<type>] . . . . . . . . . . . . . . . . . . table for list test
**
**  'IS_LIST' returns 1 if the object <hdList> is a list and 0 otherwise.
**
**  Note that 'IS_LIST' is a macro, so  do not  call  it  with arguments that
**  have sideeffects.
**
**  'IS_LIST' simply returns the value in 'TabIsList'.
**
**  A package implementing a list type must set this value to 1.  If the list
**  type  is  actually  a vector  type, then  it  must be set to 2,  and  the
**  corresponding entry for the extended matrix type must be set to 3.
**
**  'IS_LIST' is defined in the declaration part of this package as follows:
**
#define IS_LIST(LIST)           (TabIsList[GET_TYPE_BAG(LIST)] != 0)
*/
Int            TabIsList [LIST_TAB_SIZE];


/****************************************************************************
**
*F  LEN_LIST(<hdList>)  . . . . . . . . . . . . . . . . . .  length of a list
*V  TabLenList[<type>]  . . . . . . . . . . . . . . table of length functions
*F  CantLenList(<hdObj>)  . . . . . . . . . . . . . . . error length function
**
**  'LEN_LIST' returns    the logical length  of   the list  <hdList> as  a C
**  integer.  An error is signalled if <hdList> is not a list.
**
**  Note that  'LEN_LIST' is a  macro, so do  not call it with arguments that
**  have sideeffects.
**
**  'LEN_LIST' only calls the function  pointed  to by  'TabLenList[<type>]',
**  passing <hdList> as argument.  If <type> is not the type of a list,  then
**  'TabLenList[<type>]'  points  to  'CantLenList',  which  just signals  an
**  error.
**
**  A  package  implementing a list type  must  provide such  a  function and
**  install it in 'TabLenList'.
**
**  'LEN_LIST' is defined in the declaration part of this package as follows:
**
#define LEN_LIST(LIST)          ((*TabLenList[GET_TYPE_BAG(LIST)])(LIST))
*/
Int            (*TabLenList[LIST_TAB_SIZE]) ( Bag );

Int            CantLenList (Bag hdList)
{
    return HD_TO_INT( Error("Length: <list> must be a list",0,0) );
}


/****************************************************************************
**
*F  ELM_LIST(<hdList>,<pos>)  . . . . . . . . . select an element from a list
*V  TabElmList[<type>]  . . . . . . . . . . . .  table of selection functions
*F  ELMF_LIST(<hdList>,<pos>) . . .  select an element from a list w/o checks
*V  TabElmfList[<type>] . . . . . . . . . . table of fast selection functions
*F  ELML_LIST(<hdList>,<pos>) . . . . . . . . . select an element from a list
*V  TabElmlList[<type>] . . . . . . . . . . . .  table of selection functions
*F  ELMR_LIST(<hdList>,<pos>) . . . . . . . . . select an element from a list
*V  TabElmrList[<type>] . . . . . . . . . . . .  table of selection functions
*F  CantElmList(<hdList>,<pos>) . . . . . . . . . .  error selection function
**
**  'ELM_LIST' returns the handle of  the  element at position  <pos> in  the
**  list  <hdList>.  It is the  responsibility  of the caller to ensure  that
**  <pos>  is  positive.  An error is signalled if <hdList> is not a list, if
**  <hdList> has no assigned  value at position <pos>, or if  <pos> is larger
**  than 'LEN_LIST(<hdList>)'.
**
**  Note that 'ELM_LIST' is a macro, so  do not  call it  with arguments that
**  have sideeffects.
**
**  'ELMF_LIST'  does  the  same  thing  as  'ELM_LIST',  but  the  functions
**  implementing this do not need to  check that <pos>  is less than or equal
**  to 'LEN_LIST(<hdList>)'.  Also 'ELMF_LIST' returns 0  if <hdList>  has no
**  assigned value  at the position  <pos>.  It is thus the responsibility of
**  the caller to ensure that <pos> is  positive and  less  than  or equal to
**  'LEN_LIST(<hdList>)'.
**
**  'ELML_LIST'  does  the  same  thing  as  'ELMF_LIST', but  the  functions
**  implementing this are  allowed to  select the  elements into a fixed bag,
**  that will be overwritten by the next call to 'ELML_LIST'.
**
**  'ELMR_LIST' does the same thing as 'ELML_LIST' but uses a different fixed
**  bag than 'ELML_LIST'.
**
**  'ELML_LIST' and 'ELMR_LIST' are used in the generic comparison functions.
**  There the elements  selected  are only  compared and  can  be overwritten
**  afterwards.  This makes it possible to implement the comparison functions
**  in such a way that they do not allocate new memory.
**
*N  1992/12/08 martin this is critical if the comparison recurses
**
**  'ELM_LIST' only  calls  the  function pointed to by 'TabElmList[<type>]',
**  passing <hdList> and <pos> as arguments.  If <type> is not the type of  a
**  list,  then  'TabElmList[<type>]'  points  to  'CantElmList',  which just
**  signals an error.
**
**  A  package implementing  a  list  type must  provide  such functions  and
**  install   them  in   'TabElmList',  'TabElmfList',   'TabElmlList',   and
**  'TabElmrList'.
**
**  'ELM_LIST' etc.  are  defined in the declaration part of this  package as
**  follows:
**
#define ELM_LIST(LIST,POS)      ((*TabElmList[GET_TYPE_BAG(LIST)])(LIST,POS))
#define ELMF_LIST(LIST,POS)     ((*TabElmfList[GET_TYPE_BAG(LIST)])(LIST,POS))
#define ELML_LIST(LIST,POS)     ((*TabElmlList[GET_TYPE_BAG(LIST)])(LIST,POS))
#define ELMR_LIST(LIST,POS)     ((*TabElmrList[GET_TYPE_BAG(LIST)])(LIST,POS))
*/
Bag       (*TabElmList[LIST_TAB_SIZE])  ( Bag, Int );
Bag       (*TabElmfList[LIST_TAB_SIZE]) ( Bag, Int );
Bag       (*TabElmlList[LIST_TAB_SIZE]) ( Bag, Int );
Bag       (*TabElmrList[LIST_TAB_SIZE]) ( Bag, Int );

Bag       CantElmList (Bag hdList, Int pos)
{
    return Error("List Element: <list> must be a list",0,0);
}


/****************************************************************************
**
*F  ELMS_LIST(<hdList>,<hdPoss>)  . . . . . . .  select a sublist from a list
*V  TabElmsList[<type>] . . . . . . . .  table of sublist selection functions
*F  CantElmsList(<hdList>,<hdPoss>) . . . .  error sublist selection function
**
**  'ELMS_LIST' returns a new list containing  the  elements at the positions
**  given  in  the  list  <hdPoss>  from   the  list  <hdList>.   It  is  the
**  responsibility  of  the  caller  to ensure  that  <hdPoss> is  dense  and
**  contains only positive  integers.  An error  is signalled if <hdList>  is
**  not a list, if  <hdList> has no assigned value at any of the positions in
**  <hdPoss>,   or   if   an    element   of   <hdPoss>   is    larger   than
**  'LEN_LIST(<hdList>)'.
**
**  Note that 'ELMS_LIST' is a  macro, so do not call it with arguments  that
**  have sideeffects.
**
**  'ELMS_LIST' only calls the function  pointed to by 'TabElmsList[<type>]',
**  passing <hdList> and <hdPoss> as arguments.  If <type> is not the type of
**  a list, then  'TabElmsList[<type>]'  points to 'CantElmsList', which just
**  signals an error.
**
**  Note  that functions  implementing 'ELMS_LIST' *must* create a  new list,
**  even  if <positions>  is  equal  to  '[1..Length(<list>)]'.  'EvElmsList'
**  depends on this.
**
**  'ELMS_LIST'  is defined in  the  declaration  part  of  this  package  as
**  follows:
**
#define ELMS_LIST(LIST,POSS)    ((*TabElmsList[GET_TYPE_BAG(LIST)])(LIST,POSS))
*/
Bag       (*TabElmsList[LIST_TAB_SIZE]) ( Bag, Bag );

Bag       CantElmsList (Bag hdList, Bag hdPoss)
{
    return Error("List Elements: <list> must be a list",0,0);
}


/****************************************************************************
**
*F  ASS_LIST(<hdList>,<pos>,<hdVal>)  . . . . . . . . . . .  assign to a list
*V  TabAssList[<type>]  . . . . . . . . . . . . table of assignment functions
*F  CantAssList(<hdList>,<pos>,<hdVal>) . . . . . . error assignment function
**
**  'ASS_LIST'  assigns the  object <hdVal>  to  the  list  <hdList>  at  the
**  position <pos>.  Note that the assignment may change the type of the list
**  <hdList>.  It is the responsibility of the caller to ensure that <pos> is
**  positive, and that  <hdVal>  is not  'HdVoid'.  An error is  signalled if
**  <hdList> is not a list.
**
**  Note that 'ASS_LIST' is a macro,  so do not  call it with arguments  that
**  have sideeffects.
**
**  'ASS_LIST' only  calls the function pointed  to  by 'TabAssList[<type>]',
**  passing <hdList>, <pos>, and  <hdVal> as arguments.  If <type> is not the
**  type of a list,  then 'TabAssList[<type>]' points to 'CantAssList', which
**  just signals an error.
**
**  'ASS_LIST' is defined in the declaration part of this package as follows.
**
#define ASS_LIST(LIST,POS,VAL)  ((*TabAssList[GET_TYPE_BAG(LIST)])(LIST,POS,VAL))
*/
Bag       (*TabAssList[LIST_TAB_SIZE]) ( Bag, Int, Bag );

Bag       CantAssList (Bag hdList, Int pos, Bag hdVal)
{
    return Error("List Assignment: <list> must be a list",0,0);
}


/****************************************************************************
**
*F  ASSS_LIST(<hdList>,<hdPoss>,<hdVals>) . assign several elements to a list
*V  TabAsssList[<type>] . . . . . . . . . . . .  table of assignment function
*F  CantAsssList(<hdList>,<hdPoss>,<hdVals>)  . . . error assignment function
**
**  'ASSS_LIST' assignes the values  from the  list <hdVals> at the positions
**  given   in  the   list   <hdPoss>  to  the  list  <hdList>.   It  is  the
**  responsibility of  the  caller  to  ensure that  <hdPoss>  is  dense  and
**  contains only positive integers, that <hdPoss> and <hdVals> have the same
**  length, and that <hdVals> is dense.  An error is signalled if <hdList> is
**  not a list.
**
**  Note that 'ASSS_LIST' is a macro, so do not call it with  arguments  that
**  have sideeffects.
**
**  'ASSS_LIST' only calls  the function pointed to by 'TabAsssList[<type>]',
**  passing  <hdList>, <hdPoss>, and <hdVals> as arguments.  If <type> is not
**  the type of a list, then 'TabAsssList[<type>]'  points to 'CantAsssList',
**  which just signals an error.
**
**  'ASSS_LIST'  is  defined in the  declaration  part  of  this  package  as
**  follows:
**
#define ASSS_LIST(LI,POSS,VALS) ((*TabAsssList[GET_TYPE_BAG(LI)])(LI,POSS,VALS))
*/
Bag       (*TabAsssList[LIST_TAB_SIZE]) ( Bag, Bag, Bag );

Bag       CantAsssList (Bag hdList, Bag hdPoss, Bag hdVals)
{
    return Error("List Assignment: <list> must be a list",0,0);
}


/****************************************************************************
**
*F  POS_LIST(<hdList>,<hdVal>,<start>)  . . . . . . find an element in a list
*V  TabPosList[<type>]  . . . . . . . . . . . .  table of searching functions
*F  CantPosList(<hdList>,<hdVal>,<start>) . . . . .  error searching function
**
**  'POS_LIST'  returns the  position of  the first  occurence  of the  value
**  <hdVal>, which may be an object  of arbitrary type,  in the list <hdList>
**  after the position <start>  as a C  integer.  0 is returned if <hdVal> is
**  not in the list.  An error is signalled if <hdList> is not a list.
**
**  Note that 'POS_LIST'  is a macro,  so do  not call it with arguments that
**  have sideeffects.
**
**  'POS_LIST'  only calls  the function pointed to  by 'TabPosList[<type>]',
**  passing  <hdList>, <hdVal>,  and <start> as  arguments.  If <type> is not
**  the  type of  a list,  then 'TabPosList[<type>]' points to 'CantPosList',
**  which just signals an error.
**
**  'POS_LIST' is defined in the declaration part of this package as follows:
**
#define POS_LIST(LIST,VAL,START)  ((*TabPosList[GET_TYPE_BAG(LIST)])(LIST,VAL,START))
*/
Int            (*TabPosList[LIST_TAB_SIZE]) ( Bag, Bag, Int );

Int            CantPosList (Bag hdList, Bag hdVal, Int start)
{
    return HD_TO_INT( Error("Position: <list> must be a list",0,0) );
}


/****************************************************************************
**
*F  PLAIN_LIST(<hdList>)  . . . . . . . . . .  convert a list to a plain list
*V  TabPlainList[<type>]  . . . . . . . . . . . table of conversion functions
*F  CantPlainList(<hdList>) . . . . . . . . . . . . error conversion function
**
**  'PLAIN_LIST' converts the list <hdList>  to a plain list,  e.g., one that
**  has the same  representation as lists of type 'T_LIST', *in place*.  Note
**  that the type of <hdList> need not be  'T_LIST' afterwards, it could also
**  be 'T_SET' or 'T_VECTOR'.  An error is  signalled  if  <hdList> is  not a
**  list.
**
**  Note that 'PLAIN_LIST' is a macro, so do not call  it with arguments that
**  have sideeffects.
**
**  'PLAIN_LIST'    only    calls     the    function    pointed     to    by
**  'TabPlainList[<type>]',  passing <hdList> as argument.  If <type>  is not
**  the   type   of   a   list,   then   'TabPlainList[<type>]'   points   to
**  'CantPlainList', which just signals an error.
**
**  'PLAIN_LIST'  is defined in  the  declaration  part  of  this  package as
**  follows:
**
#define PLAIN_LIST(LIST)          ((*TabPlainList[GET_TYPE_BAG(LIST)])(LIST))
*/
void            (*TabPlainList[LIST_TAB_SIZE]) ( Bag );

void            CantPlainList (Bag hdList)
{
    Error("Panic: cannot convert <list> to a plain list",0,0);
}

/*N 1992/12/11 martin only here for backward compatibility                 */
Int            IsList (Bag hdObj)
{
    if ( ! IS_LIST( hdObj ) ) {
        return 0;
    }
    else {
        PLAIN_LIST( hdObj );
        return 1;
    }
}


/****************************************************************************
**
*F  IS_DENSE_LIST(<hdList>) . . . . . . . . . . . . . .  test for dense lists
*V  TabIsDenseList[<type>]  . . . . . . . table for dense list test functions
*F  NotIsDenseList(<hdObj>) . . . . .  dense list test function for non lists
**
**  'IS_DENSE_LIST' returns 1 if the  list  <hdList>  is a  dense list  and 0
**  otherwise, i.e., if either <hdList> is not a list, or if it is not dense.
**
**  'IS_DENSE_LIST'    only    calls    the    function    pointed    to   by
**  'TabIsDenseList[<type>]', passing <hdList> as argument.  If <type> is not
**  the   type  of  a   list,   then   'TabIsDenseList[<type>]'   points   to
**  'NotIsDenseList', which just returns 0.
**
**  'IS_DENSE_LIST'  is defined in the declaration  part  of this  package as
**  follows:
**
#define IS_DENSE_LIST(LIST)     ((*TabIsDenseList[GET_TYPE_BAG(LIST)])(LIST))
*/
Int            (*TabIsDenseList[LIST_TAB_SIZE]) ( Bag );

Int            NotIsDenseList (Bag hdObj)
{
    return 0;
}


/****************************************************************************
**
*F  IS_POSS_LIST(<hdList>)  . . . . . . . . . . . .  test for positions lists
*V  TabIsPossList[<type>] . . . . . . . table of positions list test function
*F  NotIsPossList(<hdObj>)  . . .  positions list test function for non lists
**
**  'IS_POSS_LIST' returns 1 if the list <hdList> is  a dense list containing
**  only positive integers  and 0 otherwise, i.e., if either <hdList> is  not
**  a list, or if it is not dense, or if it contains an element that is not a
**  positive integer.
**
**  'IS_POSS_LIST'    only    calls    the    function    pointed    to    by
**  'TabIsPossList[<type>]', passing <hdList> as argument.  If  <type> is not
**  the   type   of   a   list,   then  'TabIsPossList[<type>]'   points   to
**  'NotIsPossList', which just returns 0.
**
**  'IS_POSS_LIST' is  defined  in the  declaration  part of this  package as
**  follows:
**
#define IS_POSS_LIST(LIST)      ((*TabIsPossList[GET_TYPE_BAG(LIST)])(LIST))
*/
Int            (*TabIsPossList[LIST_TAB_SIZE]) ( Bag );

Int            NotIsPossList (Bag hdObj)
{
    return 0;
}


/****************************************************************************
**
*F  XType(<hdObj>)  . . . . . . . . . . . . . . .  extended type of an object
*F  IS_XTYPE_LIST(<type>,<hdList>)  . . . . . . . . .  test for extended type
*V  TabIsXTypeList[<type>]  . . . . . . table of extended type test functions
**
**  'XType' returns the extended type of the object <hdObj>.  For  everything
**  except objects of type 'T_LIST' and 'T_SET' this is just the type of this
**  object.  For objects of type  'T_LIST'  and 'T_SET' 'XType' examines  the
**  objects closer and returns 'T_VECTOR', 'T_VECFFE', 'T_BLIST', 'T_STRING',
**  'T_RANGE', 'T_MATRIX',  'T_MATFFE', and 'T_LISTX'.  As  a  sideeffect the
**  object <hdObj> is converted into the representation of the extended type,
**  e.g.,  if 'XType' returns 'T_MATFFE', <hdObj> is converted into a list of
**  vectors  over a common  finite  field.   'XType' is used  by  the  binary
**  operations functions for  lists to decide  to  which function they should
**  dispatch.  'T_LISTX' is  the extended type  of otherwise untypable lists.
**  The only operation defined for  such lists is  the product  with a scalar
**  (where 'PROD( <list>[<pos>], <scl> )' decides  whether the multiplication
**  is allowed or not).
**
**  'XType' calls 'IS_XTYPE_LIST(<type>,<hdList>)' with  <type>  running from
**  'T_VECTOR'  to 'T_MATFFE'  and returns  the first  type  for  which  this
**  function returns 1.   If no one returns 1, then 'XType' returns 'T_LISTX'
**  (and leaves the list as 'T_LIST' or 'T_SET').
**
**  'IS_XTYPE_LIST' is defined in  the declaration  part  of this package  as
**  follows:
**
#define IS_XTYPE_LIST(T,LIST)   ((*TabIsXTypeList[T])(LIST))
*/
Int            (*TabIsXTypeList[LIST_TAB_SIZE]) ( Bag );

Int            XType (Bag hdObj)
{
    Int                t;              /* loop variable                   */

    /* first handle non lists                                              */
    if ( GET_TYPE_BAG(hdObj) < T_LIST || GET_TYPE_BAG(hdObj) == T_REC )
        return GET_TYPE_BAG(hdObj);

    /* otherwise try the extended types in turn                            */
    /* this is done backwards to catch the more specific types first       */
    for ( t = LIST_TAB_SIZE-1; T_LIST <= t; t-- ) {
        if ( TabIsXTypeList[t] != 0 && IS_XTYPE_LIST( t, hdObj ) )
            return t;
    }

    /* nothing works, return 'T_LISTX'                                     */
    return T_LISTX;
}


/****************************************************************************
**
*F  EvList(<hdList>)  . . . . . . . . . . . . . . . . . . . . evaluate a list
**
**  'EvList' returns the value of the list <hdList>.  The  value of a list is
**  just the list itself, since lists are constants and thus selfevaluating.
*/
Bag       EvList (Bag hdList)
{
    return hdList;
}


/****************************************************************************
**
*F  PrList(<hdList>)  . . . . . . . . . . . . . . . . . . . . .  print a list
**
**  'PrList' prints the list <hdList>.  It is a generic function that can  be
**  used for all kinds of lists.
**
**  Linebreaks are preferred after the commas.
*/
void            PrList (Bag hdList)
{
    Int                lenList;        /* logical length of <list>        */
    Bag           hdElm;          /* one element from <list>         */
    Int                i;              /* loop variable                   */

    /* get the logical length of the list                                  */
    lenList = LEN_LIST( hdList );

    /* loop over the entries                                               */
    Pr("%2>[ %2>",0,0);
    for ( i = 1;  i <= lenList;  i++ ) {
        hdElm = ELMF_LIST( hdList, i );
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
*F  EqList(<hdL>,<hdR>) . . . . . . . . . . . . . test if two lists are equal
**
**  'EqList' returns 'true' if the two lists <hdL> and <hdR>  are  equal  and
**  'false' otherwise.  This is a generic function that works for  all  kinds
**  of lists.
**
**  Is called from the 'EQ' binop so both  operands  are  already  evaluated.
*/
Bag       EqList (Bag hdL, Bag hdR)
{
    Int                lenL;           /* length of the left operand      */
    Int                lenR;           /* length of the right operand     */
    Bag           hdElmL;         /* element of the left operand     */
    Bag           hdElmR;         /* element of the right operand    */
    Int                i;              /* loop variable                   */

    /* get the lengths of the lists and compare them                       */
    lenL = LEN_LIST( hdL );
    lenR = LEN_LIST( hdR );
    if ( lenL != lenR ) {
        return HdFalse;
    }

    /* loop over the elements and compare them                             */
    for ( i = 1; i <= lenL; i++ ) {
        hdElmL = ELML_LIST( hdL, i );
        hdElmR = ELMR_LIST( hdR, i );
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
*F  LtList(<hdL>,<hdR>) . . . . . . . . . . . . . test if two lists are equal
**
**  'LtList' returns 'true' if the list <hdL> is less than the list <hdR> and
**  'false' otherwise.
**
**  Is called from the 'LT' binop so both operands are already evaluated.
*/
Bag       LtList (Bag hdL, Bag hdR)
{
    Int                lenL;           /* length of the left operand      */
    Int                lenR;           /* length of the right operand     */
    Bag           hdElmL;         /* element of the left operand     */
    Bag           hdElmR;         /* element of the right operand    */
    Int                i;              /* loop variable                   */

    /* get the lengths of the lists and compare them                       */
    lenL = LEN_LIST( hdL );
    lenR = LEN_LIST( hdR );

    /* loop over the elements and compare them                             */
    for ( i = 1; i <= lenL && i <= lenR; i++ ) {
        hdElmL = ELML_LIST( hdL, i );
        hdElmR = ELMR_LIST( hdR, i );
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
*F  SumList(<hdL>,<hdR>)  . . . . . . . . . . . . . . . . . . .  sum of lists
*F  SumSclList(<hdL>,<hdR>) . . . . . . . . . . .  sum of a scalar and a list
*F  SumListScl(<hdL>,<hdR>) . . . . . . . . . . .  sum of a list and a scalar
*F  SumListList(<hdL>,<hdR>)  . . . . . . . . . . . . . . .  sum of two lists
**
**  'SumList' is the generic  dispatcher for the sums involving  lists.  That
**  is, whenever two lists are added and 'TabSum' does not point to a special
**  routine  then 'SumList' is  called.   'SumList'  determines  the extended
**  types of  the  arguments  (e.g.,  including  'T_MATRIX', 'T_MATFFE',  and
**  'T_LISTX') and then dispatches through 'TabSum' again.
**
**  'SumSclList' is a generic function for the first kind  of sum,  that of a
**  scalar with a list.
**
**  'SumListScl' is a generic function for the second kind of sum, that  of a
**  list  and a scalar.
**
**  'SumListList'  is a generic function  for  the third kind of sum, that of
**  two lists.
*/
Bag       SumList (Bag hdL, Bag hdR)
{
    return (*TabSum[XType(hdL)][XType(hdR)])( hdL, hdR );
}

Bag       SumSclList (Bag hdL, Bag hdR)
{
    Bag           hdS;            /* handle of the sum               */
    Bag           hdSS;           /* one element of sum list         */
    Bag           hdRR;           /* one element of right operand    */
    Int                len;            /* length                          */
    Int                i;              /* loop variable                   */

    /* make the result list                                                */
    len = LEN_LIST( hdR );
    hdS = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
    SET_LEN_PLIST( hdS, len );

    /* loop over the entries and add                                       */
    for ( i = 1; i <= len; i++ ) {
        hdRR = ELMR_LIST( hdR, i );
        hdSS = SUM( hdL, hdRR );
        SET_ELM_PLIST( hdS, i, hdSS );
    }

    /* return the result                                                   */
    return hdS;
}

Bag       SumListScl (Bag hdL, Bag hdR)
{
    Bag           hdS;            /* handle of the sum               */
    Bag           hdSS;           /* one element of sum list         */
    Bag           hdLL;           /* one element of left operand     */
    Int                len;            /* length                          */
    Int                i;              /* loop variable                   */

    /* make the result list                                                */
    len = LEN_LIST( hdL );
    hdS = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
    SET_LEN_PLIST( hdS, len );

    /* loop over the entries and add                                       */
    for ( i = 1; i <= len; i++ ) {
        hdLL = ELML_LIST( hdL, i );
        hdSS = SUM( hdLL, hdR );
        SET_ELM_PLIST( hdS, i, hdSS );
    }

    /* return the result                                                   */
    return hdS;
}

Bag       SumListList (Bag hdL, Bag hdR)
{
    Bag           hdS;            /* handle of the sum               */
    Bag           hdSS;           /* one element of the sum          */
    Bag           hdLL;           /* one element of the left list    */
    Bag           hdRR;           /* one element of the right list   */
    Int                len;            /* length                          */
    Int                i;              /* loop variable                   */

    /* get and check the length                                            */
    len = LEN_LIST( hdL );
    if ( len != LEN_LIST( hdR ) ) {
        return Error(
          "Vector +: lists must have the same length",
                     0, 0 );
    }
    hdS = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
    SET_LEN_PLIST( hdS, len );

    /* loop over the entries and add                                       */
    for ( i = 1; i <= len; i++ ) {
        hdLL = ELML_LIST( hdL, i );
        hdRR = ELMR_LIST( hdR, i );
        hdSS = SUM( hdLL, hdRR );
        SET_ELM_PLIST( hdS, i, hdSS );
    }

    /* return the result                                                   */
    return hdS;
}


/****************************************************************************
**
*F  DiffList(<hdL>,<hdR>) . . . . . . . . . . . . . . . . difference of lists
*F  DiffSclList(<hdL>,<hdR>)  . . . . . . . difference of a scalar and a list
*F  DiffListScl(<hdL>,<hdR>)  . . . . . . . difference of a list and a scalar
*F  DiffListList(<hdL>,<hdR>) . . . . . . . . . . . . difference of two lists
**
**  'DiffList' is the generic dispatcher for the differences involving lists.
**  That is,  whenever  two lists are subtracted and 'TabDiff' does not point
**  to  a special routine then 'DiffList'  is called.  'DiffList'  determines
**  the  extended  types  of   the  arguments  (e.g.,  including  'T_MATRIX',
**  'T_MATFFE', and 'T_LISTX') and then dispatches through 'TabDiff' again.
**
**  'DiffSclList'  is a generic function for  the first  kind of  difference,
**  that of a scalar with a list.
**
**  'DiffListScl'  is a generic function for the  second kind of  difference,
**  that of a list and a scalar.
**
**  'DiffListList'  is a  generic function for the  third kind of difference,
**  that of two lists.
*/
Bag       DiffList (Bag hdL, Bag hdR)
{
    return (*TabDiff[XType(hdL)][XType(hdR)])( hdL, hdR );
}

Bag       DiffSclList (Bag hdL, Bag hdR)
{
    Bag           hdD;            /* handle of the difference        */
    Bag           hdDD;           /* one element of difference list  */
    Bag           hdRR;           /* one element of right operand    */
    Int                len;            /* length                          */
    Int                i;              /* loop variable                   */

    /* make the result list                                                */
    len = LEN_LIST( hdR );
    hdD = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
    SET_LEN_PLIST( hdD, len );

    /* loop over the entries and subtract                                  */
    for ( i = 1; i <= len; i++ ) {
        hdRR = ELMR_LIST( hdR, i );
        hdDD = DIFF( hdL, hdRR );
        SET_ELM_PLIST( hdD, i, hdDD );
    }

    /* return the result                                                   */
    return hdD;
}

Bag       DiffListScl (Bag hdL, Bag hdR)
{
    Bag           hdD;            /* handle of the difference        */
    Bag           hdDD;           /* one element of difference list  */
    Bag           hdLL;           /* one element of left operand     */
    Int                len;            /* length                          */
    Int                i;              /* loop variable                   */

    /* make the result list                                                */
    len = LEN_LIST( hdL );
    hdD = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
    SET_LEN_PLIST( hdD, len );

    /* loop over the entries and subtract                                  */
    for ( i = 1; i <= len; i++ ) {
        hdLL = ELML_LIST( hdL, i );
        hdDD = DIFF( hdLL, hdR );
        SET_ELM_PLIST( hdD, i, hdDD );
    }

    /* return the result                                                   */
    return hdD;
}

Bag       DiffListList (Bag hdL, Bag hdR)
{
    Bag           hdD;            /* handle of the difference        */
    Bag           hdDD;           /* one element of the difference   */
    Bag           hdLL;           /* one element of the left list    */
    Bag           hdRR;           /* one element of the right list   */
    Int                len;            /* length                          */
    Int                i;              /* loop variable                   */

    /* get and check the length                                            */
    len = LEN_LIST( hdL );
    if ( len != LEN_LIST( hdR ) ) {
        return Error(
          "Vector -: lists must have the same length",
                     0, 0 );
    }
    hdD = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
    SET_LEN_PLIST( hdD, len );

    /* loop over the entries and subtract                                  */
    for ( i = 1; i <= len; i++ ) {
        hdLL = ELML_LIST( hdL, i );
        hdRR = ELMR_LIST( hdR, i );
        hdDD = DIFF( hdLL, hdRR );
        SET_ELM_PLIST( hdD, i, hdDD );
    }

    /* return the result                                                   */
    return hdD;
}


/****************************************************************************
**
*F  ProdList(<hdL>,<hdR>) . . . . . . . . . . . . . . . . .  product of lists
*F  ProdSclList(<hdL>,<hdR>)  . . . . . . . .  product of a scalar and a list
*F  ProdListScl(<hdL>,<hdR>)  . . . . . . . .  product of a list and a scalar
*F  ProdListList(<hdL>,<hdR>) . . . . . . . . . . . . .  product of two lists
**
**  'ProdList' is the generic dispatcher for  the  products  involving lists.
**  That is, whenever two lists are multiplied and  'TabProd'  does not point
**  to a special routine  then 'ProdList'  is  called.  'ProdList' determines
**  the  extended  types  of  the   arguments  (e.g.,  including  'T_MATRIX',
**  'T_MATFFE', and 'T_LISTX') and then dispatches through 'TabProd' again.
**
**  'ProdSclList' is a  generic function for the first kind  of product, that
**  of a scalar with a list.  Note that this includes the product of a matrix
**  with a list of matrices.
**
**  'ProdListScl'  is a generic function for the second kind of product, that
**  of a list and a scalar.  Note  that this includes the product of a matrix
**  with a vector.
**
**  'ProdListList' is a generic function for the third  kind of product, that
**  of  two lists.  Note that this  includes the  product  of a  vector and a
**  matrix.
*/
Bag       ProdList (Bag hdL, Bag hdR)
{
    return (*TabProd[XType(hdL)][XType(hdR)])( hdL, hdR );
}

Bag       ProdSclList (Bag hdL, Bag hdR)
{
    Bag           hdP;            /* handle of the product           */
    Bag           hdPP;           /* one element of product list     */
    Bag           hdRR;           /* one element of right operand    */
    Int                len;            /* length                          */
    Int                i;              /* loop variable                   */

    /* make the result list                                                */
    len = LEN_LIST( hdR );
    hdP = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
    SET_LEN_PLIST( hdP, len );

    /* loop over the entries and multiply                                  */
    for ( i = 1; i <= len; i++ ) {
        hdRR = ELMR_LIST( hdR, i );
        if ( hdRR != 0 ) {
            hdPP = PROD( hdL, hdRR );
            SET_ELM_PLIST( hdP, i, hdPP );
        }
    }

    /* return the result                                                   */
    return hdP;
}

Bag       ProdListScl (Bag hdL, Bag hdR)
{
    Bag           hdP;            /* handle of the product           */
    Bag           hdPP;           /* one element of product list     */
    Bag           hdLL;           /* one element of left operand     */
    Int                len;            /* length                          */
    Int                i;              /* loop variable                   */

    /* make the result list                                                */
    len = LEN_LIST( hdL );
    hdP = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
    SET_LEN_PLIST( hdP, len );

    /* loop over the entries and multiply                                  */
    for ( i = 1; i <= len; i++ ) {
        hdLL = ELML_LIST( hdL, i );
        if ( hdLL != 0 ) {
            hdPP = PROD( hdLL, hdR );
            SET_ELM_PLIST( hdP, i, hdPP );
        }
    }

    /* return the result                                                   */
    return hdP;
}

Bag       ProdListList (Bag hdL, Bag hdR)
{
    Bag           hdP;            /* handle of the product           */
    Bag           hdPP;           /* one summand of the product      */
    Bag           hdLL;           /* one element of the left list    */
    Bag           hdRR;           /* one element of the right list   */
    Int                len;            /* length                          */
    Int                i;              /* loop variable                   */

    /* get and check the length                                            */
    len = LEN_LIST( hdL );
    if ( len != LEN_LIST( hdR ) ) {
        return Error(
          "Vector *: lists must have the same length",
                     0, 0 );
    }

    /* loop over the entries and multiply and accumulate                   */
    hdLL = ELML_LIST( hdL, 1 );
    hdRR = ELMR_LIST( hdR, 1 );
    hdP  = PROD( hdLL, hdRR );
    for ( i = 2; i <= len; i++ ) {
        hdLL = ELML_LIST( hdL, i );
        hdRR = ELMR_LIST( hdR, i );
        hdPP = PROD( hdLL, hdRR );
        hdP  = SUM( hdP, hdPP );
    }

    /* return the result                                                   */
    return hdP;
}


/****************************************************************************
**
*F  QuoList(<hdL>,<hdR>)  . . . . . . . . . . . . . . . . . quotient of lists
*F  QuoLists(<hdL>,<hdR>) . . . . . . . . . . . . . . . . . quotient of lists
**
**  'QuoList'  is the generic dispatcher for  the  quotients involving lists.
**  This  is, whenever two lists are divided and 'TabQuo' does not point to a
**  special routine  then  'QuoList'  is  called.   'QuoList'  determines the
**  extended types of the arguments  (e.g., including 'T_MATRIX', 'T_MATFFE',
**  and 'T_LISTX') and then dispatches through 'TabProd' again.
**
**  'QuoLists' is a generic function, which only returns the product of <hdL>
**  and the  inverse of <hdR> (computed as <hdR>^-1).  The right operand must
**  either be a scalar or an invertable square matrix.
*/
Bag       QuoList (Bag hdL, Bag hdR)
{
    return (*TabQuo[XType(hdL)][XType(hdR)])( hdL, hdR );
}

Bag       QuoLists (Bag hdL, Bag hdR)
{
    return PROD( hdL, POW( hdR, INT_TO_HD(-1) ) );
}


/****************************************************************************
**
*F  ModList(<hdL>,<hdR>)  . . . . . . . . . . . . . . . . . quotient of lists
*F  ModLists(<hdL>,<hdR>) . . . . . . . . . . . . . . . . . quotient of lists
**
**  'ModList'  is the generic dispatcher for  left quotients involving lists.
**  This  is, whenever two lists are divided and 'TabMod' does not point to a
**  special routine  then  'ModList'  is  called.   'ModList'  determines the
**  extended types of the arguments  (e.g., including 'T_MATRIX', 'T_MATFFE',
**  and 'T_LISTX') and then dispatches through 'TabProd' again.
**
**  'ModLists' is a generic function, which only returns the product of <hdR>
**  and the  inverse of <hdL> (computed as <hdL>^-1).  The left  operand must
**  either be a scalar or an invertable square matrix.
*/
Bag       ModList (Bag hdL, Bag hdR)
{
    return (*TabMod[XType(hdL)][XType(hdR)])( hdL, hdR );
}

Bag       ModLists (Bag hdL, Bag hdR)
{
    return PROD( POW( hdL, INT_TO_HD(-1) ), hdR );
}


/****************************************************************************
**
*F  PowList(<hdL>,<hdR>)  . . . . . . . . . . . . . . . . . .  power of lists
*F  PowLists(<hdL>,<hdR>) . . . . . . . . . . . . . . . power of two matrices
**
**  'PowList'  returns  the  power  of the left  operand  <hdL>  by the right
**  operand <hdR>.  Two cases are actually  possible, <hd>  is  a matrix  and
**  <hdR>  is  an  integer, or both are matrices.   'PowList'  determines the
**  extended  types  of  the   arguments  (e.g.,   including  'T_MATRIX'  and
**  'T_MATFFE') and then dispatches through 'TabPow' again.
**
**  'PowLists' is a generic function for the third kind of power, that of two
**  matrices, which is defined as the conjugation.
*/
Bag       PowList (Bag hdL, Bag hdR)
{
    return (*TabPow[XType(hdL)][XType(hdR)])( hdL, hdR );
}

Bag       PowLists (Bag hdL, Bag hdR)
{
    return PROD( MOD( hdR, hdL ), hdR );
}


/****************************************************************************
**
*F  CommList(<hdL>,<hdR>) . . . . . . . . . . . . . . . . commutator of lists
*F  CommLists(<hdL>,<hdR>)  . . . . . . . . . . .  commutator of two matrices
**
**  'CommList'  returns the commutator of the  two  operands <hdL> and <hdR>,
**  which must both be matrices.  'CommList' determines the extended types of
**  the  arguments  (e.g.,  including  'T_MATRIX' and  'T_MATFFE')  and  then
**  dispatches through 'TabComm' again.
**
**  'CommLists' is a generic function for the commutator of two matrices.
*/
Bag       CommList (Bag hdL, Bag hdR)
{
    return (*TabComm[XType(hdL)][XType(hdR)])( hdL, hdR );
}

Bag       CommLists (Bag hdL, Bag hdR)
{
/* this long macro expansion breaks the Visual Studio.NET 2003 compiler...
        return PROD( POW( PROD( hdR, hdL ), INT_TO_HD(-1) ), PROD( hdL, hdR ) );
*/
    Bag tmp=POW( PROD( hdR, hdL ), INT_TO_HD(-1) );
        return PROD(tmp, PROD( hdL, hdR ) );

}


/****************************************************************************
**
*F  EvElmList(<hdSel>)  . . . . . . . . . . . . . select an element of a list
**
**  'EvElmList' returns the value of the element at the position '<hdSel>[1]'
**  in the list '<hdSel>[0]'.   Both '<hdSel>[0]' and '<hdSel>[1]' first have
**  to be evaluated.
*/
Bag       EvElmList (Bag hdSel)
{
    Bag           hdElm;          /* <element>, result               */
    Bag           hdList;         /* <list>, left operand            */
    Bag           hdPos;          /* <position>, right operand       */
    Int                pos;            /* <position>, as a C integer      */

    /* evaluate <list> (checking is done via the dispatching)              */
    hdList = EVAL( PTR_BAG(hdSel)[0] );

    /* evaluate and check <position>                                       */
    hdPos = EVAL( PTR_BAG(hdSel)[1] );
    if ( GET_TYPE_BAG(hdPos) != T_INT || HD_TO_INT(hdPos) <= 0 ) {
        return Error(
          "List Element: <position> must be a positive integer",
                     0, 0 );
    }
    pos = HD_TO_INT( hdPos );

    /* make the selection and return it                                    */
    if ( GET_TYPE_BAG(hdList) == T_LIST ) {
        if ( LEN_PLIST( hdList ) < pos ) {
            return Error(
              "List Element: <list>[%d] must have a value",
                         pos, 0 );
        }
        hdElm = ELM_PLIST( hdList, pos );
        if ( hdElm == 0 ) {
            return Error(
              "List Element: <list>[%d] must have a value",
                         pos, 0 );
        }
    }
    else {
        hdElm = ELM_LIST( hdList, pos );
    }
    return hdElm;
}


/****************************************************************************
**
*F  EvElmListLevel(<hdSel>) . .  select elements of several lists in parallel
**
**  'EvElmListLevel'    evaluates    a   list    selection   of   the    form
**  '<list>...{<positions>}...[<position>]',  where  there  may  actually  be
**  several  '{<positions>}'  selections between <list>  and  '[<position>]'.
**  The number of those is called the level.  'EvElmListLevel' goes that deep
**  into the  left operand and selects the element at <position> from each of
**  those lists.  Thus if the level is  one, the left operand must be a  list
**  of lists and 'EvElmListLevel' selects the element at <position> from each
**  of the lists and returns the list of those values.
**
**  We     assume     that    'EvElmsList'    (the     function    evaluating
**  '<list>...{<positions>}')  creates dense lists,  so  that  we can  select
**  elements with 'ELMF_LIST'.  We also assume that a  list of lists  has the
**  representation   of   plain  lists,   so   that  we   can   assign   with
**  'PTR_BAG(<hdLists>)[i]  =  <hdElm>;'  Finally  we  assume  that  'EvElmsList'
**  creates a new list,  so that we can overwrite this list with the selected
**  values.
*/
Bag       ElmListLevel (Bag hdLists, Int pos, Int level)
{
    Int                lenLists;       /* length of <lists>               */
    Bag           hdList;         /* one list from <lists>           */
    Bag           hdElm;          /* selected element from <list>    */
    Int                i;              /* loop variable                   */

    /* if <level> is one, loop over all lists and select the element       */
    if ( level == 1 ) {

        /* loop over its entries (which must be lists too)                 */
        lenLists = LEN_LIST( hdLists );
        for ( i = 1; i <= lenLists; i++ ) {

            /* get the list                                                */
            /* here we assume that 'EvElmsList' creates a dense list       */
            hdList = ELMF_LIST( hdLists, i );

            /* select the element                                          */
            /* this will signal an error if <lists> is not a list of lists */
            hdElm = ELM_LIST( hdList, pos );

            /* assign the element back into <lists>                        */
            /* here we assume that a list of lists has rep. 'T_LIST'       */
            SET_ELM_PLIST( hdLists, i, hdElm );

        }

    }

    /* otherwise recurse                                                   */
    else {

        /* loop over its entries (which must be lists too)                 */
        lenLists = LEN_LIST( hdLists );
        for ( i = 1; i <= lenLists; i++ ) {

            /* get the list                                                */
            /* here we assume that 'EvElmsList' creates dense lists        */
            hdList = ELMF_LIST( hdLists, i );
            if ( ! IS_LIST( hdList ) ) {
                return Error(
                  "List Element: <list> must be a list",
                             0, 0 );
            }

            /* recurse                                                     */
            ElmListLevel( hdList, pos, level-1 );

        }

    }

    /* return the selection                                                */
    return hdLists;
}

Bag       EvElmListLevel (Bag hdSel)
{
    Bag           hdLists;        /* <list>, left operand            */
    Bag           hdPos;          /* <position>, right operand       */
    Int                level;          /* <level>                         */

    /* evaluate and check <list>                                           */
    hdLists = EVAL( PTR_BAG(hdSel)[0] );
    if ( ! IS_LIST( hdLists ) ) {
        return Error(
          "List Element: <list> must be a list",
                     0, 0 );
    }

    /* evaluate and check <position>                                       */
    hdPos = EVAL( PTR_BAG(hdSel)[1] );
    if ( GET_TYPE_BAG(hdPos) != T_INT || HD_TO_INT(hdPos) <= 0 ) {
        return Error(
          "List Element: <position> must be a positive integer",
                     0, 0 );
    }

    /* get <level>                                                         */
    level = *(Int*)(PTR_BAG(hdSel)+2);

    /* make the selection and return it                                    */
    return ElmListLevel( hdLists, HD_TO_INT(hdPos), level );
}


/****************************************************************************
**
*F  EvElmsList(<hdSel>) . . . . . . . . . . . . .  select a sublist of a list
**
**  'EvElmsList' returns the sublist of the elements at the  positions  given
**  in the list '<hdSel>[1]' in the list '<hdSel>[0]'.  Both '<hdSel>[0]' and
**  '<hdSel>[1]' first have to be evaluated.  This implements  the  construct
**  '<list>{<positions>}'.
*/
Bag       EvElmsList (Bag hdSel)
{
    Bag           hdList;         /* <list>, left operand            */
    Bag           hdPoss;         /* <positions>, right operand      */

    /* evaluate <list> (checking is done via the dispatching)              */
    hdList = EVAL( PTR_BAG(hdSel)[0] );

    /* evaluate and check <positions>                                      */
    hdPoss = EVAL( PTR_BAG(hdSel)[1] );
    if ( ! IS_POSS_LIST( hdPoss ) ) {
        return Error(
      "List Elements: <positions> must be a dense list of positive integers",
                     0, 0 );
    }

    /* make the selection and return it                                    */
    return ELMS_LIST( hdList, hdPoss );
}


/****************************************************************************
**
*F  EvElmsListLevel(<hdSel>)  .  select sublists of several lists in parallel
**
**  'EvElmsListLevel'   evaluates   a   list    selection    of    the   form
**  '<list>...{<positions>}...{<positions>]',  where there  may  actually  be
**  several '{<positions>}' selections between  <list>  and  '{<positions>}'.
**  The number of those  is called  the level.   'EvElmsListLevel' goes  that
**  deep into the left operand  and selects the elements  at <positions> from
**  each of  those lists.  Thus if the level is one, the left operand must be
**  a list of lists and 'EvElmsListLevel' selects the elements at <positions>
**  from each of the lists and returns the list of those sublists.
**
**  We    assume     that    'EvElmsList'    (the     function     evaluating
**  '<list>...{<positions>}')  creates  dense lists, so  that  we can  select
**  elements  with 'ELMF_LIST'.  We also  assume that a list of lists has the
**  representation   of  plain   lists,   so  that   we   can   assign   with
**  'PTR_BAG(<hdLists>)[i]  =  <hdElm>;'  Finally  we  assume  that  'EvElmsList'
**  creates a  new list, so that we can overwrite this list with the selected
**  values.
*/
Bag       ElmsListLevel (Bag hdLists, Bag hdPoss, Int level)
{
    Int                lenLists;       /* length of <lists>               */
    Bag           hdList;         /* one list from <lists>           */
    Bag           hdElms;         /* selected elements from <list>   */
    Int                i;              /* loop variable                   */

    /* if <level> is one, loop over all lists and select the element       */
    if ( level == 1 ) {

        /* loop over its entries (which must be lists too)                 */
        lenLists = LEN_LIST( hdLists );
        for ( i = 1; i <= lenLists; i++ ) {

            /* get the list                                                */
            /* here we assume that 'EvElmsList' creates dense lists        */
            hdList = ELMF_LIST( hdLists, i );

            /* select the element                                          */
            /* this will signal an error if <lists> is not a list of lists */
            hdElms = ELMS_LIST( hdList, hdPoss );

            /* assign the element back into <list>                         */
            /* here we assume that a list of lists has rep. 'T_LIST'       */
            SET_ELM_PLIST( hdLists, i, hdElms );

        }

    }

    /* otherwise recurse                                                   */
    else {

        /* loop over its entries (which must be lists too)                 */
        lenLists = LEN_LIST( hdLists );
        for ( i = 1; i <= lenLists; i++ ) {

            /* get the list                                                */
            /* here we assume that 'EvElmsList' creates dense lists        */
            hdList = ELMF_LIST( hdLists, i );
            if ( ! IS_LIST( hdList ) ) {
                return Error(
                  "List Elements: <list> must be a list",
                             0, 0 );
            }

            /* recurse                                                     */
            ElmsListLevel( hdList, hdPoss, level-1 );

        }

    }

    /* return the selection                                                */
    return hdLists;
}

Bag       EvElmsListLevel (Bag hdSel)
{
    Bag           hdLists;        /* <list>, left operand            */
    Bag           hdPoss;         /* <positions>, right operand      */
    Int                level;          /* <level>                         */

    /* evaluate and check <lists>                                          */
    hdLists = EVAL( PTR_BAG(hdSel)[0] );
    if ( ! IS_LIST( hdLists ) ) {
        return Error(
          "List Elements: <list> must be a list",
                     0, 0 );
    }

    /* evaluate and check <positions>                                      */
    hdPoss = EVAL( PTR_BAG(hdSel)[1] );
    if ( ! IS_POSS_LIST( hdPoss ) ) {
        return Error(
      "List Elements: <positions> must be a dense list of positive integers",
                     0, 0 );
    }

    /* get <level>                                                         */
    level = *(Int*)(PTR_BAG(hdSel)+2);

    /* make the selection and return it                                    */
    return ElmsListLevel( hdLists, hdPoss, level );
}


/****************************************************************************
**
*F  EvAssList(<hdAss>)  . . . . . . . . . . .  assign to an element of a list
**
**  'EvAssList'  assigns   the  object  '<hdAss>[1]'    to  the  list element
**  '<hdAss>[0]',  i.e., to  the element  at  position '<hdAss>[0][1]' in the
**  list '<hdAss>[0][0]'.
*/
Bag       EvAssList (Bag hdAss)
{
    Bag           hdList;         /* <list>, left operand            */
    Int                plen;           /* physical length of <list>       */
    Bag           hdPos;          /* <position>, left operand        */
    Int                pos;            /* <position>, as a C integer      */
    Bag           hdVal;          /* <value>, right operand          */

    /* evaluate <list> (checking is done via dispatching)                  */
    hdList = EVAL( PTR_BAG( PTR_BAG(hdAss)[0] )[0] );

    /* evaluate and check <position>                                       */
    hdPos = EVAL( PTR_BAG( PTR_BAG(hdAss)[0] )[1] );
    if ( GET_TYPE_BAG(hdPos) != T_INT || HD_TO_INT(hdPos) <= 0 ) {
        return Error(
          "List Element: <position> must be a positive integer",
                     0, 0 );
    }
    pos = HD_TO_INT(hdPos);

    /* evaluate and check <value>                                          */
    hdVal = EVAL( PTR_BAG(hdAss)[1] );
    if ( hdVal == HdVoid ) {
        return Error(
          "List Assignment: function must return a value",
                     0, 0 );
    }

    /* make the assignment and return the assigned value                   */
    if ( GET_TYPE_BAG(hdList) == T_LIST ) {
        if ( LEN_PLIST( hdList ) < pos ) {
            plen = PLEN_SIZE_PLIST( GET_SIZE_BAG( hdList ) );
            if ( plen + plen/8 + 4 < pos )
                Resize( hdList, SIZE_PLEN_PLIST( pos ) );
            else if ( plen < pos )
                Resize( hdList, SIZE_PLEN_PLIST( plen + plen/8 + 4 ) );
            SET_LEN_PLIST( hdList, pos );
        }
        SET_ELM_PLIST( hdList, pos, hdVal );
    }
    else {
        hdVal = ASS_LIST( hdList, HD_TO_INT(hdPos), hdVal );
    }
    return hdVal;
}

Bag       EvMultiAss (Bag hdAss)
{
    Bag           hdList;         /* <list>, left operand            */
    Int                plen,i;           /* physical length of <list>       */
    Bag           hdVal;          /* <value>, right operand          */
    Bag           hd;
    /* evaluate <list> (checking is done via dispatching)                  */
    hdList = PTR_BAG(hdAss)[0];
    plen = LEN_PLIST(hdList);

    for(i=1; i<=plen; ++i)
        {
                if(GET_TYPE_BAG(PTR_BAG(hdList)[i]) != T_VAR) /*autovar*/
                        Error("Multi assignment: <%d>-th element must be a variable", i, 0);
    }

    /* evaluate and check <value>                                          */
    hdVal = EVAL( PTR_BAG(hdAss)[1] );
    if ( hdVal == HdVoid )
        return Error("Multi assignment: function must return a value", 0, 0 );

        if ( ! IS_LIST(hdVal) )
        return Error("Multi assignment: right-hand side must be a list, <%s> given",
                     (Int)TYPENAME(hdVal), 0 );

    if ( LEN_LIST(hdVal) != plen )
                return Error("Multi assignment: <%d> elements expected, but right hand side has <%d> elements",
                     plen, LEN_LIST(hdVal));

    for(i=1; i<=plen; ++i) {
        hd = PTR_BAG(hdList)[i];
        SET_VAR_VALUE(hd, ELM_LIST(hdVal, i));
    }

    return HdVoid;
}

/****************************************************************************
**
*F  EvAssListLevel(<hdSel>) . assign to elements of several lists in parallel
**
**  'EvAssListLevel'   evaluates   a    list    assignment   of   the    form
**  '<list>...{<positions>}...[<position>]  :=  <vals>;',   where  there  may
**  actually  be  several   '{<positions>}'  selections  between  <list>  and
**  '[<position>]'.    The   number   of   those   is   called   the   level.
**  'EvAssListLevel' goes that  deep into the  left  operand  and <vals>  and
**  assigns the values from <vals> to each of those lists.  Thus if the level
**  is one,  the left operand must be a list of lists, <vals> must be a list,
**  and 'EvAssListLevel'  assigns  the  element  '<vals>[<i>]'  to  the  list
**  '<left>[<i>]' at <position>.
**
**  We    assume     that    'EvElmsList'    (the     function     evaluating
**  '<list>...{<positions>}')  creates  dense lists, so  that  we can  select
**  elements  with 'ELMF_LIST'.
*/
Bag       AssListLevel (Bag hdLists, Int pos, Bag hdVals, Int level)
{
    Bag           hdList;         /* one list of <lists>             */
    Bag           hdVal;          /* one value from <values>         */
    Int                lenLists;       /* length of <lists> and <vals>    */
    Int                i;              /* loop variable                   */

    /* if <level> is one, loop over all the lists and assign the value     */
    if ( level == 1 ) {

        /* loop over the list entries (which must be lists too)            */
        lenLists = LEN_LIST( hdLists );
        for ( i = 1; i <= lenLists; i++ ) {

            /* get the list                                                */
            /* here we assume that 'EvElmsList' creates dense lists        */
            hdList = ELMF_LIST( hdLists, i );

            /* select the element to assign                                */
            hdVal = ELMF_LIST( hdVals, i );
            if ( hdVal == 0 ) {
                return Error(
                  "List Assignment: <vals> must be a dense list",
                             0, 0 );
            }

            /* assign the element                                          */
            /* this will signal an error if <lists> is not a list of lists */
            ASS_LIST( hdList, pos, hdVal );

        }

    }

    /* otherwise recurse                                                   */
    else {

        /* loop over the list entries (which must be lists too)            */
        lenLists = LEN_LIST( hdLists );
        for ( i = 1; i <= lenLists; i++ ) {

            /* get the list                                                */
            /* here we assume that 'EvElmsList' creates dense lists        */
            hdList = ELMF_LIST( hdLists, i );
            if ( ! IS_LIST( hdList ) ) {
                return Error(
                  "List Assignment: <list> must be a list",
                             0, 0 );
            }

            /* get the values                                              */
            hdVal = ELMF_LIST( hdVals, i );
            if ( hdVal == 0 ) {
                return Error(
                  "List Assignment: <vals> must be a dense list",
                             0, 0 );
            }
            if ( ! IS_LIST( hdVal ) ) {
                return Error(
                  "List Assignment: <vals> must be a dense list",
                             i, 0 );
            }
            if ( LEN_LIST( hdList ) != LEN_LIST( hdVal ) ) {
                return Error(
              "List Assignment: <list> and <vals> must have the same length",
                             0, 0 );
            }

            /* recurse                                                     */
            AssListLevel( hdList, pos, hdVal, level-1 );

        }

    }

    /* return the assigned values                                          */
    return hdVals;
}

Bag       EvAssListLevel (Bag hdAss)
{
    Bag           hdLists;        /* <list>, left operand            */
    Bag           hdPos;          /* <position>, left operand        */
    Bag           hdVals;         /* <values>, right operand         */
    Int                level;          /* <level>                         */

    /* evaluate and check <list>                                           */
    hdLists = EVAL( PTR_BAG( PTR_BAG(hdAss)[0] )[0] );
    if ( ! IS_LIST( hdLists ) ) {
        return Error(
          "List Assignment: <list> must be a list",
                     0, 0 );
    }

    /* evaluate and check <position>                                       */
    hdPos = EVAL( PTR_BAG( PTR_BAG(hdAss)[0] )[1] );
    if ( GET_TYPE_BAG(hdPos) != T_INT || HD_TO_INT(hdPos) <= 0 ) {
        return Error(
          "List Assignment: <position> must be a positive integer",
                     0, 0 );
    }

    /* evaluate <values>                                                   */
    hdVals = EVAL( PTR_BAG(hdAss)[1] );
    if ( hdVals == HdVoid ) {
        return Error(
          "List Assignment: function must return a value",
                     0, 0 );
    }

    /* get <level>                                                         */
    level = *(Int*)(PTR_BAG(PTR_BAG(hdAss)[0])+2);

    /* make the assignments and return the assigned values                 */
    return AssListLevel( hdLists, HD_TO_INT(hdPos), hdVals, level );
}


/****************************************************************************
**
*F  EvAsssList(<hdAss>) . . . . . . . . . . . . assign to a sublist of a list
**
**  'EvAssList'  assigns   the  object  '<hdAss>[1]'    to  the  list sublist
**  '<hdAss>[0]',  i.e., to  the elements at positions '<hdAss>[0][1]' in the
**  list '<hdAss>[0][0]'.
*/
Bag       EvAsssList (Bag hdAss)
{
    Bag           hdList;         /* <list>, left operand            */
    Bag           hdPoss;         /* <positions>, left operand       */
    Bag           hdVals;         /* <values>, right operand         */

    /* evaluate <list> (checking is done via dispatching)                  */
    hdList = EVAL( PTR_BAG( PTR_BAG(hdAss)[0] )[0] );

    /* evaluate and check <positions>                                      */
    hdPoss = EVAL( PTR_BAG( PTR_BAG(hdAss)[0] )[1] );
    if ( ! IS_POSS_LIST( hdPoss ) ) {
        return Error(
   "List Assignments: <positions> must be a dense list of positive integers",
                     0, 0 );
    }

    /* evaluate and check <values>                                         */
    hdVals = EVAL( PTR_BAG(hdAss)[1] );
    if ( ! IS_DENSE_LIST( hdVals ) ) {
        return Error(
          "List Assignments: <vals> must be a dense list",
                     0, 0 );
    }
    if ( LEN_LIST( hdVals ) != LEN_LIST( hdPoss ) ) {
        return Error(
          "List Assiments: <positions> and <vals> must have the same length",
                     0, 0 );
    }

    /* make the assignment and return the assigned values                  */
    return ASSS_LIST( hdList, hdPoss, hdVals );
}


/****************************************************************************
**
*F  EvAsssListLevel(<hdSel>)  assign to sublists of several lists in parallel
**
**  'EvAssListLevel'   evaluates    a   list   assignment    of   the    form
**  '<list>...{<positions>}...{<positions>}  :=  <vals>;',  where  there  may
**  actually  be  several  '{<positions>}'  selections   between  <list>  and
**  '{<positions>}'.    The   number   of   those   is  called   the   level.
**  'EvAssListLevel'  goes  that  deep into the  left  operand and <vals> and
**  assigns  the  sublists from  <vals> to each of those lists.  Thus  if the
**  level  is one, the left operand must be a list of lists, <vals> must be a
**  list, and 'EvAssListLevel' assigns the sublist '<vals>[<i>]'  to the list
**  '<left>[<i>]' at the positions <positions>.
**
**  We    assume     that    'EvElmsList'     (the     function    evaluating
**  '<list>...{<positions>}') created a new dense list.
*/
Bag       AsssListLevel (Bag hdLists, Bag hdPoss, Bag hdVals, Int lev)
{
    Int                lenLists;       /* length of <lists> and <vals>    */
    Int                lenPoss;        /* length of <positions>           */
    Bag           hdList;         /* one list of <lists>             */
    Bag           hdVal;          /* one value from <values>         */
    Int                i;              /* loop variable                   */

    /* if <lev> is one, loop over all the lists and assign the value       */
    if ( lev == 1 ) {

        /* get the length of <positions>                                   */
        lenPoss = LEN_LIST( hdPoss );

        /* loop over the list entries (which must be lists too)            */
        lenLists = LEN_LIST( hdLists );
        for ( i = 1; i <= lenLists; i++ ) {

            /* get the list                                                */
            hdList = ELMF_LIST( hdLists, i );

            /* select the element to assign                                */
            hdVal = ELMF_LIST( hdVals, i );
            if ( hdVal == 0 ) {
                return Error(
                  "List Assignments: <vals> must be a dense list",
                             0, 0 );
            }
            if ( ! IS_DENSE_LIST( hdVal ) ) {
                return Error(
                  "List Assignments: <vals> must be a dense list",
                             0, 0 );
            }
            if ( LEN_LIST( hdVal ) != lenPoss ) {
                return Error(
         "List Assigments: <positions> and <vals> must have the same lenght",
                             0, 0 );
            }

            /* assign the element                                          */
            /* this will signal an error if <lists> is not a list of lists */
            ASSS_LIST( hdList, hdPoss, hdVal );

        }

    }

    /* otherwise recurse                                                   */
    else {

        /* loop over the list entries (which must be lists too)            */
        lenLists = LEN_LIST( hdLists );
        for ( i = 1; i <= lenLists; i++ ) {

            /* get the list                                                */
            hdList = ELMF_LIST( hdLists, i );
            if ( ! IS_LIST( hdList ) ) {
                return Error(
                  "List Assignments: <list> must be a list",
                             0, 0 );
            }

            /* get the values                                              */
            hdVal = ELMF_LIST( hdVals, i );
            if ( hdVal == 0 ) {
                return Error(
                  "List Assignments: <vals> must be a dense list",
                             0, 0 );
            }
            if ( ! IS_LIST( hdVal ) ) {
                return Error(
                  "List Assignments: <vals> must be a dense list",
                             0, 0 );
            }
            if ( LEN_LIST( hdVal ) != LEN_LIST( hdList ) ) {
                return Error(
             "List Assignments: <list> and <vals> must have the same length",
                             0, 0 );
            }

            /* recurse                                                     */
            AsssListLevel( hdList, hdPoss, hdVal, lev-1 );

        }

    }

    /* return the assigned values                                          */
    return hdVals;
}

Bag       EvAsssListLevel (Bag hdAss)
{
    Bag           hdLists;        /* <list>, left operand            */
    Bag           hdPoss;         /* <positions>, left operand       */
    Bag           hdVals;         /* <values>, right operand         */
    Int                level;          /* <level>                         */

    /* evaluate and check <lists>                                          */
    hdLists = EVAL( PTR_BAG( PTR_BAG(hdAss)[0] )[0] );
    if ( ! IS_LIST( hdLists ) ) {
        return Error(
          "List Assignments: <list> must be a list",
                     0, 0 );
    }

    /* evaluate and check <positions>                                      */
    hdPoss = EVAL( PTR_BAG( PTR_BAG(hdAss)[0] )[1] );
    if ( ! IS_POSS_LIST( hdPoss ) ) {
        return Error(
   "List Assignments: <positions> must be a dense list of positive integers",
                     0, 0 );
    }

    /* evaluate and check <vals>                                           */
    hdVals = EVAL( PTR_BAG(hdAss)[1] );
    if ( hdVals == HdVoid ) {
        return Error(
          "List Assignments: function must return a value",
                     0, 0 );
    }
    if ( ! IS_LIST( hdVals ) ) {
        return Error(
          "List Assignments: <vals> must be a list",
                     0, 0 );
    }
    if ( LEN_LIST( hdVals ) != LEN_LIST( hdLists ) ) {
        return Error(
          "List Assignments: <lists> and <vals> must have the same length",
                     0, 0 );
    }

    /* get <level>                                                         */
    level = *(Int*)(PTR_BAG(PTR_BAG(hdAss)[0])+2);

    /* make the assignments and return the assigned values                 */
    return AsssListLevel( hdLists, hdPoss, hdVals, level );
}


/****************************************************************************
**
*F  PrElmList(<hdSel>)  . . . . . . . . . . . . . . .  print a list selection
**
**  'PrElmList' prints the list selection <hdSel>.
**
**  Linebreaks are preferred after the '['.
*/
void            PrElmList (Bag hdSel)
{
    Pr("%2>",0,0);  Print( PTR_BAG(hdSel)[0] );
    Pr("%<[",0,0);  Print( PTR_BAG(hdSel)[1] );
    Pr("%<]",0,0);
}


/****************************************************************************
**
*F  PrElmsList(<hdSel>) . . . . . . . . . . . . . . .  print a list selection
**
**  'PrElmsList' prints the list selection <hdSel>.
**
**  Linebreaks are preferred after the '{'.
*/
void            PrElmsList (Bag hdSel)
{
    Pr("%2>",0,0);  Print( PTR_BAG(hdSel)[0] );
    Pr("%<{",0,0);  Print( PTR_BAG(hdSel)[1] );
    Pr("%<}",0,0);
}


/****************************************************************************
**
*F  PrAssList(<hdAss>)  . . . . . . . . print an assignment to a list element
**
**  'PrAssList' prints the assignment to a list element.
**
**  Linebreaks are preferred before the ':='.
*/
void            PrAssList (Bag hdAss)
{
    Pr("%2>",0,0);       Print( PTR_BAG(hdAss)[0] );
    Pr("%< %>:= ",0,0);  Print( PTR_BAG(hdAss)[1] );
    Pr("%2<",0,0);
}


/****************************************************************************
**
*F  EvIn(<hdIn>)  . . . . . . . . . . . . . . . test for membership in a list
**
**  'EvIn' implements the membership operator '<obj> in <list>'.
**
**  'in' returns 'true' if the object <obj> is  an element of the list <list>
**  and 'false' otherwise.  'in'  works fastest if   <list> is a proper  set,
**  i.e., has no holes, is sorted and  contains no duplicates because in this
**  case  'in' can use  a binary search.   If <list> is   a general list 'in'
**  employs a linear search.
*/
Bag       EvIn (Bag hdIn)
{
    Bag           hdVal;          /* <val>, left operand             */
    Bag           hdList;         /* <list>, right operand           */
    Int                pos;            /* <position>                      */

    /* evaluate and check <val>                                            */
    hdVal = EVAL( PTR_BAG(hdIn)[0] );
    if ( hdVal == HdVoid ) {
        return Error(
          "In: function must return a value",
                     0, 0 );
    }

    /* evaluate <list>                                                     */
    hdList = EVAL( PTR_BAG(hdIn)[1] );

    /* special case for records                                            */
    /*N 1992/12/10 martin should 'TabPosList[T_REC]' be used for this?     */
    if ( GET_TYPE_BAG(hdList) == T_REC )
        return InRec( hdVal, hdList );

    /* search the element                                                  */
    pos = POS_LIST( hdList, hdVal, 0 );

    /* return the position                                                 */
    return (pos != 0) ? HdTrue : HdFalse;
}


/****************************************************************************
**
*F  FunIsList(<hdCall>) . . . . . . . . . . . . . . . . . . .  test for lists
**
**  'FunIsList' implements the internal function 'IsList( <obj> )'.
**
**  'IsList'  returns  'true' if its argument   <obj> is  a  list and 'false'
**  otherwise.
*/
Bag       FunIsList (Bag hdCall)
{
    Bag           hdObj;

    /* check the arguments                                                 */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: IsList( <obj> )",0,0);

    /* evaluate and check the object                                       */
    hdObj = EVAL( PTR_BAG(hdCall)[1] );
    if ( hdObj == HdVoid ) {
        return Error(
          "IsList: function must return a value",
                     0,0);
    }

    /* return 'true' if <obj> is a list and 'false' otherwise              */
    return IS_LIST(hdObj) ? HdTrue : HdFalse;
}


/****************************************************************************
**
*F  FunIsVector(<hdCall>) . . . . . . . . . . . .  test if a list is a vector
*F  IsVector(<hdObj>) . . . . . . . . . . . . . .  test if a list is a vector
**
**  'FunIsVector' implements the internal function 'IsVector'.
**
**  'IsVector( <obj> )'
**
**  'IsVector' return 'true' if <obj>, which can be an  object  of  arbitrary
**  type, is a vector and 'false' otherwise.  Will cause an error if <obj> is
**  an unbound variable.
*/
Int            IsVector (Bag hdObj)
{
    /* test if <hdObj> is a list and a vector                              */
    if ( IS_LIST( hdObj ) && (TabIsList[ XType( hdObj ) ] == 2) )
        return 1;
    else
        return 0;
}

Bag       FunIsVector (Bag hdCall)
{
    Bag           hdObj;          /* handle of the object            */

    /* check and get the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
        return Error("usage: IsVector( <obj> )",0,0);
    hdObj = EVAL( PTR_BAG(hdCall)[1] );
    if ( hdObj == HdVoid )
        return Error("IsVector: function must return value",0,0);

    /* test if <hdObj> is a list and a vector                              */
    if ( IS_LIST( hdObj ) && (TabIsList[ XType( hdObj ) ] == 2) )
        return HdTrue;
    else
        return HdFalse;
}


/****************************************************************************
**
*F  FunIsMat(<hdCall>)  . . . . . . . . . . . . .  test if a list is a matrix
**
**  'FunIsMat' implements the internal function 'IsMat'.
**
**  'IsMat( <obj> )'
**
**  'IsMat'  return  'true'  if <obj>, which can be an  object  of  arbitrary
**  type, is a matrix and 'false' otherwise.  Will cause an error if <obj> is
**  an unbound variable.
*/
Bag       FunIsMat (Bag hdCall)
{
    Bag           hdObj;          /* handle of the object            */

    /* check and get the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
        return Error("usage: IsMat( <obj> )",0,0);
    hdObj = EVAL( PTR_BAG(hdCall)[1] );
    if ( hdObj == HdVoid )
        return Error("IsMat: function must return value",0,0);

    /* test if <hdObj> is a list and a matrix                              */
    if ( IS_LIST( hdObj ) && (TabIsList[ XType( hdObj ) ] == 3) )
        return HdTrue;
    else
        return HdFalse;
}


/****************************************************************************
**
*F  FunLength(<hdCall>) . . . . . . . . . . . . . . . . . .  length of a list
**
**  'FunLength' implements the internal function 'Length'.
**
**  'Length( <list> )'
**
**  'Length' returns the length of a list '<list>'.
*/
Bag       FunLength (Bag hdCall)
{
    Bag           hdList;         /* handle of the list              */

    /* check the number of arguments                                       */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: Length( <list> )",0,0);

    /* evaluate <list>                                                     */
    hdList = EVAL( PTR_BAG(hdCall)[1] );

    /* return the length                                                   */
    if ( GET_TYPE_BAG(hdList) == T_LIST ) {
        /* return INT_TO_HD( LEN_PLIST( hdList ) );                        */
        return PTR_BAG(hdList)[0];
    }
    else {
        return INT_TO_HD( LEN_LIST( hdList ) );
    }
}


/****************************************************************************
**
*F  FunAdd(<hdCall>)  . . . . . . . . . . add an element to the end of a list
**
**  'FunAdd' implements the internal function 'Add(<list>,<obj>)'.
**
**  'Add' adds the object  <obj> to the end  of the list  <list>, i.e., it is
**  equivalent  to the assignment '<list>[  Length(<list>)  + 1  ] := <obj>'.
**  The  list is  automatically extended to   make room for  the new element.
**  'Add' returns nothing, it is called only for its sideeffect.
*/
Bag       FunAdd (Bag hdCall)
{
    Bag           hdList;         /* <list>, first argument          */
    Bag           hdVal;          /* <val>, second argument          */
    Int                pos;            /* position to assign to           */
    Int                plen;           /* physical length of <list>       */

    /* check the argument count                                            */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD ) {
        return Error(
          "usage: Add( <list>, <val> )",
                     0, 0 );
    }

    /* evaluate <list>                                                     */
    hdList = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IS_LIST( hdList ) ) {
        return Error(
          "Add: <list> must be a list",
                     0, 0 );
    }

    /* evaluate and check <val>                                            */
    hdVal = EVAL( PTR_BAG(hdCall)[2] );
    if ( hdVal == HdVoid ) {
        return Error(
          "Add: function must return a value",
                     0, 0 );
    }

    /* add <val> to <list>                                                 */
    if ( GET_TYPE_BAG(hdList) == T_LIST ) {
        pos = LEN_PLIST( hdList ) + 1;
        plen = PLEN_SIZE_PLIST( GET_SIZE_BAG( hdList ) );
        if ( plen + plen/8 + 4 < pos )
            Resize( hdList, SIZE_PLEN_PLIST( pos ) );
        else if ( plen < pos )
            Resize( hdList, SIZE_PLEN_PLIST( plen + plen/8 + 4 ) );
        SET_LEN_PLIST( hdList, pos );
        SET_ELM_PLIST( hdList, pos, hdVal );
    }
    else {
        pos = LEN_LIST( hdList ) + 1;
        ASS_LIST( hdList, pos, hdVal );
    }

    /* return nothing                                                      */
    return HdVoid;
}

void  ListAdd (Obj hdList, Obj hdVal) {
    Int                pos;            /* position to assign to           */
    Int                plen;           /* physical length of <list>       */

    if ( ! IS_LIST( hdList ) )
        Error("ListAdd: <list> must be a list",  0, 0 );

    /* add <val> to <list>                                                 */
    if ( GET_TYPE_BAG(hdList) == T_LIST ) {
        pos = LEN_PLIST( hdList ) + 1;
        plen = PLEN_SIZE_PLIST( GET_SIZE_BAG( hdList ) );
        if ( plen + plen/8 + 4 < pos )
            Resize( hdList, SIZE_PLEN_PLIST( pos ) );
        else if ( plen < pos )
            Resize( hdList, SIZE_PLEN_PLIST( plen + plen/8 + 4 ) );
        SET_LEN_PLIST( hdList, pos );
        SET_ELM_PLIST( hdList, pos, hdVal );
    }
    else {
        pos = LEN_LIST( hdList ) + 1;
        ASS_LIST( hdList, pos, hdVal );
    }
}

/****************************************************************************
**
*F  FunRemoveLast(<hdCall>)  . . . . remove elements from the end of the list
**
**  'FunRemoveLast' implements the internal function 'RemoveLast(<list>,<N>)'.
**
**  'RemoveLast' removes <N> elements from the end of the list.
**  'RemoveLast' returns nothing, it is called only for its sideeffect.
*/
Bag       FunRemoveLast (Bag hdCall)
{
    Bag           hdList;         /* <list>, first argument          */
    Bag           hdN;            /* <N>, second argument            */
    Int                N;
    Int                plen;           /* physical length of <list>       */
    Int                len;

    /* check the argument count                                            */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )
        return Error("usage: RemoveLast( <list>, <N> )", 0, 0 );

    /* evaluate <list>                                                     */
    hdList = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IS_LIST(hdList) )
        return Error("RemoveLast: <list> must be a plain list", 0, 0);

    PLAIN_LIST(hdList);

    /* evaluate and check <N>                                              */
    hdN = EVAL( PTR_BAG(hdCall)[2] );
    if ( hdN == HdVoid )
        return Error("RemoveLast: function must return a value", 0, 0 );
    else if ( GET_TYPE_BAG(hdN) != T_INT  ||  HD_TO_INT(hdN) < 0)
        return Error("RemoveLast: <N> must be a positive integer", 0, 0);

    N = HD_TO_INT(hdN);

    if ( GET_TYPE_BAG(hdList) == T_LIST ) {
        len = LEN_PLIST(hdList) - N;
        if ( len < 0 ) len = 0;
        plen = PLEN_SIZE_PLIST( GET_SIZE_BAG(hdList) );
        if ( len + len/8 + 4 < plen ) {
            /*Pr("plen was %d, resizing to %d\n", plen, len+len/8+4);*/
            Resize( hdList, SIZE_PLEN_PLIST(len + len/8 + 4) );
        }
        SET_LEN_PLIST( hdList, len );
    }

    /* return nothing                                                      */
    return HdVoid;
}

/****************************************************************************
**
*F  FunAppend(<hdCall>) . . . . . . . . . . . . . . append elements to a list
**
**  'FunAppend' implements the function 'Append(<list1>,<list2>)'.
**
**  'Append' adds (see "Add") the elements of the list <list2>  to the end of
**  the list <list1>.   It is allowed that  <list2> contains empty positions,
**  in which case the corresponding positions  will be left empty in <list1>.
**  'Append' returns nothing, it is called only for its side effect.
**
*N  1992/12/10 martin 'Append' should use 'ASSS_LIST'
*/

Bag Append(Bag hdList1, Bag hdList2) {
    Int                lenList1;       /* length of the first list        */
    Int                lenList2;       /* length of the second list       */
    Bag           hdElm;          /* one element of the second list  */
    Int                plen;           /* physical size of the list       */
    Int                i;              /* loop variable                   */
    Int                is_string = 0;
    is_string = (GET_TYPE_BAG(hdList1)==T_STRING) && (GET_TYPE_BAG(hdList2)==T_STRING);

    /* check the arguments                                                 */
    /* check the first list, if neccessary convert to a list  */
    if ( ! IS_LIST( hdList1 ) ) {
        return Error(
          "Append: <list1> must be a list",
                     0,0);
    }
    PLAIN_LIST( hdList1 );
    Retype( hdList1, T_LIST );
    lenList1 = LEN_PLIST( hdList1 );

    /* if neccessary convert the second list to a list           */
    if ( ! IS_LIST( hdList2 ) ) {
        return Error(
          "Append: <list2> must be a list",
                     0,0);
    }
    lenList2 = LEN_LIST( hdList2 );

    /* if the list has no room at the end, enlarge it                      */
    if ( 0 < lenList2 ) {
        plen = PLEN_SIZE_PLIST( GET_SIZE_BAG(hdList1) );
        if ( plen + plen/8 + 4 < lenList1 + lenList2 )
            Resize( hdList1, SIZE_PLEN_PLIST( lenList1 + lenList2 ) );
        else if ( plen < lenList1 + lenList2 )
            Resize( hdList1, SIZE_PLEN_PLIST( plen + plen/8 + 4 ) );
        SET_LEN_PLIST( hdList1, lenList1+lenList2 );
    }

    /* add the elements                                                    */
    if ( GET_TYPE_BAG(hdList2) == T_LIST ) {
        for ( i = 1; i <= lenList2; i++ ) {
            hdElm = ELM_PLIST( hdList2, i );
            SET_ELM_PLIST( hdList1, i+lenList1, hdElm );
        }
    }
    else {
        for ( i = 1; i <= lenList2; i++ ) {
            hdElm = ELMF_LIST( hdList2, i );
            SET_ELM_PLIST( hdList1, i+lenList1, hdElm );
        }
    }

    /* NOTE: prefer to avoid using IsString. IsString is a GAP procedure that has
       a side effect which will actually convert string to lists if possible.  */
    if (is_string) IsString(hdList1);

    /* return nothing                                                      */
    return HdVoid;
}

Bag       FunAppend (Bag hdCall)
{
    Bag           hdList1;        /* handle of the first list        */
    Bag           hdList2;        /* handle of the second list       */

    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )
        return Error("usage: Append( <list1>, <list2> )",0,0);

    /* evaluate both lists FIRST to avoid problems when evaluation
       side-effects retype the lists */
    hdList1 = EVAL( PTR_BAG(hdCall)[1] );
    hdList2 = EVAL( PTR_BAG(hdCall)[2] );
    return Append(hdList1, hdList2);
}


/****************************************************************************
**
*F  FunPosition(<hdCall>) . . . . . . . . . . . . . find an element in a list
**
**  'FunPosition' implements the internal function 'Position'
**
**  'Position(<list>,<obj>[,<start>])'.
**
**  'Position' returns the position of the object <obj> in the  list  <list>.
**  'HdFalse' is returned if the object does not occur in the list.
*/
Bag       FunPosition (Bag hdCall)
{
    Bag           hdList;         /* <list>, first argument          */
    Bag           hdVal;          /* <val>, second argument          */
    Int                start;          /* <start> position                */
    Int                k;              /* position of the value in list   */

    /* check the arguments                                                 */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD && GET_SIZE_BAG(hdCall) != 4*SIZE_HD ) {
        return Error(
          "usage: Position( <list>, <obj> )",
                     0,0);
    }

    /* evaluate and check the <start> value, 0 is the default              */
    if ( GET_SIZE_BAG(hdCall) == 4*SIZE_HD ) {
        hdVal = EVAL( PTR_BAG(hdCall)[3] );
        if ( GET_TYPE_BAG(hdVal) != T_INT || HD_TO_INT(hdVal) < 0 ) {
            return Error(
              "Position: <start> must be a nonnegative int",
                         0,0);
        }
        start = HD_TO_INT(hdVal);
    }
    else {
        start = 0;
    }

    /* evaluate and check <val>                                            */
    hdVal = EVAL( PTR_BAG(hdCall)[2] );
    if ( hdVal == HdVoid ) {
        return Error(
          "Position: function must return a value",
                     0,0);
    }

    /* evaluate and check <list>                                           */
    hdList = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IS_LIST( hdList ) ) {
        return Error(
          "Position: <list> must be a list",
                     0,0);
    }

    /*N 1990/12/25 martin must check if <start> is larger than <length>    */
    /* dispatch to the appropriate 'Pos' function                          */
    k = POS_LIST( hdList, hdVal, start );

    /* return result                                                       */
    return (k != 0) ? INT_TO_HD( k ) : HdFalse;
}


/****************************************************************************
**
*F  FunOnPoints(<hdCall>) . . . . . . . . . . . . . . . . operation on points
**
**  'FunOnPoints' implements the internal function 'OnPoints'.
**
**  'OnPoints( <point>, <g> )'
**
**  specifies  the  canonical  default operation.   Passing  this function is
**  equivalent  to  specifying no operation.   This function  exists  because
**  there are places where the operation in not an option.
*/
Bag       FunOnPoints (Bag hdCall)
{
    Bag           hdRes;          /* handle of the image, result     */
    Bag           hdPnt;          /* handle of the point, first arg  */
    Bag           hdElm;          /* handle of the element, 2nd arg  */

    /* get and check the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error("usage: OnPoints( <point>, <g> )",0,0);
    hdPnt = EVAL( PTR_BAG(hdCall)[1] );
    hdElm = EVAL( PTR_BAG(hdCall)[2] );

    /* return the result                                                   */
    hdRes = POW( hdPnt, hdElm );
    return hdRes;
}


/****************************************************************************
**
*F  FunOnPairs(<hdCall>)  . . . . . . . . . . .  operation on pairs of points
**
**  'FunOnPairs' implements the internal function 'OnPairs'.
**
**  'OnPairs( <pair>, <g> )'
**
**  specifies the componentwise operation of  group elements on pairs
**  of points, which are represented by lists of length 2.
*/
Bag       FunOnPairs (Bag hdCall)
{
    Bag           hdRes;          /* handle of the image, result     */
    Bag           hdPair;         /* handle of the pair, first arg   */
    Bag           hdElm;          /* handle of the element, 2nd arg  */
    Bag           hdTmp;          /* temporary handle                */

    /* get and check the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error("usage: OnPairs( <pair>, <g> )",0,0);
    hdPair = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IS_LIST( hdPair ) )
        return Error("OnPairs: <pair> must be a list",0,0);
    hdElm = EVAL( PTR_BAG(hdCall)[2] );

    /* check that the list is a pair                                       */
    if ( LEN_LIST( hdPair ) != 2 )
        return Error("<pair> must be a list of length 2",0,0);

    /* create a new bag for the result                                     */
    hdRes = NewBag( T_LIST, SIZE_PLEN_PLIST( 2 ) );
    SET_LEN_PLIST( hdRes, 2 );

    /* and enter the images of the points into the result bag              */
    hdTmp = POW( ELMF_LIST( hdPair, 1 ), hdElm );
    SET_ELM_PLIST( hdRes, 1, hdTmp );
    hdTmp = POW( ELMF_LIST( hdPair, 2 ), hdElm );
    SET_ELM_PLIST( hdRes, 2, hdTmp );

    /* return the result                                                   */
    return hdRes;
}


/****************************************************************************
**
*F  FunOnTuples(<hdCall>) . . . . . . . . . . . operation on tuples of points
**
**  'FunOnTuples' implements the internal function 'OnTuples'.
**
**  'OnTuples( <tuple>, <elm> )'
**
**  specifies the componentwise  operation  of  group elements  on tuples  of
**  points, which are represented by lists.  'OnPairs' is the special case of
**  'OnTuples' for tuples with two elements.
*/
Bag       FunOnTuples (Bag hdCall)
{
    Bag           hdRes;          /* handle of the image, result     */
    Bag           hdTup;          /* handle of the tuple, first arg  */
    Bag           hdElm;          /* handle of the element, 2nd arg  */
    Bag           hdTmp;          /* temporary handle                */
    UInt       i;              /* loop variable                   */

    /* get and check the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error("usage: OnTuples( <tuple>, <g> )",0,0);
    hdTup = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IS_LIST( hdTup ) )
        return Error("OnTuples: <tuple> must be a list",0,0);
    hdElm = EVAL( PTR_BAG(hdCall)[2] );

    /* special case for permutations                                       */
    if ( GET_TYPE_BAG(hdElm) == T_PERM16 || GET_TYPE_BAG(hdElm) == T_PERM32 ) {
        PLAIN_LIST( hdTup );
        return OnTuplesPerm( hdTup, hdElm );
    }

    /* create a new bag for the result                                     */
    hdRes = NewBag( T_LIST, SIZE_PLEN_PLIST( LEN_LIST(hdTup) ) );
    SET_LEN_PLIST( hdRes, LEN_LIST(hdTup) );

    /* and enter the images of the points into the result bag              */
    for ( i = LEN_LIST(hdTup); 1 <= i; i-- ) {
        hdTmp = POW( ELMF_LIST( hdTup, i ), hdElm );
        SET_ELM_PLIST( hdRes, i, hdTmp );
    }

    /* return the result                                                   */
    return hdRes;
}


/****************************************************************************
**
*F  FunOnSets(<hdCall>) . . . . . . . . . . . . . operation on sets of points
**
**  'FunOnSets' implements the internal function 'OnSets'.
**
**  'OnSets( <tuple>, <elm> )'
**
**  specifies the operation  of group elements  on  sets of points, which are
**  represented by sorted lists of points without duplicates (see "Sets").
*/
Bag       FunOnSets (Bag hdCall)
{
    Bag           hdRes;          /* handle of the image, result     */
    Bag           hdSet;          /* handle of the tuple, first arg  */
    Bag           hdElm;          /* handle of the element, 2nd arg  */
    Bag           hdTmp;          /* temporary handle                */
    UInt       len;            /* logical length of the list      */
    UInt       mutable;        /* the elements are mutable        */
    UInt       h;              /* gap width in the shellsort      */
    UInt       i, k;           /* loop variables                  */

    /* get and check the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error("usage: OnSets( <tuple>, <g> )",0,0);
    hdSet = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hdSet) != T_SET && ! IsSet( hdSet ) )
        return Error("OnSets: <tuple> must be a set",0,0);
    hdElm = EVAL( PTR_BAG(hdCall)[2] );

    /* special case for permutations                                       */
    if ( GET_TYPE_BAG(hdElm) == T_PERM16 || GET_TYPE_BAG(hdElm) == T_PERM32 ) {
        return OnSetsPerm( hdSet, hdElm );
    }

    /* create a new bag for the result                                     */
    len = LEN_LIST(hdSet);
    hdRes = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
    SET_LEN_PLIST( hdRes, 0 );
    mutable = 0;

    /* and enter the images of the points into the result bag              */
    for ( i = 1; i <= len; i++ ) {
        hdTmp = POW( ELMF_LIST( hdSet, i ), hdElm );
        SET_ELM_PLIST( hdRes, i, hdTmp );
        mutable = mutable || (T_LIST <= GET_TYPE_BAG(hdTmp));
    }

    /* sort the set with a shellsort                                       */
    h = 1;  while ( 9*h + 4 < len )  h = 3*h + 1;
    while ( 0 < h ) {
        for ( i = h+1; i <= len; i++ ) {
            hdTmp = ELM_PLIST( hdRes, i );  k = i;
            while ( h < k && LT( hdTmp, ELM_PLIST(hdRes,k-h) ) == HdTrue ) {
                SET_ELM_PLIST( hdRes, k, ELM_PLIST(hdRes,k-h) );
                k -= h;
            }
            SET_ELM_PLIST( hdRes, k, hdTmp );
        }
        h = h / 3;
    }

    /* remove duplicates, shrink bag if possible                           */
    k = 0;
    if ( 0 < len ) {
        hdTmp = ELM_PLIST( hdRes, 1 );
        k = 1;
        for ( i = 2; i <= len; i++ ) {
            if ( EQ( hdTmp, ELM_PLIST( hdRes, i ) ) != HdTrue ) {
                k += 1;
                hdTmp = ELM_PLIST( hdRes, i );
                SET_ELM_PLIST( hdRes, k, hdTmp );
            }
        }
    }

    /* retype and resize the bag if necessary                              */
    if ( ! mutable )
        Retype( hdRes, T_SET );
    if ( k < len )
        Resize( hdRes, SIZE_PLEN_PLIST(k) );
    SET_LEN_PLIST( hdRes, k );

    /* return set                                                          */
    return hdRes;
}


/****************************************************************************
**
*F  FunOnRight(<hdCall>)  . . . .  operation by multiplication from the right
**
**  'FunOnRight' implements the internal function 'OnRight'.
**
**  'OnRight( <point>, <g> )'
**
**  specifies that group elements operate by multiplication from the right.
*/
Bag       FunOnRight (Bag hdCall)
{
    Bag           hdRes;          /* handle of the image, result     */
    Bag           hdPnt;          /* handle of the point, first arg  */
    Bag           hdElm;          /* handle of the element, 2nd arg  */

    /* get and check the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error("usage: OnRight( <point>, <g> )",0,0);
    hdPnt = EVAL( PTR_BAG(hdCall)[1] );
    hdElm = EVAL( PTR_BAG(hdCall)[2] );

    /* return the result                                                   */
    hdRes = PROD( hdPnt, hdElm );
    return hdRes;
}


/****************************************************************************
**
*F  FunOnLeft(<hdCall>) . . . . . . operation by multiplication from the left
**
**  'FunOnLeft' implements the internal function 'OnLeft'.
**
**  'OnLeft( <point>, <g> )'
**
**  specifies that group elements operate by multiplication from the left.
*/
Bag       FunOnLeft (Bag hdCall)
{
    Bag           hdRes;          /* handle of the image, result     */
    Bag           hdPnt;          /* handle of the point, first arg  */
    Bag           hdElm;          /* handle of the element, 2nd arg  */

    /* get and check the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error("usage: OnLeft( <point>, <g> )",0,0);
    hdPnt = EVAL( PTR_BAG(hdCall)[1] );
    hdElm = EVAL( PTR_BAG(hdCall)[2] );

    /* return the result                                                   */
    hdRes = PROD( hdElm, hdPnt );
    return hdRes;
}


/****************************************************************************
**
*F  DepthListx( <vec> ) . . . . . . . . . . . . . . . . . . depth of a vector
*/
Bag DepthListx (Bag hdVec)
{
    Int                pos;            /* current position                */
    Bag           zero;           /* zero element                    */
    Int                len;            /* length of <hdVec>               */
    Bag           tmp;

    /* if <hdVec> is trivial return one                                    */
    len = LEN_LIST(hdVec);
    if ( len == 0 )
        return INT_TO_HD(1);

    /* construct zero                                                      */
    tmp = ELML_LIST(hdVec,1);
    if ( GET_TYPE_BAG(tmp) == T_INT )
        zero = INT_TO_HD(0);
    else
        zero = PROD( INT_TO_HD(0), tmp );

    /* loop over vector and compare                                        */
    for ( pos = 1;  pos <= len;  pos++ ) {
        tmp = ELML_LIST(hdVec,pos);
        if ( T_LIST <= GET_TYPE_BAG(tmp) && GET_TYPE_BAG(tmp) < T_REC )
            return Error( "DepthVector: <list> must be a vector", 0, 0 );
        if ( zero != tmp && EQ( zero, tmp ) == HdFalse )
            break;
    }

    /* and return the position                                             */
    return INT_TO_HD(pos);
}


/****************************************************************************
**
*F  FunDepthVector( <hdCall> )  . . . . . . . internal function 'DepthVector'
*/
Bag (*TabDepthVector[LIST_TAB_SIZE]) ( Bag );

Bag FunDepthVector (Bag hdCall)
{
    Bag           hdVec;  /* 1. argument: finite field vector        */

    /* evaluate and check the argument                                     */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: DepthVector( <vec> )", 0, 0);
    hdVec = EVAL(PTR_BAG(hdCall)[1]);

    /* jump through the table 'TabIntVecFFE'                               */
    return TabDepthVector[XType(hdVec)]( hdVec );
}

Bag CantDepthVector (Bag hdList)
{
    return Error( "DepthVector: <list> must be a vector", 0, 0 );
}

/****************************************************************************
**
*F  FunNewList( <hdCall> )  . . . . . . . internal function 'NewList'
*/

Bag FunNewList (Bag hdCall)
{
    Bag           hdLen;

    /* evaluate and check the argument                                     */
    if ( GET_SIZE_BAG(hdCall) == 2 * SIZE_HD ) {
        hdLen = EVAL(PTR_BAG(hdCall)[1]);
        if (GET_TYPE_BAG(hdLen)==T_INT && HD_TO_INT(hdLen)>=0)
            return NewList(HD_TO_INT(hdLen));
    }
    return Error("usage: NewList( <length> )", 0, 0);
}

/****************************************************************************
**
*F  InitList()  . . . . . . . . . . . . . . . . . initialize the list package
**
**  'InitList' initializes the generic list package.
*/
void            InitList (void)
{
    Int                type1, type2;   /* loop variable                   */

    /* install the error functions into the tables                         */
    for ( type1 = T_VOID; type1 < LIST_TAB_SIZE; type1++ ) {
        TabIsList     [type1] = 0;
        TabLenList    [type1] = CantLenList;
        TabElmList    [type1] = CantElmList;
        TabElmfList   [type1] = CantElmList;
        TabElmlList   [type1] = CantElmList;
        TabElmrList   [type1] = CantElmList;
        TabElmsList   [type1] = CantElmsList;
        TabAssList    [type1] = CantAssList;
        TabAsssList   [type1] = CantAsssList;
        TabPosList    [type1] = CantPosList;
        TabPlainList  [type1] = CantPlainList;
        TabIsDenseList[type1] = NotIsDenseList;
        TabIsPossList [type1] = NotIsPossList;
    }

    /* install tables for gap functions                                    */
    for ( type1 = T_VOID;  type1 < LIST_TAB_SIZE;  type1++ )
        TabDepthVector[type1] = CantDepthVector;
    TabDepthVector[T_LISTX ] = DepthListx;
    TabDepthVector[T_VECTOR] = DepthListx;


    /* install the default functions                                       */
    for ( type1 = T_LIST; type1 < T_REC; type1++ ) {
        EvTab[type1] = EvList;
        PrTab[type1] = PrList;
    }

    /* install the default comparisons                                     */
    for ( type1 = T_LIST; type1 < T_REC; type1++ ) {
        for ( type2 = T_LIST; type2 < T_REC; type2++ ) {
            TabEq[type1][type2] = EqList;
            TabLt[type1][type2] = LtList;
        }
    }

    /* install the extended dispatcher                                     */
    for ( type1 = T_INT; type1 < T_REC; type1++ ) {
        TabSum [type1  ][T_LIST ] = SumList;
        TabSum [type1  ][T_SET  ] = SumList;
        TabSum [type1  ][T_RANGE] = SumList;
        TabSum [T_LIST ][type1  ] = SumList;
        TabSum [T_SET  ][type1  ] = SumList;
        TabSum [T_RANGE][type1  ] = SumList;
        TabDiff[type1  ][T_LIST ] = DiffList;
        TabDiff[type1  ][T_SET  ] = DiffList;
        TabDiff[type1  ][T_RANGE] = DiffList;
        TabDiff[T_LIST ][type1  ] = DiffList;
        TabDiff[T_SET  ][type1  ] = DiffList;
        TabDiff[T_RANGE][type1  ] = DiffList;
        TabProd[type1  ][T_LIST ] = ProdList;
        TabProd[type1  ][T_SET  ] = ProdList;
        TabProd[type1  ][T_RANGE] = ProdList;
        TabProd[T_LIST ][type1  ] = ProdList;
        TabProd[T_SET  ][type1  ] = ProdList;
        TabProd[T_RANGE][type1  ] = ProdList;
        TabQuo [type1  ][T_LIST ] = QuoList;
        TabQuo [type1  ][T_SET  ] = QuoList;
        TabQuo [type1  ][T_RANGE] = QuoList;
        TabQuo [T_LIST ][type1  ] = QuoList;
        TabQuo [T_SET  ][type1  ] = QuoList;
        TabQuo [T_RANGE][type1  ] = QuoList;
        TabMod [type1  ][T_LIST ] = ModList;
        TabMod [type1  ][T_SET  ] = ModList;
        TabMod [type1  ][T_RANGE] = ModList;
        TabMod [T_LIST ][type1  ] = ModList;
        TabMod [T_SET  ][type1  ] = ModList;
        TabMod [T_RANGE][type1  ] = ModList;
        TabPow [type1  ][T_LIST ] = PowList;
        TabPow [type1  ][T_SET  ] = PowList;
        TabPow [type1  ][T_RANGE] = PowList;
        TabPow [T_LIST ][type1  ] = PowList;
        TabPow [T_SET  ][type1  ] = PowList;
        TabPow [T_RANGE][type1  ] = PowList;
        TabComm[type1  ][T_LIST ] = CommList;
        TabComm[type1  ][T_SET  ] = CommList;
        TabComm[type1  ][T_RANGE] = CommList;
        TabComm[T_LIST ][type1  ] = CommList;
        TabComm[T_SET  ][type1  ] = CommList;
        TabComm[T_RANGE][type1  ] = CommList;
    }

    /* install the default operations                                      */
    /* other operations are installed in the vector packages               */
    for ( type1 = T_INT; type1 < T_LIST; type1++ ) {
        TabSum [type1  ][T_LISTX] = SumSclList;
        TabSum [T_LISTX][type1  ] = SumListScl;
        TabDiff[type1  ][T_LISTX] = DiffSclList;
        TabDiff[T_LISTX][type1  ] = DiffListScl;
        TabProd[type1  ][T_LISTX] = ProdSclList;
        TabProd[T_LISTX][type1  ] = ProdListScl;
        TabQuo [T_LISTX][type1  ] = QuoLists;
        TabMod [type1  ][T_LISTX] = ModLists;
    }

    /* install the evaluation function                                     */
    EvTab[T_LISTELM  ] = EvElmList;
    EvTab[T_LISTELML ] = EvElmListLevel;
    EvTab[T_LISTELMS ] = EvElmsList;
    EvTab[T_LISTELMSL] = EvElmsListLevel;
    EvTab[T_LISTASS  ] = EvAssList;
    EvTab[T_LISTASSL ] = EvAssListLevel;
    EvTab[T_LISTASSS ] = EvAsssList;
    EvTab[T_LISTASSSL] = EvAsssListLevel;
    PrTab[T_LISTELM  ] = PrElmList;
    PrTab[T_LISTELML ] = PrElmList;
    PrTab[T_LISTELMS ] = PrElmsList;
    PrTab[T_LISTELMSL] = PrElmsList;
    PrTab[T_LISTASS  ] = PrAssList;
    PrTab[T_LISTASSL ] = PrAssList;
    PrTab[T_LISTASSS ] = PrAssList;
    PrTab[T_LISTASSSL] = PrAssList;
    EvTab[T_IN       ] = EvIn;

    EvTab[T_MULTIASS  ] = EvMultiAss;

    /* install the internal functions                                      */
    InstIntFunc( "IsList",       FunIsList   );
    InstIntFunc( "IsVector",     FunIsVector );
    InstIntFunc( "IsMat",        FunIsMat    );
    InstIntFunc( "Length",       FunLength   );
    InstIntFunc( "Add",          FunAdd      );
    InstIntFunc( "RemoveLast",   FunRemoveLast   );
    InstIntFunc( "Append",       FunAppend   );
    InstIntFunc( "Position",     FunPosition );
    InstIntFunc( "OnPoints",     FunOnPoints );
    InstIntFunc( "OnPairs",      FunOnPairs  );
    InstIntFunc( "OnTuples",     FunOnTuples );
    InstIntFunc( "OnSets",       FunOnSets   );
    InstIntFunc( "OnRight",      FunOnRight  );
    InstIntFunc( "OnLeft",       FunOnLeft   );
    InstIntFunc( "DepthVector",  FunDepthVector  );
    InstIntFunc( "NewList",      FunNewList  );
}
