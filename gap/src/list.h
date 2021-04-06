/****************************************************************************
**
*A  list.h                      GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file  declares  the  functions  and macros  that make it possible to
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
*/

#define LIST_TAB_SIZE T_NAMESPACE+1

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
*/
#define IS_LIST(LIST)           (GET_TYPE_BAG(LIST) < T_VAR && TabIsList[GET_TYPE_BAG(LIST)] != 0)
#define IS_LIST_TYPE(t)           (t < T_VAR && TabIsList[t] != 0)

extern  Int            TabIsList [LIST_TAB_SIZE];


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
*/
#define LEN_LIST(LIST)          ((*TabLenList[GET_TYPE_BAG(LIST)])(LIST))

extern  Int            (*TabLenList[LIST_TAB_SIZE]) ( Bag );

extern  Int            CantLenList (
            Bag           hdList );


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
*/
#define ELM_LIST(LIST,POS)      ((*TabElmList[GET_TYPE_BAG(LIST)])(LIST,POS))
#define ELMF_LIST(LIST,POS)     ((*TabElmfList[GET_TYPE_BAG(LIST)])(LIST,POS))
#define ELML_LIST(LIST,POS)     ((*TabElmlList[GET_TYPE_BAG(LIST)])(LIST,POS))
#define ELMR_LIST(LIST,POS)     ((*TabElmrList[GET_TYPE_BAG(LIST)])(LIST,POS))

extern  Bag       (*TabElmList[LIST_TAB_SIZE])  ( Bag, Int );
extern  Bag       (*TabElmfList[LIST_TAB_SIZE]) ( Bag, Int );
extern  Bag       (*TabElmlList[LIST_TAB_SIZE]) ( Bag, Int );
extern  Bag       (*TabElmrList[LIST_TAB_SIZE]) ( Bag, Int );

extern  Bag       CantElmList (
            Bag           hdList,
            Int                pos );


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
*/
#define ELMS_LIST(LIST,POSS)    ((*TabElmsList[GET_TYPE_BAG(LIST)])(LIST,POSS))

extern  Bag       (*TabElmsList[LIST_TAB_SIZE]) ( Bag, Bag );

extern  Bag       CantElmsList (
            Bag           hdList,
            Bag           hdPoss );


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
*/
#define ASS_LIST(LIST,POS,VAL)  ((*TabAssList[GET_TYPE_BAG(LIST)])(LIST,POS,VAL))

extern  Bag       (*TabAssList[LIST_TAB_SIZE]) ( Bag,
                                                 Int,
                                                 Bag );

extern  Bag       CantAssList (
            Bag           hdList,
            Int                pos,
            Bag           hdVal );


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
*/
#define ASSS_LIST(LI,POSS,VALS) ((*TabAsssList[GET_TYPE_BAG(LI)])(LI,POSS,VALS))

extern  Bag       (*TabAsssList[LIST_TAB_SIZE]) ( Bag,
                                                  Bag,
                                                  Bag );

extern  Bag       CantAsssList (
            Bag           hdList,
            Bag           hdPoss,
            Bag           hdVals );


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
*/
#define POS_LIST(LIST,VAL,START)  ((*TabPosList[GET_TYPE_BAG(LIST)])(LIST,VAL,START))

extern  Int            (*TabPosList[LIST_TAB_SIZE]) ( Bag,
                                                 Bag,
                                                 Int );

extern  Int            CantPosList (
            Bag           hdList,
            Bag           hdVal,
            Int                start );


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
*/
#define PLAIN_LIST(LIST)          ((*TabPlainList[GET_TYPE_BAG(LIST)])(LIST))

extern  void            (*TabPlainList[LIST_TAB_SIZE]) ( Bag );

extern  void            CantPlainList (
            Bag           hdList );

/*N 1992/12/11 martin only here for backward compatibility                 */
extern  Int            IsList (
            Bag           hdObj );


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
*/
#define IS_DENSE_LIST(LIST)     ((*TabIsDenseList[GET_TYPE_BAG(LIST)])(LIST))

extern  Int            (*TabIsDenseList[LIST_TAB_SIZE]) ( Bag );

extern  Int            NotIsDenseList (
            Bag           hdObj );


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
*/
#define IS_POSS_LIST(LIST)      ((*TabIsPossList[GET_TYPE_BAG(LIST)])(LIST))

extern  Int            (*TabIsPossList[LIST_TAB_SIZE]) ( Bag );

extern  Int            NotIsPossList (
            Bag           hdObj );


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
*/
#define IS_XTYPE_LIST(T,LIST)   ((*TabIsXTypeList[T])(LIST))

extern  Int            (*TabIsXTypeList[LIST_TAB_SIZE]) ( Bag );

extern  Int            XType (
            Bag           hdObj );


/****************************************************************************
**
*F  EvList(<hdList>)  . . . . . . . . . . . . . . . . . . . . evaluate a list
**
**  'EvList' returns the value of the list <hdList>.  The  value of a list is
**  just the list itself, since lists are constants and thus selfevaluating.
*/
extern  Bag       EvList (
            Bag           hdList );


/****************************************************************************
**
*F  PrList(<hdList>)  . . . . . . . . . . . . . . . . . . . . .  print a list
**
**  'PrList' prints the list <hdList>.  It is a generic function that can  be
**  used for all kinds of lists.
**
**  Linebreaks are preferred after the commas.
*/
extern  void            PrList (
            Bag           hdList );


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
extern  Bag       EqList (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  LtList(<hdL>,<hdR>) . . . . . . . . . . . . . test if two lists are equal
**
**  'LtList' returns 'true' if the list <hdL> is less than the list <hdR> and
**  'false' otherwise.
**
**  Is called from the 'LT' binop so both operands are already evaluated.
*/
extern  Bag       LtList (
            Bag           hdL,
            Bag           hdR );


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
extern  Bag       SumList (
            Bag           hdL,
            Bag           hdR );

extern  Bag       SumSclList (
            Bag           hdL,
            Bag           hdR );

extern  Bag       SumListScl (
            Bag           hdL,
            Bag           hdR );

extern  Bag       SumListList (
            Bag           hdL,
            Bag           hdR );


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
extern  Bag       DiffList (
            Bag           hdL,
            Bag           hdR );

extern  Bag       DiffSclList (
            Bag           hdL,
            Bag           hdR );

extern  Bag       DiffListScl (
            Bag           hdL,
            Bag           hdR );

extern  Bag       DiffListList (
            Bag           hdL,
            Bag           hdR );


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
extern  Bag       ProdList (
            Bag           hdL,
            Bag           hdR );

extern  Bag       ProdSclList (
            Bag           hdL,
            Bag           hdR );

extern  Bag       ProdListScl (
            Bag           hdL,
            Bag           hdR );

extern  Bag       ProdListList (
            Bag           hdL,
            Bag           hdR );


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
extern  Bag       QuoList (
            Bag           hdL,
            Bag           hdR );

extern  Bag       QuoLists (
            Bag           hdL,
            Bag           hdR );


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
extern  Bag       ModList (
            Bag           hdL,
            Bag           hdR );

extern  Bag       ModLists (
            Bag           hdL,
            Bag           hdR );


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
extern  Bag       PowList (
            Bag           hdL,
            Bag           hdR );

extern  Bag       PowLists (
            Bag           hdL,
            Bag           hdR );


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
extern  Bag       CommList (
            Bag           hdL,
            Bag           hdR );

extern  Bag       CommLists (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  EvElmList(<hdSel>)  . . . . . . . . . . . . . select an element of a list
**
**  'EvElmList' returns the value of the element at the position '<hdSel>[1]'
**  in the list '<hdSel>[0]'.   Both '<hdSel>[0]' and '<hdSel>[1]' first have
**  to be evaluated.
*/
extern  Bag       EvElmList (
            Bag           hdSel );


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
extern  Bag       EvElmListLevel (
            Bag           hdSel );


/****************************************************************************
**
*F  EvElmsList(<hdSel>) . . . . . . . . . . . . .  select a sublist of a list
**
**  'EvElmsList' returns the sublist of the elements at the  positions  given
**  in the list '<hdSel>[1]' in the list '<hdSel>[0]'.  Both '<hdSel>[0]' and
**  '<hdSel>[1]' first have to be evaluated.  This implements  the  construct
**  '<list>{<positions>}'.
*/
extern  Bag       EvElmsList (
            Bag           hdSel );


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
extern  Bag       EvElmsListLevel (
            Bag           hdSel );


/****************************************************************************
**
*F  EvAssList(<hdAss>)  . . . . . . . . . . .  assign to an element of a list
**
**  'EvAssList'  assigns   the  object  '<hdAss>[1]'    to  the  list element
**  '<hdAss>[0]',  i.e., to  the element  at  position '<hdAss>[0][1]' in the
**  list '<hdAss>[0][0]'.
*/
extern  Bag       EvAssList (
            Bag           hdAss );


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
extern  Bag       EvAssListLevel (
            Bag           hdAss );


/****************************************************************************
**
*F  EvAsssList(<hdAss>) . . . . . . . . . . . . assign to a sublist of a list
**
**  'EvAssList'  assigns   the  object  '<hdAss>[1]'    to  the  list sublist
**  '<hdAss>[0]',  i.e., to  the elements at positions '<hdAss>[0][1]' in the
**  list '<hdAss>[0][0]'.
*/
extern  Bag       EvAsssList (
            Bag           hdAss );


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
extern  Bag       EvAsssListLevel (
            Bag           hdAss );


/****************************************************************************
**
*F  PrElmList(<hdSel>)  . . . . . . . . . . . . . . .  print a list selection
**
**  'PrElmList' prints the list selection <hdSel>.
**
**  Linebreaks are preferred after the '['.
*/
extern  void            PrElmList (
            Bag           hdSel );


/****************************************************************************
**
*F  PrElmsList(<hdSel>) . . . . . . . . . . . . . . .  print a list selection
**
**  'PrElmsList' prints the list selection <hdSel>.
**
**  Linebreaks are preferred after the '{'.
*/
extern  void            PrElmsList (
            Bag           hdSel );


/****************************************************************************
**
*F  PrAssList(<hdAss>)  . . . . . . . . print an assignment to a list element
**
**  'PrAssList' prints the assignment to a list element.
**
**  Linebreaks are preferred before the ':='.
*/
extern  void            PrAssList (
            Bag           hdAss );


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
extern  Bag       EvIn (
            Bag           hdIn );


/****************************************************************************
**
*F  FunIsList(<hdCall>) . . . . . . . . . . . . . . . . . . .  test for lists
**
**  'FunIsList' implements the internal function 'IsList( <obj> )'.
**
**  'IsList'  returns  'true' if its argument   <obj> is  a  list and 'false'
**  otherwise.
*/
extern  Bag       FunIsList (
            Bag           hdCall );


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
extern  Int            IsVector (
            Bag           hdObj );

extern  Bag       FunIsVector (
            Bag           hdCall );


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
extern  Bag       FunIsMat (
            Bag           hdCall );


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
extern  Bag       FunLength (
            Bag           hdCall );


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
extern  Bag       FunAdd (
            Bag           hdCall );

/* Same as above, but for use in C code functions */
extern  void            ListAdd (Obj hdList, Obj hdVal);

/****************************************************************************
**
*F  FunRemoveLast(<hdCall>)  . . . . remove elements from the end of the list
**
**  'FunRemoveLast' implements the internal function 'RemoveLast(<list>,<N>)'.
**
**  'RemoveLast' removes <N> elements from the end of the list.
**  'RemoveLast' returns nothing, it is called only for its sideeffect.
*/
extern  Bag       FunRemoveLast (
            Bag           hdCall );


/****************************************************************************
**
*F  Append(<hdList1>, <hdList2>) . . . . . . . . .  append elements to a list
*/
extern  Bag      Append ( Bag hdList1, Bag hdList2 );

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
*/
extern  Bag       FunAppend (
            Bag           hdCall );


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
extern  Bag       FunPosition (
            Bag           hdCall );


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
extern  Bag       FunOnPoints (
            Bag           hdCall );


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
extern  Bag       FunOnPairs (
            Bag           hdCall );


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
extern  Bag       FunOnTuples (
            Bag           hdCall );


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
extern  Bag       FunOnSets (
            Bag           hdCall );


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
extern  Bag       FunOnRight (
            Bag           hdCall );


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
extern  Bag       FunOnLeft (
            Bag           hdCall );


/****************************************************************************
**
*F  DepthListx( <vec> ) . . . . . . . . . . . . . . . . . . depth of a vector
*/
extern Bag DepthListx (
    Bag           hdVec );


/****************************************************************************
**
*F  FunDepthVector( <hdCall> )  . . . . . . . internal function 'DepthVector'
*/
extern Bag (*TabDepthVector[LIST_TAB_SIZE]) ( Bag );

extern Bag FunDepthVector (
                Bag       hdCall );


/****************************************************************************
**
*F  InitList()  . . . . . . . . . . . . . . . . . initialize the list package
**
**  'InitList' initializes the generic list package.
*/
extern  void            InitList ( );
