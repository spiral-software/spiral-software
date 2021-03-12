/****************************************************************************
**
*A  set.h                       GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file  declares  the functions that  mainly operate  on  proper sets.
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


/****************************************************************************
**
*F  LenSet(<hdList>)  . . . . . . . . . . . . . . . . . . . . length of a set
**
**  'LenSet' returns the length of the set <hdList> as a C integer.
**
**  'LenSet' is the function in 'TabLenList' for sets.
*/
extern  Int            LenSet (
            Bag           hdList );


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
extern  Bag       ElmSet (
            Bag           hdList,
            Int                pos );

extern  Bag       ElmfSet (
            Bag           hdList,
            Int                pos );


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
extern  Bag       ElmsSet (
            Bag           hdList,
            Bag           hdPoss );


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
extern  Bag       AssSet (
            Bag           hdList,
            Int                pos,
            Bag           hdVal );


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
extern  Bag       AsssSet (
            Bag           hdList,
            Bag           hdPoss,
            Bag           hdVals );


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
extern  Int            PosSet (
            Bag           hdList,
            Bag           hdVal,
            Int                start );


/****************************************************************************
**
*F  PlainSet(<hdList>)  . . . . . . . . . . . . convert a set to a plain list
**
**  'PlainSet' converts the set <hdList> to a plain list.  Not much work.
**
**  'PlainSet' is the function in 'TabPlainList' for sets.
*/
extern  void            PlainSet (
            Bag           hdList );


/****************************************************************************
**
*F  IsDenseSet(<hdList>)  . . . . . . . . . dense list test function for sets
**
**  'IsDenseSet' returns 1, since every set is dense.
**
**  'IsDenseSet' is the function in 'TabIsDenseList' for sets.
*/
extern  Int            IsDenseSet (
            Bag           hdList );


/****************************************************************************
**
*F  IsPossSet(<hdList>) . . . . . . . . positions list test function for sets
**
**  'IsPossSet' returns 1 if the set <hdList> is a dense list containing only
**  positive integers, and 0 otherwise.
**
**  'IsPossSet' is the function in 'TabIsPossList' for sets.
*/
extern  Int            IsPossSet (
            Bag           hdList );


/****************************************************************************
**
*F  EqSet(<hdL>,<hdR>)  . . . . . . . . . . . . .  test if two sets are equal
**
**  'EqList' returns  'true' if  the two sets <hdL> and  <hdR>  are equal and
**  'false' otherwise.
**
**  Is called from the 'EQ' binop so both  operands  are  already  evaluated.
*/
extern  Bag       EqSet (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  LtSet(<hdL>,<hdR>)  . . . . . . . . . . . . .  test if two sets are equal
**
**  'LtSet'  returns 'true' if the  set <hdL> is less than the set <hdR>  and
**  'false' otherwise.
**
**  Is called from the 'LT' binop so both operands are already evaluated.
*/
extern  Bag       LtSet (
            Bag           hdL,
            Bag           hdR );


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
extern  Bag       SetList (
            Bag           hdList );


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
extern  Bag       FunSet (
            Bag           hdCall );


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
extern  Int            IsSet (
            Bag           hdList );


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
extern  Bag       FunIsSet (
            Bag           hdCall );


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
extern  Bag       FunIsEqualSet (
            Bag           hdCall );


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
extern  Bag       FunIsSubsetSet (
            Bag           hdCall );


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
extern  Bag       FunAddSet (
            Bag           hdCall );


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
extern  Bag       FunRemoveSet (
            Bag           hdCall );


/****************************************************************************
**
*F  FunUniteSet(<hdCall>) . . . . . . . . . . . .  unite one set with another
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
*/
extern  Bag       FunUniteSet (
            Bag           hdCall );


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
extern  Bag       FunIntersectSet (
            Bag           hdCall );


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
extern  Bag       FunSubtractSet (
            Bag           hdCall );


/****************************************************************************
**
*F  InitSet() . . . . . . . . . . . . . . . . . .  initialize the set package
**
**  'InitSet' initializes the set package.
*/
extern  void            InitSet ( void );


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



