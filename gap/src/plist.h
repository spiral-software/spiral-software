/****************************************************************************
**
*A  plist.h                     GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This  file declares  the  functions  that  deal  with  plain lists.   The
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
*/
#define PLEN_SIZE_PLIST(GET_SIZE_BAG)           (((GET_SIZE_BAG) - SIZE_HD) / SIZE_HD)


/****************************************************************************
**
*F  SIZE_PLEN_PLIST(<plen>)  size for a plain list with given physical length
**
**  'SIZE_PLEN_PLIST' returns the size that a plain list with room for <plen>
**  elements must at least have.
**
**  Note that 'SIZE_PLEN_PLIST' is a macro, so do not call it with  arguments
**  that have sideeffects.
*/
#define SIZE_PLEN_PLIST(PLEN)           (SIZE_HD + (PLEN) * SIZE_HD)


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
*/
#define LEN_PLIST(LIST)                 (HD_TO_INT(PTR_BAG(LIST)[0]))


/****************************************************************************
**
*F  SET_LEN_PLIST(<hdList>,<len>) . . . . . .  set the length of a plain list
**
**  'SET_LEN_PLIST' sets the length of the plain list <hdList> to <len>.  The
**  length is stored as GAP immediate integer as the zeroeth handle.
**
**  Note  that 'SET_LEN_PLIST'  is a macro, so do not call it with  arguments
**  that have sideeffects.
*/
#define SET_LEN_PLIST(LIST,LEN)         (SET_BAG(LIST, 0, INT_TO_HD(LEN)))


/****************************************************************************
**
*F  ELM_PLIST(<hdList>,<pos>) . . . . . . . . . . . . element of a plain list
**
**  'ELM_PLIST' return the <pos>-th element of the list <hdList>.  <pos> must
**  be a positive integer less than or equal to the length of <hdList>.
**
**  Note that  'ELM_PLIST' is a macro, so do  not call it with arguments that
**  have sideeffects.
*/
#define ELM_PLIST(LIST,POS)             (PTR_BAG(LIST)[POS])


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
*/
#define SET_ELM_PLIST(LIST,POS,VAL)     (SET_BAG((LIST), (POS), (VAL)))


/****************************************************************************
**
*F  LenPlist(<hdList>)  . . . . . . . . . . . . . . .  length of a plain list
**
**  'LenPlist' returns the length of the plain list <hdList> as a C integer.
**
**  'LenPlist' is the function in 'TabLenList' for plain lists.
*/
extern  Int            LenPlist (
            Bag           hdList );


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
extern  Bag       ElmPlist (
            Bag           hdList,
            Int                pos );

extern  Bag       ElmfPlist (
            Bag           hdList,
            Int                pos );


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
extern  Bag       ElmsPlist (
            Bag           hdList,
            Bag           hdPoss );


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
extern  Bag       AssPlist (
            Bag           hdList,
            Int                pos,
            Bag           hdVal );


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
extern  Bag       AsssPlist (
            Bag           hdList,
            Bag           hdPoss,
            Bag           hdVals );


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
extern  Int            PosPlist (
            Bag           hdList,
            Bag           hdVal,
            Int                start );


/****************************************************************************
**
*F  PlainPlist(<hdList>)  . . . . . . .  convert a plain list to a plain list
**
**  'PlainPlist' converts the plain list <hdList> to a plain list.  Not  much
**  work.
**
**  'PlainPlist' is the function in 'TabPlainList' for plain lists.
*/
extern  void            PlainPlist (
            Bag           hdList );


/****************************************************************************
**
*F  IsDensePlist(<hdList>)  . . . .  dense list test function for plain lists
**
**  'IsDensePlist' returns 1 if the plain list <hdList> is a dense list and 0
**  otherwise.
**
**  'IsDensePlist' is the function in 'TabIsDenseList' for plain lists.
*/
extern  Int            IsDensePlist (
            Bag           hdList );


/****************************************************************************
**
*F  IsPossPlist(<hdList>) . . .  positions list test function for plain lists
**
**  'IsPossPlist' returns  1 if  the plain  list  <hdList>  is  a  dense list
**  containing only positive integers, and 0 otherwise.
**
**  'IsPossPlist' is the function in 'TabIsPossList' for plain lists.
*/
extern  Int            IsPossPlist (
            Bag           hdList );


/****************************************************************************
**
*F  EqPlist(<hdL>,<hdR>) . . . . . . . . .  test if two plain lists are equal
**
**  'EqList' returns 'true' if the two plain lists <hdL> and <hdR> are  equal
**  and 'false' otherwise.
**
**  Is called from the 'EQ' binop so both  operands  are  already  evaluated.
*/
extern  Bag       EqPlist (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  LtPlist(<hdL>,<hdR>)  . . . . . . . . . test if two plain lists are equal
**
**  'LtList' returns 'true' if  the  plain list <hdL>  is less than the plain
**  list <hdR> and 'false' otherwise.
**
**  Is called from the 'LT' binop so both operands are already evaluated.
*/
extern  Bag       LtPlist (
            Bag           hdL,
            Bag           hdR );


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
extern  Bag       EvMakeList (
            Bag           hdLiteral );


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
extern  Bag       MakeList (
            Bag           hdDst,
            Int                ind,
            Bag           hdLiteral );


/****************************************************************************
**
*F  InitPlist() . . . . . . . . . . . . . . . . . initialize the list package
**
**  Is called during  the  initialization  to  initialize  the  list package.
*/
extern  void            InitPlist ( void );


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



