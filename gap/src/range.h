/****************************************************************************
**
*A  range.h                     GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file  defines the functions that mainly deal with ranges.  As ranges
**  are a special case of lists many things are done in the list package.
**
**  A *range* is  a list without  holes  consisting  of consecutive integers.
**  For the full definition of ranges see chapter "Ranges" in the GAP Manual.
**  Read  also   "More about Ranges"  about  the different  representation of
**  ranges.
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


/****************************************************************************
**
*F  SIZE_PLEN_RANGE(<plen>) . . . . . . size from physical length for a range
**
**  'SIZE_PLEN_RANGE' returns the size that the bag for a range with room for
**  <plen> elements must have.
**
**  Note that 'SIZE_PLEN_RANGE' is a macro, so do not call it with  arguments
**  that have sideeffects.
*/
#define SIZE_PLEN_RANGE(PLEN)           (3 * SIZE_HD)


/****************************************************************************
**
*F  LEN_RANGE(<hdRange>)  . . . . . . . . . . . . . . . . . length of a range
**
**  'LEN_RANGE' returns the logical length  of  the  range <hdRange>,  as a C
**  integer.
**
**  Note that 'LEN_RANGE' is a macro, so do not call it with  arguments  that
**  have sideeffects.
*/
#define LEN_RANGE(LIST)                 HD_TO_INT(PTR_BAG(LIST)[0])


/****************************************************************************
**
*F  SET_LEN_RANGE(<hdRange>,<len>)  . . . . . . . . set the length of a range
**
**  'SET_LEN_RANGE'  sets  the  length  of the range <hdRange>  to  the value
**  <len>, which must be a C integer larger than 1.
**
**  Note that 'SET_LEN_RANGE' is a macro,  so  do not  call it with arguments
**  that have sideeffects.
*/
#define SET_LEN_RANGE(LIST,LEN)         SET_BAG(LIST, 0, INT_TO_HD(LEN))


/****************************************************************************
**
*F  LOW_RANGE(<hdRange>)  . . . . . . . . . . . . .  first element of a range
**
**  'LOW_RANGE'  returns the  first element of  the  range  <hdRange>  as a C
**  integer.
**
**  Note that 'LOW_RANGE' is a macro,  so do not call it  with arguments that
**  have sideeffects.
*/
#define LOW_RANGE(LIST)                 HD_TO_INT(PTR_BAG(LIST)[1])


/****************************************************************************
**
*F  SET_LOW_RANGE(<hdRange>,<low>)  . . . .  set the first element of a range
**
**  'SET_LOW_RANGE' sets the  first  element  of  the range <hdRange>  to the
**  value <low>, which must be a C integer.
**
**  Note  that 'SET_LOW_RANGE' is a macro, so do not call  it with  arguments
**  that have sideeffects.
*/
#define SET_LOW_RANGE(LIST,LOW)         SET_BAG(LIST, 1, INT_TO_HD(LOW))


/****************************************************************************
**
*F  INC_RANCE(<hdRange>)  . . . . . . . . . . . . . . .  increment of a range
**
**  'INC_RANGE' returns the increment of the range <hdRange> as a C integer.
**
**  Note that 'INC_RANGE' is a macros,  so do not call it with arguments that
**  have sideeffects.
*/
#define INC_RANGE(LIST)                 HD_TO_INT(PTR_BAG(LIST)[2])


/****************************************************************************
**
*F  SET_INC_RANGE(<hdRange>,<inc>)  . . . . . .  set the increment of a range
**
**  'SET_INC_RANGE'  sets the  increment of the range <hdRange> to  the value
**  <inc>, which must be a C integer.
**
**  Note that  'SET_INC_RANGE' is a macro,  so do  not call it with arguments
**  that have sideeffects.
*/
#define SET_INC_RANGE(LIST,INC)         SET_BAG(LIST, 2, INT_TO_HD(INC))


/****************************************************************************
**
*F  ELM_RANGE(<hdRange>,<i>)  . . . . . . . . . . . . . .  element of a range
**
**  'ELM_RANGE' return the <i>-th element of the  range <hdRange>.  <i>  must
**  be a positive integer less than or equal to the length of <hdRange>.
**
**  Note that 'ELM_RANGE' is a macro, so do not call it with  arguments  that
**  have sideeffects.
*/
#define ELM_RANGE(L,POS)        INT_TO_HD(LOW_RANGE(L)+(POS-1)*INC_RANGE(L))


/****************************************************************************
**
*F  LenRange(<hdList>)  . . . . . . . . . . . . . . . . . . length of a range
**
**  'LenRange' returns the length of the range <hdList> as a C integer.
**
**  'LenRange' is the function in 'TabLenList' for ranges.
*/
extern  Int            LenRange (
            Bag           hdList );


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
extern  Bag       ElmRange (
            Bag           hdList,
            Int                pos );

extern  Bag       ElmfRange (
            Bag           hdList,
            Int                pos );


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
extern  Bag       ElmsRange (
            Bag           hdList,
            Bag           hdPoss );


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
extern  Bag       AssRange (
            Bag           hdList,
            Int                pos,
            Bag           hdVal );


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
extern  Bag       AsssRange (
            Bag           hdList,
            Bag           hdPoss,
            Bag           hdVals );


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
extern  Int            PosRange (
            Bag           hdList,
            Bag           hdVal,
            Int                start );


/****************************************************************************
**
*F  PlainRange(<hdList>)  . . . . . . . . . . convert a range to a plain list
**
**  'PlainRange' converts the range <hdList> to a plain list.
**
**  'PlainRange' is the function in 'TabPlainList' for ranges.
*/
extern  void            PlainRange (
            Bag           hdList );


/****************************************************************************
**
*F  IsDenseRange(<hdList>)  . . . . . . . dense list test function for ranges
**
**  'IsDenseRange' returns 1, since ranges are always dense.
**
**  'IsDenseRange' is the function in 'TabIsDenseList' for ranges.
*/
extern  Int            IsDenseRange (
            Bag           hdList );


/****************************************************************************
**
*F  IsPossRange(<hdList>) . . . . . . positions list test function for ranges
**
**  'IsPossRange' returns 1 if  the range <hdList> is a dense list containing
**  only positive integers, and 0 otherwise.
**
**  'IsPossRange' is the function in 'TabIsPossList' for ranges.
*/
extern  Int            IsPossRange (
            Bag           hdList );


/****************************************************************************
**
*F  PrRange(<hdRange>)  . . . . . . . . . . . . . . . . . . . . print a range
**
**  'PrRange' prints the range <hdRange>.
**
**  'PrRange' handles bags of type 'T_RANGE' and 'T_MAKERANGE'.
*/
extern  void            PrRange (
            Bag           hdRange );


/****************************************************************************
**
*F  EqRange(<hdL>,<hdR>)  . . . . . . . . . . .  test if two ranges are equal
**
**  'EqRange' returns 'true' if the two  ranges <hdL> and <hdR> are equal and
**  'false' otherwise.
**
**  Is  called from the 'EQ' binop  so both  operands are  already evaluated.
*/
extern  Bag       EqRange (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  LtRange(<hdL>,<hdR>)  . . . . . . . . . . .  test if two ranges are equal
**
**  'LtRange' returns 'true' if the range  <hdL> is less than the range <hdR>
**  and 'false' otherwise.
**
**  Is called from the 'LT' binop so both operands are already evaluated.
*/
extern  Bag       LtRange (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  EvMakeRange(<hdMake>) . . .  convert a variable range into a constant one
**
**  'EvMakeRange' turns the literal  range  <hdMake>  into  a  constant  one.
*/
extern  Bag       EvMakeRange (
            Bag           hdMake );


/****************************************************************************
**
*F  PrMakeRange(<hdMake>) . . . . . . . . . . . . . . . print a range literal
**
**  'PrMakeRange' prints the range literal  <hdMake> in the form '[  <low> ..
**  <high> ]'.
*/
extern  void            PrMakeRange (
            Bag           hdMake );


/****************************************************************************
**
*F  IsRange(<hdList>) . . . . . . . . . . . . . . . test if a list is a range
**
**  'IsRange' returns 1 if the list with the handle <hdList> is a range and 0
**  otherwise.  As a  sideeffect 'IsRange' converts proper ranges represented
**  the ordinary way to the compact representation.
*/
extern  Int            IsRange (
            Bag           hdList );


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
extern  Bag       FunIsRange (
            Bag           hdCall );


/****************************************************************************
**
*F  InitRange() . . . . . . . . . . . . . . . .  initialize the range package
**
**  'InitRange' initializes the range package.
*/
extern  void            InitRange ( void );


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



