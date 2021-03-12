/****************************************************************************
**
*A  string.h                    GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This  file  declares  the  functions  which  mainly  deal  with  strings.
**  (This is the remainder of the once important 'evbasic' package).
**
**  A *string* is a  list that  has no  holes, and  whose  elements  are  all
**  characters.  For the full definition of strings see chapter  "Strings" in
**  the {\GAP} manual.  Read also "More about Strings" about the  string flag
**  and the compact representation of strings.
**
**  This package consists of three parts.
**
**  The first part  consists of  the macros 'SIZE_PLEN_STRING', 'LEN_STRING',
**  'ELM_STRING',  and 'SET_ELM_STRING'.   They determine the respresentation
**  of strings.   For historical  reasons  however  other parts of the {\GAP}
**  kernel also know about the representation of strings.
**
**  The second part  consists  of  the  functions  'LenString',  'ElmString',
**  'ElmsStrings', 'AssString',  'AsssString', PlainString', 'IsDenseString',
**  and 'IsPossString'.  They are the functions requried by the generic lists
**  package.  Using these functions the other  parts of the {\GAP} kernel can
**  access and  modify strings  without actually  being aware  that they  are
**  dealing with a string.
**
**  The third part consists  of the functions 'PrintString', which is  called
**  by 'FunPrint', and 'IsString', which test whether an arbitrary list  is a
**  string, and if so converts it into the above format.
**
*/


/****************************************************************************
**
*V  HdChars[<chr>]  . . . . . . . . . . . . . . . . . table of character bags
**
**  'HdChars' contains the handles of all the character objects.  That way we
**  dont need to allocate new bags for new characters.
*/
extern  Bag               HdChars [256];


/****************************************************************************
**
*F  EvChar( <hdChr> ) . . . . . . . . . . . . . evaluate a character constant
**
**  'EvChar' returns  the value  of the  character constant  <hdChr>.   Since
**  characters are  constant and thus  selfevaluating, 'EvChar'  just returns
**  <hdChr>.
*/
extern  Bag       EvChar (
            Bag           hdChr );


/****************************************************************************
**
*F  EqChar( <hdL>, <hdR> )  . . . . . . . . . . . . .  compare two characters
**
**  'EqChar'  returns  'HdTrue'  if the  two  characters <hdL> and  <hdR> are
**  equal, and 'HdFalse' otherwise.
**
**  Is called from the 'Eq' binop, so both operands are already evaluated.
*/
extern  Bag       EqChar (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  LtChar( <hdL>, <hdR> )  . . . . . . . . . . . . .  compare two characters
**
**  'LtChar'  returns  'HdTrue'  if  the character  <hdL>  is  less  than the
**  character <hdR>, and 'HdFalse' otherwise.
**
**  Is called from the 'Lt' binop, so both operands are already evaluated.
*/
extern  Bag       LtChar (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  PrChar( <hdChr> ) . . . . . . . . . . . . . . . . . . . print a character
**
**  'PrChar' prints the character <hdChr>.
*/
extern  void            PrChar (
            Bag           hdChr );


/****************************************************************************
**
*F  SIZE_PLEN_STRING(<plen>)  . . . .  size from physical length for a string
**
**  'SIZE_PLEN_STRING' returns  the size that the  bag for a string with room
**  for <plen> elements must have.
**
**  Note that 'SIZE_PLEN_STRING' is a macro, so do not call it with arguments
**  that have sideeffects.
*/
#define SIZE_PLEN_STRING(PLEN)          (PLEN + NUM_TO_UINT(1))


/****************************************************************************
**
*F  LEN_STRING(<hdList>)  . . . . . . . . . . . . . . . .  length of a string
**
**  'LEN_STRING' returns the length of the string <hdList>, as a C integer.
**
**  Note that  'LEN_STRING' is a macro, so do not call it with arguments that
**  have sideeffects.
*/
#define LEN_STRING(LIST)                (GET_SIZE_BAG(LIST)-1)


/****************************************************************************
**
*F  ELM_STRING(<hdList>,<pos>)  . . . . . . . . select an element of a string
**
**  'ELM_STRING'  returns the <pos>-th element of the string <hdList>.  <pos>
**  must be a positive integer less than or equal to the length of <hdList>.
**
**  Note that 'ELM_STRING' is a  macro, so do not call it with arguments that
**  have sideeffects.
*/
#define ELM_STRING(LIST,POS)    (HdChars[((unsigned char*)PTR_BAG(LIST))[POS-1]])


/****************************************************************************
**
*F  LenString(<hdList>) . . . . . . . . . . . . . . . . .  length of a string
**
**  'LenString' returns the length of the string <hdList> as a C integer.
**
**  'LenString' is the function in 'TabLenList' for strings.
*/
extern  Int            LenString (
            Bag           hdList );


/****************************************************************************
**
*F  ElmString(<hdList>,<pos>) . . . . . . . . . select an element of a string
**
**  'ElmString' selects  the  element  at the position <pos>  of  the  string
**  <hdList>.  It is the responsibility of the caller to ensure that <pos> is
**  a positive integer.   An error is signalled if  <pos> is larger than  the
**  length of <hdList>.
**
**  'ElmfString'  does  the  same thing than 'ElmString', but need  not check
**  that <pos> is less  than or equal to the length  of <hdList>, this is the
**  responsibility of the caller.
**
**  'ElmString' is the function in 'TabElmList' for strings.  'ElmfString' is
**  the  function  in  'TabElmfList', 'TabElmlList',  and  'TabElmrList'  for
**  strings.
*/
extern  Bag       ElmString (
            Bag           hdList,
            Int                pos );

extern  Bag       ElmfString (
            Bag           hdList,
            Int                pos );


/****************************************************************************
**
*F  ElmsString(<hdList>,<hdPoss>) . . . . . .  select a sublist from a string
**
**  'ElmsString' returns a new list containing the elements  at the positions
**  given  in  the  list  <hdPoss>  from  the  string  <hdList>.   It is  the
**  responsibility  of  the  called  to  ensure  that  <hdPoss> is dense  and
**  contains  only positive integers.  An error is signalled if an element of
**  <hdPoss> is larger than the length of <hdList>.
**
**  'ElmsString' is the function in 'TabElmsList' for strings.
*/
extern  Bag       ElmsString (
            Bag           hdList,
            Bag           hdPoss );


/****************************************************************************
**
*F  AssString(<hdList>,<pos>,<hdVal>)  . . . . . . . . . . assign to a string
**
**  'AssString'  assigns the value  <hdVal> to  the  string <hdList>  at  the
**  position  <pos>.  It is the  responsibility of the  caller to ensure that
**  <pos> is positive, and that <hdVal> is not 'HdVoid'.
**
**  'AssString' is the function in 'TabAssList' for strings.
**
**  'AssString' simply  converts the string into  a plain list, and then does
**  the  same  stuff as 'AssPlist'.  This  is because  a  string is not  very
**  likely to stay a string after the assignment.
*/
extern  Bag       AssString (
            Bag           hdList,
            Int                pos,
            Bag           hdVal );


/****************************************************************************
**
*F  AsssString(<hdList>,<hdPoss>,<hdVals>)assign several elements to a string
**
**  'AsssString' assignes  the values from the list <hdVals> at the positions
**  given  in  the  list  <hdPoss>  to   the  string  <hdList>.   It  is  the
**  responsibility of the  caller  to  ensure  that  <hdPoss>  is  dense  and
**  contains only positive integers, that <hdPoss> and <hdVals> have the same
**  length, and that <hdVals> is dense.
**
**  'AsssString' is the function in 'TabAsssList' for strings.
**
**  'AsssString' simply converts the string to a plain list and then does the
**  same stuff as 'AsssPlist'.  This is because a  string  is not very likely
**  to stay a string after the assignment.
*/
extern  Bag       AsssString (
            Bag           hdList,
            Bag           hdPoss,
            Bag           hdVals );


/****************************************************************************
**
*F  PosString(<hdList>,<hdVal>,<pos>) . .  position of an element in a string
**
**  'PosString' returns  the  position  of the  value  <hdVal>  in the string
**  <hdList> after the first position <start> as a C  integer.  0 is returned
**  if <hdVal> is not in the list.
**
**  'PosString' is the function in 'TabPosList' for strings.
*/
extern  Int            PosString (
            Bag           hdList,
            Bag           hdVal,
            Int                start );


/****************************************************************************
**
*F  PlainString(<hdList>) . . . . . . . . .  convert a string to a plain list
**
**  'PlainString'  converts the string  <hdList> to  a plain list.   Not much
**  work.
**
**  'PlainString' is the function in 'TabPlainList' for strings.
*/
extern  void            PlainString (
            Bag           hdList );


/****************************************************************************
**
*F  IsDenseString(<hdList>) . . . . . .  dense list test function for strings
**
**  'IsDenseString' returns 1, since every string is dense.
**
**  'IsDenseString' is the function in 'TabIsDenseList' for strings.
*/
extern  Int            IsDenseString (
            Bag           hdList );


/****************************************************************************
**
*F  IsPossString(<hdList>)  . . . .  positions list test function for strings
**
**  'IsPossString' returns 0, since every string contains no integers.
**
**  'IsPossString' is the function in 'TabIsPossList' for strings.
*/
extern  Int            IsPossString (
            Bag           hdList );


/****************************************************************************
**
*F  EqString( <hdL>, <hdR> )  . . . . . . . .  test whether strings are equal
**
**  'EqString' returns 'HdTrue' if the two strings <hdL> and <hdR> are  equal
**  and 'HdFalse' otherwise.
**
**  Is called from the 'Eq' binop, so both operands are already evaluated.
*/
extern  Bag       EqString (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  LtString( <hdL>, <hdR> )  .  test whether one string is less than another
**
**  'LtString' returns 'HdTrue' if the string <hdL> is less than  the  string
**  <hdR> and 'HdFalse' otherwise.
**
**  Is called from the 'Lt' binop, so both operands are already evaluated.
*/
extern  Bag       LtString (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  PrString( <hdStr> ) . . . . . . . . . . . . . . . . . . .  print a string
**
**  'PrString' prints the string with the handle <hdStr>.
**
**  No linebreaks are allowed, if one must be inserted  anyhow,  it  must  be
**  escaped by a backslash '\', which is done in 'Pr'.
*/
extern  void            PrString (
            Bag           hdStr );


/****************************************************************************
**
*F  PrintString( <hdStr> )  . . . . . . . . . . .  print a string for 'Print'
**
**  'PrintString' prints the string  constant  in  the  format  used  by  the
**  'Print' and 'PrintTo' function.
*/
extern  void            PrintString (
            Bag           hdStr );


/****************************************************************************
**
*F  IsString(<hdList>)  . . . . . . . . . . . . . . . . . . test for a string
**
**  'IsString' returns 1 if the list <hdList> is a string, and 0 otherwise.
*/
extern  Int            IsString (
            Bag           hdList );


/****************************************************************************
**
*F  FunIsString( <hdCall> ) . . . . . . . . . . . . . . . . test for a string
**
**  'FunIsString' implements the internal function 'IsString'.
**
**  'IsString( <obj> )'
**
**  'IsString' returns 'true' if the object <obj> is a  string,  and  'false'
**  otherwise.  Will cause an error if <obj> is an unbound variable.
*/
extern  Bag       FunIsString (
            Bag           hdCall );


/****************************************************************************
**
*F  InitString()  . . . . . . . . . . . . . . . .  initializes string package
**
**  'InitString' initializes the string package.
*/
extern  void            InitString ( void );


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



