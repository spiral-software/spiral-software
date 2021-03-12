/****************************************************************************
**
*A  vecffe.c                    GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file  contains  the functions that mainly  operate on vectors  whose
**  elements are elements from  finite fields.  As vectors  are special lists
**  many things are done in the list package.
**
**  A *vector* is a list that has no holes,  and whose elements all come from
**  a common field.  For the full definition of vectors see chapter "Vectors"
**  in  the {\GAP} manual.   Read also about "More   about Vectors" about the
**  vector flag and the compact representation of vectors over finite fields.
**
**  A list that is  known  to be vector over a finite field is represented by
**  a bag of 'T_VECFFE', which has the following format:
**
**      +-------+----+----+- - - -+----+
**      |finite |1st |2nd |       |last|
**      |field  |elm |elm |       |elm |
**      +-------+----+----+- - - -+----+
**
**  The first handle contains the handle of  the  finite  field  bag  of  the
**  finite field.  The other entries are the  elements  of  the  vector.  For
**  each element the index is stored as an unsigned short integer.
**
**  Note that a list represented by  a bag of  type 'T_LIST' or 'T_SET' might
**  still be a vector over a  finite field.  It is  just that the kernel does
**  not know this.
**
**  This package consists of four parts.
**
**  The   first   part    consists   of   the   macros    'SIZE_PLEN_VECFFE',
**  'PLEN_SIZE_VECFFE',   'LEN_VECFFE',    'SET_LEN_VECFFE',    'VAL_VECFFE',
**  'SET_VAL_VECFFE', 'ELM_VECFFE', and 'SET_ELM_VECFFE'.  They determine the
**  representation of vectors.  Everything else in  this file and the rest of
**  the {\GAP} kernel uses those macros to access and modify vectors.
**
**  The  second  part  consists of  the functions  'LenVecFFE',  'ElmVecFFE',
**  'ElmsVecFFE',  AssVecFFE', 'AsssVecFFE',  'PlainVecFFE', 'IsDenseVecFFE',
**  and 'IsPossVecFFE'.  They are the functions required by the generic lists
**  package.  Using these functions  the other parts of the {\GAP} kernel can
**  access  and  modify vectors  without actually being  aware  that they are
**  dealing with a vector.
**
**  The   third   part    consists   of   the   functions    'IsXTypeVecFFE',
**  'IsXTypeMatFFE', 'SumFFEVecFFE', 'SumVecFFEFFE', 'SumVecFFEVecFFE',  etc.
**  They  are the function for binary operations,  which  overlay the generic
**  functions in the generic list package for better efficiency.
**
**  The fourth part contains the internal functions for vectors.
**
*/

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "scanner.h"             /* reading of tokens and printing  */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */
#include        "objects.h"
#include		"string4.h"
#include        "integer4.h"            /* TypDigit                        */
#include        "list.h"                /* generic list package            */
#include        "plist.h"               /* 'LEN_PLIST', 'SET_LEN_PLIST',.. */
#include        "range.h"               /* 'LEN_RANGE', 'LOW_RANGE', ..    */
#include        "finfield.h"            /* <everything>                    */

#include        "vecffe.h"              /* declaration part of the package */


/****************************************************************************
**
*F  PLEN_SIZE_VECFFE(<size>)  . . . .  physical length from size for a vector
**
**  'PLEN_SIZE_VECFFE' computes  the  physical length  (e.g.   the  number of
**  elements that could be stored in a list) from the <size> (as reported  by
**  'GET_SIZE_BAG') for a vector.
**
**  Note that 'PLEN_SIZE_VECFFE' is a macro, so do not call it with arguments
**  that have sideeffects.
**
**  'PLEN_SIZE_VECFFE' is defined in the declaration part of this package  as
**  follows:
**
#define PLEN_SIZE_VECFFE(GET_SIZE_BAG)          (((GET_SIZE_BAG) - SIZE_HD) / sizeof(TypFFE))
*/


/****************************************************************************
**
*F  SIZE_PLEN_VECFFE(<plen>)  .  size for a vector with given physical length
**
**  'SIZE_PLEN_VECFFE' returns  the  size that a  vector with room for <plen>
**  elements must at least have.
**
**  Note that 'SIZE_PLEN_VECFFE' is a macro, so do not call it with arguments
**  that have sideeffects.
**
**  'SIZE_PLEN_VECFFE' is defined  in the declaration part of this package as
**  follows:
**
#define SIZE_PLEN_VECFFE(PLEN)          (SIZE_HD + (PLEN) * sizeof(TypFFE))
*/


/****************************************************************************
**
*F  LEN_VECFFE(<hdList>)  . . . . . . . . . . . . . . . .  length of a vector
**
**  'LEN_VECFFE' returns the  logical  length  of  the vector <hdList> as a C
**  integer.  The  length  is stored as GAP immediate integer as  the zeroeth
**  handle.
**
**  Note that 'LEN_VECFFE' is a macro,  so do not call it with arguments that
**  have sideeffects.
**
**  'LEN_VECFFE' is defined  in  the  declaration  part of  this  package  as
**  follows:
**
#define LEN_VECFFE(LIST)              PLEN_SIZE_VECFFE( GET_SIZE_BAG( LIST ) )
*/


/****************************************************************************
**
*F  SET_LEN_VECFFE(<hdList>,<len>)  . . . . . . .  set the length of a vector
**
**  'SET_LEN_VECFFE' sets  the  length of the vector <hdList> to <len>.   The
**  length is stored as GAP immediate integer as the zeroeth handle.
**
**  Note that 'SET_LEN_VECFFE' is a macro, so do not call  it  with arguments
**  that have sideeffects.
**
**  'SET_LEN_VECFFE' is defined in  the declaration  part  of this package as
**  follows:
**
#define SET_LEN_VECFFE(LIST,LEN)        Resize( LIST, SIZE_PLEN_VECFFE(LEN) )
*/


/****************************************************************************
**
*F  FLD_VECFFE(<hdList>)  . . . . . . . . . . . . . . . . . field of a vector
**
**  'FLD_VECFFE' returns the handle of the finite field over which the vector
**  <hdList> is defined.
**
**  Note that 'FLD_VECFFE' is a macro, so do not call it  with arguments that
**  have sideeffects.
**
**  'FLD_VECFFE'  is  defined  in  the declaration part  of this  package  as
**  follows:
**
#define FLD_VECFFE(LIST)                (PTR_BAG(LIST)[0])
*/


/****************************************************************************
**
*F  SET_FLD_VECFFE(<hdList>,<hdField>)  . . . . . . set the field of a vector
**
**  'SET_FLD_VECFFE' sets the field of the vector <hdList> to <hdField>.
**
**  Note that 'SET_FLD_VECFFE' is a macro, so  do not  call it with arguments
**  that have sideeffects.
**
**  'SET_FLD_VECFFE' is defined in the declaration part  of  this  package as
**  follows:
**
#define SET_FLD_VECFFE(LIST,FLD)        (FLD_VECFFE(LIST) = (FLD))
*/


/****************************************************************************
**
*F  VAL_VECFFE(<hdVec>,<pos>) . . . . . . . value of an element from a vector
**
**  'VAL_VECFFE' returns the value of the  <pos>-th  element  of  the  finite
**  field vector <hdVec>.
**
**  Note that 'VAL_VECFFE' is a macro, so do not call it with arguments  that
**  have sideeffects.
**
**  'VAL_VECFFE' is defined in  the  declaration  part  of  this  package  as
**  follows:
**
#define VAL_VECFFE(VEC,POS)             (((TypFFE*)(PTR_BAG(VEC)+1))[(POS)-1])
*/


/****************************************************************************
**
*F  SET_VAL_VECFFE(<hdVec>,<pos>,<val>) set value of an element from a vector
**
**  'SET_VAL_VECFFE' sets the value of the <pos>-th  element  of  the  finite
**  field vector <hdVec> to <val>.
**
**  Note that 'SET_VAL_VECFFE' is a macro, so do not call it  with  arguments
**  that have sideeffects.
**
**  'SET_VAL_VECFFE' is defined in the declaration part of  this  package  as
**  follows:
**
#define SET_VAL_VECFFE(VEC,POS,VAL)     (VAL_VECFFE(VEC,POS) = (VAL))
*/


/****************************************************************************
**
*F  ELM_VECFFE(<hdVec>,<i>,<hdElm>) . . . . . . . . . . . element of a vector
**
**  'ELM_VECFFE'  assigns the   <i>-th  element of  the finite   field vector
**  <hdVec> to the finite field element bag <hdElm>.   <i> must be a positive
**  integer less than or equal to the length of <hdVec>.
**
**  Note that 'ELM_VECFFE' is a macro, so do not call  it with arguments that
**  have sideeffects.
**
**  'ELM_VECFFE' is one of the functions that packages implementing list like
**  objects must export.  It is called from 'ElmList' and 'EvFor' and various
**  other places.  Note that 'ELM_VECFFE' expects  the  bag  for  the  result
**  already allocated, unlike all the other 'ELM_<type>' functions.
**
**  'ELM_VECFFE' is  defined   in the declaration  part   of   the package as
**  follows:
**
#define ELM_VECFFE(LIST,POS,ELM)      (SET_FLD_FFE(ELM,FLD_VECFFE(LIST)), \
                                       SET_VAL_FFE(ELM,VAL_VECFFE(LIST,POS)))
*/


/****************************************************************************
**
*F  SET_ELM_VECFFE(<hdList>,<pos>,<hdVal>)  . . assign an element to a vector
**
**  'SET_ELM_VECFFE' assigns the value <hdVal> to the  vector <hdList> at the
**  position <pos>.  <pos> must be a positive integer less  than  or equal to
**  the length of <hdList>.
**
**  Note that 'SET_ELM_VECFFE'  is a macro, so do not call it  with arguments
**  that have sideeffects.
**
**  'SET_ELM_VECFFE'  is defined  in the declaration  part of this package as
**  follows:
**
#define SET_ELM_VECFFE(LIST,POS,ELM)    SET_VAL_VECFFE(LIST,POS,VAL_FFE(ELM))
*/


/****************************************************************************
**
*F  LenVecFFE(<hdList>) . . . . . . . . . . . . . . . . .  length of a vector
**
**  'LenVecFFE' returns the length of the vector <hdList> as a C integer.
**
**  'LenVecFFE' is the function in 'TabLenList' for vectors.
*/
Int            LenVecFFE (Bag hdList)
{
    return LEN_VECFFE( hdList );
}


/****************************************************************************
**
*F  ElmVecFFE(<hdList>,<pos>) . . . . . . . . . select an element of a vector
**
**  'ElmVecFFE' selects the element at position <pos> of the vector <hdList>.
**  It is the responsibility of the caller to ensure that <pos> is a positive
**  integer.  An  error is signalled if <pos>  is  larger than the  length of
**  <hdList>.
**
**  'ElmfVecFFE' does  the same thing than 'ElmList', but need not check that
**  <pos>  is less than  or  equal to the  length of  <hdList>, this  is  the
**  responsibility of the caller.
**
**  'ElmVecFFE' is the function in 'TabElmList' for vectors.  'ElmfVecFFE' is
**  the  function  in  'TabElmfList', 'TabElmlList',  and  'TabElmrList'  for
**  vectors.
*/
Bag       ElmVecFFE (Bag hdList, Int pos)
{
    Bag           hdElm;          /* the selected element, result    */

    /* check the position                                                  */
    if ( LEN_VECFFE( hdList ) < pos ) {
        return Error(
          "List Element: <list>[%d] must have a value",
                     pos,  0 );
    }

    /* select and check the element                                        */
    hdElm = NewBag( T_FFE, SIZE_HD + sizeof(TypFFE) );
    ELM_VECFFE( hdList, pos, hdElm );

    /* return the element                                                  */
    return hdElm;
}

Bag       ElmfVecFFE (Bag hdList, Int pos)
{
    Bag           hdElm;          /* the selected element, result    */

    /* select and check the element                                        */
    hdElm = NewBag( T_FFE, SIZE_HD + sizeof(TypFFE) );
    ELM_VECFFE( hdList, pos, hdElm );

    /* select and return the element                                       */
    return hdElm;
}

/*V HdVecFFEL */
Bag       HdVecFFEL;

Bag       ElmlVecFFE (Bag hdList, Int pos)
{
    ELM_VECFFE( hdList, pos, HdVecFFEL );
    return HdVecFFEL;
}

/*V HdVecFFER */
Bag       HdVecFFER;

Bag       ElmrVecFFE (Bag hdList, Int pos)
{
    ELM_VECFFE( hdList, pos, HdVecFFER );
    return HdVecFFER;
}


/****************************************************************************
**
*F  ElmsVecFFE(<hdList>,<hdPoss>) . . . . . .  select a sublist from a vector
**
**  'ElmsVecFFE' returns a new list containing the elements  at the  position
**  given  in  the  list  <hdPoss>  from  the  vector  <hdList>.  It  is  the
**  responsibility  of  the  caller  to  ensure that  <hdPoss> is  dense  and
**  contains only positive integers.   An error is signalled if an element of
**  <hdPoss> is larger than the length of <hdList>.
**
**  'ElmsVecFFE' is the function in 'TabElmsList' for vectors.
*/
Bag       ElmsVecFFE (Bag hdList, Bag hdPoss)
{
    Bag           hdElms;         /* selected sublist, result        */
    Int                lenList;        /* length of <list>                */
    TypFFE              hdElm;          /* one element from <list>         */
    Int                lenPoss;        /* length of <positions>           */
    Int                pos;            /* <position> as integer           */
    Int                inc;            /* increment in a range            */
    Int                i;              /* loop variable                   */

    /* general code                                                        */
    if ( GET_TYPE_BAG(hdPoss) != T_RANGE ) {

        /* get the length of <list>                                        */
        lenList = LEN_VECFFE( hdList );

        /* get the length of <positions>                                   */
        lenPoss = LEN_LIST( hdPoss );

        /* make the result list                                            */
        hdElms = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( lenPoss ) );
        SET_FLD_VECFFE( hdElms, FLD_VECFFE( hdList ) );

        /* loop over the entries of <positions> and select                 */
        for ( i = 1; i <= lenPoss; i++ ) {

            /* get <position>                                              */
            pos = HD_TO_INT( ELMF_LIST( hdPoss, i ) );
            if ( lenList < pos ) {
                return Error(
                  "List Elements: <list>[%d] must have a value",
                             pos,  0 );
            }

            /* select the element                                          */
            hdElm = VAL_VECFFE( hdList, pos );

            /* assign the element into <elms>                              */
            SET_VAL_VECFFE( hdElms, i, hdElm );

        }

    }

    /* special code for ranges                                             */
    else {

        /* get the length of <list>                                        */
        lenList = LEN_VECFFE( hdList );

        /* get the length of <positions>, the first elements, and the inc. */
        lenPoss = LEN_RANGE( hdPoss );
        pos = LOW_RANGE( hdPoss );
        inc = INC_RANGE( hdPoss );

        /* check that no <position> is larger than 'LEN_LIST(<list>)'      */
        if ( lenList < pos ) {
            return Error(
              "List Elements: <list>[%d] must have a value",
                         pos,  0 );
        }
        if ( lenList < pos + (lenPoss-1) * inc ) {
            return Error(
              "List Elements: <list>[%d] must have a value",
                         pos + (lenPoss-1) * inc,  0 );
        }

        /* make the result list                                            */
        hdElms = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( lenPoss ) );
        SET_FLD_VECFFE( hdElms, FLD_VECFFE( hdList ) );

        /* loop over the entries of <positions> and select                 */
        for ( i = 1; i <= lenPoss; i++, pos += inc ) {

            /* select the element                                          */
            hdElm = VAL_VECFFE( hdList, pos );

            /* assign the element into <elms>                              */
            SET_VAL_VECFFE( hdElms, i, hdElm );

        }

    }

    /* return the result                                                   */
    return hdElms;
}


/****************************************************************************
**
*F  AssVecFFE(<hdList>,<pos>,<hdVal>) . . . . . . . . . .  assign to a vector
**
**  'AssVecFFE' assigns the  value  <hdVal>  to the  vector  <hdList>  at the
**  position <pos>.   It is  the responsibility of the  caller to ensure that
**  <pos> is positive, and that <hdVal> is not 'HdVoid'.
**
**  If the position is larger  then the length of the vector <list>, the list
**  is automatically  extended.  To avoid  making this too often, the  bag of
**  the list is extended by at least '<length>/8 +  4' handles.  Thus in  the
**  loop
**
**      l := [];  for i in [1..1024]  do l[i] := i^2;  od;
**
**  the list 'l' is extended only 32 times not 1024 times.
**
**  'AssVecFFE' is the function in 'TabAssList' for vectors.
**
**  'AssVecFFE' simply converts  the  vector into a plain list, and then does
**  the same  stuff as  'AssPlist'.   This is because  a  vector is  not very
**  likely to stay a vector after the assignment.
*/
Bag       AssVecFFE (Bag hdList, Int pos, Bag hdVal)
{
    Int                plen;           /* physical length of <list>       */

    /* assignment of a scalar within the bound                             */
    if ( GET_TYPE_BAG(hdVal) == T_FFE && FLD_FFE(hdVal) == FLD_VECFFE(hdList)
      && pos <= LEN_VECFFE(hdList) ) {
        SET_ELM_VECFFE( hdList, pos, hdVal );
    }

    /* assignment of a scalar immediately after the end                    */
    else if ( GET_TYPE_BAG(hdVal) == T_FFE && FLD_FFE(hdVal) == FLD_VECFFE(hdList)
           && pos == LEN_VECFFE(hdList)+1 ) {
        Resize( hdList, SIZE_PLEN_VECFFE( pos ) );
        SET_ELM_VECFFE( hdList, pos, hdVal );
    }

    /* otherwise convert to plain list                                     */
    else {
        PLAIN_LIST( hdList );
        if ( LEN_PLIST( hdList ) < pos ) {
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
*F  AsssVecFFE(<hdList>,<hdPoss>,<hdVals>)assign several elements to a vector
**
**  'AsssVecFFE' assignes the values from the  list <hdVals> at the positions
**  given  in  the  list  <hdPoss>  to   the  vector  <hdList>.   It  is  the
**  responsibility  of the  caller to  ensure  that  <hdPoss>  is  dense  and
**  contains only positive integers, that <hdPoss> and <hdVals> have the same
**  length, and that <hdVals> is dense.
**
**  'AsssVecFFE' is the function in 'TabAsssList' for vectors.
**
**  'AsssVecFFE' simply converts the vector to a plain list and then does the
**  same stuff as 'AsssPlist'.  This is because a vector  is not  very likely
**  to stay a vector after the assignment.
*/
Bag       AsssVecFFE (Bag hdList, Bag hdPoss, Bag hdVals)
{
    /* convert <list> to a plain list                                      */
    PLAIN_LIST( hdList );
    Retype( hdList, T_LIST );

    /* and delegate                                                        */
    return ASSS_LIST( hdList, hdPoss, hdVals );
}


/****************************************************************************
**
*F  PosVecFFE(<hdList>,<hdVal>,<pos>) . .  position of an element in a vector
**
**  'PosVecFFE' returns  the  position  of the  value  <hdVal>  in the vector
**  <hdList> after the first position <start> as a C  integer.  0 is returned
**  if <hdVal> is not in the list.
**
**  'PosVecFFE' is the function in 'TabPosList' for vectors.
*/
Int            PosVecFFE (Bag hdList, Bag hdVal, Int start)
{
    Int                lenList;        /* length of <list>                */
    Bag           hdElm;          /* one element of <list>           */
    Int                i;              /* loop variable                   */

    /* get the length of <list>                                            */
    lenList = LEN_VECFFE( hdList );

    /* loop over all entries in <list>                                     */
    for ( i = start+1; i <= lenList; i++ ) {

        /* select one element from <list>                                  */
        hdElm = ELML_LIST( hdList, i );

        /* compare with <val>                                              */
        if ( hdElm != 0 && (hdElm == hdVal || EQ( hdElm, hdVal ) == HdTrue) )
            break;

    }

    /* return the position (0 if <val> was not found)                      */
    return (lenList < i ? 0 : i);
}


/****************************************************************************
**
*F  PlainVecFFE(<hdList>) . . . . . . . . .  convert a vector to a plain list
**
**  'PlainVecFFE'  converts the vector  <hdList> to  a plain list.   Not much
**  work.
**
**  'PlainVecFFE' is the function in 'TabPlainList' for vectors.
*/
void            PlainVecFFE (Bag hdList)
{
    Int                lenList;        /* logical length of the vector    */
    Bag           hdCopy;         /* handle of the list              */
    Int                i;              /* loop variable                   */

    /* find the length and allocate a temporary copy                       */
    lenList = LEN_VECFFE( hdList );
    hdCopy = NewBag( T_LIST, SIZE_PLEN_PLIST( lenList ) );
    SET_LEN_PLIST( hdCopy, lenList );

    /* create the finite field entries                                     */
    for ( i = 1; i <= lenList; i++ ) {
        SET_ELM_PLIST( hdCopy, i, ElmfVecFFE( hdList, i ) );
    }

    /* change size and type of the vector and copy back                    */
    Resize( hdList, SIZE_PLEN_PLIST( lenList ) );
    Retype( hdList, T_LIST );
    SET_LEN_PLIST( hdList, lenList );
    for ( i = 1; i <= lenList; i++ ) {
        SET_ELM_PLIST( hdList, i, ELM_PLIST( hdCopy, i ) );
    }

}


/****************************************************************************
**
*F  IsDenseVecFFE(<hdList>) . . . . . .  dense list test function for vectors
**
**  'IsDenseVecFFE' returns 1, since every vector is dense.
**
**  'IsDenseVecFFE' is the function in 'TabIsDenseList' for vectors.
*/
Int            IsDenseVecFFE (Bag hdList)
{
    return 1;
}


/****************************************************************************
**
*F  IsPossVecFFE(<hdList>)  . . . .  positions list test function for vectors
**
**  'IsPossVecFFE' returns 0, since every vector contains no integers.
**
**  'IsPossVecFFE' is the function in 'TabIsPossList' for vectors.
*/
Int            IsPossVecFFE (Bag hdList)
{
    return LEN_VECFFE( hdList ) == 0;
}


/****************************************************************************
**
*F  IsXTypeVecFFE(<hdList>) . . . . . . . . . . .  test if a list is a vector
**
**  'IsXTypeVecFFE'  returns 1  if  the list  <hdList>  is  a  vector  and  0
**  otherwise.  As a sideeffect the representation  of the list is changed to
**  'T_VECFFE'.
**
**  'IsXTypeVecFFE' is the function in 'TabIsXTypeList' for vectors.
*/
Int            IsXTypeVecFFE (Bag hdList)
{
    Int                isVecFFE;       /* result                          */
    UInt       len;            /* length of the list              */
    Bag           hdFld;          /* handle of the field             */
    UInt       p;              /* characteristic                  */
    UInt       d;              /* degree of common finite field   */
    UInt       q;              /* size of common finite field     */
    Bag           hdElm;          /* one element of the list         */
    TypFFE              v;              /* value of the element            */
    UInt       q1;             /* size of field of element        */
    UInt       d1;             /* degree of element               */
    UInt       i, k;           /* loop variables                  */

    /* if we already know that the list is a vector, very good             */
    if ( GET_TYPE_BAG(hdList) == T_VECFFE ) {
        isVecFFE = 1;
    }

    /* only lists or sets can be a vector                                  */
    else if ( GET_TYPE_BAG(hdList) != T_LIST && GET_TYPE_BAG(hdList) != T_SET ) {
        isVecFFE = 0;
    }

    /* if the list is empty, it is not a vector                            */
    else if ( LEN_PLIST( hdList ) == 0 || ELM_PLIST( hdList, 1 ) == 0 ) {
        isVecFFE = 0;
    }

    /* if the first element is a ffe, test the others                      */
    else if ( GET_TYPE_BAG( ELM_PLIST( hdList, 1 ) ) == T_FFE ) {

        /* get the length of the list                                      */
        len = LEN_PLIST( hdList );

        /* get the field and compare all other elements against that       */
        hdFld = FLD_FFE( ELM_PLIST( hdList, 1 ) );
        for ( i = 2; i <= len; i++ ) {

            /* check that the element if a ffe of the same characteristic  */
            hdElm = ELM_PLIST( hdList, i );
            if ( hdElm == 0 || GET_TYPE_BAG(hdElm) != T_FFE
              || FLD_FFE( hdElm ) != hdFld ) {
                break;
            }

        }

        /* maybe this already worked                                       */
        isVecFFE = (len < i);
        if ( isVecFFE ) {

            /* convert all elements                                        */
            for ( i = 1; i <= len; i++ ) {
                hdElm = ELM_PLIST( hdList, i );
                v = VAL_FFE( hdElm );
                SET_VAL_VECFFE( hdList, i, v );
            }

            /* retype, set the field and the length                        */
            Retype( hdList, T_VECFFE );
            SET_FLD_VECFFE( hdList, hdFld );
            SET_LEN_VECFFE( hdList, len );

        }

        /* otherwise we have to work harder                                */
        else {

            /* test all elements                                           */
            p = CharFFE( ELM_PLIST( hdList, 1 ) );
            d = 1;
            for ( i = 1; i <= len; i++ ) {

                /* check that the element if a ffe of the same char.       */
                hdElm = ELM_PLIST( hdList, i );
                if ( hdElm == 0 || GET_TYPE_BAG(hdElm) != T_FFE
                  || SIZE_FF( FLD_FFE( hdElm ) ) % p != 0 ) {
                    break;
                }

                /* get the degree of the smallest field that contains elm  */
                d1 = DegreeFFE( hdElm );

                /* get the degree of the smallest common superfield        */
                for ( k = d; d % d1 != 0; d += k )  ;

                /* make sure we can handle this field                      */
                if ( (  2 <= p && 17 <= d) || (  3 <= p && 11 <= d)
                  || (  5 <= p &&  7 <= d) || (  7 <= p &&  6 <= d)
                  || ( 11 <= p &&  5 <= d) || ( 17 <= p &&  4 <= d)
                  || ( 41 <= p &&  3 <= d) || (257 <= p &&  2 <= d) ) {
                    break;
                }

            }

            /* if this worked, convert to a vector                         */
            isVecFFE = (len < i);
            if ( isVecFFE ) {

                /* get a field that contains all elements                  */
                /* if possible take the field of the first element         */
                for ( q = 1, k = 1; k <= d; k++ )  q *= p;
                if ( (SIZE_FF(FLD_FFE(ELM_PLIST(hdList,1)))-1) % (q-1) == 0 )
                    hdFld = FLD_FFE( ELM_PLIST(hdList,1) );
                else
                    hdFld = FLD_FFE( RootFiniteField( q ) );
                q = SIZE_FF( hdFld );

                /* convert all elements                                    */
                for ( i = 1; i <= len; i++ ) {
                    hdElm = ELM_PLIST( hdList, i );
                    q1 = SIZE_FF( FLD_FFE( hdElm ) );
                    v = VAL_FFE( hdElm );
                    SET_VAL_VECFFE( hdList, i,
                                    v==0 ? v : (v-1)*(q-1)/(q1-1)+1 );
                }

                /* retype, set the field and the length                    */
                Retype( hdList, T_VECFFE );
                SET_FLD_VECFFE( hdList, hdFld );
                SET_LEN_VECFFE( hdList, len );

            }

        }

    }

    /* otherwise the list is cleary not a vector                           */
    else {
        isVecFFE = 0;
    }

    /* return the result                                                   */
    return isVecFFE;
}


/****************************************************************************
**
*F  IsXTypeMatFFE(<hdList>) . . . . . . . . . . .  test if a list is a matrix
**
**  'IsXTypeMatFFE'  returns  1  if  the list <hdList>  is  a  matrix  and  0
**  otherwise.  As a sideeffect the representation  of the rows is changed to
**  'T_VECFFE'.
**
**  'IsXTypeMatFFE' is the function in 'TabIsXTypeList' for matrices.
*/
Int            IsXTypeMatFFE (Bag hdList)
{
    Int                isMatFFE;       /* result                          */
    UInt       len;            /* length of the list              */
    UInt       col;            /* length of the rows              */
    Bag           hdFld;          /* handle of the field             */
    UInt       p;              /* characteristic                  */
    UInt       d;              /* degree of common finite field   */
    UInt       q;              /* size of common finite field     */
    Bag           hdElm;          /* one row of the list             */
    TypFFE              v;              /* value of one element            */
    UInt       q1;             /* size of field of row            */
    UInt       d1;             /* degree of field of row          */
    UInt       i, k;           /* loop variables                  */

    /* only lists or sets can be matrices                                  */
    if ( GET_TYPE_BAG(hdList) != T_LIST && GET_TYPE_BAG(hdList) != T_SET ) {
        isMatFFE = 0;
    }

    /* if the list is empty, it is not a matrix                            */
    else if ( LEN_PLIST( hdList ) == 0 || ELM_PLIST( hdList, 1 ) == 0 ) {
        isMatFFE = 0;
    }

    /* if the first element is a vector, test the others                   */
    else if ( IsXTypeVecFFE( ELM_PLIST( hdList, 1 ) ) ) {

        /* get the length of the list                                      */
        len = LEN_PLIST( hdList );
        col = LEN_VECFFE( ELM_PLIST( hdList, 1 ) );

        /* get the field and compare all other elements against that       */
        hdFld = FLD_VECFFE( ELM_PLIST( hdList, 1 ) );
        for ( i = 2; i <= len; i++ ) {

            /* check that the element if a ffe of the same characteristic  */
            hdElm = ELM_PLIST( hdList, i );
            if ( hdElm == 0 || GET_TYPE_BAG(hdElm) != T_VECFFE
              || col != LEN_VECFFE( hdElm )
              || FLD_VECFFE( hdElm ) != hdFld ) {
                break;
            }

        }

        /* maybe this already worked                                       */
        isMatFFE = (len < i);

        /* otherwise we have to work harder                                */
        if ( ! isMatFFE ) {

            /* test all elements                                           */
            p = CharVecFFE( ELM_PLIST( hdList, 1 ) );
            d = 1;
            for ( i = 1; i <= len; i++ ) {

                /* check that the element is a vecffe of the same char.    */
                hdElm = ELM_PLIST( hdList, i );
                if ( hdElm == 0 || ! IsXTypeVecFFE( hdElm )
                  || col != LEN_VECFFE( hdElm )
                  || SIZE_FF( FLD_VECFFE( hdElm ) ) % p != 0 ) {
                    break;
                }

                /* get the degree of the smallest field that contains vec. */
                d1 = DegreeVecFFE( hdElm );

                /* get the degree of the smallest common superfield        */
                for ( k = d; d % d1 != 0; d += k )  ;

                /* make sure we can handle this field                      */
                if ( (  2 <= p && 17 <= d) || (  3 <= p && 11 <= d)
                  || (  5 <= p &&  7 <= d) || (  7 <= p &&  6 <= d)
                  || ( 11 <= p &&  5 <= d) || ( 17 <= p &&  4 <= d)
                  || ( 41 <= p &&  3 <= d) || (257 <= p &&  2 <= d) ) {
                    break;
                }

            }

            /* test if it worked                                           */
            isMatFFE = (len < i);
            if ( isMatFFE ) {

                /* get a field that contains all elements                  */
                /* if possible take the field of the first element         */
                for ( q = 1, k = 1; k <= d; k++ )  q *= p;
                if ( (SIZE_FF(FLD_VECFFE(ELM_PLIST(hdList,1)))-1)%(q-1)==0 )
                    hdFld = FLD_VECFFE( ELM_PLIST(hdList,1) );
                else
                    hdFld = FLD_FFE( RootFiniteField( q ) );
                q = SIZE_FF( hdFld );

                /* convert all rows                                        */
                for ( i = 1; i <= len; i++ ) {
                    hdElm = ELM_PLIST( hdList, i );
                    if ( FLD_VECFFE( hdElm ) != hdFld ) {
                        q1 = SIZE_FF( FLD_VECFFE( hdElm ) );
                        for ( k = 1; k <= col; k++ ) {
                            v = VAL_VECFFE( hdElm, k );
                            SET_VAL_VECFFE( hdElm, k,
                                           v==0 ? v : (v-1)*(q-1)/(q1-1)+1 );
                        }
                        SET_FLD_VECFFE( hdElm, hdFld );
                    }
                }

            }

        }

    }

    /* otherwise the list is clearly not a matrix                          */
    else {
        isMatFFE = 0;
    }

    /* return the result                                                   */
    return isMatFFE;
}


/****************************************************************************
**
*F  SumFFEVecFFE(<hdL>,<hdR>) . .  sum of a finite field element and a vector
**
**  'SumFFEVecFFE' returns  the sum of the finite field element <hdL> and the
**  vector <hdR>.  The sum is a  new list, where each element is  the sum  of
**  <hdL> and the corresponding entry of <hdR>.
**
**  'SumFFEVecFFE'  is an improved version of  'SumSclList', which  does  not
**  call 'SUM'.
*/
Bag       SumFFEVecFFE (Bag hdL, Bag hdR)
{
    Bag           hdS;            /* sum, result                     */
    UInt       len;            /* length of the list              */
    UInt       p;              /* characteristic                  */
    UInt       q;              /* size of common field            */
    UInt       d;              /* degree of common field          */
    Bag           hdFld;          /* handle of common field          */
    TypFFE *            fld;            /* successor table of common field */
    UInt       qL;             /* size of left operand's field    */
    UInt       dL;             /* degree of left operand's field  */
    TypFFE              vL;             /* value of left operand           */
    UInt       qR;             /* size of right operand's field   */
    UInt       dR;             /* degee of right operand's field  */
    TypFFE              vR;             /* value of one element of list    */
    UInt       i;              /* loop variable                   */

    /* both operands lie in the same field                                 */
    if ( FLD_FFE( hdL ) == FLD_VECFFE( hdR ) ) {

        /* make the result vector                                          */
        len = LEN_VECFFE( hdR );
        hdS = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( len ) );

        /* get the field and its successor table                           */
        hdFld = FLD_FFE( hdL );
        fld = SUCC_FF( hdFld );
        SET_FLD_VECFFE( hdS, hdFld );

        /* get the left operand's value                                    */
        vL = VAL_FFE( hdL );

        /* loop over the entries and add                                   */
        for ( i = 1; i <= len; i++ ) {
            vR = VAL_VECFFE( hdR, i );
            SET_VAL_VECFFE( hdS, i, SUM_FF( vL, vR, fld ) );
        }

    }

    /* otherwise try to lift the operands into a common field              */
    else {

        /* make the result vector                                          */
        len = LEN_VECFFE( hdR );
        hdS = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( len ) );

        /* get a common field and its successor table                      */
        p = CharFFE( hdL );
        if ( SIZE_FF( FLD_VECFFE( hdR ) ) % p != 0 )
         return Error("Vector +: operands must lie in a common field", 0, 0);
        dL = DegreeFFE( hdL );
        dR = DegreeVecFFE( hdR );
        for ( d = 1, q = p; d % dR != 0 || d % dL != 0; d++ )  q *= p;
        if ( (  2 <= p && 17 <= d) || (  3 <= p && 11 <= d)
          || (  5 <= p &&  7 <= d) || (  7 <= p &&  6 <= d)
          || ( 11 <= p &&  5 <= d) || ( 17 <= p &&  4 <= d)
          || ( 41 <= p &&  3 <= d) || (257 <= p &&  2 <= d) )
         return Error("Vector +: smallest common field too large", 0, 0);
        if ( (SIZE_FF(FLD_FFE(hdL))-1) % (q-1) == 0 )
            hdFld = FLD_FFE( hdL );
        else if ( (SIZE_FF(FLD_VECFFE(hdR))-1) % (q-1) == 0 )
            hdFld = FLD_VECFFE( hdR );
        else
            hdFld = FLD_FFE( RootFiniteField( q ) );
        fld = SUCC_FF( hdFld );
        q = SIZE_FF( hdFld );
        SET_FLD_VECFFE( hdS, hdFld );

        /* get the left operand's field size and value                     */
        qL = SIZE_FF( FLD_FFE( hdL ) );
        vL = VAL_FFE( hdL );
        if ( vL != 0 )  vL = (vL-1)*(q-1)/(qL-1)+1;

        /* get the right operand's field size                              */
        qR = SIZE_FF( FLD_VECFFE( hdR ) );

        /* loop over the entries and add                                   */
        for ( i = 1; i <= len; i++ ) {
            vR = VAL_VECFFE( hdR, i );
            if ( vR != 0 )  vR = (vR-1)*(q-1)/(qR-1)+1;
            SET_VAL_VECFFE( hdS, i, SUM_FF( vL, vR, fld ) );
        }

    }

    /* return the result                                                   */
    return hdS;
}


/****************************************************************************
**
*F  SumVecFFEFFE(<hdL>,<hdR>) . .  sum of a vector and a finite field element
**
**  'SumVecFFEFFE' returns the  sum  of the vector <hdL> and the finite field
**  element  <hdR>.  The sum is a new list,  where each element is the sum of
**  <hdL> and the corresponding entry of <hdR>.
**
**  'SumVecFFEFFE'  is an improved  version of 'SumListScl',  which does  not
**  call 'SUM'.
*/
Bag       SumVecFFEFFE (Bag hdL, Bag hdR)
{
    Bag           hdS;            /* sum, result                     */
    UInt       len;            /* length of the list              */
    UInt       p;              /* characteristic                  */
    UInt       q;              /* size of common field            */
    UInt       d;              /* degree of common field          */
    Bag           hdFld;          /* handle of common field          */
    TypFFE *            fld;            /* successor table of common field */
    UInt       qL;             /* size of left operand's field    */
    UInt       dL;             /* degree of left operand's field  */
    TypFFE              vL;             /* value of left operand           */
    UInt       qR;             /* size of right operand's field   */
    UInt       dR;             /* degee of right operand's field  */
    TypFFE              vR;             /* value of one element of list    */
    UInt       i;              /* loop variable                   */

    /* both operands lie in the same field                                 */
    if ( FLD_VECFFE( hdL ) == FLD_FFE( hdR ) ) {

        /* make the result vector                                          */
        len = LEN_VECFFE( hdL );
        hdS = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( len ) );

        /* get the field and its successor table                           */
        hdFld = FLD_VECFFE( hdL );
        fld = SUCC_FF( hdFld );
        SET_FLD_VECFFE( hdS, hdFld );

        /* get the right operand's value                                   */
        vR = VAL_FFE( hdR );

        /* loop over the entries and add                                   */
        for ( i = 1; i <= len; i++ ) {
            vL = VAL_VECFFE( hdL, i );
            SET_VAL_VECFFE( hdS, i, SUM_FF( vL, vR, fld ) );
        }

    }

    /* otherwise try to lift the operands into a common field              */
    else {

        /* make the result vector                                          */
        len = LEN_VECFFE( hdL );
        hdS = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( len ) );

        /* get a common field and its successor table                      */
        p = CharVecFFE( hdL );
        if ( SIZE_FF( FLD_FFE( hdR ) ) % p != 0 )
         return Error("Vector +: operands must lie in a common field", 0, 0);
        dL = DegreeVecFFE( hdL );
        dR = DegreeFFE( hdR );
        for ( d = 1, q = p; d % dR != 0 || d % dL != 0; d++ )  q *= p;
        if ( (  2 <= p && 17 <= d) || (  3 <= p && 11 <= d)
          || (  5 <= p &&  7 <= d) || (  7 <= p &&  6 <= d)
          || ( 11 <= p &&  5 <= d) || ( 17 <= p &&  4 <= d)
          || ( 41 <= p &&  3 <= d) || (257 <= p &&  2 <= d) )
         return Error("Vector +: smallest common field too large", 0, 0);
        if ( (SIZE_FF(FLD_VECFFE(hdL))-1) % (q-1) == 0 )
            hdFld = FLD_VECFFE( hdL );
        else if ( (SIZE_FF(FLD_FFE(hdR))-1) % (q-1) == 0 )
            hdFld = FLD_FFE( hdR );
        else
            hdFld = FLD_FFE( RootFiniteField( q ) );
        fld = SUCC_FF( hdFld );
        q = SIZE_FF( hdFld );
        SET_FLD_VECFFE( hdS, hdFld );

        /* get the right operand's field size and value                    */
        qR = SIZE_FF( FLD_FFE( hdR ) );
        vR = VAL_FFE( hdR );
        if ( vR != 0 )  vR = (vR-1)*(q-1)/(qR-1)+1;

        /* get the left operand's field size                               */
        qL = SIZE_FF( FLD_VECFFE( hdL ) );

        /* loop over the entries and add                                   */
        for ( i = 1; i <= len; i++ ) {
            vL = VAL_VECFFE( hdL, i );
            if ( vL != 0 )  vL = (vL-1)*(q-1)/(qL-1)+1;
            SET_VAL_VECFFE( hdS, i, SUM_FF( vL, vR, fld ) );
        }

    }

    /* return the result                                                   */
    return hdS;
}


/****************************************************************************
**
*F  SumVecFFEVecFFE(<hdL>,<hdR>)  . . . . . . . . . . . .  sum of two vectors
**
**  'SumVecFFEVecFFE' returns  the  sum of  the two vectors <hdL>  and <hdR>.
**  The sum is a new list, where each element is the sum of the corresponding
**  entries of <hdL> and <hdR>.
**
**  'SumVecFFEVecFFE' is an improved version of 'SumListList', which does not
**  call 'SUM'.
*/
Bag       SumVecFFEVecFFE (Bag hdL, Bag hdR)
{
    Bag           hdS;            /* sum, result                     */
    UInt       len;            /* length of the list              */
    UInt       p;              /* characteristic                  */
    UInt       q;              /* size of common field            */
    UInt       d;              /* degree of common field          */
    Bag           hdFld;          /* handle of common field          */
    TypFFE *            fld;            /* successor table of common field */
    UInt       qL;             /* size of left operand's field    */
    UInt       dL;             /* degree of left operand's field  */
    TypFFE              vL;             /* value of left operand           */
    UInt       qR;             /* size of right operand's field   */
    UInt       dR;             /* degee of right operand's field  */
    TypFFE              vR;             /* value of one element of list    */
    UInt       i;              /* loop variable                   */

    /* both operands lie in the same field                                 */
    if ( FLD_VECFFE( hdL ) == FLD_VECFFE( hdR ) ) {

        /* make the result vector                                          */
        len = LEN_VECFFE( hdL );
        if ( len != LEN_VECFFE( hdR ) )
         return Error("Vector +: vectors must have the same length", 0, 0);
        hdS = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( len ) );

        /* get the field and its successor table                           */
        hdFld = FLD_VECFFE( hdL );
        fld = SUCC_FF( hdFld );
        SET_FLD_VECFFE( hdS, hdFld );

        /* loop over the entries and add                                   */
        for ( i = 1; i <= len; i++ ) {
            vL = VAL_VECFFE( hdL, i );
            vR = VAL_VECFFE( hdR, i );
            SET_VAL_VECFFE( hdS, i, SUM_FF( vL, vR, fld ) );
        }

    }

    /* otherwise try to lift the operands into a common field              */
    else {

        /* make the result vector                                          */
        len = LEN_VECFFE( hdL );
        if ( len != LEN_VECFFE( hdR ) )
         return Error("Vector +: vectors must have the same length", 0, 0);
        hdS = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( len ) );

        /* get a common field and its successor table                      */
        p = CharVecFFE( hdL );
        if ( SIZE_FF( FLD_VECFFE( hdR ) ) % p != 0 )
         return Error("Vector +: operands must lie in a common field", 0, 0);
        dL = DegreeVecFFE( hdL );
        dR = DegreeVecFFE( hdR );
        for ( d = 1, q = p; d % dR != 0 || d % dL != 0; d++ )  q *= p;
        if ( (  2 <= p && 17 <= d) || (  3 <= p && 11 <= d)
          || (  5 <= p &&  7 <= d) || (  7 <= p &&  6 <= d)
          || ( 11 <= p &&  5 <= d) || ( 17 <= p &&  4 <= d)
          || ( 41 <= p &&  3 <= d) || (257 <= p &&  2 <= d) )
         return Error("Vector +: smallest common field too large", 0, 0);
        if ( (SIZE_FF(FLD_VECFFE(hdL))-1) % (q-1) == 0 )
            hdFld = FLD_VECFFE( hdL );
        else if ( (SIZE_FF(FLD_VECFFE(hdR))-1) % (q-1) == 0 )
            hdFld = FLD_VECFFE( hdR );
        else
            hdFld = FLD_FFE( RootFiniteField( q ) );
        fld = SUCC_FF( hdFld );
        q = SIZE_FF( hdFld );
        SET_FLD_VECFFE( hdS, hdFld );

        /* get the left operand's field size                               */
        qL = SIZE_FF( FLD_VECFFE( hdL ) );

        /* get the right operand's field size                              */
        qR = SIZE_FF( FLD_VECFFE( hdR ) );

        /* loop over the entries and add                                   */
        for ( i = 1; i <= len; i++ ) {
            vL = VAL_VECFFE( hdL, i );
            if ( vL != 0 )  vL = (vL-1)*(q-1)/(qL-1)+1;
            vR = VAL_VECFFE( hdR, i );
            if ( vR != 0 )  vR = (vR-1)*(q-1)/(qR-1)+1;
            SET_VAL_VECFFE( hdS, i, SUM_FF( vL, vR, fld ) );
        }

    }

    /* return the result                                                   */
    return hdS;
}


/****************************************************************************
**
*F  SumVectorFFE(<hdL>,<hdR>) . . . . . . . . . sum of integer vector and ffe
**
**  'SumVectorFFE' returns the sum of the integer vector <hdL> and the finite
**  field  element  <hdR>.   This  is   an  important function   because such
**  constructs are used to create finite field vectors.
*/
Bag       SumVectorFFE (Bag hdL, Bag hdR)
{
    Bag           hdS;            /* sum, result                     */
    TypFFE              vS;             /* value of the sum                */
    UInt       len;            /* length of left operand vector   */
    Bag           hdFld;          /* handle of the field             */
    TypFFE              * fld;          /* successor table of field        */
    UInt       q;              /* size of finite field            */
    Bag           hdLL;           /* one element of left operand     */
    Int                l;              /* integer value of this element   */
    TypFFE              vL;             /* ffe value of this element       */
    TypFFE              vR;             /* value of right operand          */
    UInt       i;              /* loop variable                   */

    /* delegate the work if the vector does not only contain integers      */
    if ( GET_TYPE_BAG( ELMF_LIST( hdL, 1 ) ) != T_INT )
        return SumListScl( hdL, hdR );

    /* get the field                                                       */
    hdFld = FLD_FFE( hdR );
    fld = SUCC_FF( hdFld );
    q = SIZE_FF( hdFld );

    /* make the result vector                                              */
    len = LEN_LIST( hdL );
    hdS = NewBag( T_VECFFE, SIZE_PLEN_VECFFE(len) );
    SET_LEN_VECFFE( hdS, len );
    SET_FLD_VECFFE( hdS, hdFld );

    /* loop over the entries of the left operand                           */
    for ( i = 1; i <= len; i++ ) {

        /* get and check the element                                       */
        hdLL = ELMF_LIST( hdL, i );
        if ( GET_TYPE_BAG(hdLL) != T_INT ) {
            return Error("operations: sum of %s and %s is not defined",
                    (Int)NameType[GET_TYPE_BAG(hdLL)], (Int)NameType[GET_TYPE_BAG(hdR)] );
        }

        /* get the values of the two operands                              */
        l = (HD_TO_INT(hdLL) % (Int)q + q) % q;
        if ( l == 0 )  vL = 0;
        else for ( vL = 1; 1 < l; --l )  vL = (vL == 0 ? 1 : fld[vL]);
        vR = VAL_FFE(hdR);

        /* compute the sum and enter in result vector                      */
        vS = SUM_FF( vL, vR, fld );
        SET_VAL_VECFFE( hdS, i, vS );

    }

    /* return the result                                                   */
    return hdS;
}


/****************************************************************************
**
*F  SumFFEVector(<hdL>,<hdR>) . . . . . . . . . sum of ffe and integer vector
**
**  'SumVectorFFE' returns the sum of the  finite field element <hdL> and the
**  integer vector  <hdR>.    This is an   important function  because   such
**  constructs are used to create finite field vectors.
*/
Bag       SumFFEVector (Bag hdL, Bag hdR)
{
    Bag           hdS;            /* sum, result                     */
    TypFFE              vS;             /* value of the sum                */
    UInt       len;            /* length of left operand vector   */
    Bag           hdFld;          /* handle of the field             */
    TypFFE              * fld;          /* successor table of field        */
    UInt       q;              /* size of finite field            */
    TypFFE              vL;             /* value of left operand           */
    Bag           hdRR;           /* one element of right operand    */
    Int                r;              /* integer value of this element   */
    TypFFE              vR;             /* ffe value of this element       */
    UInt       i;              /* loop variable                   */

    /* delegate the work if the vector does not only contain integers      */
    if ( GET_TYPE_BAG( ELMF_LIST( hdR, 1 ) ) != T_INT )
        return SumSclList( hdL, hdR );

    /* get the field                                                       */
    hdFld = FLD_FFE( hdL );
    fld = SUCC_FF( hdFld );
    q = SIZE_FF( hdFld );

    /* make the result vector                                              */
    len = LEN_LIST( hdR );
    hdS = NewBag( T_VECFFE, SIZE_PLEN_VECFFE(len) );
    SET_LEN_VECFFE( hdS, len );
    SET_FLD_VECFFE( hdS, hdFld );

    /* loop over the entries of the left operand                           */
    for ( i = 1; i <= len; i++ ) {

        /* get and check the element                                       */
        hdRR = ELMF_LIST( hdR, i );
        if ( GET_TYPE_BAG(hdRR) != T_INT ) {
            return Error("operations: sum of %s and %s is not defined",
                    (Int)NameType[GET_TYPE_BAG(hdL)], (Int)NameType[GET_TYPE_BAG(hdRR)] );
        }

        /* get the values of the two operands                              */
        vL = VAL_FFE(hdL);
        r = (HD_TO_INT(hdRR) % (Int)q + q) % q;
        if ( r == 0 )  vR = 0;
        else for ( vR = 1; 1 < r; --r )  vR = (vR == 0 ? 1 : fld[vR]);

        /* compute the sum and enter in result vector                      */
        vS = SUM_FF( vL, vR, fld );
        SET_VAL_VECFFE( hdS, i, vS );

    }

    /* return the result                                                   */
    return hdS;
}


/****************************************************************************
**
*F  DiffFFEVecFFE(<hdL>,<hdR>)   diff. of a finite field element and a vector
**
**  'DiffFFEVecFFE' returns the  difference of the finite field element <hdL>
**  and the  vector <hdR>.  The difference is a new  list, where each element
**  is the difference of <hdL> and the corresponding entry of <hdR>.
**
**  'DiffFFEVecFFE' is an improved version  of 'DiffSclList', which  does not
**  call 'DIFF'.
*/
Bag       DiffFFEVecFFE (Bag hdL, Bag hdR)
{
    Bag           hdD;            /* difference, result              */
    UInt       len;            /* length of the list              */
    UInt       p;              /* characteristic                  */
    UInt       q;              /* size of common field            */
    UInt       d;              /* degree of common field          */
    Bag           hdFld;          /* handle of common field          */
    TypFFE *            fld;            /* successor table of common field */
    UInt       qL;             /* size of left operand's field    */
    UInt       dL;             /* degree of left operand's field  */
    TypFFE              vL;             /* value of left operand           */
    UInt       qR;             /* size of right operand's field   */
    UInt       dR;             /* degee of right operand's field  */
    TypFFE              vR;             /* value of one element of list    */
    UInt       i;              /* loop variable                   */

    /* both operands lie in the same field                                 */
    if ( FLD_FFE( hdL ) == FLD_VECFFE( hdR ) ) {

        /* make the result vector                                          */
        len = LEN_VECFFE( hdR );
        hdD = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( len ) );

        /* get the field and its successor table                           */
        hdFld = FLD_FFE( hdL );
        fld = SUCC_FF( hdFld );
        SET_FLD_VECFFE( hdD, hdFld );

        /* get the left operand's value                                    */
        vL = VAL_FFE( hdL );

        /* loop over the entries and subtract                              */
        for ( i = 1; i <= len; i++ ) {
            vR = VAL_VECFFE( hdR, i );
            vR = NEG_FF( vR, fld );
            SET_VAL_VECFFE( hdD, i, SUM_FF( vL, vR, fld ) );
        }

    }

    /* otherwise try to lift the operands into a common field              */
    else {

        /* make the result vector                                          */
        len = LEN_VECFFE( hdR );
        hdD = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( len ) );

        /* get a common field and its successor table                      */
        p = CharFFE( hdL );
        if ( SIZE_FF( FLD_VECFFE( hdR ) ) % p != 0 )
         return Error("Vector -: operands must lie in a common field", 0, 0);
        dL = DegreeFFE( hdL );
        dR = DegreeVecFFE( hdR );
        for ( d = 1, q = p; d % dR != 0 || d % dL != 0; d++ )  q *= p;
        if ( (  2 <= p && 17 <= d) || (  3 <= p && 11 <= d)
          || (  5 <= p &&  7 <= d) || (  7 <= p &&  6 <= d)
          || ( 11 <= p &&  5 <= d) || ( 17 <= p &&  4 <= d)
          || ( 41 <= p &&  3 <= d) || (257 <= p &&  2 <= d) )
         return Error("Vector -: smallest common field too large", 0, 0);
        if ( (SIZE_FF(FLD_FFE(hdL))-1) % (q-1) == 0 )
            hdFld = FLD_FFE( hdL );
        else if ( (SIZE_FF(FLD_VECFFE(hdR))-1) % (q-1) == 0 )
            hdFld = FLD_VECFFE( hdR );
        else
            hdFld = FLD_FFE( RootFiniteField( q ) );
        fld = SUCC_FF( hdFld );
        q = SIZE_FF( hdFld );
        SET_FLD_VECFFE( hdD, hdFld );

        /* get the left operand's field size and value                     */
        qL = SIZE_FF( FLD_FFE( hdL ) );
        vL = VAL_FFE( hdL );
        if ( vL != 0 )  vL = (vL-1)*(q-1)/(qL-1)+1;

        /* get the right operand's field size                              */
        qR = SIZE_FF( FLD_VECFFE( hdR ) );

        /* loop over the entries and subtract                              */
        for ( i = 1; i <= len; i++ ) {
            vR = VAL_VECFFE( hdR, i );
            if ( vR != 0 )  vR = (vR-1)*(q-1)/(qR-1)+1;
            vR = NEG_FF( vR, fld );
            SET_VAL_VECFFE( hdD, i, SUM_FF( vL, vR, fld ) );
        }

    }

    /* return the result                                                   */
    return hdD;
}


/****************************************************************************
**
*F  DiffVecFFEFFE(<hdL>,<hdR>)   diff. of a vector and a finite field element
**
**  'DiffVecFFEFFE' returns the difference of the vector <hdL> and the finite
**  field element <hdR>.  The difference is a new list, where each element is
**  the difference of <hdL> and the corresponding entry of <hdR>.
**
**  'DiffVecFFEFFE'  is an  improved version of 'DiffListScl', which does not
**  call 'DIFF'.
*/
Bag       DiffVecFFEFFE (Bag hdL, Bag hdR)
{
    Bag           hdD;            /* difference, result              */
    UInt       len;            /* length of the list              */
    UInt       p;              /* characteristic                  */
    UInt       q;              /* size of common field            */
    UInt       d;              /* degree of common field          */
    Bag           hdFld;          /* handle of common field          */
    TypFFE *            fld;            /* successor table of common field */
    UInt       qL;             /* size of left operand's field    */
    UInt       dL;             /* degree of left operand's field  */
    TypFFE              vL;             /* value of left operand           */
    UInt       qR;             /* size of right operand's field   */
    UInt       dR;             /* degee of right operand's field  */
    TypFFE              vR;             /* value of one element of list    */
    UInt       i;              /* loop variable                   */

    /* both operands lie in the same field                                 */
    if ( FLD_VECFFE( hdL ) == FLD_FFE( hdR ) ) {

        /* make the result vector                                          */
        len = LEN_VECFFE( hdL );
        hdD = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( len ) );

        /* get the field and its successor table                           */
        hdFld = FLD_VECFFE( hdL );
        fld = SUCC_FF( hdFld );
        SET_FLD_VECFFE( hdD, hdFld );

        /* get the right operand's value                                   */
        vR = VAL_FFE( hdR );
        vR = NEG_FF( vR, fld );

        /* loop over the entries and subtract                              */
        for ( i = 1; i <= len; i++ ) {
            vL = VAL_VECFFE( hdL, i );
            SET_VAL_VECFFE( hdD, i, SUM_FF( vL, vR, fld ) );
        }

    }

    /* otherwise try to lift the operands into a common field              */
    else {

        /* make the result vector                                          */
        len = LEN_VECFFE( hdL );
        hdD = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( len ) );

        /* get a common field and its successor table                      */
        p = CharVecFFE( hdL );
        if ( SIZE_FF( FLD_FFE( hdR ) ) % p != 0 )
         return Error("Vector -: operands must lie in a common field", 0, 0);
        dL = DegreeVecFFE( hdL );
        dR = DegreeFFE( hdR );
        for ( d = 1, q = p; d % dR != 0 || d % dL != 0; d++ )  q *= p;
        if ( (  2 <= p && 17 <= d) || (  3 <= p && 11 <= d)
          || (  5 <= p &&  7 <= d) || (  7 <= p &&  6 <= d)
          || ( 11 <= p &&  5 <= d) || ( 17 <= p &&  4 <= d)
          || ( 41 <= p &&  3 <= d) || (257 <= p &&  2 <= d) )
         return Error("Vector -: smallest common field too large", 0, 0);
        if ( (SIZE_FF(FLD_VECFFE(hdL))-1) % (q-1) == 0 )
            hdFld = FLD_VECFFE( hdL );
        else if ( (SIZE_FF(FLD_FFE(hdR))-1) % (q-1) == 0 )
            hdFld = FLD_FFE( hdR );
        else
            hdFld = FLD_FFE( RootFiniteField( q ) );
        fld = SUCC_FF( hdFld );
        q = SIZE_FF( hdFld );
        SET_FLD_VECFFE( hdD, hdFld );

        /* get the right operand's field size and value                    */
        qR = SIZE_FF( FLD_FFE( hdR ) );
        vR = VAL_FFE( hdR );
        if ( vR != 0 )  vR = (vR-1)*(q-1)/(qR-1)+1;
        vR = NEG_FF( vR, fld );

        /* get the left operand's field size                               */
        qL = SIZE_FF( FLD_VECFFE( hdL ) );

        /* loop over the entries and subtract                              */
        for ( i = 1; i <= len; i++ ) {
            vL = VAL_VECFFE( hdL, i );
            if ( vL != 0 )  vL = (vL-1)*(q-1)/(qL-1)+1;
            SET_VAL_VECFFE( hdD, i, SUM_FF( vL, vR, fld ) );
        }

    }

    /* return the result                                                   */
    return hdD;
}


/****************************************************************************
**
*F  DiffVecFFEVecFFE(<hdL>,<hdR>) . . . . . . . . . difference of two vectors
**
**  'DiffVecFFEVecFFE' returns the difference of the  two  vectors  <hdL> and
**  <hdR>.   The  difference  is  a  new  list,  where  each  element is  the
**  difference of the corresponding entries of <hdL> and <hdR>.
**
**  'DiffVecFFEVecFFE' is an  improved version of 'DiffListList',  which does
**  not call 'PROD'.
*/
Bag       DiffVecFFEVecFFE (Bag hdL, Bag hdR)
{
    Bag           hdD;            /* difference, result              */
    UInt       len;            /* length of the list              */
    UInt       p;              /* characteristic                  */
    UInt       q;              /* size of common field            */
    UInt       d;              /* degree of common field          */
    Bag           hdFld;          /* handle of common field          */
    TypFFE *            fld;            /* successor table of common field */
    UInt       qL;             /* size of left operand's field    */
    UInt       dL;             /* degree of left operand's field  */
    TypFFE              vL;             /* value of left operand           */
    UInt       qR;             /* size of right operand's field   */
    UInt       dR;             /* degee of right operand's field  */
    TypFFE              vR;             /* value of one element of list    */
    UInt       i;              /* loop variable                   */

    /* both operands lie in the same field                                 */
    if ( FLD_VECFFE( hdL ) == FLD_VECFFE( hdR ) ) {

        /* make the result vector                                          */
        len = LEN_VECFFE( hdL );
        if ( len != LEN_VECFFE( hdR ) )
         return Error("Vector -: vectors must have the same length", 0, 0);
        hdD = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( len ) );

        /* get the field and its successor table                           */
        hdFld = FLD_VECFFE( hdL );
        fld = SUCC_FF( hdFld );
        SET_FLD_VECFFE( hdD, hdFld );

        /* loop over the entries and subtract                              */
        for ( i = 1; i <= len; i++ ) {
            vL = VAL_VECFFE( hdL, i );
            vR = VAL_VECFFE( hdR, i );
            vR = NEG_FF( vR, fld );
            SET_VAL_VECFFE( hdD, i, SUM_FF( vL, vR, fld ) );
        }

    }

    /* otherwise try to lift the operands into a common field              */
    else {

        /* make the result vector                                          */
        len = LEN_VECFFE( hdL );
        if ( len != LEN_VECFFE( hdR ) )
         return Error("Vector -: vectors must have the same length", 0, 0);
        hdD = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( len ) );

        /* get a common field and its successor table                      */
        p = CharVecFFE( hdL );
        if ( SIZE_FF( FLD_VECFFE( hdR ) ) % p != 0 )
         return Error("Vector -: operands must lie in a common field", 0, 0);
        dL = DegreeVecFFE( hdL );
        dR = DegreeVecFFE( hdR );
        for ( d = 1, q = p; d % dR != 0 || d % dL != 0; d++ )  q *= p;
        if ( (  2 <= p && 17 <= d) || (  3 <= p && 11 <= d)
          || (  5 <= p &&  7 <= d) || (  7 <= p &&  6 <= d)
          || ( 11 <= p &&  5 <= d) || ( 17 <= p &&  4 <= d)
          || ( 41 <= p &&  3 <= d) || (257 <= p &&  2 <= d) )
         return Error("Vector -: smallest common field too large", 0, 0);
        if ( (SIZE_FF(FLD_VECFFE(hdL))-1) % (q-1) == 0 )
            hdFld = FLD_VECFFE( hdL );
        else if ( (SIZE_FF(FLD_VECFFE(hdR))-1) % (q-1) == 0 )
            hdFld = FLD_VECFFE( hdR );
        else
            hdFld = FLD_FFE( RootFiniteField( q ) );
        fld = SUCC_FF( hdFld );
        q = SIZE_FF( hdFld );
        SET_FLD_VECFFE( hdD, hdFld );

        /* get the left operand's field size                               */
        qL = SIZE_FF( FLD_VECFFE( hdL ) );

        /* get the right operand's field size                              */
        qR = SIZE_FF( FLD_VECFFE( hdR ) );

        /* loop over the entries and subtract                              */
        for ( i = 1; i <= len; i++ ) {
            vL = VAL_VECFFE( hdL, i );
            if ( vL != 0 )  vL = (vL-1)*(q-1)/(qL-1)+1;
            vR = VAL_VECFFE( hdR, i );
            if ( vR != 0 )  vR = (vR-1)*(q-1)/(qR-1)+1;
            vR = NEG_FF( vR, fld );
            SET_VAL_VECFFE( hdD, i, SUM_FF( vL, vR, fld ) );
        }

    }

    /* return the result                                                   */
    return hdD;
}


/****************************************************************************
**
*F  DiffVectorFFE(<hdL>,<hdR>)  . . . .  difference of integer vector and ffe
**
**  'DiffVectorFFE'  returns the difference  of the  integer vector <hdL> and
**  the finite field  element <hdR>.  This  is an important  function because
**  such constructs are used to create finite field vectors.
*/
Bag       DiffVectorFFE (Bag hdL, Bag hdR)
{
    Bag           hdD;            /* difference, result              */
    TypFFE              vD;             /* value of the difference         */
    UInt       len;            /* length of left operand vector   */
    Bag           hdFld;          /* handle of the field             */
    TypFFE              * fld;          /* successor table of field        */
    UInt       q;              /* size of finite field            */
    Bag           hdLL;           /* one element of left operand     */
    Int                l;              /* integer value of this element   */
    TypFFE              vL;             /* ffe value of this element       */
    TypFFE              vR;             /* value of right operand          */
    UInt       i;              /* loop variable                   */

    /* delegate the work if the vector does not only contain integers      */
    if ( GET_TYPE_BAG( ELMF_LIST( hdL, 1 ) ) != T_INT )
        return DiffListScl( hdL, hdR );

    /* get the field                                                       */
    hdFld = FLD_FFE( hdR );
    fld = SUCC_FF( hdFld );
    q = SIZE_FF( hdFld );

    /* make the result vector                                              */
    len = LEN_LIST( hdL );
    hdD = NewBag( T_VECFFE, SIZE_PLEN_VECFFE(len) );
    SET_LEN_VECFFE( hdD, len );
    SET_FLD_VECFFE( hdD, hdFld );

    /* loop over the entries of the left operand                           */
    for ( i = 1; i <= len; i++ ) {

        /* get and check the element                                       */
        hdLL = ELMF_LIST( hdL, i );
        if ( GET_TYPE_BAG(hdLL) != T_INT ) {
            return Error("operations: sum of %s and %s is not defined",
                    (Int)NameType[GET_TYPE_BAG(hdLL)], (Int)NameType[GET_TYPE_BAG(hdR)] );
        }

        /* get the values of the two operands                              */
        l = (HD_TO_INT(hdLL) % (Int)q + q) % q;
        if ( l == 0 )  vL = 0;
        else for ( vL = 1; 1 < l; --l )  vL = (vL == 0 ? 1 : fld[vL]);
        vR = VAL_FFE(hdR);

        /* compute the sum and enter in result vector                      */
        vR = NEG_FF( vR, fld );
        vD = SUM_FF( vL, vR, fld );
        SET_VAL_VECFFE( hdD, i, vD );

    }

    /* return the result                                                   */
    return hdD;
}


/****************************************************************************
**
*F  DiffFFEVector(<hdL>,<hdR>)  . . . .  difference of ffe and integer vector
**
**  'DiffVectorFFE' returns the difference of  the finite field element <hdL>
**  and the integer vector <hdR>.  This is an important function because such
**  constructs are used to create finite field vectors.
*/
Bag       DiffFFEVector (Bag hdL, Bag hdR)
{
    Bag           hdD;            /* difference, result              */
    TypFFE              vD;             /* value of the difference         */
    UInt       len;            /* length of left operand vector   */
    Bag           hdFld;          /* handle of the field             */
    TypFFE              * fld;          /* successor table of field        */
    UInt       q;              /* size of finite field            */
    TypFFE              vL;             /* value of left operand           */
    Bag           hdRR;           /* one element of right operand    */
    Int                r;              /* integer value of this element   */
    TypFFE              vR;             /* ffe value of this element       */
    UInt       i;              /* loop variable                   */

    /* delegate the work if the vector does not only contain integers      */
    if ( GET_TYPE_BAG( ELMF_LIST( hdR, 1 ) ) != T_INT )
        return DiffSclList( hdL, hdR );

    /* get the field                                                       */
    hdFld = FLD_FFE( hdL );
    fld = SUCC_FF( hdFld );
    q = SIZE_FF( hdFld );

    /* make the result vector                                              */
    len = LEN_LIST( hdR );
    hdD = NewBag( T_VECFFE, SIZE_PLEN_VECFFE(len) );
    SET_LEN_VECFFE( hdD, len );
    SET_FLD_VECFFE( hdD, hdFld );

    /* loop over the entries of the left operand                           */
    for ( i = 1; i <= len; i++ ) {

        /* get and check the element                                       */
        hdRR = ELMF_LIST( hdR, i );
        if ( GET_TYPE_BAG(hdRR) != T_INT ) {
            return Error("operations: sum of %s and %s is not defined",
                    (Int)NameType[GET_TYPE_BAG(hdL)], (Int)NameType[GET_TYPE_BAG(hdRR)] );
        }

        /* get the values of the two operands                              */
        vL = VAL_FFE(hdL);
        r = (HD_TO_INT(hdRR) % (Int)q + q) % q;
        if ( r == 0 )  vR = 0;
        else for ( vR = 1; 1 < r; --r )  vR = (vR == 0 ? 1 : fld[vR]);

        /* compute the sum and enter in result vector                      */
        vR = NEG_FF( vR, fld );
        vD = SUM_FF( vL, vR, fld );
        SET_VAL_VECFFE( hdD, i, vD );

    }

    /* return the result                                                   */
    return hdD;
}


/****************************************************************************
**
*F  ProdFFEVecFFE(<hdL>,<hdR>) product of a finite field element and a vector
**
**  'ProdFFEVecFFE' returns the product of the finite field element <hdL> and
**  the vector <hdR>.  The product is  a new list,  where each element is the
**  product of <hdL> and the corresponding entry of <hdR>.
**
**  'ProdFFEVecFFE' is an improved version  of 'ProdSclList', which  does not
**  call 'PROD'.
*/
Bag       ProdFFEVecFFE (Bag hdL, Bag hdR)
{
    Bag           hdP;            /* product, result                 */
    UInt       len;            /* length of the list              */
    UInt       p;              /* characteristic                  */
    UInt       q;              /* size of common field            */
    UInt       d;              /* degree of common field          */
    Bag           hdFld;          /* handle of common field          */
    TypFFE *            fld;            /* successor table of common field */
    UInt       qL;             /* size of left operand's field    */
    UInt       dL;             /* degree of left operand's field  */
    TypFFE              vL;             /* value of left operand           */
    UInt       qR;             /* size of right operand's field   */
    UInt       dR;             /* degee of right operand's field  */
    TypFFE              vR;             /* value of one element of list    */
    UInt       i;              /* loop variable                   */

    /* both operands lie in the same field                                 */
    if ( FLD_FFE( hdL ) == FLD_VECFFE( hdR ) ) {

        /* make the result vector                                          */
        len = LEN_VECFFE( hdR );
        hdP = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( len ) );

        /* get the field and its successor table                           */
        hdFld = FLD_FFE( hdL );
        fld = SUCC_FF( hdFld );
        SET_FLD_VECFFE( hdP, hdFld );

        /* get the left operand's value                                    */
        vL = VAL_FFE( hdL );

        /* loop over the entries and multiply                              */
        for ( i = 1; i <= len; i++ ) {
            vR = VAL_VECFFE( hdR, i );
            SET_VAL_VECFFE( hdP, i, PROD_FF( vL, vR, fld ) );
        }

    }

    /* otherwise try to lift the operands into a common field              */
    else {

        /* make the result vector                                          */
        len = LEN_VECFFE( hdR );
        hdP = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( len ) );

        /* get a common field and its successor table                      */
        p = CharFFE( hdL );
        if ( SIZE_FF( FLD_VECFFE( hdR ) ) % p != 0 )
         return Error("Vector *: operands must lie in a common field", 0, 0);
        dL = DegreeFFE( hdL );
        dR = DegreeVecFFE( hdR );
        for ( d = 1, q = p; d % dR != 0 || d % dL != 0; d++ )  q *= p;
        if ( (  2 <= p && 17 <= d) || (  3 <= p && 11 <= d)
          || (  5 <= p &&  7 <= d) || (  7 <= p &&  6 <= d)
          || ( 11 <= p &&  5 <= d) || ( 17 <= p &&  4 <= d)
          || ( 41 <= p &&  3 <= d) || (257 <= p &&  2 <= d) )
         return Error("Vector *: smallest common field too large", 0, 0);
        if ( (SIZE_FF(FLD_FFE(hdL))-1) % (q-1) == 0 )
            hdFld = FLD_FFE( hdL );
        else if ( (SIZE_FF(FLD_VECFFE(hdR))-1) % (q-1) == 0 )
            hdFld = FLD_VECFFE( hdR );
        else
            hdFld = FLD_FFE( RootFiniteField( q ) );
        fld = SUCC_FF( hdFld );
        q = SIZE_FF( hdFld );
        SET_FLD_VECFFE( hdP, hdFld );

        /* get the left operand's field size and value                     */
        qL = SIZE_FF( FLD_FFE( hdL ) );
        vL = VAL_FFE( hdL );
        if ( vL != 0 )  vL = (vL-1)*(q-1)/(qL-1)+1;

        /* get the right operand's field size                              */
        qR = SIZE_FF( FLD_VECFFE( hdR ) );

        /* loop over the entries and multiply                              */
        for ( i = 1; i <= len; i++ ) {
            vR = VAL_VECFFE( hdR, i );
            if ( vR != 0 )  vR = (vR-1)*(q-1)/(qR-1)+1;
            SET_VAL_VECFFE( hdP, i, PROD_FF( vL, vR, fld ) );
        }

    }

    /* return the result                                                   */
    return hdP;
}


/****************************************************************************
**
*F  ProdVecFFEFFE(<hdL>,<hdR>) product of a vector and a finite field element
**
**  'ProdVecFFEFFE' returns  the  product of the vector <hdL>  and the finite
**  field element <hdR>.   The product  is a new list,  where each element is
**  the product of <hdL> and the corresponding entry of <hdR>.
**
**  'ProdVecFFEFFE'  is an  improved version of 'ProdListScl', which does not
**  call 'PROD'.
*/
Bag       ProdVecFFEFFE (Bag hdL, Bag hdR)
{
    Bag           hdP;            /* product, result                 */
    UInt       len;            /* length of the list              */
    UInt       p;              /* characteristic                  */
    UInt       q;              /* size of common field            */
    UInt       d;              /* degree of common field          */
    Bag           hdFld;          /* handle of common field          */
    TypFFE *            fld;            /* successor table of common field */
    UInt       qL;             /* size of left operand's field    */
    UInt       dL;             /* degree of left operand's field  */
    TypFFE              vL;             /* value of left operand           */
    UInt       qR;             /* size of right operand's field   */
    UInt       dR;             /* degee of right operand's field  */
    TypFFE              vR;             /* value of one element of list    */
    UInt       i;              /* loop variable                   */

    /* both operands lie in the same field                                 */
    if ( FLD_VECFFE( hdL ) == FLD_FFE( hdR ) ) {

        /* make the result vector                                          */
        len = LEN_VECFFE( hdL );
        hdP = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( len ) );

        /* get the field and its successor table                           */
        hdFld = FLD_VECFFE( hdL );
        fld = SUCC_FF( hdFld );
        SET_FLD_VECFFE( hdP, hdFld );

        /* get the right operand's value                                   */
        vR = VAL_FFE( hdR );

        /* loop over the entries and multiply                              */
        for ( i = 1; i <= len; i++ ) {
            vL = VAL_VECFFE( hdL, i );
            SET_VAL_VECFFE( hdP, i, PROD_FF( vL, vR, fld ) );
        }

    }

    /* otherwise try to lift the operands into a common field              */
    else {

        /* make the result vector                                          */
        len = LEN_VECFFE( hdL );
        hdP = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( len ) );

        /* get a common field and its successor table                      */
        p = CharVecFFE( hdL );
        if ( SIZE_FF( FLD_FFE( hdR ) ) % p != 0 )
         return Error("Vector *: operands must lie in a common field", 0, 0);
        dL = DegreeVecFFE( hdL );
        dR = DegreeFFE( hdR );
        for ( d = 1, q = p; d % dR != 0 || d % dL != 0; d++ )  q *= p;
        if ( (  2 <= p && 17 <= d) || (  3 <= p && 11 <= d)
          || (  5 <= p &&  7 <= d) || (  7 <= p &&  6 <= d)
          || ( 11 <= p &&  5 <= d) || ( 17 <= p &&  4 <= d)
          || ( 41 <= p &&  3 <= d) || (257 <= p &&  2 <= d) )
         return Error("Vector *: smallest common field too large", 0, 0);
        if ( (SIZE_FF(FLD_VECFFE(hdL))-1) % (q-1) == 0 )
            hdFld = FLD_VECFFE( hdL );
        else if ( (SIZE_FF(FLD_FFE(hdR))-1) % (q-1) == 0 )
            hdFld = FLD_FFE( hdR );
        else
            hdFld = FLD_FFE( RootFiniteField( q ) );
        fld = SUCC_FF( hdFld );
        q = SIZE_FF( hdFld );
        SET_FLD_VECFFE( hdP, hdFld );

        /* get the right operand's field size and value                    */
        qR = SIZE_FF( FLD_FFE( hdR ) );
        vR = VAL_FFE( hdR );
        if ( vR != 0 )  vR = (vR-1)*(q-1)/(qR-1)+1;

        /* get the left operand's field size                               */
        qL = SIZE_FF( FLD_VECFFE( hdL ) );

        /* loop over the entries and multiply                              */
        for ( i = 1; i <= len; i++ ) {
            vL = VAL_VECFFE( hdL, i );
            if ( vL != 0 )  vL = (vL-1)*(q-1)/(qL-1)+1;
            SET_VAL_VECFFE( hdP, i, PROD_FF( vL, vR, fld ) );
        }

    }

    /* return the result                                                   */
    return hdP;
}


/****************************************************************************
**
*F  ProdVecFFEVecFFE(<hdL>,<hdR>) . . . . . . . . . .  product of two vectors
**
**  'ProdVecFFEVecFFE'  returns  the product  of the  two  vectors <hdL>  and
**  <hdR>.  The product is the sum of the corresponding entries of <hdL>  and
**  <hdR>.
**
**  'ProdVecFFEVecFFE' is an improved version  of 'ProdListList',  which does
**  not call 'PROD'.
*/
Bag       ProdVecFFEVecFFE (Bag hdL, Bag hdR)
{
    Bag           hdP;            /* product, result                 */
    TypFFE              vP;             /* value of the product            */
    UInt       len;            /* length of the list              */
    UInt       p;              /* characteristic                  */
    UInt       q;              /* size of common field            */
    UInt       d;              /* degree of common field          */
    Bag           hdFld;          /* handle of common field          */
    TypFFE *            fld;            /* successor table of common field */
    UInt       qL;             /* size of left operand's field    */
    UInt       dL;             /* degree of left operand's field  */
    TypFFE              vL;             /* value of left operand           */
    UInt       qR;             /* size of right operand's field   */
    UInt       dR;             /* degee of right operand's field  */
    TypFFE              vR;             /* value of one element of list    */
    TypFFE              vQ;             /* temporary value                 */
    UInt       i;              /* loop variable                   */

    /* both operands lie in the same field                                 */
    if ( FLD_VECFFE( hdL ) == FLD_VECFFE( hdR ) ) {

        /* make the result vector                                          */
        len = LEN_VECFFE( hdL );
        if ( len != LEN_VECFFE( hdR ) )
         return Error("Vector *: vectors must have the same length", 0, 0);
        hdP = NewBag( T_FFE, SIZE_HD + sizeof(TypFFE) );

        /* get the field and its successor table                           */
        hdFld = FLD_VECFFE( hdL );
        fld = SUCC_FF( hdFld );
        SET_FLD_FFE( hdP, hdFld );

        /* loop over the entries and multiply                              */
        vP = 0;
        for ( i = 1; i <= len; i++ ) {
            vL = VAL_VECFFE( hdL, i );
            vR = VAL_VECFFE( hdR, i );
            vQ = PROD_FF( vL, vR, fld );
            vP = SUM_FF( vP, vQ, fld );
        }

        /* enter the value                                                 */
        SET_VAL_FFE( hdP, vP );

    }

    /* otherwise try to lift the operands into a common field              */
    else {

        /* make the result vector                                          */
        len = LEN_VECFFE( hdL );
        if ( len != LEN_VECFFE( hdR ) )
         return Error("Vector *: vectors must have the same length", 0, 0);
        hdP = NewBag( T_FFE, SIZE_HD + sizeof(TypFFE) );

        /* get a common field and its successor table                      */
        p = CharVecFFE( hdL );
        if ( SIZE_FF( FLD_VECFFE( hdR ) ) % p != 0 )
         return Error("Vector *: operands must lie in a common field", 0, 0);
        dL = DegreeVecFFE( hdL );
        dR = DegreeVecFFE( hdR );
        for ( d = 1, q = p; d % dR != 0 || d % dL != 0; d++ )  q *= p;
        if ( (  2 <= p && 17 <= d) || (  3 <= p && 11 <= d)
          || (  5 <= p &&  7 <= d) || (  7 <= p &&  6 <= d)
          || ( 11 <= p &&  5 <= d) || ( 17 <= p &&  4 <= d)
          || ( 41 <= p &&  3 <= d) || (257 <= p &&  2 <= d) )
         return Error("Vector *: smallest common field too large", 0, 0);
        if ( (SIZE_FF(FLD_VECFFE(hdL))-1) % (q-1) == 0 )
            hdFld = FLD_VECFFE( hdL );
        else if ( (SIZE_FF(FLD_VECFFE(hdR))-1) % (q-1) == 0 )
            hdFld = FLD_VECFFE( hdR );
        else
            hdFld = FLD_FFE( RootFiniteField( q ) );
        fld = SUCC_FF( hdFld );
        q = SIZE_FF( hdFld );
        SET_FLD_FFE( hdP, hdFld );

        /* get the left operand's field size                               */
        qL = SIZE_FF( FLD_VECFFE( hdL ) );

        /* get the right operand's field size                              */
        qR = SIZE_FF( FLD_VECFFE( hdR ) );

        /* loop over the entries and multiply                              */
        vP = 0;
        for ( i = 1; i <= len; i++ ) {
            vL = VAL_VECFFE( hdL, i );
            if ( vL != 0 )  vL = (vL-1)*(q-1)/(qL-1)+1;
            vR = VAL_VECFFE( hdR, i );
            if ( vR != 0 )  vR = (vR-1)*(q-1)/(qR-1)+1;
            vQ = PROD_FF( vL, vR, fld );
            vP = SUM_FF( vP, vQ, fld );
        }

        /* enter the value                                                 */
        SET_VAL_FFE( hdP, vP );

    }

    /* return the result                                                   */
    return hdP;
}


/****************************************************************************
**
*F  ProdVecFFEMatFFE(<hdL>,<hdR>) . . . . .  product of a vector and a matrix
**
**  'ProdVecFFEMatFFE' returns the product of the vector <hdL> and the matrix
**  <hdR>.  The product is the sum of the  rows of <hdR>, each  multiplied by
**  the corresponding entry of <hdL>.
**
**  'ProdVecFFEMatFFE'  is  an improved version of 'ProdListList', which does
**  not  call  'PROD'  and also  acummulates the sum  into  one fixed  vector
**  instead of allocating a new for each product and sum.
*/
Bag       ProdVecFFEMatFFE (Bag hdL, Bag hdR)
{
    Bag           hdP;            /* product, result                 */
    TypFFE              vP;             /* value of the product            */
    UInt       len;            /* length of the list              */
    UInt       col;            /* length of the rows              */
    UInt       p;              /* characteristic                  */
    UInt       q;              /* size of common field            */
    UInt       d;              /* degree of common field          */
    Bag           hdFld;          /* handle of common field          */
    TypFFE *            fld;            /* successor table of common field */
    UInt       qL;             /* size of left operand's field    */
    UInt       dL;             /* degree of left operand's field  */
    TypFFE              vL;             /* value of left operand           */
    Bag           hdRR;           /* one row of the right operand    */
    UInt       qR;             /* size of right operand's field   */
    UInt       dR;             /* degee of right operand's field  */
    TypFFE              vR;             /* value of one element of row     */
    TypFFE              vQ;             /* temporary value                 */
    UInt       i, k;           /* loop variable                   */

    /* both operands lie in the same field                                 */
    if ( FLD_VECFFE( hdL ) == FLD_VECFFE( ELM_PLIST( hdR, 1 ) ) ) {

        /* make the result vector                                          */
        len = LEN_VECFFE( hdL );
        col = LEN_VECFFE( ELM_PLIST( hdR, 1 ) );
        if ( len != LEN_PLIST( hdR ) )
         return Error("Vector *: vectors must have the same length", 0, 0);
        hdP = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( col ) );

        /* get the field and its successor table                           */
        hdFld = FLD_VECFFE( hdL );
        fld = SUCC_FF( hdFld );
        SET_FLD_VECFFE( hdP, hdFld );

        /* loop over the entries and multiply                              */
        for ( i = 1; i <= len; i++ ) {
            vL = VAL_VECFFE( hdL, i );
            hdRR = ELM_PLIST( hdR, i );
            if ( vL == 1 ) {
                for ( k = 1; k <= col; k++ ) {
                    vR = VAL_VECFFE( hdRR, k );
                    vP = VAL_VECFFE( hdP, k );
                    vP = SUM_FF( vP, vR, fld );
                    SET_VAL_VECFFE( hdP, k, vP );
                }
            }
            else if ( vL != 0 ) {
                for ( k = 1; k <= col; k++ ) {
                    vR = VAL_VECFFE( hdRR, k );
                    vP = VAL_VECFFE( hdP, k );
                    vQ = PROD_FF( vL, vR, fld );
                    vP = SUM_FF( vP, vQ, fld );
                    SET_VAL_VECFFE( hdP, k, vP );
                }
            }
        }

    }

    /* otherwise try to lift the operands into a common field              */
    else {

        /* make the result vector                                          */
        len = LEN_VECFFE( hdL );
        col = LEN_VECFFE( ELM_PLIST( hdR, 1 ) );
        if ( len != LEN_PLIST( hdR ) )
         return Error("Vector *: vectors must have the same length", 0, 0);
        hdP = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( col ) );

        /* get a common field and its successor table                      */
        p = CharVecFFE( hdL );
        if ( SIZE_FF( FLD_VECFFE( ELM_PLIST( hdR, 1 ) ) ) % p != 0 )
         return Error("Vector *: operands must lie in a common field", 0, 0);
        dL = DegreeVecFFE( hdL );

        /*N maybe one should use 'DegreeMatFFE'                            */
        qR = SIZE_FF( FLD_VECFFE( ELM_PLIST( hdR, 1 ) ) );
        for ( dR = 1,  q = p;  q != qR;  q *= p,  dR++ ) ;

        for ( d = 1, q = p; d % dR != 0 || d % dL != 0; d++ )  q *= p;
        if ( (  2 <= p && 17 <= d) || (  3 <= p && 11 <= d)
          || (  5 <= p &&  7 <= d) || (  7 <= p &&  6 <= d)
          || ( 11 <= p &&  5 <= d) || ( 17 <= p &&  4 <= d)
          || ( 41 <= p &&  3 <= d) || (257 <= p &&  2 <= d) )
         return Error("Vector *: smallest common field too large", 0, 0);
        if ( (SIZE_FF(FLD_VECFFE(hdL))-1) % (q-1) == 0 )
            hdFld = FLD_VECFFE( hdL );
        else if ( (SIZE_FF(FLD_VECFFE(ELM_PLIST(hdR,1)))-1) % (q-1) == 0 )
            hdFld = FLD_VECFFE( ELM_PLIST( hdR, 1 ) );
        else
            hdFld = FLD_FFE( RootFiniteField( q ) );
        fld = SUCC_FF( hdFld );
        q = SIZE_FF( hdFld );
        SET_FLD_VECFFE( hdP, hdFld );

        /* get the left operand's field size                               */
        qL = SIZE_FF( FLD_VECFFE( hdL ) );

        /* get the right operand's field size                              */
        qR = SIZE_FF( FLD_VECFFE( ELM_PLIST( hdR, 1 ) ) );

        /* loop over the entries and multiply                              */
        for ( i = 1; i <= len; i++ ) {
            vL = VAL_VECFFE( hdL, i );
            if ( vL != 0 )  vL = (vL-1)*(q-1)/(qL-1)+1;
            hdRR = ELM_PLIST( hdR, i );
            if ( vL == 1 ) {
                for ( k = 1; k <= col; k++ ) {
                    vR = VAL_VECFFE( hdRR, k );
                    if ( vR != 0 )  vR = (vR-1)*(q-1)/(qR-1)+1;
                    vP = VAL_VECFFE( hdP, k );
                    vP = SUM_FF( vP, vR, fld );
                    SET_VAL_VECFFE( hdP, k, vP );
                }
            }
            else if ( vL != 0 ) {
                for ( k = 1; k <= col; k++ ) {
                    vR = VAL_VECFFE( hdRR, k );
                    if ( vR != 0 )  vR = (vR-1)*(q-1)/(qR-1)+1;
                    vP = VAL_VECFFE( hdP, k );
                    vQ = PROD_FF( vL, vR, fld );
                    vP = SUM_FF( vP, vQ, fld );
                    SET_VAL_VECFFE( hdP, k, vP );
                }
            }
        }

    }

    /* return the result                                                   */
    return hdP;
}


/****************************************************************************
**
*F  ProdVectorFFE(<hdL>,<hdR>)  . . . . . . product of integer vector and ffe
**
**  'ProdVectorFFE' returns  the product of  the integer vector <hdL> and the
**  finite field  element <hdR>.  This is  an important function because such
**  constructs are used to create finite field vectors.
*/
Bag       ProdVectorFFE (Bag hdL, Bag hdR)
{
    Bag           hdP;            /* product, result                 */
    TypFFE              vP;             /* value of the product            */
    UInt       len;            /* length of left operand vector   */
    Bag           hdFld;          /* handle of the field             */
    TypFFE              * fld;          /* successor table of field        */
    UInt       q;              /* size of finite field            */
    Bag           hdLL;           /* one element of left operand     */
    Int                l;              /* integer value of this element   */
    TypFFE              vL;             /* ffe value of this element       */
    TypFFE              vR;             /* value of right operand          */
    UInt       i;              /* loop variable                   */

    /* delegate the work if the vector does not only contain integers      */
    if ( GET_TYPE_BAG( ELMF_LIST( hdL, 1 ) ) != T_INT )
        return ProdListScl( hdL, hdR );

    /* get the field                                                       */
    hdFld = FLD_FFE( hdR );
    fld = SUCC_FF( hdFld );
    q = SIZE_FF( hdFld );

    /* make the result vector                                              */
    len = LEN_LIST( hdL );
    hdP = NewBag( T_VECFFE, SIZE_PLEN_VECFFE(len) );
    SET_LEN_VECFFE( hdP, len );
    SET_FLD_VECFFE( hdP, hdFld );

    /* loop over the entries of the left operand                           */
    for ( i = 1; i <= len; i++ ) {

        /* get and check the element                                       */
        hdLL = ELMF_LIST( hdL, i );
        if ( GET_TYPE_BAG(hdLL) != T_INT ) {
            return Error("operations: product of %s and %s is not defined",
                    (Int)NameType[GET_TYPE_BAG(hdLL)], (Int)NameType[GET_TYPE_BAG(hdR)] );
        }

        /* get the values of the two operands                              */
        l = (HD_TO_INT(hdLL) % (Int)q + q) % q;
        if ( l == 0 )  vL = 0;
        else for ( vL = 1; 1 < l; --l )  vL = (vL == 0 ? 1 : fld[vL]);
        vR = VAL_FFE(hdR);

        /* compute the product and enter in result vector                  */
        vP = PROD_FF( vL, vR, fld );
        SET_VAL_VECFFE( hdP, i, vP );

    }

    /* return the result                                                   */
    return hdP;
}


/****************************************************************************
**
*F  ProdFFEVector(<hdL>,<hdR>)  . . . . . . product of ffe and integer vector
**
**  'ProdVectorFFE' returns the product of the finite field element <hdL> and
**  the integer vector  <hdR>.  This  is an  important function  because such
**  constructs are used to create finite field vectors.
*/
Bag       ProdFFEVector (Bag hdL, Bag hdR)
{
    Bag           hdP;            /* product, result                 */
    TypFFE              vP;             /* value of the product            */
    UInt       len;            /* length of left operand vector   */
    Bag           hdFld;          /* handle of the field             */
    TypFFE              * fld;          /* successor table of field        */
    UInt       q;              /* size of finite field            */
    TypFFE              vL;             /* value of left operand           */
    Bag           hdRR;           /* one element of right operand    */
    Int                r;              /* integer value of this element   */
    TypFFE              vR;             /* ffe value of this element       */
    UInt       i;              /* loop variable                   */

    /* delegate the work if the vector does not only contain integers      */
    if ( GET_TYPE_BAG( ELMF_LIST( hdR, 1 ) ) != T_INT )
        return ProdSclList( hdL, hdR );

    /* get the field                                                       */
    hdFld = FLD_FFE( hdL );
    fld = SUCC_FF( hdFld );
    q = SIZE_FF( hdFld );

    /* make the result vector                                              */
    len = LEN_LIST( hdR );
    hdP = NewBag( T_VECFFE, SIZE_PLEN_VECFFE(len) );
    SET_LEN_VECFFE( hdP, len );
    SET_FLD_VECFFE( hdP, hdFld );

    /* loop over the entries of the left operand                           */
    for ( i = 1; i <= len; i++ ) {

        /* get and check the element                                       */
        hdRR = ELMF_LIST( hdR, i );
        if ( GET_TYPE_BAG(hdRR) != T_INT ) {
            return Error("operations: product of %s and %s is not defined",
                    (Int)NameType[GET_TYPE_BAG(hdL)], (Int)NameType[GET_TYPE_BAG(hdRR)] );
        }

        /* get the values of the two operands                              */
        vL = VAL_FFE(hdL);
        r = (HD_TO_INT(hdRR) % (Int)q + q) % q;
        if ( r == 0 )  vR = 0;
        else for ( vR = 1; 1 < r; --r )  vR = (vR == 0 ? 1 : fld[vR]);

        /* compute the product and enter in result vector                  */
        vP = PROD_FF( vL, vR, fld );
        SET_VAL_VECFFE( hdP, i, vP );

    }

    /* return the result                                                   */
    return hdP;
}


/****************************************************************************
**
*F  PowMatFFEInt(<hdL>,<hdR>) . . . . . . .  power of a matrix and an integer
**
**  'PowMatFFEInt' returns the <hdR>-th power of the matrix <hdL>, which must
**  be a square matrix of course.
**
**  Note that  this  function also  does the  inversion  of matrices when the
**  exponent is negative.
*/
Bag       PowMatFFEInt (Bag hdL, Bag hdR)
{
    Bag           hdP = 0;        /* power, result                   */
    Bag           hdPP;           /* one row of the power            */
    Bag           hdQQ;           /* another row of the power        */
    TypFFE              ppp;            /* one value of the row            */
    TypFFE              qqq;            /* one value of another row        */
    Bag           hdLL;           /* one row of left operand         */
    TypFFE              tmp;            /* temporary value                 */
    Bag           hdFld;          /* field                           */
    TypFFE *            fld;            /* it's successor table            */
    UInt       len;            /* length (and width) of matrix    */
    Int                e;              /* exponent                        */
    UInt       i, k, l;        /* loop variables                  */

    /* check that the operand is a square matrix                           */
    len = LEN_PLIST( hdL );
    if ( len != LEN_VECFFE( ELM_PLIST( hdL, 1 ) ) ) {
        return Error(
          "Matrix operations ^: <mat> must be square",
                      0, 0);
    }
    hdFld = FLD_VECFFE( ELM_PLIST( hdL, 1 ) );

    /* if the right operand is zero, make the identity matrix              */
    if ( GET_TYPE_BAG(hdR) == T_INT && hdR == INT_TO_HD(0) ) {
        hdP = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
        SET_LEN_PLIST( hdP, len );
        for ( i = 1; i <= len; i++ ) {
            hdPP = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( len ) );
            SET_FLD_VECFFE( hdPP, hdFld );
            SET_ELM_PLIST( hdP, i, hdPP );
        }
        for ( i = 1; i <= len; i++ ) {
            hdPP = ELM_PLIST( hdP, i );
            for ( k = 1; k <= len; k++ )
                SET_VAL_VECFFE( hdPP, k, 0 );
            SET_VAL_VECFFE( hdPP, i, 1 );
        }
    }

    /* if the right operand is one, make a copy                            */
    if ( GET_TYPE_BAG(hdR) == T_INT && hdR == INT_TO_HD(1) ) {
        hdP = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
        SET_LEN_PLIST( hdP, len );
        for ( i = 1; i <= len; i++ ) {
            hdPP = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( len ) );
            SET_FLD_VECFFE( hdPP, hdFld );
            SET_ELM_PLIST( hdP, i, hdPP );
        }
        for ( i = 1; i <= len; i++ ) {
            hdPP = ELM_PLIST( hdP, i );
            hdLL = ELM_PLIST( hdL, i );
            for ( k = 1; k <= len; k++ )
                SET_VAL_VECFFE( hdPP, k, VAL_VECFFE( hdLL, k ) );
        }
    }

    /* if the right operand is negative, invert the matrix                 */
    if ( (GET_TYPE_BAG(hdR) == T_INT && HD_TO_INT(hdR) < 0)
      || (GET_TYPE_BAG(hdR) == T_INTNEG) ) {

        /* make a matrix of the form $ ( Id_<len> | <mat> ) $              */
        hdP = NewBag( T_LIST, SIZE_PLEN_PLIST( len ) );
        SET_LEN_PLIST( hdP, len );
        for ( i = 1; i <= len; i++ ) {
            hdPP = NewBag( T_VECFFE, SIZE_PLEN_VECFFE( 2 * len ) );
            SET_FLD_VECFFE( hdPP, hdFld );
            SET_ELM_PLIST( hdP, i, hdPP );
        }
        for ( i = 1; i <= len; i++ ) {
            hdPP = ELM_PLIST( hdP, i );
            for ( k = 1; k <= len; k++ )
                SET_VAL_VECFFE( hdPP, k, 0 );
            SET_VAL_VECFFE( hdPP, i, 1 );
        }
        for ( i = 1; i <= len; i++ ) {
            hdPP = ELM_PLIST( hdP, i );
            hdLL = ELM_PLIST( hdL, i );
            for ( k = 1; k <= len; k++ )
                SET_VAL_VECFFE( hdPP, k + len, VAL_VECFFE( hdLL, k ) );
        }

        /* get the successor table                                         */
        fld = SUCC_FF( hdFld );

        /* make row operations to reach form $ ( <inv> | Id_<len> ) $      */
        /* loop over the columns of <mat>                                  */
        for ( k = len+1; k <= 2*len; k++ ) {

            /* find a nonzero entry in this column                         */
            for ( i = k-len;
                  i <= len && VAL_VECFFE( ELM_PLIST(hdP,i), k ) == 0;
                  i++ )
                ;
            if ( len < i )
                return Error("Matrix: <mat> must be invertible", 0, 0);

            /* make the row the <k>-th row and normalize it                */
            hdPP = ELM_PLIST( hdP, i );
            SET_ELM_PLIST( hdP, i, ELM_PLIST( hdP, k-len ) );
            SET_ELM_PLIST( hdP, k-len, hdPP );
            ppp = QUO_FF( 1, VAL_VECFFE( hdPP, k ), fld );
            for ( l = 1; l <= 2*len; l++ ) {
                tmp = PROD_FF( ppp, VAL_VECFFE( hdPP, l ), fld );
                SET_VAL_VECFFE( hdPP, l, tmp );
            }

            /* clear all entries in this column                            */
            for ( i = 1; i <= len; i++ ) {
                hdQQ = ELM_PLIST( hdP, i );
                ppp = NEG_FF( VAL_VECFFE( hdQQ, k ), fld );
                if ( i != k-len && ppp != 0 ) {
                    for ( l = 1; l <= 2*len; l++ ) {
                        tmp = PROD_FF( ppp, VAL_VECFFE( hdPP, l ), fld );
                        qqq = VAL_VECFFE( hdQQ, l );
                        tmp = SUM_FF( qqq, tmp, fld );
                        SET_VAL_VECFFE( hdQQ, l, tmp );
                    }
                }
            }

        }

        /* throw away the right halves of each row                         */
        for ( i = 1; i <= len; i++ ) {
            Resize( ELM_PLIST( hdP, i ), SIZE_PLEN_VECFFE( len ) );
        }

        /* assign back to left, invert exponent (only if immediate)        */
        hdL = hdP;
        if ( GET_TYPE_BAG(hdR) == T_INT )  hdR = INT_TO_HD( -HD_TO_INT(hdR) );

    }

    /* repeated squaring with an immediate integer                         */
    /* the loop invariant is: <res> = <p>^<k> * <l>^<e>, <e> < <k>         */
    /* <p> = 0 means that <p> is the identity matrix                       */
    if ( GET_TYPE_BAG(hdR) == T_INT && 1 < HD_TO_INT(hdR) ) {
        hdP = 0;
        k = NUM_TO_UINT(1) << 31;
        e = HD_TO_INT(hdR);
        while ( 1 < k ) {
            hdP = (hdP == 0 ? hdP : ProdListScl( hdP, hdP ));
            k = k / 2;
            if ( k <= e ) {
                hdP = (hdP == 0 ? hdL : ProdListScl( hdP, hdL ));
                e = e - k;
            }
        }
    }

    /* repeated squaring with a large integer                              */
    if ( GET_TYPE_BAG(hdR) != T_INT ) {
        hdP = 0;
        for ( i = GET_SIZE_BAG(hdR)/sizeof(TypDigit); 0 < i; i-- ) {
            k = NUM_TO_UINT(1) << (8*sizeof(TypDigit));
            e = ((TypDigit*) PTR_BAG(hdR))[i-1];
            while ( 1 < k ) {
                hdP = (hdP == 0 ? hdP : ProdListScl( hdP, hdP ));
                k = k / 2;
                if ( k <= e ) {
                    hdP = (hdP == 0 ? hdL : ProdListScl( hdP, hdL ));
                    e = e - k;
                }
            }
        }
    }

    /* return the result                                                   */
    return hdP;
}


/****************************************************************************
**
*F  PrVecFFE(<hdList>)  . . . . . . . . . . . . . . . . . . .  print a vector
**
**  'PrVecFFE' prints a vector.
*/
void            PrVecFFE (Bag hdList)
{
    UInt       len;            /* logical length of the list      */
    UInt       i;              /* loop variable                   */

    /* compute the length of the list                                      */
    len = LEN_VECFFE(hdList);

    /* loop over the entries                                               */
    Pr("%2>[ %2>", 0, 0);
    for ( i = 1;  i <= len;  i++ ) {
        if ( 1 < i )  Pr("%2<, %2>", 0, 0);
        PrFF( PTR_BAG(hdList)[0], VAL_VECFFE(hdList,i) );
    }
    Pr(" %4<]", 0, 0);

}


/****************************************************************************
**
*F  DepthVecFFE( <hdVec> )  . . . . . . . . .  depth of a finite field vector
*/
Bag DepthVecFFE (Bag hdVec)
{
    Int                pos;            /* current position                */
    Int                len;            /* length of <hdVec>               */

    len = LEN_VECFFE(hdVec);
    for ( pos = 1;  pos <= len;  pos++ )
        if ( VAL_VECFFE( hdVec, pos ) != 0 )
            break;
    
    /* and return the position                                             */
    return INT_TO_HD(pos);
}


/****************************************************************************
**
*F  CharVecFFE(<hdVec>) . . . . . . . . . . . . .  characteristic of a vector
**
**  'CharVecFFE' returns  the  characteristic  of  the  field  in  which  the
**  elements of the finite field vector <hdVec> lie.
*/
Int            CharVecFFE (Bag hdVec)
{
    UInt       p;              /* characteristic, result          */
    UInt       q;              /* size of the finite field        */

    /* get the size of the finite field                                    */
    q = SIZE_FF( FLD_VECFFE( hdVec ) );

    /* simply find the smallest prime that divides the size                */
    if ( q % 2 == 0 ) {
        p = 2;
    }
    else {
        for ( p = 3; q % p != 0; p += 2 )
            ;
    }

    /* return the result                                                   */
    return p;
}


/****************************************************************************
**
*F  CharMatFFE(<hdMat>) . . . . . . . . . . . . .  characteristic of a matrix
**
**  'CharMatFFE'  returns  the  characteristic of  the  field  in  which  the
**  elements of the finite field matrix <hdMat> lie.
*/
Int            CharMatFFE (Bag hdMat)
{
    return CharVecFFE( ELM_PLIST( hdMat, 1 ) );
}


/****************************************************************************
**
*F  FunCharFFE( <hdCall> )  . . . .  characteristic of a finite field element
**
**  'FunCharFFE' implements the internal function 'CharFFE'.
**
**  'CharFFE( <z> )' or 'CharFFE( <vec> )' or 'CharFFE( <mat> )'
**
**  'CharFFE' returns the  characteristic of the   smallest finite field  <F>
**  containing the element <z>, respectively all elements of the vector <vec>
**  over a finite field (see "Vectors"), or matrix  <mat> over a finite field
**  (see "Matrices").
*/
Bag       FunCharFFE (Bag hdCall)
{
    UInt       p;              /* characteristic, result          */
    Bag           hdZ;            /* finite field element            */

    /* check the arguments                                                 */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: CharFFE( <z> )", 0, 0);
    hdZ = EVAL( PTR_BAG(hdCall)[1] );

    /* dispatch                                                            */
    if ( GET_TYPE_BAG(hdZ) == T_FFE ) {
        p = CharFFE( hdZ );
    }
    else if ( IsXTypeVecFFE( hdZ ) ) {
        p = CharVecFFE( hdZ );
    }
    else if ( IsXTypeMatFFE( hdZ ) ) {
        p = CharMatFFE( hdZ );
    }
    else {
        return Error(
            "CharFFE: <z> must be a finite field element, vector, or matrix",
                      0, 0);
    }

    /* return the result                                                   */
    return INT_TO_HD( p );
}


/****************************************************************************
**
*F  DegreeVecFFE(<hdVec>) . . . . . . . . . . . . . . . .  degree of a vector
**
**  'DegreeVecFFE' returns  the  degree  of the  smallest finite  field  that
**  contains all elements of the finite field vector <hdVec>.
*/
Int            DegreeVecFFE (Bag hdVec)
{
    UInt       d;              /* degree, result                  */
    UInt       len;            /* length of the vector            */
    UInt       p;              /* characteristic, result          */
    UInt       q;              /* size of the finite field        */
    TypFFE              v;              /* value of an element             */
    UInt       q1;             /* size of subfield containing val */
    UInt       d1;             /* deg  of subfield containing val */
    UInt       i, k;           /* loop variable                   */

    /* get the size and characteristic of the finite field                 */
    q = SIZE_FF( FLD_FFE( hdVec ) );
    if ( q % 2 == 0 )  p = 2;
    else for ( p = 3; q % p != 0; p += 2 )  ;

    /* loop over all elements of the vector                                */
    d = 1;
    len = LEN_VECFFE( hdVec );
    for ( i = 1; i <= len; i++ ) {

        /* get the value of the finite field element                       */
        v = VAL_VECFFE( hdVec, i );

        /* get the degree of the smallest field that contains the element  */
        q1 = p;
        d1 = 1;
        if ( v != 0 ) {
            while ( (q-1)%(q1-1) != 0 || (v-1)%((q-1)/(q1-1)) != 0 ) {
                q1 *= p;
                d1 += 1;
            }
        }

        /* compute the lcm with the previous minimal degree                */
        for ( k = d; d % d1 != 0; d += k )  ;

    }

    /* return the result                                                   */
    return d;
}


/****************************************************************************
**
*F  DegreeMatFFE(<hdMat>) . . . . . . . . . . . . . . . .  degree of a matrix
**
**  'DegreeMatFFE' returns  the  degree  of the  smallest finite  field  that
**  contains all elements of the finite field matrix <hdMat>.
*/
Int            DegreeMatFFE (Bag hdMat)
{
    UInt       d;              /* degree, result                  */
    UInt       len;            /* length of the matrix            */
    UInt       p;              /* characteristic, result          */
    UInt       q;              /* size of the finite field        */
    Bag           hdElm;          /* one row of the matrix           */
    UInt       d1;             /* deg  of subfield containing row */
    UInt       i, k;           /* loop variable                   */

    /* get the size and characteristic of the finite field                 */
    q = SIZE_FF( FLD_FFE( ELM_PLIST( hdMat, 1 ) ) );
    if ( q % 2 == 0 ) p = 2;
    else for ( p = 3; q % p != 0; p += 2 ) ;

    /* loop over all elements of the vector                                */
    d = 1;
    len = LEN_PLIST( hdMat );
    for ( i = 1; i <= len; i++ ) {

        /* get the row                                                     */
        hdElm = ELM_PLIST( hdMat, i );

        /* get the degree of the smallest field that contains the element  */
        d1 = DegreeVecFFE( hdElm );

        /* compute the lcm with the previous minimal degree                */
        for ( k = d; d % d1 != 0; d += k )  ;

    }

    /* return the result                                                   */
    return d;
}


/****************************************************************************
**
*F  FunDegreeFFE( <hdCall> )  . . . . . . .  degree of a finite field element
**
**  'FunDegreeFFE' implements the internal function 'DegreeFFE'.
**
**  'DegreeFFE( <z> )' or 'DegreeFFE( <vec> )' or 'DegreeFFE( <mat> )'
**
**  'DegreeFFE'   returns the   degree   of  the  smallest  finite  field <F>
**  containing the element <z>, respectively all elements of the vector <vec>
**  over a finite field (see "Vectors"), or matrix <mat> over a  finite field
**  (see  "Matrices").  For vectors and matrices,  an error  is raised if the
**  smallest finite field containing all elements of the vector or matrix has
**  order larger than $2^{16}$.
*/
Bag       FunDegreeFFE (Bag hdCall)
{
    UInt       d;              /* degree, result                  */
    Bag           hdZ;            /* finite field element            */

    /* check the arguments                                                 */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: DegreeFFE( <z> )", 0, 0);
    hdZ = EVAL( PTR_BAG(hdCall)[1] );

    /* dispatch                                                            */
    if ( GET_TYPE_BAG(hdZ) == T_FFE ) {
        d = DegreeFFE( hdZ );
    }
    else if ( IsXTypeVecFFE( hdZ ) ) {
        d = DegreeVecFFE( hdZ );
    }
    else if ( IsXTypeMatFFE( hdZ ) ) {
        d = DegreeMatFFE( hdZ );
    }
    else {
        return Error(
          "DegreeFFE: <z> must be a finite field element, vector, or matrix",
                      0, 0);
    }

    /* return the result                                                   */
    return INT_TO_HD( d );
}


/****************************************************************************
**
*F  FunLogVecFFE( <hdCall> )  . . . . . . . . . internal function 'LogVecFFE'
**
**  'FunLogVecFFE' implements the internal function 'LogVecFFE'.
**
**  'LogVecFFE( <vector>, <position> )'
*/
Bag       FunLogVecFFE (Bag hdCall)
{
    Int                exp, pos;
    Bag           hdPos, hdVec;

    if ( GET_SIZE_BAG( hdCall ) != 3 * SIZE_HD )
        return Error("usage: LogVecFFE( <vector>, <position> )", 0, 0);
    hdVec = EVAL( PTR_BAG(hdCall)[1] );
    hdPos = EVAL( PTR_BAG(hdCall)[2] );
    if ( ! IsVector(hdVec) || GET_TYPE_BAG(hdVec)!=T_VECFFE || GET_TYPE_BAG(hdPos)!=T_INT )
        return Error("usage: LogVecFFE( <vector>, <position> )", 0, 0);

    pos = HD_TO_INT( hdPos );
    if (pos <= 0 || LEN_VECFFE( hdVec ) < pos)
        return Error("LogVecFFE: <position> argument is out of range", 0, 0);

    exp = VAL_VECFFE( hdVec, pos );

    if (exp == 0)   return HdFalse;
    else            return INT_TO_HD( exp-1 );
}


/****************************************************************************
**
*F  FunIntVecFFE( <hdCall> )  . . . . . . . . . internal function 'IntVecFFE'
**
**  'FunIntVecFFE' implements the internal function 'IntVecFFE'.
**
**  'IntVecFFE( <vec> )'
**  'IntVecFFE( <vec>, <pos> )'
*/
Bag (*TabIntVecFFE[T_VAR]) ( Bag, Int );

Bag FunIntVecFFE (Bag hdCall)
{
    Int                pos;
    Bag           hdPos;
    Bag           hdVec;

    /* check arguments                                                     */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD && GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )
        return Error( "usage: IntVecFFE( <vec>, <pos> )",  0,  0 );
    hdVec = EVAL(PTR_BAG(hdCall)[1]);
    if ( T_REC <= GET_TYPE_BAG(hdVec) || GET_TYPE_BAG(hdVec) < T_LIST )
        return Error( "IntVecFFE: <list> must be a finite field vector", 
                       0,  0 );
    if ( GET_SIZE_BAG(hdCall) == 2 * SIZE_HD )
        pos = 0;
    else {
        hdPos = EVAL( PTR_BAG(hdCall)[2] );
        if ( GET_TYPE_BAG(hdPos) != T_INT )
            return Error( "<pos> must be an integer",  0,  0 );
        pos = HD_TO_INT(hdPos);
        if ( pos <= 0 )
            return Error( "List Element: <pos> must be a positive integer",
                           0,  0 );
        if ( LEN_LIST(hdVec) < pos )
            return Error( "List Element: <list>[%d] must have a value",
                          (Int)pos,  0 );
    }

    /* jump through the table 'TabIntVecFFE'                               */
    return TabIntVecFFE[XType(hdVec)]( hdVec, pos );
}

Bag CantIntVecFFE (Bag hdList, Int pos)
{
    return Error("IntVecFFE: <list> must be a finite field vector", 0, 0);
}

Bag IntVecFFE (Bag hdVec, Int pos)
{
    Bag           hdRes;          /* result                          */
    Bag           hdElm;          /* converted element of <hdVec>    */
    Bag           tab;            /* conversion table                */
    Int                len;            /* length of <hdVec>               */
    Int                i;              /* loop variable                   */

    /* compute the conversion table                                        */
    tab = ConvTabIntFFE( SIZE_FF( FLD_VECFFE(hdVec) ) );
    
    /* if <pos> is 0  convert the whole vector                             */
    if ( pos == 0 ) {
        len   = LEN_LIST(hdVec);
        hdRes = NewBag( T_LIST, SIZE_PLEN_PLIST(len) );
        SET_LEN_PLIST( hdRes, len );
        for ( i = len;  0 < i;  i-- ) {
            hdElm = ELM_PLIST( tab, VAL_VECFFE(hdVec,i)+1 );
            if ( hdElm == 0 )
                return Error( "IntVecFFE: <z> must lie in the prime field",
                               0,  0 );
            SET_ELM_PLIST( hdRes, i, hdElm );
        }
    }

    /* convert a single element                                            */
    else {
        hdRes = ELM_PLIST( tab, VAL_VECFFE(hdVec,pos)+1 );
        if ( hdRes == 0 )
            return Error( "IntVecFFE: <z> must lie in the prime field",
                           0,  0 );
    }

    /* return the converted vector or element                              */
    return hdRes;
}


/****************************************************************************
**
*F  FunMakeVecFFE( <hdCall> ) . . . . . . . .  internal function 'MakeVecFFE'
**
**  'FunMakeVecFFE' implements the internal function 'MakeVecFFE'.
**
**  'MakeVecFFE( <list>, <ffe> )'
*/
Bag       FunMakeVecFFE (Bag hdCall)
{
    Bag           hdList;         /* <list>, first argument          */
    Bag           hdFFE;          /* <ffe>, second argument          */
    UInt       len;            /* logical length of <list>        */
    Bag           hdElm;          /* one element of <list>           */
    Bag           hdFF;           /* finite field of <ffe>           */
    TypFFE *            field;          /* successor table of the field    */
    UInt       p;              /* characteristic of the field     */
    TypFFE              l;              /* value of the left operand       */
    TypFFE              r;              /* value of the right operand      */
    UInt       i;              /* loop variable                   */

    /* get and check the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD ) {
        return Error("usage: MakeVecFFE( <list>, <ffe> )", 0, 0);
    }
    hdList = EVAL(PTR_BAG(hdCall)[1]);
    if ( IS_LIST(hdList) && LEN_LIST(hdList) == 0 ) {
        return HdVoid;
    }
    if ( XType(hdList) != T_VECTOR ) {
        return Error("MakeVecFFE: <list> must be a list of integers", 0, 0);
    }
    len = LEN_LIST(hdList);
    hdFFE  = EVAL(PTR_BAG(hdCall)[2]);
    if ( GET_TYPE_BAG(hdFFE) != T_FFE ) {
        return Error("MakeVecFFE: <ffe> must be finite field element", 0, 0);
    }
    hdFF  = FLD_FFE(hdFFE);
    field = SUCC_FF(hdFF);
    p     = CharFFE(hdFFE);

    /* loop over the entries and multiply them                             */
    for ( i = 1;  i <= len;  i++ ) {
        hdElm = ELM_LIST( hdList, i );
        if ( hdElm == 0 || GET_TYPE_BAG(hdElm) != T_INT ) {
            return Error(
              "MakeVecFFE: <list> must be a list of integers",
                          0, 0);
        }
        l = HD_TO_INT(hdElm);
        l = (l % p + p) % p;
        if ( l == 0 )  r = 0;
        else for ( r = 1; 1 < l; --l )  r = (r == 0 ? 1 : field[r]);
        l = VAL_FFE(hdFFE);
        SET_VAL_VECFFE( hdList, i, PROD_FF( l, r, field ) );
    }

    /* convert the list to a vector                                        */
    Retype( hdList, T_VECFFE );
    SET_FLD_VECFFE( hdList, hdFF );
    SET_LEN_VECFFE( hdList, len );
    return HdVoid;
}


/****************************************************************************
**
*F  FunNumberVecFFE( <hdCall> ) . . . . . .  internal function 'NumberVecFFE'
**
**  'FunNumberVecFFE' implements the internal function 'NumberVecFFE'.
**
**  'NumberVecFFE( <vector>, <powers>, <integers> )'
*/
Bag       FunNumberVecFFE (Bag hdCall)
{
    Int                num, dim, exp, i;
    Bag           hdVec, hdPows, hdInts;

    if ( GET_SIZE_BAG( hdCall ) != 4 * SIZE_HD )
        return Error("usage: NumberVecFFE(<vector>,<powers>,<integers>)",
                      0, 0);

    hdVec  = EVAL( PTR_BAG(hdCall)[1] );
    hdPows = EVAL( PTR_BAG(hdCall)[2] );
    hdInts = EVAL( PTR_BAG(hdCall)[3] );
    if ( ! IsVector(hdVec) || GET_TYPE_BAG(hdVec) != T_VECFFE
      || (GET_TYPE_BAG(hdPows) != T_LIST && GET_TYPE_BAG(hdPows) != T_VECTOR)
      || (GET_TYPE_BAG(hdInts) != T_LIST && GET_TYPE_BAG(hdInts) != T_VECTOR) )
        return Error("usage: NumberVecFFE(<vector>,<powers>,<integers>)",
                      0, 0);

    if (LEN_VECFFE( hdVec ) < LEN_LIST( hdPows ))
        return Error("NumberVecFFE: <vector> is shorter than <powers>",
                      0, 0);

    if (SIZE_FF( FLD_VECFFE( hdVec ) ) != (LEN_LIST( hdInts )+1))
        return Error("NumberVecFFE: <integers> has bad length", 0, 0);

    num = 1;
    dim = LEN_LIST( hdPows );
    for (i = 1; i <= dim; ++i) {
        exp = VAL_VECFFE( hdVec, i );
        if (exp != 0)
           num += HD_TO_INT( ELM_PLIST( hdPows, i ) )
                * HD_TO_INT( ELM_PLIST( hdInts, exp ) );
    }

    return INT_TO_HD( num );
}


/****************************************************************************
**
*F  InitVecFFE()  . . . . . . . . . . . . . . . . . initialize vector package
**
**  'InitVecFFE' initializes the finite field vector package.
*/
void            InitVecFFE (void)
{
    Int                type;

    /* install the list functions in the tables                            */
    TabIsList     [T_VECFFE] = 2;
    TabIsList     [T_MATFFE] = 3;
    TabLenList    [T_VECFFE] = LenVecFFE;
    TabElmList    [T_VECFFE] = ElmVecFFE;
    TabElmfList   [T_VECFFE] = ElmfVecFFE;
    TabElmlList   [T_VECFFE] = ElmlVecFFE;
    TabElmrList   [T_VECFFE] = ElmrVecFFE;
    TabElmsList   [T_VECFFE] = ElmsVecFFE;
    TabAssList    [T_VECFFE] = AssVecFFE;
    TabAsssList   [T_VECFFE] = AsssVecFFE;
    TabPosList    [T_VECFFE] = PosVecFFE;
    TabPlainList  [T_VECFFE] = PlainVecFFE;
    TabIsDenseList[T_VECFFE] = IsDenseVecFFE;
    TabIsPossList [T_VECFFE] = IsPossVecFFE;
    TabIsXTypeList[T_VECFFE] = IsXTypeVecFFE;
    TabIsXTypeList[T_MATFFE] = IsXTypeMatFFE;

    /* install tables for gap functions                                    */
    for ( type = T_LIST;  type < T_REC;  type++ )
        TabIntVecFFE[type] = CantIntVecFFE;
    TabIntVecFFE  [T_VECFFE] = IntVecFFE;
    TabDepthVector[T_VECFFE] = DepthVecFFE;

    /* install the default evaluation functions                            */
    EvTab[T_VECFFE] = EvList;
    PrTab[T_VECFFE] = PrVecFFE;

    /* install the comparision functions                                   */
    TabEq[T_VECFFE][T_VECFFE] = EqList;
    TabLt[T_VECFFE][T_VECFFE] = LtList;

    /* install the binary operations                                       */
    TabSum [T_INT   ][T_VECFFE] = SumSclList;
    TabSum [T_FFE   ][T_VECFFE] = SumFFEVecFFE;
    TabSum [T_VECFFE][T_INT   ] = SumListScl;
    TabSum [T_VECFFE][T_FFE   ] = SumVecFFEFFE;
    TabSum [T_INT   ][T_MATFFE] = SumSclList;
    TabSum [T_FFE   ][T_MATFFE] = SumSclList;
    TabSum [T_MATFFE][T_INT   ] = SumListScl;
    TabSum [T_MATFFE][T_FFE   ] = SumListScl;
    TabSum [T_VECTOR][T_VECFFE] = SumListList;
    TabSum [T_VECFFE][T_VECTOR] = SumListList;
    TabSum [T_VECFFE][T_VECFFE] = SumVecFFEVecFFE;
    TabSum [T_MATRIX][T_MATFFE] = SumListList;
    TabSum [T_MATFFE][T_MATRIX] = SumListList;
    TabSum [T_MATFFE][T_MATFFE] = SumListList;
    TabSum [T_VECTOR][T_FFE   ] = SumVectorFFE;
    TabSum [T_FFE   ][T_VECTOR] = SumFFEVector;

    TabDiff[T_INT   ][T_VECFFE] = DiffSclList;
    TabDiff[T_FFE   ][T_VECFFE] = DiffFFEVecFFE;
    TabDiff[T_VECFFE][T_INT   ] = DiffListScl;
    TabDiff[T_VECFFE][T_FFE   ] = DiffVecFFEFFE;
    TabDiff[T_INT   ][T_MATFFE] = DiffSclList;
    TabDiff[T_FFE   ][T_MATFFE] = DiffSclList;
    TabDiff[T_MATFFE][T_INT   ] = DiffListScl;
    TabDiff[T_MATFFE][T_FFE   ] = DiffListScl;
    TabDiff[T_VECTOR][T_VECFFE] = DiffListList;
    TabDiff[T_VECFFE][T_VECTOR] = DiffListList;
    TabDiff[T_VECFFE][T_VECFFE] = DiffVecFFEVecFFE;
    TabDiff[T_MATRIX][T_MATFFE] = DiffListList;
    TabDiff[T_MATFFE][T_MATRIX] = DiffListList;
    TabDiff[T_MATFFE][T_MATFFE] = DiffListList;
    TabDiff[T_VECTOR][T_FFE   ] = DiffVectorFFE;
    TabDiff[T_FFE   ][T_VECTOR] = DiffFFEVector;

    TabProd[T_INT   ][T_VECFFE] = ProdSclList;
    TabProd[T_FFE   ][T_VECFFE] = ProdFFEVecFFE;
    TabProd[T_VECFFE][T_INT   ] = ProdListScl;
    TabProd[T_VECFFE][T_FFE   ] = ProdVecFFEFFE;
    TabProd[T_INT   ][T_MATFFE] = ProdSclList;
    TabProd[T_FFE   ][T_MATFFE] = ProdSclList;
    TabProd[T_MATFFE][T_INT   ] = ProdListScl;
    TabProd[T_MATFFE][T_FFE   ] = ProdListScl;
    TabProd[T_VECTOR][T_VECFFE] = ProdListList;
    TabProd[T_VECFFE][T_VECTOR] = ProdListList;
    TabProd[T_VECFFE][T_VECFFE] = ProdVecFFEVecFFE;
    TabProd[T_VECTOR][T_MATFFE] = ProdListList;
    TabProd[T_VECFFE][T_MATRIX] = ProdListList;
    TabProd[T_VECFFE][T_MATFFE] = ProdVecFFEMatFFE;
    TabProd[T_MATRIX][T_VECFFE] = ProdListScl;
    TabProd[T_MATFFE][T_VECTOR] = ProdListScl;
    TabProd[T_MATFFE][T_VECFFE] = ProdListScl;
    TabProd[T_MATRIX][T_MATFFE] = ProdListScl;
    TabProd[T_MATFFE][T_MATRIX] = ProdListScl;
    TabProd[T_MATFFE][T_MATFFE] = ProdListScl;
    TabProd[T_VECFFE][T_LISTX ] = ProdListList;
    TabProd[T_MATFFE][T_LISTX ] = ProdSclList;
    TabProd[T_LISTX ][T_MATFFE] = ProdListScl;
    TabProd[T_VECTOR][T_FFE   ] = ProdVectorFFE;
    TabProd[T_FFE   ][T_VECTOR] = ProdFFEVector;

    TabQuo [T_VECFFE][T_INT   ] = QuoLists;
    TabQuo [T_VECFFE][T_FFE   ] = QuoLists;
    TabQuo [T_INT   ][T_MATFFE] = QuoLists;
    TabQuo [T_FFE   ][T_MATFFE] = QuoLists;
    TabQuo [T_MATFFE][T_INT   ] = QuoLists;
    TabQuo [T_MATFFE][T_FFE   ] = QuoLists;
    TabQuo [T_VECTOR][T_MATFFE] = QuoLists;
    TabQuo [T_VECFFE][T_MATRIX] = QuoLists;
    TabQuo [T_VECFFE][T_MATFFE] = QuoLists;
    TabQuo [T_MATRIX][T_MATFFE] = QuoLists;
    TabQuo [T_MATFFE][T_MATRIX] = QuoLists;
    TabQuo [T_MATFFE][T_MATFFE] = QuoLists;
    TabQuo [T_LISTX ][T_MATFFE] = QuoLists;

    TabMod [T_INT   ][T_VECFFE] = ModLists;
    TabMod [T_FFE   ][T_VECFFE] = ModLists;
    TabMod [T_INT   ][T_MATFFE] = ModLists;
    TabMod [T_FFE   ][T_MATFFE] = ModLists;
    TabMod [T_MATFFE][T_INT   ] = ModLists;
    TabMod [T_MATFFE][T_FFE   ] = ModLists;
    TabMod [T_MATFFE][T_VECTOR] = ModLists;
    TabMod [T_MATRIX][T_VECFFE] = ModLists;
    TabMod [T_MATFFE][T_VECFFE] = ModLists;
    TabMod [T_MATRIX][T_MATFFE] = ModLists;
    TabMod [T_MATFFE][T_MATRIX] = ModLists;
    TabMod [T_MATFFE][T_MATFFE] = ModLists;
    TabMod [T_MATFFE][T_LISTX ] = ModLists;

    TabPow [T_MATFFE][T_INT   ] = PowMatFFEInt;
    TabPow [T_MATFFE][T_INTPOS] = PowMatFFEInt;
    TabPow [T_MATFFE][T_INTNEG] = PowMatFFEInt;
    TabPow [T_VECTOR][T_MATFFE] = ProdListList;
    TabPow [T_VECFFE][T_MATRIX] = ProdListList;
    TabPow [T_VECFFE][T_MATFFE] = ProdVecFFEMatFFE;
    TabPow [T_MATFFE][T_MATFFE] = PowLists;

    TabComm[T_MATRIX][T_MATFFE] = CommLists;
    TabComm[T_MATFFE][T_MATRIX] = CommLists;
    TabComm[T_MATFFE][T_MATFFE] = CommLists;

    /* make the bags                                                       */
    HdVecFFEL = NewBag( T_FFE, SIZE_HD + sizeof(TypFFE) );
    HdVecFFER = NewBag( T_FFE, SIZE_HD + sizeof(TypFFE) );

    /* install the internal functions                                      */
    InstIntFunc( "CharFFE",      FunCharFFE      );
    InstIntFunc( "DegreeFFE",    FunDegreeFFE    );
    InstIntFunc( "LogVecFFE",    FunLogVecFFE    );
    InstIntFunc( "IntVecFFE",    FunIntVecFFE    );
    InstIntFunc( "MakeVecFFE",   FunMakeVecFFE   );
    InstIntFunc( "NumberVecFFE", FunNumberVecFFE );
}
