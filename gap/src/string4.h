/****************************************************************************
**
*W  string.h                    GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file declares the functions which mainly deal with strings.
**
**  A *string* is a  list that  has no  holes, and  whose  elements  are  all
**  characters.  For the full definition of strings see chapter  "Strings" in
**  the {\GAP} manual.  Read also "More about Strings" about the  string flag
**  and the compact representation of strings.
**
**  Strings in compact representation  can be accessed and handled through
**  the  macros     'NEW_STRING',   `CHARS_STRING'  (and   'CSTR_STRING'),
**  'GET_LEN_STRING',   `SET_LEN_STRING', `GROW_STRING',  'GET_ELM_STRING'
**  and `SET_ELM_STRING'.
**  
**  This  package also contains the   list  function  for ranges, which   are
**  installed in the appropriate tables by 'InitString'.  */

#include <string.h>  /* for memcpy */

/****************************************************************************
**
*F * * * * * * * * * * * * * * * string functions * * * * * * * * * * * * * *
*/

/****************************************************************************
**
*F  SIZEBAG_STRINGLEN( <len> ) . . . . size of Bag for string of length <len>
**  
*/
#define SIZEBAG_STRINGLEN(len)         ((len) + 1)

/****************************************************************************
**
*F  CSTR_STRING( <list> ) . . . . . . . . . . . . . . .  C string of a string
*F  CHARS_STRING( <list> ) . . . . . . . . . . . . . .   same pointer 
**
**  'CSTR_STRING'  returns the (address  of the)  C  character string of  the
**  string <list>. Note that the string as C string is truncated before the
**  first null character. Try to avoid this and use CHARS_STRING.
**
**  Note that 'CSTR_STRING' is a macro, so do not call it with arguments that
**  have sideeffects.
*/
#define CSTR_STRING(list)             ((Char*)PTR_BAG(list))
#define CHARS_STRING(list)            ((Char*)PTR_BAG(list))

/****************************************************************************
**
*F  GET_LEN_STRING( <list> )  . . . . . . . . . . . . . .  length of a string
**
**  'GET_LEN_STRING' returns the length of the string <list>, as a C integer.
**
**  Note that  'GET_LEN_STRING' is a macro, so  do not call it with arguments
**  that have sideeffects.
*/
#define GET_LEN_STRING(list)          (strlen(CSTR_STRING(list)))

/****************************************************************************
**
*F  SET_LEN_STRING( <list>, <len> ) . . . . . . . . . set length of a string
**
**  'SET_LEN_STRING' sets length of the string <list> to C integer <len>.
**
**  Note that  'SET_LEN_STRING' is a macro, so  do not call it with arguments
**  that have sideeffects.
*/
#define SET_LEN_STRING(list,len)

/****************************************************************************
**
*F  NEW_STRING( <len> ) . . . . . . . . . . . . . . . . . . make a new string
**
**  'NEW_STRING' returns a new string with room for <len> characters. It also
**  sets its length to len. 
**
*/
extern Obj NEW_STRING(Int len);

/****************************************************************************
**
*F  GROW_STRING(<list>, <len>) . . . .  make sure a string is large enough
**
**  'GROW_STRING' grows  the string <list>  if necessary to ensure that it
**  has room for at least <len> elements.
**
**  Note that 'GROW_STRING' is a macro, so do not call it with arguments that
**  have sideeffects.
*/
#define GROW_STRING(list,len)   ( (SIZEBAG_STRINGLEN(len) < SIZE_OBJ(list)) ? \
                                  NUM_TO_UINT(0) : GrowString(list,len) )

extern  Int             GrowString (
            Obj                 list,
            UInt                need );


/****************************************************************************
**
*F  GET_ELM_STRING( <list>, <pos> ) . . . . . . select an element of a string
**
**  'GET_ELM_STRING'  returns the  <pos>-th  element  of  the string  <list>.
**  <pos> must be  a positive integer  less than  or  equal to  the length of
**  <list>.
**
**  Note that 'GET_ELM_STRING' is a  macro, so do not  call it with arguments
**  that have sideeffects.
*/
#define GET_ELM_STRING(list,pos)      (CHARS_STRING(list)[pos-1])

/****************************************************************************
**
*F  SET_ELM_STRING( <list>, <pos>, <val> ) . . . . set a character of a string
**
**  'SET_ELM_STRING'  sets the  <pos>-th  character  of  the string  <list>.
**  <val> must be a character and <list> stay a string after the assignment.
**
**  Note that 'SET_ELM_STRING' is a  macro, so do not  call it with arguments
**  that have sideeffects.
*/
#define SET_ELM_STRING(list,pos,val)  (CHARS_STRING(list)[pos-1] = (val))

/****************************************************************************
**
*F  COPY_CHARS( <str>, <charpnt>, <n> ) . . . copies <n> chars, starting
**  from character pointer <charpnt>, to beginning of string
**  
**  This is a   macro. It assumes  that the  data  area in  <str> is  large
**  enough. It does not add a terminating null character and not change the
**  length of the string.
**
*/
#define COPY_CHARS(str,pnt,n)         (memcpy(CHARS_STRING(str), pnt, n));

/****************************************************************************
**
*F  IS_STRING( <obj> )  . . . . . . . . . . . . test if an object is a string
**
**  'IS_STRING' returns 1  if the object <obj>  is a string  and 0 otherwise.
**  It does not change the representation of <obj>.
**
**  Note that 'IS_STRING' is a  macro, so do not call  it with arguments that
**  have sideeffects.
*/
#define IS_STRING(obj)                IsString(obj)

/****************************************************************************
**
*F  C_NEW_STRING( <obj>, <cstring> ) . . . . . . . . . . . create GAP string
*/
#define C_NEW_STRING(obj,cstr) \
  do { \
    obj = NEW_STRING( strlen(cstr) ); \
    memcpy( CHARS_STRING(obj), (cstr), strlen(cstr) ); \
  } while ( 0 );

/****************************************************************************
**
*F  SINT_CHAR(a)
**
**  'SINT_CHAR' converts the character a (a UInt1) into a signed (C) integer.
*/
#define SINT_CHAR(a)    (((UInt1)a)<128 ? (Int)a : (Int)a-256)

/****************************************************************************
**
*F  CHAR_SINT(n)
**
**  'CHAR_SINT' converts the signed (C) integer n into an (UInt1) character.
*/
#define CHAR_SINT(n)    (UInt1)(n>=0 ? n : n+256)

/*
*E  string.h  . . . . . . . . . . . . . . . . . . . . . . . . . . . ends here
*/
