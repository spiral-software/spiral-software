
#include        "system.h"              /* system dependent part           */
#include        "memmgr.h"              /* garbage collector               */
#include        "objects.h"             /* objects                         */
#include		"string4.h"


/****************************************************************************
**

*F * * * * * * * * * * * * * * * string functions * * * * * * * * * * * * * *
*/

/****************************************************************************
**
*F  NEW_STRING( <len> )  . . . returns new string with length <len>, first
**  character and "first behind last" set to zero
**
*/
Obj NEW_STRING ( Int len )
{
  Obj res;
  res = NewBag( T_STRING, SIZEBAG_STRINGLEN(len)  ); 
  SET_LEN_STRING(res, len);
  /* it may be sometimes useful to have trailing zero characters */
  CHARS_STRING(res)[0] = '\0';
  CHARS_STRING(res)[len] = '\0';
  return res;
}

/****************************************************************************
**
*F  GrowString(<list>,<len>) . . . . . .  make sure a string is large enough
**
**  returns the new length, but doesn't set SET_LEN_STRING.
*/
Int             GrowString (
    Obj                 list,
    UInt                need )
{
    UInt                len;            /* new physical length             */
    UInt                good;           /* good new physical length        */

    /* find out how large the data area  should become                     */
    good = 5 * (GET_LEN_STRING(list)+3) / 4 + 1;

    /* but maybe we need more                                              */
    if ( need < good ) { len = good; }
    else               { len = need; }

    /* resize the bag                                                      */
    Resize( list, SIZEBAG_STRINGLEN(len) );

    /* return the new maximal length                                       */
    return (Int) len;
}
