/****************************************************************************
**
*T  Obj . . . . . . . . . . . . . . . . . . . . . . . . . . . type of objects
**
**  'Obj' is the type of objects.
**
**  The following is defined in "system.h"
**
#define Obj             Bag
*/


/****************************************************************************
**
*F  IS_INTOBJ( <o> )  . . . . . . . .  test if an object is an integer object
**
**  'IS_INTOBJ' returns 1 if the object <o> is an (immediate) integer object,
**  and 0 otherwise.
*/
#define IS_INTOBJ(o) \
    ((Int)(o) & 0x01)



/****************************************************************************
**
*F  ARE_INTOBJS( <o1>, <o2> ) . . . . test if two objects are integer objects
**
**  'ARE_INTOBJS' returns 1 if the objects <o1> and <o2> are both (immediate)
**  integer objects.
*/
#define ARE_INTOBJS(o1,o2) \
    ((Int)(o1) & (Int)(o2) & 0x01)


/****************************************************************************
**
*F  INTOBJ_INT( <i> ) . . . . . . .  convert a C integer to an integer object
**
**  'INTOBJ_INT' converts the C integer <i> to an (immediate) integer object.
*/
#define INTOBJ_INT(i) \
    ((Obj)(((Int)(i) << 2) + 0x01))


/****************************************************************************
**
*F  INT_INTOBJ( <o> ) . . . . . . .  convert an integer object to a C integer
**
**  'INT_INTOBJ' converts the (immediate) integer object <o> to a C integer.
*/
#define INT_INTOBJ(o) \
    ((Int)(o) >> 2)



/****************************************************************************
**
*F  EQ_INTOBJS( <o>, <l>, <r> ) . . . . . . . . . compare two integer objects
**
**  'EQ_INTOBJS' returns 'True' if the  (immediate)  integer  object  <l>  is
**  equal to the (immediate) integer object <r> and  'False'  otherwise.  The
**  result is also stored in <o>.
*/
#define EQ_INTOBJS(o,l,r) \
    ((o) = (((Int)(l)) == ((Int)(r)) ? True : False))


/****************************************************************************
**
*F  LT_INTOBJS( <o>, <l>, <r> ) . . . . . . . . . compare two integer objects
**
**  'LT_INTOBJS' returns 'True' if the  (immediate)  integer  object  <l>  is
**  less than the (immediate) integer object <r> and  'False' otherwise.  The
**  result is also stored in <o>.
*/
#define LT_INTOBJS(o,l,r) \
    ((o) = (((Int)(l)) <  ((Int)(r)) ? True : False))


/****************************************************************************
**
*F  SUM_INTOBJS( <o>, <l>, <r> )  . . . . . . . .  sum of two integer objects
**
**  'SUM_INTOBJS' returns  1  if  the  sum  of  the  (imm.)  integer  objects
**  <l> and <r> can be stored as (immediate) integer object  and 0 otherwise.
**  The sum itself is stored in <o>.
*/
#define SUM_INTOBJS(o,l,r)             \
    ((o) = (Obj)((Int)(l)+(Int)(r)-1), \
    (((Int)(o) << 1) >> 1) == (Int)(o) )


/****************************************************************************
**
*F  DIFF_INTOBJS( <o>, <l>, <r> ) . . . . . difference of two integer objects
**
**  'DIFF_INTOBJS' returns 1 if the difference of the (imm.) integer  objects
**  <l> and <r> can be stored as (immediate) integer object  and 0 otherwise.
**  The difference itself is stored in <o>.
*/
#define DIFF_INTOBJS(o,l,r)            \
    ((o) = (Bag)((Int)(l)-(Int)(r)+1), \
    (((Int)(o) << 1) >> 1) == (Int)(o) )


/****************************************************************************
**
*F  TNUM_OBJ( <obj> ) . . . . . . . . . . . . . . . . . . . type of an object
**
**  'TNUM_OBJ' returns the type of the object <obj>.
*/
#define TNUM_OBJ(obj)   (IS_INTOBJ( obj ) ? T_INT :  GET_TYPE_BAG( obj ))


/****************************************************************************
**
*F  TNAM_OBJ( <obj> ) . . . . . . . . . . . . . name of the type of an object
*/
#define TNAM_OBJ(obj)   (InfoBags[TNUM_OBJ(obj)].name)


/****************************************************************************
**
*F  SIZE_OBJ( <obj> ) . . . . . . . . . . . . . . . . . . . size of an object
**
**  'SIZE_OBJ' returns the size of the object <obj>.
*/
#define SIZE_OBJ        GET_SIZE_BAG


/****************************************************************************
**
**  ADDR_OBJ( <obj> ) . . . . . . . . . . . . . absolute address of an object
**
**  'ADDR_OBJ' returns the absolute address of the memory block of the object
**  <obj>.
**  DO NOT USE, simply use PTR_BAG() instead
*/
//   #define ADDR_OBJ(bag)        PTR_BAG(bag)

