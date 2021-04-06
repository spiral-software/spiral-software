/****************************************************************************
**
*A  record.h                    GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file declares the functions for computing with records.
**
*/

/****************************************************************************
**
*F  SetRecname( <hdRec>, <hdField>, <hdVal> ) . . . . . . . . set field value
**
** Sets the field value in a record, if needed a field is added to a record,
** by allocating additional storage 
*/
Obj    SetRecname ( Obj hdRec, Obj hdField, Obj hdVal );


/****************************************************************************
**
*F  FindRecnameRec( <hdRec>, <hdField>, <phdBag> ). find a field within a record
*F  FindRecnameRecWithDepth( <hdRec>, <hdField>, <phdBag>, <depth> )
**
** Find a field in a record, look in __bases__ list if not found,  returns 0 
** if not found neither in record nor in __bases__ 
** In phdBag returns real bag in which field is found
** FindRecnameRecWithDepth does the same as FindRecnameRec but will return record
** only after "depth" matches skipped. In other words FindRecnameRec is equal to
** FindRecnameRecWithDepth( <hdRec>, <hdField>, <phdBag>, 0 ).
*/
Obj*   FindRecnameRec ( Obj hdRec, Obj hdField, Obj* phdBag );
Obj*   FindRecnameRecWithDepth ( Obj hdRec, Obj hdField, Obj* phdBag, Int* depth );


/****************************************************************************
**
*F  RecnameObj( <hdRec>, <hdField> ) . . . convert integer/string to T_RECNAM
**
** Converts string or an integer <hdNam> to T_RECNAM, if <hdNam> is  T_RECNAM
** it is left alone. In case of other type, an error is signalled  
*/
Obj  RecnameObj ( Obj hdNam );


/****************************************************************************
**
*F  EvRec( <hdRec> )  . . . . . . . . . . . . . . . . . . . evaluate a record
**
**  'EvRec' evaluates  the record <hdRec>.   Since records are constants  and
**  thus selfevaluating this simply returns <hdRec>.
*/
extern  Bag       EvRec ( Bag hdRec );


/****************************************************************************
**
*V  HdTilde . . . . . . . . . . . . . . . . . . . . . . . . . .  variable '~'
**
**  'HdTilde' is the handle of the variable bag of  the  variable  '~'.  This
**  variable can be used inside record and list literals to refer to the list
**  or record that is currently being created.  So for example '[ ~ ]' is the
**  list that contains itself as only element.
*/
extern  Bag       HdTilde;


/****************************************************************************
**
*F  EvMakeRec(<hdLiteral>)  . . .  evaluate record literal to record constant
**
**  'EvMakeRec' evaluates the record literal, i.e., not yet evaluated, record
**  <hdLiteral> to a constant record.
**
**  'EvMakeRec' just calls 'MakeRec' telling it that the result goes into the
**  variable '~'.  Thus expressions in the variable  record can refer to this
**  variable and its subobjects to create objects that are not trees.
*/
extern  Bag       EvMakeRec ( Bag hdLiteral );


/****************************************************************************
**
*F  MakeRec(<hdDst>,<ind>,<hdLiteral>)  . evaluate record literal to constant
**
**  'MakeRec' evaluates the record literal <hdLiteral>  to a constant one and
**  puts  the result  into the bag  <hdDst>  at position <ind>.    <hdDst> is
**  either the variable '~', a list, or a record.
**
**  Because of literals like 'rec( a := rec( b := 1, c  := ~.a.b ) )' we must
**  enter the handle  of the result  into the superobject  before we begin to
**  evaluate the record literal.
**
**  A record  literal  is very much  like a  record, except that  at the even
**  places   the do not    yet contain  the  values  of  the  components, but
**  expressions, which yield the elements.   Evaluating a record literal thus
**  means looping over  the components, copying  the names at the odd entries
**  and evaluating the values at the even entries.
*/
extern  Bag       MakeRec ( Bag           hdDst,
                                    UInt       ind,
                                    Bag           hdLiteral );


/****************************************************************************
**
*F  EvRecElm( <hdElm> ) . . . . . . . . . . . . . . . .  get a record element
**
**  'EvRecElm' returns the value of 'PTR_BAG(<hdElm>)[0] . PTR_BAG(<hdElm>)[1]'.
**
**  '<record> . <name>'
**
**  This evaluates to the value of the record element with the name <name> in
**  the record <record>.  It is an  error  if  the  record  <record>  has  no
**  element with the name <name>.
**
**  'EvRecElm' simply iterates over  the record looking  for the record name.
**  If it is found the corresponding  value  is returned.  Note  that  it  is
**  not  neccessary  to compare  the strings of  the record names since every
**  record name is  store in a  unique record name  bag, i.e.,  no two record
**  name bags have the same string.  Thus we can simply compare handles.
*/
extern  Bag       EvRecElm ( Bag hdElm );


/****************************************************************************
**
*F  EvRecAss( <hdAss> ) . . . . . . . . .  assign a value to a record element
**
**  'EvRecAss' assigns the value  'EVAL( <hdAss>[1] )'  to the  element  with
**  the name '<hdAss>[0][1]' in the record '<hdAss>[0][0]'.
**
**  '<record>.<name> := <expr>;'
**
**  This assigns the value  of the expression  <expr> to the element with the
**  name <name> in  the record <record>.  Further  references to this element
**  will  return this new  value, until another assignment is  performed.  If
**  the record has  no element  with the  name   <name> it is   automatically
**  extended.
*/
extern  Bag       EvRecAss ( Bag hdAss );


/****************************************************************************
**
*F  SumRec( <hdL>, <hdR> )  . . . . . . . . . . . . . . . . sum of two record
*V  HdCallSum . . . . . . . . . . . . . handle of the 'sum' function call bag
**
**  'SumRec' returns the sum of the two operands <hdL> and <hdR>, of which at
**  least one must be a record.
**
**  '<left> + <right>'
**
**  The sum of two records or an object and a record is defined as follows:
**
**  If the right operand <right> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '+', which is a function,
**  or if the left operand <left> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '+' which is a function,
**  then this function is called with <left> and <right> as arguments
**    and '<left> + <right>' is the value returned by this function.
**
**  In all other cases an error is raised.
*/
extern  Bag       SumRec ( Bag hdL, Bag hdR );

extern  Bag       HdCallSum;


/****************************************************************************
**
*F  DiffRec( <hdL>, <hdR> ) . . . . . . . . . . . .  difference of two record
*V  HdCallDiff  . . . . . . . . . . .  handle of the 'diff' function call bag
**
**  'DiffRec' returns the difference the of two operands  <hdL> and <hdR>, of
**  which at least one must be a record.
**
**  '<left> - <right>'
**
**  The difference of two records or an object and a  record  is  defined  as
**  follows:
**
**  If the right operand <right> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '-', which is a function,
**  or if the left operand <left> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '-' which is a function,
**  then this function is called with <left> and <right> as arguments
**    and '<left> - <right>' is the value returned by this function.
**
**  In all other cases an error is raised.
*/
extern  Bag       DiffRec ( Bag hdL, Bag hdR );

extern  Bag       HdCallDiff;


/****************************************************************************
**
*F  ProdRec( <hdL>, <hdR> ) . . . . . . . . . . . . . . product of two record
*V  HdCallProd  . . . . . . . . . . .  handle of the 'prod' function call bag
**
**  'ProdRec' returns the product of the two operands  <hdL>  and  <hdR>,  of
**  which at least one must be a record.
**
**  '<left> * <right>'
**
**  The product of two records or an  object  and  a  record  is  defined  as
**  follows:
**
**  If the right operand <right> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '*', which is a function,
**  or if the left operand <left> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '*' which is a function,
**  then this function is called with <left> and <right> as arguments
**    and '<left> * <right>' is the value returned by this function.
**
**  In all other cases an error is raised.
*/
extern  Bag       ProdRec ( Bag hdL, Bag hdR );

extern  Bag       HdCallProd;


/****************************************************************************
**
*F  QuoRec( <hdL>, <hdR> )  . . . . . . . . . . . . .  quotient of two record
*V  HdCallQuo . . . . . . . . . . . . . handle of the 'quo' function call bag
**
**  'QuoRec' returns the quotient of the two operands  <hdL>  and  <hdR>,  of
**  which at least one must be a record.
**
**  '<left> / <right>'
**
**  The quotient of two records or an object  and  a  record  is  defined  as
**  follows:
**
**  If the right operand <right> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '/', which is a function,
**  or if the left operand <left> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '/' which is a function,
**  then this function is called with <left> and <right> as arguments
**    and '<left> / <right>' is the value returned by this function.
**
**  In all other cases an error is raised.
*/
extern  Bag       QuoRec ( Bag hdL, Bag hdR );

extern  Bag       HdCallQuo;


/****************************************************************************
**
*F  ModRec( <hdL>, <hdR> )  . . . . . . . . . . . . . remainder of two record
*V  HdCallMod . . . . . . . . . . . . . handle of the 'mod' function call bag
**
**  'ModRec' returns the remainder of the two operands <hdL>  and  <hdR>,  of
**  which at least one must be a record.
**
**  '<left> mod <right>'
**
**  The remainder of two records or an object and  a  record  is  defined  as
**  follows:
**
**  If the right operand <right> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element 'mod', which is a function,
**  or if the left operand <left> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element 'mod' which is a function,
**  then this function is called with <left> and <right> as arguments
**    and '<left> mod <right>' is the value returned by this function.
**
**  In all other cases an error is raised.
*/
extern  Bag       ModRec ( Bag hdL, Bag hdR );

extern  Bag       HdCallMod;


/****************************************************************************
**
*F  PowRec( <hdL>, <hdR> )  . . . . . . . . . . . . . . . power of two record
*V  HdCallPow . . . . . . . . . . . . . handle of the 'pow' function call bag
**
**  'PowRec' returns the power of the two operands <hdL> and <hdR>, of  which
**  at least one must be a record.
**
**  '<left> ^ <right>'
**
**  The power of two records or an object and a record is defined as follows:
**
**  If the right operand <right> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '^', which is a function,
**  or if the left operand <left> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '^' which is a function,
**  then this function is called with <left> and <right> as arguments
**    and '<left> ^ <right>' is the value returned by this function.
**
**  In all other cases an error is raised.
*/
extern  Bag       PowRec ( Bag hdL, Bag hdR );

extern  Bag       HdCallPow;


/****************************************************************************
**
*F  CommRec( <hdL>, <hdR> ) . . . . . . . . . . . .  commutator of two record
*V  HdCallComm  . . . . . . . . . . .  handle of the 'comm' function call bag
**
**  'CommRec' returns the commutator of the two operands <hdL> and  <hdR>, of
**  which at least one must be a record.
**
**  'Comm( <left>, <right> )'
**
**  The  commutator  of  two  records or an object and a record is defined as
**  follows:
**
**  If the right operand <right> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element 'comm', which is a function,
**  or if the left operand <left> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element 'comm' which is a function,
**  then this function is called with <left> and <right> as arguments
**    and '<left> + <right>' is the value returned by this function.
**
**  In all other cases an error is raised.
*/
extern  Bag       CommRec ( Bag hdL, Bag hdR );

extern  Bag       HdCallComm;


/****************************************************************************
**
*F  EqRec( <hdL>, <hdR> ) . . . . . . . . . . .  test if two record are equal
*V  HdCallEq  . . . . . . . . . . . . .  handle of the 'eq' function call bag
**
**  'EqRec' returns 'HdTrue' two operands <hdL> and <hdR>, of which at  least
**  one must be a record, are equal and 'HdFalse' otherwise.
**
**  '<left> = <right>'
**
**  The equal of two records or an object and a record is defined as follows:
**
**  If the right operand <right> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '=', which is a function,
**  or if the left operand <left> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '=' which is a function,
**  then this function is called with <left> and <right> as arguments
**    and '<left> = <right>' is the value returned by this function.
**
**  In all other cases the operands are considered equal if both are  records
**  and they have the same names and all corresponding elements equal.
*/
extern  Bag       EqRec ( Bag hdL, Bag hdR );

extern  Bag       HdCallEq;


/****************************************************************************
**
*F  LtRec( <hdL>, <hdR> ) . . . . . . test if one record is less than another
*V  HdCallLt  . . . . . . . . . . . . .  handle of the 'lt' function call bag
**
**  'LtRec' returns 'HdTrue' the operand <hdL> is less than the operand <hdR>
**  and 'HdFalse' otherwise .  At least one of the operands must be a record.
**
**  '<left> < <right>'
**
**  The lt of two records or an object and a record is defined as follows:
**
**  If the right operand <right> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '<', which is a function,
**  or if the left operand <left> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '<' which is a function,
**  then this function is called with <left> and <right> as arguments
**    and '<left> < <right>' is the value returned by this function.
**
**  In all other cases the operands are considered ltual if both are  records
**  and they have the same names and all corresponding elements ltual...
*/
extern  Bag       LtRec ( Bag hdL, Bag hdR );

extern  Bag       HdCallLt;


/****************************************************************************
**
*F  InRec( <hdL>, <hdR> ) . . . . . . . . . .  test if a record is in another
*V  HdCallIn  . . . . . . . . . . . . .  handle of the 'in' function call bag
**
**  'InRec' returns 'HdTrue' the operand <hdL> is in the  operand  <hdR>  and
**  'HdFalse' otherwise .  At least the right operand must be a record.
**
**  '<left> in <right>'
**
**  The 'in' of two records or an object and a record is defined as follows:
**
**  If the right operand <right> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element 'in', which is a function,
**  then this function is called with <left> and <right> as arguments
**    and '<left> in <right>' is the value returned by this function.
**
**  In all other cases an error is raised.
*/
extern  Bag       InRec ( Bag hdL, Bag hdR );

extern  Bag       HdCallIn;


/****************************************************************************
**
*F  PrRec( <hdRec> )  . . . . . . . . . . . . . . . . . . . .  print a record
*V  HdCallPrint . . . . . . . . . . . handle of the 'print' function call bag
**
**  'PrRec' prints the record with the handle <hdRec>.
**
**  If <hdRec> has an element 'operations' which is a record
**    and this record as an element 'print' which is a function
**  then this function is called with <hdRec> as argument and should print it
**
**  In all other cases the record is printed in the following form:
**
**  'rec( <name> := <expr>,... )'
**
**  'PrRec' is also called to print variable records, i.e., records that have
**  not yet been evaluated.  They are always printed in the second form.
*/
extern  void            PrRec ( Bag hdRec );

extern  Bag       HdCallPrint;


/****************************************************************************
**
*F  PrRecElm( <hdElm> ) . . . . . . . . . . . . . . .  print a record element
**
**  'PrRecElm' prints the record element in the following form:
**
**  '<record> . <name>'
*/
extern  void            PrRecElm ( Bag hdElm );


/****************************************************************************
**
*F  PrRecAss( <hdAss> ) . . . . . . . . . . . . . . print a record assignment
**
**  'PrRecAss' prints the record assignment in the form:
**
**  '<record>.<name> := <expr>;'
*/
extern  void            PrRecAss ( Bag hdAss );


/****************************************************************************
**
*F  PrRecName( <hdName> ) . . . . . . . . . . . . . print a record field name
**
*/
extern  void            PrRecName ( Bag hdNam );


/****************************************************************************
**
*F  InitRec() . . . . . . . . . . . . . . . .  intitialize the record package
**
**  'InitRec' initializes the record package.
*/
extern  void            InitRec ( void );


/****************************************************************************
**
*E  Emacs . . . . . . . . . . . . . . . . . . . . . . . local emacs variables
**
**  Local Variables:
**  outline-regexp:     "*F\\|*V\\|*T\\|*E"
**  fill-column:        73
**  fill-prefix:        "**  "
**  End:
*/



