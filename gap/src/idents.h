/****************************************************************************
**
*A  idents.h                    GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file contains the functions for mapping  identifiers  to  variables.
**
*/

enum {
    OFS_RECNAM = 0,
    OFS_IDENT = 2
};

#define VAR_NAME(hdVar)                 ((char*) (PTR_BAG(hdVar)+OFS_IDENT))

#define VAR_VALUE(hdVar)                ((Bag)(PTR_BAG(hdVar)[0]))
#define SET_VAR_VALUE(hdVar, hdVal)     SET_BAG(hdVar, 0, hdVal)

#define VAR_PROPS(hdVar)                (PTR_BAG(hdVar)[1])
#define SET_VAR_PROPS(hdVar, hdVal)     SET_BAG(hdVar, 1, hdVal)
#define RECNAM_NAME(hdVar)              ((char*) (PTR_BAG(hdVar)+OFS_RECNAM))

extern Obj HdPkgRecname;

/****************************************************************************
**
*F  PushFunction( <hdFun> ) . . . . . . . . add another function to the stack
**
**  'PushFunction' adds another function to the  function  definition  stack.
**  It is called from the reader when the reader starts to parse a  function.
**  It makes the local variables known to 'FindIdent'.
*/
extern  void            PushFunction ( Bag hdFun );


/****************************************************************************
**
*F  PopFunction() . . . . . . . . . .  remove another function from the stack
**
**  'PopFunction'  removes  the  most  recent  function  from  the   function
**  definition stack.  It is called from the reader when the reader  finished
**  parsing a function.  It makes the local variables again unknown.
*/
extern  void            PopFunction ( void );


/****************************************************************************
**
*V  HdStack . . . . . . . . .  handle of the function definition stack, local
*V  TopStack  . . . . . . . . . . top of the function definition stack, local
**
**  'HdStack' is the handle of the function definition stack.  This is a list
**  of all the functions that are currently being read organized as a  stack.
**  The entries of this list are the function  definition  bags.  'FindIdent'
**  first searches this stack to look for local  variables  before  searching
**  'HdIdenttab'.
**
**  'TopStack' is the index of the topmost entry in the  stack,  which  grows
**  upward.
*/
extern Bag             HdStack;
extern UInt   TopStack;

/****************************************************************************
**
*V  IsUndefinedGlobal . . . . .  true if a variable is an new global variable
**
**  'IsUndefinedGlobal' is set by 'FindIdent'  if  <name>  is  a  new  global
**  variable inside a function definition.  'RdVar' tests 'IsUndefinedGlobal'
**  and complains about such variables.
*/
extern  UInt   IsUndefinedGlobal;


/****************************************************************************
**
*F  MakeIdent( <name> ) . . . . . . . . . . . . . . . . . . make new variable
*F  MakeIdentSafe( <name bag>, <offset> ) . . . . . . . . . make new variable
**
**  'MakeIdent' creates the handle of  the  variable  bag  for  the  variable
**  with the identifier  <name>. It  does  not  look  in  any  tables, unlike
**  FindIdent. 'MakeIdent' will allocate space, so <name> must be guaranteed
**  to survive garbage collection (i.e. should not be part of any bag)
**
**  'MakeIdentSafe' allows to create variables from collectable strings.
**  A bag must be passed and an offset to the string within a bag.
**  String will be resolved with (char*)(PTR_BAG(hdNam)+ofs).
*/
extern  Bag       MakeIdent ( char name [] );
extern  Bag       MakeIdentSafe ( Obj hdNam, UInt ofs );


/****************************************************************************
**
*F  CopyVar( <hd> ) . . . . . . . create a copy of a variable, with same value
**
**  'CopyVar' creates variable identical to <hd> with same name and value,
**  value is not copied, pointer assignment is performed. This is useful,
**  because a copy can be put in a different namespace.
*/
extern  Bag       CopyVar ( Bag hd );


/****************************************************************************
**
*F GlobalIdent( <name> ) . . . . . . . . . . . . . . . find a global variable
**
*/
extern  Bag       GlobalIdent ( char name [] );

/****************************************************************************
**
*F  FindIdent( <name> ) . . . . . . . . . . . find variable for an identifier
**
**  'FindIdent' returns the handle of  the  variable  bag  for  the  variable
**  with the identifier  <name>.  'FindIdent'  first  searches  the  function
**  definition bags made added to the stack by 'PushFunction'.  If  no  local
**  variable has this identifier, 'FindIdent'  looks in the global identifier
**  table.  If the identifier is also not found in the  global  table  a  new
**  global variable is created.
*/
extern  Bag       FindIdent ( char name [] );
extern  Bag       FindIdentWr ( char name [] );
extern  Bag       IdentAssign ( char name [] , Bag val );


/****************************************************************************
**
*F  FindRecname( <name> ) . . . .  find the record name bag for a record name
**
**  'FindRecname' returns the record name bag for  the  record  name  <name>.
**  Note that record names are always stored unique, i.e., for  every  string
**  there is a unique record name bag for that string.  This makes it  easier
**  to find a record element for a given record  name:  We  do  not  have  to
**  compare strings, it is enough to compare handles.
*/
extern  Bag       FindRecname ( char name [] );


/****************************************************************************
**
*F  ExistsRecname( <name> ) . . . . . . check if record field name bag exists
**
**  'ExistsRecname' returns 1 if the record name bag for the rec field <name>
**  exists, 0 otherwise.
*/
extern int       ExistsRecname (char *name) ;


/****************************************************************************
**
*F  InitIdents()  . . . . . . . . . . . . . . . initialize identifier package
**
**  'InitIdents' initializes the identifier package. This must be done before
**  the  first  call  to  'FindIdent'  or  'FindRecname',  i.e.,  before  the
**  evaluator packages are initialized.
*/
extern  void            InitIdents ( void );



