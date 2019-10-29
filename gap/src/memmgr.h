#ifndef MEMMGR_H_INCLUDED
#define MEMMGR_H_INCLUDED

/****************************************************************************
**
*A  memmgr.h                    GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2019, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2019) by the GAP Group (www.gap-system.org).
**
**  This file declares the functions of  MEMMGR,  the  GAP  storage  manager.
**
**  Memmgr is the  GAP storage manager.  That  means that the other parts  of
**  GAP  request  memory  areas  from Gasman.   These  are  then  filled with
**  information.  Gasman cares about the allocation of those memory areas and
**  the collection of unused areas.  Thus these operations are transparent to
**  the rest of GAP,  enabling the programmer to concentrate on his algorithm
**  instead of caring about memory allocation and deallocation.
**
**  The basic thing is  the bag.  This is simply  a continous area of  memory
**  containing any information, including references to other bags.  If a bag
**  contains  references to other bags these  references are collected at the
**  beginning of the bag.  A bag together  with all bags he references direct
**  or indirect, is called an object.  New bags are created with 'NewBag'.
**
**  When you  create  a bag using  'NewBag' this  functions returns  a unique
**  pointer identifying this bag, this  pointer is called  a handle and is of
**  type 'Bag'.  You  specify this handle  as  first argument to  every
**  storage manager function as first argument.  Variables containing handles
**  should begin with 'hd' to make the code more readable.
**
**  Every bag belongs to a certain type, specifying what information this bag
**  contains and how  bags of this type  are evaluated.  You specify the type
**  when creating the bag, you can retrieve it using the macro 'GET_TYPE_BAG'.
**
**  Every bag has a size, which is the size of the memory area in bytes.  You
**  specify this size when you create the bag, you can  change it later using
**  'Resize' and retrieve it later using the macro 'GET_SIZE_BAG'.
*/

#include "system_types.h"
#include "system.h"
#include <assert.h>


#ifdef COUNT_BAGS
#define DEBUG_GLOBAL_BAGS		True	/* when COUNT_BAGS is turned on, turn  */
#define DEBUG_LOADING			True	/* on remaining switches for debugging */
// #define TREMBLE_HEAP			True	/* master pointers, bag counts, etc.   */
#define DEBUG_MASTERPOINTERS	True
#define DEBUG_FUNCTIONS_BAGS	True    /* provides functions to mirror macros */
#define DEBUG_BAG_MGMT			True	/* allow bag tracing by code(s) */
#else
#undef DEBUG_GLOBAL_BAGS
#undef DEBUG_LOADING
#undef TREMBLE_HEAP
#undef DEBUG_MASTERPOINTERS
#undef DEBUG_DEADSONS_BAGS
#undef DEBUG_BAG_MGMT
#define DEBUG_FUNCTIONS_BAGS	True	/* Currently always defined */
#define DEBUG_GLOBAL_BAGS		True	/* when COUNT_BAGS is turned on, turn  */
#endif


/****************************************************************************
**
**  For  every type of bags  there is a symbolic  name defined for this type.
**  The types *must* be sorted in the following order:
**
**  Type of the non-printing object:       'T_VOID',
**  Types of objects that lie in a field:  'T_INT'      to  'T_FFE',
**  Types of objects that lie in a group:  'T_PERM'     to  'T_AGWORD',
**  Types of other unstructured objects:   'T_BOOL'     to  'T_FUNCINT',
**  Types of lists:                        'T_LIST'     to  'T_RANGE',
**  Type of records:                       'T_REC',
**  Extended types of lists (see list.c):  'T_MATRIX'   to  'T_LISTX'
**  Types related to variables:            'T_VAR'      to  'T_RECASS',
**  Types of binary operators:             'T_SUM'      to  'T_COMM',
**  Types of logical operators:            'T_NOT'      to  'T_IN',
**  Types of statements:                   'T_FUNCCALL' to  'T_RETURN',
**  Types of almost constants:             'T_MAKEPERM' to  'T_MAKEREC',
**  Remaining types, order not important:  'T_CYCLE'    to  'T_FREEBAG',
**  First non assigned type:               'T_ILLEGAL'.
**
**  The types of  changable objects must  lie  between 'T_LIST' and  'T_VAR'.
**  They are the types of objects that 'Copy' must really copy.
**
**  The  types of all constants,  i.e., the types of all  objects that can be
**  the result of 'EVAL' must lie between 'T_INT' and 'T_VAR'.  They  are the
**  indices into the dispatch tables of the binary operators.
**
**  The possible types are defined in the declaration part of this package as
**  follows (immutable types represent immediate values):
**
**  NOTE: values are assigned in the definition below to aid decyphering memory 
**  dumps during debug
*/

typedef enum {
    T_VOID				= 0,			/* "void" */
	T_INT 				= 1,			/* "integer" */
	T_INTPOS 			= 2,			/* "integer (> 2^28)" */
	T_INTNEG 			= 3,			/* "integer (< -2^28)" */
	T_RAT 				= 4,			/* "rational" */
	T_CYC 				= 5,			/* "cyclotomic" */
	T_DOUBLE 			= 6,			/* "double" */
	T_CPLX 				= 7,			/* "complex" */
    T_UNKNOWN			= 8,			/* "unknown" */
	T_FFE 				= 9,			/* "finite field element" */

	T_PERM16 			= 10,			/* "permutation16" */
	T_PERM32 			= 11,			/* "permutation32" */
	T_WORD 				= 12,			/* "word" */
	T_SWORD 			= 13,			/* "sparse word" */
	T_AGWORD 			= 14,			/* "agword" */

    T_BOOL 				= 15,			/* "boolean" */
	T_CHAR 				= 16,			/* "character" */
	T_FUNCTION 			= 17,			/* "function" */
	T_METHOD 			= 18,			/* "method" */
	T_FUNCINT 			= 19,			/* "internal function" */

    T_MUTABLE			= 20,			/* mutable types -- "list" */
    T_LIST   			= T_MUTABLE,	/* "list" */
	T_SET 				= 21,			/* "set" */
	T_VECTOR 			= 22,			/* "vector" */
	T_VECFFE 			= 23,			/* "finite field vector" */
	T_BLIST 			= 24,			/* "boolean list" */
	T_STRING 			= 25,			/* "string" */
    T_RANGE 			= 26,			/* "range" */

	T_REC 				= 27,			/* "record" */
	T_MATRIX 			= 28,			/* "matrix (extended)" */
	T_MATFFE 			= 29,			/* "matffe (extended)" */
	T_LISTX 			= 30,			/* "list (extended)" */
	T_DELAY 			= 31,			/* "delayed expression" */

    T_VAR 				= 32,			/* "variable" */
	T_VARAUTO 			= 33,			/* "autoread variable" */
	T_VARASS 			= 34,			/* "var assignment" */
	T_VARMAP 			= 35,			/* "var map" */
	T_LISTELM 			= 36,			/* "list element" */
	T_LISTELML 			= 37,			/* "list element" */
    T_LISTELMS 			= 38,			/* "sublist" */
	T_LISTELMSL 		= 39,			/* "sublist" */
	T_LISTASS 			= 40,			/* "list assignment" */
	T_LISTASSL 			= 41,			/* "list assignment" */
	T_LISTASSS 			= 42,			/* "list assignment" */
    T_LISTASSSL 		= 43,			/* "list assignment" */
	T_RECELM 			= 44,			/* "record element" */
	T_RECASS 			= 45,			/* "record assignment" */
	T_MULTIASS 			= 46,			/* "multi assignment" */

    T_SUM 				= 47,			/* "+" */
	T_DIFF 				= 48,			/* "-" */
	T_PROD 				= 49,			/* "*" */
	T_QUO 				= 50,			/* "/" */
	T_MOD 				= 51,			/* "mod" */
	T_POW 				= 52,			/* "^" */
	T_COMM 				= 53,			/* "commutator" */

	T_NOT 				= 54,			/* "not" */
	T_AND 				= 55,			/* "and" */
    T_OR 				= 56,			/* "or" */
	T_EQ 				= 57,			/* "=" */
	T_NE 				= 58,			/* "<>" */
	T_LT 				= 59,			/* "<" */
	T_GE 				= 60,			/* ">=" */
	T_LE 				= 61,			/* "<=" */
	T_GT 				= 62,			/* ">" */
	T_IN 				= 63,			/* "in" */
	T_CONCAT 			= 64,			/* "::" */

    T_FUNCCALL 			= 65,			/* "function call" */
	T_FUNCINT_CALL		= 66,			/* "fast internal function call" */
	T_STATSEQ			= 67,			/* "statement sequence" */
	T_IF 				= 68,			/* "if statement" */
	T_FOR 				= 69,			/* "for loop" */
	T_WHILE 			= 70,			/* "while loop" */
	T_REPEAT 			= 71,			/* "repeat loop" */
	T_RETURN 			= 72,			/* "return statement" */

    T_MAKEPERM 			= 73,			/* "var permutation" */
	T_MAKEFUNC 			= 74,			/* "var function" */
	T_MAKEMETH 			= 75,			/* "var method" */
	T_MAKELIST 			= 76,			/* "var list" */
	T_MAKESTRING 		= 77,			/* "var string" */
    T_MAKERANGE 		= 78,			/* "var range" */
	T_MAKEREC 			= 79,			/* "var record" */

	T_MAKETAB 			= 80,			/* "var hash" */
	T_MAKELET 			= 81,			/* "let statement" */
    T_CYCLE 			= 82,			/* "cycle" */
	T_FF 				= 83,			/* "finite field" */
	T_AGEN 				= 84,			/* "abstract generator" */
	T_AGGRP 			= 85,			/* "aggroup" */
	T_PCPRES 			= 86,			/* "polycyclic presentation" */
	T_AGEXP 			= 87,			/* "ag exponent vector" */
	T_AGLIST 			= 88,			/* "ag exponent/generator" */
    T_RECNAM 			= 89,			/* "record name" */
	T_NAMESPACE 		= 90,			/* "namespace" */
	T_EXEC 				= 91,			/* "stack frame" */
    T_IS				= 92,			/* "_is" */
    T_FREEBAG 			= 93,			/* "free bag" */

	T_ILLEGAL			= 94,			/* "ILLEGAL bag */
	T_MAGIC_254			= 254, 			/* "magic" value for reserved for memory manager */
	T_RESIZE_FREE		= 255			/* flags free areas in a re-sized (smaller) bag */
} ObjType;


/****************************************************************************
**
*V  NameType . . . . . . . . . . . . . . . . . . . . printable name of a type
**
**  'NameType' is an array that contains for every possible type  a printable
**  name.  Those names can be used, for example, in error messages.
*/
extern  char            * NameType [];

/****************************************************************************
**
*V  SizeType . . . . . . . . . . size of handle and data area of a bag, local
**
**  'SizeType' is an array, that contains for information about how large the
**  handle area and the data area are for each type.
**
*/
typedef struct {
    Int        handles;
    Int        data;
    char        name[4];
} SizeTypeT;

extern SizeTypeT          SizeType [];

/****************************************************************************
**
*F  NrHandles( <type>, <size> ) . . . . . . . . .  number of handles of a bag
**
**  'NrHandles' returns the number of handles referenced from a bag with type
**  <type> and size <size>. This is used  in  the  garbage  collection  which
**  needs to know the number to be able  to  mark  all  subobjects of a given
**  object.
*/
extern  Int            NrHandles ( unsigned int      type,
                                    UInt     size );


// Flags for bags
// Old way: Flags were defined starting at 1 and were shifted left/right when
//			set/fetched in the bag header.  This was done to keep the least
//			significant byte available to store the Type  
// New way: Define flags with appropriate bit values such that the least
//			significant byte is unaffected... 
//          also see GET_FLAG_BAG, SET_FLAG_BAG, CLEAR_FLAG_BAG macros

enum BagFlags {
	BF_COPY				= 0x00000100L,
	BF_PRINT			= 0x00000200L,
	BF_UNUSED_FLAG		= 0x00000400L,	// just a free flag
	BF_DELAY			= 0x00000800L,
	BF_DELAY_EV			= 0x00001000L,
	BF_DELAY_EV0		= 0x00002000L,
	BF_NO_WARN_UNDEFINED= 0x00004000L,
	BF_NO_COPY			= 0x00008000L,
	BF_METHCALL			= 0x00010000L,
	BF_UNEVAL_ARGS		= 0x00020000L,
	BF_NS_GLOBAL		= 0x00040000L,
	BF_VAR_AUTOEVAL		= 0x00080000L,
	BF_VISITED			= 0x00100000L, /* -- for recursive traversals with cycle prevention */
	BF_PROTECT_VAR		= 0x00200000L,
	BF_ENVIRONMENT		= 0x00400000L, /* for T_FUNCTION and T_MAKEFUNC/T_MAKEMETH this*/
									   /* flag is set if function depends from environment;*/
	BF_ENV_VAR			= 0x00400000L, /* for T_VAR only, this flag means that some        */
									   /* subroutine has reference to this variable;       */
	BF_ON_CALLSTACK		= 0x00400000L, /* for T_EXEC only, flag means that T_EXEC is on the*/
									   /* call stack .                                     */
	BF_INHERITED_CALL	= 0x00400000L, /* for T_FUNCCALL means that this is inherited   */
									   /* method call, i.e. from Inherited() function.  */
	BF_ON_EVAL_STACK	= 0x00800000L, /* for all bags, FunTop() uses this flag to mark	0x00008000 */
									   /* bags on the eval stack to correctly highlight   */
									   /* current statement.                              */
	BF_WEAKREFS			= 0x01000000L, /* For all bags. References to another bags from bag marked   */
									   /* with this flag are not taken to account during garbage     */
									   /* collection. After garbage collection all dead references   */
									   /* set to zero. This flags is in use by DGC.		  */
	BF_WEAKREF_CHANGED	= 0x04000000L, /* For all bags. Internal DGC flag. To temporarily mark */
									   /* bags with BF_WEAKREFS flag that have references changed */
									   /* during garbage collection cycle */
    BF_ALIVE			= 0x80000000L, /* For all bags. DGC uses this flag internally to mark bags reachable from */
									   /* global variables. This is the max flag value.  */ 
									   /* 32, 64,..., 2^22 */
};

#define BF_ALL_FLAGS	0xFFFFFF00L		/* Allows to mask off all flags */
#define TYPE_BIT_MASK	0x000000FFL		/* Mask off least significant byte -- bag type */


/* This variable is an easy (and dirty) way to implement full copying
   of objects, needed to correctly implement spiral_delay_ev.c:Eval() */
/*V DoFullCopy */
extern int DoFullCopy;

/* goes in variables*/
#define IS_FULL_MUTABLE(hd) (   (T_MUTABLE <= GET_TYPE_BAG(hd) && GET_TYPE_BAG(hd) < T_VAR) \
                             || (T_VARAUTO < GET_TYPE_BAG(hd) && GET_TYPE_BAG(hd) < T_RECNAM))

#define IS_MUTABLE(hd)   ((!DoFullCopy && (T_MUTABLE <= GET_TYPE_BAG(hd) && GET_TYPE_BAG(hd) < T_VAR)) \
                          || (DoFullCopy && (IS_FULL_MUTABLE(hd))))
/*|| (GET_TYPE_BAG(hd)==T_VAR)  )*/

void RecursiveClearFlag(Obj hd, int flag);

void RecursiveClearFlagMutable(Obj hd, int flag);

void RecursiveClearFlagFullMutable(Obj hd, int flag);

/* Takes variable number(=howMany) of flags to clear, all flags should be int's */
void RecursiveClearFlagsFullMutable(Obj hd, int howMany, ...);



/****************************************************************************
**
*V  SIZE_HD . . . . . . . . . . . . . . . . . . . . . . . .  size of a handle
**
**  'SIZE_HD' is just short for 'sizeof(BagPtr_t)' which is used very often.
**
**  'SIZE_HD' is defined in the declaration part of this package as follows:
*/
#define SIZE_HD         ((Int)sizeof(BagPtr_t))

#ifndef IN_DGC
#define SET_BAG(bag, index, value)      PTR_BAG(bag)[index] = (value)
#endif

#define NTYPES 256

/******************************************************************************
 ** 
 ** Bag layout is shown below.  Bags can be of any arbitrary size.  
 ** NOTE: The diagram below assumes 64 bit pointers (and unsigned long ints).
 **
 **    |  0|  1|  2|  3|  4|  5|  6|  7|
 **    +-------------------------------+
 **    |    Flags (BagFlags)       |Typ|  Typ == Bag [Object] type (ObjType)
 **    +-------------------------------+
 **    |            Bag Size           |
 **    +-------------------------------+
 **    |   Copy Pointer                |
 **    +-------------------------------+
 **    | Link Pointer (to master ptr)  |
 **    +-------------------------------+
 **    | Start of bag data             |
 **   ...                             ...
 **    |                               |
 **    +-------------------------------+
 **
 ** Bag data is the number of bytes (size) plus padding to round up to a
 ** multiple of sizeof(BagPtr_t).
 **
 ** The master pointer for any bag (returned from NewBag4) is treated as a
 ** handle, the master pointer points to the start of the bag data.
 */

//
// Define a struct to give the layout of the bags...
// word 0:		Flags and bag type
// word 1:		Size of bag; size of data only, header size is NOT included
// word 2:		copy pointer
// word 3:		link pointer
// word 4:		Start of data (bag handles actually point here)
//

typedef Bag BagPtr_t;

typedef struct {
	UInt		bagFlagsType;
	UInt		bagSize;
	BagPtr_t	bagCopyPtr;
	BagPtr_t	bagLinkPtr;
#ifdef DEBUG_BAG_MGMT
	UInt		bagCodeValue;			/* coded value to track bag origin (debug purposes) */
#endif
	UInt		*bagData;				/* data is arbitrary, length is defined by size */
} BagStruct_t;

/*
 *  bagCodeMarker is a value which can be set prior to entering a section of
 *  spiral code and causes all bags created or resized during that code
 *  section or operation to be tagged with "bagCodeMarker".  This is intended
 *  for debug purposes to track where the number of bags (& memory) grows
 *  significantly; then hopefully, allowing th epossibility to destroy these
 *  data when the code section or operation is complete.
 */

UInt		bagCodeMarker;
	

// TNUM_BAG, PTR_BAG, GET_FLAGS_BAG, SET_FLAG_BAG, CLEAR_FLAG_BAG are external function (memmgr.c)

#define IS_INTOBJ(o)    ((Int)(o) & 0x01)
#define GET_TYPE_BAG(obj)       (IS_INTOBJ( obj ) ? T_INT : TNUM_BAG( obj ))
#define TYPENAME(obj)   NameType[GET_TYPE_BAG(obj)]
//   #define ADDR_OBJ(bag)   PTR_BAG(bag) 

/* We have 16 bits to set flags on a bag. This is mostly used to avoid
   infinite recursion loops in circular structures like rec(a:=~) */
//   #define FLAGS(bag)          GET_FLAGS_BAG(bag)
//   #define HAS_FLAG(bag,val)   GET_FLAG_BAG(bag, val)
//   #define SET_FLAG(bag,val)   SET_FLAG_BAG(bag, val)
//   #define CLEAR_FLAG(bag,val) CLEAR_FLAG_BAG(bag, val)

#define WORDS_BAG(size) (((size) + (sizeof(BagPtr_t)-1)) / sizeof(BagPtr_t))


/****************************************************************************
**
*F  EnterKernel() . . . . . . . . . . . . . . for compatibility, does nothing
*F  ExitKernel( <hd> )  . . . . . . . . . . . for compatibility, does nothing
**
**  'EnterKernel' and 'ExitKernel' were used by GASMAN/GAP3 to  decide  which 
**  bags are still used and which can be thrown away.
** 
**  For  GASMAN/GAP4  these  functions  don't  do anything, and  are left for 
**  compatbility.
*/ 
extern  void            EnterKernel ( void );

extern  void            ExitKernel ( BagPtr_t hd );


/****************************************************************************
**
*F  CollectGarb() . . . . . . . . . . . . . . . . . . . . collect the garbage
**
**  'CollectGarb' performs a garbage collection.  This means it  removes  the
**  unused bags from memory and compacts the  used  bags  at  the  beginning.
*/
extern  void            CollectGarb ( void );


/****************************************************************************
**
*F  NewBag( <type>, <size> )  . . . . . . . . . . . . . . .  create a new bag
**
**  'NewBag' allocates memory  for  a new bag  of  the type <type>  and  size
**  <size>.   Usually <type> is   a symbolic  constant  defined  in  the file
**  declaration file of this package.  <size> is an  unsigned long.  'NewBag'
**  returns the handle of the new bag, which must be passed as first argument
**  to  all other Gasman functions identifying  this bag.  All  entrys of the
**  new bag are initialized to 0.
*/
extern  BagPtr_t       NewBag ( UInt type,  UInt size );


/****************************************************************************
**
*F  Retype( <hdBag>, <newType> )  . . . . . . . . .  change the type of a bag
**
**  'Retype' changes the type of the bag with the handle <hdBag> to  the  new
**  type <newType>.  The handle, the size and also the  absolute  address  of
**  the bag does not change.
**
**  Note that Gasman does not  take  any  responsibility  for this operation.
**  It is the responsibility of the caller to make sure that the contents  of
**  the bag make good sense even after the retyping.  'Retype' should be used
**  to temporary turn a negative integer into a positive one,  converting  an
**  autoread variable into a normal after reading the file and similar stuff.
*/
extern  void            Retype ( BagPtr_t hdBag, UInt newType );


/****************************************************************************
**
*F  Resize( <hdBag>, <newSize> )  . . . . . . . . .  change the size of a bag
**
**  'Resize' changes the size of the bag with the handle <hdBag> to  the  new
**  size <newSize>.  The handle of the bag does not change, but the  absolute
**  address might.  New entries, whether in the handle area or  in  the  date
**  area are initializes to zero.  If the size of the handle area of the  bag
**  changes 'Resize' will move the data area of the bag.
**
**  Note that 'Resize' may cause a garbage collection if the bag is enlarged.
*/
extern  void            Resize ( BagPtr_t hdBag, UInt newSize );


/****************************************************************************
**
*F  InitGasman(<pargc>)  . . . . . initialize dynamic memory manager package
**
**  'InitGasman' initializes   Gasman, i.e., allocates   some memory  for the
**  memory managment, and sets up the bags needed  for the working of Gasman.
**  This  are the new  handles bag, which  remembers the handles of bags that
**  must not be  thrown away during a  garbage  collection and  the free bag,
**  from  which  the memory for  newly   created bags  is taken by  'NewBag'.
**  'InitGasman'  must only   be  called  once  from  'InitGap'  right  after
**  initializing  the scanner,  but before  everything else,  as none of  the
**  storage manager functions work before calling 'InitGasman'.
*/

extern  void            InitGasman ( int *pargc, char **argv );


/****************************************************************************
**
*V  NrAllBags . . . . . . . . . . . . . . . . .  number of all bags allocated
*V  SizeAllBags . . . . . . . . . . . . . .  total size of all bags allocated
*V  NrLiveBags  . . . . . . . . . .  number of bags that survived the last gc
*V  SizeLiveBags  . . . . . . .  total size of bags that survived the last gc
*V  NrDeadBags  . . . . . . . number of bags that died since the last full gc
*V  SizeDeadBags  . . . . total size of bags that died since the last full gc
**
**  'NrAllBags'
**
**  'NrAllBags' is the number of bags allocated since Gasman was initialized.
**  It is incremented for each 'NewBag4' call.
**
**  'SizeAllBags'
**
**  'SizeAllBags'  is the  total  size  of bags   allocated since Gasman  was
**  initialized.  It is incremented for each 'NewBag4' call.
**
**  'NrLiveBags'
**
**  'NrLiveBags' is the number of bags that were  live after the last garbage
**  collection.  So after a full  garbage collection it is simply  the number
**  of bags that have been found to be still live by this garbage collection.
**  After a partial garbage collection it is the sum of the previous value of
**  'NrLiveBags', which is the number  of live old  bags, and  the number  of
**  bags that  have been found to  be still live  by this garbage collection,
**  which is  the number of live   young  bags.   This  value  is used in the
**  information messages,  and to find  out  how  many  free  identifiers are
**  available.
**
**  'SizeLiveBags'
**
**  'SizeLiveBags' is  the total size of bags  that were  live after the last
**  garbage collection.  So after a full garbage  collection it is simply the
**  total size of bags that have been found to  be still live by this garbage
**  collection.  After  a partial  garbage  collection it  is the sum  of the
**  previous value of  'SizeLiveBags', which is the total   size of live  old
**  bags, and the total size of bags that have been found to be still live by
**  this garbage  collection,  which is  the  total size of  live young bags.
**  This value is used in the information messages.
**
**  'NrDeadBags'
**
**  'NrDeadBags' is  the number of bags that died since the last full garbage
**  collection.   So after a  full garbage  collection this is zero.  After a
**  partial  garbage  collection it  is  the  sum  of the  previous value  of
**  'NrDeadBags' and the  number of bags that  have been found to be dead  by
**  this garbage collection.  This value is used in the information messages.
**
**  'SizeDeadBags'
**
**  'SizeDeadBags' is  the total size  of bags that  died since the last full
**  garbage collection.  So  after a full   garbage collection this  is zero.
**  After a partial garbage collection it is the sum of the previous value of
**  'SizeDeadBags' and the total size of bags that have been found to be dead
**  by  this garbage  collection.   This  value  is used  in the  information
**  message.
**
**  'NrHalfDeadBags'
**
**  'NrHalfDeadBags'  is  the number of  bags  that  have  been  found to  be
**  reachable only by way of weak pointers since the last garbage collection.
**  The bodies of these bags are deleted, but their identifiers are marked so
**  that weak pointer objects can recognize this situation.
*/

extern  UInt                    NrAllBags;
extern  UInt                    SizeAllBags;
extern  UInt                    NrLiveBags;
extern  UInt                    SizeLiveBags;
extern  UInt                    NrDeadBags;
extern  UInt                    SizeDeadBags;
extern  UInt                    NrHalfDeadBags;

/****************************************************************************
**
*V  InfoBags[<type>]  . . . . . . . . . . . . . . . . .  information for bags
**
**  'InfoBags[<type>]'
**
**  'InfoBags[<type>]'  is a structure containing information for bags of the
**  type <type>.
**
**  'InfoBags[<type>].name' is the name of the type <type>.   Note that it is
**  the responsibility  of  the  application using  {\Gasman}   to enter this
**  name.
**
**  'InfoBags[<type>].nrLive' is the number of  bags of type <type> that  are
**  currently live.
**
**  'InfoBags[<type>].nrAll' is the total  number of all  bags of <type> that
**  have been allocated.
**
**  'InfoBags[<type>].sizeLive' is the sum of the  sizes of  the bags of type
**  <type> that are currently live.
**
**  'InfoBags[<type>].sizeAll'  is the sum of the  sizes of all  bags of type
**  <type> that have been allocated.
**
**  This  information is only  kept if {\Gasman} is  compiled with the option
**  '-DCOUNT_BAGS', e.g., with 'make <target> COPTS=-DCOUNT_BAGS'.
*/

/*
 * Added # dead & # halfdead; use this to analyze all bags during a GC
 * -- track both dead & alive bags...
 */

typedef struct  {
    const Char *            name;
    UInt                    nrLive;
    UInt                    nrAll;
	UInt					nrDead;
	UInt					nrHalfdead;
	UInt					nrRemnant;
    UInt                    sizeLive;
    UInt                    sizeAll;
} TNumInfoBags;

extern  TNumInfoBags            InfoBags [ NTYPES ];

extern const char* TNAM_BAG(BagPtr_t bag);



/****************************************************************************
**
*F  InitMsgsFuncBags(<msgs-func>) . . . . . . . . .  install message function
**
**  'InitMsgsFuncBags( <msgs-func> )'
**
**  'InitMsgsFuncBags' installs the function <msgs-func> as function used  by
**  {\Gasman} to print messages during garbage collections.  <msgs-func> must
**  be a function of  three 'unsigned  long'  arguments <full>,  <phase>, and
**  <nr>.   <msgs-func> should format  this information and  display it in an
**  appropriate way.   In fact you will usually  want to ignore  the messages
**  for partial  garbage collections, since there are  so many  of those.  If
**  you do not install a  messages  function  with  'InitMsgsFuncBags',  then
**  'CollectBags' will be silent.
**
**  If <full> is 1, the current garbage collection is a full one.  If <phase>
**  is 0, the garbage collection has just started and  <nr> should be ignored
**  in this case.  If <phase> is 1 respectively 2, the garbage collection has
**  completed the mark phase  and <nr> is  the total number  respectively the
**  total  size of live bags.   If <phase> is  3  respectively 4, the garbage
**  collection  has completed  the  sweep  phase,   and <nr>  is   the number
**  respectively the total size of bags that died since the last full garbage
**  collection.  If <phase> is  5 respectively 6,  the garbage collection has
**  completed the check phase   and <nr> is    the size of the free   storage
**  respectively the size of the workspace.  All sizes are measured in KByte.
**
**  If  <full> is 0,  the current garbage  collection  is a  partial one.  If
**  <phase> is 0, the garbage collection has just  started and <nr> should be
**  ignored  in  this  case.  If  <phase>  is 1  respectively 2,  the garbage
**  collection  has   completed the  mark   phase  and  <nr>   is the  number
**  respectively the  total size  of bags allocated  since  the last  garbage
**  collection that  are still  live.   If <phase> is  3 respectively  4, the
**  garbage collection has completed  the sweep phase and  <nr> is the number
**  respectively the   total size of   bags allocated since  the last garbage
**  collection that are already dead (thus the sum of the values from phase 1
**  and 3  is the  total number of   bags  allocated since the  last  garbage
**  collection).  If <phase> is 5 respectively 6,  the garbage collection has
**  completed the  check phase  and <nr>  is   the size of  the  free storage
**  respectively the size of the workspace.  All sizes are measured in KByte.
**
**  The message  function  should display   the information  for each   phase
**  immediatly, i.e.,  by calling 'flush' if the  output device is a file, so
**  that the user has some indication how much time each phase used.
**
**  For example {\GAP} displays messages for  full garbage collections in the
**  following form{\:}
**
**    #G  FULL  47601/ 2341KB live  70111/ 5361KB dead   1376/ 4096KB free
**
**  where 47601 is the total number of bags surviving the garbage collection,
**  using 2341 KByte, 70111 is  the total number  of bags that died since the
**  last full garbage  collection, using 5361  KByte, 1376 KByte are free and
**  the total size of the workspace is 4096 KByte.
**
**  And partial garbage collections are displayed in  {\GAP} in the following
**  form{\:}
**
**    #G  PART     34/   41KB+live   3016/  978KB+dead   1281/ 4096KB free
**
**  where  34 is the  number of young bags that  were live after this garbage
**  collection, all the old bags survived it  anyhow, using 41 KByte, 3016 is
**  the number of young bags that died since  the last garbage collection, so
**  34+3016 is the  number  of bags allocated  between  the last two  garbage
**  collections, using 978 KByte and the other two numbers are as above.
*/
typedef void            (* TNumMsgsFuncBags) (
            UInt                full,
            UInt                phase,
            Int                 nr );

extern  void            InitMsgsFuncBags (
            TNumMsgsFuncBags    msgs_func );
            
extern  TNumMsgsFuncBags    MsgsFuncBags;

/****************************************************************************
**
*V  GlobalBags  . . . . . . . . . . . . . . . . . . . . . list of global bags
*/
#ifndef NR_GLOBAL_BAGS
#define NR_GLOBAL_BAGS  20000L
#endif


typedef struct {
    BagPtr_t *                   addr [NR_GLOBAL_BAGS];
#ifdef IN_DGC
    BagPtr_t                     lastValue [NR_GLOBAL_BAGS];
#endif
    const Char *            cookie [NR_GLOBAL_BAGS];
    UInt                    nr;
} TNumGlobalBags;

extern TNumGlobalBags GlobalBags;
extern UInt GlobalSortingStatus;

/****************************************************************************
**
*F  InitGlobalBag(<addr>) . . . . . inform Gasman about global bag identifier
**
**  'InitGlobalBag( <addr>, <cookie> )'
**
**  'InitGlobalBag'  informs {\Gasman} that there is  a bag identifier at the
**  address <addr>, which must be of  type '(BagPtr_t\*)'.  {\Gasman} will look at
**  this address for a bag identifier during a garbage collection.
**
**  The application *must* call 'InitGlobalBag' for every global variable and
**  every entry of a  global array that may hold  a bag identifier.  It is no
**  problem  if  such a  variable does not   actually  hold a bag identifier,
**  {\Gasman} will simply ignore it then.
**
**  There is a limit on the number of calls to 'InitGlobalBags', which is 20,000
**  by default.   If the application has  more global variables that may hold
**  bag  identifier, you  have to  compile  {\Gasman} with a  higher value of
**  'NR_GLOBAL_BAGS', i.e., with 'make COPTS=-DNR_GLOBAL_BAGS=<nr>'.
**
**  <cookie> is a C string, which should uniquely identify this global
**  bag from all others.  It is used  in reconstructing  the Workspace
**  after a save and load
*/

extern void InitGlobalBag (
            BagPtr_t *               addr,
            const Char *        cookie );

/* byWhat can be only 2 - sort by cookie */
extern void SortGlobals( UInt byWhat );

extern Int GlobalIndexByCookie( const Char * cookie );

extern BagPtr_t * GlobalByCookie( const Char * cookie );

/* #include "gasman4.h" */


/****************************************************************************
**
** This version of GASMAN / GAP4 is ported for use in SPIRAL (GAP3 branch)
** Yevgen S. Voronenko (yevgen@drexel.edu)
**
** GASMAN/GAP4 is generational, but  requires  the  programmer  to notify it,
** when bag references change. For example, when one modifies  list elements,
** then list references change. The  notification  happens  on  CHANGED_BAG()
** macro call.
**
** Since, it  will be too much effort to modify all GAP3 modules to use this,
** we have disabled generational garbage collection. This behaviour is manda-
** tory for GAP3 modules to work, and is set by GASMAN_ALWAYS_COLLECT_FULL.
**
** GASMAN was also optimized for more aggressive memory allocation. This
** is controlled by GASMAN_MIN_ALLOC_BLOCK and GASMAN_FREE_RATIO defines.
** This behaviour was not modifiable in original GASMAN / GAP4.
**
*/

/*========= GASMAN configuration ==========================================*/


#define GASMAN_ALWAYS_COLLECT_FULL  /* absolutely required for GAP3        */
#define GASMAN_MIN_ALLOC_BLOCK 16384 /* in KB, allocate in 16MB increments   */
#define GASMAN_FREE_RATIO      0.05  /* 5% of workspace will be kept free  */


/* 
 * With the definition of BagStruct_t we don't have to assume aspecific order/place
 * for elements of the structure; these macros are defined to use the structure
 * (casting the incoming pointer as needed) to get at the required element.  The
 * macros are intended for internal memory management ONLY!!
 *
 * The reason they are in this file is that the exported functions may be inlined and
 * thus need those macros.
*/

//		#define HEADER_SIZE 4
#define HEADER_SIZE ((sizeof(BagStruct_t) - sizeof(UInt *)) / sizeof(UInt *))

#define DATA_PTR(ptr)         ((UInt **)&(((BagStruct_t *)(ptr))->bagData))
#define COPY_PTR(ptr)         (((BagStruct_t *)(ptr))->bagCopyPtr)
#define SIZE_PTR(ptr)         (((BagStruct_t *)(ptr))->bagSize)
#define FLAGS_TYPE_PTR(ptr)   (((BagStruct_t *)(ptr))->bagFlagsType)

#define GET_COPY_PTR(ptr)          (COPY_PTR(ptr))
#define GET_SIZE_PTR(ptr)          (SIZE_PTR(ptr))
#define GET_TYPE_PTR(ptr)          (FLAGS_TYPE_PTR(ptr) & TYPE_BIT_MASK)
#define TEST_FLAG_PTR(ptr,flag)     (FLAGS_TYPE_PTR(ptr) & (flag))

#ifdef DEBUG_POINTERS
#undef   LINK_PTR
BagPtr_t GET_LINK_PTR(BagPtr_t ptr);
void     SET_LINK_PTR(BagPtr_t ptr, BagPtr_t val);
#else
#define LINK_PTR(ptr)         (((BagStruct_t *)(ptr))->bagLinkPtr)
#define GET_LINK_PTR(ptr)          (LINK_PTR(ptr))
#define SET_LINK_PTR(ptr,val)       (LINK_PTR(ptr) = (BagPtr_t)val)
#endif

#define SET_COPY_PTR(ptr,val)       (COPY_PTR(ptr) = (BagPtr_t)val)
#define SET_SIZE_PTR(ptr,val)       (SIZE_PTR(ptr) = (UInt)val)
#define SET_TYPE_PTR(ptr,val)       ((FLAGS_TYPE_PTR(ptr)) = ((FLAGS_TYPE_PTR(ptr)) & BF_ALL_FLAGS) | val)
#define SET_FLAG_PTR(ptr,val)       ((FLAGS_TYPE_PTR(ptr)) |= (val))
#define CLEAR_FLAG_PTR(ptr,val)     ((FLAGS_TYPE_PTR(ptr)) &= ~(val))
#define BLANK_FLAGS_PTR(ptr)        (FLAGS_TYPE_PTR(ptr) &= TYPE_BIT_MASK)


/*========= Original (but slightly annotated) code starts here =============*/

/****************************************************************************
**
*W  gasman4.h                    GAP source                   Martin Schoenert
**
**
**
**  This file declares  the functions of  Gasman,  the  GAP  storage manager.
**
**  {\Gasman} is a storage manager for applications written in C.  That means
**  that an application requests blocks  of storage from {\Gasman}, which are
**  called bags.   After using a bag   to store data,  the application simply
**  forgets the  bag,  and  we  say that  such a  block  is dead.   {\Gasman}
**  implements   an   automatic,  cooperating,   compacting,    generational,
**  conservative storage manager.  Automatic  means that the application only
**  allocates  bags,   but need  not  explicitly   deallocate them.   This is
**  important for large or  complex application, where explicit  deallocation
**  is difficult.  Cooperating means  that the allocation must cooperate with
**  {\Gasman}, i.e., must follow certain rules.  This information provided by
**  the   application makes  {\Gasman}   use  less  storage and   run faster.
**  Compacting  means that {\Gasman} always compacts  live bags such that the
**  free storage is  one large block.  Because  there is  no fragmentation of
**  the  free  storage   {\Gasman} uses   as   little storage   as  possible.
**  Generational  means  that {\Gasman}  usually assumes  that bags that have
**  been live for some  time are still live.  This   means that it can  avoid
**  most of the tests whether a bag is still live or already dead.  Only when
**  not enough storage can be reclaimed under this assumption does it perform
**  all the  tests.  Conservative means  that {\Gasman} may  keep bags longer
**  than necessary because   the  C compiler   does  not provide   sufficient
**  information to distinguish true references to bags from other values that
**  happen to  look like references.
*/


/* 
 * This definition switches to the bigger bag header, supporting bags up to
 * 4GB in length (lists limited to 1GB for other reasons) 
 */

/****************************************************************************
**

*T  BagPtr_t  . . . . . . . . . . . . . . . . . type of the identifier of a bag
**
**  'Bag'
**
**  Each bag is identified by its *bag identifier*.  That is each bag has a
**  bag identifier and no two live bags have the same identifier.  'BagPtr_t'
**  is the type of bag identifiers.
**
**  0 is a valid value of the type 'BagPtr_t', but is guaranteed not to be the
**  identifier of any bag.
**
**  'NewBag4' returns the identifier of the newly allocated bag and the
**  application passes this identifier to every {\Gasman} function to tell it
**  which bag it should operate on (see "NewBag4", "TNUM_BAG", "GET_SIZE_BAG",
**  "PTR_BAG", "CHANGED_BAG", "RetypeBag", and "ResizeBag").
**
**  Note that the  identifier of a  bag is different from  the address of the
**  data area  of  the  bag.  This  address  may   change during  a   garbage
**  collection while the identifier of a bag never changes.
**
**  Bags  that contain references  to   other bags  must  always contain  the
**  identifiers of these other bags, never the addresses of the data areas of
**  the bags.
**
**  Note that bag identifiers are recycled.  That means that after a bag dies
**  its identifier may be reused for a new bag.
**
*/

/****************************************************************************
**
*F  PTR_BAG(<bag>)  . . . . . . . . . . . . . . . . . . . .  pointer to a bag
**
**  'PTR_BAG( <bag> )'
**
**  'PTR_BAG' returns the address of the data area of the bag with identifier
**  <bag>.  Using  this pointer the application  can then  read data from the
**  bag or write  data  into it.  The  pointer   is of  type pointer  to  bag
**  identifier, i.e., 'PTR_BAG(<bag>)[0]' is   the  identifier of the   first
**  subbag of the bag, etc.  If the bag contains  data in a different format,
**  the  application has  to  cast the pointer  returned  by 'PTR_BAG', e.g.,
**  '(long\*)PTR_BAG(<bag>)'.
**
**  Note  that the address  of the data area  of a  bag  may change  during a
**  garbage collection.  That is  the  value returned by 'PTR_BAG' may differ
**  between two calls, if 'NewBag4', 'ResizeBag', 'CollectBags', or any of the
**  application\'s functions  or macros that may   call   those functions, is
**  called in between (see "NewBag4", "ResizeBag", "CollectBags").
**
**  The first rule for using {\Gasman} is{\:} *The  application must not keep
**  any pointers to or into the data area of any  bag over calls to functions
**  that may cause a garbage collection.*
**
**  That means that the following code is incorrect{\:}
**
**      ptr = PTR_BAG( old );
**      new = NewBag4( typeNew, sizeNew );
**      *ptr = new;
**
**  because  the creation of  the new bag may  move the  old bag, causing the
**  pointer to  point  no longer to  the  data area of  the bag.   It must be
**  written as follows{\:}
**
**      new = NewBag4( typeNew, sizeNew );
**      ptr = PTR_BAG( old );
**      *ptr = new;
**
**  Note that even the following is incorrect{\:}
**
**      PTR_BAG(old)[0] = NewBag4( typeNew, sizeNew );
**
**  because a C compiler is free to compile it to  a sequence of instructions
**  equivalent to first example.  Thus whenever  the evaluation of a function
**  or  a macro  may cause a  garbage collection  there  must be   no call to
**  'PTR_BAG' in the same expression, except as argument  to this function or
**  macro.
**
**  Note that  after writing   a bag  identifier,  e.g.,  'new' in the  above
**  example, into the  data area of another  bag, 'old' in the above example,
**  the application  must inform {\Gasman}  that it  has changed  the bag, by
**  calling 'CHANGED_BAG(old)' in the above example (see "CHANGED_BAG").
**
**  Note that 'PTR_BAG' is a macro, so  do  not call it with  arguments  that
**  have sideeffects.
*/

/* _PTR_BAG is needed to be able to define PTR_BAG as a function for the debugger */
// #define _PTR_BAG(bag)    (*(Bag**)(bag))
#define _PTR_BAG(bag)    (*(BagPtr_t **)(bag))

// Make these standard functions (can optimize later if this is determined to be a bottleneck)
// Given a handle to the bag (BagPtr_t) return the address of the data
BagPtr_t *PTR_BAG(const BagPtr_t bag);

// Store the address [of the bag data] in a handle
void SET_PTR_BAG(BagPtr_t bag, BagPtr_t *dst);

//#define PTR_BAG(bag)    (*(Bag**)(bag))
/* static inline Bag* PTR_BAG(const Bag bag){ */
/*   return (*(Bag**)(bag)); */
/* } */

/* static inline void SET_PTR_BAG(Bag bag,Bag* dst){ */
/*   (*(Bag**)(bag)) = dst; */
/* } */


/****************************************************************************
**
*F  TNUM_BAG(<bag>) . . . . . . . . . . . . . . . . . . . . . . type of a bag
**
**  'TNUM_BAG( <bag> )'
**
**  'TNUM_BAG' returns the type of the the bag with the identifier <bag>.
**
**  Each bag has a certain type that identifies its structure.  The type is a
**  integer between 0 and 253.  The types 254 and 255 (T_RESIZE_FREE) are
**  reserved for {\Gasman}.  The application specifies the type of a bag when
**  it allocates it with 'NewBag4' and may later change it with 'RetypeBag'
**  (see "NewBag4" and "RetypeBag").
**
**  {\Gasman} needs to know the type of a bag so that it knows which function
**  to  call  to  mark all subbags  of a  given bag (see "InitMarkFuncBags").
**  Apart from that {\Gasman} does not care at all about types.
**
**  Note  that 'TNUM_BAG' is a macro, so do not call  it with arguments  that
**  have sideeffects.
*/

// Make these standard functions (can optimize later if this is determined to be a bottleneck)
// Set the type for the bag
void SET_TYPE_BAG(BagPtr_t bag, UInt val);
// Clear the flags associated to a bag 
void BLANK_FLAGS_BAG(BagPtr_t bag);
// Return the type associated to a bag
UInt TNUM_BAG(BagPtr_t bag);
// Get the flags associated to a bag
UInt GET_FLAGS_BAG(BagPtr_t bag);
// Set the flags associated to a bag
void SET_FLAG_BAG(BagPtr_t bag, UInt val);
// Clear a flag associated to the bag
void CLEAR_FLAG_BAG(BagPtr_t bag, UInt val);
// Test if a flag is associated to a bag
UInt GET_FLAG_BAG(BagPtr_t bag, UInt val);

/* static inline void SET_TYPE_BAG(Bag bag,UInt val){ */
/*   SET_TYPE_PTR(PTR_BAG(bag)-HEADER_SIZE,val); */
/* } */

/* static inline void BLANK_FLAGS_BAG(Bag bag){ */
/*   BLANK_FLAGS_PTR(PTR_BAG(bag)-HEADER_SIZE); */
/* } */

/* static inline UInt TNUM_BAG(Bag bag){ */
/*   if (bag==0) */
/*     return T_ILLEGAL; */
/*   else */
/*     return GET_TYPE_PTR( PTR_BAG(bag)-HEADER_SIZE ); */
/* } */

/* static inline UInt GET_FLAGS_BAG(Bag bag){ */
/*   return READ_FLAGS_PTR( PTR_BAG(bag)-HEADER_SIZE ); */
/* } */

/* static inline void SET_FLAG_BAG(Bag bag, UInt val){ */
/*   SET_FLAG_PTR( PTR_BAG(bag)-HEADER_SIZE, val ); */
/* } */


/* static inline void CLEAR_FLAG_BAG(Bag bag, UInt val){ */
/*   CLEAR_FLAG_PTR( PTR_BAG(bag)-HEADER_SIZE, val ); */
/* } */

/* static inline UInt GET_FLAG_BAG(Bag bag, UInt val){ */
/*   return (GET_FLAGS_BAG(bag) & (val)); */
/* } */

//#define TNAM_BAG(bag)                 (InfoBags[ TNUM_BAG(bag) ].name)

/*UInt BID    ( Bag bag );
  Bag  BAG    ( UInt bid );*/

UInt IS_BAG ( BagPtr_t bag );

/****************************************************************************
**
*F  GET_SIZE_BAG(<bag>) . . . . . . . . . . . . . . . . . . . . . . size of a bag
**
**  'GET_SIZE_BAG( <bag> )'
**
**  'GET_SIZE_BAG' returns  the  size of the bag   with the identifier  <bag> in
**  bytes.
**
**  Each bag has a  certain size.  The size  of a bag  is measured  in bytes.
**  The  size must  be a   value between   0 and $2^{24}-1$.  The application
**  specifies the size of a bag when it  allocates  it  with 'NewBag4' and may
**  later change it with 'ResizeBag' (see "NewBag4" and "ResizeBag").
**
**  Note that  'GET_SIZE_BAG' is  a macro,  so do not call it with arguments that
**  have sideeffects.
*/

// Make these standard functions (can optimize later if this is determined to be a bottleneck)
// Get the size of [the data in] a bag
UInt GET_SIZE_BAG(BagPtr_t bag);
// Set the size of a bag [size of data in the bag]
void SET_SIZE_BAG(BagPtr_t bag, UInt val);
// Get the link pointer associated to a bag
BagPtr_t GET_LINK_BAG(BagPtr_t bag);
// Set the link pointer to be associated to a bag
void SET_LINK_BAG(BagPtr_t bag, BagPtr_t val);
// Get the copy pointer associated to a bag
BagPtr_t GET_COPY_BAG(BagPtr_t bag);
// Set the copy pointer associated to a bag
void SET_COPY_BAG(BagPtr_t bag, BagPtr_t val);

/* static inline UInt GET_SIZE_BAG(Bag bag){ */
/*   return GET_SIZE_PTR( PTR_BAG(bag)-HEADER_SIZE ); */
/* } */

/* static inline void SET_SIZE_BAG(Bag bag, UInt val){ */
/*   SET_SIZE_PTR( PTR_BAG(bag)-HEADER_SIZE , val); */
/* } */

/* static inline Bag GET_LINK_BAG(Bag bag){ */
/*   return (Bag)(GET_LINK_PTR( PTR_BAG(bag)-HEADER_SIZE )); */
/* } */

/* static inline void SET_LINK_BAG(Bag bag, Bag val){ */
/*   SET_LINK_PTR( PTR_BAG(bag)-HEADER_SIZE, (UInt) val ); */
/* } */

/* static inline Bag GET_COPY_BAG(Bag bag){ */
/*   return (Bag)READ_COPY_PTR( PTR_BAG(bag)-HEADER_SIZE ); */
/* } */

/* static inline void SET_COPY_BAG(Bag bag, Bag val){ */
/*   SET_COPY_PTR( PTR_BAG(bag)-HEADER_SIZE, (UInt) val ); */
/* } */

/****************************************************************************
**
*F  CHANGED_BAG(<bag>)  . . . . . . . .  notify Gasman that a bag has changed
**
**  'CHANGED_BAG( <bag> )'
**
**  'CHANGED_BAG'  informs {\Gasman} that the bag   with identifier <bag> has
**  been changed by an assignment of another bag identifier.
**
**  The  second rule for using  {\Gasman} is{\:} *After  each assignment of a
**  bag identifier into a  bag the application  must inform {\Gasman} that it
**  has changed the bag before the next garbage collection can happen.*
**
**  Note that the  application need not inform {\Gasman}  if it changes a bag
**  by assigning something that is not an identifier of another bag.
**
**  For example to copy all entries from  one list into another the following
**  code must be used{\:}
**
**      for ( i = 0; i < GET_SIZE_BAG-BAG(old)/sizeof(Bag); i++ )
**          PTR_BAG(new)[i] = PTR_BAG(old)[i];
**      CHANGED_BAG( new );
**
**  On the other  hand  when the  application  allocates a matrix,  where the
**  allocation of each row may cause a garbage collection, the following code
**  must be used{\:}
**
**      mat = NewBag4( T_MAT, n * sizeof(Bag) );
**      for ( i = 0; i < n; i++ ) {
**          row = NewBag4( T_ROW, n * sizeof(Bag) );
**          PTR_BAG(mat)[i] = row;
**          CHANGED_BAG( mat );
**      }
**
**  Note that  writing 'PTR_BAG(mat)[i] = NewBag4( T_ROW, n\*\ sizeof(Bag) );'
**  is incorrect as mentioned in the section for 'PTR_BAG' (see "PTR_BAG").
**
**  Note that 'CHANGED_BAG' is a macro, so do not call it with arguments that
**  have sideeffects.
*/
extern  BagPtr_t *                   YoungBags;

extern  BagPtr_t                     ChangedBags;

void CHANGED_BAG(BagPtr_t bag);

/* static inline void CHANGED_BAG(Bag bag){ */
/*   if (   PTR_BAG(bag) <= YoungBags && PTR_BAG(bag)[-1] == (bag) ){ */
/*     PTR_BAG(bag)[-1] = ChangedBags; */
/*     ChangedBags = (bag); */
/*   } */
/* } */

/****************************************************************************
**
*F  NewBag4(<type>,<size>) . . . . . . . . . . . . . . . .  allocate a new bag
**
**  'NewBag4( <type>, <size> )'
**
**  'NewBag4' allocates a new bag  of type <type> and  <size> bytes.  'NewBag4'
**  returns the  identifier  of the new  bag,  which must be  passed as first
**  argument to all other {\Gasman} functions.
**
**  <type> must be a value between 0 and 253.  The types 254 and 255
**  (T_RESIZE_FREE) are reserved for {\Gasman}.  The application can find the
**  type of a bag with 'TNUM_BAG' and change it with 'RetypeBag' (see
**  "TNUM_BAG" and "RetypeBag").
**
**  It is probably a good idea to define symbolic constants  for all types in
**  a  system wide   header  file,  e.g.,  'types.h', if   only  to avoid  to
**  accidently use the same value for two different types.
**
**  <size> is the size of the new bag in bytes and must be a value  between 0
**  and $2^{24}-1$.  The   application can find  the    size  of a   bag with
**  'GET_SIZE_BAG'    and  change   it  with  'ResizeBag'   (see   "GET_SIZE_BAG" and
**  "ResizeBag").
**
**  If the initialization flag <dirty> is 0, all entries of  the new bag will
**  be initialized to 0; otherwise the  entries  of the  new bag will contain
**  random values (see "InitBags").
**
**  What  happens if {\Gasman}  cannot  get  enough  storage  to perform   an
**  allocation     depends on  the  behavior    of   the allocation  function
**  <alloc-func>.  If <alloc-func>  returns 0  when it   cannot do a   needed
**  extension  of  the  workspace, then 'NewBag4'   may  return 0  to indicate
**  failure, and the  application has to check the  return  value of 'NewBag4'
**  for this.  If <alloc-func> aborts when it cannot do a needed extension of
**  the workspace,  then the  application will  abort if  'NewBag4' would  not
**  succeed.  So in  this case whenever 'NewBag4'  returns,  it succeeded, and
**  the application need  not check    the return  value of 'NewBag4'     (see
**  "InitBags").
**
**  Note that 'NewBag4'  will perform a garbage collection  if no free storage
**  is available.  During  the  garbage  collection the addresses of the data
**  areas of all bags may  change.  So you  must not keep any  pointers to or
**  into the data areas of bags over calls to 'NewBag4' (see "PTR_BAG").
*/
extern BagPtr_t NewBag4( UInt type, UInt size );


/****************************************************************************
**
*F  RetypeBag(<bag>,<new>)  . . . . . . . . . . . .  change the type of a bag
**
**  'RetypeBag( <bag>, <new> )'
**
**  'RetypeBag' changes the type of the bag with identifier <bag>  to the new
**  type <new>.  The identifier, the size,  and also the  address of the data
**  area of the bag do not change.
**
**  'Retype' is usually  used to set or  reset  flags that are stored  in the
**  type of  a bag.   For  example in {\GAP}  there are  two  types of  large
**  integers, one for  positive integers and  one for negative  integers.  To
**  compute the difference of a positive integer and  a negative, {\GAP} uses
**  'RetypeBag'  to temporarily change   the second argument into  a positive
**  integer, and then adds the two operands.  For another example when {\GAP}
**  detects that a list is sorted and contains  no duplicates, it changes the
**  type  of the bag  with 'RetypeBag', and the new  type indicates that this
**  list has this property, so that this need not be tested again.
**
**  It is, as usual, the responsibility of the application to ensure that the
**  data stored in the bag makes sense when the  bag is interpreted  as a bag
**  of type <type>.
*/
extern void RetypeBag( BagPtr_t bag, UInt new_type );


/****************************************************************************
**
*F  ResizeBag(<bag>,<new>)  . . . . . . . . . . . .  change the size of a bag
**
**  'ResizeBag( <bag>, <new> )'
**
**  'ResizeBag' changes the size of the bag with  identifier <bag> to the new
**  size <new>.  The identifier  of the bag  does not change, but the address
**  of the data area  of the bag  may change.  If  the new size <new> is less
**  than the old size,  {\Gasman} throws away any data  in the bag beyond the
**  new size.  If the new size  <new> is larger than  the old size, {\Gasman}
**  extends the bag.
**
**  If the initialization flag <dirty> is 0, all entries of an extension will
**  be initialized to 0; otherwise the  entries of an  extension will contain
**  random values (see "InitBags").
**
**  What happens  if {\Gasman} cannot   get   enough storage to  perform   an
**  extension depends   on   the   behavior   of the   allocation    function
**  <alloc-func>.  If <alloc-func>   returns 0 when   it cannot do a   needed
**  extension of the  workspace, then  'ResizeBag'  may return 0 to  indicate
**  failure, and the application has to check the return value of 'ResizeBag'
**  for this.  If <alloc-func> aborts when it cannot do a needed extension of
**  the workspace, then  the application will abort  if 'ResizeBag' would not
**  succeed.  So in this case whenever 'ResizeBag' returns, it succeeded, and
**  the application   need not check   the return value  of  'ResizeBag' (see
**  "InitBags").
**
**  Note   that  'ResizeBag' will  perform a garbage   collection  if no free
**  storage is available.  During the garbage collection the addresses of the
**  data areas of all bags may change.  So you must not keep  any pointers to
**  or into the data areas of bags over calls to 'ResizeBag' (see "PTR_BAG").
*/
extern UInt ResizeBag( BagPtr_t bag, UInt new_size );


/****************************************************************************
**
*F  CollectBags(<size>,<full>)  . . . . . . . . . . . . . . collect dead bags
**
**  'CollectBags( <size>, <full> )'
**
**  'CollectBags' performs a  garbage collection.  This means  it deallocates
**  the dead   bags and  compacts the  live   bags at the  beginning   of the
**  workspace.   If    <full>  is 0, then   only   the  dead young  bags  are
**  deallocated, otherwise all dead bags are deallocated.
**
**  If the application calls 'CollectBags', <size> must be 0, and <full> must
**  be 0  or 1 depending on whether  it wants to perform  a partial or a full
**  garbage collection.
**
**  If 'CollectBags'  is called from  'NewBag4' or 'ResizeBag',  <size> is the
**  size of the bag that is currently allocated, and <full> is zero.
**
**  Note that  during the garbage collection the  addresses of the data areas
**  of all bags may change.  So you must not keep any pointers to or into the
**  data areas of bags over calls to 'CollectBags' (see "PTR_BAG").
*/
extern UInt CollectBags( UInt size, UInt full );


/****************************************************************************
**
*F  SwapMasterPoint( <bag1>, <bag2> ) . . . swap pointer of <bag1> and <bag2>
*/
extern void SwapMasterPoint( BagPtr_t bag1, BagPtr_t bag2 );



/****************************************************************************
**
*F  InitMarkFuncBags(<type>,<mark-func>)  . . . . .  install marking function
*F  MarkNoSubBags(<bag>)  . . . . . . . . marking function that marks nothing
*F  MarkOneSubBags(<bag>) . . . . . .  marking function that marks one subbag
*F  MarkTwoSubBags(<bag>) . . . . . . marking function that marks two subbags
*F  MarkAllSubBags(<bag>) . . . . . .  marking function that marks everything
*F  MARK_BAG(<bag>) . . . . . . . . . . . . . . . . . . .  mark a bag as live
**
**  'InitMarkFuncBags( <type>, <mark-func> )'
**
**  'InitMarkFuncBags' installs the function <mark-func>  as marking function
**  for bags  of  type <type>.   The  application  *must* install  a  marking
**  function for a  type before it allocates  any  bag  of  that type.  It is
**  probably best to install all marking functions before allocating any bag.
**
**  A marking function  is a function  that takes a  single  argument of type
**  'BagPtr_t' and returns nothing, i.e., has return type 'void'.  Such a function
**  must apply the  macro 'MARK_BAG' to each bag  identifier that  appears in
**  the bag (see below).
**
**  Those functions are applied during the garbage  collection to each marked
**  bag, i.e., bags  that are assumed to be  still live,  to also mark  their
**  subbags.  The ability to use the correct marking function is the only use
**  that {\Gasman} has for types.
**
**  'MARK_BAG( <bag> )'
**
**  'MARK_BAG' marks the <bag> as live so that it is  not thrown away during
**  a garbage collection.  'MARK_BAG' should only be called from the marking
**  functions installed with 'InitMarkFuncBags'.
**
**  'MARK_BAG' tests  if <bag> is  a valid identifier of a  bag  in the young
**  bags  area.  If it is not,  then 'MARK_BAG' does nothing,  so there is no
**  harm in  calling 'MARK_BAG' for  something   that is not actually  a  bag
**  identifier.
**
**  Note that 'MARK_BAG' is a macro, so do not call it with an argument that
**  has sideeffects.
**
**  'MarkBagWeakly( <bag> )'
**
**  'MarkBagWeakly' is an alternative to MARK_BAG, intended to be used by the
**  marking functions  of weak pointer objects.  A  bag which is  marked both
**  weakly and strongly  is treated as strongly marked.   A bag which is only
**  weakly marked will be recovered by garbage collection, but its identifier
**  remains, marked      in   a    way    which   can     be   detected    by
**  "IS_WEAK_DEAD_BAG". Which should  always be   checked before copying   or
**  using such an identifier.
**
**
**  {\Gasman} already provides the following marking functions.
**
**  'MarkNoSubBags( <bag> )'
**
**  'MarkNoSubBags'  is a marking function   for types whose  bags contain no
**  identifier of other   bags.  It does nothing,  as  its name implies,  and
**  simply returns.  For example   in  {\GAP} the  bags for   large  integers
**  contain only the digits and no identifiers of bags.
**
**  'MarkOneSubBags( <bag> )'
**
**  'MarkOneSubBags'  is  a  marking  function for types   whose bags contain
**  exactly one identifier of another bag as  the first entry.  It marks this
**  subbag and returns.  For example in {\GAP} bags for finite field elements
**  contain exactly one  bag  identifier  for the  finite field   the element
**  belongs to.
**
**  'MarkTwoSubBags( <bag> )'
**
**  'MarkTwoSubBags' is  a  marking function   for types  whose bags  contain
**  exactly two identifiers of other bags as  the first and second entry such
**  as the binary operations bags.  It marks those  subbags and returns.  For
**  example  in {\GAP}  bags for  rational   numbers contain exactly two  bag
**  identifiers for the numerator and the denominator.
**
**  'MarkAllSubBags( <bag> )'
**
**  'MarkAllSubBags'  is  the marking function  for  types whose bags contain
**  only identifier of other bags.  It marks every entry of such a bag.  Note
**  that 'MarkAllSubBags' assumes that  all  identifiers are at offsets  from
**  the    address of the    data area   of  <bag>   that  are divisible   by
**  'sizeof(Bag)'.  Note also that since   it does not do   any harm to  mark
**  something   which  is not    actually a   bag identifier  one   could use
**  'MarkAllSubBags' for all  types  as long as  the identifiers  in the data
**  area are  properly aligned as  explained above.  This  would however slow
**  down 'CollectBags'.  For example  in {\GAP} bags  for lists contain  only
**  bag identifiers for the elements  of the  list or 0   if an entry has  no
**  assigned value.
** */
typedef void (* TNumMarkFuncBags ) ( BagPtr_t bag );

extern void InitMarkFuncBags ( UInt type, TNumMarkFuncBags mark_func );

extern void MarkNoSubBags( BagPtr_t bag );

extern void MarkOneSubBags( BagPtr_t bag );

extern void MarkTwoSubBags( BagPtr_t bag );

extern void MarkThreeSubBags( BagPtr_t bag );

extern void MarkFourSubBags( BagPtr_t bag );

extern void MarkAllSubBags( BagPtr_t bag );

extern void MarkBagWeakly( BagPtr_t bag );

extern  BagPtr_t *                   MptrBags;
extern  BagPtr_t *                   OldBags;
extern  BagPtr_t *                   AllocBags;

extern  BagPtr_t                     MarkedBags;
//  #define IS_VALID_MPTR(ptr) (ptr >= MptrBags && ptr < FreeMptrBags)
#define IS_VALID_BAGPTR(ptr) (ptr >= OldBags && ptr <= AllocBags)

#define MARKED_DEAD(x)  (x)
#define UNMARKED_DEAD(x)  (x)

/* Marked alive adds 1 (least significant bit) to [link] pointer value; unmark clears it  */
#define MARKED_ALIVE(x) ((BagPtr_t)(((Char *)(x))+1))
#define UNMARKED_ALIVE(x) ((BagPtr_t)(((Char *)(x))-1))

/* Marked halfdead adds 2 (2nd least signif bit) to [link] pointer value; unmark clears it  */
#define MARKED_HALFDEAD(x) ((BagPtr_t)(((Char *)(x))+2))
#define UNMARKED_HALFDEAD(x) ((BagPtr_t)(((Char *)(x))-2))

#define IS_MARKED_ALIVE(bag) ((PTR_BAG(bag)[-1]) == MARKED_ALIVE(bag))
#define IS_MARKED_DEAD(bag) ((PTR_BAG(bag)[-1]) == MARKED_DEAD(bag))
#define IS_MARKED_HALFDEAD(bag) ((PTR_BAG(bag)[-1]) == MARKED_HALFDEAD(bag))


// Provide both macro and function definitions for MARK_BAG & switch mechanism

#define MARKBAG_ISFUNC 1
	
#ifdef MARKBAG_ISFUNC
void MARK_BAG( BagPtr_t bag );
#else
#define MARK_BAG(bag)                                                       \
                if ( bag && (((UInt)(bag)) & (sizeof(BagPtr_t)-1)) == 0                 \
                  && (BagPtr_t)MptrBags <= (bag)    && (bag) < (BagPtr_t)OldBags      \
                  && YoungBags < PTR_BAG(bag)  && PTR_BAG(bag) <= AllocBags \
                  && (IS_MARKED_DEAD(bag) || IS_MARKED_HALFDEAD(bag)) ) \
                  {                                                          \
                    PTR_BAG(bag)[-1] = MarkedBags; MarkedBags = (bag);      }
#endif	// MARKBAG_ISFUNC

/****************************************************************************
**
*F
*/

#define IS_WEAK_DEAD_BAG(bag) ( (((UInt)bag & (sizeof(BagPtr_t)-1)) == 0) && \
                                (BagPtr_t)MptrBags <= (bag)    &&          \
                                (bag) < (BagPtr_t)OldBags  &&              \
                                (((UInt)*bag) & (sizeof(Bag)-1)) == 1)


/****************************************************************************
**
*F  InitSweepFuncBags(<type>,<sweep-func>)  . . . . install sweeping function
**
**  'InitSweepFuncBags( <type>, <sweep-func> )'
**
**  'InitSweepFuncBags' installs the function <sweep-func> as sweeping
**  function for bags of type <type>.
**
**  A sweeping function is a function that takes two arguments src and dst of
**  type BagPtr_t *, and  a third, length of type  UInt, and returns nothing. When
**  it  is called, src points to  the start of the data  area of one bag, and
**  dst to another. The function should copy the  data from the source bag to
**  the destination, making any appropriate changes.
**
**  Those functions are applied during  the garbage collection to each marked
**  bag, i.e., bags that are assumed  to be still live  to move them to their
**  new  position. The  intended  use is  for  weak  pointer bags, which must
**  remove references to identifiers of  any half-dead objects.
**
**  If no function  is installed for a Tnum,  then the data is  simply copied
**  unchanged and this is done particularly quickly
*/

typedef void (* TNumSweepFuncBags ) ( BagPtr_t  *src, BagPtr_t *dst, UInt length);

extern void InitSweepFuncBags( UInt tnum, TNumSweepFuncBags sweep_func );


/****************************************************************************
**
*F  InitGlobalBag(<addr>) . . . . . inform Gasman about global bag identifier
**
**  'InitGlobalBag( <addr>, <cookie> )'
**
**  'InitGlobalBag'  informs {\Gasman} that there is  a bag identifier at the
**  address <addr>, which must be of  type '(BagPtr_t\*)'.  {\Gasman} will look at
**  this address for a bag identifier during a garbage collection.
**
**  The application *must* call 'InitGlobalBag' for every global variable and
**  every entry of a  global array that may hold  a bag identifier.  It is no
**  problem  if  such a  variable does not   actually  hold a bag identifier,
**  {\Gasman} will simply ignore it then.
**
**  There is a limit on the number of calls to 'InitGlobalBags', which is 20,000
**  by default.   If the application has  more global variables that may hold
**  bag  identifier, you  have to  compile  {\Gasman} with a  higher value of
**  'NR_GLOBAL_BAGS', i.e., with 'make COPTS=-DNR_GLOBAL_BAGS=<nr>'.
**
**  <cookie> is a C string, which should uniquely identify this global
**  bag from all others.  It is used  in reconstructing  the Workspace
**  after a save and load
*/

extern Int WarnInitGlobalBag;

extern void SortGlobals( UInt byWhat );

extern BagPtr_t * GlobalByCookie( const Char * cookie );

extern void StartRestoringBags( UInt nBags, UInt maxSize);

extern BagPtr_t NextBagRestoring( UInt size,  UInt type);

extern void FinishedRestoringBags( void );



/****************************************************************************
**
*F  InitFreeFuncBag(<type>,<free-func>) . . . . . .  install freeing function
**
**  'InitFreeFuncBag( <type>, <free-func> )'
**
**  'InitFreeFuncBag' installs  the function <free-func>  as freeing function
**  for bags of type <type>.
**
**  A freeing function is  a function that  takes  a single argument of  type
**  'BagPtr_t' and  returns nothing,  i.e., has return  type  'void'.  If  such  a
**  function is installed for a type <type> then it is called for each bag of
**  that type that is about to be deallocated.
**
**  A freeing function must *not* call 'NewBag4', 'ResizeBag', or 'RetypeBag'.
**
**  When such  a function is  called for a bag <bag>,  its subbags  are still
**  accessible.  Note that it it not specified, whether the freeing functions
**  for the subbags of   <bag> (if there   are freeing functions for  bags of
**  their types) are called before or after the freeing function for <bag>.
*/
typedef void (* TNumFreeFuncBags ) ( BagPtr_t bag );

extern void InitFreeFuncBag( UInt type, TNumFreeFuncBags free_func );


/****************************************************************************
**
*F  InitCollectFuncBags(<bfr-func>,<aft-func>) . install collection functions
**
**  'InitCollectFuncBags( <before-func>, <after-func> )'
**
**  'InitCollectFuncBags' installs       the   functions  <before-func>   and
**  <after-func> as collection functions.
**
**  The  <before-func> will be  called   before each garbage collection,  the
**  <after-func>  will be called after each  garbage  collection.  One use of
**  the   <before-func> is to  call 'CHANGED_BAG'  for bags  that change very
**  often, so you do not have to call 'CHANGED_BAG'  for them every time they
**  change.  One use of the <after-func> is to update a pointer for a bag, so
**  you do not have to update that pointer after every operation that might
**  cause a garbage collection.
*/
typedef void (* TNumCollectFuncBags) ( void );

extern void InitCollectFuncBags( TNumCollectFuncBags before_func, TNumCollectFuncBags after_func );


/****************************************************************************
**
*F  CheckMasterPointers() . . . . . . . . . . . . .do some consistency checks
**
**  'CheckMasterPointers()' tests for masterpoinetrs which are not one of the
**  following:
**
**  0                       denoting the end of the free chain
**  NewWeakDeadBagMarker    denoting the relic of a bag that was weakly
**  OldWeakDeadBagMarker    but not strongly linked at the last garbage
**                          collection
**  a pointer into the masterpointer area   a link on the free chain
**  a pointer into the bags area            a real object
**
*/

extern void CheckMasterPointers( void );

/****************************************************************************
**
*F  InitBags(...) . . . . . . . . . . . . . . . . . . . . . initialize Gasman
**
**  InitBags( <alloc-func>, <initial-size>,
**            <stack-func>, <stack-start>, <stack-align>,
**            <cache-size>, <dirty>, <abort-func> )
**
**  'InitBags'  initializes {\Gasman}.  It  must be called from a application
**  using {\Gasman} before any bags can be allocated.
**
**  <alloc-func> must  be the function that  {\Gasman}  uses to  allocate the
**  initial workspace and to extend the workspace.  It must accept two 'long'
**  arguments  <size> and <need>.   <size> is  the amount of  storage that it
**  must allocate, and <need>  indicates  whether {\Gasman} really needs  the
**  storage or only wants it to have a reasonable amount of free storage.
**
**  *Currently   this function must  return    immediately adjacent areas  on
**  subsequent  calls*.  So 'sbrk'  will  work on most  systems, but 'malloc'
**  will not.
**
**  If  <need> is 0,  <alloc-func> must  either  return  the  address of  the
**  extension to indicate success or return  0 if it  cannot or does not want
**  to extend the workspace.  If <need>  is 1, <alloc-func> must again return
**  the address of the extension to indicate success and can either  return 0
**  or abort if it cannot or does not  want  to  extend the workspace.   This
**  choice determines  whether  'NewBag4'  and  'ResizeBag' may  fail.  If  it
**  returns 0, then  'NewBag4' and  'ResizeBag' can fail.  If  it aborts, then
**  'NewBag4' and 'ResizeBag' can never fail (see "NewBag4" and "ResizeBag").
**
**  <size>  may also be   negative if {\Gasman} has   a large amount  of free
**  space, and wants to return  some of it  to the operating system.  In this
**  case <need>   will  always be  0.   <alloc-func>  can either  accept this
**  reduction of  the  workspace and return  a nonzero  value  and return the
**  storage to the operating system, or refuse this reduction and return 0.
**
**  <initial-size> must be the size of  the initial workspace that 'InitBags'
**  should allocate.  This   value is automatically rounded   up to the  next
**  multiple of 1/2 MByte by 'InitBags'.
**
**  <stack-func>  must be   a    function    that  applies  'MARK_BAG'   (see
**  "InitMarkFuncBags") to each possible bag identifier on the application\'s
**  stack, i.e., the stack where the applications local variables  are saved.
**  This should be a function of no  arguments  and return type 'void'.  This
**  function   is  called  from  'CollectBags'  to  mark   all bags  that are
**  accessible from  local variables.   There  is a generic function for this
**  purpose, which is called  when the application passes  0 as <stack-func>,
**  which will work all right on most machines, but *may* fail on some of the
**  more exotic machines.
**
**  If the application passes 0 as value for <stack-func>, <stack-start> must
**  be the start of the stack.   Note that the  start of the  stack is either
**  the bottom or the top of the stack, depending  on whether the stack grows
**  upward  or downward.  A  value that usually works is  the address  of the
**  argument  'argc'   of the  'main' function    of the  application,  i.e.,
**  '(BagPtr_t\*)\&argc'.  If  the application   provides  another   <stack-func>,
**  <stack-start> is ignored.
**
**  If the application passes 0 as value for <stack-func>, <stack-align> must
**  be  the alignment  of items  of  type 'BagPtr_t' on the  stack.   It must be a
**  divisor of 'sizeof(BagPtr_t)'.  The addresses of  all identifiers on the stack
**  must be a multiple  of <stack-align>.  So if  it is 1, identifiers may be
**  anywhere on the stack, and  if it is  'sizeof(BagPtr_t)', identifiers may only
**  be at addresses that are a multiple of 'sizeof(BagPtr_t)'.  This value depends
**  on  the   machine,  the  operating system,   and   the compiler.   If the
**  application provides another <stack-func>, <stack-align> is ignored.
**
**  <cache-size>  informs {\Gasman} whether  the  processor has a usable data
**  cache and how large it is  measured in bytes.   If the application passes
**  0, {\Gasman} assumes that the processor has no data cache or a data cache
**  to small to be   useful.  In this case  the  entire free storage is  made
**  available for allocations after garbage  collections.  If the application
**  passes a  nonzero value, {\Gasman}  assumes that this is  the size of the
**  part of the data cache that should  be used for  the allocation area, and
**  tries  to keep the allocation  area small enough  so that it fits.  For a
**  processor that has separate  data and instruction caches, the application
**  should pass the size of the data cache minus 65536.  For a processor with
**  a  unified cache,  the application  should  pass the size  of the unified
**  cache minus   131072.  The application probably should   not pass a value
**  less than 131072.
**
**  The initialization  flag  <dirty> determines  whether  bags allocated  by
**  'NewBag4' are initialized to contain only 0 or not.   If <dirty> is 0, the
**  bags are  initialized to  contain only 0.    If  <dirty> is  1, the  bags
**  initially contain  random values.  Note that {\Gasman}  will be  a little
**  bit faster if bags need not be initialized.
**
**  <abort-func> should be a function that {\Gasman} should call in case that
**  something goes  wrong, e.g.,     if it  cannot allocation    the  initial
**  workspace.  <abort-func> should be a function of one string argument, and
**  it  might want to display this   message before aborting the application.
**  This function should never return.
*/
typedef BagPtr_t * (* TNumAllocFuncBags) ( Int size, UInt need );

typedef void (* TNumStackFuncBags) ( void );

extern void InitBags( TNumAllocFuncBags alloc_func, UInt initial_size, TNumStackFuncBags stack_func,
					  BagPtr_t * stack_bottom, UInt stack_align, UInt cache_size, UInt dirty,
					  TNumAbortFuncBags abort_func );

/****************************************************************************
**
*F  CallbackForAllBags( <func> ) call a C function on all non-zero mptrs
**
** This calls a   C  function on every    bag, including ones  that  are  not
** reachable from    the root, and    will  be deleted  at the   next garbage
** collection, by simply  walking the masterpointer area. Not terribly safe
**
*/

extern void CallbackForAllBags( void (*func)(BagPtr_t) );


/****************************************************************************
**

*/

#endif					// MEMMGR_H_INCLUDED
