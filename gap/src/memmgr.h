#ifndef MEMMGR_H_INCLUDED
#define MEMMGR_H_INCLUDED

/****************************************************************************
**
*A  memmgr.h                    GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2020, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2020) by the GAP Group (www.gap-system.org).
**
**  This file declares the functions of  Memmgr,  the  GAP  storage  manager.
*/

#include "system_types.h"
#include "system.h"
#include <assert.h>
#include <time.h>

#ifdef COUNT_BAGS
#define DEBUG_GLOBAL_BAGS       True    // when COUNT_BAGS is turned on, turn
#define DEBUG_BAG_MGMT          True    // allow bag tracing by code(s)
#else
#undef DEBUG_GLOBAL_BAGS
#undef DEBUG_BAG_MGMT
//  #define DEBUG_GLOBAL_BAGS       True    // when COUNT_BAGS is turned on, turn
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
    T_VOID          = 0,            // "void"
    T_INT           = 1,            // "integer"
    T_INTPOS        = 2,            // "integer (> 2^28)"
    T_INTNEG        = 3,            // "integer (< -2^28)"
    T_RAT           = 4,            // "rational"
    T_CYC           = 5,            // "cyclotomic"
    T_DOUBLE        = 6,            // "double"
    T_CPLX          = 7,            // "complex"
    T_UNKNOWN       = 8,            // "unknown"
    T_FFE           = 9,            // "finite field element"

    T_PERM16        = 10,           // "permutation16"
    T_PERM32        = 11,           // "permutation32"
    T_WORD          = 12,           // "word"
    T_SWORD         = 13,           // "sparse word"
    T_AGWORD        = 14,           // "agword"

    T_BOOL          = 15,           // "boolean"
    T_CHAR          = 16,           // "character"
    T_FUNCTION      = 17,           // "function"
    T_METHOD        = 18,           // "method"
    T_FUNCINT       = 19,           // "internal function"

    T_MUTABLE       = 20,           // mutable types -- "list"
    T_LIST          = T_MUTABLE,    // "list"
    T_SET           = 21,           // "set"
    T_VECTOR        = 22,           // "vector"
    T_VECFFE        = 23,           // "finite field vector"
    T_BLIST         = 24,           // "boolean list"
    T_STRING        = 25,           // "string"
    T_RANGE         = 26,           // "range"

    T_REC           = 27,           // "record"
    T_MATRIX        = 28,           // "matrix (extended)"
    T_MATFFE        = 29,           // "matffe (extended)"
    T_LISTX         = 30,           // "list (extended)"
    T_DELAY         = 31,           // "delayed expression"

    T_VAR           = 32,           // "variable"
    T_VARAUTO       = 33,           // "autoread variable"
    T_VARASS        = 34,           // "var assignment"
    T_VARMAP        = 35,           // "var map"
    T_LISTELM       = 36,           // "list element"
    T_LISTELML      = 37,           // "list element"
    T_LISTELMS      = 38,           // "sublist"
    T_LISTELMSL     = 39,           // "sublist"
    T_LISTASS       = 40,           // "list assignment"
    T_LISTASSL      = 41,           // "list assignment"
    T_LISTASSS      = 42,           // "list assignment"
    T_LISTASSSL     = 43,           // "list assignment"
    T_RECELM        = 44,           // "record element"
    T_RECASS        = 45,           // "record assignment"
    T_MULTIASS      = 46,           // "multi assignment"

    T_SUM           = 47,           // "+"
    T_DIFF          = 48,           // "-"
    T_PROD          = 49,           // "*"
    T_QUO           = 50,           // "/"
    T_MOD           = 51,           // "mod"
    T_POW           = 52,           // "^"
    T_COMM          = 53,           // "commutator"

    T_NOT           = 54,           // "not"
    T_AND           = 55,           // "and"
    T_OR            = 56,           // "or"
    T_EQ            = 57,           // "="
    T_NE            = 58,           // "<>"
    T_LT            = 59,           // "<"
    T_GE            = 60,           // ">="
    T_LE            = 61,           // "<="
    T_GT            = 62,           // ">"
    T_IN            = 63,           // "in"
    T_CONCAT        = 64,           // "::"

    T_FUNCCALL      = 65,           // "function call"
    T_FUNCINT_CALL  = 66,           // "fast internal function call"
    T_STATSEQ       = 67,           // "statement sequence"
    T_IF            = 68,           // "if statement"
    T_FOR           = 69,           // "for loop"
    T_WHILE         = 70,           // "while loop"
    T_REPEAT        = 71,           // "repeat loop"
    T_RETURN        = 72,           // "return statement"

    T_MAKEPERM      = 73,           // "var permutation"
    T_MAKEFUNC      = 74,           // "var function"
    T_MAKEMETH      = 75,           // "var method"
    T_MAKELIST      = 76,           // "var list"
    T_MAKESTRING    = 77,           // "var string"
    T_MAKERANGE     = 78,           // "var range"
    T_MAKEREC       = 79,           // "var record"

    T_MAKETAB       = 80,           // "var hash"
    T_MAKELET       = 81,           // "let statement"
    T_CYCLE         = 82,           // "cycle"
    T_FF            = 83,           // "finite field"
    T_AGEN          = 84,           // "abstract generator"
    T_AGGRP         = 85,           // "aggroup"
    T_PCPRES        = 86,           // "polycyclic presentation"
    T_AGEXP         = 87,           // "ag exponent vector"
    T_AGLIST        = 88,           // "ag exponent/generator"
    T_RECNAM        = 89,           // "record name"
    T_NAMESPACE     = 90,           // "namespace"
    T_EXEC          = 91,           // "stack frame"
    T_IS            = 92,           // "_is"
    T_FREEBAG       = 93,           // "free bag"

    T_ILLEGAL       = 94,           // "ILLEGAL" bag
    T_MAGIC_254     = 254,          // "magic" value for reserved for memory manager
    T_RESIZE_FREE   = 255           // flags free areas in a re-sized (smaller) bag
} ObjType;


/****************************************************************************
**
*V  NameType . . . . . . . . . . . . . . . . . . . . printable name of a type
*V  SizeType . . . . . . . . . . size of handle and data area of a bag, local
*F  NrHandles( <type>, <size> ) . . . . . . . . .  number of handles of a bag
*/

extern  char       *NameType[];

typedef struct {
    Int         handles;
    Int         data;
    char        name[4];
} SizeTypeT;

extern  SizeTypeT   SizeType [];

extern  Int         NrHandles ( unsigned int type, UInt size );


//  Flags for bags Define flags with appropriate bit values such that the least
//  significant byte is unaffected... the least significant byte is reserved for
//  bag type in the flag-type word; also see GET_FLAG_BAG, SET_FLAG_BAG,
//  CLEAR_FLAG_BAG

enum BagFlags {
    BF_COPY             = 0x00000100L,
    BF_PRINT            = 0x00000200L,
    BF_UNUSED_FLAG      = 0x00000400L,          // spare flag
    BF_DELAY            = 0x00000800L,
    BF_DELAY_EV         = 0x00001000L,
    BF_DELAY_EV0        = 0x00002000L,
    BF_NO_WARN_UNDEFINED= 0x00004000L,
    BF_NO_COPY          = 0x00008000L,
    BF_METHCALL         = 0x00010000L,
    BF_UNEVAL_ARGS      = 0x00020000L,
    BF_NS_GLOBAL        = 0x00040000L,
    BF_VAR_AUTOEVAL     = 0x00080000L,
    BF_VISITED          = 0x00100000L, /* -- for recursive traversals with cycle prevention */
    BF_PROTECT_VAR      = 0x00200000L,
    BF_ENVIRONMENT      = 0x00400000L, /* for T_FUNCTION and T_MAKEFUNC/T_MAKEMETH this*/
                                       /* flag is set if function depends from environment;*/
    BF_ENV_VAR          = 0x00400000L, /* for T_VAR only, this flag means that some        */
                                       /* subroutine has reference to this variable;       */
    BF_ON_CALLSTACK     = 0x00400000L, /* for T_EXEC only, flag means that T_EXEC is on the*/
                                       /* call stack .                                     */
    BF_INHERITED_CALL   = 0x00400000L, /* for T_FUNCCALL means that this is inherited   */
                                       /* method call, i.e. from Inherited() function.  */
    BF_ON_EVAL_STACK    = 0x00800000L, /* for all bags, FunTop() uses this flag to mark bags
                                        * on the eval stack to correctly highlight current
                                        * statement. */
};

#define BF_ALL_FLAGS    0xFFFFFF00L        /* Allows to mask off all flags */
#define TYPE_BIT_MASK   0x000000FFL        /* Mask off least significant byte -- bag type */


// This variable is an easy (and dirty) way to implement full copying of
// objects, needed to correctly implement spiral_delay_ev.c:Eval()

extern  int         DoFullCopy;

/* goes in variables*/
#define IS_FULL_MUTABLE(hd) (   (T_MUTABLE <= GET_TYPE_BAG(hd) && GET_TYPE_BAG(hd) < T_VAR) \
                             || (T_VARAUTO < GET_TYPE_BAG(hd) && GET_TYPE_BAG(hd) < T_RECNAM))

#define IS_MUTABLE(hd)   ((!DoFullCopy && (T_MUTABLE <= GET_TYPE_BAG(hd) && GET_TYPE_BAG(hd) < T_VAR)) \
                          || (DoFullCopy && (IS_FULL_MUTABLE(hd))))

void RecursiveClearFlag(Obj hd, int flag);

void RecursiveClearFlagMutable(Obj hd, int flag);

void RecursiveClearFlagFullMutable(Obj hd, int flag);

/* Takes variable number(=howMany) of flags to clear, all flags should be int's */
void RecursiveClearFlagsFullMutable(Obj hd, int howMany, ...);


/****************************************************************************
**
*V  SIZE_HD . . . . . . . . . . . . . . . . . . . . . . . .  size of a handle
*/

#define SIZE_HD         ((Int)sizeof(BagPtr_t))

#define SET_BAG(bag, index, value)      PTR_BAG(bag)[index] = (value)

#define NTYPES 256

/******************************************************************************
**  
**  Bag layout is shown below.  Bags can be of any arbitrary size.  NOTE: The
**  diagram below assumes 64 bit pointers (and unsigned long ints).  For 32 bit
**  pointers/unsigned ints the structure uses the same number of words (though
**  would only be half as big) since flags & type will fit in 1 word, as of
**  course does size and the pointer values.
** 
**     |  0|  1|  2|  3|  4|  5|  6|  7|
**     +-------------------------------+
**     |    Flags (BagFlags)       |Typ|  Typ == Bag [Object] type (ObjType)
**     +-------------------------------+
**     |            Bag Size           |
**     +-------------------------------+
**     |   Copy Pointer                |
**     +-------------------------------+
**     | Link Pointer (to master ptr)  |
**     +-------------------------------+
**     | Start of bag data             |
**    ...                             ...
**     |                               |
**     +-------------------------------+
** 
**  Bag data is the number of bytes (size) plus padding to round up to a
**  multiple of sizeof(BagPtr_t).
** 
**  The master pointer for any bag (returned from NewBag4) is treated as a
**  handle, the master pointer points to the start of the bag data.
*/ 

// Define a struct to give the layout of the bags...
// word 0:        Flags and bag type
// word 1:        Size of bag; size of data only, header size is NOT included
// word 2:        copy pointer
// word 3:        link pointer
// word 4:        Start of data (bag handles actually point here)

typedef Bag BagPtr_t;

typedef struct {
    UInt        bagFlagsType;
    UInt        bagSize;
    BagPtr_t    bagCopyPtr;
    BagPtr_t    bagLinkPtr;
#ifdef DEBUG_BAG_MGMT
    UInt        bagCodeValue;            /* coded value to track bag origin (debug purposes) */
#endif
    UInt        *bagData;                /* data is arbitrary, length is defined by size */
} BagStruct_t;

/*
 *  bagCodeMarker is a value which can be set prior to entering a section of
 *  spiral code and causes all bags created or resized during that code
 *  section or operation to be tagged with "bagCodeMarker".  This is intended
 *  for debug purposes to track where the number of bags (& memory) grows
 *  significantly; then hopefully, allowing the possibility to destroy these
 *  data when the code section or operation is complete.
 */

UInt        bagCodeMarker;

/* 
**  With the definition of BagStruct_t we don't assume aspecific order/place for
**  elements of the bag header; these macros are defined to use the structure
**  (casting the incoming pointer as needed) to get at the required element.
**  The macros are intended for internal memory management ONLY!!
**  These macros all take a pointer to the header of the bag (*NOT* a bag handle).
*/

#define HEADER_SIZE ((sizeof(BagStruct_t) - sizeof(UInt *)) / sizeof(UInt *))

#define DATA_PTR(ptr)         ((UInt **)&(((BagStruct_t *)(ptr))->bagData))
#define COPY_PTR(ptr)         (((BagStruct_t *)(ptr))->bagCopyPtr)
#define SIZE_PTR(ptr)         (((BagStruct_t *)(ptr))->bagSize)
#define FLAGS_TYPE_PTR(ptr)   (((BagStruct_t *)(ptr))->bagFlagsType)

#define GET_COPY_PTR(ptr)          (COPY_PTR(ptr))
#define GET_SIZE_PTR(ptr)          (SIZE_PTR(ptr))
#define GET_TYPE_PTR(ptr)          (FLAGS_TYPE_PTR(ptr) & TYPE_BIT_MASK)
#define TEST_FLAG_PTR(ptr,flag)     (FLAGS_TYPE_PTR(ptr) & (flag))

#define DEBUG_POINTERS    True

BagPtr_t GET_LINK_PTR(BagPtr_t ptr);
void     SET_LINK_PTR(BagPtr_t ptr, BagPtr_t val);

#define SET_COPY_PTR(ptr,val)       (COPY_PTR(ptr) = (BagPtr_t)val)
#define SET_SIZE_PTR(ptr,val)       (SIZE_PTR(ptr) = (UInt)val)
#define SET_TYPE_PTR(ptr,val)       ((FLAGS_TYPE_PTR(ptr)) = ((FLAGS_TYPE_PTR(ptr)) & BF_ALL_FLAGS) | val)
#define SET_FLAG_PTR(ptr,val)       ((FLAGS_TYPE_PTR(ptr)) |= (val))
#define CLEAR_FLAG_PTR(ptr,val)     ((FLAGS_TYPE_PTR(ptr)) &= ~(val))
#define BLANK_FLAGS_PTR(ptr)        (FLAGS_TYPE_PTR(ptr) &= TYPE_BIT_MASK)


/*
**  A Memory Arena is a block of memory to manage the bags used by GAP.  Each
**  arena holds a set of master pointers (linked to active bags and a chain of
**  free pointers), bags (both live and dead), and an allocation pool where new
**  bags will be created.  The pool is managed by Memmgr, GC is invoked to clear
**  out the dead bags as necessary.  If the free space drops below a minimum
**  threshold than another arena is allocated.  Define a struct to handle the
**  pointers and values used to manage an 'arena' of memory
*/

typedef struct {
    BagPtr_t *          BagHandleStart;         // Start of bag pointers 
    BagPtr_t *          OldBagStart;            // Start of active bags
    BagPtr_t *          YoungBagStart;          // Bags allocated since last GC
    BagPtr_t *          AllocBagStart;          // Free area from which to allocate
    BagPtr_t *          StopBags;               // End of allocation area
    BagPtr_t *          EndBags;                // End of memory arena (normally == StopBags)
    BagPtr_t *          FreeHandleChain;        // Head of the free chain of bag handle in arena
    BagPtr_t *          MarkedBagChain;         // Head of 'marked bags' chain
    UInt                nrMarkedBags;           // Number of marked bags this arena (# set by MARK_BAG()) 
    UInt                SizeArena;              // Size of memory arena
    float	            FreeRatio;		        // Free ratio required in the arena
    char                ArenaNumber;            // arena number: 0 ... n
    char                ArenaFullFlag;          // Arena full, GC cannot free enough to keep using
    char                ActiveArenaFlag;        // Active memory arena -- allocate new from here
    char                SweepNeeded;            // Set if sweep needed to resolve link pointers
}   ArenaBag_t;


/*
**  Simple structure to hold time information.  Keep total time program is
**  running plus time spent in performing certain core activities (primary one
**  of interest is Garbage Collection).
*/

typedef struct {
	clock_t     gap_start;				// GAP program start time
	clock_t     gap_end;				// end time
	clock_t     gc_in;					// time on entry to GC
	clock_t     gc_out;					// time on exit form GC
	clock_t     gc_cumulative;			// cumulative time spent in GC
	Int         nrGCRuns;				// number of times GC involed
}   TimeAnalyze_t;

extern  TimeAnalyze_t   GapRunTime;

extern  void    PrintRuntimeStats ( TimeAnalyze_t *ptim );
extern  void    InitMemMgrFuncs( void );

 
/****************************************************************************
**
*F  CollectGarb() . . . . . . . . . . . . . . .  perform a garbage collection
*F  NewBag(<type>, <size>)  . . . . . . . . . . . . . . . .  create a new bag
*F  Retype(<bag>, <newType>)  . . . . . . . . . . .  change the type of a bag
*F  Resize(<bag>, <newSize> )  . . . . . . . . .  .  change the size of a bag
*F  InitGasman(<argc>, <argv>, <stackBase>)  .  initialize dynamic memory manager package
**
*/

extern  void        CollectGarb ( void );
extern  BagPtr_t    NewBag ( UInt type,  UInt size );
extern  void        Retype ( BagPtr_t hdBag, UInt newType );
extern  void        Resize ( BagPtr_t hdBag, UInt newSize );
extern  void        InitGasman ( int argc, char **argv, int* stackBase);


/****************************************************************************
**
*V  NrAllBags . . . . . . . . . . . . . . . . .  number of all bags allocated
*V  SizeAllBags . . . . . . . . . . . . . .  total size of all bags allocated
*V  SizeLiveBags  . . . . . . .  total size of bags that survived the last gc
*V  SizeAllArenas . . . . . . . . . . . . . . total size of all Memory Arenas
**
**  'NrAllBags' is the number of bags allocated since Gasman was initialized.
**  It is incremented for each 'NewBag4' call.
**
**  'SizeAllBags'  is the  total  size  of bags   allocated since Gasman  was
**  initialized.  It is incremented for each 'NewBag4' call.
**
**  'SizeLiveBags' is the total size of all the live bags after the last garbage
**  collection.  This value is used in the information messages.
**
**  'SizeAllArenas' is the total memory allocated to all Memory Arenas.  This
**  value is used in the information messages.
**
*/

extern  UInt        NrAllBags;
extern  UInt        SizeAllBags;
extern  UInt        SizeLiveBags;
extern  UInt        SizeAllArenas;


/****************************************************************************
**
*V  InfoBags[<type>]  . . . . . . . . . . . . . . . . .  information for bags
**
**  'InfoBags[]' is an array of structures containing information for bags of
**  each type <type>.
**
**  '.name' is the name of the type <type>.  Note that it is the responsibility
**  of the application using {\Gasman} to enter this name.
**
**  '.nrLive' is the number of bags of type <type> that are currently live.
**
**  '.nrAll' is the total number of all bags of <type> that have been allocated.
**
**  '.sizeLive' is the sum of the sizes of the bags of type <type> that are
**  currently live.
**
**  '.sizeAll' is the sum of the sizes of all bags of type <type> that have been
**  allocated.
**
**  '.nrDead' is the number of dead bag removed during a GC.
**
**  '.nrRemnant' is the number of resize bag remnants removed during a GC.
**
**  This information is only kept if {\Gasman} is compiled with the option
**  '-DCOUNT_BAGS', e.g., with 'make <target> COPTS=-DCOUNT_BAGS'.
*/

typedef struct  {
    const Char *            name;
    UInt                    nrLive;
    UInt                    nrAll;
    UInt                    nrDead;
    UInt                    nrRemnant;
    UInt                    sizeLive;
    UInt                    sizeAll;
} TNumInfoBags;

extern  TNumInfoBags InfoBags [ NTYPES ];

extern  const char *TNAM_BAG(BagPtr_t bag);


/****************************************************************************
**
*V  GlobalBags  . . . . . . . . . . . . . . . . . . . . . list of global bags
*/

#ifndef NR_GLOBAL_BAGS
#define NR_GLOBAL_BAGS  20000L
#endif


typedef struct {
    BagPtr_t *                   addr [NR_GLOBAL_BAGS];
//    BagPtr_t                     lastValue [NR_GLOBAL_BAGS];
    const Char *            cookie [NR_GLOBAL_BAGS];
    UInt                    nr;
} TNumGlobalBags;

extern  TNumGlobalBags GlobalBags;
extern  UInt        GlobalSortingStatus;


/****************************************************************************
**
*F  InitGlobalBag(<addr>) . . . . . inform Gasman about global bag identifier
**
**  'InitGlobalBag( <addr>, <cookie> )'
**
**  'InitGlobalBag' informs {\Memmgr} that there is a bag identifier at the
**  address <addr>, which must be of type '(BagPtr_t\*)'.  {\Memmgr} will look
**  at this address for a bag identifier during a garbage collection.
**
**  The application *must* call 'InitGlobalBag' for every global variable and
**  every entry of a global array that may hold a bag identifier.  It is not a
**  problem if such a variable does not actually hold a bag identifier,
**  {\Memmgr} will simply ignore it.
**
**  There is a limit on the number of calls to 'InitGlobalBags', which is 20,000
**  by default.  If the application has more global variables that may hold bag
**  identifier, you have to compile {\Memmgr} with a higher value of
**  'NR_GLOBAL_BAGS', i.e., with 'make COPTS=-DNR_GLOBAL_BAGS=<nr>'.
**
**  <cookie> is a C string, which should uniquely identify this global bag from
**  all others.  It is used in reconstructing the Workspace after a save and
**  load
*/

extern  void        InitGlobalBag ( BagPtr_t *addr, const Char *cookie );


//  After a normal GC we want to have a minimum of GASMAN_FREE_RATIO of the
//  active Memory Arena available for use.  If less than this is available a new
//  Arena will be created.  If a formerly full arena drops below
//  GASMAN_OCCUPIED_RATIO then make it available for future bag allocations

#define GASMAN_FREE_RATIO      0.20  /* 20% of workspace will be kept free  */
#define GASMAN_OCCUPIED_RATIO  0.50  /* < 50% full, reopen */


/****************************************************************************
**
*T  BagPtr_t  . . . . . . . . . . . . . . . . type of the identifier of a bag
*F  IS_BAG(<bag>) . . . . . .  test whether a bag identifier identifies a bag
*F  GET_FLAGS_BAG(<bag>)  . . . . . . . . . . get the flags associated to bag
*F  GET_FLAG_BAG(<bag>, <val>)  . . .  test if a specified flag is set on bag
*F  BLANK_FLAGS_BAG(<bag>)  . . . . . . . . clear all flags associated to bag
*F  CLEAR_FLAG_BAG(<bag>, <val>)  . . . . . . . clear a specified flag on bag 
*F  SET_FLAG_BAG(<bag>, <val>)  . . . . . . . . . set a specified flag on bag 
*F  GET_COPY_BAG(<bag>) . . . . . . . . . . . . get the copy pointer from bag
*F  SET_COPY_BAG(<bag>, <val>)  . . . . . . . .  set the copy pointer for bag
*F  GET_LINK_BAG(<bag>) . . . . . . . . . . . . get the link pointer from bag
*F  SET_LINK_BAG(<bag>, <val>)  . . . . . . . .  set the link pointer for bag
*F  GET_SIZE_BAG(<bag>) . . . . . . . . . . . . . . . . . get the size of bag
*F  SET_SIZE_BAG(<bag>, <val>)  . . . . . . . . . . . . . set the zize of bag
*F  GET_TYPE_BAG(<bag>) . . . . . . . . . . . . . . . . . get the type of bag
*F  SET_TYPE_BAG(<bag>, <val>)  . . . . . . . . . . . . . set the type of bag
*F  PTR_BAG(<bag>)  . . . . . . . . . . . . . . . [get data] pointer to a bag
*F  SET_PTR_BAG(<bag>, <dst>) . . . . . . . . . set the [data] pointer of bag
**
**  Each bag is identified by its *bag identifier*.  That is each bag has a bag
**  identifier and no two live bags have the same identifier.  'BagPtr_t' is the
**  type of bag identifiers.
**
**  'NewBag4' returns the identifier of newly allocated bags; the application
**  passes this identifier to every {\Memmgr} function to tell it the bag on
**  which it should operate (see "NewBag4", "GET_TYPE_BAG", "GET_SIZE_BAG",
**  "PTR_BAG", "RetypeBag", and "ResizeBag").
**
**  Note that the identifier of a bag is different from the address of the data
**  area of the bag.  This address may change during a garbage collection while
**  the identifier of a bag never changes.
**
**  Bags that contain references to other bags must always contain the
**  identifiers of these other bags, never the addresses of the data areas of
**  the bags.
**
**  'PTR_BAG( <bag> )'
**
**  'PTR_BAG' returns the address of the data area of the bag with identifier
**  <bag>.  The application uses this pointer to read or write data from the
**  bag.
**
**  Note that the address of the data area of a bag may change during a garbage
**  collection.  That is the value returned by 'PTR_BAG' may differ between two
**  calls, if 'NewBag4', 'ResizeBag', 'CollectBags', is called in between.
**  Thus, applications *must not* keep any pointers to or into the data area of
**  any bag over calls to functions that may cause a garbage collection.
**
**  'GET_TYPE_BAG' returns the type of the the bag with the identifier <bag>.
**
**  Each bag has a certain type that identifies its structure.  The type is
**  defined by 'ObjType'.  Bag (object) type is specified when creating a bag
**  (see 'NewBag4' or 'ReTypeBag').
**
**  'GET_SIZE_BAG' returns the size of the bag with the identifier <bag> in
**  bytes.  The size of a bag is measured in bytes.  The application specifies
**  the size of a bag when it allocates it with 'NewBag4' and may later change
**  it with 'ResizeBag' (see "NewBag4" and "ResizeBag").
*/

extern  UInt            IS_BAG ( BagPtr_t bag );
extern  UInt            GET_FLAGS_BAG ( BagPtr_t bag );
extern  UInt            GET_FLAG_BAG ( BagPtr_t bag, UInt val );
extern  void            BLANK_FLAGS_BAG ( BagPtr_t bag );

extern  void            CLEAR_FLAG_BAG ( BagPtr_t bag, UInt val );
extern  void            SET_FLAG_BAG ( BagPtr_t bag, UInt val );
extern  BagPtr_t        GET_COPY_BAG ( BagPtr_t bag );
extern  void            SET_COPY_BAG ( BagPtr_t bag, BagPtr_t val );

extern  BagPtr_t        GET_LINK_BAG ( BagPtr_t bag );
extern  void            SET_LINK_BAG ( BagPtr_t bag, BagPtr_t val );
extern  UInt            GET_SIZE_BAG ( BagPtr_t bag );
extern  void            SET_SIZE_BAG ( BagPtr_t bag, UInt val );

extern  ObjType         GET_TYPE_BAG ( BagPtr_t bag );
extern  void            SET_TYPE_BAG ( BagPtr_t bag, UInt val );
extern  BagPtr_t       *PTR_BAG ( const BagPtr_t bag );
extern  void            SET_PTR_BAG ( BagPtr_t bag, BagPtr_t *dst );

#define IS_INTOBJ(o)    ((Int)(o) & 0x01)
#define TYPENAME(obj)   NameType[GET_TYPE_BAG(obj)]

//  determine the number of words required by a bag
#define WORDS_BAG(size) (((size) + (sizeof(BagPtr_t)-1)) / sizeof(BagPtr_t))




/****************************************************************************
**
*F  NewBag4(<type>,<size>) . . . . . . . . . . . . . . . . allocate a new bag
*F  RetypeBag(<bag>,<new>)  . . . . . . . . . . . .  change the type of a bag
*F  ResizeBag(<bag>,<new>)  . . . . . . . . . . . .  change the size of a bag
*F  CollectBags(<size>, <spec_arena>, <caller>) . . . . . . collect dead bags
**
*/

extern  BagPtr_t    NewBag4( UInt type, UInt size );
extern  void        RetypeBag( BagPtr_t bag, UInt new_type );
extern  UInt        ResizeBag( BagPtr_t bag, UInt new_size );
extern  UInt        CollectBags( UInt size, int spec_arena, char *caller_id );


/****************************************************************************
**
*F  InitMarkFuncBags(<type>,<mark-func>)  . . . . .  install marking function
*F  MARK_BAG(<bag>) . . . . . . . . . . . . . . . . . . .  mark a bag as live
**
**  'InitMarkFuncBags( <type>, <mark-func> )'
**
**  'InitMarkFuncBags' installs the function <mark-func>  as marking function
**  for bags  of  type <type>.   The  application  *must* install  a  marking
**  function for a  type before it allocates  any  bag  of  that type.  It is
**  probably best to install all marking functions before allocating any bag.
**
**  A marking function is a function that takes a single argument of type
**  'BagPtr_t' and returns nothing, i.e., has return type 'void'.  Such a
**  function must call 'MARK_BAG' for each bag identifier that appears in the
**  bag (see below).
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
**  'MARK_BAG' tests if <bag> is a valid identifier of a bag.  All bags (old &
**  young) should be checked as we always do a full garbage collection when
**  called.  If it is not, then 'MARK_BAG' does nothing, so there is no harm
**  in calling 'MARK_BAG' for something that is not actually a bag identifier.
**
*/

typedef void (* TNumMarkFuncBags ) ( BagPtr_t bag );

extern  void        InitMarkFuncBags ( UInt type, TNumMarkFuncBags mark_func );
extern  void        MARK_BAG ( BagPtr_t bag );


// MARKED_ALIVE sets the least significant bit of the link pointer value, unmark clears it
#define MARKED_ALIVE(x)      ( (BagPtr_t) ( ((UInt)(x)) |  ((UInt)0x01) ) )
#define UNMARKED_ALIVE(x)    ( (BagPtr_t) ( ((UInt)(x)) & ~((UInt)0x01) ) ) 

#define MARKED_DEAD(x)       (x)            // No change is made to the link pointer for dead bags
#define UNMARKED_DEAD(x)     (x)



/****************************************************************************
**
*F  InitBags(...) . . . . . . . . . . . . . . . . . . . . . initialize Memmgr
**
**  InitBags( <alloc-func>, <initial-size>,
**            <stack-start>, <stack-align>,
**            <dirty>, <abort-func> )
**
**  'InitBags'  initializes {\Memmgr}.  It  must be called from an application
**  using {\Memmgr} before any bags can be allocated.
**
**  <alloc-func> is the function that {\Memmgr} uses to allocate the initial
**  workspace and to extend the workspace.  <alloc-func> must accept one
**  argument: <size>.  <size> is the amount of storage that it must allocate.
**
**  <initial-size> is the size of  the initial workspace that 'InitBags'
**  should allocate.  
**
**  <stack-start> is the start of the stack.  Note that the start of the stack
**  is either the bottom or the top of the stack, depending on whether the stack
**  grows upward or downward.  A value that usually works is the address of the
**  argument 'argc' of the 'main' function of the application, i.e.,
**  '(BagPtr_t\*)\&argc'.
**
**  <stack-align> is the alignment of items of type 'BagPtr_t' on the stack.  It
**  must be a divisor of 'sizeof(BagPtr_t)'.  The addresses of all identifiers
**  on the stack must be a multiple of <stack-align>.  So if it is 1,
**  identifiers may be anywhere on the stack, and if it is 'sizeof(BagPtr_t)',
**  identifiers may only be at addresses that are a multiple of
**  'sizeof(BagPtr_t)'.  This value depends on the machine, the operating
**  system, and the compiler.
**
**  The initialization flag <dirty> determines whether the free memory pool
**  after a GC is zeroed (i.e., cleared), so that bags allocated by 'NewBag4'
**  are initialized to contain only 0.  If <dirty> is 0, then the free pool
**  memory is initialized to contain only 0.  If <dirty> is 1, the free pool
**  memory is left unchanged.
**
**  <abort-func> is a function that {\Memmgr} will call in case something goes
**  wrong, e.g., it cannot allocate the initial workspace.  <abort-func> accepts
**  one string message argument, and will display this message before aborting
**  the application.
*/

typedef BagPtr_t * (* TNumAllocFuncBags) ( Int size );

extern  void        InitBags( TNumAllocFuncBags alloc_func, UInt initial_size,
							  BagPtr_t * stack_bottom, UInt stack_align, UInt dirty,
							  TNumAbortFuncBags abort_func );


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
**  NOTE: This is an unused feature, sweeping (i.e., consolidating live bags at
**  the beginning of the storage space) is handled generically by copying, see
**  'SweepArenaBags'.
*/

typedef void (* TNumSweepFuncBags ) ( BagPtr_t  *src, BagPtr_t *dst, UInt length);

extern  void        InitSweepFuncBags( UInt tnum, TNumSweepFuncBags sweep_func );


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
**
**  NOTE: This is an unused feature, bags are really either Alive or Dead; dead
**  bags are swept away during GC.
*/

typedef void (* TNumFreeFuncBags ) ( BagPtr_t bag );

extern  void        InitFreeFuncBag( UInt type, TNumFreeFuncBags free_func );


/****************************************************************************
**
*F  InitCollectFuncBags(<bfr-func>,<aft-func>) . install collection functions
**
**  'InitCollectFuncBags( <before-func>, <after-func> )'
**
**  'InitCollectFuncBags' installs the functions <before-func> and <after-func>
**  as collection functions.
**
**  The <before-func> will be called before each garbage collection, the
**  <after-func> will be called after each garbage collection.  One use of the
**  <after-func> is to update a pointer for a bag, so you do not have to update
**  that pointer after every operation that might cause a garbage collection.
**
**  NOTE: This is an unused feature, 'CollectBags' coordinates all the necessary
**  actions during GC.
*/

typedef void (* TNumCollectFuncBags) ( void );

extern  void        InitCollectFuncBags( TNumCollectFuncBags before_func, TNumCollectFuncBags after_func );



#endif                    // MEMMGR_H_INCLUDED
