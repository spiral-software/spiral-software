/****************************************************************************
**
*A  memmgr.c                    GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file contains the functions of Memory Manager, memmgr, (formerly called
**  Gasman), the GAP storage manager.
**
**  Memmgr is the GAP storage manager.  That means that the other parts of GAP
**  request memory areas from Memmgr.  These are then filled with information.
**  Memmgr cares about the allocation of those memory areas and the collection
**  of unused areas.  Thus these operations are transparent to the rest of GAP,
**  enabling the programmer to concentrate on his algorithm instead of caring
**  about memory allocation and deallocation.
**
**  The basic thing is the bag.  This is simply a continous area of memory
**  containing any information, including references to other bags.  If a bag
**  contains references to other bags these references are collected at the
**  beginning of the bag.  A bag together with all bags it references direct or
**  indirect, is called an object.  New bags are created with 'NewBag'.
**
**  You create a bag using 'NewBag', which returns a unique pointer identifying
**  the bag, this pointer is called a handle and is of type 'BagPtr_t'.  You
**  specify this handle as first argument to every storage manager function as
**  first argument.  Variables containing handles should begin with 'hd' to make
**  the code more readable.
**
**  Every bag belongs to a certain type, specifying what information this bag
**  contains and how bags of this type are evaluated.  You specify the type when
**  creating the bag, you can retrieve it using the function 'GET_TYPE_BAG'.
**
**  Every bag has a size, which is the size of the memory area in bytes.  You
**  specify this size when you create the bag, you can change it later using
**  'Resize' and retrieve it later using the function 'GET_SIZE_BAG'.  By
**  convention, the bag size *excludes* the header area reserved for Memmgr.  
*/

#include <stdio.h>
#include <stdlib.h>
#include        "system.h"              /* system dependent functions      */
#include        "scanner.h"             /* Pr()                            */
#include        "memmgr.h"              /* declaration part of the package */
#include        "eval.h"
#include        "integer.h"
#include        "GapUtils.h"
#include        <assert.h>

#include <time.h>


/****************************************************************************
**
*F  CollectGarb() . . . . . . . . . . . . . . .  perform a garbage collection
**
**  'CollectGarb' performs a garbage collection.  It provides an external means
**  to call 'CollectBags'; see 'CollectBags' for details of what a garbage
**  collection entails.
*/

void            CollectGarb (void)
{
    CollectBags(0, -1, "GASMAN()");
}


/****************************************************************************
**
** This version of GASMAN / GAP4 is ported for use in SPIRAL (GAP3 branch) 
** Yevgen S. Voronenko (yevgen@drexel.edu)                                 
**
** Memmgr has been extensively reworked, primarily to simplify and steamline the
** memory management tasks required in GAP.  In many cases there existed some
** provision for features either not implemented or intentionally disabled
** (e.g., partial GC); these have been intentionally removed.
**
** Memmgr always performs a full garbage collection (GC) when invoked.  As part
** of the latest overhaul we've introduced the concept of multiple memory
** arenas.  A memory arena is simply a chunk of memory used by GAP.  When memory
** runs low a GC run is performed.  If not enough memory is recovered (see
** GASMAN_FREE_RATIO) a new arena is allocated.  In general when an arena's
** occupancy exceeds the threshold it is flagged as full and a new arena is
** allocated to become the working area for most bag creation.
**
** GASMAN_FREE_RATIO is defined as the fraction of space theoretically allocable
** to bags (i.e., EndBags - OldBagStart within any arena) which must be free
** after a full GC; otherwise, try to allocate a new memory block to continue.
** If more memory cannot be allocated print amessage and the program exits.
**
*/


/****************************************************************************
**
*F  WORDS_BAG( <size> ) . . . . . . . . . . words used by a bag of given size
**
**  The structure of a bag is a follows{\:}
**
**    <identifier, aka bag handle>
**      __/
**     /
**    V
**    +---------+
**    |<masterp>|
**    +---------+
**          \____________
**                       \
**                        V
**    +---------+---------+--------------------------------------------+----+
**    |    Bag header     |         .         .         .         .    | pad|
**    | (see BagStruct_t) |         .         .         .         .    |    |
**    +---------+---------+--------------------------------------------+----+
**
**  A bag consists of a masterpointer, and a body.
**
**  The *masterpointer* is a pointer to the data area of the bag.  During a
**  garbage collection the masterpointer is the only active pointer to the data
**  area of the bag, because of the rule that no pointers to or into the data
**  area of a bag may be remembered over calls to functions that may cause a
**  garbage collection.  It is the job of the garbage collection to update the
**  masterpointer of a bag when it moves the bag.
**
**  The *identifier* of the bag is a pointer to (the address of) the
**  masterpointer of the bag.  Thus 'PTR_BAG(<bag>)' is simply '\*<bag>' plus a
**  cast.
**
**  The *body* of a bag consists of: Bag Header (flags, type, size, link
**  pointers), the data area, and padding (if necessary).  The bag [header]
**  structure is defined by the type BagStruct_t.  See the definition of
**  BagStruct_t for full details of the header layout.
**
**  'GET_SIZE_BAG' simply extracts the size; by convention bag size *excludes*
**  the header size and is simply the size of the data portion.  'GET_TYPE_BAG'
**  extracts the bag type.
**
**  The *link word* usually contains the identifier of the bag, i.e., a pointer
**  to the masterpointer of the bag.  Thus the garbage collection can find the
**  masterpointer of a bag through the link word if it knows the address of the
**  data area of the bag.  The link word is also used by memory management to
**  keep bags on a linked list (see "MarkedBagChain").
**
**  The *data area* of a bag is the area that contains the data stored by the
**  application in this bag.
**
**  The *padding* consists of up to 'sizeof(BagPtr_t)-1' bytes and pads the body
**  so that the size of a body is always a multiple of 'sizeof(BagPtr_t)'.  This
**  ensures that bags are always aligned.  The macro 'WORDS_BAG(<size>)' returns
**  the number of words occupied by the data area and padding of a bag of size
**  <size>.
**
**  A body in the workspace whose type contains the value 255 (T_RESIZE_FREE) is
**  the remainder of a 'ResizeBag'.  That is it consists either of the unused
**  words after a bag has been shrunk or of the old body of the bag after the
**  contents of the body have been copied elsewhere for an extension.  NOTE:
**  Such a body has no link word, because such a remainder does not correspond
**  to a bag (see "Implementation of ResizeBag"); the next garbage collection
**  will consolidate and eliminate these to the free pool area.
**
**  A link pointer with a value congruent to 1 mod 4 only occurs during garbage
**  collection and is so marked to indicate it is a live bag.  This mark will be
**  removed when the bag is copied / compacted to the Old Bags area during GC.
** 
*/

#ifndef C_PLUS_PLUS_BAGS
#define SIZE_MPTR_BAGS  1
#endif
#ifdef  C_PLUS_PLUS_BAGS
#define SIZE_MPTR_BAGS  2
#endif

// header definition is found in memmgr.h

#define COPY_HEADER(dst, src) {int i; for(i=0; i<HEADER_SIZE; ++i) dst[i]=src[i]; }


/***************************************************************************** 
**
**  Memory for bags is allocated in chunks (e.g., 256 MByte) at a time.  We have
**  a structure (ArenaBag_t) which holds the information needed to manage each
**  pool or arena.  The structure contains pointers to the beginning of the
**  master pointers, the beginning of the bag data (both 'old' bags and 'young'
**  bags), the allocations area, and the end of the area (or
**  arena).
**
*V  BagHandleStart  . . . . . . . . . . . . beginning of the masterpointer area
*V  OldBagStart . . . . . . . . . . . . . . . .  beginning of the old bags area
*V  YoungBagStart . . . . . . . . . . . . . .  beginning of the young bags area
*V  AllocBagStart . . . . . . . . . . . . . .  beginning of the allocation area
*V  StopBags  . . . . . . . . . . . . . . . . beginning of the unavailable area
*V  EndBags . . . . . . . . . . . . . . . . . . . . . . . . .  end of the arena
*V  FreeHandleChain . . . . . . . . . . . . . . .  list of free bag identifiers
*V  MarkedBagChain  . . . . . . . . . . . . . . . . . . . . list of marked bags
**
**  The memory manager manages large blocks of storage called *arenas*; all
**  the arenas together comprise the *workspace*.  The workspace is allocated
**  one arena at a time (first arena is allocated during program startup); the
**  workspace size is expanded as needed.  Each arena size is based on the -m
**  option specified on the command line.  The layout of each arena is as
**  follows{\:}
**
**  +--------------+--------------+--------------+--------------+--------------+
**  |masterpointer |   old bags   |  young bags  |  allocation  |  unavailable |
**  |    area      |     area     |     area     |     area     |     area     |
**  +--------------+--------------+--------------+--------------+--------------+
**  ^              ^              ^              ^              ^              ^
**  BagHandleStart OldBagStart    YoungBagStart  AllocBagStart  StopBags       EndBags
**
**  The *masterpointer area* contains all the masterpointers of the bags in a
**  given arena.  Bag handles and their respective bags live within a single
**  arena; there is currently no cross referencing or mingling across arenas.
**  'BagHandleStart' points to the beginning of the arena and 'OldBagStart' to
**  the end.  The master pointer area is allocated 1/8 of the total of each
**  arena.
**
**  The *old bags area* contains the bodies of all the bags that survived at
**  least one garbage collection.  This area is only scanned for dead bags
**  during a full garbage collection (*NOTE: only full GC is currently
**  supported).  'OldBagStart' points to the beginning of this area and
**  'YoungBagStart' to the end.
**
**  The *young bags area* contains the bodies of all the bags that have been
**  allocated since the last garbage collection.  This area is scanned for dead
**  bags during each garbage collection.  'YoungBagStart' points to the
**  beginning of this area and 'AllocBagStart' to the end.
**
**  The *allocation area* is the storage that is available for allocation of new
**  bags.  When a new bag is allocated the storage for the body is taken from
**  the beginning of this area, and this area is correspondingly reduced.  If
**  the body does not fit in the allocation area a garbage collection is
**  performed.  'AllocBagStart' points to the beginning of this area and
**  'StopBags' to the end.
**
**  The *unavailable area* is the free storage that is not available for
**  allocation.  'StopBags' points to the beginning of this area and 'EndBags'
**  to the end.  The current implementation does not maintain an unavailable
**  area (allocation area runs from end of young bags to end of arena).
**
**  'CollectBags' makes all of the free storage available for allocations by
**  setting 'StopBags' to 'EndBags' after garbage collections.  In this case
**  garbage collections are only performed when no free storage is left.
**  <cache-size> is not used in this version, further garbage collection is
**  always "full".
**
**  'FreeHandleChain' is the first free bag identifier, i.e., it points to the
**  first available masterpointer.  If 'FreeHandleChain' is 0 there are no
**  available masterpointers.  The available masterpointers are managed in a
**  forward linked list, i.e., each available masterpointer points to the next
**  available masterpointer, except for the last, which contains 0.
**
**  When a new bag is allocated it gets the identifier 'FreeHandleChain', and
**  'FreeHandleChain' is set to the value stored in this masterpointer, which is
**  the next available masterpointer.  When a bag is deallocated by garbage
**  collection its masterpointer is added to the list of available
**  masterpointers again.
**
**  'MarkedBagChain' holds a list of bags that have already been marked during a
**  garbage collection by 'MARK_BAG'.  This list is only used during garbage
**  collections, so it is always empty outside of garbage collections (see
**  Implementation of "CollectBags").
**
**  This list starts with the bag whose identifier is 'MarkedBagChain', and the
**  link word of each bag on the list contains the identifier of the next bag on
**  the list.  The link word of the last bag on the list contains 0.  If
**  'MarkedBagChain' has the value 0, the list is empty.
**
**  'MARK_BAG' puts a bag <bag> onto this list.  'MARK_BAG' has to be careful,
**  because it can be called with an argument that is not really a bag
**  identifier, and may point outside the programs address space.  So 'MARK_BAG'
**  first determines the arena in which <bag> lives and then checks that <bag>
**  points to a properly aligned location between 'BagHandleStart' and
**  'OldBagStart'.  Then 'MARK_BAG' checks that <bag> is the identifier of a bag
**  by checking that the masterpointer points to a location between
**  'YoungBagStart' and 'AllocBagStart' (if <bag> is the identifier of an old
**  bag, the masterpointer will point to a location between 'OldBagStart' and
**  'YoungBagStart', and if <bag> only appears to be an identifier, the
**  masterpointer could be on the free list of masterpointers and point to a
**  location between 'BagHandleStart' and 'OldBagStart').  Since we only do full
**  garbage collection 'YoungBagStart' is set to 'OldBagStart' at the beginning
**  of GC, thus, effectively all bags are "young".  Then 'MARK_BAG' checks that
**  <bag> is not already marked by checking that the link word of <bag> contains
**  the identifier of the bag.  If any of the checks fails, 'MARK_BAG' does
**  nothing.  If all checks succeed, 'MARK_BAG' puts <bag> onto the list of
**  marked bags by putting the current value of 'MarkedBagChain' into the link
**  word of <bag> and setting 'MarkedBagChain' to <bag>.  Note that since bags
**  are always placed at the front of the list, 'CollectBags' will mark the bags
**  in a depth-first order.  This is probably good to improve the locality of
**  reference.
** 
**  Note that the borders between the areas are not static.  In particular each
**  allocation increases the size of the young bags area and reduces the size of
**  the allocation area.  On the other hand each garbage collection empties the
**  young bags area.  */

#define MAX_MEMORY_ARENAS  50			/* maximum number of memory arenas */

int             actAR = 0;				// active memory arena
ArenaBag_t      MemArena[MAX_MEMORY_ARENAS];


/****************************************************************************
**
*F  NewBag( <type>, <size> )  . . . . . . . . . . . . . . .  create a new bag
**
**  'NewBag' allocates memory for a new bag of the type <type> and size <size>.
**  <type> is a symbolic, see 'ObjType'.  <size> is an unsigned long.  'NewBag'
**  returns the handle of the new bag, which must be passed as first argument to
**  all other Gasman functions identifying this bag.  All entrys of the new bag
**  are initialized to 0.
*/


BagPtr_t        NewBag ( UInt type, UInt size )
{
    BagPtr_t result;
    assert(type < T_ILLEGAL);
    result = NewBag4(type, size);
    return result;
}


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
void            Retype ( BagPtr_t hdBag, UInt newType )
{
    assert(newType < T_ILLEGAL);

    RetypeBag(hdBag, newType);
}


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

void            Resize ( BagPtr_t hdBag, UInt newSize )
{
    ResizeBag(hdBag, newSize);
}

//  GetArenaFromBag() returns a pointer to the memory arena in which
//  bag lives.  To be valid a bag handle must be located betwen
//  BagHandleStart and OldBagStart (i.e., the area reserved for master
//  pointers *in the current memory arena*)
	 
static ArenaBag_t *GetArenaFromBag( BagPtr_t bag )
{
    ArenaBag_t *par = &MemArena[0];

    while (par->ActiveArenaFlag) {                // Active arenas have valid start handles
        if (bag >= par->BagHandleStart && bag < par->OldBagStart) {
            return par;
        }
        par++;
    }

    // getting here means the bag was not found in any active memory arena -- return NULL
    return (ArenaBag_t *)NULL;
}


void            MarkBag_GAP3(BagPtr_t bag)
{
    BagPtr_t    *ptr;					/* pointer into the bag            */
    BagPtr_t     sub;					/* one subbag identifier           */
    UInt         i;						/* loop variable                   */
    UInt         iter = NrHandles(GET_TYPE_BAG(bag), GET_SIZE_BAG(bag));
    
    ptr = PTR_BAG( bag );
    for(i = 0; i < iter; i++) {
        sub = ptr[i];
        MARK_BAG( sub );
    }
}


void MARK_BAG( BagPtr_t bag )
{
    /*
     * Check the bag presented is:
     *    non-zero
     *    has none of the 2 or 3 least significant bits sets (is not currently tagged)
     *    Get the arena from bag pointer 
     * Next check the data pointer associated to the bag:
     *    lies between OldBagStart and AllocBagStart (we check all bags) 
     * Next check 
     *    if the link pointer points back to the bag handle (bag not previous marked)
     */
    ArenaBag_t *par;

    if (bag == 0 || ((UInt)bag & (sizeof(BagPtr_t) - 1)) != 0)
        return;							// not a valid bag handle

    if (!(par = GetArenaFromBag(bag)))            // no arena or invalid 
        return;
    
    // bag is in this arena
    BagPtr_t *ptr = (*(BagPtr_t **)(bag)); /* data pointer asociated to bag */
    
    //    for full GC (only type we do) YoungBagStart is set to OldBagStart
    if (par->OldBagStart < ptr  &&  ptr <= par->AllocBagStart) {
        BagStruct_t *pbs = (BagStruct_t *)(ptr - HEADER_SIZE); /* struct pointer */

        if ( (BagPtr_t)pbs->bagLinkPtr == bag ) {
            pbs->bagLinkPtr = par->MarkedBagChain;
            par->MarkedBagChain = bag;
            par->nrMarkedBags++;
            par->SweepNeeded = 1;
        }
    }
    else if (par->BagHandleStart <= ptr && ptr < par->OldBagStart) {
        // bag pointer is in master pointer area -- this should be a free mptr on the free chain, do nothing
    }
    else {
        printf("MARK_BAG: bag handle (%p), data ptr = %p not in valid bag range for arena #%d\n",
               bag, ptr, par->ArenaNumber);
    }

    return;
}


/****************************************************************************
**
*F  InitGasman()  . . . . . . . . . initialize dynamic memory manager package
**
**  'InitGasman' initializes   Gasman, i.e., allocates   some memory  for the
**  memory managment, and sets up the bags needed  for the working of Gasman.
**  These are the new  handles bag, which  remembers the handles of bags that
**  must not be  thrown away during a  garbage  collection and  the free bag,
**  from  which  the memory for  newly   created bags  is taken by  'NewBag'.
**  'InitGasman'  must only   be  called  once  from  'InitGap'  right  after
**  initializing  the scanner,  but before  everything else,  as none of  the
**  storage manager functions work before calling 'InitGasman'.
*/


extern void InitAllGlobals();

void            InitGasman (int argc, char** argv, int* stackBase)
{
    UInt type;

    InitSystem4(argc, argv);
    
    InitBags( SyAllocBags, SyStorMin,
              (BagPtr_t*)(((UInt)stackBase/SyStackAlign)*SyStackAlign), SyStackAlign,
              0, SyAbortBags );
    
    /* Install MarkBag_GAP3 as the marking function for all bag types */
    /* (overrides MarkAllSubBagsDefault which was installed in InitBags above)  */
    for(type = 0; type < T_ILLEGAL; ++type) {
        InitMarkFuncBags(type, MarkBag_GAP3);
        InfoBags[type].name = NameType[type];
    }

    InitAllGlobals();
}


/****************************************************************************
**
*V  NrAllBags . . . . . . . . . . . . . . . . .  number of all bags allocated
*V  SizeAllBags . . . . . . . . . . . . . .  total size of all bags allocated
*V  SizeLiveBags  . . . . . . .  total size of bags that survived the last gc
*V  SizeAllArenas . . . . . . . . . . . . . . total size of all Memory Arenas
*/

UInt            NrAllBags;
UInt            SizeAllBags;
UInt            SizeLiveBags;
UInt            SizeAllArenas;


/****************************************************************************
**
*V  InfoBags[<type>]  . . . . . . . . . . . . . . . . .  information for bags
*/

TNumInfoBags            InfoBags [ NTYPES ];

const char     *TNAM_BAG ( BagPtr_t bag )
{
    return InfoBags[ GET_TYPE_BAG(bag) ].name;
}


/****************************************************************************
**
*F  InitSweepFuncBags(<type>,<mark-func>)  . . . .  install sweeping function
*/

TNumSweepFuncBags TabSweepFuncBags [ NTYPES ];


void InitSweepFuncBags ( UInt type, TNumSweepFuncBags sweep_func )
{
#ifdef CHECK_FOR_CLASH_IN_INIT_SWEEP_FUNC
    char                str[256];

    if ( TabSweepFuncBags[type] != 0 ) {
        str[0] = 0;
        strncat( str, "warning: sweep function for type ", 33 );
        str[33] = '0' + ((type/100) % 10);
        str[34] = '0' + ((type/ 10) % 10);
        str[35] = '0' + ((type/  1) % 10);
        str[36] = 0;
        strncat( str, " already installed\n", 19 );
        SyFputs( str, 0 );
    }
#endif
    TabSweepFuncBags[type] = sweep_func;
}


/****************************************************************************
**
*F InitMarkFuncBags(<type>,<mark-func>) . . . . .  install marking function
**
**  'InitMarkFuncBags' and 'MarkAllSubBagsDefault' are really too simple for an
**  explanation.
**
**  'MarkAllSubBagsDefault' is only used by GASMAN as default.
*/

TNumMarkFuncBags TabMarkFuncBags [ NTYPES ];

extern void MarkAllSubBagsDefault ( BagPtr_t bag );

void InitMarkFuncBags ( UInt type, TNumMarkFuncBags mark_func )
{
#ifdef CHECK_FOR_CLASH_IN_INIT_MARK_FUNC
    char                str[256];

    if ( TabMarkFuncBags[type] != MarkAllSubBagsDefault ) {
        str[0] = 0;
        strncat( str, "warning: mark function for type ", 32 );
        str[32] = '0' + ((type/100) % 10);
        str[33] = '0' + ((type/ 10) % 10);
        str[34] = '0' + ((type/  1) % 10);
        str[35] = 0;
        strncat( str, " already installed\n", 19 );
        SyFputs( str, 0 );
    }
#endif
    TabMarkFuncBags[type] = mark_func;
}

void MarkAllSubBagsDefault ( BagPtr_t bag )
{
    BagPtr_t   *ptr;					/* pointer into the bag            */
    BagPtr_t    sub;					/* one subbag identifier           */
    UInt        i;						/* loop variable                   */

    /* mark everything                                                     */
    ptr = PTR_BAG( bag );
    for ( i = GET_SIZE_BAG(bag)/sizeof(BagPtr_t); 0 < i; i-- ) {
        sub = ptr[i-1];
        MARK_BAG( sub );
    }

}


/****************************************************************************
**  
*V  GlobalBags  . . . . . . . . . . . . . . . . . . . . . list of global bags
*/  
TNumGlobalBags GlobalBags;


/****************************************************************************
**
*F  InitGlobalBag(<addr>, <cookie>) inform Gasman about global bag identifier
**
**  'InitGlobalBag' simply leaves the address <addr> in a global array, where
**  it is used by 'CollectBags'. <cookie> is also recorded to allow things to
**  be matched up after loading a saved workspace.
*/

UInt    GlobalSortingStatus = 0;
Int     WarnInitGlobalBag = 0;

extern TNumAbortFuncBags   AbortFuncBags;

void    InitGlobalBag ( BagPtr_t *addr, const Char *cookie )
{
    if ( GlobalBags.nr == NR_GLOBAL_BAGS ) {
        (*AbortFuncBags)( "Panic: Gasman cannot handle so many global variables" );
    }
	
#ifdef DEBUG_GLOBAL_BAGS
    {
		UInt i;
		if (cookie != (Char *)0)
			for (i = 0; i < GlobalBags.nr; i++)
				if ( 0 == strcmp(GlobalBags.cookie[i], cookie) ) {
					if (GlobalBags.addr[i] == addr)
						Pr("Duplicate global bag entry %s\n", (Int)cookie, 0);
					else
						Pr("Duplicate global bag cookie %s\n", (Int)cookie, 0);
				}
    }
#endif

    if ( WarnInitGlobalBag ) {
        Pr( "#W  global bag '%s' initialized\n", (Int)cookie, 0 );
    } 
    GlobalBags.addr[GlobalBags.nr] = addr;
    GlobalBags.cookie[GlobalBags.nr] = cookie;
    GlobalBags.nr++;
    GlobalSortingStatus = 0;
}


/****************************************************************************
**
*F  InitFreeFuncBag(<type>,<free-func>) . . . . . .  install freeing function
**
**  'InitFreeFuncBag' is really too simple for an explanation.
*/

TNumFreeFuncBags        TabFreeFuncBags [ NTYPES ];

UInt                    NrTabFreeFuncBags;

void            InitFreeFuncBag ( UInt type, TNumFreeFuncBags free_func )
{
    if ( free_func != 0 ) {
        NrTabFreeFuncBags = NrTabFreeFuncBags + 1;
    }
    else {
        NrTabFreeFuncBags = NrTabFreeFuncBags - 1;
    }
    TabFreeFuncBags[type] = free_func;
}


/****************************************************************************
**
*F  InitCollectFuncBags(<bfr-func>,<aft-func>) . install collection functions
**
**  'InitCollectFuncBags' is really too simple for an explanation.
*/
TNumCollectFuncBags     BeforeCollectFuncBags;

TNumCollectFuncBags     AfterCollectFuncBags;

void            InitCollectFuncBags (
    TNumCollectFuncBags before_func,
    TNumCollectFuncBags after_func )
{
    BeforeCollectFuncBags = before_func;
    AfterCollectFuncBags  = after_func;
}


/****************************************************************************
**
*F  InitBags(...) . . . . . . . . . . . . . . . . . . . . . initialize Gasman
**
**  'InitBags' remembers <alloc-func>, <stack-bottom>, <stack-align>,
**  <cache-size>, <dirty>, and <abort-func> in global variables.  It also
**  allocates the initial workspace, and sets up the linked list of available
**  masterpointer.
*/

TNumAllocFuncBags       AllocFuncBags;
BagPtr_t *              StackBottomBags;
UInt                    StackAlignBags;
UInt                    DirtyBags;
TNumAbortFuncBags       AbortFuncBags;

//  set FreeRatioThreshold to GASMAN_FREE_RATIO (default), can override this via SetArenaFreeRatio()
static float FreeRatioThreshold   = (float) GASMAN_FREE_RATIO;
static float UnlockRatioThreshold = (float) GASMAN_OCCUPIED_RATIO;


static int AddMemoryArena ( UInt size )
{
    BagPtr_t *p;                        // loop variable
    int arenanr;
    ArenaBag_t *par = &MemArena[0];

    // find first free arena
    for (arenanr = 0 ; arenanr < MAX_MEMORY_ARENAS; par++, arenanr++) {
        if (par->ActiveArenaFlag == 0)	// never allocated ... use this one
            break;
    }

    if (arenanr >= MAX_MEMORY_ARENAS )
        // arena is beyond max number of arenas handled
        (*AbortFuncBags)("Cannot extend storage -- too many memory arenas.");

    // size is amount of memory in kilobytes (SyAllocBags() converts to bytes)
    par->BagHandleStart = (*AllocFuncBags)( size );
    if ( par->BagHandleStart == 0 ) {
        // Cannot get storage for the requested arena, cannot extend the workspace
        printf("AddMemoryArena: Cannot get storage, workspace maxed out;\n");
        return (-1);
    }
    
    par->EndBags = par->BagHandleStart + 1024 * (size / sizeof(BagPtr_t*));
    // 1/8th of the storage goes into the masterpointer area
    par->FreeHandleChain = (BagPtr_t)par->BagHandleStart;
    par->OldBagStart   = par->BagHandleStart + 1024 * size / 8 / sizeof(BagPtr_t*);
    par->YoungBagStart = par->OldBagStart;
    par->AllocBagStart = par->OldBagStart;

    // setup forward link chain for master pointers
    for ( p = par->BagHandleStart;
          p + 2 * (SIZE_MPTR_BAGS) <= par->OldBagStart;
          p += SIZE_MPTR_BAGS )
    {
        *p = (BagPtr_t)(p + SIZE_MPTR_BAGS);
    }

    par->StopBags = par->EndBags;
	par->SweepNeeded = 0;
	par->FreeRatio = FreeRatioThreshold;

    par->ActiveArenaFlag = 1;
    par->ArenaNumber = arenanr;
    par->SizeArena = size * 1024;
    
    actAR = arenanr;
    return (actAR);
}


void            InitBags (
    TNumAllocFuncBags   alloc_func,
    UInt                initial_size,
    BagPtr_t *          stack_bottom,
    UInt                stack_align,
    UInt                dirty,
    TNumAbortFuncBags   abort_func )
{
    UInt                i;              /* loop variable                   */

    /* install the allocator and the abort function                        */
    AllocFuncBags   = alloc_func;
    AbortFuncBags   = abort_func;

    /* install the stack marking function and values                       */
    StackBottomBags = stack_bottom;
    StackAlignBags  = stack_align;

    /* remember whether bags should be clean                               */
    DirtyBags = dirty;

    /* first get some storage from the operating system                    */
    initial_size    = (initial_size + 511) & ~(511);
    // request first memory arena (i.e., arena 0)
    i = AddMemoryArena( initial_size );
    if ( i < 0) {
        (*AbortFuncBags)("Cannot get storage for the initial workspace.");
    }
    
    /* install the marking functions                                       */
    for ( i = 0; i < NTYPES; i++ )
        TabMarkFuncBags[i] = MarkAllSubBagsDefault;

}


/****************************************************************************
**
*F  NewBag4( <type>, <size> )  . . . . . . . . . . . . . .  allocate a new bag
**
**  'NewBag4' is actually quite simple.
**
**  It first tests whether enough storage is available in the allocation area
**  and whether a free masterpointer is available.  If not, it starts a garbage
**  collection by calling 'CollectBags' passing <size> as the size of the bag it
**  is currently allocating.  If 'CollectBags' fails and returns 0, 'NewBag4'
**  also fails and also returns 0.
**
**  Then it takes the first free  masterpointer from the  linked list of free
**  masterpointers (see "FreeHandleChain").
**
**  Then it  writes  the  size and the   type  into the word   pointed to  by
**  'AllocBagStart'.  Then  it writes the identifier,  i.e.,  the location of the
**  masterpointer, into the next word.
**
**  Then it advances 'AllocBagStart' by 'HEADER_SIZE + WORDS_BAG(<size>)'.
**
**  Finally it returns the identifier of the new bag.
**
**  Note that 'NewBag4' never  initializes the new bag  to contain only 0.  If
**  this is desired because  the initialization flag <dirty> (see "InitBags")
**  was  0, it is the job  of 'CollectBags'  to initialize the new free space
**  after a garbage collection.
**
**  If {\Memmgr} was compiled with the option 'COUNT_BAGS' then 'NewBag4' also
**  updates the information in 'InfoBags' (see "InfoBags").
**
**  Most tracing/debugging or statistical information available from Memmgr is
**  available by setting the variable SyMemMgrTrace to 1.  The variable can be
**  toggled using the commands GASMAN("traceON") and GASMAN("traceOFF") in GAP.
**  Thus, recompilation is not required in order to enable diagnostice
**  information.
*/

BagPtr_t NewBag4 ( UInt type, UInt size )
{
    BagPtr_t                 bag;		/* identifier of the new bag       */
    BagPtr_t *               dst;		/* destination of the new bag      */
    ArenaBag_t *par = &MemArena[actAR];	// ptr to current (active) memory Arena
    
    /* check that a masterpointer and enough storage are available         */
    if ( ( ((par->FreeHandleChain < par->BagHandleStart) ||
            (par->FreeHandleChain >= par->OldBagStart)) ||
           par->StopBags < par->AllocBagStart + HEADER_SIZE + WORDS_BAG(size) ) &&
         CollectBags( size, -1, "NewBag4()" ) == 0 )
    {
        return 0;
    }

#ifdef  COUNT_BAGS
    /* update the statistics                                               */
    NrAllBags               += 1;
    SizeAllBags             += size;
    InfoBags[type].nrLive   += 1;
    InfoBags[type].nrAll    += 1;
    InfoBags[type].sizeLive += size;
    InfoBags[type].sizeAll  += size;
#endif

	/* get the identifier of the bag and set 'FreeHandleChain' to the next    */
    par = &MemArena[actAR];	// refresh because CollectBags could alloc a new arena
    bag = par->FreeHandleChain;
    if (bag == 0) {
        // no more free bags available ==> out of memory, exit
        // *** here or from CollectBags should allocate a new memory arena if full ***
        GuFatalMsgExit(EXIT_MEM, "Newbag4: FreeHandleChain chain is empty, no more memory\n");
    }
    else if (bag < par->BagHandleStart || bag >= par->OldBagStart) {
        // Not a valid bag handle, head of free chain is corrupt, print message & exit
        GuFatalMsgExit(EXIT_MEM, "Newbag4: FreeHandleChain chain head is corrupt ... exiting\n");
    }
    par->FreeHandleChain = *(BagPtr_t*)bag;

    /* allocate the storage for the bag                                    */
    dst       = par->AllocBagStart;
    par->AllocBagStart = dst + HEADER_SIZE + WORDS_BAG(size);

    SET_TYPE_PTR(dst,type);
    BLANK_FLAGS_PTR(dst);
    SET_SIZE_PTR(dst, size);
    SET_LINK_PTR(dst, bag);

    /* set the masterpointer                                               */
    SET_PTR_BAG(bag, dst + HEADER_SIZE);

#ifdef DEBUG_BAG_MGMT
    // insert a code value into the bag header for tracing...
    ((BagStruct_t *)(dst))->bagCodeValue = bagCodeMarker;
#endif
    
#ifdef DEBUG_POINTERS
    if (bag != GET_LINK_PTR(dst)) {
        // any new bag should always have the bag handle (bag) set as the link pointer value
        BagStruct_t *bs = (BagStruct_t *)dst;
        printf("NewBag4: Created bag (%p) but LINK ptr location (%p) not set to bag handle: bag location (%p), link ptr = %p\n",
               bag, &(bs->bagLinkPtr), dst, bs->bagLinkPtr);
    }
#endif
    
    /* return the identifier of the new bag                                */
    return bag;
}


/****************************************************************************
**
*F  RetypeBag(<bag>,<new>)  . . . . . . . . . . . .  change the type of a bag
**
**  'RetypeBag' is very simple.
**
**  All it has to do is to change the flags-type word of the bag.
**
**  If  {\Memmgr} was compiled with the  option 'COUNT_BAGS' then 'RetypeBag'
**  also updates the information in 'InfoBags' (see "InfoBags").
*/

void            RetypeBag ( BagPtr_t bag, UInt new_type )
{
    UInt        size;					/* size of the bag                 */

    /* get old type and size of the bag                                    */
    size     = GET_SIZE_BAG(bag);

#ifdef  COUNT_BAGS
    /* update the statistics      */
    {
          UInt old_type;				/* old type of the bag */

      old_type = GET_TYPE_BAG(bag);
      InfoBags[old_type].nrLive   -= 1;
      InfoBags[new_type].nrLive   += 1;
      InfoBags[old_type].nrAll    -= 1;
      InfoBags[new_type].nrAll    += 1;
      InfoBags[old_type].sizeLive -= size;
      InfoBags[new_type].sizeLive += size;
      InfoBags[old_type].sizeAll  -= size;
      InfoBags[new_type].sizeAll  += size;
    }
#endif

    /* change the flags-type word                                           */
    SET_TYPE_BAG(bag, new_type);
    BLANK_FLAGS_BAG(bag);
}


/****************************************************************************
**
*F  ResizeBag(<bag>,<new>)  . . . . . . . . . . . .  change the size of a bag
**
**  Basically 'ResizeBag' is rather  simple, but there  are a few  traps that
**  must be avoided.
**
**  If the size of the bag changes only a little bit, so that  the  number of
**  words needed for the data area does not  change, 'ResizeBag' only changes
**  the size word of the bag.
**
**  If the bag is to be shrunk and at least one word becomes free, 'ResizeBag'
**  changes the size word of the bag, and stores a magic type word in the first
**  free word.  This magic type word has type 255 (T_RESIZE_FREE).  The size of
**  the space freed up is recorded in the following word (unless just a single
**  word is freed).  The size freed is ('old # words' - 'new # words' - 1) *
**  'sizeof(BagPtr_t)'.  If just a single word is freed, no size word follows
**  the magic type; however, the flag BF_COPY is also set in the type word to
**  indicate this.  The type (T_RESIZE_FREE) allows 'CollectBags' to detect that
**  this body is the remainder of a resize operation, and the size allows it to
**  know how many bytes there are in this body (see "Implementation of
**  CollectBags").
**
**  So for example if 'ResizeBag' shrinks a bag of type 7 from 22 bytes to 10
**  bytes the situation is as follows (assume 64 bit words; however, 32 bit
**  words are exactly analogous){\:}
**
**             +----------------+                +----------------+
**       ----->| Master pointer |             -->| Master pointer |
**      /      +----------------+            /   +----------------+
**     /                      \_______      /                   \_________
**     |                              |     |                             |
**     |   +---------------------+    |     |   +---------------------+   |
**     |   |   Flags     |   7   |    |     |   |   Flags     |   7   |   |
**     |   +---------------------+    |     |   +---------------------+   |
**     |   |         18          |    |     |   |         18          |   |
**     |   +---------------------+    |     |   +---------------------+   |
**     |   | Copy Pointer        |    |     |   |   Copy Pointer      |   |
**     |   +---------------------+    |     |   +---------------------+   |
**     +-- | Link Pointer        |    |     +-- |   Link Pointer      |   |
**         +---------------------+    |         +---------------------+   |
**         | Bag Data            |<---          |   Bag Data          |<--
**         +---------------------+              +---------------------+
**         | Bag Data            |              | Bag Data |  Padding |
**         +---------------------+              +---------------------+
**         | Bag Data            |              |                0xFF |
**         +---------------------+              +---------------------+
**         | Bag Data  | Padding |              |          8          |
**         +---------------------+              +---------------------+
**
**  If the bag is to be extended and it is that last allocated bag, so that it
**  is immediately adjacent to the allocation area, 'ResizeBag' simply
**  increments 'AllocBagStart' after making sure that enough space is available
**  in the allocation area (see "Layout of the Workspace").
**
**  If the bag is to be extended and it is not the last allocated bag,
**  'ResizeBag' first allocates a new bag similar to 'NewBag4', but without
**  using a new masterpointer.  Then it copies the old contents to the new
**  bag.  Finally it resets the masterpointer of the bag to point to the new
**  address.  Then it changes the type of the old body to T_RESIZE_FREE,
**  so that the garbage collection can detect that this body is the remainder
**  of a resize (see "Implementation of NewBag4" and "Implementation of
**  CollectBags").
**
**  If {\Gasman}  was compiled with the  option 'COUNT_BAGS' then 'ResizeBag'
**  also updates the information in 'InfoBags' (see "InfoBags").
*/

static UInt nrResizeBags = 0;

UInt ResizeBag ( BagPtr_t bag, UInt new_size )
{
    UInt        type;					/* type of the bag                 */
    UInt        old_size;				/* old size of the bag             */
    UInt        flags;
    BagPtr_t   *dst;					/* destination in copying          */
    BagPtr_t   *src;					/* source in copying               */
    BagPtr_t   *end;					/* end in copying                  */

    /* get type and old size of the bag                                    */
    type     = GET_TYPE_BAG(bag);
    old_size = GET_SIZE_BAG(bag);
    flags    = GET_FLAGS_BAG(bag);

    // in which arena is the bag handle?  resized bag stays in that same arena
    ArenaBag_t *par = GetArenaFromBag(bag);
    BagPtr_t *p = (*(BagPtr_t **)(bag));
    if (p < par->OldBagStart || p >= par->EndBags) {
        // bag data is not in the allocated area of this arena
        printf("ResizeBag: bag = %llx, Arena #%d, **issue** bag data %p is not in allocated area of arena\n",
               bag, par->ArenaNumber, p);
    }
    
#ifdef  COUNT_BAGS
    /* update the statistics                                               */
    SizeAllBags             += new_size - old_size;
    InfoBags[type].sizeLive += new_size - old_size;
    InfoBags[type].sizeAll  += new_size - old_size;
#endif

    nrResizeBags++;
    /* if the real size of the bag doesn't change                          */
    if ( WORDS_BAG(new_size) == WORDS_BAG(old_size) ) {

        /* change the size word                                            */
        SET_SIZE_BAG(bag, new_size);
        return 1;
    }

    /* if the bag is shrunk                                                */
    /* we must not shrink the last bag by moving 'AllocBagStart',              */
    /* since the remainder may not be zero filled                          */
    else if ( WORDS_BAG(new_size) < WORDS_BAG(old_size) ) {

        /* leave magic word for the sweeper: copy flag (BF_COPY) is set, type must be 255 (T_RESIZE_FREE)    */
        if ((WORDS_BAG(old_size)-WORDS_BAG(new_size) == 1))
            *(UInt*)(PTR_BAG(bag) + WORDS_BAG(new_size)) = BF_COPY | T_RESIZE_FREE;
        else  {
            *(UInt*)(PTR_BAG(bag) + WORDS_BAG(new_size)) = T_RESIZE_FREE;
            *(UInt*)(PTR_BAG(bag) + WORDS_BAG(new_size) + 1) =
                (WORDS_BAG(old_size)-WORDS_BAG(new_size)-1)*sizeof(BagPtr_t);
        }

        /* change the size- word                                       */
        SET_SIZE_BAG(bag, new_size);
        return 1;
    }

    /* if the last bag is to be enlarged                                   */
    else if ( PTR_BAG(bag) + WORDS_BAG(old_size) == par->AllocBagStart ) {
        
        /* check that enough storage for the new bag is available          */
        if ( par->StopBags < PTR_BAG(bag) + WORDS_BAG(new_size)
             && CollectBags( new_size - old_size, par->ArenaNumber, "ResizeBag()" ) == 0 ) {
            return 0;
        }

        /* simply increase the free pointer                                */
        if ( par->YoungBagStart == par->AllocBagStart )
            par->YoungBagStart += WORDS_BAG(new_size) - WORDS_BAG(old_size);
        par->AllocBagStart += WORDS_BAG(new_size) - WORDS_BAG(old_size);

        /* change the size word                                       */
        SET_SIZE_BAG(bag, new_size ) ;
        return 1;
    }

    /* if the bag is enlarged                                              */
    else {

        /* check that enough storage for the new bag is available          */
        if ( par->StopBags < par->AllocBagStart + HEADER_SIZE + WORDS_BAG(new_size)
             && CollectBags( new_size, par->ArenaNumber, "ResizeBag()" ) == 0 ) {
            return 0;
        }

        /* allocate the storage for the bag                                */
        dst       = par->AllocBagStart;
        par->AllocBagStart = dst + HEADER_SIZE + WORDS_BAG(new_size);
    
        /* leave magic type word  for the sweeper, type must be T_RESIZE_FREE  */
        SET_TYPE_BAG(bag, T_RESIZE_FREE);
        BLANK_FLAGS_BAG(bag);
        SET_SIZE_BAG(bag, (((WORDS_BAG(old_size) + HEADER_SIZE - 1) * sizeof(BagPtr_t))));
    
        // set the new size, flags, & type
        SET_TYPE_PTR(dst,type);
        BLANK_FLAGS_PTR(dst);
        SET_SIZE_PTR(dst, new_size);

        BagPtr_t *dst2 = dst;
        dst2 = dst2 + HEADER_SIZE;

        /* copy the contents of the bag                                    */
        src = PTR_BAG(bag);
        end = src + WORDS_BAG(old_size);
        while ( src < end )
            *dst2++ = *src++;

        SET_LINK_PTR(dst, bag);

        /* set the masterpointer                                           */
        SET_PTR_BAG(bag, (dst + HEADER_SIZE));
        SET_FLAG_BAG(bag, flags);
    }

#ifdef DEBUG_POINTERS
    if (bag != GET_LINK_PTR(dst)) {
        // resized bag should have the bag handle (bag) set as the link pointer value 
        BagStruct_t *bs = (BagStruct_t *)dst;
        printf("ResizeBag: resized bag (%p) but LINK ptr location (%p) not set to bag handle: bag location (%p), link ptr = %p\n",
               bag, &(bs->bagLinkPtr), dst, bs->bagLinkPtr);
    }
#endif
    
    /* return success                                                      */
    return 1;
}


/****************************************************************************
**
*F  CollectBags( <size>, <spec_arena>, <caller> ) . . . . . collect dead bags
**
**  'CollectBags' is the function that does most of the work of {\Memmgr}.
**
**  An earlier implementation attempted to support the concept of a partial
**  garbage collection (GC), wherein it intended to only cycle through the young
**  bags and any bags linked on a chain of changed old bags.  This didn't work
**  and was bypassed in favour of always doing a full GC, wherein all bags
**  are considered during the cleanup.  During the current re-implementation
**  many zombie lines of code and unused items have been removed.
**
**  The concept of "old" bags and "young" bags remains, to allow for potential
**  future enhancements; however, from a practical perspective there is no
**  differentiation between young and old bags for the present.
**
**  Garbage collection  is  performed in  three phases.  The  mark phase, the
**  sweep phase, and the check phase.
**
**  In the *mark phase*, 'CollectBags' finds all young bags that are still live
**  and builds a linked list of those bags (see "MarkedBagChain").  A bag is put
**  on this list of marked bags by applying 'MARK_BAG' to its identifier.  Note
**  that 'MARK_BAG' checks that a bag is not already on the list, before it puts
**  it on the list, so no bag can be put on this list twice.
**
**  First, 'CollectBags' marks all bags that are directly accessible through
**  global variables, i.e., it marks those bags whose identifiers appear in
**  global variables.  It does this by applying 'MARK_BAG' to the values at the
**  addresses of global variables that may hold bag identifiers provided by
**  'InitGlobalBag' (see "InitGlobalBag").
**
**  Next, 'CollectBags' marks all bags that are directly accessible through
**  local variables, i.e., it marks those bags whose identifiers appear in the
**  stack.  It does this by calling 'GenStackFuncBags'.  This works by applying
**  'MARK_BAG' to all values on the stack (every value on the stack is
**  considered).  The stack is considered to extend from <stack-start> (see
**  "InitBags") to the address of a local variable of the function.  This
**  completes the set of referenced bags (i.e., bags referenced via either
**  global or local program variables).
**
**  Next 'CollectBags' marks all bags that are children (subbags) of the
**  referenced bags, i.e., it marks all bags whose identifiers appear in the
**  data areas of the referenced bags.  It does this by applying 'MARK_BAG' to
**  each identifier appearing in list of referenced bags.
**
**  Next, 'CollectBags' marks all young bags that are *indirectly* accessible,
**  i.e., it marks the subbags of the already marked bags, their subbags and so
**  on.  It does so by walking along the list of already marked bags and applies
**  the marking function of the appropriate type to each bag on this list (see
**  "InitMarkFuncBags").  Those marking functions then apply 'MARK_BAG' to each
**  identifier appearing in the bag.
**
**  After the marking function has been applied to a bag on the list of marked
**  bags, this bag is removed from the list.  Thus the marking phase is over
**  when the list of marked bags becomes empty.  Removing the bag from the list
**  of marked bags must be done at this time, because newly marked bags are
**  *prepended* to the list of marked bags.  This is done to ensure that bags
**  are marked in a depth first order, which should usually improve locality of
**  reference.  When a bag is taken from the list of marked bags it is *tagged*.
**  This tag serves two purposes.  A bag that is tagged is not put on the list
**  of marked bags when 'MARK_BAG' is applied to its identifier.  This ensures
**  that no bag is put more than once onto the list of marked bags, otherwise
**  endless marking loops could happen for structures that contain circular
**  references.  Also the sweep phase later uses the presence of the tag to
**  decide the status of the bag. There are two possible states: LIVE and
**  DEAD. The default state of a bag with its identifier in the link word, is
**  the tag for DEAD. Live bags are tagged with MARKED_ALIVE(<identifier>) in
**  the link word.
** 
**  Note that 'CollectBags' cannot put a random or magic  value into the link
**  word, because the sweep phase must be able to find the masterpointer of a
**  bag by only looking at the link word of a bag. This is done using the macros
**  UNMARKED_{ALIVE|DEAD}(<link word contents>).
**
**  In the *sweep phase*, 'CollectBags' deallocates all dead bags and compacts
**  the live bags at the beginning of the workspace.
**
**  In this phase 'CollectBags' uses a destination pointer 'dst', which points
**  to the address a body will be copied to, and a source pointer 'src', which
**  points to the address a body currently has.  Both pointers initially point
**  to the beginning of the young bags area.  Then 'CollectBags' looks at the
**  body pointed to by the source pointer.
**
**  If this body has type T_RESIZE_FREE, it is the remainder of a resize
**  operation.  In this case 'CollectBags' simply moves the source pointer to
**  the next body (see "Implementation of ResizeBag").
**
**  Otherwise, if the link word contains the identifier of the bag itself, and
**  is 'MARKED_DEAD', 'CollectBags' first adds the masterpointer to the list of
**  available masterpointers (see "FreeHandleChain") and then simply moves the
**  source pointer to the next bag.
**
**  Otherwise, if the link word contains the identifier of the bag, and is
**  'MARKED_ALIVE', this bag is still live.  In this case 'CollectBags' calls
**  the sweeping function for this bag, if one is installed.  if not, it copies
**  the body from the source address to the destination address, stores the
**  address of the masterpointer without the tag in the link word, and updates
**  the masterpointer to point to the new address of the data area of the bag.
**  After the copying the source pointer points to the next bag, and the
**  destination pointer points just past the copy.
**
**  This is repeated until  the source pointer  reaches the end of  the young
**  bags area, i.e., reaches 'AllocBagStart'.
**
**  The new free storage now is the area between  the destination pointer and
**  the source pointer.  If the initialization flag  <dirty> (see "InitBags")
**  was 0, this area is now cleared.
**
**  Next, 'CollectBags' sets   'YoungBagStart'  and 'AllocBagStart'  to   the address
**  pointed to by the destination  pointer.  So all the  young bags that have
**  survived this garbage  collection are now  promoted  to be old  bags, and
**  allocation of new bags will start at the beginning of the free storage.
**
**  Finally, the *check phase* checks  whether  the garbage collection  freed
**  enough storage and masterpointers.
**
**  'CollectBags' is called with an <arena> number, which is the arena to have GC
**  performed; if the <arena> is -1, then the currently active arena is used.
**  Otherwise, the <arena> specified is chosen (called this way by
**  ResizeBag because the resized bag *cannot* move to a different arena).  If
**  <size> is non-zero, 'CollectBags' has been called to ensure there is at
**  least that much available storage.  If <size> is zero, and <arena> is -1,
**  then 'CollectBags' checks that the free (allocation) pool is at least
**  GASMAN_FREE_RATIO percent of the total pool for bags (EndBags -
**  OldBagStart).
**
**  If the free pool is less than this or no free master pointers are available
**  then 'CollectBags' tries to extend the workspace by allocating a new memory
**  management <arena>.  If it cannot extend the workspace, 'CollectBags'
**  returns 0 to indicate failure.  The new arena is marked as the currently
**  active arena and the existing arena is marked as 'full'.  "Old" arenas are
**  only subject to GC when needed (e.g., when a ResizeBag requestfor a bag in
**  that arena requires more memory than is available in the allocation pool
**  area).
*/
#include <setjmp.h>

static jmp_buf RegsBags;

//  Walk the stack looking for anything which might be a reference to a bag
//  (i.e., a bag handle)

void GenStackFuncBags (void)
{
    BagPtr_t   *top;					/* top of stack                    */
    BagPtr_t   *p;						/* loop variable                   */
    UInt        i;						/* loop variable                   */

    top = (BagPtr_t*)&top;
    if ( StackBottomBags < top ) {
        for ( i = 0; i < sizeof(BagPtr_t*); i += StackAlignBags ) {
            for ( p = (BagPtr_t*)((char*)StackBottomBags + i); p < top; p++ )
                MARK_BAG( *p );
        }
    }
    else {
        for ( i = 0; i < sizeof(BagPtr_t*); i += StackAlignBags ) {
            for ( p = (BagPtr_t*)((char*)StackBottomBags - i); top < p; p-- )
                MARK_BAG( *p );
        }
    }

    /* mark from registers, dirty dirty hack                               */
    for ( p = (BagPtr_t*)RegsBags;
          p < (BagPtr_t*)RegsBags+sizeof(RegsBags)/sizeof(BagPtr_t);
          p++ )
        MARK_BAG( *p );

}

static void PrintBagInfo ( BagStruct_t *ps )
{
    printf("PrintBagInfo: ptr = %p, Type = %d, size = %u, link = %p, # sub bags = %u\n",
           ps, (ps->bagFlagsType & TYPE_BIT_MASK), ps->bagSize, ps->bagLinkPtr, 
           NrHandles((ps->bagFlagsType & TYPE_BIT_MASK), ps->bagSize));
    return;
}

//  Walk the free master pointer chain.  This is the list of free master
//  pointers -- bag handles.  The contents of each link should point to the next
//  link in the chain.  A value that poits outside of the master pointer handles
//  area is invalid (except for 0 which indicates end of chain).

static void CheckFreeMptrList ( int arenanr )
{
    BagPtr_t start;
    BagStruct_t *ptr;
    UInt nFound = 0, nBad = 0, nFree = 0;
    ArenaBag_t *par = &MemArena[arenanr];
    
    // Walk the master pointer area to see if we're clean there
    start = par->FreeHandleChain;

    while (start) {
        if (*start >= par->OldBagStart && *start < par->AllocBagStart) {
            // found a link to a bag, count it
            nFound++;
            ptr = (BagStruct_t *)(*start - HEADER_SIZE);
            if (*start != &ptr->bagData) {
                // a bag handle points to something not recognizable or badly linked
                nBad++;
                printf("CheckFreeMptrList: Handle/bag link suspect: handle = %p, handle link = %p, bag data address = %p\n",
                       start, *start, &ptr->bagData);
            }
        }
        else if (*start >= par->BagHandleStart && *start < par->OldBagStart) {
            // it's a free Mptr handle
            nFree++;
        }
        else if (*start != 0) {
            // next free mptr in chain isn't valid -- not supposed to happen
            printf("CheckFreeMptrList: *** Invalid *** handle = %p contains %p; can't trace list further\n",
                   start, *start);
            break;
        }
        start = *start;
    }

    BagPtr_t beg, end;
	beg   = par->BagHandleStart;
	end   = par->OldBagStart;

	char msgb[100];
	sprintf(msgb, ", and %u suspect Handle/Bag linkages", nBad);
    printf("CheckFreeMptrList: Walked Arena #%d, FreeHandleChain: Found links to %u bags (%s), %u (%.1f%%) handles free%s\n",
           par->ArenaNumber, nFound, ((nFound == 0)? "GOOD" : "BAD"), nFree, 
           (100.0 * (float)nFree / (float)(((UInt)end - (UInt)beg) / sizeof(BagPtr_t *))),
		   (nBad > 0) ? msgb : "");
    return;
}

//  Walk the active master pointer list (bag handles).  Active bag handles point
//  to an area between OldBagSTart and AllocBagStart.  Check that the link
//  pointer in the bag is equal to the bag handle.

static void CheckMptrHandles ( int arenanr )
{
    BagPtr_t start;
    BagStruct_t *ptr;
    UInt nFound = 0, nBad = 0;
    ArenaBag_t *par = &MemArena[arenanr];
    
    // Walk the master pointer area, to ensure all used handles point to valid bags
    start = par->BagHandleStart;
    while (start < par->OldBagStart) {
        if (*start >= par->OldBagStart && *start < par->AllocBagStart) {
            // found a link to a bag, count it
            nFound++;
            ptr = (BagStruct_t *)(*start - HEADER_SIZE);
            if ( (*start != &ptr->bagData) || (ptr->bagLinkPtr != start) ) {
                // a bag handle points to something not recognizable or badly linked
                nBad++;
                printf("CheckMptrHandles: Handle/Bag link not valid: handle = %p, bag data ptr = %p\n",
                       start, *start);
				printf("                  Bag data address = %p, Bag Link Pointer = %p\n",
					   &ptr->bagData, ptr->bagLinkPtr);
            }
        }
        else {
            // it's a free Mptr handle -- checked by CheckFreeMptrList elsewhere -- noop
        }
        start++;
    }

    BagPtr_t beg, end;
	beg   = par->BagHandleStart;
	end   = par->OldBagStart;

	char msgb[100];
	sprintf(msgb, ", and %u suspect Handle/Bag linkages (%s)", nBad, ((nBad == 0)? "GOOD" : "BAD") );
    printf("CheckMptrHandles: Walked master pointers for Arena #%d: Found links to %u (%.1f%%) bags%s\n",
           par->ArenaNumber, nFound,
           (100.0 * (float)nFound / (float)(((UInt)end - (UInt)beg) / sizeof(BagPtr_t))),
           (nBad > 0) ? msgb : "" );
    
    return;
}

// Walk the actual bags -- assumptions is we're called after GC, so everything
// should cleanly setup with no holes from OldBagStart to AllocBagStart; at this
// point AllocBagStart should = YoungBagStart (i.e., no new bags have been
// created since GC done).

static void WalkBagPointers ( int arenanr )
{
    BagPtr_t start, end, foo;
    BagStruct_t *ptr;
    UInt type, nbad = 0, szbad = 0;
    UInt nFound = 0, szFound = 0, sizeCurr;
    ArenaBag_t *par = &MemArena[arenanr];
    
    start = par->OldBagStart; end = par->AllocBagStart;
    printf("WalkBagPointers: Arena #%d, BagHandleStart = %p, EndBags = %p\n",
		   par->ArenaNumber, par->BagHandleStart, par->EndBags);
	printf("    Walk OldBagStart   = %p to AllocBagStart = %p, used pool = %uk (%uMb)\n", 
           start, end, ((UInt)end - (UInt)start) / 1024, ((UInt)end - (UInt)start) / (1024 * 1024) );
    printf("         YoungBagStart = %p,   AllocBagStart = %p, free pool = %uk (%uMb)\n",
           par->YoungBagStart, par->AllocBagStart,
           ((UInt)par->EndBags - (UInt)par->AllocBagStart) / 1024,
           ((UInt)par->EndBags - (UInt)par->AllocBagStart) / (1024 * 1024) );
    
    while (start < (end - 1)) {            /* last mptr in list is 0 */
        ptr = (BagStruct_t *)start;
        type = ptr->bagFlagsType & TYPE_BIT_MASK;
        if (type < T_INT || type > T_FREEBAG) {
            sizeCurr = ptr->bagSize + sizeof(BagStruct_t) - sizeof(BagPtr_t);
            nbad++; szbad += sizeCurr;
            PrintBagInfo(ptr);
        }
        nFound++;
        sizeCurr = ptr->bagSize + sizeof(BagStruct_t) - sizeof(BagPtr_t);
        szFound += sizeCurr;
        start += (sizeCurr + sizeof(BagPtr_t) - 1) / sizeof(BagPtr_t);
        // link pointer should point to the bag handle, bag handle should
        // point back to the data pointer
        if (ptr->bagLinkPtr && par->BagHandleStart <= ptr->bagLinkPtr && ptr->bagLinkPtr < par->OldBagStart) {
            BagPtr_t foo = ptr->bagLinkPtr, bar = *ptr->bagLinkPtr;
            if (bar != &ptr->bagData) {
                // we're foobar'd
                printf("        Suspect misaligned handle/link pointer: link = %p, handle = %p, *handle = %p, data ptr = %p\n",
                       foo, bar, *bar, &ptr->bagData);
            }
        }
    }

    printf("    Walked the bags, found %u bags, size = %uk (%uMb)\n",
           nFound, szFound / 1024, szFound / (1024 * 1024));
	if (nbad > 0)
		printf("    Found %u suspect bags, total size = %uk (%uMb)\n",
			   nbad, szbad / 1024, szbad / (1024 * 1024));

    // Walk the remaining memory, which should be the free pool.  It should all
    // have been initialized to zeros... Things appear to be all messed up if
    // DirtyBags != 0
    start = par->AllocBagStart; end = par->EndBags; nbad = 0;
    while (start < end) {
        if (*start)
            nbad++;
        
        start++;
    }
    
    if (nbad)
        printf("    Walked the free area, %u non-zero values -- pool is dirty!\n", nbad);
    
    float ptrpc = 100.0 * (float)((UInt)par->OldBagStart - (UInt)par->BagHandleStart)  /
                          (float)((UInt)par->EndBags - (UInt)par->BagHandleStart),
        usedpc  = 100.0 * (float)((UInt)par->YoungBagStart - (UInt)par->OldBagStart) /
                          (float)((UInt)par->EndBags - (UInt)par->BagHandleStart),
        freepc  = 100.0 * (float)((UInt)par->EndBags - (UInt)par->AllocBagStart) /
                          (float)((UInt)par->EndBags - (UInt)par->BagHandleStart);
    
    printf("WalkBagPointers:  Total pool = %uMb (%.1f%%)\n",
           (((UInt)par->EndBags - (UInt)par->BagHandleStart) / (1024 * 1024)), 100.0);
	printf("             Master pointers = %uMb (%.1f%%)\n",
           (((UInt)par->OldBagStart - (UInt)par->BagHandleStart) / (1024 * 1024)), ptrpc);
	printf("                   Live Bags = %uMb (%.1f%%)\n",
           (((UInt)par->YoungBagStart - (UInt)par->OldBagStart) / (1024 * 1024)), usedpc);
    printf("                   Free pool = %uMb (%.1f%%)\n",
           (((UInt)par->EndBags - (UInt)par->AllocBagStart) / (1024 * 1024)), freepc);
    
    CheckMptrHandles(par->ArenaNumber);	// Check the used master pointers -- bag handles
    CheckFreeMptrList(par->ArenaNumber); // Check the free master pointers
    
    return;
}

/*
 *  Gather information about the bags (all, including dead & remnants) and
 *  build a histogram for output.  For very large bags print information for
 *  these (statistics are accumulated for all bags under 2,000 words in size),
 *  so we won't have a view into them otherwise.
 */

typedef struct {
    Int4    size;	                    /* size of the bag or remnant (size in bytes incl header) */
    Int4    count;						/* count of bags this size */
    Int4    nlive;						/* number of live bags */
    Int4    ndead;						/* number of dead bags */
    Int4    nremnant;					/* number of remants */
} BagHistogram_t;

typedef enum {
    INCREMENT_LIVE,						/* increment the live bags stats */
    INCREMENT_DEAD,						/* dead */
    INCREMENT_REMNANT,					/* remnant */
}  countType_t;

#define SIZE_HIST 2000					/* track up to 2000 different sizes */
static BagHistogram_t BagSizeCount[SIZE_HIST]; /* index by WORDS_BAG */
static Int4 countHistOn = 0;    

static void IncrementBagHistogram ( Int4 size_w, countType_t typ, BagPtr_t * bagp )
{
    // increment stats for this bag size; print info for bags >= SIZE_HIST words
    BagStruct_t *ptr = (BagStruct_t *)bagp;
    int bagtype = ptr->bagFlagsType & TYPE_BIT_MASK;
    int sz;

    if (size_w >= SIZE_HIST) {
        sz = ptr->bagSize + HEADER_SIZE * sizeof(BagPtr_t);
//        printf("Big bag: Type = %d, size = %db (%dW), # subbags =%d, Status = %s\n",
//               bagtype, sz, size_w, NrHandles(bagtype, ptr->bagSize),
//               (typ == INCREMENT_LIVE) ? "Live" : (typ == INCREMENT_DEAD) ? "Dead" :
//               (typ == INCREMENT_REMNANT) ? "Remnant" : "Halfdead" );
        // record the largest size...
        size_w = SIZE_HIST - 1;
        if (sz > BagSizeCount[size_w].size)
            BagSizeCount[size_w].size = sz;
    }
    else {
        BagSizeCount[size_w].size = sz = size_w * sizeof(BagPtr_t);
    }
    
    BagSizeCount[size_w].count++;
    InfoBags[bagtype].nrAll++;
    switch (typ) {
    case INCREMENT_LIVE:    {
        BagSizeCount[size_w].nlive++;
        InfoBags[bagtype].nrLive++;
        InfoBags[bagtype].sizeLive += sz;
        break;
    }
    case INCREMENT_DEAD:    { BagSizeCount[size_w].ndead++;        InfoBags[bagtype].nrDead++;          break; }
    case INCREMENT_REMNANT:    { BagSizeCount[size_w].nremnant++;    InfoBags[bagtype].nrRemnant++;      break; }
    }
    return;
}

static void DumpBagsHistogram ( void )
{
    // for each populated entry in the histogram table print the information
    Int4 ii;

//    printf("DumpBagsHistogram: Stats for all bags (dead/alive/etc.) found in current GC run\n");
//    printf("Size\tTotal\tLive\tDead\tRemnant\n");
    
//    for (ii = 0; ii < 2000; ii++) {
//        if (BagSizeCount[ii].count > 0)  { /* found a bag of this size */
//            printf("%7d\t%7d\t%7d\t%7d\t%7d\n",
//                   BagSizeCount[ii].size, BagSizeCount[ii].count, BagSizeCount[ii].nlive,
//                   BagSizeCount[ii].ndead, BagSizeCount[ii].nremnant);
//        }
//    }

    // for each populated entry in the InfoBags table print the information
    printf("\nDumpBagsHistogram: Stats by bag type found in current GC run\n");
    printf("Type\tTotal\tLive\tDead\tRemnant\tSize Live\tType Name\n");

    for (ii = 0; ii < NTYPES; ii++)   {
        if (InfoBags[ii].nrAll > 0)  {    /* skip types not active */
            if (InfoBags[T_RESIZE_FREE].name == (char *)NULL)  {
                InfoBags[T_RESIZE_FREE].name = "Resize remnant (free)";
            }
            printf("%7d\t%7d\t%7d\t%7d\t%7d\t%9u\t%s\n",
                   ii, InfoBags[ii].nrAll, InfoBags[ii].nrLive, InfoBags[ii].nrDead,
                   InfoBags[ii].nrRemnant, InfoBags[ii].sizeLive, InfoBags[ii].name);
        }
    }
    
    return;
}

//  Print information about the memory Arenas (called during GC when tracing is
//  on and by GetArenaStats GAP function).  When <flag> is true calculate/print
//  summary data about arena usage; otherwise, don't (during GC this data will
//  change).

static UInt CountFreeChain ( ArenaBag_t *par )
{
	UInt count = 0;
	BagPtr_t  head = par->FreeHandleChain;

	while (head != 0) {
		count++;
		head = *head;
	}
	
	return count;
}

static int  DumpMemArenaData ( int flag )
{
    ArenaBag_t *par = &MemArena[0];
	int         nrArenas;

    printf("\nDumpMemArenaData: Information about allocated memory arenas\n");
    
    for (par = &MemArena[0], nrArenas = 0; par->ActiveArenaFlag; par++, nrArenas++) {
        printf("Arena: # %d, Size = %u (MB), Active = %s, Full = %s, Free ratio threshold = %.2f\n",
               par->ArenaNumber, (par->SizeArena / (1024 * 1024)),
               (par->ActiveArenaFlag ? "True" : "False"),
			   (par->ArenaFullFlag ? "True" : "False"), par->FreeRatio);
        printf("BagHandleStart = %p, OldBagStart = %p, YoungBagStart = %p\n",
               par->BagHandleStart, par->OldBagStart, par->YoungBagStart);
        printf(" AllocBagStart = %p,    StopBags = %p,       EndBags = %p\n",
               par->AllocBagStart, par->StopBags, par->EndBags);
		if (flag) {
			UInt  nFree = CountFreeChain(par);
			UInt  nMax = ( (UInt)par->OldBagStart - (UInt)par->BagHandleStart ) / sizeof(UInt);
			UInt  sizepool = (UInt)par->EndBags - (UInt)par->OldBagStart;
			UInt  freepool = (UInt)par->EndBags - (UInt)par->AllocBagStart;
			
			printf("    Free Chain = %p, Entries avail = %u (of %u max) = %.1f%% free\n",
				   par->FreeHandleChain, nFree, nMax, (100.0 * (float)nFree / (float)nMax));
			printf("Allocated bags = %u, Size = %u Kbytes (%u Mbytes)\n",
				   (nMax - nFree), ((UInt)par->AllocBagStart - (UInt)par->OldBagStart) / 1024,
				   (((UInt)par->AllocBagStart - (UInt)par->OldBagStart) / (1024 * 1024)) );
			printf("Size Free pool = %u Kbytes (%u Mbytes), -- %.1f%% free\n",
				   ((UInt)par->EndBags - (UInt)par->AllocBagStart) / 1024,
				   (((UInt)par->EndBags - (UInt)par->AllocBagStart) / (1024 * 1024)),
				   (100.0 * (float)freepool / (float)sizepool) );
		}
		else {
			printf("    Free Chain = %p, Marked bags = %p, # on Marked chain = %u\n\n",
                   par->FreeHandleChain, par->MarkedBagChain, par->nrMarkedBags);
		}
    }

    return nrArenas;
}


static void PrintArenaInfo ( ArenaBag_t *par, int eflg )
{
    if (SyMemMgrTrace > 0) {
        printf("PrintArenaInfo%s: Memory Arena #%d\n", (eflg ? ":On Entry" : ""), par->ArenaNumber);
        printf("    BagHandleStart = %p, OldBagStart   = %p, size Mptr area  = %uk (%uMb), # mptrs = %u\n",
               par->BagHandleStart, par->OldBagStart, 
               ((UInt)par->OldBagStart - (UInt)par->BagHandleStart) / 1024,
               (((UInt)par->OldBagStart - (UInt)par->BagHandleStart) / (1024 * 1024)),
               ((UInt)par->OldBagStart - (UInt)par->BagHandleStart)/  sizeof(BagPtr_t));
        printf("    OldBagStart    = %p, YoungBagStart = %p, size Old Bags   = %uk (%uMb)\n",
               par->OldBagStart, par->YoungBagStart,
               ((UInt)par->YoungBagStart - (UInt)par->OldBagStart) / 1024,
               (((UInt)par->YoungBagStart - (UInt)par->OldBagStart) / (1024 * 1024)));
        printf("    YoungBagStart  = %p, AllocBagStart = %p, size Young Bags = %uk (%uMb)\n",
               par->YoungBagStart, par->AllocBagStart,
               ((UInt)par->AllocBagStart - (UInt)par->YoungBagStart) / 1024,
               (((UInt)par->AllocBagStart - (UInt)par->YoungBagStart) / (1024 * 1024)));
        printf("    AllocBagStart  = %p, StopBags      = %p, EndBags = %p, Alloc pool area = %uk (%uMb)\n",
               par->AllocBagStart, par->StopBags, par->EndBags,
			   (   (UInt)par->EndBags - (UInt)par->AllocBagStart) / 1024,
               ( ( (UInt)par->EndBags - (UInt)par->AllocBagStart) / (1024 * 1024) ) );
        countHistOn = 1;
        fflush(stdout);
    }
    else {
        countHistOn = 0;
    }
    return;
}

static void ResetCountInfoStats ( void )
{
    if (countHistOn > 0)   {
        Int4 ii;
        for (ii = 0; ii < SIZE_HIST; ii++) {
            BagSizeCount[ii].size = BagSizeCount[ii].count = BagSizeCount[ii].nlive = 0;
            BagSizeCount[ii].ndead = BagSizeCount[ii].nremnant = 0;
        }
        // Use InfoBags to count types used -- clear at start of each GC
        for (ii = 0; ii < NTYPES; ii++) {
            InfoBags[ii].nrAll = InfoBags[ii].nrLive = InfoBags[ii].nrDead = InfoBags[ii].nrRemnant = 0;
            InfoBags[ii].sizeLive = InfoBags[ii].sizeAll = 0;
        }
    }
    return;
}

//  We have marked bag chains for each arena which consist of the references
//  found when testing the global bags and those bags found on the stack.  Walk
//  the marked chains for each Arena to mark the subbags -- some bags in an
//  Arena may have subbags in a different Arena and we need to catch and tag
//  these also

static void MarkBagsByArena ( UInt *nrBags, UInt *szBags )
{
    ArenaBag_t *par;
    BagPtr_t    first;

again:    
    par = &MemArena[0];
    while ( par->ActiveArenaFlag ) {                    // loop over the active arenas
        while ( par->MarkedBagChain != 0 ) {
            first = (BagPtr_t)((UInt)par->MarkedBagChain & ~((UInt)(sizeof(UInt) - 1))); // ensure no mark bits set
            par->MarkedBagChain = GET_LINK_BAG(first);
            SET_LINK_BAG(first, MARKED_ALIVE(first));
            (*TabMarkFuncBags[GET_TYPE_BAG(first)])( first );
            *nrBags += 1;
            *szBags += GET_SIZE_BAG(first) + HEADER_SIZE * sizeof(BagPtr_t *);
        }
        par++;
    }

	//  Marking the bags for an arena may add entries to the chain for a
	//  different arena (e.g., in the case where entries are added to a list in
	//  Arena A, but because A is full the new bags must go into the currently
	//  active arena.  After finishing checking all arenas check if any chain
	//  has entries... process until all are empty
    par = &MemArena[0];
    while ( par->ActiveArenaFlag ) {
        if ( par->MarkedBagChain != 0 ) {
            goto again;
        }
        par++;
    }
    
    return;
}

//  Compute the total memory size of all arenas
static UInt GetTotalArenaSize ( void )
{
    ArenaBag_t *par;
	UInt tsize = 0;

    par = &MemArena[0];
    while ( par->ActiveArenaFlag ) {                    // loop over the active arenas
		tsize += par->SizeArena;
        par++;
    }
	return tsize;
}

static int  nrGlobalBagsFound = 0;		// keep tabs on how many global bags we find

/****************************************************************************
**
**  gcMsgBuff is a static buffer to compose a one-line summary message to output
**  simple statistics from garbage collection when "message" is on [see
**  GASMAN("message")].  The message is composed as each phase of GC completes
**  (hence the use of a buffer) and output after the check phase.  The message
**  contains: Marking code (#GC), the Arena number,
**      number of live bags / size of live bags (KBytes) -- from Mark phase
**      number of dead bags / size of dead bags (KBytes) -- from sweep phase
**      free space (KBytes) / Arena size (KBytes) and % free -- from check phase
**
**  a sample output message (output is enabled when SyMsgsFlagBags is > 0) looks like:
**
**  #GC Arena:0  2062514 / 120186kb (live)    829 /  60kb (dead)  108621kb / 262144kb (41.4%) free
*/

static char gcMsgBuff[200];				// compose text 


//  Find and mark all bags that are referenced, or are children of bags with
//  references.  A reference is defined as the a bag is registered in the gloabl
//  array of bags or is found somewhere in the stack or registers

static void MarkAllLiveBags ( ArenaBag_t *par )
{
    UInt         i;						// loop variable
    BagPtr_t     first;					// bag handle from chains
    UInt         nrBags;                // number of live bags found
    UInt         szBags;                // size of live bags
    int          nrgbags;				// number of global bags found this run

    if (par->MarkedBagChain != 0) {		// sanity check
        printf("Collectbags: MarkedBagChain (%p) for Arena #%d is non-zero, claims %d links\n",
               par->MarkedBagChain, par->ArenaNumber, par->nrMarkedBags);
    }
    
    par->MarkedBagChain = 0;
    par->nrMarkedBags = 0;

    for ( i = 0; i < GlobalBags.nr; i++ ) // get all the global bag references
        MARK_BAG( *GlobalBags.addr[i] );
        
    if (SyMemMgrTrace > 0)
        printf("CollectBags: Building Marked bags list: MARK_BAG marked %d global bags\n",
               par->nrMarkedBags);
        
    /* mark from the stack */
    setjmp( RegsBags );
    GenStackFuncBags();

    if (SyMemMgrTrace > 0) {
        printf("Collectbags: Marked bags: After stack search, MARK_BAG total = %d bags\n",
               par->nrMarkedBags);
        printf("Collectbags: Check Changed bags, mark subbags of all Bags\n");

        if (nrGlobalBagsFound != 0) {
            // compare number found this run with prior run
            int drun = (int)par->nrMarkedBags - nrGlobalBagsFound;
            drun = (drun < 0) ? -drun : drun ;
            float delt = (float)drun / (float)nrGlobalBagsFound;
            if (delt > 0.05) {
                // print warning if delta is > 5% (just a guess)
                printf("Collectbags: delta on number globals found > 5%%, current run = %d, prior = %d\n",
                       par->nrMarkedBags, nrGlobalBagsFound);
            }
        }
        else {
            // first time thru, just set nrGlobalBagsFound
            nrGlobalBagsFound = par->nrMarkedBags;
        }
    }

    /* we've now found [or should have] all live bags... */
    if (SyMemMgrTrace > 0) {
        printf("Collectbags: Have list of all global live bags, found = %d\n", par->nrMarkedBags);
        printf("Collectbags: Start tagging the live bags...\n");
    }

    nrBags = 0;
    szBags = 0;
    MarkBagsByArena( &nrBags, &szBags );
    
    if (SyMemMgrTrace > 0) {
        printf("CollectBags: After searching subbags, all live bags now = %d\n",
               par->nrMarkedBags);
        printf("CollectBags: marked alive = %d, size of alive bags = %d\n", nrBags, szBags);
        fflush(stdout);
    }

    // Information after the mark phase
    SizeLiveBags += szBags;
	SizeAllArenas = GetTotalArenaSize();
    if (SyMsgsFlagBags > 0)
        sprintf((gcMsgBuff + strlen(gcMsgBuff)), "%9u /%8ukb (live)  ", nrBags, szBags/1024);

    return;
}

//  During GC, after all live bags are consolidated to one contiguous area we
//  must clear (zero) the free pool area

static void ClearFreePoolArea ( ArenaBag_t *par, BagPtr_t *src )
{
    BagPtr_t    *dst;
    
    if (SyMemMgrTrace > 0)    
        printf("Collectbags: Clear free area = %s\n", (DirtyBags? "No" : "Yes"));

    par->AllocBagStart = par->YoungBagStart = src; // Adjust free storage pointers

    // clear the new free area
    if ( ! DirtyBags ) {
        dst = par->EndBags;
        if (SyMemMgrTrace > 0)
            printf("Collectbags: Clear from = %p, to = %p; clear %u (%uk / %uMb) bytes\n",
                   src, dst, ((UInt)dst - (UInt)src), ((UInt)dst - (UInt)src) / 1024,
                   (((UInt)dst - (UInt)src) / (1024 * 1024)));
        while ( src < dst )
            *src++ = 0;
    }
    return;
}

// The Arena has been marked -- we should have (a) live bags, (b) resize
// remanants, and (c) dead bags.  Consolidate all the live bags at the beginning
// of the bag data area (updating the bag header values as required).  All the
// remaining area is then 1 large free pool.

static void SweepArenaBags ( ArenaBag_t *par )
{
    BagPtr_t *      dst;				/* destination in sweeping         */
    BagPtr_t *      src;				/* source in sweeping              */
    BagPtr_t *      end;				/* end of a bag in sweeping        */
    Int4            isz;
    
    dst = par->YoungBagStart;
    src = par->YoungBagStart;

    UInt nLive, nDead, nRemnant, szLive, szDead, szRemnant;
    nLive = nDead = nRemnant = szLive = szDead = szRemnant = 0;

    PrintArenaInfo( par, 0 );

    // Walk all bags:  A bag must be either: alive, dead, or resize free
    UInt nbagsCheck = 0;
    while ( src < par->AllocBagStart ) {
        nbagsCheck++;
        if ( GET_TYPE_PTR(src) == T_RESIZE_FREE ) {
            // leftover remnant of a resize of <n> bytes
            // Move source pointer (dest stays put)
            if (TEST_FLAG_PTR(src, BF_COPY) ) {            // one-word remant
                src++;
                nRemnant++; szRemnant += sizeof(BagPtr_t *);
                if (countHistOn)
                    IncrementBagHistogram(1, INCREMENT_REMNANT, src);
            }
            else {
                src += 1 + WORDS_BAG( GET_SIZE_PTR(src) ); // multi-word remnant 
                nRemnant++; szRemnant += GET_SIZE_PTR(src) + sizeof(BagPtr_t *);
                if (countHistOn) {
                    isz = 1 + WORDS_BAG( GET_SIZE_PTR(src) );
                    IncrementBagHistogram(isz, INCREMENT_REMNANT, src);
                }
            }
        }

        else if ( ((UInt)GET_LINK_PTR(src)) % sizeof(BagPtr_t) == 0 ) {
            // Dead bag -- no markers (least significant bits) are set
            nDead++; szDead += GET_SIZE_PTR(src) + HEADER_SIZE * sizeof(BagPtr_t *);
            if (countHistOn) {
                isz = HEADER_SIZE + WORDS_BAG( GET_SIZE_PTR(src) );
                IncrementBagHistogram(isz, INCREMENT_DEAD, src);
            }

#ifdef  COUNT_BAGS
            InfoBags[GET_TYPE_PTR(src)].nrLive -= 1; // update the statistics
            InfoBags[GET_TYPE_PTR(src)].sizeLive -= GET_SIZE_PTR(src);
#endif
            // put the bag on [the head of] the free list
            BagPtr_t nfhead = GET_LINK_PTR(src);
            if (nfhead >= par->BagHandleStart && nfhead < par->OldBagStart) {
                // link is valid
                *nfhead = par->FreeHandleChain;
                par->FreeHandleChain = nfhead;
            }
            else {
                printf("CollectBags: Bad link from dead bag (%p) pushed on Free chain, = %p\n",
                       src, nfhead);
                // char * msg = GuMakeMessage("CollectBags: Bad link for bag going on Free chain, = %p\n", nfhead);
                // SyAbortBags(msg);
            }

            src += HEADER_SIZE + WORDS_BAG( GET_SIZE_PTR(src) ) ; // Advance src
        }

        else if ( ((UInt)(GET_LINK_PTR(src))) % sizeof(BagPtr_t) == 1 )  {
            // live bag -- Link pointer has its least significant bit set
            nLive++; szLive += GET_SIZE_PTR(src) + HEADER_SIZE * sizeof(BagPtr_t *);

            if (countHistOn) {
                isz = HEADER_SIZE + WORDS_BAG( GET_SIZE_PTR(src) );
                IncrementBagHistogram(isz, INCREMENT_LIVE, src);
            }

            // update identifier, copy flags-type and link fields
			SET_PTR_BAG( (UNMARKED_ALIVE(GET_LINK_PTR(src))), (BagPtr_t*) DATA_PTR(dst) );
            end = src + HEADER_SIZE + WORDS_BAG( GET_SIZE_PTR(src) ) ;

            COPY_HEADER(dst, src);
            SET_LINK_PTR(dst, UNMARKED_ALIVE(GET_LINK_PTR(src)));

            dst += HEADER_SIZE;
            src += HEADER_SIZE;

            if ( dst != src ) {                        // copy data area
                while ( src < end )
                    *dst++ = *src++;
            }
            else {
                dst = end;
                src = end;
            }
        }

        else {
            // Some sort of invalid header encountered
            printf("Collectbags: Panic -- memory manager found bogus header (%p) -- exiting\n", src);
            // exit(1);
        }
    }

    if (SyMemMgrTrace > 0)
        printf("CollectBags: Swept all bags, checked %u bags\n", nbagsCheck);

    ClearFreePoolArea(par, dst);
    par->SweepNeeded  = 0;				// Arena has been swept
	par->nrMarkedBags = 0;

    // Since doing full GC have now processed all bags 
    if (SyMemMgrTrace > 0)  {
        printf("CollectBags: Arena #%d: Processed all bags...found:\n", par->ArenaNumber);
		printf("#     Live Bags = %10u, size     Live Bags = %10u (%uk / %uMb)\n",
               nLive, szLive, szLive / 1024, szLive / (1024 * 1024));
        printf("#     Dead Bags = %10u, size     Dead Bags = %10u (%uk / %uMb)\n",
               nDead, szDead, szDead / 1024, szDead / (1024 * 1024));
        printf("#  Remnant Bags = %10u, size  Remnant Bags = %10u (%uk / %uMb)\n",
               nRemnant, szRemnant, szRemnant / 1024, szRemnant / (1024 * 1024));
        fflush(stdout);
    }

    // Information after the sweep phase (print # dead bags & size of dead bags in kBytes)
    if (SyMsgsFlagBags > 0)
        sprintf((gcMsgBuff + strlen(gcMsgBuff)), "%9u /%8ukb (dead)  ", nDead, szDead/1024);

    return;
}

static void CheckArenaBags ( ArenaBag_t *par )
{
    // Called after sweeping the arena to consoldidate used and free pool space
    // Start checking..
    BagPtr_t   *p;						// loop variable
    
    if (SyMemMgrTrace > 0)   {
        printf("Collectbags: Start checking phase ...\n");
        
        WalkBagPointers(par->ArenaNumber);

        printf("CollectBags: After looping Master pointers...\n");
        PrintArenaInfo(par, 0);
    }

    // information after the check phase (print free and total size of arena + percentage free)
    if (SyMsgsFlagBags > 0) {
        UInt pfr, pwh;
        float frpc;
        pfr = ((char*)par->EndBags - (char*)par->StopBags) / 1024;
        pwh = ((char*)par->EndBags - (char*)par->BagHandleStart) / 1024;
        frpc = (float)(100.0 * (float)pfr / (float)pwh);
        sprintf((gcMsgBuff + strlen(gcMsgBuff)), "%8ukb / %8ukb (%.1f%%%%) free\n", pfr, pwh, frpc);
        fprintf(stderr, gcMsgBuff);
        fflush(stderr);
    }

    if (SyMemMgrTrace > 0) {
        printf("CollectBags: Arena #%d: EndBags < StopBags: %s, EndBags = %p, StopBags = %p\n",
               par->ArenaNumber, ((par->EndBags < par->StopBags)? "Yes" : "No"),
               par->EndBags, par->StopBags);
        printf("Number of times ResizeBag() called since last GC: %u\n", nrResizeBags);
        nrResizeBags = 0;
    }

    return;
}

static void UnmarkArenaBags( void )
{
    // Bags determined to be alive were pushed on the MarkedBagChain for each
    // arena, these chains were then traversed (recursed) to find all children
    // (subbags).  There is no chain or linkage to track all such found bags;
    // rather they are marked (their link pointer is altered with MARK_ALIVE
    // macro to set the least significant bit [LSB]).  The only way to clear
    // these is to walk the active bags and adjust the link pointer [use
    // UNMARKED_ALIVE to clear the LSB].  This must be done or future calls to
    // MARK_BAG will incorrectly think the bag (and its children) are marked
    // alive.
    
    ArenaBag_t *par = &MemArena[0];

    while (par->ActiveArenaFlag) {		// loop over active Arenas
        if (par->SweepNeeded != 0) {	// Only sweep arena needing it
            BagPtr_t     handle;		// bag handle (from BagHandleStart to OldBagStart)
            BagStruct_t *ptr;           // Bag data structure
    
            // Walk the master pointers, ensure all used bag handles are "back referenced" by the link pointer
            handle = par->BagHandleStart;
            while (handle < par->OldBagStart) {
                if (par->OldBagStart <= *handle && *handle < par->AllocBagStart) {
                    ptr = (BagStruct_t *)(*handle - HEADER_SIZE);
                    if (handle == UNMARKED_ALIVE(ptr->bagLinkPtr)) {
                        SET_LINK_BAG(handle, UNMARKED_ALIVE(ptr->bagLinkPtr));
                    }
                }
                handle++;
            }
            par->SweepNeeded = 0;
            par->nrMarkedBags = 0;
            par->MarkedBagChain = 0;
        }
        par++;
    }

    return;
}


//  Loop thru the existing 'full' arenas and GC each until we either
//  find one now occupied below the GASMAN_OCCUPIED_RATIO or we've
//  checked all full & active arenas.  Do NOT GC the 'current'
//  <arena_nr>, as it will have just been cleared by CollectBags() --
//  which is the only caller of this function

static int      ReopenOldArena (char arena_nr)
{
	ArenaBag_t *par;

	for (par = &MemArena[0]; par->ActiveArenaFlag; par++) {
		if (par->ArenaFullFlag && par->ArenaNumber != arena_nr) {
			// test how full the arena is...see if we can use w/o GC first
			UInt psiz = (char *)par->EndBags - (char *)par->OldBagStart;
			UInt fsiz = (char *)par->EndBags - (char *)par->AllocBagStart;

			if ((float)fsiz / (float)psiz < UnlockRatioThreshold) {
				// Do a GC to get accurate numbers...
				CollectBags( 0, par->ArenaNumber, "ReopenOldArena()" );
			}

			// re-test how full the arena is...
			psiz = (char *)par->EndBags - (char *)par->OldBagStart;
			fsiz = (char *)par->EndBags - (char *)par->AllocBagStart;
			if ((float)fsiz / (float)psiz >= UnlockRatioThreshold) {
				par->ArenaFullFlag = 0;
				actAR = par->ArenaNumber;		 // update 'current' active arena
				return (int)par->ArenaNumber;
			}
		}
	}
	return -1;
}

//  Gather information about the cumulative time spent performing GC
//  so GAP program total run time vs time in GC can be compared.

TimeAnalyze_t    GapRunTime;


UInt CollectBags ( UInt size, int spec_arena, char *caller_id )
{
    int             inewa = -1;			// new arena number if required
    ArenaBag_t     *par;                // Memory Arena pointer

	GapRunTime.gc_in = clock();			// start timing GC run
	GapRunTime.nrGCRuns++;				// increment number of times run
	
    if (SyMemMgrTrace > 0)
        printf("CollectBags: On Entry: GC for %s (%d)\n", (spec_arena < 0) ? "general" : "Arena #", spec_arena);

    par = &MemArena[0];
    while (par->ActiveArenaFlag) {        // loop over the active memory arenas
        if (spec_arena >= 0) {
            // GC the specific arena only
            par += spec_arena;
            if (SyMemMgrTrace > 0)
                printf("CollectBags: Perform GC for specific Arena #%d (called from %s)\n", spec_arena, caller_id);
        }
        else {
            // General GC, don't process arenas flagged as full
            if (par->ArenaFullFlag) {
                par++;
                continue;
            }
        }

        PrintArenaInfo(par, 1);			// Print arena stats
        ResetCountInfoStats();			// Clear Information and Bag histogram trackers

        // Every collection is a "full" GC, so every bag is considered to be a young bag
        par->YoungBagStart = par->OldBagStart;
        SizeLiveBags = 0;
		SizeAllArenas = 0;

        // information at the beginning of garbage collections
        if (SyMsgsFlagBags > 0)
            sprintf(gcMsgBuff, "#GC Arena:%d  ", par->ArenaNumber);

        /* * * *  mark phase  * * * */
        MarkAllLiveBags(par);

        /* * * *  sweep phase  * * * */

        SweepArenaBags(par);
        
        /* * * *  check phase  * * * */
    
        /* temporarily store in 'par->StopBags' where this allocation takes us      */
		if (par->AllocBagStart + HEADER_SIZE + WORDS_BAG(size) > par->EndBags) {
			if (spec_arena >= 0 ) {
				// not enough room in arena to resize the bag
				printf("No free space in Arena %d to ResizeBag(), maybe try larger Arena size...exiting\n",
					   par->ArenaNumber);
				exit(1);
			}
			else
				par->StopBags = par->EndBags;
		}
		else            
			par->StopBags = par->AllocBagStart + HEADER_SIZE + WORDS_BAG(size);

        CheckArenaBags(par);

        if (spec_arena < 0) {
            // Check amount of memory between par->StopBags and par->EndBags
            // (alloc pool) If this drops below 'FreeRatio' defined for the
            // arena (default value GASMAN_FREE_RATIO) of available space
            // (par->EndBags - par->OldBagStart) then we're out of usable
            // memory...allocate another arena.
            if ((par->EndBags - par->StopBags) < ((par->EndBags - par->OldBagStart) * par->FreeRatio)) {
				if (SyMemMgrTrace > 0) {
					printf("GAP - CollectBags: After GC and free pool to allocate bags is too small.\n");
					printf("MemArena[%d]: Pool size = %uk, Used = %uk, Free = %uk\n",
						   par->ArenaNumber, ((UInt)par->EndBags - (UInt)par->OldBagStart) / 1024,
						   ((UInt)par->StopBags - (UInt)par->OldBagStart) / 1024,
						   ((UInt)par->EndBags - (UInt)par->StopBags) / 1024);
					printf("Allocate another memory arena...\n");
				}
                // set flag in the current memory arena...
                par->ArenaFullFlag = 1;
                // AddMemoryArena adds another pool...
				// *** Future option: Double SyStorMin each new allocation until say max of 4 GByte
				// Unmark the bags and do GC on existing full arenas (one might open up)
				UnmarkArenaBags();
				if (ReopenOldArena( par->ArenaNumber ) >= 0) {
					break;							 // existing arena opened up
				}
                inewa = AddMemoryArena( SyStorMin ); // add a new arena
                if (inewa < 0) {
                    printf("Not enough memory to continue...exiting\n");
                    exit(1);
                }
                // new arena allocated ... done; break from loop
                break;
            }
            else {
                // Could test here for a threshold to clear arena full flag (future)
                // par->ArenaFullFlag = 0;
            }
        }

        // set StopBags to EndBags
        par->StopBags = par->EndBags;

        if (spec_arena >= 0) {
            break;                      // only do the single specified arena
        }
        par++;
    }                                   // loop over each memory arena

    UnmarkArenaBags();
    
    if (countHistOn) {
        DumpMemArenaData( 0 );
        //  DumpBagsHistogram();
    }

	GapRunTime.gc_out = clock();
	GapRunTime.gc_cumulative += ( GapRunTime.gc_out - GapRunTime.gc_in );
	
    // return success
    return 1;
}


void    PrintRuntimeStats ( TimeAnalyze_t *ptim)
{
	//  print out information about the Memory Arenas allocated and the timing
	//  information collected
	DumpMemArenaData( 1 );
	
	double dtime, gctime;				// times in seconds
	ptim->gap_end = clock();
	dtime  = (double)(ptim->gap_end - ptim->gap_start) / CLOCKS_PER_SEC;
	gctime = (double)(ptim->gc_cumulative) / CLOCKS_PER_SEC;
	printf("Spiral total run time = %.1f seconds\n", dtime);
	if (ptim->nrGCRuns > 0) {
		printf("              GC time = %.1f seconds (avg time per iteration = %.4f)\n",
			   gctime, gctime / ptim->nrGCRuns);
		printf("   GC # of iterations = %d and consumed %.1f%% of the total time\n",
			   ptim->nrGCRuns, 100 * gctime / dtime);
	}
	else
		printf("      No GC since last stats init\n");

	return;
}


/* 
** Utility functions to get or set information related to memory management
**
*F FunGetArenaStats(<bag>) . . . . . . . . . . .  Get Arena Memory Statistics
*F FunSetArenaSize(<size>) . . . . . . . .  Set Arena Memory to <size> Bbytes
*F FunGetRuntimeStats(<bag>) . . . . . . . . . . . . . . .  Get Runtime Stats
*F FunResetRuntimeStats(<bag>) . . . . . . . . . . . . .  Reset Runtime Stats
*F FunSetArenaFreeRatio(<ratio>) . . . . . . . Set Arena Free Ratio Threshold
*F FunGCAllArenas(<bag>) . . . . . . . . . . . . . . Perform GC on all Arenas
**
*/

BagPtr_t        FunGetArenaStats (BagPtr_t hdCall)
{
	BagPtr_t    hdRet;					// return value (# of arenas)
	int         nrArenas;

	nrArenas = DumpMemArenaData( 1 );
	hdRet = INT_TO_HD (nrArenas);
	
	return hdRet;
}

BagPtr_t        FunSetArenaSize (BagPtr_t size)
{
	// Accepts an argument to specify (in MBytes) the size of future memory arenas
	
    BagPtr_t    hdCmd;					// handle of an argument
	char *usageMessage =
		"usage: SetArenaSize ( <size> )    # <size> = Arena size in MBytes";
	int memsiz;
	
    // check the argument
    if ( GET_SIZE_BAG(size) == SIZE_HD )
        return Error(usageMessage, 0, 0);

	// evaluate and check the command
	hdCmd = EVAL( PTR_BAG(size)[1] );
	if ( ! IS_INTOBJ(hdCmd) )
		return Error(usageMessage, 0, 0);

	memsiz = HD_TO_INT(hdCmd) * 1024;	// convert to Kbyte
	SyStorMin = memsiz;
	
	return HdVoid;
}

BagPtr_t        FunGetRuntimeStats (BagPtr_t hdCall)
{
	PrintRuntimeStats ( &GapRunTime );
	
	return HdVoid;
}

BagPtr_t        FunResetRuntimeStats (BagPtr_t hdCall)
{
	// reset all the time tracking values
	GapRunTime.gap_start = clock();			// start timing GC run
	GapRunTime.gap_end = 0;
	GapRunTime.gc_in = GapRunTime.gc_out = GapRunTime.gc_cumulative = 0;
	GapRunTime.nrGCRuns = 0;

	return HdVoid;
}

BagPtr_t        FunSetArenaFreeRatio (BagPtr_t ratio)
{
    BagPtr_t    hdCmd;					// handle of an argument
	char *usageMessage =
		"usage: SetArenaFreeRatio(<ratio>)    # free ratio threshold: 0.1 < ratio < 0.6";
	double *memsiz;
	
    // check the argument
    if ( GET_SIZE_BAG(ratio) == SIZE_HD )
        return Error(usageMessage, 0, 0);

	// evaluate and check the command
	hdCmd = EVAL( PTR_BAG(ratio)[1] );
	if ( GET_TYPE_BAG(hdCmd) != T_DOUBLE )
		return Error(usageMessage, 0, 0);

	memsiz = (double *)PTR_BAG(hdCmd);
	if ( 0.1 <= *memsiz && *memsiz <= 0.6 ) {
		// acceptable range, set value in FreeRatioThreshold *and* current active arena
		ArenaBag_t *par = &MemArena[0];

		while (par->ActiveArenaFlag && par->ArenaFullFlag)
			par++;						// skip past the full arenas to get to current active one

		par->FreeRatio = (float)(*memsiz);
		FreeRatioThreshold = (float)(*memsiz);
	}
	else 
		return Error(usageMessage, 0, 0);

	return HdVoid;
}
			
BagPtr_t        FunGCAllArenas (BagPtr_t hdCall)
{
	// Perform GC on all areans
	ArenaBag_t *par = &MemArena[0];
	int         nrArenas;
	BagPtr_t    hdRet;					// return value (# of arenas)

    for (par = &MemArena[0], nrArenas = 0; par->ActiveArenaFlag; par++, nrArenas++) {
		if (par->ArenaFullFlag) {		// not currently active arena
			CollectBags( 0, par->ArenaNumber, "GCAllArenas()" );
		}
	}
	CollectBags( 0, -1, "GCAllArenas()" ); // General GC -- operates on current active

	hdRet = INT_TO_HD (nrArenas);
	return hdRet;
}

void InitMemMgrFuncs (void)
{
	InstIntFunc ( "GetArenaStats",      FunGetArenaStats      );
	InstIntFunc ( "SetArenaSize",       FunSetArenaSize       );
	InstIntFunc ( "GetRuntimeStats",    FunGetRuntimeStats    );
	InstIntFunc ( "ResetRuntimeStats",  FunResetRuntimeStats  );
	InstIntFunc ( "SetArenaFreeRatio",  FunSetArenaFreeRatio  );

	InstIntFunc ( "GCAllArenas",        FunGCAllArenas  );

	return;
}


/****************************************************************************
**
**  This next block of functions are the basic building block functions to get
**  or update values to/from bags.  Some expect a bag handle (XXX_BAG) and
**  others expect the bag [data] pointer (XXX_PTR)
**
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
*F  ELM_BAG(<bag>, <i>) . . . . . . . . . . . [get] the <i>-th element of bag
*F  SET_ELM_BAG(<bag>, <i>, <elm>)  . . . . . . set the <i>-th element of bag
*F  GET_LINK_PTR(<ptr>) . . . . . .  get link pointer from the bag [data] ptr
*F  SET_LINK_PTR(<ptr>, <val>)  . .  set link pointer from the bag [data] ptr
*/

UInt IS_BAG ( BagPtr_t bag )
{
    ArenaBag_t *par = &MemArena[0];
    BagPtr_t *datap;                    // ptr to data part of bag

    if ( bag == 0 || (((UInt)bag & (sizeof(BagPtr_t)-1)) != 0) )
        return 0;						// not a valid bag handle
    
    while (par->ActiveArenaFlag) {		// Allocated arenas have Active Flag set
        if ( par->BagHandleStart <= bag && bag < par->OldBagStart) {
            // bag is in this arena
            datap = (*(BagPtr_t **)(bag));
            if (datap < par->OldBagStart || datap >= par->EndBags) {
                // bag data is not in the allocated area of this arena
                printf("IS_BAG: bag = %p, Arena #%d, **issue** bag data %p is not in allocated area of arena\n",
                       bag, par->ArenaNumber, datap);
            }
            return ( par->OldBagStart < datap && datap <= par->AllocBagStart);
        }
        par++;
    }
    
    return 0;
}

// Get the flags associated to a bag
UInt GET_FLAGS_BAG(BagPtr_t bag)
{
    BagPtr_t *p = (*(BagPtr_t **)(bag)); /* PTR_BAG(bag); */
    BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
    return ptr->bagFlagsType & BF_ALL_FLAGS;
}

// Test if a flag is associated to a bag
UInt GET_FLAG_BAG(BagPtr_t bag, UInt val)
{
    BagPtr_t *p = (*(BagPtr_t **)(bag)); /* PTR_BAG(bag); */
    BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
    return ptr->bagFlagsType & val;
}

// Clear the flags associated to a bag 
void BLANK_FLAGS_BAG(BagPtr_t bag)
{
    BagPtr_t *p = (*(BagPtr_t **)(bag)); /* PTR_BAG(bag); */
    BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
    UInt flagType = ptr->bagFlagsType & TYPE_BIT_MASK;
    ptr->bagFlagsType = flagType;
    return;
}

// Clear a flag associated to the bag
void CLEAR_FLAG_BAG(BagPtr_t bag, UInt val)
{
    BagPtr_t *p = (*(BagPtr_t **)(bag)); /* PTR_BAG(bag); */
    BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
    UInt flagType = ptr->bagFlagsType & ~(val);
    ptr->bagFlagsType = flagType;
    return;
}

// Set the flags associated to a bag
void SET_FLAG_BAG(BagPtr_t bag, UInt val)
{
    BagPtr_t *p = (*(BagPtr_t **)(bag)); /* PTR_BAG(bag); */
    BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
    UInt flagType = ptr->bagFlagsType | val;
    ptr->bagFlagsType = flagType;
    return;
}

// Get the copy pointer associated to a bag
BagPtr_t GET_COPY_BAG(BagPtr_t bag)
{
    BagPtr_t *p = (*(BagPtr_t **)(bag)); /* PTR_BAG(bag); */
    BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
    return ptr->bagCopyPtr;
}

// Set the copy pointer associated to a bag
void SET_COPY_BAG(BagPtr_t bag, BagPtr_t val)
{
    BagPtr_t *p = (*(BagPtr_t **)(bag)); /* PTR_BAG(bag); */
    BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
    ptr->bagCopyPtr = val;
    return;
}

// Get the link pointer associated to a bag
BagPtr_t GET_LINK_BAG(BagPtr_t bag)
{
    BagPtr_t *p = (*(BagPtr_t **)(bag)); /* PTR_BAG(bag); */
    BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);

#ifdef DEBUG_POINTERS
    // test link ptr
    ArenaBag_t *par = GetArenaFromBag(bag);
    if (ptr->bagLinkPtr != 0 && (ptr->bagLinkPtr < par->BagHandleStart || ptr->bagLinkPtr >= par->StopBags) ) {
        printf("GET_LINK_BAG: Bag has invalid link ptr [outside Arena #%d], = %p (%p)\n",
               par->ArenaNumber, bag, ptr->bagLinkPtr);
    }
#endif

    return ptr->bagLinkPtr;
}

// Set the link pointer to be associated to a bag
void SET_LINK_BAG(BagPtr_t bag, BagPtr_t val)
{
#ifdef DEBUG_POINTERS
    ArenaBag_t *par = GetArenaFromBag(bag);
    if (val != 0 && (val < par->BagHandleStart || val >= par->StopBags) ) {
        printf("SET_LINK_BAG: Attempt to set invalid link ptr [outside Arena #%d], = %p (%p)\n",
               par->ArenaNumber, bag, val);
    }
#endif
    
    BagPtr_t *p = (*(BagPtr_t **)(bag)); /* PTR_BAG(bag); */
    BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
    ptr->bagLinkPtr = val;
    return;
}

// Get the size of [the data in] a bag
UInt GET_SIZE_BAG(BagPtr_t bag)
{
    BagPtr_t *p = (*(BagPtr_t **)(bag)); /* PTR_BAG(bag); */
    BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
    return ptr->bagSize;
}

// Set the size of a bag [size of data in the bag]
void SET_SIZE_BAG(BagPtr_t bag, UInt val)
{
    BagPtr_t *p = (*(BagPtr_t **)(bag)); /* PTR_BAG(bag); */
    BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
    ptr->bagSize = val;
    return;
}

ObjType GET_TYPE_BAG ( BagPtr_t bag )
{
    if (IS_INTOBJ(bag)) {
        return T_INT;
    }
    else if (! IS_BAG(bag)) {
        return T_ILLEGAL;
    }
    else {
        BagPtr_t *p = (*(BagPtr_t **)(bag));            // bag data pointer
        BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);

        return (ObjType)( ptr->bagFlagsType & TYPE_BIT_MASK );
    }
}

// Set the type for the bag
void SET_TYPE_BAG(BagPtr_t bag, UInt val)
{
    BagPtr_t *p = (*(BagPtr_t **)(bag)); /* PTR_BAG(bag); */
    BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
    UInt flagType = (ptr->bagFlagsType & BF_ALL_FLAGS) + val;
    ptr->bagFlagsType = flagType;
    return;
}

// Given a handle to the bag (BagPtr_t) return the address of the data
BagPtr_t *PTR_BAG(const BagPtr_t bag)
{
    BagPtr_t *p = (*(BagPtr_t **)(bag));
    return p;
}

// Store the address [of the bag data] in a handle
void SET_PTR_BAG(BagPtr_t bag, BagPtr_t *dst)
{
    (*(BagPtr_t **)(bag)) = dst;
    return;
}

BagPtr_t ELM_BAG ( BagPtr_t bag, UInt i )
{
    BagPtr_t *p = (*(BagPtr_t **)(bag));
    
    return p[i];
}

BagPtr_t SET_ELM_BAG ( BagPtr_t bag, UInt i, BagPtr_t elm )
{
    BagPtr_t *p = (*(BagPtr_t **)(bag));

    p[i] = elm;
    return elm;
}


BagPtr_t GET_LINK_PTR ( BagPtr_t ptr )
{
    // ptr points to the start of the bag [header], pull & test the link field--should be in same arena
    BagStruct_t *p = (BagStruct_t *)ptr;
    ArenaBag_t *par = &MemArena[0];

    while (par->BagHandleStart) {                 // Allocated arenas have valid start handles
        if (ptr < par->OldBagStart || ptr >= par->EndBags) {
            // bag is not in this arena
            par++;
            continue;
        }
        if (p->bagLinkPtr != 0 && (p->bagLinkPtr < par->BagHandleStart || p->bagLinkPtr >= par->EndBags) ) {
//        if ( p->bagLinkPtr < par->BagHandleStart || p->bagLinkPtr >= par->EndBags ) {
            // link ptr doesn't point to a bag in this arena
            printf("GET_LINK_PTR: bag ptr = %p, link value %p is not in allocated area of Arena #%d\n",
                   ptr, p->bagLinkPtr, par->ArenaNumber);
            //  char * msg = GuMakeMessage("GET_LINK_PTR: Bag has invalid link ptr, = %p (%p)\n",
            //                       ptr, p->bagLinkPtr);
            //  SyAbortBags(msg);
        }
        break;
    }

    return p->bagLinkPtr;
}

void SET_LINK_PTR ( BagPtr_t ptr, BagPtr_t val )
{
    // ptr points to the beginning [header] of the bag (it's *not* a bag handle)
    BagStruct_t *p = (BagStruct_t *)ptr;
    ArenaBag_t *par = GetArenaFromBag(p->bagLinkPtr);

    if (par != (ArenaBag_t *)NULL && val != 0 && (val < par->BagHandleStart || val >= par->EndBags)) {
        // new link ptr is not in the allocated area of this arena
        printf("SET_LINK_PTR: new link ptr (val) = %p, Arena #%d, bag is not in allocated area of arena\n",
               val, par->ArenaNumber);
    }
    
    p->bagLinkPtr = val;

    return;
}



/****************************************************************************
**
*v  SizeType . . . . . . . . . . size of handle and data area of a bag, local
**
**  'SizeType' is an array, that contains for information about how large the
**  handle area and the data area are for each type.
**
**  'SizeType[<type>].handles' is the size of the handle area in bytes, i.e.,
**  the size of the part at the beginning of the bag which contains  handles.
**
**  'SizeType[<type>].data' is the size of the data area in bytes, i.e.,  the
**  size of the part following the handle  area which stores arbitrary  data.
**
**  'SizeType[<type>].name' is the name  of the type <type>. This  is  useful
**  when debugging GAP, but is otherwise ignored.
**
**  For example, 'GET_SIZE_BAG[T_FFE].handles' is 'SIZE_HD' and
**  'SizeType[T_FFE].data' is 'sizeof(short)', that means the finite field
**  elements have one handle, referring to the finite field, and one short
**  value, which is the value of this finite field element, i.e., the discrete
**  logarithm.
**
**  Either value may also be negative, which  means  that  the  size  of  the
**  corresponding area has a variable size.
**
**  For example, 'SizeType[T_VAR].handles' is 'SIZE_HD' and
**  'SizeType[T_VAR].data' is -1, which means that variable bags have one
**  handle, for the value of the variable, and a variable sized data area, for
**  the name of the identi- fier.
**
**  For example, 'GET_SIZE_BAG[T_LIST].handles' is '-SIZE_HD' and
**  'SizeType[T_LIST].data' is '0', which means that a list has a variable
**  number of handles, for the elements of the list, and no other data.
**
**  If both values are negative both areas are variable sized.  The ratio  of
**  the  sizes  is  fixed,  and  is  given  by the ratio of the two values in
**  'SizeType'.
**
**  I can not give you an example, because no such type is currently in use.
*/

SizeTypeT             SizeType [] = {
{         0,               0,  "voi" },
{         0,               0,  "123" },
{         0,              -1,  "+12" },
{         0,              -1,  "-12" },
{ 2*SIZE_HD,               0,  "1/2" },
{  -SIZE_HD,  
         -(Int)sizeof(short),  "cyc" },
{         0,  sizeof(double),  "dbl" },
{         0,2*sizeof(double),  "cpx" },
{         0,    sizeof(Int),  "unk" },
{   SIZE_HD,   sizeof(short),  "ffe" },
{         0,              -1,  "prm" },
{         0,              -1,  "prm" },
{  -SIZE_HD,               0,  "wrd" },
{   SIZE_HD,              -1,  "swd" },
{   SIZE_HD,              -1,  "agw" },
{         0,               0,  "bol" },
{         0,               1,  "chr" },
{  -SIZE_HD, 2*sizeof(short),  "-> " }, /* function */
{  -SIZE_HD, 2*sizeof(short),  "->m" }, /* method */
/* internal function, it holds a function pointer + name of the the
 * defining variable  */
{         0,              -1,  "fni" },
{  -SIZE_HD,               0,  "[1]" },
{  -SIZE_HD,               0,  "{1}" },
{  -SIZE_HD,               0,  "vec" },
{   SIZE_HD,              -1,  "vff" },
{   SIZE_HD,              -1,  "bli" },
{         0,              -1,  "str" },
{ 2*SIZE_HD,               0,  "ran" },
{  -SIZE_HD,               0,  "rec" },
{  -SIZE_HD,               0,  "max" }, /* unused                          */
{  -SIZE_HD,               0,  "mfx" }, /* unused                          */
{  -SIZE_HD,               0,  "lsx" }, /* unused                          */
{   SIZE_HD,               0,  "dly" },

{ 2*SIZE_HD,              -1,  "var" },
{ 2*SIZE_HD,              -1,  "aut" },
{ 3*SIZE_HD,               0,  "v:=" }, /* third field is for documentation form comment buffer */
{ 2*SIZE_HD,               0,  "v=>" },
{ 2*SIZE_HD,               0,  "l[]" },
{ 2*SIZE_HD,    sizeof(Int),  "l[]" },
{ 2*SIZE_HD,               0,  "l{}" },
{ 2*SIZE_HD,    sizeof(Int),  "l{}" },
{ 2*SIZE_HD,               0,  "l:=" },
{ 2*SIZE_HD,    sizeof(Int),  "l:=" },
{ 2*SIZE_HD,               0,  "l:=" },
{ 2*SIZE_HD,    sizeof(Int),  "l:=" },
{ 2*SIZE_HD,               0,  "r.e" },
{ 2*SIZE_HD,               0,  "r:=" },
{ 2*SIZE_HD,               0,  "[]=" },

{ 2*SIZE_HD,               0,  "+  " },
{ 2*SIZE_HD,               0,  "-  " },
{ 2*SIZE_HD,               0,  "*  " },
{ 2*SIZE_HD,               0,  "/  " },
{ 2*SIZE_HD,               0,  "mod" },
{ 2*SIZE_HD,               0,  "^  " },
{ 2*SIZE_HD,               0,  "com" },

{   SIZE_HD,               0,  "not" },
{ 2*SIZE_HD,               0,  "and" },
{ 2*SIZE_HD,               0,  "or " },
{ 2*SIZE_HD,               0,  "=  " },
{ 2*SIZE_HD,               0,  "<> " },
{ 2*SIZE_HD,               0,  "<  " },
{ 2*SIZE_HD,               0,  ">= " },
{ 2*SIZE_HD,               0,  "<= " },
{ 2*SIZE_HD,               0,  ">  " },
{ 2*SIZE_HD,               0,  "in " },
{ 2*SIZE_HD,               0,  ":: " },

{  -SIZE_HD,               0,  "f()" },
{  0,               -1,  "i()" }, /* T_FUNCINT_CALL, this is precursor of fast internal call
                                     current implementation is bad, since function pointer
                                     (which is a true C pointer, not a collectable Obj)
                                     comes first and thus we have to tell GASMAN the whole
                                     thing is pure data (not collectable), this prevents
                                     having T_FUNCINT_CALL with parameters that are Obj's.
                                     We need -SIZE_HD,0 rather than 0,-1, or I can wait
                                     till GC is replaced */
{  -SIZE_HD,               0,  ";;;" },
{  -SIZE_HD,               0,  "if " },
{ 3*SIZE_HD,               0,  "for" },
{ 2*SIZE_HD,               0,  "whi" },
{ 2*SIZE_HD,               0,  "rep" },
{  -SIZE_HD,               0,  "ret" },

{  -SIZE_HD,               0,  "mpr" },
{  -SIZE_HD, 2*sizeof(short),  "mfu" },
{  -SIZE_HD, 2*sizeof(short),  "mme" },
{  -SIZE_HD,               0,  "mls" },
{         0,              -1,  "mst" },
{  -SIZE_HD,               0,  "mrn" },
{  -SIZE_HD,               0,  "mre" },
{  -SIZE_HD,               0,  "mta" },
{  -SIZE_HD,               0,  "mle" },

{  -SIZE_HD,               0,  "cyc" },
{         0,              -1,  "ff " },

{   SIZE_HD,              -1,  "gen" },
{  -SIZE_HD,               0,  "agp" },
{  -SIZE_HD,               0,  "pcp" },
{         0,              -1,  "age" },
{         0,              -1,  "agl" },
{         0,              -1,  "rnm" },

{  -SIZE_HD,               0,  "ns " },
{  -SIZE_HD,               0,  "stk" },

{ 2*SIZE_HD,               0,  "_is" },
{         0,              -1,  "fre" }
};



/****************************************************************************
**
*V  NameType  . . . . . . . . . . . . . . . . . . .  printable name of a type
**
**  'NameType' is an array that contains for every possible type  a printable
**  name.  Those names can be used, for example, in error messages.
*/
char            * NameType [] = {
    "void",
    "integer", "integer (> 2^28)", "integer (< -2^28)",
    "rational", "cyclotomic", "double", "complex", "unknown",
    "finite field element",
    "permutation16", "permutation32",
    "word", "sparse word", "agword",
    "boolean", "character",
    "function", "method", "internal function",
    "list", "set", "vector", "finite field vector", "boolean list",
    "string", "range",
    "record",
    "matrix (extended)", "matffe (extended)", "list (extended)",
    "delayed expression",
    "variable", "autoread variable", "var assignment", "var map",
    "list element", "list element", "sublist", "sublist",
    "list assignment","list assignment","list assignment","list assignment",
    "record element", "record assignment", "multi assignment",
    "+", "-", "*", "/", "mod", "^", "commutator",
    "not", "and", "or", "=", "<>", "<", ">=", "<=", ">", "in", "::",
    "function call", "fast internal function call", "statement sequence", "if statement",
    "for loop", "while loop", "repeat loop", "return statement",
    "var permutation", "var function", "var method", "var list", "var string", "var range",
    "var record", "var hash", "let statement",
    "cycle", "finite field",
    "abstract generator",
    "aggroup", "polycyclic presentation",
    "ag exponent vector", "ag exponent/generator",
    "record name",
    "namespace",
    "stack frame",
    "_is",
    "free bag",
    "illegal bag"
};



/****************************************************************************
**
*F  NrHandles( <type>, <size> ) . . . . . . . . .  number of handles of a bag
**
**  'NrHandles' returns the number of handles of a bag with type  <type>  and
**  size <size>.  This is used in the garbage collection which needs to  know
**  the number to be able to mark all subobjects of a given object.
**
**  'NrHandles' uses the information stored in 'SizeType'.
*/

Int             NrHandles (unsigned int type, UInt size)
{
    register Int       hs, is;

    if(type >= T_ILLEGAL) return 0;

    hs = SizeType[type].handles;
    if ( hs >= 0 )  return hs / SIZE_HD;

    is = SizeType[type].data;
    if ( is >= 0 )  return (size - is) / SIZE_HD;

    return ( hs * (Int)size / (hs + is) ) / SIZE_HD;
}



int DoFullCopy;

static void _RecursiveClearFlagMutable(Obj hd, int flag, int check_has_flag)
{
    UInt       n;						/* number of handles of <hd>       */
    UInt       i;						/* loop variable                   */
    if(hd== 0 || !IS_MUTABLE(hd) || (check_has_flag && !GET_FLAG_BAG(hd, flag)))
        return;
    CLEAR_FLAG_BAG(hd, flag);

    /* and recursively clean up the rest                                   */
    n = NrHandles(GET_TYPE_BAG(hd), GET_SIZE_BAG(hd));
    for(i = n; 0 < i; i--) {
        if(PTR_BAG(hd)[i-1] != 0 && IS_MUTABLE(PTR_BAG(hd)[i-1]) && 
           (!check_has_flag || GET_FLAG_BAG(PTR_BAG(hd)[i-1], flag))) {
            _RecursiveClearFlagMutable(PTR_BAG(hd)[i-1], flag, check_has_flag);
        }
    }
}

void RecursiveClearFlagMutable(Obj hd, int flag) 
{
    _RecursiveClearFlagMutable(hd, flag, 1);
}

static void _RecursiveClearFlagFullMutable(Obj hd, int flag, int check_has_flag) 
{
    UInt       n;						/* number of handles of <hd>       */
    UInt       i;						/* loop variable                   */
    if(hd== 0 || !IS_FULL_MUTABLE(hd) || (check_has_flag && !GET_FLAG_BAG(hd, flag)))
        return;
    CLEAR_FLAG_BAG(hd, flag);

    /* and recursively clean up the rest                                   */
    n = NrHandles(GET_TYPE_BAG(hd), GET_SIZE_BAG(hd));
    for(i = n; 0 < i; i--) {
        if(PTR_BAG(hd)[i-1] != 0 && IS_FULL_MUTABLE(PTR_BAG(hd)[i-1]) && 
           (!check_has_flag || GET_FLAG_BAG(PTR_BAG(hd)[i-1], flag))) {
            _RecursiveClearFlagFullMutable(PTR_BAG(hd)[i-1], flag, check_has_flag);
        }
    }
}

void RecursiveClearFlagFullMutable(Obj hd, int flag) 
{
    _RecursiveClearFlagFullMutable(hd, flag, 1);
}

void RecursiveClearFlag(Obj hd, int flag) 
{
    UInt       n;						/* number of handles of <hd>       */
    UInt       i;						/* loop variable                   */
    if(hd==0 || IS_INTOBJ(hd) || !IS_BAG(hd) || !GET_FLAG_BAG(hd, flag))
    return;

    CLEAR_FLAG_BAG(hd, flag);

    /*if(!IS_FULL_MUTABLE(hd))
      return;*/

    /* and recursively clean up the rest                                   */
    n = NrHandles(GET_TYPE_BAG(hd), GET_SIZE_BAG(hd));
    for(i = n; 0 < i; i--)
    RecursiveClearFlag(PTR_BAG(hd)[i-1], flag);
}

void RecursiveClearFlagsFullMutable(Obj hd, int howMany, ...) 
{
    UInt       i;						/* loop variable                   */
    unsigned int        composed_flag = 0;
    va_list ap;

    if(hd== 0 || !IS_FULL_MUTABLE(hd))
        return;

    va_start(ap, howMany);
    for(i = 0; i < howMany; ++i) {
        composed_flag |= va_arg(ap, int);
    }
    va_end(ap);

    _RecursiveClearFlagFullMutable(hd, composed_flag, 1);
}   

