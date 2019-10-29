/****************************************************************************
**
*A  gasman.c                    GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2019, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2019) by the GAP Group (www.gap-system.org).
**
**  This file contains the functions of  GASMAN,  the  GAP  storage  manager.
**
**  Gasman is the  GAP storage manager.  That  means that the other parts  of
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

#include <stdio.h>
#include <stdlib.h>
#include        "system.h"              /* system dependent functions      */
#include        "scanner.h"             /* Pr()                            */
#include        "memmgr.h"              /* declaration part of the package */
#include		"GapUtils.h"
#include        <assert.h>


/****************************************************************************
**
*F  EnterKernel() . . . . . . . . . . . establish lifetime scopes for objects
*F  ExitKernel( <hd> )  . . . . . . . . .  establish lifetime scopes for bags
*/

/*
static UInt NumBags = 0;

static void AddGlobalBag(Bag b) {
    NumBags++;
    //    InitGlobalBag(b[-1], GuMakeMessage("%d", NumBags));
}
*/

void            EnterKernel (void)
{
    /*
    static Int first_time = 1;
    if(first_time) {
    	CallbackForAllBags(AddGlobalBag);
    	first_time = 0;
    	Pr("Initialized %d global bags.\n ", NumBags, 0);
    }
    */
}

void            ExitKernel (BagPtr_t hd)
{
    /* do nothing */
}


/****************************************************************************
**
*F  CollectGarb() . . . . . . . . . . . . . . . . . . . . collect the garbage
**
**  'CollectGarb' performs a garbage collection.  This means it  removes  the
**  unused bags from memory and compacts the  used  bags  at  the  beginning.
*/
void            CollectGarb (void)
{
    CollectBags(0, 1);
}


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


BagPtr_t       NewBag ( UInt type, UInt size ) {
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
void            Retype ( BagPtr_t hdBag, UInt newType ) {
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
void            Resize ( BagPtr_t hdBag, UInt newSize ) {

    ResizeBag(hdBag, newSize);
}


static int nrMarkedBags;				/* count number of bags marked */

#ifdef MARKBAG_ISFUNC

/* 
 * Provide a function version of MARK_BAG to permit debuging...
 */

void MARK_BAG( BagPtr_t bag )
{
	/*
	 * Check the bag presented is:
	 *    non-zero
	 *    has none of the 2 or 3 least significant bits sets (is not currently tagged)
	 *    lies between MptrBags and OldBags (area reserved for master pointers)
	 * Next check the data pointer associated to the bag:
	 *    lies between YoungBags and Allocbags (area where new bags created since last GC)
	 * Next check 
	 *    if the link pointer points back to the bag handle (bag not previous marked?) or
	 *    if the link pointer points to bag handle + 2 (bag marked half dead)
	 */
	if (bag && (((UInt)(bag)) & (sizeof(BagPtr_t)-1)) == 0 &&
		(BagPtr_t)MptrBags <= (bag)	&& (bag) < (BagPtr_t)OldBags) {
		BagPtr_t *p = (*(BagPtr_t **)(bag));			/* data pointer asociated to bag */
		if (YoungBags < p  &&  p <= AllocBags) {
			BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE); /* struct pointer */
			if (((BagPtr_t)ptr->bagLinkPtr == bag) ||
				((BagPtr_t)ptr->bagLinkPtr == (BagPtr_t)(((char *)(bag)) + 2)))  {
				ptr->bagLinkPtr = MarkedBags;
				MarkedBags = bag;
				nrMarkedBags++;
			}
		}
	}
	return;
}
#endif	// MARKBAG_ISFUNC


/****************************************************************************
**
*F  InitGasman()  . . . . . . . . . initialize dynamic memory manager package
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


void            MarkBag_GAP3(BagPtr_t bag)
{
    BagPtr_t *               ptr;            /* pointer into the bag            */
    BagPtr_t                 sub;            /* one subbag identifier           */
    UInt                i;              /* loop variable                   */
    ptr = PTR_BAG( bag );
    for(i = 0; i < NrHandles(GET_TYPE_BAG(bag), GET_SIZE_BAG(bag)); ++i) {
		sub = ptr[i];
		MARK_BAG( sub );
    }
}


extern void InitAllGlobals();
extern void InitWeakPtrs();

void            InitGasman (int *pargc, char **argv)
{
    UInt type;

    InitSystem4(*pargc, argv);
    
    /* it seems that setting cache size > 0 is not a good idea, since it causes
       a lot of garbage collections, and since we always to full collection, this
       is very slow */
    SyCacheSize = 0;

    /* 1 is the most conservative value, and will *always* work. However the default
       which is sizeof(BagPtr_t) should be Ok (system4.h). */
    /*SyStackAlign = 1;*/

    /* extern  void            InitBags (
                TNumAllocFuncBags   alloc_func,
                UInt                initial_size,
                TNumStackFuncBags   stack_func,
                BagPtr_t *               stack_bottom,
                UInt                stack_align,
                UInt                cache_size,
                UInt                dirty,
                TNumAbortFuncBags   abort_func );
    */
    
    InitBags( SyAllocBags, SyStorMin,
              0, (BagPtr_t*)(((UInt)pargc/SyStackAlign)*SyStackAlign), SyStackAlign,
              SyCacheSize, 0, SyAbortBags );
    
    InitMsgsFuncBags( SyMsgsBags );
    InitWeakPtrs();

	/* Install MarkBag_GAP3 as the marking function for all bag types */
	/* (overrides MarkAllSubBagsDefault which was installed in InitBags above)  */
    for(type = 0; type < T_ILLEGAL; ++type) {
		InitMarkFuncBags(type, MarkBag_GAP3);
		InfoBags[type].name = NameType[type];
    }

    InitAllGlobals();
}

const char* TNAM_BAG(BagPtr_t bag) {
  return InfoBags[ TNUM_BAG(bag) ].name;
}



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
** GASMAN_FREE_RATIO is defined as the fraction of space theoretically
** allocable to bags (i.e., EndBags - OldBags) which must be free after a full
** GC; otherwise, a message is printed and the program exits.
**
** Please see gasman4.h
*/


/****************************************************************************
**
*F  WORDS_BAG( <size> ) . . . . . . . . . . words used by a bag of given size
**
**  The structure of a bag is a follows{\:}
**
**    <identifier>
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
**    |<sz>.<tp>|  <link> |         .         .         .         .    | pad|
**    +---------+---------+--------------------------------------------+----+
**
**  A bag consists of a masterpointer, and a body.
**
**  The *masterpointer* is a pointer  to the data  area of the bag.  During a
**  garbage collection  the masterpointer is the  only active  pointer to the
**  data area of the bag, because of the rule that no pointers to or into the
**  data  area of a  bag may be remembered  over calls  to functions that may
**  cause a garbage  collection.  It is the job  of the garbage collection to
**  update the masterpointer of a bag when it moves the bag.
**
**  The *identifier*  of  the  bag is a   pointer  to (the  address   of) the
**  masterpointer  of  the bag.   Thus   'PTR_BAG(<bag>)' is simply '\*<bag>'
**  plus a cast.
**
**  The *body* of a  bag consists of  the size-type word,  the link word, the
**  data area, and the padding.
**
**  The *size-type word* contains the size of the bag in the upper  (at least
**  24) bits, and the type (abbreviated as <tp> in the  above picture) in the
**  lower 8  bits.  Thus 'GET_SIZE_BAG'   simply extracts the size-type  word and
**  shifts it 8 bits to the right, and 'TNUM_BAG' extracts the size-type word
**  and masks out everything except the lower 8 bits.
**
**  The  *link word* usually   contains the identifier of  the  bag,  i.e., a
**  pointer to the masterpointer of the bag.  Thus the garbage collection can
**  find  the masterpointer of a  bag through the link  word if  it knows the
**  address of the  data area of the bag.    The link word   is also used  by
**  {\Gasman} to  keep   bags  on  two linked  lists  (see  "ChangedBags" and
**  "MarkedBags").
**
**  The *data area* of a  bag is the area  that  contains the data stored  by
**  the application in this bag.
**
**  The *padding* consists  of up to 'sizeof(BagPtr_t)-1' bytes  and pads the body
**  so that the size of a  body is always  a multiple of 'sizeof(BagPtr_t)'.  This
**  is to ensure that bags are always aligned.  The macro 'WORDS_BAG(<size>)'
**  returns the number  of words occupied  by the data  area and padding of a
**  bag of size <size>.
**
**  A body in the workspace whose size-type word contains the value 255
**  (T_RESIZE_FREE) in the lower 8 bits is the remainder of a 'ResizeBag'.
**  That is it consists either of the unused words after a bag has been shrunk
**  or of the old body of the bag after the contents of the body have been
**  copied elsewhere for an extension.  The upper (at least 24) bits of the
**  first word contain the number of bytes in this area excluding the first
**  word itself.  Note that such a body has no link word, because such a
**  remainder does not correspond to a bag (see "Implementation of
**  ResizeBag").
**
**  A masterpointer with a value  congruent to 1  mod 4 is   the relic of  an
**  object  that was  weakly but not   strongly  marked in  a recent  garbage
**  collection.   These persist until  after the next full garbage collection
**  by which time all references to them should have been removed.
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
*V  MptrBags  . . . . . . . . . . . . . . beginning of the masterpointer area
*V  OldBags . . . . . . . . . . . . . . . . .  beginning of the old bags area
*V  YoungBags . . . . . . . . . . . . . . .  beginning of the young bags area
*V  AllocBags . . . . . . . . . . . . . . .  beginning of the allocation area
*V  AllocSizeBags . . . . . . . . . . . . . . . . size of the allocation area
*V  StopBags  . . . . . . . . . . . . . . . beginning of the unavailable area
*V  EndBags . . . . . . . . . . . . . . . . . . . . . .  end of the workspace
**
**  The memory manager manages one large block of storage called the
**  *workspace*.  The total workspace is allocated during program startup; the
**  workspace size is fixed based on the -m option specified on the command
**  line.  Currently, the workspace is not permitted to grow; however, future
**  enhancement to the memory manager will change this.  The layout of the
**  workspace is as follows{\:}
**
**  +-------------+-----------------+------------+------------+-------------+
**  |masterpointer|    old bags     | young bags | allocation | unavailable |
**  |    area     |      area       |    area    |    area    |    area     |
**  +-------------+-----------------+------------+------------+-------------+
**  ^             ^                 ^            ^            ^             ^
**  MptrBags    OldBags         YoungBags    AllocBags     StopBags   EndBags
**
**  The *masterpointer area*  contains  all the masterpointers  of  the bags.
**  'MptrBags' points to the beginning of this area and 'OldBags' to the end.
**  The master pointer area is allocated 1/8 of the total workspace.  
**
**  The *old bags area* contains the bodies of all the  bags that survived at
**  least one  garbage collection.  This area is  only  scanned for dead bags
**  during a full garbage collection.  'OldBags'  points to the  beginning of
**  this area and 'YoungBags' to the end.
**
**  The *young bags area* contains the bodies of all  the bags that have been
**  allocated since the  last garbage collection.  This  area is scanned  for
**  dead  bags during  each garbage  collection.  'YoungBags'  points  to the
**  beginning of this area and 'AllocBags' to the end.
**
**  The *allocation area* is the storage  that is available for allocation of
**  new bags.  When a new bag is allocated the storage for  the body is taken
**  from  the beginning of   this area,  and  this  area  is  correspondingly
**  reduced.   If  the body does not   fit in the  allocation  area a garbage
**  collection is  performed.  'AllocBags' points   to the beginning of  this
**  area and 'StopBags' to the end.
**
**  The *unavailable  area* is  the free  storage that  is not  available for
**  allocation.   'StopBags'  points  to  the  beginning  of  this  area  and
**  'EndBags' to the end.
**
**  If <cache-size>  (see "InitBags") was 0,  'CollectBags' makes all of  the
**  free storage available for allocations by setting 'StopBags' to 'EndBags'
**  after garbage collections.   In   this case garbage  collections are only
**  performed when no  free storage   is left.  <cache-size> is zero in this
**  version, further garbage collection is always "full".
**
**  Note that  the  borders between the areas are not static.  In  particular
**  each allocation increases the size of the young bags area and reduces the
**  size of the  allocation area.  On the other hand each garbage  collection
**  empties the young bags area.
*/
BagPtr_t *                   MptrBags;
BagPtr_t *                   OldBags;
BagPtr_t *                   YoungBags;
BagPtr_t *                   AllocBags;
UInt                    AllocSizeBags;
BagPtr_t *                   StopBags;
BagPtr_t *                   EndBags;


/****************************************************************************
**
*V  FreeMptrBags  . . . . . . . . . . . . . . .  list of free bag identifiers
**
**  'FreeMptrBags' is the  first free bag identifier, i.e., it points  to the
**  first  available  masterpointer.   If 'FreeMptrBags'  is 0  there are  no
**  available masterpointers.  The available masterpointers are  managed in a
**  forward linked list,  i.e., each available  masterpointer  points  to the
**  next available masterpointer, except for the last, which contains 0.
**
**  When a new  bag is allocated  it gets the identifier  'FreeMptrBags', and
**  'FreeMptrBags' is set to the value stored in this masterpointer, which is
**  the next available masterpointer.  When a bag is deallocated by a garbage
**  collection  its  masterpointer  is  added   to  the  list  of   available
**  masterpointers again.
*/
BagPtr_t FreeMptrBags;


/****************************************************************************
**
*V  ChangedBags . . . . . . . . . . . . . . . . . .  list of changed old bags
**
**  'ChangedBags' holds a  list of old bags that  have been changed since the
**  last garbage collection, i.e., for  which either 'CHANGED_BAG' was called
**  or which have been resized.
**
**  This list starts with the bag whose identifier is 'ChangedBags', and the
**  link word (i.e., link pointer, see BagStruct_t) of each bag on the list
**  contains the identifier of the next bag on the list.  The link word of the
**  last bag on the list contains 0.  If 'ChangedBags' has the value 0, the
**  list is empty.
**
**  The garbage collection needs to know which young  bags are subbags of old
**  bags, since  it  must  not  throw   those away    in a partial    garbage
**  collection.  Only  those old bags that  have been changed  since the last
**  garbage collection can contain references to  young bags, which have been
**  allocated since the last garbage  collection.  The application cooperates
**  by informing {\Gasman} with 'CHANGED_BAG' which bags it has changed.  The
**  list of changed old  bags is scanned by a  partial garbage collection and
**  the young subbags of the old bags on this list are marked with 'MARK_BAG'
**  (see "MarkedBags").  Without this  list 'CollectBags' would have to  scan
**  all old bags for references to young bags, which would take too much time
**  (see "Implementation of CollectBags").
**  NOTE: This behaviour is not currently used as all garbage collections are
**  "full" only.
**
**  'CHANGED_BAG' puts a bag on the list  of changed old bags.  'CHANGED_BAG'
**  first checks that <bag> is an old bag by checking that 'PTR_BAG( <bag> )'
**  is smaller than 'YoungBags'.  Then it checks that  the bag is not already
**  on the list of changed bags by checking that the link word still contains
**  the identifier of <bag>.  If <bag> is an  old bag that  is not already on
**  the list of changed bags, 'CHANGED_BAG' puts <bag> on the list of changed
**  bags,  by  setting  the  link word of   <bag>   to the   current value of
**  'ChangedBags' and then setting 'ChangedBags' to <bag>.
*/
BagPtr_t                     ChangedBags;


/****************************************************************************
**
*V  MarkedBags  . . . . . . . . . . . . . . . . . . . . . list of marked bags
**
**  'MarkedBags' holds a list of bags that have already  been marked during a
**  garbage collection by 'MARK_BAG'.  This list is only used  during garbage
**  collections, so it is  always empty outside  of  garbage collections (see
**  "Implementation of CollectBags").
**
**  This list starts with the  bag whose identifier  is 'MarkedBags', and the
**  link word of each bag on the list contains the identifier of the next bag
**  on the list.  The link word of the  last bag on the list  contains 0.  If
**  'MarkedBags' has the value 0, the list is empty.
**
**  Note that some other  storage managers do not use  such a list during the
**  mark phase.   Instead  they simply let the  marking   functions call each
**  other.  While this is  somewhat simpler it  may use an unbound  amount of
**  space on the stack.  This is particularly  bad on systems where the stack
**  is not in a separate segment of the address space, and thus may grow into
**  the workspace, causing disaster.
**
**  'MARK_BAG'   puts a  bag <bag>  onto  this list.    'MARK_BAG'  has to be
**  careful, because it can be called  with an argument that  is not really a
**  bag identifier, and may  point  outside the programs  address space.   So
**  'MARK_BAG' first checks that <bag> points  to a properly aligned location
**  between 'MptrBags' and 'OldBags'.   Then 'MARK_BAG' checks that <bag>  is
**  the identifier  of a young bag by  checking that the masterpointer points
**  to  a  location between  'YoungBags'  and  'AllocBags'  (if <bag>  is the
**  identifier of an   old bag, the  masterpointer will  point to a  location
**  between  'OldBags' and 'YoungBags',  and if <bag>   only appears to be an
**  identifier, the masterpointer could be on the free list of masterpointers
**  and   point   to a  location  between  'MptrBags'  and  'OldBags').  Then
**  'MARK_BAG' checks  that <bag> is not  already marked by checking that the
**  link  word of <bag>  contains the identifier of the   bag.  If any of the
**  checks fails, 'MARK_BAG' does nothing.  If all checks succeed, 'MARK_BAG'
**  puts <bag> onto the  list of marked bags by  putting the current value of
**  'ChangedBags' into the link word  of <bag>  and setting 'ChangedBags'  to
**  <bag>.  Note that since bags are always placed  at the front of the list,
**  'CollectBags' will   mark the bags   in a  depth-first  order.   This  is
**  probably good to improve the locality of reference.
*/
BagPtr_t                     MarkedBags;


/****************************************************************************
**
*V  NrAllBags . . . . . . . . . . . . . . . . .  number of all bags allocated
*V  SizeAllBags . . . . . . . . . . . . . .  total size of all bags allocated
*V  NrLiveBags  . . . . . . . . . .  number of bags that survived the last gc
*V  SizeLiveBags  . . . . . . .  total size of bags that survived the last gc
*V  NrDeadBags  . . . . . . . number of bags that died since the last full gc
*V  SizeDeadBags  . . . . total size of bags that died since the last full gc
*V  NrHalfDeadBags  . . . . . number of bags that died since the last full gc
**                            but may still be weakly pointed to
*/
UInt                    NrAllBags;
UInt                    SizeAllBags;
UInt                    NrLiveBags;
UInt                    SizeLiveBags;
UInt                    NrDeadBags;
UInt                    SizeDeadBags;
UInt                    NrHalfDeadBags;


/****************************************************************************
**
*V  InfoBags[<type>]  . . . . . . . . . . . . . . . . .  information for bags
*/
TNumInfoBags            InfoBags [ NTYPES ];


/****************************************************************************
**
*F  InitMsgsFuncBags(<msgs-func>) . . . . . . . . .  install message function
**
**  'InitMsgsFuncBags'  simply  stores  the  printing  function  in a  global
**  variable.
*/
TNumMsgsFuncBags        MsgsFuncBags;

void            InitMsgsFuncBags (
    TNumMsgsFuncBags    msgs_func )
{
    MsgsFuncBags = msgs_func;
}


/****************************************************************************
**
*F  InitSweepFuncBags(<type>,<mark-func>)  . . . .  install sweeping function
*/

TNumSweepFuncBags TabSweepFuncBags [ NTYPES ];


void InitSweepFuncBags (
    UInt                type,
    TNumSweepFuncBags    sweep_func )
{
#ifdef CHECK_FOR_CLASH_IN_INIT_SWEEP_FUNC
    char                str[256];

    if ( TabSweepFuncBags[type] != 0 ) {
        str[0] = 0;
        SyStrncat( str, "warning: sweep function for type ", 33 );
        str[33] = '0' + ((type/100) % 10);
        str[34] = '0' + ((type/ 10) % 10);
        str[35] = '0' + ((type/  1) % 10);
        str[36] = 0;
        SyStrncat( str, " already installed\n", 19 );
        SyFputs( str, 0 );
    }
#endif
    TabSweepFuncBags[type] = sweep_func;
}


/****************************************************************************
**
*F  InitMarkFuncBags(<type>,<mark-func>)  . . . . .  install marking function
*F  MarkNoSubBags(<bag>)  . . . . . . . . marking function that marks nothing
*F  MarkOneSubBags(<bag>) . . . . . .  marking function that marks one subbag
*F  MarkTwoSubBags(<bag>) . . . . . . marking function that marks two subbags
*F  MarkThreeSubBags(<bag>) . . . . marking function that marks three subbags
*F  MarkFourSubBags(<bag>)  . . . .  marking function that marks four subbags
*F  MarkAllSubBags(<bag>) . . . . . .  marking function that marks everything
**
**  'InitMarkFuncBags', 'MarkNoSubBags', 'MarkOneSubBags',  'MarkTwoSubBags',
**  and 'MarkAllSubBags' are really too simple for an explanation.
**
**  'MarkAllSubBagsDefault' is the same  as 'MarkAllSubBags' but is only used
**  by GASMAN as default.  This will allow to catch type clashes.
*/
TNumMarkFuncBags TabMarkFuncBags [ NTYPES ];

extern void MarkAllSubBagsDefault ( BagPtr_t bag );

void InitMarkFuncBags ( UInt type, TNumMarkFuncBags mark_func )
{
#ifdef CHECK_FOR_CLASH_IN_INIT_MARK_FUNC
    char                str[256];

    if ( TabMarkFuncBags[type] != MarkAllSubBagsDefault ) {
        str[0] = 0;
        SyStrncat( str, "warning: mark function for type ", 32 );
        str[32] = '0' + ((type/100) % 10);
        str[33] = '0' + ((type/ 10) % 10);
        str[34] = '0' + ((type/  1) % 10);
        str[35] = 0;
        SyStrncat( str, " already installed\n", 19 );
        SyFputs( str, 0 );
    }
#endif
    TabMarkFuncBags[type] = mark_func;
}

void MarkNoSubBags ( BagPtr_t bag )
{
}

void MarkOneSubBags ( BagPtr_t bag )
{
    BagPtr_t sub;            /* one subbag identifier           */
    sub = PTR_BAG(bag)[0];
    MARK_BAG( sub );
}

void MarkTwoSubBags (
    BagPtr_t                 bag )
{
    BagPtr_t                 sub;            /* one subbag identifier           */
    sub = PTR_BAG(bag)[0];
    MARK_BAG( sub );
    sub = PTR_BAG(bag)[1];
    MARK_BAG( sub );
}

void MarkThreeSubBags (
    BagPtr_t                 bag )
{
    BagPtr_t                 sub;            /* one subbag identifier           */
    sub = PTR_BAG(bag)[0];
    MARK_BAG( sub );
    sub = PTR_BAG(bag)[1];
    MARK_BAG( sub );
    sub = PTR_BAG(bag)[2];
    MARK_BAG( sub );
}

void MarkFourSubBags (
    BagPtr_t                 bag )
{
    BagPtr_t                 sub;            /* one subbag identifier           */
    sub = PTR_BAG(bag)[0];
    MARK_BAG( sub );
    sub = PTR_BAG(bag)[1];
    MARK_BAG( sub );
    sub = PTR_BAG(bag)[2];
    MARK_BAG( sub );
    sub = PTR_BAG(bag)[3];
    MARK_BAG( sub );
}

void MarkAllSubBags (
    BagPtr_t                 bag )
{
    BagPtr_t *               ptr;            /* pointer into the bag            */
    BagPtr_t                 sub;            /* one subbag identifier           */
    UInt                i;              /* loop variable                   */

    /* mark everything                                                     */
    ptr = PTR_BAG( bag );
    for ( i = GET_SIZE_BAG(bag)/sizeof(BagPtr_t); 0 < i; i-- ) {
        sub = ptr[i-1];
        MARK_BAG( sub );
    }

}

void MarkAllSubBagsDefault (
    BagPtr_t                 bag )
{
    BagPtr_t *               ptr;            /* pointer into the bag            */
    BagPtr_t                 sub;            /* one subbag identifier           */
    UInt                i;              /* loop variable                   */

    /* mark everything                                                     */
    ptr = PTR_BAG( bag );
    for ( i = GET_SIZE_BAG(bag)/sizeof(BagPtr_t); 0 < i; i-- ) {
        sub = ptr[i-1];
        MARK_BAG( sub );
    }

}


void MarkBagWeakly(
    BagPtr_t             bag )
{
  if ( (((UInt)bag) & (sizeof(BagPtr_t)-1)) == 0 /* really looks like a pointer */
       && (BagPtr_t)MptrBags <= bag              /* in plausible range */
       && bag < (BagPtr_t)OldBags                /*  "    "       "    */
       && YoungBags < PTR_BAG(bag)          /*  points to a young bag */
       && PTR_BAG(bag) <= AllocBags         /*    "     " "  "     "  */
       && IS_MARKED_DEAD(bag) )             /*  and not marked already */
    {                          
      SET_LINK_BAG(bag, (BagPtr_t)MARKED_HALFDEAD(bag));   /* mark it now as we
                                               don't have to recurse */
    }
}



/****************************************************************************
**
*F  CallbackForAllBags( <func> ) call a C function on all non-zero mptrs
**
**  This calls a C function on every bag, including garbage ones, by simply
**  walking the masterpointer area. Not terribly safe.
**
*/

void CallbackForAllBags(
     void (*func)(BagPtr_t) )
{
  BagPtr_t ptr;
  for (ptr = (BagPtr_t)MptrBags; ptr < (BagPtr_t)OldBags; ptr ++)
    if (*ptr != 0 && !IS_WEAK_DEAD_BAG(ptr) && (BagPtr_t)(*ptr) >= (BagPtr_t)OldBags)
      {
        (*func)(ptr);
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
UInt GlobalSortingStatus = 0;
Int WarnInitGlobalBag = 0;

extern TNumAbortFuncBags   AbortFuncBags;

void InitGlobalBag (
    BagPtr_t *               addr,
    const Char *        cookie )
{

    if ( GlobalBags.nr == NR_GLOBAL_BAGS ) {
        (*AbortFuncBags)(
            "Panic: Gasman cannot handle so many global variables" );
    }
#ifdef DEBUG_GLOBAL_BAGS
    {
      UInt i;
      if (cookie != (Char *)0)
	for (i = 0; i < GlobalBags.nr; i++)
          if ( 0 == SyStrcmp(GlobalBags.cookie[i], cookie) ) {
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



static Int IsLessGlobal (
    const Char *	cookie1, 
    const Char *	cookie2,
    UInt 		byWhat )
{
  if (byWhat != 2)
    {
      (*AbortFuncBags)("can only sort globals by cookie");
    }
  if (cookie1 == NUM_TO_UINT(0) && cookie2 == NUM_TO_UINT(0))
    return 0;
  if (cookie1 == NUM_TO_UINT(0))
    return -1;
  if (cookie2 == NUM_TO_UINT(0))
    return 1;
  return SyStrcmp(cookie1, cookie2) < 0;
}



void SortGlobals( UInt byWhat )
{
  const Char *tmpcookie;
  BagPtr_t * tmpaddr;
  UInt len, h, i, k;

  if (byWhat != 2)
    {
      (*AbortFuncBags)("can only sort globals by cookie");
    }
  if (GlobalSortingStatus == byWhat)
    return;
  len = GlobalBags.nr;
  h = 1;
  while ( 9*h + 4 < len ) 
    { h = 3*h + 1; }
  while ( 0 < h ) {
    for ( i = h; i < len; i++ ) {
      tmpcookie = GlobalBags.cookie[i];
      tmpaddr = GlobalBags.addr[i];
      k = i;
      while ( h <= k && IsLessGlobal(tmpcookie,
                                     GlobalBags.cookie[k-h],
                                     byWhat))
        {
          GlobalBags.cookie[k] = GlobalBags.cookie[k-h];
          GlobalBags.addr[k] = GlobalBags.addr[k-h];
          k -= h;
        }
      GlobalBags.cookie[k] = tmpcookie;
      GlobalBags.addr[k] = tmpaddr;
    }
    h = h / 3;
  }
  GlobalSortingStatus = byWhat;
  return;
}


Int GlobalIndexByCookie(
       const Char * cookie )
{
  UInt i,top,bottom,middle;
  Int res;
  if (cookie == NUM_TO_UINT(0))
    {
      Pr("Panic -- 0L cookie passed to GlobalByCookie\n",0,0);
      SyExit(2);
    }
  if (GlobalSortingStatus != 2) 
    {
      for (i = 0; i < GlobalBags.nr; i++)
        {
          if (SyStrcmp(cookie, GlobalBags.cookie[i]) == 0)
            return i;
        }
      return -1;
    }
  else
    {
      top = GlobalBags.nr;
      bottom = 0;
      while (top >= bottom) {
        middle = (top + bottom)/2;
        res = SyStrcmp(cookie,GlobalBags.cookie[middle]);
        if (res < 0)
          top = middle-1;
        else if (res > 0)
          bottom = middle+1;
        else
          return middle;
      }
      return -1;
    }
}

                       
BagPtr_t * GlobalByCookie(
       const Char * cookie )
{
    Int i = GlobalIndexByCookie(cookie);
    if (i<0)
	return (BagPtr_t*)NUM_TO_UINT(0);
    else
	return GlobalBags.addr[i];
}
                                      

static BagPtr_t NextMptrRestoring;
extern TNumAllocFuncBags       AllocFuncBags;

void StartRestoringBags( UInt nBags, UInt maxSize)
{
  UInt target;
  BagPtr_t *newmem;
/*BagPtr_t *ptr; */
  target = (8*nBags)/7 + (8*maxSize)/7; /* ideal workspace size */
  target = (target * sizeof (BagPtr_t) + (NUM_TO_UINT(512)*NUM_TO_UINT(1024)) - 1)/(NUM_TO_UINT(512)*NUM_TO_UINT(1024))*(NUM_TO_UINT(512)*NUM_TO_UINT(1024))/sizeof (BagPtr_t);
              /* make sure that the allocated amount of memory is divisible by 512 * 1024 */
  if ((EndBags - MptrBags) < target)
    {
      newmem  = (*AllocFuncBags)(sizeof(BagPtr_t)*(target- (EndBags - MptrBags))/1024,
                                 0);
      if (newmem == 0)
        {
          target = nBags + maxSize; /* absolute requirement */
          target = (target * sizeof (BagPtr_t) + (NUM_TO_UINT(512)*NUM_TO_UINT(1024)) - 1)/(NUM_TO_UINT(512)*NUM_TO_UINT(1024))*(NUM_TO_UINT(512)*NUM_TO_UINT(1024))/sizeof (BagPtr_t);
               /* make sure that the allocated amount of memory is divisible by 512 * 1024 */
          if ((EndBags - MptrBags) < target)
            (*AllocFuncBags)(sizeof(BagPtr_t)*(target- (EndBags - MptrBags))/1024, 1);
        }
      EndBags = MptrBags + target;
    }
  OldBags = MptrBags + nBags + (EndBags - MptrBags - nBags - maxSize)/8;
  AllocBags = OldBags;
  NextMptrRestoring = (BagPtr_t)MptrBags;
  SizeAllBags = 0;
  NrAllBags = 0;
  return;
}

BagPtr_t NextBagRestoring( UInt size, UInt type)
{
  BagPtr_t bag;
  UInt i;
  *(BagPtr_t **)NextMptrRestoring = (AllocBags+HEADER_SIZE);
  bag = NextMptrRestoring;

  SET_TYPE_PTR(AllocBags,type);
  BLANK_FLAGS_PTR(AllocBags);
  SET_SIZE_PTR(AllocBags, size);
  SET_LINK_PTR(AllocBags, NextMptrRestoring);

  NextMptrRestoring++;
#ifdef DEBUG_LOADING
  if ((BagPtr_t *)NextMptrRestoring >= OldBags)
    (*AbortFuncBags)("Overran Masterpointer area");
#endif
  AllocBags += HEADER_SIZE;
  
  for (i = 0; i < WORDS_BAG(size); i++)
    *AllocBags++ = (BagPtr_t)0;
  
#ifdef DEBUG_LOADING
  if (AllocBags > EndBags)
    (*AbortFuncBags)("Overran data area");
#endif
#ifdef COUNT_BAGS
  InfoBags[type].nrLive   += 1;
  InfoBags[type].nrAll    += 1;
  InfoBags[type].sizeLive += size;
  InfoBags[type].sizeAll  += size;
#endif
  SizeAllBags += size;
  NrAllBags ++;
  return bag;
}

void FinishedRestoringBags(void)
{
  BagPtr_t p;
/*  BagPtr_t *ptr; */
  YoungBags = AllocBags;
  StopBags = AllocBags + WORDS_BAG(AllocSizeBags);
  if (StopBags > EndBags)
    StopBags = EndBags;
  FreeMptrBags = NextMptrRestoring;
  for (p = NextMptrRestoring; p +1 < (BagPtr_t)OldBags; p++)
    *(BagPtr_t *)p = p+1;
  *(UInt*)p = 0;
  NrLiveBags = NrAllBags;
  SizeLiveBags = SizeAllBags;
  NrDeadBags = 0;
  SizeDeadBags = 0;
  NrHalfDeadBags = 0;
  ChangedBags = 0;
  return;
}


/****************************************************************************
**
*F  InitFreeFuncBag(<type>,<free-func>) . . . . . .  install freeing function
**
**  'InitFreeFuncBag' is really too simple for an explanation.
*/
TNumFreeFuncBags        TabFreeFuncBags [ NTYPES ];

UInt                    NrTabFreeFuncBags;

void            InitFreeFuncBag (
    UInt                type,
    TNumFreeFuncBags    free_func )
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
**  'InitBags'   remembers   <alloc-func>,  <stack-func>,     <stack-bottom>,
**  <stack-align>, <cache-size>,  <dirty>,    and   <abort-func>  in   global
**  variables.   It also  allocates  the initial workspace,   and sets up the
**  linked list of available masterpointer.
*/
TNumAllocFuncBags       AllocFuncBags;

TNumStackFuncBags       StackFuncBags;

BagPtr_t *                   StackBottomBags;

UInt                    StackAlignBags;

UInt                    CacheSizeBags;

UInt                    DirtyBags;

TNumAbortFuncBags       AbortFuncBags;


void            InitBags (
    TNumAllocFuncBags   alloc_func,
    UInt                initial_size,
    TNumStackFuncBags   stack_func,
    BagPtr_t *               stack_bottom,
    UInt                stack_align,
    UInt                cache_size,
    UInt                dirty,
    TNumAbortFuncBags   abort_func )
{
    BagPtr_t *               p;              /* loop variable                   */
    UInt                i;              /* loop variable                   */

    /* install the allocator and the abort function                        */
    AllocFuncBags   = alloc_func;
    AbortFuncBags   = abort_func;

    /* install the stack marking function and values                       */
    StackFuncBags   = stack_func;
    StackBottomBags = stack_bottom;
    StackAlignBags  = stack_align;

    /* first get some storage from the operating system                    */
    initial_size    = (initial_size + 511) & ~(511);
    MptrBags = (*AllocFuncBags)( initial_size, 1 );
    if ( MptrBags == 0 )
        (*AbortFuncBags)("cannot get storage for the initial workspace.");
    EndBags = MptrBags + 1024*(initial_size / sizeof(BagPtr_t*));

    /* 1/8th of the storage goes into the masterpointer area               */
    FreeMptrBags = (BagPtr_t)MptrBags;
    for ( p = MptrBags;
          p + 2*(SIZE_MPTR_BAGS) <= MptrBags+1024*initial_size/8/sizeof(BagPtr_t*);
          p += SIZE_MPTR_BAGS )
    {
        *p = (BagPtr_t)(p + SIZE_MPTR_BAGS);
    }

    /* the rest is for bags                                                */
    OldBags   = MptrBags + 1024*initial_size/8/sizeof(BagPtr_t*);
    YoungBags = OldBags;
    AllocBags = OldBags;

    /* remember the cache size                                             */
    CacheSizeBags = cache_size;
    if ( ! CacheSizeBags ) {
        AllocSizeBags = 256;
        StopBags = EndBags;
    }
    else {
        AllocSizeBags = (CacheSizeBags+1023)/1024;
        StopBags  = AllocBags + WORDS_BAG(1024*AllocSizeBags) <= EndBags ?
                    AllocBags + WORDS_BAG(1024*AllocSizeBags) :  EndBags;
    }

    /* remember whether bags should be clean                               */
    DirtyBags = dirty;

    /* install the marking functions                                       */
    for ( i = 0; i < NTYPES; i++ )
        TabMarkFuncBags[i] = MarkAllSubBagsDefault;

    /* Set ChangedBags to a proper initial value */
    ChangedBags = 0;

}


/****************************************************************************
**
*F  NewBag4( <type>, <size> )  . . . . . . . . . . . . . .  allocate a new bag
**
**  'NewBag4' is actually quite simple.
**
**  It first tests whether enough storage is available in the allocation area
**  and  whether a free   masterpointer is available.   If  not, it  starts a
**  garbage collection by calling 'CollectBags' passing <size> as the size of
**  the bag it is currently allocating and 0 to indicate  that only a partial
**  garbage collection is called for.   If 'CollectBags' fails and returns 0,
**  'NewBag4' also fails and also returns 0.
**
**  Then it takes the first free  masterpointer from the  linked list of free
**  masterpointers (see "FreeMptrBags").
**
**  Then it  writes  the  size and the   type  into the word   pointed to  by
**  'AllocBags'.  Then  it writes the identifier,  i.e.,  the location of the
**  masterpointer, into the next word.
**
**  Then it advances 'AllocBags' by 'HEADER_SIZE + WORDS_BAG(<size>)'.
**
**  Finally it returns the identifier of the new bag.
**
**  Note that 'NewBag4' never  initializes the new bag  to contain only 0.  If
**  this is desired because  the initialization flag <dirty> (see "InitBags")
**  was  0, it is the job  of 'CollectBags'  to initialize the new free space
**  after a garbage collection.
**
**  If {\Gasman} was compiled with the option 'COUNT_BAGS' then 'NewBag4' also
**  updates the information in 'InfoBags' (see "InfoBags").
**
**  'NewBag4'  is implemented as  a  function  instead of a  macro  for  three
**  reasons.  It  reduces the size of  the program, improving the instruction
**  cache  hit ratio.   The compiler  can do  anti-aliasing analysis  for the
**  local  variables  of the function.  To  enable  statistics only {\Gasman}
**  needs to be recompiled.
*/
BagPtr_t NewBag4 ( UInt type, UInt size )
{
    BagPtr_t                 bag;            /* identifier of the new bag       */
    BagPtr_t *               dst;            /* destination of the new bag      */

#ifdef TREMBLE_HEAP
    CollectBags(0,0);
#endif    

    /* check that a masterpointer and enough storage are available         */
    if ( ( ((FreeMptrBags < MptrBags) || (FreeMptrBags >= OldBags)) ||
		   StopBags < AllocBags+HEADER_SIZE+WORDS_BAG(size) ) && CollectBags( size, 0 ) == 0 )
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

    /* get the identifier of the bag and set 'FreeMptrBags' to the next    */
    bag          = FreeMptrBags;
	if (bag == 0) {
		// no more free bags available ==> out of memory, exit
		GuFatalMsgExit(EXIT_MEM, "Newbag4: FreeMptrBags chain is empty, no more memory\n");
	}
	else if (bag < MptrBags || bag >= OldBags) {
		// Not a valid bag handle, head of free chain is corrupt, print message & exit
		GuFatalMsgExit(EXIT_MEM, "Newbag4: FreeMptrBags chain head is corrupt ... exiting\n");
	}
    FreeMptrBags = *(BagPtr_t*)bag;

    /* allocate the storage for the bag                                    */
    dst       = AllocBags;
    AllocBags = dst + HEADER_SIZE + WORDS_BAG(size);

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
	
    /* return the identifier of the new bag                                */
    return bag;
}


/****************************************************************************
**
*F  RetypeBag(<bag>,<new>)  . . . . . . . . . . . .  change the type of a bag
**
**  'RetypeBag' is very simple.
**
**  All it has to do is to change the size-type word of the bag.
**
**  If  {\Gasman} was compiled with the  option 'COUNT_BAGS' then 'RetypeBag'
**  also updates the information in 'InfoBags' (see "InfoBags").
*/
void            RetypeBag (
    BagPtr_t                 bag,
    UInt                new_type )
{
    UInt                size;           /* size of the bag                 */

    /* get old type and size of the bag                                    */
    size     = GET_SIZE_BAG(bag);

#ifdef  COUNT_BAGS
    /* update the statistics      */
    {
          UInt                old_type;       /* old type of the bag */

	  old_type = TNUM_BAG(bag);
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

    /* change the size-type word                                           */
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
**  the size-type word of the bag.
**
**  If the bag is to be shrunk and at least one word becomes free, 'ResizeBag'
**  changes the size-type word of the bag, and stores a magic size-type word
**  in the first free word.  This magic size-type word has type 255
**  (T_RESIZE_FREE) and the size is the number of following free bytes, which
**  is always divisible by 'sizeof(BagPtr_t)'.  The type 255 (T_RESIZE_FREE)
**  allows 'CollectBags' to detect that this body is the remainder of a resize
**  operation, and the size allows it to know how many bytes there are in this
**  body (see "Implementation of CollectBags").
**
**  So for example if 'ResizeBag' shrinks a bag of type 7 from 18 bytes to 10
**  bytes the situation before 'ResizeBag' is as follows{\:}
**
**    +---------+
**    |<masterp>|
**    +---------+
**         \_____________
**                       \
**                        V
**    +---------+---------+--------------------------------------------+----+
**    | 18  . 7 |  <link> |         .         .         .         .    | pad|
**    +---------+---------+--------------------------------------------+----+
**
**  And after 'ResizeBag' the situation is as follows{\:}
**
**    +---------+
**    |<masterp>|
**    +---------+
**         \_____________
**                       \
**                        V
**    +---------+---------+------------------------+----+---------+---------+
**    | 10  . 7 |  <link> |         .         .    | pad|  4  .255|         |
**    +---------+---------+------------------------+----+---------+---------+
**
**  If the bag is to be extended and it  is that last  allocated bag, so that
**  it  is  immediately adjacent  to the allocation  area, 'ResizeBag' simply
**  increments  'AllocBags' after making  sure that enough space is available
**  in the allocation area (see "Layout of the Workspace").
**
**  If the bag is to be extended and it is not the last allocated bag,
**  'ResizeBag' first allocates a new bag similar to 'NewBag4', but without
**  using a new masterpointer.  Then it copies the old contents to the new
**  bag.  Finally it resets the masterpointer of the bag to point to the new
**  address.  Then it changes the type of the old body to 255 (T_RESIZE_FREE),
**  so that the garbage collection can detect that this body is the remainder
**  of a resize (see "Implementation of NewBag4" and "Implementation of
**  CollectBags").
**
**  When an old bag is extended, it  will now reside  in the young bags area,
**  and thus appear  to be young.   Since old bags are   supposed to  survive
**  partial garbage  collections 'ResizeBag'  must   somehow protect this bag
**  from partial garbage collections.  This is  done by putting this bag onto
**  the linked  list  of  changed bags (see   "ChangedBags").  When a partial
**  garbage collection sees a young bag on the list of changed bags, it knows
**  that it is the result of 'ResizeBag' of an old bag, and does not throw it
**  away (see "Implementation of CollectBags").  Note  that  when 'ResizeBag'
**  tries this, the bag may already be on the linked  list, either because it
**  has been resized earlier, or because  it has been  changed.  In this case
**  'ResizeBag' simply keeps the bag on this linked list.
**
**  If {\Gasman}  was compiled with the  option 'COUNT_BAGS' then 'ResizeBag'
**  also updates the information in 'InfoBags' (see "InfoBags").
*/

UInt ResizeBag ( BagPtr_t bag, UInt new_size )
{
    UInt                type;           /* type of the bag                 */
    UInt                old_size;       /* old size of the bag             */
    UInt                flags;
    BagPtr_t *               dst;            /* destination in copying          */
    BagPtr_t *               src;            /* source in copying               */
    BagPtr_t *               end;            /* end in copying                  */

    /* check the size                                                      */
    
#ifdef TREMBLE_HEAP
    CollectBags(0,0);
#endif

    /* get type and old size of the bag                                    */
    type     = TNUM_BAG(bag);
    old_size = GET_SIZE_BAG(bag);
    flags    = GET_FLAGS_BAG(bag);

#ifdef  COUNT_BAGS
    /* update the statistics                                               */
    SizeAllBags             += new_size - old_size;
    InfoBags[type].sizeLive += new_size - old_size;
    InfoBags[type].sizeAll  += new_size - old_size;
#endif

    /* if the real size of the bag doesn't change                          */
    if ( WORDS_BAG(new_size) == WORDS_BAG(old_size) ) {

        /* change the size word                                            */
		SET_SIZE_BAG(bag, new_size);
    }

    /* if the bag is shrunk                                                */
    /* we must not shrink the last bag by moving 'AllocBags',              */
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
    }

    /* if the last bag is to be enlarged                                   */
    else if ( PTR_BAG(bag) + WORDS_BAG(old_size) == AllocBags ) {
		
        /* check that enough storage for the new bag is available          */
        if ( StopBags < PTR_BAG(bag)+WORDS_BAG(new_size)
          && CollectBags( new_size-old_size, 0 ) == 0 ) {
            return 0;
        }

        /* simply increase the free pointer                                */
        if ( YoungBags == AllocBags )
            YoungBags += WORDS_BAG(new_size) - WORDS_BAG(old_size);
        AllocBags += WORDS_BAG(new_size) - WORDS_BAG(old_size);

        /* change the size-type word                                       */
        SET_SIZE_BAG(bag, new_size ) ;
    }

    /* if the bag is enlarged                                              */
    else {

        /* check that enough storage for the new bag is available          */
        if ( StopBags < AllocBags+HEADER_SIZE+WORDS_BAG(new_size)
          && CollectBags( new_size, 0 ) == 0 ) {
            return 0;
        }

        /* allocate the storage for the bag                                */
        dst       = AllocBags;
        AllocBags = dst + HEADER_SIZE + WORDS_BAG(new_size);
	
        /* leave magic size-type word  for the sweeper, type must be 255 (T_RESIZE_FREE)  */
		SET_TYPE_BAG(bag, T_RESIZE_FREE);
		BLANK_FLAGS_BAG(bag);
        SET_SIZE_BAG(bag, (((WORDS_BAG(old_size) + HEADER_SIZE - 1) * sizeof(BagPtr_t))));
	
        /* enter the new size-type word                                    */

		SET_TYPE_PTR(dst,type);
		BLANK_FLAGS_PTR(dst);
		SET_SIZE_PTR(dst, new_size);

        /* if the bag is already on the changed bags list, keep it there   */
        if ( GET_LINK_BAG(bag) != bag ) {
            SET_LINK_PTR(dst, GET_LINK_BAG(bag));
        }

        /* if the bag is old, put it onto the changed bags list            */
        else if ( PTR_BAG(bag) <= YoungBags ) {
            SET_LINK_PTR(dst, ChangedBags);  
			ChangedBags = bag;
        }

        /* if the bag is young, enter the normal link word                 */
        else {
            SET_LINK_PTR(dst, bag);
        }

		dst = dst + HEADER_SIZE;

        /* set the masterpointer                                           */
        src = PTR_BAG(bag);
        end = src + WORDS_BAG(old_size);
        SET_PTR_BAG(bag, dst);
		SET_FLAG_BAG(bag, flags);

        /* copy the contents of the bag                                    */
        while ( src < end )
            *dst++ = *src++;

    }

    /* return success                                                      */
    return 1;
}


/****************************************************************************
**
*F  CollectBags( <size>, <full> ) . . . . . . . . . . . . . collect dead bags
**
**  'CollectBags' is the function that does most of the work of {\Gasman}.
**
**  A partial garbage collection where  every bag is  young is clearly a full
**  garbage    collection.  So  to     perform  a  full  garbage  collection,
**  'CollectBags' first sets 'YoungBags'  to   'OldBags', making every    bag
**  young, and empties the list  of changed old bags, since  there are no old
**  bags anymore, there  can be no changed old  bags anymore.  So from now on
**  we    can   assume that  'CollectBags'     is doing   a  partial  garbage
**  collection.   In  addition,    the   values 'NewWeakDeadBagMarker'    and
**  'OldWeakDeadBagMarker'  are exchanged, so  that bag idnetifiers that have
**  been  halfdead  since    before  this full    garbage  collection cab  be
**  distinguished from those which have died on this pass.
**
**  Garbage collection  is  performed in  three phases.  The  mark phase, the
**  sweep phase, and the check phase.
**
**  In the  *mark phase*, 'CollectBags' finds  all young bags that  are still
**  live and builds a linked list of those bags (see "MarkedBags").  A bag is
**  put on  this  list  of  marked bags   by   applying  'MARK_BAG' to    its
**  identifier.  Note that 'MARK_BAG' checks that a bag is not already on the
**  list of marked bags, before it puts it on the list, so  no bag can be put
**  twice on this list.
**
**  First, 'CollectBags' marks  all  young bags that are  directly accessible
**  through global   variables,  i.e.,  it   marks those young     bags whose
**  identifiers  appear  in  global variables.   It    does this  by applying
**  'MARK_BAG'  to the values at the  addresses  of global variables that may
**  hold bag identifiers provided by 'InitGlobalBag' (see "InitGlobalBag").
**
**  Next,  'CollectBags' marks  all  young bags  that are directly accessible
**  through   local  variables, i.e.,    it  marks those  young   bags  whose
**  identifiers appear in the  stack.   It  does  this by calling  the  stack
**  marking  function  <stack-func>  (see  "InitBags").   The   generic stack
**  marking function, which is called if <stack-func> (see "InitBags") was 0,
**  is described below.  The problem is  that there is usually not sufficient
**  information  available to decide  if a value on   the stack is really the
**  identifier of a bag, or is a  value of another  type that only appears to
**  be the  identifier  of a bag.  The  position  usually taken by  the stack
**  marking function is that everything  on the stack  that could possibly be
**  interpreted  as the identifier of  a bag is an  identifier of  a bag, and
**  that this bag is therefore live.  This position is what makes {\Gasman} a
**  conservative storage manager.
**
**  The generic stack marking function 'GenStackFuncBags', which is called if
**  <stack-func> (see "InitBags") was 0, works by  applying 'MARK_BAG' to all
**  the values on the stack,  which is supposed to extend  from <stack-start>
**  (see  "InitBags") to the address of  a local variable of   the  function.
**  Note that some local variables may  not  be stored on the  stack, because
**  they are  still in the processors registers.    'GenStackFuncBags' uses a
**  jump buffer 'RegsBags', filled by the C library function 'setjmp', marking
**  all bags  whose  identifiers appear in 'RegsBags'.  This  is a dirty hack,
**  that need not work, but actually works on a  surprisingly large number of
**  machines.  But it will not work on Sun  Sparc machines, which have larger
**  register  files, of which  only the part  visible to the current function
**  will be saved  by  'setjmp'.  For those machines 'GenStackFuncBags' first
**  calls the operating system to flush the whole register file.  Note that a
**  compiler may save  a register  somewhere else  if   it wants to  use this
**  register for something else.  Usually  this register is saved  further up
**  the  stack,  i.e.,   beyond the   address  of  the  local variable,   and
**  'GenStackFuncBags' would not see this value any more.   To deal with this
**  problem, 'setjmp' must be called *before* 'GenStackFuncBags'  is entered,
**  i.e.,  before the  registers may have been saved  elsewhere.   Thus it is
**  called from 'CollectBags'.
**
**  Next 'CollectBags' marks all young bags that are directly accessible from
**  old bags, i.e.,  it marks all young bags  whose identifiers appear in the
**  data areas  of  old bags.  It  does  this by applying 'MARK_BAG'  to each
**  identifier appearing in changed old bags, i.e., in those bags that appear
**  on the list of changed old bags (see "ChangedBags").   To be more precise
**  it calls the  marking function for the appropriate  type to  each changed
**  old bag (see "InitMarkFuncBags").  It need not apply the marking function
**  to each old  bag, because old bags that  have not been changed  since the
**  last garbage  collection cannot contain identifiers  of young bags, which
**  have been allocated since the last garbage collection.  Of course marking
**  the subbags of only  the changed old  bags is more efficient than marking
**  the subbags of  all old bags only  if the number of  changed old  bags is
**  smaller than the total number of old bags, but this  is a very reasonable
**  assumption.
**
**  Note that there may also be bags that  appear to be  young on the list of
**  changed old bags.  Those bags  are old bags that  were extended since the
**  last garbage  collection and therefore have their  body in the young bags
**  area (see "Implementation of  ResizeBag").  When 'CollectBags' finds such
**  a bag  on  the list of  changed  old bags  it  applies 'MARK_BAG'  to its
**  identifier and thereby  ensures that this bag will  not be thrown away by
**  this garbage collection.
**
**  Next,  'CollectBags'    marks all  young    bags  that  are  *indirectly*
**  accessible, i.e., it marks the subbags of  the already marked bags, their
**  subbags  and  so on.  It  does  so by walking   along the list of already
**  marked bags and applies  the marking function  of the appropriate type to
**  each bag on this list (see  "InitMarkFuncBags").  Those marking functions
**  then apply 'MARK_BAG' or 'MarkBagWeakly'  to each identifier appearing in
**  the bag.
**
**  After  the marking function has  been  applied to a   bag on the list  of
**  marked bag, this bag is removed from the list.  Thus the marking phase is
**  over when the list  of marked bags   has become empty.  Removing the  bag
**  from the list of marked  bags must be done  at  this time, because  newly
**  marked bags are *prepended* to the list of  marked bags.  This is done to
**  ensure that bags are marked in a  depth first order, which should usually
**  improve locality of   reference.  When a bag is   taken from the list  of
**  marked bags it is *tagged*.  This tag serves two purposes.  A bag that is
**  tagged is not put on the list  of marked bags  when 'MARK_BAG' is applied
**  to its identifier.  This ensures that  no bag is put  more than once onto
**  the list of marked bags, otherwise endless marking loops could happen for
**  structures that contain circular  references.  Also the sweep phase later
**  uses the presence of  the tag to decide the  status of the bag. There are
**  three possible statuses: LIVE, DEAD and  HALFDEAD. The default state of a
**  bag with its identifier in the link word, is  the tag for DEAD. Live bags
**  are tagged    with  MARKED_ALIVE(<identifier>)  in the   link   word, and
**  half-dead bags (ie bags pointed to weakly but not strongly) with the tage
**  MARKED_HALFDEAD(<identifier>).
** 
**  Note that 'CollectBags' cannot put a random or magic  value into the link
**  word, because the sweep phase must be able to find the masterpointer of a
**  bag by only looking at the link word of a bag. This is done using the macros
**  UNMARKED_XXX(<link word contents>).
**
**  In the   *sweep  phase*, 'CollectBags'   deallocates all dead   bags  and
**  compacts the live bags at the beginning of the workspace.
**
**  In this  phase 'CollectBags'   uses  a destination pointer   'dst', which
**  points to  the address a  body  will be copied to,   and a source pointer
**  'src',  which points to the address  a body currently has.  Both pointers
**  initially   point to  the   beginning  of  the   young bags area.    Then
**  'CollectBags' looks at the body pointed to by the source pointer.
**
**  If this body has type 255 (T_RESIZE_FREE), it is the remainder of a resize
**  operation.  In this case 'CollectBags' simply moves the source pointer to
**  the next body (see "Implementation of ResizeBag").
**
**
**  Otherwise, if the  link word contains the  identifier of the bag  itself,

**  marked dead,  'CollectBags' first adds the masterpointer   to the list of
**  available masterpointers (see  "FreeMptrBags") and then simply  moves the
**  source pointer to the next bag.
**
**  Otherwise, if the link  word contains  the  identifier of the  bag marked
**  alive, this   bag is still  live.  In  this case  'CollectBags' calls the
**  sweeping function for this bag, if one  is installed, or otherwise copies
**  the body from  the source address to the  destination address, stores the
**  address of the masterpointer   without the tag  in   the link word,   and
**  updates the masterpointer to point to the new address of the data area of
**  the bag.   After the copying  the source pointer points  to the next bag,
**  and the destination pointer points just past the copy.
**
**  Finally, if the link word contains the identifier of  the bag marked half
**  dead, then  'CollectBags' puts  the special value  'NewWeakDeadBagMarker'
**  into the masterpointer corresponding to the bag, to signify that this bag
**  has been collected as garbage.
**
**  This is repeated until  the source pointer  reaches the end of  the young
**  bags area, i.e., reaches 'AllocBags'.
**
**  The new free storage now is the area between  the destination pointer and
**  the source pointer.  If the initialization flag  <dirty> (see "InitBags")
**  was 0, this area is now cleared.
**
**  Next, 'CollectBags' sets   'YoungBags'  and 'AllocBags'  to   the address
**  pointed to by the destination  pointer.  So all the  young bags that have
**  survived this garbage  collection are now  promoted  to be old  bags, and
**  allocation of new bags will start at the beginning of the free storage.
**
**  Finally, the *check phase* checks  whether  the garbage collection  freed
**  enough storage and masterpointers.
**
**  After a partial garbage collection,  'CollectBags' wants at least '<size>
**  + AllocSizeBags' bytes  of free  storage  available, where <size> is  the
**  size of the bag that 'NewBag4' is  currently trying to allocate.  Also the
**  number  of free masterpointers should be  larger than the  number of bags
**  allocated since   the  previous garbage collection  plus 4096  more to be
**  safe.   If less free   storage  or  fewer masterpointers  are  available,
**  'CollectBags' calls itself for a full garbage collection.
**
**  After a  full  garbage collection,  'CollectBags' wants at   least <size>
**  bytes of free storage available, where <size> is the size of the bag that
**  'NewBag4' is  currently trying  to allocate.  Also it  wants  at least one
**  free  masterpointer.    If less free    storage or   no masterpointer are
**  available, 'CollectBags'  tries   to   extend  the  workspace   using the
**  allocation   function <alloc-func> (see    "InitBags").   If <alloc-func>
**  refuses  to extend  the   workspace,  'CollectBags' returns 0 to indicate
**  failure to 'NewBag4'.  In  any case 'CollectBags' will  try to  extend the
**  workspace so that at least one eigth of the storage is free, that is, one
**  eight of the storage between 'OldBags' and 'EndBags' shall be  free.   If
**  <alloc-func> refuses this extension of the workspace, 'CollectBags' tries
**  to get along with  what it  got.   Also 'CollectBags' wants at least  one
**  masterpointer per 8 words of free storage available.  If  this is not the
**  case, 'CollectBags'  extends the masterpointer area  by moving the bodies
**  of all bags and readjusting the masterpointers.  
**
**  Also,  after   a  full  garbage   collection,  'CollectBags'   scans  the
**  masterpointer area for  identifiers containing 'OldWeakDeadBagMarker'. If
**  the sweep functions have done their work then no  references to these bag
**  identifiers can exist, and so 'CollectBags' frees these masterpointers.  
*/
#include <setjmp.h>

static jmp_buf RegsBags;

/* solaris */
#ifdef __GNUC__
#ifdef SPARC
#if SPARC
        asm("           .globl  SparcStackFuncBags              ");
        asm("   SparcStackFuncBags:                             ");
        asm("           ta      0x3     ! ST_FLUSH_WINDOWS      ");
        asm("           mov     %sp,%o0                         ");
        asm("           retl                                    ");
        asm("           nop                                     ");
#endif
#endif
#else

/* sunos */
#if defined(SPARC) 
void SparcStackFuncBags( void )
{
  asm (" ta 0x3 ");
  asm (" mov %sp,%o0" );
  return;
}
#endif

#endif

void GenStackFuncBags (void)
{
    BagPtr_t *               top;            /* top of stack                    */
    BagPtr_t *               p;              /* loop variable                   */
    UInt                i;              /* loop variable                   */

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

UInt FullBags;

		
#ifdef  DEBUG_DEADSONS_BAGS
BagPtr_t OldMarkedBags;

static void CheckOldNewBagRefs()
{
   /* check for old bags pointing to new unmarked bags                    */
    BagPtr_t p = OldBags;
    UInt         pos;
    BagPtr_t	 first;

    OldMarkedBags = MarkedBags;
    while ( p < YoungBags ) {
        if ( GET_TYPE_PTR(p) == T_RESIZE_FREE ) {
			if ( TEST_FLAG_PTR(p, BF_COPY) ) 
				p++;
			else
				p += 1 + WORDS_BAG( GET_SIZE_PTR(p) );
		
        }
        else {
            (*TabMarkFuncBags[TNUM_BAG(GET_LINK_PTR(p))])( GET_LINK_PTR(p) );
            pos = 0;
            while ( MarkedBags != OldMarkedBags ) {
                Pr( "#W  Old bag (type %s, size %d, ",
                    (Int)InfoBags[ TNUM_BAG(GET_LINK_PTR(p)) ].name,
                    (Int)GET_SIZE_BAG(GET_LINK_PTR(p)) );
                Pr( "handle %d, pos %d) points to\n",
                    (Int)GET_LINK_PTR(p),
                    (Int)pos );
                Pr( "#W    new bag (type %s, size %d, ",
                    (Int)InfoBags[ TNUM_BAG(MarkedBags) ].name,
                    (Int)GET_SIZE_BAG(MarkedBags) );
                Pr( "handle %d)\n",
                    (Int)MarkedBags,
                    (Int)0 );
                pos++;
                first = GET_LINK_BAG(MarkedBags);
                SET_LINK_BAG(MarkedBags, MarkedBags);
                MarkedBags = first;
            }
            p += HEADER_SIZE + WORDS_BAG( GET_SIZE_PTR(p) );
        }
    }
	return;
}
#endif


static void PrintBagInfo(BagStruct_t *ps)
{
	printf("PrintBagInfo: ptr = %p, Type = %d, size = %u, link = %p, # sub bags = %u\n",
		   ps, (ps->bagFlagsType & TYPE_BIT_MASK), ps->bagSize, ps->bagLinkPtr, 
		   NrHandles((ps->bagFlagsType & TYPE_BIT_MASK), ps->bagSize));
	return;
}

static void CheckFreeMptrList()
{
	BagPtr_t start;
	BagStruct_t *ptr;
	UInt nFound = 0, nBad = 0, nFree = 0;
	
	// Walk the master pointer area to see if we're clean there
	start = FreeMptrBags;
	while (start) {
		if (*start >= OldBags && *start < AllocBags) {
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
		else if (*start >= MptrBags && *start < OldBags) {
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
	printf("CheckFreeMptrList: Walked FreeMptrBags list: Found links to %u bags (%s), %u (%.1f%%) master ptrs free, and %u suspect Handle/Bag linkages\n",
		   nFound, ((nFound == 0)? "GOOD" : "BAD"),
		   nFree, (100.0 * (float)nFree / (float)(((UInt)OldBags - (UInt)MptrBags) / sizeof(BagPtr_t *))), nBad);
	return;
}

static void CheckMptrHandles()
{
	BagPtr_t start;
	BagStruct_t *ptr;
	UInt nFound = 0, nBad = 0;
	
	// Walk the master pointer area, to ensure all used handles point to valid bags
	start = MptrBags;
	while (start < OldBags) {
		if (*start >= OldBags && *start < AllocBags) {
			// found a link to a bag, count it
			nFound++;
			ptr = (BagStruct_t *)(*start - HEADER_SIZE);
			if (*start != &ptr->bagData) {
				// a bag handle points to something not recognizable or badly linked
				nBad++;
				printf("CheckMptrHandles: Handle/Bag link not valid: handle = %p, handle link = %p, bag data address = %p\n",
					   start, *start, &ptr->bagData);
			}
		}
		else {
			// it's a free Mptr handle -- checked free list elsewhere -- noop
		}
		start++;
	}
	printf("CheckMptrHandles: Walked master pointer area: Found links to %u (%.1f%%) bags, and %u suspect Handle/Bag linkages (%s)\n",
		   nFound, (100.0 * (float)nFound / (float)(((UInt)OldBags - (UInt)MptrBags) / sizeof(BagPtr_t))),
		   nBad, ((nBad == 0)? "GOOD" : "BAD"));
	
	return;
}

static void WalkBagPointers()
{
	BagPtr_t start, end, foo;
	BagStruct_t *ptr;
	UInt type, nbad = 0, szbad = 0;
	UInt nFound = 0, szFound = 0, sizeCurr;
	
	/*
	 * walk the actual bags -- assumptions is we're called after GC, so
	 * everything should cleanly setup with no holes from OldBags to
	 * AllocBags; at this point AllocBags should = YoungBags (i.e., no new
	 * bags have been created since GC done).  
	 */
	start = OldBags; end = AllocBags;
	printf("WalkBagPointers: MptrBags = %p, walk OldBags = %p to AllocBags = %p, pool = %uk (%uMb)\n", 
		   MptrBags, start, end, ((UInt)end - (UInt)start)/1024, ((UInt)end - (UInt)start)/(1024*1024));
	printf("WalkBagPointers: YoungBags = %p, AllocBags = %p, EndBags = %p, free pool = %uk (%uMb)\n",
		   YoungBags, AllocBags, EndBags, ((UInt)EndBags - (UInt)AllocBags)/1024, ((UInt)EndBags - (UInt)AllocBags)/(1024*1024));
	
	while (start < (end - 1)) {			/* last mptr in list is 0 */
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
		if (ptr->bagLinkPtr && MptrBags <= ptr->bagLinkPtr && ptr->bagLinkPtr < OldBags) {
			BagPtr_t foo = ptr->bagLinkPtr, bar = *ptr->bagLinkPtr;
			if (bar != &ptr->bagData) {
				// we're foobar'd
				printf("WalkBagPointers: Suspect misaligned handle/link pointer: link = %p, handle = %p, *handle = %p, data ptr = %p\n",
					   foo, bar, *bar, &ptr->bagData);
			}
		}
	}

	printf("WalkBagPointers: Walked the bags, found %u bags, size = %uk (%uMb)\n",
		   nFound, szFound / 1024, szFound / (1024 * 1024));
	printf("WalkBagPointers: Found %u suspect bags, total size = %uk (%uMb)\n",
		   nbad, szbad / 1024, szbad / (1024 * 1024));
	/*
	 * Walk the remaining memory, which should be the free pool.  It should
	 * all have been initialized to zeros...
	 * Things appear to be all messed up if DirtyBags != 0
	 */
//  if (!DirtyBags) {
	start = AllocBags; end = EndBags; nbad = 0;
	while (start < end) {
		if (*start)
			nbad++;
		
		start++;
	}
	
	if (nbad)
		printf("WalkBagPointers: Walked the free area, %u non-zero values -- pool is dirty!\n",
			   nbad);
	else
		printf("WalkBagPointers: Walked the free area -- pool is CLEAN!\n");
//  }
	
	float ptrpc = 100.0 * (float)((UInt)OldBags - (UInt)MptrBags)  / (float)((UInt)EndBags - (UInt)MptrBags),
		usedpc  = 100.0 * (float)((UInt)YoungBags - (UInt)OldBags) / (float)((UInt)EndBags - (UInt)MptrBags),
		freepc  = 100.0 * (float)((UInt)EndBags - (UInt)AllocBags) / (float)((UInt)EndBags - (UInt)MptrBags);
	
	printf("WalkBagPointers: Total pool = %uMb (%.1f%%), Master pointers = %uMb (%.1f%%), Live Bags = %uMb (%.1f%%), Free pool = %uMb (%.1f%%)\n",
		   (((UInt)EndBags - (UInt)MptrBags) / (1024 * 1024)), 100.0,
		   (((UInt)OldBags - (UInt)MptrBags) / (1024 * 1024)), ptrpc,
		   (((UInt)YoungBags - (UInt)OldBags) / (1024 * 1024)), usedpc,
		   (((UInt)EndBags - (UInt)AllocBags) / (1024 * 1024)), freepc);
	
	CheckFreeMptrList();
	CheckMptrHandles();
	
	return;
}


/*
 * These are used to overwrite masterpointers which may still be linked from
 * weak pointer objects but whose bag bodies have been collected.  Two values
 * are used so that old masterpointers of this kind can be reclaimed after a
 * full garbage collection. The values must not look like valid pointers, and
 * should be congruent to 1 mod sizeof(BagPtr_t)
 */

BagPtr_t * NewWeakDeadBagMarker = (BagPtr_t *)(1000*sizeof(BagPtr_t) + NUM_TO_UINT(1));
BagPtr_t * OldWeakDeadBagMarker = (BagPtr_t *)(1001*sizeof(BagPtr_t) + NUM_TO_UINT(1)); 

/*
 *  Gather information about the bags (all, including dead & remnants) and
 *  build a histogram for output.  For very large bags print information for
 *  these (statistics are accumulated for all bags over 2,000 words in size,
 *  so we won't have a view into them otherwise.
 */

typedef struct {
	Int4	size;	 					/* size of the bag or remnant (size in bytes incl header) */
	Int4	count;						/* count of bags this size */
	Int4	nlive;						/* number of live bags */
	Int4	ndead;						/* number of dead bags */
	Int4	nremnant;					/* number of remants */
	Int4	nhalfdead;					/* number of half dead */
} BagHistogram_t;

typedef enum {
	INCREMENT_LIVE,						/* increment the live bags stats */
	INCREMENT_DEAD,						/* dead */
	INCREMENT_REMNANT,					/* remnant */
	INCREMENT_HALFDEAD					/* half-dead */
}  countType_t;

#define SIZE_HIST 2000					  /* track up to 2000 different sizes */
static BagHistogram_t BagSizeCount[SIZE_HIST]; /* index by WORDS_BAG */
static Int4 countHistOn = 0;	

static void IncrementBagHistogram(Int4 size_w, countType_t typ, BagPtr_t * bagp)
{
	// increment stats for this bag size; print info for bags >= SIZE_HIST words
	BagStruct_t *ptr = (BagStruct_t *)bagp;
	int bagtype = ptr->bagFlagsType & TYPE_BIT_MASK;
	int sz;

	if (size_w >= SIZE_HIST) {
		sz = ptr->bagSize + HEADER_SIZE * sizeof(BagPtr_t);
//		printf("Big bag: Type = %d, size = %db (%dW), # subbags =%d, Status = %s\n",
//			   bagtype, sz, size_w, NrHandles(bagtype, ptr->bagSize),
//			   (typ == INCREMENT_LIVE) ? "Live" : (typ == INCREMENT_DEAD) ? "Dead" :
//			   (typ == INCREMENT_REMNANT) ? "Remnant" : "Halfdead" );
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
	case INCREMENT_LIVE:	{
		BagSizeCount[size_w].nlive++;
		InfoBags[bagtype].nrLive++;
		InfoBags[bagtype].sizeLive += sz;
		break;
	}
	case INCREMENT_DEAD:	{ BagSizeCount[size_w].ndead++;		InfoBags[bagtype].nrDead++;		  break; }
	case INCREMENT_REMNANT:	{ BagSizeCount[size_w].nremnant++;	InfoBags[bagtype].nrRemnant++;	  break; }
	case INCREMENT_HALFDEAD:{ BagSizeCount[size_w].nhalfdead++;	InfoBags[bagtype].nrHalfdead++;	  break; }
	}
	return;
}

static void DumpBagsHistogram()
{
	// for each populated entry in the histogram table print the information
	Int4 ii;

//	printf("DumpBagsHistogram: Stats for all bags (dead/alive/etc.) found in current GC run\n");
//	printf("Size\tTotal\tLive\tDead\tRemnant\tHalfdead\n");
	
//	for (ii = 0; ii < 2000; ii++) {
//		if (BagSizeCount[ii].count > 0)  { /* found a bag of this size */
//			printf("%7d\t%7d\t%7d\t%7d\t%7d\t%7d\n",
//				   BagSizeCount[ii].size, BagSizeCount[ii].count, BagSizeCount[ii].nlive,
//				   BagSizeCount[ii].ndead, BagSizeCount[ii].nremnant, BagSizeCount[ii].nhalfdead);
//		}
//	}

	// for each populated entry in the InfoBags table print the information
	printf("\nDumpBagsHistogram: Stats by bag type found in current GC run\n");
	printf("Type\tTotal\tLive\tDead\tRemnant\tHalfdead\tSize Live\tType Name\n");

	for (ii = 0; ii < NTYPES; ii++)   {
		if (InfoBags[ii].nrAll > 0)  {	/* skip types not active */
			if (InfoBags[T_RESIZE_FREE].name == (char *)NULL)  {
				InfoBags[T_RESIZE_FREE].name = "Resize remnant (free)";
			}
			printf("%7d\t%7d\t%7d\t%7d\t%7d\t%7d\t%9u\t%s\n",
				   ii, InfoBags[ii].nrAll, InfoBags[ii].nrLive, InfoBags[ii].nrDead,
				   InfoBags[ii].nrRemnant, InfoBags[ii].nrHalfdead, InfoBags[ii].sizeLive, InfoBags[ii].name);
		}
	}
	
	return;
}


UInt CollectBags ( UInt size, UInt full )
{
    BagPtr_t		first;          /* first bag on a linked list      */
    BagPtr_t *      p;              /* loop variable                   */
    BagPtr_t *      dst;            /* destination in sweeping         */
    BagPtr_t *      src;            /* source in sweeping              */
    BagPtr_t *      end;            /* end of a bag in sweeping        */
    UInt            nrLiveBags;     /* number of live new bags         */
    UInt            sizeLiveBags;   /* total size of live new bags     */
    UInt            nrDeadBags;     /* number of dead new bags         */
    UInt            nrHalfDeadBags; /* number of dead new bags         */
    UInt            sizeDeadBags;   /* total size of dead new bags     */
    UInt            done;           /* do we have to make a full gc    */
    UInt            i;              /* loop variable                   */
	Int4			isz;
    BagPtr_t *      last;
    Char            type;

#ifdef DEBUG_MASTERPOINTERS
    CheckMasterPointers();
#endif

	if (SyMemMgrTrace > 0) {
		printf("CollectBags:E: MptrBags = %p, OldBags = %p, size Mptr area = %uk, # mptrs = %u\n",
			   MptrBags, OldBags, ((UInt)OldBags - (UInt)MptrBags)/1024,
			   ((UInt)OldBags - (UInt)MptrBags)/sizeof(BagPtr_t));
		printf("CollectBags:E: Oldbags = %p, YoungBags = %p, Old Bags area = %uk\n",
			   OldBags, YoungBags, ((UInt)YoungBags - (UInt)OldBags)/1024);
		printf("CollectBags:E: Youngbags = %p, AllocBags = %p, Young Bags area = %uk\n",
			   YoungBags, AllocBags, ((UInt)AllocBags - (UInt)YoungBags)/1024);
		printf("CollectBags:E: Allocbags = %p, StopBags = %p, EndBags = %p, Alloc pool area = %uk\n",
			   AllocBags, StopBags, EndBags, ((UInt)StopBags - (UInt)AllocBags)/1024);
		countHistOn = 1;
	}
	else {
		countHistOn = 0;
	}
	
	if (countHistOn > 0)   {
		Int4 ii;
		for (ii = 0; ii < SIZE_HIST; ii++) {
			BagSizeCount[ii].size = BagSizeCount[ii].count = BagSizeCount[ii].nlive = 0;
			BagSizeCount[ii].ndead = BagSizeCount[ii].nremnant = BagSizeCount[ii].nhalfdead = 0;
		}
		// Use InfoBags to count types used -- clear at start of each GC
		for (ii = 0; ii < NTYPES; ii++) {
			InfoBags[ii].nrAll = InfoBags[ii].nrLive = InfoBags[ii].nrDead = InfoBags[ii].nrRemnant = 0;
			InfoBags[ii].nrHalfdead = InfoBags[ii].sizeLive = InfoBags[ii].sizeAll = 0;
		}
	}
    
    /* call the before function (if any)                                   */
    if ( BeforeCollectFuncBags != 0 )
        (*BeforeCollectFuncBags)();

#ifdef GASMAN_ALWAYS_COLLECT_FULL
    FullBags = 1;
#else
    /* copy 'full' into a global variable, to avoid warning from GNU C     */
    FullBags = full;
#endif

    /* do we want to make a full garbage collection?                       */
again:
    if ( FullBags ) {

        /* then every bag is considered to be a young bag                  */
        YoungBags = OldBags;
        NrLiveBags = 0;
        SizeLiveBags = 0;

        /* empty the list of changed old bags                              */
		int nChangedBags = 0;
		while ( ChangedBags != 0 ) {
			first = ChangedBags;
			ChangedBags = GET_LINK_BAG(first);
			SET_LINK_BAG(first, first);
			nChangedBags++;
		}

        /* Also time to change the tag for dead children of weak
           pointer objects. After this collection, there can be no more 
           weak pointer objects pointing to anything with OldWeakDeadBagMarker
           in it */
        {
          BagPtr_t * t;
          t = OldWeakDeadBagMarker;
          OldWeakDeadBagMarker = NewWeakDeadBagMarker;
          NewWeakDeadBagMarker = t;
        }
    }

    /* information at the beginning of garbage collections                 */
    if ( MsgsFuncBags )
        (*MsgsFuncBags)( FullBags, 0, 0 );

    /* * * * * * * * * * * * * * *  mark phase * * * * * * * * * * * * * * */

    /* prepare the list of marked bags for the future                      */
    MarkedBags = 0;
	nrMarkedBags = 0;
    /* mark from the static area                                           */
    for ( i = 0; i < GlobalBags.nr; i++ ) 
		MARK_BAG( *GlobalBags.addr[i] );

	if (SyMemMgrTrace > 0)
		printf("CollectBags: Building Marked bags list: MARK_BAG marked %d global bags\n", nrMarkedBags);
    
    /* mark from the stack                                                 */
    if ( StackFuncBags ) {
        (*StackFuncBags)();
    }
    else {
        setjmp( RegsBags );
#ifdef  SPARC
#if SPARC
        SparcStackFuncBags();
#endif
#endif
        GenStackFuncBags();
    }

	if (SyMemMgrTrace > 0) {
		printf("Collectbags: Marked bags: After stack search, MARK_BAG total = %d bags\n", nrMarkedBags);
		printf("Collectbags: Check Changed bags, mark subbags of Changed OLD Bags\n");
	}
	
    /* mark the subbags of the changed old bags (bags in area *before* YoungBags)  */
    while ( ChangedBags != 0 ) {
        first = ChangedBags;
        ChangedBags = GET_LINK_BAG(first);
        SET_LINK_BAG(first, first);
        if ( PTR_BAG(first) <= YoungBags )
            (*TabMarkFuncBags[TNUM_BAG(first)])( first );
        else
            MARK_BAG(first);
    }
	/* we've now found [or should have] all live bags... */

	if (SyMemMgrTrace > 0)
		printf("Collectbags: Have list of all live bags, found = %d\n", nrMarkedBags);
	
#ifdef  DEBUG_DEADSONS_BAGS
	CheckOldNewBagRefs();
#endif

    /* tag all marked bags and mark their subbags                          */
	if (SyMemMgrTrace > 0)
		printf("Collectbags: Start tagging the live bags...\n");

    nrLiveBags = 0;
    sizeLiveBags = 0;

    while ( MarkedBags != 0 ) {
        first = MarkedBags;
        MarkedBags = GET_LINK_BAG(first);
		//  BagInfoPrint(first);
        SET_LINK_BAG(first, MARKED_ALIVE(first));
        (*TabMarkFuncBags[TNUM_BAG(first)])( first );
        nrLiveBags++;
        sizeLiveBags += GET_SIZE_BAG(first) + HEADER_SIZE * sizeof(BagPtr_t *);
    }
	
	if (SyMemMgrTrace > 0) {
		printf("CollectBags: After searching subbags, all live bags now = %d\n", nrMarkedBags);
		printf("CollectBags: marked alive = %d, size of alive bags = %d\n", nrLiveBags, sizeLiveBags);
	}

    /* information after the mark phase                                    */
    NrLiveBags += nrLiveBags;
    if ( MsgsFuncBags )
        (*MsgsFuncBags)( FullBags, 1, nrLiveBags );
    SizeLiveBags += sizeLiveBags;
    if ( MsgsFuncBags )
        (*MsgsFuncBags)( FullBags, 2, sizeLiveBags/1024 );

    /* * * * * * * * * * * * * * * sweep phase * * * * * * * * * * * * * * */

	//  TabFreeFuncBags are never set - eliminated code block

    /* sweep through the young generation -- on full GC all bags are treated as "Young" */
    nrDeadBags = 0;
    nrHalfDeadBags = 0;
    sizeDeadBags = 0;
    dst = YoungBags;
    src = YoungBags;

	UInt nLive, nDead, nHalfDead, nRemnant, szLive, szDead, szHalfDead, szRemnant;
	nLive = nDead = nHalfDead = nRemnant = szLive = szDead = szHalfDead = szRemnant = 0;

	if (SyMemMgrTrace > 0) {
		printf("CollectBags: YoungBags = %p, AllocBags = %p, # Live bags = %d\n",
			   YoungBags, AllocBags, NrLiveBags);
		printf("CollectBags: MptrBags = %p, OldBags = %p, size Mptr area = %dk, # mptrs = %d\n",
			   MptrBags, OldBags, ((UInt)OldBags - (UInt)MptrBags)/1024,
			   ((UInt)OldBags - (UInt)MptrBags)/sizeof(BagPtr_t));
	}

	/* Walk every bag that is either: alive, dead, half-dead, or resize free */
	UInt nbagsCheck = 0;
    while ( src < AllocBags ) {
		nbagsCheck++;
		
        /* leftover of a resize of <n> bytes                               */
        if ( GET_TYPE_PTR(src) == T_RESIZE_FREE ) {
            last = src;  type = 'r';

            /* advance src                                                 */
			if (TEST_FLAG_PTR(src, BF_COPY) ) {
				src++;
				nRemnant++; szRemnant += sizeof(BagPtr_t *);
				if (countHistOn)
					IncrementBagHistogram(1, INCREMENT_REMNANT, src);
			}
			else {
				src += 1 + WORDS_BAG( GET_SIZE_PTR(src) );
				nRemnant++; szRemnant += GET_SIZE_PTR(src) + sizeof(BagPtr_t *);
				if (countHistOn) {
					isz = 1 + WORDS_BAG( GET_SIZE_PTR(src) );
					IncrementBagHistogram(isz, INCREMENT_REMNANT, src);
				}
			}
        }

        /* dead bag                                                        */
        else if ( ((UInt)GET_LINK_PTR(src)) % sizeof(BagPtr_t) == 0 ) {
#ifdef DEBUG_MASTERPOINTERS
			static char dbgbuff[1000];
			BagPtr_t *q = (BagPtr_t *)GET_LINK_PTR(src);
			BagPtr_t *r = UNMARKED_DEAD(q);
			BagPtr_t *s = PTR_BAG(r);
			sprintf(dbgbuff, "CollectBags: src = %p, link ptr = %p, unmarked dead = %p, ptr Bag = %p, Data Ptr = %p\n", 
				   src, q, r, s, DATA_PTR(src));
			sprintf((dbgbuff + strlen(dbgbuff)), "CollectBags: # dead bags = %d, size dead bags = %d\n", 
				   nrDeadBags, sizeDeadBags);
            if  ( PTR_BAG( UNMARKED_DEAD(GET_LINK_PTR(src)) ) != DATA_PTR(src) )   {
				printf("%s\n", dbgbuff);
                (*AbortFuncBags)("incorrectly marked bag -- C");
			}
#endif
            last = src;  type = 'd';

            /* update count                                                */
            nrDeadBags += 1;
			sizeDeadBags += GET_SIZE_PTR(src);
			nDead++; szDead += GET_SIZE_PTR(src) + HEADER_SIZE * sizeof(BagPtr_t *);

			if (countHistOn) {
				isz = HEADER_SIZE + WORDS_BAG( GET_SIZE_PTR(src) );
				IncrementBagHistogram(isz, INCREMENT_DEAD, src);
			}

#ifdef  COUNT_BAGS
            /* update the statistics                                       */
            InfoBags[GET_TYPE_PTR(src)].nrLive -= 1;
            InfoBags[GET_TYPE_PTR(src)].sizeLive -= GET_SIZE_PTR(src);
#endif

            /* put the bag on [the head of] the free list */
			BagPtr_t nfhead = GET_LINK_PTR(src);
			if (nfhead >= MptrBags && nfhead < OldBags) {
				// link is valid
				// *(BagPtr_t*)(GET_LINK_PTR(src)) = FreeMptrBags;
				*nfhead = FreeMptrBags;
				FreeMptrBags = nfhead;			// (BagPtr_t)GET_LINK_PTR(src);
			}
			else {
				char * msg = GuMakeMessage("CollectBags: Bad link for bag going on Free chain, = %p\n", nfhead);
				SyAbortBags(msg);
			}

            /* advance src                                                 */
            src += HEADER_SIZE + WORDS_BAG( GET_SIZE_PTR(src) ) ;

        }

        /* half-dead bag                                                   */
        else if ( ((UInt)(GET_LINK_PTR(src))) % sizeof(BagPtr_t) == 2 ) {
#ifdef DEBUG_MASTERPOINTERS
            if  ( PTR_BAG( UNMARKED_HALFDEAD(GET_LINK_PTR(src)) ) != DATA_PTR(src) )  {
                (*AbortFuncBags)("incorrectly marked bag -- D");
			}
#endif
            last = src;  type = 'h';

            /* update count                                                */
            nrDeadBags += 1;
			sizeDeadBags += GET_SIZE_PTR(src);
			nHalfDead++; szHalfDead += GET_SIZE_PTR(src) + HEADER_SIZE * sizeof(BagPtr_t *);

			if (countHistOn) {
				isz = HEADER_SIZE + WORDS_BAG( GET_SIZE_PTR(src) );
				IncrementBagHistogram(isz, INCREMENT_HALFDEAD, src);
			}

#ifdef  COUNT_BAGS
            /* update the statistics                                       */
            InfoBags[GET_TYPE_PTR(src)].nrLive -= 1;
            InfoBags[GET_TYPE_PTR(src)].sizeLive -= GET_SIZE_PTR(src);
#endif

            /* don't free the identifier                                   */
            if (((UInt)UNMARKED_HALFDEAD(GET_LINK_PTR(src))) % 4 != 0)
				(*AbortFuncBags)("align error in halfdead bag");
                                              
           *(BagPtr_t**)(UNMARKED_HALFDEAD(GET_LINK_PTR(src))) = NewWeakDeadBagMarker;
           nrHalfDeadBags ++;

            /* advance src                                                 */
            src += HEADER_SIZE + WORDS_BAG( GET_SIZE_PTR(src) ) ;
        }

        /* live bag                                                        */
        else if ( ((UInt)(GET_LINK_PTR(src))) % sizeof(BagPtr_t) == 1 )  {
#ifdef DEBUG_MASTERPOINTERS
            if  ( PTR_BAG( UNMARKED_ALIVE(GET_LINK_PTR(src)) ) != DATA_PTR(src) )  {
                (*AbortFuncBags)("incorrectly marked bag -- E");
			}
#endif
            last = src;  type = 'l';
			nLive++; szLive += GET_SIZE_PTR(src) + HEADER_SIZE * sizeof(BagPtr_t *);

			if (countHistOn) {
				isz = HEADER_SIZE + WORDS_BAG( GET_SIZE_PTR(src) );
				IncrementBagHistogram(isz, INCREMENT_LIVE, src);
			}

            /* update identifier, copy size-type and link field            */
            SET_PTR_BAG(( UNMARKED_ALIVE(GET_LINK_PTR(src))),(BagPtr_t*) DATA_PTR(dst));
            end = src + HEADER_SIZE + WORDS_BAG( GET_SIZE_PTR(src) ) ;

			COPY_HEADER(dst, src);
			SET_LINK_PTR(dst, UNMARKED_ALIVE(GET_LINK_PTR(src)));

			dst += HEADER_SIZE;
			src += HEADER_SIZE;
	    
            /* copy data area                                */
			if (TabSweepFuncBags[(UInt)(src[-HEADER_SIZE]) & 0xFFL] != 0)   {
				/* Call the installed sweeping function */
				(*(TabSweepFuncBags[(UInt)(src[-HEADER_SIZE]) & 0xFFL]))(src,dst,end-src);
				dst += end-src;
				src = end;
			}
            /* Otherwise do the default thing */
			else if ( dst != src ) {
                while ( src < end )
					*dst++ = *src++;
			}
			else {
                dst = end;
                src = end;
			}
        }

        /* oops                                                            */
        else {
            (*AbortFuncBags)("Panic: Gasman found a bogus header");
        }
    }

	if (SyMemMgrTrace > 0)   {
		printf("CollectBags: Swept all bags, checked %u bags\n", nbagsCheck);
		printf("Collectbags: Clear free area = %s\n", (DirtyBags? "No" : "Yes"));
	}
	
    /* reset the pointer to the free storage                               */
    AllocBags = YoungBags = dst;
    /* clear the new free area                                             */
    if ( ! DirtyBags ) {
		src = EndBags;
		if (SyMemMgrTrace > 0)
			printf("Collectbags: Clear from = %p, to = %p; clear %u (%uk / %uMb) bytes\n",
				   dst, src, ((UInt)src - (UInt)dst), ((UInt)src - (UInt)dst) / 1024,
				   (((UInt)src - (UInt)dst) / (1024 * 1024)));
        while ( dst < src )
            *dst++ = 0;
    }

	// Since doing full GC have now processed all bags 
	if (SyMemMgrTrace > 0)  {
		printf("CollectBags: Processed all bags...found:\n#     Live Bags = %10u, size     Live Bags = %10u (%uk / %uMb)\n",
			   nLive, szLive, szLive / 1024, szLive / (1024 * 1024));
		printf("#     Dead Bags = %10u, size     Dead Bags = %10u (%uk / %uMb)\n",
			   nDead, szDead, szDead / 1024, szDead / (1024 * 1024));
		printf("#  Remnant Bags = %10u, size  Remnant Bags = %10u (%uk / %uMb)\n",
			   nRemnant, szRemnant, szRemnant / 1024, szRemnant / (1024 * 1024));
		printf("# HalfDead Bags = %10u, size HalfDead Bags = %10u (%uk / %uMb)\n",
			   nHalfDead, szHalfDead, szHalfDead / 1024, szHalfDead / (1024 * 1024));
	}

    /* information after the sweep phase                                   */
    NrDeadBags += nrDeadBags;
    NrHalfDeadBags += nrHalfDeadBags;
    if ( MsgsFuncBags )
        (*MsgsFuncBags)( FullBags, 3,
                         (FullBags ? NrDeadBags:nrDeadBags) );
    if ( FullBags )
        NrDeadBags = 0;
    SizeDeadBags += sizeDeadBags;
    if ( MsgsFuncBags )
        (*MsgsFuncBags)( FullBags, 4,
                         (FullBags ? SizeDeadBags:sizeDeadBags)/1024 );
    if ( FullBags )
        SizeDeadBags = 0;

    /* * * * * * * * * * * * * * * check phase * * * * * * * * * * * * * * */

	if (SyMemMgrTrace > 0)
		printf("Collectbags: Start checking phase ...\n");

    /* temporarily store in 'StopBags' where this allocation takes us      */
    StopBags = AllocBags + HEADER_SIZE + WORDS_BAG(size);

    /* if we only performed a partial garbage collection                   */
    if ( ! FullBags ) {
		printf("CollectBags: Always doing full GC ... shouldn't get here\n");
		
        /* maybe adjust the size of the allocation area                    */
        if ( ! CacheSizeBags ) {
            if ( nrLiveBags+nrDeadBags +nrHalfDeadBags < 512
		 
		 /* The test below should stop AllocSizeBags growing uncontrollably when
		    all bags are big */
		 && StopBags > OldBags + 4*1024*WORDS_BAG(AllocSizeBags))
                AllocSizeBags += NUM_TO_UINT(256);
            else if ( 4096 < nrLiveBags+nrDeadBags+nrHalfDeadBags
                   && 256 < AllocSizeBags )
                AllocSizeBags -= 256;
        }
        else {
            if ( nrLiveBags+nrDeadBags < 512 )
                AllocSizeBags += CacheSizeBags/1024;
            else if ( 4096 < nrLiveBags+nrDeadBags+nrHalfDeadBags
                   && CacheSizeBags < AllocSizeBags )
                AllocSizeBags -= CacheSizeBags/1024;
        }

        /* if we dont get enough free storage or masterpointers do full gc */
        if ( EndBags < StopBags + WORDS_BAG(1024*AllocSizeBags)
          || (OldBags-MptrBags)
             <
	     /*	     nrLiveBags+nrDeadBags+nrHalfDeadBags+ 4096 */
	     /*      If this test triggered, but the one below didn't
		     then a full collection would ensue which wouldn't
		     do anything useful. Possibly a version of the
		     above test should be moved into the full collection also
		     but I wasn't sure it always made sense         SL */

	     /* change the test to avoid subtracting unsigned integers */
	     WORDS_BAG(AllocSizeBags*1024)/7 +(NrLiveBags + NrHalfDeadBags) 
	     ) {
            done = 0;
        }
        else {
            done = 1;
        }

    }

    /* if we already performed a full garbage collection                   */
    else {

      /* Clean up old half-dead bags                                       */
		for (p = MptrBags; p < OldBags; p+= SIZE_MPTR_BAGS)   {
			if ((BagPtr_t *)*p == OldWeakDeadBagMarker)   {
				*p = (BagPtr_t)FreeMptrBags;
				FreeMptrBags = (BagPtr_t)p;
				NrHalfDeadBags --;
			}
		}

		if (SyMemMgrTrace > 0)   {
			printf("CollectBags: After looping Master pointers...\n");
			printf("CollectBags: MptrBags = %p, OldBags = %p, size Mptr area = %uk (%uMb), # mptrs = %d\n",
				   MptrBags, OldBags, ((UInt)OldBags - (UInt)MptrBags)/1024,
				   (((UInt)OldBags - (UInt)MptrBags) / (1024 * 1024)),
				   ((UInt)OldBags - (UInt)MptrBags)/sizeof(BagPtr_t));
			printf("CollectBags: Oldbags = %p, YoungBags = %p, Old Bags area = %uk (%uMb)\n",
				   OldBags, YoungBags, ((UInt)YoungBags - (UInt)OldBags)/1024,
				   (((UInt)YoungBags - (UInt)OldBags) / (1024 * 1024)));
			printf("CollectBags: Youngbags = %p, AllocBags = %p, Young Bags area = %uk (%uMb)\n",
				   YoungBags, AllocBags, ((UInt)AllocBags - (UInt)YoungBags)/1024,
				   (((UInt)AllocBags - (UInt)YoungBags) / (1024 * 1024)));
			printf("CollectBags: Allocbags = %p, StopBags = %p, EndBags = %p, Alloc pool area = %uk (%uMb)\n",
				   AllocBags, StopBags, EndBags, ((UInt)EndBags - (UInt)AllocBags)/1024,
				   (((UInt)EndBags - (UInt)AllocBags) / (1024 * 1024)));
			WalkBagPointers();
		}
	
        /* get the storage we absolutly need                               */
		if (SyMemMgrTrace > 0)
			printf("CollectBags: End < Stop: %s, EndBags = %p, StopBags = %p\n",
				   ((EndBags < StopBags)? "Yes" : "No"), EndBags, StopBags);

		/*
		 * Check amount of memory between StopBags and EndBags (alloc pool)
		 * If this drops below GASMAN_FREE_RATIO of available space (EndBags -
		 * OldBags) then declare we're out of memory and exit. 
		 */
		if ((EndBags - StopBags) < ((EndBags - OldBags) * GASMAN_FREE_RATIO)) {
			printf("GAP - CollectBags: After full GC and free pool to allocate bags is too small.\n");
			printf("Pool size = %uk, Used = %uk, Free = %uk\n", (EndBags - OldBags) / 1024,
				   (StopBags - OldBags) / 1024, (EndBags - StopBags) / 1024);
			printf("Not enough memory to continue...exiting\n");
			exit(1);
		}			

        /* now we are done                                                 */
        done = 1;

    }

    /* information after the check phase                                   */
    if ( MsgsFuncBags )
        (*MsgsFuncBags)( FullBags, 5,
                         ((char*)EndBags-(char*)StopBags)/1024 );
    if ( MsgsFuncBags )
        (*MsgsFuncBags)( FullBags, 6,
                         ((char*)EndBags-(char*)MptrBags)/1024 );

    /* reset the stop pointer                                              */
	StopBags = EndBags;

    /* if we are not done, then true again                                 */
    if ( ! done ) {
        FullBags = 1;
        goto again;
    }

    /* call the after function (if any)                                    */
    if ( AfterCollectFuncBags != 0 )
        (*AfterCollectFuncBags)();


#ifdef DEBUG_MASTERPOINTERS
    CheckMasterPointers();
#endif
    
	if (countHistOn)
		DumpBagsHistogram();

    /* return success                                                      */
    return 1;
}


/****************************************************************************
**
*F  CheckMasterPointers() . . . . do consistency checks on the masterpointers
**
*/

void CheckMasterPointers(void)
{
  BagPtr_t *ptr;
  for (ptr = MptrBags; ptr < OldBags; ptr++)
    {
      if (*ptr != (BagPtr_t)0 &&             /* bottom of free chain */
          *ptr != (BagPtr_t)NewWeakDeadBagMarker &&
          *ptr != (BagPtr_t)OldWeakDeadBagMarker &&
          (((BagPtr_t *)*ptr < MptrBags &&
            (BagPtr_t *)*ptr > AllocBags) ||
           (UInt)(*ptr) % sizeof(BagPtr_t) != 0))
        (*AbortFuncBags)("Bad master pointer detected in check");
    }
}


/****************************************************************************
**
*F  SwapMasterPoint( <bag1>, <bag2> ) . . . swap pointer of <bag1> and <bag2>
*/
void SwapMasterPoint (
    BagPtr_t                 bag1,
    BagPtr_t                 bag2 )
{
    BagPtr_t *               ptr1;
    BagPtr_t *               ptr2;

    if ( bag1 == bag2 )
        return;

    /* get the pointers                                                    */
    ptr1 = PTR_BAG(bag1);
    ptr2 = PTR_BAG(bag2);

    /* check and update the link field and changed bags                    */
    if ( GET_LINK_BAG(bag1) == bag1 && GET_LINK_BAG(bag2) == bag2 ) {
        SET_LINK_BAG(bag1, bag2);
        SET_LINK_BAG(bag2, bag1);
    }
    else if ( GET_LINK_BAG(bag1) == bag1 ) {
        SET_LINK_BAG(bag1, ChangedBags);
        ChangedBags = bag1; 
    }
    else if ( GET_LINK_BAG(bag2) == bag2 ) {
        SET_LINK_BAG(bag2, ChangedBags);
        ChangedBags = bag2; 
    }

    /* swap them                                                           */
    SET_PTR_BAG(bag1, ptr2);
    SET_PTR_BAG(bag2, ptr1);
}



/****************************************************************************
**

*F  BID(<bag>)  . . . . . . . . . . . .  bag identifier (as unsigned integer)
*F  IS_BAG(<bid>) . . . . . .  test whether a bag identifier identifies a bag
*F  BAG(<bid>)  . . . . . . . . . . . . . . . . . . bag (from bag identifier)
*F  GET_TYPE_BAG(<bag>) . . . . . . . . . . . . . . . . . . . . . . type of a bag
*F  GET_SIZE_BAG(<bag>) . . . . . . . . . . . . . . . . . . . . . . size of a bag
*F  PTR_BAG(<bag>)  . . . . . . . . . . . . . . . . . . . .  pointer to a bag
*F  ELM_BAG(<bag>,<i>)  . . . . . . . . . . . . . . . <i>-th element of a bag
*F  SET_ELM_BAG(<bag>,<i>,<elm>)  . . . . . . . . set <i>-th element of a bag
**
**  'BID', 'IS_BAG', 'BAG', 'GET_TYPE_BAG', 'TYPENAM', 'GET_SIZE_BAG', 'PTR_BAG', 'ELM_BAG',  and
**  'SET_ELM_BAG' are functions to support  debugging.  They are not intended
**  to be used  in an application  using {\Gasman}.  Note  that the functions
**  'GET_TYPE_BAG', 'GET_SIZE_BAG', and 'PTR_BAG' shadow the macros of the same  name  which  are 
**  usually not available in a debugger.
*/
#ifdef  DEBUG_FUNCTIONS_BAGS

#undef  GET_TYPE_BAG
#undef  GET_SIZE_BAG
#undef  PTR_BAG
/*
UInt BID ( BagPtr_t bag ) {
    return (UInt) bag;
}
BagPtr_t BAG ( UInt bid ) {
    if ( IS_BAG(bid) )  return (BagPtr_t) bid;
    else return (BagPtr_t) 0;
}
*/

UInt IS_BAG ( BagPtr_t bag ) {
    return (((UInt)MptrBags <= (UInt) bag)
         && ((UInt)bag < (UInt)OldBags)
         && (((UInt)bag & (sizeof(BagPtr_t)-1)) == 0)
	 && (OldBags < PTR_BAG(bag) && PTR_BAG(bag) <= AllocBags));
}

ObjType GET_TYPE_BAG ( BagPtr_t bag ) {
    if(IS_INTOBJ(bag)) return T_INT;
    else if(! IS_BAG(bag)) return T_ILLEGAL;
    else return (ObjType)GET_TYPE_PTR(_PTR_BAG(bag) - HEADER_SIZE);
}

const Char * TYPENAM ( BagPtr_t bag ) {
    return InfoBags[ TNUM_BAG(bag) ].name;
}

//  UInt GET_SIZE_BAG ( BagPtr_t bag ) {
//      return GET_SIZE_PTR(_PTR_BAG(bag) - HEADER_SIZE);
//  }

//  BagPtr_t * PTR ( BagPtr_t bag ) {
//      return _PTR_BAG(bag);
//  }

BagPtr_t ELM_BAG ( BagPtr_t bag, UInt i ) {
    return _PTR_BAG(bag)[i];
}

BagPtr_t SET_ELM_BAG ( BagPtr_t bag, UInt i, BagPtr_t elm ) {
    _PTR_BAG(bag)[i] = elm;
    return elm;
}

#endif


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
**  For example, 'GET_SIZE_BAG[T_FFE].handles' is 'SIZE_HD' and 'SizeType[T_FFE].data'
**  is 'sizeof(short)', that means the finite field elements have one handle,
**  referring to the finite field, and one short value, which is the value of
**  this finite field element, i.e., the discrete logarithm.
**
**  Either value may also be negative, which  means  that  the  size  of  the
**  corresponding area has a variable size.
**
**  For example, 'SizeType[T_VAR].handles' is 'SIZE_HD' and 'SizeType[T_VAR].data'
**  is -1, which means that variable bags have one handle,  for the  value of
**  the variable, and a variable sized data area, for the name of the identi-
**  fier.
**
**  For example, 'GET_SIZE_BAG[T_LIST].handles' is '-SIZE_HD' and 'SizeType[T_LIST].data'
**  is '0', which means that a list has a variable number of handles, for the
**  elements of the list, and no other data.
**
**  If both values are negative both areas are variable sized.  The ratio  of
**  the  sizes  is  fixed,  and  is  given  by the ratio of the two values in
**  'SizeType'.
**
**  I can not give you an example, because no such type is currently in use.
*/

// pb-comment: Ideally SizeType[] and NameType[] below would be static, but SizeType is referenced
//  in spiral_delay_ev.c...

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
*v  NameType  . . . . . . . . . . . . . . . . . . .  printable name of a type
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
**  'NrHandles' uses the information stored in 'Size'.
*/
Int            NrHandles (unsigned int type, UInt size)
{
    register Int       hs, is;

    if(type >= T_ILLEGAL) return 0;

    hs = SizeType[type].handles;
    if ( hs >= 0 )  return hs / SIZE_HD;

    is = SizeType[type].data;
    if ( is >= 0 )  return (size - is) / SIZE_HD;

    return ( hs * (Int)size / (hs + is) ) / SIZE_HD;
}


/********************************************************************************/
/* This next block of functions are the old static inline functions previously  */
/* defined operating on bag handles to extract various items (handle to data    */
/* pointer, size, type, etc.)												    */
/********************************************************************************/

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

// Set the type for the bag
void SET_TYPE_BAG(BagPtr_t bag, UInt val)
{
	BagPtr_t *p = (*(BagPtr_t **)(bag));						/* PTR_BAG(bag); */
	BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
	UInt flagType = (ptr->bagFlagsType & BF_ALL_FLAGS) + val;
	ptr->bagFlagsType = flagType;
	return;
}

// Clear the flags associated to a bag 
void BLANK_FLAGS_BAG(BagPtr_t bag)
{
	BagPtr_t *p = (*(BagPtr_t **)(bag));						/* PTR_BAG(bag); */
	BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
	UInt flagType = ptr->bagFlagsType & TYPE_BIT_MASK;
	ptr->bagFlagsType = flagType;
	return;
}

// Return the type associated to a bag
UInt TNUM_BAG(BagPtr_t bag)
{
	if (bag==0)
		return T_ILLEGAL;
	else {
		BagPtr_t *p = (*(BagPtr_t **)(bag));					/* PTR_BAG(bag); */
		BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
		return ptr->bagFlagsType & TYPE_BIT_MASK;
	}
}

// Get the flags associated to a bag
UInt GET_FLAGS_BAG(BagPtr_t bag)
{
	BagPtr_t *p = (*(BagPtr_t **)(bag));						/* PTR_BAG(bag); */
	BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
	return ptr->bagFlagsType & BF_ALL_FLAGS;
}

// Set the flags associated to a bag
void SET_FLAG_BAG(BagPtr_t bag, UInt val)
{
	BagPtr_t *p = (*(BagPtr_t **)(bag));						/* PTR_BAG(bag); */
	BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
	UInt flagType = ptr->bagFlagsType | val;
	ptr->bagFlagsType = flagType;
	return;
}

// Clear a flag associated to the bag
void CLEAR_FLAG_BAG(BagPtr_t bag, UInt val)
{
	BagPtr_t *p = (*(BagPtr_t **)(bag));						/* PTR_BAG(bag); */
	BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
	UInt flagType = ptr->bagFlagsType & ~(val);
	ptr->bagFlagsType = flagType;
	return;
}

// Test if a flag is associated to a bag
UInt GET_FLAG_BAG(BagPtr_t bag, UInt val)
{
	BagPtr_t *p = (*(BagPtr_t **)(bag));						/* PTR_BAG(bag); */
	BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
	return ptr->bagFlagsType & val;
}

// Get the size of [the data in] a bag
UInt GET_SIZE_BAG(BagPtr_t bag)
{
	BagPtr_t *p = (*(BagPtr_t **)(bag));						/* PTR_BAG(bag); */
	BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
	return ptr->bagSize;
}

// Set the size of a bag [size of data in the bag]
void SET_SIZE_BAG(BagPtr_t bag, UInt val)
{
	BagPtr_t *p = (*(BagPtr_t **)(bag));						/* PTR_BAG(bag); */
	BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
	ptr->bagSize = val;
	return;
}

// Get the link pointer associated to a bag
BagPtr_t GET_LINK_BAG(BagPtr_t bag)
{
	BagPtr_t *p = (*(BagPtr_t **)(bag));						/* PTR_BAG(bag); */
	BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
#ifdef DEBUG_POINTERS
	// test link ptr
	if (ptr->bagLinkPtr != 0 && (ptr->bagLinkPtr < MptrBags || ptr->bagLinkPtr >= StopBags) ) {
		GuFatalMsgExit(EXIT_MEM, "GET_LINK_BAG: Bag has invalid link ptr, = %p (%p)\n",
					   bag, ptr->bagLinkPtr);
	}
#endif
	return ptr->bagLinkPtr;
}

// Set the link pointer to be associated to a bag
void SET_LINK_BAG(BagPtr_t bag, BagPtr_t val)
{
#ifdef DEBUG_POINTERS
	if (val != 0 && (val < MptrBags || val >= StopBags) ) {
		GuFatalMsgExit(EXIT_MEM, "SET_LINK_BAG: Attempt to set invalid link ptr, = %p (%p)\n",
								   bag, val);
	}
#endif
	BagPtr_t *p = (*(BagPtr_t **)(bag));						/* PTR_BAG(bag); */
	BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
	ptr->bagLinkPtr = val;
	return;
}

// Get the copy pointer associated to a bag
BagPtr_t GET_COPY_BAG(BagPtr_t bag)
{
	BagPtr_t *p = (*(BagPtr_t **)(bag));						/* PTR_BAG(bag); */
	BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
	return ptr->bagCopyPtr;
}

// Set the copy pointer associated to a bag
void SET_COPY_BAG(BagPtr_t bag, BagPtr_t val)
{
	BagPtr_t *p = (*(BagPtr_t **)(bag));						/* PTR_BAG(bag); */
	BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);
	ptr->bagCopyPtr = val;
	return;
}

// Update ChangedBags
void CHANGED_BAG(BagPtr_t bag)
{
	BagPtr_t *p = (*(BagPtr_t **)(bag));						/* PTR_BAG(bag); */
	BagStruct_t *ptr = (BagStruct_t *)(p - HEADER_SIZE);

	if (p <= YoungBags && ptr->bagLinkPtr == bag) {
		ptr->bagLinkPtr = ChangedBags;
		ChangedBags = bag;
	}
	return;
	
  /* if (   PTR_BAG(bag) <= YoungBags && PTR_BAG(bag)[-1] == (bag) ){ */
  /*   PTR_BAG(bag)[-1] = ChangedBags; */
  /*   ChangedBags = (bag); */
  /* } */
}

#ifdef DEBUG_POINTERS
// functions to replace some macros manipulating link pointers for debug...

BagPtr_t GET_LINK_PTR(BagPtr_t ptr)
{
	// ptr points to the start of the bag [header], pull & test the link field
	BagStruct_t *p = (BagStruct_t *)ptr;
	if (p->bagLinkPtr != 0 && (p->bagLinkPtr < MptrBags || p->bagLinkPtr >= StopBags) ) {
		char * msg = GuMakeMessage("GET_LINK_PTR: Bag has invalid link ptr, = %p (%p)\n",
								   ptr, p->bagLinkPtr);
		SyAbortBags(msg);
	}
	return p->bagLinkPtr;
}

void SET_LINK_PTR(BagPtr_t ptr, BagPtr_t val)
{
	// ptr points to the start of the bag [header], validate the link before saving
	if (val != 0 && (val < MptrBags || val >= StopBags) ) {
		char * msg = GuMakeMessage("SET_LINK_PTR: Attempt to set invalid link ptr, = %p (%p)\n",
								   ptr, val);
		SyAbortBags(msg);
	}
	BagStruct_t *p = (BagStruct_t *)ptr;
	p->bagLinkPtr = val;
	return;
}
#endif


int DoFullCopy;

static void _RecursiveClearFlagMutable(Obj hd, int flag, int check_has_flag)
{
    UInt       n;              /* number of handles of <hd>       */
    UInt       i;              /* loop variable                   */
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
    UInt       n;              /* number of handles of <hd>       */
    UInt       i;              /* loop variable                   */
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
    UInt       n;              /* number of handles of <hd>       */
    UInt       i;              /* loop variable                   */
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
    UInt       i;              /* loop variable                   */
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

