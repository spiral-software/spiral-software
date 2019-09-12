
/****************************************************************************
**
*Y  Copyright (C) 2018-2019, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2019) by the GAP Group (www.gap-system.org).
**
*/

#ifndef _SYSTEM4_H
#define _SYSTEM4_H

#include        <setjmp.h>              /* jmp_buf, setjmp, longjmp        */
#include		"system_types.h"


#ifdef  SYS_HAS_STACK_ALIGN
#define SYS_STACK_ALIGN         SYS_HAS_STACK_ALIGN
#endif

#ifndef SYS_ARCH
# define SYS_ARCH = "unknown";
#endif

#ifndef SY_STOR_MIN
#  define SY_STOR_MIN   0 
#endif

#ifndef SYS_HAS_STACK_ALIGN
#define SYS_STACK_ALIGN         sizeof(UInt *)
#endif

#ifdef SYS_HAS_SIGNALS
# define HAVE_SIGNAL            1
#else
# define HAVE_SIGNAL            0
#endif

#define HAVE_ACCESS		0
#define HAVE_STAT		0
#define HAVE_UNLINK             0
#define HAVE_MKDIR              0
#define HAVE_GETRUSAGE		0
#define HAVE_DOTGAPRC		0
#define HAVE_GHAPRC             0

#ifdef SYS_IS_BSD
# undef  HAVE_ACCESS
# define HAVE_ACCESS		1
# undef  HAVE_STAT
# define HAVE_STAT              1
# undef  HAVE_UNLINK
# define HAVE_UNLINK            1
# undef  HAVE_MKDIR
# define HAVE_MKDIR             1
# undef  HAVE_GETRUSAGE
# define HAVE_GETRUSAGE		1
# undef  HAVE_DOTGAPRC
# define HAVE_DOTGAPRC          1
#endif

#ifdef SYS_IS_USG
# undef  HAVE_ACCESS
# define HAVE_ACCESS		1
# undef  HAVE_STAT
# define HAVE_STAT              1
# undef  HAVE_UNLINK
# define HAVE_UNLINK            1
# undef  HAVE_MKDIR
# define HAVE_MKDIR             1
# undef  HAVE_DOTGAPRC
# define HAVE_DOTGAPRC          1
#endif

#ifdef SYS_HAS_NO_GETRUSAGE
# undef  HAVE_GETRUSAGE
# define HAVE_GETRUSAGE		0
#endif


/****************************************************************************
**

*V  SYS_ANSI  . . . . . . . . . . . . . . . . . . . . . . . . . . . .  ANSI C
*/
#ifdef SYS_HAS_ANSI
# define SYS_ANSI       SYS_HAS_ANSI
#else
# ifdef __STDC__
#  define SYS_ANSI      1
# else
#  define SYS_ANSI      0
# endif
#endif


/****************************************************************************
**
*V  SYS_BSD . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . BSD
*/
#ifdef SYS_IS_BSD
# define SYS_BSD        1
#else
# define SYS_BSD        0
#endif


/****************************************************************************
**
*V  SYS_USG . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . USG
*/
#ifdef SYS_IS_USG
# define SYS_USG        1
#else
# define SYS_USG        0
#endif

#define FPUTS_TO_STDERR(str) fputs (str, stderr)

/****************************************************************************
**
*T  Bag . . . . . . . . . . . . . . . . . . . type of the identifier of a bag
*/
#ifdef  DGC_DEBUG
typedef UInt const * const * Bag;
#else
typedef UInt * *        Bag;
#endif

/****************************************************************************
**
*T  Obj . . . . . . . . . . . . . . . . . . . . . . . . . . . type of objects
**
**  'Obj' is the type of objects.
*/
#define Obj             Bag


/****************************************************************************
**

*F * * * * * * * * * * * command line settable options  * * * * * * * * * * *
*/

/****************************************************************************
**

*V  SyStackAlign  . . . . . . . . . . . . . . . . . .  alignment of the stack
**
**  'SyStackAlign' is  the  alignment  of items on the stack.   It  must be a
**  divisor of  'sizof(Bag)'.  The  addresses of all identifiers on the stack
**  must be  divisable by 'SyStackAlign'.  So if it  is 1, identifiers may be
**  anywhere on the stack, and if it is  'sizeof(Bag)',  identifiers may only
**  be  at addresses  divisible by  'sizeof(Bag)'.  This value is initialized
**  from a macro passed from the makefile, because it is machine dependent.
**
**  This value is passed to 'InitBags'.
*/
extern UInt SyStackAlign;


/****************************************************************************
**
*V  SyCacheSize . . . . . . . . . . . . . . . . . . . . . . size of the cache
**
**  'SyCacheSize' is the size of the data cache, in kilobytes
**
**  This is per  default 0, which means that  there is no usuable data cache.
**  It is usually changed with the '-c' option in the script that starts GAP.
**
**  This value is passed to 'InitBags'.
**
**  Put in this package because the command line processing takes place here.
*/
extern UInt SyCacheSize;


/****************************************************************************
**
*V  SyMsgsFlagBags  . . . . . . . . . . . . . . . . .  enable gasman messages
**
**  'SyMsgsFlagBags' determines whether garabage collections are reported  or
**  not.
**
**  Per default it is false, i.e. Gasman is silent about garbage collections.
**  It can be changed by using the  '-g'  option  on the  GAP  command  line.
**
**  This is used in the function 'SyMsgsBags' below.
**
**  Put in this package because the command line processing takes place here.
*/
extern UInt SyMsgsFlagBags;

/****************************************************************************
**
*V  SyStorMax . . . . . . . . . . . . . . . . . . . maximal size of workspace
**
**  'SyStorMax' is the maximal size of the workspace allocated by Gasman.
**    this is now in kilobytes.
**
**  This is per default 256 MByte,  which is often a  reasonable value.  It is
**  usually changed with the '-o' option in the script that starts GAP.
**
**  This is used in the function 'SyAllocBags'below.
**
**  Put in this package because the command line processing takes place here.
*/
extern Int SyStorMax;
extern Int SyStorOverrun;

/****************************************************************************
**
*V  SyStorKill . . . . . . . . . . . . . . . . . . maximal size of workspace
**
**  'SyStorKill' is really the maximal size of the workspace allocated by 
**  Gasman. GAP exists before trying to allocate more than this amount
**  of memory in kilobytes
**
**  This is per default disabled (i.e. = 0).
**  Can be changed with the '-K' option in the script that starts GAP.
**
**  This is used in the function 'SyAllocBags'below.
**
**  Put in this package because the command line processing takes place here.
*/
extern Int SyStorKill;

/****************************************************************************
**
*V  SyStorMin . . . . . . . . . . . . . .  default size for initial workspace
**
**  'SyStorMin' is the size of the initial workspace allocated by Gasman.
**  in kilobytes
**
**  This is per default  24 Megabyte,  which  is  often  a  reasonable  value.
**  It is usually changed with the '-m' option in the script that starts GAP.
**
**  This value is used in the function 'SyAllocBags' below.
**
**  Put in this package because the command line processing takes place here.
*/
extern Int SyStorMin;


/****************************************************************************
**

*F * * * * * * * * * * * * * * gasman interface * * * * * * * * * * * * * * *
*/


/****************************************************************************
**

*F  SyMsgsBags( <full>, <phase>, <nr> ) . . . . . . . display Gasman messages
**
**  'SyMsgsBags' is the function that is used by Gasman to  display  messages
**  during garbage collections.
*/
extern void SyMsgsBags (
            UInt                full,
            UInt                phase,
            Int                 nr );


/****************************************************************************
**
*F  SyAllocBags( <size>, <need> ) . . . allocate memory block of <size> bytes
**
**  'SyAllocBags' is called from Gasman to get new storage from the operating
**  system.  <size> is the needed amount in kilobytes (it is always a multiple of
**  512 KByte),  and <need> tells 'SyAllocBags' whether  Gasman  really needs
**  the storage or only wants it to have a reasonable amount of free storage.
**
**  Currently  Gasman  expects this function to return  immediately  adjacent
**  areas on subsequent calls.  So 'sbrk' will  work  on  most  systems,  but
**  'malloc' will not.
**
**  If <need> is 0, 'SyAllocBags' must return 0 if it cannot or does not want
**  to extend the workspace,  and a pointer to the allocated area to indicate
**  success.   If <need> is 1  and 'SyAllocBags' cannot extend the workspace,
**  'SyAllocBags' must abort,  because GAP assumes that  'NewBag'  will never
**  fail.
**
**  <size> may also be negative in which case 'SyAllocBags' should return the
**  storage to the operating system.  In this case  <need>  will always be 0.
**  'SyAllocBags' can either accept this reduction and  return 1  and  return
**  the storage to the operating system or refuse the reduction and return 0.
**
**  If the operating system does not support dynamic memory managment, simply
**  give 'SyAllocBags' a static buffer, from where it returns the blocks.
*/
extern UInt * * * SyAllocBags (
            Int                 size,
            UInt                need );


/****************************************************************************
**
*F  SyAbortBags(<msg>)  . . . . . . . . . . abort GAP in case of an emergency
**
**  'SyAbortBags' is the function called by Gasman in case of an emergency.
*/
extern void SyAbortBags (
            Char *              msg );


/****************************************************************************
**

*F * * * * * * * * * * * * * loading of modules * * * * * * * * * * * * * * *
*/

/****************************************************************************
**
*T  StructBagNames  . . . . . . . . . . . . . . . . . . . . . tnums and names
*/
typedef struct {
    Int                 tnum;
    const Char *    name;
} StructBagNames;


/****************************************************************************
**
*T  StructGVarFilt  . . . . . . . . . . . . . . . . . . . . . exported filter
*/
typedef struct {
    const Char *    name;
    const Char *    argument;
    Obj *               filter;
    Obj              (* handler)(/*arguments*/);
    const Char *    cookie;
} StructGVarFilt;


/****************************************************************************
**
*T  StructGVarAttr  . . . . . . . . . . . . . . . . . . .  exported attribute
*/
typedef struct {
    const Char *    name;
    const Char *    argument;
    Obj *               attribute;
    Obj              (* handler)(/*arguments*/);
    const Char *    cookie;
} StructGVarAttr;


/****************************************************************************
**
*T  StructGVarProp  . . . . . . . . . . . . . . . . . . . . exported property
*/
typedef struct {
    const Char *    name;
    const Char *    argument;
    Obj *               property;
    Obj              (* handler)(/*arguments*/);
    const Char *    cookie;
} StructGVarProp;


/****************************************************************************
**
*T  StructGVarOper  . . . . . . . . . . . . . . . . . . .  exported operation
*/
typedef struct {
    const Char *    name;
    Int                 nargs;
    const Char *    args;
    Obj *               operation;
    Obj              (* handler)(/*arguments*/);
    const Char *    cookie;
} StructGVarOper;


/****************************************************************************
**
*T  StructGVarFunc  . . . . . . . . . . . . . . . . . . . . exported function
*/
typedef struct {
    const Char *    name;
    Int                 nargs;
    const Char *    args;
    Obj              (* handler)(/*arguments*/);
    const Char *    cookie;
} StructGVarFunc;


/****************************************************************************
**

*F * * * * * * * * * * * * * initialize package * * * * * * * * * * * * * * *
*/

/****************************************************************************
**
*F  InitSystem4( <argc>, <argv> )  . . .  initialize system package from GAP4
**
**  'InitSystem' is called very early during the initialization from  'main'.
**  It is passed the command line array  <argc>, <argv>  to look for options.
**
**  For UNIX it initializes the default files 'stdin', 'stdout' and 'stderr',
**  installs the handler 'syAnsIntr' to answer the user interrupts '<ctr>-C',
**  scans the command line for options, tries to  find  'LIBNAME/init.g'  and
**  '$HOME/.gaprc' and copies the remaining arguments into 'SyInitfiles'.
*/
extern void InitSystem4 (
            Int                 argc,
            Char *              argv [] );


/****************************************************************************
**

*E  system.h  . . . . . . . . . . . . . . . . . . . . . . . . . . . ends here
*/

typedef void            (* TNumAbortFuncBags) (
                                Char *          msg );
                                
extern  TNumAbortFuncBags       AbortFuncBags;

#endif // _SYSTEM4_H
