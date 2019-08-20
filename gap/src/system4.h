
/****************************************************************************
**
*W  system.h                    GAP source                   Martin Schoenert
*W                                                         & Dave Bayer (MAC)
*W                                                  & Harald Boegeholz (OS/2)
*W                                                      & Frank Celler (MACH)
*W                                                         & Paul Doyle (VMS)
*W                                                  & Burkhard Hoefling (MAC)
*W                                                    & Steve Linton (MS/DOS)
**
**
*Y  Copyright (C) 2018-2019, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2019) by the GAP Group (www.gap-system.org).
**
**  The  file 'system.c'  declares  all operating system  dependent functions
**  except file/stream handling which is done in "sysfiles.h".
*/

#ifndef _SYSTEM4_H
#define _SYSTEM4_H

#include        <setjmp.h>              /* jmp_buf, setjmp, longjmp        */
#include		"system_types.h"

/****************************************************************************
**

*V  autoconf  . . . . . . . . . . . . . . . . . . . . . . . .  use "config.h"
*/
#ifdef CONFIG_H

#include "config.h"

/* define stack align for gasman (from "config.h")                         */
#define SYS_STACK_ALIGN		C_STACK_ALIGN

/* assume all prototypes are there                                         */
#define SYS_HAS_CALLOC_PROTO
#define SYS_HAS_EXEC_PROTO
#define SYS_HAS_IOCTL_PROTO
#define SYS_HAS_MALLOC_PROTO
#define SYS_HAS_MEMSET_PROTO
#define SYS_HAS_MISC_PROTO
#define SYS_HAS_READ_PROTO
#define SYS_HAS_SIGNAL_PROTO
#define SYS_HAS_STDIO_PROTO
#define SYS_HAS_STRING_PROTO
#define SYS_HAS_TIME_PROTO
#define SYS_HAS_WAIT_PROTO
#define SYS_HAS_WAIT_PROTO


/* some compiles define symbols beginning with an underscore               */
#if C_UNDERSCORE_SYMBOLS
# define SYS_INIT_DYNAMIC       "_Init__Dynamic"
#else
# define SYS_INIT_DYNAMIC       "Init__Dynamic"
#endif

/* "config.h" will redefine `vfork' to `fork' if necessary                 */
#define SYS_MY_FORK             vfork

#define SYS_HAS_SIG_T           RETSIGTYPE

/* prefer `vm_allocate' over `sbrk'                                        */
#if HAVE_VM_ALLOCATE
# undef  HAVE_SBRK
# define HAVE_SBRK              0
#endif

/* prefer "termio.h" over "sgtty.h"                                        */
#if HAVE_TERMIO_H
# undef  HAVE_SGTTY_H
# define HAVE_SGTTY_H           0
#endif

/* prefer `getrusage' over `times'                                         */
#if HAVE_GETRUSAGE
# undef  HAVE_TIMES
# define HAVE_TIMES             0
#endif

/* defualt HZ value                                                        */
/*  on IRIX we need this include to get the system value                   */

#if HAVE_SYS_SYSMACROS_H
#include <sys/sysmacros.h>
#endif

#ifndef  HZ
# define HZ                     50
#endif

/* prefer `waitpid' over `wait4'                                           */
#if HAVE_WAITPID
# undef  HAVE_WAIT4
# define HAVE_WAIT4             0
#endif

#endif


/****************************************************************************
**
*V  no autoconf . . . . . . . . . . . . . . . . . . . . do not use "config.h"
*/
#ifndef CONFIG_H

#ifdef  SYS_HAS_STACK_ALIGN
#define SYS_STACK_ALIGN         SYS_HAS_STACK_ALIGN
#endif

#ifndef SYS_ARCH
# define SYS_ARCH = "unknown";
#endif

#ifndef SY_STOR_MIN
# if SYS_MAC_MPW || SYS_TOS_GCC2
#  define SY_STOR_MIN   0
# else
#  define SY_STOR_MIN   0 
# endif
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

#ifdef SYS_IS_MACH
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

#ifdef SYS_IS_OS2_EMX
# undef  HAVE_ACCESS
# define HAVE_ACCESS		1
# undef  HAVE_STAT
# define HAVE_STAT              1
# undef  HAVE_UNLINK
# define HAVE_UNLINK            1
# undef  HAVE_MKDIR
# define HAVE_MKDIR             1
# undef  HAVE_GAPRC
# define HAVE_GAPRC             1
#endif

#ifdef SYS_HAS_NO_GETRUSAGE
# undef  HAVE_GETRUSAGE
# define HAVE_GETRUSAGE		0
#endif

#endif


/****************************************************************************
**
*V  Includes  . . . . . . . . . . . . . . . . . . . . .  include system files
*/
#ifdef CONFIG_H
#endif


/****************************************************************************
**

*V  Revision_system_h . . . . . . . . . . . . . . . . . . . . revision number
*/
#ifdef  INCLUDE_DECLARATION_PART
const char * Revision_system_h =
   "@(#)Id: system.h,v 4.53 2002/05/04 13:45:54 gap Exp";
#endif
extern const char * Revision_system_c;  /* gap.c uses this */
extern const char * Revision_system_h;


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
*V  SYS_MACH  . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  MACH
*/
#ifdef SYS_IS_MACH
# define SYS_MACH       1
#else
# define SYS_MACH       0
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


/****************************************************************************
**
*V  SYS_OS2_EMX . . . . . . . . . . . . . . . . . . . . . . OS2 using GCC/EMX
*/
#ifdef SYS_IS_OS2_EMX
# define SYS_OS2_EMX    1
#else
# define SYS_OS2_EMX    0
#endif


/****************************************************************************
**
*V  SYS_MSDOS_DJGPP . . . . . . . . . . . . . . . . . . . . . MSDOS using GCC
*/
#ifdef SYS_IS_MSDOS_DJGPP
# define SYS_MSDOS_DJGPP 1
#else
# define SYS_MSDOS_DJGPP 0
#endif


/****************************************************************************
**
*V  SYS_VMS . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . VMS
*/
#ifdef SYS_IS_VMS
# define SYS_VMS        1
#else
# define SYS_VMS        0
#endif


/****************************************************************************
**
*V  SYS_MAC_MPW . . . . . . . . . . . . . . . . . . . . . . . . MAC using MPW
*/
#ifdef SYS_IS_MAC_MPW
# define SYS_MAC_MPW    1
#else
# define SYS_MAC_MPW    0
#endif


/****************************************************************************
**
*V  SYS_MAC_MWC . . . . . . . . . . . . . . . . . . .  Mac using Metrowerks C
*/
#ifdef SYS_IS_MAC_MWC
# define SYS_MAC_MWC    1
#else
# define SYS_MAC_MWC    0
#endif

/****************************************************************************
**
*V  SYS_DARWIN . . . . . . . . . . . . . . .  DARWIN (BSD underlying MacOS X)
*/
#ifdef SYS_IS_DARWIN
# define SYS_DARWIN    1
#else
# define SYS_DARWIN    0
#endif


#if SYS_MAC_MWC  /* on the Mac, fputs does not work. Print error messages 
					using WriteToLog */
# define FPUTS_TO_STDERR(str) 	WriteToLog (str)
#else 
# define FPUTS_TO_STDERR(str) fputs (str, stderr)
#endif



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
