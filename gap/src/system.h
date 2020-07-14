/****************************************************************************
**
*A  system.h                    GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2020, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2020) by the GAP Group (www.gap-system.org).
**
**  This file declares operating system dependent functions.
**
*/

#ifndef _SYSTEM_H
#define _SYSTEM_H

#include		"system_types.h"

/****************************************************************************
**
*F  P((<args>)) . . . . . . . . . . . . enable or disable function prototypes
**
**  The macro 'P' is used to write function declarations.  For example:
**
**      char *          SyFgets ( char * line,  long length,  long fid );
**
**  For an ANSI C compiler the preproccesor will turn this into the following
**  prototype, which enables the  compiler  to  check  number  and  types  of
**  arguments in a function call for this function:
**
**      char *          SyFgets ( char * line,  long length,  long fid );
**
**  For a traditional C compiler the preprocessor will turn this into the old
**  form function declaration:
**
**      char *          SyFgets ();
*/


/****************************************************************************
**
*V  SyLibname . . . . . . . . . . . . . . . . . name of the library directory
**
**  'SyLibname' is the name of the directory where the GAP library files  are
**  located.
**
**  This is per default the subdirectory 'lib/'  of  the  current  directory.
**  It is usually changed with the '-l' option in the script that starts GAP.
**
**  Is copied into the GAP variable called 'LIBNAME'  and used by  'Readlib'.
**  This is also used in 'LIBNAME/init.g' to find the group library directory
**  by replacing 'lib' with 'grp', etc.
**
**  It must end with the pathname seperator, eg. if 'init.g' is the name of a
**  library file 'strcat( SyLibname, "init.g" );' must be a  valid  filename.
**  Further neccessary transformation of the filename are done  in  'SyOpen'.
**
**  Put in this package because the command line processing takes place here.
*/
extern  char            SyLibname [];


/****************************************************************************
**
*V  SyHelpname  . . . . . . . . . . . . . . name of the online help directory
**
**  'SyHelpname' is the name of the directory where the GAP online help files
**  are located.
**
**  By default it is computed from 'SyLibname' by replacing 'lib' with 'doc'.
**  It can be changed with the '-h' option.
**
**  It is used by 'SyHelp' to find the online documentation.
*/
extern  char            SyHelpname [];

/****************************************************************************
**
*V  SyBanner  . . . . . . . . . . . . . . . . . . . . . . . . surpress banner
**
**  'SyBanner' determines whether GAP should print the banner.
**
**  Per default it  is true,  i.e.,  GAP prints the  nice  banner.  It can be
**  changed by the '-b' option to have GAP surpress the banner.
**
**  It is copied into the GAP variable 'BANNER', which  is used  in 'init.g'.
**
**  Put in this package because the command line processing takes place here.
*/
extern  Int            SyBanner;


/****************************************************************************
**
*V  SyQuiet . . . . . . . . . . . . . . . . . . . . . . . . . surpress prompt
**
**  'SyQuit' determines whether GAP should print the prompt and  the  banner.
**
**  Per default its false, i.e. GAP prints the prompt and  the  nice  banner.
**  It can be changed by the '-q' option to have GAP operate in silent  mode.
**
**  It is used by the functions in 'gap.c' to surpress printing the  prompts.
**  Is also copied into the GAP variable 'QUIET' which is used  in  'init.g'.
**
**  Put in this package because the command line processing takes place here.
*/
extern  Int            SyQuiet;


/****************************************************************************
**
*V  SyNrCols  . . . . . . . . . . . . . . . . . .  length of the output lines
**
**  'SyNrCols' is the length of the lines on the standard output  device.
**
**  Per default this is 80 characters which is the usual width of  terminals.
**  It can be changed by the '-x' options for larger terminals  or  printers.
**
**  'Pr' uses this to decide where to insert a <newline> on the output lines.
**  'SyRead' uses it to decide when to start scrolling the echoed input line.
**
**  Put in this package because the command line processing takes place here.
*/
extern  Int            SyNrCols;


/****************************************************************************
**
*V  SyNrRows  . . . . . . . . . . . . . . . . . number of lines on the screen
**
**  'SyNrRows' is the number of lines on the standard output device.
**
**  Per default this is 24, which is the  usual  size  of  terminal  screens.
**  It can be changed with the '-y' option for larger terminals or  printers.
**
**  'SyHelp' uses this to decide where to stop with '-- <space> for more --'.
*/
extern  Int            SyNrRows;


extern	Int		SyMemMgrTrace;


/****************************************************************************
**
*V  SyMemory  . . . . . . . . . . . . . .  default size for initial workspace
**
**  'SyMemory' is the size of the  initial  workspace  allocated  by  Gasman.
**
**  This is per default  4 Megabyte,  which  is  often  a  reasonable  value.
**  It is usually changed with the '-m' option in the script that starts GAP.
**
**  This value is used in 'InitGasman' to allocate the initial workspace.
**
**  Put in this package because the command line processing takes place here.
*/
extern  Int            SyMemory;


/****************************************************************************
**
*V  SyInitfiles[] . . . . . . . . . . .  list of filenames to be read in init
**
**  'SyInitfiles' is a list of file to read upon startup of GAP.
**
**  It contains the 'init.g' file and a user specific init file if it exists.
**  It also contains all names all the files specified on the  command  line.
**
**  This is used in 'InitGap' which tries to read those files  upon  startup.
**
**  Put in this package because the command line processing takes place here.
**
**  For UNIX this list contains 'LIBNAME/init.g' and '$HOME/.gaprc'.
*/
extern  char            SyInitfiles [16] [256];

/****************************************************************************
**
*F  SyHelp( <topic>, <fid> )  . . . . . . . . . . . . . . display online help
**
*/
extern void             SyHelp ( char * topic, Int fin );

/****************************************************************************
**
*F  IsAlpha( <ch> ) . . . . . . . . . . . . .  is a character a normal letter
*F  IsDigit( <ch> ) . . . . . . . . . . . . . . . . .  is a character a digit
**
**  'IsAlpha' returns 1 if its character argument is a normal character  from
**  the range 'a..zA..Z' and 0 otherwise.
**
**  'IsDigit' returns 1 if its character argument is a digit from  the  range
**  '0..9' and 0 otherwise.
**
**  'IsAlpha' and 'IsDigit' are implemented in the declaration part  of  this
**  package as follows:
*/
#include        <ctype.h>
#define IsAlpha(ch)     (isalpha((int)ch))
#define IsDigit(ch)     (isdigit((int)ch))


/****************************************************************************
**
*F  SyFopen( <name>, <mode> ) . . . . . . . .  open the file with name <name>
**
**  The function 'SyFopen'  is called to open the file with the name  <name>.
**  If <mode> is "r" it is opened for reading, in this case  it  must  exist.
**  If <mode> is "w" it is opened for writing, it is created  if  neccessary.
**  If <mode> is "a" it is opened for appending, i.e., it is  not  truncated.
**
**  'SyFopen' returns an integer used by the scanner to  identify  the  file.
**  'SyFopen' returns -1 if it cannot open the file.
**
**  The following standard files names and file identifiers  are  guaranteed:
**  'SyFopen( "*stdin*", "r")' returns 0 identifying the standard input file.
**  'SyFopen( "*stdout*","w")' returns 1 identifying the standard outpt file.
**  'SyFopen( "*errin*", "r")' returns 2 identifying the brk loop input file.
**  'SyFopen( "*errout*","w")' returns 3 identifying the error messages file.
**
**  If it is necessary to adjust the  filename  this  should  be  done  here.
**  Right now GAP does not read nonascii files, but if this changes sometimes
**  'SyFopen' must adjust the mode argument to open the file in binary  mode.
*/
extern  Int            SyFopen ( char * name, char * mode );


/****************************************************************************
**
*F  SyFclose( <fid> ) . . . . . . . . . . . . . . . . .  close the file <fid>
**
**  'SyFclose' closes the file with the identifier <fid>  which  is  obtained
**  from 'SyFopen'.
*/
extern  void            SyFclose ( Int fid );



/****************************************************************************
**
*F  SyFgets( <line>, <lenght>, <fid> )  . . . . .  get a line from file <fid>
**
**  'SyFgets' is called to read a line from the file  with  identifier <fid>.
**  'SyFgets' (like 'fgets') reads characters until either  <length>-1  chars
**  have been read or until a <newline> or an  <eof> character is encoutered.
**  It retains the '\n' (unlike 'gets'), if any, and appends '\0' to  <line>.
**  'SyFgets' returns <line> if any char has been read, otherwise '(char*)0'.
**
**  'SyFgets'  allows to edit  the input line if the  file  <fid> refers to a
**  terminal with the following commands:
**
**      <ctr>-A move the cursor to the beginning of the line.
**      <esc>-B move the cursor to the beginning of the previous word.
**      <ctr>-B move the cursor backward one character.
**      <ctr>-F move the cursor forward one character.
**      <esc>-F move the cursor to the beginning of the next word.
**      <ctr>-E move the cursor to the end of the line.
**
**      <ctr>-H, <del> delete the character left of the cursor.
**      <ctr>-D delete the character under the cursor.
**      <ctr>-K delete up to the end of the line.
**      <esc>-D delete forward to the end of the next word.
**      <esc>-<del> delete backward to the beginning of the last word.
**      <ctr>-X delete entire input line, and discard all pending input.
**      <ctr>-Y insert (yank) a just killed text.
**
**      <ctr>-T exchange (twiddle) current and previous character.
**      <esc>-U uppercase next word.
**      <esc>-L lowercase next word.
**      <esc>-C capitalize next word.
**
**      <ctr>-L insert last input line before current character.
**      <ctr>-P redisplay the last input line, another <ctr>-P will redisplay
**              the line before that, etc.  If the cursor is not in the first
**              column only the lines starting with the string to the left of
**              the cursor are taken. The history is limitied to ~8000 chars.
**      <ctr>-N Like <ctr>-P but goes the other way round through the history
**      <esc>-< goes to the beginning of the history.
**      <esc>-> goes to the end of the history.
**      <ctr>-J accept this line and perform a <ctr>-N.
**
**      <ctr>-V enter next character literally.
**      <ctr>-U execute the next command 4 times.
**      <esc>-<num> execute the next command <num> times.
**      <esc>-<ctr>-L repaint input line.
**
**  Not yet implemented commands:
**
**      <tab>   complete the identifier before the cursor.
**      <ctr>-S search interactive for a string forward.
**      <ctr>-R search interactive for a string backward.
**      <esc>-Y replace yanked string with previously killed text.
**      <ctr>-_ undo a command.
**      <esc>-T exchange two words.
*/
extern  char *          SyFgets ( char * line,  Int length,  Int fid );


/****************************************************************************
**
*V  SyFputs( <line>, <fid> )  . . . . . . . .  write a line to the file <fid>
**
**  'SyFputs' is called to put the  <line>  to the file identified  by <fid>.
*/
extern  void            SyFputs ( char * line, Int fid );


/****************************************************************************
**
*F  SyPinfo( <nr>, <size> ) . . . . . . . . . . . . . . .  print garbage info
**
**  'SyPinfo' is called from  Gasman to inform the  window handler  about the
**  current  Gasman   statistics.  <nr> determines   the   phase the  garbage
**  collection is currently  in, and <size>  is the correspoding value, e.g.,
**  number of live bags.
*/
extern void             SyPinfo ( int nr, Int size );


/****************************************************************************
**
*F  SyWinCmd( <str>, <len> )  . . . . . . . . . . . .  . execute a window cmd
**
**  'SyWinCmd' send   the  command <str> to  the   window  handler (<len>  is
**  ignored).  In the string <str> '@' characters are duplicated, and control
**  characters  are converted to  '@<chr>', e.g.,  <newline> is converted  to
**  '@J'.  Then  'SyWinCmd' waits for  the window handlers answer and returns
**  that string.
*/
extern char *           SyWinCmd ( char * str, Int len );


/****************************************************************************
**
*F  SyIsIntr()  . . . . . . . . . . . . . . . . check wether user hit <ctr>-C
**
**  'SyIsIntr' is called from the evaluator at  regular  intervals  to  check
**  wether the user hit '<ctr>-C' to interrupt a computation.
**
**  'SyIsIntr' returns 1 if the user typed '<ctr>-C' and 0 otherwise.
*/
extern  Int            SyIsIntr ( void );


/****************************************************************************
**
*F  SyExit( <ret> ) . . . . . . . . . . . . . exit GAP with return code <ret>
**
**  'SyExit' is the offical  way  to  exit GAP, bus errors are the inoffical.
**  The function 'SyExit' must perform all the neccessary cleanup operations.
**  If ret is 0 'SyExit' should signal to a calling proccess that all is  ok.
**  If ret is 1 'SyExit' should signal a  failure  to  the  calling proccess.
*/

#define         SYEXIT_OK                       0
/* exit from brk loop while gap is running with "-i batch" interface */
#define         SYEXIT_FROM_BRK                 5
/* exiting normally but some syntax errors occured at run time while 
   gap was running with "-i batch" interface */
#define         SYEXIT_WITH_SYNTAX_ERRORS       6

extern  void            SyExit ( Int ret );


/****************************************************************************
**
*F  SyExec( <cmd> ) . . . . . . . . . . . execute command in operating system
**
**  'SyExec' executes the command <cmd> (a string) in the operating system.
**
**  'SyExec'  should call a command  interpreter  to execute the command,  so
**  that file name expansion and other common  actions take place.  If the OS
**  does not support this 'SyExec' should print a message and return.
*/
extern  int             SyExec ( char * cmd );


/****************************************************************************
**
*F  SyTime()  . . . . . . . . . . . . . . . return time spent in milliseconds
**
**  'SyTime' returns the number of milliseconds spent by GAP so far.
**
**  Should be as accurate as possible,  because it  is  used  for  profiling.
*/
extern  UInt   SyTime ( void );


/****************************************************************************
**
*F  SyTmpname() . . . . . . . . . . . . . . . . . return a temporary filename
*/
extern  char *          SyTmpname ( void );


/****************************************************************************
**
*F  SyGetmen( <size> )  . . . . . . . . allocate memory block of <size> bytes
**
**  'SyGetmem' gets  a block of  <size>  bytes from the  operating system and
**  returns a pointer  to it.  <size> must  be a multiple  of 4 and the block
**  returned by 'SyGetmem' is lonword aligned.  It is cleared to contain only
**  zeroes.  If  there  is not enough  memory  available returns '(char*)-1'.
**  'SyGetmem' returns adjacent  blocks on subsequent calls, otherwise Gasman
**  would get confused.
*/
extern  char *          SyGetmem ( Int size );


/****************************************************************************
**
*F  InitSystem( <argc>, <argv> )  . . . . . . . . . initialize system package
**
**  'InitSystem' is called very early during the initialization from  'main'.
**  It is passed the command line array  <argc>, <argv>  to look for options.
*/
extern  void            InitSystem ( int argc, char * argv [] );

/****************************************************************************
**
*F  SyFmtime( <filename> ) . . . . . . . . . . . . get file modification time
**
**  'SyFmtime' get the filename modification time
*/

extern unsigned long long SyFmtime(char *filename);


/****************************************************************************
**
*F  SyGetPid( <filename> ) . . . . . . . . . . . . get file modification time
**
**  'SyGetPid' get the process ID
*/

extern UInt SyGetPid(void);

/****************************************************************************
**
*F  SuperMakeDir( <filename> ) . . . . . . . . . . . . . . . .  create a path
**
**  'SuperMakeDir' creates a path with multiple new directories, if necessary
**  if the dir already exists, it succeeds.
*/

extern int SuperMakeDir(char *dirname);

/****************************************************************************
 **
 *F  SyLoadHistory()  . . . . . . . . . . loads a file into the history buffer
 **
 ****************************************************************************/

void SyLoadHistory();

/****************************************************************************
 **
 *F  SySaveHistory()  . . . . . . . . . writes the history buffer into a file
 **
 ****************************************************************************/

void SySaveHistory();

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
typedef UInt** Bag;


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
*V  SyMsgsFlagBags  . . . . . . . . . . . . . . . . .  enable gasman messages
**
*/
extern UInt SyMsgsFlagBags;


/****************************************************************************
**
*V  SyStorMin . . . . . . . . . . . . . .  default size for initial workspace
**
**  'SyStorMin' is the size of the initial workspace allocated by Gasman,
**  in kilobytes
**
**  The default size is 0; the value must be specifed with the '-m' option on
**  the command line to start GAP.
*/
extern Int SyStorMin;


/****************************************************************************
**

*F * * * * * * * * * * * * * * gasman interface * * * * * * * * * * * * * * *
*/

/****************************************************************************
**
*F  SyAllocBags( <size> ) . . . . . . allocate memory block of <size> bytes
**
**  See description in system4.c
*/
extern UInt*** SyAllocBags(Int size);


/****************************************************************************
**
*F  SyAbortBags(<msg>)  . . . . . . . . . . abort GAP in case of an emergency
**
**  'SyAbortBags' is the function called by Gasman in case of an emergency.
*/
extern void SyAbortBags(
    Char* msg);


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
extern void InitSystem4(
    Int                 argc,
    Char* argv[]);



typedef void            (*TNumAbortFuncBags) (
    Char* msg);

extern  TNumAbortFuncBags       AbortFuncBags;


#endif // _SYSTEM_H

