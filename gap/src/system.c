/****************************************************************************
**
*Y  Copyright (C) 2018-2020, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2020) by the GAP Group (www.gap-system.org).
**
*/

#include		<stdlib.h>
#include        "system.h"              /* declaration part of the package */
#include        "spiral.h"              /* InitLibName() */
#include        "interface.h"           /* Outside Interface */
#include        "iface.h"
#include		"GapUtils.h"


#ifdef WIN32

#define	WIN32_ANSICOLOR_EMU
#define WIN32_CTRLV_SUPPORT
#include <direct.h> // for mkdir, getdrive, etc.
#include <process.h> // for getpid()
#include <windows.h>

#endif

#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif

#ifdef SYS_HAS_ANSI
# define SYS_ANSI       SYS_HAS_ANSI
#else
# ifdef __STDC__
#  define SYS_ANSI      1
# else
#  define SYS_ANSI      0
# endif
#endif

#ifdef SYS_HAS_CONST
# define SYS_CONST      SYS_HAS_CONST
#else
# ifdef __STDC__
#  define SYS_CONST     const
# else
#  define SYS_CONST
# endif
#endif


/***************
 * Interface flag 
 */

extern int CURR_INTERFACE;
extern struct gap_iface gap_interface[];



/****************************************************************************
**
*V  SyStackSpace  . . . . . . . . . . . . . . . . . . . amount of stack space
**
**  'SyStackSpace' is the amount of stackspace that GAP gets.
**
**  On the  Mac special actions must  be  taken to ensure  that
**  enough space is available.
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
#if SYS_BSD || SYS_USG 
char            SyLibname [2048] = "lib/";
#endif
#if WIN32
char            SyLibname [2048] = "";
#endif

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
char            SyHelpname [2048];


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
Int            SyBanner = 1;


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
Int            SyQuiet = 0;


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
Int            SyNrCols = 80;


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
Int            SyNrRows = 24;


/****************************************************************************
**
*V  SyMemMgrTrace  . . . . . . . . . . . . . . . . . . enable gasman messages
**
**  'SyMemMgrTrace' is used to turn on/off diagnostic and statistic messages
**  during garbage collection. 
**
**  By default it is false, i.e., Memory manager is silent during garbage collections.
**  It can be changed by using the  '-g' option (deprecated) on the GAP command line.
**
*/
Int		SyMemMgrTrace = 0;


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
#if SYS_BSD || SYS_USG 
Int            SyMemory = 4 * 1024 * 1024;
#endif
#if WIN32
Int            SyMemory = 4 * 1024 * 1024;
#endif


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
char            SyInitfiles [16] [256];


/****************************************************************************
**
*V  syWindow  . . . . . . . . . . . . . . . .  running under a window handler
**
**  'syWindow' is 1 if GAP  is running under  a window handler front end such
**  as 'xgap', and 0 otherwise.
**
**  If running under  a window handler front  end, GAP adds various  commands
**  starting with '@' to the output to let 'xgap' know what is going on.
*/
Int            syWindow = 0;


/****************************************************************************
**
*V  syStartTime . . . . . . . . . . . . . . . . . . time when GAP was started
*V  syStopTime  . . . . . . . . . . . . . . . . . . time when reading started
*/
UInt   syStartTime;
UInt   syStopTime;



/****************************************************************************
**
*V  'syBuf' . . . . . . . . . . . . .  buffer and other info for files, local
**
**  'syBuf' is  a array used as  buffers for  file I/O to   prevent the C I/O
**  routines  from   allocating theis  buffers  using  'malloc',  which would
**  otherwise confuse Gasman.
*/
#ifndef SYS_STDIO_H                     /* standard input/output functions */
# include       <stdio.h>
# define SYS_STDIO_H
#endif

struct {
    FILE *      fp;                     /* file pointer for this file      */
    FILE *      echo;                   /* file pointer for the echo       */
    char        buf [BUFSIZ];           /* the buffer for this file        */
}       syBuf [16];


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
Int            SyFopen (char * name, char *mode )
{
    Int                fid;

    /* handle standard files                                               */
    if ( strcmp( name, "*stdin*" ) == 0 ) {
        if ( strcmp( mode, "r" ) != 0 )  return -1;
        return 0;
    }
    else if ( strcmp( name, "*stdout*" ) == 0 ) {
        if ( strcmp( mode, "w" ) != 0 )  return -1;
        return 1;
    }
    else if ( strcmp( name, "*errin*" ) == 0 ) {
        if ( strcmp( mode, "r" ) != 0 )  return -1;
        if ( syBuf[2].fp == (FILE*)0 )  return -1;
        return 2;
    }
    else if ( strcmp( name, "*errout*" ) == 0 ) {
        if ( strcmp( mode, "w" ) != 0 )  return -1;
        return 3;
    }

    /* try to find an unused file identifier                               */
    for ( fid = 4; fid < sizeof(syBuf)/sizeof(syBuf[0]); ++fid )
        if ( syBuf[fid].fp == (FILE*)0 )  break;
    if ( fid == sizeof(syBuf)/sizeof(syBuf[0]) )
        return (Int)-1;

    /* try to open the file                                                */
    syBuf[fid].fp = fopen( name, mode );
    if ( syBuf[fid].fp == (FILE*)0 )
        return (Int)-1;

    /* allocate the buffer                                                 */
    setbuf( syBuf[fid].fp, syBuf[fid].buf );

    /* return file identifier                                              */
    return fid;
}


/****************************************************************************
**
*F  SyFclose( <fid> ) . . . . . . . . . . . . . . . . .  close the file <fid>
**
**  'SyFclose' closes the file with the identifier <fid>  which  is  obtained
**  from 'SyFopen'.
*/
void            SyFclose (Int fid )
{
    /* check file identifier                                               */
    if ( syBuf[fid].fp == (FILE*)0 ) {
        fputs("gap: panic 'SyFclose' asked to close closed file!\n",stderr);
        SyExit( 1 );
    }

    /* refuse to close the standard files                                  */
    if ( fid == 0 || fid == 1 || fid == 2 || fid == 3 ) {
        return;
    }

    /* try to close the file                                               */
    if ( fclose( syBuf[fid].fp ) == EOF ) {
        fputs("gap: 'SyFclose' cannot close file, ",stderr);
        fputs("maybe your file system is full?\n",stderr);
    }

    /* mark the buffer as unused                                           */
    syBuf[fid].fp = (FILE*)0;

	syBuf[fid].buf[0] = '\0';
}

Int	SyChDir(const char* filename)
{
	if (!chdir(filename)) {
		return 1;
	}

	return 0;
}

/****************************************************************************
**
*F  SyFmtime( <filename> ) . . . . . . .  get modify time for file <filename>
**
*/
#include <sys/types.h>
#include <sys/stat.h>

unsigned long long SyFmtime(char *filename)
{
	struct stat s;

    /* try to stat the file                                               */
    if ( stat(filename, &s )  ) {
		return 0;
    }

	return (unsigned long long) s.st_mtime;
}


/****************************************************************************
**
*F  SyGetPid( ) . . . . . . . . . . . . . . . . . . . . .  get GAP process id
**
*/
UInt SyGetPid(void)
{
	return getpid();
}

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
**      <ctr>-F move the cursor forward  one character.
**      <esc>-F move the cursor to the end of the next word.
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
**      <tab>   complete the identifier before the cursor.
**      <ctr>-L insert last input line before current character.
**      <ctr>-P redisplay the last input line, another <ctr>-P will redisplay
**              the line before that, etc.  If the cursor is not in the first
**              column only the lines starting with the string to the left of
**              the cursor are taken. The history is limitied to ~8000 chars.
**      <ctr>-N Like <ctr>-P but goes the other way round through the history
**      <esc>-< goes to the beginning of the history.
**      <esc>-> goes to the end of the history.
**      <ctr>-O accept this line and perform a <ctr>-N.
**
**      <ctr>-V enter next character literally.
**      <ctr>-U execute the next command 4 times.
**      <esc>-<num> execute the next command <num> times.
**      <esc>-<ctr>-L repaint input line.
**
**  Not yet implemented commands:
**
**      <ctr>-S search interactive for a string forward.
**      <ctr>-R search interactive for a string backward.
**      <esc>-Y replace yanked string with previously killed text.
**      <ctr>-_ undo a command.
**      <esc>-T exchange two words.
*/
extern  int             syStartraw ( Int fid );
extern  void            syStopraw  ( Int fid );
extern  int             syGetch    ( Int fid );
extern  void            syEchoch   ( int ch, Int fid );
extern  void            syEchos    ( char * str, Int fid );

extern  UInt   iscomplete ( char *           name,
                                       UInt    len,
                                       UInt    rn );
extern  UInt   completion ( char *           name,
                                       UInt    len,
                                       UInt    rn );

extern  void            syWinPut   ( Int fid, char * cmd, char * str );

/* inBreakLoop from gap.c */
extern Int     inBreakLoop();

Int            syLineEdit = 1;         /* 0: no line editing              */
                                        /* 1: line editing if terminal     */
                                        /* 2: always line editing (EMACS)  */
Int            syCTRD = 1;             /* true if '<ctr>-D' is <eof>      */
Int            syNrchar;               /* nr of chars already on the line */
char           syPrompt [1024];         /* upped from 256; characters alread on the line   */

char            syHistory [32768];      /* history of command lines        */
char *          syHi = syHistory;       /* actual position in history      */
int             syCTRO;                 /* number of '<ctr>-O' pending     */

#define CTR(C)          ((C) & 0x1F)    /* <ctr> character                 */
#define ESC(C)          ((C) | 0x100)   /* <esc> character                 */
#define CTV(C)          ((C) | 0x200)   /* <ctr>V quotes characters        */

#define IS_SEP(C)       (!IsAlpha(C) && !IsDigit(C) && (C)!='_')
#define IS_PARPOINT(C)      (((C)=='(')||((C)=='.'))

extern int isReadValFromFile;
extern int addEndOfLineOnlyOnce;

char *          SyFgets (char line[], Int length, Int fid )
{
    Int                 ch,  ch2,  ch3, last;
    char                * p,  * q,  * r,  * s,  * t;
    char                * h;
    static char         yank [512];
    static char         smartcompletionsave [512];
    static int          smartcomplete=0;
    static int          viCommandMode = 0;
#ifdef WIN32_CTRLV_SUPPORT
    static char		win_clipboard_buffer[8192] = "";
    static char		* win_clipboard = win_clipboard_buffer;
#endif
    int			forRecord = 1; /* indicates that we should save line in history */
    char                old [512],  new [512];
    Int                oldc,  newc;
    Int                rep;
    char                buffer [512];
    Int                rn;
    
    /* no line editing if the file is not '*stdin*' or '*errin*'           */
    if ( fid != 0 && fid != 2 ) {
        p = fgets( line, (Int)length, syBuf[fid].fp );
		if(isReadValFromFile){
 			if(!p){
				if(addEndOfLineOnlyOnce){
					line[0] = '\n';
					line[1] = '\0';
					p = line;
					addEndOfLineOnlyOnce = 0;
				}
			}
		}
        return p;
    }

    /* Joohoon Interface reader implementation here */
    
    if( CURR_INTERFACE ){
      syStopTime = SyTime();
      p = interface_read_input_nolist( line );
      syStartTime += SyTime() - syStopTime;
      return p;
    }

    /* Should not get below this point with new interface */
    /* Must clean up this part */


    /* no line editing if the user disabled it                             */
    if ( syLineEdit == 0 ) {
      syStopTime = SyTime();
      p = fgets( line, (Int)length, syBuf[fid].fp );
      syStartTime += SyTime() - syStopTime;
      return p;
    }

    /* no line editing if the file cannot be turned to raw mode            */
    if ( syLineEdit == 1 && ! syStartraw(fid) ) {
        syStopTime = SyTime();
        p = fgets( line, (Int)length, syBuf[fid].fp );
        syStartTime += SyTime() - syStopTime;
        return p;
    }

    /* stop the clock, reading should take no time                         */
    syStopTime = SyTime();

    /* the line starts out blank                                           */
    line[0] = '\0';  p = line;  h = syHistory;
    for ( q = old; q < old+sizeof(old); ++q )  *q = ' ';
    oldc = 0;
    last = 0;

    if (smartcomplete){
      rep = 1;  ch2 = 0; ch=0;
      sprintf(line,"%s",smartcompletionsave);
      p = line + strlen(line);
    }

    while ( 1 ) {
	Int hexEsc = 0;
        /* get a character, handle <ctr>V<chr>, <esc><num> and <ctr>U<num> */
        rep = 1;  ch2 = 0;
#ifdef WIN32_CTRLV_SUPPORT
	if ( *win_clipboard != '\0') {
	    ch = *win_clipboard++;
	    if (*win_clipboard == '\0') {
		win_clipboard_buffer[0] = '\0';
		win_clipboard = win_clipboard_buffer;
	    }
	} else
#endif
	if (!smartcomplete)
        do {
            if ( syCTRO % 2 == 1  )  { ch = CTR('N'); syCTRO = syCTRO - 1; }
            else if ( syCTRO != 0 )  { ch = CTR('O'); rep = syCTRO / 2; }
            else ch = syGetch(fid);
	    //printf("---->%d<-----\n", (unsigned long)ch);
#ifndef WIN32_CTRLV_SUPPORT
            if ( ch2==0        && ch==CTR('V') ) {             ch2=ch; ch=0;}
#endif
            if ( ch2==0        && ch==CTR('[') ) {             ch2=ch; ch=0;}
            if ( ch2==0        && ch==CTR('U') ) {             ch2=ch; ch=0;}
            if ( ch2==CTR('[') && ch==CTR('V') ) { ch2=ESC(CTR('V'));  ch=0;}
            if ( ch2==CTR('[') && isdigit(ch)  ) { rep=ch-'0'; ch2=ch; ch=0;}
            if ( ch2=='['      && isdigit(ch)  ) { rep=ch-'0'; ch2=ch; ch=0;}
            if ( ch2==CTR('[') && ch=='['      ) {             ch2=ch; ch=0;}
            if ( ch2==CTR('U') && ch==CTR('V') ) { rep=4*rep;  ch2=ch; ch=0;}
            if ( ch2==CTR('U') && ch==CTR('[') ) { rep=4*rep;  ch2=ch; ch=0;}
            if ( ch2==CTR('U') && ch==CTR('U') ) { rep=4*rep;  ch2=ch; ch=0;}
            if ( ch2==CTR('U') && isdigit(ch)  ) { rep=ch-'0'; ch2=ch; ch=0;}
            if ( isdigit(ch2)  && ch==CTR('V') ) {             ch2=ch; ch=0;}
            if ( isdigit(ch2)  && ch==CTR('[') ) {             ch2=ch; ch=0;}
            if ( isdigit(ch2)  && ch==CTR('U') ) {             ch2=ch; ch=0;}
            if ( isdigit(ch2)  && isdigit(ch)  ) { rep=10*rep+ch-'0';  ch=0;}
            if ( hexEsc	       && isxdigit(ch) )  {
		rep = (rep << 4) | ((ch>'9') ? toupper(ch)-'A'+10 : ch-'0'); 
		if (ch2==';') { 
		    ch2=0; ch=0; 
		} else ch2=';';
	    }
            if ( isdigit(ch2) && rep==1 && ch==';' ) { rep=0; ch2=';'; ch=0; hexEsc=1; }
            if (viCommandMode && ch=='d'      ) {             ch2=ch; ch=0;}
            if (viCommandMode && ch=='c'      ) {             ch2=ch; ch=0;}
            
        } while ( ch == 0 );

	if (hexEsc) { 
	    ch2 = 0;
	    switch (rep) {
		case 0x5D:	/* Ctrl+Left */
		case 0x3D: {	/* Alt+Left */
		    ch = ESC('B');
		    break;
		}
		case 0x5C:	/* Ctrl+Right */
		case 0x3C: {	/* Alt+Right */
		    ch = ESC('F');
		    break;
		}
		case 0x5A:	/* Ctrl+Up */
		case 0x3A: {	/* Alt+Up */
		    ch = ESC('P');
		    break;
		}
		case 0x5B:	/* Ctrl+Down */
		case 0x3B: {	/* Alt+Down */
		    ch = ESC('N');
		    break;
		}
		default:
		    ch = ESC(rep); 
	    } 
	    rep=1; 
	}
	if ( ch =='~' && isdigit(ch2) ) { 
	    switch (rep) {
		case 1:  { ch = CTR('A'); break; } /* Home */
		case 3:  { ch = CTR('D'); break; } /* Delete */
		case 4:  { ch = CTR('E'); break; } /* End */
		case 5:  { ch = CTR('P'); break; } /* PgUp */
		case 6:  { ch = CTR('N'); break; } /* PgDown */
		default: { ch = ESC(rep); break; } /* Fxx keys, we need normal escapes parser */
	    }
	    ch2 = '['; rep = 1; 
	}
        if ( ch2==CTR('V') )       ch  = CTV(ch);
        if ( ch2==ESC(CTR('V')) )  ch  = CTV(ch | 0x80);
        if ( ch2==CTR('[') )       ch  = ESC(ch);
        if ( ch2==CTR('U') )       rep = 4*rep;
        if ( ch2=='[' ) {
	    switch (ch) {
		case 'A': { ch  = CTR('P'); break; }
		case 'B': { ch  = CTR('N'); break; }
		case 'C': { ch  = CTR('F'); break; }
		case 'D': { ch  = CTR('B'); break; }
		case 'F': { ch  = CTR('E'); break; }
		case 'H': { ch  = CTR('A'); break; }
	    }
	}
        

        if (viCommandMode) { /* Map vi keys back to the original key bindings */

          /* Note: This way, one can potentially also use emacs keys when in
           * vi-movement mode. This means that we don't have to switch
           * explicitly between emacs and vi modes, but stay in a mixed mode
           * always. Extra fun! */

          if (ch2 == 0) {
            /* Char  movement */
            if (ch == 'h') { ch = CTR('b'); }
            if (ch == 'l') { ch = CTR('f'); }
            if (ch == 'j') { ch = CTR('P'); }
            if (ch == 'k') { ch = CTR('N'); }

            /* Word movement (NOTE: not differentiating lower/upper case since
               gap, doesn't handle them differently.) */
            if (ch == 'b'|| ch == 'B') { ch = ESC('b'); }
            if (ch == 'w'|| ch == 'W') { ch = ESC('f'); }

            /* Line movement */
            if (ch == '0') { ch = CTR('A'); }
            if (ch == '$') { ch = CTR('E'); }

            /* Deletion / Change */
            if (ch == 'x') { ch = CTR('D'); }
            if (ch == 'D') { ch = CTR('K'); }
            if (ch == 'C') { ch = CTR('K'); viCommandMode = 0; }

            /* Put */
            if (ch == 'p' || ch=='P') { ch = CTR('Y'); }

            /* Commands that'll get us into insert-mode */
            if (ch == 'A') { ch = CTR('E'); viCommandMode = 0; }

          }
          else {
            if (ch2 == 'd') { /* Delete word */
              ch2 = 0;
              if (ch == 'b'|| ch == 'B') { ch = ESC(127); }
              if (ch == 'w'|| ch == 'W') { ch = ESC('d'); }
            }
            if (ch2 == 'c') { /* Change word */
              if (ch == 'b'|| ch == 'B') { ch = ESC(127); viCommandMode = 0;}
              if (ch == 'w'|| ch == 'W') { ch = ESC('d'); viCommandMode = 0;}
            }
          }
        }

        /* now perform the requested action <rep> times in the input line  */
	while (( rep-- > 0 )&&(!smartcomplete)){
            switch ( ch ) {

            case CTR('G'): /* change to viCommandMode                     */
              /* Ctrl-G is used to get us into vi's Normal mode instead
               * of the customary Esc, since a) Ctr-G is easier to access, and
               * b) Esc keys can still be used in emacs mode */
              viCommandMode = 1;
              rep = 0;
              break;

            case CTR('A'): /* move cursor to the start of the line         */
                while ( p > line )  --p;
                break;

            case ESC('B'): /* move cursor one word to the left             */
            case ESC('b'):
                if ( p > line ) do {
                    --p;
                } while ( p>line && (!IS_SEP(*(p-1)) || IS_SEP(*p)));
                break;

            case CTR('B'): /* move cursor one character to the left        */
                if ( p > line )  --p;
                break;

            case CTR('F'): /* move cursor one character to the right       */
                if ( *p != '\0' )  ++p;
                break;

            case ESC('F'): /* move cursor one word to the right            */
            case ESC('f'):
                if ( *p != '\0' ) do {
                    ++p;
                } while ( *p!='\0' && (IS_SEP(*(p-1)) || !IS_SEP(*p)));
                break;

            case CTR('E'): /* move cursor to the end of the line           */
                while ( *p != '\0' )  ++p;
                break;

            case CTR('H'): /* delete the character left of the cursor      */
            case 127:
                if ( p == line ) break;
                --p;
                /* let '<ctr>-D' do the work                               */

            case CTR('D'): /* delete the character at the cursor           */
                           /* on an empty line '<ctr>-D' is <eof>          */
                if ( p == line && *p == '\0' && syCTRD ) {
                    ch = EOF; rep = 0; break;
                }
                if ( *p != '\0' ) {
                    for ( q = p; *(q+1) != '\0'; ++q )
                        *q = *(q+1);
                    *q = '\0';
                }
                break;

            case CTR('X'): /* delete the line                              */
                p = line;
                /* let '<ctr>-K' do the work                               */

            case CTR('K'): /* delete to end of line                        */
                if ( last!=CTR('X') && last!=CTR('K') && last!=ESC(127)
                  && last!=ESC('D') && last!=ESC('d') )  yank[0] = '\0';
                for ( r = yank; *r != '\0'; ++r ) ;
                for ( s = p; *s != '\0'; ++s )  r[s-p] = *s;
                r[s-p] = '\0';
                *p = '\0';
                break;

            case ESC(127): /* delete the word left of the cursor           */
                q = p;
                if ( p > line ) do {
                    --p;
                } while ( p>line && (!IS_SEP(*(p-1)) || IS_SEP(*p)));
                if ( last!=CTR('X') && last!=CTR('K') && last!=ESC(127)
                  && last!=ESC('D') && last!=ESC('d') )  yank[0] = '\0';
                for ( r = yank; *r != '\0'; ++r ) ;
                for ( ; yank <= r; --r )  r[q-p] = *r;
                for ( s = p; s < q; ++s )  yank[s-p] = *s;
                for ( r = p; *q != '\0'; ++q, ++r )
                    *r = *q;
                *r = '\0';
                break;

            case ESC('D'): /* delete the word right of the cursor          */
            case ESC('d'):
                q = p;
                if ( *q != '\0' ) do {
                    ++q;
                } while ( *q!='\0' && (IS_SEP(*(q-1)) || !IS_SEP(*q)));
                if ( last!=CTR('X') && last!=CTR('K') && last!=ESC(127)
                  && last!=ESC('D') && last!=ESC('d') )  yank[0] = '\0';
                for ( r = yank; *r != '\0'; ++r ) ;
                for ( s = p; s < q; ++s )  r[s-p] = *s;
                r[s-p] = '\0';
                for ( r = p; *q != '\0'; ++q, ++r )
                    *r = *q;
                *r = '\0';
                break;

            case CTR('T'): /* twiddle characters                           */
                if ( p == line )  break;
                if ( *p == '\0' )  --p;
                if ( p == line )  break;
                ch2 = *(p-1);  *(p-1) = *p;  *p = ch2;
                ++p;
                break;

            case CTR('L'): /* insert last input line                       */
                for ( r = syHistory; *r != '\0' && *r != '\n'; ++r ) {
                    ch2 = *r;
                    for ( q = p; ch2; ++q ) {
                        ch3 = *q; *q = ch2; ch2 = ch3;
                    }
                    *q = '\0'; ++p;
                }
                break;

            case CTR('Y'): /* insert (yank) deleted text                   */
                for ( r = yank; *r != '\0' && *r != '\n'; ++r ) {
                    ch2 = *r;
                    for ( q = p; ch2; ++q ) {
                        ch3 = *q; *q = ch2; ch2 = ch3;
                    }
                    *q = '\0'; ++p;
                }
                break;

            case CTR('P'): /* fetch old input line                         */
                while ( *h != '\0' ) {
                    for ( q = line; q < p; ++q )
                        if ( *q != h[q-line] )  break;
                    if ( q == p )  break;
                    while ( *h != '\n' && *h != '\0' )  ++h;
                    if ( *h == '\n' ) ++h;
                }
                q = p;
                while ( *h!='\0' && h[q-line]!='\n' && h[q-line]!='\0' ) {
                    *q = h[q-line];  ++q;
                }
                *q = '\0';
                while ( *h != '\0' && *h != '\n' )  ++h;
                if ( *h == '\n' ) ++h;  else h = syHistory;
                syHi = h;
                break;

            case CTR('N'): /* fetch next input line                        */
                h = syHi;
                if ( h > syHistory ) {
                    do {--h;} while (h>syHistory && *(h-1)!='\n');
                    if ( h==syHistory )  while ( *h != '\0' ) ++h;
                }
                while ( *h != '\0' ) {
                    if ( h==syHistory )  while ( *h != '\0' ) ++h;
                    do {--h;} while (h>syHistory && *(h-1)!='\n');
                    for ( q = line; q < p; ++q )
                        if ( *q != h[q-line] )  break;
                    if ( q == p )  break;
                    if ( h==syHistory )  while ( *h != '\0' ) ++h;
                }
                q = p;
                while ( *h!='\0' && h[q-line]!='\n' && h[q-line]!='\0' ) {
                    *q = h[q-line];  ++q;
                }
                *q = '\0';
                while ( *h != '\0' && *h != '\n' )  ++h;
                if ( *h == '\n' ) ++h;  else h = syHistory;
                syHi = h;
                break;

            case ESC('<'): /* goto beginning of the history                */
                while ( *h != '\0' ) ++h;
                do {--h;} while (h>syHistory && *(h-1)!='\n');
                q = p = line;
                while ( *h!='\0' && h[q-line]!='\n' && h[q-line]!='\0' ) {
                    *q = h[q-line];  ++q;
                }
                *q = '\0';
                while ( *h != '\0' && *h != '\n' )  ++h;
                if ( *h == '\n' ) ++h;  else h = syHistory;
                syHi = h;
                break;

            case ESC('>'): /* goto end of the history                      */
                h = syHistory;
                p = line;
                *p = '\0';
                syHi = h;
                break;

	    /* 
            case CTR('S'): /* search for a line forward                    */
                /* search for a line forward, not fully implemented !!!    */
            /*    if ( *p != '\0' ) {
                    ch2 = syGetch(fid);
                    q = p+1;
                    while ( *q != '\0' && *q != ch2 )  ++q;
                    if ( *q == ch2 )  p = q;
                }
                break;

            case CTR('R'): /* search for a line backward                   */
                /* search for a line backward, not fully implemented !!!   */
            /*    if ( p > line ) {
                    ch2 = syGetch(fid);
                    q = p-1;
                    while ( q > line && *q != ch2 )  --q;
                    if ( *q == ch2 )  p = q;
                }
                break;
            */
            case ESC('U'): /* uppercase word                               */
            case ESC('u'):
                if ( *p != '\0' ) do {
                    if ('a' <= *p && *p <= 'z')  *p = *p + 'A' - 'a';
                    ++p;
                } while ( *p!='\0' && (IS_SEP(*(p-1)) || !IS_SEP(*p)));
                break;
#ifdef WIN32_CTRLV_SUPPORT
	    case CTR('V'): { /* paste text from clipboard */
		HANDLE h;
		if (!IsClipboardFormatAvailable(CF_TEXT)) break;
		if (!OpenClipboard(0)) break;
		h = GetClipboardData(CF_TEXT); 
		if (h) { 
		    char* pClipboard = GlobalLock(h); 
		    if (pClipboard){
			if (strlen(pClipboard)+strlen(p)<sizeof(win_clipboard_buffer)) {
			    /* copy first line from clipboard */
			    for ( r = pClipboard; *r != '\0' && *r != '\n' && *r != '\r'; ++r ) {
				ch2 = *r;
				for ( q = p; ch2; ++q ) {
				    ch3 = *q; *q = ch2; ch2 = ch3;
				}
				*q = '\0'; ++p;
			    }
			    /* move rest of the clipboard text and input line into local buffer */
			    if (*r ) {
				q = win_clipboard_buffer;
				while( *r ) if (*r != '\r') *q++ = *r++; else r++;
				r = p;
				while( *r ) *q++ = *r++;
				*p = '\0';
				*q = '\0';
				win_clipboard = win_clipboard_buffer;
			    }
			}
			GlobalUnlock(h); 
		    } 
		} 
		CloseClipboard(); 
		break;
	    }
#endif
            case ESC('C'): /* capitalize word                              */
            case ESC('c'):
                while ( *p!='\0' && IS_SEP(*p) )  ++p;
                if ( 'a' <= *p && *p <= 'z' )  *p = *p + 'A'-'a';
                if ( *p != '\0' ) ++p;
                /* lowercase rest of the word                              */

            case ESC('L'): /* lowercase word                               */
            case ESC('l'):
                if ( *p != '\0' ) do {
                    if ('A' <= *p && *p <= 'Z')  *p = *p + 'a' - 'A';
                    ++p;
                } while ( *p!='\0' && (IS_SEP(*(p-1)) || !IS_SEP(*p)));
                break;

            case ESC(CTR('L')): /* repaint input line                      */
                syEchoch('\n',fid);
                for ( q = syPrompt; q < syPrompt+syNrchar; ++q )
                    syEchoch( *q, fid );
                for ( q = old; q < old+sizeof(old); ++q )  *q = ' ';
                oldc = 0;
                break;

            case EOF:     /* end of file on input                          */
                break;

	    case CTR('J'): /* CTR('J') is \r and CTR('M') is \n            */
	    case CTR('M'): /* append \n and exit                           */
                while ( *p != '\0' )  ++p;
                *p++ = '\n'; *p = '\0';
                rep = 0;
                viCommandMode = 0;
                break;

            case CTR('O'): /* accept line, perform '<ctr>-N' next time     */
                while ( *p != '\0' )  ++p;
                *p++ = '\n'; *p = '\0';
                syCTRO = 2 * rep + 1;
                rep = 0;
                break;
	    /*SMARTCOMPLETION after a point or a parenthesis */
            case CTR('W'): {
                char* uppart;
                char* mybuffer;

                if (( (q = p) > line )&&(!IS_PARPOINT(*(q-1)))) {
                    do {
                	--q;
                    } while ( q>line && (!IS_PARPOINT(*(q-1)) || IS_PARPOINT(*q)));
                }
                if (line < q && (IS_PARPOINT(*(q-1)))){
                    int i, j = 0;

                    for(i=0; i<q-line; i++){
                        if ((*(q-i-2)==' ')&&(j==0)){
                            i++;
                            break;
                        }
                        if (*(q-i-2)=='('){
                            if (j==0){
                                i++;
                                break;
                            } else
                                j--;
                        }
                        if (*(q-i-2)==')')
                            j++;
                    }

                    sprintf(smartcompletionsave,"%s",line);
                    uppart = (char*) malloc((strlen(q)+1)*sizeof(char));
                    sprintf(uppart,"%s",q);
                    mybuffer = (char*) malloc(i*sizeof(char));
                    
                    for(j=0; j<i-1; j++) {
                        mybuffer[j]=*(q-i-0+j);
                    }
                    
                    mybuffer[j]=0;
                    
                    if (*(q-1)=='.')
                        sprintf(line,"SmartComplete(%s,\"%s\");\n",mybuffer,uppart);
                    else
                        sprintf(line,"Doc(%s);\n",mybuffer);
				
                    q = line;
                    p = strlen(line)+line;

                    free(mybuffer);
                    free(uppart);
                    ch='\n';

                    smartcomplete = 1;
                    forRecord = 0; /* do not save line in history */
                }
                break;
            }		    

            case CTR('I'): /* try to complete the identifier before dot    */

                // if the cursor is at the beginning of the line, or 
                // character before the cursor is a separator
                if ( p == line || IS_SEP(p[-1]) ) {
                    ch2 = ch & 0xff;
                    for ( q = p; ch2; ++q ) {
                        ch3 = *q; *q = ch2; ch2 = ch3;
                    }
                    *q = '\0'; ++p;
                } else {
                    // in costructs such as 'blahde.blXah' where X denotes
                    // cursor position, this will step backwards twice.
                    if ( (q = p) > line ) 
                        do {
                            --q;
                        } while ( q>line && (!IS_SEP(*(q-1)) || IS_SEP(*q)));

                    // true if q is a subexpression after a '.'
                    rn = (line < q && *(q-1) == '.');

                    // copy subexpression into special buffer
                    r = buffer;  s = q;
                    while ( s < p )  *r++ = *s++;
                    *r = '\0';
                    // if we find a name
                    if ( iscomplete( buffer, p-q, rn ) ) {
                        // if the user already tried to complete, beep.
                        if ( last != CTR('I') ) {
                            syEchoch( CTR('G'), fid );
                            // otherwise, display the candidates.
                        } else  {
                            syWinPut( fid, "@c", "" );
                            syEchos( "\n    ", fid );
                            syEchos( buffer, fid );
                            
                            while ( completion( buffer, p-q, rn ) ) {
                                syEchos( "\n    ", fid );
                                syEchos( buffer, fid );
                            }
                            syEchos( "\n", fid );
                            for ( q=syPrompt; q<syPrompt+syNrchar; ++q )
                                syEchoch( *q, fid );
                            for ( q = old; q < old+sizeof(old); ++q )
                                *q = ' ';
                            oldc = 0;
                            syWinPut( fid, (fid == 0 ? "@i" : "@e"), "" );
                        }
                    } else if ( ! completion( buffer, p-q, rn ) ) 
					{
                        if ( last != CTR('I') )
                            syEchoch( CTR('G'), fid );
                        else 
						{
                            syWinPut( fid, "@c", "" );
                            syEchos("\n    identifier has no completions\n",
                                    fid);
                            for ( q=syPrompt; q<syPrompt+syNrchar; ++q )
                                syEchoch( *q, fid );
                            for ( q = old; q < old+sizeof(old); ++q )
                                *q = ' ';
                            oldc = 0;
                            syWinPut( fid, (fid == 0 ? "@i" : "@e"), "" );
                        }
                    }
                    else 
					{
                        t = p;
                        for ( s = buffer+(p-q); *s != '\0'; s++ ) {
                            ch2 = *s;
                            for ( r = p; ch2; r++ ) {
                                ch3 = *r; *r = ch2; ch2 = ch3;
                            }
                            *r = '\0'; p++;
                        }
                        while ( t < p && completion( buffer, t-q, rn ) ) {
                            r = t;  s = buffer+(t-q);
                            while ( r < p && *r == *s ) {
                                r++; s++;
                            }
                            s = p;  p = r;
                            while ( *s != '\0' )  *r++ = *s++;
                            *r = '\0';
                        }
                        if ( t == p ) {
                            if ( last != CTR('I') )
                                syEchoch( CTR('G'), fid );
                            else {
                                syWinPut( fid, "@c", "" );
                                buffer[t-q] = '\0';
                                while ( completion( buffer, t-q, rn ) ) {
                                    syEchos( "\n    ", fid );
                                    syEchos( buffer, fid );
                                }
                                syEchos( "\n", fid );
                                for ( q=syPrompt; q<syPrompt+syNrchar; ++q )
                                    syEchoch( *q, fid );
                                for ( q = old; q < old+sizeof(old); ++q )
                                    *q = ' ';
                                oldc = 0;
                                syWinPut( fid, (fid == 0 ? "@i" : "@e"), "");
                            }
                        }
                    }
                }
                break;
                
            case ESC(CTR('J')):
	    case ESC(CTR('M')):
                /* Escape+Return, run EditDef() for word at cursor. */
                /* Falls into default switch branch if no word found   */
                if (line[0] != '\0') { 
                    q = p;
                    while ( q > line && IS_SEP(*q)) q--;
                    while ( q > line && (!IS_SEP(*(q-1)) || *(q-1)=='.')) q--;
                    if (*q != '\0' &&  !IS_SEP(*q)) {
                        int i;
                        p = q; while (*q != '\0' && (!IS_SEP(*q) || *q=='.')) q++;
                        /* copy line into buffer */
                        strncpy(buffer, line, sizeof(buffer)-1); buffer[sizeof(buffer)-1] = '\0';
                        i = strlen(buffer);
                        if (buffer[i-1] != '\n') {
                            if (i==sizeof(buffer)-1) i--;
                            buffer[i] = '\n'; buffer[i+1] = '\0';
                        }
                        /* wrap identifier into EditDef()\n */
                        memmove(line + 8, p, (q-p)*sizeof(char));
                        strncpy(line, "EditDef(", 8);
                        strcpy(line + (q-p) + 8, ");\n");
                        
                        forRecord = 2;  /* indicates that buffer variable contains string to put into history instead of line*/
                        p=strlen(line)+line;
                        ch='\n';
                        break;
                    }
                }

            default:      /* default, insert normal character              */
		if (inBreakLoop() && line[0] == '\0' && p == line) { /* check debugger shortcuts */
		    Int gotLine = 1;
		    switch (ch) {
		        case ESC(CTR('J')):
			case ESC(CTR('M')): { /* Escape+Return*/
			    sprintf(line,"EditTopFunc();\n");
			    break;
			}
			case CTR('\\'): { /* Ctrl+\ */
			    sprintf(line,"Top();\n");
			    break;
			}
			case ESC('P'): { /* Ctrl+Arrow Up */
			    sprintf(line,"Up();\n");
			    break;
			}
			case ESC('N'): { /* Ctrl+Arrow Down */
			    sprintf(line,"Down();\n"); 
			    break;
			}
			case ESC(19): { /* F8 */
			    sprintf(line,"StepOut(1);\n"); 
			    break;
			}
			case ESC(21): { /* F10 */
			    sprintf(line,"StepOver(1);\n"); 
			    break;
			}
			case ESC(23): { /* F11 */
			    sprintf(line,"StepInto(1);\n"); 
			    break;
			}
			default: {
			    gotLine = 0;
			    break;
			}
		    }
		    if (gotLine != 0) {   
			forRecord = 0; /* don't put into history */
			p=strlen(line)+line;
			ch='\n';
			break;
		    }
		}
                if (!viCommandMode) {
                   ch2 = ch & 0xff;
                   for ( q = p; ch2; ++q ) {
                       ch3 = *q; *q = ch2; ch2 = ch3;
                   }
                   *q = '\0'; ++p;
                }
                break;

            } /* switch ( ch ) */

            last = ch;

        }

        if ( viCommandMode && ch=='i' )
          viCommandMode = 0;

        if ( ch==EOF || ch=='\n' || ch=='\r' || ch==CTR('O') ) {
            syEchoch('\r',fid);  syEchoch('\n',fid); break;
        }
	else
	  smartcomplete=0;

        /* now update the screen line according to the differences         */
        for ( q = line, r = new, newc = 0; *q != '\0'; ++q ) {
            if ( q == p )  newc = r-new;
            if ( *q==CTR('I') )  { do *r++=' '; while ((r-new+syNrchar)%8); }
            else if ( *q==0x7F ) { *r++ = '^'; *r++ = '?'; }
            else if ( '\0'<=*q && *q<' '  ) { *r++ = '^'; *r++ = *q+'@'; }
            else if ( ' ' <=*q && *q<0x7F ) { *r++ = *q; }
            else {
                *r++ = '\\';                   *r++ = '0'+(unsigned)*q/64%4;
                *r++ = '0'+(unsigned)*q/8 %8;  *r++ = '0'+(unsigned)*q   %8;
            }
            if ( r >= new+SyNrCols-syNrchar-2 ) {
                if ( q >= p ) { q++; break; }
                new[0] = '$';   new[1] = r[-5]; new[2] = r[-4];
                new[3] = r[-3]; new[4] = r[-2]; new[5] = r[-1];
                r = new+6;
            }
        }
        if ( q == p )  newc = r-new;
        for (      ; r < new+sizeof(new); ++r )  *r = ' ';
        if ( q[0] != '\0' && q[1] != '\0' )
            new[SyNrCols-syNrchar-2] = '$';
        else if ( q[1] == '\0' && ' ' <= *q && *q < 0x7F )
            new[SyNrCols-syNrchar-2] = *q;
        else if ( q[1] == '\0' && q[0] != '\0' )
            new[SyNrCols-syNrchar-2] = '$';
        for ( q = old, r = new; r < new+sizeof(new); ++r, ++q ) {
            if ( *q == *r )  continue;
            while (oldc<(q-old)) { syEchoch(old[oldc],fid);  ++oldc; }
            while (oldc>(q-old)) { syEchoch('\b',fid);       --oldc; }
            *q = *r;  syEchoch( *q, fid ); ++oldc;
        }
        while ( oldc < newc ) { syEchoch(old[oldc],fid);  ++oldc; }
        while ( oldc > newc ) { syEchoch('\b',fid);       --oldc; }

    }

    /* Now we put the new string into the history,  first all old strings  */
    /* are moved backwards,  then we enter the new string in syHistory[].  */
    if( forRecord ) { 
        char* pLine = (forRecord==2) ? buffer : line;
        int lineLen = (forRecord==2) ? strlen(buffer) : p-line;
        
        if (*pLine != '\n' && *pLine != '\0') { /* don't add empty lines */
	    for ( q = syHistory+sizeof(syHistory)-3; q >= syHistory+lineLen; --q )
	        *q = *(q-lineLen);
	    for ( p = pLine, q = syHistory; *p != '\0'; ++p, ++q )
	        *q = *p;
	    syHistory[sizeof(syHistory)-3] = '\n';
	    if ( syHi != syHistory )
	        syHi = syHi + lineLen;
	    if ( syHi > syHistory+sizeof(syHistory)-2 )
	        syHi = syHistory+sizeof(syHistory)-2;
        }
    }
    
    /* send the whole line (unclipped) to the window handler               */
    syWinPut( fid, (*line != '\0' ? "@r" : "@x"), line );

    /* strip away prompts (usefull for pasting old stuff)                  */
    if (line[0]=='g'&&line[1]=='a'&&line[2]=='p'&&line[3]=='>'&&line[4]==' ')
        for ( p = line, q = line+5; q[-1] != '\0'; p++, q++ )  *p = *q;
    if (line[0]=='b'&&line[1]=='r'&&line[2]=='k'&&line[3]=='>'&&line[4]==' ')
        for ( p = line, q = line+5; q[-1] != '\0'; p++, q++ )  *p = *q;
    if (line[0]=='>'&&line[1]==' ')
        for ( p = line, q = line+2; q[-1] != '\0'; p++, q++ )  *p = *q;

    /* switch back to cooked mode                                          */
    if ( syLineEdit == 1 )
        syStopraw(fid);

    /* start the clock again                                               */
    syStartTime += SyTime() - syStopTime;


    /* return the line (or '0' at end-of-file)                             */
    if ( *line == '\0' )
        return (char*)0;
    return line;
}


/****************************************************************************
**
*F  syStartraw(<fid>) . . . . . . . start raw mode on input file <fid>, local
*F  syStopraw(<fid>)  . . . . . . .  stop raw mode on input file <fid>, local
*F  syGetch(<fid>)  . . . . . . . . . . . . . .  get a char from <fid>, local
*F  syEchoch(<ch>,<fid>)  . . . . . . . . . . . . echo a char to <fid>, local
**
**  This four functions are the actual system dependent  part  of  'SyFgets'.
**
**  'syStartraw' tries to put the file with the file  identifier  <fid>  into
**  raw mode.  I.e.,  disabling  echo  and  any  buffering.  It also finds  a
**  place to put the echoing  for  'syEchoch'.  If  'syStartraw'  succedes it
**  returns 1, otherwise, e.g., if the <fid> is not a terminal, it returns 0.
**
**  'syStopraw' stops the raw mode for the file  <fid>  again,  switching  it
**  back into whatever mode the terminal had before 'syStartraw'.
**
**  'syGetch' reads one character from the file <fid>, which must  have  been
**  turned into raw mode before, and returns it.
**
**  'syEchoch' puts the character <ch> to the file opened by 'syStartraw' for
**  echoing.  Note that if the user redirected 'stdout' but not 'stdin',  the
**  echo for 'stdin' must go to 'ttyname(fileno(stdin))' instead of 'stdout'.
*/


/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
**
**  For Berkeley UNIX, input/output redirection and typeahead are  supported.
**  We switch the terminal line into 'CBREAK' mode and also disable the echo.
**  We do not switch to 'RAW'  mode because  this would flush  all typeahead.
**  Because 'CBREAK' leaves signals enabled we have to disable the characters
**  for interrupt and quit, which are usually set to '<ctr>-C' and '<ctr>-B'.
**  We also turn  off  the  xon/xoff  start and  stop characters,  which  are
**  usually set  to '<ctr>-S' and '<ctr>-Q' so  we can get  those characters.
**  We  do not  change the  suspend  character, which  is usually  '<ctr>-Z',
**  instead we catch the signal, so that we  can turn  the terminal line back
**  to cooked mode before stopping GAP and back to raw mode when continueing.
*/
#if SYS_BSD

#ifndef SYS_SGTTY_H                     /* terminal control functions      */
# include       <sgtty.h>
# define SYS_SGTTY_H
#endif

struct sgttyb   syOld, syNew;           /* old and new terminal state      */
struct tchars   syOldT, syNewT;         /* old and new special characters  */

#ifndef SYS_SIGNAL_H                    /* signal handling functions       */
# include       <signal.h>
# ifdef SYS_HAS_SIG_T
#  define SYS_SIG_T     SYS_HAS_SIG_T
# else
#  define SYS_SIG_T     void
# endif
# define SYS_SIGNAL_H
typedef SYS_SIG_T       sig_handler_t ( int );
#endif

#ifdef SIGTSTP

Int            syFid;

SYS_SIG_T       syAnswerCont (int signr )
{
    syStartraw( syFid );
    signal( SIGCONT, SIG_DFL );
    kill( getpid(), SIGCONT );
#ifdef SYS_HAS_SIG_T
    return 0;                           /* is ignored                      */
#endif
}

SYS_SIG_T       syAnswerTstp (int  signr )
{
    syStopraw( syFid );
    signal( SIGCONT, syAnswerCont );
    kill( getpid(), SIGTSTP );
#ifdef SYS_HAS_SIG_T
    return 0;                           /* is ignored                      */
#endif
}

#endif

int             syStartraw ( Int fid )
{
    /* if running under a window handler, tell it that we want to read     */
    if ( syWindow ) {
        if      ( fid == 0 ) { syWinPut( fid, "@i", "" );  return 1; }
        else if ( fid == 2 ) { syWinPut( fid, "@e", "" );  return 1; }
        else {                                             return 0; }
    }

    /* try to get the terminal attributes, will fail if not terminal       */
    if ( ioctl( fileno(syBuf[fid].fp), TIOCGETP, (char*)&syOld ) == -1 )
        return 0;

    /* disable interrupt, quit, start and stop output characters           */
    if ( ioctl( fileno(syBuf[fid].fp), TIOCGETC, (char*)&syOldT ) == -1 )
        return 0;
    syNewT = syOldT;
    syNewT.t_intrc  = -1;
    syNewT.t_quitc  = -1;
    /*C 27-Nov-90 martin changing '<ctr>S' and '<ctr>Q' does not work      */
    /*C syNewT.t_startc = -1;                                              */
    /*C syNewT.t_stopc  = -1;                                              */
    if ( ioctl( fileno(syBuf[fid].fp), TIOCSETC, (char*)&syNewT ) == -1 )
        return 0;

    /* disable input buffering, line editing and echo                      */
    syNew = syOld;
    syNew.sg_flags |= CBREAK;
    syNew.sg_flags &= ~ECHO;
    if ( ioctl( fileno(syBuf[fid].fp), TIOCSETN, (char*)&syNew ) == -1 )
        return 0;

#ifdef SIGTSTP
    /* install signal handler for stop                                     */
    syFid = fid;
    signal( SIGTSTP, syAnswerTstp );
#endif

    /* indicate success                                                    */
    return 1;
}

void            syStopraw ( Int fid )
{
    /* if running under a window handler, don't do nothing                 */
    if ( syWindow )
        return;

#ifdef SIGTSTP
    /* remove signal handler for stop                                      */
    signal( SIGTSTP, SIG_DFL );
#endif

    /* enable input buffering, line editing and echo again                 */
    if ( ioctl( fileno(syBuf[fid].fp), TIOCSETN, (char*)&syOld ) == -1 )
        fputs("gap: 'ioctl' could not turn off raw mode!\n",stderr);

    /* enable interrupt, quit, start and stop output characters again      */
    if ( ioctl( fileno(syBuf[fid].fp), TIOCSETC, (char*)&syOldT ) == -1 )
        fputs("gap: 'ioctl' could not turn off raw mode!\n",stderr);
}

int             syGetch ( Int fid )
{
    char                ch;

    /* read a character                                                    */
    while ( read( fileno(syBuf[fid].fp), &ch, 1 ) != 1 || ch == '\0' )
        ;

    /* if running under a window handler, handle special characters        */
    if ( syWindow && ch == '@' ) {
        do {
            while ( read(fileno(syBuf[fid].fp), &ch, 1) != 1 || ch == '\0' )
                ;
        } while ( ch < '@' || 'z' < ch );
        if ( ch == 'y' ) {
            syWinPut( fileno(syBuf[fid].echo), "@s", "" );
            ch = syGetch(fid);
        }
        else if ( 'A' <= ch && ch <= 'Z' )
            ch = CTR(ch);
    }
    
    /* return the character                                                */
    return ch;
}

void            syEchoch ( int ch, Int fid )
{
    char                ch2;

    /* write the character to the associate echo output device             */
    ch2 = ch;
    write( fileno(syBuf[fid].echo), (char*)&ch2, 1 );

    /* if running under a window handler, duplicate '@'                    */
    if ( syWindow && ch == '@' ) {
        ch2 = ch;
        write( fileno(syBuf[fid].echo), (char*)&ch2, 1 );
    }
}

void            syEchos ( char *str, Int fid )
{
    /* if running under a window handler, send the line to it              */
    if ( syWindow && fid < 4 )
        syWinPut( fid, (fid == 1 ? "@n" : "@f"), str );

    /* otherwise, write it to the associate echo output device             */
    else
        write( fileno(syBuf[fid].echo), str, strlen(str) );
}

#endif


/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
**
**  For UNIX System V, input/output redirection and typeahead are  supported.
**  We  turn off input buffering  and canonical input editing and  also echo.
**  Because we leave the signals enabled  we  have  to disable the characters
**  for interrupt and quit, which are usually set to '<ctr>-C' and '<ctr>-B'.
**  We   also turn off the  xon/xoff  start  and  stop  characters, which are
**  usually set to  '<ctr>-S'  and '<ctr>-Q' so we  can get those characters.
**  We do  not turn of  signals  'ISIG' because  we want   to catch  stop and
**  continue signals if this particular version  of UNIX supports them, so we
**  can turn the terminal line back to cooked mode before stopping GAP.
*/
#if SYS_USG

#ifndef SYS_TERMIO_H                    /* terminal control functions      */
# include       <termio.h>
# define SYS_TERMIO_H
#endif

struct termio   syOld, syNew;           /* old and new terminal state      */

#ifndef SYS_SIGNAL_H                    /* signal handling functions       */
# include       <signal.h>
# ifdef SYS_HAS_SIG_T
#  define SYS_SIG_T     SYS_HAS_SIG_T
# else
#  define SYS_SIG_T     void
# endif
# define SYS_SIGNAL_H
typedef SYS_SIG_T       sig_handler_t ( int );
#endif

#ifdef SIGTSTP

Int            syFid;

SYS_SIG_T       syAnswerCont ( int signr )
{
    syStartraw( syFid );
    signal( SIGCONT, SIG_DFL );
    kill( getpid(), SIGCONT );
#ifdef SYS_HAS_SIG_T
#if SYS_SIG_T != void
    return 0;                           /* is ignored                      */
#endif
#endif
}

SYS_SIG_T       syAnswerTstp ( int signr )
{
    syStopraw( syFid );
    signal( SIGCONT, syAnswerCont );
    kill( getpid(), SIGTSTP );
#ifdef SYS_HAS_SIG_T
#if SYS_SIG_T != void
    return 0;                           /* is ignored                      */
#endif
#endif
}

#endif

int             syStartraw ( Int fid )
{
    /* if running under a window handler, tell it that we want to read     */
    if ( syWindow ) {
        if      ( fid == 0 ) { syWinPut( fid, "@i", "" );  return 1; }
        else if ( fid == 2 ) { syWinPut( fid, "@e", "" );  return 1; }
        else {                                             return 0; }
    }

    /* try to get the terminal attributes, will fail if not terminal       */
    if ( ioctl( fileno(syBuf[fid].fp), TCGETA, &syOld ) == -1 )   return 0;

    /* disable interrupt, quit, start and stop output characters           */
    syNew = syOld;
    syNew.c_cc[VINTR] = 0377;
    syNew.c_cc[VQUIT] = 0377;
    /*C 27-Nov-90 martin changing '<ctr>S' and '<ctr>Q' does not work      */
    /*C syNew.c_iflag    &= ~(IXON|INLCR|ICRNL);                           */
    syNew.c_iflag    &= ~(INLCR|ICRNL);

    /* disable input buffering, line editing and echo                      */
    syNew.c_cc[VMIN]  = 1;
    syNew.c_cc[VTIME] = 0;
    syNew.c_lflag    &= ~(ECHO|ICANON);
    if ( ioctl( fileno(syBuf[fid].fp), TCSETAW, &syNew ) == -1 )  return 0;

#ifdef SIGTSTP
    /* install signal handler for stop                                     */
    syFid = fid;
    signal( SIGTSTP, syAnswerTstp );
#endif

    /* indicate success                                                    */
    return 1;
}

void            syStopraw ( Int fid )
{
    /* if running under a window handler, don't do nothing                 */
    if ( syWindow )
        return;

#ifdef SIGTSTP
    /* remove signal handler for stop                                      */
    signal( SIGTSTP, SIG_DFL );
#endif

    /* enable input buffering, line editing and echo again                 */
    if ( ioctl( fileno(syBuf[fid].fp), TCSETAW, &syOld ) == -1 )
        fputs("gap: 'ioctl' could not turn off raw mode!\n",stderr);
}

int             syGetch ( Int fid )
{
    char                ch;

    /* read a character                                                    */
    while ( read( fileno(syBuf[fid].fp), &ch, 1 ) != 1 || ch == '\0' )
        ;

    /* if running under a window handler, handle special characters        */
    if ( syWindow && ch == '@' ) {
        do {
            while ( read(fileno(syBuf[fid].fp), &ch, 1) != 1 || ch == '\0' )
                ;
        } while ( ch < '@' || 'z' < ch );
        if ( ch == 'y' ) {
            syWinPut( fileno(syBuf[fid].echo), "@s", "" );
            ch = syGetch(fid);
        }
        else if ( 'A' <= ch && ch <= 'Z' )
            ch = CTR(ch);
    }

    /* return the character                                                */
    return ch;
}

void            syEchoch ( int ch, Int fid )
{
    char                ch2;

    /* write the character to the associate echo output device             */
    ch2 = ch;
    write( fileno(syBuf[fid].echo), (char*)&ch2, 1 );

    /* if running under a window handler, duplicate '@'                    */
    if ( syWindow && ch == '@' ) {
        ch2 = ch;
        write( fileno(syBuf[fid].echo), (char*)&ch2, 1 );
    }
}

void            syEchos ( char *str, Int fid )
{
    /* if running under a window handler, send the line to it              */
    if ( syWindow && fid < 4 )
        syWinPut( fid, (fid == 1 ? "@n" : "@f"), str );

    /* otherwise, write it to the associate echo output device             */
    else
        write( fileno(syBuf[fid].echo), str, strlen(str) );
}

#endif


/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
**
**  For MS-DOS we read directly from the keyboard.
**  Note that the window handler is not currently supported.
*/
#if WIN32

#ifdef WIN32

#include <conio.h>
#include <io.h>

#undef read
#define read			_read
#undef isatty
#define isatty			_isatty
#undef write
#define write			_write

# define GETKEY()       getch()
# define PUTCHAR(C)     putchar(C)
# define KBHIT()        kbhit()
# define SYS_KBD_H
#endif

#ifndef SYS_KBD_H                       /* keyboard functions              */
# include       <pc.h>
# define GETKEY()       getkey()
# define PUTCHAR(C)     putchar(C)
# define KBHIT()        kbhit()
# define SYS_KBD_H
#endif

UInt   syStopout;              /* output is stopped by <ctr>-'S'  */

char            syTypeahead [256];      /* characters read by 'SyIsIntr'   */

char            syAltMap [35] = "QWERTYUIOP    ASDFGHJKL     ZXCVBNM";

int             syStartraw ( Int fid )
{
    /* check if the file is a terminal                                     */
    if ( ! isatty( fileno(syBuf[fid].fp) ) )
        return 0;

    /* indicate success                                                    */
    return 1;
}

void            syStopraw ( Int intfid )
{
}

int             syGetch ( Int fid )
{
    int                 ch;
#ifdef WIN32
    int ch2;
    #define K_LEFT		75
    #define K_RIGHT		77
    #define K_UP		72
    #define K_PAGEUP		73
    #define K_DOWN		80
    #define K_PAGEDOWN		81
    #define K_DEL		83
    #define K_HOME		71
    #define K_END		79

    #define K_CTRL_LEFT		115
    #define K_CTRL_RIGHT	116
    #define K_CTRL_UP		141
    #define K_CTRL_DOWN		145
    
    #define K_F1		59
    #define K_F2		60
    #define K_F3		61
    #define K_F4		62
    #define K_F5		63
    #define K_F6		64
    #define K_F7		65
    #define K_F8		66
    #define K_F9		67
    #define K_F10		68
    #define K_F11		133
    #define K_F12		134
    
    ch = GETKEY();
    ch2 = ch;

    /* handle function keys                                                */
    if (( ch == '\0' ) || ( ch == 0xe0 )) {
	ch = GETKEY();
/*	define to find the correct numbers for the not yet defined K_... 
		constants experimentally by pressing the keys	*/
	switch ( ch ) {
	    case K_LEFT:            ch2 = CTR('B');  break;
	    case K_RIGHT:           ch2 = CTR('F');  break;
	    case K_UP:
	    case K_PAGEUP:          ch2 = CTR('P');  break;
	    case K_DOWN:
	    case K_PAGEDOWN:        ch2 = CTR('N');  break;
	    case K_DEL:             ch2 = CTR('D');  break;
	    case K_HOME:            ch2 = CTR('A');  break;
	    case K_END:             ch2 = CTR('E');  break;
	    case K_CTRL_LEFT:	    ch2 = ESC('B');  break;
	    case K_CTRL_RIGHT:	    ch2 = ESC('F');  break;
	    case K_CTRL_UP:	    ch2 = ESC('P');  break;
	    case K_CTRL_DOWN:	    ch2 = ESC('N');  break;
	    case K_F8:		    ch2 = ESC(19);  break;
	    case K_F10:		    ch2 = ESC(21);  break;
	    case K_F11:		    ch2 = ESC(23);  break;
	}
    }
    return ch2;

#else

    /* if chars have been typed ahead and read by 'SyIsIntr' read them     */
    if ( syTypeahead[0] != '\0' ) {
        ch = syTypeahead[0];
        strcpy( syTypeahead, syTypeahead+1 );
    }

    /* otherwise read from the keyboard                                    */
    else {
        ch = GETKEY();
    }

    /* postprocess the character                                           */
    if ( 0x110 <= ch && ch <= 0x132 )   ch = ESC( syAltMap[ch-0x110] );
    else if ( ch == 0x147 )             ch = CTR('A');
    else if ( ch == 0x14f )             ch = CTR('E');
    else if ( ch == 0x148 )             ch = CTR('P');
    else if ( ch == 0x14b )             ch = CTR('B');
    else if ( ch == 0x14d )             ch = CTR('F');
    else if ( ch == 0x150 )             ch = CTR('N');
    else if ( ch == 0x153 )             ch = CTR('D');
    else                                ch &= 0xFF;

    /* return the character                                                */
    return ch;
#endif
}






void            syEchoch ( int ch, Int fid )
{
    PUTCHAR( ch );
}

void            syEchos ( char *str, Int fid )
{
    char *              s;

    /* handle stopped output                                               */
    while ( syStopout )  syStopout = (GETKEY() == CTR('S'));

    /* echo the string                                                     */
    for ( s = str; *s != '\0'; s++ )
        PUTCHAR( *s );
}

#endif


/****************************************************************************
**
*F  SyFputs( <line>, <fid> )  . . . . . . . .  write a line to the file <fid>
**
**  'SyFputs' is called to put the  <line>  to the file identified  by <fid>.
*/
#if SYS_BSD || SYS_USG

static Int my_syNrchar = 0; 

void            SyFputs (char line[], Int fid )
{
    Int                i;

    /* Joohoon new interface implementation */
    if( CURR_INTERFACE ){
      gap_interface[CURR_INTERFACE].write_callback(line,syBuf[fid].fp);
      interface_write_output_nolist(line);
      return;
    }

    /* if outputing to the terminal compute the cursor position and length */
    if ( fid == 1 || fid == 3 ) {
        syNrchar = 0;
        for ( i = 0; line[i] != '\0'; i++ ) {
            if ( line[i] == '\n' )  
				syNrchar = 0;
            else
				syPrompt[syNrchar++] = line[i];
        }
        syPrompt[syNrchar] = '\0';
		if (syNrchar > my_syNrchar) {
			/* track and optionally report the maximal value */
			if (SyMsgsFlagBags > 0)
				printf("SyFputs: High water mark for syNrchar = %d, syPrompt = \"%s\"\n",
					   syNrchar, syPrompt);
			my_syNrchar = syNrchar;
		}
    }

    /* otherwise compute only the length                                   */
    else {
        for ( i = 0; line[i] != '\0'; i++ )
            ;
    }

    /* if running under a window handler, send the line to it              */
    if ( syWindow && fid < 4 )
        syWinPut( fid, (fid == 1 ? "@n" : "@f"), line );

    /* otherwise, write it to the output file                              */
    else
        write( fileno(syBuf[fid].fp), line, i );
}

#endif

#if WIN32

#ifdef  WIN32_ANSICOLOR_EMU

DWORD   DefaultConsoleAttributes = 0;

void ANSIEscapeValueToColor(DWORD value, DWORD *mask, DWORD *flags)
{
    DWORD   color_table[8] = { 0, FOREGROUND_RED, FOREGROUND_GREEN, 
                        FOREGROUND_RED | FOREGROUND_GREEN, FOREGROUND_BLUE,
                        FOREGROUND_BLUE | FOREGROUND_RED, FOREGROUND_BLUE | FOREGROUND_GREEN,
                        FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE };

    if (value==0) {
        *flags = DefaultConsoleAttributes; *mask = 0xFF;
    } else if (value==1) {
        *flags |= FOREGROUND_INTENSITY | BACKGROUND_INTENSITY;
        //*mask |= FOREGROUND_INTENSITY | BACKGROUND_INTENSITY;
    } else if (value>=30 && value<=37) {
        *flags = (*flags & 0xFFFFFFF8) | color_table[value-30];
        *mask |= 0x0F;
    } else if (value == 39) {
        *flags = (*flags & 0xFFFFFFF0) | (DefaultConsoleAttributes & 0x0F);
        *mask |= 0x0F;
    } else if (value>=40 && value<=47) {
        *flags = (*flags & 0xFFFFFF8F) | (color_table[value-40] << 4);
        *mask |= 0x0F0;
    } else if (value == 49) {
        *flags = (*flags & 0xFFFFFF0F) | (DefaultConsoleAttributes & 0x0F0);
        *mask |= 0x0F0;
    } else if (value == 7) {
        *flags |= COMMON_LVB_REVERSE_VIDEO;
        *mask |= COMMON_LVB_REVERSE_VIDEO;
    } else if (value == 21) {
        *flags |= COMMON_LVB_UNDERSCORE;
        *mask |= COMMON_LVB_UNDERSCORE;
    } else if (value == 24) {
        *flags &= ~COMMON_LVB_UNDERSCORE;
        *mask |= COMMON_LVB_UNDERSCORE;
    }
}

#endif
void            SyFputs ( char line[], Int fid )
{

    /* Joohoon new interface implementation */
    if( CURR_INTERFACE ){
      gap_interface[CURR_INTERFACE].write_callback(line, syBuf[fid].fp);
      interface_write_output_nolist(line);
      return;
    }

    /* handle the console                                                  */
#ifndef WIN32
    if ( isatty( fileno(syBuf[fid].fp) ) ) 
	{
	    char *s;
	    Int i;

        /* test whether this is a line with a prompt                       */
        syNrchar = 0;
        for ( i = 0; line[i] != '\0'; i++ ) {
            if ( line[i] == '\n' )  syNrchar = 0;
            else                    syPrompt[syNrchar++] = line[i];
        }
        syPrompt[syNrchar] = '\0';

        /* handle stopped output                                           */
        while ( syStopout )  syStopout = (GETKEY() == CTR('S'));

        /* output the line                                                 */
        for ( s = line; *s != '\0'; s++ )
            PUTCHAR( *s );
    }

    /* ordinary file                                                       */
    else {
#endif
#ifdef  WIN32_ANSICOLOR_EMU
        /* emulate ANSI color escape sequences */
        if (syBuf[fid].fp == stdout || syBuf[fid].fp == stderr) {
            char    *c = line;
            char    *s = line;
            int     state = 0;
            int     value = 0;
            DWORD   flags = 0;
            DWORD   mask = 0;
            CONSOLE_SCREEN_BUFFER_INFO  bf;
            while (*c) {
                if (*c==0x1B && state==0) {
                    state = 1;
                } else
                if (*c==0x5B && state==1) {
                    flags = 0;
                    mask = 0;
                    value = 0;
                    state = 2;
                } else
                if (state>=2) { // reading escape sequence
                    if (*c=='m') { // end of escape sequence
                        ANSIEscapeValueToColor(value, &mask, &flags);
                        // print line
                        *(c-state) = 0;
                        fputs( s, syBuf[fid].fp );
                        *(c-state) = 0x1B;
                        s = c+1;
                        // assign new text attributes
                        bf.wAttributes = 0;
                        GetConsoleScreenBufferInfo(GetStdHandle(syBuf[fid].fp==stderr ? STD_ERROR_HANDLE: STD_OUTPUT_HANDLE), &bf);
                        if (DefaultConsoleAttributes==0) DefaultConsoleAttributes = bf.wAttributes;
                        SetConsoleTextAttribute(GetStdHandle(syBuf[fid].fp==stderr ? STD_ERROR_HANDLE: STD_OUTPUT_HANDLE), 
                            bf.wAttributes & ~mask | flags & mask);
                        state = 0;
                    } else
                    if (*c==';') { // got some value, modify flags
                        ANSIEscapeValueToColor(value, &mask, &flags);
                        value = 0;
                        state++;
                    } else
                    if (*c-'0'<=9) { // reading decimal number
                        value = value*10+(*c-'0');
                        state++;
                    } else { // error
                        state = 0;
                    }
                } else
                    state = 0;
                c++;
            }
            if (s != c) // print remaining characters
                fputs(s, syBuf[fid].fp );
        } else
#endif
        fputs( line, syBuf[fid].fp );
   		fflush( syBuf[fid].fp );		// typically the GAP internal output buffer has just been flushed, so flush file buffer, too
                                        // otherwise piped output gets delayed
#ifndef WIN32
    }
#endif

}

#endif



/****************************************************************************
**
*F  syWinPut(<fid>,<cmd>,<str>) . . . . . . send a line to the window handler
**
**  'syWinPut'  send the command   <cmd> and the  string  <str> to the window
**  handler associated with the  file identifier <fid>.   In the string <str>
**  '@'  characters are duplicated, and   control characters are converted to
**  '@<chr>', e.g., <newline> is converted to '@J'.
*/

void            syWinPut (Int fid, char *cmd, char *str)
{
    Int                fd;             /* file descriptor                 */
    char                tmp [130];      /* temporary buffer                */
    char *              s;              /* pointer into the string         */
    char *              t;              /* pointer into the temporary      */

    /* if not running under a window handler, don't do nothing             */
    if ( ! syWindow || 4 <= fid )
        return;

    /* get the file descriptor                                             */
    if ( fid == 0 || fid == 2 )  fd = fileno(syBuf[fid].echo);
    else                         fd = fileno(syBuf[fid].fp);

    /* print the cmd                                                       */
    write( fd, cmd, strlen(cmd) );

    /* print the output line, duplicate '@' and handle <ctr>-<chr>         */
    s = str;  t = tmp;
    while ( *s != '\0' ) {
        if ( *s == '@' ) {
            *t++ = '@';  *t++ = *s++;
        }
        else if ( CTR('A') <= *s && *s <= CTR('Z') ) {
            *t++ = '@';  *t++ = *s++ - CTR('A') + 'A';
        }
        else {
            *t++ = *s++;
        }
        if ( 128 <= t-tmp ) {
            write( fd, tmp, t-tmp );
            t = tmp;
        }
    }
    if ( 0 < t-tmp ) {
        write( fd, tmp, t-tmp );
    }
}


/****************************************************************************
**
*F  SyPinfo( <nr>, <size> ) . . . . . . . . . . . . . . .  print garbage info
**
**  'SyPinfo' is called from  Gasman to inform the  window handler  about the
**  current  Gasman   statistics.  <nr> determines   the   phase the  garbage
**  collection is currently  in, and <size>  is the correspoding value, e.g.,
**  number of live bags.
*/
void            SyPinfo (int nr, Int size)
{
    char                cmd [3];
    char                buf [16];
    char *              b;

    /* set up the command                                                  */
    cmd[0] = '@';
    cmd[1] = nr + '0';
    cmd[2] = '\0';

    /* stringify the argument                                              */
    b = buf;
    while ( 0 < size ) {
        *b++ = (size % 10) + '0';
        size /= 10;
    }
    *b++ = '+';
    *b = '\0';

    /* send it to the window handler                                       */
    syWinPut( 1, cmd, buf );
}


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

char            WinCmdBuffer [8000];

char *          SyWinCmd (char *str, Int len)
{
    char                buf [130];      /* temporary buffer                */
    char *              s;              /* pointer into the string         */
    char *              b;              /* pointer into the temporary      */
    UInt       i;              /* loop variable                   */

    /* if not running under a window handler, don't do nothing             */
    if ( ! syWindow )
        return "I1+S52000000No Window Handler Present";

    /* compute the length of the (expanded) string (and ignore argument)   */
    len = 0;
    for ( s = str; *s != '\0'; s++ )
        len += 1 + (*s == '@' || (CTR('A') <= *s && *s <= CTR('Z')));

    /* send the length to the window handler                               */
    b = buf;
    for ( i = 0; i < 8; i++ ) {
        *b++ = (len % 10) + '0';
        len /= 10;
    }
    *b = '\0';
    syWinPut( 1, "@w", buf );

    /* send the string to the window handler                               */
    syWinPut( 1, "", str );

    /* read the length of the answer                                       */
    s = WinCmdBuffer;
    i = 10;
    do {
        while ( 0 < i ) {
            len = read( 0, s, i );
            i  -= len;
            s  += len;
        }
        if ( WinCmdBuffer[0] == '@' && WinCmdBuffer[1] == 'y' ) {
            for ( i = 2;  i < 10;  i++ )
                WinCmdBuffer[i-2] = WinCmdBuffer[i];
            s -= 2;
            i  = 2;
        }
    } while ( 0 < i );
    if ( WinCmdBuffer[0] != '@' || WinCmdBuffer[1] != 'a' )
        return "I1+S41000000Illegal Answer";
    for ( len = 0, i = 9;  1 < i;  i-- )
        len = len*10 + (WinCmdBuffer[i]-'0');

    /* read the arguments of the answer                                    */
    s = WinCmdBuffer;
    i = len;
    while ( 0 < i ) {
        len = read( 0, s, i );
        i  -= len;
        s  += len;
    }

    /* shrink '@@' into '@'                                                */
    for ( b = s = WinCmdBuffer;  0 < len;  len-- ) {
        if ( *s == '@' ) {
            s++;
            if ( *s == '@' )
                *b++ = '@';
            else if ( 'A' <= *s && *s <= 'Z' )
                *b++ = CTR(*s);
            s++;
        }
        else {
            *b++ = *s++;
        }
    }
    *b = 0;

    /* return the string                                                   */
    return WinCmdBuffer;
}


/****************************************************************************
**
*F  SyIsIntr()  . . . . . . . . . . . . . . . . check wether user hit <ctr>-C
**
**  'SyIsIntr' is called from the evaluator at  regular  intervals  to  check
**  wether the user hit '<ctr>-C' to interrupt a computation.
**
**  'SyIsIntr' returns 1 if the user typed '<ctr>-C' and 0 otherwise.
*/


/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
**
**  For  UNIX, OS/2  we  install 'syAnswerIntr' to  answer interrupt
**  'SIGINT'.   If two interrupts  occur within 1 second 'syAnswerIntr' exits
**  GAP.
*/
#if SYS_BSD || SYS_USG  || WIN32

#ifndef SYS_SIGNAL_H                    /* signal handling functions       */
# include       <signal.h>
# ifdef SYS_HAS_SIG_T
#  define SYS_SIG_T     SYS_HAS_SIG_T
# else
#  define SYS_SIG_T     void
# endif
# define SYS_SIGNAL_H
typedef SYS_SIG_T       sig_handler_t ( int );
#endif

#ifndef SYS_TIME_H                      /* time functions                  */
# include       <time.h>
# define SYS_TIME_H
#endif
#ifndef SYS_HAS_TIME_PROTO              /* ANSI/TRAD decl. from H&S 18.1    */
# if SYS_ANSI
extern  time_t          time ( time_t * buf );
# else
extern  Int            time ( Int * buf );
# endif
#endif

UInt   syLastIntr;             /* time of the last interrupt      */

SYS_SIG_T       syAnswerIntr(int signr)
{
    UInt       nowIntr;

    /* get the current wall clock time                                     */
    nowIntr = time(0);

    /* if the last '<ctr>-C' was less than a second ago, exit GAP          */
    if ( syLastIntr && nowIntr-syLastIntr < 1 ) {
        fputs("gap: you hit '<ctr>-C' twice in a second, goodbye.\n",stderr);
        SyExit( 1 );
    }

    /* remember time of this interrupt                                     */
    syLastIntr = nowIntr;

    /* reinstall 'syAnswerIntr' as signal handler                          */

    signal( SIGINT, syAnswerIntr );

#ifdef SYS_HAS_SIG_T
#if SYS_SIG_T != void
    return 0;                           /* is ignored                      */
#endif
#endif
}

inline Int            SyIsIntr ()
{
    Int        isIntr;

    isIntr = (syLastIntr != 0);
    syLastIntr = 0;
    return isIntr;
}

#endif


/****************************************************************************
**
*F  SyExit( <ret> ) . . . . . . . . . . . . . exit GAP with return code <ret>
**
**  'SyExit' is the offical  way  to  exit GAP, bus errors are the inoffical.
**  The function 'SyExit' must perform all the neccessary cleanup operations.
**  If ret is 0 'SyExit' should signal to a calling proccess that all is  ok.
**  If ret is 1 'SyExit' should signal a  failure  to  the  calling proccess.
*/


void            SyExit (Int ret)
{
    exit( (int)ret );
}


/****************************************************************************
**
*F  SyExec( <cmd> ) . . . . . . . . . . . execute command in operating system
**
**  'SyExec' executes the command <cmd> (a string) in the operating system.
**
**  'SyExec'  should call a command  interpreter  to execute the command,  so
**  that file name expansion and other common  actions take place.  If the OS
**  does not support this 'SyExec' should print a message and return.
**
**  For UNIX we can use 'system', which does exactly what we want.
*/


#ifndef WIN32
#include <sys/wait.h>
#endif


int             SyExec (char *cmd)
{
    int status,result;

    syWinPut( 0, "@z", "" );
    status = system( cmd );
    syWinPut( 0, "@mAgIc", "" );
#ifndef WIN32 // strange non-windows stuff.
	result = WEXITSTATUS(status);
    if (WIFSIGNALED(status)) 
	result |= WTERMSIG(status) << 8;
#else
	result = status;
#endif
	return result;
}



/****************************************************************************
**
*F  SyTime()  . . . . . . . . . . . . . . . return time spent in milliseconds
**
**  'SyTime' returns the number of milliseconds spent by GAP so far.
**
**  Should be as accurate as possible,  because it  is  used  for  profiling.
*/


/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
**
**  For Berkeley UNIX the clock ticks in 1/60.  On some (all?) BSD systems we
**  can use 'getrusage', which gives us a much better resolution.
*/
#if SYS_BSD
#ifndef SYS_HAS_NO_GETRUSAGE

#ifndef SYS_RESOURCE_H                  /* definition of 'struct rusage'   */
# include       <sys/time.h>            /* definition of 'struct timeval'  */
# include       <sys/resource.h>
# define SYS_RESOURCE_H
#endif
#ifndef SYS_HAS_TIME_PROTO              /* UNIX decl. from 'man'           */
extern  int             getrusage ( int, struct rusage * );
#endif

UInt   SyTime ()
{
    struct rusage       buf;

    if ( getrusage( RUSAGE_SELF, &buf ) ) {
        fputs("gap: panic 'SyTime' cannot get time!\n",stderr);
        SyExit( 1 );
    }
    return buf.ru_utime.tv_sec*1000 + buf.ru_utime.tv_usec/1000 -syStartTime;
}

#endif

#ifdef SYS_HAS_NO_GETRUSAGE

#ifndef SYS_TIMES_H                     /* time functions                  */
# include       <sys/types.h>
# include       <sys/times.h>
# define SYS_TIMES_H
#endif
#ifndef SYS_HAS_TIME_PROTO              /* UNIX decl. from 'man'           */
extern  int             times ( struct tms * );
#endif

UInt   SyTime ()
{
    struct tms          tbuf;

    if ( times( &tbuf ) == -1 ) {
        fputs("gap: panic 'SyTime' cannot get time!\n",stderr);
        SyExit( 1 );
    }
    return 100 * tbuf.tms_utime / (60/10) - syStartTime;
}

#endif

#endif


/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
**
**  For UNIX System V and OS/2 the clock ticks in 1/HZ,  this is usually 1/60
**  or 1/100.
*/
#if SYS_USG

#ifndef SYS_TIMES_H                     /* time functions                  */
# include       <sys/param.h>           /* definition of 'HZ'              */
# include       <sys/types.h>
# include       <sys/times.h>
# define SYS_TIMES_H
#endif
#ifndef SYS_HAS_TIME_PROTO              /* UNIX decl. from 'man'           */
extern  int             times ( struct tms * );
#endif
#include <errno.h>
UInt   SyTime ()
{
    struct tms          tbuf;
    if ( times( &tbuf ) == -1 ) {
        perror("gap: panic 'SyTime' cannot get time");
        SyExit( 1 );
    }
    return 100 * tbuf.tms_utime / (HZ / 10) - syStartTime;
}

#endif


/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
**
**  For Windows we use the function 'clock' and allow to stop the clock.
*/
#if WIN32

#ifndef SYS_TIME_H                      /* time functions                  */
# include       <time.h>
# define SYS_TIME_H
#endif
#ifndef SYS_HAS_TIME_PROTO              /* ANSI/TRAD decl. from H&S 18.2    */
# if SYS_ANSI
extern  clock_t         clock ( void );
# define SYS_CLOCKS     CLOCKS_PER_SEC
# else
extern  Int            clock ( void );
#   define SYS_CLOCKS   100
# endif
#endif

#ifdef WIN32
#define SYS_CLOCKS     CLOCKS_PER_SEC
#endif

UInt   SyTime ()
{
    return 100 * (UInt)clock() / (SYS_CLOCKS/10) - syStartTime;
}

#endif


/****************************************************************************
**
*F  SyTmpname() . . . . . . . . . . . . . . . . . return a temporary filename
**
**  'SyTmpname' creates and returns a new temporary name.
*/
#ifndef SYS_STDIO_H                     /* standard input/output functions */
# include       <stdio.h>
# define SYS_STDIO_H
#endif
#ifndef SYS_HAS_MISC_PROTO              /* ANSI/TRAD decl. from H&S 15.16  */
extern  char *          tmpnam ( char * );
#endif

char *          SyTmpname (void)
{
#ifdef WIN32
    return GuSysTmpname(config_demand_val("tmp_dir")->strval, 
		config_demand_val("path_sep")->strval, 
		"gap_XXXXXX");
#else
#ifdef HAVE_MKSTEMP
    static char * result = NULL;
    static int len = 0;
    int fd;
    if(result==NULL) {
		char * tmp_dir = config_demand_val("tmp_dir")->strval;
		char * path_sep = config_demand_val("path_sep")->strval;
		result = GuMakeMessage("%s%sgap_XXXXXX", tmp_dir, path_sep);
		len = strlen(result);
    }
    result[len-1] = 'X'; result[len-2] = 'X';  result[len-3] = 'X';
    result[len-4] = 'X'; result[len-5] = 'X';  result[len-6] = 'X';

    fd = mkstemp(result);
    if (fd == -1)
		return NULL;
    else {
		/* we are required to generate the name, but mkstemp actually      */
		/* creates an empty file                                           */
		unlink(result); 
		close(fd); 
		return result;
    }
#else
    return tmpnam( (char*)0 );
#endif
#endif
}


/****************************************************************************
**
*F  SyHelp( <topic>, <fid> )  . . . . . . . . . . . . . . display online help
**
**  This function is of course way to large.  But what the  heck,  it  works.
*/
char            syChapnames [128][16];

char            syLastTopics [16] [64] = { "Welcome to GAP" };

short           syLastIndex = 0;

void            SyHelp (char *topic, Int fin)
                                        /* topic for which help is sought  */
                                        /* file id of input and output     */
{
    char                filename [2048];/* filename of various files       */
    Int                fid;            /* file identifier of various files*/
    char                line [256];     /* single line from those files    */
    UInt       chapnr;         /* number of the chapter           */
    char                chapname [64];  /* name of the chapter             */
    UInt       secnr;          /* number of the section           */
    char                secname [1024]; /* name of the section             */
    char                secline [128];  /* '\Section <secname>'            */
    Int                match;          /* does the section match topic    */
    Int                matches;        /* how many sections matched       */
    char                last [256];     /* last line from table of contents*/
    char                last2 [256];    /* last chapter line from toc      */
    Int                offset;         /* '<' is -1, '>' is 1             */
    char                ch;             /* char read after '-- <space> --' */
    Int                spaces;         /* spaces to be inserted for just  */
    char                status;         /* 'a', '$', '|', or '#'           */
    char                * p, * q, * r;  /* loop variables                  */
    UInt       i, j;           /* loop variables                  */
    UInt       raw;            /* is input in raw mode?           */

    /* try to switch the input into raw mode                               */
    raw = (syLineEdit == 1 && syStartraw( fin ));

    /* inform the window handler                                           */
    syWinPut( fin, "@h", "" );

    /* set 'SyHelpname' to 'SyLibname' with 'lib' replaced by 'doc'        */
    if ( SyHelpname[0] == '\0' ) {
        q = SyHelpname;
        p = SyLibname;
        while ( *p != '\0' )  *q++ = *p++;
        *q = '\0';
        for ( p = SyHelpname; *p != '\0'; p++ ) ;
        while ( SyHelpname < p && (p[0]!='l' || p[1]!='i' || p[2]!='b') )
            p--;
        p[0] = 'd'; p[1] = 'o'; p[2] = 'c';
    }

    /* skip leading blanks in the topic                                    */
    while ( *topic == ' ' )  topic++;

    /* if the topic is empty take the last one again                       */
    if ( topic[0] == '\0' ) {
        topic = syLastTopics[ syLastIndex ];
    }

    /* if the topic is '<' we are interested in the one before 'LastTopic' */
    offset = 0;
    last[0] = '\0';
    if ( strcmp( topic, "<" ) == 0 ) {
        topic = syLastTopics[ syLastIndex ];
        offset = -1;
    }

    /* if the topic is '>' we are interested in the one after 'LastTopic'  */
    if ( strcmp( topic, ">" ) == 0 ) {
        topic = syLastTopics[ syLastIndex ];
        offset = 1;
    }

    /* if the topic is '<<' we are interested in the first section         */
    last2[0] = '\0';
    if ( strcmp( topic, "<<" ) == 0 ) {
        topic = syLastTopics[ syLastIndex ];
        offset = -2;
    }

    /* if the topic is '>>' we are interested in the next chapter          */
    if ( strcmp( topic, ">>" ) == 0 ) {
        topic = syLastTopics[ syLastIndex ];
        offset = 2;
    }

    /* if the topic is '-' we are interested in the previous section again */
    if ( topic[0] == '-' ) {
        while ( *topic++ == '-' )
            syLastIndex = (syLastIndex + 15) % 16;
        topic = syLastTopics[ syLastIndex ];
        if ( topic[0] == '\0' ) {
            syEchos( "Help: this section has no previous section\n", fin );
            syLastIndex = (syLastIndex + 1) % 16;
            if ( raw )  syStopraw( fin );
            return;
        }
        syLastIndex = (syLastIndex + 15) % 16;
    }

    /* if the topic is '+' we are interested in the last section again     */
    if ( topic[0] == '+' ) {
        while ( *topic++ == '+' )
            syLastIndex = (syLastIndex + 1) % 16;
        topic = syLastTopics[ syLastIndex ];
        if ( topic[0] == '\0' ) {
            syEchos( "Help: this section has no previous section\n", fin );
            syLastIndex = (syLastIndex + 15) % 16;
            if ( raw )  syStopraw( fin );
            return;
        }
        syLastIndex = (syLastIndex + 15) % 16;
    }

    /* if the subject is 'Welcome to GAP' display a welcome message        */
    if ( strcmp( topic, "Welcome to GAP" ) == 0 ) {

        syEchos( "    Welcome to GAP ______________________________", fin );
        syEchos( "_____________ Welcome to GAP\n",                    fin );
        syEchos( "\n",                                                fin );
        syEchos( "    Welcome to GAP.\n",                             fin );
        syEchos( "\n",                                                fin );
        syEchos( "    GAP is a system for computational group theor", fin );
        syEchos( "y.\n",                                              fin );
        syEchos( "\n",                                                fin );
        syEchos( "    Enter '?About GAP'    for a step by step intr", fin );
        syEchos( "oduction to GAP.\n",                                fin );
        syEchos( "    Enter '?Help'         for information how to ", fin );
        syEchos( "use the GAP help system.\n",                        fin );
        syEchos( "    Enter '?Chapters'     for a list of the chapt", fin );
        syEchos( "ers of the GAP help system.\n",                     fin );
        syEchos( "    Enter '?Copyright'    for the terms under whi", fin );
        syEchos( "ch you can use and copy GAP.\n",                    fin );
        syEchos( "\n",                                                fin );
        syEchos( "    In each case do *not* enter the single quotes", fin );
        syEchos( "(') , they are  used in help\n",                    fin );
        syEchos( "    sections only to delimit text that you actual", fin );
        syEchos( "ly enter.\n",                                       fin );
        syEchos( "\n",                                                fin );

        /* remember this topic for the next time                           */
        p = "Welcome to GAP";
        syLastIndex = (syLastIndex + 1) % 16;
        q = syLastTopics[ syLastIndex ];
        while ( *p != '\0' )  *q++ = *p++;
        *q = '\0';

        if ( raw )  syStopraw( fin );
        return;

    }

    /* if the topic is 'chapter' display the table of chapters             */
    if ( strcmp(topic,"chapters")==0 || strcmp(topic,"Chapters")==0 ) {

        /* open the table of contents file                                 */
        filename[0] = '\0';
        strncat( filename, SyHelpname, sizeof(filename)-12 );
        strncat( filename, "online.toc", 11 );
        fid = SyFopen( filename, "r" );
        if ( fid == -1 ) {
            syEchos( "Help: cannot open the table of contents file '",fin );
            syEchos( filename, fin );
            syEchos( "'\n", fin );
            syEchos( "maybe use the option '-h <hlpname>'?\n", fin );
            if ( raw )  syStopraw( fin );
            return;
        }

        /* print the header line                                           */
        syEchos( "    Table of Chapters _________________", fin );
        syEchos( "____________________ Table of Contents\n", fin );

        /* scan the table of contents for chapter lines                    */
        offset = 2;
        while ( SyFgets( line, sizeof(line), fid ) ) {

            /* parse table of contents line                                */
            for ( p = line; *p != '\0' && ! IsDigit(*p); p++ )  ;
            for ( i = 0; IsDigit(*p); p++ )  i = 10*i+*p-'0';
            if ( *p == '.' )  p++;
            for ( j = 0; IsDigit(*p); p++ )  j = 10*j+*p-'0';
            if ( *p == '}' )  p++;
            if ( i == 0 || ! IsAlpha(*p) ) {
              syEchos("Help: contentsline is garbage in 'online.toc'",fin);
              SyFclose( fid );
              if ( raw )  syStopraw( fin );
              return;
            }

            /* skip nonchapter lines                                       */
            if ( j != 0 )  continue;

            /* stop every 24 lines                                         */
            if ( offset == SyNrRows && raw ) {
              syEchos( "    -- <space> for more --", fin );
              ch = syGetch( fin );
              syEchos("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b",
                      fin);
              syEchos( "                          ", fin );
              syEchos("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b",
                      fin);
              if ( ch == 'q' )  {
                  syEchos( "\n", fin );
                  break;
              }
              else if ( ch == '\n' || ch == '\r' ) {
                  offset = SyNrRows - 1;
              }
              else {
                  offset = 2;
              }
            }

            /* display the line                                            */
            q = line;
            while ( *p != '}' )  *q++ = *p++;
            *q++ = '\n';
            *q = '\0';
            syEchos( "    ", fin );
            syEchos( line, fin );
            offset++;

        }

        /* remember this topic for the next time                           */
        p = "Chapters";
        syLastIndex = (syLastIndex + 1) % 16;
        q = syLastTopics[ syLastIndex ];
        while ( *p != '\0' )  *q++ = *p++;
        *q = '\0';

        SyFclose( fid );
        if ( raw )  syStopraw( fin );
        return;
    }

    /* if the topic is 'sections' display the table of sections            */
    if ( strcmp(topic,"sections")==0 || strcmp(topic,"Sections")==0 ) {

        /* open the table of contents file                                 */
        filename[0] = '\0';
        strncat( filename, SyHelpname, sizeof(filename)-12 );
        strncat( filename, "online.toc", 11 );
        fid = SyFopen( filename, "r" );
        if ( fid == -1 ) {
            syEchos( "Help: cannot open the table of contents file '",fin);
            syEchos( filename, fin );
            syEchos( "'\n", fin );
            syEchos( "maybe use the option '-h <hlpname>'?\n", fin );
            if ( raw )  syStopraw( fin );
            return;
        }

        /* print the header line                                           */
        syEchos( "    Table of Sections _________________", fin );
        syEchos( "____________________ Table of Contents\n", fin );

        /* scan the table of contents for chapter lines                    */
        offset = 2;
        while ( SyFgets( line, sizeof(line), fid ) ) {

            /* parse table of contents line                                */
            for ( p = line; *p != '\0' && ! IsDigit(*p); p++ )  ;
            for ( i = 0; IsDigit(*p); p++ )  i = 10*i+*p-'0';
            if ( *p == '.' )  p++;
            for ( j = 0; IsDigit(*p); p++ )  j = 10*j+*p-'0';
            if ( *p == '}' )  p++;
            if ( i == 0 || ! IsAlpha(*p) ) {
              syEchos("Help: contentsline is garbage in 'online.toc'",fin);
              SyFclose( fid );
              if ( raw )  syStopraw( fin );
              return;
            }

            /* stop every 24 lines                                         */
            if ( offset == SyNrRows && raw ) {
              syEchos( "    -- <space> for more --", fin );
              ch = syGetch( fin );
              syEchos("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b",
                      fin);
              syEchos( "                          ", fin );
              syEchos("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b",
                      fin);
              if ( ch == 'q' )  {
                  syEchos( "\n", fin );
                  break;
              }
              else if ( ch == '\n' || ch == '\r' ) {
                  offset = SyNrRows - 1;
              }
              else {
                  offset = 2;
              }
            }

            /* display the line                                            */
            q = line;
            while ( *p != '}' )  *q++ = *p++;
            *q++ = '\n';
            *q = '\0';
            if ( j == 0 )  syEchos( "    ", fin );
            else            syEchos( "        ", fin );
            syEchos( line, fin );
            offset++;

        }

        /* remember this topic for the next time                           */
        p = "Sections";
        syLastIndex = (syLastIndex + 1) % 16;
        q = syLastTopics[ syLastIndex ];
        while ( *p != '\0' )  *q++ = *p++;
        *q = '\0';

        SyFclose( fid );
        if ( raw )  syStopraw( fin );
        return;
    }

    /* if the topic is 'Copyright' print the copyright                     */
    if ( strcmp(topic,"copyright")==0 || strcmp(topic,"Copyright")==0 ) {

        /* open the copyright file                                         */
        filename[0] = '\0';
        strncat( filename, SyHelpname, sizeof(filename)-14 );
        strncat( filename, "copyrigh.tex", 13 );
        fid = SyFopen( filename, "r" );
        if ( fid == -1 ) {
            syEchos( "Help: cannot open the copyright file '",fin);
            syEchos( filename, fin );
            syEchos( "'\n", fin );
            syEchos( "maybe use the option '-h <helpname>'?\n", fin );
            if ( raw )  syStopraw( fin );
            return;
        }

        /* print the header line                                           */
        syEchos( "    Copyright _________________________", fin );
        syEchos( "____________________________ Copyright\n", fin );

        /* print the contents of the file                                  */
        offset = 2;
        while ( SyFgets( line, sizeof(line), fid ) ) {

            /* skip lines that begin with a '%'                            */
            if ( line[0] == '%' )  continue;

            /* skip the line that begins with '\thispagestyle'             */
            p = line;
            q = "\\thispagestyle";
            while ( *p == *q ) { p++; q++; }
            if ( *q == '\0' )  continue;

            /* stop every 24 lines                                         */
            if ( offset == SyNrRows && raw ) {
              syEchos( "    -- <space> for more --", fin );
              ch = syGetch( fin );
              syEchos("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b",
                      fin);
              syEchos( "                          ", fin );
              syEchos("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b",
                      fin);
              if ( ch == 'q' )  {
                  syEchos( "\n", fin );
                  break;
              }
              else if ( ch == '\n' || ch == '\r' ) {
                  offset = SyNrRows - 1;
              }
              else {
                  offset = 2;
              }
            }

            /* fixup the copyright line                                    */
            p = line;
            q = "{\\large";
            while ( *p == *q ) { p++; q++; }
            if ( *q == '\0' ) {
                syEchos( "    Copyright (c) 1992 ", fin );
                syEchos( "by Lehrstuhl D fuer Mathematik\n", fin );
                continue;
            }

            /* display the line                                            */
            p = line;
            q = last;
            spaces = 0;
            while ( *p != '\0' ) {
                if ( *p == '\\' || *p == '{' || *p == '}' ) {
                    if ( last < q && q[-1] == ' ' )
                        *q++ = ' ';
                    else
                        spaces++;
                }
                else if ( *p == ' ' ) {
                    *q++ = ' ';
                    while ( 0 < spaces ) {
                        *q++ = ' ';
                        spaces--;
                    }
                }
                else {
                    *q++ = *p;
                }
                p++;
            }
            *q = '\0';
            syEchos( "    ", fin );  syEchos( last, fin );
            offset++;
        }

        /* remember this topic for the next time                           */
        p = "Copyright";
        syLastIndex = (syLastIndex + 1) % 16;
        q = syLastTopics[ syLastIndex ];
        while ( *p != '\0' )  *q++ = *p++;
        *q = '\0';

        SyFclose( fid );
        if ( raw )  syStopraw( fin );
        return;
    }

    /* if the topic is '?<string>' search the index                        */
    if ( topic[0] == '?' ) {

        /* skip leading blanks in the topic                                */
        topic++;
        while ( *topic == ' ' )  topic++;

        /* open the index                                                  */
        filename[0] = '\0';
        strncat( filename, SyHelpname, sizeof(filename)-12 );
        strncat( filename, "online.idx", 11 );
        fid = SyFopen( filename, "r" );
        if ( fid == -1 ) {
            syEchos( "Help: cannot open the index file '", fin);
            syEchos( filename, fin );
            syEchos( "'\n", fin );
            syEchos( "maybe use the option '-h <hlpname>'?\n", fin );
            if ( raw )  syStopraw( fin );
            return;
        }

        /* make a header line                                              */
        line[0] = '\0';
        strncat( line, topic, 40 );
        strncat( line,
        " _________________________________________________________________",
                  73 - 5 );
        line[72-5] = ' ';
        line[73-5] = '\0';
        strncat( line, "Index", 6 );
        strncat( line, "\n", 2 );
        syEchos( "    ", fin );
        syEchos( line, fin );

        /* scan the index                                                  */
        offset = 2;
        while ( SyFgets( line, sizeof(line), fid ) ) {

            /* a '%' line tells us that the next entry is a section name   */
            if ( line[0] == '%' ) {
                while ( line[0] == '%' ) {
                    if ( ! SyFgets( line, sizeof(line), fid ) ) {
                        syEchos( "Help: index file is garbage\n", fin );
                        SyFclose( fid );
                        if ( raw )  syStopraw( fin );
                        return;
                    }
                }
                q = secname;
                p = line + 12;
                while ( *p != '}' )  *q++ = *p++;
                *q = '\0';
            }

            /* skip this entry if we alread had an entry for this section  */
            if ( secname[0] == '\0' )  continue;

            /* try to match topic against this index entry                 */
            for ( r = line + 12; *r != '\0'; r++ ) {
                p = topic;
                q = r;
                while ( (*p | 0x20) == (*q | 0x20) ) { p++; q++; }
                if ( *p == '\0' )  break;
            }
            if ( *r == '\0' )  continue;

            /* stop every 24 lines                                         */
            if ( offset == SyNrRows && raw ) {
              syEchos( "    -- <space> for more --", fin );
              ch = syGetch( fin );
              syEchos("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b",
                      fin);
              syEchos( "                          ", fin );
              syEchos("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b",
                      fin);
              if ( ch == 'q' )  {
                  syEchos( "\n", fin );
                  break;
              }
              else if ( ch == '\n' || ch == '\r' ) {
                  offset = SyNrRows - 1;
              }
              else {
                  offset = 2;
              }
            }

            /* print the index line                                        */
            syEchos( "    ", fin );
            syEchos( secname, fin );
            p = secname;
            q = line + 12;
            while ( *p == *q ) { p++; q++; }
            if ( *p != '\0' ) {
                syEchos( " (", fin );
                for ( p = line + 12; *p != '}'; p++ ) ;
                *p = '\0';
                syEchos( line + 12, fin );
                syEchos( ")", fin );
            }
            syEchos( "\n", fin );
            offset++;

            /* we dont want no more index entries for this section         */
            secname[0] = '\0';

        }

        /* close the index again and return                                */
        SyFclose( fid );
        if ( raw )  syStopraw( fin );
        return;

    }

    /* open the table of contents                                          */
    filename[0] = '\0';
    strncat( filename, SyHelpname, sizeof(filename)-12 );
    strncat( filename, "online.toc", 11 );
    fid = SyFopen( filename, "r" );
    if ( fid == -1 ) {
        syEchos( "Help: cannot open the table of contents file '", fin );
        syEchos( filename, fin );
        syEchos( "'\n", fin );
        syEchos( "maybe use the option '-h <hlpname>'?\n", fin );
        if ( raw )  syStopraw( fin );
        return;
    }

    /* search the table of contents                                        */
    chapnr = 0;
    secnr = 0;
    secname[0] = '\0';
    matches = 0;
    while ( SyFgets( line, sizeof(line), fid ) ) {

        /* parse table of contents line                                    */
        for ( p = line; *p != '\0' && ! IsDigit(*p); p++ )  ;
        for ( i = 0; IsDigit(*p); p++ )  i = 10*i+*p-'0';
        if ( *p == '.' )  p++;
        for ( j = 0; IsDigit(*p); p++ )  j = 10*j+*p-'0';
        if ( *p == '}' )  p++;
        if ( i == 0 || ! IsAlpha(*p) ) {
          syEchos("Help: contentsline is garbage in 'online.toc'",fin);
          SyFclose( fid );
          return;
        }

        /* compare the line with the topic                                 */
        q = topic;
        match = 2;
        while ( *p != '}' && match ) {
            if ( *q != '\0' && (*p | 0x20) == (*q | 0x20) ) {
                p++; q++;
            }
            else if ( *q == ' ' || *q == '\0' ) {
                p++;
                match = 1;
            }
            else {
                match = 0;
            }
        }
        if ( *q != '\0' )  match = 0;

        /* if the offset is '-1' we are interested in the previous section */
        if ( match == 2 && offset == -1 ) {
            if ( last[0] == '\0' ) {
                syEchos("Help: the last section is the first one\n", fin );
                SyFclose( fid );
                if ( raw )  syStopraw( fin );
                return;
            }
            q = line;
            p = last;
            while ( *p != '\0' )  *q++ = *p++;
            *q = '\0';
        }

        /* if the offset is '1' we are interested in the next section      */
        if ( match == 2 && offset == 1 ) {
            if ( ! SyFgets( line, sizeof(line), fid ) ) {
                syEchos("Help: the last section is the last one\n", fin );
                SyFclose( fid );
                if ( raw )  syStopraw( fin );
                return;
            }
        }

        /* if the offset if '-2' we are interested in the first section    */
        if ( match == 2 && offset == -2 ) {
            if ( last2[0] == '\0' ) {
                syEchos("Help: the last section is the first one\n", fin );
                SyFclose( fid );
                if ( raw )  syStopraw( fin );
                return;
            }
            q = line;
            p = last2;
            while ( *p != '\0' )  *q++ = *p++;
            *q = '\0';
        }

        /* if the offset is '2' we are interested in the next chapter      */
        if ( match == 2 && offset == 2 ) {
            while ( 1 ) {
                if ( ! SyFgets( line, sizeof(line), fid ) ) {
                  syEchos("Help: the last section is in the last chapter\n",
                          fin );
                  SyFclose( fid );
                  if ( raw )  syStopraw( fin );
                  return;
                }
                for ( p = line; *p != '\0' && ! IsDigit(*p); p++ )  ;
                for ( ; *p != '}' && *p != '.'; p++ )  ;
                if ( *p == '}' )  break;
            }
        }

        /* parse table of contents line (again)                            */
        for ( p = line; *p != '\0' && ! IsDigit(*p); p++ )  ;
        for ( i = 0; IsDigit(*p); p++ )  i = 10*i+*p-'0';
        if ( *p == '.' )  p++;
        for ( j = 0; IsDigit(*p); p++ )  j = 10*j+*p-'0';
        if ( *p == '}' )  p++;
        if ( i == 0 || ! IsAlpha(*p) ) {
          syEchos("Help: contentsline is garbage in 'online.toc'",fin);
          SyFclose( fid );
          if ( raw )  syStopraw( fin );
          return;
        }

        /* if this is a precise match remember chapter and section number  */
        if ( match == 2 ) {

            /* remember the chapter and section number                     */
            chapnr = i;
            secnr  = j;

            /* get the section name                                        */
            q = secname;
            while ( *p != '}' )  *q++ = *p++;
            *q = '\0';

            /* we dont have to look further                                */
            matches = 1;
            break;
        }

        /* append a weak match to the list of matches                      */
        else if ( match == 1 ) {

            /* remember the chapter and section number                     */
            chapnr = i;
            secnr  = j;

            /* append the section name to the list of sections             */
            q = secname;
            while ( *q != '\0' )  q++;
            if ( q != secname && q < secname+sizeof(secname)-1 )
                *q++ = '\n';
            while ( *p != '}' && q < secname+sizeof(secname)-1 )
                *q++ = *p++;
            *q = '\0';

            /* we have to continue the search                              */
            matches++;
        }

        /* copy this line into <last>                                      */
        q = last;
        p = line;
        while ( *p != '\0' ) *q++ = *p++;
        *q = '\0';

        /* if the line is a chapter line copy it into <last2>              */
        if ( j == 0 ) {
            q = last2;
            p = line;
            while ( *p != '\0' )  *q++ = *p++;
            *q = '\0';
        }

    }

    /* close the table of contents file                                    */
    SyFclose( fid );

    /* if no section was found complain                                    */
    if ( matches == 0 ) {
        syEchos( "Help: no section with this name was found\n", fin );
        if ( raw )  syStopraw( fin );
        return;
    }

    /* if several sections were found return                               */
    if ( 2 <= matches ) {
        syEchos( "Help: several sections match this topic\n", fin );
        syEchos( secname, fin );
        syEchos( "\n", fin );
        if ( raw )  syStopraw( fin );
        return;
    }

    /* if this is the first time we help collect the chapter file names    */
    if ( syChapnames[0][0] == '\0' ) {

        /* open the 'online.tex' file                                      */
        filename[0] = '\0';
        strncat( filename, SyHelpname, sizeof(filename)-12 );
        strncat( filename, "online.tex", 11 );
        fid = SyFopen( filename, "r" );
        if ( fid == -1 ) {
            syEchos( "Help: cannot open the online manual file '", fin );
            syEchos( filename, fin );
            syEchos( "'\n", fin );
            syEchos( "maybe use the option '-h <hlpname>'?\n", fin );
            if ( raw )  syStopraw( fin );
            return;
        }

        /* scan this file for '\Include' lines, each contains one chapter  */
        offset = 0;
        while ( SyFgets( line, sizeof(line), fid ) ) {
            p = line;
            q = "\\Include{";
            while ( *p == *q ) { p++; q++; }
            if ( *q == '\0' ) {
                q = syChapnames[offset];
                while ( *p != '}' )  *q++ = *p++;
                *q = '\0';
                offset++;
            }
        }

        /* close the 'online.tex' file again                               */
        SyFclose( fid );

    }

    /* try to open the chapter file                                        */
    filename[0] = '\0';
    strncat( filename, SyHelpname, sizeof(filename)-13 );
    strncat( filename, syChapnames[chapnr-1], 9 );
    strncat( filename, ".tex", 4 );
    fid = SyFopen( filename, "r" );
    if ( fid == -1 ) {
        syEchos( "Help: cannot open the chapter file '", fin );
        syEchos( filename, fin );
        syEchos( "'\n", fin );
        syEchos( "maybe use the option '-h <hlpname>'?\n", fin );
        if ( raw )  syStopraw( fin );
        return;
    }

    /* create the line we are looking for                                  */
    if ( secnr == 0 ) {
        secline[0] = '\0';
        strncat( secline, "\\Chapter{", 10 );
        strncat( secline, secname, sizeof(secline)-10 );
    }
    else {
        secline[0] = '\0';
        strncat( secline, "\\Section{", 10 );
        strncat( secline, secname, sizeof(secline)-10 );
    }

    /* search the file for the correct '\Chapter' or '\Section' line       */
    match = 0;
    while ( ! match && SyFgets( line, sizeof(line), fid ) ) {
        p = line;
        q = secline;
        while ( *p == *q ) { p++; q++; }
        match = (*q == '\0' && *p == '}');
        p = line;
        q = "\\Chapter{";
        while ( *p == *q ) { p++; q++; }
        if ( *q == '\0' ) {
            q = chapname;
            while ( *p != '}' )  *q++ = *p++;
            *q = '\0';
        }
    }

    /* raise an error if this line was not found                           */
    if ( ! match ) {
        syEchos( "Help: could not find section '", fin );
        syEchos( secname, fin );
        syEchos( "' in chapter file '", fin );
        syEchos( filename, fin );
        syEchos( "'\n", fin );
        SyFclose( fid );
        if ( raw )  syStopraw( fin );
        return;
    }

    /* remember this topic for the next time                               */
    p = secname;
    syLastIndex = (syLastIndex + 1) % 16;
    q = syLastTopics[ syLastIndex ];
    while ( *p != '\0' )  *q++ = *p++;
    *q = '\0';

    /* make a header line                                                  */
    line[0] = '\0';
    strncat( line, secname, 40 );
    strncat( line,
    " _____________________________________________________________________",
             73 - strlen(chapname) );
    line[72-strlen(chapname)] = ' ';
    line[73-strlen(chapname)] = '\0';
    strncat( line, chapname, strlen(chapname)+1 );
    strncat( line, "\n", 2 );
    syEchos( "    ", fin );
    syEchos( line, fin );

    /* print everything from here to the next section line                 */
    offset = 2;
    status = 'a';
    while ( SyFgets( line, sizeof(line), fid ) ) {

        /* skip lines that begin with '\index{'                            */
        p = line;
        q = "\\index{";
        while ( *p == *q ) { p++; q++; }
        if ( *q == '\0' )  continue;

        /* skip lines that begin with '\newpage'                           */
        p = line;
        q = "\\newpage";
        while ( *p == *q ) { p++; q++; }
        if ( *q == '\0' )  continue;

        /* skip lines that begin with '\begin{'                            */
        p = line;
        q = "\\begin{";
        while ( *p == *q ) { p++; q++; }
        if ( *q == '\0' )  continue;

        /* skip lines that begin with '\end{'                              */
        p = line;
        q = "\\end{";
        while ( *p == *q ) { p++; q++; }
        if ( *q == '\0' )  continue;

        /* break if we reach a '%%%%%%%%%%%%%%%...' line                   */
        p = line;
        q = "%%%%%%%%%%%%%%%%";
        while ( *p == *q ) { p++; q++; }
        if ( *q == '\0' )  break;

        /* skip other lines that begin with a '%'                          */
        p = line;
        q = "%";
        while ( *p == *q ) { p++; q++; }
        if ( *q == '\0' )  continue;

        /* stop every 24 lines                                             */
        if ( offset == SyNrRows && raw ) {
            syEchos( "    -- <space> for more --", fin );
            ch = syGetch( fin );
            syEchos("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b",
                    fin);
            syEchos( "                          ", fin );
            syEchos("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b",
                    fin);
            if ( ch == 'q' )  {
                syEchos( "\n", fin );
                break;
            }
            else if ( ch == '\n' || ch == '\r' ) {
                offset = SyNrRows - 1;
            }
            else {
                offset = 2;
            }
        }

        /* insert empty line for '\vspace{'                                */
        p = line;
        q = "\\vspace{";
        while ( *p == *q ) { p++; q++; }
        if ( *q == '\0' ) {
            syEchos( "\n", fin );
            offset++;
            continue;
        }

        /* display the line                                                */
        p = line;
        q = last;
        spaces = 0;
        while ( *p != '\0' ) {
            if ( *p == '\\' && status != '|' ) {
                if ( last < q && q[-1] == ' ' )
                    *q++ = ' ';
                else
                    spaces++;
            }
            else if ( *p=='{' && (line==p || p[-1]!='\\') && status!='|' ) {
                if ( status == '$' )
                    *q++ = '(';
                else if ( last < q && q[-1] == ' ' )
                    *q++ = ' ';
                else
                    spaces++;
            }
            else if ( *p=='}' && (line==p || p[-1]!='\\') && status!='|' ) {
                if ( status == '$' )
                    *q++ = ')';
                else if ( last < q && q[-1] == ' ' )
                    *q++ = ' ';
                else
                    spaces++;
            }
            else if ( *p=='$' && (line==p || p[-1]!='\\') && status!='|' ) {
                if ( last < q && q[-1] == ' ' )
                    *q++ = ' ';
                else
                    spaces++;
                if ( status != '$' )
                    status = '$';
                else
                    status = 'a';
            }
            else if ( *p == ' ' && status != '|' ) {
                *q++ = ' ';
                while ( 0 < spaces ) {
                    *q++ = ' ';
                    spaces--;
                }
            }
            else if ( *p=='|' && (line==p || p[-1]!='\\'
                                  || status=='|' || status=='#') ) {
                if ( status == '|' || status == '#' )
                    status = 'a';
                else
                    status = '|';
                spaces++;
            }
            else if ( *p == '#' ) {
                if ( status == '|' )
                    status = '#';
                *q++ = *p;
            }
            else if ( *p == '\n' ) {
                if ( status == '#' )
                    status = '|';
                *q++ = *p;
            }
            else if ( *p == '>' && line!=p && p[-1]=='\\' ) {
                spaces++;
            }
            else if ( *p == '=' && line!=p && p[-1]=='\\' ) {
                spaces++;
            }
            else {
                *q++ = *p;
            }
            p++;
        }
        *q = '\0';
        syEchos( "    ", fin );  syEchos( last, fin );
        offset++;

    }

    /* close the file again                                                */
    SyFclose( fid );
    if ( raw )  syStopraw( fin );
}


/****************************************************************************
**
*F  InitSystem( <argc>, <argv> )  . . . . . . . . . initialize system package
**
**  'InitSystem' is called very early during the initialization from  'main'.
**  It is passed the command line array  <argc>, <argv>  to look for options.
**
**  For UNIX it initializes the default files 'stdin', 'stdout' and 'stderr',
**  installs the handler 'syAnsIntr' to answer the user interrupts '<ctr>-C',
**  scans the command line for options, tries to  find  'LIBNAME/init.g'  and
**  '$HOME/.gaprc' and copies the remaining arguments into 'SyInitfiles'.
*/
#ifndef SYS_STDLIB_H                    /* ANSI standard functions         */
# if SYS_ANSI
#  include      <stdlib.h>
# endif
# define SYS_STDLIB_H
#endif
#ifndef SYS_HAS_MISC_PROTO              /* ANSI/TRAD decl. from H&S 20, 13 */
#ifndef WIN32
extern  char *          getenv ( SYS_CONST char * );
extern  int             atoi ( SYS_CONST char * );
#endif
#endif
#ifndef SYS_HAS_MISC_PROTO              /* UNIX decl. from 'man'           */
extern  int             isatty ( int );
extern  char *          ttyname ( int );
#endif

#ifndef SYS_STDLIB_H                    /* ANSI standard functions         */
# if SYS_ANSI
#  include      <stdlib.h>
# endif
# define SYS_STDLIB_H
#endif
#ifndef SYS_HAS_MALLOC_PROTO
# if SYS_ANSI                           /* ANSI decl. from H&S 16.1, 16.2  */
extern  void *          malloc ( size_t );
extern  void            free ( void * );
# else                                  /* TRAD decl. from H&S 16.1, 16.2  */
extern  char *          malloc ( unsigned );
extern  void            free ( char * );
# endif
#endif

void            InitSystem (int argc, char **argv)
{
    Int                fid;            /* file identifier                 */
    Int                pre = 63*1024;  /* amount to pre'malloc'ate        */
    int                 gaprc = 1;      /* read the .gaprc file            */
    char *              ptr;            /* pointer to the pre'malloc'ated  */
    Int                i, k;           /* loop variables                  */
    char *              progname;       /* argv[0]                         */

    /* open the standard files                                             */
#if SYS_BSD || SYS_USG
    syBuf[0].fp = stdin;   setbuf( stdin, syBuf[0].buf );
    if ( isatty( fileno(stdin) ) ) {
        if ( isatty( fileno(stdout) )
          && ! strcmp( ttyname(fileno(stdin)), ttyname(fileno(stdout)) ) )
            syBuf[0].echo = stdout;
        else
            syBuf[0].echo = fopen( ttyname(fileno(stdin)), "w" );
        if ( syBuf[0].echo != (FILE*)0 && syBuf[0].echo != stdout )
            setbuf( syBuf[0].echo, (char*)0 );
    }
    else {
        syBuf[0].echo = stdout;
    }
    syBuf[1].fp = stdout;  setbuf( stdout, (char*)0 );
    if ( isatty( fileno(stderr) ) ) {
        if ( isatty( fileno(stdin) )
          && ! strcmp( ttyname(fileno(stdin)), ttyname(fileno(stderr)) ) )
            syBuf[2].fp = stdin;
        else
            syBuf[2].fp = fopen( ttyname(fileno(stderr)), "r" );
        if ( syBuf[2].fp != (FILE*)0 && syBuf[2].fp != stdin )
            setbuf( syBuf[2].fp, syBuf[2].buf );
        syBuf[2].echo = stderr;
    }
    syBuf[3].fp = stderr;  setbuf( stderr, (char*)0 );
#endif
#if WIN32
    syBuf[0].fp = stdin;
    syBuf[1].fp = stdout;
    syBuf[2].fp = stdin;
    syBuf[3].fp = stderr;
#endif

    /* install the signal handler for '<ctr>-C'                            */
#if SYS_BSD || SYS_USG  || WIN32
    if ( signal( SIGINT, SIG_IGN ) != SIG_IGN )
        signal( SIGINT, syAnswerIntr );
#endif

    SyLibname[0] = '\0';
    progname = argv[0];

    /* scan the command line for options                                   */
    while ( argc > 1 && argv[1][0] == '-' ) {

        if ( strlen(argv[1]) != 2 ) {
            fputs("gap: sorry, options must not be grouped '",stderr);
            fputs(argv[1],stderr);  fputs("'.\n",stderr);
            goto usage;
        }

        switch ( argv[1][1] ) {

        case 'b': /* '-b', supress the banner                              */
            SyBanner = ! SyBanner;
            break;

        case 'g': /* '-g', Gasman should be verbose                        */
            SyMemMgrTrace = ! SyMemMgrTrace;
            break;

        case 'l': /* '-l <libname>', change the value of 'LIBNAME'         */
            if ( argc < 3 ) {
                fputs("gap: option '-l' must have an argument.\n",stderr);
                goto usage;
            }
            strncat( SyLibname, argv[2], sizeof(SyLibname)-2 );
#if SYS_BSD || SYS_USG 
            if ( SyLibname[strlen(SyLibname)-1] != '/'
              && SyLibname[strlen(SyLibname)-1] != ';' )
                strncat( SyLibname, "/", 1 );
#endif
#if WIN32
            if ( SyLibname[strlen(SyLibname)-1] != '\\'
              && SyLibname[strlen(SyLibname)-1] != ';' )
                strncat( SyLibname, "\\", 1 );
#endif
            ++argv; --argc;
            break;

        case 'h': /* '-h <hlpname>', change the value of 'HLPNAME'         */
            if ( argc < 3 ) {
                fputs("gap: option '-h' must have an argument.\n",stderr);
                goto usage;
            }
            SyHelpname[0] = '\0';
#if SYS_BSD || SYS_USG 
            strncat( SyHelpname, argv[2], sizeof(SyLibname)-2 );
            if ( SyLibname[strlen(SyHelpname)-1] != '/' )
                strncat( SyHelpname, "/", 1 );
#endif
#if WIN32
            strncat( SyHelpname, argv[2], sizeof(SyLibname)-2 );
            if ( SyLibname[strlen(SyHelpname)-1] != '\\' )
                strncat( SyHelpname, "\\", 1 );
#endif
            ++argv; --argc;
            break;

        case 'm': /* '-m <memory>', change the value of 'SyMemory'         */
            if ( argc < 3 ) {
                fputs("gap: option '-m' must have an argument.\n",stderr);
                goto usage;
            }
            SyMemory = atoi(argv[2]);
            if ( argv[2][strlen(argv[2])-1] == 'k'
              || argv[2][strlen(argv[2])-1] == 'K' )
                SyMemory = SyMemory * 1024;
            if ( argv[2][strlen(argv[2])-1] == 'm'
              || argv[2][strlen(argv[2])-1] == 'M' )
                SyMemory = SyMemory * 1024 * 1024;
            ++argv; --argc;
            break;

        case 'i': /* '-i <interface>', set which interface will be used         */
            if ( argc < 3 ) {
                fputs("gap: option '-i' must have an argument.\n",stderr);
                goto usage;
            }
	    InitInterface(argc, argv);
            ++argv; --argc;
            break;

        case 'a': /* '-a <memory>', set amount to pre'm*a*lloc'ate         */
            if ( argc < 3 ) {
                fputs("gap: option '-a' must have an argument.\n",stderr);
                goto usage;
            }
            pre = atoi(argv[2]);
            if ( argv[2][strlen(argv[2])-1] == 'k'
              || argv[2][strlen(argv[2])-1] == 'K' )
                pre = pre * 1024;
            if ( argv[2][strlen(argv[2])-1] == 'm'
              || argv[2][strlen(argv[2])-1] == 'M' )
                pre = pre * 1024 * 1024;
            ++argv; --argc;
            break;

        case 'n': /* '-n', disable command line editing                    */
            if ( ! syWindow )  syLineEdit = 0;
            break;

        case 'f': /* '-f', force line editing                              */
            if ( ! syWindow )  syLineEdit = 2;
            break;

        case 'q': /* '-q', GAP should be quiet                             */
            SyQuiet = ! SyQuiet;
            break;

        case 'x': /* '-x', specify the length of a line                    */
            if ( argc < 3 ) {
                fputs("gap: option '-x' must have an argument.\n",stderr);
                goto usage;
            }
            SyNrCols = atoi(argv[2]);
            ++argv; --argc;
            break;

        case 'y': /* '-y', specify the number of lines                     */
            if ( argc < 3 ) {
                fputs("gap: option '-y' must have an argument.\n",stderr);
                goto usage;
            }
            SyNrRows = atoi(argv[2]);
            ++argv; --argc;
            break;

        case 'e': /* '-e', do not quit GAP on '<ctr>-D'                    */
            if ( ! syWindow )  syCTRD = ! syCTRD;
            break;

#if SYS_BSD || SYS_USG
        case 'p': /* '-p', start GAP package mode for output               */
            syWindow     = 1;
            syLineEdit   = 1;
            syCTRD       = 1;
            syWinPut( 0, "@p", "" );
            syBuf[2].fp = stdin;  syBuf[2].echo = stdout;
            syBuf[3].fp = stdout;
            break;
#endif

        case 'r': /* don't read the '.gaprc' file                          */
            gaprc = ! gaprc;
            break;

        default: /* default, no such option                                */
            fputs("gap: '",stderr);  fputs(argv[1],stderr);
            fputs("' option is unknown.\n",stderr);
            goto usage;

        }

        ++argv; --argc;

    }

    InitLibName(progname, SyLibname, sizeof(SyLibname));

   /* try to find 'LIBNAME/init.g' to read it upon initialization         */
    i = 0;  fid = -1;
    while ( fid == -1 && i <= strlen(SyLibname) ) {
        for ( k = i; SyLibname[k] != '\0' && SyLibname[k] != ';'; k++ )  ;
        SyInitfiles[0][0] = '\0';
        if ( sizeof(SyInitfiles[0]) < k-i+6+1 ) {
            fputs("gap: <libname> is too long\n",stderr);
            goto usage;
        }
        strncat( SyInitfiles[0], SyLibname+i, k-i );
        strncat( SyInitfiles[0], "init.g", 6 );
        if ( (fid = SyFopen( SyInitfiles[0], "r" )) != -1 )
            SyFclose( fid );
        i = k + 1;
    }
    if ( fid != -1 ) {
        i = 1;
    }
    else {
        i = 0;
        SyInitfiles[0][0] = '\0';
        if ( ! SyQuiet ) {
            fputs("gap: hmm, I cannot find '",stderr);
            fputs(SyLibname,stderr);
            fputs("init.g', maybe use option '-l <libname>'?\n",stderr);
        }
    }

    if ( gaprc ) {
#if SYS_BSD || SYS_USG
      if ( getenv("HOME") != 0 ) {
          SyInitfiles[i][0] = '\0';
          strncat(SyInitfiles[i],getenv("HOME"),sizeof(SyInitfiles[0])-1);
          strncat( SyInitfiles[i], "/.gaprc",
                  (Int)(sizeof(SyInitfiles[0])-1-strlen(SyInitfiles[i])));
          if ( (fid = SyFopen( SyInitfiles[i], "r" )) != -1 ) {
              ++i;
              SyFclose( fid );
          }
          else {
              SyInitfiles[i][0] = '\0';
          }
      }
#endif
    } /* if( gaprc ) */

    /* use the files from the command line                                 */
    while ( argc > 1 ) {
        if ( i >= sizeof(SyInitfiles)/sizeof(SyInitfiles[0]) ) {
            fputs("gap: sorry, cannot handle so many init files.\n",stderr);
            goto usage;
        }
        SyInitfiles[i][0] = '\0';
        strncat( SyInitfiles[i], argv[1], sizeof(SyInitfiles[0])-1 );
        ++i;
        ++argv;  --argc;
    }

    /* start the clock                                                     */
    syStartTime = SyTime();

    /* now we start                                                        */
    return;

    /* print a usage message                                               */
 usage:
    fputs("usage: gap [-l <libname>] [-h <hlpname>] [-m <gap_memory>]\n",stderr);
    fputs("           [-a <premalloc_memory>]\n",stderr);
    fputs("           [-g] [-n] [-q] [-b] [-x <nr>]  [-y <nr>]\n",stderr);
    fputs("           [-i <interface_name>]\n",stderr);
    fputs("           <file>...\n",stderr);
    fputs("  run the Groups, Algorithms and Programming system.\n",stderr);
    SyExit( 1 );
}

#ifdef WIN32
static int SlashToBackslash(char *dst, char *src)
{
	// prepend the drive letter on windows machines -- mkdir seems
	// incapable of creating directories using the "\\dirname" syntax
	if(*src == '/')
	{
		*(dst++) = _getdrive() + 'a' - 1;
		*(dst++) = ':';
	}

	for(;*src;src++)
	{
		if(*src == '/')
		{
			*(dst++) = '\\';
		}
		else
			*(dst++) = *src;
	}
	*dst = 0;

	return 0;
}


int SuperMakeDir(char *dirname)
{
	const char delim = '\\';
	struct stat buf;
	int n;
	int len;
	char *s;
	char backslash[1024];

	// in Windows, all backslashes must be escaped
	SlashToBackslash(backslash, dirname);
	dirname = backslash;

	// if dirname is null or emptystring
	if(!dirname || (dirname && !dirname[0]))
		return 0;

	// remove trailing slashes.
	len = strlen(dirname)-1;
	while(dirname[len] == delim)
		dirname[len--] = 0;

	// set all slashes to 0, and count the number set.
	for(n=1, s=dirname; *s; s++)
	{
		if(*s == delim)
		{
			*(s++) = 0;
			n++;
		}
	}

	// restore a leading slash 
	if(!dirname[0])
	{
		dirname[0] = delim;
		n--;
	}
	// if we have a path with a drive name followed by slashes, restore the slashes now
	else if(dirname[1] == ':' && !dirname[2])
	{
		dirname[2] = delim;
		n--;
	}

	// create some directories!
	for(;n>0;n--)
	{
		if(-1 == stat(dirname, &buf) && -1 == mkdir(dirname))
			return 0;
		
		dirname[strlen(dirname)] = delim;
	}
	
	return 1;
}
#else
int SuperMakeDir(char *dirname)
{
	struct stat buf;
	int n = 0;
	int len;
	char *s;
	char delim = '/';
	#define PERM S_IRWXU|S_IRWXG|S_IRWXO

	mode_t m;

	// grab the umask
	m = umask(0); umask(m);

	// if dirname is null or emptystring
	if(!dirname || (dirname && !dirname[0]))
		return 0;

	// remove trailing slashes.
	len = strlen(dirname)-1;
	while(dirname[len] == delim)
		dirname[len--] = 0;

	// set all slashes to 0, and count the number set.
	for(n=1; (s = strrchr(dirname, delim)); n++)
		*s = 0;

	// restore a leading slash 
	if(!dirname[0])
	{
		dirname[0] = delim;
		n--;
	}

	// create some directories!
	for(;n>0;n--)
	{
	        if(-1 == stat(dirname, &buf)) 
		{
	              if ( -1 == mkdir(dirname, PERM ^ m))
	                    return 0;
	              else
			chmod(dirname,PERM);           //Force the permissions over the umask (Basically allow other 
		                                       //write on created directories which is utterly useful for 
		                                       //spiral temporaries)
	        }

		dirname[strlen(dirname)] = delim;
	}
	
	return 1;
}
#endif

#ifdef WIN32
#include <shlwapi.h>

int WinGetValue(char *key, void *val, int valsize)
{
	#define TMPBUF 1024

	DWORD dwvalsize = valsize;
	DWORD valtype;
	char *keypath = key;
	char zero = 0;
	char tmpbuf[TMPBUF];

	// check input.
	if(key == NULL || val == NULL)
		return 0;

	// separate the key from the path and reverse the
	// slashes.
	key = strrchr(keypath, '/');

	if(!key)
	{
		key = keypath;
		keypath = &zero;
	}
	else
	{
		*(key) = 0;

		// make sure we don't overrun our buffer.
		if(strlen(keypath) >= TMPBUF/2)
			return 0;

		SlashToBackslash(tmpbuf, keypath);
		keypath = tmpbuf;

		*(key++) = '/';
	}

	// get the type. 
    if(ERROR_SUCCESS != SHGetValue(HKEY_LOCAL_MACHINE, keypath, key, &valtype, val, &valsize))
        return 0;

	return (valtype == REG_DWORD) ? -valsize : ((valtype == REG_SZ) ? valsize : 0);
}

#endif

/****************************************************************************
 **
 *F  SyLoadHistory()  . . . . . . . . . . loads a file into the history buffer
 **
 ****************************************************************************/

void SyLoadHistory(){
  FILE *fp;

  if((fp = fopen(".gap_history", "r")) == NULL) {
    return;
  }

  fread(syHistory, 1, sizeof(syHistory), fp);
  fclose(fp);
}

/****************************************************************************
 **
 *F  SySaveHistory()  . . . . . . . . . writes the history buffer into a file
 **
 ****************************************************************************/

void SySaveHistory(){
  int i;
  FILE *fp;

  if((fp = fopen(".gap_history", "w")) == NULL) {
    printf("Cannot open .gap_history file.\n");
    exit(1);
  }

  for (i=0;i<sizeof(syHistory);i++)
    fprintf(fp,"%c",syHistory[i]);

  fclose(fp);
}




#ifndef SYS_STDLIB_H                    /* ANSI standard functions         */
# if SYS_ANSI
#  include      <stdlib.h>
# endif
# define SYS_STDLIB_H
#endif


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
UInt SyStackAlign = SYS_STACK_ALIGN;


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
**  Put in this package because the command line processing takes place here.
*/

UInt SyMsgsFlagBags = NUM_TO_UINT(0);


/****************************************************************************
**
*V  SyStorMin . . . . . . . . . . . . . .  default size for initial workspace
**
**  See description in system.h
*/
Int SyStorMin = SY_STOR_MIN;


/****************************************************************************
**

*F * * * * * * * * * * * * * * gasman interface * * * * * * * * * * * * * * *
*/



/****************************************************************************
**
*F  SyAllocBags( <size> ) . . . . . allocate memory block of <size> kilobytes
**
**  'SyAllocBags' is called from memory mamagement to get new storage from the
**  operating system.  <size> is the needed amount in kilobytes.  This function
**  has been simplified to just allocate a block of memory and let memory
**  manager work with it.  There is no longer any assumptions about blocks
**  being contigous or adjacent.
**
**  'SyAllocBags' must return 0 if it cannot extend the workspace (i.e.,
**  allocate more memory), and a pointer to the allocated area to indicate
**  success.
*/

UInt*** SyAllocBags(Int size)
{
    UInt* p;
    UInt*** ret;

    size *= 1024;
    ret = (UInt***)malloc(size);
    if (ret == (UInt***)NULL) {
        fputs("gap: cannot extend the workspace any more\n", stderr);
        return (UInt***)0;
    }
    else {
        for (p = ret; p < ret + size / sizeof(UInt); p++)
            *p = 0;

        return ret;
    }

    return 0;
}


/****************************************************************************
**
*F  SyAbortBags( <msg> )  . . . . . . . . . abort GAP in case of an emergency
**
**  'SyAbortBags' is the function called by Gasman in case of an emergency.
*/
void SyAbortBags(
    Char* msg)
{
    SyFputs(msg, 3);
    abort();
    /*SyExit( 2 );*/
}

/****************************************************************************
**

*F * * * * * * * * * * * * * initialize package * * * * * * * * * * * * * * *
*/


/****************************************************************************
**
*F  InitSystem4( <argc>, <argv> )  . . .  initialize system package from GAP4
**
**  'InitSystem4' is called very early during the initialization from  'main'.
**  It is passed the command line array  <argc>, <argv>  to look for options.
**
**  For UNIX it initializes the default files 'stdin', 'stdout' and 'stderr',
**  installs the handler 'syAnsIntr' to answer the user interrupts '<ctr>-C',
**  scans the command line for options, tries to  find  'LIBNAME/init.g'  and
**  '$HOME/.gaprc' and copies the remaining arguments into 'SyInitfiles'.
*/

#ifndef SYS_HAS_MALLOC_PROTO
# if SYS_ANSI                           /* ANSI decl. from H&S 16.1, 16.2  */
extern void* malloc(size_t);
extern void   free(void*);
# else                                  /* TRAD decl. from H&S 16.1, 16.2  */
extern char* malloc(unsigned);
extern void   free(char*);
# endif
#endif

static UInt ParseMemory(Char* s)
{
    UInt size;
    size = atoi(s);
    if (s[strlen(s) - 1] == 'k'
        || s[strlen(s) - 1] == 'K')
        size = size * 1024;
    if (s[strlen(s) - 1] == 'm'
        || s[strlen(s) - 1] == 'M')
        size = size * 1024 * 1024;
    if (s[strlen(s) - 1] == 'g'
        || s[strlen(s) - 1] == 'G')
        size = size * 1024 * 1024 * 1024;
    return size;
}

#define ONE_ARG(opt) \
        case opt: \
            if ( argc < 3 ) { \
                FPUTS_TO_STDERR("gap4: option " #opt " must have an argument.\n"); \
                goto usage; \
            } 

void InitSystem4(
    Int                 argc,
    Char* argv[])
{
    Int                 pre = 100 * 1024; /* amount to pre'malloc'ate        */
    Char* ptr;            /* pointer to the pre'malloc'ated  */
    Char* ptr1;           /* more pre'malloc'ated  */
    UInt                i;              /* loop variable                   */

    /* scan the command line for options                                   */
    while (argc > 1 && argv[1][0] == '-') {
        if (strlen(argv[1]) != 2) {
            FPUTS_TO_STDERR("gap: sorry, options must not be grouped '");
            FPUTS_TO_STDERR(argv[1]);  FPUTS_TO_STDERR("'.\n");
            goto usage;
        }

        switch (argv[1][1]) {
            /* '-a <memory>', set amount to pre'm*a*lloc'ate                   */
            ONE_ARG('a');
            pre = ParseMemory(argv[2]);
            ++argv; --argc; break;

            /* '-g', Gasman should be verbose                                  */
            ONE_ARG('g');
            SyMsgsFlagBags = (SyMsgsFlagBags + 1) % 3;
            ++argv; --argc; break;

            /* '-m <memory>', change the value of 'SyStorMin'                  */
            ONE_ARG('m');
            SyStorMin = ParseMemory(argv[2]) / 1024;
            ++argv; --argc; break;

            /* '-h', print a usage help                                        */
        case 'h':
            goto fullusage;

            /* default, skip unknown option, GAP3 should handle it             */
        default: break;
            /*
                  FPUTS_TO_STDERR("gap: '");  FPUTS_TO_STDERR(argv[1]);
                  FPUTS_TO_STDERR("' option is unknown.\n");
                  goto usage;*/

        }
        ++argv; --argc;
    }

    /* fix max if it is lower than min                                     */
    // if ( SyStorMax < SyStorMin )
    //    SyStorMax = SyStorMin;          // no need for a maximum memory amount
    /* now we start                                                        */
    return;

    /* print a usage message                                               */
usage:
    FPUTS_TO_STDERR("usage: gap4 [OPTIONS] [FILES]\n");
    FPUTS_TO_STDERR("       use '-h' option to get help.\n");
    FPUTS_TO_STDERR("\n");
    SyExit(1);

fullusage:
    FPUTS_TO_STDERR("usage: gap4 [OPTIONS] [FILES]\n");
    FPUTS_TO_STDERR("\n");
    FPUTS_TO_STDERR("  -g          show GASMAN messages (full garbage collections)\n");
    FPUTS_TO_STDERR("  -g -g       show GASMAN messages (all garbage collections)\n");
    FPUTS_TO_STDERR("  -m <mem>    set the initial workspace size\n");
    FPUTS_TO_STDERR("  -a <mem>    set amount to pre-malloc-ate\n");
    FPUTS_TO_STDERR("              postfix 'm' = *1024*1024, 'g' = *1024*1024*1024\n");
    FPUTS_TO_STDERR("\n");
    SyExit(1);
}

