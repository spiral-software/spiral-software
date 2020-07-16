/****************************************************************************
**
*A  scanner.c                   GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2020, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2020) by the GAP Group (www.gap-system.org).
**
**  This file contains the functions of the scanner, which is responsible for
**  all input and output processing.
**
**  The scanner  exports two very  important abstractions.  The  first is the
**  concept that an input file is  a stream of symbols,  such nasty things as
**  <space>,  <tab>,  <newline> characters or  comments (they are worst  :-),
**  characters making  up identifiers  or  digits that  make  up integers are
**  hidden from the rest of GAP.
**
**  The second is  the concept of  a current input  and output file.   In the
**  main   module   they are opened  and   closed  with the  'OpenInput'  and
**  'CloseInput' respectively  'OpenOutput' and 'CloseOutput' calls.  All the
**  other modules just read from the  current input  and write to the current
**  output file.
**
**  The scanner relies on the functions  provided  by  the  operating  system
**  dependent module 'system.c' for the low level input/output.
**
*/
#define USE_FLEX

#include        <string.h>
#include        <stdlib.h>
#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"
#include        "scanner.h"             /* definition part of this package */
#include        "comments.h"
#include        "hooks.h"
#include        "string4.h"        /* strings                         */
#include		"GapUtils.h"

#include 	"plist.h"
#include	"integer.h"

extern void Print(Bag);
#ifdef __INTEL_COMPILER
#pragma warning( disable : 170 )
#endif
/****************************************************************************
**
*V  Symbol  . . . . . . . . . . . . . . . . .  current symbol read from input
**
**  The  variable 'Symbol' contains the current  symbol read from  the input.
**  It is represented as an unsigned long integer.
**
**  The possible values for 'Symbol' are defined in the  definition  file  of
**  this package as follows:
**
#define S_ILLEGAL       (0L)

#define S_IDENT         ((1L<< 3))
#define S_INT           ((1L<< 4))
#define S_CHAR          ((1L<< 5)+0)
#define S_STRING        ((1L<< 5)+1)

#define S_DOT           ((1L<< 6))
#define S_LBRACK        ((1L<< 7)+0)
#define S_RBRACK        ((1L<< 8)+0)
#define S_LBRACE        ((1L<< 7)+1)
#define S_RBRACE        ((1L<< 8)+1)
#define S_LPAREN        ((1L<< 9))
#define S_RPAREN        ((1L<<10))
#define S_COMMA         ((1L<<11)+0)
#define S_DOTDOT        ((1L<<11)+1)

#define S_IF            ((1L<<12)+0)
#define S_THEN          ((1L<<13))
#define S_ELIF          ((1L<<14)+0)
#define S_ELSE          ((1L<<14)+1)
#define S_FI            ((1L<<15))

#define S_FOR           ((1L<<12)+1)
#define S_DO            ((1L<<16))
#define S_OD            ((1L<<17))

#define S_REPEAT        ((1L<<12)+2)
#define S_UNTIL         ((1L<<18))
#define S_WHILE         ((1L<<12)+3)

#define S_ASSIGN        ((1L<<19))
#define S_SEMICOLON     ((1L<<20))

#define S_FUNCTION      ((1L<<21))
#define S_LOCAL         ((1L<<22))
#define S_END           ((1L<<23))
#define S_RETURN        ((1L<<12)+4)

#define S_MAPTO         ((1L<<24)+0)
#define S_MAPTO_METH    ((1L<<24)+1)
#define S_ASSIGN_MAP    ((1L<<24)+2)  =>, used with Delay()

#define S_NOT           ((1L<<25)+0)
#define S_AND           ((1L<<25)+1)
#define S_OR            ((1L<<25)+2)

#define S_EQ            ((1L<<26)+0)
#define S_LT            ((1L<<26)+1)
#define S_GT            ((1L<<26)+2)
#define S_NE            ((1L<<26)+3)
#define S_LE            ((1L<<26)+4)
#define S_GE            ((1L<<26)+5)
#define S_IN            ((1L<<26)+6)

#define S_PLUS          ((1L<<27)+0)
#define S_MINUS         ((1L<<27)+1)
#define S_BACKQUOTE     ((1L<<27)+2)

#define S_MULT          ((1L<<28)+0)
#define S_DIV           ((1L<<28)+1)
#define S_MOD           ((1L<<28)+2)
#define S_POW           ((1L<<28)+3)

#define S_QUIT          ((1L<<29))
#define S_EOF           ((1L<<30))
*/
UInt           Symbol;

/****************************************************************************
**
*T  TypSymbolSet  . . . . . . . . . . . . . . . . . . type of sets of symbols
**
**  'TypSymbolSet' is the type of sets of symbols.  Sets  of symbols are used
**  in the error recovery of the  parser  to specify that 'Match' should skip
**  all symbols until finding one in a specified set.
**
**  If there were less than 32 different symbols  things would be  very easy.
**  We could  simply assign   the  symbolic constants   that are the possible
**  values for 'Symbol' values 1, 2, 4, 8, 16, ...  and so on.  Then making a
**  set  would  simply mean  or-ing the  values, as in  'S_INT|S_STRING', and
**  checking whether a symbol is in a set would be '(<symbol> & <set>) != 0'.
**
**  There  are however more  than 32 different  symbols, so  we must  be more
**  clever.  We  group some  symbols that  are syntactically  equivalent like
**  '*', '/' in a class. We use the least significant 3 bits to differentiate
**  between members in one class.  And now  every symbol class, many of which
**  contain   just  one  symbol,  has exactely  one   of  the  remaining most
**  significant 29  bits  set.   Thus   sets  of symbols  are  represented as
**  unsigned long integers, which is typedef-ed to 'TypSymbolSet'.
**
**  The classes are as follows, all other symbols are in a class themself:
**      if, for, repeat, while, return
**      elif, else
**      not, and, or
**      =, <>, <, >=, <=, >, in
**      +, -
**      *, /, mod, ^
**
**  'TypSymbolSet'  is defined in the   definition  file of  this  package as
**  follows:
**
typedef unsigned long   TypSymbolSet;
*/


/****************************************************************************
**
*F  IS_IN( <symbol>, <set> )  . . . . . . . . is a symbol in a set of symbols
**
**  'IS_IN' returns 1 if the symbol <symbol> is in the symbol set <set> and 0
**  otherwise.  Due to the grouping into classes some symbol sets may contain
**  more than mentioned, for  example 'IS_IN(S_POW,S_MULT|S_DIV|S_MOD)' is 1.
**
**  'IS_IN' is defined in the definition file of this package as follows:
**
#define IS_IN(SYMBOL,SET)       ((SYMBOL) & ((SET) & ~7))
*/


/****************************************************************************
**
*V  EXPRBEGIN . . . . . . . . . . . . set of symbols that start an expression
*V  STATBEGIN . . . . . . . . . . . . . set of symbols that start a statement
**
**  'EXPRBEGIN' is  the set   of symbols   that might  start   an expression.
**  'STATBEGIN' is the set of symbols that might  start a stament, this  is a
**  superset of 'EXPRBEGIN', since expressions are themselfs statments.
**
**  'EXPRBEGIN' and 'STATBEGIN'  are defined in  the definition  file of this
**  package as follows:
**
#define EXPRBEGIN  (S_IDENT|S_INT|S_STRING|S_LPAREN|S_FUNCTION)
#define STATBEGIN  (EXPRBEGIN|S_IF|S_FOR|S_WHILE|S_REPEAT|S_RETURN)
*/


/****************************************************************************
**
*V  Value . . . . . . . . . . . .  value of the identifier, integer or string
**
**  If 'Symbol' is 'S_IDENT', 'S_INT' or 'S_STRING' the variable 'Value' holds
**  the name of the identifier, the digits of the integer or the value of the
**  string constant.
**
**  Note that the size of  'Value' limits the  maximal number of  significant
**  characters  of an identifier,   the maximal size  of  an  integer and the
**  maximal length of a  string.   'GetIdent', 'GetInt' and 'GetStr' truncate
**  identifier, integers or strings after that many characters.
*/
char            Value [MAX_IDENT_SIZE];


/****************************************************************************
**
*V  NrError . . . . . . . . . . . . . . . .  number of errors in current expr
*V  NrErrLine . . . . . . . . . . . . . . .  number of errors on current line
*V  NrHadSyntaxErrors . . . . . . . . . . .  was there any syntax error ever
**
**  'NrError' is an integer whose value is the number of errors already found
**  in the current expression.  It is set to 0 at the beginning of 'Read' and
**  incremented with each 'SyntaxError' call, including those  from  'Match'.
**
**  If 'NrError' is greater than zero the parser functions  will  not  create
**  new bags.  This prevents the parser from creating new bags after an error
**  occured.
**
**  'NrErrLine' is an integer whose value is the number of  errors  found  on
**  the current line.  It is set to 0 in 'GetLine' and incremented with  each
**  'SyntaxError' call, including those from 'Match'.
**
**  If 'NrErrLine' is greater  than  zero  'SyntaxError' will  not  print  an
**  error message.  This prevents the printing of multiple error messages for
**  one line, since they  probabely  just reflect  the  fact that the  parser
**  has not resynchronized yet.
**
**  'NrHadSyntaxErrors' set to non zero once syntax errors is encountered. It 
**  never turns to zero again.
**
*/
Int            NrError;
Int            NrErrLine;
Int            NrHadSyntaxErrors = 0;


/****************************************************************************
**
*V  Prompt  . . . . . . . . . . . . . . . . . . . . . .  prompt to be printed
**
**  'Prompt' holds the string that is to be printed if a  new  line  is  read
**  from the interactive files '*stdin*' or '*errin*'.
**
**  It is set to 'gap> ' or 'brk> ' in the  read-eval-print loops and changed
**  to the partial prompt '> ' in 'Read' after the first symbol is read.
*/
char            * Prompt;



TypInputFile    InputFiles [SCANNER_INPUTS];
TypInputFile    * Input;
char            * In;


/****************************************************************************
**
*T  TypOutputFiles  . . . . . . . . . structure of an open output file, local
*V  OutputFiles . . . . . . . . . . . . . . stack of open output files, local
*V  Output  . . . . . . . . . . . . . . pointer to current output file, local
**
**  'TypOutputFile' describes the information stored for open  output  files:
**  'file' holds the file identifier which is  received  from  'SyFopen'  and
**  which is passed to  'SyFputs'  and  'SyFclose'  to  identify  this  file.
**  'line' is a buffer that holds the current output line.
**  'pos' is the position of the current character on that line.
**
**  'OutputFiles' is the stack of open output files.  It  is  represented  as
**  an array of structures of type 'TypOutputFile'.
**
**  'Output' is a pointer to the current output file.  It points to  the  top
**  of the stack 'OutputFiles'.
*/

Int OutputLineLen;

#define SCANNER_OUTPUTS 16
//typedef struct {
//    Int        file;
//    char        *line;
//    Int        pos;
//    Int        indent;
//    Int        spos;
//    Int        sindent;
//    char*       mem; /* holds memory buffer if output goes into memory         */
//                     /* first UInt is the buffer size without 2*sizeof(UInt);  */
//                     /* second UInt is the number of characters written        */
//                     /* null terminated characters array;                      */
//
//    Bag		hdList;
//}       TypOutputFile;

TypOutputFile   OutputFiles [SCANNER_OUTPUTS];
TypOutputFile   * Output;

/* INITIAL_MEM_SIZE - initial memory buffer size for OpenMemory() */
#define INITIAL_MEM_SIZE    2048

/****************************************************************************
**
*V  Logfile . . . . . . . . . . . . . . . . file identifier of logfile, local
**
**  'Logfile' is the file identifier of the current logfile.   If this is not
**  -1 the  scanner echoes all  input from the  files '*stdin*' and '*errin*'
**  and all output to the files '*stdout*' and '*errout*' to this file.
*/
Int            Logfile = -1;


/****************************************************************************
**
*V  InputLogfile  . . . . . . . . . . . . . file identifier of logfile, local
**
**  'InputLogfile' is the file identifier of the current logfile.   If it  is
**  not -1 the scanner echoes  input from the files  '*stdin*' and  '*errin*'
**  to this file.
*/
Int            InputLogfile = -1;




/****************************************************************************
**
*F  GetLine() . . . . . . . . . . . . . . . . . . . . . . . get a line, local
**
**  'GetLine' fetches another line from the input file 'Input->file' into the
**  buffer 'Input->line', sets  the pointer 'In'  to  the beginning  of  this
**  buffer and returns the first character from the line.
**
**  If   the input file is  '*stdin*'   or '*errin*' 'GetLine'  first  prints
**  'Prompt', unless it is '*stdin*' and GAP was called with option '-q'.
**
**  If there is a logfile in use and the input file is '*stdin*' or '*errin*'
**  'GetLine' echoes the new line to the logfile.
*/
char            GetLine (void)
{
    /* if file is '*stdin*' or '*errin*' print the prompt and flush it     */
    if ( Input->file == 0 ) {
        if ( ! SyQuiet ) Pr( "%s%c", (Int)Prompt, (Int)'\03' );
        else             Pr( "%c", (Int)'\03', 0 );
    }
    else if ( Input->file == 2 ) {
        Pr( "%s%c", (Int)Prompt, (Int)'\03' );
    }

    /* bump the line number                                                */
    if ( Input->line < In && (*(In-1) == '\n' || *(In-1) == '\r') )
        Input->number++;

    /* initialize 'In', no errors on this line so far                      */
    In = Input->line;  In[0] = '\0';
    NrErrLine = 0;

 
    /* try to read a line                                        */
   if ( ! SyFgets( In, sizeof(Input->line), Input->file ) ) {
        In[0] = '\377';  In[1] = '\0';
        return *In;
    }

    /* deal with help requests (preliminary hack)                          */
    if ( In[0] == '?' ) {
        In[strlen(In)-1] = '\0';
        SyHelp( In+1, Input->file );
        In[0] = '\n';
        In[1] = '\0';
    }

    /* if neccessary echo the line to the logfile                          */
    if ( Logfile != -1 && (Input->file == 0 || Input->file == 2) )
        SyFputs( In, Logfile );
    if ( InputLogfile != -1 && (Input->file == 0 || Input->file == 2) )
        SyFputs( In, InputLogfile );

        /* return the current character                                        */
    return *In;
}

/****************************************************************************
**
*F  GetIdent()  . . . . . . . . . . . . . get an identifier or keyword, local
**
**  'GetIdent' reads   an identifier from  the current  input  file  into the
**  variable 'Value' and sets 'Symbol' to 'S_IDENT'.   The first character of
**  the   identifier  is  the current character  pointed to  by 'In'.  If the
**  characters make  up   a  keyword 'GetIdent'  will  set   'Symbol'  to the
**  corresponding value.  The parser will ignore 'Value' in this case.
**
**  An  identifier consists of a letter  followed by more letters, digits and
**  underscores '_'.  An identifier is terminated by the first  character not
**  in this  class.  The escape sequence '\<newline>'  is ignored,  making it
**  possible to split  long identifiers  over multiple lines.  The  backslash
**  '\' can be used  to include special characters like  '('  in identifiers.
**  For example 'G\(2\,5\)' is an identifier not a call to a function 'G'.
**
**  The size  of 'Value' limits the  number  of significant characters  in an
**  identifier.   If  an  identifier   has more characters    'GetIdent' will
**  silently truncate it.
**
**  After reading the identifier 'GetIdent'  looks at the  first and the last
**  character  of  'Value' to see if  it  could possibly  be  a keyword.  For
**  example 'test'  could  not be  a  keyword  because there  is  no  keyword
**  starting and ending with a 't'.  After that  test either 'GetIdent' knows
**  that 'Value' is not a keyword, or there is a unique possible keyword that
**  could match, because   no two  keywords  have  identical  first and  last
**  characters.  For example if 'Value' starts with 'f' and ends with 'n' the
**  only possible keyword  is 'function'.   Thus in this case  'GetIdent' can
**  decide with one string comparison if 'Value' holds a keyword or not.
*/
void            GetSymbol ( void );

void            GetIdent (void)
{
    Int                i;
    Int                isQuoted;

    /* initially it could be a keyword                                     */
    isQuoted = 0;

    /* read all characters into 'Value'                                    */
    for ( i=0; IsAlpha(*In) || IsDigit(*In) || *In=='_' || *In=='\\' || *In=='@'; i++ ) {

        /* handle escape sequences                                         */
        if ( *In == '\\' ) {
            GET_CHAR();
            if      ( *In == '\n' && i == 0 )  { GetSymbol();  return; }
            else if ( *In == '\n' && i < sizeof(Value)-1 )  i--;
            else if ( *In == 'n'  && i < sizeof(Value)-1 )  Value[i] = '\n';
            else if ( *In == 't'  && i < sizeof(Value)-1 )  Value[i] = '\t';
            else if ( *In == 'r'  && i < sizeof(Value)-1 )  Value[i] = '\r';
            else if ( *In == 'b'  && i < sizeof(Value)-1 )  Value[i] = '\b';
            else if (                i < sizeof(Value)-1 )  Value[i] = *In;
            isQuoted = 1;
        }

        /* put normal chars into 'Value' but only if there is room         */
        else {
            if ( i < sizeof(Value)-1 )  Value[i] = *In;
        }

        /* read the next character                                         */
        GET_CHAR();

    }

    /* terminate the identifier and lets assume that it is not a keyword   */
    if ( i < sizeof(Value)-1 )  Value[i] = '\0';
    Symbol = S_IDENT;

    /* now check if 'Value' holds a keyword                                */
    switch ( 256*Value[0]+Value[i-1] ) {
    case 256*'a'+'d': if(!strcmp(Value,"and"))     Symbol=S_AND;     break;
    case 256*'d'+'o': if(!strcmp(Value,"do"))      Symbol=S_DO;      break;
    case 256*'e'+'f': if(!strcmp(Value,"elif"))    Symbol=S_ELIF;    break;
    case 256*'e'+'e': if(!strcmp(Value,"else"))    Symbol=S_ELSE;    break;
    case 256*'e'+'d': if(!strcmp(Value,"end"))     Symbol=S_END;     break;
    case 256*'f'+'i': if(!strcmp(Value,"fi"))      Symbol=S_FI;      break;
    case 256*'f'+'r': if(!strcmp(Value,"for"))     Symbol=S_FOR;     break;
    case 256*'f'+'n': if(!strcmp(Value,"function"))Symbol=S_FUNCTION;break;
    case 256*'i'+'f': if(!strcmp(Value,"if"))      Symbol=S_IF;      break;
    case 256*'i'+'n': if(!strcmp(Value,"in"))      Symbol=S_IN;      break;
    case 256*'l'+'l': if(!strcmp(Value,"local"))   Symbol=S_LOCAL;   break;
    case 256*'m'+'d': if(!strcmp(Value,"mod"))     Symbol=S_MOD;     break;
    case 256*'m'+'h': if(!strcmp(Value,"meth"))    Symbol=S_METHOD;  break;
    case 256*'n'+'t': if(!strcmp(Value,"not"))     Symbol=S_NOT;     break;
    case 256*'o'+'d': if(!strcmp(Value,"od"))      Symbol=S_OD;      break;
    case 256*'o'+'r': if(!strcmp(Value,"or"))      Symbol=S_OR;      break;
    case 256*'r'+'t': if(!strcmp(Value,"repeat"))  Symbol=S_REPEAT;  break;
    case 256*'r'+'n': if(!strcmp(Value,"return"))  Symbol=S_RETURN;  break;
    case 256*'t'+'n': if(!strcmp(Value,"then"))    Symbol=S_THEN;    break;
    case 256*'u'+'l': if(!strcmp(Value,"until"))   Symbol=S_UNTIL;   break;
    case 256*'w'+'e': if(!strcmp(Value,"while"))   Symbol=S_WHILE;   break;
    case 256*'q'+'t': if(!strcmp(Value,"quit"))    Symbol=S_QUIT;    break;
    case 256*'d'+'l': if(!strcmp(Value,"dbl"))     {
        /* read in a double. This dirty way seems to be the only plausible
           solution in this kind of a scanner. This scanner must be ported
           to use flex, then it would be nicer */
        Match(Symbol, "", NUM_TO_UINT(0));
        if(Symbol!=S_LPAREN)
            Match(0, "(", EXPRBEGIN);
        else {
            Int i = 0;
            char ch = *In;
            while(i < sizeof(Value)-1 && ch != ')' &&
                  (IsDigit(ch) || ch=='.' || ch=='e' || ch=='+'|| ch=='-')) {
                Value[i++] = ch;
                GET_CHAR(); ch = *In;
            }
            if(i < sizeof(Value)-1)
                Value[i] = '\0';
            if(ch!=')') Match(0, ")", EXPRBEGIN);
            else Match(Symbol, "", NUM_TO_UINT(0));
            Symbol=S_DOUBLE;
            break;
        }
    }
    }

    /* if it is quoted it is an identifier                                 */
    if ( isQuoted )  Symbol = S_IDENT;

}


/****************************************************************************
**
*F  GetInt()  . . . . . . . . . . . . . . . . . . . . . get an integer, local
**
**  'GetInt' reads  an integer number from  the  current  input file into the
**  variable  'Value' and sets  'Symbol' to 'S_INT'.   The first character of
**  the integer is the current character pointed to by 'In'.
**
**  An  integer is   a sequence of   digits  '0..9'.    The  escape  sequence
**  '\<newline>' is ignored, making it possible to  split  long integers over
**  multiple lines.
**
**  If the sequence contains characters which are not  digits  'GetInt'  will
**  interpret the sequence as an identifier and set 'Symbol' to 'S_IDENT'.
**
**  The size of 'Value' limits the maximal number of digits  of  an  integer.
**  If an integer has more digits 'GetInt' issues a warning and truncates it.
*/
void            GetInt (void)
{
    enum { INT, IDENT } state = INT;
    Int i  = 0;
    char ch = *In;

    /* read the digits into 'Value'                                        */
    while(IsDigit(ch) || IsAlpha(ch) || ch=='_' || ch=='\\') {
        /* handle escape sequences                                         */
        if ( ch == '\\' ) {
            GET_CHAR();
            if      ( ch == '\n' && i < sizeof(Value)-1 )  i--;
            else if ( ch == 'n'  && i < sizeof(Value)-1 )  Value[i] = '\n';
            else if ( ch == 't'  && i < sizeof(Value)-1 )  Value[i] = '\t';
            else if ( ch == 'r'  && i < sizeof(Value)-1 )  Value[i] = '\r';
            else if ( ch == 'b'  && i < sizeof(Value)-1 )  Value[i] = '\b';
            else if ( ch == 'c'  && i < sizeof(Value)-1 )  Value[i] = '\03';
            else if (               i < sizeof(Value)-1 )  Value[i] = ch;
        }
        /* put normal chars into 'Value' if there is room                  */
        else if ( i < sizeof(Value)-1 )
            Value[i] = ch;

        /* if the characters contain non digits it is a variable           */
        if ( ! IsDigit(ch) && ch != '\n' )  state = IDENT;

        GET_CHAR(); ch = *In; ++i;
    }

    /* check for numbers with too many digits                              */
    if ( sizeof(Value)-1 <= i )
        SyntaxError("integer must have less than 1024 digits");

    /* terminate the integer                                               */
    if ( i < sizeof(Value)-1 )  Value[i] = '\0';

    switch(state) {
    case INT  : Symbol = S_INT; break;
    case IDENT: Symbol = S_IDENT; break;
    }
}


/****************************************************************************
**
*F  GetStr()  . . . . . . . . . . . . . . . . . . . . . . get a string, local
**
**  'GetStr' reads  a  string from the  current input file into  the variable
**  'Value' and sets 'Symbol'   to  'S_STRING'.  The opening double quote '"'
**  of the string is the current character pointed to by 'In'.
**
**  A string is a sequence of characters delimited  by double quotes '"'.  It
**  must not include  '"' or <newline>  characters, but the  escape sequences
**  '\"' or '\n' can  be used instead.  The  escape sequence  '\<newline>' is
**  ignored, making it possible to split long strings over multiple lines.
**
**  An error is raised if the string includes a <newline> character or if the
**  file ends before the closing '"'.
**
**  The size of 'Value' limits the maximal number of characters in a  string.
**  If a string has more characters 'GetStr' issues a error and truncates it.
*/
void            GetStr (void)
{
    Int                i = 0;

    /* skip '"'                                                            */
    GET_CHAR();

    /* read all characters into 'Value'                                    */
    for ( i = 0; *In != '"' && *In != '\n' && *In != '\377'; i++ ) {

        /* handle escape sequences                                         */
        if ( *In == '\\' ) {
            GET_CHAR();
            if      ( *In == '\n' && i < sizeof(Value)-1 )  i--;
            else if ( *In == 'n'  && i < sizeof(Value)-1 )  Value[i] = '\n';
            else if ( *In == 't'  && i < sizeof(Value)-1 )  Value[i] = '\t';
            else if ( *In == 'r'  && i < sizeof(Value)-1 )  Value[i] = '\r';
            else if ( *In == 'b'  && i < sizeof(Value)-1 )  Value[i] = '\b';
            else if ( *In == 'c'  && i < sizeof(Value)-1 )  Value[i] = '\03';
            else if (                i < sizeof(Value)-1 )  Value[i] = *In;
        }

        /* put normal chars into 'Value' but only if there is room         */
        else {
            if ( i < sizeof(Value)-1 )  Value[i] = *In;
        }

        /* read the next character                                         */
        GET_CHAR();

    }

    /* check for error conditions                                          */
    if ( *In == '\n'  )
        SyntaxError("string must not include <newline>");
    if ( *In == '\377' )
        SyntaxError("string must end with \" before end of file");
    if ( sizeof(Value)-1 <= i )
        SyntaxError("string must have less than 1024 characters");

    /* terminate the string, set 'Symbol' and skip trailing '"'            */
    if ( i < sizeof(Value)-1 )  Value[i] = '\0';
    Symbol = S_STRING;
    if ( *In == '"' )  GET_CHAR();
}


/****************************************************************************
**
*F  GetSymbol() . . . . . . . . . . . . . . . . .  get the next symbol, local
**
**  'GetSymbol' reads  the  next symbol from   the  input,  storing it in the
**  variable 'Symbol'.  If 'Symbol' is  'T_IDENT', 'T_INT' or 'T_STRING'  the
**  value of the symbol is stored in the variable 'Value'.  'GetSymbol' first
**  skips all <space>, <tab> and <newline> characters and comments.
**
**  After reading  a  symbol the current  character   is the first  character
**  beyond that symbol.
*/

extern int yylex();

void            GetSymbol (void)
{
    Symbol = yylex();
}


/****************************************************************************
**
*F  SyntaxError( <msg> )  . . . . . . . . . . . . . . .  raise a syntax error
**
**  'SyntaxError' prints the current line, followed by the error message:
**
**      ^ syntax error, <msg> in <current file name>
**
**  with the '^' pointing to the current symbol on the current line.  If  the
**  <current file name> is '*stdin*' it is not printed.
**
**  'SyntaxError' is called from the parser to print error messages for those
**  errors that are not cought by 'Match',  for example if the left hand side
**  of an assignment is not a variable, a list element or a record component,
**  or if two formal arguments of a function have the same identifier.  It is
**  also called for warnings, for example if a statement has no effect.
**
**  'SyntaxError' first increments 'NrError' by   1.  If 'NrError' is greater
**  than zero the parser functions  will not create  new bags.  This prevents
**  the parser from creating new bags after an error occured.
**
**  'SyntaxError' also  increments  'NrErrLine'  by   1.  If  'NrErrLine'  is
**  greater than zero  'SyntaxError' will not print an  error  message.  This
**  prevents the printing of multiple error messages for one line, since they
**  probabely  just reflect the  fact  that the parser has not resynchronized
**  yet.  'NrErrLine' is reset to 0 if a new line is read in 'GetLine'.
*/
void            SyntaxError (char *msg)
{
    Int                i;
    Int                isStdIn = (strcmp( "*stdin*", Input->name ) == 0 );
    static Int         launchedEdit = 0;

    /* one more error                                                      */
    NrError++;
    NrErrLine++;

    /* do not print a message if we found one already on the current line  */
    if ( NrErrLine != 1 )
        return;

    /* print the message and the filename, unless it is '*stdin*'          */
    Pr( "Syntax error: %s (symbol: '%s')", (Int)msg, (Int)Value );
    if ( !isStdIn ) {
        Pr( " in %s line %d", (Int)Input->name, (Int)Input->number );
	if(!launchedEdit) {
	    launchedEdit = 1;
	    HooksEditFile(Input->name, (int)Input->number);
	}
    }
    Pr( "\n", 0, 0 );

    /* print the current line                                              */
    Pr( "%s", (Int)Input->line, 0 );

    /* print a '^' pointing to the current position                        */
    for ( i = 0; i < In - Input->line - 1; i++ ) {
        if ( Input->line[i] == '\t' )  Pr("\t",0,0);
        else  Pr(" ",0,0);
    }
    Pr( "^\n", 0, 0 );

}


/****************************************************************************
**
*F  Match( <symbol>, <msg>, <skipto> )  . match current symbol and fetch next
**
**  'Match' is the main  interface between the  scanner and the  parser.   It
**  performs the  4 most common actions in  the scanner  with  just one call.
**  First it checks that  the current symbol stored  in the variable 'Symbol'
**  is the expected symbol  as passed in the  argument <symbol>.  If  it  is,
**  'Match' reads the next symbol from input  and returns.  Otherwise 'Match'
**  first prints the current input line followed by the syntax error message:
**  '^ syntax error, <msg> expected' with '^' pointing to the current symbol.
**  It then  skips symbols up to one  in the resynchronisation  set <skipto>.
**  Actually 'Match' calls 'SyntaxError' so its comments apply here too.
**
**  One kind of typical 'Match' call has the form
**
**      'Match( Symbol, "", 0L );'.
**
**  This is used if the parser knows that the current  symbol is correct, for
**  example in 'RdReturn'  the   first symbol must be 'S_RETURN',   otherwise
**  'RdReturn' would not have been  called.  Called this  way 'Match' will of
**  course never raise an syntax error,  therefore <msg>  and <skipto> are of
**  no concern, they are passed nevertheless  to please  lint.  The effect of
**  this call is merely to read the next symbol from input.
**
**  Another typical 'Match' call is in 'RdIf' after we read the if symbol and
**  the condition following, and now expect to see the 'then' symbol:
**
**      Match( S_THEN, "then", STATBEGIN|S_ELIF|S_ELSE|S_FI|follow );
**
**  If the current symbol  is 'S_THEN' it is  matched  and the next symbol is
**  read.  Otherwise 'Match'  prints the  current line followed by the  error
**  message: '^ syntax error, then expected'.  Then 'Match' skips all symbols
**  until finding either  a symbol  that can begin  a statment,  an 'elif' or
**  'else' or 'fi' symbol, or a symbol that is  contained in the set <follow>
**  which is passed to  'RdIf' and contains  all symbols allowing  one of the
**  calling functions to resynchronize, for example 'S_OD' if 'RdIf' has been
**  called from 'RdFor'.  <follow>  always contain 'S_EOF', which 'Read' uses
**  to resynchronise.
**
**  If 'Match' needs to  read a  new line from  '*stdin*' or '*errin*' to get
**  the next symbol it prints the string pointed to by 'Prompt'.
*/
void            Match (UInt symbol, char *msg, TypSymbolSet skipto)
{
    char                errmsg [256];

    /* if 'Symbol' is the expected symbol match it away                    */
    if ( symbol == Symbol ) {
        GetSymbol();
    }

    /* else generate an error message and skip to a symbol in <skipto>     */
    else {
        errmsg[0] ='\0';
        strncat( errmsg, msg, sizeof(errmsg)-1 );
        strncat( errmsg, " expected",
                  (Int)(sizeof(errmsg)-1-strlen(errmsg)) );
        SyntaxError( errmsg );
        while ( ! IS_IN( Symbol, skipto ) )
            GetSymbol();
    }

}


/****************************************************************************
**
*F  PutLine() . . . . . . . . . . . . . . . . . . . . . . print a line, local
**
**  'PutLine'  prints the current output line   'Output->line' to the current
**  output file 'Output->file'.  It  is  called from 'PutChr'.
**
**  'PutLine' also compares the output line with the next  line from the test
**  input file  'TestInput'  if 'TestInput' is  not -1.   If this  input line
**  starts with '#>' and the rest of  the  line matches  the output line then
**  the output line is not printed and the input line is discarded.
**
**  'PutLine'  also   echoes  the output  line to   the  logfile 'Logfile' if
**  'Logfile' is not -1 and the output file is '*stdout*' or '*errout*'.
**
**  Finally 'PutLine' checks whether the user has hit '<ctr>-C' to  interrupt
**  the printing.
*/
void            PutLine(void)
{
	Int 	plen, i, k;
	Bag	hd;
	char* str;
	Int	slen;

	if (Output->file == -5) {

		str = malloc((strlen(Output->line) + 1) * sizeof(char));
		i = 0;
		while (Output->line[i++] == ' ');
		--i;
		for (k = 0; (i < strlen(Output->line)) && (Output->line[i] != '\n'); ++i, ++k) {
			str[k] = Output->line[i];
		}
		str[k] = '\0';

		slen = strlen(str);

		hd = NewBag(T_STRING, slen + 1);

		*((char*)PTR_BAG(hd)) = '\0';
		strncpy((char*)PTR_BAG(hd), str, slen);

		plen = PLEN_SIZE_PLIST(GET_SIZE_BAG(Output->hdList));
		Resize(Output->hdList, SIZE_PLEN_PLIST(plen + 1));
		SET_LEN_PLIST(Output->hdList, plen + 1);
		SET_BAG(Output->hdList, plen + 1, hd);

		free(str);
	}
	else if (Output->file) {
		SyFputs(Output->line, Output->file);
	}
	else if (Output->mem) {
		/* write to memory */
		UInt len = strlen(Output->line);
		UInt mem_sz = ((UInt*)(Output->mem))[0];
		UInt mem_chrs = ((UInt*)(Output->mem))[1];
		if (mem_sz + len + 1 > mem_chrs) {
			/* no error handling */
			mem_sz = (mem_sz > len) ? mem_sz * 2 : (mem_sz + len + 1);
			Output->mem = realloc(Output->mem, 2 * sizeof(UInt) + mem_sz);
			((UInt*)(Output->mem))[0] = mem_sz;
		}
		strncpy(Output->mem + 2 * sizeof(UInt) + mem_chrs, Output->line, len + 1);
		((UInt*)(Output->mem))[1] = mem_chrs + len;
	}


	/* if neccessary echo it to the logfile                                */
	if (Logfile != -1 && (Output->file == 1 || Output->file == 3))
		SyFputs(Output->line, Logfile);
}


/****************************************************************************
**
*F  PutChr( <ch> )  . . . . . . . . . . . . . . . print character <ch>, local
**
**  'PutChr' prints the single character <ch> to the current output file.
**
**  'PutChr' buffers the  output characters until  either <ch> is  <newline>,
**  <ch> is '\03' (<flush>) or the buffer fills up.
**
**  In the later case 'PutChr' has to decide where to  split the output line.
**  It takes the point at which $linelength - pos + 8 * indent$ is minimal.
*/
void            PutChr (char ch)
{
    Int                i;
//    char                str [ 256 ];
    char s0, s1;
    int n;

    /* '\01', increment indentation level                                  */
    if ( ch == '\01' ) {

        /* if this is a better place to split the line remember it         */
        if ( (Output->file == 1 || Output->file == 3)
          && Output->indent < Output->pos
          && SyNrCols-Output->pos  + 16*Output->indent
          <= SyNrCols-Output->spos + 16*Output->sindent ) {
            Output->spos     = Output->pos;
            Output->sindent  = Output->indent;
        }

        Output->indent++;

    }

    /* '\02', decrement indentation level                                  */
    else if ( ch == '\02' ) {

        /* if this is a better place to split the line remember it         */
        if ( (Output->file == 1 || Output->file == 3)
          && Output->indent < Output->pos
          && SyNrCols-Output->pos  + 16*Output->indent
          <= SyNrCols-Output->spos + 16*Output->sindent ) {
            Output->spos     = Output->pos;
            Output->sindent  = Output->indent;
        }

        Output->indent--;

    }

    /* '\03', print line                                                   */
    else if ( ch == '\03' ) {

        /* print the line                                                  */
        Output->line[ Output->pos ] = '\0';
        PutLine();

        /* start the next line                                             */
        Output->pos      = 0;

        /* first character is a very bad place to split                    */
        Output->spos     = 0;
        Output->sindent  = 660;

    }

    /* <newline> or <return>, print line, indent next                      */
    else if ( ch == '\n' || ch == '\r' ) {

        /* put the character on the line and terminate it                  */
        Output->line[ Output->pos++ ] = ch;
        Output->line[ Output->pos   ] = '\0';

        /* print the line                                                  */
        PutLine();

        /* indent for next line                                            */
        Output->pos = 0;
        for ( i = 0;  i < Output->indent; i++ )
            Output->line[ Output->pos++ ] = ' ';

        /* set up new split positions                                      */
        Output->spos     = 0;
        Output->sindent  = 660;

    }

    /* normal character, room on the current line                          */
    else if ( Output->pos < SyNrCols-2 ) {

        /* put the character on this line                                  */
        Output->line[ Output->pos++ ] = ch;

    }

    /* if we are going to split at the end of the line, discard blanks     */
    else if ( Output->spos == Output->pos && ch == ' ' ) {
        ;
    }

    /* full line, acceptable split position                                */
    else if ( Output->spos != 0 ) {
        /* add character to the line, terminate it                         */
        Output->line[ Output->pos++ ] = ch;
        Output->line[ Output->pos++ ] = '\0';

        n = Output->pos - Output->spos;

        /* terminate the line while saving those chars */
        s0 = Output->line[Output->spos];
        s1 = Output->line[Output->spos+1];
        Output->line[ Output->spos ] = '\n';
        Output->line[ Output->spos+1 ] = '\0';
        PutLine();

        /* put the chars back */
        Output->line[Output->spos] = s0;
        Output->line[Output->spos+1] = s1;

        /* print line up to the best split position                        */
        /* indent for the rest                                             */
        assert(Output->sindent + n <= OutputLineLen);

        memmove(Output->line + Output->sindent, Output->line + Output->spos, n);

        Output->pos = 0;
        for ( i = 0; i < Output->sindent; i++ )
            Output->line[ Output->pos++ ] = ' ';

        Output->pos += n-1;

        /* set new split position                                          */
        Output->spos     = 0;
        Output->sindent  = 660;

    }

    /* full line, no splitt position                                       */
    else {

        /* append a '\', and print the line                                */
        if ( Output->file == 1 || Output->file == 3 ) {
            Output->line[ Output->pos++ ] = '\\';
            Output->line[ Output->pos++ ] = '\n';
        }
        Output->line[ Output->pos   ] = '\0';
        PutLine();

        /* add the character to the next line                              */
        Output->pos = 0;
        Output->line[ Output->pos++ ] = ch;

        /* the first character is a very bad place to split                */
        Output->spos     = 0;
        Output->sindent  = 660;

    }

}


/****************************************************************************
**
*F  Pr( <format>, <arg1>, <arg2> )  . . . . . . . . .  print formatted output
**
**  'Pr' is the output function. The first argument is a 'printf' like format
**  string containing   up   to 2  '%'  format   fields,   specifing  how the
**  corresponding arguments are to be  printed.  The two arguments are passed
**  as  'long'  integers.   This  is possible  since every  C object  ('int',
**  'char', pointers) except 'float' or 'double', which are not used  in GAP,
**  can be converted to a 'long' without loss of information.
**
**  The function 'Pr' currently support the following '%' format  fields:
**  '%g'    prints GAP object (expected Bag)
**  '%c'    the corresponding argument represents a character,  usually it is
**          its ASCII or EBCDIC code, and this character is printed.
**  '%s'    the corresponding argument is the address of  a  null  terminated
**          character string which is printed.
**  '%d'    the corresponding argument is a signed integer, which is printed.
**          Between the '%' and the 'd' an integer might be used  to  specify
**          the width of a field in which the integer is right justified.  If
**          the first character is '0' 'Pr' pads with '0' instead of <space>.
**  '%>'    increment the indentation level.
**  '%<'    decrement the indentation level.
**  '%%'    can be used to print a single '%' character. No argument is used.
**
**  You must always  cast the arguments to  '(long)' to avoid  problems  with
**  those compilers with a default integer size of 16 instead of 32 bit.  You
**  must pass 0L if you don't make use of an argument to please lint.
*/
void            Pr (char *format, Int arg1, Int arg2)
{
    char                * p,  * q;
    Int                prec,  n;
    char                fill;

    /* loop over the characters of the <format> string                     */
    for ( p = format; *p != '\0'; p++ ) {

        /* if the character is '%' do something special                    */
        if ( *p == '%' ) {

            /* first look for a precision field                            */
            p++;
            if ( *p == '0' )  fill = '0';  else fill = ' ';
            for ( prec = 0; IsDigit(*p); p++ )
                prec = 10 * prec + *p - '0';

            /* '%d' print an integer                                       */
            if ( *p == 'd' ) {
                if ( arg1 < 0 ) {
                    prec--;
                    for ( n=1; n <= -(arg1/10); n*=10 )
                        prec--;
                    while ( --prec > 0 )  PutChr(fill);
                    PutChr('-');
                    for ( ; n > 0; n /= 10 )
                        PutChr( (char)(-((arg1/n)%10) + '0') );
                    arg1 = arg2;
                }
                else {
                    for ( n=1; n<=arg1/10; n*=10 )
                        prec--;
                    while ( --prec > 0 )  PutChr(fill);
                    for ( ; n > 0; n /= 10 )
                        PutChr( (char)(((arg1/n)%10) + '0') );
                    arg1 = arg2;
                }
            }

            /* '%s' print a string                                         */
            else if ( *p == 's' ) {
                for ( q = (char*)arg1; *q != '\0'; q++ )
                    prec--;
                while ( prec-- > 0 )  PutChr(' ');
                for ( q = (char*)arg1; *q != '\0'; q++ )
                    PutChr( *q );
                arg1 = arg2;
            }

            /* '%g' print a GAP object                                     */
            else if ( *p == 'g' ) {
                Print( (Bag)arg1 );
                arg1 = arg2;
            }

            /* '%c' print a character                                      */
            else if ( *p == 'c' ) {
                PutChr( (char)arg1 );
                arg1 = arg2;
            }

            /* '%>' increment the indentation level                        */
            else if ( *p == '>' ) {
                PutChr( '\01' );
                while ( --prec > 0 )
                    PutChr( '\01' );
            }

            /* '%<' decrement the indentation level                        */
            else if ( *p == '<' ) {
                PutChr( '\02' );
                while ( --prec > 0 )
                    PutChr( '\02' );
            }

            /* '%%' print a '%' character                                  */
            else if ( *p == '%' ) {
                PutChr( '%' );
            }

            /* else raise an error                                         */
            else {
                for ( p = "%format error"; *p != '\0'; p++ )
                    PutChr( *p );
            }

        }

        /* not a '%' character, simply print it                            */
        else {
            PutChr( *p );
        }

    }
}


Int	ChDir(const char* filename)
{
	return (int)SyChDir(filename);
}


/****************************************************************************
**
*F  OpenInput( <filename> ) . . . . . . . . . .  open a file as current input
**
**  'OpenInput' opens  the file with  the name <filename>  as  current input.
**  All  subsequent input will  be taken from that  file, until it is  closed
**  again  with 'CloseInput'  or  another file  is opened  with  'OpenInput'.
**  'OpenInput'  will not  close the  current  file, i.e., if  <filename>  is
**  closed again, input will again be taken from the current input file.
**
**  'OpenInput'  returns 1 if  it   could  successfully open  <filename>  for
**  reading and 0  to indicate  failure.   'OpenInput' will fail if  the file
**  does not exist or if you do not have permissions to read it.  'OpenInput'
**  may  also fail if  you have too  many files open at once.   It  is system
**  dependent how many are  too many, but  16  files should  work everywhere.
**
**  Directely after the 'OpenInput' call the variable  'Symbol' has the value
**  'S_ILLEGAL' to indicate that no symbol has yet been  read from this file.
**  The first symbol is read by 'Read' in the first call to 'Match' call.
**
**  You can open  '*stdin*' to  read  from the standard  input file, which is
**  usually the terminal, or '*errin*' to  read from the standard error file,
**  which  is  the  terminal  even if '*stdin*'  is  redirected from  a file.
**  'OpenInput' passes those  file names  to  'SyFopen' like any other  name,
**  they are  just  a  convention between the  main  and the system  package.
**  'SyFopen' and thus 'OpenInput' will  fail to open  '*errin*' if the  file
**  'stderr'  (Unix file  descriptor  2)  is  not a  terminal,  because  of a
**  redirection say, to avoid that break loops take their input from a file.
**
**  It is not neccessary to open the initial input  file, 'InitScanner' opens
**  '*stdin*' for  that purpose.  This  file on   the other   hand can not be
**  closed by 'CloseInput'.
*/
Int            OpenInput (char *filename)
{
    Int                file;

    /* fail if we can not handle another open input file                   */
    if ( Input+1 == InputFiles+(sizeof(InputFiles)/sizeof(InputFiles[0])) )
        return 0;

    /**/HookBeforeOpenInput();/**/

    /* try to open the input file                                          */
    file = SyFopen( filename, "r" );
    if ( file == -1 )
        return 0;

    /* remember the current position in the current file                   */
    if ( Input != InputFiles-1 )
        Input->ptr = In;

    /* enter the file identifier and the file name                         */
    Input++;
    Input->file = file;
    Input->name[0] = '\0';
    strncat( Input->name, filename, sizeof(Input->name) );

    /* start with an empty line and no symbol                              */
    In = Input->line;
    In[0] = In[1] = '\0';
    Symbol = S_ILLEGAL;
    Input->number = 1;

    Input->package = 0;
    Input->packages = 0;
    Input->imports = 0;
    Input->importTop = 0;
    Input->packageTop = 0;
    Input->data = 0;
    Input->global = 1;

    /**/HookAfterOpenInput();/**/

    /* indicate success                                                    */
    return 1;
}


/****************************************************************************
**
*F  CloseInput()  . . . . . . . . . . . . . . . . .  close current input file
**
**  'CloseInput'  will close the  current input file.   Subsequent input will
**  again be taken from the previous input file.   'CloseInput' will return 1
**  to indicate success.
**
**  'CloseInput' will not close the initial input file '*stdin*', and returns
**  0  if such  an  attempt is made.   This is  used in  'Error'  which calls
**  'CloseInput' until it returns 0, therebye closing all open input files.
**
**  Calling 'CloseInput' if the  corresponding  'OpenInput' call failed  will
**  close the current output file, which will lead to very strange behaviour.
*/
Int            CloseInput (void)
{
    /* refuse to close the initial input file                              */
    if ( Input == InputFiles )
        return 0;

    /**/HookBeforeCloseInput();/**/

    /* close the input file                                                */
    SyFclose( Input->file );

    /* revert to last file                                                 */
    Input--;
    In = Input->ptr;

    /* indicate that the next symbol has not yet been read                 */
    Symbol = S_ILLEGAL;

    /**/HookAfterCloseInput();/**/

    /* indicate success                                                    */
    return 1;
}


extern Int            OpenStringOutput ()
{
    Int                file;

    /* fail if we can not handle another open output file                  */
    if ( Output+1==OutputFiles+(sizeof(OutputFiles)/sizeof(OutputFiles[0])) )
        return 0;

    /* put the file on the stack, start at position 0 on an empty line     */
    Output++;
    Output->file    = -5;
    Output->line[0] = '\0';
    Output->pos     = 0;
    Output->indent  = 0;
    Output->mem     = 0;

    Output->hdList = NewBag( T_LIST, ( 1 ) * SIZE_HD );


    /* variables related to line splitting, very bad place to split        */
    Output->spos    = 0;
    Output->sindent = 660;

    /* indicate success                                                    */
    return 1;
}


Bag		GReadFile()
{
	Bag	hd, hdList;
	Int	slen, plen;

	hdList = NewBag( T_LIST, ( 1 ) * SIZE_HD );

	while(SyFgets(Input->line, 2048, Input->file)) {
		slen = strlen(Input->line);
		Input->line[slen-1] = '\0';
		hd = NewBag( T_STRING, slen );

                *( (char*) PTR_BAG( hd ) ) = '\0';
                strncpy( (char*) PTR_BAG( hd ), Input->line, slen );

                plen = PLEN_SIZE_PLIST( GET_SIZE_BAG(hdList) );
                Resize( hdList, SIZE_PLEN_PLIST( plen + 1 ) );
                SET_LEN_PLIST( hdList, plen + 1 );
                SET_BAG( hdList ,  plen + 1 ,  hd );

	}

	return hdList;
}


/****************************************************************************
**
*F  OpenOutput( <filename> )  . . . . . . . . . open a file as current output
**
**  'OpenOutput' opens the file  with the name  <filename> as current output.
**  All subsequent output will go  to that file, until either   it is  closed
**  again  with 'CloseOutput' or  another  file is  opened with 'OpenOutput'.
**  The file is truncated to size 0 if it existed, otherwise it  is  created.
**  'OpenOutput' does not  close  the  current file, i.e., if  <filename>  is
**  closed again, output will go again to the current output file.
**
**  'OpenOutput'  returns  1 if it  could  successfully  open  <filename> for
**  writing and 0 to indicate failure.  'OpenOutput' will fail if  you do not
**  have  permissions to create the  file or write   to it.  'OpenOutput' may
**  also   fail if you   have  too many files   open  at once.   It is system
**  dependent how many are too many, but 16 files should work everywhere.
**
**  You can open '*stdout*'  to write  to the standard output  file, which is
**  usually the terminal, or '*errout*' to write  to the standard error file,
**  which is the terminal  even   if '*stdout*'  is  redirected to   a  file.
**  'OpenOutput' passes  those  file names to 'SyFopen'  like any other name,
**  they are just a convention between the main and the system package.
**
**  It is not neccessary to open the initial output file, 'InitScanner' opens
**  '*stdout*' for that purpose.  This  file  on the other hand   can not  be
**  closed by 'CloseOutput'.
*/
Int            OpenOutput (char *filename)
{
    Int                file;

    /* fail if we can not handle another open output file                  */
    if ( Output+1==OutputFiles+(sizeof(OutputFiles)/sizeof(OutputFiles[0])) )
        return 0;

    /* try to open the file                                                */
    file = SyFopen( filename, "w" );
    if ( file == -1 )
        return 0;

    /* put the file on the stack, start at position 0 on an empty line     */
    Output++;
    Output->file    = file;
    Output->line[0] = '\0';
    Output->pos     = 0;
    Output->indent  = 0;
    Output->mem     = 0;
    /* variables related to line splitting, very bad place to split        */
    Output->spos    = 0;
    Output->sindent = 660;

    /* indicate success                                                    */
    return 1;
}


/****************************************************************************
**
*F  CloseOutput() . . . . . . . . . . . . . . . . . close current output file
**
**  'CloseOutput' will  first flush all   pending output and  then  close the
**  current  output  file.   Subsequent output will  again go to the previous
**  output file.  'CloseOutput' returns 1 to indicate success.
**
**  'CloseOutput' will  not  close the  initial output file   '*stdout*', and
**  returns 0 if such attempt is made.  This  is  used in 'Error' which calls
**  'CloseOutput' until it returns 0, thereby closing all open output files.
**
**  Calling 'CloseOutput' if the corresponding 'OpenOutput' call failed  will
**  close the current output file, which will lead to very strange behaviour.
**  On the other  hand if you  forget  to call  'CloseOutput' at the end of a
**  'PrintTo' call or an error will not yield much better results.
*/
Int            CloseOutput (void)
{
    /* refuse to close the initial output file '*stdout*'                  */
    if ( Output == OutputFiles )
        return 0;


    /* flush output and close the file                                     */
    Pr( "%c", (Int)'\03', 0 );
    SyFclose( Output->file );

    /* revert to previous output file and indicate success                 */
    Output->file = 0;
    if (Output->mem) {
        free(Output->mem);
        Output->mem = 0;
    }
    
    Output--;
    return 1;
}



extern Bag            ReturnStringOutput ()
{
	Bag		hd;
	Int		plen, i, k, slen;
	char		*str;

	if (Output->pos > 0) {
		Output->line[Output->pos] = '\0';

		str = malloc((strlen(Output->line)+1)*sizeof(char));
		i = 0;
		while(Output->line[i++] == ' ');
		--i;
		for(k = 0; i < strlen(Output->line); ++i, ++k) {
			str[k] = Output->line[i];
		}
		str[k] = '\0';

		slen = strlen(str);

	        hd = NewBag( T_STRING, slen + 1 );

       		*( (char*) PTR_BAG( hd ) ) = '\0';
	        strncpy( (char*) PTR_BAG( hd ), str, slen );

        	plen = PLEN_SIZE_PLIST( GET_SIZE_BAG(Output->hdList) );
	        Resize( Output->hdList, SIZE_PLEN_PLIST( plen + 1 ) );
        	SET_LEN_PLIST( Output->hdList, plen + 1 );
	        SET_BAG( Output->hdList ,  plen + 1 ,  hd );

		free(str);
	}

	return Output->hdList;

}

extern Int            CloseStringOutput (void)
{
    /* refuse to close the initial output file '*stdout*'                  */
    if ( Output == OutputFiles )
        return 0;


    /* revert to previous output file and indicate success                 */
    Output->file = 0;
    if (Output->mem) {
        free(Output->mem);
        Output->mem = 0;
    }
    
    Output--;
    return 1;
}




/****************************************************************************
**
*F  OpenMemory( )  . . . . . . . . . . . redirecting output into memory block
**
**  'OpenMemory' uses memory buffer as current output.
**  All subsequent output will go to that memory buffer, until either   
**  it is  closed with 'CloseMemory'/'CloseOutput' or  another  file is  
**  opened with 'OpenOutput'. The size of memory buffer grows to hold all 
**  redirected data.
*/

Int            OpenMemory ()
{
    char* mem;
    /* fail if we can not handle another open output file                  */
    if ( Output+1==OutputFiles+(sizeof(OutputFiles)/sizeof(OutputFiles[0])) )
        return 0;

    /* alloc buffer, first UInt there is buffer size without two UInts     */
    /* second UInt is number of number of characters in the buffer         */
    mem = malloc(2*sizeof(UInt) + INITIAL_MEM_SIZE);

    if (mem==0)
        return 0;
        
    ((UInt*)mem)[0] = INITIAL_MEM_SIZE;
    ((UInt*)mem)[1] = 0;
    *(mem + 2*sizeof(UInt)) = '\0';
    /* put the file on the stack, start at position 0 on an empty line     */
    Output++;
    Output->file    = 0;
    Output->line[0] = '\0';
    Output->pos     = 0;
    Output->indent  = 0;
    Output->mem     = mem;

    /* variables related to line splitting, very bad place to split        */
    Output->spos    = 0;
    Output->sindent = 660;

    /* indicate success                                                    */
    return 1;
}

/****************************************************************************
**
*F  CloseMemory( )  . . . . . . . . . . . close current output memory buffer
**
**  CloseMemory(Obj* hdStr) closing current output and returns all 
**  accumulated output as string in hdStr.
*/

Int            CloseMemory(Obj* hdStr) {
    if (Output->mem) {
        Obj hd = 0;
        UInt mem_chars;
        /* flush output                                                    */
        Pr( "%c", (Int)'\03', 0 );
        mem_chars = ((UInt*)(Output->mem))[1];
        /* grab memory to string                                           */
        hd = NewBag( T_STRING, SIZEBAG_STRINGLEN(mem_chars)); 
        SET_LEN_STRING(hd, mem_chars);
        strncpy(CHARS_STRING(hd), Output->mem + 2*sizeof(UInt), mem_chars+1 );
        *hdStr = hd;
    } else {
        *hdStr = NewBag( T_STRING, SIZEBAG_STRINGLEN(0)); 
    }
    return CloseOutput();
}

/****************************************************************************
**
*F  OpenAppend( <filename> )  . . open a file as current output for appending
**
**  'OpenAppend' opens the file  with the name  <filename> as current output.
**  All subsequent output will go  to that file, until either   it is  closed
**  again  with 'CloseAppend' or  another  file is  opened with 'OpenOutput'.
**  Unlike 'OpenOutput' 'OpenAppend' does not truncate the file to size 0  if
**  it exists.  Appart from that 'OpenAppend' is equal to 'OpenOutput' so its
**  description applies to 'OpenAppend' too.
*/
Int            OpenAppend (char *filename)
{
    Int                file;

    /* fail if we can not handle another open output file                  */
    if ( Output+1==OutputFiles+(sizeof(OutputFiles)/sizeof(OutputFiles[0])) )
        return 0;

    /* try to open the file                                                */
    file = SyFopen( filename, "a" );
    if ( file == -1 )
        return 0;

    /* put the file on the stack, start at position 0 on an empty line     */
    Output++;
    Output->file    = file;
    Output->line[0] = '\0';
    Output->pos     = 0;
    Output->indent  = 0;
    Output->mem     = 0;
    
    /* variables related to line splitting, very bad place to split        */
    Output->spos    = 0;
    Output->sindent = 660;

    /* indicate success                                                    */
    return 1;
}


/****************************************************************************
**
*F  OpenLog( <filename> ) . . . . . . . . . . . . . log interaction to a file
**
**  'OpenLog'  instructs  the scanner to   echo  all  input   from  the files
**  '*stdin*' and  '*errin*'  and  all  output to  the  files '*stdout*'  and
**  '*errout*' to the file with  name <filename>.  The  file is truncated  to
**  size 0 if it existed, otherwise it is created.
**
**  'OpenLog' returns 1 if it could  successfully open <filename> for writing
**  and 0  to indicate failure.   'OpenLog' will  fail if  you do  not   have
**  permissions  to create the file or   write to  it.  'OpenOutput' may also
**  fail if you have too many files open at once.  It is system dependent how
**  many   are too   many, but  16   files should  work everywhere.   Finally
**  'OpenLog' will fail if there is already a current logfile.
*/
Int            OpenLog (char *filename)
{

    /* refuse to open a logfile if we already log to one                   */
    if ( Logfile != -1 )
        return 0;

    /* try to open the file                                                */
    Logfile = SyFopen( filename, "w" );
    if ( Logfile == -1 )
        return 0;

    /* otherwise indicate success                                          */
    return 1;
}


/****************************************************************************
**
*F  CloseLog()  . . . . . . . . . . . . . . . . . . close the current logfile
**
**  'CloseLog' closes the current logfile again, so that input from '*stdin*'
**  and '*errin*' and output to '*stdout*' and '*errout*' will no  longer  be
**  echoed to a file.  'CloseLog' will return 1 to indicate success.
**
**  'CloseLog' will fail if there is no logfile active and will return  0  in
**  this case.
*/
Int            CloseLog (void)
{
    /* refuse to close a non existent logfile                              */
    if ( Logfile == -1 )
        return 0;

    /* close the logfile                                                   */
    SyFclose( Logfile );
    Logfile = -1;

    /* indicate success                                                    */
    return 1;
}


/****************************************************************************
**
*F  OpenInputLog( <filename> )  . . . . . . . . . . . . . log input to a file
**
**  'OpenInputLog'  instructs the  scanner  to echo  all input from the files
**  '*stdin*' and  '*errin*' to the file  with  name <filename>.  The file is
**  truncated to size 0 if it existed, otherwise it is created.
**
**  'OpenInputLog' returns 1  if it  could successfully open  <filename>  for
**  writing  and  0 to indicate failure.  'OpenInputLog' will fail  if you do
**  not have  permissions to create the file  or write to it.  'OpenInputLog'
**  may also fail  if you  have  too many  files open  at once.  It is system
**  dependent  how many are too many,  but 16 files  should work  everywhere.
**  Finally 'OpenInputLog' will fail if there is already a current logfile.
*/
Int            OpenInputLog (char *filename)
{

    /* refuse to open a logfile if we already log to one                   */
    if ( InputLogfile != -1 )
        return 0;

    /* try to open the file                                                */
    InputLogfile = SyFopen( filename, "w" );
    if ( InputLogfile == -1 )
        return 0;

    /* otherwise indicate success                                          */
    return 1;
}


/****************************************************************************
**
*F  CloseInputLog() . . . . . . . . . . . . . . . . close the current logfile
**
**  'CloseInputLog'  closes  the current  logfile again,  so  that input from
**  '*stdin*'  and   '*errin*'  will  no  longer   be  echoed   to  a   file.
**  'CloseInputLog' will return 1 to indicate success.
**
**  'CloseInputLog' will fail if there is no logfile active and will return 0
**  in this case.
*/
Int            CloseInputLog (void)
{
    /* refuse to close a non existent logfile                              */
    if ( InputLogfile == -1 )
        return 0;

    /* close the logfile                                                   */
    SyFclose( InputLogfile );
    InputLogfile = -1;

    /* indicate success                                                    */
    return 1;
}



/****************************************************************************
**
*F  InitScanner() . . . . . . . . . . . . . .  initialize the scanner package
**
**  'InitScanner' initializes  the  scanner  package.  This  justs  sets  the
**  current input file to '*stdin*' and current output  file  to  '*stdout*'.
*/
void            InitScanner (void)
{
    Int                ignore, i;

    OutputLineLen = SyNrCols < 1000 ? 1000 : SyNrCols + 1;
    for(i = 0; i < SCANNER_OUTPUTS; ++i) {
        /* OutputFiles[i].line = (char*) xmalloc(sizeof(char)*OutputLineLen); */
        OutputFiles[i].line = (char*) malloc(sizeof(char)*OutputLineLen);
        OutputFiles[i].file = 0;
        OutputFiles[i].indent = 0;
        OutputFiles[i].pos = 0;
        OutputFiles[i].sindent = 0;
        OutputFiles[i].spos = 0;
        OutputFiles[i].mem = 0;
    }

    Input  = InputFiles-1;   ignore = OpenInput(  "*stdin*"  );
    Output = OutputFiles-1;  ignore = OpenOutput( "*stdout*" );

    Logfile = -1;  
    InputLogfile = -1;
}



