%{
#include <string.h>
#include        "system.h"              /* system dependent functions      */
#include        "gasman.h"
#include        "scanner.h"             /* definition part of this package */
#include        "comments.h"
#include <sys_conf/xmalloc.h>
#define malloc xmalloc
#define free xfree
#define realloc xrealloc
  extern int GAP_SILENT;

static inline char decode_backslashed(char ch) {
    switch(ch) {
    case 'n': return '\n';
    case 't': return '\t';
    case 'r': return '\r';
    case 'b': return '\b';
    case 'c': return '\03';
    default: return ch;
    }
}

static void fill_Value(char *src, int maxlen) {
    char * val, * last;
    if(maxlen >= sizeof(Value)/sizeof(char))
        maxlen = sizeof(Value)/sizeof(char) - 1;
    last = Value + maxlen;
    for(val = Value; val != last; ++val) {
        char src_ch = *src;
	/* do not parse escaped characters here, fill_Value is not used for strings */
        if(src_ch == '\\') {
            ++src;
            src_ch = (*src);  
        }
        *val = src_ch;
        ++src;
    }
    *val = '\0';
}

%}

digit                [0-9]
integer              [0-9]([0-9]|\\[0-9])*
ident_char           [a-zA-Z_0-9@]
exponent             (e|E)(\+|-)?[0-9]+
whitespace           [ \n\r\t]
eof                  \377

%x comment

%x str

%%

%{

#define YY_INPUT(buf,result,max_size) { \
       int len; \
       In=In+1; \
       if(*In=='\0') GetLine(); \
       len = strlen(In); \
       memcpy(buf, In, len); \
       if(buf[len-1]=='\377') result = YY_NULL; \
       else                   result = len;\
       In=In+len-1;\
    }

  char *string_buf_ptr;
  char *string_buf_end;
%}


{whitespace}+ { /* ignore */ }

"<#"               { BEGIN(comment); AppendCommentBuffer(yytext, strlen(yytext)); }
<comment>[^#]*     { AppendCommentBuffer(yytext, strlen(yytext)); }
<comment>"#"+[^>]  { AppendCommentBuffer(yytext, strlen(yytext)); }
<comment>#+\>      { AppendCommentBuffer(yytext, strlen(yytext)); BEGIN(INITIAL); }

\#.*\n  { /* single line comment */
          AppendCommentBuffer(yytext, strlen(yytext)); }

\#.*  { /* single line comment */
  AppendCommentBuffer(yytext, strlen(yytext)); }


\"      { BEGIN(str);
          string_buf_ptr = Value;
          string_buf_end = Value + (sizeof(Value)/sizeof(char) - 1);
        }

<str>\" { /* saw closing quote, all done */
          BEGIN(INITIAL);
          *string_buf_ptr = '\0';
          return S_STRING; }

<str>\n { SyntaxError("unterminated string constant");
          BEGIN(INITIAL);
          *string_buf_ptr = '\0';
          return S_STRING;  }

<str>\\(.|\n)  { if ( string_buf_ptr != string_buf_end ) {
                     *string_buf_ptr++ = decode_backslashed(yytext[1]);
                     if ( string_buf_ptr == string_buf_end ) {
                         SyntaxError("string is too long");
                         BEGIN(INITIAL);
                     }
                 }
               }

<str>[^\\\n\"]+  { char *yptr = yytext;
                   if ( string_buf_ptr != string_buf_end ) {
                       do *string_buf_ptr++ = *yptr++;
                       while ( *yptr && string_buf_ptr != string_buf_end );
                       if ( string_buf_ptr == string_buf_end) {
                           SyntaxError("string is too long");
                           BEGIN(INITIAL);
                       }
                   }
                 }

\:\: { return S_CONCAT; }
and    { return S_AND; }
do     { return S_DO; }
elif   { return S_ELIF; }
else   { return S_ELSE; }
end    { return S_END; }
fi     { return S_FI; }
for    { return S_FOR; }
function { return S_FUNCTION; }
if     { return S_IF; }
in     { return S_IN; }
_is    { return S_IS; }
local  { return S_LOCAL; }
mod    { return S_MOD; }
meth   { return S_METHOD; }
not    { return S_NOT; }
od     { return S_OD; }
or     { return S_OR; }
repeat { return S_REPEAT; }
return { return S_RETURN; }
then   { return S_THEN; }
until  { return S_UNTIL; }
while  { return S_WHILE; }
quit   { return S_QUIT; }

\>\> { return S_MAPTO_METH; }
\-\> { return S_MAPTO; }
\=\> { return S_ASSIGN_MAP; }
\:\= { return S_ASSIGN; }

\<=  { return S_LE; }
\>=  { return S_GE; }
\<\> { return S_NE; }
\<   { return S_LT; }
\>   { return S_GT; }
\=   { return S_EQ; }

;/; { GAP_SILENT=1; return S_SEMICOLON; }
;   { GAP_SILENT=0; return S_SEMICOLON; }

\.\.\. { fill_Value(yytext, strlen(yytext));
         return S_IDENT; }
\.\. { return S_DOTDOT; }
\$ { return S_BACKQUOTE; }
\+ { return S_PLUS; }
\- { return S_MINUS; }
\* { return S_MULT; }
\/ { return S_DIV; }
\^ { return S_POW; }
\, { return S_COMMA; }
\. { return S_DOT; }
\[ { return S_LBRACK; }
\] { return S_RBRACK; }
\( { return S_LPAREN; }
\) { return S_RPAREN; }
\{ { return S_LBRACE; }
\} { return S_RBRACE; }
\~ { Value[0]='~'; Value[1]='\0'; return S_IDENT; }

\'[^\'\n\\]\' { Value[0] = yytext[1]; Value[1]='\0';
              return S_CHAR; }

\'\\[^\n]\' { Value[0] = decode_backslashed(yytext[2]); Value[1]='\0';
                return S_CHAR; }

\'[^\'\n]+\' { Value[0] = yytext[1]; Value[1]='\0';
               SyntaxError("misplaced single-quote in character constant");
               return S_CHAR; }

{integer}    { fill_Value(yytext, strlen(yytext));
               return S_INT; }

%{/*0x[0-9a-fA-F]+ { i = strtol(yytext,0,0); return S_INT; }*/

/*0X[0-9a-fA-F]+ { i = strtol(yytext,0,0); return S_INT; }*/
    %}

{integer}+(\.{integer})?{exponent}?   { fill_Value(yytext, strlen(yytext));
                                      return S_DOUBLE; }

({ident_char}|(\\.))+ { fill_Value(yytext, strlen(yytext));
                return S_IDENT; }

{eof} { *In = '\0'; /*fprintf(stderr, "matched eof."); */ return S_EOF; }
<<EOF>> { *In = '\0'; /*fprintf(stderr, "matched eof.");*/ return S_EOF; }

. { Value[0] = yytext[0]; Value[1] = '\0';  return S_ILLEGAL; }


%%

int yywrap() {
    return 1;
}
