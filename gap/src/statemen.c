/****************************************************************************
**
*A  statemen.c                  GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This module contains the functions for executing the various  statements.
**  Assignments are dealed with in 'eval.c' and functions  are  in  'func.c'.
**
*/

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "scanner.h"             /* reading of tokens and printing  */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */

#include        "list.h"                /* generic list package            */
#include        "debug.h"               /* debugging, for user interrupts  */
#include        "statemen.h"            /* definition part of this module  */

/****************************************************************************
**
*F  EvStatseq( <hdSSeq> ) . . . . . . . . . . .  execute a statement sequence
**
**  'EvStatseq' executes the statement sequence <hdSSeq>.
**
**  This is   done by executing  the   <statement> one  after another.   If a
**  'return <expr>;' is executed inside the  statement sequence the execution
**  of the statement sequence  terminates and <expr>  is returned.  Otherwise
**  'HdVoid' is returned after execution of the last statement.
**
**  A statement sequence with <n> statements is represented by a bag with <n>
**  handles, the first is the handle of the first <statement>, the second  is
**  the handle of the second <statement>, and so on.
*/
Bag       EvStatseq (Bag hdSSeq)
{
    Bag           hdRes = 0;
    UInt       k;

    /* execute the <statement> one after the other                         */
    for ( k = 0; k < GET_SIZE_BAG(hdSSeq)/SIZE_HD; ++k ) {
        hdRes = EVAL( PTR_BAG(hdSSeq)[k] );
        if ( hdRes == HdReturn )  return hdRes;
        hdRes = 0;
    }

    return HdVoid;
}


/****************************************************************************
**
*F  EvIf( <hdIf> )  . . . . . . . . . . . . . . . . . execute an if statement
**
**  'EvIf' executes the 'if' statement <hdIf>.
**
**  This  is done  by   evaluating the <conditions>  until  one  evaluates to
**  'true'.   Then the    corresponding <statements>  are  executed.   If  no
**  <condition> evaluates to 'true' the 'else'  <statements> are executed, if
**  present.  If  a 'return  <expr>;' statement is   executed inside the 'if'
**  statement the execution of the 'if' statement is terminated and <expr> is
**  returned, otherwise 'HdVoid' is returned.
**
**  An 'if' statement is represented by  a bag where the  first handle is the
**  handle of the  <condition> following the  'if', the second  handle is the
**  handle of the corresponding <statements>,  the third handle is the handle
**  of the <condition> following the  first 'elif', the  fourth handle is the
**  handle  of  the  corresponding <statements>,  and   so  on.  If  the 'if'
**  statement  has no 'else'  part  the bag has   an even number of  handles,
**  otherwise the number of handles is odd and the last  handle is the handle
**  of the <statements> following the 'else'.
*/
Bag       EvIf (Bag hdIf)
{
    Bag           hdRes = 0,  hdSSeq = 0;
    UInt       i,  k = 0;

    /* handle the 'if'/'elif' branches in order                            */
    for ( i = 0; i < GET_SIZE_BAG(hdIf) / (2*SIZE_HD); ++i ) {

        /* evaluate the <condition>                                        */
        hdRes = EVAL( PTR_BAG(hdIf)[2*i] );
        while ( hdRes != HdTrue && hdRes != HdFalse )
          hdRes = Error("if: <expr> must evaluate to 'true' or 'false'",0,0);

        /* if 'true', execute the <statements> and terminate               */
        if ( hdRes == HdTrue ) {
            hdSSeq = PTR_BAG(hdIf)[2*i+1];
            if ( GET_TYPE_BAG(hdSSeq) == T_STATSEQ ) {
                for ( k = 0; k < GET_SIZE_BAG(hdSSeq)/SIZE_HD; ++k ) {
                    hdRes = EVAL( PTR_BAG(hdSSeq)[k] );
                    if ( hdRes == HdReturn )  return hdRes;
                    hdRes = 0;
                }
            }
            else {
                hdRes = EVAL( hdSSeq );
                if ( hdRes == HdReturn )  return hdRes;
                hdRes = 0;
            }
            return HdVoid;
        }
    }

    /* if present execute the 'else' <statements> and return               */
    if ( GET_SIZE_BAG(hdIf) % (2*SIZE_HD) != 0 ) {
        hdSSeq = PTR_BAG(hdIf)[GET_SIZE_BAG(hdIf)/SIZE_HD-1];
        if ( GET_TYPE_BAG(hdSSeq) == T_STATSEQ ) {
            for ( k = 0; k < GET_SIZE_BAG(hdSSeq)/SIZE_HD; ++k ) {
                hdRes = EVAL( PTR_BAG(hdSSeq)[k] );
                if ( hdRes == HdReturn )  return hdRes;
                hdRes = 0;
            }
        }
        else {
            hdRes = EVAL( hdSSeq );
            if ( hdRes == HdReturn )  return hdRes;
        }
        return HdVoid;
    }

    return HdVoid;
}


/****************************************************************************
**
*F  EvFor( <hdFor> )  . . . . . . . . . . . . . . . . execute a for statement
**
**  'EvFor' executes the 'for' loop <hdFor>.
**
**  This is  done by evaluating   <list> and executing the <statements>  with
**  <variable> bound to the  first element  of the  list, then executing  the
**  <statements> with <variable> bound to the next element of the list and so
**  on.  Unbound entries in the list are skipped.  If  new elements are added
**  to the end of  <list> during a loop  iteration they will be iterated over
**  too.  If   a  'return <expr>;'  is executed  inside   the 'for'  loop the
**  execution of the 'for' loop terminates and <expr> is returned.  Otherwise
**  'HdVoid' is returned after execution of the last statement.
**
**  The 'for' loop  is  represented by a  bag  with three handles, the  first
**  handle is the handle  of the <variable>,  the second handle is the handle
**  of the <list> and the third is the handle of the <statements>.
*/
Bag       EvFor (Bag hdFor)
{
    Bag      hdList,  hdRes = 0,  hdVar,  hdSSeq,  hdElm = 0;
    Int           lo = 0, hi = 0;
    Int           i = 0,  k = 0;

    /* first evaluate the <list> we are to loop over                       */
    hdVar  = PTR_BAG(hdFor)[0];
    hdSSeq = PTR_BAG(hdFor)[2];
    hdList = PTR_BAG(hdFor)[1];

    /* special case for a range that appear as constant in the text        */
    /*N 1992/12/16 martin handle general range literals                    */
    if ( GET_TYPE_BAG(hdList) == T_MAKERANGE && GET_SIZE_BAG(hdList) == 2*SIZE_HD ) {

        /* get the low and the high value of the range                     */
        hdElm = EVAL( PTR_BAG(hdList)[0] );
        while ( GET_TYPE_BAG(hdElm) != T_INT )
            hdElm = Error("Range: <low> must be an integer",0,0);
        hdList = EVAL( PTR_BAG(hdList)[1] );
        while ( GET_TYPE_BAG(hdList) != T_INT )
            hdList = Error("Range: <high> must be an integer",0,0);

        lo = HD_TO_INT(hdElm);
        hi = HD_TO_INT(hdList);
        /* loop over the range                                             */
        for ( i = lo; i <= hi; i++ ) {

            /* assign the element of the range to the variable             */
            SET_BAG(hdVar, 0,  INT_TO_HD( i ) );

            /* now execute the <statements>                                */
            if ( GET_TYPE_BAG(hdSSeq) == T_STATSEQ ) {
                for ( k = 0; k < GET_SIZE_BAG(hdSSeq)/SIZE_HD; ++k ) {
                    hdRes = EVAL( PTR_BAG(hdSSeq)[k] );
                    if ( hdRes == HdReturn ) {
                        return hdRes;
                    }
                    hdRes = 0;
                }
            }
            else {
                hdRes = EVAL( hdSSeq );
                if ( hdRes == HdReturn ) {
                    return hdRes;
                }
            }

            /* give the user the chance to interrupt this loop             */
            if ( SyIsIntr() )  DbgBreak("user interrupt", 0, 0);
        }

    }

    /* general case                                                        */
    else {

        /* evaluate the list                                               */
        hdList = EVAL( hdList );
        while ( ! IS_LIST( hdList ) )
            hdList = Error("for: <list> must evaluate to a list",0,0);

        /* loop over all elements in the list                              */
        /* note that the type of the list may dynamically change in loop   */
        for ( i = 1; 1; ++i ) {

            /* get the <i>th element, break if we have reached the end     */
            if ( LEN_LIST(hdList) < i )  break;
            hdElm = ELMF_LIST( hdList, i );
            if ( hdElm == 0 )  continue;
            SET_BAG(hdVar, 0,  hdElm );

            /* now execute the <statements>                                */
            if ( GET_TYPE_BAG(hdSSeq) == T_STATSEQ ) {
                for ( k = 0; k < GET_SIZE_BAG(hdSSeq)/SIZE_HD; ++k ) {
                    hdRes = EVAL( PTR_BAG(hdSSeq)[k] );
                    if ( hdRes == HdReturn ) {
                        return hdRes;
                    }
                    hdRes = 0;
                }
            }
            else {
                hdRes = EVAL( hdSSeq );
                if ( hdRes == HdReturn ) {
                    return hdRes;
                }
            }

            /* give the user the chance to interrupt this loop             */
            if ( SyIsIntr() )  DbgBreak("user interrupt", 0, 0);
        }

    }

    /* and thats it                                                        */
    return HdVoid;
}


/****************************************************************************
**
*F  EvWhile( <hdWhile> )  . . . . . . . . . . . . . execute a while statement
**
**  'EvWhile' executes the 'while' loop <hdWhile>.
**
**  This is   done by   executing  the  <statements> while  the   <condition>
**  evaluates  to  'true'.  If   a 'return  <expr>;' is executed   inside the
**  'while' loop the  execution of the  'while' loop terminates and <expr> is
**  returned.  Otherwise  'HdVoid' is returned  after  execution of the  last
**  statement.
**
**  The 'while' loop is represented by a bag  with  two  handles,  the  first
**  handle is the handle of the <condition>, the second handle is the  handle
**  of the <statements>.
*/
Bag       EvWhile (Bag hdWhile)
{
    Bag           hdRes = 0,  hdCond,  hdSSeq;
    UInt       k = 0;

    /* get the handles                                                     */
    hdCond = PTR_BAG(hdWhile)[0];
    hdSSeq = PTR_BAG(hdWhile)[1];

    /* evaluate the <condition> for the first iteration                    */
    hdRes = EVAL( hdCond );
    while ( hdRes != HdTrue && hdRes != HdFalse )
      hdRes = Error("while: <expr> must evalate to 'true' or 'false'",0,0);
    if ( SyIsIntr() )  DbgBreak("user interrupt", 0, 0);

    /* while <condition>  do                                               */
    while ( hdRes == HdTrue ) {

        /* execute the <statements>                                        */
        if ( GET_TYPE_BAG(hdSSeq) == T_STATSEQ ) {
            for ( k = 0; k < GET_SIZE_BAG(hdSSeq)/SIZE_HD; ++k ) {
                hdRes = EVAL( PTR_BAG(hdSSeq)[k] );
                if ( hdRes == HdReturn ) {
                    return hdRes;
                }
                hdRes = 0;
            }
        }
        else {
            hdRes = EVAL( hdSSeq );
            if ( hdRes == HdReturn ) {
                return hdRes;
            }
            hdRes = 0;
        }

        /* evaluate the <condition> for the next iteration                 */
        hdRes = EVAL( hdCond );
        while ( hdRes != HdTrue && hdRes != HdFalse )
            hdRes=Error("while: <expr> must evaluate to 'true' or 'false'",
                        0,0);
        if ( SyIsIntr() )  DbgBreak("user interrupt", 0, 0);

    }

    return HdVoid;
}


/****************************************************************************
**
*F  EvRepeat( <hdRep> ) . . . . . . . . . . . . . . . . execute a repeat loop
**
**  'EvRepeat' executes the 'repeat until' loop <hdRep>.
**
**  This   is done by executing    the  <statements>  until the   <condition>
**  evaluates   to  'true'.  If a   'return <expr>;'  is executed  inside the
**  'repeat' loop the execution of the 'repeat' loop terminates and <expr> is
**  returned.    Otherwise 'HdVoid' is returned after   execution of the last
**  statement.
**
**  The 'repeat' loop is represented by a bag  with two  handles,  the  first
**  handle is the handle of the <condition>, the second handle is the  handle
**  of the <statements>.
*/
Bag       EvRepeat (Bag hdRep)
{
    Bag           hdRes = 0, hdCond, hdSSeq;
    UInt       k = 0;

    /* get the handles                                                     */
    hdCond = PTR_BAG(hdRep)[0];
    hdSSeq = PTR_BAG(hdRep)[1];

    /* repeat the <statements> until the <condition> is 'true'             */
    do {
        /* execute the <statements>                                        */
        if ( GET_TYPE_BAG(hdSSeq) == T_STATSEQ ) {
            for ( k = 0; k < GET_SIZE_BAG(hdSSeq)/SIZE_HD; ++k ) {
                hdRes = EVAL( PTR_BAG(hdSSeq)[k] );
                if ( hdRes == HdReturn ) {
                    return hdRes;
                }
                hdRes = 0;
            }
        }
        else {
            hdRes = EVAL( hdSSeq );
            if ( hdRes == HdReturn ) {
                return hdRes;
            }
            hdRes = 0;
        }

        /* evaluate the <condition>                                        */
        hdRes = EVAL( hdCond );
        while ( hdRes != HdTrue && hdRes != HdFalse )
            hdRes=Error("repeat: <expr> must evaluate to 'true' or 'false'",
                        0,0);
        if ( SyIsIntr() )  DbgBreak("user interrupt", 0, 0);

    } while ( hdRes != HdTrue );

    return HdVoid;
}


/****************************************************************************
**
*F  PrStatseq( <hdSSeq> ) . . . . . . . . . . . .  print a statement sequence
**
**  'PrStatseq' prints the statement sequence <hdSSeq>.
**
**  A linebreak is forced after each <statement> except the last one.
*/
void            PrStatseq (Bag hdSSeq)
{
    UInt       k;

    /* print the <statements> one after another, separated by linebreaks   */
    for ( k = 0; k < GET_SIZE_BAG(hdSSeq)/SIZE_HD; ++k ) {
        Print( PTR_BAG(hdSSeq)[k] );
        if ( k < GET_SIZE_BAG(hdSSeq)/SIZE_HD-1 )
            Pr(";\n",0,0);
    }
}


/****************************************************************************
**
*F  PrIf( <hdIf> )  . . . . . . . . . . . . . . . . . . print an if statement
**
**  'PrIf' prints the 'if' statement <hdIf>.
**
**  A Linebreak is forced after the 'then'  and  <statements>.  If  necessary
**  it is preferred immediately before the 'then'.
*/
void            PrIf (Bag hdIf)
{
    UInt       i;

    /* print the 'if' and 'elif' parts                                     */
    for ( i = 0; i < GET_SIZE_BAG(hdIf)/SIZE_HD/2; ++i ) {
        if ( i == 0 ) Pr("if%4> ",0,0);  else Pr("elif%4> ",0,0);
        Print( PTR_BAG(hdIf)[2*i] );
        Pr("%2<  then%2>\n",0,0);
        Print( PTR_BAG(hdIf)[2*i+1] );
        Pr(";%4<\n",0,0);
    }

    /* print the 'else' part if it exists                                  */
    if ( GET_SIZE_BAG(hdIf)/SIZE_HD % 2 != 0 ) {
        Pr("else%4>\n",0,0);
        Print( PTR_BAG(hdIf)[ GET_SIZE_BAG(hdIf)/SIZE_HD -1 ] );
        Pr(";%4<\n",0,0);
    }

    /* print the 'fi'                                                      */
    Pr("fi",0,0);
}


/****************************************************************************
**
*F  PrFor( <hdFor> )  . . . . . . . . . . . . . . . . . . .  print a for loop
**
**  'PrFor' prints the 'for' loop <hdFor>.
**
**  A linebreak is forced after the 'do' and the <statements>.  If  necesarry
**  it is preferred immediately before the 'in'.
*/
void            PrFor (Bag hdFor)
{
    Pr("for%4> ",0,0);       Print( PTR_BAG(hdFor)[0] );
    Pr("%2<  in%2> ",0,0);   Print( PTR_BAG(hdFor)[1] );
    Pr("%2<  do%2>\n",0,0);  Print( PTR_BAG(hdFor)[2] );
    Pr(";%4<\nod",0,0);
}


/****************************************************************************
**
*F  PrWhile( <hdWhile> )  . . . . . . . . . . . . . . . .  print a while loop
**
**  'PrWhile' prints the 'while' loop <hdWhile>.
**
**  A linebreak is forced after the 'do' and the <statements>.  If  necessary
**  it is preferred immediately before the 'do'.
*/
void            PrWhile (Bag hdWhile)
{
    Pr("while%4> ",0,0);     Print( PTR_BAG(hdWhile)[0] );
    Pr("%2<  do%2>\n",0,0);  Print( PTR_BAG(hdWhile)[1] );
    Pr(";%4<\nod",0,0);
}


/****************************************************************************
**
*F  PrRepeat( <hdRep> ) . . . . . . . . . . . . . . . . . print a repeat loop
**
**  'PrRepeat' prints the 'repeat until' loop <hdRep>.
**
**  A linebreak is forced after the 'repeat' and the <statements>.
*/
void            PrRepeat (Bag hdRep)
{
    Pr("repeat%4>\n",0,0);
    Print( PTR_BAG(hdRep)[1] );
    Pr(";%4<\nuntil%2> ",0,0);
    Print( PTR_BAG(hdRep)[0] );
    Pr("%2<",0,0);
}


/****************************************************************************
**
*F  InitStat()  . . . . . . . . . . . . . . . initialize the statement module
**
**  Is called from 'InitEval' to initialize the statement evaluation  module.
*/
void            InitStat (void)
{
    InstEvFunc( T_STATSEQ,  EvStatseq );
    InstEvFunc( T_IF,       EvIf      );
    InstEvFunc( T_FOR,      EvFor     );
    InstEvFunc( T_WHILE,    EvWhile   );
    InstEvFunc( T_REPEAT,   EvRepeat  );

    InstPrFunc( T_STATSEQ,  PrStatseq );
    InstPrFunc( T_IF,       PrIf      );
    InstPrFunc( T_FOR,      PrFor     );
    InstPrFunc( T_WHILE,    PrWhile   );
    InstPrFunc( T_REPEAT,   PrRepeat  );
}


/****************************************************************************
**
*E  Emacs . . . . . . . . . . . . . . . . . . . . . . . local emacs variables
**
**  Local Variables:
**  outline-regexp:     "*A\\|*F\\|*V\\|*T\\|*E"
**  fill-column:        73
**  fill-prefix:        "**  "
**  End:
*/



