/****************************************************************************
**
*A  read.c                      GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2020, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2020) by the GAP Group (www.gap-system.org).
**
**  This module contains the functions to read  expressions  and  statements.
**
**
*N  19-Jun-90 martin ';' should belong to statements, not to sequences
*/

#include		<stdio.h>
#include		<stdlib.h>
#include		<string.h>

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "scanner.h"             /* reading of single symbols       */
#include        "idents.h"              /* identifier table manager        */
#include        "integer.h"             /* arbitrary size integers         */
#include        "integer4.h"            /* SumInt / ProdInt */

#include        "eval.h"                /* 'EVAL', 'HdVoid', 'HdReturn'    */
#include        "function.h"            /* NUM_ARGS_FUNC, NUM_LOCALS_FUNC  */
#include        "read.h"                /* definition part of this module  */
#include        "objects.h"
#include		"string4.h"
#include        "comments.h"            /* AppendCommentBuffer, etc        */
#include        "plist.h"
#include		"GapUtils.h"
#include        "namespaces_bin.h"

extern UInt TopStack;

/****************************************************************************
**
**  The constructs <Expr> and <Statments> may have themself as subpart, e.g.,
**  '<Expr>+1' is <Expr> and 'if <Expr> then <Statments> fi;' is <Statments>.
**  The functions 'RdExpr' and 'RdStats' must therefor be declared forward.
*/
Bag       RdExpr ( TypSymbolSet follow );
Bag       RdStats ( TypSymbolSet follow );


/****************************************************************************
**
*F  BinBag( <type>, <hdL>, <hdR> )  . . . . . . . . . . . . make a binary bag
**
**  'BinBag' makes a new bag of the type <type> with the  two  objects  <hdL>
**  and <hdR>.  No bag is made if an error has occured  during  the  parsing.
*/
Obj  BinBag ( unsigned int type, Obj hdL, Obj hdR ) {
    Obj hdRes;
    if ( NrError >= 1 )  return 0;
    hdRes = NewBag(type, 2 * SIZE_HD);
    SET_BAG(hdRes, 0,  hdL );  SET_BAG(hdRes, 1,  hdR );
    return hdRes;
}

Obj  UniBag ( unsigned int type, Obj hd0 ) {
    Obj hdRes;
    if ( NrError >= 1 )  return 0;
    hdRes = NewBag(type, SIZE_HD);
    SET_BAG(hdRes, 0,  hd0 );
    return hdRes;
}

Obj  TriBag ( unsigned int type, Obj hd0, Obj hd1, Obj hd2 ) {
    Obj hdRes;
    if ( NrError >= 1 )  return 0;
    hdRes = NewBag(type, 3 * SIZE_HD);
    SET_BAG(hdRes, 0,  hd0 );  SET_BAG(hdRes, 1,  hd1 );  SET_BAG(hdRes, 2,  hd2 );
    return hdRes;
}

/****************************************************************************
**
*F  RdVar( <follow> ) . . . . . . . . . . . . . . . . . . . . read a variable
**
**  'RdVar' reads a variable and returns the handle to the newly created bag.
**  A variable is something that is a legal left hand side in an  assignment.
**  In case of an error it skips all symbols up to one contained in <follow>.
**
**      <Ident>         :=  a|b|..|z|A|B|..|Z { a|b|..|z|A|B|..|Z|0|..|9|_ }
**
**      <Var>           :=  <Ident>
**                      |   <Var> '.' <Ident>
**                      |   <Var> '[' <Expr> ']'
**                      |   <Var> '{' <Expr> '}'
**                      |   <Var> '(' [ <Expr> { ',' <Expr> } ] ')'
*/
/*V HdCurLHS */
Bag       HdCurLHS;

/* When set to 1, no warnings relating to undefined global variables are   *
 * produced                                                                */
static int NoWarnUndefined = 0;
static int PermBegin = 0;

Bag       RdVar (UInt backq, TypSymbolSet follow)
{
    Bag           hdVar,  hd;
    Bag           hdTmp;
    Int                level;
    Int                i;

    /* all variables must begin with an identifier                         */
    if ( Symbol == S_IDENT )  hdVar = FindIdent( Value );
    else                      hdVar = 0;
    Match( S_IDENT, "identifier", follow );
    level = 0;

    /* complain about undefined global variables                           */
    if ( IsUndefinedGlobal && !NoWarnUndefined && !PermBegin &&
         Symbol != S_MAPTO && Symbol != S_MAPTO_METH && Symbol != S_ASSIGN_MAP && hdVar != HdCurLHS ) {
        SyntaxError("warning, undefined global variable");
        NrError--;
        NrErrLine--;
        NrHadSyntaxErrors = 1;
    }

    while ( backq-- )
        hdVar = Backquote(hdVar);

    /* followed by one or more selectors                                   */
    while ( IS_IN(Symbol,S_LBRACK|S_LBRACE|S_DOT|S_LPAREN) ) {

        /* <Var> '[' <Expr> ']'  list selector                             */
        if ( Symbol == S_LBRACK ) {
            Match( S_LBRACK, "", NUM_TO_UINT(0) );
            hd = RdExpr( S_RBRACK|follow );
            Match( S_RBRACK, "]", follow );
            if ( level == 0 ) {
                hdVar = BinBag( T_LISTELM, hdVar, hd );
            }
            else {
                hdTmp = NewBag( T_LISTELML, 2*SIZE_HD+sizeof(Int) );
                SET_BAG(hdTmp, 0,  hdVar );
                SET_BAG(hdTmp, 1,  hd );
                *(Int*)(PTR_BAG(hdTmp)+2) = level;
                hdVar = hdTmp;
            }
        }

        /* <VAR> '{' <Expr> '}'  sublist selector                          */
        else if ( Symbol == S_LBRACE ) {
            Match( S_LBRACE, "", NUM_TO_UINT(0) );
            hd = RdExpr( S_RBRACE|follow );
            Match( S_RBRACE, "}", follow );
            if ( level == 0 ) {
                hdVar = BinBag( T_LISTELMS, hdVar, hd );
            }
            else {
                hdTmp = NewBag( T_LISTELMSL, 2*SIZE_HD+sizeof(Int) );
                SET_BAG(hdTmp, 0,  hdVar );
                SET_BAG(hdTmp, 1,  hd );
                *(Int*)(PTR_BAG(hdTmp)+2) = level;
                hdVar = hdTmp;
            }
            level += 1;
        }

        /* <Var> '.' <Ident>  record selector                              */
        else if ( Symbol == S_DOT ) {
            Match( S_DOT, "", NUM_TO_UINT(0) );
            if ( Symbol == S_INT ) {
                hd = FindRecname( Value );
                Match( S_INT, "", follow );
            }
            else if ( Symbol == S_IDENT ) {
                hd = FindRecname( Value );
                Match( S_IDENT, "", follow );
            }
            else if ( Symbol == S_LPAREN ) {
                Match( S_LPAREN, "", follow );
                hd = RdExpr( follow );
                Match( S_RPAREN, ")", follow );
                if ( hd != 0 && GET_TYPE_BAG(hd) == T_MAKESTRING )
                    hd = FindRecname( (char*)PTR_BAG(hd) );
            }
            else {
                SyntaxError("record component name expected");
                hd = 0;
            }
            hdVar = BinBag( T_RECELM, hdVar, hd );
            level = 0;
        }

        /* <Var> '(' [ <Expr> { ',' <Expr> } ] ')'  function call          */
        else {
          int reset_no_warn = 0;

            Match( S_LPAREN, "", NUM_TO_UINT(0) );
            hd = NewBag( T_FUNCCALL, 4 * SIZE_HD );

            SET_BAG(hd, 0,  hdVar );
            if(hdVar && GET_TYPE_BAG(hdVar)!=T_INT && GET_FLAG_BAG(hdVar, BF_NO_WARN_UNDEFINED)
               && NoWarnUndefined==0) {
                reset_no_warn = 1;
                NoWarnUndefined = 1;
            }

            hdVar = hd;
            i = 1;
            if ( Symbol != S_RPAREN ) {
                i++;
                if ( GET_SIZE_BAG(hdVar) < i * SIZE_HD )
                    Resize( hdVar, (i+i/8+4) * SIZE_HD );
                hd = RdExpr( S_RPAREN|follow );
                SET_BAG(hdVar, i-1,  hd );
            }
            while ( Symbol == S_COMMA ) {
                Match( S_COMMA, "", NUM_TO_UINT(0) );
                i++;
                if ( GET_SIZE_BAG(hdVar) < i * SIZE_HD )
                    Resize( hdVar, (i+i/8+4) * SIZE_HD );
                hd = RdExpr( S_RPAREN|follow );
                SET_BAG(hdVar, i-1,  hd );
            }
            Match( S_RPAREN, ")", follow );
            Resize( hdVar, i * SIZE_HD );
            level = 0;

            if(reset_no_warn) NoWarnUndefined = 0;
        }
    }

    /* return the <Var> bag                                                */
    if ( NrError >= 1 )  return 0;
    return hdVar;
}


/****************************************************************************
**
*F  RdList( <follow> )  . . . . . . . . . . . . . . . . . . . . . read a list
**
**  'RdList' reads a list and returns the handle to the  newly  created  bag.
**  Lists have the form '[' <Expr>',' ... ']'.  Note that  a  list  may  have
**  undefined entries, in which case there is no  <Expr>  between the commas.
**  'RdList' is also responsible for reading ranges, which have the following
**  form, '[' <Expr> '..' <Expr> ']'.
**  In case of an error it skips all symbols up to one contained in <follow>.
**
**      <List>          :=  '[' [ <Expr> ] {',' [ <Expr> ] } ']'
**                      |   '[' <Expr> '..' <Expr> ']'
*/
Bag       RdList (TypSymbolSet follow)
{
    Bag           hdList;         /* handle of the result            */
    UInt       len;            /* logical length of the list      */
    Bag           hd;             /* temporary handle                */
    UInt       i;              /* loop variable                   */

    /* '['                                                                 */
    Match( S_LBRACK, "", NUM_TO_UINT(0) );
    hdList = NewBag( T_MAKELIST, 4 * SIZE_HD );
    i = 0;
    len = 0;

    /* [ <Expr> ]                                                          */
    if ( Symbol != S_RBRACK ) {
        i++;
        if ( GET_SIZE_BAG(hdList) <= i * SIZE_HD )
            Resize( hdList, (i+i/8+4) * SIZE_HD );
        if ( Symbol != S_COMMA ) {
            hd = RdExpr( S_RBRACK|follow );
            SET_BAG(hdList, i,  hd );
            len = i;
        }
    }

    /* {',' [ <Expr> ] }                                                   */
    while ( Symbol == S_COMMA ) {
        Match( S_COMMA, "", NUM_TO_UINT(0) );
        i++;
        if ( GET_SIZE_BAG(hdList) <= i*SIZE_HD )
            Resize( hdList, (i+i/8+4) * SIZE_HD );
        if ( Symbol != S_COMMA && Symbol != S_RBRACK ) {
            hd = RdExpr( S_RBRACK|follow );
            SET_BAG(hdList, i,  hd );
            len = i;
        }
    }

    /* '..' <Expr> ']'                                                     */
    if ( Symbol == S_DOTDOT ) {
        Match( S_DOTDOT, "", NUM_TO_UINT(0) );
        i++;
        if ( 3 < i )
            SyntaxError("'..' unexpexcted");
        if ( GET_SIZE_BAG(hdList) <= i*SIZE_HD )
            Resize( hdList, (i+i/8+4) * SIZE_HD );
        hd = RdExpr( S_RBRACK|follow );
        Match( S_RBRACK, "]", follow );
        SET_BAG(hdList, i,  hd );
        if ( NrError >= 1 )  return 0;
        hd = NewBag( T_MAKERANGE, i * SIZE_HD );
        SET_BAG(hd, 0,  PTR_BAG(hdList)[1] );
        SET_BAG(hd, 1,  PTR_BAG(hdList)[2] );
        if ( i == 3 )
            SET_BAG(hd, 2,  PTR_BAG(hdList)[3] );
        return hd;
    }

    /* ']'                                                                 */
    Match( S_RBRACK, "]", follow );

    /* return the <List> bag                                               */
    Resize( hdList, (i+1)*SIZE_HD );
    SET_BAG(hdList, 0,  INT_TO_HD(len) );
    if ( NrError >= 1 )  return 0;
    return hdList;
}


/****************************************************************************
**
*F  RdRec( <follow> ) . . . . . . . . . . . . . . . . . . . . . read a record
**
**  'RdRec' reads a record, returning a handle  to  the  newly  created  bag.
**  In case of an error it skips all symbols up to one contained in <follow>.
**
**      <Record>        :=  'rec( [ <Ident>:=<Expr> {, <Ident>:=<Expr> } ] )'
**
**  The bag is resized 16 entries at at time to avoid  doing  it  too  often.
*/

#define RdRec(follow) _RdRec(follow,T_MAKEREC)
#define RdTab(follow) _RdRec(follow,T_MAKETAB)

Obj  _RdRec ( TypSymbolSet follow, int type )
{
    Bag           hdRec,  hd, hdDoc, hdDocRecname;
    UInt       i;
    char * str;

    /* 'rec('                                                              */
    Match( S_IDENT, "", NUM_TO_UINT(0) );
    if ( SAVE_DEF_LINE ) {
        str = GuMakeMessage("[[ defined in %s:%d ]]\n", Input->name, Input->number);
        AppendCommentBuffer(str, (int)strlen(str));
        free(str);
    }
    hdDoc = GetCommentBuffer(); /* we don't want to pick up anything after '(' */
    ClearCommentBuffer();

    Match( S_LPAREN, "(", follow|S_RPAREN|S_COMMA );
    hdRec = NewBag(type, 8 * SIZE_HD );

    hdDocRecname = FindRecname("__doc__");
    SET_BAG(hdRec, 0,  hdDocRecname );
    SET_BAG(hdRec, 1,  hdDoc );

    i = 1;

    /* [ <Ident> ':=' <Expr>                                               */
    if ( Symbol != S_RPAREN ) {
        i++;
        if ( GET_SIZE_BAG(hdRec) < i*2*SIZE_HD )
            Resize( hdRec, (i+i/8+4) * 2 * SIZE_HD );

        if ( Symbol == S_INT ) {
            hd = FindRecname( Value );
            Match( S_INT, "", follow );
        }
        else if ( Symbol == S_IDENT ) {
            hd = FindRecname( Value );
            Match( S_IDENT, "", follow );
        }
        else if ( Symbol == S_LPAREN ) {
            Match( S_LPAREN, "", follow );
            hd = RdExpr( follow );
            Match( S_RPAREN, ")", follow );
            if ( hd != 0 && GET_TYPE_BAG(hd) == T_MAKESTRING )
                hd = FindRecname( (char*)PTR_BAG(hd) );
        }
        else {
            SyntaxError("record component name expected");
            hd = 0;
        }
        SET_BAG(hdRec, 2*i-2,  hd );
        Match( S_ASSIGN, ":=", follow );
        hd = RdExpr( S_RPAREN|follow );
        SET_BAG(hdRec, 2*i-1,  hd );
    }

    /* {',' <Ident> ':=' <Expr> } ]                                        */
    while ( Symbol == S_COMMA ) {
        Match( S_COMMA, "", NUM_TO_UINT(0) );
        if ( Symbol == S_RPAREN ) break;

        i++;
        if ( GET_SIZE_BAG(hdRec) < i * 2 * SIZE_HD )
            Resize( hdRec, (i+i/8+4) * 2 * SIZE_HD );
        if ( Symbol == S_INT ) {
            hd = FindRecname( Value );
            Match( S_INT, "", follow );
        }
        else if ( Symbol == S_IDENT ) {
            hd = FindRecname( Value );
            Match( S_IDENT, "", follow );
        }
        else if ( Symbol == S_LPAREN ) {
            Match( S_LPAREN, "", follow );
            hd = RdExpr( follow );
            Match( S_RPAREN, ")", follow );
            if ( hd != 0 && GET_TYPE_BAG(hd) == T_MAKESTRING )
                hd = FindRecname( (char*)PTR_BAG(hd) );
        }
        else {
            SyntaxError("record component name expected");
            hd = 0;
        }
        SET_BAG(hdRec, 2*i-2,  hd );
        Match( S_ASSIGN, ":=", follow );
        hd = RdExpr( S_RPAREN|follow );
        SET_BAG(hdRec, 2*i-1,  hd );
    }

    /* ')'                                                                 */
    Match( S_RPAREN, ")", follow );

    /* return the <Record> bag                                             */
    Resize( hdRec, i * 2 * SIZE_HD );
    if ( NrError >= 1 )  return 0;
    return hdRec;
}

#include "tables.h"
#include "namespaces.h"
#include "args.h"
#include "list.h"

Obj  RdLet ( TypSymbolSet follow ) {
    Obj  hdNS, hd, hdIdent;
    Obj  hdStmts = NewList(0);
    UInt i = 1;
    UInt no_warn_old = NoWarnUndefined;
    /* 'let('                                                              */
    Match( S_IDENT, "", NUM_TO_UINT(0) );
    Match( S_LPAREN, "(", follow|S_RPAREN|S_COMMA );
    hdNS = TableCreateT(T_MAKELET, 5);
    PushPrivatePackage(hdNS);

    if ( Symbol != S_RPAREN )
    do {
        /* read an expression */
        NoWarnUndefined = 1;
        hdIdent = RdExpr( S_ASSIGN|follow );
        NoWarnUndefined = no_warn_old;

        /* if it is not followed by := then it is a standalone expression */
        if ( Symbol != S_ASSIGN )  {
            /* NOTE: function locals always have higher priority here.. */
            ASS_LIST(hdStmts, i++, hdIdent);
        }
        /* followed by := it is <Ident> ':=' <Expr>                       */
        else if ( GET_TYPE_BAG(hdIdent) == T_VAR || GET_TYPE_BAG(hdIdent) == T_VARAUTO ) {
            char * nam; UInt pos;
            /* read the assigned expression */
            Match( S_ASSIGN, ":=", follow );
            hd = RdExpr( S_RPAREN|follow );

            /* only now add identifier to namespace, so it is not accidentally
               bound in it's own assigned expression */
            nam = VAR_NAME(hdIdent);
            pos = TableLookup(hdNS, nam, OFS_IDENT);
            if ( (hdIdent = PTR_BAG(hdNS)[pos]) == 0 )
                hdIdent = TableAddIdent(hdNS, pos, nam);
            else /* hdIdent is assigned */;

            /* put something in the slot for value, so that variable does not
               look uninitialized */
            SET_BAG(hdIdent, 0,  INT_TO_HD(0) );
            /* store the expression in the slot for properties (not for value) */
            SET_BAG(hdIdent, 1,  hd );
        }
        else {
            SyntaxError("identifier expected");
            Match( S_ASSIGN, ":=", follow );
            hd = RdExpr( S_RPAREN|follow );
        }

        if ( Symbol != S_COMMA ) break;
        else Match( S_COMMA, "", NUM_TO_UINT(0) );

    } while( 1 );

    /* ')'                                                                 */
    Match( S_RPAREN, ")", follow );

    hdIdent = TableAddIdent(hdNS, TableLookup(hdNS, "__res", OFS_IDENT), "__res");
    Retype(hdStmts, T_MAKELIST);
    /* again store expression in properties slot */
    SET_BAG(hdIdent, 1,  hdStmts );

    PopPrivatePackage();

	if( i==1 ) SyntaxError("Let-expression must contain result");


    if ( NrError >= 1 )  return 0;
    return hdNS;
}


/****************************************************************************
**
*F  RdPerm( <hdFirst>, <follow> ) . . . . . . . . . . . .  read a permutation
**
**  Note: currently  permutations  are  also  used  to represent left side of
**  multivariable lambda expressions (short function definitions), for insta-
**  nce (x,y) -> x+y. If we don't handle this case specially, undefined glob-
**  al variable error will be reported for x and y in (x,y).
**
**  'RdPerm' reads the rest of a permutation, which starts '( <hdFirst>, ...'
**  'RdPerm' returns the handle of the new created variable permutation  bag.
**  In case of an error it skips all symbols up to one contained in <follow>.
**
**      <Permutation>   :=  ( <Expr> {, <Expr>} ) { ( <Expr> {, <Expr>} ) }
**
**  If the permutation is constant,  i.e., if all <Expr> are simple integers,
**  'RdPerm' converts the 'T_MAKEPERM' to a 'T_PERM' by calling 'EvMakeperm'.
**
**  'RdPerm' is somewhat special, since all other reader functions are called
**  with the current Symbol beeing the first of the  construct  to  be  read,
**  while 'RdPerm' is called with the current Symbol beeing the ',', which is
**  the first moment we know that we read a permutation and not an expression
**  in parenthesis. This is the reason for the uncommon  argument  <hdFirst>.
**
**  To speed up reading of permutations, the varperm bag is enlarged  by  256
**  handles a time and shrunk to the correct size in the end, and the  cycles
**  bags are initially created with the size of the largest cycle encountered
**  so far.  It follows that for permutations of prime  order  no  nontrivial
**  'Resize' is ever needed.  Cycles are enlarged, if ever 16 handles a time.
*/
Bag       RdPerm (Bag hdFirst, TypSymbolSet follow)
{
    Bag           hdPerm,  hdCyc,  hd;
    UInt       i,  k,  m,  isConst;
    isConst = (hdFirst != 0) && (GET_TYPE_BAG(hdFirst) == T_INT);
    hdPerm = NewBag( T_MAKEPERM, 256*SIZE_HD );  i = 1;

    /* read the rest of the first cycle                                    */
    m = 2;
    hdCyc = NewBag( T_CYCLE, m*SIZE_HD );  k = 1;
    SET_BAG(hdPerm, i-1,  hdCyc );
    SET_BAG(hdCyc, 0,  hdFirst );
    while ( Symbol == S_COMMA ) {
        Match( S_COMMA, "", NUM_TO_UINT(0) );
        if ( ++k*SIZE_HD > GET_SIZE_BAG(hdCyc) )
            Resize( hdCyc, (k+15)*SIZE_HD );
        hd = RdExpr( S_RPAREN|follow );
        SET_BAG(hdCyc, k-1,  hd );
        isConst = isConst && (hd != 0) && (GET_TYPE_BAG(hd) == T_INT);
    }
    Match( S_RPAREN, ")", follow );
    Resize( hdCyc, k*SIZE_HD );
    if ( k > m )  m = k;

    /* read the other cycles                                               */
    while ( Symbol == S_LPAREN ) {
        Match( S_LPAREN, "", NUM_TO_UINT(0) );
        if ( ++i*SIZE_HD > GET_SIZE_BAG(hdPerm) )
            Resize( hdPerm, (i+255)*SIZE_HD );

        hdCyc = NewBag( T_CYCLE, m*SIZE_HD );  k = 1;
        SET_BAG(hdPerm, i-1,  hdCyc );
        hd = RdExpr( S_RPAREN|follow );
        SET_BAG(hdCyc, 0,  hd );
        /*if ( Symbol != S_COMMA )  SyntaxError(", expected");*/
        while ( Symbol == S_COMMA ) {
            Match( S_COMMA, "", NUM_TO_UINT(0) );
            if ( ++k*SIZE_HD > GET_SIZE_BAG(hdCyc) )
                Resize( hdCyc, (k+15) * SIZE_HD );
            hd = RdExpr( S_RPAREN|follow );
            SET_BAG(hdCyc, k-1,  hd );
            isConst = isConst && (hd != 0) && (GET_TYPE_BAG(hd) == T_INT);
        }
        Match( S_RPAREN, ")", follow );
        Resize( hdCyc, k*SIZE_HD );
        if ( k > m )  m = k;

    }
    Resize( hdPerm, i*SIZE_HD );

    /* return the <Permutation> bag                                        */
    if ( NrError >= 1 )  return 0;
    if ( isConst )  return EVAL( hdPerm );
    return hdPerm;
}


/****************************************************************************
**
*F  RdFunc( <follow> )  . . . . . . . . . . . . .  read a function definition
**
**  'RdFunc' reads a function definition and returns the handle of  the  bag.
**  In case of an error it skips all symbols up to one contained in <follow>.
**
**      <Function>      :=  'function (' [ <Ident> {',' <Ident>} ] ')'
**                              [ 'local'  <Ident> {',' <Ident>} ';' ]
**                              <Statments>
**                          'end'
**
**      <Function>      :=  'method (' [ <Ident> {',' <Ident>} ] ')'
**                              [ 'local'  <Ident> {',' <Ident>} ';' ]
**                              <Statments>
**                          'end'
*/
Bag       RdFunc (TypSymbolSet follow)
{
    Bag           hdFun, hd, hdLoc;
    short               nrArg = 0,  nrLoc = 0;

    /* 'function', make the local names know to the symbol tables          */
    if( Symbol == S_FUNCTION )
        hdFun = NewBag( T_MAKEFUNC, 2*SIZE_HD + 2*sizeof(short) );
    else /* S_METHOD */
        hdFun = NewBag( T_MAKEMETH, 2*SIZE_HD + 2*sizeof(short) );

    hdLoc = MakeDefString();

    Match(Symbol, "", NUM_TO_UINT(0) );
    PushFunction( hdFun );

    /* '(' [ <Ident> {',' <Ident> } ] ')'                                  */
    Match( S_LPAREN, "(", S_IDENT|S_RPAREN|S_LOCAL|STATBEGIN|S_END|follow );
    if ( Symbol != S_RPAREN ) {
        hd = MakeIdent(Value);
        Resize( hdFun, GET_SIZE_BAG(hdFun) + SIZE_HD );
        SET_BAG(hdFun, ++nrArg,  hd );
        Match( S_IDENT, "ident", S_RPAREN|S_LOCAL|STATBEGIN|S_END|follow );
    }
    while ( Symbol == S_COMMA ) {
        Match( S_COMMA, "", NUM_TO_UINT(0) );
        hd = MakeIdent(Value);
        Resize( hdFun, GET_SIZE_BAG(hdFun) + SIZE_HD );
        SET_BAG(hdFun, ++nrArg,  hd );
        Match( S_IDENT, "ident", S_RPAREN|S_LOCAL|STATBEGIN|S_END|follow );
    }
    Match( S_RPAREN, ")", S_LOCAL|STATBEGIN|S_END|follow );

    /* [ 'local' <Ident> {',' <Ident> } ';' ]                              */
    if ( Symbol == S_LOCAL ) {
        Match( S_LOCAL, "", NUM_TO_UINT(0) );
        hd = MakeIdent(Value);
        Resize( hdFun, GET_SIZE_BAG(hdFun) + SIZE_HD );
        SET_BAG(hdFun,  nrArg+ ++nrLoc ,  hd );
        Match( S_IDENT, "identifier", STATBEGIN|S_END|follow );
        while ( Symbol == S_COMMA ) {
            Match( S_COMMA, "", NUM_TO_UINT(0) );
            hd = MakeIdent(Value);
            Resize( hdFun, GET_SIZE_BAG(hdFun) + SIZE_HD );
            SET_BAG(hdFun,  nrArg+ ++nrLoc ,  hd );
            Match( S_IDENT, "identifier", STATBEGIN|S_END|follow );
        }
        Match( S_SEMICOLON, ";", STATBEGIN|S_END|follow );
    }
    /* one bag for doc */
    Resize( hdFun, GET_SIZE_BAG(hdFun) + SIZE_HD );
    SET_BAG(hdFun,  nrArg + nrLoc + 2,  hdLoc );

    /* function ( arg ) takes a variable number of arguments               */
    if ( nrArg == 1 && ! strcmp("arg", VAR_NAME(PTR_BAG(hdFun)[nrArg])) )
        nrArg = -1;
    NUM_ARGS_FUNC(hdFun) = nrArg;
    NUM_LOCALS_FUNC(hdFun) = nrLoc;

    /* <Statments>                                                         */
    hd = RdStats( S_END|follow );
    SET_BAG(hdFun, 0,  hd );
    Match( S_END, "end", follow );

    /* remove the local names from the symbol tables                       */
    PopFunction();

    /* return the <Function> bag                                           */
    if ( NrError >= 1 ) return 0;
    return hdFun;
}


/****************************************************************************
**
*F  RdAtom( <follow> )  . . . . . . . . . . . . . . . . . . . .  read an atom
**
**  'RdAtom' reads a single atom and returns  the  handle  of  the  new  bag.
**  In case of an error it skips all symbols up to one contained in <follow>.
**
**      <Atom>          :=  <Int>
**                      |   [$..]<Var>
**                      |   [$..]'(' <Expr> ')'
**                      |   <Permutation>
**                      |   <Char>
**                      |   <String>
**                      |   <Function>
**                      |   <List>
**                      |   <Record>
**
**      <Int>           :=  0|1|..|9 { 0|1|..|9 }
**
**      <Char>          :=  ' <any character> '
**
**      <String>        :=  " { <any character> } "
**
**  In cases other then (<Expr>) and <Var> leading $'s are silently ignored.
**  Note that $ has higher precedence then indexing [], function calls (), so
**  RdVar is also aware of them, in order to make this work.
*/
Obj  ObjDbl(double d);
double DblString(char *st);

Bag       RdAtom (TypSymbolSet follow)
{
    Bag           hdAt;
    Int                i;
    UInt       nr, pow;
    UInt       backq = 0;

    /* Leading backquotes */
    while ( Symbol == S_BACKQUOTE ) {
        ++backq;
        Match( Symbol, "", NUM_TO_UINT(0) );
    }

    /* <Int>                                                               */
    /* a little tricky, to avoid calling 'SumInt' and 'ProdInt' too often  */
    if ( Symbol == S_INT ) {
        nr   = 0;
        pow  = 1;
        hdAt = INT_TO_HD(0);
        for ( i = 0; Value[i] != '\0'; ++i ) {
            nr  = 10 * nr + Value[i]-'0';
            pow = 10 * pow;
            if ( pow == NUM_TO_UINT(100000000)) {
                hdAt = SumInt( ProdInt(hdAt,INT_TO_HD(pow)), INT_TO_HD(nr) );
                nr   = 0;
                pow  = 1;
            }
        }
        if ( hdAt == INT_TO_HD(0) )
            hdAt = INT_TO_HD(nr);
        else if ( pow != 1 )
            hdAt = SumInt( ProdInt(hdAt,INT_TO_HD(pow)), INT_TO_HD(nr) );
        Match(Symbol,"",NUM_TO_UINT(0));
    }

    else if ( Symbol == S_DOUBLE ) {
        hdAt = ObjDbl(DblString(Value));
        Match(Symbol,"",NUM_TO_UINT(0));
    }

    /* '(' <Expr> ')'                                                      */
    else if ( Symbol == S_LPAREN ) {
        Match( S_LPAREN, "(", follow );
        if ( Symbol == S_RPAREN ) {
            Match( S_RPAREN, "", NUM_TO_UINT(0) );
            hdAt = NewBag( T_PERM16, NUM_TO_UINT(0) );
        }
        else {
            /* Since  permutations  are also used for left side of multi-  *
             * variable lambda, we need to handle them specially to avoid  *
             * spurious undefined global variable warnings                 */

            PermBegin = 1;
            hdAt = RdExpr( follow );
            PermBegin = 0; /* PermBegin can't be true here no matter what  */

            if ( Symbol == S_COMMA ) {
               int reset_no_warn = (NoWarnUndefined==1) ? 0 : 1;
               NoWarnUndefined = 1;
               hdAt = RdPerm( hdAt, follow );
               if(reset_no_warn) NoWarnUndefined = 0;
            }
            else {
                Match( S_RPAREN, ")", follow );
            }
        }

        while ( backq-- )
            hdAt = Backquote(hdAt);
    }

    /* '[' [ <Expr> {, [ <Expr> ] } ] ']'                                  */
    else if ( Symbol == S_LBRACK ) {
        hdAt = RdList( follow );
    }

    /* 'rec(' [ <Ident> ':=' <Expr> {',' <Ident> ':=' <Expr> } ] ')'       */
    else if ( Symbol == S_IDENT && strcmp( Value, "rec" ) == 0 ) {
        hdAt = RdRec( follow );
    }

    /* 'tab(' [ <Ident> ':=' <Expr> {',' <Ident> ':=' <Expr> } ] ')'       */
    else if ( Symbol == S_IDENT && strcmp( Value, "tab" ) == 0 ) {
        hdAt = RdTab( follow );
    }

    /* 'tab(' [ <Ident> ':=' <Expr> {',' <Ident> ':=' <Expr> } ] ')'       */
    else if ( Symbol == S_IDENT && strcmp( Value, "let" ) == 0 ) {
        hdAt = RdLet( follow );
    }

    /* <Char>                                                              */
    else if ( Symbol == S_CHAR ) {
        hdAt = NewBag( T_CHAR, 1 );
        *((char*)PTR_BAG(hdAt)) = Value[0];
        Match( S_CHAR, "", NUM_TO_UINT(0) );
    }

    /* <String>                                                            */
    else if ( Symbol == S_STRING ) {
        hdAt = NewBag( T_MAKESTRING, (UInt)(strlen(Value)+1) );
        strncat( (char*)(PTR_BAG(hdAt)), Value, strlen(Value) );
        Match( S_STRING, "", NUM_TO_UINT(0) );
    }

    /* <Function>                                                          */
    else if ( Symbol == S_FUNCTION || Symbol == S_METHOD) {
        hdAt = RdFunc( follow );
    }

    /* <Var>                                                               */
    else if ( Symbol == S_IDENT ) {
        hdAt = RdVar( backq, follow );
    }

    /* generate an error, we want to see an expression                     */
    else {
        Match( S_INT, "expression", follow );
        hdAt = 0;
    }
    PermBegin = 0;
    /* return the <Atom> bag                                               */
    if ( NrError >= 1 )  return 0;
    return hdAt;
}


/****************************************************************************
**
*F  RdFactor( <follow> )  . . . . . . . . . . . . . . . . . . . read a factor
**
**  'RdFactor' reads a  factor  and  returns  the  handle  to  the  new  bag.
**  In case of an error it skips all symbols up to one contained in <follow>.
**
**      <Factor>        :=  {'+'|'-'} <Atom> [ '^' {'+'|'-'} <Atom> ]
*/
Bag       RdFactor (TypSymbolSet follow)
{
    Bag           hdFac,  hdAt;
    Int                sign1,  sign2;

    /* { '+'|'-' }  leading sign                                           */
    sign1 = 0;
    while ( Symbol == S_MINUS  || Symbol == S_PLUS ) {
        if ( sign1 == 0 )  sign1 = 1;
        if ( Symbol == S_MINUS ) sign1 = - sign1;

        Match( Symbol, "", NUM_TO_UINT(0) );
    }

    /* <Atom>                                                              */
    hdFac = RdAtom( follow );

    /* ['^' <Atom> ] implemented as {'^' <Atom> } for better error message */
    while ( Symbol == S_POW ) {

        /* match the '^' away                                              */
        Match( S_POW, "", NUM_TO_UINT(0) );

        /* { '+'|'-' }  leading sign                                       */
        sign2 = 0;
        while ( Symbol == S_MINUS  || Symbol == S_PLUS ) {
            if ( sign2 == 0 )  sign2 = 1;
            if ( Symbol == S_MINUS ) sign2 = - sign2;
            Match( Symbol, "", NUM_TO_UINT(0) );
        }

        /* ['^' <Atom>]                                                    */
        hdAt = RdAtom(follow);

        /* add the unary minus bag                                         */
        if ( sign2 == -1 && NrError == 0 && GET_TYPE_BAG(hdFac) <= T_INTNEG )
            hdAt = ProdInt( INT_TO_HD(-1), hdAt );
        else if ( sign2 == -1 && NrError == 0 )
            hdAt = BinBag( T_PROD, INT_TO_HD(-1), hdAt );

        /* create the power bag                                            */
        hdFac = BinBag( T_POW, hdFac, hdAt );
        if ( Symbol == S_POW )  SyntaxError("'^' is not associative");

    }

    /* add the unary minus bag                                             */
    if ( sign1 == -1 && NrError == 0 && GET_TYPE_BAG(hdFac) <= T_INTNEG )
        hdFac = ProdInt( INT_TO_HD(-1), hdFac );
    else if ( sign1 == -1 && NrError == 0 )
        hdFac = BinBag( T_PROD, INT_TO_HD(-1), hdFac );

    /* return the <Factor> bag                                             */
    return hdFac;
}


/****************************************************************************
**
*F  RdTerm( <follow> )  . . . . . . . . . . . . . . . . . . . . . read a term
**
**  'RdTerm' reads a term and returns the handle of the new  expression  bag.
**  In case of an error it skips all symbols up to one contained in <follow>.
**
**      <Term>          :=  <Factor> { '*'|'/'|'mod' <Factor> }
*/
Bag       RdTerm (TypSymbolSet follow)
{
    Bag           hdTer,  hdFac;
    unsigned int        type;

    /* <Factor>                                                            */
    hdTer = RdFactor( follow );

    /* { '*'|'/'|'mod' <Factor> }                                          */
    /* do not use 'IS_IN', since 'IS_IN(S_POW,S_MULT|S_DIV|S_MOD)' is true */
    while ( Symbol==S_MULT || Symbol==S_DIV || Symbol==S_MOD ) {
        switch ( Symbol ) {
        case S_MULT:  type = T_PROD;  break;
        case S_DIV:   type = T_QUO;   break;
        default:      type = T_MOD;   break;
        }
        Match( Symbol, "", NUM_TO_UINT(0) );
        hdFac = RdFactor( follow );
        hdTer = BinBag( type, hdTer, hdFac );
    }

    /* return the <Term> bag                                               */
    return hdTer;
}


/****************************************************************************
**
*F  RdAri( <follow> ) . . . . . . . . . . . . . read an arithmetic expression
**
**  Reads an arithmetic expression,  returning  a  handle  to  the  new  bag.
**  In case of an error it skips all symbols up to one contained in <follow>.
**
**      <Arith>         :=  <Term> { '+'|'-'|'::' <Term> }
*/
Bag       RdAri (TypSymbolSet follow)
{
    Bag           hdAri,  hdTer;
    unsigned int        type;

    /* <Term>                                                              */
    hdAri = RdTerm( follow );

    /* { '+'|'-' <Term> }                                                  */
    while ( IS_IN(Symbol,S_PLUS|S_MINUS|S_CONCAT) ) {
        type = (Symbol == S_PLUS) ?  T_SUM :  (Symbol==S_MINUS ? T_DIFF : T_CONCAT);
        Match( Symbol, "", NUM_TO_UINT(0) );
        hdTer = RdTerm( follow );
        hdAri = BinBag( type, hdAri, hdTer );
    }

    /* return the <Arith> bag                                              */
    return hdAri;
}


/****************************************************************************
**
*F  RdRel( <follow> ) . . . . . . . . . . . . .. read a relational expression
**
**  'RdRel' reads a relational expression, returning a handle to the new bag.
**  In case of an error it skips all symbols up to one contained in <follow>.
**
**      <Rel>           :=  { 'not' } <Arith> { '=|<>|<|>|<=|>=|in|is' <Arith> }
*/
Bag       RdRel (TypSymbolSet follow)
{
    Bag           hdRel,  hdAri;
    unsigned int        type;
    short               isNot;

    /* { 'not' }                                                           */
    isNot = 0;
    while ( Symbol == S_NOT ) { isNot = ! isNot;  Match( S_NOT, "", NUM_TO_UINT(0) ); }

    /* <Arith>                                                             */
    hdRel = RdAri( follow );

    /* { '=|<>|<|>|<=|>=|in' <Arith> }                                     */
    if ( IS_IN(Symbol,S_EQ|S_LT|S_GT|S_NE|S_LE|S_GE|S_IN|S_IS) ) {
        switch ( Symbol ) {
        case S_EQ:  type = T_EQ;  break;
        case S_LT:  type = T_LT;  break;
        case S_GT:  type = T_GT;  break;
        case S_NE:  type = T_NE;  break;
        case S_LE:  type = T_LE;  break;
        case S_GE:  type = T_GE;  break;
        case S_IN:  type = T_IN;  break;
        default:    type = T_IS;  break;
        }
        Match( Symbol, "", NUM_TO_UINT(0) );
        hdAri = RdAri( follow );
        hdRel = BinBag( type, hdRel, hdAri );
    }

    /* return the <Rel> bag                                                */
    if ( isNot && NrError == 0 ) {
        hdAri = NewBag( T_NOT, SIZE_HD );
        SET_BAG(hdAri, 0,  hdRel );  hdRel = hdAri;
    }
    return hdRel;
}


/****************************************************************************
**
*F  RdAnd( <follow> ) . . . . . . . . . . . . . read a logical and expression
**
**  'RdAnd' reads a logical expression and returns the handle of the new bag.
**  In case of an error it skips all symbols up to one contained in <follow>.
**
**      <And>           :=  <Rel> { 'and' <Rel> }
*/
Bag       RdAnd (TypSymbolSet follow)
{
    Bag           hdAnd,  hdRel;

    /* <Rel>                                                               */
    hdAnd = RdRel( follow );

    /* { 'and' <Rel> }                                                     */
    while ( Symbol == S_AND ) {
        Match( Symbol, "", NUM_TO_UINT(0) );
        hdRel = RdRel( follow );
        hdAnd = BinBag( T_AND, hdAnd, hdRel );
    }

    /* return the <And> bag                                                */
    return hdAnd;
}

/****************************************************************************
**
*F  RdLog( <follow> ) . . . . . . . . . . . . . . . read a logical expression
**
**  'RdLog' reads a logical expression and returns the handle of the new bag.
**  In case of an error it skips all symbols up to one contained in <follow>.
**
**      <Log>           :=  <And> { 'or' <And> }
*/
Bag       RdLog (TypSymbolSet follow)
{
    Bag           hdLog,  hdAnd;

    /* <And>                                                               */
    hdLog = RdAnd( follow );

    /* { 'or' <And> }                                                      */
    while ( Symbol == S_OR ) {
        Match( Symbol, "", NUM_TO_UINT(0) );
        hdAnd = RdAnd( follow );
        hdLog = BinBag( T_OR, hdLog, hdAnd );
    }

    /* return the <Log> bag                                                */
    return hdLog;
}


/****************************************************************************
**
*F  RdExpr( <follow> )  . . . . . . . . . . . . . . . . .  read an expression
**
**  'RdExpr' an expression, returning a handle  to  the  newly  created  bag.
**  In case of an error it skips all symbols up to one contained in <follow>.
**
**      <Expr>          :=  <Log>
**                      |   <Var> [ '->' <Log> ]
**                      |   <Var> [ '=>' <Log> ]
*/
Bag       CopyVarIfNoError (Bag hdVar)
{
    if ( NrError == 0 && hdVar!=0 && GET_TYPE_BAG(hdVar)==T_VAR)
        return MakeIdentSafe(hdVar, OFS_IDENT);
    else if(hdVar!=0 && GET_TYPE_BAG(hdVar)!=T_VAR)
        SyntaxError("left hand side of '->' must have only variables");
    return MakeIdent("");
}

Bag       RdExpr (TypSymbolSet follow)
{
    Bag           hdExp,  hdFun,  hdTmp;

    /* <Var>                                                               */
    hdExp = RdLog( follow|S_MAPTO|S_MAPTO_METH );

    /* [ '->' <Expr> ]                                                     */
    if ( Symbol == S_MAPTO || Symbol == S_MAPTO_METH) {
        int numArgs = 1;
        int type = (Symbol == S_MAPTO) ? T_MAKEFUNC : T_MAKEMETH;

        if ( hdExp != 0 && GET_TYPE_BAG(hdExp) != T_VAR && GET_TYPE_BAG(hdExp) != T_MAKEPERM &&
             GET_TYPE_BAG(hdExp) != T_PERM16 )
            SyntaxError("left hand side of '->' must be a variable");

        if ( hdExp != 0 && GET_TYPE_BAG(hdExp) == T_MAKEPERM ) {
            int i;
            numArgs = GET_SIZE_BAG(PTR_BAG(hdExp)[0]) / SIZE_HD; /* num of elements in cycle */
            hdFun = NewBag( type, (1+2+numArgs)*SIZE_HD + 2*sizeof(short) );
            for(i=0; i < numArgs; i++) {
                /* make a copy of all variables */
                hdTmp = CopyVarIfNoError(PTR_BAG(PTR_BAG(hdExp)[0])[i]);
                SET_BAG(hdFun, 1+i,  hdTmp );
            }
        }
        else if ( hdExp != 0 && GET_TYPE_BAG(hdExp)==T_PERM16 ) {
            /* this has to be an () empty perm, so function takes no parameters */
            numArgs = 0;
            hdFun = NewBag( type, (1+2+0)*SIZE_HD + 2*sizeof(short) );
        }
        else {
            numArgs = 1;
            hdFun = NewBag( type, (1+2+1)*SIZE_HD + 2*sizeof(short) );
            /* make a copy of the variable returned by 'RdLog'                 */
            hdTmp = CopyVarIfNoError(hdExp);
            SET_BAG(hdFun, 1,  hdTmp );

            /* function ( arg ) takes a variable number of arguments           */
            if ( strcmp("arg", VAR_NAME(hdTmp)) == 0 )
                numArgs = -1;
        }

        SET_BAG( hdFun, (numArgs>=0) ? numArgs+2 : 3, MakeDefString());
        NUM_ARGS_FUNC(hdFun) = numArgs;
        NUM_LOCALS_FUNC(hdFun) = 0;
        PushFunction( hdFun );

        /* match away the '->'                                             */
        Match( Symbol, "", NUM_TO_UINT(0) );

        /* read the expression                                             */
        hdExp = RdLog( follow );
        hdTmp = NewBag( T_RETURN, SIZE_HD );
        SET_BAG(hdTmp, 0,  hdExp );
        SET_BAG(hdFun, 0,  hdTmp );

        /* the function is the expression                                  */
        hdExp = hdFun;
        PopFunction();
    }
    /* [ '=>' <Expr> ]                                                     */
    else if ( Symbol == S_ASSIGN_MAP ) {
        Bag hdVar = hdExp;
        Bag hdValue = 0;

        if ( hdVar != 0 && GET_TYPE_BAG(hdVar) != T_VAR )
            SyntaxError("left hand side of '=>' must be a variable");

	/* NB: Remove 0 below to enable warning for undefined attributes */
	else if (0 && hdVar != 0 && ! ExistsRecname(VAR_NAME(hdVar)) && ! NoWarnUndefined ) {
 	    SyntaxError("attribute does not exist");
	    NrError--;
	    NrErrLine--;
	    NrHadSyntaxErrors = 1;
 	    FindRecname(VAR_NAME(hdVar));
	}

        /* match away the '->'                                             */
        Match( Symbol, "", NUM_TO_UINT(0) );

        /* read the expression                                             */
        hdValue = RdLog( follow );

        hdExp = BinBag(T_VARMAP, hdVar, hdValue);
    }

    /* return the <Expr> bag                                               */
    return hdExp;
}


/****************************************************************************
**
*F  RdIf( <follow> )  . . . . . . . . . . . . . . . . .  read an if statement
**
**  'RdIf' reads an 'if'-statement,  returning  a  handle  to  the  new  bag.
**  In case of an error it skips all symbols up to one contained in <follow>.
**
**      <Statement>     :=    'if'   <Expr> 'then' <Statments>
**                          { 'elif' <Expr> 'then' <Statments> }
**                          [ 'else'               <Statments> ]
**                            'fi'
*/
Bag       RdIf (TypSymbolSet follow)
{
    Bag hd = 0;
    Bag hdIf = 0;
    int i = 0;

    hd = NewList(0);
    /* 'if' <Expr>  'then' <Statments>                                     */
    Match( S_IF, "", NUM_TO_UINT(0) );
    ListAdd(hd, RdExpr( S_THEN|S_ELIF|S_ELSE|S_FI|follow ));
    Match( S_THEN, "then", STATBEGIN|S_ELIF|S_ELSE|S_FI|follow );
    ListAdd(hd, RdStats( S_ELIF|S_ELSE|S_FI|follow ));

    /* { 'elif' <Expr>  'then' <Statments> }                               */
    while ( Symbol == S_ELIF ) {
        Match( S_ELIF, "", NUM_TO_UINT(0) );
        ListAdd(hd, RdExpr( S_THEN|S_ELIF|S_ELSE|S_FI|follow ));
        Match( S_THEN, "then", STATBEGIN|S_ELIF|S_ELSE|S_FI|follow );
        ListAdd(hd, RdStats( S_ELIF|S_ELSE|S_FI|follow ));
    }

    /* [ 'else' <Statments> ]                                              */
    if ( Symbol == S_ELSE ) {
        Match( S_ELSE, "", NUM_TO_UINT(0) );
        ListAdd(hd, RdStats( S_FI|follow ));
    }

    /* 'fi'                                                                */
    Match( S_FI, "fi", follow );

    /* create and return the 'if'-statement bag                            */
    if ( NrError >= 1 )  return 0;
    i = LEN_PLIST(hd);
    hdIf = NewBag( T_IF, i * SIZE_HD );
    while ( i >= 1 ) { --i;  SET_BAG(hdIf, i,  ELM_PLIST(hd, i+1) ); }
    return hdIf;
}


/****************************************************************************
**
*F  RdFor( <follow> ) . . . . . . . . . . . . . . . . .  read a for statement
**
**  'RdFor' reads a 'for'-loop, returning a handle to the newly created  bag.
**  In case of an error it skips all symbols up to one contained in <follow>.
**
**      <Statement>     :=  'for' <Var>  'in' <Expr>  'do'
**                              <Statments>
**                          'od'
*/
Bag       RdFor (TypSymbolSet follow)
{
    Bag           hdVar,  hdList,  hdStats,  hdFor;

    /* 'for' <Var>                                                         */
    Match( S_FOR, "", NUM_TO_UINT(0) );
    hdVar = FindIdent( Value );
    Match( S_IDENT, "identifier", S_IN|S_DO|S_OD|follow );

    /* complain about undefined global variables                           */
    if ( IsUndefinedGlobal && !NoWarnUndefined) {
        SyntaxError("warning, undefined global variable");
        NrError--;
        NrErrLine--;
        NrHadSyntaxErrors = 1;
    }

    /* 'in' <Expr>                                                         */
    Match( S_IN, "in", S_DO|S_OD|follow );
    hdList = RdExpr( S_DO|S_OD|follow );

    /* 'do' <Statments>                                                    */
    Match( S_DO, "do", STATBEGIN|S_OD|follow );
    hdStats = RdStats( S_OD|follow );

    /* 'od'                                                                */
    Match( S_OD, "od", follow );

    /* create and return the 'for'-loop bag                                */
    if ( NrError >= 1 )  return 0;
    hdFor = NewBag( T_FOR, 3 * SIZE_HD );
    SET_BAG(hdFor, 0,  hdVar );  SET_BAG(hdFor, 1,  hdList );
    SET_BAG(hdFor, 2,  hdStats );
    return hdFor;
}


/****************************************************************************
**
*F  RdWhile( <follow> ) . . . . . . . . . . . . . . .  read a while statement
**
**  'RdWhile' reads a 'while'-loop,  returning  a  handle  to  the  new  bag.
**  In case of an error it skips all symbols up to one contained in <follow>.
**
**      <Statement>     :=  'while' <Expr>  'do'
**                              <Statments>
**                          'od'
*/
Bag       RdWhile (TypSymbolSet follow)
{
    Bag       hdCond,  hdStats,  hdWhile;

    /* 'while' <Expr>  'do'                                                */
    Match( S_WHILE, "", NUM_TO_UINT(0) );
    hdCond = RdExpr( S_DO|S_OD|follow );
    Match( S_DO, "do", STATBEGIN|S_DO|follow );

    /*     <Statments>                                                     */
    hdStats = RdStats( S_OD|follow );

    /* 'od'                                                                */
    Match( S_OD, "od", follow );

    /* create and return the 'while'-loop bag                              */
    if ( NrError >= 1 )  return 0;
    hdWhile = NewBag( T_WHILE, 2 * SIZE_HD );
    SET_BAG(hdWhile, 0,  hdCond );  SET_BAG(hdWhile, 1,  hdStats );
    return hdWhile;
}


/****************************************************************************
**
*F  RdRepeat( <follow> )  . . . . . . . . . . . . . . read a repeat statement
**
**  'RdRepeat' reads a 'repeat'-loop, returning a  handle  to  the  new  bag.
**  In case of an error it skips all symbols up to one contained in <follow>.
**
**      <Statement>     :=  'repeat'
**                              <Statments>
**                          'until' <Expr>
*/
Bag       RdRepeat (TypSymbolSet follow)
{
    Bag       hdStats,  hdCond,  hdRep;

    /* 'repeat' <Statments>                                                */
    Match( S_REPEAT, "", NUM_TO_UINT(0) );
    hdStats = RdStats( S_UNTIL|follow );

    /* 'until' <Expr>                                                      */
    Match( S_UNTIL, "until", EXPRBEGIN|follow );
    hdCond = RdExpr( follow );

    /* create and return the 'repeat'-loop bag                             */
    if ( NrError >= 1 )  return 0;
    hdRep = NewBag( T_REPEAT, 2 * SIZE_HD );
    SET_BAG(hdRep, 0,  hdCond );  SET_BAG(hdRep, 1,  hdStats );
    return hdRep;
}


/****************************************************************************
**
*F  RdReturn( <follow> )  . . . . . . . . . . . . . . read a return statement
**
**  'RdReturn' reads a 'return'-statement, returning a handle of the new bag.
**  Return with no expression following is used in functions to return  void.
**  In case of an error it skips all symbols up to one contained in <follow>.
**
**      <Statement>     :=  'return' [ <Expr> ]
**
**  It is still legal to use parenthesis but they  are  no  longer  required,
**  a return statememt is not a function call and should not look  like  one.
*/
Bag       RdReturn (TypSymbolSet follow)
{
    Bag           hdRet,  hdExpr;

    /* skip the return symbol                                              */
    Match( S_RETURN, "", NUM_TO_UINT(0) );

    /* 'return' with no expression following                               */
    if ( Symbol == S_SEMICOLON ) {
        if ( NrError >= 1 )  return 0;
        hdRet = NewBag( T_RETURN, SIZE_HD );
        SET_BAG(hdRet, 0,  HdVoid );
    }

    /* 'return' with an expression following                               */
    else {
        hdExpr = RdExpr( follow );
        if ( NrError >= 1 )  return 0;
        hdRet = NewBag( T_RETURN, SIZE_HD );
        SET_BAG(hdRet, 0,  hdExpr );
    }

    /* return the 'return'-statement bag                                   */
    return hdRet;
}


/****************************************************************************
**
*F  RdQuit( <follow> )  . . . . . . . . . . . . . . . . read a quit statement
**
**  'RdQuit' reads a 'quit' statement, returning a handle  of  the  new  bag.
**  In case of an error it skips all symbols up to one contained in <follow>.
**
**      <Statement>     :=  'quit'
*/
Bag       RdQuit (TypSymbolSet follow)
{
    Bag           hdQuit;
    Match( S_QUIT, "", follow );
    hdQuit = NewBag( T_RETURN, SIZE_HD );
    SET_BAG(hdQuit, 0,  HdReturn );
    return hdQuit;
}


/****************************************************************************
**
*F  RdStat( <follow> )  . . . . . . . . . . . . . . . . . .  read a statement
**
**  Reads a single statement, returning the handle to the newly created  bag.
**  In case of an error it skips all symbols up to one contained in <follow>.
**
**      <Statment>      :=  <Expr>
**                      |   <Var> ':=' <Expr>
**                      |   'if'   <Expr>  'then' <Statments>
**                        { 'elif' <Expr>  'then' <Statments> }
**                        [ 'else'                <Statments> ] 'fi'
**                      |   'for' <Var>  'in' <Expr>  'do' <Statments>  'od'
**                      |   'while' <Expr>  'do' <Statments>  'od'
**                      |   'repeat' <Statments>  'until' <Expr>
**                      |   'return' [ <Expr> ]
**                      |   'quit'
*/

Bag       RdStat (TypSymbolSet follow)
{
    Bag           hd,  hdExpr,  hdAss,  hdComment=0, hdElm;
    UInt t, i, plen;

    /* handle those cases where the statement has a unique prefix symbol   */
    if ( Symbol == S_IF      )  return RdIf( follow );
    if ( Symbol == S_FOR     )  return RdFor( follow );
    if ( Symbol == S_WHILE   )  return RdWhile( follow );
    if ( Symbol == S_REPEAT  )  return RdRepeat( follow );
    if ( Symbol == S_RETURN  )  return RdReturn( follow );
    if ( Symbol == S_QUIT    )  return RdQuit( follow );

    /* read an expression                                                  */
    hd = RdExpr( S_ASSIGN|follow );
    if ( Symbol != S_ASSIGN )  return hd;

    t = GET_TYPE_BAG(hd);
    /* if the expression is followed by := it is an assignment             */
    if ( hd != 0  && t != T_VAR       && t != T_VARAUTO  && t != T_MAKELIST
                  && t != T_LISTELM   && t != T_LISTELML && t != T_LISTELMS
                  && t != T_LISTELMSL && t != T_RECELM  )
        SyntaxError("left hand side of assignment must be a variable");
    Match( S_ASSIGN, "", NUM_TO_UINT(0) );

    /* need identifiers for writing, could be in different namespace, remap using FindIdentWr */
    if(hd!=0) {
        if (t==T_VAR || t==T_VARAUTO)
            hd = FindIdentWr(VAR_NAME(hd));
        else if(t==T_MAKELIST) {
            /* remap all identifiers in a list */
            plen = LEN_PLIST(hd);
            for(i=1; i<=plen; ++i) {
                hdElm = PTR_BAG(hd)[i];
                if (GET_TYPE_BAG(hdElm) != T_VAR)
                    SyntaxError("left hand side of multi-assignment must contain variables only");
                else {
                    hdElm = FindIdentWr(VAR_NAME(hdElm));
                    SET_BAG(hd, i,  hdElm );
                }
            }
        }
    }
    /* doc stuff */
    if(TopStack == 0) {
        if ( SAVE_DEF_LINE ) {
            char * str = GuMakeMessage("[[ defined in %s:%d ]]\n", Input->name, Input->number);
            AppendCommentBuffer(str, (int)strlen(str));
            free(str);
        }

        if(hd!=0 && (GET_TYPE_BAG(hd)==T_VAR || GET_TYPE_BAG(hd)==T_VARAUTO)) {
            DocumentVariable(hd, GetCommentBuffer());
        }
        hdComment = GetCommentBuffer();
        ClearCommentBuffer();
    }
    else { ClearCommentBuffer(); } //hdComment = GetCommentBuffer(); }

    /* end doc stuff */

    if ( HdCurLHS == 0 ) {
        HdCurLHS = hd;
        hdExpr = RdExpr( follow );
        HdCurLHS = 0;
    }
    else {
        hdExpr = RdExpr( follow );
    }


    /* create an assignment bag and return it                              */
    if ( NrError >= 1 )  return 0;
    if      ( GET_TYPE_BAG(hd)==T_VAR       )  hdAss = TriBag(T_VARASS,   hd,hdExpr,hdComment);
    else if ( GET_TYPE_BAG(hd)==T_VARAUTO   )  hdAss = TriBag(T_VARASS,   hd,hdExpr,hdComment);
    else if ( GET_TYPE_BAG(hd)==T_LISTELM   )  hdAss = BinBag(T_LISTASS,  hd,hdExpr);
    else if ( GET_TYPE_BAG(hd)==T_MAKELIST  )  hdAss = BinBag(T_MULTIASS, hd,hdExpr);
    else if ( GET_TYPE_BAG(hd)==T_LISTELML  )  hdAss = BinBag(T_LISTASSL, hd,hdExpr);
    else if ( GET_TYPE_BAG(hd)==T_LISTELMS  )  hdAss = BinBag(T_LISTASSS, hd,hdExpr);
    else if ( GET_TYPE_BAG(hd)==T_LISTELMSL )  hdAss = BinBag(T_LISTASSSL,hd,hdExpr);
    else                               hdAss = BinBag(T_RECASS,   hd,hdExpr);

    return hdAss;
}


/****************************************************************************
**
*F  RdStats( <follow> ) . . . . . . . . . . . . . . read a statement sequence
**
**  Reads a statement sequence,  returning a handle to the newly created bag.
**  In case of an error it skips all symbols up to one contained in <follow>.
**
**      <Statments>     :=  { <Statment> ; }
**                      |   ;
**
**  A single semicolon is an empty statement sequence not an empty statement.
*/
Bag       RdStats (TypSymbolSet follow)
{
    Bag           hdStats,  hd [1024];
    short               i = 0;

    /* a single semicolon is an empty statement sequence                   */
    if ( Symbol == S_SEMICOLON ) {
        Match( S_SEMICOLON, "", NUM_TO_UINT(0) );
    }

    /* { <Statement> ; }                                                   */
    else {
        while ( IS_IN(Symbol,STATBEGIN) || i == 0 ) {
            if ( i == 1024 ) {
                SyntaxError("sorry, can not read more than 1024 statements");
                i = 0;
            }
            hd[i++] = RdStat( S_SEMICOLON|follow );
            if ( Symbol == S_SEMICOLON
              && hd[i-1] != 0                && GET_TYPE_BAG(hd[i-1]) != T_VARASS
              && GET_TYPE_BAG(hd[i-1]) != T_LISTASS  && GET_TYPE_BAG(hd[i-1]) != T_LISTASSL
              && GET_TYPE_BAG(hd[i-1]) != T_LISTASSS && GET_TYPE_BAG(hd[i-1]) != T_LISTASSSL
              && GET_TYPE_BAG(hd[i-1]) != T_RECASS && GET_TYPE_BAG(hd[i-1]) != T_MULTIASS
              && !(T_FUNCCALL<=GET_TYPE_BAG(hd[i-1]) && GET_TYPE_BAG(hd[i-1])<=T_RETURN)) {
                SyntaxError("warning, this statement has no effect");
                NrError--;
                NrErrLine--;
                NrHadSyntaxErrors = 1;
            }
            Match( S_SEMICOLON, ";", follow );
        }
    }

    /* create and return the statement sequence bag                        */
    if ( NrError >= 1 )  return 0;
    if ( i == 1 )  return hd[0];
    hdStats = NewBag( T_STATSEQ, i * SIZE_HD );
    while ( i >= 1 ) { --i; SET_BAG(hdStats, i,  hd[i] ); }
    return hdStats;
}


/****************************************************************************
**
*F  ReadIt()  . . . . . . . . . . . . . . . . . read a statement interactivly
**
**  'ReadIt' reads a single statement, returning the handle to the  new  bag.
**  This is the only reading function that doesn't expect the first symbol of
**  its input already read and wont read the first symbol of the  next input.
**  This is the main interface function for the  various  ReadEvalPrintLoops.
**
**  It has this funny name, because 'Read' would give name clash with  'read'
**  from the C library on the stupid VAX, which turns all names to uppercase.
*/
Bag       ReadIt (void)
{
    Bag           hd;

    /* get the first symbol from the input                                 */
    Match( Symbol, "", NUM_TO_UINT(0) );

    /* print only a partial prompt from now on                             */
    Prompt = "> ";

    if ( Symbol == S_SEMICOLON || Symbol == S_EOF  )
        return 0;

    /* read a statement                                                    */
	hd = RdStat( S_SEMICOLON|S_EOF );

    /* every statement must be terminated by a semicolon                   */
    if ( Symbol != S_SEMICOLON )
        SyntaxError("; expected");

    if ( Symbol == S_EOF )
        return 0;
    if ( NrError >= 1 )
        return 0;

    return hd;
}


/****************************************************************************
**
*E  Emacs . . . . . . . . . . . . . . . . . . . . . . . local emacs variables
**
**  Local Variables:
**  outline-regexp:   "*A\\|*F\\|*V\\|*T\\|*E"
**  fill-column:      73
**  fill-prefix:      "**  "
**  End:
**
*/



