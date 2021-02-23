/****************************************************************************
**
*A  read.h                      GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This module declares the functions to read  expressions  and  statements.
**
**      <Ident>         :=  a|b|..|z|A|B|..|Z { a|b|..|z|A|B|..|Z|0|..|9|_ }
**
**      <Var>           :=  <Ident>
**                      |   <Var> '.' <Ident>
**                      |   <Var> '[' <Expr> ']'
**                      |   <Var> '{' <Expr> '}'
**                      |   <Var> '(' [ <Expr> { ',' <Expr> } ] ')'
**
**      <List>          :=  '[' [ <Expr> ] {',' [ <Expr> ] } ']'
**                      |   '[' <Expr> '..' <Expr> ']'
**
**      <Record>        :=  'rec( [ <Ident>:=<Expr> {, <Ident>:=<Expr> } ] )'
**
**      <Permutation>   :=  ( <Expr> {, <Expr>} ) { ( <Expr> {, <Expr>} ) }
**
**      <Function>      :=  'function (' [ <Ident> {',' <Ident>} ] ')'
**                              [ 'local'  <Ident> {',' <Ident>} ';' ]
**                              <Statments>
**                          'end'
**
**      <String>        :=  " { <any character> } "
**
**      <Int>           :=  0|1|..|9 { 0|1|..|9 }
**
**      <Atom>          :=  <Int>
**                      |   <Var>
**                      |   '(' <Expr> ')'
**                      |   <Permutation>
**                      |   <String>
**                      |   <Function>
**                      |   <List>
**                      |   <Record>
**
**      <Factor>        :=  {'+'|'-'} <Atom> [ '^' {'+'|'-'} <Atom> ]
**
**      <Term>          :=  <Factor> { '*'|'/'|'mod' <Factor> }
**
**      <Arith>         :=  <Term> { '+'|'-' <Term> }
**
**      <Rel>           :=  { 'not' } <Arith> { '=|<>|<|>|<=|>=|in|==' <Arith> }
**
**      <And>           :=  <Rel> { 'and' <Rel> }
**
**      <Log>           :=  <And> { 'or' <Rel> }
**
**      <Expr>          :=  <Log>
**                      |   <Var> [ '->' <Log> ]
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
**
**      <Statments>     :=  { <Statment> ; }
**                      |   ;
**
*/

/****************************************************************************
**
*F  Backquote() . .  to be implemented elsewhere as a constructor for `<expr>
**  
**  'Backquote' provides a hook to customize behavior of `<expr> syntax, 
**  currently it is defined in spiral_delay_ev.c
*/
Bag       Backquote ( Bag );

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
Bag       ReadIt ( void );


/****************************************************************************
**
*F  BinBag( <type>, <hdL>, <hdR> )  . . . . . . . . . . . . make a binary bag
**
**  'BinBag' makes a new bag of the type <type> with the  two  objects  <hdL>
**  and <hdR>.  No bag is made if an error has occured  during  the  parsing.
*/
Bag       BinBag ( unsigned int type, Bag hdL, Bag hdR );

/****************************************************************************
**
*F  UniBag( <type>, <hd> )  . . . . . . . . . . . . . . . .  make a unary bag
**
**  'UniBag' makes a new bag of the type <type> with  the  one  objects  <hd>
*/
Bag       UniBag ( unsigned int type, Bag hd );
