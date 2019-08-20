/****************************************************************************
**
*A  spiral_delay_ev.h            SPIRAL source               Yevgen Voronenko
**
**
**  This  package  implements  the  delayed  evaluation  modeled  after  LISP 
**  backquote operator.
**
**  Evaluation of any expression in GAP can be delayed using Delay() internal
**  function. It returns an object of type T_DELAY, which is just a container
**  for an unevaluated expression. T_DELAY object can be evaluated using 
**  Eval(). 
**
**  Like in Lisp, Delay()/Eval() work like brackets, eg. the following predi-
**  cates are true:
**
**    Delay(Eval(x)) = x
**    Eval(Eval(Delay(Delay(x)))) = x
**
**  Unlike in Lisp, Arithmetic can be done on T_DELAY objects to construct
**  new T_DELAY objects:
**
**    Delay(x) + 1 = Delay(x + 1)
**
*/

/****************************************************************************
**
*F  ExprFromDelay(<hd>) . . . . . . returns an expression from T_DELAY object
*F  DelayFromExpr(<hd>  . . . . . . makes a T_DELAY object from an expression
*F  RawDelayFromExpr(<hd>  . . .  makes a T_DELAY without inner D() expansion
*/
Obj  ExprFromDelay ( Obj hdDelay );
Obj  DelayFromExpr ( Obj hdExpr );
Obj  RawDelayFromExpr ( Obj hdExpr );

#define PROJECTION_D(obj)  (GET_TYPE_BAG(obj)<=T_DELAY ? (obj) : RawDelayFromExpr(obj))
#define INJECTION_D(obj)   (GET_TYPE_BAG(obj)==T_DELAY ? PTR_BAG(obj)[0] : (obj))

/****************************************************************************
**
*F  FunD(<hdCall>)  . . . . . . . . . . . delays the evaluation of expression
*F  FunDelay(<hdCall>)  . . . . . . . . . delays the evaluation of expression
**
**  'FunDelay' implements the internal function 'Delay'.
**  'FunD' implements the internal function 'D'.
**
**     D( <expr> )
**     Delay( <expr> )
**     Delay( <expr>, <var1>, .... <varN> )
**     Delay( <expr>, <var1> => <value1>, ..., <var2> => <value2> )
**
**  Both 'D'  and  'Delay'  delay  the  evaluation  of  expression <expr> by 
**  creating a T_DELAY object. 
**
**  'Delay' can optionally evaluate specified variables <var1> .. <varN> and
**  thus can be used to d create a return value of a function, where it is a
**  delayed expression parametrized by function arguments.
**
**  The value of the variables can  be  explicitly specified using a special 
**  '=>' (variable map) operator.  Using  a  variable map will not alter the 
**  values of the variables in the containing scope.
**
**  Example: 
**    spiral> x:=1;;
**    spiral> expr:=Delay(x+2, x);
**       D(1 + 2)
**    spiral> expr:=Delay(x+2, x=>2);
**       D(2 + 2)
**    spiral> x;
**       1   <-- variable map 'x=>2' does not alter 'x'
**    spiral> Eval(expr);
**       4
**
**  The T_DELAY objects can be later evaluated using Eval().
**  Delay() statements can be nested, then Eval() will strip one level
**  of Delay. 
**
**  There is an important difference between 'Delay' and 'D'.  'Delay' should 
**  be understood as a helper function, and 'D' as an object constructor.  As
**  a special constrictor, evaluation of 'D' is not standard - it can *never*
**  be delayed. This can only be demonstrated by nesting these functions:
**
**    spiral> a := D(D(1));
**       D(D(1))  <-- doubly delayed object is constructed
**    spiral> EvalDeep(a); 
**       1        
**    spiral> b := Delay(Delay(1));
**       D(Delay( 1 )) <-- delayed function call Delay(1) is constructed
**    spiral> EvalDeep(b);
**       D(1)  <-- EvalDeep evaluates 1->1, Delay( 1 )->D(1), D(D(1))->D(1)
*/
Obj  FunDelay ( Obj hdCall );
Obj  FunD     ( Obj hdCall );


/****************************************************************************
**
*F  FunEval(<hdCall>) . . . . . . . . . . . . . evaluate a delayed expression
*F  FunEvalDeep(<hdCall>) . . . . . evaluate a delayed expression recursively
*F  FunEvalVar(<hdCall>) . . . evaluate variables inside a delayed expression
**
**  'FunEval' implements the internal function 'Eval'.
**  'FunEvalDeep' implements the internal function 'Eval'.
**  'FunEvalVar' implements the internal function 'EvalVar'.
**
**  'Eval( <delayed-obj> )'
**
**  Evaluates an expression if the evaluation was earlier delayed with 'Delay'
**  otherwise nothing is done.
**
**  'EvalDeep( <delayed-obj> )'
**
**  Same as Eval() but does the evaluation recursively,  stripping all levels
**  of Delay().
**
**  'EvalVar( <delayed-obj>, <var1>, .... <varN> )
**  'EvalVar( <delayed-obj>, <var1> => <value1>, ..., <var2> => <value2> )
**
**  Evaluates  variables  <var1> .. <varN>  within  a  <delayed-obj>.  In the
**  alternative form, the values of the  variables  are  explicitly specified
**  with '=>' (variable map) operator.  Using  a  variable map will not alter 
**  the values of the variables in the containing scope.
**
**  EvalVar(D(expr), var1, ..., varN) 
**     is equivalent to
**  Delay(expr, var1, ..., varN)
**
*/
Obj  FunEval     ( Obj hdCall );
Obj  FunEvalDeep ( Obj hdCall );
Obj  FunEvalVar  ( Obj hdCall );


/****************************************************************************
**
*F  FunIsDelay( <obj> ) . . . . . . check whether an object is T_DELAY object
**
**  'FunIsDelay' implements the internal function 'IsDelay'
**  'IsDelay( <obj> )' return 'true' if <obj> is a T_DELAY object.
*/
Obj  FunIsDelay ( Obj hdCall );

 
/****************************************************************************
**
*F  DelaySum(<hdL>,<hdR>) . . . . . .  construct a delayed sum of two objects
*F  DelayDiff(<hdL>,<hdR>) . .  construct a delayed difference of two objects
*F  DelayProd(<hdL>,<hdR>) . . . . construct a delayed product of two objects
*F  DelayQuo(<hdL>,<hdR>) . . . . construct a delayed quotient of two objects
*F  DelayMod(<hdL>,<hdR>) . . . . . .  construct a delayed mod of two objects
*F  DelayPow(<hdL>,<hdR>) . . . . .  construct a delayed power of two objects
** 
*F  EvDelay(<hd>) . . . . . . . . .  evaluate a delayed object (returns <hd>)
*F  PrDelay(<hd>) . . . . . . . . . . . . . . . . . .  print a delayed object
**
*F  EvVarMap(<hd>) . . . . . . . . . . . . . . evaluate a variable map object
*F  PrVarMap(<hd>) . . . . . . . . . . . . . . .  print a variable map object
**
*/
Obj  DelaySum  ( Obj hdL, Obj hdR );
Obj  DelayDiff ( Obj hdL, Obj hdR );
Obj  DelayProd ( Obj hdL, Obj hdR );
Obj  DelayQuo  ( Obj hdL, Obj hdR );
Obj  DelayMod  ( Obj hdL, Obj hdR );
Obj  DelayPow  ( Obj hdL, Obj hdR );

Obj  EvDelay ( Obj hd );
void PrDelay ( Obj hd );

Obj  EvVarMap( Obj hd );
void PrVarMap( Obj hd );


/****************************************************************************
**
*F  InitSPIRAL_DelayEv()  . . . . . initialize the delayed evaluation package
*/
void            InitSPIRAL_DelayEv();

