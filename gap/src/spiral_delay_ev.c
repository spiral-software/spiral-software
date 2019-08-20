/****************************************************************************
**
*A  spira_delay_ev.c             SPIRAL source               Yevgen Voronenko
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
**  cate is true:
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

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* Bag, NewBag, T_STRING, .. */
#include        "objects.h"
#include		"string4.h"
#include        "integer.h"             /* arbitrary size integers         */
#include        "eval.h"
#include	"idents.h"
#include        "scanner.h"             /* Pr()                            */
#include        "list.h"
#include        "read.h"                /* BinBag                          */
#include        "args.h"
/* #include        "flags.h" */			// defns & decls moved to memmgr.h
#include        "function.h"            /* NUM_ARGS_FUNC, NUM_LOCAL_FUNC   */
#include        "spiral_delay_ev.h"
#include        "spiral.h"
#include        "namespaces.h"

Obj  HdEval;
Obj  HdEvalFromD;
Obj  HdBackquote;

/* This  macro  can  be  used to  debug  RecursiveEval  or  other  recursive 
 * functions.  It prints <hd> on a separate line. Print can't print T_RECNAM 
 * bags, thus we have a check for this type.
 */
#define TRACE(hd) if(hd) { Pr("->%d| ",GET_TYPE_BAG(hd),0); \
                           if(GET_TYPE_BAG(hd)==T_RECNAM) Pr("recnam",0,0); \
			   else Print(hd); \
	                   Pr("\n",0,0); }


/* returns list{[head..Length(list)]} for a 0-indexed function argument list */
Obj  TailArgList ( Obj list, UInt head ) {
    Obj result;
    UInt len = GET_SIZE_BAG(list) / SIZE_HD;
    UInt result_len, p;
    assert(head <= len);

    /* remember GAP lists are indexed starting with 1,
       but argument lists are indexed starting with 0 */
    result_len = len - head;
    result = NewList(result_len);
    for(p = head; p < len; ++p) 
	ASS_LIST(result, p-head+1, ELM_ARGLIST(list, p));
    return result;
}

/* ================ Predicate functions for RecursiveEval() ================
 * The following functions customize behaviour of RecursiveEval 
 * =========================================================================
 */
enum {
    EVAL_RECURS_TAIL, /* first evaluation, then recursive call */
    EVAL_RECURS_HEAD, /* first recursive call, then evaluation */
    EVAL_NO_RECURS, /* evaluation but no recursive call */
    EVAL_NO_EVAL, /* recursive call, but no evaluation */
    EVAL_DELAY, /* Get delayed object, and evaluate it (recursively) */
    EVAL_SUBST, /* substitute object by HdSubst, no recursive call */
    EVAL_NOTHING /* do nothing */
};

#define IS_VAR_SUBST(x) (GET_TYPE_BAG(x)==T_VAR || GET_TYPE_BAG(x)==T_VARMAP)

/* RecursiveEval evaluates an expression recursively using head recursion   */
int recurs_head_eval(Obj arg, Obj hdObj) {
    /*
	 * NOTE: The only reason not to go inside function definitions is that
	 * there might be unevaluatable (local) variables
	 */
    switch( GET_TYPE_BAG(hdObj) ) {
    case T_MAKEFUNC: return EVAL_NO_RECURS;
    case T_VARAUTO: return EVAL_RECURS_TAIL;
    case T_DELAY: return EVAL_DELAY;
    default: return EVAL_RECURS_HEAD;
    }
}

/* RecursiveEval evaluates an expression recursively using tail recursion   */
int recurs_tail_eval(Obj arg, Obj hdObj) {
    return EVAL_RECURS_TAIL;
}

Obj HdSubst = 0;

/* RecursiveEval will evaluate only a variable hdVar                        */
int single_var_eval(Obj hdVar, Obj hdObj) {
    int isVar = GET_TYPE_BAG(hdObj) == T_VAR;
    int useVarMap = GET_TYPE_BAG(hdVar) == T_VARMAP;

    if(isVar && 
       ((!useVarMap && hdVar==hdObj) || (useVarMap && PTR_BAG(hdVar)[0]==hdObj)))
    {
	if(useVarMap) HdSubst = PTR_BAG(hdVar)[1];
	else          HdSubst = EVAL(hdVar);

	/*if(GET_TYPE_BAG(HdSubst)==T_DELAY)
	  HdSubst = ExprFromDelay(HdSubst);*/
	return EVAL_SUBST;
    }
    else 
	return EVAL_NO_EVAL;
}

/* RecursiveEval evaluates only variables in hdVarList                      */
int multi_var_eval(Obj hdVarList, Obj hdObj) {
    int i;
    assert(GET_TYPE_BAG(hdVarList) == T_LIST);
    for(i = 1; i <= LEN_LIST(hdVarList); ++i) {
	int res = 
	    single_var_eval( ELM_LIST(hdVarList,i), hdObj );
	if(res!=EVAL_NO_EVAL)
	    return res;
    }
    return EVAL_NO_EVAL;
}

/****************************************************************************
**
*F  RecursiveEval(<hd>, <evalPred>, <hdPredArg>)  . . . recursive evaluation
**
**  'RecursiveEval' evaluates  an  expression <hd> or its parts recursively, 
**  the behavior is customized using a predicate function <evalPred>,  which
**  will be given <hdPredArg>  as  an argument,  and  its  return value will 
**  determine  whether  a  subexpression  will  be  evaluated,  an whether a 
**  recursive call will be made.
*/
Obj  _RecursiveEval ( Obj hd, int (*evalPred)(Obj,Obj), Obj hdPredArg );

Obj  RecursiveEval ( Obj hd, int (*evalPred)(Obj,Obj), Obj hdPredArg ) {
    Obj hdRes = _RecursiveEval(hd, evalPred, hdPredArg);
    RecursiveClearFlagFullMutable(hdRes, BF_DELAY_EV);
    return hdRes;
}

Obj  RecursiveEvalChildren ( Obj hd, int (*evalPred)(Obj,Obj), Obj hdPredArg ) {
    /* don't look at non-mutable types at T_RECNAM */
    if(!hd || GET_TYPE_BAG(hd) < T_MUTABLE || GET_TYPE_BAG(hd)==T_RECNAM) 
	return hd;
    else {
	unsigned int i;
	for ( i = NrHandles(GET_TYPE_BAG(hd), GET_SIZE_BAG(hd)); 0 < i; --i ) {
	    Obj hdSubObj = PTR_BAG(hd)[i-1];
	    hdSubObj = _RecursiveEval(hdSubObj, evalPred, hdPredArg);
	    SET_BAG(hd, i-1,  hdSubObj );
	}
	return hd;
    }
}

Obj  _RecursiveEval ( Obj hd, int (*evalPred)(Obj, Obj), Obj hdPredArg ) { 
    /*TRACE(hd);*/
    /* don't look at non-mutable types at T_RECNAM */
    if(!hd || GET_TYPE_BAG(hd) < T_MUTABLE || GET_TYPE_BAG(hd)==T_RECNAM || GET_TYPE_BAG(hd)==T_NAMESPACE) 
	return hd;
    /* prevent infinite recursion for circular structures */
    else if(GET_FLAG_BAG(hd, BF_DELAY_EV))
	return hd;
    else {
	unsigned int e;
	/* Never mark variables, they can be substituted, and the mark will stay forever */
	if(GET_TYPE_BAG(hd)!=T_VAR)
	    SET_FLAG_BAG(hd, BF_DELAY_EV);
	e = evalPred(hdPredArg, hd);
	if     (e == EVAL_RECURS_TAIL) {
	    hd = EVAL(hd);
	    hd = RecursiveEvalChildren(hd, evalPred, hdPredArg);
	}
	else if(e == EVAL_RECURS_HEAD) {
	    hd = RecursiveEvalChildren(hd, evalPred, hdPredArg);
	    hd = EVAL(hd);
	}
	else if(e == EVAL_NO_RECURS) {
	    hd = EVAL(hd);
	}
	else if(e == EVAL_NO_EVAL) {
	    hd = RecursiveEvalChildren(hd, evalPred, hdPredArg);
	}
	else if(e == EVAL_DELAY) {
	    hd = ExprFromDelay(hd);
	    hd = _RecursiveEval(hd, evalPred, hdPredArg);
	}
	else if(e == EVAL_SUBST) {
	    hd = HdSubst;
	}
	else if(e == EVAL_NOTHING) {
	    /* do nothing */
	}
	else 
	    assert("evalPred() returned bad value"==0);
	return hd;
    }
}

/****************************************************************************
**
*F  ExprFromDelay(<hd>) . . . . . . returns an expression from T_DELAY object
*F  DelayFromExpr(<hd>  . . . . . . makes a T_DELAY object from an expression
*F  RawDelayFromExpr(<hd>  . . .  makes a T_DELAY without inner D() expansion
*/

/* _ExpandD is mutually recursive with _DelayFromExprInPlace.  Its purpose
 *  is to find all calls to FunD (internal function 'D'), which correspond
 *  to construction of T_DELAY objects and construct these objects.
 *    
 *  This implements that D(D(x)) is a doubly delayed x, and not a delayed
 *  function call D(x).
 *
 *  _ExpandD is not used for RawDelayFromExpr.
 */
static Obj _DelayFromExprInPlace(Obj hd);
static int _funcIs(Obj hd, void *func) {
    if(hd == NULL)
	return 0;
    else if(GET_TYPE_BAG(hd)==T_VAR) {
	Obj val = PTR_BAG(hd)[0];
	if(val != NULL && HD_TO_FUNCINT(val) == func)
	    return 1;
	else return 0;
    }
    else if(HD_TO_FUNCINT(hd)==func) return 1;
    else return 0;
}

static Obj _ExpandD(Obj hd) {
    int i;
    /* don't look at non-mutable types and undefined variables */
    if(!hd || !IS_FULL_MUTABLE(hd)) return hd;

    /* avoid infinite loops in circular structures                  */
    if(GET_FLAG_BAG(hd, BF_DELAY)) return hd;

    SET_FLAG_BAG(hd, BF_DELAY);
    /* process children */
    for(i = NrHandles(GET_TYPE_BAG(hd), GET_SIZE_BAG(hd)); i > 0; --i) {
	Obj hdTmp = _ExpandD( PTR_BAG(hd)[i-1] );
	SET_BAG(hd, i-1,  hdTmp );
    }
    
    if(GET_TYPE_BAG(hd)==T_FUNCCALL && _funcIs(PTR_BAG(hd)[0], FunD) ) 
	return _DelayFromExprInPlace(PTR_BAG(hd)[1]);
    else if(GET_TYPE_BAG(hd)==T_FUNCCALL && _funcIs(PTR_BAG(hd)[0], FunDelay) ) 
	return EVAL(hd);
    else if(GET_TYPE_BAG(hd)==T_FUNCCALL && _funcIs(PTR_BAG(hd)[0], FunEval) ) {
	SET_BAG(hd, 0,  HdEvalFromD );
	return EVAL(hd);
    }
    else return hd;
}

static Obj _DelayFromExprInPlace(Obj hd) {
    Obj hdRes = NewBag(T_DELAY, SIZE_HD);
    Obj hdExp = _ExpandD( hd );
    SET_BAG(hdRes, 0,  hdExp ); 
    return hdRes;
}


Obj  ExprFromDelay ( Obj hdDelay ) {
    assert(GET_TYPE_BAG(hdDelay)==T_DELAY);
    return PTR_BAG(hdDelay)[0];
}

Obj  DelayFromExpr ( Obj hdExpr ) {
    Obj hdRes = _DelayFromExprInPlace(FullCopy(hdExpr));
    RecursiveClearFlagFullMutable(hdRes, BF_DELAY);
    return hdRes;
}

Obj  RawDelayFromExpr(Obj hd) {
    Obj hdRes = NewBag(T_DELAY, SIZE_HD);
    SET_BAG(hdRes, 0,  hd ); 
    return hdRes;
}

/****************************************************************************
**
*F  FunIsDelay( <obj> ) . . . . . . check whether an object is T_DELAY object
**
**  'FunIsDelay' implements the internal function 'IsDelay'
**  'IsDelay( <obj> )' return 'true' if <obj> is a T_DELAY object.
*/
Obj  FunIsDelay ( Obj hdCall ) {
    char * usage = "usage: IsDelay( <obj> )";
    Obj hdObj;
    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hdObj = EVAL( PTR_BAG(hdCall)[1] );
    if( GET_TYPE_BAG(hdObj) == T_DELAY ) 
	return HdTrue;
    else
	return HdFalse;
}

/****************************************************************************
**
*F  FunIsVarMap( <obj> ) . . . . . check whether an object is T_VARMAP object
**
**  'FunIsVarMap' implements the internal function 'IsVarMap'
**  'IsVarMap( <obj> )' return 'true' if <obj> is a T_VARMAP object.
*/
Obj  FunIsVarMap ( Obj hdCall ) {
    char * usage = "usage: IsVarMap( <obj> )";
    Obj hdObj;
    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hdObj = EVAL( PTR_BAG(hdCall)[1] );
    if( GET_TYPE_BAG(hdObj) == T_VARMAP ) 
	return HdTrue;
    else
	return HdFalse;
}
/****************************************************************************
**
*F  FunVarMap( <from>, <to> ) . . . . . . . . . . . . . . .  create a var map 
**
**  'FunVarMap' implements the internal function 'VarMap'
*/
Obj  FunVarMap ( Obj hdCall ) {
    char * usage = "usage: VarMap( <var_from>, <to> )";
    Obj hd1, hd2;
    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )  return Error(usage, 0,0);
    hd1 = INJECTION_D(EVAL(PTR_BAG(hdCall)[1]));
    hd2 = EVAL(PTR_BAG(hdCall)[2]);

    if( GET_TYPE_BAG(hd1) == T_VARAUTO ) EVAL(hd1);
    if( GET_TYPE_BAG(hd1) != T_VAR ) return Error(usage, 0, 0);
    return BinBag(T_VARMAP, hd1, hd2);
}

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
Obj  FunD ( Obj hdCall ) {
    char * usage = "usage: D( <expr> )";
    int argc = GET_SIZE_BAG(hdCall) / SIZE_HD;
    if(argc != 2 ) return Error(usage, 0,0);
    return DelayFromExpr(PTR_BAG(hdCall)[1]);
}

void _prepareAndCheckSubstList(Obj hdVars, char * usage, int open_delays) {
    UInt i;
    Obj ev;
    for(i=1; i <= LEN_LIST(hdVars); ++i) {
	Obj elm = ELM_LIST(hdVars,i);
	if(! IS_VAR_SUBST(elm) ) 
	    Error(usage, 0, 0);

	if ( GET_TYPE_BAG(elm) == T_VARMAP )
	    elm = FullShallowCopy(elm);
	else {
	    Obj map = NewBag(T_VARMAP, 2*SIZE_HD);
	    SET_BAG(map, 0,  elm );
	    ev = EVAL(elm);
	    SET_BAG(map, 1,  ev );
	    elm = map;
	}

	ev = EVAL(PTR_BAG(elm)[1]);
	SET_BAG(elm, 1,  ev );
	if ( open_delays && GET_TYPE_BAG(PTR_BAG(elm)[1]) == T_DELAY )
	  SET_BAG(elm, 1,  INJECTION_D(PTR_BAG(elm)[1]) );

	ASS_LIST(hdVars, i, elm);
    }
}

Obj  FunDelay ( Obj hdCall ) {
    char * usage = 
	"usage: Delay( <expr>, [ <var> | <var-list> ] )\n"
	"     Delays evaluation of expression <expr>, expanding only\n"
	"     values of variables <var> or all variables in <var-list>";
    int argc = GET_SIZE_BAG(hdCall) / SIZE_HD;
    if(argc < 2) 
	return Error(usage, 0,0);
    /* simplest case: just return the argument without evaluation           */
    else if(argc==2)  
	return DelayFromExpr(PTR_BAG(hdCall)[1]);
    /* evaluate specific variable references                                */
    else { /* argc >= 3 */
	Obj hd = FullCopy(PTR_BAG(hdCall)[1]);
	Obj hdVars = TailArgList(hdCall,2); 
	_prepareAndCheckSubstList(hdVars, usage, 1);
	return DelayFromExpr(
	      RecursiveEval(hd, multi_var_eval, hdVars));
    }
}

Obj  FunSubst ( Obj hdCall ) {
    char * usage = "usage: Subst( <expr>, [ <var> | <var-list> ] )\n"
    "     Delays evaluation of expression <expr>, expanding only\n"
    "     values of variables <var> or all variables in <var-list>";
    int argc = GET_SIZE_BAG(hdCall) / SIZE_HD;
    if(argc < 2) 
	return Error(usage, 0,0);
    /* simplest case: just return the evaluated argument                    */
    else if(argc==2)  
	return EVAL(ExprFromDelay(DelayFromExpr(PTR_BAG(hdCall)[1])));
    /* evaluate specific variable references                                */
    else { /* argc >= 3 */
	Obj hd = FullCopy(PTR_BAG(hdCall)[1]);
	Obj hdVars = TailArgList(hdCall,2); 
	/* do NOT open delays in variable substitutions, unlike in FunDelay */
	_prepareAndCheckSubstList(hdVars, usage, 0); 
	hd = RecursiveEval(hd, multi_var_eval, hdVars);
	return EVAL(hd);
    }
}

Obj  FunLet ( Obj hdCall ) {
    Obj hdCallNew = FullShallowCopy(hdCall);
    int len = GET_SIZE_BAG(hdCall) / SIZE_HD;
    SET_BAG(hdCallNew, 1,  PTR_BAG(hdCall)[len-1] );
    SET_BAG(hdCallNew, len-1,  PTR_BAG(hdCall)[1] );
    return FunSubst(hdCallNew);
}

/* void _bindLetList(Obj hdVars, char * usage) { */
/*     UInt i; */
/*     Obj ev; */
/*     for(i=1; i <= LEN_LIST(hdVars); ++i) { */
/* 	Obj elm = ELM_LIST(hdVars,i); */
/* 	if ( GET_TYPE_BAG(elm) == T_VARMAP ) */
/* 	    elm = FullShallowCopy(elm); */
/* 	else  */
/* 	    Error(usage, 0, 0); */

/* 	ev = EVAL(PTR_BAG(elm)[1]); */
/* 	SET_BAG(elm, 1,  ev ); */
/* 	ASS_LIST(hdVars, i, elm); */
/*     } */
/* } */
/* Obj  FunNewLet ( Obj hdCall ) { */
/*     char * usage = "usage: Let([ <var> | <var-list> ], <expr> )\n"; */
/*     int argc = GET_SIZE_BAG(hdCall) / SIZE_HD; */
/*     if(argc < 2)  */
/* 	return Error(usage, 0,0); */
/*     /\* simplest case: just return the evaluated argument                    *\/ */
/*     else if(argc==2)   */
/* 	return EVAL(ExprFromDelay(DelayFromExpr(PTR_BAG(hdCall)[1]))); */
/*     /\* evaluate specific variable references                                *\/ */
/*     else { /\* argc >= 3 *\/ */
/* 	Obj hd = PTR_BAG(hdCall)[argc-1]; */
/* 	_bindLetList(hdCall, usage); */
/* 	hd = EVAL(hd); */
/* 	_unbindLetList(hdCall, usage); */
/* 	return hd; */
/*     } */
/* } */

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
Obj  FunEval ( Obj hdCall ) {
    char * usage = "usage: Eval( <expr> )";
    Obj hd;
    /* get and check the argument                                           */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage, 0,0);
    hd = FullCopy( EVAL(PTR_BAG(hdCall)[1]) );
    if(GET_TYPE_BAG(hd)==T_DELAY)
	hd = ExprFromDelay(hd);
    return EVAL(hd); 
}

Obj  FunEvalFromD ( Obj hdCall ) {
    char * usage = "usage: Eval( <expr> )";
    Obj hd;
    /* get and check the argument                                           */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage, 0,0);
    hd = FullCopy( EVAL(PTR_BAG(hdCall)[1]) );
    if(GET_TYPE_BAG(hd)==T_DELAY)
	hd = ExprFromDelay(hd);
    return (hd); /* when Eval() is called from within D() constructor, final
		    evaluation step is unnecessary */
}

Obj  Backquote ( Obj hd ) {
    Obj res = NewBag(T_FUNCCALL, 2*SIZE_HD);
    SET_BAG(res, 0,  HdBackquote );
    SET_BAG(res, 1,  hd );
    return res;
}

Obj  FunEvalDeep ( Obj hdCall ) {
    char * usage = "usage: EvalDeep( <expr> )";
    Obj hd;
    /* get and check the argument                                           */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage, 0,0);
    hd = FullCopy( EVAL(PTR_BAG(hdCall)[1]) );
    if(GET_TYPE_BAG(hd)==T_DELAY)
	hd = ExprFromDelay(hd);
    return RecursiveEval(hd, recurs_head_eval,0); 
}

Obj  FunEvalVar ( Obj hdCall ) {
    char * usage = "usage: EvalVar( <delayed-obj>, <var1>, ..., <varN> )  or\n"
    "       EvalVar( <delayed-obj>, <var1> => <value1>, ..., <varN> => <valueN> )";
    int argc = GET_SIZE_BAG(hdCall) / SIZE_HD;
    if(argc < 3) 
	return Error(usage, 0,0);
    else {
	Obj hd = EVAL(PTR_BAG(hdCall)[1]);
	Obj hdVars;

	if(GET_TYPE_BAG(hd)!=T_DELAY) return Error(usage, 0, 0);
	hd = ExprFromDelay(hd);
	hd = FullCopy( hd );

	hdVars = TailArgList(hdCall, 2);
	_prepareAndCheckSubstList(hdVars, usage, 1);
	return DelayFromExpr(
	   RecursiveEval(hd, multi_var_eval, hdVars));
    }
}

Obj  FunEvalVarIndirect ( Obj hdCall ) {
    char * usage = "usage: EvalVarIndirect( <delayed-obj>, <varMap1>, ..., <varMapN> )\n"
	"       varMap arguments are evaluated, so VarMaps can be held in variables";
    int argc = GET_SIZE_BAG(hdCall) / SIZE_HD;
    if(argc < 3) 
	return Error(usage, 0,0);
    else {
	int i;
	Obj hd = EVAL(PTR_BAG(hdCall)[1]);
	Obj hdSecondArg = EVAL(PTR_BAG(hdCall)[2]);
	Obj hdVars;

	if(GET_TYPE_BAG(hd)!=T_DELAY) return Error(usage, 0, 0);
	hd = ExprFromDelay(hd);
	hd = FullCopy( hd );

	if(IS_LIST(hdSecondArg))  hdVars = hdSecondArg;
	else	                  hdVars = TailArgList(hdCall, 2);

	for(i=1; i <= LEN_LIST(hdVars); ++i) {
	    Obj elm = ELM_LIST(hdVars,i);
	    Obj ev;
	    elm = EVAL(elm);
	    if( GET_TYPE_BAG(elm) != T_VARMAP )
		return Error(usage, 0, 0);
	    else {
		/* varMaps are evaluated in-place, copy to leave original intact */
		elm = FullShallowCopy(elm);
		ev = EVAL(PTR_BAG(elm)[1]);
		if ( GET_TYPE_BAG(ev) == T_DELAY )
		    ev = INJECTION_D(ev);

		SET_BAG(elm, 1,  ev );
		ASS_LIST(hdVars, i, elm); 
	    }
	}
	return DelayFromExpr(
	   RecursiveEval(hd, multi_var_eval, hdVars));
    }
}

/****************************************************************************
**
*F  DelaySum(<hdL>,<hdR>) . . . . . .  construct a delayed sum of two objects
*F  DelayDiff(<hdL>,<hdR>) . .  construct a delayed difference of two objects
*F  DelayProd(<hdL>,<hdR>) . . . . construct a delayed product of two objects
*F  DelayQuo(<hdL>,<hdR>) . . . . construct a delayed quotient of two objects
*F  DelayMod(<hdL>,<hdR>) . . . . . .  construct a delayed mod of two objects
*F  DelayPow(<hdL>,<hdR>) . . . . .  construct a delayed power of two objects
** 
*F  EvDelay(<hd>) . . . . . . . . .  evaluate a delayed object (does nothing)
*F  PrDelay(<hd>) . . . . . . . . . . . . . . . . . .  print a delayed object
**
*/
Obj  DBinBag ( unsigned int type, Obj hdL, Obj hdR ) {
    if(GET_TYPE_BAG(hdL)==T_DELAY) hdL = ExprFromDelay(hdL);
    if(GET_TYPE_BAG(hdR)==T_DELAY) hdR = ExprFromDelay(hdR);
    return DelayFromExpr(BinBag(type,  hdL, hdR));
}

#define isNegation(x) (GET_TYPE_BAG(x)==T_DELAY && GET_TYPE_BAG(PTR_BAG(x)[0])==T_PROD && \
     (PTR_BAG(PTR_BAG(x)[0])[0]==INT_TO_HD(-1) || \
      PTR_BAG(PTR_BAG(x)[0])[1]==INT_TO_HD(-1)))

#define unnegate(x) ( (PTR_BAG(PTR_BAG(x)[0])[0]==INT_TO_HD(-1)) ? \
		      (PTR_BAG(PTR_BAG(x)[0])[1]) : (PTR_BAG(PTR_BAG(x)[0])[0]) )
		       
Obj  DelaySum  ( Obj hdL, Obj hdR ) { 
    if ( hdL == INT_TO_HD(0) ) return hdR;
    else if ( hdR == INT_TO_HD(0) ) return hdL;
    else if ( isNegation(hdR) )
	return DBinBag(T_DIFF,  hdL, unnegate(hdR)); 
    else if ( isNegation(hdL) )
	return DBinBag(T_DIFF,  hdR, unnegate(hdL)); 
    else return DBinBag(T_SUM,  hdL, hdR); 
}

Obj  DelayDiff ( Obj hdL, Obj hdR ) { 
    if ( hdL == INT_TO_HD(0) ) return hdR;
    else if ( hdR == INT_TO_HD(0) ) return hdL;
    else if ( isNegation(hdR) )
	return DBinBag(T_SUM,  hdL, unnegate(hdR)); 
    else return DBinBag(T_DIFF,  hdL, hdR); 
}

Obj  DelayProd ( Obj hdL, Obj hdR ) { 
    if ( hdL == INT_TO_HD(0) ) return INT_TO_HD(0);
    else if ( hdL == INT_TO_HD(1) ) return hdR;
    else if ( hdR == INT_TO_HD(0) ) return INT_TO_HD(0);
    else if ( hdR == INT_TO_HD(1) ) return hdL;
    else if ( hdL == INT_TO_HD(-1) && isNegation(hdR))
	return RawDelayFromExpr(unnegate(hdR));
    else if ( hdR == INT_TO_HD(-1) && isNegation(hdL))
	return RawDelayFromExpr(unnegate(hdL));
    else return DBinBag(T_PROD,  hdL, hdR); 
}

Obj  DelayQuo  ( Obj hdL, Obj hdR ) { 
    if ( hdL == INT_TO_HD(0) ) return INT_TO_HD(0);
    else if ( hdR == INT_TO_HD(0) ) 
	return Error("DelayQuo: Division by 0",0, 0);
    else if ( hdR == INT_TO_HD(1) ) 
	return hdL;
    else return DBinBag(T_QUO,  hdL, hdR); 
}

Obj  DelayMod  ( Obj hdL, Obj hdR ) { return DBinBag(T_MOD,  hdL, hdR); }

Obj  DelayPow  ( Obj hdL, Obj hdR ) { 
    if ( hdL == INT_TO_HD(0) ) return INT_TO_HD(0);
    else if ( hdL == INT_TO_HD(1) ) return INT_TO_HD(1);

    else if ( hdR == INT_TO_HD(0) ) return INT_TO_HD(1);
    else if ( hdR == INT_TO_HD(1) ) return hdL;
    else return DBinBag(T_POW,  hdL, hdR); 
}

Obj  EqDelay ( Obj hdL, Obj hdR ) { return (PTR_BAG(hdL)[0] == PTR_BAG(hdR)[0]) ? HdTrue : HdFalse; }
Obj  LtDelay ( Obj hdL, Obj hdR ) { return (PTR_BAG(hdL)[0] <  PTR_BAG(hdR)[0]) ? HdTrue : HdFalse; }
Obj  EvDelay ( Obj hd ) { return hd; }

/* Allows to manipulate VarMaps as first-class objects w/o evaluation */
Obj  EvVarMap( Obj hd ) { return hd; } 

extern Int prFull;
void PrDelay ( Obj hd ) {
    /* print function bags in the full form, so that output is reparseable */
    prFull = 1; 
    Pr("%2>D(",0,0);
    Print(ExprFromDelay(hd));
    Pr(")%2<",0,0);
    prFull = 0;
}

void PrVarMap ( Obj hd ) {
    Pr("%2>",0,0);
    Print(PTR_BAG(hd)[0]);
    Pr("%< %>=> ",0,0);
    Print(PTR_BAG(hd)[1]);
    Pr("%2<",0,0);
}

/****************************************************************************
**
*F  Children( <Obj> ) . . . . . . . . . . .  returns all subbags of given bag 
**
** Implements internal function  Children()  which  allows  inspection of GAP
** objects  from  within  the  interpreter. Using  Children  enables one, for 
** example to inspect function definitions, or even compile them.
*/
Obj  FunChildren ( Obj hdCall ) {
    char * usage = "usage: Children( <Obj> )";
    unsigned int i, siz;
    Obj hd;
    Obj children;
    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);

    hd = EVAL(PTR_BAG(hdCall)[1]); 
    siz =  GET_TYPE_BAG(hd)==T_INT  ?  0  :  NrHandles(GET_TYPE_BAG(hd), GET_SIZE_BAG(hd));

    children = NewList(1 + siz);
    for ( i = siz; 0 < i; --i ) {
	Obj hdSubObj = PTR_BAG(hd)[i-1];
	if(hdSubObj==0)
	    hdSubObj = StringToHd("null");
	else if(hdSubObj==HdVoid)
	    hdSubObj = StringToHd("void");
	SET_BAG(children, 1 + i,  hdSubObj );
    }
    hd = StringToHd(SizeType[GET_TYPE_BAG(hd)].name);
    SET_BAG(children, 1,  hd );
    return children;
}

Obj  FunChild ( Obj hdCall ) {
   char * usage = "usage: Child( <Obj>, <childNum> )";
    Obj   hd, hdChildNum;
    Int  num, siz;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )  return Error(usage, 0,0);
    /* <Obj> */
    hd = EVAL(PTR_BAG(hdCall)[1]); 
    /* <childNum> */
    hdChildNum = EVAL(PTR_BAG(hdCall)[2]); 
    if ( GET_TYPE_BAG(hdChildNum) != T_INT ) return Error(usage, 0, 0);
    num = HD_TO_INT(hdChildNum);

    siz =  GET_TYPE_BAG(hd)==T_INT  ?  0  :  NrHandles(GET_TYPE_BAG(hd), GET_SIZE_BAG(hd));
    /* in GAP lists are indexed starting with 1 */
    if ( num < 1 || num > siz ) 
	Error("<childNum> must be in [%d, %d] for the given object", NUM_TO_INT(1), siz);

    /* but in C they are indexed starting with 0 */
    hd = PTR_BAG(hd)[num-1];
    if(hd==0)
	return StringToHd("null");
    else if(hd==HdVoid)
	return StringToHd("void");
    else 
	return hd;
}

Obj  FunSetChild ( Obj hdCall ) {
    char * usage = "usage: SetChild( <Obj>, <childNum>, <newChildObj> )";
    Obj   hd, hdNewChild, hdChildNum;
    Int  num, siz;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 4 * SIZE_HD )  return Error(usage, 0,0);
    /* <Obj> */
    hd = EVAL(PTR_BAG(hdCall)[1]); 
    /* <childNum> */
    hdChildNum = EVAL(PTR_BAG(hdCall)[2]); 
    if ( GET_TYPE_BAG(hdChildNum) != T_INT ) return Error(usage, 0, 0);
    num = HD_TO_INT(hdChildNum);
    /* <newChildObj> */
    hdNewChild = EVAL(PTR_BAG(hdCall)[3]); 

    siz =  GET_TYPE_BAG(hd)==T_INT  ?  0  :  NrHandles(GET_TYPE_BAG(hd), GET_SIZE_BAG(hd));
    /* in GAP lists are indexed starting with 1 */
    if ( num < 1 || num > siz ) 
	Error("<childNum> must be in [%d, %d] for the given object", NUM_TO_INT(1), siz);

    /* but in C they are indexed starting with 0 */
    SET_BAG(hd, num-1,  hdNewChild ); 
    return HdVoid;
}

Obj  FunNumArgsLocals ( Obj hdCall ) {
    char * usage = "usage: NumArgsLocals( <func> )";
    Obj hd;
    Obj res;
    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hd = EVAL(PTR_BAG(hdCall)[1]); 
    if ( GET_TYPE_BAG(hd) != T_FUNCTION && GET_TYPE_BAG(hd) != T_METHOD ) return Error(usage, 0, 0);
    res = NewList(2);
    SET_BAG(res, 1,  INT_TO_HD( NUM_ARGS_FUNC(hd) ) );
    SET_BAG(res, 2,  INT_TO_HD( NUM_LOCALS_FUNC(hd) ) );
    return res;
}
Obj  FunNumArgs ( Obj hdCall ) {
    char * usage = "usage: NumArgs( <func> )";
    Obj hd;
    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hd = EVAL(PTR_BAG(hdCall)[1]); 
    if ( GET_TYPE_BAG(hd) != T_FUNCTION && GET_TYPE_BAG(hd) != T_METHOD ) return Error(usage, 0, 0);
    return INT_TO_HD( NUM_ARGS_FUNC(hd) );
}
Obj  FunNumLocals ( Obj hdCall ) {
    char * usage = "usage: NumArgs( <func> )";
    Obj hd;
    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hd = EVAL(PTR_BAG(hdCall)[1]); 
    if ( GET_TYPE_BAG(hd) != T_FUNCTION && GET_TYPE_BAG(hd) != T_METHOD ) return Error(usage, 0, 0);
    return INT_TO_HD( NUM_LOCALS_FUNC(hd) );
}

int  _CanBeFullyEvaluated ( Obj hd ) {
    int T, res=0;
    if(IS_INTOBJ(hd)) return 1;
    else if(!hd || GET_FLAG_BAG(hd, BF_DELAY_EV0)) return 0;
    else if(GET_FLAG_BAG(hd, BF_DELAY_EV)) return 1;

    T = GET_TYPE_BAG(hd);
    if(T==T_VAR) {
	Obj varValue = PTR_BAG(hd)[0];
	/* Undefined variables, can't be fully evaluated */
	if(varValue==NULL) 	               return 0;
	/* Variables defining functions are ok */
	else if(GET_TYPE_BAG(varValue)==T_FUNCTION)    return 1;
	/* Look at values of defined variables */
	else return _CanBeFullyEvaluated(varValue);
    }

    if(!IS_FULL_MUTABLE(hd) || T==T_MAKEFUNC)
	return 1;
    else if(T==T_FUNCCALL && _funcIs(PTR_BAG(hd)[0], FunD)) 
    	res = _CanBeFullyEvaluated(PTR_BAG(hd)[1]); /* Look at argument */
    else if(T==T_FUNCCALL && _funcIs(PTR_BAG(hd)[0], FunDelay)) 
    	res = _CanBeFullyEvaluated(_ExpandD(hd)); /* Look at argument */
    else if(T==T_DELAY) {
	res = _CanBeFullyEvaluated(PTR_BAG(hd)[0]);
    } 

    SET_FLAG_BAG(hd, (res==0) ? BF_DELAY_EV0 : BF_DELAY_EV);
    { /* else */
	unsigned int i;
	res = 1;
	for ( i = NrHandles(GET_TYPE_BAG(hd), GET_SIZE_BAG(hd)); 0 < i; --i ) {
	    Obj hdSubObj = PTR_BAG(hd)[i-1];
	    if(! _CanBeFullyEvaluated(hdSubObj)) { res = 0; break; }
	}
    }
    /* doesn't mark variables, there is possibility of mark staying forever */
    SET_FLAG_BAG(hd, (res==0) ? BF_DELAY_EV0 : BF_DELAY_EV);

    return res;
}

int  CanBeFullyEvaluated ( Obj hd ) {
    int res = _CanBeFullyEvaluated(hd);
    /*RecursiveClearFlagsFullMutable(hd, 3, BF_DELAY, BF_DELAY_EV, BF_DELAY_EV0);*/
    RecursiveClearFlagFullMutable(hd, BF_DELAY);
    RecursiveClearFlagFullMutable(hd, BF_DELAY_EV);
    RecursiveClearFlagFullMutable(hd, BF_DELAY_EV0);
    return res;
}

Obj  FunCanBeFullyEvaluated ( Obj hdCall ) {
    char * usage = "usage: CanBeFullyEvaluated( <expr> )";
    Obj hd;    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hd = (PTR_BAG(hdCall)[1]); 
    return CanBeFullyEvaluated(hd) == 1 ? HdTrue : HdFalse;
}

Obj  FuncFunccall( Obj hd ) {
    Obj f = PTR_BAG(hd)[0];
    if ( GET_TYPE_BAG(f) == T_VAR && PTR_BAG(f)[0] != NULL ) return PTR_BAG(f)[0];
    else return f;
}

Obj  _PartialEval ( Obj hd, Obj hdDontEvalFuncs ) {
    int T;
    if(!hd || IS_INTOBJ(hd) || GET_FLAG_BAG(hd, BF_DELAY_EV)) return hd;

    T = GET_TYPE_BAG(hd);
    if(T==T_VAR) {
	Obj varValue = PTR_BAG(hd)[0];
	/* Create T_DELAY for undefined variables */
	if(varValue==NULL) 
	    return RawDelayFromExpr(hd);
	/* Leave variables defining functions as is */
	else if(GET_TYPE_BAG(varValue)==T_FUNCTION)
	    return hd;
	/* Symbols defined as a simple cycle: x = D(x) */
	else if(INJECTION_D(varValue) == hd) 
	    return RawDelayFromExpr(hd);
	/* Expand defined variables */
	else return _PartialEval(FullCopy(varValue), hdDontEvalFuncs);
    }    

    /* don't mark variables, there is possibility of mark staying forever */
    SET_FLAG_BAG(hd, BF_DELAY_EV);
    
    if(!IS_FULL_MUTABLE(hd))
	return hd;
    else if(T==T_MAKEFUNC)
	return EVAL(hd);
    else if(T==T_FUNCCALL && _funcIs(PTR_BAG(hd)[0], FunD)) 
    	return _PartialEval(_ExpandD(hd), hdDontEvalFuncs);
    else if(T==T_FUNCCALL && _funcIs(PTR_BAG(hd)[0], FunDelay)) 
    	return _PartialEval(_ExpandD(hd), hdDontEvalFuncs);
    else if(T==T_DELAY) {
	return _PartialEval(PTR_BAG(hd)[0], hdDontEvalFuncs);
    }
    else {
	unsigned int i;
	int canFullyEval = 1;
	for ( i = NrHandles(GET_TYPE_BAG(hd), GET_SIZE_BAG(hd)); 0 < i; --i ) {
	    Obj hdSubObj = PTR_BAG(hd)[i-1];
	    Obj hdPE = _PartialEval(hdSubObj, hdDontEvalFuncs);
	    if(hdPE && GET_TYPE_BAG(hdPE) == T_DELAY) {
		Obj hdExpr = ExprFromDelay(hdPE);
		SET_BAG(hd, i-1,  hdExpr );
		canFullyEval = 0;
	    }
	    else SET_BAG(hd, i-1,  hdPE );
	}
	if(canFullyEval && ! (GET_TYPE_BAG(hd)==T_FUNCCALL && hdDontEvalFuncs != NULL &&
			      POS_LIST(hdDontEvalFuncs, FuncFunccall(hd), 0)!=0)) 
	    return EVAL(hd);
	else 
	    return RawDelayFromExpr(hd);
    }
}

Obj  PartialEval ( Obj hd, Obj hdDontEvalFuncs ) {
    Obj hdRes = _PartialEval(FullCopy(hd), hdDontEvalFuncs);
    /*RecursiveClearFlagsFullMutable(hdRes, 2, BF_DELAY, BF_DELAY_EV);*/
    RecursiveClearFlagFullMutable(hdRes, BF_DELAY);
    RecursiveClearFlagFullMutable(hdRes, BF_DELAY_EV);
    return hdRes;
}

Obj  FunPartialEval ( Obj hdCall ) {
    char * usage = "usage: PartialEval( <expr>, [ <dont-eval-funcs-list> ] )";
    Obj hd, hdDontEvalFuncs;
    int argc = GET_SIZE_BAG(hdCall) / (SIZE_HD);
    if ( argc < 2 || argc > 3 )  return Error(usage, 0,0);
    hd = PTR_BAG(hdCall)[1]; 
    if ( argc == 2 ) hdDontEvalFuncs = NULL;
    else hdDontEvalFuncs = EVAL(PTR_BAG(hdCall)[2]);

    return PartialEval(hd, hdDontEvalFuncs);
}

/****************************************************************************
**
*F  DelayedValueOf(<name>) . . . returns the delayed variable with given name
**
** This function makes possible to create a batch of variables automatically.
*/
Bag       FunDelayedValueOf (Bag hdCall)
{
    char * usage = "usage: DelayedValueOf(<name>)";
    Bag hdName, hdVar;
    char * name;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hdName = EVAL( PTR_BAG(hdCall)[1] );
    if( GET_TYPE_BAG(hdName) != T_STRING ) return Error(usage,0,0);
    name = (char*) PTR_BAG(hdName);
    hdVar = RawDelayFromExpr(FindIdent(name));
    return hdVar;
}

/****************************************************************************
**
*F  ConstraintD( <cond> ) . .make sure condition holds if it can be evaluated
**
*/
void ConstraintD(Obj cond) {
    /* evaluate condition but save copy to unevaluated, for nicer error
       reporting */
    Obj evalCond = PartialEval(cond, NULL);
    if(evalCond && GET_TYPE_BAG(evalCond)!=T_DELAY) {
	if(evalCond == HdTrue) 
	    return;
	else if(evalCond == HdFalse)
	    Error("Condition %g doesn't hold", (Int)cond, 0);
	else
	    Error("Condition must evaluate to a boolean", 0, 0);	
    }
    else {
	/*Pr("%g must hold\n", (Int)PTR_BAG(evalCond)[0], 0);*/
    }
}

Bag       FunConstraintD (Bag hdCall)
{
    char * usage = "usage: ConstraintD( <cond> )";
    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    ConstraintD(PTR_BAG(hdCall)[1]); /* note that condition is not evaluated here */
    return HdTrue;
}

/****************************************************************************
**
*F  CheckedD( <cond1>, <cond2>, ..., <expr> ) . . . .  constrained evaluation 
**
**  CheckedD()  is similar  to  Checked()  but  uses  ConstraintD() to verify 
**  conditions. This allows to deal with delayed expressions easier.
**
*/
Obj FunCheckedD ( Obj hdCall ) {
    char * usage = "usage: CheckedD( <cond1>, <cond2>, ..., <expr> )";
    Int nargs, i;
    /* get and check the argument                                          */
    nargs = GET_SIZE_BAG(hdCall) / SIZE_HD - 1;
    if ( nargs < 1 )  return Error(usage, 0,0);
    for ( i = 0; i < nargs-1; ++i ) 
        ConstraintD(PTR_BAG(hdCall)[1+i]);
    return EVAL(PTR_BAG(hdCall)[nargs]);
}


/****************************************************************************
**
*F  DetachFunc( <func> ) . . . . . . . . remove link to enclosing environment 
**
*/
Obj FunDetachFunc ( Obj hdCall ) {
    char * usage = "usage: DetachFunc( <func> )";
    Int nargs;
    Obj hd;
    /* get and check the argument                                          */
    nargs = GET_SIZE_BAG(hdCall) / SIZE_HD - 1;
    if ( nargs != 1 )  return Error(usage, 0,0);
    hd = EVAL(PTR_BAG(hdCall)[1]);
    if ( GET_TYPE_BAG(hd) != T_FUNCTION && GET_TYPE_BAG(hd) != T_METHOD )
	return Error("DetachFunc: <func> must be an interpreted function or a method", 0, 0);
    else {
	SET_BAG(hd, NUM_ARGS_FUNC(hd) + NUM_LOCALS_FUNC(hd) + 1,  0 );
	return hd;
    }
}


/****************************************************************************
**
*F  InitSPIRAL_DelayEv()  . . . . . initialize the delayed evaluation package
*/
void InitSPIRAL_DelayEv(void) {
    unsigned int        type;
    
    InitGlobalBag(&HdSubst, "HdSubst");
    InitGlobalBag(&HdEval, "HdEval");
    InitGlobalBag(&HdEvalFromD, "HdEvalFromD");
    InitGlobalBag(&HdBackquote, "HdBackquote");
    
    InstEvFunc(T_DELAY, EvDelay);
    InstPrFunc(T_DELAY, PrDelay);

    InstEvFunc(T_VARMAP, EvVarMap);
    InstPrFunc(T_VARMAP, PrVarMap);

    /* any_type OP delayed_type = delayed_type                              */
    /* this nested loop assigns some table entries twice, for simplicity    */
    for ( type = 0; type < T_VAR; ++type ) {
	if ( type != T_LIST && type != T_VECTOR && type != T_MATRIX ) {
	TabSum [type][T_DELAY] = DelaySum;
	TabDiff[type][T_DELAY] = DelayDiff;
	TabProd[type][T_DELAY] = DelayProd;
	TabQuo [type][T_DELAY] = DelayQuo;
	TabMod [type][T_DELAY] = DelayMod;
	TabPow [type][T_DELAY] = DelayPow;

	TabSum [T_DELAY][type] = DelaySum;
	TabDiff[T_DELAY][type] = DelayDiff;
	TabProd[T_DELAY][type] = DelayProd;
	TabQuo [T_DELAY][type] = DelayQuo;
	TabMod [T_DELAY][type] = DelayMod;
	TabPow [T_DELAY][type] = DelayPow;
	}
    }

    TabEq[T_DELAY][T_DELAY] = EqDelay;
    TabLt[T_DELAY][T_DELAY] = LtDelay;

    for ( type = T_LIST; type <= T_LISTX; type++ ) {
	if ( type != T_STRING && type != T_REC && type != T_BLIST ) {
	    TabSum [T_DELAY][type   ] = SumSclList;
	    TabSum [type   ][T_DELAY] = SumListScl;
	    TabDiff[T_DELAY][type   ] = DiffSclList;
	    TabDiff[type   ][T_DELAY] = DiffListScl;
	    TabProd[T_DELAY][type   ] = ProdSclList;
	    TabProd[type   ][T_DELAY] = ProdListScl;

	    //TabQuo [type   ][T_DELAY] = QuoLists;
	    //TabMod [T_DELAY][type   ] = ModLists;
	}
    }

    /* NOTE: What to do about Equal /  LessThen operators ?
    //    for ( typeL = 0; typeL < EV_TAB_SIZE; ++typeL ) {
    //        for ( typeR = 0; typeR <= typeL; ++typeR ) {
    //            TabEq[typeL][typeR] = IsFalse;
    //            TabLt[typeL][typeR] = IsFalse;
    //        }
    //        for ( typeR = typeL+1; typeR < EV_TAB_SIZE; ++typeR ) {
    //            TabEq[typeL][typeR] = IsFalse;
    //            TabLt[typeL][typeR] = IsTrue;
    //        }
    //    }
    */

    /**/ GlobalPackage2("spiral", "delay"); /**/
    InstIntFunc( "D",                    FunD ); 
    InstIntFunc( "Delay",                FunDelay ); 
    InstIntFunc( "IsDelay",              FunIsDelay ); 
    InstIntFunc( "IsVarMap",             FunIsVarMap ); 
    InstIntFunc( "VarMap",               FunVarMap ); 
    InstIntFunc( "Eval",                 FunEval );
    InstIntFunc( "EvalDeep",             FunEvalDeep );
    InstIntFunc( "EvalVar",              FunEvalVar );
    InstIntFunc( "EvalVarIndirect",      FunEvalVarIndirect );

    InstIntFunc( "Subst",                FunSubst ); 
    InstIntFunc( "Let",                  FunLet ); 

    InstIntFunc( "PartialEval",          FunPartialEval );
    InstIntFunc( "DelayedValueOf",       FunDelayedValueOf );
    InstIntFunc( "CanBeFullyEvaluated",  FunCanBeFullyEvaluated );

    InstIntFunc( "Child",                FunChild );
    InstIntFunc( "Children",             FunChildren );
    InstIntFunc( "SetChild",             FunSetChild );
    InstIntFunc( "NumArgsLocals",        FunNumArgsLocals );
    InstIntFunc( "NumLocals",            FunNumLocals );
    InstIntFunc( "NumArgs",              FunNumArgs );

    InstIntFunc( "ConstraintD",          FunConstraintD );
    InstIntFunc( "CheckedD",             FunCheckedD );
    InstIntFunc( "DetachFunc",           FunDetachFunc );

    SET_FLAG_BAG(FindIdent("Subst"),   BF_NO_WARN_UNDEFINED);
    SET_FLAG_BAG(FindIdent("Let"),     BF_NO_WARN_UNDEFINED);
    SET_FLAG_BAG(FindIdent("Delay"),   BF_NO_WARN_UNDEFINED);
    SET_FLAG_BAG(FindIdent("EvalVar"), BF_NO_WARN_UNDEFINED);
    SET_FLAG_BAG(FindIdent("D"),       BF_NO_WARN_UNDEFINED);

    HdEval = InstIntFunc("__HdEval", FunEval);
    HdEvalFromD = InstIntFunc("__HdEvalFromD", FunEvalFromD);
    //HdEval = NewBag(T_FUNCINT, SIZE_HD);
    //HdEvalFromD = NewBag(T_FUNCINT, SIZE_HD);
    HdBackquote = FindIdent("Eval");
    //HD_TO_FUNCINT(HdEvalFromD) = FunEvalFromD;
    //HD_TO_FUNCINT(HdEval) = FunEval;
    /**/ EndPackage(); /**/
}

/****************************************************************************
**
*E  Emacs . . . . . . . . . . . . . . . . . . . . . . . local emacs variables
**
**  Local Variables:
**  c-basic-offset:     4
**  outline-regexp:     "*A\\|*F\\|*V\\|*T\\|*E"
**  fill-column:        76
**  fill-prefix:        "**  "
**  End:
*/
