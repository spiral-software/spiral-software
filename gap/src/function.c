/****************************************************************************
**
*A  function.c                  GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This package contains the functions  that  mainly  deal  with  functions.
**
*/

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "scanner.h"             /* reading of tokens and printing  */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */

#include        "list.h"                /* generic list package            */
#include        "plist.h"               /* plain list package              */
#include        "record.h"              /* 'HdTilde'                       */
#include        "statemen.h"            /* 'HdStat', 'StrStat'             */
#include        "spiral.h"              /* Try, Catch, exc                 */

#include        "function.h"            /* declaration file of the package */
/* #include        "flags.h" */			// defns & decls moved to memmgr.h
#include        "spiral_delay_ev.h"     /* PROJECTION_D                    */
#include        "idents.h"              /* FindRecname                     */
#include        "args.h"                /* NewList                         */
#include        "debug.h"
#include		"GapUtils.h"


/*V HdCallRecname */
Bag HdCallRecname;


/****************************************************************************
**
*V  HdExec  . . . . . . . . . . . . . . . handle of the topmost execution bag
**
**  'HdExec' is the handle of the topmost execution bag on the execution  bag
**  linked list.  For every active function there is  one  execution  bag  on
**  this list.  'HdExec' is the execution bag of the current function.
**
**  The execution bag list is the equivalent  of  the  stack  in  programming
**  languages like C or Pascal.  'HdExec' is therefor the equivalent  of  the
**  Stackpointer in those languages.
**
**  To be precise a function execute bag has the following form:
**
**      handle 0:               handle of enclosing environment.
**      handle 1:               handle 'HdTrue' if this frame is current.
**      handle 2:               handle of function definition.
**      handle 3:               handle of function call bag, debug only.
**      handle 4:               handle of calling enviroment, debug only.
**      handle EXEC_ARG_START..n+EXEC_ARG_START-1:          handles of old arguments.
**      handle n+EXEC_ARG_START..n+m+EXEC_ARG_START-1:      handles of old local variables.
**      handle n+m+EXEC_ARG_START..2*n+m+EXEC_ARG_START-1:  handles of new arguments.
*/
Bag       HdExec;


/****************************************************************************
** Limits
**
*V RecursionLimit . . . . . . . . . . . . . . . . . . maximum recursion depth
*V RecursionDepth . . . . . . . . . . . . . . . . . . current recursion depth
*/
UInt   RecursionLimit = 4096;
UInt   RecursionLimitIncreased = 0;
volatile UInt   RecursionDepth;

/****************************************************************************
**
*V  IsProfiling . . . . . . . . . . . . . .  is 1 if profiling is switched on
*V  HdTimes . . .  handle of the list that contains the profiling information
*V  Timesum . .  total time spent in all functions that have completed so far
**
**  'IsProfiling' is 1 if profiling is enabled and 0 otherwise.
**
**  'HdTimes'  is  the  handle  of  the  list  that  contains  the  profiling
**  information.  This list contains for every function the following entries
**
**      <function>      handle of the function bag (or body of the function)
**      <name>          handle of the first entry in the function call
**      <count>         number of times this function was called
**      <time>          time spent in this function without its children
**      <total>         time spent in this function with its childer
**
**  'Timesum' is the total time spent in all functions that have completed so
**  far.  When a function  is  called  the  current  value  of  'Timesum'  is
**  remembered.  When the function completes 'Timesum' - <old>  is  the  time
**  spent in all childeren of this function.  If this is subtracted from  the
**  total time spent in this function we have the time spent in this function
**  without its children.
*/
Int            IsProfiling;
Bag       HdTimes;
UInt   Timesum;


/****************************************************************************
**
*F  ChangeEnv( <hdEnv> )  . . . .  change the environment for a function call
**
**  'ChangeEnv' changes the environment for a function call.  A *environment*
**  is  the set of  bindings of  identifiers to variables.   GAP has  lexical
**  binding, i.e., the environment in effect  when a function is created, the
**  so  called *definition  environment*,  determines the  variable bindings.
**  Thus when a  function is called its   definition environment is made  the
**  current environment.   When the function  terminates the old environment,
**  the so called *execution environment* is made current again.
**
**  An environment is stored  as a linked list  of exec-bags.  Every exec-bag
**  contains, among other things, the changes that a certain function made to
**  the environment.  I.e., when a function is  called it introduces a set of
**  new  arguments and local variables.   If the function was already active,
**  i.e., if the call was recursive, the exec-bag remembers the old values of
**  the  variables.  If  the  function was  not  already active, the exec-bag
**  remembers the fact that the variables had no values prior to the call.
**
**  The following picture should make the  operation  of  'ChangeEnv'  clear:
**
**      <hdEnv>  -> <exec 1> -> <exec 2> -\
**                                         \
**      'HdExec' -> <exec 3> -> <exec 4> ---+-> <exec 5> -> <exec 6> ... -> 0
**
**  'HdExec' is the handle of the current environment.  <hdEnv> is the handle
**  of an environment of a function that is  just beeing called.  'ChangeEnv'
**  must now change  the environment from 'HdExec' to  <hdEnv>.  To do  so it
**  must undo the changes stored in <exec 3> and  <exec 4> and then must redo
**  the changes stored in <exec 2> and <exec 1>, in that order.
**
**  Note that functions which are defined globally  can not access non-local,
**  non-global  variables.  Therefor it makes no  difference in  which such a
**  function  is executed.  In this  case  'EvFunccall'  does not change  the
**  environment at all.  Thus instead of:
**
**      <hdEnv> -> <exec 1> -------------\
**                                        \
**      'HdExec' -> <exec 3> -> <exec 4> --+-> 0
**
**  'EvFuncall' acts as if the situation was:
**
**      <hdEnv> -> <exec 1> -\
**                            \
**      'HdExec' --------------+-> <exec 3> -> <exec 4> -> 0
*/

void            ChangeEnv (Bag hdEnv, int flag)
{
    register Bag  hdDo, hdComm, hdTmp, hdUndo;
    register Bag  * ptUndo,  * ptDef,  * ptDo;
    register short      nr,  i;

    /* first walk down the new chain until we find a active exec bag       */
    /* we reverse the links, so we can walk back later                     */
    hdDo   = 0;
    hdComm = hdEnv;
    while ( hdComm != 0 && PTR_BAG(hdComm)[1] != HdTrue ) {
        hdTmp          = PTR_BAG(hdComm)[0];
        SET_BAG(hdComm, 0,  hdDo );
        hdDo           = hdComm;
        hdComm         = hdTmp;
    }

    /* then we undo all changes from the topmost down to the common exec   */
    hdUndo = HdExec;
    while ( hdUndo != hdComm ) {
        int nrArg, nrLoc;
        
        ptUndo = PTR_BAG(hdUndo);
        ptDef  = PTR_BAG(ptUndo[2]);
        ACT_NUM_ARGS_FUNC(ptUndo[2], nrArg);
        ACT_NUM_LOCALS_FUNC(ptUndo[2], nrLoc);

        if (!GET_FLAG_BAG(hdUndo, BF_ON_CALLSTACK) || (flag==CEF_DBG_DOWN && hdUndo==HdExec)) { // if not on the call stack
            if (flag == CEF_CLEANUP) {
                for ( i = 1; i <= nrArg + nrLoc; ++i ) {
                    hdTmp = ptUndo[EXEC_ARGS_START + i - 1];
                    ptUndo[EXEC_ARGS_START + i - 1] = (GET_FLAG_BAG(ptDef[i], BF_ENV_VAR)) ? VAR_VALUE(ptDef[i]) : 0 ;
                    SET_VAR_VALUE(ptDef[i], hdTmp);
                }
                for ( i = nrArg+nrLoc+1; i <= 2*nrArg+nrLoc; ++i ) {
                    ptUndo[EXEC_ARGS_START + i - 1] = 0;
                }
            } else {
                nr = nrArg + nrLoc;
                for ( i = 1; i <= nr; ++i ) {
                    hdTmp = ptUndo[EXEC_ARGS_START + i - 1];
                    ptUndo[EXEC_ARGS_START + i - 1] = VAR_VALUE(ptDef[i]);
                    SET_VAR_VALUE(ptDef[i], hdTmp);
                }
            }
        }
        ptUndo[1] = HdFalse;
        //  CHANGED_BAG(hdUndo);
        hdUndo    = ptUndo[0];
    }

    /* then we redo all changes from the common up to the new topmost exec */
    /* hdDo : stack frame */
    while ( hdDo != 0 ) {
        ptDo   = PTR_BAG(hdDo);
        if (!GET_FLAG_BAG(hdDo, BF_ON_CALLSTACK) || (flag==CEF_DBG_UP && hdDo==hdEnv) ) {
	    int nrArg, nrLoc;
	    ACT_NUM_ARGS_FUNC(ptDo[2], nrArg);
	    ACT_NUM_LOCALS_FUNC(ptDo[2], nrLoc);
	    nr = nrArg+nrLoc;
	    ptDef  = PTR_BAG(ptDo[2]);
            for ( i = 1; i <= nr; ++i ) {
                hdTmp = ptDo[EXEC_ARGS_START+i-1]; /* new value */
                ptDo[EXEC_ARGS_START+i-1] = VAR_VALUE(ptDef[i]);
                SET_VAR_VALUE(ptDef[i],  hdTmp);
            }
        }
        ptDo[1] = HdTrue;
        hdTmp   = ptDo[0];
        ptDo[0] = hdComm;
        hdComm  = hdDo;
        //  CHANGED_BAG(hdDo);
        hdDo    = hdTmp;
    }

    /* reflect the new environment in HdExec                               */
    HdExec = hdComm;
}


/****************************************************************************
**
*F  EvFunccall( <hdCall> )  . . . . . . . . . . . . evaluates a function call
**
**  'EvFunccall' evaluates the function call with  the  handle  <hdCall>  and
**  returns the value returned by the function or 'HdVoid'  if  the  function
**  did not return any value at all.
**
**  The function call bag <hdCall has the following form:
**
**      handle 0:               handle of the function definition bag.
**      handle 1.. :            handles of arguments (not yet evaluated).
**
**  'EvFunccall' first creates a new execute bag.  Then it evaluates the  new
**  arguments, and puts the values in the execute bag.  Then it saves the old
**  values of the arguments and local variables in the execute bag.  Then  it
**  calles 'ChangeEnv' to copy the new values from the execute bag  into  the
**  variables.  Now the binding is complete, and  'EvFunccall'  executes  the
**  statement sequence.  After that 'EvFunccall' calls 'ChangeEnv'  again  to
**  restore the old values from the execute bag.
*/
int IsMethodFunc( Bag hdFunc ) {
    return GET_TYPE_BAG(hdFunc)==T_METHOD;
}

UInt EvFunccallProfiling_begin(Bag hdDef, UInt* time) {
    UInt sime = 0;
    Int i;
    IsProfiling = 1;
    *time = SyTime()-Timesum;
    for ( i = 0; i < GET_SIZE_BAG(HdTimes)/SIZE_HD; i += 5 ) {
	if ( PTR_BAG(HdTimes)[i] == hdDef ) {
	    sime = SyTime() - HD_TO_INT( PTR_BAG(HdTimes)[i+4] );
            break;
        }
    }
    if ( i == GET_SIZE_BAG(HdTimes)/SIZE_HD ) {
	sime = SyTime();
    }
    return sime;
}


void EvFunccallProfiling_end(Bag hdDef, Bag hdCall, UInt time, UInt sime) {
    Int i, j;
    Bag hd;
    time = SyTime()-Timesum-time; Timesum += time;
    for ( i = 0; i < GET_SIZE_BAG(HdTimes)/SIZE_HD; i += 5 ) {
	if ( PTR_BAG(HdTimes)[i] == hdDef ) {
	    SET_BAG(HdTimes, i+2, INT_TO_HD(HD_TO_INT(PTR_BAG(HdTimes)[i+2])+1) );
            SET_BAG(HdTimes, i+3, INT_TO_HD(HD_TO_INT(PTR_BAG(HdTimes)[i+3])+time) );
            SET_BAG(HdTimes, i+4, INT_TO_HD(SyTime()-sime) );
            break;
        }
    }
    if ( i == GET_SIZE_BAG(HdTimes)/SIZE_HD ) {
        hd = PTR_BAG(hdCall)[0];

	Resize( HdTimes, GET_SIZE_BAG(HdTimes) + 5*SIZE_HD );
        SET_BAG(HdTimes, i,  hdDef );
        SET_BAG(HdTimes, i+1,  hd );
        SET_BAG(HdTimes, i+2,  INT_TO_HD(1) );
        SET_BAG(HdTimes, i+3,  INT_TO_HD(time) );
        SET_BAG(HdTimes, i+4,  INT_TO_HD(SyTime()-sime) );
    }
    while ( 0 < i
	&& (Int)PTR_BAG(HdTimes)[i-2] < (Int)PTR_BAG(HdTimes)[i+3] ) {
        hd = PTR_BAG(HdTimes)[i-5];
        SET_BAG(HdTimes, i-5,  PTR_BAG(HdTimes)[i] );
        SET_BAG(HdTimes, i,  hd );
        hd = PTR_BAG(HdTimes)[i-4];
        SET_BAG(HdTimes, i-4,  PTR_BAG(HdTimes)[i+1] );
        SET_BAG(HdTimes, i+1,  hd );
        hd = PTR_BAG(HdTimes)[i-3];
        SET_BAG(HdTimes, i-3,  PTR_BAG(HdTimes)[i+2] );
        SET_BAG(HdTimes, i+2,  hd );
        hd = PTR_BAG(HdTimes)[i-2];
        SET_BAG(HdTimes, i-2,  PTR_BAG(HdTimes)[i+3] );
        SET_BAG(HdTimes, i+3,  hd );
        hd = PTR_BAG(HdTimes)[i-1];
        SET_BAG(HdTimes, i-1,  PTR_BAG(HdTimes)[i+4] );
        SET_BAG(HdTimes, i+4,  hd );
        i -= 5;
    }
}

/* Try {} Catch {} block moved out from EvFunccall to not confuse msvc and 
** intel compilers. EvFunccall + EvFunccallTryCatchEval stack footprint is
** much smaller than of EvFunccall alone with Try/Catch block inside.
*/

Bag EvFunccallTryCatchEval( Bag hdDef, Bag hdExec) {
    /* remember the old value of '~' to recover later                      */
    Bag		hdRes = 0;
    exc_type_t  e = 0;
    Bag		hdTilde = PTR_BAG(HdTilde)[0];
    SET_BAG(HdTilde, 0,  0 );

    Try {
        RecursionDepth++;
        /* well here's what all is about                                       */
        hdRes =  EVAL( PTR_BAG(hdDef)[0] );
        if ( hdRes == HdReturn )
            hdRes = PTR_BAG(hdRes)[0];
        else
            hdRes = HdVoid;
    } Catch(e) {
        /* restore old environment                                             */
        CLEAR_FLAG_BAG(hdExec, BF_ON_CALLSTACK);
        /* recover the value of '~'                                            */
        SET_BAG(HdTilde, 0,  hdTilde );

        if ( RecursionLimitIncreased ) {
            RecursionLimit = RecursionLimit >> 1;
            RecursionLimitIncreased--;
        }
        RecursionDepth--;
        Throw(e);
    }
    RecursionDepth--;

    /* recover the value of '~'                                            */
    SET_BAG(HdTilde, 0,  hdTilde );
    return hdRes;
}

#define INIT_FAST_CALL()

Bag       EvFunccall (Bag hdCall)
{
    Bag           hdDef,  hdExec = 0,  hdRes = 0, hdOld = 0, hdRecElm = 0;
    Bag           hd = 0, hdMethodSelf = 0;
    Bag           hdTilde = 0;
    Bag*	  pt = 0;
    Bag*	  ptSrc = 0;
    int			nrArg = 0,  nrLoc = 0,  i = 0,  trace = 0, evalArgs = 0;
    UInt       time = 0,  sime = 0, ftype = 0;


    if (EvalStackTop>=EVAL_STACK_COUNT && !InBreakpoint) {
	DbgBreak("Evaluation stack overflow. GAP compiled with %d evaluation stack depth. Change EVAL_STACK_COUNT in eval.h if you really need more.", EVAL_STACK_COUNT, 0);
	return HdVoid;
    }

    hdDef = PTR_BAG(hdCall)[0];
    
    /* Method calls <uneval_object>.method()                               */
    if( GET_TYPE_BAG(hdDef) == T_RECELM  ) {
        Bag hdM = PTR_BAG(hdDef)[1]; /* save method ref */
        hdMethodSelf = EVAL(PTR_BAG(hdDef)[0]); /* evaluate object */

        /* Create new hdDef with evaluated object, original hdDef can't be *
         * touched (!!), function calls can reside in function bodies, and *
         * modifying a function call object will alter function body       */

        hdDef = NewBag(T_RECELM, GET_SIZE_BAG(hdDef));

	pt = PTR_BAG(hdDef);
	pt[0] = hdMethodSelf;
        pt[1] = RecnameObj(hdM);
        
        hdRecElm = hdDef;
    }

    hdDef = EVAL(hdDef);
    ftype = GET_TYPE_BAG(hdDef);

    if ( ftype == T_REC ) {
      /* look for __call__ */
      Obj  tmp = 0;
      Obj *hd = FindRecnameRec(hdDef, HdCallRecname, &tmp);
      if ( hd == 0 )
        DbgBreak("__call__ field not found, can't use this record as a function", 0, 0);
      else {
        hdMethodSelf = hdDef;
        hdDef = hd[1]; /* function definition is record's __call__ field */
        ftype = GET_TYPE_BAG(hdDef);

	if (hdRecElm==0)
	    hdRecElm = NewBag(T_RECELM, 2*SIZE_HD);
        pt = PTR_BAG(hdRecElm);
	pt[0] = hdMethodSelf;
        pt[1] = HdCallRecname;
      }
    }

    if ( ! IsMethodFunc(hdDef) )
        hdMethodSelf = 0;
    else if( hdMethodSelf != 0 ) {
        /* Method calls have 'self' (origin object) as first argument */
        Bag hdCallNew;

	hdCallNew = NewBag(T_FUNCCALL, GET_SIZE_BAG(hdCall) + SIZE_HD);
	pt = PTR_BAG(hdCallNew);
	ptSrc = PTR_BAG(hdCall);
	pt[0] = hdRecElm; 
        pt[1] = hdMethodSelf;
        for(i=1; i < GET_SIZE_BAG(hdCall)/SIZE_HD; ++i) {
            pt[i+1] = ptSrc[i];
        }
        hdCall = hdCallNew;
        SET_FLAG_BAG(hdCall, BF_METHCALL);
    }

    /* treat the special case of internal functions                        */
    if ( ftype == T_FUNCINT ) {
        if (InDebugMode) 
            CheckBreakpoints(hdDef, 0);
        if ( IsProfiling==0 ) {
            return (** (Bag(**)())PTR_BAG(hdDef)) ( hdCall );
        } else {
            sime = EvFunccallProfiling_begin(hdDef, &time);
            hdRes = (** (Bag(**)())PTR_BAG(hdDef)) ( hdCall );
            EvFunccallProfiling_end(hdDef, hdCall, time, sime);
            return hdRes ;
        }
    }
    
    if ( ftype != T_FUNCTION && ftype != T_METHOD)
        return Error("Call: %g\nFunction: <function> must be a function",(Int)hdCall,0);
    
    /* compute the number of arguments and locals                          */
    trace = 0;
    nrArg = NUM_ARGS_FUNC(hdDef);
    nrLoc = NUM_LOCALS_FUNC(hdDef);
    evalArgs = ! GET_FLAG_BAG(hdDef, BF_UNEVAL_ARGS);

    /* handle functions with variable number of arguments -> function(arg) */
    if ( nrArg == -1 ) {
        hdRes = NewBag( T_LIST, SIZE_PLEN_PLIST( GET_SIZE_BAG(hdCall)/SIZE_HD-1 ) );
        SET_LEN_PLIST( hdRes, GET_SIZE_BAG(hdCall)/SIZE_HD-1 );
        for ( i = 1; i < GET_SIZE_BAG(hdCall)/SIZE_HD; i++ ) {
            hd =  PTR_BAG(hdCall)[i];
            if ( evalArgs ) hd = EVAL(hd);
            else hd = PROJECTION_D(hd);
            if ( GET_TYPE_BAG(hd) == T_VOID )
                hd = Error("illegal void argument",0,0);
            SET_ELM_PLIST( hdRes, i, hd );
        }
        nrArg = 1;
        trace = 2;
    }
    else if ( nrArg != GET_SIZE_BAG(hdCall) / SIZE_HD - 1 )
        return Error("Call: %g\nFunction: number of args must be %d",(Int)hdCall, (Int)nrArg);

    /* check if the function is to be traced                               */
    if ( nrLoc < 0 ) {
        trace |= 1;  nrLoc = -nrLoc-1;
        Pr("\n%2>",0,0);  Print( PTR_BAG(hdCall)[0] );  Pr("%<( ",0,0);
    }

    /* Now create the new execute bag                                      */
  
    hdExec = NewBag( T_EXEC, SIZE_HD*(2*nrArg+nrLoc+EXEC_ARGS_START) );

    /* $$let$$ */
    /* enter all relevant information into the execbag                     */
    pt = (Bag*)PTR_BAG(hdExec);
    pt[0] = PTR_BAG(hdDef)[nrArg+nrLoc+1];
    pt[1] = HdFalse;           /* this frame is not yet current   */
    pt[2] = hdDef;             /* function definition             */
    pt[3] = hdCall;            /* function call, for debug only   */
    pt[4] = HdExec;            /* calling environment, dbg only   */
    /* enter the new evaluated arguments in the execbag                    */
    for ( i = 1; i <= nrArg; ++i ) {
        if ( ! (trace & 2) ) {
            hdRes = PTR_BAG(hdCall)[i];
            if ( evalArgs ) hdRes = EVAL(hdRes);
            else hdRes = PROJECTION_D(hdRes);
        }
        if ( GET_TYPE_BAG(hdRes) == T_VOID )
            hdRes = Error("illegal void argument",0,0);
        SET_BAG(hdExec, EXEC_ARGS_START+i-1,  hdRes );
        SET_BAG(hdExec, EXEC_ARGS_START + nrArg + nrLoc + i - 1,  hdRes );
        if ( trace & 1 ) {
            Pr("%>",0,0);  Print( hdRes );
            if ( i < nrArg )  Pr("%<, ",0,0);
            else              Pr("%< )",0,0);
        }
    }
    
    if (InDebugMode) 
        CheckBreakpoints(hdDef, hdExec);
    /* $$let$$ */

    if ( RecursionDepth == RecursionLimit ) {
        RecursionLimitIncreased++;
        RecursionLimit = RecursionLimit << 1; /* increase the limit for error handling */
        DbgBreak("Recursion limit of %d reached", RecursionLimit >> 1, 0);
    }

    /* If there are timed functions compute the timing                     */
    if ( IsProfiling ) sime = EvFunccallProfiling_begin(PTR_BAG(hdDef)[0], &time);
    /* And now change the environment                                      */
    hdOld = HdExec;
    ChangeEnv( hdExec, CEF_NONE );
    // mark hdExec as it's on the call stack
    SET_FLAG_BAG(hdExec, BF_ON_CALLSTACK);
    // push hdExec to the EVAL call stack
    EVAL_STACK_PUSH(hdExec);
    // save hdExec position in the EVAL call stack
    SET_BAG(hdExec, EXEC_EVAL_STACK,  INT_TO_HD(EvalStackTop) );
    
    hdRes = EvFunccallTryCatchEval(hdDef, hdExec);
    
    /* restore old environment                                             */

    pt = (Bag*)PTR_BAG(hdExec);
    pt[3] = 0;
    pt[4] = 0;
    pt[EXEC_EVAL_STACK] = 0;

    EVAL_STACK_POP;
    CLEAR_FLAG_BAG(hdExec, BF_ON_CALLSTACK);
    ChangeEnv( hdOld, CEF_CLEANUP );

    /* If there are timed functions compute the timing                     */
    if ( IsProfiling == 1 ) EvFunccallProfiling_end(PTR_BAG(hdDef)[0], hdCall, time, sime);
    
    /* If the function is traced, print the return value                   */
    if ( trace & 1 ) {
        Pr("\n%>",0,0); Print( PTR_BAG(hdCall)[0] );  Pr("%< returns ",0,0);
        if ( hdRes != HdVoid )  Print( hdRes );
        Pr("%< ",0,0);
    }

    return hdRes;
}


/****************************************************************************
**
*F  EvFunction( <hdFun> ) . . . . . . . . . . . . . . . . evaluate a function
**
**  'EvFunction' returns the value of the function <hdFun>.  Since  functions
**  are constants and thus selfevaluating it just returns <hdFun>.
*/
Bag       EvFunction (Bag hdDef)
{
    return hdDef;
}


/****************************************************************************
**
*F  EvMakefunc( <hdFun> ) . . . . . . . . . . . . . . . . . . make a function
*F  EvMakemeth( <hdFun> ) . . . . . . . . . . . . . . . . . . . make a method
**
**  'EvMakefunc' makes a function, i.e., turns a  variable  function  into  a
**  constant one.  GAP has lexical binding.  This means that the binding from
**  identifiers to variables is determined by the environment that was active
**  when a function was created and not by the one active  when  the function
**  is  executed.  'ChangeEnv'  performs  the  task  of  switching  from  the
**  active execution environment to the definition environment of a function.
**  But in order to do this it needs to know the  definition  environment  of
**  a function.  'EvMakefunc' copies the function  definition  bag  and  adds
**  the handle of the current  environment  to  that  bag.  This  process  is
**  usually called closing the function and the result is called  a  closure.
**
**  To be precise, the make-function bag created by the parser has the form:
**
**      handle 0:               handle of the statement sequence.
**      handle 1..n:            handles of the arguments.
**      handle n+1..n+m:        handles of the local variables.
**      handle n+m+1:           0.
**      handle n+m+2:           handle of the definition location T_STRING 
**      data 1:                 (short) number of arguments (n).
**      data 2:                 (short) number of local variables (m).
**
**  And 'EvMakefunc' makes a copy of the form:
**
**      handle 0:               handle of the statement sequence.
**      handle 1..n:            handles of the arguments.
**      handle n+1..n+m:        handles of the local variables.
**      handle n+m+1:           handle of the definition environment.
**      handle n+m+2:           handle of the definition location T_STRING
**      data 1:                 (short) number of arguments (n).
**      data 2:                 (short) number of local variables (m).
*/

Bag       EvMakemake (Bag hdFun, Int type)
{
    Bag           Result;
    short               nrArg,  nrLoc, i;

    Result = NewBag( type, GET_SIZE_BAG(hdFun) );

    /* copy the info about the number of arguments and locals              */
    nrArg = NUM_ARGS_FUNC(hdFun);
    nrLoc = NUM_LOCALS_FUNC(hdFun);

    NUM_ARGS_FUNC(Result) = nrArg;
    NUM_LOCALS_FUNC(Result) = nrLoc;

    /* now copy the formal arguments and locals                            */
    ACT_NUM_ARGS_FUNC(hdFun, nrArg);
    ACT_NUM_LOCALS_FUNC(hdFun, nrLoc);
    for ( i = 0; i <= nrArg+nrLoc; ++i )
        SET_BAG(Result, i,  PTR_BAG(hdFun)[i] );

    /* add the environment, i.e., close the function                       */
    if (GET_FLAG_BAG(hdFun, BF_ENVIRONMENT))
	SET_BAG(Result, nrArg+nrLoc+1,  HdExec );
    else
        SET_BAG(Result, nrArg+nrLoc+1,  0 );
    SET_BAG(Result, nrArg+nrLoc+2,  PTR_BAG(hdFun)[nrArg+nrLoc+2] );

    /* return the new function                                             */
    return Result;
}

Bag       EvMakefunc ( Bag hdFun ) {
    Bag res = EvMakemake(hdFun, T_FUNCTION);
    /* doc stuff
    if(res!=0) {
        if(*( (char*)PTR_BAG(GetCommentBuffer()) ) != '\0') {
            char * str = GuMakeMessage("[[ defined in %s:%d ]]\n",
                                      Input->name, Input->number);
            char * name = GuMakeMessage("%ld", (long)res);
            AppendCommentBuffer(str, strlen(str));
            free(str);
            CommentBufferToDoc(name);
            free(name);
        }
        ClearCommentBuffer();
    }
     end doc stuff */
    return res;
}

Bag       EvMakemeth ( Bag hdFun ) {
    Bag res = EvMakemake(hdFun, T_METHOD);
    return res;
}

/****************************************************************************
**
*F  EvReturn( <hdRet> ) . . . . . . . . . . . . . evaluate a return-statement
**
**  'EvReturn' executes the return-statement with the handle <hdRet>.
**
**  'EvReturn' evaluates the expression in the return bag and puts the  value
**  in the 'HdReturn' bag.  This bag is then  passed  back  through  all  the
**  statement execution functions, until  it  finally  reaches  'EvFunccall'.
**  'EvFunccall' then returns the value in the 'HdResult' bag.
**
**  Note that a quit statement is implemented as a return bag with the  value
**  'HdReturn' in it.  When 'EvReturn' sees this it does not try to  evaluate
**  it but just puts it into the 'HdReturn' bag.  The rules for  'EvFunccall'
**  now say that the function call will return 'HdReturn', thus it will make
**  its way back to the mail loop.
*/
Bag       EvReturn (Bag hdRet)
{
    Bag           hd;

    if ( PTR_BAG(hdRet)[0] == HdReturn )
        hd = HdReturn;
    else if ( PTR_BAG(hdRet)[0] == HdVoid )
        hd = HdVoid;
    else
        hd = EVAL( PTR_BAG(hdRet)[0] );
    SET_BAG(HdReturn, 0,  hd );
    return HdReturn;
}


/****************************************************************************
**
*F  FunIsFunc( <hdCall> ) . . . . . . . . . . . .  internal function 'IsFunc'
**
**  'IsFunc'  returns 'true' if the object <obj> is  a  function and  'false'
**  otherwise.  May cause an error if <obj> is an unbound variable.
*/
Bag       FunIsFunc (Bag hdCall)
{
    Bag           hdObj;

    /* evaluate and check the argument                                     */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: IsFunc( <obj> )",0,0);
    hdObj = EVAL( PTR_BAG(hdCall)[1] );
    if ( hdObj == HdVoid )
        return Error("IsFunc: function must return a value",0,0);

    if ( GET_TYPE_BAG(hdObj) == T_FUNCTION || GET_TYPE_BAG(hdObj) == T_FUNCINT )
        return HdTrue;
    else
        return HdFalse;
}

/****************************************************************************
**
*F  FunIsMeth( <hdCall> ) . . . . . . . . . . . .  internal function 'IsMeth'
**
**  'IsMeth' returns 'true' if <obj> is a method and 'false' otherwise.
*/
Bag       FunIsMeth (Bag hdCall)
{
    Bag           hdObj;

    /* evaluate and check the argument                                     */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: IsFunc( <obj> )",0,0);
    hdObj = EVAL( PTR_BAG(hdCall)[1] );
    if ( hdObj == HdVoid )
        return Error("IsMeth: function must return a value",0,0);

    /* return 'true' if <obj> is a rational and 'false' otherwise          */
    if ( IsMethodFunc(hdObj) )
        return HdTrue;
    else
        return HdFalse;
}

/****************************************************************************
**
*F  FunUnevalArgs( <hdCall> ) . . . . . . . .  internal function 'UnevalArgs'
**
**  UnevalArgs(<func>) marks function with BF_UNEVAL_ARGS, so that its argum-
**  ents are not evaluated.
*/
Bag       FunUnevalArgs (Bag hdCall)
{
    Bag           hdFunc;
    /* evaluate and check the argument                                     */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: UnevalArgs( <func> )",0,0);
    hdFunc = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hdFunc) != T_FUNCTION )
        return Error("UnevalArgs: <func> must be a function",0,0);

    SET_FLAG_BAG(hdFunc, BF_UNEVAL_ARGS);
    return hdFunc;
}


/****************************************************************************
**
*F  FunTraceFunc( <hdCall> )  . . . . . . . . . . . . . internal function 'TraceFunc'
**
**  'FunTraceFunc' implements the internal function 'TraceFunc'.
**
**  'TraceFunc( <function>... )'
**
**  'TraceFunc' switches on  tracing  for  the  functions  passed  as  arguments.
**  Whenever such a function is called GAP prints a message of the form:
**
**      <function1>( <arg1>, <arg2>, ... )
**       <function2>()
**        ...
**       <function2> returns
**      <function1> returns <value>
**
**  Where <function1>, <function2>, <arg1>, <arqg2> and <value>  are  replaced
**  by the respective values.
**
**  'UntraceFunc' switches this off again.
*/
Bag       FunTraceFunc (Bag hdCall)
{
    Bag           hdDef;
    short               nrLoc,  i;
    UInt t;

    for ( i = 1; i < GET_SIZE_BAG(hdCall)/SIZE_HD; ++i ) {
        hdDef = EVAL( PTR_BAG(hdCall)[i] );
        t = GET_TYPE_BAG(hdDef);
        if ( t == T_FUNCINT )
            return Error("sorry I can not trace internal function",0,0);
        if ( t != T_FUNCTION && t != T_METHOD )
            return Error("usage: TraceFunc( <function>... )",0,0);
        /* use negative nrLoc -- number of locals -- as flag to enable tracing in EvFunccall */
        nrLoc = ((short*)((char*)PTR_BAG(hdDef)+GET_SIZE_BAG(hdDef)))[-1];
        if ( 0 <= nrLoc )  nrLoc = -nrLoc-1;
        ((short*)((char*)PTR_BAG(hdDef)+GET_SIZE_BAG(hdDef)))[-1] = nrLoc;
    }
    return HdVoid;
}


/****************************************************************************
**
*F  FunUntraceFunc( <hdCall> )  . . . . . . . . . . . internal function 'UntraceFunc'
**
**  'FunUntrace' implements the internal function 'UntraceFunc'.
**
**  'UntraceFunc( <function>... )'
**
**  'UntraceFunc' switches off tracing for the functions passed as  arguments.
*/
Bag       FunUntraceFunc (Bag hdCall)
{
    Bag           hdDef;
    short               nrLoc, i;
    UInt t;

    for ( i = 1; i < GET_SIZE_BAG(hdCall)/SIZE_HD; ++i ) {
        hdDef = EVAL( PTR_BAG(hdCall)[i] );
        t = GET_TYPE_BAG(hdDef);
        if ( t != T_FUNCTION && t != T_METHOD )
            return Error("usage: UntraceFunc( <function>... )",0,0);
        nrLoc = ((short*)((char*)PTR_BAG(hdDef)+GET_SIZE_BAG(hdDef)))[-1];
        if ( nrLoc < 0 )  nrLoc = -nrLoc-1;
        ((short*)((char*)PTR_BAG(hdDef)+GET_SIZE_BAG(hdDef)))[-1] = nrLoc;
    }
    return HdVoid;
}


/****************************************************************************
**
*F  FunProfile( <hdCall> )  . . . . . . . . . . . internal function 'Profile'
**
**  'FunProfile' implements the internal function 'Profile'.
**
**  'Profile( true )'
**  'Profile( false )'
**  'Profile()'
**
**  'Profile' controls the function profiling.
**
**  In the first form,  with  the  argument  'true', 'Profile'  switches  the
**  profiling on.  From that moment on for every function GAP  remembers  the
**  number of times  this  function  was  called,  the  time  spent  in  this
**  function without its children, i.e., the functions it  called  and  their
**  children, and the time spent in this function together with them.  If the
**  profiling was already on, 'Profile' clears the profiling information.
**
**  In the second form, with the  argument  'false', 'Profile'  switches  the
**  profiling off again.  Note that programs run faster without profiling.
**
**  In the third form, without  arguments,  'Profile'  prints  the  profiling
**  information.
*/
Bag       FunProfile (Bag hdCall)
{
    Bag           hdArg;
    short               i;
    Int                total;

    /* check argument count                                                */
    if ( 2 * SIZE_HD < GET_SIZE_BAG(hdCall) ) {
        return Error("usage: Profile( true|false ) or Profile()",0,0);
    }

    /* switch profiling on or off                                          */
    else if ( GET_SIZE_BAG(hdCall) == 2 * SIZE_HD ) {
        hdArg = EVAL( PTR_BAG(hdCall)[1] );
        if ( hdArg == HdTrue ) {
            IsProfiling = 2;
            Resize( HdTimes, 0 * SIZE_HD );
        }
        else if ( hdArg == HdFalse ) {
            IsProfiling = 0;
        }
        else {
            return Error("usage: Profile( true|false ) or Profile()",0,0);
        }
    }

    /* print profiling information, this should be formatted much nicer    */
    else {
        total = 0;
        for ( i = 0; i < GET_SIZE_BAG(HdTimes)/SIZE_HD; i += 5 )
            total = total + HD_TO_INT( PTR_BAG(HdTimes)[i+3] );
        if ( total == 0 )  total = 1;
        Pr(" count    time percent time/call child function\n",0,0);
        for ( i = 0; i < GET_SIZE_BAG(HdTimes)/SIZE_HD; i += 5 ) {
            Pr("%6d  ", HD_TO_INT( PTR_BAG(HdTimes)[i+2] ), 0 );
            Pr("%6d  ", HD_TO_INT( PTR_BAG(HdTimes)[i+3] ), 0 );
            Pr("%6d  ", 100 * HD_TO_INT(PTR_BAG(HdTimes)[i+3]) / total, 0 );
            Pr("%6d  ", HD_TO_INT( PTR_BAG(HdTimes)[i+3] ) /
                        HD_TO_INT( PTR_BAG(HdTimes)[i+2] ), 0 );
            Pr("%6d  ", HD_TO_INT( PTR_BAG(HdTimes)[i+4] ), 0 );
            Print( PTR_BAG(HdTimes)[i+1] );
            Pr("\n",0,0);
        }
        Pr("        %6d     100                  TOTAL\n",total-1,0);
    }

    return HdVoid;
}


/****************************************************************************
**
*F  FunApplyFunc( <hdCall> )  . . . . . . . . . . . . .  internal 'ApplyFunc'
*/
Bag FunApplyFunc (Bag hdCall)
{
    Bag           hdNew = 0;      /* the new function call bag       */
    Bag           hdFunc;         /* the function                    */
    Bag           hdList = 0;     /* and the list                    */
    Bag		  hdRes = 0;
    Int                i = 0;          /* loop                            */

    /* check arguments                                                     */
    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error( "usage: ApplyFunc( <func>, <list> )", 0, 0 );
    /* Don't evaluate function definition for nicer Backtrace() output     */
    hdFunc = PTR_BAG(hdCall)[1];
    hdList = EVAL(PTR_BAG(hdCall)[2]);
    if ( ! IS_DENSE_LIST(hdList) )
        return Error( "<list> must be a dense list", 0, 0 );

    /* create a new function call bag                                      */
    hdNew = NewBag( T_FUNCCALL, SIZE_HD*(1+LEN_LIST(hdList)) );
    SET_BAG(hdNew, 0,  hdFunc );
    /* copy arguments into it                                              */
    for ( i = LEN_LIST(hdList);  0 < i;  i-- ) {
        Obj hd = ELMF_LIST( hdList, i );
        SET_BAG(hdNew, i,  hd );
    }

    /* evaluate this call                                                  */
    hdRes = EVAL(hdNew);
    return hdRes;
}


/****************************************************************************
**
*F  PrFuncint( <hdFun> )  . . . . . . . . . . . .  print an internal function
**
**  'PrFuncint' prints the internal function with the handle  <hdFun> in  the
**  short form: 'function (...) internal; end'.
*/
/*ARGSUSED*/

void            PrFuncint (Bag hdFun)
{
    Pr("%2>%s%2<",
       (Int) ((char*)PTR_BAG(hdFun) + sizeof(PtrIntFunc)), 0);
}


/****************************************************************************
**
*F  PrFunction( <hdFun> ) . . . . . . . . . . . . . . . . .  print a function
*F  PrMethod( <hdFun> ) . . . . . . . . . . . . . . . . . . .  print a method
**
**  'PrFunction' prints the function with the handle <hdFun>  either  in  the
**  short format:
**
**      function ( <args> ) ... end
**
**  if 'prFull' is 0, or in the long format:
**
**      function ( <args> )
**          local <locals>;
**          <statements>
**      end
**
**  otherwise.
*/
Int            prFull;

int MaybePrintShortFunc (Bag hdFun, char *shortKeyword)
{
    /* we can print in short form if the first statment is return ... */
    if(NUM_LOCALS_FUNC(hdFun)==0 && GET_TYPE_BAG(PTR_BAG(hdFun)[0]) == T_RETURN) {
        int i;
        int nrArg;
        Pr("(%>",0,0);
        ACT_NUM_ARGS_FUNC(hdFun, nrArg);
        for ( i = 1; i <= nrArg; ++i ) {
            Print( PTR_BAG(hdFun)[i] );
            if ( i != nrArg )  Pr("%<, %>",0,0);
        }
        Pr("%<) %s %>%g%<", (Int)shortKeyword, (Int)PTR_BAG(PTR_BAG(hdFun)[0])[0]);
        return 1;
    }
    else return 0;
}

void            PrFunc (Bag hdFun, char *keyword, char *shortKeyword)
{
    short               nrArg,  nrLoc,  i;

    if(MaybePrintShortFunc(hdFun, shortKeyword))
        return;

    Pr("%5>%s%< ( %>",(Int)keyword,0);
    ACT_NUM_ARGS_FUNC(hdFun, nrArg);
    for ( i = 1; i <= nrArg; ++i ) {
        Print( PTR_BAG(hdFun)[i] );
        if ( i != nrArg )  Pr("%<, %>",0,0);
    }
    Pr(" %<)",0,0);

    if ( prFull == 0 ) {
        Pr(" ...%4< ",0,0);
    }
    else {
        Pr("\n",0,0);
        nrLoc = ((short*)((char*)PTR_BAG(hdFun) + GET_SIZE_BAG(hdFun)))[-1];
        if ( nrLoc < 0 )  nrLoc = -nrLoc-1;
        if ( nrLoc >= 1 ) {
            Pr("%>local  ",0,0);
            for ( i = 1; i <= nrLoc; ++i ) {
                Print( PTR_BAG(hdFun)[i+nrArg] );
                if ( i != nrLoc )  Pr("%<, %>",0,0);
            }
            Pr("%<;\n",0,0);
        }
        Print( PTR_BAG(hdFun)[0] );
        Pr(";%4<\n",0,0);
    }

    Pr("end",0,0);
}

void            PrFunction (Bag hdFun) {
    PrFunc(hdFun, "function", "->");
}

void            PrMethod (Bag hdFun) {
    PrFunc(hdFun, "meth", ">>");
}

/****************************************************************************
**
*F  PrintFunction( <hdFun> )  . . . . . . . print a function in the full form
**
**  'PrintFunction' prints the function with the handle <hdFun> in  the  full
**  form, i.e., with the statement sequence.  It is called from main read-eval
**  loop.
*/
void            PrintFunction (Bag hdFun)
{
    prFull = NUM_TO_INT(1);
    PrFunction( hdFun );
    prFull = 0;
}

/****************************************************************************
**
*F  PrintMethod( <hdFun> )  . . . . . . . . . print a method in the full form
**
**  Same as 'PrintFunction' but for a method.
*/
void            PrintMethod (Bag hdFun)
{
	prFull = NUM_TO_INT(1);
	PrMethod(hdFun);
	prFull = 0;
}

/****************************************************************************
**
*F  PrFunccall( <hdCall> )  . . . . . . . . . . . . . . print a function call
**
**  'PrFunccall' prints the function call with the  handle  <hdCall>  in  the
**  usual form:  '<function>( <args> )'.
**
**  Linebreaks are preffered after the opening  parenthesis  and  the  commas
**  between the arguments.
*/
void            PrFunccall (Bag hdCall)
{
    Int                i;
    Int                start = 1;
    if ( GET_FLAG_BAG(hdCall, BF_METHCALL) ) start = 2;
    Pr("%2>",0,0);  Print( PTR_BAG(hdCall)[0] ); Pr("%<(%>",0,0);
    for ( i = start; i < GET_SIZE_BAG(hdCall)/SIZE_HD; ++i ) {
        Print( PTR_BAG(hdCall)[i] );
        if ( i != GET_SIZE_BAG(hdCall)/SIZE_HD-1 )
            Pr("%<, %>",0,0);
    }
    Pr("%2<)",0,0);
}


/****************************************************************************
**
**  PrReturn( <hdRet> ) . . . . . . . . . . . . . .  print a return statement
**
**  'PrReturn' prints the return statement with the  handle  <hdRet>  in  the
**  usual form 'return;' or 'return <expr>;'.
*/
void            PrReturn (Bag hdRet)
{
    if ( PTR_BAG(hdRet)[0] == HdReturn ) {
        Pr("quit",0,0);
    }
    else if ( PTR_BAG(hdRet)[0] == HdVoid ) {
        Pr("return",0,0);
    }
    else {
        Pr("%2>return%< %>",0,0);
        Print( PTR_BAG(hdRet)[0] );
        Pr("%2<",0,0);
    }
}


/****************************************************************************
**
*F  InitFunc()  . . . . . . . . . . .  initialize function evaluation package
**
**  'InitFunc' initializes the function evaluation package.
*/

void            InitFunc (void)
{
    InstEvFunc( T_FUNCCALL, EvFunccall  );
    InstEvFunc( T_FUNCTION, EvFunction  );
    InstEvFunc( T_METHOD,   EvFunction  );
    InstEvFunc( T_FUNCINT,  EvFunction  );
    InstEvFunc( T_MAKEFUNC, EvMakefunc  );
    InstEvFunc( T_MAKEMETH, EvMakemeth  );
    InstEvFunc( T_RETURN,   EvReturn    );

    InstPrFunc( T_FUNCCALL, PrFunccall  );
    InstPrFunc( T_FUNCINT,  PrFuncint   );

    InstPrFunc( T_FUNCTION, PrFunction  );
    InstPrFunc( T_MAKEFUNC, PrFunction  );

    InstPrFunc( T_METHOD,   PrMethod    );
    InstPrFunc( T_MAKEMETH, PrMethod    );
    InstPrFunc( T_RETURN,   PrReturn    );

    InstIntFunc( "IsFunc",      FunIsFunc      );
    InstIntFunc( "IsMeth",      FunIsMeth      );
    InstIntFunc( "TraceFunc",   FunTraceFunc   );
    InstIntFunc( "UntraceFunc", FunUntraceFunc );
    InstIntFunc( "ApplyFunc",   FunApplyFunc   );
    InstIntFunc( "UnevalArgs",  FunUnevalArgs  );

    INIT_FAST_CALL();

    HdTimes = NewBag( T_LIST, 0 );
    InstIntFunc( "Profile", FunProfile );
    HdReturn = NewBag( T_RETURN, SIZE_HD );

    HdCallRecname = FindRecname( "__call__" );
    InitGlobalBag(&HdCallRecname, "HdCallRecname");

    RecursionDepth = 0;
}
