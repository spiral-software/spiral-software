#include "GapUtils.h"
#include "memmgr.h"
#include "eval.h"
#include "list.h"
#include "plist.h"
#include "idents.h"
#include "record.h"
#include "integer.h"
#include "args.h"
#include "namespaces.h"
#include "function.h"
#include "read.h"
#include "scanner.h"
#include "hooks.h"
#include "comments.h"
#include        "objects.h"
#include		"string4.h"
#include "debug.h"

#define DEEP_STACK_STEP 1000000

// defined in function.c
extern Int  prFull;

// defined in gap.c
extern Int  DbgStackTop;
extern UInt DbgEvalStackTop;
extern Obj  DbgStackExec();
extern Int  DbgDown();
extern Int  DbgUp();
extern Int  inBreakLoop();
extern void PrintBacktraceExec(Bag hdExec, UInt execDepth, UInt execStackDepth, UInt printValues );
extern UInt DbgExecStackDepth();

static char brkField_statement[] = "statement";
static Bag  HdFieldStat = 0;
static char brkField_object_read[] = "object_read";
static Bag  HdFieldObjRd = 0;
static char brkField_object_write[] = "object_write";
static Bag  HdFieldObjWr = 0;
static char brkField_hitCount[] = "uncondHits";
static Bag  HdFieldHitCount = 0;
static char brkField_condition[] = "condition";
static Bag  HdFieldCond = 0;


Bag     HdEvalBreakpoints = 0;
UInt    EvalStackBreakLevel = 0;
// InDebugMode - EvTab hooked if non zero.
UInt    InDebugMode = 0;
// InBreakpoint - in breakpoint handler if non zero.
UInt    InBreakpoint = 0;

static Obj (* OrigEvTab[ T_ILLEGAL ]) ( Obj hd );

/* PrTab hooked by FunTop function in order to be able to highlight
current statement while printing top function */
static void (* OrigPrTab[ T_ILLEGAL ]) ( Obj hd );

static inline UInt isRetTrueFunc(Obj hdFunc) {
    Obj hdStat = PTR_BAG(hdFunc)[0];
    return (GET_TYPE_BAG(hdStat)==T_RETURN) && (PTR_BAG(hdStat)[0] == HdTrue);
}

static UInt evalCondition(Obj* recHdCond, Obj hdExec) {
    if (recHdCond==0 || recHdCond[1]==0 || recHdCond[1] == HdTrue)
        return 1;
    if (recHdCond[1] == HdFalse)
        return 0;
    if (GET_TYPE_BAG(recHdCond[1]) == T_FUNCTION) {
        Int numArg;
        if (isRetTrueFunc(recHdCond[1])) 
            return 1;
        ACT_NUM_ARGS_FUNC(recHdCond[1], numArg);
        if (numArg==0 && hdExec==0) {
            return EVAL(UniBag(T_FUNCCALL, recHdCond[1] ))==HdTrue;
        } else {
            if (hdExec!=0) {
                Int i;
                ACT_NUM_ARGS_FUNC(PTR_BAG(hdExec)[2], i);
                if (numArg==0) {
                    return EVAL(UniBag(T_FUNCCALL, recHdCond[1] ))==HdTrue;
                } else if (i==numArg) {
                    Obj hdCall = NewBag(T_FUNCCALL, (1+numArg)*SIZE_HD);
                    SET_BAG(hdCall, 0, recHdCond[1]);
                    for ( i = 1;  i<=numArg;  i++)
                        SET_ELM_PLIST(hdCall, i, PTR_BAG(hdExec)[EXEC_ARGS_START+i-1] );
                    return EVAL(hdCall)==HdTrue;
                } else {
                    Pr("Breakpoint condition function has %d arguments when statement function has %d arguments.\n", numArg, i);
                }
            }
        }
        return 0;
    }
    Pr("Breakpoint condition must have boolean value or must be a function.\n", 0, 0);
    return 1;
}

static UInt evalMemCondition(Obj* recHdCond, Obj hdObj, Obj hdField, Obj hdValue) {
    if (recHdCond==0 || recHdCond[1]==0 || recHdCond[1] == HdTrue)
        return 1;
    if (recHdCond[1] == HdFalse)
        return 0;
    if (GET_TYPE_BAG(recHdCond[1]) == T_FUNCTION) {
        Int numArg;
        if (isRetTrueFunc(recHdCond[1])) 
            return 1;
        ACT_NUM_ARGS_FUNC(recHdCond[1], numArg);
        if (numArg==0) {
            return EVAL(UniBag(T_FUNCCALL, recHdCond[1] ))==HdTrue;
        } else {
            Obj hdCall = NewBag(T_FUNCCALL, (1+numArg)*SIZE_HD);
            SET_BAG(hdCall, 0, recHdCond[1]);
            SET_ELM_PLIST(hdCall, 1, hdObj );
            if (numArg==2) {
                SET_ELM_PLIST(hdCall, 2, hdValue );
            } else if (numArg==3) {
                SET_ELM_PLIST(hdCall, 2, hdField );
                SET_ELM_PLIST(hdCall, 3, hdValue );
            } else {
                Pr("Breakpoint condition function has unexpected number of arguments.\n", 0, 0);
                return 0;
            }
            return EVAL(hdCall)==HdTrue;
        }
    }
    Pr("Breakpoint condition must have boolean value or must be a function.\n", 0, 0);
    return 1;
}

static inline void incBreakpointHits(Obj hdBrk) {
    Obj     hdJunk;
    Obj*    pRec = FindRecnameRec(hdBrk, HdFieldHitCount, &hdJunk);
    if (pRec) 
        pRec[1] = INT_TO_HD(HD_TO_INT(pRec[1])+1);

}

static UInt evalBreakPoint(Obj hdBrk, Obj hdCurStat, Obj hdExec) {
    Obj     hdJunk;
    Bag     hdStat;
    Obj*    pRec = FindRecnameRec(hdBrk, HdFieldStat, &hdJunk);
    if (pRec==0) return 0; /* no statement field */
    hdStat = pRec[1];
    if (hdStat == 0 || hdStat == HdVoid || hdStat == hdCurStat) {
        incBreakpointHits(hdBrk);
        return evalCondition(FindRecnameRec(hdBrk, HdFieldCond, &hdJunk), hdExec);
    }
    return 0;
}

static UInt evalMemBreakpoint(Obj hdBrk, Obj hdMode, Obj hdObj, Obj hdField, Obj hdValue) {
    Obj     hdJunk;
    Bag     hdBrkObj;
    Obj*    pRec = FindRecnameRec(hdBrk, hdMode, &hdJunk);
    if (pRec==0) return 0; /* no hdMode field */
    hdBrkObj = pRec[1];
    if (hdBrkObj == 0 || hdBrkObj == HdVoid 
        || hdBrkObj == hdObj 
        || (GET_TYPE_BAG(hdBrkObj) == T_RECELM 
            && (PTR_BAG(hdBrkObj)[0]==0 || PTR_BAG(hdBrkObj)[0]==hdObj) 
            && PTR_BAG(hdBrkObj)[1] == hdField)  
        ) {
        incBreakpointHits(hdBrk);
        return evalMemCondition(FindRecnameRec(hdBrk, HdFieldCond, &hdJunk), hdObj, hdField, hdValue);
    }
    return 0;
}

void    CheckBreakpoints(Obj hdE, Obj hdExec) {
    exc_type_t  e = 0;
    if (!inBreakLoop() && !InBreakpoint) {
        InBreakpoint = 1;
        Try {
            Bag hdBrk = 0;
            Bag	hdLst = VAR_VALUE(HdEvalBreakpoints);
            if (GET_TYPE_BAG(hdLst)==T_LIST) {
		UInt cnt = LEN_PLIST(hdLst);
		UInt i = 0;
		for (i=1; i<=cnt; i++) {
		    hdBrk = ELM_PLIST(hdLst, i);
		    if (evalBreakPoint( hdBrk, hdE, hdExec))
			break;
		    hdBrk = 0;
		}
		if (hdBrk != 0 || (EvalStackBreakLevel>0 
		    && EvalStackBreakLevel>EvalStackTop
		    && hdExec == 0 ) ) 
		{
		    if (HdExec==0) {
			// cannot break in the main loop
		    } else {
			EvalStackBreakLevel = 0;
			DbgBreak("GapBreakpoint", 0, 0);
		    }
		}
	    } else {
		Pr("Breakpoints is not a list", 0, 0);
	    }
	    InBreakpoint = 0;
        } Catch(e) {
            InBreakpoint = 0;
            Throw(e);
        }
    }
}

void    CheckMemBreakpoints( Obj hdMode, Obj hdObject, Obj hdField, Obj hdValue) {
    exc_type_t  e = 0;
    if (!inBreakLoop() && !InBreakpoint) {
        InBreakpoint = 1;
        Try {
            Bag hdBrk = 0;
            Bag	hdLst = VAR_VALUE(HdEvalBreakpoints);
            if (GET_TYPE_BAG(hdLst)==T_LIST) {
		UInt cnt = LEN_PLIST(hdLst);
		UInt i = 0;
		for (i=1; i<=cnt; i++) {
		    hdBrk = ELM_PLIST(hdLst, i);
		    if (evalMemBreakpoint( hdBrk, hdMode, hdObject, hdField, hdValue))
			break;
		    hdBrk = 0;
		}
		if (hdBrk != 0) {
		    if (HdExec==0) {
			// cannot break in the main loop
		    } else {
			EvalStackBreakLevel = 0;
			DbgBreak((hdMode==HdFieldObjRd) ? "GapBreakpointRd" : "GapBreakpointWr", 0, 0);
		    }
		}
	    }
	    InBreakpoint = 0;
        } Catch(e) {
            InBreakpoint = 0;
            Throw(e);
        }
    }
}

/****************************************************************************
**
*F  void DbgBreak(char* message)  . . . . . . . . . . . break to dbg prompt 
** 
**  DbgBreak stops execution and enters dbg> loop. This function must be 
**  called instead of Error() if you need just to stop execution ( for 
**  example as response to Ctrl+C).
*/

void    DbgBreak(char* message, Int arg1, Int arg2) {
    if (!inBreakLoop() && !InBreakpoint) {
        exc_type_t  e = 0;
        InBreakpoint = 1;
        Try {
            Error(message, arg1, arg2);
            InBreakpoint = 0;
        } Catch(e) {
            InBreakpoint = 0;
            Throw(e);
        }
    } else {
        Error(message, arg1, arg2);
    }
}

static int _printTopFunction = 0;
static Bag HdFuncTop = 0;

/* DbgErrorLoopStarting - called each time dbg> or brk> loop started */
void DbgErrorLoopStarting() {
    if (_printTopFunction) {
        _printTopFunction  = 0;
        if (InBreakpoint) 
            EVAL(UniBag(T_FUNCCALL, HdFuncTop));
    }
}

static Obj DebugEVAL ( Obj hdE ) {
    CheckBreakpoints(hdE, 0);
    return (* OrigEvTab[GET_TYPE_BAG( hdE )])(hdE);
}

static Obj DebugVarEVAL ( Obj hdVar ) {
    Obj hdValue;
    CheckBreakpoints(hdVar, 0);
    hdValue = (* OrigEvTab[T_VAR])(hdVar);
    CheckMemBreakpoints( HdFieldObjRd, hdVar, 0, hdValue);
    return hdValue;
}

static Obj DebugVarAssEVAL ( Obj hdVarAss ) {
    CheckBreakpoints(hdVarAss, 0);
    CheckMemBreakpoints( HdFieldObjWr, PTR_BAG(hdVarAss)[0], 0, PTR_BAG(hdVarAss)[1]);
    return (* OrigEvTab[T_VARASS])(hdVarAss);
}

static Obj DebugRecElmEVAL ( Obj hdRecElm ) {
    Obj hdValue;
    CheckBreakpoints(hdRecElm, 0);
    hdValue = (* OrigEvTab[T_RECELM])(hdRecElm);
    CheckMemBreakpoints( HdFieldObjRd, EVAL(PTR_BAG(hdRecElm)[0]), RecnameObj( PTR_BAG(hdRecElm)[1] ), hdValue);
    return hdValue;
}

static Obj DebugRecElmAssEVAL ( Obj hdRecAss ) {
    Obj hdRec, hdNam, hdNewValue;
    
    CheckBreakpoints(hdRecAss, 0);

    hdRec = EVAL( PTR_BAG(PTR_BAG(hdRecAss)[0])[0] );
    hdNam = RecnameObj( PTR_BAG(PTR_BAG(hdRecAss)[0])[1] ); 
    hdNewValue = PTR_BAG(hdRecAss)[1];               

    CheckMemBreakpoints( HdFieldObjWr, hdRec, hdNam, hdNewValue);
    
    return (* OrigEvTab[T_RECASS])(hdRecAss);
}

static char EnteringDbgMode[] = "\
Entering debug mode (slower execution), to exit call Debug(false). \
To get help on debugging call DebugHelp().\n\
Shortcuts: Down() - Ctrl+Down; Up() - Ctrl+Up; Top() - Ctrl+\\; \
StepOver() - F10; StepInto() - F11; StepOut() - F8.\n";

void    EnterDebugMode() {
    int t;
    if (InDebugMode == 0) {
        Pr(EnteringDbgMode, 0, 0);
        for(t=0; t<T_ILLEGAL; ++t) {
            OrigEvTab[t] = EvTab[t];
            EvTab[t] = DebugEVAL;
        }
        EvTab[T_VAR] = DebugVarEVAL;
        EvTab[T_VARASS] = DebugVarAssEVAL;
        EvTab[T_RECELM] = DebugRecElmEVAL;
        EvTab[T_RECASS] = DebugRecElmAssEVAL;
        InDebugMode = 1;
    }
}

void LeaveDebugMode() {
    int t;
    if (InDebugMode != 0) {
        for(t=0; t<T_ILLEGAL; ++t) {
            EvTab[t] = OrigEvTab[t];
        }
        InDebugMode = 0;
    }
}

static Obj  NewBreakpoint( Obj hdMainField, Obj hdMainFieldValue, UInt maxStackDepth, Obj hdCondition) {
    Obj result;
    result = NewBag(T_REC, SIZE_HD*2*3);
    SET_BAG(result, 0, hdMainField);
    SET_BAG(result, 1, hdMainFieldValue);
    SET_BAG(result, 2, HdFieldCond);
    SET_BAG(result, 3, hdCondition);
    SET_BAG(result, 4, HdFieldHitCount);
    SET_BAG(result, 5, INT_TO_HD(0));
    return result;
}

/****************************************************************************
**
*F  Debug( <bool> )  . . . . . . . . . . . .  switches Debug mode on / off
**
** If parameter is 'true' enables the debug mode, if 'false' disables it.
** In debug mode, the evaluation table is replaced by the special debug mode
** evaluation functions, which allow stepping thru evaluation and setting
** breakpoints.
*/
Obj  FunDebug ( Obj hdCall ) {
    char * usage = "usage: Debug( <bool> )";
    Obj hdEn;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage, 0,0);
    hdEn = EVAL( PTR_BAG(hdCall)[1] );
    if( GET_TYPE_BAG(hdEn) != T_BOOL ) return Error(usage,0,0);

    if (hdEn == HdTrue)
        EnterDebugMode();
    else
        LeaveDebugMode();
    return HdVoid;
}

static char HelpText[] = "\
  Breakpoints - breakpoints list;\n\
  Debug(<bool>) - turns on/off debug mode. In debug mode Spiral checking \n\
breakpoints each time something is evaluated. It runs slower in this mode\n\
but you can leave it by executing Debug(false). Functions Breakpoint(), \n\
StepOver(), StepInto(), StepOut() and ReturnAndBreak() enable debug mode \n\
automatically.\n\
  Breakpoint(<func>|<method>) - new breakpoint. Creating a new breakpoint \n\
and placing it into Breakpoints list. Returns breakpoint record.\n\
  BreakpointOnRead(<var>|<record>|<record>.<field>|RecName(\"field\")) - new\n\
breakpoint. Creating and placing new read access breakpoint into Breakpoints \n\
list. Returns breakpoint record.\n\
  BreakpointOnWrite(<var>|<record>|<record>.<field>|RecName(\"field\")) - new\n\
breakpoint. Creating and placing new write access breakpoint into Breakpoints\n\
list. Returns breakpoint record.\n\
  RemoveBreakpoint([<index>]) - removing breakpoint from Breakpoints list.\n\
\"Index\" is position of breakpoint to remove. If called without arguments\n\
removes all breakpoints.\n\
  DbgBreak(<text>) - same as Error() internal function but hits into dbg\n\
loop. This function always returns HdVoid.\n\
  ReturnAndBreak(<obj>) - function for switching from brk to dbg loop.\n\
Returns given object as result of Error() and placing breakpoint to next\n\
statement.\n\
\n\
    While user in brk or dbg loop he can move stack pointer using Down()\n\
    and Up() functions and watch variables values.\n\
  Down([<levels>]) - moves stack pointer down (as printed on the screen).\n\
<levels> is how many levels down you want to move stack pointer. Moves one\n\
level down if called without paramenters.\n\
  Up([<levels>]) - same as Down() but moves stack pointer up.\n\
  Top() - print function/method on top of the stack.\n\
\n\
    While in dbg loop user can execute statements step by step:\n\
  StepOver() - step over current statement.\n\
  StepInto() - step into current statement.\n\
  StepOut() - step out from current statement.\n\
\n\
    Breakpoint record fields:\n\
  statement - function or method on which execution breakpoint fires.\n\
  condition - function evaluated if statement match. If function returns\n\
true execution stops. By default dummy function generated, you can replace\n\
it by yours with the same parameters list.\n\
  uncondHits - integer counter. This counter incremented each time statement\n\
is matched, even if condition function returns false.\n\
\n";
    
    
Obj FunDebugHelp ( Obj hdCall ) {
    Pr(HelpText, 0, 0);
    return HdVoid;
}

/****************************************************************************
**
*F  Breakpoint( <obj>) . . . . . . . . . . . . . . . . . . . . new breakpoint 
** 
**  Creating new breakpoint on given object and placing it into Breakpoints
**  list. Turning on debug mode automatically.
*/

Obj  FunBreakpoint ( Obj hdCall ) {
    char * usage = "usage: Breakpoint( <obj> )";
    Obj hdObj, hdLst, hdDef, hdFunc;
    Int i, nrArg = 0;
    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage, 0,0);

    hdDef = EVAL(PTR_BAG(hdCall)[1]);
    if (GET_TYPE_BAG(hdDef)==T_FUNCTION || GET_TYPE_BAG(hdDef)==T_METHOD) {
        ACT_NUM_ARGS_FUNC(hdDef, nrArg);
    }
    
    hdFunc = NewBag(T_FUNCTION, (4+nrArg)*SIZE_HD);
    NUM_ARGS_FUNC(hdFunc) = (nrArg>0) ? (NUM_ARGS_FUNC(hdDef)) : 0;
    NUM_LOCALS_FUNC(hdFunc) = 0;
    for ( i=1; i<=nrArg; i++) {
        SET_BAG(hdFunc, i, PTR_BAG(hdDef)[i]);
    }
    SET_BAG(hdFunc, 0, UniBag(T_RETURN, HdTrue));
            
    hdObj = NewBreakpoint(HdFieldStat, hdDef, 0, hdFunc);
    hdLst = VAR_VALUE(HdEvalBreakpoints);
    AssPlist(hdLst, LEN_PLIST(hdLst)+1, hdObj);

    EnterDebugMode();
    
    return hdObj;
}

/****************************************************************************
**
*F  RemoveBreakpoint() . . . . . . . . . . . . . remove breakpoint from list
** 
**  Removing breakpoint from the Breakpoints list by index. If called without
**  parameters or with -1 whole Breakpoints list cleared.
*/

Obj  FunRemoveBreakpoint ( Obj hdCall ) {
    char * usage = "usage: RemoveBreakpoint( [<index>] )";
    Obj hdObj;
    Int i, cnt;
    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD  && GET_SIZE_BAG(hdCall) != SIZE_HD) return Error(usage, 0,0);
    if ( GET_SIZE_BAG(hdCall) == 2 * SIZE_HD ) {
        hdObj = EVAL(PTR_BAG(hdCall)[1]);
        if( GET_TYPE_BAG(hdObj) != T_INT ) return Error(usage,0,0);
        i = HD_TO_INT(hdObj);
    } else
        i = -1;
    
    hdObj = VAR_VALUE(HdEvalBreakpoints);
    if (i>0) {
        cnt = LEN_PLIST(hdObj);
        while (i<cnt) {
            SET_ELM_PLIST(hdObj, i, ELM_PLIST(hdObj, i+1));
            i++;
        }
        if (cnt>0 && i==cnt)
            SET_LEN_PLIST(hdObj, cnt-1);
    } else {
        SET_LEN_PLIST(hdObj, 0);
        Resize(hdObj, SIZE_PLEN_PLIST(0));
    }
    if (LEN_PLIST(hdObj)==0)
        LeaveDebugMode();
    return hdObj;
}

static Obj FunStep(Obj hdCall, char * usage, Int step) {
    Bag hdExec;
    Int	level = 0;
    if ( GET_SIZE_BAG(hdCall) > 2*SIZE_HD || !InBreakpoint ) return Error(usage, 0,0);
    _printTopFunction = (GET_SIZE_BAG(hdCall) == 2*SIZE_HD) && (IS_INTOBJ(PTR_BAG(hdCall)[1])) && (HD_TO_INT(PTR_BAG(hdCall)[1])!=0);
    if (DbgStackTop>0) {
	DbgStackTop--;
	hdExec = DbgStackExec();
	DbgStackTop++;
	if (hdExec)
	    level = HD_TO_INT(PTR_BAG(hdExec)[EXEC_EVAL_STACK]);
    }
    if (level==0 && EvalStackTop>1) {
	level = EvalStackTop + step;
    }
    if (level>0) {
	EnterDebugMode();
        EvalStackBreakLevel = level;
        SET_BAG(HdReturn, 0, HdVoid);
        return HdReturn;
    } else {
        return HdVoid;
    }
}

/****************************************************************************
**
*F  StepOver() . . . . . . . . . . . . . . . . . . . . break when execution
**    returns to same EVAL stack level.
*/

Obj FunStepOver( Obj hdCall ) {
    return FunStep(hdCall, "usage (in dbg mode only): StepOver()", 0);
}

/****************************************************************************
**
*F  StepInto() . . . . . . . . . . . . . . . . . . . . break on next EVAL. 
*/

Obj FunStepInto( Obj hdCall ) {
    return FunStep(hdCall, "usage (in dbg mode only): StepInto()", DEEP_STACK_STEP);
}

/****************************************************************************
**
*F  StepOut() . . . . . . . . . . . . . . . . . . . . break when parent EVAL
**    stack frame reached.
*/

Obj FunStepOut( Obj hdCall ) {
    return FunStep(hdCall, "usage (in dbg mode only): StepOut()", -1);
}

/****************************************************************************
**
*F  ReturnAndBreak( [<obj>] ) . . . . . . .returns object and does StepInto()
** 
**  While in the break loop this function allows to return some value (and
**  exit break loop), switch into debug mode and stop execution on next EVAL.
*/

Obj FunReturnAndBreak( Obj hdCall ) {
    char * usage = "usage (in brk mode only): ReturnAndBreak( [<obj>] )";
    Obj hdRet = HdVoid;
    if ( (GET_SIZE_BAG(hdCall) != SIZE_HD && GET_SIZE_BAG(hdCall) != 2*SIZE_HD) 
        || !inBreakLoop() || InBreakpoint ) return Error(usage, 0,0);
    if (GET_SIZE_BAG(hdCall) == 2*SIZE_HD) {
        hdRet = EVAL(PTR_BAG(hdCall)[1]);
    }
    EnterDebugMode();
    if (EvalStackTop>1) {
        EvalStackBreakLevel = EvalStackTop + DEEP_STACK_STEP;
        SET_BAG(HdReturn, 0, hdRet);
        return HdReturn;
    } else {
        return hdRet;
    }
}

/****************************************************************************
**
*F  Up() . . . . . . . . . . . . . . . . . . . . . . .  go one stack frame up
*F  Down() . . . . . . . . . . . . . . . . . . . . .  go one stack frame down
*F  Top() . . . . . . . . . . . . . . . .  print function on top of the stack
*
*/

/* We are hooking PrTab from FunTop() in order to highlight statements from EVAL call stack */

Int     PrTabHooked = 0;
Int     TopPr_Printing;
UInt    TopPr_CurDepth;
UInt    TopPr_MaxDepth;

static void TopPr ( Obj hd ) {
    UInt hasFlag, highlight = 0; 
    if (hd==0 || IS_INTOBJ(hd)) {
        (*OrigPrTab[GET_TYPE_BAG(hd)])(hd);
        return;
    }
    hasFlag = GET_FLAG_BAG(hd, BF_ON_EVAL_STACK);
    // special case: PrFunction may swallow T_RETURN
    if(hasFlag==0 && (GET_TYPE_BAG(hd)==T_FUNCTION || GET_TYPE_BAG(hd)==T_METHOD)) {
        hasFlag = (NUM_LOCALS_FUNC(hd)==0) && (GET_TYPE_BAG(PTR_BAG(hd)[0]) == T_RETURN)
            && GET_FLAG_BAG(PTR_BAG(hd)[0], BF_ON_EVAL_STACK);
    }
    if ( hasFlag )
        TopPr_CurDepth++;
    if (TopPr_Printing)
        // if we printing function determine should we highlight this statement or not
        highlight = hasFlag && (TopPr_CurDepth == TopPr_MaxDepth); 
    else {
        // if we traversing print tree update TopPr_MaxDepth
        if (TopPr_MaxDepth<TopPr_CurDepth) 
            TopPr_MaxDepth = TopPr_CurDepth;
    }
    if (highlight) HooksBrkHighlightStart();
    // call original Pr function
    (*OrigPrTab[GET_TYPE_BAG(hd)])(hd);
    if (highlight) HooksBrkHighlightEnd();
    if ( hasFlag )
        TopPr_CurDepth--;
}

static void HookPrTab()
{
    int t;
    PrTabHooked++;
    if (PrTabHooked==1) {
        for(t=0; t<T_ILLEGAL; ++t) {
	    OrigPrTab[t] = PrTab[t];
	    PrTab[t] = TopPr;
        }
    }
}
static void UnhookPrTab()
{
    int t;
    if (PrTabHooked>0) {
        PrTabHooked--;
        if (PrTabHooked==0) {
            for(t=0; t<T_ILLEGAL; ++t) {
	        PrTab[t] = OrigPrTab[t];
            }
        }
    }
}
	
Obj  FunTop ( Obj hdCall ) {
    char *      usage = "usage (in brk or dbg modes only): Top()";
    Obj         top, doc;
    exc_type_t  e;
    UInt        i;
    
    if ( GET_SIZE_BAG(hdCall) != 1 * SIZE_HD || !inBreakLoop())  return Error(usage, 0,0);
    // mark statements on the eval stack that belong to current function
    top = DbgStackExec();
    i = HD_TO_INT(PTR_BAG(top)[EXEC_EVAL_STACK])+1;
    while (i<=DbgEvalStackTop && GET_TYPE_BAG(EvalStack[i])!=T_EXEC) {
        SET_FLAG_BAG(EvalStack[i++], BF_ON_EVAL_STACK);
    }
    // get current function definition
    top = PTR_BAG(top)[2];
    doc = FindDocString(top);
    if(doc != 0 && GET_TYPE_BAG(doc)==T_STRING)
        Pr("%s", (Int)CSTR_STRING(doc), 0);
    else
        Pr("--no documentation--\n", 0, 0);
    prFull = 1;
    HookPrTab();
    Try {
        // first time figure out what to highlight, this is not precise
        // but better than nothing. Redirect output to /dev/null and traverse
        // bags that will be printed.
#ifdef WIN32
        OpenOutput("NUL");
#else
        OpenOutput("/dev/null");
#endif
        TopPr_CurDepth = 0; // current tree depth (in marked bags)
        TopPr_MaxDepth = 0; // maximal tree depth (in marked bags)
        Try {
            TopPr_Printing = 0; // we are going to calculate  max depth first
            Pr("%g\n", (Int)top, 0);
            // closing /dev/null and return to previous output
            CloseOutput();
        } Catch(e) {
            CloseOutput();
            Throw(e);
        }
        TopPr_Printing = 1; // now we are going to print function
        TopPr_CurDepth = 0; // reset tree depth, all what marked by BF_ON_EVAL_STACK
                            // flag and has TopPr_MaxDepth will be highlighted (with 
                            // children).
        Pr("%g\n", (Int)top, 0);
    } Catch(e) {
        UnhookPrTab();
        Throw(e);
    }
    UnhookPrTab();
    // reset BF_ON_EVAL_STACK flag
    i = HD_TO_INT(PTR_BAG(DbgStackExec())[EXEC_EVAL_STACK])+1;
    while (i<=DbgEvalStackTop && GET_TYPE_BAG(EvalStack[i])!=T_EXEC) {
        CLEAR_FLAG_BAG(EvalStack[i++], BF_ON_EVAL_STACK);
    }
    prFull = 0;
    return HdVoid;
}

Obj  FunEditTopFunc ( Obj hdCall ) {
    char *      usage = "usage (in brk or dbg modes only): EditTopFunc()";
    char	fileName[512];
    Int         line;
	
    if ( GET_SIZE_BAG(hdCall) != 1 * SIZE_HD || !inBreakLoop())  return Error(usage, 0,0);
    
    switch(FindDocAndExtractLoc(PTR_BAG(DbgStackExec())[2], fileName, &line)) {
        case  0: { Pr("--no documentation--\n", 0, 0); break; }
        case -1: { Pr("--definition not found--\n", 0, 0); break; }
        case  1: { HooksEditFile(fileName, line); break; }
    }
    return HdVoid;
}

static Int GetLevels( Obj hdCall ) {
    UInt levels = 1;
    if (GET_SIZE_BAG(hdCall) == 2 * SIZE_HD) {
        Obj hdLevels = EVAL(PTR_BAG(hdCall)[1]);
        if (GET_TYPE_BAG(hdLevels) == T_INT) {
            levels = HD_TO_INT(hdLevels);
            if (levels<1)
                levels = 1;
        }
    }
    return levels;
}

Obj  FunDown ( Obj hdCall ) {
    char * usage = "usage  (in brk or dbg modes only): Down( [<levels>] )";
    char * errText = "--initial frame selected (can't go down)--\n";
    UInt levels = GetLevels(hdCall);
    if ( (GET_SIZE_BAG(hdCall) != 1 * SIZE_HD && GET_SIZE_BAG(hdCall) != 2 * SIZE_HD) 
        || !inBreakLoop()) return Error(usage, 0,0);
    while (levels-->0) {
        if ( !DbgDown() ) { 
            Pr(errText, 0, 0);
            break;
        }
    }
    levels = DbgExecStackDepth();
    PrintBacktraceExec(HdExec, levels-DbgStackTop, levels, 0);
    return HdVoid;
}

Obj  FunUp ( Obj hdCall ) {
    char * usage = "usage (in brk or dbg modes only): Up( [<levels>] )";
    char * errText = "--topmost frame selected (can't go up)--\n";
    UInt levels = GetLevels(hdCall);
    if ( (GET_SIZE_BAG(hdCall) != 1 * SIZE_HD && GET_SIZE_BAG(hdCall) != 2 * SIZE_HD) 
        || !inBreakLoop()) return Error(usage, 0,0);
    while (levels-->0) {
        if ( !DbgUp() ) { 
            Pr(errText, 0, 0);
            break;
        }
    }
    levels = DbgExecStackDepth();
    PrintBacktraceExec(HdExec, levels-DbgStackTop, levels, 0);
    return HdVoid;
}


Bag FunNoDbg (Obj hdCall)
{
	return HdVoid;
}


/****************************************************************************
**
*F  FunDbgBreak( <hdCall> )  . . . . . . . . . . internal function 'DbgBreak'
**
**  'FunDbgBreak' implements the internal function 'DbgBreak'.
**
**  'DbgBreak( <message> )'
**
**  'FunDbgBreak' simply calls the GAP  kernel  function  'DbgBreak', 
**  which stops execution and starting debug loop. 
*/

Bag       FunDbgBreak (Obj hdCall)
{
    char* msg = "\0";
    if ( GET_SIZE_BAG(hdCall) ==  2*SIZE_HD ) {
        hdCall = EVAL(PTR_BAG(hdCall)[1]);
        if (GET_TYPE_BAG(hdCall)==T_STRING)
            msg = HD_TO_STRING(hdCall);
    }
    DbgBreak(msg, 0, 0);
    return HdVoid;
}

static Bag BreakpointObj (Obj hd, Obj hdMode) {
    Obj hdObj = hd, hdLst, hdFunc;
    Int nrArg = 0;

    if (GET_TYPE_BAG(hd)==T_FUNCCALL) hd = EVAL(hdObj);
    
    switch (GET_TYPE_BAG(hd)) {
        case T_VAR: {
            nrArg = 2;
            break;
        }
        case T_REC:
        case T_NAMESPACE: {
            nrArg = 3;
            break;
        }
        case T_RECELM: {
            hd = BinBag(T_RECELM, EVAL(PTR_BAG(hd)[0]), EVAL(PTR_BAG(hd)[1]));
            nrArg = 3;
            break;
        }
        case T_RECNAM: {
            hd = BinBag(T_RECELM, 0, hd);
            nrArg = 3;
            break;
        }
        default:
            return Error("Can set data access breakpoints on variables, records and tables only.", 0, 0);
    }
    
    hdFunc = NewBag(T_FUNCTION, (4+nrArg)*SIZE_HD);
    NUM_ARGS_FUNC(hdFunc) = (short)nrArg;
    NUM_LOCALS_FUNC(hdFunc) = 0;
    SET_BAG(hdFunc, 1, MakeIdent("obj"));
    if (nrArg==2) {
        SET_BAG(hdFunc, 2, MakeIdent("value"));
    } else {
        SET_BAG(hdFunc, 2, MakeIdent("field"));
        SET_BAG(hdFunc, 3, MakeIdent("value"));
    }
    SET_BAG(hdFunc, 0, UniBag(T_RETURN, HdTrue));
            
    hdObj = NewBreakpoint(hdMode, hd, 0, hdFunc);
    hdLst = VAR_VALUE(HdEvalBreakpoints);
    AssPlist(hdLst, LEN_PLIST(hdLst)+1, hdObj);

    EnterDebugMode();
    
    return hdObj;
}

/****************************************************************************
**
*F  BreakpointOnRead( <obj> )  . .  . . . . . . . . . . . . . .set breakpoint
**
**  'BreakpointOnRead' creating new breakpoint on read access for variables,
**  records, tables, record elements and recordnames.
**
*/

Bag       FunBreakpointOnRead (Obj hdCall) {
    char * usage = "usage: BreakpointOnRead( <obj> )";
    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage, 0,0);

    return BreakpointObj(PTR_BAG(hdCall)[1], HdFieldObjRd);
}

/****************************************************************************
**
*F  BreakpointOnWrite( <obj> )  . .  . . . . . . . . . . . . . .set breakpoint
**
**  'BreakpointOnWrite' creating new breakpoint on write access for variables,
**  records, tables, record elements and recordnames.
**
*/

Bag       FunBreakpointOnWrite (Obj hdCall) {
    char * usage = "usage: BreakpointOnWrite( <obj> )";
    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage, 0,0);

    return BreakpointObj(PTR_BAG(hdCall)[1], HdFieldObjWr);
}

void    InitDebug() {
    InitGlobalBag(&HdFieldStat, "HdFieldStat");
    InitGlobalBag(&HdFieldCond, "HdFieldCond");
    InitGlobalBag(&HdFieldHitCount, "HdFieldHitCount");
    InitGlobalBag(&HdFieldObjRd, "HdFieldObjRd");
    InitGlobalBag(&HdFieldObjWr, "HdFieldObjWr");
    InitGlobalBag(&HdEvalBreakpoints, "HdEvalBreakpoints");
    
    HdFieldStat = FindRecname(brkField_statement);
    HdFieldCond = FindRecname(brkField_condition);
    HdFieldHitCount = FindRecname(brkField_hitCount);
    HdFieldObjRd = FindRecname(brkField_object_read);
    HdFieldObjWr = FindRecname(brkField_object_write);

    /**/ GlobalPackage2("gap", "debug"); /**/

    HdEvalBreakpoints = FindIdent("Breakpoints");
    SET_VAR_VALUE(HdEvalBreakpoints, NewList(0));

    InstIntFunc( "Debug",             FunDebug );
    InstIntFunc( "DebugHelp",         FunDebugHelp );
    InstIntFunc( "Breakpoint",        FunBreakpoint );
    InstIntFunc( "BreakpointOnRead",  FunBreakpointOnRead);
    InstIntFunc( "BreakpointOnWrite", FunBreakpointOnWrite);
    InstIntFunc( "RemoveBreakpoint",  FunRemoveBreakpoint);
    
    InstIntFunc( "DbgBreak",        FunDbgBreak);
    InstIntFunc( "ReturnAndBreak",  FunReturnAndBreak);
    
    InstIntFunc( "StepOver",        FunStepOver);
    InstIntFunc( "StepInto",        FunStepInto);
    InstIntFunc( "StepOut",         FunStepOut);
    
    InstIntFunc( "Down",             FunDown);
    InstIntFunc( "Up",               FunUp);
    HdFuncTop = InstIntFunc( "Top",  FunTop);

    InstIntFunc( "EditTopFunc",      FunEditTopFunc);

	// Dummy init for no debug
	// InstIntFunc( "DbgBreak",        FunNoDbg);

    /**/ EndPackage(); /**/
}


