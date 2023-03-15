/****************************************************************************
**
*A  hooks.c                    SPIRAL source                Yevgen Voronenko
**
*A  $Id: hooks.c 7468 2008-10-29 12:59:11Z vlad $
**
**  Implements user-definable hooks, that are called on special GAP events,
**  for example, when new file is opened for reading.
*/
#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* Bag, NewBag, T_STRING, .. */
#include	"idents.h"
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */
#include        "scanner.h"             /* Pr(), Input                     */
#include        "objects.h"
#include		"string4.h"
#include        "comments.h"
#include        "args.h"
#include        "list.h"
#include        "plist.h"
#include        "namespaces.h"
/* #include        "flags.h" */			// defns & decls moved to memmgr.h

static int HOOKS_INITIALIZED = 0;

void RunHooks(Obj lst, Obj args) {
    int i, j, nargs;
    nargs = LEN_LIST(args);
    for(i = 1; i <= LEN_LIST(lst); ++i) {
	Obj hdFun = ELM_LIST(lst, i);
	Obj hdCall = NewBag(T_FUNCCALL, (1+nargs) * SIZE_HD);
	SET_BAG(hdCall, 0,  hdFun );
	for(j = 1; j <= nargs; ++j) {
	    Obj elm = ELM_LIST(args, j);
	    SET_BAG(hdCall, j,  elm );
	}

	EVAL(hdCall);
    }
}

Obj _mkArgs(TypInputFile *i1, TypInputFile *i2) {
    Obj args = NewList(2);
    Obj a1 = StringToHd(i1->name);
    Obj a2 = StringToHd(i2->name);
    ASS_LIST(args, 1, a1);
    ASS_LIST(args, 2, a2);
    return args;
}

void HookBeforeOpenInput(void) {
    if(HOOKS_INITIALIZED) {
	Obj var = FindIdent("HooksBeforeOpenInput");
	Obj val = PTR_BAG(var)[0];
	RunHooks(val, _mkArgs(Input, Input+1));
    }
}

void HookAfterOpenInput(void) {
    if(HOOKS_INITIALIZED) {
	Obj var = FindIdent("HooksAfterOpenInput");
	Obj val = PTR_BAG(var)[0];
	RunHooks(val, _mkArgs(Input-1, Input));
    }
}

void HookBeforeCloseInput(void) {
    if(HOOKS_INITIALIZED) {
	Obj var = FindIdent("HooksBeforeCloseInput");
	Obj val = PTR_BAG(var)[0];
	RunHooks(val, _mkArgs(Input, Input-1));
    }
}

void HookAfterCloseInput(void) {
    if(HOOKS_INITIALIZED) {
	Obj var = FindIdent("HooksAfterCloseInput");
	Obj val = PTR_BAG(var)[0];
	RunHooks(val, _mkArgs(Input+1, Input));
    }
}

void HookSessionStart(void) {
    if(HOOKS_INITIALIZED) {
	Obj var = FindIdent("HooksSessionStart");
	Obj val = PTR_BAG(var)[0];
	RunHooks(val, NewList(0));
    }
}

void HookSessionEnd(void) {
    if(HOOKS_INITIALIZED) {
	Obj var = FindIdent("HooksSessionEnd");
	Obj val = PTR_BAG(var)[0];
	RunHooks(val, NewList(0));
    }
}

void HooksBrkHighlightStart(void) {
    if(HOOKS_INITIALIZED) {
	Obj var = FindIdent("HooksBrkHighlightStart");
	Obj val = PTR_BAG(var)[0];
	RunHooks(val, NewList(0));
    }
}

void HooksBrkHighlightEnd(void) {
    if(HOOKS_INITIALIZED) {
	Obj var = FindIdent("HooksBrkHighlightEnd");
	Obj val = PTR_BAG(var)[0];
	RunHooks(val, NewList(0));
    }
}

void HooksEditFile(char* fileName, int lineNumber) {
    if(HOOKS_INITIALIZED) {
	Obj var = FindIdent("HooksEditFile");
	Obj val = PTR_BAG(var)[0];
	Obj arg = NewList(2);
	Obj str;
	C_NEW_STRING(str, fileName);
	SET_ELM_PLIST(arg, 1, str);
	SET_ELM_PLIST(arg, 2, INT_TO_HD(lineNumber));
	RunHooks(val, arg);
    }
}

Obj FunRunHooksBeforeOpenInput  (Obj call) { HookBeforeOpenInput();     return HdVoid; }
Obj FunRunHooksAfterOpenInput   (Obj call) { HookAfterOpenInput();      return HdVoid; }
Obj FunRunHooksBeforeCloseInput (Obj call) { HookBeforeCloseInput();    return HdVoid; }
Obj FunRunHooksAfterCloseInput  (Obj call) { HookAfterCloseInput();     return HdVoid; }
Obj FunRunHooksSessionStart     (Obj call) { HookSessionStart();        return HdVoid; }
Obj FunRunHooksSessionEnd       (Obj call) { HookSessionEnd();          return HdVoid; }
Obj FunRunHooksBrkHighlightStart(Obj call) { HooksBrkHighlightStart();  return HdVoid; }
Obj FunRunHooksBrkHighlightEnd  (Obj call) { HooksBrkHighlightEnd();    return HdVoid; }
Obj FunRunHooksEditFile		(Obj call) { 
    char* usage = "usage: EditFile(<fileName>, <lineNumber>)";
    if (GET_SIZE_BAG(call) == 3*SIZE_HD) {
	Obj str = EVAL(PTR_BAG(call)[1]);
	Obj ln = EVAL(PTR_BAG(call)[2]);
	if (GET_TYPE_BAG(str)!=T_STRING || GET_TYPE_BAG(ln) != T_INT)
	    Error(usage, 0, 0);
	else {
	    HooksEditFile(CSTR_STRING(str), HD_TO_INT(ln));
	}
    } else
	Error(usage, 0, 0);
    return HdVoid;
}

void InitHooks(void) {
    Obj l;
    GlobalPackage2("spiral", "hooks");
    HOOKS_INITIALIZED = 1;

    l=NewList(0); SET_BAG(FindIdent("HooksBeforeOpenInput"), 0, l);
    l=NewList(0); SET_BAG(FindIdent("HooksBeforeCloseInput"), 0, l);
    l=NewList(0); SET_BAG(FindIdent("HooksAfterOpenInput"), 0, l);
    l=NewList(0); SET_BAG(FindIdent("HooksAfterCloseInput"), 0, l);
    l=NewList(0); SET_BAG(FindIdent("HooksSessionStart"), 0, l);
    l=NewList(0); SET_BAG(FindIdent("HooksSessionEnd"), 0, l);
    l=NewList(0); SET_BAG(FindIdent("HooksBrkHighlightStart"), 0, l);
    l=NewList(0); SET_BAG(FindIdent("HooksBrkHighlightEnd"), 0, l);
    l=NewList(0); SET_BAG(FindIdent("HooksEditFile"), 0, l);

    InstIntFunc("RunHooksBeforeOpenInput",  FunRunHooksBeforeOpenInput);
    InstIntFunc("RunHooksAfterOpenInput",   FunRunHooksAfterOpenInput);
    InstIntFunc("RunHooksBeforeCloseInput", FunRunHooksBeforeCloseInput);
    InstIntFunc("RunHooksAfterCloseInput",  FunRunHooksAfterCloseInput);

    InstIntFunc("RunHooksSessionStart",     FunRunHooksSessionStart);
    InstIntFunc("RunHooksSessionEnd",       FunRunHooksSessionEnd);
    InstIntFunc("RunHooksBrkHighlightStart",FunRunHooksBrkHighlightStart);
    InstIntFunc("RunHooksBrkHighlightEnd",  FunRunHooksBrkHighlightEnd);
    
    InstIntFunc("RunHooksEditFile",	    FunRunHooksEditFile);
    EndPackage();
}
