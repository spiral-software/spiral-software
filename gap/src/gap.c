/* Changes:
   - prompt changed to "spiral>"
   - support for environmant variables
*/
/****************************************************************************
**
*A  gap.c                       GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file contains the various read-eval-print loops and  related  stuff.
**
*/

#include	<stdio.h>
#include	<stdlib.h>
#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "scanner.h"             /* reading of single tokens        */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */

#include        "idents.h"              /* 'InitIdents', 'FindIdent'       */
#include        "read.h"                /* 'ReadIt'                        */

#include        "list.h"                /* generic list package            */
#include        "plist.h"               /* 'LEN_PLIST', 'SET_LEN_PLIST',.. */
#include        "gstring.h"             /* 'IsString', 'PrintString'       */
#include        "string4.h"        /* strings                         */

#include        "statemen.h"            /* 'HdStat', 'StrStat'             */
#include        "function.h"            /* 'HdExec', 'ChangeEnv', 'PrintF' */
#include        "record.h"              /* 'HdCall*', 'HdTilde'            */

#include        "spiral.h"              /* InitSPIRAL, Try, Catch, exc     */
#include        "comments.h"            /* InitCommentBuffer, GetCommentB..*/
#include        "hooks.h"               /* InitHooks                       */
#include        "namespaces.h"          /* InitNamespaces                  */
#include        "args.h"

#include        "interface.h"           /* New Spiral Interface            */
#include        "iface.h"               
#include        "tables.h"
#include        "debug.h"
#include		"GapUtils.h"


extern Bag                  HdStack;
extern UInt        TopStack;

int CURR_INTERFACE = DEFAULT_INTERFACE;
int LAST_INTERFACE = DEFAULT_INTERFACE;

int ERROR_QUIET = 0;
int BACKTRACE_DEFAULT_LEVEL = 5;
/*V HdLastErrorMsg */
Obj HdLastErrorMsg;

/****************************************************************************
**
*V  HdLast  . . . . . . . . . . . . . .  handle of the variable 'last', local
*V  HdLast2 . . . . . . . . . . . . . . handle of the variable 'last2', local
*V  HdLast3 . . . . . . . . . . . . . . handle of the variable 'last3', local
**
**  'HdLast' is the handle of  the  variable  'last'.  This  global  variable
**  holds the result of the last evaluation in the main read-eval-print loop.
**  'HdLast2' likewise holds the next to last result, and 'HdLast3' holds the
**  result before that.
*/
Bag       HdLast, HdLast2, HdLast3;


/****************************************************************************
**
*V  HdTime  . . . . . . . . . . . . . .  handle of the variable 'time', local
**
**  'HdTime' is the handle of the variable 'time'.
**
**  'time' holds the time in milliseconds that  the  execution  of  the  last
**  statement took.  This variable is set at the end of  the  read-eval-print
**  cycle.
*/
Bag       HdTime;

/****************************************************************************
*V HdDbgStackRoot - value of HdExec when error loop starts
*V DbgStackTop - call stack depth from HdDbgStackRoot, Up()/Down() change
**  this variable.
*V DbgEvalStackTop - value of the EvalStackTop variable when error loop starts
*/
Int         DbgStackTop = 0;
Bag         HdDbgStackRoot = 0;
UInt        DbgEvalStackTop = 0;

Int         DbgInBreakLoop = 0;


/****************************************************************************
**
*F  main( <argc>, <argv> )  . . . . . . .  main program, read-eval-print loop
**
**  'main' is the entry point of GAP.  The operating sytem transfers  control
**  to this point when GAP is started.  'main' calls 'InitGap' to  initialize
**  everything.  Then 'main' starts the read-eval-print loop, i.e.,  it reads
**  expression, evaluates it and prints the value.  This continues until  the
**  end of the input file.
*/
int             main (int argc, char **argv)
{
    extern void         InitGap (int argc, char **argv, int *stackBase);
    exc_type_t          e;

   
    /* Save this information to reset */
    CURR_INTERFACE = DEFAULT_INTERFACE;
    interface_save_args(argc, argv);
 
	/*************************************************************************/
	/* printf("message from gap...\n");										 */
	/* char *p;																 */
	/* Int n = 50 * sizeof(char *);											 */
	/* p = (char *)malloc(n);												 */
	/* strcpy(p, "1234567890ABCDEF FEDCBA0987654321");						 */
	/* printf("# bytes allocated = %d, ptr = %X, value = %s\n", n, p, p);	 */
	/*************************************************************************/

	GapRunTime.gap_start = clock();
	
    Try {
		/* initialize everything                                             */
		InitGap( argc, argv, &argc );
    }
	Catch(e) {
		exc_show();
		return 1;
    }

    Try { 
		HookSessionStart(); 
    }
    Catch(e) {
		/* exceptions raised using Error() are already printed at this point */
		if(e!=ERR_GAP) 
		{ 
			exc_show(); 
			while ( HdExec != 0 )  
				ChangeEnv( PTR_BAG(HdExec)[4], CEF_CLEANUP ); 
		        while ( EvalStackTop>0 )
		                EVAL_STACK_POP;
		}
    }

    /* Load static history buffer */
    SyLoadHistory();
 
    /* Start Interface Main Evaluation here */
    start_interface(CURR_INTERFACE);
 
    /* Write static history buffer */
    SySaveHistory();
   
    Try { 
		HookSessionEnd();
    }
    Catch(e) {
		/* exceptions raised using Error() are already printed at this point */
		if(e!=ERR_GAP) 
		{ 
			exc_show(); 
			while ( HdExec != 0 )  
				ChangeEnv( PTR_BAG(HdExec)[4], CEF_CLEANUP ); 
			while ( EvalStackTop>0 )
		                EVAL_STACK_POP;
		}
    }

	//  GapRunTime.gap_end = clock();
	//  PrintRuntimeStats(&GapRunTime);
	
     /* exit to the operating system, the return is there to please lint    */
    if (NrHadSyntaxErrors && (LAST_INTERFACE == ID_BATCH)) 
        SyExit(SYEXIT_WITH_SYNTAX_ERRORS);
    else
        SyExit(SYEXIT_OK);

    return SYEXIT_OK;
}

/****************************************************************************
**
*F  FunBacktrace( <hdCall> )  . . . . . . . . . internal function 'Backtrace'
**
**  'FunBacktrace' implements the internal function 'Backtrace'.
**
**  'Backtrace()' \\
**  'Backtrace( <level> )'
**
**  'Backtrace' can be used inside a break loop to print  a  history  of  the
**  computation.  'Backtrace' prints a list of  all  active  functions,  most
**  recent first, up to maximal <level>  nestings.  If  <level>  is  positive
**  the names of the formal arguments of the  functions  calls  are  printed,
**  otherwise the  values  of  the  actual  arguments  are  printed  instead.
**  <level> default to 5, i.e., calling 'Backtrace'  with  no  argument  will
**  print the 5 most recent functions with the names of the formal arguments.
**
**  When a break loop (see "Break Loops") is entered  'Backtrace'  is  called
**  automatically.
*/
/*  PrintBacktraceExec(Bag hdExec, int execDepth, int printValues )
**      Called from Backtrace to print single T_EXEC bag
*/

UInt DbgExecStackDepth() {
    Obj	    hdExec;
    UInt    result = 0;
    for ( hdExec=HdDbgStackRoot; hdExec!=0; hdExec=PTR_BAG(hdExec)[4] ) {
        result++;
    }
    return result;
}

void PrintBacktraceExec(Bag hdExec, UInt execDepth, UInt execStackDepth, UInt printValues )
{
    int         nrArg,  nrLoc,  i;
    Bag   hdDef;
    
    if (execDepth==execStackDepth-DbgStackTop)
        Pr("* ", 0, 0);
    else
        Pr("  ", 0, 0);
    /* Print the depth like gdb does                                   */
    execDepth++;
    if(execDepth < 10)
        Pr("#%d -> ", execDepth, 0);
    else
        Pr("#%d-> ", execDepth, 0);
    
    if ( hdExec == 0 ) {
        Pr("main loop\n",0,0);
    } else {
        if ( PTR_BAG(hdExec)[3] == HdCallSum )       Pr("<rec1> + <rec2>",0,0);
        else if ( PTR_BAG(hdExec)[3] == HdCallDiff ) Pr("<rec1> - <rec2>",0,0);
        else if ( PTR_BAG(hdExec)[3] == HdCallProd ) Pr("<rec1> * <rec2>",0,0);
        else if ( PTR_BAG(hdExec)[3] == HdCallQuo )  Pr("<rec1> / <rec2>",0,0);
        else if ( PTR_BAG(hdExec)[3] == HdCallMod )  Pr("<rec1> mod <rec2>",0,0);
        else if ( PTR_BAG(hdExec)[3] == HdCallPow )  Pr("<rec1> ^ <rec2>",0,0);
        else if ( PTR_BAG(hdExec)[3] == HdCallComm ) Pr("Comm(<rec1>,<rec2>)",0,0);
        else if ( PTR_BAG(hdExec)[3] == HdCallEq )   Pr("<rec1> = <rec2>",0,0);
        else if ( PTR_BAG(hdExec)[3] == HdCallLt )   Pr("<rec1> < <rec2>",0,0);
        else if ( PTR_BAG(hdExec)[3] == HdCallIn )   Pr("<elm> in <rec>",0,0);
        else if ( PTR_BAG(hdExec)[3] == HdCallPrint ) Pr("Print( <rec> )",0,0);
        else {
            if (printValues) {
                Print( PTR_BAG( PTR_BAG(hdExec)[3] )[0] );
                Pr("%>( %>",0,0);
                hdDef = EVAL( PTR_BAG( PTR_BAG(hdExec)[3] )[0] );
                ACT_NUM_ARGS_FUNC(hdDef, nrArg);
                ACT_NUM_LOCALS_FUNC(hdDef, nrLoc);
                for ( i = 1; i <= nrArg; ++i ) {
                    Print( PTR_BAG(hdExec)[EXEC_ARGS_START+i+nrArg+nrLoc-1] );
                    if ( i < nrArg )  Pr("%<, %>",0,0);
                }
                Pr(" %2<)",0,0);
            } else {
                Print( PTR_BAG(hdExec)[3] );
            }
        }
        Pr("\n",0,0);
    }
}

void PrintBacktraceEval(Bag hdExec)
{
    int StackPnt = HD_TO_INT(PTR_BAG(hdExec)[EXEC_EVAL_STACK]) + 1;
    int StackStart = StackPnt;
    
    while (StackPnt<DbgEvalStackTop && GET_TYPE_BAG(EvalStack[StackPnt+1])!=T_EXEC)
        StackPnt++;
    // do not print statement just before T_EXEC, that should be the same statement.
    if (StackPnt<DbgEvalStackTop && GET_TYPE_BAG(EvalStack[StackPnt+1])==T_EXEC)
        StackPnt--;
    while (StackPnt>=StackStart) {
        Obj item = EvalStack[StackPnt];
        if (GET_TYPE_BAG(item)==T_FUNCCALL)
            Pr("          %g\n", (Int)PTR_BAG(item)[0], 0);
        else
            Pr("          %s\n", (Int)NameType[GET_TYPE_BAG(item)], 0);
        StackPnt--;
    }
}

Bag       FunBacktrace (Bag hdCall)
{
    Int	    level;
    UInt    depth = 0, execStackDepth;
    Bag     hdExec, hdDef;
    
    /* so that it is possible to call it FunBacktrace(INT_TO_HD(<level>))  */
    if ( hdCall != (Bag) 0 && GET_TYPE_BAG(hdCall) == T_INT ) {
	level = HD_TO_INT(hdCall);
    }
    /* get the value of <level>                                            */
    else if ( hdCall == (Bag) 0 || GET_SIZE_BAG(hdCall) == SIZE_HD ) {
        level = BACKTRACE_DEFAULT_LEVEL;
    }
    else if ( GET_SIZE_BAG(hdCall) == 2 * SIZE_HD ) {
        hdDef = EVAL( PTR_BAG(hdCall)[1] );
        if ( GET_TYPE_BAG(hdDef) != T_INT )
            return Error("usage: Backtrace( <level> )",0,0);
        else
            level = HD_TO_INT( hdDef );
    }
    else {
        return Error("usage: Backtrace( <level> )",0,0);
    }

    if ( level == 0 ) return HdVoid;

    execStackDepth = DbgExecStackDepth();
    depth = execStackDepth;
    /* for <level> frames                                                  */
    for ( hdExec=HdDbgStackRoot; hdExec!=0 && level!=0; hdExec=PTR_BAG(hdExec)[4] ) {
        /* if <level> is positive print only the names of the formal args  */
        if ( 0 < level ) {
            PrintBacktraceExec(hdExec,  depth, execStackDepth, 0);
            --level;
        }
        /* if <level> is negative print the values of the arguments        */
        else {
            PrintBacktraceExec(hdExec,  depth, execStackDepth, 1);
            ++level;
        }
        --depth;
    }

    /* print the bottom of the function stack                              */
    if ( hdExec == 0 ) {
        PrintBacktraceExec(0, depth, execStackDepth, 0);
    }
    else {
        Pr("...\n",0,0);
    }
    
    return HdVoid;
}


Bag       FunBacktrace2 (Bag hdCall)
{
    Int	    level;
    UInt    depth = 0, execStackDepth;
    Bag           hdExec, hdDef;
    
    /* so that it is possible to call it FunBacktrace2(INT_TO_HD(<level>))  */
    if ( hdCall != (Bag) 0 && GET_TYPE_BAG(hdCall) == T_INT ) {
	level = HD_TO_INT(hdCall);
    }
    /* get the value of <level>                                            */
    else if ( hdCall == (Bag) 0 || GET_SIZE_BAG(hdCall) == SIZE_HD ) {
        level = BACKTRACE_DEFAULT_LEVEL;
    }
    else if ( GET_SIZE_BAG(hdCall) == 2 * SIZE_HD ) {
        hdDef = EVAL( PTR_BAG(hdCall)[1] );
        if ( GET_TYPE_BAG(hdDef) != T_INT )
            return Error("usage: Backtrace2( <level> )",0,0);
        else
            level = HD_TO_INT( hdDef );
    }
    else {
        return Error("usage: Backtrace2( <level> )",0,0);
    }

    if ( level == 0 ) return HdVoid;
    /* calculate stack depth */
    depth = DbgExecStackDepth();
    execStackDepth = depth;
    /* for <level> frames                                                  */
    for ( hdExec=HdDbgStackRoot; hdExec!=0 && level!=0; hdExec=PTR_BAG(hdExec)[4] ) {
        /* if <level> is positive print only the names of the formal args  */
        PrintBacktraceEval(hdExec);
        if ( 0 < level ) {
            PrintBacktraceExec(hdExec,  depth, execStackDepth, 0);
            --level;
        }
        /* if <level> is negative print the values of the arguments        */
        else {
            PrintBacktraceExec(hdExec,  depth, execStackDepth, 1);
            ++level;
        }
        --depth;
    }

    /* print the bottom of the function stack                              */
    if ( hdExec == 0 ) {
        PrintBacktraceExec(0, depth, execStackDepth, 0);
    }
    else {
        Pr("...\n",0,0);
    }
    
    return HdVoid;
}

/****************************************************************************
**
*F  FunBacktraceTo( <hdCall> ) . . . . . . . . internal function 'BacktraceTo'
**
**  'BacktraceTo( <filename>, <level> )'
**
**  'BacktraceTo' prints output of Backtrace(<level>) to a file with name 
**  <filename>.
**
**  'BacktraceTo' is a procedure, i.e., does not return a value.
*/
Bag       FunBacktraceTo (Bag hdCall)
{
    Bag           hdName, hdLevel;

    char * usage = "usage: BacktraceTo( <file>, <level> )";
    /* check the number and type of the arguments, nothing special         */
    if ( GET_SIZE_BAG(hdCall) != SIZE_HD*3 ) return Error(usage,0,0);
    hdName = EVAL( PTR_BAG(hdCall)[1] );
    hdLevel = EVAL( PTR_BAG(hdCall)[2] );
    if ( ! IsString(hdName) ) return Error(usage,0,0);
    if ( GET_TYPE_BAG(hdLevel)!=T_INT) return Error(usage,0,0);

    /* try to open the given output file, raise an error if you can not    */
    if ( OpenOutput( (char*)PTR_BAG(hdName) ) == 0 )
        return Error("BacktraceTo: can not open '%s' for writing",
		     (Int)PTR_BAG(hdName), 0);

    FunBacktrace(hdLevel);

    /* close the output file again, and return nothing                     */
    if ( ! CloseOutput() )
        Error("BacktraceTo: can not close output, this should not happen",0,0);
    return HdVoid;
}

/****************************************************************************
**
*F  Error( <msg>, <arg1>, <arg2> )  . . . . . . . . . . . . . . error handler
**
**  'Error' is the GAP kernel error handler.
*/
/*static int InError = 0;*/

Int     inBreakLoop()
{
    return DbgInBreakLoop != 0;
}

Obj  DbgStackExec() {
    Obj root = HdDbgStackRoot; 
    int top = DbgStackTop;
    while (root != 0 && top>0) {
        root = PTR_BAG(root)[4];
        top--;
    }
    return root;
}

void EmptyStack() {
    while ( TopStack >= 1 )
	SET_BAG(HdStack, TopStack--,  0 );
}

/* Enter all funcdef bags from the exec list onto the        
 * stack, so that we can access args and locals in the loop  
 */

void PopulateStack() {
    Obj hd, hdRoot;
    int i;
    
    EmptyStack();
    
    hdRoot = DbgStackExec();
    
    hd = hdRoot;  TopStack = 0;
    while ( hd != 0 && TopStack+1 < GET_SIZE_BAG(HdStack)/SIZE_HD ) {
	++TopStack;
	hd = PTR_BAG(hd)[0];
    }
    hd = hdRoot;  i = 0;
    while ( hd != 0 && TopStack-i+1 > 0 ) {
	++i;
	SET_BAG(HdStack, TopStack-i+1,  PTR_BAG(hd)[2] );
	hd = PTR_BAG(hd)[0];
    }
}

void    DbgWalkEvalStackBackward(int exchangeVars) {
    Obj item;
    Obj hdExec = DbgStackExec();
    // walk on eval stack and pop T_MAKELET packages,
    // restore previous local variables values for each package 
    int i = HD_TO_INT(PTR_BAG(hdExec)[EXEC_EVAL_STACK]);
    while (i<DbgEvalStackTop && GET_TYPE_BAG(EvalStack[i+1])!=T_EXEC) i++;
    item = EvalStack[i];
    while (item != hdExec) {
        if (GET_TYPE_BAG(item)==T_LIST) { // this is some variable values on the stack
            Obj hd = PTR_BAG(item)[1];
            if (GET_TYPE_BAG(hd)==T_MAKELET) {  //let statement
                PopPackage();
                if (exchangeVars) {
                    int j, size = TableNumEnt(hd) - 1;
                    for ( j=0; j<size; j++ ) {
                        Obj var = PTR_BAG(hd)[j];
                        Obj oldbinding = PTR_BAG(item)[j+2];
                        SET_BAG(item, j+2,  VAR_VALUE(var) );
                        SET_VAR_VALUE(var, oldbinding);
                    }
                }
            }
        }
        item = EvalStack[--i];
    }
}

void    DbgWalkEvalStackForward(int exchangeVars) {
    int i = HD_TO_INT(PTR_BAG(DbgStackExec())[EXEC_EVAL_STACK])+1;
    while (i<=DbgEvalStackTop && GET_TYPE_BAG(EvalStack[i])!=T_EXEC) {
        Obj item = EvalStack[i++];
        if (GET_TYPE_BAG(item)==T_LIST) { // this is some variable values on the stack
            Obj hd = PTR_BAG(item)[1];
            if (GET_TYPE_BAG(hd)==T_MAKELET) { //let statement
                PushPackage(hd);
                if (exchangeVars) {
                    int j, size = TableNumEnt(hd) - 1;
                    for ( j=0; j<size; j++ ) {
                        Obj var = PTR_BAG(hd)[j];
                        Obj oldbinding = PTR_BAG(item)[j+2];
                        SET_BAG(item, j+2,  VAR_VALUE(var) );
                        SET_VAR_VALUE(var, oldbinding);
                    }
                }
            }
        }
    } // while end
}

Int     DbgUp() {
    if ( DbgStackTop>0 ) {
        DbgWalkEvalStackBackward(0);    
        DbgStackTop--;
	ChangeEnv(DbgStackExec(), CEF_DBG_UP);
	DbgWalkEvalStackForward(1);    
        PopulateStack();
        return 1;
    }
    return 0;
}

Int     DbgDown() {
    Obj hdExec = DbgStackExec();
    if (hdExec) {
        if (PTR_BAG(hdExec)[4]) {
            DbgWalkEvalStackBackward(1);
            ChangeEnv(PTR_BAG(hdExec)[4], CEF_DBG_DOWN);
            DbgStackTop++;
            DbgWalkEvalStackForward(0);
            PopulateStack();
            return 1;
        }
    }
    return 0;
}

void EnterDbgStack() {
    if (DbgInBreakLoop) {
        // ??? crash 
    }
    DbgEvalStackTop = EvalStackTop;
    DbgStackTop = 0;
    HdDbgStackRoot = HdExec;
    DbgInBreakLoop = 1;
    DbgWalkEvalStackForward(0);
    PopulateStack();
}

void LeaveDbgStack() {
    if (DbgInBreakLoop==0) {
        // ??? crash
    }
    
    while (DbgStackTop>0) DbgUp();
    DbgWalkEvalStackBackward(0);

    // clean EVAL stack in the case of exception
    while (EvalStackTop>DbgEvalStackTop) EVAL_STACK_POP;
        
    DbgEvalStackTop = 0;
    HdDbgStackRoot = 0;
    DbgStackTop = 0;
    DbgInBreakLoop = 0;
    
    PopulateStack();
}


static char BrkPrompt[80] = "brk> ";
static char DbgPrompt[80] = "dbg> ";

Bag       Error (char *msg, Int arg1, Int arg2)
{
	Bag           hd;
	Bag           hdTilde; 
	Int                ignore;
	extern Bag    FunPrint(Bag hdCall);
	extern char         * In;
	TypInputFile        * parent;
	exc_type_t          e;
	Int                isBreakpoint;

	if ( ! ERROR_QUIET ) {

		/* open the standard error output file                                */
		/*if ( ! InError )*/
		isBreakpoint = 0;
		if (strcmp(msg, "GapBreakpoint")==0) isBreakpoint = 1; 
		if (strcmp(msg, "GapBreakpointRd")==0) isBreakpoint = 2;
		if (strcmp(msg, "GapBreakpointWr")==0) isBreakpoint = 3;
		if ( DbgInBreakLoop==0 ) {
			ignore = OpenOutput( "*errout*" );
			if (!isBreakpoint)
				Pr("[[ while reading %s:%d ]]\n", (Int)Input->name, (Int)Input->number);
		}
		if (isBreakpoint) {
			switch(isBreakpoint) {
			case 2: { Pr("Read Access Breakpoint",0,0); break; } 
			case 3: { Pr("Write Access Breakpoint",0,0); break; }
			default:
				Pr("Breakpoint",0,0); 
			}
		} else {
			/* print the error message, special if called from 'FunError'      */
			if ( strcmp( msg, "FunError" ) != 0 ) {
				Pr("Error, ",0,0);  Pr( msg, arg1, arg2 );
			} else {
				Pr("Error, ",0,0);  FunPrint( (Bag)arg1 );
			}
		}

		/* print the error message                                             */
		if ( HdExec != 0 && DbgInBreakLoop==0 ) {
			/* we have to do something about this as we have more detailed
			call stack now
			if ( HdStat != 0 && strcmp( msg, "FunError" ) != 0 ) {
			Pr(" at\n%s", (long)StrStat, 0 );
			Print( HdStat );
			Pr(" ...",0,0);
			}
			Pr(" in\n",0,0);
			*/
			if (DbgEvalStackTop>0 && strcmp( msg, "FunError" ) != 0) {
				Pr(" at\n", 0, 0 );
				Print( EvalStack[DbgEvalStackTop] );
				Pr(" ...",0,0);
			}

			Pr(" in\n",0,0);
		}
		else {
			Pr("\n",0,0);
		}

		if ( DbgInBreakLoop == 0 ){
			/* we must disable tilde during break loop processing, since it points to
			a broken object. We may need to restore it if the user 'return's from the
			break loop, so we keep a local copy */
			hdTilde = PTR_BAG(HdTilde)[0];
			SET_BAG(HdTilde, 0,  0 );

			parent = Input;

			/* if requested enter a break loop                                     */
			if ( HdExec != 0 && OpenInput( "*errin*" ) ) {

				if(parent->packages) PushPackages(parent->packages);
				if(parent->imports) PushNamespaces(parent->imports);

				EnterDbgStack();
				Try {
					if (isBreakpoint)
						FunBacktrace2( (Bag)0 );
					else {
						FunBacktrace( (Bag)0 );
						Pr("web:error\n", 0, 0);
					}
					DbgErrorLoopStarting();
				} Catch(e) { if (e != ERR_GAP) { LeaveDbgStack(); Throw(e); } }
				/* now enter a read-eval-print loop, just as in main               */
				while ( Symbol != S_EOF ) {
					if (CURR_INTERFACE == ID_BATCH) {
						/* inconsistency in interfaces: ReadIt ignoring interfaces,
						quit from here if we are in batch interface */
						SyExit(SYEXIT_FROM_BRK);
					}
					/* read an expression                                          */
					if (InBreakpoint) {
						Prompt = DbgPrompt; 
					} else {
						Prompt = BrkPrompt;
					}
					NrError = 0;
					hd = ReadIt();
					/* if there we no syntax error evaluate the expression         */
					if ( hd != 0 ) {
						SyIsIntr();
						Try {
							hd = EVAL( hd );
						} Catch(e) {
							if (e != ERR_GAP) { LeaveDbgStack(); Throw(e); }
							hd = HdVoid;
						}
						if ( hd == HdReturn && PTR_BAG(hd)[0] != HdReturn ) {
							LeaveDbgStack(); 
							SET_BAG(HdTilde, 0,  hdTilde ); /* restore ~ */
							ignore = CloseInput();
							ignore = CloseOutput();
							/*InError = 0;*/
							return PTR_BAG(hd)[0];
						}
						else if ( hd == HdReturn ) {
							hd = HdVoid;
							Symbol = S_EOF;
						}

						/* assign the value to 'last' and then print it            */
						if ( GET_TYPE_BAG(hd) != T_VOID ) {
							SET_BAG(HdLast, 0,  hd );
							if ( *In != ';' ) {
								Try {
									Print( hd );
									Pr("\n",0,0);
								} Catch(e) { if (e != ERR_GAP) { LeaveDbgStack(); Throw(e); } }
							}
						}
					}
				}
				/* remove function definitions from the stack and close "*errin*"  */
				LeaveDbgStack();
				ignore = CloseInput();
			} else {
				if (CURR_INTERFACE == ID_BATCH) {
					/* quit with the first error */
					SyExit(SYEXIT_FROM_BRK);
				}
			}

			while ( HdExec != 0 )  ChangeEnv( PTR_BAG(HdExec)[4], CEF_CLEANUP );
			while ( EvalStackTop > 0 ) EVAL_STACK_POP;

			/* close "*errout*" and return to the main read-eval-print loop        */
			while ( CloseOutput() ) ;
			while ( CloseInput() ) ;
		} else { // if we are already in error loop just cleanup stack
			while ( HdExec != DbgStackExec() )  ChangeEnv( PTR_BAG(HdExec)[4], CEF_CLEANUP );
			while ( EvalStackTop > DbgEvalStackTop ) EVAL_STACK_POP;
		}
	} else {
		HdLastErrorMsg = StringToHd(msg);
	}

	/*InError = 0;*/
	Throw exc(ERR_GAP);
	return 0;                           /* just to please lint ...         */
}


/****************************************************************************
**
*F  FunIgnore( <hdCall> ) . . . . . . . . . . . .  internal function 'Ignore'
**
**  'FunIgnore' implements the internal function 'Ignore'.
**
**  'Ignore( <arg1>, <arg2>, ... )'
**
**  'Ignore' ignores all its arguments,  it does not even evaluate  them.  So
**  for tracing a GAP function,  use a function 'InfoSomething'  which either
**  has value 'Print' and prints its arguments or has value 'Ignore' and does
**  nothing at all.
*/
Bag       FunIgnore(Bag hdCall)
{
    return HdVoid;
}


/****************************************************************************
**
*F  FunError( <hdCall> )  . . . . . . . . . . . . . internal function 'Error'
**
**  'FunError' implements the internal function 'Error'.
**
**  'Error( <arg1>, <arg2>,... )'
**
**  raises an error.
**  ...A lot of bla about errors and break loops...
**
**  'FunError' simply calls the GAP  kernel  function  'Error',  which  knows
**  that it has been called from 'FunError' because the  format  argument  is
**  'FunError'.  'FunError' passes <hdCall> as the first extra argument.
*/
Bag       FunError (Bag hdCall)
{
    return Error("FunError", (Int)hdCall, 0 );
}


/****************************************************************************
**
*F  FunWindowCmd( <hdCall> )  . . . . . . . . . . .  execute a window command
*/
Bag	FunWindowCmd (Bag hdCall)
{
    Bag       hdStr;
    Bag       hdTmp;
    Bag       hdCmd;
    Bag       hdLst;
    Int            len;
    Int            n,  m;
    Int            i;
    char          * ptr;
    char          * qtr;

    /* check arguments                                                     */
    if ( GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
	return Error( "usage: WindowCmd( <cmds> )", 0, 0 );
    hdCmd = EVAL(PTR_BAG(hdCall)[1]);
    if ( !IsList(hdCmd) )
	return Error( "usage: WindowCmd( <cmds> )", 0, 0 );
    hdTmp = ELM_LIST(hdCmd,1);
    if ( GET_TYPE_BAG(hdTmp) != T_STRING )
	return Error( "<cmd> must be a string", 0, 0 );
    if ( GET_SIZE_BAG(hdTmp) != 4 )
	return Error( "<cmd> is not a valid command", 0, 0 );

    /* compute size needed to store argument string                        */
    len   = 13;
    hdLst = NewBag( T_LIST, (LEN_LIST(hdCmd)+1)*SIZE_HD );
    for ( i = LEN_LIST(hdCmd);  1 < i;  i-- )
    {
	hdTmp = ELM_LIST(hdCmd,i);
	if ( GET_TYPE_BAG(hdTmp) != T_INT && ! IsString(hdTmp) )
	    return Error("%d.th argument must be a string or integer",i,0);
	SET_BAG(hdLst, i,  hdTmp );
	if ( GET_TYPE_BAG(hdTmp) == T_INT )
	    len += 12;
	else
	    len += 5 + 2*GET_SIZE_BAG(hdTmp);
    }

    /* convert <hdCall> into an argument string                            */
    hdStr  = NewBag( T_STRING, len + 13 );
    ptr    = (char*) PTR_BAG(hdStr);
    *ptr   = '\0';

    /* first the command name                                              */
    strncat( ptr, (char*)PTR_BAG(ELM_LIST(hdCmd,1)), 3 );
    ptr += 3;

    /* and at last the arguments                                           */
    for ( i = 2;  i < GET_SIZE_BAG(hdLst)/SIZE_HD;  i++ )
    {
	hdTmp = PTR_BAG(hdLst)[i];
	if ( GET_TYPE_BAG(hdTmp) == T_INT )
	{
	    *ptr++ = 'I';
	    m = HD_TO_INT(hdTmp);
	    for ( m = (m<0)?-m:m;  0 < m;  m /= 10 )
		*ptr++ = (m%10) + '0';
	    if ( HD_TO_INT(hdTmp) < 0 )
		*ptr++ = '-';
	    else
		*ptr++ = '+';
	}
	else
	{
	    *ptr++ = 'S';
	    m = GET_SIZE_BAG(hdTmp)-1;
	    for ( n = 7;  0 <= n;  n--, m /= 10 )
		*ptr++ = (m%10) + '0';
	    qtr = (char*) PTR_BAG(hdTmp);
	    for ( m = GET_SIZE_BAG(hdTmp)-1;  0 < m;  m-- )
		*ptr++ = *qtr++;
	}
    }
    *ptr = 0;

    /* compute correct length of argument string                           */
    qtr = (char*) PTR_BAG(hdStr);
    len = (Int)(ptr - qtr);

    /* now call the window front end with the argument string              */
    ptr = SyWinCmd( qtr, len );
    len = strlen(ptr);

    /* now convert result back into a list                                 */
    hdLst = NewBag( T_LIST, SIZE_PLEN_PLIST(11) );
    SET_LEN_PLIST( hdLst, 0 );
    i = 1;
    while ( 0 < len )
    {
	if ( *ptr == 'I' )
	{
	    ptr++;
	    for ( n=0,m=1; '0' <= *ptr && *ptr <= '9'; ptr++,m *= 10,len-- )
			n += ((Int)(*ptr) - (Int)'0') * m;
	    if ( *ptr++ == '-' )
		n *= -1;
	    len -= 2;
	    AssPlist( hdLst, i, INT_TO_HD(n) );
	}
	else if ( *ptr == 'S' )
	{
	    ptr++;
	    for ( n = 0, m = 7;  0 <= m;  m-- )
			n = n*10 + ((Int)ptr[m] - (Int)'0');
	    hdTmp = NewBag( T_STRING, n+1 );
	    *(char*)PTR_BAG(hdTmp) = '\0';
	    ptr += 8;
	    strncat( (char*)PTR_BAG(hdTmp), ptr, n );
	    ptr += n;
	    len -= n+9;
	    AssPlist( hdLst, i, hdTmp );
	}
	else
	    return Error( "unknown return value '%s'", (Int)ptr, 0 );
	i++;
    }

    /* if the first entry is one signal an error */
    if ( ELM_LIST(hdLst,1) == INT_TO_HD(1) )
    {
	hdStr = NewBag( T_STRING, 30 );
	strncat( (char*) PTR_BAG(hdStr), "window system: ", 15 );
	SET_ELM_PLIST( hdLst, 1, hdStr );
	Resize( hdLst, i*SIZE_HD );
	return Error( "FunError", (Int)hdLst, 0 );
    }
    else
    {
	for ( m = 1;  m <= i-2;  m++ )
	    SET_ELM_PLIST( hdLst,m, ELM_LIST(hdLst,m+1) );
	SET_LEN_PLIST( hdLst, i-2 );
	return hdLst;
    }
}


/****************************************************************************
**
*F  FunREAD( <hdCall> ) . . . . . . . . . . . . . .  internal function 'READ'
**
**  'FunREAD' implements the internal function 'READ'.
**
**  'READ( <filename> )'
**
**  'READ' instructs GAP to read from the file with the  name  <filename>. If
**  it is not found or could not be opened for reading  'false'  is returned.
**  If the file is found GAP reads all expressions and statements  from  this
**  file and evaluates respectively executes them and finally returns 'true'.
**  Then GAP continues evaluation or execution of what it was  doing  before.
**  'READ' can be nested, i.e., it is legal to execute a 'READ' function call
**  in a file that is read with 'READ'.
**
**  If a syntax error is found 'READ' continues reading the  next  expression
**  or statement, just  as  GAP  would  in  the  main  read-eval-print  loop.
**  If an evaluation error occurs, 'READ' enters a break loop.  If you 'quit'
**  this break loop, control returns to the  main  read-eval-print  loop  and
**  reading of <filename> terminates.
**
**  Note that this function is a helper function for  'Read',  which  behaves
**  similar, but causes an error if a file is not found.  'READ'  could  also
**  be used for a 'ReadLib' which searches for a file in various directories.
*/
Bag       FunREAD (Bag hdCall)
{
    Bag           hd,  hdName,  hdPkg;
    TypInputFile * parent;
    exc_type_t e;

	UInt processInclude;

    /* check the number and type of arguments                              */
    if ( GET_SIZE_BAG(hdCall) != 2*SIZE_HD && GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error("usage: READ( <filename>, [<pkg>] )",0,0);
    hdName = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IsString(hdName) ) return Error("usage: READ( <filename> )",0,0);
    if ( GET_SIZE_BAG(hdCall) == 3*SIZE_HD ) {
        hdPkg = EVAL(PTR_BAG(hdCall)[2]);
	hdPkg = StartPackageSpec(hdPkg); /* try it out */
	EndPackage(); 
    } 
    else hdPkg = 0;

    parent = Input;
    /* try to open the given file, if the file is not found return 'false' */
    if ( ! OpenInput( (char*)PTR_BAG(hdName) ) )
        return HdFalse;

    if ( hdPkg ) { 
      if(parent->packages) PushPackages(parent->packages);
      if(parent->imports) PushNamespaces(parent->imports);
      StartPackageSpec(hdPkg); 
    }

    /* now comes a read-eval-noprint loop, similar to the one in 'main'    */
	Try {
        while ( Symbol != S_EOF ) {
            hd = ReadIt();
			if ( hd != 0 ) { 
				hd = EVAL( hd );
			}				
			if ( hd == HdReturn && PTR_BAG(hd)[0] != HdReturn )
				return Error("READ: 'return' must not be used here",0,0);
			else if ( hd == HdReturn )
				return Error("READ: 'quit' must not be used here",0,0);
        }
    } Catch(e) {
        if ( hdPkg ) EndPackage();
        Throw(e);
    }
    if ( hdPkg ) EndPackage();
    /* close the input file again, and return 'true'                       */
    if ( ! CloseInput() )
        Error("READ: can not close input, this should not happen",0,0);
	
	return HdTrue;
}


Bag     FunChangeDir (Bag hdCall)
{
    Bag           hd,  hdName,  hdPkg;
    exc_type_t e;

    /* check the number and type of arguments                              */
    if ( GET_SIZE_BAG(hdCall) != SIZE_HD && GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
        return Error("usage: CHANGEDIR( <filename> )",0,0);
    hdName = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IsString(hdName) ) return Error("usage: CHANGEDIR( <filename> )",0,0);

    if ( ! (int)ChDir( (const char*)PTR_BAG(hdName) ) )
        return HdFalse;

    return HdTrue;
}


Bag       FunReadString (Bag hdCall)
{
    Bag           hdList,  hdName;
    TypInputFile * parent;

    /* check the number and type of arguments                              */
    if ( GET_SIZE_BAG(hdCall) != SIZE_HD && GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
        return Error("usage: READSTR( <filename> )",0,0);
    hdName = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IsString(hdName) ) return Error("usage: READSTR( <filename> )",0,0);

    parent = Input;
    /* try to open the given file, if the file is not found return 'false' */
    if ( ! OpenInput( (char*)PTR_BAG(hdName) ) )
        return HdFalse;

        hdList = GReadFile();

    /* close the input file again, and return 'true'                       */
    if ( ! CloseInput() )
        Error("READSTR: can not close input, this should not happen",0,0);
        
        return hdList;
}



/****************************************************************************
**
*F  FunAUTO( <hdCall> ) . . . . . . . . . . . . . .  internal function 'AUTO'
**
**  'FunAUTO' implements the internal function 'AUTO'.
**
**  'AUTO( <expression>, <var1>, <var2>,... )'
**
**  'AUTO' associates the expression <expression> with the variables <var1>,
**  <var2> etc.  Whenever one those variables is evaluated, i.e.,  when  its
**  value is required, <expression> is automatically  evaluated.  This  must
**  assign a new value to the variable, otherwise an error  is  raised.  The
**  new value is then returned.
**
**  Here is an example of the most important special usage of 'AUTO':
**
**  |    AUTO( ReadLib("integer"), Int, Abs, Sign, Maximum, Minimum ); |
**
**  When one of the variables, 'Int', 'Abs', etc., is  evaluated  the  libary
**  file 'integer.g' is automatically read.  This then defines the functions.
**  This makes it possible to load the library function only on demand.
**
**  'AUTO' is a procedure, i.e., does not return a value.
*/
Bag       FunAUTO (Bag hdCall)
{
    Bag           hdExpr,  hdVar;
    Int                i;

    /* check the number of arguments                                       */
    if ( GET_SIZE_BAG(hdCall) < 3 * SIZE_HD )
        return Error("usage: AUTO( <expr>, <var>, <var>... )",0,0);

    /* get the expression                                                  */
    hdExpr = PTR_BAG(hdCall)[1];

    /* for all remaining arguments                                         */
    for ( i = 2; i < GET_SIZE_BAG(hdCall)/SIZE_HD; ++i ) {
        hdVar = PTR_BAG(hdCall)[i];

        /* check that they are variables                                   */
        if ( GET_TYPE_BAG(hdVar) != T_VAR && GET_TYPE_BAG(hdVar) != T_VARAUTO )
            return Error("usage: AUTO( <expr>, <var>, <var>... )",0,0);

        /* turn them into automatic variables and bind them to <expr>      */
        Retype( hdVar, T_VARAUTO );
        SET_BAG(hdVar, 0,  hdExpr );

    }

    return HdVoid;
}


/****************************************************************************
**
**  FunPrint( <hdCall> )  . . . . . . . . . . . . . internal function 'Print'
**
**  'FunPrint' implements the internal function 'Print'.
**
**  'Print( <obj1>, <obj2>... )'
**
**  'Print' prints the objects <obj1>, <obj2>,  etc.  one  after  the  other.
**  Strings are printed without the double quotes and special characters  are
**  not escaped, e.g., '\n' is printed as <newline>.  This  makes  a  limited
**  amount of formatting possible.  Functions are printed in the  full  form,
**  i.e., with the function body, not in the abbreviated form.
**
**  'Print' is a procedure, i.e., does not return a value.
**
**  Note that an empty string literal '""' prints empty (remember strings are
**  printed without the double quotes), while an empty list  '[]'  prints  as
**  '[ ]'.
**
**      gap> s := "";;  l := [];;  s = l;
**      gap> Print( s, "\n", l, "\n" );
**      
**      [ ]
**
**  To achieve this 'Print' must be able to distinguish between empty  string
**  literals and other empty lists.  For that it relies on  'IsString'  *not*
**  to convert empty lists to type 'T_STRING'.  This is ugly.
*/
Bag       FunPrint (Bag hdCall)
{
    Bag           hd;
    Int                i;

    /* print all the arguments, take care of strings and functions         */
    for ( i = 1; i < GET_SIZE_BAG(hdCall)/SIZE_HD; ++i ) {
	Int type;
        hd = EVAL( PTR_BAG(hdCall)[i] );
	type = GET_TYPE_BAG( hd );
        if ( IsString( hd ) && GET_TYPE_BAG(hd) == T_STRING )  PrintString( hd );
        else if ( type == T_MAKEFUNC )           PrintFunction( hd );
        else if ( type == T_FUNCTION )           PrintFunction( hd );
        else if ( type == T_MAKEMETH )           PrintMethod( hd );
        else if ( type == T_METHOD )             PrintMethod( hd );
        else if ( type != T_VOID )               Print( hd );
        else  /*hd = Error("function must return a value",0,0);*/;
    }

    return HdVoid;
}

/****************************************************************************
**
*F  Fun_Pr( <hdCall> ) . . . . . . . . . . . . . . . internal function '_Pr'
**
**  'Fun_Pr' implements direct interface for Pr() function. It can print
**  strings only. String passed directly into Pr() without any processing.
**
**  _Pr( <string>, <string>, ... )
*/

Bag       Fun_Pr (Bag hdCall)
{
    Bag           hd;
    Int                i;

    /* print all the arguments, take care of strings and functions         */
    for ( i = 1; i < GET_SIZE_BAG(hdCall)/SIZE_HD; ++i ) {
	Int type;
        hd = EVAL( PTR_BAG(hdCall)[i] );
	type = GET_TYPE_BAG( hd );
        if ( IsString( hd ) && GET_TYPE_BAG(hd) == T_STRING )  Pr( CSTR_STRING (hd), 0, 0 );
        else  /*hd = Error("function must return a value",0,0);*/;
    }

    return HdVoid;
}

/****************************************************************************
**
*F  FunPrntTo( <hdCall> ) . . . . . . . . . . . . internal function 'PrintTo'
**
**  'FunPrntTo' implements the internal function 'PrintTo'.  The stupid  name
**  is neccessary to avoid a name conflict with 'FunPrint'.
**
**  'PrintTo( <filename>, <obj1>, <obj2>... )'
**
**  'PrintTo' prints the objects <obj1>, <obj2>, etc. to the  file  with  the
**  name <filename>.
**
**  'PrintTo' works as follows.  It opens the file with the name  <filename>.
**  If the file does not exist it is  created,  otherwise  it  is  truncated.
**  If you do not want to truncate the file use 'AppendTo'  (see "AppendTo").
**  After opening the file 'PrintTo' evaluates  its  arguments  in  turn  and
**  prints the values to  <filename>.  Finally  it  closes  the  file  again.
**  During evaluation of the arguments <filename> is the current output file.
**  This means that output printed with 'Print' during  the  evaluation,  for
**  example to inform the user about the progress, also goes  to  <filename>.
**  To make this feature more useful 'PrintTo' will silently ignore if one of
**  the arguments is a procedure call, i.e., does not return a value.
**
**  'PrintTo' is a procedure, i.e., does not return a value.
**
**  See the note about empty string literals and empty lists in 'Print'.
*/
Bag       FunPrntTo (Bag hdCall)
{
    Bag           hd;
    Int                i;

    /* check the number and type of the arguments, nothing special         */
    if ( GET_SIZE_BAG(hdCall) == SIZE_HD )
        return Error("usage: PrintTo( <file>, <obj>, <obj>... )",0,0);
    hd = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IsString(hd) )
        return Error("usage: PrintTo( <file>, <obj>, <obj>... )",0,0);

    /* try to open the given output file, raise an error if you can not    */
    if ( OpenOutput( (char*)PTR_BAG(hd) ) == 0 )
        return Error("PrintTo: can not open the file for writing",0,0);

    /* print all the arguments, take care of strings and functions         */
    for ( i = 2; i < GET_SIZE_BAG(hdCall)/SIZE_HD; ++i ) {
	Int type;
        hd = EVAL( PTR_BAG(hdCall)[i] );
	type = GET_TYPE_BAG( hd );
        if ( IsString( hd ) && GET_TYPE_BAG(hd) == T_STRING )  PrintString( hd );
        else if ( type == T_MAKEFUNC )           PrintFunction( hd );
        else if ( type == T_FUNCTION )           PrintFunction( hd );
        else if ( type == T_MAKEMETH )           PrintMethod( hd );
        else if ( type == T_METHOD )             PrintMethod( hd );
	else if ( type != T_VOID )               Print( hd );
        else                                           Pr("",0,0);
    }

    /* close the output file again, and return nothing                     */
    if ( ! CloseOutput() )
        Error("PrintTo: can not close output, this should not happen",0,0);
    return HdVoid;
}

Int            OpenStringOutput ();
Bag            ReturnStringOutput ();
Int            CloseStringOutput (void);

Bag       FunPrntToString (Bag hdCall)
{
    Bag           hd, hdList;
    Int                i;

    /* check the number and type of the arguments, nothing special         */
    if ( GET_SIZE_BAG(hdCall) == SIZE_HD )
        return Error("usage: PrintTo( <file>, <obj>, <obj>... )",0,0);
    //hd = EVAL( PTR_BAG(hdCall)[1] );

    /* try to open the given output file, raise an error if you can not    */
    if ( OpenStringOutput( ) == 0 )
        return Error("PrintTo: can not open the file for writing",0,0);

    /* print all the arguments, take care of strings and functions         */
    for ( i = 1; i < GET_SIZE_BAG(hdCall)/SIZE_HD; ++i ) {
        Int type;
        hd = EVAL( PTR_BAG(hdCall)[i] );
        type = GET_TYPE_BAG( hd );
        if ( IsString( hd ) && GET_TYPE_BAG(hd) == T_STRING )  PrintString( hd );
        else if ( type == T_MAKEFUNC )           PrintFunction( hd );
        else if ( type == T_FUNCTION )           PrintFunction( hd );
        else if ( type == T_MAKEMETH )           PrintMethod( hd );
        else if ( type == T_METHOD )             PrintMethod( hd );
        else if ( type != T_VOID )               Print( hd );
        else                                           Pr("",0,0);
    }

    hdList = ReturnStringOutput();

    /* close the output file again, and return nothing                     */
    if ( ! CloseStringOutput() )
        Error("PrintTo: can not close output, this should not happen",0,0);
    return hdList;
}

/****************************************************************************
**
*F  FunStringPrint( <hdCall> ) . . . . . . . . . .  Prints to a string
**
****************************************************************************/

Bag FunStringPrint (Bag hdCall)
{
    Obj hdRes = 0;
    
    /* try to to redirect output to a memory buffer                        */
    if ( OpenMemory() == 0 )
        return Error("StringPrint: can not redirect output to string.",0,0);

    /* Print the stuff */
    FunPrint(hdCall);
    
    /* close the output and return string                                  */
    if ( ! CloseMemory(&hdRes) )
        Error("StringPrint: can not close output, this should not happen",0,0);

    return hdRes; 
}



/****************************************************************************
**
*F  FunAppendTo( <hdCall> ) . . . . . . . . . .  internal function 'AppendTo'
**
**  'FunAppendTo' implements the internal function 'AppendTo'.
**
**  'AppendTo( <filename>, <obj1>, <obj2>... )'
**
**  'AppendTo' appends the obejcts <obj1>, <obj2>, etc. to the file with  the
**  name <filename>.  'AppendTo' works like 'PrintTo' (see "PrintTo")  except
**  that it does not truncate the file if it exists.
**
**  'AppendTo' is a procedure, i.e., does not return a value.
**
**  See the note about empty string literals and empty lists in 'Print'.
*/
Bag       FunAppendTo (Bag hdCall)
{
    Bag           hd;
    Int                i;

    /* check the number and type of the arguments, nothing special         */
    if ( GET_SIZE_BAG(hdCall) == SIZE_HD )
        return Error("usage: AppendTo( <file>, <obj>, <obj>... )",0,0);
    hd = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IsString(hd) )
        return Error("usage: AppendTo( <file>, <obj>, <obj>... )",0,0);

    /* try to open the given output file, raise an error if you can not    */
    if ( OpenAppend( (char*)PTR_BAG(hd) ) == 0 )
        return Error("AppendTo: can not open the file for appending",0,0);

    /* print all the arguments, take care of strings and functions         */
    for ( i = 2; i < GET_SIZE_BAG(hdCall)/SIZE_HD; ++i ) {
	Int type;
        hd = EVAL( PTR_BAG(hdCall)[i] );
	type = GET_TYPE_BAG( hd );
        if ( IsString( hd ) && GET_TYPE_BAG(hd) == T_STRING )  PrintString( hd );
        else if ( type == T_MAKEFUNC )           PrintFunction( hd );
        else if ( type == T_FUNCTION )           PrintFunction( hd );
	else if ( type == T_MAKEMETH )           PrintMethod( hd );
        else if ( type == T_METHOD )             PrintMethod( hd );
        else if ( type != T_VOID )               Print( hd );
        else                                           Pr("",0,0);
    }

    /* close the output file again, and return nothing                     */
    if ( ! CloseOutput() )
       Error("AppendTo: can not close output, this should not happen",0,0);
    return HdVoid;
}


/****************************************************************************
**
*F  FunLogTo( <hdCall> )  . . . . . . . . . . . . . internal function 'LogTo'
**
**  'FunLogTo' implements the internal function 'LogTo'.
**
**  'LogTo( <filename> )' \\
**  'LogTo()'
**
**  'LogTo' instructs GAP to echo all input from the  standard  input  files,
**  '*stdin*' and '*errin*' and all output  to  the  standard  output  files,
**  '*stdout*'  and  '*errout*',  to  the  file  with  the  name  <filename>.
**  The file is created if it does not  exist,  otherwise  it  is  truncated.
**
**  'LogTo' called with no argument closes the current logfile again, so that
**  input   from  '*stdin*'  and  '*errin*'  and  output  to  '*stdout*'  and
**  '*errout*' will no longer be echoed to a file.
*/
Bag       FunLogTo (Bag hdCall)
{
    Bag           hdName;

    /* 'LogTo()'                                                           */
    if ( GET_SIZE_BAG(hdCall) == SIZE_HD ) {
        if ( ! CloseLog() )
            return Error("LogTo: can not close the logfile",0,0);
    }

    /* 'LogTo( <filename> )'                                               */
    else if ( GET_SIZE_BAG(hdCall) == 2 * SIZE_HD ) {
        hdName = EVAL( PTR_BAG(hdCall)[1] );
        if ( ! IsString(hdName) )
            return Error("usage: LogTo() or LogTo( <string> )",0,0);
        if ( ! OpenLog( (char*)PTR_BAG(hdName) ) )
            return Error("LogTo: can not log to %s",(Int)PTR_BAG(hdName),0);
    }

    return HdVoid;
}


/****************************************************************************
**
*F  FunLogInputTo( <hdCall> ) . . . . . . . .  internal function 'LogInputTo'
**
**  'FunLogInputTo' implements the internal function 'LogInputTo'.
**
**  'LogInputTo( <filename> )' \\
**  'LogInputTo()'
**
**  LogInputTo'  instructs  GAP  to echo  all  input from the  standard input
**  files, '*stdin*' and  '*errin*',  to the file  with the  name <filename>.
**  The file is created if it does not exist, otherwise it is truncated.
**
**  'LogInputTo' called with no argument closes the current logfile again, so
**  that input  from '*stdin*' and '*errin*' will  no longer  be echoed  to a
**  file.
*/
Bag       FunLogInputTo (Bag hdCall)
{
    Bag           hdName;

    /* 'LogInputTo()'                                                      */
    if ( GET_SIZE_BAG(hdCall) == SIZE_HD ) {
        if ( ! CloseInputLog() )
            return Error("LogInputTo: can not close the logfile",0,0);
    }

    /* 'LogInputTo( <filename> )'                                          */
    else if ( GET_SIZE_BAG(hdCall) == 2 * SIZE_HD ) {
        hdName = EVAL( PTR_BAG(hdCall)[1] );
        if ( ! IsString(hdName) )
           return Error("usage: LogInputTo() or LogTo( <string> )",0,0);
        if ( ! OpenInputLog( (char*)PTR_BAG(hdName) ) )
           return Error("LogInputTo: cannot log to %s",(Int)PTR_BAG(hdName),0);
    }

    return HdVoid;
}




/****************************************************************************
**
*F  FunHelp( <hdCall> ) . . . . . . . . . . . . . .  internal function 'Help'
**
**  'FunHelp' implements the internal function 'Help'.
**
**  'Help( <topic> )'
**
**  'Help' prints a section from the on-line documentation about <topic>.
*/
Bag       FunHelp (Bag hdCall)
{
    Pr("Use Dir(spiral) and Dir(gap) to see a list of SPIRAL and GAP packages\n"
       "Use Dir(spiral.<pkg>) and Dir(gap.<pkg>) to see contents of <pkg>\n"
       "Use ?<func> or Doc(<func>) to learn about a function or a package\n", 0, 0);
    return HdVoid;
}


/****************************************************************************
**
*F  FunExec( <hdCall> ) . . . . . . . . . . . . . .  internal function 'Exec'
**
**  'FunExec' implements the internal function 'Exec'.
**
**  'Exec( <command> )'
**
**  'Exec' passes the string <command> to  the  command  interpreter  of  the
**  operating system.  The precise mechanismen of this is  system  dependent.
**  Also operating system dependent are the possible commands.
**
**  'Exec' is a procedure, i.e., does not return a value.
*/
Bag       FunExec (Bag hdCall)
{
    Bag           hdCmd;
    if ( GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
        return Error("usage: Exec( <command> )",0,0);
    hdCmd = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IsString(hdCmd) )
        return Error("usage: Exec( <command> )",0,0);
    SyExec( (char*)PTR_BAG(hdCmd) );
    return HdVoid;
}

/****************************************************************************
**
*F  FunIntExec( <hdCall> ) . . . . . . . . . . . .  internal function 'Exec'
**
**  'FunIntExec' implements the internal function 'IntExec'.
**
**  'IntExec( <command> )'
**
**  'IntExec' passes the string <command> to the command interpreter
**  of the operating system.  The precise mechanismen of this is
**  system dependent.  Also operating system dependent are the
**  possible commands.
**
**  'IntExec' is a function, it returns completion status of the command.
*/
Bag       FunIntExec (Bag hdCall)
{
    Bag           hdCmd;
    int status;

    if ( GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
        return Error("usage: Exec( <command> )",0,0);
    hdCmd = EVAL( PTR_BAG(hdCall)[1] );
    if ( ! IsString(hdCmd) )
        return Error("usage: Exec( <command> )",0,0);
    status = SyExec( (char*)PTR_BAG(hdCmd) );
    return INT_TO_HD( (Int)status );
}


/****************************************************************************
**
*F  FunRuntime( <hdCall> )  . . . . . . . . . . . internal function 'Runtime'
**
**  'FunRuntime' implements the internal function 'Runtime'.
**
**  'Runtime()'
**
**  'Runtime' returns the time spent since the start of GAP in  milliseconds.
**  How much time execution of statements take is of course system dependent.
**  The accuracy of this number is also system dependent.
*/
Bag       FunRuntime (Bag hdCall)
{
    if ( GET_SIZE_BAG(hdCall) != SIZE_HD )
        return Error("usage: Runtime()",0,0);
    return INT_TO_HD( SyTime() );
}


/****************************************************************************
**
*F  FunSizeScreen( <hdCall> ) . . . . . . . .  internal function 'SizeScreen'
**
**  'FunSizeScreen' implements the internal function 'SizeScreen' to  get  or
**  set the actual screen size.
**
**  'SizeScreen()'
**
**  In this form 'ScreeSize' returns the size of the screen as  a  list  with
**  two entries.  The first is the length of each line,  the  second  is  the
**  number of lines.
**
**  'SizeScreen( [ <x>, <y> ] )'
**
**  In this form 'SizeScreen' sets the size of the screen.  <x> is the length
**  of each line, <y> is the number of lines.  Either value may  be  missing,
**  to leave this value unaffected.  Note that those parameters can  also  be
**  set with the command line options '-x <x>' and '-y <y>'.
*/
Bag       FunSizeScreen (Bag hdCall)
{
    Bag           hdSize;         /* argument and result list        */
    Int                len;            /* length of lines on the screen   */
    Int                nr;             /* number of lines on the screen   */

    /* check the arguments                                                 */
    if ( GET_SIZE_BAG(hdCall) != SIZE_HD && GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
        return Error("usage: SizeScreen() or SizeScreen([<x>,<y>])",0,0);

    /* no argument is equivalent to the empty list                         */
    if ( GET_SIZE_BAG(hdCall) == SIZE_HD ) {
        hdSize = NewBag( T_LIST, SIZE_PLEN_PLIST(0) );
        SET_LEN_PLIST( hdSize, 0 );
    }

    /* otherwise check the argument                                        */
    else {
        hdSize = EVAL( PTR_BAG(hdCall)[1] );
        if ( ! IS_LIST(hdSize) || 2 < LEN_LIST(hdSize) )
          return Error("usage: SizeScreen() or SizeScreen([<x>,<y>])",0,0);
    }

    /* extract the length                                                  */
    if ( LEN_LIST(hdSize) < 1 || ELMF_LIST(hdSize,1) == 0 ) {
        len = SyNrCols;
    }
    else {
        if ( GET_TYPE_BAG( ELMF_LIST(hdSize,1) ) != T_INT )
            return Error("SizeScreen: <x> must be an integer",0,0);
        len = HD_TO_INT( ELMF_LIST(hdSize,1) );
        if ( len < 20  )  len = 20;
        if ( 256 < len )  len = 256;
    }

    /* extract the number                                                  */
    if ( LEN_LIST( hdSize ) < 2 || ELMF_LIST(hdSize,2) == 0 ) {
        nr = SyNrRows;
    }
    else {
        if ( GET_TYPE_BAG( ELMF_LIST(hdSize,2) ) != T_INT )
            return Error("SizeScreen: <y> must be an integer",0,0);
        nr = HD_TO_INT( ELMF_LIST(hdSize,2) );
        if ( nr < 10 )  nr = 10;
    }

    /* set length and number                                               */
    SyNrCols = len;
    SyNrRows = nr;

    /* make and return the size of the screen                              */
    hdSize = NewBag( T_LIST, SIZE_PLEN_PLIST(2) );
    SET_LEN_PLIST( hdSize, 2 );
    SET_ELM_PLIST( hdSize, 1, INT_TO_HD(len) );
    SET_ELM_PLIST( hdSize, 2, INT_TO_HD(nr) );
    return hdSize;
}


/****************************************************************************
**
*F  FunTmpName( <hdCall> )  . . . . . . . . . . . internal function 'TmpName'
**
**  'TmpName()' returns a file names that can safely be used for a temporary
**  file.  It returns 'false' in case of failure.
*/
Bag	FunTmpName (Bag hdCall)
{
    Bag       hdStr;
    char          * str;

    if ( GET_SIZE_BAG(hdCall) != SIZE_HD )
	return Error( "usage: TmpName()", 0, 0 );
    str = SyTmpname();
    if ( str == (char*)0 )
	return HdFalse;
    hdStr = NewBag( T_STRING, strlen(str)+1 );
    *((char*)PTR_BAG(hdStr)) = 0;
    strncat( (char*)PTR_BAG(hdStr), str, strlen(str) );
    return hdStr;
}


/****************************************************************************
**
*F  FunIsIdentical( <hdCall> )  . . . . . . . internal function 'IsIdentical'
**
**  'FunIsIdentical' implements 'IsIdentical'
*/
Bag       FunIsIdentical (Bag hdCall)
{
    Bag           hdL;
    Bag           hdR;

    if ( GET_SIZE_BAG(hdCall) != 3*SIZE_HD )
        return Error( "usage: IsIdentical( <l>, <r> )", 0, 0 );
    hdL = EVAL( PTR_BAG(hdCall)[1] );
    hdR = EVAL( PTR_BAG(hdCall)[2] );
    if ( GET_TYPE_BAG(hdL) < T_LIST && GET_TYPE_BAG(hdR) < T_LIST )
	return EQ( hdL, hdR );
    else if ( GET_TYPE_BAG(hdL) < T_LIST || GET_TYPE_BAG(hdR) < T_LIST )
	return HdFalse;
    else
	return ( hdL == hdR ) ? HdTrue : HdFalse;
}


/****************************************************************************
**
*F  FunHANDLE( <hdCall> ) . . . . . . . . . . . . .  expert function 'HANDLE'
**
**  'FunHANDLE' implements the internal function 'HANDLE'.
**
**  'HANDLE( <obj> )'
**
**  'HANDLE' returns the handle  of  the  object  <obj>  as  an  integer.  It
**  exists only for debugging purposes and should only be  used  by  experts.
*/
Bag       FunHANDLE (Bag hdCall)
{
    Bag           hdHD;
    Bag           hdObj;

    if ( GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
        return Error("usage: HANDLE( <obj> )",0,0);
    hdObj = EVAL( PTR_BAG(hdCall)[1] );
    hdHD  = INT_TO_HD( (Int)hdObj );
    if ( HD_TO_INT(hdHD) != (Int)hdObj )
        return Error("HANDLE: %d does not fit into 28 bits",(Int)hdObj,0);

    return hdHD;
}


/****************************************************************************
**
*F  FunOBJ( <hdCall> )  . . . . . . . . . . . . . . . . expert function 'OBJ'
**
**  'FunOBJ' implements the internal function 'OBJ'.
**
**  'OBJ( <int> )'
**
**  'OBJ' returns the object with the handle given by the integer  <int>.  It
**  is the inverse function to 'HD'.  Note that passing an integer  to  'OBJ'
**  which is not a valid handle is likely to crash GAP.  Thus  this  function
**  is only there for debugging purposes and should only be used by experts.
*/
Bag       FunOBJ (Bag hdCall)
{
    Bag           hdObj;
    Bag           hdHD;

    if ( GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
        return Error("usage: OBJ( <hd> )",0,0);
    hdHD = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG( hdHD ) != T_INT )
        return Error("OBJ: <hd> must be a small integer",0,0);
    hdObj = (Bag)HD_TO_INT( hdHD );

    return hdObj;
}


/****************************************************************************
**
*F  FunTYPE( <hdCall> ) . . . . . . . . . . . . . . .  expert function 'TYPE'
**
**  'FunTYPE' implements the internal function 'TYPE'.
**
**  'TYPE( <obj> )'
**
**  'TYPE' returns the type of the object <obj> as a string.
*/
Bag       FunTYPE (Bag hdCall)
{
    Bag           hdType;
    Bag           hdObj;

    if ( GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
        return Error("usage: GET_TYPE_BAG( <obj> )",0,0);
    hdObj  = EVAL( PTR_BAG(hdCall)[1] );
    if ( hdObj == 0 ) {
        hdType = NewBag( T_STRING, 5 );
        strncat( (char*)PTR_BAG(hdType), "null", 4 );
    }
    else {
		char *objtyp = InfoBags[GET_TYPE_BAG(hdObj)].name;
        hdType = NewBag( T_STRING, strlen( objtyp ) + 1 );
        strncat( (char*)PTR_BAG(hdType), objtyp,
                   strlen( objtyp ) + 1 );
    }

   return hdType;
}


/****************************************************************************
**
*F  FunSIZE( <hdCall> ) . . . . . . . . . . . . . . .  expert function 'SIZE'
**
**  'FunSIZE' implements the internal function 'SIZE'.
**
**  'SIZE( <obj> )'
**
**  'SIZE' returns the size of the object <obj> including all its subobjects.
**
**  First the  all   bags of  the object  are marked by  'MarkObj' by  adding
**  'T_ILLEGAL'  to their type.  Then 'SizeObj'   only counts marked bags and
**  unmarks them before recursing to subobjects.   This way every bag is only
**  counted once, even if it  appear several times in  the object.  This also
**  helps  to   avoid  infinite recursion if    an  object contains itself as
**  subobject.
*/
void            MarkObj (Bag hdObj)
{
    UInt       i;

    /* void and small integers do not have a handle structure              */
    if ( hdObj == 0 || GET_TYPE_BAG(hdObj) == T_INT || GET_TYPE_BAG(hdObj)==T_NAMESPACE)
        return;

    /* do not mark a bag twice                                             */
    if ( GET_FLAG_BAG(hdObj, BF_VISITED) )
        return;

    /* mark this bag                                                       */
    SET_FLAG_BAG(hdObj, BF_VISITED);

    /* mark the subobjects                                                 */
    for ( i = NrHandles( GET_TYPE_BAG(hdObj), GET_SIZE_BAG(hdObj) ); 0 < i; i-- )
        MarkObj( PTR_BAG(hdObj)[i-1] );
}

UInt   SizeObj (Bag hdObj)
{
    UInt       size;
    UInt       i;

    /* void and small integers do not use any memory at all                */
    if ( hdObj == 0 || GET_TYPE_BAG(hdObj) == T_INT )
        return NUM_TO_UINT(0);

    /* do not count unmarked bags                                          */
    if ( ! GET_FLAG_BAG(hdObj, BF_VISITED) )
        return NUM_TO_UINT(0);

    /* unmark this bag                                                     */
    CLEAR_FLAG_BAG(hdObj, BF_VISITED);

    /* start with the size of this bag                                     */
    size = GET_SIZE_BAG( hdObj );

    /* add the sizes of the subobjects                                     */
    for ( i = NrHandles( GET_TYPE_BAG(hdObj), GET_SIZE_BAG(hdObj) ); 0 < i; i-- )
        size += SizeObj( PTR_BAG(hdObj)[i-1] );

    /* return the size                                                     */
    return size;
}

Bag       FunSIZE (Bag hdCall)
{
    UInt       size;
    Bag           hdObj;

    if ( GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
        return Error("usage: GET_SIZE_BAG( <obj> )",0,0);
    hdObj  = EVAL( PTR_BAG(hdCall)[1] );
    MarkObj( hdObj );
    size = SizeObj( hdObj );

    return INT_TO_HD( size );
}


/****************************************************************************
**
*F  FunGASMAN( <hdCall> ) . . . . . . . . . . . . .  expert function 'GASMAN'
**
**  'FunGASMAN' implements the internal function 'GASMAN'
**
**  'GASMAN( "display" | "clear" | "collect" | "message" | "messageSTAT"
**           "traceON" | "traceOFF" | "traceSTAT" )'
*/
Bag       FunGASMAN (Bag hdCall)
{
    Bag     hdCmd;				        // handle of an argument
    UInt    i,  k;						// loop variables
	Bag     hdRet = HdVoid;				// return value
	char *usageMessage =
		"usage: GASMAN( \"display\"|\"clear\"|\"collect\"|\"message\"|\"messageSTAT\"|\"traceON\"|\"traceOFF\"|\"traceSTAT\" )";

    /* check the argument                                                  */
    if ( GET_SIZE_BAG(hdCall) == SIZE_HD )
        return Error(usageMessage, 0, 0);

    /* loop over the arguments                                             */
    for ( i = 1; i < GET_SIZE_BAG(hdCall)/SIZE_HD; i++ ) {

        /* evaluate and check the command                                  */
        hdCmd = EVAL( PTR_BAG(hdCall)[i] );
        if ( ! IsString(hdCmd) )
           return Error(usageMessage, 0, 0);

        /* if request display the statistics                               */
        if ( strcmp( (char*)PTR_BAG(hdCmd), "display" ) == 0 ) {
            Int sumNrLive = 0;
            Int sumSizeLive = 0;
            Int sumNrAll = 0;
            Int sumSizeAll = 0;
            Pr("\t\t    type     alive     size     total     size\n",0,0);
            for ( k = T_VOID; k < T_ILLEGAL-1; k++ ) {
                Pr("%24s  ",   (Int)InfoBags[k].name, 0 );
                sumNrLive += InfoBags[k].nrLive;
                sumSizeLive += InfoBags[k].sizeLive;
                Pr("%8dk %8dk  ",(Int)InfoBags[k].nrLive >> 10,
                               (Int)InfoBags[k].sizeLive >> 10);
                sumNrAll += InfoBags[k].nrAll;
                sumSizeAll += InfoBags[k].sizeAll;
                Pr("%8dk %8dk\n",(Int)InfoBags[k].nrAll >> 10,
                               (Int)InfoBags[k].sizeAll >> 10);
            }
            Pr("%24s  ",   (Int)"SUMMARY", 0 );
            if (sumSizeLive<1000000000) {
                Pr("%9d %9d  ",(Int)sumNrLive, (Int)sumSizeLive);
            } else {
                Pr("%8dk %8dk  ",(Int)sumNrLive >> 10, (Int)sumSizeLive >> 10);
            }
            if (sumSizeAll<1000000000) {
                Pr("%9d %9d\n",(Int)sumNrAll, (Int)sumSizeAll);
            } else {
                Pr("%8dk %8dk\n",(Int)sumNrAll >> 10, (Int)sumSizeAll >> 10);
            }
        } 

        else if ( strcmp( (char*)PTR_BAG(hdCmd), "display1" ) == 0 ) { // vvv
            Int sumNrLive = 0;
            Int sumSizeLive = 0;
            Int sumNrAll = 0;
            Int sumSizeAll = 0;
            Pr("\t\t    type     alive     size     total     size\n",0,0);
            for ( k = T_VOID; k < T_ILLEGAL-1; k++ ) {
                    if ( (InfoBags[k].sizeLive>0) && (InfoBags[k].sizeAll>0) ) {
                        Pr("%24s  ",   (Int)InfoBags[k].name, 0 );
                        sumNrLive += InfoBags[k].nrLive;
                        sumSizeLive += InfoBags[k].sizeLive;
                        if (InfoBags[k].sizeLive<1000000000)
                            Pr("%9d %9d  ",(Int)InfoBags[k].nrLive,
                               (Int)InfoBags[k].sizeLive);
                        else 
                            Pr("%8dk %8dk  ",(Int)InfoBags[k].nrLive >> 10,
                               (Int)InfoBags[k].sizeLive >> 10);
                        sumNrAll += InfoBags[k].nrAll;
                        sumSizeAll += InfoBags[k].sizeAll;
                        if (InfoBags[k].sizeAll<1000000000)
                            Pr("%9d %9d\n",(Int)InfoBags[k].nrAll,
                               (Int)InfoBags[k].sizeAll);
                        else
                            Pr("%8dk %8dk\n",(Int)InfoBags[k].nrAll >> 10,
                               (Int)InfoBags[k].sizeAll >> 10);
                }
            }
            Pr("%24s  ",   (Int)"SUMMARY", 0 );
            if (sumSizeLive<1000000000) {
                Pr("%9d %9d  ",(Int)sumNrLive, (Int)sumSizeLive);
            } else {
                Pr("%8dk %8dk  ",(Int)sumNrLive >> 10, (Int)sumSizeLive >> 10);
            }
            if (sumSizeAll<1000000000) {
                Pr("%9d %9d\n",(Int)sumNrAll, (Int)sumSizeAll);
            } else {
                Pr("%8dk %8dk\n",(Int)sumNrAll >> 10, (Int)sumSizeAll >> 10);
            }
        }

        /* if request clear the statistics                               */
        else if ( strcmp( (char*)PTR_BAG(hdCmd), "clear" ) == 0 ) {
            for ( k = T_VOID; k < T_ILLEGAL; k++ ) {
                InfoBags[k].nrAll   = InfoBags[k].nrLive;
                InfoBags[k].sizeAll = InfoBags[k].sizeLive;
            }
        }

        /* or collect the garbage                                          */
        else if ( strcmp( (char*)PTR_BAG(hdCmd), "collect" ) == 0 ) {
            float usedpc;
            CollectGarb();
			usedpc = (float)(100 * SizeLiveBags) / SizeAllArenas;
			if (SyMsgsFlagBags) {
				fprintf(stderr, "%dk live bags, %.1f%% of total Memory Arenas (%dk)\n",
						SizeLiveBags / 1024, usedpc, SizeAllArenas / 1024);
				fflush(stderr);
			}
			hdRet = INT_TO_HD( SizeLiveBags / 1024 ); // return value is size of bags in Kbytes
        }

        /* or toggle Gasman messages                               */
        else if ( strcmp( (char*)PTR_BAG(hdCmd), "message" ) == 0 ) {
			if(SyMsgsFlagBags==0)
                SyMsgsFlagBags = 2;
			else
				SyMsgsFlagBags = 0;
        }

		// get the current state (value) of GC summary message printing
		else if ( strcmp( (char*)PTR_BAG(hdCmd), "messageSTAT" ) == 0 ) {
			hdRet = INT_TO_HD (SyMsgsFlagBags);
		}

		// turn memory manager tracing messages and statistics ON
		else if ( strcmp( (char*)PTR_BAG(hdCmd), "traceON" ) == 0 ) {
			SyMemMgrTrace = 1;
		}

		// turn memory manager tracing messages and statistics OFF
		else if ( strcmp( (char*)PTR_BAG(hdCmd), "traceOFF" ) == 0 ) {
			SyMemMgrTrace = 0;
		}

		// get the current state (value) of Memory Manager tracing
		else if ( strcmp( (char *)PTR_BAG(hdCmd), "traceSTAT" ) == 0 ) {
			hdRet = INT_TO_HD (SyMemMgrTrace);
		}

        /* otherwise complain                                              */
        else {
           return Error(usageMessage, 0, 0);
        }
    }

    /* return nothing, this function is a procedure                        */
    return hdRet;
}


/****************************************************************************
**
*F  FunCoefficients( <hdCall> ) . . . . . .  internal function 'Coefficients'
**
**  'FunCoefficients' implements the internal function 'Coefficients'.
**
**  'Coefficients( <list>, <number> )'
**
*N  15-Jan-91 martin this function should not be here
*N  15-Jan-91 martin this function should not be called 'Coefficients'
*/
Bag       FunCoefficients (Bag hdCall)
{
    Int                pos, num, val;
    Bag           hdRes, hdList, hdInt;


    if ( GET_SIZE_BAG( hdCall ) != 3 * SIZE_HD )
        return Error("usage: Coefficients( <list>, <int> )",0,0);

    hdList = EVAL( PTR_BAG(hdCall)[1] );
    hdInt  = EVAL( PTR_BAG(hdCall)[2] );
    if ( ! IS_LIST(hdList) || GET_TYPE_BAG(hdInt) != T_INT)
        return Error("usage: Coefficients( <list>, <int> )",0,0);

    pos   = LEN_LIST( hdList );
    hdRes = NewBag( T_LIST, SIZE_PLEN_PLIST( pos ) );
    SET_LEN_PLIST( hdRes, pos );

    num = HD_TO_INT( hdInt );
    if ( num < 0 )
        return Error("Coefficients: <int> must be non negative",0,0);

    while ( 0 < num && 0 < pos ) {
        hdInt = ELMF_LIST( hdList, pos );
        if ( hdInt == 0 || GET_TYPE_BAG( hdInt ) != T_INT )
          return Error("Coefficients: <list>[%d] must be a positive integer",
                       (Int)pos,0);
        val = HD_TO_INT(hdInt);
        if ( val <= 0 )
          return Error("Coefficients: <list>[%d] must be a positive integer",
                        (Int)pos,0);
        SET_ELM_PLIST( hdRes, pos, INT_TO_HD( num % val ) );
        pos--;
        num /= val;
    }

    while ( 0 < pos ) {
        SET_ELM_PLIST( hdRes, pos, INT_TO_HD( 0 ) );
        pos--;
    }

    return hdRes;
}


/****************************************************************************
**
*F  FunNUMBERHANDLES( <hdCall> )  . . . .  internal function 'NUMBER_HANDLES'
**
**  'FunNUMBERHANDLES' implements the internal function 'NUMBER_HANDLES'.
**
**  'NUMBER_HANDLES( <type> )'
*/
Bag       FunNUMBERHANDLES (Bag hdCall)
{
    Int                typ;
    Bag           hdTyp;


    if ( GET_SIZE_BAG( hdCall ) != 2 * SIZE_HD )
        return Error("usage: NUMBER_HANDLES( <type> )",0,0);

    hdTyp = EVAL( PTR_BAG(hdCall)[1] );
    if (GET_TYPE_BAG(hdTyp) != T_INT)
        return Error("usage: NUMBER_HANDLES( <type> )",0,0);

    typ = HD_TO_INT( hdTyp );
    if (typ < 0 || typ >= T_ILLEGAL)
        return Error("NUMBER_HANDLES: <type> must lie in [%d,%d]",
                     0,(Int)(T_ILLEGAL-1));

    return INT_TO_HD( InfoBags[typ].nrAll );
}


/****************************************************************************
**
*F  FunSIZEHANDLES( <hdCall> )  . . . . . .  internal function 'SIZE_HANDLES'
**
**  'FunSIZEHANDLES' implements the internal function 'SIZE_HANDLES'.
**
**  'SIZE_HANDLES( <type> )'
*/
Bag       FunSIZEHANDLES (Bag hdCall)
{
    Int                typ;
    Bag           hdTyp;


    if ( GET_SIZE_BAG( hdCall ) != 2 * SIZE_HD )
        return Error("usage: SIZE_HANDLES( <type> )",0,0);

    hdTyp = EVAL( PTR_BAG(hdCall)[1] );
    if (GET_TYPE_BAG(hdTyp) != T_INT)
        return Error("usage: SIZE_HANDLES( <type> )",0,0);

    typ = HD_TO_INT( hdTyp );
    if (typ < 0 || typ >= T_ILLEGAL)
        return Error("SIZE_HANDLES: <type> must lie in [%d,%d]",
                     0,(Int)(T_ILLEGAL-1));

    return INT_TO_HD( InfoBags[typ].sizeAll );
}

Bag       FunWeakRef(Bag hdCall)
{
    Bag           hd;

    if ( GET_SIZE_BAG( hdCall ) != 2 * SIZE_HD )
        return Error("usage: WeakRef( <obj> )",0,0);

    hd = EVAL( PTR_BAG(hdCall)[1] );
    return hd;
}

Bag     FunTabToList(Bag hdCall) {
    Bag hd;
    if ( GET_SIZE_BAG( hdCall ) != 2 * SIZE_HD )
        return Error("usage: TabToList( <table> )",0,0);
    hd = EVAL( PTR_BAG(hdCall)[1] );
    if (GET_TYPE_BAG(hd) != T_NAMESPACE)
        return Error("usage: TabToList( <table> )",0,0);
    return TableToList(hd);
}


/****************************************************************************
**
*F  InitGap( <pargc>, <pargv> ) . . . . . . . . . . . . . . . initializes GAP
**
**  'InitGap' initializes GAP.
*/
void            InitGap (int argc, char** argv, int* stackBase) {
    Bag           hd;
    Int                i;
    Int                ignore;
    char *              version;
    exc_type_t          e;
    char*		prompt;
    /* Initialize all subpackages of GAP.                                  */

#ifdef DEBUG
#ifndef WIN32
	mtrace();							/* trace memory calls */
#endif				// !WIN32
#endif				// DEBUG
	
    InitSystem( argc, argv );
    InitScanner();
    InitGasman(argc, argv, stackBase);
    InitIdents();
    InitCommentBuffer();
    InitEval();
    InitDebug();
    InitSPIRAL();
    InitHooks();
    InitNamespaces();

    /* create the variables last, last2, last3                             */
    HdLast  = FindIdent( "last"  );
    HdLast2 = FindIdent( "last2" );
    HdLast3 = FindIdent( "last3" );
    HdTime  = FindIdent( "time"  );
    
    InitGlobalBag(&HdDbgStackRoot, "HdDbgStackRoot");
    
    hd = FindIdent( "LIBNAME" );
    SET_BAG(hd, 0,  NewBag( T_STRING, (UInt)(strlen(SyLibname)+1) ) );
    strncat( (char*)PTR_BAG(PTR_BAG(hd)[0]), SyLibname, strlen(SyLibname) );

    hd = FindIdent( "QUIET" );
    if ( SyQuiet )  SET_BAG(hd, 0,  HdTrue );
    else            SET_BAG(hd, 0,  HdFalse );

    hd = FindIdent( "BANNER" );
    if ( SyBanner )  SET_BAG(hd, 0,  HdTrue );
    else             SET_BAG(hd, 0,  HdFalse );

    /**/ GlobalPackage2("gap", "gap"); /**/
    /* install all internal function from this package                     */
    InstIntFunc( "Ignore",     FunIgnore     );
    InstIntFunc( "Error",      FunError      );
    InstIntFunc( "Backtrace",  FunBacktrace  );
    InstIntFunc( "Backtrace2", FunBacktrace2 );
    InstIntFunc( "BacktraceTo",FunBacktraceTo);
    InstIntFunc( "WindowCmd",  FunWindowCmd  );

    InstIntFunc( "READ",       FunREAD       );
    InstIntFunc( "READSTR",    FunReadString );
    InstIntFunc( "CHANGEDIR",  FunChangeDir  );
    InstIntFunc( "AUTO",       FunAUTO       );
    InstIntFunc( "Print",      FunPrint      );
    InstIntFunc( "PrintToString",    FunPrntToString     );
    InstIntFunc( "_Pr",        Fun_Pr        );
    InstIntFunc( "PrintTo",    FunPrntTo     );
    InstIntFunc( "StringPrint",FunStringPrint);
    InstIntFunc( "AppendTo",   FunAppendTo   );
    InstIntFunc( "LogTo",      FunLogTo      );
    InstIntFunc( "LogInputTo", FunLogInputTo );

    InstIntFunc( "Help",        FunHelp        );
    InstIntFunc( "Exec",        FunExec        );
    InstIntFunc( "IntExec",     FunIntExec     );
    InstIntFunc( "Runtime",     FunRuntime     );
    InstIntFunc( "SizeScreen",  FunSizeScreen  );
    InstIntFunc( "TmpName",     FunTmpName     );
    InstIntFunc( "IsIdentical", FunIsIdentical );
    InstIntFunc( "HANDLE",      FunHANDLE      );
    InstIntFunc( "OBJ",         FunOBJ         );
    InstIntFunc( "TYPE",        FunTYPE        );
    InstIntFunc( "SIZE",        FunSIZE        );
    InstIntFunc( "GASMAN",      FunGASMAN      );

    InstIntFunc( "NUMBER_HANDLES",   FunNUMBERHANDLES );
    InstIntFunc( "SIZE_HANDLES",     FunSIZEHANDLES   );

    InstIntFunc( "WeakRef",     FunWeakRef   );
    InstIntFunc( "TabToList",   FunTabToList );
    /*N  15-Jan-91 martin this function should not be here                 */
    InstIntFunc( "CoefficientsInt", FunCoefficients );

	InitMemMgrFuncs();
	
    /**/ EndPackage(); /**/

    /* read all init files, stop doing so after quiting from Error         */
    Try {
        for ( i=0; i<sizeof(SyInitfiles)/sizeof(SyInitfiles[0]); ++i ) {
	    char * file = SyInitfiles[i];
	    Obj pkg;
            if ( file[0] != '\0' ) {
		/*Pr("Reading %s...\n", file, 0);*/
                if ( OpenInput( file ) ) {
                    while ( Symbol != S_EOF ) {
                        hd = ReadIt();
                        if ( hd != 0 )  hd = EVAL( hd );
                        if ( hd == HdReturn && PTR_BAG(hd)[0] != HdReturn )
                            Error("Read: 'return' must not be used",0,0);
                        else if ( hd == HdReturn )
                             Error("Read: 'quit' must not be used",0,0);
		    }		    
		    pkg = Input->package;
                    ignore = CloseInput();
		    if(pkg != 0) /* pkg is 0 if no variables were defined */
		      PushNamespace(pkg);
                }
                else {
                    Error("can't read from \"%s\"",(Int)file,0);
                }
            }
        }
        /* read prompts from GAPInfo.prompts */
	hd = GetPromptString(PROMPT_FIELD_DBG);
	if (hd) {
	    prompt = HD_TO_STRING(hd);
	    if (strlen(prompt)<sizeof(DbgPrompt))
		strncpy(DbgPrompt, prompt, strlen(prompt)+1);
	}
	hd = GetPromptString(PROMPT_FIELD_BRK);
	if (hd) {
	    prompt = HD_TO_STRING(hd);
	    if (strlen(prompt)<sizeof(BrkPrompt))
		strncpy(BrkPrompt, prompt, strlen(prompt)+1);
	}
        /* read prompts from environment */
	prompt = getenv("SPIRAL_DBG_PROMPT");
	if (prompt != NULL && strlen(prompt)<sizeof(DbgPrompt)) {
	    strncpy(DbgPrompt, prompt, strlen(prompt)+1);
	}
	prompt = getenv("SPIRAL_BRK_PROMPT");
	if (prompt != NULL && strlen(prompt)<sizeof(BrkPrompt)) {
	    strncpy(BrkPrompt, prompt, strlen(prompt)+1);
	}

    }
    Catch(e) {
      /* exceptions raised using Error() are already printed at this point */
      if(e!=ERR_GAP) {
	while ( HdExec != 0 )  ChangeEnv( PTR_BAG(HdExec)[4], CEF_CLEANUP );
	while ( EvalStackTop>0 ) EVAL_STACK_POP;
	exc_show();
      }
    }

}



