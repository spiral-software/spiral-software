#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <errno.h>

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "scanner.h"             /* reading of single tokens        */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */

#include        "idents.h"              /* 'InitIdents', 'FindIdent'       */
#include        "read.h"                /* 'ReadIt'                        */

#include        "list.h"                /* generic list package            */
#include        "plist.h"               /* 'LEN_PLIST', 'SET_LEN_PLIST',.. */
#include        "gstring.h"              /* 'IsString', 'PrintString'       */

#include        "statemen.h"            /* 'HdStat', 'StrStat'             */
#include        "function.h"            /* 'HdExec', 'ChangeEnv', 'PrintF' */
#include        "record.h"              /* 'HdCall*', 'HdTilde'            */

#include        "spiral.h"              /* InitSPIRAL, Try, Catch, exc     */
/* #include        "flags.h" */			// defns & decls moved to memmgr.h
#include        "args.h"

#include        "iface.h"
#include		"GapUtils.h"			/* Gap Utilities */


extern Bag            HdStack;
extern UInt        TopStack;

int GAP_SILENT = 1;

extern Bag       HdLast, HdLast2, HdLast3;
extern Bag       HdTime;

typedef void handler_t(int);

int execute()
{
  Bag		    hd;
  UInt     start;
  exc_type_t        e;
  char*		    prompt;
  static char	    spiralPrompt[80] = "spiral> ";

  if ( Symbol==S_EOF ){
    //interface_read_output_nolist(output);
    return EXEC_QUIT;
  }

  /* read prompts from GAPInfo.prompts */
  Try {
    hd = GetPromptString(PROMPT_FIELD_SPIRAL);
  } Catch(e) {
    hd = 0;
  }
  if (hd) {
    prompt = HD_TO_STRING(hd);
    if (strlen(prompt)<sizeof(spiralPrompt))
	strncpy(spiralPrompt, prompt, strlen(prompt)+1);
  }
  /* read prompts from environment */
  prompt = getenv("SPIRAL_PROMPT");
  if (prompt != NULL && strlen(prompt)<sizeof(spiralPrompt)) {
     strncpy(spiralPrompt, prompt, strlen(prompt)+1);
  }
	
  /* repeat the read-eval-print cycle until end of input                 */
  Try {
    /* read an expression                                              */
    Prompt = spiralPrompt;
    NrError = 0;
    
    hd = ReadIt();
    
    if ( hd != 0 ) {
      SyIsIntr();
      start = SyTime();
      hd = EVAL( hd );
      if ( hd == HdReturn && PTR_BAG(hd)[0] != HdReturn )
	    Error("'return' must not be used in main loop",0,0);
      else if ( hd == HdReturn ) {
	    hd = HdVoid;
	    Symbol = S_EOF;
	    return EXEC_QUIT;
      }
      SET_BAG(HdTime, 0,  INT_TO_HD( SyTime() - start ) );
      
      /* assign the value to 'last' and then print it                */
      if ( hd != 0 && GET_TYPE_BAG(hd) != T_VOID ) {
    	SET_BAG(HdLast3, 0,  PTR_BAG(HdLast2)[0] );
        SET_BAG(HdLast2, 0,  PTR_BAG(HdLast)[0] );
	    SET_BAG(HdLast, 0,  hd );
	    if ( ! GAP_SILENT ) {
	        IsString( hd );
	        //Print( hd );
            PrintObj(stdout_stream, hd, 0);
	        //Pr("\n",0,0);
            SyFmtPrint(stdout_stream, "\n");
	    }
      }
    }
  }
  Catch(e) {
    /* exceptions raised using Error() are already printed at this point */
    if (e!=ERR_GAP) {
      exc_show();
      while ( HdExec != 0 )  ChangeEnv( PTR_BAG(HdExec)[4], CEF_CLEANUP );
      while ( EvalStackTop > 0 ) EVAL_STACK_POP;
 
      return EXEC_ERROR;
    }
  }
  return EXEC_SUCCESS;
}

