/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */

#include        <math.h>
#include        <stdio.h>
#include        <stdlib.h>
#include        <string.h>

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "scanner.h"             /* reading of tokens and printing  */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* HD_TO_INT / INT_TO_HD           */
#include        "read.h"                /* ReadIt()                        */
#include        "idents.h"
#include        "args.h"
#include        "namespaces.h"

#include		"GapUtils.h"

Bag CheckInputType(int input_type)
{
	if (input_type < 0 || input_type >= INPUT_LAST) {
		return Error("Bad input type %d, must be in range [0,%d]",
			input_type, INPUT_LAST - 1);
	}
	else {
		return HdVoid;
	}
}



/****************************************************************************
**
*F  ReadVal( <fname> ) . . . . . . . . .  read a single GAP value from a file
**
*/

extern void yypush_new_buffer_state();
extern void yypop_buffer_state();

int isReadValFromFile;
int addEndOfLineOnlyOnce;

Bag ReadValFromFile(char* fname)
{
	Bag hdResult;
	/* Read and evaluate test output from temporary file                   */
	isReadValFromFile = 1;
	addEndOfLineOnlyOnce = 1;
	int res = OpenInput(fname);
	if (res == 0) {
		return HdFalse;
	}
	yypush_new_buffer_state();
	hdResult = ReadIt();
	if (hdResult != NULL) {
		hdResult = EVAL(hdResult);
	}
	if (hdResult == NULL || hdResult == HdVoid) {
		hdResult = HdFalse;
	}
	yypop_buffer_state();
	CloseInput();
	isReadValFromFile = 0;
	addEndOfLineOnlyOnce = 0;
	return hdResult;
}

Bag FunReadVal(Bag hdCall)
{
	char* usage = "usage: ReadVal( <fname> )";
	Bag hdFname;
	char* fname;

	/* get and check the argument                                          */
	if (GET_SIZE_BAG(hdCall) != 2 * SIZE_HD) return Error(usage, 0, 0);
	hdFname = EVAL(PTR_BAG(hdCall)[1]);
	if (GET_TYPE_BAG(hdFname) != T_STRING) 
		return Error(usage, 0, 0);

	fname = (char*)PTR_BAG(hdFname);
	GuSysCheckExists(fname);
	return ReadValFromFile(fname);
}


/****************************************************************************
**
*F  InitSPIRAL_SPLProg() . . . . . . . . .
**
**  'InitSPIRAL_SPLProg' initializes
*/
void InitSPIRAL_SPLProg(void) {
	GlobalPackage2("spiral", "splprog");
	InstIntFunc("ReadVal", FunReadVal);
	EndPackage();
}
