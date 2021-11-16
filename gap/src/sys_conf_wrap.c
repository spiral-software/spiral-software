




/* Implementation : GAP */

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#include "system.h"
#include "memmgr.h"
#include "integer.h"
#include        "objects.h"
#include		"string4.h"
#include "eval.h"
#include "idents.h"
#include "spiral.h"
#include "plist.h"
#include "args.h"
#include "double.h"
#include "GapUtils.h"







static char * _usage_sys_exists = "sys_exists (const char *fname)";
Obj _wrap_sys_exists(Obj argv) {
    char * usage = _usage_sys_exists;
    int  argc = GET_SIZE_BAG(argv) / SIZE_HD;
    Obj  result;

    int  _result;
    char * _arg0;

    /* Check number of arguments */ 
    if ((argc < 2) || (argc > 2)) 
        return Error("Wrong number of arguments.\nUsage: %s", (Int)usage, 0);
    /* Get argument0 */ 
    {	
	_arg0 = (char *) HdToString(ELM_ARGLIST(argv, 1), 
	   "<fname> must be a String.\nUsage: %s", (Int)usage, 0); 
    }

    /*===== Call the C function =====*/ 
    _result = (int )sys_exists(_arg0);
    /*===============================*/ 

    /* Convert the result */ 
    {
        result = IntToHd(_result); 
    }
    return result;
}


static char * _usage_sys_mkdir = "sys_mkdir (const char *name)";
Obj _wrap_sys_mkdir(Obj argv) {
    char * usage = _usage_sys_mkdir;
    int  argc = GET_SIZE_BAG(argv) / SIZE_HD;
    Obj  result;

    int  _result;
    char * _arg0;

    /* Check number of arguments */ 
    if ((argc < 2) || (argc > 2)) 
        return Error("Wrong number of arguments.\nUsage: %s", (Int)usage, 0);
    /* Get argument0 */ 
    {	
	_arg0 = (char *) HdToString(ELM_ARGLIST(argv, 1), 
	   "<name> must be a String.\nUsage: %s", (Int)usage, 0); 
    }

    /*===== Call the C function =====*/ 
    _result = (int )sys_mkdir(_arg0);
    /*===============================*/ 

    /* Convert the result */ 
    {
        result = IntToHd(_result); 
    }
    return result;
}

static char * _usage_sys_rm = "sys_rm (const char *name)";
Obj _wrap_sys_rm(Obj argv) {
    char * usage = _usage_sys_rm;
    int  argc = GET_SIZE_BAG(argv) / SIZE_HD;
    Obj  result;

    int  _result;
    char * _arg0;

    /* Check number of arguments */ 
    if ((argc < 2) || (argc > 2)) 
        return Error("Wrong number of arguments.\nUsage: %s", (Int)usage, 0);
    /* Get argument0 */ 
    {	
	_arg0 = (char *) HdToString(ELM_ARGLIST(argv, 1), 
	   "<name> must be a String.\nUsage: %s", (Int)usage, 0); 
    }

    /*===== Call the C function =====*/ 
    _result = (int )sys_rm(_arg0);
    /*===============================*/ 

    /* Convert the result */ 
    {
        result = IntToHd(_result); 
    }
    return result;
}




void Init_sys_conf(void) {

    InstIntFunc("sys_exists", _wrap_sys_exists);
    InstIntFunc("sys_mkdir", _wrap_sys_mkdir);
    InstIntFunc("sys_rm", _wrap_sys_rm);


    
}

