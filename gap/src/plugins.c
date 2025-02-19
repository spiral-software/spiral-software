
//  Copyright (c) 2025, Carnegie Mellon University
//  See LICENSE for details

//
// plugins.c -- support for loading plugins
//

#include <stdlib.h>
#include <stdio.h>

#ifdef WIN32
    #include "win_dlfcn.h"
#else
    #include <dlfcn.h>
#endif


// for GAP to C functions
#include "system.h"
#include "memmgr.h" 
#include "integer.h"
#include "args.h"
#include "eval.h"


typedef void* (*lookup_func)(char *);
typedef int (*init_func)(lookup_func);

void* LookupGlobalName(char* name) {
    if (strcmp(name, "InstIntFunc") == 0) {
        return (void*)InstIntFunc;
    }
    else if (strcmp(name, "HdVoid") == 0) {
        return (void*)HdVoid;
    }
    else {
        return (void*)0;
    }
}


Obj FunLoadPlugin(Obj hdCall) {
    char * usage = "usage: LoadPlugin(<plugin>)";
    Obj  hd1;
    char* libname;
    char* funcname = "init_plugin";
    void *handle;
    void *funcptr;
    int init_ret;
    
    if (GET_SIZE_BAG(hdCall) != 2 * SIZE_HD) {
        return Error(usage, 0, 0);
    }
    hd1 = EVAL(PTR_BAG(hdCall)[1]);
    
    libname = HdToString(hd1, "<plugin> must be a String.\n%s", usage, 0);
    
    handle = dlopen(libname, RTLD_LAZY);
    if (handle == 0) {
        return Error("cannot open plugin %s", libname, 0);
    }
    
    funcptr = dlsym(handle, funcname);
    if (funcptr == 0) {
        return Error("cannot find function %s in library %s", funcname, libname);
    }
    
    init_ret = ((init_func)funcptr)(LookupGlobalName);
    if (init_ret != 0) {
        return Error("unable to initialize plugin %s", libname, 0);
    }
    
    return HdVoid;
}


void Init_Plugins() {
    InstIntFunc("LoadPlugin", FunLoadPlugin);
}