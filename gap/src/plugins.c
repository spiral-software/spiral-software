
//  Copyright (c) 2025, Carnegie Mellon University
//  See LICENSE for details

//
// plugins.c -- support for loading plugins
//

#include <stdlib.h>
#include <stdio.h>

#include		"GapUtils.h"

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
#include "gstring.h"

extern Bag EvalString(char *str);
extern const char* LastEVErrorString();

typedef void* (*lookup_func)(char *);
typedef int (*init_func)(lookup_func);


UInt GetArgCount(Bag argv)
{
    return GET_SIZE_BAG(argv) / SIZE_HD;
}


Bag GetArg(Bag argv, UInt n)
{
    if ((n < 0) || (n >= GetArgCount(argv))) {
        return HdVoid;
    }
    return PTR_BAG(argv)[n];
}


void* LookupGlobalName(char *name) {
    if (strcmp(name, "InstIntFunc") == 0) {
        return (void*)InstIntFunc;
	}
	else if (strcmp(name, "Error") == 0) {
		return (void*)Error;
	}
    else if (strcmp(name, "HdVoid") == 0) {
        return (void*)HdVoid;
    }	
    else if (strcmp(name, "EVAL") == 0) {
        return (void*)EVAL;
    }	
    else if (strcmp(name, "GetArgCount") == 0) {
        return (void*)GetArgCount;
    }
    else if (strcmp(name, "GetArg") == 0) {
        return (void*)GetArg;
    }
	else if (strcmp(name, "IsString") == 0) {
		return (void*)IsString;
	}
	else if (strcmp(name, "GET_TYPE_BAG") == 0) {
		return (void*)GET_TYPE_BAG;
	}
	else if (strcmp(name, "HdToString") == 0) {
		return (void*)HdToString;
    }
	else if (strcmp(name, "HdToInt") == 0) {
		return (void*)HdToInt;
    }
	else if (strcmp(name, "StringToHd") == 0) {
		return (void*)StringToHd;
    }
    else if (strcmp(name, "EvalString") == 0) {
		return (void*)EvalString;
    }     
    else if (strcmp(name, "LastEVErrorString") == 0) {
		return (void*)LastEVErrorString;
    }
    else {
        return (void*)0;
    }
}


Obj LoadPlugin(char *libname) {
    char *funcname = "init_plugin";
    void* handle;
    void* funcptr;
    int init_ret;
    
    handle = dlopen(libname, RTLD_LAZY);
    
    #ifndef WIN32
    // try appending ".so" to library name, Windows automatically adds ".dll"
    if ((handle == 0) && (strstr(libname, ".so") == 0)) {
        int newlen = strlen(libname) + 5;
        char *libname2 = malloc(newlen);
        strcpy(libname2, libname);
        strcat(libname2, ".so");
        handle = dlopen(libname2, RTLD_LAZY);
        free(libname2);
    }
    #endif
    
    if (handle == 0) {
		#ifdef WIN32
        return Error("cannot open plugin %s", libname, 0);
		#else
		return Error("%s", dlerror(), 0);	
		#endif
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


Obj FunLoadPlugin(Obj hdCall) {
    char *usage = "usage: LoadPlugin(<plugin>)";
    Obj  hd1;
    char *libname;
    
    if (GET_SIZE_BAG(hdCall) != 2 * SIZE_HD) {
        return Error(usage, 0, 0);
    }
    hd1 = EVAL(PTR_BAG(hdCall)[1]);
    
    libname = HdToString(hd1, "<plugin> must be a String.\n%s", usage, 0);
    
    return LoadPlugin(libname);
}


void LoadOptionalPlugins(int argc, char **argv) {
    // assume ill-formed arguments caught already by InitSystem()
    char *libname;
    exc_type_t e;
    while (argc > 1) {
        if ((argv[1][0] == '-') && (argv[1][1] == 'p')) {
            argc--;
            argv++;
            if (argc > 1) {
                libname = argv[1];
                Try {
                    LoadPlugin(libname);
                }
                Catch(e) {
                    SyExit(1);
                }
            } 
            else {
                Error("usage: -p <plugin>", 0, 0);
                SyExit(1);
            }
        }
        argc--;
        argv++;
    }
}


void Init_Plugins() {
    InstIntFunc("LoadPlugin", FunLoadPlugin);
}