#include "system.h"
/* #include "types.h" */
/* #include        "flags.h" */			// defns & decls moved to memmgr.h
#include "memmgr.h"
#include "eval.h"
#include "integer.h"
#include "integer4.h"
#include "args.h"
#include <sys/stat.h>
#include <string.h>
#ifdef WIN32
#include <windows.h>
#include <io.h>
#endif

/* does not return consistent results yet :(( */
UInt  _InternalHash ( Obj hd ) {
    if(!hd || GET_TYPE_BAG(hd)==T_NAMESPACE) return 0;
    else if(IS_INTOBJ(hd)) return HD_TO_INT(hd);
    else if(GET_FLAG_BAG(hd, BF_VISITED)) return 0;
    else if(GET_TYPE_BAG(hd)==T_VAR) return (Int) hd;   /* address of identifier */
    else if(GET_TYPE_BAG(hd)==T_VARAUTO) return (Int) EVAL(hd);

    {
	UInt result = 0;
	UInt i;
	UInt nhandles = NrHandles(GET_TYPE_BAG(hd), GET_SIZE_BAG(hd));
	/* don't mark variables, there is possibility of mark staying forever */
	if ( nhandles > 0 )
	  SET_FLAG_BAG(hd, BF_VISITED);    
	
	for (i = 0; i < nhandles; ++i )
	    result += _InternalHash(PTR_BAG(hd)[i]);

	for (i = nhandles * SIZE_HD; i < GET_SIZE_BAG(hd); ++i)
	    result += (UInt) (((unsigned char*)PTR_BAG(hd))[i]);

	return result;
    }
}

UInt  InternalHash ( Obj hd ) {
    UInt res = _InternalHash(hd);
    RecursiveClearFlag(hd, BF_VISITED);
    return res;
}

Obj  FunInternalHash ( Obj hdCall ) {
    char * usage = "usage: InternalHash( <obj> )";
    Obj hd;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  
		return Error(usage, 0,0);
    hd = EVAL(PTR_BAG(hdCall)[1]); 
    return INT_TO_HD(InternalHash(hd));
}


