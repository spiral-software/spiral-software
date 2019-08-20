/* SPIRAL FFT package */
#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic memory manager          */
#include        "scanner.h"             /* reading of tokens and printing  */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "list.h"                /* generic list package            */
#include        "plist.h"               /* 'LEN_PLIST', 'SET_LEN_PLIST',.. */

#include        "spiral_fft.h"          /* declaration part of the package */
#include        "fft.h"             /* Sebastian Egner's FFT package   */
#include        "double.h"
#include        "complex.h"
#include        "namespaces.h"
#ifndef NULL
#define NULL 0
#endif

void  _ToGapCplx ( Obj hdResult, Obj dlist ) {
    int i;
    for(i=0; i < LEN_LIST(hdResult); i++) {
        double re = ((double*) PTR_BAG(dlist))[2*i];
        double im = ((double*) PTR_BAG(dlist))[2*i+1];
	SET_BAG(hdResult, i+1,  ObjCplx(re, im) ); 
    }
}

/* returns HdVoid or Error */
Obj  _ToCDoubles ( Obj hdList, Obj dlist ) {
    int i;
    Obj elem = 0;
    for(i=0; i < LEN_LIST(hdList); i++) {
	elem = ELMF_LIST(hdList, i+1);
	elem = CplxAny(elem);
	((double*) PTR_BAG(dlist))[2*i]   = RE(elem);
	((double*) PTR_BAG(dlist))[2*i+1] = IM(elem);
    }
    /* return HdVoid if everything went Ok                                 */
    return HdVoid;
}

/* Empty lists are returned unchanged */
Obj  FunComplexFFT ( Obj hdCall ) {
    Obj     hdList, result, data; 
    int     len;
    fft_t * FFT;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD)
        return Error("Usage: ComplexFFT ( <compelx_list> )",0,0);

    hdList = EVAL( PTR_BAG(hdCall)[1] );

    if ( hdList == HdVoid )
        return Error("ComplexFFT: function must return a value",0,0);

    if ( ! IS_LIST(hdList) )
        return Error("usage: ComplexFFT( <complex_list> )",0,0);

    len = LEN_LIST(hdList);
   
    if ( len == 0 ) return hdList;
    
    /* Allocate space for double data using GASMAN, we can't use xmalloc   */
    /* since it prevents SyGetmem from getting adjacent blocks of memory   */
    data = NewBag(T_STRING, 2 * len * sizeof(double) + SIZE_HD);

    /* each element must be IntPair, which converts to a double            */
    _ToCDoubles(hdList, data); 

    /* Perform FFT (S. Egner's FFT package does it in-place)               */
    /*alloc*/ FFT = fft_new_prettyGood(len); 
    if(FFT==NULL) {
      /*free*/ fft_delete(FFT);
	return Error("ComplexFFT: could not initialize FFT", 0, 0);
    }
    fft_apply(1, FFT, 1, (double*) PTR_BAG(data));
    /*free*/ fft_delete(FFT);

    /* Convert double data back to IntPair's and return the result         */
    /* No need to deallocate 'data', GASMAN does garbage collection        */
    result = Copy(hdList);
    _ToGapCplx(result, data); /* hdList is just for copying   */
    
    return result;
}

/****************************************************************************
**
*F  InitSPIRAL_FFT() . . . . . . . . . . . . . initializes SPIRAL FFT package
**
**  'InitSPIRAL_FFT' initializes the SPIRAL FFT package that allows to compu-
**  te floating-point FFT from within SPIRAL. 
*/
void            InitSPIRAL_FFT (void)
{
    GlobalPackage2("spiral", "fft");
    InstIntFunc( "ComplexFFT",         FunComplexFFT);
    EndPackage();
}

