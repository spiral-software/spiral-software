/* use FindGlobals.sh to generate most of this file, however 
   there are portions outlined below that need to be added by hand */

#include "system.h"
#include "memmgr.h"
#include "scanner.h"
#include "GapUtils.h"

extern Bag HdBases;
extern Bag HdCall1;
extern Bag HdCall2;
extern Bag HdCallComm;
extern Bag HdCallDiff;
extern Bag HdCallEq;
extern Bag HdCallIn;
extern Bag HdCallLt;
extern Bag HdCallMod;
extern Bag HdCallOop1;
extern Bag HdCallOop2;
extern Bag HdCallPow;
extern Bag HdCallPrint;
extern Bag HdCallProd;
extern Bag HdCallQuo;
extern Bag HdCallSum;
extern Bag HdCallSumAgWord;
extern Bag HdCPC;
extern Bag HdCPL;
extern Bag HdCPS;
extern Bag HdCurLHS;
extern Bag HdCycLastE;
extern Bag HdExec;
extern Bag HdFalse;
extern Bag HdFields;
extern Bag HdIdenttab;
extern Bag HdIdWord;
extern Bag HdIntFFEs;
extern Bag HdLast;
extern Bag HdLast2;
extern Bag HdLast3;
extern Bag HdLastIntFFE;
extern Bag HdLocal;
extern Bag HdPerm;
extern Bag HdPkgRecname;
extern Bag HdRectab;
extern Bag HdResult;
extern Bag HdReturn;
extern Bag HdRnAvec;
extern Bag HdRnCentralWeight;
extern Bag HdRnComm;
extern Bag HdRnDepth;
extern Bag HdRnDiff;
extern Bag HdRnDifferenceAgWord;
extern Bag HdRnEq;
extern Bag HdRnExponentAgWord;
extern Bag HdRnExponentsAgWord;
extern Bag HdRnIn;
extern Bag HdRnInformationAgWord;
extern Bag HdRnIsAgWord;
extern Bag HdRnIsCompatibleAgWord;
extern Bag HdRnLeadingExponent;
extern Bag HdRnLt;
extern Bag HdRnMod;
extern Bag HdRnNormalizeIgs;
extern Bag HdRnOp;
extern Bag HdRnPow;
extern Bag HdRnPrint;
extern Bag HdRnProd;
extern Bag HdRnQuo;
extern Bag HdRnReducedAgWord;
extern Bag HdRnRelativeOrder;
extern Bag HdRnSum;
extern Bag HdRnSumAgWord;
extern Bag HdRnTailDepth;
extern Bag HdStack;
extern Bag HdStrComm;
extern Bag HdStrDiff;
extern Bag HdStrEq;
extern Bag HdStrIn;
extern Bag HdStrLt;
extern Bag HdStrMod;
extern Bag HdStrPow;
extern Bag HdStrPrint;
extern Bag HdStrProd;
extern Bag HdStrQuo;
extern Bag HdStrSum;
extern Bag HdTilde;
extern Bag HdTime;
extern Bag HdTimes;
extern Bag HdTrue;
extern Bag HdUnion;
extern Bag HdVecFFEL;
extern Bag HdVecFFER;
extern Bag HdVoid;
extern Bag HdLastErrorMsg;
extern Bag HdTildePr;

/* added by hand */
extern Bag HdChars[];
/* end of by hand */
/* NOTE: What about agcollec.c: Powers (Bag*) ?? */

void InitAllGlobals(void) {
    /* added by hand */
    Int i=0;
    for(i=0; i < 256; ++i)
	InitGlobalBag(& HdChars[i], GuMakeMessage("HdChars[%d]",i));
    for(i=0; i < SCANNER_INPUTS; ++i) {
	InitGlobalBag(& (InputFiles[i].packages), GuMakeMessage("InputFiles[%d].packages",i));
	InitGlobalBag(& (InputFiles[i].package), GuMakeMessage("InputFiles[%d].package",i));
	InitGlobalBag(& (InputFiles[i].data),    GuMakeMessage("InputFiles[%d].data",i));
	InitGlobalBag(& (InputFiles[i].imports), GuMakeMessage("InputFiles[%d].imports",i));
    }
    /* end of by hand */
    InitGlobalBag(&HdLastErrorMsg, "HdLastErrorMsg");    
    InitGlobalBag(&HdBases, "HdBases");    
    InitGlobalBag(&HdCall1, "HdCall1");
    InitGlobalBag(&HdCall2, "HdCall2");
    InitGlobalBag(&HdCallComm, "HdCallComm");
    InitGlobalBag(&HdCallDiff, "HdCallDiff");
    InitGlobalBag(&HdCallEq, "HdCallEq");
    InitGlobalBag(&HdCallIn, "HdCallIn");
    InitGlobalBag(&HdCallLt, "HdCallLt");
    InitGlobalBag(&HdCallMod, "HdCallMod");
    InitGlobalBag(&HdCallOop1, "HdCallOop1");
    InitGlobalBag(&HdCallOop2, "HdCallOop2");
    InitGlobalBag(&HdCallPow, "HdCallPow");
    InitGlobalBag(&HdCallPrint, "HdCallPrint");
    InitGlobalBag(&HdCallProd, "HdCallProd");
    InitGlobalBag(&HdCallQuo, "HdCallQuo");
    InitGlobalBag(&HdCallSumAgWord, "HdCallSumAgWord");
    InitGlobalBag(&HdCallSum, "HdCallSum");
    InitGlobalBag(&HdCPC, "HdCPC");
    InitGlobalBag(&HdCPL, "HdCPL");
    InitGlobalBag(&HdCPS, "HdCPS");
    InitGlobalBag(&HdCurLHS, "HdCurLHS");
    InitGlobalBag(&HdCycLastE, "HdCycLastE");
    InitGlobalBag(&HdExec, "HdExec");
    InitGlobalBag(&HdFalse, "HdFalse");
    InitGlobalBag(&HdFields, "HdFields");
    InitGlobalBag(&HdIdenttab, "HdIdenttab");
    InitGlobalBag(&HdIdWord, "HdIdWord");
    InitGlobalBag(&HdIntFFEs, "HdIntFFEs");
    InitGlobalBag(&HdLast2, "HdLast2");
    InitGlobalBag(&HdLast3, "HdLast3");
    InitGlobalBag(&HdLast, "HdLast");
    InitGlobalBag(&HdLastIntFFE, "HdLastIntFFE");
    InitGlobalBag(&HdLocal, "HdLocal");
    InitGlobalBag(&HdPerm, "HdPerm");
    InitGlobalBag(&HdPkgRecname, "HdPkgRecname");
    InitGlobalBag(&HdRectab, "HdRectab");
    InitGlobalBag(&HdResult, "HdResult");
    InitGlobalBag(&HdReturn, "HdReturn");
    InitGlobalBag(&HdRnAvec, "HdRnAvec");
    InitGlobalBag(&HdRnCentralWeight, "HdRnCentralWeight");
    InitGlobalBag(&HdRnComm, "HdRnComm");
    InitGlobalBag(&HdRnDepth, "HdRnDepth");
    InitGlobalBag(&HdRnDifferenceAgWord, "HdRnDifferenceAgWord");
    InitGlobalBag(&HdRnDiff, "HdRnDiff");
    InitGlobalBag(&HdRnEq, "HdRnEq");
    InitGlobalBag(&HdRnExponentAgWord, "HdRnExponentAgWord");
    InitGlobalBag(&HdRnExponentsAgWord, "HdRnExponentsAgWord");
    InitGlobalBag(&HdRnInformationAgWord, "HdRnInformationAgWord");
    InitGlobalBag(&HdRnIn, "HdRnIn");
    InitGlobalBag(&HdRnIsAgWord, "HdRnIsAgWord");
    InitGlobalBag(&HdRnIsCompatibleAgWord, "HdRnIsCompatibleAgWord");
    InitGlobalBag(&HdRnLeadingExponent, "HdRnLeadingExponent");
    InitGlobalBag(&HdRnLt, "HdRnLt");
    InitGlobalBag(&HdRnMod, "HdRnMod");
    InitGlobalBag(&HdRnNormalizeIgs, "HdRnNormalizeIgs");
    InitGlobalBag(&HdRnOp, "HdRnOp");
    InitGlobalBag(&HdRnPow, "HdRnPow");
    InitGlobalBag(&HdRnPrint, "HdRnPrint");
    InitGlobalBag(&HdRnProd, "HdRnProd");
    InitGlobalBag(&HdRnQuo, "HdRnQuo");
    InitGlobalBag(&HdRnReducedAgWord, "HdRnReducedAgWord");
    InitGlobalBag(&HdRnRelativeOrder, "HdRnRelativeOrder");
    InitGlobalBag(&HdRnSumAgWord, "HdRnSumAgWord");
    InitGlobalBag(&HdRnSum, "HdRnSum");
    InitGlobalBag(&HdRnTailDepth, "HdRnTailDepth");
    InitGlobalBag(&HdStack, "HdStack");
    InitGlobalBag(&HdStrComm, "HdStrComm");
    InitGlobalBag(&HdStrDiff, "HdStrDiff");
    InitGlobalBag(&HdStrEq, "HdStrEq");
    InitGlobalBag(&HdStrIn, "HdStrIn");
    InitGlobalBag(&HdStrLt, "HdStrLt");
    InitGlobalBag(&HdStrMod, "HdStrMod");
    InitGlobalBag(&HdStrPow, "HdStrPow");
    InitGlobalBag(&HdStrPrint, "HdStrPrint");
    InitGlobalBag(&HdStrProd, "HdStrProd");
    InitGlobalBag(&HdStrQuo, "HdStrQuo");
    InitGlobalBag(&HdStrSum, "HdStrSum");
    InitGlobalBag(&HdTilde, "HdTilde");
    InitGlobalBag(&HdTime, "HdTime");
    InitGlobalBag(&HdTimes, "HdTimes");
    InitGlobalBag(&HdTrue, "HdTrue");
    InitGlobalBag(&HdUnion, "HdUnion");
    InitGlobalBag(&HdVecFFEL, "HdVecFFEL");
    InitGlobalBag(&HdVecFFER, "HdVecFFER");
    InitGlobalBag(&HdVoid, "HdVoid");
    InitGlobalBag(&HdTildePr, "HdTildePr");
}
