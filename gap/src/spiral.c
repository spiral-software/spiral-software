#include        <stdlib.h>
#include        <time.h>
#ifdef WIN32
#include        <windows.h>
#endif
#include		<spiral_build_info.h>
#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* Bag, NewBag, T_STRING, .. */
#include        "idents.h"
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */
#include        "scanner.h"             /* Pr()                            */
#include        "objects.h"
#include		"string4.h"
#include        "integer4.h"
#include        "namespaces.h"
#include        "record.h"
#include        "function.h"            /* ChangeEnv()                     */
#include        "spiral_fft.h"
#include        "spiral_spl_prog.h"
#include        "spiral_delay_ev.h"
#include        "spiral_bag_list.h"
#include        "statemen.h"
#include        "comments.h"
#include		"GapUtils.h"
#include        "list.h"
#include        "plist.h"
#include		"conf.h"

#include        <string.h>              /* strstr()                        */
#include        "args.h"
#include        "tables.h"
#include        "hash.h"
#include        "hooks.h"
#include        "double.h"

extern Bag HdIdenttab;            /* used by FunApropos()            */

Bag     HdListClass = 0;          /* used by Fun_ObjId()              */


/****************************************************************************
**
*F  InitLibName(argv, SyLibName, maxLen) . . . . . . . . .  sets library path
**
**  'InitLibName' initializes GAP library path variable by reading the  value
**  from libsys_conf - SPIRAL's configuration layer library.
**
**  The value is appended to SyLibname, so SyLibname must be already initial-
**  ized to a valid string. maxLen should  contain the  available storage  of
**  SyLibName.
*/
void            InitLibName (char *progname, char *SyLibname, int maxLen)
{
    char *lib_ptr;
    char *path_sep;

    // InitSysConf(progname);	don't need, only sets pgm name, called directly below.
	GuSysSetProgname("gap");
	
    if(SyLibname[0] == '\0') {
        /* read gap_lib_dir from configuration */
		config_val_t * temp = config_demand_val("gap_lib_dir");
		if (temp != NULL)
			lib_ptr = temp->strval;
		else
			lib_ptr = "";

		temp = config_demand_val("path_sep");
		if (temp != NULL)
			path_sep = temp->strval;
		else
			path_sep = "";

        strncat( SyLibname, lib_ptr, maxLen - strlen(SyLibname)-2 );
        strncat( SyLibname, path_sep, maxLen - strlen(SyLibname)-2 );
    }
}

/****************************************************************************
**
*F  InitSPIRAL_Paths() . . . . . . . . . .  Initialize paths from libsys_conf
**
** This function currently does nothing.
*/
void            InitSPIRAL_Paths (void)
{
}

/****************************************************************************
**
*F  Apropos( <sub_string> ) . . .  Print identifiers that contain a substring
**
**  'Apropos' prints a list of all defined identifiers that contain a given
**  substring in their name. It is useful  to  find related  functions. For
**  example:
**      spiral> Apropos("ARep");
**       IsFaithfulARep
**       InsertedInductionARep
**       KernelARep
**       ...
*/
void Apropos(Obj hdSubStr, Obj tab, int recursive);

Bag       FunApropos (Bag hdCall)
{
    Bag           hdSubStr;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: Apropos( <sub_string> )", 0,0);

    hdSubStr = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hdSubStr)!=T_STRING)
        return Error("usage: Apropos( <sub_string> )", 0,0);

    Apropos(hdSubStr, HdIdenttab, 1);
    RecursiveClearFlag(HdIdenttab, BF_VISITED);
    return HdVoid;
}

void Apropos(Obj hdSubStr, Obj tab, int recursive) {
    char * sub_str = (char*) PTR_BAG(hdSubStr);
    UInt i, printed_header = 0;

    if ( GET_FLAG_BAG(tab, BF_VISITED) ) return;
    else SET_FLAG_BAG(tab, BF_VISITED);

    for ( i = 0; i < TableSize(tab); i++ ) {
        Obj ent = PTR_BAG(tab)[i];
        Obj val;
        if ( ent == 0 )  continue;
        val = VAR_VALUE(ent);

        if(val != 0 && strstr(VAR_NAME(ent), sub_str) != NULL) { /* found something */
            if(! printed_header) {
                Pr("**** %g ****\n", (Int)tab, (Int)tab);
            }
            printed_header = 1;
            Pr("%s\n", (Int)VAR_NAME(ent), 0);
        }
    }
    for ( i = 0; i < TableSize(tab); i++ ) {
        Obj ent = PTR_BAG(tab)[i];
        Obj val;
        if ( ent == 0 )  continue;
        val = VAR_VALUE(ent);

        if ( recursive && val != 0 && val != tab && val != HdIdenttab && GET_TYPE_BAG(val) == T_NAMESPACE ) {
            SET_FLAG_BAG(ent, BF_VISITED);
            Apropos(hdSubStr, val, 1);
        }
    }
}

/****************************************************************************
**
*F  TimeInSecs( ) . .  returns the time in seconds since the Epoch (1/1/1970)
*/
Bag       FunTimeInSecs (Bag hdCall)
{
#ifndef SYS_IS_64_BIT
	signed long hi; /* long should be 32 bits */
    unsigned long lo;
    unsigned long long timeval;
#else
	signed int hi; /* long should be 32 bits */
    unsigned int lo;
    UInt timeval;
#endif
	Bag hdResult;

   /* get and check the argument                                           */
    if ( GET_SIZE_BAG(hdCall) != 1 * SIZE_HD )
        return Error("usage: TimeInSecs()", 0,0);

#ifndef SYS_IS_64_BIT    
	timeval = (unsigned long long) time(0);
#else
	timeval = (UInt) time(0);
#endif

    /* GAP can accept only 28-bit integers in INT_TO_HD, the time value    */
    /* has to be represented as a long integer, but instead of constructing*/
    /* the long integer we compute it, see next comment */
    lo = 0x0FFFFFFF & timeval;
    hi = ((0xF0000000 & timeval) >> 28);

    /* hdResult = hi * 2^28 + lo = hi * 2^26 * 4 + lo
       signed 2^28 is 30 bits, so we have to multiply by 2^26 and then by 4 to
       cumulatively multiply by 2^28 */
    hdResult = INT_TO_HD(hi);
    hdResult = PROD( PROD(hdResult, INT_TO_HD(1<<26)), INT_TO_HD(4) );
    hdResult = SUM( hdResult, INT_TO_HD(lo) );
    return hdResult;
}



/****************************************************************************
**
*F  TimezoneOffset( ) . .  returns the time in seconds since the Epoch (1/1/1970)
*F                          adjusted for the local timezone
*/
Bag       FunTimezoneOffset (Bag hdCall)
{
	int seconds = 0;

#ifdef WIN32
	TIME_ZONE_INFORMATION timeZoneInformation;
	long timezoneid;
#else
    time_t t = time(NULL);
    struct tm lt = {0};
#endif

   /* get and check the argument                                           */
	if ( GET_SIZE_BAG(hdCall) != 1 * SIZE_HD )
		return Error("usage: TimezoneOffset()", 0,0);

#ifdef WIN32
    timezoneid = GetTimeZoneInformation(&timeZoneInformation);
	seconds = timeZoneInformation.Bias;
	if (timezoneid == TIME_ZONE_ID_DAYLIGHT)
		seconds += timeZoneInformation.DaylightBias;
	else
		seconds += timeZoneInformation.StandardBias;
    seconds =  -(seconds * 60);
#else
	localtime_r(&t, &lt);
	seconds = lt.tm_gmtoff;
#endif
 
	return INT_TO_HD(seconds);
}

#ifdef WIN32
Bag   dtime() {
    LARGE_INTEGER freq, tick;
    QueryPerformanceCounter(&tick); 
    QueryPerformanceFrequency(&freq);
    return ObjDbl((double)tick.QuadPart/(double)freq.QuadPart); 
}
#else
#if HAVE_GETTIMEOFDAY
#include <sys/time.h>
Bag   dtime() {
    static struct timeval t;
    gettimeofday(&t, 0);
    return ObjDbl( (double)t.tv_sec+(double)t.tv_usec*1e-6 );
}
#else
Bag   dtime() { return HdFalse; }
#endif
#endif

/****************************************************************************
**
*F  TimeInMicroSecs( ) . . as TimeInSecs but returns a double with usec resolution
*/
Bag       FunTimeInMicroSecs (Bag hdCall)
{
    Bag hd;
    if ( GET_SIZE_BAG(hdCall) != 1 * SIZE_HD ) return Error("usage: TimeInMicroSecs()", 0,0);
    hd = dtime();
    if (hd == HdFalse) return Error("TimeInMicroSecs: microsecond timer is disabled."
                                    "GAP must be compiled with HAVE_GETTIMEOFDAY", 0, 0);
    else return  hd;
}

/****************************************************************************
**
*F  SysVerbose( 0 | 1 | 2 ) . . . . . . . . .  sets verbose mode for sys_conf
**
*/
Bag       FunSysVerbose (Bag hdCall)
{
    char * usage = "usage: SysVerbose( 0 | 1 | 2 )";
    Bag hdVerbose;
    int verbose;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) == SIZE_HD ) return INT_TO_HD(GuSysGetVerbose());
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage, 0,0);

    hdVerbose = EVAL( PTR_BAG(hdCall)[1] );
    if( GET_TYPE_BAG(hdVerbose) != T_INT ) return Error(usage,0,0);

    verbose = HD_TO_INT(hdVerbose);
    if(verbose<0 || verbose > 2) return Error(usage,0,0);

    GuSysSetVerbose(verbose);

    return HdVoid;
}


void ErrorExit(int code) {
    Throw exc(ERR_OTHER, "");
};

/****************************************************************************
**
*F  FunInternalFunction(<var>) . . .return a bag containing internal function
**
*/
Bag       FunInternalFunction (Bag hdCall)
{
    char * usage = "usage: InternalFunction( <var> )";
    Bag hdVar, hdFunc = 0;
    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage, 0,0);
    hdVar = PTR_BAG(hdCall)[1];
    if( GET_TYPE_BAG(hdVar) != T_VAR ) return Error(usage,0,0);
    hdFunc = EVAL(hdVar);
    if( GET_TYPE_BAG(hdFunc) != T_FUNCINT ) Error("no such internal function", 0, 0);
    return hdFunc;
}

/****************************************************************************
**
*F  FunBlanks(<num>) . . . . . . . return a string consisting of <num> blanks
**
*/
Bag       FunBlanks (Bag hdCall)
{
    char * usage = "usage: Blanks( <positive number of blanks> )";
    Bag result;
    int i;
    char * str;
    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage, 0,0);
    i = HdToInt( PTR_BAG(hdCall)[1], usage, 0, 0 );
    if(i < 0) return Error(usage, 0, 0);

    result = NEW_STRING(i);
    str = CHARS_STRING(result);
    for( ; i>0; --i, ++str)
        *str = ' ';
    return result;
}


/****************************************************************************
**
*F  ValueOf(<name>) . . . . . returns the value of a variable with given name
**
** This function makes possible to read a batch of variables automatically.
*/
Bag       FunValueOf (Bag hdCall)
{
    char * usage = "usage: ValueOf(<name>)";
    Obj hdName = 0, var = 0, val = 0;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hdName = EVAL( PTR_BAG(hdCall)[1] );
    if( GET_TYPE_BAG(hdName) != T_STRING ) return Error(usage,0,0);

    var = FindIdent(HD_TO_STRING(hdName));
    if ( GET_TYPE_BAG(var) == T_VARAUTO ) EVAL(var);
    val = VAR_VALUE(var);
    if(val == 0) return Error("Variable: '%s' must have a value",
                              (Int)HD_TO_STRING(hdName), 0);
    else return val;
}

Bag       Fun_GlobalVar (Bag hdCall)
{
    char * usage = "usage: _GlobalVar(var)";
    Obj hdVar = 0;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hdVar = INJECTION_D(EVAL(PTR_BAG(hdCall)[1]));
    if( GET_TYPE_BAG(hdVar) != T_VAR && GET_TYPE_BAG(hdVar) != T_VARAUTO) return Error(usage,0,0);

    hdVar = GlobalIdent(VAR_NAME(hdVar));
    if ( GET_TYPE_BAG(hdVar) == T_VARAUTO ) EVAL(hdVar);
    return PROJECTION_D(hdVar);
}
/****************************************************************************
**
*F  NameOf(<var>) . . . . . . . . . . .  returns the string for variable name
**
*/
Bag       FunNameOf (Bag hdCall)
{
    char * usage = "usage: NameOf(<var>)";
    Bag hdVar = 0, hdRes = 0;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hdVar = INJECTION_D(EVAL(PTR_BAG(hdCall)[1]));
    if( GET_TYPE_BAG(hdVar) != T_VAR && GET_TYPE_BAG(hdVar) != T_VARAUTO) return Error(usage,0,0);

    hdRes = NEW_STRING(strlen(VAR_NAME(hdVar)));
    strncpy(CHARS_STRING(hdRes), VAR_NAME(hdVar), strlen(VAR_NAME(hdVar))+1);
    return hdRes;
}

/****************************************************************************
**
*F  Assign(<name>, <value>) . . assigns a value to a variable with given name
**
** This function makes possible to create variables automatically:
**
**   for i in [1..10] do
**      Assign(Concatenation("z",String(i)), 1);
**   od;
**
** Will create and assign 1 to 10 variables: z1, z2, ..., z10
*/
Bag       FunAssign (Bag hdCall)
{
    char * usage = "usage: Assign(<name>, <value>)";
    Bag hdName = 0, hdVal = 0;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )  return Error(usage, 0,0);
    hdName = EVAL( PTR_BAG(hdCall)[1] );
    if( GET_TYPE_BAG(hdName) != T_STRING && GET_TYPE_BAG(hdName) != T_DELAY) return Error(usage,0,0);
    if( GET_TYPE_BAG(hdName) == T_DELAY ) {
        hdName = PTR_BAG(hdName)[0];
        if ( GET_TYPE_BAG(hdName) != T_VAR && GET_TYPE_BAG(hdName) != T_VARAUTO ) return Error(usage,0,0);
        hdName = FindIdentWr(VAR_NAME(hdName));
        hdVal = EVAL( PTR_BAG(hdCall)[2] );
        SET_VAR_VALUE(hdName, hdVal);
        return hdVal;
    }
    else {
        hdVal = EVAL( PTR_BAG(hdCall)[2] );
        IdentAssign((char*)PTR_BAG(hdName), hdVal);
        return hdVal;
    }
}

/****************************************************************************
**
*F  Declare(<var1>, <var2>, ...)  . . . . . create variables in current scope
**
**  Declare will initialize a variable to HdVoid, if it didn't exist.
**  Declare is non-destructive.
*/
Bag       FunDeclare (Bag hdCall)
{
    char * usage = "usage: Declare(<var1>, <var2>, ...)";
    Bag hdVar = 0;
    UInt i;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) < 2 * SIZE_HD )  return Error(usage, 0,0);

    for ( i = 1; i < GET_SIZE_BAG(hdCall) / SIZE_HD; ++i) {
        hdVar = PTR_BAG(hdCall)[i];
        if ( GET_TYPE_BAG(hdVar) != T_VAR && GET_TYPE_BAG(hdVar) != T_VARAUTO ) return Error(usage,0,0);
        hdVar = FindIdentWr(VAR_NAME(hdVar));
        if ( VAR_VALUE(hdVar) == 0 )
            SET_VAR_VALUE(hdVar, HdVoid);
    }
    return HdVoid;
}

/****************************************************************************
**
*F  IsVoid(<expr>)  . . . . . . .  returns true if <expr> evaluates to HdVoid
**
**  Also can be used to find Declare()'d but uninitialized variables.
*/
Bag       FunIsVoid (Bag hdCall)
{
    char * usage = "usage: IsVoid(<expr>)";
    Bag hd = 0;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);

    hd = EVAL(PTR_BAG(hdCall)[1]);
    if ( hd == HdVoid ) return HdTrue;
    else return HdFalse;
}


/****************************************************************************
**
*F  Same( <obj1>, <obj2> ) . . . . . . . . . . . . . .  comparison by pointer
**
**  Returns true of obj1 and obj2 refer to the same object
*/
Obj  FunSame ( Obj hdCall ) {
    char * usage = "usage: Same( <obj1>, <obj2> )";
    Obj o1 = 0, o2 = 0;
    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )  return Error(usage, 0,0);
    o1 = EVAL(PTR_BAG(hdCall)[1]);
    o2 = EVAL(PTR_BAG(hdCall)[2]);
    return (o1==o2) ? HdTrue : HdFalse;
}

/****************************************************************************
**
*F  Constraint( <cond> ) . . . . . . . . . . . . .  make sure condition holds
**
*/
void Constraint( Obj cond ) {
    /* keeps copy to unevaluated cond, for nicer error reporting           */
    Obj evalCond = 0;
    evalCond = EVAL(cond);
    if(evalCond == HdTrue)
        return;
    else if(evalCond == HdFalse)
        Error("Condition %g doesn't hold", (Int)cond, 0);
    else
        Error("Condition %g must evaluate to a boolean", (Int)cond, 0);
}

Obj  FunConstraint ( Obj hdCall ) {
    char * usage = "usage: Constraint( <cond> )";
    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    Constraint(PTR_BAG(hdCall)[1]); /* condition is not evaluated here yet     */
    return HdTrue;
}

/****************************************************************************
**
*F  Checked( <cond1>, <cond2>, ..., <expr> ) . . . . . constrained evaluation
b**
**  Checked() verifies all given conditions first, if any of them is false,
**  an error is reported, otherwise <expr> is evaluated and returned.
**
**  <expr> will not be evaluated if any of the conditions are false,  also if
**  a condition is false, then none of the further conditions are evaluated.
**
*/
Obj FunChecked ( Obj hdCall ) {
    char * usage = "usage: Checked( <cond1>, <cond2>, ..., <expr> )";
    Int nargs, i;
    /* get and check the argument                                          */
    nargs = GET_SIZE_BAG(hdCall) / SIZE_HD - 1;
    if ( nargs < 1 )  return Error(usage, 0,0);
    for ( i = 0; i < nargs-1; ++i )
        Constraint(PTR_BAG(hdCall)[1+i]);
    return EVAL(PTR_BAG(hdCall)[nargs]);
}

/****************************************************************************
**
*F  When( <cond>, <result-if-true>, <result-if-false> ) . . . . functional IF
**
**  FunWhen implements internal function 'When' which is a functional equiva-
**  lent of an 'if'  statement. When is special, since it never evaluates the
**  argument when it  is  not necessary. For example, when condition is true,
**  <result-if-true> is returned, and <result-if-false> is not evaluated.
*/
Obj  FunWhen( Obj hdCall ) {
    Obj cond = 0;
    Int nargs;
    char * usage = "usage: When( <cond>, <result-if-true>, <result-if-false> )";
    nargs = GET_SIZE_BAG(hdCall) / SIZE_HD - 1;
    /* get and check the argument                                          */
    if ( nargs != 3 && nargs != 2 ) return Error(usage, 0,0);
    cond = EVAL( PTR_BAG(hdCall)[1] );
    if ( cond==0 || GET_TYPE_BAG(cond) != T_BOOL )  return Error(usage, 0,0);
    else if ( cond == HdTrue ) return EVAL(PTR_BAG(hdCall)[2]);
    else if ( nargs == 2 ) return HdVoid;
    else return EVAL(PTR_BAG(hdCall)[3]); /* nargs==3 && cond==HdTrue */
}

/****************************************************************************
**
*F  Cond( <cond1>, <res1>, <cond2>, ..., <res-else> ) . . . . functional CASE
**
**  FunCond implements internal function 'Cond' which is a functional equiva-
**  lent of a 'case' statement. Cond takes a list of conditions and correspo-
**  nding results. Cond  evaluates  the  conditions,  until it finds one that
**  evaluates to true, then it evaluates corresponding result and returns it.
**  No other results (or further conditions) are evaluated.
*/
Obj  FunCond( Obj hdCall ) {
    char * usage = "usage: Cond( <cond1>, <res1>, <cond2>, ..., <res-else> )";
    Int i, nargs;
    /* get and check the argument                                          */
    nargs = GET_SIZE_BAG(hdCall) / SIZE_HD - 1;
    if ( nargs < 3 ) return Error(usage, 0,0);

    for ( i = 1; i < nargs; i += 2 ) {
        Obj cond = EVAL( PTR_BAG(hdCall)[i] );
        if ( cond == HdTrue )
            return EVAL(PTR_BAG(hdCall)[i+1]);
        else if ( cond == HdFalse )
            /* continue */;
        else
            return Error(usage, 0, 0);
    }

    if ( i == nargs ) return EVAL(PTR_BAG(hdCall)[nargs]);
    else return Error("No else clause, when no conditions are true", 0, 0);
}

/****************************************************************************
**
*F  BagInfo() . . . . . . . . . . . . . . . . . . . . display bag information
*/
void BagInfo ( Obj hd ) {
    if(hd == NULL)
        Pr("NULL bag\n", 0, 0);
    else if(! IS_BAG(hd))
        Pr("INVALID bag\n", 0, 0);
    else {
        Pr("Type    : %d ( %s )\n", (Int)GET_TYPE_BAG(hd),  (Int)NameType[GET_TYPE_BAG(hd)]);
        if(GET_TYPE_BAG(hd)==T_INT) {
            Pr("Value   : %g\n", (Int)hd, 0);
            return;
        }
        Pr("Size    : %d \t Addr : 0x%s \n",
           (Int) GET_SIZE_BAG(hd),
           (Int) PTR_BAG(HexStringInt(INT_TO_HD(hd))));
        Pr("Handles : %d \t PTR_BAG  : 0x%s \n",
           (Int) NrHandles(GET_TYPE_BAG(hd), GET_SIZE_BAG(hd)),
           (Int) PTR_BAG(HexStringInt(INT_TO_HD(PTR_BAG(hd)))));
        Pr("Flags   : %d\n", (Int)GET_FLAGS_BAG(hd), 0);
        Pr("Value   : %g\n", (Int)hd, 0);
    }
}

Obj  FunBagInfo ( Obj hdCall ) {
    char * usage = "usage: BagInfo( <obj> )";
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    BagInfo(EVAL(PTR_BAG(hdCall)[1]));
    return HdVoid;
}

Obj  FunBagAddr ( Obj hdCall ) {
    char * usage = "usage: BagAddr( <obj> )";
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    return INT_TO_HD(EVAL(PTR_BAG(hdCall)[1]));
}

Obj  FunBagType ( Obj hdCall ) {
    char * usage = "usage: BagType( <obj> )";
    Obj hd;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hd = EVAL(PTR_BAG(hdCall)[1]);
    if ( hd == 0 ) return INT_TO_HD(0);
    else return INT_TO_HD(GET_TYPE_BAG(EVAL(PTR_BAG(hdCall)[1])));
}

Obj  FunBagFlags ( Obj hdCall ) {
    char * usage = "usage: BagFlags( <obj> )";
    Obj hd;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hd = EVAL(PTR_BAG(hdCall)[1]);
    if ( hd == 0 || GET_TYPE_BAG(hd) == T_INT ) return INT_TO_HD(0);
    else return INT_TO_HD(GET_FLAGS_BAG(hd));
}

Obj  FunBagFromAddr ( Obj hdCall ) {
    char * usage = "usage: BagFromAddr( <obj> )";
    Obj bag;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    bag = (Obj) HD_TO_INT(EVAL(PTR_BAG(hdCall)[1]));
    if ( ! IS_BAG(bag) ) Error("No such bag", 0, 0);
    return bag;
}


extern ArenaBag_t    MemArena[];
extern int           actAR;


Obj  FunBagBounds ( Obj hdCall ) {
    char * usage = "usage: BagBounds( ). Returns triple [first, last, freebags].\n"
" first is the address of the first master pointer, last is the address of the last"
" master pointer, and freebags is the list of unused master pointers in the range"
" [first, last]. ";
    Obj res, freebags, addr;  Bag fst;
    UInt nfreebags, i;

    if ( GET_SIZE_BAG(hdCall) != SIZE_HD )  return Error(usage, 0,0);

    nfreebags = 0;
    fst = MemArena[actAR].FreeHandleChain;
    while(fst != 0) { ++nfreebags; fst = *(Bag*)(fst); }

    freebags = NewList((int)nfreebags);
    fst = MemArena[actAR].FreeHandleChain; i = 1;
    while(fst != 0) {
        addr = PROD(INT_TO_HD(4), INT_TO_HD(((UInt)fst) >> 2));
        SET_BAG(freebags, i,  addr );
        fst = *(Bag*)fst;
        ++i;
    }

    res = NewList(3);
    addr = PROD(INT_TO_HD(4), INT_TO_HD(((UInt)MemArena[actAR].BagHandleStart) >> 2));
    SET_BAG(res, 1,  addr );
    addr = PROD(INT_TO_HD(4), INT_TO_HD(((UInt)(MemArena[actAR].OldBagStart-1)) >> 2));
    SET_BAG(res, 2,  addr );

    SET_BAG(res, 3,  freebags );

    return res;
}

Obj  FunBagsOfType ( Obj hdCall ) {
    char * usage = "usage: BagsOfType( <type> [, <index>] )";
    Obj res;
    UInt mptr, mptrEnd, cnt, type;
    int index = -1;
    int i;

    if ( GET_SIZE_BAG(hdCall) < 2*SIZE_HD )  return Error(usage, 0,0);
    type = HD_TO_INT(EVAL(PTR_BAG(hdCall)[1]));
    if ( GET_SIZE_BAG(hdCall) > 2*SIZE_HD ) {
        if ( GET_SIZE_BAG(hdCall) > 3*SIZE_HD ) return Error(usage, 0,0);
        index = HD_TO_INT(EVAL(PTR_BAG(hdCall)[2]));
    }

    cnt = 0;
    mptr = (UInt)MemArena[actAR].BagHandleStart;
    mptrEnd = (UInt)MemArena[actAR].OldBagStart - 1;
    while (mptr<mptrEnd) {
        if (*(UInt*)mptr >= mptrEnd) {
            if (GET_TYPE_BAG((Obj)mptr)==type)
                cnt++;
        }
        mptr += SIZE_HD;
    }
    if (index>=0 && index<cnt)
        cnt = 1;
    else
      if (index>=0)
          cnt = 0;
    res = NewList((int)cnt);
    if (cnt>0) {
        i = 1;
        mptr = (UInt)MemArena[actAR].BagHandleStart;
        mptrEnd = (UInt)MemArena[actAR].OldBagStart - 1;
        while (mptr<mptrEnd) {
            if (*(UInt*)mptr >= mptrEnd) {
                if (GET_TYPE_BAG((Obj)mptr)==type && (UInt)mptr != (UInt)res)
                    if (index>=0) {
                        if (index==0) {
                            SET_BAG(res, 1,  PROD(INT_TO_HD(4), INT_TO_HD(mptr >> 2)) );
                            break;
                        }
                        index--;
                    } else {
                        SET_BAG(res, i++,  PROD(INT_TO_HD(4), INT_TO_HD(mptr >> 2)) );
                    }
            }
            mptr += SIZE_HD;
        }
    }
    return res;
}



/****************************************************************************
**
*F  SetEnv( <name>, <value> ) . . . . . . . . .  set the environment variable
**
*/
Bag       FunSetEnv (Bag hdCall)
{
    char * usage = "usage: SetEnv( <name>, <value> )";
    Bag hdName;
    Bag hdValue;
    char * name;
    char * value;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )  return Error(usage, 0,0);
    hdName = EVAL( PTR_BAG(hdCall)[1] );
    hdValue = EVAL( PTR_BAG(hdCall)[2] );
    if( GET_TYPE_BAG(hdName) != T_STRING ) return Error(usage,0,0);
    if( GET_TYPE_BAG(hdValue) != T_STRING ) return Error(usage,0,0);

    name = (char*) PTR_BAG(hdName);
    value = (char*) PTR_BAG(hdValue);

    return INT_TO_HD(GuSysSetenv(name, value, 1));
}

/****************************************************************************
**
*F  GetEnv( <name> ) . . . . . . . . . . . . . . get the environment variable
**
*/
Bag       FunGetEnv (Bag hdCall)
{
    char * usage = "usage: GetEnv( <name> )";
    Bag hdName;
    char * name, * value;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hdName = EVAL( PTR_BAG(hdCall)[1] );
    if( GET_TYPE_BAG(hdName) != T_STRING ) return Error(usage,0,0);

    name = (char*) PTR_BAG(hdName);
    value = getenv(name);
    if ( value == NULL ) return StringToHd("");
    else return StringToHd(value);
}

Obj  Props ( Obj hd ) {
    if ( GET_TYPE_BAG(hd) != T_VAR  && GET_TYPE_BAG(hd) != T_VARAUTO)
        return Error("Props: <hd> is not a variable", 0, 0);
    if( VAR_PROPS(hd) == 0 ) {
        Obj props = NewBag(T_REC, 0);
        return (SET_VAR_PROPS(hd, props));
    }
    else return VAR_PROPS(hd);
}

Obj  FunProps ( Obj hdCall ) {
    char * usage = "usage: Props( <delayed-var> )";
    Obj hd;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hd = EVAL(PTR_BAG(hdCall)[1]);
    if ( hd == 0 || GET_TYPE_BAG(hd) != T_DELAY )  return Error(usage, 0,0);
    hd = PTR_BAG(hd)[0];
    if ( GET_TYPE_BAG(hd) != T_VAR )
        return Error("Props: Delayed expression is not a variable", 0, 0);
    return Props(hd);
}
Obj  FunPkg ( Obj hdCall ) {
    char * usage = "usage: Pkg( <var> )";
    Obj hd, props, pkg;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hd = PTR_BAG(hdCall)[1];
    if ( hd == 0 || (GET_TYPE_BAG(hd) != T_VAR && GET_TYPE_BAG(hd) != T_VARAUTO) )
        return Error(usage, 0,0);

    props = Props(hd);
    pkg = FindRecnameRec(props, HdPkgRecname, &hd)[1];
    if ( pkg == 0 ) return INT_TO_HD(0);
    else {
        pkg = TableId(pkg);
        if ( pkg == 0 ) return HdFalse;
        return pkg;
    }
}


int  reachableFrom ( Obj root, Obj hd ) {
    if ( ! IS_BAG(root) || (! IS_INTOBJ(root) && GET_FLAG_BAG(root, BF_VISITED)) )
        return 0;
    else if ( root == hd )
        return 1;
    else {
        UInt nhandles = NrHandles(GET_TYPE_BAG(root), GET_SIZE_BAG(root));
        UInt i;
        SET_FLAG_BAG(root, BF_VISITED);

        /*for(i = 0; i < GET_SIZE_BAG(root)/sizeof(Bag); i++ ) {*/
        for(i = 0; i < nhandles; ++i) {
            Obj child = PTR_BAG(root)[i];
            if(! IS_BAG(child)) continue;
            if(reachableFrom(child, hd)) {
                Pr("%d %s ", (Int)child, (Int)NameType[GET_TYPE_BAG(child)]);
                if ( GET_TYPE_BAG(child) == T_VAR || GET_TYPE_BAG(child) == T_VARAUTO || GET_TYPE_BAG(child) < T_MUTABLE ||
                     (GET_TYPE_BAG(child) > T_DELAY && GET_TYPE_BAG(child) < T_STATSEQ) || GET_TYPE_BAG(child) >= T_RETURN)
                    Pr("%g", (Int)child, 0);
                Pr("\n", 0, 0);
                return 1;
            }
        }
        return 0;
    }
}

int  reachable ( Obj hd ) {
    UInt i, j, found = 0;
    for (i = 0; i < GlobalBags.nr && !found; i++) {
        if(reachableFrom(*GlobalBags.addr[i], hd)) {
            Pr("global : %d : %s\n", (Int)*GlobalBags.addr[i],
                                     (Int) GlobalBags.cookie[i] );
            found = 1;
        }
    }
    for (j = 0; j < i; j++) {
        RecursiveClearFlag(*GlobalBags.addr[j], BF_VISITED);
    }
    return (int)found;
}


Obj  FunReachability ( Obj hdCall ) {
    char * usage = "usage: Reachability( <obj> [, <root>] )";
    Obj hd, hdRoot = 0;
    if ( GET_SIZE_BAG(hdCall) < 2 * SIZE_HD || GET_SIZE_BAG(hdCall) > 3 * SIZE_HD)
        return Error(usage, 0,0);
    if ( GET_SIZE_BAG(hdCall) == 3 * SIZE_HD )
        hdRoot = INJECTION_D(EVAL(PTR_BAG(hdCall)[2]));

    hd = EVAL(PTR_BAG(hdCall)[1]);
    hd = INJECTION_D(hd); /* strip T_DELAY, if needed */

    if ( hdRoot == 0 )
        return reachable(hd) ? HdTrue : HdFalse;
    else {
        Obj res = reachableFrom(hdRoot, hd) ? HdTrue : HdFalse;
        RecursiveClearFlag(hdRoot, BF_VISITED);
        return res;
    }
}

Obj  FunHdExec ( Obj hdCall ) {
    return HdExec;
}

extern Int  XType ( Obj hd );

Obj  FunXType ( Obj hdCall ) {
    char * usage = "usage: XType(<obj>)";
    Obj hd;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hd = EVAL(PTR_BAG(hdCall)[1]);
    return INT_TO_HD(XType(hd));
}

/****************************************************************************
**
*F  Try( <expr> ) . . . . . . . . catches exceptions in evaluation of <expr>
**
**  Try returns [true, <result>] if <expr> successfully evaluates to <result>,
**  and [false, <errmsg>] if an exception is caught with error message <errmsg>.
*/
extern int BACKTRACE_DEFAULT_LEVEL;
extern int ERROR_QUIET;
extern Obj HdLastErrorMsg;
Obj  FunTry ( Obj hdCall ) {
    char * usage = "usage: Try(<expr>)";
    Obj hd;
    volatile Int level, quiet;
    volatile UInt stack, evalStack;
    volatile Obj exec;
	volatile TypInputFile  * inputStackTop;
	volatile TypOutputFile * outputStackTop;
    exc_type_t e;

    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hd = PTR_BAG(hdCall)[1];
    level = BACKTRACE_DEFAULT_LEVEL;
    quiet = ERROR_QUIET;

    BACKTRACE_DEFAULT_LEVEL=0;
    ERROR_QUIET = 1;
    exec = HdExec;
    stack = TopStack;
    evalStack = EvalStackTop;
	inputStackTop  = Input;
	outputStackTop = Output;

    Try {
        hd = EVAL(hd);
    }
    Catch(e) {
		int ok;
        Obj res = NewList(2);
        SET_BAG(res, 1,  HdFalse );
        /* exceptions raised using Error() are already printed at this point */
        if(e!=ERR_GAP) {
            SET_BAG(res, 2,  StringToHd(exc_err_msg()) );
        }
        else SET_BAG(res, 2,  Copy(HdLastErrorMsg) );

        while ( HdExec != exec )
            ChangeEnv( PTR_BAG(HdExec)[4], CEF_CLEANUP );
        while ( TopStack > stack )
            SET_BAG(HdStack, TopStack--,  0 );
        while ( EvalStackTop > evalStack )
            EVAL_STACK_POP;

		// close files opened during context of Try()

		ok = 1;
		while ( ( Output > outputStackTop) && ok )
			ok = CloseOutput();
		ok = 1;
		while ( ( Input > inputStackTop) && ok )
			ok = CloseInput();

        BACKTRACE_DEFAULT_LEVEL = level;
        ERROR_QUIET = quiet;
        return res;
    }

    {
        Obj res = NewList(2);
        SET_BAG(res, 1,  HdTrue );
        SET_BAG(res, 2,  hd );
        BACKTRACE_DEFAULT_LEVEL = level;
        ERROR_QUIET = quiet;
        return res;
    }
}

/****************************************************************************
**
*F  PathRelativeToSPIRAL( <path> ) . . . . . . . . this function does that
**
** This function does that and that and that and that.
*/
Obj FunPathRelativeToSPIRAL( Obj hdCall ) {
     char * usage = "usage: PathRelativeToSPIRAL( <path> )";
     Obj hdPath, hdRes;
     char * path;
     char * s;
     char * spiral_dir;
     /* get and check the argument                                          */
     if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage, 0,0);
     hdPath = EVAL( PTR_BAG(hdCall)[1] );
     if( GET_TYPE_BAG(hdPath) != T_STRING ) return Error(usage,0,0);
     path = strdup( (char*) PTR_BAG(hdPath) );
     spiral_dir = config_demand_val("spiral_dir")->strval;

     /* strip install directory */
     if ( strncmp(path, spiral_dir, strlen(spiral_dir)) == 0 ) {
         s = strdup(path+strlen(spiral_dir));
         free(path);
         path = s;
     }
     else s = path;

     /*while ( *s != '\0' ) {
         / look for .. /
         if ( *s == '.' && *(s+1) == '.' ) */
     hdRes = StringToHd(path);
     free(path);
     return hdRes;
}


/****************************************************************************
**
*F  Version()
**
** Returns version string generated during build
*/

Obj FunVersion(Obj hdCall) {
	Obj hdRes;
	hdRes = StringToHd(SPIRAL_VERSION_STRING);
	return hdRes;
}


/****************************************************************************
**
*F  BuildInfo()
**
** Prints name:value pairs of build info
*/

Obj FunBuildInfo(Obj hdCall) {
	Pr("Version : %s\n", SPIRAL_VERSION_STRING, 0);
	Pr("GitHash : %s\n", SPIRAL_GIT_COMMIT_HASH, 0);
	Pr("GitRemote : %s\n", SPIRAL_GIT_REMOTE_URL, 0);
	Pr("GitBranch : %s\n", SPIRAL_GIT_COMMIT_BRANCH, 0);
	Pr("DateTimeUTC : %s %s\n", SPIRAL_BUILD_DATE_UTC, SPIRAL_BUILD_TIME_UTC);
	Pr("Compiler : %s %s\n", SPIRAL_C_COMPILER_ID, SPIRAL_C_COMPILER_VERSION);
    return HdVoid;
}



/****************************************************************************
**
*F  GetPromptString( char* RecFieldName) . . . . . .looking for prompt string
**
** This function returns value of LocalConfig.gapinfo.prompts.(RecFieldName).
** Returns 0 if error occured or value found is not a T_STRING.
*/

Bag GetPromptString(char* RecFieldName) {
    Bag hdRec = FindIdent("LocalConfig");
    if (hdRec==0) return 0;
    if (GET_TYPE_BAG(hdRec)==T_VAR && VAR_VALUE(hdRec)==0)
        return 0;
    hdRec = EVAL(hdRec);
    if (hdRec) {
        Bag* pRec = FindRecField(hdRec, "gapinfo");
        if (pRec == 0) return 0;
        pRec = FindRecField(pRec[1], "prompts");
        if (pRec == 0) return 0;
        pRec = FindRecField(pRec[1], RecFieldName);
        if (pRec == 0) return 0;
        hdRec = EVAL(pRec[1]);
        if (GET_TYPE_BAG(hdRec)==T_STRING)
            return hdRec;
    }
    return 0;
}

/****************************************************************************
**
*F  FunEditDef( <obj> ) . . . . . . . . . . . . . internal function EditDef()
**
** Extracting file name and line number from <obj> documentation and executing
** HooksEditFile.
*/

Bag  FunEditDef ( Bag hdCall ) {
    char *      usage = "usage: EditDef(<obj>)";
    char        fileName[512];
    Int         line;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);

    switch(FindDocAndExtractLoc(PTR_BAG(hdCall)[1], fileName, &line)) {
        case  0: { Pr("--no documentation--\n", 0, 0); break; }
        case -1: { Pr("--defnition not found--\n", 0, 0); break; }
        case  1: { HooksEditFile(fileName, line); break; }
    }
    return HdVoid;
}

/****************************************************************************
**
*F  FunFindRefs( <obj> [, <by value> = false] ) . internal function FindRefs()
**
**  Returns list of objects which have references to <obj> (might be indirect 
**  link) and each object has documentaion, i.e. this objects are closest 
**  documented objects. 
**  When <by value> is true it will search references to <obj> value, 
**  when false - to <obj> itself.
*/

typedef struct {
    Bag* list;
    UInt list_capacity;
    UInt list_count;
    UInt mem_err;
} refs_search_t;

int  findrefs_recursion ( Obj root, Obj hd, refs_search_t* result ) {
    if (root == hd)
        return 1;
    else if ( IS_BAG(root) && !IS_INTOBJ(root) && !GET_FLAG_BAG(root, BF_VISITED) ) {
        UInt nhandles = NrHandles(GET_TYPE_BAG(root), GET_SIZE_BAG(root));
        UInt i;
        SET_FLAG_BAG(root, BF_VISITED);

        for(i = 0; i < nhandles; ++i) {
            Obj child = PTR_BAG(root)[i];
            if (findrefs_recursion(child, hd, result)) {
                char    fileName[512];
                Int     line;
                UInt    type = GET_TYPE_BAG(root);
                if ( (type == T_VAR || type == T_FUNCTION || type == T_METHOD 
                    || type == T_MAKEFUNC || type == T_MAKEMETH || type==T_REC 
                    || type==T_NAMESPACE) && !result->mem_err
                    && FindDocAndExtractLoc(root, fileName, &line) == 1) {
                    if (result->list_count == result->list_capacity) {
                        Int  new_cap = 3*result->list_capacity/2;
                        Bag* new_list = realloc( result->list, new_cap*SIZE_HD );
                        if (new_list) {
                            result->list = new_list;
                            result->list_capacity = new_cap;
                        } else {
                            result->mem_err = 1;
                            return 1; 
                        }
                    }
                    result->list[result->list_count++] = root;
                    return 0;
                } else
                    return 1; 
            };
        }
    }
    return 0;
}

Bag  findrefs_global( Obj hd ) {
    char* mem_err = "Cannot allocate enough memory.\n";
    UInt i;
    Bag list;
    refs_search_t result;
    
    result.mem_err = 0;
    result.list_count = 0;
    result.list_capacity = 10;
    result.list = malloc(result.list_capacity*SIZE_HD);
    
    if (result.list==0) {
        Pr(mem_err, 0, 0);
        return NewList(0);
    }
    
    for (i = 0; i < GlobalBags.nr; i++) {
        if(findrefs_recursion(*GlobalBags.addr[i], hd, &result)) {
            if (result.mem_err) {
                Pr(mem_err, 0, 0);
                free(result.list);
                return NewList(0);
            }
        }
    }
    
    for (i = 0; i < GlobalBags.nr; i++) {
        RecursiveClearFlag(*GlobalBags.addr[i], BF_VISITED);
    }
    
    list = NewList((int)result.list_count);
    for (i = 0; i < result.list_count; i++) {
        SET_ELM_PLIST(list, i+1, result.list[i]);
    }
    
    free(result.list);
    
    return list;
}

Bag  FunFindRefs( Bag hdCall ) {
    char *      usage = "usage: FindRefs( <obj> [, <by value> = false])";
    int         by_value = 0;
    Obj         hd;
    
    switch(GET_SIZE_BAG(hdCall)) {
        case 3 * SIZE_HD: 
            by_value = EVAL(PTR_BAG(hdCall)[2]) == HdTrue;
        case 2 * SIZE_HD: {
            hd = (by_value) ? EVAL(PTR_BAG(hdCall)[1]) : PTR_BAG(hdCall)[1];
            return findrefs_global(hd);
            break;
        }
    }   
    return Error(usage, 0,0);
}


extern Bag       HdBases;
extern Bag*   _FindRecnameRec_nobases ( Bag hdRec, Bag hdField );


Bag  _ObjId ( Bag hd ) {
    UInt        t;

    t = GET_TYPE_BAG(hd);

    if(t == T_STRING) { 
	return hd;
    }
    else if(IS_LIST_TYPE(t)) {
        if(HdListClass==0) HdListClass = VAR_VALUE(FindIdent("ListClass"));
        return HdListClass;
    }
    else if(t == T_REC) {
        Bag * ptBases = _FindRecnameRec_nobases(hd, HdBases);
        if ( ptBases != 0 && ptBases[1] != 0 && LEN_PLIST(ptBases[1]) >= 1) {
            return ELM_PLIST(ptBases[1], 1);
        }
        else return hd;
    }
    else return hd;
}

Bag  Fun_ObjId ( Bag hdCall ) {
    char *      usage = "usage: ObjId(<obj>)";

    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    
    return _ObjId(EVAL(PTR_BAG(hdCall)[1]));
}


/****************************************************************************
**
*F  Exit(val=0) . . . . . . . . . . . .  Exit immediately returning value
**
**  FunExit calls system exit(val), default val = 0
**
**  This is useful in test scripts that need to return an exit status
**  to the invoking process.
*/


Bag FunExit( Bag hdCall ) {
	int val = 0;
	Obj hdVal;
	char* usage = "usage: Exit(<int>)";

	if (GET_SIZE_BAG(hdCall) > 2 * SIZE_HD) {
		return Error(usage, 0, 0);
	}
	else if (GET_SIZE_BAG(hdCall) == 2 * SIZE_HD) {
		hdVal= EVAL(PTR_BAG(hdCall)[1]);
		if (GET_TYPE_BAG(hdVal) != T_INT) {
			return Error(usage, 0, 0);
		}
		val = HD_TO_INT(hdVal);
	}

	SyExit(val);

	return HdVoid;
}



/****************************************************************************
**
*F  InitSPIRAL() . . . . . . . . . . . .  initializes SPIRAL related packages
**
**  'InitSPIRAL' initializes packages needed by SPIRAL
*/
extern void Init_sys_conf();
extern void Init_types();
extern void Init_Double();
extern void Init_Complex();

void            InitSPIRAL (void) {
    InitSPIRAL_Paths();
    InitSPIRAL_FFT();
    InitSPIRAL_SPLProg();
    InitSPIRAL_DelayEv();
    InitSPIRAL_BagList();
    Init_Double();
    Init_Complex();
    Init_types();
    /**/ GlobalPackage2("spiral", "util"); /**/
    InitGlobalBag( &HdListClass, "HdListClass" );
    InstIntFunc( "Apropos",          FunApropos );
    InstIntFunc( "SysVerbose",       FunSysVerbose );
    InstIntFunc( "BagInfo",          FunBagInfo );
    InstIntFunc( "BagAddr",          FunBagAddr );
    InstIntFunc( "BagType",          FunBagType );
    InstIntFunc( "BagFromAddr",      FunBagFromAddr );
    InstIntFunc( "BagFlags",         FunBagFlags );
    InstIntFunc( "BagBounds",        FunBagBounds );
    InstIntFunc( "BagsOfType",       FunBagsOfType );
    InstIntFunc( "TimeInSecs",       FunTimeInSecs );
    InstIntFunc( "TimezoneOffset",   FunTimezoneOffset );
    InstIntFunc( "TimeInMicroSecs",  FunTimeInMicroSecs );
    InstIntFunc( "InternalFunction", FunInternalFunction );
    InstIntFunc( "InternalHash",     FunInternalHash);
    InstIntFunc( "Blanks",           FunBlanks);
    InstIntFunc( "Assign",           FunAssign);
    InstIntFunc( "Declare",          FunDeclare);
    InstIntFunc( "IsVoid",           FunIsVoid);
    InstIntFunc( "ValueOf",          FunValueOf);
    InstIntFunc( "NameOf",           FunNameOf);
    InstIntFunc( "_GlobalVar",       Fun_GlobalVar);
    InstIntFunc( "Constraint",       FunConstraint);
    InstIntFunc( "Checked",          FunChecked);
    InstIntFunc( "When",             FunWhen);
    InstIntFunc( "Cond",             FunCond);
    InstIntFunc( "Same",             FunSame);
    InstIntFunc( "MD5File",          FunMD5File);
    InstIntFunc( "MD5String",        FunMD5String);
    InstIntFunc( "FileTime",         FunFileMTime);
    InstIntFunc( "GetPid",           FunGetPid);
    InstIntFunc( "MakeDir",          FunMakeDir);

    InstIntFunc( "WinGetValue",      FunWinGetValue);
    InstIntFunc( "WinPathFixSpaces", FunWinPathFixSpaces);
	InstIntFunc( "WinShortPathName", FunWinShortPathName);

    InstIntFunc( "SetEnv",  FunSetEnv );
    InstIntFunc( "GetEnv",  FunGetEnv );

    InstIntFunc( "Props",              FunProps);
    InstIntFunc( "Pkg",              FunPkg);
    InstIntFunc( "Reachability",              FunReachability);
    InstIntFunc( "PathRelativeToSPIRAL",  FunPathRelativeToSPIRAL );
	InstIntFunc( "Version", FunVersion);
	InstIntFunc( "BuildInfo", FunBuildInfo);
    IdentAssign( "NULL", HdVoid);
    IdentAssign( "HdStack", HdStack);
    InstIntFunc( "HdExec",  FunHdExec);
    InstIntFunc( "XType",  FunXType);
    InstIntFunc( "Try",  FunTry);
    InstIntFunc( "EditDef", FunEditDef);
    InstIntFunc( "FindRefs", FunFindRefs);
    
    InstIntFunc( "_ObjId", Fun_ObjId);

	InstIntFunc("Exit", FunExit);

    /**/ EndPackage(); /**/

    GuSysSetProgname("gap");
    GuSysSetExitFunc(ErrorExit);

    /**/ GlobalPackage2("spiral", "sys_conf"); /**/
	Init_sys_conf();
    /**/ EndPackage(); /**/
}

/****************************************************************************
**
*E  Emacs . . . . . . . . . . . . . . . . . . . . . . . local emacs variables
**
**  Local Variables:
**  c-basic-offset:     4
**  outline-regexp:     "*A\\|*F\\|*V\\|*T\\|*E"
**  fill-column:        76
**  fill-prefix:        "**  "
**  End:
*/
