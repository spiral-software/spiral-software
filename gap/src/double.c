#include        <stdio.h>
#include		<stdlib.h>
#include		<string.h>
#include        <math.h>

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* Bag, NewBag, T_STRING, .. */
#include        "objects.h"
#include		"string4.h"
#include        "integer.h"             /* arbitrary size integers         */
#include        "integer4.h"            /* QuoInt */
#include        "rational.h"
#include        "eval.h"
#include        "idents.h"
#include        "scanner.h"             /* Pr()                            */
#include        "list.h"
#include        "args.h"
#include        "ieee754.h"      /* union ieee754_double */
#include        "namespaces.h"
#include        "double.h"
#include		"GapUtils.h"

/**********************************************************************
**
*F  DBL_INT(<hd>)  . . . . . . . convert small integer object to double
*F  DBL_OBJ(<hd>) . . . . . . . . . . . convert double object to double
*F  ObjDbl(<double>) . . . . . . . . create a double object from double
*F  DblAny(<hd>) . . . . . convert int/rational/string object to double
*/

Obj  ObjDbl(double d) {
    Obj hd = NewBag(T_DOUBLE, sizeof(double));
    *((double*)PTR_BAG(hd)) = d;
    return hd;
}

double DblString(char *st) {
    double res=0.0;
    int k = sscanf(st, "%lf", &res);
    if(k==0) Error("DblAny: String '%s' can't be parsed as a double",
                   (Int)st, 0);
    return res;
}


long double LongDblString(char *st) {
    long double res=0.0;
    int k = sscanf(st, "%Lf", &res);
    if(k==0) Error("LongDblAny: String '%s' can't be parsed as a long double",
                   (Int)st, 0);
    return res;
}


double DblAny(Obj hd) {
    switch(GET_TYPE_BAG(hd)) {
    case T_DOUBLE:
        return DBL_OBJ(hd);
    case T_INT:
        return DBL_INT(hd);
    case T_STRING:
        return DblString((char*)PTR_BAG(hd));
    case T_INTPOS:
    case T_INTNEG:
        return DblString((char*)PTR_BAG(StringInt(hd)));
    case T_RAT:
        { Obj nom = PTR_BAG(hd)[0];
          Obj den = PTR_BAG(hd)[1];
          return DblAny(nom) / DblAny(den); }
    default:
        Error("DblAny: Can't convert %s to a double", (Int)TNAM_BAG(hd), 0);
        return 0.0; /* never reached */
    }
}


long double LongDblAny(Obj hd) {
    switch(GET_TYPE_BAG(hd)) {
    case T_DOUBLE:
        return (long double) DBL_OBJ(hd);
    case T_INT:
        return LONG_DBL_INT(hd);
    case T_STRING:
        return LongDblString((char*)PTR_BAG(hd));
    case T_INTPOS:
    case T_INTNEG:
        return LongDblString((char*)PTR_BAG(StringInt(hd)));
    case T_RAT:
        { Obj nom = PTR_BAG(hd)[0];
          Obj den = PTR_BAG(hd)[1];
          return LongDblAny(nom) / LongDblAny(den); }
    default:
        Error("LongDblAny: Can't convert %s to a long double", (Int)TNAM_BAG(hd), 0);
        return 0.0; /* never reached */
    }
}

/****************************************************************************
**
*F  FunRatDouble() . . . . . . . . . . . converts double to an exact rational
**
**  Implements internal function 'RatDouble'. Note that no precision is lost,
**  since GAP allows infinite-precision integers in numerator/denominator.
**
**  RatDouble(<double>)
*/

Obj RatDouble ( double d ) {
    union ieee754_double dbl;
    Int mantissa0;
    UInt mantissa1;
    Int exponent;

    Obj hdMantissa;
    Obj hdExponent;
    Obj hdResult;

    dbl.d = d;

    /* GAP can accept only 29-bit signed integers as immediate */
    mantissa1 = 0x0FFFFFFF & dbl.ieee.mantissa1; /* low 28 bits (unsigned) */

    /* the remaining 4 bits of mantissa1 go here, also the leading 1 is    */
    /* implicit, so it has to be added                                     */
    mantissa0 = ((0xF0000000 & dbl.ieee.mantissa1) >> 28) +
                (dbl.ieee.mantissa0 << 4) +
                (1 << 24);

    exponent  = dbl.ieee.exponent - IEEE754_DOUBLE_BIAS;

    /* signed 2^28 is 30 bits, so we have to multiply by 2^27 and then by 2*/
    /* to get it                                                           */
    hdMantissa = INT_TO_HD(mantissa0);
    hdMantissa = ProdInt( ProdInt(hdMantissa, INT_TO_HD(1<<27)), INT_TO_HD(2) );
    hdMantissa = SumInt( hdMantissa, INT_TO_HD(mantissa1) );
    if(dbl.ieee.negative)
        hdMantissa = ProdInt( hdMantissa, INT_TO_HD(-1) );

    hdExponent = INT_TO_HD(-52+exponent);

    hdResult = ProdRat(hdMantissa, POW(INT_TO_HD(2), hdExponent));
    return hdResult;
}

Obj  FunRatDouble ( Obj hdCall ) {
    char * usage = "usage: RatDouble( <double> )";
    Obj hd;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hd = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hd) != T_DOUBLE) return Error(usage, 0, 0);
    return RatDouble(DBL_OBJ(hd));
}

/****************************************************************************
**
*F  FunIntDouble() . . . . . . . . . . . . . truncates a double to an integer
**
**  Implements internal function 'IntDouble'.
**
**  IntDouble(<double>)
*/
Obj  FunIntDouble ( Obj hdCall )
{
    char * usage = "usage: IntDouble( <double> )";
    Obj hd;

    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
                return Error(usage, 0,0);

    hd = EVAL( PTR_BAG(hdCall)[1] );

    if ( GET_TYPE_BAG(hd) != T_DOUBLE)
                return Error(usage, 0, 0);

    if ( fabs(DBL_OBJ(hd)) >= (1<<28) )
        {
                Obj rat = RatDouble(DBL_OBJ(hd));
                int t = GET_TYPE_BAG(rat);

                if ( t==T_INTPOS || t==T_INTNEG || t==T_INT )
                        return rat;

                else if ( t == T_RAT )
                        return QuoInt(PTR_BAG(rat)[0], PTR_BAG(rat)[1]);

                else
                        assert(0);

                return NULL;
    }
    else
                return INT_TO_HD( (Int)DBL_OBJ(hd) );
}

/****************************************************************************
**
*F  FunDoubleRep64() . . . . . . explicit 64-bit representation of a double
**
**  Implements internal functions 'DoubleRep64'
**
**  DoubleRep64(<double>)
**  DoubleRep64(<double>)
*/

Obj  FunDoubleRep64 ( Obj hdCall ) {
        char usage[] = "usage: DoubleRep64( <double> )";
    Obj hd, hdRes = NULL;

        double d;
    UInt8 rep;
    UInt4 k; Obj K;

    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);

        hd = EVAL( PTR_BAG(hdCall)[1] );

    if ( GET_TYPE_BAG(hd) != T_DOUBLE) return Error(usage, 0, 0);

    d = DBL_OBJ(hd);
    rep = *((UInt8*)&d);

    k = 1 << 28; // on 32-bit machines GAP small integer can hold 28 bit values
    K = INT_TO_HD(k);

    hd = INT_TO_HD(rep % k);
        hdRes = PROD(INT_TO_HD((rep >> 28) %k), K);
        hdRes = SUM(hd, hdRes);
        hd = PROD(INT_TO_HD((rep >> 56) % k), K);
        hd = PROD(hd, K);
        hdRes = SUM(hdRes, hd);
/*
        this statement does not compile in MS VC++, it has been split up into multiple
        statements, and now works.

        hdRes = SUM(      SUM(INT_TO_HD((rep      ) % k),
                     PROD(INT_TO_HD((rep >> 28) % k), K)),
                PROD(PROD(INT_TO_HD((rep >> 56) % k), K), K));
*/
        return hdRes;
}

/****************************************************************************
**
*F  FunDoubleString() . . . . . . . . . . . . . parses a string into a double
**
**  Implements internal function 'DoubleString'.
**
**  DoubleString(<double>)
*/
Obj  FunDoubleString ( Obj hdCall ) {
    char * usage = "usage: DoubleString( <string> )";
    Obj hd;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hd = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hd) != T_STRING) return Error(usage, 0, 0);
    return ObjDbl(DblString(HD_TO_STRING(hd)));
}

/****************************************************************************
**
*F  FunStringDouble(<hdCall>)  . . . . . . . . . convert a double to a string
**
**  Implements internal function 'StringDouble'
**
**  'StringDouble( <format_string>, <int_pair> )'
**
**  'StringDouble' returns a formatted string created with sprintf called
**  with a format string and a double.
**
**  <format_string> should not contain more than one %.
**  Example: StringDouble("%f", dbl(1.0)) returns "1.000000",
**           since %f defaults to 6 digits of precision.
*/
Obj  FunStringDouble ( Obj hdCall ) {
    Obj  hdFmt, hdResult;
    double d;
    char * usage = "usage: StringDouble( <fmt_string>, <double> )";
    char * result;
    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD ) return Error(usage, 0, 0);
    hdFmt = EVAL( PTR_BAG(hdCall)[1] );
    d = DblAny( EVAL( PTR_BAG(hdCall)[2] ) );
    if ( GET_TYPE_BAG(hdFmt) != T_STRING ) return Error(usage, 0, 0);

    /*alloc*/ result = GuMakeMessage(HD_TO_STRING(hdFmt), d);
    hdResult = NewBag( T_STRING, strlen(result)+1 );
    strncpy( HD_TO_STRING(hdResult), result, strlen(result)+1);
    /*free*/ free(result);
    return hdResult;
}



/**********************************************************************
**
*F  DblSum(<hdL>,<hdR>) . . . . . . . . . . . . . sum of two doubles
*F  DblDiff(<hdL>,<hdR>) . . . . . . . . . difference of two doubles
*F  DblProd(<hdL>,<hdR>)  . . . . . . . . . . product of two doubles
*F  DblQuo(<hdL>,<hdR>)  . . . . . . . . . . . . . . double quotient
*F  DblPow(<hdL>,<hdR>) . . . . . . . . . . . . . . . . double power
**
*F  DblAnySum(<hdL>,<hdR>)  . . . . . . . . sum of scalar and double
*F  DblAnyDiff(<hdL>,<hdR>)  . . . . difference of scalar and double
*F  DblAnyProd(<hdL>,<hdR>) . . . . . . product of scalar and double
*F  DblAnyQuo(<hdL>,<hdR>)  . . . . . quotient of scalar and sdouble
*F  DblAnyPow(<hdL>,<hdR>)  . . . . . . . power of scalar and double
**
*F  EvDbl(<hd>)  . . . . . . . . . . . . . . . . . . evaluate double
*F  PrDbl(<hd>) . . . . . . . . . . . . . . . . . . . print a double
**
*/
Obj  DblSum  (Obj l, Obj r) { return ObjDbl(DBL_OBJ(l) + DBL_OBJ(r)); }
Obj  DblDiff (Obj l, Obj r) { return ObjDbl(DBL_OBJ(l) - DBL_OBJ(r)); }
Obj  DblProd (Obj l, Obj r) { return ObjDbl(DBL_OBJ(l) * DBL_OBJ(r)); }
Obj  DblQuo  (Obj l, Obj r) { return ObjDbl(DBL_OBJ(l) / DBL_OBJ(r)); }
Obj  DblPow  (Obj l, Obj r) { return ObjDbl(pow(DBL_OBJ(l), DBL_OBJ(r))); }

Obj  DblAnySum  (Obj l, Obj r) { return ObjDbl(DblAny(l) + DblAny(r)); }
Obj  DblAnyDiff (Obj l, Obj r) { return ObjDbl(DblAny(l) - DblAny(r)); }
Obj  DblAnyProd (Obj l, Obj r) { return ObjDbl(DblAny(l) * DblAny(r)); }
Obj  DblAnyQuo  (Obj l, Obj r) { return ObjDbl(DblAny(l) / DblAny(r)); }
Obj  DblAnyPow  (Obj l, Obj r) { return ObjDbl(pow(DblAny(l), DblAny(r))); }

Obj  EqDbl    (Obj l, Obj r) { return (DBL_OBJ(l) ==DBL_OBJ(r)) ? HdTrue : HdFalse; }
Obj  LtDbl    (Obj l, Obj r) { return (DBL_OBJ(l) < DBL_OBJ(r)) ? HdTrue : HdFalse; }

Obj  EqDblAny (Obj l, Obj r) { return (DblAny(l) ==DblAny(r)) ? HdTrue : HdFalse; }
Obj  LtDblAny (Obj l, Obj r) { return (DblAny(l) < DblAny(r)) ? HdTrue : HdFalse; }

Obj  EvDbl ( Obj hd ) { return hd; }



void PrDbl ( Obj hd ) {
    char buf[30];
    double n = DBL_OBJ(hd);
    double intpart;
#ifdef _MSC_VER
   /* ugly hackery */
	static long long __ininity = 0x7FF0000000000000;
    double inf = *((double*)(&__ininity));
#else
    double zero = 0.0;
    double inf = 1/zero; /* portable way to obtain a double infinity value */
#endif
    /* using snprintf prevents any buffer overflowing */
    if (n!=inf && n!=-inf && modf(n, &intpart)==0.0)
        snprintf(buf, sizeof(buf)/sizeof(char), "%.17g.0", DBL_OBJ(hd));
    else
        snprintf(buf, sizeof(buf)/sizeof(char), "%.17g", DBL_OBJ(hd));
    Pr("%s", (Int)buf, 0);
}

/****************************************************************************
**
*F  FunDouble() . . . . . . . . . . . . implements internal function Double()
*F  FunIsDouble() . . . . . . . . . . implements internal function IsDouble()
**
*/
Obj FunDouble ( Obj hdCall ) {
    char * usage = "usage: Double( <rational> )";
    Obj hd;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hd = EVAL( PTR_BAG(hdCall)[1] );
    return ObjDbl( DblAny(hd) );
}

Obj FunIsDouble ( Obj hdCall ) {
    char * usage = "usage: IsDouble( <obj> )";
    Obj hd;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hd = EVAL( PTR_BAG(hdCall)[1] );
    return GET_TYPE_BAG(hd)==T_DOUBLE ? HdTrue : HdFalse;
}

/****************************************************************************
**
*F  Func1Dbl(<name>, <fPtr>, <hdCall>) . .  C function with 1 arg wrapper
**
**  'Func1Dbl' is a  wrapper for C  math functions  with 1 argument.
*/
Obj Func1Dbl ( char * name, double (*fPtr)(double), Obj hdCall ) {
    char * usage = "usage: Func1Dbl( <double> )";
    Obj  hd;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD) return Error(usage, 0, 0);
    hd = EVAL( PTR_BAG(hdCall)[1] );
    return ObjDbl( fPtr(DblAny(hd)) );
}

Obj Func2Dbl ( char * name, double (*fPtr)(double,double), Obj hdCall ) {
    char * usage = "usage: Func2Dbl( <double>, <double> )";
    Obj  hd1, hd2;
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD) return Error(usage, 0, 0);
    hd1 = EVAL( PTR_BAG(hdCall)[1] );
    hd2 = EVAL( PTR_BAG(hdCall)[2] );
    return ObjDbl( fPtr(DblAny(hd1), DblAny(hd2)) );
}

/* This macro expands to a function definition FunCMath_name, that
   implements GAP internal function CMath_name. It is a wrapper around
   C math function 'name' with one double argument, and double return
   value */
#define FUNC1(name) \
Obj  FunDbl_##name (Obj hdCall) { return Func1Dbl("", &name, hdCall); }

/* Same as FUNC1 but for C functions with 2 double arguments. */
#define FUNC2(name) \
Obj  FunDbl_##name (Obj hdCall) { return Func2Dbl("", &name, hdCall); }

/* Define internal GAP function wrappers */
FUNC1(exp); /* exponential of xdouble */
FUNC1(log);  /* natural logarithm of x */
FUNC1(log10) /* base-10 logarithm of x */
FUNC2(pow);  /* x raised to power y*/
FUNC1(sqrt); /* square root of x */
FUNC1(ceil); /* smallest integer not less than x */
FUNC1(floor); /* largest integer not greater than x */
FUNC1(fabs); /* absolute value of x */
FUNC2(fmod); /* if y non-zero, floating-point remainder of x/y, with same sign as x;
                if y zero, result is implementation-defined */
FUNC1(sin);  /* sine of x */
FUNC1(cos);  /* cosine of x */
FUNC1(tan);  /* tangent of x */
FUNC1(asin); /* arc-sine of x */
FUNC1(acos); /* arc-cosine of x */
FUNC1(atan); /* arc-tangent of x */
FUNC2(atan2); /* arc-tangent of y/x */
FUNC1(sinh); /* hyperbolic sine of x */
FUNC1(cosh); /* hyperbolic cosine of x */
FUNC1(tanh); /* hyperbolic tangent of x */

/* Unimplemented math calls:
double ldexp(double x, int n);   // x times 2 to the power n
double frexp(double x, int* exp); // if x non-zero, returns value,
   with absolute value in interval [1/2, 1), and assigns to *exp integer
   such that product of return value and 2 raised to the power *exp equals
   x; if x zero, both return value and *exp are zero
double modf(double x, double* ip); // returns fractional part and assigns
   to *ip integral part of x, both with same sign as x
*/

/****************************************************************************
**
*F  Init_Double() . . . . . . . . . . . . . . . initialize the Double package
*/
void Init_Double(void) {
    unsigned int        type;

    /**/ GlobalPackage2("gap", "double"); /**/
    InstEvFunc(T_DOUBLE, EvDbl);
    InstPrFunc(T_DOUBLE, PrDbl);
    InstIntFunc( "Double",    FunDouble );
    InstIntFunc( "IsDouble",  FunIsDouble );

    InstIntFunc( "DoubleString",    FunDoubleString );
    InstIntFunc( "StringDouble",    FunStringDouble );
    InstIntFunc( "IntDouble",    FunIntDouble );
    InstIntFunc( "RatDouble",    FunRatDouble );

    InstIntFunc( "DoubleRep64",    FunDoubleRep64 );

    InstIntFunc("d_exp",    FunDbl_exp);
    InstIntFunc("d_log",    FunDbl_log);
    InstIntFunc("d_log10",  FunDbl_log10);
    InstIntFunc("d_pow",    FunDbl_pow);
    InstIntFunc("d_sqrt",   FunDbl_sqrt);
    InstIntFunc("d_ceil",   FunDbl_ceil);
    InstIntFunc("d_floor",  FunDbl_floor);
    InstIntFunc("d_fabs",   FunDbl_fabs);
    InstIntFunc("d_fmod",   FunDbl_fmod);

    InstIntFunc("d_sin",    FunDbl_sin);
    InstIntFunc("d_cos",    FunDbl_cos);
    InstIntFunc("d_tan",    FunDbl_tan);
    InstIntFunc("d_asin",   FunDbl_asin);
    InstIntFunc("d_acos",   FunDbl_acos);
    InstIntFunc("d_atan",   FunDbl_atan);
    InstIntFunc("d_atan2",  FunDbl_atan2);
    InstIntFunc("d_sinh",   FunDbl_sinh);
    InstIntFunc("d_cosh",   FunDbl_cosh);
    InstIntFunc("d_tanh",   FunDbl_tanh);

    SET_BAG(FindIdentWr("d_PI"), 0, ObjDbl(M_PI));

    for ( type = T_INT; type <= T_RAT; ++type ) {
        TabSum [T_DOUBLE][type] = TabSum [type][T_DOUBLE] = DblAnySum;
        TabDiff[T_DOUBLE][type] = TabDiff[type][T_DOUBLE] = DblAnyDiff;
        TabProd[T_DOUBLE][type] = TabProd[type][T_DOUBLE] = DblAnyProd;
        TabQuo [T_DOUBLE][type] = TabQuo [type][T_DOUBLE] = DblAnyQuo;
        TabPow [T_DOUBLE][type] = TabPow [type][T_DOUBLE] = DblAnyPow;
        TabEq  [T_DOUBLE][type] = TabEq  [type][T_DOUBLE] = EqDblAny;
        TabLt  [T_DOUBLE][type] = TabLt  [type][T_DOUBLE] = LtDblAny;
    }

    TabSum [T_DOUBLE][T_DOUBLE] = DblSum;
    TabDiff[T_DOUBLE][T_DOUBLE] = DblDiff;
    TabProd[T_DOUBLE][T_DOUBLE] = DblProd;
    TabQuo [T_DOUBLE][T_DOUBLE] = DblQuo;
    TabPow [T_DOUBLE][T_DOUBLE] = DblPow;
    TabEq  [T_DOUBLE][T_DOUBLE] = EqDbl;
    TabLt  [T_DOUBLE][T_DOUBLE] = LtDbl;

    for ( type = T_LIST; type <= T_LISTX; type++ ) {
        if ( type != T_STRING && type != T_REC && type != T_BLIST ) {
            TabSum [T_DOUBLE][type   ] = SumSclList;
            TabSum [type   ][T_DOUBLE] = SumListScl;
            TabDiff[T_DOUBLE][type   ] = DiffSclList;
            TabDiff[type   ][T_DOUBLE] = DiffListScl;
            TabProd[T_DOUBLE][type   ] = ProdSclList;
            TabProd[type   ][T_DOUBLE] = ProdListScl;
            TabQuo [type   ][T_DOUBLE] = QuoLists;
        }
    }
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
