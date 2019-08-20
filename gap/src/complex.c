#include        <stdio.h>
#include        <stdlib.h>
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
#include        "double.h"
#include        "complex.h"
#include        "namespaces.h"
#include		"GapUtils.h"

/**********************************************************************
**
*F  CPLX_RE_OBJ(<hd>) . . . . . . . . . . real part of a complex object
*F  CPLX_IM_OBJ(<hd>) . . . . . . .  imaginary part of a complex object
*F  ObjCplx(<double>, <double>) . . . . . . . . create a complex object
*F  CplxAny(<hd>) . . . . . . . . . . .  convert int/rat/cyc to complex
*/

Obj  ObjCplx(double re, double im) {
    Obj hd = NewBag(T_CPLX, 2*sizeof(double));
    RE(hd) = re;
    IM(hd) = im;
    return hd;
}


void _long_double_W(int n, int k, long double *p_re, long double *p_im) {
    static int NEGI = 1, NEGR = 2, SWAP = 4;
    static long double sqrt05  = 0.7071067811865475244L;
    static long double sqrt075 = 0.866025403784438646764L;
    static long double pi      = 3.141592653589793238462643383279L;

    int s, q, g, r;
    long double re, im;

    s = 0;
    if (n < 0) { k = -k; n = -n; } /* now 0 <= n */
    if (k < 0) { k = -k; s ^= NEGI; } /* now 0 <= k */
    k = k % n; /* now 0 <= k < n */
    if (2*k > n) { k = n-k; s ^= NEGI; } /* now 0 <= k <= n/2 */
    if (4*k > n) { k = n-2*k; n *= 2; s ^= NEGR; } /* 0 <= k <= n/4 */
    if (8*k > n) { k = n-4*k; n *= 4; s ^= SWAP; } /* 0 <= k <= n/8 */
    /* cancel gcd */
    for (q=k,g=n; q>0; ) {
        r = g % q;
        g = q;
        q = r;
    }
    if (g>1) { k /= g; n /= g; }
    if (k==0) { re = 1; im = 0; }
    else if (k==1 && n==8) { re = im = sqrt05; }
    else if (k==1 && n==12) { re = sqrt075; im = 0.5; }
    else {
        long double t = (2*pi*k)/n;
        re = cos(t);
        im = sin(t);
    }
    /* apply s:  */
    if (s & SWAP) { long double t=re; re=im; im=t; }
    if (s & NEGR) re = -re;
    if (s & NEGI) im = -im;

    *p_re = re;
    *p_im = im;
}


Obj CplxW(int n, int k) {
    long double r, i;
    _long_double_W(n,k,&r,&i);
    return ObjCplx(r, i);
}

Obj  CplxAny(Obj hd) {
    if(GET_TYPE_BAG(hd)==T_CPLX) return hd;
    else if(GET_TYPE_BAG(hd)==T_CYC) {
        Obj res = ObjCplx(0,0);
        int n = HD_TO_INT( PTR_BAG(hd)[0] );
        int len = GET_SIZE_BAG(hd)/(SIZE_HD+sizeof(unsigned short));
        int i;

        long double res_re=0.0, res_im=0.0;
        for ( i = 1; i < len; i++ ) {
            long double coeff = LongDblAny(PTR_BAG(hd)[i]);
            unsigned short pow = ((unsigned short*)(PTR_BAG(hd)+len))[i];
            long double re, im;
            _long_double_W(n, pow, &re, &im);
            res_re += re * coeff;
            res_im += im * coeff;
        }
        return ObjCplx(res_re, res_im);
    }
    else return ObjCplx(DblAny(hd), 0.0);
}


Obj  CplxAny2(Obj re, Obj im) {
    return ObjCplx(DblAny(re), DblAny(im));
}
/****************************************************************************
**
*F  FunStringComplex(<hdCall>)  . . . . . . . . . convert a double to a string
**
**  Implements internal function 'StringComplex'
**
**  'StringComplex( <format_string>, <int_pair> )'
**
**  'StringComplex' returns a formatted string created with sprintf called
**  with a format string and a double.
**
**  <format_string> must contain exactly two %.
**  Example: StringComplex("%f", cplx(1.0)) returns "1.000000",
**           since %f defaults to 6 digits of precision.
*/
Obj  FunStringComplex ( Obj hdCall ) {
    Obj  hdFmt, hdResult, hdCplx;
    double re, im;
    char * usage = "usage: StringComplex( <fmt_string>, <double> )";
    char * result;
    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD ) return Error(usage, 0, 0);
    hdFmt = EVAL( PTR_BAG(hdCall)[1] );
    hdCplx = CplxAny( EVAL( PTR_BAG(hdCall)[2] ) );
    re = RE(hdCplx);
    im = IM(hdCplx);
    if ( GET_TYPE_BAG(hdFmt) != T_STRING ) return Error(usage, 0, 0);

    /*alloc*/ result = GuMakeMessage(HD_TO_STRING(hdFmt), re, im);
    hdResult = NewBag( T_STRING, strlen(result)+1 );
    strncpy( HD_TO_STRING(hdResult), result, strlen(result)+1);
    /*free*/ free(result);
    return hdResult;
}

/**********************************************************************
**
*F  CplxSum(<hdL>,<hdR>) . . . . . . . . . . . . . sum of two doubles
*F  CplxDiff(<hdL>,<hdR>) . . . . . . . . . difference of two doubles
*F  CplxProd(<hdL>,<hdR>)  . . . . . . . . . . product of two doubles
*F  CplxQuo(<hdL>,<hdR>)  . . . . . . . . . . . . . . double quotient
*F  CplxPow(<hdL>,<hdR>) . . . . . . . . . . . . . . . . double power
**
*F  CplxAnySum(<hdL>,<hdR>)  . . . . . . . . sum of scalar and double
*F  CplxAnyDiff(<hdL>,<hdR>)  . . . . difference of scalar and double
*F  CplxAnyProd(<hdL>,<hdR>) . . . . . . product of scalar and double
*F  CplxAnyQuo(<hdL>,<hdR>)  . . . . . quotient of scalar and sdouble
*F  CplxAnyPow(<hdL>,<hdR>)  . . . . . . . power of scalar and double
**
*F  EvCplx(<hd>)  . . . . . . . . . . . . . . . . . . evaluate double
*F  PrCplx(<hd>) . . . . . . . . . . . . . . . . . . . print a double
**
*/
Obj  CplxSum  (Obj l, Obj r) { return ObjCplx(RE(l) + RE(r),
                                              IM(l) + IM(r)); }

Obj  CplxDiff  (Obj l, Obj r) { return ObjCplx(RE(l) - RE(r),
                                               IM(l) - IM(r)); }

Obj  CplxProd (Obj l, Obj r) { return ObjCplx(RE(l) * RE(r) - IM(l) * IM(r),
                                              RE(l) * IM(r) + IM(l) * RE(r)); }

Obj  CplxQuo  (Obj l, Obj r) {
    double re_l = RE(l), im_l = IM(l);
    double re_r = RE(r), im_r = IM(r);
    double denom = re_r * re_r  +  im_r * im_r;
    return ObjCplx( (re_l * re_r + im_l * im_r) / denom,
                    (im_l * re_r - re_l * im_r) / denom );
}

Obj  CplxPow  (Obj l, Obj r) {
    int pow;
    if(GET_TYPE_BAG(r) != T_INT)
        return Error("Only integer exponents allowed for complex numbers", 0, 0);
    pow = HD_TO_INT(r);
    if(pow == 0) return INT_TO_HD(1);
    else {
        double rr = RE(l), ii = IM(l);
        double r = rr, i = ii;
        int neg = 0;
        if(pow < 0) { neg = 1; pow = -pow; }

        for( ; pow > 1; --pow) {
            double newr = r*rr - i*ii;
            double newi = r*ii + i*rr;
            r = newr;
            i = newi;
        }

        if(neg) {
            double denom = r*r + i*i;
            return ObjCplx(r / denom, -i / denom);
        }
        else return ObjCplx(r, i);
    }
}

Obj  EqCplx    (Obj l, Obj r) { return (RE(l)==RE(r) && IM(l)==IM(r)) ? HdTrue : HdFalse; }
Obj  LtCplx    (Obj l, Obj r) { return (RE(l) < RE(r) || (RE(l)==RE(r) && IM(l)<IM(r))) ? HdTrue : HdFalse; }

Obj  CplxAnySum  (Obj l, Obj r) { return CplxSum(CplxAny(l), CplxAny(r)); }
Obj  CplxAnyDiff (Obj l, Obj r) { return CplxDiff(CplxAny(l), CplxAny(r)); }
Obj  CplxAnyProd (Obj l, Obj r) { return CplxProd(CplxAny(l), CplxAny(r)); }
Obj  CplxAnyQuo  (Obj l, Obj r) { return CplxQuo(CplxAny(l), CplxAny(r)); }
Obj  CplxAnyPow  (Obj l, Obj r) { return CplxPow(CplxAny(l), r); }

Obj _toRealMaybe(Obj o) {
    if(IM(o)==0)
        return ObjDbl(RE(o));
    else return o;
}

Obj  DblCplxAnySum  (Obj l, Obj r) { return _toRealMaybe(CplxSum(CplxAny(l), CplxAny(r))); }
Obj  DblCplxAnyDiff (Obj l, Obj r) { return _toRealMaybe(CplxDiff(CplxAny(l), CplxAny(r))); }
Obj  DblCplxAnyProd (Obj l, Obj r) { return _toRealMaybe(CplxProd(CplxAny(l), CplxAny(r))); }
Obj  DblCplxAnyQuo  (Obj l, Obj r) { return _toRealMaybe(CplxQuo(CplxAny(l), CplxAny(r))); }
Obj  DblCplxAnyPow  (Obj l, Obj r) { return _toRealMaybe(CplxPow(CplxAny(l), r)); }


Obj  EqCplxAny (Obj l, Obj r) { return EqCplx(CplxAny(l), CplxAny(r)); }
Obj  LtCplxAny (Obj l, Obj r) { return LtCplx(CplxAny(l), CplxAny(r)); }


Obj  EvCplx ( Obj hd ) { return hd; }

void PrCplx ( Obj hd ) {
    char buf[64];
    /* using snprintf prevents any buffer overflowing */
    snprintf(buf, sizeof(buf)/sizeof(char), "Cplx(%.17g, %.17g)", RE(hd), IM(hd));
    Pr("%s", (Int)buf, 0);
}

/****************************************************************************
**
*F  FunComplex() . . . . . . . . . . . . implements internal function Complex()
*F  FunIsComplex() . . . . . . . . . . implements internal function IsComplex()
**
*/
Obj FunComplex ( Obj hdCall ) {
    char * usage = "usage: Complex( <num> )";

    if ( GET_SIZE_BAG(hdCall) == 2 * SIZE_HD ) {
        Obj hd = EVAL( PTR_BAG(hdCall)[1] );
        return CplxAny(hd);
    }
    else if ( GET_SIZE_BAG(hdCall) == 3 * SIZE_HD ) {
        Obj re = EVAL( PTR_BAG(hdCall)[1] );
        Obj im = EVAL( PTR_BAG(hdCall)[2] );
        return CplxAny2(re, im);
    }
    else return Error(usage, 0,0);
}

Obj FunIsComplex ( Obj hdCall ) {
    char * usage = "usage: IsComplex( <obj> )";
    Obj hd;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hd = EVAL( PTR_BAG(hdCall)[1] );
    return GET_TYPE_BAG(hd)==T_CPLX ? HdTrue : HdFalse;
}

Obj FunReComplex ( Obj hdCall ) {
    char * usage = "usage: ReComplex( <complex> )";
    Obj hd; UInt t;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hd = EVAL( PTR_BAG(hdCall)[1] );
    t = GET_TYPE_BAG(hd);
    if ( t == T_DOUBLE ) return hd;
    else if (t == T_CPLX) return ObjDbl(RE(hd));
    else return Error(usage, 0, 0);
}

Obj FunImComplex ( Obj hdCall ) {
    char * usage = "usage: ImComplex( <complex> )";
    Obj hd; UInt t;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hd = EVAL( PTR_BAG(hdCall)[1] );
    t = GET_TYPE_BAG(hd);
    if ( t == T_DOUBLE ) return ObjDbl(0.0);
    else if (t == T_CPLX) return ObjDbl(IM(hd));
    else return Error(usage, 0, 0);
}

Obj FunAbsComplex ( Obj hdCall ) {
    char * usage = "usage: AbsComplex( <complex> )";
    Obj hd; UInt t;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hd = EVAL( PTR_BAG(hdCall)[1] );
    t = GET_TYPE_BAG(hd);
    if ( t == T_DOUBLE ) {
        if (t < 0.0) return ObjDbl(-DBL_OBJ(hd)); else return hd;
    }
    else if (t == T_CPLX) {
        double re,im;
        re = RE(hd);
        im = IM(hd);
        return ObjDbl(sqrt(re*re + im*im));
    }
    else return Error(usage, 0, 0);
}


/****************************************************************************
**
*F  ComplexW( <n>, <pow> )  . . . . . . . . . . . . . . complex root of unity
**
** Implements internal function 'ComplexW' which returns a complex root of
** unity of order <n>, raised to the power <pow>.
*/
Obj  FunComplexW ( Obj hdCall ) {
    char * usage = "usage: ComplexW( <n>, <pow> )";
    Obj hdN, hdPow;
    int n, pow;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD ) return Error(usage, 0,0);
    hdN = EVAL( PTR_BAG(hdCall)[1] );
    hdPow = EVAL( PTR_BAG(hdCall)[2] );
    if( GET_TYPE_BAG(hdN) != T_INT ) return Error(usage,0,0);
    if( GET_TYPE_BAG(hdPow) != T_INT ) return Error(usage,0,0);

    n = HD_TO_INT(hdN);
    pow = HD_TO_INT(hdPow);
    return CplxW(n, pow);
}


/****************************************************************************
**
*F  Init_Complex() . . . . . . . . . . . . . . . initialize the Complex package
*/
void Init_Complex(void) {
    unsigned int        type;

    /**/ GlobalPackage2("gap", "complex"); /**/
    InstEvFunc(T_CPLX, EvCplx);
    InstPrFunc(T_CPLX, PrCplx);
    InstIntFunc( "Cplx",    FunComplex );
    InstIntFunc( "Complex",    FunComplex );
    InstIntFunc( "IsComplex",  FunIsComplex );
    InstIntFunc( "StringComplex",  FunStringComplex );
    InstIntFunc( "ComplexW",    FunComplexW );
    InstIntFunc( "ReComplex",    FunReComplex );
    InstIntFunc( "ImComplex",    FunImComplex );
    InstIntFunc( "AbsComplex",    FunAbsComplex );

    SET_BAG(FindIdentWr("c_I"), 0, ObjCplx(0, 1));

    for ( type = T_INT; type <= T_CYC; ++type ) {
        TabSum [T_CPLX][type] = TabSum [type][T_CPLX] = CplxAnySum;
        TabDiff[T_CPLX][type] = TabDiff[type][T_CPLX] = CplxAnyDiff;
        TabProd[T_CPLX][type] = TabProd[type][T_CPLX] = CplxAnyProd;
        TabQuo [T_CPLX][type] = TabQuo [type][T_CPLX] = CplxAnyQuo;
        TabPow [T_CPLX][type] = TabPow [type][T_CPLX] = CplxAnyPow;
        TabEq  [T_CPLX][type] = TabEq  [type][T_CPLX] = EqCplxAny;
        TabLt  [T_CPLX][type] = TabLt  [type][T_CPLX] = LtCplxAny;
    }
    TabSum [T_DOUBLE][T_CYC] = TabSum [T_CYC][T_DOUBLE] = DblCplxAnySum;
    TabDiff[T_DOUBLE][T_CYC] = TabDiff[T_CYC][T_DOUBLE] = DblCplxAnyDiff;
    TabProd[T_DOUBLE][T_CYC] = TabProd[T_CYC][T_DOUBLE] = DblCplxAnyProd;
    TabQuo [T_DOUBLE][T_CYC] = TabQuo [T_CYC][T_DOUBLE] = DblCplxAnyQuo;
    TabPow [T_DOUBLE][T_CYC] = TabPow [T_CYC][T_DOUBLE] = DblCplxAnyPow;

    TabSum [T_DOUBLE][T_CPLX] = TabSum [T_CPLX][T_DOUBLE] = CplxAnySum;
    TabDiff[T_DOUBLE][T_CPLX] = TabDiff[T_CPLX][T_DOUBLE] = CplxAnyDiff;
    TabProd[T_DOUBLE][T_CPLX] = TabProd[T_CPLX][T_DOUBLE] = CplxAnyProd;
    TabQuo [T_DOUBLE][T_CPLX] = TabQuo [T_CPLX][T_DOUBLE] = CplxAnyQuo;
    TabPow [T_DOUBLE][T_CPLX] = TabPow [T_CPLX][T_DOUBLE] = CplxAnyPow;

    TabSum [T_CPLX][T_CPLX] = CplxSum;
    TabDiff[T_CPLX][T_CPLX] = CplxDiff;
    TabProd[T_CPLX][T_CPLX] = CplxProd;
    TabQuo [T_CPLX][T_CPLX] = CplxQuo;
    TabPow [T_CPLX][T_CPLX] = CplxPow;
    TabEq  [T_CPLX][T_CPLX] = EqCplx;
    TabLt  [T_CPLX][T_CPLX] = LtCplx;

    for ( type = T_LIST; type <= T_LISTX; type++ ) {
        if ( type != T_STRING && type != T_REC && type != T_BLIST ) {
            TabSum [T_CPLX][type   ] = SumSclList;
            TabSum [type   ][T_CPLX] = SumListScl;
            TabDiff[T_CPLX][type   ] = DiffSclList;
            TabDiff[type   ][T_CPLX] = DiffListScl;
            TabProd[T_CPLX][type   ] = ProdSclList;
            TabProd[type   ][T_CPLX] = ProdListScl;
            TabQuo [type   ][T_CPLX] = QuoLists;
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
