#define RE(hd) (((double*)PTR_BAG(hd))[0])
#define IM(hd) (((double*)PTR_BAG(hd))[1])

void Init_Complex();

Obj  ObjCplx(double re, double im);
Obj  CplxW(int n, int pow);
Obj  CplxAny(Obj hd);

Obj  CplxSum  (Obj l, Obj r) ;
Obj  CplxDiff  (Obj l, Obj r);
Obj  CplxProd (Obj l, Obj r) ;
Obj  CplxQuo  (Obj l, Obj r) ;
Obj  CplxPow  (Obj l, Obj r) ;
Obj  EqCplx    (Obj l, Obj r);
Obj  LtCplx    (Obj l, Obj r);
Obj  CplxAnySum  (Obj l, Obj r);
Obj  CplxAnyDiff (Obj l, Obj r);
Obj  CplxAnyProd (Obj l, Obj r);
Obj  CplxAnyQuo  (Obj l, Obj r);
Obj  CplxAnyPow  (Obj l, Obj r);
Obj  EqCplxAny (Obj l, Obj r) ;
Obj  LtCplxAny (Obj l, Obj r) ;
Obj  EvCplx ( Obj hd ) ;
void PrCplx ( Obj hd );
