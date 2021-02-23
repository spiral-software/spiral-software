
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#
# Functions
#
IsASPFreqBasis := x -> IsRec(x) and IsBound(x.isASPFreqBasis) and x.isASPFreqBasis;
IsASPTimeBasis := x -> IsRec(x) and IsBound(x.isASPTimeBasis) and x.isASPTimeBasis;

Class(BasisOps, PrintOps, rec(
    \= := (b1, b2) -> Cond(
        not IsASPFreqBasis(b1) or not IsASPFreqBasis(b2), false,
        ObjId(b1)=ObjId(b2) and b1.rChildren() = b2.rChildren())
));

#
# Base classes
#
Class(ASPTimeBasis, rec(
    isAtomic := true, # NOTE: these probably should not be atomic, maybe use TimeComp(Freq_E) for composed bases
    isASPTimeBasis := true,
    doNormalize := false,
    operations := PrintOps,
    print := self >> Print(self.__name__, When(self.doNormalize, ".norm()")),
    norm := self >> CopyFields(self, rec(doNormalize:=true)),
    rChildren := self >> [],
    from_rChildren := (self, rch) >> self,
    unparse := "ConstClass",
)); 

Class(ASPComposedTimeBasis, ASPTimeBasis, rec(
    composedFrom := self >> Error("must be defined in the subclass"),
    toX := (self, alg) >> Checked(ObjId(alg) = XN_skew, IsEvenInt(alg.n), 
        let(C := When(self.doNormalize, self.composedFrom().norm(), 
                                        self.composedFrom()),
            Tensor(C.to1X(XN_skew(2, alg.a, alg.rot)), I(alg.n/2))))
));

Class(ASPFreqBasis, rec(
    doNormalize := false,
    isASPFreqBasis := true,
    dim := self >> Error("must be defined in the subclass"),
    from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), rch),
    unparse := "RewritableObjectExp",
));

Class(ASPFreqBasis1d,  ASPFreqBasis, rec(
    __call__ := (self, scale1d) >> WithBases(self, rec(
            scale1d := scale1d, operations := BasisOps
    )),
    rChildren := self >> [self.scale1d],
    rSetChild := rSetChildFields("scale1d"),
    print := self >> Print(self.name, "(", self.scale1d, ")", When(self.doNormalize, ".norm()")),
    dim := self >> 1,
    hashAs := self >> ObjId(self)(1)
)); 

Class(ASPFreqBasis1d2d,  ASPFreqBasis, rec(
    __call__ := (self, scale1d, scale2d) >> WithBases(self, rec(
            scale1d := scale1d, scale2d := scale2d, operations := BasisOps
    )),
    rChildren := self >> [self.scale1d, self.scale2d],
    rSetChild := rSetChildFields("scale1d", "scale2d"),
    print := self >> Print(self.name, "(", self.scale1d, ", ", self.scale2d, ")",
        When(self.doNormalize, ".norm()")),

    norm := self >> CopyFields(self, rec(doNormalize:=true)),

    dim := self >> 2,
    hashAs := self >> ObjId(self)(1, 1),

    to1X_2d0 := (self, rot) >> I(2),
    to1X_2d := (self, a, rot) >> Error("Must be implemented in a subclass"),
    to1X := (self, alg) >> let(t := ObjId(alg), s1 := self.scale1d, s2 := self.scale2d, Cond(
        alg.n=1, 1/s1 * I(1),

        alg.n=2 and (t in [XN_min_1, XN_min_1U]),
            1/s2 * self.to1X_2d0(alg.rot),

        alg.n=2 and (t in [XN_skew, XN_plus_1]), let(
            a := alg.a, #EvalScalar(alg.a),
            aa := Cond(self.doNormalize, cond(gt(a, 0.5), 1-a, a), a), #Cond(self.doNormalize and a > 1/2, 1-a, a), 
            1/s2 * self.to1X_2d(aa, alg.rot)),

        Error("Algebra <alg> must have dimension of 1 or 2"))),


    from1X_2d0 := (self, rot) >> I(2),
    from1X_2d := (self, a, rot) >> Mat(MatSPL(self.to1X_2d(a, rot))^-1), 
    from1X := (self, alg) >> let(t := ObjId(alg), s1 := self.scale1d, s2 := self.scale2d, Cond(
        alg.n=1, s1 * I(1),

        alg.n=2 and (t in [XN_min_1, XN_min_1U]),
            s2 * self.from1X_2d0(alg.rot),

        alg.n=2 and (t in [XN_skew, XN_plus_1]), let(
            a := alg.a, #EvalScalar(alg.a),
            aa := Cond(self.doNormalize, cond(gt(a, 0.5), 1-a, a), a), 
            s2 * self.from1X_2d(aa, alg.rot)),

        Error("Algebra <alg> must have dimension of 1 or 2"))),
)); 

#
# TIME BASIS Classes
#
Declare(Freq_1, Freq_1H, Freq_T, Freq_S, Freq_TH, Freq_THU,  Freq_E, Freq_EH,  Freq_H, Freq_HH);

Declare(Time_TX, Time_SX, Time_THX, Time_THUX,  Time_EX, Time_EHX,  Time_HX, Time_HHX);

# ASP time basis: standard "time" basis (1, x, ..., x^{n-1}) 
Class(Time_TX,  ASPTimeBasis, rec(
    toX := (self, alg) >> I(alg.n)
));

# ASP time basis: symmetric "time" basis (-x^{n/2}, ..., 1, x, ..., x^{n/2-1}) 
Class(Time_SX,  ASPTimeBasis, rec(
    toX := (self, alg) >> let(t := ObjId(alg), hf := Int(alg.n/2), 
        Cond(t in [XN_min_1, XN_min_1U], 
                Z(alg.n, hf),
             t = XN_plus_1 or (t = XN_skew and alg.a = 1/4),
                 DirectSum(I(Int((alg.n+1)/2)), -I(hf)) * Z(alg.n, hf),
             t = XN_skew, 
                 Tensor(Freq_S(1,1).to1X(XN_skew(2, alg.a, alg.rot)), I(alg.n/2)), 
             Error("Unrecognized algebra <alg>")))
)); 

Class(Time_THX,  ASPComposedTimeBasis, rec(composedFrom := self >> Freq_TH(1,1)));
Class(Time_THUX, ASPComposedTimeBasis, rec(composedFrom := self >> Freq_THU(1,1)));
Class(Time_EX,   ASPComposedTimeBasis, rec(composedFrom := self >> Freq_E(1,1))); # e_a(x^n) (*) t_n
Class(Time_EHX,  ASPComposedTimeBasis, rec(composedFrom := self >> Freq_EH(1,1)));
Class(Time_HX,   ASPComposedTimeBasis, rec(composedFrom := self >> Freq_H(1,1))); # h_a(x^n) (*) t_n
Class(Time_HHX,  ASPComposedTimeBasis, rec(composedFrom := self >> Freq_HH(1,1)));


#
# FREQUENCY BASIS Classes
#

# ASP frequency basis: (1)
Class(Freq_1,  ASPFreqBasis1d, rec(
    timeBasis := self >> Time_TX
));

Class(Freq_1H,  ASPFreqBasis1d, rec(
    timeBasis := self >> Error("not implemented"),
    to1X := (self, alg) >> Error("not implemented"),
));

# ASP frequency basis: (1, x), also implies 2d spectral components, ie real DFT
Class(Freq_T, ASPFreqBasis1d2d, rec(
    timeBasis := self >> Time_TX,
    to1X := (self, alg) >>  let(t := ObjId(alg), s1 := self.scale1d, s2 := self.scale2d, Cond(
        alg.n = 2, 1/s2 * I(2),
        alg.n = 1, 1/s1 * I(1),
        Error("Algebra <alg> must have dimension of 1 or 2"))),
    from1X := (self, alg) >>  let(t := ObjId(alg), s1 := self.scale1d, s2 := self.scale2d, Cond(
        alg.n = 2, s2 * I(2),
        alg.n = 1, s1 * I(1),
        Error("Algebra <alg> must have dimension of 1 or 2"))),
));

# ASP frequency basis: (x^-1, 1), also implies 2d spectral components, ie real DFT
Class(Freq_S, ASPFreqBasis1d2d, rec(
    timeBasis := self >> Time_SX,
    to1X_2d0 := (self, rot) >> J(2), 
    to1X_2d := (self, a, rot) >> Mat(_stripval([[ 2*cospi(2*a*rot), 1], 
                                                [             -1  , 0]])),
));

# ASP frequency basis: (x^1/2, x^3/2) for type 2--4 transforms, also implies 2d spectral components, ie real DFT
Class(Freq_TH, ASPFreqBasis1d2d, rec(
    timeBasis := self >> Time_THX,
    to1X_2d := (self, a, rot) >> let(c := cospi(2*a*rot), q := 1/(2*cospi(a*rot)), 
        Mat(_stripval(q * [[1+2*c, 1],
                           [-1,    1]])))
));

Class(Freq_THU, ASPFreqBasis1d2d, rec(
    timeBasis := self >> Time_THUX,
    to1X_2d := (self, a, rot) >> let(c := cospi(2*a*rot), q := 1/(2*cospi(a * rot)), 
        Mat(_stripval(q^2 * [[1+2*c, 1],
                             [-1,    1]])))
));

# ASP frequency basis: (1, (x-cos(a)) / sin(a)), also implies 2d spectral components, ie real DFT
Class(Freq_E,  ASPFreqBasis1d2d, rec(
    timeBasis := self >> Time_EX,
    to1X_2d := (self, a, rot) >> let(c := cospi(2*rot*a), s := sinpi(2*rot*a),     
        Mat(_stripval([[1, -c/s], 
                       [0,  1/s]]))),
    from1X_2d := (self, a, rot) >> let(c := cospi(2*rot*a), s := sinpi(2*rot*a),     
        Mat(_stripval([[1,  c], 
                       [0,  s]])))
));

Class(Freq_EH,  ASPFreqBasis1d2d, rec(
    timeBasis := self >> Time_EX,
    to1X_2d := (self, a, rot) >> let(c := r->cospi(r*2*rot*a), s := r->sinpi(r*2*rot*a),     
        Mat(_stripval(1/s(1) * [[ s(3/2), -c(3/2)], 
                                [-s(1/2),  c(1/2)]])))
));

# ASP frequency basis: (1 + (x-cos(a)) / sin(a), 1 - (x-cos(a))/sin(a)), ie Hartley transform
Class(Freq_H,  ASPFreqBasis1d2d, rec(
    timeBasis := self >> Time_HX,
    to1X_2d := (self, a, rot) >> let(c := cospi(2*rot*a), s := sinpi(2*rot*a),     
        Mat(_stripval(1/2 * [[1-c/s, 1+c/s], 
                             [  1/s,  -1/s]])))
));

Class(Freq_HH,  ASPFreqBasis1d2d, rec(
    timeBasis := self >> Time_EX,
    to1X_2d := (self, a, rot) >> let(s := sinpi(2*rot*a), 
        cas := r -> cospi(r*2*rot*a)+sinpi(r*2*rot*a), cms := r -> cospi(r*2*rot*a)-sinpi(r*2*rot*a),
        Mat(_stripval(1/(2*s) * [[-cms(3/2), -cas(3/2)], 
                                 [ cms(1/2),  cas(1/2)]])))
));

