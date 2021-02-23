
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


IsASPAlgebra := x -> IsRec(x) and IsBound(x.isASPAlgebra) and x.isASPAlgebra;

Class(AlgebraOps, PrintOps, rec(
    \= := (b1, b2) -> Cond(
        not IsASPAlgebra(b1) or not IsASPAlgebra(b2), false,
        ObjId(b1)=ObjId(b2) and b1.rChildren() = b2.rChildren())
));

Class(ASPAlgebra, rec(
    isASPAlgebra:=true,
    codeletShape := self >> Concatenation([ObjId(self)], List(self.rChildren(), CodeletShape)),
    from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), rch),
    unparse := "RewritableObjectExp",
)); 


Declare(XN_skew, XN_plus_1, XN_min_1, XN_min_w);

#F XN_min_1(n) -- represents x^n - 1 polynomial
#F   Spectral decomposition: 1d and 2d components 
#F   <n> even or odd
#F
Class(XN_min_1, ASPAlgebra, rec(
    __call__ := (self, n, rot) >> let(nn:=_unwrap(n), Checked(IsPosIntSym(nn), IsIntSym(rot), WithBases(self, 
        rec(n := nn, rot := rot, operations := PrintOps)))),

    hashAs := self >> ObjId(self)(self.n, 1),
    rChildren := self >> [self.n, self.rot],
    rSetChild := rSetChildFields("n", "rot"),
    print := self >> Print(self.__name__, "(", self.n, ", ", self.rot,")"),

    conj := self >> CopyFields(self, rec(rot := -self.rot)),

    prettyPrint := self >> PrintEval("x$1 - 1", When(self.n=1, "", Concat("^", StringInt(self.n)))),

    rspectrum := self >> let(rot := EvalScalar(self.rot), n := EvalScalar(self.n), Cond(
        n <= 1, [self], # if n<=1 then then can't decompose further
        Concatenation(
            [ XN_min_1(1, rot) ],
            Cond(IsOddInt(n), [], [XN_plus_1(1, rot)]),
            List([1..Int((n-1)/2)], i -> XN_skew(2, i/n, rot)))
    )),

    cspectrum := self >> let(rot := EvalScalar(self.rot), n := EvalScalar(self.n), Cond(
        n = 1, [self],
        n = 2, [XN_min_1(1,self.rot), XN_plus_1(1, self.rot)],
        n > 2, ConcatList(self.rspectrum(), x -> x.cspectrum())
    )),

    # rfactor(1) == rspectrum()
    # rfactor(n) == self
    rfactor := (self, m) >> Checked((self.n mod m) = 0,
        List(XN_min_1(self.n/m, self.rot).rspectrum(), x -> CopyFields(x, rec(n := x.n * m)))),

    shift := self >> Z(self.n, -1)
));

#F XN_min_1U(n) -- represents x^n - 1 polynomial
#F   Spectral decomposition: 2d components only
#F   <n> must be even
#F
Class(XN_min_1U, XN_min_1, rec(
    __call__ := (self, n, rot) >> let(nn:=_unwrap(n), Checked(IsPosIntSym(nn), IsSymbolic(nn) or IsEvenInt(nn),
        IsIntSym(rot), WithBases(self, 
            rec(n := nn, rot := rot, operations := PrintOps)))),

    rspectrum := self >> let(rot := EvalScalar(self.rot), n := EvalScalar(self.n), Cond(
        self.n <= 1, [self], # if n==1 then then can't decompose further
        Concatenation(
            Cond(IsOddInt(n), [XN_min_1(1, rot)], [XN_min_1(2, rot)]),
            List([1..Int((n-1)/2)], i -> XN_skew(2, i/n, rot)))
    ))
)); 

Class(XN_skew_base, ASPAlgebra, rec(
    prettyPrint := self >> Cond(IsOddInt(self.n) or self.a=1/4,
        PrintEval("x$1 + 1", When(self.n=1, "", Concat("^", StringInt(self.n)))),
        PrintEval("x^$1 - 2 x$2 cos($3*pi) + 1", self.n, When(self.n=2, "", Concat("^", StringInt(self.n/2))), 2*self.a)
    ),

    conj := self >> CopyFields(self, rec(rot := -self.rot)),

    shift := self >> let(n:=self.n, cos:=CosPi(self.rot*2*self.a), i := Ind(n),
        HStack(VStack(O(1,n-1),I(n-1)), 
            Mat([List([0..n-1], i -> Cond(i=0, -1, i=n/2, 2*cos, 0))]).transpose())),

    cspectrum := self >> let(rot := EvalScalar(self.rot), n := EvalScalar(self.n), Cond(
        n = 1, [self],
        n = 2, [XN_min_w(1,self.a, rot), XN_min_w(1,1-self.a, rot)],
        n > 2, ConcatList(self.rspectrum(), x -> x.cspectrum())
    ))
));

#F XN_skew(n, a, rot) -- represents x^n - 2x^{n/2} cospi(2*a) + 1 polynomial
#F   Spectral decomposition: 2d components only
#F   <n> must be even
#F
Class(XN_skew, XN_skew_base, rec(
    __call__ := (self, n, a, rot) >> let(nn := _unwrap(n), 
	Checked(IsPosIntSym(nn), IsSymbolic(nn) or IsEvenInt(nn), IsIntSym(rot), 
	        IsRatSym(a), IsSymbolic(a) or (0 < a and a < 1), 
		WithBases(self, 
		    rec(n := nn, a := a, rot := rot, operations := PrintOps)))),

    hashAs := self >> XN_skew(self.n, 1/16, 1),
    rChildren := self >> [self.n, self.a, self.rot],
    rSetChild := rSetChildFieldsF(_unwrap, "n", "a", "rot"),
    print := self >> Print(self.__name__, "(", self.n, ", ", self.a, ", ", self.rot, ")"),

    rspectrum := self >> let(rot := EvalScalar(self.rot), n := EvalScalar(self.n), Cond(
        n <= 2, [self], # if n<=2 then then can't decompose further
        List([0..Int(n/2)-1], i -> XN_skew(2, (i + self.a)/(n/2), rot)))),

    # rfactor(1) == rspectrum(), rfactor(n) == self
    rfactor := (self, m) >> Checked(IsEvenInt(m), m > 1, (self.n mod m) = 0, 
        List(XN_skew(2*self.n/m, self.a, self.rot).rspectrum(), x -> CopyFields(x, rec(n := x.n * m/2)))),
));

#F XN_plus_1(n, rot) -- represents x^n + 1 polynomial
#F   Spectral decomposition: 1d and 2d components 
#F   <n> even or odd
#F
Class(XN_plus_1, XN_skew_base, rec(
    __call__ := (self, n, rot) >> let(nn:=_unwrap(n), Checked(IsPosIntSym(nn), IsIntSym(rot), 
        WithBases(self, rec(n := nn, a := 1/4, rot := rot, operations := PrintOps)))),

    rspectrum := self >> let(rot := EvalScalar(self.rot), n := EvalScalar(self.n), Cond(
        n <= 2, [self], # if n<=2 then then can't decompose further
        Concatenation(
            List([0..Int(n/2)-1], i -> XN_skew(2, (i + 1/2)/(n), rot)),
            Cond(IsEvenInt(n), [], [XN_plus_1(1, rot)]))
    )),

    rfactor := (self, mm) >> Checked(mm >= 1, (self.n mod mm) = 0, Cond(
        IsOddInt(mm) and not IsOddInt(self.n), Error("<mm> can only be odd if polynomial degree (self.n) is odd"),
        let(m := When(IsOddInt(self.n), mm, mm/2), 
            List(XN_plus_1(self.n/m, self.rot).rspectrum(), x -> CopyFields(x, rec(n := x.n * m)))))),

    hashAs := self >> XN_plus_1(self.n, 1),
    rChildren := self >> [self.n, self.rot],
    rSetChild := rSetChildFields("n", "rot"),
    print := self >> Print(self.__name__, "(", self.n, ", ", self.rot, ")"),
));

#F XN_min_w(n, a, rot) -- represents x^n - w_a polynomial
#F   Spectral decomposition: 1d components only over complex numbers
#F
Class(XN_min_w, XN_skew_base, rec(
    __call__ := (self, n, a, rot) >> let(nn:=_unwrap(n), Checked(
	IsPosIntSym(nn), IsRatSym(a), IsSymbolic(a) or (0 < a and a < 1), IsIntSym(rot), 
        WithBases(self, rec(n := nn, a := a, rot := rot, operations := PrintOps)))),

    hashAs := self >> XN_min_w(self.n, 1/16, 1),
    rChildren := self >> [self.n, self.a, self.rot],
    rSetChild := rSetChildFields("n", "a", "rot"),
    print := self >> Print(self.__name__, "(", self.n, ", ", self.a, ", ", self.rot, ")"),

    rspectrum := self >> Error("Polynomial algebra is complex, no real spectrum"),

    cspectrum := self >> let(rot := EvalScalar(self.rot), n := EvalScalar(self.n), Cond(
        n = 1, [self], # if n<=2 then then can't decompose further
        List([0..n-1], i -> XN_min_1(1, (i + self.a)/n, rot)))),

    prettyPrint := self >> PrintEval("x^$1 - w_$2", self.n, self.a),

    shift := self >> let(n:=self.n, cos:=CosPi(self.rot*2*self.a), i := Ind(n),
        DirectSum(_omega(EvalScalar(self.a))*I(1), When(n=1, [], I(n-1)))*
        Prm(Z(n,-1)))
));
