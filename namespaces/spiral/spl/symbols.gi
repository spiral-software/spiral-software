
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#P SPL Parametrized Symbols
#P ========================
#P
#P This package contains parametrized symbols - shortcut notation for common
#P idiomatic constructs.
#P

#F -----------------------------------------------------------------------------
#F I(N) : NxN identity matrix
#F -----------------------------------------------------------------------------
Class(I, Sym, rec(
    abbrevs := [ (x,y) -> Checked(x=y, IsPosInt0Sym(x), x),
                 (n)   -> Checked(IsPosInt0Sym(n), n) ],
    def := n -> Perm((), n),
    transpose := self >> self,
    conjTranspose := self >> self,
    inverse := self >> self,
    isIdentity := True,
    isSymmetric := True,
    doNotMeasure := true,
    printlatex := self >> Print(" \\one_{",self.params[1],"} "),
    area := self >> self.params[1],
    normalizedArithCost := self >> 0,
));

#F -----------------------------------------------------------------------------
#F O(M,N) : MxN matrix of zeros
#F -----------------------------------------------------------------------------
Class(O, BaseMat, rec(
    abbrevs := [ n -> Checked(IsPosInt(n), [n,n]) ],

    new := (self, r, c) >> SPL(WithBases(self, rec(
        params := [r, c], dimensions := [r, c]))),

    isReal := True,
    isPermutation := False,
    arithmeticCost := (self, costMul, costAddMul) >> costMul(0) - costMul(0),

    transpose     := self >> ObjId(self)(self.params[2], self.params[1]), 
    conjTranspose := self >> ObjId(self)(self.params[2], self.params[1]), 

    area := self >> Maximum(self.dimensions),

    toAMat := self >> AMatMat(NullMat(
	EvalScalar(self.params[1]), EvalScalar(self.params[2]))),

    print := Sym.print,
    dims  := self >> [ self.params[1], self.params[2] ],
    rChildren := Sym.rChildren,
    rSetChild := Sym.rSetChild
));

#F -----------------------------------------------------------------------------
#F RI(M,N) : MxN rectangular identity
#F           MIN(M,N) x MIN(M,N) identity padded with 0s
#F -----------------------------------------------------------------------------
Class(RI, Sym, rec(
    def := (r, c) -> Checked(IsPosInt(r), IsPosInt(c),
    Cond(r = c, I(r),
         r < c, Gath(fAdd(c,r,0)),
                Scat(fAdd(r,c,0)))),

    transpose := self >> ApplyFunc(self.__bases__[1], Reversed(self.params)),
    conjTranspose := self >> ApplyFunc(self.__bases__[1], Reversed(self.params)),
));

#F -----------------------------------------------------------------------------
#F F(N) - NxN Discrete Fourier Transform matrix
#F -----------------------------------------------------------------------------
Class(F, Sym, rec(
    def := size -> Checked(IsPosInt(size),
    Cond(size = 1, Mat([[1]]),
         size = 2, Mat([[1,1], [1,-1]]),
         Mat(Global.DFT(size)))),

    isReal    := self >> self.params[1] <= 2,
    isPermutation := False,
    transpose := self >> self,
    conjTranspose := self >> self,
    inverse := self >> let(n:=self.params[1], Cond(n=1, self, n=2, 1/2*F(2), Error("Inverse not supported"))),
    toAMat    := self >> DFTAMat(self.params[1]),
    printlatex := (self) >> Print(" F_{", self.params[1], "} ")
));

#F -----------------------------------------------------------------------------
#F R(a)     : 2x2 rotation matrix with angle a*pi, namely
#F
#F    cos(a) sin(a)
#F   -sin(a) cos(a)
#F
#F alternatively:
#F
#F R(c,s)   : specified by a pair such as c^2 + s^2 = 1
#F     <c>  <s>
#F    -<s>  <c>
#F
#F -----------------------------------------------------------------------------
Class(Rot, Sym, rec(
    abbrevs := [ (a) -> [cospi(a), sinpi(a)] ],

    def := (a,b) ->
    Mat( [[a, b], [-b, a]] ),

    transpose := self >>
        self.__bases__[1](self.params[1], -self.params[2]),

    conjTranspose := self >> self.transpose(),
    inverse := self >> self.transpose(),

    toAMat := self >> let(a := self.params[1], b := self.params[2],
        When(IsScalar(a) and IsScalar(b) and ScalarIsCos(a) and ScalarIsSin(b),
         RotationAMat(EvalScalar(ScalarCosArg(a))),
         AMatMat([[EvalScalar(a),  EvalScalar(b)],
              [EvalScalar(-b), EvalScalar(a)]]))),

    #-----------------------------------------------------------------------
    # Expansions
    expandLifting := self >>
        let(c := self.params[1],
            s := self.params[2],
            ev := (not IsScalar(c) or CanBeFullyEvaluated(c.val)) and
                  (not IsScalar(s) or CanBeFullyEvaluated(s.val)),
            Cond(ev and c = s,  c * F(2) * Perm((1,2), 2),
                 ev and c = -s, c * Perm((1,2), 2) * F(2),
                 Mat([[1,1,0], [0,-1,1]]) *
                 Diag([c-s, s, c+s]) *
                 Mat([[1,0], [1,1], [0,1]]))),

    expandWinograd:= self >>
        let(c := self.params[1],
            s := self.params[2],
            ev := (not IsScalar(c) or CanBeFullyEvaluated(c.val)) and
                  (not IsScalar(s) or CanBeFullyEvaluated(s.val)),
            Cond(ev and c = s,  c * F(2) * Perm((1,2), 2),
                 ev and c = -s, c * Perm((1,2), 2) * F(2),
                 Mat([[1, (1-c)/s], [0, 1]]) *
                 Mat([[1, 0], [-s, 1]]) *
                 Mat([[1, (1-c)/s], [0, 1]]))),

    expandDef := self >> self.obj,
));


Declare(toeplitz);

#F -----------------------------------------------------------------------------
#F toeplitz(elements) : toeplitz matrix given by elements of the first row and
#F                      column, starting from the upper right corner
#F -----------------------------------------------------------------------------
#F NOTE: use generating function not list (see 'Toeplitz')
Class(toeplitz, Sym, rec(
    abbrevs := [ arg -> When(Length(arg) > 1 or not IsList(arg[1]), [arg], arg) ],

    def := function(elements)
        local n;
        Constraint(IsList(elements) and Length(elements) > 0);
        Constraint(IsOddInt(Length(elements)));
        DoForAll(elements, x->Constraint(IsScalarOrNum(x)));
        n := (Length(elements) + 1) / 2;
        return Mat(ToeplitzMat(elements));
    end,

    transpose := self >> toeplitz(Reversed(self.params[1])),
    conjTranspose := self >> toeplitz(List(Reversed(self.params[1]), x->Global.Conjugate(x))),
    isReal := True,
    isPermutation := False,
    area := self >> Product(self.dimensions)
));
