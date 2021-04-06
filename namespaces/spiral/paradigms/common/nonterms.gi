
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(APar, ASingletonTag);
Class(AVec, ASingletonTag);

IsParPar := P -> P[3] = APar and P[4] = APar;
IsVecVec := P -> P[3] = AVec and P[4] = AVec;
IsVecPar := P -> P[3] = AVec and P[4] = APar;
IsParVec := P -> P[3] = APar and P[4] = AVec;
IsParXXX := P -> P[3] = APar;
IsVecXXX := P -> P[3] = AVec;
IsXXXPar := P -> P[4] = APar;
IsXXXVec := P -> P[4] = AVec;

BaseOperation.doNotMeasure := true;
PermClass.doNotMeasure := true;

###################################################################
#F tSPL - base class for all tSPL constructs
#F *OBSOLETE*. Use Tagged_tSPL
#F

#D Class(tSPL, NonTerminal, rec(
#D    isTSPL := true
#D ));

###################################################################
#F Tagged_tSPL - base class for all tSPL constructs with tag support
#F
#F *!!* Should be used for all new constructs instead of tSPL
#F
Class(Tagged_tSPL, TaggedNonTerminal, rec(
    isTSPL := true
));

###################################################################
#F Tagged_tSPL_Container - base class for a generic tSPL container
#F
Class(Tagged_tSPL_Container, Tagged_tSPL, rec(
    abbrevs :=  [ nt -> Checked(IsSPL(nt), [nt]) ],
    dims := self >> self.params[1].dims(),
    advdims := self >> self.params[1].advdims(),
    terminate := self >> self.params[1].terminate(),
    transpose := self >> ObjId(self)(self.params[1].transpose())
                             .withTags(self.getTags()),
    isReal := self >> self.params[1].isReal(),
    isInplace := self >> self.params[1].isInplace(),
    normalizedArithCost := self >> self.params[1].normalizedArithCost(),
    doNotMeasure := true,
));


IsTSPL := o -> IsRec(o) and IsBound(o.isTSPL) and o.isTSPL;

Declare(TGrp, TTensor, TTensorI, TTensorInd, TDirectSum, TInplace);

#F TGrp(<s>)
#F  tSPL grouping container (i.e. parenthesis)
#F
Class(TGrp, Tagged_tSPL_Container, rec(
    abbrevs :=  [ s -> Checked(IsSPL(s), [s]) ],
    HashId := self >> let(
	p := self.params[1],
	h := When(IsBound(p.HashId), p.HashId(), p),
        [ TGrp, h ] :: When(IsBound(self.tags), self.tags, [])
    )
));

#F TTensor(<A>, <B>)
#F    Nonterminal for Tensor(A, B)
#F
Class(TTensor, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (A, B) -> Checked(IsNonTerminal(A), IsNonTerminal(B), [A,B]) ],

    dims := self >> let(
	a := self.params[1].dims(), b := self.params[2].dims(),
	[ a[1]*b[1], a[2]*b[2] ] ),

    terminate := self >> Tensor(self.params[1], self.params[2]),

    transpose := self >>
        TTensor(self.params[1].transpose(), self.params[2].transpose())
           .withTags(self.getTags()),

    isReal := self >> self.params[1].isReal() and self.params[2].isReal(),

    normalizedArithCost := self >>
        self.params[1].normalizedArithCost() * Rows(self.params[2]) +
	self.params[2].normalizedArithCost() * Cols(self.params[1]),

    HashId := self >> let(
	h := List(self.params, i -> When(IsBound(i.HashId), i.HashId(), i)),
        When(IsBound(self.tags), Concatenation(h, self.tags), h))
));

#F TDirectSum(<A>, <B>)
#F    Nonterminal for DirectSum(A, B)
#F
Class(TDirectSum, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (A, B) -> Checked(IsSPL(A), IsSPL(B), [ A, B ]) ],
    dims := self >> let(
	a:=self.params[1].dims(), b:=self.params[2].dims(),
	[ a[1]+b[1], a[2]+b[2] ]),

    terminate := self >> DirectSum(self.params[1], self.params[2]),

    transpose := self >>
        TDirectSum(self.params[1].transpose(), self.params[2].transpose())
	    .withTags(self.getTags()),

    isReal := self >> self.params[1].isReal() and self.params[2].isReal(),

    doNotMeasure := true,

    HashId := self >> let(
	h := List(self.params, i->When(IsBound(i.HashId), i.HashId(), i)),
        When(IsBound(self.tags), Concatenation(h, self.tags), h))
));

#F TTensorI(<nt>, <s>, <l>, <r>)
#F
#F tSPL nonterminal for A x I, I x A, (A x I)L, (I x A)L
#F
#F Example: TTensorI(F(2), 32, APar, APar)
#F
Class(TTensorI, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (nt, s, l, r) -> Checked(
        IsSPL(nt), IsPosInt(s), l in [APar, AVec], r in [APar, AVec],
	[nt, s, l, r])
    ],

    dims := self >> self.params[1].dims()*self.params[2],

    SPLtSPL := (self, nt, P) >> let(
	A := nt[1], n := P[2], l := P[3], r := P[4],
        Cond(IsParPar(P), Tensor(I(n), A),
             IsVecVec(P), Tensor(A, I(n)),
             IsParVec(P), Tensor(I(n), A) * L(A.dims()[2]*n, n),
             IsVecPar(P), let(m:=A.dims()[2], Tensor(A, I(n)) * L(m*n, m))
    )),

    terminate := self >> let(
	P := self.params, A:=P[1].terminate(), n:= P[2], l:=P[3],  r:=P[4],
        Cond(IsParPar(P), Tensor(I(n), A),
             IsVecVec(P), Tensor(A, I(n)),
             IsParVec(P), Tensor(I(n), A) * L(A.dims()[2]*n, n),
             IsVecPar(P), let(m:=A.dims()[2], Tensor(A, I(n)) * L(m*n, m)))),

    transpose := self >> let(p := self.params,
        TTensorI(p[1].transpose(), p[2], p[4], p[3]).withTags(self.getTags())),

    isReal := self >> self.params[1].isReal(),

    normalizedArithCost := self >>
        self.params[1].normalizedArithCost() * self.params[2],

    doNotMeasure := true,

    HashId := self >> let(
	p := self.params,
	h := When(IsBound(p[1].HashId), p[1].HashId(), p[1]),
        [h, p[2], p[3], p[4]] :: When(IsBound(self.tags), self.tags, [])
    ),
));

Class(TTensorInd, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (nt, idx, l, r) -> Checked(
        IsSPL(nt), IsVar(idx), l in [APar, AVec], r in [APar, AVec],
	[nt, idx, l, r])
    ],

    dims := self >> self.params[1].dims() * self.params[2].range,

    SPLtSPL := (self, nt, P) >> let(
	A := nt[1], idx := P[2], m := A.dims()[2], n := idx.range,
        dsA := IDirSum(idx, A),
        Cond(IsParPar(P), dsA,
             IsVecVec(P), L(m*n, m) * dsA * L(m*n, n),
             IsParVec(P), dsA * L(m*n, n),
             IsVecPar(P), L(m*n, m) * dsA)
    ),

    terminate := self >> let(
	P := self.params, A := P[1], idx:= P[2],
        lstA := List([0..idx.range-1],
	    i -> SubstVars(Copy(A), rec((idx.id) := V(i))).terminate()),
        d := A.dims(),
	n := idx.range,
        dsA := DirectSum(lstA),
        Cond(IsParPar(P), dsA,
             IsVecVec(P), L(d[1]*n, d[1]) * dsA * L(d[2]*n, n),
             IsParVec(P), dsA * L(m*n, n),
             IsVecPar(P), L(m*n, m) * dsA)
    ),

    transpose := self >> TTensorInd(
       self.params[1].transpose(), self.params[2], self.params[4], self.params[3])
          .withTags(self.getTags()),

    isReal := self >> self.params[1].isReal(),

    normalizedArithCost := self >>
        self.params[1].normalizedArithCost() * self.params[2].range,

#    doNotMeasure := true,

    HashId := self >> let(
	p := self.params,
	h := When(IsBound(p[1].HashId), p[1].HashId(), p[1]),
        [h, p[2].range, p[3], p[4]] :: When(IsBound(self.tags), self.tags, [])
    )
));


Declare(TL);

#F  tSPL I x L x I
#F  TL(m, n, k, j)
#F  I(k), L(m, n), I(j)
Class(TL, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (size, stride) -> Checked(ForAll([size, stride], IsPosIntSym), [size, stride, 1, 1]),
                  (size, stride, left, right) -> Checked(ForAll([size, stride, left, right], IsPosIntSym), [size, stride, left, right]) ],
    __call__ := arg >> let(self := arg[1], args := Drop(arg, 1),
        Cond(args=[1,1,1,1], I(1),
             ApplyFunc(Inherited,args))),

    dims := self >> Replicate(2, self.params[1]*self.params[3]*self.params[4]),
    terminate := self >> Tensor(I(self.params[3]), L(self.params[1], self.params[2]), I(self.params[4])),
    transpose := self >> TL(self.params[1], self.params[1]/self.params[2], self.params[3], self.params[4]).withTags(self.getTags()),
    isReal := self >> true,
    noCodelet := true,
    # need that for DPBench!
    doNotMeasure := true,
    transposeSymmetric := False,
    isSymmetric := self >> self.params[2]^2 = self.params[1],
    hashAs := self >> ObjId(self)(self.params[1], self.params[2], self.params[3], self.params[4]).withTags(self.getTags()),
));

#F  tSPL Bit Reversal
Class(TBR, Tagged_tSPL_Container, rec(
    abbrevs :=  [ n -> Checked(IsInt(n), [n])],
    dims := self >> [self.params[1], self.params[1]],
    terminate := self >> BR(self.params[1]),
    transpose := self >> self,
    isReal := True,
    doNotMeasure := true,
));

Declare(TRDiag);

#F TRDiag(<nrows>, <ncols>, <diagfunc>)
#F
#F TRDiag is a rectangular tall matrix, which is a diagonal
#F zero-padded on the bottom.
#F
#F To obtain wide matrix (diagonal zero-padded on the right),
#F use TRDiag(...).transpose()
#F
#F Example:
#F  spiral> PrintMat(MatSPL(  TRDiag(6, 3, FList(TInt, [1,2,3])) ));
#F  [ [ 1,  ,   ],
#F    [  , 2,   ],
#F    [  ,  , 3 ],
#F    [  ,  ,   ],
#F    [  ,  ,   ],
#F    [  ,  ,   ] ]
#F
Class(TRDiag, Tagged_tSPL, rec(
    abbrevs :=  [ (m,n,diag) -> Checked(IsPosIntSym(m), IsPosIntSym(n),
	    AnySyms(m, n) or (m>=n),
	    IsFunction(diag) or IsFuncExp(diag),
	    [m, n, diag])],

    dims :=  self >> let( d := [ self.params[1], self.params[2] ], Cond(self.transposed, Reversed(d), d)),

    # tells autolib's ConstrainPSums that params[2] must be = params[3].domain()
    # NOTE: get rid of params[2]
    paramConstraints := self >> [[self.params[2], self.params[3].domain()]],

    terminate := self >> let(
        res := RI(self.params[1], self.params[2]) * Diag(self.params[3]),
	When(self.transposed, res.transpose(), res)),
	
    conjTranspose := self >> CopyFields(self,
	rec(params := [self.params[1], self.params[2], FConj(self.params[3])]))
            .transpose(),

    isReal := self >> not IsComplexT(self.params[3].range()),

    doNotMeasure := true,

    hashAs := self >> let(
	t := ObjId(self)(self.params[1], self.params[2],
	                 fUnk(self.params[3].range(), self.params[3].domain())),
	tt := t.withTags(self.getTags()),
	When(self.transposed, tt.transpose(), tt))
));


#F  tSPL Digit Reversal
Class(TDR, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (n, r) -> Checked(IsInt(n) and  IsInt(r) and n=r^LogInt(n,r), [n, r])],
    dims := self >> [self.params[1], self.params[1]],
    terminate := self >> DR(self.params[1], self.params[2]),
    transpose := self >> self,
    isReal := True,
    doNotMeasure := true,
));

Declare(TReflL);
#F  tSPL Refl(L(mn,m))
Class(TReflL, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (mn, m) -> Checked(IsInt(mn) and  IsInt(m), [mn, m]) ],
    dims := self >> self.terminate().dims(),
    terminate := self >> let(mn := self.params[1], m:=self.params[2], N := mn/2,
                             Refl(N, 2*N-1, N, L(mn, m))),
    transpose := self >> TReflL(self.params[1], self.params[1]/self.params[2]).withTags(self.getTags()),
    isReal := True,
    doNotMeasure := true,
));


Declare(TRC, TCompose, TDiag, TICompose);


#F   tSPL RC(.)
Class(TRC, Tagged_tSPL_Container, rec(
    _short_print := true,
    abbrevs :=  [ (A) -> Checked(IsNonTerminal(A) or IsSPL(A), [A]) ],
    dims := self >> 2*self.params[1].dims(),
    terminate := self >> Mat(MatSPL(RC(self.params[1]))),

    transpose := self >> ObjId(self)(
	self.params[1].conjTranspose()).withTags(self.getTags()),

    conjTranspose := self >> self.transpose(),

    isReal := self >> true,

    # Do not use doNotMeasure, this will prevent TRC_By_Def from ever being found!
    doNotMeasure := false,
    normalizedArithCost := self >> self.params[1].normalizedArithCost(),

    HashId := self >> let(
	h := [ When(IsBound(self.params[1].HashId), self.params[1].HashId(),
		    self.params[1]) ],
        When(IsBound(self.tags), Concatenation(h, self.tags), h))
));

Class(TMat, Tagged_tSPL, rec(
    abbrevs :=  [ m -> Checked(IsMat(m), [m]) ],
    dims := self >> DimensionsMat(self.params[1]),
    terminate := self >> Mat(self.params[1]),
    transpose := self >> ObjId(self)(TransposedMat(self.params[1])).withTags(self.getTags()),
    isReal := self >> IsRealMat(self.params[1]),
    isInplace := self >> false,
    normalizedArithCost := self >> 2 * Rows(self.params[1]) * Cols(self.params[1]),
    doNotMeasure := false,
));

#F   tSPL Diag(d)
Class(TDiag, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (D) -> [D] ],
    dims := self >> let(n:=self.params[1].domain(), [n,n]),
    terminate := self >> Diag(self.params[1]),
    transpose := self >> Copy(self),
    conjTranspose := self >> ObjId(self)(FConj(self.params[1])).withTags(self.getTags()),

    isReal := self >> Diag(self.params[1]).isReal(),

    doNotMeasure := true,
    noCodelet := true,
    hashAs := self >> ObjId(self)(FUnk(self.params[1].domain())).withTags(self.getTags()),
    #NOTE: depends on wether the whole transform is complex...
    normalizedArithCost := self >> When(self.params[1].isReal(), 2*Rows(self), 6*Rows(self))
));

#F   tSPL RCDiag(d)
Class(TRCDiag, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (D) -> [D] ],
    dims := self >> let(n:=self.params[1].domain(), [n,n]),
    terminate := self >> RCDiag(self.params[1]),
    # NOTE: transpose on real == conjugate; transpose on complex == id. What to use?
#    transpose := self >> let(d := self.params[1].var.value, t := d.t.t, size := d.t.size,
#        newd := List(d.v, i-> t.value(Complex(ReComplex(i.v), -ImComplex(i.v)))),
#        FData(newd)),
    transpose := self >> CopyFields(self, rec(transposed := not self.transposed)),
    isReal := True,
    doNotMeasure := true,
    noCodelet := true,
    hashAs := self >> ObjId(self)(FUnk(self.params[1].domain())).withTags(self.getTags()),
    normalizedArithCost := self >> 3*Rows(self)
));



#F   tSPL Id(f)
Class(TId, TDiag);


Declare(TCond);
#F   tSPL Cond
Class(TCond, Tagged_tSPL_Container, rec(
    abbrevs   := [ (f, a, b) -> Checked(IsBound(f.ev), [f,a,b]) ],
    dims      := self >> self.params[2].dims(),
    terminate := self >> COND(self.params[1], self.params[2].terminate(), self.params[3].terminate()),
    transpose := self >> TCond(self.params[1], self.params[2].transpose(), self.params[3].transpose()), # maybe
));

Declare(TRaderMid);
#F   tSPL RaderMid(N, k, root, transp)
Class(TRaderMid, Tagged_tSPL_Container, rec(
    # -- Rader infrastructure --
    raderDiag := (N, k, root) -> SubList(ComplexFFT(List([0..N-2], x->ComplexW(N, k*root^x mod N))), 2) / (N-1),
    raderMid := (self, N, k, root) >>
                 DirectSum(Mat([[1, 1], [1, -1/(N-1)]]),
                   Diag(FData(self.raderDiag(N, k, root)))),
    # -- tSPL infrastructure --
    abbrevs :=  [ (N, k, root) -> [N, k, root, false],
                  (N, k, root, transp) -> [N, k, root, transp] ],
    dims := self >> let(N:=self.params[1], [N,N]),
    terminate := self >> let(spl := self.raderMid(self.params[1], self.params[2], self.params[3]),
        When(self.params[4], spl.transpose(), spl)),
    transpose := self >> TRaderMid(self.params[1], self.params[2], self.params[3], not self.params[4]).withTags(self.getTags()),
    isReal := False,
    doNotMeasure := true,
    noCodelet := true,
));


#F   tSPL Compose([A,B,...])
Class(TCompose, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (l) -> Checked(IsList(l), [l]) ],
    dims := self >> [self.params[1][1].dims()[1], self.params[1][Length(self.params[1])].dims()[2]],
    terminate := self >> Compose(List(self.params[1], i->i.terminate())),
    transpose := self >> TCompose(Reversed(List(self.params[1], i->i.transpose()))).withTags(self.getTags()),
    isReal := self >> ForAll(self.params[1], i->i.isReal()),
    doNotMeasure := true,
    normalizedArithCost := self >> Sum(List(self.params[1], i->i.normalizedArithCost())),

    HashId := self >> let(h := List(self.params[1], i->When(IsBound(i.HashId), i.HashId(), i)),
        When(IsBound(self.tags), Concatenation(h, self.tags), h))
));


#F   tSPL ICompose(<var>, <domain>, <nonterm>, <attrib>)
Class(TICompose, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (var,domain,l) -> [var, domain, l] ],
    dims := self >> self.params[3].dims(),
    terminate := self >> ICompose(self.params[1], self.params[2], self.params[3]).unroll(),

    unroll := self >> let(
        var := self.params[1],
        domain := self.params[2],
        expr := self.params[3],
        tag := self.getTags(),
        TCompose(
            List([ 0 .. domain - 1 ],
                index_value -> SubstBottomUp(
                    Copy(expr), var,
                    e -> V(index_value))))),

    isReal := self >> ICompose(self.params[1], self.params[2], self.params[3]).unroll().isReal(),

    transpose := self >> let(
        var := self.params[1],
        dom := self.params[2],
        ch := self.params[3],
        newvar := var.clone(),
        tags := self.getTags(),
        TICompose(newvar, dom,
            SubstVars(Copy(ch.transpose()), tab((var.id) := dom-1-newvar))).withTags(tags)
    )
));


Declare(TGath, TScat, TPrm, TConj, TS);
#F   tSPL Gath(d)
Class(TGath, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (tspl) -> [tspl] ],
    dims := self >> [self.params[1].domain(), self.params[1].range()],
    terminate := self >> Gath(self.params[1]),
    transpose := self >> TScat(self.params[1]).withTags(self.getTags()),
    isReal := True,
    doNotMeasure := true,
    noCodelet := true,
    normalizedArithCost := self >> 0
));


Class(TScat, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (tspl) -> [tspl] ],
    dims := self >> [self.params[1].range(),self.params[1].domain()],
    terminate := self >> Scat(self.params[1]),
    transpose := self >> TGath(self.params[1]).withTags(self.getTags()),
    isReal := True,
    doNotMeasure := true,
    noCodelet := true,
    normalizedArithCost := self >> 0
));


Class(TPrm, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (tspl) -> [tspl] ],
    dims := self >> Cond(IsBound(self.params[1].domain), [self.params[1].domain(), self.params[1].range()], self.params[1].dims()),
    terminate := self >> Prm(self.params[1]),
    transpose := self >> TPrm(self.params[1].transpose()).withTags(self.getTags()),
    isReal := True,
    doNotMeasure := true,
    noCodelet := true,
    normalizedArithCost := self >> 0
));


Class(TConj, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (tspl, prm) -> [tspl,prm.transpose(),prm],
                  (tspl, prml, prmr) -> [tspl,prml,prmr] ],
    dims := self >> [self.params[2].domain(), self.params[3].range()],
    terminate := self >> ConjLR(self.params[1].terminate(), self.params[2], self.params[3]),
    transpose := self >> TConj(self.params[1].transpose(), self.params[3].transpose(), self.params[2].transpose()).withTags(self.getTags()),
    isReal := self >> self.params[1].isReal(),
    doNotMeasure := false,
    noCodelet := true,
    normalizedArithCost := self >> self.params[1].normalizedArithCost()
));


Class(TS, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (N) -> [N, false],
                  (N, t) -> [N, t] ],
    dims := self >> [self.params[1], self.params[1]],
    terminate := self >> When(self.params[2], S(self.params[1]).transpose(), S(self.params[1])),
    transpose := self >> TS(self.params[1], not self.params[2]).withTags(self.getTags()),
    isReal := True,
    doNotMeasure := true,
    noCodelet := true,
    normalizedArithCost := self >> self.params[1] - 1
));

Class(TInplace, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (tspl) -> [tspl] ],
    dims := self >> self.params[1].dims(),
    terminate := self >> self.params[1].terminate(),
    transpose := self >> TInplace(self.params[1].transpose()).withTags(self.getTags()),
    isReal := self >> self.params[1].isReal(),
    isInplace := True,
    doNotMeasure := true,
    noCodelet := true,
    normalizedArithCost := self >> self.params[1].normalizedArithCost()
));



InterleavedComplexT := t -> TRC(t);
SplitComplexT := t -> let(r:= Rows(t), c:=Cols(t), TConj(TRC(t), L(2*r, 2), L(2*c, c)));
ComplexT := (t,opts) -> When((not IsBound(opts.interleavedComplex)) or (IsBound(opts.interleavedComplex) and opts.interleavedComplex), InterleavedComplexT(t), SplitComplexT(t));

InterleavedSplitComplexT := t -> let(r:= Rows(t), c:=Cols(t), TConj(TRC(t), fId(2*r), L(2*c, c)));
