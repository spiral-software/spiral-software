
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F NthRootCyc(<cyc>, <nroot>) - <nroot>-th root of <cyc>
#F   to get a root we take all powers of cyc of order n,
#F   and make them powers of cyc of order nroot*n (effectively dividing these
#F   powers by nroot). This is done by appending n*(nroot-1) zeros to list of
#F   coeffs and using CoeefsCyc.
NthRootCyc := (cyc, nroot) -> let(n := OrderCyc(cyc),
    coeffs := CoeffsCyc(cyc,n), # -1 must be E(2), not -E(1)
    sum := Sum(coeffs),
    Cond( # GAP normalization of cyc's (to [-j,j], and not [1,-1]) give headache here
	sum =  1, CycList(Concatenation(coeffs, Replicate(n*(nroot-1), 0))),
	sum = -1, CycList(Concatenation(Replicate(n/2, 0), -coeffs{[1..n/2]}, Replicate(n*(nroot-1), 0))),
	Error("E(N) or E(-N) where N is positive integer is expected")));
	

#F SkewDFT(n, alpha, k) - Fourier transform for algebra C[x]/x^n - omega_1^alpha
#F   k - rotation
#F   matrix [ w_n ^ (r+alpha)c ]_{r,c}
#F
Class(SkewDFT, TaggedNonTerminal, rec(
    abbrevs := [ (n)       -> Checked(IsPosIntSym(n), [n, 1, 1]),
	         (n,alpha) -> Checked(IsPosIntSym(n), IsRatSym(alpha), [n, alpha, 1]),
	         (n,alpha,k) -> Checked(IsPosIntSym(n), IsIntSym(k), AnySyms(n,k) or Gcd(_unwrap(n),_unwrap(k))=1, IsRatSym(alpha), [n, alpha, k]) ],

    dims := self >> [ self.params[1], self.params[1] ],
    terminate := self >> let(
	N := self.params[1], rot := self.params[3], a := self.params[2], j := Ind(N-1), 
        mat := DFT(N).terminate() *
               Diag(diagDirsum(fConst(TReal, 1, 1.0), fPrecompute(
                      Lambda(j, cospi(fdiv(2*a*rot*(j+1), N)) + E(4)*sinpi(fdiv(2*a*rot*(j+1), N)))))),
        Cond(self.transposed, mat.transpose(), mat)),

    conjTranspose := self >> ObjId(self)(self.params[1], self.params[2], -self.params[3]).transpose(),

    isReal := False,

    hashAs := self >> let(t:=ObjId(self)(self.params[1], 1/16, 1).withTags(self.getTags()),
        When(self.transposed, t.transpose(), t)),

    normalizedArithCost := self >> let(n := self.params[1], 
        floor(5.0 * n * log(n) / log(2.0))),
));

Class(OnlineDiag, Diag, rec(
    isReal := self >> let(t := self.element.range(),
	Cond(IsVecT(t), t.t <> TComplex and ObjId(t.t) <> T_Complex,
	     t <> TComplex and ObjId(t)<>T_Complex)), 
));

Global.compiler.DefaultCodegen.OnlineDiag := (self, o, y, x, opts) >> let(
    cc := o.element.compute(),
    decl(cc.values, 
        chain(
            cc.code,
            List([1..Length(cc.values)], i -> assign(nth(y, i-1), cc.values[i] * nth(x, i-1)))
        )
    ));

Class(RCOnlineDiag, Diag, rec(
    dims := self >> Replicate(2, 2*self.element.domain())
));

Global.compiler.DefaultCodegen.RCOnlineDiag := (self, o, y, x, opts) >> let(
    cxmul      := (y, x, c) -> assign(y, c*x),
    cxmul_conj := (y, x, c) -> assign(y, conj(c)*x),
    cc         := o.element.computeWithOps(cxmul, cxmul_conj),

    decl(cc.values, chain(
	cc.code,
	List([0..Length(cc.values)-1], i -> chain(
		assign(nth(y, 2*i),   nth(x, 2*i) * re(cc.values[i+1]) - nth(x, 2*i+1) * im(cc.values[i+1])),
		assign(nth(y, 2*i+1), nth(x, 2*i) * im(cc.values[i+1]) + nth(x, 2*i+1) * re(cc.values[i+1]))))
    ))
);

RewriteRules(RulesDiag, rec(
   RC_OnlineDiag := Rule([RC, [OnlineDiag, @(1)]], e-> RCOnlineDiag(@(1).val))
));


#F SkewTwid(<n>, <a>)  -- twiddle factor function II(n-1) -> TComplex
#F   j -> omegapi(2*a*(j+1) / n)
#F
Class(SkewTwid, RewritableObject, Function, rec(
    updateParams := self >> Checked(IsPosIntSym(self.params[1]), IsRatSym(self.params[2]), 0),

    lambda := self >> let(
        n := self.params[1], a := self.params[2], j := Ind(n-1),
        Lambda(j, omegapi(fdiv(2*a*(j+1), n)))),

    range := self >> TComplex,

    domain := self >> self.params[1]-1,
));


#F OnlineSkewTwid(<n>, <a>, <seed_exps>, <seed_twiddle_func>)  -- same as SkewTwid but twiddles are computed
#F   online from seed values given by <seed_twiddle_func>.tolist()
#F
#F Example:
#F   OnlineSkewTwid(13,1/7,[1], fCompose(SkewTwid(13,1/7), FList(TInt, [0]))).compute();
Class(OnlineSkewTwid, SkewTwid, rec(
    # params[2] is only used by super-class's .lambda(), but not here, not in compute()
    updateParams := self >> Checked(IsPosIntSym(self.params[1]), IsRatSym(self.params[2]), 
                                    IsList(self.params[3]), ForAll(self.params[3], IsIntSym),
                                    IsFunction(self.params[4])),

    range := self >> self.params[4].range(),

    # compute() returns [ code := ..., values := ... ],
    compute := self >> self.computeWithOps((res,a,b) -> assign(res, a*b), 
	                                   (res,a,b) -> assign(res, a*conj(b))),

    computeWithOps := meth(self, cxmul, cxmul_conj)
        local exp, W, n, c, new_exp, new_W, i, a, b, v, val, num, perm;
        n := EvalScalar(self.params[1]);

        exp := List(self.params[3], EvalScalar);
        W := List([1..Length(exp)], i -> var.fresh_t(Concat("W",StringInt(exp[i]), "_"), self.range()));
        c := List([1..Length(exp)], i -> assign(W[i], self.params[4].at(i-1)));
                                 

        # NOTE: check if stuck
        while Length(exp) <> (n-1) do
            new_exp := Cartesian(exp, exp);
            new_W := Cartesian(W, W);
            num := Length(exp);
            for i in [1..Length(new_exp)] do
                [a, b] := new_exp[i];
                if (a+b) > 0 and (a+b) < n and not ((a+b) in exp) then
                    v := var.fresh_t(Concat("W",StringInt(a+b), "_"), self.range());
                    Add(exp, a+b);
                    Add(W, v);
                    Add(c, cxmul(v, new_W[i][1], new_W[i][2]));
                fi;
                if (a-b) > 0 and (a-b) < n and not ((a-b) in exp) then
                    v := var.fresh_t(Concat("W",StringInt(a-b), "_"), self.range());
                    Add(exp, a-b);
                    Add(W, v);
                    Add(c, cxmul_conj(v, new_W[i][1], new_W[i][2]));
                fi;
            od;
            if Length(exp) = num then Error("OnlineSkewTwid.compute is stuck, not enough twiddle seed values given"); fi;
        od;
        perm := SortingPerm(exp);
        return rec(values := Permuted(W,perm), exp := Permuted(exp,perm), code := chain(c)); 
    end
));


Class(ATwidOnline, AGenericTag, rec(isPushTag := true));
Class(ATwidSplit, AGenericTag, rec(isPushTag := true));

NewRulesFor(SkewDFT, rec(
    SkewDFT_Base2 := rec(
	applicable := t -> t.params[1] = 2,
	apply := (t, C, Nonterms) -> let(
            den:=Denominator(t.params[2]), num:=Numerator(t.params[2]), k:=t.params[3],
	    F(2) * Diag(1,E(2*den)^(k*num)))),

    SkewDFT_toDFT := rec(
	applicable := (self, t) >> logic_or(eq(self.a.maxSize, -1), leq(t.params[1], self.a.maxSize)),
        children := t -> [[ DFT(_unwrap(t.params[1]), _unwrap(t.params[3])).withTags(
		    Filtered(t.getTags(), t -> not ObjId(t) in [ATwidOnline, ATwidSplit])) ]],

        a := rec(
            diagMode := "normal", # "normal" | "split" | "online" 
            maxSize := -1
        ),

        forTransposition := true,

        onlineSeeds := n -> Cond(n<=4, [1],
                                 n<=8, [1, 3], 
                                       [1, 3, n-1]),

	apply := (self, t, C, Nonterms) >> let(
            N := t.params[1], rot := t.params[3], a := t.params[2], j := Ind(N-1), 
            tw := SkewTwid(N, a*rot),
            mode := Cond(t.firstTag()=ATwidOnline(), 
		         # if other tags are present, do not use "online" mode
		         # this is necessary because we can't vectorize a standalone SkewDFT in this mode
		              Cond(Length(t.getTags())>1, self.a.diagMode, "online"),
                         t.firstTag()=ATwidSplit(), "split", 
                         self.a.diagMode),

	    C[1] * 
            Cond(mode = "split", 
                    DiagCpxSplit(diagDirsum(FList(TReal, [1.0, 0.0]), fPrecompute(RCData(tw)))),
                 mode = "normal" or N=2, 
                    Diag(        diagDirsum(fConst(TReal, 1, 1.0),    fPrecompute(tw))), # XXX

                 mode = "online", let(seeds := self.onlineSeeds(N), 
                     DirectSum(I(1), OnlineDiag(OnlineSkewTwid(N, a*rot, seeds, fPrecompute(fCompose(tw, FList(TInt, seeds-1))))))),
                 Error("<self>.diagMode must be \"split\" | \"normal\" | \"online\""))
        )               
    ),

    #F SkewDFT_Fact : PDFT_2n_a -> L (PDFT_n_r0 dirsum PDFT_n_r1) L 
    #F                               (I2 tensor [[1,r0],[1,r1]]) L
    #F Derived using polynomial factorization:
    #F    (x^2n - a) == (x^n - r0) * (x^n - r1)
    #F where r0 and r1 are two different quadratic roots of a
    #F
    SkewDFT_Fact := rec(
	applicable := t -> t.params[1] > 2 and t.params[1] mod 2 = 0,

	children := t -> let(
	    n:=t.params[1]/2, r0:=t.params[2]/2, r1:=r0+1/2, k:=t.params[3],
	    [[ SkewDFT(n, r0, k), SkewDFT(n, r1, k) ]]),

	apply := (t, C, Nonterms) -> let(
	    n:=t.params[1]/2, den:=Denominator(t.params[2]), num:=Numerator(t.params[2]), k:=t.params[3],
	    DirectSum(C[1], C[2]) ^ L(2*n, 2) * 
	    Tensor(I(n), F(2)*Diag(1,E(2*den)^(k*num))) *
	    L(2*n, n))
    ), 

    #F SkewDFT_CT : SkewDFT_mn(r) -> (SkewDFT_m(.) tensor I_n) (I_m tensor SkewDFT_n(r)) L
    #F
    #F Derived using polynomial decomposition
    #F    x^mn - a == (x^m)^n - a
    #F
    SkewDFT_CT := rec(
        applicable := (self, t) >> logic_and(gt(t.params[1], 2), logic_neg(t.hasTags()), logic_neg(isPrime(t.params[1]))), 
        freedoms := t -> [ divisorsIntNonTriv(t.params[1]) ], 
        child := (t, fr) -> let(
            N := t.params[1],  m := fr[1],          n := div(N, m),  
            a := t.params[2],  rot := t.params[3],  j := Ind(n), 

            [ IDirSum(j, SkewDFT(m, fdiv(a+j, n), rot)), 
              SkewDFT(n, a, rot) ]),

        apply := (nt, C, Nonterms) -> let(m := Rows(Nonterms[1].child(1)), n := Rows(Nonterms[2]),
            Tr(n, m) * C[1] * Tr(m, n) * Tensor(I(m), C[2]) * Tr(n, m)
        )
    )
));
