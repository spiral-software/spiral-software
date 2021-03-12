
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# alpha is a function of P (divisor of N)
# m = N/P

# Frequency domain, u in [0, N)
# Per processor, total pts=2*m, total useful pts=m
# We start to see aliasing at 3*m/2, m=N/P.
W := (alpha, u) -> exp(-alpha * u^2);

# Time domain, t in [0, 1]
w := (alpha, t) -> sqrt(d_PI / alpha) * exp(-d_PI^2 * t^2 / alpha);

# digits - eg. 17, number of exact (unaffected by aliasing) decimal digits
gauss_alpha := (N, p, digits) -> let(m := N/p, 4/9 * digits * d_log(10) / m^2);

# how many samples of the filter to keep in the time domain, assuming gauss_alpha(N, p, digits)
# was used.
edge := digits -> IntDouble(d_ceil(2/(3*d_PI) * d_log(10) * digits));


Class(fPerExt, FuncClass, rec(
    def := (N, l, r) -> rec(N:=N, n:=N+l+r),
    lambda := self >> let(N := self.params[1], l:=self.params[2], r:=self.params[3],
 	j := Ind(self.n),
	Lambda(j, imod(j-l, N)))
));

# below we should not use HZ, since HZ assumes that n * stride = N, and if used as below
# assumption will be violated
#RewriteRules(RulesFuncSimp, rec(
#    fPerExt_H := ARule(fCompose, [@(1,fPerExt), @(2,H)], e -> 
#	let(base := @(2).val.params[3], stride := @(2).val.params[4], 
#	[ HZ(@(1).val.N, @(2).val.n, base - @(1).val.params[2], stride) ]))
#));


NewRulesFor(DFT, rec(

   DFT_Gauss := rec(
	forTransposition := true,
	minSize     := 4,
	maxSize     := false,
	p      := 4,
	digits := 15,

	applicable  := (self, t) >> (self.minSize<>false and Rows(t) > self.minSize) and
	                            not IsPrime(Rows(t)) and t.params[2] in [1,-1 mod Rows(t)],

        libApplicable := (self, t) >> eq(imod(t.params[1], self.p), 0),
	freedoms := (self, t) >> [],

	child := (self, t, freedoms) >> let(N := Rows(t), p := self.p,
	    [ DFT(2*N/p, t.params[2]), DFT(p, t.params[2]) ]), 
       
	apply := meth(self, t,C,Nonterms) 
	    local N, m, n, rot, k, p, alpha, time_filter, freq_filter, j;
	    N := Rows(t); 
	    n := Rows(C[2]);
	    k := edge(self.digits);
	    p := self.p;
	    m := N/p;
	    alpha := gauss_alpha(N, p, self.digits);

	    time_filter := let(i:=Ind(2*k*p), Lambda(i, w(alpha, (i-k*p)/N)).setRange(TDouble));
	    freq_filter := let(i:=Ind(m),     Lambda(i, p/(2*N*W(alpha, i-m/2))).setRange(TDouble)); 
	    
	    j := Ind(p);
	    return 
	    Grp(
		IterDirectSum(j, j.range, 
		    Diag(freq_filter) * 
		    Gath(H(2*m, m, m*imod(j,2), 1)).toloop(4) * C[1]
		) *
		(Tr(2*N/p, p))
	    ) * 
	    RowTensor(2*N/p, 2*k*p-p/2, 
		C[2] * 
		let(i:=Ind(p), Diag(Lambda(i, omega(2*p, i)).setRange(TComplex))) *
		Tensor(RowVec(List([0..2*k-1], x->(-1)^x)), I(p)) * 
		Diag(time_filter)
	    ) * 
	    Gath(fPerExt(N, k*p, k*p-p/2));
	end
	    
   )
));

# scaling by p/(2*N) belongs to the time_filter, but freq_filter is shorter
#	    time_filter := List([-k*p .. k*p-1], i-> w(alpha, i/N));
#	    freq_filter := List([-m/2..m/2-1], i-> p/(2*N*W(alpha, i))); 

#	    return
#	    Z(N, m/2) *
#	    Tensor(I(p/2), 
#		   DirectSum(
#		       Diag(freq_filter) * Gath(fStack(H(2*m, m/2, 3*m/2, 1), H(2*m, m/2, 0, 1))) * C[1],
#		       Diag(freq_filter) * Gath(H(2*m, m, m/2, 1)) * C[1])
#	    ) * 


testGauss := function(N, p, k)
    local s, me, them;
    DFT_Gauss.p := p;
    SwitchRulesByNameQuiet(DFT, [DFT_Gauss]);
    s := SPLRuleTree(ExpandSPL(DFT(N,k))[1]);
    me := MatSPL(s);
    them := MatSPL(DFT(N,k));
    VisualizeMat(me-them, " ");
    return [s, me, them];
end;

splGauss := function(N, p, k)
    local s, me, them;
    DFT_Gauss.p := p;
    SwitchRulesByNameQuiet(DFT, [DFT_Gauss]);
    s := SPLRuleTree(ExpandSPL(DFT(N,k))[1]);
    return s;
end;

sums := function(s, opts)
    local real;
    real := s.isReal();
    s := SumsSPL(s);
    s := SubstBottomUp(s, @.cond(IsNonTerminal), x->RecursStep(0,0,x));
    s := ApplyStrategy(s, opts.formulaStrategies.sigmaSpl, UntilDone);
    if not opts.generateComplexCode and not real then
        s := ApplyStrategy(RC(s), opts.formulaStrategies.rc, UntilDone); fi;
    return ApplyStrategy(s, opts.formulaStrategies.postProcess, UntilDone);
end;

# opts := CopyFields(SpiralDefaults, rec(generateComplexCode := true));
# opts := CopyFields(opts, rec(unparser := CMacroUnparser));

# N := 128;
# k := edge(17);
# p := 4;
# alpha := alpha(N, p, 17);

NewRulesFor(DFT, rec(

   DFT_PrunedGauss := rec(
	forTransposition := true,
	minSize     := 4,
	maxSize     := false,
	p := 2,
	digits := 15,

	applicable  := (self, t) >> (self.minSize<>false and Rows(t) > self.minSize) and
	                            not IsPrime(Rows(t)) and t.params[2] in [1,-1 mod Rows(t)],

	children := (self, t) >> let(N := Rows(t), p := self.p,
	    [[ DFT(2*N/p, t.params[2]), DFT3(p, t.params[2]) ]]), #DFT3
       
	apply := meth(self, t,C,Nonterms) 
	    local N, m, n, mat, rot, k, p, alpha, time_filter, freq_filter, freq_filter2;
	    N := Rows(t); 
	    n := Rows(C[2]);
            mat := MatSPL(C[2]);
	    k := edge(self.digits);
	    p := self.p;
	    m := N/p;
	    alpha := gauss_alpha(N, p, self.digits);
	    time_filter := List([-k*p .. k*p-1], i-> w(alpha, i/N).eval());
	    freq_filter := List([-m/2..m/2-1], i-> p/(2*N*W(alpha, i).eval())); 

	    return 
	    Diag(freq_filter) * Gath(H(2*m, m, 0, 1)) * C[1] *
	    RowTensor(2*N/p, 2*k*p-p/2, 
		RowVec(mat[1]) *
		Tensor(RowVec(List([0..2*k-1], x->(-1)^x)), I(p)) * 
		Diag(time_filter)) *
	    transforms.filtering.Ext(N, transforms.filtering.per(k * p), 
		                        transforms.filtering.per(k * p - p/2));
	end
   )
));

#2N/p  * 2kp
#4Nk complex fmas overhead for filter (one stripe out), k=edge(digits)
