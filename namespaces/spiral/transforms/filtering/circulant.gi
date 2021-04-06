
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F Circulant(<n>, <filt-func>, <valuation>) -- n x n  circulant matrix non-terminal
#F
#F The circulant matrix with coefficients of the first row given by
#F the filter function <filt-func>.
#F
#F <valuation> is the offset, default = 0, meaning no offset.
#F
#F Examples:
#F
#F spiral> PrintMat(MatSPL( Circulant(5, FList(TInt, [1..3]), 0)));
#F [ [ 1, 2, 3,  ,   ], 
#F   [  , 1, 2, 3,   ], 
#F   [  ,  , 1, 2, 3 ], 
#F   [ 3,  ,  , 1, 2 ], 
#F   [ 2, 3,  ,  , 1 ] ]
#F spiral> PrintMat(MatSPL( Circulant(5, FList(TInt, [1..3]), -1)));
#F [ [ 2, 3,  ,  , 1 ], 
#F   [ 1, 2, 3,  ,   ], 
#F   [  , 1, 2, 3,   ], 
#F   [  ,  , 1, 2, 3 ], 
#F   [ 3,  ,  , 1, 2 ] ]

Class(Circulant, NonTerminal, rec(
    _short_print := true,

    abbrevs := [ 
     (L)     -> [ toSize(L), toFunc(L), toValuation(L) ],
     (n,L)   -> [ Checked(IsPosInt(n), n), toFunc(L), toValuation(L) ],
     (n,L,v) -> [ Checked(IsPosInt(n), n), toFunc(L), Checked(IsInt(v), v) ], 
    ], 

    dims := self >> Replicate(2, self.params[1]), 

    setData := meth(self, newdata) self.params[2] := newdata; return self; end,

    column := self >> let(
	coeffs := List(self.params[2].tolist(), x->x.ev()),
	valuation := self.params[3],
	CircularWrap([coeffs, valuation], self.params[1])),

    terminate := meth(self)
      local i, j, n, L, Ls, mat;
      L := self.column();
      n := Length(L);
      mat := [ ];
      for i in [0..n-1] do
          Ls := Replicate(n, 0);
	  for j in [0..n-1] do Ls[(i-j) mod n + 1] := L[j+1]; od;
	  Add(mat,Ls);
      od;
      return Mat(mat);      
    end,

    filtlen := self >> self.params[2].domain(),

    transpose := self >> Circulant(
	self.params[1],
	fCompose(self.params[2], J(self.filtlen())),
	-self.params[3]-self.filtlen()+1),
  
    isReal := self >> not IsComplexT(self.params[2].range()),

    hashAs := self >> let(
	t := ObjId(self)(
	    self.params[1], fUnk(self.params[2].range(), self.params[2].domain()), self.params[3]),
	When(self.transposed, t.transpose(), t)),

    SmallRandom := () -> let(n := Random([2..16]),
	List([1..n], i -> Float(Random([-2^10..2^10]), Random([-20..10])))),

    LargeRandom := () -> let(n := Random([16, 32, 64]),
	List([1..n], i -> Float(Random([-2^10..2^10]), Random([-20..10]))))
));

## 
## Rules
##
RulesFor(Circulant, rec(
    ###################################################################
    ## Time domain methods
    ###################################################################

    #F Circulant_Base: (base case)
    #F
    Circulant_Base := rec (
	info             := "Circulant -> Mat",
	forTransposition := false,
	isApplicable     := P -> P[1] <= 32, 
	allChildren      := P -> [[ ]],
	rule := (P, C) -> ApplyFunc(Circulant, P).terminate()
    ),

#     Circulant_toFilt := rec (
# 	info             := "Circulant -> VStack(.., Filt, ...)",
# 	forTransposition := false,
# 	isApplicable     := P -> P[1] > 2 and (P[1]-P[2].domain()) > 2 and P[2].domain() <= 32 and P[3] <= 0, 
# 	allChildren      := P -> let(
# 	    n := P[1], l := -P[3], r := P[3] + P[2].domain() - 1,
# 	    [[ Filt(n-l-r, Poly(List(P[2].tolist(),EvalScalar), P[3])) ]]),

# 	rule := (P, C) -> let(
# 	    coeffs := P[2], nc := coeffs.domain(), n := P[1], l := -P[3], 
# 	    r := -l + nc - 1, 
# 	    j := Ind(l), k := Ind(r), v := Ind(nc),
# 	    VStack(
# 		BB(ISum(j, j.range, 
# 		    Scat(fBase(j)) * 
# 		    RowVec(fCompose(coeffs, Lambda(v, cond(leq(v,r+j), v-j+l, v-j+l-nc))))) *
# 		    Gath(n, nc,	Lambda(v, cond(leq(v,r+j), v, v+(n-l-r)-1 )))),
# 		C[1],
# 		BB(ISum(k, k.range, 
# 		    Scat(fBase(k)) * 
# 		    RowVec(fCompose(coeffs, Lambda(v, cond(leq(v,k), v-k+(nc-1), v-k-1)))) *
# 		    Gath(n, nc, Lambda(v, cond(leq(v,k), v, v+(n-l-r)-1 )))))))
#     ),

    #F Circulant_Blocking
    #F 
    #F Circulant -> Blocks (Toeplitzes)
    #F
    #F Circulant is partitioned into square blocks of sizes 
    #F given by all proper divisors of the input size.
    #F 
    #F NOTE: this rule is invalid, Circulant_BlockingDense seems valid
    #F        whats the difference between the two ???
    Circulant_Blocking := rec(
	info             := "Circulant -> Toeplitz",
	forTransposition := false,
	isApplicable     := P -> let(n := P[1],
	    n > 2 and not IsPrime(n)), # and n <= 2*Length(P[2])),

	allChildren := P -> let(
	    divs := DivisorPairs(P[1]),
	    ratio := P[2].domain() / P[1], 
	    List(divs, d -> [ Toeplitz(FUnk(2*d[1]-1), 0, CeilingRat(ratio * (2*d[1]-1))) ])
	),

        prepData := (self, L, b, nc) >> let(
	    n := Length(L),
	    Concatenation(
		List([1..nc], k -> 
		    List([1..2*b-1], j -> L[((n-k*b+j) mod n) + 1])))),

	rule := (self,P,C,Nonterms) >> let(
	    n    := P[1],
	    b    := Rows(C[1]), 
	    nc   := n/b,
	    new_data := FData(self.prepData(ApplyFunc(Circulant, P).column(), b, nc)),
	    i := Ind(nc),
	    k := Ind(nc),
	    ofs := DataInd(TInt, nc * (2*b-1)).setAttr("live_out"),
	    bnum := DataInd(TInt, nc).setAttr("live_out"),
	    gfunc := fCompose(new_data, fAdd(nc*(2*b-1), 2*b-1, ofs)),

	    len := P[2].domain(),
	    val := P[3] mod n,
	    btoep_len := 2*b-1,

	    left := val,
	    right := When(len+b >= n, val-1, (val + (len-1) + (b-1)) mod n), 
	    exact_right := (val + (len-1)) mod n,

	    # we "quantize" the condition to tell us which block #s (ofs) are 0
	    lblock := btoep_len * Int(left/b),
	    rblock := btoep_len * Int(right/b),

	    rproj := b*bnum + (b - 1 - left),
	    lproj := b*bnum + (b - 1 - exact_right), 
	    nt    := Nonterms[1].setData(gfunc)
	                        .setNonZero(lproj, rproj),

	    open   := leq(rblock+1, lblock), 
	    A := leq(lblock, ofs), 
	    B := leq(ofs, rblock),
	    isNonZero := logic_or(logic_and(A, B), logic_and(open, logic_or(A, B))),

	    ISum(i, i.range, 
		Scat(fTensor(fBase(i), fId(b))) * 
		ISumAcc(k, k.range, 
		    Data(bnum, imod(nc+k-i, nc), 
			Data(ofs, (2*b-1)*bnum, 
			    COND(isNonZero, C[1], O(b)))) * 
		    Gath(fTensor(fBase(k), fId(b)))))
        )
        # RowDirectSum( nc*b, List([1..nc], i-> 
        # ColDirectSum( b, List([1..nc], k-> C[((k-i) mod nc) + 1]))))
    ),

    #F Circulant_BlockingDense
    #F 
    #F Circulant -> Blocks (Toeplitzes)
    #F
    #F Circulant is partitioned into square blocks of sizes 
    #F given by all proper divisors of the input size.
    #F 
    Circulant_BlockingDense := rec(
	info             := "Circulant -> Toeplitz",
	forTransposition := false,
	isApplicable     := P -> let(n := P[1],
	    n > 2 and not IsPrime(n) and n <= 2*P[2].domain()),

	allChildren := P -> let(
	    divs := DivisorPairs(P[1]),
	    ratio := P[2].domain() / P[1], 
	    List(divs, d -> [ Toeplitz(FUnk(2*d[1]-1), 0, CeilingRat(ratio * (2*d[1]-1))) ])
	),

        prepData := (self, L, b, nc) >> let(
	    n := Length(L),
	    Concatenation(
		List([1..nc], k -> 
		    List([1..2*b-1], j -> L[((n-k*b+j) mod n) + 1])))),

	rule := (self,P,C,Nonterms) >> let(
	    n    := P[1],
	    b    := Rows(C[1]), 
	    nc   := n/b,
	    new_data := FData(self.prepData(ApplyFunc(Circulant, P).column(), b, nc)),
	    i := Ind(nc),
	    k := Ind(nc),
	    ofs := DataInd(TInt, nc * (2*b-1)).setAttr("live_out"),
	    bnum := DataInd(TInt, nc).setAttr("live_out"),

	    gfunc := fCompose(new_data, fAdd(nc*(2*b-1), 2*b-1, ofs)),

	    len := P[2].domain(),
	    val := P[3] mod n,
	    left := val,
	    exact_right := (val + (len-1)) mod n,

	    rproj := b*bnum + (b - 1 - left),
	    lproj := b*bnum + (b - 1 - exact_right), 
	    nt    := Nonterms[1].setData(gfunc)
	                        .setNonZero(lproj, rproj),

	    ISum(i, i.range, 
		Scat(fTensor(fBase(i), fId(b))) * 
		ISumAcc(k, k.range, 
		    Data(bnum, imod(nc+k-i, nc), 
			Data(ofs, (2*b-1)*bnum, 
			    C[1])) *
		    Gath(fTensor(fBase(k), fId(b)))))
        )
        # RowDirectSum( nc*b, List([1..nc], i-> 
        # ColDirectSum( b, List([1..nc], k-> C[((k-i) mod nc) + 1]))))
    ),

    #F Circulant_DiagonalizeStep
    #F
    #F Circulant_n -> (F2 tensor I2) 
    #F                (Circulant_n/2 dirsum Toeplitz_n/2) 
    #F                (F2 tensor I2)
    #F 
    #F Computes circulant matrix through DFT.
    #F h is the first column of the circulant 
    #F (params of Circulant nonterminal).
    #F
    Circulant_DiagonalizeStep := rec(
	info             := "Circulant_n -> Circulant_n/2 dirsum Toeplitz_n/2",
	forTransposition := false,
	isApplicable     := P -> let(n := P[1],
	    n > 2 and n mod 2 = 0 and n<=16),

	allChildren := P -> let(n := P[1], 
	     [[ Circulant(FUnk(n/2)), Toeplitz(FUnk(n-1)) ]]),

	prepData := function(L)
	    local l, L1, L2, circ, toep;   
	    l := Length(L);
	    L1 := L{[1     .. l/2]}; # first half
	    L2 := L{[l/2+1 ..  l ]}; # second half

	    circ := 1/2 * (L1 + L2);
	    toep := 1/2 * Concat(Drop(L2,1) - Drop(L1,1), 
		                      L1    - L2);
	    return [circ, toep];
	end,

	rule := (self, P, C, Nonterms) >> let(
	    n := P[1], 
	    data := self.prepData(ApplyFunc(Circulant,P).column()),
	    circ := Nonterms[1].setData(FData(data[1])),
	    toep := Nonterms[2].setData(FData(data[2])),

	    Tensor(F(2), I(n/2)) *
	    DirectSum(C[1], C[2]) *
	    Tensor(F(2), I(n/2))
	)
    ),

    ###################################################################
    ## Frequency domain methods
    ###################################################################

    #F Circulant_DFT
    #F
    #F Circulant -> invDFT * diag(DFT(h)) * DFT
    #F 
    #F Computes circulant matrix through DFT.
    #F h is the first column of the circulant 
    #F (params of Circulant nonterminal).
    #F
    Circulant_DFT := rec(
	info             := "Circulant -> invDFT * diag(DFT(h)) * DFT",
	forTransposition := false,
	switch           := false,
	isApplicable     := P -> P[1] > 1,
	allChildren      := P -> [[ transforms.DFT(P[1], -1), transforms.DFT(P[1], 1) ]],

	rule := function(P, C)
            local l, coef, L;
	    L := ApplyFunc(Circulant, P).column();
	    l := Length(L);
	    coef := List([1..l], i -> ComplexAny(L[i]));
	    coef := 1/l * ComplexFFT(coef);
	    return C[1] * Diag(FData(coef)) * C[2];
	end
    ),

    #F Circulant_RDFT
    #F
    #F Circulant -> invRDFT * block(RDFT(h)) * RDFT
    #F 
    #F Computes circulant matrix through the Real DFT.
    #F h is the first column of the circulant 
    #F (params of nonterminal Circulant).
    #F
    Circulant_RDFT := rec(
	info             := "Circulant -> invRDFT * block(RDFT(h)) * RDFT",
	forTransposition := false,
	switch           := false,

	isApplicable     := P -> let(n := P[1],
	    n > 2 and n <= 64 and (n mod 2 = 0) and
	    ForAll(Factors(n), f -> f in [2, 3])),

	allChildren      := P -> let(n := P[1],
	    [[ RDFT(n).transpose(), RDFT(n) ]]),

	rule := function( P, C )
 	    local i, L, M, X, l, coef, ccoef;

	    L := ApplyFunc(Circulant, P).column();
	    l := Length(L);
	    ccoef := List([1..l], i -> ComplexAny(L[i]));
	    ccoef := ComplexFFT(ccoef);   
	    coef  := List([1..Int(l/2)+1], i -> 2/l * ReComplex(ccoef[i]));
	    Append(coef, List([Int(l/2)+2..l], i -> 2/l * (-ImComplex(ccoef[i])) ));

	    X := Diag([ 1/2 * coef[1]]);
	    for i in [2..Int(l/2)] do
	        M := Mat([[coef[i],      coef[l-i+2]], 
			  [-coef[l-i+2], coef[i]]]);
		X := DirectSum(X, M);
	    od;
	    if (l mod 2 = 0) then 
		X := DirectSum(X, Diag( [1/2 * coef[l/2+1]] ));
	    fi;
	    X := X^perm4(l);
	    return C[1] * X * C[2];
	end
    ),

    #F Circulant_PRDFT
    #F
    #F Circulant -> IPRDFT * diag(PRDFT(h)) * PRDFT
    #F 
    #F Computes circulant matrix through the Packed Real DFT.
    #F h is the first column of the circulant 
    #F (params of nonterminal Circulant).
    #F
    Circulant_PRDFT := rec(
	info             := " Circulant -> IPRDFT * diag(PRDFT(h)) * PRDFT",
	forTransposition := false,
	switch           := false,

	isApplicable     := P -> true,

	allChildren      := P -> let(n := P[1],
	    [[ IPRDFT(n), PRDFT(n) ]]),

	rule := function( P, C )
 	    local i, n, L, l, coef, D;
	    n := Cols(C[1]);
	    L := ApplyFunc(Circulant, P).column();
	    l := Length(L);
	    coef := List(L, EvalScalar);
	    coef := 1/l * ComplexFFT(coef);
	    D:=Diag(FData(coef{[1..n/2]}));
	    return C[1] * RC(D) * C[2]; 
	end
    ), 

    #F Circulant_DHT
    #F
    #F Circulant -> DHT * block(DHT(h)) * DHT
    #F 
    #F Computes circulant matrix through the Discrete Hartley transform.
    #F h is the first column of the circulant 
    #F (params of nonterminal Circulant).
    #F
    Circulant_DHT := rec(
	info             := "Circulant -> DHT * block(DHT(h)) * DHT",
	forTransposition := false,
	switch           := false, 
	isApplicable     := P -> let(n := P[1], n > 1 and Is2Power(n)),

	allChildren      := P -> let(n := P[1], [[ DHT(n) ]]),

	rule := function(P, C)
	    local i,M,X,l,L,coef,ccoef, rcoef, icoef;
	    L := ApplyFunc(Circulant, P).column();
	    l := Length(L);
	    ccoef := List([1..l], i -> ComplexAny(L[i]));
	    ccoef := ComplexFFT(ccoef);   
	    rcoef := List([1..l/2+1], i -> 1/l * ReComplex(ccoef[i]));
	    icoef := List([2..l/2], i -> 1/l * ImComplex(ccoef[i]));
	    X := Diag([rcoef[1]]);
	    for i in [2..Int(l/2)] do
	        M := Mat([[rcoef[i],   -icoef[i-1]], 
			  [icoef[i-1], rcoef[i]]]);
		X := DirectSum(X,M);
	    od;
	    X := DirectSum(X, Diag( [rcoef[l/2+1]] ));
	    X := X^perm4(l);
	    return C[1] * X * C[1];
	end
    )
));
