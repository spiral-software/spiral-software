
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Transform symbol: Toeplitz
#
# Parameters:       <list>
#
# Definition:       Toeplitz(L) represents the (n X n) toeplitz matrix. 
#                   L must contain the coefficients of the first row (from
#                   right to left) followed by all but first coefficents 
# 		    on the first column specified. Toeplitz matrices have 
#                   constant elements along all diagonals.
#
# 		    n = (len(L)+1) / 2.
#
# Example:          Transform( "Toeplitz", [1, 2, 3, 4, 5, 6, 7]) is the matrix
#                     [ [4, 3, 2, 1],
#                       [5, 4, 3, 2],
#                       [6, 5, 4, 3],
#                       [7, 6, 5, 4] ]
Class(Toeplitz, NonTerminal, DataNonTerminalMixin, rec(
    # above DataNonTerminalMixin must come after NonTerminal to pickup
    # NonTerminal.operations instead of DataNonTerminalMixin.operations
    # which is actually ClassBase.operations
    abbrevs := [ 
	L -> let(f:=toFunc(L), 
	         [ Checked(IsOddInt(f.domain()), f), 0, f.domain()-1 ]),
	(L, l, r) -> let(f:=toFunc(L), 
	         [ Checked(IsOddInt(f.domain()), f), l, r ] )
    ],

    dims := self >> Replicate(2, (self.params[1].domain()+1)/2),

    terminate := self >> let(
	coeffs := self.params[1],
	#lcoeffs := coeffs.lambda(),
	coeffs_ev := coeffs.lambda().tolist(),
	len := coeffs.domain(),
	l := self.params[2],
	r := self.params[3],
	#v := Ind(len),
        #toeplitz( Lambda(v, cond(leq(l,v,r), lcoeffs.at(v), 0)).tolist() )),
        toeplitz(coeffs_ev)),

    transpose := self >> let(nums := self.params[1],
        Toeplitz(fCompose(nums, J(nums.domain())))),
  
    isReal := self >> true,

    HashId := self >> let(n := self.params[1].domain(), l := self.params[2], r := self.params[3],
#	[n, When(IsInt(l) and IsInt(r), QuantizeQuarters(((r-l) mod n)/(n-1)), 1)]),
	n),

    hprint := self >> let(n := self.params[1].domain(), h := self.HashId, 
	Print(self.name, "(", n, ")")), #, ", 0, ", ", Int((n-1)*h[2]), ")")),

    SmallRandom  := () -> [1 .. 1+2*Random([1..8])],

    setNonZero := meth(self, l, r)
        self.params[2] := l;
	self.params[3] := r;
    end
));

## 
## Rules
##
RulesFor(Toeplitz, rec(
    ###################################################################
    ## Time domain methods
    ###################################################################
     
    #F Toeplitz_Base: (base case)
    #F
    #F NonTerm_Filter -> SPLMat
    #F Computes convolution by definition
    #F
    Toeplitz_Base := rec(
	info             := "Toeplitz -> Mat",
	forTransposition := false,
	limit            := 8,
	isApplicable     := (self, P) >> let(n := (P[1].domain()+1)/2, n <= self.limit),
	allChildren      := P -> [[ ]],
	rule := (P, C) -> ApplyFunc(Toeplitz, P).terminate()
    ),

    Toeplitz_SmartBase := rec(
	info             := "Toeplitz -> Lower-Triangular | Upper-Triangular | Full matrix",
	forTransposition := false,
	limit            := 5,
	isApplicable     := (self, P) >> let(n := (P[1].domain()+1)/2, n <= self.limit),
	allChildren      := P -> [[ ]],

	rule := (P, C) -> let(
	    coeffs := P[1],
	    coeffs_ev := coeffs.lambda().tolist(),
	    len := coeffs.domain(), # 7 (r 6,5,4,<3>,2,1,0 l)
	    halflen := (len-1) / 2, # 3 
	    l := P[2],
	    r := P[3],
	    closed := leq(l, r),

	    A := leq(halflen, l),
	    B := leq(r, halflen),

	    lout := leq(len, l),
	    rout := leq(r, -1),

	    COND(logic_and(A, logic_or(closed, rout)), # lower triangular
		 toeplitz(Concat(Replicate(halflen,0), Drop(coeffs_ev, halflen))), 
		 COND(logic_and(B, logic_or(closed, lout)), # upper triangular
		      toeplitz(Concat(Take(coeffs_ev, halflen+1), Replicate(halflen,0))),
		      toeplitz(coeffs_ev)))) # full
    ),

    Toeplitz_PreciseBase := rec(
	info             := "Toeplitz -> toeplitz of non-zero elements only",
	forTransposition := false,
	switch := false,
	isApplicable     := P -> let(len := P[1].domain(), len = 3),
	allChildren      := P -> [[ ]],
	rule := (P, C) -> let(
	    lcoeffs := P[1].lambda(),
	    len := P[1].domain(),
	    l := P[2],
	    r := P[3],
	    v := Ind(len),
	    toeplitz( Lambda(v, cond(leq(l,v,r), lcoeffs.at(v), 0)).tolist() ))
    ),

    #F Toeplitz_Blocking
    #F 
    #F Toeplitz -> Blocks (Toeplitzes)
    #F
    #F Toeplitz is divided into square blocks of sizes given by all
    #F proper divisors of the input size.
    #F 
    Toeplitz_Blocking := rec(
	info             := "Toeplitz_nk -> Toeplitz_k",
	forTransposition := false,

	isApplicable     := P -> let(N := (P[1].domain() + 1) / 2,
	    N > 2 and not IsPrime(N)),

	allChildren := P -> let(
	    len := P[1].domain(), N := (len+1)/2, 
	    l := P[2], r:= P[3], nzlen := r-l+1, ratio := When(IsInt(nzlen), nzlen/len, 1),
	    List(DivisorPairs(N), d -> let(newlen := 2*d[1]-1, 
		[ Toeplitz(FUnk(newlen), newlen - CeilingRat(newlen*ratio), newlen-1) ]))),

	rule := (P, C, Nonterms) -> let(
	    len := P[1].domain(), 
	    N := (len + 1) / 2,
	    bksize := Rows(C[1]),
	    bklen  := 2*bksize - 1,
	    bkdim := N / bksize,	    
	    numblocks := 2 * bkdim - 1, 
	    i := Ind(bkdim),
	    k := Ind(bkdim),
	    ofs := DataInd(TInt, N*2-bklen).setAttr("live_out"), 
	    gfunc := fAdd(len, bklen, ofs), 

	    l := P[2], 
	    r := P[3],
	    open   := leq(r+1, l), 
	    A := leq(l-bklen+1, ofs), 
	    B := leq(ofs, r),
	    isNonZero := logic_or(logic_and(A, B), logic_and(open, logic_or(A, B))),

            nt := Nonterms[1].setData(fCompose(P[1], gfunc))
	                     .setNonZero(l-ofs, r-ofs),
	    
	    ISum(i, i.range, 
		Scat(fTensor(fBase(i), fId(bksize))) * 
		ISumAcc(k, k.range, 
		    Data(ofs, bksize*((bkdim-1)+i-k),
		       COND(isNonZero, C[1],
			    O(bksize))) * 
		    Gath(fTensor(fBase(k), fId(bksize)))))
	)
    ),

    Toeplitz_BlockingDense := rec(
	info             := "Toeplitz_nk -> Toeplitz_k",
	forTransposition := false,

	isApplicable     := P -> let(N := (P[1].domain() + 1) / 2,
	    N > 2 and not IsPrime(N)),

	allChildren := P -> let(
	    len := P[1].domain(), N := (len+1)/2, 
	    l := P[2], r:= P[3], nzlen := r-l+1, ratio := When(IsInt(nzlen), nzlen/len, 1),
	    List(DivisorPairs(N), d -> let(newlen := 2*d[1]-1, 
		[ Toeplitz(FUnk(newlen), newlen - CeilingRat(newlen*ratio), newlen-1) ]))),

	rule := (P, C, Nonterms) -> let(
	    len := P[1].domain(), 
	    N := (len + 1) / 2,
	    bksize := Rows(C[1]),
	    bklen  := 2*bksize - 1,
	    bkdim := N / bksize,	    
	    numblocks := 2 * bkdim - 1, 
	    i := Ind(bkdim),
	    k := Ind(bkdim),
	    ofs := DataInd(TInt, N*2-bklen).setAttr("live_out"), 
	    gfunc := fAdd(len, bklen, ofs), 
	    isNonZero := When(P[3]-P[2]+1=len, V(1), leq(P[2]-bklen+1, ofs, P[3])), 
            nt := Nonterms[1].setData(fCompose(P[1], gfunc))
	                     .setNonZero(P[2]-ofs, P[3]-ofs),
	    
	    ISum(i, i.range, 
		Scat(fTensor(fBase(i), fId(bksize))) * 
		ISumAcc(k, k.range, 
		    Data(ofs, bksize*((bkdim-1)+i-k),
		       C[1]) * 
		    Gath(fTensor(fBase(k), fId(bksize)))))
	)
    ),
    ###################################################################
    ## Frequency domain methods
    ###################################################################

    #F Toeplitz_ExpandedConvolution
    #F
    #F Toeplitz -> Toeplitz_n -> submatrix Circulant(k)  where k>=2*n-1
    #F 
    #F Computes Toeplitz matrix through convolution of expanded size.
    #F
    #F Van Loan, C. "Computational Frameworks for the Fast Fourier Transform",
    #F SIAM Philadelphia 1992, p208.
    #F
    Toeplitz_ExpandedConvolution := rec(
	info             := "Toeplitz_n -> submatrix Circulant(k), k>=2*n-1 ",
	forTransposition := false,
	switch           := false,
	isApplicable     := P -> P[1].domain() >= 5, 
	allChildren      := function( P )
	    local conv_size,l,L,toep_size;
	    l := P[1].domain();
	    L := P[1].lambda().tolist();
	    conv_size := 2^LogInt(l, 2);
	    while conv_size < l do conv_size := conv_size*2; od;
	    return [[ Circulant(
		Concatenation(
		    List([1..(l+1)/2], x->L[(l+1)/2+x-1]),
		    List([1..conv_size-l], x->0),
		    List([1..(l-1)/2], x->L[x]))) ]];
	end, 
	
	rule := (P, C) -> let(n := (P[1].domain() + 1)/2, 
	    RI(n, Rows(C[1])) * 
	    C[1] *
	    RI(Rows(C[1]), n)
	)
    )
));

