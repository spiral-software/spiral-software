
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(DFT_Rader, DFT_GoodThomas);

_accurate_dft := vec -> List(TransposedMat(MatSPL(transforms.DFT(Length(vec)))*TransposedMat([vec]))[1], ComplexCyc);

Class(fdft, FuncExp, rec(
    computeType := self >> self.args[1].t
));

Class(fDFT, DiagFunc, rec(
    abbrevs := [ func -> Checked(IsFunction(func), [func]) ],
    range  := self >> TComplex, 
    domain := self >> self.params[1].domain(), 

    lambda := self >> fdft(self.params[1].lambda()),

    tolist := self >> let(
	elts := List(self.params[1].tolist(), x->Complex(EvalScalar(x))),
	ComplexFFT(elts))
));

NewRulesFor(DFT, rec(
    #F DFT_Base: DFT_2 = F_2
    #F
    DFT_Base := rec(
        info             := "DFT_2 -> F_2",
        forTransposition := false,
        applicable       := nt -> nt.params[1] = 2 and not nt.hasTags(),
        apply            := (nt, C, cnt) -> F(2)
    ),

    DFT_Base1 := rec(
        info             := "DFT_1 -> I(1)",
        forTransposition := false,
        applicable       := nt -> nt.params[1] = 1 and not nt.hasTags(),
        apply            := (nt, C, cnt) -> I(1)
    ),

    #F DFT_Canonize: DFT_n_k = perm * DFT_n
    #F
    DFT_Canonize := rec(
       switch           := false,
       info             := "DFT_n_k -> perm * DFT_n",
       forTransposition := false,
       applicable       := nt -> nt.params[1] > 2 and nt.params[2] <> 1 and not nt.hasTags(),
       children         := nt -> [[ DFT(nt.params[1], 1) ]],
       apply            := (nt, C, cnt) -> OddStride(nt.params[1], nt.params[2]) * C[1]
    ),

    #F DFT_CT: 1965
    #F   General Cooley-Tukey Rule
    #F   DFT_n = (DFT_n/d tensor I_d) * diag * (I_n/d tensor F_d) * perm
    #F
    #F Cooley/Tukey:
    #F   An Algorithm for the Machine Calculation of Complex Fourier Series.
    #F   Mathematics of Computation, Vol. 19, 1965, pp. 297--301.
    #F
    DFT_CT := rec(
        info          := "DFT(mn,k) -> DFT(m, k%m), DFT(n, k%n)",

        maxSize       := false,
        forcePrimeFactor := false,

        applicable := (self, nt) >> nt.params[1] > 2
            and not nt.hasTags()
            and (self.maxSize=false or nt.params[1] <= self.maxSize)
            and not IsPrime(nt.params[1])
            and When(self.forcePrimeFactor, not DFT_GoodThomas.applicable(nt), true),

        children  := nt -> Map2(DivisorPairs(nt.params[1]),
            (m,n) -> [ DFT(m, nt.params[2] mod m), DFT(n, nt.params[2] mod n) ]
        ),

        apply := (nt, C, cnt) -> let(mn := nt.params[1], m := Rows(C[1]), n := Rows(C[2]),
            Tensor(C[1], I(n)) *
            Diag(fPrecompute(Tw1(mn, n, nt.params[2]))) *
            Tensor(I(m), C[2]) *
            L(mn, m)
        )
    ),

    DFT_CT_Mincost := rec(
        maxSize := false,
        applicable := (self, nt) >> let(N := nt.params[1],
	    N > 2 and not nt.hasTags() and IsEvenInt(N) and
            (self.maxSize = false or nt.params[1] <= self.maxSize)),

        children  := nt -> let(N := nt.params[1], k := nt.params[2], Map2( 
	    Filtered(DivisorPairs(N), divpair -> IsEvenInt(divpair[2])),
            (m,n) -> [ DFT(m, k mod m), DFT3(m, k mod (2*m)), DFT(n, k mod n) ])),

        apply := (nt, C, cnt) -> let(
	    N := nt.params[1], m := Rows(C[1]), n := Rows(C[3]), k := nt.params[2],
            hf := (n-2)/2,     i := Ind(hf),    j := Ind(hf),

            SUM(        _hhi(N,  m, 0,      n,  C[1]), # first iteration has no twiddles
              When(hf=0, [],
		ISum(i, _hhi(N,  m, i+1,    n,  C[1] * Diag(fPrecompute(Twid(N,m,k,0,0,i+1)))))),
                        _hhi(N,  m, hf+1,   n,  C[2]), # middle is a DFT3
	      When(hf=0, [],
		ISum(j, _hhi(N,  m, j+hf+2, n,  C[1] * Diag(fPrecompute(Twid(N,m,k,0,0,j+2+hf))))))
            ) *
            Tensor(I(m), C[3]) *
            Tr(n, m)
        )
    ),

    #F DFT_CosSinSplit:
    #F
    #F   DFT_n = CosDFT_n + i * SinDFT_n
    #F
    #F This rule has been restricted in applicability to small size
    #F (see config.g), Connects to Vetterli rules for CosDFT/SinDFT.
    #F
    DFT_CosSinSplit := rec(
        info             := "DFT_n -> CosDFT_n, SinDFT_n",
        forTransposition := false,
        maxSize := 64,

        applicable := (self, nt) >> let(N:=nt.params[1], rot:=nt.params[2],
            N > 2 and N <= self.maxSize and Is2Power(N) and AbsInt(rot)=1 
	    and not nt.hasTags()),

        children := nt -> [[ CosDFT(nt.params[1]), SinDFT(nt.params[1]) ]],

        apply := (nt, C, cnt) -> let(N := nt.params[1], rot := nt.params[2],
	    Tensor(Mat([[1,1]]), I(N)) * 
            VStack(C[1], (E(4)^rot) * C[2])
	)
    ),

    #F DFT_Rader: Rader's Algorithm for Prime size DFT
    #F
    #F   DFT(p) -> P(g)' * (1 dirsum DFT(p-1)) * Tp * (1 dirsum DFT(p-1)) * P(g)",
    #F   P(g) = perm
    #F   Tp   = [[1,1],[1,-1/(p-1)]] dirsum diag
    #F
    DFT_Rader := rec(
        info             := "DFT(p) -> P(g)' * DFT(p-1) * Tp ",
        forTransposition := false,
        minSize          := 3,
        maxSize          := 64,
        accurate         := true,

        raderDiag := (self, N, k, root) >> let(
	    dft := When(self.accurate, _accurate_dft, vec->ComplexFFT(List(vec,ComplexCyc))),
            SubList(dft(List([0..N-2], x -> E(N)^(k*root^x mod N))), 2) / (N-1)),

        raderMid := (self, N, k, root) >>
                     DirectSum(Mat([[1, 1], [1, -1/(N-1)]]),
                       Diag(FData(self.raderDiag(N, k, root)))),

        applicable := (self, nt) >> let(n:=nt.params[1],
            n >= self.minSize and n <= self.maxSize and IsPrime(EvalScalar(n)) 
	    and not nt.hasTags()),

        children      := nt -> [[ DFT(nt.params[1]-1, -1), DFT(nt.params[1]-1, -1) ]],

        apply := (self, nt, C, cnt) >> let(
	    N := EvalScalar(nt.params[1]), k := EvalScalar(nt.params[2]), root := PrimitiveRootMod(N),
            RR(N, 1, root).transpose() *
            DirectSum(I(1), C[1]) *
            self.raderMid(N,k,root) *
            DirectSum(I(1), C[2]) *
            RR(N, 1, root)
        )
    ),

    DFT_RealRader := CopyFields(~.DFT_Rader, rec(
        children      := nt -> [[ PRDFT1(nt.params[1]-1).transpose(), PRDFT1(nt.params[1]-1) ]],

        realRaderDiag := (self, N, k, root) >> let(cdiag := self.raderDiag(N, k, root),
            # cdiag is of length N-2.
            # cdiag = sublist( DFT_{N-1} * omega_vector, [2..N-1])
            # now we want to obtain
            #   rdiag = sublist( RDFT_N * omega_vector, [2..N-1]), we can do this using
            #   the property RDFT = (1 dirsum (I tensor [[1, 1], [-j, j]]) dirsum 1) * LIJ * DFT
            #   which means that rdiag = (1 dirsum (I tensor [[1, 1], [-j, j]]) dirsum 1) * LIJ * cdiag
            #   the code below does this ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ plus pads a zero (for PRDFT)
            #   and also the following trick.

            #  The expected rdiag steady state is [[a, b], [b, -a]], we convert this to rotations (=RCDiag)
            #  by applying J(2), ie we get [[b, -a], [a, b]], which would be a rotation by b + E(4)*a,
            #  but in this case both a and b are complex, so its not. We still use RCDiag though.
            #
            Concatenation(
                ConcatList([1..(N-3)/2], i -> Reversed([cdiag[i]+cdiag[N-1-i], -E(4)*(cdiag[i]-cdiag[N-1-i])])), # DFT->RDFT
                [0, cdiag[(N-3)/2+1]])),

        raderMid := (self, N, k, root) >>
            DirectSum(Mat([[1, 1, 0], [1, -1/(N-1), 0], [0, 0, 0]]), # zero padding for PRDFT
                Tensor(I((N-1)/2), J(2))*
                RCDiag(FData(self.realRaderDiag(N, k, root))).toloop()), 
                # RCDiag with complex entries, max weird :) !
                # .toloop() is used because in vector code RCDiag is 
                # converted to VRCDiag which is confused by complex entries AFAIK
        apply := (self,nt,C,cnt) >> let(
        N := EvalScalar(nt.params[1]), k := EvalScalar(nt.params[2]), root := PrimitiveRootMod(N),
            RR(N, 1, root).transpose() *
            DirectSum(I(1), C[1]) *
            self.raderMid(N,k,root) *
            DirectSum(I(1), C[2]) *
            RR(N, 1, root))
    )),

    #F DFT_Bluestein : Convert FFT to Toeplitz matrix, then embed in a larger Circulant
    #F
    #F    DFT_n -> diag * Toeplitz * diag
    #F
    # DFT on inputs, some of which are 0:
    #
    DFT_Bluestein := rec(
        info             := "DFT_n -> diag * Toeplitz * diag",
        forTransposition := false,
        minSize := 3,
        maxSize := false,
        switch := false,

        # Applicable only for non-powers of 2 to prevent cycles
        applicable := (self, nt) >> let(N := nt.params[1],
            IsSymbolic(N) or 
	    (N >= self.minSize and When(IsInt(self.maxSize), N <= self.maxSize, true) and not Is2Power(N)
		and not nt.hasTags())),

	# NOTE: extend this to be a true degree of freedom
	freedoms := t -> let(min := 2*t.params[1]+1,
	    [ integersBetween(min, 4 * pow(2, floor(log(t.params[1])/log(2)))) ]),

        child := (t, fr) -> let(
	    N := t.params[1], Ncirc := fr[1], 
            [ DFT(Ncirc, -1).withTags(t.getTags()), DFT(Ncirc, 1).withTags(t.getTags()) ]),

        diag := (N, k) -> let(i:=Ind(N), Lambda(i, omegapi(fdiv(i^2 * k, N)))),

	circ := (N, k, circsize) -> let(
	    j1 := Ind(N),
	    j2 := Ind(N-1),
	    diagDirsum(
		Lambda(j1, fdiv(omegapi(fdiv(-j1^2*k, N)), circsize)),
		fConst(TReal, circsize-2*N+1, 0),
		Lambda(j2, fdiv(omegapi(fdiv(-(N-j2-1)^2*k, N)), circsize)))),

        apply := (self, nt, C, cnt) >> let(
            N := nt.params[1],
            k := nt.params[2],
            Ncirc := Rows(C[1]),
            diag := fPrecompute(self.diag(N,k)),
	    pad  := spiral.paradigms.common.TRDiag(Ncirc, N, diag).withTags(nt.getTags()),

	    #Diag(diag) * Gath(fCompose(fAdd(Ncirc, N, 0))).toloop(1) * 
	    pad.transpose() * 
	    Inplace(Grp(C[1] * Diag(fPrecompute(fDFT(self.circ(N, k, Ncirc)))))) *
	    Inplace(C[2]) *
	    pad
	    #Scat(fAdd(Ncirc, N, 0)).toloop(1) * Diag(diag) ##, O(Ncirc-N, N))
	)
    ),

    #F DFT_GoodThomas : Prime Factor FFT
    #F
    #F     DFT_n*k -> perm * (DFT_n_a tensor DFT_k_b) * perm
    #F     when gcd(n,k) = 1
    #F
    DFT_GoodThomas := rec(
        info             := "DFT_n*k -> perm * (DFT_n_a tensor DFT_k_b) * perm",
        forTransposition := false,
        applicable     := (self, nt) >> let(N := nt.params[1],
	    N > 2 and DivisorPairsRP(N) <> [] and N <= self.maxSize and
	    not nt.hasTags()),

        maxSize := 64,

        children := nt -> let(mn := nt.params[1], k:= nt.params[2], Map2(DivisorPairsRP(mn),
            (m,n) -> [ DFT(m, k*n mod m), DFT(n, k*m mod n) ])),

        apply := (nt, C, cnt) -> let(
            r  := Rows(C[1]),
            s  := Rows(C[2]),
            alpha := 1 / s mod r,
            beta  := 1 / r mod s,
            CRT(r,s,1,1).transpose() * Tensor(C[1], C[2]) * CRT(r,s,1,1)
        )
    ),

    #F DFT_PFA_SUMS : Prime Factor FFT using Sigma-SPL and cyclic shifts
    #F
    #F     DFT_n*k -> perm * (DFT_n_a tensor DFT_k_b) * perm
    #F     when gcd(n,k) = 1
    #F
    DFT_PFA_SUMS := CopyFields(~.DFT_GoodThomas, rec(
        apply := (nt, C, cnt) -> let(
	    r := Rows(C[1]), s := Rows(C[2]),
	    j := Ind(s), k := Ind(r),
	    alpha := (1/s) mod r, beta := (1/r) mod s,
	    ISum(j,
                Scat(fTensor(fId(r), fBase(j))) *
                (C[1] ^ Z(r, -alpha*j)) *
                Gath(fTensor(fId(r), fBase(j)))) *
	    ISum(k,
                Scat(fTensor(fId(s), fBase(k))) *
                (C[2] ^ Z(s, -beta*k)) *
                Gath(fTensor(fId(s), fBase(k))))
	)
    )),

    DFT_PFA_RaderComb := rec(
        forTransposition := false,
        applicable := nt -> let(N := nt.params[1], divs := DivisorPairsRP(N),
            N > 2 and Length(divs)=2 and N mod 2 = 1 and
            IsPrime(divs[1][1]) and IsPrime(divs[1][2]) 
	    and not nt.hasTags()),

        children := nt -> let(
	    N:=nt.params[1], k:=nt.params[2], divs := DivisorPairsRP(nt.params[1]),
             r := divs[1][1], s := divs[1][2],
                 [ [DFT(r-1,-1), DFT(r-1,-1), DFT(s-1,-1), DFT(s-1,-1)],
		   [DFT(s-1,-1), DFT(s-1,-1), DFT(r-1,-1), DFT(r-1,-1)] ]),
	
        apply := (nt, C, cnt) -> let(
            r := Rows(C[1])+1,
            s := Rows(C[3])+1,
            er := ChineseRem([r,s], [1,0]),
            es := ChineseRem([r,s], [0,1]),
            dr := DFT_Rader.raderMid(r, nt.params[2]*s), # * er/s mod r),
            ds := DFT_Rader.raderMid(s, nt.params[2]*r), # * es/r mod s),

            CRT(r,s,1,1).transpose() *
            Tensor(RR(r).transpose(), RR(s).transpose()) *
            Tensor(
                DirectSum(I(1), C[1]) * dr * DirectSum(I(1), C[2]),
                DirectSum(I(1), C[3]) * ds * DirectSum(I(1), C[4])
            ) *
            Tensor(RR(r), RR(s)) *
            CRT(r,s,1,1)
        )
    ),

    #F DFT_SplitRadix: 1984
    #F
    #F DFT_n = B * (DFT_n/2 dirsum DFT_n/4 dirsum DFT_n/4) * perm
    #F
    #F B = (DFT_2 tensor I_n/2) * S * diag
    #F S = (I_n/2 dirsum (diag([1,E(4)] tensor I_n/4))
    #F
    #F Duhamel, Pierre:
    #F   Split Radix FFT Algorithm, Electronics Letters, Vol. 20, No. 1,
    #F   pp. 14--16, 1984
    #F
    DFT_SplitRadix := rec(
        info             := "DFT_n -> DFT_n/2, DFT_n/4, DFT_n/4",
        forTransposition := true,
        maxSize := 64,

        applicable := (self, nt) >> let(N := nt.params[1],
            N >= 8 and N mod 4 = 0 and N <= self.maxSize and not nt.hasTags()
        ),

        children := nt -> let(N := nt.params[1], w := nt.params[2],
            [[ DFT(N/2, w), DFT(N/4, w) ]]
        ),

        apply := (nt, C, cnt) -> let(N := nt.params[1], w := nt.params[2],
            Tensor(F(2), I(N/2)) *
            DirectSum(
                C[1],
                Tensor(Diag([1, E(4)^w]) * F(2), I(N/4)) *
                Diag(fPrecompute(fCompose(Tw3(N/2, 2, w), L(N/2, 2)))) *
                Tensor(I(2), C[2])*L(N/2,2)
            ) *
            L(N, 2)
        )
    ),

    #F DFT_DCT1andDST1: 1984
    #F
    #F   DFT_n = blocks * (DCT1_(n/2+1) dirsum DST1_(n/2-1)) * blocks
    #F
    #F   Wang:
    #F     Fast Algorithms for the Discrete W Transform and the
    #F     Discrete Fourier Transform.
    #F     IEEE Trans. on ASSP, 1984, pp. 803--814
    #F   Britanak/Rao:
    #F     The fast generalized discrete Fourier transforms: A unified
    #F     approach to the discrete sinusoidal transforms computation.
    #F     Signal Processing 79, 1999, pp. 135--150
    #F
    DFT_DCT1andDST1 := rec(
        info    := "DFT_n -> DCT1_(n/2+1), DST1_(n/2-1)",
        maxSize := 64,

        applicable := (self, nt) >> let(N:=nt.params[1], k:=nt.params[2],
            N > 2 and N <= self.maxSize and Is2Power(N) and AbsInt(k)=1) and not nt.hasTags(),

        children := nt -> 
	    [[ DCT1( nt.params[1]/2 + 1 ), DST1( nt.params[1]/2 - 1 ) ]],

        apply := (nt, C, cnt) -> let(N:=nt.params[1], w:=nt.params[2],
            DirectSum(I(1), blocks4(N - 1, w)) *
            DirectSum(C[1], C[2] ^ J(N/2 - 1)) *
            DirectSum(I(1), blocks1(N - 1))
        )
    )
));
