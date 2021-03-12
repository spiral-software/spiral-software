
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

#F   tSPL Twiddle factors
Class(TTwiddle, Tagged_tSPL_Container, rec(
    abbrevs :=  [ 
        (mn, n) -> [mn, n, 1],
        (mn, n, k) -> [mn, n, k]
     ],
    dims := self >> let(n:=self.params[1], [n,n]),
    terminate := self >> Diag(Tw1(self.params[1], self.params[2], self.params[3])),
    transpose := self >> Copy(self),
    isReal := False,
    doNotMeasure := true,
    noCodelet := true,
    normalizedArithCost := self >> 6*Rows(self)
));

Class(TTwiddle_Stockham, Tagged_tSPL_Container, rec(
    abbrevs :=  [ 
        (n, blocksize, rdx, j) -> [n, blocksize, rdx, j]
     ],
    dims := self >> let(n:=self.params[1], [n,n]),
    #params := self >> let(params := [self.params[1], self.params[2], self.params[3], self.params[4]]),
    terminate := self >> Tw1(self.params[1], self.params[1]/self.params[3], 1),
#F 	D tensor I has to be done in the TTwiddle_St rule, not here
#F    terminate := self >> diagTensor(Tw1(self.params[1], self.params[2], self.params[3]), fConst(TComplex, self.params[2], 1)),
#F    terminate := self >> fCompose(dOmega(self.params[1],1), fId(self.params[1]/self.params[2])),
    transpose := self >> Copy(self),
    isReal := False,
    doNotMeasure := true,
    noCodelet := true,
    domain := self >> TComplex,
    normalizedArithCost := self >> 6*Rows(self)
));

NewRulesFor(TTwiddle_Stockham, rec(
    TTwiddle_St := rec(
        minSize := false,
        maxSize := false,
        applicable := (self, nt) >> ((IsBool(self.minSize) and not self.minSize) or nt.params[1] >= self.minSize) and 
            ((IsBool(self.maxSize) and not self.maxSize) or nt.params[1] <= self.maxSize),
        forTransposition := false,
        apply := function(t, C, Nonterms)
            local access, blocksize, i, i1, j, n, d, k, rdx, stride, t;

			n := t.params[1];
			blocksize := t.params[2];
			rdx := t.params[3];
			j := t.params[4];
			t := LogInt(n, rdx);
			i := Ind(n);
			i1 := Ind(n/(rdx^j));
			stride := rdx^j;
			access := Lambda(i1, i1 * stride);

#F	Older versions, not using an access function, but returning the full matrix
#F			TDiag(fPrecompute(diagTensor(fConst(TComplex, rdx^j, 1), Stockham_radix.gen(rdx, LogInt(N, rdx)-j-1))))
#F			omega := fCompose(FData(fCompose(dOmega(n,1),diagTensor(dLin(div(n,blocksize), 1, 0, TInt), dLin(blocksize, 1, 0, TInt))).tolist()), Lambda(i, i));
#F			omega := fCompose(fPrecompute(fCompose(dOmega(n,1),diagTensor(dLin(div(n,blocksize), 1, 0, TInt), dLin(blocksize, 1, 0, TInt))).tolist()), Lambda(i, i));

			omega := fCompose(fPrecompute(Tw1(n, n/rdx,1)), access);
			omega := diagTensor(omega, fConst(TComplex, rdx^j, 1));

        	return Diag(omega);
       	
		end
    ),
));

NewRulesFor(TTwiddle, rec(
    TTwiddle_Tw1 := rec(
        minSize := false,
        maxSize := false,
        applicable := (self, nt) >> ((IsBool(self.minSize) and not self.minSize) or nt.params[1] >= self.minSize) and 
            ((IsBool(self.maxSize) and not self.maxSize) or nt.params[1] <= self.maxSize),
        forTransposition := false,
        apply := (t, C, Nonterms) -> Diag(fPrecompute(Tw1(t.params[1], t.params[2], t.params[3])))
    ),

    TTwiddle_TwoTables := rec(
        minSize := false,
        maxSize := false,
        applicable := (self, nt) >> ((IsBool(self.minSize) and not self.minSize) or nt.params[1] >= self.minSize) and 
            ((IsBool(self.maxSize) and not self.maxSize) or nt.params[1] <= self.maxSize),
        forTransposition := false,
        apply := function(t, C, Nonterms)
            local mn, n, _k, rdx, dp, i, i1, i2, j, k, f, omega_1, omega_2;
            
            #t in this case is the TTwiddle(mn,n,k) function, so the t.params[1]  = first parameter of TTwiddle() = mn
            mn := t.params[1];
            n := t.params[2];
            _k := t.params[3];
            dp := DivisorPairs(mn);
            rdx := Maximum(dp[Int(Length(dp)/2+1)]);
            
            i := Ind(mn);
            i1 := Ind(mn);
            i2 := Ind(mn);
            j := Ind(mn/rdx);
            k := Ind(rdx);

            f := Lambda(i, idiv(i, n) * imod(i, n));	# -> guarantees contiguous access
            
#   NOTE: fPrecompute does not work, so I have to force table generation through FData()
#            omega_1 := fCompose(fPrecompute(dOmega(mn/rdx, _k)), fComputeOnline(Lambda(i1, idiv(f.at(i1), rdx))));
#            omega_2 := fCompose(fPrecompute(dOmega(mn, _k)), fComputeOnline(Lambda(i2, imod(f.at(i2), rdx))));
#            omega_1 := fCompose(fPrecompute(fCompose(dOmega(mn/rdx, _k), fId(mn/rdx))), fComputeOnline(Lambda(i1, idiv(f.at(i1), rdx))));
#            omega_2 := fCompose(fPrecompute(fCompose(dOmega(mn, _k), fId(rdx))), fComputeOnline(Lambda(i2, imod(f.at(i2), rdx))));

#   NOTE: the domain of dOmega cannot be set to an integer, so I need to compose with fId()...
            omega_1 := fCompose(FData(fCompose(dOmega(mn/rdx, _k), fId(mn/rdx)).tolist()), Lambda(i1, idiv(f.at(i1), rdx)));
            omega_2 := fCompose(FData(fCompose(dOmega(mn, _k), fId(rdx)).tolist()), Lambda(i2, imod(f.at(i2), rdx)));

            return Diag(omega_1) * Diag(omega_2);

# old version with FData without fCompose
#            omega_1 := FData(Lambda(j, omega(mn/rdx, j*imod(_k, mn/rdx))).tolist());
#            omega_2 := FData(Lambda(k, omega(mn, k*_k)).tolist());
#            return Diag(Lambda(i1, omega_1.at(idiv(f.at(i1), rdx)))) * Diag(Lambda(i2, omega_2.at(imod(f.at(i2), rdx))));
        end
    ),
));
