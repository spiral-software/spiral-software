
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


TrPRDFT1 := arg -> let(x:=ApplyFunc(PRDFT1, arg).transpose(), Chain(PrintLine(x, " ", x.dimensions), x));
TrPRDFT2 := arg -> ApplyFunc(PRDFT2, arg).transpose();
TrPRDFT3 := arg -> ApplyFunc(PRDFT3, arg).transpose();
TrPRDFT4 := arg -> ApplyFunc(PRDFT4, arg).transpose();

DFTSymmetries := rec(    #n-even, #n-odd
    DFT  := [
	[Real ,                 [CE0_R2Cpx(W_RFT, 1), [PRDFT1,   PRDFT1], 1]],
	[CE0_R2Cpx(W_URFTT, 1), [Real,                [TrPRDFT1, TrPRDFT1], 1]],
	[Even0, [Even0, [DCT1, DCT5], 1]],
	[Odd00, [Odd00, [DST1, DST5], E(4)]]  # *j
    ],

    DFT2 := [
	[Real ,                 [CO0_R2Cpx(W_RFT, -E(4)), [PRDFT2,   PRDFT2], 1]],
	[CE1_R2Cpx(W_URFTT, 1), [Real,                    [TrPRDFT3, TrPRDFT3], 1]],
	[Even1, [Odd0,   [DCT2, DCT6], 1]], 
	[Odd1 , [Even00, [DST2, DST6], E(4)]]
    ],

    DFT3 := [
	[Real ,                    [CE1_R2Cpx(W_RFT, 1), [PRDFT3,   PRDFT3], 1]],
	[CO1_R2Cpx(W_URFTT, E(4)), [Real,                [TrPRDFT2, TrPRDFT2], 1]],
	[Real , [CE1_R2Cpx(W_RFT, 1), [PRDFT3, PRDFT3], 1]],
	[Odd0 , [Even1, [DCT3, DCT7], 1]],
	[Even00,[Odd1,  [DST3, DST7], E(4)]]
    ],

    DFT4 := [
	[Real ,                    [CO1_R2Cpx(W_RFT, -E(4)), [PRDFT4,   PRDFT4], 1]],
	[CO1_R2Cpx(W_URFTT, E(4)), [Real,                    [TrPRDFT4, TrPRDFT4], 1]],
	[Odd1 , [Odd1,  [DCT4, DCT8], 1]],
	[Even1, [Even1, [DST4, DST8], E(4)]] 
    ]
);

_dftSym := function(nt, sym) 
    local pos, entry, dftsym;
    if not IsBound(DFTSymmetries.(nt.name)) 
	then return false; fi;

    dftsym := DFTSymmetries.(nt.name);
    pos := PositionProperty(dftsym, x->x[1]=sym);
    if pos = false then 
	return false; fi;

    entry := dftsym[pos][2];
    if IsEvenInt(Rows(nt)) then return [entry[1], entry[2][1], entry[3]];
    else                        return [entry[1], entry[2][2], entry[3]];
    fi;
end;


# let Q := JDFTp_Reconstruct(n), p = 1..4 (DFT type)
# Q is defined to be the matrix that satisfies
#    Q * DFTp(n) = DFTp(n) * J(n)
#    Q = J(n) ^ (DFTp(n)^-1)
#
#spiral> PrintMat( MatSPL(J(8))^(MatSPL(DFT(8))^-1) );

Class(JRecTwid1, DiagFunc, rec(
    def := n -> rec(size:=n),
    range := self >> TComplex,
    lambda := self >> let(n:=self.params[1], i:=Ind(n), 
	Lambda(i, omega(n, -i)))
));

Class(JRecTwid3, DiagFunc, rec(
    def := n -> rec(size:=n),
    range := self >> TComplex,
    lambda := self >> let(n:=self.params[1], i:=Ind(n), 
	Lambda(i, omega(2*n, (-2*i + n - 1))))
));

mk_InvDFT_Reconstruct := dft -> (let(n:=dft.params[1], Cond(
    ObjId(dft) = DFT1, DirectSum(I(1), J(n-1)),
    ObjId(dft) = DFT2, DirectSum(I(1), -J(n-1)),
    ObjId(dft) = DFT3, J(n),
    ObjId(dft) = DFT4, -J(n),
    Error("<dft> must be a non-terminal DFT1, DFT2, DFT3, or DFT4"))));

mk_JInvDFT_Reconstruct := dft -> (let(n:=dft.params[1], Cond(
    ObjId(dft) = DFT1, Diag(JRecTwid1(n)),
    ObjId(dft) = DFT2, I(n), 
    ObjId(dft) = DFT3, Diag(JRecTwid3(n)),
    ObjId(dft) = DFT4, -I(n), 
    Error("<dft> must be a non-terminal DFT1, DFT2, DFT3, or DFT4"))));

mk_JDFT_Reconstruct := dft -> (let(n:=dft.params[1], Cond(
    ObjId(dft) = DFT1, Diag(JRecTwid1(n)) * mk_InvDFT_Reconstruct(dft),
    ObjId(dft) = DFT2, mk_InvDFT_Reconstruct(dft),
    ObjId(dft) = DFT3, Diag(JRecTwid3(n)) * mk_InvDFT_Reconstruct(dft),
    ObjId(dft) = DFT4, -mk_InvDFT_Reconstruct(dft),
    Error("<dft> must be a non-terminal DFT1, DFT2, DFT3, or DFT4"))));

verify_JInvDFT_Reconstruct := dft -> let(T:=ObjId(dft), n:=dft.params[1], k:=dft.params[2], 
    MatSPL(dft * J(n)) - MatSPL(mk_JInvDFT_Reconstruct(dft) * T(n, -k))
);
verify_JDFT_Reconstruct := dft -> let(T:=ObjId(dft), n:=dft.params[1], k:=dft.params[2], 
    MatSPL(dft * J(n)) - MatSPL(mk_JDFT_Reconstruct(dft) * dft)
);



Class(RulesDFTSymmetry, RuleSet);
RewriteRules(RulesDFTSymmetry, rec(

     DFT1_upgrade := ARule(Compose, [@(1,DFT), [@(2,GathExtend), @, @(3).cond(e->e in [Even1, Odd1])]], 
	 e -> let(n := Cols(@(1).val), [ Diag(Twid(n,n,-1,1/2,0,0)), DFT2(n), @(2).val])),

     DFT3_upgrade := ARule(Compose, [@(1,DFT3), [@(2,GathExtend), @, @(3).cond(e->e in [Even1, Odd1])]], 
	 e -> let(n := Cols(@(1).val), [ Diag(Twid(n,n,-1,1/2,1/2,0)), DFT4(n), @(2).val])),

     DFT1_upgradeCE := ARule(Compose, [@(1,DFT), [@(2,GathExtend), @, @(3, [CE1_R2Cpx, CO1_R2Cpx])]], 
	 e -> let(n := Cols(@(1).val), [ Diag(Twid(n,n,-1,1/2,0,0)), DFT2(n), @(2).val])),

     DFT3_upgradeCE := ARule(Compose, [@(1,DFT3), [@(2,GathExtend), @, @(3, [CE1_R2Cpx, CO1_R2Cpx])]], 
	 e -> let(n := Cols(@(1).val), [ Diag(Twid(n,n,-1,1/2,1/2,0)), DFT4(n), @(2).val])),

     DFT_EO_Symmetry := ARule(Compose,
	 [ @(1,[DFT,DFT2,DFT3,DFT4]), 
	   [ @(2, GathExtend), @, @(3).cond(e->_dftSym(@(1).val, e)<>false)] ], 
	 e -> 
	     let(sym := _dftSym(@(1).val, @(3).val), 
		 out := sym[1],
		 transf := sym[2],
		 n := When(@(3).val.tsize = "cols", Cols(@(2).val), Rows(@(2).val)),
		 scale := sym[3],
		 [PushL(GathExtendU(Rows(@(1).val), out)), scale*transf(n)])),

     L_Real := ARule(Compose, [[L, @(1), @(2)], [@(3,GathExtend), @, Real]], 
	 e -> let(N:=@(1).val, k:=@(2).val, 
	     [ GathExtend(N, Real), L(N, k) ])),

     L_Conj0_R2Cpx := ARule(Compose, [[L, @(1), @(2)], [@(3,GathExtend), @, @(4, [CE0_R2Cpx, CO0_R2Cpx])]], 
	 e -> let(N:=@(1).val, k:=@(2).val, m:=N/k, nn:=Cols(@(3).val)/2, 
	          sym  := @(4).val, 
	          sym1 := When(ObjId(sym)=CE0_R2Cpx, CE1_R2Cpx(sym.w, sym.j), CO1_R2Cpx(sym.w, sym.j)), 
		  diag := When(ObjId(sym)=CE0_R2Cpx, Diag(BHD(m, 1, -1)), Diag(BHD(m, -1, 1))),
	     [ DirectSum(GathExtend(m, sym), GathExtend(N-m, sym1)),
	       DirectSum(I(Int(m+1+_even(m))), Tensor(I(Int((k-1)/2)), diag), I(_even(k)*(m+_odd(m)))),
	       RC(VStack(Gath(Refl(nn, N, Int((m+2)/2), L(N,k))), 
		         Gath(Refl(nn, N, nn-Int((m+2)/2), fCompose(L(N,k), fAdd(N, N-m, m)))))) ])), 

     L_Conj1_R2Cpx := ARule(Compose, [[L, @(1), @(2)], [@(3,GathExtend), @, @(4, [CE1_R2Cpx, CO1_R2Cpx])]], 
	 e -> let(N:=@(1).val, k:=@(2).val, m:=N/k, nn:=Cols(@(3).val)/2,
	          sym  := @(4).val, 
		  diag := When(sym=CE1_R2Cpx, Diag(BHD(m, 1, -1)), Diag(BHD(m, -1, 1))),
	     [ GathExtend(N, sym), 
	       DirectSum(Tensor(I(Int(k/2)), diag), When(IsOddInt(k), I(m+_odd(m)), [])),
	       RC(Gath(Refl(nn, N-1, nn, L(N,k)))) ])), 

     L_Even0 := ARule(Compose, [[L, @(1), @(2)], [@(3,GathExtend), @, Even0]], 
	 e -> let(N:=@(1).val, k:=@(2).val, m:=N/k, nn:=Cols(@(3).val),
	     [ DirectSum(GathExtend(m, Even0), GathExtend(N-m, Even1)),
	       VStack(Gath(Refl(nn, N, Int((m+2)/2), L(N,k))), 
		      Gath(Refl(nn, N, nn-Int((m+2)/2), fCompose(L(N,k), fAdd(N, N-m, m))))) ])), 

     L_Even00 := ARule(Compose, [[L, @(1), @(2)], [@(3,GathExtend), @, Even00]], 
	 e -> let(N:=@(1).val, k:=@(2).val, m:=N/k, nn:=Cols(@(3).val),
	     [ DirectSum(GathExtend(m, Even00), GathExtend(N-m, Even1)),
	       VStack(Gath(Refl(nn, N-2, Int(m/2), fCompose(fAdd(N,N,-1), L(N,k), fAdd(N, N-1, 1)))), 
		      Gath(Refl(nn, N-2, nn-Int(m/2), fCompose(fAdd(N,N,-1), L(N,k), fAdd(N, N-m, m))))) ])), 

     L_Odd0 := ARule(Compose, [[L, @(1), @(2)], [@(3,GathExtend), @, Odd0]], 
	 e -> let(N:=@(1).val, k:=@(2).val, m:=N/k, nn:=Cols(@(3).val),
	     [ DirectSum(GathExtend(m, Odd0),  GathExtend(m*(k-1), Odd1)),
	       DirectSum(I(Int((m+1)/2)), Tensor(I(Int((k-1)/2)), Diag(BHN(m))), When(IsEvenInt(k), I(Int(m/2)), [])),
	       VStack(Gath(Refl(nn, N, Int((m+1)/2), L(N,k))), 
		      Gath(Refl(nn, N, nn-Int((m+1)/2), fCompose(L(N,k), fAdd(N, N-m, m))))) ])), 

     L_Odd00 := ARule(Compose, [[L, @(1), @(2)], [@(3,GathExtend), @, Odd00]], 
	 e -> let(N:=@(1).val, k:=@(2).val, m:=N/k, nn:=Cols(@(3).val),
	     [ DirectSum(GathExtend(m, Odd00), GathExtend(m*(k-1), Odd1)),
	       DirectSum(I(Int((m-1)/2)), Tensor(I(Int((k-1)/2)), Diag(BHN(m))), When(IsEvenInt(k), I(Int(m/2)), [])),
	       VStack(Gath(Refl(nn, N-2, Int((m-1)/2), fCompose(fAdd(N,N,-1), L(N,k), fAdd(N, N-1, 1)))), 
		      Gath(Refl(nn, N-2, nn-Int((m-1)/2), fCompose(fAdd(N,N,-1), L(N,k), fAdd(N, N-m, m))))) ])), 

     L_Even1 := ARule(Compose, [[L, @(1), @(2)], [@(3,GathExtend), @, Even1]], 
	 e -> let(N:=@(1).val, k:=@(2).val, m:=N/k, nn:=Cols(@(3).val),
	     [ GathExtend(N, Even1), 
	       Gath(Refl(nn, N-1, nn, L(N,k))) ])), 

     L_Odd1 := ARule(Compose, [[L, @(1), @(2)], [@(3,GathExtend), @, Odd1]], 
	 e -> let(N:=@(1).val, k:=@(2).val, m:=N/k, nn:=Cols(@(3).val),
	     [ GathExtend(N, Odd1), 
	       DirectSum(Tensor(I(Int(k/2)), Diag(BHN(m))), When(IsOddInt(k), I(Int(m/2)), [])),
	       Gath(Refl(nn, N-1, nn, L(N,k))) ])), 

     CRT_Even1 := ARule(Compose, [@(1, CRT), [@(2,GathExtend), @, Even1]], 
	 e -> let(N:=Rows(@(2).val),  nn:=Cols(@(2).val),
	     [ GathExtend(N, Even1), 
	       Gath(Refl(nn, N-1, nn, @(1).val)) ])), 

     Tensor_Dirsum_split := ARule(Compose, 
	 [[Tensor, [I, @(1)], @(2)], [DirectSum, @(3).cond(e->Rows(e)=Cols(@(2).val)), @(4)]],
	 e -> [ DirectSum(@(2).val * @(3).val, Tensor(I(@(1).val-1), @(2).val) * @(4).val) ]),

     IxDFT_EO1 := ARule(Compose,
	 [[Tensor, [I, @(1)], [@(2,[DFT,DFT2,DFT3,DFT4]), @, 1, ...]], [@(3,GathExtend), @, @(4).cond(e->e in [Even1, Odd1])]],
	 e -> let(k:=@(1).val, n:=Cols(@(2).val), transf:=@(2).val, sign := When(@(4).val=Odd1, -1, 1),
	        When(k mod 2 = 0,
	         [ # Below is actually a scaled type 1 extension! exploit this for final rewriting phase
                   PushL(DirectSum(I(n*k/2), Tensor(I(k/2), mk_JDFT_Reconstruct(transf)*J(n))) * 
                         @(3).val) * 
		   Tensor(I(k/2), transf) ],

                   # again, a scaled type 1 extension!
		 [ PushL(VStack(I(n*(k+1)/2),
			        Tensor(J((k-1)/2), sign * mk_JDFT_Reconstruct(transf)) 
				   * Gath(fAdd(n*(k+1)/2, n*(k-1)/2, 0)))), 
		   DirectSum(Tensor(I((k-1)/2), 1/2*transf), 
			     transf * GathExtend(n, @(4).val))]))),

 #     IxDFT_EO1 := ARule(Compose,
# 	 [[Tensor, [I, @(1)], [@(2,[DFT,DFT2,DFT3,DFT4]), @, 1, ...]], [GathExtend, @(3), @(4).cond(e->e in [Even1, Odd1])]],
# 	 e -> let(k:=@(1).val, n:=Cols(@(2).val), transf:=@(2).val, sign := When(@(4).val=Odd1, -1, 1),
# 	        When(k mod 2 = 0,
# 	         [ # Below is actually a scaled type 1 extension! exploit this for final rewriting phase
#                    PushL(VStack(I(n*k/2), Tensor(J(k/2), sign * mk_JDFT_Reconstruct(transf)))), 
# 		   Tensor(I(k/2), 1/2*transf) ],

#                    # again, a scaled type 1 extension!
# 		 [ PushL(VStack(I(n*(k+1)/2),
# 			        Tensor(J((k-1)/2), sign * mk_JDFT_Reconstruct(transf)) 
# 				   * Gath(fAdd(n*(k+1)/2, n*(k-1)/2, 0)))), 
# 		   DirectSum(Tensor(I((k-1)/2), 1/2*transf), 
# 			     transf * GathExtend(n, @(4).val))]))),

     IxDFT_CEO1_RFTT_DHT := ARule(Compose,
	 [[Tensor, [I, @(1)], @(2,[DFT,DFT2,DFT3,DFT4])], 
	  [GathExtend, @(3), @(4,[CE1_R2Cpx, CO1_R2Cpx],e->e.w in [W_RFTT, W_URFTT, W_DHT, W_DHT])]], 

	 e -> let(k:=@(1).val, n:=Cols(@(2).val), 
	          transf := @(2).val, sign := When(ObjId(@(4).val)=CO1_R2Cpx, -1, 1),
	          newtransf := RC(ObjId(transf)(transf.params[1], -transf.params[2])), 
		  winv := @(4).val.w^-1,
		  pack1 := Mat([winv[1]]),
		  pack2 := Mat([winv[2]]),

	        When(k mod 2 = 0,
	         [ PushL(VStack(Tensor(I(n*k/2), pack1), 
			        Tensor(J(k/2), sign * Tensor(I(n), pack2)*
				               RC(mk_JInvDFT_Reconstruct(transf))))), 
		   Tensor(I(k/2), newtransf) ],

		 [ PushL(VStack(DirectSum(Tensor(I(n*(k-1)/2), pack1), I(n)),
			        Tensor(J((k-1)/2), sign * Tensor(I(n), pack2) *
				                   RC(mk_JInvDFT_Reconstruct(transf))) * Gath(fAdd(n*k, n*(k-1), 0)))), 
		   DirectSum(Tensor(I((k-1)/2), newtransf), 
			     transf * GathExtend(n, @(4).val))]))),

     IxDFT_Real := ARule(Compose,
	 [[Tensor, [I, @(1)], [@(2,[DFT,DFT2,DFT3,DFT4]), @, 1, ...]], [GathExtend, @(3), Real]],
	 e -> [ Tensor(I(@(1).val), @(2).val * GathExtend(Cols(@(2).val), Real)) ]),

     DFTxI_Ext := ARule(Compose,
	 [[@(1,Tensor), [@(2,[DFT,DFT2,DFT3,DFT4]), @, 1, ...], I], @(4, GathExtend)],
	 e -> [ @(1).val.parallelForm() * @(4).val]),

     Tensor_J1 := Rule([Tensor, [J, 1], @(1)], e -> @(1).val),     
     Tensor_I1 := Rule([Tensor, [I, 1], @(1)], e -> @(1).val),
     Tensor_I0 := Rule([Tensor, ..., [I, 0], ...], e -> I(0)),
     Compose_I := ARule(Compose, [I], e -> []),
     DirectSum_I0 := ARule(DirectSum, [[I, 0]], e -> []),
     DirectSum_Assoc := ARule(DirectSum, [@(1,DirectSum)], e -> [@(1).val.children()]),
     DirectSum_Single := Rule([DirectSum, @(1)], e -> @(1).val),
     Compose_Assoc := ARule(Compose, [@(1,Compose)], e -> [@(1).val.children()]),
     Compose_Single := Rule([Compose, @(1)], e -> @(1).val),
     J_J := ARule(Compose, [@(1,J), @(2,J)], e -> [ I(@(1).val.params[1]) ]),

     # ===============
     # Diags
     # ================
     Diag_Diag := ARule(Compose, [[Diag, @(1)], [Diag, @(2)]], e -> 
         [ Diag(diagMul(@(1).val, @(2).val)) ]),

     Diag_L_Diag := ARule(Compose, [[Diag, @(1)], @(2, L), [Diag, @(3)]], e -> 
         [ Diag(diagMul(@(1).val, fCompose(@(3).val, @(2).val))) * @(2).val ]),

     Tensor_Diag := Rule([Tensor, [I, @(1)], [Diag, @(2)]], 
         e -> Diag(diagTensor(fConst(@(1).val, 1), @(2).val))),
     DirectSum_Diag := Rule([DirectSum, [I, @(1)], [Diag, @(2)]], 
         e -> Diag(diagDirsum(fConst(@(1).val, 1), @(2).val))),
     
     E1_Diag_E1 := ARule(Compose, [[@(1,GathExtend,e->e.transposed), @, Odd1], 
                                   [Diag, @(2)], 
                                   [@(1,GathExtend,e->not e.transposed), @, Odd1]],
      e -> let(n:=@(2).val.domain(), f := @(2).val, 
           [Diag(diagMul(fConst(n/2, 1/4), 
                       diagAdd(fCompose(f, fTensor(fBase(2, 0), fId(n/2))),
                               fCompose(f, fTensor(fBase(2, 1), J(n/2))))))])),

     Scale_Scale := Rule([Scale, @(1), [Scale, @(2), @(3)]], e->Scale(@(1).val * @(2).val, @(3))),
     Scale_1 := Rule([Scale, @(1).cond(e->e=1), @(2)], e->@(2).val),

     # =============
     # PushL/PushR
     # =============
     PushL_within_PushL := Rule(@@(1,PushL, (e,cx)->IsBound(cx.PushL) and cx.PushL<>[]), e -> e.child(1)),

     Compose_PushL_PushL := ARule(Compose, [[PushL, @(1)], [PushL, @(2)]], e -> [ PushL(@(1).val * @(2).val) ]),
     Compose_PushR_PushL := ARule(Compose, [[PushR, @(1)], [PushL, @(2)]], e -> [ PushR(@(1).val * @(2).val) ]),
     XXX_PushL := ARule(Compose, [@(1, [Diag, L]), [PushL, @(2)]], e -> [PushL(@(1).val * @(2).val)]),
     
     Tensor_PushL := Rule([Tensor, @(1, I), [Compose, @(2, PushL), @(3)]], 
	 e -> PushL(Tensor(@(1).val, @(2).val.child(1))) * Tensor(@(1).val, @(3).val)),

     DirectSum_PushL := Rule(@(1, DirectSum, 
	     e -> ForAny(e.children(), 
		 c -> ObjId(c)=PushL or (ObjId(c)=Compose and ObjId(c.child(1)) = PushL))),

	 e -> let(ch := @(1).val.children(),
	          split := List(ch, e -> Cond(ObjId(e) = PushL, 
			                          [e, I(Cols(e))], 
			                      ObjId(e) = Compose and ObjId(e.child(1))=PushL, 
					          [e.child(1).child(1), Drop(e.children(), 1)], 
					      [I(Rows(e)), e])),
		  PushL(DirectSum(List(split, x->x[1]))) * DirectSum(List(split, x->x[2]))))
));

RewriteRules(RulesDFTSymmetry, rec(
     BRDFT3_ExtendOdd0 := ARule(Compose, [[BRDFT3, @(1).cond(e->IsEvenInt(e)), 1/4], [GathExtend, @, Odd0]], 
	 e -> let(k:=@(1).val,
	     [ GathExtend(k,Upsample0),SkewDTT(DCT3(k/2),1/2)])),

     Odd1_split := ARule(Compose, [[Tensor, [I,@(1)],@(2)],[GathExtend, @, Odd1]],
	 e -> let(n:=@(1).val,A:=@(2).val,k:=When(IsEvenInt(n),n/2, (n-1)/2),When(IsEvenInt(n),
	     [VStack(Tensor(I(k),A)*(1/2),Tensor(J(k),A*J(Cols(A))*(-1/2)))],
	     [VStack(DirectSum(Tensor(I(k),A)*(1/2),A*GathExtend(Cols(A),Odd1)),Tensor(J(k),A*J(Cols(A))*(-1/2)))]))),

     ReduceUpsample0_RC := ARule(Compose, [[ScatReduce, @(1), Upsample0],[RC, @(2)]], 
	 e -> [@(2).val*ScatReduce(@(1).val,Upsample0)]),

     Push_ScatReduceWithinSum :=  ARule(Compose, [[ScatReduce, @, Upsample0],[@(1,IterDirectSum).cond(e->IsEvenInt(e.domain)),@(2)]],
	 e -> [IterDirectSum(@(1).val.var,@(1).val.domain,ScatReduce(Rows(@(2).val),Upsample0)*@(2).val)]
),

     Cut_SkewPRDFT := ARule(Compose, [[ScatReduce, @, Upsample0],[BSkewPRDFT,@(1),@(2)]],
	 e ->  let(n:=@(1).val,a:=@(2).val,f:=2*cospi(2*a),
	        [SkewDTT(DCT3(n/2),2*a)*When(IsEvenInt(n/2),
		     HStack(I(n/2),DirectSum(Mat([[f/2]]),Tensor(I(((n/2)-2)/2),Mat([[f,-1],[-1,f]])),Mat([[f-1]]))^M((n)/2,n/4)),
		     HStack(I(n/2),DirectSum(Mat([[f/2]]),Conjugate(Tensor(I((n/2-1)/2),Mat([[f,-1],[-1,f]])),M(n/2-1,(n/2-1)/2)))))])),

     SplitIterDirectSum := Rule([@(1,IterDirectSum), [Compose,@(2,SkewDTT),@(3,HStack)]],
	 e->let(i:=@(1).val.var,IterDirectSum(@(1).val.var,@(1).val.domain,@(2).val)*IterDirectSum(@(1).val.var,@(1).val.domain,@(3).val))),

     BRDFT3_ExtendOdd1 := ARule(Compose, [[BRDFT3, @(1).cond(e->IsEvenInt(e)), 1/4], [GathExtend, @, Odd1]],
         e -> let(k:=@(1).val,i:=Ind(),
	        [IterDirectSum(i,k/2,Mat([[1/2+cospi((2*i+1)/k)],[-1/2]]))*PolyDTT(DCT4(k/2))])),
     
     BRDFT3_J := ARule(Compose, [[BRDFT3, @(1).cond(e->IsEvenInt(e)), 1/4], [J, @]],
         e -> let(k:=@(1).val,i:=Ind(),
                [IterDirectSum(i,k/2,Mat([[-2*cospi((2*i+1)/k),-1-2*cospi((2*i+1)/(k/2))],[1,-2*cospi((2*((k/2)-i)-1)/k)]]))*BRDFT3(k)]))
));

   
#z := L(15,3) * GathExtend(15, Odd0);; zz := RulesDFTSymmetry(z);; PrintMat(MatSPL(z)-MatSPL(zz));
#z := L(16,4) * GathExtend(16, Odd0);; zz := RulesDFTSymmetry(z);; PrintMat(MatSPL(z)-MatSPL(zz));

#z := L(15,3) * GathExtend(15, Even00);; zz := RulesDFTSymmetry(z);; PrintMat(MatSPL(z)-MatSPL(zz));
#z := L(16,4) * GathExtend(16, Odd1);; zz := RulesDFTSymmetry(z);; PrintMat(MatSPL(z)-MatSPL(zz));

#RulesDFTSymmetry(Tensor(I(3), DFT(5))*L(15,3)*GathExtend(15, Even0));
#RulesDFTSymmetry(Tensor(I(2), DFT3(8))*L(16,2)*GathExtend(16, Even0));


dft1 := (N, k) -> Tensor(DFT1(k), I(N/k)) * Diag(Tw1(N, N/k, 1)) * Tensor(I(k), DFT1(N/k)) * L(N, k);
dft2 := (N, k) -> Tensor(DFT2(k), I(N/k)) * Diag(Tw2(N, N/k, 1)) * Tensor(I(k), DFT1(N/k)) * L(N, k);
dft3 := (N, k) -> Tensor(DFT1(k), I(N/k)) * Diag(Tw3(N, N/k, 1)) * Tensor(I(k), DFT3(N/k)) * L(N, k);
dft4 := (N, k) -> Tensor(DFT2(k), I(N/k)) * Diag(Tw4(N, N/k, 1)) * Tensor(I(k), DFT3(N/k)) * L(N, k);

pdft1 := (N, k, lsym, rsym) -> GathExtend(N, lsym).transpose() * dft1(N, k) * GathExtend(N, rsym);
pdft2 := (N, k, lsym, rsym) -> GathExtend(N, lsym).transpose() * dft2(N, k) * GathExtend(N, rsym);
pdft3 := (N, k, lsym, rsym) -> GathExtend(N, lsym).transpose() * dft3(N, k) * GathExtend(N, rsym);
pdft4 := (N, k, lsym, rsym) -> GathExtend(N, lsym).transpose() * dft4(N, k) * GathExtend(N, rsym);

# DCT6(8)
f := L(15, 3) * Tensor(I(5), DFT2(3, 1)) * L(15, 5) * 
     Diag(Tw2(15, 5, 1)) * 
     Tensor(I(3), DFT(5, 1)) * L(15, 3);
ff := GathExtend(Rows(f), Odd0).transpose() * f * GathExtend(Cols(f), Even1);

# f := L(16,4) * Tensor(I(4), DFT(4)) * L(16,4) * Diag(Tw1(16,4,1)) * Tensor(I(4), DFT(4)) * L(16,4);
# ff := GathExtend(Rows(f), Odd00).transpose() * f * GathExtend(Cols(f), Odd00);

# DCT2/DCT6
# f := L(16,4) * Tensor(I(4), DFT2(4)) * L(16,4) * Diag(Tw2(16,4)) * Tensor(I(4), DFT(4)) * L(16,4);
#ff := GathExtend(Rows(f), Odd0).transpose() * f * GathExtend(Cols(f), Even1);

# f4 := L(16,4) * Tensor(I(4), DFT2(4)) * L(16,4) * Diag(Tw4(16,4)) * Tensor(I(4), DFT3(4)) * L(16,4);
# ff := GathExtend(Rows(f), Odd1).transpose() * f4 * GathExtend(Cols(f), Odd1);

#  fo := L(21,3) * Tensor(I(7), DFT2(3)) * L(21,7) * Diag(Tw4(21,7)) * Tensor(I(3), DFT3(7)) * L(21,3);
# ffo := GathExtend(Rows(fo), Odd1).transpose() * fo * GathExtend(Cols(fo), Odd1);

# fr1 := GathExtend(16, CE0_R2Cpx(W_RFT, 1).invertW()).transpose() * f * GathExtend(16, Real);

 fr2 := GathExtend(21, CE0_R2Cpx(W_RFT, 1).invertW()).transpose() * 
        L(21,3) * Tensor(I(7), DFT(3)) * L(21,7) * Diag(Tw1(21,7,1)) * Tensor(I(3), DFT(7)) * L(21,3) *
	GathExtend(21, Real);

 fr4 := GathExtend(21, CO1_R2Cpx(W_RFT, -E(4)).invertW()).transpose() * 
        L(21,3) * Tensor(I(7), DFT2(3)) * L(21,7) * Diag(Tw4(21,7,1)) * Tensor(I(3), DFT3(7)) * L(21,3) *
	GathExtend(21, Real);

# fr4 := GathExtend(16, CO1_R2Cpx).transpose() * L(16,4) * Tensor(I(4), DFT2(4)) * L(16,4) * Diag(Tw4(16,4)) * Tensor(I(4), DFT3(4)) * L(16,4) * GathExtend(16, Real);

#p  := Mat(1/2*[[1,  E(4)]]);
#pp := Mat(1/2*[[1, -E(4)]]);

#them := MatSPL(DFT(4) * Tensor(I(4), 2*pp));;
#me   := MatSPL(Tensor(I(4), 2*pp) * RC(DFT(4,1)));;

# y = DFT x* = (DFT* x)* ?
#
# C IxP = IxP' RC(C) 
# IxP' = C IxP RC(C)^-1

# C IxP =IxP' C Ix(1,j)

dftproj := function(f)
    f := RulesDFTSymmetry(f);
    f := f.transpose();
    f := RulesDFTSymmetry(f);
    f := f.transpose();
    return f;
end;
