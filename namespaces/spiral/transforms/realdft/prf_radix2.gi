
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#
# Using these "radix-2" rules will yield optimal 2nlog(n)-4n+6 
# arithmetic cost for PRDFT1 (Split-Radix), and an almost equivalent 
# cost of 2nlog(n)-4n+8 for PDHT1 (better than the literature).
#

# Each entry is J(2) times conjugated twiddle w* times PDHT3(2)=F(2), 
# twiddles are accessed at stride 2.
# (w^*) * F_2 = [[r,i],[-i,r]] * F_2 = [[r+i, r-i], [r-i], [-i-r]]
Class(H3_CasTwid, Sym, rec(
    def := (n, k) -> 
	DirectSum(List([0..n/2-1], j -> let(w := E(2*n)^(k*j),
		    J(2) * Mat([[Re(w)+Im(w),  Re(w)-Im(w)],
			        [Re(w)-Im(w), -Re(w)-Im(w)]]))))
));

# These twiddles are combined with PRDFT3(2) (just as for PDHT3 above)
# since PRDFT3(2,1)=I(2), and PRDFT3(2,-1)=Diag(1,-1) we have the 
# variable 'm' scaling the second column of rotations.
Class(R3_Twid, Sym, rec(
    def := (n, k) -> let(m := When(k mod 4 = 1, 1, -1), 
	DirectSum(List([0..n/2-1], j -> let(w := E(2*n)^(k*j),
		           Mat([[Re(w), -Im(w)*m],
				[Im(w),  Re(w)*m]])))))
));
# since IPRDFT2(2,1)=Diag(1,-1), and IPRDFT2(2,-1)=I(2) we have 'm' again,
# it scales second row
Class(IR3_Twid, Sym, rec(
    def := (n, k) -> let(m := When(k mod 4 = 1, -1, 1), 
	DirectSum(List([0..n/2-1], j -> let(w := E(2*n)^(k*j),
		           Mat(2*  [  [Re(w), -Im(w)],
				    m*[Im(w),  Re(w)]])))))
));

RulesFor(PDHT3, rec(
    PDHT3_CT_Radix2 := rec(
	isApplicable := P -> P[1] > 2 and IsEvenInt(P[1]),
	allChildren  := P -> [[ DFT1(P[1]/2, P[2]) ]],
	rule := (P,C) -> let(N := P[1], k := P[2], 
	        RC(LIJ(N/2)) *
		DirectSum(Tensor(I(N/4), J(2)), I(N/2)) * 
		RC(C[1]) * 
		H3_CasTwid(N,k) *
		L(N,N/2)
	))
));
RulesFor(PRDFT3, rec(
    PRDFT3_CT_Radix2 := rec(
	isApplicable := P -> P[1] > 2 and IsEvenInt(P[1]),
	allChildren  := P -> [[ DFT1(P[1]/2, P[2]) ]], # SkewDFT(n, E(4)), and then no twids are necessary
	rule := (P,C) -> let(N := P[1], k := P[2], 
	        RC(LIJ(N/2)) *
		Diag(BHD(N/2, 1, -1)) *
		RC(C[1]) * 
		R3_Twid(N,k) *
		L(N,N/2)
	))
));
RulesFor(IPRDFT2, rec(
    IPRDFT2_CT_Radix2 := rec(
	isApplicable := P -> P[1] > 2 and IsEvenInt(P[1]),
	allChildren  := P -> [[ DFT1(P[1]/2, P[2]) ]], # SkewDFT(n, E(4)), and then no twids are necessary
	rule := (P,C) -> let(N := P[1], k := P[2], 
		L(N,2) *
		IR3_Twid(N,k) *
		RC(C[1]) * 
		Diag(BHD(N/2, 1, -1)) *
	        RC(LIJ(N/2).transpose())
	))
));

RulesFor(PDHT1, rec(
    PDHT1_CT_Radix2 := rec(
	isApplicable := P -> P[1] > 2 and IsEvenInt(P[1]),
	allChildren  := P -> [[ PDHT1(P[1]/2, P[2]), PDHT3(P[1]/2, P[2]) ]],
	rule := (P,C) -> let(N := P[1], k := P[2], 
	        RC(OddStride(N/2+1, N/4+1)) * 
		DirectSum(C[1], C[2]) * 
		Tensor(F(2), I(N/2))
	))
));

RulesFor(PRDFT1, rec(
    PRDFT1_CT_Radix2 := rec(
	isApplicable := P -> P[1] > 2 and IsEvenInt(P[1]),
	allChildren  := P -> [[ PRDFT1(P[1]/2, P[2]), PRDFT3(P[1]/2, P[2]) ]],
	rule := (P,C) -> let(N := P[1], k := P[2], 
	        RC(L_or_OS(N/2+1, Int(N/4)+1)) * 
		DirectSum(C[1], C[2]) * 
		Tensor(F(2), I(N/2))
	))
));
RulesFor(IPRDFT1, rec(
    IPRDFT1_CT_Radix2 := rec(
	isApplicable := P -> P[1] > 2 and IsEvenInt(P[1]),
	allChildren  := P -> [[ IPRDFT1(P[1]/2, P[2]), IPRDFT2(P[1]/2, P[2]) ]],
	rule := (P,C) -> let(N := P[1], k := P[2], 
		Tensor(F(2), I(N/2)) *
		DirectSum(C[1], C[2]) * 
	        RC(OddStride(N/2+1, 2))
	))
));

RulesFor(PRDFT2, rec(
    PRDFT2_CT_Radix2 := rec(
	isApplicable := P -> P[1] > 2 and IsEvenInt(P[1]),
	allChildren  := P -> [[ PRDFT2(P[1]/2, P[2]), PRDFT4(P[1]/2, P[2]) ]],
	rule := (P,C) -> let(N := P[1], k := P[2], 
	        RC(OddStride(N/2+1, N/4+1)) * 
		DirectSum(C[1], C[2]) * 
		Tensor(I(N/2), F(2)) ^ L(N, N/2)
	))
));

# SwitchRules(PRDFT3, [1,3]);
# SwitchRules(PRDFT1, [1,7]);#H3_CasTwid.cost := n -> 4 + (n/2-1)*6;

#DHT1.cost := n -> n + DHT1.cost(n/2) + DHT3.cost(n/2);
#DHT3.cost := n -> H3_CasTwid.cost(n) + DFT.cost(n/2);

#DHT1.cost := n -> n + DHT1.cost(n/2) + H3_CasTwid.cost(n/2) + DFT.cost(n/4);
# DHT1(32) cost = 16*2 (F2) + size 64

# DFT_8 = 56
# DFT_16 = 168
# DFT_32 = 456
# DFT_64 = 1160
