
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Base change for DST-3
#    def := (mn, n) -> Let(m=>mn/n,
#                  Conjugate(DirectSum(Tensor(I(n-1), S(m)), I(m)),
#	                        L(mn, n) * IP(mn, DirectSum(J(n-1), I(1))))),
Class(B_DST3_U, Sym, rec(
    def := (mn, m) -> let(n := mn/m,
	DoNotReorder(SUMAcc(
		Mon(fId(mn), 
		    II(mn), 
		    II(mn)),
		Mon(fTensor(Z(m,1), fDirsum(J(n-1), fId(1))),
		    II(mn),
		    diagTensor(II(m, 1, m), II(n,n-1))))))));
#B_DST3_U := (mn,m) -> I(mn);

Class(B_DST4_U, Sym, rec(
    def := (mn, m) -> let(n := mn/m,
	DoNotReorder(SUMAcc(
		Mon(fId(mn), 
		    II(mn),
		    II(mn)),
		Mon(fTensor(Z(m,1), J(n)),
		    II(mn),
		    diagTensor(II(m, 1, m), II(n))))))));

Class(B_DCT4_U, Sym, rec(
    def := (mn, m) -> let(n := mn/m,
	DoNotReorder(SUMAcc(
		Mon(fId(mn), 
		    II(mn), 
		    II(mn)),
		Mon(fTensor(Z(m,1), J(n)),
		    fConst(mn, -1), 
		    diagTensor(II(m, 1, m), II(n))))))));

Class(B_DCT3_U, Sym, rec(
    def := (mn, m) -> let(n := mn/m,
	DoNotReorder(SUMAcc(
		Mon(fId(mn), II(mn),          diagTensor(II(m,0,1), II(n,0,1))),
		Mon(fId(mn), fConst(mn, 1/2), diagTensor(II(m,1,m), II(n,0,1))),
		Mon(fId(mn), II(mn),          diagTensor(II(m,0,m), II(n,1,n))),

		Mon(fTensor(Z(m,1), fDirsum(fId(1), J(n-1))),
		    II(mn),
		    diagTensor(II(m, 1, m), II(n,1,n))),

		Mon(fTensor(Z(m,2), fId(n)),
		    fConst(mn, -1/2), 
		    diagTensor(II(m, 2, m), II(n,0,1))))))));

# inverse-transpose T-basis for DCT3
Class(B_DCT3_T_IT, Sym, rec(
    def := (mn, m) -> let(n := mn/m,
	DoNotReorder(SUMAcc(
		Mon(fId(mn),
		    II(mn),
		    II(mn)),
		Mon(fTensor(Z(m,m-1), fDirsum(fId(1), J(n-1))),
		    II(mn),
		    diagTensor(II(m, 0, m-1), II(n,1,n))))))));


Class(B_DCT3_T_Radix2, Sym, rec(
    def := (mn, r) -> let(n := mn/2, 
	DoNotReorder(SUMAcc(
		Mon(fId(mn), II(mn),               diagTensor(II(2,0,1), II(n,0,n))),
		Mon(fId(mn), fConst(mn, cos(r)),   diagTensor(II(2,1,2), II(n,0,1))),
		Mon(fId(mn), fConst(mn, 2*cos(r)), diagTensor(II(2,1,2), II(n,1,n))),

		Mon(fTensor(Z(2,1), fDirsum(fId(1), J(n-1))),
		    fConst(mn, -1), 
		    diagTensor(II(2, 1, 2), II(n,1,n))))))));


