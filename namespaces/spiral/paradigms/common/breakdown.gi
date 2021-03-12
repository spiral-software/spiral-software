
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# NOTE: find a better place for these! What about a single standard drop tag rule?
NewRulesFor(TRC, rec(
    TRC_tag := rec(
        forTransposition := false,
        applicable := (self, nt) >>
            (nt.isTag(1, spiral.paradigms.smp.AParSMP) or not nt.hasTags())
            # AVecReg is taken from a namespace that is NOT YET LOADED
	    # hence the fully qualified name
            and not nt.hasTag(spiral.paradigms.vector.AVecReg)
            and not nt.hasTag(spiral.paradigms.vector.AVecRegCx),

        children := nt -> [[ nt.params[1].withTags(nt.getTags()) ]],
        apply := (nt, c, cnt) -> RC(c[1])
    )
));

NewRulesFor(TDiag, rec(
    TDiag_tag := rec(
        forTransposition := false,

	# YSV: Below limits applicability to the cases where diag size is divisible by vlen
	#      which is a safe thing to do. Because VectorCodegen, can't  generated code
	#      for VDiags of size non-divisible by vlen. HOWEVER, if VDiag is propagate
	#      past any kind of VGath, this problem goes away. So having no restriction,
	#      will work MOST of the time, but not all the time.
        #
	# applicable := (self, nt) >> let(
	#     vtags := [spiral.paradigms.vector.AVecReg, spiral.paradigms.vector.AVecRegCx],
	#     dom   := nt.params[1].domain(),
	#     not nt.hasAnyTag(vtags) or (dom mod nt.getAnyTag(vtags).v) = 0
	# ),

        apply := (t, C, Nonterms) -> let(
	    vtags := [spiral.paradigms.vector.AVecReg, spiral.paradigms.vector.AVecRegCx],
	    Cond(t.hasAnyTag(vtags),
		 spiral.paradigms.vector.sigmaspl.VDiag(t.params[1], t.getAnyTag(vtags).v),
		 Diag(t.params[1])
	    )
	)
    )
));

RulesFor(TRCDiag, rec(
    TRCDiag_tag := rec(
        forTransposition := false,
        applicable := (self, nt) >> not nt.transposed,
        rule := (P, C) -> RC(Diag(P[1])))
));

RulesFor(TId, rec(
    TId_tag := rec(
        forTransposition := false,
        switch := false,
        rule := (P, C) -> P[1])
));

NewRulesFor(TRaderMid, rec(
    TRaderMid_tag := rec(
        forTransposition := false,
        apply := (t, C, Nonterms) -> t.raderMid(t.params[1], t.params[2], t.params[3])
    )
));

NewRulesFor(TRDiag, rec(
    TRDiag_RT_Diag := rec(
        forTransposition := true,
        apply := (t, C, Nonterms) -> t.terminate()
    )
));

NewRulesFor(TCompose, rec(
    TCompose_tag := rec(
        forTransposition := false,
        applicable := (self, nt) >> true,
        children := nt -> [ List(nt.params[1], e -> e.withTags(nt.getTags())) ],
        apply := (nt, c, nt) -> Grp(Compose(c))
    )
));

NewRulesFor(TCond, rec(
    TCond_tag := rec(
        forTransposition := false,
        applicable := (self, nt) >> true,
        children := nt -> [[
	    nt.params[2].withTags(nt.getTags()), nt.params[3].withTags(nt.getTags()) ]],
        apply := (t, C, Nonterms) -> COND(t.params[1], C[1], C[2])
    )
));

NewRulesFor(TGrp, rec(
    TGrp_tag := rec(
        forTransposition := false,
        children := nt -> [[ nt.params[1].withTags(nt.getTags()) ]],
        apply := (nt, c, cnt) -> Grp(c[1])
    )
));

NewRulesFor(TInplace, rec(
    TInplace_tag := rec(
        forTransposition := false,
        children := nt -> [[ nt.params[1].withTags(nt.getTags()) ]],
        apply := (nt, c, cnt) -> Inplace(c[1])
    )
));

NewRulesFor(TICompose, rec(
    TICompose_tag := rec(
        forTransposition := false,
        children := nt -> [[ nt.params[3].withTags(nt.getTags()) ]],
        apply := (nt, c, cnt) -> ICompose(nt.params[1], nt.params[2], c[1])
    )
));



########################################################################
#   (A + B) rules
NewRulesFor(TDirectSum, rec(
#   (A + B) terminate
    A_dirsum_B := rec(
        forTransposition := false,
        children := (self, t) >> let( tags := t.getTags(),
            [[ t.params[1].withTags(tags), t.params[2].setTags(tags) ]]
        ),
        apply := (t, C, Nonterms) -> DirectSum(C)

#D        children := (self, t) >> let (tags:=GetTags(t),
#D            [[ AddTag(t.params[1], tags), SetTag(t.params[2], tags) ]]),
    )
));




########################################################################
#   (A x B) rules
NewRulesFor(TTensor, rec(
#   (A x B) -> (A x I)(I x B)
    AxI_IxB := rec(
        info := "(A x B) -> (A x I)(I x B)",
        forTransposition := false,
        applicable := nt -> true,
        inplace := false,
        children := (self, nt) >> let(inp := When(self.inplace, TInplace, x->x),
            [[ TCompose([
                inp(TTensorI(nt.params[1], nt.params[2].dims()[1], AVec, AVec)),
                TTensorI(nt.params[2], nt.params[1].dims()[2], APar, APar)
            ]).withTags(nt.getTags()) ]]),
        apply := (nt, c, cnt) -> c[1],
#D        isApplicable := P -> true,
#D        allChildren := P -> [[TCompose([TTensorI(P[1], P[2].dims()[1], AVec, AVec), TTensorI(P[2], P[1].dims()[2], APar, APar)], P[3])]],
#D        rule := (P, C) -> C[1]
    ),
#   (A x B) -> (I x B)(A x I)
    IxB_AxI := rec(
        info := "(A x B) -> (I x B)(A x I)",
        forTransposition := false,
        applicable := nt -> true,
        inplace := false,
        children := (self, nt) >> let(inp := When(self.inplace, TInplace, x->x),
            [[ TCompose([
                inp(TTensorI(nt.params[2], nt.params[1].dims()[1], APar, APar)),
                TTensorI(nt.params[1], nt.params[2].dims()[2], AVec, AVec)
            ]).withTags(nt.getTags()) ]]),
        apply := (nt, c, cnt) -> c[1]

#D        isApplicable := P -> true,
#D        allChildren := P -> [[TCompose([TTensorI(P[2], P[1].dims()[1], APar, APar), TTensorI(P[1], P[2].dims()[2], AVec, AVec)], P[3])]],
#D        rule := (P, C) -> C[1]
    ),
#   (A x B) -> (L(B x I))(L(A x I))
    L_BxI__L_AxI := rec(
        info := "(A x B) -> (L(B x I))(L(A x I))",
        forTransposition := false,
        applicable := nt -> true,
        children := nt -> [[ TCompose([
            TTensorI(nt.params[2], nt.params[1].dims()[1], APar, AVec),
            TTensorI(nt.params[1], nt.params[2].dims()[2], APar, AVec)
        ]).withTags(nt.getTags()) ]],
        apply := (nt, c, cnt) -> c[1]

#D        isApplicable := P -> true,
#D        allChildren := P -> [[TCompose([TTensorI(P[2], P[1].dims()[1], APar, AVec), TTensorI(P[1], P[2].dims()[2], APar, AVec)], P[3])]],
#D        rule := (P, C) -> C[1]
    ),
#   (A x B) -> ((A x I)L)((B x I)L)
    AxI_L__BxI_L := rec(
        info := "(A x B) -> ((A x I)L)((B x I)L)",
        forTransposition := false,
        applicable := nt -> true,
        children := nt -> [[ TCompose([
            TTensorI(nt.params[1], nt.params[2].dims()[1], AVec, APar),
            TTensorI(nt.params[2], nt.params[1].dims()[2], AVec, APar)
        ]).withTags(nt.getTags()) ]],
        apply := (nt, c, cnt) -> c[1],

#D        isApplicable := P -> true,
#D        allChildren := P -> [[TCompose([TTensorI(P[1], P[2].dims()[1], AVec, APar), TTensorI(P[2], P[1].dims()[2], AVec, APar)], P[3])]],
#D        rule := (P, C) -> C[1]
    ),
));

########################################################################
#   rules for A x I, I x A, (A x I)L, (I x A)L
NewRulesFor(TTensorI, rec(
    TTensorI_toGT := rec(
        applicable := t -> true,
        freedoms := t -> [], # no degrees of freedom
        child := (t, fr) -> [ GT_TTensorI(t) ], # fr will be an empty list
        apply := (t, C, Nonterms) -> C[1]
    )
));


NewRulesFor(TTensorI, rec(
#   base cases
#   I x A
    IxA_base := rec(
        info := "IxA base",
        forTransposition := false,
        applicable := nt -> (not nt.hasTags() or nt.firstTag() = ANoTag) and IsParPar(nt.params),
        children := nt -> [[ nt.params[1].withTags(nt.getTags()) ]],
        apply := (nt, c, cnt) -> When(nt.params[2] > 1,
            Tensor(I(nt.params[2]), c[1]),
            c[1]
        )
#D        isApplicable := (self, P) >> PUntagged(self.nonTerminal, P) and IsParPar(P),
#D        allChildren := P -> [[P[1]]],
#D        rule := (P, C) -> When(P[2]>1,Tensor(I(P[2]),C[1]),C[1])
    ),
#   A x I
    AxI_base := rec(
        info := "AxI base",
        forTransposition := false,
        applicable := nt -> (not nt.hasTags() or nt.firstTag() = ANoTag) and IsVecVec(nt.params),
        children := nt -> [[ nt.params[1].withTags(nt.getTags()) ]],
        apply := (nt, c, cnt) -> When( nt.params[2] > 1,
            Tensor(c[1], I(nt.params[2])),
            c[1]
        ),
#D        isApplicable := (self, P) >> PUntagged(self.nonTerminal, P) and IsVecVec(P),
#D        allChildren := P -> [[P[1]]],
#D        rule := (P, C) -> When(P[2]>1,Tensor(C[1], I(P[2])),C[1])
    ),
#   (I x A)L
    IxA_L_base := rec(
        info := "(IxA)L base",
        forTransposition := false,
        applicable := nt -> (not nt.hasTags() or nt.firstTag() = ANoTag) and IsParVec(nt.params),
        children := nt -> [[ nt.params[1].withTags(nt.getTags()) ]],
        apply := (nt, c, cnt) -> Tensor(I(nt.params[2]), c[1]) * L(c[1].dims()[2] * nt.params[2], nt.params[2]),

#D        isApplicable := (self, P) >> PUntagged(self.nonTerminal, P) and IsParVec(P),
#D        allChildren := P -> [[P[1]]],
#D        rule := (P, C) -> Tensor(I(P[2]), C[1])*L(C[1].dims()[2]*P[2], P[2])
    ),
#   L(I x A)
    L_IxA_base := rec(
        info := "L(IxA) base",
        forTransposition := false,
        applicable := nt -> (not nt.hasTags() or nt.firstTag() = ANoTag) and IsVecPar(nt.params),
        children := nt -> [[ nt.params[1].withTags(nt.getTags()) ]],
        apply := (nt, c, cnt) -> L(c[1].dims()[1] * nt.params[2], c[1].dims()[1]) * Tensor(I(nt.params[2]), c[1])

#D        isApplicable := (self, P) >> PUntagged(self.nonTerminal, P) and IsVecPar(P),
#D        allChildren := P -> [[P[1]]],
#D        rule := (P, C) -> L(C[1].dims()[1]*P[2], C[1].dims()[1]) * Tensor(I(P[2]), C[1])
    ),
#   splitting rules ##############################################################
#   (A _m x I_n)L_mn_m
    AxI_L_split := rec(
        info := "split (A_m x I_n) L^mn_m --> (L_mn/u_m x I_u) * (I_n/u x (A_m x I_u) * L_mu_m )",
        forTransposition := false,
        applicable := nt -> (nt.firstTag().kind() = AGenericTag) and IsVecPar(nt.params),
        children := nt -> let(t := nt.getTags(), p := nt.params, d  := p[1].dims(), mu := t[1].params[1], [
            TTensorI(TL(d[1] * p[2]/mu, d[1],1,1), mu, AVec, AVec).withTags(t),
            TTensorI(p[1], mu, AVec, APar).withTags(t)
        ]),
        apply := (nt, c, cnt) -> let(t := nt.getTags(), n := nt.params[2], mu := t[1].params[1],
            c[1] * Tensor(I(n/mu), c[2])
        ),

        # Example
        # =======
        # t:=TTensorI(DFT(4, 1), 4, AVec, APar).withTags([ AGenericTag(2) ]);
        # c:=AxI_L_split.children(t);
        # res := AxI_L_split.apply(t,c,false);

        switch:=false
    ),

#   (I_n x A_rxs) L^ns_n
    IxA_L_split := rec(
        info := "split (I_n x A_rxs) L^ns_n",
        forTransposition := false,
        applicable := nt -> IsParVec(nt.params),
        children := nt -> let(t := nt.getTags(), p := nt.params, d := p[1].dims(), [[
            TTensorI(p[1], p[2], APar, APar).withTags(t),
            TL(d[2]*p[2], p[2], 1, 1).withTags(t)
        ]]),
        apply := (nt, c, cnt) -> c[1] * c[2],

#D        isApplicable := P -> P[3].isPar and P[4].isVec,
#D        allChildren := P -> let(pv:=P[5], d:=P[1].dims(), [[TTensorI(P[1], P[2], APar, APar, pv), TL(d[2]*P[2], P[2], 1, 1, pv)]]),
#D        rule := (P, C) -> C[1] * C[2],
        switch := false
    ),
#   L^nr_n (A_rxs x I_n)
    L_AxI_split := rec(
        info := "split L^nr_n (A_rxs x I_n) ",
        forTransposition := false,
        applicable := nt -> IsParVec(nt.params),
        children := nt -> let( t := nt.getTags(), p := nt.params, d := p[1].dims(), [[
            TL(d[1] * p[2], p[2], 1, 1).withTags(t),
            TTensorI(p[1], p[2], AVec, AVec).withTags(t)
        ]]),
        apply := (nt, c, cnt) -> c[1] * c[2],
        switch := false
#D        isApplicable := P -> P[3].isPar and P[4].isVec,
#D        allChildren := P -> let(pv:=P[5], d:=P[1].dims(), [[ TL(d[1]*P[2], P[2], 1, 1, pv), TTensorI(P[1], P[2], AVec, AVec, pv) ]]),
#D        rule := (P, C) -> C[1] * C[2],
    ),
#   L^nr_r (I_n x A_rxs)
    L_IxA_split := rec(
        info := "split L^nr_r (I_n x A_rxs)",
        forTransposition := false,
        applicable := nt -> IsVecPar(nt.params),
        children := nt -> let( t := nt.getTags(), p := nt.params, d := p[1].dims(), [[
            TL(d[1]*p[2], d[1], 1, 1).withTags(t),
            TTensorI(p[1], p[2], APar, APar).withTags(t)
        ]]),
        apply := (nt, c, cnt) -> c[1] * c[2],

#D        isApplicable := P -> P[3].isVec and P[4].isPar,
#D        allChildren := P -> let(pv:=P[5], d:=P[1].dims(), [[TL(d[1]*P[2], d[1], 1, 1, pv), TTensorI(P[1], P[2], APar, APar, pv)]]),
#D        rule := (P, C) -> C[1] * C[2],
        switch := false
    ),
#   (A_rxs x I_n) L^nr_s
    AxI_L_split := rec(
        info := "split (A_rxs x I_n) L^nr_s ",
        forTransposition := false,
        applicable := nt -> IsVecPar(nt.params),
        children := nt -> let( t := nt.getTags(), p := nt.params, d := p[1].dims(), [[
            TTensorI(p[1], p[2], APar, APar).withTags(t),
            TL(d[2]*p[2], d[2], 1, 1).withTags(t)
        ]]),
        apply := (nt, c, cnt) -> c[1] * c[2],
#D        isApplicable := P -> P[3].isVec and P[4].isPar,
#D        allChildren := P -> let(pv:=P[5], d:=P[1].dims(), [[ TTensorI(P[1], P[2], APar, APar, pv), TL(d[2]*P[2], d[2], 1, 1, pv)]]),
#D        rule := (P, C) -> C[1] * C[2],
        switch := false
    ),
##   vector recursion #############################################################
#   (I x (I x A)L)L
    IxA_L_vecrec := rec(
        info := "(I x (I x A)L)L vector recursion",
        forTransposition := false,
        applicable := nt -> ObjId(nt.params[1]) = TTensorI and IsParVec(nt.params) and IsParVec(nt.params[1].params),
        children := nt -> let(k := nt.params[2], m := nt.params[1].params[2], n := nt.params[1].params[1].dims(), [[
            TL(k*m, k, 1, n[1]).withTags(nt.getTags()),
            TTensorI(nt.params[1].params[1], nt.params[2], APar, AVec).withTags(nt.getTags()),
            TL(m*n[2], m, 1, k).withTags(nt.getTags())
        ]]),
        apply := (nt, c, cnt) -> let(m := nt.params[1].params[2],
            c[1] * Tensor(I(m), c[2]) * c[3]
        ),
#D        isApplicable := P -> P[1].name = "TTensorI" and P[3].isPar and P[4].isVec and P[1].params[3].isPar and P[1].params[4].isVec,
#D        allChildren := P -> let(k:=P[2], m:=P[1].params[2], n:=P[1].params[1].dims(),
#D                [[ TL(k*m, k, 1, n[1], P[5]), TTensorI(P[1].params[1], P[2], APar, AVec, P[5]),  TL(m*n[2], m, 1, k, P[5])]]),
#D        rule := (P, C) -> let(k:=P[2], m:=P[1].params[2], n:=P[1].params[1].dims(),
#D                C[1] * Tensor(I(m), C[2]) * C[3]
#D            ),
        switch := false
    ),
#   L(I x L(I x A))
    L_IxA_vecrec := rec(
        info := "L(I x L(I x A)) vector recursion",
        forTransposition := false,
        applicable := nt -> ObjId(nt.params[1]) = TTensorI and IsVecPar(nt.params) and IsVecPar(nt.params[1].params),
        children := nt -> let( k := nt.params[2], m := nt.params[1].params[2], n := nt.params[1].params[1].dims(), [[
            TL(m*n[1], n[1], 1, k).withTags(nt.getTags()),
            TTensorI(nt.params[1].params[1], nt.params[2], AVec, APar).withTags(nt.getTags()),
            TL(k*m, m, 1, n[2]).withTags(nt.getTags())
        ]]),
        apply := (nt, c, cnt) -> let(m := nt.params[1].params[2],
            c[1] * Tensor(I(m), c[2]) * c[3]
        ),
#D        isApplicable := P -> P[1].name = "TTensorI" and P[3].isVec and P[4].isPar and P[1].params[3].isVec and P[1].params[4].isPar,
#D        allChildren := P -> let(k:=P[2], m:=P[1].params[2], n:=P[1].params[1].dims(),
#D                [[ TL(m*n[1], n[1], 1, k, P[5]), TTensorI(P[1].params[1], P[2], AVec, APar, P[5]),  TL(k*m, m, 1, n[2], P[5])]]),
#D        rule := (P, C) -> let(k:=P[2], m:=P[1].params[2], n:=P[1].params[1].dims(),
#D                C[1] * Tensor(I(m), C[2]) * C[3]
#D            ),
        switch := false
    )
));


########################################################################
#   rules for L

#D isVec := P->Length(P[5]) > 0 and P[5][1].isVec;

NewRulesFor(TL, rec(
#   TL(N,n,l,r,[]) -> I_l x L(N,n) x I_r
    L_base := rec(
        forTransposition := false,
        applicable := nt -> nt.isTag(1, spiral.paradigms.smp.AParSMP) or not nt.hasTags(),
        apply := (nt, c, cnt) -> let(
            c1 := When(nt.params[3]=1, [], [I(nt.params[3])]),
            c2 := When(nt.params[4]=1, [], [I(nt.params[4])]),
            Tensor(Concat(c1, [ L(nt.params[1], nt.params[2]) ], c2))
        )
    ),
#   TL(N,n,l,r,[]) -> I_l x L(N,n) x I_r
    L_func := rec(
        forTransposition := false,
        applicable := nt -> nt.isTag(1, spiral.paradigms.smp.AParSMP) or not nt.hasTags(),
        apply := (nt, c, cnt) -> let(
            c1 := When(nt.params[3]=1, [], [fId(nt.params[3])]),
            c2 := When(nt.params[4]=1, [], [fId(nt.params[4])]),
            Prm(fTensor(Concat(c1, [ L(nt.params[1], nt.params[2]) ], c2)))
        )
    ),
#   recursion rules
    IxLxI_kmn_n := rec (
        info             := "I(l) x L(kmn, n) x I(r) -> (I_l x L(kn,n) x I(mr))(I(kl) x L(mn, n) x I(r))",
        forTransposition := false,
        applicable := nt -> Length(DivisorsIntDrop(nt.params[1]/nt.params[2])) > 0,
        children := nt -> let(
            N := nt.params[1], n := nt.params[2],
            km := N/n, ml := DivisorsIntDrop(km),
            l := nt.params[3], r := nt.params[4],
            List(ml, m -> let( k := km/m, [
                TL(k*n, n, l, r*m).withTags(nt.getTags()),
                TL(m*n, n, k*l, r).withTags(nt.getTags())
            ]))
        ),
        apply := (nt, c, cnt) -> let(
            spl := c[1] * c[2],
            When(nt.params[1] = nt.params[2]^2,
                SymSPL(spl),
                spl
            )
        ),

#D        isApplicable     := P -> #isVec(P) and let(v:=P[5][1].v, (P[1]*P[2] >= v or P[1]*P[3] >= v) and
#D                                Length(DivisorsIntDrop(P[1]/P[2])) > 0,
#D        allChildren := P -> let(N:=P[1], n:=P[2], km:=N/n, ml:=DivisorsIntDrop(km), l:=P[3], r:=P[4], vp:=P[5],
#D            List(ml, m->let(k:=km/m, [TL(k*n, n, l, r*m, vp), TL(m*n,n, k*l, r, vp)])) ),
#D        rule := (P, C) -> let(spl := C[1]*C[2], When(P[1]=P[2]^2, SymSPL(spl), spl)),
        switch := false
    ),
    IxLxI_kmn_km := rec (
        info             := "I(l) x L(kmn, km) x I(r) -> (I(kl) x L(mn,m) x I(r))(I(l) x L(kn, k) x I(r))",
        forTransposition := false,
        applicable := nt -> Length(DivisorsIntDrop(nt.params[2])) > 0,
        children := nt -> let(
            N := nt.params[1], km := nt.params[2],
            n := N/km, ml := DivisorsIntDrop(km),
            l := nt.params[3], r := nt.params[4],
            List(ml, m->let(
                k := km/m,
                [
                    TL(m*n, m, k*l, r).withTags(nt.getTags()),
                    TL(k*n,k, l, m*r).withTags(nt.getTags())
                ]
            ))
        ),
        apply := (nt, C, cnt) -> let(P := nt.params, spl := C[1]*C[2], When(P[1]=P[2]^2, SymSPL(spl), spl)),
#D        isApplicable     := P -> #isVec(P) and let(v:=P[5][1].v, (P[1]*P[2] >= v or P[1]*P[3] >= v) and
#D                                Length(DivisorsIntDrop(P[2])) > 0,
#D        allChildren := P -> let(N:=P[1], km:=P[2], n:=N/km, ml:=DivisorsIntDrop(km), l:=P[3], r:=P[4], vp:=P[5],
#D            List(ml, m->let(k:=km/m, [TL(m*n, m, k*l, r, vp), TL(k*n,k, l, m*r, vp)])) ),
#D        rule := (P, C) -> let(spl := C[1]*C[2], When(P[1]=P[2]^2, SymSPL(spl), spl)),
        switch := false
    ),
    IxLxI_IxLxI_up := rec (
        info             := "I(l) x L(kmn, km) x I(r) -> (I(l) x L(kmn, k) x I(r))(I(l) x L(kmn, m) x I(r))",
        forTransposition := false,
        applicable       := nt -> Length(DivisorPairs(nt.params[2])) > 0,
        children := nt -> let(
            N := nt.params[1], km := DivisorPairs(nt.params[2]),
            l := nt.params[3], r := nt.params[4], t := nt.getTags(),
            List(km, i->[TL(N, i[1], l, r).withTags(t), TL(N, i[2], l, r).withTags(t)])
        ),
        apply := (nt, c, nt) -> c[1] * c[2],

#D        isApplicable     := P -> Length(DivisorPairs(P[2])) > 0,
#D        allChildren := P -> let(N:=P[1], km:=DivisorPairs(P[2]), l:=P[3], r:=P[4], vp:=P[5],
#D            List(km, i->[TL(N, i[1], l, r, vp), TL(N, i[2], l, r, vp)])),
#D        rule := (P, C) -> C[1]*C[2],
        switch := false
    ),
    IxLxI_IxLxI_down := rec (
        info             := "I(l) x L(kmn, k) x I(r) -> (I(l) x L(kmn, km) x I(r))(I(l) x L(kmn, kn) x I(r))",
        forTransposition := false,
        applicable       := nt -> Length(DivisorPairs(nt.params[1]/nt.params[2])) > 0,
        children         := nt -> let(
            N := nt.params[1], km := DivisorPairs(nt.params[1]/nt.params[2]),
            l := nt.params[3], r := nt.params[4], t := nt.getTags(),
            List(km, i->[TL(N, N/i[1], l, r).withTags(t), TL(N, N/i[2], l, r).withTags(t)])
        ),
        apply := (nt, c, cnt) -> c[1] * c[2],

#D        isApplicable     := P -> Length(DivisorPairs(P[1]/P[2])) > 0,
#D        allChildren := P -> let(N:=P[1], km:=DivisorPairs(P[1]/P[2]), l:=P[3], r:=P[4], vp:=P[5],
#D            List(km, i->[TL(N, N/i[1], l, r, vp), TL(N, N/i[2], l, r, vp)])),
#D        rule := (P, C) -> C[1]*C[2],
        switch := false
    ),
    IxLxI_loop1 := rec(
        info := "I x L x I loop1",
        forTransposition := false,
        applicable := nt -> not nt.hasTags(),
        apply := (nt, c, cnt) -> let(
            m := nt.params[2], n := nt.params[1]/nt.params[2], j:=Ind(m), fid := fId(n), fbase := fBase(m,j),
            gath := Gath(fTensor(fid, fbase)), scat := Scat(fTensor(fbase, fid)),
            c0 := [ISum(j, m, scat*gath)],
            c1 := When(nt.params[3]=1, [], [I(nt.params[3])]),
            c2 := When(nt.params[4]=1, [], [I(nt.params[4])]),
            Tensor(Concat(c1,c0,c2))
        ),

#D        isApplicable := P -> Length(P[5]) = 0,
#D        rule := (P, C) -> let(m:=P[2], n:=P[1]/P[2], j:=Ind(m), fid := fId(n), fbase := fBase(m,j),
#D                gath := Gath(fTensor(fid, fbase)), scat := Scat(fTensor(fbase, fid)),
#D                C0 := [ISum(j, m, scat*gath)], C1:=When(P[3]=1, [], [I(P[3])]), C2:=When(P[4]=1, [], [I(P[4])]), Tensor(Concat(C1, C0, C2))),
        switch := false
    ),
    IxLxI_loop2 := rec(
        info := "I x L x I loop2",
        forTransposition := false,
        applicable := nt -> not nt.hasTags(),
        apply := (nt, c, cnt) -> let(
            m := nt.params[2], n := nt.params[1]/nt.params[2], j:=Ind(m), fid := fId(n), fbase := fBase(m,j),
            gath := Gath(fTensor(fbase, fid)), scat := Scat(fTensor(fid, fbase)),
            c0 := [ISum(j, m, scat*gath)],
            c1 := When(nt.params[3]=1, [], [I(nt.params[3])]),
            c2 := When(nt.params[4]=1, [], [I(nt.params[4])]),
            Tensor(Concat(c1,c0,c2))
        ),

#D        isApplicable := P -> Length(P[5]) = 0,
#D        rule := (P, C) -> let(m:=P[2], n:=P[1]/P[2], j:=Ind(n), fid := fId(m), fbase := fBase(n,j),
#D                gath := Gath(fTensor(fbase, fid)), scat := Scat(fTensor(fid, fbase)),
#D                C0 := [ISum(j, n, scat*gath)], C1:=When(P[3]=1, [], [I(P[3])]), C2:=When(P[4]=1, [], [I(P[4])]), Tensor(Concat(C1, C0, C2))),
        switch := false
    )
));

###################################################################
NewRulesFor(TICompose, rec(
    TICompose_unroll := rec(
        forTransposition := false,
        applicable := nt -> true,
        children := nt -> [[
            TCompose(
                List([0..nt.params[2]-1], i -> RulesStrengthReduce(SubstBottomUp(Copy(nt.params[3]), nt.params[1], e -> V(i))))
            ).withTags(nt.getTags())
        ]],
        apply := (nt, c, cnt) -> c[1]
    )
));


NewRulesFor(TDR, rec(
    TDR_base := rec(
        forTransposition := false,
        applicable := nt -> true,
        apply := (nt, c, cnt) -> DR(nt.params[1], nt.params[2])
#D        isApplicable := True,
#D        rule := (P, C) -> DR(P[1], P[2])
    )
));

NewRulesFor(TGath, rec(
    TGath_base := rec(
        applicable := True,
        apply := (t, C, nt) -> t.terminate()
    )
));

NewRulesFor(TScat, rec(
    TScat_base := rec(
        applicable := True,
        apply := (t, C, nt) -> t.terminate()
    )
));


NewRulesFor(TConj, rec(
    TConj_tag := rec(
        applicable := True,
        children := t -> [[ t.params[1].withTags(t.getTags()) ]],
        apply := (t, C, nt) -> ConjLR(C[1], t.params[2], t.params[3])
    ),

    TConj_perm := rec(
        applicable := True,

	_cvtPerm := (t,p, use_tl) -> Cond(
	    ObjId(p) = fId,
	        I(p.params[1]),
	    ObjId(p) = L and use_tl,
	        TL(p.params[1], p.params[2], 1, 1).withTags(t.getTags()),
	    # else
		FormatPrm(p)
	),

	# one degree of freedom -- use TL (true) or use FormatPrm(L) (false)
	freedoms := (self, t) >> [[ true, false ]],

        child := (self, t, fr) >> [
	    self._cvtPerm(t, t.params[2], fr[1]),
	    t.params[1].withTags(t.getTags()),
	    self._cvtPerm(t, t.params[3], fr[1])
	],

	apply := (self, t, C, Nonterms) >> C[1]*C[2]*C[3]
    ),

    TConj_cplx := rec(
        applicable := t -> t.params[1] _is TRC,

	_cvtPerm := (t, p) -> Cond(
	    ObjId(p) = fId,
	        I(p.params[1]),
	    ObjId(p) = L,
	        TL(p.params[1], p.params[2], 1, 1).withTags(t.getTags()),
	    # else
		FormatPrm(p)
	),

	freedoms := (self, t) >> [],

        child := (self, t, fr) >> [
	    self._cvtPerm(t, t.params[2]),
	    t.params[1].withTags(List(t.getTags(), t->Cond(t.kind()=spiral.paradigms.vector.AVecReg, spiral.paradigms.vector.AVecRegCx(t.isa.cplx()), t))),
	    self._cvtPerm(t, t.params[3])
	],

	apply := (self, t, C, Nonterms) >> C[1]*C[2]*C[3]
    ),

));

#########################################################################

NewRulesFor(TTensorInd, rec(
#   base cases
#   I x A
    dsA_base := rec(
        info := "IxA base",
        forTransposition := false,
        applicable := nt -> not nt.hasTags() and IsParPar(nt.params),
        children := nt -> [[ nt.params[1], InfoNt(nt.params[2]) ]],
        apply := (nt, c, cnt) -> IDirSum(cnt[2].params[1], c[1])
    ),
#   A x I
    L_dsA_L_base := rec(
        info := "AxI base",
        forTransposition := false,
        applicable := nt -> not nt.hasTags() and IsVecVec(nt.params),
        children := nt -> [[ nt.params[1], InfoNt(nt.params[2]) ]],
        apply := (nt, c, cnt) ->
            L(c[1].dims()[1] * nt.params[2].range, c[1].dims()[1]) *
            IDirSum(cnt[2].params[1], c[1]) *
            L(c[1].dims()[2] * nt.params[2].range, nt.params[2].range)
    ),
#   (I x A)L
    dsA_L_base := rec(
        info := "(IxA)L base",
        forTransposition := false,
        applicable := nt -> not nt.hasTags() and IsParVec(nt.params),
        children := nt -> [[ nt.params[1], InfoNt(nt.params[2]) ]],
        apply := (nt, c, cnt) ->
            IDirSum(cnt[2].params[1], c[1]) *
            L(c[1].dims()[2] * nt.params[2].range, nt.params[2].range),
    ),
#   L(I x A)
    L_dsA_base := rec(
        info := "L(IxA) base",
        forTransposition := false,
        applicable := nt -> not nt.hasTags() and IsVecPar(nt.params),
        children := nt -> [[ nt.params[1], InfoNt(nt.params[2]) ]],
        apply := (nt, c, cnt) ->
            L(c[1].dims()[1] * nt.params[2].range, c[1].dims()[1]) *
            IDirSum(cnt[2].params[1], c[1])
    )
));
