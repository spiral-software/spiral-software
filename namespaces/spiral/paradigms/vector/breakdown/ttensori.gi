
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


########################################################################
#   rules for A x I, I x A, (A x I)L, (I x A)L
Declare(AxI_vec);


NewRulesFor(TTensorI, rec(
#   vectorization cases

#   I_n x A_rxs -> I_n/v x L^rv_v(A_rxs x I_v)L^sv_s
    IxA_vec_push := rec(
        info := "IxA vec",
        forTransposition := false,
        applicable := nt -> IsParPar(nt.params) and (nt.isTag(1,AVecReg) or nt.isTag(1,AVecRegCx)),
        children := nt -> let(
            pv := nt.firstTag(),
            v := pv.v,
            d := nt.params[1].dims(),
            krnl := nt.params[1].withTags(nt.getTags()),
            [[ krnl ]]
	    # YSV: below is the original piece of code and comment. This is no longer works
	    #      and does not seem to be necessary. Above works quite well. Delete the below
	    #      as soon as everything is tested.
            #D  When(krnl.isReal(), krnl.setWrap(VWrapId), krnl.setWrap(VWrapTRC(pv.isa))) ]]
        ),  #D  thats bad but required to keep wrapping working :(
        apply := (nt, c, cnt) -> Tensor(I(nt.params[2]), c[1])

    ),

#   A x I_n -> (A x I_n/v) x I_v
    AxI_vec := rec(
        info := "A x I_n -> (A x I_n/v) x I_v",
        forTransposition := false,
        applicable := nt -> nt.hasTags() and IsVecVec(nt.params) and (nt.isTag(1,AVecReg) or nt.isTag(1,AVecRegCx)) and IsInt(nt.params[2]/nt.firstTag().v),
        children := nt -> let(
            r := nt.params[2] / nt.firstTag().v,
            isa := nt.firstTag().isa,
            [[ When(r = 1,
                When(nt.numTags() = 1,
                    nt.params[1].setWrap(VWrap(isa)),
                    nt.params[1].setWrap(Drop(nt.getTags(), 1)).setWrap(VWrap(isa))
                ),
                TTensorI(nt.params[1].setWrap(VWrap(isa)), r, AVec, AVec).withTags(Drop(nt.getTags(), 1))
            )]]
        ),
        apply := (nt, c, cnt) -> VTensor(c[1], nt.firstTag().v)
    ),
#   I_n x A_rxs -> I_n/v x L^rv_v(A_rxs x I_v)L^sv_s
    IxA_vec := rec(
        info := "IxA vec",
        forTransposition := false,
        applicable := nt -> IsParPar(nt.params) and nt.hasTags() and (nt.isTag(1,AVecReg) or nt.isTag(1,AVecRegCx)) and IsInt(nt.params[2]/nt.firstTag().v),
        children := nt -> let(
            pv := nt.getTags(),
            v := pv[1].v,
            isa := pv[1].isa,
            d := nt.params[1].dims(),
            [[
                TL(d[1]*v, v, 1, 1).withTags(pv).setWrap(VWrapId),
                When(Length(pv)=1, nt.params[1].setWrap(VWrap(isa)), nt.params[1].setpv(Drop(pv, 1)).setWrap(VWrap(isa))),
                TL(d[2]*v, d[2], 1, 1).withTags(pv).setWrap(VWrapId)
            ]]
        ),
        apply := (nt, c, cnt) -> let(
            l := nt.params[2] / nt.firstTag().v,
            A := c[1] * VTensor(c[2], nt.firstTag().v) * c[3],
            NoDiagPullin(When(l=1, A , Tensor(I(l), A)))
        )
    ),
#   (I_n x A_rxs)L^nr_n -> (I_n/v x L^rv_v(A_rxs x I_v))(L^ns/v_n/v x I_v)
    IxA_L_vec := rec(
        info := "(IxA)L vec",
        forTransposition := false,
        maxSize := -1,
        applicable := (self, nt) >>
            ObjId(nt.params[1]) <> TTensorI
            and IsParVec(nt.params)
            and nt.hasTags()
            and (nt.isTag(1,AVecReg) or nt.isTag(1,AVecRegCx))
            and IsInt(nt.params[2] / nt.firstTag().v)
            and (
                self.maxSize = -1
                or (
                    Rows(nt.params[1]) <= self.maxSize
                    and Cols(nt.params[1]) <= self.maxSize
                )
            ),
        children := nt -> let(
            pv := nt.getTags(),
            v := pv[1].v,
            isa := pv[1].isa,
            d := nt.params[1].dims(),
            [[
                TL(d[1]*v, v).withTags(pv).setWrap(VWrapId),
                When(Length(pv)=1,
                    nt.params[1].setWrap(VWrap(isa)),
                    nt.params[1].withTags(Drop(pv, 1)).setWrap(VWrap(isa))
                )
            ]]
        ),
        apply := (nt, c, cnt) -> let(
            v := nt.firstTag().v,
            m := nt.params[2] / v,
            d := nt.params[1].dims(),
            NoDiagPullinLeft(
                When(m = 1,
                    c[1] * VTensor(c[2], v),
                    Tensor(I(m), c[1] * VTensor(c[2], v)) * VTensor(L(m*d[2], m), v)
                )
            )
        ),
        switch := false

    ),
#   L^nr_r(I_n x A_rxs) -> (L^nr/v_r/v x I_v)(I_n/v x (A_rxs x I_v)L^sv_s)
    L_IxA_vec := rec(
        info := "L(IxA) vec",
        forTransposition := false,
        maxSize := -1,
        applicable := (self, nt) >>
            ObjId(nt.params[1]) <> TTensorI
            and IsVecPar(nt.params)
            and nt.hasTags()
            and (nt.isTag(1,AVecReg) or nt.isTag(1,AVecRegCx))
            and IsInt(nt.params[2] / nt.firstTag().v)
            and (
                self.maxSize = -1
                or (
                    Rows(nt.params[1]) <= self.maxSize
                    and Cols(nt.params[1]) <= self.maxSize
                )
            ),
        children := nt -> let(
            pv := nt.getTags(),
            v := pv[1].v,
            isa := pv[1].isa,
            d := nt.params[1].dims(),
            [[
                When(nt.numTags() = 1,
                    nt.params[1].setWrap(VWrap(isa)),
                    nt.params[1].withTags(Drop(pv, 1)).setWrap(VWrap(isa))
                ),
                TL(d[2]*v, d[2]).withTags(pv).setWrap(VWrapId)
            ]]
        ),
        apply := (nt, c, cnt) -> let(
            v := nt.firstTag().v,
            m := nt.params[2] / v,
            d := nt.params[1].dims(),
            NoDiagPullinRight(
                When(m = 1,
                    VTensor(c[1], v) * c[2],
                    VTensor(L(m*d[1], d[1]), v) * Tensor(I(m), VTensor(c[1], v) * c[2])
                )
            )
        ),

        switch := false
    ),
############################################################################
#   L splitting variant
#   (I_n x A_rxs) L^ns_n
    IxA_L_split_vec := rec(
        info := "split (I_n x A_rxs) L^ns_n",
        forTransposition := false,
        applicable := nt -> IsParVec(nt.params) and nt.hasTags() and (nt.isTag(1,AVecReg) or nt.isTag(1,AVecRegCx)),
        children := nt -> [[ TTensorI(nt.params[1], nt.params[2], AVec, AVec).withTags(nt.getTags()) ]],
        apply := (nt, c, cnt) -> let(d := nt.params[1].dims()[1],
            L(d * nt.params[2], nt.params[2]) * c[1]
        ),
        switch := false
    ),
#   L^nr_r (I_n x A_rxs)
    L_IxA_split_vec := rec(
        info := "split L^nr_r (I_n x A_rxs)",
        forTransposition := false,
        applicable := nt -> IsVecPar(nt.params) and (nt.isTag(1,AVecReg) or nt.isTag(1,AVecRegCx)),
        children := nt -> [[ TTensorI(nt.params[1], nt.params[2], AVec, AVec).withTags(nt.getTags()) ]],
        apply := (nt, c, cnt) -> let(d := nt.params[1].dims()[2],
            c[1] * L(d * nt.params[2], d)
        ),
        switch := false
    ),
    IxA_split_vec := rec(
        info := "split (I_n x A_rxs)",
        forTransposition := false,
        applicable := nt -> IsParPar(nt.params) and (nt.isTag(1,AVecReg) or nt.isTag(1,AVecRegCx)),
        children := nt -> [[ TTensorI(nt.params[1], nt.params[2], AVec, AVec).withTags(nt.getTags()) ]],
        apply := (nt, c, cnt) -> let(d := nt.params[1].dims(),
            L(d[1] * nt.params[2], nt.params[2]) * c[1] * L(d[2] * nt.params[2], d[2])
        ),
        switch := false
    ),
    IxA_conj_vec := rec(
        info := "split (I_n x A_rxs)",
        forTransposition := false,
        applicable := nt -> IsParPar(nt.params) and (nt.isTag(1,AVecReg) or nt.isTag(1,AVecRegCx)),
        children := nt -> [[ TTensorI(nt.params[1], nt.params[2], AVec, AVec).withTags(nt.getTags()) ]],
        apply := (nt, c, cnt) -> let(d := nt.params[1].dims(),
            ConjLR(c[1], L(d[1] * nt.params[2], nt.params[2]), L(d[2] * nt.params[2], nt.params[2]))
        ),
        switch := false
    ),

############################################################################
#   for now, these rules are for unrolled cod only
#   non-vref TTensorI
#   A x I_n, v\nmid n
    AxI_svec := rec(
        info := "A x I_n for v\nmid n",
        forTransposition := false,
        applicable := nt ->
            IsVecVec(nt.params)
            and (nt.isTag(1,AVecReg) or nt.isTag(1,AVecRegCx))
            #and let( d := nt.params[1].dims(), d[1] = d[2] )
            and not IsInt(nt.params[2] / nt.firstTag().v),

        children := nt -> let(
            v := nt.firstTag().v,
            r := v * QuoInt(nt.params[2] + v - 1, v),
            [[ TTensorI(nt.params[1], r, AVec, AVec).withTags(nt.getTags()) ]]
        ),
        apply := (nt, c, cnt) -> let(
            d := nt.params[1].dims(),
            v := nt.firstTag().v,
            IxVScat_pc(d[1], nt.params[2], nt.params[2], 0, v)
            * c[1]
            * IxVGath_pc(d[2], nt.params[2], nt.params[2], 0, v)
        ),
    ),
#   (I_m x A_n )L^mn_m, v\nmid m
    IxA_L_svec := rec(
        info := "(I_m x A_n )L^mn_m, v\nmid m",
        forTransposition := false,
        applicable := nt -> IsParVec(nt.params) and (nt.isTag(1,AVecReg) or nt.isTag(1,AVecRegCx))
            and let(
                v := nt.firstTag().v,
                d := nt.params[1].dims(),
                d[1] = d[2]
                and (
                    not IsInt(nt.params[2] / v)
                    or (not IsInt(d[1] / v))
                )
            ),
        children := nt -> let(
            v := nt.firstTag().v,
            d := nt.params[1].dims(),
            r := v * QuoInt(nt.params[2]+v-1, v),
            s := v * QuoInt(d[1]+v-1, v),
            [[
                TL(r*s, r, 1, 1).withTags(nt.getTags()),
                TTensorI(nt.params[1], r, AVec, AVec).withTags(nt.getTags())
            ]]
        ),
        apply := (nt, C, cnt) -> let(
            d:=nt.params[1].dims()[1],
            v := nt.firstTag().v,
            c1c := Cols(C[1])/v, c1r := Rows(C[1])/v,
            c2c := Cols(C[2])/v, c2r := Rows(C[2])/v,
            C0 := IxVScat_pc(nt.params[2],d,d,0,v),
            c0c := Cols(C0)/v,
            C0
#           * VGath(fAdd(c1r,c0c,0), v)
            * VGath_zero(c1r,c0c, v)
#           * VTensor(HStack(I(c0c), O(c0c,c1r-c0c)),v)
            * C[1]
            * VScat_zero(c1c,c2r,v)
#           * VTensor(VStack(I(c2r), O(c1c-c2r,c2r)),v)
            * C[2]
            * IxVGath_pc(d,nt.params[2],nt.params[2],0,v)
        )
    ),
    TTensorI_oddvloop := rec(
        switch  := false,
        maxSize := false,
        filter  := e->true,
        applicable := nt ->
            IsVecVec(nt.params)
            and (nt.isTag(1,AVecReg) or nt.isTag(1,AVecRegCx))
            and not IsInt(nt.params[2] / nt.firstTag().v),

        children := nt -> let(
            v := nt.firstTag().v,
           [[ TTensorI(nt.params[1], v, AVec, AVec).withTags(nt.getTags()) ]]
        ),

        apply := (t, C, Nonterms) -> let(
  	        v := t.firstTag().v,
            kernel := C[1],
            m := Rows(t.params[1]),
            n := Cols(t.params[1]),
            k := t.params[2],
            kd := _rounddown(k,v).v,
            kr := k-kd,
            i := Ind(kd/v),
            #---------------------
            SUM(
                ISum(i,
                    IxVScat_pc(n, k, v, v*i, v) *
                    kernel *
                    IxVGath_pc(m, k, v, v*i, v)
                ),
                IxVScat_pc(n, k, k-kd, kd, v) *
                kernel *
                IxVGath_pc(m, k, k-kd, kd, v)
            )
        )
    )
));
