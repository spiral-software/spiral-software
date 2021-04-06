
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_SVCT_THRSHOLD := 2^13;

NewRulesFor(GT, rec(
    # Vectorize AxI: A x I_n -> (A x I_n/v) x I_v
    GT_Vec_AxI := rec(
        forTransposition := false,
        requiredFirstTag := [AVecReg, AVecRegCx],
        applicable := t -> t.rank()=1 and t.params[2]=GTVec and t.params[3]=GTVec and
                           IsInt(t.params[4][1] / t.firstTag().v),
        children := t -> let(
        r := t.params[4][1] / t.firstTag().v,
        isa := t.firstTag().isa,
        spl := t.params[1], #.setWrap(VWrap(isa)),
        tags := Drop(t.getTags(), 1),
        [[ GT(spl, GTVec, GTVec, [r]).withTags(tags).setWrap(VWrap(isa)) ]]),

        apply := (t, C, Nonterms) -> VTensor(C[1], t.firstTag().v),
    ),

    GT_Vec_IxA := rec(
        forTransposition := false,
        requiredFirstTag := [AVecReg, AVecRegCx],
        applicable := t -> t.rank()=1 and t.params[2]=GTPar and t.params[3]=GTPar and
                           IsInt(t.params[4][1] / t.firstTag().v),
        children := t -> let(
            tags := t.getTags(),
            v := t.firstTag().v,
#            r := t.params[4][1] / v,
            spl := t.params[1],

            [[  TL(Rows(spl)*v, v, 1, 1).withTags(tags).setWrap(VWrapId),
                spl.withTags(Drop(tags, 1)).setWrap(VWrap(tags[1].isa)),
                TL(Cols(spl)*v, Cols(spl), 1, 1).withTags(tags).setWrap(VWrapId)
            ]]),

        apply := (t, C, Nonterms) ->
            let(P := t.params, v := t.firstTag().v, r := P[4][1] / v,
                A := C[1] * VTensor(C[2], v) * C[3], When(r=1, A, Tensor(I(r), A)))
    ),

    # Push vector tag down to preserve locality: vec(I x A) -> (I x vec(A))
    GT_Vec_IxA_Push := rec(
        forTransposition := false,
        requiredFirstTag := [AVecReg, AVecRegCx],
        applicable := t -> t.rank()=1 and t.params[2]=GTPar and t.params[3]=GTPar,
        children := t -> [[ t.params[1].withTags(t.getTags()) ]],
        apply := (t, C, Nonterms) -> Tensor(I(t.params[4][1]), C[1])
    ),

    # Split off L:  vectorizes (I x A), (I x A) L, or L (I x A)
    #
    GT_Vec_SplitL := rec(
        minSize := _SVCT_THRSHOLD,
        forTransposition := false,
        requiredFirstTag := [AVecReg, AVecRegCx],

        applicable := (self, t) >> t.rank()=1 and Maximum(t.dims()) > self.minSize and
            (t.params{[2,3]} in [[GTPar, GTPar], [GTVec, GTPar], [GTPar, GTVec]]),

        children := t -> let(tags := t.getTags(), spl := t.params[1], n := t.params[4][1],
            g := t.params[2], s := t.params[3],
            [[ GT(spl, GTVec, GTVec, [n]).withTags(tags) ]]),

        apply := (t, C, Nonterms) -> let(spl := t.params[1], n := t.params[4][1], g := t.params[2], s := t.params[3],
                        c := Concatenation(
                            When(s=GTPar, [ L(Rows(spl)*n, n) ], []),
                            [ C[1] ],
                            When(g=GTPar, [ L(Cols(spl)*n, Cols(spl)) ], [])), Product(c))
    ),

    GT_Vec_IxA_L := rec(
        maxSize := _SVCT_THRSHOLD,
        forTransposition := false,
        requiredFirstTag := [AVecReg, AVecRegCx],

        applicable := (self, t) >> t.rank()=1 
            and t.params[2]=GTVec 
            and t.params[3]=GTPar 
            and Maximum(t.dims()) <= self.maxSize 
            and IsInt(t.params[4][1] / t.firstTag().v),

        children := t -> let(tags := t.getTags(), 
            spl := t.params[1], v:=tags[1].v,
            isa:=tags[1].isa, spl := t.params[1], d:=spl.dims(),

            [[ TL(d[1]*v, v).withTags(tags).setWrap(VWrapId),
               spl.withTags(Drop(tags,1)).setWrap(VWrap(isa)) ]] 
        ),

        apply := (t, C, Nonterms) -> let(spl := t.params[1], 
            d:=spl.dims(), n := t.params[4][1], tag := t.firstTag(), 
            v := tag.v, m := n / v,

            NoDiagPullinLeft(When(m=1, 
                C[1] * VTensor(C[2], v), 
                Tensor(I(m), C[1] * VTensor(C[2], v)) * VTensor(L(m*d[2], m), v)
            ))
        ),
    ),

    GT_Vec_L_IxA := rec(
        maxSize := _SVCT_THRSHOLD,
        forTransposition := false,
        requiredFirstTag := [AVecReg, AVecRegCx],

        applicable := (self, t) >> t.rank()=1 and t.params[2]=GTPar and t.params[3]=GTVec and Maximum(t.dims()) <= self.maxSize and
                           IsInt(t.params[4][1] / t.firstTag().v),

        children := t -> let(tags := t.getTags(), spl := t.params[1], v:=tags[1].v,
            isa:=tags[1].isa, spl := t.params[1], d:=spl.dims(),
            [[ spl.withTags(Drop(tags,1)).setWrap(VWrap(isa)),
               TL(d[2]*v, d[2]).withTags(tags).setWrap(VWrapId) ]] ),

        apply := (t, C, Nonterms) -> let(spl := t.params[1], d:=spl.dims(), n := t.params[4][1], tag := t.firstTag(), v := tag.v, m := n / v,
            NoDiagPullinRight(When(m=1, VTensor(C[1], v) * C[2], VTensor(L(m*d[2], d[2]), v) * Tensor(I(m), VTensor(C[1], v) * C[2]))))
    )
));
