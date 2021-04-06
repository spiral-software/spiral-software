
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


NewRulesFor(GT, rec(
    GT_Par := rec(
        maxSize       := false,
        minSize       := false,
        parEntireLoop := true,    # parallelize entire loop
        splitLoop := true,        # split off <nthreads> iterations

        requiredFirstTag := AParSMP,

    applicable := (self, t) >> let(
        rank := Length(t.params[4]), nthreads := t.firstTag().params[1],
        rank = 1 and let(its := t.params[4][1],
        PatternMatch(t, [GT, @(1), @(2,XChain), @(3,XChain), ...], empty_cx()) and
        IsPosInt(its/nthreads))),

        children := (self, t) >> let(
            spl := t.params[1], g := t.params[2], s := t.params[3], its := t.params[4][1],
            nthreads := t.firstTag().params[1], tags := Drop(t.getTags(), 1),
            Concatenation(
                  When(self.splitLoop,
                      [[ GT(spl, g, s, [its / nthreads]).withTags(tags), InfoNt(nthreads) ]], []),
                  When(self.parEntireLoop,
                      [[ GT(spl, XChain([0]), XChain([0]), []).withTags(tags), InfoNt(its) ]], [])
            )
        ),

        apply := (self, t, C, Nonterms) >> let(
            spl := t.params[1], N := Minimum(spl.dimensions),
            g := t.params[2], s := t.params[3], its := t.params[4][1],
            gg := When(g.params[1]=[0,1], XChain([0,1,2]), XChain([1,2,0])),
            ss := When(s.params[1]=[0,1], XChain([0,1,2]), XChain([1,2,0])),
            nthreads := t.firstTag().params[1],
            tid      := t.firstTag().params[2],
            par_its := When(IsBound(Nonterms[2]), Nonterms[2].params[1], nthreads), i := Ind(par_its),

            SMPBarrier(nthreads, tid,
                SMPSum(nthreads, tid, i, par_its,
                    Scat(ss.part(1, i, Rows(spl), [par_its, its/par_its])) *
                    C[1] *
                    Gath(gg.part(1, i, Cols(spl), [par_its, its/par_its]))))
        )
    ),

    # Push parallel tag down to preserve locality: smp(I x A) -> (I x smp(A))
    # may be good for MD DFTs
    GT_Par_IxA_Push := rec(
        forTransposition := false,
        requiredFirstTag := AParSMP,
        applicable := t -> t.rank()=1 and t.params[2]=GTPar and t.params[3]=GTPar,
        children := t -> [[ t.params[1].withTags(t.getTags()) ]],
        apply := (t, C, Nonterms) -> Tensor(I(t.params[4][1]), C[1])
    ),

    GT_Par_odd := rec(
        maxSize       := false,
        minSize       := false,
        requiredFirstTag := AParSMP,

    applicable := (self, t) >> let(
        rank := Length(t.params[4]), nthreads := t.firstTag().params[1],
        rank = 1 and let(its := t.params[4][1], not IsInt(its/nthreads) and 
        PatternMatch(t, [GT, @(1), @(2,XChain), @(3,XChain), ...], empty_cx()))),

    children := (self, t) >> let(
            spl := t.params[1], g := t.params[2], s := t.params[3], its := t.params[4][1],
            nthreads := t.firstTag().params[1], tags := Drop(t.getTags(), 1),
                      [[ spl.withTags(tags) ]]),

    apply := (t, C, Nonterms) -> let(
            spl := Nonterms[1],
            its := t.params[4][1],
            i := Ind(its),
            g := fTensor(fBase(i), fId(Cols(spl))),
            s := fTensor(fBase(i), fId(Rows(spl))),
            nthreads := t.firstTag().params[1],
            tid      := t.firstTag().params[2],

            SMPSum(its, i, i, its,
                Scat(s) * C[1] * Gath(g)))
    )
));


NewRulesFor(TTensorInd, rec(
#   base cases
#   I x A
    dsA_base_smp := rec(
        forTransposition := false,
        applicable := nt -> IsParPar(nt.params) and nt.isTag(1, AParSMP) and nt.params[2].range = nt.getTags()[1].params[1],
        children := nt -> let(jp := IndPar(nt.params[2].range),
            [[ SubstVars(Copy(nt.params[1]), rec((nt.params[2].id) := jp)).withTags(Drop(nt.getTags(), 1)), InfoNt(jp) ]]),
        apply := (self, t, C, Nonterms) >> let(
            spl := C[1],
            nthreads := t.firstTag().params[1],
            tid      := t.firstTag().params[2],
            jp := Nonterms[2].params[1],
            g := fTensor(fBase(jp), fId(Cols(spl))),
            s := fTensor(fBase(jp), fId(Rows(spl))),

            SMPBarrier(nthreads, tid,
                SMPSum(nthreads, tid, jp, nthreads,
                    Scat(s) *
                    C[1] *
                    Gath(g))
        ))
    ),

    dsA_smp := rec(
        forTransposition := false,
        applicable := nt -> IsParPar(nt.params) and nt.isTag(1, AParSMP) and
            let(r := nt.params[2].range/nt.getTags()[1].params[1], IsInt(r) and r > 1),
        children := nt -> let(jp := IndPar(nt.getTags()[1].params[1]), j := Ind(nt.params[2].range/nt.getTags()[1].params[1]),
            [[ TTensorInd(
                SubstVars(Copy(nt.params[1]), rec((nt.params[2].id) := jp * V(j.range) + j)),
                j, APar, APar).withTags(Drop(nt.getTags(), 1)), InfoNt(jp) ]]),
        apply := (self, t, C, Nonterms) >> let(
            spl := C[1],
            nthreads := t.firstTag().params[1],
            tid      := t.firstTag().params[2],
            jp :=  Nonterms[2].params[1],
            g := fTensor(fBase(jp), fId(Cols(spl))),
            s := fTensor(fBase(jp), fId(Rows(spl))),

            SMPBarrier(nthreads, tid,
                SMPSum(nthreads, tid, jp, nthreads,
                    Scat(s) *
                    C[1] *
                    Gath(g))
            )
        )
    ),
    L_dsA_L_base_smp := rec(
        forTransposition := false,
        applicable := nt -> IsVecVec(nt.params) and nt.isTag(1, AParSMP) and nt.params[2].range = nt.getTags()[1].params[1],
        children := nt -> let(jp := IndPar(nt.getTags()[1].params[1]),
            [[ SubstVars(Copy(nt.params[1]), rec((nt.params[2].id) := jp)).withTags(Drop(nt.getTags(), 1)),
             InfoNt(jp) ]]),
        apply := (self, t, C, Nonterms) >> let(
            spl := C[1],
            nthreads := t.firstTag().params[1],
            tid      := t.firstTag().params[2],
            jp := Nonterms[2].params[1],
            g := fTensor(fId(Cols(spl)), fBase(jp)),
            s := fTensor(fId(Rows(spl)), fBase(jp)),

            SMPBarrier(nthreads, tid,
                SMPSum(nthreads, tid, jp, nthreads,
                    Scat(s) *
                    C[1] *
                    Gath(g))
        ))
    ),
    L_dsA_L_smp := rec(
        forTransposition := false,
        applicable := nt -> IsVecVec(nt.params) and nt.isTag(1, AParSMP) and
            let(r := nt.params[2].range/nt.getTags()[1].params[1], IsInt(r) and r > 1),
        children := nt -> let(jp := IndPar(nt.getTags()[1].params[1]), j := Ind(nt.params[2].range/nt.getTags()[1].params[1]),
            [[ TTensorInd(
                SubstVars(Copy(nt.params[1]), rec((nt.params[2].id) := jp * V(j.range) + j)),
                j, AVec, AVec).withTags(Drop(nt.getTags(), 1)), InfoNt(jp), InfoNt(j) ]]),

        apply := (self, t, C, Nonterms) >> let(
            spl := C[1],
            kernel := Nonterms[1].params[1],
            nthreads := t.firstTag().params[1],
            tid      := t.firstTag().params[2],
            jp := Nonterms[2].params[1],
            j := Nonterms[3].params[1],
            g := fCompose(fTensor(L(nthreads*Cols(kernel), nthreads), fId(j.range)), fTensor(fBase(jp), fId(Cols(spl)))),
            s := fCompose(fTensor(L(nthreads*Rows(kernel), nthreads), fId(j.range)), fTensor(fBase(jp), fId(Rows(spl)))),

            SMPBarrier(nthreads, tid,
                SMPSum(nthreads, tid, jp, nthreads,
                    Scat(s) *
                    C[1] *
                    Gath(g))
            )
        )
    )
));
