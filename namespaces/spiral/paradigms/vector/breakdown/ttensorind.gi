
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

NewRulesFor(TTensorInd, rec(
#   base cases
#   I x A
    dsA_base_vec_push := rec(
        info := "IxA base",
        forTransposition := false,
        applicable := nt -> IsParPar(nt.params) and nt.isTag(1, AVecReg),
        children := nt -> let(_krnl := nt.params[1].withTags(nt.getTags()), krnl := When(_krnl.isReal(), _krnl.setWrap(VWrapId), _krnl.setWrap(VWrapTRC(nt.firstTag().isa))),
            [[ krnl, InfoNt(nt.params[2]) ]]),
        apply := (nt, c, cnt) -> IDirSum(cnt[2].params[1], c[1])
    ),
#   A x Iv
    L_dsA_L_base_vec := rec(
        info := "AxI base",
        forTransposition := false,
        applicable := nt -> IsVecVec(nt.params) and nt.isTag(1, AVecReg) and nt.params[2].range = nt.getTags()[1].v,
        children := nt -> let(jv := Ind(nt.getTags()[1].v), 
            [[ SubstVars(Copy(nt.params[1]), rec((nt.params[2].id) := jv)).setWrap(VWrap(nt.firstTag().isa)).withTags(Drop(nt.getTags(), 1)), 
            InfoNt(jv) ]]),
        apply := (nt, c, cnt) -> let(v := nt.getTags()[1].v, jv := cnt[2].params[1], 
            VTensorInd(c[1], jv))
    ),
#   A x In
    L_dsA_L_vec := rec(
        info := "AxI base",
        forTransposition := false,
        applicable := nt -> IsVecVec(nt.params) and nt.isTag(1, AVecReg) and nt.params[2].range > nt.getTags()[1].v,
        children := nt -> let(v := nt.getTags()[1].v, jv := Ind(nt.getTags()[1].v), j := Ind(nt.params[2].range/nt.getTags()[1].v), 
            [[ TTensorInd(
                SubstVars(Copy(nt.params[1]), rec((nt.params[2].id) := j*V(v)+jv)), 
                j, AVec, AVec).setWrap(VWrap(nt.firstTag().isa)).withTags(Drop(nt.getTags(), 1)), InfoNt(jv) ]]),
        apply := (nt, c, cnt) -> let(jv := cnt[2].params[1],
            VTensorInd(c[1], jv))
    )
));
