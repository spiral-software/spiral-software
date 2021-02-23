
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


NewRulesFor(TRaderMid, rec(
    Pad_vec := rec(
        applicable := (self, t) >> t.isTag(1, AVecReg) or t.isTag(1, AVecRegCx),
        forTransposition := false,
        apply := (t, C, Nonterms) -> let(
            v := t.firstTag().v,
            ds := Rows(t)-1,
            When(IsInt(ds/v),
                DelayedDirectSum(VScat_sv(fId(1), v , 1), I(ds))
                * VecRaderMid(t.params[1], t.params[2], t.params[3], v)
                * DelayedDirectSum(VGath_sv(fId(1), v , 1), I(ds)),
                DelayedDirectSum(VScat_sv(fId(1), v , 1), VScat_sv(fId(ds), v, 1))
                * VecRaderMid(t.params[1], t.params[2], t.params[3], v)
                * DelayedDirectSum(VGath_sv(fId(1), v , 1), VGath_sv(fId(ds), v, 1))
            )
        )
    )
));
