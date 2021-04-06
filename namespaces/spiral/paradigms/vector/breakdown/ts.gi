
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


NewRulesFor(TS, rec(
    TS_vect := rec(
        forTransposition := false,
        applicable := t -> t.isTag(1, AVecReg) and IsInt(t.params[1] / t.firstTag().v),
        apply := (t, C, Nonterms) -> let(
            v := t.firstTag().v,
            N := t.params[1],
            VS(N, v)
        )
    )
));
