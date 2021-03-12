
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


########################################################################
#   (A x B) rules
NewRulesFor(TTensor, rec(
#   (A x B) -> L (B x I)(L(A x I))
    splitL_BxI__L_AxI := rec(
        info := "(A x B) -> L (B x I)(L(A x I))",
        forTransposition := false,
        applicable := nt -> let(P := nt.params,
            Rows(P[1]) = Cols(P[1])
            and Rows(P[2]) = Cols(P[2])
        ),
        children := nt -> let(P := nt.params,
            [[TCompose([TTensorI(P[2], P[1].dims()[1], AVec, AVec), TTensorI(P[1], P[2].dims()[2], APar, AVec)]).withTags(nt.getTags())]]
        ),
        apply := (t, C, nt) -> let(
            p:=nt[1].params[1][1],
            mn := Rows(nt[1]),
            n := p.params[2],
            DelayedPrm(L(mn, n)) * C[1]
        )

#D        isApplicable := P -> Rows(P[1]) = Cols(P[1]) and Rows(P[2]) = Cols(P[2]),
#D        allChildren := P -> [[TCompose([TTensorI(P[2], P[1].dims()[1], AVec, AVec), TTensorI(P[1], P[2].dims()[2], APar, AVec)], P[3])]],
#D        rule := (P, C, nt) -> let(p:=nt[1].params[1][1], mn := Rows(nt[1]), n := p.params[2], DelayedPrm(L(mn, n)) * C[1])
    ),
#   (A x B) -> ((A x I)L)(B x I) L
    AxI_L__BxI_splitL := rec(
        info := "(A x B) -> ((A x I)L)(B x I) L",
        forTransposition := false,

        applicable := nt -> let(P := nt.params,
            Rows(P[1]) = Cols(P[1])
            and Rows(P[2]) = Cols(P[2])
        ),

        children := nt -> let(P := nt.params,
            [[TCompose([TTensorI(P[1], P[2].dims()[1], AVec, APar), TTensorI(P[2], P[1].dims()[2], AVec, AVec)]).withTags(nt.getTags())]]
        ),

        apply := (t, C, nt) -> let(
            p := nt[1].params[1][1],
            mn := Rows(nt[1]),
            n := Rows(p.params[1]),
            C[1] * DelayedPrm(L(mn, n))
        )
#D        isApplicable := P -> Rows(P[1]) = Cols(P[1]) and Rows(P[2]) = Cols(P[2]),
#D        allChildren := P -> [[TCompose([TTensorI(P[1], P[2].dims()[1], AVec, APar), TTensorI(P[2], P[1].dims()[2], AVec, AVec)], P[3])]],
#D        rule := (P, C, nt) -> let(p:=nt[1].params[1][1], mn := Rows(nt[1]), n := Rows(p.params[1]), C[1] * DelayedPrm(L(mn, n)))
    ),
));
