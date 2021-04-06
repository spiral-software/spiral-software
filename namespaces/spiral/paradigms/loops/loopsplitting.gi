
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


########################################################################
#   rules for A x I, I x A, (A x I)L, (I x A)L
NewRulesFor(TTensorI, rec(
#   loop splitting for A x I
    AxI_LS_L := rec(
        info := "(A_rxs x I_n)(L|I) -> Sum(Sum(SAG))",
        forTransposition := false,
        applicable := nt -> let(P := nt.params,
            P[3] = AVec 
            and nt.isTag(1, AVecMemL)
            and IsInt(P[2]/nt.firstTag().v) 
            and P[2]/nt.firstTag().v>1
        ),
        children := nt -> [[
            When(nt.numTags() = 1, 
                nt.params[1], 
                nt.params[1].setTags(nt.params[1].withoutFirstTag())
            )
        ]],
        apply := (nt, C, cnt) -> let(
            v := nt.firstTag().v,
            i := Ind(v), 
            m_by_v := nt.params[2]/v, 
            j := Ind(m_by_v), 
            d := C[1].dims(),
            fr := When(nt.params[4] = AVec, 
                fTensor(fId(d[2]), fBase(m_by_v, j), fBase(v, i)), 
                fTensor(fBase(m_by_v, j), fBase(v, i), fId(d[2]))
            ),

            ISumLS(j, m_by_v, 
                Buf(Scat(fTensor(
                    fId(d[1]), fBase(m_by_v, j), fId(v)
                ))) 
                * ISum(i, v, Scat(fTensor(
                    fId(d[1]), fBase(v, i))) * C[1] * Gath(fr)
                )
            )
        )

#D        isApplicable := P -> P[3].isVec and Length(P[5]) > 0 and P[5][1].isMemL and IsInt(P[2]/P[5][1].v) and P[2]/P[5][1].v>1,
#D        allChildren := P -> let(pv:=P[5], v:=pv[1].v, d:=P[1].dims(), [[When(Length(pv)=1, P[1], P[1].setpv(Drop(pv, 1)))]]),
#D        rule := (P, C) -> let(
#D                v:=P[5][1].v, i:=Ind(v), m_by_v := P[2]/v, j:=Ind(m_by_v), d:=C[1].dims(),
#D                fr:=When(P[4].isVec, fTensor(fId(d[2]), fBase(m_by_v, j), fBase(v, i)), fTensor(fBase(m_by_v, j), fBase(v, i), fId(d[2]))),
#D                ISumLS(j, m_by_v, Buf(Scat(fTensor(fId(d[1]), fBase(m_by_v, j), fId(v)))) *
#D                    ISum(i, v, Scat(fTensor(fId(d[1]), fBase(v, i))) * C[1] * Gath(fr)))
#D            )
    ),
    AxI_LS_R := rec(
        info := "(I|L)(A_rxs x I_n) -> Sum(Sum(SAG))",
        forTransposition := false,
        applicable := nt -> let(P := nt.params,
            P[4] = AVec 
            and nt.isTag(1, AVecMemR)
            and IsInt(P[2]/nt.firstTag().v) 
            and P[2]/nt.firstTag().v > 1
        ),
        children := nt -> [[
            When(nt.numTags() = 1, 
                nt.params[1], 
                nt.params[1].setTags(nt.params[1].withoutFirstTag())
            )
        ]],
        apply := (nt, C, cnt) -> let(
            v := nt.firstTag().v, 
            i := Ind(v), 
            m_by_v := nt.params[2]/v, 
            j := Ind(m_by_v), 
            d := C[1].dims(),
            fw := When(nt.params[3] = AVec, 
                fTensor(fId(d[1]), fBase(m_by_v, j), fBase(v, i)), 
                fTensor(fBase(m_by_v, j), fBase(v, i), fId(d[1]))
            ),

            ISumLS(j, m_by_v,
                ISum(i, v, Scat(fw) * C[1] * Gath(fTensor(fId(d[2]), fBase(v, i)))) 
                * Buf(Gath(fTensor(
                    fId(d[2]), fBase(m_by_v, j), fId(v)
                )))
            )
        )

#D        isApplicable := P -> P[4].isVec and Length(P[5]) > 0 and P[5][1].isMemR and IsInt(P[2]/P[5][1].v) and P[2]/P[5][1].v>1,
#D        allChildren := P -> let(pv:=P[5], v:=pv[1].v, d:=P[1].dims(), [[When(Length(pv)=1, P[1], P[1].setpv(Drop(pv, 1)))]]),
#D        rule := (P, C) -> let(
#D                v:=P[5][1].v, i:=Ind(v), m_by_v := P[2]/v, j:=Ind(m_by_v), d:=C[1].dims(),
#D                fw:=When(P[3].isVec, fTensor(fId(d[1]), fBase(m_by_v, j), fBase(v, i)), fTensor(fBase(m_by_v, j), fBase(v, i), fId(d[1]))),
#D                ISumLS(j, m_by_v,
#D                    ISum(i, v, Scat(fw) * C[1] * Gath(fTensor(fId(d[2]), fBase(v, i)))) *
#D                            Buf(Gath(fTensor(fId(d[2]), fBase(m_by_v, j), fId(v)))))
#D            )
    ),
    LS_drop := rec(
        info := "drop tag",
        forTransposition := false,
        applicable := nt -> let(P := nt.params,
            (nt.isTag(1, AVecMemL) or nt.isTag(1, AVecMemR))
            and (
                (not IsInt(P[2]/nt.firstTag().v))
                or P[2]/nt.firstTag().v = 1
            )
        ),
        children := nt -> let(P := nt.params,
            [[ TTensorI(P[1], P[2], P[3], P[4]) ]]
        ),
        apply := (nt, C, cnt) -> C[1],

#D        isApplicable := P -> Length(P[5]) > 0 and (P[5][1].isMemL or P[5][1].isMemR) and 
#D                            ((not IsInt(P[2]/P[5][1].v)) or P[2]/P[5][1].v=1),
#D        allChildren := P -> [[TTensorI(P[1], P[2], P[3], P[4], [])]],
#D        rule := (P, C) -> C[1],
    )
));

########################################################################
#   TCompose rules
NewRulesFor(TCompose, rec(
    TCompose_LS_L := rec(
        info := "TCompose loop splitting left",
        forTransposition := false,
        applicable := nt -> nt.isTag(1, AVecMemL),
        children := nt -> [ Concat(
            [ nt.params[1][1].setTags(nt.getTags()) ], 
            Drop(nt.params[1], 1)
        )],
        apply := (nt, C, cnt) -> Compose(C)
#D        isApplicable := P -> Length(P[2]) > 0 and P[2][1].isMemL,
#D        allChildren := P -> [Concat([P[1][1].setpv(P[2])], Drop(P[1], 1))],
#D        rule := (P, C) -> Compose(C)
    ),
    TCompose_LS_R := rec(
        info := "TCompose loop splitting right",
        forTransposition := false,
        applicable := nt -> nt.isTag(1, AVecMemR),
        children := nt -> [ Concat(
            DropLast(nt.params[1], 1), 
            [
                nt.params[1][Length(nt.params[1])].setTags(nt.getTags())
            ]
        )],
        apply := (nt, C, cnt) -> Compose(C)

#D        isApplicable := P -> Length(P[2]) > 0 and P[2][1].isMemR,
#D        allChildren := P -> [Concat(DropLast(P[1], 1), [P[1][Length(P[1])].setpv(P[2])])],
#D        rule := (P, C) -> Compose(C)
    )
));
