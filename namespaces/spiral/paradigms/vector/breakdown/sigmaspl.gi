
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#D TPrm.getTags := self >> GetTags(self);
#D TPrm.firstTag := self >> let(tags := GetTags(self), When(tags=[], ANoTag, tags[1]));
#D TPrm.withTags := (self, tags) >> AddTag(self, tags);

# Breakdowns for vectorized TPrm( IJ )
#
NewRulesFor(TPrm, rec(
    # IP(N, any perm of size 1) -> I(N)
    TPrm_IP_Base1 := rec(
        applicable := (self,t) >> t.hasTags() and let(v := t.firstTag().v, p := t.params[1],
            ObjId(p) = IP and Rows(p.params[2]) = 1),

        apply := (self, t, C, Nonterms) >> I(Rows(t))
    ),

    TPrm_IJ_Vec := rec(
        requiredFirstTag := AVecReg,
        applicable := (self,t) >> t.hasTags() and let(v := t.firstTag().v, p := t.params[1],
            ObjId(p) = IP and ObjId(p.params[2]) = J and
            (Rows(p.params[2]) mod v) = 0
        ),

        children := (self, t) >> let(tags := t.getTags(), v := tags[1].v,
            [[ TPrm(J(v)).withTags(tags) ]]),

        apply := (self, t, C, Nonterms) >> let(
            v := t.firstTag().v,
            N := Rows(t),
            m := Rows(t.params[1].params[2]),
            k := N/m,
            j := Ind(N/v),

# NOTE: Pulling in breaks for vector code due to some missing comutative rule to pull it all over PushLR() -> VContainer() for now
            VContainer(IDirSum(j,
                COND(fCompose(diagTensor(II(k/2), II(2,0,1), II(m/v)), fBase(j)),
                    VTensor(Prm(fId(1)), v), C[1])) *
                    VTensor(Prm(condIJ(N/v, m/v)), v), t.firstTag().isa)
        )
    ),
    TPrm_format := rec(
        applicable := (self,t) >> t.hasTags() and t.hasTag(AVecReg),
        apply := (self, t, C, Nonterms) >> FormatPrm(t.params[1])
    )
));


# Breakdowns for vectorized TPrm( J )
#
NewRulesFor(TPrm, rec(
    TPrm_J := rec(
        requiredFirstTag := AVecReg,

        applicable := (self,t) >> t.hasTags() and let(p := t.params[1], v := t.firstTag().v, s := Rows(p),
            ObjId(p) = J and IsInt(s/v) and s<>v),

        children := (self, t) >> let(tags := t.getTags(), v := t.firstTag().v,
            [[ TPrm(J(v)).withTags(tags) ]]),

        apply := (self, t, C, Nonterms) >> let(v:=t.firstTag().v, s:=t.params[1].params[1], s2:=s/v,
                                    VTensor(J(s2), v) * Tensor(I(s2), C[1]))
    ),

    TPrm_Jv := rec(
        requiredFirstTag := AVecReg,

        applicable := (self,t) >> t.hasTags() and let(p := t.params[1], v := t.firstTag().v,
            ObjId(p) = J and Rows(p) = v),

        apply := (self, t, C, Nonterms) >> let(
            v := t.firstTag().v, isa := t.firstTag().isa,
            VPerm(J(v), isa.reverse, v, 0))
    ),
));
