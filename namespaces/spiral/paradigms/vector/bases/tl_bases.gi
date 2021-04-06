
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_TL_applicable := (e, p1, p2, p3, p4) ->
    p1 = e.N and p2 = e.n and IsPosInt(p3 / e.l) and p4 = e.r;

SIMD_ISA_DB.buildRules := meth(self)
    local isa;
    for isa in self.active() do
        if self.verbose then Print("Building rules for ", isa, "...\n"); fi;
        isa.buildRules();
    od;
end;

SIMD_ISA.buildRules := meth(self)
    self.rules := rec();
    self.rules.binrules := let(binrules := BuildBinRules(self), Concat(
        binrules,
#        BuildBinRulesLxI(self, binrules, true, true),
        BuildBinRulesLxI(self, binrules, true, false),
        BuildBinRulesLxI(self, binrules, false, true)
    ));
    self.rules.unrules := BuildUnRules(self);
    self.rules.x_I_vby2 := Build_x_I_vby2(self);
    self.rules_built := true;
end;

SIMD_ISA.supportedTL := self >> List(Concat(self.rules.binrules, self.rules.unrules),
        e -> rec(l := e.perm.l, N := e.perm.N, n := e.perm.n, r := e.perm.r));

SIMD_ISA.getTL := (self, p) >> let(res := Filtered(Concat(self.rules.binrules, self.rules.unrules),
        e -> _TL_applicable(e.perm, p[1], p[2], p[3], p[4])), When(Length(res) >= 1, res[1], false));

L_x_I_vby2_code := function(instr, N, n, vby2, y, x)
    local ichain, l, v, l1, l2, op, i;
    v := 2*vby2;
    l := L(N, n).lambda();
    ichain := [];
    for i in [0..N/2-1] do
        l1 := EvalScalar(l.at(2*i).ev());
        l2 := EvalScalar(l.at(2*i+1).ev());
        op := instr[(l1 mod 2)+1][(l2 mod 2)+1];
        Add(ichain, assign(vref(y, v*i, v), op.instr(vref(x, v*QuoInt(l1, 2), v), vref(x, v*QuoInt(l2, 2), v), op.p)));
    od;

    return chain(ichain);
end;

NewRulesFor(TL, rec(
    SIMD_ISA_Bases1 := rec(
        forTransposition := false,
        applicable := (self, t) >> t.isTag(1, AVecReg) and
                                     let(isa := t.firstTag().isa, P:=t.params,
                                         isa.active and ForAny(isa.supportedTL(),
                         e -> _TL_applicable(e, P[1], P[2], P[3], P[4]))
                                     ),
        apply := function(nt,C,cnt)
                    local isa, tl, ll, vprm, P;
            P:=nt.params;
                    isa := nt.firstTag().isa;
                    tl := isa.getTL(P);
                    ll := P[3] / tl.perm.l;
                    vprm := tl.vperm;
                    return When(ll = 1, vprm, BlockVPerm(ll, isa.v, vprm, tl.perm.spl));
                end,
    ),

    SIMD_ISA_Bases2 := rec(
        forTransposition := false,
        applicable := (self, t) >> t.isTag(1, AVecReg) and
                                     let(isa := t.firstTag().isa,
                                         isa.active and t.params[4] = isa.v / 2 and
                                         Flat(isa.rules.x_I_vby2) <> []
                                     ) and not(t.params[1]=t.params[2] or t.params[2]=1),
         apply := function(nt,C,cnt)
                    local isa, vperm, P;
            P:=nt.params;
                    isa := nt.firstTag().isa;
                    vperm := VPerm(Tensor(L(P[1], P[2]), I(P[4])),
                                  (y, x) -> L_x_I_vby2_code(isa.rules.x_I_vby2, P[1], P[2], P[4], y, x),
                                   isa.v, P[1] * P[4] / isa.v);
                    return When(P[3] = 1, vperm, Tensor(I(P[3]), vperm));
                end
    )

));
