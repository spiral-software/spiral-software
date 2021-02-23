
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


########################################################
#   Machinery for being able to translate
#   vector-sigma-sums into vector code
########################################################

vref := (loc, idx, vlen) -> When(vlen=1, 
    nth(loc, idx), 
    nth(tcast(TPtr(TVect(loc.t.t, vlen)), loc), idx/vlen));

nth.toPtr := (self, t) >> let( 
    exp := When(IsPtrT(self.loc.t), self.loc, tcast(self.loc.t.toPtrType(), self.loc)) + self.idx,
    exp_t := exp.t, 
    new_t := When( IsPtrT(exp_t), TPtr(t, exp_t.qualifiers).withAlignment(exp_t), TPtr(t)),
    When(new_t=exp_t, exp, tcast(new_t, exp))
);

DeclareVars := function(code)
    local vars, vects,good;

    vars  := Set(List(Collect(code, [assign, @(1,var), @]),     e->e.loc));
    vects := Set(List(Collect(code, [assign, [vref,@(1,var),@], @]), e->e.loc.loc));
    good:=Set(Flat(List(Collect(code, decl), e->e.vars)));
    SubtractSet(vects, good);
    SubtractSet(vects, Set([X,Y]));
    SubtractSet(vars, good);

    if Length(vars) > 0 then code := decl(vars, code); fi;
    if Length(vects) > 0 then code := decl(vects, code); fi;
    return code;
end;