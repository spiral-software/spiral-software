
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#######################################################################################
# Pointers
#
vvref := (loc, idx, vlen) -> nth(tcast(TPtr(TVect(loc.t.t, vlen)), loc), idx);
vtref := (vtype, loc, idx) -> nth(tcast(TPtr(vtype), loc), idx);

#######################################################################################
#   Shuffle instructions and instruction sets

Class(assign_sv, assign, rec(
   __call__ := (self, loc, exp, p) >> WithBases(self,
       rec(operations := CmdOps,
       loc := toAssignTarget(loc),
       exp := toExpArg(exp),
       p := p)),
    print := (self,i,is) >> Chain(Print(Blanks(i), self.name, "("), self.loc.print(), Print(", "), self.exp.print(), Print(", "), Print(self.p), Print(");\n")),
    getNoScalar := self >> [self.loc.loc]
));

_vcprintcs := function ( lst )
    local  i;
    if Length(lst) = 0  then
        return;
    fi;
    lst[1].vcprint();
    for i  in [ 2 .. Length(lst) ]  do
        Print(", ", lst[i].vcprint());
    od;
end;


# vec_shr(<a>, <n>) 
#   shift vector <a> elements "right" ("left" in spiral - toward first vector element)
#   by <n> items, shifting in zeros.

Class(vec_shr, AutoFoldExp, rec(
    ev := self >> Error("make ev method"),
    computeType := self >> self.args[1].t
));

# vec_shl(<a>, <n>) 
#   shift vector <a> elements "left" ("right" in spiral - toward last vector element)
#   by <n> items, shifting in zeros.

Class(vec_shl,  Exp, rec(
    ev := self >> Error("make ev method"),
    computeType := self >> self.args[1].t
));

# alignr(<x0>, <x1>, <shift>)
#   takes two vectors <x0> and <x1>, concatenates their values, and pulls out a 
#   vector-length section from an offset given by <shift>.

Class(alignr,  Exp, rec(
    ev := self >> let(
        x := self.args[1].ev() :: self.args[2].ev(),
        s := self.args[3].ev(),
        x{[s..s+self.t.size-1]}
    ),
    computeType := self >> UnifyPair(self.args[1].t, self.args[2].t),
));


