
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


prep_hex_perm_spu := function(p)
    local retval;
    retval :=  ConcatenationString(
                "0x", p{[  1 ..  8]}, ", ",
                "0x", p{[  9 .. 16]}, ", ",
                "0x", p{[ 17 .. 24]}, ", ",
                "0x", p{[ 25 .. 32]});
    return(retval);
end;


prep_perm_string_spu := function(p)
    local base_string, final_string, i_counter, zeroOutString, v;
    # This handles both 4x32f and 2x64f perm strings
    v := Length(p);
    base_string := [0..((16/v)-1)];
    zeroOutString := List([1..(16/v)], i->128);
    final_string := [];
    # A value of '128' in the permutation param list causes the spu_shuffle to zero out the corresponding bytes.
    for i_counter in (p-1) do # changed Reversed(p-1) to (p-1)
       if i_counter = 127 then
          Append(final_string, zeroOutString);
        else
           Append(final_string, (base_string + i_counter * (16/v)));
        fi;
    od;
    return final_string;
end;

# NOTE: The first section is Intel SSE/2/3 specific, and has to be ported
# over to SPU

unpacklo := (l1,l2,n,k) -> Flat(List([1..n/(2*k)], i-> [
        List([1..k], j->l1[(i-1)*k+j]),
        List([1..k], j->l2[(i-1)*k+j])
]));

unpackhi := (l1,l2,n,k) -> Flat(List([1..n/(2*k)], i -> [
        List([1..k], j->l1[n/2+(i-1)*k+j]),
        List([1..k], j->l2[n/2+(i-1)*k+j])
]));

sparams := (l,n) -> List([1..l], i->[1..n]);

shuffle := (in1, in2, p, n, k) -> Flat([
    List([1..n/(2*k)],     i->List([1..k], j->in1[(p[i]-1)*k+j])),
    List([n/(2*k)+1..n/k], i->List([1..k], j->in2[(p[i]-1)*k+j]))
]);

iperm4 := self >> Filtered(Cartesian(self.params()), i->i[1]<>i[2] and i[3] <> i[4]);


# Generic instructions
#
Class(vop_new_mixin, rec(
    params := self >> [],
    permparams := self >> Cartesian(self.params()),
    isBinop := self >> self.numargs = 2,
    isUnop := self >> self.numargs = 1,
    isLoadop := False,
    isStoreop := False,
));

Class(vop_new, vop_new_mixin, Exp, rec(
    __call__ := arg >> let(self:=arg[1], 
    _computeExpType(WithBases(self, rec(
        p          := When(Length(arg) >= self.numargs+2, arg[self.numargs+2], []),
        args       := Concat(List(Sublist(arg, [2..self.numargs+1]), toExpArg),
                        When(Length(arg) >= self.numargs+2 and arg[self.numargs+2] <> [],
                        [vparam(arg[self.numargs+2])], [])),
        operations := ExpOps
    )))
    ),
));

Class(vbinop_new, vop_new, rec(numargs:=2));
Class(vunop_new, vop_new, rec(numargs:=1));

Class(vunbinop_new, vop_new, rec(
    numargs := 1,
    params := self >> self.binop.params(),
    v := self >> self.binop.v,
    semantic := (self, in1, p) >> self.binop.semantic(in1, in1, p)
));

Class(vloadop_new, vop_new, rec(
    numargs := 0,
    isLoadop := True,
    # in case of explicit type cast (YSV modification), we don't need getNoScalar,
    # and below returns [], compiler understands not to mess with typecasts
    getNoScalar := self >> When(IsBound(self.noscalar) and IsBound(self.args[self.noScalar].loc),
	self.args[self.noScalar].loc, [])
));

Class(vstoreop_new, assign, vop_new_mixin, rec(
    isStoreop := True,
    # in case of explicit type cast (YSV modification), we don't need getNoScalar,
    # and below returns [], compiler understands not to mess with typecasts
    getNoScalar := self >> let(s := self.loc,
    When(IsBound(s.loc), s.loc, []))
));

Class(vstoremsk, assign, vop_new_mixin, rec(
    isStoreop := True,
   __call__ := (self, loc, exp, p) >> WithBases(self,
       rec(operations := CmdOps,
       loc := toAssignTarget(loc),
       exp := toExpArg(exp),
       p := p)),

    print := (self,i,is) >> Print(Blanks(i), self.name, "(",
    self.loc.print(), ", ", self.exp.print(), ", ", self.p, ");\n"),

    # in case of explicit type cast (YSV modification), we don't need getNoScalar,
    # and below returns [], compiler understands not to mess with typecasts
    getNoScalar := self >> When(IsBound(self.noscalar) and IsBound(self.args[self.noScalar].loc),
	self.args[self.noScalar].loc, [])
));
