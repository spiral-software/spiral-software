
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#Declare (CellCGenSIMD, CellCompileStrategyVector, CellVHackUnparser);

#   IBM AltiVec specific (av is for "AltiVec")

vconstpr_av := (self, v) >> Cond(v.t=TInt, v.idx_cprint(),
                                  v.t = TDouble, Print("(", self.vconst1, "((float)", v.cprint(), "))"),
                                  Print("(", self.vconstv, "{", PrintCS(v.v), "})")); #changed Reversed(v.v) to v.v

prep_perm_string := function(p)
    local base_string, final_string, i_counter;
    base_string := [0,1,2,3];
    final_string := [];
    for i_counter in (p-1) do # changed Reversed(p-1) to (p-1)
        Append(final_string, (base_string+i_counter*4));
    od;
    return final_string;
end;
avshuffle := (p) -> Print("((vector unsigned char){", PrintCS(prep_perm_string(p)), "})");
avprintop := (self) >> Chain(Print(self.altivec, "("), spiral.compiler._cprintcs(self.args), Print(")"));

spushuffle := (p) -> Print("((vector unsigned char){", PrintCS(prep_perm_string(p)), "})");
spuprintop := (self) >> Chain(Print(self.spu, "("), spiral.compiler._cprintcs(self.args), Print(")"));

vpermop := (in1, in2, p, v)-> List(p, i->When(i<=v, in1[i], in2[i-v]));

#aperm := meth(self)
#    local l1, l2, l3, res, el, v;
#    res := [];
#    v := self.v();
#    l1 := Cartesian(self.params());
#    for el in l1 do
#        l2 := Filtered(el, i->i<=v);
#        l3 := Filtered(el, i->i>v);
#        if Length(l2) = Length(l3) and Try([PermList(l2-Minimum(l2)+1)])[1] and Try(PermList(l3-Minimum(l3)+1))[1] then
#            Add(res, el);
#        fi;
#    od;
#    return res;
#end;

aperm := meth(self)
    local l1, l2, l3, res, el, v;
    res := [];
    v := self.v();
    res := Cartesian(self.params());
    return Filtered(res, i->Length(Set(Copy(i)))=v);
end;





Class(vparam_av, Value,
    rec(
        __call__ := arg >> WithBases(
            arg[1],
            rec(
                p := arg[2],
                args := [],
                operations := PrintOps
            )
        ),
        t := TUnknown,
        print := self >> Print(self.p),
        cprint := self >> avshuffle(self.p),
        isVparam := true
    )
); #Class added for AltiVec

Class(vparam_spu, Value,
    rec(
        __call__ := arg >> WithBases(
            arg[1],
            rec(
                p := arg[2],
                args := [],
                operations := PrintOps
            )
        ),
        t := TUnknown,
        print := self >> Print(self.p),
        cprint := self >> spushuffle(self.p),
        #from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), [rch[1], rch[2]]),
        isVparam := true
    )
); #Class added for SPU

Class(vop_av, Exp, rec(
        __call__ := arg >> _computeExpType(WithBases(arg[1], rec(
                    p := When(Length(arg) >= arg[1].numargs+2, arg[arg[1].numargs+2], []),
                    args  := Concat(List(Sublist(arg, [2..arg[1].numargs+1]), toExpArg),
                                When(Length(arg) >= arg[1].numargs+2 and arg[arg[1].numargs+2] <> [],
                                    [vparam_av(arg[arg[1].numargs+2])], [])),
                    operations := PrintOps
                )
            )),
        params := self >> [],
        permparams := self >> Cartesian(self.params()),
        cprint := avprintop,
        vcprint := self>>self.cprint(),
        isBinop := self >> self.numargs = 2,
        isUnop := self >> self.numargs = 1,
        isLoadop := False,
        isStoreop := False,
    )
); #Class added for AltiVec

Class(vop_spu, Exp, rec(
        __call__ := arg >> code._computeExpType(
            WithBases(arg[1], rec(
                    p := When(Length(arg) >= arg[1].numargs+2, arg[arg[1].numargs+2], []),
                    args  := Concat(List(Sublist(arg, [2..arg[1].numargs+1]), toExpArg),
                                When(Length(arg) >= arg[1].numargs+2 and arg[arg[1].numargs+2] <> [],
                                    [vparam_spu(arg[arg[1].numargs+2])], [])),
                    operations := PrintOps
                )
                )),
        params := self >> [],
        permparams := self >> Cartesian(self.params()),
        cprint := spuprintop,
        vcprint := self>>self.cprint(),
        isBinop := self >> self.numargs = 2,
        isUnop := self >> self.numargs = 1,
        isLoadop := False,
        isStoreop := False,
    )
); #Class added for SPU


Class(vbinop_av, vop_av, rec(numargs:=2, vcost := 1)); #Class added for AltiVec
Class(vbinop_spu, vop_spu, rec(numargs:=2, vcost := 1)); #Class added for SPU

Class(vunop_av, vop_av, rec(numargs:=1, vcost := 1)); #Class added for AltiVec
Class(vunop_spu, vop_spu, rec(numargs:=1, vcost := 1)); #Class added for SPU


Class(vunbinop_av, vop_av,
    rec(
        numargs := 1,
        params := self >> self.binop.params(),
        v := self >> self.binop.v(),
        ctype := self >> self.binop.ctype(),
        cprint := self >> self.binop(self.args[1], self.args[1], self.p).cprint(),
        semantic := (self, in1, p) >> self.binop.semantic(in1, in1, p),
        vcost := 1
    )
); #Class added for AltiVec

Class(vunbinop_spu, vop_spu,
    rec(
        numargs := 1,
        params := self >> self.binop.params(),
        v := self >> self.binop.v(),
        ctype := self >> self.binop.ctype(),
        cprint := self >> self.binop(self.args[1], self.args[1], self.p).cprint(),
        semantic := (self, in1, p) >> self.binop.semantic(in1, in1, p),
        vcost := 1
    )
); #Class added for SPU
