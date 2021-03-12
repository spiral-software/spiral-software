
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Parameter container
Class(vparam, Value, rec(
    __call__ := arg >> let(self := arg[1], WithBases(self,
    rec(p := arg[2], operations := PrintOps))),

    t := TSym("vparam"),
    isVparam := true,
    print := self >> Print(self.__name__, "(", self.p, ")")
));

# Container for hex numbers
Class(vhex, Value, rec(
    __call__ := arg >> let(self := arg[1], WithBases(self,
    rec(p := arg[2], t:=TVectDouble(Length(arg[2])), operations := PrintOps))),

    isVhex := true,
    print := self >> Print(self.name, "(", self.p, ")"),
    can_fold := False,
));

# Generic instructions
#
_vparam := vp -> When(ObjId(vp)=vparam, vp, vparam(vp));

#F VecExp
#F
#F This class serves as base class for vector instructions. It has
#F some extra fields to describe semantics, so that the magic in
#F paradigms/vector/bases can automatically build some simple base
#F cases like stride permutations.
#F
Class(VecExp, Exp, rec(
    __call__ := arg >> let(self:=arg[1], WithBases(self, rec(
        args  := Concat(List(Sublist(arg, [2..self.numargs+1]), toExpArg),
            When(Length(arg) >= self.numargs+2 and arg[self.numargs+2] <> [],
            [_vparam(arg[self.numargs+2])], [])),
        operations := ExpOps
    )).setType()),

    # Needs to be defined in subclasses:
    # v := ...    # vector length
    # numargs := ... # number of arguments, not counting vparam, which could come last

    # NOTE: this is ugly, because it derives type from 1st argument only
    computeType := self >> let(
        t       := self.args[1].t,
        deref_t := When(IsPtrT(t), t.t, t),
        el_t    := Cond(IsVecT(deref_t), deref_t.t, deref_t),
        TVect(el_t, self.v)),

    # _vval returns unwrapped (e.g. list) value by index
    _vval := (self, i) >> _unwrap( When( IsValue(self.args[i]) and not IsVecT(self.args[i].t),
                                       self.t.value(self.args[i]),
                                       self.args[i] )),

    params := self >> [],
    permparams := self >> Cartesian(self.params()),
    countAsVectOp := True,

    isBinop := self >> self.numargs = 2,
    isUnop := self >> self.numargs = 1,

    _unaryFromBinopFields := rec(
        numargs  := 1,
        params   := self >> self.binop.params(),
        semantic := (self, in1, p) >> self.binop.semantic(in1, in1, p),
        ev       := self >> self.toBinop().ev()
    ),

    # Example: Class(ushuffle, VecExp.unaryFromBinop(shuffle));
    unaryFromBinop := (self, binop) >> CopyFields(self, self._unaryFromBinopFields,
        rec(binop := binop, v := binop.v, vcost := binop.vcost)),
    toBinop := self >> ApplyFunc(self.binop, [self.args[1]] :: self.args),

    unary   := self >> CopyFields(self, rec(numargs := 1)),
    binary  := self >> CopyFields(self, rec(numargs := 2)),
    ternary := self >> CopyFields(self, rec(numargs := 3)),
    quad    := self >> CopyFields(self, rec(numargs := 4)),

    # instruction cost, used to estimate TL cost when building TL bases 
    vcost   := 1,
));

Class(VecExp_2, VecExp, rec(v := 2));
Class(VecExp_4, VecExp, rec(v := 4));
Class(VecExp_8, VecExp, rec(v := 8));
Class(VecExp_16, VecExp, rec(v := 16));
Class(VecExp_32, VecExp, rec(v := 32));
Class(VecExp_64, VecExp, rec(v := 64));
Class(VecExp_128, VecExp, rec(v := 128));


#F VecExpCommand
#F
#F Base classes for vector statements (commands) such as stores, which do not return
#F a value.
#F
Class(VecExpCommand, ExpCommand, rec(
    # Needs to be defined in subclasses:
    #   nothing
    unary   := self >> CopyFields(self, rec(numargs := 1)),
    binary  := self >> CopyFields(self, rec(numargs := 2)),
    ternary := self >> CopyFields(self, rec(numargs := 3)),
    quad    := self >> CopyFields(self, rec(numargs := 4)),
));

#F VecStoreCommand
#F
#F  Base class for store operations
#F
Class(VecStoreCommand, VecExpCommand, rec(
    op_in    := self >> ConcatList(self.args{[1..self.numargs]} , ArgsExp),
    op_out   := self >> [deref(self.args[1])], # ok, here is a problem
    op_inout := self >> [],

));

# evaluating binary operation using 'semantic' method
_ev_binop_semantic_mixin := rec(
    ev := self >> self.t.value(self.semantic(self._vval(1),self. _vval(2), [])),
);

