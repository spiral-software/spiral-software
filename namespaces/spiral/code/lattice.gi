
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details



Class(vdup, Exp, rec(
    __call__ := (self, a, n) >> WithBases(self, rec(
	    args       := [toExpArg(a), toExpArg(n)],
	    operations := ExpOps,
	)).setType(),

    computeType := self >> let(t := self.args[1].t, 
	Cond(IsVecT(t), 
	         ObjId(t)(t.t, t.size * self.args[2].ev()),
	     # else
	         TVect(t, self.args[2].ev())
	)),

    ev := self >> Value(self.t, Flat(Replicate(self.args[2].ev(), self.args[1].ev())))
));

# tcast(<newtype>, <exp>)
#
Class(tcast, Exp, rec(
    isLoc := true,
    __call__ := (self, t, exp) >> Checked(IsType(t), WithBases(self, rec(
	 args := [t, toExpArg(exp)],
	 t := t, 
	 operations := ExpOps
    ))),

    zero := self >> self.args[1].zero(),
    
    computeType := self >> self.args[1]
));

# Packs n arguments into an n-way vector,
# this should be considered a pseudo-instruction
#
Class(vpack, AutoFoldExp, rec(
    ev := self >> self.t.value(List(self.args, x->x.ev())),
    computeType := self >> TVect(UnifyTypes(List(self.args, x->x.t)), Length(self.args))
));

Class(velem, nth, rec(
    __call__ := (self, loc, idx) >> Cond(
	# velem(vpack(a, b), 1) == a
	ObjId(loc)=vpack and IsInt(idx), loc.args[idx+1],	
        WithBases(self, rec(loc := loc, idx := toExpArg(idx), t := loc.t.t, operations := ExpOps))),

    computeType := self >> self.loc.t.t,
    cprint := self >> Print(self.name, "(", self.loc.cprint(), ", ", self.idx.cprint(), ")"),
    print := self >> Print(self.name, "(", self.loc, ", ", self.idx, ")")
));

# tcvt(<newtype>, <exp>)
#
Class(tcvt, tcast);

# sizeof(<type>)
#
Class(sizeof, Exp, rec(
    __call__ := (self, t) >> Checked(IsType(t), 
        WithBases(self, rec(args := [t], t := TInt, operations := ExpOps))),
    
    computeType := self >> TInt
));
