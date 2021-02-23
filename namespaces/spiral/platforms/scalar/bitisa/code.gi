
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

# -------------------------------------------------------------------------
# Scalar bit-level instructions
# 16x1i: 16-way bit register
# -------------------------------------------------------------------------

Class(sklr_bcast_16x1i,  VecExp_16.binary());
Class(sklr_loadu_16x1i,  VecExp_16.ternary());
Class(sklr_storeu_16x1i, VecExpCommand.quad(), rec(
    isStoreop := True,
    __call__ := (self, loc, offs, exp, p) >> WithBases(self,
       rec(operations := CmdOps,
       loc := toAssignTarget(loc),
       offs := toExpArg(offs),
       exp := toExpArg(exp),
       p := p)),

    rChildren := self >> [self.loc, self.offs, self.exp, self.p],
    rSetChild := rSetChildFields("loc", "offs", "exp", "p"),

    print := (self,i,is) >> Print(self.__name__, "(", self.loc.print(), ", ", self.offs.print(), ", ", self.exp.print(), ", ", self.p, ")"),

    # in case of explicit type cast (YSV modification), we don't need getNoScalar,
    # and below returns [], compiler understands not to mess with typecasts
    getNoScalar := self >> When(IsBound(self.noscalar) and IsBound(self.args[self.noScalar].loc),
    self.args[self.noScalar].loc, [])
));

# -------------------------------------------------------------------------
# Scalar bit-level instructions
# 32x1i: 32-way bit register
# -------------------------------------------------------------------------

Class(sklr_bcast_32x1i,  VecExp_32.binary());
Class(sklr_loadu_32x1i,  VecExp_32.ternary());
Class(sklr_storeu_32x1i, VecExpCommand.quad(), rec(
    isStoreop := True,
    __call__ := (self, loc, offs, exp, p) >> WithBases(self,
       rec(operations := CmdOps,
       loc := toAssignTarget(loc),
       offs := toExpArg(offs),
       exp := toExpArg(exp),
       p := toExpArg(p))),

    rChildren := self >> [self.loc, self.offs, self.exp, self.p],
    rSetChild := rSetChildFields("loc", "offs", "exp", "p"),

    print := (self,i,is) >> Print(self.__name__, "(", self.loc.print(), ", ", self.offs.print(), ", ", self.exp.print(), ", ", self.p, ")"),

    # in case of explicit type cast (YSV modification), we don't need getNoScalar,
    # and below returns [], compiler understands not to mess with typecasts
    getNoScalar := self >> When(IsBound(self.noscalar) and IsBound(self.args[self.noScalar].loc),
    self.args[self.noScalar].loc, [])
));

# -------------------------------------------------------------------------
# Scalar bit-level instructions
# 64x1i: 64-way bit register
# -------------------------------------------------------------------------

Class(sklr_bcast_64x1i, VecExp_64.binary());
Class(sklr_loadu_64x1i, VecExp_64.ternary());
Class(sklr_storeu_64x1i, VecExpCommand.quad(), rec(
    isStoreop := True,
    __call__ := (self, loc, offs, exp, p) >> WithBases(self,
       rec(operations := CmdOps,
       loc := toAssignTarget(loc),
       offs := toExpArg(offs),
       exp := toExpArg(exp),
       p := toExpArg(p))),

    rChildren := self >> [self.loc, self.offs, self.exp, self.p],
    rSetChild := rSetChildFields("loc", "offs", "exp", "p"),

    print := (self,i,is) >> Print(self.__name__, "(", self.loc.print(), ", ", self.offs.print(), ", ", self.exp.print(), ", ", self.p, ")"),

    # in case of explicit type cast (YSV modification), we don't need getNoScalar,
    # and below returns [], compiler understands not to mess with typecasts
    getNoScalar := self >> When(IsBound(self.noscalar) and IsBound(self.args[self.noScalar].loc),
    self.args[self.noScalar].loc, [])
));

