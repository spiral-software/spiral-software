
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(VIxL, VL, VPrm_x_I);

RecursStep.isSymmetric := self >> self.child(1).isSymmetric();
RTWrap.isSymmetric := self >> self.child(1).isSymmetric();
RuleTreeClass.isSymmetric := self >> self.node.isSymmetric();

#   in-register permutations to change complex format
Class(VIxL, BaseContainer, rec(
    new := (self, m, n, v) >> SPL(WithBases(self, rec(
        dimensions := [2*m*v, 2*m*v],
        m:=m,
        n:=n,
        v:=v
    ))),
    dims := self >> Replicate(2, 2*self.m*self.v),

    _children := [],
    rChildren := self >> [],
    from_rChildren := (self, rch) >> ObjId(self)(self.m, self.n, self.v),

    isPermutation := False,
    isReal := True,
    toAMat := self >> AMatSPL(Tensor(I(self.m), L(2*self.v, self.n))),
    sums := self >> self,
    print := (self,i,is) >> Print(self.name, "(", self.m, ", ", self.n, ", ", self.v, ")"),

    implement := meth(self, isa)
        local t, hentry, sums;
        t := TL(2*self.v, self.n, 1, 1).withTags(isa.getTags());
        hentry := HashLookup(SIMD_ISA_DB.getHash(), t);
        if hentry=false then
            return Error("SIMD_ISA_DB lookup failed, tried ", t); 
	else
            sums := Tensor(I(self.m), _SPLRuleTree(hentry[1].ruletree)).sums();
            sums := When(IsBound(sums.unroll), sums.unroll(), sums);
            return sums;
	fi;
    end,

    transpose := self >> VIxL(self.m, 2*self.v/self.n, self.v)
));


#   in-register stride perm
Class(VL, BaseContainer, rec(
    new := (self, mn, m) >> SPL(WithBases(self, rec(
        dimensions := [mn, mn],
        mn:=mn,
        m:=m
    ))),
    dims := self >> Replicate(2, self.mn),
    _children := [],

    rChildren := self >> [],
    from_rChildren := (self, rch) >> ObjId(self)(self.mn, self.m),

    isPermutation := False,
    isReal := True,
    toAMat := self >> AMatSPL(L(self.mn, self.m)),
    sums := self >> self,
    print := (self,i,is) >> Print(self.name, "(", self.mn, ", ", self.m, ")"),
    implement := (self, isa) >> _SPLRuleTree(HashLookup(SIMD_ISA_DB.getHash(), TL(self.mn, self.m, 1, 1).withTags(isa.getTags()))[1].ruletree).sums(),
    transpose := self >> VL(self.mn, self.nm/self.m)
));




#   Block diagonal permutation
# NOTE: Franz pls make sure rChildren properly exposes all parameters
Class(BlockVPerm, BaseContainer, rec(
    new := (self, n, vlen, spl, perm) >> SPL(WithBases(self, rec(
        _children := [spl],
        n := n,
        perm := perm,
        dimensions := n * spl.dimensions,
        vlen := vlen
    ))),

    from_rChildren := (self, rch) >> ObjId(self)(self.n, self.vlen, rch[1], self.perm),

    isPermutation := self >> self._children[1].isPermutation(),
    isReal := self >> self._children[1].isReal(),
    toAMat := self >> AMatSPL(Tensor(I(self.n), self._children[1])),

    print := (self,i,is) >> Print(self.name, "(", self.n, ", ", self._children[1].print(i+is, is), ")"),
    sums := self >> self,
    dims := self>>self.dimensions,
    needInterleavedLeft := False,
    needInterleavedRight := False,
#    code := (self, y, x) >> ApplyStrategy(
#   Tensor(I(self.n), self._children[1]).sums(),
#   VectorStrategySum, UntilDone).code(y, x);,
    transpose := self >> When(self.child(1).isSymmetric(), self, Error("transpose not supported"))
));

Class(BlockVPerm2, BlockVPerm);

Declare(VPerm);

#   in-register permutations
Class(VPerm, BaseContainer, rec(
    new := (self, spl, code, vlen, vcost) >> SPL(WithBases(self, rec(
        _children := [spl],
        dimensions := spl.dimensions,
        code:= code,
        vlen :=  vlen,
        _vcost := vcost
    ))),

    # NOTE: Franz pls make sure rChildren properly exposes all parameters
    from_rChildren := (self, rch) >> ObjId(self)(rch[1], self.code, self.vlen, self._vcost),

    isPermutation := False,
    isReal := True,
    toAMat := self >> AMatSPL(self._children[1]),

    printl := (self,i,is) >> Print(self.name, "(", self._children[1].print(i+is, is), ", ", self.code, ", ", self.vlen, ", ", self._vcost, ")"),
    prints := (self,i,is) >> Print(self.name, "(", self._children[1].print(i+is, is), ")"),

    sums := self >> self,
    vcost := self >> self._vcost,
    pshort := meth(self) self.print := self.prints; end,
    plong := meth(self) self.print := self.printl; end,
    needInterleavedLeft := False,
    needInterleavedRight := False,
    transpose := self >> When(self.child(1).isSymmetric(), self, Error("transpose not supported")),
    isSymmetric := self >> self.child(1).isSymmetric(),
    area := (self) >> self.dims()[1]
));

VPerm.pshort();

# ==========================================================================
# VPrm_x_I(<func>, <vlen>) - permutation
# ==========================================================================
Class(VPrm_x_I, Prm, SumsBase, rec(
    #-----------------------------------------------------------------------
    abbrevs := [ (func, v) -> 
	Checked(IsFunction(func) or IsFuncExp(func), IsInt(v), [func, v]) ],
    #-----------------------------------------------------------------------
    rChildren := self >> [self.func],
    rSetChild := rSetChildFields("func"), 
    from_rChildren := (self, rch) >> ObjId(self)(rch[1], self.v),
    #-----------------------------------------------------------------------
    area := self >> Rows(self) * self.v,
    #-----------------------------------------------------------------------
    new := (self, func, v) >> SPL(WithBases(self, rec(
        func := func, v := v))).setDims(), 
    #-----------------------------------------------------------------------
    dims := self >> let(n := self.func.domain(), [n * self.v, n * self.v]),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(
	self.name, "(", self.func, ", ", self.v, ")", self.printA()),
    #-----------------------------------------------------------------------
    toAMat := self >> let(f := self.func.lambda(),
	Tensor(Perm(PermList(List(f.tolist(), EvalScalar) + 1),
                    f.domain()), I(self.v)).toAMat()),
    #-----------------------------------------------------------------------
    sums := self >> self,
    needInterleavedLeft := False,
    needInterleavedRight := False,
    cannotChangeDataFormat := True,
    totallyCannotChangeDataFormat := True,
    transpose := self >> VPrm_x_I(self.func.transpose(), self.v)
));
