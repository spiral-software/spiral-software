
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# ==========================================================================
# BaseIterative - base class for iterative SPL operators
#
# Subclasses must provide
#    unroll := self >> ...unrolled (non-iterative) spl...
#    dims := self >> ...compute dimensions..
#    transpose := self >> ...transposed spl...
#
# if default implementation below is not applicable, subclasses must redefine
#   arithmeticCost := (self, costMul, costAddMul) >> ...
#
# ==========================================================================

IsIterative := x -> IsRec(x) and IsBound(x.isIterative) and x.isIterative;

Class(BaseIterative, BaseOperation, rec(
    
    isIterative := true,
    #-----------------------------------------------------------------------
    # Subclasses must provide
    unroll := self >> Error("This method must be provided by the subclass of BaseIterative"),
    dims := self >> Error("This method must be provided by the subclass of BaseIterative"),
    transpose := self >> Error("This method must be provided by the subclass of BaseIterative"),

    free := self >>
        Difference(Union(List(self.children(), x->x.free())), [self.var]),

    from_rChildren := (self, rch) >> CopyFields(self, rec(_children := rch)),

    # if default implementation below is not applicable, subclasses must redefine
    # arithmeticCost := (self, costMul, costAddMul) >> ...
    #
    #-----------------------------------------------------------------------
    abbrevs := [ (v, expr) -> [v, v.range, expr] ],

    new := meth(self, var, domain, expr)
        local res;
        Constraint(IsSPL(expr));
        # if domain is an integer (not symbolic) it must be positive
        Constraint(not IsInt(domain) or domain >= 0);
        var.isLoopIndex := true;
        res := SPL(WithBases(self, rec(_children := [expr], var := var, domain := domain)));
        res.dimensions := res.dims();
        return res;
    end,
    #-----------------------------------------------------------------------
    unrolledChildren := self >> List(listRange(EvalScalar(self.domain)), u -> self.at(TInt.value(u))),

    at := (self, u) >> SubstVars(Copy(self._children[1]), rec((self.var.id):=u)),

    #-----------------------------------------------------------------------
    # .split(m), splits the original loop into two nested loops with m
    #            iterations in the inner loop
    # This operations works with any kind of iterative operator, like ISum,
    # and IDirSum. It requres the operator class to provide .directOper, which
    # is the non-iterative version of the same operator (e.g. SUM for ISum)
    #
    split := (self, m) >> Cond(
	m=1,
	    self, 
	let(
            N  := self.domain,      _It := ObjId(self),  _Ch := self.directOper,
            j1 := Ind(idiv(N, m)),  j2  := Ind(m),       j3  := Ind(imod(N, m)),

            is := _It(j1, j1.range, _It(j2, j2.range, self.at(j1*m + j2))),
            When(j3.range = 0, is,
		_Ch(is, _It(j3, j3.range, self.at(m * j1.range + j3)))))
    ),
    #-----------------------------------------------------------------------
    print := (self, i, is) >> Print(
        self.name, "(", self.var, ", ", self.domain, ",\n",
        Blanks(i+is), self._children[1].print(i+is, is), "\n",
        Blanks(i), ")", self.printA(),
        When(IsBound(self._setDims), Print(".overrideDims(", self._setDims, ")"), Print(""))
    ),
    #-----------------------------------------------------------------------
    arithmeticCost := (self, costMul, costAddMul) >>
        self.domain * self._children[1].arithmeticCost(costMul, costAddMul),
    #-----------------------------------------------------------------------
    isPermutation := False,
    isReal := self >> self._children[1].isReal(),
    toAMat := self >> self.unroll().toAMat()
));

