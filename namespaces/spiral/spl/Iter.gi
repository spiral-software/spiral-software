
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# ==========================================================================
# IDirSum(<var>, <range>, <spl>) - Iterative direct sum.
#    Dimensions of <spl> are assumed to not depend on <var>.
# ==========================================================================
Class(IDirSum, BaseIterative, rec(
    directOper := DirectSum,
    #-----------------------------------------------------------------------
    dims := self >> let(d:=self._children[1].dimensions,
        [d[1]* self.domain, d[2]*self.domain]),
#    dims := self >> _evInt(self.unroll().dims()),
    #-----------------------------------------------------------------------
    unroll := self >> DirectSum(self.unrolledChildren()),
    #-----------------------------------------------------------------------
    transpose := self >> CopyFields(self, rec(
        _children := [self._children[1].transpose()],
        dimensions := Reversed(self.dimensions))),
    #-----------------------------------------------------------------------
    conjTranspose := self >> CopyFields(self, rec(
        _children := [self._children[1].conjTranspose()],
        dimensions := Reversed(self.dimensions))),
    #-----------------------------------------------------------------------
    inverse := self >> CopyFields(self, rec(
        _children := [self._children[1].inverse()],
        dimensions := Reversed(self.dimensions)))
));

Declare(IColDirSum, IRowDirSum);

# ==========================================================================
# IRowDirSum(<var>, <range>, <overlap>, <spl>) - Iterative direct sum.
#    Dimensions of <spl> are assumed to not depend on <var>.
# ==========================================================================
Class(IRowDirSum, BaseIterative, rec(
    directOper := RowDirectSum,
    transposeOper := self >> IColDirSum,
    abbrevs := [ (v, expr) -> [v, v.range, 0, expr] ],

    rChildren := self >> Concat([self.var, self.overlap, self._children], When(IsBound(self._setDims), [self._setDims],[])),
    rSetChild := rSetChildFields("var", "overlap", "_children", "_setDims"),

    new := meth(self, var, domain, overlap, expr)
        local res;
        Constraint(IsSPL(expr));
        Constraint(not IsInt(domain) or domain > 0);
        var.isLoopIndex := true;
        res := SPL(WithBases(self, rec(_children := [expr], var := var, domain := domain, overlap := overlap)));
        res.dimensions := res.dims();
        return res;
    end,
    #-----------------------------------------------------------------------
    print := meth(self, indent, indentStep)
        Print(self.name, "(", self.var, ", ", self.domain, ", ", self.overlap, ",");
        self._newline(indent + indentStep);
        SPLOps.Print(self._children[1], indent+indentStep, indentStep); #, ", ", self.nt_maps);
        self._newline(indent);
        Print(")");
        if IsBound(self._setDims) then
            Print(".overrideDims(", self._setDims, ")");
        fi;
    end,
    #-----------------------------------------------------------------------
    _dims := self >> self.unroll().dims(),
    dims := self >> When(IsBound(self._setDims), self._setDims, let(d := Try(self._dims()), When(d[1], d[2], [errExp(), errExp()]))),
    #-----------------------------------------------------------------------
    unroll := self >> ApplyFunc(self.directOper, [self.overlap, self.unrolledChildren()]),
    #-----------------------------------------------------------------------
    transpose := self >> ApplyFunc(self.transposeOper(), [self.var, self.domain, self.overlap, self.child(1).transpose()]),
    #-----------------------------------------------------------------------
    conjTranspose := self >> ApplyFunc(self.transposeOper(), [self.var, self.domain, self.overlap, self.child(1).conjTranspose()])
));


# ==========================================================================
# IColDirSum(<var>, <range>, <overlap>, <spl>) - Iterative direct sum.
#    Dimensions of <spl> are assumed to not depend on <var>.
# ==========================================================================
Class(IColDirSum, IRowDirSum, rec(
    directOper := ColDirectSum,
    transposeOper := self >> IRowDirSum
));


# ==========================================================================
# IterCompose(<var>, <domain>, <spl>) - iterative compose
#     <spl> must be square
# ==========================================================================
Class(IterCompose, BaseIterative, rec(
    directOper := Compose,
    #-----------------------------------------------------------------------
    dims := self >> self._children[1].dimensions,
    #-----------------------------------------------------------------------
    unroll := self >> Compose(self.unrolledChildren()),
    #-----------------------------------------------------------------------
    # NOTE: this is incorrect, one has to reverse the order
    transpose := self >> CopyFields(self, rec(
        _children := [self._children[1].transpose()],
        domain := self.domain,
        dimensions := self.dimensions))
    # NOTE: conjTranspose() is missing
));

Declare(IterVStack);

# ==========================================================================
# IterHStack(<var>, <domain>, <spl>)
#    Dimensions of <spl> are assumed to not depend on <var>.
# ==========================================================================
Class(IterHStack, BaseIterative, rec(
    directOper := HStack,
    #-----------------------------------------------------------------------
    unroll := self >> HStack(self.unrolledChildren()),
    #-----------------------------------------------------------------------
    dims := self >> [ self.child(1).dimensions[1],
                  self.child(1).dimensions[2] * self.domain ],
    #-----------------------------------------------------------------------
    transpose := self >> IterVStack(self.var, self.domain, self.child(1).transpose()),
    conjTranspose := self >> IterVStack(self.var, self.domain, self.child(1).conjTranspose())
));

# this is a HStack that is guaranteed not to perform any ops, and doesnt need ScatAcc in SumsGen
Class(IterHStack1, IterHStack);
# ==========================================================================
# IterVStack(<var>, <domain>, <spl>)
#    Dimensions of <spl> are assumed to not depend on <var>.
# ==========================================================================
Class(IterVStack, BaseIterative, rec(
    directOper := VStack,
    #-----------------------------------------------------------------------
    unroll := self >> VStack(self.unrolledChildren()),
    #-----------------------------------------------------------------------
    dims := self >> [ self.child(1).dimensions[1] * self.domain,
                  self.child(1).dimensions[2] ],
     #-----------------------------------------------------------------------
    transpose := self >> IterHStack(self.var, self.domain, self.child(1).transpose()),
    conjTranspose := self >> IterHStack(self.var, self.domain, self.child(1).conjTranspose())
));

#IterDirectSum := IDirSum;
# ==========================================================================
# IterDirectSum(<var>, <domain>, <spl>)
#    Dimensions of <spl> are assumed to not depend on <var>.
# ==========================================================================
Class(IterDirectSum, BaseIterative, rec(
    directOper := DirectSum, 
    #-----------------------------------------------------------------------
    unroll := self >> DirectSum(self.unrolledChildren()),
    #-----------------------------------------------------------------------
    dims := self >> [self.child(1).dimensions[1] * self.domain, self.child(1).dimensions[2]*self.domain],
    #-----------------------------------------------------------------------
    transpose := self >> IterHStack(self.var, self.domain, self.child(1).transpose()),
    conjTranspose := self >> IterHStack(self.var, self.domain, self.child(1).conjTranspose())
));
