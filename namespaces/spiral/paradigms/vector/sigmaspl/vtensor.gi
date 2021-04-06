
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(VTensor, RCVTensor, VTensorInd);

#   A x I_v
Class(VTensor, Tensor, rec(
    new := (self, L) >> SPL(WithBases(self, rec(
        _children := [L[1]],
        dimensions := When(IsBound(L[1].dims), L[1].dims(), L[1].dimensions) * L[2],
        vlen := L[2]))),
    # NOTE: vlen not exposed to rChildren
    from_rChildren := (self, rch) >> ObjId(self)(rch[1], self.vlen),

    normalizedArithCost := (self) >> self.vlen * self.child(1).normalizedArithCost(),
    print := (self,i,is) >> Print(self.name, "(",
    self.child(1).print(i+is,is), ", ", self.vlen, ")"),
    toAMat := self >> Tensor(self.child(1), I(self.vlen)).toAMat(),
    sums := self >> Inherit(self, rec(_children := [self.child(1).sums()])),
    isPermutation := False,
    dims := self >> self.child(1).dims() * self.vlen,
    needInterleavedLeft := False,
    needInterleavedRight := False,
    transpose := self >> VTensor(self.child(1).transpose(), self.vlen),
    # for BlockSums
    isBlockTransitive := true,
    area := self >> self.child(1).area() * self.vlen,
    #NOTE: check if these two should be set to true or false
    cannotChangeDataFormat := False,
    totallyCannotChangeDataFormat := False
));


#   RC(A x I_v)
Class(RCVTensor, Tensor, rec(
    new := (self, L) >> SPL(WithBases(self, rec(
        _children := [L[1]],
        dimensions := 2 * L[1].dimensions * L[2],
        vlen := L[2]))),
    # NOTE: vlen not exposed to rChildren
    from_rChildren := (self, rch) >> ObjId(self)(rch[1], self.vlen),
    print := (self,i,is) >> Print(self.name, "(",
    self.child(1).print(i+is,is), ", ", self.vlen, ")"),
    toAMat := self >> RC(Tensor(self.child(1), I(self.vlen)).toAMat()),
    sums := self >> Inherit(self, rec(_children := [self.child(1).sums()])),
    isPermutation := False,
    dims := self >> self.child(1).dims() * self.vlen * 2,
    needInterleavedLeft := True,
    needInterleavedRight := True,
    transpose := self >> RCVTensor(self.child(1).transpose(), self.vlen),
    # for BlockSums
    isBlockTransitive := true
));


Class(VTensorInd, Tensor, rec(
    new := (self, L) >> SPL(WithBases(self, rec(
        _children := [L[1]],
        dimensions := When(IsBound(L[1].dims), L[1].dims(), L[1].dimensions) * L[2].range,
        vlen := L[2]))),
    from_rChildren := (self, rch) >> ObjId(self)(rch[1], self.vlen),

    rChildren := self >> [self._children[1], self.vlen],

    rSetChild := meth ( self, n, what )
            if n=1 then
                self._children[1] := what;
            elif n=2 then
                self.vlen := what;
            else
                Error("VTensorInd only has 2 rChildren");
            fi;
        end,
#    from_rChildren := (self, rch) >> ObjId(self)(rch[1], rch[2]),

    print := (self,i,is) >> Print(self.name, "(",
    self.child(1).print(i+is,is), ", ", self.vlen, ")"),
    toAMat := self >> let(d := self.child(1).dims(), v := self.vlen.range, L(d[1]*v, d[1]).toAMat() * IDirSum(self.vlen, self.child(1)).toAMat() * L(d[2]*v, v).toAMat()),
    sums := self >> Inherit(self, rec(_children := [self.child(1).sums()])),
    isPermutation := False,
    dims := self >> self.child(1).dims() * self.vlen.range,
    needInterleavedLeft := False,
    needInterleavedRight := False,
    transpose := self >> VTensorInd(self.child(1).transpose(), self.vlen),
    # for BlockSums
    isBlockTransitive := true,
    area := self >> self.child(1).area() * self.vlen.range,
    free := meth(self)
        local fvar;
        fvar := self.child(1).free();
        SubtractSet(fvar, Set([self.vlen]));
        return fvar;
    end
));
