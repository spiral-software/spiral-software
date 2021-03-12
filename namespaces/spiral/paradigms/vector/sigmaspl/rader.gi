
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(VecRaderMid, BaseMat, SumsBase, rec(
    sums := self >> self,
    isReal := False,
    dims := self >> self.dimensions,
    #-----------------------------------------------------------------------
    new := (self, p, k, r, v) >> SPL(WithBases(self,
        rec(p := p, k := k, r := r, v := v, dimensions := [v+_roundup(p-1,v), v+_roundup(p-1,v)]))),
    #-----------------------------------------------------------------------
    area := self >> self.dimensions[1] + 2,
    transpose := self >> self,
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.name, "(", self.p, ", ", self.k, ", ", self.r, ", ", self.v, ")"),
    #-----------------------------------------------------------------------
    toAMat := self >> self.term().toAMat(),
    rChildren := self >> [],
    from_rChildren := (self, rch) >> self,
    #-----------------------------------------------------------------------
    term := meth(self)
        local dl, d, e0, ep, blk, p, k, root, v;
        p := self.p; k:=self.k; root :=self.r; v:= self.v;
        dl := Concat(TRaderMid.raderDiag(p, k, root), List([p-1..v], i->V(0.0)));
        e0 := Concat([V(1.0)], List([2..v], i->V(0.0)));
        ep := Concat([V(-1/(p-1))], dl{[1..v-1]});
        blk := VBlk([[V(e0),V(e0)],[V(e0),V(ep)]], v).setSymmetric();

        if p > 2*v then
            d := VDiag(fStretch(FList(TComplex, dl{[v..p-2]}), _roundup(p-1-v, v), p-1-v), v);
            return DelayedDirectSum(blk, d);
        else
            return blk;
        fi;
    end,
    #-----------------------------------------------------------------------
    stretchCT := meth(self, N)
        local dl, d, e0, ep, blk, p, k, root, v;
        p := self.p;
        k:=self.k;
        root:=self.r;
        v:= self.v;
        dl := fStretch(FList(TComplex, Concat([-1/(p-1)], TRaderMid.raderDiag(p, k, root))), _roundup(N, v), N).tolist();
        d := VDiag(FList(TComplex, Drop(dl, v)), v);
        e0 := Concat([V(1.0)], List([2..v], i->V(0.0)));
        ep := dl{[1..v]};
        blk := VBlk([[e0,e0],[e0,ep]], v).setSymmetric();
        return DelayedDirectSum(blk, d);
    end,
    #-----------------------------------------------------------------------
    stretchPFA := meth(self, part, N, func)
        local dl, d, e0, ep, blk, p, k, root, v;
        p := self.p;
        k:=self.k;
        root:=self.r;
        v:= self.v;
#        Error();
        dl := fStretch(fCompose(FList(TComplex, Concat([-1/(p-1)], TRaderMid.raderDiag(p, k, root))), func), _roundup(N/part, v), N/part).tolist();
        d := VDiag(FList(TComplex, Drop(dl, v)), v);
        e0 := Concat([V(1.0)], List([2..v], i->V(0.0)));
        ep := dl{[1..v]};
        blk := VBlk([[e0,e0],[e0,ep]], v).setSymmetric();
        return DelayedDirectSum(blk, d);
    end,
));
