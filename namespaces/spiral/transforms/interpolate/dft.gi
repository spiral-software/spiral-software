
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Class(InterpolateDFT, TaggedNonTerminal, rec(
    abbrevs := [ (dsfunc,usfunc) -> [dsfunc, usfunc] ],
    dims      := self >> [self.params[1].domain(), self.params[2].domain()],
    terminate := self >> let(n:=self.params[2].domain(), N := self.params[1].range(), 
        Downsample(self.params[1]).terminate() * 
        DFT(N, 1).terminate() * 
        Upsample(self.params[2]).terminate() *
        Scale(1/n, DFT(n, -1).terminate())),
    isReal    := False,
    normalizedArithCost := self >> let(n:=self.params[2].domain(), N := self.params[1].range(), 
        n + IntDouble(5 * n * d_log(n) / d_log(2)) + IntDouble(5 * N * d_log(N) / d_log(2))),
    TType := T_Complex(T_Real(64))
));


NewRulesFor(InterpolateDFT, rec(
    InterpolateDFT_base := rec(
        applicable := (self, nt) >> not nt.hasTags(),
        children := nt -> let(n:=nt.params[2].domain(), N := nt.params[1].range(), 
            [[ Downsample(nt.params[1]), DFT(N, 1), Upsample(nt.params[2]), DFT(n, -1) ]]),
        apply := (nt, C, cnt) -> C[1] * C[2] * C[3] * Scale(1/nt.params[2].domain(), C[4]) 
    ),
    InterpolateDFT_PrunedDFT := rec(
        applicable := (self, nt) >> not nt.hasTags() and ObjId(nt.params[2]) = fZeroPadMiddle,
        children := nt -> let(n:=nt.params[2].domain(), N := nt.params[1].range(), us := N/n, blk := n/2, 
            [[ PrunedDFT(N, 1, blk, [0, 2*us-1]), DFT(n, -1) ]]),
        apply := (nt, C, cnt) -> let(n:=nt.params[2].domain(), N := nt.params[1].range(), us := N/n, blk := n/2,
                Gath(nt.params[1]) * C[1] * Scale(1/n, C[2]) 
       )
    )
));



