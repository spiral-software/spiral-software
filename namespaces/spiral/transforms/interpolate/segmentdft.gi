
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Class(InterpolateSegmentDFT, TaggedNonTerminal, rec(
    abbrevs := [ (numseg, outsize, insize, overlap, dsfunc,usfunc) -> [numseg, outsize, insize, overlap, dsfunc, usfunc] ],

    dims := self >> [self.params[2], self.params[3]],

    terminate := self >> let(
        numseg  := self.params[1], 
        rowlength := self.params[2],
        insize  := self.params[3], 
        j       := Ind(numseg),
        overlap := self.params[4], 
        downsample := self.params[5].at(j), 
        upsample   := self.params[6],
        n       := upsample.domain(), 
        N       := downsample.range(), 
        substrec := i -> rec((j.id):=V(i)),
 
        kernel := Mat(MatAMat(RowDirectSum(
            overlap,
            List([1..numseg], i -> 
                Downsample(SubstVars(Copy(downsample), substrec(i-1))).terminate() *
                DFT(N, 1).terminate() * 
                Upsample(upsample).terminate() *
                Scale(1/n, DFT(n, -1).terminate())
            )).toAMat() * 
            Scat(fAdd(n*numseg-(numseg-1)*overlap, insize, 0)).toAMat()
        )),
        When(Rows(kernel) = rowlength, kernel, VStack(kernel, O(rowlength-Rows(kernel), Cols(kernel)))) 
    ),
    isReal    := False,
    normalizedArithCost := self >> let(numseg := self.params[1], n:=self.params[6].domain(), N := self.params[6].range(), 
        numseg * (n + IntDouble(5 * n * d_log(n) / d_log(2)) + IntDouble(5 * N * d_log(N) / d_log(2)))),
    TType := T_Complex(T_Real(64)),

    HashId := self >> let(h := [ self.params[1], self.params[2], self.params[3], self.params[4], self.params[5].domain(), self.params[5].range()],
        When(IsBound(self.tags), Concatenation(h, self.tags), h)),
    doNotMeasure := true
));


_sumSegDims := function(numsegs, seglen)
    local l, lfact, lsum, divsum, ssum;

    l := List([0..numsegs-1], i->seglen.at(i).domain());
    
    # check to pull out the mul
    if ForAll(l, i->ObjId(i) = mul and IsValue(i.args[1]) and i.args[1].v = l[1].args[1].v) then
        lfact := l[1].args[1];
        lsum := List(l, i -> i.args[2]);
        # we may need to create a table, but not sure yet...
        divsum := ApplyFunc(add, lsum);
        ssum := mul(lfact, divsum);
    else
        ssum := ApplyFunc(add, l);
    fi;
    
    return ssum;
end;

NewRulesFor(InterpolateSegmentDFT, rec(
    InterpolateSegmentDFT_base := rec(
        switch := false,
        applicable := (self, nt) >> not nt.hasTags() and ObjId(nt.params[6]) = fZeroPadMiddle,
        children := nt -> let(numseg := nt.params[1], j := Ind(numseg), downsample := nt.params[5].at(j), upsample := nt.params[6],
                              insize := nt.params[3], overlap := nt.params[4], n:=upsample.domain(), N := downsample.range(), 
                              us := N/n, blk := n/2, 
                              inp := n + (numseg-1)*(n-overlap), over := inp - insize, l := n - over,
            [[ 
                Downsample(downsample),
                DFT(N, 1), 
                Upsample(upsample),
                DFT(n, -1),
                Upsample(fAdd(n, l, 0)),
                InfoNt(j)
            ]]),
        apply := (nt, C, cnt) -> let(numseg := nt.params[1], _j := cnt[6].params[1], j := Ind(_j.range-1), downsample := nt.params[5].at(j), upsample := nt.params[6], 
                              outsize := nt.params[2], insize := nt.params[2], overlap := nt.params[4], n:=upsample.domain(), N := downsample.range(), 
                              ids_rows := _sumSegDims(numseg-1, nt.params[5]),
                RowDirectSum(overlap,[
                    IRowDirSum(j, numseg-1, overlap, 
                        SubstVars(Copy(C[1]), rec((_j.id) := j)) * C[2] * C[3] * Scale(1/n, C[4])).overrideDims([ids_rows, insize - C[4].dims()[2] + overlap]),
                    SubstVars(Copy(C[1]), rec((_j.id) := V(numseg-1))) * C[2] * C[3] * Scale(1/n, C[4] * C[5]) 
                ]).overrideDims([nt.params[2], nt.params[3]])
       )
    ),
    InterpolateSegmentDFT_PrunedDFT := rec(
        applicable := (self, nt) >> not nt.hasTags() and ObjId(nt.params[6]) = fZeroPadMiddle,
        children := nt -> let(numseg := nt.params[1], j := Ind(numseg), downsample := nt.params[5].at(j), upsample := nt.params[6], 
                              insize := nt.params[3], overlap := nt.params[4], n:=upsample.domain(), N := downsample.range(), 
                              us := N/n, blk := n/2, 
                              inp := n + (numseg-1)*(n-overlap), over := inp - insize, l := n - over, g := Gcd(n, l), 
            [[ 
                Downsample(downsample),
                PrunedDFT(N, 1, blk, [0, 2*us-1]), 
                DFT(n, -1),
                PrunedDFT(n, -1, g, [0..l/g-1]),
                InfoNt(j) 
            ]]),
        apply := (nt, C, cnt) -> let(numseg := nt.params[1], _j := cnt[5].params[1], j := Ind(_j.range-1), downsample := nt.params[5].at(j), upsample := nt.params[6],
                              outsize := nt.params[2], insize := nt.params[3], overlap := nt.params[4], n := upsample.domain(), N := downsample.range(), 
                              ids_rows := _sumSegDims(numseg-1, nt.params[5]),
                RowDirectSum(overlap,[
                    IRowDirSum(j, numseg-1, overlap, 
                        SubstVars(Copy(C[1]), rec((_j.id) := j)) * C[2] * Diag(fConst(n, 1/n)) * C[3]).overrideDims([ids_rows, insize - C[4].dims()[2] + overlap]),
                    SubstVars(Copy(C[1]), rec((_j.id) := V(numseg-1))) * C[2] * Diag(fConst(n, 1/n)) * C[4]
                ]).overrideDims([nt.params[2], nt.params[3]])
       )
    )
));
