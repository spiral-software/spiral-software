
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details



NewRulesFor(InterpolateDFT, rec(
    InterpolateDFT_tSPL_PrunedDFT := rec(
       applicable := (self, nt) >> nt.hasTags() and ObjId(nt.params[2]) = fZeroPadMiddle,
       children := nt -> let(n:=nt.params[2].domain(), N := nt.params[1].range(), us := N/n, blk := n/2,
            [[ 
                TCompose([
                    Downsample(nt.params[1]),
                    PrunedDFT(N, 1, blk, [0, 2*us-1]),
                    TDiag(fConst(n, 1/n)), 
                    DFT(n, -1)
                ]).withTags(nt.getTags())
            ]]),
       apply := (nt, C, cnt) -> C[1] 
    )
));

_maxBkSize := function(exp)
    local i, l, d;
    
    if IsExp(exp) and ObjId(exp)=mul and IsValue(exp.args[1]) then
        return exp.args[1];
    fi;

    if IsInt(exp.eval()) or IsValue(exp.eval()) then return EvalScalar(exp); fi;
    l := Lambda(Filtered(exp.free(), IsLoopIndex), exp);
    d := List(spiral.sigma.GenerateData(l).tolist(), EvalScalar);
    return Gcd(d);
end;

_divideFunc := function(exp, dval)
    local l, d, i;

    if IsValue(exp) or IsInt(exp) then return exp/dval; fi;

    if IsExp(exp) and ObjId(exp)=mul and IsValue(exp.args[1]) then
        return ApplyFunc(mul, Concat([idiv(exp.args[1], dval)], Drop(exp.args, 1)));
    fi;

    if IsInt(exp.eval()) or IsValue(exp.eval()) then return EvalScalar(exp/dval); fi;
    i := Filtered(exp.free(), IsLoopIndex)[1];
    l := Lambda(i, exp);
    d := List(spiral.sigma.GenerateData(l).tolist(), EvalScalar)/dval;
    return FData(List(d, j->i.t.value(j))).at(i);
end;

NewRulesFor(Downsample, rec(
    Downsample_tag := rec(
        applicable := (self, nt) >> nt.hasTags(),
        apply := (nt, C, cnt) -> let(gcd := _maxBkSize(nt.params[1].domain()),
            NeedInterleavedComplex(ScatGath(fTensor(fId(_divideFunc(nt.params[1].domain(), gcd)), fId(gcd)), nt.params[1])))
    )
));


NewRulesFor(InterpolateSegmentDFT, rec(
    InterpolateSegmentDFT_tSPL_PrunedDFT := rec(
        applicable := (self, nt) >> nt.hasTags() and ObjId(nt.params[6]) = fZeroPadMiddle,
        children := nt -> let(tags := nt.getTags(),
                              numseg := nt.params[1], j := Ind(numseg), downsample := nt.params[5].at(j), upsample := nt.params[6], 
                              insize := nt.params[3], overlap := nt.params[4], n := upsample.domain(), N := downsample.range(), 
                              us := N/n, blk := n/2, 
                              inp := n + (numseg-1)*(n-overlap), over := inp - insize, l := n - over, g := Gcd(n, l), 
                              
            [[ 
                Downsample(downsample).withTags(tags),
                PrunedDFT(N, 1, blk, [0, 2*us-1]).withTags(tags), 
                DFT(n, -1).withTags(tags),
                PrunedDFT(n, -1, g, [0..l/g-1]).withTags(tags),
                InfoNt(j) 
            ]]),
        apply := (nt, C, cnt) -> let(numseg := nt.params[1], _j := cnt[5].params[1], j := Ind(_j.range-1), downsample := nt.params[5].at(j), upsample := nt.params[6],
                              outsize := nt.params[2], insize := nt.params[3], overlap := nt.params[4], n := upsample.domain(), N := downsample.range(), 
                              ids_rows := _sumSegDims(numseg-1, nt.params[5]),
#                              ids_rows := RulesStrengthReduce(outsize - nt.params[5].at(numseg-1).domain()),
                RowDirectSum(overlap,[
                    IRowDirSum(j, numseg-1, overlap, 
                       SubstVars(Copy(C[1]), rec((_j.id) := j)) * C[2] * Diag(fConst(n, 1/n)) * C[3]).overrideDims([ids_rows, insize - C[4].dims()[2] + overlap]),
                    SubstVars(Copy(C[1]), rec((_j.id) := V(numseg-1))) * C[2] * Diag(fConst(n, 1/n)) * C[4]
                 ]).overrideDims([nt.params[2], nt.params[3]])
       )
    )
));

