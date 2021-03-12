
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#######################################################################################
#   tSPL DFT rules
NewRulesFor(PrunedDFT, rec(
    PrunedDFT_tSPL_CT := rec(

    forTransposition := true,

    maxSize       := false,

    applicable    := (self, nt) >> nt.params[1] > 2
        and (self.maxSize = false or nt.params[1] <= self.maxSize)
        and not IsPrime(nt.params[1])
        and nt.params[3] > 1
        and nt.hasTags(),

    children      := nt -> Map2(Filtered(DivisorPairs(nt.params[1]), (l) -> IsInt(nt.params[3]/l[1])), (m,n) -> [
        TCompose([
            TGrp(TCompose([
                TTensorI(DFT(m, nt.params[2] mod m), n, AVec, AVec),
                TDiag(fPrecompute(Tw1(m*n, n, nt.params[2])))
            ])),
            TGrp(TTensorI(PrunedDFT(n, nt.params[2] mod n, nt.params[3]/m, nt.params[4]), m, APar, AVec))
        ]).withTags(nt.getTags())
    ]),

    apply := (nt, c, cnt) -> c[1],

    switch := false
    )
));


NewRulesFor(IOPrunedDFT, rec(
    IOPrunedDFT_tSPL_CT := rec(

    forTransposition := true,

    maxSize       := false,

    applicable    := (self, nt) >> nt.params[1] > 2
        and (self.maxSize = false or nt.params[1] <= self.maxSize)
        and not IsPrime(nt.params[1])
        and nt.params[3] * nt.params[5] >= nt.params[1]
        and nt.hasTags(),

    children      := nt -> Map2(Filtered(DivisorPairs(nt.params[1]), (l) -> IsInt(nt.params[3]/l[2]) and IsInt(nt.params[5]/l[1])), (m,n) -> [
        TCompose([
            TGrp(TCompose([
                TTensorI(PrunedDFT(m, nt.params[2] mod m, nt.params[3]/n, nt.params[4]).transpose(), n, AVec, AVec),
                TDiag(fPrecompute(Tw1(m*n, n, nt.params[2])))
            ])),
            TGrp(TTensorI(PrunedDFT(n, nt.params[2] mod n, nt.params[5]/m, nt.params[6]), m, APar, AVec))
        ]).withTags(nt.getTags())
    ]),

    apply := (nt, c, cnt) -> c[1],

    switch := false
    )
));
