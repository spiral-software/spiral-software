
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


ImportAll(paradigms.smp); # for AParSMP
ImportAll(paradigms.vector); # for VRCLR
Import(approx); # for CeilingRat.

_swrap := (nt) -> let(t:=nt.firstTag(), ScratchWrap(t.size,t.nsgmts,t.linesize));

_nodeFitsTensorDoesnt := (nt) -> let (
    k := nt.getTag(1).size,
    m := nt.params[1].dims()[2],
    n := nt.params[2],

    2*m <= k and m*n >= k
);

_isALStoreTag := (nt) -> nt.isTag(1,ALStore) or nt.isTag(1,ALStoreCx);

NewRulesFor(TTensorI, rec(
#   (In x Am)_{LS(<=k)} -> In x (Am)_{LS(<=k)} for m > k
    IxA_scratch_push := rec(
        forTransposition := false,
        applicable := nt -> nt.hasTags() 
            and _isALStoreTag(nt)
            and IsParPar(nt.params) 
            and nt.params[1].dims()[2] > nt.getTag(1).size,

        children := nt -> [[ nt.params[1].withTags(nt.getTags()) ]],
        apply := (nt, c, cnt) -> Tensor(I(nt.params[2]), c[1])
    ),
   
    AxI_scratch_push := rec(
        forTransposition := false,
        applicable := nt -> nt.hasTags()
            and _isALStoreTag(nt)
            and IsVecVec(nt.params)
            and nt.params[1].dims()[2] > nt.getTag(1).size,
        children := nt -> [[nt.params[1].withTags(nt.getTags())]],
        apply := (nt, c, cnt) -> Tensor(c[1], I(nt.params[2]))
    ),

#   ==========================================================
# synchronous rules

#   (In x Am)_{LS(<=k)} -> Imn/k x (Ik/m x Am) for m|k and k|mn
   IxA_scratch := rec(
        forTransposition := false,
        applicable := nt -> nt.hasTags() 
            and _isALStoreTag(nt)
	        and IsParPar(nt.params)
            and IsPosInt(nt.getTag(1).size / nt.params[1].dims()[2])
            and IsPosInt(nt.params[1].dims()[2] * nt.params[2] / nt.getTag(1).size),

        children := nt -> [[ 
            TTensorI(
                nt.params[1],
                nt.getTag(1).size / nt.params[1].dims()[2], 
                APar, APar
            ).withTags(Drop(nt.getTags(),1)).setWrap(
                _swrap(nt)
            )
        ]],
        apply := (nt, c, cnt) -> let(k := nt.getTag(1).size,  m := nt.params[1].dims()[2], n:= nt.params[2],
                DMAFence(Tensor(
                    I(m*n/k),
                    LSKernel(c[1], nt.params[1].normalizedArithCost() * k/m)
                )))
    ),

#   (Am x In)_{LS(<=k)} -> (L^m^2n/k_m x Ik/m) (Imn/k x (Am x Ik/m)) (L^m^2n/k_mn/k x Ik/m) for m|k and k|mn
    AxI_scratch := rec(
        forTransposition := false,
        applicable := nt -> nt.hasTags() 
                            and _isALStoreTag(nt)
	                        and IsVecVec(nt.params) 
                            and IsPosInt(nt.getTag(1).size / nt.params[1].dims()[2])
                            and IsPosInt(nt.params[1].dims()[2]*nt.params[2] / nt.getTag(1).size),

        children := nt -> [[ let(t := nt.firstTag(),
            TTensorI(
                nt.params[1], 
                t.size / nt.params[1].dims()[2], 
                AVec, AVec
            ).withTags(Drop(nt.getTags(),1)).setWrap(_swrap(nt))
        ) ]],

        apply := (nt, c, cnt) -> let(
            k := nt.getTag(1).size,  
            m := nt.params[1].dims()[2], 
            n:= nt.params[2],

            DMAFence(Prm(fTensor(L(m^2*n/k, m), fId(k/m))) *
            Tensor(
                I(m*n/k),
                LSKernel(c[1], nt.params[1].normalizedArithCost() * k/m)
            ) *
            Prm(fTensor(L(m^2*n/k, m*n/k), fId(k/m))))
        )
    ),

    IxAL_scratch := rec(
        forTransposition := false,
        applicable := nt -> nt.hasTags()
                            and _isALStoreTag(nt)
                            and IsParVec(nt.params)
                            and IsPosInt(nt.getTag(1).size/ nt.params[1].dims()[2])
                            and IsPosInt(nt.params[1].dims()[2]*nt.params[2]/nt.getTag(1).size),
        children := nt -> let(
            k := nt.firstTag().size,
            m := nt.params[1].dims()[2],

            [[ TTensorI(
                nt.params[1], 
                k/m, 
                APar,AVec
            ).withTags(Drop(nt.getTags(),1)).setWrap(_swrap(nt)) 
        ]]),

        apply := (self, nt, c, cnt) >> let(
            k := nt.getTag(1).size,  
            m := nt.params[1].dims()[2], 
            n := nt.params[2],

            DMAFence(
                Tensor(
                    I(m*n/k),
                    LSKernel(c[1],nt.normalizedArithCost() * k/m)
                ) *
                Prm(fTensor(L(m^2*n/k,m*n/k),fId(k/m)))
            )
        ),
    ),

    L_IxA_scratch :=  rec(
        forTransposition := false,
        applicable := nt -> nt.hasTags() 
            and _isALStoreTag(nt) 
	        and IsVecPar(nt.params),

        children := nt -> [[ 
            TTensorI(
                nt.params[1], 
                nt.getTag(1).size / nt.params[1].dims()[2], 
                AVec, APar
            ).withTags(Drop(nt.getTags(),1)).setWrap(
                _swrap(nt)
            )
        ]],

        apply := (self, nt, c, cnt) >> let(
            k := nt.getTag(1).size,
            m := nt.params[1].dims()[2], 
            n:= nt.params[2],

            DMAFence( 
                Prm(fTensor(L(m^2*n/k, m*n/k), fId(k/m))
                ) * Tensor(LSKernel(
                    c[1], nt.normalizedArithCost() * k/m
                ), I(m*n/k))
            )
        )
     ),
#====================================================================
# SWP Rules - double buffering is applied
   IxA_scratch_swp := rec(
        forTransposition := false,
        applicable := nt -> nt.hasTags() 
            and _isALStoreTag(nt)
	        and IsParPar(nt.params)
            and IsPosInt(nt.getTag(1).size / (2 * nt.params[1].dims()[2]))
            and IsPosInt(2 * nt.params[1].dims()[2] * nt.params[2] / nt.getTag(1).size),

        children := nt -> [[ 
            TTensorI(
                nt.params[1],
                nt.getTag(1).size / (2 * nt.params[1].dims()[2]), 
                APar, APar
            ).withTags(Drop(nt.getTags(),1)).setWrap(
                _swrap(nt)
            )
        ]],
        apply := (nt, c, cnt) -> let(k := nt.getTag(1).size,  m := nt.params[1].dims()[2], n:= nt.params[2],
                DMAFence(Tensor(
                    I(2*m*n/k),
                    LSKernel(c[1], nt.params[1].normalizedArithCost() * k/(2*m))
                )))
    ),

    AxI_scratch_swp := rec(
        forTransposition := false,
        applicable := nt -> nt.hasTags() 
                            and _isALStoreTag(nt)
	                        and IsVecVec(nt.params) 
                            and IsPosInt(nt.getTag(1).size / (2*nt.params[1].dims()[2]))
                            and IsPosInt(2*nt.params[1].dims()[2]*nt.params[2] / nt.getTag(1).size),

        children := nt -> [[ let(t := nt.firstTag(),
            TTensorI(
                nt.params[1], 
                t.size / (2*nt.params[1].dims()[2]), 
                AVec, AVec
            ).withTags(Drop(nt.getTags(),1)).setWrap(_swrap(nt))
        ) ]],

        apply := (nt, c, cnt) -> let(
            k := nt.getTag(1).size,  
            m := nt.params[1].dims()[2], 
            n:= nt.params[2],

            DMAFence(Prm(fTensor(L(2*m^2*n/k, 2*m), fId(k/(2*m)))) *
            Tensor(
                I(2*m*n/k),
                LSKernel(c[1], nt.params[1].normalizedArithCost() * k/(2*m))
            ) *
            Prm(fTensor(L(2*m^2*n/k, 2*m*n/k), fId(k/(2*m)))))
        )
    ),

    IxAL_scratch_swp := rec(
        forTransposition := false,
        applicable := nt -> nt.hasTags()
                            and _isALStoreTag(nt)
                            and IsParVec(nt.params)
                            and IsPosInt(nt.getTag(1).size/ (2*nt.params[1].dims()[2]))
                            and IsPosInt(2*nt.params[1].dims()[2]*nt.params[2]/nt.getTag(1).size),
        children := nt -> let(
            k := nt.firstTag().size,
            m := nt.params[1].dims()[2],

            [[ TTensorI(
                nt.params[1], 
                k/(2*m), 
                APar,AVec
            ).withTags(Drop(nt.getTags(),1)).setWrap(_swrap(nt)) 
        ]]),

        apply := (self, nt, c, cnt) >> let(
            k := nt.getTag(1).size,  
            m := nt.params[1].dims()[2], 
            n := nt.params[2],

            DMAFence(
                Tensor(
                    I(2*m*n/k),
                    LSKernel(c[1],nt.normalizedArithCost() * k/(2*m))
                ) *
                Prm(fTensor(L(2*m^2*n/k,2*m*n/k),fId(k/(2*m))))
            )
        ),
    ),

    L_IxA_scratch_swp :=  rec(
        forTransposition := false,
        applicable := nt -> nt.hasTags() 
            and _isALStoreTag(nt) 
	        and IsVecPar(nt.params)
            and IsPosInt(nt.getTag(1).size/ (2*nt.params[1].dims()[2]))
            and IsPosInt(2*nt.params[1].dims()[2]*nt.params[2]/nt.getTag(1).size),

        children := nt -> [[ 
            TTensorI(
                nt.params[1], 
                nt.getTag(1).size / (2*nt.params[1].dims()[2]), 
                AVec, APar
            ).withTags(Drop(nt.getTags(),1)).setWrap(
                _swrap(nt)
            )
        ]],

        apply := (self, nt, c, cnt) >> let(
            k := nt.getTag(1).size,
            m := nt.params[1].dims()[2], 
            n:= nt.params[2],

            DMAFence( 
                Prm(fTensor(L(2*m^2*n/k, 2*m*n/k), fId(k/(2*m)))
                ) * Tensor(LSKernel(
                    c[1], nt.normalizedArithCost() * k/(2*m)
                ), I(2*m*n/k))
            )
        )
     )
));
