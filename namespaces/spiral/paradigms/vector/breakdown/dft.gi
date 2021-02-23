
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

NewRulesFor(DFT, rec(
    #F DFT_Base: DFT_2 = F_2
    #F
    DFT_tSPL_VBase := rec(
        switch := false,
        applicable := (t) -> Rows(t) = 2 and t.isTag(1,AVecReg),
        children := t -> [[ TTensorI(DFT(2), 1, AVec, AVec).withTags(t.getTags()) ]],
        apply := (t, C, Nonterms) -> C[1]
    ),
    #F DFT_Base: DFT_2 = F_2 for complex v/2-way vectorization
    #F
    DFT_tSPL_CxVBase := rec(
        switch := false,
        applicable := (t) -> Rows(t) = 2 and t.isTag(1,AVecRegCx),
        apply := (t, C, Nonterms) -> let(
	    v := t.getTags()[1].v, 
	    Cond(v = 1, VTensor(F(2), 1), 
		 # else
		 let(zeros    := Replicate(v-2, 0.0), 
		     one_one  := TVect(TReal, v).value( [1.0,  1.0] :: zeros ), 
		     one_mone := TVect(TReal, v).value( [1.0, -1.0] :: zeros ), 
		     blk      := VBlk([[ one_mone, one_one ]], v),
		     scat     := _VVStack([VTensor(I(1), v), VIxJ2(v)], v),
		     IxVScat_pc(1,2,2,0,v) * blk * scat * IxVGath_pc(1,2,2,0,v)))
        )
    ),

    #F DFT_Base: DFT_4 for complex v/2-way vectorization
    #F Tensor(F(2),I(4)) * RC(Diag(Tw1(4,2,-1))) * Tensor(I(2), F(2), I(2)) * Tensor(L(4,2),I(2))
    DFT_tSPL_CxVBase2 := rec(
        switch := false,
        applicable := (t) -> t.params[1] = 4 and t.firstTagIs(AVecRegCx) and t.firstTag().v>=4,
        children := nt -> [[TL(4, 2, 1, 1).withTags(nt.getTags()).setWrap(VWrapId)]],

        apply := (t, C, Nonterms) -> let(
	    v  := t.firstTag().v,
            vt := TVect(TReal, v),

            IxVScat_pc(1,4,4,0,v) * 
            VBlk( [[ vt.value([1, 1, -1,-1]), vt.one() ]], v) * 
            _VVStack([VTensor(I(1), v), VJxI(2, v)], v) *
            VDiag(Tw1(4,2,t.params[2]), v) * 
            VBlk( [[ vt.value([1, -1, 1,-1]), vt.one() ]], v) *
            _VVStack([VTensor(I(1), v), VIxJ2(v)], v) *
            C[1] *
            IxVGath_pc(1,4,4,0,v)
        )
    ),
    #F DFT_tSPL_CT_oddvloop
    #F DFT(mn, k)->Tensor(t2 * RI(m, mu), RI(n,nu)) * (Tensor(RI(mu,m), RI(nu,n)) * Diag(twf) * Tensor(RI(m,mu), RI(n,nu))) * L(mu*nu, mu) * Tensor(RI(nu,n) * t1, RI(mu, m))
    DFT_tSPL_CT_oddvloop := rec(
        switch  := false,
        maxSize := false,
        filter  := e->true,
        applicable := (self, nt) >> nt.params[1] > 2
            and (self.maxSize = false or nt.params[1] <= self.maxSize)
            and not IsPrime(nt.params[1])
            and (nt.firstTagIs(AVecReg) or nt.firstTagIs(AVecRegCx)),
        
        children := (self, nt) >> let(
            pv := nt.getTags(),
            v := pv[1].v,
            isa := pv[1].isa,
            _wrp := _nt -> When(Length(pv)=1, _nt.setWrap(VWrap(isa)), _nt.withTags(Drop(pv, 1)).setWrap(VWrap(isa))),
            Map2(Filtered(Filtered(DivisorPairs(nt.params[1]), a->IsOddInt(a[1])), self.filter), 
                (m,n) -> [ _wrp(DFT(m, nt.params[2] mod m)), 
                            _wrp(DFT(n, nt.params[2] mod n)),
                            TL(_roundup(n,v).v*v,v).withTags(pv).setWrap(VWrapId)
                         ])),
      
        apply := (t, C, Nonterms) -> let(
  	        v := t.firstTag().v,
            lprm := C[3],
            m := Rows(C[1]),
            n := Rows(C[2]),
            N := m * n,
            k := t.params[2],
            dftm := C[1],
            dftn := C[2],
            mu := _roundup(m,v).v,
            md := _rounddown(m,v).v,
            nu := _roundup(n,v).v,
            nd := _rounddown(n,v).v,
            f1 := fCompose(Tw1(N, n, k), fTensor(fId(m), fAdd(n,nd,0))),
            f2 := When(mu = m, 
                fCompose(Tw1(N, n, k), fTensor(fId(m), fAdd(n,n-nd,nd))), 
                fStretch(fCompose(Tw1(N, n, k), fTensor(fId(m), fAdd(n,n-nd,nd))), v, mu-m)),
            i1 := Ind(md/v),
            i2 := Ind(nd/v),
            #---------------------            
            stage1 := ApplyFunc(SUM, Concat([
                ISum(i1, 
                    VScat(fTensor(fAdd(mu/v,1,i1), fId(nu)),v) * 
                    lprm * VScat_zero(nu, n, v) * VTensor(dftn,v) * 
                    IxVGath_pc(n, m, v, v*i1, v)
                )],
                Cond(IsOddInt(m), 
                    [VScat(fTensor(fAdd(mu/v,1,md/v), fId(nu)),v) * 
                    lprm * VScat_zero(nu, n, v) * VTensor(dftn,v) * 
                    IxVGath_pc(n, m, m-md, md, v)],
                    [])
                )
            ),
            #---------------------            
            stage2 := ApplyFunc(SUM, Concat([
                ISum(i2, 
                    IxVScat_pc(m, n, v, v*i2, v) * 
                    VTensor(dftm, v) * VDiag(fPrecompute(fCompose(f1, fTensor(fId(m), fBase(i2), fId(v)))), v) * VGath(fAdd(mu, m, 0), v) * 
                    VGath(fTensor(fId(mu), fAdd(nu/v, 1,i2)),v)
                )],
                Cond(IsOddInt(n),
                    [IxVScat_pc(m, n, n-nd, nd, v) * 
                    VTensor(dftm, v) * VDiag(f2, v) * VGath(fAdd(mu, m, 0), v) * 
                    VGath(fTensor(fId(mu), fAdd(nu/v, 1,nd/v)),v)],
                    [])
                )
            ),
#            #---------------------            
#            stage1 := SUM(
#                ISum(i1, 
#                    VScat(fTensor(fAdd(mu/v,1,i1), fId(nu)),v) * 
#                    lprm * VScat_zero(nu, n, v) * VTensor(dftn,v) * 
#                    IxVGath_pc(n, m, v, v*i1, v)
#                ),
#                VScat(fTensor(fAdd(mu/v,1,md/v), fId(nu)),v) * 
#                lprm * VScat_zero(nu, n, v) * VTensor(dftn,v) * 
#                IxVGath_pc(n, m, m-md, md, v)
#            ),
#            #---------------------            
#            stage2 := SUM(
#                ISum(i2, 
#                    IxVScat_pc(m, n, v, v*i2, v) * 
#                    VTensor(dftm, v) * VDiag(fPrecompute(fCompose(f1, fTensor(fId(m), fBase(i2), fId(v)))), v) * VGath(fAdd(mu, m, 0), v) * 
#                    VGath(fTensor(fId(mu), fAdd(nu/v, 1,i2)),v)
#                ),
#                IxVScat_pc(m, n, n-nd, nd, v) * 
#                VTensor(dftm, v) * VDiag(f2, v) * VGath(fAdd(mu, m, 0), v) * 
#                VGath(fTensor(fId(mu), fAdd(nu/v, 1,nd/v)),v)
#            ),
            #---------------------            
            stage2 * stage1
        )
    ),

#    #F DFT_tSPL_CT_evenoddvloop
#    #F DFT(mn, k)->Tensor(t2 * RI(m, mu), RI(n,nu)) * (Tensor(RI(mu,m), RI(nu,n)) * Diag(twf) * Tensor(RI(m,mu), RI(n,nu))) * L(mu*nu, mu) * Tensor(RI(nu,n) * t1, RI(mu, m))
#    DFT_tSPL_CT_evenoddvloop := rec(
#        switch  := false,
#        maxSize := false,
#        filter  := e->true,
#        applicable := (self, nt) >> nt.params[1] > 2
#            and (self.maxSize = false or nt.params[1] <= self.maxSize)
#            and not IsPrime(nt.params[1])
#            and IsOddInt(nt.params[1]/2)
#            and IsEvenInt(nt.params[1])
#            and (nt.firstTagIs(AVecReg) or nt.firstTagIs(AVecRegCx)),
#        
#        children := (self, nt) >> let(
#            pv := nt.getTags(),
#            v := pv[1].v,
#            isa := pv[1].isa,
#            _wrp := _nt -> When(Length(pv)=1, _nt.setWrap(VWrap(isa)), _nt.withTags(Drop(pv, 1)).setWrap(VWrap(isa))),
#            Map2(Filtered(DivisorPairs(nt.params[1]), self.filter), 
#                (m,n) -> [ _wrp(DFT(m, nt.params[2] mod m)), 
#                            _wrp(DFT(n, nt.params[2] mod n)),
#                            TL(_roundup(n,v).v*v,v).withTags(pv).setWrap(VWrapId)
#                         ])),
#      
#        apply := (t, C, Nonterms) -> let(
#  	        v := t.firstTag().v,
#            lprm := C[3],
#            m := Rows(C[1]),
#            n := Rows(C[2]),
#            N := m * n,
#            k := t.params[2],
#            dftm := C[1],
#            dftn := C[2],
#            mu := _roundup(m,2).v,
#            md := _rounddown(m,2).v,
#            nu := _roundup(n,2).v,
#            nd := _rounddown(n,2).v,
#            f1 := fCompose(Tw1(N, n, k), fTensor(fId(m), fAdd(n,nd,0))),
#            f2 := fStretch(fCompose(Tw1(N, n, k), fTensor(fId(m), fAdd(n,n-nd,nd))), v, mu-m),
#            i1 := Ind(md/v),
#            i2 := Ind(nd/v),
#            #---------------------            
#            stage1 := ApplyFunc(SUM, Concat([
#                ISum(i1, 
#                    VScat(fTensor(fAdd(mu/v,1,i1), fId(nu)),v) * 
#                    lprm * VScat_zero(nu, n, v) * VTensor(dftn,v) * 
#                    IxVGath_pc(n, m, v, v*i1, v)
#                )],
#                Cond(IsOddInt(m), 
#                    [VScat(fTensor(fAdd(mu/v,1,md/v), fId(nu)),v) * 
#                    lprm * VScat_zero(nu, n, v) * VTensor(dftn,v) * 
#                    IxVGath_pc(n, m, m-md, md, v)],
#                    [])
#                )
#            ),
#            #---------------------            
#            stage2 := ApplyFunc(SUM, Concat([
#                ISum(i2, 
#                    IxVScat_pc(m, n, v, v*i2, v) * 
#                    VTensor(dftm, v) * VDiag(fPrecompute(fCompose(f1, fTensor(fId(m), fBase(i2), fId(v)))), v) * VGath(fAdd(mu, m, 0), v) * 
#                    VGath(fTensor(fId(mu), fAdd(nu/v, 1,i2)),v)
#                )],
#                Cond(IsOddInt(n),
#                    [IxVScat_pc(m, n, n-nd, nd, v) * 
#                    VTensor(dftm, v) * VDiag(f2, v) * VGath(fAdd(mu, m, 0), v) * 
#                    VGath(fTensor(fId(mu), fAdd(nu/v, 1,nd/v)),v)],
#                    [])
#                )
#            ),
#            #---------------------            
#            stage2 * stage1
#        )
#    )
));
