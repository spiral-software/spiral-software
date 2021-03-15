
# Copyright (c) 2018-2020, Carnegie Mellon University
# See LICENSE for details


Class(PrunedPRDFT, TaggedNonTerminal, rec(
    abbrevs := [
        (n,blk,pat) -> Checked(IsPosIntSym(n), IsPosIntSym(blk), IsList(pat),
            AnySyms(n,blk) or (IsInt(_unwrap(n)/_unwrap(blk)) and ForAll(pat, i->IsInt(i) and i < n/blk)),
            [_unwrap(n), 1, _unwrap(blk), pat]),
        (n,k,blk,pat) -> Checked(IsPosIntSym(n), IsIntSym(k), IsPosIntSym(blk), IsList(pat),
            AnySyms(n,k) or Gcd(_unwrap(n),_unwrap(k))=1,
            AnySyms(n,blk) or (IsInt(_unwrap(n)/_unwrap(blk)) and ForAll(pat, i->IsInt(i) and i < n/blk)),
            [_unwrap(n), When(AnySyms(n,k), k, k mod _unwrap(n)), _unwrap(blk), pat])
        ],

    dims := self >> let(
        size := let(n:=self.params[1], 2*(Int(n/2)+1)),
        d := [size, self.params[3]*Length(self.params[4])],
        When(self.transposed, [d[2], d[1]], d)
    ),

    terminate := self >> let(
        size := self.params[1], blk := self.params[3], pat := self.params[4],
        res := Compose(PRDFT(size, self.params[2]),
               Tensor(Mat(List(pat, i->BasisVec(size/blk, i))).transpose(), I(blk))).terminate(),
        When(self.transposed, res.transpose(), res)
    ),

    isReal    := True,
    normalizedArithCost := self >> let(n := self.params[1], IntDouble(2.5 * n * d_log(n) / d_log(2))),
    TType := TReal
));

_pruned_PRF12_CT_Children := (N,k,PRFt,DFTt,PRFtp,PRF1) -> Map2(DivisorPairs(N),
    (m,n) -> When(IsEvenInt(n),
	[ PRFt(m,k), DFTt(m,k), PRFtp(m,k) ] :: PRF1(m,n,k),
	[ PRFt(m,k), DFTt(m,k) ] :: PRF1(m,n,k))
);

_pruned_PRF12_CT_Stage1 := (N,k,C,Conj,Tw) ->
    let(m:=Cols(C[1]),
        n:=N/m,
        Nf:=Int(N/2),
        nf:=Int(n/2),
        nc:=Int((n+1)/2),
        mf:=Int(m/2),
        mc:=Int((m+1)/2),
        j:=Ind(nc-1),
        SUM(
            RC(Scat(H(Nf+1,mf+1,0,n))) * C[1] * Gath(H(2*(nf+1)*m, m, 0, 2*(nf+1))),
            When(nc=1, [],
                ISum(j,
                    RC(Scat(BH(Nf+1,N,m,j+1,n))) *
                    Conj * RC(C[2]) * Tw(j) *
                    RC(Gath(H((nf+1)*m, m, j+1, nf+1)))
                )
            ),
            When(IsOddInt(n), [],
                RC(Scat(H(Nf+1,mc,nf,n))) * C[3] * Gath(H(2*(nf+1)*m, m, 2*nf, 2*(nf+1)))
            )
        )
    );


_pruned_IPRF12_CT_Stage2 := (N,k,C,Conj,Tw) ->
    let(m:=Rows(C[1]),
        n:=N/m,
        Nf:=Int(N/2),
        nf:=Int(n/2),
        nc:=Int((n+1)/2),
        mf:=Int(m/2),
        mc:=Int((m+1)/2),
        j:=Ind(nc-1),
        SUM(
            Scat(H(2*(nf+1)*m, m, 0, 2*(nf+1))) * C[1] * RC(Gath(H(Nf+1,mf+1,0,n))),
        	When(nc=1, [],
                ISum(j,
                    RC(Scat(H((nf+1)*m, m, j+1, nf+1))) *
                    Tw(j) * RC(C[2]) * Conj *
    	            RC(Gath(BH(Nf+1,N,m,j+1,n)))
                )
            ),
        	When(IsOddInt(n), [],
    	       Scat(H(2*(nf+1)*m, m, 2*nf, 2*(nf+1))) * C[3] * RC(Gath(H(Nf+1,mc,nf,n))))
        )
    );


NewRulesFor(PrunedPRDFT, rec(
    PrunedPRDFT_base := rec(
       forTransposition := false,
       maxSize := 256,
       applicable := (self, nt) >> not nt.hasTags() and nt.params[1] <= self.maxSize and nt.params[3] = 1,
       children := nt -> [[PRDFT(nt.params[1], nt.params[2])]],
       apply := (nt, C, cnt) -> let(size := nt.params[1], blk := nt.params[3], pat := nt.params[4],
            C[1] * Tensor(Mat(List(pat, i->BasisVec(size/blk, i))).transpose(), I(blk)).terminate())
    ),
    PrunedPRDFT_DFT := rec(
       forTransposition := false,
       applicable := (self, nt) >> nt.params[1] = nt.params[3] and nt.params[4] = [0],
       children := nt -> [[ PRDFT(nt.params[1], nt.params[2]).withTags(nt.getTags()) ]],
       apply := (nt, C, cnt) -> C[1]
    ),
    PrunedPRDFT_CT_rec_block := rec(
       forTransposition := true,
       fPrecompute := f->fPrecompute(f),
       applicable := (self, nt) >> nt.params[1] > 2
            and not nt.hasTags()
            and not IsPrime(nt.params[1])
            and nt.params[3] = 1
            and Last(nt.params[4])+1-nt.params[4][1] = Length(Set(nt.params[4]))
            and ForAny(Map2(DivisorPairs(nt.params[1]), (m,n)->_ctpr_applicable(m, n, nt.params[4])), i->i),
       children := (nt) -> Filtered(_pruned_PRF12_CT_Children(nt.params[1],nt.params[2],PRDFT1, DFT1, PRDFT3,
                                        (m, n, k)->_pruned_children(m, n, nt.params[4], (r, sp) -> PrunedPRDFT(r, k mod n, nt.params[3], sp))),
                              lst -> ForAll(Filtered(lst, i->ObjId(i) = PrunedPRDFT), j->j.params[4] <> [])),
       apply := (self, nt, C, cnt) >> let(N:=nt.params[1], k:=nt.params[2], m:=Cols(C[1]), n := N/m,
            _pruned_PRF12_CT_Stage1(N, k, C, Diag(BHD(m,1,-1)), j->RC(Diag(self.fPrecompute(Twid(N,m,k,0,0,j+1))))) *
            _build_stack(m, n, nt.params[4], let(s1len := When(IsEvenInt(n), 3, 2), C{[s1len+1..Length(C)]}), nt.dims()[2], Gath, IterVStack, VStack, (a,b)->a*b))
    )
));






Class(PrunedIPRDFT, TaggedNonTerminal, rec(
    abbrevs := [
        (n,blk,pat) -> Checked(IsPosIntSym(n), IsPosIntSym(blk), IsList(pat),
            AnySyms(n,blk) or (IsInt(_unwrap(n)/_unwrap(blk)) and ForAll(pat, i->IsInt(i) and i < n/blk)),
            [_unwrap(n), 1, _unwrap(blk), pat]),
        (n,k,blk,pat) -> Checked(IsPosIntSym(n), IsIntSym(k), IsPosIntSym(blk), IsList(pat),
            AnySyms(n,k) or Gcd(_unwrap(n),_unwrap(k))=1,
            AnySyms(n,blk) or (IsInt(_unwrap(n)/_unwrap(blk)) and ForAll(pat, i->IsInt(i) and i < n/blk)),
            [_unwrap(n), When(AnySyms(n,k), k, k mod _unwrap(n)), _unwrap(blk), pat])
        ],

    dims := self >> let(
        size := let(n:=self.params[1], 2*(Int(n/2)+1)),
        d := [self.params[3]*Length(self.params[4]), size],
        When(self.transposed, [d[2], d[1]], d)
    ),

    terminate := self >> let(
        size := self.params[1], blk := self.params[3], pat := self.params[4],
        res := Compose(Tensor(Mat(List(pat, i->BasisVec(size/blk, i))), I(blk)),
                       IPRDFT(size, self.params[2])).terminate(),
        When(self.transposed, res.transpose(), res)
    ),

    isReal    := True,
    normalizedArithCost := self >> let(n := self.params[1], IntDouble(2.5 * n * d_log(n) / d_log(2))),
    TType := TReal
));


NewRulesFor(PrunedIPRDFT, rec(
    PrunedIPRDFT_base := rec(
       forTransposition := false,
       maxSize := 256,
       applicable := (self, nt) >> not nt.hasTags() and nt.params[1] <= self.maxSize and nt.params[3] = 1,
       children := nt -> [[IPRDFT(nt.params[1], nt.params[2])]],
       apply := (nt, C, cnt) -> let(size := nt.params[1], blk := nt.params[3], pat := nt.params[4],
            Tensor(Mat(List(pat, i->BasisVec(size/blk, i))), I(blk)).terminate() * C[1] )
    ),
    PrunedIPRDFT_DFT := rec(
       forTransposition := false,
       applicable := (self, nt) >> nt.params[1] = nt.params[3] and nt.params[4] = [0],
       children := nt -> [[ IPRDFT(nt.params[1], nt.params[2]).withTags(nt.getTags()) ]],
       apply := (nt, C, cnt) -> C[1]
    ),
    PrunedIPRDFT_CT_rec_block := rec(
       forTransposition := true,
       applicable := (self, nt) >> nt.params[1] > 2
            and not nt.hasTags()
            and not IsPrime(nt.params[1])
            and nt.params[3] = 1
            and Last(nt.params[4])+1-nt.params[4][1] = Length(Set(nt.params[4]))
            and ForAny(Map2(DivisorPairs(nt.params[1]), (m,n)->_ctpr_applicable(m, n, nt.params[4])), i->i),
       children := (nt) -> Filtered(_pruned_PRF12_CT_Children(nt.params[1],nt.params[2], IPRDFT1, DFT1, IPRDFT2,
                                        (m, n, k)->_pruned_children(m, n, nt.params[4], (r, sp) -> PrunedIPRDFT(r, k mod n, nt.params[3], sp))),
                              lst -> ForAll(Filtered(lst, i->ObjId(i) = PrunedIPRDFT), j->j.params[4] <> [])),
       apply := (nt, C, cnt) -> let(N:=nt.params[1], k:=nt.params[2], m:=Rows(C[1]), n := N/m,
            _build_stack(m, n, nt.params[4], let(s1len := When(IsEvenInt(n), 3, 2), C{[s1len+1..Length(C)]}), nt.dims()[1], Scat, IterHStack1, HStack1, (a,b)->b*a) *
            _pruned_IPRF12_CT_Stage2(N, k, C, Diag(BHD(m,1,-1)), j->RC(Diag(fPrecompute(Twid(N,m,k,0,0,j+1)))))
       )
    ),
));

Class(IOPrunedConv, TaggedNonTerminal, rec(
    abbrevs := [
        (n,h,oblk,opat,iblk,ipat) -> Checked(IsPosIntSym(n), IsPosIntSym(oblk), IsList(opat), IsPosIntSym(iblk), IsList(ipat),
            AnySyms(n,iblk,oblk) or (IsInt(_unwrap(n)/_unwrap(iblk)) and IsInt(_unwrap(n)/_unwrap(oblk)) and
                ForAll(ipat, i->IsInt(i) and i < n/iblk) and ForAll(opat, i->IsInt(i) and i < n/oblk)),
            [_unwrap(n), h, _unwrap(oblk), opat, _unwrap(iblk), ipat, false]),
        (n,h,oblk,opat,iblk,ipat,isFreqData) -> Checked(IsPosIntSym(n), IsPosIntSym(oblk), IsList(opat), IsPosIntSym(iblk), IsList(ipat),
            AnySyms(n,iblk,oblk) or (IsInt(_unwrap(n)/_unwrap(iblk)) and IsInt(_unwrap(n)/_unwrap(oblk)) and
                ForAll(ipat, i->IsInt(i) and i < n/iblk) and ForAll(opat, i->IsInt(i) and i < n/oblk)),
            [_unwrap(n), h, _unwrap(oblk), opat, _unwrap(iblk), ipat, isFreqData])
        ],
    dims      := self >> [self.params[3]*Length(self.params[4]), self.params[5]*Length(self.params[6])],
    terminate := self >> let(size := self.params[1],
        taps := When(self.params[7], MatSPL(DFT(self.params[1], -1)) * 
               Cond(IsList(self.params[2]), self.params[2], IsLambda(self.params[2]), self.params[2].tolist(), IsBound(self.params[2].list), self.params[2].list), 
           self.params[2]),
        oblk := self.params[3], opat := self.params[4], iblk := self.params[5], ipat := self.params[6],
        Tensor(Mat(List(opat, i->BasisVec(size/oblk, i))), I(oblk)).terminate() *
        spiral.transforms.filtering.Circulant(size, taps, -size).terminate() *
        Tensor(Mat(List(ipat, i->BasisVec(size/iblk, i))).transpose(), I(iblk)).terminate()),
    isReal    := self >> false,
    normalizedArithCost := self >> let(n := self.params[1], IntDouble(5 * n * d_log(n) / d_log(2))),
    TType := TComplex,
    hashAs := self >> ApplyFunc(ObjId(self), [self.params[1], fUnk(self.params[2].range(), self.params[2].domain())]::Drop(self.params, 2))
));

NewRulesFor(IOPrunedConv, rec(
    IOPrunedConv_PrunedDFT_IPrunedDFT := rec(
       forTransposition := false,
       applicable :=  (self, nt) >> not nt.hasTags(),
       children := nt -> [[ PrunedIDFT(nt.params[1], 1, nt.params[3], nt.params[4]),
                            PrunedDFT(nt.params[1], -1, nt.params[5], nt.params[6]) ]],
       apply := (nt, C, cnt) -> Cond(ObjId(nt.params[2]) = var, # a variable is frequency domain data -> FData
                                    Checked(nt.params[7], C[1] * Diag(FData(nt.params[2])) * C[2]), # if data is provided it needs to be frequency domain
                                    ObjId(nt.params[2]) = Lambda,  # data is inside a larger array
                                    Checked(nt.params[7], C[1] * Diag(nt.params[2]) * C[2]),
                                    nt.params[7],   # if params[7] is true then data is frequency domain data
                                    C[1] * Diag(When(IsList(nt.params[2]), FList(TComplex, nt.params[2]), nt.params[2])) * C[2],
                                    let(cxfftdiag := 1/nt.params[1]*ComplexFFT(List(nt.params[2].tolist(), i->ComplexAny(_unwrap(i)))),
                                        C[1] * Diag(FData(cxfftdiag)) * C[2]))
    ),
));


#===========================================

Class(IOPrunedRConv, TaggedNonTerminal, rec(
    abbrevs := [
        (n,h,oblk,opat,iblk,ipat) -> Checked(IsPosIntSym(n), IsPosIntSym(oblk), IsList(opat), IsPosIntSym(iblk), IsList(ipat),
            AnySyms(n,iblk,oblk) or (IsInt(_unwrap(n)/_unwrap(iblk)) and IsInt(_unwrap(n)/_unwrap(oblk)) and
                ForAll(ipat, i->IsInt(i) and i < n/iblk) and ForAll(opat, i->IsInt(i) and i < n/oblk)),
            [_unwrap(n), h, _unwrap(oblk), opat, _unwrap(iblk), ipat, false]),
        (n,h,oblk,opat,iblk,ipat,isFreqData) -> Checked(IsPosIntSym(n), IsPosIntSym(oblk), IsList(opat), IsPosIntSym(iblk), IsList(ipat),
            AnySyms(n,iblk,oblk) or (IsInt(_unwrap(n)/_unwrap(iblk)) and IsInt(_unwrap(n)/_unwrap(oblk)) and
                ForAll(ipat, i->IsInt(i) and i < n/iblk) and ForAll(opat, i->IsInt(i) and i < n/oblk)),
            [_unwrap(n), h, _unwrap(oblk), opat, _unwrap(iblk), ipat, isFreqData])
        ],
    dims      := self >> [self.params[3]*Length(self.params[4]), self.params[5]*Length(self.params[6])],
    terminate := self >> let(size := self.params[1],
        taps := When(self.params[7], MatSPL(IPRDFT(self.params[1], -1)) * When(IsList(self.params[2]), self.params[2], self.params[2].list), self.params[2]),
        oblk := self.params[3], opat := self.params[4], iblk := self.params[5], ipat := self.params[6],
        Tensor(Mat(List(opat, i->BasisVec(size/oblk, i))), I(oblk)).terminate() *
        spiral.transforms.filtering.Circulant(size, taps, -size).terminate() *
        Tensor(Mat(List(ipat, i->BasisVec(size/iblk, i))).transpose(), I(iblk)).terminate()),
    isReal    := self >> true,
    normalizedArithCost := self >> let(n := self.params[1], IntDouble(5 * n * d_log(n) / d_log(2))),
    TType := TReal
));

NewRulesFor(IOPrunedRConv, rec(
    IOPrunedRConv_PrunedIPRDFT_PrunedPRDFT := rec(
       forTransposition := false,
       applicable :=  (self, nt) >> not nt.hasTags(),
       children := nt -> [[ PrunedIPRDFT(nt.params[1], 1, nt.params[3], nt.params[4]),
                            PrunedPRDFT(nt.params[1], -1, nt.params[5], nt.params[6]) ]],
       apply := (nt, C, cnt) -> Cond(ObjId(nt.params[2]) = var, # a variable is frequency domain data -> FData
                                Checked(nt.params[7], C[1] * RCDiag(FData(nt.params[2])) * C[2]), # if data is provided it needs to be frequency domain
                                nt.params[7],   # if params[7] is true then data is frequency domain data
                                C[1] *  RCDiag(When(IsList(nt.params[2]), FList(TReal, nt.params[2]), nt.params[2])) * C[2],
                                let(cxfftdiag := 1/nt.params[1]*ComplexFFT(List(nt.params[2].tolist(), i->ComplexAny(_unwrap(i)))),
                                    rcdiag := cxfftdiag{[1..Rows(PRDFT(nt.params[1], -1))/2]},
                                    rlist := Flat(List(rcdiag, i->[ReComplex(_unwrap(i)), ImComplex(_unwrap(i))])),
                                    C[1] * RCDiag(FList(TReal, rlist)) * C[2]))
    ),
));

_complexify := l -> List([1..Length(l)/2], i->l[2*i-1] + E(4) * l[2*i]);

#=========================================================================================================================================================
Declare(IOPrunedMDRConv);
Class(IOPrunedMDRConv, TaggedNonTerminal, rec(
    abbrevs := [
        (n,h,oblk,opat,iblk,ipat) -> Checked(ForAll(n, IsPosIntSym), IsPosIntSym(oblk), ForAll(opat, IsList), IsPosIntSym(iblk), ForAll(ipat, IsList),
            [_unwrap(n), h, _unwrap(oblk), opat, _unwrap(iblk), ipat, false]),
        (n,h,oblk,opat,iblk,ipat,isFreqData) -> Checked(ForAll(n, IsPosIntSym), IsPosIntSym(oblk), ForAll(opat, IsList), IsPosIntSym(iblk), ForAll(ipat, IsList),
            [_unwrap(n), h, _unwrap(oblk), opat, _unwrap(iblk), ipat, isFreqData])
        ],
    dims      := self >> [self.params[3]*Product(self.params[4], i->Length(i)), self.params[5]*Product(self.params[6], i->Length(i))],
    isReal    := self >> true,
    normalizedArithCost := self >> let(n := self.params[1], IntDouble(5 * n * d_log(n) / d_log(2))),
    TType := TReal,
    terminate := self >> When(self.params[7],
                            let(nlist := self.params[1],
                                n := nlist[1],
                                nfreq := n/2+1,
                                nrem := Drop(nlist, 1),
                                idft := Tensor(IPRDFT(n, -1), I(Product(nrem))) *
                                    Tensor(I(nfreq), L(2*Product(nrem), 2)) * Tensor(I(nfreq), RC(MDDFT(nrem, -1))),
                                tlist := MatSPL(idft) * 
                                    Cond(IsList(self.params[2]), self.params[2], IsLambda(self.params[2]), self.params[2].tolist(), IsBound(self.params[2].list), self.params[2].list),
                                IOPrunedMDRConv(self.params[1], FList(TReal, tlist), self.params[3], self.params[4],
                                    self.params[5], self.params[6], false).terminate()
                            ),
                            let(nlist := self.params[1],
                                oblk := self.params[3], opat_md := self.params[4],
                                iblk := self.params[5], ipat_md := self.params[6],
                                scat3d := Tensor(List(Zip2(ipat_md, nlist), j->let(ipat := j[1], size := j[2],
                                    Tensor(Mat(List(ipat, i->BasisVec(size/iblk, i))).transpose(), I(iblk)).terminate()))::[Mat([[1], [0]])]),
                                gath3d := Tensor(List(Zip2(opat_md, nlist), j->let(opat := j[1], size := j[2],
                                    Tensor(Mat(List(opat, i->BasisVec(size/oblk, i))), I(oblk)).terminate()))::[Mat([[1, 0]])]),

                                dft3dr := RC(MDDFT(nlist, -1)),
                                idft3dr := RC(MDDFT(nlist, 1)),
                                gfd := List(1/Product(nlist) * MatSPL(MDDFT(nlist, 1)) * self.params[2].list, i->ComplexAny(_unwrap(i))),
                                gdiagr := RC(Diag(gfd)),
                                t := gath3d * idft3dr * gdiagr * dft3dr * scat3d,
                                t.terminate()
                            )
                        ),
    hashAs := self >> ApplyFunc(ObjId(self), [self.params[1], fUnk(self.params[2].range(), self.params[2].domain())]::Drop(self.params, 2))
));

NewRulesFor(IOPrunedMDRConv, rec(
    IOPrunedMDRConv_time2freq := rec(
       forTransposition := false,
       applicable :=  (self, nt) >> not nt.hasTags() and not nt.params[7],
       children := nt -> let(htime := nt.params[2].list,
                            nlist := nt.params[1],
                            dft := 1/Product(nlist) * MatSPL(MDDFT(nlist, 1)),
                            nfreq := nlist[1]/2+1,
                            hfreq := dft*htime,
                            hfreqc := List(hfreq, ComplexAny),
                            hfreqce := hfreqc{[1..nfreq*Product(Drop(nlist,1))]},
                            hfreqr := Flat(List(hfreqce, i->[ReComplex(i), ImComplex(i)])),
                            [[IOPrunedMDRConv(nt.params[1], FList(TReal, hfreqr), nt.params[3], nt.params[4], nt.params[5], nt.params[6], true)]]
                        ),
       apply := (nt, C, cnt) -> C[1]
    ),

    ## GPU/TITAN V Hockney algotithm variant
    ## 2-trip, 5-step, ZYX ====================================================
    IOPrunedMDRConv_3D_2trip_zyx_freqdata := rec(
       forTransposition := false,
       applicable :=  (self, nt) >> not nt.hasTags() and Length(nt.params[1]) = 3 and nt.params[7],
       children := nt -> let(nlist := nt.params[1],
                            diag := nt.params[2],
                            oblk := nt.params[3],
                            opats := nt.params[4],
                            iblk := nt.params[5],
                            ipats := nt.params[6],
                            nfreq := nlist[1]/2+1,
                            i := Ind(nfreq*nlist[2]),
                            hfunc := Cond(ObjId(diag) = Lambda,
                                let(j := Ind(nlist[3]),
                                    # Lambda(j, cxpack(diag.at(2*(j + i*nlist[3])), diag.at(2*(j + i*nlist[3])+1)))
                                    pos := i +j*nfreq*nlist[2],
                                    Lambda(j, cxpack(diag.at(2*pos), diag.at(2*pos+1)))
                                ),
                                ObjId(diag) = fUnk,
                                fUnk(TComplex, nlist[3]),
                                let(list := nt.params[2].list,  # here we assume FList(TReal, [...])
                                    clist := List([1..Length(list)/2], i->Cplx(list[2*i-1], list[2*i])),
                                    fc := FList(TComplex, clist),
                                    gf := fTensor(fBase(i), fId(nlist[3])),
                                    fCompose(fc, gf)
                                )
                            ),
                            [[ PrunedPRDFT(nlist[1], -1, iblk, ipats[1]),  # stage 1: PRDFT z
                                PrunedDFT(nlist[2], -1, iblk, ipats[2]),    # stage 2: DFT y
                                IOPrunedConv(nlist[3], hfunc, oblk, opats[3], iblk, ipats[3], true), # stage 3+4+5: complex conv in x
                                PrunedIDFT(nlist[2], 1, oblk, opats[2]), # stage 6: iDFT in y
                                PrunedIPRDFT(nlist[1], 1, oblk, opats[1]),   # stage 7: iPRDFT in z
                                InfoNt(i)
                            ]]),

       apply := (nt, C, cnt) -> let(prdft1d := C[1],
                                    pdft1d := C[2],
                                    iopconv := C[3],
                                    ipdft1d := C[4],
                                    iprdft1d := C[5],
                                    i := cnt[6].params[1],
                                    nlist := nt.params[1],
                                    n1 := nlist[1],
                                    nfreq := nlist[1]/2+1,
                                    n2 := nlist[2],
                                    n3 := nlist[3],
                                    oblk := nt.params[3],
                                    opats := nt.params[4],
                                    iblk := nt.params[5],
                                    ipats := nt.params[6],
                                    ns1 := iblk * Length(ipats[1]),
                                    ns2 := iblk * Length(ipats[2]),
                                    ns3 := iblk * Length(ipats[3]),
                                    nd1 := oblk * Length(opats[1]),
                                    nd2 := oblk * Length(opats[2]),
                                    nd3 := oblk * Length(opats[3]),
                                    stage1 := L(2*nfreq*ns3*ns2, ns3) * Tensor(I(ns2), Tensor(L(2*nfreq, 2) * prdft1d, I(ns3))) * Tensor(L(ns2*ns1, ns2), I(ns3)),
                                    stage2 := Tensor(I(ns3), Tensor(RC(pdft1d), I(nfreq))),
                                    pp := Tensor(L(ns3*n2*nfreq, n2*nfreq), I(2)) * Tensor(I(ns3), L(2*nfreq*n2, nfreq)),
                                    ppi := Tensor(I(nd3), L(2*nfreq*n2, 2*n2)) * Tensor(L(nd3*n2*nfreq, nd3), I(2)),
                                    stage543 := ppi * IDirSum(i, RC(iopconv)) * pp,
                                    stage76 := Tensor(L(nd2*nd1, nd1), I(nd3)) * Grp(Tensor((Tensor(I(nd2), iprdft1d * L(2*nfreq, nfreq)) *
                                        Tensor(RC(ipdft1d), I(nfreq))), I(nd3)) * L(2*nfreq*nd3*n2, 2*nfreq*n2)),
                                    conv3dr := stage76 * stage543 * stage2 * stage1,
                                    conv3dr
                            ),
    ),

    ## X dimension last (convolution)
    ## 5-step, ZYX ====================================================
    IOPrunedMDRConv_3D_5step_zyx_freqdata := rec(
       forTransposition := false,
       applicable :=  (self, nt) >> not nt.hasTags() and Length(nt.params[1]) = 3 and nt.params[7],
       children := nt -> let(nlist := nt.params[1],
                            diag := nt.params[2],
                            oblk := nt.params[3],
                            opats := nt.params[4],
                            iblk := nt.params[5],
                            ipats := nt.params[6],
                            nfreq := nlist[1]/2+1,
                            i := Ind(nfreq*nlist[2]),
                            hfunc := Cond(ObjId(diag) = Lambda,
                                let(j := Ind(nlist[3]),
                                    Lambda(j, cxpack(diag.at(2*(j + i*nlist[3])), diag.at(2*(j + i*nlist[3])+1)))
                                ),
                                ObjId(diag) = fUnk,
                                fUnk(TComplex, nlist[3]),
                                let(list := nt.params[2].list,  # here we assume FList(TReal, [...])
                                    clist := List([1..Length(list)/2], i->Cplx(list[2*i-1], list[2*i])),
                                    fc := FList(TComplex, clist),
                                    gf := fTensor(fBase(i), fId(nlist[3])),
                                    fCompose(fc, gf)
                                )
                            ),
                            [[ PrunedPRDFT(nlist[1], -1, iblk, ipats[1]),  # stage 1: PRDFT z
                                PrunedDFT(nlist[2], -1, iblk, ipats[2]),    # stage 2: DFT y
                                IOPrunedConv(nlist[3], hfunc, oblk, opats[3], iblk, ipats[3], true), # stage 3+4+5: complex conv in x
                                PrunedIDFT(nlist[2], 1, oblk, opats[2]), # stage 6: iDFT in y
                                PrunedIPRDFT(nlist[1], 1, oblk, opats[1]),   # stage 7: iPRDFT in z
                                InfoNt(i)
                            ]]),

       apply := (nt, C, cnt) -> let(prdft1d := Grp(C[1]),
                                    pdft1d := Grp(C[2]),
                                    iopconv := Grp(C[3]),
                                    ipdft1d := Grp(C[4]),
                                    iprdft1d := Grp(C[5]),
                                    i := cnt[6].params[1],
                                    nlist := nt.params[1],
                                    n1 := nlist[1],
                                    nfreq := nlist[1]/2+1,
                                    n2 := nlist[2],
                                    n3 := nlist[3],
                                    oblk := nt.params[3],
                                    opats := nt.params[4],
                                    iblk := nt.params[5],
                                    ipats := nt.params[6],
                                    ns1 := iblk * Length(ipats[1]),
                                    ns2 := iblk * Length(ipats[2]),
                                    ns3 := iblk * Length(ipats[3]),
                                    nd1 := oblk * Length(opats[1]),
                                    nd2 := oblk * Length(opats[2]),
                                    nd3 := oblk * Length(opats[3]),
                                    stage1 := Tensor(I(nfreq), L(2*ns2*ns3, ns2*ns3)) * Tensor(prdft1d, I(ns2*ns3)),
                                    stage2 := RC(Tensor(I(nfreq), pdft1d, I(ns3))),
                                    stage543c := IDirSum(i, iopconv),
                                    stage543 := RC(stage543c),
                                    stage6 := RC(Tensor(I(nfreq), ipdft1d, I(nd3))),
                                    stage7 := Tensor(iprdft1d, I(nd2*nd3)) * Tensor(I(nfreq), L(2*nd2*nd3, 2)),
                                    conv3dr := stage7 * stage6 * stage543 * stage2 * stage1,
                                    conv3dr
                            ),
    ),
    ## 5-step, YZX ====================================================
    IOPrunedMDRConv_3D_5step_yzx_freqdata := rec(
       forTransposition := false,
       applicable :=  (self, nt) >> false, #not nt.hasTags() and Length(nt.params[1]) = 3 and nt.params[7], Breaks!
       children := nt -> let(nlist := nt.params[1],
                            diag := nt.params[2],
                            oblk := nt.params[3],
                            opats := nt.params[4],
                            iblk := nt.params[5],
                            ipats := nt.params[6],
                            nfreq := nlist[1]/2+1,
                            i := Ind(nfreq*nlist[2]),
                            hfunc := Cond(ObjId(diag) = Lambda,
                                let(j := Ind(nlist[3]),
                                    Lambda(j, cxpack(diag.at(2*(j + i*nlist[3])), diag.at(2*(j + i*nlist[3])+1)))
                                ),
                                ObjId(diag) = fUnk,
                                fUnk(TComplex, nlist[3]),
                                let(list := nt.params[2].list,  # here we assume FList(TReal, [...])
                                    clist := List([1..Length(list)/2], i->Cplx(list[2*i-1], list[2*i])),
                                    fc := FList(TComplex, clist),
                                    gf := fTensor(fBase(i), fId(nlist[3])),
                                    fCompose(fc, gf)
                                )
                            ),
                            [[ PrunedPRDFT(nlist[2], -1, iblk, ipats[2]),  # stage 1: PRDFT y
                                PrunedDFT(nlist[1], -1, iblk, ipats[1]),    # stage 2: DFT z
                                IOPrunedConv(nlist[3], hfunc, oblk, opats[3], iblk, ipats[3], true), # stage 3+4+5: complex conv in x
                                PrunedIDFT(nlist[1], 1, oblk, opats[1]), # stage 6: iDFT in z
                                PrunedIPRDFT(nlist[2], 1, oblk, opats[2]),   # stage 7: iPRDFT in y
                                InfoNt(i)
                            ]]),

       apply := (nt, C, cnt) -> let(prdft1d := C[1],
                                    pdft1d := C[2],
                                    iopconv := C[3],
                                    ipdft1d := C[4],
                                    iprdft1d := C[5],
                                    i := cnt[6].params[1],
                                    nlist := nt.params[1],
                                    n1 := nlist[1],
                                    nfreq := nlist[2]/2+1,
                                    n2 := nlist[2],
                                    n3 := nlist[3],
                                    oblk := nt.params[3],
                                    opats := nt.params[4],
                                    iblk := nt.params[5],
                                    ipats := nt.params[6],
                                    ns1 := iblk * Length(ipats[1]),
                                    ns2 := iblk * Length(ipats[2]),
                                    ns3 := iblk * Length(ipats[3]),
                                    nd1 := oblk * Length(opats[1]),
                                    nd2 := oblk * Length(opats[2]),
                                    nd3 := oblk * Length(opats[3]),
                                    stage1 := Tensor(I(nfreq*ns2), L(2*ns3, ns3)) * Tensor(I(ns2), prdft1d, I(ns3)),
                                    stage2 := RC(Tensor(pdft1d, I(nfreq*ns3))),
                                    stage543c := IDirSum(i, iopconv),
                                    stage543 := RC(stage543c),
                                    stage6 := RC(Tensor(ipdft1d, I(nfreq*nd3))),
                                    stage7 := Tensor(I(nd2), iprdft1d, I(nd3)) * Tensor(I(nfreq*nd2), L(2*nd3, 2)),
                                    conv3dr := stage7 * stage6 * stage543 * stage2 * stage1,
                                    conv3dr
                            ),
    ),
    ## 3-step, (ZY)X ====================================================
    IOPrunedMDRConv_3D_3step_zyx_freqdata := rec(
       forTransposition := false,
       applicable :=  (self, nt) >> not nt.hasTags() and Length(nt.params[1]) = 3 and nt.params[7],
       children := nt -> let(nlist := nt.params[1],
                            diag := nt.params[2],
                            oblk := nt.params[3],
                            opats := nt.params[4],
                            iblk := nt.params[5],
                            ipats := nt.params[6],
                            nfreq := nlist[1]/2+1,
                            i := Ind(nfreq*nlist[2]),
                            hfunc := Cond(ObjId(diag) = Lambda,
                                let(j := Ind(nlist[3]),
                                    Lambda(j, cxpack(diag.at(2*(j + i*nlist[3])), diag.at(2*(j + i*nlist[3])+1)))
                                ),
                                ObjId(diag) = fUnk,
                                fUnk(TComplex, nlist[3]),
                                let(list := nt.params[2].list,  # here we assume FList(TReal, [...])
                                    clist := List([1..Length(list)/2], i->Cplx(list[2*i-1], list[2*i])),
                                    fc := FList(TComplex, clist),
                                    gf := fTensor(fBase(i), fId(nlist[3])),
                                    fCompose(fc, gf)
                                )
                            ),
                            [[ PrunedPRDFT(nlist[1], -1, iblk, ipats[1]),  # stage 1a: PRDFT z
                                PrunedDFT(nlist[2], -1, iblk, ipats[2]),    # stage 1b: DFT y
                                IOPrunedConv(nlist[3], hfunc, oblk, opats[3], iblk, ipats[3], true), # stage 3+4+5: complex conv in x
                                PrunedIDFT(nlist[2], 1, oblk, opats[2]), # stage 6b: iDFT in y
                                PrunedIPRDFT(nlist[1], 1, oblk, opats[1]),   # stage 6a: iPRDFT in z
                                InfoNt(i)
                            ]]),

       apply := (nt, C, cnt) -> let(prdft1d := Grp(C[1]),
                                    pdft1d := Grp(C[2]),
                                    iopconv := Grp(C[3]),
                                    ipdft1d := Grp(C[4]),
                                    iprdft1d := Grp(C[5]),
                                    i := cnt[6].params[1],
                                    nlist := nt.params[1],
                                    n1 := nlist[1],
                                    nfreq := nlist[1]/2+1,
                                    n2 := nlist[2],
                                    n3 := nlist[3],
                                    oblk := nt.params[3],
                                    opats := nt.params[4],
                                    iblk := nt.params[5],
                                    ipats := nt.params[6],
                                    ns1 := iblk * Length(ipats[1]),
                                    ns2 := iblk * Length(ipats[2]),
                                    ns3 := iblk * Length(ipats[3]),
                                    nd1 := oblk * Length(opats[1]),
                                    nd2 := oblk * Length(opats[2]),
                                    nd3 := oblk * Length(opats[3]),
                                    stage1 := Tensor(I(nfreq), L(2*ns2, ns2)) * Tensor(prdft1d, I(ns2)),
                                    stage2 := RC(Tensor(I(nfreq), pdft1d)),
                                    stage12 := Tensor(I(nfreq * n2), L(2*ns3, ns3)) * Tensor(stage2 * stage1, I(ns3)),
                                    stage543c := IDirSum(i, iopconv),
                                    stage543 := RC(stage543c),
                                    stage6 := RC(Tensor(I(nfreq), ipdft1d)),
                                    stage7 := Tensor(iprdft1d, I(nd2)) * Tensor(I(nfreq), L(2*nd2, 2)),
                                    stage67 := Tensor(stage7 * stage6, I(nd3)) * Tensor(I(nfreq * n2), L(2*nd3, 2)),
                                    conv3dr := stage67 * stage543 * stage12,
                                    conv3dr
                            ),
    ),
    ## 3-step, (YZ)X ====================================================
    IOPrunedMDRConv_3D_3step_yzx_freqdata := rec(
       forTransposition := false,
       applicable :=  (self, nt) >> false, #not nt.hasTags() and Length(nt.params[1]) = 3 and nt.params[7], Breaks!
       children := nt -> let(nlist := nt.params[1],
                            diag := nt.params[2],
                            oblk := nt.params[3],
                            opats := nt.params[4],
                            iblk := nt.params[5],
                            ipats := nt.params[6],
                            nfreq := nlist[1]/2+1,
                            i := Ind(nfreq*nlist[2]),
                            hfunc := Cond(ObjId(diag) = Lambda,
                                let(j := Ind(nlist[3]),
                                    Lambda(j, cxpack(diag.at(2*(j + i*nlist[3])), diag.at(2*(j + i*nlist[3])+1)))
                                ),
                                ObjId(diag) = fUnk,
                                fUnk(TComplex, nlist[3]),
                                let(list := nt.params[2].list,  # here we assume FList(TReal, [...])
                                    clist := List([1..Length(list)/2], i->Cplx(list[2*i-1], list[2*i])),
                                    fc := FList(TComplex, clist),
                                    gf := fTensor(fBase(i), fId(nlist[3])),
                                    fCompose(fc, gf)
                                )
                            ),
                            [[ PrunedPRDFT(nlist[2], -1, iblk, ipats[2]),  # stage 1a: PRDFT y
                                PrunedDFT(nlist[1], -1, iblk, ipats[1]),    # stage 1b: DFT z
                                IOPrunedConv(nlist[3], hfunc, oblk, opats[3], iblk, ipats[3], true), # stage 3+4+5: complex conv in x
                                PrunedIDFT(nlist[1], 1, oblk, opats[1]), # stage 6b: iDFT in z
                                PrunedIPRDFT(nlist[2], 1, oblk, opats[2]),   # stage 6a: iPRDFT in y
                                InfoNt(i)
                            ]]),

       apply := (nt, C, cnt) -> let(prdft1d := C[1],
                                    pdft1d := C[2],
                                    iopconv := C[3],
                                    ipdft1d := C[4],
                                    iprdft1d := C[5],
                                    i := cnt[6].params[1],
                                    nlist := nt.params[1],
                                    n1 := nlist[1],
                                    nfreq := nlist[2]/2+1,
                                    n2 := nlist[2],
                                    n3 := nlist[3],
                                    oblk := nt.params[3],
                                    opats := nt.params[4],
                                    iblk := nt.params[5],
                                    ipats := nt.params[6],
                                    ns1 := iblk * Length(ipats[1]),
                                    ns2 := iblk * Length(ipats[2]),
                                    ns3 := iblk * Length(ipats[3]),
                                    nd1 := oblk * Length(opats[1]),
                                    nd2 := oblk * Length(opats[2]),
                                    nd3 := oblk * Length(opats[3]),
                                    stage1 := Tensor(I(ns1), prdft1d),
                                    stage2 := RC(Tensor(pdft1d, I(nfreq))),
                                    stage12 := Tensor(I(nfreq * n2), L(2*ns3, ns3)) * Tensor(stage2 * stage1, I(ns3)),
                                    stage543c := IDirSum(i, iopconv),
                                    stage543 := RC(stage543c),
                                    stage6 := RC(Tensor(ipdft1d, I(nfreq))),
                                    stage7 := Tensor(I(nd1), iprdft1d),
                                    stage67 := Tensor(stage7 * stage6, I(nd3)) * Tensor(I(nfreq * n2), L(2*nd3, 2)),
                                    conv3dr := stage67 * stage543 * stage12,
                                    conv3dr
                            ),
    ),
#======================

    IOPrunedMDRConv_2D_2trip_xy_freqdata := rec(
       forTransposition := true,
       applicable :=  (self, nt) >> not nt.hasTags() and Length(nt.params[1]) = 2 and nt.params[7],
       children := nt -> let(nlist := nt.params[1],
                            diag := nt.params[2],
                            oblk := nt.params[3],
                            opats := nt.params[4],
                            iblk := nt.params[5],
                            ipats := nt.params[6],
                            nfreq := nlist[2]/2+1,
                            i := Ind(nfreq),
                            hfunc := Cond(ObjId(diag) = Lambda,
                                let(j := Ind(nlist[1]),
                                    pos := i + j*nfreq,
                                    Lambda(j, cxpack(diag.at(2*pos), diag.at(2*pos+1)))
                                ),
                                ObjId(diag) = fUnk,
                                fUnk(TComplex, nlist[1]),
                                let(list := nt.params[1].list,  # here we assume FList(TReal, [...])
                                    clist := List([1..Length(list)/2], i->Cplx(list[2*i-1], list[2*i])),
                                    fc := FList(TComplex, clist),
                                    gf := fTensor(fBase(i), fId(nlist[1])),
                                    fCompose(fc, gf)
                                )
                            ),
                            [[ PrunedPRDFT(nlist[2], -1, iblk, ipats[2]),  # stage 1: PRDFT x
                                IOPrunedConv(nlist[1], hfunc, oblk, opats[1], iblk, ipats[1], true), # stage 2+3+4: complex conv in y
                                PrunedIPRDFT(nlist[2], 1, oblk, opats[1]),   # stage 5: iPRDFT in x
                                InfoNt(i)
                            ]]),

       apply := (nt, C, cnt) -> let(prdft1d := C[1],
                                    iopconv := C[2],
                                    iprdft1d := C[3],
                                    i := cnt[4].params[1],
                                    nlist := nt.params[1],
                                    n1 := nlist[1],
                                    nfreq := nlist[1]/2+1,
                                    n2 := nlist[2],
                                    oblk := nt.params[3],
                                    opats := nt.params[4],
                                    iblk := nt.params[5],
                                    ipats := nt.params[6],
                                    ns1 := iblk * Length(ipats[1]),
                                    ns2 := iblk * Length(ipats[2]),
                                    nd1 := oblk * Length(opats[1]),
                                    nd2 := oblk * Length(opats[2]),
                                    
#                                    stage1 := L(2*nfreq*ns3*ns2, ns3) * Tensor(I(ns2), Tensor(L(2*nfreq, 2) * prdft1d, I(ns3))) * Tensor(L(ns2*ns1, ns2), I(ns3)),
#                                    stage2 := Tensor(I(ns3), Tensor(RC(pdft1d), I(nfreq))),
#                                    pp := Tensor(L(ns3*n2*nfreq, n2*nfreq), I(2)) * Tensor(I(ns3), L(2*nfreq*n2, nfreq)),
#                                    ppi := Tensor(I(nd3), L(2*nfreq*n2, 2*n2)) * Tensor(L(nd3*n2*nfreq, nd3), I(2)),
#                                    stage543 := ppi * IDirSum(i, RC(iopconv)) * pp,
#                                    stage76 := Tensor(L(nd2*nd1, nd1), I(nd3)) * Grp(Tensor((Tensor(I(nd2), iprdft1d * L(2*nfreq, nfreq)) *
#                                        Tensor(RC(ipdft1d), I(nfreq))), I(nd3)) * L(2*nfreq*nd3*n2, 2*nfreq*n2)),

                                    stage1 := Tensor(I(ns2), prdft1d),
                                    pp := Tensor(L(ns2*nfreq, nfreq), I(2)),
                                    ppi := Tensor(L(nd2*nfreq, nd2), I(2)),
                                    stage432 := IDirSum(i, RC(iopconv)),
                                    stage5 :=  Tensor(I(nd2), iprdft1d), 
                                    
                                    conv2dr := Grp(stage5 * ppi) * stage432 * Grp(pp * stage1),
                                    conv2dr
                            ),
    ),
#     IOPrunedMDRConv_2D_2trip_yx_freqdata := rec(
#       forTransposition := true,
#       applicable :=  (self, nt) >> not nt.hasTags() and Length(nt.params[1]) = 2 and nt.params[7],
#       children := nt -> let(nlist := nt.params[1],
#                            diag := nt.params[2],
#                            oblk := nt.params[3],
#                            opats := nt.params[4],
#                            iblk := nt.params[5],
#                            ipats := nt.params[6],
#                            nfreq := nlist[1]/2+1,
#                            i := Ind(nfreq),
#                            hfunc := Cond(ObjId(diag) = Lambda,
#                                let(j := Ind(nlist[2]),
#                                    pos := i + j*nfreq,
#                                    Lambda(j, cxpack(diag.at(2*pos), diag.at(2*pos+1)))
#                                ),
#                                ObjId(diag) = fUnk,
#                                fUnk(TComplex, nlist[2]),
#                                let(list := nt.params[2].list,  # here we assume FList(TReal, [...])
#                                    clist := List([1..Length(list)/2], i->Cplx(list[2*i-1], list[2*i])),
#                                    fc := FList(TComplex, clist),
#                                    gf := fTensor(fBase(i), fId(nlist[2])),
#                                    fCompose(fc, gf)
#                                )
#                            ),
#                            [[ PrunedPRDFT(nlist[1], -1, iblk, ipats[1]),  # stage 1: PRDFT y
#                                IOPrunedConv(nlist[2], hfunc, oblk, opats[2], iblk, ipats[2], true), # stage 2+3+4: complex conv in x
#                                PrunedIPRDFT(nlist[1], 1, oblk, opats[1]),   # stage 5: iPRDFT in y
#                                InfoNt(i)
#                            ]]),
#
#       apply := (nt, C, cnt) -> let(prdft1d := C[1],
#                                    iopconv := C[2],
#                                    iprdft1d := C[3],
#                                    i := cnt[4].params[1],
#                                    nlist := nt.params[1],
#                                    n1 := nlist[1],
#                                    nfreq := nlist[1]/2+1,
#                                    n2 := nlist[2],
#                                    oblk := nt.params[3],
#                                    opats := nt.params[4],
#                                    iblk := nt.params[5],
#                                    ipats := nt.params[6],
#                                    ns1 := iblk * Length(ipats[1]),
#                                    ns2 := iblk * Length(ipats[2]),
#                                    nd1 := oblk * Length(opats[1]),
#                                    nd2 := oblk * Length(opats[2]),
#                                    
##                                    stage1 := L(2*nfreq*ns3*ns2, ns3) * Tensor(I(ns2), Tensor(L(2*nfreq, 2) * prdft1d, I(ns3))) * Tensor(L(ns2*ns1, ns2), I(ns3)),
##                                    stage2 := Tensor(I(ns3), Tensor(RC(pdft1d), I(nfreq))),
##                                    pp := Tensor(L(ns3*n2*nfreq, n2*nfreq), I(2)) * Tensor(I(ns3), L(2*nfreq*n2, nfreq)),
##                                    ppi := Tensor(I(nd3), L(2*nfreq*n2, 2*n2)) * Tensor(L(nd3*n2*nfreq, nd3), I(2)),
##                                    stage543 := ppi * IDirSum(i, RC(iopconv)) * pp,
##                                    stage76 := Tensor(L(nd2*nd1, nd1), I(nd3)) * Grp(Tensor((Tensor(I(nd2), iprdft1d * L(2*nfreq, nfreq)) *
##                                        Tensor(RC(ipdft1d), I(nfreq))), I(nd3)) * L(2*nfreq*nd3*n2, 2*nfreq*n2)),
#
#                                    stage1 := L(2*nfreq*ns3*ns2, ns3) * Tensor(I(ns2), Tensor(L(2*nfreq, 2) * prdft1d, I(ns3))) * Tensor(L(ns2*ns1, ns2), I(ns3)),
#                                    pp := Tensor(L(ns3*n2*nfreq, n2*nfreq), I(2)) * Tensor(I(ns3), L(2*nfreq*n2, nfreq)),
#                                    ppi := Tensor(I(nd3), L(2*nfreq*n2, 2*n2)) * Tensor(L(nd3*n2*nfreq, nd3), I(2)),
#                                    stage432 := ppi * IDirSum(i, RC(iopconv)) * pp,
#                                    stage5 := Tensor(L(nd2*nd1, nd1), I(nd3)) * Grp(Tensor((Tensor(I(nd2), iprdft1d * L(2*nfreq, nfreq)) *
#                                        Tensor(RC(ipdft1d), I(nfreq))), I(nd3)) * L(2*nfreq*nd3*n2, 2*nfreq*n2)),
#
#                                    conv2dr := stage5 * stage432 * stage1,
#                                    conv2dr
#                            ),
#    ),
   
    
));
