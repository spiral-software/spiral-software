
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


ImportAll(paradigms.smp); # for AParSMP
ImportAll(paradigms.vector); # for VRCLR
Import(approx); # for CeilingRat.

Load(spiral.paradigms.dram.hacks); # for TTensor3 etc.
Import(hacks);

#_swrap := (nt) -> let(t:=nt.firstTag(), ScratchWrap(t.size,t.nsgmts,t.linesize));

#_nodeFitsTensorDoesnt := (nt) -> let (
#    k := nt.getTag(1).size,
#    m := nt.params[1].dims()[2],
#    n := nt.params[2],
#
#    2*m <= k and m*n >= k
#);
	
#_isALStoreTag := (nt) -> nt.isTag(1,ALStore) or nt.isTag(1,ALStoreCx);

_isADramTag := (nt) -> nt.isTag(1,ADram);
_isATileTag := (nt) -> nt.isTag(1,ATile);
_isATileRdTag := (nt) -> nt.isTag(1,ATileRd);
_isATileWrTag := (nt) -> nt.isTag(1,ATileWr);
_isACubeTag := (nt) -> nt.isTag(1,ACube);
_isACubeRdTag := (nt) -> nt.isTag(1,ACubeRd);
_isACubeWrTag := (nt) -> nt.isTag(1,ACubeWr);
_isTL_I := (nt) -> nt.params[1]=1 and nt.params[2]=1 and nt.params[3]>1 and nt.params[4]=1;
_isTL_L := (nt) -> nt.params[1]>1 and nt.params[2]>1 and nt.params[3]=1 and nt.params[4]=1;
_isTL_I_L := (nt) -> nt.params[1]>1 and nt.params[2]>1 and nt.params[3]>1 and nt.params[4]=1;
_isReduced1D := (nt) -> let(p := Length(nt.params[1].params), When(p=3, (Length(nt.params[1].params[1]) = 1), true));

_DetermineTag := function(t)
	local ll,tag,k;
	ll := When(Length(t.params)=3,Length(t.params[1]),1); #if MDDFT,x,1
	
	if(ll = 2 or ll = 1) then
		k := Int(Log2Int(t.getTag(1).rb)/2);
		k := 2^k;
		if(not (k^2 = t.getTag(1).rb)) then
			PrintLine("WARNING: Tile <-> Row-Buffer cannot be mathced perfectly!");
		fi;
		tag := ATile(k,t.getTag(1).m);
	else if(ll = 3) then
		k := Int(Log2Int(t.getTag(1).rb)/3);
		k := 2^k;
		if(not (k^3 = t.getTag(1).rb)) then
			PrintLine("WARNING: Cube <-> Row-Buffer cannot be mathced perfectly!");
		fi;
		tag := ACube(k,t.getTag(1).m);
	else
		PrintLine("I cannot handle an algorithm with ",ll," stages for now, sorry..");
		Error("");
	fi; fi;

	return tag;
end;
#  NewRulesFor(TCompose, rec(
#  # (AB) -> (A) fence (B)
#  	AB_tile := rec(
#  	)
#  ));

NewRulesFor(MDDFT, rec(
    MDDFT_tSPL_RowCol_break_2D := rec(
        info := "tSPL MDDFT_n -> MDDFT_n/d, MDDFT_d",

        applicable := (self, t) >> Length(t.params[1]) > 1 and not _isADramTag(t) and _isATileTag(t),
        freedoms := t -> [ [1..Length(t.params[1])-1] ],

        child := (t, fr) -> let(
            newdims := SplitAt(t.params[1], fr[1]),
            rot := t.params[2],
            [ TTensor(
                MDDFT(newdims[1], rot),
                MDDFT(newdims[2], rot)
            ).withTags(t.getTags())]
        ),

        apply := (t, C, Nonterms) -> C[1],
        switch := false
    ),
	
	MDDFT_tSPL_RowCol_break_3D := rec(
        info := "tSPL MDDFT_n -> MDDFT_n/d, MDDFT_d",

        applicable := (self, t) >> Length(t.params[1]) > 1 and not _isADramTag(t) and _isACubeTag(t),
        freedoms := t -> [ [1..Length(t.params[1])-1] ],

        children := (t) -> let(
            newdims := t.params[1],
            
            [[ TTensor3(
                MDDFT([newdims[1]]),
                MDDFT([newdims[2]]),
                MDDFT([newdims[3]])
            ).withTags(t.getTags())]]
        ),

        apply := (t, C, Nonterms) -> C[1],
        switch := false
    ),
	
	MDDFT_tSPL_RowCol_push := rec(
        info := "tSPL MDDFT_n -> MDDFT_n/d, MDDFT_d",

        applicable := (self, t) >> Length(t.params[1]) > 1 and _isADramTag(t),
        freedoms := t -> [ [1..Length(t.params[1])-1] ],

        child := (t, fr) -> let(
			tag := _DetermineTag(t),
			newTag := Concat([tag], Drop(t.getTags(), 1)),
            [ t.withoutFirstTag().withTags(newTag) ]
        ),

        apply := (t, C, Nonterms) -> C[1],
        switch := false
    )
	
));


NewRulesFor(TTwiddle, rec(
    TTwiddle_dram := rec(
        minSize := false,
        maxSize := false,
        applicable := (self, nt) >> ((IsBool(self.minSize) and not self.minSize) or nt.params[1] >= self.minSize) and 
            ((IsBool(self.maxSize) and not self.maxSize) or nt.params[1] <= self.maxSize),
        forTransposition := false,
        apply := (t, C, Nonterms) -> TwiddleROM(t.params[1], t.params[2], t.params[3])
    )
));


#######################################################################################
#   tSPL DFT rules
NewRulesFor(DFT, rec(
    #F DFT_CT: 1965
    #F   General Cooley-Tukey Rule
    #F   DFT_n = (DFT_n/d tensor I_d) * diag * (I_n/d tensor F_d) * perm
    #F
    #F Cooley/Tukey:
    #F   An Algorithm for the Machine Calculation of Complex Fourier Series.
    #F   Mathematics of Computation, Vol. 19, 1965, pp. 297--301.
    #F
    DFT_tSPL_CT_tiled := rec(
    info          := "tSPL DFT(mn,k) -> DFT(m, k%m), DFT(n, k%n)",

    maxSize       := false,

    applicable    := (self, nt) >> nt.params[1] > 2
        and (self.maxSize = false or nt.params[1] <= self.maxSize)
		and _isATileTag(nt)
        and not IsPrime(nt.params[1])
		and IsPosInt(Sqrt(nt.params[1]))
        and nt.hasTags(),

    children      := nt -> let(n := Sqrt(nt.params[1]), m := n,
	
	[[
        TCompose([
            TGrp(TCompose([
                TTensorI(DFT(m, nt.params[2] mod m), n, AVec, AVec),
                TTwiddle(m*n, n, nt.params[2])
            ])),
            TGrp(TTensorI(DFT(n, nt.params[2] mod n), m, APar, AVec))
        ]).withTags(nt.getTags())
    ]]),

    apply := (nt, c, cnt) -> c[1],

    switch := false
    ),
	
	DFT_tSPL_push_tiled := rec(
        info := "tSPL DFT(mn,k) -> DFT(m, k%m), DFT(n, k%n)",
		
		maxSize       := false,
        
		applicable := (self, t) >> t.params[1] > 2 and _isADramTag(t),

        children := (t) -> let(
			tag := _DetermineTag(t),
			newTag := Concat([tag], Drop(t.getTags(), 1)),
            [[ t.withoutFirstTag().withTags(newTag) ]]
        ),

        apply := (t, C, Nonterms) -> C[1],
        switch := false
    )
));


############################## 3D Rules ##############################

#   (A x B) rules
NewRulesFor(TTensor3, rec(
#   (A x B x C) -> (A x B x I)(I x I x C)
    AxBxI_IxIxC := rec(
        info := "(A x B x C) -> (A x B x I)(I x I x C)",
        forTransposition := false,
        applicable := nt -> true,
        #inplace := false,
        children := (self, nt) >> let(#inp := When(self.inplace, Inplace, x->x),
            [[ TCompose([
                TTensorI3(nt.params[1], nt.params[2], I(nt.params[3].dims()[2])),
                TTensorI3(I(nt.params[1].dims()[2]), I(nt.params[2].dims()[2]), nt.params[3])
            ]).withTags(nt.getTags()) ]]),
        apply := (nt, c, cnt) -> c[1],
#D        isApplicable := P -> true,
#D        allChildren := P -> [[TCompose([TTensorI(P[1], P[2].dims()[1], AVec, AVec), TTensorI(P[2], P[1].dims()[2], APar, APar)], P[3])]],
#D        rule := (P, C) -> C[1]
    ),
	#   (A x B x C) -> (A x I x I)(I x B x C)
    AxIxI_IxBxC := rec(
        info := "(A x B x C) -> (A x I x I)(I x B x C)",
        forTransposition := false,
        applicable := nt -> true,
        #inplace := false,
        children := (self, nt) >> let(#inp := When(self.inplace, Inplace, x->x),
            [[ TCompose([
				TTensorI3(nt.params[1], I(nt.params[2].dims()[2]), I(nt.params[3].dims()[2])),
				TTensorI3(I(nt.params[1].dims()[2]), nt.params[2], nt.params[3])
            ]).withTags(nt.getTags()) ]]),
        apply := (nt, c, cnt) -> c[1],
	),
	
	# (A x B x C) -> (A x I x I)(I x B x I)(I x I x C)
	AxIxI_IxBxI_IxIxC := rec(
        info := "(A x B x C) -> (A x I x I)(I x B x I)(I x I x C)",
        forTransposition := false,
        applicable := nt -> true,
        #inplace := false,
        children := (self, nt) >> let(#inp := When(self.inplace, Inplace, x->x),
            [[ TCompose([
				TTensorI3(nt.params[1], I(nt.params[2].dims()[2]), I(nt.params[3].dims()[2])),
				TTensorI3(I(nt.params[1].dims()[2]), nt.params[2], I(nt.params[3].dims()[2])),
				TTensorI3(I(nt.params[1].dims()[2]), I(nt.params[2].dims()[2]), nt.params[3])
            ]).withTags(nt.getTags()) ]]),
        apply := (nt, c, cnt) -> c[1],
	)
));

NewRulesFor(TTensorI3, rec(
	# (In x In x An) -> In3 (In2/k2 x (Ik x Ik x DFTn)) In3
	IxIxA_cube := rec(
		forTransposition := false,
		applicable := nt -> nt.hasTags()
							and _isACubeTag(nt)
							and IsIIA(nt.params)
							and IsPosInt(nt.params[3].dims()[2] / nt.getTag(1).k / nt.getTag(1).k)
							and (Length(nt.params[3].params[1]) = 1) # reduced to 1D-DFT kernel
							and (nt.dims()[2] > nt.getTag(1).m),
							
		children := nt -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := nt.params[3].dims()[2],
			RdTag := Concat([ACubeRd(k,m)], Drop(nt.getTags(), 1)),
			WrTag := Concat([ACubeWr(k,m)], Drop(nt.getTags(), 1)),
			
			[[ 
				TL(1,1,n*n*n,1).withTags(WrTag),
				TTensorI3(I(k),I(k),nt.params[3]).withTags(nt.getTags()), #.withTags(Drop(nt.getTags(), 1)) ,
				TL(1,1,n*n*n,1).withTags(RdTag) 
			]]
		),
		
		apply := (nt, c, cnt) -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := nt.params[3].dims()[2],
			
			MemFence(
				c[1] *
				DTensor(
					I(n*n/k/k),
					(c[2])
				) *
				c[3]
			)
		)
	),
	
	# (An x In x In) -> Ln3_n (In2/k2 x (Ik x Ik x DFTn)) Ln3_n2
	AxIxI_cube := rec(
		forTransposition := false,
		applicable := nt -> nt.hasTags()
							and _isACubeTag(nt)
							and IsAII(nt.params)
							and IsPosInt(nt.params[1].dims()[2] / nt.getTag(1).k / nt.getTag(1).k)
							and (Length(nt.params[1].params[1]) = 1) # reduced to 1D-DFT kernel
							and (nt.dims()[2] > nt.getTag(1).m),
							
		children := nt -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := nt.params[1].dims()[2],
			RdTag := Concat([ACubeRd(k,m)], Drop(nt.getTags(), 1)),
			WrTag := Concat([ACubeWr(k,m)], Drop(nt.getTags(), 1)),
			
			[[ 
				TL(n*n*n,n,1,1).withTags(WrTag) ,
				TTensorI3(I(k),I(k),nt.params[1]).withTags(nt.getTags()), #.withTags(Drop(nt.getTags(), 1)) ,
				TL(n*n*n,n*n,1,1).withTags(RdTag) 
			]]
		),
		
		apply := (nt, c, cnt) -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := nt.params[1].dims()[2],
			
			MemFence(
				c[1] *
				DTensor(
					I(n*n/k/k),
					(c[2])
				) *
				c[3]
			)
		)	
	),

	# (In x An x In) -> In x Ln2_n (In2/k2 x (Ik x Ik x DFTn)) In x Ln2_n
	IxAxI_cube := rec(
		forTransposition := false,
		applicable := nt -> nt.hasTags()
							and _isACubeTag(nt)
							and IsIAI(nt.params)
							and IsPosInt(nt.params[2].dims()[2] / nt.getTag(1).k / nt.getTag(1).k)
							and (Length(nt.params[2].params[1]) = 1) # reduced to 1D-DFT kernel
							and (nt.dims()[2] > nt.getTag(1).m),
							
		children := nt -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := nt.params[2].dims()[2],
			RdTag := Concat([ACubeRd(k,m)], Drop(nt.getTags(), 1)),
			WrTag := Concat([ACubeWr(k,m)], Drop(nt.getTags(), 1)),
			
			[[ 
				TL(n*n,n,n,1).withTags(WrTag) ,
				TTensorI3(I(k),I(k),nt.params[2]).withTags(nt.getTags()), #.withTags(Drop(nt.getTags(), 1)) ,
				TL(n*n,n,n,1).withTags(RdTag) 
			]]
		),
		
		apply := (nt, c, cnt) -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := nt.params[2].dims()[2],
			
			MemFence(
				c[1] *
				DTensor(
					I(n*n/k/k),
					(c[2])
				) *
				c[3]
			)
		)	
	),
	
	IxIxA_cube_base := rec(
		forTransposition := false,
		applicable := nt -> nt.hasTags()
							and _isACubeTag(nt)
							and IsIIA(nt.params)
							and (Length(nt.params[3].params[1]) = 1) # reduced to 1D-DFT kernel
							and (nt.dims()[2] <= nt.getTag(1).m),
							
		children := nt -> [[ ]],
		
		
		apply := (nt, c, cnt) -> let(
			n := nt.params[3].dims()[2],
			k2 := nt.params[2].dims()[2],
			k1 := nt.params[1].dims()[2],
			
			CompKern(I(k1),I(k2),DFT(n)) 			
		)
		
	)
	

));


NewRulesFor(TL, rec(
# In^2_{read} -> (In/k x L^n_k x Ik)(In/k x L^n_n/k x Ik)	
	I_cubeRd := rec(
		applicable := nt -> nt.hasTags()
							and _isACubeRdTag(nt)
							and _isTL_I(nt)
							and IsPosInt(RootInt(nt.params[3],3)) #n3 -> n
							and IsPosInt(RootInt(nt.params[3],3) / nt.getTag(1).k),
		
		apply := (nt, c, cnt) -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := RootInt(nt.params[3],3),
			#mem := MemRdPrm(fTensor(fId(n/k),L(n*n,n*n/k/k),fId(k))) * MemRdPrm(fTensor(fId(n),L(n,k),fId(n))),
			mem := MemRdPrm(fTensor(fId(n/k),L(n*n,n*n/k/k),fId(k))) * MemRdPrm(fTensor(fId(n),L(n,k),fId(n/k),fId(k))),
			loc := DTensor(I(n*n/k/k), LocalRdPrm(fTensor(L(n*k,k*k), fId(k)))),
			
			loc*mem
			
			
		)
	),
# In^2_{write} -> (In/k x L^n_k x Ik)(In/k x L^n_n/k x Ik)	
	I_cubeWr := rec(
		applicable := nt -> nt.hasTags()
							and _isACubeWrTag(nt)
							and _isTL_I(nt)
							and IsPosInt(RootInt(nt.params[3],3)) #n3 -> n
							and IsPosInt(RootInt(nt.params[3],3) / nt.getTag(1).k),
		
			
		apply := (nt, c, cnt) -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := RootInt(nt.params[3],3),
			#mem := MemRdPrm(fTensor(fId(n/k),L(n*n,n*n/k/k),fId(k))) * MemRdPrm(fTensor(fId(n),L(n,k),fId(n))),
			mem := MemRdPrm(fTensor(fId(n/k),L(n*n,n*n/k/k),fId(k))) * MemRdPrm(fTensor(fId(n),L(n,k),fId(n/k),fId(k))),
			loc := DTensor(I(n*n/k/k), LocalRdPrm(fTensor(L(n*k,k*k), fId(k)))),
			
			TransposedSPL(loc*mem)
		)
	),
# Ln^2_n_{read} -> (In/k x L^nk_k)(L^n^2/k_n/k x Ik)
	Ln3n2_cubeRd := rec(
		applicable := nt -> nt.hasTags()
							and _isACubeRdTag(nt)
							and _isTL_L(nt)
							and IsPosInt(RootInt(nt.params[1],3)) #n3 -> n
							and IsPosInt(RootInt(nt.params[1],3) / nt.getTag(1).k),
							
		apply := (nt, c, cnt) -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := RootInt(nt.params[1],3),
			#mem := 	MemRdPrm(fTensor(L(n*n*n/k,n*n/k/k), fId(k))) * MemRdPrm(fTensor(fId(n),L(n,k),fId(n))),
			mem := 	MemRdPrm(fTensor(L(n*n*n/k,n*n/k/k), fId(k))) * MemRdPrm(fTensor(fId(n),L(n,k),fId(n/k),fId(k))),
			loc := DTensor(I(n*n/k/k), LocalRdPrm(fTensor(L(n*k*k,k*k)))),
			
			loc*mem
		)
	),
# Ln^2_n_{write} -> (L^n^2/k_n x Ik)(In/k x L^nk_n)
	Ln3n_cubeWr := rec(
		applicable := nt -> nt.hasTags()
							and _isACubeWrTag(nt)
							and _isTL_L(nt)
							and IsPosInt(RootInt(nt.params[1],3)) #n3 -> n
							and IsPosInt(RootInt(nt.params[1],3) / nt.getTag(1).k),
		
				
		apply := (nt, c, cnt) -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := RootInt(nt.params[1],3),
			#mem := 	MemRdPrm(fTensor(L(n*n*n/k,n*n/k/k), fId(k))) * MemRdPrm(fTensor(fId(n),L(n,k),fId(n))),
			mem := 	MemRdPrm(fTensor(L(n*n*n/k,n*n/k/k), fId(k))) * MemRdPrm(fTensor(fId(n),L(n,k),fId(n/k),fId(k))),
			loc := DTensor(I(n*n/k/k), LocalRdPrm(fTensor(L(n*k*k,k*k)))),
			
			TransposedSPL(loc*mem)
		)
			
	),
# Ln^2_n_{read} -> (In/k x L^nk_k)(L^n^2/k_n/k x Ik)
	InLn2n_cubeRd := rec(
		applicable := nt -> nt.hasTags()
							and _isACubeRdTag(nt)
							and _isTL_I_L(nt)
							and IsPosInt(RootInt(nt.params[1],2)) #n2 -> n
							and IsPosInt(RootInt(nt.params[1],2) / nt.getTag(1).k),
							
		apply := (nt, c, cnt) -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := RootInt(nt.params[1],2),
			#mem := MemRdPrm(fTensor(fId(n/k), L(n*n,n/k), fId(k))) * MemRdPrm(fTensor(fId(n/k), L(n,n/k), fId(n*k))),
			mem := MemRdPrm(fTensor(fId(n/k), L(n*n,n/k), fId(k))) * MemRdPrm(fTensor(fId(n/k), L(n,n/k), fId(n),fId(k))),
			loc := DTensor(I(n*n/k/k), LocalRdPrm(L(n*k*k,k*k))) * DTensor(I(n*n/k/k), LocalRdPrm(fTensor(fId(n/k),L(k*k,k),fId(k)))),
			
			loc*mem
		)
	),

	InLn2n_cubeWr := rec(
		applicable := nt -> nt.hasTags()
							and _isACubeWrTag(nt)
							and _isTL_I_L(nt)
							and IsPosInt(RootInt(nt.params[1],2)) #n2 -> n
							and IsPosInt(RootInt(nt.params[1],2) / nt.getTag(1).k),
							
		apply := (nt, c, cnt) -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := RootInt(nt.params[1],2),
			#mem := MemRdPrm(fTensor(fId(n/k), L(n*n,n/k), fId(k))) * MemRdPrm(fTensor(fId(n/k), L(n,n/k), fId(n*k))),
			mem := MemRdPrm(fTensor(fId(n/k), L(n*n,n/k), fId(k))) * MemRdPrm(fTensor(fId(n/k), L(n,n/k), fId(n),fId(k))),
			loc := DTensor(I(n*n/k/k), LocalRdPrm(L(n*k*k,k*k))) * DTensor(I(n*n/k/k), LocalRdPrm(fTensor(fId(n/k),L(k*k,k),fId(k)))),
			
			TransposedSPL(loc*mem)
		)		
	)

));

############################## 2D Rules ##############################

NewRulesFor(TTensorI, rec(
 
# ================================================================
# nt.params; => [DFT_matix, Size_of_I, <APar,Avec>, <APar,Avec>]
# So,
# nt.params[1].dims()[2] -> Size of DFT matrix
# nt.params[2] -> Size of I matrix
# IsParPar actually checks nt.params[3]=APar and nt.params[4]=APar
#
# nt.getTag(1) -> Tag object. So access its methods via "."
#
# ================================================================
#  tSPL I x L x I
#  TL(m, n, k, j)
#  I(k), L(m, n), I(j)
# ================================================================


# (In x An) -> In^2 (In/k x Ik x An) In^2
# (In x An)_{Tile(k,m)} -> In^2_{TileWr(k,m)} (In/k x Ik x An) In^2_{TileRd(k,m)} for k|n and m>kn
	IxA_tile := rec(
		forTransposition := false,
		applicable := nt -> nt.hasTags()
							and _isATileTag(nt)
							and IsParPar(nt.params)
							and IsPosInt(nt.params[1].dims()[2] / nt.getTag(1).k)
							and _isReduced1D(nt) # reduced to 1D-DFT kernel
							#and (Length(nt.params[1].params[1]) = 1) # reduced to 1D-DFT kernel
							and (nt.params[1].dims()[2]*nt.params[2] > nt.getTag(1).m),
							
		children := nt -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := nt.params[1].dims()[2],
			RdTag := Concat([ATileRd(k,m)], Drop(nt.getTags(), 1)),
			WrTag := Concat([ATileWr(k,m)], Drop(nt.getTags(), 1)),
			
			[[ 
				TL(1,1,n*n,1).withTags(WrTag) ,
				TTensorI(nt.params[1],k,APar,APar).withTags(nt.getTags()), #.withTags(Drop(nt.getTags(), 1)) ,
				TL(1,1,n*n,1).withTags(RdTag) 
			]]
		),
		
		apply := (nt, c, cnt) -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := nt.params[1].dims()[2],
			
			MemFence(
				c[1] *
				DTensor(
					I(n/k),
					(c[2])
				) *
				c[3]
			)
		)
	),

# (An x In) -> L^{n^2}_n (In/k x Ik x An) L^{n^2}_n
	AxI_tile := rec(
		forTransposition := false,
		applicable := nt -> nt.hasTags()
							and _isATileTag(nt)
							and IsVecVec(nt.params)
							and IsPosInt(nt.params[1].dims()[2] / nt.getTag(1).k)
							and _isReduced1D(nt) # reduced to 1D-DFT kernel
							#and (Length(nt.params[1].params[1]) = 1) # reduced to 1D-DFT kernel
							and (nt.params[1].dims()[2]*nt.params[2] > nt.getTag(1).m),
							
		children := nt -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := nt.params[1].dims()[2],
			RdTag := Concat([ATileRd(k,m)], Drop(nt.getTags(), 1)),
			WrTag := Concat([ATileWr(k,m)], Drop(nt.getTags(), 1)),
			
			[[ 
				TL(n*n,n,1,1).withTags(WrTag) ,
				TTensorI(nt.params[1],k,APar,APar).withTags(nt.getTags()), #.withTags(Drop(nt.getTags(), 1)) ,
				TL(n*n,n,1,1).withTags(RdTag) 
			]]
		),
		
		apply := (nt, c, cnt) -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := nt.params[1].dims()[2],
			
			MemFence(
				c[1] *
				DTensor(
					I(n/k),
					(c[2])
				) *
				c[3]
			)
		)
	),
# L^{n^2}_n (In x An) -> L^{n^2}_n (In/k x Ik x An) In^2
	L_IxA_tile := rec(
		forTransposition := false,
		applicable := nt -> nt.hasTags()
							and _isATileTag(nt)
							and IsVecPar(nt.params)
							and IsPosInt(nt.params[1].dims()[2] / nt.getTag(1).k)
							and _isReduced1D(nt) # reduced to 1D-DFT kernel
							#and (Length(nt.params[1].params[1]) = 1) # reduced to 1D-DFT kernel
							and (nt.params[1].dims()[2]*nt.params[2] > nt.getTag(1).m),
							
		children := nt -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := nt.params[1].dims()[2],
			RdTag := Concat([ATileRd(k,m)], Drop(nt.getTags(), 1)),
			WrTag := Concat([ATileWr(k,m)], Drop(nt.getTags(), 1)),
			
			[[ 
				TL(n*n,n,1,1).withTags(WrTag) ,
				TTensorI(nt.params[1],k,APar,APar).withTags(nt.getTags()), #.withTags(Drop(nt.getTags(), 1)) ,
				TL(1,1,n*n,1).withTags(RdTag) 
			]]
		),
		
		apply := (nt, c, cnt) -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := nt.params[1].dims()[2],
			
			MemFence(
				c[1] *
				DTensor(
					I(n/k),
					(c[2])
				) *
				c[3]
			)
		)
	),
# (In x An) L^{n^2}_n -> In^2 (In/k x Ik x An) L^{n^2}_n
	IxA_L_tile := rec(
		forTransposition := false,
		applicable := nt -> nt.hasTags()
							and _isATileTag(nt)
							and IsParVec(nt.params)
							and IsPosInt(nt.params[1].dims()[2] / nt.getTag(1).k)
							and _isReduced1D(nt) # reduced to 1D-DFT kernel
							#and (Length(nt.params[1].params[1]) = 1) # reduced to 1D-DFT kernel
							and (nt.params[1].dims()[2]*nt.params[2] > nt.getTag(1).m),
							
		children := nt -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := nt.params[1].dims()[2],
			RdTag := Concat([ATileRd(k,m)], Drop(nt.getTags(), 1)),
			WrTag := Concat([ATileWr(k,m)], Drop(nt.getTags(), 1)),
			
			[[ 
				TL(1,1,n*n,1).withTags(WrTag) ,
				TTensorI(nt.params[1],k,APar,APar).withTags(nt.getTags()), #.withTags(Drop(nt.getTags(), 1)) ,
				TL(n*n,n,1,1).withTags(RdTag) 
			]]
		),
		
		apply := (nt, c, cnt) -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := nt.params[1].dims()[2],
			
			MemFence(
				c[1] *
				DTensor(
					I(n/k),
					(c[2])
				) *
				c[3]
			)
		)
	)
));

# Termination Rules, i.e. computation fits in local memory
NewRulesFor(TTensorI, rec(
	IxA_tile_base := rec(
		applicable := nt -> nt.hasTags()
							and _isATileTag(nt)
							and IsParPar(nt.params)
							and (nt.params[1].dims()[2] * nt.params[2] <= nt.getTag(1).m),
							
		children := nt -> [[ ]],
		
		apply := (nt, c ,cnt) -> let(
			k := nt.params[2],
			n := nt.params[1].dims()[2],
			
			CompKern(I(k), DFT(n))
		)
	),
	
	
	AxI_tile_base := rec(
		applicable := nt -> nt.hasTags()
							and _isATileTag(nt)
							and IsVecVec(nt.params)
							and (nt.params[1].dims()[2] * nt.params[2] <= nt.getTag(1).m),
							
		children := nt -> [[ ]],
		
		apply := (nt, c ,cnt) -> let(
			k := nt.params[2],
			n := nt.params[1].dims()[2],
			
			# NOTE: Check the L index!
			LocalWrPrm(fTensor(L(k*n,n))) * CompKern(I(k), DFT(n)) * LocalRdPrm(fTensor(L(n*k,k)))
			# TPrm and TL for splhdl
			#LocalWrPrm(Tensor(L(k*n,n))) * CompKern(I(k), DFT(n)) * LocalRdPrm(Tensor(L(n*k,k)))
		)
	),
	
	
	L_IxA_tile_base := rec(
		applicable := nt -> nt.hasTags()
							and _isATileTag(nt)
							and IsVecPar(nt.params)
							and (nt.params[1].dims()[2] * nt.params[2] <= nt.getTag(1).m),
							
		children := nt -> [[ ]],
		
		apply := (nt, c ,cnt) -> let(
			k := nt.params[2],
			n := nt.params[1].dims()[2],
			
			# NOTE: Check the L index!
			# TTensorI(DFTn,k,AVec,APar) -> L(n*k,???) * (Ik x DFTn)
			# 
			# 
			# 
			LocalWrPrm(fTensor(L(n*k,k))) * CompKern(I(k),DFT(n))
			# TPrm and TL for splhdl
			#LocalWrPrm(Tensor(L(n*k,k))) * CompKern(I(k),DFT(n))
		)
	),
	
	
	IxA_L_tile_base := rec(
		applicable := nt -> nt.hasTags()
							and _isATileTag(nt)
							and IsParVec(nt.params)
							and (nt.params[1].dims()[2] * nt.params[2] <= nt.getTag(1).m),
							
		children := nt -> [[ ]],
		
		apply := (nt, c ,cnt) -> let(
			k := nt.params[2],
			n := nt.params[1].dims()[2],
			
			# NOTE: Check the L index!
			CompKern(I(k),DFT(n)) * LocalRdPrm(fTensor(L(k*n,n)))
			# TPrm and TL for splhdl
			#CompKern(I(k),DFT(n)) * LocalRdPrm(Tensor(L(k*n,n)))
		)
	)
));


NewRulesFor(TL, rec(
# In^2_{read} -> (In/k x L^n_k x Ik)(In/k x L^n_n/k x Ik)	
	I_tileRd := rec(
		applicable := nt -> nt.hasTags()
							and _isATileRdTag(nt)
							and _isTL_I(nt)
							and IsPosInt(Sqrt(nt.params[3]))
							and IsPosInt(Sqrt(nt.params[3]) / nt.getTag(1).k),
		
		apply := (nt, c, cnt) -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := Sqrt(nt.params[3]),
			
				
			DTensor(I(n/k), LocalRdPrm(fTensor(L(n,k),fId(k)))) * MemRdPrm(fTensor(fId(n/k), L(n,n/k), fId(k))) 
			# TPrm and TL for splhdl		
			#DTensor(I(n/k), LocalRdPrm(Tensor(L(n,k),I(k)))) * MemRdPrm(fTensor(fId(n/k), L(n,n/k), fId(k))) 
			
		)
	),
# In^2_{write} -> (In/k x L^n_k x Ik)(In/k x L^n_n/k x Ik)	
	I_tileWr := rec(
		applicable := nt -> nt.hasTags()
							and _isATileWrTag(nt)
							and _isTL_I(nt)
							and IsPosInt(Sqrt(nt.params[3]))
							and IsPosInt(Sqrt(nt.params[3]) / nt.getTag(1).k),
		
			
		apply := (nt, c, cnt) -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := Sqrt(nt.params[3]),
			
			
			MemWrPrm(fTensor(fId(n/k), L(n,k), fId(k))) * DTensor(I(n/k), LocalWrPrm(fTensor(L(n,n/k),fId(k))))
			# TPrm and TL for splhdl
			#MemWrPrm(fTensor(fId(n/k), L(n,k), fId(k))) * DTensor(I(n/k), LocalWrPrm(Tensor(L(n,n/k),I(k))))
		)
	),
# Ln^2_n_{read} -> (In/k x L^nk_k)(L^n^2/k_n/k x Ik)
	L_tileRd := rec(
		applicable := nt -> nt.hasTags()
							and _isATileRdTag(nt)
							and _isTL_L(nt)
							and IsPosInt(Sqrt(nt.params[1]))
							and IsPosInt(Sqrt(nt.params[1]) / nt.getTag(1).k),
							
		apply := (nt, c, cnt) -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := Sqrt(nt.params[1]),
				
			DTensor(I(n/k), LocalRdPrm(fTensor(L(n*k,k)))) * MemRdPrm(fTensor(L(n*n/k,n/k), fId(k))) 
			# TPrm and TL for splhdl
			#DTensor(I(n/k), LocalRdPrm(Tensor(L(n*k,k)))) * MemRdPrm(fTensor(L(n*n/k,n/k), fId(k))) 
			
		)
	),
# Ln^2_n_{write} -> (L^n^2/k_n x Ik)(In/k x L^nk_n)
	L_tileWr := rec(
		applicable := nt -> nt.hasTags()
							and _isATileWrTag(nt)
							and _isTL_L(nt)
							and IsPosInt(Sqrt(nt.params[1]))
							and IsPosInt(Sqrt(nt.params[1]) / nt.getTag(1).k),
		
				
		apply := (nt, c, cnt) -> let(
			k := nt.getTag(1).k,
			m := nt.getTag(1).m,
			n := Sqrt(nt.params[1]),
			
			MemWrPrm(fTensor(L(n*n/k,n), fId(k))) * DTensor(I(n/k), LocalWrPrm(fTensor(L(n*k,n))))
			# TPrm and TL for splhdl
			#MemWrPrm(fTensor(L(n*n/k,n), fId(k))) * DTensor(I(n/k), LocalWrPrm(Tensor(L(n*k,n))))
		)
			
	)
));

#==============================================================
# First version of TL rules
# Generates children for permutations that is to be further optimized
#==============================================================
#	NewRulesFor(TL, rec(
#	# In^2_{read} -> (In/k x L^n_k x Ik)(In/k x L^n_n/k x Ik)	
#		I_tileRd := rec(
#			applicable := nt -> nt.hasTags()
#								and _isATileRdTag(nt)
#								and _isTL_I(nt)
#								and IsPosInt(Sqrt(nt.params[3]))
#								and IsPosInt(Sqrt(nt.params[3]) / nt.getTag(1).k),
#			
#			children := nt -> let(
#				k := nt.getTag(1).k,
#				m := nt.getTag(1).m,
#				n := Sqrt(nt.params[3]),
#				
#				[[
#					TL(n, k, n/k, k).withTags(Drop(nt.getTags(), 1)) , 
#					TL(n, n/k, n/k, k).withTags(Drop(nt.getTags(), 1))
#				]]
#			),
#			
#			apply := (nt, c, cnt) -> c[1] * c[2]
#		),
#	# In^2_{write} -> (In/k x L^n_k x Ik)(In/k x L^n_n/k x Ik)	
#		I_tileWr := rec(
#			applicable := nt -> nt.hasTags()
#								and _isATileWrTag(nt)
#								and _isTL_I(nt)
#								and IsPosInt(Sqrt(nt.params[3]))
#								and IsPosInt(Sqrt(nt.params[3]) / nt.getTag(1).k),
#			
#			children := nt -> let(
#				k := nt.getTag(1).k,
#				m := nt.getTag(1).m,
#				n := Sqrt(nt.params[3]),
#				
#				[[
#					TL(n, k, n/k, k).withTags(Drop(nt.getTags(), 1)) , 
#					TL(n, n/k, n/k, k).withTags(Drop(nt.getTags(), 1))
#				]]
#			),
#			
#			apply := (nt, c, cnt) -> c[1] * c[2]
#		),
#	# Ln^2_n_{read} -> (In/k x L^nk_k)(L^n^2/k_n/k x Ik)
#		L_tileRd := rec(
#			applicable := nt -> nt.hasTags()
#								and _isATileRdTag(nt)
#								and _isTL_L(nt)
#								and IsPosInt(Sqrt(nt.params[1]))
#								and IsPosInt(Sqrt(nt.params[1]) / nt.getTag(1).k),
#								
#			children := nt -> let(
#				k := nt.getTag(1).k,
#				m := nt.getTag(1).m,
#				n := Sqrt(nt.params[1]),
#				
#				[[
#					TL(n*k, k, n/k, 1).withTags(Drop(nt.getTags(), 1)) ,
#					TL(n*n/k, n/k, 1, k).withTags(Drop(nt.getTags(), 1))
#				]]
#			),
#			
#			apply := (nt, c, cnt) -> c[1] * c[2]
#		),
#	# Ln^2_n_{write} -> (L^n^2/k_n x Ik)(In/k x L^nk_n)
#		L_tileWr := rec(
#			applicable := nt -> nt.hasTags()
#								and _isATileWrTag(nt)
#								and _isTL_L(nt)
#								and IsPosInt(Sqrt(nt.params[1]))
#								and IsPosInt(Sqrt(nt.params[1]) / nt.getTag(1).k),
#			
#			children := nt -> let(
#				k := nt.getTag(1).k,
#				m := nt.getTag(1).m,
#				n := Sqrt(nt.params[1]),
#				
#				[[
#					TL(n*n/k, n, 1, k).withTags(Drop(nt.getTags(), 1)),  
#					TL(n*k, n, n/k, 1).withTags(Drop(nt.getTags(), 1))
#				]]
#			),
#			
#			apply := (nt, c, cnt) -> c[1] * c[2]
#				
#		)
#	));
#	





