
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Import(approx);

SubMat := (mat, ri, ci, rsize, csize) -> let(
    d := DimensionsMat(mat),
    nrows := d[1],
    ncols := d[2],    
    List(mat{[ri .. Min2(ri+rsize-1, nrows)]}, 
	row -> row{[ci .. Min2(ci+csize-1, ncols)]})
);

GetBlocksMat := (mat, rblk, cblk) -> Checked(IsPosInt(rblk), IsPosInt(cblk), let(
    d := DimensionsMat(mat),
    nrows := d[1],
    ncols := d[2],    
    List([0 .. CeilingRat(nrows/rblk)-1], rb -> 
	List([0 .. CeilingRat(ncols/cblk)-1], cb -> 
	    SubMat(mat, 1+rb*rblk, 1+cb*cblk, rblk, cblk)))
));

# Cyclic diagonal vectorization:
VectorizedZMatSPL := ( isa, spl ) -> Checked( Rows(spl)=Cols(spl) and isa.v = Rows(spl), let(
    mat := Cond(IsMat(spl), spl, MatSPL(spl)),
    VBlk( [List( [0 .. isa.v-1], j -> isa.t.value(
                List( [0 .. isa.v-1], i -> mat[1 + i][1 + (i-j) mod isa.v])
         ))], 
         isa.v)
));

# Straight diagonal vectorization:
VectorizedDMatSPL := ( isa, spl ) -> Checked(Rows(spl) <= isa.v, let(
    mat   := Cond(IsMat(spl), spl, MatSPL(spl)),
    r     := Rows(mat),
    c     := Cols(mat),
    padd  := (offs, l) -> Replicate(offs, 0) :: l :: Replicate(isa.v - offs - Length(l), 0),
    VBlk( [ List( [0 .. r-2], i -> isa.t.value(padd(r-i-1, List( [0 .. _unwrap(min(i,c-1))], j -> mat[r-i+j][j+1] )))) ::
            List( [0 .. c-1], i -> isa.t.value(padd(    0, List( [0 .. When(i+r<=c, r-1, c-i-1)], j -> mat[j+1][i+j+1] )))) ], 
         isa.v)
));

# Zero pad both dimensions:
ZeroPadMat := (mat, v) -> let(
    r   := Rows(mat),
    c   := Cols(mat),
    dr  := CeilingRat(r/v)*v - r,
    dc  := CeilingRat(c/v)*v - c,
    List( mat, row -> row :: Replicate(dc, 0) ) :: Replicate(dr, Replicate(c+dc, 0))
);

# Zero pad matrix by adding zero rows:
ZeroPadMatV := (mat, v) -> let(
    r   := Rows(mat),
    c   := Cols(mat),
    dr  := CeilingRat(r/v)*v - r,
    mat :: Replicate(dr, Replicate(c, 0))
);

NewRulesFor(TMat, rec(
    TMat_Base := rec(
	applicable := (self, t) >> not t.hasTags(), 
        apply := (self, t, C, Nonterms) >> Mat(t.params[1])
    ),

    TMat_Vec := rec(
	forTransposition := false,
	maxSize := 128,
	applicable := (self, t) >> t.hasTags() and t.isTag(1, AVecReg) and 
	    (self.maxSize<0 or (Rows(t) <= self.maxSize and Cols(t) <= self.maxSize)),

        apply := (self, t, C, Nonterms) >> let(
	    isa := t.getTag(AVecReg, 1).isa,
	    VectorizedMatSPL(isa, t)
        )
    ),

    # vectorization of square matrix with dimensions equal to vector width using cyclic shifts
    TMat_VecZ := rec(
        forTransposition := false,
	applicable := (self, t) >> t.hasTags() and t.isTag(1, AVecReg) and 
	    Rows(t) = Cols(t) and Rows(t)=t.getTag(AVecReg).v,

        apply := (self, t, C, Nonterms) >> let( isa := t.getTag(AVecReg).isa, i := Ind(isa.v),
	    VectorizedZMatSPL(isa, t)
	     * ISum(i, VScat(fBase(isa.v, i), isa.v) * VPerm(Z(isa.v,-i), isa.rotate_left(i), isa.v, 1))
	     * VGath(fId(1), isa.v)
        )
    ),

    # Vectorization of non square matrix by diagonals
    TMat_VecD := rec(
        forTransposition := false,
        maxSize := 256,
	applicable := (self, t) >> t.hasTags() and t.isTag(1, AVecReg) and 
	    Rows(t) <= t.getTag(AVecReg).v and (self.maxSize<0 or Cols(t) <= self.maxSize),

        apply := (self, t, C, Nonterms) >> let( 
            isa := t.getTag(AVecReg).isa,
            v   := isa.v,
            r   := Rows(t),
            c   := Cols(t),
            ru  := _roundup(c + v-1, v),
            i   := Ind(r-1),
            j   := Ind(c),
            VScat_pc( r, r, 0, v) *
            VectorizedDMatSPL(isa, t) *
            SUM(
                When(r>1, 
                [ ISum(i, 
                    VScat(fBase(r+c-1,     i), v) * VPerm(Z(v,v-r+i+1), isa.rotate_left(r-i-1), v, 1) 
                     * VGath_pc(c, min(c, v), 0, v))], []) ::
                [ISum(j, 
                    VScat(fBase(r+c-1, j+r-1), v) 
                     * VGath_pc(c, min(v, c-j), j, v)).unrolledChildren()]
            )
        )
    ),

    TMat_UnrolledBlkH := rec(
        forTransposition := false,
	applicable := (self, t) >> t.hasTags() and t.isTag(1, AVecReg) and 
	    Rows(t) > t.getTag(AVecReg).v,

        freedoms := (self, t) >> [[t.getTag(AVecReg).v]],

	child := (self, t, fr) >> Flat(MapMat(GetBlocksMat(ZeroPadMatV(t.params[1], t.getTag(AVecReg).v), fr[1], Cols(t)), e -> TMat(e).withTags(t.getTags()))),

        apply := (self, t, C, Nonterms) >> let( 
            isa := t.getTag(AVecReg).isa,
            v   := isa.v,
            c   := Cols(t),
            r   := Rows(t),
            VScat_pc(r, r, 0, v) * SUM( 
                List( [1..Length(C)], i -> VScat(fAdd(Length(C), 1, i-1), v) * C[i] ))
        )
    ),

    TMat_UnrolledBlkV := rec(
        forTransposition := false,
	applicable := (self, t) >> t.hasTags() and t.isTag(1, AVecReg) and 
	    Cols(t) > t.getTag(AVecReg).v,

        freedoms := (self, t) >> [[1]],

	child := (self, t, fr) >> let(
	    v := t.getTag(AVecReg).v,
	    m := ZeroPadMat(t.params[1], v),
	    Flat(MapMat(GetBlocksMat(m, Rows(m), v), e -> TMat(e).withTags(t.getTags())))
	),

        apply := (self, t, C, Nonterms) >> let( 
            isa := t.getTag(AVecReg).isa,
            v   := isa.v,
            c   := Cols(t),
            r   := Rows(t),
            h   := CeilingRat(r/v),
            VScat_pc(r, r, 0, v) * 
            VBlk( MapMat(TensorProductMat( [Replicate( CeilingRat(c/v), 1)], IdentityMat(h) ), e -> When(e=0, isa.t.zero(), isa.t.one())), v)*
            SUM( 
                List( [1..Length(C)], i -> VScat(fAdd(Length(C)*h, h, (i-1)*h), v) * C[i] * VGath(fAdd(Length(C), 1, i-1), v) ))*
            #HStack(C) *
            VGath_pc(c, c, 0, v)
        )
    ),

    TMat_Blk := rec(
    	forTransposition := false,

    	applicable := (self, t) >> t.hasTags() and t.isTag(1, AVecReg) 
	    and (Rows(t) > self.rblk or Cols(t) > self.cblk),

	# Matrix partition 
	# N - normal block (rblk x cblk), T - tall blocks, S - short blocks, C - corner bock
	#
	# N ... N T
	# . \   . T
	# .   \ . T
	# N ... N T
        # S S S S S
	#
	_NTSC_Split := meth(self, mat, rblk, cblk) 
	    local r, c, nr, nc, N, T, S, C;
	    [ r,  c] := Dimensions(mat);
	    [nr, nc] := [r - r mod rblk,  c - c mod cblk];
	    N := mat{[1..nr]}{[1..nc]};
	    T := mat{[1..nr]}{[nc+1..c]};
	    S := mat{[nr+1..r]}{[1..nc]};
	    C := mat{[nr+1..r]}{[nc+1..c]};
	    return [N, T, S, C];
	end,
	    
        apply := meth(self, t, C, Nonterms) 
	    local isa, N, T, S, C, blks, dat, dvar, i, j;

    	    isa := t.getTag(AVecReg, 1).isa;

	    [N, T, S, C] := self._NTSC_Split(t.params[1], self.rblk, self.cblk);
	    blks := rec(
		N := MapMat(GetBlocksMat(N, self.rblk, self.cblk), x -> VectorizedMatSPL(isa, x)),
		T := MapMat(GetBlocksMat(T, self.rblk, self.cblk), x -> VectorizedMatSPL(isa, x)),
		S := MapMat(GetBlocksMat(S, self.rblk, self.cblk), x -> VectorizedMatSPL(isa, x))
	    );
	    dat := rec(
		N := V(List(Collect(blks.N, VBlk), x->x.element)),
		T := V(List(Collect(blks.T, VBlk), x->x.element)),
		S := V(List(Collect(blks.S, VBlk), x->x.element))
	    );
	    dvar := rec(
		N := var.fresh_t("DN", dat.N.t),
		T := var.fresh_t("DT", dat.T.t),
		S := var.fresh_t("DS", dat.S.t)
	    );

	    [i, j] := [Ind(Rows(blks.N)), Ind(Cols(blks.N))];

	    return
	    Data(dvar.N, dat.N, 
	    Data(dvar.T, dat.T, 
	    Data(dvar.S, dat.S, 
		VStack(
		    IterVStack(i, 
			HStack(
			    IterHStack(j, 
				BB(SubstTopDownNR(blks.N[1][1], VBlk, x->VBlk(nth(dvar.N, i*Cols(blks.N)+j), isa.v)))
			    ),
			    BB(SubstTopDownNR(blks.T[1][1], VBlk, x->VBlk(nth(dvar.T, i), isa.v)))
			)
		    ),
		    HStack(
			IterHStack(j, 
			    BB(SubstTopDownNR(blks.S[1][1], VBlk, x->VBlk(nth(dvar.S, j), isa.v)))
			),
			VectorizedMatSPL(isa, C)
		    )
		)
	    )));
	end
    )
));

