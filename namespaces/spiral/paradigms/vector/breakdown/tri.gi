
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


NewRulesFor(TRDiag, rec(
    # TRDiag is a tall matrix.
    # TRDiag.transpose() is a wide matrix
    #
    TRDiag_Vec := rec(
        forTransposition := true,
	requiredFirstTag := [AVecReg, AVecRegCx],
        applicable := t -> not AnySyms(t.params[1], t.params[2]), 

        apply := (t, C, Nonterms) -> let(
            v := t.firstTag().v,
            N := t.params[1],
            n := t.params[2],
            nv := _roundup(n, v),

            VScat_zero(N/v, nv/v, v) *
	    VDiag(fStretch(t.params[3], nv, n), v) *
	    VGath_pc(n, n, 0, v)   
        )
    ),

    TRDiag_VecN := rec(
        forTransposition := true,
	requiredFirstTag := [AVecReg, AVecRegCx],
        applicable := t -> true, #AnySyms(t.params[1], t.params[2]), 

        apply := (t, C, Nonterms) -> let(
            v := t.firstTag().v,
            N := t.params[1],
            n := t.params[2],
            ndiv := idiv(n, v), 
	    nmod := imod(n, v),
	    nmodrem := imod(v-nmod, v),
	    j := Ind(ndiv),

	    # * NOTE: redundant zero-padding operations when transposing the below
	    #   VGath_zero/VScat_zero type operations fix this
	    # * Virtual pad seems to be a hack
	    # * O.toloop() should be implemented, VO is missing
	    # * Middle summand is super ugly, use ScatGath?
	    #
	    SUM(
		VirtualPad(N, n, 
		    ISum(j, j.range,
			VScat(HH(N/v, ndiv, 0, [1]), v) *
			VDiag(fCompose(t.params[3], fAdd(n, ndiv*v, j*v)), v) *
			VGath(HH(n/v, ndiv, 0, [1]), v))),

		# Below could be simplified, because the block completely disappears
		# when nmod==0
		Scat(HH(N, nmod+nmodrem, ndiv*v, [1])) *
		VStack(
		    Diag(fCompose(t.params[3], fAdd(n, nmod, ndiv*v))).toloop(1),
		    O(nmodrem, nmod) # XXX NOTE: should have O.toloop() here 
		) *
		Gath(HH(n, nmod, ndiv*v, [1])),
		
                VirtualPad(N, n, 
		    VScat_zero(idiv(N,v), idiv(N,v)-ndiv, 0, v)) # XXX NOTE: VScat_zero needs to take 4th parameter (here it is 2nd) which is the scatter offset. This code won't work until VScat_zero is patched.
	    )
        )
    )
 ));
