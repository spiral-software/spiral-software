
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


##########################################################################################
#   tSPL Pease DFTDR rule
NewRulesFor(DFTDR, rec(
    # Pease tSPL DFT_(k)*DR(k,r) -> \Prod(L(I tensor F)Tc)
    DFTDR_tSPL_Pease   := rec (
        forTransposition := true,
        minSize := 2,

	# set to true to wrap TC in fPrecompute (i.e., for software)
	# leave as false to allow TC to be simplified by other rules
	# (i.e., for hardware)
	precompute := false,

        applicable := (self, t) >> let(
            tags := t.getTags(), n := t.params[1], k := t.params[2], radix := t.params[3], 
            stream := Cond(Length(tags)=1 and IsBound(tags[1].bs), tags[1].bs, -1),
               Length(tags) >= 1    and 
               (IsSymbolic(n)    or ((n >= self.minSize) and IsIntPower(n, radix)))
	       and IsInt(LogInt(n, radix) / self.unroll_its)
#           and k = 1
        ),

        freedoms := t -> [[ t.params[3] ]], # radix is the only degree of freedom

	# Used in HW to have multiple stages inside the TICompose.  For "normal"
	# operation, keep this at 1.
	unroll_its := 1,

        child := (self, t, fr) >> let(
            n     := t.params[1],
            radix := fr[1],
            e     := t.params[2],
            tags  := t.getTags(),
            j     := Ind(LogInt(n, radix)/self.unroll_its),
            stage := TTensorI(DFT(radix, e), n/radix, AVec, APar),
            fPre  := Cond(self.precompute, fPrecompute, x->x), 
	    twid  := i >> TDiag(fPre(TC(n, radix, self.unroll_its*j+i, e))),
	    full_stage := List([1..self.unroll_its], t->(TCompose([stage, twid(t-1)]))),
	    
            [ TICompose(j, j.range, TCompose(full_stage)).withTags(tags) ]
        ),

        apply := (t, C, Nonterms) -> C[1]
    )
));

# it is a transpose, Spiral will transpose automatically
DRDFT_tSPL_Pease := DFTDR_tSPL_Pease;
