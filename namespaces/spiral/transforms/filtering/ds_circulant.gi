
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# DSCirculant(<n>, <P>, <v>, <ds>, <ofs>)
#    Downsampled circulant
#
Class(DSCirculant, Circulant, rec(
    abbrevs := [ (n, P, v, ds, ofs) -> 
	    [ Checked(IsPosInt(n), n),
	      toFunc(P),
	      Checked(IsInt(v), v),
	      Checked(IsPosInt(ds), ds),
	      Checked(IsPosInt0(ofs), ofs < ds, ofs) ]
    ],

    dims := self >> let(p:=self.params, n:=p[1], ds:=p[4], ofs:=p[5],
	[ FloorRat((n+ofs)/ds), n ]),

    terminate := self >> let(cterminate := Circulant.terminate,
        DownSample(self.params[1], self.params[4], self.params[5]) * 
        cterminate(self)),

    transpose := NonTerminal.transpose,
  
    HashId := self >> let(p := self.params, [p[4], p[1], p[2].domain(), p[3]])
));

# since Circulant is already added, DSCirculant won't be added automatically
AddNonTerminal(DSCirculant); 

## 
## Rules
##
RulesFor(DSCirculant, rec(
    ###################################################################
    ## Time domain methods
    ###################################################################

    #F RuleCirculant_Base: (base case)
    #F
    DSCirculant_Base := rec (
	info             := "DSCirculant -> Mat",
	forTransposition := false,
	isApplicable     := P -> P[1] <= d_sqrt(SPL_DEFAULTS.globalUnrolling),
	allChildren      := P -> [[ ]],
	rule := (P, C) -> ApplyFunc(DSCirculant, P).terminate()
    ),

    DSCirculant_toFilt := rec (
	info             := "DSCirculant -> VStack(.., Filt, ...)",
	forTransposition := false,
	isApplicable     := P -> P[1] > 2 and (P[1]-P[2].domain()) > 2 and P[3] <= 0, 
	allChildren      := P -> [[ ]], 

	rule := function(P, C)
	    local n,l,r,coeffs,nc,ds,ofs,ofs_l,ofs_r,ofs_filt,ds_l,ds_r,ds_filt,j,jj,k,kk,v,
	         filtpoly,filtcoef,filtlen,f;

	    n := P[1];  coeffs := P[2]; nc := coeffs.domain();
	    l := -P[3]; r := -l + nc - 1; 

	    ds := P[4]; ofs := P[5];

	    ofs_l := ofs;
	    ofs_filt := (l - ofs_l) mod ds;
	    ofs_r := (n-l-r - ofs_filt) mod ds; 

	    ds_l    := CeilingRat((l - ofs_l) / ds);
	    ds_filt := CeilingRat((n-l-r - ofs_filt) / ds);
	    ds_r    := CeilingRat((r - ofs_r) / ds);

	    j := Ind( ds_l ); 
	    k := Ind( ds_r );
	    f := Ind( ds_filt );
	    v := Ind(nc);
	    jj := Ind(l);
	    kk := Ind(r);

	    filtpoly := Poly(List(coeffs.tolist(),EvalScalar), -l);
	    filtcoef := FillZeros(filtpoly)[1];
	    filtlen := Length(filtcoef);

	    return 
	    VStack(
	        BB(ISum(j, j.range, 
		     Data(jj, ds*j+ofs_l,
			 Scat(fBase(j)) * 
			 RowVec(fCompose(coeffs, Lambda(v, cond(leq(v,r+jj), v-jj+l, v-jj+l-nc)))) *
			 Gath(n, nc, Lambda(v, cond(leq(v,r+jj), v, v+(n-l-r)-1)))))),

		ISum(f, f.range,
		    Scat(fBase(f)) *
		    Blk([ filtcoef ]) *
		    Gath(fAdd(n, filtlen, ds*f + ofs_filt))),

		BB(ISum(k, k.range, 
		     Data(kk, ds*k+ofs_r,
			 Scat(fBase(k)) *
			 RowVec(fCompose(coeffs, Lambda(v, cond(leq(v,kk), v-kk+(nc-1), v-kk-1)))) *
			 Gath(n, nc, Lambda(v, cond(leq(v,kk), v, v+(n-l-r)-1))))))
	    );
	end
    )
));
