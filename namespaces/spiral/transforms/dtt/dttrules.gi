
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Local.qmat := Mat(
    [[1,1,1],
     [1,-1,0],
     [1,0,-1]]
);
Local.wmat := Mat(
    [[1,0,1],
     [0,1,0],
     [0,0,1]]);

## C3 = DCT-3
## C4 = DCT-4
## S3 = DST-3
## S4 = DST-4

RulesFor(DTT, rec(
    #
    # DCT-3
    #
    DTT_C3_Base2 := SimpleRule(
	[ DTT_C3, 2, @, @ ], 
	(P, C) -> F(2) * Diag(1, cospi(P[3]))
    ),
    DTT_IC3_Base2 := SimpleRule(
	[ DTT_IC3, 2, @, @ ], 
	(P, C) -> Diag(1, 1/(2*cospi(P[3]))) * F(2)
    ),
    DTT_C3_Base3 := SimpleRule(
	[ DTT_C3, 3, @, @ ], 
	(P, C) -> let(r:=P[3],
	    qmat * DirectSum(I(1), Mat([[ cospi((1+r)/3),  cospi((1-2*r)/3) ],
			             [ cospi((1-r)/3),  cospi((1+2*r)/3) ]])))
    ),
    DTT_C3_Base3 := SimpleRule(
	[ DTT_C3, 3, @, @ ], 
	(P, C) -> let(r:=P[3],
	    DirectSum(I(1), 
		      1/sinpi(r) *
		      Mat([[ cospi(( 1-2*r)/6),  sinpi((-1+2*r)/3) ],
			   [ cospi((-1+2*r)/6),  cospi(( 1-2*r)/3) ]]) *
		      Mat([[0, sqrt(3)/2],
			   [1, -1/2]]))
	    * qmat)
    ),

    #
    # DST-3
    #
    DTT_S3_Base2 := SimpleRule(
	[ DTT_S3, 2, @, false ], 
	(P, C) -> let(r:=P[3],  F(2) * Diag(sinpi(r/2), sinpi(r)))
    ),
    DTT_S3Poly_Base2 := SimpleRule(
	[ DTT_S3, 2, @, true ], 
	(P, C) -> let(r:=P[3],  F(2) * Diag(1, 2*cospi(r/2)))
    ),
    DTT_S3Poly_Base3 := SimpleRule(
	[ DTT_S3, 3, @, true ], 
	(P, C) -> let(r:=P[3],  
	    qmat * DirectSum(I(1), 2* Mat([[ cospi((1+r)/3),  cospi((1-2*r)/3) ],
	       	                           [ cospi((1-r)/3),  cospi((1+2*r)/3) ]]) * wmat))
    ),
    DTT_S3_Base2 := SimpleRule(
	[ DTT_S3, 2, @, false ], 
	(P, C) -> let(r:=P[3],  F(2) * Diag(sinpi(r/2), sinpi(r)))
    ),

    #
    # DCT-4
    #
    DTT_C4_Base2 := SimpleRule(
	[ DTT_C4, 2, @, false ], 
	(P, C) -> let(r:=P[3],  Mat([[cospi(r/4), cospi(3*r/4)], 
		                     [sinpi(r/4), -sinpi(3*r/4)]]))
    ),
    DTT_C4_Base2_LS := SimpleRule(
	[ DTT_C4, 2, @, false ], 
	(P, C) -> let(r:=P[3],  Diag(cospi(r/4), sinpi(r/4)) * F(2) * Mat([[1,-1],[0,2*cospi(r/2)]]))
    ),
    DTT_C4Poly_Base2 := SimpleRule(
	[ DTT_C4, 2, @, true ], 
	(P, C) -> let(r:=P[3],  F(2) * Mat([[1,-1],[0,2*cospi(r/2)]]))
    ),

    #
    # DST-4
    #
    DTT_S4_Base2 := SimpleRule(
	[ DTT_S4, 2, @, false ], 
	(P, C) -> let(r:=P[3],  Mat([[sinpi(r/4), sinpi(3*r/4)], 
		                     [cospi(r/4), -cospi(3*r/4)]]))
    ),
    DTT_S4_Base2_LS := SimpleRule(
	[ DTT_S4, 2, @, false ], 
	(P, C) -> let(r:=P[3],  Diag(sinpi(r/4), cospi(r/4)) * F(2) * Mat([[1,1],[0,2*cospi(r/2)]]))
    ),
    DTT_S4Poly_Base2 := SimpleRule(
	[ DTT_S4, 2, @, true ], 
	(P, C) -> let(r:=P[3],  F(2) * Mat([[1,1],[0,2*cospi(r/2)]]))
    ),

    #
    # Recursive rules
    #
    DTT_DIF_T := rec(
	isApplicable     := P -> not IsPrime(P[2]) and P[2] > 2,
	forTransposition := false,

	baseChange := (self, P, m) >> let(t := P[1], mn := P[2], 
	    Cond(Same(t, DTT_C3), B_DCT3_U(mn, m),
		 Same(t, DTT_C4), B_DCT4_U(mn, m),
		 Same(t, DTT_S3), B_DST3_U(mn, m),
		 Same(t, DTT_S4), B_DST4_U(mn, m))),

	recursType := (self, t) >> 
	    Cond(Same(t, DTT_C3), DTT_S3,
		 Same(t, DTT_C4), DTT_S3,
		 Same(t, DTT_S3), DTT_S3,
		 Same(t, DTT_S4), DTT_S3),

	allChildren      := (self,P) >> let(
	    type := P[1], MN := P[2], r := P[3],  poly := P[4],
	    divs := DivisorPairs(MN),
	    List(divs, d -> let(j := Ind(d[2]),
		[ DTT(type,                     d[1], fr(d[2], j, r), poly), 
		  DTT(self.recursType(type),    d[2], r,              true) ]))),
	
	rule := (self, P, C, Nonterms) >> let(MN := P[2], N := Rows(C[1]), M := Rows(C[2]), 
	                 j := Nonterms[1].params[3].args[2],
	    K(MN, N) * 
	    IterDirectSum(j, j.range, C[1]) *
	    Tensor(C[2], I(N)) *
	    self.baseChange(P, M)
	)
    )
));

