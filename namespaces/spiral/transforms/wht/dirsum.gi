
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# This rule seems to be unnecessary
#

RulesFor(WHT, rec(
    #F WHT_Dirsum: switched off...
    #F
    #F see tensor products as direct sums so that they can be 
    #F split separately
    #F
    #F   WHT_(2^k) = 
    #F     prod_(i = 1)^r 
    #F       dirsum_(a = 1)^2^(k_1 + .. + k_(i-1))
    #F         ( dirsum_(b = 1)^2^(k_(i+1) + .. + k_r) WHT_(2^(k_i))) ^ 
    #F           L^(2^(k_i + .. + k_r))_(2^(k_(i+1) + .. + k_r))
    #F         )
    #F
    WHT_Dirsum := rec (
	info             := "WHT_(2^k) -> prod dirsum (dirsum WHT_(2^ki))^L",
	forTransposition := false,
	switch           := false,
	isApplicable     := P -> P <> 1,
	isDerivable := meth ( self, S, C )
	    local C1, i, c, j;
	    if not (IsApplicableRule(self, S) and
		    ForAll(C, c -> ObjId(c) = WHT)) then return false; fi;
	    i  := 1;
	    C1 := [ ];
	    while i <= Length(C) do
	        c := C[i];
		for j in [i..i + 2^(S.params - c.params) - 1] do
		    if c.params <> C[j].params then
			return false;
		    fi;
		od;
		Add(C1, c);
		i := i + 2^(S.params - c.params);
	    od;
	    return Sum(C1, c -> c.params) = S.params;
	end,

	allChildren := function ( P )
	    local C, C1, p, c, i, k;
	    C := OrderedPartitions(P);
	    Unbind(C[Length(C)]);
	    C1 := [ ];
	    for p in C do
	         c := [ ];
		 for i in p do
		     for k in [1..2^(P - i)] do Add(c, WHT(i)); od;
		 od;
		 Add(C1, c);
	    od;
	    return C1;
	end,

	randomChildren := function ( P )
	    local C, sum, rand;
	    C   := [ ];
	    sum := 0;
	    while sum < P do
	        if sum = 0 then	rand := RandomList( [1..(P-1)] );
		else		rand := RandomList( [1..(P-sum)] );
		fi;
		Add( C, rand );
		sum := sum + rand;
	    od;
	    return Concatenation(List(C, c -> List([1..2^(P - c)], i -> WHT(c))));
	end,

	rule := function ( P, C )
	    local C1, c, left, right, factors, i;
	    i  := 1;
	    C1 := [ ];
	    while i <= Length(C) do
	        c := C[i];
		Add(C1, c);
	        i := i + 2^P/c.dimensions[1];
	    od;

	    left    := 1;
	    right   := 2^P / C1[1].dimensions[1];
	    factors := [ ];
   
            # first factor
	    Add(factors,
		Conjugate( 
		    DirectSum(List([1..right], l -> C1[1])),
		    L(2^P, right))
	    );

	    left  := left * C1[1].dimensions[1];
	    right := right / C1[2].dimensions[1];
	    
            # middle factors
	    for i in [2..Length(C1) - 1] do
  	        Add(factors, DirectSum(  
		    List([1..left],
			k -> Conjugate(
			    DirectSum(List([1..right], l -> C1[i])),
			    L(right * C1[i].dimensions[1], right))
		    )
		));
		left  := left * C1[i].dimensions[1];
		right := right / C1[i+1].dimensions[1];
	    od;

            # last factor
	    Add(factors, DirectSum(List([1..left], k -> C1[Length(C1)])));
	    return Compose(factors);
	 end
    )
));
