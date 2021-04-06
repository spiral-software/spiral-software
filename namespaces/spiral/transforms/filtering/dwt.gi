
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(DWT, NonTerminal, DataNonTerminalMixin, rec(
    abbrevs := [ 
       (n,w) -> Checked(IsPosInt(n), IsEvenInt(n), n>=2, IsList(w), Length(w)=2, 
	   [n, toFiltFunc(w[1]), toFiltFunc(w[2])]),
       (n,g,h) -> Checked(IsPosInt(n), IsEvenInt(n), n>=2, 
	   [n, toFiltFunc(g), toFiltFunc(h)]),
    ], 
    isReal := self >> true,

    setData := meth(self, gdata, hdata) 
       self.params[2][1] := gdata; self.params[3][1] := hdata; return self; 
    end,

    dims := self >> let(n:=self.params[1], k:=self.params[2][1].domain(),
                        [n, n + k - 2]), 

    terminate := self >> let(
	n := self.params[1], 
	g := self.params[2],
	h := self.params[3],
	VStack(Filt(n/2,g[1],g[2],2).terminate(), Filt(n/2,h[1],h[2],2).terminate())),

    HashId := self >> [ self.params[1], self.params[2][1].domain() ]
));

KnownCoeffs := filtfunc -> ObjId(filtfunc[1])=FList or
                           (ObjId(filtfunc[1])=FData and IsBound(filtfunc[1].var.value));

RulesFor(DWT, rec(
    #F DWT_Base: (base case)
    #F
    #F Computes DWT by definition
    #F 
    DWT_Filt := rec(
	forTransposition := true,
	limit            := -1, # means no limit
	isApplicable     := (self, P) >> When(self.limit < 0, true,
	    P[1] <= self.limit or P[2][1].domain() <= self.limit),

	allChildren := P -> let(n := P[1], g := P[2], h := P[3],
	    [[ Filt(n/2,g[1],g[2],2), Filt(n/2,h[1],h[2],2) ]]),

	rule := (P, C) -> VStack(C[1], C[2])
    ),

    DWT_Base2 := rec(
	isApplicable := P -> true,
	rule := (P, C) -> let(n := P[1], g := P[2][1].tolist(), h := P[3][1].tolist(),
	    diff := Length(h) - Length(g),
	    gg := Cond(diff <= 0, g, Concatenation(g, Replicate(diff,0))),
	    hh := Cond(diff >= 0, h, Concatenation(h, Replicate(-diff,0))),
	    k := Maximum(Length(g), Length(h)),
	    When(n=1, Mat([gg,hh]), L(n,2)*RowTensor(n/2,k-2,Mat([gg,hh]))))
    ),

    DWT_Base4 := rec(
	isApplicable := P -> P[1] mod 4 = 0,
	rule := (P, C) -> let(n := P[1], g := P[2][1].tolist(), h := P[3][1].tolist(),
	    diff := Length(h) - Length(g),
	    gg := Cond(diff <= 0, g, Concatenation(g, Replicate(diff,0))),
	    hh := Cond(diff >= 0, h, Concatenation(h, Replicate(-diff,0))),
	    k := Maximum(Length(g), Length(h)),
	    When(n=1, Mat([gg,hh]), L(n,2)*RowTensor(n/4,k-2,BB(RowTensor(2,k-2,Mat([gg,hh]))))))
    ),
    DWT_Base8 := rec(
	isApplicable := P -> P[1] mod 8 = 0,
	rule := (P, C) -> let(n := P[1], g := P[2][1].tolist(), h := P[3][1].tolist(),
	    diff := Length(h) - Length(g),
	    gg := Cond(diff <= 0, g, Concatenation(g, Replicate(diff,0))),
	    hh := Cond(diff >= 0, h, Concatenation(h, Replicate(-diff,0))),
	    k := Maximum(Length(g), Length(h)),
	    When(n=1, Mat([gg,hh]), L(n,2)*RowTensor(n/8,k-2,BB(RowTensor(4,k-2,Mat([gg,hh]))))))
    ),

    #F DWT_Lifting: lifting steps, does not work yet
    #F 
    DWT_Lifting := rec(
        info             := "DWT(n,g,h) -> Lifting steps",
        forTransposition := false,
        isApplicable     := P -> P[1]>2 and P[1] mod 2 =0 and KnownCoeffs(P[2]),
	switch := false,
        allChildren := function ( P )
            local n,g,h,vg,vh, lift_schemes,children,scheme,step,filts,pol;
            n       := P[1];
	    [g, h]  := [List(P[2][1].tolist(), EvalScalar), List(P[3][1].tolist(), EvalScalar)];
	    [vg,vh] := [P[2][2], P[3][2]];

            lift_schemes := Copy(HashLookupWav(HashTableWavelets, [[g,h],[vg,vh]]));
	    children := [];

            for scheme in lift_schemes do 
	        filts := [];
	        for step in Drop(scheme,1) do
		    pol := FillZeros(ListPoly(step));
		    Add(filts, Filt(n/2, pol[1], pol[2]));
		od;
                # attach the indicator of the type of the first liftings step
		filts[1].lift := scheme[1];
		Add(children, filts);
            od;
            return children;
        end,

        rule := function ( P, C, Nonterms )
            local n, i, ind, b, l, M, ind0, ind1, lstep;
            n := P[1];          
            b := Nonterms[1].lift;
            l:=Length(C);

	    ind0 := fTensor(fBase(2,0), fId(Rows(C[3])));
	    ind1 := fTensor(fBase(2,1), fId(Rows(C[3])));
	    lstep := (f,b) -> When(b=0, 
		LStep(Scat(ind0) * f * Gath(ind1)),
		LStep(Scat(ind1) * f * Gath(ind0)));

            M := When(b < 0, 
		SUM(Scat(ind0)*C[2]*Gath(ind1), Scat(ind1)*C[1]*Gath(ind0)),
		SUM(Scat(ind0)*C[2]*Gath(ind0), Scat(ind1)*C[1]*Gath(ind1)));

	    b := (b+2) mod 2; # make b positive
	    for i in [1 .. l-2] do
	        M := M * lstep(C[i+2], b);
		b := (b+1) mod 2;
            od; 

	    return M.child(1) * Inplace(Compose(Drop(M.children(),1))) * L(n,2);
       end
    )
));

