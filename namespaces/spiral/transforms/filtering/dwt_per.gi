
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(DWTper);

#F DWTper(<n>, <j>, [<h(z)>,<g(z)>])
#F DWTper(<n>, <j>, <L>, <V>)
#F   returns a 2-channel <j>-stage discrete wavelet transform with periodic extensions 
#F   (circulant transforms) where the 
#F   low pass and the high pass analysis filters given by polynomials 
#F   [<h(z)>,<g(z)>] 
#F
#F   <L> = [<Lh>,<Lg>] - coefficient lists for h and g
#F   <V> = [<vh>,<vg>] - valuation for h and g
#F
#F  <n> - size of the output of the reconstructed sequence
#F  <j> - number of filter bank stages
Class(DWTper, NonTerminal, rec(

  abbrevs := [
    function(n,j,M)
    local S,L,V, j;
    if IsList(M[1]) then
     S:=List([1,2], j->FillZeros([M[1][j],M[2][j]]));
     V:=List(S, j->j[2]);
     L:=List(S, j->j[1]);
     return [n,j,L,V];
    else 
    Checked(IsPosInt(n), Checked(ForAll(M,i->IsPolynomial(i)),true));
      S:=List(M, i-> FillZeros(i));
      V:=List(S, i->i[2]);
      L:=List(S, i->i[1]);
    return [n,j,L,V];
    fi;
    end,
    function(n,j,L,V)
    local S,V,L,i,j;
     S:=List([1,2], j->FillZeros([L[j],V[j]]));
     V:=List(S, j->j[2]);
     L:=List(S, j->j[1]);
     return [n,j,L,V];
    end
  ],

  dims := self >> [self.params[1], self.params[1]],

  half1 := self >> let(
      n  := self.params[1],
      l1 := self.params[3][1],
      v1 := self.params[4][1],
      DownSample(n,2,0) * Circulant(n,l1,v1).terminate() ),

  half2 := self >> let(
      n  := self.params[1],
      l2 := self.params[3][2], 
      v2 := self.params[4][2],
      DownSample(n,2,0) * Circulant(n,l2,v2).terminate() ),
        
  terminate := self >> let(
      n := self.params[1],
      j := self.params[2],
      L := self.params[3],
      V := self.params[4],               
      # DWT as filter bank stage + downsampling
      res := Cond(j = 1, I(n), 
	          DirectSum(DWTper(n/2,j-1,L,V).terminate(), I(n/2))) * 
             VStack(self.half1(), self.half2()),
      When(self.transposed, res.transpose(), res)
  ),

  LiftingScheme := self >> self.lifting(),

  lifting := meth(self)
               local n,j,LS,h,g,he,ho,ge,go;         
               n := self.params[1];          
               j := self.params[2];
               h := DownsampleTwo([self.params[3][1],self.params[4][1]]);
               g := DownsampleTwo([self.params[3][2],self.params[4][2]]);
               # Lifting scheme does not converge for wavelets longer than 9 
               # (need to be fixed)
               if (Maximum(List(h, i-> Length(i[1])))>9 or 
                   Maximum(List(g, i-> Length(i[1])))>9) then return [[]];fi;
               he := Poly(h[1][1],h[1][2]); 
               ho := Poly(h[2][1],h[2][2]); 
               ge := Poly(g[1][1],g[1][2]); 
               go := Poly(g[2][1],g[2][2]); 
               LS := LiftingScheme([[he,ho],[ge,go]]);
               return LS;
             end,

  isReal := True,
));


#F RuleFilter_DWT: (base case) DWT -> Mat,  
#F   Computes filter by definition
#F
RulesFor(DWTper, rec(
    #F RuleFilt_Mallat_2:
    #F 
    #F DWTper(n,j,[h,g])->DWTper(n/2,j-1,[h,g]),DWTper(n,1,[h,g]) 
    #F
    #F Mallat rule recursive (single stage DWT)
    #F The I(n/2) is inefficient because it copies the output
    #F However, this rule allows application of Lifting, POlyphase, etc. 
    #F on DWTper(n,1,[h,g])
    #F 
    DWTper_Mallat_2 := rec(
	info             := "DWTper(n,j,[h,g])->DWTper(n,j-1,[h,g])",
	forTransposition := false,
	isApplicable     := P -> P[1] > 2 and P[1] mod 2 =0 and P[2] > 1,

	allChildren := P -> [[ DWTper(P[1]/2,P[2]-1,P[3],P[4]), DWTper(P[1],1,P[3],P[4]) ]],

	rule := (P, C) -> DirectSum(C[1],I(P[1]/2))*C[2]
    ),

    #F DWTper_Mallat:
    #F 
    #F DWTper(n,j,[h,g])->DWTper(n/2,j-1,[h,g]), DSCirculant 
    #F 
    DWTper_Mallat := rec(
        info             := "DWTper(n,j,[h,g])->DWTper(n/2,j-1,[h,g])",
        forTransposition := false,
        isApplicable     := P -> P[1]>2 and P[1] mod 2 = 0,

        allChildren := P -> let(
	    n        := P[1],
	    dcirc1 := DSCirculant(n, P[3][1], P[4][1], 2, 0),
	    dcirc2 := DSCirculant(n, P[3][2], P[4][2], 2, 0),
	    When(P[2]=1, [[ dcirc1, dcirc2 ]],
		         [[ dcirc1, dcirc2, DWTper(P[1]/2, P[2]-1, P[3], P[4]) ]])),

         rule := (P, C) -> When(P[2]=1, VStack(C[1], C[2]),
                                        VStack(C[3]*C[1], C[2]))
    ),

    #F DWTper_Polyphase:
    #F 
    #F DWTper(n,1,[h,g])->[ [Circulant(he), Circulant(ho)],
    #F                      [Circulant(ge), Circulant(go)] ]
    #F
    #F Single-stage periodic DWT into a matrix of circulants of downsampled filters 
    #F 
    DWTper_Polyphase := rec(
	info             := "DWTper(n,1,[h,g]) -> [[Circ(he), Circ(ho)], [Circ(ge), Circ(go)]]",
	forTransposition := false,
	isApplicable     := P -> P[1]>2 and P[1] mod 2 =0 and P[2]=1,
	allChildren := function(P)
	    local n,h,g,he,ho,ge,go;         
	    n := P[1];          
	    h := DownsampleTwo([P[3][1], P[4][1]]);
	    g := DownsampleTwo([P[3][2], P[4][2]]);
	    he := Circulant(n/2, h[1][1], h[1][2]); 
	    ho := Circulant(n/2, h[2][1], h[2][2]);
	    ge := Circulant(n/2, g[1][1], g[1][2]);
	    go := Circulant(n/2, g[2][1], g[2][2]);
	    return [[ he, ho, ge, go ]];
	end,
	rule := (P, C) -> BlockMat( [[ C[1], C[2] ],
		                     [ C[3], C[4] ]] ) * L(P[1],2)
    ),

    #F RuleDWTper_Lifting:
    #F 
    DWTper_Lifting := rec(
        info             := "DWTper(n,1,[h,g]) -> Lifting steps",
        forTransposition := false,
        isApplicable     := ( L ) -> L[1]>2 and L[1] mod 2 =0 and L[2]=1,

        allChildren := function ( P )
            local n,j,LS,Lc,scheme,step,pol;         
            n := P[1];
            LS := Copy(HashLookupWav(HashTableWavelets, [P[3],P[4]]));
            Lc := List(LS, scheme->
                     List(scheme{[2..Length(scheme)]}, step->
                        let(pol := FillZeros(ListPoly(step)),
                            Circulant(n/2,pol[1],pol[2]))));

            for i in [1..Length(LS)] do 
             # attach the indicator of the type of the first liftings step
              Lc[i][1].lift :=LS[i][1];

             # fuse in the constants/shifts in the last lifting step
             #if (LS[i][1]=0 or LS[i][1]=-2) then LS[i][4]:=LS[i][4]*LS[i][3];
             #else  LS[i][4]:=LS[i][4]*LS[i][2];
             #fi;

             pol := FillZeros(ListPoly(LS[i][4]));
             Lc[i][3] := Circulant(n/2,pol[1],pol[2]);
            od;
            return Lc;
        end,

        rule := function ( P, C, Nonterms )
            local n, i, ind, b, l, M, first, last, ind0, ind1, lstep, last_ls;
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

	    first := M.child(1);
	    M := Inplace(Compose(Drop(M.children(),1)));
	    return first * M * L(n,2);
       end
    )
));
