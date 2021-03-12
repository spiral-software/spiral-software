
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# -----------------------------
# Special extension definitions
# ------------------------------

# periodic
Class(per, rec(
    __call__ := (self, k) >> WithBases(self, rec(n:=k, operations:=PrintOps)),
    print := self >> Print(self.name, "(", self.n, ")"),
    left  := (self, N) >> fAdd(N, self.n, N - self.n),
    right := (self, N) >> fAdd(N, self.n, 0),
));

# half-point symmetric
Class(hs, rec(
    __call__ := (self, k) >> WithBases(self, rec(n:=k, operations:=PrintOps)),
    print := self >> Print(self.name, "(", self.n, ")"),

    left  := (self, N) >> fCompose(fAdd(N, self.n, 0), J(self.n)),
    right := (self, N) >> fCompose(fAdd(N, self.n, N-self.n),J(self.n))
));

# whole-point symmetric
Class(ws, rec(
    __call__ := (self, k) >> WithBases(self, rec(n:=k, operations:=PrintOps)),
    print := self >> Print(self.name, "(", self.n, ")"),

    left  := (self, N) >> fCompose(fAdd(N, self.n, 1), J(self.n)),
    right := (self, N) >> fCompose(fAdd(N, self.n, N-self.n-1),J(self.n))
));

# zero padding - doing nothing, just returning the extension length
Class(zer, rec(
    __call__ := (self, k) >> WithBases(self, rec(n:=k, operations:=PrintOps)),
    print := self >> Print(self.name, "(", self.n, ")"),
    left  := (self, N) >> self.n,
    right := (self, N) >> self.n,
));



# ******************************
#  Operations on polynomials
# ******************************

ListPoly := (p) -> [p.coefficients, p.valuation];

ListSum := function(l1,l2)
  local i, d1, d2, dm, l;

  l := [];
  d1 := Length(l1);
  d2 := Length(l2);
  dm := Minimum(d1,d2);
  for i in [1..dm] do 
     l[i] := l1[i]+l2[i];od;
  if d2 > dm then Append(l,l2{[dm+1..d2]});
  else Append(l,l1{[dm+1..d1]});
  fi;
  return l;
end; 


ListPolyMat := function(M) 
   local S,L,V;
   S:=List(M, j-> ListPoly(j));
   L:=List(S, i->i[1]);
   V:=List(S, i->i[2]);

  return [L,V];
end;

Shift := function(h, sh)
local g;

if IsPolynomial(h) then
g := ShallowCopy(h);
g.valuation := h.valuation + sh;
return g;
else 
return [h[1],h[2]+sh];
fi;
end;

Upsample := function(h)
local i, L, g;
L:=[];
 g:=ShallowCopy(h);
 if Length(h.coefficients)=1 then
  Add(L,h.coefficients[1]);
 elif Length(h.coefficients)>1 then
  for i in [1..Length(h.coefficients)-1] do
    Add(L, h.coefficients[i]);
    Add(L, h.baseRing.zero);
  od;
  Add(L, h.coefficients[i+1]);
 fi;
  g.coefficients := L;
  g.valuation := h.valuation * 2;
  return g;
end;



# Downsample creates apolynomial of even coefficients
# h_e(z^2) =1/2*( h(z)+h(-z) )

Downsample := function(h)
local i, L, g, even, odd, v, l, k, coef;

 if IsPolynomial(h) then L:= ListPoly(h); coef:=L[1]; l:=L[2];
 else coef:=h[1]; l:=h[2]; fi;
 L:=[];
 k := l mod 2;
    even := Filtered([1..Length(coef)], i-> i mod 2 = 0);
    odd := Filtered([1..Length(coef)], i-> i mod 2 = 1);    

  if (k=0) then 
    L:= Sublist(coef,odd);
    v := l/2;
  else 
    L := Sublist(coef,even);
    v := Int((l+1)/2);
  fi;

 
  #g.coefficients := L;
  #g.valuation :=v;
  return [L,v];
end;


DownsampleTwo := function(h)
local he, ho;

if IsPolynomial(h) then h:=ListPoly(h); fi;
he:=Downsample(h);
ho:=Downsample(Shift(h,1));
if h[2] mod 2 = 0 then
ho:=Shift(ho,-1);
fi;

return[he,ho];
end;

DownsampleTwoNeg := function(h)
local he, ho;

he:=Downsample(h);
ho:=Downsample(Shift(h,-1));
if h[2] mod 2 = 0 then
ho:=Shift(ho,1);
fi;

return[he,ho];
end;



PosInt := (int) -> (int + AbsInt(int))/2;


#F Positive(n)
#F returns positive value of n thresholded at 0
#F
Positive := (n) -> When(n>0, n, 0);

#F Poly(L,v)
#F Creates a polynomial L[1]*z^(v+k-1) + ... + L[k]*z^v
#F

Poly := function(arg)
local z,L,v;
z:=Indeterminate(Cyclotomics);
z.name:="z";
L := When(IsList(arg[1]),arg[1],[arg[1]]);
v := When(Length(arg)<2,0,arg[2]); 
return Polynomial(Cyclotomics, L, v);
end;

#F PolyMat(L,V)
#F Creates a polynomial matrix out of coefficient and valuation matrices
#F

PolyMat := function(L,V)
local M,dim;
dim:=Dimensions(V);
M := List([1..dim[1]],i->List([1..dim[2]],j->Poly(L[i][j],V[i][j])));
return M;
end;



#F PolyExtension(p(z)) 
#F Computes extension lengths for poynomial p(z)
#F 
PolyExtension := p -> let(
    l := Positive(-p.valuation),
    r := Positive(LaurentDegree(p)+p.valuation),
    [l, r]
);

#F FillZeros := function(L,v)
#F fills zeros in the list of polynomial coefficients up to z^0
#F 
FillZeros := function(p)
local w,Ls,L,v;
if IsList(p) then
L:=p[1];
v:=p[2];
else
L:= p.coefficients;
v:= p.valuation;
fi;

w:= v+Length(L)-1;
Ls:= L;
if w<=0 then 
  Ls := Concat(L,List([1..-w], i->0));
  #k := -w;
elif v>=0 then
  Ls := Concat(List([1..v], i->0), L);
  #k := v;
fi;
return [Ls,-Positive(-v)];
end;

#F FillZerosMat := function(L,v)
#F fills zeros in the list of polynomial coefficients up to z^0
#F 

FillZerosMat := function(p)
local S,L,V,dim;

dim := Dimensions(p);
L:=List(p, i->List(i, j->  j.coefficients));
V:=List(p, i->List(i, j->  j.valuation));

S:=List([1..dim[1]],i->List([1..dim[2]],j-> FillZeros([L[i][j],V[i][j]])));
L:=List(S, i->List(i, j-> j[1])); 
V:=List(S, i->List(i, j-> j[2])); 

return [L,V];
end;


TimeReverse := (L,v) -> let(
               deg := Length(L),
               [Reversed(L),-(v+deg)+1]);


TimeReverseMat := function(arg) 
   local S,L,V,Ls,Vs,deg;

if Length(arg)=1 then L:=arg[1][1]; V:=arg[1][2]; 
else L:=arg[1]; V:=arg[2]; fi;

deg := Length(V);
S:=List([1..deg],i->TimeReverse(L[i],V[i]));
Ls:=List(S,i->i[1]);
Vs:=List(S,i->i[2]);
return [Ls,Vs];
end;


Alternate := (L) -> [List([1..Length(L[1])], i->(-1)^(L[2]+i-1)*L[1][i]),L[2]];
 

                         
Synth2AnalFilts := function(L,V) 
 
 local ht,gt,h,g;

 h:=TimeReverse(L[1],V[1]);
 g:=TimeReverse(L[2],V[2]);

 h:=Alternate(h);
 g:=Alternate(g); 

 ht:=[];
 gt:=[];

 ht[1] := g[1];
 ht[2] := g[2]+1;
 gt[1] := -h[1];
 gt[2] := h[2]+1;

 return [[ht[1],gt[1]],[ht[2],gt[2]]];

end;


Anal2SynthFilts := function(L,V) 
 
 local ht,gt,g,h;                
 h:=TimeReverse(L[1],V[1]);
 g:=TimeReverse(L[2],V[2]);
 
 h:=Alternate(h);
 g:=Alternate(g); 

 ht:=[];
 gt:=[];

 ht[1] := g[1];
 ht[2] := g[2]+1;
 gt[1] := -h[1];
 gt[2] := h[2]+1;

 return [[ht[1],gt[1]],[ht[2],gt[2]]];

end;

# wraps a polynomial given as Poly or as List+valuation periodically
# for circulants
CircularWrap := function(arg)
local L,v,l,r,n;
if (Length(arg) = 2) then 
if IsPolynomial(arg[1]) then
l:=ListPoly(arg[1]); 
else l:=arg[1];fi;
L:=l[1]; 
v:=l[2];
n:=arg[2]; 
else 
L:=arg[1]; 
v:=arg[2];
n:=arg[3];
fi;

r:= Length(L)+v-1;
if v>0 then L:=Concat(Replicate(v,0),L);fi;

if r <n then 
L:=Concat(L, Replicate(n-r-1,0));
L:=L*MatSPL(Ext(n,per(PosInt(-v)),per(0)));
else
L:=L*MatSPL(Ext(n,per(PosInt(-v)),per(r-n+1)));
fi;

# Circulant needs first column as param instead of first row (need to be changed)

l:= Reversed(L{[2..Length(L)]});
L:=Concat([L[1]],l);

return L;
end;



# 
# Helper functions for non-terminal arguments 
#
toSize := o -> Cond(
    IsPosInt(o), o,
    IsList(o), Length(o),
    IsPolynomial(o), Length(o.coefficients),
    IsFunction(o), o.domain(),
    Error("List, polynomial, or generating function expected"));
    
toFunc := o -> Cond(
    IsPosInt(o), FUnk(o),
    IsList(o), Checked(o<>[], FList(UnifyTypesV(o), o)),
    IsPolynomial(o), FList(UnifyTypesV(o.coefficients), o.coefficients),
    IsFunction(o),   o,
    Error("List, polynomial, or generating function expected"));

toValuation := o -> Cond(
    IsPosInt(o), 0,
    IsList(o), 0,
    IsPolynomial(o), o.valuation,
    IsFunction(o), 0,
    Error("List, polynomial, or generating function expected"));

QuantizeQuarters := rat -> Cond(0  <= rat and rat <= 3/8, 1/4,
                                3/8 < rat and rat <= 5/8, 1/2,
                                5/8 < rat and rat <= 7/8, 3/4,
				1);


toFiltFunc := o -> Cond(
    IsPosInt(o), 
        [ fUnk(TReal, o), -o+1],
    IsList(o), Checked(o<>[], 
	[ toFunc(o), -Length(o)+1 ]),
    IsPolynomial(o), let(zp := FillZeros(o),
        [ toFunc(zp[1]), zp[2] ]), 
    IsFunction(o) or IsFuncExp(o), 
        [ o, -o.domain()+1 ],
    Error("List, polynomial, or generating function expected"));

toFiltFuncV := (o,v) -> Cond(
    IsPosInt(o), let(w := v+o-1,
	When(w < 0 or v > 0, 
	    Error("Valuation does not match domain of <o>, can't zero-pad a gener. function"),
        [ fUnk(TReal, o), v])),
    IsList(o), Checked(o<>[], let(zp := FillZeros([o,v]),
	[ toFunc(zp[1]), zp[2] ])),
    IsFunction(o) or IsFuncExp(o), let(dom := o.domain(), w := v+dom-1,
	When(w < 0 or v > 0, 
	    Error("Valuation does not match domain of <o>, can't zero-pad a gener. function"),
	    [ o, -o.domain()+1 ])),
    Error("Only list or generating function can be specified with separate valuation"));
