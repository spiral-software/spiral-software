
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


ZeroCoef := function(p)
local i, L;
L:=p.coefficients;
for i in [1..Length(p.coefficients)] do
  if AbsFloat(ReComplex(ComplexAny(p.coefficients[i]))) < FloatString("1e-4") then
    L[i] := p.baseRing.zero;
  fi;
od;
return Polynomial(p.baseRing, L, p.valuation);
end;


# Recursive implementation of the long division for Laurent polynomials h(z) and g(z)
# There are deg(h)-deg(g)+2 different schemes for division 
LaurentRemainder := function ( arg )

local a, b, q, q1, r1, q2, r2, Ls, Lt, z, 
      deghigha, deglowa, deghighb, deglowb, dega, degb, coefa, coefb,  divs;

if IsList(arg[1]) then arg:=arg[1];fi;
a:=arg[1];
b:=arg[2];

#Print("check");
if not (IsPolynomial(a) and IsPolynomial(a)) then
  Error("<a> and <b> must be polynomials");
elif not a.baseRing.name = b.baseRing.name then
  Error("Both polynomials have to be over the same field");
elif not IsBound(PolynomialRing(a.baseRing).isEuclideanRing) then
  Error("Polymonials must be constructed over a field, not over a ring");
fi;

z:=Indeterminate(a.baseRing);
z.name:="z";
if Length(arg)=3 then q:=arg[3]; else q:=0*z^0; fi;

deglowa := a.valuation;
deglowb := b.valuation;
dega := Length(a.coefficients)-1;
degb := Length(b.coefficients)-1;
deghigha := deglowa + dega;
deghighb := deglowb + degb;


coefa := a.coefficients;
coefb := b.coefficients; 

if dega<degb then return [[q,a]];fi;

  q1 := Polynomial(a.baseRing,[coefa[dega+1]/coefb[degb+1]],deghigha-deghighb);
  r1 := a - q1 * b;
  q1 := q1 + q;
#  Print(q1);
#  Print("\n");
#  Print(r1);
#Print("\n");

#  q2 := coefa[1]/coefb[1] * z^(deglowa-deglowb);
  q2 := Polynomial(a.baseRing,[coefa[1]/coefb[1]],deglowa-deglowb);
  r2 := a - q2 * b;
  q2 := q2 + q;
  Ls := [];
  for divs in [[r1,b,q1],[r2,b,q2]] do
     Lt := LaurentRemainder(divs);
     Add(Ls,Lt);
  od;
  Ls := Flat(Ls);
  Ls := List([1..Length(Ls)/2], i->[Ls[2*i-1],Ls[2*i]]); 
  Ls := List(Collected(Ls), col->col[1]);
  if not IsList(Ls[1]) then return [Ls];
  else return Ls;
  fi;  
 

end;

#
# Note:
# ! Because we now use double floats, the LiftingScheme generates more LS than
# possible because of finite precision. There will be a bunch of the same lifting
# schemes generated multiple times with coefficients different after 10th
# decimal or so. This requies fixing !
#

#F LiftingScheme( <poly1>, <poly2> )
#F   returns a list of Laurent polynomials that represent 
#F   primal and dual lifting steps factorization of the polyphase 
#F   matrix of polynomials <poly1> and <poly2>:
#F            P(z)= [[ <poly1_even>, <poly1_odd>],
#F                   [ <poly2_even>, <poly2_odd>]]
#F   In Discrete Wavelet Transform notation, <poly1> represents
#F   a low-pass filter impulse response, whereas <poly2> represents
#F   a high-pass one. 
#F   
#F   The output is in the following format:
#F   [ b, l1(z), l2(z), .... , ln(z), c1, c2]
#F   
#F   b - a binary symbol that determines the type of the 
#F       initial lifting step (0 - primal, 1 - dual)
#F   li(z) - polynomial representing i-th lifting step
#F   c1, c2 - scaling constants
#F
#F   Example:
#F   
#F   P(z)=  [ c1, 0  ] * [ 1, l2(z)] * [ 1    , 0 ] 
#F          [  0, c2 ]   [ 0, 1    ]   [ l1(z), 1 ]
#F          ----------   -----------   ------------   
#F           scaling      primal l.s.   dual l.s. (b=1)
#F
#F   Note: Factorization is implemented in Laurent Polynomial
#F   ring over the base field inherited from <poly1>, <poly2>.
#F

LiftingScheme := function ( M )

local he, ho, ge, go, hen, hon, gen, gon, b, QR, z, Le, Lo, L, Lc, Ls, ve, vo, P, factor, scheme;


#Print("Recursion \n");
# Check if h(z) and g(z) in fact represent a perfect reconstruction 
# filter bank -> determinant of the polyphase matrix should be
# a monomial 
he := M[1][1];ho := M[1][2]; ge := M[2][1]; go := M[2][2];
  if not Length(ZeroCoef(Determinant([[he, ho],[ge, go]])).coefficients) = 1 then
     Error("h(z) and g(z) must satisfy the perfect reconstruction condition P(z)*P'(z)^(-1) = I where P and P' are analysis and synthesis polyphase matrices");
  fi;

  z:=Indeterminate(he.baseRing);
  z.name:="z";
# Factoring Algorithm
  b:=0;

  if LaurentDegree(he) >= LaurentDegree(ho) then b:=1; fi;

  if (ho=0*z^0 or ge=0*z^0) then
    L:=[];
    if ho=0*z^0 then Add(L,ZeroCoef(ge)*
         Polynomial(he.baseRing,[1/ ZeroCoef(go).coefficients[1]], -go.valuation));
    else Add(L,ho/he); fi;
    Add(L, he);
    Add(L, go);
    Add(L, b);
    return [Reversed(L)];
  fi;
  if (he=0*z^0 or go=0*z^0) then
    L:=[];
    if he=0*z^0 then Add(L,ZeroCoef(go)*
         Polynomial(he.baseRing,[1/ ZeroCoef(ge).coefficients[1]], -ge.valuation));
    else Add(L,he/ho); fi;
    Add(L, ho);
    Add(L, ge);
    Add(L, b-2);
    return [Reversed(L)];
  fi;
  L:=[];    
    if b=0 then
      QR:=LaurentRemainder(ho,he);
      for factor in QR do
        hon := factor[2];
        hon := ZeroCoef(hon);
        gon := go - ge * factor[1];
        gon := ZeroCoef(gon);

        Lc:=LiftingScheme([[he,hon],[ge,gon]]);
        for scheme in Lc do Add(scheme,factor[1]); od;
        Append(L,Lc);
      od;


    else    
      QR:=LaurentRemainder(he,ho);
      for factor in QR do
        hen := factor[2];
        hen := ZeroCoef(hen);
        gen := ge - go * factor[1];
        gen := ZeroCoef(gen);
        Lc:=LiftingScheme([[hen,ho],[gen,go]]);
        for scheme in Lc do Add(scheme,factor[1]); od;
        Append(L,Lc);
      od;

    fi;

return(L);

end;


Lifting := function(M)
local Lt, Ls, L, factor, p, normM, normDiff;

L:=LiftingScheme(M,[]);
Lt:=[];
Ls:=[];

for factor in L do 
  Add(Ls, factor);
 if IsInt(factor) then  Add(Lt, Ls); Ls:=[];fi;
od;
return Lt;
end;


VerifyLiftingScheme:=function(P,L2)

local P, M, b, i, l,z,p, normM, normDiff;

  z:=Indeterminate(P[1].baseRing);
  z.name:="z";

#P:=[[P[1],P[2]],[P[3],P[4]]];
M:= [[1, 0],[0,1]];

L2:=Reversed(L2);
l:=Length(L2);
b:=L2[l];
for i in [1 .. l-3] do
 if b=-2 then 
   M:= M * [[0*z^0,1*z^0 ], [1*z^0, L2[l-i-2]]]; b:=1;
 elif b=-1 then 
   M:= M * [[L2[l-i-2],1*z^0 ], [1*z^0, 0*z^0]]; b:=0;
 elif b=0 then
   M:= M * [[1*z^0, L2[l-i-2]], [0*z^0, 1*z^0]]; b:=1;
 else 
   M:= M * [[1*z^0, 0*z^0], [L2[l-i-2], 1*z^0]]; b:=0;
 fi;
od;
M:=[[L2[l-2], 0*z^0],[0*z^0, L2[l-1]]]*M;
#Print([[P[1],P[2]],[P[3],P[4]]]);
if Same(P[1][1].baseRing, Doubles) then 
  normM:=Sum(M, l->Sum(l, p->(Sum(p.coefficients,i->i^2))^1/2))^1/2;
  normDiff:=Sum(M-P, l->Sum(l, p->(Sum(p.coefficients,i->i^2))^1/2))^1/2;
  Print("error: ",normDiff/normM,"\n");
  if (normDiff/normM > FloatString("1e-4")) then return false;
  else return true;
  fi;
else return[M,P,M=P];
fi;
end;

ArithmeticCostLS := function(L)
local step, mult, add, i;

mult:=0;
add:=0;
L:=Reversed(L);
for step in [1..Length(L)-3] do
  for i in L[step].coefficients do
     if not i = 0 then mult:=mult+1; add:=add+1;fi;
  od;
od;
mult:=mult+2;
return [add,mult,Length(L)-3];
end;


#F MatPoly := function(h,g)
#F   returns a matrix formed of two filters h(z) and g(z)
#F   also returns largest and the smallest degree of the 
#F   coefficients in the matrix. 
#F   Format: [<mat>, q, p]

MatPoly := function(h,g)

local hc, gc, hsmall, hlarge,
      gsmall, glarge, q, p, M;

    # Check the degrees of the filters   
    hc := h.coefficients;
    gc := g.coefficients; 
    hsmall := h.valuation;
    hlarge := Length(hc) + hsmall - 1;
    gsmall := g.valuation;
    glarge := Length(gc) + gsmall - 1;
    q := Minimum([hsmall,gsmall]);
    p := Maximum([hlarge,glarge]);
    
# Construct the 2 x l matrix of filter coefficients

    M := 
      [
       Concat(List([1..AbsInt(q-hsmall)],i->h.baseRing.zero),
               hc, List([1..p-hlarge],i->h.baseRing.zero)
             ),
       Concat(List([1..AbsInt(q-gsmall)],i->h.baseRing.zero),
               gc, List([1..p-glarge],i->h.baseRing.zero)
             )
      ];

    return ([M, hsmall, hlarge, gsmall, glarge]);
end;


Polyphase := function(p1,p2)

local  he, ho, ge, go, z, h, g;

if not (IsPolynomial(p1) and IsPolynomial(p1)) then
  Error("<p1> and <p2> must be polynomials");
elif not p1.baseRing.name = p2.baseRing.name then
  Error("Both polynomials have to be over the same field");
elif not IsBound(PolynomialRing(p1.baseRing).isEuclideanRing) then
  Error("Polymonials must be constructed over a field, not over a ring");
fi;

  z:=Indeterminate(p1.baseRing);
  z.name:="z";

        h := DownsampleTwo(ListPoly(p1));
        g := DownsampleTwo(ListPoly(p2));
        he := Polynomial(p1.baseRing, h[1][1], h[1][2]); 
        ho := Polynomial(p1.baseRing, h[2][1], h[2][2]); 
        ge := Polynomial(p1.baseRing, g[1][1], g[1][2]); 
        go := Polynomial(p1.baseRing, g[2][1], g[2][2]); 

# Find he(z) and ho(z)
#  Lo:=Sublist(p1.coefficients, Filtered([1..Length(p1.coefficients)],IsEvenInt));
#  Le:=Sublist(p1.coefficients, Filtered([1..Length(p1.coefficients)],IsOddInt));
#  if (p1.valuation mod 2 = 1) then L:=Le; Le:=Lo; Lo:=L; 
#    ve := Int((p1.valuation + 1)/2); vo := ve;
#  else ve := Int(p1.valuation/2); vo := ve + 1;
#  fi;
#  he:=Polynomial(p1.baseRing, Le, ve);
#  ho:=Polynomial(p1.baseRing, Lo, vo);  
#    
# Find ge(z) and go(z)

#  Lo:=Sublist(p2.coefficients, Filtered([1..Length(p2.coefficients)],IsEvenInt));
#  Le:=Sublist(p2.coefficients, Filtered([1..Length(p2.coefficients)],IsOddInt));
#  if (p2.valuation mod 2 = 1) then L:=Le; Le:=Lo; Lo:=L; 
#    ve := Int((p2.valuation + 1)/2); vo := ve;
#  else ve := Int(p2.valuation/2); vo := ve+1;
#  fi;
#  ge:=Polynomial(p2.baseRing, Le, ve);
#  go:=Polynomial(p2.baseRing, Lo, vo);  
 
return [[he,ho],[ge,go]];
end;


# PolyPolyphaseMat:=function(M)
# Returns a list of 2 polynomials from a polyphase matrix 

PolyPolyphaseMat:=function(M)

local he,ho,ge,go,h,g,z;

he:=M[1][1];
ho:=M[1][2];
ge:=M[2][1];
go:=M[2][2];
z:=Indeterminate(he.baseRing);
z.name:="z";

h:=Upsample(he)+z^(1)*Upsample(ho);
g:=Upsample(ge)+z^(1)*Upsample(go);

return([h,g]);
end;


