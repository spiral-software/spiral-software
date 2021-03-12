# -*- Mode: shell-script -*- 

#############################################################################
##
#A  morpheus.g                  GAP library                  Alexander Hulpke
##
##
#Y  Copyright (C) 2018-2021, Carnegie Mellon University
#Y  All rights reserved.  See LICENSE for details.
#Y  
#Y  This work is based on GAP version 3, with some files from version 4.  GAP is
#Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
##
##  This file contains functions for isomorphisms between groups and
##  automorphism groups.
##
##  The filename is NOT ment to give any allusions to computing speed...
##

#############################################################################
##
#V  InfoMorph1/2(<arg>) . . . . . . . . . . . . . . . . information functions
##
if not IsBound(InfoMorph1) then InfoMorph1:=Ignore;fi;
if not IsBound(InfoMorph2) then InfoMorph2:=Ignore;fi;

#############################################################################
##
#V  MORPHEUSELMS . . . .  limit up to which size to store element lists
##
MORPHEUSELMS := 50000;

#############################################################################
##
#F  MorFroWords(<gens>) . . . . . . create some pseudo-random words in <gens>
##                                                featuring the MeatAxe's FRO
MorFroWords := function(gens)
local list,a,b,ab,i;
  list:=[];
  ab:=gens[1];
  for i in [2..Length(gens)] do
    a:=ab;
    b:=gens[i];
    ab:=a*b;
    list:=Concatenation(list,
	 [ab,ab^2*b,ab^3*b,ab^4*b,ab^2*b*ab^3*b,ab^5*b,ab^2*b*ab^3*b*ab*b,
	 ab*(ab*b)^2*ab^3*b]);
  od;
  return list;
end;

#############################################################################
##
#F  PermAutImg(<elms>,<hom>) . . . . Permutation operation of <hom> on <elms>
##
PermAutImg := function(elms,hom)
  if not IsBound(hom.elmsPerm) or not IsIdentical(elms,hom.permElms) then
    hom.permElms:=elms;
    hom.elmsPerm:=PermList(List([1..Length(elms)],i->Position(elms,
       Image(hom,elms[i]))));
  fi;
  return hom.elmsPerm;
end;

#############################################################################
##
#V  PermAutomorphismGroupOps . .  operations record for automorphisms knowing
##                                the element permutation.
##
PermAutomorphismGroupOps :=
  OperationsRecord("PermAutomorphismGroupOps",GroupOps);

PermAutomorphismGroupOps.Size := function(G)
  return Size(G.permGroup);
end;

PermAutomorphismGroupOps.\in := function(el,G)
  return IsGeneralMapping(el) and
         PermAutImg(Parent(G).elms,el) in G.permGroup;
end;

PermAutomorphismGroupOps.Subgroup := function(G,gens)
local U;
  if G.identity in gens  then
    gens:=Filtered(gens,function(gen)
	      return gen<>G.identity;
      end);
  else
    gens:=ShallowCopy(gens);
  fi;
  if IsEqualSet(G.generators,gens) then
    U:=G;
  else
    U:=rec(
	   );
    U.isDomain:=true;
    U.isGroup:=true;
    U.parent:=G;
    U.identity:=G.identity;
    U.generators:=gens;
    U.operations:=PermAutomorphismGroupOps;
    U.permGroup:=Subgroup(G.permGroup,List(gens,i->PermAutImg(G.elms,i)));
  fi;
  return U;
end;

PermAutomorphismGroupOps.Closure := function(G,obj)
local i;
  if IsGroup(obj)
    then obj:=obj.generators;
  elif not IsList(obj) then
    obj:=[obj];
  fi;
  for i in obj do
    if not i in G then
      G:=Subgroup(Parent(G),Concatenation(G.generators,[i]));
    fi;
  od;
  return G;
end;

PermAutomorphismGroupOps.RightTransversal := function(G,U)
local a,e,t,grp,ge;
  t:=RightTransversal(G.permGroup,U.permGroup); 
  e:=Parent(G).elms;
  ge:=Parent(G).elmsgens;
  grp:=G.identity.source;
  a:=List(ge,i->Position(e,i));
  t:=List(t,i->GroupHomomorphismByImages(grp,grp,ge,
                 List(a,j->e[j^i])));
  for e in t do
    e.isMapping:=true;
  od;
  return t;
end;


#############################################################################
##
#F  MorRatClasses(<G>) . . . . . . . . . . . local rationalization of classes
##
MorRatClasses := function(GR)
local r,c,u,j,i,flag;
  InfoMorph2("#I  RationalizeClasses\n");
  if IsBound(GR.operations.RationalClasses) then
    r:=[];
    for c in RationalClasses(GR) do
      u:=Subgroup(GR,[c.representative]);
      j:=c.operations.Decomposed(c);
#j:=Set(j);
      Add(r,rec(representative:=u,
		  class:=j[1],
		  classes:=j,
		  size:=Size(c)));

    od;

  else
    c:=Filtered(ConjugacyClasses(GR),i->i.representative<>GR.identity);
    if false and IsBound(GR.rationalClasses) then
      r:=GR.rationalClasses;
    else
      r:=[];
      for i in c do 
	flag:=true;
	j:=0;
	while flag and j<Length(r) do
	  j:=j+1;
	  if ForAny(r[j].classes,k->i.representative in k) then
	    flag:=false;
	  fi;
	od;
	if flag then
	  u:=Subgroup(GR,[i.representative]);
	  Add(r,rec(representative:=u,
		    class:=i,
		    classes:=[i]
		    ));
	else
	  Add(r[j].classes,i);
	fi;
      od;
    fi;
  fi;

  for i in r do
    i.size:=Sum(i.classes,Size);
  od;
  return r;
end;

#############################################################################
##
#F  MorMaxFusClasses(<l>) . .  maximal possible morphism fusion of classlists
##
MorMaxFusClasses := function(r)
local i,j,flag,cl;
  # cl is the maximal fusion among the rational classes.
  cl:=[]; 
  for i in r do
    j:=0;
    flag:=true;
    while flag and j<Length(cl) do
      j:=j+1;
      flag:=not(Size(i.class)=Size(cl[j][1].class) and
		  i.size=cl[j][1].size and
		  Size(i.representative)=Size(cl[j][1].representative));
    od;
    if flag then
      Add(cl,[i]);
    else
      Add(cl[j],i);
    fi;
  od;

  # sort classes by size
  Sort(cl,function(a,b) return
    Sum(a,i->i.size)
      <Sum(b,i->i.size);end);
  return cl;
end;

#############################################################################
##
#F  MorClassLoop(<G>,<classes>,<params>) . . . loop over classes list to find
##             generating sets or Iso/Automorphisms up to inner automorphisms
##
##  params is an record, containing
##    type  =1 for generating sets
##          =2 for morphisms
##    what  =1 for automs
##           2 for one isom
##           3 for mappings into
##           4 for G-Quotient
##    gens     generators of preimage
##    from     preimage
##    size     size of image
##    elms     list of elements
##    isom     group permuting elms, approx. of autom. group
##    free     free group on <gens> generators
##    rels     some relations between gens: [relator,orderImage]
##
MorClassLoop := function(GR,clali,params)
local i,isom,l,len,cla,ind,m,mp,cen,hom,type,elms,what,gens,G,free,rels,
      size;
  type:=params.type;
  if IsBound(params.what) then
    what:=params.what;
    if type=2 then
      if not IsBound(params.free) then
	G:=params.from;
	gens:=params.gens;
	# find some relations, that hopefully will be defining
	free:=FreeGroup(Length(gens));
	free:=free.generators;
	rels:=List(MorFroWords(free),i->[i,Order(G,MappedWord(i,free,gens))]);
      else
        free:=params.free;
	rels:=params.rels;
      fi;
    fi;
  fi;
  if IsBound(params.elms) then
    elms:=params.elms;
  fi;
  if IsBound(params.size) then
    size:=params.size;
  else
    size:=Size(GR);
  fi;
  if IsBound(params.isom) then
    isom:=params.isom;
  elif type=1 then
    isom:=false;
  else
    isom:=[];
  fi;
  len:=Length(clali);
  # 'free for loop' over all classes in clali
  l:=0*[1..len]+1;
  ind:=len;
  while ind>0 do
    ind:=len;
    # test class combination indicated by l:
    cla:=List([1..len],i->clali[i][l[i]]); 
    # test, whether a gen.sys. can be taken from the classes in <cla>
    # candidates

    # this makes up another free for loop ...
    m:=[];
    m[len]:=[cla[len].representative];
    # positions
    mp:=[];
    mp[len]:=1;
    mp[len+1]:=-1;
    # centralizers
    cen:=[];
    cen[len]:=cla[len].centralizer;
    cen[len+1]:=GR; # just for the recursion
    i:=len-1;

    # set up the lists
    while i>0 do
      m[i]:=List(DoubleCosets(cla[i].group,cla[i].centralizer,
                              Intersection(cla[i].group,cen[i+1])),
		    j->cla[i].representative^j.representative);
      mp[i]:=1;
      if i>1 then
	cen[i]:=Centralizer(cen[i+1],m[i][1]);
      fi;
      i:=i-1;
    od;
    i:=1; 

    while i<len do
      hom:=List([1..len],i->m[i][mp[i]]);
      # the size can be nasty. Thus try the short presentation first.
      if type<>1 or Size(Subgroup(GR,hom))=size then
	# otherwise not gen. set.
	if type=1 then
	    # found gen set. UFF!
	    return hom;
	else
#Print(hom,List(rels,i->MappedWord(i[1],free,hom)^i[2]),"\n");
	  if (what>1 or hom<>gens) # if we check for Isom(g,g) &c, we might
				   # have the same gens
	    and
	     ((what<>4 and 
	       ForAll(rels,i->Order(GR,MappedWord(i[1],free,hom))=i[2])) or
	      (what=4 and
	       ForAll(rels,i->MappedWord(i[1],free,hom)^i[2]=GR.identity)))
	    and
	      Size(Subgroup(GR,hom))=size then
	    InfoMorph2("#I  testing");
	    hom:=GroupHomomorphismByImages(G,GR,gens,hom);
	    if IsHomomorphism(hom) and (what=4 or Size(Kernel(hom))=1) then
	      InfoMorph2(",found");
	      if Size(Kernel(hom))=1 then
		hom.isInjective:=true;
	      fi;
	      if Size(GR)=size then
		hom.isSurjective:=true;
	      fi;
	      if what=2 then
		InfoMorph2("\n");
		isom:=[hom];
		return isom;
	      elif elms=false then
		Add(isom,hom);
	      else
		hom:=PermAutImg(elms,hom);
		if not hom in isom then
		  InfoMorph2(",new");
		  # avoid the large S_n, so make a new group each time.
		  isom:=Group(Concatenation(isom.generators,[hom]),());
		fi;
	      fi;
	    fi;
	  InfoMorph2("\n");
	  fi;
	fi;
      fi;

#      if flag then
      mp[i]:=mp[i]+1;
      while i<=len and mp[i]>Length(m[i]) do
	mp[i]:=1;
	i:=i+1;
	mp[i]:=mp[i]+1;
      od;
      if i<=len then
	while i>1 do
	  cen[i]:=Centralizer(cen[i+1],m[i][mp[i]]);
	  i:=i-1;
	  m[i]:=List(DoubleCosets(GR,cla[i].centralizer,cen[i+1]),
		      j->cla[i].representative^j.representative);
	  mp[i]:=1;
	od;
      fi;
#      fi;

    od;

#    if flag then

    # 'free for increment'
    l[ind]:=l[ind]+1;
    while ind>0 and l[ind]>Length(clali[ind]) do
      l[ind]:=1;
      ind:=ind-1;
      if ind>0 then
	l[ind]:=l[ind]+1;
      fi;
    od;

#    fi;
  od;
  return isom;
end;

#############################################################################
##
#F  MorFindGeneratingSystem(<G>,<cl>) . .  find generating system with an few 
##                      as possible generators from the first classes in <cl>
##
MorFindGeneratingSystem := function(G,cl)
local lcl,len,comb,combc,com,a;
  InfoMorph1("#I  FindGenerators\n");
  #create just a list of ordinary classes.
  lcl:=List(cl,i->Concatenation(List(i,j->j.classes)));
  len:=Maximum(1,Length(IrreducibleGeneratingSet(
		    AgGroup((G/DerivedSubgroup(G)))))-1);
  while true do
    len:=len+1;
    # now search for <len>-generating systems
    comb:=UnorderedTuples([1..Length(lcl)],len); 
    combc:=List(comb,i->List(i,j->lcl[j]));

    # test all <comb>inations
    com:=0;
    while com<Length(comb) do
      com:=com+1;
      a:=MorClassLoop(G,combc[com],rec(type:=1));
      if a<>false then
        return a;
      fi;
    od;
  od;
end;

#############################################################################
##
#F  Morphium(<G>,<H>,<what>) . . . . . . . .Find isomorphisms between G and H
##       modulo inner automorphisms. If what=1 all possibilities are computed.
##       This function thus does the main combinatoric work for creating 
##       Iso- and Automorphisms.
##       It needs, that both groups are not cyclic.
##
Morphium := function(G,H,what)
local 
      len,comb,combc,com,combi,l,m,mp,cen,Gr,Gcl,Ggc,Hr,Hcl,
      ind,gens,i,j,c,cla,u,lcl,hom,isom,free,elms,price;

  # try the given generating system
  if IsAgGroup(G) then
    gens:=IrreducibleGeneratingSet(G);
  else
    gens:=G.generators;
  fi;
  len:=Length(gens);
  Gr:=MorRatClasses(G);
  Gcl:=MorMaxFusClasses(Gr);

  Ggc:=List(gens,i->First(Gcl,j->ForAny(j,j->ForAny(j.classes,k->i in k))));
  combi:=List(Ggc,i->Concatenation(List(i,i->i.classes)));
  price:=Product(combi,i->Sum(i,Size));
  InfoMorph1("#I  generating system of price:",price,"\n");

  if not IsAgGroup(G) and price>20000  then

    if IsSolvable(G) and what=2 then
      gens:=AgGroup(G);
      gens:=List(IrreducibleGeneratingSet(gens),i->Image(gens.bijection,i));
    else
      gens:=MorFindGeneratingSystem(G,Gcl);
    fi;

    Ggc:=List(gens,i->First(Gcl,j->ForAny(j,j->ForAny(j.classes,k->i in k))));
    combi:=List(Ggc,i->Concatenation(List(i,i->i.classes)));
    price:=Product(combi,i->Sum(i,Size));
    InfoMorph1("#I  generating system of price:",price,"\n");
  fi;

  if what=2 then
    Hr:=MorRatClasses(H);
    Hcl:=MorMaxFusClasses(Hr);
  fi;

  # now test, whether it is worth, to compute a finer congruence
  # then ALSO COMPUTE NEW GEN SYST!
  # [...]

  if what=2 then
    combi:=[];
    for i in Ggc do
      c:=Filtered(Hcl,
	   j->Set(List(j,k->k.size))=Set(List(i,k->k.size))
		and Length(j[1].classes)=Length(i[1].classes) 
		and Size(j[1].class)=Size(i[1].class)
		and Size(j[1].representative)=Size(i[1].representative)
      # This test assumes maximal fusion among the rat.classes. If better
      # congruences are used, they MUST be checked here also!
	);
      if Length(c)<>1 then
	# Both groups cannot be isomorphic, since they lead to different 
	# congruences!
	InfoMorph2("#I  different congruences\n");
	return false;
      else
	Add(combi,c[1]);
      fi;
    od;
    combi:=List(combi,i->Concatenation(List(i,i->i.classes)));
  fi;

  # combi contains the classes, from which the
  # generators are taken.

  isom:=[];
  elms:=false;
  if what=1 then
    if Size(G)<=MORPHEUSELMS then
      elms:=Union(List(Set(List(Flat(combi),Representative)),i->Orbit(G,i)));
      isom:=Group(IdentityMapping(G));
    fi;
  fi;

  # now take all possible images, and test for morphism:

  isom:=MorClassLoop(H,combi,rec(type:=2,what:=what,gens:=gens,
                           elms:=elms,from:=G,isom:=isom));

  if elms<>false then
    isom.elms:=elms;
    isom.elmsgens:=gens;
  fi;
  return isom;

end;


# Martin's standard generator routines
IndependentGeneratorsAbelianPPermGroup := function ( P, p )
    local   inds,       # independent generators, result
            pows,       # their powers
            base,       # the base of the vectorspace
            size,       # the size of the group generated by <inds>
            orbs,       # orbits
            trns,       # transversal
            gens,       # remaining generators
            gens2,      # remaining generators for next round
            exp,        # exponent of <P>
            g,          # one generator from <gens>
            h,          # its power
            b,          # basepoint
            c,          # other point in orbit
            i, j, k;    # loop variables

    # initialize the list of independent generators
    inds := [];
    pows := [];
    base := [];
    size := 1;
    orbs := [];
    trns := [];

    # gens are the generators for the remaining group
    gens := P.generators;

    # loop over the exponents
    exp := Maximum( List( P.generators, g -> LogInt( Order( P, g ), p ) ) );
    for i  in [exp,exp-1..1]  do

        # loop over the remaining generators
        gens2 := [];
        for j  in [1..Length(gens)]  do
            g := gens[j];
            h := g ^ (p^(i-1));

            # reduce <g> and <h>
            while h <> h^0
              and IsBound(trns[SmallestMovedPointPerm(h)^h])
            do
                g := g / pows[ trns[SmallestMovedPointPerm(h)^h] ];
                h := h / base[ trns[SmallestMovedPointPerm(h)^h] ];
            od;

            # if this is linear indepenent, add it to the generators
            if h <> h^0  then
                Add( inds, g );
                Add( pows, g );
                Add( base, h );
                size := size * p^i;
                b := SmallestMovedPointPerm(h);
                if not IsBound( orbs[b] )  then
                    orbs[b] := [ b ];
                    trns[b] := [ () ];
                fi;
                for c  in ShallowCopy(orbs[b])  do
                    for k  in [1..p-1]  do
                        Add( orbs[b], c ^ (h^k) );
                        trns[c^(h^k)] := Length(base);
                    od;
                od;

            # otherwise reduce and add to gens2
            else
                Add( gens2, g );

            fi;

        od;

        # prepare for the next round
        gens := gens2;
        pows := OnTuples( pows, p );

    od;

    # return the indepenent generators
    return inds;
end;

IndependentGeneratorsAbelianPermGroup := function ( G )
    local   inds,       # independent generators, result
            p,          # prime factor of group size
            gens,       # generators of <p>-Sylowsubgroup
            g,          # one generator
            o;          # its order

    # loop over all primes
    inds := [];
    for p  in Union(List( G.generators, g -> Set(Factors(Order(G,g)))) )  do

        # compute the generators for the <p>-Sylowsubgroup
        gens := [];
        for g  in G.generators  do
            o := Order(G,g);
            while o mod p = 0  do o := o / p; od;
            if g^o <> g^0  then Add( gens, g^o );  fi;
        od;

        # append the independent generators for the <p>-Sylowsubgroup
        Append( inds,
                IndependentGeneratorsAbelianPPermGroup(Group(gens,()),p) );
        
    od;

    # return the independent generators
    return inds;
end;

AutomorphismGroupAbelianGroup := function(G)
local i,j,k,l,m,o,nl,nj,max,r,e,au,p,gens,offs;

  # get standard generating system
  if not IsPermGroup(G) then
    p:=PermGroup(G);
    gens:=IndependentGeneratorsAbelianPermGroup(p);
    gens:=List(gens,i->Image(p.bijection,i));
  else
    gens:=IndependentGeneratorsAbelianPermGroup(G);
  fi;

  au:=[];
  # run by primes
  p:=Set(Factors(Size(G)));
  for i in p do
    l:=Filtered(gens,j->IsInt(Order(G,j)/i));
    nl:=Filtered(gens,i->not i in l);

    #sort by exponents
    o:=List(l,j->LogInt(Order(G,j),i));
    e:=[];
    for j in Set(o) do
      Add(e,[j,l{Filtered([1..Length(o)],k->o[k]=j)}]);
    od;

    # construct automorphisms by components
    for j in e do
      nj:=Concatenation(List(Filtered(e,i->i[1]<>j[1]),i->i[2]));
      r:=Length(j[2]);

      # the permutations and addition
      if r>1 then
	Add(au,GroupHomomorphismByImages(G,G,Concatenation(nl,nj,j[2]),
	    #(1,2)
	    Concatenation(nl,nj,j[2]{[2]},j[2]{[1]},j[2]{[3..Length(j[2])]})));
	Add(au,GroupHomomorphismByImages(G,G,Concatenation(nl,nj,j[2]),
	    #(1,..,n)
	    Concatenation(nl,nj,j[2]{[2..Length(j[2])]},j[2]{[1]})));
	#for k in [0..j[1]-1] do
        k:=0;
	  Add(au,GroupHomomorphismByImages(G,G,Concatenation(nl,nj,j[2]),
	      #1->1+i^k*2
	      Concatenation(nl,nj,[j[2][1]*j[2][2]^(i^k)],
	                          j[2]{[2..Length(j[2])]})));
        #od;
      fi;
  
      # multiplications
      for k in List(PrimeResidueClassGroup(i^j[1]).generators,
                    Representative) do
	Add(au,GroupHomomorphismByImages(G,G,Concatenation(nl,nj,j[2]),
	    #1->1^k
	    Concatenation(nl,nj,[j[2][1]^k],j[2]{[2..Length(j[2])]})));
      od;

    od;
    
    # the mixing ones
    for j in [1..Length(e)] do
      for k in [1..Length(e)] do
	if k<>j then
	  nj:=Concatenation(List(e{Difference([1..Length(e)],[j,k])},i->i[2]));
	  offs:=Maximum(0,e[k][1]-e[j][1]);
	  if Length(e[j][2])=1 and Length(e[k][2])=1 then
	    max:=Minimum(e[j][1],e[k][1])-1;
	  else
	    max:=0;
	  fi;
	  for m in [0..max] do
	    Add(au,GroupHomomorphismByImages(G,G,
	       Concatenation(nl,nj,e[j][2],e[k][2]),
	       Concatenation(nl,nj,[e[j][2][1]*e[k][2][1]^(i^(offs+m))],
				    e[j][2]{[2..Length(e[j][2])]},e[k][2])));
	  od;
	fi;
      od;
    od;
  od;

  for i in au do
    i.isMapping:=true;
    i.isInjective:=true;
    i.isSurjective:=true;
  od;

  au:=Group(au,IdentityMapping(G));

  if Size(G)<50000 then
    # note permutation action
    o:=Filtered(Elements(G),i->Order(G,i)>1);
    p:=List(au.generators,i->PermAutImg(o,i));
    p:=Group(p,());
    au.permGroup:=p;
    au.elms:=o;
    au.elmsgens:=G.generators;
    au.operations:=PermAutomorphismGroupOps;
    au.morphismDomain:=G;
    au.innerAutomorphisms:=[];
  fi;

  return au;
end;

IsomorphismAbelianGroups := function(G,H)
local o,p,gens,hens;

  # get standard generating system
  if not IsPermGroup(G) then
    p:=PermGroup(G);
    gens:=IndependentGeneratorsAbelianPermGroup(p);
    gens:=List(gens,i->Image(p.bijection,i));
  else
    gens:=IndependentGeneratorsAbelianPermGroup(G);
  fi;

  # get standard generating system
  if not IsPermGroup(H) then
    p:=PermGroup(H);
    hens:=IndependentGeneratorsAbelianPermGroup(p);
    hens:=List(hens,i->Image(p.bijection,i));
  else
    hens:=IndependentGeneratorsAbelianPermGroup(H);
  fi;

  o:=List(gens,i->Order(G,i));
  p:=List(hens,i->Order(H,i));

  SortParallel(o,gens);
  SortParallel(p,hens);

  if o<>p then
    return false;
  fi;

  o:=GroupHomomorphismByImages(G,H,gens,hens);
  o.isMapping:=true;
  o.isInjective:=true;
  o.isSurjective:=true;

  return o;
end;

