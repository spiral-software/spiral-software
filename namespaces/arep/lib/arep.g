# -*- Mode: shell-script -*-
# Term-Algebra for Group Representations
# SE, MP, 13.5.97 - , GAPv3.4

# Nach Submission (AREP 1.1)
# --------------------------
# 27.07.99: KernelARep f"ur monreps besser gemacht, benutzt
#           Formel f"ur Induktionen. RandomMonRep geschrieben.
# AREP 1.2
# --------
# 09.03.00: Kleinen Fehler in ImageARep beseitigt.
# 26.01.01: RandomMonRep gibts nur wenn SmallGroup da ist.
#
# AREP 1.3
# --------
# 08.01.02: kleinen bug in ARepOps.KernelCharacter beseitigt.
#

#F Abstract Representations of Finite Groups (ARep)
#F ================================================
#F
#F The class ARep provides a term algebra for matrix 
#F representations of finite groups. This means there
#F are a number of symbolic constructions for AReps
#F like direct sum or Galois conjugation which return
#F expressions (terms) in the original representations.
#F   The strategy for simplification of the expressions
#F is to construct even trivial expressions and provide
#F functions to apply simplifications. Hence, a term 
#F can be R^id unless you explicitly request simplification.
#F
#F <ARep> ::= 
#F ; atomic cases
#F     <perm>                                 ; "perm" 
#F   | <mon>                                  ; "mon"  
#F   | <mat>                                  ; "mat"
#F 
#F ; composed cases
#F   | <ARep> ^ <AMat>                        ; "conjugate"
#F   | DirectSum(<ARep>, .., <ARep>)          ; "directSum"
#F   | InnerTensorProduct(<ARep>, .., <ARep>) ; "innerTensorProduct"
#F   | OuterTensorProduct(<ARep>, .., <ARep>) ; "outerTensorProduct"
#F   | Restriction(<ARep>, <subgrp>)          ; "restriction"
#F   | Induction(<ARep>, <supergrp>)          ; "induction"
#F   | Extension(<ARep>, <ext-char>)          ; "extension"
#F   | GaloisConjugate(<ARep>, <aut>)         ; "galoisConjugate".
#F
#F An ARep is a GAP-Rec R with the following 
#F mandatory fields common to all types of ARep
#F
#F   R.isARep       := true, identifies AReps
#F   R.operations   := ARepOps, GAP-operations record of R
#F   R.char         : characteristic of the base field
#F   R.degree       : degree of the representation
#F   R.source       : group being represented; the group must
#F                   contain a list R.source.theGenerators
#F                   of generators which are never changed;
#F                   representations are specified by giving
#F                   images for R.source.theGenerators
#F   R.type         : string identifying the type of ARep
#F
#F The following fields are mandatory to special types
#F
#F R.type = "perm" 
#F   R.theImages    : list of Perms for the images of 
#F                    R.source.theGenerators
#F   R.isTransitive : optional field to indicate, that
#F                    R is transitive
#F   R.transitivity : optional field containing the degree of
#F                    transitivity of R, at the moment
#F                    this is an integer
#F   R.induction    : optional field containing a conjugated
#F                    (by a "mon"-AMat) "induction"-ARep of a 
#F                    one dimensional "mon"-ARep equal to R
#F
#F R.type = "mon"
#F   R.theImages    : list of Mons (cf. mon.g)
#F   R.isTransitive : optional field to indicate, that
#F                    R is transitive
#F   R.transitivity : optional field containing the degree of
#F                    transitivity of R, at the moment
#F                    this is an integer
#F   R.induction    : optional field containing a conjugated
#F                    (by a "mon"-AMat) "induction"-ARep of a 
#F                    one dimensional "mon"-ARep equal to R
#F                 
#F R.type = "mat"
#F   R.theImages    : list of Mats
#F
#F R.type = "conjugate"
#F   R.rep          : an ARep to be conjugated
#F   R.conjugation  : an AMat conjugating R.rep
#F
#F R.type = "directSum"
#F   R.summands     : list of AReps of the same R.source, R.char
#F                   The sources of the areps is the same GAP-Rec.
#F
#F R.type = "innerTensorProduct"
#F   R.factors      : list of AReps of the same R.source, R.char
#F                   The sources of the areps is the same GAP-Rec.
#F
#F R.type = "outerTensorProduct"
#F   R.isOuter      : flag to indicate that R.source is the
#F                   outer direct product of the sources of
#F                   the factors; otherwise R.source is the
#F                   inner direct product of the sources of 
#F                   the factors (which are normal subgroups)
#F   R.factors      : list of AReps of the same R.char; note that
#F                   the R.source is the direct product of all
#F                   Rk.source for Rk in R.factors
#F   R.projections  : list of group homomorphisms from R.source
#F                   into R.factors[i].source; the individual
#F                   entries of this list may be empty if 
#F                   the projection has not been used before
#F   R.embeddings   : list of group homomorphisms from the 
#F                   sources of the factors into R.source; the
#F                   entries may be empty if the embeddings
#F                   have not been used before
#F
#F R.type = "galoisConjugate"
#F   R.rep          : an ARep to be conjugated
#F   R.galoisAut    : a Galois automorphism or an integer for 
#F                   the Galois conjugation in CF(n) or GF(p^n)
#F
#F R.type = "restriction"
#F   R.rep          : an ARep of a supergroup of R.source;
#F                   the group R.source and R.rep.source
#F                   have a common parent group
#F
#F R.type = "induction"
#F   R.rep          : an ARep of a subgroup of R.source;
#F                   the group R.rep.source has the same
#F                   parent group as R.source
#F   R.transversal  : a transversal of Cosets(R.source, R.rep.source)
#F
#F R.type = "extension"
#F   R.rep          : an irreducible ARep of a subgroup of 
#F                   R.source with R.rep.character being bound;
#F                   the groups R.source and R.rep.source share
#F                   a common parent group
#F   R.character    : the irreducible character of the extended
#F                   representation of R.source such that the
#F                   restriction to R.rep.source is R.rep.character
#F 
#F Optional fields common to all types of ARep
#F
#F   R.isFaithful    : field to indicate that R is faithful
#F   R.character     : the GAP-character belonging to R
#F   R.isIrreducible : set if R is known to be irreducible or not
#F   R.kernel        : the kernel of R if known
#F   R.hom           : the GAP-group homomorphism corresponding
#F                     to R constructed as
#F                       GroupHomomorphismByImages(
#F                         R.source,              Group(R.theImages),
#F                         R.source.theGenerators, R.theImages )
#F   R.name          : field to give a name to R
#F   R.isPermRep     : flag to indicate if R can be replaced by
#F                     a "perm"-ARep equal to R
#F   R.permARep      : the "perm"-ARep equal to R 
#F   R.isMonRep      : flag to indicate if R can be replaced by
#F                     a "mon"-ARep equal to R
#F   R.monARep       : the "mon"-ARep equal to R
#F   R.matARep       : a "mat"-ARep equal to R
#F            

IsARep    := "defined below";
IsPermRep := "defined below";
IsMonRep     := "defined below";
PermARepARep := "defined below";
MonARepARep  := "defined below";
MatARepARep  := "defined below";


# Internal tools
# --------------

ARepOps :=
  OperationsRecord("ARepOps");

# ARepOps.PrimeFieldOne( <char> )
#   the one element in a given prime field.

ARepOps.PrimeFieldOne := function ( char )
  if char = 0 then
    return 1;
  else
    return Z(char)^0;
  fi;
end;

# ARepOps.CheckGroup( <grp> )
#   make sure <grp> is a group and <grp>.theGenerators is set.
#   Issue a warning if theGenerators is completed.

ARepOps.CheckGroup := function ( grp )
  if not IsGroup(grp) then
    Error("<grp> must be a group");
  fi;
  if not IsBound(grp.theGenerators) then
    Print(
      "#W Warning: unbound <grp>.theGenerators; ",
      "use GroupWithGenerators\n"
    );
    if IsTrivial(grp) then
      grp.theGenerators := [ grp.identity ];
    else
      grp.theGenerators := grp.generators;
    fi;
  fi;
end;

# ARepOps.CheckDegree( <degree>, <int/permgrp/list-of-perm/"infinity" )
#   make sure <degree> specifies a degree >= minDegree which
#   is given as an Int, the largest moved point of a permgroup 
#   or a list of perms or the string "infinity".

ARepOps.CheckDegree := function ( degree, minDegree )
  local m, x;

  if not ( IsInt(degree) and degree >= 1 ) then
    Error("<degree> must be positive integer");
  fi;
  if IsPermGroup(minDegree) then

    # <permgrp>
    if IsTrivial(minDegree) then
      minDegree := 1;
    else
      minDegree := PermGroupOps.LargestMovedPoint(minDegree);
    fi;

  elif IsList(minDegree) and ForAll(minDegree, IsPerm) then

    # <list-of-perm>
    if Length(minDegree) = 0 then
      minDegree := "infinity";
    else
      m := 1;
      for x in minDegree do
        if x <> () then
          m := Maximum(m, LargestMovedPointPerm(x));
        fi;
      od;
      minDegree := m;
    fi;

  fi;
  if IsInt(minDegree) and degree < minDegree then
    Error("<degree> is too small");
  fi;
end;

# ARepOps.CheckedHomomorphism( <grp>, <images> )
#   checks if pointwise mapping of grp.theGenerators onto 
#   images defines a group homomorphism from grp to the
#   group generated by images and if so returns this
#   homomorphism.

ARepOps.CheckedHomomorphism := function ( grp, images )
  local deg, imagegroup, hom;

  if not ( Length(grp.theGenerators) = Length(images) ) then
    Error("grp.theGenerators and images must have the same length");
  fi;

  # construct and check homomorphism
  hom := 
    GroupHomomorphismByImages(
      grp,               Group(images, images[1]^0),
      grp.theGenerators, images
    );
  if not( IsMapping(hom) and IsGroupHomomorphism(hom) ) then
    Error("images define no group homomorphism");
  fi;

  return hom;
end;

# ARepOps.Hom( <arep> )
#   this function computes the .hom-field and 
#   returns the homomorphism.

ARepOps.Hom := function ( R )
  local R1;

  if IsBound(R.hom) then
    return R.hom;
  fi;

  if R.type = "perm" then
    R1    := PermARepARep(R);
    R.hom :=
      GroupHomomorphismByImages(
	R.source,               Group(R1.theImages, ()),
	R.source.theGenerators, R1.theImages
      );
  elif R.type = "mon" then
    R1    := MonARepARep(R);
    R.hom :=
      GroupHomomorphismByImages(
	R.source,               
        Group(R1.theImages, Mon((), R.degree, R.char)),
	R.source.theGenerators, 
        R1.theImages
      );
  elif R.type = "mat" then
    R1    := MatARepARep(R);
    R.hom :=
      GroupHomomorphismByImages(
	R.source,               
        Group(R1.theImages, R1.theImages[1]^0),
	R.source.theGenerators, 
        R1.theImages
      );
  elif IsPermRep(R) then
    R1    := PermARepARep(R);
    R.hom :=
      GroupHomomorphismByImages(
	R.source,               Group(R1.theImages, ()),
	R.source.theGenerators, R1.theImages
      );
  elif IsMonRep(R) then
    R1    := MonARepARep(R);
    R.hom :=
      GroupHomomorphismByImages(
	R.source,               
        Group(R1.theImages, Mon((), R.degree, R.char)),
	R.source.theGenerators, 
        R1.theImages
      );
  else
    R1    := MatARepARep(R);
    R.hom :=
      GroupHomomorphismByImages(
	R.source,               
        Group(R1.theImages, R1.theImages[1]^0),
	R.source.theGenerators, 
        R1.theImages
      );
  fi;

  return R.hom;
end;

# ARepOps.CopyWithNewSource( <arep>, <grp> )
#   constructs a new ARep equal to <arep> having source
#   <grp> if <arep>.source is not *identical* to <grp>.
#   The group <grp> must be equal to <arep>.source and
#   it is updated with information from <arep>.source
#   if this groups has some interesting knowledge.

ARepOps.CopyWithNewSource := function ( R, G )
  local RG;

  if IsIdentical(R.source, G) then
    return R;
  fi;
  ARepOps.CheckGroup(G);
  RG        := ShallowCopy(R); # may be dangerous
  RG.source := G;

  # update knowledge components
  if IsBound(R.size) then

    # copy the size
    RG.size := R.size;

  fi;
  if IsBound(R.name) then

    # copy the name
    RG.name := R.name;

  fi;
  if 
    IsBound(R.source.conjugacyClasses) and
    not IsBound(RG.source.conjugacyClasses)
  then

    # reorganize the conjugacy classes
    RG.source.conjugacyClasses :=
      List(
        R.source.conjugacyClasses,
        C -> ConjugacyClass(G, C.representative)
      );

  fi;
  if IsBound(RG.character) then

    # reorganize the character
    RG.character :=
      Character(
        RG.source,
        List(
          ConjugacyClasses(RG.source), 
          c -> c.representative ^ RG.character
        )
      );
  fi;
  return RG;
end;

# ARepOps.OuterTensorProductProjection( <arep>, <i> )
#   constructs a group homomorphism from <arep>.source
#   into <arep>.factors[i].source as the i-th projection.
#   The homomorphisms are stored in <arep>.projections.

InnerDirectProductProjectionOps :=
  OperationsRecord(
    "InnerDirectProductProjectionOps",
    GroupHomomorphismOps
  );

InnerDirectProductProjectionOps.ImageElm := function ( prj, elm )
  local i, DP, Ps;

  # check if it is in one factor
  for i in [1..Length(prj.factors)] do
    if elm in prj.factors[i] then
      if i = prj.component then
        return elm;
      else
        return prj.range.identity;
      fi;
    fi;
  od;

  # use bijection prj.source -> DirectProduct(prj.factors)
  if not IsBound(prj.bijection) then
    DP := GroupOps.DirectProduct(prj.factors);
    Ps := 
      List(
        [1..Length(prj.factors)], 
        i -> Projection(DP, prj.factors[i], i)
      );

    prj.bijection :=
      GroupHomomorphismByImages(
        prj.source, 
        DP,
        List(
          DP.generators,
          g -> Product(List(Ps, P -> g^P))
        ),
        DP.generators
      );
    prj.bijection.isMapping      := true;
    prj.bijection.isHomomorphism := true;
    prj.bijection.isSurjective   := true;
    prj.bijection.isInjective    := true;
    prj.bijection.isBijective    := true;
    prj.bijection.image          := DP;
    prj.bijection.kernel         := TrivialSubgroup(prj.source);
  fi;

  return 
    (elm ^ prj.bijection) ^
    Projection(
      prj.bijection.range, 
      prj.factors[prj.component], 
      prj.component
    );
end;

InnerDirectProductProjectionOps.ImagesElm := function ( prj, elm )
  return 
    [ InnerDirectProductProjectionOps.ImageElm(prj, elm) ];
end;

InnerDirectProductProjectionOps.ImageRepresentative := 
  InnerDirectProductProjectionOps.ImageElm;

InnerDirectProductProjectionOps.PreImagesRepresentative :=
  function ( prj, img )
    return img;
  end;

InnerDirectProductProjectionOps.Print := function ( prj )
  ProjectionDirectProductOps.Print(prj);
end;

ARepOps.OuterTensorProductProjection := function ( R, i )
  if IsBound(R.projections[i]) then
    return R.projections[i];
  fi;
  if R.isOuter then
    R.projections[i] := 
      Projection(R.source, R.factors[i].source, i);
  else
    R.projections[i] :=
      rec(
        isGeneralMapping := true,
        isMapping        := true,
        isHomomorphism   := true,
        isProjection     := true,
        isSurjective     := true,
        source           := R.source,
        factors          := List(R.factors, Ri -> Ri.source),
        range            := R.factors[i].source,
        component        := i,
        operations       := InnerDirectProductProjectionOps
      );
  fi;
  return R.projections[i];
end;


# ARepOps.OuterTensorProductEmbedding( <arep>, <i> )
#   constructs a group homomorphism from 
#   <arep>.factors[i].source into <arep>.source as the 
#   i-th embedding. The homomorphisms are stored in 
#   <arep>.embedding.

ARepOps.OuterTensorProductEmbedding := function ( R, i )
  if IsBound(R.embeddings[i]) then
    return R.embeddings[i];
  fi;
  if R.isOuter then
    R.embeddings[i] := 
      Embedding(R.factors[i].source, R.source, i);
  else
    R.embeddings[i] := 
      IdentityMapping(R.factors[i].source);
  fi;
  return R.embeddings[i];
end;


# ARepOps.PermConjugacyClasses( <conjClasses>, <reps> )
#   returns a permutation which sorts the list <reps> of
#   representatives of the conjugacy classes <conjClasses>
#   into the ordering in <conjClasses>.

ARepOps.PermConjugacyClasses := function ( ccs, reps )
  local s, r, i;

  s := [ ];
  for r in reps do
    i := PositionProperty(ccs, c -> c.representative = r);
    if i = false then
      i := PositionProperty(ccs, c -> r in c);
      if i = false then
        Error("panic! Cannot order <reps> to conjugacyClasses <ccs>");
      fi;
    fi;
    Add(s, i);
  od;
  return PermList(s);
end;


# ARepOps.KernelCharacter( <character> )
#   determines the kernel of the character

ARepOps.KernelCharacter := function ( chi )
  local C, cc, size, N, g;

  if not IsCharacter(chi) then
    Error("usage: ARepOps.KernelCharacter( <character> )");
  fi;

  # catch trivial case
  if ForAll(chi.values, x -> x = Degree(chi) * x^0) then
    return chi.source;
  fi;

  # determine conjugacy classes c where chi is
  # trivial, i.e. chi(c) = deg(chi)
  C := [ ];
  for cc in ConjugacyClasses(chi.source) do
    if cc.representative^chi = Degree(chi) then
      Add(C, cc);
    fi;
  od;

  # construct the kernel N by drawing random
  # elements until full size is reached
  size  := Sum(List(C, Size));
  N     := TrivialSubgroup(chi.source);
  while not( Size(N) = size ) do
    g     := Random(Random(C));
    if not( g in N ) then
      N     := Closure(N, g);
    fi;
  od;

  return N;
end;


# ARepOps.IsOneRep( <arep> )
#   tests if <arep> is the one-representation of 
#   arbitrary degree

ARepOps.IsOneRep := function ( R )
  if not IsARep(R) then 
    Error("usage: ARepOps.IsOneRep( <arep> )");
  fi;
  return
    IsPermRep(R) and 
    ForAll(List(R.source.theGenerators, g -> g ^ R), IsIdentityMat);
end;

# ARepOps.IsTrivialOneRep( <arep> )
#   tests if <arep> is the one-representation of degree 1

IsARep    := "defined below";
IsPermRep := "defined below";

ARepOps.IsTrivialOneRep := function ( R )
  if not IsARep(R) then 
    Error("usage: ARepOps.IsTrivialOneRep( <arep> )");
  fi;
  return R.degree = 1 and IsPermRep(R);
end;


# ARepOps.MinkwitzExtension( <arep> )
#   constructs a homomorphism from the "extension"-ARep
#   <arep>.source into a MatGroup. This is done with
#   Minkwitz' extension formula summing over the subgroup.

ImageARep := "defined below";

ARepOps.MinkwitzExtension := function ( R )
  local gens, images, grp, g, ig, h, Rh, id;

  # find a short generating set for R.source/R.rep.source
  # (which is a group iff R.rep.source is normal in R.source)
  gens := [ ];
  grp  := R.rep.source;
  while Size(grp) < Size(R.source) do
    g := Random(R.source);
    if not g in grp then
      Add(gens, g);
      grp := Closure(grp, g);
    fi; 
  od;

  # compute the images for gens
  id     := IdentityMat(R.degree, ARepOps.PrimeFieldOne(R.char));
  images := List(gens, g -> 0*id);
  for h in Elements(R.rep.source) do
    Rh := MatAMat( ImageARep(h, R.rep) );
    for ig in [1..Length(gens)] do
      images[ig] := 
        images[ig] + 
        ((gens[ig] * h^-1) ^ R.character) * Rh;
    od;
  od;
  for ig in [1..Length(gens)] do
    images[ig] := 
      Degree(R.rep.character)/Size(R.rep.source) * 
      images[ig];
  od;

  # add R.rep.source.theGenerators and R.rep.theImages
  Append(
    gens,
    R.rep.source.theGenerators
  );
  Append(
    images, 
    List(ImageARep(R.rep.source.theGenerators, R.rep), MatAMat)
  );
  return
    GroupHomomorphismByImages(
      R.source, Group(images, 
                      IdentityMat(R.degree, ARepOps.PrimeFieldOne(R.char))),
      gens,     images
    );
end;


# ARepOps.ConjugationPermTransversalNC( 
#   <grp>, <subgrp>, <transversal1>, <transversal2> 
# )
#   calculates a permutation p on Index( <grp>, <subgrp> )
#   many points mapping the cosets given in <transversal1> 
#   onto the cosets given in <transversal2>, namely
#     <subgrp> * <transversal1>[i ^ p] = 
#     <subgrp> * <transversal2>[i], for all i
#   The arguments are not checked.

ARepOps.ConjugationPermTransversalNC := function ( G, H, T1, T2 )
  local T1a, T2a, cos1, cos2;

  # catch the case, that H is trivial, hence
  # T1 and T2 are lists of all elements in G
  if IsTrivial(H) then
    T1a := ShallowCopy(T1);
    T2a := ShallowCopy(T2);
    return Sortex(T1) * Sortex(T2) ^ -1;
  fi;

  # build cosets and sort
  cos1 := List(T1, x -> Coset(H, x));
  cos2 := List(T2, x -> Coset(H, x));
  return Sortex(cos1) * Sortex(cos2) ^ -1;
end;


#F Fixing generators
#F -----------------
#F

#F GroupWithGenerators( <group> )
#F GroupWithGenerators( <list-of-grpelts> )
#F   construct a group with the field .theGenerators being
#F   set to a fixed non-empty generating list. If <group>
#F   is given then this group record is modified by copying
#F   the field .generators to the field .theGenerators 
#F   if this does not exist already.
#F   If <list-of-grpelts> is given then this must be a non-
#F   empty list of group elements acting as the fixed generating
#F   list of the resulting group.
#F

GroupWithGenerators := function ( groupOrGens )
  local G, gens;

  if IsGroup(groupOrGens) then

    # <group> 
    G := groupOrGens;
    if IsBound(G.theGenerators) then
      return G;
    fi;
    if IsTrivial(G) then
      G.theGenerators := [ G.identity ];
    else
      G.theGenerators := G.generators;
    fi;
    return G;

  elif IsList(groupOrGens) then

    # <list-of-grpelts>
    gens := groupOrGens;
    if Length(gens) = 0 then
      Error("<gens> must be not empty");
    fi;
    G := Group(gens, gens[1]^0);
    G.theGenerators := gens;
    return G;

  else
    Error(
      "usage:\n",
      "  GroupWithGenerators( <group> )\n",
      "  GroupWithGenerators( <list-of-grpelts> )"
    );
 fi;
end;


#F Fundamental Constructors for AReps
#F ----------------------------------
#F 

#F IsARep( <obj> )
#F   tests of <obj> is an ARep.
#F

IsARep := function ( x )
  return IsRec(x) and IsBound(x.isARep) and x.isARep;
end;

#F TrivialPermARep( <grp> [, <degree> [,<char/field> ] )
#F TrivialMonARep(  <grp> [, <degree> [,<char/field> ] )
#F TrivialMatARep(  <grp> [, <degree> [,<char/field> ] )
#F   the "perm"/"mon"/"mat"-ARep of mapping every group element 
#F   onto a one of degree <degree>. The default for the degree is 1, 
#F   the  default for the characteristic is 0.
#F

TrivialPermARep := function ( arg )
  local grp, deg, char;

  if Length(arg) = 1 then
    grp  := arg[1];
    deg  := 1;
    char := 0;
  elif Length(arg) = 2 then
    grp  := arg[1];
    deg  := arg[2];
    char := 0;
  elif Length(arg) = 3 then
    grp  := arg[1];
    deg  := arg[2];
    char := arg[3];
  else
    Error("usage: TrivialPermARep( <grp>, [, <char/field> ] )");
  fi;

  ARepOps.CheckGroup(grp);
  ARepOps.CheckDegree(deg, "infinity");
  char := AMatOps.CheckChar(char);

  return
    rec(
      isARep        := true,
      operations    := ARepOps,
      char          := char,
      degree        := deg,
      source        := grp,
      type          := "perm",
      theImages     := List(grp.theGenerators, g -> ()),

      isIrreducible := true,
      kernel        := grp
    );
end;

TrivialMonARep := function ( arg )
  local grp, deg, char;

  if Length(arg) = 1 then
    grp  := arg[1];
    deg  := 1;
    char := 0;
  elif Length(arg) = 2 then
    grp  := arg[1];
    deg  := arg[2];
    char := 0;
  elif Length(arg) = 3 then
    grp  := arg[1];
    deg  := arg[2];
    char := arg[3];
  else
    Error("usage: TrivialMonARep( <grp>, [, <char/field> ] )");
  fi;

  ARepOps.CheckGroup(grp);
  ARepOps.CheckDegree(deg, "infinity");
  char := AMatOps.CheckChar(char);

  return
    rec(
      isARep        := true,
      operations    := ARepOps,
      char          := char,
      degree        := deg,
      source        := grp,
      type          := "mon",
      theImages     := 
        List(
          grp.theGenerators, 
          g -> Mon(List([1..deg], i -> ARepOps.PrimeFieldOne(char)))
        ),

      isIrreducible := true,
      kernel        := grp
    );
end;

TrivialMatARep := function ( arg )
  local grp, deg, char;

  if Length(arg) = 1 then
    grp  := arg[1];
    deg  := 1;
    char := 0;
  elif Length(arg) = 2 then
    grp  := arg[1];
    deg  := arg[2];
    char := 0;
  elif Length(arg) = 3 then
    grp  := arg[1];
    deg  := arg[2];
    char := arg[3];
  else
    Error("usage: TrivialMatARep( <grp>, [, <char/field> ] )");
  fi;

  ARepOps.CheckGroup(grp);
  ARepOps.CheckDegree(deg, "infinity");
  char := AMatOps.CheckChar(char);

  return
    rec(
      isARep        := true,
      operations    := ARepOps,
      char          := char,
      degree        := deg,
      source        := grp,
      type          := "mat",
      theImages     := 
        List(
          grp.theGenerators, 
          g -> IdentityMat(deg, ARepOps.PrimeFieldOne(char))
        ),

      isIrreducible := (deg = 1),
      kernel        := grp
    );
end;

#F RegularARep( <grp> [, <char/field> ] )
#F   returns an "induction"-arep of the onerep on 
#F   the trivial subgroup of <grp>.
#F

InductionARep       := "defined below";

RegularARep := function ( arg )
  local grp, char, R;

  # decode and check arguments
  if Length(arg) = 1 then
    grp  := arg[1];
    char := 0;
  elif Length(arg) = 2 then
    grp  := arg[1];
    char := arg[2];
  else
    Error("usage: RegularARep( <grp> [, <char/field> ] )");
  fi;
  ARepOps.CheckGroup(grp);
  char := AMatOps.CheckChar(char);

  R := 
    InductionARep(
      TrivialPermARep(
        GroupWithGenerators(TrivialSubgroup(grp)), 
        1, 
        char
      ), 
      grp
    );

  # store valuable information
  R.isFaithful   := true;
  R.transitivity := 1;

  return R;
end;


#F NaturalARep( <matgrp> )
#F NaturalARep( <mongrp> )
#F NaturalARep( <permgrp>, <degree> [, <char/field> ] )
#F   a group taken as a representation of itself.
#F

NaturalARep := function ( arg )
  local grp, degree, char;

  if Length(arg) = 1 and IsMatGroup(arg[1]) then

    # <matgrp>
    grp := arg[1];
    ARepOps.CheckGroup(grp);
    return
      rec(
        isARep     := true,
        operations := ARepOps,
        char       := DefaultField([grp.identity[1][1]]).char,
        degree     := Length(grp.identity),
        source     := grp,
        type       := "mat",
        theImages  := grp.theGenerators,

        kernel     := TrivialSubgroup(grp),
        hom        := IdentityMapping(grp)
      );

  elif Length(arg) = 1 and IsMon(arg[1].identity) then
    
    # <mongrp>
    grp := arg[1];
    ARepOps.CheckGroup(grp);
    return
      rec(
        isARep     := true,
        operations := ARepOps,
        char       := DefaultField([grp.identity.diag[1]]).char,
        degree     := Length(grp.identity.diag),
        source     := grp,
        type       := "mon",
        theImages  := grp.theGenerators,

        kernel     := TrivialSubgroup(grp),
        hom        := IdentityMapping(grp)
      );
    
  elif Length(arg) in [2, 3] and IsPermGroup(arg[1]) then

    # <permgrp>, <degree> [, <char/field> ]
    grp    := arg[1];
    degree := arg[2];
    if Length(arg) = 3 then
      char := arg[3];
    else
      char := 0;
    fi;

    ARepOps.CheckGroup(grp);
    ARepOps.CheckDegree(degree, grp);
    char := AMatOps.CheckChar(char);
    return
       rec(
        isARep        := true,
        operations    := ARepOps,
        char          := char,
        degree        := degree,
        source        := grp,
        type          := "perm",
        theImages     := grp.theGenerators,

        kernel        := TrivialSubgroup(grp),
        hom           := IdentityMapping(grp)
      );

  else
    Error(
      "usage:\n",
      "  NaturalARep( <matgrp> )\n",
      "  NaturalARep( <mongrp> )\n",
      "  NaturalARep( <permgrp>, <degree> [, <char/field> ] )"
    );
  fi;
end;

#F ARepByImages( <grp>, <list-of-perm>, <degree> 
#F               [, <char/field> ] [, <hint> ] )
#F ARepByImages( <grp>, <list-of-mon> [, <hint> ] )
#F ARepByImages( <grp>, <list-of-mat> [, <hint> ] )
#F   the representation defined by mapping the list of generators
#F   grp.theGenerators pointwise onto the elements of the list
#F   given as the second argument. The optional argument <hint>
#F   is a string which gives a hint to avoid the check if the 
#F   list of images actually define a group homomorphism. The
#F   possible hints are "hom" (image do define a homomorphism) 
#F   and "faithful" (image define an injective homomorphism).
#F

ARepByImages := function ( arg )
  local grp, images, degree, char, hint, imagegrp, hom, R; 

  if 
    Length(arg) in [3, 4, 5] and 
    IsList(arg[2]) and 
    ForAll(arg[2], IsPerm) and
    IsInt(arg[3])
  then

    # <grp>, <list-of-perm>, <degree> [, <char/field> ] [, <hint> ]
    grp    := arg[1];
    images := arg[2];
    degree := arg[3];
    ARepOps.CheckGroup(grp);
    if not ( Length(images) = Length(grp.theGenerators) ) then
      Error("Length(<images>) must equal Length(<grp>.theGenerators)");
    fi;
    ARepOps.CheckDegree(degree, images);
    if Length(arg) = 3 then
      char := 0;
      hint := "no hint";
    elif Length(arg) = 4 then
      if arg[4] in ["no hint", "hom", "faithful"] then
        char := 0;
        hint := arg[4];
      else
        char := arg[4];
        hint := "no hint";
      fi;
    else # Length(arg) = 5 
      char := arg[4];
      hint := arg[5];
    fi;
    char := AMatOps.CheckChar(char);
    if not( hint in ["no hint", "hom", "faithful"] ) then
      Error("<hint> must be \"hom\" or \"faithful\"");
    fi;
    
    R := 
      rec(
       isARep        := true,
       operations    := ARepOps,
       char          := char,
       degree        := degree,
       source        := grp,
       type          := "perm",
       theImages     := images,

       isIrreducible := degree = 1,
       isPerm        := true,
       isMon         := true
     );

    # check if input defines a group homomorphism
    # if no hint is given
    if not hint in ["hom", "faithful"] then
      R.hom := ARepOps.CheckedHomomorphism(grp, images);
    fi;
    if hint = "faithful" then
      R.kernel := TrivialSubgroup(grp);
    fi;
    return R;

  elif
    Length(arg) in [2, 3] and
    IsList(arg[2]) and
    Length(arg[2]) > 0 and
    ForAll(arg[2], IsMon)
  then

    # <grp>, <list-of-mon> [, <hint> ]
    grp    := arg[1];
    images := arg[2];
    degree := Length(images[1].diag);
    if not ForAll(images, p -> Length(p.diag) = degree) then
      Error("<images> must have the same degree");
    fi;
    char   := images[1].char;
    if not ForAll(images, p -> p.char = char) then
      Error("<images> must have the same char");
    fi;
    ARepOps.CheckGroup(grp);
    if Length(arg) = 3 then
      hint := arg[3];
    else
      hint := "no hint";
    fi;
    if not ( hint in ["no hint", "hom", "faithful"] ) then
      Error("<hint> must be \"hom\" or \"faithful\"");
    fi;

    R := 
      rec(
       isARep        := true,
       operations    := ARepOps,
       char          := char,
       degree        := degree,
       source        := grp,
       type          := "mon",
       theImages     := images,

       isMon         := true
     );

    # check if input defines a group homomorphism
    # if no hint is given
    if not hint in ["hom", "faithful"] then
      R.hom := ARepOps.CheckedHomomorphism(grp, images);
    fi;
    if hint = "faithful" then
      R.kernel := TrivialSubgroup(grp);
    fi;
    return R;

  elif 
    Length(arg) in [2, 3] and
    IsList(arg[2]) and
    Length(arg[2]) > 0 and
    ForAll(arg[2], IsMat)
  then

    # <grp>, <list-of-mat> [, <hint> ]
    grp    := arg[1];
    images := arg[2];
    degree := Length(images[1]);
    char   := DefaultField([images[1][1][1]]).char;
    ARepOps.CheckGroup(grp);
    if not( 
      ForAll(images, m -> DimensionsMat(m) = [degree, degree]) 
    ) then
      Error("mats must be square and have equal size");
    fi;
    if Length(arg) = 3 then
      hint := arg[3];
    else
      hint := "no hint";
    fi;        
    if not( hint in ["no hint", "hom", "faithful"] ) then
      Error("<hint> must be \"hom\" or \"faithful\"");
    fi;
    
    R := 
      rec(
       isARep        := true,
       operations    := ARepOps,
       char          := char,
       degree        := degree,
       source        := grp,
       type          := "mat",
       theImages     := images
     );

    # check if input defines a group homomorphism
    # if no hint is given
    if not hint in ["hom", "faithful"] then
      R.hom := ARepOps.CheckedHomomorphism(grp, images);
    fi;
    if hint = "faithful" then
      R.kernel := TrivialSubgroup(grp);
    fi;
    if degree = 1 then
      R.isIrreducible := true;
    fi;
    return R;

  else
    Error("usage:\n", 
          "  ARepByImages( <grp>, <list-of-perm>, ",
            "<degree> [, <char/field> ] [, <hint> ] )\n",
          "  ARepByImages( <grp>, <list-of-mon> [, <hint> ] )\n",
          "  ARepByImages( <grp>, <list-of-mat> [, <hint> ] )"
    );
  fi;     
end;


#F ARepByHom( <grphom-to-matgrp> )
#F ARepByHom( <grphom-to-mongrp> )
#F ARepByHom( <grphom-to-permgrp>, <degree> [, <char/field> ] )
#F   the representation defined by a group homomorphism.
#F

ARepByHom := function ( arg )
  local hom, grp, degree, char, R;

  if 
    Length(arg) = 1 and
    IsMapping(arg[1]) and
    IsGroupHomomorphism(arg[1]) and
    IsMatGroup(arg[1].range)
  then

    # <grphom-to-matgrp>
    hom    := arg[1];
    grp    := hom.source;
    degree := Length(hom.range.identity);
    char   := DefaultField([hom.range.identity[1][1]]).char;
    ARepOps.CheckGroup(grp);

    R := 
      rec(
       isARep        := true,
       operations    := ARepOps,
       char          := char,
       degree        := degree,
       source        := grp,
       type          := "mat",
       theImages     := List(grp.theGenerators, g -> g^hom)
     );
    if degree = 1 then
      R.isIrreducible := true;
    fi;
    if IsBound(hom.kernel) then
      R.kernel := hom.kernel;
    fi;

    return R;

  elif 
    Length(arg) = 1 and
    IsMapping(arg[1]) and
    IsGroupHomomorphism(arg[1]) and
    IsMon(arg[1].range.identity)
  then

    # <grphom-to-mongrp>
    hom    := arg[1];
    grp    := hom.source;
    degree := Length(hom.range.identity.diag);
    char   := DefaultField([hom.range.identity.diag[1]]).char;
    ARepOps.CheckGroup(grp);

    R := 
      rec(
       isARep        := true,
       operations    := ARepOps,
       char          := char,
       degree        := degree,
       source        := grp,
       type          := "mon",
       theImages     := List(grp.theGenerators, g -> g^hom),
      
       isMon         := true
     );
    if degree = 1 then
      R.isIrreducible := true;
    fi;
    if IsBound(hom.kernel) then
      R.kernel := hom.kernel;
    fi;

    return R;
    
  elif
    Length(arg) in [2, 3] and
    IsMapping(arg[1]) and
    IsGroupHomomorphism(arg[1]) and
    IsPermGroup(arg[1].range)
  then

    # <grphom-to-permgrp>, <degree> [, <char/field> ]
    hom    := arg[1];
    grp    := hom.source;
    degree := arg[2];
    if Length(arg) = 3 then
      char := arg[3];
    else
      char := 0;
    fi;
    ARepOps.CheckGroup(grp);
    ARepOps.CheckDegree(degree, hom.range);
    char := AMatOps.CheckChar(char);

    R := 
      rec(
       isARep        := true,
       operations    := ARepOps,
       char          := char,
       degree        := degree,
       source        := grp,
       type          := "perm",
       theImages     := List(grp.theGenerators, g -> g^hom),

       isIrreducible := (degree = 1),
       isPerm        := true,
       isMon         := true
     );
    if IsBound(hom.kernel) then
      R.kernel := hom.kernel;
    fi;

    return R;

  else
    Error(
      "usage:\n",
      "  ARepByHom( <grphom-to-matgrp> )\n",
      "  ARepByHom( <grphom-to-mongrp> )\n",
      "  ARepByHom( <grphom-to-permgrp>, <degree> [, <char/field> ] )"
    );      
  fi;
end;


#F ARepByCharacter( <1dim-character> )
#F   the monomial representation defined by a 
#F   1-dimensional character.
#F

ARepByCharacter := function ( chi )
  local grp, field, R;

  if not( 
    IsMapping(chi) and 
    IsCharacter(chi) 
  ) then
    Error("usage: ARepByCharacter( <1dim-character> )");
  fi;
  
  # check if chi is one-dimensional
  grp := chi.source;
  field := DefaultField(chi.values);
  if not( grp.identity^chi = field.one ) then
    Error("character must be one-dimensional");
  fi;
  ARepOps.CheckGroup(grp);

  R :=
    rec(
      isARep        := true,
      operations    := ARepOps,
      char          := field.char,
      degree        := 1,
      source        := grp,
      type          := "mon",
      theImages     := 
        List(grp.theGenerators, g -> Mon((), [g^chi])),

      character     := chi,
      isIrreducible := true,
      isMon         := true
    );
  if IsBound(chi.kernel) then
    R.kernel := chi.kernel;
  fi;

  return R;      
end;


#F Structural Symbolic Constructors for AReps
#F ------------------------------------------
#F

#F ConjugateARep( <arep>, <amat> [, <hint> ] )
#F <arep> ^ <amat> ; shorthand
#F   the representation <arep> conjugated with an invertible matrix 
#F   represented by <amat>, an object of type AMat. Note that the
#F   <hint> can be the string "invertible" to avoid the check for
#F   <amat> to be invertible.
#F

ConjugateARep := function ( arg )
  local R, A, hint;

  # decode and check arg
  if Length(arg) = 2 then
    R    := arg[1];
    A    := arg[2];
    hint := "no hint";
  elif Length(arg) = 3 then
    R    := arg[1];
    A    := arg[2];
    hint := arg[3];
  else
    Error("usage: ConjugateARep( <arep>, <amat> [, <hint> ] )");
  fi;
  if not IsARep(R) then
    Error("<R> must be an ARep");
  fi;
  if not (
    IsAMat(A) and 
    A.dimensions[1] = A.dimensions[2] and
    A.dimensions[1] = R.degree
  ) then
    Error("<A> must be a square AMat of size <R>.degree");
  fi;
  if not R.char = A.char then
    Error("<R> and <A> must have common char");
  fi;
  if not hint in ["no hint", "invertible"] then
    Error("<hint> must be \"no hint\" or \"invertible\"");
  fi;
  if hint = "no hint" then
    if not IsInvertibleMat(A) then
      Error("<A> must be invertible");
    fi;
  fi;

  # return result
  return
    rec(
      isARep      := true,
      operations  := ARepOps,
      char        := R.char,
      degree      := R.degree,
      source      := R.source,
      type        := "conjugate",
      rep         := R,
      conjugation := A
    );
end;


#F DirectSumARep( <arep1>, .., <arepN> ) ; N >= 1
#F DirectSumARep( <list-of-areps> )
#F   direct sum of AReps. Note that the areps have to represent a
#F   common group in a common characteristic. 
#F

DirectSumARep := function ( arg )
  local i;

  # make arg the list of AMats and check source, char
  if Length(arg) = 1 and IsList(arg[1]) then
    arg := arg[1];
  fi;
  if Length(arg) = 0 then
    Error("must have at least one summand");
  fi;
  if not ForAll(arg, IsARep) then
    Error("<arg> must consist of AReps");
  fi;
  for i in [2..Length(arg)] do
    if not arg[i].char = arg[1].char then
      Error("sorry, no common characteristic in <arg>");
    fi;
    if not arg[i].source = arg[1].source then
      Error("sorry, no common source group in <arg>");
    fi;
  od;

  # make the sources *identical*
  for i in [2..Length(arg)] do
    arg[i] := ARepOps.CopyWithNewSource(arg[i], arg[1].source);
  od;

  # construct the result
  return
    rec(
      isARep        := true,
      operations    := ARepOps,
      char          := arg[1].char,
      degree        := Sum(List(arg, Ri -> Ri.degree)),
      source        := arg[1].source,
      type          := "directSum",
      summands      := arg
    );
end;


#F InnerTensorProductARep( <arep1>, .., <arepN> ) ; N >= 1
#F InnerTensorProductARep( <list-of-areps> )
#F   inner tensor product of AReps. Note that the areps have to 
#F   represent a common group in a common characteristic. 
#F

InnerTensorProductARep := function ( arg )
  local i;

  # make arg the list of AMats and check source, char
  if Length(arg) = 1 and IsList(arg[1]) then
    arg := arg[1];
  fi;
  if Length(arg) = 0 then
    Error("must have at least one factor");
  fi;
  if not ForAll(arg, IsARep) then
    Error("<arg> must consist of AReps");
  fi;
  for i in [2..Length(arg)] do
    if not arg[i].char = arg[1].char then
      Error("sorry, no common characteristic in <arg>");
    fi;
    if not arg[i].source = arg[1].source then
      Error("sorry, no common source group in <arg>");
    fi;
  od;

  # make the sources *identical*
  for i in [2..Length(arg)] do
    arg[i] := ARepOps.CopyWithNewSource(arg[i], arg[1].source);
  od;
  
  # construct the result
  return
    rec(
      isARep        := true,
      operations    := ARepOps,
      char          := arg[1].char,
      degree        := Product(List(arg, Ri -> Ri.degree)),
      source        := arg[1].source,
      type          := "innerTensorProduct",
      factors       := arg
    );
end;


#F OuterTensorProductARep( [ <grp> ,] <arep1>, .., <arepN> ) ; N >= 1
#F OuterTensorProductARep( [ <grp> ,] <list-of-arep> )
#F   outer tensor product of AReps. Note that the areps have to 
#F   have a common characteristic. If the first argument is the
#F   optional <grp>, then this group is the source of the result.
#F   Note that <grp> has to be the inner direct product of the
#F   sources of all factors and that this is not tested! If no
#F   <grp> is given, then the GAP-function DirectProduct() is
#F   used to construct the outer direct product of the sources
#F   of the factors.
#F

OuterTensorProductARep := function ( arg )
  local isOuter, grp, i;

  # split off <grp> if present
  if Length(arg) >= 1 and IsGroup(arg[1]) then
    isOuter := false;
    grp     := arg[1];
    arg     := Sublist(arg, [2..Length(arg)]);
  else
    isOuter := true;
  fi;

  # reduce <list-of-arep> to the <arep1>, .., <arepN>
  if Length(arg) = 1 and IsList(arg[1]) then
    arg := arg[1];
  fi;

  # make sure there N >= 1 AReps of the same char
  if Length(arg) = 0 then
    Error("must have at least one factor");
  fi;
  for i in [2..Length(arg)] do
    if not arg[i].char = arg[1].char then
      Error("sorry, no common characteristic in <arg>");
    fi;
  od;

  # provide grp 
  if isOuter then
    if Length(arg) = 1 then
      grp := arg[1].source;
    else
      grp := 
        GroupOps.DirectProduct(
          List(arg, Ri -> Ri.source)
        );
      grp.theGenerators :=
        grp.generators;
    fi;
  fi;
  ARepOps.CheckGroup(grp);

  # construct the result
  return
    rec(
      isARep      := true,
      operations  := ARepOps,
      char        := arg[1].char,
      degree      := Product(List(arg, Ri -> Ri.degree)),
      source      := grp,
      type        := "outerTensorProduct",
      isOuter     := isOuter,
      factors     := arg,
      projections := [ ],
      embeddings  := [ ]
    );
end;


#F GaloisConjugateARep( <arep>, <gal-aut/int> )
#F   the Galois conjugate of <arep> with the galois conjugation
#F   defined by <gal-aut/int>. This can be a field automorphism
#F   or an integer k, in which case x -> x^(FrobeniusAut^k) or 
#F   x -> GaloisCyc(x, k) is meant.
#F

GaloisConjugateARep := function ( R, galoisAut ) 
  if not IsARep(R) then
    Error("<R> must be an ARep");
  fi;
  if not AMatOps.IsGaloisAut(galoisAut) then
    Error("<galoisAut> must specify a Galois automorphism");
  fi;

  return 
    rec(
      isARep     := true,
      operations := ARepOps,
      char       := R.char,
      degree     := R.degree,
      source     := R.source,
      type       := "galoisConjugate",
      rep        := R,
      galoisAut  := galoisAut
    );
end;


#F RestrictionARep( <arep>, <subgrp> )
#F   the representation <arep> restricted to a subgroup.
#F   It is allowed that <subgrp> does not have the same
#F   parent group as <arep>.source.
#F

RestrictionARep := function ( R, subgrp )
  local subgrp1;

  if not IsARep(R) then
    Error("<R> must be an ARep");
  fi;

  # make sure Parent(subgrp) identical Parent(R.source)
  ARepOps.CheckGroup(subgrp);
  if not ForAll(subgrp.theGenerators, g -> g in R.source) then
    Error("<subgrp> must be a subgroup of <R>.source");
  fi;
  if not IsIdentical(Parent(subgrp), Parent(R.source)) then
    subgrp1 := AsSubgroup(R.source, subgrp);
    subgrp1.theGenerators := subgrp.theGenerators;
    if IsBound(subgrp.name) then
      subgrp1.name := subgrp.name;
    fi;
  else
    subgrp1 := subgrp;
  fi;

  return
    rec(
      isARep     := true,
      operations := ARepOps,
      char       := R.char,
      degree     := R.degree,
      source     := subgrp1,
      type       := "restriction",
      rep        := R
    );
end;


#F InductionARep( <arep>, <supergroup> [, <transversal> ] )
#F   the induced representation <arep> on the <supergroup>.
#F   If no transversal is provided then the function will
#F   choose one as [t_1, .., t_r]. The convention for the
#F   induction is RG = g -> [ RHDot(t_i g t_j^-1) | i, j ],
#F   what implies, that [t_1, .., t_r] is a right transversal.
#F   It is allowed that <supergroup> does not have the same
#F   parent group as <arep>.source. The given <transversal>
#F   is not checked to be one.
#F

InductionARep := function ( arg )
  local R, grp, supgrp, transv;

  # decode and check arg
  if Length(arg) = 2 then
    R      := arg[1];
    supgrp := arg[2];
    transv := "automatic";
  elif Length(arg) = 3 then
    R      := arg[1];
    supgrp := arg[2];
    transv := arg[3];
  fi;
  if not IsARep(R) then
    Error("<R> must be an ARep");
  fi;
  ARepOps.CheckGroup(supgrp);

  # make sure R.source <= supgrp with the same Parent group
  if not ForAll(R.source.theGenerators, g -> g in supgrp) then
    Error("<supgrp> must contain <R>.source");
  fi;
  if not IsIdentical(Parent(R.source), Parent(supgrp)) then
    grp               := Subgroup(supgrp, R.source.theGenerators);
    grp.theGenerators := R.source.theGenerators;
    if IsBound(R.source.name) then
      grp.name := R.source.name;
    fi;
    R := 
      ARepOps.CopyWithNewSource(
        R,
        grp
      );
  fi; 

  # get a transversal
  if transv = "automatic" then

    # catch regular case
    if IsTrivial(R.source) then
      transv := Elements(supgrp);
      if not IsSet(transv) then
        Error("GAP did not enumerate elements of <supgrp> as a set");
      fi;
    else
      transv := RightTransversal(supgrp, R.source);
    fi;

  else

    if not (
      IsList(transv) and 
      Length(transv) = Index(supgrp, R.source) and
      ForAll(transv, x -> x in supgrp)
    ) then
      Error(
        "<transv> must be a right transversal of <R>.source\\<supgrp>"
      );
    fi;

  fi;

  return
    rec(
      isARep      := true,
      operations  := ARepOps,
      char        := R.char,
      degree      := R.degree * Index(supgrp, R.source),
      source      := supgrp,
      type        := "induction",
      rep         := R,
      transversal := transv
    );
end;


#F ExtensionARep( <arep>, <extending-character> )
#F   the representation <arep> extended to a representation
#F   of a supergroup affording the extending character.
#F   Note that the extending character and the character of 
#F   <arep> must both be irreducible. (The extension is 
#F   evaluated using Minkwitz's extension formula.)
#F   It is allowed that <supergroup> does not have the same
#F   parent group as <arep>.source.
#F   This function only works for <R>.char = 0.
#F

CharacterARep := "defined below";

ExtensionARep := function ( R, chi )
  local subgrp, chiR, i;

  # check simple conditions on arguments
  if not IsARep(R) then
    Error("<R> must be an ARep");
  fi;
  if not IsCharacter(chi) then
    Error("<chi> must be a Character");
  fi;

  # check char = 0
  if R.char <> 0 then
    Error("<R>.char must be zero");
  fi;

  # make sure R.source <= chi.source with the same Parent group
  ARepOps.CheckGroup(chi.source);
  if not 
    ForAll(R.source.theGenerators, g -> g in chi.source) 
  then
    Error("<chi>.source must contain <R>.source");
  fi;
  if not IsIdentical(Parent(R.source), Parent(chi.source)) then
    subgrp := Subgroup(chi.source, R.source.theGenerators);
    subgrp.theGenerators := R.source.theGenerators;
    if IsBound(R.source.name) then
      subgrp.name := R.source.name;
    fi;
    R := 
      ARepOps.CopyWithNewSource(
        R,
        subgrp
      );
  fi; 

  # check expensive conditions
  if not IsIrreducible(chi) then
    Error("<chi> must be irreducible");
  fi;
  chiR := CharacterARep(R);
  if not IsIrreducible(chiR) then
    Error("<R> must be irreducible");
  fi;
  for i in [1..Length(chiR.values)] do
    if not 
      chiR.source.conjugacyClasses[i].representative ^ chi = 
      chiR.values[i]
    then
      Error("<chi> must extend CharacterARep(<R>)");
    fi;
  od;

  return
    rec(
      isARep        := true,
      operations    := ARepOps,
      char          := R.char,
      degree        := R.degree,
      source        := chi.source,
      type          := "extension",
      rep           := R,

      character     := chi,
      isIrreducible := true
    );
end;


#F Comparison of AReps
#F -------------------
#F

#F ARepOps.\=( <arep1>, <arep2> )
#F   tests if <arep1> and <arep2> represent equal representations
#F   in the mathematical sense which means that the images of 
#F   <arep1> and <arep2> are pointwise equal. The groups represented 
#F   have to be equal in the GAP-sense, i.e.
#F     <arep1>.source = <arep2>.source
#F   has to be true.
#F

ARepOps.\= := function ( R1, R2 )

  if not IsARep(R1) and not IsARep(R2) then
    Error("sorry, don't know how to compare <R1> = <R2>");
  fi;
  if IsARep(R1) and not IsARep(R2) then
    return false;
  fi;
  if not IsARep(R1) and IsARep(R2) then
    return false;
  fi;
  
  # check if groups are equal
  if not R1.source = R2.source then
    return false;
  fi;

  # cheap checks
  if R1.degree <> R2.degree then
    return false;
  fi;
  if R1.char <> R2.char then
    return false;
  fi;
  if 
    IsBound(R1.character) and 
    IsBound(R2.character) and
    R1.character.values <> R2.character.values 
  then
    return false;
  fi;

  # the expensive check
  # for aggroups check only minimal generating set
  if IsAgGroup(R1.source) then
    return
      ForAll(
        MinimalGeneratingSet(R1.source),
        g -> 
          ImageARep(g, R1) = ImageARep(g, R2)
      );
  fi;
            
  # else check all generators
  return 
    ForAll(
      R1.source.theGenerators,
      g -> 
        ImageARep(g, R1) = ImageARep(g, R2)
    );
end;


#F Pretty Printing of AReps
#F ------------------------
#F

#F ARepOps.Print( <arep> [, <indent> ] )
#F   prints the <arep> beginning at the current cursor
#F   position which is assumed to be at the beginning
#F   of a line, indented at <indent> spaces.
#F

ARepOps.PrintGroup := function ( G )
  if IsBound(G.name) then
    Print(G.name);
  else
    Print("GroupWithGenerators( ", G.theGenerators, " )");
  fi;
end;

ARepOps.PrintMat := function ( mat, indent, indentStep )
  Print(mat); # ...noch nicht
end;

ARepOps.PrintTheImages := function ( imgs, indent, indentStep )
  local i, newline, printImage;

  newline := function ( )
    local i;

    Print("\n");
    for i in [1..indent] do
      Print(" ");
    od;
  end;

  printImage := function ( im, indent, indentStep )
    if IsPerm(im) then
      Print(im);
    elif IsMon(im) then
      MonOps.PrintNC(im, indent, indentStep);
    elif IsMat(im) then
      ARepOps.PrintMat(im, indent, indentStep);
    else
      Print(im);
    fi;
  end;

  if Length(imgs) = 0 then
    Print("[ ]");
    return;
  fi;
  Print("[");
  for i in [2..indentStep] do
    Print(" ");
  od;
  indent := indent + indentStep;
  for i in [1..Length(imgs)] do
    printImage(imgs[i], indent, indentStep);
    if i < Length(imgs) then
      Print(",");
      newline();
    fi;
  od;
  indent := indent - indentStep;
  newline();
  Print("]");
end;

ARepOps.PrintNC := function ( R, indent, indentStep )
  local newline, i;

  newline := function ( )
    local i;

    Print("\n");
    for i in [1..indent] do
      Print(" ");
    od;
  end;

  if IsBound(R.name) then
    Print(R.name);
    return;
  fi;

  if R.type = "perm" then

    # use TrivialARep/ TrivialPermARep
    if ForAll(R.theImages, x -> x = ()) then
      Print("TrivialPermARep( ");
      ARepOps.PrintGroup(R.source);
      if R.char > 0 then
        Print(", ",R.degree);
        Print(", GF(", R.char, ")");
      elif R.degree > 1 then
        Print(", ",R.degree);
      fi;
      Print(" )");
      return;
    fi;

    # use NaturalARep
    if 
      IsPermGroup(R.source) and
      R.theImages = R.source.theGenerators
    then
      Print("NaturalARep( ");
      ARepOps.PrintGroup(R.source);
      Print(", ", R.degree);
      if R.char > 0 then
        Print(", GF(", R.char, ")");
      fi;
      Print(" )");
      return;
    fi;

    # use ARepByImages
    Print("ARepByImages("); 
    indent := indent + indentStep; 
      newline();
      ARepOps.PrintGroup(R.source);
      Print(",");
      newline();
      ARepOps.PrintTheImages(R.theImages, indent, indentStep);
      Print(",");
      newline();
      Print(R.degree, ", # degree");
      newline();
      if R.char > 0 then
        Print("GF(", R.char, "),");
        newline();
      fi;
      if IsBound(R.kernel) and IsTrivial(R.kernel) then
        Print("\"faithful\"");
      else
        Print("\"hom\"");
      fi;
    indent := indent - indentStep;
    newline();
    Print(")");
    return;

  elif R.type = "mon" then

    # use TrivialMonARep
    if 
      ForAll(
        R.theImages, 
        x -> x.perm = () and 
        ForAll(x.diag, t -> t = t^0)
      )
    then
      Print("TrivialMonARep( ");
      ARepOps.PrintGroup(R.source);
      if R.char > 0 then
        Print(", ", R.degree);
        Print(", GF(", R.char, ")");
      elif R.degree > 1 then
        Print(", ", R.degree);
      fi;
      Print(" )");
      return;
    fi;

    # use NaturalARep
    if 
      IsMon(R.source.identity) and 
      R.theImages = R.source.theGenerators
    then
      Print("NaturalARep( ");
      ARepOps.PrintGroup(R.source);
      Print(" )");
      return;
    fi;

    # use ARepByImages
    Print("ARepByImages("); 
    indent := indent + indentStep; 
      newline();
      ARepOps.PrintGroup(R.source);
      Print(",");
      newline();
      ARepOps.PrintTheImages(R.theImages, indent, indentStep);
      Print(",");
      newline();
      if IsBound(R.kernel) and IsTrivial(R.kernel) then
        Print("\"faithful\"");
      else
        Print("\"hom\"");
      fi;
    indent := indent - indentStep;
    newline();
    Print(")");
    return;

  elif R.type = "mat" then

    # use TrivialMatARep
    if 
      ForAll(R.theImages, x -> x = x^0)
    then
      Print("TrivialMatARep( ");
      ARepOps.PrintGroup(R.source);
      if R.char > 0 then
        Print(", ", R.degree);
        Print(", GF(", R.char, ")");
      elif R.degree > 1 then
        Print(", ", R.degree);
      fi;
      Print(" )");
      return;
    fi;

    # use NaturalARep
    if 
      IsMon(R.source.identity) and 
      R.theImages = R.source.theGenerators
    then
      Print("NaturalARep( ");
      ARepOps.PrintGroup(R.source);
      Print(" )");
      return;
    fi;

    # use ARepByImages
    Print("ARepByImages("); 
    indent := indent + indentStep; 
      newline();
      ARepOps.PrintGroup(R.source);
      Print(",");
      newline();
      ARepOps.PrintTheImages(R.theImages, indent, indentStep);
      Print(",");
      newline();
      if IsBound(R.kernel) and IsTrivial(R.kernel) then
        Print("\"faithful\"");
      else
        Print("\"hom\"");
      fi;
    indent := indent - indentStep;
    newline();
    Print(")");
    return;

  elif R.type = "conjugate" then

    # use ConjugateARep
    Print("ConjugateARep(");
    indent := indent + indentStep;
      newline();
      ARepOps.PrintNC(R.rep, indent, indentStep);
      Print(",");
      newline();
      AMatOps.Print(R.conjugation, indent, indentStep, 0);
    indent := indent - indentStep;
    newline();
    Print(")");
    return;

  elif R.type = "directSum" then

    # use DirectSumARep
    Print("DirectSumARep(");
    indent := indent + indentStep;
      newline();
      for i in [1..Length(R.summands)] do
        ARepOps.PrintNC(R.summands[i], indent, indentStep);
        if i < Length(R.summands) then
          Print(",");
          newline();
        fi;
      od;
    indent := indent - indentStep;
    newline();
    Print(")");
    return;

  elif R.type = "innerTensorProduct" then

    # use InnerTensorProductARep
    Print("InnerTensorProductARep(");
    indent := indent + indentStep;
      newline();
      for i in [1..Length(R.factors)] do
        ARepOps.PrintNC(R.factors[i], indent, indentStep);
        if i < Length(R.factors) then
          Print(",");
          newline();
        fi;
      od;
    indent := indent - indentStep;
    newline();
    Print(")");
    return;

  elif R.type = "outerTensorProduct" then

    # use OuterTensorProductARep
    Print("OuterTensorProductARep(");
    indent := indent + indentStep;
      newline();
      if not R.isOuter then
        ARepOps.PrintGroup(R.source);
        Print(",");
        newline();
      fi;
      for i in [1..Length(R.factors)] do
        ARepOps.PrintNC(R.factors[i], indent, indentStep);
        if i < Length(R.factors) then
          Print(",");
          newline();
        fi;
      od;
    indent := indent - indentStep;
    newline();
    Print(")");
    return;

  elif R.type = "galoisConjugate" then

    # use GaloisConjugateARep
    Print("GaloisConjugateARep(");
    indent := indent + indentStep;
      newline();
      ARepOps.PrintNC(R.rep, indent, indentStep);
      Print(",");
      newline();
      Print(R.galoisAut);
    indent := indent - indentStep;
    newline();
    Print(")");
    return;

  elif R.type = "restriction" then

    # use RestrictionARep
    Print("RestrictionARep(");
    indent := indent + indentStep;
      newline();
      ARepOps.PrintNC(R.rep, indent, indentStep);
      Print(",");
      newline();
      ARepOps.PrintGroup(R.source);
    indent := indent - indentStep;
    newline();
    Print(")");
    return;

  elif R.type = "induction" then

    # use RegularARep
    if IsTrivial(R.rep.source) and IsSet(R.transversal) then
      Print("RegularARep( ");
      ARepOps.PrintGroup(R.source);
      if R.char <> 0 then
        Print(", ");
        Print("GF(", R.char, ")");
      fi;
      Print(" )");
      return;
    fi;

    # use InductionARep
    Print("InductionARep(");
    indent := indent + indentStep;
      newline();
      ARepOps.PrintNC(R.rep, indent, indentStep);
      Print(",");
      newline();
      ARepOps.PrintGroup(R.source);
      Print(",");
      newline();
      Print(R.transversal);
    indent := indent - indentStep;
    newline();
    Print(")");
    return;

  elif R.type = "extension" then

    # use ExtensionARep
    Print("ExtensionARep(");
    indent := indent + indentStep;
      newline();
      ARepOps.PrintNC(R.rep, indent, indentStep);
      Print(",");
      newline();
      Print(R.character);
    indent := indent - indentStep;
    newline();
    Print(")");
    return;

  else
    Error("unrecognized <R>.type of ARep");
  fi;
end;

ARepOps.Print := function ( arg )
  local R, indent;

  if Length(arg) = 1 then
    R      := arg[1];
    indent := 0;
  elif Length(arg) = 2 then
    R      := arg[1];
    indent := arg[2];
  else
    Error("usage: ARepOps.Print( <arep> [, <indent> ] )");
  fi;
  if not IsARep(R) then
    Error("<R> must be an ARep");
  fi;
  if not ( IsInt(indent) and indent >= 0 ) then
    Error("<indent> must be non-negative integer");
  fi;
   
  ARepOps.PrintNC(R, indent, 2);
end; 


#F Fundamental Operations with AReps
#F ---------------------------------
#F
#F ImageARep( <grpelt/list-of-grpelts>, <arep> )
#F <grpelt> ^ <arep> ; shorthand 
#F   evaluates the ARep at the group element; 
#F   the result is an AMat.
#F

ARepOps.ImageNC := function ( g, R )
  local i, j, P, D, T;

  if R.type = "perm" then

    # check for a generator first
    i := Position(R.source.theGenerators, g);
    if i <> false then
      return AMatPerm(R.theImages[i], R.degree, R.char);
    fi;

    # check if R is trivial
    if ARepOps.IsOneRep(R) then
      return IdentityPermAMat(R.degree, R.char);
    fi;
    
    return AMatPerm(g ^ ARepOps.Hom(R), R.degree, R.char);

  elif R.type = "mon" then
  
    # check for a generator first
    i := Position(R.source.theGenerators, g);
    if i <> false then
      return AMatMon(R.theImages[i]);
    fi;

    # check if R is trivial
    if ARepOps.IsOneRep(R) then
      return IdentityMonAMat(R.degree, R.char);
    fi;

    return AMatMon(g ^ ARepOps.Hom(R)); 

  elif R.type = "mat" then
  
    # check for a generator first
    i := Position(R.source.theGenerators, g);
    if i <> false then
      return AMatMat(R.theImages[i]);
    fi;

    # check if R is trivial
    if ARepOps.IsOneRep(R) then
      return IdentityMatAMat(R.degree, R.char);
    fi;

    return AMatMat(g ^ ARepOps.Hom(R)); 

  elif R.type = "conjugate" then

    return ARepOps.ImageNC(g, R.rep) ^ R.conjugation;

  elif R.type = "directSum" then
  
    return 
      DirectSumAMat(
        List(
          R.summands,
          Ri -> ARepOps.ImageNC(g, Ri)
        )
      );

  elif R.type = "innerTensorProduct" then

    return 
      TensorProductAMat(
        List(
          R.factors,
          Ri -> ARepOps.ImageNC(g, Ri)
        )
      );

  elif R.type = "outerTensorProduct" then

    return
      TensorProductAMat(
        List(
          [1..Length(R.factors)],
          i -> 
            ARepOps.ImageNC(
              g ^ ARepOps.OuterTensorProductProjection(R, i), 
              R.factors[i]
            )
        )
      );

  elif R.type = "galoisConjugate" then

    return
      GaloisConjugateAMat(
        ARepOps.ImageNC(g, R.rep),
        R.galoisAut
      );

  elif R.type = "restriction" then

    return ARepOps.ImageNC(g, R.rep);

  elif R.type = "induction" then

    # construct induced AMat
    T := R.transversal;
    P := [ ];
    D := [ ];
    for i in [1..Length(T)] do

      # find j such that T[i] g T[j]^-1 in R.rep.source
      j := 1;
      while not T[i]*g/T[j] in R.rep.source do
        j := j + 1;
      od;
      Add(P, j);

      # evaluate R.rep at T[i] g T[j]^-1
      Add(D, ImageARep(T[i]*g/T[j], R.rep));
    od;
    P := PermList(P);
    D := Permuted(D, P);

    return
      TensorProductAMat(
        AMatPerm(P, Length(R.transversal), R.char),
        IdentityPermAMat(R.rep.degree, R.char)
      ) *
      DirectSumAMat(D);

  elif R.type = "extension" then

    if not IsBound(R.hom) then
      R.hom := ARepOps.MinkwitzExtension(R);
    fi;
    return AMatMat(g ^ R.hom);

  else
    Error("unrecognized <R>.type in ARep");
  fi;
end;

ImageARep := function ( g, R )
  if not IsARep(R) then
    Error("<R> must be an ARep");
  fi;
  if g in R.source then
    return ARepOps.ImageNC(g, R);
  elif IsList(g) and ForAll(g, gi -> gi in R.source) then
    return List(g, gi -> ARepOps.ImageNC(gi, R));
  else
    Error("<g> is neither in <R>.source nor a list of these");
  fi;
end;


#F IsEquivalentARep( <arep1>, <arep2> )
#F   determines, whether <arep1> and <arep2> are equivalent 
#F   representations of the same source and char.
#F   This function works only, if the Maschke condition holds
#F   for both, <arep1> and <arep2>.
#F

IsEquivalentARep := function ( R1, R2 )
  if not( IsARep(R1) and IsARep(R2) ) then
    Error("usage: IsEquivalentARep( <arep1>, <arep2> )");
  fi;
  if not( 
    IsIdentical(R1.source, R2.source) and 
    R1.char = R2.char 
  ) then
    Error("<R1>, <R2> must be areps of same source and char");
  fi;

  # check Maschke
  if 
    R1.char <> 0 and
    ( Size(R1.source) mod R1.char = 0 or
      Size(R2.source) mod R2.char = 0 )
  then
    Error("Maschke condition must hold for <R1> and <R2>");
  fi;

  if R1.char = 0 then
    return 
      R1.degree = R2.degree and 
      CharacterARep(R1) = CharacterARep(R2);
  fi;

  # for characteristic <> 0 don't construct a character
  return 
    List(
      List(
        ConjugacyClasses(R1.source), 
        cc -> cc.representative
      ),
      x -> TraceAMat(x ^ R1)
    ) = 
    List(
      List(
        ConjugacyClasses(R1.source), 
        cc -> cc.representative
      ),
      x -> TraceAMat(x ^ R2)
    );
 end;


#F CharacterARep( <arep> )
#F   the Character of the representation. The result
#F   is stored in <arep>.character. 
#F   <arep>.char must be zero.
#F

# ARepOps.OuterTensorProductCharacter( <arep> )
#   computes the character for a the outer tensor product of
#   the factors; this function is used because the case is
#   too complex to put into CharacterARep immediately.

ARepOps.OuterTensorProductCharacter := function ( R )
  local chi, embed, vs, cs, cs1, c, ci, i;

  # There are three ways to do the tensor product here
  #   1. Get the character of the factors; tensor the value vectors
  #      and form the conjugacy classes; store the conjugacy classes
  #      in R.source.
  #   2. As 1. but sort the value vector according to the existing 
  #      order of R.source.conjugacyClasses.
  #   3. Use R.projections to evaluate the characters of the factors
  #      at the existing R.source.conjugacyClasses.
  # We use 1. if possible, 2. if necessary and 3. not at all.

  # compute all characters of the factors
  chi := List(R.factors, CharacterARep);

  # form chi[1] tensor .. tensor chi[n]
  embed := ARepOps.OuterTensorProductEmbedding(R, 1);
  vs    := chi[1].values;
  cs    := 
    List(
      chi[1].source.conjugacyClasses, 
      c -> c.representative ^ embed
    );
  for i in [2..Length(R.factors)] do

    # {vs, cs} := {vs, cs} tensor chi[i]
    embed := ARepOps.OuterTensorProductEmbedding(R, i);
    vs    := Concatenation(List(vs, v -> v * chi[i].values));
    cs1   := [ ];
    for c in cs do
      for ci in chi[i].source.conjugacyClasses do
        Add(cs1, c * (ci.representative ^ embed));
      od;
    od;
    cs := cs1; 
  od;

  # sort {cs, vs} to match R.source.conjugacyClasses
  if not IsBound(R.source.conjugacyClasses) then
    R.source.conjugacyClasses := 
      List(cs, c -> ConjugacyClass(R.source, c));
    R.character := 
      Character(R.source, vs);
  else
    R.character :=
      Character(
        R.source,
        Permuted(
          vs, 
          ARepOps.PermConjugacyClasses(
            R.source.conjugacyClasses,
            cs
          )
        )
      );
  fi;
  return R.character;
end;

CharacterARep := function ( R )
  local chi;

  if not IsARep(R) then
    Error("<R> must be an ARep");
  fi;
  if IsBound(R.character) then
    return R.character;
  fi;

  # check char = 0
  if R.char <> 0 then
    Error("<R>.char must be zero");
  fi;  

  # make sure Length(R.theImages) > 0  
  if IsTrivial(R.source) then
    R.character :=
      Character(
        R.source, 
        [ R.degree * ARepOps.PrimeFieldOne(R.char) ]
      );
    return R.character;
  fi;

  # dispatch on the types
  if R.type = "perm" then

    # make sure R.hom is present
    R.hom       := ARepOps.Hom(R);
    R.character :=
      Character(
        R.source,
        List(
          ConjugacyClasses(R.source),
          c -> 
            R.degree - 
            NrMovedPointsPerm(c.representative ^ R.hom)
        )
      );

  elif R.type = "mon" then

    # make sure R.hom is present
    R.hom       := ARepOps.Hom(R);
    R.character :=
      Character(
        R.source,
        List(
          ConjugacyClasses(R.source),
          c -> TraceMon(c.representative ^ R.hom)
        )
      );

  elif R.type = "mat" then

    # make sure R.hom is present
    R.hom       := ARepOps.Hom(R);
    R.character :=
      Character(
        R.source,
        List(
          ConjugacyClasses(R.source),
          c -> Trace(c.representative ^ R.hom)
        )
      );

  elif R.type = "conjugate" then

    # the character is invariant under conjugation
    R.character := CharacterARep(R.rep);

  elif R.type = "directSum" then

    # R.summands[i].source is *identical* to R.source for all i
    R.character :=
      Sum(
        List(
          R.summands, 
          CharacterARep
        )
      );

  elif R.type = "innerTensorProduct" then

    # R.factors[i].source is *identical* to R.source for all i
    R.character :=
      Product(
        List(
          R.factors, 
          CharacterARep
        )
      );

  elif R.type = "outerTensorProduct" then

    # form the tensor product of the character value vectors
    chi         := List(R.factors, CharacterARep);
    R.character :=
      ARepOps.OuterTensorProductCharacter( R );

  elif R.type = "galoisConjugate" then

    # conjugate the character
    R.character :=
      Character(
        R.source,
        List(
          CharacterARep(R.rep).values,
          v -> AMatOps.GaloisConjugation(v, R.galoisAut)
        )
      );

  elif R.type = "restriction" then

    # evaluate the extended character at the conjugacy classes
    chi         := CharacterARep(R.rep);
    R.character :=
      Character(
        R.source,
        List(
          ConjugacyClasses(R.source),
          c -> c.representative ^ chi
        )
      );
 
  elif R.type = "induction" then
 
    # evaluate the induced character directly
    chi         := CharacterARep(R.rep);
    R.character :=
      Character(
        R.source,
        List(
          ConjugacyClasses(R.source),
          function ( c )
            local v, t, ct;

            v := 0 * ARepOps.PrimeFieldOne(R.char);
            for t in R.transversal do
              ct := t * c.representative / t;
              if ct in R.rep.source then
            v := v + ct ^ chi;
              fi;
            od;
            return v;
          end
        )
      );
    
  elif R.type = "extension" then

    # the character is mandatory for "extension"-AReps
    if not IsBound(R.character) then
      Error("panic! no <R>.character in extension");
    fi;

  else
    Error("unrecognized <R>.type in ARep");
  fi;

  return R.character;
end;


#F IsIrreducibleARep( <arep> )
#F   determines if <arep> is an irreducible representation.
#F   This function only works for characteristic zero, 
#F   since the character is used.
#F

IsIrreducibleARep := function ( R )
  if not IsARep(R) then
    Error("<R> must be an ARep");
  fi;
  if R.char <> 0 then
    Error("<R>.char must be zero");
  fi;
  if IsBound(R.isIrreducible) then
    return R.isIrreducible;
  fi;

  # check the character first
  if IsBound(R.character) then
    R.isIrreducible := IsIrreducible(R.character);
    return R.isIrreducible;
  fi;

  # catch permreps
  if IsPermRep(R) then
    R.isIrreducible := (R.degree = 1);
    return R.isIrreducible;
  fi;

  # catch abelian groups
  if IsAbelian(R.source) then
    R.isIrreducible := (R.degree = 1);
    return R.isIrreducible;
  fi;

  # dispatch on the types
  if R.type in ["mon", "mat"] then

    # use the character
    R.isIrreducible := IsIrreducible(CharacterARep(R));

  elif R.type = "conjugate" then

    # the character is invariant under conjugation
    R.isIrreducible := IsIrreducibleARep(R.rep);

  elif R.type = "directSum" then

    # look at nr. of summands and the summand
    R.isIrreducible := 
      Length(R.summands) = 1 and
      IsIrreducibleARep(R.summands[1]);

  elif R.type = "innerTensorProduct" then

    # irred iff there is at most one irred factor of degree > 1
    if Number(R.factors, Ri -> Ri.degree > 1) > 1 then
      return false;
    fi;
    R.isIrreducible := 
      ForAll(
        R.factors, 
        Ri -> Ri.degree = 1 or IsIrreducibleARep(Ri)
      );

  elif R.type = "outerTensorProduct" then

    # irred iff all factors are
    R.isIrreducible := 
      ForAll(R.factors, IsIrreducibleARep);

  elif R.type = "galoisConjugate" then

    R.isIrreducible := IsIrreducibleARep(R.rep);

  elif R.type = "restriction" then

    # check the character
    R.isIrreducible := IsIrreducible(CharacterARep(R));
 
  elif R.type = "induction" then

    # check the character
    R.isIrreducible := IsIrreducible(CharacterARep(R));
    
  elif R.type = "extension" then

    # check the character
    R.isIrreducible := IsIrreducible(CharacterARep(R));

  else
    Error("unrecognized <R>.type in ARep");
  fi;
  return R.isIrreducible;
end;


#F KernelARep( <arep> )
#F   the kernel of the representation.
#F

KernelARep := function ( R )
  if not( IsARep(R) ) then
    Error("usage: KernelARep( <arep> )");
  fi;
  if IsBound(R.kernel) then
    return R.kernel;
  fi;
  if IsBound(R.isFaithful) and R.isFaithful = true then
    return TrivialSubgroup(R.source);
  fi;

  # conjugation does not change the kernel
  if R.type = "conjugate" then
    return KernelARep(R.rep);
  fi;

  # induction case can be treated specially:
  # ker(L_H ind R) = core(ker(L_H) intersect H)
  if R.type = "induction" then
    R.kernel :=
      Core(
        R.source, 
        Intersection(
          AsSubgroup(Parent(R.source), KernelARep(R.rep)), 
          AsSubgroup(Parent(R.source), R.rep.source)
        )
      );
    return R.kernel;
  fi;

  # if R is a transitive monrep, consider the 
  # corresponding induction
  if IsMonRep(R) and IsTransitiveMonRep(R) and R.degree > 1 then
    R.kernel :=
      KernelARep(
        TransitiveToInductionMonRep(R)
      );
    return R.kernel;
  fi;

  # calculate kernel via the character
  R.kernel := ARepOps.KernelCharacter(CharacterARep(R));
  return R.kernel;
end;


#F IsFaithfulARep( <arep> )
#F   determines if <arep> is an injective homomorphism.
#F

IsFaithfulARep := function ( R )
  if not IsARep(R) then
    Error("<R> must be an ARep");
  fi;
  return IsTrivial(KernelARep(R));
end;


#F ARepWithCharacter( <chi> )
#F   calculates an arep with character <chi>.
#F   The group must be solvable.
#F   This function only works for GAP3r4p4
#F   after bugfix #9!
#F

ARepWithCharacter := function ( chi )
  local G, mults, reps, rep, i, j, N, chiN, repN, irrs;

  if not IsCharacter(chi) then
    Error("usage: ARepWithCharacter( <chi> )");
  fi;

  G := GroupWithGenerators(chi.source);
  if not IsSolvable(G) then
    Error("<G> must be solvable");
  fi;

  # decompose in irreducibles
  if not IsIrreducible(chi) then
    mults := List(Irr(G), c -> ScalarProduct(c, chi));
    reps  := [ ];
    for i in [1..Length(Irr(G))] do
      rep := ARepWithCharacter(Irr(G)[i]);
      for j in [1..mults[i]] do
        Add(reps, rep);
      od;
    od;
    return DirectSumARep(reps);
  fi;

  # catch onedimensional case
  if chi.values[1] = 1 then
    return ARepByCharacter(chi);
  fi;

  # recurse with normal subgroup of prime index
  N    := CompositionSeries(G)[2];
  chiN := 
    Character(
      N, 
      List(
        ConjugacyClasses(N), 
        cc -> cc.representative ^ chi
      )
    );
  
  # check, if chiN is irreducible
  if IsIrreducible(chiN) then

    repN := ARepWithCharacter(chiN);
    
    # extend repN
    return ExtensionARep(repN, chi);
  
  else

    # induce the first component
    return
      InductionARep( 
        ARepWithCharacter(
          First(Irr(N), c -> ScalarProduct(c, chiN) = 1)
        ),
        G
      );

  fi;
end;


#F GeneralFourierTransform( <group> )
#F   returns an amat A representing a general Fourier transform 
#F   for <group> (cf. Clausen/Baum: Fast Fourier Transforms, 1993).
#F   This means that A decomposes the regular representation
#F   of <group> into a direct sum of irreducibles which are
#F   ordered according to their character.
#F   The <group> must be solvable. Since the function uses
#F   the function ARepWithCharacter (see above) it works only
#F   for GAP3r4p4 after bugfix #9.
#F   Note that it is possible to obtain a fast Fourier transform
#F   for <group> by using the function DecompositionMonRep
#F   for the regular representation of <group>.
#F

GeneralFourierTransform := function( G )
  local L, irrs, M, i, row, R;

  # check argument
  if not IsGroup(G) then
    Error("<G> must be group");
  fi;
  if not IsSolvable(G) then
    Error("<G> must be solvable");
  fi;

  G := GroupWithGenerators(G);
  L := Elements(G);

  # create reps for each irreducible character
  irrs := List(Irr(G), ARepWithCharacter);
 
  # Build up the DFT matrix M following Baum-Clausen
  # Chap. 2, p. 44: Choose an ordering of the Elements
  # of G and a system of representatives rho_1, ..., rho_k
  # of equivalence classes of irreducible reps. of G.
  # Then write the matrix coefficients of the images of
  # each group element to the rows of M
  M := [ ];
  for i in [1..Size(G)] do
    row := [ ];
    for R in irrs do
       Append(row, Flat(MatAMat(L[i]^R)));
    od;
    Add(M, row);
  od;
  return AMatMat(TransposedMat(M))^-1;
end;


#F Flattening Out AReps
#F --------------------
#F
#F IsPermRep( <arep> )
#F IsMonRep(  <arep> )
#F   test if <arep> can be turned into a "perm" or "mon" ARep.
#F   The result is memorized in .isPermRep/.isMonRep. Note that
#F   the names of the operations are *not* 'IsPermARep' etc. 
#F   since <arep> can be any type but it *represents* a 
#F   permutation representation in the mathematical sense.
#F  
#F PermARepARep( <arep> )
#F MonARepARep(  <arep> )
#F MatARepARep(  <arep> )
#F   constructs an "perm"/"mon"/"mat"-ARep equal to the
#F   given <arep> or returns false if this is not possible.
#F   The result is memorized in .permARep/.monARep/.matARep.
#F

IsPermRep := function ( R )
  local images, im, g;

  if not IsARep(R) then 
    Error("usage: IsPermRep( <arep> )");
  fi;
  if IsBound(R.isPermRep) then
    return R.isPermRep;
  fi;

  # use R.monARep if present
  if IsBound(R.monARep) then
    images := [ ];
    for im in R.monARep.theImages do
      Add(images, PermMon(im));
      if images[Length(images)] = false then
        R.isPermRep := false;
        return R.isPermRep;
      fi;
    od;
    R.isPermRep := true;
    R.permARep  :=
      ARepByImages(R.source, images, R.degree, "hom");
    return R.isPermRep;
  fi;

  # dispatch on types
  if R.type = "perm" then

    R.isPermRep := true;

  elif R.type = "mon" then

    images := [ ];
    for im in R.theImages do
      Add(images, PermMon(im));
      if images[Length(images)] = false then
        R.isPermRep := false;
        return R.isPermRep;
      fi;
    od;
    R.isPermRep := true;
    R.permARep  :=
      ARepByImages(R.source, images, R.degree, "hom");

  elif R.type = "mat" then

    images := [ ];
    for im in R.theImages do
      Add(images, PermMat(im));
      if images[Length(images)] = false then
        R.isPermRep := false;
        return R.isPermRep;
      fi;
    od;
    R.isPermRep := true;
    R.permARep  :=
      ARepByImages(R.source, images, R.degree, "hom");

  elif R.type = "conjugate" then

    # catch the case, that the conjugation
    # is a permmat
    if IsPermMat(R.conjugation) then
      R.isPermRep := IsPermRep(R.rep);
    else

      images := [ ];
      for g in R.source.theGenerators do
        Add(images, PermAMat(ImageARep(g, R)));
        if images[Length(images)] = false then
          R.isPermRep := false;
          return R.isPermRep;
        fi;
      od;
      R.isPermRep := true;
      R.permARep  :=
        ARepByImages(R.source, images, R.degree, "hom");

    fi;

  elif R.type = "directSum" then

    R.isPermRep := ForAll(R.summands, IsPermRep);

  elif R.type = "innerTensorProduct" then

    if ForAll(R.factors, IsPermRep) then

      # all factors are permreps => R is permrep
      R.isPermRep := true;

    elif ForAny(R.factors, r -> not IsMonRep(r)) then

      # any factor not monomial => R not monomial
      R.isPermRep := false;

    else

      # calculate images
      images := [ ];
      for g in R.source.theGenerators do
        Add(images, PermAMat(ImageARep(g, R)));
        if images[Length(images)] = false then
          R.isPermRep := false;
          return R.isPermRep;
        fi;
      od;
      R.isPermRep := true;
      R.permARep  :=
        ARepByImages(R.source, images, R.degree, "hom");

    fi;

  elif R.type = "outerTensorProduct" then
    
    # outer tensorproduct is a permrep iff the factors are
    R.isPermRep := ForAll(R.factors, IsPermRep);

  elif R.type = "galoisConjugate" then

    R.isPermRep := IsPermRep(R.rep);

  elif R.type = "restriction" then

    if IsPermRep(R.rep) then
      R.isPermRep := true;
    else
      images := [ ];
      for g in R.source.theGenerators do
        Add(images, PermAMat(ImageARep(g, R)));
        if images[Length(images)] = false then
          R.isPermRep := false;
          return R.isPermRep;
        fi;
      od;
      R.isPermRep := true;
      R.permARep  :=
        ARepByImages(R.source, images, R.degree, "hom");
    fi;

  elif R.type = "induction" then

    R.isPermRep := IsPermRep(R.rep);

  elif R.type = "extension" then
    
    # an extension is irreducible and hence cannot 
    # be a permrep except the trivial case
    R.isPermRep := 
      R.degree = 1 and
      ForAll(CharacterARep(R).values, x -> x = x ^ 0);

  else
    Error("unrecognized <R>.type in ARep");
  fi;

  # update knowledge of R.permARep if present
  if IsBound(R.permARep) then
    if 
      IsBound(R.character) and 
      not IsBound(R.permARep.character) 
    then
      R.permARep.character := R.character;
    fi;
    if IsBound(R.isIrreducible) then
      R.permARep.isIrreducible := R.isIrreducible;
    fi;
    if IsBound(R.kernel) then
      R.permARep.kernel := R.kernel;
    fi;
  fi;

  # update isMon, too
  if R.isPermRep then
    R.isMonRep := true;
  fi;
  return R.isPermRep;    
end;


IsMonRep := function( R )
  local images, im, g;

  if not IsARep(R) then 
    Error("usage: IsMonRep( <arep> )");
  fi;
  if IsBound(R.isMonRep) then
    return R.isMonRep;
  fi;
  if IsBound(R.isPermRep) and R.isPermRep then
    R.isMonRep := true;
    return R.isMonRep;
  fi;

  # dispatch on types
  if R.type = "perm" then

    R.isMonRep := true;

  elif R.type = "mon" then

    R.isMonRep := true;

  elif R.type = "mat" then

    images := [ ];
    for im in R.theImages do
      Add(images, MonMat(im));
      if images[Length(images)] = false then
        R.isMonRep := false;
        return R.isMonRep;
      fi;
    od;
    R.isMonRep := true;
    R.monARep  := ARepByImages(R.source, images, "hom");

  elif R.type = "conjugate" then

    # catch the case, that the conjugation
    # is a monmat
    if IsMonMat(R.conjugation) then
      R.isMonRep := IsMonRep(R.rep);
    else

      images := [ ];
      for g in R.source.theGenerators do
        Add(images, MonAMat(ImageARep(g, R)));
        if images[Length(images)] = false then
          R.isMonRep := false;
          return R.isMonRep;
        fi;
      od;
      R.isMonRep := true;
      R.monARep  := ARepByImages(R.source, images, "hom");

    fi;

  elif R.type = "directSum" then

    R.isMonRep := ForAll(R.summands, IsMonRep);

  elif R.type = "innerTensorProduct" then

    # if any factor of a tensor product is non-monomial
    # then the tensor product is non-monomial
    R.isMonRep := ForAll(R.factors, IsMonRep);

  elif R.type = "outerTensorProduct" then
    
    # outer tensorproduct is a permrep iff the factors are
    R.isMonRep := ForAll(R.factors, IsMonRep);

  elif R.type = "galoisConjugate" then

    R.isMonRep := IsMonRep(R.rep);

  elif R.type = "restriction" then

    if IsMonRep(R.rep) then
      R.isMonRep := true;
    else
      images := [ ];
      for g in R.source.theGenerators do
	Add(images, MonAMat(ImageARep(g, R)));
	if images[Length(images)] = false then
	  R.isMonRep := false;
	  return R.isMonRep;
	fi;
      od;
      R.isMonRep := true;
      R.monARep  := ARepByImages(R.source, images, "hom");
    fi;

  elif R.type = "induction" then

    R.isMonRep := IsMonRep(R.rep);

  elif R.type = "extension" then

    images := [ ];
    for g in R.source.theGenerators do
      Add(images, MonAMat(ImageARep(g, R)));
      if images[Length(images)] = false then
        R.isMonRep := false;
	return R.isMonRep;
      fi;
    od;
    R.isMonRep := true;
    R.monARep  := ARepByImages(R.source, images, "hom");

  else
    Error("unrecognized <R>.type in ARep");
  fi;

  # update knowledge of R.monARep if present
  if IsBound(R.monARep) then
    if 
      IsBound(R.character) and 
      not IsBound(R.monARep.character) 
    then
      R.monARep.character := R.character;
    fi;
    if IsBound(R.isIrreducible) then
      R.monARep.isIrreducible := R.isIrreducible;
    fi;
    if IsBound(R.kernel) then
      R.monARep.kernel := R.kernel;
    fi;
  fi;
  return R.isMonRep;    
end;


PermARepARep := function ( R )
  local P, Rs, images, L, T, g, i, j, listUntilFalse;

  if not IsARep(R) then
    Error("usage: PermARepARep( <arep> )");
  fi;
  if IsBound(R.permARep) then
    return R.permARep;
  fi;
  if IsBound(R.isPermRep) and not R.isPermRep then
    return false;
  fi;

  # listUntilFalse( <list>, <func> )
  #   compute List(<list>, <func>) or return false
  #   if any component of the result is false.

  listUntilFalse := function ( list, func )
    local result, x;

    result := [ ];
    for x in list do
      Add(result, func(x));
      if result[Length(result)] = false then
        return false;
      fi;
    od;
    return result;
  end;

  if IsBound(R.monARep) then

    # try building R.permARep from R.monARep
    images := 
      listUntilFalse(
        R.monARep.theImages,
        PermMon
      );
    if images = false then
      P := false;
    else
      P := ARepByImages(R.source, images, R.degree, R.char, "hom");
    fi;

  elif R.type = "perm" then

    P := R;

  elif R.type = "mon" then

    images := 
      listUntilFalse(
        R.theImages,
        PermMon
      );
    if images = false then
      P := false;
    else
      P := ARepByImages(R.source, images, R.degree, R.char, "hom");
    fi;

  elif R.type = "mat" then

    images := 
      listUntilFalse(
        R.theImages,
        PermMat
      );
    if images = false then
      P := false;
    else
      P := ARepByImages(R.source, images, R.degree, R.char, "hom");
    fi;

  elif R.type = "conjugate" then

    images := 
      listUntilFalse(
        R.source.theGenerators,
        g -> PermAMat(ImageARep(g, R))
      );
    if images = false then
      P := false;
    else
      P := ARepByImages(R.source, images, R.degree, R.char, "hom");

      # check if R is an induction of a onedimensional rep
      # conjugated by a monomial matrix
      if 
        R.rep.type = "induction" and
        R.rep.rep.degree = 1 and
        IsMonMat(R.conjugation)
      then
        P.induction := 
          ConjugateARep(
            InductionARep(
              MonARepARep(R.rep.rep),
              R.source,
              R.rep.transversal
            ),
            MonAMatAMat(R.conjugation)
          );
      fi;
    fi;

  elif R.type = "directSum" then

    Rs := listUntilFalse(R.summands, PermARepARep);
    if Rs = false then
      P := false;
    else
      P :=
        ARepByImages(
          R.source,
          List(
            [1..Length(R.source.theGenerators)],
            g -> 
              DirectSumPerm(
                List(Rs, Ri -> Ri.degree),
                List(Rs, Ri -> Ri.theImages[g])
              )
          ),
          R.degree,
          "hom"
        );
    fi;

  elif R.type = "innerTensorProduct" then

    # note that the inner tensor product of two representations
    # which are no permreps can be a permrep
    images := 
      listUntilFalse(
        R.source.theGenerators,
        g -> PermAMat(ImageARep(g, R))
      );
    if images = false then
      P := false;
    else
      P := ARepByImages(R.source, images, R.degree, R.char, "hom");
    fi; 

  elif R.type = "outerTensorProduct" then

    # an outer tensor product is a permrep iff the factors are
    Rs := listUntilFalse(R.factors, PermARepARep);
    if Rs = false then
      P := false;
    else
      images :=
        List(
          R.source.theGenerators,
          g -> 
            TensorProductPerm(
              List(Rs, p -> p.degree),
              List(
                [1..Length(R.factors)], 
                i -> 
                  PermAMat(
                    ImageARep(
                      g ^ ARepOps.OuterTensorProductProjection(R, i), 
                      Rs[i]
                    )
                  )
              )
            )
        );
      P := ARepByImages(R.source, images, R.degree, R.char, "hom");
    fi;

  elif R.type = "galoisConjugate" then
    
    P := PermARepARep(R.rep);

  elif R.type = "restriction" then

    images := 
      listUntilFalse(
        R.source.theGenerators,
        g -> PermAMat(ImageARep(g, R))
      );
    if images = false then
      P := false;
    else
      P := ARepByImages(R.source, images, R.degree, R.char, "hom");
    fi;

  elif R.type = "induction" then

    # R.rep must be a permrep
    Rs := PermARepARep(R.rep);
    if Rs = false then
      P := false;
    else 

      # construct induction
      if IsTrivial(Rs.source) then

        # regular case
        images :=
          List(
            R.source.theGenerators,
            g -> 
            PermList(
              List(R.transversal, t -> Position(R.transversal, t * g))
            )
          );
	P := ARepByImages(R.source, images, R.degree, R.char, "hom");

      else

	# construct by formula
	images := [ ];
	T      := R.transversal;
	for g in R.source.theGenerators do
	  L := [ ];
	  for i in [1..Length(R.transversal)] do

	    # look which block is <> 0
	    j := 1;
	    while not T[i]*g/T[j] in Rs.source do
	      j := j + 1;
	    od;

	    # the image under R.rep gives one block
	    Append( L,
	      OnTuples(
		[1..Rs.degree], 
		PermAMat(ImageARep(T[i]*g/T[j], Rs))
	      ) +
	      (j - 1)*Rs.degree
	    );
	  od;

	  Add(images, PermList(L));
	od;

	P := ARepByImages(R.source, images, R.degree, R.char, "hom");
      fi;

      # check if R is an induction of a onedimensional rep
      if Rs.degree = 1 then
      
        # store induction decomposition
        P.induction :=
          ConjugateARep(
            InductionARep(
              MonARepARep(Rs),
              P.source,
              R.transversal
            ),
            IdentityMonAMat(R.degree, R.char)
          );
      fi;
    fi;
    
  elif R.type = "extension" then

    # since R is irreducible it can only be a permutation
    # representation if it is trivial
    if ARepOps.IsTrivialOneRep(R) then
      P := TrivialPermARep(R.source, 1, R.char);
    else 
      P := false;
    fi;

  else
    Error("unrecognized <R>.type in ARep");
  fi;

  # prepare the result
  if P = false then
    R.isPermRep := false;
    return false;
  fi;

  # copy optional fields
  if IsBound(R.character) then
    P.character := R.character;
  fi;
  if IsBound(R.isIrreducible) then
    P.isIrreducible := R.isIrreducible;
  fi;
  if IsBound(R.kernel) then
    P.kernel := R.kernel;
  fi;
  if IsBound(R.induction) then
    P.induction := R.induction;
  fi;
  P.isPermRep := true;
  P.isMonRep  := true;

  R.permARep := P;
  return R.permARep;
end;


MonARepARep := function ( R )
  local P, Rs, images, L, D, T, g, img, i, j, k, listUntilFalse;

  if not IsARep(R) then
    Error("usage: MonARepARep( <arep> )");
  fi;
  if IsBound(R.monARep) then
    return R.monARep;
  fi;
  if IsBound(R.isMonRep) and not R.isMonRep then
    return false;
  fi;

  # listUntilFalse( <list>, <func> )
  #   compute List(<list>, <func>) or return false
  #   if any component of the result is false.

  listUntilFalse := function ( list, func )
    local result, x;

    result := [ ];
    for x in list do
      Add(result, func(x));
      if result[Length(result)] = false then
        return false;
      fi;
    od;
    return result;
  end;

  if IsBound(R.permARep) then

    # build R.monARep from R.permARep
    P := 
      ARepByImages(
        R.source, 
        List(
          R.permARep.theImages, 
          x -> Mon(x, R.degree, R.char)
        ),
        "hom"
      );

  elif R.type = "perm" then

    P := 
      ARepByImages(
        R.source, 
        List(
          R.theImages, 
          x -> Mon(x, R.degree, R.char)
        ),
        "hom"
      );

  elif R.type = "mon" then

    P := R;

  elif R.type = "mat" then

    images := listUntilFalse(R.theImages, MonMat);
    if images = false then
      P := false;
    else
      P := ARepByImages(R.source, images, "hom");
    fi;

  elif R.type = "conjugate" then

    images := 
      listUntilFalse(
        R.source.theGenerators,
        g -> MonAMat(ImageARep(g, R))
      );
    if images = false then
      P := false;
    else
      P := ARepByImages(R.source, images, "hom");

      # check if R is an induction of a onedimensional rep
      # conjugated by a monomial matrix
      if 
        R.rep.type = "induction" and
        R.rep.rep.degree = 1 and
        IsMonMat(R.conjugation)
      then
        P.induction := 
          ConjugateARep(
            InductionARep(
              MonARepARep(R.rep.rep),
              R.source,
              R.rep.transversal
            ),
            MonAMatAMat(R.conjugation)
          );
      fi;
    fi;

  elif R.type = "directSum" then

    Rs := listUntilFalse(R.summands, MonARepARep);
    if Rs = false then
      P := false;
    else
      P :=
        ARepByImages(
          R.source,
          List(
            [1..Length(R.source.theGenerators)],
            g -> 
              DirectSumMon(
                List(Rs, Ri -> Ri.theImages[g])
              )
          ),
          "hom"
        );
    fi;

  elif R.type = "innerTensorProduct" then

    # inner tensor product is monomial iff the factors are
    Rs := listUntilFalse(R.factors, MonARepARep);
    if Rs = false then
      P := false;
    else
      P :=
        ARepByImages(
          R.source,
          List(
            [1..Length(R.source.theGenerators)],
            g -> 
              TensorProductMon(
                List(Rs, Ri -> Ri.theImages[g])
              )
          ),
          "hom"
        );
    fi;

  elif R.type = "outerTensorProduct" then

    # outer tensor product is monomial iff the factors are
    Rs := listUntilFalse(R.factors, MonARepARep);
    if Rs = false then
      P := false;
    else
      P :=
        ARepByImages(
          R.source,
          List(
            R.source.theGenerators,
            g -> 
              TensorProductMon(
                List(
                  [1..Length(Rs)],
                  i -> 
                    MonAMat(
                      ImageARep(
                        g ^ ARepOps.OuterTensorProductProjection(R, i),
                        R.factors[i]
                      )
                    )
                )
              )
          ),
          "hom"
        );
    fi;

  elif R.type = "galoisConjugate" then
    
    Rs := MonARepARep(R);
    if Rs = false then
      P := false;
    else
      P := 
        ARepByImages(
          R.source, 
          List(
            Rs.theImages,
            x -> GaloisMon(x, R.galoisAut)
          ),
          "hom"
        );
    fi;

  elif R.type = "restriction" then

    images := 
      listUntilFalse(
        R.source.theGenerators,
        g -> MonAMat(ImageARep(g, R))
      );
    if images = false then
      P := false;
    else
      P := ARepByImages(R.source, images, "hom");
    fi;

  elif R.type = "induction" then

    # R.rep must be a monrep
    Rs := MonARepARep(R.rep);
    if Rs = false then
      P := false;
    else 

      # construct the induction
      if IsTrivial(Rs.source) then

        # regular case
        P := MonARepARep(PermARepARep(R));

      else
    
        # construct by formula
	images := [ ];
	T      := R.transversal;
	for g in R.source.theGenerators do
	  L := [ ];
	  D := [ ];
	  for i in [1..Length(R.transversal)] do

	    # look which block is <> 0
	    j := 1;
	    while not T[i]*g/T[j] in Rs.source do
	      j := j + 1;
	    od;

	    # the image under R.rep gives one block
	    img := MonAMat(ImageARep(T[i]*g/T[j], Rs));
	    Append( L,
	      OnTuples(
		[1..Rs.degree], 
		img.perm
	      ) +
	      (j - 1)*Rs.degree
	    );
	    for k in [1..Rs.degree] do
	      D[k + (j - 1)*Rs.degree] := img.diag[k];
	    od;
	  od;

	  Add(images, Mon(PermList(L), D));
	od;

	P := ARepByImages(R.source, images, "hom");

	# check if R is an induction of a onedimensional rep
	if R.rep.degree = 1 then
	  P.induction :=
	    ConjugateARep(
	      InductionARep(
		MonARepARep(R.rep),
		P.source,
		R.transversal
	      ),
	      IdentityMonAMat(R.degree, R.char)
	    );
	fi;
      fi;
    fi;
    
  elif R.type = "extension" then

    # R is monomial implies R.rep is 
    Rs := MonARepARep(R.rep);
    if Rs = false then
      P := false;
    else
      images := 
        listUntilFalse(
          R.source.theGenerators,
          g -> MonAMat(ImageARep(g, R))
        );
      if images = false then
        P := false;
      else
        P := ARepByImages(R.source, images, "hom");
      fi;
    fi;

  else
    Error("unrecognized <R>.type in ARep");
  fi;

  # prepare the result
  if P = false then
    R.isMonRep := false;
    return false;
  fi;

  # copy optional fields
  if IsBound(R.character) then
    P.character := R.character;
  fi;
  if IsBound(R.isIrreducible) then
    P.isIrreducible := R.isIrreducible;
  fi;
  if IsBound(R.kernel) then
    P.kernel := R.kernel;
  fi;
  if IsBound(R.induction) then
    P.induction := R.induction;
  fi;
  P.isMonRep  := true;

  R.monARep := P;
  return R.monARep;
end;


MatARepARep := function ( R )
  local P;

  if not IsARep(R) then
    Error("usage: MatARepARep( <arep> )");
  fi;
  if IsBound(R.matARep) then
    return R.matARep;
  fi;

  if IsBound(R.permARep) then

    # build R.matARep from R.permARep
    P := 
      ARepByImages(
        R.source, 
        List(
          R.permARep.theImages, 
          x -> MatPerm(x, R.degree, R.char)
        ),
        "hom"
      );

  elif IsBound(R.monARep) then

    # build R.matARep from R.monARep
    P := 
      ARepByImages(
        R.source, 
        List(R.monARep.theImages, MatMon),
        "hom"
      );

  elif R.type = "perm" then

    P := 
      ARepByImages(
        R.source, 
        List(
          R.theImages, 
          x -> MatPerm(x, R.degree, R.char)
        ),
        "hom"
      );

  elif R.type = "mon" then

    P := 
      ARepByImages(
        R.source, 
        List(R.theImages, MatMon),
        "hom"
      );

  elif R.type = "mat" then

    P := R;

  elif R.type in 
    [ "conjugate", "directSum", "innerTensorProduct",
      "outerTensorProduct", "galoisConjugate", "restriction", 
      "induction", "extension"]
  then

    P := 
      ARepByImages(
        R.source, 
        List(ImageARep(R.source.theGenerators, R), MatAMat),
        "hom"
      );

  else
    Error("unrecognized <R>.type in ARep");
  fi;

  # prepare the result
  # copy optional fields
  if IsBound(R.character) then
    P.character := R.character;
  fi;
  if IsBound(R.isIrreducible) then
    P.isIrreducible := R.isIrreducible;
  fi;
  if IsBound(R.kernel) then
    P.kernel := R.kernel;
  fi;
  R.matARep := P;
  return R.matARep;
end;


# Random Monreps
# --------------

#F RandomMonRep( [ <groupsize/group> ] )
#F   creates a "random" monrep of a group G of size <groupsize>
#F   or chooses randomly, if not supplied. The size must be 
#F   in [1..1000]\[512, 768]. A random subgroup H <= G and
#F   a random rep L of H of degree 1 is chosen. 
#F   The induction of L to G is returned.
#F   THis function requires the small group library
#F

if not IsBound(SmallGroup) then
  SmallGroup := "dummy";
  NumberSmallGroups := "dummy";
fi;

if SmallGroup <> "dummy" then

RandomMonRep := function ( arg  )
  local sizes, s, G, H, L;  

  # dispatch
  if Length(arg) = 0 then
    s := 1; # dummy value
  elif Length(arg) = 1 and IsInt(arg[1]) and arg[1] > 0 then
    s := arg[1];
  elif Length(arg) = 1 and IsGroup(arg[1]) then
    s := arg[1];
  else
    Error("usage: RandomMonRep( [ <groupsize/group> ] )");
  fi;

  # possible sizes
  sizes := 
    Filtered(
      [1..1000], 
      i -> i <> 512 and i <> 768
    );
  if not ( s in sizes or IsGroup(s) ) then
    Error("<s> must be a valid group size <= 1000, <> 512, 768");
  fi;
  if Length(arg) = 0 then
    s := Random(sizes);
  fi;

  # choose group
  if IsGroup(s) then
    G := s;
  else
    G := 
      GroupWithGenerators(
        SmallGroup(s, Random([1..NumberSmallGroups(s)]))
      );
  fi;

  # subgroup
  H := 
    GroupWithGenerators(
      Random(ConjugacyClassesSubgroups(G)).representative
    );
  H.name := "H"; # avoids warning, is unbound below

  # representation of degree 1 of H
  L := 
    ARepByCharacter(
      Random(Filtered(Irr(H), chi -> Degree(chi) = 1))
    );
  Unbind(H.name);
  
  return InductionARep(L, G);
end;

else

RandomMonRep := function ( arg )
  Error("small groups library not installed");
end;

fi; # IsBound(SmallGroup)




# Dispatcher for Operators
# ------------------------

ARepOps.\^ := function ( L, R )
  if IsGroupElement(L) and IsARep(R) then
    return ImageARep(L, R);
  elif IsARep(L) and IsAMat(R) then
    return ConjugateARep(L, R);
  else
    Error("unrecognized operation <L>^<R>");
  fi;
end;
