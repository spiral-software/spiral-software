# -*- Mode: shell-script -*- 

GlobalPackage(gap.perm);
#############################################################################
##
#A  permutat.g                  GAP library                  Martin Schoenert
#A                                                                & Udo Polis
##
##
#Y  Copyright (C) 2018-2021, Carnegie Mellon University
#Y  All rights reserved.  See LICENSE for details.
#Y  
#Y  This work is based on GAP version 3, with some files from version 4.  GAP is
#Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
##
##  This  file  contains  those  functions  that  deal   with   permutations.
##
##


#############################################################################
##
#F  InfoOperation?(...) . . .  information function for the operation package
##
if not IsBound(InfoOperation1)  then InfoOperation1 := Ignore;  fi;
if not IsBound(InfoOperation2)  then InfoOperation2 := Ignore;  fi;


#############################################################################
##
#V  Permutations  . . . . . . . . . . . . . . . .  domain of all permutations
#V  PermutationsOps . . . . . . . . . . .  operations record for permutations
##
PermutationsOps := Copy( GroupElementsOps );

PermutationsOps.\in            := function ( g, Permutations )
    return IsPerm( g );
end;

Permutations                    := rec();
Permutations.isDomain           := true;

Permutations.name               := "Permutations";

Permutations.isFinite           := false;
Permutations.size               := "infinity";

Permutations.operations         := PermutationsOps;


#############################################################################
##
#F  PermutationsOps.Cycle(<g>,<d>)  . .  orbit of a point under a permutation
##
PermutationsOps.Cycle := function ( g, d, opr )
    local   cyc;

    # standard operation on integers
    if   opr = OnPoints  and IsInt( d )  then
        cyc := CyclePermInt( g, d );

    # other operation
    else
        cyc := GroupElementsOps.Cycle( g, d, opr );

    fi;

    # return the cycle <cyc>
    return cyc;
end;


#############################################################################
##
#F  PermutationsOps.CycleLength(<g>,<d>)  . . . .  length of orbit of a point
#F                                                        under a permutation
##
PermutationsOps.CycleLength := function ( g, d, opr )
    local   len;

    # standard operation on integers
    if  opr = OnPoints  and IsInt( d )  then
        len := CycleLengthPermInt( g, d );

    # other operation
    else
        len := GroupElementsOps.CycleLength( g, d, opr );

    fi;

    # return the length <len>
    return len;
end;


#############################################################################
##
#F  PermutationsOps.Cycles(<g>,<D>)   cycles of a domain under an permutation
##
#N  26-Apr-91 martin 'CyclesPerm' should be internal
##
PermutationsOps.Cycles := function ( g, D, opr )
    local   cycs,       # cycles, result
            cyc,        # cycle
            mark,       # boolean list to mark elements already processed
            i,j,k;      # loop variables

    # standard operation
    if   opr = OnPoints  and ForAll( D, IsInt )  then

     if g = () then
        cycs := List(D, x-> [x]);
     else 
        mark := BlistList([1..Maximum(LargestMovedPointPerm(g),Maximum(D))], 
               []);

        # compute the cycles <cycs>
        cycs := []; 
        k:=0;
        for i in D do
            if not mark[i] then 
               k := k+1;
               cyc := CyclePermInt( g, i );
               cycs[k]:= cyc;
               for j in cyc do
                  mark[j] := true;
               od;
            fi; 
        od;
     fi;

    # other operation
    else
        cycs := GroupElementsOps.Cycles( g, D, opr );

    fi;

    # return the cycles <cycs>
    return cycs;
end;


#############################################################################
##
#F  PermutationsOps.CycleLengths(<g>,<D>) . . . . . . . lengths of the cycles
#F                                                           of a permutation
##
#N  26-Apr-91 martin 'CycleLengthsPermInt' should be internal
##
PermutationsOps.CycleLengths := function ( g, D, opr )
    local   lens,       # cyclelengths, result
            cyc,        # cycle
            mark,       # boolean list to mark elements already processed
            i,j,k;      # loop variables

    # standard operation
    if   opr = OnPoints  and ForAll( D, IsInt )  then

      if g = () then
        lens := List( [1..Length(D)], x -> 1 );
      else
        mark := BlistList([1..Maximum(LargestMovedPointPerm(g),Maximum(D))], 
               []);

        # compute the cyclelengths <lens>
        lens := []; 
        k:=0;
        for i in D do
            if not mark[i] then 
               k := k+1;
               cyc := CyclePermInt( g, i );
               lens[k]:= Length( cyc );
               for j in cyc do
                  mark[j] := true;
               od;
            fi; 
        od;
      fi;

    # other operation
    else
        lens := GroupElementsOps.CycleLengths( g, D, opr );

    fi;

    # return the cyclelengths <lens>
    return lens;
end;


#############################################################################
##
#F  PermutationsOps.EvenPerm(<g>,<D>,opr) returns `true' for even permutation
##
PermutationsOps.EvenPerm := function ( g, D, opr )
    local   cycs;       # cyclelenghts of g
 
    cycs := PermutationsOps.CycleLengths( g, D, opr );
    return IsInt( ( Size(D)-Length(cycs) )/2 );
end;


#############################################################################
##
#F  CycleStructurePerm( <perm> )  . . . . . . . . .  length of cycles of perm
##
CycleStructurePerm := function ( perm )
    local   cys,    # collected cycle lengths, result
            degree, # degree of perm
            mark,   # boolean list to mark elements already processed
            i,j,    # loop variables 
            len,    # length of a cycle 
            cyc;    # a cycle of perm

    if perm = () then
        cys := [];
    else
        degree := LargestMovedPointPerm(perm);
        mark := BlistList([1..degree], []);
        cys := [];
        for i in [1..degree] do
            if not mark[i] then 
               cyc := CyclePermInt( perm, i );
               len := Length(cyc) - 1;
               if 0 < len  then
                  if IsBound(cys[len])  then
                     cys[len] := cys[len]+1;
                  else
                     cys[len] := 1;
                  fi;
               fi;
               for j in cyc do
                  mark[j] := true;
               od;
            fi;
        od;
    fi;
    return cys;
end;


#############################################################################
##
#F  PermutationsOps.Permutation(<g>,<D>)  . . .  permutation of a permutation
#F                                                                on a domain
##
PermutationsOps.Permutation := function ( g, D, opr )
    local   prm,        # permutation, result
            pos,        # position of point in D
            inc,        # increment if <D> is a range
            i;          # loop variable

    # standard operation
    if   opr = OnPoints  then

        # standard operation on a range
        if IsRange( D )  then
            InfoOperation1("#I  PermutationPerm(<g>,<range>)\c");
            if Length( D ) = 1  then
               inc := 1;
            else
               inc := D[2] - D[1];
            fi;

            # compute the permutation <prm>
            prm := [];
            for i  in [1..Length(D)]  do
                prm[i] := (D[i]^g - D[1]) / inc + 1;
            od;

            prm := PermList( prm );
            InfoOperation1("\r#I  PermutationPerm(<g>,<range>) returns\n");

        # standard operation on another list of integers
        elif D = []  or ForAll( D, IsInt )  then
            InfoOperation1("#I  PermutationPerm(<g>,<D>)\c");

            # first find the position of every point of <D>
            pos := [];
            for i  in [1..Length(D)]  do
                pos[D[i]] := i;
            od;

            # then compute the permutation <prm>
            prm := [];
            for i  in [1..Length(D)]  do
                prm[i] := pos[ D[i]^g ];
            od;

            prm := PermList( prm );
            InfoOperation1("\r#I  PermutationPerm(<g>,<D>) returns\n");

        # standard operation on a list of nonintegers
        else
            prm := GroupElementsOps.Permutation( g, D, opr );
        fi;

    # other operation
    else
        prm := GroupElementsOps.Permutation( g, D, opr );

    fi;

    # return the permutation <prm>
    return prm;
end;


#############################################################################
##
#F  SmallestMovedPointPerm( <g> ) . . . smallest point moved by a permutation
##
SmallestMovedPointPerm := function ( g )
    local   p;
    if g = ()  then
        Error("<g> must not be the identity");
    else
        p := 1;
        while p^g = p  do
            p := p + 1;
        od;
    fi;
    return p;
end;


#############################################################################
##
#F  ListPerm( <g> ) . . . . . . . . . . . . . list of images of a permutation
##
ListPerm := function ( g )

    if g = ()  then
      return [];
    else
      return OnTuples( [ 1 .. LargestMovedPointPerm(g) ], g );
    fi;
    end;

#############################################################################
##
#F  MappingPermListList(<src>,<dst>)  permutation mapping one list to another
##
MappingPermListList := function( src, dst )

    if not IsList(src) or not IsList(dst) or Length(src) <> Length(dst)  then
       Error("usage: MappingPermListList( <lst1>, <lst2> )");
    fi;

    if Length(src) = 0  then
        return ();
    fi;

    src := Concatenation( src, Difference( [1..Maximum(src)], src ) );
    dst := Concatenation( dst, Difference( [1..Maximum(dst)], dst ) );

    return PermList( src ) mod PermList( dst );
end;


##############################################################################
##
#F  RestrictedPerm(<g>,<D>) . restriction of a permutation to an invariant set
##
RestrictedPerm := function( g, D )
    local   res, d, e, max;

    # check the arguments
    if not IsPerm( g )  then
        Error("<g> must be a permutation");
    elif not IsList( D )  then
        Error("<D> must be a list");
    fi;

    # special case for the identity
    if g = ()  then return ();  fi;

    # compute the largest point that we must consider
    max := 1;
    for d  in D  do
        e := d ^ g;
        if d <> e  and max < d  then
            max := d;
        fi;
    od;

    # compute the restricted permutation <res>
    res := [ 1 .. max ];
    for d  in D  do
        e := d ^ g;
        if d <= max  then
            res[d] := e;
        fi;
    od;

    # return the restricted permutation <res>
    return PermList( res );
end;


#############################################################################
##
#F  PermutationsOps.CyclicGroup(<P>,<n>)  . . . . .  cyclic permutation group
#F  PermutationsOps.AbelianGroup(<P>,<ords>)  . . . abelian permutation group
#F  PermutationsOps.ElementaryAbelianGroup(<P>,<n>) . . .  elementary abelian
#F                                                          permutation group
#F  PermutationsOps.DihedralGroup(<P>,<n>)  . . .  dihedral permutation group
#F  PermutationsOps.PolyhedralGroup(<P>,<p>,<q>) polyhedral permutation group
#F  PermutationsOps.AlternatingGroup(<P>,<n>) . alternating permutation group
#F  PermutationsOps.SymmetricGroup(<P>,<n>) . . . symmetric permutation group
#F  PermutationsOps.GeneralLinearGroup(<P>,<n>,<q>)
#F  PermutationsOps.SpecialLinearGroup(<P>,<n>,<q>)
#F  PermutationsOps.SymplecticGroup(<P>,<n>,<q>)
#F  PermutationsOps.GeneralUnitaryGroup(<P>,<n>,<q>)
#F  PermutationsOps.SpecialUnitaryGroup(<P>,<n>,<q>)
##
PermutationsOps.CyclicGroup := function ( P, n )
    return PermGroupLib( "CyclicGroup", n );
end;

PermutationsOps.AbelianGroup := function ( P, ords )
    return PermGroupLib( "AbelianGroup", ords );
end;

PermutationsOps.ElementaryAbelianGroup := function ( P, n )
    return PermGroupLib( "ElementaryAbelianGroup", n );
end;

PermutationsOps.DihedralGroup := function ( P, n )
    return PermGroupLib( "DihedralGroup", n );
end;

PermutationsOps.PolyhedralGroup := function ( P, p, q )
    return PermGroupLib( "PolyhedralGroup", p, q );
end;

PermutationsOps.AlternatingGroup := function ( P, n )
    return PermGroupLib( "AlternatingGroup", n );
end;

PermutationsOps.SymmetricGroup := function ( P, n )
    return PermGroupLib( "SymmetricGroup", n );
end;

PermutationsOps.GeneralLinearGroup := function ( P, n, q )
    return PermGroupLib( "GeneralLinearGroup", n, q );
end;

PermutationsOps.SpecialLinearGroup := function ( P, n, q )
    return PermGroupLib( "SpecialLinearGroup", n, q );
end;

PermutationsOps.SymplecticGroup := function ( P, n, q )
    return PermGroupLib( "SymplecticGroup", n, q );
end;

PermutationsOps.GeneralUnitaryGroup := function ( P, n, q )
    return PermGroupLib( "GeneralUnitaryGroup", n, q );
end;

PermutationsOps.SpecialUnitaryGroup := function ( P, n, q )
    return PermGroupLib( "SpecialUnitaryGroup", n, q );
end;

#############################################################################
##
##  other functions are in 'permgrp.g'
##
##  AUTO( ReadLib("permgrp"),
##      PermutationsOps.Group )
##
ReadLib("permgrp");



