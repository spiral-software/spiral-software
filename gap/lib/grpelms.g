# -*- Mode: shell-script -*- 

#############################################################################
##
#A  grpelms.g                   GAP library                      Frank Celler
##
##
#Y  Copyright (C) 2018-2021, Carnegie Mellon University
#Y  All rights reserved.  See LICENSE for details.
#Y  
#Y  This work is based on GAP version 3, with some files from version 4.  GAP is
#Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
##
##  This file contains all  functions  and  operations  for  groups elements.
##
##


#############################################################################
##
#F  IsGroupElement( <g> ) . . . . . . .  test if an object is a group element
##
IsGroupElement := function ( g )
    return    IsPerm( g )
           or IsWord( g )
           or IsAgWord( g )
           or IsMat( g )
           or IsRec( g ) and IsBound(g.isGroupElement) and g.isGroupElement;
end;


#############################################################################
##
#V  GroupElements . . . . . . . . . . . . . . .  domain of all group elements
#V  GroupElementsOps  .  operation record of the domain of all group elements
##
##  'GroupElements' is the domain of all group  elements, i.e., permutations,
##  abstract  words, words in  solvable  groups,  matrices, and records  that
##  implement group elements.  Note that 'GroupElements' is not a group.
##
##  'GroupElementsOps' is   the operations  record for   the  'GroupElements'
##  domain.   Other  domains that  consist  of group  elements,  for  example
##  'Permutations' and 'Matrices' inherit from this operations record.
##
##  The most important function defined for 'GroupElements' is 'Group', i.e.,
##  'GroupElements' knowns how to make a group from a list of elements.
##
GroupElements                   := rec( );

GroupElements.isDomain          := true;

GroupElements.name              := "GroupElements";

GroupElements.isFinite          := false;
GroupElements.size              := "infinity";

GroupElements.operations        := Copy( DomainOps );
GroupElementsOps                := GroupElements.operations;

GroupElementsOps.\in := function ( g, GroupElements )
    return    IsPerm( g )
           or IsWord( g )
           or IsAgWord( g )
           or IsMat( g )
           or IsRec( g ) and IsBound(g.isGroupElement) and g.isGroupElement;
end;

GroupElementsOps.Order := function ( G, g )
    local   ord,        # order of <g>, result
            pow,        # power of <g>
            id;         # identity of <G>

    id  := g^0;
    pow := g;
    ord := 1;
    while pow <> id  do
        pow := pow * g;
        ord := ord + 1;
    od;
    return ord;
end;


#############################################################################
##
#F  GroupElementOps . . . . . . . operation record for generic group elements
##
GroupElementOps         := rec( );

GroupElementOps.\= := function ( a, b )
    if      IsRec( a ) and IsBound( a.isGroupElement ) and a.isGroupElement
        and IsRec( b ) and IsBound( b.isGroupElement ) and b.isGroupElement
        and a.domain = b.domain
    then
        return a.element = b.element;
    else
        return false;
    fi;
end;

GroupElementOps.\< := function ( a, b )
    if      IsRec( a ) and IsBound( a.isGroupElement ) and a.isGroupElement
        and IsRec( b ) and IsBound( b.isGroupElement ) and b.isGroupElement
    then
        if a.domain = b.domain  then
            return a.element < b.element;
        else
            return a.domain < b.domain;
        fi;
    elif    IsRec( a ) and IsBound( a.isGroupElement ) and a.isGroupElement
    then
        return not IsGroupElement( b );
    elif    IsRec( b ) and IsBound( b.isGroupElement ) and b.isGroupElement
    then
        return IsGroupElement( a );
    else
        Error("panic, either <a> or <b> must be a group element");
    fi;
end;

GroupElementOps.\* := function ( a, b )
    Error("no default method to multiply generic group elements");
end;

GroupElementOps.\/ := function ( a, b )
    return a * b^-1;
end;

GroupElementOps.\mod := function ( a, b )
    return a^-1 * b;
end;

GroupElementOps.\^ := function ( a, b )
    if IsGroup( a )  then
        return a.operations.\^( a, b );
    else
        Error("no default method to multiply generic group elements");
    fi;
end;

GroupElementOps.Comm := function ( a, b )
    return a^-1 * b^-1 * a * b;
end;

GroupElementOps.Print := function ( a )
    Print( "GroupElement( ", a.element, " )" );
end;


#############################################################################
##
#F  Order( <g> )  . . . . . . . . . . . . . . . . . order of an group element
##
Order := function ( G, g )
    return G.operations.Order( G, g );
end;

GroupOps.Order := function ( G, g )
    local   ord,        # order of <g>, result
            pow,        # power of <g>
            id;         # identity of <G>

    id := G.identity;
    pow := g;
    ord := 1;
    while pow <> id  do
        pow := pow * g;
        ord := ord + 1;
    od;
    return ord;
end;


#############################################################################
##
#F  LeftNormedComm( <a>, ... )	. . . . . . . . . . . left normed commutators
##
##  The following function computes left normed commutators.
##
LeftNormedComm := function( arg )
    local   c, i, l;

    if Length(arg) = 1 and IsList(arg[1])  then
        l := arg[1];
    else
	l := arg;
    fi;

    c := Comm( l[1], l[2] );
    for i  in [3..Length(l)]  do
        c := Comm( c, l[i] );
    od;
    return c;

end;


#############################################################################
##
#F  RightNormedComm( <a>, ... )	. . . . . . . . . .  right normed commutators
##
##  The following function computes right normed commutators.
##
RightNormedComm := function( arg )
    local   c, i, l;

    if Length(arg) = 1 and IsList(arg[1])  then
        l := arg[1];
    else
	l := arg;
    fi;

    c := Comm( l[Length(l)-1], l[Length(l)] );
    for i  in [ Length(l)-2, Length(l)-3 .. 1 ]  do
        c := Comm( l[i], c );
    od;
    return c;

end;


#############################################################################
##
#F  GroupElementsOps.CyclicGroup(<D>,<n>) . . . . . . . . . . .  cyclic group
#F  GroupElementsOps.AbelianGroup(<D>,<ords>) . . . . . . . . . abelian group
#F  GroupElementsOps.ElementaryAbelianGroup(<D>,<n>) elementary abelian group
#F  GroupElementsOps.DihedralGroup(<D>,<n>) . . . . . . . . .  dihedral group
#F  GroupElementsOps.PolyhedralGroup(<D>,<p>,<q>) . . . . .  polyhedral group
#F  GroupElementsOps.AlternatingGroup(<D>,<n>)  . . . . . . alternating group
#F  GroupElementsOps.SymmetricGroup(<D>,<n>)  . . . . . . . . symmetric group
##
GroupElementsOps.CyclicGroup := function ( D, n )
    Error( "sorry, cannot compute cyclic group for this domain" );
end;

GroupElementsOps.AbelianGroup := function ( D, ords )
    return DirectProduct( List( ords, x -> CyclicGroup( D, x ) ) );
end;

GroupElementsOps.ElementaryAbelianGroup := function ( D, n )
    local   f,  C;

    f := Factors( n );
    C := CyclicGroup( D, f[ 1 ] );
    return DirectProduct( List( f, x -> C ) );

end;

GroupElementsOps.DihedralGroup := function ( D, n )
    Error("sorry, cannot compute dihedral group for this domain");
end;

GroupElementsOps.PolyhedralGroup := function ( D, n )
    Error("sorry, cannot compute polyhedral group for this domain");
end;

GroupElementsOps.AlternatingGroup := function ( D, n )
    Error("sorry, cannot compute alternating group for this domain");
end;

GroupElementsOps.SymmetricGroup := function ( D, n )
    Error("sorry, cannot compute symmetric group for this domain");
end;
