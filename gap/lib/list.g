# -*- Mode: shell-script -*- 

#############################################################################
##
#A  list.g                      GAP library                  Martin Schoenert
#A                                                           &  Werner Nickel
##
##
#Y  Copyright (C) 2018-2021, Carnegie Mellon University
#Y  All rights reserved.  See LICENSE for details.
#Y  
#Y  This work is based on GAP version 3, with some files from version 4.  GAP is
#Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
##
##  This file contains the library functions that  deal  mainly  with  lists.
##  It does not contain functions that deal with sets, vectors  or  matrices.
##
##
GlobalPackage(gap.list);

#NSFields := ns -> RecFields(RecNamespace(ns));

#############################################################################
##
#F  NewList( length ) . . . . . . . . . . . . . . . . empty list of length
##
##  Returns empty list of requested length.
## 

if not IsBound(NewList) then 
# latest GAP kernel already has this function build in
NewList := function ( length )
    local result;
    result := [];
    if length>0 then 
        result[length] := 0;
    fi;
    return result;
end;

fi;

if not IsBound(DocLoc) then 
# latest GAP kernel already has this function build in
DocLoc := ( obj ) -> ["rebuild GAP to have DocLoc function", 0];
fi;

#############################################################################
##
#F  List( <obj>, [<func>] ) . . . . . . . . . . . . . . . . convert to a list
##
List := function ( arg )
    local  lst,  i,  fun, j, R, nargs, pair;

    if Length(arg) = 1  then
        if IsString(arg[1])  then
            lst := [];
            for i  in [1..LengthString(arg[1])]  do
                lst[i] := SubString( arg[1], i, i );
            od;
        elif IsList(arg[1])  then
            lst := arg[1];
        elif IsPerm(arg[1])  then
            lst := [];
            for i  in [1..LargestMovedPointPerm(arg[1])]  do
                lst[i] := i ^ arg[1];
            od;
        elif IsWord(arg[1])  then
            lst := [];
            for i  in [1..LengthWord(arg[1])]  do
                lst[i] := Subword( arg[1], i, i );
            od;
        else
            Error("can't convert <arg[1]> into a list");
        fi;

    elif Length(arg) = 2  and IsList(arg[1])  and (IsFunc(arg[2]) or IsRec(arg[2]))  then
        lst := NewList(Length( arg[1] )); fun := arg[2];
        for i  in [ 1 .. Length( arg[1] ) ] do
            lst[i]:= fun( arg[1][i] );
        od;
    elif Length(arg) = 2  and IsRec(arg[1]) and (IsFunc(arg[2]) or IsRec(arg[2])) then

	lst := [];  R := arg[1];  fun := arg[2];
	nargs := NumArgs(fun);

	if nargs = 1 then
	    j := 1;
	    for i in RecFields(R) do
	        lst[j] := fun( R.(i) );
		j := j+1;
	    od;
	    return lst;

	elif nargs = 2 then
	    j := 1;
	    #for i in [1..NumRecFields(R)] do
	    for i in RecFields(R) do
	        pair := fun( RecName(i), R.(i) );
		Append(lst, When(IsList(pair) and not IsDelay(pair), pair, [pair]));
		j := j+2;
	    od;
	    return lst;
	else
	    Error("<arg[2]> must be a function that takes 1 or 2 arguments");
	fi;

    else
        Error("usage: List( <obj> ) or List( <list>, <func> )");

    fi;

    return lst;
end;

#############################################################################
##
#F  RecList( <list> ) . . . . . . . . . . . . . .  convert a list to a record
##
RecList := function ( list )
    local res, fld, i;
    if not IsList(list) then Error("<list> must be a list"); fi;
    if Length(list) mod 2 <> 0 then Error("<list> must have even number of elements"); fi;
    res := rec();
    i := 1;
    while i < Length(list) do
        fld := list[i];
	if not BagType(fld) = T_RECNAM then fld := RecName(String(fld)); fi;
	res.(fld) := list[i+1];
	i := i+2;
    od;
    return res;
end;

#############################################################################
##
#F  TabList( <list> ) . . . . . . . . . . . . . .  convert a list to a record
##
##  Example: TabList("a", 1, "a", 2, "b", 3);
##   result: tab(a:=2, b:=3);
## 
##  See also: TabListAccumulate
##
TabList := function ( list )
    local res, fld, i;
    if not IsList(list) then Error("<list> must be a list"); fi;
    if Length(list) mod 2 <> 0 then Error("<list> must have even number of elements"); fi;
    res := tab();
    i := 1;
    while i < Length(list) do
        fld := list[i];
	if not IsString(fld) then fld := (String(fld)); fi;
	res.(fld) := list[i+1];
	i := i+2;
    od;
    return res;
end;

#############################################################################
##
#F  TabListAccumulate( <list> ) . . . . . as TabList, but accumulates entries 
## 
##  Example: TabListAccumulate("a", 1, "a", 2, "b", 3);
##   result: tab(a:=[1,2], b:=[3]);
##
##  See also: TabList
##
TabListAccumulate := function ( list )
    local res, fld, i;
    if not IsList(list) then Error("<list> must be a list"); fi;
    if Length(list) mod 2 <> 0 then Error("<list> must have even number of elements"); fi;
    res := tab();
    i := 1;
    while i < Length(list) do
        fld := list[i];
	if not IsString(fld) then fld := (String(fld)); fi;
	if not IsBound(res.(fld)) then res.(fld) := [list[i+1]];
	else Add(res.(fld), list[i+1]);
	fi;
	i := i+2;
    od;
    return res;
end;

#############################################################################
##
#F  Map( <list>, <func> ) . . . . . .  apply <func> to each element of a list 
##
##  Similar to List(<list>, <func>) but does not perform the check, and does
##  not support Permutations, Words, and calls with one argument
##
Map := function ( list, func )
    local i;
    if IsRec(list) then
	list := ShallowCopy(list);
	for i in RecFields(list) do
	    list.(i) := func( list.(i) );
	od;
	return list;
    elif BagType(list) = T_NAMESPACE then
	list := ShallowCopy(list);
	if NumArgs(func)=1 then
	    for i in NSFields(list) do
	        list.(i) := func( list.(i) );
	    od;
	else 
	    for i in NSFields(list) do
	        list.(i) := func(i, list.(i));
	    od;
	fi;
	return list;
    else 
	list := ShallowCopy(list);
	for i in [ 1 .. Length( list ) ] do
            list[i] := func( list[i] );
	od;
	return list;
    fi;
end;
 
Map2 := function ( list, func )
    local i, el;
    list := ShallowCopy(list);
    for i in [ 1 .. Length( list ) ] do
        el := list[i];
        list[i] := func( el[1], el[2] );
    od;
    return list;
end;

MapN := function ( list, func )
    local i, el;
    list := ShallowCopy(list);
    for i in [ 1 .. Length( list ) ] do
        el := list[i];
        list[i] := ApplyFunc(func, el);
    od;
    return list;
end;

ConcatList := function ( list, func )
    local i, el, res;
    res := [];
    for el in list do
        Append(res, func(el));
    od;
    return res;
end;


 

#############################################################################
##
#F  Apply( <list>, <func> ) . . . . . . . .  apply a function to list entries
##
##  'Apply' will  apply <func>  to  every  member  of <list> and replaces  an
##  entry by the corresponding return value.  Warning:  The previous contents
##  of <list> will be lost.
##
Apply := function ( list, func )
    local i;

    if not IsList( list ) or not IsFunc( func ) then
        Error( "usage: Apply( <list>, <func> )" );
    fi;

    for i in [1..Length( list )] do
        list[i] := func( list[i] );
    od;
end;


#############################################################################
##
#F  Concatenation( <list>, <list> ) . . . . . . . . . concatentation of lists
##
Concatenation := function ( arg )
    local  cat, lst;

    if Length(arg) = 1  and IsList(arg[1]) then
        cat := [];
        for lst  in arg[1]  do
            Append( cat, lst );
        od;

    else
        cat := [];
        for lst  in arg  do
            Append( cat, lst );
        od;

    fi;

    return cat;
end;

#############################################################################
##
#F  Flat( <list> )  . . . . . . . list of elements of a nested list structure
##
Flat := function ( lst )
    local   flt,        # list <lst> flattened, result
            elm;        # one element of <lst>

    if IsString(lst) and (Length(lst)>0 or BagType(lst)=T_STRING) then
        return lst;
    fi;
    # make the flattened list
    flt := [];
    for elm  in lst  do
        if not IsList(elm) or (IsString(elm) and (Length(elm)>0 or BagType(elm)=T_STRING)) then
            Add( flt, elm );
        else
            Append( flt, Flat(elm) );
        fi;
    od;

    # and return it
    return flt;
end;




#############################################################################
##
#F  Reversed( <list> )  . . . . . . . . . . .  reverse the elements in a list
##
##        modified at Jean Michel's suggestion

Reversed := function( list )
    local  rev,  len,  i;

    len := Length( list );
    if TYPE( list ) = "range"  then
        rev := [ list[ len ], list[ len - 1 ] .. list[ 1 ] ];
    else
        rev := list{[Length(list),Length(list)-1..1]};
    fi;

    return rev;
end;


ContainsElement := function( list, element)
	local i;

	for i in list do
		if i = element then
			return true;
		fi;
	od;

	return false;
end;


#############################################################################
##
#F  Sublist( <lst>, <indices> ) . . . . . . . . . .  extract a part of a list
#F  Sublist( <lst>, <from>, [<to>] ) . . . . . . . . extract a part of a list
##
Sublist := function(arg)
    local lst, from, to;
    if Length(arg) < 2 or Length(arg) > 3
	then Error("Usage: Sublist(<lst>, <from>, [<to>])\n     Sublist(<lst>, <indices>)");  
    fi;
    lst := arg[1];
    from := arg[2];
    to := When(Length(arg)=3, arg[3], Length(lst));

    if IsList(from) then return lst{from};
    else return lst{[from..to]};
    fi;
end;

# See Sublist
SubList := Sublist;

#############################################################################
##
#F  Filtered( <list>, <func> )  . . . . extract elements that have a property
##
Filtered := function ( list, func )
    local  flt,  elm;

    flt := [];
    for elm  in list  do
        if func( elm )  then
            Add( flt, elm );
        fi;
    od;

    return flt;
end;


#############################################################################
##
#F  SplitBy( <list>, <func> )  . . . . . . divide list according to predicate
##
##  SplitBy returns two lists, first containing elements that satisfy predi-
##  ate function <func>, and the other elements that do not. 
##  
##  <func> must take a single argument and return a boolean.
##
SplitBy := function ( list, func )
    local  T, F,  elm;

    T := []; F := [];
    for elm  in list  do
        if func( elm )  then  Add( T, elm );
	else Add( F, elm );
        fi;
    od;

    return [T, F];
end;


#############################################################################
##
#F  Number( <list> [, <func>] ) . . . . . count elements that have a property
##
Number := function ( arg )
    local  nr,  elm;

    if Length(arg) = 1  then
        nr := 0;
        for elm  in arg[1]  do
            nr := nr + 1;
        od;

    elif Length(arg) = 2  then
        nr := 0;
        for elm  in arg[1]  do
            if arg[2]( elm )  then
                nr := nr + 1;
            fi;
        od;

    else
        Error("usage: Number( <list> ) or Number( <list>, <func> )");
    fi;

    return nr;
end;


#############################################################################
##
#F  Compacted( <list> ) . . . . . . . . . . . . . .  remove holes from a list
##
Compacted := function ( list )
    local    res,       # compacted of <list>, result
             elm;       # element of <list>
    res := [];
    for elm  in list  do
        Add( res, elm );
    od;
    return res;
end;


#############################################################################
##
#F  Collected( <list> ) . . . . . 
##
Collected := function ( list )
    local   col,        # collected of <list>, result
            elm,        # element of <list>
            nr,         # number of elements of <list> equal to <elm>
            i;          # loop variable

    col := [];
    for elm  in Set( list )  do
        nr := 0;
        for i  in list  do
            if i = elm  then
                nr := nr + 1;
            fi;
        od;
        Add( col, [ elm, nr ] );
    od;
    return col;
end;


#############################################################################
##
#F  Equivalenceclasses( <list>, <function> )  . calculate equivalence classes
##
##
##  returns
##
##      rec(
##          classes := <list>,
##          indices := <list>
##      )
##
Equivalenceclasses := function( list, isequal )
    local ecl, idx, len, new, i, j;

    if not IsList( list ) or not IsFunc( isequal ) then
        Error( "usage: Equivalenceclasses( <list>, <function> )" );
    fi;

    len := 0;
    ecl := [];
    idx := [];
    for i in [1..Length( list )] do
        new := true;
        j   := 1;
        while new and j <= len do
            if isequal( list[i], ecl[j][1] ) then
                Add( ecl[j], list[i] );
                Add( idx[j], i );
                new := false;
            fi;
            j := j + 1;
        od;
        if new then
            len := len + 1;
            ecl[len] := [ list[i] ];
            idx[len] := [ i ];
        fi;
    od;
    return rec( classes := ecl, indices := idx );
end;


#############################################################################
##
#F  ForAll( <list>, <func> )  . .  test a property for all elements of a list
##
ForAll := function ( list, func )
    local  l;

    for l  in list  do
        if not func( l )  then
            return false;
        fi;
    od;

    return true;
end;


#############################################################################
##
#F  ForAny( <list>, <func> )  . . . test a property for any element of a list
##
ForAny := function ( list, func )
    local  l;

    for l  in list  do
        if func( l )  then
            return true;
        fi;
    od;

    return false;
end;


#############################################################################
##
#F  First( <list>, <func> ) . .  find first element in a list with a property
##
First := function ( list, func )
    local  l;

    for l  in list  do
        if func( l )  then
            return l;
        fi;
    od;

    Error("at least one element of <list> must fulfill <func>");
end;

#############################################################################
##
#F  FirstDef( <list>, <func>, <default> ) . . . . First() with default value
##
##  Same as first but returns <default> value if <func> returns false on all
##  <list> elements.
##

FirstDef := function ( list, func, def )
    local  l;

    for l  in list  do
        if func( l )  then
            return l;
        fi;
    od;

    return def;
end;


#############################################################################
##
#F  PositionProperty( <list>, <func> ) position of an element with a property
##
PositionProperty := function ( list, func )
    local   i;
    # this function works correctly on lists with holes
    for i  in [ 1 .. Length( list ) ]  do
        if IsBound(list[i]) and func( list[ i ] )  then
            return i;
        fi;
    od;
    return false;

end;


#############################################################################
##
#F  PositionBound( <list> ) . . . . . . . . . . position of first bound entry
##
PositionBound := function ( list )
    local   i;

    # look for the first bound element
    for i  in [1..Length(list)]  do
        if IsBound( list[i] )  then
            return i;
        fi;
    od;

    # no bound element found
    return false;
end;


#############################################################################
##
#F  Cartesian( <list>, <list>.. ) . . . . . . . .  cartesian product of lists
##
Cartesian2 := function ( list, n, tup, i )
    local  tups,  l;
    if i = n+1  then
        tup := ShallowCopy(tup);
        tups := [ tup ];
    else
        tups := [];
        for l  in list[i]  do
            tup[i] := l;
            Append( tups, Cartesian2( list, n, tup, i+1 ) );
        od;
    fi;
    return tups;
end;

Cartesian := function ( arg )
    if Length(arg) = 1  then
        return Cartesian2( arg[1], Length(arg[1]), [], 1 );
    else
        return Cartesian2( arg, Length(arg), [], 1 );
    fi;
end;


#############################################################################
##
#F  Sort( <list> )  . . . . . . . . . . . . . . . . . . . . . . . sort a list
##
##  Sort() uses Shell's diminishing increment sort, which extends bubblesort.
##  The bubble sort works by  running  through  the  list  again  and  again,
##  each time exchanging pairs of adjacent elements which are out  of  order.
##  Thus large elements "bubble" to the top, hence the name  of  the  method.
##  However elements need many moves to come close to their  final  position.
##  In shellsort the first passes do not compare element j with its  neighbor
##  but with the element j+h, where h is larger than one.  Thus elements that
##  aren't at their final position make large moves towards the  destination.
##  This increment h is diminished, until during the last  pass  it  is  one.
##  A good sequence of incremements is given by Knuth:  (3^k-1)/2,... 13,4,1.
##  For this sequence shellsort uses on average  approximatly  N^1.25  moves.
##
##  Shellsort is the method of choice to  sort  lists  for  various  reasons:
##  Shellsort is quite easy to get right, much easier than,  say,  quicksort.
##  It runs as fast as quicksort for lists with  less  than  ~5000  elements.
##  It handles both  almost  sorted  and  reverse  sorted  lists  very  good.
##  It works well  in  the  presence  of  duplicate  elements  in  the  list.
##  Says Sedgewick: "In short, if you have a sorting problem,  use the  above
##  program, then determine whether the extra effort required to  replace  it
##  with a sophisticated method will be worthwile."
##
##  Donald Knuth, The Art of Computer Programming, Vol.3, AddWes 1973, 84-95
##  Donald Shell, CACM 2, July 1959, 30-32
##  Robert Sedgewick, Algorithms 2nd ed., AddWes 1988, 107-123
##
##  In the case where theer is just one argument, we use the code supplied by Jean
##  Michel, using the internal function Set
##
##  Sort does in-place sorting. Returned list is the same object as original list.
Sort := function ( arg )
    local   list,  isLess,  i,  k,  h,  v, l, both;

    if Length(arg) = 1 and IsList(arg[1])  then
        list := arg[1];
        l:=Length(list);
        both := [  ];
        for i  in [ 1 .. l ]  do
            both[i] := [ list[i], i ];
        od;
        both:=Set( both );
        list{[1..l]} := both{[1..l]}[1];
    elif Length(arg) = 2  and IsList(arg[1])  and IsFunc(arg[2])  then
        list := arg[1];  isLess := arg[2];
        h := 1;  while 9 * h + 4 < Length(list)  do h := 3 * h + 1;  od;
        while 0 < h  do
            for i  in [ h+1 .. Length(list) ]  do
                v := list[i];  k := i;
                while h < k  and isLess( v, list[k-h] )  do
                    list[k] := list[k-h];   k := k - h;
                od;
                list[k] := v;
            od;
            h := QuoInt( h, 3 );
        od;

    else
        Error("usage: Sort( <list> ) or Sort( <list>, <func> )");
    fi;
    return list;
end;


#############################################################################
##
#F  SortParallel(<list>,<list2>)  . . . . . . . .  sort two lists in parallel
##
SortParallel := function ( arg )
    local   lst,        # list <lst> to be sorted, first argument
            par,        # list <par> to be sorted parallel, second argument
            isLess,     # comparison function, optional third argument
            gap,        # gap width
            l, p,       # elements from <lst> and <par>
            i, k,       # loop variables
            both;       # special list to pass to Set for fast sorting

    if Length(arg) = 2  and IsList(arg[1])  then
        lst := arg[1];
        par := arg[2];
        l:=Length(lst);
        both := [  ];
        for i  in [ 1 .. l ]  do
            both[i] := [ lst[i], i , par[i]];
        od;
        both:=Set( both );
        for i  in [ 1 .. l ]  do
            lst[i] := both[i][1];
            par[i] := both[i][3];
        od;

        
    elif Length(arg) = 3  and IsList(arg[1])  and IsFunc(arg[3])  then
        lst := arg[1];
        par := arg[2];
        isLess := arg[3];
        gap := 1;  while 9*gap+4 < Length(lst)  do gap := 3*gap+1;  od;
        while 0 < gap  do
            for i  in [ gap+1 .. Length(lst) ]  do
                l := lst[i];  p := par[i];  k := i;
                while gap < k  and isLess( l, lst[k-gap] )  do
                    lst[k] := lst[k-gap];  par[k] := par[k-gap];  k := k-gap;
                od;
                lst[k] := l;  par[k] := p;
            od;
            gap := QuoInt( gap, 3 );
        od;

    else
        Error("usage: SortParallel(<lst>,<par>[,<func>])");
    fi;

end;



#############################################################################
##
#F  Sortex(<list>) . . . sort a list (stable), return the applied permutation
##
Sortex := function ( list )
    local   both, perm, i;

    # make a new list that contains the elements of <list> and their indices
    both := [];
    for i  in [1..Length(list)]  do
        both[i] := [ list[i], i ];
    od;

    # sort the new list according to the first item (stable)
    both := Set( both );

    # copy back and remember the permutation
    perm := [];
    for i  in [1..Length(list)]  do
        list[i] := both[i][1];
        perm[i] := both[i][2];
    od;

    # return the permutation mapping old <list> onto the sorted list
    return PermList( perm )^(-1);
end;


############################################################################
##
#F  Permuted( <list>, <perm> ) . . .apply permutation <perm> to list <list>
##  
##  make maximum use of kernel functions
##
Permuted := function ( list, perm )
  return list{OnTuples([1..Length(list)],perm^-1)}; 
end;



#############################################################################
##
#F  PositionSorted( <list>, <elm> ) . . . .  find an element in a sorted list
##
##  'PositionSorted' uses a binary search instead of the linear  search  used
##  'Position'.  This takes log to base 2 of  'Length( <list> )' comparisons.
##  The list <list> must be  sorted  however  for  'PositionSorted'  to work.
##
##  Jon Bentley, Programming Pearls, AddWes 1986, 85-88
##
PositionSorted := function ( arg )
    local   list,  elm,  isLess,  l,  m,  h;

    if Length(arg) = 2  and IsList(arg[1])  then
        list := arg[1];  elm := arg[2];
        l := 0;  h := Length(list)+1;
        while l+1 < h  do               # list[l]<elm & elm<=list[h] & l+1<h
            m := QuoInt( l + h, 2 );    # l < m < h
            if list[m] < elm  then l := m;
            else                   h := m;
            fi;
        od;
        return h;                       # list[l]<elm & elm<=list[h] & l+1=h

    elif Length(arg) = 3  and IsList(arg[1])  and IsFunc(arg[3])  then
        list := arg[1];  elm := arg[2];  isLess := arg[3];
        l := 0;  h := Length(list)+1;
        while l+1 < h  do               # list[l]<elm & elm<=list[h] & l+1<h
            m := QuoInt( l + h, 2 );    # l < m < h
            if isLess( list[m], elm )  then l := m;
            else                            h := m;
            fi;
        od;
        return h;                       # list[l]<elm & elm<=list[h] & l+1=h

    else
        Error("usage: PositionSorted( <list>, <elm> [, <func>] )");
    fi;
end;


#############################################################################
##
#F  Product( <list> ) . . . . . . . . . . . product of the elements in a list
##
##  'Product( <list> )' \\
##  'Product( <list>, <func> )'
##
##  When used in the first way 'Product' returns the product of the  elements
##  of the list <list>.  When used in the second way  'Product'  applies  the
##  function <func>, which must  be  a  function  taking  one  argument,  and
##  returns the product of the results.  In either case if  <list>  is  empty
##  'Product' returns 1.
##
Product := function ( arg )
    local  list,  func,  prod,  i;

    if Length(arg) = 1  then
        list := arg[1];
        if Length(list) = 0  then
            prod := 1;
        else
            prod := list[1];
            for i  in [ 2 .. Length(list) ]  do
                prod := prod * list[i];
            od;
        fi;

    elif Length(arg) = 2  and IsList(arg[1])  and IsFunc(arg[2])  then
        list := arg[1];  func := arg[2];
        if Length(list) = 0  then
            prod := 1;
        else
            prod := func( list[1] );
            for i  in [ 2 .. Length(list) ]  do
                prod := prod * func( list[i] );
            od;
        fi;

    else
        Error("usage: Product( <list> ) or Product( <list>, <func> )");

    fi;

    return prod;
end;


#############################################################################
##
#F  Sum( <list> ) . . . . . . . . . . . . . . . sum of the elements of a list
##
Sum := function ( arg )
    local  list,  func,  sum,  i;

    if Length(arg) = 1  then
        list := arg[1];
        if Length(list) = 0  then
            sum := 0;
        else
            sum := list[1];
            for i  in [ 2 .. Length(list) ]  do
                sum := sum + list[i];
            od;
        fi;

    elif Length(arg) = 2  and IsList(arg[1])  and IsFunc(arg[2])  then
        list := arg[1];  func := arg[2];
        if Length(list) = 0  then
            sum := 0;
        else
            sum := func( list[1] );
            for i  in [ 2 .. Length(list) ]  do
                sum := sum + func( list[i] );
            od;
        fi;

    else
        Error("usage: Sum( <list> ) or Sum( <list>, <func> )");

    fi;

    return sum;
end;

#############################################################################
##
#F  Xor(<list>)                 . . . . . . .  Xor list members together
##
Xor := function ( arg )
    local  list,  func,  xor,  i;

    if Length(arg) = 1  then
        list := arg[1];
        if Length(list) = 0  then
            xor := 1;
        else
            xor := list[1];
            for i  in [ 2 .. Length(list) ]  do
                xor := BinXor(xor, list[i]);
            od;
        fi;

    elif Length(arg) = 2  and IsList(arg[1])  and IsFunc(arg[2])  then
        list := arg[1];  func := arg[2];
        if Length(list) = 0  then
            xor := 1;
        else
            xor := func( list[1] );
            for i  in [ 2 .. Length(list) ]  do
                xor := BinXor(xor, func( list[i] ));
            od;
        fi;

    else
        Error("usage: Xor( <list> ) or Xor( <list>, <func> )");

    fi;

    return xor;
end;

#############################################################################
##
#F  Iterated( <list>, <func> )  . . . . . . .  iterate a function over a list
##
Iterated := function ( list, func )
    local  res,  i;
    if Length(list) = 0  then
        Error("Iterated: <list> must contain at least one element");
    fi;
    res := list[1];
    for i  in [ 2 .. Length(list) ]  do
        res := func( res, list[i] );
    od;
    return res;
end;


#############################################################################
##
#F  Maximum( <obj>, <obj>... )  . . . . . . . . . . . . . maximum of integers
##
Maximum := function ( arg )
    local   max, elm;
    if   Length(arg) = 1  and IsRange(arg[1])  then
        if Length(arg[1]) = 0  then
            Error("Maximum: <list> must contain at least one element");
        fi;
        max := arg[1][Length(arg[1])];
        if max < arg[1][1] then max:= arg[1][1]; fi;
    elif Length(arg) = 1  and IsList(arg[1])  then
        if Length(arg[1]) = 0  then
            Error("Maximum: <list> must contain at least one element");
        fi;
        max := arg[1][Length(arg[1])];
        for elm  in arg[1]  do
            if max < elm  then
                max := elm;
            fi;
        od;
    elif  Length(arg) = 2  then
        if arg[1] > arg[2]  then return arg[1];
        else                     return arg[2];
        fi;
    elif Length(arg) > 2  then
        max := arg[Length(arg)];
        for elm  in arg  do
            if max < elm  then
                max := elm;
            fi;
        od;
    else
        Error("usage: Maximum( <obj>, <obj>... ) or Maximum( <list> )");
    fi;
    return max;
end;

#############################################################################
##
#F  MaximumPosition( <list> ) . . . . . . . . position of maximum of integers
##

MaximumPosition := function ( arg )
    local   max, i, pos;
    if   Length(arg) = 1 and IsList(arg[1])  then
        if Length(arg[1]) = 0  then
            Error("MaximumPosition: <list> must contain at least one element");
        fi;
        pos:=1;
        max := arg[1][1];
        for i  in [1..Length(arg[1])]  do
            if max < arg[1][i]  then
                max := arg[1][i];
                pos := i;
            fi;
        od;
        return pos;
    else
        Error("MaximumPosition: arg is not a <list>");
    fi;
end;

#F Maximum0(<list>)
#F Maximum0(<obj>, <obj>, ...)
#F
#F Same as Maximum() but returns 0 if empty list is given
#F
Maximum0 := function ( arg )
    local   max, elm;
    if   Length(arg) = 1  and IsRange(arg[1])  then
        if Length(arg[1]) = 0  then return 0; fi;
        max := arg[1][Length(arg[1])];
        if max < arg[1][1] then max:= arg[1][1]; fi;
    elif Length(arg) = 1  and IsList(arg[1])  then
        if Length(arg[1]) = 0  then return 0; fi;
        max := arg[1][Length(arg[1])];
        for elm  in arg[1]  do
            if max < elm  then
                max := elm;
            fi;
        od;
    elif  Length(arg) = 2  then
        if arg[1] > arg[2]  then return arg[1];
        else                     return arg[2];
        fi;
    elif Length(arg) > 2  then
        max := arg[Length(arg)];
        for elm  in arg  do
            if max < elm  then
                max := elm;
            fi;
        od;
    else
        Error("usage: Maximum( <obj>, <obj>... ) or Maximum( <list> )");
    fi;
    return max;
end;


#############################################################################
##
#F  Minimum( <obj>, <obj>... )  . . . . . . . . . . . . . minimum of integers
##
Minimum := function ( arg )
    local   min, elm;
    if   Length(arg) = 1  and IsRange(arg[1])  then
        if Length(arg[1]) = 0  then
            Error("Minimum: <list> must contain at least one element");
        fi;
        min := arg[1][Length(arg[1])];
        if min > arg[1][1] then min:= arg[1][1]; fi;
    elif Length(arg) = 1  and IsList(arg[1])  then
        if Length(arg[1]) = 0  then
            Error("Minimum: <list> must contain at least one element");
        fi;
        min := arg[1][Length(arg[1])];
        for elm  in arg[1]  do
            if min > elm  then
                min := elm;
            fi;
        od;
    elif  Length(arg) = 2  then
        if arg[1] < arg[2]  then return arg[1];
        else                     return arg[2];
        fi;
    elif Length(arg) > 2  then
        min := arg[Length(arg)];
        for elm  in arg  do
            if min > elm  then
                min := elm;
            fi;
        od;
    else
        Error("usage: Minimum( <obj>, <obj>... ) or Minimum( <list> )");
    fi;
    return min;
end;

#############################################################################
##
#F Minimum0(<list>)
#F Minimum0(<obj>, <obj>, ...)
#F
#F Same as Minimum() but returns 0 if empty list is given
#F
Minimum0 := arg -> Cond(arg=[] or arg[1]=[], 0, ApplyFunc(Minimum, arg));

#############################################################################
##
#F  RandomList( <list> )  . . . . . . . . return a random element from a list
##
#N  31-May-91 martin 'RandomList' should be internal
##
R_N := 1;
R_X := [];

RandomList := function ( list )
    R_N := R_N mod 55 + 1;
    R_X[R_N] := (R_X[R_N] + R_X[(R_N+30) mod 55+1]) mod 2^28;
    return list[ QuoInt( R_X[R_N] * Length(list), 2^28 ) + 1 ];
end;

RandomSeed := function ( n )
    local  i;
    R_N := 1;  R_X := [ n ];
    for i  in [2..55]  do
        R_X[i] := (1664525 * R_X[i-1] + 1) mod 2^28;
    od;
    for i  in [1..99]  do
        R_N := R_N mod 55 + 1;
        R_X[R_N] := (R_X[R_N] + R_X[(R_N+30) mod 55+1]) mod 2^28;
    od;
end;

if R_X = []  then RandomSeed( 1 );  fi;



#############################################################################
##
#F  PositionSet( <l>, <x>[, <less> ) . . . . like 'Position', but the user
#F  is responsible for <l> beeing sorted
##  
##  This is only a slight variation of 'PositionSorted'. Due to Chevie team
##  The difference from PositionSorted is that it returns false if the object is
##  not present, rather than returning the position where it would be inserted
##  

PositionSet := function(arg)
    local  list, elm, isLess, l, m, h;
    if Length( arg ) = 2 and IsList( arg[1] )  then
        list := arg[1];
        elm := arg[2];
        l := 0;
        h := Length( list ) + 1;
        while l + 1 < h  do
            m := QuoInt( l + h, 2 );
            if list[m] < elm  then
                l := m;
            else
                h := m;
            fi;
        od;
    elif Length( arg ) = 3 and IsList( arg[1] ) and IsFunc( arg[3] )  then
        list := arg[1];
        elm := arg[2];
        isLess := arg[3];
        l := 0;
        h := Length( list ) + 1;
        while l + 1 < h  do
            m := QuoInt( l + h, 2 );
            if isLess( list[m], elm )  then
                l := m;
            else
                h := m;
            fi;
        od;
    else
        Error( "usage: PositionSet( <list>, <elm> [, <func>] )" );
    fi;
    if IsBound(list[h]) and list[h]=elm then
      return h;
    else
      return false;
    fi;
end;


#############################################################################
##
#F  SortingPerm( <list> ) . . . . . . . returns the same as 'Sortex( <list> )'
#F  but does *not* change the argument
##  
##    

SortingPerm := function ( list )
  local  both, perm, i, l;
  l:=Length(list);
  both:=[];
  for i in [1..l]  do
    both[i]:=[list[i],i];
  od;
  both:=Set(both);
  perm := [];
  perm{[1..l]} := both{[1..l]}[2];
  return PermList(perm)^-1;
end;

#############################################################################
##
#F  PermListList( <lst>, <lst2> ) . . . . what permutation of <lst> is <lst2>
##
##  PermListList finds which permutation p of [1..Length(lst)] is such 
##  that lst[i^p]=lst2[i] 
##  It returns false if there is no such permutation.
##
PermListList := function(l1,l2)
local res;
  l1:=ShallowCopy(l1);l2:=ShallowCopy(l2); # to not destroy l1 and l2
  res:=Sortex(l2)* Sortex(l1)^-1;
  if l1<>l2 then return false; else return res;fi;
end;


#############################################################################
##
#F Take( <lst>, <n> )
##
## Take(lst, n) returns the prefix of lst of length n, 
##              or lst itself if n > Length(lst).
##
## See also: TakeLast, Drop, DropLast, SplitAt
##
Take := (lst, n) -> lst{[1..Minimum(n,Length(lst))]};

#############################################################################
##
#F TakeLast( <lst>, <n> )
##
## TakeLast(lst, n) returns the suffix of lst of length n, 
##              or lst itself if n > Length(lst).
##
## See also: Take, Drop, DropLast, SplitAt
##
TakeLast := (lst, n) -> let(L:=Length(lst),lst{[Maximum(1,L-n+1) .. L]});

#############################################################################
##
#F Drop( <lst>, <n> )
##
## Take(lst, n) returns the prefix of lst of length n, 
##              or lst itself if n > Length(lst).
## Drop(lst, n) returns the suffix of lst of length n, or lst itself if n > Length(lst).
##              or lst itself if n > Length(lst).
## DropLast(lst, n) returns all but the last n elements of lst
##
## SplitAt(lst, n) is equivalent to [Take(lst, n), Drop(lst, n)]
##
Drop := (lst, n) -> lst{[(n+1)..Length(lst)]};

#############################################################################
##
#F DropCond( <lst>, <func>,  <n> )
##
## Returns a copy of the <lst> without first <n> elements for which <func>
## predicate evaluated to true.
##
DropCond := function( lst, func, n)
    local e, r;
    r := [];
    for e in lst do
        if not func(e) or n=0 then
            Add(r, e);
        else
            n := n-1;
        fi; 
    od;
    return r;
end;

#############################################################################
##
#F DropLast( <lst>, <n> )
##
## DropLast(lst, n) returns all but the last n elements of lst
##
DropLast := (lst, n) -> lst{[1..Length(lst)-n]};

#############################################################################
##
#F SplitAt( <lst>, <n> )
##
## SplitAt(lst, n) is equivalent to [Take(lst, n), Drop(lst, n)]
##
SplitAt := (lst, n) -> [lst{[1..n]}, lst{[(n+1)..Length(lst)]}];

#############################################################################
##
#F FoldL( <lst>, <func>, <z> )
##
## Applied to a binary operator <func>, a starting value <z> (typically the
## left-identity of the operator), and a list <lst>, reduces the list using
## the binary operator, from left to right:
##
##   FoldL([x1, x2, ..., xn], f, z) == f( ...( f( f(z, x1), x2 ) ... ), xn)
##
FoldL := function ( list, func, z )
    local  res,  i;
    res := z; 
    for i  in list do
        res := func(res, i);
    od;
    return res;
end;

#############################################################################
##
#F FoldR( <lst>, <func>, <z> )
##
## Applied to a binary operator <func>, a starting value <z> (typically the
## left-identity of the operator), and a list <lst>, reduces the list using
## the binary operator, from right to left:
##
##   FoldR([x1, x2, ..., xn], f, z) == f(x1, f(x2, ... f(xn, z) )..)
##
FoldR := (list, func, z) -> FoldL(Reversed(list), func, z);

#############################################################################
##
#F FoldL1( <lst>, <func> )
## 
## Same as FoldL, but uses lst[1] as the starting value. 
##    FoldL1([x1, x2, ..., xn], f) == FoldL([x2, ..., xn], f, x1)
##
FoldL1 := Iterated;

#############################################################################
##
#F FoldR1( <lst>, <func> )
## 
## Same as FoldR, but uses lst[N] (last list element) as the starting value. 
##    FoldR1([x1, x2, ..., xn], f) == FoldR([x1, ..., x(n-1)], f, xn)
##
FoldR1 := (lst,func) -> FoldL1(Reversed(lst), func);

#############################################################################
##
#F ScanL( <lst>, <func>, <z> )
##
##  ScanL is similar to FoldL, but returns a list of successive reduced values
##  from the left:
##
##       ScanL([x1, x2, ...], f, z) == [z, f(z, x1), f(f(z, x1), x2), ...]
##
##  Note that Last(ScanL(lst, f, z)) == FoldL(lst, f, z)
##
ScanL := function ( lst, func, z )
    local  res,  i, pred;
    # Length(res) == Length(lst)+1  [contents irrelevant]
    res := ShallowCopy(lst); 
    Add(res, 0);

    res[1] := z; pred := z;
    for i in [1..Length(lst)] do
        pred := func(pred, lst[i]);
	res[i+1] := pred;
    od;
    return res;
end;

#############################################################################
##
#F ScanR( <lst>, <func>, <z> )
##
##  ScanR is similar to FoldR, but returns a list of successive reduced values
##  from the right to left:
##
##       ScanR([x1, x2, ...], f, z) == [..., f(f(z, xn), x(n-1)), f(z, xn), z ]
##
##  Note that first element of ScanR(lst, f, z) is equal to FoldR(lst, f, z)
##

ScanR := (lst, func, z) -> Reversed(ScanL(Reversed(lst), func, z));


#############################################################################
##
#F ScanL1( <lst>, <func> )
## 
## Same as ScanL, but uses lst[1] as the starting value. 
##    ScanL1([x1, x2, ..., xn], f) == ScanL([x2, ..., xn], f, x1)
##
ScanL1 := function ( lst, func )
    local  res,  i, pred;
    if Length(lst) = 0  then
        Error("ScanL1: <lst> must contain at least one element");
    fi;

    # Length(res) == Length(lst) 
    res := ShallowCopy(lst); 

    pred := res[1];
    for i in [2..Length(lst)] do
        pred := func(pred, lst[i]);
	res[i] := pred;
    od;
    return res;
end;

ScanR1 := (lst, func) -> Reversed(ScanL1(Reversed(lst), func));

#############################################################################
##
#F Replicate( <n>, <val> )
##
## Returns a list of length <n> with <val> the value of every element
##
Replicate := function ( n, val )
    local i, res;
    res := [1..n];
    for i in [1..n] do
        res[i] := val;
    od;
    return res;
end;


#############################################################################
##
#F Choose( <lst>, <indices> )  . . . . . . . . . . . . . . .  same as Sublist
##
Choose := Sublist;


#############################################################################
##
#F Last( <lst> ) . . . . . . . . . . . . . . . . . . . last element of a list
##
Last := lst -> lst[Length(lst)];


#############################################################################
##
#F BasisVec( <size>, <n> ) . . . . . . . .  <n>-th basis vector of given size
##
## Returns a list of <size> elements with 1 in <n>-th position, and 0 in others.
#
BasisVec := (size, n) -> Concatenation(Replicate(n, 0), [1], Replicate(size-n-1, 0));

#############################################################################
##
#F ScaledBasisVec( <size>, <n>, <v>  . . . . . basis vector multiplied by <v>
##
## Returns a list of <size> elements with <v> in <n>-th position, and 0 in others.
##
ScaledBasisVec := (size, n, v) -> Concatenation(Replicate(n, 0), [v], Replicate(size-n-1, 0));

############################################################################
##
#F RemoveList(<lst>, <elt>)
#F   returns a list after removing all occurences of element i in L
#F   input is a list L and element i. Return type is a list shorter
#F   than L or unchanged L if i is not found in L.
#F   added by Riddhi, 11/15/04
RemoveList := function(lst, elt)
   local j, result;
   result := [];
   for j in [1..Length(lst)] do
	if lst[j]<>elt then
		Append(result,[lst[j]]);
	fi;
   od;
   return result;
end;

############################################################################
##
#F ListDifference(<lst1>, <lst2>)
#F    Returns a new list obtained by taking <lst1> and one by one removing elements 
#F    that appear in <lst2>. At every removal only the first instance of an element is 
#F    removed from <lst1>. 
#F
#F  For example: ListDifference([1,2,3,4,3], [1, 3, 4]) == [2, 3]
#F
#F  Also elements of <lst1> are not reordered (unlike in Difference(), which is a set operation)
#F    
ListDifference := function(lst1, lst2)
   local wk, elt, cut, p;
   cut := []; 
   wk := ShallowCopy(lst1);
   for elt in lst2 do
       p := Position(wk, elt);
       if p<>false then
           Add(cut, p);
           Unbind(wk[p]);
       fi;
   od;
   return lst1{Difference([1..Length(lst1)], cut)};
end;

############################################################################
##
#F ListWithout(<lst>, <idx>)
#F   returns a list without an element at position <idx>
#F
ListWithout := (lst, idx) -> Concatenation(lst{[1..idx-1]}, lst{[idx+1..Length(lst)]});

############################################################################
##
#F ListReplace(<lst>, <idx>, <elt>)
#F   returns a new list, identical to <lst>, but with element <elt> at position <idx>
#F
ListReplace := function(lst, idx, elt)
    lst := ShallowCopy(lst);
    lst[idx] := elt;
    return lst;
end;

############################################################################
##
#F LastPosition( <lst>, <pred> )
#F     Returns position of the last element in lst for which pred is true.
LastPosition := function( lst, pred )
   local i,r,hit;
   hit := When( IsFunc(pred), pred, x -> x = pred );
   r := Length(lst) + 1;
   for i in [1..Length(lst)] do
       if hit(lst[r-i]) then return r-i; fi;
   od;
   return 0;
end;

############################################################################
##
#F FirstPosition( <lst>, <pred> )
#F     Returns position of the first element in lst for which pred is true.
FirstPosition := function( lst, pred )
   local i,hit;
   hit := When( IsFunc(pred), pred, x -> x = pred );
   for i in [1..Length(lst)] do
       if hit(lst[i]) then return i; fi;
   od;
   return 0;
end;


############################################################################
##
#F StripList( <lst> )
#F     If the list has only one element, remove parenthesis
StripList:=function(l)
  if IsList(l) and Length(l)=1 then
    return l[1];
  else
    return l;
  fi;
end;

############################################################################
##
#F SortRecordList( <rec lst>, <fctn> )
#F     Sorts the <rec lst> according to the result of the <fctn>

SortRecordList := function(reclst,fctn)
   local l, bucket, lasttype, v;

   reclst := ConcatList(reclst, x->[[fctn(x), x]]);
   Sort(reclst);
   
   l:=[];
   bucket:=[];
   lasttype:=0;
   for v in reclst do
      if v[1]=lasttype then
          Add(bucket,v[2]);
      else
          if (Length(bucket)>0) then
              Add(l,bucket);
          fi;
          bucket:=[v[2]];
          lasttype:=v[1];
      fi;
   od;
   Add(l,bucket);

   return l;
end;

############################################################################
##
#F Zip2 ( <lst1>, <lst2> )
#F     Zip2 "zips" two lists of the same size by returning the list of pairs
#F     of corresponding arguments
#F
#F     ex: Zip2([1,2,3],[13,14,15]) = [ [ 1, 13 ], [ 2, 14 ], [ 3, 15 ] ]
#F
#F     By extension, it also supports arguments that are not lists and pairs
#F     of pairs them.

Zip2 := (lst1, lst2) -> Cond(
    not(IsList(lst1) and IsList(lst2)), [[lst1,lst2]],
    Checked(IsList(lst1) and IsList(lst2) and Length(lst1)=Length(lst2),
            List([1..Length(lst1)], i -> [lst1[i], lst2[i]])));

Declare(Zip2E);
Zip2E := (lst1, lst2) -> Cond(
    not IsList(lst1) or not IsList(lst2) or IsString(lst1) or IsString(lst2), [lst1, lst2],
    Checked(Length(lst1)=Length(lst2), List([1..Length(lst1)], i -> Zip2E(lst1[i], lst2[i]))));
 
ListSubstRecursive := (lst, from, to) -> Cond(
    lst = from, to,
    BagType(lst)<>T_STRING and IsList(lst), List(lst, c -> ListSubstRecursive(c, from, to)),
    lst);

############################################################################# 
##
#F ListOrderMat( <lst> )
#F
#F This function verifies the ordering function on list elements, as defined
#F by operations.\= and operations.\<. 
#F 
#F The expected output should be have 1s, 0s and =s on diagonal.
#F Should have no Is or Es.
#F
#F The function is useful when validating ordering of elements of different
#F types, with user defined ordering functions.
#F
#F I = inconsistent 
#F E = crash
#F
#F Example:
#F
#F spiral> PrintMat(ListOrderMat([1, 2, "string", rec()]));
#F [ [ =, 1, 1, 1 ], 
#F   [  , =, 1, 1 ], 
#F   [  ,  , =, 1 ], 
#F   [  ,  ,  , = ] ]
#F
ListOrderMat := l -> let(n:=Length(l),
    List([1..n], i -> List([1..n], j -> let(res := Try(Cond(
         l[i] < l[j] and l[j] > l[i] and l[i]<>l[j], 1,
         l[i] > l[j] and l[j] < l[i] and l[i]<>l[j], 0,
         l[i] = l[j] and (not (l[i]<l[j])) and (not (l[i]>l[j])) 
                     and (not (l[j]<l[i])) and (not (l[j]>l[i])), "=",
         "I")), 
         When(res[1], res[2], "E")))));


############################################################################# 
## List Statistics
############################################################################# 

############################################################################# 
##
#F Mean( <list )
#F
Mean := list -> Sum(list) / Length(list);

############################################################################# 
##
#F Variance( <list )
#F
Variance := list -> let(
    n := Length(list), 
    m := Mean(list),
    Mean(List(list, x -> (x-m)^2)) * n / (n-1)
);

############################################################################# 
##
#F StdDev( <list )
#F
StdDev := list -> Double(Variance(list)) ^ (1/2);


############################################################################# 
##
#F RemoveAdjacentDuplicates(<lst>)
#F Removes an element is it is similar to on of the ones next to it
##
RemoveAdjacentDuplicates := function(l)
   local res, cur, e, i;

   res :=[];
   for i in [1..Length(l)] do
      e := l[i];
      if ((i=1) or (e<>cur)) then
          Add(res, e);
          cur:=e;
      fi;
   od;
   return res;
end;

############################################################################
##
#F  SepList( <list>, [<func>,] <sep> ) . . . list with <sep> between <func>(<list>) elements
##
##

SepList := function( arg )
    local i, result, list, sep, func;
    list := arg[1];
    if Length(list)=0 then return []; fi;
    func := When(Length(arg)=3, arg[2], e -> e);
    sep  := When(Length(arg)=3, arg[3], arg[2]);
    result := [func(list[1])];
    for i in [2..Length(list)] do
        Add(result, sep);
        Add(result, func(list[i]));
    od;
    return result;
end; 

ConcatSepList := (arg) -> let( 
    list := arg[1],
    func := When(Length(arg)=2, e -> e, arg[2]),
    sep  := When(Length(arg)=3, arg[3], arg[2]),
    r := ConcatList(SepList(list, func, sep), e -> e), 
    When(IsString(sep) and IsString(r), String(r), r)
);

############################################################################
##
#F ShiftList(<lst>, <steps>, <c>) - shifting list left or right by <steps>
#F ( lst[i] := lst[i-steps] ) and shifting in <c>. 
##

ShiftList := (lst, steps, c) -> let( l := Length(lst), 
    Cond( steps>=l or steps<=-l, Replicate(l, c), 
          steps<0, lst{[-steps..l]} :: Replicate(-steps, c),
          steps>0, Replicate(steps, c) :: lst{[1..l-steps]},
          lst));

############################################################################
#F 
#F GroupList(<list>, <func>) groups <list> elements by categories returned 
#F by <func>(element).
#F ex: GroupList( [1, 2, 3, 4, 5], e -> e mod 3 )
#F     [ [0, [ 3 ]], [1, [ 1, 4 ]], [2, [ 2, 5 ]] ]
##

GroupList := function(lst, func)
    local slst, r, e;
    slst := Sort(ShallowCopy(lst), (a, b) -> func(a)<func(b));
    r    := [];
    for e in slst do
        if r = [] or Last(r)[1] <> func(e) then
            r := r :: [[func(e), [e]]];
        else
            Add(Last(r)[2], e);
        fi;
    od;
    return r;
end;

