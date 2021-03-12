# -*- Mode: shell-script -*- 

#############################################################################
##
#A  saggroup.g                  GAP library                      Bettina Eick
##
##
#Y  Copyright (C) 2018-2021, Carnegie Mellon University
#Y  All rights reserved.  See LICENSE for details.
#Y  
#Y  This work is based on GAP version 3, with some files from version 4.  GAP is
#Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
##
##  This file contains all functions for creating special ag groups.
##
##


#############################################################################
##
#F  InfoSagGroup? . . . . . . . . . . . . . . . . . . . . . print information
##
if not IsBound(InfoSagGroup1)  then InfoSagGroup1 := Ignore;  fi;
if not IsBound(InfoSagGroup2)  then InfoSagGroup2 := Ignore;  fi;
if not IsBound(InfoSagGroup3)  then InfoSagGroup3 := Ignore;  fi;

#############################################################################
##
#V  SagWeights  . . . . . . . . .  weight functions for the special ag series
##
##  The weight  of  a group  element g  is  a threetuple  w  = [w1,   w2, w3]
##  indicating the  subgroup of the  Leedham-Green series corresponding to g,
##  such that w1 corresponds to the factor  of the lower nilpotent series, w3
##  is the prime  of the Sylowsubgroup  of the factor  in the lower nilpotent
##  series, w2 corresponds to the factor of the lower-w3-central series of the
##  Sylow-w3-subgroup of the w1's factor of the lower nilpotent series.
##
##  weight functions are records of functions  constructing the new weight of
##  the generators of a group such that they refer to the leedhamgreen series
##  of the group.
##
##  'adj( <g>, <weight> )'
##
##  'adj' is called in 'Modify', it calculates the  weight of a component
##  gg of the generator <g> and returns the weight.
##
##  'one( <g> )'
##
##  'one' is called in 'Initialize' and  returns the initialization of the
##  weight of the group element.
##
##  'relevant( <weights>, <i>, <j> )'
##
##  'relevant' is called in 'Echelonise' and  returns true if it is necessary 
##  to modify the base with the power/commutator.
## 
##  'weight( <weights>, <i>, <j>, <power/commutator> )'
##
##  'weight' is called  in 'Echelonise', calculates the weight  of a power or
##  commutator  and  returns  a  weight.  'relevant' is true  if  'weight' is
##  called.
##
SagWeights := rec(

    adj := function( g, wt )
        wt := Copy( wt );
        wt[ 3 ] := RelativeOrderAgWord( g );
        return wt;
    end,

    one := function( g )
        return [ 1, 1, RelativeOrderAgWord( g ) ];
    end,

    relevant := function( w, i, j, h )
        if i = j  and (w[i][1] = h-1 or w[i][1] = h+1)  then
            return true;
        else
            if w[i][1] = w[j][1]  then
                if w[i][1] = h-1 and w[i][3] = w[j][3] and 
                  (w[i][2] = 1 or w[j][2] = 1)  then
                    return true;
                elif w[i][1] = h  and w[i][3] <> w[j][3]  then
                    return true;
                elif w[i][1] >= h+1  then
                    return true;
                else
                    return false;
                fi;
            elif w[i][1] >= h+1 and w[j][1] >= h+1 then
                 return true;
            elif w[i][1] = h+1  and w[j][1] <= h  and w[j][2] = 1  then 
                 return true;
            elif w[i][1] <= h  and w[j][1] = h+1  and w[i][2] = 1  then 
                 return true;
            else 
                 return false;
            fi;
        fi;
    end,

    weight := function( w, i, j, h, g )
        local p;

        p := RelativeOrderAgWord(g);
        if i = j  then
            if w[i][1] = h-1  then
                return [ w[i][1], w[i][2]+1, w[i][3] ];
            else
                return w[i];
            fi;
        else
            if w[i][1] = w[j][1]  and w[i][1] = h-1  then
                return [ w[i][1], w[i][2]+w[j][2], w[i][3] ];
            elif w[i][1] = w[j][1]  and w[i][1] = h  then
                return [ w[i][1]+1, 1, p ];
            elif w[i][1] = w[j][1]  and w[j][1] >= h+1  then
                if w[i][3] <> w[j][3] or w[i][3] <> p then
                    return [w[i][1]+1, 1, p];
                else
                    return [w[i][1], 1, p];
                fi;
            else
                return [ Maximum( w[i][1],w[j][1] ), 1, p ];
            fi;
        fi;
    end,

    useful := function( w, i, j, h )

        if i = j and w[i][1] >= h+1 then
            return false;
        elif i<>j and w[i][1] = w[j][1] and w[i][1] >= h+1 and
             w[i][3] = w[j][3] then
            return false;
        else
            return true;
        fi;
    end
    
);

#############################################################################
##
#F  AgGroupOps.ModifyBase( <system>, <g>, <weight> )   modify 'base' with <g>
##
##  'ModifyBase' modifies  the record entry 'base'  of <system>  with <g> and
##  returns the place in which <g> is inserted.  <g> is either a base element
##  or a power/commutator of a base element with another.
##
AgGroupOps.ModifyBase := function ( system, g, wt )

    local d,         #  depth of g
          gg,        #  g*B[d]^n
          S,         #  list of components of gg
          s,         #  component of gg
          min,       #  minimal position 
          tmp, i, j;       

    # the trivial case
    if g = system.G.identity  then
        return Length( system.base ) + 1;
    fi;
    d := DepthAgWord( g );

    # an easy case occuring in the 'Initialize' function
    if system.base[ d ] = system.G.identity then
        system.base[ d ] := g;
        system.weights[ d ] := wt;
        return d;

    # the other cases
    else
        gg := ReducedAgWord( system.base[ d ], g );
        S := Components( system.G, gg );

        # insert g in base 
        if system.weights[ d ] < wt  then
            tmp := system.weights[ d ];
            system.weights[ d ] := wt;
            system.base[ d ] := g;
            InfoSagGroup3( "#I  insert: ",g, "\n#I  at position: ", d,
                           " with weight: ", wt, "\n" );

            # correct work-flag
            system.work[ d ] := List( system.work[ d ], x -> true );
            for i in [d..Length( system.base )]  do
                system.work[ i ][ d ] := true;
            od;

            # ModifyBase with components of gg
            for s  in S  do 
                system.G.operations.ModifyBase( system, s, 
                        system.wf.adj( s,tmp ));
            od;
            return d;

        # base is not changed 
        else
         
            # modify with components of gg
            min := Length( system.base ) + 1;
            for s  in S  do
                tmp := system.wf.adj( s, wt );
                min := Minimum( 
                           min,
                           system.G.operations.ModifyBase(system,s,tmp) );
            od;
            return min;
        fi;
    fi;
end;


#############################################################################
##
#F  AgGroupOps.InitializeSystem ( <G>, <gens>, <wf> ) . .   initialize 'base'
##
##  'InitializeSystem'  computes a PAG-system  of <G>, such that each element
##  of the  PAG-system has a  prime power order.  Furthermore  it induces the
##  weight of each element  in the PAG-system.  The function returns a record
##  consisting of the entries
##
##  'G',        the group <G>
##  'base',     the calculated PAG-system
##  'weights',  the induced weights
##  'wf',       the weightfunction
##  'work',     the list that gives the elements which have to be considered.
##
AgGroupOps.InitializeSystem := function( G, gens, wf )

    local maxDepth,     # Length of composition series
          system,       # record with all information
          g,            # one element of gens 
          s,            # one component of g
          S;            # list of components of g

    InfoSagGroup2("#I  initializing base\n" );
    maxDepth := CompositionLength( gens[ 1 ] ) - 1;

    # set up record system
    system  := rec(
    G       := G,
    base    := List( [1..maxDepth], x -> G.identity ),
    weights := List( [1..maxDepth], x -> false ),
    work    := List( [1..maxDepth], x -> List( [1..x], y -> true ) ),
    wf      := wf   );

    # run through gens
    for g  in gens  do
        S := Components( G, g );
        for s  in S  do
            G.operations.ModifyBase( system, s, wf.one( s ) );
        od;
    od;

    # return the information
    return system;

end;


#############################################################################
##
#F  AgGroupOps.EcheloniseBase( <system> ) . . . . . . . . . echelonise 'base'
##
##  'EcheloniseBase' modifies the initialized base  with all necessary powers
##  and commutators. In the record  <system> the entries 'base' and 'weights'
##  are changed.
##
AgGroupOps.EcheloniseBase := function ( system )

    local depth,        # actual depth of an element
          maxDepth,     # length of composition series
          nilp,         # actuall length of lower nilpotent series
          g,            # power/commutator of old generator
          s,            # component of g
          S,            # list of components of g
          wt,           # new weight of s 
          pos,          # position of s  in new basis 
          i, j, k, h;

    # set up variables
    InfoSagGroup2( "#I  echelonising base\n" );
    maxDepth := Length( system.base );
    nilp := 1;

    # run down lower nilpotent series
    h := 1;
    while h <= nilp+1 do

        # run through powers and commutators 
        i := 1;
        while i <= maxDepth  do
            j := 1;
            while j <= i  do
                if not (IsBool(system.weights[i]) or IsBool(system.weights[j]))
                   and system.wf.relevant( system.weights, i, j, h ) 
                   and system.work[ i ][ j ]  then
 
                    # set work flag new
                    if system.wf.useful( system.weights, i, j, h ) then
                        system.work[ i ][ j ] := false;
                    fi;

                    # modify with components of power or commutator
                    if i = j  then
                        g := system.base[ i ]^system.weights[ i ][ 3 ];
                    else
                        g := Comm( system.base[ i ], system.base[ j ] );
                    fi;
                    InfoSagGroup3( "#I  ", i, " ", system.weights[i], "  = i ",
                                   j, " ", system.weights[j], " = j\n" );
                    S := Components( system.G, g );
                    pos := maxDepth + 1;
                    for s  in S  do
                        wt := system.wf.weight( system.weights, i, j, h, s );
                        pos := Minimum( pos, 
                               system.G.operations.ModifyBase( system,s,wt ) ); 
                    od;
    
                    # if necessary, set indices new
                    if pos <= i  then
                        i := pos;
                        j := 0;
                    fi;
                fi;
                j := j+1;
            od;
            i := i+1;
        od;
        h := h+1;

        # set nilp
        for i in [1..maxDepth] do
            if not IsBool(system.weights[i]) then
                nilp := Maximum( nilp, system.weights[i][1] );
            fi;
        od;
    od;
    system.base := Filtered( system.base, x -> x <> system.G.identity );
    system.weights :=  Filtered( system.weights, x -> not IsBool( x ) );

    Unbind( system.work );
    Unbind( system.wf );

end;

#############################################################################
##
#F  AgGroupOps.GetLgLayer( <system> ) . . . . . . . . get layers of LG-series
##
##  'GetLgLayer' sorts the 'weights' lexicographically and it sorts the 'base'
##  in the corresponding order. Furthermore it calculates the lists 'layers',
##  'first', 'head' and 'tail' as decribed in  'SpecialAgGroup' and appends
##  them to the record <system>.
##
AgGroupOps.GetLgLayer := function( system )

    local perm,             # permutation
          weight,           # actual weight
          layer,            # actual layer
          layers,           # list of all layers
          nilpotlayer,      # layer of lower nilpotent series
          dealayer,         # layer of dea-series
          first,            # list that  indicates first base element of each
                            # layer
          head,             # list that  indicates first base element of each
                            # nilpotent layer
          tail,             # list that  indicates first base element of each 
                            # tail
          i;

    InfoSagGroup2( "#I  getting layers of LG-series\n" );

    # Sort weights and base
    perm        := Sortex( system.weights );
    system.base := Permuted( system.base, perm );

    # set up first and layers
    first  := [ ];
    layers := [ ];
    i := 1;
    layer := 0;
    while i <= Length( system.weights )  do
        weight := system.weights[ i ];
        layer := layer + 1;
        first[ layer ] := i;
        while i <= Length(system.weights) and weight = system.weights[i]  do
            layers[ i ] := layer;
            i := i + 1;
        od;
    od;
    Add( first, i );

    # set up head and tail
    head := [ ];
    tail := [ ];
    nilpotlayer := 0;
    dealayer := 0;
    for i  in [ 1..Length( first )-1 ]  do
        if system.weights[ first[i] ][ 1 ] <> nilpotlayer  then
            if dealayer = 1  then
                tail[ nilpotlayer ] := first[ i ];
            fi;
            nilpotlayer := nilpotlayer + 1;
            dealayer := 1;
            head[ nilpotlayer ] := first[ i ];
        elif system.weights[ first[i] ][2] <> dealayer  then
            dealayer := dealayer + 1;
            if dealayer = 2  then
                tail[ nilpotlayer ] := first[ i ];
            fi;
        fi;
    od;
    Add( head, first[ Length( first ) ] );
    if dealayer = 1  then
        Add( tail, first[ Length( first ) ] );
    fi;

    # update system
    system.layers  := layers;
    system.first   := first;
    system.head    := head;
    system.tail    := tail;

end;


#############################################################################
##
#F  AgGroupOps.CompositionBase ( <system> ) . . .  construct isomorphic group 
##
##  'CompositionBase' constructs the isomorphism to a group 'H' that has an ag
##  system in 'base' which refines the LG-series.  It adds to the record the 
##  'bijection' between the old group 'G' and the new group 'H'.
##
AgGroupOps.CompositionBase := function( system )

    local first,           # list containing the first element of each layer
          maxDepth,        # length of system.base
          series,          # list of groups
          N,               # group of series
          generators,      # generators of N
          alpha,           # isomorphism
          i;

    InfoSagGroup2("#I  constructing semispecial ag group");
    maxDepth := Length( system.base );

    # catch trivial case
    if ForAll( [1..maxDepth], x -> x = DepthAgWord( system.base[x] ) ) then
        InfoSagGroup2(" without isomorphism \n");
        system.H := Copy( system.G );
        system.base := Copy( Cgs( system.G ) );
        system.bijection := IdentityMapping( system.H );
        return;
    fi;
    InfoSagGroup2(" with isomorphism \n");

    # construct series
    series := [];
    for i  in [ 1..Length( system.first )-1 ]  do
        generators := Sublist( system.base, [ system.first[i]..maxDepth ] );
        Sort( generators, 
              function(a, b) return DepthAgWord(a) < DepthAgWord(b);
              end );
        N := system.G.operations.AgSubgroup(system.G, generators, false);
        Normalize ( N );
        Add ( series, N );
    od;
    Add ( series, system.G.operations.AgSubgroup(system.G, [], true));
    alpha := IsomorphismAgGroup ( series ) ;
    Unbind( system.G );

    # include information in system 
    system.H           := alpha.range;
    system.base        := Copy( Cgs ( alpha.range ) );
    system.bijection   := alpha;

end; 

#############################################################################
##
#F  AgGroupOps.LeastBadHallIndex( <system>, <index> ) . . . . least bad index
##
AgGroupOps.LeastBadHallIndex := function ( system, i )

    local U,               # group below the ith base element
          pj, pi,          # primes of j and i
          bad,             # index
          w,               # power/commutator
          exponents,       # list of exponents
          maxDepth,        # 
          j, k;

    maxDepth := Length( system.base );
    U := system.H.operations.AgSubgroup( system.H, 
                  system.base{[ i+1..maxDepth ]}, false );
 
    # get primes
    pi := system.weights[ i ][ 3 ];

    # run through powers/commutators and search for bad one
    bad := maxDepth + 1;
    for j  in [ i .. maxDepth ]  do
        if j = i  then
            w := system.base[ i ] ^ pi;
            pj := pi;
        else
            w := Comm( system.base[ j ], system.base[ i ] );
            pj := system.weights[ j ][ 3 ];
        fi;
        if w <> system.H.identity  then
            exponents := Exponents( U, w );
            k := 1;

            # run through exponent list until bad entry is found
            while k <= Length( exponents )  do

                # test primes
                if exponents[k] <> 0 and 
                   pi <> system.weights[k+i][3] and 
                   pj <> system.weights[k+i][3] 
                then
                    bad := Minimum( bad, k+i );
                    k := Length( exponents ) + 1;  
                else
                    k := k + 1;
                fi;
            od;
        fi;

        # if bad is minimal return; otherwise go on 
        if i = bad -1  then
            return bad;
        fi;
    od;
    return bad;
end;


#############################################################################
##
#F  AgGroupOps.LeastBadComplementIndex( <system>, <index> ) . least bad index
##
AgGroupOps.LeastBadComplementIndex := function ( system, i )

    local U,                 # composition subgroup
          maxDepth,          # composition length
          bad,               # least bad  index
          w,                 # commutator
          exponents,         # exponent vector of w  in layer
          p,                 # important prime
          j, k, h;

    maxDepth := Length( system.base );
    bad      := maxDepth + 1;
    p        := system.weights[i][3];
    U        := system.H.operations.AgSubgroup( system.H, 
                system.base{[i+1..maxDepth]}, false );

    for j in [system.head[system.weights[i][1]]..maxDepth] do
        if system.weights[j][3] <> p then
            w := Comm( system.base[j], system.base[i] );
            if w <> system.H.identity   then
                exponents := Exponents( U, w );
                k := 1;

                # run through exponent list until bad entry is found
                while k <= Length( exponents )  do
                    if exponents[k] <> 0 and 
                       system.weights[i+k][1] = system.weights[j][1] + 1 and  
                       system.weights[i+k][2] = 1  and
                       system.weights[i+k][3] = p  then
                        if i+k < bad  then
                            bad := i+k;
                        fi;
                        k := Length( exponents ) + 1;
                    else
                        k := k + 1;
                    fi;
                od;
            fi;
        fi;

        ## if bad is minimal return; otherwise go on
        if i = bad - 1  then
            return bad;
        fi;
    od;
    return bad;
end;

#############################################################################
##
#F  AgGroupOps.ChangeBase( <system>, <i>, <flag> ) change <i>.th base element
##
##  'ChangeBase'   runs through the  base elements  below  the <i>-th one and
##  corrects the <i>-th element of 'base' for  all base elements which have a
##  bad index. 
##
AgGroupOps.ChangeBase := function ( system, i, flag )

    local k,                 # first bad  index
          layer,             # layer with bad  index
          first,             # first element of this layer
          next,              # first element of next layer
          size,              # size of layer
          head, tail,        # for complements
          maxDepth,          # composition length
          N, M,              # subgroups of system.H such that N/M is layer
          gensNM, NM,        # N/M and generators
          U,                 # composition subgroup below system.base[ i ]
          gensUN,            # generators of U/N
          A,                 # operation on layer
          ai, aij,           # operating elements
          B, v,              # one equation system
          E, V,              # simultaneuos linear equation system
          F,                 # enlarged simultaneuos linear system
          pi, pj, pk,        # involved primes
          g,                 # power/commutator
          solution,          # one solution of simultaneuos system or false
          gens,              # relevant generators
          I,                 # idmat
          j, l, h; 

    maxDepth := Length( system.base );

    # get in the case that flag indicates
    if flag = "head"  then
        k := system.H.operations.LeastBadComplementIndex( system, i );
        InfoSagGroup2( "#I  change complement base: ");
    elif flag = "hall"  then
        k := system.H.operations.LeastBadHallIndex( system, i );
        InfoSagGroup2( "#I  change hall base: ");
    fi;

    # trivial case
    if k > Length( system.base )  then
        InfoSagGroup2( i, " has no bad index\n" );
        return i;
    fi;
    InfoSagGroup2( i, " has bad index = ", k, "\n" );

    # composition subgroup
    U := system.H.operations.AgSubgroup(system.H, 
                  system.base{[i+1..maxDepth]}, false );

    # get the layer
    layer := system.layers[ k ];
    first := system.first[ layer ];
    next  := system.first[ layer + 1 ];
    size  := next - first;

    # get factor group of this layer
    N := system.H.operations.AgSubgroup(system.H, 
                  system.base{[first..maxDepth]}, false );
    M := system.H.operations.AgSubgroup(system.H, 
                  system.base{[next ..maxDepth]}, false );
    NM := N mod M;
    gensNM := NM.generators;

    # InitializeSystem inhomogenous system  
    V := [];
    E := List([1..size], x -> []);

    # get primes
    pi := system.weights[ i ][ 3 ];
    pk := system.weights[ k ][ 3 ];

    if  flag = "hall"  then
        gens := system.base{ 
                Filtered( [i+1..first-1], x -> system.weights[x][3] <> pk ) };

        # and we have to add the power
        g := system.base[ i ] ^ pi;

        # exponent vector of g  in NM
        v := Sublist( Exponents( U, g, GF(pk) ), [first-i..next-i-1] );
 
        # set up matrix
        A := List( gensNM, x -> Exponents( NM, x^system.base[i], GF(pk) ) );
        I := A ^ 0;
        B := I;
        for l  in [ 1..pi-1 ]  do
            B := B * A + I;
        od;
        B := - B;

        # append to system
        for l  in [ 1..size ]  do
            Append( E[ l ], B[ l ] );
        od;
        Append( V, v );

    else

        # pic the p'-generators in the head above
        head := system.head[system.weights[k][1]-1];
        tail := system.tail[system.weights[k][1]-1];
        gens := system.base{ Filtered( [ head .. tail-1 ], x ->
                system.weights[x][3] <> pi ) };
    fi;

    # run through commutators 
    for h  in  gens  do
        g := Comm( h, system.base[ i ] );

        # exponent vector of g  in NM
        v := Sublist( Exponents( U, g, GF(pk) ), [first-i..next-1-i] );

        # corresponding matrix
        aij := h ^ system.base[ i ];
        A := List( gensNM, x -> Exponents( NM, x^aij, GF(pk) ) );
        B := A - A ^ 0;

        # append to system
        for l  in [ 1..size ]  do
            Append( E[ l ], B[ l ] );
        od;
        Append( V, v );
    od;

    # try to solve inhomogenous systems simultaneously
    solution := SolutionMat( E, V );
    if IsBool( solution )  then
        Error("cannot find solution \n");
    fi;

    # calculate new i-th base element
    ai := system.base[ i ];
    for j  in [ 1..Length( gensNM ) ]  do
        ai := ai * gensNM[ j ] ^ Int( solution[ j ] );
    od;
    system.base[ i ] := ai;

    # and start recursion
    system.H.operations.ChangeBase( system, i, flag );
end;


#############################################################################
##
#V  SagGroupOps . . . . . . . . . . . operations record for special ag groups
##
SagGroupOps := OperationsRecord( "SagGroupOps", AgGroupOps );


#############################################################################
##
#F  SpecialAgGroup( <G>, <flag> ) . . . . . . . . compute a special ag system
##
##  'SpecialAgGroup' returns   an isomorphic ag  group to  <G>   which has a
##  Leedham-Green series  refined by the  ag system,  exhibited Hall subgroups
##  and exhibited head-complements. The group record  has additionally the 
##  following entries.
##
##  'weights'	    a list of LG-weights of the gens of the output group
##  'layers'	    a list,  giving the number of the layer in the  LG-series
##                  of the corresponding generator
##  'first'	    a list,  giving the number of the first gen in a layer
##  'head'          a list,  giving the number of the first gen in a head
##  'tail'          a list,  giving the number of the first gen in a tail
##  'bijection'     the isomorphism from the output group to <G>
##
##  if <flag> = "noPublic", then only a semispecial ag system is calculated.
##  if <flag> = "noHall", then no public Hall groups are calculated.
##  if <flag> = "noHead", then no public head-complements are calculated.
##
SpecialAgGroup := function( arg )
    local  G,  H,  K,  system,  wf,  i,  alpha;
            
    # check if a special ag group is already known
    G := arg[ 1 ];
    if IsBound(G.sagGroup)  then
        return G.sagGroup;
    fi;

    # get trivial case 
    if 0 = Length(Cgs(G))  then
      H                    := Copy(G);
      H.weights            := [];
      H.layers             := [];
      H.first              := [];
      H.head               := [];
      H.tail               := [];
      H.isHallSystem       := true;
      H.isHeadSystem       := true;
      H.isNormalizerSystem := true;
      H.bijection          := GroupHomomorphismByImages( G, H, [], [] );
      H.operations         := SagGroupOps;
      G.sagGroup           := H;
      return H;
    fi;

    # use Leedham-Green weights
    wf := SagWeights;

    # construct LG-series
    InfoSagGroup2("#I  constructing LG-series and semispecial ag system\n");

    system := G.operations.InitializeSystem( G, G.cgs, wf );
    G.operations.EcheloniseBase( system );
    G.operations.GetLgLayer( system );

    # compute the isomorphic semispecial ag group
    G.operations.CompositionBase( system );

    # construct exhibited subgroups
    if Length( arg ) = 1 or not arg[2] = "noPublic"  then
        InfoSagGroup2("#I  constructing exhibited subgroups\n");
    else
        system.base := system.H.generators;
    fi;
    system.isHallSystem := false;
    system.isHeadSystem := false;

    # compute Hall system
    if Length( arg ) = 1  or  arg[ 2 ] = "noHead"  then
        i := system.first[ Length( system.first ) -1 ]-1;
        while i >= 1  do
            system.H.operations.ChangeBase( system, i, "hall" );
            i := i - 1;
        od;
        system.isHallSystem       := true;
    fi;

    # compute head complements
    if Length( arg ) = 1  or  arg[ 2 ] = "noHall"  then
        i := system.head[ Length( system.head ) - 1 ] - 1;
        while i >= 1  do
            system.H.operations.ChangeBase( system, i, "head" );
            i := i - 1;
        od;
        system.isHeadSystem       := true;
    fi;

    if system.H.generators <> system.base  then
        InfoSagGroup2("#I  computing the new group and isomorphism\n");

        # compute a new group <K>
        H     := Subgroup( system.H, system.base );
        H.igs := system.base;
        H.operations.AddShiftInfo(H);
        K := AgGroupFpGroup( FpGroup ( H ) );

        # compute isomorphism
        alpha := GroupHomomorphismByImages( H, K, Igs(H), K.generators );
        K.bijection := InverseMapping( system.bijection * alpha );
    else
        K := system.H;
        K.bijection := InverseMapping( system.bijection );
        Unbind( K.normalized );
    fi;

    # change ops entry
    K.operations         := AgGroupOps;
    K.isHallSystem       := system.isHallSystem;
    K.isHeadSystem       := system.isHeadSystem;
    K.cgs                := K.generators;

    # and update information
    K.weights := system.weights;
    K.layers  := system.layers;
    K.first   := system.first;
    K.head    := system.head;
    K.tail    := system.tail;
    
    # store information
    if K.isHallSystem and K.isHeadSystem  then
        G.sagGroup   := K;
        K.isSagGroup := true;
        K.operations := SagGroupOps;
    fi;

    # and return <K>
    return K;
end;


#############################################################################
##
#V  Read  . . . . . . . . . . . . . . . . . read other special ag group stuff
##
ReadLib("sagsbgrp");
