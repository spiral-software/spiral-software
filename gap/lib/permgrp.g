# -*- Mode: shell-script -*- 

#############################################################################
##
#A  permgrp.g                   GAP library                         Udo Polis
#A                                                         & Martin Schoenert
##
##
#Y  Copyright (C) 2018-2021, Carnegie Mellon University
#Y  All rights reserved.  See LICENSE for details.
#Y  
#Y  This work is based on GAP version 3, with some files from version 4.  GAP is
#Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
##
##  This file contains the basic functions that work with permutation groups.
##  There  are  functions to compute   a stabilizer chain  for  a permutation
##  group, to change to a stabilizer chain for  a  different base, and simple
##  functions that use stabilizer chain, e.g., 'Size' and 'in'.
##
##


#############################################################################
##
#F  InfoPermGroup1( ... ) . .  information function for the permgroup package
#F  InfoPermGroup2( ... ) . .  information function for the permgroup package
##
if not IsBound( InfoPermGroup1 )  then InfoPermGroup1 := Ignore;  fi;
if not IsBound( InfoPermGroup2 )  then InfoPermGroup2 := Ignore;  fi;


#############################################################################
##
#F  IsPermGroup(<D>)  . . . . . . . . . . . . is a domain a permutation group
##
IsPermGroup := function ( D )
    return IsRec( D )
       and IsBound( D.isPermGroup )  and D.isPermGroup;
end;


#############################################################################
##
#F  PermutationsOps.Group(<D>,<gens>,<id>)  . . .  create a permutation group
##
PermutationsOps.Group := function ( Permutations, gens, id )
    local   G;          # permutation group <G>, result

    # let the default function do the main work
    G := GroupElementsOps.Group( Permutations, gens, id );

    # add the permutation group tag
    G.isPermGroup       := true;

    # add known information
    G.isFinite          := true;

    # add the operations record
    G.operations        := PermGroupOps;

    # return the permuation group
    return G;
end;


#############################################################################
##
#V  PermGroupOps  . . . . . . operation record for permutation group category
##
##  'PermGroupOps'  is  the  operation  record for   permutation  groups.  It
##  contains   the  function  for   domain  operation,    e.g.,    'Size' and
##  'Intersection' as well  as  the functions  for group    operations, e.g.,
##  'Centralizer' and 'Orbit'.
##
##  'PermGroupOps' is initially a copy of 'GroupOps', thus permutation groups
##  inherit the default group functions, e.g., 'DerivedSubgroup' and 'Index'.
##  However  'PermGroupOps'   overlays  some of  those    functions with more
##  efficient ones, e.g., 'Elements' and 'Size'.
##
PermGroupOps := Copy( GroupOps );


#############################################################################
##
#F  PermGroupOps.Subgroup(<G>,<gens>) . . . . . . make a permutation subgroup
##
PermGroupOps.Subgroup := function ( G, gens )
    local   S;          # subgroup, result

    # let the default function do the main work
    S := GroupOps.Subgroup( G, gens );

    # add permgroup tag and permgroup operations record
    if IsBound( S.parent )  then
        S.isPermGroup       := true;
        S.operations        := PermGroupOps;
    fi;

    # return the subgroup
    return S;
end;

#############################################################################
##
#F  PermGroupOps.AsSubgroup(<G>,<U>) . . . . . . make a permutation subgroup
##
PermGroupOps.AsSubgroup := function ( G, U )
    local   S;          # subgroup, result

    # let the default function do the main work
    S := PermGroupOps.Subgroup( G, U.generators );

    # if U has a stabChain, keep it
    if IsBound( U.stabChain) then 
       S.stabChain := Copy( U.stabChain );
       if IsBound( S.stabChain.orbit ) then
          S.orbit := S.stabChain.orbit;
          S.transversal := S.stabChain.transversal;
          S.stabilizer := S.stabChain.stabilizer;
       fi;
    #N akos 6/25/94 whose randomness info should we keep?
    fi; 

    return S;
end;

############################################################################
#
#F  PermGroupOps.Closure(<G>,<obj>[,options]) . . . . .  closure < <G>, <g> >
##
PermGroupOps.Closure := function ( arg )
    local   G,          # the group to close
            obj,        # element(s) to close with
            C,          # closure of < <G>, <obj> >, result
            gens,       # generators of <obj> that are not in <G>
            g,          # an element of gens
            chain,      # the stabilizer chain created
            options;    # options record for closure

    G := arg[1];
    obj := arg[2];
    if Length(arg) = 3 then
       options := arg[3];
    else 
       options := rec ();
    fi;

    # get the elements that must be added to <G>
    if IsGroup( obj )  then
        gens := obj.generators;
    elif IsList(obj) then
        gens := obj;
    else
        gens := [ obj ];
    fi;

    # handle a closure in the parent
    if IsParent( G )  then
        C := G;

    # handle the closure with a group that has a stabilizer chain
    elif IsBound( G.stabChain )  then

        # if all generators are in <G>, <G> is the closure
        gens := Filtered( gens, gen -> not G.operations.\in( gen, G ) );
        if Length(gens) = 0  then
            C := G;

        # otherwise decide between random and deterministic methods
        else 
           if not IsBound(options.random) then 
              if IsBound(Parent(G).stabChainOptions) 
                 and IsBound(Parent(G).stabChainOptions.random) then
                 options.random := Parent(G).stabChainOptions.random;
              elif IsBound( G.stabChainOptions )
                 and IsBound( G.stabChainOptions.random) then
                 options.random := G.stabChainOptions.random;
              elif IsBound(StabChainOptions.random) and 
                 PermGroupOps.LargestMovedPoint
                  ( rec( generators:=Union(G.generators,gens) ) ) > 100 then
                 options.random := StabChainOptions.random;
              else 
                 options.random := 1000;
              fi;
           fi;
           if options.random = true then
              options.random := 900;
           elif  options.random = false then 
              options.random := 1000;
           fi;

           # add info to options record
           if IsBound(options.size) then
              options.limit := options.size;
           fi;
           if IsBound(Parent(G).stabChain) then 
              if not IsBound(options.knownBase) then
                 options.knownBase := Base(Parent(G));
              fi;
              if not IsBound(options.limit) then 
                 options.limit := Size(Parent(G));
              fi;
           fi;
           # make the base of G compatible with options.base
           if IsBound(options.base) then
              MakeStabChain(G,options.base);
           fi;

           if options.random < 1000 then
              chain := PermGroupOps.ClosureRandom
                            ( Copy(G.stabChain), gens, options );
           else
              chain := Copy( G.stabChain);
              for g in gens do
                  if not G.operations.\in( g, chain) then
##                     chain := 
                               G.operations.StabChainStrong
                           ( chain, [g], G.operations.Base( chain ));
                  fi;
              od;
           fi;

           C := Subgroup( Parent(G), ShallowCopy(chain.generators) );
           C.stabChainOptions := rec( random := options.random );
           C.stabChain := chain;
           # for compatibility 
           C.orbit    := C.stabChain.orbit;
           C.transversal   := C.stabChain.transversal;
           C.stabilizer := C.stabChain.stabilizer;
        fi;

    # handle the closure with a group that has no stabilizer chain
    else

        # if all generators are in <G>, <G> is the closure
        gens := Filtered( gens, gen ->     not gen    in G.generators
                                       and not gen^-1 in G.generators
                                       and not (IsBound( G.elements )
                                                and gen in G.elements) );
        if Length(gens) = 0  then
            C := G;

        # otherwise make the closure group
        else
            C := G.operations.Subgroup( Parent( G ),
                                       Concatenation( G.generators, gens ) );

        fi;

    fi;

    # return the closure
    return C;
end;


##############################################################################
##
#F  PermGroupOps.ClosureRandom(<G>,genlist,options) . . . closure of G,genlist
##
PermGroupOps.ClosureRandom := function ( G,genlist,options )
    local  k,          # number of pairs of subproducts of generators in 
                       # testing result
           givenbase,  # ordering from which initial base points should
                       # be chosen
           gens,       # generators in genlist that are not in <G>
           g,          # element of gens
           degree,     # degree of closure
           orbits,     # list of orbits of closure
           orbits2,    # list of orbits of closure
           i,j,        # loop variables
           param,       # list of parameters guiding number of repetitions
                        # in random constructions
           where,       # list indicating which orbit contains points in domain
           basesize,    # list; i^th entry = number of base points in orbits[i]
           ready,       # boolean; true if stabilizer chain ready
           new,         # list of permutations to be added to stab. chain
           result,      # output of checking phase; nontrivial if stabilizer
                        # chain is incorrect
           base,        # ordering of domain from which base points are taken
           missing,     # if a correct base was provided by input, missing
                       # contains those points of it which are not in 
                       # constructed base
           correct;     # boolean; true if a correct base is given

# warning:  options.base should be compatible with Base(G)


    gens := Filtered( genlist, gen -> SCRSift(G,gen) <> () );
    if Length(gens) > 0  then
        
        G.identity := () ;
        k := First([0..14],x->(3/5)^x <= 1-options.random/1000);
        if IsBound(options.knownBase) then 
            param := [k,4,0,0,0,0];
        else
            param := [QuoInt(k,2),4,QuoInt(k+1,2),4,50,5];
        fi;
        if options.random <= 200 then 
            param[2] := 2;
            param[4] := 2;
        fi;
        
#param[1] = number of pairs of random subproducts from generators in 
#           first checking phase
#param[2] = (number of random elements from created set)/S.diam
#           in first checking phase 
#param[3] = number of pairs of random subproducts from generators in 
#           second checking phase
#param[4] = (number of random elements from created set)/S.diam
#           in second checking phase 
#param[5] = maximum size of orbits in  which we evaluate words on all
#           points of orbit
#param[6] = minimum number of random points from orbit to plug in to check 
#           whether given word is identity on orbit

        degree := PermGroupOps.LargestMovedPoint(
                          rec(generators:=Union(G.generators,gens)));


        # prepare input of construction
        if IsBound(options.base) then 
            givenbase := options.base;
        else
            givenbase := [];
        fi;
        
        if IsBound(options.knownBase) then
            correct := true;
        else
            correct := false;
        fi;
        
        if correct then 
            # if correct  base was given as input,
            # no need for orbit information
            base := Set( givenbase );
            for i in PermGroupOps.Base(G) do
                if not i in base then
                    Add( givenbase, i );
                fi;
            od;
            base := Concatenation(givenbase,Difference(options.knownBase,
                                                       givenbase));
            missing := Difference(options.knownBase,PermGroupOps.Base(G));
            basesize := []; 
            where := [];
            orbits := [];
        else 
            # create ordering of domain used in choosing base points and
            # compute orbit information
            base := Set( givenbase );
            for i in PermGroupOps.Base(G) do
                if not i in base then
                    Add( givenbase, i );
                fi;
            od;
            base := Concatenation(givenbase,Difference([1..degree],givenbase));
            missing := [];
            orbits2 := PermGroupOps.Orbits(rec(generators:=Union(G.generators,
                               gens)),[1..degree],OnPoints);
            #throw away one-element orbits
            orbits:=[];
            j:=0;
            for i in [1..Length(orbits2)] do 
                if Length(orbits2[i]) >1 then
                    j:=j+1; orbits[j]:= orbits2[i];
                fi;
            od;
            basesize:=[]; 
            where:=[];
            for i in [1..Length(orbits)] do 
                basesize[i]:=0;
                for j in [1..Length(orbits[i])] do
                    where[orbits[i][j]]:=i;
                od;
            od;
            # temporary solution to speed up of handling
            # of lots of small orbits until compiler
            if Length(orbits) > degree/40 then
                param[1] := 0;
                param[3] := k;
            fi;
        fi;
        
        if not IsBound(G.aux) then
            SCRExtendRecord(G);
        fi;
        new := gens;
       
        #the first call of SCRMakeStabStrong has top:=false
        #in order to add gens to the generating set of G;
        #further calls have top:=true, in order not to add
        #output of SCRStrongGenTest to generating set.
        #remark: adding gens to the generating set of G before 
        #calling SCRMakeStabStrong gives a nasty error if first base
        #point changes
        for g in gens do
            if SCRSift(G,g) <> () then 
                SCRMakeStabStrong (G,[g],param,orbits,
                        where,basesize,base,correct,missing,false);
            fi;
        od;
        if    IsBound(options.limit)
          and PermGroupOps.Size(G)=options.limit 
        then
            ready := true;
        else 
            Unbind(G.size);
            result:=SCRStrongGenTest
                    (G,param,orbits,basesize,base,correct,missing);
            if result <> () then
                new := [result];
                ready := false;
            elif correct then
                ready := true;
            else 
                result := SCRStrongGenTest2(G,param);
                if result = () then
                    ready := true;
                else 
                    new := [result];
                    ready := false;
                fi;
            fi;
        fi;
        
        while not ready do 
            SCRMakeStabStrong
              (G,new,param,orbits,where,basesize,base,correct,missing,true);
            if    IsBound(options.limit)
              and PermGroupOps.Size(G)=options.limit
            then
                ready := true;
            else
                Unbind(G.size);
                result:=SCRStrongGenTest
                        (G,param,orbits,basesize,base,correct,missing);
                if result <> () then
                    new := [result];
                    ready := false;
                elif correct then
                    ready := true;
                else
                    result := SCRStrongGenTest2(G,param);
                    if result = () then
                        ready := true;
                    else
                        new := [result];
                        ready := false;
                    fi;
                fi;
            fi;
        od;
        
        # if not IsBound(options.temp) or options.temp = false then
        SCRRestoreRecord(G);
        # fi;
        Unbind(G.base);
        Unbind(G.size);
        
    fi; # if Length(gens) > 0
    
    
    # return the closure
    return G;

end;


#############################################################################
##
#F  PermGroupOps.CommutatorSubgroup( <U>,<V> )  . .  commutator subgrp of U,V 
##
PermGroupOps.CommutatorSubgroup := function ( U, V )
    local   C,       # the commutator subgroup
            CUV,     # closure of U,V
            doneCUV, # boolean; true if CUV is computed
            u,       # random subproduct of U.generators
            v,       # random subproduct of V.generators 
            comm,    # commutator of u,v
            list,    # list of commutators
            i,j;     # loop variables

    # [ <U>, <V> ] = normal closure of < [ <u>, <v> ] >.
    C := TrivialSubgroup( Parent(U) );
    StabChain(C);
    doneCUV := false;
    # if there are lot of generators, use random subproducts
    if Length(U.generators)*Length(V.generators) > 10 then
        repeat
           list := [];
           for i in [1..10] do
               u := SCRRandomSubproduct(U.generators); 
               v := SCRRandomSubproduct(V.generators);
               comm := Comm( u, v );
               if not comm in C then
                   Add( list, comm ) ;
               fi;
           od;
           if Length(list) > 0 then
               C := PermGroupOps.Closure( C,list );
               if not doneCUV then
                   CUV := Closure( U, V );
                   doneCUV := true;
               fi;
               C := PermGroupOps.NormalClosure(  CUV , C );
           fi;
        until list = [];
    fi;

    # do the deterministic method; it will also check correctness 
    list := [];
    for i in [1..Length(U.generators)] do
        for j in [1..Length(V.generators)] do
           comm := Comm(U.generators[i],V.generators[j]);
           if not comm in C then 
               Add( list, comm );
           fi;
        od;
    od;
    if Length(list) > 0 then 
        C := PermGroupOps.Closure( C,list );
        if not doneCUV then
           CUV := Closure( U, V );
           doneCUV := true;
        fi;
        C := PermGroupOps.NormalClosure(  CUV , C );
    fi;

    return C;

end;

###############################################################################
##
#F  PermGroupOps.DerivedSubgroup(<G>) . . . . . . . . . derived subgroup of G
##
PermGroupOps.DerivedSubgroup := function ( G )
    local   D,          # derived subgroup of <G>, result
            g, h,       # random subproducts of generators 
            comm,       # their commutator
            list,       # list of commutators
            count,i,j;  # loop variables

    # find the subgroup generated by the commutators of the generators
    D := TrivialSubgroup( Parent(G) ); 
    StabChain(D);

    # if there are >4 generators, use random subproducts
    if Length(G.generators) > 4 then 
        repeat 
            list := [];
            for i in [1..10] do
                g := SCRRandomSubproduct(G.generators);
                h := SCRRandomSubproduct(G.generators);
                comm := Comm( g, h );
                if not comm in D then  
                   Add( list, comm );
                fi;
            od;
            if Length(list) > 0 then 
               D := PermGroupOps.Closure(D,list);
               D := PermGroupOps.NormalClosure( G, D );
            fi;
        until list = [];
    fi;

    # do the deterministic method; it will also check random result
    list := [];
    for i in [2..Length(G.generators)] do
         for j  in [ 1 .. i - 1 ]  do
             comm := Comm( G.generators[i], G.generators[j] );
             if not comm in D then 
                 Add( list, comm );
             fi;
         od;
    od;
    if Length(list) > 0 then
        D := PermGroupOps.Closure(D,list);
        D := PermGroupOps.NormalClosure( G, D );
    fi;

    return D;
end;

###############################################################################
##
#F  PermGroupOps.NormalClosure(<G>,<U>) . . . . . normal closure of <U> in <G>
##
PermGroupOps.NormalClosure := function ( G, U )
    local   N,          # normal closure of <U> in <G>, result
            g,          # one random subproduct generator of the group <G>
            cnj,        # conjugate to be added to <N>
            list,       # list of conjugates to be added
            i;          # loop variable

    # handle trivial case
    if Length(U.generators) = 0 then
       return U;
    fi;

    N := U;
    # make list of conjugates to be added to N
    repeat 
        list := [];
        for i in [1..10] do 
            g   := SCRRandomSubproduct(G.generators);
            cnj := SCRRandomSubproduct(Concatenation(N.generators,list))^g;
            if not (cnj in N) then 
               Add(list,cnj);
            fi;
        od;
        if Length(list) > 0 then
           N := PermGroupOps.Closure( N, list );
        fi;
    until list = [];

    # use deterministic method to check that we got normal subgroup

    N := GroupOps.NormalClosure(G,N);
    # return the normal closure
    return N;

end;

#############################################################################
##
#F  PermGroupOps.Elements(<G>)  . . . . . . . elements of a permutation group
##
PermGroupOps.Elements := function ( G )
    local   elms;               # element set, result

    # make sure that <G> has a stabchain
    if not IsBound(G.stabilizer)  then MakeStabChain(G); fi;

    # compute the elements of <G>
    elms := PermGroupOps.ElementsStab( G );

    # return the elements
    return elms;
end;

PermGroupOps.ElementsStab := function ( G )
    local   elms,               # element list, result
            stb,                # elements of the stabilizer
            pnt,                # point in the orbit of <G>
            rep;                # inverse representative for that point

    # if <G> is trivial then it is easy
    if Length(G.generators) = 0  then
        elms := [ G.identity ];

    # otherwise
    else

        # start with the empty list
        elms := [];

        # compute the elements of the stabilizer
        stb := PermGroupOps.ElementsStab( G.stabilizer );

        # loop over all points in the orbit
        for pnt  in G.orbit  do

           # add the corresponding coset to the set of elements
           rep := G.identity;
           while G.orbit[1] ^ rep <> pnt  do
                rep := LeftQuotient( G.transversal[pnt/rep], rep );
           od;
           UniteSet( elms, stb * rep );

        od;

   fi;

   # return the result
   return elms;
end;


#############################################################################
##
#F  PermGroupOps.IsFinite(<P>)  . . . . test if a permutation group is finite
##
PermGroupOps.IsFinite := function ( G )
    return true;
end;


#############################################################################
##
#F  PermGroupOps.Size(<G>)  . . . . . . . . . . . size of a permutation group
##
PermGroupOps.Size := function ( G )
    local   S,          # stabilizer of <G>
            size;       # size of <G>, result

    # make sure that <G> has a stabchain
    if not IsBound(G.stabilizer)  then MakeStabChain( G );  fi;

    # go down the stabchain and multiply the orbitlengths
    size := 1;
    S := G;
    while Length(S.generators) <> 0  do
        size := size * Length( S.orbit );
        S := S.stabilizer;
    od;

    # return the size
    return size;
end;


#############################################################################
##
#F  PermGroupOps.\in(<g>,<G>)  . . . . membership test for permutation group
##
PermGroupOps.\in := function ( g, G )
    local   S,          # stabilizer of <G>
            bpt;        # basepoint of <S>

    # make sure that we can proceed with the rest
    if not IsPerm( g )
        and (not IsRec(g)  or not IsBound(g.isPerm)  or g.isPerm <> true)
    then
        return false;
    fi;

    # handle special cases
    if   g in G.generators  then
        return true;
    elif g = G.identity then
        return true;
    elif 0 = Length(G.generators)  then
    	return false;
    fi;

    # make sure that <G> has a stabchain
    if not IsBound(G.stabilizer)  then MakeStabChain( G );  fi;

    # go down the stabchain and reduce the permutation
    S := G;
    while Length(S.generators) <> 0  do
        bpt := S.orbit[1];

        # if '<bpt>^<g>' is not in the orbit then <g> is not in <G>
        if not IsBound(S.transversal[bpt^g])  then
            return false;
        fi;

        # reduce <g> into the stabilizer
        while bpt ^ g <> bpt  do
            g := g * S.transversal[bpt^g];
        od;

        # and test if the reduced <g> lies in the stabilizer
        S := S.stabilizer;
    od;

    # <g> is in the trivial iff <g> is the identity
    return g = G.identity;
end;


#############################################################################
##
#F  PermGroupOps.Random(<G>)  . . . . . random element of a permutation group
##
PermGroupOps.Random := function ( G )
    local   S,          # stabilizer of <G>
            rnd,        # random element of <G>, result
            pnt;        # random point in <S>.orbit

    # make sure that <G> has a stabchain
    if not IsBound(G.stabilizer)  then MakeStabChain( G );  fi;

    # go down the stabchain and multiply random representatives
    rnd := G.identity;
    S := G;
    while Length(S.generators) <> 0  do
        pnt := RandomList(S.orbit) ^ rnd;
        while S.orbit[1]^rnd <> pnt  do
            rnd := LeftQuotient( S.transversal[pnt/rnd], rnd );
        od;
        S := S.stabilizer;
    od;

    # return the random element
    return rnd;
end;


#############################################################################
##
#F  PermGroupOps.Order(<G>,<g>) . . . . . . . . . . .  order of a permutation
##
PermGroupOps.Order := function ( G, g )
    return OrderPerm( g );
end;


#############################################################################
##
#F  PermGroupOps.ConjugacyClass(<G>,<g>)  . . . conjugacy class of an element
#F                                                     in a permutation group
#V  ConjugacyClassPermGroupOps  . . .  operation record for conjugacy classes
#V                                                     in a permutation group
##
##  Conjugacy  classes in  permutation  groups  are   almost like   conjugacy
##  classes  in  generic groups,  except that  'Representative'   accepts the
##  centralizer of the second element as optional parameter.
##
PermGroupOps.ConjugacyClass := function ( G, g )
    local   C;

    # make the domain
    C := GroupOps.ConjugacyClass( G, g );

    # enter the operations record
    C.operations := ConjugacyClassPermGroupOps;

    # return the conjugacy class
    return C;
end;

ConjugacyClassPermGroupOps := Copy( ConjugacyClassGroupOps );

ConjugacyClassPermGroupOps.\= := function ( C, D )
    local    isEql;
    if    IsRec( C )  and IsBound( C.isConjugacyClass )
      and IsRec( D )  and IsBound( D.isConjugacyClass )
      and C.group = D.group
    then
        if not IsBound( C.centralizer )  then
            C.centralizer := Centralizer( C.group, C.representative );
        fi;
        isEql := Size(C) = Size(D)
             and Order( C.group, C.representative )
               = Order( D.group, D.representative )
             and C.group.operations.RepresentativeConjugationElements(
                        C.group,
                        D.representative,
                        C.representative,
                        C.centralizer ) <> false;
    else
        isEql := DomainOps.\=( C, D );
    fi;
    return isEql;
end;

ConjugacyClassPermGroupOps.\in := function ( g, C )
    if IsBound( C.elements ) then
      return g in C.elements;
    else
      if not IsBound( C.centralizer )  then
	  C.centralizer := Centralizer( C.group, C.representative );
      fi;
      return g in C.group
	 and Order( C.group, g ) = Order( C.group, C.representative )
	 and C.group.operations.RepresentativeConjugationElements(
		  C.group,
		  g,
		  C.representative,
		  C.centralizer ) <> false;
    fi;
end;


#############################################################################
##
#F  PermGroupOps.ConjugacyClasses( <G> )  . . . . . . . . . conjugacy classes
##
PermGroupOps.ConjugacyClasses := function( G )
    local  classes,  cl;

    # in certain cases, we prefer the old (random) method
    if     not IsBound(G.rationalClasses)
       and ( Size(G) <= 1000 or IsBound(G.elements) or IsSimple(G) )
    then
    	return GroupOps.ConjugacyClasses(G);

     # otherwise compute the rational classes and split them
     else
	classes := [];
	for cl  in RationalClasses(G)  do
	    Append( classes, cl.operations.Decomposed(cl) );
	od;
	InfoPermGroup1( "#I  ", Collected( List( classes, 
	              cl -> Order(G,cl.representative) ) ), "\n" );
	return classes;
    fi;
end;


#############################################################################
##
#F  PermGroupOps.RationalClasses( <G> ) . .  rational classes for perm groups
##
PermGroupOps.RationalClasses := function( G )
    return RationalClassesPermGroup( G );
end;

#############################################################################
##
#F  PermGroupOps.ConjugateSubgroup(<G>,<g>)  conjugate of a permutation group
##
PermGroupOps.ConjugateSubgroup := function ( G, g )
    local   H,          # conjugated subgroup, result
            S,          # stabilizer of <G>
            T,          # stabilizer of <H>
            str,        # strong generators of <G>
            cnj,        # conjugated generators
            i;          # loop variable

    # first conjugate the generators (and the element list if present)
    H := GroupOps.ConjugateSubgroup( G, g );

    # now conjugate the stabchain if present
    if IsBound( G.stabilizer )  and not IsBound( H.stabilizer )  then

        str := Concatenation( [ G.identity ], G.generators );
        cnj := Concatenation( [ H.identity ], H.generators );

        # go down the stabchain and conjugate every stabilizer
        S := G;
        T := H;
        while Length(S.generators) <> 0  do

            # conjugate the generators of this stabilizer
            T.generators := [];
            for i  in [1..Length(S.generators)]  do
                if not S.generators[i] in str  then
                    Add( str, S.generators[i] );
                    Add( cnj, S.generators[i] ^ g );
                fi;
                T.generators[i]:=cnj[Position(str,S.generators[i])];
            od;

            # conjugate the orbit and the transversal of this stabilizer
            T.orbit       := [];
            T.transversal := [];
            for i  in [1..Length(S.orbit)]  do
                T.orbit[i]                := S.orbit[i] ^ g;
                T.transversal[T.orbit[i]] :=
                        cnj[ Position(str,S.transversal[S.orbit[i]]) ];
            od;

            # make a new stabilizer
            T.stabilizer := Group( [], T.identity );

            # on to the next stabilizer
            S := S.stabilizer;
            T := T.stabilizer;
        od;

    fi;

    # return the conjugated subgroup
    return H;
end;


#############################################################################
##
#F  PermGroupOps.IsSimple(<G>)  . . . . test if a permutation group is simple
##
##  This  is  a most interesting function.   It  tests whether  a permutation
##  group is  simple  by testing whether the group is  perfect and then  only
##  looking at the size of the group and the degree of a primitive operation.
##  Basically  it uses  the O'Nan--Scott theorem, which gives  a pretty clear
##  description of perfect primitive groups.  This algorithm is described  in
##  William M. Kantor,
##  Finding Composition Factors of Permutation Groups of Degree $n\leq 10^6$,
##  J. Symbolic Computation, 12:517--526, 1991.
##
PermGroupOps.IsSimple := function ( G )
    local   D,          # operation domain of <G>
            hom,        # transitive constituent or blocks homomorphism
            d,          # degree of <G>
            n, m,       # $d = n^m$
            simple,     # list of orders of simple groups
            transperf,  # list of orders of transitive perfect groups
            s, t;       # loop variables

    # if <G> is the trivial group, it is simple
    if Size( G ) = 1  then
        return true;
    fi;

    # first find a transitive representation for <G>
    D := Orbit( G, PermGroupOps.SmallestMovedPoint( G ) );
    if not IsEqualSet( PermGroupOps.MovedPoints( G ), D )  then
        hom := OperationHomomorphism( G, Operation( G, D ) );
        if Size( G ) <> Size( Image( hom ) )  then
            return false;
        fi;
        G := Image( hom );
    fi;

    # next find a primitive representation for <G>
    D := Blocks( G, PermGroupOps.MovedPoints( G ) );
    while Length( D ) <> 1  do
        hom := OperationHomomorphism( G, Operation( G, D, OnSets ) );
        if Size( G ) <> Size( Image( hom ) )  then
            return false;
        fi;
        G := Image( hom );
        D := Blocks( G, PermGroupOps.MovedPoints( G ) );
    od;

    # compute the degree $d$ and express it as $d = n^m$
    D := PermGroupOps.MovedPoints( G );
    d := Length( D );
    n := SmallestRootInt( Length( D ) );
    m := LogInt( Length( D ), n );
    if 10^6 < d  then
        Error("cannot decide whether <G> is simple or not");
    fi;

    # if $G = C_p$, it is simple
    if    IsPrime( Size( G ) )  then
        return true;

    # if $G = A_d$, it is simple (unless $d < 5$)
    elif  Size( G ) = Factorial( d ) / 2  then
        return 5 <= d;

    # if $G = S_d$, it is not simple (except $S_2$)
    elif  Size( G ) = Factorial( d )  then
        return 2 = d;

    # if $G$ is not perfect, it is not simple (unless $G = C_p$, see above)
    elif  Size( DerivedSubgroup( G ) ) < Size( G )  then
        return false;

    # if $\|G\| = d^2$, it is not simple (Kantor's Lemma 4)
    elif  Size( G ) = d ^ 2  then
        return false;

    # if $d$ is a prime, <G> is simple
    elif  IsPrime( d )  then
        return true;

    # if $G = U(4,2)$, it is simple (operation on 27 points)
    elif  d = 27 and Size( G ) = 25920  then
        return true;

    # if $G = PSL(n,q)$, it is simple (operations on prime power points)
    elif  (  (d =      8 and Size(G) = (7^3-7)/2          )  # PSL(2,7)
          or (d =      9 and Size(G) = (8^3-8)            )  # PSL(2,8)
          or (d =     32 and Size(G) = (31^3-31)/2        )  # PSL(2,31)
          or (d =    121 and Size(G) =        237783237120)  # PSL(5,3)
          or (d =    128 and Size(G) = (127^3-127)/2      )  # PSL(2,127)
          or (d =   8192 and Size(G) = (8191^3-8191)/2    )  # PSL(2,8191)
          or (d = 131072 and Size(G) = (131071^3-131071)/2)  # PSL(2,131071)
          or (d = 524288 and Size(G) = (524287^3-524287)/2)) # PSL(2,524287)
      and IsTransitive( Stabilizer( G, D[1] ), Difference( D, [ D[1] ] ) )
    then
        return true;

    # if $d$ is a prime power, <G> is not simple (except the cases above)
    elif  IsPrimePowerInt( d )  then
        return false;

    # if we don't have at least an $A_5$ acting on the top, <G> is simple
    elif  m < 5  then
        return true;

    # otherwise we must check for some special cases
    else

        # orders of simple subgroups of $S_n$ with primitive normalizer
        simple := [ ,,,,,
          [60,360],,,,                  #  5: A(5), A(6)
          [60,360,1814400],,            # 10: A(5), A(6), A(10)
          [660,7920,95040,239500800],,  # 12: PSL(2,11), M(11), M(12), A(12)
          [1092,43589145600],           # 14: PSL(2,13), A(14)
          [360,2520,20160,653837184000] # 15: A(6), A(7), A(8), A(15)
        ];

        # orders of transitive perfect subgroups of $S_m$
        transperf := [ ,,,,
          [60],                         # 5: A(5)
          [60,360],                     # 6: A(5), A(6)
          [168,2520],                   # 7: PSL(3,2), A(7)
          [168,8168,20160]              # 8: PSL(3,2), AGL(3,2), A(8)
        ];

        # test the special cases (Kantor's Lemma 3)
        for s  in simple[n]  do
            for t  in transperf[m]  do
                if    Size( G ) mod (t * s^m) = 0
                  and (((t * (2*s)^m) mod Size( G ) = 0 and s <> 360)
                    or ((t * (4*s)^m) mod Size( G ) = 0 and s =  360))
                then
                    return false;
                fi;
            od;
        od;

        # otherwise <G> is simple
        return true;

    fi;

end;


#############################################################################
##
#F  PermGroupOps.Base(<G>)  . . . . . . . . . .  base for a permutation group
##
PermGroupOps.Base := function ( G )
    local   S,          # stabilizer of <G>
            base;       # base <base>, result

    # handle trivial case
    if Length(G.generators) = 0 and not IsBound(G.stabilizer) then
       return [];
    fi;

    # make sure there is a stabchain
    if not IsBound(G.stabilizer)  then MakeStabChain(G);  fi;

    # go down the stabchain and collect the basepoints
    base := [];
    S := G;
    while IsBound(S.stabilizer)  do
        Add( base, S.orbit[1] );
        S := S.stabilizer;
    od;

    # return the base
    return base;
end;


#############################################################################
##
#F  PermGroupOps.StrongGenerators(<G>)  . . . . . .  strong generating system
#F                                                     of a permutation group
##
PermGroupOps.StrongGenerators := function ( G )
    local   S,          # stabilizer of <G>
            gens;       # strong generators, result

    # handle trivial case
    if Length( G.generators ) = 0 then
       return [];
    fi;

    # make sure that <G> has a stabchain
    if not IsBound(G.stabilizer)  then MakeStabChain( G );  fi;

    # go down the stabchain and collect the strong generators
    gens := [];
    S := G;
    while Length(S.generators) <> 0  do
        UniteSet( gens, S.generators );
        S := S.stabilizer;
    od;

    # return the strong generators
    return gens;
end;


#############################################################################
##
#F  PermGroupOps.Indices(<G>) . . . . . . . . . . indices of stabilizer chain
#F                                                     of a permutation group
##
PermGroupOps.Indices := function ( G )
    local   S,          # stabilizer of <G>
            inds;       # indices, result

    # make sure that <G> has a stabchain
    if not IsBound(G.stabilizer)  then MakeStabChain( G );  fi;

    # go down the stabchain and collect the indices
    inds := [];
    S := G;
    while IsBound(S.stabilizer)  do
        Add( inds, Length( S.orbit ) );
        S := S.stabilizer;
    od;

    # return the indices
    return inds;
end;


#############################################################################
##
#F  PermGroupOps.SmallestGenerators(<G>)  . . . .  smallest generating system
#F                                                     of a permutation group
##
PermGroupOps.SmallestGenerators := function ( G )

  if not IsBound(G.smallestGenerators) then
    # call the recursive function to do the work
    G.smallestGenerators:=G.operations.SmallestGeneratorsStab( 
	    # we need a stabilizer chain with respect to the smallest base
	    PermGroupOps.MinimalStabChain(G));
  fi;
  return G.smallestGenerators;
end;

PermGroupOps.SmallestGeneratorsStab := function ( S )
    local   gens,       # smallest generating system of <S>, result
            gen,        # one generator in <gens>
            orb,        # basic orbit of <S>
            pnt,        # one point in <orb>
            T;          # stabilizer in <S>

    # handle the anchor case
    if Length(S.generators) = 0  then
        return [];
    fi;

    # now get the smallest generating system of the stabilizer
    gens := PermGroupOps.SmallestGeneratorsStab( S.stabilizer );

    # get the sorted orbit (the basepoint will be the first point)
    orb := Set( S.orbit );
    SubtractSet( orb, [S.orbit[1]] );

    # handle the other points in the orbit
    while Length(orb) <> 0  do

        # take the smallest point (coset) and one representative
        pnt := orb[1];
        gen := S.identity;
        while S.orbit[1] ^ gen <> pnt  do
           gen := LeftQuotient( S.transversal[ pnt / gen ], gen );
        od;

        # the next generator is the smallest element in this coset
        T := S.stabilizer;
        while Length(T.generators) <> 0  do
            pnt := Minimum( OnTuples( T.orbit, gen ) );
            while T.orbit[1] ^ gen <> pnt  do
                gen := LeftQuotient( T.transversal[ pnt / gen ], gen );
            od;
            T := T.stabilizer;
        od;

        # add this generator to the generators list and reduce orbit
        Add( gens, gen );
        SubtractSet( orb, Orbit( Group( gens, () ), S.orbit[1] ) );

    od;

    # return the smallest generating system
    return gens;
end;


#############################################################################
##
#F  PermGroupOps.\<( <G>, <H> ) . . . . . . comparison for permutation groups
##
PermGroupOps.\< := function(G,H)
  if not (IsPermGroup(G) and IsPermGroup(H)) then
    return GroupOps.\<(G,H);
  else
    return  PermGroupOps.SmallestGenerators(G)
          < PermGroupOps.SmallestGenerators(H);
  fi;
end;

#############################################################################
##
#F  PermGroupOps.SylowSubgroup(<G>,<p>) . . . . . . . . . . .  Sylow subgroup
#F                                                     of a permutation group
##
PermGroupOps.SylowSubgroup := function ( G, p )
    local   S,          # <p>-Sylow subgroup of <G>, result
            q,          # largest power of <p> dividing the size of <G>
            D,          # domain of operation of <G>
            O,          # one orbit of <G> in this domain
            B,          # blocks of the operation of <G> on <D>
            f,          # operation homomorphism of <G> on <O> or <B>
            T,          # <p>-Sylow subgroup in the image of <f>
            g, g2,      # one <p> element of <G>
            C, C2;      # centralizer of <g> in <G>

    # get the size of the <p>-Sylow subgroup
    q := 1;  while Size( G ) mod (q * p) = 0  do q := q * p;  od;
    InfoGroup1("#I  ",p,"-SylowSubgroup in ",GroupString(G,"G"),"\n");

    # handle trivial subgroup
    if   q = 1  then
        InfoGroup1("#I  ",p,"-SylowSubgroup returns trivial subgroup\n");
        return TrivialSubgroup( G );
    fi;

    # go down in stabilizers as long as possible
    if not IsBound( G.orbit )  then MakeStabChain( G );  fi;
    while Length( G.orbit ) mod p <> 0  do
        InfoGroup2("#I    go down to stabilizer\n");
        G := Stabilizer( G, G.orbit[1] );
    od;

    # handle full group
    if q = Size( G )  then
        InfoGroup2("#I  ",p,"-SylowSubgroup returns full group\n");
        return G;
    fi;

    # handle <p>-Sylow subgroups of size <p>
    if q = p  then
        InfoGroup2("#I  ",p,"-SylowSubgroup returns cyclic group\n");
        repeat g := Random( G );  until Order( G, g ) mod p = 0;
        g := g ^ (Order( G, g ) / p);
        return Subgroup( Parent(G), [ g ] );
    fi;

    # if the group is not transitive work with the transive constituents
    D := PermGroupOps.MovedPoints( G );
    if not IsTransitive( G, D )  then
        S := G;
        while q < Size( S )  do
            InfoGroup2("#I    approximation is ",GroupString(S,"S"),"\n");
            O := Orbit( S, D[1] );
            f := OperationHomomorphism( S, Operation( S, O ) );
            T := PermGroupOps.SylowSubgroup( Image( f ), p );
            S := PreImage( f, T );
            SubtractSet( D, O );
        od;
        InfoGroup1("#I  ",p,"-SylowSubgroup returns ",
                   GroupString(S,"S"),"\n");
        return S;
    fi;

    # if the group is not primitive work in the image first
    B := Blocks( G, D );
    if Length( B ) <> 1  then
        f := OperationHomomorphism( G, Operation( G, B, OnSets ) );
        T := PermGroupOps.SylowSubgroup( Image( f ), p );
        if Size( T ) < Size( Image( f ) )  then
            S := PermGroupOps.SylowSubgroup( PreImage( f, T ), p );
            InfoGroup1("#I  ",p,"-Sylow subgroup returns ",
                        GroupString(S,"S"),"\n");
            return S;
        fi;
    fi;

    # find a <p> element whose centralizer contains a full <p>-Sylow subgroup
    repeat g := Random( G );  until Order( G, g ) mod p = 0;
    g := g ^ (Order( G, g ) / p);
    C := Centralizer( G, g );
    Size( C );
    InfoGroup2("#I  ","  ",p,"-element centralizer is ",
                GroupString(C,"C"),"\n");
    while GcdInt( q, Size( C ) ) < q  do
        repeat g2 := Random( C );  until Order( G, g2 ) mod p = 0;
        g2 := g2 ^ (Order( G, g2 ) / p);
        C2 := Centralizer( G, g2 );
        if GcdInt( q, Size( C ) ) < GcdInt( q, Size( C2 ) )  then
            C := C2;  g := g2;
            InfoGroup2("#I  ","  ",p,"-element centralizer is ",
                       GroupString(C,"C"),"\n");
        fi;
    od;

    # the centralizer operates on the cycles of the <p> element
    B := List( Cycles( g, D ), Set );
    f := OperationHomomorphism( C, Operation( C, B, OnSets ) );
    T := PermGroupOps.SylowSubgroup( Image( f ), p );
    S := PreImage( f, T );
    InfoGroup1("#I  ",p,"-SylowSubgroup returns ",GroupString(S,"S"),"\n");
    return S;

end;


#############################################################################
##
#F  PermGroupOps.FpGroup( <U> ) . . . . . . . . . . presentation of a PermGrp
##
PermGroupOps.FpGroup := function( arg )
local U,gens,h,F;

  # check trivial case
  U:=arg[1];
  if 0 = Length(U.generators)  then
    F:=FreeGroup(0);
    F.relators:=[];
    F.bijection:=GroupHomomorphismByImages(F,U,[],[]);
    return F;
  fi;

  # Try to find suitable names for this generators.
  if Length(arg) = 2 then
    gens:=WordList(Length(U.generators), arg[2] );
  else
    gens:=WordList(Length(U.generators), "F" );
  fi;

  # compute the presentation
  F:=Group( gens, IdWord );
  h:=GroupHomomorphismByImages(U,F,U.generators,gens);
  F.relators:=CoKernelGensPermHom(h);
  h:=GroupHomomorphismByImages(F,U,gens,U.generators);
  h.isMapping:=true;
  h.isHomomorphism:=true;
  F.bijection:=h;

  # Return the presentation.
  return F;

end;


#############################################################################
##
#R  Read  . . . . . . . . . . . . .  read other function from the other files
##
ReadLib( "permstbc" );
ReadLib( "permoper" );
ReadLib( "permbckt" );
ReadLib( "permnorm" );
ReadLib( "permcose" );
ReadLib( "permhomo" );
ReadLib( "permcser" );
ReadLib( "permag"   );
ReadLib( "permctbl" );
ReadLib( "ratclass" );
ReadLib( "permprod" );



