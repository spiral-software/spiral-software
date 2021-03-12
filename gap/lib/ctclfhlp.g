# -*- Mode: shell-script -*- 

#############################################################################
##
#A  ctclfhlp.g                  GAP library                      Ute Schiffer
##
##
#Y  Copyright (C) 2018-2021, Carnegie Mellon University
#Y  All rights reserved.  See LICENSE for details.
#Y  
#Y  This work is based on GAP version 3, with some files from version 4.  GAP is
#Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
##
##  This file contains functions that help to fill the cliffordmatrices and
##  columnweights, especially CentralizerOrbits[Multiple].
##
##
################################################################################
##
#F WhichClm( <clms>, <nr> ) ........ to which cliffordrecord of clms belongs nr
##
## finds out which cliffordmatrix the conjugacyclass <nr> belongs to 
## (only by counting)
## and returns the index of the cliffordrecord and the columnnumber
##
WhichClm := function( clms, nr )

    local 	i, lb, hb;	# lowbound and highbound

    i := 1;
    lb := 1; hb := clms.1.size;
    while not nr in [lb..hb]  do
        i:= i+ 1;
        lb := hb + 1;
        hb := hb + clms.(i).size;
    od;

    return [ clms.(i).elname, i, nr-lb+1 ];
end;
################################################################################
##
#F Findmi( <size>, <sum>, <centralizersize> )
##
## test the possibilities for mi, the columnweights with given
## total sum <summe> and the centralizerorder of the element
##
Findmi := function( size, sum, cent )

    local i, akt, vec,	# momentarily tested row
	liste,	# list of possible entries
	lastel,	# last element of slist
	ct,	# ct[i] := index of element of row[i] in slist
	p,	# prime of order of N
	erg, lerg;# resultvector and its length


    lastel := sum-size+1;

    # make list of possible entries of the colweights
    liste := [1];
    akt := 1;
    for i in [2..lastel] do
        if cent mod i = 0  then
  	    akt := akt + 1;
	    liste[akt] := i;
        fi;
    od;

    lastel := liste[akt];
    ct := [,0];
    akt := 2; vec := [1]; # vec[1]:= fest is fixed!!
    erg := []; lerg := 0;

    # backtracking algorithm for searching possibilities
    while akt > 1  do
 
        ct[akt] := ct[akt] + 1;
        vec[akt] := liste[ct[akt]];
        for i in [akt+1..size] do
    	    vec[i] := vec[akt];
	    ct[i] := ct[akt];
        od;
        if Sum( [1..size], x-> vec[x] ) = sum  then
	    lerg := lerg + 1;
	    erg[lerg] := Copy( vec );
        fi;

        akt := size;
        while vec[akt] = lastel and akt > 1  do # back for next listelement
	    akt := akt - 1;
        od;

    od;

    # if p=3 the module corresponding to one mi must have one or >= orbits
    # of length one
    liste := []; akt := 0;
    p := Factors(sum)[1];
    if p = 3 then

        for i in [1..Length( erg )] do
	    if not( erg[i][2] = 1 and erg[i][3] <> 1)  then
	        akt := akt + 1;
	        liste [akt] := i;
	    fi;
        od;
        erg := Sublist( erg, liste );

    fi;

    return( erg );
end;
################################################################################
##
#F  Findeo1( <fest>, <columnweights>, <liste> ) ... find row without testing
##                                                  second OR with first row
##
## <fest>  is the first entry not as square
## This function finds missing entries, if the possible entries are given
## in <liste> as square of the absolute value.
## it calculates only the second OR of the row with itself so that no
## complex entries are necessary, the square of the absolute value is taken for
## each entry (except the first one = <fest>).
## a list of possible rows, with entries the square of the absolute value for
## all entries except the first is returned.
#
Findeo1 := function( fest, m, liste )

    local i, akt, size, m2n,	# sum of m * fest = sum for 2. OR
	vec,	# momentarily tested row
	s,	# upper bound for entries of this row of cliffordmatrix
	slist,ls,# sorted and shortened list of possible entries and its length
	sm,	# sorted m
	lastel,	# last element of slist
	ct,	# ct[i] := index of element of row[i] in slist
	erg, lerg;# resultvector and its length

    size := Length( m );
    m2n := Sum( [1..size], x -> m[x] ) * fest;

    # sort out the listelements that are too large 
    slist := Copy( liste ); Sort( slist );
    sm := Copy( m ); Sort( sm );

    s :=  Int( (m2n-fest*fest) / sm[2] );
    ls := Copy( Length( slist ));
    while s < slist[ls]  do
        ls := ls - 1;
    od;
    slist := slist{[1..ls]};

    akt := 2; vec := [fest*fest]; #vec[1]:= fest is fixed!!
    ct := [,0]; 
    while m[akt] = 1  do

        vec[akt] := fest*fest;
        akt := akt+1;
        ct[akt] := 0;

    od;
    erg := []; lerg := 0;
    lastel := slist[ls];

    # backtracking for searching possibilities
    while akt > 1  do

        ct[akt] := ct[akt] + 1;
        vec[akt] := slist[ct[akt]];
        for i in [akt+1..size] do
    	    vec[i] := slist[1];
	    ct[i] := 1;
        od;
        if Sum( [1..size], x-> vec[x]*m[x] ) = m2n  then
    	    lerg := lerg + 1;
	    erg[lerg] := Copy( vec );
    	    erg[lerg][1] := fest;
        fi;

        akt := size;
        while (vec[akt] = lastel or vec[akt] = fest*fest) and akt > 1  do	
        # back for next listelement
    	    akt := akt - 1;
        od;
    od;

    return( erg );
end;
################################################################################
##
#F  o1m1( <list>, <colw> ) ...... list returned by Findeo1 tested with first row
##
## after the use of "Findeo1" this procedures nows changes those possibilities
## to "real" possibilities fulfilling additionally the 2. OR with first row:
## the square of the absolute value is substituted by a list of possibilities 
## for this entry
## This function is only for p=3 !! for p =2 useless, for p > 3 not implemented
## output is the list of possibilities for this row, as in FindRow
##
o1m1 := function( list, m )

    local  i,j, size, fest, 	# liste[1][1] is always the same
	ll,		# length of liste
	vec, akt,	# momentarily tested row and used liste-element
	lastel,		# last elements of liste[akt]
	ct,		# ct[i] := index of element of row[i] in liste
	liste,		# list of the sums instead of square 
	listr, lict,	#list of possible entries and its length
	ints, listint, whereint, #variables to change from square to rootsums
	erg, lerg;	# resultvector and its length

    size := Length( m );
    liste := Copy( list );
    fest:=liste[1][1];

    # change liste:
    # the entries(the squares of the a.v.) are substituted by lists of sums
    # of 3rd roots of unity with this absolute values
    # find all possible sums of fest roots
    #
    listr :=[]; lict := 0;
    for i in [0..fest] do
        for j in [0..fest-i] do
            lict := lict + 1;
	    listr[lict] := i + j*E(3) + (fest-i-j)*E(3)^2;
        od;
    od;
    listr := Set( listr ); lict := Length( listr );
    # calculate the squares of the absolute values
    listint := [];
    for i in [1..lict] do
        listint[i]:= GaloisCyc( listr[i],-1 )*listr[i];
    od;
    ints := Set( listint );

    # for each possible sum, find the index of its squares of the absolute 
    # value and store it, after that substitue the entries by a list
    #
    whereint := [];
    for i in [1..Length( ints )] do
        whereint[i] :=[]; 
    od;	

    for i in [1..lict] do
        Add( whereint[Position( ints, listint[i] )], listr[i] );
    od;

    for i in [1..Length(liste)] do
        for j in [2..size] do
            liste[i][j] := whereint[Position( ints, liste[i][j] )];
        od;
    od;

    erg := []; lerg := 0;
    for ll in [1..Length( liste )] do

        akt := 2; vec := [fest]; # vec[1]:= fest is fixed!!
        lastel :=[];
        for i in [2..size] do
            lastel[i] := liste[ll][i][Length( liste[ll][i] )];
        od;
        ct := [,0]; #first element is fixed

        # backtracking for searching possibilities
        while akt > 1  do

            ct[akt] := ct[akt] + 1;
            vec[akt] := liste[ll][akt][ct[akt]];
            for i in [akt+1..size] do
                vec[i] := liste[ll][i][1];
	        ct[i] := 1;
            od;
            if Sum( [1..size], x-> vec[x]*m[x] ) = 0  then
    	        lerg := lerg + 1;
	        erg[lerg] := Copy( vec );
            fi;
            akt := size;
            while (akt > 1 and vec[akt] = lastel[akt])  do # back for next 
    	        akt := akt - 1;
            od;

        od;

    od;	#all liste-lists

    return erg;
end;
################################################################################
##
#F FindRow( <fest>, <colw> [, { <liste>, "irrational", "rational"}] ) ..........
##         complete a row by restricting the possible row entries in <liste> 
##         and testing the OR with first row 
##
## <fest> first entry of row
## <colw>  columnweights
## <liste> list of possible entries of row
##
FindRow := function( arg )

    local  akt, i, j, k, kk, fest, colw, liste, p,	
	bound,	# upper bound for liste
	m2n,	# sum of m * fest = sum for 2. OR
	listgiven,# true if listis given by the procedurecall
	ergint, ints, listint, # in "irrational case " the absolute values first
	lict,	# listcounter 
	size,	# Length of m, i.e. the size of the matrix
	s,	# upper bound for entries of this row of cliffordmatrix
	slist,ls,# sorted and shortened list of possible entries and its length
	sm,	# sorted m
	vec,	# momentarily tested row
	lastel,	# last element of slist
	msum,	# sum of second or of row with itself
	ct,	# ct[i] := index of element of row[i] in slist
	erg, lerg;# resultvector and its length

    if Length( arg ) < 2 or Length( arg ) > 3  then
        Error( "usage: FindRow( fest, colw [, { liste, \"irrational\", ",
               "\"rational\"}] )" );
    elif not (IsInt( arg[1] ) and IsList( arg[2] ) )  then
        Error( "usage: FindRow( <integer>, <list of colw> [, ",
               "{ liste, \"irrational\", \"rational\"}] )" );
    fi;
    fest := arg[1]; colw := arg[2];
    size := Length( colw );
    m2n := 1;
    for i in [2..size] do
        m2n := m2n + colw[i];
    od;
    # find the prime of the order of N
    p := Factors(m2n)[1];

    m2n := m2n * fest;

    if Length( arg ) = 3 and IsInt( arg[3][1] )  then

        listgiven := true;
        liste := arg[3];

        # sort elements of liste that are too large or too small (negative ones)
        slist := Copy( liste ); Sort( slist );
        sm := Copy( colw ); Sort( sm );
        s :=  RootInt( Int( (m2n-fest*fest) / sm[2] ));

        # those too small
        ls := 1; 
        while -s > slist[ls]  do
            ls := ls + 1;
        od;
        slist := Sublist( slist, [ls..Length(slist)] );
        ls := Copy( Length( slist ));

        # those too large
        while s < slist[ls]  do 
            ls := ls - 1;
        od;
        liste := Sublist( slist, [1..ls] );

    else 
        listgiven := false;
    fi;

    if not listgiven then

        if p = 2 or (p >= 3 and IsBound( arg[3] ) and arg[3] = "rational") then

            liste := [];
            bound := Minimum( fest, RootInt( 1+QuoInt( m2n -fest*fest, 
                              Minimum( Sublist( colw, [2..size] ) ) ) ) );

            if EuclideanRemainder( Integers, fest, p ) = 0  then

                liste :=[0]; lict := 1;
	        kk := p;
                while kk <= bound do
	            lict := lict + 1;
	            liste[lict] := kk;
	            lict := lict + 1;
	            liste[lict] := -kk;
	            kk := kk + p;
	        od;

            else

	        liste := [fest]; lict := 1;
	        kk := bound - EuclideanRemainder( Integers, bound, p) 
	    	            + EuclideanRemainder( Integers, fest, p);
	        if fest = bound  then
	            kk := kk - p;
	        fi;
	        while AbsInt( kk ) <= bound+p  do
	            lict := lict + 1;
	            liste[lict] := kk;
	            kk := kk - p;
	        od;

            fi;
            listgiven := true;

        elif p = 3 then
        # "irrational" case
        # all sums of fest p-th roots of unity into liste

            liste :=[]; lict := 0;
            for i in [0..fest] do
                for j in [0..fest-i] do
                    lict := lict + 1;
  	            liste[lict] := i + j*E(3) + (fest-i-j)*E(3)^2;
                od;
            od;
            liste := Set( liste ); lict := Length( liste );
            listint := [];
            for i in [1..lict] do
  	        listint[i]:= GaloisCyc( liste[i],-1 )*liste[i];
            od;
            ints := Set( listint );

            # first test with the 2.OR with the row itself 
            # then substitute the integer by the list of elements of list with 
            # this absolute value
            ergint := Findeo1( fest, colw, ints );

            if Length( ergint ) > 0  then
                erg := o1m1( ergint, colw ); 
            else
                erg := [];
            fi;
            listgiven := false;

        else

            Print( "#E: FindRow: Sorry, irrational case for p > 3 not ",
                   "implemented!\n" );
            erg := [];
        fi;
    fi;	  

    if listgiven  then
        if ForAny( liste, x -> IsInt( x ) )  then
            Sort( liste, function( x, y ) return AbsInt( x ) < AbsInt( y ); 
                         end );
            akt := 2; vec := [fest]; #vec[1]:= fest is fixed!!
            erg := []; lerg := 0;
            ls := Length( liste ); lastel := liste[ls];
            ct := [,0];

            # backtracking for searching possibilities
            while akt > 1  do

                ct[akt] := ct[akt] + 1;
                vec[akt] := liste[ct[akt]];
                for i in [akt+1..size] do
   	            vec[i] := liste[1];
	            ct[i] := 1;
                od;
                msum := Sum( [1..size], x-> vec[x]*vec[x]*colw[x] );
                if msum > m2n  then # in liste only absolute larger elements
	            akt := size - 1;
                else
                    if msum = m2n and Sum( [1..size], x-> vec[x]*colw[x] ) = 0 
                      then
	                lerg := lerg + 1;
	                erg[lerg] := Copy( vec );
	                erg[lerg][1] := fest;
                    fi;
                    akt := size;
                fi;
                while (vec[akt] = lastel and akt > 1)  do #back for next 
	            akt := akt - 1;
                od;

            od;

        else   #shorter backtracking because of irrationalities not possible

            akt := 2; vec := [fest]; #vec[1]:= fest is fixed!!
            erg := []; lerg := 0;
            ls := Length( liste ); lastel := liste[ls];
            ct := [,0]; 
            # backtracking for searching possibilities
            while akt > 1  do
                ct[akt] := ct[akt] + 1;
                vec[akt] := liste[ct[akt]];
                for i in [akt+1..size] do
   	            vec[i] := liste[1];
  	            ct[i] := 1;
                od;
                if Sum( [1..size], x-> vec[x]*vec[x]*colw[x] ) = m2n  
                   and Sum( [1..size], x-> vec[x]*colw[x] ) = 0  then
	            lerg := lerg + 1;
	            erg[lerg] := Copy( vec );
  	            erg[lerg][1] := fest;
                fi;
                akt := size;
                while vec[akt] = lastel and akt > 1  do #back for next 
                    akt := akt - 1;
                od;
	    od;

        fi;  #IsInt or not

    fi;  	 #listgiven

    return erg;
end;
################################################################################
##
#F  CompleteClm( <clm> [, <colws>] ) ........ find all possible matrices for clm
##                                    for all ossible or for given columnweights
##
## if there is more than one possibility, all those are returned, otherwise 
## the record with filled matrix and colw are returnd
##
CompleteClm := function( arg )

    local  i, ii, j, nj, k, kk, rowct, ect, pct,
	ct,	# counter, which nj-tupel is regarded
	up,	# upper bound for tupel during the backtracking
	lp,	# length of poss
	clm,	# the CliffordRecord 
	sc, perm, # sorted first column permutation that sorts the first column
	perm2, ilist, subilist,# variables to sort after i.f.g
	case, lct, lc, # different cases for first rowentries with length and ct
	mis,	# possible colweights
	permmi,	# possible perms of the columns because of identical mi
	pg, pm,	# Elements of group of permmi's and possi[i].mat
	miakt,	# helpvariable for permmi
	posscl,	# possible matrices + colw
	possi,	# posscl[i] momentarily used
	testclm,# cliffordrecord to test the first OR of the found posscl
	ergrat, ergirr, # [ir]rational result of FindRow
	erg, lerg,ergakt,#resultvector and counter and often used element of erg
	tor, 	# posscls, that fulfill 2. resp. 1. and 2. OR
	solve;	# roww solvable

    if Length( arg ) < 1 or Length(arg) > 2 then
        Error( "usage: CompleteClm( clm [, colws] )");
    fi;
    clm := arg[1];

    if not ( IsBound(clm.mat) and IsBound(clm.size) and IsBound(clm.inertiagrps)
         and IsBound(clm.colw) ) then
        Error( "mat, size, colw and inertiagrps must be bound" );
    fi;

    if not IsBound( clm.full ) or IsBound( clm.full ) and not clm.full  then

        # which columnweights are to take
        if Length( clm.colw ) = clm.size  then
            mis := [clm.colw];
        elif IsBound( arg[2] )  then
            if not ForAll(arg[2],IsList) and 
                 ForAll(arg[2], x -> Length(x) = clm.size) then
                Error( "second argument must be a list of lists of size ", 
                        clm.size );
            else
                mis := arg[2];
            fi;
        else
            mis := Findmi( clm.size, Sum( [1..clm.size], x -> clm.mat[x][1] ), 
                           clm.roww[1] );
        fi;

        # sort first column to find duplicates, search for few poss. first
        sc := clm.mat{[1..clm.size]}[1]; 
        perm :=Sortex( sc );
        sc := Sublist( sc, [2..clm.size] );
        ilist := Copy( clm.inertiagrps );
        ilist := Sublist( Permuted( ilist, perm ), [2..clm.size]);

        # find out how many different rowtupels with have to regard:
        # first rowentry unique => extra case
        # first rowentry multiple => those belonging to the same inertia f.g.
        #			are in the same case

        case := []; ii := 0;
        rowct := 1;
        while rowct <= clm.size -1  do

            # find out, how many sc[j] are the same
            if rowct < clm.size - 1  then
                nj := rowct+1; 
                while nj < clm.size and sc[rowct] = sc[nj] do
	            nj := nj + 1;
	        od;
	        nj := nj - rowct;	

                # now we have nj rows with the same first entry. find out, 
                # which belong to same i.f.g.
	        if nj = 1  then
	            ii := ii + 1; case[ii] := 1;
	        else
	            subilist := Sublist( ilist, [rowct..rowct+nj-1]); 
	            perm2 := Sortex( subilist );
	            j := 2; kk := 1;
	            while kk <= nj  do
	                while j <= nj and subilist[j] = subilist[kk]  do
	                    j := j + 1;
	                od;
	                ii := ii + 1; case[ii] := j-kk;	
	                kk := Copy( j ); j := Copy( kk )+1;
	            od;
	        fi;

            else

    	        ii := ii + 1; case[ii] := 1;
                nj := 1;
            fi;

            rowct := rowct + nj;

        od;
        lc := Length( case ); 
        posscl := [];

        for i in [1..Length( mis )] do

            posscl[i] := [rec( mat:=[clm.mat[1]], colw:=mis[i] )];
            solve := true;
            ergrat := [[0]]; ergirr := [[0]];
            rowct := 1;

            lct := 1;
            while lct <= lc and solve  do

  	        nj := case[lct];
                # find a list possibilities for rows: 
                # a list of possibilities is created in "Findmi"

  	        if nj = 1  then
	            if ergrat[1][1] <> sc[rowct]  then	
                    # new first rowentry else FindRow for this entry is known
	                ergrat := FindRow( sc[rowct], mis[i], "rational" );
	            fi;
	            erg := ergrat;
  	        else
	            if ergirr[1][1] <> sc[rowct]  then	
                        # new first rowentry else FindRow for entry is known
	                ergirr := FindRow( sc[rowct], mis[i], "irrational" );
	            fi;
	            erg := ergirr;
	        fi;
	        lerg := Length( erg );

	        if lerg < nj  then	  #not enough possible rows
	            solve := false;
	        elif nj = 1  then     #only one row with this first entry

	            lp := Copy( Length( posscl[i] ));
	            possi := []; pct := 0;	# temporary possibilities
	            for ect in [1..lerg] do
	                for ii in [1..lp] do
                            # fill all possibilities with new row
                            tor := ForAll( [2..rowct], y -> Sum( [1..clm.size], 
                                   x -> erg[ect][x]*GaloisCyc(
			           posscl[i][ii].mat[y][x],-1 )*mis[i][x] )=0 );
	                    if tor  then
		                pct := pct + 1;
		                possi[pct] := Copy( posscl[i][ii] );
		                possi[pct].mat[rowct+1] := Copy( erg[ect] );
	                    fi;
	                od;
	            od;
	            posscl[i] := Copy( possi );

	        else  
                # more than one row with this first entry, test OR between rows
	            lp := Copy( Length( posscl[i] ));
	            possi := []; pct := 0;	# temporary possibilities

	            # test nj-tupel for OR, backtracking for rows
	            ct := [ 1, 2 ];  up := 2; tor := true;	
	            while ct[up] <= lerg - nj + up  do
	                ergakt := erg[ct[up]];
                        tor := ForAll( [1..up-1], y -> Sum( [1..clm.size], 
                               x -> erg[ct[y]][x]*GaloisCyc( ergakt[x], -1 )*
                                               mis[i][x] )=0 );
                        # nj-tupel found or other row to complete nj-tupel 
	                if nj = up  then 
	                    if tor  then  #test nj-tupel with poss. up to now
	                        for ii in [1..lp] do
		                    for kk in [1..nj] do
                                        tor := tor and ForAll( [2..rowct], 
                                               y -> Sum( [1..clm.size], 
                                               x -> erg[ct[kk]][x]*GaloisCyc(
				               posscl[i][ii].mat[y][x],-1 )*
                                               mis[i][x] )=0);
		                    od;
	                            if tor  then  
                                    # store new possibility with nj-tupel
    		                        pct := pct + 1;
		                        possi[pct] := Copy( posscl[i][ii] );
                                        # permute rows back before adding them 
                                        # by permuting the indices in ct
	          	                ct := Permuted( ct, perm2^-1 );
		                        for ii in [1..nj] do
		                            possi[pct].mat[rowct+ii] := 
                                                    Copy( erg[ct[ii]] );
		                        od;
	                            fi;
		                    tor := true;
	                        od;
	                    fi;

                            # find next nj-tupel fulfilling second OR
  	                    if ct[up] < lerg-nj+up  then   
	                        ct[up] := ct[up] + 1;
	                    else
                                while up >= 1 and ct[up] = lerg-nj+up  do 
                                    up := up - 1; 
                                od;
                                if up = 0 then # end of backtracking
                                    up := 1; 
                                fi;
                                ct[up] := ct[up]+1;
	                    fi;
	                else	#nj < up
	                    if tor  then
	                        up := up + 1; ct[up] := ct[up-1]+1;
	                    else
                                while up >= 1 and ct[up] = lerg-nj+up  do 
                                    up := up - 1; 
                                od;
                                if up = 0 then 	# end of backtracking
                                    up := 1; 
                                fi; 
	                        ct[up] := ct[up]+1;  
                                tor := true; #last poss. wrong
	                    fi;
	                fi;
	            od;

	        fi; # nj =, > 1
	        rowct := rowct + nj;
                solve := solve and pct > 0;

                if solve  then
	            posscl[i] := Copy( possi );
	        else
	            posscl[i] := [];
	        fi;	

                lct := lct + 1;
            od;  # rowct

            # find possible double entries in one mi so that permutation 
            # of such columns lead to a matrix already found
            j := 3; miakt:= 2; pg:= [1]; permmi := []; k := 0; 
            for j in [3..clm.size]  do
                if mis[i][j] = mis[i][miakt]  then
	            k := k + 1;
	            permmi[k] := (miakt, j);
	        else
	            miakt := Copy( j);
	        fi;
            od;
            if Length( permmi ) > 0  then
                pg := Elements( Group( permmi, permmi[1]^0 ));
            else
  	        pg := [()];
            fi;

            # test if some matrices are double because of possible column perms
            pct :=  Length(posscl[i]);
            if Length( pg ) > 1  and pct > 1  then # nontrivial perm
                pg := Filtered( pg, x -> x <> () );
                kk := 1;
                while kk <= pct  do
	            for ii in pg  do
	                pm := TransposedMat( Permuted( TransposedMat( 
	                                     posscl[i][kk].mat ), ii )); 
	                k := kk+1;
	                while k <= pct and pm <> posscl[i][k].mat  do
	                    k := k+1;
	                od;	
	                if k <= pct  then	#possibility double
	                    posscl[i] := Sublist( posscl[i], Filtered( 
                                         [1..pct], x -> x <> k ));
	                    pct := pct -1;
	                fi;
	            od;
	            kk := kk + 1;
	        od;
            fi;

        od;  # mis

        pct := 0; possi :=[];	#final possibilities
        testclm := Copy( clm );
        # for those that fulfill 2. OR test 1. OR
        for i in [1..Length( posscl )] do
            for j in [1..Length( posscl[i] )] do
                testclm.colw := posscl[i][j].colw;
                # sort the rows for roww
                testclm.mat := Permuted( posscl[i][j].mat, perm^-1 );
                if TestCliffordRec( testclm, [1] ) then 
                    pct := pct + 1;
                    possi[pct] := rec( mat  := testclm.mat, 
                                       colw := testclm.colw );
                fi;
            od;
        od;

        if Length( possi ) = 1  then
            clm.mat := possi[1].mat;
            clm.colw := possi[1].colw;
            clm.full := true;
            return clm;
        else
            return possi;
       fi;

    else # clm already complete
        return( "#I CompleteClm: Record is already complete.\n" );
    fi;
end;
################################################################################
##
#F  CompleteRows( <clm> [, <list>] [, { "irrational", "rational" }] )
##      .................. completes single rows of clm, all or those in <list>
##
## In contrary to "CompleteClm" the filled rows are not tested on correctness.
## clm.colw must be given completely!!
##
CompleteRows := function( arg )

    local i, j, k, l, larg, erg, clm, list,
	opt,	       # option: "[ir]rational" for FindRow
	filled,lfilled,# indices of rows that are filled in the matrix
	ok,	       # boolean variable: 2.OR fulfilled?
	ct, lastel,    # for the backtracking (to find possible matrices)
	akt, mat,      # momentarily regarded row and filled matrix
	ergmat, lmat;  #results for clm.mat and its length

    list := []; opt := ""; ergmat := [];
    larg := Length( arg );
    clm  := arg[1];

    if larg < 1 or larg > 3  then
        Error( "usage: CompleteRows( clm [, list] [, option] )\n" );
    elif larg = 3  then
        list := arg[2];
        opt  := arg[3];
    elif larg = 2  then
        if IsInt( arg[2][1] )  then
            list := arg[2];
        else
    	    opt := arg[2];
        fi;
    fi;

    # if <list> is not given, find empty rows of the matrix
    if Length( list ) = 0  then
        list := []; j := 0;
        for i in [2..clm.size] do
            if Length( clm.mat[i] ) <> clm.size  then
	        j := j+1;
	        list[j] := i;
           fi;
        od;
    fi;

    if not IsBound( clm.colw ) or Length( clm.colw ) < clm.size  then
        Error( "<colw> must have length ", clm.size, " in the cliffordrecord" );
    fi;
    if Length( list ) = clm.size -1  then # all rows missing
        return CompleteClm( clm );
    fi;

    # find the possibilities for each row, if it is to find, else
    # copy row into erg 
    # then check second OR with other rows
    erg := [clm.mat[1]]; mat := Copy( clm.mat );

    # filled contains the rows that are to be filled or are filled completely
    lfilled := 1; filled := [1];  lastel:= [clm.mat[1]];
    for i in [2..clm.size]  do

        if i in list  then 
            if opt <> ""  then
                erg[i] := FindRow( clm.mat[i][1], clm.colw, opt );
            else
                erg[i] := FindRow( clm.mat[i][1], clm.colw );
            fi;
	    if Length( erg[i] ) > 0  then
	        lfilled := lfilled + 1;
	        filled[lfilled] := i;
                lastel[i] := erg[i][Length(erg[i])];
	    fi;
        else 
	    if Length( clm.mat[i] ) = clm.size  then
	        erg[i] := [clm.mat[i]];
	        lfilled := lfilled + 1;
	        filled[lfilled] := i;
                lastel[i] := clm.mat[i];
            else
	        mat[i] := Copy( clm.mat[i] ); # keep uncompleted rows
	    fi;
        fi;

    od;

    # now backtracking for possible matrices (fulfilling the second OR)

    ct := []; lmat := 0; 
    for i in filled  do
        ct[i] := 0;
    od;
    clm.mat := [];
    akt := 2;
    while akt > 1  do

        ct[filled[akt]] := ct[filled[akt]] + 1;
        mat[filled[akt]] := erg[filled[akt]] [ct[filled[akt]]];
        for i in Sublist( filled, [akt+1..lfilled] ) do
    	    mat[i] := erg[i][1];
	    ct[i] := 1;
        od;
        ok := true;
        for i in filled do
    	    for j in [i+1..lfilled] do
	        ok := ok and Sum( [1..clm.size], x-> mat[filled[i]][x]*
		         GaloisCyc( mat[filled[j]][x], -1)*clm.colw[x] ) = 0;
	    od;
        od;
        if ok  then 
    	    lmat := lmat + 1;
	    ergmat[lmat] := Copy( mat );
        fi;
        akt := lfilled;
        # back for next listelement
        while mat[filled[akt]] = lastel[filled[akt]] and akt > 1  do
    	    akt := akt - 1;
        od;

    od;

    if lmat = 1  then
        clm.mat:= Copy(ergmat[1]);  #clm.mat is vector with all possibilities
        return clm;
    else
        return ergmat;
    fi;
end;
################################################################################
#F NUMBER( <string> ) ........... return initial part of string that is a number
##
NUMBER := function( string )

    local i, digits, number;
    digits:= "0123456789";
    i:= 1;
    number:= 0;
    while i <= Length( string ) and string[i] in digits do
      number:= 10 * number + Position( digits, string[i] ) - 1;
      i:= i+1;
    od;
    return number;
    end;

#############################################################################
##
#F  CentralizerOrbits := function( cgrp, element ) 
##
## the matrixgroup <cgrp> is the centralizer of the element <element>
## find out [element,N] of the vectorspace N where N is a p-normal subgroup
## (p is named char in the procedure)
## and calculate the orbits of cgrp on the characters of N/[element,N].
## <element> must be a matrix, the function returns the cliffordmatrix belonging
## to <element> as a representative of a conjugacy class of G
##
    orbitsFiniteFieldMatGroup := function ( G )
##
##  calculates one orbit for finite field vectorspaces with matrixoperation 

    local  pow,  moduli, size,  i,  int,  blt,  orb,  orbs,  d,  pnt,  gen,  
           img,  num;

    # set up an enumerator for finite field vectors
    pow  := []; moduli := [];
    size := Size( G.field );
    num  := 1;
    for i  in [ G.dimension, G.dimension-1 .. 1]  do
        pow[i] := num;
        num := num*size;
        moduli[i] := G.field.char; 
    od;
    int := [ 1 .. size-1 ];
    IsVector( int );

    # construct a bit list
    size := size^G.dimension;
    blt  := BlistList( [1..size], [] );

    # first orbit always is the trivial vector, spare it
    pnt :=  1;
    blt[pnt] := true;
    d := CoefficientsInt( moduli, pnt-1 );
    MakeVecFFE( d, One( G.field ) );
    orbs := [[ d ]];

    # start the orbit algorithm for all first elements in bitlist not taken
    # with vector <d>
    pnt := 2;
    while IsInt( pnt ) do
        blt[pnt] := true;
        d := CoefficientsInt( moduli, pnt-1 );
        MakeVecFFE( d, One( G.field ) );
        orb := [d];
        for pnt  in orb  do
            for gen  in G.generators  do
                img := pnt ^ gen;
                num := NumberVecFFE( img, pow, int );
                if not blt[num]  then
                    Add( orb, img );
                    blt[num] := true;
                fi;
            od;
        od;
        orbs[Length( orbs )+1] := Copy( orb );
        pnt := Position( blt, false );
    od;

    return orbs;
end;
###############################################################################
CentralizerOrbits := function( arg )

    local   i,j,k, cgrp, element,# centralizermatrixgroup and central element
	oe, 	    # element's order
	image,	    # base of the image of element-id
	wechsel,    # basis transforming matrix
        wechseli,   # the inverse of wechsel
        vecs, dualvecs, rep,
        buildmatrix,
	gen, ngrp,  # the generators of the transformed group and the group
      	nelem,lnelem, # image of elemi and its size
      	orders,
      	char,	    # the characteristic of the field the "element" belongs to
        field,	    # GF(char)
	l,	    # dimension(image)
	ct,	    # counts the nullentries of the triangulized mat
	fgrp,fgen,  # the group operating on the factor n/[element,N]
        tgrp,tfgen, # the group "transposed", i.e. generated by tfgen
	vs,	    # the vectorspace N/[element,N]
	orbv,orbc,  # the orbits of vs made by fgrp and of the characters of vs
	chi,	    # product vector(orbc)*vector(orbj)
	cliff,cliffmat,mi, # cliffordrecord, cliffordmatrix and columnweights 
        perm,
	eins, e3, e32; #variables to sum up the the p-th roots of unity

    if Length( arg ) > 3 or Length( arg ) < 2 then
        Error( "usage: CentralizerOrbits( cgrp, element ",
                               "[, rec( extraspecial := scprmat ) ] )" );
    else
        cgrp := arg[1];
        element := arg[2];
    fi;
    oe := OrderMat( element );

    if not IsBound( cgrp.dimension )  then
         cgrp.dimension := Copy( Length( element ));
    fi;

    ## calculate the operation on N/[element,N]
    ##
    image := BaseMat( element - cgrp.identity );#upper triangular form of (x-1)N
    wechsel := Copy( image );
    char := CharFFE( element[1][1] ); 
    field := GF( char );
    l := Length( image );
    ct := 0;
    i := 1;
    while i <= l  do
        if image[i][i+ct] = field.zero  then   #fill up basechange matrix 
            wechsel[l+ct+1] := cgrp.identity[i+ct];
    	    ct := ct + 1;
        else
    	    i := i + 1;
        fi;
    od;

    for i in [l+ct+1..cgrp.dimension] do
        wechsel[i] := cgrp.identity[i];
    od;

    # transform the generators of cgrp to the new basis
    # and build new group
    #
    wechseli:=wechsel^-1;
    gen := wechsel * cgrp.generators * wechseli;
    nelem := wechsel*element*wechseli;
    ngrp := Group( gen, gen[1]^0 );

    #
    # first check whether we have a nontrivial quotient space
    #
    if l >= ngrp.dimension  then
        return rec( mat := [[1]], colw := [1], orders := [oe], order := oe );
    fi;
    # make the list of generators of the factor group
    fgen :=[];
    for i in [1..Length( ngrp.generators )] do
        fgen[i] := Sublist( gen[i], [l+1..ngrp.dimension]); 
        for j in [1..ngrp.dimension - l] do 
            fgen[i][j] := Sublist( fgen[i][j], [l+1..ngrp.dimension]);
        od;
    od;
    fgrp := Group( fgen, fgen[1]^0 );
    #
    ## calculate the orbits of the vectors and the charactersums
    #
    orbv := orbitsFiniteFieldMatGroup( fgrp );
    #
    # we now try to find the order of the preimage elements of element;
    # we pull back the representatives of the orbits in N/[N,element]
    # and then work out the order by applying the matrix of element
    # in the new basis over and over again, till we hit zero:
    # applying means that we do v*element + v.
    #
    orders:=[];
    vecs:=[];
    lnelem := Length( nelem ) + 1;	#create   nelem   |0   and calculate its
    rep:=Copy(nelem);    		#	          |0	order
    for j in rep do			#	 0..0 orbv|1
        Add(j,0*Z(char));
    od;
    for i in [1..Length(orbv)] do
        vecs[i]:=Concatenation(List([1..l],x->0*Z(char)),orbv[i][1])*wechsel;
        rep[lnelem] := Concatenation( List( [1..l],x->0*Z(char) ),
    				      orbv[i][1],[Z(char)^0] );
        orders[i]:=OrderMat( rep );
    od;

    # "transpose" group because the vectors of the transposed representation can
    # be interpreted as characters
    # first check that the group fgrp is not trivial
    #
    tfgen := [];
    if Length( fgrp.generators ) = 0  then 
        tgrp:=Group( IdentityMat( fgrp.dimension, field ) );
    else
        for i in [1..Length( fgrp.generators )] do
            tfgen[i] := TransposedMat( fgrp.generators[i] )^-1;
        od;
        tgrp:=Group( tfgen, tfgen[1]^0 );
    fi;
    orbc := orbitsFiniteFieldMatGroup( tgrp ); 
    cliffmat :=[];
    mi := List( [1..Length( orbv )], x-> Length( orbv[x] ) );

    dualvecs:=[];
    for i in [1..Length( orbc )] do
        cliffmat[i] := [];
        dualvecs[i]:=Concatenation(
           List( [1..l], x -> 0*Z(char)), orbc[i][1]) * Transposed( wechseli );
        for j in [1..Length( orbv )] do

    	    eins := 0; e3 := 0; e32 := 0;
	    if char = 3  then
	        for k in [1..Length( orbc[i] )] do
	            chi := orbc[i][k] * orbv[j][1];
	            if chi = field.zero  then  eins := eins + 1;
	            elif chi = field.one  then e3 := e3 + 1;
	            else e32 := e32 + 1;
	            fi;
	        od;
	        cliffmat[i][j] := eins + e3*E(char) + e32*E(char)^2;
	    elif char = 2  then
	        for k in [1..Length( orbc[i] )] do
	            chi := orbc[i][k] * orbv[j][1];
	            if chi = field.zero  then  eins := eins + 1;
	            else e3 := e3 + 1;
	            fi;
	        od;
	        cliffmat[i][j] := eins - e3;
	    else
	        Error("CentralizerOrbits is implemented only for p = 2,3!\n");	
	    fi;

        od;
    od;

    # sort columns, so that columnsweights are growing
    perm := Sortex( mi );
    j := Length( mi);
    k := List( [1..j], x -> cliffmat{[1..j]}[x] );
    k := Permuted( k, perm );
    cliffmat := List( [1..j], x -> k{[1..j]}[x] );

    cliff:= rec( mat := cliffmat, colw := mi, orders:= Permuted( orders, perm ),
                 order:= oe, vecs := Permuted( vecs, perm), 
                 dualvecs := Permuted( dualvecs, perm ) );
################################################################################
# function to create a matrix, its order should be the elementorder of the
# extraspecial extension.
#
    buildmatrix:=function(matrix,vector,bimat, prime)

    local newmat, f, one, null, dim, dimhalf, vectrans, i, j;
    newmat := [];
    f := GF(prime);
    one := f.one; null := f.zero;
    dim := Length( matrix );
    dimhalf := dim / 2;
    vectrans := vector * bimat;

    newmat[1] := [];
    newmat[1][1] := one;
    for i in [2..dim+1]  do
        newmat[1][i] := vector[i-1];
    od;
    newmat[1][dim+2] := null;
    for i in [2..dim+1]  do
        newmat[i] := [];
        newmat[i][1] := null;
        for j in [2..dim+1]  do
            newmat[i][j] := matrix[i-1][j-1];
        od;
        newmat[i][dim+2] := vectrans[i-1];
    od;
    newmat[dim+2] := [];
    for i in [1..dim+1]  do
        newmat[dim+2][i] := null;
    od;
    newmat[dim+2][dim+2] := one;

    return newmat;
    end;
    #
    # now fill in the orders of the extraspecial normal subgroup version 
    #
    if IsBound( arg[3] ) and IsRec( arg[3] ) and IsBound( arg[3].extraspecial )
       and IsMat( arg[3].extraspecial ) then
        cliff.eorders := [];
        for j in [1..Length( cliff.orders )] do
            cliff.eorders[j] := OrderMat( buildmatrix( element, cliff.vecs[j],
                                          arg[3].extraspecial, char ) );
        od;
    fi;

    return( cliff );
end;
################################################################################
##
#F CentralizerOrbitsMultiple( <permgroup>, <classes>, <matrixgroup> [, <list> ]
##                          [, rec( extraspecial )] )
##                  ....... calculates all cliffordmatrices of group for classes
##
## determine all clifford matrices that index is given in list of the 
## semidirect product
## list contain the indices of "classes", that needn't be the same as the 
## indices of the cliffordtable
## <extraspecial> must be a matrix of the symplectic scalarproduct that is 
## invariant under the group
##
CentralizerOrbitsMultiple := function( arg )

    local  hom, cliff, i, rep, cen, repm, cenm, esp,
           g, classes, matg, list;

    if not ( Length( arg ) in [3,4,5] and IsPermGroup( arg[1] ) 
       and IsMatGroup( arg[3] ) and IsList( arg[2] ) ) then
        Error( "usage: CentralizerOrbitsMultiple( g, classes, matg [, list ]",
               "[, rec( extraspecial )]" );
    fi;
    g:= arg[1]; matg:=arg[3]; classes:=arg[2]; 

    if IsBound( arg[4] ) then
        if IsList( arg[4] ) then
            list := arg[4];
            if IsBound( arg[5] ) and not IsRec( arg[5] ) then
                Error( " invalid fifth argument. ");
            elif IsBound( arg[5] ) then
                esp := arg[5].extraspecial;
            fi;
        elif IsRec( arg[4] ) then
            esp := arg[4].extraspecial;
            list := Filtered( [1..Length( classes )], 
                              x -> IsBound( classes[x] ) ); 
        else
            Error( " invalid fourth argument. ");
        fi;
    else
        list := Filtered([1..Length(classes)], x-> IsBound(classes[x])); 
    fi;

    cliff := [];
    hom := GroupHomomorphismByImages( g, matg, g.generators, matg.generators );
    hom.isMapping := true;
    hom.isHomomorphism := true;

    for i in list do

        if IsRec( classes[i] ) then
            rep := classes[i].representative;
        else
            rep := classes[i];	
        fi;
        cen := Centralizer( g, rep );

        repm := Image( hom, rep );
        cenm := List( cen.generators, x->Image(hom,x) );
        cliff[i]:=CentralizerOrbits( Group(cenm,cenm[1]^0), repm,
                                     rec( extraspecial:= esp ) );
        cliff[i].roww:=[Size( cen )];
    od;

    return cliff;
end;
################################################################################
##
#F AdaptCOMatricesToCliffordTable( <clt>, <mats> ) ... find for clt-records the
##                                                     correct mat in mats
##
## adapts the by CentralizerOrbitsMultiple calculated records with
## mat, colw, order, orders, roww[1] to the cliffordtable
## therefore find for every element of the cliffordtable a record that has the
## same elementorder, roww[1] and same first column of mat
##
AdaptCOMatricesToCliffordTable := function( clt, mats )

    local i, j, lmats, perms, col1s, permc, col1c, clm, corec,
  	  hlp, trips;

    mats := Copy( mats );
    mats := Filtered( mats, x -> IsBound( x ) );
    lmats := Length( mats );
    perms := List( mats, x-> Sortex( x.mat{[1..Length( x.mat[1] )]}[1] ) );
    col1s := List( [1..Length( mats )], x-> Permuted( mats[x].mat
                              {[1..Length( mats[x].mat[1] )]}[1], perms[x] ) );

    for i in [1..Size( clt )] do

        clm := clt.(i);
        permc := Sortex( clm.mat{[1..clm.size]}[1] );
        col1c := Permuted( clm.mat{[1..clm.size]}[1], permc );
        j := 1; 
        while  j <= lmats and not( mats[j].order = clm.order and 
               mats[j].roww[1] = clm.roww[1] and col1c = col1s[j])  do
            j := j+1;
        od;

        if j <= lmats then

            corec := mats[j];
            # write permuted mat and colw in the clt element
            clm.colw := corec.colw;
            clm.orders := corec.orders;
	    if IsBound( corec.eorders ) then 
                clm.eorders := corec.eorders; 
            fi;
	    if IsBound( corec.vecs ) then 
                clm.vecs := corec.vecs; 
            fi;
	    if IsBound( corec.dualvecs ) then 
                clm.dualvecs := corec.dualvecs; 
            fi;
            clm.mat := Permuted( corec.mat, perms[j]*permc^-1 );

            # if there are different inertiagrps for same first column entry,
            # check rational and complex entries
            trips := List( [2..clm.size], 
                           x -> [clm.inertiagrps[x], clm.mat[x][1]] );
            for i in [1..clm.size-1] do
                if not (trips[i] in trips{[i+1..clm.size-1]} or 
                        trips[i] in trips{[1..i-1]} ) and 
                   not ForAll( [2..clm.size], x -> IsInt( clm.mat[i+1][x] ) )  
                   then
                    # change row: find apt one, if not used yet, exchange
                    j := 1;
                    while j <= clm.size-1 and (clm.mat[j+1][1] <> trips[i][2] or
                      not ForAll( [2..clm.size], 
                                  x-> IsInt( clm.mat[j+1][x] ) ) ) do
                       j := j+1;
                    od;
                    if j <= clm.size-1 then
                         hlp := Copy( clm.mat[j+1] );
	                 clm.mat[j+1] := Copy( clm.mat[i+1] );
                         clm.mat[i+1] := hlp;
                     else
	                Print( "#I AdaptCO: rows of matrix ", clm.elname, 
                               " cannot be adapted automatically.\n");
                    fi;
                elif (trips[i][2] in trips{[i+1..clm.size-1]}[2] or 
                      trips[i][2] in trips{[1..i-1]}[2] ) and 
                   ForAll( [2..clm.size], x -> IsInt( clm.mat[i+1][x] ) )  then
                    Print( "#I AdaptCO: Row ", i+1, " of matrix ", clm.elname,
                           " may be exchanged.\n" );
                fi;
            od;

        fi;

    od;

    return clt;
end; 
