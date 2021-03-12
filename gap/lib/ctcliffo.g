# -*- Mode: shell-script -*- 

#############################################################################
##
#A  ctcliffo.g                  GAP library                      Ute Schiffer
##
##
#Y  Copyright (C) 2018-2021, Carnegie Mellon University
#Y  All rights reserved.  See LICENSE for details.
#Y  
#Y  This work is based on GAP version 3, with some files from version 4.  GAP is
#Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
##
##  This file contains primary functions for CliffordRecords and CliffordTables.
##  They are domains and they get the CharTableOps as basis for the operations
##
##  The file contains:
##  CliffordRecord(Ops), CliffordTable(Ops), DetermFusions, ClmInit
##  PrintCliffordRec, IsCliffordRec, IsCliffordTable
##  and MakeHead (function to complete the head of the charactertable)
##  and SplitClass, SplitCliffordTable
##
##
################################################################################
#F  IsCliffordRec( <C> ) ................ is the input is a record with mat, 
##                             colw and operations, fusionsclasses, inertiagrps 
##
IsCliffordRec := function( C )

    return IsRec( C ) and 
      IsBound( C.operations ) and IsBound( C.mat ) and IsBound( C.colw ) and
      IsBound( C.inertiagrps ) and IsBound( C.fusionclasses );
end;
################################################################################
##  make domain "CliffordRecord". First declare all operations
##
CliffordRecordOps := OperationsRecord( "CliffordRecordOps", CharTableOps );

CliffordRecordOps.Representative := function( C )

    return [ C.mat, C.colw ];
end;

CliffordRecordOps.Elements := function( C )

    return RecFields( C );
end;

CliffordRecordOps.\= := function( C, D )

    return Sort( Elements( C ) ) = Sort( Elements( D ) ) and 
           ForAll( Elements( C ), r -> C.(r) = D.(r) );
end;
################################################################################
## returns the first resp second orthogonality relation for <C>.mat 
## z1, z2 may be the whole column/row or the index of the column/row
## ScalarProduct( <clm>, <z1>, <z2>, {1|2} )
##
CliffordRecordOps.ScalarProduct := function( arg )

    local  i, z1,z2, l1, sum, weights;

    ## check the input
    if not arg[4] in [1,2]  then
        Error( "last argument must be in [1,2]" );
    fi;

    z1:= []; z2 := [];
    if IsList( arg[2] )  then
        z1:= arg[2];
    elif IsInt( arg[2] )  then
        if arg[4] = 1  then z1 := arg[1].mat{[1..arg[1].size]}[arg[2]];
        else                z1 := arg[1].mat[arg[2]];
        fi;
    else
        Error( "second argument is invalid" );
    fi;

    if IsList( arg[3] )  then
        z2 := arg[3];
    elif IsInt( arg[3] )  then
        if arg[4] = 1  then z2 := arg[1].mat{[1..arg[1].size]}[arg[3]];
        else                z2 := arg[1].mat[arg[3]];
        fi;
    else
        Error( "third argument is invalid" );
    fi;

    if arg[4] = 1  then weights := arg[1].roww;
    else                weights := arg[1].colw;
    fi;

    l1 := Length( z1 );
    sum := 0;
    if l1 = Length( z2 ) and l1 = Length( weights )  then 
        for i in [1..l1] do
    	    sum := sum + weights[i]  * z1[i]* GaloisCyc( z2[i], -1 );
        od;
    else
        Error( "length of first row/col, second row/col and ",
               "weights different" );
    fi;

    return sum;
end;
################################################################################
## MatScalarProducts( <clm>, {1.OR|2.OR} )
## returns the matrix of the first resp second orthogonality relations for 
## <clm>.mat 
##
CliffordRecordOps.MatScalarProducts := function ( clm, OR )

    local  i, j, k, sum, z, weights, scprmatrix, scpr;

    if not IsCliffordRec( clm ) then
        Error( "<clm> is not a cliffordrecord" );
    elif not (IsBound( clm.mat )  and IsMat( clm.mat ) )
         or (OR = 1 and not (IsBound( clm.colw ) 
                    and Length( clm.colw ) = Length( clm.mat[1] ) ))
         or (OR = 2 and not (IsBound( clm.roww ) 
                    and Length( clm.roww ) = Length( clm.mat[1] ) )) then
        Error( "<clm> is not complete for ", OR, ".OR" );
    elif not OR in [1,2]  then
        Error( "<OR> must be in [1,2]" );
    fi;

    z := clm.mat;
    scprmatrix := [  ];

    if OR = 1 then
        weights := clm.roww;
        for i  in [ 1 .. clm.size ]  do
            scprmatrix[i] := [  ];
            for j in [ 1 .. clm.size ] do
                sum := 0;
                for k in [1.. clm.size] do
	             sum := sum + weights[k] * z[k][i]*GaloisCyc( z[k][j], -1 );
                od;
                scprmatrix[i][j] := Copy(sum);
                if not IsInt( sum )  then
                    if IsRat( sum )  then
                        InfoCharTable2( "#E MatScalarProducts: sum not",
                          " divisible by group order\n" );
                    elif IsCyc( sum )  then
                         InfoCharTable2( "#E MatScalarProducts: summation not",
                          " integer valued\n" );
                    fi;
                fi;
            od; 
        od;

    else   # OR = 2

        weights := clm.colw;
        for i  in [ 1 .. clm.size ]  do
            scprmatrix[i] := [  ];
            for j in [ 1 .. clm.size ] do
                sum := 0;
                for k in [1.. clm.size] do
	            sum := sum + weights[k] * z[i][k]*GaloisCyc( z[j][k], -1 );
                od;
                scprmatrix[i][j] := Copy(sum);
                if not IsInt( sum )  then
                    if IsRat( sum )  then
                        InfoCharTable2( "#E MatScalarProducts: sum not ",
                            "divisible by group order\n" );
                    elif IsCyc( sum )  then
                        InfoCharTable2( "#E MatScalarProducts: summation not"
                         , " integer valued\n" );
                    fi;
                fi;
        od; od;
    fi;
    return scprmatrix;
end;

################################################################################
# displays the cliffordmatrix in a readable form
#
CliffordRecordOps.Print := function( C )

  local i, ii, j, k, l, clm,	# record of clms to be displayed (clms[nr_name])
	sum,		# sum of first col without last row[s](splitclass)
	faccen, clen, censet,	# information about the cen-factors
	facroww, rset, rlen,	# information about the cl.roww-factors
	mat,			# matrix to display
	cllen,			# number of matrices to display
	lmat,			# length, that is the number of rows in mat
	aktrow,			# actual row of displaymatrix
	indanf,			# index of first column in chartable
	leer,			# emptystring
	prz,			# prime in a factorset
	arr, rowmax, colmax;	# variables for printing the mat

    # if clms is the libraryversion, get full clm
    if not IsCliffordRec( C )  then
        Error( " input is not a cliffordrecord " );
    else
        clm := Copy( C );
    fi;
    if not IsBound( clm.size ) then clm.size := Length( clm.mat[1] ); fi;

    # calculate and factor centralisators and row-weights 
    faccen := []; facroww:=[];

    # there is a difference in centralizercalculation between splitted matrices
    # and non-splitted
    if IsBound( clm.splitinfos )  then
        if clm.splitinfos.classindex > 0  then 
            sum :=Sum( [1..clm.size-clm.splitinfos.numclasses+1], 
	    	        x -> clm.mat[x][1] );
        else
            sum :=Sum( [1..clm.size], x -> clm.mat[x][1] );
        fi;
        cllen := clm.roww[1] * sum * clm.splitinfos.p;
    else
        cllen := clm.roww[1] * Sum( [1..clm.size], x -> clm.mat[x][1] );
    fi;

    if Length( clm.colw ) = clm.size  then	# columnweights filled
        for i in [1..clm.size] do
            faccen[i]:= Factors( cllen/clm.colw[i] );
            facroww[i] := Factors( clm.roww[i] );
        od;
    else
        faccen[1]:= Factors( cllen );
        facroww[1] := Factors( clm.roww[1] );
        for i in [2..clm.size] do
            faccen[i]:= [1]; 
            facroww[i] := Factors( clm.roww[i] );
        od;
    fi;

    censet := Set( faccen[1] ); clen := Length( censet );
    rset := Set( facroww[1] ); rlen := Length( rset );

    # start to fill matrix to display
    #
    mat := [[clm.elname]];
    if IsBound( clm.order )  then
        mat[2] := [clm.order];
    else
        mat[2] := [" "];
    fi;
    leer:=[" "];
    for i in [1..rlen] do
        leer[i+1] := " ";
        mat[1][i+1] := " ";
        mat[2][i+1] := " ";
    od;

    # centralizer orders are factorized
    for i in [1..clen] do
        if i > 2 then
    	    mat[i] := Copy( leer );
        fi;
        prz := censet[clen-i+1];
        mat[i][rlen+2] := prz;
        for j in [1..clm.size] do
    	    mat[i][rlen+2+j] := Length( Filtered( [1..Length( faccen[j])], x -> 
	    		          faccen[j][x] = prz ) );
        od;
    od;
    aktrow := clen + 1;

    # row for new elementorders if known
    mat[aktrow] :=Copy( leer ); mat[aktrow+1] := [];
    mat[aktrow][rlen+2] := "or";
    if IsBound( clm.orders )  then
        Append( mat[aktrow], clm.orders );
    fi;

    aktrow := aktrow + 1;

    # fill roww-factors
    for j in [1..clm.size] do
        mat[aktrow+j] := [];
    od;
    for i in [1..rlen] do
        prz := rset[rlen-i+1];
        mat[aktrow][i] := prz;
        for j in [1..clm.size] do
    	    mat[aktrow+j][i] := Length( Filtered( [1..Length( facroww[j])], 
                                 x -> facroww[j][x] = prz ));
        od;
    od;

    # fill Ti and classes and the matrix	
    mat[aktrow][rlen+1] := "cl";
    mat[aktrow][rlen+2] := "Ti";
    # fill first row of cliffordmatrix
    if IsList( clm.mat[1][1] )  then	#one matrix to display
        cllen := Length( clm.mat );
    else
        cllen := 1;
    fi;

    for ii in [1..cllen] do	# display mats one after another
        if cllen > 1  then	
	    Print( "\n");
	    clm.mat := Copy( C.mat[ii] );
        fi;

        for j in [2..clm.size] do
            mat[aktrow+1][rlen+2+j] := 1;
        od;
        for i in [1..clm.size] do
            k := aktrow+i;
            mat[k][rlen+1] := clm.fusionclasses[i];
            mat[k][rlen+2] := clm.inertiagrps[i];
            for j in [1..clm.size] do
                l := rlen+2+j;
                if IsBound( clm.mat[i][j] )  then mat[k][l] := clm.mat[i][j];
    	        else mat[k][l] := "?"; fi;
            od;
        od;

        # put the colweights into mat
        k := aktrow + clm.size+1;
        mat[k] := Copy( leer );
        Add( mat[k], " ");

        for i in [1..Length( clm.colw )] do
            mat[k][rlen+2+i] := clm.colw[i];
        od;

        lmat := Length( mat );

        # Print the matrix in an apt form
        arr := List( mat, x -> String( x ) );
        #find longest row
        rowmax := 1;
        for i in [1..lmat] do
            k := Length( mat[i] );
            for j in [1..k]  do
                if IsBound( mat[i][j] ) then
                    arr[i][j] := String( mat[i][j] );
                else
                    arr[i][j] := " ";
                fi;
            od;
            if k > rowmax  then
    	        rowmax := Copy( k );
            fi;
        od;

        colmax := [[]];
        for i in [1..rowmax] do
            colmax[i] := 0;
            for j in [1..lmat] do
                if i <= Length( mat[j] ) and 
                   Length( arr[j][i] ) > colmax[i] then
	            colmax[i] := Length( arr[j][i] );
	        fi;
            od;
            colmax[i] := colmax[i]+1;
        od;
        colmax[rlen+3] := colmax[rlen+3]+1;	#space before matrix starts

        # Print arr
        for i in [1..lmat] do
            for j in [1..Length( mat[i] )] do
	        Print( String( arr[i][j], colmax[j] ) );
            od;
            Print( "\n" );
        od;

    od;	#all mats
end;
################################################################################
##
#F  TestCliffordRec( <clm> [, <t12> ] ) .... tests t12's ORs and entries of clm
##
##  Tests the lengths of the entries and ORs
##  the function returns "true" if the relations are fulfilled,
##  otherwise all pairs of rows/columns not fulfilling the relation. 
##  either centrsize must be given or roww must be bound in the record
##
TestCliffordRec := function( arg )

    local  i, j, clm, t12, sum,	mat,
      	mn,	# expected result of OR (a,a)
      	error,err1, err2,	# relations still fulfilled?
      	stelle,	# contains rhe rows/cols which don't fulfil the OR
      	m;	# the right weights for the OR

    if not (Length( arg ) in [1,2] and IsCliffordRec( arg[1] )) then
        Error( "usage: TestCliffordRec( clm.ind [, t12 ] )" ); 
    elif Length( arg ) = 2 then
        t12 := arg[2];
        if not IsList( t12 )  then
            Error( "last argument must be a list." );
        elif not t12 in [[1],[2],[1,2],[2,1]]  then
            Error( "invalid number for OR (or double number)" );
        fi;
    else
        t12 := [1,2];
    fi;

    clm := Copy( arg[1] ); 
    if not (IsBound( clm.roww ) and IsBound( clm.colw ) and IsBound( clm.mat )
       and  IsBound( clm.fusionclasses ) and IsBound( clm.inertiagrps ) 
       and  IsBound( clm.size ) ) then 
        Print( "#I TestCliffordRec: item(s) missing in cliffordrecord.\n" );
    fi; 
    error := false;
    err1 := false;
    err2 := false;
    if not IsBound( clm.size ) then 
        clm.size := Length( clm.mat[1] ); 
    fi;
    if IsBound( clm.inertiagrps ) and Length( clm.inertiagrps ) < clm.size  then
        Print( "#E TestCliffordRec: 'inertiagrps' not complete.\n" );
        error := true;
    fi;
    if IsBound( clm.fusionclasses ) and Length( clm.fusionclasses ) < clm.size
       then
        Print( "#E TestCliffordRec: 'fusionclasses' not complete.\n" );
        error := true;
    fi;
    if Length( clm.roww ) < clm.size  then
        Print( "#E TestCliffordRec: 'roww' not complete.\n" );
        error := true;
    fi;
    if Length( clm.colw ) < clm.size  then
        Print( "#E TestCliffordRec: 'colw' not complete.\n" );
        error := true;
    fi;

    # test orthogonality relations
    #
    if not IsMat( clm.mat ) or Length( clm.mat ) < clm.size  then
        Print( "#E TestCliffordRec: 'mat' not complete. ",
               "No orthogonality test.\n" );
        error := true;
    else 
        if IsBound( clm.splitinfos ) then
            if clm.splitinfos.classindex > 0  then
                sum := clm.splitinfos.p * 
                       Sum( [1..clm.size-clm.splitinfos.numclasses+1], 
                       x-> clm.mat[x][1] );
            else
                sum := clm.splitinfos.p * 
                       Sum( [1..clm.size], x-> clm.mat[x][1] );
            fi;
        else
            sum := Sum( [1..clm.size], x-> clm.mat[x][1] );
        fi;
        mn := [];
        if 1 in t12  and Length( clm.roww ) = clm.size  then 
            stelle := [];
            # fill variables for first OR
            for i in [1..clm.size] do   
                mn[i] := clm.roww[1] * sum / clm.colw[i]; od;
            err1 := false; 
            # OR between different/same columns
            for i in [1..clm.size] do
                for j in [i+1..clm.size] do
    	            if (ScalarProduct( clm, i, j, 1 ) <> 0) then
      	                err1 := true;
                        stelle[Length( stelle )+ 1] := [i,j];
                    fi;
                od; 
                if ScalarProduct( clm, i, i, 1 ) <> mn[i]  then
    	            err1 := true;
                    stelle[Length( stelle )+ 1] := [i,i];
                fi;
            od;

            if err1  then
                Print( "#E TestCliffordRec: wrong scalarproduct between ",
                       "columns ", stelle, "\n" );
            fi;
        fi; 

        if 2 in t12 and Length( clm.colw ) = clm.size  then
            stelle := [];
            # fill variables for second OR
            for i in [1..clm.size] do   
                mn[i] := clm.mat[i][1] * sum;
            od;
            if IsBound(clm.splitinfos) and clm.splitinfos.classindex > 0 then 
	        if clm.splitinfos.p in [2, 3] then  
                     mn[clm.size]   := sum;  
                fi;
                if clm.splitinfos.numclasses = 3 then  
                    mn[clm.size-1] := sum; 
                fi;
            fi;

            err2 := false; 

            # OR between different/same rows
            for i in [1..clm.size] do
                for j in [i+1..clm.size] do
    	            if ScalarProduct( clm, i, j, 2 ) <> 0  then
      	                err2 := true;
                        stelle[Length( stelle )+ 1] := [i,j];
                    fi;
                od; 
                if ScalarProduct( clm, i, i, 2 ) <> mn[i]  then
    	            err2 := true;
                    stelle[Length( stelle )+ 1] := [i,i];
                fi;
            od;
            if err2  then
                Print( "#E TestCliffordRec: wrong scalarproduct between ",
                       "rows ", stelle, "\n" );
            fi;
        fi;
    fi;
    if not ( error or err1 or err2 ) then
        arg[1].full := true;
    fi;  
    return not ( error or err1 or err2 );
end;
################################################################################
##
#F CliffordRecords( <Ti> ) ........ creates the cliffordrecords out of Ti
##
##  determine the fusions of the inertia factorgroups into first inertia
##  factor H and call for each element of H "CmInit"
##  to make the matrices and fill them as far as possible
##
##  possible entries for Ti.fusions:
##  - totally missing: all fusions must be unique
##  - some single entries missing: this fusion must be unique
##  - integer: index of the fusion stored in the charactertable
##  - a list: the fusionlist to be taken. it is not tested in here
##
##  in Ti.tables explicit charactertables are expected
################################################################################
##  first find fusions of classes of inertia groups into the first inertia group
##  i.e. the factor group
##  possible entries for Ti.fusions:
##  - totally missing: all fusions must be unique
##  - some single entries missing: this fusion must be unique
##  - integer: index of the fusion stored in the charactertable
##  - a list: the fusionlist to be taken. it is not tested in here
##
DetermFusions := function( Ti )

    local  i, l,	# Number conjugacy classes of H
        list,	# list of fusions to be found
	AnzTi,	# number of inertiagroups
	fuses,	# explicit fusionlists
      	fus;	# fusions of the classes as written in the CharTable

    fuses := []; list:= [];
    AnzTi := Length( Ti.tables );
    l := Length( Ti.tables[1].classes );

    if not IsBound( Ti.fusions )  then list := [2..AnzTi]; 
    else

        for i in [2..AnzTi] do
            if not IsBound( Ti.fusions[i] ) or not IsList( Ti.fusions[i] )  then
                list[Length( list )+1]:= i;
            else
                fuses[i] := Copy( Ti.fusions[i] );
            fi;
        od;

    fi;

    for i in list do   # find the fusion[s] calculated or stored by GAP 

        fus := GetFusionMap( Ti.tables[i], Ti.tables[1] );
        if IsBound( Ti.fusions ) and IsBound( Ti.fusions[i] ) and 
           IsInt( Ti.fusions[i] ) then
            if not IsList( fus )  then  # but no list for the index is known
                 Error( "list of fusions from " , Ti.tables[i].identifier, 
                        " into ", Ti.tables[1].identifier, " is missing" );
	    elif IsInt( fus[1] ) and Ti.fusions[i] = 1  then 
                fuses[i] := Copy( fus );
            elif IsBound( fus[Ti.fusions[i]] )  then
	        fuses[i] := Copy( fus[Ti.fusions[i]] );
	    else
	        Error( "No fusion for " , Ti.tables[i].identifier,
			   " is stored with index ", Ti.fusions[i] );
	    fi;
        else

            # if table is not one of projectives, find (for testing) fusion
            if not IsList( fus ) and Length( Ti.tables[i].irreducibles[1] ) = 
   		               Length( Ti.tables[i].irreducibles ) then
                # representative fusions not possible, 
                # because Ti[1].automorphisms more than 1x
                fus:= SubgroupFusions( Ti.tables[i], Ti.tables[1],
	  		    	       rec( quick := "true" ) );
	        if Length( fus ) = 1  then 
                    fuses[i] := Copy( fus[1] ); 
                else
                    Error( "Fusion from " , Ti.tables[i].identifier, " into ",
	                   Ti.tables[1].identifier, 
                           " not unique by subgroup fusion" );
                fi;
	    else
                if IsInt( fus[1] ) then fuses[i] := Copy( fus );
                else
                    Error( "Fusion from " , Ti.tables[i].identifier, " into ",
	             Ti.tables[1].identifier, " not unique by GetFusionMap" );
                fi;
	    fi;

        fi;
    od;
    return( fuses );
 
end;
###############################################################################
##  find fusions of classes of inertia groups in the first inertia group
##  i.e. the factor group, calculate the columnweights and the first
##  column of the clifford-matrix
##  fusions of the classes into classes of Ti.tables[1] must be given in 
##  Ti.fusions as an explicit list
## 
ClmInit := function( ind, Ti )

    local i, k, size,	# final size of the cliffordmatrix
      	rw,		# rowweights
      	nxi,		# helpvar & sum of the first column = [N:Nxi]
      	col1,		# first column of the cliffordmatrix
	sum,		# sum of first column of the matrix
      	fus,		# copy of fusions of the classes 
      	name,		# classname of the actual class 
	nr,		# index of the class
      	ingr,		# number of inertia group for each row
      	irr, rowfus,	# classnumber of inertia group corresponding to the row
      	p, 		# fusioning classes of Ti.tables[size]
	lmat, 		# library-cl.matrix 
	l, clmlist,	# list of cliffordmatrices in the library
      	retrec;		# return the partly filled record

    ClassNamesCharTable( Ti.tables[1] );
    if not IsInt( ind )  then
        nr := Position( Ti.tables[1].classnames );
        name := ind;
    else
        nr := ind;
        name := Ti.tables[1].classnames[nr];
    fi;

    nxi := Ti.tables[1].centralizers[nr];

    # Initialisation for first inertia group, that is the factorgroup G/N =: H
    col1 := [1]; sum := 1;
    rw := [nxi];
    ingr := [1]; rowfus := [nr];

    # for the other inertia groups determine all fusioning classes and their
    # centralizer orders

    size := 1;
    for i in [2..Length( Ti.tables )] do # for all inertia factors really in H
        fus := Ti.fusions[i];
        # if there are projective chars, test if class is regular
        if Ti.ident[i][1] = "projectives"  then
            k := PositionProperty( [1..Length( Ti.tables[i].projectives )],
                                    x -> Ti.tables[i].projectives[x].name =
                                         Ti.ident[i][3] );
            irr := Ti.tables[i].projectives[k].chars;
            p := Filtered( [1..Length(fus)], ii -> fus[ii] = nr and
                            ForAny( irr, x -> x[ii] <> 0 ) );
        else
            p:= Filtered( [1..Length( fus )], ii -> fus[ii] = nr ); 
        fi;

        # for each class fusioning into class "nr" in H fill initial dates 
        # as roww, inertiagrp and first rowentry
        for k in [1..Length(p)] do
            size := size + 1;
            rowfus[size] := p[k];
            rw[size] := Ti.tables[i].centralizers[p[k]];
            col1[size] := nxi / rw[size]; 
            sum := sum + col1[size];
            ingr[size] := i;
        od;
    od;

    retrec:= rec(
                  isDomain      := true,
                  operations    := CliffordRecordOps, 
  		  nr            := nr,
                  full          := false,
                  elname        := name,
		  order         := Ti.tables[1].orders[nr],
		  orders        := [Ti.tables[1].orders[nr]],
#T Is there any place where this is completed ??
                  size          := size, 
	  	  inertiagrps   := ingr,
		  fusionclasses := rowfus,
                  roww          := rw,
                  colw          := [1],
                  mat           := [[1]]
                 );

    # Fill first row and column.
    for i in [1..size] do
      retrec.mat[i] := [col1[i]];
      retrec.mat[1][i] := 1;
    od;

    # Fill the matrix as far as possible for small sizes.
    if   size = 1 then
        retrec.full := true;
    elif size = 2 then
        retrec.mat[2][2] := -1;
        retrec.colw      := col1;
        retrec.full      := true;
    elif size = 3 and Sum( [1..3], x -> col1[x] ) = 3  then
        retrec.mat := [[1, 1, 1], [1, E(3), E(3)^2], [1, E(3)^2, E(3)]];
        retrec.colw := [1,1,1];
        retrec.full := true;
    elif size = 3 and col1 = [1, 1, sum-2]  then
        retrec.mat := [[1, 1, 1], [1, 1, -1], [sum-2, -2, 0]];
        i := (sum-2)/2;
        retrec.colw := [1, i, i+1];
        retrec.full := true;
    elif size = 3 and col1 = [1, sum-2, 1]  then
        retrec.mat := [[1, 1, 1], [sum-2, -2, 0], [1, 1, -1]]; 
        i := (sum-2)/2;
        retrec.colw := [1, i, i+1];
        retrec.full := true;
    elif size = 4 and col1 = [ 1, 1, 1, 1]  then
        retrec.mat := [ [1, 1, 1, 1], [1, 1, -1, -1],
                        [1, -1, -1, 1], [1, -1, 1, -1]];
        retrec.colw := [1, 1, 1, 1];
        retrec.full := true;
    elif size = 4 and sum > 4 and col1 = [ 1, 1, 1, sum-3]  then
        retrec.mat := [ [1, 1, 1, 1], [1, 1, E(3), E(3)^2],
                        [1, 1, E(3)^2, E(3)], [sum-3, -3, 0, 0]];
        i := sum/3;
        retrec.colw := [1, i-1, i, i];
        retrec.full := true;
    fi;

    # If the matrix is not complete, try to find proposals from the library.
    # when using ClmInit no split-version is expected
    if not retrec.full  then

        # sort first column of each matrix
        clmlist := LibraryTables( "clmelab" );
        l := ConcatenationString( "elab", String(retrec.size) );
        retrec.proposal := ["elab", retrec.size, []];

        if IsRec( clmlist ) and IsBound( clmlist.(l) )  then

	    # Compare (unsorted) first column of library matrix with 'col1'.
            for i in [ 1 .. Length( clmlist.(l) ) ] do
	      lmat := clmlist.(l)[i][1]; 
              if col1 = List( [1..retrec.size], x-> lmat[x][1] ) then 
                Add( retrec.proposal[3], i );
              fi;
            od;

            if retrec.proposal = [] then

              # Try to find library matrices that match up to
              # permutations of rows (and columns).
              Sort( col1 );
              for i in [1..Length( clmlist.(l) )] do

                # Compare first column of library matrix with 'col1'.
  	        lmat := clmlist.(l)[i][1]; 
                k := List( [1..retrec.size], x-> lmat[x][1] );
                Sort( k );
  	        if col1 = k  then 
	          Add( retrec.proposal[3], i );
                fi;
              od;

            fi;
        fi;
        if retrec.proposal[3] = [] then
          Unbind( retrec.proposal );
        fi;

    fi;

    return retrec;
    end;
 
################################################################################
CliffordRecords := function( Ti )

    local  clms, i;

    if not ForAll( Ti.tables, x-> IsCharTable( x ) ) then
        Error(" all Ti.tables must be explicit charactertables" );
    fi;

    if not IsBound( Ti.fusions )  or  ForAny( [2..Length( Ti.tables )], 
				  x -> not IsBound( Ti.fusions[x] ) )  then
        Ti.fusions := DetermFusions( Ti );
    fi;

    clms := [];
    for i in [1..Length( Ti.tables[1].classes )] do
        clms[i] := ClmInit ( i, Ti );
    od;

    return( clms );
end;	
################################################################################
##
#F  PrintCliffordRec( <clm> ) ............ prints the cliffordrecord as a record
##  is like 'PrintCharTable'
##
PrintCliffordRec := function( C )

    local  len, i, nam, lst, printRecIndent;
 
    if not IsCliffordRec( C ) then
        return( "#E PrintCliffordRec: input is not a cliffordrecord." );
    fi;

    printRecIndent := "  ";
    len  := 0;
    for nam in RecFields( C )  do
        if len < LengthString( nam )  then
            len := LengthString( nam );
        fi;
        lst := nam;
    od;
    Print( "rec(\n" );
    for nam  in RecFields( C )  do
        if not nam = "operations" then
            Print( printRecIndent, nam );
            for i  in [LengthString( nam )..len]  do
                Print( " " );
            od;
    	    if IsString( C.(nam) )  then
	        Print( ":= \"", C.(nam), "\"");
	    else
                Print( ":= ", C.(nam) );
	    fi;
            if nam <> lst  then  Print( ",\n" );  fi;
        else
            Print( printRecIndent, nam );
            for i  in [LengthString( nam )..len]  do
                Print( " " );
            od;
            Print( ":= CliffordRecordOps" );
            if nam <> lst  then Print( ",\n" );  fi;
        fi;
    od;
    Print( " )" );

end;
################################################################################
## start of procedures for CliffordTable
################################################################################
##
#F IsCliffordTable( <clms> ) ........................... is clms a cliffordtable
##
IsCliffordTable := function( C )

    return  IsRec( C ) and IsBound( C.size ) and IsBound( C.operations ) and
	    ForAll( [1..C.size], x -> IsBound( C.(x) ) and 
				      IsCliffordRec( C.(x) ) );
end;
################################################################################
##
#F  PrintCliffordTable( <clms> ) .......... prints the cliffordtable as a record
##  the function corresponds to 'PrintCharTable'
##
PrintCliffordTable := function( C )

    local  len, i, nam, lst, printRecIndent, RecSpecial, RecPrints, RecIgnore;
 
    if not IsCliffordTable( C )  then
       return( "#E PrintCliffordTable: Input is not a cliffordtable.\n" );
    fi;

    printRecIndent := "  ";
    RecSpecial := List( [1..Size( C )], x-> String( x ) );
    RecIgnore := Filtered( RecFields( C ), x -> IsRec( C.(x) ) and 
    		           IsBound( C.(x).elname ) and 
                           String( C.(x).elname ) = x );
    Add( RecIgnore, "charTable"); 
    RecPrints := Filtered( RecFields( C ), x -> not x in RecIgnore );

    len  := 0;
    for nam in RecPrints  do
        if len < LengthString( nam )  then
            len := LengthString( nam );
        fi;
        lst := nam;
    od;
    Print( "rec(\n" );
    for nam  in RecPrints  do
        if nam = "operations" then
            Print( printRecIndent, nam );
            for i  in [LengthString( nam )..len]  do
                Print( " " );
            od;
            Print( ":= CliffordTableOps" );
            if nam <> lst  then Print( ",\n" );  fi;
        elif  nam in RecSpecial  then
            Print( printRecIndent, nam );
            for i  in [LengthString( nam )..len]  do
                Print( " " );
            od;
            Print( ":= "); PrintCliffordRec( C.(nam) );
            if nam <> lst  then  Print( ",\n" );  fi;
        else
            Print( printRecIndent, nam );
            for i  in [LengthString( nam )..len]  do
                Print( " " );
            od;
            if IsString( C.(nam) )  then
    	        Print( " := \"", C.(nam), "\"");
    	    else
                Print( ":= ", C.(nam) );
    	    fi;
            if nam <> lst  then  Print( ",\n" );  fi;
        fi;
    od;
    Print( " )" );

end;

###############################################################################
##
#F  MakeHead( <table> [, rec( expN:= <expN>, powermap := true|false)] ) .......
##  ....... tries to establish the head of the new table, that is the powermaps
##
##  if N is elementary abelian then the element orders are nearly known
##  so then the orders can be filled.
##  Otherwise if exp(N) is known this might give a better bound for element
##  orders because the calculation of power maps can take a long time,
##  only do it when wished
##
##  *Note* that the argument <table> is changed by 'MakeHead'.
##
#T determine element orders from power maps:
#T work with parametrized maps
#T (instead of unknowns, as 'ElementOrdersPowermap' does)!
##
MakeHead := function ( arg )

    local i, j, table, cl,   # vector with the record of cliffordmatrices 
	ct,   	    # counter of classes
	expN, pmap, # exponent of the normal subgroup N and option: powermap?
	p,	    # factors of the order of N or, if given, its exponent
	lp,	    # Length of table.powermap
	tpp,        # table.powermap[p]
	clord,	    # order of element of regarded cl
	vecord,	    # vector of possible orders for one element (paramap)
	found,	    # found a powermap with unique power 
	ord;	    # orders of classes of new group

    if Length( arg ) > 2 or Length( arg ) < 1  or not IsCharTable( arg[1] ) or
       (IsBound( arg[2] ) and not IsRec( arg[2] )) then
        Error( "usage: MakeHead( <table> [, <options>] )"); 
    fi;
    table := arg[1];
    cl := table.cliffordTable; 
    pmap := true;
    if IsBound( cl.expN )  then
        expN := cl.expN;
    else
        expN := Sum( [1..cl.1.size], x -> cl.1.mat[x][1] );
    fi;
    if IsBound( arg[2] ) then
        if IsBound( arg[2].expN ) then
            expN := arg[2].expN;
        fi;
        if IsBound( arg[2].powermap ) then
            pmap := arg[2].powermap;
        fi;
    fi;
    # case: N elementary abelian
    p := Factors( expN )[1];
    # calculate orders or list of possible orders
    ct := 0;
    for i in [1..cl.size] do

        clord := cl.(i).order;
        for j in [1..cl.(i).size] do 
            ct := ct + 1;

            if not IsBound( table.orders[ct] ) then
            # p divides the elementorder table.orders[ct]

	        if EuclideanRemainder( Integers, clord, p ) = 0  then 
	            vecord := [ clord ];
	            ord := clord*p;
	 	    while ord <= expN*clord and EuclideanRemainder( 
				Integers, table.centralizers[ct], p ) = 0  do
		        vecord[Length( vecord )+1] := ord;
		        ord := ord*p;
		    od;
		    table.orders[ct] := Copy( vecord );
	        else # Schur-Zassenhaus 
		    table.orders[ct] := [ clord*p ];
		    ord := Copy( p );
		    while ord < expN  do
		        ord := ord * p;
		        table.orders[ct][Length( table.orders[ct] )+1] 
                                                            := clord * ord;
		    od;
		    if Length( table.orders[ct] ) = 1  then
		        table.orders[ct] := table.orders[ct][1];
		    fi;
	        fi;

	    else	# test order against centralizer order

  	        if IsInt( table.orders[ct] ) then
	            if EuclideanRemainder( Integers, table.centralizers[ct],
		    	        table.orders[ct] ) <> 0 then
		        Print( "#E MakeHead: elementorder of class ", ct, 
                               " does not divide centralizerorder!\n" );
	            fi;
	        elif IsList( table.orders[ct] )  then
	            j := Copy( Length( table.orders[ct] ) );
	            for i in [1..j] do
	                if EuclideanRemainder( Integers, table.centralizers[ct],
		  	            table.orders[ct][j-i+1] ) <> 0 then
		            Unbind(table.orders[ct][j-i+1] );
		        fi;
	            od;
	            if Length( table.orders[ct] ) = 1 then
		        table.orders[ct] := table.orders[ct][1];
	            fi;
	        else
	    	    Error( "element order of class ", ct, " not valid!");
	        fi;

            fi;

        od;
    od;

    InitClassesCharTable( table );

 #  ord := Set( Factors( table.size ) );
    ord := Union( Set( Factors( table.size ) ),
                  Filtered( [ 1 .. Length( table.powermap ) ],
                            x -> IsBound( table.powermap[x] ) ) );

    # try to establish the power maps for G if 'pmap = true'
    if pmap then
        for i in ord  do
            if not IsBound( table.powermap[i] ) or
                IsBound( table.powermap[i]) and table.powermap[i] = []  then
                table.powermap[i] := Parametrized( Powermap( table, i, 
		  		                   rec( quick:=true) ));
            else
                table.powermap[i] := Parametrized( Powermap( table, i, 
		    	    rec( powermap := table.powermap[i], quick:=true) ));
            fi;
        od; 
    fi;

    # try to turn paramaps of element orders into unique integer
    if IsBound( table.powermap ) then
        tpp := table.powermap[p];
        if IsList( tpp ) and tpp <> []  then
            for i in [1..Length(table.orders )] do
                if IsList( table.orders[i] )  then
                # take p-th powermap to restrict the possibilities 
                # for the element orders

                    if IsList( tpp[i] )  then
	                clord := [];
    	                for j in [1..Length(tpp[i])] do
	  	            if IsInt( table.orders[tpp[i][j]] ) then
		                clord[Length(clord)+1] := 
                                                       table.orders[tpp[i][j]];
		            else
		                Append( clord, table.orders[tpp[i][j]]);
		            fi;
	                od;
	                clord := Set( clord );
	                if Length( clord ) = 1  then clord := clord[1]; fi;
	            else
	                clord := Copy( table.orders[tpp[i]] );
	            fi;
	            if IsInt( tpp[i] ) and tpp[tpp[i]] = i then 
	                # p-th power of p-th power is same class
	  	        clord := Filtered( table.orders[i], 
                                           x -> not p in Factors( x ));
	            elif IsList( clord )  then
	                if ForAll( [1..Length(clord)], 
                                    x-> p in Factors(clord[x])) then 
	                    # p | clord, muss potenz.
	                    clord := Set( clord*p );
	                else 
	                    Append( clord, clord*p );
	                    clord := Set( clord );
	                fi;
	            else # IsInt( clord )  then
	                if p in Factors( clord ) then   clord := clord*p;	
	                else   clord := [clord, clord*p];
	                fi;
	            fi;

	            if IsList( clord )  then
	                IntersectSet( clord, table.orders[i] ); 
	                if clord <> []  then
          	            if Length( clord ) = 1  then 
                                clord := clord[1]; 
                            fi;
	                    table.orders[i] := clord;
	                else
	  	            Print( "#E MakeHead: regarding powermap[p] no",
                                   " order for class ", i, "!\n");
	                fi;
	            else
	                if clord in table.orders[i]  then
	                    table.orders[i] := clord;
	                else
	  	            Print( "#E MakeHead: regarding powermap[p] no",
                                   " order for class ", i, "!\n");
	                fi;
	            fi;
                fi;
            od;
        fi;

        # try to reduce the possibilities of the elementorders by the
        # powermaps of table all primes in ord

        for j in ord do
            if IsBound( table.powermap[j] )  then
                tpp := table.powermap[j];
                if IsList( tpp ) and tpp <> []  then
                    for i in [1..Length( table.orders )] do
                        if IsList( table.orders[i] ) and IsInt( tpp[i] )  then

                            # take j-th powermap to restrict the 
                            # possibilities for the elementorders
	                    clord := Copy( table.orders[tpp[i]] );
	                    if tpp[tpp[i]] = i then 
	                        # j-th power of j-th power is same class
		                clord := Filtered( table.orders[i], 
		    	                           x -> not j in Factors( x ));
	                    elif IsList( clord )  then
	                        if ForAll( [1..Length(clord)], 
			                   x-> j in Factors(clord[x])) then
	                        # j | clord, muss potenz.
	                           clord := Set( clord*j );
	                        else 
	                            Append( clord, clord*j );
	                            clord := Set( clord );
	                        fi;
	                    elif IsInt( clord )  then
	                        if j in Factors( clord ) then 
                                    clord := clord*j;	
	                        else  
                                    clord := [clord, clord*j];
	                        fi;
	                    fi;

	                    if IsList( clord )  then
	                        IntersectSet( clord, table.orders[i] ); 
	                        if clord <> []  then
            	                    if Length( clord ) = 1  then 
                                        clord := clord[1]; 
                                    fi;
	                            table.orders[i] := clord;
	                        else
	    	                    Print( "#E MakeHead: regarding ",
                                           "powermap[", j, "] no order ",
                                           "for class ", i, "!\n" );
	                        fi;
	                    else
	                        if clord in table.orders[i]  then
	                            table.orders[i] := clord;
	                        else
	    	                    Print( "#E MakeHead: regarding ",
                                           "powermap[", j, "] no order ",
                                           "for class ", i, "!\n" );
	                        fi;
	                    fi;
                        fi;
                    od;
                fi;
            fi;
        od;
    fi; # tbl.powermap is bound

end;

################################################################################
## 
CliffordTableOps := OperationsRecord( "CliffordTableOps", CharTableOps );
################################################################################
CliffordTableOps.\in := function( c, C )

    if IsBound( c.elname )  then
        return  c.elname in Elements(C);
    else
        Error( "<elname> must be bound in <c>" );
    fi;
end;
################################################################################
##  creates the charactertable out of cliffordtable with 
##  the complete cliffordrecords
##
CliffordTableOps.CharTable := function( clms )

     local  i, j, k, xx, table, name, Ti, ClmMultOne,
	extrec,	# record of the extension
	multerg,# result of multiplication "clmult"
	centr,	# centralizerorders
	class,	# classlengths
	ct,	# counter of the actually calculated class, as it is columnwise
		#    calculated
	fusmap,	# fusion of into factor group G/N
	pmap,	# powermap as far as known from the powermap of G/N
	p,	# index of the class in G/N on that another is mapping on
	os,	# known elementorders
	irr, 	# the irreducibles of the "big" group G 
	fname;	# name of first inertiagrp for fusion

###############################################################################
##  multiplies the cliffordmatrix <clm> with the charactertables of the
##  inertia groups
##
##  Ti is a record with the explicit charactertables, their idents and fusions
##  clm is a cliffordrecord or a "cll"-version
##
    ClmMultOne := function( Ti, clm )

    local  newct,	    # new CharTable.irreducibles 
           centr,           # centralizer of new classes 
           ct,    	    # CharTable[inertiagroup].irreducibles
      	   colct, rowct,    #counter for row and column of new CharTable
     	   j, n, klind, lind, 	    # indices
	   AnzTi,	    # number of inertia groups
	   AnzClm,	    # number of Cliffordmatrices
	   l, clmlist,      # list of cliffordmatrices of the library
 	   clmexp,   # expanded cliffordmatrix-record, if clm is library version
      	   clmat, 
	   ind,	            #index of the clm in the library 
      	   sum, quo;        #helpvariables

    centr := []; newct := []; 
    AnzTi := Length( Ti.tables );

    # for factor no projective characters possible
    #
    ct := [Ti.tables[1].irreducibles];
    for j in [2..AnzTi] do	
        if Ti.ident[j][1] = "projectives"  then
        # search characters for name in ident[j][3]
            ind := PositionProperty( [1..Length( Ti.tables[j].projectives )],
                                    x -> Ti.tables[j].projectives[x].name =
                                         Ti.ident[j][3] );
            ct[j] := Ti.tables[j].projectives[ind].chars;
        else
            ct[j] := Ti.tables[j].irreducibles;
        fi;
    od;
    n := Sum( [1..AnzTi], x -> Length( ct[x] ) );

    for rowct in [1..n] do
        newct[rowct] := [];
    od;
    colct := 0;

    clmexp := Copy( clm );
    # if cliffordmatrix is in the library, get it and calculate necessary data
    # if the size of the matrix is <= 2, automatic calculation

    if not IsCliffordRec( clmexp ) then #cll-version expected
        clm := CllToClf( Ti, clmexp );
    else	# clmexp is explicitly filled record
        clmat := Copy( clmexp.mat );
    fi;

    # calculate then centralizerorders and the part of the irreducibles
    #
    if IsBound( clmexp.splitinfos )  then
        if clmexp.splitinfos.classindex > 0  then # splitted class
            sum := Sum ( [1..clmexp.size-clmexp.splitinfos.numclasses+1], 
	    	         x -> clmat[x][1] );
        else # in a splitversion, only columnweights are different
            sum := Sum ( [1..clmexp.size], x -> clmat[x][1] );
        fi;
        quo := Ti.tables[1].centralizers[clmexp.nr] * clmexp.splitinfos.p * sum;
    else
        quo := Ti.tables[1].centralizers[clmexp.nr] 
    	        * Sum ( [1..clmexp.size], x -> clmat[x][1] );
    fi;
    for n in [1..clmexp.size] do	#for all columns in the matrix
        rowct := 0;
        colct := colct + 1;

    #
    # calculate the entries [rowct,colct] in the newct
    # for first inertia group special treatment because of the special structure
    #
        for klind in [1..Length( ct[1] )] do	#for all classes 
            rowct := rowct + 1;
            newct [rowct][colct] := clmat[1][n] * ct[1] [klind][clmexp.nr];
        od;
        for j in [2..AnzTi] do	# for all inertiagroups 
            for klind in [1..Length( ct[j] )] do	#for all classes 
                rowct := rowct + 1;
                newct[rowct][colct] :=  Sum( Filtered( [2..clmexp.size],
                                     ll -> clmexp.inertiagrps[ll] = j ),
	       x -> clmat[x][n] * ct[j][klind][clmexp.fusionclasses[x]] );
            od;
        od;
        centr[colct] := quo / clmexp.colw[n];
    od;

    return [ centr, newct ];
    end;

## now start CharTable
##
    if not IsCliffordTable( clms )  then
        Error( "usage: CharTable( cliffordtable )" );
    fi;	

    # test whether the character tables are explicitly given and get them
    # if not explicitly given, in Ti.ident the argument for CharTableLibrary is
    # bound from CliffordTable

    if not IsBound( clms.Ti.ident ) then
        Print("#I CharTable: chartables of inertiagroups only recognizable by ",
    	      "identifier. \n");
        Ti := rec( tables:=[], ident := [] );
        for i in [1..Length(clms.Ti.tables)] do
            if IsCharTable( clms.Ti.tables[i] )  then
    	        Ti.tables[i] := clms.Ti.tables[i];
                Ti.ident[i] := clms.Ti.tables[i].identifier;
            else
    	        Error( "no identifier and no table for inertiagrp no.", i );
            fi;
        od;
    else
        Ti := rec( tables:=[], ident := ShallowCopy( clms.Ti.ident ) );
        k := Maximum( Length( Ti.ident), Length(  clms.Ti.tables ) );
        for i in [1..k] do
            if IsBound( clms.Ti.tables[i] ) and 
	       IsCharTable( clms.Ti.tables[i] ) then
	        Ti.tables[i] := clms.Ti.tables[i];
	        if not IsBound( Ti.ident[i] ) then
	          Ti.ident[i] := [ Ti.tables[i].identifier ]; fi;
            else
	        if IsBound( Ti.ident[i] ) then
	            Ti.tables[i] := CharTableLibrary( Ti.ident[i] );
	        else 
                    Error( "no identifier and no table for inertiagrp no.", i );
	        fi;
            fi;
        od;
    fi;
 
    # special treatment for first clm
    multerg := ClmMultOne( Ti, clms.1 );
    centr := multerg[1]; irr := multerg[2];
    class := []; fusmap := [1]; pmap := [];

    for i in [2..Length( Ti.tables[1].powermap )] do
        if IsBound( Ti.tables[1].powermap[i] )  then
            pmap[i] := [1];
        fi;
    od;

    ct := 1;
    for i in [2..clms.1.size] do
        ct:= ct + 1;
        fusmap[ct] := 1;
        for j in [2..Length( Ti.tables[1].powermap )] do
            if IsBound( Ti.tables[1].powermap[j] )  then
        	if IsBound( Ti.expN ) and Ti.expN = j  then
    	            pmap[j][i] := 1;
  	        else
	  	    pmap[j][i] := [1..clms.1.size];
	        fi;
            fi;
        od;
    od;

    if IsBound( clms.1.orders )  then os := Copy( clms.1.orders );
    else                              os := [1];
    fi;
    for i in [1..ct] do
        class[i] := multerg[1][1] / centr[i];
    od;

    for i in [2..clms.size] do
    # elementorders as far as known from elsewhere
        if IsBound( clms.(i).orders )  then
            for j in [1..Length( clms.(i).orders )] do
	        if IsBound( clms.(i).orders[j] ) then
                    os[ct+j] := clms.(i).orders[j];
	        fi;
            od;
        else
            os[ct+1] := Ti.tables[1].orders[i];
        fi;

        # multiply for one cliffordmatrix, get part of the charactertable and 
        # the centralizerorders
        multerg := ClmMultOne( Ti, clms.(i) );
        Append( centr, multerg[1] );

        # append the new part to each character
        for j in [1..Length( multerg[2] )] do
  	    Append( irr[j], multerg[2][j] );
        od;
        for j in [1..clms.(i).size] do
            class[ct+j] := centr[1] / multerg[1][j];
            fusmap[ct+j] := i;
            # a part of the powermap is known from the powermap of Ti.tables[1]
            for k in [2..Length( Ti.tables[1].powermap )]  do
	        if IsBound( Ti.tables[1].powermap[k] ) then
	            p := Ti.tables[1].powermap[k][i];
	      	    xx := Sum( [1..p-1], x -> clms.(x).size ) + 1;
	            if IsBound( clms.(i).splitinfos ) or j > 1  then  
			# powermap of first class of a clm is known
	    	        pmap[k][ct+j] := [xx..xx+clms.(p).size-1];
	            else 
		        pmap[k][ct+1] := xx;
		    fi;
  	        fi;
	    od;
        od;
        ct := ct + clms.(i).size;
    od;

    # only the name or identifier of the inertiagroups must be kept, not the 
    # whole table
    ct := Copy( clms );
    ct.Ti.ident := Copy( Ti.ident );

    if IsBound( ct.Ti.tables[1].identifier ) then 
        fname := ct.Ti.tables[1].identifier;
    else
        fname := ct.Ti.tables[1].name;
    fi;
    Unbind( ct.Ti.tables );

    table := rec (identifier := ct.grpname, centralizers := centr, 
                  size := centr[1],
  	          classes := class, orders:= os, powermap := pmap,
                  irreducibles := irr, operations := CharTableOps,
	          fusions := [rec(name := fname,map := fusmap,type :="factor")],
                  text := ConcatenationString( "table computed with ",
                                               clms.name ),
  	          cliffordTable := ct, construction := function( tbl )
                                       ConstructClifford( tbl ); end);

    if IsBound( clms.expN )  then
        MakeHead( table, rec(expN := clms.expN, powermap := false) );
    elif IsBound( clms.Ti.expN )  then
        MakeHead( table, rec(expN := clms.Ti.expN, powermap := false) );
    else
        MakeHead(table, rec(powermap := false) );
    fi;

    return table;
end;

################################################################################
##
#F  CliffordTable( <Ti> ) create the domain, the records of the cliffordmatrices
##
CliffordTable := function( Ti )
################################################################################
local 	i, C, elements, fname;	

    C := rec( isDomain := true, Ti := Copy( Ti ));

    # keep the information about the tables how they are given for library 
    # or reconstruction and get explicit chartables
    #
    C.Ti.ident := [];
    for i in [1..Length(Ti.tables)] do

        if IsCharTable(Ti.tables[i]) then
            C.Ti.ident[i] := [Ti.tables[i].identifier];
        elif IsString( Ti.tables[i] ) then
            C.Ti.ident[i] := [Copy( C.Ti.tables[i] )];
            C.Ti.tables[i] := CharTable( C.Ti.tables[i] );
        elif IsList( Ti.tables[i] ) then
            C.Ti.ident[i] := Copy( C.Ti.tables[i] );
            if Ti.tables[i][1] = "projectives"  then # proj. characters
                if i = 1 then	
                   Print( "#E CliffordTable: projective characters for first ",
                          "inertia factor?? Usual ones taken. \n" );
                   C.Ti.ident := [ C.Ti.ident[1][2] ];
                fi; 
                C.Ti.tables[i] := CharTable( C.Ti.tables[i][2] );
            else
                C.Ti.tables[i] := CharTableLibrary( C.Ti.tables[i] );
            fi;
        else
            Error("Ti.tables[i] must be a string, a list or a charactertable");
        fi;

    od;

    elements := CliffordRecords( C.Ti );
    C.size := Length( elements );
    for i in [1..C.size] do
        C.(i) := Copy( elements[i] );
        C.(elements[i].elname) := C.(i);
    od;
    C.elements := List( [1..C.size], x-> C.(x).elname );

    if not IsBound(Ti.grpname)  then
        C.grpname := "??";
    else
        C.grpname := Copy( Ti.grpname );
        Unbind( C.Ti.grpname );
    fi;
    if IsBound( Ti.expN )  then
        C.expN := Copy(C.Ti.expN);
        Unbind( Ti.expN );
    fi;
    if IsBound(C.Ti.tables[1].identifier )  then
        fname := C.Ti.tables[1].identifier;
    else
        fname := C.Ti.tables[1].name;
    fi;
    C.name := ConcatenationString("CliffordTable( ", fname, " -> ", 
					          C.grpname, " )" );
    C.operations := CliffordTableOps;

    return C;
end;

################################################################################
##
#F  SplitClass( <clm>, <p>, <indTi> [, rec( colindex, newtable, fusion, 
##                                          numclasses, rt )] )
##  splits the <i>-th column of the cliffordmatrix <cl> and changes colw
##  if no columnindex is given, the first column is splitted
##  because the new inertiafactorgroup must be added into the record, the index
##  <indTi> in Ti must be given.
##  Because of the possible splitting of 1x1-matrices p must be given, <p> then
##  is not known from the matrix
##  <rt> is the root that shall be taken for splitting
##
SplitClass := function( arg )

    local  i, j, k, p, newct,
	ind, nc,   # colindex and number of classes coming of splitting class
        fus, fusclasses,#find fusionclasses for new ti
	sum,fac,   # Sum of first column and its factors
        rt, rts2, rts3, # roots for p = 2,3
        irr, list,
	rnxi,	# root of sum 
	clsp;	# new cliffordmatrixrecord

    rts2 := [ 1-E(4), 2, 2-2*E(4), 4, 4-4*E(4), 16, 16-16*E(4), 32, 32-32*E(4), 
              64, 64-64*E(4), 128, 128-128*E(4) ];
    rts3 := [ 1-E(3), 3, 3-3*E(3), 9, 9-9*E(3), 27, 27-27*E(3), 81, 81-81*E(3), 
              243, 243-243*E(3), 729, 729-729*E(3) ];

    if Length( arg ) < 3 or Length( arg ) > 4 or 
       (IsBound( arg[4] ) and not IsRec( arg[4] ) ) or 
       not (IsCliffordRec( arg[1] ) and IsInt( arg[2] ) and IsInt( arg[3] ) ) 
       then
        Error( "usage: SplitClass( clm, p, indTi [,rec( classindex, ",
               "numclasses, root, newtable, fusion )] )" );
    fi;
    if IsBound( arg[4] ) and IsBound( arg[4].classindex ) then 
        ind := arg[4].classindex; 
    else
        ind := 1;
    fi;
    if IsBound( arg[4] ) and IsBound( arg[4].root )  then 
        rnxi  := arg[4].root; 
    else
        if ind > 0  then  #otherwise rnxi not necessary
            sum := Sum( [1..arg[1].size], x -> arg[1].mat[x][1] );
            fac := Factors( sum );
            if not (ForAll(fac, x-> x = arg[2]) or fac = [1])  then
               Error( "sum of first column is not a power of p" );
            fi;
            if fac = [1] then rnxi := 1;
            else
                if arg[2] = 3  then rnxi := rts3[Length(fac)];
                else                rnxi := rts2[Length(fac)]; #p = 2
                fi;
            fi;
        fi;
    fi;
    if IsBound( arg[4] ) and IsBound( arg[4].numclasses ) then 
        nc  := arg[4].numclasses; 
        if nc = 1 then ind := 0; fi;  #from one to one class, i.e. not splitting
    else
        nc := arg[2];
    fi;

    if IsBound( arg[4] ) and IsBound( arg[4].newtable ) then 

        newct := ShallowCopy( arg[4].newtable ); 
        # which kind of input
        if not IsCharTable( newct ) then
            if IsString( newct ) then
                newct := CharTable( newct );
            elif IsList( newct )  then
                if newct[1] = "projectives"  then
                    list := Copy( newct );
                    newct := CharTable( list[2] );
                    i := PositionProperty( [1..Length( newct.projectives )],
                                 x -> newct.projectives[x].name = list[3] );
                    irr := newct.projectives[i].chars;
                else
                    newct := CharTableLibrary( newct );
                fi;
            else
                Error( "option <newtable> is no correct input " );
            fi;
        fi;

        if IsBound( arg[4].fusion ) then
            fus := arg[4].fusion;
            if IsBound( irr ) then
                fusclasses := Filtered( [1..Length( fus )], 
                        x -> fus[x] = arg[1].fusionclasses[1]
                             and ForAny( irr, y -> y[x] <> 0 ) );
            else
                fusclasses := Filtered( [1..Length( fus )], 
                        x -> fus[x] = arg[1].fusionclasses[1] );
            fi;
            if Length( fusclasses ) = 0 then ind := 0; fi;# not splitting
            if ind > 0 and Length( fusclasses ) <> nc-1 then
                if IsBound( arg[4].numclasses ) then
                    Print( "#E SplitClass: Number of classes fusioning in ",
                           "splitclass not equal number of new classes. \n" );
                else
                    nc := Length(fusclasses)+1;
                fi;
            fi;
        else
           Error( "fusion from newtable into factor group is missing" );
        fi;
    else
        fusclasses := [arg[1].fusionclasses[1]];
    fi;

    if not arg[2] in [2, 3]  then
        Error( "sorry, for p > 3 not implemented" );
    fi;
    clsp := Copy( arg[1] );
    if ind = 0  then

        clsp.splitinfos := rec( p:= arg[2], classindex := ind );
        clsp.colw := arg[2] * arg[1].colw;
        # elementorders can change
        if IsBound( arg[1].eorders ) then
            clsp.orders := arg[1].eorders;
        elif IsBound( clsp.orders ) then
	    for i in [1..Length( clsp.orders )] do
	        if IsBound( clsp.orders[i] )  then
	            clsp.orders[i] := [arg[1].orders[i], 
                                       arg[2] * arg[1].orders[i]];
	        fi;
	    od;
        fi;

    else

        clsp.splitinfos := rec( p:= arg[2], classindex := ind, 
                                numclasses := nc, root := rnxi );
        clsp.size := clsp.size + nc-1;
        clsp.mat[clsp.size] := [];
        clsp.fusionclasses[clsp.size] := fusclasses[Length( fusclasses )]; 
        clsp.inertiagrps[clsp.size] := arg[3];
        clsp.roww[clsp.size] := clsp.roww[1];

        if arg[2] = 3  then
            if nc = 3  then
                clsp.mat[clsp.size-1] := [];
                clsp.fusionclasses[clsp.size-1] := fusclasses[1]; 
                clsp.inertiagrps[clsp.size-1] := arg[3];
                clsp.roww[clsp.size-1] := clsp.roww[1];
            else
                clsp.roww[clsp.size] := clsp.roww[1] / 2;
            fi;
        fi;

        clsp.colw[ind] := arg[1].colw[ind];
        clsp.colw[ind+1] :=  arg[1].colw[ind];
        for j in [1..ind-1] do    #correct columns lower than ind
            clsp.colw[j] := arg[2] * arg[1].colw[j];
            clsp.mat[clsp.size][j] := 0;
        od;
        # split ind-column
        for j in [1..arg[1].size] do
            clsp.mat[j][ind+1] := arg[1].mat[j][ind];
        od;
        for j in [ind+1..arg[1].size] do   # move columns higher than ind
            clsp.colw[j+nc-1] := arg[2] *arg[1].colw[j];
            for k in [1..arg[1].size] do	# copy column
                clsp.mat[k][j+nc-1] := arg[1].mat[k][j];
            od;
            clsp.mat[clsp.size][j+nc-1] := 0;
        od;

        if arg[2] = 2  then
            clsp.mat[clsp.size][ind] :=  rnxi;
            clsp.mat[clsp.size][ind+1] := -rnxi;
        elif arg[2] = 3  then
            if nc = 3  then
                clsp.colw[ind+2] :=  arg[1].colw[ind];
                clsp.mat[clsp.size-1][ind] := rnxi;
                clsp.mat[clsp.size][ind] :=  GaloisCyc( rnxi, -1 );
                clsp.mat[clsp.size-1][ind+1] := E(3) * rnxi;
                clsp.mat[clsp.size-1][ind+2] := E(3)^2 * rnxi;
                clsp.mat[clsp.size][ind+1] := GaloisCyc( E(3) * rnxi, -1 );
                clsp.mat[clsp.size][ind+2] := GaloisCyc( E(3)^2 * rnxi, -1 ); 
                for j in [1..arg[1].size] do
    	            clsp.mat[j][ind+2] := arg[1].mat[j][ind];
                od;
                for j in [ind+1..arg[1].size] do   # fill second additional row 
   	            clsp.mat[clsp.size-1][j+2] := 0;
   	        od;
            else
                clsp.colw[ind+1] := 2*clsp.colw[ind+1];
                clsp.mat[clsp.size][ind]  := 2*rnxi;
                clsp.mat[clsp.size][ind+1] := -rnxi;
            fi;
        fi;

        if IsBound( arg[1].eorders ) then
            for j in [1..ind-1]  do
  	        if IsBound( arg[1].eorders[j] )  then
	            clsp.orders[j] := arg[1].eorders[j]; 
	        fi;
            od;
            for j in [ind+1..arg[1].size]  do
  	        if IsBound( arg[1].eorders[j] )  then
	            clsp.orders[j+nc-1] := arg[1].eorders[j]; 
	        fi;
            od;
        elif IsBound( arg[1].orders )  then
            for j in [1..ind-1]  do
  	        if IsBound( arg[1].orders[j] )  then
	            clsp.orders[j] := Set( Flat( [arg[1].orders[j], 
                                           arg[1].orders[j]*arg[2]] ));
	        fi;
            od;
            for j in [ind+1..arg[1].size]  do
  	        if IsBound( arg[1].orders[j] )  then
	            clsp.orders[j+nc-1] := Set( Flat( [arg[1].orders[j], 
		    		                arg[1].orders[j]*arg[2]] ));
	        fi;
            od;
        fi;

        if IsBound( arg[1].orders ) and IsBound( arg[1].orders[ind] ) then
            if arg[1].orders[ind] = 1 then
                clsp.orders[ind] := 1; 
                clsp.orders[ind+1] := arg[2];
                if arg[2] = 3 and nc = 3 then
                    clsp.orders[ind+2] := arg[2];
                fi;
            else
                clsp.orders[ind] := Set(Flat( [arg[1].orders[ind], 
                 	 	               arg[1].orders[ind]*arg[2]] ));
                clsp.orders[ind+1] := Set(Flat( [arg[1].orders[ind], 
    			                       arg[2] * arg[1].orders[ind]] ));
                if arg[2] = 3 and nc = 3 then
                    clsp.orders[ind+2] := Set(Flat( [arg[1].orders[ind], 
	                                       arg[2] * arg[1].orders[ind]] ));
                fi;
            fi;
        fi;

    fi;

    if IsBound( arg[1].proposal ) then
        Unbind( clsp.proposal );
    fi;

    return clsp;
end;
################################################################################
##
#F  SplitCliffordTable( <clms>, <p>, <newct> [, rec( roots, fusion )] )
##  takes the cliffordtable <clms> and splits in dependeance of the splitlist
##  the matrices of the cliffordrecords 
##  Additionally some informations in Ti and other items of <clt> are changed
##  spllist must be a record, the fourth argument of SplitClass
## 
SplitCliffordTable := function( arg )

    local  i, p, tiind, cls, newct, t1ct, fusion, roots, els, grpname;	

    if not Length( arg ) in [3,4] or not IsCliffordTable( arg[1] )  then
        Error( "usage: SplitCliffordTable( clifftable, p, newct [, rec( roots",
               ", fusion )] ) ");
    fi;

    cls := Copy( arg[1] );
    newct := arg[3];
    p := arg[2];

    if Length( arg ) = 4  then
        if IsBound( arg[4].roots ) then
            roots := arg[4].roots;
        fi;
        if IsBound( arg[4].fusion ) then
            cls.Ti.fusions[Length(cls.Ti.fusions) + 1] := arg[4].fusion;
        fi;
    fi;

    # find new Cliffordtable-name
    i:= Position( List( cls.name ), ">" )+2;
    grpname := SubString( cls.name, i, i+4 );
    if cls.grpname{[1..5]} = grpname  then
        cls.grpname:= ConcatenationString( String( p ), "^1+", 
                                  cls.grpname{[3..Length( cls.grpname )]} ); 
        cls.name := ConcatenationString( 
                    SubString( cls.name, 1, Length( cls.name)-1 ), " -> ", 
                    cls.grpname, " )" );
    else
        cls.name := ConcatenationString( 
                          SubString( cls.name, 1, Length( cls.name)-1 ),
                          " -> ", cls.grpname, " )" );
    fi;

    # find the charactertables
    #
    tiind := Length( cls.Ti.tables )+1; 
    if IsCharTable( newct ) then
       cls.Ti.tables[tiind] := ShallowCopy( newct );
       cls.Ti.ident[tiind] := newct.identifier;
    elif IsString( newct )  then
       cls.Ti.ident[tiind] := [Copy( newct )];
       cls.Ti.tables[tiind] := CharTable( newct );
    elif IsList( newct )  then
       cls.Ti.ident[tiind] := Copy( newct );
       if newct[1] = "projectives"  then
           cls.Ti.tables[tiind] := CharTable( newct[2] );
       else
           cls.Ti.tables[tiind] := CharTableLibrary( newct );
       fi;
    else
        Error( "<newct> must be a charactertable, a name or a list " );
    fi;

    # determine fusion for new ifg, otherwise in SplitClass does it each call
    if not IsBound( arg[4] ) or 
        IsBound( arg[4] ) and not IsBound( arg[4].fusion ) then
        t1ct:= Copy( cls.Ti.tables[1] );
        if not IsCharTable( t1ct ) then
            if IsString( t1ct )  then
                t1ct := CharTable( t1ct );
            elif IsList( t1ct )  then
                if t1ct[1] = "projectives"  then
                    t1ct := CharTable( t1ct[2] );
                else
                    t1ct := CharTableLibrary( t1ct );
                fi;
            else
                 Error( "<Ti.tables[1]> must be a table, a name or a list " );
            fi;
        fi;
        fusion := SubgroupFusions( cls.Ti.tables[tiind], t1ct );
        if Length( fusion ) = 0 or Length( fusion ) > 1 then
            Error( "Fusion from ", cls.Ti.tables[tiind].identifier, 
                   " into ", t1ct.identifier, " is missing and not unique." );
        else
            cls.Ti.fusions[tiind] := fusion[1];
        fi;
    fi;

    # because SplitClass copies cliffordrecord, split only, combine later
    els := Elements( cls );
    for i in els  do
        Unbind( cls.(i) );
    od;

    for i in [1..Size(cls)] do
        if IsBound( roots ) and IsBound( roots[i] )  then
            cls.(i) := SplitClass( arg[1].(i), p, tiind, 
                                   rec( newtable := newct,
                                        fusion := cls.Ti.fusions[tiind], 
                                        root := roots[i] ) );
        else
            cls.(i) := SplitClass( arg[1].(i), p, tiind, 
                                   rec( newtable := newct,
                                        fusion := cls.Ti.fusions[tiind] ) ); 
        fi;
        cls.(els[i]) := cls.(i); 
    od;

    # if charTable is bound, then it is the one of not-splitted
    Unbind( cls.charTable );
    if newct[1] = "projectives"  then
       cls.Ti.ident[tiind] := [newct[1], [newct[2]], newct[3]];
    fi;
    if IsBound( cls.expN ) then
       cls.expN := cls.expN * arg[2];
    fi;

    return( cls );
end;
