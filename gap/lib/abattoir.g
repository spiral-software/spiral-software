# -*- Mode: shell-script -*-

# changes for spiral:
# - look for packages in GAP_DIR/..
#   (search for PKGNAME1)

#############################################################################
##
#A  abattoir.g                  GAP library                  Martin Schoenert
##
##
#Y  Copyright (C) 2018-2021, Carnegie Mellon University
#Y  All rights reserved.  See LICENSE for details.
#Y  
#Y  This work is based on GAP version 3, with some files from version 4.  GAP is
#Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
##
##  This file is only there to catch some loose ends.
##
##

GlobalPackage(gap.string);

#############################################################################
##
#F  LengthString(<string>)  . . . . . . . . . . . . . . .  length of a string
##
##  'LengthString' is obsolete since strings a lists.
##
LengthString := Length;


#############################################################################
##
#F  SubString(<args>) . . . . . . . . . . . . . . . . . substring of a string
##
##  'SubString' is obsolete since strings are lists.
##
SubString := function ( arg )
    local    string, from, to;
    string := arg[1];
    if not IsString( string )  then
        Error("usage: SubString( <string>, <from> [, <to>] )");
    fi;
    from := arg[2];
    if not IsInt( from )  then
        Error("usage: SubString( <string>, <from> [, <to>] )");
    fi;
    if Length(arg) = 3  then
        to := arg[3];
    else
        to := Length(string);
    fi;
    if to < from  then return "";  fi;
    if from <= 0  then  from := 1;  fi;
    if Length(string) < from   then from := Length(string);  fi;
    if to   <= 0  then  to   := 1;  fi;
    if Length(string) < to     then to   := Length(string);  fi;
    return string{[from..to]};
end;


#############################################################################
##
#F  ConcatenationString(<string>,..)  . . . . . . .  concatenation of strings
##
##  'ConcatenationString' is obsolete since strings are lists.
##
ConcatenationString := function ( arg )
    local   res,  str;
    res := "";
    for str  in arg  do
        Append( res, str );
    od;
    IsString( res );
    return res;
end;

EndPackage();

#############################################################################
##
#F  Edit(<name>)  . . . . . . . . . . . . . . . . . . . . . . . . edit a file
##
if not IsBound( EDITOR )  then EDITOR := "vi";  fi;

Edit := function ( name )
    Exec( ConcatenationString( EDITOR, " ", name ) );
    Read( name );
end;


#############################################################################
##
#F  ProductPol( <f>, <g> )  . . . . . . . . . . .  product of two polynomials
##
ProductPol := function ( f, g )
    local  prod,  q,  m,  n,  i,  k;
    m := Length(f);  while 1 < m  and f[m] = 0  do m := m-1;  od;
    n := Length(g);  while 1 < n  and g[n] = 0  do n := n-1;  od;
    prod := [];
    for i  in [ 2 .. m+n ]  do
        q := 0;
        for k  in [ Maximum(1,i-n) .. Minimum(m,i-1) ]  do
            q := q + f[k] * g[i-k];
        od;
        prod[i-1] := q;
    od;
    return prod;
end;


#############################################################################
##
#F  ValuePol( <f>, <x> )  . . . . . . . . . . . evaluate a polynom at a point
##
ValuePol := function ( f, x )
    local  value, i, id;
    id := x ^ 0;
    value := 0 * id;
    i := Length(f);
    while 0 < i  do
        value := value * x + id * f[i];
        i := i-1;
    od;
    return value;
end;


#############################################################################
##
#F  MergedRecord(<rec1>,<rec2>...)  . . . . . . . merge the fields of records
##
MergedRecord := function ( arg )
    local   res,        # merged record, result
            record,     # one of the arguments
            name;       # name of one component of <record>
    res := rec();
    for record  in  arg do
        for name  in RecFields( record )  do
            if IsBound( res.(name) )  then
                Unbind( res.(name) );
            else
                res.(name) := record.(name);
            fi;
        od;
    od;
    return res;
end;


#############################################################################
##
#F  UnionBlist( <blist1>, <blist2> )  . . . . . . . . . . . . union of blists
##
UnionBlist := function ( arg )
    local  U, i;
    if Length( arg ) = 1  then
        arg := arg[1];
    fi;
    U := Copy( arg[1] );
    for i  in [2..Length(arg)]  do
        UniteBlist( U, arg[i] );
    od;
    return U;
end;


#############################################################################
##
#F  IntersectionBlist( <blist1>, <blist2> ) . . . . .  intersection of blists
##
IntersectionBlist := function ( arg )
    local  I, i;
    if Length( arg ) = 1  then
        arg := arg[1];
    fi;
    I := Copy( arg[1] );
    for i  in [2..Length(arg)]  do
        IntersectBlist( I, arg[i] );
    od;
    return I;
end;


#############################################################################
##
#F  DifferenceBlist( <blist1>, <blist2> ) . . . . . . .  difference of blists
##
DifferenceBlist := function ( blist1, blist2 )
    local  D;
    D := Copy( blist1 );
    SubtractBlist( D, blist2 );
    return D;
end;


#############################################################################
##
#F  SetPrintLevel( <L>, <lev> ) . . . . . . . . . . .  set print level of <L>
##
SetPrintLevel := function( L, lev )
   L.operations.SetPrintLevel( L, lev );
end;


#############################################################################
##
#F  Save( <file>, <obj>, <name> ) . . . . . . . . . save some strange objects
##
Save := function( F, G, N )
    if not IsRec(G) or not IsBound(G.operations.Save)  then
        Error( "sorry, I do not know how to save <G>" );
    fi;
    G.operations.Save( F, G, N );
end;


#############################################################################
##
#V  PKGNAME . . . . . . . . . . . . . . . . . . . location of share libraries
##
SetPkgname := function( path )
    local   i,  l,  p;

    # copy old path
    path := Copy(path);

    # append final ';'
    if path[Length(path)] <> ';'  then
        Add( path, ';' );
    fi;

    # replace "lib/;" by "pkg/;"
    for i  in [ 1 .. Length(path)-4 ]  do
        if path{[i..i+4]} = "lib/;"  then
            path{[i..i+4]} := "pkg/;";
        elif path{[i..i+4]} = "lib\\;"  then  # DOS
            path{[i..i+4]} := "pkg\\;";
        elif path{[i..i+4]} = "lib:;"  then   # MacOS
            path{[i..i+4]} := "pkg:;";
        fi;
    od;

    # now split paths
    p := [];
    l := 1;
    for i  in [ 1 .. Length(path) ]  do
        if path[i] = ';'  then
            Add( p, path{[l..i-1]} );
            IsString( p[Length(p)] );
            l := i+1;
        fi;
    od;

    # and return
    return p;

end;

# change for spiral - start
PKGNAME := SetPkgname( LIBNAME );

Add(PKGNAME, ConcatenationString(Conf("spiral_dir"), Conf("path_sep")));
Add(PKGNAME, ConcatenationString(Conf("spiral_dir"), Conf("path_sep"), "namespaces", Conf("path_sep")));
# change for spiral - end


#############################################################################
##
#F  ReadPkg( <lib>, <name> )  . . . . . . . . . .   read a share library file
##
LOADED_PACKAGES := rec();

ReadPkg := function( arg )
    local   ind,  fln,  i, fln1;

    # store old indent value, add two spaces
    ind := ReadIndent;
    ReadIndent := ConcatenationString( ReadIndent, "  " );

    # construct complete path
    fln := Copy( LOADED_PACKAGES.(arg[1]) );
    for i  in [ 2 .. Length(arg)-1 ]  do
        Append( fln, arg[i] );
        Add( fln, '/' );
    od;
    Append( fln, arg[Length(arg)] );
    IsString(fln);

    # read in file -- try bare name first. If that fails, try with .g
    if not READ(fln)  then
        Append( fln, ".g" );
        InfoRead1( "#I", ReadIndent, "ReadPkg( \"", fln, "\" )\n" );

        # read in file
        if not READ(fln)  then
            Error("share library file \"",fln,"\" must exist and be readable");
         fi;
    else
        InfoRead1( "#I", ReadIndent, "ReadPkg( \"", fln, "\" )\n" );
    fi;

    # restore old indentation
    ReadIndent := ind;

end;


#############################################################################
##
#F  ExecPkg( <lib>, <cmd>, <ags>, <dir> ) . . . . .  execute a package binary
##
##  Change to the directory <dir> and execute <cmd> with arguments <ags>.
##
ExecPkg := function( lib, cmd, ags, dir )
    local   del,  new,  i,  sub;

    # prefix <cmd> with path
    new := Copy( LOADED_PACKAGES.(lib) );
    Append( new, cmd );

    # construct the command line
    cmd := ConcatenationString( "cd ", dir, "; ", new, " ", ags );
    InfoRead1( "#I  ExecPkg: executing ", cmd, "\n" );
    Exec(cmd);

end;


#############################################################################
##
#F  LoadPackage( <name> ) . . . . . . . . . . .  load a share library package
##
LoadPackage := function( name )
    local   path,  init,  ind;

    # store old indent value, add two spaces
    ind := ReadIndent;
    ReadIndent := ConcatenationString( ReadIndent, "  " );

    # find the share library <name>
    for path  in PKGNAME  do

        # check next <path>
        init := Copy(path);
        Append( init, name );
        Append( init, "/" );
        IsString(init);
        LOADED_PACKAGES.(name) := Copy(init);
        Append( init, "init.g" );
        IsString(init);

        # give read info
        InfoRead1( "#I  LoadPackage tries \"", init, "\"\n" );

        # try to read the init file
        if READ(init)  then
            ReadIndent := ind;
            return init;
        fi;
    od;

    # signal an error
    Unbind( LOADED_PACKAGES.(name) );
    ReadIndent := ind;
    Error( "share library \"", name, "\" is not installed" );

end;


#############################################################################
##
#F  RequirePackage( <name> )  . . . . . . . . . .  make sure <name> is loaded
##
RequirePackage := function( name )

    # check if <name> is already loaded
    if not IsBound( LOADED_PACKAGES.(name) )  then
    LoadPackage( name );
    fi;

end;

#############################################################################
##
#F  IsOperationsRecord( <obj> ) . . . . . . . . . . .  category test function
##
IsOperationsRecord := function( obj )
    return     IsRec( obj )
           and IsBound( obj.name )
           and IsBound( obj.operations )
           and IsRec( obj.operations )
           and IsBound( obj.operations.name )
           and obj.operations.name = "OpsOps";
end;

##############################################################################
##
#V  OpsOps
##
OpsOps := rec( name:= "OpsOps",
               Print:= function( obj ) Print( obj.name ); end );

OpsOps.( "=" ) := function( oprec1, oprec2 )
    return IsOperationsRecord( oprec1 )  and
           IsOperationsRecord( oprec2 )  and
       oprec1.name = oprec2.name;
end;

OpsOps.operations := OpsOps;

##############################################################################
##
#F  OperationsRecord( <name> )
#F  OperationsRecord( <name>, <parent> )
##
OperationsRecord := function( arg )

    local oprec;

    if Length( arg ) = 1 then
      oprec:= rec();
    else
      oprec:= Copy( arg[2] );
    fi;
    oprec.name:= arg[1];
    oprec.operations:= OpsOps;
    return oprec;
    end;

##############################################################################
##
#F  EXEC( <str1>, <str2>, ... )
##
##  This should become the standard of 'Exec'
##
EXEC := function( arg )
    Exec( Concatenation( List( arg, String ) ) );
    end;

False := function( arg )
    return false;
    end;

True := function( arg )
    return true;
    end;

#############################################################################
##
#F  PrintFactorsInt( <n> )  . . . . . . . . print factorization of an integer
##
##  'PrintFactorsInt'  prints the prime decomposition of the given integer n.
##
PrintFactorsInt := function ( n )
    local decomp, i;

    if -4 < n and n < 4 then
        Print( n );
    else
        decomp := Collected( Factors( AbsInt( n ) ) );
        if n > 0 then
            Print( decomp[1][1] );
        else
            Print( -decomp[1][1] );
        fi;
        if decomp[1][2] > 1 then
            Print( "^", decomp[1][2] );
        fi;
        for i in [ 2 .. Length( decomp ) ] do
            Print( "*", decomp[i][1] );
            if decomp[i][2] > 1 then
                Print( "^", decomp[i][2] );
            fi;
        od;
    fi;
end;
