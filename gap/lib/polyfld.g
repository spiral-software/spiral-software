# -*- Mode: shell-script -*- 

#############################################################################
##
#A  polyfld.g                   GAP library                      Frank Celler
##
##
#Y  Copyright (C) 2018-2021, Carnegie Mellon University
#Y  All rights reserved.  See LICENSE for details.
#Y  
#Y  This work is based on GAP version 3, with some files from version 4.  GAP is
#Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
##
##  This file contains functions for polynomials over fields.
##
##


#############################################################################
##
#F  InfoPoly1(...)  . . . . . . . . . . . infomation function for polynomials
#F  InfoPoly2(...)  . . . . . . . . . . . . .  debug function for polynomials
##
if not IsBound(InfoPoly1)    then InfoPoly1   := Ignore;  fi;
if not IsBound(InfoPoly2)    then InfoPoly2   := Ignore;  fi;


#############################################################################
##

#V  FieldPolynomialRingOps  . . . . . . . . . . . .  full polynomial ring ops
##
FieldPolynomialRingOps := Copy( PolynomialRingOps );
FieldPolynomialRingOps.name := "FieldPolynomialRingOps";


#############################################################################
##
#F  FieldPolynomialRingOps.StandardAssociate( <R>, <f> )  . . . .  normed pol
##
FieldPolynomialRingOps.StandardAssociate := function( R, f )
    if Length(f.coefficients)=0 then
      return Zero(R);
    fi;
    return f * f.coefficients[ Length( f.coefficients ) ]^-1;
end;


#############################################################################
##
#F  FieldPolynomialRingOps.EuclideanRemainder( <R>, <f>, <g> )  .  remnainder
##
FieldPolynomialRingOps.EuclideanRemainder := function( R, f, g )
    local   m,  n,  i,  k,  c,  q,  R,  val;

    # <f> and <g> must have the same field
    if f.baseRing <> g.baseRing  then
    	Error( "<f> and <g> must have the same base ring" );
    fi;

    # and they must be ordinary polynomials
    if f.valuation < 0  then
    	Error( "<f> must not be laurent polynomial" );
    fi;
    if g.valuation < 0  then
    	Error( "<g> must not be laurent polynomial" );
    fi;

    # if <g> is zero signal an error
    if 0 = Length(g.coefficients)  then
    	Error( "<g> must not be zero" );
    fi;

    # if <f> is zero return it
    if 0 = Length(f.coefficients)  then
	return f;
    fi;

    # remove the valuation of <f> and <g>
    f := ShiftedCoeffs( f.coefficients,  f.valuation );
    g := ShiftedCoeffs( g.coefficients,  g.valuation );

    # Try to divide <f> by <g>
    q := [];
    n := Length(g);
    m := Length(f) - n;
    for i  in [ 0 .. m ]  do
        c := f[m-i+n] / g[n];
        for k  in [ 1 .. n ]  do
            f[m-i+k] := f[m-i+k] - c * g[k];
        od;
        q[m-i+1] := c;
    od;

    # return the polynomial
    return R.baseRing.operations.FastPolynomial( R.baseRing, f, 0 );

end;


#############################################################################
##
#F  FieldPolynomialRingOps.EuclideanQuotient( <R>, <f>, <g> ) . . . . <l>/<r>
##
FieldPolynomialRingOps.EuclideanQuotient := function ( R, f, g )
    local   m,  n,  i,  k,  c,  q,  R,  val;

    # <f> and <g> must have the same field
    if f.baseRing <> g.baseRing  then
    	Error( "<f> and <g> must have the same base ring" );
    fi;

    # and they must be ordinary polynomials
    if f.valuation < 0  then
    	Error( "<f> must not be laurent polynomial" );
    fi;
    if g.valuation < 0  then
    	Error( "<g> must not be laurent polynomial" );
    fi;

    # if <g> is zero signal an error
    if 0 = Length(g.coefficients)  then
    	Error( "<g> must not be zero" );
    fi;

    # if <f> is zero return it
    if 0 = Length(f.coefficients)  then
	return f;
    fi;

    # remove the valuation of <f> and <g>
    f := ShiftedCoeffs( f.coefficients,  f.valuation );
    g := ShiftedCoeffs( g.coefficients,  g.valuation );

    # Try to divide <f> by <g>
    q := [];
    n := Length(g);
    m := Length(f) - n;
    for i  in [ 0 .. m ]  do
        c := f[m-i+n] / g[n];
        for k  in [ 1 .. n ]  do
            f[m-i+k] := f[m-i+k] - c * g[k];
        od;
        q[m-i+1] := c;
    od;

    # return the polynomial
    return R.baseRing.operations.FastPolynomial( R.baseRing, q, 0 );

end;


#############################################################################
##
#F  FieldPolynomialRingOps.QuotientRemainder( <R>, <f>, <g> ) . . rem and quo
##
FieldPolynomialRingOps.QuotientRemainder := function ( R, f, g )
    local   m,  n,  i,  k,  c,  q,  R,  val;

    # <f> and <g> must have the same field
    if f.baseRing <> g.baseRing  then
    	Error( "<f> and <g> must have the same base ring" );
    fi;

    # and they must be ordinary polynomials
    if f.valuation < 0  then
    	Error( "<f> must not be laurent polynomial" );
    fi;
    if g.valuation < 0  then
    	Error( "<g> must not be laurent polynomial" );
    fi;

    # if <g> is zero signal an error
    if 0 = Length(g.coefficients)  then
    	Error( "<g> must not be zero" );
    fi;

    # if <f> is zero return it
    if 0 = Length(f.coefficients)  then
	return f;
    fi;

    # remove the valuation of <f> and <g>
    f := ShiftedCoeffs( f.coefficients,  f.valuation );
    g := ShiftedCoeffs( g.coefficients,  g.valuation );

    # Try to divide <f> by <g>
    q := [];
    n := Length(g);
    m := Length(f) - n;
    for i  in [ 0 .. m ]  do
        c := f[m-i+n] / g[n];
        for k  in [ 1 .. n ]  do
            f[m-i+k] := f[m-i+k] - c * g[k];
        od;
        q[m-i+1] := c;
    od;

    # return the polynomials
    return [ R.baseRing.operations.FastPolynomial(R.baseRing, q, 0),
             R.baseRing.operations.FastPolynomial(R.baseRing, f, 0) ];

end;


#############################################################################
##

#V  FieldLaurentPolynomialRingOps . . . . . . . . .  full polynomial ring ops
##
FieldLaurentPolynomialRingOps := Copy( LaurentPolynomialRingOps );
FieldLaurentPolynomialRingOps.name := "FieldLaurentPolynomialRingOps";


#############################################################################
##
#F  FieldLaurentPolynomialRingOps.StandardAssociate( <R>, <f> ) .  normed pol
##
FieldLaurentPolynomialRingOps.StandardAssociate := function( R, f )
    local   l;

    l := f.coefficients[Length(f.coefficients)];
    return R.baseRing.operations.FastPolynomial(
                   R.baseRing, f.coefficients*l^-1, 0 );
end;


#############################################################################
##
#F  FieldLaurentPolynomialRingOps.EuclideanRemainder( <R>, <f>, <g> ) . . rem
##
FieldLaurentPolynomialRingOps.EuclideanRemainder := function( R, f, g )
    local   F,  G,  r;

    # construct <F>, <G> such that <F>*x^i = <f> and <G>*x^j = <g>
    F := Polynomial( f.baseRing, f.coefficients, 0 );
    G := Polynomial( g.baseRing, g.coefficients, 0 );

    # compute <r> such that <F> = h * <G> + <r>
    r := EuclideanRemainder( PolynomialRing(R.baseRing), F, G );

    # now (<F>*x^i) = h*x^(i-j) * (<G>*x^j) + <r>*x^i
    return R.baseRing.operations.FastPolynomial(
                   R.baseRing, r.coefficients, r.valuation+f.valuation );

end;


#############################################################################
##
#F  FieldLaurentPolynomialRingOps.EuclideanQuotient( <R>, <l>, <r> )  <l>/<r>
##
FieldLaurentPolynomialRingOps.EuclideanQuotient := function ( R, f, g )
    local   m,  n,  i,  k,  c,  q,  R,  val;

    # get the base ring
    R := R.baseRing;

    # if <f> is zero return it
    if 0 = Length(f.coefficients)  then
	return f;
    fi;

    # get the value of the valuation of <f> / <g>
    val := f.valuation - g.valuation;

    # Try to divide <f> by <g>
    q := [];
    n := Length( g.coefficients );
    m := Length( f.coefficients ) - n;
    f := ShallowCopy( f.coefficients );
    if IsField(R)  then
        for i  in [0 .. m ]  do
            c := f[m-i+n] / g.coefficients[n];
            for k  in [ 1 .. n ]  do
                f[m-i+k] := f[m-i+k] - c * g.coefficients[k];
            od;
            q[m-i+1] := c;
        od;
    else
        for i  in [0 .. m ]  do
            c := Quotient( R, f[m-i+n], g.coefficients[n] );
    	    if c = false  then return false;  fi;
            for k  in [ 1 .. n ]  do
                f[m-i+k] := f[m-i+k] - c * g.coefficients[k];
            od;
            q[m-i+1] := c;
        od;
    fi;

    # return the result
    return R.operations.FastPolynomial( R, q, val );

end;


#############################################################################
##
#F  FieldLaurentPolynomialRingOps.QuotientRemainder( <R>, <l>, <r> )  <l>/<r>
##
FieldLaurentPolynomialRingOps.QuotientRemainder := function ( R, f, g )
    return [ EuclideanQuotient(R,f,g), EuclideanRemainder(R,f,g) ];
end;
