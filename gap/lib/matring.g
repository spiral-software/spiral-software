# -*- Mode: shell-script -*- 

#############################################################################
##
#A  matring.g                   GAP library                  Martin Schoenert
##
##
#Y  Copyright (C) 2018-2021, Carnegie Mellon University
#Y  All rights reserved.  See LICENSE for details.
#Y  
#Y  This work is based on GAP version 3, with some files from version 4.  GAP is
#Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
##
##  This file contains  those  functions that mainly deal with matrix  rings.
##
##


#############################################################################
##
#F  IsMatrixRing(<obj>) . . . . . . . . .  test if an object is a matrix ring
##
IsMatrixRing := function ( obj )
    return IsRec( obj )
       and IsBound( obj.isMatrixRing )  and obj.isMatrixRing;
end;


#############################################################################
##
#F  MatricesOps.Ring(<gens>)  . . . . . . . . . . . . .  create a matrix ring
##
MatricesOps.Ring := function ( gens )
    local   R;

    # make the ring record
    R := rec(
        isDomain                := true,
        isRing                  := true,
        isMatrixRing            := true,

        generators              := gens,
        one                     := gens[1]^0,
        zero                    := gens[1] - gens[1],

        dimension               := Length( gens[1] ),
        field                   := Field( Flat( gens ) ),

        operations              := MatrixRingOps
    );

    # return the ring record
    return R;
end;


#############################################################################
##
#F  MatricesOps.DefaultRing(<gens>) . . . . .  create the default matrix ring
##
MatricesOps.DefaultRing := MatricesOps.Ring;


#############################################################################
##
#V  MatrixRingOps . . . . . . . . . operation record for matrix ring category
##
##  'MatrixRingOps' is the  operation record for  matrix  rings.  It contains
##  the domain functions,  e.g., 'Size'  and   'Intersection', and the   ring
##  functions, e.g., 'IsUnit' and 'Factors'.
##
##  'MatrixRingOps' is initially a copy  of 'RingOps', and thus inherits  the
##  default ring  functions.    Currently  we  overlay   very few   of  those
##  functions.
##
MatrixRingOps := Copy( RingOps );


#############################################################################
##
#F  MatrixRingOps.IsFinite(<R>) . . . . . . . test if a matrix ring is finite
##
MatrixRingOps.IsFinite := function ( R )
    if IsFinite( R.field )  then
        return true;
    else
        return RingOps.IsFinite( R );
    fi;
end;


#############################################################################
##
#F  MatrixRingOps.IsUnit(<R>,<m>) . . . . . . . .  test if a matrix is a unit
##
MatrixRingOps.IsUnit := function ( R, m )
    return DeterminantMat( m ) <> R.field.zero
       and m^-1 in R;
end;


#############################################################################
##
#F  MatrixRingOps.Quotient := function ( R, m, n )
##
MatrixRingOps.Quotient := function ( R, m, n )
    if IsFinite( R )  then
        if RankMat( n ) = Length( n )  then
            return m / n;
        else
            Error("<n> must be invertable");
        fi;
    else
        Error("sorry, cannot compute the quotient of <m> and <n>");
    fi;
end;


#############################################################################
##
#E  Emacs . . . . . . . . . . . . . . . . . . . . . . . local emacs variables
##
##  Local Variables:
##  mode:               outline
##  outline-regexp:     "#F\\|#V\\|#E"
##  fill-column:        73
##  fill-prefix:        "##  "
##  eval:               (hide-body)
##  End:
##



