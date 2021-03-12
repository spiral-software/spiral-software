# -*- Mode: shell-script -*- 

##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

#############################################################################
#V  Doubles  . . . . . . . . . . . . . . . . domain of floating-poiny numbers
##
Doubles := CantCopy(rec(
    isDomain   := true,
    isField    := true,
    name       := "Doubles",
    size       := "infinity",
    isFinite   := false,
    field      := 0,
    char       := 0,
    generators := [ 1.0 ],
    zero       := 0.0, 
    one        := 1.0,
    operations := rec(
	\in := (x, dom) -> IsDouble(x) or IsRat(x),
	Field := elms -> Doubles, 
	DefaultField := elms -> Doubles,
	Print := d -> Print(d.name)
    )
));


#############################################################################
#V  Complexes  . . . . . . . . . . . . . . . . . . domain of complex numbers
##
Complexes := CantCopy(rec(
    isDomain   := true,
    isField    := true,
    name       := "Complexes",
    size       := "infinity",
    isFinite   := false,
    char       := 0,
    field      := 0,          
    operations := rec(
	\in := (x, dom) -> IsComplex(x) or IsRat(x) or IsDouble(x) or IsCyc(x),
	Field := elms -> Complexes,
	DefaultField := elms -> Complexes,
	Print := d -> Print(d.name)
    ),
    generators := [ Complex(1,0), Complex(0,1) ],
    zero := Complex(0,0),
    one := Complex(1,0)
));


#############################################################################
## Wrappers for new built-in Double
##

IsFloat  := IsDouble;
Float    := (mant,exp) -> Double(mant) * d_pow(2, exp); 
FloatInt := Double;
DivFloatInt := (i1, i2) -> Double(i1)/i2;
FloatRat := Double;
IntFloat := IntDouble;
RatFloat := RatDouble;
AbsFloat := F -> When(F<0, -F, F);

StringFloat := StringDouble;
FloatString := DoubleString;

RoundDouble := x -> d_floor(x + 0.5);
