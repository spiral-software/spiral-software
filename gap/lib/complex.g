# -*- Mode: shell-script -*- 

##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

## Complex imaginary I = sqrt(-1)
##
c_I := c_I;

#F IsComplex( <obj> ) 
#F   tests whether <obj> is a Complex
#F
DocumentVariable(IsComplex);

#F AbsComplex( <complex> )
#F   Returns an absolute value of a complex number (|a+bi| = sqrt(a^2+b^2))
#F
DocumentVariable(AbsComplex);

#F ComplexW ( <int> N, <int> pow )
#F    N-th Root of unity taken to the power 'pow' as a Double-based
#F    Complex number.
#F
DocumentVariable(ComplexW);

#F ComplexCyc ( <cyclotomic> )
#F    Convert cyclotomic to a complex floating point number
#F
ComplexCyc := num -> Complex(num); 

#F ComplexAny ( <num> )
#F    Converts any known number to a complex floating point value
#F    Suported types are Integer, Rational, Double, Cyclotomic
#F
ComplexAny := num -> Complex(num);
