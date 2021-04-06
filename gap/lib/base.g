# -*- Mode: shell-script -*- 

##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

GlobalPackage(gap.numbase);

# Represent number <num> as a list of coefficients for powers of base
#
BaseN := function(num, base)
    local pow, res, pow, factor;
    res := [ ];
    if num = 0 then return [ 0 ]; fi;
    pow := LogInt(num, base);
    while pow >= 0 do
	factor := Int(num / base^pow);
	Add(res, factor);
	num := num - factor*base^pow;
	pow := pow - 1;
    od;
    
    return res;
end;

# Tests baseN for correctness on given input
#
TestBaseN := function(num, base)
    local z;
    z := BaseN(num, base);
    return num = Sum([0..Length(z)-1], x->base^x * z[Length(z)-x]);
end;

HexInt := function(num)
    local hex;
    hex := "0123456789abcdef";
    return String( List(BaseN(num, 16), x->hex[x+1] ) );
end;

AlphInt := function(num)
    local alp;
    alp := "0123456789abcdefghijklmnopqrstuv"; # length = 32
    return String( List(BaseN(num, Length(alp)), x->alp[x+1] ) );
end;

ExtAlphInt := function(num)
    local alp;
    alp := "0123456789abcdefghijklmnopqrstuvwxyz"; # length = 36
    alp := Concat(alp, 
	   "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    return String( List(BaseN(num, Length(alp)), x->alp[x+1] ) );
end;

# first character is a digit
VarNameInt := function(num)
    local alp, lead, res;
    lead := num mod 10;
    num := Int(num / 10);
    alp := "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    res := [alp[1+lead]];
    if num > 0 then
	Append(res, String( List(BaseN(num, Length(alp)), x->alp[x+1] ) ));
    fi;
    return res;
end;
