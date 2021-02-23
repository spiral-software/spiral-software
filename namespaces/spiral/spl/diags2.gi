
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Here I try to be very clean, when defining functions.
# For example, no longer use .size since this often contains info from .params.
# Instead have explicit .domain / .range
# use checkParams instead of def, default def will just set .params
#


#F dLin(<N>, <a>, <b>, <t>) - interval -> diagonal linear scaling function
#F    f: i -> a*i + b,
#F    dom(f) = interval(0..N-1)
#F    range(f) = <t>, must be a type
#F
#F Example:
#F  spiral> dLin(8,1,1/2,TReal).tolist();
#F  [ 1/2, 3/2, 5/2, 7/2, 9/2, 11/2, 13/2, 15/2 ]
#F
Class(dLin, DiagFunc, rec(
    checkParams := (self, params) >> Checked(Length(params)=4,
	IsPosInt0Sym(params[1]), IsType(params[4]), params),
    lambda := self >> let(i:=Ind(self.params[1]), a := self.params[2], b := self.params[3],
	Lambda(i, a*i+b).setRange(self.params[4])),
    range := self >> self.params[4],
    domain := self >> self.params[1]
));

#F dOmega(<N>, <k>) - N-th root of unity diagonal function
#F   f : TInt -> Tcomplex : i -> omega(N, k*i)
#F
Class(dOmega, DiagFunc, rec(
    checkParams := (self, params) >> Checked(Length(params)=2,
	IsPosIntSym(params[1]), IsIntSym(params[2]), params),
    lambda := self >> let(i:=Ind(), Lambda(i, omega(self.params[1], self.params[2]*i)).setRange(TComplex)),
    range := self >> TComplex,
    domain := self >> TInt
));

Class(dOmegaPow, DiagFunc, rec(
    checkParams := (self, params) >> Checked(Length(params)=3,
	IsPosIntSym(params[1]), IsIntSym(params[2]), IsInt(params[3]), params),
    lambda := self >> let(i:=Ind(), Lambda(i, omega(self.params[1], (self.params[2]*i))^self.params[3]).setRange(TComplex)),
    range := self >> TComplex,
    domain := self >> TInt
));
