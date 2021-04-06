
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F _IPCodeSums(<sums-spl>, <x-input-output-var>)
#F
#F Generates unoptimize inplace loop code for <sums-spl>
#F if we know how (i.e. 'ipcode' methods exist for <sums-spl>)
#F
_IPCodeSums := function(sums, x)
    local code;
    code := sums.ipcode(x);
    code.dimensions := sums.dimensions;
    code.root := sums;
    return code;
end;

Inplace.ipcode := (self, x) >> _IPCodeSums(self.child(1), x);

Inplace.code := (self, y, x) >> self.child(1).code(y,x);
#let(i := Ind(),
#    chain(self.child(1).ipcode(x),
#	  loop(i, Rows(self), assign(nth(y,i), nth(x,i)))));
#
LStep.ipcode := (self, x) >> _CodeSumsAcc(self.child(1), x, x);

Compose.ipcode := (self, x) >> chain(
    List(Reversed(self.children()), c->_IPCodeSums(c, x)));

SUM.ipcode := (self, x) >> chain(
    List(self.children(), c->_IPCodeSums(c, x)));


#2x2 inplace blockmat (one entry must be 0)
#A B 
#0 C
#Inplace(A%G0 + S0''*B*G1 + C%G1);
BB.ipcode := (self, x) >> MarkForUnrolling(_CodeSums(self.child(1), x, x));
Blk.ipcode := (self, x) >> MarkForUnrolling(_CodeSums(self, x, x));
ISum.ipcode := (self, x) >> _CodeSums(self,x,x);

