
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#   count number of vector instructions in a ruletree
#   used to build vector base cases with minimal instruction count

VCost := function(R, O)
   return SPLRuleTree(R).vcost();
end;

Tensor.vcost := self >> let(C := self.children(), When(IsIdentitySPL(C[1]), C[1].params*C[2].vcost(), 1E15));
VTensor.vcost := self >> 0;
Compose.vcost := self >> Sum(List(self.children(), i->i.vcost()));
BlockVPerm.vcost := self >> self.n * self.child(1).vcost();
