
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


DCTDiag  := n -> Diag(diagDirsum(fConst(1,1/Sqrt(n)), fConst(n-1, Sqrt(2/n))));
DCTDiag2 := n -> Diag(diagDirsum(fConst(1,1/n), fConst(n-1, 2/n)));

#ForwardDCT := DCT2;

Class(InverseDCT, DTTBase, rec(
    terminate := self >> DCT3(self.params[1]).terminate() * DCTDiag2(self.params[1]),
#    transpose := self >> DCTDiag2(self.params[1]) * DCT2(self.params[1]),
));

RulesFor(InverseDCT, rec(
     InverseDCT_toDCT3 := rec(
	 isApplicable     := P -> true,
	 allChildren      := P -> [[ DCT3(P[1]) ]],
	 rule := (P, C) -> C[1] * DCTDiag2(P[1])
     )
));

