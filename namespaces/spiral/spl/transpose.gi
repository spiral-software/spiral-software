
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(InertTranspose, BaseContainer, SumsBase, rec(
    dims := self >> Reversed(self.child(1).dims()),
    toAMat := self >> AMatMat(TransposedMat(MatSPL(self.child(1))))
));

Class(InertConjTranspose, BaseContainer, SumsBase, rec(
    dims := self >> Reversed(self.child(1).dims()),
    toAMat := self >> AMatMat(Global.Conjugate(TransposedMat(MatSPL(self.child(1)))))
));

InertConjTranspose.conjTranspose := self >> self.child(1);
