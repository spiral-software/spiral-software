
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#
# NB: Contents of this file should be moved to a directory local to the 
#     particular Spiral Extension
#

Class(f2DTr, Tr, rec(
    advdomain := self >> [[self.params[2], self.params[1]]],
    advrange := self >> [[self.params[1], self.params[2]]]
));

Class(f2DTrExplicitL, f2DTr);
Class(f2DTrExplicitR, f2DTr);

Class(f4DHalfFlip, PermClass, rec(
    def := (a1,a2,b1,b2) -> rec(size := a1*a2*b1*b2),
    lambda := self >> let(
        a1 := self.params[1], a2 := self.params[2], 
        b1 := self.params[3], b2 := self.params[4], 
        fTensor(Tr(a1, a2), Tr(b1, b2)).lambda()
        ),
    domain := self >> self.params[1] * self.params[2] * self.params[3] * self.params[4],
    range := self >> self.params[1] * self.params[2] * self.params[3] * self.params[4],
    
    dims := self >> [self.range(), self.domain()],

    transpose := self >> Error("not supported"),
    
    advdomain := self >> [[self.params[4], self.params[3], self.params[2], self.params[1]]],
    advrange := self >> [[self.params[3], self.params[4], self.params[1], self.params[2]]]
));

Class(f4DMacroFlip, PermClass, rec(
    def := (a1,a2,b1,b2) -> rec(size := a1*a2*b1*b2),
    lambda := self >> let(
        a1 := self.params[1], a2 := self.params[2], 
        b1 := self.params[3], b2 := self.params[4], 
        Tr(a1*a2, b1*b2).lambda()),
#         i := Ind(a1*a2*b1*b2),
#         k := a1*a2, str := b1*b2,
#         Lambda(i, add(mul(b2, idiv(idiv(i, k), b2)), imod(idiv(i, k), b2)) 
#             + str * add(mul(a2, idiv(imod(i, k), a2)), imod(imod(i, k), a2)))),

    domain := self >> self.params[1] * self.params[2] * self.params[3] * self.params[4],
    range := self >> self.params[1] * self.params[2] * self.params[3] * self.params[4],
    
    dims := self >> [self.range(), self.domain()],

    transpose := self >> Error("not supported"),
    
    advdomain := self >> [[self.params[3], self.params[4], self.params[1], self.params[2]]],
    advrange := self >> [[self.params[1], self.params[2], self.params[3], self.params[4]]]
));

Class(f4DFullFlip, PermClass, rec(
    def := (a1,a2,b1,b2) -> rec(size := a1*a2*b1*b2),
    lambda := self >> let(
        a1 := self.params[1], a2 := self.params[2], 
        b1 := self.params[3], b2 := self.params[4], 
        i := Ind(a1*a2*b1*b2),
        Lambda(i, imod(i,b2) * a1 * a2 * b1 
            + imod(idiv(i,b2),b1)*a1*a2 
            + imod(idiv(i, b2*b1),a2) * a1 
            + imod(idiv(i, b2*b1*a2), a1))),
    domain := self >> self.params[1] * self.params[2] * self.params[3] * self.params[4],
    range := self >> self.params[1] * self.params[2] * self.params[3] * self.params[4],
    
    dims := self >> [self.range(), self.domain()],

    transpose := self >> Error("not supported"),
    
    advdomain := self >> [[self.params[4], self.params[3], self.params[2], self.params[1]]],
    advrange := self >> [[self.params[1], self.params[2], self.params[3], self.params[4]]]
));


#F
#F Multi-dimensional fTensor's for OL
#F

Class(f2DTensor, fTensor, rec(
    advdomain := self >> [List(self._children, x->x.domain())],
    advrange := self >> [List(self._children, x->x.range())]
));

Class(f4DTensor, fTensor, rec(
    advdomain := self >> [List(self._children, x->x.domain())],
    advrange := self >> [List(self._children, x->x.range())]
));
