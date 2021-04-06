
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(VBase);

Class(VBase, SumsBase, BaseContainer, rec(
    __call__ := (self, spl, vlen) >> Checked(IsSPL(spl), IsInt(vlen),
        SPL(WithBases(self, rec(_children := [spl], 
                                vlen := vlen,
                                dimensions := spl.dimensions)))),
    rChildren := self >> [self._children[1], self.vlen],
    rSetChild := meth(self, n, newChild)
        if n=1 then self._children[1] := newChild; self.dimensions := newChild.dimensions;
        elif n=2 then self.vlen := newChild;
        else Error("<n> must be in [1..2]");
        fi;
    end,

    print := (self, i, is) >> Print(self.name, "(", self._children[1], ", ", self.vlen, ")"),

    transpose := self >> VBase(self.child(1).transpose(), self.vlen),

    sums := self >> self
)); 
