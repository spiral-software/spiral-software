
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(VWrapBase, rec(
    opts := (self, t, opts) >> opts,
));

Class(VWrapId, VWrapBase, rec(
    __call__ := self >> self,
    wrap := (self,r,t,opts) >> r,
    twrap := (self,t,opts) >> t
));

Class(DPWrapper, SumsBase, BaseContainer, rec(

    _short_print := true,
    
    new := (self, spl, wrap) >> SPL(WithBases(self, rec(
        _children  := [spl],
        dimensions := spl.dimensions,
        wrap       := wrap,
        ))),

    rChildren := self >> [self._children[1], self.wrap],

    rSetChild := meth(self, n, newC)
        if n=1 then self._children[1] := newC;
        elif n=2 then self.wrap := newC;
        else Error("<n> must be in [1..2]");
        fi;
    end,

    sums := self >> self._children[1].sums(),

    print := (self, i, is) >> Cond(self._short_print,
        Print(self.__name__, "(", self._children[1].print(i+is, is), ", ", self.wrap, ")"),
        Print(self.__name__, "(\n", Blanks(i+is), 
	    self._children[1].print(i+is, is), ",\n", Blanks(i+is), self.wrap, "\n", Blanks(i), ")")),

    HashId := self >> let(h := [ When(IsBound(self._children[1].HashId), self._children[1].HashId(), self._children[1]) ],
        When(IsBound(self.tags), Concatenation(h, self.tags), h)),

    vcost := self >> self.child(1).vcost()
));

#F DPSWrapper - wrapper for stackable VWraps;
#F all stackable wrappers applied to formula in _DPSPLRec, innermost first.

Class(DPSWrapper, DPWrapper);

ClassSPL.setWrap := (self, wrap) >> DPWrapper(self, wrap).takeAobj(self);
ClassSPL.addWrap := (self, wrap) >> DPSWrapper(self, wrap).takeAobj(self);
