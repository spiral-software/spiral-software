
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(SymSPL, BaseContainer, SumsBase, rec(
    isBlock:=true,
    transpose := self >> self,
    rng:=self>>self._children[1].rng(),
    dmn:=self>>self._children[1].dmn(),
    vcost := self >> self._children[1].vcost()
));
