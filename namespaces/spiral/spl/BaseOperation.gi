
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# ==========================================================================
# BaseOperation
# ==========================================================================
Class(BaseOperation, ClassSPL, rec(
    #-----------------------------------------------------------------------
    child       := (self, n) >> self._children[n],
    children    := self >> self._children,
    numChildren := self >> Length(self.children()),
    isReal      := self >> ForAll(self.children(), IsRealSPL),
    setChild := meth(self, n, what) self._children[n] := what; end,
    # -------- Transformation rules support ---------------------------------
    rChildren := ~.children, 
    rSetChild := ~.setChild, 
));

