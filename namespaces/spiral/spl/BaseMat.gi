
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F IsSPLMat(<x>) - returns true for parametrized matrices (Gath, Scat, Diag, etc)
#F
IsSPLMat := x -> IsRec(x) and IsBound(x._sym) and x._sym;

# ==========================================================================
# BaseMat
# ==========================================================================
Class(BaseMat, ClassSPL, rec(
    _mat := true,
    _short_print := true,
    #-----------------------------------------------------------------------
    isTerminal := True,
    isPermutation := False,
    children   := self >> [], 
    terminate  := self >> self,
    # -------- Transformation rules support ---------------------------------
    rChildren := self >> [self.element],
    rSetChild := rSetChildFields("element"), 
));

