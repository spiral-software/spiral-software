
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# ==========================================================================
# Mon(<perm-func>, <value-diag>, <structural-diag>) = perm * diag 
# ==========================================================================
Class(Mon, BaseMat, SumsBase, rec(
    new := (self, perm, vdiag, sdiag) >> Checked(
       domain(perm) = domain(vdiag) and domain(perm) = domain(sdiag),
       SPL(WithBases(self, 
		rec( perm := perm,
		     vdiag := vdiag,
		     sdiag := sdiag ))).setDims()),

    #-----------------------------------------------------------------------
    dims := self >> [self.perm.domain(), self.perm.range()],
    setDims := meth(self) self.dimensions := self.dims(); return self; end,
    #-----------------------------------------------------------------------
    sums := self >> self,
    rChildren := self >> [self.perm, self.vdiag, self.sdiag],
    rSetChild := rSetChildFields("perm", "vdiag", "sdiag"),

    children := ~.rChildren,
    setChild := ~.rSetChild,
    child := (self, n) >> Cond(n=1, self.perm, n=2, self.vdiag, n=3, self.sdiag,
	Error("<n> must be in [1..3]")),
    #-----------------------------------------------------------------------
    isPermutation := False, 
    #-----------------------------------------------------------------------
    isReal := self >> self.vdiag.isReal(),
    #-----------------------------------------------------------------------
    toAMat := self >> AMatSPL( Gath(self.perm) * Diag(self.vdiag) * Diag(self.sdiag) ),
    #-----------------------------------------------------------------------
    transpose := self >>   # we use inherit to copy all fields of self
        Inherit(self, rec(
		perm := self.perm.transpose(),
		vdiag := fCompose(self.vdiag, self.perm),
		sdiag := fCompose(self.sdiag, self.perm),
		dimensions := Reversed(self.dimensions))),
    #-----------------------------------------------------------------------
    print := (self, i, is) >>
       Print(self.name, "(", self.perm, ", ", self.vdiag, ", ", self.sdiag, ")"),
    #-----------------------------------------------------------------------
    arithmeticCost := (self, costMul, costAddMul) >> Sum(self.vdiag.tolist(), costMul)
));

Declare(SMon, GMon, SGMon);

# ==========================================================================
# GMon(<gath-func>, <value-diag>, <structural-diag>) = diag * gath
# ==========================================================================
Class(GMon, Mon, rec(
    #-----------------------------------------------------------------------
    dims := self >> [domain(self.perm), range(self.perm)],
    #-----------------------------------------------------------------------
    toAMat := self >> AMatSPL( self.vdiag * self.sdiag * Gath(self.perm) ),
    #-----------------------------------------------------------------------
    transpose := self >> SMon(self.perm, self.vdiag, self.sdiag)
    #-----------------------------------------------------------------------
));

# ==========================================================================
# SMon(<scat-func>, <value-diag>, <structural-diag>) = scat * diag 
# ==========================================================================
Class(SMon, Mon, rec(
    #-----------------------------------------------------------------------
    dims := self >> [range(self.perm), domain(self.perm)],
    #-----------------------------------------------------------------------
    toAMat := self >> AMatSPL( Scat(self.perm) * self.vdiag * self.sdiag ),
    #-----------------------------------------------------------------------
    transpose := self >> GMon(self.perm, self.vdiag, self.sdiag)
    #-----------------------------------------------------------------------
));

# ==========================================================================
# SGMon(<scat-func>, <gath-func>, <value-diag>, <structural-diag>) = scat * diag * gath
# ==========================================================================
Class(SGMon, Mon, rec(
    new := (self, scat, gath, vdiag, sdiag) >> Checked(
       domain(scat) = domain(vdiag) and domain(vdiag) = domain(sdiag) and
       domain(sdiag) = domain(gath),
       SPL(WithBases(self, 
		rec( scat := scat, 
		     gath := gath, 
		     vdiag := vdiag,
		     sdiag := sdiag ))).setDims()),

    rChildren := self >> [self.scat, self.gath, self.vdiag, self.sdiag],
    rSetChild := rSetChildFields("scat", "gath", "vdiag", "sdiag"),
    child := (self, n) >> Cond(n=1, self.scat,  n=2, self.gath, 
	                       n=3, self.vdiag, n=4, self.sdiag,
			       Error("<n> must be in [1..3]")),
    #-----------------------------------------------------------------------
    dims := self >> [range(self.scat), range(self.gath)],
    #-----------------------------------------------------------------------
    toAMat := self >> AMatSPL(Scat(self.scat) * self.vdiag * self.sdiag * 
	                      Gath(self.gath)),
    #-----------------------------------------------------------------------
    transpose := self >> SGMon(self.gath, self.scat, self.vdiag, self.sdiag),
    #-----------------------------------------------------------------------
    print := (self, i, is) >>
       Print(self.name, "(", self.scat, ", ", self.gath, ", ", self.vdiag, ", ", self.sdiag, ")")
 ));
