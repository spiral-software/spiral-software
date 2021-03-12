
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(VContainer);

###############################################################################
#F VContainer(<spl>, <isa>)
#F
#F   self.child(1) -- spl
#F   self.isa      -- isa
#F
Class(VContainer, VRC, rec(
    #-----------------------------------------------------------------------
    new := (self, spl, isa) >> SPL(WithBases(self, rec(
        _children := [spl], isa := isa))).setDims(), 
    #-----------------------------------------------------------------------
    dims := self >> self.child(1).dims(),
    #-----------------------------------------------------------------------
    toAMat := (self) >> AMatMat(MatSPL(self.child(1))),
    #-----------------------------------------------------------------------
    isReal := self >> self.child(1).isReal(),
    #-----------------------------------------------------------------------
    conjTranspose := self >> ObjId(self)(self.child(1).conjTranspose(), self.isa),
    #-----------------------------------------------------------------------
    # NOTE: make sure rChildren properly exposes all parameters
    from_rChildren := (self, rch) >> VContainer(rch[1], self.isa),
    #-----------------------------------------------------------------------
    print := (self, i, is) >> Print(self.name, "(\n", Blanks(i+is), 
	self.child(1).print(i+is,is), ",\n", Blanks(i+is), self.isa, ")", 
	self.printA()),
    #-----------------------------------------------------------------------
    unroll := self >> self,
    #-----------------------------------------------------------------------
    transpose := self >> VContainer(self.child(1).transpose(), self.isa),
    #-----------------------------------------------------------------------
    vcost := self >>  self.child(1).vcost()
));

Class(VirtualPad, BaseContainer, SumsBase, rec(
    #-----------------------------------------------------------------------
    new := (self, rows, cols, ch) >> SPL(WithBases(self, rec(
        dimensions := [rows, cols], _children := [ch]))).setDims(),
    #-----------------------------------------------------------------------
    dims := self >> self.dimensions,
    #-----------------------------------------------------------------------
    toAMat := self >> let(m := MatSPL(self.child(1)), ch := self.child(1),
	AMatMat(
	    List(m, x -> x :: Replicate(Cols(self)-Cols(ch), 0)) ::
	    Replicate(Rows(self)-Rows(ch), Replicate(Cols(self), 0)))),
    #-----------------------------------------------------------------------
    rChildren := self >> [self.dimensions[1], self.dimensions[2], self._children[1]],
    #-----------------------------------------------------------------------
    rSetChild := meth(self, n, what)
        if   n=1 then self.dimensions[1] := Checked(IsIntSym(what), what);
        elif n=2 then self.dimensions[2] := Checked(IsIntSym(what), what);
	elif n=3 then self._children[1]  := Checked(IsSPL(what), what);
	else Error("<n> must be in [1..3]");
	fi;
    end,
));
