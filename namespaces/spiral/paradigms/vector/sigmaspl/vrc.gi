
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(VRC, VRCL, VRCR, VRCLR, VBlkInt);

##############################################################################
Class(VRC, RC, rec(
    toAMat := (self) >> AMatMat(RCMatCyc(MatSPL(self.child(1)))),
    new := meth(self, spl, v)
        local res;
        res := SPL(WithBases(self, rec(_children:=[spl], v:=v,
                                dimensions := spl.dimensions)));
        res.dimensions := res.dims();
        return res;
    end,
    print := (self, i, is) >> Print(self.__name__, "(\n", Blanks(i+is), self.child(1).print(i+is,is), ", ",
	#"\n", Blanks(i+is), 
	self.v, 
	#"\n", Blanks(i),
	")", self.printA()),

    unroll := self >> self,
    transpose := self >> VRC(self.child(1).conjTranspose(), self.v),
    vcost := self >> self.child(1).vcost(),

    from_rChildren := (self, rch) >> ObjId(self)(rch[1], self.v)
));

Class(VRCL, VRC, rec(
    toAMat := self >> AMatMat(MatSPL(Tensor(I(self.child(1).dims()[1]/self.v), L(2*self.v, 2)))) *
                      VRC(self.child(1), self.v).toAMat(),
    unroll := self >> self,
    transpose := self >> VRCR(self.child(1).conjTranspose(), self.v)
));

Class(VRCR, VRC, rec(
    toAMat := self >> VRC(self.child(1), self.v).toAMat() *
                      AMatMat(MatSPL(Tensor(I(self.child(1).dims()[2]/self.v), L(2*self.v, self.v)))),
    unroll := self >> self,
    transpose := self >> VRCL(self.child(1).conjTranspose(), self.v)
));

Class(VRCLR, VRC, rec(
    toAMat := self >> let(
    rows := Rows(self.child(1)), cols := Cols(self.child(1)), v := self.v,
    AMatMat(MatSPL(
        Tensor(I(rows/v), L(2*v, 2)) *
        RC(self.child(1)) *
        Tensor(I(cols/v), L(2*v, v))))
    ),
    unroll := self >> self,
    transpose := self >> VRCLR(self.child(1).conjTranspose(), self.v)
));

Class(vRC, RC, rec(
    print := (self, i, is) >> Print(self.name, "(", self.child(1).print(i+is,is), ")", self.printA()),
    vcost := self >> self.child(1).vcost()
));


Class(VBlkInt, BaseMat, SumsBase, rec(
    new := (self, spl, v) >> SPL(WithBases(self,
        rec(dimensions := spl.dims(), _children:=[spl], v := v))),
    #-----------------------------------------------------------------------
    dims := self >> self.dimensions,
    child := (self, i) >> self._children[i],
    rChildren := self >> [self._children[1], self.v],
    rSetChild := meth(self, n, what)
        if n=1 then self._children[1] := what;
        elif n=2 then self.v := what;
        else Error("<n> must be in [1..2]");
        fi;
    end,
    #-----------------------------------------------------------------------
    print := (self, i, is) >> Print(self.name, "(", self.child(1).print(i+is,is), ", ", self.v, ")", self.printA()),
    unroll := self >> self,
    transpose := self >> VBlkInt(self.child(1).transpose(), self.v),
    #-----------------------------------------------------------------------
    toAMat := self >> let(
        rows := Rows(self.child(1)), cols := Cols(self.child(1)), v := self.v,
        AMatMat(MatSPL(
            Tensor(I(rows/(2*v)), L(2*v, v)) *
            self.child(1) *
            Tensor(I(cols/(2*v)), L(2*v, 2))))
        )
));
