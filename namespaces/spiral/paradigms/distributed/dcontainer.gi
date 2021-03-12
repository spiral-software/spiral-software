
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(DContainer);

##############################################################################
Class(DContainer, VRC, rec(
    toAMat := (self) >> AMatMat(MatSPL(self.child(1))),
    new := meth(self, spl, procs)
        local res;
        res := SPL(WithBases(self, rec(_children:=[spl], procs:=procs,
                                dimensions := spl.dimensions)));
        res.dimensions := res.dims();
        return res;
    end,
    print := (self, i, is) >> Print(self.name, "(", self.child(1).print(i+is,is), ", ", self.procs, ")"),
    unroll := self >> self,
    transpose := self >> DContainer(self.child(1).transpose(), self.procs),
    dims := self >> self.child(1).dims()
));
