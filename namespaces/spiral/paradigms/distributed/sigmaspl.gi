
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(GathRecv);
Declare(ScatSend);
Declare(GathDist);
Declare(ScatDist);
Declare(Comm_Cell);
Declare(PTensor);

Declare(DistSum);
Declare(DistSumLoop);

#F GathRecv(func, pkSize, P, i).
#F Gather Receive, for parallel code.
Class(GathRecv, BaseMat, SumsBase, rec(
    sums := self >> self,
    isReal := self >> true,
    dims := self >> self.dimensions,
    needInterleavedLeft := self >> false,
    needInterleavedRight := self >> false,
    cannotChangeDataFormat := self >> true,
    totallyCannotChangeDataFormat := self >> true,
    #-----------------------------------------------------------------------
    # NOTE: we should have 2 children, not 1!
    rChildren := self >> [self.func, self.pkSize, self.P, self.i],
    rSetChild := rSetChildFields("func", "pkSize", "P", "i"),
    #-----------------------------------------------------------------------
    new := (self, func, pkSize, P, i) >> SPL(WithBases(self,
        rec(func       := FF(func),
            pkSize     := pkSize,
            i          := i,
            P          := P,
            dimensions := When(func.__name__ = "FData",
                  [pkSize*func.domain()/P, pkSize*func.domain()],
                  [pkSize*func.domain(), pkSize*func.range()] )
            ))),
    #-----------------------------------------------------------------------
    area := self >> When(func.__name__ = "FData",
                    self.pkSize * self.func.domain()/self.P,
                    self.pkSize * self.func.domain()),

    transpose := self >> ScatSend(self.func, self.pkSize, self.P, self.i),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.name, "(", self.func, ", ", self.pkSize, ", ", self.P, ", ", self.i,")"),
    #-----------------------------------------------------------------------
    toAMat := self >> 
       Cond(self.func.__name__ = "FData",

          # NOTE Problem: The following considers self.i with FData, but not
          # otherwise! (So in the otherwise case, func contains this info).

          # This is if we've already converted to FData. Don't change the func below!
          let(pkSize   := self.pkSize, 
              n        := pkSize*(self.func.domain()/self.P),
              N        := pkSize*self.func.domain(),
              func     := fTensor(
                            fCompose(self.func, fTensor( fBase(self.P, self.i), fId(self.func.domain()/self.P) )),
                            fId(pkSize)
                          ).lambda(),
              AMatMat(List([0..n-1], row -> BasisVec(N, func.at(row).ev())))
          ),
          # This is if we haven't converted to FData yet
          let(pkSize   := self.pkSize, 
              n        := pkSize*self.func.domain(),
              N        := pkSize*self.func.range(),
              func     := fTensor(self.func, fId(pkSize)).lambda(),
              AMatMat(List([0..n-1], row -> BasisVec(N, func.at(row).ev())))
          )
       )
));

#F ScatSend(func, pkSize, P, i).
# For now, ScatSend will specify the target explicitly. GathRecv will just put
# things in place
Class(ScatSend, BaseMat, rec(
    sums := self >> self,
    dims := self >> self.dimensions,
    isReal := self >> true,
    needInterleavedLeft := self >> false,
    needInterleavedRight := self >> false,
    cannotChangeDataFormat := self >> true,
    totallyCannotChangeDataFormat := self >> true,
    area := self >> self.transpose().area(),
    toAMat := self >> TransposedAMat(self.transpose().toAMat()),
    #-----------------------------------------------------------------------
    rChildren := self >> [self.func, self.pkSize, self.P, self.i],
    rSetChild := rSetChildFields("func", "pkSize", "P", "i"),
    #-----------------------------------------------------------------------
    new := (self, func, pkSize, P, i) >> SPL(WithBases(self,
        rec(func       := FF(func),
            pkSize     := pkSize,
            P          := P,
            i          := i,
            dimensions := When(func.__name__ = "FData",
                  [pkSize*func.domain(), pkSize*func.domain()/P],
                  [pkSize*func.range(), pkSize*func.domain()] )
            ))),
    #-----------------------------------------------------------------------
    transpose := self >> GathRecv(self.func, self.pkSize, self.P, self.i),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.name, "(", self.func, ", ", self.pkSize, ", ", self.P, ", ", self.i,")"),
));

#F GathDist(N, pkSize, P, i)
#F Distributed Gather, for parallel code.
#F N=Vector length. pkSize = packetSize P=Number of processors i=processor#
#F Gathers (N/P)*pkSize elements corresponding to the ones to be processed by processor i.
Class(GathDist, BaseMat, SumsBase, rec(
    sums := self >> self,
    dims := self >> self.dimensions,
    isReal := self >> true,
    needInterleavedLeft := self >> false,
    needInterleavedRight := self >> false,
    cannotChangeDataFormat := self >> true,
    totallyCannotChangeDataFormat := self >> true,
    #-----------------------------------------------------------------------
    rChildren := self >> [self.N, self.pkSize, self.P, self.i],
    rSetChild := rSetChildFields("N", "pkSize", "P", "i"),
    #-----------------------------------------------------------------------
    new := (self, N, pkSize, P, i) >> SPL(WithBases(self,
        rec(N          := N,
            pkSize     := pkSize,
            P          := P,
            i          := i,
            func       := FF( fAdd(N, N/P, i*(N/P))  ),
            dimensions := [pkSize*(N/P), pkSize*N]))),
    #-----------------------------------------------------------------------
    area := self >> self.func.domain() * self.func.range(),
    transpose := self >> ScatDist(self.N, self.pkSize, self.P, self.i),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.name, "(", self.N, ", ", self.pkSize, ", ", self.P, ", ", self.i, ")"),
    #-----------------------------------------------------------------------
    toAMat := self >> let(
             pkSize:=self.pkSize,
             n:=pkSize*(self.N/self.P),
             N:=pkSize*self.N,
             i:=self.i,
             P:=self.P,
             func := FF( fAdd(N, N/P, i*(N/P))  ).lambda(),
             AMatMat(List([0..n-1], row -> BasisVec(N, func.at(row).ev())))
    ),
));

#F ScatDist(N, pkSize, P, i)
#F Distributed Scatter. See Doc(GathDist) for more info.
Class(ScatDist, BaseMat, rec(
    sums := self >> self,
    dims := self >> self.dimensions,
    isReal := self >> true,
    needInterleavedLeft := self >> false,
    needInterleavedRight := self >> false,
    cannotChangeDataFormat := self >> true,
    totallyCannotChangeDataFormat := self >> true,
    #-----------------------------------------------------------------------
    rChildren := self >> [self.N, self.pkSize, self.P, self.i],
    rSetChild := rSetChildFields("N", "pkSize", "P", "i"),
    #-----------------------------------------------------------------------
    new := (self, N, pkSize, P, i) >> SPL(WithBases(self,
        rec(N           := N,
            pkSize      := pkSize,
            P           := P,
            i           := i,
            func        := FF( fAdd(N, N/P, i*(N/P))  ),
            dimensions  := [pkSize*N, pkSize*(N/P)]))),
    #-----------------------------------------------------------------------
    area := self >> self.transpose().area(),
    transpose := self >> GathDist(self.N, self.pkSize, self.P, self.i),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.name, "(", self.N, ", ", self.pkSize, ", ", self.P, ", ", self.i, ")"),
    #-----------------------------------------------------------------------
    toAMat := self >> TransposedAMat(self.transpose().toAMat()),
));

#NOTE: P and domain are the same thing, and are redundant for DistSum. Remove
#one of them. Assign P to domain or something like that.

#F ==========================================================================
#F DistSum(<P>, <var>, <domain>, <spl>) - as ISum but makes parallel loop in the code
#F ==========================================================================

Class(DistSum, ISum, rec(

    doNotMarkBB := true,
    new := meth(self, P, var, domain, expr)
        local res;
        Constraint(IsSPL(expr)); 
        # if domain is an integer (not symbolic) it must be non-zero
        Constraint(not IsInt(domain) or domain > 0);
        var.isParallelLoopIndex := true;
        res := SPL(WithBases(self, rec(P:=P, _children := [expr], var := var, domain := domain)));
        res.dimensions := res.dims();
        return res;
    end,
    
    rChildren := self >> Concatenation([self.P], self._children),
    rSetChild := meth(self, n, newChild) 
        if n=1 then self.P := newChild;
          else self._children[n-1] := newChild;
        fi;
    end,

    from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), [rch[1], self.var, self.domain, rch[2]]),

    leftMostParScat  := self >> Collect(self._children[1], @(1,[ScatSend, ScatDist]))[1],
    rightMostParGath := self >> Collect(self._children[1], @(1,[GathRecv, GathDist]))[1],

    print := meth(self, indent, indentStep)
        Print(self.name, "(", self.P, ", ", self.var, ", ", self.domain, ",");
        self._newline(indent + indentStep);
        SPLOps.Print(self._children[1], indent+indentStep, indentStep); #, ", ", self.nt_maps);
        self._newline(indent);
        Print(")");
    end,

));

Class(DistSumLoop, DistSum, rec(

    dims := (self) >> self._children[1].dims()*self.domain

));

#F Comm_Cell(p, pkSize)
#F All-to-all communication, specifically implementing L(p^2, p), where p=# of procs

Class(Comm_Cell, BaseMat, rec(
    cannotChangeDataFormat :=  self >> true,
    totallyCannotChangeDataFormat := self >> true,
    sums := self >> self,
    dims := self >> self.dimensions,
    isReal := self >> true,
    #-----------------------------------------------------------------------
    rChildren := self >> [self.P, self.pkSize],
    rSetChild := rSetChildFields("P", "pkSize"),
    #-----------------------------------------------------------------------
    new := (self, P, pkSize) >> SPL(WithBases(self,
        rec(P           := P,
            pkSize      := pkSize,
            dimensions  := [P*P*pkSize, P*P*pkSize]))),
    #-----------------------------------------------------------------------
    area := self >> self.P*self.P*self.pkSize*self.P*self.P*self.pkSize,
    #NOTE: check.
    transpose := self >> self,
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.name, "(", self.P, ", ", self.pkSize, ")"),
    #-----------------------------------------------------------------------
    toAMat := self >> let(
             pkSize := self.pkSize,
             P      := self.P,
             func   := FF( fTensor(L(P*P, P), fId(pkSize))  ).lambda(),
             AMatMat(List([0..(P*P*pkSize)-1], row -> BasisVec((P*P*pkSize), func.at(row).ev())))
          ),
));



Class(PTensor, BaseMat, SumsBase, rec(
    cannotChangeDataFormat :=  self >> true,
    totallyCannotChangeDataFormat := self >> true,
    sums := self >> PTensor(self.L.sums(), self.P),
    dims := self >> self.L.dims() * self.P,
    isReal := self >> true,
    #-----------------------------------------------------------------------
    rChildren := self >> [self.L, self.P],
    rSetChild := rSetChildFields("L", "P"),
    #-----------------------------------------------------------------------
    new := (self, L, P) >> SPL(WithBases(self,
        rec(L   := L,
            P   := P,
            dimensions := L.dims()*P)
    )),

    #-----------------------------------------------------------------------
    #area := self >> self.L.dimen
    transpose := self >> PTensor(self.L.transpose(), self.P),
    #-----------------------------------------------------------------------

    print := (self,i,is) >> Print(self.name, "(",
        self.L.print(i+is,is), ", ", self.P, ")"),
    #-----------------------------------------------------------------------
    toAMat := self >> Tensor(I(self.P), self.L).toAMat(),
    #-----------------------------------------------------------------------
    isPermutation := False,
    needInterleavedLeft := true,
    needInterleavedRight := true,
    #NOTE: We should be able to stuff data format change into the PTensor
));


#Class(DontTouchMe, Buf, rec(
#    totallyCannotChangeDataFormat := self >> true,
#    cannotChangeDataFormat := self >> true
#));
