
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(GathMem);
Declare(ScatMem);

Declare(MultiBufISum);
Declare(MultiBufISumFinal);
Declare(MemISum);
Declare(R2Sum);

Declare(MultiBufDistSum);

Class(FDataOfs_mbuf, FDataOfs);
Class(RemoteData, Data, rec(
    new := (self, var, altbuf, value, ofs, spl) >> Checked(IsVar(var), IsSPL(spl), SPL(WithBases(self, rec(
           var:=var,
           altbuf:=altbuf,
           value:=value,
           ofs := ofs,
           _children:=[spl],
           dimensions := spl.dimensions)))),
    #print := (self) >> Print(DarkRedStr(self.name), "(", self.var, ", ", self.len, ", ", self.ofs, ")")


));

# Used to mark BBs which have RemoteDataUnrolled 
Class(RemoteDataInit, BB, rec(
    doNotMarkBB := true
));

Class(RemoteDataUnrolled, RemoteData);
Class(RemoteDataNoBody, RemoteData);

#F GathMem(func, pkSize)
#F Explicit Gather from memory (like DMA)
Class(GathMem, BaseMat, SumsBase, rec(
    sums := self >> self,
    isReal := self >> true,
    dims := self >> self.dimensions,
    needInterleavedLeft := self >> false,
    needInterleavedRight := self >> false,
    cannotChangeDataFormat := self >> true,
    totallyCannotChangeDataFormat := self >> true,
    #-----------------------------------------------------------------------
    rChildren := self >> [self.func, self.pkSize],
    rSetChild := rSetChildFields("func", "pkSize"),
    #-----------------------------------------------------------------------
    new := (self, func, pkSize) >> SPL(WithBases(self,
        rec(func       := FF(func),
            pkSize     := pkSize,
            dimensions := [pkSize*func.domain(), pkSize*func.range()]
            ))),
    #-----------------------------------------------------------------------
    area := self >> self.pkSize * self.func.domain(),

    transpose := self >> ScatMem(self.func, self.pkSize),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.name, "(", self.func, ", ", self.pkSize, ")"),
    #-----------------------------------------------------------------------
    toAMat := self >> 
          let(pkSize   := self.pkSize, 
              n        := pkSize*self.func.domain(),
              N        := pkSize*self.func.range(),
              func     := fTensor(self.func, fId(pkSize)).lambda(),
              AMatMat(List([0..n-1], row -> BasisVec(N, func.at(row).ev())))
          )
));

#F ScatMem(func, pkSize)
#F Explicit Scatter to memory (like DMA)
Class(ScatMem, BaseMat, rec(
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
    rChildren := self >> [self.func, self.pkSize],
    rSetChild := rSetChildFields("func", "pkSize"),
    #-----------------------------------------------------------------------
    new := (self, func, pkSize) >> SPL(WithBases(self,
        rec(func       := FF(func),
            pkSize     := pkSize,
            dimensions := [pkSize*func.range(), pkSize*func.domain()]
            ))),
    #-----------------------------------------------------------------------
    transpose := self >> GathMem(self.func, self.pkSize),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.name, "(", self.func, ", ", self.pkSize, ")"),
));

Class(GathMemDummy, GathMem);
Class(ScatMemDummy, ScatMem);

#F ==========================================================================
#F MultiBufSum(<var>, <domain>, <scat>, <spl>, <gath>) 
#F - Like ISum but makes multibuffered loop in the code
#F ==========================================================================

Class(MultiBufISum, ISum, rec(
    doNotMarkBB := true,
    cannotChangeDataFormat := self >> true, # Because ScatMem/GathMem (hidden) cannot change data format
    #needSeparateVRC := self >> true,

    new := meth(self, var, domain, scatmem, expr, gathmem)
        local res;
        Constraint(IsSPL(expr)); 
        # if domain is an integer (not symbolic) it must be non-zero
        Constraint(not IsInt(domain) or domain > 0);
        res := SPL(WithBases(self, rec(
               var := var,
               domain := domain,
               scatmem := scatmem,
               _children := [expr],
               gathmem := gathmem
        )));
        res.dimensions := expr.dims()*domain;
        return res;
    end,

    rChildren := self >> [self.scatmem, self._children[1], self.gathmem],

    rSetChild := meth(self, n, what)
       if n=1 then self.scatmem := what;
       elif n=2 then self._children[1] := what;
       elif n=3 then self.gathmem := what;
       fi;
    end,



    from_rChildren := (self, rch) >> ObjId(self)(self.var, self.domain, rch[1], rch[2], rch[3]),

    at := (self, u) >> SubstVars(Copy(self.scatmem*self._children[1]*self.gathmem), rec((self.var.id):=u)),

    dims := (self) >> self._children[1].dims()*self.domain,
    rng := (self) >> self.scatmem.rng(), # Since this is hidden
    dmn := (self) >> self.gathmem.dmn(), # Since this is hidden

    print := meth(self, indent, indentStep)
        local expr, scatmem, gathmem;
        expr := self._children[1];
        scatmem := self.scatmem;
        gathmem := self.gathmem;

        Print(self.name, "(", self.var, ", ", self.domain);
        self._newline(indent + indentStep);
        SPLOps.Print(scatmem, indent+indentStep, indentStep);

        self._newline(indent + indentStep);
        SPLOps.Print(expr, indent+indentStep, indentStep);

        self._newline(indent + indentStep);
        SPLOps.Print(gathmem, indent+indentStep, indentStep);
        self._newline(indent);
        Print(")");
    end
    
));


#NOTE: this probably needs a needSeparateVRC. Check and add.
Class(MultiBufDistSum, DistSum, rec(
    new := meth(self, var, domain, scatmem, expr, gathmem)
        local res;
        Constraint(IsSPL(expr)); 
        # if domain is an integer (not symbolic) it must be non-zero
        Constraint(not IsInt(domain) or domain > 0);
        var.isParallelLoopIndex := true;
        res := SPL(WithBases(self, rec(
               var := var,
               domain := domain,
               scatmem := scatmem,
               _children := [expr],
               gathmem := gathmem
        )));
        res.dimensions := expr.dims()*domain;
        return res;
    end,

    rChildren := self >> [self.scatmem, self._children[1], self.gathmem],

    rSetChild := meth(self, n, what)
       if n=1 then self.scatmem := what;
       elif n=2 then self._children[1] := what;
       elif n=3 then self.gathmem := what;
       fi;
    end,

    from_rChildren := (self, rch) >> ObjId(self)(self.var, self.domain, rch[1], rch[2], rch[3]),

    at := (self, u) >> SubstVars(Copy(self.scatmem*self._children[1]*self.gathmem), rec((self.var.id):=u)),

    dims := (self) >> self._children[1].dims()*self.domain,

    print := meth(self, indent, indentStep)
        local expr, scatmem, gathmem;
        expr := self._children[1];
        scatmem := self.scatmem;
        gathmem := self.gathmem;

        Print(self.name, "(", self.var, ", ", self.domain);
        self._newline(indent + indentStep);
        Print(scatmem);

        self._newline(indent + indentStep);
        SPLOps.Print(expr, indent+indentStep, indentStep); #, ", ", self.nt_maps);

        self._newline(indent + indentStep);
        Print(gathmem);
        self._newline(indent);
        Print(")");
    end
    
));

Class(MultiBufISumFinal, MultiBufISum);
Class(MemISum, MultiBufISum);
Class(MemISumFinal, MemISum);

Class(R2Sum, ISum, rec(
    doNotMarkBB := true,
    cannotChangeDataFormat := self >> true, # Because ScatMem/GathMem (hidden) cannot change data format
    #needSeparateVRC := self >> true,

    new := meth(self, var1, domain1, var2, domain2, expr)
        local res;
        Constraint(IsSPL(expr)); 
        # if domain is an integer (not symbolic) it must be non-zero
        Constraint(not IsInt(domain1) or domain1 > 0);
        Constraint(not IsInt(domain2) or domain2 > 0);
        res := SPL(WithBases(self, rec(
               var1 := var1,
               domain1 := domain1,
               var2 := var2,
               domain2 := domain2,
               _children := [expr]
        )));
        res.dimensions := expr.dims()*domain1*domain2;
        return res;
    end,

    rChildren := self >> [self._children[1]],

    rSetChild := meth(self, n, what)
       if n=1 then self._children[1]  := what;
       fi;
    end,

    from_rChildren := (self, rch) >> ObjId(self)(self.var1, self.domain1, self.var2, self.domain2, rch[1]),
    
    at := (self, u) >> SubstVars(Copy(self.scatmem*self._children[1]*self.gathmem), rec((self.var.id):=u)),

    dims := (self) >> self._children[1].dims()*self.domain1*self.domain2,

    #rng := (self) >> self.scatmem.rng(), # Since this is hidden
    #dmn := (self) >> self.gathmem.dmn(), # Since this is hidden

    print := meth(self, i, is)
        local expr, scatmem, gathmem;
        expr := self._children[1];
        #scatmem := self.scatmem;
        #gathmem := self.gathmem;

        Print(self.name, "(", self.var1, ", ", self.domain1, ", ", self.var2, ", ", self.domain2);
        #self._newline(i + is);
        #SPLOps.Print(scatmem, i+is, is);

        self._newline(i + is);
        SPLOps.Print(expr, i+is, is);

        #self._newline(i + is);
        #SPLOps.Print(gathmem, i+is, is);
        self._newline(i);
        Print(")");
    end
    
));
