
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F ==========================================================================
#F AParSMP(<num_threads>, <tid>) - SMP parallelization tag
#F    .params[1] becomes num_threads
#F    .params[2] becomes tid
#F   
Class(AParSMP, AGenericTag, rec(
    isSMP := true,
    updateParams := meth(self)
        if Length(self.params)=1 then self.params := [self.params[1], threadId()];
        elif Length(self.params)=2 then ;
        else Error("Usage: AParSMP(<num_threads>, [<tid>])");
        fi;
    end
));

#F ==========================================================================
#F SMPBarrier(<nthreads>, <tid>, <spl>) 
#F
#F ==========================================================================
Class(SMPBarrier, Buf, BaseContainer, rec( 
    # inheriting from Buf helps to get extra methods from Buf, for example 
    # .cannotChangeDataFormat (from paradigms/vector). Buf is considered a general wrapper. 
    doNotMarkBB := true,
    new := (self, nthreads, tid, spl) >> Checked(IsPosInt0Sym(tid), IsPosIntSym(nthreads), IsSPL(spl),
        SPL(WithBases(self, rec(_children:=[spl], tid := tid, nthreads := nthreads, dimensions := spl.dims())))),

    dims := self >> self._children[1].dims(),

    rChildren := self >> [self.nthreads, self.tid, self._children[1]],
    rSetChild := meth(self, n, newC)
        if n=1 then self.nthreads := newC;
        elif n=2 then self.tid := newC;
        elif n=3 then self._children[1] := newC;
        else Error("<n> must be in [1..3]");
        fi;
    end
));

#F ==========================================================================
#F SMPSum(<nthreads>, <tid>, <var>, <domain>, <spl>) - parallel loop with <nthreads> threads 
#F
#F As a matrix SMPSum(p, var, domain, spl) == ISum(var, domain, spl)
#F ==========================================================================
Class(SMPSum, ISum, rec(
    doNotMarkBB := true,
    abbrevs := [ (p, var, domain, spl) -> Checked(
                    IsInt(p) or IsScalar(p), IsVar(var), IsInt(p) or IsScalar(domain), IsSPL(spl), 
                    [p, threadId(), var, domain, spl]) ],

    new := meth(self, nthreads, tid, var, domain, spl)
        local res;
        Constraint(IsSPL(spl)); 
	# if domain is an integer (not symbolic) it must be non-zero
	Constraint(IsPosIntSym(domain));
        var.isLoopIndex := true;
        var.range := domain;
	res := SPL(WithBases(self, rec(nthreads:=nthreads, tid:=tid, _children := [spl], var := var, domain := domain)));
	res.dimensions := res.dims();
	return res;
    end,
    
    rChildren := self >> [self.nthreads, self.tid, self.var, self.domain, self._children[1]],

    from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), rch),

    rSetChild := meth(self, n, newChild) 
	if n=1 then self.nthreads := newChild;
	elif n=2 then self.tid := newChild;
	elif n=3 then self.var := newChild;
	elif n=4 then self.var.range := newChild; self.var.isLoopIndex := true;
	elif n=5 then self._children[1] := newChild;
        else Error("<n> must be in [1..5]");
	fi;
    end, 

    print := (self, i, is) >> Print(self.__name__, "(", self.nthreads, ", ", self.tid, ", ", 
        self.var, ", ", self.domain, ",\n", Blanks(i+is), self.child(1).print(i+is, is), "\n",
	Blanks(i), ")", self.printA())
));

#F ==========================================================================
#F SMP(<num_threads>, <tid>, <spl>) - parallel container for rewriting
#F
#F SMP(ISum(...)) -- becomes --> SMPSum(...)
#F ==========================================================================
Class(SMP, BaseContainer, rec(
    doNotMarkBB := true,

    abbrevs := [ (nthreads, spl) -> Checked(IsInt(nthreads) or IsScalar(nthreads), IsSPL(spl), 
                                            [nthreads, threadId(), spl]) ],

    new := (self, nthreads, tid, spl) >>
	SPL(WithBases(self, rec(nthreads:=nthreads, tid:=tid, dimensions := spl.dimensions, _children := [spl]))),
	
    rChildren := self >> [self.nthreads, self.tid, self._children[1]],

    rSetChild := meth(self, n, newChild)
        if   n = 1 then self.nthreads := newChild;
        elif n = 2 then self.tid := newChild;
	elif n = 3 then self._children[1] := newChild;
	else Error("<n> must be in [1..3]"); fi;
	return self;
    end,

    sums := self >> ObjId(self)(self.nthreads, self.tid, self.child(1).sums()),

    normalizedArithCost := self >> self.child(1).normalizedArithCost()
));

