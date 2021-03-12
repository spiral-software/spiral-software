
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# ==========================================================================
# SPLSymbol(<name>, <params>)  - obsolete function for creating spl symbols
#
# Note: in this revision all symbol names can be used as functions.
#       SPLSymbol("F", 2) is now F(2)
#
# ==========================================================================
SPLSymbol := function(s, params)
    local sym, perm;
    if not IsList(params) then params := [params]; fi;
    sym  := IsBound(spl.(s));
    if not sym then 
	Error("Symbol '", s, "' does not exist. See Dir(spl).");
    else
	return ApplyFunc(spl.(s), params);
    fi;
end;

#F IsSPLSym(<x>) - returns true for parametrized symbols (F, I, ...)
#F
IsSPLSym := x -> IsRec(x) and IsBound(x._sym) and x._sym;

# ==========================================================================
# Sym - base class for SPL symbols
# ==========================================================================
Class(Sym, BaseMat, rec(
    _sym := true,
    transposed := false,
    type := "Sym",

    new := (self, s, obj) >>
        SPL(Inherit(self, rec( symbol := s,
                   obj := obj,
                   dimensions := obj.dims()))),

   # -------- Transformation rules support ---------------------------------
    rChildren := self >> When(IsList(self.params), self.params, [self.params]),
    rSetChild := meth(self, n, newChild) 
        if IsList(self.params) then
            self.params[n] := newChild;
        else 
            if n<>1 then Error("<n> must be 1"); fi;
            self.params := newChild;
        fi;
        # self.canonizeParams(); ??
        self.dimensions := self.dims();
    end,

    from_rChildren := (self, rch) >> let(
	res := ApplyFunc(ObjId(self), rch).appendAobj(self),
	When(self.transposed, res.transpose(), res)),
    # ----------------------------------------------------------------------

    checkParams := meth(self, params) 
        local nargs, nump;
	nargs := NumArgs(self.def);
	nump := When(IsList(params), Length(params), 1);
        if nargs <> -1 and nargs <> nump then
	    Error("Symbol '", self.name, "' needs ", NumArgs(self.def), " parameters: ",
		ParamsFunc(self.def), "\n");
	fi;
    end,

    canonizeParams := meth(self, params)
        local A, nump;
	nump := Length(params);
        if IsBound(self.abbrevs) then
	    for A in self.abbrevs do
                if NumArgs(A) = -1 or NumArgs(A) = nump then
		    return ApplyFunc(A, params);
		fi;
	    od;
	    return params;
	else
	    return params;
	fi;
    end,
    
    fromDef := meth(arg)
        local result, self, params, h, lkup;
	self := arg[1];
	params := arg{[2..Length(arg)]};
	params := self.canonizeParams(params);
        self.checkParams(params);
	params := When(IsList(params), params, [params]);
	
	h := self.hash;
	if h<>false then 
	    lkup := h.objLookup(self, params);
	    if lkup[1] <> false then return lkup[1]; fi;
	fi;

	result := SPL(WithBases(self, rec(params := params, transposed := false)));
	result.obj := ApplyFunc(result.def, params);
	result.dimensions := Dimensions(result.obj);

	if h<>false then return h.objAdd(result, lkup[2]);
	else return result;
	fi;
    end,

    __call__ := ~.fromDef,

    getObj := self >> When(self.transposed, self.obj.transpose(), self.obj),

    dims          := self >> self.getObj().dims(),
    isPermutation := self >> self.obj.isPermutation(),
    isReal        := self >> self.obj.isReal(), 
    toAMat        := self >> self.getObj().toAMat(),
    free          := self >> Union(List(self.params, FreeVars)),

    #-----------------------------------------------------------------------
    transpose := self >> 
        Inherit(self, rec(transposed := not self.transposed, 
                  dimensions := Reversed(self.dimensions))),
    #-----------------------------------------------------------------------
    equals := meth(self, other)
        return ObjId(other) = ObjId(self)
           and other.params = self.params 
           and self.transposed = other.transposed
	   and self.a = other.a;
    end,
    #-----------------------------------------------------------------------
    print := (self, i, is) >> let(
        params := When(IsList(self.params), self.params, [self.params]), 
        Print(self.__name__, "(", PrintCS(params), ")",
              When(self.transposed, ".transpose()", ""),
              self.printA())),
    #-----------------------------------------------------------------------
    arithmeticCost := (self, costMul, costAddMul) >>
        self.getObj().arithmeticCost(costMul, costAddMul)
));

#F SymFunc(<name>, <func>)
#F      Convert a function (which returns SPL) to SPL symbol. 
#F      Resulting symbol is automatically added to spiral.spl.symbols namespace.
SymFunc := function(name, func)
    local sym;
    sym := rec(notSupported := true, def := func);
    sym.name := name;
    sym.operations := ClassOps;
    sym.__bases__ := [ Sym ];
    CantCopy(sym);
    return sym;
end;

