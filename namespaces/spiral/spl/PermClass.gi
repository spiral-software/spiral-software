
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#P Permutations
#P ------------
#P
#P Under different circumstances different objects are called permutations,
#P even in the context of linear algrebra.
#P
#P Following list describes these objects and their representation in SPIRAL:
#P
#P  1. Regular permutations
#P     (a) in cycle notation : (1,2)(3,4)
#P     (b) as lists          : [2,1,4,3]
#P     (c) as index space mapping function:  i -> i+1 mod N
#P
#P  2. Parametrized permutation classes
#P     [For example stride permutations L(size,str) | size mod str = 0]
#P     Represented as a construction function, which returns an index mapping
#P     function - (c) above.
#P
#P  Conversions: ListPerm     (b) <- (a)   [gap.perm]
#P               PermList     (a) <- (b)   [gap.perm]
#P               PermFunc     (a) <- (c)   [spiral.spl.perm]
#P               ListPermFunc (b) <- (c)   [spiral.spl.perm]
#P
#P -------------------

#F PermFunc(<func>, <size>) . . . . . convert perm. function into explicit GAP perm.
#F
#F  PermFunc converts a 0-based permutation function used in SPLs, into an explicit
#F  GAP permutation. Recall that GAP permutations are 1-based, and are in cycle
#F  representation.
#F
PermFunc := (func, size) ->
    PermList( List([0..size-1], func) + 1 );

#F PermFunc(<func>, <size>) . . . . . convert perm. function into explicit list perm.
#F
#F  PermFunc converts a 0-based permutation function used in SPLs, into an explicit
#F  1-based list permutation. List permutations can be converted to GAP permutations
#F  using PermList. Alternatively PermFunc can be used, wihch returns GAP permutation.
#F
ListPermFunc := (func, size) ->
    List([0..size-1], func) + 1;

# ==========================================================================
# FuncClass
#
# Base class for symbolic functions
# ==========================================================================
Class(FuncClass, BaseMat, Function, rec(
    #-----------------------------------------------------------------------
    # Must be implemented in subclasses
    #-----------------------------------------------------------------------
    lambda := self >> Error("not implemented"), 
    domain := self >> Error("not implemented"), 
    range := self >> Error("not implemented"), 

    #-----------------------------------------------------------------------
    _perm := true,
    isReal := True,
    isPermutation := self >> false,
    perm := self >> Checked(self.isPermutation(), PermFunc(x->self.lambda().at(x), self.range())),

    #-----------------------------------------------------------------------
    print   := (self,i,is) >> Print(
	self.name, "(", PrintCS(self.params), ")", When(self.transposed, ".transpose()")),
    #-----------------------------------------------------------------------
    equals := (self, other) >> ObjId(other) = ObjId(self) and self.params=other.params,
    dims := self >> [self.range(), self.domain()],

    advdims   := self >> [ [[ self.range() ]], [[ self.domain() ]] ],

    # size along each dimension, for a multidimensional range
    # for compatibility with ClassSPL, we wrap this into another list 
    # (list of outputs, since ClassSPL can have >1 output)
    advrange  := self >> [[ self.range() ]],

    # size along each dimension, for a multidimensional domain
    # for compatibility with ClassSPL, we wrap this into another list 
    # (list of outputs, since ClassSPL can have >1 output)
    advdomain := self >> [[ self.domain() ]],

    # dimensionality of range (1-d, 2-d, etc)
    advrangeDim := self >> Length(self.advrange()[1]),
    # dimensionality of domain (1-d, 2-d, etc)
    advdomainDim := self >> Length(self.advdomain()[1]),
    #-----------------------------------------------------------------------
    toAMat := self >> Gath(self).toAMat(),
    #-----------------------------------------------------------------------
    arithmeticCost := (self, costMul, costAddMul) >> costMul(0) - costMul(0),
    #-----------------------------------------------------------------------
    transpose := self >> CopyFields(self, rec(transposed := not self.transposed )),
    #-----------------------------------------------------------------------
    conjTranspose := self >> self.transpose(),
    #-----------------------------------------------------------------------
    normalizedArithCost := self >> 0,
    #-----------------------------------------------------------------------
    free := self >> Union(List(self.params, FreeVars)),
    #-----------------------------------------------------------------------
    # Rewrite rules support 
    #
    from_rChildren := (self, rch) >> CopyFields(ApplyFunc(ObjId(self), rch), 
	rec(transposed:=self.transposed)),

    # NOTE: self.transposed not exposed
    rChildren := self >> self.params, 

    rSetChild := meth(self, n, newChild)
        self.params[n] := newChild;
        # self.canonizeParams(); ??
	self.dimensions := self.dims();
    end,
    # ----------------------------------------------------------------------
    checkParams := meth(self, params)
        local nargs, nump;
	nargs := NumArgs(self.def);
	nump := Length(params);
	if nargs <> -1 and nargs <> nump then
            Error(self.name, " needs ", NumArgs(self.def), " parameters: ",
		ParamsFunc(self.def), "\n");
	fi;
    end,
    #-----------------------------------------------------------------------
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
    #-----------------------------------------------------------------------
    __call__ := meth(arg)
        local result, self, params, lkup, h;
        self := arg[1];
        params := arg{[2..Length(arg)]};
        params := self.canonizeParams(params);
        self.checkParams(params);

        h := self.hash;
        if h<>false then
            lkup := h.objLookup(self, params);
            if lkup[1] <> false then return lkup[1]; fi;
        fi;
        
        result := SPL(WithBases(self, rec(params := params, transposed := false)));
        result := Inherit(result, ApplyFunc(result.def, params));
	result.dimensions := result.dims();
        
        if h<>false then return h.objAdd(result, lkup[2]);
        else return result;
        fi;
    end,

# obsolete
#    checkInverse := self >> Checked(self.isPermutation(),
#        PermFunc(self.direct, self.range()) * PermFunc(self.inverse, self.domain())),
));

Class(PermClass, FuncClass, rec(
    isPermutation := self >> true,
    toAMat := self >> Gath(self).toAMat(),
    equals := (self, other) >> ObjId(other) = ObjId(self) and self.params=other.params
));
