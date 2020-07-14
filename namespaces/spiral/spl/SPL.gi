
# Copyright (c) 2018-2020, Carnegie Mellon University
# See LICENSE for details


Rows := s -> Cond(
    IsMat(s),  
        Length(s),

    IsBound(s.rng) and IsBound(s.dmn),
        s.dims()[1],
	
    IsValue(s) or IsSymbolic(s),
        Checked(IsArrayT(s.t), IsArrayT(s.t.t), 
	    s.t.size),

    s.dimensions[1]
);

Cols := s -> Cond(
    IsMat(s),  
        Length(s[1]),

    IsBound(s.rng) and IsBound(s.dmn),
        s.dims()[2],
	
    IsValue(s) or IsSymbolic(s),
        Checked(IsArrayT(s.t), IsArrayT(s.t.t), 
	    s.t.t.size),

    s.dimensions[2]
);

olRows := s -> Flat([Rows(s)]);
olCols := s -> Flat([Cols(s)]);

#F SPL(<rec>)
#F    Set operations field to SPLOps. This function should be called
#F    to initialize SPL instances.
SPL := function(record)
   record.operations := SPLOps;
   return record;
end;

HashAsSPL := o -> Cond(
    IsList(o), 
        List(o, HashAsSPL),
    not IsRec(o) or (not IsBound(o.hashAs) and not IsBound(o.from_rChildren)), 
        o,
    IsBound(o.hashAs), 
        o.hashAs(),
    IsValue(o),
        o.v,
    o.from_rChildren(List(o.rChildren(), HashAsSPL))
);

# ==========================================================================
# ClassSPL
#
# Base class for all SPL constructs
# ==========================================================================
Class(ClassSPL, AttrMixin, rec(
    isSPL := true,
    transposed := false,

#DD this allows objects which are not TaggedNonTerminals to drop tags automagically
#DD and without error.
    withTags := (self, t) >> self,

    _short_print := false,
    _newline := i ->  Print("\n", Blanks(i)),
    _indent  := i ->  Print(Blanks(i)),
    _indentStr := Blanks,


    __call__ := meth(arg)
        local self, params, nump, A,p,res,h,lkup;
        self := arg[1];
        params := arg{[2..Length(arg)]};
        nump := Length(params);

        if not IsBound(self.new)  then
            Error("Constructor for this class is not implemented");
        elif IsBound(self.abbrevs) and self.abbrevs <> [] then
            for A in self.abbrevs do
                if NumArgs(A) = -1 or NumArgs(A) = nump then
                    params := ApplyFunc(A, params);
                fi;
            od;
        fi;

        if NumArgs(self.new)-1 <> Length(params) then
            Error("Constructor requires ", NumArgs(self.new)-1, " parameters (",
                Length(arg)-1, " given): ",
                ParamsMeth(self.new));
        else
            h := self.hash;
            if h<>false then
                lkup := h.objLookup(self, params);
                if lkup[1] <> false then return lkup[1]; fi;
            fi;

            res := ApplyFunc(self.new, params);

            if h<>false then return h.objAdd(res, lkup[2]);
            else return res;
            fi;
        fi;
    end,

    hash := false,

    checkDims := self >> DimensionsMat(MatSPL(self)) = self.dimensions,

    #-----------------------------------------------------------------------
    # create a new object with .<name> field set to true
    setAttr := meth(self, name)
        local s;
        s:= Copy(self);
        s.(name) := true;
        return s;
    end,
    # ----------------------------------------------------------------------
    # create a new object with .<name> field set to <val>
    setAttrTo := meth(self, name, val)
        local s;
        s:= Copy(self);
        s.(name) := val;
        return s;
    end,

    #---------Backwards Compatibility for the new dimension system------
    setDims := meth(self) self.dimensions := self.dims(); return self; end,

    dims := self >> [ StripList(List(self.rng(), l -> l.size)), 
	              StripList(List(self.dmn(), l -> l.size)) ],

    advdims := (self) >> let(d := self.dims(), [ [[ d[1] ]], [[ d[2] ]] ]),
    arity   := (self) >> List(self.dims(), e -> Length(Flat([e]))),

    TType:=TUnknown,

    rng := meth(self) local d;
        if IsBound(self.dims) then
            d := Flat([self.dims()[1]]);
        else 
            d := [self.dimensions[1]];
        fi;
        if IsBound(self.a.t_out) then
            return List(TransposedMat([self.a.t_out, d]), e -> TArray(e[1], e[2]));
        else
            return List(d, e -> TArray(self.TType, e));
        fi;
    end,

    dmn := meth(self) local d;
        if IsBound(self.dims) then
            d := Flat([self.dims()[2]]);
        else 
            d := [self.dimensions[2]];
        fi;
        if IsBound(self.a.t_in) then
            return List(TransposedMat([self.a.t_in, d]), e -> TArray(e[1], e[2]));
        else
            return List(d, e -> TArray(self.TType, e));
        fi;
    end,

    free := self >> Union(List(self.rChildren(), FreeVars)),

    equals := (self, o) >>
        ObjId(self) = ObjId(o) and self.rChildren() = o.rChildren() and self.a = o.a,

    lessThan := (self, o) >> Cond(
        ObjId(self) <> ObjId(o), ObjId(self) < ObjId(o),
        [ ObjId(self), self.rChildren(), self.a ] < [ ObjId(o), o.rChildren(), o.a ]
    ),

    from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), rch).appendAobj(self),


    print := (self, i, is) >> self._print(self.rChildren(), i, is),

    _print := meth(self, ch, indent, indentStep)
        local s, first, newline;

	if self._short_print or ForAll(ch, x->not IsRec(x) or IsSPLSym(x) or IsSPLMat(x)) then
	    newline := Ignore;
	else 
	    newline := self._newline;
	fi;

	first := true;
        Print(self.__name__, "(");
	for s in ch do
            if(first) then first:=false;
            else Print(", "); fi;
            newline(indent + indentStep);
            When(IsSPL(s) or (IsRec(s) and IsBound(s.print) and NumGenArgs(s.print)=2),
                 s.print(indent + indentStep, indentStep), 
		 Print(s));
	od;
	newline(indent);
	Print(")");
        self.printA();

	if IsBound(self._setDims) then
            Print(".overrideDims(", self._setDims, ")");
	fi;
    end,

    printlatex := meth(self)
        local s, first, newline, i;

	first := true;
        Print("(");
        i := Length(self.rChildren());
        for s in self.rChildren() do
            When(IsSPL(s) or (IsRec(s) and IsBound(s.printlatex) and NumGenArgs(s.printlatex)=0),
                 s.printlatex(),
		 Print(s));
                 if i >=2 then 
                    Print(" ", When(IsBound(self.latexSymbol), self.latexSymbol, ""), " ");
                 fi;
                 i := i-1;
	od;
	Print(")");
    end,



    overrideDims := (self, dims) >> CopyFields(self, rec(_setDims := dims, dimensions := dims)),

    terminate     := self >> self.from_rChildren(List(self.rChildren(), x->When(IsSPL(x), x.terminate(), x))),

    # ------------------------- Required methods ---------------------------\
    transposeSymmetric := True,
    dims          := meth(self) Error("Not implemented"); end,
    isPermutation := meth(self) Error("Not implemented"); end,
    isTerminal    := meth(self) Error("Not implemented"); end,
    isReal        := meth(self) Error("Not implemented"); end,
    isInplace     := self >> Rows(self)=Cols(self) and let(ch:=self.children(), Cond(Length(ch)=0, false, ForAll(ch, x->x.isInplace()))),
    children      := meth(self) return []; end,
    numChildren   := meth(self) return 0; end,
    child         := meth(self,n) Error("Not implemented"); end,
    setChild      := meth(self,n,what) Error("Not implemented"); end,
    toAMat        := meth(self) Error("Not implemented"); end,
    transpose     := meth(self) Error("Not implemented"); end,
    conjTranspose  := meth(self) Error("Not implemented"); end,
));


#F <SPL> * <SPL>
#F <scalar> * <SPL>
#F   is equivalent to ComposeSPL and ScalarMultiple resp.
#F
SPLOps.\* := (S1, S2) ->
    Cond(IsSPL(S1) and IsSPL(S2),   Compose(S1, S2),
         IsSPL(S2),                 Scale(S1, S2),
     IsSPL(S1),                 Scale(S2, S1),
     Error("do not know how to compute <S1> * <S2>"));

#F <SPL> + <SPL>
#F   is equivalent to SUM(<S1>, <S2>)
#F
SPLOps.\+ := (S1, S2) ->
    Cond(IsSPL(S1) and IsSPL(S2),   SUM(S1, S2),
     Error("do not know how to compute <S1> + <S2>"));

#F S1 ^ S2
#F   is equivalent to ConjugateSPL(S1, S2).
#F
SPLOps.\^ := (S1, S2) ->
    When(IsSPL(S1) and IsSPL(S2),
         Conjugate(S1, S2),
     Error("do not know how to compute S1 ^ S2"));
