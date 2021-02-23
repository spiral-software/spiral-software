
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F IsNonTerminal ( <obj> )
#F   returns true, if <obj> is a non-terminal
#F
IsNonTerminal := N -> IsRec(N) and IsBound(N.isNonTerminal) and N.isNonTerminal = true;


Class(NonTerminalOps, SPLOps, rec(
));

Declare(NonTerminal);

# ==========================================================================
# NonTerminal
# ==========================================================================
Class(NonTerminal, BaseMat, rec(
    #F create getter methods
    #F    ex: NonTerminal.WithParams("param1_getter_name", "param2_getter_name")
    WithParams := arg >> CopyFields(arg[1], 
	RecList( ConcatList( [2..Length(arg)], 
	    i -> [arg[i], DetachFunc(Subst(self >> self.params[$(i-1)]))] ))),

    isNonTerminal := true, # for nonterm.g and IsNonTerminal() function
    _short_print := false,

    #--------- Transformation rules support --------------------------------
    from_rChildren := (self, rch) >> let(
        t := ApplyFunc(ObjId(self), DropLast(rch, 1)),
        When(Last(rch), t.transpose(), t)
    ),

    rChildren := self >>
        Concatenation(self.params, [self.transposed]),

    rSetChild := meth(self, n, newChild)
        if n <= 0 or n > Length(self.params) + 1
            then Error("<n> must be in [1..", Length(self.params)+1, "]"); fi;
        if n <= Length(self.params) then
            self.params[n] := newChild;
        else
            self.transposed := newChild;
        fi;
        # self.canonizeParams(); ??
        self.dimensions := self.dims();
    end,

    #-----------------------------------------------------------------------
    canonizeParams := meth(self)
        local A, nump;
        nump := Length(self.params);
    if nump = 0 then return; fi;
        for A in self.abbrevs do
            if NumArgs(A) = -1 or NumArgs(A) = nump then
                self.params := ApplyFunc(A, self.params);
                return;
            fi;
        od;
        Error("Nonterminal ", self.name, " can take ", List(self.abbrevs, NumArgs),
              "  arguments, but not ", nump);
    end,

    new := meth(arg)
        local result, self, params;
        self := arg[1];
        params := arg{[2..Length(arg)]};
        result := SPL(WithBases(self, rec(params := params, transposed := false )));
        result.canonizeParams();
        result.dimensions := result.dims();
        return result;
    end,
    #-----------------------------------------------------------------------
    __call__ := ~.new,
    #-----------------------------------------------------------------------
    isTerminal := False,
    #-----------------------------------------------------------------------
    isPermutation := False,
    #-----------------------------------------------------------------------
    transpose := self >>
        Inherit(self, rec(transposed := not self.transposed,
                  dimensions := Reversed(self.dimensions))),
    setTransposed := (self, v) >> When( self.transposed <> v, 
        self.transpose(), self),
    #-----------------------------------------------------------------------
    # .conjTranspose() is needed for RC(T).transpose() == RC(T.conjTranspose())
    # The inert form will only work is non-terminal is not used inside RC,
    # or used inside RC, but not transposed
    conjTranspose := self >> InertConjTranspose(self),
    isInertConjTranspose := self >> self.conjTranspose = NonTerminal.conjTranspose,

    #-----------------------------------------------------------------------
    print := (self, i, is) >> 
	Cond(
            not IsBound(self.params), Print(self.name),
	    Print(self._print(self.params, i, is), 
		  When(self.transposed, ".transpose()"))),
    #-----------------------------------------------------------------------
    toAMat := self >> self.terminate().toAMat(),
    #-----------------------------------------------------------------------
    export := arg >> Error("Can't export a non-terminal"),
    #-----------------------------------------------------------------------
    arithmeticCost := arg >> Error("Can't compute arithmetic cost of a non-terminal"),

    transposeSymmetric := True,
    isSymmetric := False,

    area := self >> Product(self.dims())
));
