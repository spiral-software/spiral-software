
# Copyright (c) 2018-2020, Carnegie Mellon University
# See LICENSE for details


Declare(ICompose, ISum, fTensor, fBase, fId, fCompose);
Declare(RC);

# ===========================================================================
# SPL Sums Notation
# ===========================================================================
# Blk - block
# Blk1 - 1x1 block
# Data(var, value, expr) - data
# ISum - iterative sum
# SUM - sum
#
# Gath(N,n,func) - gather matrix
# Scat(N,n,func) - scatter matrix
# Prm(N, read_func, write_func) - read: output->input, write: input->output
# Conj, ConjL, ConjR, ConjLR - generalized conjugation (arbitrary/no matrix to left and right)
# ConjDiag to conjugate block diagonals
# ===========================================================================

Class(SumsBase, rec(
    isSums := true,
    area := self >> Sum(self.children(), x->x.area()),
    sums := meth(self)
       local children, i, res;
       res := Copy(self);
       children := Map(res.rChildren(), c -> Cond(IsSPL(c), c.sums(), c));
       for i in [1..Length(children)] do
           res.rSetChild(i, children[i]);
       od;
       return res;
    end
));

Compose.area   := self >> Sum(self.children(), x->x.area());
Diag.area      := self >> Rows(self);
Scale.area     := self >> 0;
IsSumsSPL := o -> IsRec(o) and
                ((IsBound(o.isSums) and o.isSums) or
                 (Same(ObjId(o), Compose) and ForAll(o.children(), IsSumsSPL)));

# ==========================================================================
# TCast(<n>, <to_type>, <from_type>) - type conversion on <n> elements
# ==========================================================================
Class(TCast, SumsBase, Sym, rec(
    abbrevs := [
        (n, to_type)            -> Checked(IsPosInt0Sym(n), IsType(to_type), [n, to_type, TUnknown]),
        (n, to_type, from_type) -> Checked(IsPosInt0Sym(n), IsType(to_type), [n, to_type, from_type])
    ],
    def := (n, to_type, from_type) -> Perm((), n),
    dmn := self >> [ TArray(self.params[3], self.params[1]) ],
    rng := self >> [ TArray(self.params[2], self.params[1]) ],

    transpose := self >> self,
    conjTranspose := self >> self,
    inverse := self >> self,
    isSymmetric := True,
    isPermutation := False,
));

# ==========================================================================
# BB(<spl>) - basic block container, serves as barrier for rule application
# ==========================================================================
Class(BB, SumsBase, BaseContainer, rec(isBlock:=true,
    rng := meth(self) return self._children[1].rng(); end,
    dmn := meth(self) return self._children[1].dmn(); end,
));

Class(Buf, SumsBase, BaseContainer, rec(
    rng := meth(self) return self._children[1].rng(); end,
    dmn := meth(self) return self._children[1].dmn(); end
));

Declare(PushL, PushR, PushLR, NoPullLeft, NoPullRight);

# Forces propagation into *left* construct (e.g. ISum, RecursStep, BB, etc)
# Also .sums() conversion always returns same object, ie conversion of
# child is not forced. This is done to avoid Sigma-SPLizing constructs
# that we want to pull in .
Class(PushL, Buf, rec(
    sums := self >> self,
    transpose := self >> PushR(self.child(1).transpose()),
    normalizedArithCost := self >> self.child(1).normalizedArithCost(),
));

# Forces propagation into *right* construct (e.g. ISum, RecursStep, BB, etc)
# Also .sums() conversion always returns same object, ie conversion of
# child is not forced. This is done to avoid Sigma-SPLizing constructs
# that we want to pull in .
Class(PushR, Buf, rec(
    sums := self >> self,
    transpose := self >> PushL(self.child(1).transpose()),
    normalizedArithCost := self >> self.child(1).normalizedArithCost(),
));

# Allows propagation into either right or left construct (e.g. ISum, RecursStep, BB, etc)
# Also .sums() conversion always returns same object, ie conversion of
# child is not forced. This is done to avoid Sigma-SPLizing constructs
# that we want to pull in .
Class(PushLR, Buf, rec(sums := self >> self));

# Prevents propagation into constructs on both sides (e.g. ISum, RecursStep, BB, etc)
Class(NoPull, Buf);

# Prevents pull-in. Used for distributed stuff.
Class(NoPull_Dist, Buf);

# Prevents propagation pulling in of diags
Declare(NoDiagPullinRight);
Class(NoDiagPullin, Buf);
Class(NoDiagPullinLeft, Buf, rec(
    transpose := self >> NoDiagPullinRight(self.child(1).transpose())
));
Class(NoDiagPullinRight, Buf, rec(
    transpose := self >> NoDiagPullinLeft(self.child(1).transpose())
));

# Prevents propagation into left construct (e.g. ISum, RecursStep, BB, etc)
Class(NoPullLeft, Buf, rec(transpose := self >> NoPullRight(self.child(1).transpose())));

# Prevents propagation into right construct (e.g. ISum, RecursStep, BB, etc)
Class(NoPullRight, Buf, rec(transpose := self >> NoPullLeft(self.child(1).transpose())));

# Top level wrapper for Sigma-SPL formulas
Class(Formula, BaseContainer, SumsBase);

#F RecursStep(<expr>)
#F RecursStep(<yofs>, <xofs>, <expr>) - with implicit
#F     fAdd on Gath (xofs) and Scat (yofs) side
#F
Class(RecursStep, SumsBase, BaseContainer, rec(
    abbrevs := [ ch -> [0,0,ch],
                 (yofs,xofs,ch) -> [yofs, xofs, ch] ],
    new := (self, yofs, xofs, ch) >> SPL(WithBases(self,
    rec(yofs:=yofs, xofs:=xofs, _children := [ch], dimensions := ch.dims()))),
    rChildren := self >> [self.yofs, self.xofs, self._children[1]],
    rSetChild := meth(self, n, what)
        if n=1 then self.yofs := what;
        elif n=2 then self.xofs := what;
        elif n=3 then self._children[1] := what;
        else Error("<n> must be in [1..3]");
        fi;
    end,
));

Class(Inplace, SumsBase, BaseContainer, rec(
  rng:=self>>self._children[1].rng(),
  dmn:=self>>self._children[1].dmn(),
  numops:=self >>0, # YSV: what is this? pls remove or document
  toNonInplace := self >> self._children[1],
  isInplace := self >> true,
  normalizedArithCost := self >> self._children[1].normalizedArithCost(),
));

Class(LStep, SumsBase, BaseContainer, rec(
    toAMat := self >> AMatMat(Sum([I(Rows(self)), self.child(1)], MatSPL))
));

Declare(RTWrap); # to avoid complaints in .transpose

Class(RTWrap, SumsBase, BaseContainer, rec(
    new := (self, rt) >> Checked(Global.formgen.IsRuleTree(rt),
    SPL(WithBases(self, rec(
        rt   := rt,
        root := rt.node))).setDims()),

    area := self >> Rows(self) * Cols(self),
    children := self >> [self.rt],
    child := (self, n) >> When(n=1, self.rt, Error("<n> must be 1")),
    setChild := rSetChildFields("rt"),
    rSetChild := ~.setChild,
    rChildren := ~.children,

    dims          := self >> self.rt.dims(),
    isPermutation := self >> self.rt.node.isPermutation(),
    isReal        := self >> self.rt.node.isReal(),
    toAMat        := self >> self.rt.node.toAMat(),

    transpose     := self >> RTWrap(self.rt.transpose()),
    conjTranspose := self >> InertConjTranspose(self),
    isInertConjTranspose := True,
));

# ==========================================================================
# COND(<spl>) - This is a 'switch' statement for SPLs 
# ==========================================================================
Class(COND, SumsBase, BaseContainer, rec(
    abbrevs := [ arg -> let(f:=Flat(arg), [f[1], Drop(f, 1)]) ],
    new := (self, cond, spls) >> SPL(WithBases(self, rec(
        _children := spls,
        dimensions := spls[1].dimensions,
        cond := cond))),

    toAMat := self >> When(self.cond.ev()=V(true) or self.cond.ev()=1,
        self.child(1).toAMat(),
        self.child(2).toAMat()),

    rChildren := self >> Concatenation([self.cond], self._children),
    rSetChild := meth(self, n, newC)
        if n = 1 then self.cond := newC;
        else self.setChild(n-1, newC);
        fi;
    end,

    area := self >> Maximum(List(self.children(), x->x.area())),

    sums := self >> CopyFields(self, rec(_children := List(self._children, x->x.sums()))),
    transpose := self >> CopyFields(self, rec(_children := List(self._children, x->x.transpose()))),
));

Class(RC, SumsBase, BaseContainer, rec(
    dims := self >> let(d:=self.child(1).dims(), [2*d[1], 2*d[2]]),
    isReal := self >> true,

    # when RC(M) is transposed with M - complex, not only M is transposed, but also
    # each complex element of M as a 2x2 matrix is transposed == complex conjugation
    transpose := self >> CopyFields(self, rec(_children := [self.child(1).conjTranspose()],
        dimensions := [self.dimensions[2], self.dimensions[1]])),

    # RC(.) is real, conjTranspose is just a regular transpose
    conjTranspose := self >> self.transpose(),
    inverse := self >> CopyFields(self, rec(_children := [self.child(1).inverse()],
        dimensions := [self.dimensions[2], self.dimensions[1]])),

    sums := self >> CopyFields(self, rec(_children := [self.child(1).sums()])),
    area := self >> 2*self.child(1).area(),
    toAMat := self >> AMatMat(RealMatComplexMat(MatSPL(self.child(1)))),
    createCode := self >> Cond(IsBound(self.child(1).createCode), RC(self.child(1).createCode()), self),

    # assume that normalizedArithCost() always returns cost in real ops
    normalizedArithCost := self >> self.child(1).normalizedArithCost(),

));

# This takes a real matrix that can be seen as RC(A) and returns A as complex matrix
Class(CR, SumsBase, BaseContainer, rec(
    dims := self >> List(self.child(1).dimensions, e -> _unwrap(div(e,2))),

    # the derived matrix is real, but over the complex field
    isReal := self >> false,

    transpose := self >> CopyFields(self, rec(
	_children := [self.child(1).transpose()],
        dimensions := [self.dimensions[2], self.dimensions[1]])),

    # CR(.) is complex, but all entries are real, conjTranspose is just a regular transpose
    conjTranspose := self >> self.transpose(),

    inverse := self >> CopyFields(self, rec(_children := [self.child(1).inverse()],
        dimensions := [self.dimensions[2], self.dimensions[1]])),

    sums := self >> CopyFields(self, rec(_children := [self.child(1).sums()])),

    area := self >> 1/2*self.child(1).area(),

    toAMat := self >> let(mat := MatSPL(self.child(1)),
        rmat := List(mat{2*[1..Length(mat)/2]}, m -> m{2*[1..Length(m)/2]}),
        AMatMat(rmat)
        ),

    # assume that normalizedArithCost() always returns cost in real ops
    normalizedArithCost := self >> self.child(1).normalizedArithCost(),
    vcost := self >> self.child(1).vcost()

));

# ==========================================================================
# Blk(<mat>) - matrix block
# ==========================================================================
# Note: Blk should not check for M being a matrix, 
#   otherwise cant reuse Blk for vector code
Class(Blk, SumsBase, Mat, rec(
    new := (self, M) >> SPL(WithBases(self, rec(
            element := M,
            TType   := Cond( # NOTE: add checks to M
                            IsList(M),     UnifyTypes(List(Flat(M), InferType)),
                            IsValue(M),    M.t.t,
			    IsSymbolic(M), M.t.t),
			))).setDims(),
    area := self >> Length(Filtered(Flat(self.element), k -> k<>0)),
    new  := (self, M) >> SPL(WithBases(self, rec(element := M))).setDims(),
    dims := self >> Dimensions(self.element)
));

# ==========================================================================
# Blk1(<val>) - 1x1 block
# ==========================================================================
Class(Blk1, SumsBase, BaseMat, rec(
    # Compare mathematically Blks disregarding differences in way to express code-level elements.
#    new := (self, val) >> SPL(WithBases(self, rec(dimensions:=[1,1], element:=val))),
#    toAMat := self >> AMatMat([[EvalScalar(Eval(self.element))]]),
    new := (self, val) >> SPL(WithBases(self, rec(dimensions:=[1,1], element:=EvalScalar(Eval(val))))),
    toAMat := self >> AMatMat([[self.element]]),
    transpose := self >> self,
    conjTranspose := self >> CopyFields(self, rec(element := Global.Conjugate(self.element))),
    inverse := self >> CopyFields(self, rec(element := 1 / self.element)),
    area := self >> 1,
    dims := self >> [1,1],
));


# ==========================================================================
# BlkConj() - pseudo 1x1 matrix when multiplied w/ complex number conjugates it
# ==========================================================================
Class(BlkConj, SumsBase, BaseMat, rec(
    rChildren := self >> [],
    rSetChild := (self, n, what) >> Error("no children"),
    new := (self) >> SPL(WithBases(self, rec(dimensions:=[1,1]))),
    toAMat := self >> AMatMat([[1]]),
    transpose := self >> self,
    conjTranspose := self >> self,
    inverse := self >> self,
    area := self >> 1
));

# ==========================================================================
# Data(<var>, <value>, <spl>) - introduces a data constant bound in <spl>
# ==========================================================================
Class(Data, SumsBase, BaseContainer, rec(
    new := (self, var, value, spl) >> Checked(IsVar(var), IsSPL(spl),
	SPL(WithBases(self, rec(
	    var := var, value := value, _children := [spl],
            dimensions := spl.dims())))),
    #-----------------------------------------------------------------------
    area := self >> self.child(1).area(),
    #-----------------------------------------------------------------------
    eval := meth(self)
        local d, c;
        if IsBound(self._evaluated) then return self._evaluated;
        else
            d := Cond(
                IsValue(self.value) or IsSymbolic(self.value), self.value,
                IsBound(self.value.tolist), V(self.value.tolist()),
                IsSPL(self.value), V(MatSPL(self.value)),
                self.value);
            c := Copy(self._children[1]);
            self._evaluated := SubstBottomUp(c, @(1, var, e->Same(e,self.var)), e -> d);
            return self._evaluated;
        fi;
    end,

    rChildren := self >> [self.var, self.value, self.child(1)],
    rSetChild := meth(self, n, newC)
        if n=1 then self.var := newC;
        elif n=2 then self.value := newC;
        elif n=3 then self._children[1] := newC;
        else Error("<n> must be between [1..3]");
        fi;
    end,
    from_rChildren := (self, rch) >> CopyFields(self, rec(
        var := rch[1], value := rch[2], _children := [rch[3]])),
    #-----------------------------------------------------------------------
    uneval := meth(self) Unbind(self._evaluated); return self; end,
    #-----------------------------------------------------------------------
    transpose := self >> CopyFields(self, rec(
        _children := [self.child(1).transpose()])).uneval(),
    #-----------------------------------------------------------------------
    toAMat := self >> self.eval().toAMat(),
    #-----------------------------------------------------------------------
    sums := self >> CopyFields(self, rec(_children := [self.child(1).sums()])),
));

Declare(Scat);

#F ==========================================================================
#F Gath(<func>) - gather (read) matrix
#F NOTE: implements affine transformations via funcExp hack.

#F as of Dec '2010, <func> can contain fInsert/fPad, which will create
#F funcExp(..) in the code. This is used for simulating affine (rather
#F than linear) transformation.
#F
#F Affine transformations can only be experessed using matrices, if we
#F use homogeneous coordinates, i.e., instead of [x_1, ..., x_n] use
#F always [x_1, ..., x_n, 1], for input/output vectors
#F
#F Gath.toAMat and everything else does NOT use homogeneous
#F coordinates, and thus we can't represent affine "gathers" with a
#F proper matrix, the only special case is when funcExp(0) is used to
#F insert 0s (this preserves linearity).
#F
#F Currently, we use the following semantics (to implement affine transf. using a hack)
#F
#F  nth(X, i)          == X[i]
#F  nth(X, funcExp(i)) == i
#F
#F The proper way of doing this would be instead (using h. coords, X[len(x)] = 1)
#F nth(X, funcExp(i)) -> i * nth(X, len(X)) = i * X[len(X)] = i 
#F
#F See http://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations
# ==========================================================================
Class(Gath, SumsBase, BaseMat, rec(
    #-----------------------------------------------------------------------
    rChildren := self >> [self.func],
    rSetChild := rSetChildFields("func"),
    #-----------------------------------------------------------------------
    new := (self, func) >> SPL(WithBases(self, rec(
      	func := Checked(IsFunction(func) or IsFuncExp(func), func)))).setDims(),
    #-----------------------------------------------------------------------
    dims := self >> [self.func.domain(), self.func.range()],
    sums := self >> self,
    area := self >> Sum(Flat([self.func.domain()])),
    isReal := self >> true,
    transpose := self >> Scat(self.func),
    conjTranspose := self >> self.transpose(),
    inverse := self >> self.transpose(),
    #-----------------------------------------------------------------------
    toAMat := self >> let(
	n := EvalScalar(self.func.domain()),
        N := EvalScalar(self.func.range()),
        func := self.func.lambda(),
        AMatMat(List([0..n-1], row -> let(
            idx := EvalScalar(func.at(row)),
	    Cond(idx _is funcExp,
		     When(idx.args[1]=0, Replicate(N, 0), 
			 Error("<self> is an affine (non-linear) transformation ",
			       "and can't be represented as a matrix")),
		 BasisVec(N, idx)))))
    ),
    #-----------------------------------------------------------------------
    toloop := (self, bksize) >> let(
	i := Ind(self.func.domain()),
	ISum(i, 
            Scat(fTensor(fBase(i), fId(1))) *
            Gath(fCompose(self.func, fTensor(fBase(i), fId(1))))
	).split(bksize)
    ),
    #-----------------------------------------------------------------------
    normalizedArithCost := self >> 0,
    #-----------------------------------------------------------------------
    isIdentity := self >> IsIdentity(func),
));

#F ==========================================================================
#F Prm(<func>) - permutation, semantically same as Gath(<func>), but square
#F
#F NB: Prm should not be used with fPad/fInsert, which lead to affine 
#F     transformations when used with Gath.
#F
#F Prm(f).transpose() = Prm(f.transpose())
#F
#F ==========================================================================
Class(Prm, Gath, rec(
    transpose := self >> CopyFields(self, rec(func:=self.func.transpose())),
    toAMat := self >> Perm(PermList(List(self.func.lambda().tolist(), e->e.v)+1),
                           self.func.domain()).toAMat(),
    #-----------------------------------------------------------------------
    normalizedArithCost := self >> 0
));

#   special perm to be gotten rid of in rewriting
Class(DelayedPrm, Prm);
Class(FormatPrm, Prm);

#F ==========================================================================
#F Scat(<func>) - scatter (write) matrix,  Scat(f) = Gath(f).transpose()
#F
#F NOTE: implements affine transformations via funcExp workaround. 
#F        See Doc(Gath) for explanation
#F ==========================================================================
Class(Scat, SumsBase, BaseMat, rec(
    #-----------------------------------------------------------------------
    rChildren := self >> [self.func],
    rSetChild := rSetChildFields("func"),
    #-----------------------------------------------------------------------
    new := (self, func) >> SPL(WithBases(self, rec(
	func := Checked(IsFunction(func) or IsFuncExp(func), func)))).setDims(),
    #-----------------------------------------------------------------------
    dims := self >> [self.func.range(), self.func.domain()],
    sums := self >> self,
    area := self >> Sum(Flat([self.func.domain()])),
    isReal := self >> true,
    transpose := self >> Gath(self.func),
    conjTranspose := self >> self.transpose(),
    inverse := self >> self.transpose(),
    #-----------------------------------------------------------------------
    toAMat := self >> TransposedAMat(Gath(self.func).toAMat()),
    #-----------------------------------------------------------------------
    toloop := (self, bksize) >> Gath(self.func).toloop(bksize).transpose(),
    #-----------------------------------------------------------------------
    normalizedArithCost := self >> 0,
    #-----------------------------------------------------------------------
    isIdentity := self >> IsIdentity(self.func),
));

Class(ScatAcc, Scat, rec(
    codeletName:="SA", 
    toloop := (self, bksize) >> Error("Not implemented")
));

Declare(ScatGath);

# ==========================================================================
# ScatGath(<sfunc>, <gfunc>)
# ==========================================================================
Class(ScatGath, SumsBase, BaseMat, rec(
    rChildren := self >> [self.sfunc, self.gfunc],
    rSetChild := rSetChildFields("sfunc", "gfunc"),
    #-----------------------------------------------------------------------
    new := (self, sfunc, gfunc) >> SPL(WithBases(self,
        rec(dimensions := [sfunc.range(), gfunc.range()], 
	    sfunc := Checked(IsFunction(sfunc) or IsFuncExp(sfunc), sfunc),
	    gfunc := Checked(IsFunction(gfunc) or IsFuncExp(gfunc), gfunc)))),
    #-----------------------------------------------------------------------
    dims := self >> [self.sfunc.range(), self.gfunc.range()],
    area := self >> self.sfunc.domain(),
    isReal := self >> true,
    transpose := self >> ScatGath(self.gfunc, self.sfunc),
    conjTranspose := self >> self.transpose(),
    inverse := self >> self.transpose(),
    #-----------------------------------------------------------------------
    toAMat := meth(self)
        local s, gfunc, g, sdomain, gdomain, idx;
        # NOTE: FF: this is a temporary solution to work around Lamda 
	#        problems with symbolic domains and variable substitution
        s := Scat(self.sfunc);
        sdomain := EvalScalar(self.sfunc.domain());
        gdomain := spiral.code.RulesStrengthReduce(self.gfunc.domain());
        if ObjId(self.gfunc) = Lambda and IsExp(gdomain) then
            idx := Ind(sdomain);
            gfunc := Lambda(idx, self.gfunc.at(idx)).setRange(self.gfunc.range());
        else
            gfunc := self.gfunc;
        fi;
        g :=Gath (gfunc);
        return s.toAMat() * g.toAMat();
    end,
       # Correct semantics: Scat(self.sfunc).toAMat() * Gath(self.gfunc).toAMat(),
    #-----------------------------------------------------------------------
    sums := self >> self, #self.toloop(self.maxBkSize()),
    #-----------------------------------------------------------------------
    maxBkSize := meth(self)
        local exp, l, d;
        exp := self.gfunc.domain();

        if IsValue(exp) or IsInt(exp) then return exp; fi;

        if IsExp(exp) and ObjId(exp)=mul and IsValue(exp.args[1]) then
            return exp.args[1];
        fi;

        if IsInt(exp.eval()) or IsValue(exp.eval()) then return EvalScalar(exp); fi;
        l := Lambda(Filtered(exp.free(), IsLoopIndex), exp);
        if Length(l.vars) > 1 then return 1; fi;
        d := List(spiral.sigma.GenerateData(l).tolist(), EvalScalar);
        return Gcd(d);
    end,
    #-----------------------------------------------------------------------
    toloop := (self, bksize) >> let(
	i := Ind(self.gfunc.domain()),
        ISum(i, 
            Scat(fCompose(self.sfunc, fTensor(fBase(i), fId(1)))) *
            Gath(fCompose(self.gfunc, fTensor(fBase(i), fId(1))))
        ).split(bksize)
    )
));


# ==========================================================================
# SUM(<spl1>, <spl2>, ...) - non-overlapping matrix sum
# ==========================================================================
Declare(SUM, SUMAcc);
Class(SUM, SumsBase, BaseOperation, rec(
    area := self >> Sum(self.children(), x->x.area()),
    abbrevs := [ arg ->
    [ Flat(List(Flat(arg),
        s -> When(IsSPL(s) and Same(ObjId(s), SUM), s.children(), s))) ] ],
    #-----------------------------------------------------------------------
    new := meth(self, L)
        local dims;
        Constraint(Length(L) >= 1); Constraint(ForAll(L, IsSPL));
        if Length(L) = 1 then return L[1]; fi;
        dims := L[1].dims();
        if not (IsSymbolic(dims[1]) or IsSymbolic(dims[2])) and
           not ForAll(Drop(L, 1), x -> let(d:=x.dims(),
                   IsSymbolic(d[1]) or IsSymbolic(d[2]) or d = dims))
            then Error("Dimensions of summands do not match"); fi;
        return SPL(WithBases(self, rec( _children := L, dimensions := dims)));
    end,
    #-----------------------------------------------------------------------
    rng := self >> self.child(1).rng(),
    #-----------------------------------------------------------------------
    dmn := self >> self.child(1).dmn(),

    advdims := self >> self._children[1].advdims(),
    #-----------------------------------------------------------------------
#    dims := self >> self.child(1).dimensions,
    #-----------------------------------------------------------------------
    toAMat := self >> AMatMat(Sum(self._children, MatSPL)),
    #-----------------------------------------------------------------------
    isPermutation := self >> false,
    #-----------------------------------------------------------------------
    transpose := self >>   # we use CopyFields to copy all fields of self
        CopyFields(self, rec(
           _children := List(self._children, x->x.transpose()),
           dimensions := Reversed(self.dimensions))),
    inverse := self >>   # we use CopyFields to copy all fields of self
        CopyFields(self, rec(
           _children := List(self._children, x->x.inverse()),
           dimensions := Reversed(self.dimensions))),
    conjTranspose := self >>   # we use CopyFields to copy all fields of self
        CopyFields(self, rec(
           _children := List(self._children, x->x.conjTranspose()),
           dimensions := Reversed(self.dimensions)))
));

# ==========================================================================
# SUMAcc(<spl1>, <spl2>, ...) - overlapping matrix sum
# ==========================================================================
Class(SUMAcc, SUM, rec(
   abbrevs := [ arg ->
    [ Flat(List(Flat(arg),
        s -> When(IsSPL(s) and Same(ObjId(s), SUMAcc), s.children(), s))) ] ]
 ));

# ==========================================================================
# ISum(<var>, <domain>, <spl>) - non-overlapping iterative matrix sum
# ==========================================================================
Class(ISum, SumsBase, BaseIterative, rec(
    needInterleavedLeft := self >> self.child(1).needInterleavedLeft(),
    needInterleavedRight := self >> self.child(1).needInterleavedRight(),
    cannotChangeDataFormat := self >> self.child(1).cannotChangeDataFormat(),
    totallyCannotChangeDataFormat := self >> self.child(1).totallyCannotChangeDataFormat(),

    directOper := SUM,
    area := self >> let(ac:=self._children[1].area(), ac * self.domain),
    #-----------------------------------------------------------------------
    rng := self >> self._children[1].rng(),
    dmn := self >> self._children[1].dmn(),
    dims := self >> [StripList(List(self.rng(),l->l.size)),StripList(List(self.dmn(),l->l.size))],

    advdims := self >> self._children[1].advdims(),

    #-----------------------------------------------------------------------
    transpose := self >> CopyFields(self, rec(
           _children := [self._children[1].transpose()],
           dimensions := Reversed(self.dimensions))),
    conjTranspose := self >> CopyFields(self, rec(
           _children := [self._children[1].conjTranspose()],
           dimensions := Reversed(self.dimensions))),
    inverse := self >> CopyFields(self, rec(
           _children := [self._children[1].inverse()],
           dimensions := Reversed(self.dimensions))),
    #-----------------------------------------------------------------------
    unroll := self >> SUM(self.unrolledChildren()),
    #-----------------------------------------------------------------------
    sums := self >> CopyFields(self, rec(
        _children := [self._children[1].sums()]))
));

Class(ISumLS, ISum);
Class(JamISum, ISum, rec(isBlockTransitive := true));
Class(Grp, Buf);

# ==========================================================================
# ICompose(<var>, <domain>, <spl>) - iterative matrix product
# ==========================================================================
Class(ICompose, SumsBase, BaseIterative, rec(
    area := self >> self._children[1].area() * self.domain,
    #-----------------------------------------------------------------------
    dims := self >> self._children[1].dimensions,
    #-----------------------------------------------------------------------
    unroll := self >> Compose(self.unrolledChildren()),
    #-----------------------------------------------------------------------
    transpose := self >> ICompose(self.var, self.domain,
        SubstVars(Copy(self._children[1].transpose()), 
	          tab((self.var.id) := self.domain-1-self.var))),

    createCode := self >> Cond(IsBound(self._children[1].createCode),
        ICompose(self.var, self.domain, self._children[1].createCode()), self),

    prods := self >> let(base := self.__bases__[1],
        base(self.var, self.domain, self._children[1].prods())),

#    rChildren := self >> [self.var, self.domain, self._children[1]],
#
#    from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), rch),
#
#    rSetChild := meth(self, n, newChild)
#        if n=1 then self.var := newChild;
#        elif n=2 then self.domain := newChild;
#        elif n=3 then self._children := [newChild];
#        else Error("<n> must be in [1..3]");
#        fi;
#    end
));


# ==========================================================================
# ISumAcc(<var>, <domain>, <spl>) - overlapping iterative matrix sum
# ==========================================================================
Class(ISumAcc, ISum);

# ==========================================================================
# IParSeq(<var>, <domain>, <fb>, <spl>) 
# ==========================================================================
Declare(ParSeq);
Class(IParSeq, SumsBase, BaseIterative, rec(
    abbrevs := [ (v, fb_cnt, expr) -> [v, v.range, fb_cnt, expr]  ],
    new := meth(self, v, domain, fb_cnt, expr)
        local obj;
        #NOTE: check dimensions
        obj := Inherited(v, domain, expr);
        obj.fb_cnt := fb_cnt;
        return obj;
    end,

    dims := self >> self._children[1].dims(),
    unroll := self >> ParSeq( self.fb_cnt, Reversed(self.unrolledChildren())),

    filtCompL := (self, lst) >> lst{[1..self.fb_cnt]},
    filtCompR := (self, lst) >> lst{[1..self.fb_cnt]},
    filtSUML  := (self, lst) >> lst{[self.fb_cnt+1..Length(lst)]},
    filtSUMR  := (self, lst) >> lst{[self.fb_cnt+1..Length(lst)]},

    # area doesn take into account that we have composition and sum
    area := self >> self._children[1].area() * self.domain,

    print := (self, i, is) >> Print(
        self.name, "(", self.var, ", ", self.domain, ", ", self.fb_cnt, ",\n",
        Blanks(i+is), self._children[1].print(i+is, is), "\n",
        Blanks(i), ")", self.printA(),
        When(IsBound(self._setDims), Print(".overrideDims(", self._setDims, ")"), Print(""))
    ),
));

##############################################################################
Declare(Conj, ConjL, ConjR, ConjLR, ConjDiag);

Class(Conj, SumsBase, BaseOperation, rec(
    new    := (self, spl) >> SPL(WithBases(self, rec(_children:=[spl], dimensions := spl.dimensions))),
    dims   := self >> self._children[1].dims(),
    toAMat := self >> self.child(1).toAMat(),
    sums   := self >> self,
    transpose := self >> Conj(self.child(1).transpose())
));

Class(ConjL, Conj, rec(
    new    := (self, spl, lprm) >> SPL(WithBases(self, rec(_children:=[spl, lprm]))).setDims(),
    dims   := self >> [self.child(2).dims()[1], self.child(1).dims()[2]],
    toAMat := self >> self.child(2).toAMat() * self.child(1).toAMat(),
    sums   := self >> ConjL(self.child(1).sums(), self.child(2)),
    transpose := self >> ConjR(self.child(1).transpose(), self.child(2).transpose()),
));

Class(ConjR, Conj, rec(
    new    := (self, spl, rprm) >> SPL(WithBases(self, rec(_children:=[spl, rprm]))).setDims(),
    dims   := self >> [self._children[1].dims()[1], self._children[2].dims()[2]],
    toAMat := self >> self.child(1).toAMat() * self.child(2).toAMat(),
    sums   := self >> ConjR(self.child(1).sums(), self.child(2)),
    transpose := self >> ConjL(self.child(1).transpose(), self.child(2).transpose()),
));

Class(ConjLR, Conj, rec(
    new    := (self, spl, lprm, rprm) >> SPL(WithBases(self, rec(_children:=[spl, lprm, rprm]))).setDims(),
    dims   := self >> [ self.child(2).dims()[1], self.child(3).dims()[2] ],
    toAMat := self >> self.child(2).toAMat() * self.child(1).toAMat() * self.child(3).toAMat(),
    sums   := self >> ConjLR(self.child(1).sums(), self.child(2), self.child(3)),
    transpose := self >> ConjLR(self.child(1).transpose(), self.child(3).transpose(), self.child(2).transpose()),
));

Class(ConjDiag, Conj, rec(
    new    := (self, spl, lprm, rprm) >> SPL(WithBases(self, rec(_children:=[spl, lprm, rprm]))).setDims(),
    dims   := self >> [Rows(self.child(2)), Cols(self.child(3))],
    toAMat := self >> self.child(2).toAMat() * self.child(1).toAMat() * self.child(3).toAMat(),
    sums   := self >> ConjDiag(self.child(1).sums(), self.child(2), self.child(3)),
    transpose := self >> ConjDiag(self.child(1).transpose(), self.child(3).transpose(), self.child(2).transpose()),
));

Class(NeedInterleavedComplex, BaseContainer, rec(
    needInterleavedLeft := True,
    needInterleavedRight := True,
    sums := self >> self,
    area:= self >> self.child(1).area()
));
