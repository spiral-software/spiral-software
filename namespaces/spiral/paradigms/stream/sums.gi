
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(StreamGath);
Declare(StreamScat);
Declare(STensor);
Declare(SIterDirectSum);

Import(compiler);

Class(RootsOfUnity, Sym, rec(
    abbrevs := [ (n) -> [n]],

    def := (n) -> let(j := Ind(n),
	Diag(Lambda(j, omega(n, j)).setRange(Complexes))),

    tolist := self >> self.lambda().tolist(),
    lambda := self >> self.obj.element,
    sums := self >> Diag(self),
    isFunction := true,
    range := self >> TComplex,
    domain := self >> self.params[1],

    transpose := self >> self,

    isReal := self >> (self.params[1] <= 2),

    exportParams := self >> self.params

));

Class(streamDiagFunc, FuncClass, rec(
    def := (N, n, r) -> rec(
        N := N/2,
        n := N,
	r := r,
    ),


    tolist := self >> self.lambda().tolist(),
    lambda := self >> let(
	i := Ind(self.n),
	Lambda(i, imod(i, self.params[2]) * idiv(i, self.params[2]))),
   domain := self >> self.n,
   range := self >> self.N
));


# ==========================================================================
# StreamGath(<i>, <range>, <bs>) - vector gather (read) matrix 
# ==========================================================================
Class(StreamGath, BaseMat, SumsBase, rec(
    #-----------------------------------------------------------------------
#    rChildren := self >> [self.func],
#    rSetChild := rSetChildFields("func"),
    rChildren := self >> [self.i],
    rSetChild := rSetChildFields("i"),
    #-----------------------------------------------------------------------
    new := (self, i, range, bs) >> SPL(WithBases(self, 
        rec(dimensions := [bs, bs*range], bs := bs, i := i, range := range))),
    #-----------------------------------------------------------------------
    sums := self >> self,
    area := self >> self.bs * self.range,
    isReal := self >> true,
    transpose := self >> StreamScat(self.i, self.range, self.bs),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.name, "(", self.i, ", ", self.range,", ", self.bs,")"),
    #-----------------------------------------------------------------------
    toAMat := self >> Gath(fTensor(fBase(self.range, self.i), fId(self.bs))).toAMat()
    # make verilog
));


# ==========================================================================
# StreamScat(<func>, <v>) - vector scatter (write) matrix
# ==========================================================================
Class(StreamScat, BaseMat, SumsBase, rec(
    #-----------------------------------------------------------------------
    rChildren := self >> [self.i],
    rSetChild := rSetChildFields("i"),
    #-----------------------------------------------------------------------
    new := (self, i, range, bs) >> SPL(WithBases(self, 
        rec(dimensions := [bs*range, bs], bs := bs, i := i, range := range))),
    #-----------------------------------------------------------------------
    sums := self >> self,
    area := self >> self.func.domain() * self.v,
    isReal := self >> true,
    transpose := self >> StreamGath(self.i, self.range, self.bs),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.name, "(", self.i, ", ", self.range,", ", self.bs,")"),
    #-----------------------------------------------------------------------
    toAMat := self >> TransposedAMat(StreamGath(self.i, self.range, self.bs).toAMat())
#    code := (self, y, x) >> VTensor(Scat(self.func), self.v).code(y,x)    
));

RCExpandData := function(value)
   return ConcatList(value.v, x->[ReComplex(Complex(x.v)), ImComplex(Complex(x.v))]);
end;

#FixLocalize := (o) -> SubstBottomUpRules(o, [
#    [[Diag, @(1)], e -> Diag(GenerateData(ResolveMemos(e.element))), "FixLocalize" ]]);
   

Declare(BRAMPerm);

# ===============================================
# CodeBlock(<SPL>)
# ==============================================
Class(CodeBlock, BaseContainer, SumsBase, rec(
    isBlock:=true,	

    sums := self >> CopyFields(self, rec(_children := [self.child(1).sums()])),
	
    code := false,

	createCode := meth(self)  
	   local i, s, res, d, dl;
	   if (IsBound(self._children[1].isBlock)) then
	     res := self._children[1].createCode();
	   else 
	     s := self.child(1); 
	     res := CopyFields(self, rec(code := CodeSums(s, StreamDefaults)));
	   fi;	    	   
       res.code := BinSplit(res.code);
	   return res;
	end,

    rChildren := self >> [self._children[1], self.code],

    rSetChild := meth(self, n, what)
        if n=1 then
            self._children[1] := what;
        elif n=2 then
            self.code := what;
        else Error("<n> must be in [1,2]");
        fi;
    end,

    from_rChildren := (self, rch) >> CopyFields(ObjId(self)(rch[1]), rec(code:=rch[2])),

    print := (self,i,is) >> Print(self.name, "(", Cond(self.code<>false, self.code.print(i,is), self._children[1]), ")")
));

Class(StreamSum, ISum, BaseIterative, SumsBase);

# A streaming I tensor J primitive
Class(StreamIxJ, BaseMat, SumsBase, rec(
    new := (self, k, p, bs) >> SPL(WithBases(self, rec(dimensions:=[k*p,k*p], element:=k, p:=p, bs:=bs))),
    print := (self,i,is) >> Print(self.name, "(", self.element, ", ", self.p, ", ", self.bs, ")"),
    toAMat := self >> let(J := When(self.element = 2, I(2), Compose(Tensor(I(2), L(self.element/2, self.element/4)), L(self.element, 2))), Tensor(I(self.p), J).toAMat()),
    transpose := self >> self,
    area := self >> self.element,
    sums := self >> self,
    rChildren := self >> [],
    dimensions := self >> [self.element * self.p, self.element * self.p],
    dims := self >> self.dimensions,
   isReal := True
));




# parameters:
#   L[1]: SPL
#   L[2]: p: p in I_p x SPL
#   L[3]: bs: stream buffer size
Class(STensor, Tensor, 
  rec(
    new := (self, L) >> SPL(WithBases(self, rec(
            bs := L[3],
            dimensions := L[1].dimensions * Cond(IsValue(L[2]), L[2].ev(), L[2]), 
            p := Cond(L[1].dims()[1] >= L[3], L[2], L[2]*L[1].dims()[1]/L[3]),
            _children := [Checked(L[1].dims()[1] = L[1].dims()[2], 
                          Cond(L[1].dims()[1] >= L[3], L[1], Tensor(I(L[3] / L[1].dims()[1]), L[1]))), Cond(L[1].dims()[1] >= L[3], L[2], L[2]*L[1].dims()[1]/L[3]), L[3]]
#            _children := [Checked(L[1].dims()[1] = L[1].dims()[2], 
#                          Cond(L[1].dims()[1] >= L[3], L[1], Tensor(I(L[3] / L[1].dims()[1]), L[1])))]
                          ))),

    print := (self,i,is) >> Print(self.name, "(", self.child(1), ", ", self.p, ", ", self.bs, ")"),
    createCode := self >> Cond(IsBound(self.child(1).createCode), STensor(self.child(1).createCode(), self.p, self.bs), self),
    isPermutation := False,
    toAMat := self >> Tensor(I(self.p), self.child(1)).toAMat(),
    sums := self >> self,
    dims := self >> Cond(IsValue(self.p), self.p.ev(), self.p) * self.child(1).dims()
));


# parameters: I_m x L^n_r x I_p
#   L[1]: n
#   L[2]: r
#   L[3]: p 
#   L[4]: m 
#   L[5]: stream buffer size
Class(SIxLxI, Tensor, 
  rec(
    new := (self, l) >> SPL(WithBases(self, rec(
            _children := [],
            dimensions := [l[1] * l[3] * l[4], l[1] * l[3] * l[4]],
            bs := l[5],
            p := l[3],
            m := l[4],

    createCode := self >> self,
    print := (self,i,is) >> Print(self.name, "(", l[1], ", ", l[2], ", ", self.p, ", ", self.m, ", ", self.bs, ")"),
    isPermutation := False,
    toAMat := self >> Tensor(Tensor(I(self.m), L(l[1], l[2]), I(self.p))).toAMat(),
    transpose := self >> self,
    dims := self >> self.dimensions,
#    dimensions := l[1] * l[3] * l[4],
    sums := self >> self,
)))));


# ==========================================================================
# Streaming iterative direct sum
# SIterDirectSum(<var>, <range>, <spl>, <stream buffer size>)
#    Size of the <spl> must not depend on <var>. 
# ==========================================================================
Class(SIterDirectSum, BaseIterative, rec(
    abbrevs := [ (v,d,e, bs) -> [v,d,e,bs],
	         (v,e,bs) -> [v, v.range, e, bs]],
    new := meth(self, var, domain, expr, bs) 
        local res;
        var.isLoopIndex := true;
        res := SPL(WithBases(self, rec(
#		_children := [Checked(expr.dims()[1] <= bs and 
		_children := [Checked(expr.dims()[1] = expr.dims()[2],
				      Cond(expr.dims()[1] >= bs,
					  expr,
					  IterDirectSum(Ind(bs/expr.dims()[1]), expr)))],		
		var := var,
		bs := bs,
		dimensions := expr.dimensions * domain,
		domain := domain)));
        return res;
        end,


    rChildren := self >> self._children,
    rSetChild := meth(self, n, what)  self._children[n] := what; end,
    child := (self, n) >> self._children[n],
    children := self >> self._children,

#         Constraint(IsSPL(expr));
# 	Constraint(not IsList(domain));
# 	#domain := toRange(domain);
# 	if domain = 1 then 
# 	    return SubstBottomUp(expr, @.cond(e->Same(e,var)), e->V(0));
# 	fi;
# 	var.range := domain;
#         return SPL(WithBases( self, 
# 	    rec( expr := expr,
# 		 var := var,
# 		 domain := domain,
# 		 dimensions := domain * Dimensions(expr) )));
#     end,

    print := (self, i, is) >> Print(self.name, "(", self.var, ", ",  self.domain, ", ", self.child(1).print(i+is, is), ", ", self.bs, ")"),

    toAMat := self >> self.unroll().toAMat(),

    unroll := self >> DirectSum(
	List([ 0 .. self.domain - 1 ], 
	    index_value -> SubstBottomUp(
		Copy(self._children[1]), self.var, 
		e -> V(index_value)))),

    dims := self >> self.dimensions, 
    #-----------------------------------------------------------------------
    isPermutation := False,
    #-----------------------------------------------------------------------
    createCode := self >> Cond(IsBound(self._children[1].createCode), 
	SIterDirectSum(self.var, self.domain, self._children[1].createCode(), self.bs), self),

    transpose :=   # we use inherit to copy all fields of self
        self >> Inherit(self, rec(
		   expr := TransposedSPL(self._children[1]), 
		   dimensions := Reversed(self.dimensions))),

    sums := self >> self
));

NoPull.createCode := meth ( self )
    local i, s, res, d, dl;
    if IsBound(self._children[1].isBlock)  then
        res := self._children[1].createCode();
    else
        s := self.child(1);
        res := SPL(CopyFields(self, rec(
                code := CodeSums(self.dims()[1] * 2, s) )));
    fi;
    for i  in [ 1 .. 10 ]  do
        BinSplit(res.code);
    od;
    return res;
end;


Class(Omega, BaseIterative, rec(
	
    abbrevs := [ (N,n,i) ->[N, n, i] ],



    new := (self, N, n, i) >> let(j := Ind(n),
        Lambda(j, omega(N, i*j)).setRange(TComplex)),


    rChildren := self >> self._children,
    rSetChild := meth(self, n, what)  self._children[n] := what; end,
    child := (self, n) >> self._children[1],
    children := self >> self._children,
    
    

    tolist := self >> self.lambda().tolist(),
    lambda := self >> self.obj.element,
    sums := self >> Diag(self),
    isFunction := true,
    range := self >> TComplex,
    domain := self >> self.params[1],

    transpose := self >> self,

#    isReal := self >> (self.params[1] = self.params[2]  or self.params[2] = 1),

    exportParams := self >> self.params
));

Prm.createCode := (self) >> SPL(CopyFields(self, rec(code := CodeSums(self, StreamDefaults))));
ISum.createCode := (self) >> CodeBlock(self).createCode();

Tensor.createCode := self >>
   Tensor(
      List([1..Length(self.children())], i->
        Cond(IsBound(self.child(i).createCode),
	    self.child(i).createCode(),
	    self.child(i)
         )
      )
   );
      
Tw1Stride := (n,d,k,which,stride) -> Checked(IsPosIntSym(n), IsPosIntSym(d), IsPosIntSym(k),
    fCompose(dOmega(n,k), diagTensor(dLin(n/d, 1, 0, TInt), dLin(d/stride, 1, which, TInt))));

# fReal := (f) -> fCompose(RCData(f), fTensor(fId(f.n), fBase(2, 0)));
Class(fReal, FuncClass, rec(
   def := (f) -> rec(N := f.range(), n := f.domain(), f := f),
   tolist := self >> self.lambda().tolist(),
   lambda := self >> fCompose(RCData(self.f), fTensor(fId(self.n), fBase(2,0))).lambda(),
   domain := self >> self.n,
   range := self >> self.N
));

# fImag := (f) -> fCompose(RCData(f), fTensor(fId(f.n), fBase(2, 1)));
Class(fImag, FuncClass, rec(
   def := (f) -> rec(N := f.range(), n := f.domain(), f := f),
   tolist := self >> self.lambda().tolist(),
   lambda := self >> fCompose(RCData(self.f), fTensor(fId(self.n), fBase(2,1))).lambda(),
   domain := self >> self.n,
   range := self >> self.N
));

Class(Stream1DFT2, BaseMat, SumsBase, rec(
    new := (self) >> SPL(WithBases(self, rec(dimensions:=[2,2]))),
    print := (self,i,is) >> Print(self.name),
    toAMat := self >> DFT(2,1).toAMat(),
    transpose := self >> self,
    sums := self >> self,
    rChildren := self >> [],
    dimensions := [2,2],
    dims := self >> self.dimensions,
));


Class(StreamPadDiscard, BaseMat, SumsBase, rec(
    abbrevs := [(n, m, w_in, w_out) -> [n, m, w_in, w_out]],

    new := (self, n, m, w_in, w_out) >> SPL(WithBases(self, rec(
          _children := [n,m, w_in, w_out],
          dimensions := [n, m],
    ))),

    rChildren := self >> self._children,
	rSetChild := meth(self, n, what) self._children[n] := what; end,
	child     := (self, n) >> self._children[n],
	children  := self >> self._children,

    toAMat := self >> RI(self.child(1), self.child(2)).toAMat(),
    dims := self >> [self.child(1), self.child(2)],

));