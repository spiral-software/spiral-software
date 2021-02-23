
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# -------------------------------------------------------------------------------
# NOTE: Make at()/lambda() methods consistent
#
#  Either implement at via lambda consistently or the other way around
#  Currently some classes implement at() via lambda(), and some lambda() via at()
#  This is error prone!
# -------------------------------------------------------------------------------

#################################################################################
#F VData( <diagfunc>, <vlen> ) - new diagonal gen. function returning vectors
#F
#F Given a diagonal generating function F with domain N, and range (=element
#F type) T, VData(F, v) is a new "vector" diagonal function with domain N/v,
#F and range (=element type) T.vectorType(v).
#F
#F It uses vpack(_, _, ...) pseudo-instruction to pack scalars into vectors.
#F This instrucion has to be converted into architecture specific sequence
#F by the backend.
#F
Class(VData, Function, rec(
   __call__ := (self, func, v) >> WithBases(self, rec(
       func := Checked(IsFunction(func) or IsFuncExp(func), func),
       v := v,
       operations := PrintOps)),

   print := self >> Print(self.name, "(", self.func, ", ", self.v, ")"),
   rChildren := self >> [ self.func, self.v ],
   rSetChild := rSetChildFields("func", "v"),

   at := (self, i) >> let(
       n := self.func.domain(), v := self.v, f := self.func.lambda(),
       ApplyFunc(vpack, List([0..v-1], j->f.at(v*i + j)))),

   lambda := self >> let(
       n := self.func.domain(), v := self.v, f := self.func.lambda(),
       i := Ind(n/v),
       Lambda(i, ApplyFunc(vpack, List([0..v-1], j->f.at(v*i + j))))),

   domain := self >> div(self.func.domain(), self.v),
   range := self >> TVect(self.func.range(), self.v),
   free := self >> self.func.free()
));

#################################################################################
#F VDup( <diagfunc>, <vlen> ) - new diagonal gen. function returning vectors
#F
#F
#F Given a diagonal generating function F with domain N, and range (=element
#F type) T, VData(F, v) is a new "vector" diagonal function with domain N,
#F and range (=element type) T.vectorType(v). Every vector element is the
#F replicated original scalar.
#F
#F It uses vdup(<elt>, <vlen>) pseudo-instruction to duplicate scalars
#F to form vectors.  This instrucion has to be converted into
#F architecture specific sequence by the backend.
#F
Class(VDup, Function, rec(
   __call__ := (self, func, v) >> WithBases(self, rec(
       func := Checked(IsFunction(func) or IsFuncExp(func), func),
       v := v,
       operations := PrintOps)),

   print := self >> Print(self.name, "(", self.func, ", ", self.v, ")"),
   rChildren := self >> [ self.func, self.v ],
   rSetChild := rSetChildFields("func", "v"),

   at := (self, i) >> vdup(self.func.at(i), self.v),

   lambda := self >> let(
       i := Ind(self.func.domain()),
       Lambda(i, vdup(self.func.at(i), self.v))),

   domain := self >> self.func.domain(),
   range := self >> TVect(self.func.range(), self.v),
   free := self >> self.func.free()
));


#################################################################################
#F VDupOnline( <diagfunc>, <vlen> ) - is an online version of VDup.
#F
#F VDupOnline semantically the same as VDup(), but unlike VDup() it is pulled out
#F of fPrecompute and executed at compute time. 
#F 

Class(VDupOnline, VDup, rec(
    range := (self) >> let( t := self.func.range(), 
        Cond( IsVecT(t), TVect(t.t, t.size*self.v), TVect(t, self.v)))
));

#################################################################################
#F RCVData( <cplx_vdiagfunc> ) - func returning real vecs given complex vec func
#F
#F Given a diagonal generating function F with domain N, and range of
#F of complex vectors TVect(CT, v), RCVData(F) is a new real vector diagonal
#F function with domain N and range TVect(CT.realType(), 2*v) 
#F
Class(RCVData, Function, rec(
   __call__ := (self, func) >> WithBases(self, rec(
       func := Checked((IsFunction(func) or IsFuncExp(func)) and IsVecT(func.range()), func),
       operations := PrintOps)),

   print := self >> Print(self.name, "(", self.func, ")"),
   rChildren := self >> [ self.func ],
   rSetChild := rSetChildFields("func"),

   at := (self, i) >> let(
       t := self.func.range(), 
       f := self.func.lambda(),
       Cond(IsVecT(t), 
	    ApplyFunc(vpack, ConcatList([0..t.size-1], j -> [ re(velem(f.at(i), j)), im(velem(f.at(i), j)) ])),
	    ApplyFunc(vpack, [ re(f.at(i)), im(f.at(i)) ]))
   ),

   lambda := self >> let(j := Ind(self.domain()), Lambda(j, self.at(j))), 
   domain := self >> self.func.domain(), 
   range := self >> let(t := self.func.range(), 
       Cond(IsVecT(t), TVect(t.t.realType(), 2*t.size), TVect(t.realType(), 2))),
   free := self >> self.func.free()
));

#################################################################################
#F RCVDataSplit(<cplx_func>, <v>)
#F
#F Given a complex function f RCVDataSplit(f, v)
#F is a new function that returns v-way vectors of the form
#F [ re0, re0, re1, re1, ... ], [-im0, im0, -im1, im1, ...]
#F
Class(RCVDataSplit, Function, rec(
   __call__ := (self, cplx_func, v) >> WithBases(self, rec(
       func := Checked(IsFunction(cplx_func) or IsFuncExp(cplx_func), cplx_func),
       v := v,
       operations := PrintOps)),

   print := self >> Print(self.name, "(", self.func, ", ", self.v, ")"),
   rChildren := self >> [ self.func, self.v ],
   rSetChild := rSetChildFields("func", "v"),

   lambda := self >> let(
       i := Ind(self.domain()),
       n := self.func.domain(), 
       v := self.v, 
       f := self.func.lambda(),
       vars := DropLast(f.mkVars(), 1),
       ip := idiv(i, 2),
       Lambda(vars :: [i], 
       cond(imod(i, 2), 
            ApplyFunc(vpack, ConcatList([0..v/2-1], j ->
	        [ -im(f.at(vars :: [(v/2)*ip+j])), im(f.at(vars :: [(v/2)*ip+j])) ])),
            ApplyFunc(vpack, ConcatList([0..v/2-1], j -> 
		[  re(f.at(vars :: [(v/2)*ip+j])), re(f.at(vars :: [(v/2)*ip+j])) ]))))
   ),

   at := (self, i) >> self.lambda().at(i), 

   # each cpx element -> 2 duplicated reals (r r -i i) = 4 reals -> packed into vectors of length <v>
   domain := self >> let(d := self.func.domain(), 
       Cond(IsType(d), d, 4 * self.func.domain() / self.v)), 

   range := self >> let(t := self.func.range(), TVect(t.realType(), self.v)),
   free := self >> self.func.free()
));

#################################################################################
#F RCVDataSplitVec( <cplx_vdiagfunc> ) 
#F
#F Given a v/2-way complex vector function f RCVDataSplitVec(f, v)
#F is a new function that returns v-way real vectors of the form
#F [ re0, re0, re1, re1, ... ], [-im0, im0, -im1, im1, ...]
#F
Class(RCVDataSplitVec, RCVData, rec(
   # lambda := <inherited from RCVData, implemented via at>
   #
   at := (self, i) >> let(
       v := self.func.range().size, 
       f := self.func.lambda(),
       ip := idiv(i, 2),
       cond(imod(i, 2), 
            ApplyFunc(vpack, ConcatList([0..v-1], j -> 
		        [ -im(velem(f.at(ip), j)), im(velem(f.at(ip), j)) ])),
            ApplyFunc(vpack, ConcatList([0..v-1], j ->
		        [  re(velem(f.at(ip), j)), re(velem(f.at(ip), j)) ])))
   ),

   # each cpx element -> 2 duplicated reals (r r -i i) = 4 reals
   #  -> packed into vectors of length <v>
   domain := self >> let(d:=self.func.domain(), 
       Cond(IsType(d), d, 2 * self.func.domain())), 
));


#################################################################################
#F VDataRDup(<real-vector-func>)
#F
#F Returns new function of twice the vector length, with every element
#F duplicated.
#F
#F
Class(VDataRDup, RCVData, rec(
   # lambda := <inherited from RCVData, implemented via at>
   #
   at := (self, i) >> let(
       v := self.func.range().size, 
       f := self.func.lambda(),
       ApplyFunc(vpack, ConcatList([0..v-1],
	           j -> [ velem(f.at(i), j), velem(f.at(i), j) ]))
   ),

   # domain doesn't change, but output vectors get longer (ie range is different)
   domain := self >> self.func.domain(),
));

#################################################################################
#F fStretch( <diagfunc>, num, den ) - new diagonal gen. function stretching the old one
#F
Class(fStretch, Function, rec(
   __call__ := (self, func, num, den) >> WithBases(self, rec(
       func := Checked(IsFunction(func) or IsFuncExp(func), func),
       den := den,
       num := num,
       operations := PrintOps)),

   print := self >> Print(self.__name__, "(", self.func, ", ", self.num, ", ", self.den, ")"),
   rChildren := self >> [ self.func, self.num, self.den ],
   rSetChild := rSetChildFields("func", "num", "den"),

   at := (self, i) >> self.lambda().at(i).eval(),

   lambda := self >> let(
       n := self.func.domain(), 
       f := self.func.lambda(),
       i := Ind(n * self.num / self.den),
       i_old := self.den * idiv(i, self.num) + imod(i, self.num),
       Lambda(i, cond(leq(imod(i, self.num), self.den - 1), f.at(i_old), 0))
   ),

   domain := self >> self.func.domain() * self.num/self.den,
   range := self >> self.func.range(),
   free := self >> self.func.free()
));

#F ==========================================================================
#F VDiag(<func>, <v>)
#F
#F Vectorized diagonal. <func> must be a plain scalar function.
#F
#F Diag(f) == VDiag(f, v)
#F
Class(VDiag, Diag, rec(
    abbrevs := [ 
	(f, v) -> Checked(IsInt(v), 
	    Cond(IsList(f), let(ff := List(f, toExpArg),
		    [ FList(UnifyTypes(List(f, x -> x.t)), f), v]),
		 IsFunction(f), 
		    [ f, v ]))
    ],

    new := (self, f, v) >> SPL(WithBases(self, rec(element := f, v := v))).setDims(), 

    print := (self, indent, indentStep) >> Print(self.name, "(", self.element, ", ", self.v, ")"),
    needInterleavedLeft := False,
    needInterleavedRight := False,
    transpose := self >> self,
    conjTranspose := self >> ObjId(self)(FConj(self.element), self.v),
    # self.v is not exposed in rChildren
    from_rChildren := (self, rch) >> ObjId(self)(rch[1], self.v),
));

#################################################################################
#F VDiag_x_I(<func>, <v>)
#F
#F VTensor(Diag(f), v) == VDiag_x_I(f, v)
#F
Class(VDiag_x_I, VDiag, rec(
    toAMat := self >> Tensor(Diag(self.element), I(self.v)).toAMat(),
    dims := self >> self.v * Diag(self.element).dims(),
    needInterleavedLeft := False,
    needInterleavedRight := False,
    transpose := self >> self
));

#################################################################################
#F VRCDiag(<func>, <v>)
#F
#F Semantically equivalent to (I tensor L) * RCDiag * (I tensor L), and vectorized.
#F
#F Expects <func> and Input/output data to be in vector-interleaved complex format.
#F I.e. that packing is r0,r1,r2,...,r_(v-1); i0,i1,..,i_(v-1); ... into vectors.
#F
#F Example, given a complex function <f> (range = TComplex), and vector length <v>:
#F (different SIMD slots perform subsequent complex mults)
#F
#F    VRCDiag(VData(fCompose(RCData(f.domain()), fTensor(fId(d.domain()/v), L(2*v, 2))), v), v)
#F
#F Example, given a complex function <f> (range = TComplex), and vector length <v>:
#F (different SIMD slots perform SAME complex mults)
#F
#F    VRCDiag(VDup(RCData(f), v), v)
#F
Declare(VRCDiag);
Class(VRCDiag, VDiag, rec(
    print := (self, indent, indentStep) >> Print(self.name, "(", self.element, ", ", self.v, ")"),

    toAMat := meth(self)
        local elts, k, res, r, i;
        elts := self.element.tolist();
        Constraint(Length(elts) mod 2 = 0);
        res := [];
        for k in [ 1 .. Length(elts)/2 ] do
            r := EvalScalar(elts[2 * k - 1]);
            i := EvalScalar(elts[2 * k]);
            Add(res, BlockMat([[Diag(r), Diag(-i)], [Diag(i), Diag(r)]]).toAMat());
        od;
        return DirectSumAMat(res);
    end,

    toloop := self >> let(
        func := self.element, # r,i,r,i,...
        j    := Ind(idiv(func.domain(),2)),
        re   := func.at(2*j),
        im   := func.at(2*j+1),
        ISum(j, j.range, 
                VScat(H(2*j.range, 2, 2*j, 1), self.v) *
                VBlk([[re, -im],
                      [im,  re]], self.v) *
                VGath(H(2*j.range, 2, 2*j, 1), self.v))),

    dims := self >> let(n:=self.element.domain() * self.v, [n,n]),  # data stores r,i pairs
    needInterleavedLeft := False,
    needInterleavedRight := False,
    transpose := self >> VRCDiag(FRConj(self.element), self.v)
));

#################################################################################
#F RCVDiag(<func>, <v>)
#F
#F Semantically equivalent to RCDiag, but vectorized.
#F Expects <func> to pack r0,i0,r1,i1,.. into vectors. Input/Output data format is same.
#F
#F Example, given a complex function <f> (range = TComplex), and vector length <v>:
#F (different SIMD slots perform subsequent complex mults)
#F
#F    RCDiag(RCData(f)) == RCVDiag(VData(RCData(f), v), v)
#F
#F Example, given a complex function <f> (range = TComplex), and vector length <v>:
#F (different SIMD slots perform SAME complex mults)
#F
#F    RCDiag(RCData(f)) == RCVDiag(RCVData(VDup(f, v/2)), v)
#F
#F Example, given a complex vector function <f> (range = TVect(TComplex, v/2))
#F
#F    RCVDiag(RCVData(f), v),
#F
Class(RCVDiag, VDiag, rec(
    print := (self, indent, indentStep) >> Print(self.name, "(", self.element, ", ", self.v, ")"),

    toAMat := self >> RCDiagonalAMat(
	ConcatList(self.element.tolist(), x -> Cond(IsValue(x), x.v, List([1..self.v], i->velem(x,i)))), 
	IdentityPermAMat(2)),

    dims := self >> let(n:=self.element.domain() * self.v, [n,n]),  # data stores r,i pairs
    needInterleavedLeft := True,
    needInterleavedRight := True,

    transpose := self >> self  # this is invalid! need to conjugate complex function
));

#################################################################################
#F RCVDiagSplit(<func>, <v>)
#F
#F Semantically equivalent to RCDiag, but vectorized, and expects different <func>
#F data format than RCVDiag. 
#F
#F Expects <func> to pack r0,r0,r1,r1,...; -i0,i0,-i1,i1,...; ... into vectors.
#F Expects Input/Output data to pack r0,i0,r1,i1,.. into vectors.
#F
#F Example, given a complex function <f> (range = TComplex), and vector length <v/2>:
#F (different SIMD slots perform SAME complex mults)
#F
#F  RCVDiagSplit(RCVDataSplitVec(VDup(f, v/2)), v); 
#F
#F Example, given a complex function <f> (range = TComplex), and vector length <v/2>:
#F (different SIMD slots perform different complex mults)
#F
#F  RCVDiagSplit(RCVDataSplit(f, v), v);
#F
Class(RCVDiagSplit, VDiag, rec(
    print := (self, indent, indentStep) >> Print(self.name, "(", self.element, ", ", self.v, ")"),

    toAMat := meth(self)
        local elts, k, res, r, i, p;
        elts := self.element.tolist();
        Constraint(Length(elts) mod 2 = 0);
        res := [];
        p := Tensor(I(self.v/2), J(2));
        for k in [ 1 .. Length(elts)/2 ] do
            r := EvalScalar(elts[2 * k - 1]);
            i := EvalScalar(elts[2 * k]);
            Add(res, SUM(Diag(r), Diag(i)*p).toAMat());
        od;
        return DirectSumAMat(res);
    end,

    dims := self >> let(n:=self.element.domain() * (self.v/2), [n,n]),  # data stores r,i pairs
    needInterleavedLeft := True,
    needInterleavedRight := True,
    transpose := self >> self   # this is invalid! need to conjugate complex function
));

#################################################################################
#F VScale(<spl>, <constant>, <v>)
#F
#F Vecorized version of Scale
#F 
Declare(VScale);
Class(VScale, BaseMat, SumsBase, rec(
    new := (self, spl, s, v) >> SPL(WithBases(self,
        rec(dimensions := spl.dims(), _children:=[spl], s:=s, v := v))),
    #-----------------------------------------------------------------------
    dims := self >> self.dimensions,
    child := (self, i) >> self._children[i],
    rChildren := self >> [self._children[1], self.s, self.v],
    rSetChild := meth(self, n, what)
        if n=1 then self._children[1] := what;
        elif n=2 then self.s := what;
        elif n=3 then self.v := what;
        else Error("<n> must be in [1..3]");
        fi;
    end,
    #-----------------------------------------------------------------------
    print := (self, i, is) >> Print(self.name, "(", self.child(1).print(i+is,is), ", ", self.s, ", ", self.v, ")"),
    unroll := self >> self,
    transpose := self >> VScale(self.child(1).transpose(), self.s, self.v),
    isReal := self >> self.child(1).isReal(),
    #-----------------------------------------------------------------------
    toAMat := self >> AMatMat(MatSPL(Scale(self.s, self.child(1))))
));


#################################################################################
# Below are the version of above objects hacked up for online twiddle computation
# The difference is in the codegen, as it uses a different style of underlying
# function, that is recursive.
#
# NOTE: Find a cleaner and more general solution
# NOTE: document this
# NOTE: .toAMat() does not work for the below object, because self.func is different
#
Class(RCVOnlineDiag, RCVDiag);
Class(VOnlineDiag, VDiag);
Class(VOnlineDiag_x_I, VDiag_x_I);
