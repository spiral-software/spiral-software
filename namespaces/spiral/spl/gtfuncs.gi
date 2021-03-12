
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Here I define the new layer of "index-free" Sigma-SPL.  I will use
# this for library generation. The goal of index-free formulas is to
# eliminate loop indices completely. In such formulas objects may
# still depend on the loop indices, because gather/scatter/diagonal
# functions become multivariate (vs. univariate) functions, taking
# loop indices + point index as arguments. 
#
# See "Generalized Problem Spec" write-up.
#

Class(GTIndexFunction, FuncClass, rec(
    isGTIndexFunction := true,

    # .upRank (=lift in spl.maude) is is the inverse of reduction .downRank, 
    # it adds implicit dependency on the (new) inner loop, this happens when, eg.
    # gather is pulled into a loop:  Gath(f) ISum(s) => ISum(Gath(f.upRank()) * s)
    upRank := self >> Error("Not implemented"),

    upRankBy := (self, n) >> Checked(IsPosInt0(n), 
        FoldL([1..n], (f, i) -> f.upRank(), self)),

    # .downRank (=red in spl.maude) is the "reduction" operator for
    # index mapping functions it eliminates implicit deendencies on loops 
    # by introducing explicit references to loop variables ind(..)
    #
    # Eliminates dependency on loop <loopid>, 1 denotes innermost loop
    # <ind> is a loop index
    downRank := (self, loopid, ind) >> Error("Not implemented"),

    # returns the rank, i.e., the number of implicit nested loops
    rank := self >> Error("Not implemented"),

    # eliminated all loop dependencies 
    downRankFull := (self, inds) >> 
        FoldL(Reversed([1..Length(inds)]), (f, i) -> f.downRank(i, inds[i]), self),

    # split a loop, increase rank by 1, 
    # if a loop had n iterations, the new outer loop will have <outer_its> iterations, 
    # and next inner <inner_its> iterations
    split := (self, loopid, inner_its, outer_its) >> Error("not implemented"),

    # rotate the ranks (see Doc(frotate) for explanation)
    rotate := (self, n) >> Error("not implemented"),

    isUnitStride := self >> false,
    vecStride := (self, rank) >> self.params[4][rank+1],

));

Class(GTOps, rec(
    Print := PrintOps.Print,
    \= := (a,b) -> When(IsBound(a.equals), a.equals(b), b=a)
));

IsGTIndexFunction := o -> IsRec(o) and IsBound(o.isGTIndexFunction) and o.isGTIndexFunction;

# slow implementation, Haskell would love this
_dropTail := (lst, arg) -> Cond(Length(lst)=0,lst,Cond(Last(lst)=arg, _dropTail(DropLast(lst, 1),arg), lst));

Class(HHBase, GTIndexFunction, rec(
    abbrevs := [ (N,n,b,strides) -> [toExpArg(N), toExpArg(n), toExpArg(b), 
                                     List(_dropTail(strides, 0), toExpArg)] ],
    def := (N, n, b, strides) -> Checked(IsList(strides), ForAll(strides, IsPosInt0Sym), 
	rec(N := N, n := n)), 

    nloops := self >> Length(self.params[4]),
    free := self >> FreeVars(self.params),

    toSpl := (self, inds, kernel_size) >> self.downRankFull(inds),
    rank := self >> Maximum(0, Length(self.params[4])-1),  # rank = number of implicit loop dependencies

    domain := self >> When(IsValue(self.params[2]), self.params[2].v, self.params[2]),
    range := self >> When(IsValue(self.params[1]), self.params[1].v, self.params[1]),

    base := self >> self.params[3],
    strides := self >> Cond(self.params[4] = [], [0], self.params[4]),

    # rotate(<n>)
    #   switch the order of loops (represented by ranks), by making <n>-th loop innermost
    rotate := meth(self, n)
       local params, strides, rank;
       rank := self.rank();
       if rank<=1 and n<=1 then return self; 
       elif n > rank then return self.upRank(); 
       fi;
                 
       params := ShallowCopy(self.params);
       strides := ShallowCopy(params[4]);
       strides := [strides[1], strides[n+1]] :: strides{[2..n]} :: strides{[n+2 .. rank+1]};
       params[4] := strides;
       return self.from_rChildren(params);
    end,

    isUnitStride := self >> false, # redefine if unit stride is possible
));

Declare(KH, BHH, HH, HHZ);

#F HH(<N>, <n>, <b>, <strides>) - generalized stride index mapping function
#F   N - range
#F   n - domain
#F
#F   HH represents a multivariate function (j1 .. jn are indices of enclosing loops)
#F       (i, j1, ..., jn) -> b + strides[1]*i + strides[2]*j1 + ... + strides[n+1]*jn
#F
#F   Observe that dependence on the loop indices exists, even though no explicit
#F   loop index is present. This is how index free representation works.
#F
Class(HH, HHBase, rec(
    upRank := self >> let(s:=self.strides(), 
	HH(self.params[1], self.params[2], self.params[3], Concatenation([s[1], 0], Drop(s, 1)))),

    downRank := (self, loopid, ind) >> Cond(loopid > self.rank(), self, let(s:=self.strides(),
	HH(self.params[1], self.params[2], s[loopid+1] * ind + self.params[3], ListWithout(s, loopid+1)))),
    
    split := (self, loopid, inner_its, outer_its) >> Cond(loopid > self.rank(), self, let(
        strides := self.params[4],
        vs := self.params[4][1+loopid],
	HH(self.params[1], 
           self.params[2], 
           self.params[3], 
           Concatenation(strides{[1..loopid+1]}, [inner_its * vs], strides{[loopid+2..Length(strides)]})))),
    
    lambda := self >> let(
        n := self.params[2],
        b := self.params[3],
        s := self.strides(),
        ind := List([1..Length(s)], i -> When(i=1, Ind(n), var.fresh_t("q", TInt))),
        Lambda(Reversed(ind), b + Sum([1..Length(s)], i -> s[i] * ind[i]))),

    isUnitStride := self >> self.strides()[1] = 1
));

Class(BHH, HHBase, rec(
    abbrevs := [ (N,n,b,strides,refl) -> [toExpArg(N), toExpArg(n), toExpArg(b), List(_dropTail(strides,0), toExpArg), toExpArg(refl)] ],
    def := (N, n, b, strides, refl) -> Checked(
        IsList(strides), ForAll(strides, IsPosInt0Sym), IsPosIntSym(refl), 
	rec(N := N, n := n)), 

    upRank := self >> let(s:=self.strides(), 
	BHH(self.params[1], self.params[2], self.params[3], Concatenation([s[1], 0], Drop(s, 1)), self.params[5])),

    downRank := (self, loopid, ind) >> Cond(loopid > self.rank(), self, let(s:=self.strides(),
	BHH(self.params[1], self.params[2], self.params[3] + s[loopid+1] * ind, 
           ListWithout(s, loopid+1), self.params[5]))),
    
    split := (self, loopid, inner_its, outer_its) >> Cond(loopid > self.rank(), self, let(
        strides := self.strides(),
        vs := self.params[4][1+loopid],
	BHH(self.params[1], 
           self.params[2], 
           self.params[3], 
           Concatenation(strides{[1..loopid+1]}, [inner_its * vs], strides{[loopid+2..Length(strides)]}),
           self.params[5]))),
    
    lambda := self >> let(
        n := self.params[2],
        b := self.params[3],
        s := self.strides(),
        refl := self.params[5],
        inds := List([1..Length(s)], i -> When(i=1, Ind(n), var.fresh_t("q", TInt))),
        res := b + Sum([1..Length(s)], i -> s[i] * inds[i]),
        Lambda(Reversed(inds), cond(leq(inds[1], idiv(n-1,2)), res, refl-res))
    ),
));

#F KH(<N>, <n>, <b>, <strides>, <corr>) - generalized (I_m dirsum J_m dirsum I_m ...) L^mn_n thingy
#F   N - range
#F   n - domain
#F
#F   KH represents a multivariate function (j1 .. jn are indices of enclosing loops)
#F       (i, j1, ..., jn) -> b + strides[1]*i + strides[2]*j1 + ... + strides[n+1]*jn
#F
#F   Observe that dependence on the loop indices exists, even though no explicit
#F   loop index is present. This is how index free representation works.
#F
Class(KH, HHBase, rec(
    abbrevs := [ (N,n,b,strides,corr) -> [toExpArg(N), toExpArg(n), toExpArg(b), List(_dropTail(strides,0), toExpArg), List(corr, toExpArg)] ],
    def := (N, n, b, strides, corr) -> Checked(
        IsList(strides), ForAll(strides, IsPosInt0Sym), 
        IsList(corr),    ForAll(corr, IsIntSym), Length(corr) = 2, 
	rec(N := N, n := n)), 

    upRank := self >> let(s:=self.strides(), 
	KH(self.params[1], self.params[2], self.params[3], 
           Concatenation([s[1], 0], Drop(s, 1)), 
           self.params[5])),

    downRank := (self, loopid, ind) >> Checked(loopid=1, self.rank()=1, let(
        s:=self.strides(), corr:=self.params[5], b:=self.params[3],
	KH(self.params[1], 
           self.params[2], 
           b,
           [s[1]], 
           [ s[2]*ind      + cond(neq(imod(ind,2),0), corr[2], corr[1]), 
             s[1]-s[2]-ind + cond(neq(imod(ind,2),0), corr[1], corr[2]) ]))),
    
    lambda := self >> Checked(Length(self.strides())=1, let(
        n := self.params[2],  
        b := self.params[3], 
        s := self.strides()[1],
        i := Ind(n),
        corr_even := self.params[5][1], corr_odd := self.params[5][2],
        Lambda(i, b + s*i + imod(i+1, 2)*corr_even + imod(i, 2)*corr_odd)))
));

#F HHZ(<N>, <n>, <b>, <strides>) - generalized stride mod N index mapping (prime-factor FFT)
#F   N - range
#F   n - domain
#F
#F   HHZ represents a multivariate function (j1 .. jn are indices of enclosing loops)
#F       (i, j1, ..., jn) -> (b + strides[1]*i + strides[2]*j1 + ... + strides[n+1]*jn) mod N
#F
#F   Observe that dependence on the loop indices exists, even though no explicit
#F   loop index is present. This is same as in HH. 
#F
Class(HHZ, HHBase, rec(
    split := (self, loopid, inner_its, outer_its) >> Cond(loopid > self.rank(), self, let(
        strides := self.params[4],
        vs := self.params[4][1+loopid],
	HHZ(self.params[1], 
           self.params[2], 
           self.params[3], 
           Concatenation(strides{[1..loopid+1]}, [inner_its * vs], strides{[loopid+2..Length(strides)]})))),

    upRank := self >> let(s:=self.strides(), 
	HHZ(self.params[1], self.params[2], self.params[3], Concatenation([s[1], 0], Drop(s, 1)))),

    downRank := (self, loopid, ind) >> Cond(loopid > self.rank(), self, let(s:=self.strides(),
	HHZ(self.params[1], self.params[2], self.params[3] + s[loopid+1]*ind, ListWithout(s, loopid+1)))),

    lambda := self >> let(hlambda := ApplyFunc(HH, self.params).lambda(),
        CopyFields(hlambda, rec(expr := imod(hlambda.expr, self.params[1]))))
));

Declare(UU);

##    UU(<N>,<n>,<base>,<stride>,<columns>,<leading_dim>,
##                                 < <vstride_cols, vstride_rows>, ... > )
## 
##    index mapping: 
##       (i, j1, ..., jn) -> 
##          base + stride * (i / columns) + leading_dim * (i mod columns)
##               + vstride_cols[1]*j1 + ... + vstride_cols[n]*jn
##               + (vstride_rows[1]*j1 + ... + vstride_rows[n]*jn)* leading_dim
##
##    This is roughly the equivalent of an HH function for 2D inputs.

Class(UU, GTIndexFunction, rec(
    abbrevs := [ (N, n, bX, bY, s, c, ld, strides) -> [toExpArg(N), toExpArg(n), toExpArg(bX), toExpArg(bY),
                                            toExpArg(s), toExpArg(c), toExpArg(ld),
                                     List(_dropTail(strides,[0,0]), x->List(x,toExpArg))] ],
    def := (N, n, bX, bY, s, c, ld, strides) -> Checked(IsList(strides), ForAll(strides, x-> 
                                              IsList(x) and Length(x)=2 and ForAll(x,IsPosInt0Sym)), 
                                                rec(N := N, n := n)), 

    nloops := self >> Length(self.params[8]),
    free := self >> FreeVars(self.params),

    toSpl := (self, inds, kernel_size) >> self.downRankFull(inds),
    rank := self >> Length(self.params[8]),   # rank = number of implicit loop dependencies

    domain := self >> When(IsValue(self.params[2]), self.params[2].v, self.params[2]),
    range := self >> When(IsValue(self.params[1]), self.params[1].v, self.params[1]),

    upRank := self >> let(ss:=self.params[8], 
	UU(self.params[1], self.params[2], self.params[3], self.params[4], self.params[5], self.params[6], self.params[7],  
            Concatenation([[0,0]], ss))),

    downRank := (self, loopid, ind) >> Cond(loopid > self.rank(), self, let(ss:=self.params[8],
	UU(self.params[1], self.params[2], self.params[3] + ss[loopid][1] * ind, self.params[4] + ss[loopid][2] * ind * self.params[7],
            self.params[5], self.params[6], self.params[7], ListWithout(ss, loopid)))),

    split := (self, loopid, inner_its, outer_its) >> Cond(loopid > self.rank(), self, let(
        strides := self.params[8],
        vs := self.params[8][loopid],
        UU(self.params[1], 
           self.params[2], 
           self.params[3], 
           self.params[4],
           self.params[5],
           self.params[6],
           self.params[7],
           Concatenation(strides{[1..loopid]}, [inner_its * vs], strides{[loopid+1..Length(strides)]})))),

    
    lambda := self >> let(
        n := self.params[2],
        bX := self.params[3],
        bY := self.params[4],
        s := self.params[5],
        c := self.params[6],
        ld := self.params[7],
        ss := self.params[8],
        ind := List([1..Length(ss)], i -> var.fresh_t("q", TInt)),
        index := Ind(n),
        Lambda(Concatenation(Reversed(ind),[index]), (bX  + imod(index,c) * s) + (bY+idiv(index,c) * ld) + Sum([1..Length(ss)], i -> ss[i][1] * ind[i]) + Sum([1..Length(ss)], i -> ss[i][2] * ind[i] * ld))),

    isUnitStride := self >> Error("isUnitStride is not supported for now with UU"),
));

Declare(XChain);

#F XChain( <permuted [1..n]> ) - fTensor chain of fBase's and single fId.
#F
#F   This represents a subset of functions captured by HH. Namely those
#F   functions that described tightly packed index space, i.e.,
#F   if f is an XChain in k-nested loop with dimensions in loop_dims, then
#F
#F   domain(f) * Product(loop_dims) = range(f)
#F
Class(XChain, GTIndexFunction, rec(
    def := perm -> Checked(IsList(perm), ForAll(perm, IsPosInt0), 
	Set(Copy(perm))=[0..Length(perm)-1],
        rec()),

    range := self >> 0,
    domain := self >> 0,

    equals := (self, o) >> ObjId(self)=ObjId(o) and self.params[1]=o.params[1],

    printFull := self >> Print(self.name, "(", self.params[1], ")"),
    printShort := self >> Print("x(", PrintCS(self.params[1]), ")"),

    toSpl := (self, inds, kernel_size) >> let(fbases := List(inds, fBase),
	ApplyFunc(fTensor, List(self.params[1], i -> When(i=0, fId(kernel_size), fbases[i])))),

    toDiag := (self, loop_dims, kernel) >> let(fconst := List(loop_dims, d->fConst(d,1)),
	ApplyFunc(diagTensor, List(self.params[1], i -> When(i=0, kernel, fconst[i])))),

    without := (self, loopid) >> Checked(loopid >= 1, loopid <= Length(self.params[1])-1, let(
	lst := ListWithout(self.params[1], Position(self.params[1], loopid)),
	XChain(List(lst, x -> When(x < loopid, x, x-1))))),

    part := (self, loopid, ind, kernel_size, loop_dims) >> 
        ApplyFunc(fTensor, List(self.params[1], i ->
	    Cond(i=0,      fId(kernel_size),
		 i=loopid, fBase(ind),
		 fId(loop_dims[i])))),
 
    composeWith := (self, f) >> When(ObjId(f)<>XChain, Error("Can only compose with another XChain"),
	let(fperm := f.params[1], 
	    myperm := self.params[1],
	    pos := Position(myperm, 0),
	    XChain(Concatenation(1+myperm{[1..pos-1]}, fperm, 1+myperm{[pos+1..Length(myperm)]})))),
));


# NOTE: toSpl -> SubstTopDown( ind -> inds[i] )
# Normal functions retrofitted for GT
fId.toSpl  := (self, inds, ksize) >> self; 
fId.without:= (self, loopid) >> self;
fId.part   := (self, loopid, loopvar, kernel_size, loop_dims) >> self;
#fId.toDiag := (self, loop_dims) >> fConst(TReal, self.params[1], 1);

fBase.toSpl  := (self, inds, ksize) >> Cond(
    ObjId(self.params[2])=ind, fBase(inds[self.params[2].n]), 
    self);

fBase.without := (self, loopid) >> Cond(
    self.params[2]=ind(self.params[1], loopid), fBase(self.params[1], 0),
    self);

fBase.part := (self, loopid, loopvar, ksize, loop_dims) >> Cond(
    self.params[2]=ind(self.params[1], loopid), fAdd(self.params[1], self.params[1], loopvar), 
    fId(self.params[1]));

fTensor.part := (self, loopid, loopvar, ksize, loop_dims) >>
     ApplyFunc(fTensor, List(self.children(), c -> c.part(loopid, loopvar, ksize, loop_dims)));

fTensor.without := (self, loopid) >> 
     ApplyFunc(fTensor, List(self.children(), c -> c.without(loopid)));

fTensor.toSpl := (self, inds, ksize) >> 
     ApplyFunc(ObjId(self), List(self.children(), c -> c.toSpl(inds, ksize)));

fCompose.toSpl := (self, inds, ksize) >> 
     ApplyFunc(fCompose, List(self.children(), c -> c.toSpl(inds, ksize)));

FuncClass.toSpl := (self, inds, ksize) >> self;

#FuncClassOper.rank := self >> Maximum(List(self.children(), x->x.rank()));
#FuncClass.rank := self >> 0;
#DiagFunc.rank := self >> 0;

FuncClassOper.isUnitStride := self >> false; 
FuncClass.isUnitStride     := self >> false;
fId.isUnitStride := self >> true;
fTensor.isUnitStride := self >> ForAll(self._children, x->x.isUnitStride());
#fBase.toDiag := (self, loop_dims) >> fConst(TReal, self.params[1], 1);
#fId.part   := (self, loopid, ind, kernel_size, loop_dims) >> self;

ListAddP := function(l1, l2, c)
    local i, lenl1, lenl2, res, maxlen, minlen;
    lenl1 := Length(l1);
    lenl2 := Length(l2);
    minlen := Minimum(lenl1, lenl2);
    maxlen := Maximum(lenl1, lenl2);
    res := Concatenation(l1, Replicate(maxlen-lenl1, c));   
    for i in [1..lenl2] do 
        res[i] := res[i] + l2[i];
    od;
    return res;
end;

ListAddZP := (l1, l2) -> ListAddP(l1,l2,0);
ListAddLP := (l1, l2) -> ListAddP(l1,l2,[0,0]);

