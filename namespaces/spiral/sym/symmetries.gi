
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# equality of scalars
scalareq := (a,b) -> AbsFloat(a-b) < 1e-10;
# equality of lists
lseq := (a,b) -> Length(a)=Length(b) and ForAll([1..Length(a)], scalareq(a[i], b[i]));

lequiv := (a,b) -> Cond(
    lseq(a,b),   "eq", 
    lseq(a, -b), "eqneg",
    lseq(a, Reversed(b)), "rev", 
    lseq(a, -Reversed(b)), "revneg",
    "none");

tovec := lst -> TransposedMat([lst]);
tolst := vec -> TransposedMat(vec)[1];
rpart := vec -> List(vec, x->ReComplex(Complex(x)));
ipart := vec -> List(vec, x->ImComplex(Complex(x)));

rand := arg -> RandomList([-100..100]);
#rand := x->x;

##
## Constructors for symmetric sequences
##
real := n -> List([1..n], rand);
cplx := (rsym, isym) -> (n -> rsym(n) + E(4)*isym(n));

even0 := n -> let(len := Int((n-1)/2), lst := List([2..len+1], rand),
    Flat([rand(1), lst, When(IsEvenInt(n),rand(len+2),[]), Reversed(lst)]));

even00 := n -> let(len := Int((n-1)/2), lst := List([1..len], rand),
    Flat([0, lst, When(IsEvenInt(n),rand(len+1),[]), Reversed(lst)]));
odd0 := n -> let(len := Int((n-1)/2), lst := List([2..len+1], rand),
    Flat([rand(1), lst, When(IsEvenInt(n),0,[]), -Reversed(lst)]));
odd00 := n -> let(len := Int((n-1)/2), lst := List([1..len], rand),
    Flat([0, lst, When(IsEvenInt(n),0,[]), -Reversed(lst)]));

even1 := n -> let(len := Int(n/2), lst := List([1..len], rand),
    Flat([lst, When(IsOddInt(n),rand(len+1),[]), Reversed(lst)]));
odd1 := n -> let(len := Int(n/2), lst := List([1..len], rand),
    Flat([lst, When(IsOddInt(n),0,[]), -Reversed(lst)]));

jeven1 := n -> Checked(IsEvenInt(n), let(lst:=List([1..n/2], rand), Flat([lst,lst])));
jodd1  := n -> Checked(IsEvenInt(n), let(lst:=List([1..n/2], rand), Flat([lst,-lst])));
jeven0 := n -> Checked(IsOddInt(n), Concatenation([rand(0)], jeven1(n-1)));
jeven00 := n -> Checked(IsOddInt(n), Concatenation([0], jeven1(n-1)));
jodd0  := n -> Checked(IsOddInt(n), Concatenation([rand(0)], jodd1(n-1)));
jodd00 := n -> Checked(IsOddInt(n), Concatenation([0], jodd1(n-1)));

ce0 := cplx(even0, odd00);
co0 := cplx(odd0, even00);
ce1 := cplx(even1, odd1);
co1 := cplx(odd1, even1);

jce0 := cplx(jeven0, jodd00);
jco0 := cplx(jodd0, jeven00);
jce1 := cplx(jeven1, jodd1);
jco1 := cplx(jodd1, jeven1);

raderce0 := n -> Checked(IsPrime(n+1), Drop(
    tolst(MatSPL(DFT_Rader.raderMid(n+1,1,PrimitiveRootMod(n+1))) * 
	  tovec(Concat([rand(0)], ce0(n)))), 1));

upsample0 := n -> Checked(IsEvenInt(n),  List([1..n], i -> When(IsEvenInt(i),0,rand())));

upsample1 := n -> Checked(IsEvenInt(n),  List([1..n], i -> When(IsOddInt(i),0,rand())));

_even := n -> ((n+1) mod 2);
_odd := n -> (n mod 2);

_minsetPoints := function(points, n)
    local uniq, cur, p;
    cur := []; uniq := Set([]);
    for p in points do
        if not (p in uniq) and p < n then 
	    Add(cur, p); 
	    AddSet(uniq, p); 
	fi;
    od;
    return cur;
end;

_CommuteExt := function(perm, sym, refl_diff, n, drop0) 
    local mat, N, ext, points, minset, pmat;
    mat:=MatSPL(Prm(perm));
    N := perm.domain();
    ext := sym.ext(N,false);
    points := List(Refl(n, N-refl_diff, N, perm).tolist(), x->x.ev());
    minset := When(not drop0, _minsetPoints(points, n), Drop(_minsetPoints(points,n), 1)-1);
    pmat := MatSPL(Scat(FList(minset).setRange(Cols(ext))));
    return 
       mat * MatSPL(ext) * pmat;
end;


# CommuteExt(spl, sym, scaled) - find X, such that spl * sym.ext(n, scaled) = X * pspl
#  X = spl * sym.ext(n, scaled) * pspl^-1
#

CommuteExt1 := (perm, sym) -> _CommuteExt(perm, sym, -1, Cols(sym.ext(perm.domain(),false)), false);
CommuteExt0 := (perm, sym) -> _CommuteExt(perm, sym, 0,  Cols(sym.ext(perm.domain(),false)), false);
CommuteExt00 := (perm, sym) -> _CommuteExt(perm, sym, 0, 1+Cols(sym.ext(perm.domain(),false)), true);

##
## Constructors for symmetric sequences
##
Class(Symmetry, rec(
    isSymmetry := true,
    __call__ := (self, n) >> self.seq(n),
    red := (self, n, scaled) >> self.ext(n, scaled).transpose(),
    verify := (self, n) >> MatSPL(self.red(n,false)*self.ext(n, true))
));

IsSymmetry := x -> IsRec(x) and IsBound(x.isSymmetry) and x.isSymmetry;

_gathI := (nn, top_hole, bot_hole) -> let(n:=Int(nn), Gath(fAdd(n, n-top_hole-bot_hole, top_hole)));
_gathI2 := (nn, top_hole, bot_hole) -> let(n:=Int(nn), Gath(H(n, Int((n-top_hole-bot_hole+1)/2), top_hole, 2)));
_gathJ := (nn, top_hole, bot_hole) -> let(n:=Int(nn), Gath(fCompose(fAdd(n, n-top_hole-bot_hole, top_hole), J(n-top_hole-bot_hole))));

Class(GathExtend, Sym, rec(def := (n, symmetry) -> symmetry.ext(n, true)));
Class(GathExtendU, Sym, rec(def := (n, symmetry) -> symmetry.ext(n, false)));
Class(GathExtendZ, Sym, rec(def := (n, symmetry) -> DirectSum(I(1),-J(n-1))*symmetry.ext(n, true)));
Class(GathExtendZU, Sym, rec(def := (n, symmetry) -> DirectSum(I(1),-J(n-1))*symmetry.ext(n, false)));
Class(ScatReduce, Sym, rec(def := (n, symmetry) -> symmetry.red(n, true)));
Class(ScatReduceU, Sym, rec(def := (n, symmetry) -> symmetry.red(n, false)));


Class(EOSymmetry, Symmetry, rec(tsize := "cols"));

Class(Upsample0, EOSymmetry, rec(
    seq:=  n -> Checked(IsEvenInt(n),  List([1..n], i -> When(IsEvenInt(i),0,rand()))),
    ext := (n, scaled) -> Checked(IsEvenInt(n), Tensor(I(n/2),Mat([[1],[0]]))),
    red := (n, scaled) -> Checked(IsEvenInt(n), Tensor(I(n/2),Mat([[1,0]]))),
));

Class(Upsample1, EOSymmetry, rec(
    seq:=  n -> Checked(IsEvenInt(n),  List([1..n], i -> When(IsEvenInt(i),0,rand()))),
    ext := (n, scaled) -> Checked(IsEvenInt(n), Tensor(I(n/2),Mat([[0],[1]]))),
    red := (n, scaled) -> Checked(IsEvenInt(n), Tensor(I(n/2),Mat([[0,1]]))),
));

Class(Even0, EOSymmetry, rec(
    seq := n -> let(len := Int((n-1)/2), lst := List([2..len+1], rand),
	Flat([rand(1), lst, When(IsEvenInt(n),rand(len+2),[]), Reversed(lst)])),
    ext := (n, scaled) -> let(hf := When(scaled, 1/2, 1), 
	DirectSum(I(1), 
	            VStack(
			DirectSum(hf*I(Int((n-1)/2)), I(_even(n))),
			hf*_gathJ(n/2, 0, _even(n))))),
#    red := n -> Gath(fAdd(n, Int((n+2)/2), 0))
));

Class(Even0star, EOSymmetry, rec(
    seq := n -> let(len := Int((n-1)/2), lst := List([2..len+1], rand),
	Flat([rand(1), lst, When(IsEvenInt(n),0,[]), Reversed(lst)])),
    ext := (n, scaled) -> let(hf := When(scaled, 1/2, 1), 
	DirectSum(I(1), 
	            VStack(
			hf*I(Int((n-1)/2)), 
                        When(IsEvenInt(n), O(1, Int((n-1)/2)), []), 
			hf*J(Int((n-1)/2))))),
#    red := n -> Gath(fAdd(n, Int((n+2)/2), 0))
));

Class(Even00, EOSymmetry, rec(
    seq := n -> let(len := Int((n-1)/2), lst := List([2..len+1], rand),
	Flat([0, lst, When(IsEvenInt(n),rand(len+2),[]), Reversed(lst)])),
    ext := (n, scaled) -> let(hf := When(scaled, 1/2, 1), 
	VStack(O(1,Int(n/2)), 
	       DirectSum(hf*I(Int((n-1)/2)), I(_even(n))),
	       hf*_gathJ(n/2, 0, _even(n)))),
#    red := n -> Gath(fAdd(n, Int(n/2), 1))
));

Class(Even00star, EOSymmetry, rec(
    seq := n -> let(len := Int((n-1)/2), lst := List([2..len+1], rand),
	Flat([0, lst, When(IsEvenInt(n),0,[]), Reversed(lst)])),
    ext := (n, scaled) -> let(hf := When(scaled, 1/2, 1), 
	VStack(O(1,Int(n/2)), 
	       DirectSum(hf*I(Int((n-1)/2)), O(_even(n))),
	       hf*_gathJ(n/2, 0, _even(n)))),
#    red := n -> Gath(fAdd(n, Int(n/2), 1))
));

Class(Even1, EOSymmetry, rec(
    seq := n -> let(len := Int(n/2), lst := List([1..len], rand),
	Flat([lst, When(IsOddInt(n),rand(len+1),[]), Reversed(lst)])),
    ext := (n, scaled) -> let(hf := When(scaled, 1/2, 1), 
	When(IsEvenInt(n), 
	    hf * VStack(I(n/2), J(n/2)), 
	    VStack(DirectSum(hf*I((n-1)/2), I(1)), hf*_gathJ((n+1)/2, 0, 1))))
#    red := n -> Gath(fAdd(n, Int((n+1)/2), 0))
));


Class(Odd0, EOSymmetry, rec(
    seq := n -> let(len := Int((n-1)/2), lst := List([2..len+1], rand),
	Flat([rand(1), lst, When(IsEvenInt(n),0,[]), -Reversed(lst)])),
    ext := (n, scaled) -> let(hf := When(scaled, 1/2, 1),
	DirectSum(I(1), 
	         hf*VStack(I(Int((n-1)/2)), 
		            When(IsEvenInt(n), O(1, Int((n-1)/2)), []), 
			    -J(Int((n-1)/2))))) 
#    red := n -> Gath(fAdd(n, Int((n+2)/2), 0))
));

Class(Odd00, EOSymmetry, rec(
    seq := n -> let(len := Int((n-1)/2), lst := List([2..len+1], rand),
	Flat([0, lst, When(IsEvenInt(n),0,[]), -Reversed(lst)])),
    ext := (n, scaled) -> let(hf := When(scaled, 1/2, 1), 
	        hf * VStack(O(1,Int((n-1)/2)), 
	                     I(Int((n-1)/2)),
			     When(IsEvenInt(n), O(1,Int((n-1)/2)), []), 
	                     -J(Int((n-1)/2)))),
#    red := n -> Gath(fAdd(n, Int((n-1)/2), 1))
));

Class(Odd00star, EOSymmetry, rec(
    seq := n -> let(len := Int((n-1)/2), lst := List([2..len+1], rand),
	Flat([0, lst, When(IsEvenInt(n),rand(len+2),[]), -Reversed(lst)])),
    ext := (n, scaled) -> let(hf := When(scaled, 1/2, 1), 
	        hf * VStack(O(1,Int((n-1)/2)), 
	                    I(Int((n-1)/2)),
                            When(IsEvenInt(n), RowVec( Replicate(Int((n-3)/2),0)::[1]  ), []), 
                            -J(Int((n-1)/2)))),
#    red := n -> Gath(fAdd(n, Int((n-1)/2), 1))
));


Class(Odd1, EOSymmetry, rec(
    seq := n -> let(len := Int(n/2), lst := List([1..len], rand),
	Flat([lst, When(IsOddInt(n),0,[]), -Reversed(lst)])),
    ext := (n, scaled) -> let(hf := When(scaled, 1/2, 1), 
	     hf * VStack(I(Int(n/2)), 
		         When(IsOddInt(n), O(1, Int(n/2)),[]), 
			 -J(Int(n/2)))),
#    red := n -> Gath(fAdd(n, Int(n/2), 0))
));


Class(Real, EOSymmetry, rec(
    seq := n -> List([1..n], rand),
    ext := (n, scaled) -> I(n),
    tsize := "rows", # does not matter, since ext is square
));

# NOTE: red != ext^T, how to handle?
# - silently drop scaling?

# DirectSum((I tensor W), j * I(1))
#
Class(Conj_Base, Symmetry, rec(
    __call__ := (self, w, j) >> Checked(IsMat(w), DimensionsMat(w)=[2,2], 
	WithBases(self, rec(
		w := w, 
		j := j, 
		operations := PrintOps
    ))),
    tsize := "rows",
    print := self >> Print(self.name, "(", self.w, ", ", self.j, ")"),
    W := self >> When(IsBound(self.isOdd) and self.isOdd, self.w * DiagonalMat([1, -1]), self.w),
    invertW := self >> let(base := self.__bases__[1], base(TransposedMat(self.w^-1), 1/self.j)),
    seq := (self, n) >> self.re.seq(n) + E(4) * self.im.seq(n),
    verify := (self, n) >> MatSPL(self.red(n,false)*self.ext(n, false))
));

Class(Conj0_Base, Conj_Base, rec(
    P := n -> DirectSum(I(1), L_or_OS(n-1, Int(n/2)) * DirectSum(I(Int(n/2)), J(Int((n-1)/2)))),
    ext := (self, n, scaled) >> self.invertW().red(n, scaled).transpose(),
    red := (self, n, scaled) >> let(
	W := Mat(When(scaled, self.W(), self.W())),
        self.PP(n) * DirectSum(I(1), Tensor(I(Int((n-1)/2)), W), self.j * I(_even(n))) * self.P(n)
    )
));

Class(Conj1_Base, Conj_Base, rec(
    P := n -> L_or_OS(n, Int((n+1)/2)) * DirectSum(I(Int((n+1)/2)), J(Int(n/2))),
    seq := (self, n) >> self.re.seq(n) + E(4) * self.im.seq(n),
    ext := (self, n, scaled) >> self.invertW().red(n, scaled).transpose(),
    red := (self, n, scaled) >> let(
	W := Mat(When(scaled, self.W(), self.W())),
        self.PP(n) * DirectSum(Tensor(I(Int(n/2)), W), self.j * I(_odd(n))) * self.P(n)
    )
));

Class(Conj0_R2R, Conj0_Base, rec(PP := (self, n) >> self.P(n).transpose()));
Class(Conj1_R2R, Conj1_Base, rec(PP := (self, n) >> self.P(n).transpose()));

Class(CE0_R2R, Conj0_R2R, rec(re := Even0, im := Odd00,  isOdd := false));
Class(CO0_R2R, Conj0_R2R, rec(re := Odd0,  im := Even00, isOdd := true, j := -E(4)));
Class(CE1_R2R, Conj1_R2R, rec(re := Even1, im := Odd1,   isOdd := false));
Class(CO1_R2R, Conj1_R2R, rec(re := Odd1,  im := Even1,  isOdd := true, j := -E(4)));

_m := n -> When(n=1, Mat([[1], [0]]), I(0));
_mj := n -> When(n=1, Mat([[0], [1]]), I(0));

Class(Conj0_R2Cpx, Conj0_Base, rec(PP := (self, n) >> 
	DirectSum(_m(1), I(n-1-_even(n)), _m(_even(n)))));

Class(Conj1_R2Cpx, Conj1_Base, rec(PP := (self, n) >> 
	DirectSum(I(n-_odd(n)), When(self.isOdd, _mj(_odd(n)), _m(_odd(n))))));

Class(CE0_R2Cpx, Conj0_R2Cpx, rec(re := Even0, im := Odd00,  isOdd := false));
Class(CO0_R2Cpx, Conj0_R2Cpx, rec(re := Odd0,  im := Even00, isOdd := true, j := -E(4)));
Class(CE1_R2Cpx, Conj1_R2Cpx, rec(re := Even1, im := Odd1,   isOdd := false));
Class(CO1_R2Cpx, Conj1_R2Cpx, rec(re := Odd1,  im := Even1,  isOdd := true, j := -E(4)));

W_RFT := 1/2*[[1, 1], [-E(4), E(4)]];
W_RFTT := 1/2*[[1, 1], [E(4), -E(4)]];
W_URFT := [[1, 1], [-E(4), E(4)]];
W_URFTT := [[1, 1], [E(4), -E(4)]];
W_DHT := 1/2*[[1-E(4), 1+E(4)], [1+E(4), 1-E(4)]];

rsym := function(x)
    local n, nc, nf, mid, uniq, zeros, fst_0, mid_0, fstmid;
    n := Length(x); nc := Int((n+1)/2); nf := Int(n/2); mid := nf;
    if n=0 then return "empty"; fi;

    fst_0 := scalareq(x[1], 0);
    mid_0 := scalareq(x[nf+1], 0);
    fstmid := [Cond(fst_0,0, 1), Cond(mid_0, 0, 1)];

    if ForAll(x, e->scalareq(e,0)) then return "zero";

    # normal symmetries
    elif ForAll([1..nc-1], i->scalareq(x[1+ i],  x[1+ (n-i) mod n])) then
        fstmid[2] := Cond(IsOddInt(n), 1, fstmid[2]);
        if   fstmid=[0,0] then return "even00*"; 
	elif fstmid=[0,1] then return "even00"; 
        elif fstmid=[1,0] then return "even0*"; 
        elif fstmid=[1,1] then return "even0"; 
        fi;
    elif ForAll([1..nc-1], i->scalareq(x[1+ i], -x[1+ (n-i) mod n])) then
        fstmid[2] := Cond(IsOddInt(n), 0, fstmid[2]);
        if   fstmid=[0,0] then return "odd00"; 
	elif fstmid=[0,1] then return "odd00*"; 
        elif fstmid=[1,0] then return "odd0"; 
        elif fstmid=[1,1] then return "odd0*"; 
        fi;
    elif ForAll([0..mid-1], i->scalareq(x[1+ i],  x[1+ (n-1-i)])) then 
        if IsOddInt(n) and mid_0 then return "even1*"; 
        else return "even1";
        fi;
    elif ForAll([0..mid-1], i->scalareq(x[1+ i], -x[1+ (n-1-i)])) then 
        if IsOddInt(n) and not mid_0 then return "odd1*"; 
        else return "odd1";
        fi;

    # symmetries with flipped bottom, i.e. abcabc
    elif n>2 and IsOddInt(n) and ForAll([1..nc-1], i->scalareq(x[1+ i],  x[1+ mid+i])) then
	if fst_0 then return "jeven00"; else return "jeven0"; fi;
    elif n>2 and IsOddInt(n) and ForAll([1..nc-1], i->scalareq(x[1+ i], -x[1+ mid+i])) then
	if fst_0 then return "jodd00"; else return "jodd0"; fi;
    elif n>2 and ForAll([0..mid-1], i->scalareq(x[1+ i],  x[1+ mid+i])) then return "jeven1";
    elif n>2 and ForAll([0..mid-1], i->scalareq(x[1+ i], -x[1+ mid+i])) then return "jodd1";

    # rader symmetries, similar to normal but every other sample in bottom half is negated
    elif  ForAll([1..nc-1], i->scalareq(x[1+i], (-1)^i * x[1+(n-i) mod n])) then
	if fst_0 then return "rader_even00"; else return "rader_even0"; fi;
    elif  ForAll([1..nc-1], i->scalareq(x[1+i], - (-1)^i * x[1+(n-i) mod n])) then
	if fst_0 then return "rader_odd00"; else return "rader_odd0"; fi;

    # symmetries with normal shape, but elements negated in some non-recognized way
    elif  ForAll([1..nc-1], i->scalareq(x[1+i], x[1+(n-i) mod n]) or scalareq(x[1+i], -x[1+(n-i) mod n])) then
	if fst_0 then return "eo00"; else return "eo0"; fi;
    elif  ForAll([0..mid-1], i->scalareq(x[1+i], x[1+(n-1-i)]) or scalareq(x[1+i], -x[1+(n-1-i)])) then
	return "eo1"; 
    else
	uniq := Length(Set(List(x, e->V(AbsFloat(e)))));
	zeros := Length(Filtered(x, e->scalareq(e,0)));
	if uniq <= Int((n+2)/2) then
            # no obvious symmetry, but only approx half unique elements
	    # so sequence is still redundant
	    return Concat("red_", String(uniq-zeros), "/", String(zeros), "/",String(Length(x))); 
	else return "none";
	fi;
    fi;
end;

risym := function(x)
    local rs, is, re, im;
    re := List(x, e->ReComplex(Complex(e)));
    im := List(x, e->ImComplex(Complex(e)));
    rs := rsym(re);
    is := rsym(im);
    return [rs,is];
end;

csym := function(x)
   local s;
   s := sym(x);
   if s = ["even0", "odd00"] then return "ce0";
   elif s = ["odd0", "even00"] then return "co0";
   elif s = ["even1", "odd1"] then return "ce1";
   elif s = ["odd1", "even1"] then return "co1";
   elif s = ["jeven1", "jodd1"] then return "jce1";
   elif s = ["jodd1", "jeven1"] then return "jco1";
   elif s = ["rader_even0", "rader_odd00"] then return "rader_ce0";
   elif s = ["rader_odd0", "rader_even00"] then return "rader_co0";
   elif s = ["none", "zero"] then return "real";
   elif s = ["zero", "none"] then return "imag";
   elif s = ["zero", "zero"] then return "zero";
   elif s = ["none", "none"] then return "cplx";
   else return s;
   fi;
end;

       
part := (v, str, n) -> v{1 + [ 0 .. (Length(v) / str - 1) ] * str + n};
    
fxi_inputs := (inp, fsize) -> let(
    m := Length(inp)/fsize, 
    List([0..m-1], x->part(inp,m,x)));

fxi_rsyms := (inp, fsize) -> List(fxi_inputs(inp, fsize), rsym);
fxi_syms := (inp, fsize) -> List(fxi_inputs(inp, fsize), risym);

transf_syms := (n, transforms, syms) -> 
   List(syms, s ->
       List(transforms, transform -> 
	   let(inpvec := TransposedMat([s(n)]),
	       outvec := TransposedMat(MatSPL(transform(n))*inpvec)[1],
	       outsym := sym(outvec),
	       Cond(outsym[2]="zero", outsym[1],
		    outsym[1]="zero", Concat("j*", outsym[2]),
		    "none"))));

ctransf_syms := (n, transforms, syms) -> 
   List(syms, s ->
       List(transforms, transform -> 
	   let(inpvec := TransposedMat([s(n)]),
	       outvec := TransposedMat(MatSPL(transform(n))*inpvec)[1],
	       risym(outvec))));


# transf_syms(18, [ DFT1, DFT2, DFT3, DFT4 ], [even00, even1, odd00, odd1]);

# fxi_syms(even0(36),6);
# fxi_syms(even00(36),6);
# fxi_syms(even1(36),6);
# fxi_syms(odd0(36),6);
# fxi_syms(odd00(36),6);
# fxi_syms(odd1(36),6);


#    if rs="none" and is="none" then return "none";
#    elif rs="none" and is="zero" then return "real";
#    elif rs="even" and is="odd" then return "ce";
#    elif rs="even" and is="odd" then return "ce";
#end;
twpow :=  (N,n) -> List([0..N-1], j -> (j mod n) * Int(j / n));

tw1 :=  (N,n) -> List([0..N-1], j -> E(4*N)^((2*(j mod n))   * (2*Int(j / n))));
tw2 :=  (N,n) -> List([0..N-1], j -> E(4*N)^((2*(j mod n))   * (2*Int(j / n)+1)));
tw3 :=  (N,n) -> List([0..N-1], j -> E(4*N)^((2*(j mod n)+1) * (2*Int(j / n))));
tw4 :=  (N,n) -> List([0..N-1], j -> E(4*N)^((2*(j mod n)+1) * (2*Int(j / n)+1)));
dct2diag := N -> List([0..N-1], i->E(2*N)^i);
dct4diag := N -> List([0..N-1], i->E(4*N)^(2*i+1));
idct2diag := N -> List([0..N-1], i->E(2*N)^-i);
listmul := (l1,l2) -> List([1..Length(l1)], i->l1[i]*l2[i]);

ComplexFFT2 := input -> TransposedMat(MatSPL(DFT2(Length(input))) * TransposedMat([input]))[1];
ComplexFFT3 := input -> TransposedMat(MatSPL(DFT3(Length(input))) * TransposedMat([input]))[1];
ComplexFFT4 := input -> TransposedMat(MatSPL(DFT4(Length(input))) * TransposedMat([input]))[1];


# Macro Extensions
#
MacroExt0 := (k,a,b,c,d) -> Checked(IsSymmetry(a), IsSPL(b), IsSPL(d), b.dims()=d.dims(),
    IsOddInt(k) or IsSymmetry(c), 
    Cond(IsOddInt(k),
         let(kk := (k-1)/2, n := Cols(b), 
             top := DirectSum(GathExtendU(n, a), Tensor(I(kk), b)),
             VStack(top,
                    Tensor(J(kk), d) * Gath(fAdd(Cols(top), kk*Cols(d), Cols(a.ext(n,true)))))),
         let(kk := (k-2)/2, n := Cols(b), 
             top := DirectSum(GathExtendU(n, a), Tensor(I(kk), b), GathExtendU(n, c)),
             VStack(top, 
                    Tensor(J(kk), d) * Gath(fAdd(Cols(top), kk*Cols(d), Cols(a.ext(n,true))))))
         ));

MacroExt1 := (k,b,c,d) -> Checked(IsSPL(b), IsSPL(d), b.dims()=d.dims(),
    IsEvenInt(k) or IsSymmetry(c), 
    Cond(IsOddInt(k),
         let(kk := (k-1)/2, n := Cols(b), 
             top := DirectSum(Tensor(I(kk), b), GathExtendU(n, c)),
             VStack(top,
                    Tensor(J(kk), d) * Gath(fAdd(Cols(top), kk*Cols(d), 0)))),
         let(kk := k/2, 
             VStack(Tensor(I(kk), b),
                    Tensor(J(kk), d)))
         ));

Jp := x -> OS(x, -1);

listExtMat := mat -> List(mat, row -> let(n:=PositionProperty(row, x->x<>0), 
    Cond(n=false, 0, 
         row[n]<0, -n,
         +n)));

analyze := (stride, extmat) -> let(
    lst := listExtMat(MatSPL(extmat)),
    fsize := Length(lst)/stride,
    rec(inp := fxi_inputs(lst, fsize),
        sym := fxi_rsyms(lst, fsize)));

analyzeT := (stride, extmat, partsizes) -> let(
    fsize := Rows(extmat)/stride,
    lst := listExtMat(TransposedMat(MatSPL(L(fsize*stride, stride)*extmat))),
    part := Drop(ScanL(partsizes, (p, x) -> [Last(p)+1..Last(p)+x], [0]), 1),
    rec(out := List(part, p->lst{p}))); 

panalyze := function(stride, extmat) 
    local r, zip, i, e, sym, n, mid, j, cur, absprev, prev, sign;
    r := analyze(stride, extmat);
    zip := Zip2(r.sym, r.inp);
    n := Length(zip);
    mid := Int((n+1)/2);
    for i in [1..Length(zip)] do
        e := zip[i];
        if e[1] = "none" then 
            sym := "I";
            if i > mid then
                cur := List(e[2], AbsInt);
                for j in [1..mid] do                 
                   prev := r.inp[j];
                   absprev := List(r.inp[j], AbsInt);
                   if cur=Reversed(absprev) then 
                       prev := Reversed(prev);
                       sign := Cond(e[2]=prev, "", e[2]=-prev, "-", "#");
                       sym := sign :: "J/" :: StringInt(j); 
                   elif cur=[absprev[1]]::Reversed(Drop(absprev, 1)) then 
                       prev := [prev[1]]::Reversed(Drop(prev, 1)); 
                       sign := Cond(e[2]=prev, "", e[2]=-prev, "-", "#");
                       sym := sign :: "J'/"::StringInt(j);  
                   fi;
                od;
            fi;
        else
            sym := YellowStr(e[1]);
        fi;
        Print(sym, ": ", e[2], "\n");
    od;
end;

