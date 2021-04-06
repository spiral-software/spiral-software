
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#P Parametrized permutation classes
#P --------------------------------
#P

## -----------------------------------------------------------------------------
#F L(<n>, <str>) - stride permutation
##
Class(L, PermClass, rec(
    def := (n,str) -> Checked(IsPosIntSym(n), IsPosIntSym(str),
        (not (IsInt(n) and IsInt(str)) or n mod str = 0), rec()),

    domain := self >> self.params[1],
    range  := self >> self.params[1],

    lambda := self >> let(
        n := self.params[1], str := self.params[2], i := Ind(n),
        Lambda(i, idiv(i, n/str) + str * imod(i, n/str))),

    transpose := self >> self.__bases__[1](self.params[1], self.params[1] / self.params[2]),
    isSymmetric := self >> (self.params[1] = self.params[2]^2) or (self.params[2] = 1) or (self.params[2] = self.params[1]),
    printlatex := self >> Print(" \\stride^{", self.params[1], "}_{",self.params[2],"} ")

));

## -----------------------------------------------------------------------------
#F Tr(<k>, <str>) - stride permutation L(k*str, str),
##
## This L equivalent is used to simplify rewriting, since k is not explicit in L,
## and if symbolics are used, to extract it we have to do simplification tricks,
## eg.  n/f1(n) = f2(n).
##
## Tr stands for Transpose.
##
Class(Tr, PermClass, rec(
    def := (k,str) -> Checked(IsPosIntSym(k), IsPosIntSym(str), rec()),

    lambda := self >> let(
        k := self.params[1], str := self.params[2], i := Ind(k*str),
        Lambda(i, idiv(i, k) + str * imod(i, k))),

    domain := self >> self.params[1] * self.params[2],
    range := self >> self.params[1] * self.params[2],

    dims := self >> [self.range(), self.domain()],
  
    transpose := self >> self.__bases__[1](self.params[2], self.params[1])
));

## -----------------------------------------------------------------------------
#F Z(N,k) : Cyclic shift by <k>, permutation {0, ..., n-1} -> {k,...,n-1, 0,...,k-1}
##
# if one redefines .lambda, then .transposed should be also handled in custom .lambda
Class(Z, PermClass, rec(
    abbrevs := [ n -> [n,1] ],
    def := (n,k) -> rec(), 
    domain := self >> self.params[1],
    range  := self >> self.params[1],
    lambda := self >> let(n:=self.params[1], k:=self.params[2], i := Ind(n),
        Lambda(i, imod(i + k, n))),
    transpose := self >> self.__bases__[1](self.params[1], self.params[1] - self.params[2])
));

## -----------------------------------------------------------------------------
#F J(N) : NxN reverse identity, also known as reversal permutation (1,n)(2,n-1)...
##
Class(J, PermClass, rec(
    def := n -> rec(), 
    lambda := self >> let(i := Ind(self.params[1]), Lambda(i,self.params[1]-i-1)),
    domain := self >> self.params[1],
    range  := self >> self.params[1],
    transpose := self >> self,
    isSymmetric := True
));

## -----------------------------------------------------------------------------
#F OddStride(<n>, <str>)
#F OS(<n>, <str>)        - odd stride permutation (i -> i * str mod n)
##
Class(OS, PermClass, rec(
    isCyclic := true,
    def := (n,str) -> Checked(IsPosIntSym(n), IsIntSym(str), n > 0,
        AnySyms(n,str) or (Gcd(EvalScalar(n),EvalScalar(str))=1), rec()),
    domain := self >> self.params[1],
    range  := self >> self.params[1],
    lambda := self >> let(i := Ind(self.params[1]),
        Lambda(i, imod(self.params[2] * i, self.params[1]))),
    transpose := self >> self.__bases__[1](self.params[1], self.params[2]^-1 mod self.params[1])
));
OddStride := OS;

Declare(gammaTensor);
## -----------------------------------------------------------------------------
#F CRT(<r>, <s>) - Chinese remainder theorem permutation function of size r*s
##
Class(CRT, PermClass, rec(
    isCyclic := true,

    abbrevs := [ (r, s) -> Checked(IsPosIntSym(r), IsPosIntSym(s), AnySyms(r,s) or Gcd(r,s)=1,
        let(alpha := 1/s mod r,
        beta  := 1/r mod s,
        [r, s, alpha, beta])) ],

    toGammaTensor := self >> gammaTensor(
        OddStride(self.params[1], self.params[3]),
        OddStride(self.params[2], self.params[4])),

    def := (r, s, alpha, beta) -> Checked(
        IsPosIntSym(r), IsPosIntSym(s), AnySyms(r, s) or Gcd(r,s)=1,
        rec()),

    domain := self >> self.params[1] * self.params[2],
    range  := self >> self.params[1] * self.params[2],

    lambda := self >> let(
        r := self.params[1], s := self.params[2], N := r*s,
        alpha := self.params[3], beta := self.params[4],
        aa := (1/s/alpha) mod r, bb := (1/r/beta) mod s,
        i := Ind(N),
        When(not self.transposed,
            Lambda(i, ((s*alpha*idiv(i, s)) + (r*beta*imod(i, s))) mod N),
            Lambda(i,   s*(imod(i*aa, r))   + (imod(i*bb, s))))),
));

Class(fCond, FuncClass, rec(
    def := (cond, f1, f2) -> Checked(
        f1.range() = f2.range(),
        f1.domain() = f2.domain(),
	rec()),

    domain := self >> self.params[1].domain(),
    range := self >> self.params[2].range(),

    lambda := self >> let(f1 := self.params[2], f2:= self.params[3], mycond := self.params[1],
        i := Ind(f1.domain()),
        Lambda(i, cond(mycond.lambda().at(i), f1.lambda().at(i), f2.lambda().at(i)))),

    transpose := self >> let(base := self.__bases__[1],
        base(self.params[1], self.params[2].transpose(), self.params[3].transpose()))
));

# ----------------------------------------------------------
# Bit and Digit reversals
# ----------------------------------------------------------

# Return x as a b-bit value, formed as a vector
_numToBits := (k, b) >> Reversed(List([1..b], i-> imod(idiv(k, 2^(i-1)), 2)));

# Return number corresponding to bit vector b
_bitsToNum := b -> let(
   l := Reversed(b),
   Sum([1..Length(l)], i -> 2^(i-1) * l[i]));

# Calculate base-r digit reversed value of k, where k is between 0 and n-1
_digitRev := function(k, n, r)
   local bitval, numbits, bitsperchunk, numchunks, res_bits, i, s; 

   numbits      := Log2Int(n);
   bitsperchunk := Log2Int(r);
   numchunks    := numbits / bitsperchunk;

   bitval := _numToBits(k, numbits);
   
   res_bits := [];
   for i in Reversed([1..numchunks]) do
       s := bitval{([bitsperchunk*i-(bitsperchunk-1) .. bitsperchunk * i])};
       Append(res_bits, s);
   od;

   return _bitsToNum(res_bits);
end;

#F DR(<k, r>) - digit reversal permutation R^k_r
##
Class(DR, PermClass, rec(
    def := (k, r) -> Checked(IsInt(k), Is2Power(k), (r^(LogInt(k,r)) = k), rec()),

    domain := self >> self.params[1],
    range  := self >> self.params[1],
 # This is very slow because of how Gap evaluates functions
 #    lambda := self >> let(
 #         n := self.params[1],  base := self.params[2],  t := LogInt(n, base),
 #         rev := List([1..t], i-> fTensor(fId(base^(t-i)), L(base^i, base))),
 #         fCompose(Reversed(rev)).lambda()),

# This is faster but a little bit uglier.
    lambda := self >> let(
        i := Ind(self.params[1]),
        Lambda(i, _digitRev(i, self.params[1], self.params[2]))),

    transpose := self >> self
));

#F BR(<k>) - bit reversal permutation
##
Class(BR, PermClass, rec(
    def := k -> Checked(IsInt(k), Is2Power(k),  rec()), 
    domain := self >> self.params[1],
    range  := self >> self.params[1],
    lambda := self >> DR(self.params[1], 2).lambda(),
    transpose := self >> self
));
