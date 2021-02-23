
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


##################################
# I currently have a streaming algorithm (from Nikara et al.) for two variations of DCT-2.
#
# First, we have Spiral's standard DCT-2.  This is called TDCT2, defined: 
#    [ cos(k*(l+1/2)*pi/n) | k,l = 0...n-1 ] 
#
# Then, I have added a scaled version, defined:
#    [ Diag(1/sqrt(2), 1, 1, ...  ] * [ cos(k*(l+1/2)*pi/n) | k,l = 0...n-1 ]
# which matches the definition in the Nikara paper.

# We use essentially the same algorithm for both; the only change is in one diagonal.

# Using opts := InitStreamUnrollHw(); will enable these rules.

# Other DCT-2 definitions also have a sqrt(2/n) scaling factor in front of the whole matrix.

###########################

Declare(Sc_DCT2_func);



#F Sc_DCT2(<n>) - Scaled Discrete Cosine Transform, Type II, non-terminal
#F Definition: (n x n)-matrix [ Diag(1/sqrt(2), 1, 1, ...  ] * [ cos(k*(l+1/2)*pi/n) | k,l = 0...n-1 ]
#F Example:    DCT2(8)
Class(Sc_DCT2, TaggedNonTerminal, rec(

    abbrevs := [
    (n)       -> Checked(IsPosIntSym(n),
        [_unwrap(n)]),
    ],

    hashAs := self >> ObjId(self)(self.params[1], 1).withTags(self.getTags()),

    dims := self >> [ self.params[1], self.params[1] ],

    isReal := self >> true,

    terminate := self >> Mat(Sc_DCT2_func(self.params[1])),
    transpose := self >> Mat(TransposedMat(Sc_DCT2_func(self.params[1]))),
));




# hadamard_func(n,i): n-point Hadamard permutation, position i. 
# Defined as in:
#    Z. Wang, Pruning the fast discrete Cosine transform, IEEE 
#    Tr. Communications 39(5), May 1991, 640-643.
hadamard_func := function(n,i)
   if ((n=1) and (i=0)) then return 0; fi;
   if (imod(n,2) <> 0) then return -1;  fi; # ERROR
   if (imod(i,2) = 0) then return hadamard_func(n/2, i/2); fi;

   return n-1-hadamard_func(n/2, (i-1)/2);

end;

# An Exp wrapper for hadamard_func.  Used so Spiral does not try to evaluate the expression 
# until Process_fPrecompute is called.

Class(hadamard_func_exp, Exp, rec(
    ev := self >> hadamard_func(self.args[1].ev(), self.args[2].ev())
));

# Had(n): Hadamard permutation on n points.
# See:
#    Z. Wang, Pruning the fast discrete Cosine transform, IEEE 
#    Tr. Communications 39(5), May 1991, 640-643.
Class(Had, PermClass, rec(
#    exportSymbol := self>>self.name,
#    exportParams := self>>self.params,
#    export := Sym.export,

    def := (n) -> Checked(
        IsPosIntSym(n),
        rec(size := n)),

    lambda := self >> let(
        n := self.params[1], i := Ind(n),
        lt := List([0..n-1], it->hadamard_func(n,it)),
        FList(TInt, lt).lambda()
    ),

    transpose := self >> Error("Transpoed Hadamard permutation not currently supported."),

    # Only symmetric for size 2.
    isSymmetric := self >> (self.params[1] = 2)
));


# Don't think I need this anymore.
# tSPL Hadamard permutation on n points
# See:
#    Z. Wang, Pruning the fast discrete Cosine transform, IEEE 
#    Tr. Communications 39(5), May 1991, 640-643.
Class(THad, Tagged_tSPL_Container, rec(
    abbrevs :=  [ size -> [size] ],

    dims := self >> Replicate(2, self.params[1]),

    terminate := self >> Had(self.params[1]),
    transpose := self >> Error("Transposed Hadamard permutation not currently supported."),
    isReal := self >> true,
    isSymmetric := self >> self.params[1] = 2,

    # We can verify this with:
    # PermMatrixToBits(MatSPL(Had(n)))
    permBits := self >> let(n:=self.params[1], logn:=Log2Int(n),
        (MatSPL(DirectSum(J(logn-1), O(1,1))) + MatSPL(J(logn))) * 
        GF(2).one
    )

));


########################################
## The following are helper functions used in the streaming DCT2
## diagonal given in
##    Nikara et al., Discrete cosine and sine transforms--regular 
##    algorithms and pipeline architectures, Signal Processing 86, 
##    2006.

mu := (s,i) -> Cond(imod(i,2^s)=0, 0, 1);
tau := (i,s) -> Cond(s=i, 0, 1);

Class(mu_exp, Exp, rec(
   ev := self >> mu(self.args[1].ev(), self.args[2].ev())
));

Class(tau_exp, Exp, rec(
   ev := self >> tau(self.args[1].ev(), self.args[2].ev())
));

dct_diag_f := (k,i,s) -> (imod(i,2) + (1-tau_exp(0,i)) * (1-tau_exp(k-1,s)));


#dct_diag_d := i -> let(K := 2^(Log2Int(i)), t := i-K,
#    h := hadamard_func(K, t),
#    cospi((h+1/2)/(2*K)));

dct_diag_d := i -> let(K := 2^(floor(fdiv(log(i), log(2)))), t := i-K,
    h := hadamard_func_exp(K, t),
    cospi(fdiv((h+1/2), (2*K))));

dct_sc_diag_g := (k,i,s) -> ((2^(mu_exp(s, floor(fdiv(i,2))))) * 
    (dct_diag_d(2^(k-s-1) + floor(fdiv(i, (2^(s+1)))))))^dct_diag_f(k,i,s);

dct_unsc_diag_g := (k,i,s) -> ((2^(mu_exp(s, floor(fdiv(i,2))))) * 
    (dct_diag_d(2^(k-s-1) + floor(fdiv(i, (2^(s+1)))))))^imod(i,2);

## This represents the diagonal matrix used in the streaming
## DCT2 algorithm. The problem is of size 2^k, and s represents 
## the iteration
#Str_DCT2_Diag := (k, s) -> Diag(List([0..((2^k)-1)], 
#    i-> dct_diag_g(k, i, s)));

Class(Str_Sc_DCT2_Diag, DiagFunc, rec(
    abbrevs := [(k, s) -> [k, s]],
    def := (k, s) -> rec(size := 2^k),
    lambda := self >> let(k := self.params[1], s := self.params[2], i := Ind(2^k), Lambda(i, dct_sc_diag_g(k, i, s))),
    range := self >> TReal,
));

Class(Str_DCT2_Diag, DiagFunc, rec(
    abbrevs := [(k, s) -> [k, s]],
    def := (k, s) -> rec(size := 2^k),
    lambda := self >> let(k := self.params[1], s := self.params[2], i := Ind(2^k), Lambda(i, dct_unsc_diag_g(k, i, s))),
    range := self >> TReal,
));

Class(Str_DCT2_Perm, PermClass, rec(
    def := (k) -> Checked(
        IsPosIntSym(k),
        rec(size := 2^k)),

    # Sigh, this is broken it seems.
    lambda := self >> let(
        k := self.params[1],
	fCompose(Reversed(List([0..(k-2)], i-> let(rsize := 2^k - 2^(k-i),

            fCompose(
		fTensor(fId(2^i), L(2^(k-i), 2^(k-i-1))),
		fDirsum(fId(2^(k-i)), fTensor(fId(rsize/4), fDirsum(fId(2), J(2))))		
	    )

	)))).lambda()),

    transpose := self >> self,
    isSymmetric := self >> true,

));


Class(Str_DCT2_Perm_tspl, TaggedNonTerminal, rec(

    abbrevs := [(k)      -> [k]],

    dims := self >> [ 2^self.params[1], 2^self.params[1] ],

    terminate := self >> let(k := self.params[1],
	    Compose(List([0..(k-2)], i-> let(rsize := 2^k - 2^(k-i),
            DirectSum(I(2^(k-i)), 
                Tensor(I(rsize/4), DirectSum(I(2), J(2)))
            ) * 
            Tensor(I(2^i), L(2^(k-i), 2^(k-i-1)))
	    )
        ))        
    ),

    isReal := self >> true,

    print := meth(self, indent, indentStep)
        local lparams, mparams;
        if not IsBound(self.params) then Print(self.name); return; fi;
        Print(self.name, "(");
        if IsList(self.params) then
            lparams := Filtered(self.params, i->not (IsList(i) and i=[]));
            mparams := Filtered(lparams, i->not (IsBool(i) and not i));
            DoForAllButLast(mparams, x -> Print(x, ", "));
            Print(Last(mparams));
        else
            Print(self.params);
        fi;
    Print(")", When(self.transposed, ".transpose()", ""));
    end,

));

Str_DCT2_M := (n,s) -> let(l := Ind(n/2), TTensorInd(COND(eq(imod(l, 2^s), 0), I(2), Mat([[1,0],[-1,1]])), l, APar, APar));

Str_DCT2_H := (n,s) -> let(k := Log2Int(n), l := Ind(n/4), Cond(s=0, I(n), TTensorInd(COND(eq(imod(l, 2^(s-1)),0), I(4), Mat([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])), l, APar, APar)));

#Sc_DCT2_func := (N) -> List([0..N-1], m-> List([0..N-1], n-> ((Sqrt(2/N) * Cond(m=0, 1/Sqrt(2), 1) * CosPi(m*(n+1/2)/N)))));
Sc_DCT2_func := (N) -> List([0..N-1], m-> List([0..N-1], n-> (Cond(m=0, 1/Sqrt(2), 1) * CosPi(m*(n+1/2)/N))));

NewRulesFor(Sc_DCT2, rec(
    Sc_DCT2_Stream := rec(
        forTransposition := false,
        applicable := (self, nt) >> nt.params[1] > 4 and nt.isTag(1, AStream) and nt.firstTag().bs >= 4,
        children := (self, t) >> let(
            n := t.params[1],
            k := Log2Int(n),
            [[ TCompose(Concatenation(
                  [TPrm(Str_DCT2_Perm(k))],
                  Reversed(List([1..k-1], s-> TCompose([Str_DCT2_M(n, s), TDiag(fPrecompute(Str_Sc_DCT2_Diag(k, s))), Str_DCT2_H(n, s), TTensorI(F(2), n/2, APar, APar), TTensorI(TPrm(L(2^(s+1), 2^s)), 2^(k-s-1), APar, APar),
#TL(2^(s+1), 2^s, 2^(k-s-1), 1)
]))),
                  [TDiag(fPrecompute(Str_Sc_DCT2_Diag(k, 0))), TTensorI(F(2), n/2, APar, APar), TPrm(Had(n))])).withTags(t.getTags())
            ]]
         ),

        apply := (t,c,nt) -> c[1],
    )
));

NewRulesFor(TDCT2, rec(
    DCT2_Stream := rec(
        forTransposition := false,
        applicable := (self, nt) >> nt.params[1] > 4 and nt.isTag(1, AStream) and nt.firstTag().bs >= 4,
        children := (self, t) >> let(
            n := t.params[1],
            k := Log2Int(n),
            [[ TCompose(Concatenation(
                  [Cond(t.firstTag().bs = t.params[1],
			  TPrm(Str_DCT2_Perm(k)),
			  TPrm(Str_DCT2_Perm_tspl(k))
		      )
		      ],
                  Reversed(List([1..k-1], s-> TCompose([Str_DCT2_M(n, s), TDiag(fPrecompute(Str_DCT2_Diag(k, s))), Str_DCT2_H(n, s), TTensorI(F(2), n/2, APar, APar), TTensorI(TPrm(L(2^(s+1), 2^s)), 2^(k-s-1), APar, APar),
]))),
                  [TDiag(fPrecompute(Str_DCT2_Diag(k, 0))), TTensorI(F(2), n/2, APar, APar), TPrm(Had(n))])).withTags(t.getTags())
            ]]
         ),

        apply := (t,c,nt) -> c[1],
    )
));

