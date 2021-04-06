
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_ToDec := v -> let(d:=Length(v), Sum(List([1..d], e -> v[e]*2^(d-e))));
_ToBin := (i,d) -> List([1..d], e -> BinAnd(QuoInt(i, 2^(d-e)), 1));

_TB := (i, n) -> GF(2).one * _ToBin(i, n);
_TD := (i) -> _ToDec(IntVecFFE(i));

PrintHex := function(lst, del)
    local i;
    if Length(lst) = 0 then
        return;
    fi;
    Print(HexInt(lst[1]));
    for i in [2..Length(lst)] do
        Print(del, HexInt(lst[i]));
    od;
end;

Declare(B);
Class(B, PermClass, rec(
    def := (n,l) -> Checked(
        # make sure the 1s are inside the matrix.
        ForAll(l, e ->
            IsPosInt(e[1]) and IsPosInt(e[2])
            and e[1] <= Log2Int(n) and e[2] <= Log2Int(n)),
        # make sure 'l' is a list.
        IsList(l),
	# done
	rec()),

    domain := self >> self.params[1],
    range  := self >> self.params[1],

    # return the bit matrix
    bm := meth(self)
        local n, m, i;

        n := Log2Int(self.params[1]);

        # generate matrix of zeros.
        m := List([1..n], e -> List([1..n], ee -> 0));

        # fill in identity
        for i in [1..n] do
            m[i][i] := 1;
        od;

        # xor in individual 1s from 'l'
        for i in self.params[2] do
            m[i[1]][i[2]] := BinXor(m[i[1]][i[2]], 1);
        od;

        return m * GF(2).root;
    end,

    # convert the sparse bit matrix info into a permutation matrix
    toAMat := meth(self)
        local m, n, i, M;

        m := self.bm();
        n := self.params[1];

        # generate full permutation matrix
        M := [];
        for i in [0..(n-1)] do
            Add(M, BasisVec(
                n,
                _TD(m * _TB(i, Log2Int(n)))
            ));
        od;

        return AMatMat(M);
    end,

    transpose := (self) >> self(
        self.params[1],
        self._mklist(
            MatAMat(
                InverseAMat(
                    AMatMat(self.bm())
                )
            )
        )
    ),

    _mklist := function(b)
        local l, i, j;

        Constraint(Dimensions(b)[1] = Dimensions(b)[2]);

        b := b - MatSPL(I(Dimensions(b)[1])) * GF(2).root;

        l := [];
        for i in [1..Length(b)] do
            for j in [1..Length(b)] do
                if b[i][j] = GF(2).root then
                    Add(l, [i,j]);
                fi;
            od;
        od;

        return l;
    end,

    transpose := self >> B(self.params[1], List(self.params[2], e -> [e[2],e[1]])),

    sums := (self) >> Chain(Error("Don't use legacy sumsgen!!"), true),
));

Declare(B2);

Class(B2, PermClass, rec(
    def := (N, E, S) -> Checked(
            Is2Power(N),
            Is2Power(E),
            Is2Power(S),
            E*(S^2) >= N,
            rec( transposed := false)
    ),

    domain := self >> self.params[1],
    range  := self >> self.params[1],

    transpose := self >> CopyFields(
        ApplyFunc(B2, self.params), rec(transposed := not self.transposed)),

    getES := (self) >> Product(self.params{[2..3]}),

    bm := meth(self)
        local n, nes, b, i;

        n := Log2Int(self.params[1]);
        nes := n - Log2Int(self.params[2] * self.params[3]);

        b := MatSPL(I(n)) * GF(2).root;

        if self.transposed then
            for i in [1..nes] do
                b[i+nes][i] := GF(2).root;
            od;
        else
            for i in [1..nes] do
                b[i][i+nes] := GF(2).root;
            od;
        fi;

        return b * GF(2).root;
    end,

    toAMat := meth(self)
        local b, n, i, M;

        b := self.bm();
        n := self.params[1];

        # generate full permutation matrix
        M := [];
        for i in [0..(n-1)] do
            Add(M, BasisVec(
                n,
                _TD(_TB(i, Log2Int(n)) * b)
            ));
        od;

        return AMatMat(M);
    end,
));

Declare(BP);

BPI := (n) -> BP(MatSPL(I(Log2Int(n)))*GF(2).root);
BPL := (n,str) -> BP(MatSPL(Z(Log2Int(n), Log2Int(str)))*GF(2).root);

Class(BP, PermClass, rec(
    # p must be a full rank 2-d square matrix of size no larger than 29x29.
    def := function(arg)
        local n, m;

        n := Length(arg);

        if n = 1 and IsMat(arg[1]) then
            m := arg[1];
            n := Dimensions(m)[1];
        else if n > 0 and IsInt(arg[1]) then

            Constraint(ForAny(arg, e -> e < 2^n));
            m := List(arg, e -> _TB(e, n));
        fi; fi;

        # ints in SPIRAL are 29bits long, the upper bits are used for flags
        Constraint(IsInvertibleMat(AMatMat(m)) and Dimensions(m)[1] < 30);

        # NOTE: this is buggy, because fields below are not exposed through .rChildren
        return rec(
            size := 2^Dimensions(m)[1],
            mat:= m,
            fast := List(m, e -> _TD(e)),
            operations := rec(
                Print := (self) >> Print(self.name, "(", PrintCS(self.fast), ")")
            )
        );
    end,

    # NOTE: this is buggy, because .size is not exposed through .rChildren
    domain := self >> self.size,
    range  := self >> self.size,

#    this method converts 'i' to an array of bits, multiplies by the matrix, and converts back
#
#    lambda := self >> let(
#       d := Dimensions(self.params[1])[1],
#       i := Ind(self.size),
#       Lambda(i, todec(self.params[1] * tobin(i,d)))),

#   alternative method: ands a row of the matrix with the input and uses a parity
#   check to see if the bit is set in the output. shifts the output to the proper place. repeat
    lambda := self >> let(
        f := Length(self.fast),
        i := Ind(self.size),
        Lambda(i, Sum(List([1..f], e -> bin_parity(bin_and(self.fast[e], i))*2^(f-e))))),

    transpose := self >> self.__bases__[1](TransposedMat(self.mat))
));

## -----------------------------------------------------------------------------
# CL(N, STR, [E,S,A,R]) - counters a stride L(N, STR) for cache given by [E,S,A,R]
#
# this returns a compound object, namely, P * B where
#
# P = I(N/(E*S)) x L(E*S, Ceil(E*S/STR))
#
# and B is a permutation specified by a function linear on the bits
# which forces the P*B object to span the sets in the cache after L.
#

CL := function(N, STR, CS)
    local E,S,A,Q,P,l,k,i,j, n;

    E := CS[1];
    S := CS[2];

    # deal with degenerate cases and check params.

    # if cache is larger/equal to transform size, no
    # fancy footwork is necessary.
    if E * S >= N then
        return I(N);
    fi;

    # the cache must be large enough, otherwise this method
    # will not build an adequate perm to counteract the stride.
    if S^2 * E < N then
        Error("CL failure: Cache inadequately sized.");
    fi;

    # build the P portion as a bit matrix
    P := MatSPL(DirectSum(
        I(Log2Int(N/(E*S))),
        Z(Log2Int(E*S), Log2Int(E*S) - Log2Int(Maximum(E*S/ Maximum(2*S/E, STR),1)))
#        Z(Log2Int(E*S), Log2Int(E*S) - Log2Int(Maximum(E*S/STR,1)))
    )) * GF(2).root;

    # build the list of 1's in the B matrix.
    l := [];
    k := Log2Int(Minimum(STR, (N/(E*S))));
    i := Log2Int(N/(E*S)) + 1;
    n := Log2Int(N);

    # paranoia check
    if i-k < 1 then
        Error("Problem!!");
    fi;

    A := Copy(P);

    for j in [0..i-2] do
        A[j+1][j+i] := GF(2).root;
    od;

    Q := (TransposedMat(P) * A) - (MatSPL(I(n))*GF(2).root);

    for j in [1..n] do
        for i in [1..n] do
            if Q[j][i] = GF(2).root then
                Add(l, [j,i]);
            fi;
        od;
    od;

    # rebuild P as a normal perm matrix that corresponds to the
    # bit matrix from earlier.
    j := Maximum(E*S/ Maximum(2*S/E, STR),1);

    P := When(E*S = j or j = 1,
        I(N),
        Tensor(
            I(N/(E*S)),
            L(E*S, j)
        )
    );

    # combine the two and return
    # temp -- remove the B matrix, to remove the xor/and/shifts
    # this means we no longer guarantee spanning of the elements/sets
    # but the expressions should reduce better.
    return Compose(P,B(N, l));
end;



#F fB describes a permutation which can be expressed as a sparse linear
#F     function on the bits.
#F
#F I_2^n -> I_2^N
#F
#F params: 
#F  n (input size)
#F  N (output size)
#F  l (list of location of 1s in the bit matrix)
#F
#F additional details:
#F   l defines a list of pairs. each pair describes the position
#F   of a 1 in bit matrix of size Nxn. The output mapping is described
#F   by (I + BM)(index), where BM is the sparse bit matrix generated by the
#F   pairs. the pairs themselves are in SPIRAL parlance, that is, [ROW, COL] 
#F   where the first row/col is index 1.
#F
#F NOTE:
#F   in order for composition to work like matrix multiplication, the index
#F   is now applied from the LEFT (eg: (index)(I + BM)) instead of from the
#F   right (eg: (I + BM)(index)). 
#F
#F example: fB(8,8, [[1,2],[2,3]])
#F
#F [ 1 1 0
#F   0 1 1
#F   0 0 1 ]
#F
#F
#F
Class(fB, FuncClass, rec(
    # full input definition.
    def := (n,N,l) -> rec(), 
    range := self >> self.params[2],
    domain := self >> self.params[1],

    # abbreviated input forms, from most complex to simplest.
    abbrevs := [
        (n, N, l) -> Checked(Is2Power(n), Is2Power(N), IsList(l),
            [n,N,l]),
        (n, l) -> Checked(Is2Power(n), IsList(l),
            [n,n,l]),
        (n) -> Checked(Is2Power(n),
            [n,n,[]])
    ],

    transpose := (self) >> self(
        self.params[2], 
        self.params[1], 
        self._mklist(
            MatAMat(
                InverseAMat(
                    AMatMat(self.bm())
                )
            )
        )
    ),

    # LAMBDA with index on the LEFT: (index)(I + BM)
    lambdaLEFT := self >> let(
        i := Ind(self.params[1]),
        Lambda(i, 
            When(Length(self.params[3]) = 0,
                i,
                ApplyFunc(
                    xor, 
                    Concat([i], List(self.params[3], e -> let(
                        a := Log2Int(self.params[1]) - e[1],
                        b := Log2Int(self.params[2]) - e[2],
                        When(b = a,
                            bin_and(i, V(2^a)),
                            When(b > a,
                                shl(bin_and(i, V(2^a)), b - a),
                                shr(bin_and(i, V(2^a)), a - b)
                            )
                        )
                    )))
                )
            )
        )
    ),
    
    # LAMBDA with index on the RIGHT: (I + BM)(index)
    lambdaRIGHT := self >> let(
        i := Ind(self.params[1]),
        Lambda(i, 
            When(Length(self.params[3]) = 0,
                i,
                ApplyFunc(
                    xor, 
                    Concat([i], List(self.params[3], e -> let(
                        a := Log2Int(self.params[1]) - e[2],
                        b := Log2Int(self.params[2]) - e[1],
                        When(b = a,
                            bin_and(i, V(2^a)),
                            When(b > a,
                                shl(bin_and(i, V(2^a)), b - a),
                                shr(bin_and(i, V(2^a)), a - b)
                            )
                        )
                    )))
                )
            )
        )
    ),

    lambda := self >> self.lambdaLEFT(),

    # a bit more complicated than the B case since matrix is not necessarily
    # square.
    bm := meth(self)
        local n, N, m, minn, minN, j, i;
    
        n := Log2Int(self.params[1]);
        N := Log2Int(self.params[2]);

        # since we multiply the index from the right, i * BM,
        # and n -> N, the bit matrix is nxN. (n rows, N cols)

        # generate matrix of zeros.
        m := List([1..n], e -> List([1..N], ee -> 0));

        # fill in identity
        minn := When(N > n, 1, n - N + 1);
        minN := When(n > N, 1, N - n + 1);
        for i in [minn..n] do
            m[i][i-minn+minN] := 1;
        od;

        # xor in individual 1s from 'l'
        for i in self.params[3] do
            m[i[1]][i[2]] := BinXor(m[i[1]][i[2]], 1);
        od;

        return m * GF(2).root;
    end,

    # given a bit matrix, make a list.
    _mklist := function(b)
        local n,N, m, minn, minN, i, j, l;
        
        ## generate identity...
        #

        n := Dimensions(b)[1];
        N := Dimensions(b)[2];

        # generate matrix of zeros.
        m := List([1..n], e -> List([1..N], ee -> GF(2).root*0));

        # fill in identity
        minn := When(N > n, 1, n - N + 1);
        minN := When(n > N, 1, N - n + 1);
        for i in [minn..n] do
            m[i][i-minn+minN] := GF(2).root;
        od;

        ## subtract out identity.
        b := b - m;

        ## add a list entry for each nonzero elem in b.

        l := [];
        for j in [1..Length(b)] do #rows
            for i in [1..Length(b[1])] do # cols
                if b[j][i] = GF(2).root then
                    Add(l, [j,i]);
                fi;
            od;
        od;

        return l;
    end,
));


Class(fB2, fTensorBase, rec(
    __call__ := meth(arg)
        local self, children, lkup, res, h, es;
        self := arg[1];
        es := arg[2];
        children := Flat(Drop(arg, 2));
        if self.skipOneChild and Length(children)=1 then return children[1]; fi;

        h := self.hash;
        if h<>false then
            lkup := h.objLookup(self, children);
            if lkup[1]<>false then return lkup[1]; fi;
        fi;
        res := WithBases(self, rec(es := es, operations := RewritableObjectOps, _children := children));
        if h<>false then return h.objAdd(res, lkup[2]);
        else return res;
        fi;
    end,

    rightBinary  := self >> FoldR1(self._children, (p,x) -> let(base:=self.__bases__[1], base(x, p))),
    leftBinary := self >> FoldL1(self._children, (p,x) -> let(base:=self.__bases__[1], base(p, x))),

    # this works just like the normal fTensor except that
    # subject to some conditions, the variables responsible for the
    # top-end of the range get XORed with variables responsible
    # for the middle of the range.

    lambda := meth(self)
        local es;

        es := self.es;

        # flatten the fTensors first
        # ensure we always work with 2 children
        if self.numChildren() > 2 then
            return self.leftBinary().lambda();
        fi;

        # the expression is now guaranteed to be a left sided
        # binary tree. we need to tell each node/leaf how it is
        # to be split according to the params available to the fB2.

    end,

    combine_op := (self, jv, split, f, g) >> let(a := Error("fB2.lambda()!"), self),
    transpose := self >> self.__bases__[1](List(self.children(), c->c.transpose()))
));
