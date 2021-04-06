
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F WHT( <log2(size)> ) - Walsh Hadamard Transform non-terminal
#F Definition: (2^n x 2^n)-matrix  DFT_2 tensor ... tensor DFT_2 (n-fold)
#F Note:       DFT_2 denotes the matrix [[1, 1], [1, -1]],
#F             WHT is symmetric.
#F Example:    WHT(3)
#F
#F Tweakable parameters: WHT_GeneralSplit.maxSize,
#F                       WHT_BinSplit.minSize
#F                       WHT_DDL.minSize
#F
#F WHT_GeneralSplit is a generalization of BinSplit, so it only makes
#F sense to have WHT_BinSplit.minSize := WHT_GeneralSplit.maxSize + 1
#F (not to have 2 rules enabled at the same time)
#F
#F By default, WHT_GeneralSplit is disabled.
#
Class(WHT, TaggedNonTerminal, rec(
  abbrevs   := [ n -> Checked(IsPosInt(n), [n]) ],
  dims      := self >> let(size := 2^self.params[1], [size, size]),
  terminate := self >> When(self.params[1] = 1, F(2), Tensor(Replicate(self.params[1], F(2)))),
  transpose := self >> Copy(self),
  isReal    := self >> true,
  SmallRandom := () -> Random([2..5]),
  LargeRandom := () -> Random([6..15]),
  normalizedArithCost := self >> self.params[1]*2^self.params[1],
  TType := T_Real(64)
));

# NOTE: A lot of WHT rules were moved to paradigms/common/wht.g, UGLY!
#
NewRulesFor(WHT, rec(
    #F WHT_Base: WHT_(2^1) = F_2
    #F
    WHT_Base := rec(
        info             := "WHT_(2^1) -> F_2",
        forTransposition := false,
        applicable       := (self, nt) >> nt.params[1]=1 and not nt.hasTags(),
        apply            := (nt, c, cnt) -> F(2),
    ),

    #F WHT_Base2: WHT_(2^1) = F_2
    # Added by Kyle 7-11-07
    # Need a better way to represent base cases for WHT so it is not fully expanded.
    WHT_Base2 := rec(
        info             := "WHT_(2^2) -> F_2",
        forTransposition := false,
        switch           := false,
        applicable       := (self, nt) >> nt.params[1]=2 and not nt.hasTags(),
        apply            := (nt, c, cnt) -> WHT(2),

    ),

    #F WHT_GeneralSplit: follows from definition
    #F
    #F   WHT_(2^k) =
    #F                        WHT_(2^k1) tensor I_(2^(k-k1)) *
    #F     I_(2^k1)    tensor WHT_(2^k2) tensor I_(2^(k-k1-k2)) *
    #F       ...
    #F       ...
    #F     I_(2^(k-kr)) tensor WHT_(2^kr)
    #F
    #F This rule has been restricted to small sizes
    #F
    WHT_GeneralSplit := rec (
        info             := "WHT_(2^k) -> (WHT_(2^k1) tensor I) .. (I tensor WHT_(2^kr))",
        forTransposition := false,
        switch           := false,
        maxSize          := 7,
        applicable       := (self, nt) >> nt.params[1] <> 1 and nt.params[1] <= self.maxSize,

        children := nt -> let(C := DropLast(OrderedPartitions(nt.params[1]), 1),
            List(C, p -> List(p, WHT))),

        # MRT 10-2007: .randomChildren does appear to be called in a certain case during RandomRuleTree()
        # but is it necessary? It only appears here!
        randomChildren := function ( Pin )
            local C, sum, rand, P;
            P := Pin[1];
            C := [];
            sum := 0;
            while sum < P do
                if sum = 0 then  rand := RandomList( [1..(P-1)] );
                else             rand := RandomList( [1..(P-sum)] );
                fi;
                Add( C, rand );
                sum := sum + rand;
            od;
            When(not (Sum(C)=P and Length(C) > 1),
                Error( "WHT_GeneralSplit.randomChildren has a bug"));
            return List(C, WHT);
        end,

        rule := function ( P, C )
            local P, left, right, i;
            right := 2^P / C[1].dimensions[1];
            left  := 1;
            P := [ Tensor(C[1], I(right)) ];

            # list of factors
            for i in [2..Length(C)-1] do
                left  := left * C[i-1].dimensions[1];
                right := right/C[i].dimensions[1];
                Add(P, Tensor(I(left), C[i], I(right)));
            od;
            left := left * C[Length(C)-1].dimensions[1];
            Add(P, Tensor(I(left), C[Length(C)]));
        
            return Compose(P); # return product
        end
    ),

    WHT_BinSplit_binloops := rec (
        info             := "WHT_(2^k) -> (WHT_(2^k1) tensor I) (I tensor WHT_(2^k2))",
        forTransposition := false,
        minSize          := 2,
        applicable       := (self, nt) >> nt.params[1] >= self.minSize and not nt.hasTags(),

        children := nt -> List([1..nt.params[1]-1], i -> [ WHT(i), WHT(nt.params[1]-i) ] ),

        apply := function(nt, c, cnt)
            local r1, r2, a, i, b;

            r1 := Rows(c[1]);
            r2 := Rows(c[2]);

            a := c[1];
            for i in [1..Log2Int(r2)] do
                a := Grp(Tensor(I(2), a));
            od;
            a := Grp(L(r1 * r2, r1) * a * L(r1 * r2, r2));

            b := c[2];
            for i in [1..Log2Int(r1)] do
                b := Grp(Tensor(I(2), b));
            od;

            return a * b;
        end,

        switch := true,
    )
));
