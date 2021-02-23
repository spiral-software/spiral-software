
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


NewRulesFor(DFT, rec(

    DFT_PD := rec(
        forTransposition := false,
        maxSize          := 13,
    
        core := meth(self, N, k, root, is_real)
            local coeffs, l1, l2, n, circ, toep;
            coeffs := List([0..N-2], x->E(N)^(k*root^(-x) mod N));
            n := N-1;
            l1 := coeffs{[1..n/2]};
            l2 := coeffs{[n/2+1..n]};
            circ := 1/2 * (l1 + l2);
            toep := Reversed(1/2 * Concat(Drop(l2 - l1, 1), l1 - l2));
            if is_real then toep := toep / E(4); fi;
            return
            DirectSum(I(1),
                  transforms.filtering.Circulant(circ).terminate(),
                  transforms.filtering.Toeplitz(toep).terminate());
        end,
    
        A := N -> let(nn := (N-1)/2, ones := m -> fConst(m, 1),
             VStack(HStack(RowVec(ones(nn+1)), O(1, nn)),
               HStack(VStack(-2*ColVec(ones(nn)), O(nn, 1)), I(2*nn)))),
    
        applicable     := (self, nt) >> nt.params[1] > 2 and nt.params[1] <= self.maxSize and IsPrime(nt.params[1]) and not nt.hasTags(),
    
        apply := (self,nt,C,cnt) >> let(N := nt.params[1], k := nt.params[2], root := PrimitiveRootMod(N),
            Scat(RR(N, 1, root)) *
            DirectSum(I(1), Tensor(F(2), I((N-1)/2))) *
            Mat(MatSPL(
                self.core(N, k, root, false) *
                self.A(N))) *
            DirectSum(I(1), Tensor(F(2), I((N-1)/2))) *
    #        DirectSum(I(1), OS(N-1, -1)) *
            Gath(RR(N, 1, 1/root mod N))
        )
#D    isApplicable     := (self, P) >> P[1] > 2 and P[1] <= self.maxSize and IsPrime(P[1]) and P[3] = [],
#D
#D    rule := (self,P,C) >> let(N := P[1], k := P[2], root := PrimitiveRootMod(N),
#D        Scat(RR(N, 1, root)) *
#D        DirectSum(I(1), Tensor(F(2), I((N-1)/2))) *
#D        Mat(MatSPL(
#D            self.core(N, k, root, false) *
#D            self.A(N))) *
#D        DirectSum(I(1), Tensor(F(2), I((N-1)/2))) *
#D#        DirectSum(I(1), OS(N-1, -1)) *
#D        Gath(RR(N, 1, 1/root mod N))
#D    )
    )
));
