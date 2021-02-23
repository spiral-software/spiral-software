
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# NOTE: Support odd sizes ie with (I_4 dirsum J_3)
condIJ := (N, m) -> let(k := N/m, Checked(IsSymbolic(k) or IsEvenInt(k), 
    fCond(diagTensor(II(k/2), II(2,0,1), II(m)), fTensor(fId(k), fId(m)), 
                                                 fTensor(fId(k), J(m))) 
));

# NOTE: Support odd sizes ie with (I_4 dirsum OS_3)
condIOS := (N, m) -> let(k := N/m, Checked(IsSymbolic(k) or IsEvenInt(k), 
    fCond(diagTensor(II(k/2), II(2,0,1), II(m)), fTensor(fId(k), fId(m)), 
                                                 fTensor(fId(k), OS(m, -1))) 
));

condM := (N, m) -> L(N, m) * Prm(condIJ(N, m));
condK := (N, m) -> Prm(condIJ(N, N/m)) * L(N, m);

condMp := (N, m) -> L(N, m) * Prm(condIOS(N, m));
condKp := (N, m) -> Prm(condIOS(N, N/m)) * L(N, m);

condB := (N, m) -> Checked(IsEvenInt(m), let(k := N/m, 
    condK(N, m) * Tensor(I(N/m), condM(m,m/2))));

condBp := (N, m) -> Checked(IsEvenInt(m), let(k := N/m, 
    condKp(N, m) * DirectSum(L(m,m/2), Tensor(I(k-1), condM(m,m/2)))));

condBpu := (N, m) -> Checked(IsEvenInt(m), let(k := N/m, 
    condKp(N, m) * DirectSum(I(m), Tensor(I(k-1), condM(m,m/2)))));

# condBpu(k*m,m) == Scat(Refl0_u(k,m)) = Refl0_u(k,m).transpose()
# u = stands for URDFT. i.e. this is for URDFT based algorithms
# NOTE: not valid for odd <k> (but will need new ASPF rule anyway with an extra child)
# GOOD:  valid for odd <m>.
Class(Refl0_u, PermClass, rec(
    def := (k, m) -> rec(), 
    domain := self >> self.params[1] * self.params[2], 
    range := self >> self.params[1] * self.params[2], 
    lambda := self >> let(k := self.params[1], m := self.params[2], i := Ind(k*m), 
        Lambda(i, cond(leq(i, m-1), k*i, 
                                    cond(leq(imod(i, m), idiv(m-1,2)), 2*k*imod(i,m) + idiv(i,m), 
                                                                      2*k*m - 2*k*imod(i,m) - idiv(i,m)))))
));

# Valid for odd <k>, even <m>
# <omit> is how many initial points we want to omit from the permutation
# the only sensible values for it are 0 and 1
Class(Refl0_odd, PermClass, rec(
    def := (k, m, omit) -> rec(), #size := k*m+idiv(m+1,2)-omit),
    range := self >> self.domain(),
    domain := self >> let(k:=self.params[1], m:=self.params[2], omit:=self.params[3],
        k*m + idiv(m+1, 2) - omit),

    lambda := self >> let(
        k := self.params[1],   m := self.params[2], mc := idiv(m+1, 2),
        N := self.domain(), omit := self.params[3], mf := idiv(m, 2),   
        N2:= 2*k*m+m,         ii := Ind(N-omit),    i  := ii+omit,
        
        Lambda(ii, cond(leq(i, mc-1), (2*k+1)*i, 
                                     cond(leq(imod(i-mc, m), mc-1),       (2*k+1)*imod(i-mc,m) + idiv(i+mf,m), 
                                                                    N2  - (2*k+1)*imod(i-mc,m) - idiv(i+mf,m)))
            -omit))
));

# condB(k*m,m) == Scat(Refl1(k,m)) = Refl1(k,m).transpose()
# NOTE: not valid for odd <k> (but will need new ASPF rule anyway with an extra child)
# GOOD:  valid for odd <m>.
Class(Refl1, PermClass, rec(
    def := (k, m) -> rec(), #size := k*m),
    domain := self >> self.params[1]*self.params[2], 
    range := self >> self.params[1]*self.params[2], 
    lambda := self >> let(k := self.params[1], m := self.params[2], i := Ind(k*m), 
        Lambda(i, cond(leq(imod(i, m), idiv(m-1,2)), 2*k*imod(i,m) + idiv(i,m), 
                                       2*k*m - 1 - 2*k*imod(i,m) - idiv(i,m))))
));

Declare(KK, MM, KKp, MMp);

Class(KK, PermClass, rec(
    def := (k, m) -> rec(), #size := k*m),
    domain := self >> self.params[1]*self.params[2], 
    range := self >> self.params[1]*self.params[2], 
    transpose := self >> MM(self.params[2], self.params[1]),
    lambda := self >> let(k := self.params[1], m := self.params[2],
        fCompose(Tr(k,m), condIJ(k*m, k)).lambda()),
));

Class(MM, PermClass, rec(
    def := (k, m) -> rec(), #size := k*m),
    domain := self >> self.params[1]*self.params[2], 
    range := self >> self.params[1]*self.params[2], 
    transpose := self >> KK(self.params[2], self.params[1]),
    lambda := self >> let(k := self.params[1], m := self.params[2],
        fCompose(condIJ(k*m, m), Tr(k, m)).lambda()),
));

Class(KKp, PermClass, rec(
    def := (k, m) -> rec(), #size := k*m),
    domain := self >> self.params[1]*self.params[2], 
    range := self >> self.params[1]*self.params[2], 
    transpose := self >> MMp(self.params[2], self.params[1]),
    lambda := self >> let(k := self.params[1], m := self.params[2],
        fCompose(Tr(k,m), condIOS(k*m, k)).lambda()),
));

Class(MMp, PermClass, rec(
    def := (k, m) -> rec(), #size := k*m),
    domain := self >> self.params[1]*self.params[2], 
    range := self >> self.params[1]*self.params[2], 
    transpose := self >> KKp(self.params[2], self.params[1]),
    lambda := self >> let(k := self.params[1], m := self.params[2],
        fCompose(condIOS(k*m, m), Tr(k, m)).lambda()),
));

Declare(IP);

libB := (k, m) -> Checked(IsSymbolic(m) or IsEvenInt(m), 
    KK(k, m) * Tensor(I(k), MM(2,m/2)));

libBp := (k, m) -> Checked(IsSymbolic(m) or IsEvenInt(m), 
    KKp(k, m) * DirectSum(Tr(2,m/2), Tensor(I(k-1), MM(2,m/2))));

libBpu := (k, m) -> Checked(IsSymbolic(m) or IsEvenInt(m), 
    KKp(k, m) * DirectSum(I(m), Tensor(I(k-1), MM(2,m/2))));

#F IJ(<size>, <n>) - I_n dirsum J_n dirsum I_n dirsum ... (repeats N/n times, n|N)
#F
#
# Subst(i -> BinXor(i, BinAnd(i,$n) - BinShr(BinAnd(i,$n), $(LogInt(n,2))))),
# Class(IJ, PermClass, rec(
#   def := (N, n) -> Checked(IsInt(N), IsInt(n), N mod n = 0,
#       rec(
#       size := N,
#       direct  := Subst(x -> x + (QuoInt(x,$n) mod 2) * ($(n-1) - 2*(x mod $n))),
#       inverse := Subst(x -> x + (QuoInt(x,$n) mod 2) * ($(n-1) - 2*(x mod $n)))
#       )),
#   transpose := self >> self
#);
IJ := (N, n) -> IP(N, J(n));

#F IP(<N>, <splperm>) - I_n dirsum P_n dirsum I_n dirsum ... (repeats N/n times, n|N)
#F
#F    P_n given by <splperm>, which is a permutation function
#F
Class(IP, PermClass, rec(
    def := (N, splperm) -> Checked(IsInt(N), IsSPL(splperm), N mod Rows(splperm) = 0, rec()), 
    range := self >> self.params[1],
    domain := self >> self.params[1],

    lambda := self >> let(
	N := self.params[1], 
	P := self.params[2],
	Pl := P.lambda(),
	n := P.domain(), 
	i := Ind(N),
	Lambda(i, n*idiv(i,n) + imod(1+idiv(i,n), 2)*imod(i, n) +
                                imod(  idiv(i,n), 2)*Pl.at(imod(i,n)))
    ),

    transpose := self >> IP(self.params[1], self.params[2].transpose())
));



# -----------------------------------------------------------------------------#
#F LIJ(<N>) - L(N, N/2) * (I(N/2) dirsum J(N/2))
##   works also for odd sizes
Class(LIJ, PermClass, rec(
    def := N -> Checked(IsInt(N), rec()),
    range := self >> self.params[1],
    domain := self >> self.params[1],
    lambda := self >> let(
	N := self.params[1],
	N1 := When(IsOddInt(N), N, N-1),
	i := Ind(N),
	When(not self.transposed,
            Lambda(i, QuoInt(N,2) + idiv(N-i, 2)*cond(neq(imod(N1-i,2),0), -1, 1)),
            Lambda(i, 2*i + idiv(i, QuoInt(N+1,2)) * (2*N - 1 - 4*i))
        ))
));


# -- Originals
# K ( <n>, <m> )
#   returns the equivalent of the stride permutation for Cooley-Tukey FFT
#   for the DCT3 and 4. <m> has to divide <n>
#     (I_n/m dirsum J_n/m dirsum I_n/m ....) * L^n_m  (direct sum alternately)
#   This permutation is to be used if it occurs on the left side in the rule.
#   It is K(n, m)^-1 =  M(n, n/m).
K := (mn, m) -> Checked(IsPosInt(mn), IsPosInt(m), mn mod m = 0, 
    IJ(mn, mn/m) * L(mn, m));

M := (mn, m) -> Checked(IsPosInt(mn), IsPosInt(m), mn mod m = 0, 
    L(mn, m) * IJ(mn, m));
# -------------


########################
# Generating functions
########################

L_or_OS := (n, str) -> When((n mod str) <> 0, OS(n, str), L(n, str));

Kp := (NN, m) -> let(n := NN / m,
    Prm(IP(NN, OS(n, -1)))* L(NN, m));

Mp := (NN, m) -> Kp(NN, NN/m).transpose(); 

# Type 0 symmetries  (ce0, co0) aka whole-point
# BH0(<Nreal>, <n>, <base>, <stride>), Nreal = number of output real pts = Rows(PRDFT(.))
# BH0 := (Nreal,n,base,stride) -> BH(Int(Nreal/2)+1, Nreal, n, base, stride);

# Type 1 symmetries  (ce1, co1) aka half-point
# BH1(<Nreal>, <n>, <base>, <stride>), Nreal = number of output real pts = Rows(PRDFT3(.))
# BH1 := (Nreal,n,base,stride) -> BH(Int((Nreal+1)/2), Nreal-1, n, base, stride);


# BH(<N>, <reflect>, <n>, <b>, <s>) - Reflective stride function
#   Defined as  i -> When(i < ceil(n/2)-1, si + b, reflect - si - b)
# 
Class(BH, FuncClass, rec(
    def := (N,reflect,n,base,stride) -> rec(), #N := N, n := n),
    domain := self >> self.params[3],
    range  := self >> self.params[1],
    lambda := self >> let(
	N:=self.params[1], reflect:=self.params[2], n:=self.params[3], b:=self.params[4], 
	s:=self.params[5], i:=Ind(n),
	Lambda(i, cond(leq(i, idiv(n-1, 2)), s*i + b, reflect - s*i - b)))
));

Declare(Refl);
# Refl(<N>, <reflect>, <n>, <func>) - Reflective arbitrary function
#   Defined as  i -> When(func(i) < N, func(i), reflect - func(i))
#   with domain of <n> points (usually smaller than domain(func))
# 
Class(Refl, FuncClass, rec(
    def := (N,reflect,n,func) -> rec(), #N := N, n := n),
    domain := self >> self.params[3],
    range  := self >> self.params[1],
    transpose := self >> Refl(self.params[1], self.params[2], self.params[3], self.params[4].transpose()),
    lambda := self >> let(
	func:=self.params[4].lambda(), reflect:=self.params[2], N:=self.params[1], 
	i := Ind(self.params[3]),
	Lambda(i, cond(leq(func.at(i), N-1), func.at(i), reflect-func.at(i))))
));
