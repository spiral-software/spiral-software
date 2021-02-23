
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Convention: N - range
#             n - domain

## H(<N>,<n>,<base>,<stride>) 
##    Stride index mapping: i -> base + stride * i
#
Class(H, FuncClass, rec(
    def := (N,n,base,stride) -> rec(),
    domain := self >> self.params[2],
    range := self >> self.params[1], 
    lambda := self >> let(
	base := self.params[3], str := self.params[4], i := Ind(self.params[2]),
	Lambda(i, base + str * i))
)); 

# HZ(<N>,<n>,<base>,<stride>) 
##    Mod-stride index mapping: i -> (base + stride * i) mod N
##
Class(HZ, FuncClass, rec(
    isCyclic := true,

    def := (N,n,base,str) -> rec(), 
    domain := self >> self.params[2],
    range := self >> self.params[1], 

    lambda := self >> let(
	n := self.params[2], N := self.params[1], base := self.params[3], str := self.params[4],
	i := Ind(n), 
	ii := When(str * n = N, no_mod(i), i), # condition to omit the mod is str*n = N
	Lambda(i, imod(base + str * ii, N)))
)); 

#F RM(<N>,<n>,<phi>,<g>) 
##    Exponential index mapping: i -> (phi * g^i) mod N
##    <g> - generating element for integers mod <N>.
##    <phi> - "phase"
##
Class(RM, FuncClass, rec(
    isCyclic := true,
    abbrevs := [ N    -> [N, N-1, 1, PrimitiveRootMod(N)],
	        (N,n) -> [N, n,   1, PrimitiveRootMod(N)] ],

    domain := self >> self.params[2],
    range := self >> self.params[1], 
    def := (N,n,phi,g) -> Checked(IsPosIntSym(N), IsPosIntSym(n), n <= N, rec()),

    lambda := self >> let(
	phi := self.params[3], g := self.params[4], i := Ind(self.params[2]),
	Lambda(i, powmod(phi, g, i, self.params[1]))),
));

#F RR(<N>,<phi>,<g>) 
##    Rader permutation: i -> {    0,     if i = 0 }
##                            { RM(i-1),  if i > 0 }
##    <phi> and <g> are parameters to RM
##
Class(RR, PermClass, rec(
    isCyclic := true,

    abbrevs := [ N -> [N,1,PrimitiveRootMod(N)] ],

    def := (N,phi,g) -> Checked(IsPosInt(N), IsPrime(N), rec()),

    domain := self >> self.params[1],
    range := self >> self.params[1], 

    # alternatively cond(i, imod(phi * pow(g, i-1), N), 0) we do g^-1 for easier poweropt
    lambda  := self >> let(
	N := self.params[1], phi := self.params[2], g := self.params[3], i:= Ind(N),
	When(not self.transposed,
	    Lambda(i, cond(i, powmod(phi*(g^-1 mod N) mod N, g, i, N), 0)),
	    Lambda(i, cond(i, 1+ilogmod(cond(i,i,1)/phi mod N, g, N), 0)))
    )
));

