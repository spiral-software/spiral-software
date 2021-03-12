
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


BlocksInt := (n, min, max) ->
    List(Filtered([0..Log2Int(max)+1], i -> min <= (2^i) and (2^i) <= max and (2^i) < n), i->2^i);


# ---------
Class(divisorsIntNonTriv, AutoFoldExp, rec(
    ev := self >> DivisorsIntNonTriv(self.args[1].ev()),
    computeType := self >> TList(TInt)
)); 

Class(oddDivisorsIntNonTriv, AutoFoldExp, rec(
    ev := self >> Filtered(DivisorsIntNonTriv(self.args[1].ev()), IsOddInt),
    computeType := self >> TList(TInt)
)); 


Class(isPrime, AutoFoldExp, rec(
    ev := self >> let(a:=self.args[1].ev(), IsInt(a) and IsPrime(a)),
    computeType := self >> TBool
));

Class(hasOddDivisors, AutoFoldExp, rec(
    ev := self >> Filtered(DivisorsIntNonTriv(self.args[1].ev()), IsOddInt) <> [], 
    computeType := self >> TBool
));

Class(divisorsIntNonSelf, AutoFoldExp, rec(
    ev := self >> [1] :: DivisorsIntNonTriv(self.args[1].ev()),
    computeType := self >> TList(TInt)
)); 

#F integersBetween(<min>, <max>)
#F    returns a list min..max, as a code object
#F
Class(integersBetween, AutoFoldExp, rec(
    ev := self >> [self.args[1].ev() .. self.args[2].ev()],
    computeType := self >> TList(TInt)
)); 

#F blocksInt(<n>, <min>, <max>)
#F    returns a list of possible block sizes min..max, as a code object
#F    block sizes is really any integers in the interval, that is less than <n>
#F    it should be defined in autolib .c files separately, depending on strategy, etc
#F    In Spiral, evaluation will use BlocksInt which will try 2-power sized blocks.
#F    This does not need to be consistent with autolib.c implementation.
#F
Class(blocksInt, AutoFoldExp, rec(
    ev := self >> BlocksInt(self.args[1].ev(), self.args[2].ev(), self.args[3].ev()),
    computeType := self >> TList(TInt)
));

# ---------------
# YSV NOTE: Refactor below as above
#
Class(rpDivisorsIntNonTriv, Exp, rec(computeType := self>> TPtr(TInt))); 
Class(hasCoprimeFactors, Exp, rec(computeType := self >> TBool));

# ---------------
#DivisorsIntNonTrivSym := n ->
#    Cond(IsSymbolic(n), divisorsIntNonTriv(n), # inert form 
#                        DivisorsIntNonTriv(EvalScalar(n))); # computes divisors

#DivisorsIntNonSelfSym := n ->
#    Cond(IsSymbolic(n), divisorsIntNonSelf(n), # inert form 
#                        Concatenation([1],DivisorsIntNonTriv(EvalScalar(n)))); # computes divisors

RPDivisorsIntNonTrivSym := n ->
    Cond(IsSymbolic(n), rpDivisorsIntNonTriv(n), # inert form 
                        RPDivisorsIntNonTriv(EvalScalar(n))); # computes divisors
