
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(DTTType, rec(
   isDTTType:=true,
    __call__ := arg >> ApplyFunc(spiral.transforms.DTT, arg)
));

Class(DTT_C3,  DTTType);  Class(DTT_C4,  DTTType);
Class(DTT_S3,  DTTType);  Class(DTT_S4,  DTTType);
Class(DTT_IC3, DTTType);  Class(DTT_IC4, DTTType);
Class(DTT_IS3, DTTType);  Class(DTT_IS4, DTTType);
IsDTTType := x -> IsRec(x) and IsBound(x.isDTTType) and x.isDTTType;

#F DTT(<type>,<size>,<skew-angle>,<is-poly?>,<is-inv?>) 
#F    Discrete Trigonometric Transform
#F 
Class(DTT, NonTerminal, rec(
    dims := self >> [self.params[2], self.params[2]],
    isReal := self >> true,
    abbrevs := [ (t,N,skew)     -> Checked(IsPosInt(N), IsDTTType(t), [t,N,skew,false]),
             (t,N,skew,poly) -> Checked(IsPosInt(N), IsDTTType(t), [t,N,skew,poly]),
           ],
    terminate := self >> Error("not implemented"),
   
    SmallRandom := () ->
        [ RandomList([DTT_C3, DTT_S3, DTT_C4, DTT_S4]), 
      Random([2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 27, 30, 32]),
      1 / RandomList([2..10]),
      RandomList([true, false]) ]
#  print := (self,i,is) >> let(p:=self.params,Print(p[1],"(",PrintCS(Drop(p, 1)),")")
));
