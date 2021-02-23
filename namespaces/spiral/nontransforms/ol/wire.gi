
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# The Wire class implements OL connection system
# usage: Wire(inputs, interconnection matrix)
# ex: Wire([1,2,3,4],MatSPL(L(4,2)));
 Class(Wire, BaseMat, rec(
   abbrevs := [ (inp, mat) ->  Checked(IsList(inp), IsMat(mat), [[inp, mat]])],
   new := meth(self, p)
       local res;
       res := SPL(WithBases(self, rec(params := p, TType:=Replicate(Length(p[1]),TUnknown))));
       return res;
   end,
   isPermutation := self >> false,
   #now just make it work, assuming all inputs have the same type
   dmn:=self >> List(self.params[1],x->TArray(self.TType[1],x)),
   rng:=self >> List(self.params[2]*self.params[1],x->TArray(self.TType[1],x)),
   sums:= self>>self,
   numops:=self>>0,
   rChildren := self >> [],
   print := meth(self,i,is)
     Print(self.name, "(", PrintCS(self.params), ")");
     return;
   end
 ));

 Class(LinkIO, BaseMat, rec(
   abbrevs := [ (sums, mat) ->  Checked(IsMat(mat), [[sums, mat]])],
   new := meth(self, p)
       local res;
       res := SPL(WithBases(self, rec(params := p, TType:=Replicate(Length(p[1]),TUnknown))));
       return res;
   end,
   isPermutation := self >> false,
   dmn:=self >> self.params[1].dmn(),
   rng:=self >> self.params[1].rng(),
   sums:= self>>self,
   numops:=self>>0,
   rChildren := self >> self.params,
   rSetChild := meth ( self, n, newChild )
     if n = 1  then
         self.params[1] := newChild;
     elif n = 2 then
         self.params[2] := newChild;
    else
        Error("<n> not in range");
     fi;
   end,
   print := meth(self,i,is)
     Print(self.name, "(", PrintCS(self.params), ")");
     return;
   end
 ));
