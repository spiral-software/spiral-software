
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(LoopGath, Gath);
Class(LoopScat, Scat);
Class(LoopPrm, Prm);

FuncClass.gath := self >> Gath(self);
FuncClass.scat := self >> Scat(self);

fId.gath := self >> I(self.params[1]);
fId.scat := self >> I(self.params[1]);

L.scat := self >> let(m:=self.params[2], n:=self.params[1]/self.params[2], j:=Ind(m), fid := fId(n), fbase := fBase(m,j),
                      gath := Gath(fTensor(fid, fbase)), scat := Scat(fTensor(fbase, fid)),
                      ISum(j, m, scat*gath));

L.gath := self >> let(m:=self.params[2], n:=self.params[1]/self.params[2], j:=Ind(n), fid := fId(m), fbase := fBase(n,j),
                      gath := Gath(fTensor(fbase, fid)), scat := Scat(fTensor(fid, fbase)),
                      ISum(j, n, scat*gath));

fCompose.gath := self >> Compose(Reversed(List(self.children(), i->i.gath())));
fCompose.scat := self >> Compose(List(self.children(), i->i.scat()));

fTensorBuf := function(func, combine)
    local ch, n, loopvars, i, lv, newch1, newch2, res; 
    ch := func.children();
    n := Length(ch);
    loopvars := List(Filtered(ch, c -> ObjId(c)=fId), c -> Ind(c.size));
    
    lv := 1; newch1 := []; newch2 := [];
    for i in [1..n] do
        if ObjId(ch[i]) <> fId then
	    Add(newch1, ch[i]);
	else
	    Add(newch1, fBase(loopvars[lv]));
	    Add(newch2, fBase(loopvars[lv]));
	    lv := lv+1;
	fi;
    od;

    res := combine(fTensor(newch1), fTensor(newch2));

    # lv here is number of loopvars + 1
    for i in Reversed([1..lv-1]) do
        res := ISum(loopvars[i], res);
	# unroll inner loop
	#if i=lv-1 then res:=BB(res); fi;
    od;
    return res;
end;

#fTensor.gath := self >> Tensor(List(self.children(), i->i.gath()));

fTensor.scat := self >> fTensorBuf(self, (f1, f2) -> BB(Scat(f1) * Gath(f2)));
fTensor.gath := self >> fTensorBuf(self, (f1, f2) -> BB(Scat(f2) * Gath(f1)));

