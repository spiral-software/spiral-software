
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

bc := n -> HStack(2/n*ColVec(fConst(TInt, n, 1)), I(n)) * 
           VStack(RowVec(fConst(TInt, n, 1)), Circulant(n,[-1,1],-1));

split := n -> DirectSum(I(n/2), bc(n/2)) * L(n, 2);

raderBrenner := n -> Tensor(F(2), I(n/2)) * 
    DirectSum(DFT(n/2), Diag(List([0..n/2-1], e -> When(e=0, 1/2, E(4)/2/SinPi(2*e/n)))) * DFT(n/2)) * split(n);

check := n -> inf_norm(MatSPL(raderBrenner(n))-MatSPL(DFT(n)));

NewRulesFor(DFT, rec(
   DFT_RaderBrenner := rec(
       applicable := t -> t.params[1] > 2 and IsEvenInt(t.params[1]),
       freedoms := t -> [],
       child := (t, fr) -> [ DFT(t.params[1]/2, t.params[2]) ],
       apply := (t, C, Nonterms) -> let(n := t.params[1], 
           rot := t.params[2],           
           bc := HStack(4/n*ColVec(fConst(TInt, n/2, 1)), I(n/2)) * 
                 VStack(RowVec(fConst(TInt, n/2, 1)), transforms.filtering.Circulant(n/2,[-1,1],-1).terminate()),

           Tensor(F(2), I(n/2)) * 
           DirectSum(C[1], Diag(List([0..n/2-1], e -> When(e=0, 1/2, E(4)/2/SinPi(rot*2*e/n)))) * C[1]) *  
           DirectSum(I(n/2), Mat(MatSPL(bc))) * 
           L(n, 2)
       )
   )
));

opts := CopyFields(SpiralDefaults, rec(breakdownRules := rec(DFT := [DFT_CT, DFT_Base])));
rbopts := CopyFields(SpiralDefaults, rec(breakdownRules := rec(DFT := [DFT_RaderBrenner, DFT_Base])));

d := DPBench(rec(raderbrenner := rbopts, default := opts), rec(verbosity:=0));
d.run([DFT(4), DFT(8), DFT(16), DFT(32)]);
