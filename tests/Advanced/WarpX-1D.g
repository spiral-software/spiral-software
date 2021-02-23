
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

comment("");
comment("WarpX 1D");
comment("");

##  if CheckBasicProfilerTest() ==> profiler tests passed

if not CheckBasicProfilerTest() then
    PrintLine("Basic Profiler test NOT PASSED, skipping test");
    if FileExists(".") then
        TestSkipExit();
    else
        TestFailExit();
    fi;
fi;

opts := SpiralDefaults;

ni:= 4;
n := 8;
no := 6;

## 1D case -- getting started
padg := fAdd(n, no, (n-no)/2);
pads := fAdd(n, ni, (n-ni)/2);

scat := Scat(pads);
gath := Gath(padg);
dft := DFT(n, 1);
idft := DFT(n, -1);
krn := Diag(List([0..n-1], i->i+1));

op := gath * idft * krn * dft * scat;

## pruned DFT
sl := [(n-ni)/2 .. n - (n-ni)/2 - 1];
gl := [(n-no)/2 .. n - (n-no)/2 - 1];

pdft := PrunedDFT(n, 1, 1, sl);
pidft := PrunedIDFT(n, -1, 1, gl);
InfinityNormMat(MatSPL(gath * idft) - MatSPL(pidft));
InfinityNormMat(MatSPL(dft * scat) - MatSPL(pdft));

krn := Diag(List([0..n-1], i->i+1));
pop := pidft * krn * pdft;

InfinityNormMat(MatSPL(op) - MatSPL(pop));

## constant vector field
vi := 3;
vo := 2;

vpdft := Tensor(pdft, I(vi));
vpidft := Tensor(pidft, I(vo));

cmat := Mat([[1,2,3], [4,5,6]]);
vkrn := Tensor(I(n), cmat);

vpop := vpidft * vkrn * vpdft;

## location dependent vector field
nnz := 4;
mf := FData(List([0..nnz * n-1], i-> Complex(i+E(4))));
i := Ind(n);
cimat := Mat([[mf.at(nnz*i),0,mf.at(nnz*i+1)], [0,mf.at(nnz*i+2),mf.at(nnz*i+3)]]);

vikrn := IterDirectSum(i, cimat);
vipop := vpidft * vikrn * vpdft;

## real arithmetic
rvipop := RC(vipop);
rvipopm := List(MatSPL(rvipop), r->List(r, ReComplex));

# SPL Matrix
rt := RandomRuleTree(rvipop, opts);
spl := SPLRuleTree(rt);
splm := List(MatSPL(spl), r->List(r, ReComplex));
InfinityNormMat(rvipopm - splm);

# CMatrix
tvpdft := TTensorI(pdft, vi, AVec, AVec);
tvpidft := TTensorI(pidft, vo, AVec, AVec);
tvikrn := TTensorInd(cimat,i, APar, APar);
tvipop := TRC(TCompose([tvpidft, tvikrn, tvpdft]));

rt := RandomRuleTree(tvipop, opts);
c := CodeRuleTree(rt, opts);

cm := CMatrix(c, opts);
if not IsList(cm) then
    Print("CMatrix failed -- returned: ", cm, "\n");
    TestFailExit();
fi;

inorm := 1;
inorm := InfinityNormMat(rvipopm - cm);
if inorm > 1e-5 then
    Print("InfinityNormMat failed -- max diff: ", inorm, "\n");
    TestFailExit();
fi;
