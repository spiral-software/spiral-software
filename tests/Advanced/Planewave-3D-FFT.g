comment("");
comment("Pruned 3D FFT from Planewave");
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

n := 8;
k := 1;
np := 2;
pl1d := [0..np-1]::[n-np..n-1];

opts := SpiralDefaults;

t := PrunedMDDFT([n, n, n], k, 1, [pl1d, pl1d, pl1d]);
tm := MatSPL(t);

rt := RandomRuleTree(t, opts);

spl := SPLRuleTree(rt);
sm := MatSPL(spl);

InfinityNormMat(sm-tm);

c := CodeRuleTree(rt, opts);
cm := CMatrix(c, opts);
if not IsList(cm) then
    Print("CMatrix failed -- returned: ", cm, "\n");
    TestFailExit();
fi;

trm := MatSPL(RC(t));
inorm := 1;
inorm := InfinityNormMat(cm-trm);
if inorm > 1e-5 then
    Print("InfinityNormMat failed -- max diff: ", inorm, "\n");
    TestFailExit();
fi;
