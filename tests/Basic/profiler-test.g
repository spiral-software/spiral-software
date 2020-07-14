comment("");
comment("A minimal test for the profiler... ");
comment("Check CMeasure gives a valid result...");
comment("");

ClearBasicProfilerTestResults();

opts := SpiralDefaults;
transform := DFT(4);
ruletree := RuleTreeMid(transform, opts);
icode := CodeRuleTree(ruletree, opts);
meas := CMeasure(icode, opts);

if (meas <= 0) or (meas >= 1e+100) then
    Print("CMeasure failed -- returned: ", meas, ", Profiler did not run correctly\n");
    MarkBasicProfilerTest(false);
    TestFailExit();
fi;

cmat := CMatrix(icode, opts);
if not IsList(cmat) then
    Print("CMatrix failed -- returned: ", cmat, ", Profiler did not run correctly \n");
    MarkBasicProfilerTest(false);
    TestFailExit();
fi;

smat := MatSPL(RC(transform));
diff := 1;
diff := cmat - smat;
if not IsList(diff) then
    Print("CMatrix failed -- matrix size mismatch -- bad result from Profiler\n");
    MarkBasicProfilerTest(false);
    TestFailExit();
fi;

inorm := InfinityNormMat(diff);
if inorm > 1e-5 then
    Print("Transform failed -- max diff: ", inorm, "\n");
    MarkBasicProfilerTest(false);
    TestFailExit();
fi;

MarkBasicProfilerTest(true);

