comment("");
comment("Generate a Simple FFT ");
comment("Check CMeasure and CMatrix");
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
transform := DFT(4);
ruletree := RandomRuleTree(transform, opts);
icode := CodeRuleTree(ruletree, opts);

##  PrintCode("DFT4", icode, opts);
##  PrintTo("DFT4.c", PrintCode("DFT4", icode, opts));

meas := CMeasure(icode, opts);
if (meas <= 0) or (meas >= 1e+100) then
    Print("CMeasure failed -- returned: ", meas, "\n");
    TestFailExit();
fi;

cmat := CMatrix(icode, opts);
if not IsList(cmat) then
    Print("CMatrix failed -- returned: ", cmat, "\n");
    TestFailExit();
fi;

smat := MatSPL(RC(transform));
diff := 1;
diff := cmat - smat;
if not IsList(diff) then
    Print("CMatrix failed -- matrix size mismatch\n");
    TestFailExit();
fi;

inorm := InfinityNormMat(diff);
if inorm > 1e-5 then
    Print("InfinityNormMat failed -- max diff: ", inorm, "\n");
    TestFailExit();
fi;
