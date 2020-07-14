comment("");
comment("Generate a Threaded FFT with OpenMP ");
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

opts := LocalConfig.getOpts(
            rec(dataType := T_Real(64), globalUnrolling := 512),
            rec(numproc := 2, api := "OpenMP"),
            rec(svct := true, splitL := false, oddSizes := false, stdTTensor := true, tsplPFA := false)
    );

transform := TRC(DFT(64)).withTags(opts.tags);
ruletree := RandomRuleTree(transform, opts);
icode := CodeRuleTree(ruletree, opts);

##  PrintTo("SSE_OMP2_DFT256.c", PrintCode("SSE_OMP2_DFT256", icode, opts));

meas := CMeasure(icode, opts);
if (meas <= 0) or (meas >= 1e+100) then
    Print("CMeasure failed -- returned: ", meas, ", Profiler did not run correctly\n");
    TestFailExit();
fi;

cmat := CMatrix(icode, opts);
if not IsList(cmat) then
    Print("CMatrix failed -- returned: ", cmat, ", Profiler did not run correctly \n");
    TestFailExit();
fi;

##  smat := MatSPL(RC(transform));  ## generates a matrix twice as large

smat := MatSPL(transform);
diff := 1;
diff := cmat - smat;
if not IsList(diff) then
    Print("CMatrix failed -- matrix size mismatch -- bad result from Profiler\n");
    TestFailExit();
fi;

inorm := InfinityNormMat(diff);

if inorm > 1e-5 then
    Print("Transform failed -- max diff: ", inorm, "\n");
    TestFailExit();
fi;

