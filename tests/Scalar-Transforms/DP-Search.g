comment("");
comment("Dynamic Programming Search Test");

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
transform := DFT(8);
best := DP(transform, rec(), opts);

if (best[1].measured <= 0) or (best[1].measured >= 1e+100) then
    Print("DP / CMeasure failed -- returned: ", best[1].measured, "\n");
    TestFailExit();
fi;

ruletree := best[1].ruletree;
icode := CodeRuleTree(ruletree, opts);
