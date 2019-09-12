comment("");
comment("Generate a Threaded FFT with OpenMP ");
comment("");
opts := LocalConfig.getOpts(
            rec(dataType := T_Real(64), globalUnrolling := 512),
            rec(numproc := 2, api := "OpenMP"),
            rec(svct := true, splitL := false, oddSizes := false, stdTTensor := true, tsplPFA := false)
    );
transform := TRC(DFT(64)).withTags(opts.tags);
ruletree := RandomRuleTree(transform, opts);
icode := CodeRuleTree(ruletree, opts);
PrintTo("SSE_OMP2_DFT256.c", PrintCode("SSE_OMP2_DFT256", icode, opts));
