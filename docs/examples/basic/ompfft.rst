
Threaded FFT with OpenMP
++++++++++++++++++++++++


.. code-block:: none

    opts := LocalConfig.getOpts(
		rec(dataType := T_Real(64), globalUnrolling := 512), 
		rec(numproc := 2, api := "OpenMP"),
		rec(svct := true, splitL := false, oddSizes := false, stdTTensor := true, tsplPFA := false)
	);
    transform := TRC(DFT(32)).withTags(opts.tags);
    ruletree := RandomRuleTree(transform, opts);
    icode := CodeRuleTree(ruletree, opts);
    PrintTo("SSE_OMP2_DFT32.c", PrintCode("SSE_OMP2_DFT32", icode, opts));

