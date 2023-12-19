
Threaded FFT with OpenMP
++++++++++++++++++++++++


.. code-block:: none

	N := 64;
    threads := 2;
    opts := LocalConfig.getOpts(
		rec(dataType := T_Real(64), globalUnrolling := 512), 
		rec(numproc := threads, api := "OpenMP"),
		rec(svct := true, splitL := false, oddSizes := false, stdTTensor := true, tsplPFA := false)
	);
    transform := TRC(DFT(N)).withTags(opts.tags);
    ruletree := RuleTreeMid(transform, opts);
    icode := CodeRuleTree(ruletree, opts);
	name := "AVX_OMP"::String(threads)::"_DFT"::String(N);
    PrintTo(name::".c", PrintCode(name, icode, opts));

