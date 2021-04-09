
3D FFT with AVX Using Dynamic Programming Search
++++++++++++++++++++++++++++++++++++++++++++++++


.. code-block:: none

	opts := SIMDGlobals.getOpts(AVX_4x64f);
	transform := TRC(MDDFT([64,64,64])).withTags(opts.tags);
	best := DP(transform, rec(), opts);
	ruletree := best[1].ruletree;
	icode := CodeRuleTree(ruletree, opts);
	PrintTo("AVX_3DDFT64.c", PrintCode("AVX_3DDFT64", icode, opts));


