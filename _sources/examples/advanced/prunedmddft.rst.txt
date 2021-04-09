
Pruned 3D FFT from Planewave
============================

.. code-block:: none

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

	trm := MatSPL(RC(t));
	InfinityNormMat(cm-trm);




