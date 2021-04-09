.. _end2end:

End to End, from Transform to C Code
====================================


DFT(8) to C
+++++++++++

.. code-block:: none

	n := 8; k := -1;		# transform parameters
	opts := SpiralDefaults;	# default options
	opts.useDeref := false;	# prefer array[] over *(deref)
	t := DFT(n, k);		# transform
	rt := RandomRuleTree(t, opts);	# get rule tree
	spl := SPLRuleTree(rt);	# Debug: SPL formula
	ss1 := spl.sums();	# Debug: SPL->Sigma-SPL w/o optimization
	ss := SumsRuleTree(rt, opts);	# Correct: from rt -> Sigma-SPL
	c1 := CodeSums(ss, opts);	# Debug: Sigma-SPL->code
	c := CodeRuleTree(rt, opts);	# Correct: rt-> code in one shot
	PrintCode("dft8", c, opts);	# final code


Using DP and CodeRuleTree
+++++++++++++++++++++++++

.. code-block:: none

	n := 1024; k := -1;		# transform parameters
	opts := SpiralDefaults;	# default options
	opts.globalUnrolling := 16;	# set smaller unrolling
	t := DFT(n, k);		# transform
	best := DP(t, rec(), opts);	# run search
	rt := best[1].ruletree;
	c := CodeRuleTree(rt, opts);	# Correct: rt-> code in one shot
	PrintCode("dft"::StringInt(n), c, opts);	# final code


Correctness Checks
++++++++++++++++++

.. code-block:: none

	tm := MatSPL(t);		# symbolic complex cyclotomic matrix
	tmr := MatSPL(RC(t));		# symbolic real cyclotomic matrix
	splm := MatSPL(spl);		# symbolic complex cyclotomic matrix
	tmr := MatSPL(RC(t));		# symbolic real cyclotomic matrix
	ssm := MatSPL(ss);		# symbolic double-precision matrix
	cm := CMatrix(c, opts);	# symbolic double-precision matrix
	tm = splm;			# symbolically equivalent
	InfinityNormMat(tmr - ssm);	# only equivalent up to rounding error
	InfinityNormMat(tmr - cm);	# only equivalent up to rounding error


Other Examples
++++++++++++++

.. code-block:: none

	Import(dct_dst, realdft);	# load DCT/DST and Real DFT package
	opts := SpiralDefaults;	# default options
	t1 := DFT(31);		# a larger prime size
	t2 := DCT3(32);	# a larger cosine transform of type 3
	t3 := PRDFT(17);	# Real DFT in the "pack" format
	t4 := PrunedDFT(128, 16, [0,1,5,6,7]);

	ts := [t1, t2, t3, t4];
	rts := List(ts, tt->RandomRuleTree(tt, opts));
	cs := List(rts, rr->CodeRuleTree(rr, 
			  CopyFields(SpiralDefaults, rec(globalUnrolling := 64))));




















