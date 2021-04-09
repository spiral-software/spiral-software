.. _searching:


Search
======

Dynamic Programming
+++++++++++++++++++

Standard Dynamic Programming
----------------------------

.. code-block:: none

	n := 15; k := -1;           # transform parameters
	opts := SpiralDefaults;     # default options
	opts.globalUnrolling := 16; # set smaller unrolling
	t := DFT(n, k);             # transform
	best := DP(t, rec(), opts); # run search
	rt := best[1].ruletree;     # get best rule tree


Hashing and Custom Measure Functions
------------------------------------

.. code-block:: none

	dpopts := rec(verbosity := 5, hashTable := HashTableDP());
	dpopts.measureFunction := (rt, opts) -> 
		let(c:= CodeRuleTree(rt, opts),    # generate code
			Length(Filtered(           # count flops in code
				Collect(c, @(1,[add, sub, mul, neg])), i->i.t=TReal)));
	best := DP(t, dpopts, opts);	# run search with flop minimization
	#look what’s in the hash table
	HashLookup(dpopts.hashTable, DFT(5, 1))[1].ruletree;
	HashLookup(dpopts.hashTable, DFT(5,-1));

	# now time the tree found through flop minimization
	rt1 := HashLookup(dpopts.hashTable, DFT(15, 1))[1].ruletree;
	DPMeasureRuleTree(rt1, opts);

