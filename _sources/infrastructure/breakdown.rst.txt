.. _breakdown:


Breakdown Rules
===============


Base Rules
++++++++++

Definition
----------

.. code-block:: none

	# In spiral-core\namespaces\spiral\transforms\dft\dft_rules.gi
	 DFT_Base := rec(
		forTransposition := false,
		applicable       := nt -> nt.params[1] = 2 and not nt.hasTags(),
		apply            := (nt, C, cnt) -> F(2)
	)


Twiddle Function for DFT
------------------------

.. code-block:: none

	Tw1 := (n,d,k) -> Checked(
		IsPosIntSym(n), IsPosIntSym(d), IsIntSym(k),
		fCompose(dOmega(n,k),
			diagTensor(dLin(div(n,d), 1, 0, TInt), 
				dLin(d, 1, 0, TInt))));

			
Rule Methods
------------

.. code-block:: none

	PrintActiveRules(DFT);		# rules for DFT currently active
	DFT_Base.switch;		# filed in rule to determine active
	t := DFT(2);			
	DFT_Base.applicable(t);	# is the rule applicable
	DFT_Base.children(t);		# all possible Algorithmic choices
	DFT_Base.apply(t, [], []);	# t->spl for a particular choice


Cooley-Tukey Rule
+++++++++++++++++


Definition
----------

.. code-block:: none

	# In spiral-core\namespaces\spiral\transforms\dft\dft_rules.gi
	DFT_CT := rec(
		maxSize       := false,
		forcePrimeFactor := false,
		applicable := (self, nt) >> nt.params[1] > 2
			and not nt.hasTags()
			and (self.maxSize=false or nt.params[1] <= self.maxSize)
			and not IsPrime(nt.params[1])
			and When(self.forcePrimeFactor, not
					 DFT_GoodThomas.applicable(nt), true),
		children  := nt -> Map2(DivisorPairs(nt.params[1]),
				(m,n) -> [ DFT(m, nt.params[2] mod m), 
						   DFT(n, nt.params[2] mod n) ]),
		apply := (nt, C, cnt) -> let(mn := nt.params[1], 
					m := Rows(C[1]), n := Rows(C[2]),
				Tensor(C[1], I(n)) *
				Diag(fPrecompute(Tw1(mn, n, nt.params[2]))) *
				Tensor(I(m), C[2]) *
				L(mn, m)
			)
		)


Applicability
-------------

Cooley Tukey requires a non-prime size.

.. code-block:: none

	t := DFT(2);
	t1 := DFT(4);
	t2 := DFT(8);
	t3 := DFT(20);

	DFT_CT.applicable(t);		# see for which sized DFT_CT 
	DFT_CT.applicable(DFT(5));	# is applicable
	DFT_CT.applicable(t1);
	DFT_CT.applicable(t2);
	DFT_CT.applicable(t3);

Children: Algorithmic Choices
-----------------------------

.. code-block:: none

	c1 := DFT_CT.children(t2);	# enumerate all algorithmic choices
	c2 := DFT_CT.children(t2);
	c3 := DFT_CT.children(t3);


Expand DFT(8) by Hand
---------------------

.. code-block:: none

	s := DFT_Base.apply(t, [], []);	# expand DFT(2) -> F(2)
	s1 := DFT_CT.apply(t1, [s, s], [t, t]); 	# DFT(4) -> SPL
	s2 := DFT_CT.apply(t2, [s1, s], [t1, t]);	# DFT(8) -> SPL
	MatSPL(t2) = MatSPL(s2);


Ruletrees and SPL Revisited
+++++++++++++++++++++++++++


From Transform to SPL Formula
-----------------------------

.. code-block:: none

	n := 8; k := -1;			# transform parameters
	opts := CopyFields(SpiralDefaults, 	# local configuration
		rec(breakdownRules := rec(
			DFT := [DFT_Base,		
			CopyFields(DFT_CT, rec(maxSize := 20))])));
	t := DFT(n, k);			# transform
	rt := RandomRuleTree(t, opts);	# get rule tree
	spl := SPLRuleTree(rt);		# SPL formula


Impact of Configuration
-----------------------

.. code-block:: none

	PrintActiveRules(DFT);	
	opts.breakdownRules.DFT;
	DFT_CT.maxSize;		# global configuration unchanged
	ct := Filtered(opts.breakdownRules.DFT, i->i.name =  DFT_CT.name)[1];
	ct.maxSize;			# access local configuration
	t2 := DFT(21);			# works with global but not local opts
	rt := RandomRuleTree(t2, SpiralDefaults);
	rt2 := RandomRuleTree(t2, opts);
	FindUnexpandableNonterminal(t2, opts);	# Where do we fail?
	ct.maxSize := false;				# remove guard in DFT_CT
	rt2 := RandomRuleTree(t2, opts);		# try again
	FindUnexpandableNonterminal(t2, opts);	# Where do we fail now?






