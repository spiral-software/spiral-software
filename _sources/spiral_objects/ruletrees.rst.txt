.. _ruletrees:

Transforms and Rule Trees
=========================

Non-Terminals (Transforms)
++++++++++++++++++++++++++

Definitions
-----------

.. code-block:: none

	t1 := DFT(4);		# complex DFT of size 4
	t2 := MDDFT([4,4]);	# 2D DFT
	t3 := DFT(5);		# non 2-power DFT
	Import(dct_dst);	# load DCT/DST package
	t4 := DCT3(8);		# cosine transform of type 3, size 8
	Import(filtering);	# load package filtering
	t5 := Filt(4, [1,2,3,4]); # FIR filter with constant taps
	Import(wht);		# load Walsh-Hadamard Transform
	t6 := WHT(3);		# WHT of size 8
				
Operations on Functions
-----------------------

.. code-block:: none

	DoForAll([t1,t2,t3,t4,t5,t6], # print them all as matrices
		t->Print(pm(t), "\n"));
	t1.terminate();		# translate into matrix
	t4.transpose();		# transposed transform
	t1.conjTranspose();	# conjugated transposed transform
	t3.inverse();		# inverse transform transform
	t2.dims();		# transforms have a size

	SpiralDefaults.breakdownRules; # all transforms known to the system


Rule Trees
++++++++++

Expand Non-Terminals
--------------------

.. code-block:: none

	opts := SpiralDefaults;
	t1 := DFT(4);				# complex DFT of size 4
	rt1 := RandomRuleTree(t1, opts);	# create a random rule tree
	t2 := DFT(80);				# complex DFT of size 80
	rt2 := RandomRuleTree(t2, opts);	# create a random rule tree

	
Exploring a Rule Tree
---------------------

.. code-block:: none

	rt1.node;				# this node
	rt1.rule;				# rule applied at node
	rt1.transposed;				# rule applied transposed ?
	rt1.children;				# a level down in the tree
	rt1.children[1];			# first child node
	rt1.children[1].node;			# same as root node
	rt1.children[1].rule;
	rt1.children[1].children;
	rt1.children[1].transposed;
	rt1.children[2];			# second child node
	rt1.children[2].node;			# again tree node structure
	rt1.children[2].rule;
	rt1.children[2].children;
	rt1.children[2].transposed;


Non-Terminal Example: DFT
+++++++++++++++++++++++++

Definition
----------

.. code-block:: none

	# In spiral-core\namespaces\spiral\transforms\dft\dft.gi
	Class(DFT, DFT_NonTerm, rec(
		transpose     := self >> DFT(self.params[1], 
						 self.params[2]).withTags(self.getTags()),
		conjTranspose := self >> DFT(self.params[1], 
						 -self.params[2]).withTags(self.getTags()),
		inverse := self >> self.conjTranspose(),
		omega4pow := (r,c) -> 4*r*c,
	));


Base Class
----------

.. code-block:: none

	Class(DFT_NonTerm, TaggedNonTerminal, rec(
		abbrevs := [(n) -> Checked(IsPosIntSym(n), [_unwrap(n), 1]),...],
		hashAs := ...,
		dims := ...,
		terminate := ...,
		isReal := ...,
		SmallRandom := () -> Random([2..16]),
		LargeRandom := () -> 2 ^ Random([6..15]),
		normalizedArithCost := ...
		TType := T_Complex(TUnknown)
	));







