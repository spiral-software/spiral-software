.. _spl:


SPL
===

Matrices and Symbolic Matrices
++++++++++++++++++++++++++++++

SPL Objects
-----------

.. code-block:: none

	s1 := I(4);			# Identity matrix
	s2 := F(2);			# Butterfly matrix
	s3 := L(8, 2);			# Stride permutation matrix
	s4 := Mat([[1,2],[3,4]]);	# GAP Matrix as SPL object
	RowVec(4); ColVec(4);		# row and column vectors
	O(4); O(3, 4);			# zero matrix, square and rectangular


SPL Operations
--------------

.. code-block:: none

	s2 * s4; Compose(s2, s4);	# product of SPL
	DirectSum(s1, s2);		# matrix direct sum
	Tensor(s1, s2);		# Kronecker product of matrices
	HStack(s2, s4);		# [[s2,s4]]
	VStack(s2, s4);		# [[s2],[s4]]


Operations for SPL Objects
--------------------------

.. code-block:: none

	pm(s1);				# print as matrix
	MatSPL(s2);			# convert to GAP matrix
	s3.transpose();		# symbolic transposition
	# List all SPL objects
	List(Filtered(Dir(spiral.spl), o->IsSPL(spiral.spl.(o))),
		 e->spiral.spl.(e));


Still SPL, But Towards Σ-SPL
++++++++++++++++++++++++++++

Permutations
------------

.. code-block:: none

	s1 := L(8,4);		# The stride permutation...
	MatSPL(s1);		# ...is a matrix...
	s1.lambda();		# ...and a symbolic function
	L(8, 4) * L(8, 2);	# == I(8)
	Tensor(I(2), L(4,2)) * L(8,2);	# digit perm needs expression

	# other permutation matrices/functions
	J(4); Z(4,1); CRT(4,5); RR(13,3,2); 
	Tensor(J(4), Z(4, 1));

Diagonals
---------

.. code-block:: none

	d:= Diag(fConst(4,1.1));		# Diagonal matrix
	d.element;
	e:= RCDiag(FList(TReal, [1..16]));	# RC(Diag(...))
	e.element;
	f := DiagCpxSplit(FList(TReal, [1..16]));
	f.element;

	# diagonal matrix with dependency on free variable
	i := Ind(2);
	Diag(fCompose(FList(TReal, [1..4]), fTensor(fId(2), fBase(i))));	


More SPL Operators
++++++++++++++++++

Various
-------

.. code-block:: none

	s1 := DFT(4).terminate();	# Get the complex DFT(4) matrix
	s2 := RC(s1);			# convert it to a real 8x8 matrix
	s3 := COND(Ind(), I(2), F(2));	# conditional matrix
	s4:= Tensor(DFT(4), I(4));		# transforms are SPL objects
	MatSPL(DFT(4)) * [1..4];	# and support SPL and Matrix operations
	ConjLR(Tensor(I(2), F(2)), L(4,2), L(4,2));	# classical identity

	RowDirectSum(1, F(2), J(2));	# overlapped direct sum
	RowTensor(5, 1, F(2));		# overlapped tensor
	ColDirectSum(1, F(2), J(2));	# overlapped direct sum
	ColTensor(5, 1, F(2));		# overlapped tensor


Iterative
---------

.. code-block:: none

	i := Ind(4);
	s5 := IterDirectSum(i, F(2));		# iterative direct sum
	s6 := IterDirectSum(i, Mat([[i+1, 2*i],[-3*i, 4*i+5]]));		
	MatSPL(s6);

	s7 := IterHStack(i, F(2));		# iterative HStack
	s8 := IterVStack(i, F(2));		# iterative VStack


Defining SPL Objects
++++++++++++++++++++

SPL Matrices
------------

.. code-block:: none

	# spiral-core\namespaces\spiral\spl\symbols.gi
	Class(F, Sym, rec(
		def := size -> Checked(IsPosInt(size),
		Cond(size = 1, Mat([[1]]),
			 size = 2, Mat([[1,1], [1,-1]]),
			 Mat(Global.DFT(size)))),

		isReal    := self >> self.params[1] <= 2,
		isPermutation := False,
		transpose := self >> self,
		conjTranspose := self >> self,
		inverse := self >> let(n:=self.params[1], 
			Cond(n=1, self, n=2, 1/2*F(2), 
			Error("Inverse not supported"))),
		toAMat    := self >> DFTAMat(self.params[1]),
		printlatex := (self) >> Print(" F_{", self.params[1], "} ")
	));


Example
-------

.. code-block:: none

	s := F(2);
	MatSPL(s);


SPL Operator
------------

.. code-block:: none

	# spiral-core\namespaces\spiral\spl\Conjugate.gi
	Class(Conjugate, BaseOperation, rec(
		new := (self, spl, conj) >> Checked(IsSPL(spl), IsSPL(conj),
		When(Dimensions(conj) = [1,1], spl,
			 SPL(WithBases( self, 
				 rec( _children := [spl, conj],
				 dimensions := spl.dimensions ))))),
		isPermutation := False, 
		dims := self >> self.dimensions,
		toAMat := self >> 
			ConjugateAMat(AMatSPL(self._children[1]), 
				AMatSPL(self._children[2])),
		print := (self, i, is) >>
			Print("(", self.child(1).print(i, is), ") ^ (", 
			self.child(2).print(i+is, is), ")", self.printA()),
		transpose := self >> Inherit(self, rec(_children := [  
			TransposedSPL(self._children[1]), self._children[2] ], 
		dimensions := Reversed(self.dimensions))),
		arithmeticCost := (self, costMul, costAddMul) >>
			self._children[1].arithmeticCost(costMul, costAddMul)
	));


SPL Permutation
---------------

.. code-block:: none

	# spiral-core\namespaces\spiral\spl\perms.gi
	Class(L, PermClass, rec(
		def := (n,str) -> Checked(IsPosIntSym(n), IsPosIntSym(str),
			(not (IsInt(n) and IsInt(str)) or n mod str = 0), rec()),

		domain := self >> self.params[1],
		range  := self >> self.params[1],

		lambda := self >> let(
			n := self.params[1], str := self.params[2], i := Ind(n),
			Lambda(i, idiv(i, n/str) + str * imod(i, n/str))),

		transpose := self >> self.__bases__[1](self.params[1], 
			self.params[1] / self.params[2]),
		isSymmetric := self >> (self.params[1] = self.params[2]^2) or 
			(self.params[2] = 1) or (self.params[2] = self.params[1]),
	));


Transforms, RuleTrees, and SPL Formulas
+++++++++++++++++++++++++++++++++++++++

From Transform to SPL
---------------------

.. code-block:: none

	# create the objects and lower them
	t := DFT(4);
	rt := RandomRuleTree(t, SpiralDefaults);
	s := SPLRuleTree(rt);


Conversions Transform/SPL/GAP Matrix
------------------------------------

.. code-block:: none

	t.terminate();			# transform -> SPL
	tm := MatSPL(t);		# transform -> GAP matrix
	sm := MatSPL(s);		# SPL formula -> GAP matrix


Symbolic Correctness
--------------------

.. code-block:: none

	tm = sm;			# same, as GAP matrices
	InfinityNormMat(tm - sm);	# computes the matrix norm

	t1 := DFT(13);			# for this we lose symbolic equivalency
	rt1 := RandomRuleTree(t1, SpiralDefaults);
	s1 := SPLRuleTree(rt1);	# size 13 triggers Rader 
	tm1 := MatSPL(t1);		# this is fully symbolic, but
	sm1 := MatSPL(s1);		# Rader requires the FFT
	tm1 = sm1;			# thus floating-point matrix
	InfinityNormMat(tm1 - sm1);	# but equivalent wrt. floating-point 








