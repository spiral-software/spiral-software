.. _sigma_spl:


Σ-SPL
=====

.. only:: html

   .. contents::

Gather and Scatter
++++++++++++++++++

Gather and Scatter Matrices
---------------------------

.. code-block:: none

	g1 := Gath(fId(2));			# Gath(fId(.)) = I(.)
	g2 := Gath(fBase(4, 0));		# Gath(fBase(.,.)) = base vec
	g3 := Gath(fTensor(fBase(4, 0), fId(2)));	# standard pattern

	s1 := Scat(fId(2));			# Scat(fId(.)) = I(.)
	s2 := Scat(fBase(4, 2));		# Scat(fBase(.,.)) = base vec
	s3 := Scat(fTensor(fId(2), fBase(4, 3)));	# standard pattern

Scatter/Kernel/Gather Pattern
-----------------------------

.. code-block:: none

	A := F(2); j := 0;
	# iteration j of Tensor(I(4), F(2))
	sag1 := Scat(fTensor(fBase(4, j), fId(2))) * A * 
			Gath(fTensor(fBase(4, j), fId(2)));

	# iteration j of Tensor(F(2), I(4))
	sag2 := Scat(fTensor(fId(2), fBase(4, j))) * A * 
			Gath(fTensor(fId(2), fBase(4, j)));

	# iteration j of Tensor(I(4), F(2)) * L(8, 4)
	sag3 := Scat(fTensor(fBase(4, j), fId(2))) * A * 
			Gath(fTensor(fId(2), fBase(4, j)));
	pm(sag1); pm(sag2); pm(sag3);


Tensor Product and Gath/Scat/ISum
---------------------------------

.. code-block:: none

	A := F(2); 
	j := Ind(4);
	# Sigma-SPL for Tensor(I(4), F(2))
	sag1 := Scat(fTensor(fBase(j), fId(2))) * A * 
			Gath(fTensor(fBase(j), fId(2)));
	s1 := ISum(j, sag1);
	MatSPL(s1) = MatSPL(Tensor(I(4), F(2)));

Other Tensor Patterns
---------------------

.. code-block:: none

	# Sigma-SPL for Tensor(F(2), I(4))
	s2 := ISum(j, Scat(fTensor(fId(2), fBase(j))) * A * 
			Gath(fTensor(fId(2), fBase(j))));
	MatSPL(s2) = MatSPL(Tensor(F(2), I(4)));

	# Sigma-SPL for Tensor(I(4), F(2)) * L(8, 4)
	s3 := ISum(j, Scat(fTensor(fBase(j), fId(2))) * A * 
			Gath(fTensor(fId(2), fBase(j))));
	MatSPL(s3) = MatSPL(Tensor(I(4), F(2)) * L(8, 4));

	# Direct sum
	ISum(j, Scat(fTensor(fBase(j), fId(2))) * Mat([[j,-j],[j, j]]) * 
			Gath(fTensor(fBase(j), fId(2))));


Advanced Loops
++++++++++++++

Tensor Product and Gath/Scat/ISum
---------------------------------

.. code-block:: none

	A := F(2); 
	j := Ind(4);
	k := Ind(2);

	# Sigma-SPL for Tensor(I(4), F(2), I(2))
	sag := Scat(fTensor(fBase(j), fId(2), fBase(k))) * A * 
			Gath(fTensor(fBase(j), fId(2), fBase(k)));
	s := ISum(k, ISum(j, sag));
	MatSPL(s) = MatSPL(Tensor(I(4), F(2), I(2)));


More Complex Example
--------------------

.. code-block:: none

	i := Ind(8);
	j := Ind(4);
	k := Ind(2);
	A := Mat([[j+1,-2*j],[j*k, j+1]]); B := F(2); 

	s := ISum(k, ISum(j, Scat(fTensor(fBase(j), fBase(k), fId(2))) * A *
						 Gath(fTensor(fId(2), fBase(k), fBase(j))))) *
		 ISum(i, Scat(fTensor(fBase(i), fId(2))) * B 
			   * Gath(fTensor(J(2), fCompose(Z(8,3), fBase(i)))));
	MatSPL(s);


Gath/Scat, Diag and Perms
+++++++++++++++++++++++++

Gather Functions
----------------

.. code-block:: none

	# use Lambda functions directly in Gath/Scat
	i := Ind(8);
	f1 := Lambda(i, imod(5*i+7, 32)).setRange(32);
	g := Gath(f1);

	# Use indirection tables in Gath/Scat. By default not supported
	f2 := CopyFields(FData(List([0..7], j->V(Mod(5*j+7, 32)))),
					 rec(range := self >> 32));
	s := Scat(f2);


Diagonals
---------

.. code-block:: none

	# the true Twiddle diagonal in DFT(8)
	d := Diag(fPrecompute(fCompose(dOmega(8, 1), 
			  diagTensor(dLin(V(4), 1, 0, TInt), dLin(2, 1, 0, TInt)))));
	MatSPL(d);
	e:= RCDiag(fCompose(FData(List([1..32], u->Value(TReal, u))), 
			   fTensor(fId(4), fBase(i))));
	# print out the function values
	e.element.tolist();
	# access the indirection table
	e.element.children()[1].var.value;


Defining Σ-SPL Objects
++++++++++++++++++++++

Gather
------

.. code-block:: none

	# spiral-core\namespaces\spiral\spl\sums.gi
	Class(Gath, SumsBase, BaseMat, rec(
		rChildren := self >> [self.func],
		rSetChild := rSetChildFields("func"),
		new := (self, func) >> SPL(WithBases(self, rec(func := 
		  Checked(IsFunction(func) or IsFuncExp(func), func)))).setDims(),
		dims := self >> [self.func.domain(), self.func.range()],
		sums := self >> self,
		area := self >> Sum(Flat([self.func.domain()])),
		isReal := self >> true,
		transpose := self >> Scat(self.func),
		conjTranspose := self >> self.transpose(),
		inverse := self >> self.transpose(),
		toAMat := self >> let(
		  n := EvalScalar(self.func.domain()),
		  N := EvalScalar(self.func.range()),
		  func := self.func.lambda(),
		  AMatMat(List([0..n-1], row -> let(idx:=evalScalar(func.at(row)),
			Cond(idx _is funcExp, When(idx.args[1]=0, Replicate(N, 0), 
				 Error("... ")),
			 BasisVec(N, idx)))))),
	));


Σ-SPL Index Mapping Functions
+++++++++++++++++++++++++++++

Definition
----------

.. code-block:: none

	f := fId(4);			# I4->I4: i->i
	j := Ind(4);			# variable with range
	g := fBase(j);			# I1->I4: i->j
	h := fAdd(4,2,1);		# I2->I4: i->i+1
	u := L(16, 4);			# permutation i -> \ell(16,4)(i)

Operations on Functions
-----------------------

.. code-block:: none

	f.at(0);			# evaluate function
	f.tolist();			# create table for function
	f.lambda();			# convert to Lambda function	
	r := fTensor(f, g);		# tensor product of functions
	s := fCompose(r, u);		# function composition

Function Properties/Fields
--------------------------

.. code-block:: none

	f.domain();
	f.range();

Σ-SPL Diagonal Functions
++++++++++++++++++++++++

Definition
----------

.. code-block:: none

	f := fConst(4, 1.1);		# I4->R: i->1.1
	g := dOmega(8, 2);		# f_N,k : N -> C : i -> omega(N, k*i)
	h := FList(TReal, [1.1, 1.2, 1.4, 1.4]);	# table lookup
	u := FData([V(1.1), V(1.2), V(1.3), V(1.4)]);	# table lookup w/var

Operations on Functions
-----------------------

.. code-block:: none

	g.at(3);			# evaluate function
	g.at(3).ev();			# simplify the result
	f.tolist();			# create table for function
	g.lambda();			# convert to Lambda function	
	r := diagTensor(f, u);		# tensor product of functions
	r.tolist();			# what does diagTensor do?
	s := fCompose(r, L(16,4));	# composition of permutation and 
	s.tolist();			#   tensor product of functions

Function Properties/Fields
--------------------------

.. code-block:: none

	f.domain();
	f.range();
	u.var;
	u.var.t;
	u.var.value;
	
Defining Σ-SPL Symbolic Functions
+++++++++++++++++++++++++++++++++

fId and fBase
-------------

.. code-block:: none

	# spiral-core\namespaces\spiral\spl\perms2.gi
	Class(fId, PermClass, rec(
		domain := self >> self.params[1],
		range  := self >> self.params[1],
		def    := size -> Checked(IsPosInt0Sym(size), rec(size := size)),
		lambda := self >> let(i := Ind(self.params[1]), Lambda(i,i)),
		transpose := self >> self,
		isIdentity := True
	));

	Class(fBase, FuncClass, rec(
		abbrevs := [ var -> Checked(IsVar(var) or 
					 ObjId(var)=ind, [var.range, var]) ],
		def    := (N, pos) -> rec(),
		domain := self >> 1,
		range  := self >> self.params[1],
		print  := (self, i, is) >> Print(self.name, "(", 
			When(ObjId(self.params[2]) in [var, ind],
				Print(self.params[2]), PrintCS(self.params)), ")"),
		lambda := self >> let(i := Ind(1), Lambda(i, self.params[2]))
	));

fCompose
--------

.. code-block:: none

	# spiral-core\namespaces\spiral\spl\perms2.gi
	Class(fCompose, FuncClassOper, rec(
		domain := self >> Last(self._children).domain(),
		range := self >> self._children[1].range(),

		lambda := self >>
			FoldL1(List(self.children(), z->z.lambda()), 
				_rankedLambdaCompose),

		transpose := self >> self.__bases__[1](
			List(Reversed(self.children()), c->c.transpose())),

		isIdentity := self >> ForAll(self._children, IsIdentity),
	));

dOmega and dLin for Twiddle Diagonal
------------------------------------

.. code-block:: none

	Class(dLin, DiagFunc, rec(
		checkParams := (self, params) >> Checked(Length(params)=4,
		IsPosInt0Sym(params[1]), IsType(params[4]), params),
		lambda := self >> let(i:=Ind(self.params[1]), 
			a := self.params[2], b := self.params[3],
			Lambda(i, a*i+b).setRange(self.params[4])),
		range := self >> self.params[4],
		domain := self >> self.params[1]
	));

	Class(dOmega, DiagFunc, rec(
		checkParams := (self, params) >> Checked(Length(params)=2,
		IsPosIntSym(params[1]), IsIntSym(params[2]), params),
		lambda := self >> let(i:=Ind(), Lambda(i, 
			omega(self.params[1], self.params[2]*i)).setRange(TComplex)),
		range := self >> TComplex,
		domain := self >> TInt
	));













