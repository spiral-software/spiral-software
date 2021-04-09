.. _values_types:

Values and Types
================


Value Wrapper and Object
++++++++++++++++++++++++

.. code-block:: none

	V(1);
	v := V(1.1);
	_unwrap(v);
	v.v;
	v.t;

Scalar Data Types
+++++++++++++++++

.. code-block:: none

	Value(TReal, 1.0);
	Value(TDouble, 1.0);
	Value(T_Real(32), 1.0);
	Value(T_Real(64), 1.0);
	Value(TInt, 1);
	Value(T_Int(32), 1);
	Value(TComplex, Cplx(0, 1));

Array Data Types
+++++++++++++++++

.. code-block:: none

	t := TArray(TReal, 4);			# double[4]
	v := Value(t, [1,2,3,4]);		# double[4] = {1.0,2.0,3.0,4.0}
	mt := TArray(TArray(TReal, 4), 4);
	Value(mt, Replicate(4, v));		# initialize a 4x4 matrix

Pointers
++++++++

.. code-block:: none

	TPtr(TReal);				# *double

Variables
+++++++++

.. code-block:: none

	i := Ind();				# integer index
	j := Ind(4);				# index, 0 <= j < 4
	k := var.fresh_t("j", TInt);		# create a "fresh" variable
	var.table.(v.id);			# global variable table

Expressions
+++++++++++

.. code-block:: none

	a := var.fresh_t("a", TReal);		# a few real variables
	b := var.fresh_t("b", TReal);	
	c := var.fresh_t("c", TReal);	
	a + b;					# + is overloaded
	add(a, V(2.0));
	a + 2;
	e := add(a, b);				# use add function
	e.args;					# the operands
	e.t;					# expressions carry a type
	g := add(a, b, c);			# not just binary
	h := add(a, mul(b, c));			# expressions
	k := add(neg(a), mul(b, c, V(1.1)));	# expressions
	mul(V(1.0), V(2.0));			# evaluates at construction
	sub(V(1), V(2.0)).t;			# type unification

Comparisons
+++++++++++

.. code-block:: none

	i := Ind();				# an integer variable
	j := Ind();				# an integer variable
	eq(i, V(1));				# i == 1
	leq(i, j);				# comparison: <=
	geq(i, j);				# comparison: >=
	lt(i, j);				# comparison: <
	gt(i, j);				# comparison: >
	cond(leq(i, 0), 0, i);	 		# C ? : conditional assignment
	min(i, j);				# minimum
	max(i, j);				# maximum

Error Handling
++++++++++++++

.. code-block:: none

	i := Ind();				# an integer variable
	errExp(TInt);				# illegal value of type int
	noneExp(TInt);				# undefined value, type double

Lambda Functions
++++++++++++++++

Definition
----------

.. code-block:: none

	i := Ind(4);				# variable with range
	f := Lambda(i, i+1);			# i -> i+1
	j := Ind(4);				# variable with range
	g := Lambda([i, j], imod(i*j, 4));	# function in 2 variables

Operations on Functions
-----------------------

.. code-block:: none

	f.at(0);				# evaluate function
	f.tolist();				# create table for function
	k := Ind(4);
	Lambda(k, g.at(k, V(1)));		# partially evaluate g(.,1)
	m := Ind(4);
	h := Lambda(m, 2*m);			# i -> 2*i
	LambdaCompose(f, h);			# function composition

Function Properties/Fields
--------------------------

.. code-block:: none

	f.domain();
	f.range();
	f.t;
	f.vars;
	f.expr;



