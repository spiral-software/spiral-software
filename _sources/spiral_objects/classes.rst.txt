.. _classes:


Classes
=======

Class Definition
++++++++++++++++

Uses ``self`` (class and instance are both records).

.. code-block:: none

	Class(A, rec(
	   __call__ := meth(self, v) return WithBases(self, # copy base fields
		   rec(a := v,		# initialize values
		   operations := rec(	# need that to print with state
			  Print := self >> Print("A(", self.a, ")")))); end,
	   a := 1,			# state of the object
	   geta := self >> self.a,	# access functions: get
	   seta := meth(self, v) self.a := v; end, # access functions: set
	));


Class Operations
++++++++++++++++

.. code-block:: none

	a := A(3);		# instantiate A by calling constructor
	a.seta(a.geta()+1);	# use access functions
	a.a;			# updated value of a.a
	a.name;			# Class name
	a.__bases__;		# base classes
	a.__bases__[1].a;	# get value of a from the base class
	Unbind(a.a);
	a.a;			# refers to the base class value



