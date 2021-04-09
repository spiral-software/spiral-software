.. _icode:

Abstract Code (icode)
=====================

.. code-block:: none

	program(
	   chain(
		  func(TVoid, "init", [  ],
			 chain()
		  ),
		  func(TVoid, "transform", [ Y, X ],
			 decl([ t57, t58, t59, t60, t61, t62, t63, t64 ],
				chain(
				   assign(t57, add(deref(X), deref(add(X, V(4))))),
				   assign(t58, add(deref(add(X, V(1))), deref(add(X, V(5))))),
				   assign(t59, sub(deref(X), deref(add(X, V(4))))),
				   assign(t60, sub(deref(add(X, V(1))), deref(add(X, V(5))))),
				   assign(t61, add(deref(add(X, V(2))), deref(add(X, V(6))))),
				   assign(t62, add(deref(add(X, V(3))), deref(add(X, V(7))))),
				   assign(t63, sub(deref(add(X, V(2))), deref(add(X, V(6))))),
				   assign(t64, sub(deref(add(X, V(3))), deref(add(X, V(7))))),
				   assign(deref(Y), add(t57, t61)),
				   assign(deref(add(Y, V(1))), add(t58, t62)),
				   assign(deref(add(Y, V(4))), sub(t57, t61)),
				   assign(deref(add(Y, V(5))), sub(t58, t62)),
				   assign(deref(add(Y, V(2))), sub(t59, t64)),
				   assign(deref(add(Y, V(3))), add(t60, t63)),
				   assign(deref(add(Y, V(6))), add(t59, t64)),
				   assign(deref(add(Y, V(7))), sub(t60, t63))
				)
			 )
		  )
	   )
	)

Basics
++++++

.. code-block:: none

	c1 := skip();				# NOP
	a := var.fresh_t("j", TInt);		# create a "fresh" variable
	c2 := assign(a, V(0));			# assignment
	c3 := assign(a, fcall("foo"));	# call to foo()
	c4 := chain(c1, c2, c3);		# basic block
	i := Ind(4);				# loop index
	c5 := loop(i, 4, c4);			# loop
	c6 := decl([a], c5);			# declare a variable

	PrintCode("", c6, SpiralDefaults);	# pretty print as C code

Internal Fields
+++++++++++++++

.. code-block:: none

	a.id; a.t;
	c2.exp; c2.loc;
	c3.cmds;
	c4.var; c4.range; c4.cmd;
	c5.vars; c5.cmd;


Arrays and Pointers
+++++++++++++++++++

.. code-block:: none

	X; 					# default input
	Y;					# default output
	X.t; Y.t;				# they are pointers
	deref(X + V(4));			# *(X+4)
	t := var.fresh_t("t", TArray(TReal, 4));	# double[4]
	nth(t, V(2));				# T[2]
	tcast(TInt, deref(X + V(4)));		# typecast

Constants
+++++++++

.. code-block:: none
	
	d := var.fresh_t("D", TReal);		# data variable
	v := V(1.1);				# value
	c := data(d, v, 			# declare constant
			  assign(nth(X, 0), nth(X, 0) * d)
		 );			


Expressions and Commands
++++++++++++++++++++++++

.. code-block:: none

	# spiral-core\namespaces\spiral\code\ir.gi
	Class(neg, AutoFoldExp, rec(
		ev := self >> -self.args[1].ev(),
		computeType := self >> let(t := self.args[1].t,
		Cond(IsPtrT(t),
			 t.aligned([t.alignment[1], 
				 -t.alignment[2] mod t.alignment[1]]), t)),
	));

	Class(chain, multiwrap, rec(
	   isChain := true,
	   flatten := self >> let(cls := self.__bases__[1],
		   CopyFields(self, rec(cmds := ConcatList(self.cmds,
			   c -> Cond(IsChain(c) and not IsBound(c.doNotFlatten),  
						 c.cmds, ObjId(c) = skip, [], [c]))))),
	   __call__ := meth(arg)
		   local self, cmds;
		   [self, cmds] := [arg[1], Flat(Drop(arg, 1))];
		   return WithBases(self, rec(
			   operations := CmdOps,
			   cmds       := Checked(ForAll(cmds, IsCommand), cmds)));
	   end
	));


Find All icode Expressions, Types, Commands, Etc.
+++++++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: none

	# all objects defined in spiral.code
	allobs := Dir(spiral.code);

	# filter function, checking that base classes are in a given list
	flt := bl -> (o -> (ForAny(bl, b -> b in let(ob := spiral.code.(o), 
		When(IsRec(ob) and IsBound(ob.__bases__), ob.__bases__, [])))));

	# list all expressions 
	Filtered(allobs, flt([Exp, AutoFoldExp]));

	# list all types 
	Filtered(allobs, flt([AtomicTyp]));

	# list all commands 
	Filtered(allobs, flt([Command]));
























