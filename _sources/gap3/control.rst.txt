.. _control_flow:

Control Flow
============

Procedures and Procedural Functions
+++++++++++++++++++++++++++++++++++

.. code-block:: none

	add1 := function(n)
	   local m;
	   m := n + 1;
	   return m;
	end;

	vaarg:= function(arg) return Length(arg); end;
	vaarg(1, 2); vaarg(1, 2, 3);	# variable number of arguments

Loops
+++++

.. code-block:: none

	for i in [1..5] do Print(i); od;
	i:= 5; while i > 0 do Print(i); i := i-1; od;
	DoForAll([1..5], PrintLine);

Conditionals
++++++++++++

.. code-block:: none

	a := 3;
	b := When(a < 3, a+1, 2*a);
	if a < 3 then c := a+1; else c := 2*a; fi;
	c;
	a -> Cond(a<0, 0, a>10, 20, 2*a);	# functional switch statement











