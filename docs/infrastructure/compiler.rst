.. _compiler:


Compiler
========


Compiling from Σ-SPL to icode
+++++++++++++++++++++++++++++

Top-Level Flow
--------------

.. code-block:: none

	opts := SpiralDefaults;
	s := SumsRuleTree(RandomRuleTree(DFT(4), opts), opts);
	c := CodeSums(s, opts);

Basic Block Compilation
-----------------------

.. code-block:: none

	# the actual code generator is a configuration option
	opts.codegen;
	# What happens in CodeSums
	DefaultCodegen(Formula(s), Y, X, opts);

	# without Formula() the basic block compiler is not run
	c := DefaultCodegen(s, Y, X, opts);
	# invoke the basic block compiler
	Compile(c, opts);

	# Compile calls a number of compile strategies
	opts.compileStrategy;
	for i  in [ 1 .. Length(opts.compileStrategy) ] do
		c := let(stage := opts.compileStrategy[i],  
			When(IsCallableN(stage, 2), stage(c, opts), stage(c)));
	od;
	c;


Basic Block Compiler
++++++++++++++++++++

Top-Level Flow
--------------

.. code-block:: none

	opts := SpiralDefaults;
	s := SumsRuleTree(RandomRuleTree(DFT(8), opts), opts);
	c := DefaultCodegen(s, Y, X, opts);
	Compile(c, opts);

Basic Block Compilation, Stage by Stage
---------------------------------------

.. code-block:: none

	c := Compile.pullDataDeclsRefs(c);
	c := Compile.fastScalarize(c);
	c := UnrollCode(c);
	c := FlattenCode(c);
	c := UntangleChain(c);
	c := CopyPropagate.initial(c, opts);
	c := HashConsts(c, opts);
	c := MarkDefUse(c);
	c := BinSplit(c, opts);
	c := MarkDefUse(c);
	c := CopyPropagate(c, opts);
	c := BinSplit(c, opts);
	c := FixValueTypes(c);
	c := Compile.declareVars(c);
	PrintCode("dft8", c, opts);


A Closer Look at Compiler Stages
++++++++++++++++++++++++++++++++

Implementation of Important Stages
---------------------------------------

.. code-block:: none

	Print(Compile.pullDataDeclsRefs);
	Print(Compile.fastScalarize);
	UnrollCode;
	FlattenCode;
	UntangleChain;
	Print(CopyPropagate.copyProp);
	Print(CSE.__call__);
	Print(HashConsts);
	Print(_MarkDefUse);
	Print(BinSplit.__call__);
	FixValueTypes;
	Compile.declareVars;

















