.. _rewrite:


Rewriting
=========

.. only:: html

   .. contents::

Pattern Matching
++++++++++++++++

Collect
-------

.. code-block:: none

	opts := SpiralDefaults;
	s := SumsRuleTree(RandomRuleTree(DFT(8), opts), opts);
	c := CodeSums(s, opts);

	Collect(s, Scat);		# get list of scatter operations
	Set(Collect(s, Value));	# get all unique values


Simple Patterns
---------------

.. code-block:: none

	Collect(c, @(1, [add, sub, neg, mul]));	# get all arith ops...
	Collect(c, @(1, [add, sub, neg, mul], e->e.t=TReal)); #...on reals

	List(Collect(s, @(1, ISum)), e->e.var);	# all loop variables
	Set(Collect(s, @@(1, Value, 	# all values inside Blk objects
		(e, cx)->IsBound(cx.Blk) and Length(cx.Blk) > 0)));


Subtree Patterns
----------------

.. code-block:: none

	Collect(c, [deref, add, sub]);
	Collect(c, [mul, @(1), sub]);
	Collect(c, [mul, Value, ...]);
	Collect(c, [mul, @(1), [sub, deref, @(2)]]);
	Collect(c, [mul, @(1), [sub, @(2, deref, e->X in e.free()), @(3)]]);


Substitutions
+++++++++++++

SubstTopDown/SubstBottomUp
--------------------------

.. code-block:: none

	opts := SpiralDefaults;
	c := CodeSums(SumsRuleTree(RandomRuleTree(DFT(8), opts), opts), opts);

	# Ordered substitution: traversal order can matter greatly
	SubstTopDown(Copy(c), @(1, Value, e->e.v=1), e->V(25));
	SubstBottomUp(Copy(c), @(1, Value, e->e.v=1), e->V(-25));

Variable Substitutions
----------------------

.. code-block:: none

	vars := Collect(c, @(1, var, e->e.t=TReal));	# all the real variables
	SubstVars(Copy(c), rec((vars[1].id) := V(1.1)));	# substitute one

	# record of assignment of consecutive numbers to all real variables
	substrec := FoldR(Zip2(vars, [1..Length(vars)]), 
		(a,b) -> CopyFields(a, rec((b[1].id) := V(b[2]))), rec());
	SubstVars(Copy(c), substrec);	# substitute them

	# loop unrolling example
	i := Ind(4);
	c2 := loop(i, 4, assign(nth(X, i), i));	# loop to be unrolled
	chain(List(c2.range, 	# chain of partially evaluated loop iterations
		i->SubstVars(Copy(c2.cmd), rec((c2.var.id) := V(i)))));


Rules
+++++

Simple Rules
------------

.. code-block:: none

	Rule([neg, [neg, @1]], e -> @1.val);
	Rule([add, Value, Value], 
		e->Value.new(e.args[1].t, e.args[1].v + e.args[2].v));
	Rule([im, [conj, @(1)]], x->-im(@(1).val));
	Rule([IF, @(1), skip, skip], e -> skip());

	Rule([RC, @(1, Compose)], e -> Compose(List(@(1).val.children(), RC)));
	Rule([RC, @(1, Gath)], e -> Gath(fTensor(@(1).val.func, fId(2))));

	Rule([Tensor, ..., @(1,O), ...], e -> O(Rows(e), Cols(e)));

Complex Rules
-------------

.. code-block:: none

	_v0none := @(0).target([ Value, noneExp ]).cond(
		(e) -> Cond(ObjId(e) = noneExp, true, isValueZero(e)));
	_0noneOrZero :=(t) -> When(
		ObjId(@(0).val) = noneExp, noneExp(t), t.zero());
	Rule([mul, ..., _v0none, ...], e -> _0noneOrZero(e.t));
	Rule([@@(0,mul,(e,cx)->IsBound(cx.nth) and cx.nth<>[]), @(1), @(2,add)],
		 e -> ApplyFunc(add, List(@(2).val.args, a->@(1).val * a)));
	Rule( [im, [mul, [cxpack, @(1), @(2)], [conj, [cxpack, 
		@(3).cond(x->x=@(1).val), @(4).cond(x->x=@(2).val)]]]],
		e -> e.t.zero() );


Associative Rules
+++++++++++++++++

Simple Rules
------------

.. code-block:: none

	ARule(add, [ @(1,add) ], e -> @1.val.args);
	ARule(fTensor,  [@(1, fTensor) ], e -> @(1).val.children());
	ARule(fCompose, [@(1), fId ], e -> [@(1).val]);

	ARule(Compose, [ @(1, Prm), @(2, Prm) ], 
		 e -> [ Prm(fCompose(@(2).val.func, @(1).val.func)) ])
	ARule(Compose, [ @(1, Gath), @(2, [Gath, Prm]) ],
		 e -> [ Gath(fCompose(@(2).val.func, @(1).val.func)) ]);

Complex Rules
-------------

.. code-block:: none

	ARule(leq, [@(1, Value, x->x.v<=0), [@(0,mul), @(2, Value, x->x.v>0), 
		@(3,var,IsLoopIndex)]], e -> [@(0).val]);

	ARule( Compose, [ @(1, [Prm, Scat, ScatAcc, Conj, ConjL, ConjR, ConjLR]),
		@(2, [RecursStep, Grp, BB, SUM, ISum, Data, COND]) ],
		e -> [ CopyFields(@(2).val, rec(
				 _children :=  List(@(2).val._children, c -> @(1).val * c),
				 dimensions := [Rows(@(1).val), Cols(@(2).val)] )) ]);

	ARule(fCompose, [ @(1, L), [ @(3, fTensor),
		@(2).cond(e->range(e) = @(1).val.params[2] and domain(e)=1), ... ] ],
		e->[ fTensor(Copy(Drop(@(3).val.children(), 1)), Copy(@(2).val)) ] 


Rule Sets
+++++++++

Define a Rule Set
-----------------

.. code-block:: none

	# spiral-core\namespaces\spiral\code\sreduce.gi
	Class(RulesStrengthReduce, RuleSet);
	RewriteRules(RulesStrengthReduce, rec(
		 leq_single := Rule([leq, @(1)], e-> V_true),
		 add_assoc  := ARule(add, [ @(1,add) ], e -> @1.val.args),
	# hundreds of rules
	));

Add Rules to Existing Rule Set
------------------------------

.. code-block:: none

	# somewhere else in the source code
	RewriteRules(RulesStrengthReduce, rec(	# add more rules
		logic_single := Rule([@(1, [logic_and, logic_or]), @1], e->@1.val)
	));

Using Rule Sets
---------------

.. code-block:: none

	RulesStrengthReduce.rules.leq_single;
	opts := SpiralDefaults;
	s := SPLRuleTree(RandomRuleTree(DFT(8), opts)).sums();
	s := Rewrite(s, RulesSums, opts);
	s := Rewrite(s, RulesDiag, opts);
	s := RulesDiagStandalone(s);


Rule Strategies
+++++++++++++++

Define a Rule Strategy
----------------------

.. code-block:: none

	# spiral-core\namespaces\spiral\code\sreduce.gi
	LibStrategy := [ StandardSumsRules, HfuncSumsRules ]; 

Combining Rule Sets
-------------------

.. code-block:: none

	StandardSumsRules := MergedRuleSet(
		RulesSums, RulesFuncSimp, RulesDiag, RulesDiagStandalone, 
		RulesStrengthReduce, RulesRC, RulesII, OLRules
	);

Use of Rule Strategies
----------------------

.. code-block:: none

	SpiralDefaults.formulaStrategies.sigmaSpl;
	SpiralDefaults.formulaStrategies.rc;
	SReduce := (c,opts) -> 	# handy shortcut
		ApplyStrategy(c, [RulesStrengthReduce], BUA, opts);

	opts := SpiralDefaults;
	s := SumsRuleTree(RandomRuleTree(DFT(4), opts), opts);
	c := DefaultCodegen(s, Y, X, opts);
	c := SubstTopDown(c, @(1, loop), e->e.unroll());
	Collect(c, mul);
	c := SReduce(c, opts);
	Collect(c, mul);


General Mechanics
+++++++++++++++++

Interface for Rewriting
-----------------------

.. code-block:: none

	# spiral-core\namespaces\spiral\code\ir.gi
	Class(deref, nth, rec(
		__call__ := (self, loc) >> Inherited(loc, TInt.value(0)),
		rChildren := self >> [self.loc],
		rSetChild := rSetChildFields("loc"),
	));
	deref.from_rChildren;

Unifying Interface Across All Rewritable Objects
------------------------------------------------

.. code-block:: none

	c := deref(X); 	# code objects
	c.rChildren(); 	# get rewritable fields
	c.rSetChild(1, Y);	# change a rewritable field

	DFT.rChildren;		# Transform level
	RandomRuleTree(DFT(4), SpiralDefaults).rChildren;	# ruletree level
	F.rChildren;		# SPL level
	L.rChildren;		# Permutations
	Gath.rChildren;	# Sigma-SPL
	Lambda.rChildren;	# Lambda function
	fId.rChildren;		# symbolic functions
	add.rChildren;		# expressions
	T_Real.rChildren;	# data types


Implementing Recursive Descent: Visitors 
++++++++++++++++++++++++++++++++++++++++

Simple Example
--------------

.. code-block:: none

	# spiral-core\namespaces\spiral\rewrite\visitor.gi
	Class(LispGen, Visitor, rec(
		add := (self, o) >> Print("(+ ", self(o.args[1]), " ", 
			self(o.args[2]), ")"),
		mul := (self, o) >> Print("(* ", self(o.args[1]), " ", 
			self(o.args[2]), ")"),
		sub := (self, o) >> Print("(- ", self(o.args[1]), " ", 
			self(o.args[2]), ")"),
		var := (self, o) >> Print("(var ", o.id, ")"),
		Value := (self, o) >> Print("(value ", o.v, ")")
	));
	LispGen(4*X+2);

Visitors Used in Standard Translation Flow
------------------------------------------

.. code-block:: none

	opts := SpiralDefaults;
	opts.sumsgen;
	DefaultSumsGen;
	opts.codegen;
	DefaultCodegen;
	opts.unparser;
	CUnparser;

		
SumsGen
+++++++

The DefaultSumsGen Visitor
--------------------------

.. code-block:: none

	# spiral-core\namespaces\spiral\sigma\sumsgen.gi
	# all the fields
	Filtered(RecFields(DefaultSumsGen), i->not IsSystemRecField(i));
	Print(DefaultSumsGen.__call__);

	# the recursive definitions needed for DFT
	DefaultSumsGen.Compose;
	Print(DefaultSumsGen.Tensor);
	DefaultSumsGen.I;
	DefaultSumsGen.F;
	DefaultSumsGen.Diag;
	DefaultSumsGen.L;

Visitors Used in Standard Translation Flow
------------------------------------------

.. code-block:: none

	opts := SpiralDefaults;
	s := SPLRuleTree(RandomRuleTree(DFT(8), opts));
	SumsSPL(s, opts);
	opts.sumsgen(s, opts);

	# legacy and backwards compatibility framework
	F(2).sums();
	Tensor(F(2), I(2)).sums();


CodeGen
+++++++

The DefaultCodegen Visitor
--------------------------

.. code-block:: none

	# spiral-core\namespaces\spiral\compiler\codegen.gi
	# all the fields
	Filtered(RecFields(DefaultCodegen), i->not IsSystemRecField(i));
	Print(DefaultCodegen.__call__);

	# Some of the fields
	Print(DefaultCodegen.Formula);
	Print(DefaultCodegen.Compose);
	DefaultCodegen.ISum;
	Print(DefaultCodegen.Gath);
	Print(DefaultCodegen.Scat);
	DefaultCodegen.Diag;

Visitors Used in Standard Translation Flow
------------------------------------------

.. code-block:: none

	opts := SpiralDefaults;
	s := SumsRuleTree(RandomRuleTree(DFT(8), opts), opts);
	# only translate Sigma-SPL to icode
	opts.codegen(s, Y, X, opts);
	# also invoke the basic block compiler
	opts.codegen(Formula(s), Y, X, opts);


C Pretty Printer
++++++++++++++++

The CUnparser Visitor
---------------------

.. code-block:: none

	# spiral-core\namespaces\spiral\compiler\unparse.gi
	# all the fields
	Filtered(RecFields(CUnparser), i->not IsSystemRecField(i));
	Filtered(RecFields(CUnparserBase), i->not IsSystemRecField(i));
	Print(CUnparser.gen);

	# Some of the fields
	Print(CUnparser.loop);
	CUnparser.deref;
	CUnparser.add;
	CUnparser.Value;
	Print(CUnparser.decl);
	CUnparser.chain;

Visitors Used in Standard Translation Flow
------------------------------------------

.. code-block:: none

	opts := SpiralDefaults;
	c := CodeRuleTree(RandomRuleTree(DFT(8), opts), opts);
	# Print full header etc.
	PrintCode("dft8", c, opts);
	# unparser needs opts as context
	Unparse(c.cmds[1].cmds[2].cmd, 
		CopyFields(CUnparser, rec(opts:=opts)), 0, 1);

















