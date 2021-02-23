
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(TDA, _ApplyAllRulesTopDown, limit_cx);

IsRewriteRule := r -> IsRec(r) and IsBound(r.isRewriteRule) and r.isRewriteRule;

Class(RewriteRuleBase, rec(
	isRewriteRule := true,
	
	regularize := self >> Error("Not implemented. This method needs to convert ",
	"a special kind of rewrite rule into a regular RewriteRule"),
));

#F RewriteRule(<rewrite-rule>)	-- returns <rewrite-rule> object unchanged
#F RewriteRule(<from>, <to_func>) -- creates an unnamed rewrite rule
#F RewriteRule(<from>, <to_func>, <name>) -- creates a rewrite rule
#F
Class(RewriteRule, RewriteRuleBase, rec(
	regularize := self >> self,

	__call__ := meth(arg)
		local op, from, to, name, self;
	
		self := arg[1]; 
		arg := Drop(arg, 1);

		if Length(arg)=1 and IsList(arg[1]) then
			arg := arg[1];
		fi;
		if Length(arg)=1 then 
			return Checked(IsRewriteRule(arg[1]), arg[1]); 
		elif Length(arg)=2 then
			from := arg[1];
			to   := Checked(IsCallableN(arg[2], 1) or IsCallableN(arg[2], 2), arg[2]);
			name := "unnamed";
		elif Length(arg)=3 then
			from := arg[1];
			to   := Checked(IsCallableN(arg[2], 1) or IsCallableN(arg[2], 2), arg[2]);
			name := Checked(IsString(arg[3]), arg[3]); 
		else 
			Error("Rule requires 1, 2 or 3 arguments. See Doc(ARule)");
		fi;

		if IsList(from) then
			if Length(from)=0 then
				Error("Rule must have non-empty left-side");
			fi;
			op := from[1];
		else
			op := from;
		fi;
		if Is@(op) or not IsRec(op) or not IsBound(op.name) then
			if IsBound(op._target) and Length(op._target)=1 then 
				op := op._target[1];
			else 
				op := @;
			fi;
		fi;
		return WithBases(self, rec(op := op, from := from, to := to, name := name, operations := PrintOps));
	end,

	print := self >> PrintEval("$1($2, $3, \"$4\")", self.__name__, self.from, self.to, self.name),

));

# Rule(..), alias for RewriteRule, see Doc(RewriteRule)
#
Rule := arg -> ApplyFunc(RewriteRule, arg);

#F ARule(<rewrite-rule>) - returns <rewrite-rule> object unchanged
#F ARule(<op>, <from>, <to_func>) - creates an unnamed rewrite rule for associative <op>
#F ARule(<op>, <from>, <to_func>, <name>) - creates a rewrite rule for associative <op>
#F
#F Create a rewrite rule for an associative operator
#F
Class(AssociativeRewriteRule, RewriteRuleBase, rec(
	__call__ := meth(arg)
		local op, from, to, name, lhs, rhs, self;
		
		self := arg[1];
		arg := Drop(arg, 1);

		if Length(arg)=1 and IsList(arg[1]) then
			arg := arg[1];
		fi;
		
		if Length(arg)=1 then 
			return Checked(IsRewriteRule(arg[1]), arg[1]); 
		elif Length(arg)=3 then
			op   := Checked(IsClass(arg[1]), arg[1]);
			from := Checked(IsList(arg[2]),  arg[2]);
			to   := Checked(IsCallableN(arg[3], 1) or IsCallableN(arg[3], 2), arg[3]);
			name := "unnamed";
		elif Length(arg)=4 then
			op   := Checked(IsClass(arg[1]), arg[1]);
			from := Checked(IsList(arg[2]),  arg[2]);
			to   := Checked(IsCallableN(arg[3], 1) or IsCallableN(arg[3], 2), arg[3]);
			name := Checked(IsString(arg[4]), arg[4]); 
		else 
			Error("ARule requires 1, 3 or 4 arguments. See Doc(ARule)");
		fi;

		return WithBases(self, rec(op := op, from := from, to := to, name := name, operations := PrintOps));
	end,

	regularize := meth(self)
		local op, from, to, lhs, rhs;
		
		[op, from, to] := [self.op, self.from, self.to];
		lhs := [op, ...] :: from :: [...];
		if NumArgs(to)=1 then
			rhs := DetachFunc(Subst(
				function(e)
				local rch;
				rch := _children(e); 
					# ....left means O.left, where O is the '...' object, defined in rules.gi
				return _fromChildren(e, Concatenation(
				rch{[1 .. ....left]}, 
				$to(e),
				rch{[....right .. Length(rch)]}));
			end));
		else
			rhs := DetachFunc(Subst(
			function(e,cx)
				local rch;
				rch := _children(e); 
					 # ....left means O.left, where O is the '...' object, defined in rules.gi
				return _fromChildren(e, Concatenation(
					rch{[1 .. ....left]},
					$to(e,cx),
					rch{[....right .. Length(rch)]}));
			end));
		fi;
		return RewriteRule(lhs, rhs, self.name);
	end,

	print := self >> PrintEval("$1($2, $3, $4, \"$5\")", self.__name__, self.op, self.from, self.to, self.name),
));

# ARule(..), alias for AssociativeRewriteRule, see Doc(AssociativeRewriteRule)
#
ARule := arg -> ApplyFunc(AssociativeRewriteRule, arg);

_ARule_Transparent := (op, transp_ops, from, to) -> Checked(
	IsClass(op), IsList(transp_ops), ForAll(transp_ops, IsClass),
	IsList(from), IsCallableN(to, 1) or IsCallableN(to, 2),
	List(transp_ops, tr_op -> 
	ARule(op, List(from, f -> [tr_op, f]), 
		When(IsCallableN(to, 1), Subst(x -> List($to(x), y->$tr_op(y))),
							 Subst((x,cx) -> List($to(x, cx), y->$tr_op(y))))))
);


IsRuleSet := x -> IsRec(x) and IsBound(x.isRuleSet) and x.isRuleSet;

Class(RuleSet, rec(
	isRuleSet	:= true,
	rules		:= rec(),
	# counts how many time this RuleSet changed
	_changes	 := 0,
	# change index of compiled ruleset, need to recompile if
	# '_comp_id' is not equal to '_changes'
	_comp_id	 := -1,
	
	__transparent__ := [],

	addRules := meth(self, rules)
		local f, rr;
		Constraint(IsRec(rules));
		self.rules := ShallowCopy(self.rules);
		rules := ShallowCopy(rules);

		for f in UserRecFields(rules) do
			rr := rules.(f);
			Constraint(IsRewriteRule(rr) or (IsList(rr) and ForAll(rr, IsRewriteRule)));
			rr := When(IsList(rr), rr, [rr]);
			rr := ConcatList(rr, r -> 
			Cond(r _is AssociativeRewriteRule, 
				[r] ::
				ConcatList(Filtered(self.__transparent__, t->t[1]=r.op),
				t -> _ARule_Transparent(r.op, t[2], r.from, r.to)),
				[r]));
			rules.(f) := rr;
		od;

		MergeIntoRecord(self.rules, rules);
	
		# ruleset changed
		self._changes := self._changes + 1;
		return self;
	end,

	compileRule := meth(self, name, rule)
		local op, from, to_func, owner, head;
		owner := rule.owner;
		rule := rule.regularize();
		[op, from, to_func] := [rule.op, rule.from, rule.to];
		op := op.__name__;
		head := When(IsList(from) and from<>[], from[1], from);

		if op = "@" and IsRec(head) and IsBound(head._target) then
			DoForAll(head._target, t ->
				self.compileRule(name, CopyFields(rule, rec(op := t))));
		else
			if not IsBound(self._compiled.(op)) then
				self._compiled.(op) := [ CopyFields(rule, rec(name := name)) ]; #[from, to_func, name, owner] ];
			else
				Add(self._compiled.(op), CopyFields(rule, rec(name := name))); #[from, to_func, name, owner]);
			fi;
		fi;
	end,

	compile := meth(self)
		self._compiled := tab();
		# save the owner ruleset, it will be used to produce warnings of duplicate 
		# rule definitions, when dealing with merged rule sets
		DoForAll(self.rules, function(name, rule)
			local r;
			if IsRewriteRule(rule) then rule.owner := self; 
			elif IsList(rule) and not IsString(rule) then  for r in rule do r.owner := self; od;
			fi;
		end);

		DoForAll(self.rules, (name, rule) ->
			Cond(IsRewriteRule(rule), self.compileRule(name, rule),
				IsList(rule) and not IsString(rule), DoForAll(rule, r -> self.compileRule(name, r)), 0)
		);
		# remember change number
		self._comp_id := self._changes;
		return self;
	end,

	__call__ := (self, s) >> TDA(s, self, rec()),

	apply1 := (self, s) >> _ApplyAllRulesTopDown(s, limit_cx(1), self),

	get_changes  := self >> self._changes,
	get_comp_ids := self >> self._comp_id,

	changed  := self >> self.get_comp_ids() <> self.get_changes(),

	compiled := self >> When(IsBound(self._locked) or not self.changed(), self, self.compile())._compiled, 

	# locked ruleset doesn't check changes and doesn't recompile rules when asked for compiled() table
	locked   := self >> CopyFields(When(self.changed(), self.compile(), self), rec(_locked := true)),
));


Class(EmptyRuleSet, RuleSet);

AnonRuleSet := rules -> WithBases(RuleSet, rec()).addRules(rules);

Class(MergedRuleSet, RuleSet, rec(
	 _mergedConflictWarnings := [],
	 warnConflicts := false,
	 warnDuplicates := true,

	 __call__ := arg >> let(
		 self	 := arg[1],
		 rulesets := Drop(arg, 1),
		 comp_id  := Sum(rulesets, e -> e.get_changes())-1,
		 Checked(Length(rulesets) >= 1,
			 When( Length(rulesets)=1,
				 rulesets[1],
				 WithBases(self, rec(operations := PrintOps,
					 __call__ := RuleSet.__call__,
					 rulesets := rulesets,
					 _comp_id := comp_id))))),

	 get_changes  := self >> Sum(self.rulesets, e -> e.get_changes()),
	 get_comp_ids := self >> self._comp_id,

	 checkConflicts := meth(self, rules1, rules2, rulesets)
		 local conflicts, c;
		 conflicts := Intersection(UserRecFields(rules1), UserRecFields(rules2));
		 # make sure the origins of the rules are different
		 # origin == RuleSet where the rule first appears, i.e., not the MergedRuleSet
		 conflicts := Filtered(conflicts, c -> rules1.(c).owner.__name__ <> rules2.(c).owner.__name__);

		 # prevent duplicate warnings
		 if not self.warnDuplicates then
			 SubtractSet(conflicts, self._mergedConflictWarnings); fi;

		 if conflicts <> [] then
			 PrintErr("Warning: rewrite rule conflict when merging ", rulesets, "\n");
			 for c in conflicts do
				 Add(self._mergedConflictWarnings, c);
				 PrintErr(Blanks(9), c, "  ", rules1.(c).owner, "<->", rules2.(c).owner, "\n");
			 od;
		 fi;
	 end,

	 compile := meth(self)
		local rules, r;
		rules := rec();
		for r in self.rulesets do
			r.compile();
			if self.warnConflicts then
				self.checkConflicts(rules, r.rules, self.rulesets);
			fi;
			MergeIntoRecord(rules, r.rules);
		od;
		self.rules := rules;

		self._compiled := tab();

		DoForAll(self.rules, (name, rule) ->
			Cond(IsRewriteRule(rule), self.compileRule(name, rule),
				IsList(rule) and not IsString(rule), 
				DoForAll(rule, r -> self.compileRule(name, r)),0)
		);

		self._comp_id := self.get_changes();
		return self;
	end,

	print := self >> Print(self.name, "(", PrintCS(self.rulesets), ")")
));

RewriteRules := (rule_set, rules) -> rule_set.addRules(rules);


_LookupRules := (expr, ruleset) -> let(
	name := ObjId(expr).__name__, 
	R := When(IsBound(ruleset._locked), ruleset._compiled, ruleset.compiled()),
	When(IsBound(R.(name)), R.(name), []) :: When(IsBound(R.@), R.@, []));

#_AddRule2 := function(op, from, to_func)
#	 local rules;
#	 rules := _AddRule_old(op, from, to_func);
#	 if not IsBound(op._rules) then op._rules := BagAddr(rules); fi;
#end;
#_LookupRules2 := op -> When(IsBound(op._rules), BagFromAddr(op._rules), []);


ChkSPL := function(a,b, rule)
	local diff;
#	Error();
	if spiral.spl.IsSPL(a) and spiral.spl.IsSPL(b) then
		diff := spiral.code.InfinityNormMat(spiral.spl.MatSPL(a) -spiral.spl.MatSPL(b));
		if diff > 1E-4 then
			Print("Broken SPL rule: ", rule, "\nold:\n", a, "\nnew:\n", b);
			Error("bad SPL rule");
		fi;
	else
		return true;
	fi;
end;

RuleTrace := Ignore;
RuleStrategyTrace := Ignore;
RuleStrategyTiming := Ignore;
RuleStatus := Ignore;
RuleCheckSPL := false;

apply_rules := function(rules, expr, context)
	local rule, lhs, rhs, old;
	for rule in rules do
		lhs := rule.from;
		rhs := rule.to;
		while PatternMatch(expr, lhs, context) and context.rlimit <> 0 do
			context.rlimit := context.rlimit - 1;
			context.applied := context.applied + 1;
			old := Copy(expr);
			RuleTrace(rule);
			RuleStatus(rule, "OLD: ", [expr, "\n"]);
			if RuleCheckSPL then old := Copy(expr); fi;
			if NumArgs(rhs)=1 then expr := rhs(expr);
			else expr := rhs(expr, context);
			fi;
			RuleStatus(rule, "NEW: ", [expr, "\n"]);
			trace_log.addRewrite(rule.name,old,expr, []);
			if RuleCheckSPL then ChkSPL(old, expr, rule); fi;
		od;
	od;
	return expr;
end;

# non-iterative version (no more: while PatternMatch(...) do ...)
apply_rules_ni := function(rules, expr, context)
	local rule, lhs, rhs, old;
	for rule in rules do
		lhs := rule.from;
		rhs := rule.to;
		if PatternMatch(expr, lhs, context) and context.rlimit <> 0 then
			context.rlimit := context.rlimit - 1;
			context.applied := context.applied + 1;
			old := Copy(expr);
			RuleTrace(rule);
			RuleStatus(rule, "OLD: ", [expr, "\n"]);
			if NumArgs(rhs)=1 then expr := rhs(expr);
			else expr := rhs(expr, context);
			fi;
			RuleStatus(rule, "NEW: ", [expr, "\n"]);
			trace_log.addRewrite(rule.name,old,expr, []);
		fi;
	od;
	return expr;
end;

cx_enter := function(cx, expr)
	local opname;
	if IsRec(expr) and IsBound(expr.name) then
	opname := expr.name;
	if not IsBound(cx.(opname)) then cx.(opname) := [ expr ];
	else Add(cx.(opname), expr); fi;
	fi;
	Add(cx.parents, expr);
end;

cx_leave := function(cx, expr)
	# unupdate context back to original
	if IsRec(expr) and IsBound(expr.name) then
	RemoveLast(cx.(expr.name), 1);
	fi;
	RemoveLast(cx.parents, 1);
end;

map_children := function(expr, to_func)
	local ch, i;
	ch := _children(expr);
	for i in [1..Length(ch)] do
		_setChild(expr, i, to_func(ch[i]));
	od;
	return expr;
end;

map_children_safe := function(expr, to_func)
	local ch, i;
	ch := ShallowCopy(_children(expr));
	if ch=[] then return expr; fi;
	for i in [1..Length(ch)] do
		ch[i] := to_func(ch[i]);
	od;
	return _fromChildren(expr, ch);
end;

# Rule application context fields
#	parents - list of parents of current node,
#			  immediate parent is last, farthest parent is first,
#			  this field is updated by apply_rules()
#	rlimit  - maximum number of rules to apply, this is a parameter
#			  to apply_rules()
#	applied - number of rules applied so far, updated by apply_rules()
#
#	<objid> - for each <objid> list of parents with that id only,
#			  immediate parent is last. For instance Last(context.ISum)
#			  is the enclosing ISum.
#

_meth_cx_isInside := (self, oid) >> let(nam := Cond(IsString(oid), oid, oid.__name__),
	IsBound(self.(nam)) and self.(nam)<>[]
);

# construct an empty context, but set a limit on the number of applied rules
limit_cx := lim -> tab(
	isInside := _meth_cx_isInside,
	parents := [], rlimit := lim, applied := 0
);

# construct an empty initial context
empty_cx := () -> tab(
	isInside := _meth_cx_isInside,
	parents := [], rlimit := -1, applied := 0
);


_ApplyAllRulesTopDown := function(expr, context, ruleset)
	if (not IsRec(expr) or not IsBound(expr.name)) and (not IsList(expr) or BagType(expr) in [T_STRING, T_RANGE]) then
		return expr;
	fi;
	if IsBound(ruleset.__avoid__) and ObjId(expr) in ruleset.__avoid__ then
		return expr;
	fi;
	# apply rules
	expr := apply_rules(_LookupRules(expr, ruleset), expr, context);
	# recurse
	# NOTE: do not enter context if expr has no children!
	cx_enter(context, expr);
	expr := map_children(expr, c -> _ApplyAllRulesTopDown(c, context, ruleset));
	cx_leave(context, expr);
	return expr;
end;

ApplyAllRulesTopDown := (expr, ruleset) ->
	_ApplyAllRulesTopDown(expr, empty_cx(), ruleset.locked());

_ApplyAllRulesBottomUp := function(expr, context, ruleset)
	if (not IsRec(expr) or not IsBound(expr.name)) and (not IsList(expr) or BagType(expr) in [T_STRING, T_RANGE]) then
		return expr;
	fi;
	if IsBound(ruleset.__avoid__) and ObjId(expr) in ruleset.__avoid__ then
		return expr;
	fi;
	# recurse
	cx_enter(context, expr);
	expr := map_children(expr, c -> _ApplyAllRulesBottomUp(c, context, ruleset));
	cx_leave(context, expr);
	# apply rules
	expr := apply_rules(_LookupRules(expr, ruleset), expr, context);
	return expr;
end;

ApplyAllRulesBottomUp := (expr, ruleset) ->
	_ApplyAllRulesBottomUp(expr, empty_cx(), ruleset.locked());

_apply_strategy_step := function(expr, rset, apply_func, opts)
	if IsFunc(rset) then
		if NumArgs(rset)=2 then
			expr := rset(expr, opts);
		else
			expr := rset(expr);
		fi;
	else
		expr := apply_func(expr, rset.locked(), opts); 
	fi;
	return expr;
end;

##
## ApplyStrategy(<expr>, <list-of-rulesets>, <apply_func>, <opts>)
##
## <expr>			 - expression to transform
## <list-of-rulesets> - list of rulesets to apply
## <apply_func>	   - of type (expr, rset) -> expr
##
ApplyStrategy := function(expr, strategy, apply_func, opts)
	local rset, l, r, i, t;
	# a hack to make this function reentrant
	# (e.g. to make it possible to call this function from within rhs of a rule)
	l := ....left; r := ....right; i := 1;
	for rset in strategy do
		RuleStrategyTrace(i, rset, expr);
	[expr, t] := UTimedAction(_apply_strategy_step(expr, rset, apply_func, opts));
		RuleStrategyTrace(i, "DONE", expr);
	RuleStrategyTiming(i, t);
		i := i + 1;
	od;
	....left := l;
	....right := r;
	return expr;
end;

TD := ApplyAllRulesTopDown;
BU := ApplyAllRulesBottomUp;

TDA := function(s, ruleset, opts)
	local cx;
	cx := empty_cx();
	cx.opts := opts;
	cx.applied := 1;
	while cx.applied > 0 do
		cx.applied := 0; 
		s := _ApplyAllRulesTopDown(s, cx, ruleset);
	od;
	return s;
end;

BUA := function(s, ruleset, opts)
	local cx;
	
	cx := empty_cx();
	cx.opts := opts;
	cx.applied := 1;
	while cx.applied > 0 do
		cx.applied := 0;
		s := _ApplyAllRulesBottomUp(s, cx, ruleset);
	od;
	return s;
end;

UntilDone := TDA;

Rewrite := function(expr, rset, opts)
	local res;
	
	if IsList(rset) then
		return FoldL(rset, (a,b)->Rewrite(a, b, opts), expr); 
	fi;

	trace_log.beginRuleset(rset, expr);
	res := UntilDone(expr, rset, opts);	
	trace_log.endRuleset(rset, res);
		
	return res;
end;


#F TDA_1by1(<s>, <expr>, <after_each_rule>)
#F
#F This function applies rules in a top-down fashion to <s> until
#F it converges, just like TDA.  However after each application of the
#F rule it runs after_each_rule(s) on the newly transformed <s>.
#F This allows stepwise verification among other things.
#F
#F Example:
#F	r := RandomRuleTree(DFT(4));
#F	# following line (unlike SumsRuleTree) gives non-simplified Sigma-SPL
#F	s := SumsSPL(SPLRuleTree(r));
#F	TDA_1by1(s, MergedRuleSet(RulesSums, RulesDiag, RulesFuncSimp),
#F				x->PrintLine(MatSPL(x)-MatSPL(DFT(4))));
#F
TDA_1by1 := function(s, ruleset, after_each_rule)
	local cx;
	
	cx := limit_cx(1); cx.applied := 1;
	while cx.applied > 0 do
		cx := limit_cx(1); cx.applied := 0;
		s := _ApplyAllRulesTopDown(s, cx, ruleset);
		after_each_rule(s);
	od;
	return s;
end;
