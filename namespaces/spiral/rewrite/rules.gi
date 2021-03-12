
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


rSetChildFields := arg -> Checked(IsList(arg), 
	Subst(
		meth(self, n, newChild)
			if not n in [1..numfields] then 
				Error("<n> must be in ", [1..numfields]);
			else 
				self.(RecName(fields[n])) := newChild;
			fi;
		end,
		numfields => Length(arg),
		fields => arg)
);

rSetChildFieldsCh := arg -> Checked(IsList(arg), 
	Subst(
		meth(self, n, newChild)
			if not n in [1..chCount+numfields] then
				Error("<n> must be in ", [1..chCount+numfields]);
			elif n in [1..chCount] then
				self._children[n] := newChild;
			else 
				self.(RecName(fields[n-chCount])) := newChild;
			fi;
		end,
		numfields => Length(arg)-1,
		chCount => arg[1],
		fields => Drop(arg, 1))
);

#  rChildrenFieldsCh: gives rChildren method where rChildren is self.children() :: [self.field1, self.field2, ... ],
#  arg[1] is "field1", arg[2] is "field2" etc.
#  ex: rChildren := rChildrenFieldsCh("stride", "v")
#
rChildrenFieldsCh := arg -> Checked(IsList(arg), 
	Subst( (self) >> self._children :: List([1..numfields], i -> self.(RecName(fields[i]))),
		numfields => Length(arg),
		fields => arg )
);

rSetChildFieldsF := arg -> Checked(IsList(arg), Length(arg)>1, 
	Subst(
		meth(self, n, newChild)
			if not n in [1..numfields] then
				Error("<n> must be in ", [1..numfields]);
			else 
				self.(RecName(fields[n])) := func(newChild);
			fi;
		end,
		numfields => Length(arg)-1,
		fields => Drop(arg, 1),
		func => arg[1])
);

Local._print := function(arg)
	local a;
	for a in arg do 
		if IsRec(a) and IsBound(a.__bases__) then
			Print(a.__bases__[1]);
		else
			Print(a);
		fi;
	od;
end;

Local._children := function(obj)
	if IsRec(obj) then
		if not IsBound(obj.rChildren) then
			return [];
		else
			return obj.rChildren();
		fi;
	elif BagType(obj) in [T_STRING, T_RANGE] then 
		return [];
	elif IsList(obj) then
		return obj;
	else
		return [];
	fi;
end;

Local._setChild := function(obj, n, newchild)
	if IsRec(obj) then
		if not IsBound(obj.rSetChild) then 
			Error("<obj> must have rSetChild(<n>, <newChild>) method");
		fi;
		obj.rSetChild(n, newchild);
	elif IsList(obj) then
		obj[n] := newchild;
	else
		Error("<obj> must be a record or a list");
	fi;
end;

Local._fromChildren := function(obj, newchildren)
	if IsRec(obj) then
		if not IsBound(obj.from_rChildren) then 
			Error("<obj> must have from_rChildren(<newChildren>) method");
		fi;
		return obj.from_rChildren(newchildren); 
	elif BagType(obj) in [T_STRING, T_RANGE] then 
		return  obj;
	elif IsList(obj) then
		return newchildren;
	elif newchildren=[] then
		return obj;
	else
		return Error("<obj> must be a record or a list");
	fi;
end;

_PrintShape := function(obj, maxDepth, printCutoff, info, i, is)
	if maxDepth > 0 then 
		Print(Blanks(i));
		_print(obj, " ", info(obj), "\n");
		DoForAll(_children(obj), c -> _PrintShape(c, maxDepth-1, printCutoff, info, i+is, is));
	elif printCutoff then
		Print(Blanks(i));
		Print("...\n");
	fi;
end;

PrintShape := (obj, maxDepth) -> Checked(IsInt(maxDepth) and maxDepth >= 0, 
	_PrintShape(obj, maxDepth, true, x->"", 0, 4));

# info is a function : obj -> what to print
PrintShapeInfo := (obj, maxDepth, info) -> Checked(IsInt(maxDepth) and maxDepth >= 0, 
	_PrintShape(obj, maxDepth, true, info, 0, 4));

PrintShape2 := (obj, maxDepth) -> Checked(IsInt(maxDepth) and maxDepth >= 0, 
	_PrintShape(obj, maxDepth, false, x->"", 0, 4));

Local.id := ObjId; # ObjId now handles lists

#F ShapeObject(<obj>)
#F   Convert an object to a list-based pattern,
#F   i.e. add(X, Y) becomes [add, var, var]
#F 
ShapeObject := obj -> Cond(
	BagType(obj) in [T_STRING,T_RANGE], obj,
	IsRec(obj) or IsList(obj), let(
		res := Concatenation([ObjId(obj)], List(_children(obj), ShapeObject)),
		When(Length(res)=1, res[1], res)),
	obj
);

#F ShapeObject@(<obj>, <func>)
#F 
#F Same as ShapeObject(obj) but takes a boolean function <func>, and objects
#F for which <func> returns true are replaced by @.
#F
ShapeObject@ := (obj, func_replace_by_wildcard) -> Cond(
	BagType(obj) in [T_STRING,T_RANGE], obj,
	func_replace_by_wildcard(obj), @,
	IsRec(obj) or IsList(obj), let(
		res := Concatenation([ObjId(obj)], List(_children(obj), x -> ShapeObject@(x, func_replace_by_wildcard))),
		When(Length(res)=1, res[1], res)),
	obj
);

Class(..., rec(
	left := false,
	right := false
));

Left := z -> z.left;
Right := z -> z.right;

# @ : basic pattern object
# .parent is the pointer to the unique object in @.table 
# That table does not store any precondition functions
#
Class(@, rec(
	is@ := true,

	table := WeakRef(tab()),

	__call__ := meth(arg)
		local res, self, num, target, precond;
		if Length(arg) < 2 or Length(arg) > 4 then 
			Error("Usage: @(<num>, [<target>], [<cond>])");
		fi;

		self := arg[1];
		num := arg[2];
		if Length(arg) >= 3 then
			target := arg[3];
		fi;
		if Length(arg) = 4 then
			precond := arg[4];
		fi;

		if not IsBound(self.table.(num)) then
			res := self.new(num); 
			self.table.(num) := res;
			if IsBound(target) then
				res := res.target(target);
			fi;
			if IsBound(precond) then
				res := res.cond(precond);
			fi;
			return res;
		else
			res := self.table.(num);
			Unbind(res._target);
			Unbind(res._precond);
			if IsBound(target) then
				res := res.target(target);
			fi;
			if IsBound(precond) then
				res := res.cond(precond);
			fi;
			return res;
		fi;
	end,

	new := (self, num) >> CantCopy(
		WithBases(self, rec(num := num, operations := PrintOps))),

	match := meth(self, obj, cx)
		if (not IsBound(self._target) or ObjId(obj) in self._target) and
		   (not IsBound(self._precond) or self._precond(obj)) 
		then 
			if IsBound(self.parent) then
				self.parent.val := obj;
			else
				self.val := obj;
			fi;
			return true;
		else
			return false;
		fi;
	end,
	
	cond := (self, precond_func) >> Checked(IsCallableN(precond_func, 1), CantCopy(
		WithBases(self, rec(_precond := precond_func, 
							 parent := When(IsBound(self.parent), self.parent, self))))),

	target := (self, target_id) >> CantCopy(
		WithBases(self, rec(#_taddr  := When(IsList(target_id), List(target_id,BagAddr), [BagAddr(target_id)]),
							_target := When(IsList(target_id), target_id,			   [target_id]),
							parent := When(IsBound(self.parent), self.parent, self)))),

	val := false,

	print := self >> Print("@(", self.num, ")", 
						   When(IsBound(self._target), 
								Print(".target(", self._target, ")"), 
								""),
						   When(IsBound(self._precond), 
								Print(".cond(", self._precond, ")"), 
								"")),

	clear := meth(self)
		local e, l;
		l := TabToList(self.table);
		for e in l do
			if IsRec(e) and IsBound(e.val) then
				e.val := false;
			fi;
		od;
		if IsBound(self.val) then
			self.val := false;
		fi;
	end
));

# @@ : basic pattern object for context sensitive matching
# @@ has its own table.
# It still uses a .parent to point into @@.table where matches are stored
# That table does not store any precondition functions.
# Both lhs and rhs of rewrite rules must use either @ or @@. Otherwise wrong
# table will be used.

Class(@@, @, rec(
	table := WeakRef(tab()),

	match := meth(self, obj, cx)
		local base_match;
		base_match := @.match;
		if not IsBound(self._cxcond) then
			return base_match(self, obj, cx); 
		else
			return base_match(self, obj, cx) and self._cxcond(obj, cx);
		fi;
	end,

	cond := (self, cxcond_func) >> CantCopy(Checked(IsCallableN(cxcond_func, 2), 
		WithBases(self, rec(_cxcond := cxcond_func, 
							 parent := When(IsBound(self.parent), self.parent, self))))),

	print := self >> Print("@@(", self.num, ")", 
						   When(IsBound(self._target), 
								Print(".target(", self._target, ")"), 
								""),
						   When(IsBound(self._cxcond), 
								Print(".cond(", self._cxcond, ")"), 
								""))
));


Is@ := x -> IsRec(x) and IsBound(x.is@) and x.is@;

Local._midmatch := arg->false;
Local._normalmatch := arg->false;

Local._match_id := (obj, shape, cx) -> Cond(
	IsRec(shape) and IsBound(shape.is@),
		shape.match(obj, cx), 
	BagType(shape) < T_FUNCTION or BagType(shape) in [T_STRING,T_RANGE], 
		BagType(shape) = BagType(obj) and shape=obj,
	Same(obj, shape) or Same(ObjId(obj), shape));

##
Declare(cx_enter, cx_leave, empty_cx, apply_rules_ni, _SubstBottomUp, _SubstTopDown);
##

# PatternMatch(<obj>, <shape>, <cx>)
#	Returns true if <obj> matches the given <shape> in a given context <cx>.
#	For plain matches use empty context table empty_cx().
#
PatternMatch := function(obj, shape, cx)
	local ch, numch, shlen, res;
	
	if not IsList(shape) or BagType(shape) in [T_STRING,T_RANGE] then
		return _match_id(obj, shape, cx);
	elif shape = [] then
		return false; 
	elif not _match_id(obj, shape[1], cx) then
		return false;
	else 
		shlen := Length(shape);
		ch := _children(obj);
		numch := Length(ch);
		if shlen = 1 
			then return numch = 0;
		else 
			cx_enter(cx, obj);
			if shape[2] = ... then 
				if shape[shlen] = ... then
					res := _midmatch(ch, shape{[3..shlen-1]}, cx);
				elif numch < shlen-2 then
					res := false;
				else 
					....left := numch-shlen+2;
					....right := numch+1;
					res := _normalmatch(ch, numch-shlen+3, shlen-2,  shape, 3, shlen-2, cx);
				fi;
			elif Last(shape) = ... then
				if numch < shlen-2 then
					res := false;
				else  
					....left := 0;
					....right := shlen-1;
					res := _normalmatch(ch, 1, shlen-2,  shape, 2, shlen-2, cx);
				fi;
			else 
				....left := false;
				....right := false;
				res := _normalmatch(ch, 1, numch,  shape, 2, shlen-1, cx);
			fi;
			cx_leave(cx, obj);
			return res;
		fi;
	fi;
end;

_normalmatch := function(lst, lstart, llen, shape, sstart, slen, cx)
	local i, ch;
	if llen <> slen then
		return false;
	else 
		for i in [0..slen-1] do
			if not PatternMatch(lst[lstart+i], shape[sstart+i], cx) then
				return false;
			fi;
		od;
		return true;
	fi;
end;

_midmatch := function(lst, shape, cx)
	local i, ch, shlen, res, llen;
	shlen := Length(shape);
	llen := Length(lst);
	if llen < shlen then
		return false;
	elif llen = shlen then 
		res := _normalmatch(lst, 1, llen, shape, 1, shlen, cx);
		if res then
			....left := 0;
			....right := llen + 1;
		fi;
		return res;
	else 
		for i in [1 .. llen - shlen + 1] do
			if PatternMatch(lst[i], shape[1], cx) then 
				if shlen = 1 then 
					....left := i-1;
					....right := i+shlen;
					return true;
				else
					if _normalmatch(lst, i+1, shlen-1,  shape, 2, shlen-1, cx) then
						....left := i-1;
						....right := i+shlen;
						return true;
					fi;
				fi;
			fi;
		od;
		return false;
	fi;
end;

#F AlternativesRewrite( <expr>, <from>, <to_func> )
#F
#F Creates all alternatives for substitution of an expression tree.
#F	<expr> - expression to substitute in
#F	<from> - shape to substitute
#F	<to_func> - a substitution function of the form e->f(e), where
#F			 as <e> will be passed the subtree matching <from>
#F

Declare(_AlternativesRewrite, _ConditionalAlternativesRewrite);

AlternativesRewrite := (expr, from, to_func) ->
	_AlternativesRewrite(expr, from, to_func, empty_cx(), []);

_AlternativesRewrite := function(expr, from, to_func, cx, list)
	local ch, i, clist;

	ch := _children(expr);
	if Length(ch) <> 0 then 
		# not a leaf
		cx_enter(cx, expr);
		for i in [1..Length(ch)] do
			clist := _AlternativesRewrite(ch[i], from, to_func, cx, []);
			Append(list, List(clist, function(x) local a; a:=Copy(expr); _setChild(a, i, x); return a; end));
		od;					 
		cx_leave(cx, expr);
	fi;

	if PatternMatch(expr, from, cx) then
		Add(list, When(NumArgs(to_func)=2, to_func(cx, expr), to_func(expr)));
	fi;

	return list;
end;

ConditionalAlternativesRewrite := (expr, from, condition_to_func, rewrite_to_func) ->
	_ConditionalAlternativesRewrite(expr, from, condition_to_func, rewrite_to_func, empty_cx(), []);

_ConditionalAlternativesRewrite := function(expr, from, condition_to_func, rewrite_to_func, cx, list)
	local ch, i, clist;

	ch := _children(expr);
	if Length(ch) <> 0 then 
		# not a leaf
		cx_enter(cx, expr);
		for i in [1..Length(ch)] do
			clist := _ConditionalAlternativesRewrite(ch[i], from, condition_to_func, rewrite_to_func, cx, []);
			Append(list, List(clist, function(x) local a; a:=Copy(expr); _setChild(a, i, x[2]); return [x[1], a]; end));
		od;
		cx_leave(cx, expr);
	fi;

	if PatternMatch(expr, from, cx) then
		Add(list, [When(NumArgs(condition_to_func)=2, condition_to_func(cx, expr), condition_to_func(expr)),
				When(NumArgs(rewrite_to_func)=2, rewrite_to_func(cx, expr), rewrite_to_func(expr))]);
	fi;

	return list;
end;


# Return a list of rewrite rule objects, from either a list of a record
# passing in a record has the advantage of clean rule naming
#
_parseRuleSet := rset -> Cond(
	IsList(rset), List(rset, Rule),
	IsRec(rset),  List(UserRecFields(rset), 
	function(fld) local r; r := Rule(rset.(fld)); r.name := fld; return r; end),
	Error("<rset> must be a list of rules, or a record rec(rule1 := Rule(...), ...)")
);

#F SubstBottomUp( <expr>, <from>, <to_func> )
#F
#F Destructive bottom up substitution on an expression tree.
#F	<expr> - expression to substitute in
#F	<from> - shape to substitute
#F	<to_func> - a substitution function of the form e->f(e), where
#F			 as <e> will be passed the subtree matching <from>
#F
SubstBottomUp := (expr, from, to_func) ->
	_SubstBottomUp(expr, [ Rule(from, to_func, "unnamed(SubstBottomUp)") ], empty_cx());

SubstBottomUpRules := (expr, ruleset) ->
	_SubstBottomUp(expr, _parseRuleSet(ruleset), empty_cx());

_SubstBottomUp := function(expr, rules, cx)
	local ch, i;
	ch := _children(expr);
	if ch <> [] then
		cx_enter(cx, expr);
		for i in [1..Length(ch)] do
			_setChild(expr, i, _SubstBottomUp(ch[i], rules, cx));
		od;
		cx_leave(cx, expr);
	fi;
	return apply_rules_ni(rules, expr, cx);
end;

_SubstLeaves := function(expr, from, to_func, cx)
	local ch, i;
	ch := _children(expr);
	if Length(ch) <> 0 then 
		# not a leaf
		cx_enter(cx, expr);
		for i in [1..Length(ch)] do
			_setChild(expr, i, _SubstLeaves(ch[i], from, to_func, cx));
		od;
		cx_leave(cx, expr);
		return expr;
	else
		# a leaf
		if PatternMatch(expr, from, cx) then 
			return When(NumArgs(to_func)=2, to_func(cx, expr), to_func(expr));
		else 
			return expr;
		fi;
	fi;
end;

SubstLeaves := (expr, from, to_func) -> _SubstLeaves(expr, from, to_func, empty_cx());

SubstChildren := function(expr, from, to_func)
	local ch, i;
	ch := _children(expr);
	#expr := map_children_safe(expr, c -> When(PatternMatch(c,from,empty_cx()), to_func(c), c));
	for i in [1..Length(ch)] do
		_setChild(expr, i, 
			When(PatternMatch(ch[i],from,empty_cx()), to_func(ch[i]), ch[i]));
	od;
	return expr;
end;

SubstTopDown := (expr, from, to_func) ->
	_SubstTopDown(expr, [ Rule(from, to_func, "unnamed(SubstTopDown)") ], empty_cx());

SubstTopDown_named := (expr, from, to_func, name) ->
	_SubstTopDown(expr, [ Rule(from, to_func, name) ], empty_cx());

SubstTopDownRules := (expr, ruleset) ->
	_SubstTopDown(expr, _parseRuleSet(ruleset), empty_cx());

_SubstTopDown := function(expr, rules, cx)
	local ch, newch, res;
	expr := apply_rules_ni(rules, expr, cx);

	ch := _children(expr);
	if ch <> [] then
		cx_enter(cx, expr);
		newch :=  List(ch, c -> _SubstTopDown(c, rules, cx));
		res := _fromChildren(expr, newch);
		cx_leave(cx, expr);
	else 
		res := expr;
	fi;
	return res;
end;

Declare(_SubstTopDownNR);

# same as SubstTopDown but doesn't recurse on the substitution
SubstTopDownNR := (expr, from, to_func) -> 
	_SubstTopDownNR(expr, [ Rule(from, to_func, "unnamed(SubstTopDownNR)") ], empty_cx());

SubstTopDownNR_named := (expr, from, to_func, name) -> 
	_SubstTopDownNR(expr, [ Rule(from, to_func, name) ], empty_cx());

SubstTopDownRulesNR := (expr, ruleset) -> 
	_SubstTopDownNR(expr, _parseRuleSet(ruleset), empty_cx());

_SubstTopDownNR := function(expr, rules, cx)
	local ch, newch, res, n_applied;
	n_applied := cx.applied;
	cx.rlimit := 1;
	expr := apply_rules_ni(rules, expr, cx);
	if cx.applied > n_applied then
		return expr;
	fi;

	ch := _children(expr);
	if ch <> [] then
		cx_enter(cx, expr);
		newch :=  List(ch, c -> _SubstTopDownNR(c, rules, cx));
		res := _fromChildren(expr, newch);
		cx_leave(cx, expr);
	else
		res := expr;
	fi;
	return res;
end;


#F MatchSubst(<expr>, <rulelist>)
#F
#F Attempts matching 1 of the rules from the list to <expr> if it
#F succeeds the transformation is applied, otherwise original 
#F expression is returned.
#F
#F MatchSubst is similar to SubstTopDown, but it does not recurse
#F on children.
#F
MatchSubst := function(expr, rules)
	local ch, i, r, cx;
	cx := empty_cx();
	for r in _parseRuleSet(rules) do
		if PatternMatch(expr, r.from, cx) then 
			RuleTrace(r);
			return r.to(expr); 
		fi;
	od;
	return expr;
end;

Declare(_Collect);

#F Collect(<expr>, <shape>)
#F
#F Returns a list of all subtrees of <expr> that match <shape>
#F
Collect := (expr, shape) -> _Collect(expr, shape, empty_cx(), true);

#F Contains(<expr>, <shape>)
#F
#F Returns 'true' if <expr> contains <shape>
#F NOTE: optimize this
Contains := (expr, shape) -> _Collect(expr, shape, empty_cx(), false)<>[];

#F CollectNR(<expr>, <shape>)
#F
#F Returns a list of all subtrees of <expr> that match <shape>.
#F Unlike 'Collect', once 'CollectNR' finds an object matching <shape>,
#F it does not search inside its children.
#F
CollectNR := (expr, shape) -> _Collect(expr, shape, empty_cx(), false);

_Collect := function(expr, shape, cx, do_recurse)
	local res, ch, c;
	if PatternMatch(expr, shape, cx) then
		if not do_recurse then 
			return [expr];
		else 
			cx_enter(cx, expr);
			res := Concatenation(List(_children(expr), c -> _Collect(c, shape, cx, do_recurse)));
			cx_leave(cx, expr);
			return Concatenation([expr], res);
		fi;
	else 
		cx_enter(cx, expr);
		res := Concatenation(List(_children(expr), c -> _Collect(c, shape, cx, do_recurse)));
		cx_leave(cx, expr);
		return res;
	fi;
end;

Declare(_Pull);

#F Pull(<expr>, <shape>, <to_func>, <pull_func>)
#F
#F Pull is a hybrid of Collect and SubstBottomUp. First three parameters
#F specify substitution, and the last one collection:
#F
#F	<expr> - expression to substitute in
#F	<from> - shape to substitute
#F	<to_func> - a substitution function of the form e->f(e), where
#F			 as <e> will be passed the subtree matching <from>
#F	<pull_func> - applied to matching subtree <e> to collect data
#F
#F Pull retuns a tuple [<data>, <newtree>], where <data> is a list 
#F of all matched subtrees (before substitutions) with <pull_func>
#F applied, and <newtree> is a tree obtained by substitution.
#F
## Note: This is Top-down recursive Pull, same as PullTD
Pull := (expr, shape, to_func, pull_func) -> _Pull(expr, shape, to_func, pull_func, empty_cx(), 
	true, true);
# Top-down recursive Pull
PullTD := (expr, shape, to_func, pull_func) -> _Pull(expr, shape, to_func, pull_func, empty_cx(), 
	true, true);
	
# Bottom-up recursive Pull (non-recursive bottom up does not exist)
PullBU := (expr, shape, to_func, pull_func) -> _Pull(expr, shape, to_func, pull_func, empty_cx(), 
	true, false);
	
# Top-down non-recursive Pull
PullNR := (expr, shape, to_func, pull_func) -> _Pull(expr, shape, to_func, pull_func, empty_cx(), 
	false, true);

_Pull := function(expr, shape, to_func, pull_func, cx, do_recurse, top_down)
	local ch, i, t, data, newdata, newexpr;
	data := [];

	if top_down then
		if PatternMatch(expr, shape, cx) then
			Add(data, When(NumArgs(pull_func)=2, pull_func(cx, expr), pull_func(expr)));
			expr := When(NumArgs(to_func)=2, to_func(cx, expr), to_func(expr));
			if not do_recurse then
				return [ data, expr ];
			fi;
		fi;
	fi;

	cx_enter(cx, expr);
	ch := _children(expr);
	for i in [1..Length(ch)] do
		t := _Pull(ch[i], shape, to_func, pull_func, cx, do_recurse, top_down);
		Append(data, t[1]);
		_setChild(expr, i, t[2]);
	od;
	cx_leave(cx, expr);

	if not top_down then
		if PatternMatch(expr, shape, cx) then
			Add(data, When(NumArgs(pull_func)=2, pull_func(cx, expr), pull_func(expr)));
			expr := When(NumArgs(to_func)=2, to_func(cx, expr), to_func(expr));
			if not do_recurse then
				return [ data, expr ];
			fi;
		fi;
	fi;
	return [data, expr]; 
end;

#F Harvest( <expr>, <shape-func-list> )
#F
#F Harvest is a generalization of Collect( <expr>, <shape> ).
#F It returns concatenated list of <func>(child) for all children 
#F of <expr> that match <shape>, for each <shape>-<func> pair.
Harvest := function( expr, shape_func_list )
	local res, sf, c;
	res := [];
	for sf in shape_func_list do
		Append(res, List(Collect(expr, sf[1]), sf[2]));
	od;
	return res;
end;

#F SubstObj(<expr>, <obj>, <new_obj>
#F replacing all occurences of <obj> by <new_obj>
SubstObj := (expr, obj, new_obj) -> SubstTopDownNR_named(expr, @.cond(x -> x=obj), e -> new_obj, "SubstObj");
