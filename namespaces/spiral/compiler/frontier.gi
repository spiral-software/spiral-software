
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#
# Frontier
#

#F _DFUnion(<list_frontiers>) merges frontiers.
#F     Multiple frontiers result from multiple outgoing paths.
#F
#F     For example t3 = add(t1,t2) will have the frontier
#F       [ [t3], _DFUnion([DefFrontier(t1), DefFrontier(t2)]) ]
#F
#F     The parameter is a list of frontiers, where each frontier
#F     is a list of sets of variables. 
#F
_DFUnion := function(list_frontiers)
    local depths, maxdepth, curset, res, numfrontiers, d, j;
    numfrontiers := Length(list_frontiers); 
    depths := List(list_frontiers, Length);
    maxdepth := Maximum0(depths);
    res := [];
    for d in [1..maxdepth] do 
        curset := Set([]);
	for j in [1..numfrontiers] do
	    if d <= depths[j] then 
		curset := UnionSet(curset, list_frontiers[j][d]); 
	    else 
		# if frontier is shorter than the longest, we carry over its last terms. 
		# For instance, the frontier of t5 in { t3=add(t1,t2); t5=add(t3, t4) }
		# is [[t5], [t3, t4], [t1, t2, t3]].
		curset := UnionSet(curset, list_frontiers[j][depths[j]]); 
	    fi;
	od;
	Add(res, curset);
    od;
    return res;
end;

#F DefFrontier(<cmd>, <depth>) - computes the so-called def-frontier for a command.
#F    The function can be defined inductively as:
#F       DF(t, d) == DF(t.def, d), where t.def is the 'assign' defining t
#F       DF(assign(t,      ...    ), 0)  == [ [t] ]
#F       DF(assign(t, f(a1,a2,...)), 1)  == [ [t], [a1,a2,...] ]
#F       DF(assign(t, f(a1,a2,...)), n)  == [ [t], [a1,a2,...] ] concat
#F                                                 _DFUnion(DF(a1,n-1), DF(a2,n-1), ...)
#F
#F    In plain words it returns the minimal set of variables that fully define a given
#F    location (or cmd.loc). 
#F
#F    For instance, the frontier of t5 in { t3=add(t1,t2); t5=add(t3, t4) }
#F    is [[t5], [t3, t4], [t1, t2, t4]].
#F
#F    Note (!!!): MarkDefUse(code) must be called on the corresponding code object,
#F                so that locations can be connected with their definitions
#F
#F    If <depth> is negative, then the maximally deep frontier is computed.
#F
DefFrontier := (cmd, depth) -> Cond(
    IsLoc(cmd), let(def := DefLoc(cmd), When(def=false, [[cmd]], DefFrontier(def, depth))),
    depth = 0,      [Set([cmd.loc])],
#    IsLoc(cmd.exp), [Set([cmd.loc])],
    let(args := ArgsExp(cmd.exp),
	Concatenation([Set([cmd.loc])], 
	               _DFUnion(
			   List(args, a -> DefFrontier(a, depth-1))))));

DefCollect := (cmd, depth) -> Cond(
    depth = 0,      cmd.loc,
    IsLoc(cmd.exp), cmd.loc,
    depth = 1,      cmd.exp,
    SubstLeaves(Copy(cmd.exp), var, 
	v -> let(def := DefLoc(v), When(def=false, v, DefCollect(def, depth-1)))));

AddFrontier := x -> Cond(
    IsLoc(x), let(def := DefLoc(x),
	Cond(def = false, x, AddFrontier(def.exp))),
    ObjId(x) in [add,sub],
        ApplyFunc(ObjId(x), List(x.args, AddFrontier)), 
    x);

AddFrontierD := (x,d) -> Cond(
    d = 0,    x,
    IsLoc(x), let(def := DefLoc(x),
	Cond(def = false, x, AddFrontierD(def.exp, d))),
    ObjId(x) in [add,sub],
        ApplyFunc(ObjId(x), List(x.args, a -> AddFrontierD(a, d-1))), 
    x);

AddFrontierDM := (x,d) -> Cond(
    d = 0,    
        [x,false],
    IsLoc(x), let(def := DefLoc(x),
	Cond(def = false, [x,false], AddFrontierDM(def.exp, d))),
    d = 1,   
        [x,ObjId(x)=mul],
    ObjId(x) in [add,sub],
        let(args := List(x.args, a -> AddFrontierDM(a, d-1)),
	    Cond(ForAll(args, a->a[2]=false), [x,false],
		 [ApplyFunc(ObjId(x), List([1..Length(args)], i->Cond(args[i][2], args[i][1], x.args[i]))), true])),
    [x,ObjId(x)=mul]);

# NOTE: memoization
#
Class(madd, add, rec(
    redundancy := self >> self.constantRedundancy(), # or self.termRedundancy(),

    terms               := self >> Map(self.args, a->a.args[2]),
    constants           := self >> Map(self.args, a->a.args[1].v),
    nonTrivialConstants := self >> Filtered(Map(self.args, a->AbsFloat(a.args[1].v)), c->c<>1),

    constantRedundancy := self >> let(
	constants := self.nonTrivialConstants(),
	constants <> [] and Length(constants)<>Length(Set(constants))),

    termRedundancy := self >> let(
	locs := Filtered(self.terms(), IsLoc),
	locs <> [] and Length(locs) <> Length(Set(locs))),

    factorConstants := meth(self)
        local constants, res, t, c, a, i, fac_indices, l1, l2;
	res := []; constants := []; l1:=[]; fac_indices := Set([]);
	for a in self.args do
	    a := Copy(a);
	    c := a.args[1].v; t := a.args[2];

	    if c in [1,-1] then 
		Add(constants, c); Add(res, a); 
	    elif c in constants then
		i := Position(constants, c);
		res[i].args[2] := add(res[i].args[2], t); AddSet(fac_indices,i);
	    elif -c in constants then
		i := Position(constants, -c);
		res[i].args[2] := sub(res[i].args[2], t); AddSet(fac_indices,i);
	    else 
		Add(constants, c); Add(res, a);
	    fi;
	od;

	l1 := res{fac_indices};
	l2 := res{Difference([1..Length(res)], fac_indices)};

	if l1=[] then return FoldL1(l2,add); fi;
	l1 := FoldL1(l1,add);

	if l2=[] then return l1; fi;
	l2 := FoldL1(l2,add);
	return add(l1, l2);
    end,

    factorTerms := meth(self)
        local terms, res, t, c, a, i, fac_indices, l1, l2;
	res := []; terms := []; fac_indices := Set([]);
	for a in self.args do
	    a := Copy(a);
	    c := a.args[1].v; t := a.args[2];

	    if IsLoc(t) and t in terms then 
		i := Position(terms, t);
		res[i].args[1].v := res[i].args[1].v + c; AddSet(fac_indices,i);
	    else 
		Add(terms, t); Add(res, a);
	    fi;
	od;

	l1 := res{fac_indices};
	l2 := res{Difference([1..Length(res)], fac_indices)};
	l1 := Filtered(l1, a->a.args[1].v <> 0);
	l2 := Filtered(l2, a->a.args[1].v <> 0);

	if l1=[] and l2=[] then return V(0); fi;

	if l1=[] then return FoldL1(l2,add); fi;
	l1 := FoldL1(l1,add);

	if l2=[] then return l1; fi;
	l2 := FoldL1(l2,add);
	return add(l1, l2);
    end
));

# "MultiAdd" is a normal form for linear expressions
#  MultiAdd = madd(mul(c1,e1), mul(c2,e2), ...)
#
Class(ToMultiAdd, RuleSet, rec(
    __call__ := (self, e) >> TDA(madd(mul(1,e)), self, SpiralDefaults)));

RewriteRules(ToMultiAdd, rec(
    EliminateAdd := Rule(add, e->ApplyFunc(madd, List(e.args, a->mul(1,a)))),
   
    EliminateSub := Rule(sub, e->madd(mul(1,e.args[1]), mul(-1, e.args[2]))),

    MulMul := Rule([mul, @(1,Value), [mul, @(2,Value), @(3)]], 
	e -> mul(@(1).val.v * @(2).val.v, @(3).val)),

    MaddSinkMul := ARule(madd, [[mul, @(1,Value), @(2,madd)]], 
	e -> List(@(2).val.args, a->mul(@(1).val*a.args[1], a.args[2]))),

    FlattenMaddMadd := ARule(madd, [@(1,madd)], e->@(1).val.args)
));



NODE := x -> RCSE.node(TDouble, x);
REMOVE_NODE := x -> RCSE.remove_node(x);

deepCheck := function(c, d)
    local frontier, linexp;
    frontier := AddFrontierD(c.exp, d);
    linexp := ToMultiAdd(frontier);
    if not ObjId(linexp)=madd then return [c.loc,c.loc,"none"]; fi;
    if linexp.constantRedundancy() then	return [c.loc, linexp, "constant"];
    elif linexp.termRedundancy()   then	return [c.loc, linexp, "term"];
    else	                        return [c.loc, c.loc, "none"];
    fi;
end;

deepCheckM := function(c, d)
    local frontier, linexp;
    frontier := AddFrontierDM(c.exp, d)[1];
    linexp := ToMultiAdd(frontier);
    if not ObjId(linexp)=madd then return [c.loc,c.loc,"none"]; fi;
    if linexp.constantRedundancy() then	return [c.loc, linexp, "constant"];
    elif linexp.termRedundancy()   then	return [c.loc, linexp, "term"];
    else	                        return [c.loc, c.loc, "none"];
    fi;
end;

_DeepSimplifyCmd := function(c, d, frontier_func)
    local node, frontier, linexp;

    frontier := frontier_func(c.exp, d);
    linexp := ToMultiAdd(frontier);

    if not ObjId(linexp)=madd then return c; fi;

    if linexp.constantRedundancy() then
	linexp := linexp.factorConstants();
	if IsValue(linexp) then node := linexp;
	else node := ApplyFunc(ObjId(linexp), List(linexp.args, NODE));
	fi;
	c.exp := node;
    elif linexp.termRedundancy() then
	linexp := linexp.factorTerms();
	if IsValue(linexp) then node := linexp;
	else node := ApplyFunc(ObjId(linexp), List(linexp.args, NODE));
	fi;
	c.exp := node;
    fi;
    return c;
end;

DeepSimplifyCmd := function(c, d1, d2)
    if PatternMatch(c.exp, [mul, @(1,Value), @(2,var,v->DefLoc(v)<>false)], empty_cx()) and
	PatternMatch(DefLoc(@(2).val).exp, [mul, @(3,Value), @(4)], empty_cx()) then
	c.exp := mul(@(1).val.v * @(3).val.v, @(4).val);
    fi;
    c := _DeepSimplifyCmd(c, d1, AddFrontierD);
    c := _DeepSimplifyCmd(c, d2, (x,d)->AddFrontierDM(x,d)[1]);
    return c;
end;

Declare(DeepSimplifyChain);


DEPTH1 := 2; # AddFrontierD
DEPTH2 := 3; # AddFrontierDM

DefUnbound := function(e, bound)
    local unbound, defs;
    unbound := Difference(e.free(), bound);
    defs := Filtered(List(unbound, DefLoc), d->d<>false and not Same(d,e));
    return Concatenation(Concatenation(List(defs, d -> DefUnbound(d, bound))), defs);
end;

_DeepSimplifyChain := function(code, bound)
   local orig, c, res, cmds, args, supp, supplocs;
   res := chain();

   for c in code.cmds do
       if ObjId(c) <> assign then Error("Can't handle 'c' of type '", ObjId(c), "'"); fi;
       orig := c.exp;

       DeepSimplifyCmd(c,DEPTH1,DEPTH2);	   
       if IsVar(c.loc) then AddSet(bound, c.loc); fi;

       supp := chain(DefUnbound(c.exp, bound));
       supplocs := List(supp.cmds, cmd->cmd.loc);
       DoForAll(supplocs, REMOVE_NODE);
       UniteSet(bound, supplocs);

       supp := FlattenCode(_DeepSimplifyChain(supp, bound)); 
       Add(supp.cmds, c);
       supp := RCSE(supp);

       Append(res.cmds, supp.cmds);
   od;

   return res;
end;

DeepSimplifyChain := code -> _DeepSimplifyChain(code, code.free());

DeepSimplify := function(code)
   local c, frontier, linexp, free, undefined;
   code := BinSplit(code); 
   RCSE.flush();
   MarkDefUse(code);
   return SubstBottomUp(code, chain, DeepSimplifyChain);
end;

_CheckUninitializedChain := function(code, def, uninit)
    local def1, def2, c, use;
    for c in code.cmds do
        if ObjId(c)=assign then
	    use := VarArgsExp(c.exp);
	    UniteSet(uninit, Filtered(use, x->not x in def));
	    AddSet(def, c.loc);

        elif ObjId(c)=chain then
            [def, uninit] := _CheckUninitializedChain(c, def, uninit);

        elif ObjId(c)=IF then
            [def1, uninit] := _CheckUninitializedChain(c.then_cmd, ShallowCopy(def), uninit);
            [def2, uninit] := _CheckUninitializedChain(c.else_cmd, ShallowCopy(def), uninit);
            def := Union(def1, def2);
	fi;
    od;
    return [def, uninit];
end;

CheckUninitializedChain := function(code)
    local def, uninit;
    Constraint(ObjId(code)=chain);
    def := Set([]);
    uninit := Set([]);
    [def, uninit] := _CheckUninitializedChain(code, def, uninit);
    return uninit;
end;

reds := chain -> Filtered(chain.cmds, c -> deepCheck(c,DEPTH1)[3] <>"none" or
    deepCheckM(c, DEPTH2)[3] <>"none");

DoDeepSimplify := function(code, num_iterations)
   local c, frontier, linexp, free, undefined, i;
   Print("Orig cost : ", ArithCostCode(code), "\n");
   for i in [1..num_iterations] do
       code := DeepSimplify(code);  Print("DS : ", ArithCostCode(code), "\n");
       code := CopyPropagate(code); Print("CP : ", ArithCostCode(code), "\n");
       code := CopyPropagate(code); Print("CP : ", ArithCostCode(code), "\n");
       code := CopyPropagate(code); Print("CP : ", ArithCostCode(code), "\n");
       code := BinSplit(code); 
       code := RCSE(code); Print("RCSE : ", ArithCostCode(code), "\n");
   od;
   return code;
end;


# err := [];

# LIMIT := 1; ww:=DeepSimplify(Copy(w));; cm := CMatrix(pp(ww));
# if inf_norm(cm-dm) > 1e-10 then Print("ERROR\n"); Add(err, [Copy(ww), Copy(w), inf_norm(cm-dm)]); else Print("OK\n"); fi;

# LIMIT := 1; w:=DeepSimplify(Copy(ww));; cm := CMatrix(pp(w));
# if inf_norm(cm-dm) > 1e-10 then Print("ERROR\n"); err := Add(err, [Copy(w), Copy(ww), inf_norm(cm-dm)]); else Print("OK\n"); fi;
