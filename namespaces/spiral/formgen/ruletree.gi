
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details



#F RuleTreeOps.\=( <ruletree1>, <ruletree2> )
#F    returns true if <ruletree1> and <ruletree2> are equal, i.e., they
#F    represent the same breakdown strategy.  Otherwise, false is returned.
#F

RuleTreeOps := CantCopy(OperationsRecord("RuleTreeOps"));


IsRuleTree := T -> IsRec(T) and IsBound(T.isRuleTree) and T.isRuleTree;


#F Ruletrees
#F =========
#F
#F A ruletree is a record with the following mandatory fields:
#F
#F   isRuleTree  = "true"              # identifies ruletrees
#F   operations  = RuleTreeOps         # operations record
#F   node        = <spl>               # the spl expanded
#F   transposed  = false/true          # indicates whether <rule> or
#F                                       transpose - <rule> - transpose
#F   rule        = <rule>              # the rule chosen
#F   children    = <list>              # the children, a list of
#F                                       ruletrees/non-terminal spls
#F
#F Note that children can be a list of ruletrees as well as non-terminal spls.
#F The field children is empty iff the ruletree is a leaf.
#F In general, the .node is a non-terminal spl that is expanded by .rule.
#F
#F The following fields are optional:
#F   splOptions = [ <option1>, <option2>, .. ]
#F     valid <options> : "unrolled", "pcl"
#F

Class(RuleTreeClass, rec(
    spl  := self >> SPLRuleTree(self),
    measure := self >> SPLRuleTree(self).measure(),
    shortPrint := false,
    dims := self >> self.node.dims(),
    dmn  := self >> self.node.dmn(),
    rng  := self >> self.node.rng(),

    print := meth(self, i, is)
        local ch;

        if Length(self.children) = 0 then
           # print leaves in one line
            Print(self.rule, "( ");
            When (self.transposed, Print("\"T\","));
            self.node.print(i+is, is);
            When (IsBound(self.splOptions),
                Print(", ", self.splOptions, " )"),
                Print(" )"));
        else
            # print children on separate lines
            Print(self.rule, "( ",
                When(self.transposed, "\"T\", ", ""),
                When(not self.shortPrint, self.node.print(i+is, is), ""), ",\n", Blanks(i+is)
            );
            self.children[1].print(i+is, is);
            for ch in Drop(self.children, 1) do
                Print(",\n", Blanks(i+is));
                ch.print(i+is, is);
            od;

            When (IsBound(self.splOptions),
                Print("\n", Blanks(i), ", ", self.splOptions, " )"),
                Print(" )"));
        fi;
    end,

    # new with tSPL and opts.baseHash:
    # we cannot blindly transpose everything, as vector
    # permutations that are automatically generated
    # may not support transposition
    transpose := self >>
        When((not self.node.transposeSymmetric()) and self.node.isSymmetric(),
            self,
            RuleTree(
                self.rule,
                When(self.transposed, "", "T"),
                self.node.transpose(),
                List(self.children, x->x.transpose())
        )
    ),
    # RuleTreeRewriting enabled by default
    # one can disable it with "EnableRuleTreeRewriting(false)"
    rChildren := (self) >> [ self.node, self.children ],
    rSetChild := rSetChildFields("node", "children"),
    from_rChildren := (self, rch) >>
        RuleTree(self.rule, When(self.transposed, "T", ""), rch[1], rch[2])
));

RuleTreeOps.\= := function( T1, T2 )
    return
        IsRuleTree(T1) and IsRuleTree(T2) and
        T1.rule = T2.rule and
        T1.transposed = T2.transposed and
        IsIdenticalSPL(T1.node, T2.node) and
        T1.children = T2.children and
        (   (   IsBound(T1.splOptions) and 
		        IsBound(T2.splOptions) and
                T1.splOptions = T2.splOptions
			) or
            (   not IsBound(T1.splOptions) and 
			    not IsBound(T2.splOptions)
			)
        );
end;

#F RuleTreeNC( <rule>, <non-terminal spl>, <children> )
#F   returns the corresponding ruletree without any checking.

RuleTreeNC := function ( R, S, C )
    return WithBases(RuleTreeClass,
        rec(
            isRuleTree := true,
            operations := RuleTreeOps,
            node       := S,
            transposed := false,
            rule       := R,
            children   := C
		)
    );
end;


#F RuleTree(
#F   <rule>,
#F   [ "T" ,]
#F   <non-terminal spl>
#F   [, <children> ]
#F   [, <list-of-sploptions> ]
#F )
#F
#F   Creates a ruletree for <non-terminal spl>
#F   using <rule> with <children>. The <children> can be non-terminal spls
#F   or ruletrees. It is checked whether <children> can be derived
#F   from <non-terminal spl> using <rule>.
#F   If the string "T" is supplied, then instead of <rule>, the
#F   sequence transpose - <rule> - transpose is denoted. In this case
#F   <non-terminal spl> and <children> are also transposed. This construction
#F   avoids unnecessary recoding of rules that are just transposes of
#F   other rules. In this case, <rule> must have the field
#F   .forTransposition set to true.
#F
#F   The argument <list-of-sploptions> provides options used for spl compilation.
#F   See top of this file for valid options; examples are "unrolled" and "pcl".
#F   <children> might be empty (default) in which case the ruletree
#F   returned is a leaf.
#F
RuleTree := function ( arg )
    local a, R, t, S, C, u, S1, C1, RT;
    S := false;
    t := false;
    u := [ ];
    C := [ ];
    R := false;

    for a in arg do
        if IsString(a) and a = "T" then 
	        t := true;
        elif IsList(a) and ForAll(a, IsString) then 
	        u := a;
        elif IsSPL(a) then
			if S=false then 
				S := a;
			else 
				Add(C, a); 
			fi;
		elif IsRuleTree(a) then 
			Add(C, a);
		elif IsList(a) then 
			Append(C, a);
		elif IsRule(a) and R=false then 
			R := a;
		else Error(
			"usage: \n",
			"  RuleTree( \n",
			"    <rule>, [ \"T\" ,] <non-terminal spl> \n",
			"    [, <children> ] [, <list-of-sploptions> ] )");

		fi;
    od;

    Constraint(IsRule(R));
    Constraint(IsSPL(S));

    if not IsList(u) and ForAll(u, IsString) then
        Error("<u> must be a list of spl options");
    elif not (IsList(C) and ForAll(C, c -> IsSPL(c) or IsRuleTree(c))) then
        Error("<C> must be a list of ruletrees or non-terminal spls");
    fi;

    RT := RuleTreeNC(R, S, C);
    if u <> [ ]
	    then RT.splOptions := u;
	fi;
    return RT;
end;


#F ExtractChildrenSPL ( <spl> )
#F   returns the list of non-terminal spls contained in <spl>.
#F   The order is left first - depth first (non-terminals are
#F   always leaves.
#F
ExtractChildrenSPL := S -> Collect(S, @.cond(IsNonTerminal));


#F CopyRuleTree( <ruletree> )
#F   returns a copy of <ruletree>. All fields apart from .children
#F   are copied by reference.
#F
CopyRuleTree := function( orig )
    local new;
    new := ShallowCopy(orig);
    if IsRuleTree(new) then 
	    new.children := List(orig.children, CopyRuleTree);
	fi;
    return new;
end;

_matchChildren := (C1, C2) ->
    Length(C1)=Length(C2) and
    ForAll([1..Length(C1)],
       j -> IsHashIdenticalSPL(HashAsSPL(C1[j]), HashAsSPL(C2[j])) or
            (ObjId(C1[j])=InfoNt and ObjId(C2[j])=InfoNt));

#F ApplyRuleTreeSPL( <ruletree>, <spl>, <opts> )
#F   creates a ruletree for <spl> using rules from the <ruletree>
#F   (if possible)
#F   Uses: opts.breakdownRules (for backwards compatibility of R.switch only)
ApplyRuleTreeSPL := function ( rtree, spl, opts )
    local i, R, C, C1, CC, children, L;
    Constraint(IsRuleTree(rtree));
    Constraint(IsSPL(spl));

    if rtree.transposed then
        return ApplyRuleTreeSPL(rtree.transpose(), spl.transpose(), opts).transpose();
    fi;

    R := rtree.rule;
    children:=[];

    # if <spl> was not compatible with <ruletree>.node to start with
    # then at some point there will be a split in <ruletree> that is not
    # applicable to the generated ruletree.
    if not IsApplicableRule( R, spl, opts.breakdownRules ) then
        Error("<rtree> can not be expanded from <spl>");
    fi;

    # determine the right set of children from allChildren to be used
    # in expansion
    C1 := _allChildren(R, rtree.node, opts);
    C := List([1..Length(rtree.children)], c -> rtree.children[c].node);
    i := PositionProperty([1..Length(C1)], j -> _matchChildren(C1[j], C));

    # DEBUG HACK: AppendTo("applytree", R, "\n", C, "\n", C1, "\n---------\n");
    if i = false then
        Error("<ruletree> can not be expanded from <spl>, could not find matching children"); 
	fi;

    # determine all children for <spl> and pick the right set
    L :=  _allChildren(R, spl, opts);
    if i > Length(L) then
        Error("<ruletree> can not be expanded from <spl>");
    else
        C1 := L[i];
    fi;

    # recurse and return
    children := List([1..Length(rtree.children)], i -> ApplyRuleTreeSPL(rtree.children[i], C1[i], opts));

    return RuleTreeNC(R, spl, children);
end;


#F Printing RuleTrees
#F ------------------
#F

#F RuleTreeOps.Print( <ruletree> [, <indent> , <indentStep> ] )
#F   prints <ruletree> with <indent>. Further indenting is done
#F   in steps of size <indentStep>. The default is
#F   indent = 0, indentStep = 2.
#F
RuleTreeOps.Print := function ( arg )
    local T, indent, indentStep;  
	if Length(arg) = 1 then
        T          := arg[1];
        indent     := 0;
        indentStep := 2;
    elif Length(arg) = 3 then
        T          := arg[1];
        indent     := arg[2];
        indentStep := arg[3];
    else
        Error("usage: RuleTreeOps.Print( <ruletree> [, <indent> , <indentStep> ] )");
    fi;
    Constraint(IsRuleTree(T));
    Constraint(IsPosInt0(indent));
    Constraint(IsPosInt0(indentStep));
    T.print(indent, indentStep);
end;

PrintRuleTreeCustom := function (T, i, is, spl_print, ruletree_print)
    local ch;
    Constraint(IsRuleTree(T) or IsSPL(T));
    Constraint(IsPosInt0(i));
    Constraint(IsPosInt0(is));

    if IsSPL(T) then 
        spl_print(T);
    else # T is a ruletree
        Print(Blanks(i), ruletree_print(T), "\n");
        for ch in T.children do
            PrintRuleTreeCustom(ch, i+is, is, spl_print, ruletree_print);
        od;
    fi;
end;

#F PrintNodesRuleTree(
#F   <ruletree/non-terminal spl> [, <indent> , <indentStep> ]
#F )
#F   pretty prints <ruletree> by displaying only the nodes (non-terminals)
#F   and its parameters. The tree structure is expressed by indentation.
#F   Non-terminals are being marked by (nt).
#F

PrintNodesRuleTree := function ( arg )
    local T, i, is;
    if Length(arg) = 1 then
        T := arg[1]; 
		i := 0; 
		is := 2;
    elif Length(arg) = 3 then
        T := arg[1];
		i := arg[2];
		is := arg[3];
    else
        Error("usage: PrintNodesRuleTree( <ruletree> [, <indent> , <indentStep> ] )");
    fi;
    PrintRuleTreeCustom(T, i, is,
        T -> Print(T, " (nt)"),
        T -> Print(T.node));
end;


#F PrintRulesRuleTree(
#F   <ruletree/non-terminal spl> [, <indent> , <indentStep> ]
#F )
#F   pretty prints <ruletree> by displaying only the rules and
#F   the node sizes in parentheses. Transposed rules are marked
#F   by ^T. The tree structure is expressed by indentation.
#F   Non-terminals are denoted by nt.
#F
PrintRulesRuleTree := function(arg)
    local T, i, is;
    if Length(arg) = 1 then
        T := arg[1]; i := 0; is := 2;
    elif Length(arg) = 3 then
        T := arg[1]; i := arg[2]; is := arg[3];
    else
        Error("usage: PrintRulesRuleTree( <ruletree> [, <indent> , <indentStep> ] )");
    fi;
    PrintRuleTreeCustom(T, i, is,
    T -> Print("nt (", T.dimensions[1], ")"),
    T -> Print(T.rule, When(T.transposed," ^ T",""), "(", T.node.dimensions[1], ")"));
end;



#F PrettyPrintRuleTree(
#F   <ruletree/non-terminal spl> [, <whiteIndent> [, <linedIndent> ] ]
#F )
#F   pretty prints <ruletree> by displaying only the nodes (non-terminals),
#F   their parameters, and the rules. The tree structure is expressed by
#F   indentation.  Non-terminals are being marked by (nt).
#F

PrettyPrintRuleTree := function ( arg )
    local T, chIndent, whiteIndent, linedIndent, oldLinedIndent, newline, params, i;

    # new line plus indents
    newline := function ( )
    local i;

    Print("\n");
    for i in [1..whiteIndent] do
        Print(" ");
    od;
    if not linedIndent = "" then
        Print( linedIndent, "--" );
    fi;
    end;

    # decode arg
    if Length(arg) = 1 then
        T           := arg[1];
        whiteIndent := 0;
        linedIndent := "";
    elif Length(arg) = 2 then
        T           := arg[1];
        whiteIndent := arg[2];
        linedIndent := "";
    elif Length(arg) = 3 then
        T           := arg[1];
        whiteIndent := arg[2];
        linedIndent := arg[3];
    else
        Error("usage: PrintNodesRuleTree( <ruletree> [, <indent> , <indentStep> ] )");
    fi;

    # check arguments
    if not ( IsRuleTree(T) or IsSPL(T) and T.type = "nonTerminal" ) then
        Error("<T> must be a ruletree or a non-terminal spl");
    fi;
    if not IsInt(whiteIndent) and whiteIndent >= 0 then
        Error("<whiteIndent> must be pos-int");
    fi;
    if not IsString(linedIndent) then
        Error("<linedIndent> must be a string");
    fi;

    chIndent := whiteIndent+Length(linedIndent)+4;
    # spl case
    if IsSPL(T) then
        T.print(chIndent, 2); Print(" (nt)");
        return;
    fi;
  
    T.node.print(chIndent, 2);
    Print("     {" );
    Print(T.rule);
    if T.transposed then
        Print(" ^ T");
    fi;
    if IsBound(T.splOptions) then
        Print(", ", T.splOptions);
    fi;
    Print( "}" );

    if Length(T.children) > 0 then
        if linedIndent = "" then
            oldLinedIndent := linedIndent;
        else
            oldLinedIndent := Concatenation( linedIndent, "  " );
        fi;
        linedIndent := Concatenation( oldLinedIndent, " |" );
        for i in [1..Length(T.children)-1] do
            newline();
            PrettyPrintRuleTree(T.children[i], whiteIndent, linedIndent);
        od;
        linedIndent := Concatenation( oldLinedIndent, " \`" );
        newline();
        linedIndent := Concatenation( oldLinedIndent, "  " );
        PrettyPrintRuleTree(T.children[Length(T.children)],
                            whiteIndent, linedIndent);
        linedIndent := oldLinedIndent;
    fi;
    if whiteIndent = 0 and linedIndent = "" then
        newline();
    fi;
end;


#F Converting RuleTrees
#F --------------------
#F

#F AMatRuleTree( <ruletree> ) - returns an amat corresponding to ruletree.
#F
AMatRuleTree := T -> AMatSPL(SPLRuleTree(T));

#F MatRuleTree( <ruletree> ) -  returns the matrix corresponding to ruletree.
#F
MatRuleTree := T -> MatSPL(SPLRuleTree(T));


#F Creating Random RuleTrees
#F -------------------------
#F

Declare(SemiRandomRuleTree, _SemiRandomRuleTree);

#F RandomRuleTree( <spl> , <opts>)
#F    returns a random rule tree for <spl>
#F    or 'false' if no breakdown rule combination leads to a
#F    fully expanded ruletree.
#F
#F   Uses: opts.baseHashes
#F         opts.breakdownRules
#F   Optional:  (see Doc(_allChildren) ) for usage
#F         opts.restrictSplit
#F         opts.restrictSplitSize
#F
RandomRuleTree := (spl, opts) -> _SemiRandomRuleTree(spl, false, s->false, opts);

RandomRuleTreeCutoff := (spl, cutoff_func, opts) -> _SemiRandomRuleTree(spl, false, cutoff_func, opts);

RandomRuleTreeDP := (t, opts) -> let(res := spiral.search.DP(t, rec(measureFunction := (R,O)->Random([1..10000]), verbosity := 0), opts), When(res = [], false, res[1].ruletree));

#F SemiRandomRuleTree( <spl>, <top_rule>, opts )
#F    returns a random rule tree for <spl> with fixed top-level rule
#F    or 'false' if no breakdown rule combination leads to a
#F    fully expanded ruletree.
#F
#F   Uses: opts.baseHashes
#F         opts.breakdownRules
#F   Optional:  (see Doc(_allChildren) ) for usage
#F         opts.restrictSplit
#F         opts.restrictSplitSize
#F
SemiRandomRuleTree := (spl, top_rule, opts) -> _SemiRandomRuleTree(spl, top_rule, s->false, opts);


_SemiRandomRuleTree := function(spl, top_rule, cutoff_func, opts)
    local ch, i, R, rules, rule, rand, candidates, ch, children, ch_candidates;

    if not (IsSPL(spl)) then
        Error("usage: SemiRandomRuleTree( <spl>, <top_rule>, <cutoff>, <opts> )");
    elif cutoff_func(spl) then
	    return spl;
    elif MultiHashLookup(opts.baseHashes, spl) <> false then
        return MultiHashLookup(opts.baseHashes, spl)[1].ruletree;
    else
        if top_rule <> false then
            rules := [top_rule];
        else
            rules  := AllApplicableRulesDirect(spl, opts.breakdownRules);
            if rules = [ ] then
                return false;
            fi;
        fi;

        candidates := Set([1..(Length(rules))]);
        while candidates <> [] do
            rand := RandomList(candidates);
            RemoveSet(candidates, rand);

            if rand <= Length(rules) then
                rule := rules[rand];
                ch_candidates := _allChildren(rule, spl,opts);
                ch_candidates := Permuted(ch_candidates, Random(SymmetricGroup(Length(ch_candidates))));
                for children in ch_candidates do
                    children := List( children, s -> _SemiRandomRuleTree(s, false, cutoff_func, opts) );
                    if not ForAny(children, c -> IsBool(c) and c=false) then
                        return RuleTreeNC( rule, spl, children );
                    fi;
                od;
            fi;
        od;
        return false;
   fi;
end;


#F Creating all RuleTrees
#F ----------------------
#F
#ExpandSPLRules := function( S, ruleset )
ExpandSPLRules := function( arg )
    local i, L, R, S, ruleset, opts;
    S := arg[1];
    ruleset := arg[2];
    Constraint(IsSPL(S));

    if IsBound(arg[3]) then
	    opts := arg[3];
    else 
	    opts := rec();
	fi;

    L  := [ ];
    for R in AllApplicableRulesDirect(S, ruleset) do
        Append(L, List(_allChildren(R,S,opts), c -> RuleTreeNC(R, S, c)));
    od;

    return L;
end;

#F ExpandSPL( <spl>, <opts> )
#F   performs one expansion of <spl> in all possible ways,
#F   and returns the list of the derived ruletrees.
#F   If there are no breakdown rules for <spl> (i.e., not a non-terminal)
#F   then the default base cases rule @_Base is used, which means that
#F   the recursion will continue on its children
#F
#F   Uses: opts.baseHashes
#F         opts.breakdownRules
#F
ExpandSPL := (S, opts) -> ExpandSPLRules(S, opts.breakdownRules, opts);


_AllRuleTrees := function ( S, cutoff, memohash, opts )
    local L, L1, T, Lc, i, cs, T1, lkup;

    # check cutoff, baseHashes, and our memoization hash
    if cutoff(S) then
		return [S];
	fi;
    lkup := MultiHashLookup(opts.baseHashes, S);
    if lkup <> false then
		return [lkup[1].ruletree];
	fi;

    lkup := HashLookup(memohash, S);
    if lkup <> false
		then return lkup;
	fi;

    L := ExpandSPL(S, opts);
    if ForAll(L, r -> r.children = [ ]) then
		HashAdd(memohash, S, L);
		return L;
    fi;

    L1 := [ ];
    for T in L do
        Lc := [ ];
		for i in [1..Length(T.children)] do
            Lc[i] := _AllRuleTrees(T.children[i], cutoff, memohash, opts);
		od;
		for cs in Cartesian(Lc) do
            T1 := ShallowCopy(T);
			T1.children := cs;
			Add(L1, T1);
		od;
    od;

    HashAdd(memohash, S, L1);
    return L1;
end;

#F AllRuleTrees( <spl>, <opts> )
#F   expands <spl> in all possible ways and returns the obtained list
#F   of fully expanded ruletrees (ruletrees containing no non-terminals).
#F   Note that subtrees are "by reference", i.e., modifying one subtree
#F   alters the same subtree in some other expansions.
#F
#F   Uses: opts.baseHashes
#F         opts.breakdownRules
#F   Optional:  (see Doc(_allChildren) ) for usage
#F         opts.restrictSplit
#F         opts.restrictSplitSize
#F
AllRuleTrees := (S, opts) -> _AllRuleTrees(S, x->false, HashTableSPL(), opts);

#F AllRuleTreesCutoff( <spl>, <cutoff_func>, <opts> )
#F   expands <spl> in all possible ways and returns the obtained list
#F   of fully expanded ruletrees (ruletrees containing no non-terminals).
#F   Note that subtrees are "by reference", i.e., modifying one subtree
#F   alters the same subtree in some other expansions.
#F   <cutoff_func> is of the form e -> e = <cutoff>
#F
#F   Uses: opts.baseHashes
#F         opts.breakdownRules
#F
AllRuleTreesCutoff := (S, cutoff_func, opts) -> _AllRuleTrees(S, cutoff_func, HashTableSPL(), opts);



_NofRuleTrees := function ( S, cutoff, memohash, opts, level, trace )
    local p, Cs, n, lkup, indentstr, i;
    Constraint(IsSPL(S));
    Constraint(IsRec(opts));

    if trace then
        indentstr := "";
        for i in [2..level] do
            indentstr := Concat(indentstr, ".  ");
        od;
        indentstr := Concat(indentstr, String(level), " ");
        Print(indentstr, ">> _NofRuleTrees(", S, ")\n");
    fi;

    # check cutoff, baseHashes, and our memoization hash
    if cutoff(S) then
        if trace then
            Print(indentstr, "<< (cutoff) 1\n");
        fi;
        return 1; 
    fi;
    if MultiHashLookup(opts.baseHashes, S) <> false then
        if trace then
            Print(indentstr, "<< (MultiHashLookup) 1\n");
        fi;
        return 1; 
    fi;
    lkup := HashLookup(memohash, S);
    if lkup <> false then
        if trace then
            Print(indentstr, "<< hashed: ", lkup, "\n");
        fi;
        return lkup; 
    fi;

    # first apply rules as they are
    Cs := List(AllApplicableRulesDirect(S, opts.breakdownRules), r -> _allChildren(r,S, opts));
	if trace then
        Print(indentstr, "   Cs: ", Cs, "\n");
    fi;

    if Cs = [ [ ] ] then
        n := 1;
    else
        n := Sum(Cs, c -> Sum(c, l -> Product(l, x -> _NofRuleTrees(x, cutoff, memohash, opts, level+1, trace))));
    fi;

    HashAdd(memohash, S, n);
  
    if trace then
        Print(indentstr, "<< (", S, ") ", n, "\n");
    fi;
  
    return n;
end;

#F NofRuleTrees ( <spl>, <opts> )
#F   returns the number of ruletrees for <spl>.
#F
#F   Uses: opts.baseHashes
#F         opts.breakdownRules
#F   Optional:  (see Doc(_allChildren) ) for usage
#F         opts.restrictSplit
#F         opts.restrictSplitSize
#F
#F   if opts.verbosity > 3 prints trace info

NofRuleTrees := (spl, opts) -> _NofRuleTrees(spl, x->false, HashTableSPL(), opts, 1, IsBound(opts.verbosity) and (opts.verbosity > 3));

NofRuleTreesCutoff := (spl, cutoff_func, opts) -> _NofRuleTrees(spl, cutoff_func, HashTableSPL(), opts, 1, IsBound(opts.verbosity) and (opts.verbosity > 3));


#F RulesInRuleTree( <ruletree> )
#F   return the set of rules contained in <ruletree>.
#F
RulesInRuleTree := function ( T )
    local C;
    Constraint(IsRuleTree(T));

    # base case
    if T.children = [ ] then
	    return Set([T.rule]);
	fi;

    # recurse with ruletree children
    C := Filtered(T.children, IsRuleTree);
    C := Set(Concatenation(List(C, RulesInRuleTree)));
    AddSet(C, T.rule);
    return C;
end;


#F RulesInRuleTreeAll( <ruletree> )
#F   return the histogram of all rules contained in <ruletree> as set
#F   of pairs [rule, number].
#F
RulesInRuleTreeAll := function ( T )
    local C, C1, c, i;
    Constraint(IsRuleTree(T));

    # base case
    if T.children = [ ] then
	    return Set([ [T.rule, 1] ]);
	fi;

    # recurse with ruletree children
    C := Filtered(T.children, IsRuleTree);
    C := Concatenation(List(C, RulesInRuleTreeAll));

    # merge
    C1 := [ ];
    for c in C do
        i := PositionProperty(C1, p -> c[1] = p[1]);
        if i = false then
            AddSet(C1, c);
        else
            C1[i][2] := C1[i][2] + c[2];
        fi;
    od;

    i := PositionProperty(C1, p -> p[1] = T.rule);
    if i = false then
        AddSet(C1, [T.rule, 1]);
    else
        C1[i][2] := C1[i][2] + 1;
    fi;
        return C1;
end;


ApplyRuleTreeStep := function ( rt, recurse )
    local nt, C, Nonterms, S, c;
    if IsNonTerminal(rt) or IsSPL(rt) then 
	    return rt;
	fi;

    Constraint(IsRuleTree(rt));

    C := List(rt.children, recurse);
    nt := rt.node;
    Nonterms := List(rt.children, c -> When(IsRuleTree(c), c.node, c));

    if rt.transposed then
        nt := TransposedSPL(nt);
        C := List(C, TransposedSPL);
        Nonterms := List(Nonterms, TransposedSPL);
    fi;

    S := _apply(rt.rule, nt, C, Nonterms);
    if rt.transposed then 
	    S := TransposedSPL(S); 
	fi;

    trace_log.addTreeExpansion(rt.rule,nt,S,rt.children,CopyFields(rt.node, rec(params := C)), var);

    S.root := rt.node;
    return S;
end;

SPLRuleTreeStep := rt -> ApplyRuleTreeStep(rt, c -> RecursStep(
    When(IsRuleTree(c), RTWrap(c), c)));

_SPLRuleTree := rt -> ApplyRuleTreeStep(rt, _SPLRuleTree);

SPLRuleTree := function(rt)
    local spl, t, tag;
    trace_log.beginStage("tSPL","SPL", rt.node);    
    spl := ApplyRuleTreeStep(rt, _SPLRuleTree);

    # tags may inject container objects
    t := rt.node;
    if IsBound(t.getTags) then
        for tag in Reversed(t.getTags()) do
            if IsBound(tag.container)
			    then spl := tag.container(spl);
			fi;
        od;
    fi;
    trace_log.endStage("tSPL","SPL", spl);    
    return spl;
end;

# forward declaration for function that does all the work.
Declare(_ruleTreeN);

#F RandomRuleTrees( <spl>, <num>, <opts>)
#F    returns <num> random rule trees for <spl>
#F    uniformly distributed amongst the potential trees
#F    or 'false' if no breakdown rule combination leads to a
#F    fully expanded ruletree.
#F
#F   Uses: opts.baseHashes
#F         opts.breakdownRules
#F   Optional:  (see Doc(_allChildren) ) for usage
#F
RandomRuleTrees := function(spl, num, opts)
    local n, rand, memohash;

    memohash := HashTableSPL();

    n := _NofRuleTrees(spl, s -> false, memohash, opts, 1, false);

    if num >= n then
        rand := [1..n];
    else
        rand := [];
        while Length(rand) <> num do
            rand := Set(Concat(rand, [RandomList([1..n])]));
        od;
    fi;

    return List(rand, e -> _ruleTreeN(spl, e-1, memohash, opts));
end;

# helper functions/arrays for _ruleTreeN and _ruleTree1

_getNumTrees := (s, memohash, opts) -> let(
    n := HashLookup(memohash, s), 
    When(n = false,
        _NofRuleTrees(s, e -> false, memohash, opts, 1, false),
        n
    )
);
#F RuleTreeN(<spl>, <num>, <opts>)
#F   returns ruletree of index <num>
#F
#F   Uses: opts.basHashes, opts.breakdownRules.
#F
RuleTreeN := function(spl, num, opts)
    local memohash, n, r;

    memohash := HashTableSPL();
    n := _NofRuleTrees(spl, s -> false, memohash, opts, 1, false);

    if num in [1..n] then
        r := _ruleTreeN(spl, num-1, memohash, opts);

    else
        r := false;
    fi;

    return r;
end;


# fold list inside out, favoring the right hand side

_foldedMidFirst := function(origlist)
	local len, midpt, front, back, newlist;
	
	len := Length(origlist);
	if len < 2 then
		return origlist;
	fi;
	midpt := Int(len/2);
	front := Reversed(Sublist(origlist, [1..midpt]));
	back  := Sublist(origlist, [midpt+1..len]);
	
	newlist := [];
	
	if IsOddInt(len) then
		Append(newlist, [back[1]]);
		back := Drop(back,1);
	fi;
	
	while (front <> []) or (back <> []) do
		if back <> [] then
			Append(newlist, [back[1]]);
			back := Drop(back,1);
		fi;
		if front <> [] then
			Append(newlist, [front[1]]);
			front := Drop(front, 1);
		fi;	
	od;
	
	return newlist;
end;


Declare(_ruleTreeMid);

#F RuleTreeMid(spl, opts)
#F
#F   Similar to RuleTree1, but tries to grab the middle of each range of choices
#F 

RuleTreeMid := (spl, opts) -> _ruleTreeMid(spl, HashTableSPL(), opts);

_ruleTreeMid := function(spl, memohash, opts)
    local h, R, r, C, c, ch, npc, idx, idxlist, ridx, ridxlist;

    h := MultiHashLookup(opts.baseHashes, spl);

    if h <> false then
        return h[1].ruletree;
    fi;

    R := AllApplicableRulesDirect(spl, opts.breakdownRules);
	ridxlist := [1..Length(R)];
	if Length(ridxlist) > 1 then
		if Length(_allChildren(R[1], spl, opts)) < 2 then
			ridxlist := _foldedMidFirst(ridxlist);
		fi;
	fi;
    for ridx in ridxlist do
		r := R[ridx];
		C := _allChildren(r, spl, opts);
		idxlist := _foldedMidFirst([1..Length(C)]);
		for idx in idxlist do 
			c := C[idx];
			npc := Product(c, e -> _getNumTrees(e, memohash, opts));
			if npc > 0 then
				ch := List(c, e -> _ruleTreeMid(e, memohash, opts));
				if not ForAny(ch, e -> IsBool(e) and e=false) then
					return RuleTreeNC(r, spl, ch);
				fi;
			fi;

        od;
    od;
	# no tree, return original spl
    return spl;
end;


Declare(_ruleTree1);

#F RuleTree1(spl, opts)
#F
#F   returns the first ruletree. this is often useful for
#F   testing whether a particular rule is firing, and you
#F   want to avoid the randomness of RandomRuleTree
#F
#F   NOTE: this function does not check if a tree can
#F   actually just be created, it just tries to return
#F   the first one. RuleTreeN(spl, 1, opts) does the
#F   same thing but first checks.
#F 

RuleTree1 := (spl, opts) -> _ruleTree1(spl, HashTableSPL(), opts);

_ruleTree1 := function(spl, memohash, opts)
    local h, R, r, i, c, ch, npc;

    h := MultiHashLookup(opts.baseHashes, spl);

    if h <> false then
        return h[1].ruletree;
    fi;

    R := AllApplicableRulesDirect(spl, opts.breakdownRules);
    for r in R do
		for c in _allChildren(r, spl, opts) do 
			npc := Product(c, e -> _getNumTrees(e, memohash, opts));
			if npc > 0 then
				ch := List(c, e -> _ruleTree1(e, memohash, opts));
				if not ForAny(ch, e -> IsBool(e) and e=false) then
					return RuleTreeNC(r, spl, ch);
				fi;
			fi;
        od;
    od;
	# no tree, return original spl
    return spl;
end;
    

#F _ruleTreeN(<spl>, <num>, <memohash>, <opts>)
#F   internal function which returns a ruletree for 
#F   <spl> with index given by <num>. This is a recursive
#F   routine. Note that <num> is 0-based, unlike everything
#F   else in SPIRAL, which is 1-based.
#F 
_ruleTreeN := function(spl, num, memohash, opts)
    local R, rtNC, h, r, c, npc, off, ch, i, n, p, offset;

    # lookup stuff like SSE base cases.
    h := MultiHashLookup(opts.baseHashes, spl);

    if h <> false then
       return h[1].ruletree;
    fi;

    # get list of normal rules
    R := AllApplicableRulesDirect(spl, opts.breakdownRules);
    offset := 0;

    for r in R do
        for c in _allChildren(r, spl, opts) do 
            # figure out which group of children we should be looking at.
            npc := Product(c, e -> _getNumTrees(e, memohash, opts));

            # is this the correct group?
            if num < offset + npc then
                off := npc;
                ch := [];

                # step through the kids, and push down the ruletree number
                for i in [1..Length(c)] do
                    n := _getNumTrees(c[i], memohash, opts);

                    off := When(n > 0, off / n, off);

                    p := When(n <= 0 or num <= offset, 
                        0, 
                        RemInt(QuoInt(num - offset, off), n)
                    );

                    # all the kids get evaluated
                    Add(ch, _ruleTreeN(c[i], p, memohash, opts));
                od;

                # no kids should be false, but just in case.
                if not ForAny(ch, e -> IsBool(e) and e=false) then
                    return RuleTreeNC(r, spl, ch);
                fi;
            fi;

            # fixup offset
            offset := offset + npc;
        od;
    od;

	# no tree, return original spl
    return spl;
end;

Declare(_ruleTreeGetN);

#F RuleTreeGetN(<ruletree>, <opts>)
#F given a ruletree, determine the index
#F
#F uses:
#F      opts.breakdownRules
#F      opts.baseHashes
#F

RuleTreeGetN := function(ruletree, opts)
    local memohash, r;
    
    memohash := HashTableSPL();

    # prime the memohash.
    _NofRuleTrees(ruletree.node, s -> false, memohash, opts, 1, false);

    r := _ruleTreeGetN(ruletree, memohash, opts, 0);

    # ruleTreeGetN may return false if we couldn't match the tree in the passed
    # in breakdownRules/baseHashes.
    return When(IsInt(r), r + 1, false);
end;

_ruleTreeGetN := function(rt, memohash, opts, offset)
    local h, R, r, c, ch, off, n;

    # is the ruletree something from the baseHashes?
    h := MultiHashLookup(opts.baseHashes, rt.node);
    if IsList(h) and rt = h[1].ruletree then
        return offset;
    fi;

    R := AllApplicableRulesDirect(rt.node, opts.breakdownRules);

    for r in R do
        # does rule match?
        if r = rt.rule then
            for c in _allChildren(r, rt.node, opts) do
                # does child match?
                if c = List(rt.children, e -> e.node) then
                    off := Product(rt.children, e -> _getNumTrees(e.node, memohash, opts));
                    for ch in rt.children do
                        n := _getNumTrees(ch.node, memohash, opts);
                        off := When(n > 0, off / n, off);
                        offset := offset + (off * _ruleTreeGetN(ch, memohash, opts, 0));
                    od;
                    return offset;
                # child doesn't match, increase offset
                else
                    offset := offset + Product(c, e -> _getNumTrees(e, memohash, opts));
                fi; 
            od;
        # rule doesn't match, so increase offset by all possible expansions we didn't use
        else
            offset := offset 
                + Sum(_allChildren(r, rt.node, opts), C ->  
                    Product(C, c -> _getNumTrees(c, memohash, opts))
                );
        fi;
    od;

    return offset;
end;
