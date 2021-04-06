
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


FindUnexpandableNonterminal := function(t, opts)
    local trees, res;
    Constraint(IsSPL(t));
    trees := ExpandSPL(t, opts);
    if trees=[] then return [t];
    else
        return ConcatList(trees, tr -> 
	    ConcatList(tr.children, c -> FindUnexpandableNonterminal(c, opts)));
    fi;
end;

RuleTreeClass_rChildren := self >> [self.node, self.children];
RuleTreeClass_rSetChild := rSetChildFields("node", "children");
   

EnableRuleTreeRewriting := function(switch)
    if switch then
        RuleTreeClass.rChildren := RuleTreeClass_rChildren;
        RuleTreeClass.rSetChild := RuleTreeClass_rSetChild;
    else 
        Unbind(RuleTreeClass.rChildren);
        Unbind(RuleTreeClass.rSetChild);
    fi;
end;

VerifyRulesInRuleTree := function(r, opts)
    local nt, n, bad;
    EnableRuleTreeRewriting(true);
    nt := CollectNR(r, @.cond(IsNonTerminal));
    bad := [];
    for n in nt do
        #if n.free() <> [] then 
            n:=HashAsSPL(n); 
        #fi;
        if ObjId(n) <> InfoNt then
            Print(Red("-------", n, "-------"), "\n");
            if VerifyRulesForSPL(n, opts) = false then
                Add(bad, n);
            fi;
        fi;
    od;
    EnableRuleTreeRewriting(true);
    Print(Red("Failed:\n    "), bad, "\n");
end;

_VerifySubRuleTrees := function(r, opts, rt_tomat_func, ind)
    local subtrees, rt, bugs, hashrt, t, innerbugs;
    EnableRuleTreeRewriting(true);
    bugs := [];
    for rt in r.children do
        t := HashAsSPL(rt.node);
	hashrt := ApplyRuleTreeSPL(rt, t, opts);	
        if InfinityNormMat(MatSPL(rt_tomat_func(hashrt)) - MatSPL(hashrt.node)) > 1e-13 then
	    PrintLine(Blanks(ind), hashrt.node, RedStr("  FAIL"));
	    bugs := [hashrt];
	    innerbugs := _VerifySubRuleTrees(hashrt, opts, rt_tomat_func, ind+4);
	    return Cond(innerbugs=[], bugs, innerbugs);
	else 
	    PrintLine(Blanks(ind), hashrt.node, GreenStr("  OK"));
	fi;
    od;
    EnableRuleTreeRewriting(false);
    return bugs;
end;

VerifySubRuleTreesSPL := (r, opts) -> _VerifySubRuleTrees(r, opts, SPLRuleTree, 0);

#VerifySubRuleTreesSums := (r, opts) -> _VerifySubRuleTrees(r, opts, x->SumsRuleTree(x, opts), 0);
