
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_dbgPrintRuleName := rule -> Cond(
    IsBound(rule.owner),   
        Print(spiral._color(Blue, rule.name), "   /  ", 
	      spiral._color(DarkBlue, rule.owner), "\n"),
    Print(spiral._color(Blue, rule.name), "\n")
);

_dbgRuleBelongsTo := (rule, rset) -> 
    IsBound(rule.owner) and (rule.owner in rset);

#F DebugRewriting(<rsets.)
DebugRewriting := function(rsets)
    if rsets=true then
        rewrite.RuleTrace := rule -> _dbgPrintRuleName(rule); 
        rewrite.RuleStatus := (rule, hd, str) -> Print(
	    spiral._color(Yellow, hd), ApplyFunc(Print, str));
    elif rsets=false then
        rewrite.RuleTrace := Ignore;
        rewrite.RuleStatus := Ignore;
    else
		if not IsList(rsets) then 
			rsets := [rsets];
		fi;
        rewrite.RuleTrace := rule -> 
			When(_dbgRuleBelongsTo(rule, rsets), _dbgPrintRuleName(rule));
        rewrite.RuleStatus := (rule, hd, str) -> 
			When(_dbgRuleBelongsTo(rule, rsets), 
		Print(spiral._color(Yellow, hd), ApplyFunc(Print, str)));
    fi;
end;

DebugRuleStrategies := function(switch)
    Constraint(IsBool(switch));
    if switch then
        rewrite.RuleTrace := rule -> _dbgPrintRuleName(rule); 
        rewrite.RuleStrategyTrace := (i, rset, expr) -> Print(
			spiral._color(Red, i), 
			spiral._color(Red, " ----------------\n"), 
			rset, "\n",
			Doc(rset), 
			expr, "\n"); 
    else
        rewrite.RuleTrace := Ignore;
        rewrite.RuleStrategyTrace := Ignore;
    fi;
end;

