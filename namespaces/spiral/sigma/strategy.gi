
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# RuleStrategy := [
#     RulesSums, RulesFuncSimp, RulesGMonTensorPullOut, RulesFuncSimp,
#     RulesGMonToSGMon, RulesFuncSimp,
#     RulesSGMonTensorPullIn, RulesSums, RulesFuncSimp
# ];

StandardSumsRules := MergedRuleSet(
    RulesSums, RulesFuncSimp, RulesDiag, RulesDiagStandalone, RulesStrengthReduce, RulesRC, RulesII, OLRules
);

StandardSumsRulesNoRC := MergedRuleSet(
    RulesSums, RulesFuncSimp, RulesDiag, RulesDiagStandalone, RulesStrengthReduce, RulesII, OLRules
);

HfuncSumsRules := MergedRuleSet(
    StandardSumsRules, RulesHfunc
);

LibStrategy := [ StandardSumsRules, HfuncSumsRules ]; 

# HasRM := e -> ObjId(e) in [Gath, Scat] and e.func.range() > 16 and
#               Collect(e.func, @(1,RM,e->e.range()>16)) <> [];

# PrecomputeRM := s -> SubstBottomUp(s, @.cond(HasRM), e->ObjId(e)(GenerateData(e.func)));

#F RecompileStrategies(<opts>)
#F
RecompileStrategies := function(opts)
    local fld, lst, rset;
    for fld in UserRecFields(opts.formulaStrategies) do
        lst := opts.formulaStrategies.(fld);
        if not IsList(lst) then lst := [lst]; fi;
        for rset in lst do
            if IsRec(rset) and IsBound(rset.compile) then rset.compile(); fi;
        od;
    od;
end;
