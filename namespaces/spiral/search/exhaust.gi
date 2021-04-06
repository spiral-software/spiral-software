
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# dummy to restart doc string
_dummy := 0;





#F ExhaustiveSearch(spl, opts)
#F

ExhaustiveSearch := function(spl, opts)
	local idxList, resList, bestRec;
	
	if not IsSPL(spl) then
		Error("invalid SPL");
	fi;
	
	idxList := [1 .. NofRuleTrees(spl, opts)];
	
	resList := TimeRuleTrees(spl, opts, idxList);

	bestRec := BestTimedRuleTree(resList);
	bestRec.ruletree := RuleTreeN(spl, bestRec.index, opts);
	
	return bestRec;
end;
