# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_unTspl := (c, info) -> When(spiral.paradigms.common.IsTSPL(c), SumsSPL(c.toSpl(), info.opts), c);

SumsRuleTreeStep := function ( rtwrap, info )
  local S, rt, t, tag, tags, processed_tags;
  rt := rtwrap.rt;
  info.rsteps := info.rsteps + 1;

  tags := When(IsBound(rt.node.getTags), rt.node.getTags(), []);

  if IsBound(rt.node.noCodelet) then
      S := SumsSPL(_SPLRuleTree(rt), info.opts);

  else
      processed_tags := Union(info.processed_tags, List(tags, x->x.kind()));
      S := ApplyRuleTreeStep(rt, c -> Cond(
          not IsRuleTree(c) and IsBound(c.noCodelet) and c.noCodelet, 
	      _unTspl(c, info), 
	  not IsRuleTree(c), 		  
	      RecursStep(_unTspl(c, info)).setA(processed_tags => processed_tags),
          IsBound(c.node.noCodelet) and c.node.noCodelet, 
	      _SPLRuleTree(c),
	  # else
              RecursStep(RTWrap(c)).setA(processed_tags => processed_tags)
      ));
      S := SumsSPL(S, info.opts);
      S.root := rt.node;
  fi;

  # tags may inject container objects
  for tag in Reversed(tags) do
      if IsBound(tag.container) and not (tag.kind() in info.processed_tags) then
          S := tag.container(S); 
      fi;
  od;
  return S;
end;

SumsRecursStep := function ( rstep, info )
  local rsteps;
  rsteps := info.rsteps;
  info.processed_tags := When(IsBound(rstep.a.processed_tags), rstep.a.processed_tags, []);
  rstep := SubstTopDownNR(rstep, RTWrap, e -> SumsRuleTreeStep(e, info));
  if rsteps = info.rsteps then # nothing has changed
      return rstep;
  else
      rstep := ApplyStrategy(rstep, info.strategy, UntilDone, info.opts);
      return rstep.child(1); # strip outer RecursStep container
  fi;
end;

Recurse := function(sums,info)
    local rt;
    if IsRuleTree(sums) then
	rt := sums;
	# keep track of tags that were converted into containers, via an attribute
	# this mechanism does not work well, and more VContainers will be created
	# than neeeded, however this is safe, because redundant ones are eliminated
	# by rewrite rules.
        return Recurse(RecursStep(RTWrap(rt)).setA(processed_tags=>[]), info);

    else
        return SubstTopDownNR(sums, RecursStep, x -> SumsRecursStep(x,info));
    fi;
end;

SumsRuleTreeStrategy := function ( rt, strategy, opts )
    local info;
    info := rec(rsteps := 0, strategy := strategy, opts := opts); #, cutoff := cutoff_func);
    rt := Recurse(rt, info);
    while info.rsteps > 0 do info.rsteps := 0; rt := Recurse(rt, info); od;
    return rt;
end;

#F SumsRuleTree(<rt>, <opts>)
#F
#F <opts> flags used:
#F   opts.formulaStrategies.sigmaSpl    Sigma-SPL rewriting strategy
#F   opts.formulaStrategies.rc          RC(.) rewriting strategy
#F   opts.generateComplexCode == bool   if set to false, then RC rewriting strategy is applied
#F
SumsRuleTree := function(rt, opts)
        local sums, rsums, t, tag;
        if IsNonTerminal(rt) then rt := RandomRuleTree(rt,opts); fi;
        sums := SumsRuleTreeStrategy(rt, [], opts);
        sums := ApplyStrategy(sums, opts.formulaStrategies.sigmaSpl, UntilDone, opts);
        sums := ApplyStrategy(sums, opts.formulaStrategies.preRC, UntilDone, opts);
        if (not opts.generateComplexCode) and (not rt.node.isReal()) then # NOTE: when t_in/t_out available check them instead of isReal
            rsums := ApplyStrategy(RC(sums), opts.formulaStrategies.rc, UntilDone, opts);
        else
            rsums := sums;
        fi;
        rsums := ApplyStrategy(rsums, opts.formulaStrategies.postProcess, UntilDone, opts);
        rsums := SumsUnification(rsums, opts);
        rsums.ruletree := rt;
     
        return rsums;
end;

#F See SumsRuleTree
#F
SumsRuleTreeOpts := SumsRuleTree;

RecurseOpts := function(rt, opts)
    local sums, rsums;
    if IsNonTerminal(rt) then rt := RandomRuleTree(rt,opts); fi;
    sums := Recurse(rt, rec(strategy:=opts.formulaStrategies.sigmaSpl, rsteps:=0, opts := opts));
    return sums;
    #return ApplyStrategy(sums, opts.formulaStrategies.postProcess, UntilDone, opts);
end;

######
SumsVerifyRulesForSPL := (S,opts) -> VerifyRules(S,
    (rt, s) -> InfinityNormMat(MatSPL(SumsRuleTree(rt,opts)) - MatSPL(s)) < 1e-11,
    opts);