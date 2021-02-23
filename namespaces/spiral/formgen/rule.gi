
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F Rules
#F =====
#F
#F A rule is a record with the following mandatory fields:
#F
#F   isRule           = "true"         # identifies rules
#F   operations       = RuleOps        # operations record
#F   name             = <string>       # the name of the rule
#F   info             = <string>       # a string with info about the rule
#F   nonTerminal      = <symbol>       # the non-terminal the rule is meant
#F                                       for, given by its symbol (e.g. "WHT")
#F   forTransposition = true/false     # identifies whether the should also
#F                                       be used in its transposed form
#F   switch           = true/false     # allows to switch rules on (true) or off
#F   isApplicable     = func( params ) # rule applicable for params?
#F   allChildren      = func( params ) # returns a list of ordered lists of
#F                                       all possible children for the rule 
#F   randomChildren   = func( params ) # returns an ordered list of children
#F                                       for the rule, chosen at random
#F   rule             =                # the actual rule, given as a formula
#F     func( params, children )          combining children
#F
#F The following are optional fields:
#F
#F The function .isDerivable checks whether a given set of children can
#F be derived from a given spl. This can also be decided by using doing
#F   children in .allChildren(spl.params)
#F In some cases, namely when there are to many children configurations, 
#F this field should be present. It is only used in the function RuleTree
#F and nowhere during code generation.
#F
#F   isDerivable = func( spl, children )
#F
#F Note that the field "nonTerminal" must contain a symbol known
#F from the NonTerminalTable in spl.g
#F All rules have to be in the global variable RuleTable in order
#F to be used.
#F
#F
#F new rule:
#F    .children := nt -> list of ordered lists of all possible children 
#F    .applicable := nt -> true/false
#F    .apply := (nt, children) -> apply the rule
#F    or .apply := (nt, children, child_nonterms) -> apply the rule
#F Functions for Rules
#F -------------------
#F

RuleOps := OperationsRecord("RuleOps");

#F IsRule( <rule> ) - true for breakdown rules (base class = BreakdownRule)
#F
IsRule := R -> IsRec(R) and IsBound(R.isRule) and R.isRule = true;

#F IsNewRule(<rule>) - true for new style rules (base class = NewBreakdownRule)
#F
IsNewRule := R -> IsRec(R) and IsBound(R.isRule) and R.isRule and IsBound(R.isNewRule) and R.isNewRule;

#F IsAlternativeRewriteRule(<rule>) - true for new style rules (base class = AlternativeRewriteRule)
#F
IsAlternativeRewriteRule := R -> IsRec(R) and IsBound(R.isRule) and R.isRule and IsBound(R.isAlternativeRewriteRule) and R.isAlternativeRewriteRule;

#F RuleOps.Print( <rule> ) -  prints the name of <rule>.
#F
RuleOps.Print:= R -> Print(R.name, Cond(IsBound(R.a), R.printA(), ""));

#F RuleOps.\=( <rule1>, <rule2> ) - equality of rules based on names
#F
RuleOps.\= := (R1, R2) -> IsRule(R1) and IsRule(R2) and R1.name = R2.name and 
    When(IsBound(R1.a), R1.a = R2.a, true);

#F RuleOps.\<( <rule1>, <rule2> ) - ordering of rules based on their names. 
#F
RuleOps.\< := (R1, R2) -> Cond(
    IsRule(R1) and IsRule(R2), 
        R1.name < R2.name or (R1.name = R2.name and When(IsBound(R1.a), R1.a < R2.a, true)),
    ObjId(R1) < ObjId(R2));


# Default rule for SPLs which not non-terminals
Declare(@_Base);

_applicable := (R, nt, ruleset) -> 
    (not Same(ruleset, ApplicableTable) or R.switch) and 
    (R.nonTerminal=@ or R.nonTerminal = ObjId(nt)) and
    ((nt.transposed = R.transposed) or (nt.transpose()=nt)) and 
    # R.requiredFirstTag is obsolete
    When(not IsNewRule(R) or not IsBound(R.requiredFirstTag), true,
	When(IsBound(nt.firstTag), 
            When(IsList(R.requiredFirstTag), 
                 nt.firstTag().kind() in R.requiredFirstTag,
                 nt.firstTag().kind()  = R.requiredFirstTag), false)) and
    # R.a.requiredFirstTag is the better way of setting mandatory tags
    When(not IsNewRule(R) or not IsBound(R.a.requiredFirstTag), true,
	When(IsBound(nt.firstTag), 
            When(IsList(R.a.requiredFirstTag), 
                 nt.firstTag().kind() in R.a.requiredFirstTag,
                 nt.firstTag().kind()  = R.a.requiredFirstTag), false)) and
    When(IsNewRule(R), 
            SReduceSimple(R.applicable(nt)) <> false, 
            R.isApplicable(nt.params));
    
#F _allChildren( <rule>, <non-terminal>[, <opts>])
#F   returns all possible children obtained by applying <rule> to <non-terminal>
#F   Use of opts.restrictSplit and opts.restrictSplitSize
#F      Set opts.restrictSplit:=true and opts.restrictSplitSize to the desired
#F      size.
#F      Can be used to obtain ruletrees with no leaves <= the specified size
#F      If this is used there will be a problem with generating ruletrees
#F      unless opts.baseHashes is set so that small sizes are fetched from the 
#F      hash table. Otherwise generating rule trees will return false

 #_allChildren := (R, nt) -> When(IsNewRule(R), R.children(nt), R.allChildren(nt.params));
_allChildren := function( arg )
    local flag, i, R, nt, opts, ch, ch_copy, children;
    R := arg[1];
    nt := arg[2];
    children :=  When(IsNewRule(R), R.children(nt), R.allChildren(nt.params));
    if IsBound(arg[3]) then 
        opts := arg[3]; 
        ch_copy := Copy(children);
        flag := false;
        i := 1;
        if IsBound(opts.restrictSplit) and opts.restrictSplit = true and
                                         IsBound(opts.restrictSplitSize) then
            for ch in ch_copy do
                while flag=false and i<=Length(ch) do
                    if ch[i].params[1] <= opts.restrictSplitSize then
                        flag := true;
                    fi;
                    i := i+1;
                od;
                if flag then
                    RemoveSet(children, ch);
                    flag := false;
                fi;
                i := 1;
                    
            od;
        fi;
    fi;
    return children;
end;
# children[i] could be a complicated SPL expansion of the nonterms[i]
# or children could be same as nonterms, depending on the workflow you are using
#_apply := (R, nt, children, nonterms) -> Checked(IsRule(R), IsSPL(nt),
#    When(IsNewRule(R), 
#	 R.apply(nt, children, nonterms),
#	 When(NumGenArgs(R.rule)=2, 
#	      R.rule(nt.params, children),
#	      R.rule(nt.params, children, nonterms)))
#);

# Current rule wrap for fftx needs attributes from nt object. 
# The wrapper has var. arg. This fact is used here
# to recognize the wrapper and pass nt instead of its params.
_apply := (R, nt, children, nonterms) -> Checked(IsRule(R), IsSPL(nt),
    When(IsNewRule(R), 
     R.apply(nt, children, nonterms),
     Cond(NumGenArgs(R.rule)=-1,
            R.rule(nt, children, nonterms),
            NumGenArgs(R.rule)=2, 
            R.rule(nt.params, children),
            R.rule(nt.params, children, nonterms)))
);

#F IsApplicableRule( <rule>, <non-terminal>, <ruleset> )
#F   returns true if <rule> can be applied to <non-terminal>
#F   and false else.
#F
IsApplicableRule := (R, nt, ruleset) -> When(IsAlternativeRewriteRule(R),
    Length(Collect(nt, R.pattern))<>0
    ,
    Checked(IsRule(R), IsSPL(nt),
    (not Same(ruleset, ApplicableTable) or R.switch) and 
    (
	_applicable(R, nt, ruleset) or 
	(R.forTransposition and _applicable(R, nt.transpose(), ruleset)))));

#F ApplyRuleSPL( <rule>, <non-terminal> )
#F   returns result of application of <rule> to <non-terminal>, if
#F   it is non-applicable an error is reported
#F
ApplyRuleSPL := (R, nt) -> 
	Checked(IsRule(R), IsSPL(nt), 
		let(c := _allChildren(R,nt)[1],	_apply(R, nt, c, c))
	);

AllApplicableRulesDirect := (spl, ruleset) -> 
    Concatenation(
		When(not IsBound(ruleset.(spl.__name__)), [ ],
			Filtered(ruleset.(spl.__name__), r -> not r.forTranspositionOnly and _applicable(r, spl, ruleset))),
		Filtered([@_Base], r ->not r.forTranspositionOnly and _applicable(r, spl, ruleset))
	);

#F AllApplicableRules( <non-terminal>, <ruleset> )
#F   returns list of all rules applicable to a non-terminal
#F
AllApplicableRules := (nt, ruleset) -> 
	Checked(IsSPL(nt), Set(AllApplicableRulesDirect(nt, ruleset)));

#F RandomChildrenRule( <rule>, <non-terminal>, <ruleset> )
#F   returns a random set of children for <rule> applied to <spl>.
#F   Calls the rule's randomChildren function if available.
#F   Otherwise, calls RandomList on the rule's allChildren function.
#F
RandomChildrenRule := (R, nt, ruleset) -> Checked(IsApplicableRule(R, nt, ruleset),
    When(IsBound(R.randomChildren), 
	 R.randomChildren(nt.params),
	 RandomList(_allChildren(R, nt))));

#F Verification of Rules
#F ---------------------
#F

#F VerifyRules(<non-terminal>, <verify-func>)
#F
#F  Expands <non-terminal> using all applicable rules and all
#F  possible children sets (just one expansion step) and runs
#F  <verify-func> on the obtained partial ruletrees and the non
#F  terminal to check correctness.
#F
#F  verify_func = (rt, nt) -> boolean 
#F      <rt> is partial ruletree, 
#F      <nt> is original non-terminal
#F 
#F  Example:
#F   VerifyRulesForSPL := nt -> 
#F       VerifyRules(nt, (rt, nt) -> 
#F           InfinityNormMat(MatSPL(SPLRuleTree(rt)) - MatSPL(nt)) < 1e-11);
#F 
VerifyRules := function ( nt, verify_func, opts )
  local rule, Csets, C, ruletree, res, direct, transp, transp_nt;
  Constraint(IsSPL(nt));
  res := true;

  for ruletree in ExpandSPL(nt, opts) do 
      Print("-- ", ruletree, " --\n");
      Print("checking rule ", ruletree.rule, ": ");	
      if verify_func(ruletree, nt) then
	  Print(Green("correct\n"));
      else
	  Print(Red("incorrect!\n"));
	  res := false;
      fi;
  od;
  return res;
end;


_checkMatRuleTree := function(rt, nt)
    local me, them, diff;
    me := MatSPL(SPLRuleTree(rt));
    them := MatSPL(nt);
    diff := InfinityNormMat(me-them);
    return diff < 1e-11;
end;

#F VerifyRulesForSPL( <non-terminal>, <opts> )
#F   expands <non-terminal> using all applicable rules and all
#F   possible children sets (just one expansion step) and checks 
#F   whether the resulting matrix matches the non-terminal matrix.
#F
#F   Matrices are obtained with MatSPL, and infinity-norm of the
#F   difference is thresholded, with threshold of 1e-11. 
#F
#F   See also VerifyRules, it is a more general function.
#F
VerifyRulesForSPL := (S, opts) -> VerifyRules(S, _checkMatRuleTree, opts); 


#F Rule switching
#F ----------------
#F


#F AllRules(<non-terminal> | <non-terminal-name>) 
#F    Returns a list of all rules for <non-terminal>.
#F  
AllRules := nt -> let(
    name := Cond(IsSPL(nt), nt.__name__, IsString(nt), nt, 
	Error("<nt> must be a nonterminal or its name (string)")),
    When(IsBound(ApplicableTable.(name)), ApplicableTable.(name), 
	Error("No rules exist for '", name, "'")));

#F Adding your own rules
#F ---------------------
#F

#F BreakdownRule - base class for breakdown rules created with RulesFor(...)
#F
Class(BreakdownRule, rec(
    isRule := true,
    operations := RuleOps,
    info             := "-not specified-",
    forTransposition := false,
    forTranspositionOnly := false,
    switch           := true,
    transposed       := false,
    allChildren      := P -> [[ ]],
    isApplicable     := P -> true,
    __call__         := arg >> ApplyFunc(RuleTree, arg)
));

#F NewBreakdownRule - base class for breakdown rules created with NewRulesFor(...)
#F
Class(NewBreakdownRule, AttrMixin, rec(
    isRule := true,
    isNewRule := true,
    operations := RuleOps,
    info             := "-not specified-",
    forTransposition := false,
    forTranspositionOnly := false,
    switch           := true,
    transposed       := false,

    freedoms := (self, nt) >> [],
    child := (self, nt, fr) >> [],

    children := (self, nt) >> let(
	ff := List(_unwrap(self.freedoms(nt)), _unwrap),
	cart := Cartesian(ff),   # Cartesian([])==[[ ]]
	List(cart, f -> self.child(nt, f))
    ),
    
    applicable       := nt -> true,
    __call__         := arg >> ApplyFunc(RuleTree, arg),

# NOTE: YSV finish this (--parametrization of rules)
#     print := self >> let(rch := self.rChildren(),
#         Print(self.name, Cond(rch<>[], Print(".from_rChildren(", PrintCS(rch), ")"), ""))),

#     operations := RewritableObjectOps,
#     lessThan := RewritableObject.lessThan,
#     equals := RewritableObject.equals,
#     rChildren := self >> [],
#     from_rChildren := (self, rch) >> Error("not supported"),    
));
     
MakeRule := function ( R, name, nt )
    local doc;
    Constraint(IsRec(R) and IsString(name));
    doc := R.__doc__;
    # R.__doc__ is overwritten upon assignment
    R := WithBases(BreakdownRule, R);
    R.__doc__ := doc;
    R.name    := name;
    R.nonTerminal := nt;
    return R;
end;

NewMakeRule := function ( R, name, nt )
    local doc;
    Constraint(IsRec(R) and IsString(name));
    doc := R.__doc__;
    # R.__doc__ is overwritten upon assignment
    R := WithBases(NewBreakdownRule, R);
    R.__doc__ := doc;
    R.name    := name;
    R.nonTerminal := nt;
    return R;
end;

@_Base := NewMakeRule(
    rec(applicable       := nt -> not IsNonTerminal(nt) and not IsBound(ApplicableTable.(nt.__name__)),
	forTransposition := false,
        forTranspositionOnly := false,
	children             := nt -> [nt.children()],
	apply := (nt,ch,nonterms) -> Inherit(nt, rec(_children := ch))),
        "@_Base", @); 

#F ApplicableTable - mapping from non-terminal names to applicable rules
#F
ApplicableTable := rec(
    @ := [ @_Base ]
);

_RulesFor := function(nt, rulesRec, makeRuleFunc)
    local rules, r, ntname, fields, nam;
    Constraint(IsSPL(nt));
    if not IsBound(nt.index) then AddNonTerminal(nt); fi;

    if not IsRec(rulesRec)
        then Error("<rulesRec> must be a record containing rule records as elements"); fi;

    fields := Filtered(RecFields(rulesRec), f -> not IsSystemRecField(f));
    if not ForAll(fields, x -> IsRec(rulesRec.(x))) 
        then Error("<rulesRec> must be a record containing rule records as elements");
    fi;

    rules := [];
    for nam in fields do
        rulesRec.(nam) := makeRuleFunc(rulesRec.(nam), nam, nt);
	Add(rules, rulesRec.(nam));
    od;

    if IsBound(ApplicableTable.(nt.__name__)) then Append(ApplicableTable.(nt.__name__), rules);
    else ApplicableTable.(nt.__name__) := rules; fi;

    # NOTE: Assign() was here for backwards compatibility
    for r in rules do Assign(r.name, r); od;
end;

#F RulesFor( <nonterm>, <rulesRec> )
#F
RulesFor := (nt, rulesRec) -> _RulesFor(nt, rulesRec, MakeRule);

NewRulesFor := (nt, rulesRec) -> _RulesFor(nt, rulesRec, NewMakeRule);
    
SimpleRule := (pat, rule) -> rec(
    isApplicable := Subst(P -> PatternMatch(P, $pat, empty_cx())),
    rule := rule
);

BaseRule := (transform, pat) -> let(lpat := When(IsList(pat), pat, [pat]),
    rec(
	isApplicable := DetachFunc(Subst(P -> PatternMatch(P, $(Concatenation([ListClass],lpat)), empty_cx()))),
	rule := DetachFunc(Subst((P,C) -> ApplyFunc($transform, P).terminate()))
    ));

Class(InfoNt, NonTerminal, rec(
    abbrevs := [ arg -> arg ], 
    dims := self >> [1,1],
    doNotExpand := true,
    doNotMeasure := true,
    doNotSaveInHashtable := true,
    isReal := True,
    transpose := self >> self
));

NewRulesFor(InfoNt, rec(Info_Base := rec(applicable := True, apply := arg -> I(1))));

############

Class(AlternativeRewriteRule, AttrMixin, rec(
    isRule := true,
    isAlternativeRewriteRule := true,
    operations := RuleOps,
));


MakeAlternativeRewriteRule := function ( R, name)
    local doc;
    Constraint(IsRec(R) and IsString(name));
    doc := R.__doc__;
    # R.__doc__ is overwritten upon assignment
    R := WithBases(AlternativeRewriteRule, R);
    R.__doc__ := doc;
    R.name    := name;
    return R;
end;

AlternativeRewriteRules := function(rulesRec)
    local fields, nam;

    if not IsRec(rulesRec)
        then Error("<rulesRec> must be a record containing rule records as elements"); fi;

    fields := Filtered(RecFields(rulesRec), f -> not IsSystemRecField(f));
    if not ForAll(fields, x -> IsRec(rulesRec.(x))) 
        then Error("<rulesRec> must be a record containing rule records as elements");
    fi;

    for nam in fields do
        rulesRec.(nam) := MakeAlternativeRewriteRule(rulesRec.(nam), nam);
        Assign(nam, rulesRec.(nam));
    od;
end;
