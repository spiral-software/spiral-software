
# Copyright (c) 2018-2020, Carnegie Mellon University
# See LICENSE for details


Import(rewrite);

VectorStrategySum := [
    MergedRuleSet(JoinDirectSums,StretchRaderMid),
    MergedRuleSet(StandardSumsRules, RulesPropagate),
    # when to terminate Grp() has become an issue: too early -> diags go the wrong way; too late->Rader stretch breaks
	RulesSMP,
    MergedRuleSet(StandardSumsRules, RulesPropagate, StretchRaderMid, JoinDirectSums, RulesTermGrp),
    MergedRuleSet(StandardSumsRules, RulesPropagate, StretchRaderMid, JoinDirectSums, RulesTermGrp, RulesVDiag),
    MergedRuleSet(StandardSumsRules, RulesPropagate, StretchRaderMid, JoinDirectSums, RulesTermGrp, RulesSMP, RulesVec),
    RulesKickout
];

VectorStrategyTerm := [
#    MergedRuleSet(RulesConjTerm, RulesFuncSimp),
    MergedRuleSet(StretchRaderMid, JoinDirectSums),
    RulesTerm, RulesTermGrp, RulesKickout,
];

VectorStrategyTerm2 := [
#    RulesConjTerm,
    TerminateDirectSums, TerminateRaderMid, TerminateDirectSums, TerminateSymSPL, TerminateDPrm,
];


VectorStrategyVDiag := [
    RulesVDiag, RulesVec, RulesSMP, RulesPropagate,
    StandardSumsRules
#   MergedRuleSet(RulesVDiag, StandardSumsRules)
];


VectorStrategyRC := [
    MergedRuleSet(RulesSMP, RulesSplitComplex),
    RulesKickout,
    RulesVBlkInt,
    RulesVDiag,
    MergedRuleSet(RulesPropagate, RulesSMP, RulesSplitComplex, RulesVRC, RulesSums),
    MergedRuleSet(RulesPropagate, RulesVRC, RulesCxRC, RulesVRCTerm, RulesSMP, RulesVec, TerminateSymSPL, RulesRC, RulesSplitComplex),
    RulesTerm,
    RulesTermGrp,
    RulesSMP,
    RulesVec,
    RulesPropagate,
    StandardSumsRules,
    RulesKickout
];

VectorStrategyRCUnterm := [
    MergedRuleSet(RulesVRC, RulesCxRC, RulesRC),
    MergedRuleSet(RulesSMP, RulesVec, RulesCxRC, RulesPropagate)
];

# VectorStrategy := [
#     MergedRuleSet(StandardSumsRules)
#     MergedRuleSet(RulesVec, RulesPropagate, RulesDiag),
#     RulesVDiag,
#     MergedRuleSet(RulesVec, RulesPropagate),
#     MergedRuleSet(StandardSumsRules)
#     MergedRuleSet(RulesVRC, RulesCxRC, RulesRC), RulesVec,
#     RulesKickout,
#     MergedRuleSet(StandardSumsRules)
# ];

VectorStrategyRCRec := [
    MergedRuleSet(RulesVRC, RulesCxRC, RulesRC),     RulesKickout,
    MergedRuleSet(RulesSMP, RulesVec, RulesPropagate),
    StandardSumsRules
];

fix_fAdd := [
    MergedRuleSet(StandardSumsRules, Rules_fAdd),
];
