
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

# Formula Generator 
# -----------------
# This package defines data structures for decomposition rules, 
# ruletrees, SPL and Sigma-SPL formulas, and required functionality to generate
# formulas from transform and decomposition rule definitions.
#@P

Import(rewrite, code, approx, spl);

Include(optrec);
Include(nonterm);

Declare(ExpandSPL, ApplicableTable, RuleTree, SPLRuleTree);
Include(rule);
Include(acarule);
Include(ruletree);
Include(implement);
Include(external);
Include(bug);


