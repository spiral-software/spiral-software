
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Top-Level Functions for SPIRAL
# ==============================
# BWS, MP, from 02/15/00


#F Creating Transforms
#F -------------------
#F

#F Transforms
#F   the list of symbols for valid transforms, e.g., "DFT".
#F
Transforms := NonTerminalListSPL;

#F Transform ( <symbol>, <parameters> )
#F   alias for SPLNonTerminal, returns the transform defined by 
#F   <symbol> and <parameters>, e.g. Transform( "DFT", 8 ).
#F
#Transform := SPLNonTerminal;

#F Info ( <symbol> )
#F   prints information on how to create the transform <symbol>
#F
Info := Doc;

#F toSPL( <obj> )
#F   Converts ruletree or nonterminal (using random ruletree) into an SPL,
#F   If <obj> is an SPL it is returned as is.
toSPL := x -> 
    Cond(IsNonTerminal(x), SPLRuleTree(RandomRuleTree(x)),
	 IsSPL(x), x,
	 IsRuleTree(x), SPLRuleTree(x),
	 Error("<x> must be an SPL or a RuleTree"));
