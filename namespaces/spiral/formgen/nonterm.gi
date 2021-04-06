
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Non-Terminals (Transforms)
# ==========================
# MP, from 03/24/01, GAPv3.4.4

#F Non-Terminaltable (Transform Table)
#F -----------------------------------
#F

#F The non-terminal table represents all symbols that can be used
#F to define an spl of type "nonTerminal". The table is meant to be
#F extended. In contrast to spls of type "symbol", an spl of type
#F "nonTerminal" has no equivalent in the SPL language. The non-terminals
#F usually represent discrete signal transforms and are used by the
#F formula generator to represent non-terminal expressions in SPL objects.
#F These are meant to be further expanded. Thus, an SPL object can be exported
#F to a valid SPL program iff it contains no non-terminals.
#F Every non-terminal is represented by a record contained in the 
#F following list NonTerminalTableSPL.
#F The record has the form:
#F rec(
#F   isNonTerminal,   true, identifying a non-terminal
#F   NonTerminalOps,  operations record
#F   symbol,          a string for the symbol
#F   CheckParams,     a function for checking and canonifying
#F                    of the parameters
#F   Dim,             a function to return the dimension from
#F                    the parameters in canonical form
#F   Params,          a function that returns a shorter representation
#F                    of parameters used for printing in gap
#F   Terminate SPL,   a function to convert into an spl without non-terminals
#F   Transposed,      a function for transposing
#F   isRealTransform, = true if the transform is considered as real transform,
#F                    i.e., the matrix is real for all choices of parameters
#F )
#F
#F the following fields are optional:
#F for verification:
#F SmallRandom        a function that returns a parameter choice for a small
#F                    instantiation of the transform, i.e., one that can be verified
#F                    against the definition
#F LargeRandom        correspondingly, a function that returns a parameter choice
#F                    for a large instantiation of the transform, used for verifying
#F                    on a random vector
#F
#F HashIndexSPL       a function that returns index used for hashing. It is stored
#F                    in .hashIndex field. Overrides the standard
#F                    HashIndex function used in the search.
#F
#F hashIndex          overrides HashIndex function used in 
#F                    search. Precomputed by optional HashIndexSPL
#F                    function and should not be specified manually.
#F
#F
#F Nonterminals are created in the directory formgen/transforms
#F
#F The Non-Terminal table can easily be extended by adding new non-terminals
#F (transforms).
#F

#F Non-Terminal Table
#F ------------------
#F

#F NonTerminalTableSPL
#F   is the set of all known non-terminals (transforms)
#F
NonTerminalTableSPL := Set ( [ ] );

#F NonTerminalListSPL
#F   is a set containing all known non-terminals.
#F   Note that NonTerminalListSPl is ordered exactly as NonTerminalTableSPL.
#F
NonTerminalListSPL := Set( [ ] );


#F Adding new Non-Terminals (Transforms)
#F -------------------------------------
#F

#F AddNonTerminal ( <non-terminal> )
#F   adds <non-terminal> to the non-terminal table. This function is 
#F   meant to import new non-terminals into NonTerminalTableSPL,
#F   and NonTerminalListSPL, and update .index field.
#F
#F Note: Currently, this function is invoked automatically, when one
#F       for example, defines new breakdown rules for a non-terminal,
#F       
AddNonTerminal := function ( S )
  local i;
  Constraint(IsNonTerminal(S));
  if not IsBound(S.__bases__) then 
      Error("Old-style non-terminals are not supported anymore");
  fi;

  if S in NonTerminalTableSPL then RemoveSet(NonTerminalTableSPL, S); fi;
  AddSet(NonTerminalTableSPL, S);
  AddSet(NonTerminalListSPL, S.name);
  for i in [1..Length(NonTerminalTableSPL)] do 
      NonTerminalTableSPL[i].index := i;
  od;
  #UpdateApplicableTable();
end;

