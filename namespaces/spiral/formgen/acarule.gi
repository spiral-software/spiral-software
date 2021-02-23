
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F PrintActiveRules(<non-terminal>)
#F PrintActiveRules(<non-terminal.name>)   
#F PrintActiveRules([<rulelist>])
#F 
#F Prints out all applicable rules for for the transform <non-terminal>
#F together with their switches.
#F Accepts a nonterminal spl, its symbol, or a rule list as an argument.
#F  

PrintActiveRules := function (nonterm)

local rules, switches, i;

# Check if rules are already found
if IsList(nonterm) and not IsString(nonterm) then 
  if ForAll(nonterm, IsRule) then rules:=nonterm;
  else Error("List must be a list of rules");
  fi; 
else rules := AllRules(nonterm);
fi;

switches := 
  List([1..Length(rules)], 
    function(i) 
      if rules[i].switch = false then return "OFF";
      else return "ON";
      fi;
    end
  );
Print("\n");
Print("No. \t\b\bSwitch \tRule\n");
Print("--------------------------------------------\n"); 
for i in [1..Length(rules)] do
  Print(i,".\t",switches[i],"\t",rules[i],"\n");
od;
Print("\n");
end;

#F SwitchRulesOn( <non-terminal>, <ind> )
#F SwitchRulesOn( <non-terminal.name>, <ind> )
#F SwitchRulesOn( [<rulelist>], <ind> )
#F 
#F Switches on the rules with indices <ind> for a given <non-terminal>
#F   <non-terminal> can be either a nonterminal spl or its symbol.
#F   <ind> is either a single index or a list of indices
#F
#F Hint: Use PrintActiveRules(<non-terminal>) to find out more about 
#F       rules and their numbering.
#F 

SwitchRulesOn := function ( nonterm , index )
local i, L;

# Check if rules are already found
if IsList(nonterm) and not IsString(nonterm) then 
  if ForAll(nonterm, IsRule) then L := nonterm;
  else Error("List must be a list of rules");
  fi; 
else L := AllRules(nonterm);
fi;

if not IsList(index) then index := [index];fi; 
if ForAny(index, k-> not (IsInt(k) and k <= Length(L) and k > 0)) then
     Error("Index to the rule list must be an integer in the proper range");
fi;
for i in index do
  L[i].switch:=true;
od;

PrintActiveRules(L);
end;

Declare(SwitchRulesQuiet, SwitchRulesByNameQuiet);

#F SwitchRules( <non-terminal>, <ind> )
#F SwitchRules( <non-terminal.name>, <ind> )
#F SwitchRules( [<rulelist>], <ind> )
#F 
#F Switches on the rules with indices <ind> for a given <non-terminal>
#F   and off all the other rules for the same nonterm
#F   <non-terminal> can be either a nonterminal spl or its symbol.
#F   <ind> is either a single index or a list of indices
#F
#F Hint: Use PrintActiveRules(<non-terminal>) to find out more about 
#F       rules and their numbering.
#F 
SwitchRules := function ( nonterm , index )
    SwitchRulesQuiet(nonterm, index);
    PrintActiveRules(nonterm);
end;

SwitchRulesQuiet := function ( nonterm , index )
local i, L;

# Check if rules are already found
if IsList(nonterm) and not IsString(nonterm) then 
  if ForAll(nonterm, IsRule) then L := nonterm;
  else Error("List must be a list of rules");
  fi; 
else L := AllRules(nonterm);
fi;

if not IsList(index) then index := [index];fi; 
if ForAny(index, k-> not (IsInt(k) and k <= Length(L) and k > 0)) then
     Error("Index to the rule list must be an integer in the proper range");
fi;

for i in [1..Length(L)] do 
  L[i].switch:=false;
od;

for i in index do
  L[i].switch:=true;
od;
end;


SwitchRulesByName := function ( nonterm , rules )
    SwitchRulesByNameQuiet(nonterm, rules);
    PrintActiveRules(nonterm);
end;

SwitchRulesByNameQuiet := function ( nonterm , rules )
    local r, L;
    if not IsNonTerminal(nonterm) then Error("<nonterm> must be a non-terminal");
    elif not (IsList(rules) and ForAll(rules, IsRule)) then Error("<rules> must be a list of rules");
    fi;
    SwitchRulesQuiet(nonterm, []);
    for r in rules do
        r.switch:=true;
    od;
end;

#F SwitchRulesOff( <non-terminal>, <ind> )
#F SwitchRulesOff( <non-terminal.name>, <ind> )
#F SwitchRulesOff( [<rulelist>], <ind> )
#F 
#F Switches off the rules with indices <ind> for a given <non-terminal>
#F   <non-terminal> can be either a nonterminal spl or its symbol.
#F   <ind> is either a single index or a list of indices
#F
#F Hint: Use PrintActiveRules(<non-terminal>) to find out more about 
#F       rules and their numbering.
#F 

SwitchRulesOff := function ( nonterm , index )
local i, L;

# Check if rules are already found
if IsList(nonterm) and not IsString(nonterm) then
  if ForAll(nonterm, IsRule) then L := nonterm;
  else Error("List must be a list of rules");
  fi; 
else L := AllRules(nonterm);
fi;

if not IsList(index) then index := [index]; fi; 
if ForAny(index, k-> not (IsInt(k) and k <= Length(L) and k > 0)) then
     Error("Index to the rule list must be an integer in the proper range");
fi;
for i in index do
  L[i].switch:=false;
od;

PrintActiveRules(L);
end;

# VIENNA ADDED:
#F SwitchRulesName( [<rulelist>], true|false )
#F 
#F Switches the rules on and off by names 
#F   <rulelist> is a list of names of rules
#F
SwitchRulesName := function(l,s)
local i;
for i in l do i.switch:=s; od;
end;
