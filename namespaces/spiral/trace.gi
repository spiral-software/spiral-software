
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Class(TraceBase, rec());

Class(TraceRewrite, TraceBase, rec(
        __call__ := (self, name,input,output,env) >> WithBases(self, rec(name:=name, input:=input,output:=output,env:=env,operations := PrintOps)),
       print := self >> Print("Rewrite(", self.name,",",self.input,",",self.output,")")
));

Class(TraceExpansion, TraceBase, rec(
        __call__ := (self, name,input,output,env) >> WithBases(self, rec(name:=name, input:=input,output:=output,env:=env,operations := PrintOps)),
       print := self >> Print("Expansion(", self.name,",",self.input,",",self.output,")")
));

Class(TraceTreeExpansion, TraceBase, rec(
        __call__ := (self, name,input,output,children,subtrees,env) >> WithBases(self, rec(name:=name, input:=input,output:=output,children:=children,subtrees:=subtrees, env:=env,operations := PrintOps)),
       print := self >> Print("TreeExpansion(", self.name,",",self.input,",",self.output,",",self.children,")")
));

Class(TraceConversion, TraceBase, rec(
        __call__ := (self, name,input,output,env) >> WithBases(self, rec(name:=name, input:=input,output:=output, env:=env, operations := PrintOps)),
       print := self >> Print("Conversion(", self.name,",",self.input,",",self.output,")")
));

Class(TraceLogPrinter, rec(
        __call__ := (self) >> WithBases(self, rec()),
                           
        addRewrite := (self, name,input,output,env) >> 
                           Print("\n------------------------------------------------------------------\n", 
                                 "RULE: ", name,
                                 "\n------------------------------------------------------------------\n", 
                                 "old expression",
                                 "\n------------------------------------------------------------------\n", 
                                 input,
                                 "\n------------------------------------------------------------------\n",
                                 "new expression",
                                 "\n------------------------------------------------------------------\n", 
                                 output,
                                 "\n------------------------------------------------------------------\n"),
                           
        addExpansion := (self, name,input,output,env) >>
                           Print("\n==================================================================\n",
                                 "EXPANSION RULE: ", name, 
                                 "\n------------------------------------------------------------------\n", 
                                 "SPL expression:",
                                 "\n------------------------------------------------------------------\n", 
                                 input,
                                 "\n------------------------------------------------------------------\n", 
                                 "Sigma-SPL expression:",
                                 "\n------------------------------------------------------------------\n", 
                                 output,
                                 "\n\n"),
                           
        addTreeExpansion := meth(self, name,input,output,children,subtrees,env)
                           local d;
                           Print("\n==================================================================\n",
                                 "TREE EXPANSION RULE: ", name, 
                                 "\n------------------------------------------------------------------\n", 
                                 "original expression:",
                                 "\n------------------------------------------------------------------\n", 
                                 input, "\n");

                           if Length(children) <> 0 then
                               Print("\n------------------------------------------------------------------\n", 
                                     "subtree substitutions: ", 
                                     "\n------------------------------------------------------------------\n");
                               DoForAll(children, i->Print(i, "\n\n"));
                               Print("\n------------------------------------------------------------------\n", 
                                     "subtrees substituted: ", 
                                     "\n------------------------------------------------------------------\n");
                               Print(subtrees, "\n");      
                           fi;
                           # NOTE: pass and save var
                           # d := Collect(output, @(1, var, e->IsBound(e.value)));
                           #if (d <>[]) then
                           #   Print("\n------------------------------------------------------------------\n", 
                           #          "data tables: ", 
                            #         "\n------------------------------------------------------------------\n");
                            #   DoForAll(d, i->Print(i.id, " := ", i.value, "\n\n"));
                           #fi;

                           Print("------------------------------------------------------------------\n", 
                                 "substituted expression:",
                                 "\n------------------------------------------------------------------\n", 
                                 output, "\n\n");
                       end,
                           
        addConversion := (self, name,input,output,env) >> 
                           Print("\n==================================================================\n",
                                 "CONVERSION RULE: ", name, 
                                 "\n------------------------------------------------------------------\n", 
                                 "icode:",
                                 "\n------------------------------------------------------------------\n", 
                                 input,
                                 "\n------------------------------------------------------------------\n", 
                                 "icode:",
                                 "\n------------------------------------------------------------------\n", 
                                 output,
                                 "\n\n"),
                           
        beginRuleset := (self, ruleset, input) >> 
                           Print("\n******************************************************************\n",
                                 "BEGIN RULESET: ", ruleset.inType, " -> ", ruleset.outType,
                                 "\n------------------------------------------------------------------\n", 
                                 "Ruleset:", ruleset,
                                 "\n------------------------------------------------------------------\n", 
                                 "Initial ", ruleset.inType,
                                 "\n------------------------------------------------------------------\n", 
                                 input,
                                 "\n******************************************************************\n"),

        endRuleset := (self, ruleset,output) >> 
                           Print("\n******************************************************************\n",
                                 "END RULESET: ", ruleset.inType, " -> ", ruleset.outType,
                                 "\n------------------------------------------------------------------\n", 
                                 "Final ", ruleset.outType,
                                 "\n------------------------------------------------------------------\n", 
                                 output,
                                 "\n******************************************************************\n\n"),
                           
        beginStage := (self, from, to, input) >> 
                           Print("******************************************************************\n",
                                 "BEGIN TRACE: ",from," -> ", to,
                                 "\n------------------------------------------------------------------\n", 
                                 from, " expression:",
                                 "\n------------------------------------------------------------------\n", 
                                 input,
                                 "\n******************************************************************\n"),
                           
        endStage := (self, from, to, output) >>
                           Print("\n******************************************************************\n",
                                 "END TRACE: ",from," -> ", to,
                                 "\n------------------------------------------------------------------\n", 
                                 to," expression:",
                                 "\n------------------------------------------------------------------\n", 
                                 output,
                                 "\n******************************************************************\n\n"),

		addNote := (self, note) >>
                           Print("\n------------------------------------------------------------------\n",
                                 "NOTE:\n", note, "\n",
                                  "------------------------------------------------------------------\n")
								 
                         #NOTE:
                         #d := Collect(spl, @(1, var, e->IsBound(e.value)));
                         #if (d <>[]) then
                         #    Print("\n------------------------------------------------------------------\n", 
                         #          "data tables: ", 
                         #          "\n------------------------------------------------------------------\n");
                         #    DoForAll(d, i->Print(i.id, " := ", i.value, "\n\n"));
                        #fi;
));

Class(TraceLogToFile, TraceLogPrinter, rec(
        __call__ := (self, filename) >> WithBases(self, rec(filename:=filename)),
                            
        addRewrite    := (self, name,input,output,env) >> AppendTo(self.filename, self.__bases__[1].__bases__[1].addRewrite           (name,input,output,env)),
        addExpansion  := (self, name,input,output,env) >> AppendTo(self.filename, self.__bases__[1].__bases__[1].addExpansion   (name,input,output,env)),
        addTreeExpansion  := (self, name,input,output,children,subtrees,env) >> AppendTo(self.filename, self.__bases__[1].__bases__[1].addTreeExpansion(name,input,output,children,subtrees,env)),
        addConversion := (self, name,input,output,env) >> AppendTo(self.filename, self.__bases__[1].__bases__[1].addConversion (name,input,output,env)),
                            
        beginRuleset  := (self, ruleset, input   ) >> AppendTo(self.filename, self.__bases__[1].__bases__[1].beginRuleset  (ruleset, input   )),
        endRuleset    := (self, ruleset, output  ) >> AppendTo(self.filename, self.__bases__[1].__bases__[1].endRuleset    (ruleset, output  )),
        beginStage    := (self, from, to, input  ) >> AppendTo(self.filename, self.__bases__[1].__bases__[1].beginStage    (from, to, input  )),
        endStage      := (self, from, to, output ) >> AppendTo(self.filename, self.__bases__[1].__bases__[1].endStage      (from, to, output )),
		addNote       := (self, note             ) >> AppendTo(self.filename, self.__bases__[1].__bases__[1].addNote       (note ))
));


Class(TraceLogCollector, rec(
        __call__ := (self) >> WithBases(self, rec(log:=[])),
                       
        addRewrite := meth(self, name,input,output,env)
                    local e;
                    e := TraceRewrite(name, input,output,env);
                    Add(self.log, e);
                    return e;
                end,
                  
        addExpansion := meth(self, name,input,output,env)
                    local e;
                    e := TraceExpansion(name, input,output, env);
                    Add(self.log, e);
                    return e;
                end,
                  
        addTreeExpansion := meth(self, name,input,output,children,subtrees,env)
                    local e;
                    e := TraceTreeExpansion(name, input,output,children,subtrees,env);
                    Add(self.log, e);
                    return e;
                end,
                  
        addConversion := meth(self, name,input,output,env)
                    local e;
                    e := TraceConversion(name, input,output, env);
                    Add(self.log, e);
                    return e;
                end,
                  
        beginRuleset := (self, ruleset, input) >> 0,
        endRuleset := (self, ruleset, output) >> 0,
        beginStage := (self, from, to, input) >> 0,
        endStage := (self, from, to, output) >> 0,
		addNote := (self, note) >> 0
                  
));

Class(TraceLog, rec(
                        
        __call__ := (self) >> WithBases(self, rec(plugins := [])),
                    
        addPlugin := (self, plugin) >>  Add(self.plugins, plugin),
                    
        addRewrite := meth(self, name,input,output,env)
                    local p;
                    for p in self.plugins do
                          p.addRewrite(name,input,output,env);
                     od;
                end,
                        
        addExpansion := meth(self, name,input,output,env)
                    local p;
                    for p in self.plugins do
                        p.addExpansion(name,input,output,env);
                    od;
                end,
                  
        addTreeExpansion := meth(self, name,input,output,children,subtrees,env)
                    local p;
                    for p in self.plugins do
                        p.addTreeExpansion(name,input,output,children,subtrees,env);
                    od;
                end,
                  
        addConversion := meth(self, name,input,output,env)
                    local p;
                    for p in self.plugins do
                        p.addConversion(name,input,output,env);
                    od;
                end,
                  
        beginRuleset := meth(self, ruleset, input)
                    local p;
                    for p in self.plugins do
                        p.beginRuleset(ruleset, input);
                    od;
                end,
                  
        endRuleset := meth(self, ruleset,output)
                    local p;
                    for p in self.plugins do
                        p.endRuleset(ruleset,output);
                    od;
                end,
                  
        beginStage := meth(self, from, to, input)
                    local p;
                    for p in self.plugins do
                        p.beginStage(from, to, input);
                    od;
                end,
                  
        endStage := meth(self, from, to, output)
                    local p;
                    for p in self.plugins do
                        p.endStage(from, to, output);
                    od;
                end,
				
		addNote := meth(self, note)
                    local p;
                    for p in self.plugins do
                        p.addNote(note);
                    od;
                end		
));
                
# Global variable. Will be moved to 'opts' eventually.
trace_log := TraceLog();

# Convenience functions

TraceToConsole := function() 
    trace_log.addPlugin(TraceLogPrinter()); 
end;

TraceToFile := function(filename) 
    trace_log.addPlugin(TraceLogToFile(filename));
end;

trace_collector := TraceLogCollector();
TraceToMemory := function() 
    trace_log.addPlugin(trace_collector);
    return(trace_collector);
end;

TraceNote := function(note)
	trace_log.addNote(note);
end;

  
# --- Example of trace system initialization:---
#trace_log.addPlugin(TraceLogPrinter());
#trace_log.addPlugin(TraceLogToFile("trace.g"));
# Save reference to collector for later use
#trace_collector := TraceLogCollector();
#trace_log.addPlugin(trace_collector);
