
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_IsCompute  := (asgn) >> (IsBound(asgn.isCompute) and asgn.isCompute);
_IsLoad  := (asgn) >> (IsBound(asgn.isLoad) and asgn.isLoad);
_IsStore  := (asgn) >> (IsBound(asgn.isStore) and asgn.isStore);

#NOTE: Need a much better marking heuristic
MarkSWPLoops := function(loop)
  # Obviously, we can't sw-pipeline if we don't have adequate iterations
  if Length(loop.range) >=4 then
    # Current heuristic: Mark loops where the loop body has at least 10 commands.
    if IsBound(loop.cmd) and IsBound(loop.cmd.cmd) and IsBound(loop.cmd.cmd.cmds) and Length(loop.cmd.cmd.cmds) >= 10 then
      return(loop_sw(loop.rChildren()[1], loop.rChildren()[2], loop.rChildren()[3]));
    else
      return(loop);
    fi;
  else
    return(loop);
  fi;
end;

MarkLoadinPreds := function(asgn)
  local i;
   if not _IsCompute(asgn) then # Mark as load
      asgn.isLoad := true;
   fi;

   # Mark all predecessors as loads
   if IsBound(asgn.loc) and IsBound(asgn.loc.pred) then
     for i in [1..Length(asgn.loc.pred)] do
        if IsBound(asgn.loc.pred[i].def) then
           MarkLoadinPreds(asgn.loc.pred[i].def);
        fi;
     od;
   fi;
end;

MarkStoreinPreds := function(asgn)
  local i;
   if not _IsCompute(asgn) then # Mark as store
      asgn.isStore := true;

     # Mark all predecessors as stores (stops at computes)
     for i in [1..Length(asgn.loc.pred)] do
        if IsBound(asgn.loc.pred[i].def) then
           MarkStoreinPreds(asgn.loc.pred[i].def);
        fi;
     od;
   fi;
end;

MarkAllSucceedingLoadsAsComputes := function(asgn)
  local i;
  # Mark all successors that are loads as computes
  #
  for i in [1..Length(asgn.loc.succ)] do
     if IsBound(asgn.loc.succ[i].def) and _IsLoad(asgn.loc.succ[i].def) then
       #Error("Aha!");
       asgn.loc.succ[i].def.isLoad := false;
       asgn.loc.succ[i].def.isCompute := true;
     fi;
  od;

end;

SubstLoopVarCopy := function(asgns, loopvar, value)
   local retval, i;
   retval := Copy(asgns);
   for i in retval do
      SubstVars(i, rec((loopvar.id) := value));
   od;
   return(retval);
end;



SoftwarePipeline := function(lsw)
local alllsws, allasgns, loads, stores, loopvar, n1, n2, computes, isCompute, arg, asgn, asgns, g0, g1, c0, sn2, sn1, cn1, bodyload, vars, computeCount; 
# Collect all loops to be software pipelined

   # Mark all computes
   computes := Collect(lsw, [assign, ..., @(2, [add, mul, sub]), ...]);

   #Error("BP");
   
   # For each compute: kick out all those that don't have a TReal type for all exp.arg
   # NOTE: Include TVects in TReal. NOTE: Won't work for code where loopvar datatype = compute datatype
   computeCount := 0;
   for asgn in computes do
      isCompute := true;
      for arg in asgn.exp.args do
       if arg.t <> TReal and arg.t.__name__ <> "TVect" then
          isCompute := false;
       fi;
      od;

      if (isCompute) then
         asgn.isCompute := true;
         computeCount := computeCount + 1;
         #Print(".");
      fi;
   od;

   # Heuristic: if there're no computes, this is probably a huge permute (load or
   # store) block, and should not be sw-pipelined.

   if computeCount = 0 then
     return(loop(lsw.rChildren()[1], lsw.rChildren()[2], lsw.rChildren()[3]));
   fi;


   #Error("BP");
   
   # Mark all Loads
   # Loads are all predcessors of all the computes
   asgns := Collect(lsw, assign);
   for asgn in asgns do
      if IsBound(asgn.isCompute) and asgn.isCompute then
         #Print(asgn);
         MarkLoadinPreds(asgn);
      fi;
   od;

   #Error("BP");
   
   # HACK: must really check to ensure this is a store!
   asgns := Collect(lsw, assign);
   for asgn in asgns do
      if not _IsCompute(asgn) and not _IsLoad(asgn) then
         asgn.isStore := true;
         MarkStoreinPreds(asgn);
         #NOTE: the loopvar assign should also be a store
      fi;
   od;

   # Mark loads which are to be marked as computes because they are
   # both preceded by and succeeded by a compute

   # For each compute, find all successors that are loads, and mark them as
   # computes.

   asgns := Collect(lsw, assign);
   for asgn in asgns do
      if IsBound(asgn.isCompute) and asgn.isCompute then
         MarkAllSucceedingLoadsAsComputes(asgn);
      fi;
   od;

   
   # Now, change the loop_sw to a software pipelined loop
   # Cmds should be strictly in order.

   # Make a list of loads, computes, and stores
   loads    := [];
   computes := [];
   stores   := [];
   allasgns := Collect(lsw, chain)[1].cmds;
   for asgn in allasgns do
      if _IsLoad(asgn) then
         loads := Concatenation(loads, [asgn]);
      fi;
      if _IsCompute(asgn) then
         computes := Concatenation(computes, [asgn]);
      fi;
      if _IsStore(asgn) then
         stores := Concatenation(stores, [asgn]);
      fi;
   od;

   #Error("BP");


   loopvar := lsw.var;
   n1 := Length(lsw.range)-1;
   n2 := n1-1;

   g0  := SubstLoopVarCopy(loads, loopvar, V(0));
   g1  := SubstLoopVarCopy(loads, loopvar, V(1));
   c0  := Copy(computes);

   sn2 := SubstLoopVarCopy(stores, loopvar, V(n2));
   sn1 := SubstLoopVarCopy(stores, loopvar, V(n1));
   cn1 := Copy(computes);

   bodyload  := SubstLoopVarCopy(loads,  loopvar, add(loopvar, V(2)));
   #prologue := chain(g0,   c0,  g1); #epilogue := chain(sn2, cn1, sn1);

   vars := Collect(lsw, decl)[1].vars;
   lsw := decl(vars, chain(g0, c0, g1, loop(loopvar, n2, chain(stores, c0, bodyload)), sn2, cn1, sn1));
   #Print(lsw);
   return(lsw);
end;


