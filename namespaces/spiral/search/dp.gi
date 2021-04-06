
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Dynamic Programming Search for good trees
# =========================================
# BS, MP, from 08/17/00

#F TimeLimitExpired( <startTime>, <timeLimit> )
#F    calculates if the time limit has expired
#F

TimeLimitExpired := function( startTime, timeLimit )
   return ( TimeInSecs()-startTime > 60*timeLimit );
end;

#F HashTableDP()
#F   returns a hash table for use with dynamic programming
#F

DPMeasureRuleTree := CMeasureRuleTree;

HashTableDP := function()
   return HashTable( HashIndexSPL, IsHashIdenticalSPL, "HashTableDP()" );
end;

_dpMeasure := function(wrappedTree, fullTree, spl, opts, dpopts, indent, isBaseCase)
    local mtree, time, bf;
    # Time tree
    When(dpopts.verbosity>=4, Print(Blanks(indent), "Timing: ", PrettyPrintRuleTree(fullTree, indent+6), "\n"));

#    Print("** _dpMeasure: \n* ", wrappedTree, "\n* ", fullTree, "\n* ", spl, "\n");

    ##################################################
    #    tSPL hack: do not measure everything
    if dpopts.verbosity>=5 then
    Print("Now doing measurments:\n");
    Print("spl: ", spl, "\n");
    if IsBound(dpopts.wraps) then Print("dpopts.wraps: ", dpopts.wraps, "\n"); fi;
    if IsBound(spl.wrap) then Print("spl.wrap", spl.wrap, "\n"); fi;
    fi;

    When(dpopts.verbosity>=5, Print(Blanks(indent), "tSPL: Measuring: ", wrappedTree,"\n"));

    if (IsBound(spl.doNotMeasure) and (spl.doNotMeasure)) then
        When(dpopts.verbosity>=5, Print(Blanks(indent), "tSPL: Not measuring ", spl, "\n"));
        time := 0;
    else
        if not dpopts.timeBaseCases and isBaseCase then
            time := 0; # do not measure
        elif IsBound(dpopts.measureFunction) then
            time := dpopts.measureFunction( wrappedTree, opts );
        else
            time := DPMeasureRuleTree( wrappedTree, opts );
        fi;
        When(dpopts.verbosity>=4, PrintLine("Measurment done: ", time));
    fi;

    ResetCodegen();

    When(dpopts.verbosity>=4, Print(Blanks(indent), "   ! ", time, "\n" ));
    return time;
end;

_dpReuseHash := function(W, spl, opts, dpopts, rvars)
    local lkup, lkup2, i, wrappedspl;

    wrappedspl := spl;
    for i in Reversed(W) do
        wrappedspl := i.twrap(spl,opts);
    od;

    # First lookup in readonly base cases hashtable, if not found, use the DP hashTable
    lkup := MultiHashLookup(opts.baseHashes, spl); # NOTE: why not wrappedspl? because it does not work!
    if lkup <> false then
        # Save in DP hash table also
        if HashLookup(rvars.hashTable, spl) = false then
            HashAdd(rvars.hashTable, spl, lkup);
        fi;
    else
        lkup := HashLookup( rvars.hashTable, wrappedspl);
    fi;


    # Found something -- just return it
    if not lkup = false then
    When(dpopts.verbosity>=5, Print("reuse lkup:\nlkup: ", lkup, "\n"));
    lkup2 := Copy(lkup);
    for i in lkup2 do
            if IsBound(i.origtree) then
                # special case in real A_x_I wrapped guys -- NOTE!!!
        if W <> VWrapId then
            When(dpopts.verbosity>=5, Print("VWrapId: using ruletree instead of origtree:\n"));
            i.ruletree := i.origtree;
        fi;
        Unbind(i.origtree);
        fi;
    od;
    return lkup2;
    fi;
    return lkup;
end;


_dpMax := (a,b) -> a.measured > b.measured;
_dpMin := (a,b) -> a.measured < b.measured;

Declare(_DPSPLRec);

# DPSPLRec( <spl>, <search-options-rec>, <SPL-options-rec>, <recVars> )
#    does the actual dynamic programming recursivedly.
#    do not call directly -- call DPSPL.
#
# HELP -- no checks for infinite loops (YSV: infinite loops in what?)
#
DPSPLRec := function( spl, dpopts, opts, rvars )
    local hspl, res, i;
    opts := TypeOpts(spl, opts);
    hspl := HashAsSPL(spl);
    res  := _DPSPLRec(hspl, dpopts, opts, rvars);
    for i in res do
        i.ruletree := ApplyRuleTreeSPL(i.ruletree, spl, opts);
    od;
    return res;
end;

_DPSPLRec := function( spl, dpopts, opts, rvars )
   local hash, hash2, chash, i, bf, wrappedspl, tree, trees, mtree, W, w, children, childrenSets,
         index, time, bestTrees, bestTrees2, numTimed, result, sorter, dpopts2, IsTopLevel,
         fullTree, mopts;

    ################################################
    if dpopts.verbosity>=5 then
        Print("Recursive call, spl: ", spl, "\n");
        if IsBound(dpopts.wraps) then Print("dpopts.wraps: ", dpopts.wraps, "\n"); fi;
        if IsBound(spl.wrap) then Print("spl.wrap", spl.wrap, "\n"); fi;
    fi;

    dpopts2 := Copy(dpopts);

    if not IsBound(dpopts2.wraps) then
        dpopts2.wraps := [VWrapId];
        dpopts2.wrap_index := 1;
    fi;

    # replaceable wrapper
    if ObjId(spl)=DPWrapper then 
        dpopts2.wraps := ListWithout(dpopts2.wraps, dpopts2.wrap_index) :: [spl.wrap]; 
        dpopts2.wrap_index := Length(dpopts2.wraps);
    fi;
    # stackable wrappers
    if ObjId(spl)=DPSWrapper then 
        Add(dpopts2.wraps, spl.wrap); 
    fi;

    W := dpopts2.wraps; 

    #################################################

    ############################################################
    ## Temporary Hack because spl.doNotMeasure is not checked ##
    ############################################################
    if (ObjId(spl)=InfoNt) then
        W:=[VWrapId];
    fi;

    # Check Time Limit
    if rvars.stopNow or (IsInt(dpopts.timeLimit) and TimeLimitExpired(rvars.startTime, dpopts.timeLimit)) then
        rvars.stopNow := true;
        When(dpopts.verbosity > 1, Print("\nTime Limit Expired!\n\n"));

        return [];
    fi;

    # unify data types to have spl with correct data type attributes before searching in hash 
    spl := SumsUnification(spl, opts);

    hash := _dpReuseHash(W, HashAsSPL(spl), opts, dpopts, rvars);
    if hash <> false then
        return hash;
    fi;

    # Set up some local variables and functions
    sorter := When(dpopts.optimize = "maximize", _dpMax, _dpMin);
    numTimed := 0;
    bestTrees := [];
    if dpopts.verbosity>=1 then 
        Print(Blanks(rvars.indent), "DP called on ", spl.print(rvars.indent, 2), "\n" );
    fi;
    trees := ExpandSPL(spl, opts);
    When(dpopts.verbosity>=3, Print(Blanks(rvars.indent), Length(trees), " tree(s) to fully expand\n"));
    rvars.indent := rvars.indent + 2;

    # For each possible new one level tree
    for tree in trees do

      # Check Time Limit
      if IsInt(dpopts.timeLimit) and not rvars.stopNow and TimeLimitExpired(rvars.startTime, dpopts.timeLimit) then
        rvars.stopNow := true;
        When(dpopts.verbosity > 1, Print("\nTime Limit Expired!\n\n"));
      else
        # For each possible set of subtrees of <tree>'s children
        childrenSets := Cartesian(List(tree.children, c -> List(DPSPLRec(c,dpopts2,opts,rvars), n->n.ruletree)));

        for children in childrenSets do
          if dpopts.verbosity>=5 then ################################################
            PrintLine("Back again:", "\nspl: ", spl, "\nchildren: ", children);
            When(IsBound(dpopts.wraps), Print("dpopts.wraps: ", dpopts.wraps, "\n"));
            When(IsBound(spl.wrap),    Print("spl.wrap", spl.wrap, "\n"));
          fi;   ######################################################################

          # Construct full tree, wrap it, and measure
          fullTree := CopyRuleTree(tree);
          fullTree.children := ShallowCopy(children);
          mopts := ShallowCopy(opts);
          mtree := fullTree;
          for w in Reversed(W) do
            mtree := w.wrap(mtree, spl, mopts);
            mopts := w.opts(spl, mopts);
          od;
          time := _dpMeasure(mtree, fullTree, spl, mopts, dpopts, rvars.indent,
              Length(childrenSets)=1 and Length(trees)=1);

          rvars.numTimed := rvars.numTimed + 1;
          numTimed := numTimed + 1;

              # See if one of the best times
          Add(bestTrees, rec( ruletree := mtree, origtree:= fullTree, measured := time,
                          globalUnrolling := opts.globalUnrolling ));
          Sort(bestTrees, sorter);
          When(IsBound(bestTrees[dpopts.nBest+1]), Unbind(bestTrees[dpopts.nBest+1]));

          od;
      fi;
   od;
   rvars.indent := rvars.indent - 2;

   # Save final result
   if not(IsBound(spl.doNotSaveInHashtable)) or
       (IsBound(spl.doNotSaveInHashtable) and not spl.doNotSaveInHashtable) then
       wrappedspl := HashAsSPL(spl);
       for w in Reversed(W) do
           wrappedspl := w.twrap(wrappedspl,opts);
       od;
       HashAdd(rvars.hashTable, wrappedspl, bestTrees);

       if IsBound(opts.unsafeDpHashUnwrapped) and opts.unsafeDpHashUnwrapped then
	   if HashLookup(rvars.hashTable, HashAsSPL(spl))=false then
	       HashAdd(rvars.hashTable, HashAsSPL(spl), List(bestTrees, x->CopyFields(x, rec(ruletree:=x.origtree))));
	   fi;
       fi;
   fi;

   if dpopts.verbosity>=2 then
       When(dpopts.verbosity>=3, Print(Blanks(rvars.indent), numTimed, " tree(s) timed at this level\n"));
       Print(Blanks(rvars.indent), "Best Trees:\n" );
       for result in bestTrees do
           Print(Blanks(rvars.indent+3), PrettyPrintRuleTree(result.ruletree, rvars.indent+3), " ! ", result.measured, "\n");
       od;
   fi;

    # NOTE: Hack to avoid wrapped trees in bestfoundtable
    bestTrees2 := Copy(bestTrees);
    for i in bestTrees2 do
        if IsBound(i.origtree) then
            i.ruletree := i.origtree;
            Unbind(i.origtree);
        fi;
    od;

    return bestTrees2;
end;


# GlobalUnrollingDPSPL( <spl>, <search-options-rec>,
#                       <SPL-options-rec>, <rvars> )
#   performs search over global unrolling for DP
#

GlobalUnrollingDPSPL := function( spl, dpopts, opts, rvars )
   local result, bestResult, bestUnrolling, bestHashTable;

   bestResult := false;

   # Initialize to first globalUnrolling setting
   opts.globalUnrolling := dpopts.globalUnrollingMin;

   repeat

      if dpopts.verbosity > 0 then
         Print( "Global_Unrolling := ", opts.globalUnrolling, "\n" );
      fi;

      # Set up hashTable
      if IsBound( dpopts.hashTable ) then
         rvars.hashTable := Copy(dpopts.hashTable);
      else
     rvars.hashTable := HashTableDP();
      fi;

      # Do DP
      result := DPSPLRec( spl, dpopts, opts, rvars );

      # Keep track of best result found over all unrollings
      if bestResult = false or bestResult = [] then
         bestResult := result;
     bestUnrolling := opts.globalUnrolling;
     bestHashTable := rvars.hashTable;
      elif    (     bestResult[1].measured < result[1].measured
                and dpopts.optimize = "maximize" )
           or (     bestResult[1].measured > result[1].measured
                and dpopts.optimize = "minimize" ) then
         bestResult := result;
     bestUnrolling := opts.globalUnrolling;
     bestHashTable := rvars.hashTable;
      fi;

      # Increase globalUnrolling
      opts.globalUnrolling := opts.globalUnrolling * 2;

   until opts.globalUnrolling > dpopts.globalUnrollingMax or
         rvars.stopNow;

   if dpopts.verbosity > 0 then
      Print( "Optimal unrolling is ", bestUnrolling, "\n" );
   fi;

   if IsBound( dpopts.hashTable ) then
      RecCopy(dpopts.hashTable, bestHashTable);
   fi;

   return bestResult;
end;


#F DPSPL( <spl> [, <DP-options-record>, <SPL-options-record> ] )
#F   runs dynamic programing on the given spl.
#F   maintins the n best formulas for each spl.
#F   returns a list of records of the best trees and their times.
#F   also may pass a DPOptionsRecord and a SPLOptionsRecord.
#F   to specify one set of options but not the other, pass just a "rec()"
#F      in place of the options you do not wish to set
#F   call PrintSpecDPOptionsRecord() and PrintSpecSPLOptionsRecord() to get
#F      info on possible options
#F   Verbosity levels:
#F      0 = Print nothing
#F      1 = Show recursive calls to DP
#F      2 = Show best trees found found each recursive call to DP
#F      3 = Show how many trees must be fully expanded and how many were timed
#F      4 = Show formulas that are being timed.
#F

DPSPL := function( arg )
   local spl,
     dpopts, opts,
     rvars,
     result;

   # process arg
   if Length(arg) = 1 then
      spl := arg[1];
      dpopts := MergeDPOptionsRecord(rec());
      opts := MergeSPLOptionsRecord(rec());
   elif Length(arg) = 3 then
      spl := arg[1];
      dpopts := MergeDPOptionsRecord(arg[2]);
#opts := MergeSPLOptionsRecord(arg[3]);
      opts := (arg[3]);
   else
      Error( "usage: DPSPL( <spl>",
             " [, <DP-options-record>, <SPL-options-record> ] )\n" );
   fi;

   # check spl
   if not IsSPL(spl) then
      Error("<spl> must be provided and a valid spl");
   fi;
   # check dataType
   SearchCheckDataType( spl, opts );
   # setup variables used during recursive calls

   rvars := rec( indent := 0, numTimed := 0, stopNow := false);

   # check timeLimit
   if IsInt( dpopts.timeLimit ) then
      rvars.startTime := TimeInSecs();
   fi;

   # check globalUnrolling because it requires extra work
   if dpopts.globalUnrolling = true then
      result := GlobalUnrollingDPSPL( spl, dpopts, opts, rvars );
   else

      # setup hashTable
      if IsBound( dpopts.hashTable ) then
         rvars.hashTable := dpopts.hashTable;
      elif IsBound( opts.hashTable ) then
         rvars.hashTable := opts.hashTable;
      else
         rvars.hashTable := HashTableDP();
      fi;

      result := DPSPLRec( spl, dpopts, opts, rvars );

   fi;

   if dpopts.verbosity >= 3 then
      Print( rvars.numTimed, " total trees timed\n" );
   fi;

   return result;
end;


#F DP(...) alias for DPSPL(...)
#F

DP := DPSPL;
