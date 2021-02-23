
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


## The hoister is a compiler pass that separates looped code in three 
## parts:
##  - the hoisted code a.k.a before the loop
##  - the body of the loop, a.k.a the loop itself
##  - the epilogue of the loop, a.k.a what is run after the loop
##
## Essentially, it is done by separating statements that depend on the loop
## index, that have to be in the body from statements that do not which can 
## be hoisted.
##
## On top of this, the following code performs two optimizations:
##  - if code is being stored to a memory location that is loop independent, 
##    then this memory location is scalarized, i.e. the store is performed in 
##    register and the memory write is delayed (put inside the epilogue)
##  - The code performs loop variable induction. This replaces statements that
##    depend of param*i, by accumulators that add plus param at each pass of
##    the loop. The code also attempts to discover the minimal accumulation
##    value to have a few number of window pointers

Hoister := function(c)
    local loop_idx, loop_range, idxpool, hoist, body, epilogue, lvlist, lvfifo, asns, asn, lv, freevars, 
    candidate_memory_acc, confirmed_memory_acc, stmt, stmts, lv, freevars, equal_second_member, d, step, 
    lvfamilies, fams, fam, s, cc, pointer, j, var_usage, members, member_usage, rootexpr, m, newloop_idx, newloop_range, newloop_start, newloop_inc ;
    #Hoister only applies to loopn(decl(chain(...)))
    if ((ObjId(c)=loopn) and (ObjId(c.cmd)=decl) and (ObjId(c.cmd.cmd)=chain)) then

         c := Compile.pullDataDeclsRefs(c);
         loop_idx := c.var;
         loop_range := c.range;
         c := c.cmd;
         MarkDefUse(c);
         #idxpool will contain the set of all variables that change with the loop index
         #(and therefore should not be hoisted)
         idxpool := Set([loop_idx]);

         #Loop variable induction
         #This gathers all expressions that are linear in the index variable
         #and stores them as (v, start, inc) in the loop-variable list (lvlist)
         lvlist := [];
         lvfifo := [];

         Add(lvfifo, rec(v := loop_idx,
                     start := loop_idx.t.zero(),
                     inc := V(1)));

         while(Length(lvfifo)>0) do

             lv := lvfifo[1];
             lvfifo := ListWithout(lvfifo, 1);
             Add(lvlist, lv);

             #If lv.start is a variable, add this variable to the lvfifo
             if IsVar(lv.start) then
                  #If lv.start is used twice, it cannot be used as an induction variable.
                  if (Length(Collect(c, [assign, lv.start, @(0)]))=1
                      and Length(Collect(c, lv.start))=1) then 
                      SubstBottomUp(c, lv.v, e->lv.start);
                      SubstBottomUp(lvfifo, lv.v, e->lv.start);
                      SubstBottomUp(lvlist, lv.v, e->lv.start);

                      [asns, c] := Pull(c, [assign, lv.start, @(0)],e->skip(), e->e);

                      Add(lvfifo, rec(v := lv.start, 
                              start := asns[1].exp, 
                              inc := lv.inc));
                      AddSet(idxpool, lv.start);                  

                 fi;
             fi;

             #Gather all mults and adds that depend on lv and add them too
             [asns, c] := Pull(c, [assign, @(0), [add, @(1).cond(e->ObjId(e)<>Value), lv.v]], 
                 e->skip(), e->e);
             for asn in asns do
                 Add(lvfifo, rec(v := asn.loc, 
                         start := add(asn.exp.args[1], lv.start), 
                         inc := lv.inc));
                 AddSet(idxpool, asn.loc);
                 od;

             [asns, c] := Pull(c, [assign, @(0), [add, lv.v, @(2).cond(e->ObjId(e)<>Value)]], 
                 e->skip(), e->e);
             for asn in asns do
                 Add(lvfifo, rec(v := asn.loc, 
                         start := add(asn.exp.args[2], lv.start), 
                         inc := lv.inc));
                 AddSet(idxpool, asn.loc);
                 od;                

             [asns, c] := Pull(c, [assign, @(0), [mul, @(1), lv.v]], 
                 e->skip(), e->e);
             for asn in asns do
                 Add(lvfifo, rec(v := asn.loc, 
                     start := mul(asn.exp.args[1], lv.start), 
                     inc := mul(asn.exp.args[1],  lv.inc)));
                 AddSet(idxpool, asn.loc);
                 od;

             [asns, c] := Pull(c, [assign, @(0), [mul, lv.v, @(2)]], 
                 e->skip(), e->e);
             for asn in asns do
                 Add(lvfifo, rec(v := asn.loc, 
                     start := mul(asn.exp.args[2], lv.start), 
                     inc := mul(asn.exp.args[2],  lv.inc)));
                 AddSet(idxpool, asn.loc);
                 od;

         od;

         #Some of the lvs were only used by other lvs, let's kick
         #them out
         freevars := Set(c.free());
         lvlist := Filtered(lvlist, lv -> lv.v in freevars);

         #Now we partition the lvs into bins that have the same increment
         #We call it an lvfamily
         lvfamilies:=[];
         for lv in lvlist do
             fams := Filtered(lvfamilies, fam->fam.inc=lv.inc);
             if (Length(fams)>0) then
                 Add(fams[1].list, lv);
             else
                 Add(lvfamilies, rec(inc:=lv.inc, list:=[lv]));
             fi;
         od;

         #If the increment is an operation, hoist this operation
         for lv in lvfamilies do
             if (ObjId(lv.inc) in [add, mul]) then
                d := var.fresh_t("d", lv.inc.t);
                Add(c.cmds, assign(d, lv.inc));
                lv.inc := d;
             fi;
         od;

         #OK, so this is the tricky part.
         #we want to fuse lvfamilies together if it is possible.
         #it is only possible if members of the family are used sequentially in the 
         #loop AND if we know the pattern

         var_usage := [];
         DoForAll(c.rChildren(), function(x) if IsAssign(x) then Append(var_usage, x.op_in()); fi; end);

         for j in [1..Length(lvfamilies)] do
           fam:=lvfamilies[j];
           members:=List(fam.list, x->x.v);
           member_usage := RemoveAdjacentDuplicates(Filtered(var_usage, x -> x in members));
	   
           if (Length(Set(member_usage))=Length(member_usage) and Length(members)>1) then
               #ok so this is a good candidate, but can we match it?
               #our target is that they all are add( add(param, i*param), param)
               rootexpr := fam.list[1].start;

               if (ObjId(rootexpr)=add) then 
                   d := rootexpr.args[2];
                   equal_second_member:=true;
                   for m in fam.list do
                   if ((ObjId(m.start)<>add) or (m.start.args[2]<>d)) then
                       equal_second_member:=false;
                   fi;
                   od;
                   
                   if ((equal_second_member) and (rootexpr.args[1] in fam.list[2].start.args[1].pred)) then
                           step := Difference(fam.list[2].start.args[1].pred, Set([rootexpr.args[1]]));
                           if (Length(step)=1) then 
                               pointer := 2;
                               cc := [];
                               for d in c.rChildren() do
                               if ((Length(fam.list)>=pointer) and (members[pointer] in d.op_in())) then
                                   pointer:=pointer+1;
                              Add(cc, assign(members[1], add(members[1], step[1])));
                               fi;
                               Add(cc, d);                               
                               od;
                               c := chain(cc);
                               
                               d := sub(fam.inc, mul(V(Length(members)-1), step[1]));
                               s := var.fresh_t("inc", d.t);
                               Add(cc, assign(s, d));
                               
                               c := chain(cc);
                               lvfamilies[j] := rec(inc := s,
                                   list:=[rec(v:=members[1], 
                                           start:=rootexpr, 
                                           inc:=lvfamilies[j].inc)]);
                               for d in [2..Length(members)] do
                               SubstBottomUp(c, members[d], e->members[1]);
                               od;
                           fi;
                   fi;
               fi;
           fi;
        od;


         #This is the hoister logic
         candidate_memory_acc := Set([]);
         confirmed_memory_acc := Set([]);
         hoist := [];
         body := [];
         epilogue := [];


         for asn in c.cmds do
             if IsAssign(asn) then
                 s := asn.op_in();
                 IntersectSet(s, idxpool);
                 if (Length(s)=0) then                     
                     if (ObjId(asn.exp)=deref) then
                         #cannot assume SSA on derefs
                         #so we mark it and leave it
                         #we'll fix that later
                         AddSet(candidate_memory_acc, asn.exp);
                     fi;
                     Add(hoist, asn);
                 else
                     if ((ObjId(asn.loc)=deref) and (asn.loc in candidate_memory_acc)) then
                         AddSet(confirmed_memory_acc, asn.loc);
                     fi;
                     Add(body, asn);
                     UniteSet(idxpool, asn.op_out());
                 fi;
             else
                 if (ObjId(asn)<> skip) then
                     Error("Non assigns cannot be handled");
                 fi;
             fi;
         od;

         #Plug back the induced vars
         newloop_idx := [];
         for lv in lvfamilies do
            if ((newloop_idx=[])and(Length(lv.list)=1)) then
                newloop_idx := lv.list[1].v;
                d := lv.list[1].start;
                s := var.fresh_t("init", d.t);
                Add(hoist, assign(s, d));
                lv.list[1].start := s;
                d := add(lv.list[1].start, mul(loop_range, lv.list[1].inc));
                s := var.fresh_t("ubound", d.t);
                Add(hoist, assign(s, d));
                newloop_range := s;
                newloop_start := lv.list[1].start;
                newloop_inc := lv.inc;                
            else
                for d in lv.list do
                   Add(hoist, assign(d.v, d.start));
                   Add(body, assign(d.v, add(d.v, lv.inc)));
                od;
            fi;
         od;
         if (newloop_idx=[]) then
             newloop_idx := loop_idx;
             newloop_range := loop_range;
             newloop_start := V(0);
             newloop_inc := V(1);
         fi;
         Add(hoist, assign(newloop_idx, newloop_start));
         Add(body, assign(newloop_idx, add(newloop_idx, newloop_inc)));


         #This is the memory accumulator logic
         for d in confirmed_memory_acc do
             asns := Collect(hoist, [assign, @(1), [deref, d.loc]]);
             if Length(asns)<>1 then
                 Error("Multi assign in hoist????");
             else
                 SubstBottomUp(body, [deref, d.loc], e->asns[1].loc);
                 Add(epilogue, assign(d, asns[1].loc));
             fi;
         od;

         c := Compile.declareVars(chain(
                 chain(hoist), 
                 doloop(newloop_idx, newloop_range, chain(body)), 
                 chain(epilogue)));
    fi;
    return c;
end;
