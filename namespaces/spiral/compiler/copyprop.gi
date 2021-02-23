
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


neg.doPeel := true;

#F CopyPropagate(<code>) 
#F     Performs copy propagation and strength reduction
#F     If <code> has IFs it MUST be in SSA form, i.e., 
#F     SSA(<code>) must be run first.
#F
Class(SubstVarsRules, RuleSet, rec(
    create := (self, varmap) >> Inherit(self, 
	rec(
            rules := rec(
		v := Rule(@(1,var,e->IsBound(varmap.(e.id))), e -> varmap.(e.id))
	    )
	)).compile()
));

# apply linear normalization to index expressions, this will simplify cases like
# 3x + 2x = 5x
#
Class(SimpIndicesRules, RuleSet, rec(
    create := (self, opts) >> Inherit(self, 
	rec(
	    rules := rec(
		simp_indices := Rule(@(1, opts.simpIndicesInside), x -> GroupSummandsExp(x))
	    )
	)).compile()
));

# Transforms nth(x, idx) to deref(x + idx)
Class(RulesDerefNth, RuleSet);
RewriteRules(RulesDerefNth, rec(
    deref_nth := Rule(
	[nth, @(1).cond( x -> not (x _is [Value, param]) or not IsBound(x.value)), @(2)], 
	e -> let(
	    b := @(1).val, idx := @(2).val, 
	    Cond(
		ObjId(idx) = add, deref(ApplyFunc(add, [b] :: idx.args)),
		ObjId(idx) = sub, deref(ApplyFunc(add, [b] :: [idx.args[1], neg(idx.args[2])])),
                                  deref(b + idx))))
));

# sorts the incoming array of assignments by the offset of the
# variable being assigned TO, rather than the one assigned FROM.
# improves locality hence cache performance.
_sortByIdx := function(array)
    local newarray;
    newarray := Copy(array);

    SortParallel(
        List(array, e -> Double(SubString(e.args[2].id, 2))),
        newarray
    );

    return newarray;
end;

CopyTab := function (t)
   local r, a;
   r := tab();
   for a in NSFields(t) do
      r.(a):=Copy(t.(a));
   od;
   return r;
end;

Class(CopyPropagate, CSE, rec(
    propagate := (self, cloc, cexp) >> let(cebase := ObjId(cexp),
        ((cebase in [var, Value, noneExp])
            or (self.propagateNth and cebase in [nth, deref] and IsValue(cexp.idx)) 
            or (IsBound(self.opts.autoinline) and IsBound(cloc.succ) and Length(cloc.succ)<=1)
            or (IsBound(cloc.succ) and Length(cloc.succ)=0))),

    # sreduce_and_subst(c.exp) can lead to inf loop - why?
    procRHS := (self, x) >> self.sreduce(self.subst3(self.sreduce(self.subst2(x)))),

    procLHS := (self, x) >> self.sreduce3(self.sreduce(self.subst2(x))),

    # NOTE: explain this
    prepVarMap := meth(self, initial_map, do_sreduce, do_init_scalarized, do_idx)
        local v, varmap, rs_subst, rs2, rs_deref, rs_idx;
        self.varmap := initial_map;

        if do_sreduce then
	    # do_idx is done initially, at the same time as scalarization
	    # deref prevents scalarization, so  must be disabled at that moment
	    rs_deref := When(not do_idx and self.opts.useDeref, RulesDerefNth, EmptyRuleSet);
	    rs_idx   := When(do_idx, SimpIndicesRules.create(self.opts), EmptyRuleSet);

            rs_subst := SubstVarsRules.create(self.varmap);
	    rs2 := MergedRuleSet(rs_subst, rs_deref);
	    rs_subst.__avoid__ := [Value];
	    rs_idx.__avoid__ := [Value];
	    rs2.__avoid__ := [Value];
	    
            self.subst   := x -> SubstVars(x, self.varmap);
            self.subst2  := x -> BU(x, rs2);
            self.subst3  := x -> SubstBottomUpRules(BU(x, rs_subst), rs_idx.rules);
            self.sreduce := x -> SReduce(x, self.opts);
            self.sreduce3 := x -> SubstBottomUpRules(x, rs_idx.rules);
            self.sreduce_and_subst := MergedRuleSet(rs_subst, rs_deref, RulesStrengthReduce);
        else
            self.subst   := x -> SubstVars(x, self.varmap);
            self.subst2  := x -> SubstVars(x, self.varmap);
            self.subst3  := x -> SubstVars(x, self.varmap);
            self.sreduce := x -> x;
            self.sreduce_and_subst := self.subst;
        fi;

        if do_init_scalarized then
            for v in compiler.Compile.scalarized do 
                self.varmap.(v.id) := noneExp(v.t); 
            od;
        fi;
        return self;
    end,

    assumeSSA := false,
    afterSSA := self >> WithBases(self, rec(assumeSSA:=true)),

    closeVarMap := meth(self, other, newcmds)
        local v;
        for v in UserNSFields(other.varmap) do
            if (not IsBound(self.varmap.(v)) or self.varmap.(v) <> other.varmap.(v)) and SuccLoc(var(v))<>[] then
                if self.assumeSSA then ;
                #    self.varmap.(v) := other.subst(other.varmap.(v));
                #    PrintLine("close ", v, " => ", other.subst(other.varmap.(v)));
                else
                    Unbind(self.varmap.(v));
                    Add(newcmds.cmds, assign(var(v), other.subst(other.varmap.(v))));
                fi;
            fi;
        od;
        #Print("-----\n");
    end,

    init := meth(self, opts)
        self.opts := opts;
        self.prepVarMap(tab(), true, true, false);
        self.doScalarReplacement := opts.doScalarReplacement;
        self.propagateNth := opts.propagateNth;
        self.flush();
        return self;
    end,

    initial := meth(self, code, opts)
        self.opts := opts;
        self.prepVarMap(tab(), true, true, true);
        self.doScalarReplacement := opts.doScalarReplacement;
        self.propagateNth := opts.propagateNth;
        self.flush();
        return self.copyProp(code);
    end,

    __call__ := (self, code, opts) >> self.init(opts).copyProp(code),

    flush := meth(self)
         self.csetab := tab();
         if (self.doScalarReplacement) then self.lhsCSE := CSE.init(); fi;
    end,

    fast := meth(self, code, opts)
        self.prepVarMap(tab(), false, false, false);
        self.doScalarReplacement := opts.doScalarReplacement;
        self.propagateNth := opts.propagateNth;
        self.opts := opts;
        self.flush();
        return self.copyProp(code);
    end,

    procAssign := meth(self, c, newcmds, live_out)
        local newloc, newexp, cid, cloc, op, varmap, lkup;
        varmap := self.varmap;

        # NOTE: generalize this, use .in()/.out()/.inout() somehow
        if IsBound(c.p) then  
            cid := (loc, exp) -> ObjId(c)(loc, exp, c.p); 
        else cid := ObjId(c); fi;

        # to my current understanding, if we assign a phi function, nothing can be propagated.
        # If you propagate inside a phi, you lose the branch information.
        if (ObjId(c.exp)=phi) then Add(newcmds, c); return; fi;

        # Run strength reduction, variable remapping, and all other rewrite rules
        newexp := self.procRHS(c.exp); 
        cloc := When(not IsVar(c.loc) or (c.loc in c.op_inout()), 
	             self.procLHS(c.loc), c.loc);

        # check if newexp was already computed
        lkup := self.cseLookup(newexp);
        if lkup <> false then newexp := lkup; fi;

	# if cloc is marked as live_out, save it in the list, so that its not kicked out
        When(IsVar(cloc) and IsBound(cloc.live_out), AddSet(live_out, cloc));

        # invalidate the LHS in the CSE tables
        if (self.cseLookup(cloc) <> false) 
            then self.cseInvalidate(cloc); fi;      
        if (self.doScalarReplacement and self.lhsCSE.cseLookup(cloc) <> false) 
            then self.lhsCSE.cseInvalidate(cloc); fi;
    
        # propagate
        if IsVar(cloc) and cid=assign and self.propagate(cloc,newexp) then
            # this should not happen due to sreduce/subst above 
            When(IsVar(newexp) and IsBound(varmap.(newexp.id)), Error("Should not happen"));
            varmap.(cloc.id) := newexp; 

        # do not propagate
        else
            # NOTE: YSV: is this right???
            #        it seems that its correct, above case catches var=noneExp,
            #        so this looks like a mem-store, e.g.,  nth(..) = noneExp
            if ObjId(newexp)=noneExp then return; fi;

            # do not propagate AND cloc is a variable
            if IsVar(cloc) and not (cloc in c.op_inout()) then          
                # propagate 'neg' like unary operators outwards
                if IsBound(newexp.doPeel) and newexp.doPeel then
                    op := ObjId(newexp); 
                    newexp := newexp.args[1];
                    if self.propagate(cloc, newexp) then
                        if IsVar(newexp) and IsBound(varmap.(newexp.id)) then
                            varmap.(cloc.id) := op(varmap.(newexp.id));
                        else
                            varmap.(cloc.id) := op(newexp);
                        fi;
                    else
                        newloc := cloc.clone();
                        varmap.(cloc.id) := op(newloc);
                        Add(newcmds, cid(newloc, newexp)); 
                        # careful! cid can be any storeop, not always <assign>
                        When(cid=assign, self.cseAdd(newloc, newexp));
                    fi;
                else 
                    # variable that is not propagated is remapped to a fresh name (to get code in SSA form)
                    newloc := cloc.clone(); 
                    varmap.(cloc.id) := newloc;
                    Add(newcmds, cid(newloc, newexp));   # cid == assign | assign_nop | ...
                    # careful! cid can be any storeop, not always <assign>
                    When(cid=assign, self.cseAdd(newloc, newexp));
                fi;

            #  do not propagate AND cloc is *not* a variable (ie. nth(X, i)) or it is an inout variable
            else
                if self.doScalarReplacement or not (self.propagateNth or IsVar(newexp) or IsValue(newexp)) then
                    # for non-variable newexp a fresh temporary var sXX is created to hold the result
                    newloc := var.fresh_t("s", newexp.t);
                    self.cseAdd(newloc, newexp);
                    Add(newcmds, assign(newloc, newexp));
                    newexp := newloc;
                fi;

                if self.doScalarReplacement then
                    When(cid<>assign or not (ObjId(cloc) in [nth,deref]), 
                        Error("Scalar replacement can't handle <cloc> of type ", ObjId(cloc))); 
                    self.cseAdd(newexp, cloc);
                    self.lhsCSE.cseAdd(newexp, cloc);
                else
                    Add(newcmds, cid(cloc, newexp));
                fi;
            fi;
        fi;
    end,

    procIF := meth(self, c, newcmds, live_out)
        local lo_map, v, orig, then_cmd, else_cmd, then_cp, else_cp;
        c.cond := self.procRHS(c.cond);

        if IsValue(c.cond) then
            Error("This is a constant IF, it should be folded away before reaching copyprop");
            # Following code should work if needed (if you uncomment the error) but it is not optimized
            # self.flush(); # YSV: why is this needed??? I will comment this out.
            if c.cond.v=0 or c.cond.v=false then  Add(newcmds, self.copyProp(c.else_cmd));
            else                                  Add(newcmds, self.copyProp(c.then_cmd));
            fi;
        else
            # Here the idea is that each part of the branch should work with
            # the original csetab (there cannot be crossings) and the final thing
            # should be restored to the original csetab

            then_cp := CopyFields(self, rec(csetab := CopyTab(self.csetab)))
                .prepVarMap(CopyTab(self.varmap), true, false, false);

            else_cp := CopyFields(self, rec(csetab := CopyTab(self.csetab)))
                .prepVarMap(CopyTab(self.varmap), true, false, false);

            then_cmd := then_cp.copyProp(c.then_cmd);
            self.closeVarMap(then_cp, then_cmd);

            else_cmd := else_cp.copyProp(c.else_cmd);
            self.closeVarMap(else_cp, else_cmd);

            Add(newcmds, IF(c.cond, then_cmd, else_cmd));
            #self.flush();
        fi;
    end,
        
    #if we do Scalar Replacement on the fly, reinject the final writes as assigns (for now)
    finalizeScalarReplacement := meth(self, newcmds)
        local entry;
            if IsBound(self.lhsCSE.csetab.nth) then 
                # MRT: sort by order of variable being assigned TO rather
                # than variable assigned FROM. This reduces cache misses by
                # exploiting possible locality in the output array.
                #
                # eg: Y[0]  := a1   ===>  Y[0]   := a1
                #     Y[32] := a2         Y[1]   := a3
                #     Y[1]  := a3         Y[32]  := a2

                for entry in _sortByIdx(self.lhsCSE.csetab.nth) do
                    if Length(entry.args)=2 then
                        Add(newcmds, 
                            assign(nth(entry.args[1],entry.args[2]),
                                entry.loc));
                    fi;
                od;
            fi;
            if IsBound(self.lhsCSE.csetab.deref) then 
                for entry in self.lhsCSE.csetab.deref do
                    if Length(entry.args)=1 then
                        Add(newcmds, 
                            assign(deref(entry.args[1]),
                        entry.loc));
                    fi;
                od;
            fi;
    end,

    # create the reverse mapping, to map last assignment to live_out variable 
    #  to the actual variable intead of an SSA related substitute
    substLiveOut := meth(self, newcmds, live_out)
        local varmap, lo_map, active, v;
        varmap := self.varmap;
        lo_map := tab();
        active := Set(Collect(newcmds, var));  # explain this
        for v in live_out do 
           # NOTE: The commented out lines with #X removed extra copies 
           #  associated with live_out variables. But it turns out that this 
           #  does not work if a live_out variable is never actually remapped,
           #  but has an entry in varmap due to copy propagation.
           #  For example (a = t11; b = a) if a is live_out will be compiled to
           #  (b=t11) b CopyPropagate and converted to (b=a) by function below.
           #   which is incorrect 

           #X if not IsVar(varmap.(v.id)) or not (varmap.(v.id) in active) then
                Add(newcmds, assign(v, varmap.(v.id)));
           #X else
           #X     lo_map.(varmap.(v.id).id) := v;
           #X fi;
            Unbind(varmap.(v.id));   # prevent duplication of cleanup code due to closeVarMap
        od;
        newcmds := SubstVars(newcmds, lo_map); 
        return newcmds;
    end,

    copyProp := meth(self, code)
        local c, cid, cmds, newcmds, live_out;
        cmds := When(ObjId(code)=chain, code.cmds, [code]); 
        newcmds := [];
        live_out := Set([]);
        for c in cmds do
            cid := ObjId(c); 
            if   IsAssign(c)     then  self.procAssign(c, newcmds, live_out);
            elif (cid = IF)      then  self.procIF(c, newcmds, live_out);
            elif IsExpCommand(c) then  Add(newcmds, self.procRHS(c));
            elif (cid = skip)    then  ;  # do nothing

            # if .sideeffect is bound, the command is assumed have side effects
            # so one has to enable array scalarization inside all its children
            elif IsRec(c) and IsBound(c.sideeffect) and c.sideeffect then
               Add(newcmds, map_children_safe(c, x -> self.procRHS(x)));

            # the command is a container for other functions, copyprop all children
            else Add(newcmds, map_children_safe(c, 
			x -> Cond(IsCommand(x), self.copyProp(x), self.procRHS(x))));
            fi;
        od;

        When(self.doScalarReplacement, self.finalizeScalarReplacement(newcmds));
        newcmds := self.substLiveOut(newcmds, live_out);
        # self.flush();
        return chain(newcmds);
    end
));
