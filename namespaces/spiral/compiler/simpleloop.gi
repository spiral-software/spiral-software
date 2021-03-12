
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(_ExpandInvars, RuleSet);
RewriteRules(_ExpandInvars, rec(
    shift_depends_left_in_add := ARule(add, [@(1).cond(e-> ObjId(e)<>depends), @(2,depends)], 
        e->[@(2).val, @(1).val]),

    shift_depends_left_in_mul := ARule(mul, [@(1).cond(e-> ObjId(e)<>depends), @(2,depends)], 
        e->[@(2).val, @(1).val]),

    fuse_depends_in_add := ARule(add, [@(1,depends), @(2,depends)], 
        e->[depends(add(@(1).val.args[1], @(2).val.args[1]), Union(@(1).val.args[2].v, @(2).val.args[2].v))]),

    fuse_depends_in_mul := ARule(mul, [@(1,depends), @(2,depends)], 
        e->[depends(mul(@(1).val.args[1], @(2).val.args[1]), Union(@(1).val.args[2].v, @(2).val.args[2].v))]),

    leq_depends := Rule([leq, @(1, depends), @(2, depends), @(3, depends)],
        e->depends(leq(@(1).val.args[1], @(2).val.args[1], @(3).val.args[1]), 
                Union(@(1).val.args[2].v, @(2).val.args[2].v, @(3).val.args[2].v))),

    imod_depends := Rule([imod, @(1, depends), @(2, depends)],
        e-> depends(imod(@(1).val.args[1],@(2).val.args[1]), Union(@(1).val.args[2].v, @(2).val.args[2].v))),

    extract_depends := Rule([@(0, [add,mul]), @(1,depends)],
        e-> @(1).val),

    div_depends := Rule([div, @(1, depends), @(2, depends)],
        e-> depends(div(@(1).val.args[1],@(2).val.args[1]), Union(@(1).val.args[2].v, @(2).val.args[2].v))),

    idiv_depends := Rule([idiv, @(1, depends), @(2, depends)],
        e-> depends(idiv(@(1).val.args[1],@(2).val.args[1]), Union(@(1).val.args[2].v, @(2).val.args[2].v))),

    tcast_depends := Rule([tcast, @(1), @(2, depends)],
        e -> depends(tcast(@(1).val,@(2).val.args[1]), @(2).val.args[2])),

    add_assoc := RulesStrengthReduce.rules.add_assoc,
    
    mul_assoc := RulesExpensiveStrengthReduce.rules.mul_assoc,

    depends_deref := Rule( [deref, @(1,depends)],   #One needs to introduce depends memory to handle loads and stores differently
        e -> depends_memory(deref(@(1).val.args[1]), @(1).val.args[2])),

    nth_depends := Rule([nth, @(1, depends), @(2, depends)],           
        e -> depends_memory(nth(@(1).val.args[1],@(2).val.args[1]),  Union(@(1).val.args[2].v, @(2).val.args[2].v))),

    vdup_depends_memory := Rule([vdup, @(1,depends_memory), @(2)],
        e -> depends_memory(vdup(@(1).val.args[1], @(2).val), @(1).val.args[2])),

    assign_load_depends := Rule([assign, @(1), @(2, depends_memory)], #This is a load, good invariant!
        e -> assign(@(1).val, depends(@(2).val.args[1], @(2).val.args[2]))),

    mul_depends_accu := ARule(mul, [@(1,depends, e->Length(e.args[2].v)=0), @(2,accu)],
        e -> [accu(mul(@(1).val, @(2).val.rChildren()[1]), 
            mul(@(1).val, @(2).val.rChildren()[2]), 
            mul(@(1).val, @(2).val.rChildren()[3]))]),

    add_depends_accu := ARule(add, [@(1,depends, e->Length(e.args[2].v)=0), @(2,accu)],
        e -> [accu(add(@(1).val, @(2).val.rChildren()[1]), 
            @(2).val.rChildren()[2], 
            add(@(1).val, @(2).val.rChildren()[3]))]),

    div_depends_accu := Rule([div, @(1, accu), @(2,depends, e->Length(e.args[2].v)=0)],
        e -> accu(div(@(1).val.rChildren()[1], @(2).val),
            div(@(1).val.rChildren()[2], @(2).val),
            div(@(1).val.rChildren()[3], @(2).val))),

    accu_init := Rule( [accu, @(1,depends), @(2), @(3)], #The initialization step doesn't need to be hoisted 
        e -> accu(@(1).val.args[1], @(2).val, @(3).val)),        #Since it is gonna be out of the loop
    
    #These are security checks
    depends_depends := Rule([depends, depends],
        e->Error("Your code is somehow aliased and SimplifyLoop will probably mess it up. Fix it first then come back!")),

    assign_load_depends := Rule([assign, depends_memory, @], #This is a store invariant! Too good to be true...
        e -> Error("A store invariant has been found which is unlikely to be correct")),
));


Class(_ExpandInvarsUnsafeXXX, RuleSet);
RewriteRules(_ExpandInvarsUnsafeXXX, rec(
    unsafe_memory_depends := Rule(@(1,depends_memory), e-> depends(@(1).val.args[1], @(1).val.args[2]))
));

_ExpandInvarsUnsafe := MergedRuleSet(_ExpandInvarsUnsafeXXX, _ExpandInvars);

Declare(_CreateVirtualVars);
_CreateVirtualVars := function(dep, str, csetable, opts)
    local idx, rem, n, a, v, l;
    if Length(dep.args[2].v)=0 then
        v := SReduce(dep.args[1], opts);
        if IsValue(v) or IsVar(v) or ObjId(v)=param then
            return [v, skip()];
        else
            if csetable<>false then
                l := csetable.cseLookup(v);
                if (l<>false) then 
                    return [l, skip()]; 
                else
                    n := var.fresh_t(str, dep.t);
                    csetable.cseAdd(n, v);
                    return [n, assign(n, v)];
                fi;
            else
                n := var.fresh_t(str, dep.t);
                return [n, assign(n, v)];
            fi;
        fi;
    else
        idx := dep.args[2].v[1];
        rem := Drop(dep.args[2].v, 1);
        [n, a] := TransposedMat(List([1.. idx.range], 
                i -> _CreateVirtualVars(depends(SubstVars(Copy(dep.args[1]), tab((idx.id) := V(i-1))), rem), str, csetable, opts)));
        return [virtual_var(n, idx), a];
    fi;
end;

_SimplifyLoop := function(c, to_be_unrolled, opts)
    local free_vars, invars, l, v, assign_accs_init, assign_accs_final, x, accu_init, accu_acc, _DisableAccumulatorBound, 
          loop_var, accu_bound, virtual_free_vars, csetable, selectedaccus, allaccus, allincs, i, j, t;

    if not(ObjId(c) in [loop,loopn]) then
            Error("SimplifyLoop has to be called on a loop!");
    fi;

    #Make sure the loop is not simplified again later!
    #Note that this is *critical* : we do not do any dependence analysis and thus
    #it is assumed that all free variables are loop independent which is not the case
    #anymore after the loop has been processed
    c.simplified := true;

    #If the loop is not gonna be unrolled, we will convert it to an accumulator loop
    #That is, we will use a while() loop instead of a for loop and this will allow us
    # to manipulate the loop index aka do a loop with multiple indices that start at different
    #values and increments differently
    if not(to_be_unrolled) then
        c := When(ObjId(c)=loopn,
            SubstBottomUp(c, c.var, x->accu(V(0), V(1), V(c.range))),
            SubstBottomUp(c, c.var, x->accu(V(0), V(1), V(Last(c.range)+1))));
    fi;

    #All free variables, params and values are assumed to be pure loop invariants (depends on nothing)
    free_vars := Set(c.free());
    c := SubstBottomUp(c, @(1, [param, Value], e -> not (ObjId(e.t) = TSym)), x->depends(x, Set([])));
    c := SubstBottomUp(c, @(1,var, e-> e in free_vars ), x->depends(x, Set([])));

    #Propagation of these pure loop invariants allow us to propagate inside the loop accumulators and
    #Define them properly.
    #NOTE: we could have add accumulators that actually depend on internal unrolled loops (as explained later) but then
    #it would actually require as many accumulators as the internal loop range which would waste registers. Note that
    #it's just a feeling, it wasn't tested.


    #This is a bad HACK
    #depends_memory exists because of the DGEMM assign_accs, but it
    #also breaks the DFTs. So when there's no assign acc, we turn them off.
    c := When(Length(Collect(c, assign_acc))>0, _ExpandInvars(c), _ExpandInvarsUnsafe(c));

    #Loop indices that will be unrolled later are somehow loop invariant of the external loop without
    #being loop invariants of the internal loop. As we want to hoist these variables anyhow, we introduce them as depends(exp, [idx])
    virtual_free_vars := When(to_be_unrolled,
        Set(List(Collect(c.cmd, loop), l -> l.var)),
        Set(List(Collect(c.cmd, @@(0, loop,(e, cx) -> (IsBound(cx.unroll_cmd)) and (Length(cx.unroll_cmd)>0))), l -> l.var)));
    c := SubstBottomUp(c, @(1,var, e-> e in virtual_free_vars ), x->depends(x, Set([x])));

    #Propagation now extends dependent variables. Note that dependent variables do not mix with accu so this prevents 
    #dependent accumulators
    #Again, the hack.
    c := When(Length(Collect(c, assign_acc))>0, _ExpandInvars(c), _ExpandInvarsUnsafe(c));

    #Kick out trivial invariants. e.g.: constant are always invariants, useless ones!
    c := SubstBottomUp(c, [depends, @(1, [var, Value, param, tcast]), @], x->@(1).val);

    #This rule speeds up processing. It is not necessary but it works well... usually
    #Invariants that do not comprise params are probably constants so forget them!
    c := SubstBottomUp(c, @(0, depends, e-> Length(Collect(e, param))=0), x->@(0).val.args[1]);

    #As scalar increments are essentially free, we merge accumulators that have the same 
    #increment if the difference of their initial values is a scalar
    allaccus := Set(Collect(c, accu));
    allincs := Set(List(allaccus, x->x.args[2]));
    for i in allincs do
        selectedaccus := Filtered(allaccus, x-> x.args[2] = i);
        for j in DropLast(selectedaccus,1) do
            t := j.args[1]-Last(selectedaccus).args[1];
            if ObjId(t)=Value then
                j.substitute := add(Last(selectedaccus), t);
            fi;
        od;
    od;
    c := SubstBottomUp(c, @(1, accu, e ->IsBound(e.substitute)), e->e.substitute);

    #If we are to transform in a while loop, we only really need one loop counter
    #Therefore, we kick out all loop counters but one
    if not(to_be_unrolled) then
        x := true;
        _DisableAccumulatorBound := function(e)
            if x then
                x:=false;
                e.isLoopBound := true;
                return e;
            else
                return accu(e.rChildren()[1], e.rChildren()[2], V(0));
            fi;
        end;
        SubstBottomUp(c.cmd, accu, e -> _DisableAccumulatorBound(e));
    fi;

    #Create new variables for all invariants
    csetable := CSE.init();
    l := Set(Collect(c, depends));
    [v, invars] := let(a := List(l, x->_CreateVirtualVars(x, "ivr", csetable, opts)), 
        When(Length(a)>0, TransposedMat(a), [[],[]]));
    c := SubstBottomUp(c, @(1, depends), e->v[Position(l,e)]);
    
    #Same for all assign_accs!
    l := Set(List(Collect(c, [assign_acc, depends_memory, @]), x->x.loc));
    [v, assign_accs_init] := let(a := List(l, x->_CreateVirtualVars(x, "acc", false, opts)), 
        When(Length(a)>0, TransposedMat(a), [[],[]]));
    assign_accs_final := SubstTopDownNR(Copy(assign_accs_init), assign, x->assign(x.exp, x.loc));
    c := SubstBottomUp(c, @(1, depends_memory), e->v[Position(l,e)]);

    #If the loop is to be unrolled then we're done
    if to_be_unrolled then
        return chain(invars, assign_accs_init, c, assign_accs_final);
    else #if not, then we need to finish the transformation in a while loop

        #drop the loop itself, we will replace it by a doloop later
        c := c.cmd;

        #Replace all accus by new vars
        #retrieve the loop bound and set all bounds to V(0) so they can merge in the Set()
        accu_bound := Copy(Collect(c, @(1, accu, x->IsBound(x.isLoopBound)))[1]);
        [loop_var, accu_bound] := [accu(accu_bound.rChildren()[1], accu_bound.rChildren()[2], V(0)), 
            accu_bound.rChildren()[3]];
        c := SubstBottomUp(c, [accu, @(1), @(2), @(3)], x->accu(@(1).val, @(2).val, V(0)));
        l := Set(Collect(c, accu));
        v := List(l, x->var.fresh_t("accu", x.t));
        accu_init := List([1..Length(l)], i->assign(v[i], l[i].rChildren()[1]));
        #Copy here prevents aliasing with previous
        accu_acc := List([1..Length(l)], i->assign_acc(Copy(v[i]), l[i].rChildren()[2]));  

        loop_var := let(p:=FirstPosition(l, x->(x=loop_var)), v[p]);
        c := SubstBottomUp(c, @(1, accu), e->v[Position(l,e)]);

        #if we cross the unroll_cmd boundary, we need to flag the new variables as alive
        #so copypropagation doesn't mess with it later
        for x in Collect([assign_accs_init, invars, accu_init], assign) do x.loc.live_out := true ; od;

        c := chain(unroll_cmd(chain(invars, assign_accs_init, accu_init)), 
             doloop(loop_var, accu_bound, chain(c, accu_acc)), unroll_cmd(chain(assign_accs_final)));
        return c;
    fi;
end;


SimplifyLoops := function(c, opts)
    c := FlattenCode0(c); #Flatten chains, keep unroll_cmds
    c := SReduce(c, opts); #Simplify index expressions

    c := SubstTopDown(c, @.cond( #Remove all aliases!
            e->IsRec(e) and IsBound(e.rChildren) and IsBound(e.from_rChildren)), 
        e -> e.from_rChildren(List(e.rChildren(), Copy)));

    c := SubstTopDown(c, @(1, [loop,loopn], e->not(IsBound(e.simplified))), 
        (e, cx) -> _SimplifyLoop(e, (IsBound(cx.unroll_cmd)) and (Length(cx.unroll_cmd)>0), opts));
    return c;
end;

