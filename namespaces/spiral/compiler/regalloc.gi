
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# This file implements a register allocator for straight line code
# Based on Belady's farthest-first eviction strategy.  For normal
# three-operand code without memory ops, this should yield the optimal
# (w.r.t. number of spill-related load-stores) code. 
#
# For two-operand code without memory ops, this can still be optimal,
# but I'm not sure.
#
# For two-operand code *with* memory ops (=x86) it is not clear at
# all, whether this simple allocator is best. Main problem is that we
# have a choice of spilling versus using memop in an instruction.  
#
# The output of this pass should be fed into C-compiler, and spill
# space here is represented as a regular array. The hope is that
# C-compiler won't touch the spill array (S), and therefore will not
# run out of registers.  
#
# Potential problems: on 2-operand architectures (x86), the compiler
# can internally operate on 3-operand code, and then convert to
# 2-operand at some point. This may render this allocator useless,
# because 3->2 conversion is likely to destroy lot's of information.
# NOTE: Investigate this further.
#

# we are assuming SSA code
MarkLiveness := function(code)
    local live, add, remove, v, c, already_dead, n;
    MarkDefUse(code);
    
    # Starting from the statement, invariant : variable is dead afters its last use
    already_dead := Set([]); 
    for c in Reversed(code.cmds) do
        c.dead := Difference(PredCmd(c), already_dead);
	UniteSet(already_dead, c.dead);
    od;

    live := Set([]);
    n := 1;
    for c in code.cmds do
        Constraint(IsAssign(c));
	if SuccCmd(c) <> [] then AddSet(live, c.loc); fi;
	SubtractSet(live, c.dead);
	c.live := Copy(live);
	c.n := n; # position in a chain
	n := n+1;
    od;
    return code;
end;

# RequiredVars(<chain>) - minimum number of required variables
# MarkLiveness should be run beforehand
#
RequiredVars := (code) -> Maximum0(List(code.cmds, c->Length(c.live)));

# ReuseVars(<chain>) - reusing variables in the chain
# MarkLiveness should be run beforehand,
# output is no longer in SSA form
#

ReuseVars := function(code)
    local free, current, map, reduced, new, out, v, c, f, alive;
    reduced := [];
    free    := [];
    current := [];
    
    map     := tab();

    for c in code.cmds do
        
        alive := Set(c.live :: Filtered(c.dead, e -> IsVar(e) and not IsArrayT(e.t) and not IsPtrT(e.t)));
        new   := Difference(alive, current);
        out   := Difference(current, alive);

        for v in out do
            if IsBound(map.(v.id)) then
                free := [map.(v.id)] :: free;
                Unbind(map.(v.id));
            else
                free := free :: [v];
            fi;
        od;
        if Length(new)=1 then
            f := PositionProperty(free, e -> e.t=new[1].t);
            if f<>false then
                map.(new[1].id) := free[f];
                free := ListWithout(free, f);
            fi;
        fi;

        Add(reduced, SubstParamsCustom(c, map, [var]));
        current := alive;
    od;

    return chain(reduced);
end;

# RequiredRegs(<chain>) - minimum number of required registers
# MarkLiveness should be run beforehand
#
RequiredRegs := (code, t) -> Maximum(List(code.cmds, c->Length(
	    Filtered(c.live, x->IsBound(x.t) and x.t=t))));


# spill using Belady's MIN heuristic - spill the value used farthest away
_spill := function(n, regmap, regvals, S, free_slots, newcmds)
    local dists, maxdist, spill, evict, r;

    dists := List(regvals, v -> let(
	    dd := Filtered(List(SuccLoc(v), succ -> succ.def.n), succn -> succn >= n),
            # if successor list (beyond current n) is empty, the variable is dead and should
	    # be evicted, assign it a really large dist value 2^20
	    When(dd=[], 2^20, Minimum(dd))));  

    maxdist := Maximum(dists);
    evict := regvals[Position(dists, maxdist)];
    r := regmap.(evict.id); # freed up register

    if not IsBound(evict.value) then # do not save constants into spill slot
	Constraint(Length(free_slots) > 0);
	spill := nth(S, free_slots[1]);
	RemoveSet(free_slots, free_slots[1]);
	Add(newcmds, assign(spill, r));
	regmap.(evict.id) := spill;
    else
	Unbind(regmap.(evict.id));
    fi;

    RemoveSet(regvals, evict);

    return r;
end;

_getreg := function(n, newvar, free, regmap, regvals, S, free_slots, newcmds)
    local r;
    if Length(free) > 0 then 
        # we have free registers, grab <r> from free list
	r := free[1];
	RemoveSet(free, r);
    else
 	# no free registers spill something to free up <r>
	r := _spill(n, regmap, regvals, S, free_slots, newcmds);
    fi;
    AddSet(regvals, newvar);
    regmap.(newvar.id) := r;
    return r;
end;

_reload := function(n, vars, free, regmap, regvals, S, free_slots, newcmds)
    local v, r, reload_from;
    for v in vars do
        reload_from := When(IsBound(regmap.(v.id)), regmap.(v.id), v);
        r := _getreg(n, v, free, regmap, regvals, S, free_slots, newcmds);
	Add(newcmds, assign(r, reload_from));
        # if <reload_from> is a spill location (ie not a variable),
        # then return the  spill slot to the free list
	if not IsVar(reload_from) then 
	AddSet(free_slots, reload_from.idx); fi;
    od;
end;

# MarkLiveness should be run beforehand
#
RegAlloc := function(code, numregs, t) 
    local regs, stack, regmap, regvals, c, cc, free, dead, r, v,
         dists, maxdist, evict, newcmds, 
	 S, free_slots, reqregs, spill, deadregs,
	 reloads;

    reqregs := RequiredRegs(code, t);
    S := var("S", TArray(t, When(reqregs-numregs+1 > 0, reqregs-numregs+21, 0)));
    # free spill slots
    free_slots := Set([0..S.t.size-1]);

    regmap := rec();
    free := Set(List([0..numregs-1], x -> var(Concat("r", StringInt(x)), t)));
    for r in free do r.isReg := true; od;

    regvals := Set([]); # set of all variables that reside in registers (=not spilled)

    newcmds := [];
    for c in code.cmds do
        Constraint(IsAssign(c));
	# plug in allocated registers instead of SSA vars
	cc := Copy(c);

	reloads := Union(
	    # do not reload constants (in register for which regmap.(..) == var, and external)
	    Filtered(cc.op_inout(), x->IsVar(x) and (not IsBound(regmap.(x.id)) or # external
		                                     not IsVar(regmap.(x.id))) and not (TPtr = ObjId(x.t) or TInt = x.t)), # or spilled
	    
	    # do not reload constants or other inputs used only once
	    # ideally we should use Belady heuristic again
	    Filtered(cc.op_in(), x->IsVar(x) and (not IsBound(regmap.(x.id)) or
		                                  not IsVar(regmap.(x.id))
	                                     and (not Length(SuccCmd(x.def))=1)) and not (TPtr = ObjId(x.t) or  TInt = x.t)));

	_reload(cc.n, reloads, free, regmap, regvals, S, free_slots, newcmds);
	if (cc.loc in cc.op_inout()) or (cc.loc in cc.op_in()) then 
	    cc := SubstVars(cc, regmap);
	else
	    cc.exp := SubstVars(cc.exp, regmap);
	fi;
 	# allocate a register if c.loc only is a variable of type t 
	if IsVar(cc.loc) and not IsBound(cc.loc.isReg) and cc.loc.t = t then
	    deadregs := Filtered(cc.dead, v -> IsVar(v) and v in regvals);
	    if deadregs <> [] then
		# grab reg from the dead list (this automatically implements coalescing)
		v := deadregs[1]; 

		RemoveSet(regvals, v);
		AddSet(regvals, cc.loc);
		r := regmap.(v.id);
		Unbind(regmap.(v.id));
		regmap.(cc.loc.id) := r;
		cc.loc := r;
	    else 
		r := _getreg(cc.n, cc.loc, free, regmap, regvals, S, free_slots, newcmds);
		cc.loc := r;
	    fi;
	fi;
	# do not output dummy moves (ie assign(r1, r1)).
	if not (ObjId(cc)=assign and cc.loc = cc.exp) then
	    Add(newcmds, cc);
	fi;
	
	# put registers associated with dead variables on the free list
	for v in cc.dead do
	    if IsVar(v) and (v in regvals) then
		AddSet(free, regmap.(v.id));
		RemoveSet(regvals, v);
		Unbind(regmap.(v.id));
	    fi;
	od;

    od;
    return chain(newcmds);
end;

