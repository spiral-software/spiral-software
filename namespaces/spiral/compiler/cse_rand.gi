
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


RandInput := () -> Complex(
    RandomList([1..2^27])/(1.0*2^27),
    RandomList([1..2^27])/(1.0*2^27));

RandValExp := function(exp)
    local e, rargs;
    if IsBound(exp.randVal) then 
	return exp.randVal;

    elif IsValue(exp) then
	exp.randVal := exp.v;
	return exp.randVal;

    elif IsLoc(exp) then
	exp.randVal := RandInput();
	#Print("RandInput : ", exp, " = ", exp.randVal, "\n");
	return exp.randVal;
    else 
	rargs := List(exp.args, RandValExp);
	e := ApplyFunc(ObjId(exp), rargs);
	e := e.ev();
	Constraint(not IsExp(e) and not IsLoc(e));
	exp.randVal := e;
	return exp.randVal;
    fi;
end;

RandValAssign := function(cmd) 
    local r;
    if IsBound(cmd.loc.randVal) then return cmd.loc.randVal;
    elif IsBound(cmd.exp.randVal) then
	cmd.loc.randVal := cmd.exp.randVal;
	return cmd.loc.randVal;
    else
	r := RandValExp(cmd.exp);
	cmd.exp.randVal := r;
	cmd.loc.randVal := r;
	return r;
    fi;    
end;
    
Class(RCSE, rec(
    cseLookup := (self, randVal) >> HashLookup(self.csetab, V(randVal)),

    cseAdd := meth(self, cmd)
        local op, ent, v;
	if not IsBound(cmd.exp.isExpComposite) then return; fi;
	HashAdd(self.csetab, V(cmd.loc.randVal), cmd.loc);
	HashAdd(self.csetab, V(-cmd.loc.randVal), neg(cmd.loc));
    end,

    cseCmd := meth(self, cmd)
        local lkup;
	RandValAssign(cmd);
	lkup := self.cseLookup(cmd.loc.randVal); 
	if lkup <> false then
	    cmd.exp := lkup;
	else
	    self.cseAdd(cmd);
	fi;
    end,

    remove_node := (self, exp) >> Chain(
	HashDelete(self.csetab, V(exp.randVal)), 
	HashDelete(self.csetab, V(-exp.randVal))),

    node := meth(self, t, exp)
        local lkup, v;
	if IsValue(exp) or IsLoc(exp) then
	    RandValExp(exp); 
	    return exp; 
	fi;

	exp := ApplyFunc(ObjId(exp), List(exp.args, a->self.node(t, a)));
	RandValExp(exp);

	lkup := self.cseLookup(exp.randVal); 
	if lkup <> false then
	    return lkup;
	else
	    v := var.fresh_t("t", t);
	    v.randVal := exp.randVal;
	    v.def := assign(v, exp);
	    self.cseAdd(v.def);
	    return v;
	fi;
    end,

    flush := meth(self)
        self.csetab := HashTable( (val,size) -> val.t.hash(val.v, size) );
    end,

    __call__ := meth(self, code)
        self.csetab := HashTable( (val,size) -> val.t.hash(val.v, size) );
	DoForAll(Collect(code, assign), cmd -> self.cseCmd(cmd));
	return code;
    end
));
