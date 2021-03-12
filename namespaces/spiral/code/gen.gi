
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#IsDigit := c -> c in "0123456789";
#var.prefix := meth( self )
#    local p;
#    p := FirstPosition(self.id,IsDigit);
#    p := SplitAt(self.id,When(p>0,p-1,0))[1];
#    Constraint(p <> "");
#    return p;
#end;

Class(VarGenBase, rec(
    __call__ := self >> WithBases(self, rec(counter:=rec(), pfx := "")),

    suffix := n -> StringInt(n),

    reset := meth(self) self.counter := rec(); end,

    nextu := (self, t) >> When(self.pfx="", self.next(t, "u"), self.next(t, "")),

    next := meth(self, t, pfx)
        local p;
	Constraint(IsString(pfx)); Constraint(IsType(t));
	pfx := Concat(self.pfx, pfx);
	if not IsBound(self.counter.(pfx)) then 
            # first will be plain "pfx", then "pfx2" 
	    p := self.var(t, pfx); 
	    self.counter.(pfx) := 2;
	else
	    p := self.var(t, Concat(pfx,  self.suffix(self.counter.(pfx))));
	    self.counter.(pfx) := self.counter.(pfx) + 1;
	fi;
	return p;
    end,

    nextName := (self, pfx) >> self.next(TInt, pfx).id,
        
    var := (t, pfx) -> Error("must be implemented in subclasses"),

    # creates a cloned generator with aliased .counter, and an extra prefix
    # that will be added to all params created with .next()
    withPrefix := (self, pfx) >> When(
        pfx="i",
        Error("i prefix is RESERVED for loop variables!"),
        WithBases(self, rec(pfx := Concat(self.pfx, pfx))))
));


Class(VarGenNumeric, VarGenBase, rec(
    var := (t, name) -> var(name, t)
));


Class(VarGenSymbolic, VarGenBase, rec(
    var := (t, name) -> var(name, t),
    suffix := VarNameInt
));


Class(ParamGenNumeric, VarGenBase, rec(
    var := (t, name) -> param(t, name)
));


Class(ParamGenSymbolic, VarGenBase, rec(
    var := (t, name) -> param(t, name),
    suffix := VarNameInt
));


Class(NewInt,rec(
	__call__ := meth(self)
	    self.n := self.n + 1;
	    return self.n;
	end,

	n := 0,
));


Class(VarMapper, rec(
    __call__ := (self, vargen) >> WithBases(self, rec(
	    sn := NewInt(),
	    bindings := tab(),
	    vargen := vargen)),

    reset := meth(self)
	self.vargen.reset();
	self.bindings := tab();
	self.sn := NewInt();
    end,

    ignore := (self, var) >> false,

    alreadyMapped := (self,var) >> IsBound(var.sn) and var.sn = self.sn,

    map := meth(self, var) 
        local newvar;
	if self.ignore(var) or (IsBound(var.NoReMap) and var.NoReMap) or self.alreadyMapped(var) then
	    return var;
	fi;
        if IsBound(self.bindings.(var.id)) then 
	    return self.bindings.(var.id);
	else
	    newvar := self.vargen.next(var.t, var.id{[1]});
	    newvar.sn := self.sn;
	    self.bindings.(var.id):=newvar;
	    return newvar;
	fi;
    end
));

ProperName := function(o)
    o.properName := true;
    return o;
end;

_RemapVars := function(c, ignore_list, varGen )
    local vmapper;
    vmapper := CopyFields(VarMapper(varGen), 
        rec(ignore := (self, var) >> IsBound(var.properName) and var.properName=true
                                     or var.id in ignore_list));
    return SubstTopDownRules(c, [
	    [var,v->vmapper.map(v)],
	    [decl,d->decl(List(d.vars,v->vmapper.map(v)),d.cmd)],
	    [data,d->data(vmapper.map(d.var),d.value,d.cmd)] ]);
end;

RemapVars := c -> _RemapVars(c, ["X", "Y"], VarGenNumeric);

RemapVarsIgnore := (c, ignore_list) -> _RemapVars(c, Concatenation(["X", "Y"], ignore_list));

RemapVarsSafe := function(c, ignore_list)
    local free, args, init;
    free := c.free();
    args := Set(ConcatList( Collect(c, @(1, [func])), x->x.params));
    init := Collect(c, @(1, [var,param], x->IsBound(x.value) or IsBound(x.init))); # .value and .init: this is a hack!
    return  _RemapVars(c, List(free::args, x->x.id) :: init :: ignore_list, ParamGenSymbolic());
end;


Class(RemoveUnusedVars,rec(__call__:=function(code)
    local usedvars,declvars,unusedvars;
  
    usedvars := Set(Flat([
		  	Collect(code,var)	# all used vars in operands of assign-cmds
		]));

    declvars := Set(Flat([
##  			 X, Y, # implicitly they are arguments of DFT kernel
			 List(Collect(code,decl),d->d.vars)
		]));

    unusedvars := declvars;
    SubtractSet(unusedvars,usedvars);

    # remove unused vars
	code := SubstTopDown(code,decl,d->decl(Difference(d.vars, unusedvars),d.cmd));

    return code;
end));

##  # Maybe, it would be usefull to add someting else to remove unused loops, data...
##  Class(RemoveUnused...,rec(__call__:=function(code)
##      local usedvars,declvars,unusedvars,v;
##      declvars := Set(Flat([
##  			 List(Collect(code,data),d->d.var),
##  			 List(Collect(code,loop),l->l.var)
##  		]));
##      unusedvars := declvars;
##      SubtractSet(unusedvars,usedvars);
##  
##			# NOTE: prefixes for data and loop var-names?
##      for v in unusedvars do
##        if v.id[1] = 'D' then
##          code := SubstTopDown(code,data,d->Cond(d.var=v,d.cmd,d));
##        elif v.id[1] = 'i' then
##          code := SubstTopDown(code,loop,d->Cond(d.var=v,d.cmd,d));
##        else # decls
##          code := SubstTopDown(code,decl,d->decl(RemoveList(d.vars,v),d.cmd));
##        fi;
##      od;
