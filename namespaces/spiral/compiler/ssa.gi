
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(phi, Exp);

Class(SSAMap, rec(
     __call__ := self >> WithBases(self, rec(
	         operations:=PrintOps,
		 top := 0 ,
		 stack := [ ],       
		 alternatives := [ ],
		 phi_map := tab()
	     )).enter(),
     print := self >> Print("SSAMap(", self.top, ")"),
     
     insert := meth(self, oldvar, newvar) 
         self.stack[self.top].(oldvar.id) := newvar;
     end,

     _lookup := meth(self, v, i)
	 while i >= 1 do
	     if IsBound(self.stack[i].(v)) then return self.stack[i].(v); fi;
	     i := i-1;
	 od;
	 return false;
     end,

     lookup := (self, v) >> self._lookup(v.id, self.top),

     enter := meth(self)
         local newtab;
	 newtab := tab();
         Add(self.stack, newtab);
	 Add(self.alternatives, [newtab]);
	 self.top := self.top + 1;
	 return self;
     end,

     next := meth(self)
         local newtab;     
	 newtab := tab();
         Add(self.alternatives[self.top], newtab);
	 self.stack[self.top] := newtab;
	 return self;
     end,

     join := meth(self)
         local i, alt, tt, id, phi_expr, phi_var, outer_tab, type, alts, tmp;

     #Create a list of all the changed variables in merging scope
#      changed := tab();

#      for alt in self.alternatives[self.top] do
#          for id in Filtered(NSFields(alt),x->x<>"__doc__") do
#              if not(IsBound(changed.(id))) then
#                  changed.(id) := (id);
#              fi;
#          od;
#      od;

     #add alternatives for when variable appear in two different branches
	 tt := tab();
	 for alt in self.alternatives[self.top] do
	     for id in Filtered(NSFields(alt),x->x<>"__doc__") do
	         if IsBound(tt.(id)) then Add(tt.(id).args, alt.(id));
		 else 
		     tt.(id) := phi(alt.(id));
		     tt.(id).t := (alt.(id).t);
		 fi;
	     od;
	 od;

     #if a variable does not appear in all branches, maybe it should be set do its 
     #father version
     alts := Length(self.alternatives[self.top]);
     for id in Filtered(NSFields(tt),x->x<>"__doc__") do
         if (Length(tt.(id).args)<>alts)  then
             tmp := self._lookup((id), self.top-1);
             if tmp<>false then
                 Add(tt.(id).args, tmp);
             fi;
         fi;
     od;

	 outer_tab := self.stack[self.top-1];
	 self.phi_map := tab(); # phiXX -> phi(a, b, c)
	 for id in Filtered(NSFields(tt),x->x<>"__doc__") do 
	     phi_expr := tt.(id);
	     if Length(phi_expr.args) > 1 then # non-trivial phi function
		 phi_var := var.fresh_t("phi", phi_expr.args[1].t);
                 phi_expr.t := phi_var.t;
		 self.phi_map.(phi_var.id) := phi_expr;
		 outer_tab.(id) := phi_var;
		 # we make the following assumptions here
		 # 1. variables inside phi(...) are live outside the if
		 #    (violated when result of phi(...) is never used, eg. phi itself is dead,
		 #     this is safe, but will not remove dead code)
		 # 2. variables not inside phi(...) are dead outside the if
		 #    (violated in obscure cases like if(x) {t1=1} else {}; if(x) {y=t1}else{y=1},
		 #     we can prove that these cases do not arise in linear transforms)
		 for i in phi_expr.args do 
		     i.live_out := true; 
		 od;
	     else  # trivial phi(x)->x
             outer_tab.(id) := phi_expr.args[1];
	     fi;
	 od;
	 return self;
     end,

     leave := meth(self)
         local phi_map;
         phi_map := self.join();
         RemoveLast(self.stack, 1);
         RemoveLast(self.alternatives, 1);
	 self.top := self.top - 1;
	 return phi_map;
     end,

     subst_vars := (self, exp) >>
         FoldR(self.stack, (ex, t) -> SubstVars(ex, t), exp),
	 
     phi_map_assigns := self >> List(Filtered(NSFields(self.phi_map),x->x<>"__doc__"), 
	 f -> assign(var(f), self.phi_map.(f)))

));
     
Declare(_SSA);

SSA := function(code)
    local ssamap, icode;
    ssamap := SSAMap();
    icode := FlattenCode(_SSA(code, ssamap));
    return icode;
end;

_SSA := function(code, ssamap)
    local i, c, newvar, newc, phi_map;

    if ObjId(code) = IF then 
	code.cond := ssamap.subst_vars(code.cond);
	ssamap.enter();	code.then_cmd := _SSA(code.then_cmd, ssamap);
	ssamap.next();	code.else_cmd := _SSA(code.else_cmd, ssamap);
	ssamap.leave();
	return chain(code, ssamap.phi_map_assigns());

    elif ObjId(code) = assign then
	return _SSA(chain(code), ssamap);

    elif IsExpCommand(code) then 
	return ssamap.subst_vars(code);

    elif ObjId(code) = chain then
	for i in [1..Length(code.cmds)] do
	    c := code.cmds[i];
	    if ObjId(c) = assign then
		    c.exp := ssamap.subst_vars(c.exp);
		    if IsVar(c.loc) then
			newvar := var.fresh_t("t", c.loc.t);
			ssamap.insert(c.loc, newvar);
			c.loc := newvar;
		    else
			c.loc := ssamap.subst_vars(c.loc);
		    fi;
	    else
		code.cmds[i] := _SSA(c, ssamap);
	    fi;
	od;
	return code;
    elif IsRec(code) and IsBound(code.sideeffect) and code.sideeffect then
        return map_children_safe(code, x->ssamap.subst_vars(x));
    elif IsCommand(code) then
        return code;
    else
        Error("Can't handle ", ObjId(code));
    fi;
 end;

StronglyConnectedComponents := function(ll)
    local result, l, bool, r;

    result := [];
    for l in ll do
        bool := true;
        for r in [1..Length(result)] do
            if (bool=true) then
                if Length(Intersection(l, result[r]))<>0 then
                    result[r]:=Concatenation(result[r], l);
                    bool:=false;
                fi;
            fi;
        od;
        if (bool=true) then
            Add(result, l);
        fi;
    od;

    return result;
end;


EliminatePhiSSA := function(code)
    local elim_map, phis, p, v, master_var;
    elim_map := tab();
    phis := Collect(code, phi);

    phis := StronglyConnectedComponents(List(phis, x->x.args));

    for p in phis do
        master_var := p[1];
        for v in Drop(p, 1) do
	        elim_map.(v.id) := master_var;
    	od;
    od;
    code := SubstVars(code, elim_map);
    code := SubstBottomUp(code, phi, e->e.args[1]);
    return code;
end;

