# -*- Mode: shell-script -*- 

##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# NOTE:  INJECTION_D for T_REC
#
#------------------------------
GlobalPackage(spiral.delay);
#------------------------------

PEV := PartialEval;
GapVar := () -> VariableStore.newVar();
VarGenerator := base -> WithBases(VariableStore, rec(_base := base));
Type := x -> When(IsDelay(x), BagType(Child(x, 1)), BagType(x));
Addr := x -> When(IsDelay(x), BagAddr(Child(x, 1)), BagAddr(x));

# DetachFunc(<func>) - removes the link to enclosing (runtime) environment  
#
# Higher-order functions (i.e. functions created at runtime inside other 
# functions) have link to the entire runtime stack. This space overhead
# may become very large. This link is necessary so that created inner function
# can create variables from the outer (parent) function.
#
# The solution is to inject variable values using Subst, and then detach 
# the function.
#
# Example:
#   Regular higher order function (will keep a link runtime stack)
#       adder := val -> (x -> x+val)
#
#   Detached function
#       adder := val -> DetachFunc(Subst(x -> x + $val));
#
# Note: DetachFunc is destructive.
#
DocumentVariable(DetachFunc);

#--------------------------------------------------------------------------
IsBoundName := name -> When(IsString(name), 
    Eval(Delay(IsBound(x), x=>DelayedValueOf(name))),
    Error("Usage: IsBoundName( <string> )"));

UnbindName := name -> When(IsString(name), 
    Eval(Delay(Unbind(x), x=>DelayedValueOf(name))),
    Error("Usage: UnbindName( <string> )"));

# ----- Constructors ------------------------------------------------------
Funccall := (nam, args) -> BagList(T_FUNCCALL, Concatenation([nam], args));

FuncDef := func -> Let(f=>$func, Checked(Type(f) in [T_FUNCTION, T_METHOD],
                     D($(Child(f,1)))));

IsPureFunc := f -> IsFunc(f) and NumLocals(f)=0 and Type(Child(f, 1))=T_RETURN;

PureFuncDef := f -> D($(Child(Child(f, 1), 1)));

FuncArgs := func -> Let(f=>$func, Checked(Type(f) in [T_FUNCTION, T_METHOD],
    Let(ch => Children(f), nargs => NumArgs(f),
	List(SubList(ch, 3, 2+nargs),  c -> D($c)))));

FuncLocals := func -> Checked(Type(func) in [T_FUNCTION, T_METHOD],
    Let(ch => Children(func), nargs => NumArgs(func),
	List(SubList(ch, 3+nargs, Length(ch)-1),  c -> D($c))));

SimpleSubst := "to be defined later";
CallInline := call -> Checked(Type(call) = T_FUNCCALL, 
    Let(f=>Eval(call[1]), 
	SimpleSubst(FuncDef(f), 
	    x -> x in FuncArgs(f),
	    x -> call[When(Type(f)=T_FUNCTION, 1, 0) + Position(FuncArgs(f), x)])));

# ----- 
IsGapVar := x -> Type(x) in [T_VAR, T_VARAUTO];

Enum := UnevalArgs(
    function(arg)
       local id, var, func;
       id := Eval(arg[1]);
       func := Eval(arg[2]);
       arg := Drop(arg, 2);
       for var in arg do 
         Constraint(IsGapVar(var));
	 Assign(var, func(id));
	 id := id + 1;
       od;
    end
);

Syms := UnevalArgs(
    arg -> DoForAll(arg, x -> Checked(IsGapVar(x),
	                              Assign(x, x))));

Funcs := UnevalArgs(
    arg -> DoForAll(arg, x -> Checked(IsGapVar(x),
	                              Assign(x, arg -> Funccall(x, arg)))));

SetProp := function(var, fld, val)
    Props(var).(fld) := val; return var;
end;

SetProps := UnevalArgs(arg -> 
    let(attr:=Eval(arg[1]), val:=Eval(arg[2]), vars:=SubList(arg, 3),
	DoForAll(vars, x -> SetProp(x, attr, val))));
    
# ----- Substitution

_SimpleSubst := function(expr, avoid_pred, subst_pred, to, visited, recurs_to) 
    local addr;
    if avoid_pred(expr) then return expr; fi;
    addr := Addr(expr);
#    if addr in visited then return expr; fi;

    if subst_pred(expr) then 
	return When(recurs_to, _SimpleSubst(to(expr), avoid_pred, subst_pred, 
		                            to, visited, recurs_to),
	                       to(expr));
    else 
	;#AddSet(visited, addr);
    fi;

    return 
    Cond( Type(expr) in [T_VAR, T_VARAUTO, T_FUNCTION, T_METHOD, T_STRING], expr,
	  IsList(expr), Map(expr, x -> _SimpleSubst(x, avoid_pred, subst_pred, 
		                                    to, visited, recurs_to)),
	  IsRec(expr),  Map(expr, x -> _SimpleSubst(x, avoid_pred, subst_pred, 
		                                    to, visited, recurs_to)),
	  expr);
end;

isOpRec := expr -> IsDomain(expr) or 
                   (IsRec(expr) and IsBound(expr.operations) and 
	              (expr.operations = OpsOps or
	               expr.operations = ClassOps));

SimpleSubst := (expr,pred,to) -> 
    _SimpleSubst(expr, isOpRec,
	               When(IsFunc(pred), pred, x->x=pred), 
	               When(IsFunc(to),   to,   x->to), 
                       Set([]), 
		       false);

SimpleSubst2 := (expr,pred,to) -> 
    _SimpleSubst(expr, isOpRec,
	               When(IsFunc(pred), pred, x->x=pred), 
	               When(IsFunc(to),   to,   x->to), 
                       Set([]), 
		       true);


# ----- Sampled values (stored in variable's proprty list Props(var)

TakeSample := z -> SimpleSubst(z, x -> Type(x)=T_VAR and IsBound(Props(x).sample),
                                  x -> Props(x).sample);
#SampledVar := function(sample) 
#    local v;
#    v := VariableStore.newVar();
#    Props(v).sample := Eval(TakeSample(sample));
#    return v;
#end;

# -----
_DelaySubst := (expr, sel, to, avoid) ->
    Cond( avoid(expr), 
	      expr, 
	  Type(expr) in [T_VAR, T_VARAUTO, T_STRING],
	      When(sel(expr), to(expr), expr),
	  IsList(expr) or IsRec(expr), 
	      let(r := Map(expr, x -> _DelaySubst(x, sel, to, avoid)),
		  When(sel(r), to(r), r)),
	  expr 
    );

_DelaySubst2 := (expr, sel, to, avoid) ->
    Cond( avoid(expr), 
	      expr, 
	  Type(expr) in [T_VAR, T_VARAUTO, T_STRING],
	      When(sel(expr), to(expr), expr),
	  IsList(expr) or IsRec(expr), 
	      When(sel(expr), Map(to(expr), x->_DelaySubst2(x, sel, to, avoid)), 
		              Map(expr,     x->_DelaySubst2(x, sel, to, avoid))), 
	  expr 
    );

DelaySubst  := (expr, sel, to) -> _DelaySubst(expr, sel, to, isOpRec);

DelaySubst2 := (expr, sel, to) -> _DelaySubst2(expr, sel, to, isOpRec);

FunccallsDelay := function(expr)
    local t, r;
    t := Type(expr);
    if t in [T_VAR, T_VARAUTO] then 
	return expr;
    elif t in [T_REC, T_STRING, T_INT, T_DOUBLE, T_CYC, T_RAT] then 
	return Eval(expr);
    elif IsList(expr) then 
	r := Map(expr, FunccallsDelay);
	if IsDelay(r) and Type(r) <> T_FUNCCALL then 
	    return Funccall(DelayedValueOf(TYPES[1+Type(r)]), r);
	else 
	    return r;
	fi;
    else
	return expr;
    fi;
end;
