
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# 2-operand intermediate representation.
# Here we define 2-op commands and 3-op -> 2-op
# conversion functions.
#
# This is relevant if one needs very low-level x86 representation,
# eg. to optimize register use and generate assembly code.
#

Class(assign_two_op, assign, rec(
    op_in := self >> Set(ArgsExp(self.exp)),
    op_out := self >> Set([]),
    op_inout := self >> Set([self.loc])
));

Class(assign_add, assign_two_op);
Class(assign_mul, assign_two_op);
Class(assign_sub, assign_two_op);
Class(assign_neg, assign_two_op);

# Examples: 
#   assign(t1, add(t1, t2)) == assign_add(t1, t2),
#   assign(t1, add(t2, t3)) == chain(assign(t1, t2), assign_add(t2, t3)).
#

Class(TwoOpCodeRuleSet, RuleSet);
RewriteRules(TwoOpCodeRuleSet, rec(
    add := Rule([assign, @(1), [add, @(2), @(3)]], e -> let(
	    s2:=Length(SuccLoc(@(2).val)), s3:=Length(SuccLoc(@(3).val)),
	When(Length(Filtered(List(@(2).val.succ,x->x.def.n),y->y>@(1).val.def.n))>0,
#	When(IsBound(@(2).val.live_out) or (s3 < s2 and not IsBound(@(3).val.live_out)),
	    chain(assign(@(1).val, @(3).val), # copy+destroy @(3)
		  assign_add(@(1).val, @(2).val)), 
	    chain(assign(@(1).val, @(2).val), # copy+destroy @(2)
		  assign_add(@(1).val, @(3).val))))),

    mul := Rule([assign, @(1), [mul, @(2), @(3)]], e -> let(
	    s2:=Length(SuccLoc(@(2).val)), s3:=Length(SuccLoc(@(3).val)),
	When(Length(Filtered(List(@(2).val.succ,x->x.def.n),y->y>@(1).val.def.n))>0,
#	When(IsBound(@(2).val.live_out) or (s3 <= s2 and not IsBound(@(3).val.live_out)),
	    chain(assign(@(1).val, @(3).val), # copy+destroy @(3)
		  assign_mul(@(1).val, @(2).val)),
	    chain(assign(@(1).val, @(2).val), # copy+destroy @(2)
		  assign_mul(@(1).val, @(3).val))))),

    sub := Rule([assign, @(1), [sub, @(2), @(3)]], e -> 
	    chain(assign(@(1).val, @(2).val),
		  assign_sub(@(1).val, @(3).val))),

    neg := Rule([assign, @(1), [neg, @(2)]], e -> 
            chain(assign(@(1).val, @(2).val), 
	          assign_neg(@(1).val)))
));

ElimRedundantCopyChain := function(code)
    local varmap, c, newcmds, succs;
    varmap := tab();
    newcmds := [];
    for c in code.cmds do
        c := SubstVars(c, varmap);
        if ObjId(c)=assign and ObjId(c.loc)=var and ObjId(c.exp)=var and
	   not IsBound(c.exp.live_out) and
	   Filtered(SuccLoc(c.exp), x->x<>c.loc and x.def.n > c.n)=[] then
	    varmap.(c.loc.id) := c.exp; 
	else
	    Add(newcmds, c);
	fi;
    od;
    return chain(newcmds);
end;

# Convert three-operand code -> two-operand code
#
TwoOpCode := function(code)
    # NOTE: write a better binsplit
    code := BinSplit(BinSplit(BinSplit(code)));
    MarkLiveness(code);
    code := TwoOpCodeRuleSet(code);
    code := FlattenCode(code);
    MarkLiveness(code);
#    code := ElimRedundantCopyChain(code);
#    code := CopyPropagate(code);
    return code;
end;

X86Code := function(code, numregs)
    local dims;
    if IsBound(code.dimensions) then dims := code.dimensions; fi;
    MarkLiveness(code);
    code := HashConsts(code, rec(declareConstants := true));
    code := TwoOpCode(code);
    MarkLiveness(code);
    code := RegAlloc(code, numregs, TDouble);
    code := DeclareHidden(code);
    if IsBound(dims) then code.dimensions := dims; fi;
    return code;
end;


# assumes <code> is a chain of assigns
TwoOpSSA := function(code)
    local varmap, c, newcmds, succs, a, b, sa, sb, so, v, overwrite, antidep;
    Constraint(IsChain(code) and ForAll(code.cmds, IsAssign));
    varmap := tab();
    newcmds := [];
    for c in code.cmds do
        c.exp := SubstVars(c.exp, varmap);

        if (ObjId(c.exp) in [add, sub, mul]) and ForAny(c.exp.args, IsVar) then
	    Constraint(Length(c.exp.args)=2);
	    [a,b] := c.exp.args;
	    sa := Filtered(SuccLoc(a), x -> x.def.n > c.n);
	    sb := Filtered(SuccLoc(b), x -> x.def.n > c.n);

	    if not IsVar(b) then overwrite := a;
	    elif not IsVar(a) then overwrite := b;
	    elif ObjId(c.exp)=sub then overwrite :=a;
	    elif IsBound(b.live_out) or 
		 ((Length(sa) < Length(sb)) and not IsBound(a.live_out)) then
		overwrite := a;
	    else 
		overwrite := b;
	    fi;
	    antidep := false;
	    so := When(overwrite=a, sa, sb);
	    if so <> [] then
		v := overwrite.clone();
		v.succ := so;
		overwrite.succ := Difference(overwrite.succ, so);
		Add(newcmds, assign(v, overwrite));
		varmap.(overwrite.id) := v;
		antidep := v;
	    fi;
	    if antidep<>false then c.antidep := [antidep]; fi;
	    Add(newcmds, c);

	else
	    Add(newcmds, c);
	fi;
    od;
    return chain(newcmds);
end;

Class(DirectTwoOpMapping, RuleSet);
RewriteRules(DirectTwoOpMapping, rec(
    add := Rule([assign, @(1), [add, @(2), @(3)]], e -> 
	Cond(@(1).val = @(2).val, assign_add(@(2).val, @(3).val), 
	     @(1).val = @(3).val, assign_add(@(3).val, @(2).val),
	     Error("Statement ", e, " is not 2-op ready, must be a=a+b, or b=a+b"))),

    mul := Rule([assign, @(1), [mul, @(2), @(3)]], e -> 
	Cond(@(1).val = @(2).val, assign_mul(@(2).val, @(3).val), 
	     @(1).val = @(3).val, assign_mul(@(3).val, @(2).val),
	     Error("Statement ", e, " is not 2-op ready, must be a=a*b, or b=a*b"))),

    sub := Rule([assign, @(1), [sub, @(2), @(3)]], e -> 
	Cond(@(1).val = @(2).val, assign_sub(@(2).val, @(3).val), 
	     Error("Statement ", e, " is not 2-op ready, must be a=a-b")))
));
# def b1
# a1: consume b1
# a2: consume b1

# def b1
# cpy b1->b2

# a1: consume b1
# a2: consume b2
