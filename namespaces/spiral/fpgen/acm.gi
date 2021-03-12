
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


CUnparser.bin_shl := (self,o,i,is) >> self.pinfix(o.args, "\<\<");
CUnparser.bin_shr := (self,o,i,is) >> self.pinfix(o.args, ">>");
CUnparser.fpmul := (self,o,i,is) >> Print("(",self(o.args[2],i,is), "*", self(o.args[3],i,is), " >> ", self(o.args[1],i,is), ")");
#TDouble.ctype:="int";
#CUnparser.TDouble := (self, vars, i, is) >> Print("int ", self.infix(vars, ", "));

synth_op := [add, sub, bin_shl, arith_shr, mul, neg]; #, addbin_shl, subbin_shl, bin_shlsub];

CheckSynthFormat := result -> true;

RunSynth := function(nums)
    local rfile, numst, result, msg;
    Constraint(ForAll(nums, IsInt));
    rfile := SysTmpName();
    numst := Concatenation(List(nums, x->Concat(StringInt(x), " ")));
    SYS_EXEC(ACM, " ", numst, " -gap > ", rfile);
    result := Try(ReadVal(rfile));
    if result[1] = false  then
        Error(ACM, " did not produce valid result, offending output kept in '", rfile, 
              "'\n error message = ", result[2]);
    fi;
    msg := CheckSynthFormat(result[2]);
    if msg <> true then 
        Error(ACM, " did not produce result in correct format, offending output kept in '", rfile, 
              "'\n ", msg);

    else
        return result[2];
    fi;
end;

Synth := function(x, destvars, targets)
    local rfile, result, cmd, vmap, destreg, numt, dest, dvar, C, exp, op;
    Constraint(ForAll(targets, IsInt));
    Constraint(Length(destvars) = Length(targets));

    numt := Length(targets);
    result := RunSynth(targets);
    
    C := [];
    vmap := Concatenation([x], destvars);
    for cmd in result do
        Constraint(IsList(cmd) and Length(cmd)=2 and IsInt(cmd[1]) and cmd[1] > 0);
	dest := cmd[1];
	exp := cmd[2];

	# when we hit a target (not intermediate result), vmap will already have an entry
	if dest <= numt then dvar := vmap[1+dest];
	else dvar := var.fresh_t("t", x.t); vmap[1+dest] := dvar;
	fi;
	
	if IsList(exp) then
	    op := synth_op[exp[1]];
	    if op in [add,sub] then
		exp := op(vmap[1+exp[2]], vmap[1+exp[3]]);
	    elif op in [bin_shl, arith_shr] then
		exp := op(vmap[1+exp[2]], exp[3]);
	    elif op = neg then
		exp := neg(vmap[1+exp[2]]);
	    fi;
	else 
	    Constraint(IsInt(exp));
	    exp := vmap[1+exp];
	fi;
	Add(C, assign(dvar, exp));
    od;
    return C;
end;

testSynth := function(targets)
    local C,D,x;
    x := var("x", Y.t.t);
    D := List([1..Length(targets)], i -> nth(Y, i-1));
    C := chain(Synth(x, D, targets));
    return Compile(C, SpiralDefaults);
end;

SynthChain := function(code) 
    local c, i, succ, mults, targets, dests, res, mult_code;
    Constraint(IsChain(code));
    res := [];
    for c in code.cmds do
        if ObjId(c)=assign then
	    succ := SuccLoc(c.loc);
	    mults := Filtered(succ, s->IsBound(s.def) and ObjId(s.def.exp) = fpmul 
		                       and IsValue(s.def.exp.args[2]) and s.def.exp.args[2].v<>0);
	    #PrintLine(mults);
	    if Length(mults)=0 then
		Add(res,c);
	    else 
		dests := List([1..Length(mults)], x -> var.fresh_t("d", TInt));
		#PrintLine(dests);
		targets := List(mults, x->x.def.exp.args[2].v);
		When(targets<>[],PrintLine(targets),0);
   	        # go ahead and synthesize
		mult_code := Synth(c.loc, dests, targets);
		#PrintLine(mult_code);
		#PrintLine("=====");
	        # plug in dests instead of the mults
		for i in [1..Length(mults)] do
	            mults[i].def.exp := arith_shr(dests[i], mults[i].def.exp.args[1]);
		od;
		Add(res, c);
		Append(res, mult_code);
	    fi;
	else
	    Add(res, c);
	fi;
    od;
    return chain(res);
end;

FPStrat := fracbits ->  Concatenation(BaseIndicesCS, [
    MarkDefUse, #
    (c, opts) -> CopyPropagate.fast(c, opts), # --12 kicks out vars used only once or never

    c -> FixedPointCode(c, 32, fracbits),
    Compile.declareVars
]);


SynStrat := fracbits -> Concatenation(BaseIndicesCS, [
    MarkDefUse, #
    (c, opts) -> CopyPropagate.fast(c, opts), # --12 kicks out vars used only once or never
    BinSplit, 
    c -> FixedPointCode(c, 32, fracbits),
    MarkDefUse, 
    SynthChain,
    Compile.declareVars
]);

# compiler.CodeSums := function ( bksize, sums )
#     local  code;
#     sums := BB(Gath(fId(Rows(sums)))*sums*Gath(fId(Cols(sums))));
#     code := _CodeSums(sums, Y, X);
#     code := RemoveAssignAcc(code);
#     code := BlockUnroll(code);
#     code := DeclareHidden(code);
#     code.dimensions := sums.dimensions;
#     return code;
# end;
# Add(RuleStrategy, x-> Gath(fId(Rows(x)))*x*Gath(fId(Cols(x))));

RewriteRules(code.RulesStrengthReduce, rec(
   arith_shrl := Rule([arith_shr, [bin_shl, @(1), @(2,Value)], @(3, Value)], e ->
       let(l := @(2).val.v, r:=@(3).val.v,
	   Cond(l > r, bin_shl(@(1).val, l-r), 
	        l < r, arith_shr(@(1).val, r-l),
		l = r, @(1).val))),
));
