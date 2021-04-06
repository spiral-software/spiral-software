
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


X := var("X", TPtr(TReal));
Y := var("Y", TPtr(TReal));
X1 := var("X1", TPtr(TReal));
Y1 := var("Y1", TPtr(TReal));
XY1 := var("XY1", TPtr(TReal));

# defined in cgstrat.gi
Declare(ApplyCodegenStrat);

RemoveAssignAcc := code -> SubstTopDownRulesNR(code, rec(
    assign_acc_toassign := Rule(assign_acc, e -> assign(Copy(e.loc), e.loc+e.exp)),
    assign_donothing := Rule(assign, e -> e), 
));


#F CodeSums(<sums>, <opts>)
#F    Creates code for Sigma-SPL <sums>.
#F    Sigma-SPL rewrite rules are not applied.
#F    Compilation proceeds according to <opts>.compileStrategy
#F

CodeSums := function(sums, opts)
    local X, Y, code;

        if IsList(sums.dims()[2]) then
            X := List([1..Length(sums.dims()[2])],
                x -> var(Concatenation("X",String(x)), TPtr(TReal)));
        else
            X := Cond(IsBound(opts.X), opts.X, var("X", TPtr(TReal)));
        fi;

        if IsList(sums.dims()[1]) then
            Y := List([1..Length(sums.dims()[1])],
                x -> var(Concatenation("Y",String(x)), TPtr(TReal)));
        else
            Y := Cond(
                opts.inplace, X,
                IsBound(opts.Y), opts.Y, 
                var("Y", TPtr(TReal)));
        fi;
    
        code := opts.codegen(Formula(sums), Y, X, opts);
        code.ruletree := Cond(IsBound(sums.ruletree), sums.ruletree, rec());
        return code;
end;

#F See CodeSums
#F
CodeSumsOpts := CodeSums;


#F CodeRuleTree(<rt>, <opts>)
#F
#F <opts> flags used:
#F   opts.formulaStrategies.sigmaSpl    Sigma-SPL rewriting strategy
#F   opts.formulaStrategies.rc          RC(.) rewriting strategy
#F   opts.generateComplexCode == bool   if set to false, then RC rewriting strategy is applied
#F
CodeRuleTree := function(rt, opts)
    local sums, code;
    sums := SumsRuleTree(rt, opts);
    code := CodeSums(sums, opts);
    return code;
end;

#F See CodeRuleTree
#F
CodeRuleTreeOpts := CodeRuleTree;

#F RealSums(<sums>)
#F    Convert complex Sigma-SPL formula to a real formula.
#F
RealSums := sums -> StandardSumsRules(RC(sums));


#F CodeSPL(<unroll>, <spl>, opts)
#F
CodeSPL := function(spl, opts)
    local sums;
    sums := SumsSPL(spl, opts);
    sums := ApplyStrategy(sums, opts.formulaStrategies.sigmaSpl, UntilDone, opts);
    sums := ApplyStrategy(sums, opts.formulaStrategies.preRC, UntilDone, opts);
    if not spl.isReal() and not opts.generateComplexCode then
        sums := ApplyStrategy(RC(sums), opts.formulaStrategies.rc, UntilDone, opts); fi;
    sums := ApplyStrategy(sums, opts.formulaStrategies.postProcess, UntilDone, opts);
    return CodeSums(sums, opts);
end;


#F PrintCode(<funcname>, <code>, <opts>)
#F    Prints the code using opts.unparser
#F
PrintCode := (funcname, code, opts) -> opts.unparser.gen(funcname, code, opts);

_ExportCodeRuleTree := function(ruletree, file, funcname, opts)
    local code;
    Constraint(IsRuleTree(ruletree));
    if opts.verbosity > 0 then Print("Generating code...\n"); fi;
    code := CodeRuleTree(ruletree, opts);
    PrintTo(file, opts.unparser.gen(funcname, code, opts));
end;

#F ExportCodeRuleTree(<ruletree>, <funcname>, <opts>)
#F    Generates and exports C code to a file for <ruletree>.
#F    The C transform function will have the name <funcname>.
#F    Code will be saved in <funcname>.c
#F
ExportCodeRuleTree := (ruletree, funcname, opts) ->
    _ExportCodeRuleTree(ruletree, Concat(funcname, ".c"), funcname, opts);

#F PrintCodeRuleTree(<ruletree>, <opts>)
#F    Generates C code and prints it out.
#F
PrintCodeRuleTree := (ruletree, opts) ->
    _ExportCodeRuleTree(ruletree, "*stdout*", "sub", opts);

#F ImplementRuleTree(<ruletree>, <file>, <opts>)
#F
ImplementRuleTree := (ruletree, file, opts) ->
    _ExportCodeRuleTree(ruletree, file,
    When(IsBound(opts.subName), opts.subName, ruletree.node.name), opts);

#F VerifyMatrixRuleTree(<ruletree>, <opts>)
#F
VerifyMatrixRuleTree := function(ruletree, opts)
    local code, mat;
    Constraint(IsRuleTree(ruletree));
    if opts.verbosity > 0 then Print("Generating code...\n"); fi;
    code := CodeRuleTree(ruletree, opts);
    if opts.verbosity > 0 then Print("Computing reference matrix...\n"); fi;
    mat := When(ruletree.node.isReal() or opts.dataType = "complex" or opts.generateComplexCode,
            MatSPL(ruletree.node),
        RCMatCyc(MatSPL(ruletree.node)));
    if opts.verbosity > 0 then Print("Running code and computing the norm...\n"); fi;
    return InfinityNormMat(CMatrix(code, opts) - mat);
end;

#F VerifyMatrixCode(<code>, <definition-matrix>, <opts>)
#F
VerifyMatrixCode := (code, def_matrix, opts) -> Checked(IsCommand(code),
    InfinityNormMat(CMatrix(code, opts) - def_matrix)
);


FailedTrees := [];
InaccurateTrees := [];

CMeasureRuleTree := function(rt, opts)
    local res, c, mfunc, tol;

    c := CodeRuleTree(rt, opts);

    mfunc := When(IsBound(opts.profile) and IsBound(opts.profile.meas), opts.profile.meas, CMeasure);

    if opts.faultTolerant then
		res := Try(mfunc(c, opts));
    else
		res := [true, mfunc(c, opts)];
    fi;

    if res[1]=false then
        Add(FailedTrees, rt);
        return 1e20;
    else
        if IsBound(opts.verifyDP) and opts.verifyDP then
            tol := VerifyMatrixRuleTree(rt, opts);
            if tol > opts.verifyTolerance then
                Add(InaccurateTrees, rec(rt:=rt, opts := opts, tol :=tol));
            fi;
        fi;
        return res[2];
    fi;
end;
