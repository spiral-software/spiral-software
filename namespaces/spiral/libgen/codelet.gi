
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


@_RecursBase := NewMakeRule(
    rec(applicable       := nt -> true,
	forTransposition := false,
	children         := nt -> [[ ]],
	apply := (nt,ch,nonterms) -> RecursStep(nt)),
        "@_RecursBase", @); 

#F CreateRecursBaseHash(<smallHash>) 
#F
#F Returns new hash table with T->@_RecursBase(T) ruletrees for each T in <smallHash>
#F
CreateRecursBaseHash := function(smallHash)
    local e, f, t, basehash;
    basehash := HashTableDP();
    for e in smallHash.entries do
	if e<>[] then
            for f in e do
              if f.data<>[] then 
                  HashAdd(basehash, f.key, When(IsBound(f.data[1].measuredGP),[ rec(ruletree := @_RecursBase(f.key),measured := f.data[1].measuredGP) ],[ rec(ruletree := @_RecursBase(f.key))]));
              fi;
            od;
	fi;
    od;
    return basehash;
end;

Class(Codelet, BaseContainer);

_lookupFormula := function(smallHash, transform, opts) 
    local lkup;
    lkup := HashLookup(smallHash, transform);
    if lkup = false 
	then Error("Transform ", transform, " missing in opts.libgen.baseBench");
    else
	if not IsBound(lkup[1].sums) then
	    VPrintLine(opts.verbosity, "Sigma-splizing ", lkup[1].ruletree.node, "...");
	    # NOTE: generateComplexCode
	    lkup[1].sums := SumsRuleTree(lkup[1].ruletree, CopyFields(opts, rec(generateComplexCode := true)));
	fi;
	return Copy(lkup[1].sums);
    fi;
end;

Class(RecursStepCall, BaseOperation, rec(  
    new := (self, dimensions, func, bindings) >> SPL(WithBases(self, 
        rec(dimensions := dimensions,  func := func, bindings := bindings))),

    dims := self >> self.dimensions,
    rChildren := self >> [self.dimensions, self.func, self.bindings], 
    rSetChild := rSetChildFields("dimensions", "func", "bindings"),
    numops := self >> 10^300,

    print := (self, i, is) >> Print(
        self.name, "(", self.dimensions, ", \"", self.func, "\", ", self.bindings, ")")
));

Class(RecursStepTerm, RuleSet);
RewriteRules(RecursStepTerm, rec(

    RecursStepToCall := Rule(RecursStep, 
        meth(e, cx)
            local rs, opts, bindings, func, shape, lkup, params, clrec, smallhash, preimpl;
            opts := cx.opts;
            e := e.child(1);
	    e := SubstTopDown(e, BB, x->x.child(1));
            shape := CodeletShape(e);
            func := CodeletName(shape);
            rs := MkCodelet(e);
            params := CodeletParams(rs);
            bindings := Zip2(params, CodeletParams(e));

            lkup := HashLookup(opts.libgen.codeletTab, shape);
            smallhash := opts.libgen.baseBench.exp.bases.hashTable;
#should it be NoPull(_lookupFormula(...))??
            if lkup = false then
                preimpl := Copy(rs);
                rs := SubstBottomUpRules(rs, [
                    [RTWrap, r -> _lookupFormula(smallhash, r.rt.node, opts) ],
                    [@.cond(IsNonTerminal), r -> _lookupFormula(smallhash, r, opts) ]]);
                rs := ApplyStrategy(rs, opts.libgen.terminateStrategy, UntilDone, opts);
                clrec := rec(name := func, shape := shape, sums := rs, preimpl := preimpl, params := params); 
                HashAdd(opts.libgen.codeletTab, shape, clrec);
                HashAdd(opts.libgen.codeletTab, func, clrec);
            fi;
            return RecursStepCall(rs.dimensions, func, bindings);
        end)
));


CreateCodeletHashTable := () -> 
    HashTable((val,size)->InternalHash(CodeletName(val)));

CompileCodelets := function(c, opts)
    local calls, call, clet_code, lkup, res, funcs,outputs,inputs, u, t;
    calls := Collect(c, RecursStepCall);
    res := []; funcs := Set([]);
    for call in calls do
        if not call.func in funcs then
	    Add(funcs, call.func);
	    lkup := HashLookup(opts.libgen.codeletTab, call.func);
	    if lkup = false then Error(call.func, " is missing from codelet hash. ", 
                                                  "Run SumsRuleTree again?"); fi;
	    if not IsBound(lkup.code) then
		VPrintLine(opts.verbosity, "Compiling ", call.func, "...");
                t := TPtr(Cond(opts.generateComplexCode, TComplex, TReal));
                outputs:=StripList(List([1..Length(Flat([lkup.sums.dims()[1]]))],x->var(Concat("Y",When(x>1,String(x),"")), t)));
                inputs:=StripList(List([1..Length(Flat([lkup.sums.dims()[2]]))],x->var(Concat("X",When(x>1,String(x),"")), t)));
                u:=opts.libgen.basesUnrolling;
                if IsBound(lkup.unrolling) then
                    u:=lkup.unrolling;
                fi;
		lkup.code := opts.codegen(Codelet(lkup.sums), outputs, inputs, 
		    CopyFields(opts, rec(globalUnrolling := u)));
	    fi;
            #if Collect(lkup.code, errExp) <> [] then 
            #    lkup.badcode := lkup.code; Unbind(lkup.code); Error("errExp"); fi;
	    Add(res, lkup);
	fi;
    od;
    return res;
end;
 
