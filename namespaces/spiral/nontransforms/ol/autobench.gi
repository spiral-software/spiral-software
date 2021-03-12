
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_prepVars := function(vars, vargen, pfx) 
    local v, res, newpfx, newv;
    res := []; newpfx := Concat("p", pfx);
    for v in vars do
        if IsArrayT(v.t) then
	    newv := vargen.next(TPtr(v.t.t), newpfx);
	    Add(res, [ v, newv, zallocate(newv, v.t) ]);
	else
	    Add(res, [ v, v, skip() ]);
	fi;
    od;
    return res;
end;

GenerateMultiIOBench := function(t,code, domain, range, opts)
    local vargen, code, inp, out, inout, args, allocs;
    vargen := VarGenNumeric();
    inp := List(domain, t -> vargen.next(t, "_in"));
    inp := _prepVars(inp, vargen, "_in");

    out := List(range, t -> vargen.next(t, "_out"));
    out := _prepVars(out, vargen, "_out");
    
    inout := Concatenation(out, inp);
    args := List(inout, x->x[2]);
    allocs := List(inout, x->x[3]);
    
    code := program(
	decl(args, 
	    chain(
		SubstTopDown(code, program, x->chain(x.cmds)),
		func(TVoid, "init", [], 
		    chain(allocs,
		    ApplyFunc(call, [var(opts.subInitName)]))),
		func(TVoid, "transform", [When(IsBound(opts.subParams), opts.subParams, [])],
		    ApplyFunc(call, Concatenation([var(opts.subName)], When(IsBound(opts.subParams), opts.subParams, []),args))))));
    return code;
end;

# Generate Bench does allocation for a program (a function and its initialization).
# It indirects both the init and the transform, does memory allocation for parameters
# in the new init and uses them to call the function in the new body.
#
#     init(){...} 
#     transform(...){...}
# 
# will be thus be renamed to:
#
#      <opts.subInitName>(){...}
#      <opts.subName>(...){...}
#
# and the following functions will be added:
#
#      init(){params=malloc();<opts.subInitName>();}
#      transform(){<opts.subName>(params);}
#
# GenerateBench(<t>,<code>,<opts>)
# <t> is the sums corresponding to the program <code> compiled with options <opts> 
GenerateBench := (t,code,opts) -> 
    GenerateMultiIOBench(t, code, t.dmn(), t.rng(), opts);
