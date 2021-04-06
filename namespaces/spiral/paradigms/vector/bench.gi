
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_md := function(lst)
    local i, str;
    str:="";
    if Length(lst) = 0  then
        return str;
    fi;
    str := Concat(str, StringInt(lst[1]));
    for i in [ 2 .. Length(lst) ]  do
        str := Concat(str, "x", StringInt(lst[i]));
    od;
    return str;
end;

#   doSimdDft argument parser
#
#   spiral> doSimdDft();
#   [ [ 4 ], SSE_2x64f, rec(
#      svct := true,
#      oddSizes := false ) ]
#   spiral> doSimdDft(1);
#   [ [ 4, 8 ], SSE_2x64f, rec(
#      svct := true,
#      oddSizes := false ) ]
#   spiral> doSimdDft([4,6]);
#   [ [ 4, 6 ], SSE_2x64f, rec(
#      svct := true,
#      oddSizes := true ) ]
#   spiral> doSimdDft(SSE_4x32f);
#   [ [ 16 ], SSE_4x32f, rec(
#      svct := true,
#      oddSizes := false ) ]
#   spiral> doSimdDft(1, SSE_4x32f);
#   [ [ 16, 32 ], SSE_4x32f, rec(
#      svct := true,
#      oddSizes := false ) ]
#   spiral> doSimdDft([4,6], SSE_4x32f);
#   [ [ 4, 6 ], SSE_4x32f, rec(
#      svct := true,
#      oddSizes := true ) ]
#   spiral> doSimdDft([4,6], SSE_4x32f, rec(svct:=false, oddSizes:=true));
#   [ [ 4, 6 ], SSE_4x32f, rec(
#      svct := false,
#      oddSizes := true ) ]

_parse := function(arglist, multiplier)
    local isa, n, sizes, optrec;

    isa := SIMD_ISA_DB.active()[1];
    optrec := rec(svct := true, oddSizes :=false);

    if Length(arglist) = 0 then
        n := 0;
    elif Length(arglist) = 1 then
        if IsInt(arglist[1]) then
            n := arglist[1];
        elif IsList(arglist[1]) then
            n := arglist[1];
            optrec := rec(svct := true, oddSizes :=true);
        else
            n := 0;
            isa := arglist[1];
        fi;
    elif Length(arglist) = 2 then
        n := arglist[1];
        isa := arglist[2];
        if IsList(arglist[1]) then
            optrec := rec(svct := true, oddSizes :=true);
        fi;
    else
        n := arglist[1];
        isa := arglist[2];
        optrec := arglist[3];
    fi;

    sizes := When(IsList(n), n, List([0..n], i->multiplier*isa.v^2*2^i));

    return [sizes, isa, optrec];
end;

codeSimdDft := function(arg)
    local sizes, opts, isa, res, t,rt, s, c, optrec;
    [sizes, isa, optrec] := _parse(arg, 1);
    opts := SIMDGlobals.getOpts(isa, optrec);

    t := ComplexT(DFT(sizes[1]), opts).withTags(opts.tags);
    rt := RandomRuleTree(t, opts);
    s := SumsRuleTreeOpts(rt, opts);
    c := CodeSumsOpts(s, opts);
    return rec(opts:=opts,t:=t,rt:=rt,s:=s,c:=c);
end;


_doSimd := function(sizes, isa, opts, dpr, name, trafo, mflops, stringint)
    local results, res, i;

    results := [];
    if not IsBound(dpr.hashTable) then dpr.hashTable := HashTableDP(); fi;

    Print(name, ": running ", isa, ": ", sizes, "...\n");
    for i in sizes do
      res := TimedAction(DP(trafo(i), dpr, opts));
      if Length(res[1]) > 0 then
        Add(results, res[1]);
        ExportCodeRuleTree(res[1][1].ruletree,
            Concat(name, "_", isa.name, "_", stringint(i)),
            CopyFields(opts, rec(verbosity := 0, fileinfo := rec(
                cycles := res[1][1].measured,
                flops := mflops(i),
                file := Concat(name, "_", isa.name, "_", stringint(i)),
                algorithm := res[1][1].ruletree
        ))));

        if opts.verify then
            _seqPerfStatsMflopsAcc(Concat(name, "_", isa.name, ".txt"),
                i, mflops(i), res[1][1].measured, res[2], IntDouble(d_log(1e-16+CVerifyRT(res[1][1].ruletree, opts))/d_log(10)));
        else
            _seqPerfStatsMflops(Concat(name, "_", isa.name, ".txt"),
                i, mflops(i), res[1][1].measured, res[2]);
        fi;
        HashSave(dpr.hashTable, Concat(name, "_", isa.name, ".hash"));
      fi;
    od;

    return results;
end;


#F doSimdDft(sizes, isa, optrec)
doSimdDft := function(arg)
    local sizes, opts, dpr, isa, optrec, trafos, dpbench, direction;

    [sizes, isa, optrec] := _parse(arg, 1);
	
	direction := When(IsBound(optrec.transInverse) and optrec.transInverse, 1, -1);

    opts := SIMDGlobals.getOpts(isa, optrec);
    opts.benchTransforms := List(sizes, k -> ComplexT(DFT(k, direction), opts).withTags(opts.tags));

    if not IsBound(optrec.globalUnrolling) and ForAny(sizes, i -> not IsInt(i/isa.v^2)) then
        optrec.globalUnrolling := 16384;
    fi;
    dpr  := rec(verbosity := 0, timeBaseCases:=true);
    dpbench := spiral.libgen.DPBench(rec((Concat(isa.name, When(opts.vector.conf.interleavedComplex, "_ic", "_sc"))) := opts), dpr);
    if IsBound(optrec.verify) then dpbench.matrixVerify := optrec.verify; fi;
	
	dpbench.sizes := sizes;

    return dpbench;
end;


#F doSimdWht(sizes, isa, optrec)
doSimdWht := function(arg)
    local sizes, isa, opts, optrec, defrec, dpr, dpbench;

    [sizes, isa, optrec] := _parse(arg, 1);

    defrec := rec(
        oddSizes := false,
        svct := true,
        stdTTensor := true,
        tsplPFA := false,
        splitL:=false,
        realVect := true,
        cplxVect := false,
        useConj := false,
    );
    opts := SIMDGlobals.getOpts(isa, CopyFields(optrec, defrec));
    opts.benchTransforms := List(sizes, k -> WHT(Log2Int(k)).withTags(opts.tags));
    #opts.benchTransforms := List(sizes, k -> GT(WHT(Log2Int(k)), GTPar, GTPar, [4]).withTags(opts.tags));
    #opts.benchTransforms := List(sizes, k -> GT(WHT(Log2Int(k)), GTPar, GTPar, [8]).withTags(opts.tags));
    dpr  := rec(verbosity := 0, timeBaseCases:=true);

    dpbench := spiral.libgen.DPBench(rec((isa.name) := opts), dpr);
    if IsBound(optrec.verify) then dpbench.matrixVerify := optrec.verify; fi;
	
	dpbench.sizes := sizes;

    return dpbench;
end;



doSimdMddft := function(arg)
    local sizes, opts, tags, dpr, isa, defrec, optrec, dpbench, direction;

    [sizes, isa, optrec] := _parse(arg, 1);
	
	direction := When(IsBound(optrec.transInverse) and optrec.transInverse, 1, -1);

    # for cannot do larger sizes at this point
    defrec := rec(
        realVect := true,
        cplxVect := false,
        useConj := false,
    );
    opts := SIMDGlobals.getOpts(isa, CopyFields(optrec, defrec));

    opts.benchTransforms := List(sizes, k -> ComplexT(MDDFT(k, direction), opts).withTags(opts.tags));
    dpr  := rec(verbosity := 0, timeBaseCases:=true);

    dpbench := spiral.libgen.DPBench(rec((Concat(isa.name, When(opts.vector.conf.interleavedComplex, "_ic", "_sc"))) := opts), dpr);
    if IsBound(optrec.verify) then dpbench.matrixVerify := optrec.verify; fi;
	
	dpbench.sizes := sizes;

    return dpbench;
end;


codeSimdSymDFT := function(arg)
    local sizes, opts, dpr, isa, defrec, optrec, t, r, s, c, ttype;

    ttype := arg[1];
    [sizes, isa, optrec] := _parse(Drop(arg, 1), 2);
    defrec := rec(oddSizes := false, svct := true, splitL:=false, realVect := true,
                  cplxVect := false, useConj := false, globalUnrolling := 1024);
    opts := SIMDGlobals.getOpts(isa, CopyFields(optrec, defrec));
    t := ttype(sizes[1]).withTags(opts.tags);
    r := RandomRuleTree(t, opts);
    s := SumsRuleTree(r, opts);
    c := CodeSums(s, opts);
    return rec(opts:=opts, t:=t, rt:=r, s:=s, c:=c);
end;


doSimdSymDFT := function(arg)
    local sizes, opts, dpr, isa, optrec, results, defrec, ur, ttype, dpbench;

    ttype := arg[1];
    [sizes, isa, optrec] := _parse(Drop(arg, 1), 2);

    defrec := rec(
        oddSizes := false,
        svct := true,
        splitL:=false,
        realVect := true,
        cplxVect := false,
        useConj := false,
        globalUnrolling := 1024
    );
    ur := rec(
        globalUnrollingMin := 1024,
        globalUnrollingMax := 1024
    );
    opts := SIMDGlobals.getOpts(isa, CopyFields(defrec, optrec));
    opts.benchTransforms := List(sizes, k -> ttype(k).withTags(Cond(IsBound(optrec.cplxVect) and optrec.cplxVect, opts.cxtags, opts.tags)));
    if not IsBound(optrec.globalUnrolling) and ForAny(sizes, i -> not IsInt(i/isa.v^2)) then
        optrec.globalUnrolling := 16384;
    fi;
    dpr  := rec(verbosity := 0, timeBaseCases:=true);

    dpbench := spiral.libgen.DPBench(rec((isa.name) := opts), dpr);
    if IsBound(optrec.verify) then dpbench.matrixVerify := optrec.verify; fi;
	
	dpbench.sizes := sizes;

    return dpbench;

#    results := _doSimd(sizes, isa, opts, dpr, arg[1], k -> ttype(k).withTags(opts.tags), i->2*i*d_log(i)/d_log(2) + i, StringInt);
#    return rec(opts:=opts, results:=results);
end;



doSimdSymMDDFT := function(arg)
    local sizes, isa, opts, optrec, defrec, dpr, dpbench, ttype;

    ttype := arg[1];
    [sizes, isa, optrec] := _parse(Drop(arg, 1), 2);

    defrec := rec(svct := true, splitL := false, stdTTensor := true, tsplRader:=false, tsplBluestein:=false,
        tsplPFA:=false, oddSizes:=false, interleavedComplex := false, pushTag := false, flipIxA := true);
    opts := SIMDGlobals.getOpts(isa, CopyFields(optrec, defrec));

    opts.benchTransforms := List(sizes, k -> TTensor(ttype(k), ttype(k)).withTags(opts.tags));
    dpr  := rec(verbosity := 0, timeBaseCases:=true);

    dpbench := spiral.libgen.DPBench(rec((isa.name) := opts), dpr);
    if IsBound(optrec.verify) then dpbench.matrixVerify := optrec.verify; fi;
	
	dpbench.sizes := sizes;

    return dpbench;
end;


doSimdMdConv := function(arg)
    local sizes, isa, opts, optrec, defrec, dpbench, transforms;

    [sizes, isa, optrec] := _parse(arg, 1);
	
    defrec := rec();
	
    opts := SIMDGlobals.getOpts(isa, CopyFields(optrec, defrec));
	
	transforms := Flat(List(sizes, n -> let(tt := TRConv2D(ImageVar([n, n])).withTags(opts.tags), [tt, tt.forwardTransform()])));
	
	dpbench := spiral.libgen.DPBench.build(transforms, opts);
	
	dpbench.sizes := sizes;

    return dpbench;
end;



