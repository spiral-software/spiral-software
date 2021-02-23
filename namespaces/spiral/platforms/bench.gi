
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_svctSizes := (N, p, v) -> Filtered([1..N], i->ForAll(Factors(i), j->j<=p) and IsInt(i/v^2));
_defaultSizes := (func, default) -> ((arg) -> When(Length(arg) = 1, func(When(IsList(arg[1]), arg[1], [arg[1]])), func(default)));

_parse := function(arglist)
    local isa, n, sizes, optrec;

    isa := "f64re";
    optrec := rec(interleavedComplex := true);

    if Length(arglist) = 0 then
        n := 1;
    elif Length(arglist) = 1 then
        if IsInt(arglist[1]) then
            n := arglist[1];
        elif IsList(arglist[1]) then
            n := arglist[1];
        else
            n := 0;
            isa := arglist[1];
        fi;
    elif Length(arglist) = 2 then
        n := arglist[1];
        isa := arglist[2];
    else
        n := arglist[1];
        isa := arglist[2];
        optrec := arglist[3];
    fi;

    sizes := When(IsList(n), n, List([1..n], i->2^i));

    return [sizes, isa, optrec];
end;


doDft := function(arg)
    local sizes, opts, dpr, isa, trafos, t, optrec, dpbench, defaults;

    [sizes, isa, optrec] := _parse(arg);

    defaults := rec(
        breakdownRules := rec(
            DFT := [ DFT_Base, DFT_PD, DFT_CT, DFT_GoodThomas, DFT_Rader ],
            WHT := [ WHT_Base, WHT_BinSplit ],
            MDDFT := [ MDDFT_Base, MDDFT_RowCol ],
            TRC := [ TRC_tag ]
        ),
        unparser := CUnparserProg
    );
    opts := CopyFields(When(optrec.generateComplexCode, CplxSpiralDefaults, SpiralDefaults), defaults, optrec);
    opts := InitDataType(opts, isa);

    if opts.generateComplexCode then
        t := i -> RC(DFT(i));
    else
        t := When(IsBound(opts.tags), i -> ComplexT(DFT(i), opts).withTags(opts.tags), i ->ComplexT(DFT(i), opts));
    fi;
    opts.benchTransforms := List(sizes, k -> t(k));

    dpr  := rec(verbosity := 0, timeBaseCases:=true);
    dpbench := spiral.libgen.DPBench(rec((Concat(isa, When(opts.interleavedComplex, "_ic", "_sc"))) := opts), dpr);
    if IsBound(optrec.verify) then dpbench.matrixVerify := optrec.verify; fi;

    return dpbench;
end;

# doDft auto-vectorization interface
#b := doDft(10, "f32re", _autovect(IntelC, rec(interleavedComplex := true, language := "c.icl.opt.core2")));
#b := doDft(16, "f32c", _autovect(IntelC, rec(interleavedComplex := true, language := "c.icl.opt.core2", generateComplexCode:=true)));

_autovect := (cc, opts) -> CopyFields(opts, rec(
    useRestrict := true, 
    looppragma := cc.looppragma,
    restrict := cc.restrict, 
    postalign := cc.postalign,
    arrayBufModifier := Concat("static ", cc.alignmentSpecifier()), 
    arrayDataModifier := Concat("static ", cc.alignmentSpecifier()),
    compileStrategy:=NoCSE
));

benchScalar := () -> rec(
    float := rec(
        dft_ic := rec(
            medium := _defaultSizes(s->doDft(s, "f32re", rec(interleavedComplex := true)), 13)
        ),
        dft_sc := rec(
            medium := _defaultSizes(s->doDft(s, "f32re", rec(interleavedComplex := false)), 13)
        )
    ),
    double := rec(
        dft_ic := rec(
            medium := _defaultSizes(s->doDft(s, "f64re", rec(interleavedComplex := true)), 13)
        ),
        dft_sc := rec(
            medium := _defaultSizes(s->doDft(s, "f64re", rec(interleavedComplex := false)), 13)
        )
    )
);

