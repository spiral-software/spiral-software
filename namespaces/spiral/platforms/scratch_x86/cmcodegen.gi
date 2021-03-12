
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Class(CMScratchMainCodegen, ScratchMainCodegen, rec(
    genMain := (self, opts, sub, func1, func2, xy, params) >>
        func(TVoid, sub, Concatenation(xy, params), par_exec(
            ApplyFunc(call, Flat(Concatenation([func1],[false]))),
            ApplyFunc(call, Flat(Concatenation([func2], xy)))
        ))
));

Class(CMContextScratchCodegen, ScratchCodegen, rec(
    CPUCodegen := CPUCodegen,
    DMACodegen := DMACodegen,
    MainCodegen := CMScratchMainCodegen,
    Formula := meth(self, o, y, x, opts)
        local main, initfunc, cpufunc, dmafunc, auxfunc, prog, params, sub, initsub, memvars, scratchvars, v, io, loadvar, storevar, v, substrec, svars, initcode, datas, dvars, dv, tag;

        datas := Collect(o, FDataOfs);
        params := Set(Concatenation(Collect(o, param), Filtered(Collect(o, var), IsParallelLoopIndex)));
        sub := Cond(IsBound(opts.subName), opts.subName, "transform");

        initsub := Cond(IsBound(opts.subName), Concat("init_", opts.subName), "init");
        if IsBound(opts.generateInitFunc) and opts.generateInitFunc then
            initcode := chain(List(datas, x -> SReduce(x.var.init, opts)));
            initfunc := func(TVoid, initsub, params :: Set(Collect(initcode, param)), initcode);
        else
            initfunc := func(TVoid, initsub, params, chain());
        fi;

        dvars := List(datas, x->x.var);
        for dv in dvars do
            dv.t.qualifiers := [opts.romModifier];
        od;

        self.DMACodegen.loadbuffers := Set([]);
        self.DMACodegen.storebuffers := Set([]);
        self.DMACodegen.membuffers := Set([]);
        dmafunc := self.DMACodegen.Formula(o, y, x, opts);
        self.DMACodegen.membuffers := Set(self.DMACodegen.membuffers);
        SubtractSet(self.DMACodegen.membuffers, Set([x, y]));
        for v in self.DMACodegen.membuffers do v.t.qualifiers := [opts.memModifier]; od;

        self.CPUCodegen.loadbuffers := Set([]);
        self.CPUCodegen.storebuffers := Set([]);
        cpufunc := self.CPUCodegen.Formula(o, y, x, opts);

        memvars := Set(Concat(
            Filtered(Flat(List(Collect(cpufunc, decl), i->i.vars)), j->IsBound(j.t.qualifiers) and opts.memModifier in j.t.qualifiers),
            self.DMACodegen.membuffers));
        scratchvars := Set(Filtered(Flat(List(Collect(cpufunc, decl), i->i.vars)), j->IsBound(j.t.qualifiers) and opts.scratchModifier in j.t.qualifiers));
        
        tag := opts.tags[1];

        loadvar := var.fresh_t("S", TArray(opts.XType.t, Cond(tag.isRegCx, 2 * tag.size, tag.size)));
        loadvar.t.qualifiers := [opts.scratchModifier];
        storevar := var.fresh_t("S", TArray(opts.XType.t, Cond(tag.isRegCx, 2 * tag.size, tag.size)));
        storevar.t.qualifiers := [opts.scratchModifier];

        substrec := rec();
        for v in self.CPUCodegen.loadbuffers do substrec.(v.id) := loadvar; od;
        for v in self.CPUCodegen.storebuffers do substrec.(v.id) := storevar; od;
        for v in self.DMACodegen.loadbuffers do substrec.(v.id) := loadvar; od;
        for v in self.DMACodegen.storebuffers do substrec.(v.id) := storevar; od;
        cpufunc :=  SubstVars(cpufunc, substrec);
        dmafunc :=  SubstVars(dmafunc, substrec);

        cpufunc := SubstTopDown(cpufunc, @(1, decl), e->decl(Filtered(@(1).val.vars, i->not i in Concat(memvars, scratchvars, [loadvar, storevar])), @(1).val.cmd));
        dmafunc := SubstTopDown(dmafunc, @(1, decl), e->decl(Filtered(@(1).val.vars, i->not i in Concat(memvars, scratchvars, [loadvar, storevar])), @(1).val.cmd));

        [cpufunc, dmafunc, svars] := When(opts.swp,_double_buffer(cpufunc, dmafunc, [loadvar, storevar], opts),[cpufunc,dmafunc,[loadvar,storevar]]);

        x.t.qualifiers := Set(x.t.qualifiers);
        y.t.qualifiers := Set(y.t.qualifiers);

        main := self.MainCodegen.genMain(opts, sub, func(TVoid,"DMA_jump",[ false ],dma_jump(false)), dmafunc, [y,x], params);
        prog := program(
            decl(Concat(Set(memvars), svars, dvars), chain(
                initfunc,
                cpufunc,
		dmafunc,
                main
        )));

        prog.dimensions := o.dims();
        return prog;
    end
));
