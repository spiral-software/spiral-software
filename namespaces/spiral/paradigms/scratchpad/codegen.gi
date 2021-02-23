
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(DMACodegen, DefaultCodegen, rec(
    Formula := meth(self, o, y, x, opts)
        local icode, datas, prog, params, sub, initsub, io, t, initcode;
        [x, y] := self.initXY(x, y, opts);

        Add(x.t.qualifiers, opts.memModifier);
        Add(y.t.qualifiers, opts.memModifier);

        o := o.child(1);
        params := Set(Concatenation(Collect(o, param), Filtered(Collect(o, var), IsParallelLoopIndex)));

        datas := Collect(o, FDataOfs);
        [o,t] := UTimedAction(BlockSumsOpts(o, opts)); #PrintLine("BlockSums ", t);
        [icode,t] := UTimedAction(self(o, y, x, opts)); #PrintLine("codegen ", t);
        [icode,t] := UTimedAction(ESReduce(icode, opts)); #PrintLine("ESReduce ", t);
        icode := RemoveAssignAcc(icode);
        Unbind(Compile.times);
        [icode,t] := UTimedAction(BlockUnroll(icode, opts)); #PrintLine("BlockUnroll ", t);
        #PrintLine("---compile--");

        icode := DeclareHidden(icode);
        if IsBound(opts.isFixedPoint) and opts.isFixedPoint then
            icode := FixedPointCode(icode, opts.bits, opts.fracbits);
        fi;

        io := When(x=y, [x], [y, x]);
        sub := Cond(IsBound(opts.subNameDMA), opts.subNameDMA, "sub_dma");
        icode := func(TVoid, sub, Concatenation(io, params), icode);

        return icode;
    end,
    
    LSKernel := meth(self, o, y, x, opts)
        return chain(dma_signal(self.swp_var), cpu_wait(self.swp_var));
    end,


    DMAGath := meth(self, o, y, x, opts)
        local i, func, rfunc, ix, size;
        Add(self.loadbuffers, y);
        Add(self.membuffers, x);
        i := Ind();

        func := o.func;
        size := 1;
        if ObjId(func) = fTensor and ObjId(Last(func.children())) = fId then
            size := Last(func.children()).domain();
            func := fTensor(Concat(DropLast(func.children(), 1), [fBase(size, 0)]));
        fi;

        func := func.lambda();
        return loop(i, func.domain(), dma_load(y+(i*size), x+func.at(i), size));
    end,

    DMAScat := meth(self, o, y, x, opts)
        local i, func, rfunc, ix, size;

        Add(self.storebuffers, x);
        Add(self.membuffers, y);
        i := Ind();

        func := o.func;
        size := 1;
        if ObjId(func) = fTensor and ObjId(Last(func.children())) = fId then
            size := Last(func.children()).domain();
            func := fTensor(Concat(DropLast(func.children(), 1), [fBase(size, 0)]));
        fi;
        
        func := func.lambda();
        return loop(i, func.domain(), dma_store(y+func.at(i), x+(i*size), size));
    end,

    DMAFence := (self,o,y,x,opts) >> chain(self(o.child(1), y, x, opts), dma_fence()),

    swp_var := false,
    SWPSum := meth(self, o, y, x, opts)
        local old_swp, c;

        old_swp := self.swp_var;
        self.swp_var := o.var;
        c := swp_loop(o.var, o.domain, self(o.child(1), y, x, opts));
        self.swp_var := old_swp;
        return c;
    end

));


Class(CPUCodegen, DefaultCodegen, rec(
    Formula := meth(self, o, y, x, opts)
        local icode, datas, prog, params, sub, initsub, io, t, initcode;
        [x, y] := self.initXY(x, y, opts);

        o := o.child(1);
        params := Set(Concatenation(Collect(o, param), Filtered(Collect(o, var), IsParallelLoopIndex)));

        datas := Collect(o, FDataOfs);
        [o,t] := UTimedAction(BlockSumsOpts(o, opts)); #PrintLine("BlockSums ", t);
        [icode,t] := UTimedAction(self(o, y, x, opts)); #PrintLine("codegen ", t);
        [icode,t] := UTimedAction(ESReduce(icode, opts)); #PrintLine("ESReduce ", t);
        icode := RemoveAssignAcc(icode);
        Unbind(Compile.times);
        [icode,t] := UTimedAction(BlockUnroll(icode, opts)); #PrintLine("BlockUnroll ", t);
        #PrintLine("---compile--");

        icode := DeclareHidden(icode);
        if IsBound(opts.isFixedPoint) and opts.isFixedPoint then
            icode := FixedPointCode(icode, opts.bits, opts.fracbits);
        fi;

        io := When(x=y, [x], [y, x]);
        Add(params, Filtered(icode.free(), i->not IsBound(i.init)));
        sub := Cond(IsBound(opts.subNameCompute), opts.subNameCompute, "sub_cpu");

        icode := func(TVoid, sub, params, icode);

        return icode;

    end,

    DMAGath := meth(self, o, y, x, opts)
        local dma;
        if not IsBound(x.t.qualifiers) then x.t.qualifiers := []; fi;
        if not IsBound(y.t.qualifiers) then y.t.qualifiers := []; fi;
        Add(x.t.qualifiers, opts.memModifier);
        y.t.qualifiers := [ opts.scratchModifier ];
        Add(self.loadbuffers, y);
        return dma_wait(self.swp_var);
    end,

    DMAScat := meth(self, o, y, x, opts)
        local dma;
        if not IsBound(x.t.qualifiers) then x.t.qualifiers := []; fi;
        if not IsBound(y.t.qualifiers) then y.t.qualifiers := []; fi;
        x.t.qualifiers := [ opts.scratchModifier ];
        Add(y.t.qualifiers, opts.memModifier);
        Add(self.storebuffers, x);
        return cpu_signal(self.swp_var);
    end,

    LSKernel := (self, o, y, x, opts) >> self(o.child(1), y, x, opts),

    swp_var := false,
    SWPSum := meth(self, o, y, x, opts)
        local old_swp, c;

        old_swp := self.swp_var;
        self.swp_var := o.var;
        c := swp_loop(o.var, o.domain, self(o.child(1), y, x, opts));
        self.swp_var := old_swp;
        return c;
    end
));

_scratch_variables := function(vars, opts)
   local new_vars, i, nv;
   new_vars := [];
    for i in vars do
        nv := Copy(i);
        nv.t := TArray(nv.t, 2);
        nv.t.qualifiers := [opts.scratchModifier];
        Add(new_vars, nv);
    od;

    return new_vars;
end;

_double_buffer := function(cpu, dma, vars, opts)
    local new_cpu, new_dma, loops, new_loops, l, ld, st, s, w, c, i, s1, s2, ll, lst, new_vars, nv, svars, srec, v, vv, dvars, ass;

    svars := [];
    loops := Collect(dma, swp_loop);
    new_loops := [];
    new_dma := Copy(dma);
    new_cpu := cpu;

    for l in loops do
        i := l.var;

        ld := Collect(l, [loop, @(1), @(2), dma_load]);
        if Length(ld) = 0 then ld := Collect(l, dma_load)[1]; else ld := ld[1]; fi;
        st := Collect(l, [loop, @(1), @(2), dma_store]);
        if Length(st) = 0 then st := Collect(l, dma_store)[1]; else st := st[1]; fi;
        s := Collect(l, dma_signal)[1];
        w := Collect(l, cpu_wait)[1];
        s1 := Filtered(ld.free(), i->IsBound(i.t.qualifiers) and i.t.qualifiers=[opts.scratchModifier])[1];
        s2 := Filtered(st.free(), i->IsBound(i.t.qualifiers) and i.t.qualifiers=[opts.scratchModifier])[1];
        Add(svars, s1);
        Add(svars, s2);

        ll := Copy(l);
        lst := SubstVars(Copy(st), rec((s2.id) := nth(s2, bin_and(i, V(2)))));
        lst := SubstVars(lst, rec((i.id) := i-V(1)));
        lst := Collect(lst, dma_store)[1];

        ll.range := [Minimum(l.range)+1..Maximum(l.range)];
        ll := SubstTopDown(ll, @(1, dma_load), e->dma_load(nth(@(1).val.loc, bin_and(i, V(2))), @(1).val.exp, @(1).val.size));
        ll := SubstTopDown(ll, @(1, dma_store), e->lst);
        ll := SubstTopDown(ll, @(1, cpu_wait), e->cpu_wait(@(1).val.args[1]-1));

        c := chain([
            SubstVars(chain([Copy(ld), Copy(s)]), rec((i.id) := V(0), (s1.id) := nth(s1, 0))),
            ll,
            SubstVars(chain([Copy(w), Copy(st)]), rec((i.id) := i.range-V(1), (s2.id) := nth(s2, 1))),
        ]);
        c := RulesStrengthReduce(c);
        new_dma := SubstBottomUp(new_dma, @(1, swp_loop, e->e.var = i), e->c);
    od;

    svars := Set(svars);
    loops := Collect(cpu, swp_loop);
    new_cpu := Copy(cpu);
    new_loops := [];
    for l in loops do
        i := l.var;
        srec := rec();
        dvars := [];
        ass := [];
        for v in svars do
            vv := var.fresh_t("R", TPtr(v.t.t));
            Add(dvars, vv);
            Add(ass, assign(vv, nth(v, bin_and(i, V(2)))));
            srec.(v.id) := vv;
        od;
        c := SubstVars(l, srec);
        c := swp_loop(c.var, c.range, decl(dvars, chain(Concat(ass, [c.cmd]))));
        new_cpu := SubstBottomUp(new_cpu, @(1, swp_loop, e->e.var = i), e->c);
    od;

    new_vars := _scratch_variables(vars,opts);

    return [new_cpu, new_dma, new_vars];
end;

Class(ScratchMainCodegen, rec(
    genMain := (self, opts, sub, dmafunc, cpufunc, xy, params) >>
        func(TVoid, sub, Concatenation(xy, params), par_exec(
            ApplyFunc(call, Flat(Concatenation([dmafunc], xy))),
            call(cpufunc)
        ))
));

Class(ScratchCodegen, DefaultCodegen, rec(
    CPUCodegen := CPUCodegen,
    DMACodegen := DMACodegen,
    MainCodegen := ScratchMainCodegen,

    Formula := meth(self, o, y, x, opts)
        local tag, main, initfunc, cpufunc, dmafunc, prog, params, sub, initsub, memvars, scratchvars, v, io, loadvar, storevar, v, substrec, svars, initcode, datas, dvars, dv;

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

        main := self.MainCodegen.genMain(opts, sub, dmafunc, cpufunc, [x,y], params);
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
