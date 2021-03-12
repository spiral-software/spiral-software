
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

#the multiple number of buffers is equal as size
multiple_buffers := function(elements)
                local number, s;
               
                number := Length(elements);
               
                s := elements[1];

                return [ var("S", TArray(TArray(TReal, s.t.size), 2)) ];
end;

construct_loop := function(i, low, high, ld, st, val, br, s1, s2, s)
               local new_chain, new_loop, new_high;

               new_chain := chain(
                         SubstVars(Copy(st), rec((s2.id) := nth(s, imod(i,2)))),
                         SubstVars(Copy(ld), rec((i.id) := add(i,val), (s1.id) := nth(s, imod(i,2)))),
                         br);
               new_high := high - val;
               new_loop := swp_loop(i, [low..new_high.v], new_chain);

               return new_loop;
end; 

swp_peel := function(fun, opts) 
         #peeling instruction from loops
         local l, loops, ld, st, br, i, low, high, prologue, epilogue, new_chain, new_fun, s1, s2, svars, s;

         svars := []; 

         loops := Collect(fun, swp_loop);
         
         new_fun := Copy(fun);

         for l in loops do
             i := l.var;
             low := Minimum(l.range);
             high := Maximum(l.range);

             ld := Collect(l, [loop, @1, @2, dma_load]);
             if Length(ld) = 0 then ld := Collect(l, dma_load)[1]; else ld := ld[1]; fi;
             st := Collect(l, [loop, @1, @2, dma_store]);
             if Length(st) = 0 then st := Collect(l, dma_store)[1]; else st := st[1]; fi;
             br := Collect(l, barrier_cmd)[1];
             
             s1 := Filtered(ld.free(), i->IsBound(i.t.qualifiers) and i.t.qualifiers=[opts.scratchModifier])[1];
             s2 := Filtered(st.free(), i->IsBound(i.t.qualifiers) and i.t.qualifiers=[opts.scratchModifier])[1];
             Add(svars, s1);
             Add(svars, s2);
             
             s := multiple_buffers(svars)[1];

             prologue := chain(
                      SubstVars(Copy(ld), rec((i.id) := low, (s1.id) := nth(s, low))),
                      br,
                      SubstVars(Copy(ld), rec((i.id) := low + 1, (s1.id) := nth(s, low + 1))),
                      br);
             
             epilogue := chain(
                      SubstVars(Copy(st), rec((i.id) := high - 1, (s2.id) := nth(s, bin_and(high - 1, 1)))),
                      br,
                      SubstVars(Copy(st), rec((i.id) := high, (s2.id) := nth(s, bin_and(high, 1)))));

             new_chain := When((low = 0 and high = 1), chain(prologue, epilogue), chain(prologue, construct_loop(i, low, high, ld, st, V(2), br, s1, s2, s), epilogue));
             new_chain := RulesStrengthReduce(new_chain);
             new_fun := SubstBottomUp(new_fun, @(1, swp_loop, e->e.var = i), e->new_chain);
         od;

         return new_fun;
end;

swp_merge_variables := function(fun, opts)
                   local l, loops, i, new_loop, new_fun, s1, s2, svars, s;

                   svars := []; 

                   loops := Collect(fun, swp_loop);
         
                   new_fun := Copy(fun);

                   for l in loops do
                       i := l.var;
             
                       s1 := Filtered(l.free(), i->IsBound(i.t.qualifiers) and i.t.qualifiers=[opts.scratchModifier])[1];
                       s2 := Filtered(l.free(), i->IsBound(i.t.qualifiers) and i.t.qualifiers=[opts.scratchModifier])[2];
                       Add(svars, s1);
                       Add(svars, s2);
             
                       s := multiple_buffers(svars)[1];
                       
                       new_loop := SubstVars(Copy(l), rec((s1.id) := nth(s, imod(i, 2)), (s2.id) := nth(s, imod(i, 2))));

                       new_loop := RulesStrengthReduce(new_loop);
                       new_fun := SubstBottomUp(new_fun, @(1, swp_loop, e->e.var = i), e->new_loop);
                   od;

                   return new_fun;
end;

Class(SWPDMACodegen, DefaultCodegen, rec(
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
        return barrier_cmd(self.swp_var);
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
        return swp_loop(o.var, o.domain, self(o.child(1), y, x, opts));
    end

));

Class(SWPCPUCodegen, DefaultCodegen, rec(
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
        return nop_cmd(self.swp_var);
    end,
    
    DMAScat := meth(self, o, y, x, opts)
        local dma;
        if not IsBound(x.t.qualifiers) then x.t.qualifiers := []; fi;
        if not IsBound(y.t.qualifiers) then y.t.qualifiers := []; fi;
        x.t.qualifiers := [ opts.scratchModifier ];
        Add(y.t.qualifiers, opts.memModifier);
        Add(self.storebuffers, x);
        return barrier_cmd(self.swp_var);
    end,

    LSKernel := (self, o, y, x, opts) >> self(o.child(1), y, x, opts),
    
    DMAFence := (self,o,y,x,opts) >> chain(barrier_cmd(self.swp_var), self(o.child(1), y, x, opts)),

    swp_var := false,
    SWPSum := meth(self, o, y, x, opts)
        return swp_loop(o.var, o.domain, self(o.child(1), y, x, opts));
    end
));

Class(SWPBarrierScratchCodegen, DefaultCodegen, rec(
    CPUCodegen := SWPCPUCodegen,
    DMACodegen := SWPDMACodegen,
    MainCodegen := ScratchMainCodegen,

    Formula := meth(self, o, y, x, opts)
        local compute_time, tag, main, initfunc, cpufunc, dmafunc, prog, params, sub, initsub, memvars, scratchvars, v, io, loadvar, storevar, v, substrec, svars, initcode, datas, dvars, dv, counter;
 
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
 
        [cpufunc, dmafunc, svars] := When(opts.swp,[swp_merge_variables(cpufunc, opts), swp_peel(dmafunc, opts), multiple_buffers([loadvar, storevar])],[cpufunc,dmafunc,[loadvar,storevar]]);

        x.t.qualifiers := Set(x.t.qualifiers);
        y.t.qualifiers := Set(y.t.qualifiers);
        
        compute_time := true;

        counter := var("count",TInt);

        cpufunc := func(cpufunc.ret, cpufunc.id, When(compute_time, Concat(cpufunc.params, [counter]), cpufunc.params), chain(register(), cpufunc.cmd));
	    dmafunc := func(dmafunc.ret, dmafunc.id, When(compute_time, Concat(dmafunc.params, [counter]), dmafunc.params), chain(register(), dmafunc.cmd));
        initfunc := func(initfunc.ret, initfunc.id, initfunc.params, chain(initialization(), initfunc.cmd));
        #PrintLine(svars);
        #main := self.MainCodegen.genMain(opts, sub, dmafunc, cpufunc, [y,x], params);
        prog := program(
            decl(Concat(Set(memvars), svars, dvars), chain(
                initfunc,
                cpufunc,
		        dmafunc
        )));

        prog.dimensions := o.dims();
        return prog;
    end
));
