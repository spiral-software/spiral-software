
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(CellCodegen, VectorCodegen, rec(
    Formula := meth(self, o, y, x, opts)
        local icode, datas, prog, params, sub, initsub, io;
        if IsBound(opts.XType) and not IsList(x) then
            x.t := TPtr(opts.XType);
            if IsBound(opts.useRestrict) and opts.useRestrict then
                x.t := x.t.restrict();
            fi;
        fi;
        if IsBound(opts.YType) and not IsList(y) then
            y.t := TPtr(opts.YType);
            if IsBound(opts.useRestrict) and opts.useRestrict then
                y.t := y.t.restrict();
            fi;
        fi;

        o := o.child(1);
        params := Set(Collect(o, param));

        datas := Collect(o, FDataOfs);
        o := BlockSums(opts.globalUnrolling, o);
        icode := ESReduce(self(o, y, x, opts),opts);
        icode := RemoveAssignAcc(icode);
        #Error("BP");
        icode := BlockUnroll(icode, opts);
        # icode := PowerOpt(icode);
        icode := DeclareHidden(icode);
        if IsBound(opts.isFixedPoint) and opts.isFixedPoint then
            icode := FixedPointCode(icode, opts.bits, opts.fracbits);
        fi;

        # Insert sw pipelining here
        icode := MarkPreds(icode);
        icode := MarkDefUse(icode);
        #SubstTopDown(icode, loop, e->MarkSWPLoops(e));
        #SubstTopDown(icode, loop_sw, e->SoftwarePipeline(e));

        io := When(x=y, [x], [y, x]);
        sub := Cond(IsBound(opts.subName), opts.subName, "transform");
        initsub := Cond(IsBound(opts.subName), Concat("init_", opts.subName), "init");
        icode := func(TVoid, sub, Concatenation(io, params), icode);

        if IsBound(opts.generateInitFunc) and opts.generateInitFunc then
            prog := program(
                decl(List(datas, x->x.var),
                    chain(
                        func(TVoid, initsub, params, chain(List(datas, x -> SReduce(x.var.init, opts)))),
                        icode
                    )));
        else
            prog := program( func(TVoid, initsub, params, chain()), icode);
        fi;
        prog.dimensions := o.dims();
        return prog;
    end,

));

