
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# slab allocator.


_LuaInit := function(datas, slab, opts)
    local init;

    Error("haha");


end;

#
## _DefaultWrapInitCompute
#
# this is the first function which gets called on the code. It wraps
# the pure transform code in a function and also extracts any necessary
# init code.
#
# in addition, for the slab allocator, this code defines the slab struct.

Class(_SlabInitCompute, CodegenStrat, rec(
    __call__ := function(code, sums, opts)
        local x, y, io, params, datas, slab_struct,
            slab, sub, initsub, init, compute, stackvars,
            replacer,  type, length;

        # replacer function: replaces reference of a slab
        # variable in the code with a reference to the variable
        # in the slab. EG: X ==> slab->X
        replacer := (slab, code) ->
            DoForAll(slab.t.t.getVars(), e -> 
                SubstTopDown(code, e, ee -> fld(e.t, slab, e.id))
            );

        type := (a) -> a;
        length := (a) -> a;

        # -------
        # here we build the slab structure
        # -------

        # get the standard input output
        [x, y] := _getXY(sums, opts);
        io := When(x=y, x, Concat(y, x));

        # any parameters in our sigma-spl expression
        params := Set(Collect(sums, param));

        # any data required by the transform
        datas := Collect(sums, FDataOfs);

        [stackvars, code] := Pull(code, 

            # compare shape
            @(1, decl, e -> ForAny(e.vars, ee -> ObjId(ee.t) = TArray)),

            # subst shape: always drop array declarations, possibly drop decl.
            e -> let(nonarrayvars := Filtered(e.vars, ee -> ObjId(ee.t) <> TArray),
                When(nonarrayvars = [],
                    @(1).val.cmd,
                    decl(nonarrayvars, @(1).val.cmd)
                )
            ),

            # pull shape: pull out the array vars
            e -> Filtered(@(1).val.vars, ee -> ObjId(ee.t) = TArray)
        );

        stackvars := Flat(stackvars);

        # build the slab structure, and a variable for it
        slab_struct := T_Struct("slab_t", Concat(params, List(datas, e -> e.var), io, stackvars));
        slab := var("slab", TPtr(slab_struct));

        # names
        sub := Cond(IsBound(opts.subName), opts.subName, "compute");
        initsub := Cond(IsBound(opts.subName), Concat("init_", opts.subName), "initz");

        # generate the 'init' code
        if IsBound(opts.generateInitFunc) and opts.generateInitFunc then

            if IsBound(opts.luaInit) and opts.luaInit then
                init := _LuaInit(datas, slab, opts);
            else
                init := chain( List(datas, e -> SReduce(e.var.init, opts)) );
                replacer(slab, init);
            fi;

            init := func(TVoid, initsub, [slab], init);

        else
            init := func(TVoid, initsub, [slab], code);
        fi;

        # convert the direct array references to input, output, and data to
        # references to fields inside the slab alloced block.

        replacer(slab, code);

        compute := func(TVoid, sub, [slab], code);

        return program(
            define([slab_struct]),
            init,
            compute
        );

    end
));

Class(_SlabAlloc, CodegenStrat, rec(
    __call__ := function(code, sums, opts)
        local sub, initsub, funcs, slab_struct, slab, pslab, alloc, free, tmp;

        # names
        sub := Cond(IsBound(opts.subName), opts.subName, "compute");
        initsub := Cond(IsBound(opts.subName), Concat("init_", opts.subName), "initz");

        # we expect a program wrapper
        Constraint(ObjId(code) = program);

        # we expect a slab to be defined first.
        Constraint(ObjId(code.cmds[1]) = define);
        Constraint(ObjId(code.cmds[1].types[1]) = T_Struct);

        # extract functions for additional checks.
        funcs := Collect(code, func);

        # ensure ordering, code must already be wrapped by init/compute
        Constraint(2 = Length(Filtered(funcs, e -> e.id = sub or e.id = initsub)));

        slab_struct := code.cmds[1].types[1];
        slab := var("slab", TPtr(slab_struct));
        pslab := var("pslab", TPtr(TPtr(slab_struct)));
        tmp := var("tmp", slab_struct);
        tmp.size := 1;

        # allocation function.
        alloc := func(TVoid, "alloc", [pslab],
            allocate(deref(pslab), tmp)
        );

        free := func(TVoid, "dealloc", [slab], 
            deallocate(slab, slab_struct)
        );

        Add(code.cmds, alloc);
        Add(code.cmds, free);

        return code;
    end
));

Class(_SlabTimer, CodegenStrat, rec(
    __call__ := function(code, sums, opts)
        local sub, initsub, funcs, i, numruns, slab_struct, slab, t, timer;

        # names
        sub := Cond(IsBound(opts.subName), opts.subName, "compute");
        initsub := Cond(IsBound(opts.subName), Concat("init_", opts.subName), "initz");

        # we expect a program wrapper
        Constraint(ObjId(code) = program);

        # we expect a slab to be defined first.
        Constraint(ObjId(code.cmds[1]) = define);
        Constraint(ObjId(code.cmds[1].types[1]) = T_Struct);

        # extract functions.
        funcs := Collect(code, func);

        # ensure ordering, make sure we have an init and compute
        Constraint(2 = Length(Filtered(funcs, e -> e.id = sub or e.id = initsub)));

        # ensure ordering, make sure we have the alloc/dealloc functions too!
        Constraint(2 = Length(Filtered(funcs, e -> e.id = "alloc" or e.id = "dealloc")));

        # these two are used interchangably, MUST be same type.
        i := var.fresh_t("i", TInt);
        numruns := var.fresh_t("numruns", TInt);

        slab_struct := code.cmds[1].types[1];
        slab := var("slab", TPtr(slab_struct));

        t := var("t", TPtr(TVoid));

        timer := func(TVoid, "timer", [t, numruns],
            decl([slab], chain(
                _fCall("alloc", [addrof(slab)]),
                _fCall(initsub, [slab]),
                # this is used by simics to switch from a fast functional
                # to a slow timed execute mode
                When(IsBound(opts.extraTimerCall) and opts.extraTimerCall,
                    _fCall("timer_start", [t]),
                    skip()
                ),
                When(IsBound(opts.coldcache) and opts.coldcache,
                    skip(),
                    _fCall(sub, [slab]) # warmup the cache by default.
                ),
                _fCall("timer_start", [t]),
                loop(i, numruns, 
                    _fCall(sub, [slab])
                ),
                _fCall("timer_end", [t]),
                _fCall("dealloc", [slab])
            ))
        );
    
        Add(code.cmds, timer);

        return code;
    end,
));


Class(_SlabVerify, CodegenStrat, rec(
    __call__ := function(code, sums, opts)
        Add(code.cmds, func(TVoid, "verify", [], chain()));
        return code;
    end
));

#
## _SlabHackInit
#
# this object removes the sines/cosines from the init function. It was written to support
# the simplescalar backend. Simplescalar takes forever (and a day) to compute sin/cos, so
# this hack was necessary to improve the execution turnaround time.
#
# NOTE: It ONLY removes cos/sin calls from the init function, not any other.
#
Class(_SlabHackInit, CodegenStrat, rec(
    __call__ := function(code, sums, opts)
        local initsub, init;

        # get the init func name, and find it in the code.
        initsub := Cond(IsBound(opts.subName), Concat("init_", opts.subName), "initz");
        init := Collect(code, @1(1, func, e -> e.id = initsub));

        # replace all sin/cos calculations with constant 1.0
        SubstTopDown(init, sinpi, e -> V(1.0));
        SubstTopDown(init, cospi, e -> V(1.0));

        # replace the init function in the main code.
#        SubstTopDown(code, @(1, func, e -> e.id = initsub), e -> Error("a"));

        return code;
    end
));
