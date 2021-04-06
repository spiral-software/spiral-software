
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details



#
#######################
#
# local helper functions
#
#######################

#
## _fProto
#
# create a TFunc(...) object from an array of variables
#
# eg: _fProto([
#        var.fresh_t("i", TPtr(TInt)), 
#        var.fresh_t("b", TReal)
#     ])
#     ----> TFunc( TPtr(TInt), TReal )
#
# NOTE: useful for converting lists of fully typed variables into a function proto

_fProto := (a) -> ApplyFunc(TFunc, List(a, e -> e.t));

#
## _fCall
#
# create a function call in 'spiral-code' reprentation given an array and a fully typed variable list
#
# eg: _fCall("foo", [
#        var.fresh_t("i", TPtr(TInt)), 
#        var.fresh_t("b", TReal)
#     ])
#     ----> call("foo", TFunc(TPtr(TInt), TReal), i, b)
#
# NOTE: this is generally unparsed to
#
#     foo(i, b)
#
_fCall := (name, var_array) -> ApplyFunc(call, Concat([var(name, _fProto(var_array))], var_array));

#
## _wrapInPtr
#
# copy a list of vars, creating new vars that are pointers
#
# eg: _wrapInPtr([
#        var.fresh_t("i", TPtr(TInt))
#        var.fresh_t("b", TReal)
#     ])
#     ---> [ var.fresh_t("i", TPtr(TPtr(TInt))), var.fresh_t("b", TPtr(TReal)) ]
#
# this is useful if you want to pass by reference, as in the case of allocation functions.
#
# NOTE: this function does not modify the input array

_wrapInPtr := (var_array) -> List(var_array, e -> CopyFields(e, rec(t := TPtr(e.t))));


#
## _arraysToPtrs
#
# turns arrays into pointers so that the arrays are not statically defined.
#
_arraysToPtrs := (var_array) -> List(var_array, 
    e -> When(ObjId(e.t) = TArray, 
        CopyFields(e, rec(t := TPtr(e.t.t))),
        CopyFields(e)
    )
);
    
#
## _getXY
#
# get input and output arrays. returns a nested array, 
#   IsArray(X) and IsArray(Y) = true
#
# NOTE: var.fresh cannot be used here, as this function needs to ALWAYS 
# return the same data, as it may be called >1 
#
_getXY := function(sums, opts)
    local X, Y, precision;

    precision := 64;

    if IsBound(opts.precision) then
        if opts.precision = "single" then
            precision := 32;
        elif opts.precision = "double" then
            precision := 64;
        else
            Error("Unknown precision");
        fi;
    fi;
    
    if IsBound(opts.doSumsUnification) and opts.doSumsUnification then
        X := List([1..Length(sums.dmn())],
           (i) -> var(Concatenation("X", String(i)), sums.dmn()[i], sums.dmn()[i].size));
        Y := List([1..Length(sums.rng())],
           (i) -> var(Concatenation("Y", String(i)), sums.rng()[i], sums.rng()[i].size));

        return [X, Y];
    fi;

    if IsList(sums.dims()[2])  then
        X := List([ 1 .. Length(sums.dims()[2]) ], 
            (x) -> var(Concatenation("X", String(x)), TPtr(T_Real(precision)), sums.dims()[2]));
    else
        X := Cond(IsBound(opts.X), opts.X, [var("X", TArray(T_Real(precision), sums.dims()[2]), sums.dims()[2])]);
    fi;

    if IsList(sums.dims()[1])  then
        Y := List([ 1 .. Length(sums.dims()[1]) ], 
            (x) -> var(Concatenation("Y", String(x)), TPtr(T_Real(precision)), sums.dims()[1]));
    else
        Y := Cond(
            opts.inplace, X, 
            IsBound(opts.Y), opts.Y, 
            [var("Y", TArray(T_Real(precision), sums.dims()[1]), sums.dims()[1])]
        );
    fi;
    
    return [X, Y];
end;

######################
#
## default wrappers: init/compute, alloc, timer, verify, data
#
# these defaults generate the above functions and our output has the following
# characteristics:
#
# temp variables on the stack
# statically declared data (twiddles in case of DFT)
# calloc'ed (default allocate func) input/output pointers
#
######################

#
## _DefaultWrapInitCompute
#
#
# wraps passed in code in compute() function, generates init() function.
#
Class(_DefaultWrapInitCompute, CodegenStrat, rec(
    __call__ := function(code, sums, opts)
        local x, y, io, params, datas, sub, initsub, init, compute;

        # extract data from 'sums'
        [x, y] := _getXY(sums, opts);
        io := When(x=y, x, Concat(y, x));
        params := Set(Collect(sums, param));
        datas := Collect(sums, FDataOfs);

        # names
        sub := Cond(IsBound(opts.subName), opts.subName, "compute");
        initsub := Cond(IsBound(opts.subName), Concat("init_", opts.subName), "initz");

        # generate the 'init' code
        if IsBound(opts.generateInitFunc) and opts.generateInitFunc then
            init := func(TVoid, initsub, params, chain(
                List(datas, e -> SReduce(e.var.init, opts))
            ));
        else
            init := func(TVoid, initsub, When(params = [], [TVoid], params), code);
        fi;

        # wrap the 'compute'
        compute := func(TVoid, sub, Concatenation(io, params), code);

        return program(chain(
            init,
            compute
        ));
    end
));

#
## _DefaultWrapAlloc
#
# sets up allocation functions.
#
# NOTE: only functions in 'code' are preserved, everything else is trashed.

Class(_DefaultWrapAlloc, CodegenStrat, rec(
    __call__ := function(code, sums, opts)
        local x, y, io, iop, sub, initsub, funcs, alloc, free;

        [x, y] := _getXY(sums, opts);
        io := When(x=y, x, Concat(y, x));
        iop := _wrapInPtr(io);

        # names
        sub := Cond(IsBound(opts.subName), opts.subName, "compute");
        initsub := Cond(IsBound(opts.subName), Concat("init_", opts.subName), "initz");

        # extract functions.
        funcs := Collect(code, func);

        # ensure ordering, code must already be wrapped by init/compute
        Constraint(2 = Length(Filtered(funcs, e -> e.id = sub or e.id = initsub)));

        # we expect a program wrapper
        Constraint(ObjId(code) = program);

        alloc := func(TVoid, "alloc", iop, chain(
            List(iop, e -> allocate(
                deref(e), 
                e.t.t
#                When(IsBound(e.range) and ObjId(e.t) = TPtr,
#                    TArray(e.t.t, e.range),
#                    e.t
#                )
            ))
        ));

        free := func(TVoid, "dealloc", io, chain(
            List(io, e -> deallocate(
                e, 
                e.t
#                When(IsBound(e.range) and ObjId(e.t) = TPtr,
#                    TArray(e.t.t, e.range),
#                    e.t
#                )
            ))
        ));

#        alloc.countedArithCost := (self, countrec) >> countrec.arithcost(0);
#        free.countedArithCost := (self, countrec) >> countrec.arithcost(0);

        Add(code.cmds, alloc);
        Add(code.cmds, free);

        return code;
    end
));

#
## _DefaultWrapTimer
#
# adds a timer function. 
#
# relies on existing init, compute, alloc, and dealloc.

Class(_DefaultWrapTimer, CodegenStrat, rec(
    __call__ := function(code, sums, opts)
        local x, y, io, params, sub, initsub, funcs, i, numruns, t, timer;

        [x, y] := _getXY(sums, opts);
        io := When(x=y, x, Concat(y, x));
        params := Set(Collect(sums, param));

        # names
        sub := Cond(IsBound(opts.subName), opts.subName, "compute");
        initsub := Cond(IsBound(opts.subName), Concat("init_", opts.subName), "initz");

        # extract functions.
        funcs := Collect(code, func);

        # ensure ordering, make sure we have an init and compute
        Constraint(2 = Length(Filtered(funcs, e -> e.id = sub or e.id = initsub)));

        # ensure ordering, make sure we have the alloc/dealloc functions too!
        Constraint(2 = Length(Filtered(funcs, e -> e.id = "alloc" or e.id = "dealloc")));

        # we expect a program wrapper
        Constraint(ObjId(code) = program);

        # these two are used interchangably, MUST be same type.
        i := var.fresh_t("i", TInt);
        numruns := var.fresh_t("numruns", TInt);

        t := var("t", TPtr(TVoid));

        timer := func(TVoid, "timer", [t, numruns],
            decl(Concat(_arraysToPtrs(io), params), chain(
                _fCall("alloc", List(io, addrof)),
                _fCall(initsub, params),

                # this is used by simics to switch from a fast functional
                # to a slow timed execute mode
                When(IsBound(opts.extraTimerCall) and opts.extraTimerCall,
                    _fCall("timer_start", [t]),
                    skip()
                ),
                When(IsBound(opts.coldcache) and opts.coldcache, 
                    skip(),
                    _fCall(sub, Concat(io, params)) # warm up the cache by default
                ), 
                _fCall("timer_start", [t]),
                loop(i, numruns, 
                    _fCall(sub, Concat(io, params))
                ),
                _fCall("timer_end", [t]),
                _fCall("dealloc", io)
            ))
        );
    
#        timer.countedArithCost := (self, countrec) >> countrec.arithcost(0);

        Add(code.cmds, timer);

        return code;
    end,
));

#
## _DefaultWrapVerify
#
#
Class(_DefaultWrapVerify, CodegenStrat, rec(
    __call__ := function(code, sums, opts)
        local x, y, io, params, sub, initsub, funcs, i, j, basis, printY, verify;
        
        [x, y] := _getXY(sums, opts);
        io := When(x=y, x, Concat(y, x));
        params := Set(Collect(sums, param));

        # names
        sub := Cond(IsBound(opts.subName), opts.subName, "compute");
        initsub := Cond(IsBound(opts.subName), Concat("init_", opts.subName), "initz");

        # extract functions.
        funcs := Collect(code, func);

        # ensure ordering, make sure we have an init and compute
        Constraint(2 = Length(Filtered(funcs, e -> e.id = sub or e.id = initsub)));

        # ensure ordering, make sure we have the alloc/dealloc functions too!
        Constraint(2 = Length(Filtered(funcs, e -> e.id = "alloc" or e.id = "dealloc")));
        
        # we expect a program wrapper
        Constraint(ObjId(code) = program);

        i := var.fresh_t("i", TInt);
        j := var.fresh_t("j", TInt);

        basis := func(TVoid, "basis", Concat(x, [j]),
            chain(
                loop(i, x[1].range,
                    assign(nth(x[1], i), V(0))
                ),
                assign(nth(x[1], j), V(1))
            )
        );

        # outputs only entries in y[1].
        printY := func(TVoid, "printY", y, chain(
            PRINT("["),
            PRINT("%lf", nth(y[1], 0)),
            loop(i, (y[1].range-1), 
                PRINT(", %lf", nth(y[1], add(i,1)))
            ),
            PRINT("]")

        ));

        verify := func(TVoid, "verify", [], 
            decl(_arraysToPtrs(io), chain(

                _fCall("alloc", List(io, addrof)),

                _fCall(initsub, params),

                PRINT("[\\n"),
                loop(i, x[1].range, chain(
                    _fCall("basis", Concat(x, [i])),
                    _fCall(sub, Concat(io, params)),
                    _fCall("printY", y),


                    IF(neq(i, x[1].range-1), PRINT(",\\n"), skip())
                )),
                PRINT("\\n];\\n"),
                _fCall("dealloc", io)
            ))
        );

#        basis.countedArithCost := (self, countrec) >> countrec.arithcost(0);
#        printY.countedArithCost := (self, countrec) >> countrec.arithcost(0);
#        verify.countedArithCost := (self, countrec) >> countrec.arithcost(0);

        Append(code.cmds, [basis, printY, verify]);
        
        return code;
    end
));

#
## _DefaultWrapData
#
# must be done after all function wraps
#
# allocates data arrays (for things like twiddles) globally in the file.
#
Class(_DefaultWrapData, CodegenStrat, rec(
    __call__ := function(code, sums, opts)
        local datas, data;

        datas := Collect(sums, FDataOfs);

        # for static data declared in file.
        data := List(datas, e -> e.var);

        # we expect a program wrapper
        Constraint(ObjId(code) = program);

        # put data chunk at start of program

        code.cmds := Concat([decl(data, skip())], code.cmds);

        return code;
    end
));

#
## _DefaultWrapAll
#
# same structure as the new legacy wrap except uses the new timer paradigm.
#
# by structure, we have 
# data (like twiddles) in the data segment, not allocated
# input/output arrays allocated
# temporary arrays on the stack
#
Class(_DefaultWrap, CodegenStrat, rec(
    __call__ := function(code, sums, opts)
        local 
            x, y, io, iop, params, datas,                # sums related data
            sub, initsub,                                # strings
            t, j, i,                                   # var names
            data,                                        # twiddle/other data
            init, compute, alloc, free, timer, verify,   # functions
            basis, printY,                               # 
            prog;                                        # full program

        #
        # data from sums
        #

        # in/out and sizes
        [x, y] := _getXY(sums, opts);
        io := When(x=y, x, Concat(y, x));
        iop := _wrapInPtr(io);
        params := Set(Collect(sums, param));
        datas := Collect(sums, FDataOfs);

        # names
        sub := Cond(IsBound(opts.subName), opts.subName, "compute");
        initsub := Cond(IsBound(opts.subName), Concat("init_", opts.subName), "initz");

        #
        # code sections start here:
        #

        # generate the 'init' code
        if IsBound(opts.generateInitFunc) and opts.generateInitFunc then
            init := func(TVoid, initsub, params, chain(
                List(datas, e -> SReduce(e.var.init, opts))
            ));
        else
            init := func(TVoid, initsub, params, code);
        fi;

        # wrap the 'compute'
        compute := func(TVoid, sub, Concatenation(io, params), code);

        # create the allocs/free
        #
        # NOTE: extra level of derefs because we modify incoming pointers
        alloc := func(TVoid, "alloc", iop, chain(
            List(iop, e -> allocate(
                deref(e), 
                When(IsBound(e.range) and ObjId(e.t.t) = TPtr,
                    TArray(e.t.t.t, e.range),
                    e.t.t
                )
            ))
        ));

        free := func(TVoid, "dealloc", io, chain(
            List(io, e -> deallocate(
                e, 
                When(IsBound(e.range) and ObjId(e.t) = TPtr,
                    TArray(e.t.t, e.range),
                    e.t
                )
            ))
        ));

        #
        # setup timer
        # 

        # these two are used interchangably, MUST be same type.
        j := var.fresh_t("j", TInt);
        i := var.fresh_t("i", TInt);

        t := var("t", TPtr(TVoid));

        timer := func(TVoid, "timer", [t, j],
            decl(Concat(io, params), chain(
                _fCall("alloc", List(io, addrof)),
                _fCall(initsub, params),
                _fCall("timer_start", [t]),
                loop(i, j, 
                    _fCall(sub, Concat(io, params))
                ),
                _fCall("timer_end", [t]),
                _fCall("dealloc", io)
            ))
        );
    
        #
        # verifier
        #

        # verifier helpers
        #
        # 

        basis := func(TVoid, "basis", Concat(x, [j]),
            chain(
                loop(i, x[1].range,
                    assign(nth(x[1], i), V(0))
                ),
                assign(nth(x[1], j), V(1))
            )
        );

        printY := func(TVoid, "printY", y, skip());

        verify := func(TVoid, "verify", [], 
            decl(io, chain(

                _fCall("alloc", List(io, addrof)),

                _fCall(initsub, params),

                loop(i, x[1].range, chain(
                    _fCall("basis", Concat(x, [i])),
                    _fCall(sub, Concat(io, params)),
                    _fCall("printY", y)
                )),

                _fCall("dealloc", io)
            ))
        );

        # for static data declared in file.
        data := List(datas, e -> e.var);

        #
        # put pieces together
        #

        prog := program(
            decl(data, chain(
                init,
                compute,
                comment(""),
                comment("****************************************"),
                comment("* timer/verifier code below this point *"),
                comment("****************************************"),
                alloc,
                free,
                timer,
                basis,
                printY,
                verify
            ))
        );

        return prog;
    end,
));
