
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# 
#################################
#
# CodegenStrategy Stages
#
# NOTE: these are wrapped in classes because their names are simpler in a list.
#################################

#
## _EnumBlocks
#
# Add a bbnum tag to each BB, enumerating them. 
#
# changes the sums INPLACE. Has to appear before _ApplyCodegen 
# to be effective.
Class(_EnumBlocks, CodegenStrat, rec(
    __call__ := function(code, sums, opts)
        EnumBlocks(sums, opts);

        return code;
    end,
));

#
## Apply Codegen
#
# convert SigmaSPL to Code using opts.codegen
#
Class(_ApplyCodegen, CodegenStrat, rec(
    __call__ := function(code, sums, opts)
        local X, Y;

        [X, Y] := _getXY(sums, opts);

        return opts.codegen(sums, StripList(Y), StripList(X), opts);
    end,
));

#
## _BlockSums
#
#
Class(_BlockSums, CodegenStrat, rec(
    __call__ := function(code, sums, opts)
        return BlockSums(code, opts);
    end,
));

#
## _ESReduce
#
#
Class(_ESReduce, CodegenStrat, rec(
    __call__ := function(code, sums, opts)
        return ESReduce(code, opts);
    end,
));

#
## _RemoveAssignAcc
#
#
Class(_RemoveAssignAcc, CodegenStrat, rec(
    __call__ := function(code, sums, opts)
        return RemoveAssignAcc(code);
    end
));

#
## __BlockUnroll
#
#
Class(__BlockUnroll, CodegenStrat, rec(
    __call__ := function(code, sums, opts)
        return BlockUnroll(code, opts);
    end,
));

#
## _DeclareHidden
#
#
Class(_DeclareHidden, CodegenStrat, rec(
    __call__ := function(code, sums, opts)
        return DeclareHidden(code);
    end,
));

#
## _FixedPointCode
#
#
Class(_FixedPointCode, CodegenStrat, rec(
    __call__ := function(code, sums, opts)
        if IsBound(opts.isFixedPoint) and opts.isFixedPoint then
            code := FixedPointCode(code, opts.bits, opts.fracbits);
        fi;

        return code;
    end,
));


#
## new legacy wrap
#
# temporaries on stack
# out of place arrays, global ptrs, malloced on 128b boundary
#
# format is that used in the new type, with all the new functions
#
Class(_NewLegacyWrap, CodegenStrat, rec(
    __call__ := function(code, sums, opts)
        local x, y, io, params, datas, sub, initsub, init, compute, alloc, free, timer, data, prog;

        # data from sums
        [x, y] := _getXY(sums, opts);
        io := When(x=y, [x], [y, x]);
        params := Set(Collect(sums, param));
        datas := Collect(sums, FDataOfs);

        # names
        sub := Cond(IsBound(opts.subName), opts.subName, "transform");
        initsub := Cond(IsBound(opts.subName), Concat("init_", opts.subName), "initz");

        #
        # code sections start here:
        #

        # generate the 'init' code.
        if IsBound(opts.generateInitFunc) and opts.generateInitFunc then
            init := func(TVoid, initsub, params, chain(
                List(datas, x -> SReduce(x.var.init, opts))
            ));
        else
            init := func(TVoid, initsub, params, code);
        fi;
        

        # wrap the 'compute'
        compute := func(TVoid, sub, Concatenation(io, params), code);

        # create the allocs/free
        alloc := skip();
        free := skip();

        # setup timer
        timer := skip();

        # generate data
        data := List(datas, x -> x.var);

        #
        # put pieces together
        #

        prog := program(
            decl(data, chain(
                init,
                compute,
                alloc,
                free,
                timer
            ))
        );

        return prog;
    end,
));

#
## _LegacyWrap
#
#
Class(_LegacyWrap, CodegenStrat, rec(
    __call__ := function(code, sums, opts)
        local x, y, params, datas, io, sub, initsub, prog;

        [x, y] := _getXY(sums, opts);

        params := Set(Collect(sums, param));
        datas := Collect(sums, FDataOfs);

        io := When(x=y, [x], [y, x]);
        sub := Cond(IsBound(opts.subName), opts.subName, "transform");
        initsub := Cond(IsBound(opts.subName), Concat("init_", opts.subName), "initz");
        code := func(TVoid, sub, Concatenation(io, params), code);

        if IsBound(opts.generateInitFunc) and opts.generateInitFunc then
            prog := program(
                decl(List(datas, x->x.var),
                    chain(
                        func(TVoid, initsub, params, chain(List(datas, 
                            x -> SReduce(x.var.init, opts)
                        ))), 
                        code
                    )));
        else
            prog := program( func(TVoid, initsub, params, chain()), code);
        fi;

        prog.dimensions := sums.dims();

        return prog;
    end,
));

###
#
#
_BasicCodegenStrat := [
    _EnumBlocks, 
    _ApplyCodegen,
    _ESReduce,
    _RemoveAssignAcc,
    __BlockUnroll,
    _DeclareHidden,
    _FixedPointCode
];

# these functions are in cgwrap.gi
_DefaultWrapStrat := [
    _DefaultWrapInitCompute,
    _DefaultWrapData,
    _DefaultWrapAlloc,
    _DefaultWrapTimer,
    _DefaultWrapVerify
];

_SlabWrapStrat := [
    _SlabInitCompute,
    _SlabAlloc,
    _SlabTimer,
    _SlabVerify
];


DefaultCodegenStrat := Concat(_BasicCodegenStrat, _DefaultWrapStrat);
SlabCodegenStrat := Concat(_BasicCodegenStrat, _SlabWrapStrat);

# a version where all sin/cos computation has been removed from the init function
# ::: useful for getting something running on the simulator.
SlabHackedCodegenStrat := Flat([_BasicCodegenStrat, _SlabWrapStrat, _SlabHackInit]);

#
## ApplyCodegenStrat
#
# applies the codegen strategy.
#
ApplyCodegenStrat := function(sums, opts)
    local code, s;

    # some default
    code := skip();

    # apply each function in the strategy in order
    # sums and opts do not change (hopefully), but code does.
    for s in opts.codegenStrat do
        code := s(code, sums, opts);
    od;
    
    return code;
end;
