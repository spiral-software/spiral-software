
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(Codegen, HierarchicalVisitor, rec(
    initXY := function(x, y, opts)
        if IsBound(opts.XType) and not IsList(x) then
            if not IsArrayT(opts.XType) and not IsPtrT(opts.XType) 
                then Error("opts.XType must be a pointer or array type. This has recently changed.",
                           "If you used TReal before, use TPtr(TReal) now"); fi;
            x.t := opts.XType;
            x.t := When(IsBound(opts.useRestrict) and opts.useRestrict, x.t.restrict(), x.t);
        fi;
        if IsBound(opts.YType) and not IsList(y) then
            if not IsArrayT(opts.YType) and not IsPtrT(opts.YType) 
                then Error("opts.YType must be a pointer or array type. This has recently changed.",
                           "If you used TReal before, use TPtr(TReal) now"); fi;
            y.t := opts.YType;
            y.t := When(IsBound(opts.useRestrict) and opts.useRestrict, y.t.restrict(), y.t);
        fi;
        return [x, y];
    end,

    # NOTE: get rid of _acc_

    # This function substitutes assign by assign_acc (eliminated in a subsequent pass)
    # This is done to achieve accumulation, and assign_acc is used to
    #   prevent double accumulation.. Used in Codegen.ISumAcc and Codegen.SUMAcc.
    # Must be handled better somehow.
    #
    _acc := (icode, y) ->
        SubstTopDownNR(icode, [assign, [@(0,nth), @(1), @(2)], @(3)],
            e -> assign_acc(@(0).val, @(3).val)),

 
    _interleave := function(codes)
        local decls, c, i, j, cmds;
        decls := Set([]);
        for i in [1..Length(codes)] do
            c := codes[i];
            while ObjId(c)=decl do UniteSet(decls, c.vars); c := c.cmd; od;
            if ObjId(c)<>chain then
                codes[i] := [c];
            else
                codes[i] := c.cmds;
            fi;
        od;

        cmds := [];
        for i in [1..Maximum(List(codes, Length))] do
            for j in [1..Length(codes)] do
                if IsBound(codes[j][i]) then Add(cmds, codes[j][i]); fi;
            od;
        od;
        return decl(decls, chain(cmds));
    end
));

#fAdd.rlambda := self >> let(i:=Ind(), Lambda(i,i+1));
#H.rlambda := self >> let(i:=Ind(), Lambda(i,i+self.params[4]));


# a version of the Dat1d call that also sets the 'no scalarize' flag on the variable 't'.
_dat1d_donotscalarize := function(a,b)
    local t;

    t := Dat1d(a,b);
    t.doNotScalarize := true;

    return t;
end;


Class(DefaultCodegen, Codegen, rec(
    Formula := meth(self, o, y, x, opts)
        local icode, datas, prog, params, sub, initsub, destroysub, io, t, initcode, initparams;
        
        o := SumsUnification(o.child(1), opts);

        [x, y] := self.initXY(x, y, opts);

        #o :=  Process_fPrecompute(o, opts);
        
        params := Set(Concatenation(Collect(o, param), Filtered(Collect(o, var), IsParallelLoopIndex)));

        datas := Collect(o, FDataOfs);
        [o,t] := UTimedAction(BlockSumsOpts(o, opts)); #PrintLine("BlockSums ", t);
        [icode,t] := UTimedAction(self(o, y, x, opts)); #PrintLine("codegen ", t);
        #[icode,t] := UTimedAction(ESReduce(icode, opts)); #PrintLine("ESReduce ", t);
        icode := RemoveAssignAcc(icode);
        Unbind(Compile.times);
        [icode,t] := UTimedAction(BlockUnroll(icode, opts)); #PrintLine("BlockUnroll ", t);
        #PrintLine("---compile--");
        #DoForAll([1..Length(Compile.times)], i -> PrintLine(i, " ", Compile.times[i], " ",
        #        let(f:=opts.compileStrategy[i], When(IsFunc(f) or IsMeth(f), "---", f))));

        # icode := PowerOpt(icode);
        icode := DeclareHidden(icode);
        if IsBound(opts.isFixedPoint) and opts.isFixedPoint then
            icode := FixedPointCode(icode, opts.bits, opts.fracbits);
        fi;

        initparams := Copy(params);
        if IsBound(opts.symbol) then
          params := Concatenation(params, opts.symbol);
        fi;

        if IsBound(opts.accStrategy) then
          icode.iy := y;
          icode.iy.n := o.dims()[1];
          icode.ix := x;
          icode.ix.n := o.dims()[2];
          icode.ivars := Concatenation(params, List(datas, x->x.var));
          icode := opts.accStrategy(icode);
        fi;

        io := When(x=y, [x], [y, x]);
        sub := Cond(IsBound(opts.subName), opts.subName, "transform");
        initsub := Cond(IsBound(opts.subName), Concat("init_", opts.subName), "init");
        destroysub := Cond(IsBound(opts.subName), Concat("destroy_", opts.subName), "destroy");
        icode := func(TVoid, sub, Concatenation(io, params), icode);

        if IsBound(opts.generateInitFunc) and opts.generateInitFunc then
	    initcode := chain(List(Filtered(datas, e -> IsBound(e.var.init)), x -> SReduce(x.var.init, opts)));
            prog := program(
                decl(List(datas, x->x.var),
                    chain(
                        func(TVoid, initsub, initparams :: Set(Collect(initcode, param)), initcode), 
                        icode,
                        func(TVoid, destroysub, [], skip()) 
                    )));
        else
            prog := program( func(TVoid, initsub, params, chain()), icode);
        fi;
        prog.dimensions := o.dims();
        return prog;
    end,

    Cross := meth(self, o, y, x, opts)
        local i, mychain, xdims, ydims, myYdims, myXdims;
        mychain:=[];xdims:=1;ydims:=1;
        for i in [1..Length(o._children)] do
            myYdims:=DimLength(o._children[i].dims()[1]);
            myXdims:=DimLength(o._children[i].dims()[2]);
            Add(mychain,self(o._children[i],
                    StripList(y{[ydims..ydims+myYdims-1]}),
                    StripList(x{[xdims..xdims+myXdims-1]}),opts));
            ydims:=ydims+myYdims;
            xdims:=xdims+myXdims;
        od;
        return chain(mychain);
    end,

    Glue:= meth(self,o,y,x,opts)
        local size, als,iterator,n;
        size := o.element[2];
        n := EvalScalar(o.element[1]);
        iterator :=Ind(size);
        als  := List([0..n-1],t -> assign(nth(StripList(y),add(iterator,t*size)),nth(x[t+1],iterator)));
        return loop(iterator, size, chain(als));
    end,

    Split:= meth(self,o,y,x,opts)
        local size, als,iterator,n;
        n := EvalScalar(o.element[2]);
        size := o.element[1];
        iterator :=Ind(idiv(size, n));
        als  := List([0..(n-1)],t -> assign(nth(y[t+1],iterator),nth(StripList(x),add(iterator,t*size/n))));
        return loop(iterator, idiv(size, n), chain(als));
     end,


    BB := (self,o,y,x,opts) >> MarkForUnrolling(
        When(IsBound(o.bbnum),
            o.bbnum,
            0
        ), 
        self(o.child(1), y, x, opts),
        opts
    ),
    IDirSum := (self,o,y,x,opts) >> self(o.sums(), y, x, opts),
    TTag := (self,o,y,x,opts) >> self(o.params[1], y, x, opts),
    DPWrapper := (self,o,y,x,opts) >> self(o.child(1), y, x, opts),

    Buf := (self,o,y,x,opts) >> self(o.child(1), y, x, opts),
    NoPull := (self,o,y,x,opts) >> self(o.child(1), y, x, opts),
    PushL := (self,o,y,x,opts) >> self(o.child(1), y, x, opts),
    PushR := (self,o,y,x,opts) >> self(o.child(1), y, x, opts),
    PushLR := (self,o,y,x,opts) >> self(o.child(1), y, x, opts),
    Grp := (self,o,y,x,opts) >> self(o.child(1), y, x, opts),
    NoDiagPullin := (self,o,y,x,opts) >> self(o.child(1), y, x, opts),
    NoDiagPullinLeft := (self,o,y,x,opts) >> self(o.child(1), y, x, opts),
    NoDiagPullinRight := (self,o,y,x,opts) >> self(o.child(1), y, x, opts),

    COND := (self,o,y,x,opts) >> IF(
        When(IsFunction(o.cond), o.cond.at(0), o.cond),
        self(o.child(1), y, x, opts), self(o.child(2), y, x, opts)),

    Diag := (self, o, y, x, opts) >> let(i := Ind(), elt := o.element.lambda(),
        loop(i, elt.domain(), assign(nth(y,i), elt.at(i) * nth(x,i)))),
    
    DiagCpxSplit := (self, o, y, x, opts) >> let(i := Ind(), elt := o.element.lambda(),
        re := elt.at(2*i), im := elt.at(2*i+1),
        loop(i, elt.domain()/2, 
             assign(nth(y,i),   re * nth(x,i) + E(4) * im * nth(x,i)))),

    TCast := (self, o, y, x, opts) >> let(i := Ind(), 
        loop(i, o.params[1], assign(nth(y,i), tcast(o.params[2], nth(x,i))))),

    RCDiag := (self, o, y, x, opts) >> let(i := Ind(), elt := o.element.lambda(),
        re := elt.at(2*i), im := elt.at(2*i+1),
        loop(i, elt.domain()/2, chain(
             assign(nth(y,2*i),   re * nth(x,2*i) - im * nth(x,2*i+1)),
             assign(nth(y,2*i+1), im * nth(x,2*i) + re * nth(x,2*i+1))))),

    ColVec := (self, o, y, x, opts) >> let(i := Ind(), func := o.element.lambda(),
        loop(i, func.domain(), assign(nth(y,i), mul(func.at(i), nth(x,0))))),

    RowVec := (self, o, y, x, opts) >> let(i := Ind(), func := o.element.lambda(),
        t := TempVar(x.t.t),
        chain(assign(t,0),
            loop(i, func.domain(), assign(t, add(t, mul(func.at(i), nth(x,i))))),
            assign(nth(y,0), t))),

    Scale := (self, o, y, x, opts) >> let(i := Ind(),
        chain(self(o.child(1), y, x, opts),
              loop(i, Rows(o), assign(nth(y,i), mul(o.scalar, nth(y,i)))))),

    I := (self, o, y, x, opts) >> Cond(x<>y,let(i := Ind(Rows(o)),
            loop(i, i.range, assign(nth(y,i), nth(x,i)))),skip()),

    2DI := (self, o, y, x, opts) >> Cond(x<>y, Error("Should not happen"), skip()),

    # this one will always unroll Blk's
    Blk := (self, o, y, x, opts) >> let(
	# this is a hack, and it is needed, because without it we will be assigning to a 
	# variable of type TArray (after binsplit), and some compilers (=icc) don't like that 
	# (not an l-value)
	# NOTE: suggestion -- move this somewhere, as a compatibility patch (maybe postprocess?)
	tcst := Cond(IsSymbolic(o.element), x->tcast(TPtr(o.element.t.t.t), x), x->x),
	Cond(
            not IsSymbolic(o.element) and Rows(o)=2 and Cols(o)=2, Blk2code(o, y, x),
            not IsSymbolic(o.element) and Rows(o)=4 and Cols(o)=4, Blk4code(o, y, x),
            chain(
		List([0..Rows(o)-1], j -> let(
		    row := tcst(nth(o.element, j)),
                    assign(nth(y,j), ApplyFunc(add, List([0..Cols(o)-1], i -> nth(row, i) * nth(x,i))))
		))
	    )
	)
    ),

# This one can loop Blk's if needed, but is much slower for large matrices, and eventually blows up in storage requirements
#    Blk := (self, o, y, x, opts) >> Cond(
#        Rows(o)=2 and Cols(o)=2, Blk2code(o, y, x),
#        Rows(o)=4 and Cols(o)=4, Blk4code(o, y, x),
#        let(j:=Ind(), i:=Ind(), t:=TempVar(x.t.t), mat:=V(o.element), d:=Dat(mat.t),
#            data(d, mat,
#                loop(j, Rows(o), decl(t, chain(
#                     assign(t, 0),
#                     loop(i, Cols(o), assign(t, t + nth(nth(d,j),i) * nth(x,i))),
#                     assign(nth(y,j), t))))))),

    toeplitz := (self, o, y, x, opts) >> self.Blk(o.obj, y, x, opts),

    Blk1 := (self, o, y, x, opts) >> assign(nth(y,0), mul(toExpArg(o.element), nth(x,0))),

    BlkConj := (self, o, y, x, opts) >> assign(nth(y,0), conj(nth(x,0))),

    Prm := (self, o, y, x, opts) >> Cond(
        x = y,
            When(ObjId(o.func) = fId, skip(), Error("Inplace Permutation is not dealt with...")),

        let(i:=Ind(), func:=o.func.lambda(),
            loop(i, Rows(o), assign(nth(y, i), nth(x, func.at(i)))))
    ),

    O := (self, o, y, x, opts) >> let(i:=Ind(),
        loop(i, o.params[1], assign(nth(y, i), V(0)))),


    Gath := meth(self, o, y, x, opts)
        local i, func, rfunc, ix;
        i := Ind(); func := o.func.lambda();

        if IsBound(o.func.rlambda) then
            rfunc := o.func.rlambda();
            ix := var.fresh_t("ix", TInt);
            return decl(ix, chain(
                    assign(ix, func.at(0)),
                    assign(nth(y, 0), nth(x, ix)),
                    loop(i, o.func.domain()-1,
                        chain(assign(ix, rfunc.at(ix)),
                            assign(nth(y, i+1), nth(x, ix))))));
        else
            return loop(i, o.func.domain(), assign(nth(y,i), nth(x, func.at(i))));
        fi;
    end,

    Scat := meth(self, o, y, x, opts)
        local i, func, rfunc, ix;
        i := Ind(); func := o.func.lambda();
        if IsBound(o.func.rlambda) then
        rfunc := o.func.rlambda();
        ix := var.fresh_t("ix", TInt);
        return decl(ix, chain(
                assign(ix, func.at(0)),
                assign(nth(y, ix), nth(x, 0)),
                loop(i, o.func.domain()-1,
                    chain(assign(ix, rfunc.at(ix)),
                        assign(nth(y, ix), nth(x, i+1))))));
        else
            return loop(i, o.func.domain(), assign(nth(y,func.at(i)), nth(x, i)));
        fi;
    end,

    ScatGath := meth(self, o, y, x, opts)
        local i, sfunc, gfunc, decls;

        i := Ind();
        sfunc := o.sfunc.lambda();
        gfunc := o.gfunc.lambda();
        decls := Set(Concat(sfunc.free(),gfunc.free()));
        return loopn(i, o.sfunc.domain(), assign(nth(y,sfunc.at(i)), nth(x, gfunc.at(i))));
    end,

    SUM := (self, o, y, x, opts) >> chain(List(o.children(), c -> self(c, y, x, opts))),

    ISum := (self, o, y, x, opts) >> let(
        myloop := When(IsSymbolic(o.domain), loopn, loop),
        myloop(o.var, o.domain,
            self(o.child(1), y, x, opts))),

    JamISum := (self, o, y, x, opts) >> let(its := EvalScalar(o.domain),
        Cond(IsSymbolic(its), 
                 self.ISum(o, y, x, opts),            
             its > 32, Error("<o.domain> is too big (> 32), probably something went wrong. ",
                             "If you know what you are doing, then change DefaultCodegen.JamISum"),
             let(s := o.child(1),
                 bodies := List([0..its-1], j ->
                     Compile(self(SubstVars(Copy(s), rec((o.var.id):= V(j))), y, x, opts), opts)),
                 self._interleave(bodies)))),

    # NOTE: get rid of _acc
    SUMAcc := (self, o, y, x, opts) >> let(ii := Ind(),
        When(not Same(ObjId(o.child(1)), Gath),  # NOTE: come up with a general condition
            chain(
                loop(ii, Rows(o), assign(nth(y, ii), V(0))),
                List(o.children(), c -> self._acc(self(c, y, x, opts), y))),
            chain(
                self(o.child(1), y, x, opts),
                List(Drop(o.children(), 1), c -> self._acc(self(c, y, x, opts), y))))),

    ISumAcc := (self, o, y, x, opts) >> let(ii := Ind(), chain(
            loop(ii, Rows(o), assign(nth(y, ii), V(0))),
            loop(o.var, o.domain, self._acc(self(o.child(1), y, x, opts), y)))),

    ScatAcc := (self, o, y, x, opts) >> self._acc(self(Scat(o.func),y,x,opts),y),

    # _composePropagate(<ch>, <y>, <x>, <mfunc>) - <x> and <y> and temp arrays propagation through 
    # identity/inplace operators and creating intermediate arrays using <mfunc> which
    # has (c) -> ... signature (<c> is child operator). There is no assumption made on what is <x>, 
    # <y> and <mfunc> return so this function can be used to propagate other information same way 
    # as arrays in composition.

    _composePropagate := function(ch, y, x, mfunc)
        local numch, vecs, i, j, cmd, code, isI, crossCh, apos;
        numch := Length(ch);
        vecs  := [y];

        isI   := (x) -> IsIdentitySPL(x) or x.isInplace();

        # we like to evaluate right (last) to left (first), where the values
        # in parens refer to the position of the elements in the array.
        #
        # So here, we start at the output (first entry in array) and
        # walk towards the input, creating temporary arrays as necessary
        
        for i in [1..numch-1] do
            vecs[i+1] := When(isI(ch[i]),
                vecs[i],
                mfunc(ch[i])
            );
            if ObjId(ch[i]) = Cross then
                apos := [ 1, 1 ]; # apos holds starting input/output indexes 
                                  # as children's arity on input may be different from output arity.
                crossCh := ch[i].rChildren();
                for j in [1 .. Length(crossCh)] do
                    if isI(crossCh[j]) then
                        vecs[i+1][apos[2]] := vecs[i][apos[1]];
                    fi;
                    apos := apos + crossCh[j].arity();
                od;
            fi;
        od;

        # the last entry must be the input.
        vecs[numch+1] := x;

        # now we walk in the opposite direction, copying through
        # input as far as we can.
        # if there is a cross and one of the inputs is an identity,
        # there's no point actually doing it
        for i in Reversed([1..numch]) do
            if isI(ch[i]) then
                vecs[i] := vecs[i+1];
            elif ObjId(ch[i]) = Cross then
                apos := [ 1, 1 ]; # apos holds starting input/output indexes 
                                  # as children's arity on input may be different from output arity.
                crossCh := ch[i].rChildren();
                for j in [1 .. Length(crossCh)] do
                    if isI(crossCh[j]) then
                        vecs[i][apos[1]] := vecs[i+1][apos[2]];
                    fi;
                    apos := apos + crossCh[j].arity();
                od;
            fi;
        od;

        # If all children were evaluated inplace, the output will be in x
        # Make it go from x -> y as expected
        if vecs[1] = vecs[numch+1] then vecs[1] := y; fi;
        if vecs[1] = vecs[numch+1] then vecs[numch+1] := x; fi;

        return vecs;
    end,

    Compose := meth(self, o,y,x,opts)
        local ch, numch, vecs;
        ch    := o.children();
        numch := Length(ch);

        # propagate x and y arrays and create temporary arrays
        vecs  := self._composePropagate(ch, y, x, c -> TempArray(y,x,c));

        # order them so that first to be evaluated is first in array.
        vecs := Reversed(vecs);
        ch   := Reversed(ch);

        # Wrap code in variable declaration. Each entry in vecs will contain multiple
        # arrays in the case of multi-input/output (i.e. OL)
        return decl(
            Difference(Flat(vecs{[2..Length(vecs)-1]}), Flat([x,y])),
            chain(
                List([1..numch], i ->
                     self(ch[i], vecs[i+1], vecs[i], opts)))
        );
    end,

    # ComposeDists is meant to be like a Compose, but for parallel ISums. We
    # need it for 2 reasons: a) The arrays declared are distributed among the
    # nodes, so we need them to be of size N/p, and not N. b) We can ping-pong
    # between 2 parallel buffers instead of declaring a ton of temp arrays, as
    # long as a barrier-sync exists between the stages. NOTE: This won't work
    # if we do full overlapping with micro-barriers. NOTE: This also might not
    # work too well for partial/dirty-overlap.

    ComposeDists := meth(self, o,y,x,opts)
        local ch, numch, vecs, i, cmd, code, pt1, pt2;
        ch    := o.children();
        numch := Length(ch);

        # order them so that first to be evaluated is first in array.
        ch   := Reversed(ch);

        # NOTE: We assume that the # of procs doesn't change across the ComposeDists!
        pt1 := TempArraySeq(y,x,ch[1]);
        pt2 := TempArraySeq(y,x,ch[1]);

        return(
            decl([pt1, pt2], chain( 
            List([1..numch], i -> 
                let(ppx := When(i mod 2 = 0, pt1, pt2),
                    ppy := When(i mod 2 = 0, pt2, pt1),
                    px  := When(i=1,     x, ppx),
                    py  := When(i=numch, y, ppy),
                    self(ch[i], py, px, opts))
                )
            ))
        );
    end,

    ComposeStreams:= meth(self, o,y,x,opts)

        # This is a compose of 2 or more multibuffered streams. The streams can
        # ping-pong data between X and Y. This is okay only if we're allowed to
        # clobber the input (reasonable to assume only when Inplace is
        # requested). Also, whether the last stage is a ping or a pong will
        # determine where the output is written to (which is not great).
        # Currently, just a hack.

        local ch, numch;
        ch    := o.children();
        numch := Length(ch);

        # order them so that first to be evaluated is first in array.
        ch   := Reversed(ch);

        return chain( List([1..numch], i -> 
            let(ppy := When(i mod 2 = 0, x, y),
                ppx := When(i mod 2 = 0, y, x),
                self(ch[i], ppy, ppx, opts))
            )
        );
    end,


    Inplace := (self, o, y, x, opts) >>
        self(o.child(1), y, x, opts), # Compose will handle these somehow
#MRT: it really should do this... but it doesn't
#        self(o.child(1), x, x, opts),

    Data := meth(self, o, y, x, opts)
        local val;
        o.var.isData := true;
        val := When(IsFunction(o.value), o.value.tolist(), o.value);
        val := When(IsValue(val), val, o.var.t.value(val));
        return data(o.var, val, self(o.child(1), y, x, opts));
    end,

    #   NOTE: use pointers to do pingpong?
    ICompose := (self, o, y, x, opts) >> let(
        t      := Dat1d(x.t.t, Rows(o)),
        its    := o.domain,
        newind := Ind( Int((its-1)/2) ),
        # if orig loops has even # iterations, we peel 2 iterations, otherwise peel 1
        peel_its := Cond(IsEvenInt(o.domain), 2, 1),

        decl([t], chain(
           Cond(IsOddInt(o.domain),
                SubstVars(Copy(self(o.child(1), y, x, opts)), tab((o.var.id) := (V(its-1)))),

                chain(
                   SubstVars(Copy(self(o.child(1), t, x, opts)), tab((o.var.id) := (V(its-1)))),
                   SubstVars(Copy(self(o.child(1), y, t, opts)), tab((o.var.id) := (V(its-2)))))),

           When(o.domain <= 2, [],
               loop(newind, newind.range,
                   chain(
                       SubstVars(Copy(self(o.child(1), t, y, opts)), tab((o.var.id) := (its-1-peel_its)-2*newind)),
                       SubstVars(Copy(self(o.child(1), y, t, opts)), tab((o.var.id) := (its-2-peel_its)-2*newind)))))
       ))),

    Multiplication:= meth(self, o, y, x, opts)
        local iterator;

        iterator:=Ind();
        return loop(iterator, [ 0 .. o.element[2]-1 ],
            assign(nth(StripList(y), iterator), mul(nth(x[1], iterator),nth(x[2], iterator))));
    end,

    OLMultiplication := meth(self, o, y, x, opts)
        local iterator;
        iterator:=Ind();
        return loop(iterator, [ 0 .. o.rChildren()[2]-1 ],
            assign(nth(StripList(y), iterator),
                ApplyFunc(mul, List([1..o.rChildren()[1]], i -> nth(x[i], iterator)))));
    end,

    OLConjMultiplication := meth(self, o, y, x, opts)
        local iterator;
        iterator:=Ind();
        return loop(iterator, [ 0 .. o.rChildren()[2]-1 ],
            assign(nth(StripList(y), iterator),
                ApplyFunc(mul, [nth(x[1], iterator)] :: List([2..o.rChildren()[1]], i -> conj(nth(x[i], iterator))))));
    end,

    __RCOLMultiplication := (self, o, y, x, conj) >> let(
        i  := Ind(),
        n  := o.rChildren()[2], 
        m  := o.rChildren()[1],
        yy := StripList(y),
        re := List([1..m], e -> var.fresh_t("re", yy.t.t)),
        im := List([1..m], e -> var.fresh_t("re", yy.t.t)),
        loop(i, n, decl( re :: im, chain(
            assign( re[1], nth(x[1], 2*i) ),
            assign( im[1], nth(x[1], 2*i+1) ),
            chain( List( [2..m], j -> 
                chain(
                    assign(re[j], re[j-1] * nth(x[j],2*i) - conj * im[j-1] * nth(x[j],2*i+1)),
                    assign(im[j], im[j-1] * nth(x[j],2*i) + conj * re[j-1] * nth(x[j],2*i+1))
                ))),
            assign(nth(yy,2*i),   re[m]),
            assign(nth(yy,2*i+1), im[m]))))),

    RCOLMultiplication     := (self, o, y, x, conj) >> self.__RCOLMultiplication(o, y, x, 1),
    RCOLConjMultiplication := (self, o, y, x, conj) >> self.__RCOLMultiplication(o, y, x, -1),


    OLDup := (self, o, y, x, opts) >> let(
        i := Ind(o.params[2]),
        loop(i, i.range, chain( 
            List( Flat([y]), yy -> assign(nth(yy, i), nth(x, i)))
        ))
    ),

    SMAP := (self, o, y, x, opts) >> assign(nth(y, 0), o.at(List([1..Cols(o)], i -> nth(x, i-1)))),

    ParSeqWrap := (self, o, y, x, opts) >> let( yy := Flat([y]), xx := Flat([x]),
        self( o.p.child(o.ci),
            StripList(yy :: o.p.filtSUMR(o.y)),
            StripList(xx :: o.p.filtSUMR(o.x)),
            opts)),

    ParSeq := (self, o, y, x, opts) >> let( ch := o.children(), yy := Flat([y]), xx := Flat([x]),
        self( Compose(List([1..Length(ch)], i -> ParSeqWrap(o, i, yy, xx))),
            StripList(o.filtCompR(yy)), StripList(o.filtCompL(xx)), opts)),

    IParSeq := (self, o, y, x, opts) >> let(
            its := Ind(o.domain),
            xs  := o.filtSUML(Flat([x])),
            ys  := o.filtSUMR(Flat([y])),
            xc  := o.filtCompL(Flat([x])),
            yc  := o.filtCompR(Flat([y])),

            rc  := o.filtCompR(Flat([Rows(o)])),
            tc  := List(Zip2(yc, rc), a -> Dat1d(a[1].t.t, a[2])),

            src := List( xc, e -> var.fresh_t("pX", TPtr(e.t.t))),
            dst := List( yc, e -> var.fresh_t("pY", TPtr(e.t.t))),

            ptr := (p) -> When(IsPtrT(p.t), p, nth(p, 0).toPtr(p.t.t)),

            decl(src :: dst :: tc, chain(chain(
                List( TransposedMat([src, xc]),
                    e -> assign(e[1], ptr(e[2])) ) ::
                List( TransposedMat([dst, tc, yc]),
                    e -> assign(e[1], cond( eq(imod(o.domain, 2), 0), ptr(e[2]), ptr(e[3]))) )),
                loopn( its, its.range, chain(
                    self( SubstVars(Copy(o.child(1)), tab((o.var.id) := its)), StripList(dst :: ys), StripList(src :: xs), opts ),
                    chain(
                        List( TransposedMat([src, dst]),
                            e -> assign(e[1], e[2])) :: 
                        List( TransposedMat([dst, yc, tc]),
                            e -> assign(e[1], cond( eq(imod(add(o.domain, its), 2), 0), ptr(e[2]), ptr(e[3]) )))
                )))
            ))
    ),

    Cvt := (self, o, y, x, opts) >> o.params[1].code(y, x, opts),
));

StackAllocsToPtrs := function(icode, x, size, vars)
    local arrayvars, replvars, offset, i;

    # extract all temp array definitions
    arrayvars := Flat(List(
        Collect(icode, @(1, decl, e -> ForAny(e.vars, i -> ObjId(i.t) = TArray))),
        f -> Filtered(f.vars, g -> ObjId(g.t) = TArray)
    ));

    # build a set of replacement pointers for these variables
    replvars := List(arrayvars, e -> var.fresh_t("T", TPtr(e.t.t)));

    offset := 2*size;

    for i in [1..Length(arrayvars)] do

        # remove declaration of array
        icode := SubstTopDown(icode, @(1, decl, e -> arrayvars[i] in e.vars), ee -> ee.cmd);

        # change variables from TArray -> TPtr
        icode := SubstTopDown(icode, arrayvars[i], e -> replvars[i]);

        # prepend declaration and setup of ptr
        icode := decl(replvars[i], chain(
            assign(replvars[i], add(x, offset)),
            icode
        ));

        offset := offset + size;
    od;

    Append(vars, arrayvars);

    return icode;
end;

Class(SingleAllocCodegen, DefaultCodegen, rec(
    Formula := meth(self, o, y, x, opts)
        local icode, datas, prog, params, sub, initsub, io, vars, size;
        
        o := SumsUnification(o.child(1), opts);

        [x, y] := self.initXY(x, y, opts);

        size := Maximum(o.dimensions);
        params := Set(Collect(o, param));

        datas := Collect(o, FDataOfs);
        o := BlockSums(opts.globalUnrolling, o);
        icode := self(o, y, x, opts);
        icode := RemoveAssignAcc(icode);
        icode := BlockUnroll(icode, opts);

        # replace all stack allocated arrays with offsets into input array
        vars := [];
        icode := StackAllocsToPtrs(icode, x, size, vars);

        # icode := PowerOpt(icode);
        icode := DeclareHidden(icode);
        if IsBound(opts.isFixedPoint) and opts.isFixedPoint then
            icode := FixedPointCode(icode, opts.bits, opts.fracbits);
        fi;

        io := When(x=y, [x], [y, x]);
        sub := Cond(IsBound(opts.subName), opts.subName, "transform");
        initsub := Cond(IsBound(opts.subName), Concat("init_", opts.subName), "init");
        icode := func(TVoid, sub, Concatenation(io, params), icode);

        if IsBound(opts.generateInitFunc) and opts.generateInitFunc then
            prog := program(
                decl(List(datas, x->x.var),
                    chain(
                        func(TVoid, initsub, params, chain(List(datas, x -> SReduce(x.var.init)))),
                        icode
                    )));
        else
            prog := program( func(TVoid, initsub, params, chain()), icode);
        fi;
        prog.dimensions := o.dimensions;
        return prog;
    end,
));

Class(RecCodegenMixin, rec(
    RecursStep := (self, o, y, x, opts) >> let(
        name := spiral.libgen.CodeletName(spiral.libgen.CodeletShape(o.child(1))),
        ApplyFunc(call, Concatenation(
                [ var(name), y + o.yofs, x + o.xofs ],
                spiral.libgen.CodeletParams(o.child(1))))),

    RecursStepCall := (self, o, y, x, opts) >>
        ApplyFunc(call, Concatenation(Flat([var(o.func), y, x,]), List(o.bindings, x->x[2]))),

    Codelet := meth(self, o, y, x, opts)
        local code;
        [x, y] := self.initXY(x, y, opts);
        o := o.child(1);
        ## Generating code : main body
        o := BlockSums(opts.libgen.basesUnrolling, o);
        code := SReduce(self(o, y, x, opts), opts);
        code := BlockUnroll(RemoveAssignAcc(code), opts);
        code := DeclareHidden(code);
        return code;
    end,
));


