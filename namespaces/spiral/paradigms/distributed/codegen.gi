
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


## Hackity.
#Class(fcall_addr, fcall, rec(
#        computeType := self >> self.args[3].t
#));

# A dist_loop is a parallel loop with domain=#spus

Class(dist_loop, loop_base, rec(
   __call__ := meth(self, P, loopvar, range, cmd) 
       local result;
       Constraint(IsVar(loopvar)); 
       Constraint(IsCommand(cmd)); 
       range := toRange(range);

       loopvar.setRange(range);
       #loopvar.isLoopIndex := true;
       return WithBases(self, rec(
           operations := CmdOps, 
           P := P, 
           var := loopvar,
           cmd := cmd, 
           range := listRange(range)
       ));
   end,

   rChildren := self >> [self.P, self.var, self.range, self.cmd],
   rSetChild := rSetChildFields("P", "var", "range", "cmd"),

   #rChildren := self >> [self.P, self.var, self.cmd],
   #rSetChild := rSetChildFields("P", "var", "cmd", "range"),
   #from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), [rch[1], rch[2], self.range, rch[3]]),

   print := (self, i, is) >> Print(self.name, "(", self.P, ", ", self.var, ", ",
       Blanks(i+is),
       self.cmd.print(i+is, is),
       Print("\n", Blanks(i), ")")),


   #NOTE: This should go in loop_bases, which should free self.vars only if it's bound.
   free := meth(self) local c;
       c := self.cmd.free();
       return c;
   end

));

Class(DistCodegen, VectorCodegen, rec(
    # Wed 16 Jul 2008 07:17:33 PM EDT
    #NOTE: commenting this out! Check to see if this is now equivalent to DefaultCodegen.Formula.
    # This had BlockSums() commented out. Why?
  
    #NOTE: Need to find a way (hack?) to callback DefaultCodegen's formula with one extra stage, etc.
    #Formula := meth(self, o, y, x, opts)
    #    local icode, datas, prog, params, sub, initsub, io;
    #    if IsBound(opts.XType) then x.t := TPtr(opts.XType); fi;
    #    if IsBound(opts.YType) then y.t := TPtr(opts.YType); fi;

    #    o := o.child(1);
    #    params := Set(Collect(o, param));

    #    datas := Collect(o, FDataOfs);
    #    #o := BlockSums(opts.globalUnrolling, o);
    #    icode := self(o, y, x, opts);
    #    icode := RemoveAssignAcc(icode);
    #    icode := BlockUnroll(icode, opts);
    #    # icode := PowerOpt(icode);
    #    icode := DeclareHidden(icode);
    #    # icode := InsertBarriers(icode);
    #    if IsBound(opts.isFixedPoint) and opts.isFixedPoint then
    #        icode := FixedPointCode(icode, opts.bits, opts.fracbits);
    #    fi;

    #    io := When(X=Y, [X], [Y, X]);
    #    #io := When(IsBound(opts.multibuffer_its) and opts.multibuffer_its > 1,
    #    #                Concatenation(io, [Yprev, Xnext]),
    #    #                io);
    #    sub := Cond(IsBound(opts.subName), opts.subName, "transform");
    #    initsub := Cond(IsBound(opts.subName), Concat("init_", opts.subName), "init");
    #    icode := func(TVoid, sub, Concatenation(io, params), icode);

    #    if IsBound(opts.generateInitFunc) and opts.generateInitFunc then
    #        prog := program(
    #            decl(List(datas, x->x.var),
    #                chain(
    #                    func(TVoid, initsub, params, chain(List(datas, x -> SReduce(x.var.init, opts)))),
    #                    icode
    #                )));
    #    else
    #        prog := program( func(TVoid, initsub, params, chain()), icode);
    #    fi;
    #    prog.dimensions := o.dimensions;
    #    return prog;
    #end,


    # ---------------
    # Container
    # ---------------

#    DContainer := meth(self, o, y, x, opts)
#         local spuid;
#         spuid := var.fresh_t("spuid", TInt);
#         return( decl(
#             spuid, 
#             self(o.child(1), y, x, opts)
#             ));
#      end,

    DContainer := (self, o, y, x, opts) >> self(o.child(1), y, x, opts),

    # ---------------
    # Gather
    # ---------------

    #GathDist := (self, o, y, x, opts) >> self(Gath(fId( (o.N/o.P)*o.pkSize )), y, x, opts),
    GathDist := (self, o, y, x, opts) >> call(var("// GathDist"), y, x),

    # Pass the base pointer of the data to be transferred as a param to the DMA_GET

    GathRecv := meth(self, o, y, x, opts)
        local i, func, rfunc, ix;
        i := Ind(); func := o.func.lambda();

        # Standard Gather
        #return loop(i, o.func.domain()/o.P, assign(nth(y, i), nth(x, func.at(i))));

        # We shouldn't use func.at(i), since we've done ScatSend-side normalizing already
        #return chain(
        #    #dist_barrier(),
        #    # Moving dist_barrier to scatsend for now
        #    call(var("// BLOCK_ON_READ")),
        #    loop(i, o.pkSize*o.func.domain()/o.P, assign(nth(y, i), nth(x, i)))
        #);

        return call(var("// Gath_Recv"), y, x);
    end,

    # ---------------
    # Scatter
    # ---------------



    #ScatDist := (self, o, y, x, opts) >> self(Scat(fId( (o.N/o.P)*o.pkSize )), y, x, opts),
    ScatDist := (self, o, y, x, opts) >> call(var("// ScatDist"), y, x),

    #NOTE: Assuming that Length(Y) is the full size of the transform.


                   #idiv(func.at( (i*pkSize)+(chunkSize*o.i) ), chunkSize), # This is the SPU#
    ScatSend := meth(self, o, y, x, opts)
        local i, func, rfunc, ix, pkSize, numPktsPerSPU;
        func          := o.func.lambda();
        pkSize        := o.pkSize;
        numPktsPerSPU := o.func.domain()/o.P;
        i             := Ind(numPktsPerSPU);

        return chain(loop(i, numPktsPerSPU, chain(                            # Loop over number of packets
           call(var("SCATSEND_PUT"), 
               x,                                                       # Base address of X
               i*pkSize,                                                # Offset of X in elements
               fcall(var("ADDR", TFunc(TInt, TInt, TULongLong)),
                   idiv(func.at((numPktsPerSPU*o.i)+i), numPktsPerSPU), # This is the SPU#
                   y                                                    # Base address of Y
               ),
               mul(
                imod(func.at((numPktsPerSPU*o.i)+i), numPktsPerSPU),    # Offset of Y in elements
                pkSize),
               pkSize                                                   # Size of DMA transfer in elements
           )
        )),
        dist_barrier());

        # Standard Scatter

        # return loop(i, numPktsPerSPU,                                    # Loop over number of packets * pkSize
        #     assign(nth(y, func.at(i)), nth(x, i))
        # );
    end,

    # ---------------
    # Sum
    # ---------------

    # For now, assume loop range = # of spus

    DistSum := (self, o, y, x, opts) >> let(
        dist_loop(o.P, o.var, o.domain, self(o.child(1), y, x, opts))
        ),

    DistSumLoop := (self, o, y, x, opts) >> chain(
        dist_loop(o.P, o.var, o.domain, self(o.child(1), y, x, opts)),
        dist_barrier()
        ),

# Doesn't work for unrolled code.
#   DistSum := (self, o, y, x, opts) >> let(
#       Constraint(o.domain = opts.spus),
#       # If we're GathRecv'ing inside, we must sync before getting into the loop.
#       When(o._children[1]._children[Length(o._children[1]._children)].name = "GathRecv",
#         chain(dist_barrier(), dist_loop(o.P, o.var, o.domain, self(o.child(1), y, x, opts))),
#         dist_loop(o.P, o.var, o.domain, self(o.child(1), y, x, opts))
#       )
#    )



    # PTensor and Comm_Cell are used by ParCellDMP only.
    PTensor := (self, o, y, x, opts) >> self(o.L, y, x, opts),

    # Looped version (NOTE: works, except since we'll never do more than 8 SPUs for now, we always want a fully unrolled version

    Comm_Cell := meth(self, o, y, x, opts)
        local i;
        i := Ind(o.P);

        return(chain(loop(i, o.P,             # Loop over P
            call(var("SCATSEND_PUT"),
                x,                      # Base address of X
                i*o.pkSize,             # Offset of X in elements
                fcall(var("ADDR"),      
                    i,                  # This is the SPU#
                    y                   # Base address of Y
                ),
                fcall(var("SPUID_TIMES"), o.pkSize),             # Offset of Y in elements (MUST BE spuid*pkSize!)
                o.pkSize                # Size of DMA transfer in elements
                )
        ),
        dist_barrier()
        ));
    end

));

