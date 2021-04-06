
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F Replaces RemoteData inside a BB with RemoteDataUnrolled
UnrollRemoteData := function(e, cx)
   local extvars, loopvar, i, unrolledLoopVars, BBLoopVars, unrolledRange, newobj, newvar, newFofs, newofs, newfdataofs, f;

   # Compute list of loopvars that will be unrolled (ones inside BB)
   # These are all loops under the BB
   i := Length(cx.parents);
   unrolledLoopVars := [];
   unrolledRange := 1;
   while (ObjId(cx.parents[i])<>MultiBufISum and ObjId(cx.parents[i]) <> MultiBufISumFinal) do
      i := i-1;
      if ObjId(cx.parents[i])=ISum then
         unrolledLoopVars := unrolledLoopVars :: [cx.parents[i].var];
      fi;
   od;

   #Error("BP");

  # Take intersection because twiddle might not involve all loops under BB.
  BBLoopVars := Intersection(e.ofs.free(), unrolledLoopVars);
  unrolledRange := ListProduct(List(unrolledLoopVars, i->i.range));
  unrolledRange := When(unrolledRange=0, 1, unrolledRange);

  #NOTE: why was this an issue?
  #if BBLoopVars = [] then
  #   PrintLine("------------------> Going ahead with no Isum inside BB");
  #fi;

  # a) return RemoteDataUnrolled
  # b) with FDataOfs_mbuf that has a new range

  #Error("BP");

  newobj := e.child(1);    # This is the VRCDiag or similar obj


  # Change the FDataOfs_mbuf to include a different length and offset


  # Offset for FDataOfs_mbuf is RemoteData.ofs, with freevars that are not inside the BB set to zero.
  newFofs := Copy(e.ofs);

  extvars := newFofs.free();
  SubtractSet(extvars, BBLoopVars);

  for loopvar in extvars do
     SubstVars(newFofs, rec( (loopvar.id) := 0));
  od;

  newobj := SubstTopDown(newobj, FDataOfs_mbuf, f->FDataOfs_mbuf(f.var, f.len*unrolledRange, newFofs));

  e.var.t.size    := e.var.t.size    * unrolledRange;
  e.altbuf.t.size := e.altbuf.t.size * unrolledRange;

  # Must manually set unrolled loopvars to zero here because this will go to
  # init, and won't be taken care of by unrolling
  newofs := e.ofs;
  for loopvar in BBLoopVars do
     SubstVars(newofs, rec( (loopvar.id) := V(0) ));
  od;

  return(RemoteDataUnrolled(e.var, e.altbuf, e.value, e.ofs, newobj));
end;


Class(MBufCodegen, DistCodegen, rec(

    RemoteDataInit := meth(self, o, y, x, opts)
    #-----------------------------------------------------------------------------
        local r;
        r := Collect(o, RemoteDataUnrolled);
        # NOTE: handle multiple RemoteDataUnrolleds here.

        if Length(r) > 1 then
           Error("ERROR: Multi RemoteDataInit case NOT implemented!");
        fi;
        r := r[1];
        return(chain(
          self( RemoteDataNoBody(r.var, r.value, r.ofs, r.child(1)), y, x, opts ),
          self(o.child(1), y, x, opts)
        ));
    end,

    # This is called only from the BB body. So codegen only o.child(1)
    RemoteDataUnrolled := (self, o, y, x, opts) >> self( o.child(1), y, x, opts ),
    #-----------------------------------------------------------------------------

    # This is called in situations where there is no unrolled code.
    RemoteDataNoBody := meth(self, o, y, x, opts)
    #-----------------------------------------------------------------------------

      local pkSizeInBytes, dmaCommand;

      dmaCommand := var("GATHMEM_DIAG");
      pkSizeInBytes := o.child(1).element.len * opts.vector.vlen * (opts.vector.isa.bits/8);

      if pkSizeInBytes > 16384 then
         if pkSizeInBytes = 32768 then
            dmaCommand := var("GATHMEM_DIAG_32K");
         elif pkSizeInBytes = 65536 then
            dmaCommand := var("GATHMEM_DIAG_64K");
         else
           Error("Too large a packet size for getting twiddles");
         fi;
      fi;
     return(chain(
       call(dmaCommand,
       o.altbuf.id,    # Destination base address
       0,              # Destination offset
       "spe_info."::o.value.id,     # Source base address
       o.ofs*opts.vector.vlen,       # Source offset
       o.child(1).element.len * opts.vector.vlen # Packet size in elements
       )

       # Only if 

       #call(var("spu_writech"),
       #"MFC_WrTagMask",
       #"1 << 3"),
       #
       #call(var("spu_mfcstat"),
       #"MFC_TAG_UPDATE_ALL"
       #)
     ));
    end,


    # # These should've gotten converted to RemoteDataUnrolled and RemoteDataInit
    # RemoteData := meth(self, o, y, x, opts)
    #     Error("# These should've gotten converted to RemoteDataUnrolled and RemoteDataInit");
    # end,


    # These should've gotten converted to RemoteDataUnrolled and RemoteDataInit
    # But during DP, these might not always be contained in a MultiBufISum. In
    # this case, we should ideally add a new MultiBufISum. Here's a hack instead:
    #RemoteData := (self, o, y, x, opts) >> self(o.child(1), y, x, opts),
    RemoteData := (self, o, y, x, opts) >> skip(),

#-----------------------------------------------------------------------------
    Formula := meth(self, o, y, x, opts)
#-----------------------------------------------------------------------------
#NOTE: Formula needs major cleanup Tue 31 Mar 2009 12:58:40 AM EDT
        local icode, datas, datas_ppe, prog, params, sub, initsub, initsub_ppe, io, e, d;

        [x, y] := self.initXY(x, y, opts);

        o :=  Process_fPrecompute(o, opts);

        o := o.child(1);
        params := Set(Concatenation(Collect(o, param), Filtered(Collect(o, var), IsParallelLoopIndex)));


        SubstTopDown(o, @@(1, VRCDiag,
            (e,cx)->(ObjId(e.element)=FDataOfs
                     and e.element.var.t.size * opts.vector.vlen >= opts.maxSPETwiddles
                     #and IsBound(cx.MultiBufISum)
                     #and IsBound(cx.BB)
                     )
            ),
          e->let(v := @@(1).val,
                 f     := v.element,
                 dmbuf     := var.fresh_t("Tmbuf",  TArray(f.var.t.t, f.len)),
                 dmbuf_alt := var.fresh_t("TmbufAlt", TArray(f.var.t.t, f.len)),
                 RemoteData(dmbuf, dmbuf_alt, f.var, f.ofs, VRCDiag(FDataOfs_mbuf(dmbuf, f.len, 0), v.v))
              )
        );


        # Replace FDataOfs that won't fit inside an SPE with FDataOfs_mbuf
        #SubstTopDown(o, @@(1, FDataOfs,
        #    (e,cx)->(e.var.t.size*opts.vector.vlen >= opts.maxSPETwiddles
        #             #and IsBound(cx.MultiBufISum)
        #             #and IsBound(cx.BB)
        #             )
        #    ),
        #e->let(f:=@@(1).val, FDataOfs_mbuf(f.rChildren()[1], f.rChildren()[2], f.rChildren()[3]))
        #);

        datas := Collect(o, FDataOfs);

        datas_ppe := Collect(o, RemoteData);

        #datas_ppe := [];
        #for d in datas do
        #   if d.var.t.size * opts.vector.vlen >= opts.maxSPETwiddles then
        #      datas_ppe := datas_ppe :: [d];
        #      datas := RemoveList(datas, d);
        #   fi;
        #od;

        # Replace RemoteData with RemoteDataUnroll where necessary
        #NOTE: Checking for context is not enough: must check for length
        #SubstTopDown(o, @@(1, RemoteData, (e,cx)->IsBound(cx.BB)), UnrollRemoteData);


        #SubstTopDown(o, @@(1, RemoteData, (e,cx)->(IsBound(cx.MultiBufISum) or IsBound(cx.MultiBufISumFinal))),
        #    UnrollRemoteData);

        SubstTopDown(o, @@(1, RemoteData, (e,cx)->IsBound(cx.MultiBufISum) or IsBound(cx.MultiBufISumFinal)),
            UnrollRemoteData);

        # Enclose BBs within a RemoteDataInit if needed (if the BB contains one or more
        # RemoteDataUnrolled's)
        #SubstTopDown(o, @@(1, MultiBufISum, (e,cx)->(
        #    Length(Collect(e, RemoteDataUnrolled))>=1
        #    and (not IsBound(cx.RemoteDataInit) or Length(cx.RemoteDataInit) = 0)
        #  )),
        #  f->RemoteDataInit(f)
        #);


        #return(o);

        o := BlockSumsOpts(o, opts);
        icode := self(o, y, x, opts);
        icode := ESReduce(icode, opts);
        icode := RemoveAssignAcc(icode);
        icode := BlockUnroll(icode, opts);
        # icode := PowerOpt(icode);
        icode := DeclareHidden(icode);
        if IsBound(opts.isFixedPoint) and opts.isFixedPoint then
            icode := FixedPointCode(icode, opts.bits, opts.fracbits);
        fi;

        io := When(x=y, [x], [y, x]);
        sub := Cond(IsBound(opts.subName), opts.subName, "transform");
        initsub := Cond(IsBound(opts.subName), Concat("init_", opts.subName), "init");
        initsub_ppe := Cond(IsBound(opts.subName), Concat("init_ppe_", opts.subName), "init_ppe");
        icode := func(TVoid, sub, Concatenation(io, params), icode);

        #Error("BP");

        if IsBound(opts.generateInitFunc) and opts.generateInitFunc then
            prog := program(
                #decl(List(datas, x->x.var)::List(datas_ppe, x->x.var),
                decl(List(datas, x->x.var),
                    chain(
                        func(    TVoid, initsub,     params, chain(List(datas,     x -> SReduce(x.var.init, opts)))),
                        func_ppe(TVoid, initsub_ppe, params, decl(List(datas_ppe, x->x.value), chain(List(datas_ppe, x -> SReduce(x.value.init, opts)))) ),
                        icode
                    )));
        else
            prog := program( func(TVoid, initsub, params, chain()), icode);
        fi;

        # FF: I really don't know why suddenly AVX_8x32f requires me to do that !!
        if IsBound(opts.vector.isa.fixProblems) then prog := opts.vector.isa.fixProblems(prog, opts); fi;

        prog.dimensions := o.dims();

        # Perform software pipelining
        # prog := MarkPreds(prog);
        # prog := MarkDefUse(prog);
        # SubstTopDown(prog, @(1, loop, e->(Length(Collect(e, loop))=1 and Length(e.range) >= 4)),
        #     e->let(l := @(1).val, loop_sw(l.rChildren()[1], l.rChildren()[2], l.rChildren()[3]) )
        # );
        # SubstTopDown(prog, loop_sw, e->SoftwarePipeline(e));

        return prog;
    end,



    GathMem := meth(self, o, y, x, opts)
        #Gath: loop(i, o.func.domain(), assign(nth(y,i), nth(x, func.at(i))));
        #DO_DMA(Xalt, spe_info.X, IDEAL_DMA_SIZE_BYTES, MFC_GET_CMD); // Xalt = spe_info.X
        local func, pkSize, i, numPkts, pkSizeInBytes, dmaCommand;
        func        := o.func.lambda();
        pkSize      := o.pkSize;
        numPkts     := o.func.domain();
        #i          := 0;
        i           := Ind(numPkts);


        dmaCommand := var("GATHMEM_GET");
        pkSizeInBytes := o.pkSize * (opts.vector.isa.bits/8);

        #NOTE: 16384 is the largest packet we can send on the Cell
        # If we're not using tables, then we can go higher using backend macros
        if (pkSizeInBytes > 16384) then
          # NOTE: How to handle cases where pkSize is greater than 16k?
          if (pkSizeInBytes = 32768) then
             dmaCommand := var("GATHMEM_GET_32K");
          elif (pkSizeInBytes = 65536) then
             dmaCommand := var("GATHMEM_GET_64K");
          elif (pkSizeInBytes = 131072) then
             dmaCommand := var("GATHMEM_GET_128K");
          else
          Error("Selected packet size - ", o.pkSize, " - is higher than the Cell's max allowed (using macros).");
          fi;
       fi;

        # NOTE: remove this after testing
        if (numPkts <= 16 ) then
            #16 is the size of the SPU's DMA Command queue. Exceeding this
            #means taking a huge performance loss because in effect, DMAs won't
            #be done in the background.
            # NOTE: this (16) is hardcoded for the implementation than for the ISA
            return(loop(i, numPkts, call(
                   dmaCommand,
                   #y,
                   var(Concatenation(y.id, "alt")),
                   i*pkSize,
                   var(Concatenation("spe_info.", y.id)),
                   func.at(i)*pkSize,
                   pkSize
                )));
        fi;

        if (numPkts > 2048) then
          Error("Selected attemping to DMA more than 2048 packets at one time.  The Cell architecture has no facility for this. (Doing multiple sets is possible, but in most cases, a better algorithm is needed).");
        fi;

        #NOTE: 16384 is the largest packet we can send on the Cell using DMA tables
        if (pkSizeInBytes > 16384) then
          Error("Selected packet size - ", o.pkSize, " - is higher than the Cell's max allowed.");
        fi;


        return(chain(
            # Build DMA list
            loop(i, numPkts, chain(
              call(var("GATH_LIST_SETSIZE"), pkSize, i.id),
              call(var("GATH_LIST_SETADDR"), func.at(i)*pkSize, i.id, var(Concatenation("spe_info.", y.id)))
            )),

            # Execute DMA list
            call(var("GATHMEM_LIST"),
                var(Concatenation(y.id, "alt")),
                var(Concatenation("spe_info.", y.id)),
                numPkts
            )
        ));


    end,

    ScatMem := meth(self, o, y, x, opts)
        #Scat: return loop(i, o.func.domain(), assign(nth(y,func.at(i)), nth(x, i)));
        local func, pkSize, i, numPkts, pkSizeInBytes, dmaCommand;
        func          := o.func.lambda();
        pkSize        := o.pkSize;
        numPkts     := o.func.domain();
        #i          := 0;
        i           := Ind(numPkts);

        pkSizeInBytes := o.pkSize * (opts.vector.isa.bits/8);

        #NOTE: 16384 is the largest packet we can send on the Cell
        # If we're not using tables, then we can go higher using backend macros
        dmaCommand := var("SCATMEM_PUT");
        pkSizeInBytes := o.pkSize * (opts.vector.isa.bits/8);

        #NOTE: 16384 is the largest packet we can send on the Cell
        # If we're not using tables, then we can go higher using backend macros
        if (pkSizeInBytes > 16384) then
          # NOTE: How to handle cases where pkSize is greater than 16k?

          if (pkSizeInBytes = 32768) then
             dmaCommand := var("SCATMEM_PUT_32K");
          elif (pkSizeInBytes = 65536) then
             dmaCommand := var("SCATMEM_PUT_64K");
          elif (pkSizeInBytes = 131072) then
             dmaCommand := var("SCATMEM_PUT_128K");
          else
          Error("Selected packet size - ", o.pkSize, " - is higher than the Cell's max allowed (using macros).");
          fi;
        fi;

        #DO_DMA(Yalt, spe_info.Y, IDEAL_DMA_SIZE_BYTES, MFC_PUT_CMD); // spe_info.Y = Yalt

        if (numPkts <= 16 ) then
            return(loop(i, numPkts, call(
                dmaCommand,
                #x,
                var(Concatenation(x.id, "alt")),
                i*pkSize,
                var(Concatenation("spe_info.", x.id)),
                func.at(i)*pkSize,
                pkSize
            )));
        fi;

        if (numPkts > 2048) then
          Error("Selected attemping to DMA more than 2048 packets at one time. The Cell architecture has no facility for this");
        fi;

        #NOTE: 16384 is the largest packet we can send on the Cell using DMA tables
        if (pkSizeInBytes > 16384) then
          Error("Selected packet size - ", o.pkSize, " - is higher than the Cell's max allowed for DMA lists.");
        fi;


        return(chain(
            # Build DMA list
            loop(i, numPkts, chain(
              call(var("SCAT_LIST_SETSIZE"), pkSize, i.id),
              call(var("SCAT_LIST_SETADDR"), func.at(i)*pkSize, i.id, var(Concatenation("spe_info.", x.id)))
            )),

            # Execute DMA list
            call(var("SCATMEM_LIST"),
                var(Concatenation(x.id, "alt")),
                var(Concatenation("spe_info.", x.id)),
                numPkts
            )
        ));

    end,

     R2Sum := (self, o, y, x, opts) >> 
            loop(o.var1, o.domain1, loop(o.var2, o.domain2, self(o.child(1), y, x, opts))),


    # Note: to perform multibuffering, x,y are reversed in some of the following calls.
    MultiBufISum := meth(self, o, y, x, opts)
       local r, mbufvars, mloop, loopbody;

        r := Collect(o, RemoteDataUnrolled);

        # NOTE: handle multiple RemoteDataUnrolleds here.
        if Length(r) > 1 then Error("ERROR: Multi RemoteDataInit case NOT implemented!"); fi;

        mbufvars := [];
        if Length(r) > 0 then
           r := r[1];
           mbufvars := [r.var]::[r.altbuf];
        fi;

        mloop := multibuffer_loop;
        if IsBound(opts.nombuf) and opts.nombuf=true then mloop := mem_loop; fi;

#loopvar, range, y, x, gathmem, twiddles, bufs, cmd, scatmem)
        return(decl(mbufvars,
          mloop(o.var, o.domain, y, x,
               self(o.gathmem, x, y, opts),
               When(r=[], [], self( RemoteDataNoBody(r.var, r.altbuf, r.value, r.ofs, r.child(1)), y, x, opts )),
               mbufvars,
               self(o.child(1), y, x, opts),
               self(o.scatmem, x, y, opts)
          )
        ));
    end,

    MultiBufISumFinal := (self, o, y, x, opts) >>
        self(MultiBufISum(o.var, o.domain, o.scatmem, o.child(1), o.gathmem), y, x, opts),

    #HACK: this won't work if twiddles need to be streamed.
    MemISum := (self, o, y, x, opts) >> 
       mem_loop(o.var, o.domain, y, x,
            self(o.gathmem, x, y, opts),
            [],
            [],
            self(o.child(1), y, x, opts),
            self(o.scatmem, x, y, opts)
       ),

    MemISumFinal := (self, o, y, x, opts) >>
       mem_loop(o.var, o.domain, y, x,
            self(o.gathmem, x, y, opts),
            self(o.child(1), y, x, opts),
            self(o.scatmem, x, y, opts)
       ),


    MultiBufDistSum := (self, o, y, x, opts) >>
        self(MultiBufISum(o.var, o.domain, o.scatmem, o.child(1), o.gathmem), y, x, opts)
));
