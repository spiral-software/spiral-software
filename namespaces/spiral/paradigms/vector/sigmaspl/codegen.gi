
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(VectorCodegen, DefaultCodegen, rec(

    Formula := meth(self, o, y, x, opts)
        local icode, datas, prog, params, sub, initsub, io, initcode;

        o := SumsUnification(o.child(1), opts);

        [x, y] := self.initXY(x, y, opts);

        o :=  Process_fPrecompute(o, opts);

        params := Set(Filtered(Collect(o, @(1, [param, var])), x -> ObjId(x)=param or IsParallelLoopIndex(x)));

        datas := Filtered(Collect(o, FDataOfs), x->IsBound(x.var.init));
        o := BlockSumsOpts(o, opts);
        icode := self(o, y, x, opts);
        if IsBound(opts.finalSReduce) and opts.finalSReduce then
            icode := SReduce(icode, opts);   #SReduce -> ESReduce sequence is faster than just ESReduce
            icode := ESReduce(icode, opts);
        fi;
        icode := RemoveAssignAcc(icode);
        icode := BlockUnroll(icode, opts);
        icode := DeclareHidden(icode);
        if IsBound(opts.isFixedPoint) and opts.isFixedPoint then
            icode := FixedPointCode(icode, opts.bits, opts.fracbits);
        fi;

        io := When(x=y, [x], [y, x]);
        sub := Cond(IsBound(opts.subName), opts.subName, "transform");
        initsub := Cond(IsBound(opts.subName), Concat("init_", opts.subName), "init");
        icode := func(TVoid, sub, Concatenation(io, params), icode);

        if IsBound(opts.generateInitFunc) and opts.generateInitFunc then
	    initcode := chain(List(datas, x -> SReduce(x.var.init, opts)));
            prog := program(
                decl(List(datas, x->x.var),
                    chain(
                        func(TVoid, initsub, params :: Set(Collect(initcode,param)), initcode),
                        icode
                    )));
        else
            prog := program( func(TVoid, initsub, params, chain()), icode);
        fi;

        # FF: I really don't know why suddenly AVX_8x32f requires me to do that !!
        #if IsBound(opts.vector.isas) then
        #    prog := FoldR(opts.isas, (r, e) -> e.fixProblems(r, opts), prog);
        #fi;
        #if IsBound(opts.vector.isa.fixProblems) then prog := opts.vector.isa.fixProblems(prog, opts); fi;

        prog.dimensions := o.dims();
        if IsBound(opts.finalSReduce) and opts.finalSReduce then
            icode := SReduce(icode, opts);   #SReduce -> ESReduce sequence is faster than just ESReduce
            icode := ESReduce(icode, opts);
        fi;
        return prog;
    end,

###########################################################################

    NoPull := (self, o, y, x, opts) >> self(o.child(1), y, x, opts),

    VContainer := (self, o, y, x, opts) >>
        self(o.child(1), y, x, CopyFields(opts, rec(
            vector := rec(
                isa  := o.isa,
                SIMD := LocalConfig.cpuinfo.SIMDname )))),

    # VirtualPad changes dimensions of the SPL object but does not
    # affect the code
    VirtualPad := (self, o, y, x, opts) >> self(o.child(1), y, x, opts),

    VPrm_x_I := (self, o, y, x, opts) >> self(VTensor(Prm(o.func), o.v), y, x, opts),

    VPerm := (self, o, y, x, opts) >> o.code(y, x), # these codes are generated

    #VTensor for multi inputs
    VTensor := (self, o, y, x, opts) >> let(
        CastToVect := p -> StripList(List(Flat([p]), e -> tcast(TPtr(TVect(opts.vector.isa.t.t, o.vlen)), e))),
        self(o.child(1), CastToVect(y), CastToVect(x), opts)),

    # ---------------
    # Gather
    # ---------------
    VGath := (self, o, y, x, opts) >> Cond(IsUnalignedPtrT(x.t),
        self(VGath_u(fTensor(o.func, fBase(o.v, 0)), o.v), y, x, opts),
        self(VTensor(Gath(o.func), o.v), y, x, opts)),

    VGath_u := (self, o, y, x, opts) >> let(
        j    := Ind(o.func.domain()),
        xofs := o.func.at(j),
        loop(j, j.range,
            opts.vector.isa.loadCont(o.v, y, j, x, xofs, imod(xofs, o.v), opts))),

    VGath_dup := (self, o, y, x, opts) >> let(
        v := o.v, func := o.func.lambda(), j := Ind(o.func.domain()),
        isa := opts.vector.isa,
        loop(j, j.range,
             isa.dupload(vtref(isa.t, y, j), nth(x, func.at(j))))),

    VReplicate := (self, o, y, x, opts) >> let(
	v := o.v, isa := opts.vector.isa,
        chain(List([0..v-1], i -> isa.duploadn(vtref(isa.t, y, i), vtref(isa.t, x, 0), i+1)))),

    _VHAdd := (self,xlist,opts) >> let(
       l := Length(xlist),
       Cond(l=2, opts.vector.isa.hadd(xlist[1], xlist[2]),
	    # else
	         opts.vector.isa.hadd(self._VHAdd(xlist{[1..l/2]}, opts),
		                      self._VHAdd(xlist{[l/2+1..l]}, opts)))),

    VHAdd := (self, o, y, x, opts) >> let(v := o.v, isa := opts.vector.isa,
        assign(vtref(isa.t, y, 0), self._VHAdd(List([0..v-1], i -> vtref(isa.t, x, i)), opts))),

    VGath_zero := (self, o, y, x, opts) >> self(VGath(fAdd(o.N, o.n, 0), o.v), y, x, opts),

    VScat_sv := meth(self, o, y, x, opts)
        local v, sv, func, nv1, rem, ii, id, svstore, isa;
        [v, sv, func] := [o.v, o.sv, o.func.lambda()];
        isa := opts.vector.isa;
        nv1 := idiv(func.domain()*sv, v);
        rem := o.getConstantRem();
        ii  := Ind(nv1);
        id  := 1+Log2Int(sv);
        svstore  := (n, id2, idx) -> isa.svstore[id][id2](
            List([0..n-1], i -> nth(y, sv*func.at(i + (v/sv)*idx))),
            vtref(isa.t, x, idx), opts);
        return chain(
            loop(ii, ii.range, svstore(v/sv, v/sv, ii)),
            When(rem=0, [],    svstore(rem, rem, nv1)));
    end,

    VScat_svAcc := meth(self, o, y, x, opts)
        local v, sv, func, nv1, rem, ii, id, svaccstore, tmp, tmp2, isa;
        [v, sv, func] := [o.v, o.sv, o.func.lambda()];
        nv1 := idiv(func.domain()*sv, v);
        #ESReduce should do the equivalent of an EvalScalar if possible
        #and maybe come up with smarter reductions if not possible
        # rem := ESReduce(imod(func.domain()*sv, v) / sv, opts);  <-- keep this??
        rem := o.getConstantRem();
        isa := opts.vector.isa;
        ii  := Ind(nv1);
        id  := 1+Log2Int(sv);
        tmp := TempVar(isa.t);
        tmp2:= TempVar(isa.t);
        svaccstore  := (n, id2, idx) -> chain(
            isa.svload[id][id2](
                tmp,
                List([0..n-1], i -> nth(y, sv*func.at(i + (v/sv)*idx))), opts),
            assign(tmp2, tmp + vtref(isa.t, x, idx)),
            isa.svstore[id][id2](
                List([0..n-1], i -> nth(y, sv*func.at(i + (v/sv)*idx))),
                tmp2, opts));
        return chain(
            loop(ii, ii.range, svaccstore(v/sv, v/sv, ii)),
            When(rem=0, [],    svaccstore(rem, rem, nv1)));
    end,

    VGath_sv := meth(self, o, y, x, opts)
        local v, sv, func, nv1, rem, ii, id, svload, isa;
        [v, sv, func] := [o.v, o.sv, o.func.lambda()];
        nv1 := idiv(func.domain()*sv, v);
        rem := o.getConstantRem();
        isa := opts.vector.isa;
        ii := Ind(nv1);
        id := 1+Log2Int(o.sv);
        svload  := (n, id2, idx) -> isa.svload[id][id2](
            vtref(isa.t, y, idx),
            List([0..n-1], i -> nth(x, sv*func.at(i + (v/sv)*idx))), opts);
        return chain(
            loop(ii, ii.range, svload(v/sv, v/sv, ii)),
            When(rem=0, [],    svload(rem, rem, nv1)));
    end,

    RCVGath_sv := meth(self, o, y, x, opts)
        local v, sv, func, nv1, res, ii, id, svload, n, isa;
        [v, sv, func] := [o.v, o.sv, fTensor(o.func, fId(2*o.sv)).lambda()];
        n := o.func.domain()*sv;
        nv1 := idiv(n, v);
        res := EvalScalar((n - v*nv1)/sv);
        ii  := Ind(2*nv1);
        id := Log2Int(2*sv)+1;
        isa := opts.vector.isa;
        svload := (n,id2, idx) -> isa.svload[id][id2](
            vtref(isa.t, y, idx),
            List([0..n-1], i -> nth(x, func.at(2*i*sv + v*idx))), opts);
        # FF: NOTE: This strength reduction call should not be necessary. However, looks like at this point SAR requires it
        return RulesMergedStrengthReduce(chain(
            loop(ii, ii.range, svload(v/sv, v/id, ii)),
            Cond(res=0,         [  ],
                 sv*res <= v/2, [ svload(res, res, 2*nv1),
                          assign(vtref(isa.t, y, (2*nv1+1)), isa.t.zero()) ],
                        [ svload(v/sv, v/id, 2*nv1),
                  svload(res-v/(2*sv), res-v/id, 2*nv1+1) ])));
    end,

    RCVScat_sv := meth(self, o, y, x, opts)
        local v, sv, func, nv1, res, ii, id, svstore, n, isa;
        [v, sv, func] := [o.v, o.sv, fTensor(o.func, fId(2*o.sv)).lambda()];
        n := o.func.domain()*sv;
        nv1 := idiv(n, v);
        res := EvalScalar((n - v*nv1)/sv);
        ii  := Ind(2*nv1);
        id := Log2Int(2*sv)+1;
        isa := opts.vector.isa;
        svstore := (n,id2, idx) -> isa.svstore[id][id2](
        List([0..n-1], i -> nth(y, func.at(2*i*sv + v*idx))),
        vtref(isa.t, x, idx), opts);
        return chain(
            loop(ii, ii.range, svstore(v/sv, v/id, ii)),
            Cond(res=0,         [  ],
                 sv*res <= v/2, [ svstore(res, res, 2*nv1) ],
                        [ svstore(v/sv, v/id, 2*nv1),
                  svstore(res-v/(2*sv), res-v/id, 2*nv1+1) ]));
    end,

   VStretchGath := (self, o, y, x, opts) >> let(
       v := o.v,
       func := o.func.lambda(),
       rat := func.domain() / o.part,
       nv1 := idiv(rat, v),
       res := EvalScalar(rat - v*nv1),
       ii := Ind(),
       it := Ind(),
       xbase := rat * it,
       ybase := (_roundup(rat, v) / v) * it,
       isa := opts.vector.isa,
       svload := isa.svload[1][v],

       footer := When(res=0, skip(),
           let(svloadR := isa.svload[1][res],
               svloadR(vtref(isa.t, y, nv1 + ybase), List([0..res-1],
                       i->nth(x, func.at(nv1*v + i + xbase))), opts))),
       loop(it, o.part,
           chain(
               loop(ii, nv1,
                   svload(vtref(isa.t, y, ii + ybase),
                       List([0..v-1], i->nth(x, func.at(v * ii + i + xbase))), opts)),
               footer))
    ),

   vRCStretchGath := (self, o, y, x, opts) >> let(
       v := o.v/2,
       func := o.func.lambda(),
       rat := func.domain() / o.part,
       nv1 := idiv(rat, v),
       res := EvalScalar(rat - v*nv1),
       ii := Ind(),
       it := Ind(),
       xbase := rat * it,
       ybase := (_roundup(rat, v) / v) * it,
       isa := opts.vector.isa,
       svload := isa.svload[2][v],
       footer := When(res=0, skip(),
           let(svloadR := isa.svload[2][res],
               svloadR(vtref(isa.t, y, nv1 + ybase), List([0..res-1], i -> nth(x, 2*func.at(nv1*v + i + xbase))), opts))),
       loop(it, o.part,
           chain(
               loop(ii, nv1,
                   svload(vtref(isa.t, y, ii + ybase), List([0..v-1], i -> nth(x, 2*func.at(v*ii + i + xbase))), opts)),
               footer))
    ),

   RCVStretchGath := (self, o, y, x, opts) >> let(
       v    := o.v,
       func := o.func.lambda(),
       rat  := o.func.domain()/o.part,
       nv1  := idiv(rat, v),
       res  := EvalScalar(rat-v*nv1),
       ii   := Ind(),
       it   := Ind(),
       isa  := opts.vector.isa,
       svload := isa.svload[2][v/2],
       ybase := (2 * _roundup(rat, v) / v) * it,
       xbase := rat * it,

       footer := Cond(
           res=0,
              skip(),

           res<=v/2, let(svloadR := isa.svload[2][res],
               chain(
                   svloadR(vtref(isa.t, y, 2*nv1   + ybase), List([0..res-1], i -> nth(x, 2*func.at(nv1*v + i + xbase))), opts),
                   assign (vtref(isa.t, y, 2*nv1+1 + ybase), isa.t.zero()))),

           let(svloadR := isa.svload[2][res-v/2],
               chain(
                   svload (vtref(isa.t, y, 2*nv1   + ybase), List([0..v/2-1],   i -> nth(x, 2*func.at(nv1*v + i + xbase))), opts),
                   svloadR(vtref(isa.t, y, 2*nv1+1 + ybase), List([v/2..res-1], i -> nth(x, 2*func.at(nv1*v + i + xbase))), opts)))
       ),

       loop(it, o.part,
           chain(
               loop(ii, 2*nv1,
                   svload(vtref(isa.t, y, ii + ybase), List([0..v/2-1], i->nth(x, 2*func.at(ii*(v/2) + i + xbase))), opts)),
           footer))
    ),

    ##
    _VScat_pc := (self, N, n, ofs, v, y, x, xofs, opts) >> Checked(IsPosInt0Sym(n), let(isa := opts.vector.isa,
        nv1 := idiv(n, v),
	res := imod(n, v),
	i   := Ind(nv1),
        chain(loop(i, nv1,        isa.storeCont(v,   y, v*i   + ofs, imod(ofs, v), x, xofs +   i, opts)),
              Cond(res=0, skip(), isa.storeCont(res, y, v*nv1 + ofs, imod(ofs, v), x, xofs + nv1, opts)))
    )),

    # NOTE: _pcAcc is a hack
    _VScat_pcAcc := (self, N, n, ofs, v, y, x, xofs, opts) >> Checked(IsPosInt0Sym(n), let(isa := opts.vector.isa,
        nv1 := idiv(n, v),
	res := imod(n, v),
	i   := Ind(nv1),
        chain(loop(i, nv1,        isa.storeContAcc(v,   y, v*i   + ofs, imod(ofs, v), x, xofs +   i, opts)),
              Cond(res=0, skip(), isa.storeContAcc(res, y, v*nv1 + ofs, imod(ofs, v), x, xofs + nv1, opts)))
    )),

    _VGath_pc := (self, N, n, ofs, v, y, yofs, x, opts) >> Checked(IsPosInt0Sym(n), let(isa := opts.vector.isa,
        nv1 := idiv(n, v),
	res := imod(n, v),
        i   := Ind(nv1),
        chain(loop(i, nv1,        isa.loadCont(v,   y, yofs +   i, x, v*i + ofs,   imod(ofs, v), opts)),
              When(res=0, skip(), isa.loadCont(res, y, yofs + nv1, x, v*nv1 + ofs, imod(ofs, v), opts)))
    )),

    VScat_pc    := (self, o, y, x, opts) >> self._VScat_pc   (o.N, o.n, o.ofs, o.v, y, x, 0, opts),
    VScat_pcAcc := (self, o, y, x, opts) >> self._VScat_pcAcc(o.N, o.n, o.ofs, o.v, y, x, 0, opts),
    VGath_pc    := (self, o, y, x, opts) >> self._VGath_pc   (o.N, o.n, o.ofs, o.v, y, 0, x, opts),

    ##
    IxVGath_pc := (self, o, y, x, opts) >> Checked(IsPosInt0(o.n), let(
	nn := _roundup(o.n, o.v)/o.v,
	jj := Ind(),
	chain(Cond(IsPosInt0(o.k) and IsPosInt0(o.ofs) and IsPosInt0(o.N),
	     # if all values are constant, we fully unroll to keep alignments constant
	     List([0..o.k-1], jj -> self._VGath_pc(o.N, o.n, o.ofs + o.N*jj, o.v, y, nn*jj, x, opts)),
	     # no need to fully unroll, alignment will still be unknown
	     loop(jj, o.k,  	    self._VGath_pc(o.N, o.n, o.ofs + o.N*jj, o.v, y, nn*jj, x, opts))))
    )),
    IxVScat_pc := (self, o, y, x, opts) >> Checked(IsPosInt0(o.n), let(
	nn := _roundup(o.n, o.v)/o.v,
	jj := Ind(),
	chain(Cond(IsPosInt0(o.k) and IsPosInt0(o.ofs) and IsPosInt0(o.N),
	     # if all values are constant, we fully unroll to keep alignments constant
	     List([0..o.k-1], jj -> self._VScat_pc(o.N, o.n, o.ofs + o.N*jj, o.v, y, x, nn*jj, opts)),
	     # no need to fully unroll, alignment will still be unknown
	     loop(jj, o.k,  	    self._VScat_pc(o.N, o.n, o.ofs + o.N*jj, o.v, y, x, nn*jj, opts))))
    )),
    ##
    _RCVGath_pc := (self, N, n, ofs, v, y, yofs, x, opts) >> let(
        nv1 := idiv(n, v), res := imod(n, v), isa := opts.vector.isa,
        chain(self._VGath_pc(2*N, 2*n, 2*ofs, v, y, 2*yofs, x, opts),
	      When(res > v/2, skip(),
		   assign(vtref(isa.t, y, 2*yofs + 2*nv1 + 1), isa.t.zero()))) # extra 0 pad is needed
    ),
    _RCVScat_pc := (self, N, n, ofs, v, y, x, xofs, opts) >> let(
        nv1 := idiv(n, v), res := imod(n, v),
        self._VScat_pc(2*N, 2*n, 2*ofs, v, y, x, 2*xofs, opts) # no extra 0 pads unlike in _RCVGath_pc case
    ),
    ##
    IxRCVGath_pc := (self, o, y, x, opts) >> Checked(IsPosInt0(o.n), let(
	nn := _roundup(o.n, o.v)/o.v,
	jj := Ind(),
	Cond(IsPosInt0(o.k) and IsPosInt0(o.ofs) and IsPosInt0(o.N),
	     # if all values are constant, we fully unroll to keep alignments constant
	     chain(List([0..o.k-1], jj -> self._RCVGath_pc(o.N, o.n, o.ofs + o.N*jj, o.v, y, nn*jj, x, opts))),
	     # no need to fully unroll, alignment will still be unknown
	     loop(jj, o.k,                self._RCVGath_pc(o.N, o.n, o.ofs + o.N*jj, o.v, y, nn*jj, x, opts)))
    )),
    IxRCVScat_pc := (self, o, y, x, opts) >> Checked(IsPosInt0(o.n), let(
	nn := _roundup(o.n, o.v)/o.v,
	jj := Ind(),
	Cond(IsPosInt0(o.k) and IsPosInt0(o.ofs) and IsPosInt0(o.N),
	     # if all values are constant, we fully unroll to keep alignments constant
	     chain(List([0..o.k-1], jj -> self._RCVScat_pc(o.N, o.n, o.ofs + o.N*jj, o.v, y, x, nn*jj, opts))),
	     # no need to fully unroll, alignment will still be unknown
	     loop(jj, o.k,                self._RCVScat_pc(o.N, o.n, o.ofs + o.N*jj, o.v, y, x, nn*jj, opts)))
    )),

    # ---------------
    # Scatter
    # ---------------
    VScat := (self, o, y, x, opts) >> Cond(IsUnalignedPtrT(y.t),
        self(VScat_u(fTensor(o.func, fBase(o.v, 0)), o.v), y, x, opts),
        self(VTensor(Scat(o.func), o.v), y, x, opts)),

    VScat_u := (self, o, y, x, opts) >> let(
        j    := Ind(o.func.domain()),
	yofs := o.func.at(j),
        loop(j, j.range,
            opts.vector.isa.storeCont(o.v, y, yofs, imod(yofs, o.v), x, j, opts))),

    VScatAcc := (self, o, y, x, opts) >>
	Cond(IsUnalignedPtrT(y.t),
             let(j      := Ind(o.func.domain()),
		 yofs   := o.func.at(j),
		 loop(j, j.range,
		     opts.vector.isa.storeContAcc(o.v, y, o.v*yofs, 0, x, j, opts))),
	     # else
	     self._acc(self(VScat(o.func, o.v),y,x,opts), y)),

    VScatAcc_u := (self, o, y, x, opts) >> let(
	j      := Ind(o.func.domain()),
	yofs   := o.func.at(j),
	nfunc  := fTensor(o.func, fBase(o.v, 0)),
	loop(j, j.range,
	    opts.vector.isa.storeContAcc(o.v, y, yofs, imod(yofs, o.v), x, j, opts))),

    VScat_zero := (self, o, y, x, opts) >> let(
	i := Ind(), j := Ind(), v := o.v, isa := opts.vector.isa,
        chain(
            loop(i, o.n,     assign(vtref(isa.t, y, i    ),  vtref(isa.t, x, i))),
            loop(j, o.N-o.n, assign(vtref(isa.t, y, o.n+j),  isa.t.zero())))),

    VStretchScat := (self, o, y, x, opts) >> let(
        v := o.v,
        func := o.func.lambda(),
        rat := func.domain() / o.part,
	nv1 := idiv(rat, v),
	res := EvalScalar(rat - v*nv1),
        ii := Ind(),
	it := Ind(),
	xbase := rat * it,
	ybase := it * ((_roundup(rat, v) / v)),
	isa := opts.vector.isa,
        svstore := isa.svstore[1][v],

        footer := Cond(
            res=0, skip(),
            let(svstoreR := isa.svstore[1][res],
                svstoreR(List([0..res-1], i->nth(y, func.at(nv1*v + i + xbase))), vtref(isa.t, x, nv1 + ybase), opts))
        ),
        loop(it, o.part,
            chain(
                loop(ii, nv1,
                    svstore(List([0..v-1], i->nth(y, func.at(v*ii + i + xbase))), vtref(isa.t, x, ii + ybase), opts)),
                footer)
        )
    ),

    vRCStretchScat := (self, o, y, x, opts) >> let(
        v := o.v/2,
        func := o.func.lambda(),
        rat := func.domain() / o.part,
	nv1 := idiv(rat, v),
	res := EvalScalar(rat-v*nv1),
        ii := Ind(),
	it := Ind(),
	ybase := it * (_roundup(rat, v) / v),
	xbase := rat * it,
	isa := opts.vector.isa,
        svstore := isa.svstore[2][v],

        footer := Cond(
            res=0, skip(),
            let(svstoreR := isa.svstore[2][res],
                svstoreR(List([0..res-1], i -> nth(y, 2 * func.at(nv1*v + i + xbase))), vtref(isa.t, x, nv1 + ybase), opts))
        ),
        loop(it, o.part,
            chain(
                loop(ii, nv1,
                    svstore(List([0..v-1], i -> nth(y, 2 * func.at(v*ii + i + xbase))), vtref(isa.t, x, ii + ybase), opts)),
                footer)
        )
    ),

    RCVStretchScat := (self, o, y, x, opts) >> let(
        v    := o.v,
        func := o.func.lambda(),
        rat  := func.domain()/o.part,
        nv1  := idiv(rat, v),
        res  := EvalScalar(rat-v*nv1),
        ii := Ind(),
        it := Ind(),
        ybase := 2 * (_roundup(rat, v) / v) * it,
	xbase := rat * it,
	isa := opts.vector.isa,
        svstore := isa.svstore[2][v/2],

        footer := Cond(
            res=0,
                skip(),

            res <= v/2, let(svstoreR := isa.svstore[2][res],
                svstoreR(List([0..res-1], i->nth(y, 2 * func.at(nv1*v + i + xbase))), vtref(isa.t, x, 2*nv1 + ybase), opts)),

            let(svstoreR := isa.svstore[2][res-v/2],
                chain(
                  svstore (List([0..v/2-1],   i->nth(y, 2 * func.at(nv1*v + i + xbase))), vtref(isa.t, x, 2*nv1   + ybase), opts),
                  svstoreR(List([v/2..res-1], i->nth(y, 2 * func.at(nv1*v + i + xbase))), vtref(isa.t, x, 2*nv1+1 + ybase), opts))
            )
        ),
        loop(it, o.part,
            chain(
                loop(ii, 2*nv1,
                    svstore(List([0..v/2-1], i->nth(y, 2 * func.at(v/2*ii + i + xbase))), vtref(isa.t, x, ii + ybase), opts)),
            footer)
        )
    ),


    # ---------------
    # Diagonal
    # ---------------
    VDiag := (self, o, y, x, opts) >>
        self(VTensor(Diag(VData(o.element, o.v)), o.v), y, x, opts),

    VDiag_x_I := (self,o,y,x,opts) >>
        self(VTensor(Diag(o.element), o.v), y, x, opts),

    # need to do VRef into the twiddles
    VRCDiag := (self, o, y, x, opts) >>
        self(VTensor(RCDiag(o.element), o.v), y, x, opts),

    VBlk := (self, o, y, x, opts) >> self(VTensor(Blk(o.velement()), o.v), y, x, opts),

    RCVBlk := (self, o, y, x, opts) >> let(
        #self(BlockMat(MapMat(o.velement(), el->RCVDiag(el, o.v))), y, x, opts),
        isa := opts.vector.isa,
        cx_mul := isa.mul_cx(opts),
        j   := Ind(),
        i   := Ind(),
        t   := TempVar(isa.t),
        u   := TempVar(isa.t),
        mat := o.velement(),
        d   := Dat(mat.t),
	data(d, mat,
            loop(j, Rows(o) / o.v,
		decl([t, u], chain( # FF: fixed for v>2
                    assign(t, 0),
                    loop(i, Cols(o) / o.v, chain(
			cx_mul(u, vtref(isa.t, x, i), nth(nth(d, j), i)),
			assign(t, add(t, u)))
		    ),
                    assign(vtref(isa.t, y, j), t)
		))
    ))),

    # dv := VData(o.element, o.v);
    RCVDiag := (self, o, y, x, opts) >> let(
        i     := Ind(),
        isa   := opts.vector.isa,
        cxmul := isa.mul_cx(opts),
        loop(i, Rows(o)/o.v,
             cxmul(vtref(isa.t, y, i), vtref(isa.t, x, i), o.element.at(i)))
    ),

    RCVOnlineDiag := (self, o, y, x, opts) >> let(
        isa        := opts.vector.isa,
        cxmul      := isa.mul_cx(opts),
        cxmul_conj := isa.mul_cx_conj(opts),
	cc         := o.element.computeWithOps(cxmul, cxmul_conj),
	decl(cc.values, chain(
	    cc.code,
	    List([1..Length(cc.values)], i -> cxmul(vtref(isa.t, y, i-1), cc.values[i], vtref(isa.t, x, i-1)))
	))
    ),

    RCVDiagSplit := (self, o, y, x, opts) >> let(
        isa := opts.vector.isa,
        i   := Ind(),
        yy  := vtref(isa.t, y, i),
        xx  := vtref(isa.t, x, i),
        t   := var.fresh_t("U", isa.t),
        t2  := var.fresh_t("U", isa.t),
        loop(i, Rows(o)/o.v,
	    decl([t, t2], chain(
                isa.swap_cx(t, xx, opts),
                assign(yy, add(mul(o.element.at(2*i), xx), mul(o.element.at(2*i+1), t))))))
    ),


    VS := meth(self, o, y, x, opts)
        local xv, yv, i, n, isa, u, w;
	isa := opts.vector.isa;
        xv := tcast(TPtr(TVect(x.t.t, o.v)), x);
        yv := tcast(TPtr(TVect(y.t.t, o.v)), y);
        n := o.n / o.v;
        i := Ind();
	u := var.fresh_t("U", TVect(x.t.t, o.v));
        if o.transposed then
	    return decl([u], chain(
		loop(i, n-1, chain(
		    isa.bin_shl2(u, [nth(xv,i), nth(xv, i+1)], opts),
                    assign(nth(yv, i+1), add(u, nth(xv, i+1))))
		),
		isa.bin_shl1(u, nth(xv, 0), opts),
                assign(nth(yv, 0), add(u, nth(xv, 0)))
	    ));
        else
	    return decl([u], chain(
                loop(i, n-1, chain(
                    isa.bin_shr2(u, [nth(xv, i), nth(xv, i+1)], opts),
                    assign(nth(yv, i), add(u, nth(xv, i))))
		),
                isa.bin_shr1(u, nth(xv, n-1), opts),
                assign(nth(yv, n-1), add(u, nth(xv, n-1)))));
        fi;
    end,

    VO1dsJ := meth(self, o, y, x, opts)
        local xv, yv, i, n, isa, t;
        isa := opts.vector.isa;
        t   := Checked( IsVecT(isa.t) and isa.t.size = o.v, TPtr(isa.t));
        xv  := tcast(t, x);
        yv  := tcast(t, y);
        n   := o.n / o.v;
	i   := Ind(n-1);
	return chain(
	    loop(i, n-1, isa.bin_shrev(nth(yv, i+1), [nth(xv, n-i-1), nth(xv, n-i-2)], opts)),
            isa.bin_shrev(nth(yv, 0), [isa.t.zero(), nth(xv, n-1)], opts));
    end,

    VScale := (self, o, y, x, opts) >>
        self(VDiag(fConst(TReal, Rows(o), o.s), o.v) * o.child(1), y, x, opts),

    VOLMultiplication := (self, o, y, x, opts) >>
        self(VTensor(OLMultiplication(o.rChildren()[1], o.rChildren()[2]),  o.rChildren()[3]), y, x, opts),

    VRCOLMultiplication := (self, o, y, x, opts) >>
        self(VTensor(RCOLMultiplication(o.rChildren()[1], o.rChildren()[2]), o.rChildren()[3]), y, x, opts),

    RCVOLMultiplication := (self, o, y, x, opts) >> let(
        n  := o.rChildren()[2],
        m  := o.rChildren()[1],
        v  := o.rChildren()[3],
        t  := TPtr(opts.vector.isa.t),
        xv := List(x, e -> tcast(t, e)),
        yv := tcast(t, StripList(y)),
        mul_cx := opts.vector.isa.mul_cx(opts),
        i  := Ind(),
        a  := var.fresh_t("a", opts.vector.isa.t),
        loop( i, n, decl([a], chain(
            assign(a, nth(xv[1], i)),
            chain( List( [2..m], j -> mul_cx(a, nth(xv[j], i), a) )),
            assign(nth(yv, i), a))))
    ),

    VOLConjMultiplication := (self, o, y, x, opts) >>
        self(VTensor(OLConjMultiplication(o.rChildren()[1], o.rChildren()[2]),  o.rChildren()[3]), y, x, opts),

    VRCOLConjMultiplication := (self, o, y, x, opts) >>
        self(VTensor(RCOLConjMultiplication(o.rChildren()[1], o.rChildren()[2]), o.rChildren()[3]), y, x, opts),

    RCVOLConjMultiplication := (self, o, y, x, opts) >> let(
        n  := o.rChildren()[2],
        m  := o.rChildren()[1],
        v  := o.rChildren()[3],
        t  := TPtr(opts.vector.isa.t),
        xv := List(x, e -> tcast(t, e)),
        yv := tcast(t, StripList(y)),
        mul_cx_conj := opts.vector.isa.mul_cx_conj(opts),
        i  := Ind(),
        a  := var.fresh_t("a", opts.vector.isa.t),
        loop( i, n, decl([a], chain(
            assign(a, nth(xv[1], i)),
            chain( List( [2..m], j -> mul_cx_conj(a, nth(xv[j], i), a) )),
            assign(nth(yv, i), a))))
    ),

    RCVIxJ2 := (self, o, y, x, opts) >> let(
        isa := opts.vector.isa,
        t   := Checked( IsVecT(isa.t) and isa.t.size = o.v, TPtr(isa.t)),
        xv  := tcast(t, x),
        yv  := tcast(t, y),
        opts.vector.isa.RCVIxJ2(nth(yv,0), nth(xv, 0), opts)),

    VIxJ2 := (self, o, y, x, opts) >> let(
        isa := opts.vector.isa,
        t := Checked(IsVecT(isa.t) and isa.t.size = o.v, TPtr(isa.t)),
        xv := tcast(t, x),
        yv := tcast(t, y),
        opts.vector.isa.swap_cx(nth(yv, 0), nth(xv, 0), opts) ),

    VJxI := (self, o, y, x, opts) >> let(
        isa := opts.vector.isa,
        t   := Checked( IsVecT(isa.t) and isa.t.size = o.v, TPtr(isa.t)),
        xv  := tcast(t, x),
        yv  := tcast(t, y),
        opts.vector.isa.VJxI(o.m, nth(yv,0), nth(xv, 0), opts)),

));
