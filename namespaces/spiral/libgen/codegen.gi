
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F AllocBuffersSMP(pfx, code, map, opts)
#F
#F Returns [newbuffers, allocation_code, code]
#F and fills the <map> with the remapping from original array-typed variables
#F to the new dynamically allocated pointer-typed variables.
#F
#F if opts.zeroallocate is true, the buffers are zeroed out
#F if opts.buffPrefixExtra is set, it's appended to pfx
AllocBuffersSMP := function(pfx, code, map, opts)
    local nameprefix, newbuffers, buffers, b, bufmap, v, nthreads, tid, newv, alloc_code;
    ## Pull out the buffer declarations
    [buffers, code] := PullBuffersSMP(code, IsArrayT);
    alloc_code := [];
    newbuffers := [];
	if IsBound(opts.buffPrefixExtra) then
		nameprefix := Concat(pfx, String(opts.buffPrefixExtra));
	else
		nameprefix := pfx;
	fi;
    for b in buffers do
        [v, nthreads, tid] := b;
        if nthreads = 1 then
            newv := var.fresh_t(nameprefix, TPtr(v.t.t));
        else
            newv := var.fresh_t(Concat("thr", nameprefix), TPtr(v.t.t));
        fi;
        if IsBound(v.value) then
            newv.value := v.value;
        fi;
        Add(newbuffers, newv);
        if IsBound(opts.zeroallocate) and opts.zeroallocate then
            Add(alloc_code, zallocate(newv, TArray(v.t.t, v.t.size*nthreads)));
        else
            Add(alloc_code, allocate(newv, TArray(v.t.t, v.t.size*nthreads)));
        fi;
        map.(v.id) := When(nthreads=1, newv, newv + tid*v.t.size);
    od;
    return [newbuffers, alloc_code, code];
end;

#F UnifyBuffersSMP(pfx, code, map, ubuf)
#F
#F Returns [ubuf_size, ptr_init_code, code]
#F and fills the <map> with the remapping from original array-typed variables
#F to the new dynamically allocated pointer-typed variables.
#F
#F ptr_init_code initializes the pointer-typed variables as pointers to
#F sections of <ubuf>
#F
UnifyBuffersSMP := function(pfx, code, map, ubuf, opts)
    local map, vars, alloc, ofs, usize, ptr_init_code;
    [vars, alloc, code] := AllocBuffersSMP(pfx, code, map,opts);
    ofs   := ScanL(alloc, (prev, c) -> prev + c.exp.size * sizeof(c.exp.t), 0);
    usize := Sum(alloc, a -> a.exp.size * sizeof(a.exp.t));

    ptr_init_code := List([1..Length(vars)], i -> assign(vars[i], tcast(vars[i].t, (ubuf + ofs[i]))));
    return [ usize, ptr_init_code, code ];
end;

Class(RecCodegen, RecCodegenMixin, SMPCodegenMixin, DefaultCodegen, rec(
    Formula := meth(self, o, y, x, opts)
        local code, datas, prog, params, init_code, destroy_code, codelet_codes, codelet_recs,
          datvars, dalloc, bufvars, bufalloc, map, ignore, num_threads, smp, io, _data, v;

        [x, y] := self.initXY(x, y, opts);
    o := o.child(1);
    params := Set(Collect(o, param));
    datas := Collect(o, @(1, var, e->IsBound(e.init)));
    smp := Collect(o, SMPSum);
    num_threads := When(smp=[], 1, smp[1].nthreads);
#    if not ForAll(smp, x->x.nthreads=num_threads) then Error("Non-uniform num_threads in SMPSum's"); fi;
    io := When(x=y, [x], [y, x]);

    ## Generating code : codelets
    codelet_recs := CompileCodelets(o, opts);
    codelet_codes := List(codelet_recs, clrec ->
        func(TVoid, clrec.name, Concatenation([Y, X], clrec.params), clrec.code));

    map := tab();
    ## Generating code : main body
    code := SReduce(self(o, y, x, opts), opts);
    code := BlockUnroll(code, opts);
    code := DeclareHidden(code); # NOTE: do I need this?
    code := func(TVoid, "transform", Concatenation(
        When(IsBound(opts.subParams), opts.subParams, []), params, io),
                When(num_threads=1, code,
                                    smp_fork(num_threads, code)));

    [bufvars, bufalloc, code] := AllocBuffersSMP("buf", code, map, opts);

    ## Generating code : initialization
    [datvars, dalloc, ignore] := AllocBuffersSMP("dat", decl(datas, skip()), map, opts);
    init_code := func(TVoid, "init", [],
        chain(bufalloc, dalloc, List(datas, x -> SReduce(x.init, opts))));
	destroy_code := func(TVoid, "destroy", [], skip());

    for v in bufvars do Add(v.t.qualifiers, "static"); od;
    for v in datvars do Add(v.t.qualifiers, "static"); od;
    _data := When(not IsBound(opts.smp) or (IsBound(opts.smp) and IsBound(opts.smp.OmpMode) and opts.smp.OmpMode = "for"), (i,j,k)->k, (i,j,k)->data(i, j, k));

    prog := program(
        codelet_codes,
        decl(Concatenation(datvars, bufvars),
        _data(var("NUM_THREADS", TInt), V(num_threads),
            SubstVars(chain(init_code, code, destroy_code), map))));
    prog.dimensions := o.dimensions;
    return prog;
    end,
));
