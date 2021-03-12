
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(RollingPointers, rec(
    __call__ := (self, code, opts) >> CopyFields(self, rec(opts := opts)).apply(code),

    _linearPatterns := Set([[], [mul]]),

    _contains    := (v, U) -> CollectNR(U, @.cond(u->u=v))<>[],
    _linear_exp  := (self, e, v) >> Length(Collect(e, @@.cond((x, cx)-> x=v and (List(cx.parents, ObjId) in self._linearPatterns))))=1,

    _basePtr  := (summands) -> let( p := Filtered(summands, e->IsPtrT(e.t) or IsArrayT(e.t)), Checked(Length(p)=1, p[1]) ),

    # _collectPtrs(<l>) -- <l> loop 
    _collectPtrs := meth(self, l)
        local free, vptr, pattern, ptradds, ptrs, e, p, d, id, clusters, loopinv, i;

        # Collecting innermost ponter arithmetic additions.
        # Index computation expression expected to be in normalized form.
        pattern := @@(1, [add], (x, cx) -> IsPtrT(x.t) and (Filtered(cx.parents, IsLoop)=[] or Last(Filtered(cx.parents, IsLoop))=l));
         
        ptradds := Collect(l, pattern);
        ptradds := Filtered(ptradds, x -> Collect(x.args, pattern)=[]);
        
        ptrs := tab();

        free := Set(l.free() :: [l.var]);
        # extracting base pointer and common subexpression with this pointer.
        # relying on expression print method to get string id. 
        for e in ptradds do
            p  := self._basePtr(e.args);
            # we may have buffer defined in the loop referenced by loop variable
            vptr := Collect(p, var);
            if vptr=[] or Intersection(vptr, free)<>[] then
                id := StringPrint(p);
                if IsBound(ptrs.(id)) then
                    ptrs.(id).cmn := Checked(ptrs.(id).ptr = p, Intersection(ptrs.(id).cmn, Set(e.args)));
                else
                    ptrs.(id) := rec( ptr := p, cmn := Set(e.args), offs := [] );
                fi;
            fi;
        od;
        # make sure common expression doesn't have local loop variables
        for id in UserNSFields(ptrs) do
            ptrs.(id).cmn := Filtered(ptrs.(id).cmn, a -> IsSubset(free, Collect(a, var)));
        od;
        # preparing offsets list for each base expression.
        #
        # offset record: rec( 
        #    ref  := <full original expression>,
        #    tail := <offset from common base expression, as summands list> 
        # )
        for e in ptradds do
            id := StringPrint(self._basePtr(e.args));
            if IsBound(ptrs.(id)) then
                d := Difference(e.args, ptrs.(id).cmn);
                Add(ptrs.(id).offs, rec( ref := e, tail := d ));
            fi;
        od;

        # when loop variable is not in common subexpression 'cmn' but still in 'tail' 
        # try to split pointer so that variable goes to 'cmn' in each resulting pointer
        for id in UserNSFields(ptrs) do
            p := ptrs.(id);
            if not self._contains(l.var, p.cmn) then
                [ clusters, loopinv ] := SplitBy(ConcatList(p.offs, o -> o.tail), s -> self._linear_exp(s, l.var));
                if ForAll(loopinv, s -> not self._contains(l.var, s)) then
                    clusters := Set(clusters);
                    # NOTE: hardcoded max number of resulting rolling pointers
                    if Length(clusters)<=4 then
                        for i in [1..Length(clusters)] do
                            ptrs.(id :: "_" :: StringInt(i)) := rec(
                                ptr  := p.ptr, 
                                cmn  := Set( p.cmn :: [clusters[i]] ),
                                # each original <offs> can contain only one expression from <clusters>
                                # because previousely summands were grouped by loop variables
                                offs := List(Filtered(p.offs, o -> clusters[i] in o.tail), 
                                          o -> rec( ref := o.ref, tail := RemoveList(o.tail, clusters[i]) )),
                            );
                        od;
                        Unbind(ptrs.(id));
                    fi;
                fi;
            fi;
        od;

        return ptrs;
    end,

    _offsetsInit := function(vars, init, coeff)
        vars.linv := vars.linv :: vars.offs;
        init.linv := init.linv :: List( [1..Length(coeff)], i -> assign(vars.offs[i], vars.strides*coeff[i]));
    end,

    # cost is number of rolling pointers + their stride + number of offsets from rolling pointers 
    _offsetsCost := (self, k) >> Cond(Length(k.rp_coeff)>4, 100000, self.rpCost*Length(k.rp_coeff) + 1 + self.coeffCost*Length(k.coeff)),
   
    # default cost of rolling pointer and coefficient
    rpCost    := 3,
    coeffCost := 1,

    # includeValues
    includeValues := false,

    
    _strideGrp := s -> CondPat( s, @(1, Value),       V(1), 
                                   [mul, Value, @],   s.args[2],
                                # else
                                   s ),
    _strideVal := s -> CondPat( s, @(1, Value),       s.v,   
                                   [mul, Value, @],   s.args[1].v, 
                                # else
                                   1 ),

    _getStrides := meth(self, l, offs)
        local s, strides, offsets, free;

        if self.includeValues then
            s := ConcatList(offs, e -> e.tail);
        else
            s := ConcatList(offs, e -> Filtered(e.tail, x -> not (x _is Value)));
        fi;
        # ignore expressions with loop-local variables
        free := Set(l.free() :: [l.var]);
        s := Filtered(s, a -> IsSubset(free, Collect(a, var)));
        
        # get strides vector
        strides := Filtered(GroupList(s, self._strideGrp), e -> Length(e[2])>1);
        if strides<>[] then
            strides := TransposedMat(strides)[1];
        fi;

        # get stride coefficients for each offset + leftover expression
        offsets := TransposedMat(List(offs, function(e)
            local c, leftovers, x, p;
            leftovers := [];
            c := Replicate(Length(strides), 0);
            for x in e.tail do
                p := Position(strides, self._strideGrp(x));
                if p<>false then
                    c[p] := self._strideVal(x);
                else
                    Add(leftovers, x);
                fi;
            od;
            return [c, rec(ref := e.ref, leftovers := leftovers)];
        end));

        return [strides, offsets[1], offsets[2]];
    end,

    # reduce <coeff> offsets list to coefficients relative to <k> "evenly" spaced rolling pointers
    _k_func := function(k, coeff)
        local n, rp, offs, i;
        n     := Length(coeff);
        rp    := List([0..k-1], i -> QuoInt(n*i, k)+1) :: [n+1];
        offs := [];
        for i in [1..k] do
            offs := UnionSet(offs, List(coeff{[rp[i]+1..rp[i+1]-1]}, c -> c - coeff[rp[i]]));
        od;
        return rec(
            rp_coeff := List(DropLast(rp, 1), i -> coeff[i]), 
            coeff    := offs,
        );
    end,

    # finds best split [<rolling ptrs>, <offsets>] for <coeff> offsets list.
    # returns record:
    #    .rp_coeff - list of coefficients for rolling pointers
    #    .coeff    - reduced <coeff> list so that every original <coeff>
    #                can be expressed as some <rp_coeff> plus some <coeff>
    #                from this list.
    #
    # Brute force, _offsetsCost can be redefined

    _bestSplit := meth(self, coeff)
        local scoeff, min_k, min_cost, i, j, k, o, cost;
        scoeff := Set(coeff);
        min_k := self._k_func(1, scoeff);
        min_cost := self._offsetsCost(min_k);
        for i in [1..Length(scoeff[1])] do
            o := Sort(ShallowCopy(scoeff), (a,b) -> Cond(a[i]=b[i], a<b, a[i]<b[i]));
            for j in [1..Length(Set(List(o, e -> e[i])))] do
                k := self._k_func(j, o);
                cost := self._offsetsCost(k);
                if cost < min_cost then
                    min_k := k;
                    min_cost := cost;
                fi;
            od;
        od;
        return min_k;
    end,

    
    # find indices in <offsets> of a given <offs> (not including crossed-out in <refs>) 
    _offsetsIndices := (offsets, offs, refs) -> Filtered([1..Length(offsets)], i -> offsets[i]=offs and refs[i].ref<>false),

    # add mapping pair to <map> and cross-out index.
    _mapRecord := function(map, rem, summands)
        Add(map, rec(
            ref := rem.ref,
            subst := ApplyFunc(add, summands :: rem.leftovers)
        ));
        rem.ref := false;
    end,

    # split pointer expressions into rolling pointers + offsets
    _splitPtr := meth(self, ptr, loop, vars, init, incr, map)
        local strides, offsets_coeff, offsets_rem, cfg, i, j, k, ld, li, p;

        [strides, offsets_coeff, offsets_rem] := self._getStrides(loop, ptr.offs);
        
        cfg := self._bestSplit(offsets_coeff);

        # temporary variables lists.
        # Doing CSE is smarter as we may have same stride and offset computation expressions
        # for other pointers. I don't have this situation in DFTs though.
        vars.rp      := List(cfg.rp_coeff, e -> var.fresh_t("rp",   Cond(IsArrayT(ptr.ptr.t), ptr.ptr.t.toPtrType(), ptr.ptr.t)));
        vars.offs    := List(cfg.coeff,    e -> var.fresh_t("offs", TInt));
        vars.strides := List(strides,      e -> var.fresh_t("s",    TInt));
        vars.inc     := var.fresh_t("inc", TInt);

        # vars.offs is not added here as they might be declared in or outside of the loop
        # this is done by initOffsets method later
        vars.linv    := vars.linv :: vars.rp :: vars.strides :: [vars.inc];

        # preparing mapping from original pointer expressions to rolling pointers expressions
        for i in [1..Length(cfg.rp_coeff)] do
            # for each expression which correspond to this rolling pointer
            for k in self._offsetsIndices(offsets_coeff, cfg.rp_coeff[i], offsets_rem) do
                self._mapRecord(map, offsets_rem[k], [vars.rp[i]]);
            od;

            for j in [1..Length(cfg.coeff)] do
                # for each expression which correspond to this offset from current rolling pointer
                for k in self._offsetsIndices(offsets_coeff, cfg.rp_coeff[i]+cfg.coeff[j], offsets_rem) do
                    self._mapRecord(map, offsets_rem[k], [vars.rp[i], vars.offs[j]]);
                od;
            od;
        od;

        Checked(ForAll(offsets_rem, e -> e.ref=false), "paranoid");
       
        # <ld> loop var dependent summands, <li> - loop invariant summands
        [ld, li] := SplitBy(ptr.cmn, e -> self._contains(loop.var, e));

        # initialization code
        init.strides := List([1..Length(strides)], i -> assign(vars.strides[i], strides[i]));
        init.rp      := List([1..Length(cfg.rp_coeff)], i -> assign(vars.rp[i], SReduce(ApplyFunc(add, li :: SubstVars( Copy(ld), rec( (loop.var.id) := V(0))) :: Cond(vars.strides<>[], [vars.strides*cfg.rp_coeff[i]], [])), self.opts)));
        init.inc     := assign(vars.inc, SReduce(ApplyFunc(add, SubstVars( Copy(ld), rec( (loop.var.id) := V(1)))), self.opts));

        init.linv    := init.linv :: init.strides :: init.rp :: [init.inc];

        for p in vars.rp do
            Add(incr, assign(p, add(p, vars.inc)));
        od;
        
        self._offsetsInit(vars, init, cfg.coeff);
    end,
        

    _processPointer := meth(self, ptr, loop, vars, init, incr, map)
        local ld;
        # process pointers with separable base and loop dependent linear offset
        ld := Filtered(ptr.cmn, e -> self._contains(loop.var, e));
        if ld<>[] 
           and ForAll(ld, s -> self._linear_exp(s, loop.var))
           and ForAll(ptr.offs, e -> not self._contains(loop.var, e.tail)) 
        then
            self._splitPtr(ptr, loop, vars, init, incr, map);
        fi;
    end,

    inject := (self, code, vars, prologue, epilogue) >> Cond(
         code _is decl,  decl(code.vars, self.inject(code.cmd, vars, prologue, epilogue)),
         code _is data,  data(code.var, code.value, self.inject(code.cmd, vars, prologue, epilogue)),
         # else
             decl(vars, chain(prologue, code, epilogue))),

    transformLoop := meth(self, l)
        local new_loop, ptrs, field, vars, init, incr, map, idx, oid, cmd, cp, subst;

        # collect references grouped by base pointers and extracted common summands
        ptrs := self._collectPtrs(l);

        # variables updated by _processPointer method
        #   vars.linv - loop invariant (outer) variables list
        #   vars.loop - inner variables list (declared in the loop body)
        #
        # initialization:
        #   init.linv - outer commands
        #   init.loop - inner commands
        #
        # incr - list of commands that increment pointers (loop body footer)
        # map - list of mapping records:
        #   map[i].ref - expression to replace
        #   map[i].subst 
        # 
        vars := rec( linv := [], loop := [] );
        init := rec( linv := [], loop := [] );
        incr := [];
        map  := [];

        for field in UserNSFields(ptrs) do
            self._processPointer( ptrs.(field), l, vars, init, incr, map );
        od;

        if map<>[] then
            # copy propagation of simple loop invariant expressions
            cp  := tab();
            init.linv := Filtered(init.linv, function(e) local prop;
                    prop := CondPat( e,
                              [assign, var, Value], true,
                              [assign, var, param], true,
                              false);
                    if prop then
                        cp.(e.loc.id) := e.exp;
                    fi;
                    return not prop;
                end);
            vars.linv := Filtered(vars.linv, e -> not IsBound(cp.(e.id)));

            subst := (c) -> SubstVars(c, cp);

            idx := Set(List(map, e -> e.ref));
            oid := Set(List(map, e -> ObjId(e.ref)));
            cmd := SubstTopDownNR_named(l.cmd, @(1, oid, x -> x in idx),
                    e -> subst(map[PositionProperty( map, x -> x.ref=e )].subst), "toRollingPointer");
            new_loop := decl( vars.linv, chain( subst(init.linv) :: [
                                loop( l.var, l.range, self.inject(cmd, subst(vars.loop), subst(init.loop), subst(incr)))
                        ]));
        else
            new_loop := l;
        fi;

        return new_loop;
    end,

    
    apply := meth(self, code)

        # normalize linear expressions, group summands and pull out loop variables
        code := SubstTopDownNR_named(code, @(1, self.opts.simpIndicesInside), e -> 
                   SubstBottomUp(GroupSummandsExp(ESReduce(e, self.opts)), add, a->__groupSummandsVar(a.args)), "rollingPtrNormalize");

        # transform loops, innermost first
        code := SubstBottomUp(code, @(1, [loop, loopn]), e -> self.transformLoop(e));

        return code;
    end,
));



