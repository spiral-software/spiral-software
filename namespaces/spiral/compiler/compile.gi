
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# NOTE: too slow
PackCode := function(c)
     local pulled, decls, datas, dims;
     if IsBound(c.dimensions) then dims := c.dimensions; fi;
     pulled := PullBU(Copy(c), decl, e -> e.cmd, e -> e.vars);
     decls := Set(Concatenation(pulled[1]));
     c := pulled[2];

     pulled := Pull(Copy(c), data, e -> e.cmd, e -> [e.var, e.value]);
     datas := pulled[1];
     c := pulled[2];

     c := SubstBottomUp(c, chain, e -> e.flatten());
     c := FoldL(datas, (cod,d) -> data(d[1], d[2], cod), decl(decls, c));
     if IsBound(dims) then c.dimensions := dims; fi;
     return c;
end;

FoldIf := c -> SubstTopDown(c,
    @(1).target(IF).cond(e->IsValue(e.cond)),
    e -> When(e.cond.v=0, e.else_cmd, e.then_cmd));

DeclareHidden := function(code)
    local free, v, dims;
    free := code.free();
    if IsBound(code.dimensions) then dims := code.dimensions; fi;
    for v in free do
        if IsBound(v.value) then code := data(v, v.value, code); fi;
    od;
    if IsBound(dims) then code.dimensions := dims; fi;
    return code;
end;

# This eliminates aliasing between subexpressions of different commands
# (ie pointers can alias, and this is bad)
UntangleChain := c -> SubstTopDownNR(c, chain, e -> chain(List(e.cmds, Copy)));

# reduce complex constants with im(c)=0 to float
_prepConst := e -> Cond(
    e.t <> TComplex, e,
    IsCyc(e.v) and Im(e.v)=0, TDouble.value(e.v),
    IsComplex(e.v) and ImComplex(e.v) = 0, TDouble.value(ReComplex(e.v)),
    e);

Class(_dropTRealInRefs, RuleSet);
RewriteRules(_dropTRealInRefs, rec(
    TReal_left  := ARule(ListClass, [ [@(1, TPtr), @(2, AtomicTyp, x->x=TReal), ...], 
        [@(3, TPtr, x->x.alignment=@(1).val.alignment), T_Real, ...]],
        e -> [@(3).val]),
    TReal_right := ARule(ListClass, [ [@(1, TPtr), T_Real, ...], 
        [@(2, TPtr), @(3, AtomicTyp, x->x=TReal and @(1).val.alignment=@(2).val.alignment), ...]],
        e -> [@(1).val]),
    TVect_TReal_left  := ARule(ListClass, [ [@(1, TPtr), [TVect, @(2, AtomicTyp, x->x=TReal), @(3)], ...], 
        [@(4, TPtr), [TVect, T_Real, @(5).cond(x->x=@(3).val and @(1).val.alignment=@(4).val.alignment)], ...]], 
        e -> [@(4).val]),
    TVect_TReal_right := ARule(ListClass, [ [@(1, TPtr), [TVect, T_Real, @(2)], ...],
        [@(3, TPtr), [TVect, @(4, AtomicTyp, x->x=TReal), @(5).cond(x->x=@(2).val and @(1).val.alignment=@(3).val.alignment)], ...]], 
        e -> [@(1).val]),
));

#F Compile(<code>)
#F
#F Fully unroll and optimize <code>
#F
Class(Compile, rec(
    datas := 0,
    decls := 0,
    refs := 0,

    status := Ignore,
    timingStatus := Ignore,

    pullDataDeclsRefs := meth(self, code)
        local pulled, c, d, p, id, typecasts, tcast, ref, loopvars, doNotScalarize;
        self.free := code.free();
        pulled := PullBU(code, @(1, [decl, data, nth, deref]),
            e -> When(IsBound(e.cmd), e.cmd, e), e -> e);

        self.decls := Set([]);
        self.datas := tab();
        self.refs  := tab();

        loopvars := Pull(code, @@(1).cond( (x, cx)->IsLoop(x) ),
            e -> e, (cx, e) -> e.var)[1];

        doNotScalarize := (ref) -> ref=false or Collect(ref, @(1, [var, param], x -> (IsLoopIndex(x) and not(x in loopvars)) or x _is param))<>[];

        for p in pulled[1] do
            if   ObjId(p)=decl then UniteSet(self.decls, p.vars);
            elif ObjId(p)=data then self.datas.(p.var.id) := p.value; if IsArrayT(p.var.t) then p.var.value := p.value; fi;
            # nth, deref
            elif IsVar(p.loc) then
                id := p.loc.id;
                if not IsBound(self.refs.(id)) then self.refs.(id) := Set([]); fi;
                AddSet(self.refs.(id), TPtr(p.t));
                if doNotScalarize(p) then # prevent scalarization when there is a dependency on outer loop variable or param
                    AddSet(self.refs.(id), TPtr(TVoid));
                fi;
            fi;
        od;

        # Pull out type casts and figure out granularity of accesses to given variable
        # the complicated condition tries to figure out <..what?..>
        typecasts := Pull(code,
            @@(1,var, (e,cx) -> IsVar(e) and IsArrayT(e.t) and IsBound(cx.tcast) and
                                cx.tcast<>[] and Last(cx.tcast).args[2].t in [e.t, e.t.toPtrType()] ),
            e -> e,
            (cx, e) -> [e, Last(cx.tcast), Cond(IsBound(cx.nth) and Length(cx.nth)>0, Last(cx.nth), false)])[1];

        for p in typecasts do
            [id, tcast, ref] := [p[1].id, p[2], p[3]];
            if not IsBound(self.refs.(id)) then self.refs.(id) := Set([]); fi;
            AddSet(self.refs.(id), tcast.args[1]);
            # below prevents scalarization if pointer arithmetic was used or
            # if there is dependency on outer loop variable or param
            if not IsVar(tcast.args[2]) or doNotScalarize(ref) then 
                AddSet(self.refs.(id), TPtr(TVoid));
            fi;
        od;
        # TReal pops up in diffrent places
        for id in UserNSFields(self.refs) do
            self.refs.(id) := _dropTRealInRefs(self.refs.(id));
        od;
        # plugs in data definitions into code, this process could be recursive, since one datavar can depend on another
        return SubstTopDown(pulled[2], @(1,var,e->IsBound(self.datas.(e.id))), e->self.datas.(e.id));
    end,
    
    scalarizationBarriers := [call, fcall],

    fastScalarize := meth(self,code)
        local d,len,tentry, barriers, typecasts, newlen, newt, newscal;
        barriers := Collect(code, @(1, self.scalarizationBarriers));
        self.doNotScalarize := Union(
            # do not scalarize arrays from barriers (like from function calls)
            Union(List(barriers, c->Collect(c, @(1,var,x->IsArrayT(x.t))))),
            # arrays with vector accesses of different granularity
            Set(Filtered(self.decls, v -> not IsBound(self.refs.(v.id)) or
                    (IsBound(self.refs.(v.id)) and (Length(self.refs.(v.id)) > 1 or ForAny(self.refs.(v.id), IsUnalignedPtrT))) or
                    (IsBound(v.doNotScalarize) and v.doNotScalarize)))
        );

        self.scalarized := Set([]);
        newscal := [];
        for d in Difference(self.decls, self.doNotScalarize) do
            newt := self.refs.(d.id)[1];
            if IsArrayT(d.t) and IsPtrT(newt) then
                newt := newt.t; # base type of ptr
                newlen := d.t.size * When(IsVecT(d.t.t), d.t.t.size, 1) / When(IsVecT(newt), newt.size, 1);
                if IsSymbolic(newlen) or IsValue(newlen) then newlen := newlen.ev(); fi;
                tentry := List([0..newlen-1], i -> var.fresh_t("scal", newt));
                Append(newscal, tentry);
                d.value := Value(TArray(newt, newlen), tentry);
            fi;
        od;
        self.scalarized := Set(newscal);

        return code;
    end,

    # NOTE: make this more general, in particular, if variable does not appear
    # on the left-hand side of the assignment it won't be declared, scalar
    # variables won't be declared if they are only used inside call() (pathological case)
    # Vectors can in fact be only used inside call, so this situtation is handled correctly.
    declareVars := meth(self, code)
        local vars, vects;
        vars  := Filtered(Set(ConcatList(Collect(code, @.cond(e->IsCommand(e) and IsBound(e.op_out))), e->e.op_out())), IsVar);
        vects := Filtered(code.free(), x -> IsArray(x.t));

        IntersectSet(vects, self.decls);
	SubtractSet(vects, self.free);
	SubtractSet(vars,  self.free);
	
        if Length(vars) > 0 then code := decl(vars, code); fi;
        if Length(vects) > 0 then code := decl(vects, code); fi;
        return code;
    end,

    # Old and slow scalarizer no longer works, but old scalarizer did not make an 
    # assumption of "constant geomery code" and the current scalarizer does make that
    # assumption. It is becoming a limitation for certain applications, so we might need
    # to redesign our current fast scalarizer to be more like the old one.
    #
    #D scalarize := (self, c) >> Scalarize(c, self.decls),

    showTimes := meth(self)
        local i;
        Print("times := [\n");
        for i in [1..Length(self.times)] do
	    PrintEval("$1,  $2,\n", i, StringDouble("%.3g", self.times[i]));
	od;
        Print("];\n");
    end,
	    
    __call__ := meth(self, c, opts)
        local dims, root, stage, compileStrategy, t, i;

        if IsBound(c.dimensions) then dims := c.dimensions; fi;
        if IsBound(c.root) then root := c.root; fi;

        if opts.printWebMeasure then Print("web:measure\n"); fi;
        self.curcode := [Copy(c)];

        # self.times keeps compilation time information to be used for profiling
        if not IsBound(self.times) or Length(self.times)<>Length(opts.compileStrategy) then 
            self.times := Replicate(Length(opts.compileStrategy), 0.0); fi;

        for i in [1..Length(opts.compileStrategy)] do
            stage := opts.compileStrategy[i];
            if IsMeth(stage) and IsCallableN(stage, 2) then 
                [c,t] := UTimedAction(stage(self, c, opts));
            elif IsMeth(stage) then 
                [c,t] := UTimedAction(stage(self, c));
            elif IsCallableN(stage, 2) then 
                [c,t] := UTimedAction(stage(c, opts));
            else 
                [c,t] := UTimedAction(stage(c));
            fi;
            Add(self.curcode, Copy(c));
            self.times[i] := self.times[i] + t;
	    self.timingStatus(i, stage, t); # print timing info

            if opts.printWebMeasure then Print("web:measure\n"); fi;
        od;

        if IsBound(dims) then c.dimensions := dims; fi;
        if IsBound(root) then c.root := root; fi;
        return c;
    end
));
