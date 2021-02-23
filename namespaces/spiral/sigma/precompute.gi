
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F ResolveMemos(<o>)
#F   This function removes all precomputed variables (with Data(v, expr, ...))
#F   by inserting the corresponding expressions
#F
ResolveMemos := o ->
   SubstTopDown(o, @(1,var,e->IsBound(e.mapping)), e->e.mapping);

_GenerateData := function(map, free, lambda)
    local newlambda, v, val, d, free1;
    if free = [] then
	newlambda := Copy(lambda);
	# NOTE: hardcoded [var, param] is ugly
	newlambda.expr := SubstParamsCustom(newlambda.expr, map, [var, param]);
	return newlambda.tolist();
    else
	v := free[1];
	d := [];
	free1 := Drop(free, 1);
	for val in [0..v.range-1] do
	    map.(v.id) := V(val);
	    Append(d, _GenerateData(map, free1, lambda));
	od;
	return d;
    fi;
end;

#Declare(fComputeOnline);

# PrecalculatedData - accumulating precalculated data variables so
# same constant data table (variable) could be reused in different places in code.
# Beware: search for table by key is inefficient here.

Class(PrecalculatedData, rec(

     # _tables is a list of rec(key, data, opts) items.
    _tables := [],

    __call__ := (self) >> self._tables,

    # keyForFunc - returns search key for function 'func' that has 'var' variables.
    # The key is just the same function with unnamed variables.

    keyForFunc := meth( self, func, vars)
        local i, j, map;
        map := rec();
        i := 1; j := Length(vars);
        while j>0 do
            map.(vars[i].id) := ind(vars[i].range, j);
            i := i+1; j := j-1;
        od;
	# NOTE: hardcoded [var, param] is ugly
        return SubstParamsCustom( Copy(func), map, [var, param]);
    end,

    # find(key, opts) - searching if data table already exists.
    find := meth( self, key, opts)
        local item;
        for item in self._tables do  
            #NOTE: opts.dataType - just to distinguish single precision from double 
            if item.key=key and item.opts.dataType=opts.dataType then
                return item.data;
            fi;
        od;
        return false;
    end,

    # add - remember data table assosiated with key, data is FData object.
    add := meth( self, key, data, opts)
        Add(self._tables, rec( key := key, data := data, opts := opts));
    end,

    clear := meth(self) 
        self._tables := []; 
    end,

));

GenerateData := function ( func, opts )
    local f, n, free, lambda, data, t, olvars, key, map, i;

    if ObjId(func)=FDataOfs then return func; fi;

    olvars := [];
    map := rec();

    func := SubstTopDown(Copy(func), fComputeOnline, 
        function(s)
            local v;
            v := var.fresh_t("o", s._children[1].t);
            v.range := s._children[1].range();
            Add(olvars, v);
            map.(v.id) := s._children[1];
            return v;
        end
    );

    lambda := func.lambda();
    # NOTE: Collect is a hack
    # NOTE: hardcoded 'param' is ugly
    free := Difference(lambda.expr.free() :: Collect(lambda.expr, param), lambda.vars);
    free := Filtered(free, IsLoopIndex);

    for i in free do map.(i.id) := fBase(i); od;

    Append(free, olvars);

    Constraint(ForAll(free, v->IsBound(v.range)));
    Sort(free, (x,y) -> DoubleString(Drop(x.id,1)) <= DoubleString(Drop(y.id,1)));

    if IsBound(opts.dataSharing) and opts.dataSharing.enabled then
        key := PrecalculatedData.keyForFunc(func, free);
        data := PrecalculatedData.find(key, opts);
        if ObjId(data)=false then 
            data := FData(_GenerateData(tab(), free, lambda));
            PrecalculatedData.add(key, data, opts);
        fi;
    else data := FData(_GenerateData(tab(), free, lambda));
    fi;

    n := lambda.domain();

    if (free=[] and olvars=[]) then return data.part(n,0);
    else return data.part(n, n * fTensor(List(free, i->map.(i.id))).at(0));
    fi;
end;

_GenerateInitCode := function(data, N, idx, free, lambda)
    local i, v, res;
    if free = [] then
	i := lambda.vars[1];
	res := loop(i, lambda.domain(), 
		assign(nth(data, idx+i), 
		       When(data.t.t = lambda.expr.t, 
			        lambda.expr,
			    # else
				tcast(data.t.t, lambda.expr))));

        # ObjId<>loop, when i.range was 1, loop just returned its body
	if i.range <= 16 and ObjId(res)=loop then return SReduceSimple(res.unroll()); 
	else return SReduceSimple(res);
	fi;
    else
	v := free[1];
        # careful here, since N and v.range can be symbols we have to use div ,
        # since N/v.range will become an fdiv (floating point division) in the code,
        # div = integer division with divisible arguments (allows normal simplification as in fdiv)
	N := div(N, v.range); 
	return loop(v, v.range,
	    _GenerateInitCode(data, N, idx+v*N, Drop(free,1), lambda));
    fi;
end;

GenerateInitCode := function ( func, data_container, opts )
    local f, N, n, free, lambda, data, map, initcode, res, range, key;
    if ObjId(func)=FDataOfs then return func; fi;
    lambda := func.lambda();

    # NOTE: Collect is a hack
    # NOTE: hardcoded 'param' is ugly
    free := Difference(lambda.expr.free() :: Collect(lambda.expr, param), lambda.vars);
    free := Filtered(free, IsLoopIndex);

    Constraint(ForAll(free, v->IsBound(v.range)));
    Sort(free, (x,y) -> DoubleString(Drop(x.id,1)) <= DoubleString(Drop(y.id,1)));

    n := func.domain();
    N := n * Product(free, x->x.range);

    range := func.range();

    if IsBound(opts.dataSharing) and opts.dataSharing.enabled then
        key := PrecalculatedData.keyForFunc(func, free);
        data := PrecalculatedData.find(key, opts);
    else
        data := false;
    fi;

    if data=false then 
        if IsInt(range) or IsSymbolic(range) then 
            data := data_container(TInt, N); # index mapping function, not diag
        else 
	    if opts.unifyStoredDataType = "input" then
		range := UnifyTypes([range, opts.XType.t]);
	    elif opts.unifyStoredDataType = "output" then
		range := UnifyTypes([range, opts.YType.t]);
	    elif IsType(opts.unifyStoredDataType) then 
		range := UnifyTypes([range, opts.unifyStoredDataType]);
	    fi;
            data := data_container(range, N);
        fi;
        data.init := _GenerateInitCode(data, N, 0, free, lambda);

        if IsBound(opts.dataSharing) and opts.dataSharing.enabled then    
            PrecalculatedData.add(key, data, opts);
        fi;
    fi;

    if free=[] then
        res := FData(data).part(n,0);
    else
        res := FData(data).part(n, n * fTensor(List(free, fBase)).at(0));
    fi;
    res := res.setRange(range);
    res.init := res.var.init;
    
    return res;
end;

_Process_fPrecompute := (o, precompute_func, precond) -> SubstBottomUpRules(o, [
    [ [fPrecompute, @(1)],
      (e,cx) -> When(precond(e,cx) and e.rank()=0, precompute_func(ResolveMemos(e)), e),
      "ContextDependentLocalize_fPrecompute"],
]);

# Process_fPrecompute(<sums>, <opts>)
#
Process_fPrecompute := function(sums, opts)
    local bb, b, has_free_vars, precompute_func;

    # Autolib needs ability to disable fPrecompute when it uses strategies from LocalConfig.bench
    if IsBound(opts.disablePrecompute) and opts.disablePrecompute then
        return sums;
    fi;

    if IsBound(opts.dataSharing) and opts.dataSharing.enabled then
    	sums := ApplyStrategy(sums, opts.dataSharing.precomputeStrategy, UntilDone, opts);
    fi;

    bb := Collect(sums, @(1, [BB, RecursStep]));
    for b in bb do
        b.precomputed_free := b.free();
    od;

    has_free_vars := (e, cxentry) -> 
        (CollectNR(e, param) <> []) or
        (Intersection(e.free(), cxentry[1].precomputed_free)<>[]);

    precompute_func := When(opts.generateInitFunc, 
        e -> GenerateInitCode(e, Dat1d, opts), 
        e -> GenerateData(e, opts));

    sums := _Process_fPrecompute(sums, precompute_func,
           (e, cx) -> (
                        ((not IsBound(cx.BB) or cx.BB=[]) and 
                        (not IsBound(cx.RecursStep) or cx.RecursStep=[]) and
                        (not IsBound(cx.RS) or cx.RS=[]) and        
                        (not IsBound(cx.RSBase) or cx.RSBase=[])) or 
                        (IsBound(cx.BB) and cx.BB<>[] and has_free_vars(e, cx.BB)) or 
	                    (IsBound(cx.RecursStep) and cx.RecursStep<>[] and has_free_vars(e, cx.RecursStep)) 
                    )
                    );
    return sums;
end;
