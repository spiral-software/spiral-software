
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# This is the code generator I use for generating HDL descriptions.  The
# output from this is sent to my back-end. 

Class(HDLCodegen, Codegen, rec(
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

    Formula := meth(self, o, y, x, opts)

        local code, datas, prog, params, sub, initsub;

	[x, y] := self.initXY(x,y,opts);

	o := o.child(1);
	params := Set(Collect(o, param));

	datas := Collect(o, FDataOfs);
	o := BlockSums(opts.globalUnrolling, o);
	code := self(o, y, x, opts);
    code := ESReduce(code, opts);
	code := RemoveAssignAcc(code);
	code := BlockUnroll(code, opts);
        # code := PowerOpt(code);
	code := DeclareHidden(code);
	if IsBound(opts.isFixedPoint) and opts.isFixedPoint then
	    code := FixedPointCode(code, opts.bits, opts.fracbits);
	fi;

	sub := Cond(IsBound(opts.subName), opts.subName, "transform");
	initsub := Cond(IsBound(opts.subName), Concat("init_", opts.subName), "init");
#	code := func(TVoid, sub, Concatenation(params, [y, x]), code);
#	code := 

#	if IsBound(opts.generateInitFunc) and opts.generateInitFunc then
#	    prog := program(
#		decl(List(datas, x->x.var),
#		    chain(
#			func(TVoid, initsub, [], chain(List(datas, x -> SReduce(x.var.init, opts)))),
#			code
#		    )));
#	else
#	    prog := program(code);
#	fi;
	prog := code;
	prog.dimensions := o.dimensions;
	return prog;
    end,

    

#    BB := (self,o,y,x,opts) >> MarkForUnrolling(self(o.child(1), y, x, opts)),
    BB := (self,o,y,x,opts) >> MarkForUnrolling(
        When(IsBound(o.bbnum),
            o.bbnum,
            0
        ), 
        self(o.child(1), y, x, opts)
    ),

    Buf := (self,o,y,x,opts) >> self(o.child(1), y, x, opts),
    Grp := (self,o,y,x,opts) >> self(o.child(1), y, x, opts),

    COND := (self,o,y,x,opts) >> IF(
        When(IsFunction(o.cond), o.cond.at(0), o.cond),
        self(o.child(1), y, x, opts), self(o.child(2), y, x, opts)),

#    COND := (self,o,y,x,opts) >> IF(o.cond, self(o.child(1), y, x, opts), self(o.child(2), y, x, opts)),

    Diag := (self, o, y, x, opts) >> let(i := Ind(), elt := o.element.lambda(),
	loop(i, elt.domain(), assign(nth(y,i), elt.at(i) * nth(x,i)))),

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

    I := (self, o, y, x, opts) >> let(i := Ind(Rows(o)),
	loop(i, i.range, assign(nth(y,i), nth(x,i)))),

    Blk := (self, o, y, x, opts) >> Cond(
	Rows(o)=2 and Cols(o)=2, Blk2code(o, y, x),
        Rows(o)=4 and Cols(o)=4, Blk4code(o, y, x),
	let(j:=Ind(), i:=Ind(), t:=TempVar(x.t.t), mat:=V(o.element), d:=Dat(mat.t),
	    data(d, mat,
		loop(j, Rows(o), decl(t, chain(
		     assign(t, 0),
		     loop(i, Cols(o), assign(t, t + nth(nth(d,j),i) * nth(x,i))),
		     assign(nth(y,j), t))))))),

    toeplitz := (self, o, y, x, opts) >> self.Blk(o.obj, y, x, opts),

    Blk1 := (self, o, y, x, opts) >> assign(nth(y,0), mul(toExpArg(o.element), nth(x,0))),

    BlkConj := (self, o, y, x, opts) >> assign(nth(y,0), conj(nth(x,0))),


    Prm := (self, o, y, x, opts) >> let(i:=Ind(), func:=o.func.lambda(),
	loop(i, Rows(o), assign(nth(y, i), nth(x, func.at(i))))),
    
    L := meth(self, o, y, x, opts)
        return let(i:=Ind(), func:=o.lambda(),
            loop(i, Rows(o), assign(nth(y, i), nth(x, func.at(i)))).unroll());
    end,
   
    
    O := (self, o, y, x, opts) >> let(i:=Ind(),
	loop(i, o.params[1], assign(nth(y, i), V(0)))),

    DirectSum := (self,o,y,x,opts) >> self(o.sums(), y, x, opts),
    Tensor := (self,o,y,x,opts) >> self(o.sums(), y, x, opts),
    IDirSum := (self,o,y,x,opts) >> self(o.sums(), y, x, opts),

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

    SUM := (self, o, y, x, opts) >> chain(List(o.children(), c -> self(c, y, x, opts))),

    ISum := (self, o, y, x, opts) >> let(myloop := When(IsSymbolic(o.domain), loopn, loop),
    myloop(o.var, o.domain,
        self(o.child(1), y, x, opts))),

    # NOTE: get rid of _acc
    SUMAcc := (self, o, y, x, opts) >> let(ii := Ind(),
       When(not Same(ObjId(o.child(1)), Gath),  # NOTE: come up with a general condition
       chain(
           loop(ii, Rows(o), assign(nth(y, ii), V(0))),
           List(o.children(), c -> _acc(self(c, y, x, opts), y))),
       chain(
           self(o.child(1), y, x, opts),
           List(Drop(o.children(), 1), c -> _acc(self(c, y, x, opts), y))))),

    ISumAcc := (self, o, y, x, opts) >> let(ii := Ind(), chain(
    loop(ii, Rows(o), assign(nth(y, ii), V(0))),
    loop(o.var, o.domain, _acc(self(o.child(1), y, x, opts), y)))),

#     Compose := meth(self, o,y,x,opts)
#         local ch, numch, vecs, allow, i;
# #   if IsBound(opts._inplace) and opts._inplace then
# #       return chain(
# #       List(Reversed(o.children()), c -> self(c, x, x, opts))); fi;
#         #VIENNA filtering DMPGath, DMPScat out here because they do not generate code
#     ch := Filtered(o.children(), i-> not i.name in ["DMPGath","DMPScat"]);
#     numch := Length(ch);
#     vecs := [y];
#     allow := (x<>y);
#     for i in [1..numch-1] do
#             if allow and ObjId(ch[i])=Inplace then vecs[i+1] := vecs[i];
#         else vecs[i+1] := TempVec(TArray(TempArrayType(y, x), Cols(ch[i])));
#         fi;
#     od;
#     vecs[numch+1] := x;
#     for i in Reversed([1..numch]) do
#             if allow and ObjId(ch[i])=Inplace
#         then vecs[i] := vecs[i+1]; fi;
#     od;

#         # everything was inplace, make it go from x -> y as expected
#     if vecs[1] = vecs[numch+1] then vecs[1] := y; fi;
#     if vecs[1] = vecs[numch+1] then vecs[numch+1] := x; fi;

#     [vecs, ch] := [Reversed(vecs), Reversed(ch)];
#     return decl( Difference(vecs{[2..Length(vecs)-1]}, [x,y]),
#         chain( List([1..numch], i -> When(vecs[i+1]=vecs[i],
#             self(ch[i], vecs[i],   vecs[i], CopyFields(opts, rec(_inplace:=true))),
#             self(ch[i], vecs[i+1], vecs[i], opts)))));
#     end,

    Compose := meth(self, o,y,x,opts)
        local ch, numch, vecs, i, cmd, code;
        ch    := o.children();
        numch := Length(ch);
        vecs  := [y];

        # we like to evaluate right (last) to left (first), where the values
        # in parens refer to the position of the elements in the array.
        #
        # So here, we start at the output (first entry in array) and
        # walk towards the input, creating temporary arrays as necessary
        for i in [1..numch-1] do
            vecs[i+1] := When(ObjId(ch[i]) = Inplace,
                vecs[i],
                TempArray(y,x,ch[i])
            );
        od;

        # the last entry must be the input.
        vecs[numch+1] := x;

        # now we walk in the opposite direction, copying through
        # input as far as we can.
        for i in Reversed([1..numch]) do
            if ObjId(ch[i]) = Inplace then
                vecs[i] := vecs[i+1];
            fi;
        od;

        # If all children were evaluated inplace, the output will be in x
        # Make it go from x -> y as expected
        if vecs[1] = vecs[numch+1] then vecs[1] := y; fi;
        if vecs[1] = vecs[numch+1] then vecs[numch+1] := x; fi;

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


    Inplace := (self, o, y, x, opts) >>
#        self(o.child(1), x, x, CopyFields(opts, rec(_inplace:=true))),
        self(o.child(1), y, x, opts), # Compose will handle these somehow

    Data := meth(self, o, y, x, opts)
        local val;
        o.var.isData := true;
        val := When(IsFunction(o.value), o.value.tolist(), o.value);
        val := When(IsValue(val), val, o.var.t.value(val));
        return data(o.var, val, self(o.child(1), y, x, opts));
    end,

#     Data := (self, o, y, x, opts) >> decl(o.var, chain(
# 	    assign(o.var, o.value.tolist()),
# 	    self(o.child(1), y, x, opts))
#     ),

    #   NOTE: use pointers to do pingpong?
    ICompose := (self, o, y, x, opts) >>
        chain([let(
            t :=  Dat1d(x.t.t, Rows(o)),
            newind := Ind(o.domain),
            decl([t], chain(
            When(IsOddInt(o.domain),
            SubstVars(Copy(self(o.child(1), y, x, opts)), tab((o.var.id) := (V(0)))),
                skip()
            ),
            When(IsEvenInt(o.domain),
                chain(
                    SubstVars(Copy(self(o.child(1), t, x, opts)), tab((o.var.id) := (V(0)))),
                    SubstVars(Copy(self(o.child(1), y, t, opts)), tab((o.var.id) := (V(1))))
                ),
                skip()
            ),
            loop(newind, (o.domain/2),
                chain([
                    SubstVars(Copy(self(o.child(1), x, y, opts)), tab((o.var.id) := (2*newind)+2)),
                    SubstVars(Copy(self(o.child(1), y, x, opts)), tab((o.var.id) := (2*newind)+3))
                ])
            )
            )))])
));
