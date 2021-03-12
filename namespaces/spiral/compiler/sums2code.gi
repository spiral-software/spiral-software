
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


TempArrayType := (child, y, x, index) ->     
    When( IsBound(child.a.t_in), child.a.t_in[index], let(
          X := Flat([x])[1], # without type unification everything has same type
          When(X.t.t=TComplex or Flat([y])[1].t.t=TComplex, TComplex, X.t.t)
));

TempArray := (y, x, child) -> let(
    cols := Flat([Cols(child)]),
    StripList( List( [1..Length(cols)], i ->
        TempVec( TArray( TempArrayType(child, y, x, i), cols[i]))
)));

# HACKed in. This should just be rolled into TempArray or otherwise be handled cleanly.
TempArraySeq := function(y, x, child) 
    local dims, P;
    if ObjId(child)=BB then 
      dims := child.child(1).dims();
      P    := child.child(1).P;
    else
      dims := child.dims();
      P    := child.P;
    fi;

    return(
        StripList(List(
          Flat([ dims[2]/P ]), z ->
            TempVec(
                #TArray(TempArrayType(child, Flat([y])[1], Flat([x])[1]), z))
                TArray(TempArrayType(child, y, x, 1), z))
            )));
end;

#F _CodeSums(<sums-spl>, <y-output-var>, <x-input-var>)
#F
#F Generates unoptimize loop code for <sums-spl>
#F
_CodeSums := function(sums, y, x)
    local code;
    code := sums.code(y,x);
    code.dimensions := sums.dimensions;
    code.root := sums;
    return code;
end;

#CodeSums := _CodeSums;

_CodeSumsAcc := function(sums, y, x)
    local code;
    code := _CodeSums(sums, y, x);
    code := SubstTopDownNR(code, [assign, [@(0,nth), @(1, var, e->Same(e,y)), ...], @(3)], 
                     e -> assign_acc(@(0).val, @(3).val));
    return code;
end;

# =======================================================================================
# Code generation templates
# [ Data, Blk, Blk1, Prm, R, W, SUM, ISum, Compose ]
# =======================================================================================
Formula.code := (self,y,x) >> self.child(1).code(y, x);

BB.code := (self,y,x) >> MarkForUnrolling(_CodeSums(self.child(1), y, x));

Buf.code := (self,y,x) >> _CodeSums(self.child(1), y, x);

COND.code := (self,y,x) >>
    IF(self.cond, 
    _CodeSums(self.child(1), y, x),
    _CodeSums(self.child(2), y, x));

Data.code := meth(self, y, x) 
    local val, t;
    if self.inline then
    # we set live_out to prevent copy propagation 
    # self.var.setAttr("live_out")
    return decl(self.var, chain(
        assign(self.var, self.value.tolist()),
        _CodeSums(self.child(1), y, x)));
    else
    val := self.value;
    if IsBound(val.tolist) then
        if IsBound(val.range) and val.range() <> false then
        t := TArray(val.range(), val.domain());
        val := t.value(val.tolist());
        else
        val := V(val.tolist());
        fi;
    else val := Cond(IsSPL(val), V(MatSPL(val)), val);
    fi;
    return data(self.var, val, _CodeSums(self.child(1), y, x));
    fi;
end;

Diag.code := meth(self, y, x)
    local i, elt;
    i := Ind(); 
    elt := self.element.lambda().at(i);
    return loop(i, self.element.domain(), assign(nth(y,i), elt * nth(x,i)));
end;

RCDiag.code := meth(self, y, x)
    local i, elt, re, im;
    i := Ind(); 
    elt := self.element.lambda();
    re := elt.at(2*i);
    im := elt.at(2*i+1);
    return loop(i, self.element.domain()/2, chain(
	    assign(nth(y,2*i),   re * nth(x,2*i) - im * nth(x,2*i+1)), 
	    assign(nth(y,2*i+1), im * nth(x,2*i) + re * nth(x,2*i+1))));
end;

ColVec.code := meth(self, y, x)
    local i, datavar, diag, func;
    i := Ind();
    func := self.element;
    if IsBound(self.inline) and self.inline then
    return loop(i, Rows(self), assign(nth(y,i), mul(func.lambda().at(i), nth(x,0))));
    else
    diag := V(func.lambda().tolist());
    datavar := Dat(diag.t);
    return
    data(datavar, diag,
        loop(i, Rows(self), assign(nth(y,i), mul(nth(datavar,i), nth(x,0)))));
    fi;
end;

RowVec.code := meth(self, y, x)
    local i, datavar, diag, func, t;
    i := Ind();
    t := TempVar(x.t.t);
    func := self.element;
    if IsBound(self.inline) and self.inline then
    return chain(
        assign(t,0),
        loop(i, Cols(self), assign(t, add(t, mul(func.lambda().at(i), nth(x,i))))),
        assign(nth(y,0), t));
    else
    diag := V(func.lambda().tolist());
    datavar := Dat(diag.t);
    return
    data(datavar, diag,
        chain(
        assign(t,0),
        loop(i, Cols(self), assign(t, add(t, mul(nth(datavar,i), nth(x,i))))),
        assign(nth(y,0), t)));
    fi;
end;

Scale.code := (self, y, x) >> let(i := Ind(), 
    chain(
	_CodeSums(self.child(1), y, x),
	loop(i, Rows(self), 
	    assign(nth(y,i), mul(self.scalar, nth(y,i))))));

I.code := (self, y, x) >> let(i := Ind(), 
    loop(i, Rows(self), assign(nth(y,i), nth(x,i))));


Blk2code := (blk, y, x) -> let(b := V(blk.element).v, 
    t0 := TempVar(x.t.t), t1 := TempVar(x.t.t), 
    decl([t0, t1], chain(
	    assign(t0, mul(b[1].v[1], nth(x,0)) + mul(b[1].v[2], nth(x,1))),
	    assign(t1, mul(b[2].v[1], nth(x,0)) + mul(b[2].v[2], nth(x,1))),
	    assign(nth(y,0), t0),
	    assign(nth(y,1), t1)
	)));

Blk4code := (blk, y, x) -> let(b := V(blk.element).v, 
    t0 := TempVar(x.t.t), t1 := TempVar(x.t.t), 
    t2 := TempVar(x.t.t), t3 := TempVar(x.t.t), 

    x0 := TempVar(x.t.t), x1 := TempVar(x.t.t), 
    x2 := TempVar(x.t.t), x3 := TempVar(x.t.t), 
    decl([x0,x1,x2,x3, t0, t1, t2, t3], chain(
	    assign(x0, nth(x,0)),
	    assign(x1, nth(x,1)),
	    assign(x2, nth(x,2)),
	    assign(x3, nth(x,3)),

	    assign(t0, mul(b[1].v[1], x0) + mul(b[1].v[2], x1) + 
		       mul(b[1].v[3], x2) + mul(b[1].v[4], x3)),
	    assign(t1, mul(b[2].v[1], x0) + mul(b[2].v[2], x1) + 
		       mul(b[2].v[3], x2) + mul(b[2].v[4], x3)),
	    assign(t2, mul(b[3].v[1], x0) + mul(b[3].v[2], x1) + 
		       mul(b[3].v[3], x2) + mul(b[3].v[4], x3)),
	    assign(t3, mul(b[4].v[1], x0) + mul(b[4].v[2], x1) + 
		       mul(b[4].v[3], x2) + mul(b[4].v[4], x3)),
	    assign(nth(y,0), t0),
	    assign(nth(y,1), t1),
	    assign(nth(y,2), t2),
	    assign(nth(y,3), t3)
    )));

Blk.code := meth(self, y, x) 
    local i, j, t, datavar, matval;
    if Rows(self)=2 and Cols(self)=2 then return Blk2code(self, y, x); fi;
    if Rows(self)=4 and Cols(self)=4 then return Blk4code(self, y, x); fi;
    j := Ind();
    i := Ind();
    t := TempVar(x.t.t);
    matval := V(self.element);
    datavar := Dat(matval.t);
    return 
    data(datavar, matval,
    loop(j, Rows(self),
        decl(t, 
        chain(
            assign(t, 0),
            loop(i, Cols(self),
            assign(t, add(t, mul(nth(nth(datavar, j), i), nth(x, i))))),
            assign(nth(y,j), t)))));
end;

toeplitz.code := meth(self, y, x) 
    local i, j, t, datavar, matval, dataload;
    j := Ind();
    i := Ind();
    t := TempVar(x.t.t);
    matval := V(self.obj.element);
    datavar := Dat(matval.t);
    dataload := TempVec(TArray(TDouble, Rows(self)*Cols(self)));
    return 
    data(datavar, matval, decl(dataload, chain(
    loop(j, Rows(self),
        loop(i, Cols(self),
        assign(nth(dataload, j*Cols(self)+i), nth(nth(datavar,j),i)))),
    
    loop(j, Rows(self),
        decl(t, 
        chain(
            assign(t, 0),
            loop(i, Cols(self),
            assign(t, add(t, mul(nth(dataload, j*Cols(self)+i), nth(x, i))))),
            assign(nth(y,j), t)))))));
end;

Blk1.code := (self, y, x) >> 
   assign(nth(y,0), mul(toExpArg(self.element), nth(x,0)));

BlkConj.code := (self, y, x) >> 
   assign(nth(y,0), conj(nth(x,0)));

Prm.code := (self, y, x) >> let(i := Ind(), func := self.func.lambda(),
    loop(i, Rows(self), 
    assign(nth(y, i),
           nth(x, func.at(i)))));

Z.code := (self, y, x) >> let(i := Ind(), ii := Ind(), z := self.params[2],
    chain(
    loop(i, Rows(self)-z, 
        assign(nth(y, i), nth(x, i+z))),
    loop(i, z, 
        assign(nth(y, i+(Rows(self)-z)), nth(x, i)))));

O.code := (self, y, x) >> let(i := Ind(), 
    loop(i, self.params[1], 
        assign(nth(y, i), V(0))));

Gath.code := meth(self, y, x)
    local i, func, rfunc, ix;
    i := Ind(); func := self.func.lambda();

    if IsBound(self.func.rlambda) then
	rfunc := self.func.rlambda();
	ix := var.fresh_t("ix", TInt);
	return chain(
	    assign(ix, func.at(0)),
	    assign(nth(y, 0), nth(x, ix)),
	    loop(i, self.func.domain()-1, 
		chain(assign(ix, rfunc.at(ix)), 
		      assign(nth(y, i+1), nth(x, ix)))));
    else 
	return loop(i, self.func.domain(), assign(nth(y,i), nth(x, func.at(i))));
    fi;
end;

Scat.code := meth(self, y, x)
    local i, func, rfunc, ix;
    i := Ind(); func := self.func.lambda();
    if IsBound(self.func.rlambda) then
	rfunc := self.func.rlambda();
	ix := var.fresh_t("ix", TInt);
	return chain(
	    assign(ix, func.at(0)),
	    assign(nth(y, ix), nth(x, 0)),
	    loop(i, self.func.domain()-1, 
		chain(assign(ix, rfunc.at(ix)), 
		      assign(nth(y, ix), nth(x, i+1)))));
    else 
	return loop(i, self.func.domain(), assign(nth(y,func.at(i)), nth(x, i)));
    fi;
end;

SUM.code := (self, y, x) >>
       chain(List(self.children(), c -> _CodeSums(c, y, x))); 

SUMAcc.code := (self, y, x) >>
   let(ii := Ind(), 
       When(not Same(ObjId(self.child(1)), Gath),  # NOTE: come up with a general condition
       chain(
           loop(ii, Rows(self), assign(nth(y, ii), V(0))), 
           List(self.children(), c -> _CodeSumsAcc(c, y, x))),
       chain(
           _CodeSums(self.child(1), y, x),
           List(Drop(self.children(), 1), c -> _CodeSumsAcc(c, y, x)))));


ISum.code := (self, y, x) >> let(myloop := When(IsSymbolic(self.domain), loopn, loop),
   myloop(self.var, self.domain, 
        _CodeSums(self.child(1), y, x)));

ISumAcc.code := (self, y, x) >> 
   let(ii := Ind(),
       chain(
       loop(ii, Rows(self), 
           assign(nth(y, ii), V(0))), 
       loop(self.var, 
            self.domain, 
        _CodeSumsAcc(self.child(1), y, x))));

Compose.code := meth(self,y,x) 
    local ch, numch, vecs, allow, i;
#VIENNA filtering DMPGath, DMPScat out here because they do not generate code 
#    ch := self.children();
    ch := Filtered(self.children(), i-> not i.name in ["DMPGath","DMPScat"]);
    numch := Length(ch);
    vecs := [y];
    allow := (x<>y);
    for i in [1..numch-1] do
        if allow and ObjId(ch[i])=Inplace then vecs[i+1] := vecs[i]; 
	else vecs[i+1] := TempVec(TArray(TempArrayType(y, x), Cols(ch[i])));
	fi;
    od;
    vecs[numch+1] := x;
    for i in Reversed([1..numch]) do
        if allow and ObjId(ch[i])=Inplace then vecs[i] := vecs[i+1]; 
	fi;
    od;

#Print(vecs, "\n");
    if vecs[1] = vecs[numch+1] then # everything was inplace
	vecs[1] := y;
    fi;
    if vecs[1] = vecs[numch+1] then # everything was inplace
	vecs[numch+1] := x;
    fi;
#Print(vecs, "\n");
    vecs := Reversed(vecs);
    ch := Reversed(ch);
    return decl( Difference(vecs{[2..Length(vecs)-1]}, [x,y]),
	chain( List([1..numch], i -> When(vecs[i+1]=vecs[i], 
		    compiler._IPCodeSums(ch[i], vecs[i]), 
		    _CodeSums(ch[i], vecs[i+1], vecs[i])))));
end;

ICompose.code := (self, y, x) >> 

#DoneSpiralHelp - How to make this work for any datatype (var t)
#Look into Compose.code for tempvec

  chain([let(
    t :=  Dat1d(x.t.t, Rows(self)),
    newind := Ind(self.domain),
    decl([t], chain(
       When(IsOddInt(self.domain),
       SubstVars(Copy(_CodeSums(self.child(1), y, x)), tab((self.var.id) := (V(0)))),
          skip()
       ),
       When(IsEvenInt(self.domain),
          chain(
          SubstVars(Copy(_CodeSums(self.child(1), t, x)), tab((self.var.id) := (V(0)))),
          SubstVars(Copy(_CodeSums(self.child(1), y, t)), tab((self.var.id) := (V(1))))
          ),
          skip()
       ),
      loop(newind, (self.domain/2), 
        chain([
        SubstVars(Copy(_CodeSums(self.child(1), x, y)), tab((self.var.id) := (2*newind)+2)),
        SubstVars(Copy(_CodeSums(self.child(1), y, x)), tab((self.var.id) := (2*newind)+3))
        ])
      )
    ))
  )]);

