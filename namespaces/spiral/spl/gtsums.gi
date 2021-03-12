
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(ISumn, ISum, rec(
    abbrevs := [ (dom, expr) -> Checked(IsPosIntSym(dom), IsSPL(expr), [dom, expr]) ],

    new := meth(self, domain, expr)
        local res;
	# ??? if domain = 1 then return SubstBottomUp(expr, var, e->V(0));
	res := SPL(WithBases(self, rec(_children := [expr], domain := domain)));
	res.dimensions := res.dims();
	return res;
    end,
    #-----------------------------------------------------------------------
    unrolledChildren := self >> Error("not implemented"),
    #    List(listRange(self.domain), u ->
    #	       SubstBottomUp(Copy(self._children[1]), self.var, e->V(u))),
    #-----------------------------------------------------------------------
    rChildren := self >> [self.domain, self._children[1]],
    rSetChild := meth(self, n, what) 
        if n=1 then self.domain := what; 
	elif n=2 then self._children[1] := what; 
	else Error("<n> must be in [1,2]"); fi;
    end,
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.name, "(", self.domain, ",\n", 
	Blanks(i+is), self._children[1].print(i+is, is), "\n", Blanks(i), ")"),
    #-----------------------------------------------------------------------
    sums := self >> let(base := self.__bases__[1],
	base(self.var, self.domain, self._children[1].sums())),
));

Tensor.sumsn := meth(self)
    local ch, col_prods, row_prods, col_prods_rev, row_prods_rev, i, c, 
          cp1, cp2, rp1, rp2, cols, rows, prod, term, scat, gath;
    if self.isPermutation() then return
	Prm(ApplyFunc(self.fTensor, List(self.children(), c->c.sums().func)));
    elif ForAll(self.children(), x->ObjId(x)=Diag) then return
	Diag(ApplyFunc(diagTensor, List(self.children(), c->c.element)));
    fi;

    ch := self.children();
    col_prods := ScanL(ch, (x,y)->x*Cols(y), 1);
    col_prods_rev := Drop(ScanR(ch, (x,y)->x*Cols(y), 1), 1);
    #row_prods := ScanL(ch, (x,y)->x*Rows(y), 1);
    #row_prods_rev := DropLast(ScanR(ch, (x,y)->x*Rows(y), 1), 1);

    prod := [];
    for i in [1..Length(ch)] do
        c := ch[i];
        if not IsIdentitySPL(c) then
	    [cp1, cp2] := [col_prods[i], col_prods_rev[i]];
	    #[rp1, rp2] := [row_prods[i], row_prods_rev[i]];
	    [rows, cols] := Dimensions(c);
	    [scat, gath] := [[cp2], [cp2]];
	    if cp2 <> 1 then 
		Add(scat, 1);
		Add(gath, 1); fi;
	    if cp1 <> 1 then 
		Add(scat, cp2*rows); 
		Add(gath, cp2*cols); fi;

	    term := Scat(HH(cp1*rows*cp2, rows, 0, scat)) * 
	            c.sums() * 
		    Gath(HH(cp1*cols*cp2, cols, 0, gath));

	    if cp2 <> 1 then term := ISumn(cp2, term); fi;
	    if cp1 <> 1 then term := ISumn(cp1, term); fi;
	    Add(prod, term);
	fi;
    od;
    return Compose(prod);
end;

IterDirectSum.sumsn := self >> let(A := self.child(1),
    ISumn(self.domain,  
	  Scat(HH(Rows(self), Rows(A), 0, [1, Rows(A)])) *
#	  spiral.sigma.SumsSPL( # NOTE: subst to ind -- potentially unsafe! (other loops!)
	      SubstVars(Copy(A), rec((self.var.id) := ind(self.var.range, 1))).sums() *
	  Gath(HH(Cols(self), Cols(A), 0, [1, Cols(A)]))));

RowTensor.sumsn := self >> let(A := self.child(1),
    ISumn(self.isize,
	  Scat(HH(Rows(self), Rows(A), 0, [1, Rows(A)])) *
#	  spiral.sigma.SumsSPL(A) *
	  A.sums() * 
	  Gath(HH(Cols(self), Cols(A), 0, [1, Cols(A) - self.overlap]))));

