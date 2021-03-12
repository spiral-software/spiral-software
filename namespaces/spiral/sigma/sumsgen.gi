
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(LegacySumsGen, HierarchicalVisitor, rec(

   __call__ := meth(arg)

        Constraint(Length(arg) >= 2);

        #BWD: I believe LegacySumsGen.visit only expects Formula objects
        if not PatternMatch(arg[2], Formula, empty_cx()) then
            arg[2] := Formula(arg[2]);
        fi;

        return ApplyFunc(arg[1].visit, arg{[2..Length(arg)]});

   end,

	Formula := meth(self, o, opts)
		local spl, sums;
		spl := _children(o)[1];

    	if IsBound(spl.memo_sums) then
    		sums := spl.memo_sums;
    	else
    	sums := spl.sums();
    		spl._sums := sums;
    	fi;
    	sums.root := spl;
    	#sums.sums := self >> self;
    	return sums;
	end,
));

Class(DefaultSumsGen, HierarchicalVisitor, rec(
            
    __call__ := meth(arg)
        local res;
        res := ApplyFunc(arg[1].visit, arg{[2..Length(arg)]});
        trace_log.addExpansion(ObjId(arg[2]), arg[2], res, var);
        return res;
   end, 

    _directSumSums := (self, obj, roverlap, coverlap, opts) >>
        let(cspans  :=  BaseOverlap.spans(roverlap, List(obj.children(), Cols)),
            rspans  :=  BaseOverlap.spans(coverlap, List(obj.children(), Rows)),
            nblocks  :=  obj.numChildren(),
            C  :=  Cols(obj),
            R  :=  Rows(obj),
            OP  :=  When(coverlap > 0, SUMAcc, SUM),
            OP( Map([1..nblocks],
                 i -> Compose( #W(N, bksize_cols, fTensor(fBase(nblocks, ind), fId(bksize)))
                      WID(R, 1 + rspans[i][2] - rspans[i][1], rspans[i][1]-1),
                      self(obj.child(i), opts),
                      RID(C, 1 + cspans[i][2] - cspans[i][1], cspans[i][1]-1) )))),

    Formula     := (self, o, opts) >> self(_children(o)[1], opts),
    NonTerminal := (self, o, opts) >> o,
    PermClass   := (self, o, opts) >> Prm(o), 

    Rot := (self, o, opts) >> self(ExpandRotationsSPL(o), opts),
    Sym := (self, o, opts) >> self(o.getObj(), opts),

    O  := (self, o, opts) >> o,
    L  := (self, o, opts) >> Prm(o), 
    F  := (self, o, opts) >> self(o.getObj(), opts),
    I  := (self, o, opts) >> Prm(fId(o.params[1])), 
    
    RC := (self, o, opts) >> CopyFields(o, rec(_children := [self(o.child(1), opts)])),

    Perm := (self, o, opts) >> let(
	N := o.size,
	plist_r := Permuted([1..N], o.element^-1) - 1, # we want 0-based perms
	#plist_w := Permuted([1..N], o.element) - 1,    # we want 0-based perms
	Gath(FList(N, plist_r).setRange(N))
    ),

    Diag := (self, o, opts) >> o,
    RCDiag := (self, o, opts) >> o,
    Mat := (self, o, opts) >> Blk(o.element),
    Sparse := (self, o, opts) >> Blk(MatSparseSPL(o)),

    B := (self, o, opts) >> Prm(fB(o.size, o.l)),
    B2 := (self, o, opts) >> Prm(o), 
    Scale := (self, o, opts) >> Inherit(o, rec(_children := [self(o.child(1), opts)])),

    Conjugate := (self, o, opts) >> let(
	O := o.child(1), perm := o.child(2), Tperm := perm.transpose(),
	self(Compose(Tperm, O, perm), opts)
    ),

    Compose := (self, o, opts) >> Cond(o.isPermutation(),
	Prm(ApplyFunc(fCompose, List(Reversed(o.children()), c -> self(c, opts).func))),
	Compose(Map(o.children(), c -> self(c, opts)))
    ),

    Cross := (self, o, opts) >> Cond(o.isPermutation(),
	Prm(ApplyFunc(fCross, List(o.children(), c -> self(c, opts).func))),
	Cross(Map(o.children(), c -> self(c, opts)))
    ),

    ICompose := (self, o, opts) >> CopyFields(o, rec(
        _children := List(o.children(), c -> self(c, opts))
    )),
		
    DirectSum := (self, o, opts) >> Cond(o.isPermutation(),
	Prm(ApplyFunc(fDirsum, List(o.children(), c->self(c, opts).func))),
        self._directSumSums(o, 0, 0, opts)
    ),
    
    DelayedDirectSum := (self, o, opts) >> CopyFields(o, rec(
        _children := List(o.children(), c -> self(c, opts))
    )),
		
    RowDirectSum := (self, o, opts) >> self._directSumSums(o, o.overlap, 0, opts),
    ColDirectSum := (self, o, opts) >> self._directSumSums(o, 0, o.overlap, opts),

    RowTensor := (self, o, opts) >> let(
        A := o.child(1),
	i := Ind(o.isize),
	ISum(i, i.range,
	    Scat(fTensor(fBase(o.isize, i), fId(Rows(A)))) *
	    SumsSPL(A, opts) *
	    Gath(fAdd(Cols(o), Cols(A), i * (Cols(A) - o.overlap))))
    ),

    ColTensor := (self, o, opts) >> let(
        A := o.child(1),
	i := Ind(o.isize),
	ISumAcc(i, i.range,
	    Scat(fAdd(Rows(o), Rows(A), i * (Rows(A) - o.overlap))) *
	    SumsSPL(A, opts) *
	    Gath(fTensor(fBase(o.isize, i), fId(Cols(A)))))
    ),

    Tensor := meth(self, o, opts)
    	local ch, col_prods, row_prods, col_prods_rev, row_prods_rev, i, j1, j2, j1b, j2b, bkcols, bkrows, prod, term;

    	if o.isPermutation() then 
	    return Prm(ApplyFunc(o.fTensor, List(o.children(), c->self(c, opts).func)));
	elif ForAll(o.children(), x->ObjId(x)=Diag) then 
	    return Diag(ApplyFunc(diagTensor, List(o.children(), c->c.element)));
	fi;

	ch := o.children();
	col_prods := ScanL(ch, (x,y)->x*Cols(y), 1);
	col_prods_rev := Drop(ScanR(ch, (x,y)->x*Cols(y), 1), 1);

	row_prods := ScanL(ch, (x,y)->x*Rows(y), 1);
	row_prods_rev := Drop(ScanR(ch, (x,y)->x*Rows(y), 1), 1);

	prod := [];
	for i in [1..Length(ch)] do
	    if not IsIdentitySPL(ch[i]) then
		bkcols := Cols(ch[i]);
		bkrows := Rows(ch[i]);
		j1 := Ind(col_prods[i]);
		j2 := Ind(row_prods_rev[i]);
		j1b := When(j1.range = 1, 0, fBase(j1));
		j2b := When(j2.range = 1, 0, fBase(j2));
	
		term := Scat(o.scatTensor(i)(Filtered([j1b, fId(bkrows), j2b], x->x<>0))) *
		        self(ch[i], opts) *
			Gath(o.gathTensor(i)(Filtered([j1b, fId(bkcols), j2b], x->x<>0)));
	
		if j2.range <> 1 then
		    term := ISum(j2, j2.range, term); 
		fi;
	
		if j1.range <> 1 then
		    term := ISum(j1, j1.range, term); 
		fi;
	
		Add(prod, term);
	    fi;
	od;
	return Compose(prod);
    end,

    IDirSum := (self, o, opts) >> let(
	bkcols := Cols(o.child(1)),
	bkrows := Rows(o.child(1)),
	nblocks := o.domain,
	cols := Cols(o), 
	rows := Rows(o),

	ISum(o.var, o.domain,
	    Compose(Scat(fTensor(fBase(nblocks, o.var), fId(bkrows))),
		    self(o.child(1), opts),
		    Gath(fTensor(fBase(nblocks, o.var), fId(bkcols)))))
    ),
        
    HStack := (self, o, opts) >> let(		
	ch   := List(o.children(), e -> self(e, opts)),
	l    := Length(ch),
	it0  := Scat(fId(Rows(o))) * ch[1] * Gath(H(Cols(o), Cols(ch[1]), 0, 1)), 
	ofs  := ScanL(ch, (p,x)->p+Cols(x), 0),
	its  := List([2..l], i -> 
	           ScatAcc(fId(Rows(o))) * ch[i] * Gath(H(Cols(o), Cols(ch[i]), ofs[i], 1))), 
	SUM( [it0] :: its )
    ),

    VStack := (self, o, opts) >> let(		
	ch   := List(o.children(), e -> self(e, opts)),
	l    := Length(ch),
	ofs  := ScanL(ch, (p,x)->p+Rows(x), 0),
	SUM(List([1..l], i -> 
	        Scat(H(Rows(o), Rows(ch[i]), ofs[i], 1)) * ch[i] * Gath(fId(Cols(o)))))
    ),

    HStack1 := (self, o, opts) >> let(		
    	ch   := List(o.children(), e -> self(e, opts)),
    	l    := Length(ch),
    	it0  := Scat(fId(Rows(o))) * ch[1] * Gath(H(Cols(o), Cols(ch[1]), 0, 1)), 
    	ofs  := ScanL(ch, (p,x)->p+Cols(x), 0),
    	its  := List([2..l], i -> 
    	           Scat(fId(Rows(o))) * ch[i] * Gath(H(Cols(o), Cols(ch[i]), ofs[i], 1))), 
    	SUM( [it0] :: its )
    ),

    VStack := (self, o, opts) >> let(		
	ch   := List(o.children(), e -> self(e, opts)),
	l    := Length(ch),
	ofs  := ScanL(ch, (p,x)->p+Rows(x), 0),
	SUM(List([1..l], i -> 
	        Scat(H(Rows(o), Rows(ch[i]), ofs[i], 1)) * ch[i] * Gath(fId(Cols(o)))))
    ),

    IterHStack := (self, o, opts) >> let(
	bkcols := Cols(o.child(1)),
	bkrows := Rows(o.child(1)),
	nblocks := o.domain,
	cols := Cols(o), rows := Rows(o),
	j := Ind(nblocks-1),
	ch := self(o.child(1), opts),

	SUM(
	    Scat(fId(bkrows)) * 
	    SubstVars(Copy(ch), rec((o.var.id) := V(0))) *
	    Gath(fTensor(fBase(nblocks, 0), fId(bkcols))),

	    ISum(j, j.range,
		ScatAcc(fId(bkrows)) *
		SubstVars(Copy(ch), rec((o.var.id) := j+1)) *
		Gath(fTensor(fBase(nblocks, j+1), fId(bkcols)))
	    )
	)
    ),

    IterHStack1 := (self, o, opts) >> let(
    	bkcols := Cols(o.child(1)),
    	bkrows := Rows(o.child(1)),
    	nblocks := o.domain,
    	cols := Cols(o), rows := Rows(o),
    	j := Ind(nblocks),
    	ch := self(o.child(1), opts),

	    ISum(j, j.range,
		Scat(fId(bkrows)) *
		SubstVars(Copy(ch), rec((o.var.id) := j)) *
		Gath(fTensor(fBase(nblocks, j), fId(bkcols)))
    	)
    ),


    DPWrapper := (self, o, opts) >> self(o.child(1), opts),

    IterVStack := (self, o, opts) >> let(
	bkcols := Cols(o.child(1)),
	bkrows := Rows(o.child(1)),
	nblocks := o.domain,
	cols := Cols(o), rows := Rows(o),
	ISum(o.var, o.domain,
	    Scat(fTensor(fBase(nblocks, o.var), fId(bkrows))) *
	    self(o.child(1), opts) *
	    Gath(fId(bkcols)))
    ),

    ISum := (self, o, opts) >> CopyFields(o, rec(_children := [self(o._children[1], opts)])),

    RecursStep := ~.ISum,
    VPerm      := (self, o, opts) >> o,
    BlockVPerm := (self, o, opts) >> o,
    ExpDiag    := (self, o, opts) >> o,

    VTensor    := (self, o, opts) >> ObjId(o)(self(o.child(1), opts), o.vlen),

    SumsBase := meth(self, o, opts) 
       local children, i, res;
       res := Copy(o);
       ## MRT, aug08
       # IsSPL -> IsSPL and IsFunction
       # joint SPL/Sigma objects like fId were improperly triggering the
       # SPL condition, resulting in illegal rewrites like:
       #   Scat(fId(2)) -> Scat(Prm(fId(2))
       
       # YSV: This is still buggy, because VTensor(L) --> VTensor(L)
       children := Map(res.rChildren(), c -> Cond(IsSPL(c) and not IsFunction(c), self(c, opts), c));
       for i in [1..Length(children)] do
           res.rSetChild(i, children[i]);
       od;
       return res;
    end,

     Split := (self, o, opts) >> o,
     Glue := (self, o, opts) >> o,
     VIxL := (self, o, opts) >> o,

));

