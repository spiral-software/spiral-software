
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

# utility functions

Range0 := (a) -> When(IsList(a), [0..Length(a)-1], [0..a-1]);
Range1 := (a) -> When(IsList(a), [1..Length(a)], [1..a]);

# special rules for annotating the recursion level for all variables. 
# allows us to just keep the variables, instead of also needing to keep
# the ISums.

_checkTag := function(e, cx)
    local sums;

    if IsBound(e.var.order) then
        return false;
    fi;

    sums :=  Filtered(cx.parents, e -> ObjId(e) = ISum);

    return (sums = [] or IsBound(Last(sums).var.order));
end;

_setTag := function(e, cx)
    local sums;

    sums :=  Filtered(cx.parents, e -> ObjId(e) = ISum);

    if sums = [] then
        e.var.order := 0;
    else
        e.var.order := Last(sums).var.order + 1;
    fi;

    return e;
end;

Class(_MissEstPreprocessJams, RuleSet);
Class(_MissEstSanitize, RuleSet);
Class(_MissEstRules, RuleSet);
Class(_MissEstCleanup, RuleSet);
Class(_MissEstInplace, RuleSet);

Class(_GSWrap, SumsBase, rec(
    __call__ := (self, rng, dmn, payload) >> WithBases(self, rec(
        _dmn := dmn,
        _rng := rng,
        _payload := payload,
        _children := []
    )),
    rChildren := (self) >> [],
    dims := (self) >> [
        StripList(List(self.rng(), (l) -> l.size)),
        StripList(List(self.dmn(), (l) -> l.size))
    ],
    rSetChild := (self, n, what) >> Error("no kids"),
    rng := (self) >> self._rng,
    dmn := (self) >> self._dmn,
    print := (self,i,is) >> Print(self.name)
));


Class(_AltBB, BB);
Class(_AltInplace, Inplace);

# wrapper needs an identifier, for later matching since two wraps 
# are always spawned. See _MissEstInplace.InplaceExpand rule.
#
Class(_InplaceWrap, _GSWrap);
#SumsBase, BaseMat, rec(
#    new := (self, id) >> SPL(WithBases(self, rec(
#        id := id
#    )))
#));
        
Class(_KeepScat, Scat);
Class(_KeepGath, Gath);

RewriteRules(_MissEstPreprocessJams, rec(

 ComposeGathGath := ARule(Compose, [ @(1, Gath), @(2, [Gath, Prm]) ], # o 1-> 2->
     e -> [ Gath(fCompose(@(2).val.func, @(1).val.func)) ]),

 ComposeScatScat := ARule(Compose, [ @(1, [Scat, ScatAcc]), @(2, [Scat, ScatAcc]) ], # <-1 <-2 o
     e -> [ Cond(ObjId(@(1).val)=ScatAcc or ObjId(@(2).val)=ScatAcc,
                 ScatAcc(fCompose(@(1).val.func, @(2).val.func)),
                 Scat   (fCompose(@(1).val.func, @(2).val.func))) ]),


 PullInRight := ARule( Compose,
       [ @(1, [Prm, Scat, ScatAcc, TCast, PushR, PushLR, Conj, ConjL, ConjR, ConjLR, FormatPrm]),
         @(2, [RecursStep, Grp, BB, SUM, JamISum, Buf, ISum, ICompose, Data, COND, NoDiagPullin, NoDiagPullinLeft, NoDiagPullinRight, NeedInterleavedComplex]) ],
  e -> [ CopyFields(@(2).val, rec(
             _children :=  List(@(2).val._children, c -> @(1).val * c),
             dimensions := [Rows(@(1).val), Cols(@(2).val)] )) ]),

 PullInLeft := ARule( Compose,
       [ @(1, [RecursStep, Grp, BB, SUM, SUMAcc, JamISum, Buf, ISum, ICompose, ISumAcc, Data, COND, NoDiagPullin, NoDiagPullinLeft, NoDiagPullinRight, NeedInterleavedComplex]),
         @(2, [Prm, Gath, TCast, PushL, PushLR, Conj, ConjL, ConjR, ConjLR, FormatPrm]) ],
     e -> [ CopyFields(@(1).val, rec(
                _children := List(@(1).val._children, c -> c * @(2).val),
                dimensions := [Rows(@(1).val), Cols(@(2).val)] )) ]),
));

RewriteRules(_MissEstSanitize, rec(

    # sometimes there is a BB inside a BB. this is not good for me.
    # this removes all interior BBs
    RemoveInnerBB := Rule(@@(1,BB,(e,cx) -> ForAny(cx.parents, e -> ObjId(e) = BB)), e -> @@(1).val.rChildren()[1]),

    # BB cannot have an inplace inside of it, since all operations inside
    # the BB do not spill to memory.
    RemoveInnerInplaceFromBB := Rule(@@(1,Inplace,(e,cx) -> ForAny(cx.parents, e -> ObjId(e) = BB)), e -> @@(1).val.rChildren()[1]),

    # same thing with Inplace wrappers. leave only the outer.
    RemoveInnerInplace := Rule(@@(1,Inplace,(e,cx) -> ForAny(cx.parents, e -> ObjId(e) = Inplace)), e -> @@(1).val.rChildren()[1])
));

# we tag the BB with an inplace tag, if that basic block is to be performed
# inplace.

_tagBBin := function(bb, v)
    bb.inplin := v;
    return bb;
end;

_tagBBout := function(bb, v)
    bb.inplout := v;
    return bb;
end;

_tagInplace := function(k, v)
    k.inplace := v;
    return k;
end;

#
# this thing works very similarly to the keepgath/keepscat below. we start with
# the inplace tag, generate some wrappers from it, propagate the wrappers
# into the sigm-spl tree looking for a BB. once the wrappers hit the BB, they
# tag it. 

RewriteRules(_MissEstInplace, rec(
    InplaceExpand := Rule(Inplace,  e ->
        let(i := Ind(2),
            _AltInplace(Compose(Concat(
                [_InplaceWrap(e.rng(), e.rng(), i)], 
                e.rChildren(), 
                [_InplaceWrap(e.dmn(), e.dmn(), i)]
            )))
        )
    ),

    WrapCompose := ARule(Compose, [@(1, _InplaceWrap), @(2, Compose)], e -> [
        Compose(
            Concat([@(1).val], @(2).val.rChildren())
        )
    ]),

    ComposeWrap := ARule(Compose, [@(1, Compose), @(2, _InplaceWrap)], e -> [
        Compose(
            Concat(@(2).val.rChildren(), [@(1).val])
        )
    ]),

    WrapISum := ARule(Compose, [@(2, _InplaceWrap), @(1, [ISum, JamISum])], e -> [
        CopyFields(@(1).val, rec(
            _children := [
                Compose(
                    @(2).val, # _GSWrap(@(1).val.rng(), @(1).val.rng()),
                    @(1).val.child(1)
                )
            ],
            dimensions := [Rows(@(1).val), Cols(@(2).val)]
        ))
    ]),

    ISumWrap := ARule(Compose, [@(1, [ISum, JamISum]), @(2, _InplaceWrap)], e -> [
        CopyFields(@(1).val, rec(
            _children := [
                Compose(
                    @(1).val.child(1),
                    @(2).val # _GSWrap(@(1).val.rng(), @(1).val.dmn()) # have to adjust the size as we move it in.
                )
            ],
            dimensions := [Rows(@(1).val), Cols(@(2).val)]
        ))
    ]),

    InplaceWrapBB := ARule(Compose, [@(1, _InplaceWrap), @(2,BB)], e -> [
        _tagBBout(@(2).val, @(1).val._payload)
    ]),

    BBInplaceWrap := ARule(Compose, [@(1, BB), @(2, _InplaceWrap)], e -> [
        _tagBBin(@(1).val, @(2).val._payload)
    ]),

    ComposeAssoc := ARule( Compose, [ @(1,Compose) ],  e -> @(1).val.children() )

));

#
# we push into the basic block to figure out the gath/scat which access memory.
#
RewriteRules(_MissEstRules, rec(

    # insert a first gather and last scatter. replace BB with
    # a fake for a while.
    BBExpand := Rule(BB,  e ->
        _AltBB(Compose(Concat(
            [_GSWrap(e.rng(), e.rng(), When(IsBound(e.inplout), e.inplout, false))], 
            e.rChildren(), 
            [_GSWrap(e.dmn(), e.dmn(), When(IsBound(e.inplin), e.inplin, false))])
        ))
    ),
    WrapCompose := ARule(Compose, [@(1, _GSWrap), @(2, Compose)], e -> [
        Compose(
            Concat([@(1).val], @(2).val.rChildren())
        )
    ]),

    ComposeWrap := ARule(Compose, [@(1, Compose), @(2, _GSWrap)], e -> [
        Compose(
            Concat(@(2).val.rChildren(), [@(1).val])
        )
    ]),

    WrapISum := ARule(Compose, [@(2, _GSWrap), @(1, [JamISum, ISum])], e -> [
        CopyFields(@(1).val, rec(
            _children := [
                Compose(
                    @(2).val, # _GSWrap(@(1).val.rng(), @(1).val.rng()),
                    @(1).val.child(1)
                )
            ],
            dimensions := [Rows(@(1).val), Cols(@(2).val)]
        ))
    ]),

    ISumWrap := ARule(Compose, [@(1, [JamISum, ISum]), @(2, _GSWrap)], e -> [
        CopyFields(@(1).val, rec(
            _children := [
                Compose(
                    @(1).val.child(1),
                    @(2).val # _GSWrap(@(1).val.rng(), @(1).val.dmn()) # have to adjust the size as we move it in.
                )
            ],
            dimensions := [Rows(@(1).val), Cols(@(2).val)]
        ))
    ]),

    GSWrapScat := ARule(Compose, [@(1, _GSWrap), @(2,Scat)], e -> [
        _tagInplace(_KeepScat(@(2).val.func), @(1).val._payload)
    ]),

    GathGSWrap := ARule(Compose, [@(1, Gath), @(2, _GSWrap)], e -> [
        _tagInplace(_KeepGath(@(1).val.func), @(2).val._payload)
    ]),

    ComposeAssoc := ARule( Compose, [ @(1,Compose) ],  e -> @(1).val.children() )
));

RewriteRules(_MissEstCleanup, rec(
    ISumTagVar := Rule(@@(1,[JamISum, ISum], _checkTag), _setTag),

    RemoveBlk := ARule(Compose, [Blk], e -> []),
    RemoveRCDiag := ARule(Compose, [RCDiag], e -> []),

#    RemoveScatGath := ARule(Compose, [Scat, Gath], e -> []),
#    RemoveGath := ARule(Compose, [Gath], e -> []),
#    RemoveScat := ARule(Compose, [Scat], e -> []),
    # collapse ISums with just one G or S inside. 
#    ISumGath := Rule([ISum, @(1, [Gath, Scat])], e -> @(1).val),

    ComposeAssoc := ARule( Compose, [ @(1,Compose) ],  e -> @(1).val.children() ),

    AltBBtoBB := Rule(@(1, _AltBB), e -> BB(@(1).val.child(1))),

    AltInplacetoInplace := Rule(@(1,_AltInplace), e -> Inplace(@(1).val.child(1))),

#    ComposeAltInplacetoInplace := ARule(Compose, [@(1, _AltInplace)], e -> 
#        [Inplace(@(1).val.child(1))]
#    ),

#    buggy rule. huh?
#    DropInplaceWrap := ARule(Compose, [@(1, _InplaceWrap)], e -> [])
));


# plot 1
# naive iterative inplace WHT(1M)

_returnAndReset := function(a)
    local t;

    t := a.switch;

    a.switch := false;

    return t;
end;

_set := function(a, v)
    a.switch := v;
end;

Class(cachesim, rec(
    __call__ := (self, e, s, a) >> WithBases(self, rec(
        e := e,
        s := s,
        a := a,

        tagstore := List(Range1(s), e -> 
            List(Range1(a), ee -> 
                rec(tag:=-1, time:=0)
            )
        ),
        count := 0,
		hash := false
    )),

    reset := meth(self)
        self.tagstore := List(Range1(self.s), e -> 
            List(Range1(self.a), ee -> 
                rec(tag:=-1, time:=0)
            )
        );
        self.count := 0;
    end,

    access := meth(self, addr)
        local setidx, tag, tagidx, new, bits, mask, shift;

		if self.hash then
			bits := Log2Int(self.a * self.s);
			mask := self.e * ((self.a * self.s) - 1);
			shift := bits;

			# we clip it (somewhat arbitrarily) at 30 bits.
			while (shift < 30) do
				addr := BinXor(
					addr, 
					BinAnd(
						mask,
						Int(addr / shift)
					)
				);
				shift := shift + bits;
			od;
		fi;

		# add 1 for array offset
       	setidx := 1 + (Int(addr / self.e) mod self.s); 

        tag := Int(addr / (self.e * self.s));

        tagidx := Filtered(Range1(self.a), e -> 
            tag = self.tagstore[setidx][e].tag
        );

        # paranoia
        Constraint(Length(tagidx) <= 1);

        if Length(tagidx) = 0 then
            tagidx := self.a;
            self.tagstore[setidx][tagidx].tag := tag;
            new := 1;
        else
            tagidx := tagidx[1];
            new := 0;
        fi;

        self.count := self.count + 1;
        self.tagstore[setidx][tagidx].time := self.count;

        # sort based on access time, promotion to MRU happens here.
        Sort(self.tagstore[setidx], (a,b) -> a.time > b.time);

		# AppendTo("accesses", addr, " (", setidx, ", ", tagidx, "): ", new, "\n");
        return new;
    end,

    evaljam := meth(self, l, spl, res, jams)
        local l2, last, i, memaddr;
    
        if jams = [] then
            # get the actual address.
            memaddr := spl.array.offset + l.eval().v;
    
			# AppendTo("accesses", When(ObjId(spl) = _KeepGath, "G ", "S "));
            # perform the memory access!
            res.activ := res.activ + self.access(memaddr);
            res.access := res.access + 1;
        else
    
            last := Last(jams);
    
            for i in Range0(last.range) do
                l2 := Copy(l);
    
                SubstVars(l2, tab((last.id) := V(i)));
    
                self.evaljam(l2, spl, res, DropLast(jams, 1));
    
            od;
        fi;
    end,

	evalinner := meth(self, l, spl, res, inners, jams)
		local i, l2, last;

		if inners = [] then
        	for i in Range0(l.vars[1].range) do

            	l2 := Copy(l.at(i));

            	# now we have to deal with multiple jammed loops
            	# that means we have a nice little recursion here.
            	self.evaljam(l2, spl, res, jams);
        	od;
		else
			last := Last(inners);

			for i in Range0(last.range) do 
				l2 := Copy(l);

				SubstVars(l2, tab((last.id) := V(i)));

				self.evalinner(l2, spl, res, DropLast(inners, 1), jams);
			od;
		fi;
	end,

    evalspl := meth(self, spl, data)
        local res, id, i, j, tres, l, l2;
        res := rec(access := 0, activ := 0);

        id := ObjId(spl);

        if id = ISum then

			# we evaluate ISums inside of BB differently
			if data.currsize <= data.K then
				# AppendTo("traverse", "inner ", spl.var.id, " ", spl.var.range,", ", data.currsize, "\n");
				Add(data.innervar, spl.var);
				data.currsize := data.currsize / spl.var.range;
            	tres := self.evalspl(spl.child(1), data);
				data.currsize := data.currsize * spl.var.range;
            	data.innervar := DropLast(data.innervar, 1);

            	res.access := res.access + tres.access;
            	res.activ := res.activ + tres.activ;

			else
				# AppendTo("traverse", "outer", spl.var.id, " ", spl.var.range,", ", data.currsize, "\n");
				data.currsize := data.currsize / spl.var.range;
            	for i in Range0(spl.var.range) do
                	Add(data.var, spl.var.id);
                	Add(data.val, V(i));
                	tres := self.evalspl(spl.child(1), data);
                	data.var := DropLast(data.var, 1);
                	data.val := DropLast(data.val, 1);
	
                	res.access := res.access + tres.access;
                	res.activ := res.activ + tres.activ;
            	od;
				data.currsize := data.currsize * spl.var.range;
			fi;


		elif id = BB then
			data.inBB := true;
            tres := self.evalspl(spl.child(1), data);
			data.inBB := false;

            res.access := res.access + tres.access;
            res.activ := res.activ + tres.activ;

        elif id = JamISum then
            Add(data.jamvar, spl.var);
            tres := self.evalspl(spl.child(1), data);
            data.jamvar := DropLast(data.jamvar, 1);

            res.access := res.access + tres.access;
            res.activ := res.activ + tres.activ;

        elif id = Compose then
            for i in Reversed(spl.children()) do
                tres := self.evalspl(i, data);
                res.access := res.access + tres.access;
                res.activ := res.activ + tres.activ;
            od;
                
        elif id = _KeepGath or id = _KeepScat then

            l := spl.func.lambda();

            # subst the normal loops
            DoForAll(Range1(data.var), e -> 
                SubstVars(l, tab((data.var[e]) := data.val[e]))
            );

			# handle the innermost outer loop, nice little
			# recursion here.
			self.evalinner(l, spl, res, data.innervar, data.jamvar);

        else
            tres := List(spl.children(), e -> self.evalspl(e, data));
            res.access := Sum(List(tres, e -> e.access));
            res.activ := Sum(List(tres, e -> e.activ));
        fi;

        return res;
    end,

	applyRules := (self, s, o) >>
        ApplyStrategy(s, [
            _MissEstPreprocessJams,
            RulesFuncSimp,
            _MissEstSanitize, 
            _MissEstInplace, 
            _MissEstRules, 
            _MissEstCleanup
        ], UntilDone, o),

    eval := meth(self, s, _K)
        local gath, scat, gf, sf, arrays, i;

        gath := Reversed(Collect(s, _KeepGath));
        scat := Reversed(Collect(s, _KeepScat));

        arrays := [TArray(TUnknown, gath[1].func.range())];

		# PrintTo("accesses", "\n");
		# PrintTo("traverse", "\n");

        # assign input/output arrays
        for i in Range1(gath) do

            # paranoia
            # Constraint(gath[i].func.domain() = scat[i].func.domain());

            gath[i].array := Last(arrays);

			# we can match any of the previous gather matrices.
			if scat[i].inplace <> false 
				and ForAny(gath{[1..i]}, e -> e.inplace = scat[i].inplace) then

				scat[i].array := 
					Filtered(gath{[1..i]}, e -> e.inplace = scat[i].inplace)[1].array;
			else

                Add(arrays, TArray(TUnknown, scat[i].func.range()));
                scat[i].array := Last(arrays);
            fi;
        od;
         
        # descending order by size of array
        Sort(arrays, (a,b) -> a.size > b.size);

        # mark offsets
        arrays[1].offset := 0;

        # propagate offsets
        for i in [2..Length(arrays)] do
            arrays[i].offset := arrays[i-1].offset + arrays[i-1].size;
        od;

        # traverse the spl expression.
        return self.evalspl(s, rec(
			m := s.dims()[1], 
			K := _K,

			currsize := s.dims()[1], 
			var:=[], 
			val:=[], 
			innervar := [], 
			jamvar:=[]
		));
    end,
));

