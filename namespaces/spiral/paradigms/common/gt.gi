
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(GT);
Declare(ListProduct);

ListProduct := function(l)
    local prod, i;
    if Length(l) <=0 then return 0; fi;
    prod := 1;
    for i in l do
      prod := prod * i;
    od;
    return(prod);
end;

GTVec := XChain([0,1]);
GTPar := XChain([1,0]);

TTensorI_GT := function(gt)
    local spl, g, s, v, tags;
    [spl,g,s,v] := gt.params;
    tags := gt.getTags();
    Constraint(IsList(v) and Length(v)=1 and IsPosInt(v[1]));
    g := When(g=XChain([0,1]), AVec, APar);
    s := When(s=XChain([0,1]), AVec, APar);
    return TTensorI(spl, v[1], s, g).withTags(tags);
end;

GT_TTensorI := function(tt)
    local spl, g, s, v, tags;
    [spl,v,s,g] := tt.params;
    tags := tt.getTags();
    g := When(g=AVec, GTVec, GTPar);
    s := When(s=AVec, GTVec, GTPar);
    return GT(spl, g, s, [v]).withTags(tags);
end;

IsGT := x -> IsRec(x) and IsBound(x.isGT) and x.isGT;

Class(GTBase, rec(
    isGT := true,

    freeTags := self >> Filtered(self.tags, x -> not x.isSticky),
    stickyTags := self >> Filtered(self.tags, x -> x.isSticky),

    stickyRank := self >> Length(self.stickyTags()),
    freeRank := self >> self.rank() - self.stickyRank(),
    freeLoops := self >> [self.stickyRank()+1 .. self.rank()],

    transposedTags := self >> List(self.tags, x->x.transpose()),

    getSpl := self >> self.params[1],
    setSpl := (self, spl) >> ApplyFunc(ObjId(self), Concatenation([spl], Drop(self.params, 1)))
        .withTags(self.tags),

    # need setIts(its) and getIts()
    _fmanip := meth(self, func_manip, gt_manip, newits) 
         local cpy, z;
	 cpy := Copy(self);
 	 z :=  SubstTopDownNR_named(cpy, @.cond(x -> (not Same(x, cpy) and IsGT(x)) or IsFunction(x) or IsFuncExp(x) or ObjId(x)=ind), 
	                e -> When(IsGT(e), gt_manip(e), func_manip(e)), "__GT._fmanip");
         return z.setIts(newits);
    end,

    bodyRank := self >> Maximum( [0] ::
	List(Collect(self.getSpl(), @.cond(e->IsFunction(e) or IsFuncExp(e))), _rank)),

    downRankFull := (self, inds) >> let(thisrank := self.rank(), 
	self._fmanip(f -> f.downRankFull(inds), 
	             e -> Cond(e.bodyRank()=0, e, Error(".downRankFull does not work with nested GTs")),
		     [])),


    downRank := (self, loopid, ind) >>
        self._fmanip(f -> f.downRank(loopid, ind), 
	             e -> e.downRankOuter(loopid + e.rank(), ind), 
	             ListWithout(self.getIts(), loopid)),
    # downRankOuter: notification that downRank performed on some outer GT, 
    #   so that this GT can update its functions and notify children.
    downRankOuter := (self, loopid, ind) >>
        self._fmanip(f -> f.downRank(loopid + self.rank(), ind), 
	             e -> e.downRankOuter(loopid + e.rank(), ind), 
	             self.getIts()),

    rotate := (self, n) >> When(n=1, self, let(
	its := self.getIts(), 
	tags := self.getTags(),	stags := self.stickyTags(), ftags := self.freeTags(),
        res := self._fmanip(
	    f -> f.rotate(n), 
	    e -> Cond(e.bodyRank() <= e.rank(), e, 
		      Error(".rotate() doesn't support nested GTs where inner GTs depend on outer indices")),
	    [its[n]] :: its{[1..n-1]} :: its{[n+1..Length(its)]}),
	Cond(self.stickyRank() = 0, 
	        res,
	     n <= self.stickyRank(),
	        res.setTags( ftags :: [stags[n]] :: stags{[1..n-1]} :: stags{[n+1..Length(stags)]}),
	     # else, n > self.stickyRank(),
 	        Error("Can't make non-sticky rank <n> an inner loop, because sticky tags exist")
                # it could look like this instead: res.setTags( [ANoTag] :: tags )
	))),

    rotateIntoSticky := (self, n, newtag) >> let(
	its := self.getIts(), 
	tags := self.getTags(),	stags := self.stickyTags(), ftags := self.freeTags(),
        res := self._fmanip(
	    f -> f.rotate(n),
	    e -> Cond(e.bodyRank() <= e.rank(), e, 
		      Error(".rotate() doesn't support nested GTs where inner GTs depend on outer indices")),
	    [its[n]] :: its{[1..n-1]} :: its{[n+1..Length(its)]}),
	Cond(self.stickyRank() = 0, 
	        res.setTags(tags :: [newtag]),
	     n <= self.stickyRank(),
	        res.setTags( ftags :: [newtag] :: stags{[1..n-1]} :: stags{[n+1..Length(stags)]}),
	     # else, n > self.stickyRank(),
                res.setTags( ftags :: [newtag] :: stags )
	)),

    # NOTE: upRank() leaves GT in inconsistent state, because sticky tags still need to be shifted
    upRank := self >> self._fmanip(
	f -> f.upRank(), 
	e -> Cond(e.bodyRank() <= e.rank(), e, 
	          Error(".upRank() doesn't support nested GTs where inner GTs depend on outer indices")),
	Concatenation([0], self.getIts())),

    # NOTE: upRank() leaves GT in inconsistent state, because sticky tags still need to be shifted
    upRankBy := (self, n) >> self._fmanip(
	f -> f.upRankBy(n), 
	e -> Cond(e.bodyRank() <= e.rank(), e, 
	          Error(".upRankBy() doesn't support nested GTs where inner GTs depend on outer indices")),
	Concatenation(Replicate(n, 0), self.getIts())),

    split := (self, loopid, inner_its, outer_its) >> let(its := self.getIts(),
        self._fmanip(
	    f -> f.split(loopid, inner_its, outer_its),
	    e -> e.splitOuter(loopid + e.rank(), inner_its, outer_its), 
            Concatenation(its{[1..loopid-1]}, [inner_its, outer_its], its{[loopid+1..Length(its)]}))),
    # splitOuter: notification that some outer GT is splitting loop, 
    #   so that this GT can update its functions and notify children.
    splitOuter := (self, loopid, inner_its, outer_its) >> let(its := self.getIts(),
        self._fmanip(
	    f -> f.split(loopid + self.rank(), inner_its, outer_its),
	    e -> e.splitOuter(loopid + e.rank(), inner_its, outer_its), 
            its)),

    getGath := self >> Error("Must be implemented in a subclass"),

    getScat := self >> Error("Must be implemented in a subclass"),

    isReal := self >> self.params[1].isReal(),
    toAMat := self >> self.toSpl().toAMat(),
    # when self.getIts() returns empty list it actually means single iteration,
    # that's why [1] appended to the list
    normalizedArithCost := self >> self.getSpl().normalizedArithCost() * ListProduct(self.getIts() :: [1]),

    children := self >> [self.params[1]],
    child    := (self, n) >> Checked( n = 1, self.params[1] ),

    # overrideing withTags() to make sure non-sticky/sticky tags order is preserved
    # returns a copy of the object with 'tags' added on to any existing tags.
    withTags := (self, tags) >> self.setTags(Flat(TransposedMat(
        List([self.tags, tags], t -> SplitBy(t, e -> not e.isSticky))))),
));

# GT(<spl>, <gath>, <scat>, <v>) - generalized problem spec
#
# NOTE: assumes tightly packed data, i.e. Dims(GT(spl)) = Dims(spl) * Product(v)
#        (if gath/scat functions have domain&range == 0)
Class(GT, GTBase, Tagged_tSPL, rec(
    abbrevs := [
    (spl, gath, scat, v) -> Checked(
        IsSPL(spl), IsIndexMapping(gath), IsIndexMapping(scat), IsList(v),
        ForAll(v, IsPosInt0Sym),
        [spl, gath, scat, v]
    ) ],

    dims := self >> let(p := self.params, r := range(p[3]), c := range(p[2]), d := p[1].dims(),
        When(r<>0 and c<>0,
             [r, c],
         let(prod := Product(p[4]), [d[1]*prod, d[2]*prod]))), # NOTE: assumes XChain.

    advdims := self >> let(p := self.params, [p[3].advrange(), p[2].advrange()]),

    getIts := self >> self.params[4],
    setIts := (self,its) >> ObjId(self)(self.params[1], self.params[2], self.params[3], its).withTags(self.tags),
    getGath := self >> self.params[2],
    getScat := self >> self.params[3],

    rank := self >> Length(self.params[4]),

    _scat := Scat,
    _gath := Gath,

    # NOTE: toSpl doesn't work with nested GTs. It should use downRank method for function manipulations
    toSpl := self >> self.toSplCx([]),
    toSplCx := (self, outer_inds) >> let(
        p := self.params, spl := Copy(p[1]), g := Copy(p[2]), s := Copy(p[3]), dims := p[4],
        inds := List(dims, Ind), allinds := Concatenation(inds, outer_inds),
        kernel := self._scat(s.toSpl(allinds, Rows(spl))) * spl * self._gath(g.toSpl(allinds, Cols(spl))),
        kerneld := Cond(inds=[], kernel, # downRank only if not rank-0
            SubstTopDownNR(kernel, @.cond(e->IsFunction(e) or IsFuncExp(e)), e->e.downRankFull(allinds))),
        FoldL(inds, (ker, idx) -> ISum(idx.setAttr("GT"), ker), kerneld)
    ),

    toISums := self >> let(
    inds := List(self.getIts(), Ind),
    kernel := self._scat(self.getScat()) * self.params[1] * self._gath(self.getGath()),
    FoldL(inds, (ker, idx) -> ISum(idx.setAttr("GT"), ker), kernel)
    ),

    transpose := self >> let(p:=self.params,
        GT(p[1].transpose(), p[3], p[2], p[4]).withTags(self.transposedTags())),

    conjTranspose := self >> let(p:=self.params,
        GT(p[1].conjTranspose(), p[3], p[2], p[4]).withTags(self.transposedTags())),

    hashAs := self >> let(p:=self.params,
        ObjId(self)(HashAsSPL(p[1]), p[2], p[3], p[4]).withTags(self.tags)),
));

Declare(GTAccT, GTAcc0T);

# GTAcc(<spl>, <gath>, <scat>, <v>) - generalized problem spec with accumulation on output side
#
Class(GTAcc, GT, rec(
    transpose := self >> let(p:=self.params,
        GTAccT(p[1].transpose(), p[3], p[2], p[4]).withTags(self.transposedTags())),
    _scat := ScatAcc
));

Class(GTAcc0, GT, rec(
    transpose := self >> let(p:=self.params,
        GTAcc0T(p[1].transpose(), p[3], p[2], p[4]).withTags(self.transposedTags())),
    _scat := ScatAcc,

    toSplCx := (self, outer_inds) >> Checked(self.rank()=1, let( # NOTE: works only for rank-1
        dims := self.getIts(),
        inds0 := Concatenation(List(DropLast(dims,1), Ind), [0]),
        inds  := Concatenation(List(DropLast(dims,1), Ind), [Ind(Last(dims)-1)]),
        inds_shft := Concatenation(DropLast(inds,1), [Last(inds)+1]),

        dr0 := self.downRankFull(Concatenation(inds0, outer_inds)),
        dr  := self.downRankFull(Concatenation(inds_shft, outer_inds)),
        kernel0 := Scat(dr0.getScat())   * dr0.params[1] * self._gath(dr0.getGath()),
        kernel  := ScatAcc(dr.getScat()) * dr .params[1] * self._gath(dr .getGath()),
        When(Last(dims)=1,
            kernel0,
            SUM(kernel0, FoldL(inds, (ker, idx) -> ISum(idx.setAttr("GT"), ker), kernel))))
    ),
));

# GTAccT == GTAcc.transpose(), even though semantically GTAccT == GT, we need a separate object, so that
# GTAccT.transpose() is back to GTAcc itself.
Class(GTAccT, GT, rec(
    transpose := self >> let(p:=self.params,
        GTAcc(p[1].transpose(), p[3], p[2], p[4]).withTags(self.transposedTags())),
));

Class(GTAcc0T, GT, rec(
    transpose := self >> let(p:=self.params,
        GTAcc0(p[1].transpose(), p[3], p[2], p[4]).withTags(self.transposedTags())),
));

Declare(GTInplace);

Class(GTInplace, GTBase, Tagged_tSPL, rec(
    abbrevs := [ (spl, gathscat, v) -> Checked(
        IsSPL(spl), IsIndexMapping(gathscat), IsList(v), ForAll(v, IsPosInt0Sym),
        [spl, gathscat, v]) ],

    dims := self >> let(p := self.params, rc := range(p[2]), d := p[1].dims(),
        When(rc<>0,
             [rc, rc],
             let(prod:=Product(p[3]), [d[1]*prod, d[2]*prod]))), # NOTE: assumes XChain.

    getIts := self >> self.params[3],
    setIts := (self,its) >> GTInplace(self.params[1], self.params[2], its).withTags(self.tags),

    getGath := self >> self.params[2],
    getScat := self >> self.params[2],

    rank := self >> Length(self.params[3]),
    toGT := self >> let(p:=self.params, GT(p[1], p[2], p[2], p[3]).withTags(self.tags)),
    toNonInplace := self >> self.toGT(),
    toSpl := self >> Inplace(self.toGT().toSpl()),
    isInplace := self >> true,

    conjTranspose := self >> let(p:=self.params,
        GTInplace(p[1].conjTranspose(), p[2], p[3]).withTags(self.transposedTags())),

    transpose := self >> let(p:=self.params,
        GTInplace(p[1].transpose(), p[2], p[3]).withTags(self.transposedTags())),

    hashAs := self >> let(p:=self.params,
        ObjId(self)(HashAsSPL(p[1]), p[2], p[3]).withTags(self.tags))
));


#
# NOTE: what if there is no SUM scatters or gathers
#

Class(GTPS, GTBase, Tagged_tSPL, rec(
    abbrevs := [
    (spl, fb_cnt, gath, scat, v) -> Checked(
        IsSPL(spl), IsPosInt(fb_cnt), IsIndexMapping(gath), IsIndexMapping(scat), IsList(v),
        ForAll(v, IsPosInt0Sym),
        [spl, fb_cnt, gath, scat, v]
    ) ],

    dims := self >> let( p := self.params, r := range(p[4]), c := range(p[3]), 
        d := p[1].dims(), 
        [ StripList(Flat([d[1]]){[1..p[2]]} :: Flat([r])), 
          StripList(Flat([d[2]]){[1..p[2]]} :: Flat([c])) ]),

    advdims := self >> let(p := self.params, r := p[4].advrange(), c := p[3].advrange(), 
        d := p[1].advdims(), 
        [ StripList(Flat([d[1]]){[1..p[2]]} :: Flat([r])), 
          StripList(Flat([d[2]]){[1..p[2]]} :: Flat([c])) ]),

    getIts := self >> self.params[5],
    setIts := (self,its) >> ObjId(self)(self.params[1], self.params[2], self.params[3], self.params[4], its).withTags(self.tags),
    getGath := self >> fCross(List(Flat([self.params[1].advdims()[1]]){[1..self.params[2]]}, e -> fId(e)) :: [self.params[3]]),
    getScat := self >> fCross(List(Flat([self.params[1].advdims()[2]]){[1..self.params[2]]}, e -> fId(e)) :: [self.params[4]]),

    rank := self >> Length(self.params[5]),

    toSpl := self >> let(
        inds   := List(self.params[5], Ind),
        kernel := FoldR([1..self.rank()], (ker, i) -> ker.downRank(i, inds[i]), Copy(self)),
        spl    := Scat(kernel.getScat()) * kernel.child(1) * Gath(kernel.getGath()),
        FoldL(inds, (ker, idx) -> IParSeq(idx.setAttr("GT"), kernel.params[2], ker), spl)),

    hashAs := self >> let(p:=self.params,
        ObjId(self)(HashAsSPL(p[1]), p[2], p[3], p[4], p[5]).withTags(self.tags)),
    
));

# Useful identities:
#
# Tr(m,n) = GT(I(m), XChain([0,1]), XChain([1,0]), [n]) =
#           GT(I(n), XChain([1,0]), XChain([0,1]), [m]) = L(m*n, n)
GT_Tr1 := (m,n) -> GT(I(m), XChain([0,1]), XChain([1,0]), [n]);
GT_Tr2 := (m,n) -> GT(I(n), XChain([1,0]), XChain([0,1]), [m]);

GT_Tr1u := (m,n,u) -> let(uu:=When(m mod u = 0, u, 1),
    GT(BB(I(uu)), XChain([1,0,2]), XChain([2,1,0]), [m/uu, n]));

GT_Tr2u := (m,n,u) -> let(uu:=When(n mod u = 0, u, 1),
    GT(BB(I(uu)), XChain([2,1,0]), XChain([1,0,2]), [n/uu, m]));


NewRulesFor(GT, rec(
    GT_Base := rec(
	switch := false,
	a := rec(maxSize := false),
        # requiredFirstTag := ANoTag,
	applicable := (self, t) >> let(rank := Length(t.params[4]), spl := t.params[1],
            rank = 0 and (self.a.maxSize=false
		or Rows(spl) <= self.a.maxSize or Cols(spl) <= self.a.maxSize)),
        freedoms := (self, t) >> [],
	child := (t, fr) -> [ t.params[1].withTags(t.getTags()) ],
	apply := (t, C, Nonterms) -> let(
            g := t.params[2], s := t.params[3],
            Scat(s.toSpl([], Rows(C[1]))) * C[1] * Gath(g.toSpl([], Cols(C[1])))
        )
    ),

    GT_NthLoop := rec(
	    requiredFirstTag := [ANoTag, ALimitNthLoop],
	    applicable := t -> let(rank := Length(t.params[4]), rank > 0), 

        # restrict to innermost first (i.e. loop interchange)
	    # to search over loop orders use [1..nloops]

        # Limit tag reduces the number of loop interchanges.
        # it is useful when you don't want the number of potential
        # ruletrees to explode, and you are not overtly concerned
        # with having access to all the potential ones.
        freedoms := t -> let(
            fr := [1..Length(t.params[4])], 
            When(t.hasTag(ALimitNthLoop),
                [[1]],
                [fr]
            )
        ),

	    child := (t, fr) -> let(
	        spl := t.params[1], 
            g := t.params[2], 
            s := t.params[3], 
            loopid := fr[1],
	        [ 
                GT(spl, g.without(loopid), 
                    s.without(loopid), ListWithout(t.params[4], loopid)), 
                InfoNt(loopid) 
            ]
        ),

	    apply := (t, C, Nonterms) -> let(
	        loopid := Nonterms[2].params[1], 
            dft := Nonterms[1].params[1], 
	        g := t.params[2], 
            s := t.params[3],
	        loop_dims := t.params[4],
	        i := Ind(loop_dims[loopid]),

	        ISum(i, Scat(s.part(loopid, i, Rows(dft), loop_dims)) 
            * C[1] 
            * Gath(g.part(loopid, i, Cols(dft), loop_dims)))
        )
    ),

    GT_BufReshape := rec(
    bufIters      := [2, 4, 8, 16, 32],
    requiredFirstTag := ANoTag,

    applicable := (self, t) >> Length(t.params[4])=1 and let(
        N := Minimum(t.params[1].dimensions), its := t.params[4][1],
        PatternMatch(t, [GT, @, @(2,XChain), @(3,XChain), ...], empty_cx()) and
        (@(2).val.params[1] = [0,1] or @(3).val.params[1] = [0,1]) and
        ForAny(self.bufIters, bi -> bi < its and IsInt(its/bi))),

    u := [2,4,8,16],

    children := (self, t) >> let(
        spl := t.params[1], its := t.params[4][1],
        bufiters := Filtered(self.bufIters, bi -> (bi < its) and IsInt(its/bi)),
        Map2(Cartesian(bufiters, self.u), (bi, u) ->
        [ GT(spl, XChain([1,0]), XChain([1,0]), [bi]),
          InfoNt(u) ])),

    apply := (self, t, C, Nonterms) >> let(
        g := t.params[2].params[1], s := t.params[3].params[1],
        gg := When(g=[0,1], XChain([0,1,2]), XChain([1,2,0])),
        ss := When(s=[0,1], XChain([0,1,2]), XChain([1,2,0])),
        spl := t.params[1], its := t.params[4][1], inner := Nonterms[1].params[4][1],
        i := Ind(its / inner),
        u := When(IsBound(Nonterms[2]), Nonterms[2].params[1], 4),

        ISum(i, Scat(ss.part(1, i, Rows(spl), [i.range, inner])) *
            When(s=[0,1], GT_Tr2u(inner, Rows(spl), u).toSpl(), I(Rows(spl)*inner)) *
            C[1] *
            When(g=[0,1], GT_Tr1u(Cols(spl), inner, u).toSpl(), I(Cols(spl)*inner)) *
            Gath(gg.part(1, i, Cols(spl), [i.range, inner]))
        )
    )
    )
));
