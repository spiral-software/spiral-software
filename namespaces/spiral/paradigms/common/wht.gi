
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

# Cross-compatibility
#

# MAJOR FORWARD DECLARATION HACK
#
# the current module is loaded prior to paradigms.cache, but we rely on the ACache tag
# declared there. To get around this we create a function, which is evaluated on 
# execution, to map ACache to a shorter name. This is necessary so that ObjId comparisons
# work at the tag level.
#
# NOTE: this cannot be done with
# 
# _ACache := spiral.paradigms.cache.ACach
#
# because assignments are evaluated immediately, and paradigms.cache does not yet exist.
#
# NOTE: we do the same with the CacheSpec and AInplace

_ACache := () -> spiral.paradigms.cache.ACache;

_CacheDesc := () -> spiral.paradigms.cache.CacheDesc;

_AInplace := () -> spiral.paradigms.cache.AInplace;

_AExpRight := () -> spiral.paradigms.cache.AExpRight;

_AIO := () -> spiral.paradigms.common.AIO;

_ABuf := () -> spiral.paradigms.loops.ABuf;
#
## _divisorCacheTags
#
# takes the ACache(N, ...) with the largest N from 't' and
# tests whether the divisor pair 'd' would work well with the cache
#
# We assume a (W x I) (I x W) expansion, so the left side is
# strided. We also assume out-of-place expansion.
#
# At some point, support for an 'Inplace' tag will be added.
#
# the criteria for working well with cache are as follows:
#
# let m = d[1], n = d[2]
#
# LEFT side: kernel size 'm', stride 'n'
# RIGHT side: kernel size 'n', stride 1
#
# cache specs are [B,S,A,R]
# T is the datatype, eg. float, double
# 
# let E = B / sizeof(T)
#
# criteria:
# 
# --> see the comments in the code
#
# NOTE: additions to support Inplace tag added, but changes incomplete.

_divisorCacheTags := function(d, t)
    local taglist, N, tag, cs, m, n, elems, csize, cond, inplace;
    
    taglist := t.getTag(_ACache());

    # tagged for inplace?
    inplace := t.getTag(_AInplace());

    # if we don't have any ACache tags, accept this divisor pair.
    if taglist = false then
        return true;
    fi;

    # we need taglist to be a list
    taglist := When(IsList(taglist), taglist, [taglist]);

    # find the max cache level amongst the tags.
    N := Maximum(List(taglist, e -> e.params[1]));

    # use the first tag which matches it.
    tag := Filtered(taglist, e -> e.params[1] = N)[1];

    # extract the cache spec from the tag
    cs := tag.params[2];

    # paranoia
    Constraint(ObjId(cs) = _CacheDesc());

    # d[1] LEFT side expansion
    # d[2] RIGHT side expansion
    m := d[1]; n := d[2];

    # cache sizes
    elems := cs.blksize;
    csize := elems * cs.nsets * cs.assoc;

    # associativity must be at least 2 for outofplace, otherwise read and 
    # write overwrite each other
    if inplace = false and cs.assoc < 2 then
        return false;
    fi;

    # if any of these conditions are true -- we allow this split
    cond := [];

    # if the whole thing fits in cache, let it through
    Add(cond, n * m < csize);

    # if right side is larger than cache, left side only uses the assoc so
    # size must be smaller than it.. (also accounting for read/write)
    Add(cond, (n > elems*cs.nsets) and (m <= (cs.assoc / 2)));

    # if right side fits in cache, left side strides partially into the cache, 
    # and the size constraint on 'm' is a function of 'n'
    Add(cond, (n <= csize / 2) and (m <= (csize / (2 * n))));

    #PrintLine(d, ": ", cond);

    return ForAny(cond, e -> e);
end;

#
## _dropCacheTags
#
# drop the cache tag if the problem size fits entirely into cache.
#
_dropCacheTags := (t, size, inplace) -> 
    
    (ObjId(t) <> _ACache())           # let other tags through
    or When(inplace = true,  # inplace?
        ((t.params[2].csize) < size), # only let through ACache tags smaller than problem size 
        (t.params[2].csize/2 < size)  # (div 2 accounts for rd/wr)
    )
;

#
## _expandRight
#
_expandRight := function(d, t)
    local tag, leftmax;
    
    tag := t.getTag(_AExpRight());
    
    # if the tag isn't present, pass through
    if tag = false then
        return true;
    fi;

    # get the limit on left side size, handle multiple tags
    leftmax := When(IsList(tag),
        Minimum(List(tag, e -> e.params[1])),
        tag.params[1]
    );

    # if left expansion exceeds our max, don't allow this breakdown
    return d[1] <= 2^leftmax;
end;

#
## _allowedExpRight
#
#

_allowedExpRight := (nt) -> let(
    t := nt.getTag(_AExpRight()),
    tt := When(IsList(t), t, [t]),
    maxx := Minimum(List(tt, e -> e.params[1])),
    minn := Maximum(List(tt, e -> When(IsBound(e.params[2]) and nt.params[1] > e.params[2], e.params[2], 0))), # optional 2nd param is minimum
#    debugout := Chain(Print(nt.params[1], ": ", minn, "-", maxx, "\n"), 0),
    [minn..maxx]
);

#
## _dropExpandRight
#
# drop all expandright tags at smallest size.
#
# NOTE: this function returns TRUE if we should keep the tag, and FALSE if the
# tag is to be dropped.
#

_dropExpRightTags := (t, size) ->
    ObjId(t) <> _AExpRight() 
    or (
        size > 2^Minimum(t.params)
        and not IsBound(t.params[3]) # the presence of a third param says "drop this tag asap"
    )
;
 
#
#
## DivisorProds
#
# returns lists made up of DivisorsInt(N) which are \leq M and their product = N
#
Declare(_getdivisors);

DivisorProds := function(N, M)
    local l, r;

    # list of valid divisors.
    l := Filtered(DivisorsInt(N), e -> e <= M and e <> 1);

    r := _getdivisors(N, l, [1]);

    # the last element is a 1, we don't want it in the output.
    return List(r, e -> DropLast(e, 1));
end;

_getdivisors := function(N, list, prod)
    local result, e, p; 
    
    p := Product(prod);

    result := List(
        Filtered(list, e -> e * p = N),
        ee -> Concat([ee], prod)
    );

    for e in Filtered(list, i -> i * p < N) do
        Append(result, _getdivisors(N, list, Concat([e], prod)));
    od; 

    return result;
end;

_loop := function(p, i)
    local idx, ii, a, c, size;

    # paranoia, 'i' in range.
    Constraint(i >=1 and i <= Length(p));

    idx := [
        XChain([0]),
        GTPar,
        GTVec,
        XChain([1,0,2])
    ];

    # 
    a := Product(p{[1..(i-1)]});
    c := Product(p{[i+1..Length(p)]});

    ii := When(a > 1, 1, 0) + When(c > 1, 2, 0) + 1;

    size := [
        [],
        [a],
        [c],
        [a,c]
    ];

    return [idx[ii], size[ii]];
end;

#
## WHT RULES ###############
#

NewRulesFor(WHT, rec(

    #F WHT_BinSplit:
    #F
    #F same as WHT_GeneralSplit, but only allows for binary splits
    #F
    #F   WHT_(2^k) =
    #F     WHT_(2^k1)   tensor I_(2^(k-k1)) *
    #F     I_(2^(k-k2)) tensor WHT_(2^k2)
    #F
    #F We use this rule only, when WHT_GeneralSplit is turned off.
    #F
    #F NOTE: propagates tags, handles expand-right tag.
    WHT_BinSplit := rec (
        info             := "WHT_(2^k) -> (WHT_(2^k1) tensor I) (I tensor WHT_(2^k2))",
        forTransposition := false,
        minSize          := 2,
        maxRadix         := 32,
        applicable       := (self, nt) >> nt.params[1] >= self.minSize,
        requiredFirstTag := ANoTag,

        children := (self, nt) >> List([1.. Minimum(self.maxRadix, nt.params[1]-1)], 
            i -> [ WHT(i), WHT(nt.params[1]-i) ]
        ),

        apply := (nt, c, cnt) ->
            Tensor(c[1], I(Rows(c[2])))
            * Tensor(I(Rows(c[1])), c[2])
    ),

    # handle the right expanded WHT.
    WHT_ExpandRight := rec (
        info             := "WHT_(2^k) -> (WHT_(2^k1) tensor I) (I tensor WHT_(2^k2))",
        forTransposition := false,
        minSize          := 2,
        applicable       := (self, nt) >> nt.params[1] >= self.minSize and nt.hasTag(_AExpRight()),

        children := nt -> List(Intersection([1..nt.params[1]-1], _allowedExpRight(nt)), 
            i -> [ 
                WHT(i).withTags(
                    Filtered(nt.getTags(), e ->
                        _dropExpRightTags(e, 2^i)
                    )
                ), 
                WHT(nt.params[1]-i).withTags(
                    Filtered(nt.getTags(), e ->
                        _dropExpRightTags(e, 2^(nt.params[1]-i))
                    )
                )
            ]
        ),

        apply := (nt, c, cnt) ->
            Tensor(c[1], I(Rows(c[2])))
            * Tensor(I(Rows(c[1])), c[2])
    ),

    # top level bin split rule (requires ATopLevel tag) which inserts some permutations which are
    # linear on the bits into the ruletree.
    # this rule expects ATopLevel param1 to be a size at which to stop
    # propogating the ATopLevel tag and param2 to be a cache specification
    # of the form [E,S,A,R]
    WHT_BinSplitB := rec (
        info             := "WHT_(2^k) -> (WHT_(2^k1) tensor I) (I tensor WHT_(2^k2))",
        forTransposition := false,
        minSize          := 2,
        applicable       := (self, nt) >> nt.hasTag(ATopLevel) and Rows(nt) > self.minSize,

        children := nt -> let(t := nt.getTag(ATopLevel).params[1],
            List([1..(nt.params[1]-1)], i -> 
                When(i >= t or (nt.params[1]-i) >= t,
                    [ WHT(i).withTags([ATopLevel(t)]), WHT(nt.params[1]-i).withTags([ATopLevel(t)]) ],
                    [ WHT(i), WHT(nt.params[1]-i) ]
                )
            )
        ),

        apply := (nt, c, cnt) ->
                Tensor(c[1], I(Rows(c[2])))
                * PushL(
                    CL(Rows(nt), Rows(c[2]), nt.getTag(ATopLevel).params[2])) 
                * PushR(
                    CL(Rows(nt), Rows(c[2]), nt.getTag(ATopLevel).params[2]).transpose())
                * Tensor(I(Rows(c[1])), c[2]),

        switch := true
    ),

    WHT_BinSplitB_binloops := rec (
        info             := "WHT_(2^k) -> (WHT_(2^k1) tensor I) (I tensor WHT_(2^k2))",
        forTransposition := false,
        minSize          := 2,
        applicable       := (self, nt) >> nt.hasTag(ATopLevel) and Rows(nt) > self.minSize,

        children := nt -> let(t := nt.getTag(ATopLevel).params[1],
            List([1..(nt.params[1]-1)], i -> 
                When(i >= t or (nt.params[1]-i) >= t,
                    [ WHT(i).withTags([ATopLevel(t)]), WHT(nt.params[1]-i).withTags([ATopLevel(t)]) ],
                    [ WHT(i), WHT(nt.params[1]-i) ]
                )
            )
        ),

        apply := function(nt, c, cnt)
            local a, b;

            a := WHT_BinSplit_binloops.apply(nt, c, cnt);

            b := CL(Rows(nt), Rows(c[2]), nt.getTag(ATopLevel).params[2]);

            if b = I(Rows(nt)) then
                return a;
            else
                return Grp(a._children[1] * b) * Grp(b.transpose() * a._children[2]);
            fi;
        end,

        switch := true,
    ),
    #######################################################################################################
    #   tSPL WHT rule
    # tSPL WHT_(2^k) -> (WHT_(2^k1) tensor I) (I tensor WHT_(2^k2))
    WHT_tSPL_BinSplit := rec (
        forTransposition := false,
        minSize          := 2, 
        applicable       := (self, nt) >> 
            nt.hasTags() 
            and nt.params[1] >= self.minSize
            and not nt.hasTag(_AInplace()),
        children         := nt -> List([1..nt.params[1] - 1], i -> [ TTensor(WHT(i), WHT(nt.params[1]-i)).withTags(nt.getTags()) ]), 
        apply            := (nt, c, cnt) -> c[1],
        switch := true,
    ),

    #F WHT_Base: WHT_1 = F_2
    #F
    WHT_tSPL_Base := rec(
        switch := false,
        applicable := (t) -> Rows(t) = 2 and t.hasTags(),
        children := t -> [[ TTensorI(F(2), 1, AVec, AVec).withTags(t.getTags()) ]],
        apply := (t, C, Nonterms) -> C[1]
    ),


    #######################################################################################################
    #   tSPL Pease WHT rule
    # Pease tSPL WHT_(2^k) -> \Prod((I tensor F_2)L)
    WHT_tSPL_Pease   := rec (
        forTransposition := false,
        minSize          := 2, 
        applicable       := (self, nt) >> nt.hasTags()
                                          and nt.params[1] >= self.minSize
                                          and IsBound(nt.firstTag().legal_kernel)
                                          and ForAny(self.radices, i -> nt.firstTag().legal_kernel(2^i)),

        radices          := [1, 2, 3, 4, 5],

        children         := (self, nt) >> 
            let( 
                k := nt.params[1],
                streamsize := Cond(
                    nt.hasTags() and IsBound(nt.firstTag().bs),
                    nt.firstTag().bs, 
                    NULL
                ),
                ap := Filtered(self.radices, n -> IsInt(k/n) and streamsize >= 2^n),
                List(ap, i -> let(
                    r := i,
                    Cond(
                        IsInt(k/r) and nt.hasTags() and IsBound(nt.firstTag().bs)
                            and nt.firstTag().bs >= 2^r,
                        [ TICompose(var("j"), k/r, TTensorI(WHT(i), 2^(k-r), APar, AVec)).withTags(nt.getTags())],
                        []
                    )
                ))
            ),
            

        apply := (nt, c, cnt) -> c[1],
        switch := false
    ),
    
    #######################################################################################################
    #   tSPL Korn-Lambiotte WHT rule
    WHT_tSPL_KornLambiotte   := rec (
        forTransposition := false,
        minSize          := 2, 
        applicable       := (self, nt) >> nt.hasTags() 
                                and nt.params[1] >= self.minSize 
                                and ForAny(self.radices, i->IsInt(nt.params[1]/i) 
                                and IsBound(nt.firstTag().legal_kernel)
                                and nt.firstTag().legal_kernel(2^i)
                            ),
                                     
        radices          := [1, 2, 3, 4, 5],

        children         := (self, nt) >> let(
                                k := nt.params[1],
                                ap := Filtered(self.radices, n -> IsInt(k/n) and nt.firstTag().legal_kernel(2^n)),
                                List(ap, i ->
                                    [ TICompose(var("j"), k/i, 
                                        TTensorI(WHT(i), 2^(k-i), AVec, APar), 
                                        nt.getTags()
                                    )]
                                )
                            ),



        apply            := (nt, c, cnt) -> c[1],
        switch := false
    ),

    # WHT GT rule. Direct copy of DFT_GT_CT minus the twiddles
    #
    # WHT_nm -> GT(WHT_n, ...) * GT(WHT_m, ...)
    #
    # supports the ACache tag.
    #
    WHT_GT_CT := rec(
        switch := true,
        maxSize := false,
        minSize := false,
        forTransposition := false,

        applicable := (self, t) >> let(
            n := Rows(t),

            n > 2 
            and (self.maxSize=false or n <= self.maxSize) 
            and (self.minSize=false or n >= self.minSize) 
            and not IsPrime(n))
            and not t.hasTag(_AInplace()),

        children := (self, t) >> Map2(
            # _divisorCacheTags determines if the split is allowed (according to the cache params)
            Filtered( 
                DivisorPairs(Rows(t)), 
                d -> _divisorCacheTags(d, t) and _expandRight(d, t)
            ),
            (m,n) -> [
                GT(WHT(Log2Int(m)), XChain([0, 1]), XChain([0, 1]), [n]).withTags(
                    Filtered(t.getTags(), e -> 
                        _dropCacheTags(e, m, false) and _dropExpRightTags(e, m)
                    )
                ),
                GT(WHT(Log2Int(n)), XChain([1, 0]), XChain([1, 0]), [m]).withTags(
                    Filtered(t.getTags(), e ->
                        _dropCacheTags(e, n, false) and _dropExpRightTags(e, n)
                    )
                )
            ]
        ),

        apply := (self, t, C, Nonterms) >>  C[1] * C[2]
    ),

    #
    ## WHT_TopInplace
    #
    # the input and output strides must be the same, and intermediate
    # arrays are *full length*, which means the code generated is not
    # recursive.
    #
    # each 'stage' is <= kernel size.
    #
    WHT_TopInplace := rec(
        switch := true,
        maxSize := false,
        minSize := false,

        applicable := (self, t) >> let(
            n := Rows(t),

            t.hasTag(_AInplace())
            and n > 2 
            and (self.maxSize=false or n <= self.maxSize) 
            and (self.minSize=false or n >= self.minSize) 
            and not IsPrime(n)),

        children := meth(self, t)
            local iptag, divs, wa, w, p, i, loopdata, newtags;

            iptag := t.getTag(_AInplace());

            divs := DivisorProds(
                Rows(t), 
                When(iptag <> false, iptag.params[1], Rows(t))
            );

            wa := [];
            newtags := Filtered(t.getTags(), e -> ObjId(e) <> _AInplace());

            for p in divs do
                w := [];
                for i in [1..Length(p)] do
                    loopdata := _loop(p, i);
                    Add(w, Inplace(
                        GT(
                            WHT(Log2Int(p[i])), 
                            loopdata[1],
                            loopdata[1],
                            loopdata[2]
                        ).withTags(newtags)
                    ));
                od;
                Add(wa, w);
            od;

            return wa;
        end,

        apply := (self, t, C, Nonterms) >>  ApplyFunc(Compose, C)
    ),

    #
    ## WHT_GT_Inplace
    #
    # standard Binsplit CT rule except in place, tags ignored and not
    # propagated.

    WHT_GT_Inplace := rec(
        switch := true,
        maxSize := false,
        minSize := false,

        applicable := (self, t) >> let(
            n := Rows(t),

            t.hasTag(_AInplace()) <> false
            and n > 2 
            and not IsPrime(n)
        ),

        children := (self, t) >> Map2(
            DivisorPairs(Rows(t)), 
            (m,n) -> [
                GT(WHT(Log2Int(m)), XChain([0, 1]), XChain([0, 1]), [n]).withTags(t.getTags()),
                GT(WHT(Log2Int(n)), XChain([1, 0]), XChain([1, 0]), [m]).withTags(t.getTags())
            ]
        ),


        apply := (self, t, C, nt) >> Inplace(ApplyFunc(Compose, C))
    )
));

#
## GT-WHT RULES ########################
#
# rules for GT(WHT, ...)
# these rules are a copy of the rules for the DFT

NewRulesFor(GT, rec(
    # copy and simplification of GT_DFT_Base2
    # matches GT(WHT(n), ...) where n = 1
    GT_WHT_Base := rec(
        forTransposition := false,
        switch           := false,
        applicable := (self, t) >> let(
            rank := Length(t.params[4]),                                # rank is number of outer loops

            rank = 0                                                    # means no outer loops
            and PatternMatch(t, [GT, WHT, @, @, @, @, @], empty_cx())   # t.rChildren(), used by patternmatch appends t.transposed and t.tags, so GT, with its 4 params, actually has 6.
            and Rows(t.params[1])=2                                     # we're looking for a WHT of size 2
        ),

        apply := (t, C, Nonterms) -> F(2)
    ),

    # matches GT(WHT(n), ...) where n > 1
    GT_WHT_Inplace := rec(
        maxSize       := false,
        minSize       := false,
        minRank       := 0,
        maxRank       := 3,
        forTransposition := false,
        switch           := false,

        applicable := (self, t) >> let(
            rank := Length(t.params[4]), 
            wht := t.params[1],

            t.getTag(_AInplace()) <> false
            and rank >= self.minRank 
            and rank <= self.maxRank 
            and rank = 0 # MRT: we use NthLoop for rank > 0
            and (self.maxSize=false or Rows(wht) <= self.maxSize)
            and (self.minSize=false or Rows(wht) >= self.minSize) 
            and PatternMatch(t, [GT, WHT, XChain, XChain, @, @, @], empty_cx()) 
            and WHT_GT_CT.applicable(wht)
        ),

        children := (self, t) >> let(
            wht := t.params[1], 
            g := t.params[2], 
            s := t.params[3], 
            loop_dims := t.params[4],  
            nloops := Length(loop_dims), 
            tags := t.getTags(),

            Map2( 
                DivisorPairs(Rows(wht)),
                (m,n) -> [
                    GT(
                        WHT(Log2Int(m)),
                        s.composeWith(XChain([0, 1])), 
                        s.composeWith(XChain([0, 1])),
                        Concatenation([n], loop_dims)
                    ).withTags(Filtered(t.getTags(), e -> 
                        ObjId(e) <> _AInplace() or m > e.params[1]
                    )),
                    GT(
                        WHT(Log2Int(n)),
                        g.composeWith(XChain([1, 0])), 
                        s.composeWith(XChain([1, 0])),
                        Concatenation([m], loop_dims)
                    ).withTags(Filtered(t.getTags(), e -> 
                        ObjId(e) <> _AInplace() or n > e.params[1]
                    ))
                ]
            )
        ),

        apply := (self, t, C, Nonterms) >> Inplace(C[1] * C[2])
    ),

    GT_WHT_CT := rec(
        maxSize       := false,
        minSize       := false,
        minRank       := 0,
        maxRank       := 3,
        codeletSize   := 32,
        forTransposition := false,
        switch           := false,

        applicable := (self, t) >> let(
            rank := Length(t.params[4]), 
            wht := t.params[1],

#            not t.hasTag(_ACache())
#            and 
            not t.hasTag(_AInplace())
            and not t.hasTag(_ABuf())
            and rank >= self.minRank 
            and rank <= self.maxRank 
#            and When(rank>0, t.hasTags(), true) 
            and rank = 0 # MRT: we use NthLoop for rank > 0
            and (self.maxSize=false or Rows(wht) <= self.maxSize)
            and (self.minSize=false or Rows(wht) >= self.minSize) 
            and PatternMatch(t, [GT, WHT, XChain, XChain, @, @, @], empty_cx()) 
            and WHT_GT_CT.applicable(wht)
        ),

        children := (self, t) >> let(
            wht := t.params[1], 
            g := t.params[2], 
            s := t.params[3], 
            loop_dims := t.params[4],  
            nloops := Length(loop_dims), 
            tags := t.getTags(),
            inp := t.getTag(_AInplace()) <> false,

            Map2( 
                Filtered(
                    DivisorPairs(Rows(wht)), 
                    d -> ( d[1] <= self.codeletSize) and _divisorCacheTags(d, t) and _expandRight(d, t)
                ),
                (m,n) -> [
                    GT(
                        WHT(Log2Int(m)),
                        s.composeWith(XChain([0, 1])), 
                        s.composeWith(XChain([0, 1])),
                        Concatenation([n], loop_dims)
                    ).withTags(
                        Filtered(t.getTags(), e ->
                            _dropCacheTags(e, m, inp) and _dropExpRightTags(e, m)
                        )
                    ),

                    GT(
                        WHT(Log2Int(n)),
                        g.composeWith(XChain([1, 0])), 
                        s.composeWith(XChain([1, 0])),
                        Concatenation([m], loop_dims)
                    ).withTags(
                        Filtered(t.getTags(), e -> 
                            _dropCacheTags(e, n, inp) and _dropExpRightTags(e, n)
                        )
                    )
                ]
            )
        ),

        apply := (self, t, C, Nonterms) >> C[1] * C[2]
    ),

    GT_NthLoop_PassTags := rec(
        switch := false,

        applicable := t -> let(
            rank := Length(t.params[4]), 
            
            rank > 0 and t.hasTags()
        ), 

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
#            codeletSize := 32,
            spl := t.params[1], 
            g := t.params[2], 
            s := t.params[3], 
            loopid := fr[1],
            gt := GT(spl, g.without(loopid), 
                s.without(loopid), ListWithout(t.params[4], loopid)
            ),
            [ 
                gt.withTags(t.getTags()),
                # Filtered(t.getTags(), e ->
                #     ObjId(e) <> _AInplace() or Rows(gt) >= e.params[1]
                # )), 
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
));

#   (WHTmn x Ik) -> (WHTm x Ikn) (Im x (WHTn x Ik))
NewRulesFor(TTensorI, rec(
    WHTxI_vecrec := rec(
        switch           := false,
        forTransposition := false,
        minKernel := false,
        maxKernel := false,
        applicable := (self, nt) >> nt.hasTags() and IsVecVec(nt.params) and
            (not IsInt(self.minKernel) or nt.params[1].dims()[2] > self.minKernel) and
            (not IsInt(self.maxKernel) or nt.params[1].dims()[2] <= self.maxKernel),
        children := nt -> let(mn := nt.params[1].dims()[2], k := nt.params[2],
            List(Flat(List(DivisorPairs(mn), i -> let(m := i[1], n := i[2],
                [
                    TCompose([
                        TTensorI(WHT(Log2Int(m)), n*k, AVec, AVec),
                        TTensorI(TTensorI(WHT(Log2Int(n)), k, AVec, AVec), m, APar, APar)
                    ]).withTags(nt.getTags()),
                    TCompose([
                        TTensorI(TTensorI(WHT(Log2Int(n)), k, AVec, AVec), m, APar, APar),
                        TTensorI(WHT(Log2Int(m)), n*k, AVec, AVec)
                    ]).withTags(nt.getTags()),
                 ])
            )), j->[j])
        ),
        apply := (nt, c, cnt) -> c[1]
    )
));
