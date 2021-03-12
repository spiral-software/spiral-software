
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_testJam := (t) -> let(
    xx1 := Last(t.params[2].params[1]),
    xx2 := Last(t.params[3].params[1]),

    (xx1 <> 0 or xx2 <> 0)
    and let(
        M := t.params[1].dims()[1],
        K := t.getTag(ABB).params[1],
        n := When(xx1 <> 0, t.params[4][xx1], t.params[4][xx2]),
		
		n <= K/2
    )
);

_testIxA := (t) -> let(
    @par := @.cond(e -> e = GTPar),

	PatternMatch(t, [GT, @(1), @par, @par, \.\.\. ], empty_cx())
    and let(
        M := @(1).val.dims()[1],
        K := t.getTag(ABB).params[1],
        m := t.dims()[1],

        K > M
        and m > K
    )
);

_testIxAxI := (t) -> let(
    @ixi := @.cond(e -> e = XChain([1,0,2])),

    PatternMatch(t, [GT, @(1), @ixi, @ixi, \.\.\. ], empty_cx())
    and let(
        M := @(1).val.dims()[1],
        K := t.getTag(ABB).params[1],
        m := t.dims()[1],
        n := t.params[4][1],
        p := t.params[4][2],

		(K/2 < p
		and 1 < K/2)
		or (
			m > K 
			and K > M*p
		)
    )
);

_testAxI := (t) -> let(
    @vec := @.cond(e -> e = GTVec),

    PatternMatch(t, [GT, @(1), @vec, @vec, \.\.\. ], empty_cx())
    and let(
        M := @(1).val.dims()[1],
        K := t.getTag(ABB).params[1],
        m := t.dims()[1],
        n := t.params[4][1],

		K/2 < n
		and 1 < K/2
    )
);

NewRulesFor(GT, rec(

    # splits a loop based on basic block size K
    # K is passed in with the tag ABB
    # I_N x A_M -> I_(N/K/M) x I_(K/M) x A_M

    GT_BB_IxA := rec(
        forTransposition := false,
        switch := false,
        applicable := (t) -> 
            t.hasTag(ABB)
            and _testIxA(t),

        children := (self, t) >> let(
            K := t.getTag(ABB).params[1],
            spl := t.params[1],
            M := spl.dims()[1],
            n := t.params[4][1],

            r := (K / M),
            s := n / r,

            # if q <=1, just drop the tag, otherwise split the loop
            [[
                GT(
                    spl, 
                    XChain([2,1,0]), 
                    XChain([2,1,0]),
                    [r, s]
                ).withTags(Filtered(t.getTags(), e -> ObjId(e) <> ABB))
            ]]
        ),

        apply := (self, t, C, nt) >> C[1]
    ),

    # I_n x A_m x I_p, tagged with (K, mu)
    # 1) mp < K
    # 2) mp >= K
    #    a) m > K/mu
    #    b) m <= K/mu
    #    a) p > mu
    #    b) p <= mu

    GT_BB_IxAxI := rec(
        forTransposition := false,
        switch := false,
        applicable := (t) -> 
            t.hasTag(ABB)
			and _testIxAxI(t),

        children := (self, t) >> let(
            K := t.getTag(ABB).params[1],
            M := t.params[1].dims()[1], 
            spl := t.params[1],
            n := t.params[4][1],
            p := t.params[4][2],
			m := t.dims()[1],

            Cond(
                # cond1
                m > K and K > M*p,
                let(
                    s := K / (M*p),
                    u := n/s,

                    [[GT(
                         spl, XChain([2,1,0,3]), XChain([2,1,0,3]), [s,u,p]
                    ).withTags(t.getTags())]]
                ),

                # cond2
				# p > K/2
                let(
                    s := K / 2,
                    u := p/s,

                    [[GT(
                        spl, XChain([1,0,3,2]), XChain([1,0,3,2]), [n,s,u]
                    ).withTags(t.getTags())]]
                )
            )
        ),

        apply := (self, t, C, nt) >> C[1]
    ),

    #
    GT_BB_AxI := rec(
        forTransposition := false,
        switch := false,
        applicable := (t) -> 
            t.hasTag(ABB)
			and _testAxI(t),

        children := (self, t) >> let(
            M := t.params[1].dims()[1],
            K := t.getTag(ABB).params[1],
            n := t.params[4][1],
            spl := t.params[1],

            r := K / 2,
            s := n / r,

            [[GT(
                 spl, XChain([0,2,1]), XChain([0,2,1]), [r,s]
            ).withTags(t.getTags())]]
        ),

        apply := (self, t, C, nt) >> C[1]
    ),

    # when K <= M, we just drop the tag.
 
    GT_BB_DropTag := rec(
        forTransposition := false,
        switch := false,
        applicable := (self, t) >> 
            t.hasTag(ABB)
            and ObjId(t) = GT
            and let(
                K := t.getTag(ABB).params[1],

                K = 2
            ),

        # build a new GT object without the ABB tag.
        children := (self, t) >>
            [[
                t.withoutTag(ABB)
            ]],

        apply := (self, t, C, nt) >> C[1],
    ),

    # just like nthloop except only applicable when
    # the Jam version below isn't.

    GT_BB_NthLoop := CopyFields(GT_NthLoop, rec(
        requiredFirstTag := [AExpRight,ANoTag,ALimitNthLoop,ABB],
        applicable := (t) -> 
            Length(t.params[4]) > 0
			and (
				not t.hasTag(ABB)
				or (
					t.hasTag(ABB)
					and not _testIxA(t)
					and not _testIxAxI(t)
					and not _testAxI(t)
					and not _testJam(t)
				)
			)
    )),

    GT_BB_NthLoopJam := rec(
        requiredFirstTag := [ABB,ALimitNthLoop],
        applicable := (t) -> 
            Length(t.params[4]) > 0
			and t.hasTag(ABB)
            and _testJam(t),

        # choose the rightmost index.
        freedoms := t -> let(
            xx1 := Last(t.params[2].params[1]),
            xx2 := Last(t.params[3].params[1]),
        
            When(xx1 <> 0, [[xx1]], [[xx2]])
        ),
            

        child := (t, fr) -> let(
            spl := t.params[1],
            M := spl.dims()[1],
            g := t.params[2], 
            s := t.params[3], 
            loopid := fr[1],
            newK := t.getTag(ABB).params[1]/t.params[4][loopid],

            When(t.hasTag(ALimitNthLoop),
                [ 
                    GT(spl, 
                        g.without(loopid), 
                        s.without(loopid), 
                        ListWithout(t.params[4], loopid)).withTags(
                            When(newK = 2,
                                Filtered(t.getTags(), e -> ObjId(e) <> ABB),
                                Concat(
                                    [ABB(newK)],
                                    Filtered(t.getTags(), e -> ObjId(e) <> ABB)
                                )
                            )
                        ),
                    InfoNt(loopid) 
                ],

                [ 
                    GT(spl, g.without(loopid), 
                        s.without(loopid), ListWithout(t.params[4], loopid)), 
                    InfoNt(loopid) 
                ]
            )
        ),


        apply := (t, C, nt) -> let(
            loopid := nt[2].params[1],
            dft := nt[1].params[1],
            g := t.params[2],
            s := t.params[3],
            loop_dims := t.params[4],
            i := Ind(loop_dims[loopid]),

            JamISum(i, 
                Scat(s.part(loopid, i, Rows(dft), loop_dims)) 
                * C[1] 
                * Gath(g.part(loopid, i, Cols(dft), loop_dims))
            ) 
        )
    )
));
