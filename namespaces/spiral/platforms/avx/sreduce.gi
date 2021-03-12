
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(FixCodeAVX, RuleSet);
RewriteRules(FixCodeAVX, rec(
    fix_noneExp := Rule( noneExp, e -> e.t.zero()),

    vpermf128_8x32f_to_vextract := Rule( [assign, [deref, @(1)], [vpermf128_8x32f, @(2), @(3), @(4)]],
        e -> let( p := @(4).val.p, 
                  a := [[@(2).val, [0]], [@(2).val, [1]], [@(3).val, [0]], [@(3).val, [1]]],
                  dst := tcast(TPtr(TVect(T_Real(32), 4)), @(1).val),
                  chain(
                      assign( deref(dst  ), ApplyFunc(vextract_4l_8x32f, a[p[1]])),
                      assign( deref(dst+1), ApplyFunc(vextract_4l_8x32f, a[p[2]]))
                  )
        )),

    flatten_chains := ARule( chain, [@(1, chain)], e -> [@(1).val.cmds] ),

    addsub_4x64f_to_mul := Rule( [addsub_4x64f, _0, @(1)],
        e -> mul(e.t.value([-1,1,-1,1]), @(1).val)),
    addsub_8x32f_to_mul := Rule( [addsub_8x32f, _0, @(1)],
        e -> mul(e.t.value([-1,1,-1,1,-1,1,-1,1]), @(1).val)),

    # replacing 32 byte load by 16 byte load which is faster for out of L1 problems.
    deref_to_doubleload := Rule( 
        @@(1, deref, (x, cx) -> _avxT(x.t, cx.opts) and cx.isInside(assign) and Last(cx.assign).loc<>x and 
                                cx.isInside(rdepth_marker) and Last(cx.rdepth_marker).depth=1),
        (e, cx) -> Cond( 
            e.t = TVect(T_Real(64), 4) or e.t = TVect(TReal, 4),
                let( p := tcast(TPtr(TVect(T_Real(64), 2)), e.loc), 
                     vinsert_2l_4x64f( vinsert_2l_4x64f(e.t.zero(), deref(p), [0]), deref(p + 1), [1])),
            e.t = TVect(T_Real(32), 8) or e.t = TVect(TReal, 8),
                let( p := tcast(TPtr(TVect(T_Real(32), 4)), e.loc), 
                     vinsert_4l_8x32f( vinsert_4l_8x32f(e.t.zero(), deref(p), [0]), deref(p + 1), [1])),
            Error("unexpected type"))
    ),

    vloadu_8x32f_to_doubleload := Rule( @@(1, vloadu_8x32f, (x, cx) -> cx.isInside(rdepth_marker) and Last(cx.rdepth_marker).depth=1),
        (e, cx) -> vinsert_4l_8x32f( vinsert_4l_8x32f(e.t.zero(), vloadu_4x32f(e.args[1]), [0]), vloadu_4x32f(e.args[1] + 4), [1])
    ),

    vloadu_4x64f_to_doubleload := Rule( @@(1, vloadu_4x64f, (x, cx) -> cx.isInside(rdepth_marker) and Last(cx.rdepth_marker).depth=1),
        (e, cx) -> vinsert_2l_4x64f( vinsert_2l_4x64f(e.t.zero(), vloadu_2x64f(e.args[1]), [0]), vloadu_2x64f(e.args[1] + 2), [1])
    ),

));

RewriteRules(RulesStrengthReduce, rec(
    # a dupe
    #tcast_fold := Rule([tcast, @(1), [tcast, @(2), @(3)]], e->tcast(@(1).val, @(3).val)),

    avx_add_addsub_vzero := Rule([add, @(1), [@(2, [addsub_4x64f, addsub_8x32f]), _0, @(3)]], 
        e -> ObjId(@(2).val)(@(1).val, @(3).val)), 

    avx_addsub_vzero := Rule([@(1, [addsub_4x64f, addsub_8x32f]), @(2), _0], e -> @(2).val),

    # is it valid?
    avx_addsub_noneExp := Rule([@(1, [addsub_4x64f, addsub_8x32f]), @(2, noneExp), @(3)], 
	e -> ObjId(@(1).val)(@(2).val.t.zero(), @(3).val)),

    avx_addsub_mul_mul_values := Rule([@(0, [add, sub, addsub_4x64f, addsub_8x32f]), [mul, @(1, Value), @(2)], [mul, @@(3, Value, (x, cx)->x=@(1).val and _avxT(@(0).val.t, cx.opts)), @(4)]],
        e -> mul(@(1).val, ObjId(@(0).val)(@(2).val, @(4).val))),

    # replace by mul const?
    avx_addsub_zero_neg := Rule( [@(1, [addsub_4x64f, addsub_8x32f]), _0, [neg, @(2)]],
        e -> neg(ObjId(@(1).val)(e.t.zero(), @(2).val))),

    avx_vunpack_4x64f_neg := Rule([@(1, [vunpacklo_4x64f, vunpackhi_4x64f]), [neg, @(2)], [neg, @(3)]],
       e -> neg(ObjId(e)(@(2).val, @(3).val))),

    #NOTE: this is real hack, unpacking from vdup(TVect(TReal,2),2) is simply a vdup(TReal, 4), how to do this right?
    # need the same for AVX_8x32f
    avx_vunpacklo_4x64f_vdup_2 := Rule([vunpacklo_4x64f, [@(1, vdup), nth, _2], 
           [@(2, vdup, x -> x=@(1).val), @(3, nth, x -> x.loc _is var and not IsBound(x.loc.value)), _2]],
       e -> let(n := @(3).val, vdup(deref(nth(n.loc, 0).toPtr(e.t.t) + 2*n.idx), 4))),
    avx_vunpackhi_4x64f_vdup_2 := Rule([vunpackhi_4x64f, [@(1, vdup), nth, _2], 
           [@(2, vdup, x -> x=@(1).val), @(3, nth, x -> x.loc _is var and not IsBound(x.loc.value)), _2]],
       e -> let(n := @(3).val, vdup(deref(nth(n.loc, 0).toPtr(e.t.t) + 2*n.idx+1), 4))),

    
    avx_vperm_neg := Rule([@(0, [vperm_4x64f, vperm_8x32f]), [neg, @(1)], @(2)],
        e -> neg(ObjId(e)(@(1).val, @(2).val))),
    
    avx_vshuffle_neg := Rule([@(0, [vshuffle_4x64f, vshuffle_8x32f]), [neg, @(1)], [neg, @(2)], @(3)],
       e -> neg(ObjId(@(0).val)(@(1).val, @(2).val, @(3).val))),

    avx_addsub_neg_neg := Rule([@(0, [addsub_4x64f, addsub_8x32f]), [neg, @(1)], [neg, @(2)]],
        e -> neg(ObjId(e)(@(1).val, @(2).val))),
	
    avx_fold_vshuffle := Rule(
	[@(1, [vshuffle_4x64f, vshuffle_8x32f]), @(2, Value, e->IsVecT(e.t)), 
	                                         @(3, Value, e->IsVecT(e.t)), @(4)],
        e -> Value(@(2).val.t, @(1).val.semantic(@(2).val.v, @(3).val.v, @(4).val.p))),

    # op(v1*a, v2*b) where v1*v2=0 turns into op(v1,v2)*vblend(a,b)
    avx_addsub_to_blend := Rule([@(1, [add, sub, addsub_4x64f, addsub_8x32f]), [mul, @(2, Value), @(3)], [mul, @@(4, Value, (x, cx) -> _avxT(x.t, cx.opts) and x*@(2).val=x.t.zero()), @(5)]],
        e -> let( c1 := @(2).val, 
                  c2 := @@(4).val,
                  c  := ObjId(e)(c1, c2),
                  b  := Cond( e.t.size = 4, vblend_4x64f, vblend_8x32f ),
                  c*b(@(3).val, @(5).val, List(c1.v, x -> Cond(x=0, 2, 1))))),

    avx_addsub_ushuffles := Rule([@(0, [add, sub]), 
            [@(1, [vshuffle_4x64f, vshuffle_8x32f]), @(2), @(3).cond(x -> x=@(2).val), @(4)],
            [@(5, [vshuffle_4x64f, vshuffle_8x32f]), @(6), @(7).cond(x -> x=@(6).val), @(8).cond(x -> x = @(4).val and ObjId(@(5).val)=ObjId(@(1).val))]],
        e -> ObjId(@(1).val)(ObjId(e)(@(2).val, @(6).val), ObjId(e)(@(2).val, @(6).val), @(4).val)),
    avx_addsub_vperm := Rule([@(0, [add, sub]), 
            [vperm_4x64f, @(1), @(2)], [vperm_4x64f, @(3), @(4).cond(x -> x = @(2).val)]],
        e -> vperm_4x64f(ObjId(e)(@(1).val, @(3).val), @(2).val)),

    avx_fmaddsub_4x64f := Rule(
	[@@(1, addsub_4x64f, (e,cx)->IsBound(cx.opts.vector) and cx.opts.vector.SIMD = "AVX3"), 
	    @(2, mul, e->Length(e.args)=2), @(3)],
        e-> fmaddsub_4x64f(@(2).val.args[1], @(2).val.args[2], @(3).val)),

    avx_fmaddsub_8x32f := Rule(
	[@@(1, addsub_8x32f, (e,cx)->IsBound(cx.opts.vector) and cx.opts.vector.SIMD = "AVX3"), 
	    @(2, mul, e->Length(e.args)=2), @(3)],
        e-> fmaddsub_8x32f(@(2).val.args[1], @(2).val.args[2], @(3).val)),

    avx_add_mul_to_addsub := Rule([add, [mul, @(1, Value), @(2)], 
                                        [mul, @@(3, Value, (x, cx) -> _avxT(x.t, cx.opts) and 
                                                ForAll([1..x.t.size], i -> x.v[i]*(1 - 2*(i mod 2))=@(1).val.v[i])), @(4)]],
        e -> mul(@(1).val, Cond(e.t.size=4, addsub_4x64f, addsub_8x32f)(@(2).val, @(4).val))),
    avx_sub_mul_to_addsub := Rule([sub, [mul, @(1, Value), @(2)], 
                                        [mul, @@(3, Value, (x, cx) -> _avxT(x.t, cx.opts) and 
                                                ForAll([1..x.t.size], i -> x.v[i]*(-1 + 2*(i mod 2))=@(1).val.v[i])), @(4)]],
        e -> mul(@(1).val, Cond(e.t.size=4, addsub_4x64f, addsub_8x32f)(@(2).val, @(4).val))),


    avx_drop_insert_mulmask := Rule( [mul, @(1, Value), [@(2, vinsert_2l_4x64f), _0, @, 
            @(3).cond( x -> (x.p[1]=0 and @(1).val=@(2).val.t.value([1, 1, 0, 0])) or 
                            (x.p[1]=1 and @(1).val=@(2).val.t.value([0, 0, 1, 1])))]],
        e -> @(2).val),

    avx_drop_store_mulmask_a := Rule( [@(1, vextract_2l_4x64f), [@(2, mul), @(3, Value), @(4)], 
            @(5).cond( x -> (x.p[1]=0 and @(3).val=@(2).val.t.value([1, 1, 0, 0])) or 
                            (x.p[1]=1 and @(3).val=@(2).val.t.value([0, 0, 1, 1])))],
        e -> vextract_2l_4x64f(@(4).val, @(5).val)),

    avx_drop_store_mulmask_b := Rule( [@(1, vextract_2l_4x64f), [@(2, [add, sub]), @(3), [mul, @(4, Value), @(5)]], 
            @(6).cond( x -> (x.p[1]=0 and @(4).val=@(2).val.t.value([1, 1, 0, 0])) or 
                            (x.p[1]=1 and @(4).val=@(2).val.t.value([0, 0, 1, 1])))],
        e -> vextract_2l_4x64f(ObjId(@(2).val)(@(3).val, @(5).val), @(6).val)),

    avx_drop_mulmask32f_a := Rule( [mul, @(1, Value), [@(2, vpermf128_8x32f), @(3), _0,
            @(4).cond( x -> x.p[2]>=3 and @(1).val=@(2).val.t.value([1,1,1,1,0,0,0,0]) )]],
        e -> @(2).val),

    avx_drop_mulmask32f_b := Rule( [vextract_4l_8x32f, [@(1, [add, sub]), @(2), [@(3, vperm_8x32f), [mul, @(4, Value), @(5)], @(6)]], 
            @(7).cond( x -> (x.p[1]=0 and @(4).val = @(1).val.t.value([1,1,1,1,0,0,0,0])) or
                            (x.p[1]=1 and @(4).val = @(1).val.t.value([0,0,0,0,1,1,1,1])))],
        e -> vextract_4l_8x32f(ObjId(@(1).val)(@(2).val, vperm_8x32f(@(5).val, @(6).val)), @(7).val)),

    avx_drop_mulmask32f_c := Rule( [mul, @(1, Value), [@(2, vinsert_4l_8x32f), _0, [vload_2l_4x32f, _0, @(3)], 
            @(4).cond( x -> x.p[1]=0 and @(1).val = @(2).val.t.value([1,1,0,0,0,0,0,0]) )]],
        e -> @(2).val ),

    avx_term_cxtr_8x32f := Rule( vcxtr_8x32f,
        e -> vblend_8x32f(e.args[1], vperm_8x32f(vupermf128_8x32f(e.args[1], [ 2, 1 ]), [3,4,1,2]), [1,1,2,2,2,2,1,1])),

    # complex vectorization loop rules
    # 4-way AVX
    AVX_4x64f_load_TComplex := Rule([@@(1, vpack, (e,cx)-> IsBound(cx.opts) and IsBound(cx.opts.vector) and IsBound(cx.opts.vector.isa) and
        cx.opts.vector.isa = AVX_4x64f),
        [@(2,re), nth], @(3, im, e->e.args[1]=@(2).val.args[1]), @(4,re, e->e.args[1]=@(2).val.args[1]), @(5,im, e->e.args[1]=@(2).val.args[1])],

        e->let(dptr := @(2).val.args[1].toPtr(TVect(T_Real(64), 2)), src := vinsert_2l_4x64f(vzero_4x64f(),deref(dptr),[0]),
            vpermf128_4x64f(src, src, [1,1]))),

    AVX_4x64f_fold_shuffle := Rule([@(1, vshuffle_4x64f), [@(2, vpermf128_4x64f), @(3, vinsert_2l_4x64f), @(6), @(7)],
        @(4).cond(e->e=@(2).val), @(5, vparam, e->Length(Set(e.p))=1)],
        e->vdup(deref(@(3).val.args[2].toPtr(T_Real(64)) + (@(5).val.p[1]-1)), 4)),

    AVX_4x64f_load_TVect := Rule([@@(1, vpack, (e,cx)-> IsBound(cx.opts) and IsBound(cx.opts.vector) and IsBound(cx.opts.vector.isa) and
        cx.opts.vector.isa = AVX_4x64f),
        @(2,nth,e->IsBound(e.loc.init)), @(3, nth, e->e.loc=@(2).val.loc), @(4,nth, e->e.loc=@(2).val.loc), @(5,nth, e->e.loc=@(2).val.loc)],
        e->deref(@(2).val.toPtr(TVect(T_Real(64), 4)))),

#    # 8-way AVX

    AVX_8x32f_broadcast_re := Rule([@(0, vshuffle_8x32f), [@@(1, vpack, (e,cx)-> IsBound(cx.opts) and IsBound(cx.opts.vector) and IsBound(cx.opts.vector.isa) and cx.opts.vector.isa = AVX_8x32f),
                [@(2,re), nth], @(3, im, e->e.args[1]=@(2).val.args[1]),
                @(4,re, e->e.args[1]=@(2).val.args[1]), @(5,im, e->e.args[1]=@(2).val.args[1]),
                @(6,re, e->e.args[1]=@(2).val.args[1]), @(7,im, e->e.args[1]=@(2).val.args[1]),
                @(8,re, e->e.args[1]=@(2).val.args[1]), @(9,im, e->e.args[1]=@(2).val.args[1])],
            @(10).cond(e->e=@@(1).val), @(11,vparam, e->e.p=[1,1,3,3])],
        e->vdup(@(2).val, 8)),

    AVX_8x32f_broadcast_im := Rule([@(0, vshuffle_8x32f), [@@(1, vpack, (e,cx)-> IsBound(cx.opts) and IsBound(cx.opts.vector) and IsBound(cx.opts.vector.isa) and cx.opts.vector.isa = AVX_8x32f),
                [@(2,re), nth], @(3, im, e->e.args[1]=@(2).val.args[1]),
                @(4,re, e->e.args[1]=@(2).val.args[1]), @(5,im, e->e.args[1]=@(2).val.args[1]),
                @(6,re, e->e.args[1]=@(2).val.args[1]), @(7,im, e->e.args[1]=@(2).val.args[1]),
                @(8,re, e->e.args[1]=@(2).val.args[1]), @(9,im, e->e.args[1]=@(2).val.args[1])],
            @(10).cond(e->e=@@(1).val), @(11,vparam, e->e.p=[2,2,4,4])],
        e->vdup(@(3).val, 8)),

    AVX_8x32f_load_TVect := Rule([@@(1, vpack, (e,cx)-> IsBound(cx.opts) and IsBound(cx.opts.vector) and IsBound(cx.opts.vector.isa) and
        cx.opts.vector.isa = AVX_8x32f),
        @(2,nth,e->IsBound(e.loc.init)), @(3, nth, e->e.loc=@(2).val.loc), @(4,nth, e->e.loc=@(2).val.loc), @(5,nth, e->e.loc=@(2).val.loc),
        @(6,nth,e->IsBound(e.loc.init)), @(7, nth, e->e.loc=@(2).val.loc), @(8,nth, e->e.loc=@(2).val.loc), @(9,nth, e->e.loc=@(2).val.loc)],
        e->deref(@(2).val.toPtr(TVect(T_Real(64), 8))))
));


AVXFixProblems := function(c, opts)
    local zerovar, emptyvar, cc;

#    emptyvar := var.fresh_t("ZERO", TVect(TReal, opts.vector.isa.v));
#    zerovar := var.fresh_t("ZERO", TVect(TReal, opts.vector.isa.v));
#
#    cc := SubstTopDown(c, [@(1, neg), @(2)], e->sub(zerovar, @(2).val));
#    cc := SubstTopDown(c, @(1, var, e->IsBound(e.value) and IsList(e.value.v) and ForAll(e.value.v, IsZero)), e->zerovar);
#
#    c := decl([zerovar, emptyvar], chain(assign(zerovar, bin_xor(emptyvar, emptyvar)), cc));
#
#    if Collect(c, vpack) <> [] then Error("Caught in FixProblems"); fi;
    c := UntilDone(c, MergedRuleSet(RulesStrengthReduce, FixCodeAVX), opts);

    return c;
end;

AVX_4x64f.fixProblems := AVXFixProblems;
AVX_8x32f.fixProblems := AVXFixProblems;
