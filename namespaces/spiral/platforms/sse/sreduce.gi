
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


##################################################################################
#   vector strength reduction - experimental!!!

zero_one   := @.cond(e -> (IsValue(e) and (e.v=[0,1]  or e.v=[0.0,1.0])));
one_zero    := @.cond(e -> (IsValue(e) and (e.v=[1,0]  or e.v=[1.0,0.0])));

zero_mone   := @.cond(e -> (IsValue(e) and (e.v=[0,-1]  or e.v=[0.0,-1.0])));
mone_zero    := @.cond(e -> (IsValue(e) and (e.v=[-1,0]  or e.v=[-1.0,0.0])));


#   WARNING: these rules may be SSE2-specific. Especially the order of @2, @1 may be because of _mm_shuffle_pd
# NOTE: YSV reenable this

# Checks alignment of an expr.
# Pointers are assumed to be aligned.
# Type of the expression is assumed to be the "natural" one (i.e. float for 4-way)
# e.g.: for SSE_4x32f: X+3 is unaligned and X+4 is aligned
Declare(_isAligned);
_isAligned := function(c, v)
     if ObjId(c)=add then
         return ForAll(c.args, x->_isAligned(x, v));
     elif ObjId(c)=tcast then
         return _isAligned(c.args[2], v);
     elif ObjId(c) = var and ObjId(c.t)=TPtr then
         return not IsUnalignedPtrT(c.t);
     elif ObjId(c) = mul then
         return ForAny(c.args, x->_isAligned(x, v));

     elif ObjId(c) = Value and (IsVecT(c.t) or IsArrayT(c.t)) then # XXX YSV: is this correct?
	 return true;

     elif ObjId(c) = Value then
         return (Mod(c.v,v)=0);
     elif ObjId(c) = var then
         return false;
     else
         return false; #By default, return false
     fi;
     return true;
end;

_isISA := (t, vec_isa, opts) -> IsVecT(t) and t.size=vec_isa.t.size
                             and ((IsBound(opts.isas) and vec_isa in opts.isas) or
                                  (IsBound(opts.vector) and vec_isa = opts.vector.isa));

# pull "change sign" from add and sub
_rule_addSub_chs_chs := (chs) -> Rule( [@(3, [add, sub]), [chs, @(1)], [chs, @(2)]], e -> chs(ObjId(@(3).val)(@(1).val, @(2).val)));

RewriteRules(RulesStrengthReduce, rec(
    #
    # These rules replaces unaligned accesses by aligned accesses whenever possible
    #
    # This rules are disabled because _isAligned may incorrectly return true when pointer expression
    # splitted in compiler
    #
    #replace_unaligned_load := Rule(
    #    [ @(1, [vloadu_2x64f, vloadu_4x32f, vloadu_8x16i, vloadu_16x8i]),
    #      @(2).cond(e -> _isAligned(e, @(1).val.v)) ],
    #    e -> deref(tcast(TPtr(@(1).val.t), @(2).val))
    #),

    #replace_unaligned_store := Rule(
    #    [ @(1, [vstoreu_2x64f, vstoreu_4x32f, vstoreu_8x16i, vstoreu_16x8i]),
    #      @(2),
    #      @(3).cond(e -> _isAligned(@(2).val, e.t.size)) ],
    #    e -> assign(deref(tcast(TPtr(@(3).val.t), @(2).val)),@(3).val)
    #),

#     add_mul1001a := Rule([add, [mul, @(1), one_zero], [mul, @(2), zero_one]],  e ->
#         let(ins := _SIMDArch().rules.x_I_vby2[1][2], ins.instr(@(1).val, @(2).val, ins.p))), #oddly I had to switch order of @1, @2 and not [1][2]->[2][1] ??
#     add_mul1001b := Rule([add, [mul, one_zero, @(1)], [mul, zero_one, @(2)]],  e ->
#         let(ins := _SIMDArch().rules.x_I_vby2[1][2], ins.instr(@(1).val, @(2).val, ins.p))),

#     add_mul0110a := Rule([add, [mul, @(1), zero_one], [mul, @(2), one_zero]],  e ->
#         let(ins := _SIMDArch().rules.x_I_vby2[1][2], ins.instr(@(2).val, @(1).val, ins.p))),
#     add_mul0110b := Rule([add, [mul, zero_one, @(1)], [mul, one_zero, @(2)]],  e ->
#         let(ins := _SIMDArch().rules.x_I_vby2[1][2], ins.instr(@(2).val, @(1).val, ins.p))),

#     sub_mul1001b := Rule([sub, [mul, one_zero, @(1)], [mul, zero_one, @(2)]],  e ->
#         let(ins := _SIMDArch().rules.x_I_vby2[1][2], ins.instr(@(1).val, neg(@(2).val), ins.p))),
#     sub_mul0110b := Rule([sub, [mul, zero_one, @(1)], [mul, one_zero, @(2)]],  e ->
#         let(ins := _SIMDArch().rules.x_I_vby2[1][2], ins.instr(@(1).val, @(2).val, ins.p))),

# replace unaligned->aligned if possible
#    align_load:= Rule([@(1, [vloadu_2x64f, vloadu_4x32f, vloadu_8x16i]), @(2, vref, e->IsValue(e.idx) and IsInt(e.idx.ev()/@1.val.v)) ], e->vref(@2.val.loc, @2.val.idx, @1.val.v)),

    ushuffle_neg := Rule([@(1, [vushuffle_2x64f, vushuffle_4x32f]), [neg, @(2)], @(3)],
        e -> neg(ObjId(@(1).val)(@(2).val, @(3).val))),

    addsub00 := Rule([@(1,addsub_2x64f), _0, _0], e->v0),
    addsubA0 := Rule([@(1,addsub_2x64f), @(2).cond(e->ObjId(e) <> Value), _0], e->@(2).val),

    mul1A := Rule([@(1, addsub_2x64f), [@(2, mul), zero_one, @(3)], [@(4, mul), one_zero, @(5)]], e->
        addsub_2x64f(0, vshuffle_2x64f(@(5).val, @(3).val, [1,2]))),

    mul1B := Rule([@(1, addsub_2x64f), [@(2, mul), one_zero, @(3)], [@(4, mul), zero_one, @(5)]], e->
        vshuffle_2x64f(@(3).val, @(5).val, [2,1])),

    mul2A := Rule([@(1, addsub_2x64f), [@(2, mul), zero_mone, @(3)], [@(4, mul), mone_zero, @(5)]], e->
        let(v01 := Value(TVectDouble(2), [0, 1]), v10 := Value(TVectDouble(2), [1, 0]),
            vushuffle_2x64f(addsub_2x64f(v01 * @(5).val, v10 * @(3).val), [2,1]))),

    mul2B := Rule([@(1, addsub_2x64f), [@(2, mul), mone_zero, @(3)], [@(4, mul), zero_mone, @(5)]], e->
        let(v01 := Value(TVectDouble(2), [0, 1]), v10 := Value(TVectDouble(2), [1, 0]),
            neg(vshuffle_2x64f(@(3).val, @(5).val, [2,1])))),

    shuf1 := Rule([@(1, vshuffle_2x64f), @(2, vushuffle_2x64f), @(3, vushuffle_2x64f), @(4, vparam)], e->
        let(p :=@(4).val.p, p1:=@(2).val.args[2].p, p2:=@(3).val.args[2].p,
            v1 := @(2).val.args[1], v2 := @(2).val.args[1],
            vshuffle_2x64f(v1, v2, [p1[p[1]], p2[p[2]]]))),

    shuf2 := Rule([@(1, vshuffle_2x64f), @(2), @(3).cond(e->e=@(2).val), @(4, vparam)], e->
        vushuffle_2x64f(@(2).val, @(4).val.p)),

    shuf3 := Rule([@(1, vushuffle_2x64f), @(2), @(3, vparam, e->e.p = [1,2])], e-> @(2).val),

    _mm_min_4way_noSSE41_replace := Rule(@@(1,min, (e,cx) -> not(LocalConfig.cpuinfo.SIMD().hasSSE4_1()) and
        IsBound(cx.opts.vector) and
            cx.opts.vector.isa=SSE_4x32i and
            e.t=TVect(TReal, 4)),
	e -> bin_or(bin_andnot(gt(@@(1).val.args[2],@@(1).val.args[1]),@@(1).val.args[2]),
	            bin_and(gt(@@(1).val.args[2],@@(1).val.args[1]),@@(1).val.args[1]))),

    _rshift_16way_replace := Rule(@@(1, bin_shr, (e,cx) -> (
            IsBound(cx.opts.vector) and
            cx.opts.vector.isa=SSE_16x8i and
            e.t=cx.opts.vector.isa.t)),
	e -> bin_and(tcast(e.t, bin_shr(tcast(TVect(TReal, 8), @@(1).val.args[1]), @@(1).val.args[2])),
	             vdup(TInt.value(bin_shr(256,@@(1).val.args[2])-1), 16))),

    #note: this does a gte and not a gt
    _gt_16way_unsigned_replace := Rule(@@(1, gt, (e,cx) -> (
            IsBound(cx.opts.vector) and
            cx.opts.vector.isa=SSE_16x8i and
            not (cx.opts.vector.isa.isSigned) and
            e.t=cx.opts.vector.isa.t)),
        e -> eq(min(@@(1).val.args[2], @@(1).val.args[1]), @@(1).val.args[2])
    ),


    vload1_4x32f_cond := Rule( [vload1_4x32f, @(1, cond)],
        e -> _map_cond(@(1).val, p->p, exp->vload1_4x32f(exp))),
    vload1_4x32f_funcExp := Rule( [vload1_4x32f, [funcExp, @(1, Value)]],
        e -> e.t.value([@(1).val, 0, 0, 0])),  # must be _mm_set_ss

    vload_2h_4x32f_cond := Rule( [vload_2h_4x32f, @(1), @(2, cond)],
        e -> _map_cond(@(2).val, p->p, exp->vload_2h_4x32f(@(1).val, exp))),
    vload_2l_4x32f_cond := Rule( [vload_2l_4x32f, @(1), @(2, cond)],
        e -> _map_cond(@(2).val, p->p, exp->vload_2l_4x32f(@(1).val, exp))),

    vload_2h_4x32f_funcExp := Rule( [vload_2h_4x32f, @(1, Value), [funcExp, @(2, Value)]],
        e -> vshuffle_4x32f(@(1).val, @(2).val, [1,2,3,4])),
    vload_2l_4x32f_funcExp := Rule( [vload_2l_4x32f, @(1, Value), [funcExp, @(2, Value)]],
        e -> vshuffle_4x32f(@(2).val, @(1).val, [1,2,3,4])),

    vshuffle_4x32f_cond_r := Rule( [vshuffle_4x32f, @(1), @(2, cond), @(3)],
        e -> _map_cond(@(2).val, p->p, exp->vshuffle_4x32f(@(1).val, exp, @(3).val))),
    vshuffle_4x32f_cond_l := Rule( [vshuffle_4x32f, @(1, cond), @(2), @(3)],
        e -> _map_cond(@(1).val, p->p, exp->vshuffle_4x32f(exp, @(2).val, @(3).val))),
    vushuffle_4x32f_cond := Rule( [vushuffle_4x32f, @(1, cond), @(2)],
        e -> _map_cond(@(1).val, p->p, exp->vshuffle_4x32f(exp, @(2).val))),

));

Class(RulesPreStrengthReduce, RuleSet);
RewriteRules(RulesPreStrengthReduce, rec(
    _avg_16way_replace := Rule([@@(1, add, (e,cx) -> (
            IsBound(cx.opts.vector) and
            cx.opts.vector.isa=SSE_16x8i and
            e.t=cx.opts.vector.isa.t)),[bin_shr,@(2),@(3,Value,e->e.v=1)],[bin_shr,@(4),@(5,Value,e->e.v=1)]],
        (e, cx)-> cx.opts.vector.isa.average(@(2).val,@(4).val)),
));

# recursive condition function for push_change_sign rule.
# if tree consists from sub and add with [mul, Value, @] leaves only  we can
# merge "change sign" with values.

Declare(_canEliminateChangeSign);
_canEliminateChangeSign := (tree) -> CondPat( tree,
    add, ForAll(tree.args, _canEliminateChangeSign),
    sub, ForAll(tree.args, _canEliminateChangeSign),
    [mul, Value, @], true,
    false);

RewriteRules(RulesStrengthReduce, rec(
    require_load := Rule(@@(1, deref, (e,cx) -> IsBound(cx.opts.vector) and
	    IsBound(cx.opts.vector.isa.requireLoad) and cx.opts.vector.isa.requireLoad and
        ((IsBound(cx.assign) and cx.assign<>[] and cx.assign[1].exp = @@(1).val) or (ForAny([add, sub, mul,
            vunpacklo_4x32f, vunpackhi_4x32f, addsub_4x32f, hadd_4x32f, vshuffle_4x32f, vushuffle_4x32f], i-> IsBound(cx.(i.name)) and cx.(i.name)<>[] ))) ),
#        ((IsBound(cx.assign) and cx.assign<>[] and ObjId(cx.assign[1].exp) = deref and ObjId(cx.assign[1].loc) <> deref) or (ForAny([add, sub, mul, vunpacklo_4x32f], i-> IsBound(cx.(i.name)) and cx.(i.name)<>[] ))) ),
        (e, cx)-> cx.opts.vector.isa.loadop(@@(1).val.loc)),

    require_store := Rule([@@(1, assign, (e,cx) -> (
            IsBound(cx.opts.vector) and IsBound(cx.opts.vector.isa.requireStore) and cx.opts.vector.isa.requireStore)),
            @(2, deref), @(3)],
        (e, cx)-> cx.opts.vector.isa.storeop(@(2).val, @(3).val)),

    require_store_var_SSE_2x32f := Rule([@@(1, assign, (e,cx) -> (IsBound(cx.opts.vector) and cx.opts.vector.isa = SSE_2x32f)),
            @(2, deref), @(3, Value)],
        (e, cx)-> let(vv := var.fresh_t("V", TVect(TReal, cx.opts.vector.isa.v)),
            decl(vv, chain(assign(vv, @(3).val), cx.opts.vector.isa.storeop(@(2).val, vv))))),

    fix_unpackhi_2x32f := Rule(@(1, vunpackhi_2x32f),
        e -> vshuffle_4x32f(vunpacklo_4x32f(@(1).val.args[1], @(1).val.args[2]), vunpacklo_4x32f(@(1).val.args[1], @(1).val.args[2]), vparam([3,4,3,4]))),

    fix_unpacklo_2x32f := Rule(@(1, vunpacklo_2x32f), e -> vunpacklo_4x32f(@(1).val.args[1], @(1).val.args[2])),

    # fix_vzero_2x32f := Rule(@(1, vzero_2x32f), e -> vzero_4x32f()),

    fix_vload_2x32f := Rule(@(1, vload_2x32f), e -> vload_2l_4x32f(@(1).val.args[1], @(1).val.args[2])),
    fix_vstore_2x32f := Rule(@(1, vstore_2x32f), e -> vstore_2l_4x32f(@(1).val.loc, @(1).val.exp)),
    fix_addsub_2x32f := Rule(@(1, addsub_2x32f), e -> addsub_4x32f(@(1).val.args[1], @(1).val.args[2])),

    addsub_zero_r := Rule([@(1, addsub_4x32f), @(2), _0], e -> @(2).val),

    # NOTE: verify this rule
    addsub_zero_l := Rule([@(1, addsub_4x32f), _0, @(2)], e -> chslo_4x32f(@(2).val)),

    chslo_4x32f_neg := Rule([chslo_4x32f, [neg, @(1)]], e -> chshi_4x32f(@(1).val)),
    chshi_4x32f_neg := Rule([chshi_4x32f, [neg, @(1)]], e -> chslo_4x32f(@(1).val)),
    chslo_2x64f_neg := Rule([chslo_2x64f, [neg, @(1)]], e -> chshi_2x64f(@(1).val)),
    chshi_2x64f_neg := Rule([chshi_2x64f, [neg, @(1)]], e -> chslo_2x64f(@(1).val)),

    chslo_4x32f_neg_2 := Rule([neg, [chslo_4x32f, @(1)]], e -> chshi_4x32f(@(1).val)),
    chshi_4x32f_neg_2 := Rule([neg, [chshi_4x32f, @(1)]], e -> chslo_4x32f(@(1).val)),
    chslo_2x64f_neg_2 := Rule([neg, [chslo_2x64f, @(1)]], e -> chshi_2x64f(@(1).val)),
    chshi_2x64f_neg_2 := Rule([neg, [chshi_2x64f, @(1)]], e -> chslo_2x64f(@(1).val)),

    # try to use same "change sign" everywhere
    add_@_chshi_4x32f := Rule([add, @(1), @(2, chshi_4x32f, x -> not (ObjId(@(1).val) in [chslo_4x32f, chshi_4x32f]))],
        e -> sub(@(1).val, chslo_4x32f(@(2).val.args[1]))),

    # put value to front in mul
    # NOTE: ESReduce has something like this one; rules below work only when Value is in front.
    mul_value_to_front := Rule( [mul, ...,  @(1), @(2, Value, x -> not IsValue(@(1).val)), ...],
        e -> let( a := SplitBy(e.args, IsValue), ApplyFunc(mul, [Product(a[1])] :: a[2]))),

    # "change sign" elimination
    mul_val_chs := ARule( mul, [@(1, Value), [@(2, [chshi_2x64f, chslo_2x64f, chshi_4x32f, chslo_4x32f]), @(3)] ],
        e -> let( chs := @(2).val,
                  [@(1).val * ObjId(chs)(chs.t.one()).ev(), @(3).val] )),

    chs_mul_val := Rule( [@(2, [chshi_2x64f, chslo_2x64f, chshi_4x32f, chslo_4x32f]), [mul, @(1, Value), @(3)]],
        e -> let( chs := @(2).val,
                  mul(@(1).val * ObjId(chs)(chs.t.one()).ev(), @(3).val))),

    mul_val_vushuffle_2x64f_chs := ARule( mul, [@(1, Value), [vushuffle_2x64f, [@(2, [chshi_2x64f, chslo_2x64f]), @(3)], @(4)] ],
        e -> let( chs := @(2).val,
                  vp  := @(4).val,
                  val := vushuffle_2x64f( ObjId(chs)(chs.t.one()).ev(), vp).ev(),
                  [@(1).val * val, vushuffle_2x64f( @(3).val, vp)] )),

    mul_val_vushuffle_4x32f_chs := ARule( mul, [@(1, Value), [vushuffle_4x32f, [@(2, [chshi_4x32f, chslo_4x32f]), @(3)], @(4)] ],
        e -> let( chs := @(2).val,
                  vp  := @(4).val,
                  val := vushuffle_4x32f( ObjId(chs)(chs.t.one()).ev(), vp).ev(),
                  [@(1).val * val, vushuffle_4x32f( @(3).val, vp)] )),

    # pull "change sign" from [add, sub]
    addSub_chshi_2x64f_chshi_2x64f := _rule_addSub_chs_chs(chshi_2x64f),
    addSub_chslo_2x64f_chslo_2x64f := _rule_addSub_chs_chs(chslo_2x64f),
    addSub_chshi_4x32f_chshi_4x32f := _rule_addSub_chs_chs(chshi_4x32f),
    addSub_chslo_4x32f_chslo_4x32f := _rule_addSub_chs_chs(chslo_4x32f),

    # push change sign into mul so it can be eliminated later
    push_change_sign := Rule([@(1, [chshi_2x64f, chslo_2x64f, chshi_4x32f, chslo_4x32f]), @(2, [add, sub], _canEliminateChangeSign)],
       e -> SubstTopDownNR( @(2).val, [mul, @(3, Value), @(4)], x -> mul(@(3).val, ObjId(@(1).val)(@(4).val)))),

    # pull same unary shuffle from add, sub and mul
    addSubMul_vushuffle_2x64f_vushuffle_2x64f := Rule( [@(5, [add, sub, mul]), [vushuffle_2x64f, @(1), @(2)], [vushuffle_2x64f, @(3), @(4).cond(x -> x=@(2).val)]],
        e -> vushuffle_2x64f( ObjId(@(5).val)( @(1).val, @(3).val ), @(2).val )),
    addSubMul_vushuffle_4x32f_vushuffle_4x32f := Rule( [@(5, [add, sub, mul]), [vushuffle_4x32f, @(1), @(2)], [vushuffle_4x32f, @(3), @(4).cond(x -> x=@(2).val)]],
        e -> vushuffle_4x32f( ObjId(@(5).val)( @(1).val, @(3).val ), @(2).val )),

    # pull value multiplier from unary shuffle. Disabled for now as it causing stalls on Core i7 2x64f
    # vushuffle_mul_value := Rule( [@(1, [vushuffle_2x64f, vushuffle_4x32f]), [mul, @(2, Value), @(3)], @(4)],
    #    e -> mul( ObjId(@(1).val)(@(2).val, @(4).val), ObjId(@(1).val)(@(3).val, @(4).val)) ),

    # V1*A + V2*vushuffle(A, P), V1*V2=0 => (V1+V2)*vushuffle(A, P`)
    merge_add_mul_mul_vushuffle_4x32f := ARule( add, [ [mul, @(1, Value), @(2)], [mul, @(3, Value), [@(0, vushuffle_4x32f), @(4), @(5).cond(x -> @(4).val = @(2).val and @(3).val*@(1).val = @(0).val.t.zero())]] ],
        e -> let( vp := @(5).val.p,
                  V1 := @(1).val,
                  V2 := @(3).val,
                  new_vp := List([1..Length(vp)], i -> When( V1.v[i] = 0, vp[i], i)),
                  [ mul( add(V1 + V2), vushuffle_4x32f( @(4).val, new_vp ))] )),

    # V1*A + vushuffle(V2*A, P), V1*vushuffle(V2,P)=0 => (V1+vushuffle(V2,P))*vushuffle(A, P`)
    merge_add_mul_vushuffle4x32f_mul := ARule( add, [ [mul, @(1, Value), @(2)], [@(0, vushuffle_4x32f), [mul, @(3, Value), @(4)], @(5).cond(x -> @(4).val = @(2).val and vushuffle_4x32f(@(3).val, x).ev()*@(1).val = @(0).val.t.zero())] ],
        e -> let( vp := @(5).val.p,
                  V1 := @(1).val,
                  V2 := @(3).val,
                  new_vp := List([1..Length(vp)], i -> When( V1.v[i] = 0, vp[i], i)),
                  [ mul( add(V1 + vushuffle_4x32f(V2, vp).ev()), vushuffle_4x32f( @(4).val, new_vp ))] )),

    remove_mul_load_2l_4x32f := Rule( [mul, @(1, Value), [@(2, vload_2l_4x32f, x -> @(1).val * x.t.value([ 1, 1, 0, 0]) = @(1).val), _0, @]],
        e -> @(2).val),

    store_zero := Rule([@(1, vstore_2l_4x32f), @(2), _0], e -> vstore_2l_4x32f(@(2).val, vzero_4x32f())),

    fix_vushuffle_2x32f := Rule([@(1, vushuffle_2x32f), @(2).cond(e->ObjId(e) <> Value), @(3)],
        e -> vshuffle_4x32f(@(1).val.args[1], @(1).val.args[1], vparam(Concat(@(1).val.args[2].p, @(1).val.args[2].p)))),

#   FF: NOTE: Why are all thes unaligned stores showing up?? This is a temporary fix. Yevgen, please tell me!
    fix_vstoreu_2x32f := Rule(@(1, vstoreu_2x32f), e -> vstore_2l_4x32f(@(1).val.loc, @(1).val.exp)),

    remove_shuffle := Rule([@(1, vstore_2l_4x32f), @(2), @(3, vushuffle_4x32f, e->e.args[2].p=[3,4,3,4])],
        e -> vstore_2h_4x32f(@(2).val, @(3).val.args[1])),

#    propagate_vushuffle := Rule(@(1, [add, sub, mul], e->ForAll(e.args, i->ObjId(i)=vushuffle_4x32f)),
#        e -> Error("Chaught")),
#
    assign_zero_xor := Rule([@@(1, assign, (e,cx)->IsBound(cx.opts.vector) and cx.opts.vector.isa = SSE_2x32f), @(2), @(3, Value, e->e.v=0)],
        (e, cx) -> let(zv := var.fresh_t("Z", TVect(TReal, cx.opts.vector.isa.v)), decl(zv, assign(@(2).val, vzero_2x32f())))),

    sreduce_vushuffle_1 := Rule([@(1, vushuffle_2x32f), @(2, Value, e->ObjId(e.t)=TVect), @(3)],
        e -> Value(@(2).val.t, List([1..@(2).val.t.size], i->@(2).val.v[@(3).val.p[i]]))),

    sreduce_vushuffle_2 := Rule([@(1, vushuffle_2x32f), @(2, Value, e->e.t=TInt), @(3)],
        e -> @(2).val),

#    fix_noneExp := Rule(@(1, noneExp), e->e.t.value(0)),

    addsub_noneExp_l := Rule([@(1, [addsub_2x64f, addsub_4x32f]), @(2, noneExp), @(3)], (e,cx)-> ObjId(@(1).val)(cx.opts.vector.isa.vzero(), @(3).val)),
    addsub_noneExp_r := Rule([@(1, [addsub_2x64f, addsub_4x32f]), @(2), @(3, noneExp)], (e,cx)-> @(2).val),

    add_addsub_zero_l := Rule([add, @(3), [@(1, [addsub_2x64f, addsub_4x32f]), _0, @(2)]], e -> ObjId(@(1).val)(@(3).val, @(2).val)),
    add_addsub_zero_r := Rule([add, [@(1, [addsub_2x64f, addsub_4x32f]), _0, @(2)], @(3)], e -> ObjId(@(1).val)(@(3).val, @(2).val)),

    addsub_addsub_zero_l := Rule( [@(1, [addsub_2x64f, addsub_4x32f]), [@(2, [addsub_2x64f, addsub_4x32f]), _0, @(3)], @(4)],
        e -> ObjId(@(1).val)(@(0).val, add(@(3).val, @(4).val))),
    addsub_addsub_zero_r := Rule( [@(1, [addsub_2x64f, addsub_4x32f]), @(4), [@(2, [addsub_2x64f, addsub_4x32f]), _0, @(3)]],
        e -> add(@(3).val, @(4).val)),

    addsub_zero_neg  := Rule([@(1, [addsub_2x64f, addsub_4x32f]), _0, [neg, @(2)]], e -> neg(ObjId(@(1).val)(@(0).val, @(2).val))),
    sub_addsub_zero := Rule([sub, [@(1, [addsub_2x64f, addsub_4x32f]), _0, @(2)], [@(3, [addsub_2x64f, addsub_4x32f]), _0, @(4)]],
        e -> ObjId(@(1).val)(@(0).val, sub(@(2).val, @(4).val))),

    sse_vshuffle_noneExp2 := Rule([@(1, [vshuffle_2x64f, vshuffle_4x32f]), @(2, noneExp), @(3, noneExp), @(4)], e->@(2).val),

    sse_vshuffle_noneExp1L := Rule([@(1, [vshuffle_2x64f, vshuffle_4x32f]), @(2, noneExp), @(3), @(4)],
	(e,cx)-> ObjId(@(1).val)(cx.opts.vector.isa.vzero(), @(3).val, @(4).val)),

    sse_vshuffle_noneExp1R := Rule([@(1, [vshuffle_2x64f, vshuffle_4x32f]), @(2), @(3, noneExp), @(4)],
	(e,cx)-> ObjId(@(1).val)(@(2).val, cx.opts.vector.isa.vzero(), @(4).val)),

    sse_unary_noneExp := Rule(
	[@(1, [
          vushuffle_16x8i, vushuffle_2x32f, vushuffle_2x64f, vushuffle_2x64i,
	  vushuffle_4x32f, vushuffle_4x32i,
	  chslo_2x32f, chslo_2x64f, chslo_2x64i, chslo_4x32f,
	  chshi_2x32f, chshi_2x64f, chshi_2x64i, chshi_4x32f ]),
	 @(2, noneExp), ...],
	e -> @(2).val),

## SSE_8x16i SSSE3 rules
    vushuffle_8x16i := Rule([@(1, vushuffle_8x16i), @(2, Value, e->ObjId(e.t)=TVect), @(3)],
        e -> Value(@(2).val.t, List(2*[0..@(2).val.t.size-1], i->@(2).val.v[@(3).val.v[i+1].v/2+1]))),

# complex loopcode load operations rules
    SSE_4x32f_load_TComplex := Rule([@@(1, vpack, (e,cx)-> IsBound(cx.opts) and IsBound(cx.opts.vector) and IsBound(cx.opts.vector.isa) and
        cx.opts.vector.isa = SSE_4x32f),
        [@(2,re), nth], @(3, im, e->e.args[1]=@(2).val.args[1]), @(4,re, e->e.args[1]=@(2).val.args[1]), @(5,im, e->e.args[1]=@(2).val.args[1])],
        e->let(dptr := @(2).val.args[1].toPtr(TVect(TReal, 2)),
            vushuffle_4x32f(vload_2l_4x32f(vzero_4x32f(),dptr), [1,2,1,2]))),

    SSE_4x32f_fold_vushuffle := Rule([@(1, vushuffle_4x32f), @(2, vushuffle_4x32f), @(3)],
        e->vushuffle_4x32f(@(2).val.args[1], List(@(1).val.args[2].p, i->@(2).val.args[2].p[i]))),

    SSE_4x32f_fold_vushuffle_vshuffle := Rule([@(1, vushuffle_4x32f), [vshuffle_4x32f, @(2), @(3), @(4)], @(5)],
        e->vshuffle_4x32f(@(2).val, @(3).val, List(@(5).val.p, i->@(4).val.p[i]))),

    SSE_4x32f_load_TVect := Rule([@@(1, vpack, (e,cx)-> IsBound(cx.opts) and IsBound(cx.opts.vector) and IsBound(cx.opts.vector.isa) and
        cx.opts.vector.isa = SSE_4x32f),
        @(2,nth,e->IsBound(e.loc.init)), @(3, nth, e->e.loc=@(2).val.loc), @(4,nth, e->e.loc=@(2).val.loc), @(5,nth, e->e.loc=@(2).val.loc)],
        e->deref(@(2).val.toPtr(TVect(TReal, 4)))),

    # movldup is an SSE3 instruction
    SSE_4x32f_movldup := Rule([@@(1, vushuffle_4x32f,
        (e,cx) -> IsBound(cx.opts) and IsBound(cx.opts.vector) and IsBound(cx.opts.vector.SIMD) and not cx.opts.vector.SIMD in ["SSE2", "SSE", "MMX"]),
        @(2).cond(e->ObjId(e) <> Value), @(3, vparam, e->e.p=[1,1,3,3])],
        e->vldup_4x32f(@(2).val)),

    # movhdup is an SSE3 instruction
    SSE_4x32f_movhdup := Rule([@@(1,vushuffle_4x32f,
        (e,cx) -> IsBound(cx.opts) and IsBound(cx.opts.vector) and IsBound(cx.opts.vector.SIMD) and not cx.opts.vector.SIMD in ["SSE2", "SSE", "MMX"]),
        @(2).cond(e->ObjId(e) <> Value), @(3, vparam, e->e.p=[2,2,4,4])],
        e->vhdup_4x32f(@(2).val)),

    SSE_2x64f_load_TVect := Rule([@@(1, vpack, (e,cx)-> IsBound(cx.opts) and IsBound(cx.opts.vector) and IsBound(cx.opts.vector.isa) and
        cx.opts.vector.isa = SSE_2x64f),
        @(2,nth,e->IsBound(e.loc.init)), @(3, nth, e->e.loc=@(2).val.loc)],
        e->deref(@(2).val.toPtr(TVect(TReal, 2)))),

    SSE_16x8i_add_ddiv2 := ARule(add, [[ddiv, @(1), @(2, Value, x -> x=2)], ..., [@(3, ddiv), @(4),
        @@(5, Value, (x, cx) -> x=2 and _isISA(@(3).val.t, SSE_16x8i, cx.opts) and IsUIntT(@(3).val.t.base_t()))]],
        e -> [average_16x8i(@(1).val, @(4).val)]),
    SSE_16x8i_add_ddiv2_absdiff2 := ARule(add, [[ddiv, @(1, absdiff2), @(2, Value, x -> x=2)], ..., [@(3, ddiv), @(4, absdiff2),
        @@(5, Value, (x, cx) -> x=2 and _isISA(@(3).val.t, SSE_16x8i, cx.opts))]],
        e -> [average_16x8i(@(1).val, @(4).val)]),

    SSE_16x8i_didiv2_add := Rule([@(0, [ddiv, idiv]), [add, @(1), @(2)],
        @@(3, Value, (x, cx) -> x=2 and _isISA(@(0).val.t, SSE_16x8i, cx.opts) and IsUIntT(@(0).val.t.base_t()))],
        e -> [average_16x8i(@(1).val, @(2).val)]),
    SSE_16x8i_didiv2_add_absdiff2 := Rule([@(0, [ddiv, idiv]), [add, @(1, absdiff2), @(2, absdiff2)],
        @@(3, Value, (x, cx) -> x=2 and _isISA(@(0).val.t, SSE_16x8i, cx.opts))],
        e -> [average_16x8i(@(1).val, @(2).val)]),
));

Class(RulesSSEPostProcess, RuleSet);
RewriteRules(RulesSSEPostProcess, rec(
    div_16x := Rule( [ @@(1, [idiv, ddiv], (x, cx) -> IsOrdT(x.t.base_t()) and _isISA(x.t, SSE_16x8i, cx.opts)),
                       @(2), @(3, Value, x -> let( v := Set(Flat([x.v])), Length(v)=1 and 2^CeilLog2(_unwrap(v[1])) = v[1])) ],
        e -> let( v :=  _unwrap(Set(Flat([@(3).val.v]))[1]),
                  bin_and(tcast(e.t, bin_shr(tcast(TVect(T_Int(16), 8), @(2).val), CeilLog2(v))), e.t.value(idiv(255, v))))),

    SSE_absdiff2_to_xor := Rule( @(1, absdiff2, x -> IsOrdT(x.t.base_t())),
        e -> bin_xor(e.args[1], e.args[2])),

    SSE_fixUnalignedLoadStore := Rule(@@(1,[vstoreu2_4x32f, vstoreu_4x32f],(e,cx)->ObjId(e.args[2])=noneExp and cx.opts.fixUnalignedLoadStore = true), e->skip())
));
