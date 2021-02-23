
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# ================================================================
# Integer Strength Reduction Rules
# ================================================================

Class(RulesStrengthReduce, RuleSet);

#F SReduceSimple(<c>) -- applies strength reduction rules to any object <c>
#F                       disables <opts> sensitive rules
SReduceSimple  := c -> ApplyStrategy(c, [RulesStrengthReduce], BUA, rec());

#F SReduce(<c>, <opts>) - applies strength reduction rules to any object <c>
#F
SReduce  := (c,opts) -> ApplyStrategy(c, [RulesStrengthReduce], BUA, opts);

#@1:=@(1);
#@2:=@(2);

@1int := @(1).cond(e -> IsValue(e) and IsInt(e.v));
@2int := @(2).cond(e -> IsValue(e) and IsInt(e.v));
@2ge_var := than@ -> @(2).cond(
    e -> IsValue(e) and IsBound(than@.val.range) and than@.val.range <= e.v);

isNegV := o -> o.v<0;
isNonZero := o -> o <> 0;

@negVal := @.target(Value).cond(isNegV);

@2Teq1 := @(2).cond( x -> x.t = @(1).val.t );

_0noneOrZero := (t) -> When(ObjId(@(0).val)=noneExp, noneExp(t), t.zero());
_sameTasOp := (opname, func) -> @@(0, [Value, noneExp]).cond((e, cx) -> 
    let( op := Last(cx.(opname)), 
        func(e) and ForAny(op.args, (a) -> not Same(a, e) and a.t=op.t)
    )
);

v0:=V(0);
v1:=V(1);

@vpos  := (n) -> @(n, Value, e -> IsPosInt(e.v));
@vpos0 := (n) -> @(n, Value, e -> IsPosInt0(e.v));

RewriteRules(RulesStrengthReduce, rec(

 # FF: SAR-specific rule. Not sure if they should be turned on by default.
 # YSV: below rule is invalid, because idiv(a+b, c) <> idiv(a) + idiv(b)
 #      example: idiv(2+3, 4) == 1     
 #               idiv(2,4) + idiv(3,4) == 0
 # div_sar := Rule([@(1,idiv), @(2, add), @(3,Value) ], e -> ApplyFunc(add, List(@(2).val.args, i->idiv(i, @(3).val)))),

 ################
 # leq 
 # ===========
 leq_single := Rule([leq, @(1)], e-> V_true),
 leq_eq := Rule([leq, @(1), @(2), @(3).cond(e->e=@(1).val)], e -> eq(@(1).val, @(2).val)),

 leq_val_mulindex     := Rule( [leq, @vpos0(1), [mul, @vpos0(2), @(3, [var, param], IsLoopIndex)]],
     e -> leq(idiv(@(1).val + @(2).val - 1, @(2).val), @(3).val)),
 
 leq_val_mulindex_val := Rule( [leq, @vpos0(1), [mul, @vpos(2), @(3, [var, param], IsLoopIndex)], @vpos0(4)],
     e -> leq(idiv(@(1).val + @(2).val - 1, @(2).val), @(3).val, idiv(@(4).val, @(2).val))),
 
 leq_mulindex_val     := Rule( [leq, [mul, @vpos0(2), @(3, [var, param], IsLoopIndex)], @vpos0(4)],
     e -> leq(@(3).val, idiv(@(4).val, @(2).val))),

 leq_index_neg := Rule([leq, @(2,var,IsLoopIndex), @(3, Value, x->x.v<0), ...], e -> V_false), 
 leq_index_big := ARule(leq, [@(2,var,x->IsLoopIndex(x) and IsInt(x.range)), @(3, Value, x->x.v >= @(2).val.range-1)],
     e -> [@(2).val]),

 leq_neg_mul_index := Rule([leq, [mul, @(2, Value, x->x.v>0), @(3,var,IsLoopIndex)], @(4, Value, x->x.v<0), ...], 
     e -> V_false), 

 leq_index_0 := Rule([leq, @(2,var,IsLoopIndex), @(3, Value, x->x.v=0)], e -> eq(@(2).val, @(3).val)), 
 leq_mul_index_0 := Rule([leq, [mul, @(2, Value, x->x.v<>0), @(3,var,IsLoopIndex)], @(4, Value, x->x.v=0)], 
     e -> eq(@(3).val, @(4).val)), 

 leq_neg_index_A := ARule(leq, [@(1, Value, x->x.v<=0), @(2,var,IsLoopIndex)], e -> [@(2).val]),
 leq_neg_mul_index_A := ARule(leq, [@(1, Value, x->x.v<=0), [@(0,mul), @(2, Value, x->x.v>0), @(3,var,IsLoopIndex)]], 
     e -> [@(0).val]),

 leq_val_val_l := Rule([leq, @(1, Value), @(2, Value, x->@(1).val<=x), @, ...],
     e -> ApplyFunc(leq, Drop(e.args, 1))),
 leq_val_val_r := Rule([leq, ..., @, @(1, Value), @(2, Value, x->@(1).val<=x)],
     e -> ApplyFunc(leq, DropLast(e.args, 1))),
     
 ################
 # add/sub/mul 0
 # =============
 subzr := Rule([sub, @1, _sameTasOp("sub", _is0none)], e -> @1.val),
 subzl := Rule([sub, _sameTasOp("sub", _is0none), @1], e -> neg(@1.val)),

 div0 := Rule([@(1, [div, idiv, fdiv]), _0none, @], e->_0noneOrZero(e.t)),
 div1 := Rule([@(1, [div, idiv, fdiv]), @, @@(0).cond((e, cx) -> let( op := Last(cx.(ObjId(@(1).val).__name__)),
     Cond(IsValue(e), isValueOne(e), e=1) and ForAny(op.args, (a) -> not Same(a, e) and a.t=op.t)))], 
     e -> @.val),

 add_assoc  := ARule(add, [ @(1,add) ], e -> @1.val.args),
 add_zero   := ARule(add, [_sameTasOp("add", _is0none)], e -> [ ]),
 add_single := Rule([add, @1],          e -> @1.val),

 add_cfold := Rule(@(1,add,e->Length(e.args)>2 and Length(Filtered(e.args, IsValue)) >= 2),
     e -> let(s := SplitBy(e.args, IsValue), values := s[1], exps := s[2],
          ApplyFunc(add, Concatenation([ApplyFunc(add,values)], exps)))),

 sub_sub := Rule( [@(1, sub), @(2), [sub, @(3), @(4)]], e -> sub(add(@(2).val, @(4).val), @(3).val)),

 ################
 # logic and/or
 # ============
 logic_single := Rule([@(1, [logic_and, logic_or]), @1], e->@1.val),
 logic_empty := Rule([@(1, [logic_and, logic_or])], e->V_true),

 logic_and_assoc := ARule(logic_and, [ @(1,logic_and) ], e -> @1.val.args),
 logic_and_false := Rule([logic_and, ..., _vfalse, ...],  e -> V_false),
 logic_and_true := ARule(logic_and, [_vtrue],  e -> [ ]),

 logic_or_assoc := ARule(logic_or, [ @(1,logic_or) ], e -> @1.val.args),
 logic_or_true := Rule([logic_or, ..., _vtrue, ...],  e -> V_true),
 logic_or_false := ARule(logic_or, [_vfalse],  e -> [ ]),
 
 logic_neg_neg := Rule([logic_neg, [logic_neg, @(1)]], e -> @(1).val),
 logic_neg_eq := Rule([logic_neg, [eq, @(1), @(2)]], e -> neq(@(1).val, @(2).val)),

 # 'maybe' is a "magic" boolean function that behaves as 'true' inside 'and'/'or' operators,
 # but also satisfies the uncertainty rule logic_not(maybe()) = maybe()
 # 
 logic_neg_maybe := Rule([logic_neg, @(1, maybe)], e -> @(1).val),
 logic_and_maybe := ARule(logic_and, [maybe], e -> [ ]),
 logic_or_maybe := ARule(logic_or, [maybe], e -> [ ]),

 eq_logic_true := Rule( [eq, @(1), _vtrue], e -> @(1).val ),
  
 ################
 # mul
 # ===

 # mul_assoc is slow, and we enable it in RulesExpensiveStrength reduce
 #   mul_assoc := ARule(mul, [ @(1,mul) ], e -> @1.val.args),
 mul_one := ARule(mul, [ _sameTasOp("mul", _is1) ],  e -> [ ]),

 mul_single := Rule([mul, @1], e->@1.val),
 mul_zero := Rule([mul, ..., _v0none, ...], e -> _0noneOrZero(e.t)),  # must be @(0).val so that none*x -> none
 mul_mul := Rule([mul, @(1,Value), [mul, @(2,Value), @(3)]], e -> let(
     t := UnifyPair(@(1).val.t, @(2).val.t),
     val := t.value(t.product(@(1).val.v, @(2).val.v)),
     mul(val, @(3).val))),

 # apply distributivity a * (b+c) -> a*b + a*c, only if inside nth()
 mul_add_nth := Rule([@@(0,mul,(e,cx)->IsBound(cx.nth) and cx.nth<>[]), @(1), @(2,add)],
     e -> ApplyFunc(add, List(@(2).val.args, a->@(1).val * a))),

 # FF: NOTE: Need to guard this rule in the SAR case. Not sure how to properly do this.
 # Guard against nth indexing integer tables. This is at the moment only used in SAR.
 # NB: Without guard, there is an infinite loop with sar_sub rule
 mul_sub_nth := Rule([@@(0, mul, (e,cx) -> cx.isInside(nth) and cx.nth[1].loc.t.t <> TInt), @(1), @(2,sub)],
     e -> ApplyFunc(sub, List(@(2).val.args, a->@(1).val * a))),  # Above guard to prevent inf loop with sar_sub

 # NB: Below rule became obsolete due to change in segment length computation
 # sar_sub := Rule([sub, @(1,Value,e->e.t=TInt), [mul, @(2, Value, e->e.t=TInt and IsInt(@(1).val.v/e.v)), @(3)]],
 #    e->mul(@(2).val, sub(@(1).val/@(2).val, @(3).val))),

 # Value * (Value * a +/- Value * b) -> Value * a +/- Value * b
 mul_add_mul := Rule([mul, @(1, Value), [@(0,[add,sub]), [mul, @(2, Value), @(3)], [mul, @(4, Value), @(5)]]],
     e -> let(t1 := UnifyPair(@(1).val.t, @(2).val.t),
              t2 := UnifyPair(@(1).val.t, @(4).val.t),
              op := ObjId(@(0).val),
              op(t1.value(t1.product(@(1).val.v, @(2).val.v)) * @(3).val,
                 t2.value(t2.product(@(1).val.v, @(4).val.v)) * @(5).val))),

 #  mul_cfold := Rule(@(1,mul,e->Length(e.args)>2 and Length(Filtered(e.args, IsValue)) >= 2),
 #   e -> let(s := SplitBy(e.args, IsValue), values := s[1], vars := s[2],
 #            ApplyFunc(mul, Concatenation( [Product(values,v->v.v)], vars)))),

 fpmul_2power := Rule([fpmul, @(1, Value), @(2, Value, e->Is2Power(e.v)), @(3)], e ->
    let(sh := Log2Int(@(2).val.v),
        Cond(sh = @(1).val.v, @(3).val,
             sh > @(1).val.v, arith_shl(@(3).val, sh - @(1).val.v),
             sh < @(1).val.v, arith_shr(@(3).val, @(1).val.v - sh)))),

 # mul_add_expand := Rule([mul, @(1,Value,e->e.t=TInt), @(2,add)],
 #    e -> ApplyFunc(add, List(@2.val.args, a->mul(@1.val, a).eval()))),
 #
 # mul_sub_expand := Rule([mul, @(1,Value,e->e.t=TInt), @(2,sub)],
 #    e -> ApplyFunc(sub, List(@2.val.args, a->mul(@1.val, a).eval()))),


 ##############
 # neg
 # =====
 mul_negone := Rule([mul, _neg1, @2], e -> neg(@2.val)),
 neg_neg := Rule([neg, [neg, @1]], e -> @1.val), # -(-1) -> 1
 neg_sub := Rule([neg, [sub, @1, @2]], e -> sub(@2.val, @1.val)), # -(1-2) -> 2-1
 neg_zero := Rule([neg, _v0none], e -> e.args[1]),

 sub_neg1 := Rule([sub, @1, [neg, @2]], e -> add(@1.val, @2.val)),  # 1-(-2) -> 1+2
 sub_neg2 := Rule([sub, [neg, @1], @2], e -> neg(add(@1.val, @2.val))), # (-1)-2 -> -(1+2)
 add_neg1 := Rule([add, @1, [neg, @2]], e -> sub(@1.val, @2.val)), # 1+(-2) -> 1-2
 add_neg2 := Rule([add, [neg, @1], @2], e -> sub(@2.val, @1.val)), # (-1)+2 -> 2-1

 mul_neg1 := Rule([mul, @1, [neg, @2]], e -> neg(mul(@1.val, @2.val))), # 1*(-2) -> -(1*2)
 mul_neg2 := Rule([mul, [neg, @1], @2], e -> neg(mul(@1.val, @2.val))), # (-1)*2 -> -(1*2)
 mul_negc1 := Rule([mul, @1, @(2,Value,isNegV)], e -> neg(mul(Value.new(@2.val.t, -@2.val.v), @1.val))),
 mul_negc2 := Rule([mul, @(1,Value,isNegV), @2], e -> neg(mul(Value.new(@1.val.t, -@1.val.v), @2.val))),


 #################################
 # general constant folding
 # ==========================

 # The rules below can be used instead of generic cfold to speed up this ruleset
 # cfold_add  := Rule([add, Value,Value], e->Value.new(e.args[1].t, e.args[1].v + e.args[2].v)),
 # cfold_mul  := Rule([mul, Value,Value], e->Value.new(e.args[1].t, e.args[1].v * e.args[2].v)),
 # cfold_sub  := Rule([sub, Value,Value], e->Value.new(e.args[1].t, e.args[1].v - e.args[2].v)),
 # cfold_imod := Rule([imod, Value,Value], e->Value.new(e.args[1].t, e.args[1].v mod e.args[2].v)),
 # cfold_idiv := Rule([idiv, Value,Value], e->Value.new(e.args[1].t, QuoInt(e.args[1].v, e.args[2].v))),
 # cfold_nth  := Rule([nth, @(1,var,e->IsBound(e.value)), Value], e->e.loc.value.v[e.idx.v+1]),
 # cfold_nthv := Rule([nth, Value,Value],                         e->e.loc.v[e.idx.v+1]),
 # cfold_re   := Rule([re, Value], e->Value.new(TDouble, ReComplex(Complex(e.args[1].v)))),
 # cfold_im   := Rule([im, Value], e->Value.new(TDouble, ImComplex(Complex(e.args[1].v)))),
 # cfold_neg  := Rule([neg, Value], e->Value.new(e.args[1].t, -e.args[1].v)),

 # cfold_arith := Rule([@(1,[add,mul,sub]), Value,Value], e->e.eval()),
 # cfold_divmod := Rule([@(1,[imod,idiv]), Value,Value], e->e.eval()),
 # cfold_unary := Rule([@(1,[re,im,neg]), Value], e->e.eval()),
 # cfold_nth := Rule([nth, @(1,var,e->IsBound(e.value)), Value], e->e.eval()),
 # cfold_nthv := Rule([nth, Value, Value], e->e.eval()),

 cfold := Rule(@(1).cond(e -> IsExp(e) and e.can_fold()), e -> e.eval()),

 # In below rules @(1) is a type, and if t is a type, 
 # t.value creates a value of this type, and thus performs the typecast 
 tcast_varvalue := Rule([tcast, @(1), @(2,var,e->IsBound(e.value))], 
     e -> Cond(IsPtrT(@(1).val), @(2).val, @(1).val.value(@(2).val.v))), # NOTE: IsPtrT branch is unsafe, and relies on correct
                                                                         #        scalarization/non-scalarization
# tcast_value    := Rule([tcast, @(1), @(2,Value)], e -> Cond(
#     # if an array value is typecast to a pointer of same type, just drop the typecast
#     IsPtrT(@(1).val) and IsArrayT(@(2).val.t) and @(1).val.t = @(2).val.t.t, @(2).val, 
#     @(1).val.value(@(2).val.v))),
 
 # only applies to pointers, otherwise is not valid -- for example (double)((int)3.2) is not (double)3.2
 tcast_tcast := Rule([tcast, @(1).cond(t->IsPtrT(t)), [tcast, @(2).cond(t->IsPtrT(t)), @(3)]], 
     e -> tcast(@(1).val, @(3).val)), 

 # allow adding zero to ptrs.
 tptr_addzero := ARule(add, [@@(0, Value).cond( (e, cx) -> 
    let( op := Last(cx.("add")),
        _is0none(e) and ObjId(op.t) = TPtr
    ))],
    ee -> []
 ),   

 # cfold_vref := Rule(@(1).cond(e -> IsRec(e) and IsBound(e.isVref) and e.isVref and e.canScalarize() and not IsBound(e.didEval)),
 #     e -> e.eval()),

 # cfold_vref := Rule(@(1).cond(e -> IsRec(e) and IsBound(e.isVref) and e.isVref and e.canScalarize() and not IsBound(e.didEval)),
 #     e -> e.eval()),

 ##################
 # cond rules
 # ============
 cond_rid := Rule([cond, @(1), @(2), @(3).cond(e->e=@(2).val)], e -> @(2).val),

 cond_fold := Rule([cond, @(1,Value), @2, ...], e -> Cond(
         @1.val.v = false or @1.val.v = 0,
             When(Length(e.args) = 3, e.args[3], ApplyFunc(cond, Drop(e.args, 2))),
         @1.val.v = true or IsInt(@1.val.v),
             (@2.val),
         Error("non boolean condition"))),

 IF_fold := Rule([IF, @(1,Value), @(2), @(3)], e -> Cond(
         @(1).val.v = false or @1.val.v = 0,
             @(3).val,
         @(1).val.v = true or IsInt(@1.val.v),
             @(2).val,
         Error("non boolean condition"))),

 IF_skip := Rule([IF, @(1), skip, skip], e -> skip()),

 multi_if2 := Rule([multi_if, @(1), @(2)], e -> IF(@(1).val, @(2).val, skip())),
 multi_if3 := Rule([multi_if, @(1), @(2), @(3)], e -> IF(@(1).val, @(2).val, @(3).val)),
 multi_if_skip := Rule([multi_if, skip], e->skip()),
 multi_if_skip_skip := Rule([@(1, multi_if), @(2, e->ObjId(Last(@(1).val.rChildren()))=skip), skip, ...],
     e -> ApplyFunc(multi_if, Drop(@(1).val.rChildren(), 2))),
 #NOTE: code in autolib's _genPlan messing with 'args' directly, have to drop empty multi_if even though it could fold automatically
 multi_if_drop := ARule( chain, [[multi_if]], e -> [] ),


 ###################
 # chain rules
 # =============
 chain_empty := Rule([chain], e-> skip()),

 chain_skip := Rule([chain, skip], e-> skip()),


 ##########################
 # rules for functions
 # =====================
 frotate_rank1 := Rule([frotate, @(1).cond(e->e.rank()<=1)], e -> @(1).val),
 fcall_fexp  := Rule(
     [@(0,fcall), 
	 @(1,[fcurry,flift,fsplit,frotate,Lambda], e -> e.rank()+2=Length(@(0).val.args)), 
	 ...], 
     e -> e.eval()),

 fexp_Lambda := Rule(
     [@(1,[fcurry,flift,fsplit,frotate]), 
	 Lambda, 
	 ...], 
     e -> e.eval()),

 fcurry_void := Rule(
     [fcurry,
	 @(1), 
	 @(2).cond(x -> IsValue(x) and ObjId(@(1).val.t)=TFunc and @(1).val.t.params[x.v] in [TVoid, TDummy]), 
         @(3).cond(x -> x<>0)], 
     x -> fcurry(@(1).val, @(2).val, 0)),

 fcall_void := Rule(
     [@(0,fcall), 
	 @(1).cond(x -> ObjId(x.t)=TFunc and
	                let(pos := Position(x.t.params, TDummy), 
			    pos <> false and
			    Length(@(0).val.args) >= 1+pos and
			    not IsValue(@(0).val.args[1+pos]))),
	 ...],
     e -> let(types := e.args[1].t.params,
	      len := Length(e.args),
	      ApplyFunc(fcall, [e.args[1]] :: List([2..len], i -> Cond(types[i-1]=TDummy, 0, e.args[i]))))),

 # if lambdaWrap(Lambda(..)) has Lambda of the form x -> exp, where exp does not depend on x, then eliminate the Lambda
 #lambdaWrap_dummyLambda := Rule([lambdaWrap, @(1, Lambda, x->x.t.params[1]=TDummy and Length(x.t.params)=2)], e -> @(1).val.at(0)),

 #lambdaWrap_rank0 := Rule([lambdaWrap, @(1).cond(x->x.rank()=0)], e -> fcall(@(1).val, 0)), 


 ###############
 # omega
 # =======
 re_omega := Rule([re, [omega, @(1), @(2)]], e -> cospi(fdiv(2*@(2).val, @(1).val))),
 im_omega := Rule([im, [omega, @(1), @(2)]], e -> sinpi(fdiv(2*@(2).val, @(1).val))),

 re_omegapi := Rule([re, [omegapi, @(1)]], e -> cospi(@(1).val)),
 im_omegapi := Rule([im, [omegapi, @(1)]], e -> sinpi(@(1).val)),

 re_conj_omega := Rule([re, [conj, [omega, @(1), @(2)]]], e -> cospi(fdiv(2*@(2).val, @(1).val))),
 im_conj_omega := Rule([im, [conj, [omega, @(1), @(2)]]], e -> -sinpi(fdiv(2*@(2).val, @(1).val))),


 ######################
 # complex arith
 # ===============
 conj_conj := Rule([conj, [conj, @(1)]], e->@(1).val),
 re_conj := Rule([re, [conj, @(1)]], x->re(@(1).val)),
 im_conj := Rule([im, [conj, @(1)]], x->-im(@(1).val)),

 # the re_cond/im_cond rules prevent cond from having different types in different branches (ie TReal vs TComplex)
 # 
 re_cond := Rule([re, @(1,cond)], x -> ApplyFunc(cond, 
	 List(@(1).val.args, x->Cond(x.t=TBool, x, re(x))))),
 im_cond := Rule([im, @(1,cond)], x -> ApplyFunc(cond, 
	 List(@(1).val.args, x->Cond(x.t=TBool, x, im(x))))),

 re_real := Rule([re, @.cond(e->e.t=TReal or ObjId(e.t)=T_Real)], e -> e.args[1]), 
 im_real := Rule([im, @.cond(e->e.t=TReal or ObjId(e.t)=T_Real)], e -> e.t.zero()), 

 re_mul := Rule([re, [@(1, mul), @(2).cond(e->e.t=TReal), ...]], e -> @(2).val * re(ApplyFunc(mul, Drop(@(1).val.args, 1)))),
 im_mul := Rule([im, [@(1, mul), @(2).cond(e->e.t=TReal), ...]], e -> @(2).val * im(ApplyFunc(mul, Drop(@(1).val.args, 1)))),

 re_cxpack := Rule([re, [cxpack, @(1), @]], e -> @(1).val),
 im_cxpack := Rule([im, [cxpack, @, @(1)]], e -> @(1).val),

 re_vdup := Rule([re, [vdup, @(1), @(2)]], e -> vdup(re(@(1).val), @(2).val)),
 im_vdup := Rule([im, [vdup, @(1), @(2)]], e -> vdup(im(@(1).val), @(2).val)),

 mul_cxpack := ARule( mul, [[cxpack, @(1), @(2)], [cxpack, @(3), @(4)]],
     e -> [cxpack( @(1).val*@(3).val - @(2).val*@(4).val,
                   @(1).val*@(4).val + @(2).val*@(3).val )]),

 add_cxpack := ARule( add, [[cxpack, @(1), @(2)], [cxpack, @(3), @(4)]],
     e -> [cxpack( add(@(1).val, @(3).val), add(@(2).val, @(4).val) )]),

 sub_cxpack := Rule( [sub, [cxpack, @(1), @(2)], [cxpack, @(3), @(4)]],
     e -> [cxpack( sub(@(1).val, @(3).val), sub(@(2).val, @(4).val) )]),

 # re( (a + i*b) * (a - i*b) ) -> a*a + b*b
 re_cxmulconj := Rule( [re, [mul, [cxpack, @(1), @(2)], [conj, [cxpack, @(3).cond(x->x=@(1).val), @(4).cond(x->x=@(2).val)]]]],
     e -> add(mul(@(1).val, @(1).val), mul(@(2).val, @(2).val))),
 # im( (a + i*b) * (a - i*b) ) -> 0
 im_cxmulconj := Rule( [im, [mul, [cxpack, @(1), @(2)], [conj, [cxpack, @(3).cond(x->x=@(1).val), @(4).cond(x->x=@(2).val)]]]],
     e -> e.t.zero() ),

 #############
 # imod
 # ======
 imod_by_one := Rule([imod, @1, _1], e -> v0),
 imod_zero := Rule([imod, _0none, @], e -> @(0).val),
 imod_one := Rule([imod, _1, @], e -> v1),

 imod_imod := Rule([imod, [imod, @(1,Value), @(2)], @(3,Value, e->@(2).val.v <= e.v )],
     e -> imod(@(1).val, @(2).val)),

 imod_small := Rule([imod, @(1,var,e->IsBound(e.range) and not IsSymbolic(e.range)),  @(2,Value,e->e.v >= @(1).val.range)],
     e -> @(1).val),

 # this is a less general version of imod_mul_reduce_const, commented out below
 # v1*exp mod v3 -> ((v1 mod v3)*exp) mod v3
 imod_mul := Rule([imod, [@(2, mul), @(1,Value), ...], @(3,Value, e->@(1).val.v >= e.v)],
     e -> imod(ApplyFunc(mul, [@(1).val.v mod @(3).val.v] :: Drop(@(2).val.args, 1)), @(3).val)),

 # (v1*exp + exp4) mod v3 -> ((v1 mod v3)*exp + exp4) mod v3
 imod_addmul_l := Rule([imod, [add, [@(2,mul), @(1,Value), ...], @(4)], @(3,Value, e->@(1).val.v >= e.v)],
     e -> imod(ApplyFunc(mul, [@(1).val.v mod @(3).val.v] :: Drop(@(2).val.args, 1)) + @(4).val, @(3).val)),
 imod_addmul_r := Rule([imod, [add, @(4), [@(2,mul), @(1,Value), ...]], @(3,Value, e->@(1).val.v >= e.v)],
     e -> imod(ApplyFunc(mul, [@(1).val.v mod @(3).val.v] :: Drop(@(2).val.args, 1)) + @(4).val, @(3).val)),

 no_mod_imod := Rule([no_mod, [imod, @1, @2]], e -> @1.val),
 no_mod_small_mod := Rule([no_mod, [small_mod, @1, @2]], e -> @1.val),

 # imod_mul_reduce_const := Rule( [imod, @(1,mul),
 #        @(2,Value, e -> Filtered(@1.val.args, a->IsValue(a) and a.v >= e.v) <>[]) ],
 #    e -> imod(
 #      ApplyFunc(mul, List(@1.val.args, a->Cond(IsValue(a), V(a.v mod @2.val.v), a))),
 #      @2.val)),

 # NOTE: These rules are very general but lead to an infinite loop.
 #
 # imod_mul := Rule([imod, @(1, mul, NOT_ALL_ARGS_ARE(imod)), @(2)],
 #     e -> imod(ApplyFunc(mul, List(@(1).val.args, a -> imod(a, @(2).val))), @(2).val)),
 #
 # imod_add := Rule([imod, @(1, [add,sub], NOT_ALL_ARGS_ARE(imod)), @(2)],
 #     e -> imod(ApplyFunc(ObjId(@(1).val), List(@(1).val.args, a -> imod(a, @(2).val))), @(2).val)),


 #############
 # idiv
 # ======
 idiv_small := Rule([idiv, @(1, var, e -> IsBound(e.range) and not IsSymbolic(e.range)), @(2, Value, e-> e.v >= @(1).val.range)],
     e -> V(0)),

 # exp / value ->  (1/value) * exp,  for floating-point types.  NOTE: use .t.isFloat()
 div_byV_tomul := Rule([@(0, [div, fdiv]), @(1), @(2, Value, x->x.t in [TReal, TComplex])],
     e -> div(@(2).val.t.one(), @(2).val) * @(1).val),

 # (v1*exp2 + exp3)/v4 -> v1/v4*exp2 + exp3/v4; v4 | v1;
 # NOTE: these rules are invalid if sign(exp2)<>sign(exp4) !!!
  
 #idiv_addmul_l := Rule([idiv, [add,  [mul, @(1, Value), @(2)], @(3)], @(4, Value, x -> _divides(x, @(1).val))],
 #    e -> add(mul(div(@(1).val, @(4).val), @(2).val), idiv(@(3).val, @(4).val))),
 
 #idiv_addmul_r := Rule([idiv, [add,  @(3), [mul, @(1, Value), @(2)]], @(4, Value, x -> _divides(x, @(1).val))],
 #    e -> add(mul(div(@(1).val, @(4).val), @(2).val), idiv(@(3).val, @(4).val))),

 # (v1*loop_idx + v2)/v3 -> v1/v3*loop_idx + v2/v3;  when  (v3 | v1) & (v2 >= 0) & (v1 >= 0);
 # assumption: loop index cannot be negative 
 idiv_addmul_pos_l := Rule([idiv, [add,  [mul, @vpos0(1), @(2).cond(IsLoopIndex)], @vpos0(3)], @(4, Value, x -> _divides(x, @(1).val))],
     e -> add(mul(idiv(@(1).val, @(4).val), @(2).val), idiv(@(3).val, @(4).val))),
 
 idiv_addmul_pos_r := Rule([idiv, [add,  @vpos0(3), [mul, @vpos0(1), @(2).cond(IsLoopIndex)]], @(4, Value, x -> _divides(x, @(1).val))],
     e -> add(mul(idiv(@(1).val, @(4).val), @(2).val), idiv(@(3).val, @(4).val))),

 ############
 # pow, sqrt, rsqrt
 # =====
 pow_zero := Rule([pow, @1, _0none], e->v1),

 powmod_mul_const := Rule([powmod, @, Value, @(1,mul,e->ForAny(e.args,IsValue)), @],
    e -> let(split := SplitBy(@1.val.args, IsValue),
         value := Product(split[1], x->x.v),
         vars := split[2],
         powmod( e.args[1],
            (e.args[2].v ^ value) mod e.args[4].v,
                 ApplyFunc(mul, vars),
             e.args[4]))),

 powmod_cfold := Rule([powmod, @1, Value, Value, Value],
    e -> imod(((e.args[2].v ^ e.args[3].v) mod e.args[4].v)*e.args[1], e.args[4].v)),

 sqrt_pow2 := Rule( [sqrt, [pow, @(1), _2]], e -> abs(@(1).val) ),
 pow2_sqrt := Rule( [pow, [sqrt, @(1)], _2], e -> @(1).val ),

 rsqrt_pow2 := Rule( [rsqrt, [pow, @(1), _2]], e -> fdiv(e.t.one(), abs(@(1).val)) ),
 pow2_rsqrt := Rule( [pow, [rsqrt, @(1)], _2], e -> fdiv(e.t.one(), @(1).val) ),
 
 #############
 # bin
 # ====

 bin_xor_zero      := Rule([bin_xor, ..., _0none, ...], e -> When(@(0).val = e.args[2], e.args[1], e.args[2])),
 bin_and_zero      := Rule([bin_and, ..., _0none, ...], e -> @(0).val),

 bin_shr_zero      := Rule([bin_shr, _0none,      @], e -> e.args[1]),
 bin_shr_zeroshift := Rule([bin_shr,      @, _0none], e -> e.args[1]),
 bin_shl_zero      := Rule([bin_shl, _0none,      @], e -> e.args[1]),
 bin_shl_zeroshift := Rule([bin_shl,      @, _0none], e -> e.args[1]),

 absdiff2_to_abssub := Rule([@(0, absdiff2, x -> not IsOrdT(x.t.base_t())), @(1), @(2)],
     e -> Cond( IsValue(@(1).val) and isValueZero(@(1).val),
                    @(2).val,
                IsValue(@(1).val) and Length(Set(Flat([@(1).val.v])))=1,
                    sub(@(1).val, @(2).val),
                # else
                    abs(sub(@(1).val, @(2).val)))),

 # Memory allocation
 allocate_0 := Rule([allocate, @(1), [TArray, @(2), _0]], e -> assign(@(1).val, null())),
 deallocate_0 := Rule([deallocate, @(1), [TArray, @(2), _0]], e -> skip()),

 # funcExp rule

 expr_funcExp := Rule( [@(0, [mul, add, sub, div, idiv, imod]), ..., @(1, funcExp), ...], 
     e -> @(1).val ),

 expr_cond_funcExp := Rule( [@(1, [mul, add]), ..., [@(2, cond), ..., funcExp, ...], ...], 
     e -> let( args := RemoveList(e.args, @(2).val), 
         _map_cond(@(2).val, p -> p, exp -> ApplyFunc(ObjId(e), args :: [exp])))),

 divmod_cond_funcExp := Rule( [@(1, [div, idiv, imod]), [@(2, cond), ..., funcExp, ...], @(3)], 
     e -> _map_cond(@(2).val, p -> p, exp -> ObjId(e)(exp, @(3).val))),

 # always pull cond() from nth() for nth(funcExp) annihilation.
 nth_cond := Rule( [nth, @(1), @(2, cond)], 
     e -> _map_cond(@(2).val, p->p, exp->nth(@(1).val, exp))),


 #############
 vdup_noneExp := Rule([vdup, [noneExp, @(1)], @(2)], e -> vdup(@(1).val.zero(), @(2).val) ),

 vdup_vdup := Rule([vdup, [vdup, @(1), @(2)], @(3)], e -> vdup(@(1).val, @(2).val * @(3).val)),

 vdup_vpack := Rule([vdup, @(1, vpack), @(3)], e -> ApplyFunc(vpack, ConcatList(@(1).val.args, x -> [x,x])))
));

Class(stickyNeg, neg, rec(doPeel := false, unparse := "neg"));

Class(RulesExpensiveStrengthReduce, RuleSet);

__groupSummands := function(lst, is_full)
    local e, grp, non, grpid, id, mult, i, one, ptrs;
    [grp, grpid, non] := [rec(), Set([]), []];
    [ptrs, lst] := SplitBy(lst, e -> IsPtrT(e.t) or IsArrayT(e.t));
    for e in lst do
        one := e.t.one();
        if ObjId(e) in [var, param] then
	    if not IsBound(grp.(e.id)) then  grp.(e.id) := [one, e];
	    else 		             grp.(e.id)[1] := grp.(e.id)[1] + one;
	    fi;
	    AddSet(grpid, e.id);
        elif ObjId(e) in [neg, stickyNeg] and ObjId(e.args[1]) in [var, param] then
	    id := e.args[1].id;
	    if not IsBound(grp.(id)) then  grp.(id) := [-one, e.args[1]];
	    else 		           grp.(id)[1] := grp.(id)[1] - one;
	    fi;
	    AddSet(grpid, id);
        elif ObjId(e)=mul and Length(e.args)=2 and ObjId(e.args[1])=Value and ObjId(e.args[2]) in [var, param] then
	    id := e.args[2].id;
	    if not IsBound(grp.(id)) then  grp.(id) := [e.args[1], e.args[2]];
	    else 		           grp.(id)[1] := grp.(id)[1] + e.args[1];
	    fi;
	    AddSet(grpid, id);
	elif ObjId(e) in [ neg, stickyNeg ] and ObjId(e.args[1]) = mul and Length(e.args[1].args) = 2
               and ObjId(e.args[1].args[1]) = Value and ObjId(e.args[1].args[2]) in [ var, param ]  then
            id := e.args[1].args[2].id;
            if not IsBound(grp.(id))  then grp.(id) := [ -e.args[1].args[1], e.args[1].args[2] ];
            else                           grp.(id)[1] := grp.(id)[1] - e.args[1].args[1];
            fi;
            AddSet(grpid, id);
        elif not is_full then 
	    Add(non, [one, e]);
	else
	    mult := one;
	    if ObjId(e) in [neg, stickyNeg] then mult := -mult; e := e.args[1]; fi;
	    if ObjId(e) = mul and IsValue(e.args[1]) then mult := mult*e.args[1]; e := ApplyFunc(mul, Drop(e.args, 1)); fi;
	    if ObjId(e) in [neg, stickyNeg] then mult := -mult; e := e.args[1]; fi;

	    i := PositionProperty(non, x->x[2]=e);
	    if i=false then Add(non, [mult, e]);
	    else 
		non[i][1] := non[i][1] + mult;
	    fi;
	fi;
    od;
    return ApplyFunc(add, ptrs :: List(grpid, id -> grp.(id)[1] * grp.(id)[2]) :: List(non, x->x[1]*x[2])); 
end;

#F __groupSummandsVar(<lst>) is similar to __groupSummands but pulls out
#F variables. This helps to implement rolling pointers in Autolib 
#F because grouping this way helps to get stride expression for a loop index variable.
#F

__groupSummandsVar := function(lst)
    local e, grp, non, grpid, id, mult, i, one, ptrs;
    [grp, grpid, non] := [rec(), Set([]), []];
    [ptrs, lst] := SplitBy(lst, e -> IsPtrT(e.t) or IsArrayT(e.t));
    for e in lst do
        one := e.t.one();
        if ObjId(e) in [var, param] then
	    if not IsBound(grp.(e.id)) then  grp.(e.id) := [one, e];
	    else 		             grp.(e.id)[1] := grp.(e.id)[1] + one;
	    fi;
	    AddSet(grpid, e.id);
        elif ObjId(e) in [neg, stickyNeg] and ObjId(e.args[1]) in [var, param] then
	    id := e.args[1].id;
	    if not IsBound(grp.(id)) then  grp.(id) := [-one, e.args[1]];
	    else 		           grp.(id)[1] := grp.(id)[1] - one;
	    fi;
	    AddSet(grpid, id);
        elif ObjId(e)=mul and ObjId(Last(e.args))=var then
	    id := Last(e.args).id;
	    if not IsBound(grp.(id)) then  grp.(id) := [ApplyFunc(mul, DropLast(e.args, 1)), Last(e.args)];
	    else 		           grp.(id)[1] := grp.(id)[1] + ApplyFunc(mul, DropLast(e.args, 1));
	    fi;
	    AddSet(grpid, id);
	elif ObjId(e) in [ neg, stickyNeg ] and ObjId(e.args[1]) = mul
               and ObjId(Last(e.args[1].args)) = var then
            id := Last(e.args[1].args).id;
            if not IsBound(grp.(id))  then grp.(id) := [ -ApplyFunc(mul, DropLast(e.args[1].args, 1)), Last(e.args[1].args) ];
            else                           grp.(id)[1] := grp.(id)[1] - ApplyFunc(mul, DropLast(e.args[1].args, 1));
            fi;
            AddSet(grpid, id);
        else 
	    Add(non, [one, e]);
	fi;
    od;
    return ApplyFunc(add, ptrs :: List(grpid, id -> grp.(id)[1] * grp.(id)[2]) :: List(non, x->x[1]*x[2])); 
end;

_groupSummands := lst -> __groupSummands(lst, true);
_groupSummandsSimple := lst -> __groupSummands(lst, false);


RewriteRules(RulesExpensiveStrengthReduce, rec(
  mul_add := Rule([mul, @(1,Value), [add, @(2, Value), @(3)]], e -> @(1).val*@(2).val + @(1).val*@(3).val),

  mul_assoc := ARule(mul, [ @(1,mul) ], e -> @1.val.args),

  mul_cfold := Rule(@(1,mul,e->Length(e.args)>2 and Length(Filtered(e.args, IsValue)) >= 2),
   e -> let(s := SplitBy(e.args, IsValue), values := s[1], exps := s[2],
            ApplyFunc(mul, Concatenation( [Product(values,v->v.v)], exps)))),

  add_assoc := Rule([add, ..., @(1,add), ... ], e -> let(
	  args := ConcatList(e.args, x->Cond(ObjId(x)=add, x.args, [x])),
	  #args)),
	  _groupSummands(args))), 

  add_cfold := Rule(@(1,add,e->Length(e.args)>2 and Length(Filtered(e.args, IsValue)) >= 2),
        e -> let(s := SplitBy(e.args, IsValue), values := s[1], exps := _groupSummands(s[2]), 
            ApplyFunc(add, Concatenation([Sum(values,v->v.v)], [exps])))),

  add_sub_val := Rule( [add, ..., [@(1, sub), @(2), @(3, Value)], ...], 
      e -> ApplyFunc(add, [@(2).val, neg(@(3).val)] :: Filtered(e.args, x -> not Same(x, @(1).val)))),
  sub_add_val := Rule( [@(1, sub), @(2, add), @(3, Value)], 
      e -> ApplyFunc(add, [neg(@(3).val)] :: @(2).val.args)),

  # super ugly:  ((a+b+c)-b) -> a+c
  # group summands could do this
  sub_add_same_exp := Rule( [sub, @(1, add), @(2).cond(x -> x in @(1).val.args)],
      e -> let( a := SplitBy(@(1).val.args, x -> x=@(2).val),
                ApplyFunc(add, a[2] :: Drop(a[1], 1) :: [a[1][1].t.zero()]))),
  add_sub_same_exp := Rule( [@(1, add), ..., [@(2, sub), @(3), @(4).cond(x -> x in @(1).val.args)], ...],
      e -> let( a := SplitBy(@(1).val.args, x -> x=@(4).val),
                b := SplitBy(a[2], x -> x=@(2).val),
                ApplyFunc(add, b[2] :: Drop(a[1], 1) :: Drop(b[1], 1) :: [@(3).val, a[1][1].t.zero()]))),

# fold the constants in an xor expression with >2 arguments of which >=2 arguments are constants
  xor_cfold := Rule(@(1,xor,e -> Length(e.args)>2 and Length(Filtered(e.args, IsValue))>=2),
    e -> let(
      s := SplitBy(e.args, IsValue),
      values := s[1],
      vars := s[2],

      ApplyFunc(xor, Concatenation( [Xor(values, v -> v.v)], vars ) )
    )
  ),

  xor_zero := ARule(xor, [_0none], e -> []),

  xor_one := Rule(@(1,xor, e -> Length(e.args) = 1),
    e -> e.args[1]
  ),

# NOTE: removed <idiv> since the rule is invalid for non-divisible numbers
#        can one create a new rule for idiv?

   mul_div := Rule([mul, ..., @(1, div), ...], e ->
       let(facs := e.args, divkind := ObjId(@(1).val), dd := Filtered(facs, f->ObjId(f)=divkind),
       divkind(Product(List(facs, f->When(ObjId(f)=divkind, f.args[1], f))),
           Product(List(dd, d->d.args[2]))))),

   imod_imod := Rule([imod, [imod, @(1), @(2, Value)], @(3, Value, e->@(2).val.v mod e.v = 0)],
       e -> imod(@(1).val, @(3).val)),

   imod_addmul := Rule([imod,[@(1,add),...,[@(2,mul),...,@(3,Value),...],...],@(4,Value,e->Mod(@(3).val.v,e.v)=0)],
       e->let(p1:=FirstPosition(@1.val.rChildren(),e->e=@(2).val),a:=ApplyFunc(add,ListWithout(@(1).val.rChildren(),p1)),imod(a,@(4).val))),

   imod_sub := Rule([imod,  [@(0,sub), @(1), @(2)],  @(3).cond(x -> _divides(x, @(1).val) or _divides(x, @(2).val))], 
       e -> When(_divides(@(3).val, @(2).val), imod(@(1).val, @(3).val), imod(-@(2).val, @(3).val))),

   imod_add := Rule([@(1, imod), [ add, ..., @(2).cond(x -> _divides(@(1).val.args[2], x)), ... ], @], 
       e -> imod(ApplyFunc(add, Filtered(e.args[1].args, x -> not _divides(e.args[2], x))), e.args[2])),

   imod_divides := Rule( [imod, @(1), @(2).cond( x -> _divides(x, @(1).val) )],
       e -> e.t.zero() ),

   # NOTE: idiv rule disabled, because its invalid if summands have different signs (a+2)/2 != a/2 + 1, if a<0
   #        can this be overcome?
   #idiv_addmul  := Rule([idiv,[@(1,add),...,[@(2,mul),...,@(3,Value),...],...],@(4,Value,e->Mod(@(3).val.v,e.v)=0)],
   #    e->let(p1:=FirstPosition(@1.val.rChildren(),e->e=@(2).val),a:=ApplyFunc(add,ListWithout(@(1).val.rChildren(),p1)),add(idiv(a,@(4).val),idiv(@(2).val,@(4).val)))),

   # NOTE: idiv rule disabled, because its invalid if summands have different signs (a+2)/2 != a/2 + 1, if a<0
   #        can this be overcome?
   divs_addmul2 := Rule([@(1, [div, fdiv]), [ add, ..., [ @(2, mul), ..., @(3).cond(x -> x = @(1).val.args[2]), ...], ... ], @], 
       e -> add( ApplyFunc(mul, DropCond(@(2).val.args, x -> x=e.args[2], 1)),
                 ObjId(e)(ApplyFunc(add, DropCond(e.args[1].args, x -> x = @(2).val, 1)), e.args[2]))),

   # NOTE: idiv rule disabled, because its invalid if summands have different signs (a+2)/2 != a/2 + 1, if a<0
   #        can this be overcome?
   divs_addmul3 := Rule([@(1, [div, fdiv]), [ add, ..., @(2).cond(x -> x = @(1).val.args[2]), ...], @], 
       e -> add( e.t.one(), ObjId(e)(ApplyFunc(add, DropCond(e.args[1].args, x -> x = @(2).val, 1)), e.args[2]))),  

   div_div1 := Rule([div, @(1), [div, @(2), @(3)]], e -> div(@(1).val*@(3).val, @(2).val)),
   div_div2 := Rule([div, [div, @(1), @(2)], @(3)], e -> div(@(1).val, @(2).val*@(3).val)),

# NOTE: removed <idiv> since the rule is invalid for non-divisible numbers
#        can one create a new rule for idiv?
#   idiv_idiv1 := Rule([idiv, @(1), [idiv, @(2), @(3)]], e -> idiv(@(1).val*@(3).val, @(2).val)),
#   idiv_idiv2 := Rule([idiv, [idiv, @(1), @(2)], @(3)], e -> idiv(@(1).val, @(2).val*@(3).val)),

   fdiv_fdiv1 := Rule([fdiv, @(1), [fdiv, @(2), @(3)]], e -> fdiv(@(1).val*@(3).val, @(2).val)),
   fdiv_fdiv2 := Rule([fdiv, [fdiv, @(1), @(2)], @(3)], e -> fdiv(@(1).val, @(2).val*@(3).val)),

   mul_middleValue := Rule(@(1,mul,e -> PositionProperty(e.args, IsValue) > 1 and
                                    Length(Filtered(e.args, IsValue))=1),
       e -> let(i := PositionProperty(e.args, IsValue),
       Product(Permuted(e.args, (1,i))))),

   add_Value_1st := Rule(@(1,add,e->ForAny(e.args, IsValue) and not IsValue(e.args[1])),
    e -> let(s := SplitBy(e.args, IsValue), values := s[1], vars := s[2],
        ApplyFunc(add, [ApplyFunc(add, values)] :: vars))),


# NOTE: removed <idiv> since the rule is invalid for non-divisible numbers
#        can one create a new rule for idiv?

   div_add_Values := Rule([@(0,[div,fdiv]), [@(1,add), @(2, Value), ...],
                                             @(3, Value, e->IsInt(@(2).val.v) and (@(2).val.v mod e.v = 0))],
       e -> add(@(2).val.v / @(3).val.v,
            ObjId(@(0).val)(ApplyFunc(add, Drop(@(1).val.args, 1)), @(3).val))),

   # NOTE: this rule ignores the 3rd parameter of bin_shl (!!), which is 'bits'
   div_add_shl := Rule([@(0,[div,fdiv]), [@(1,add), ..., @(2, arith_shl)],
                                             @(3, Value, e->IsInt(@(2).val.args[2].v)
                                                 and Is2Power(e.v)
                                                 and Log2Int(e.v)<=@(2).val.args[2].v)],
       e -> add(
              arith_shl(@(2).val.args[1],@(2).val.args[2].v - Log2Int(@(3).val.v)),
            ObjId(@(0).val)(ApplyFunc(add, DropLast(@(1).val.args, 1)), @(3).val))),


   div_mul_mul_cfold1 := Rule([@(0,[idiv,div,fdiv]),
       [@(1,mul), @(2,Value), ...],
       [@(3,mul), @(4,Value,
           e -> (@(2).val.t = TInt and e.t=TInt and (@(2).val.v mod e.v)=0)), ...]],
   e -> ObjId(@(0).val)(ApplyFunc(mul, Concatenation([@(2).val.v/@(4).val.v], Drop(@(1).val.args, 1))),
                        ApplyFunc(mul, Drop(@(3).val.args, 1)))),

   div_mul_mul_cfold2 := Rule([@(0,[idiv,div,fdiv]),
       [@(1,mul), @(2,Value), ...],
       [@(3,mul), @(4,Value,
           e -> (@(2).val.t = TInt and e.t=TInt and (e.v mod @(2).val.v)=0)), ...]],
   e -> ObjId(@(0).val)(ApplyFunc(mul, Drop(@(1).val.args, 1)),
                        ApplyFunc(mul, Concatenation([@(4).val.v/@(2).val.v], Drop(@(3).val.args, 1))))),

   div_mul_Value_cfold1 := Rule([@(0,[div, idiv, fdiv]),
       [@(1,mul), @(2,Value), ...],
       @(4,Value, e->(@(2).val.t = TInt and e.t=TInt and (@(2).val.v mod e.v)=0))],
   e -> ApplyFunc(mul, Concatenation([@(2).val.v/@(4).val.v], Drop(@(1).val.args, 1)))),

   div_mul_Value_cfold2 := Rule([@(0,[div,idiv, fdiv]),
       [@(1,mul), @(2,Value), ...],
       @(4,Value, e->(@(2).val.t = TInt and e.t=TInt and (e.v mod @(2).val.v)=0))],
   e -> ObjId(@(0).val)(Product(Drop(@(1).val.args, 1)),
                        @(4).val.v/@(2).val.v)),

   div_mulmul_cfold_vars1 := Rule([@(0,[div,idiv,fdiv]),
       @(1, mul, e->Length(Filtered(e.args, IsLoc)) >= 1),
       @(2, mul, e->let(vv:=Filtered(e.args, IsLoc), Length(vv) >= 1 and Intersection(vv, @(1).val.args)<>[]))],
   e -> let(s1 := SplitBy(@(1).val.args, IsLoc), vars1 := s1[1], nonvars1 := s1[2],
            s2 := SplitBy(@(2).val.args, IsLoc), vars2 := s2[1], nonvars2 := s2[2],
            ObjId(@(0).val)(
                Product(nonvars1) * Product(ListDifference(vars1, vars2)),
                Product(nonvars2) * Product(ListDifference(vars2, vars1))))),

   div_mulmul_cfold_vars2 := Rule([@(0,[div,idiv,fdiv]),
       @(1, mul, e->Length(Filtered(e.args, IsLoc)) >= 1),
       @(2).cond(e->IsLoc(e) and e in @(1).val.args) ],
   e -> let(a := @(1).val.args, ApplyFunc(mul, ListWithout(a, Position(a, @(2).val))))),

   div_mulmul_cfold_vars3 := Rule([@(0,[div,idiv,fdiv]),
       @(1).cond(IsLoc),
       @(2, mul, e->@(1).val in e.args) ],
   e -> let(a := @(2).val.args, ObjId(@(0).val)(1, ApplyFunc(mul, ListWithout(a, Position(a, @(1).val)))))),

   div_cfold_vars4 := Rule([@(0,[div, idiv, fdiv]), 
       @(1).cond(IsLoc),
       @(2).cond(x->IsLoc(x) and x=@(1).val)],
   e -> e.t.one()),

   pdiv_param := Rule([pdiv, @(0, param), @(1, Value)], e -> div(@(0).val, @(1).val)),

   pdiv_add := Rule([pdiv, @(0, add), @(1, Value)], e -> ApplyFunc(add, List(@(0).val.rChildren(), x->pdiv(x, @(1).val)))),

   #idiv is a trick to avoid inversion of this rule by mul_div
   pdiv_mul := Rule([pdiv, [mul, @(1, param), @(2)], @(3, Value)], e -> mul(idiv(@(1).val, @(3).val), @(2).val)),

   pdiv_const := Rule([pdiv, [mul, @(1, Value), @(2)], @(3, Value, e->Mod(@(1).val.v, e.v)=0)],
       e -> mul(div(@(1).val, @(3).val), @(2).val)),

   merge_logic_and := Rule([ logic_and, ..., logic_and, ... ], 
       x -> ApplyFunc(logic_and, ConcatList(x.args, e->When(ObjId(e)=logic_and, e.args, [e]) )))
));

RulesMergedStrengthReduce := MergedRuleSet(RulesStrengthReduce, RulesExpensiveStrengthReduce);

ESReduce := (c, opts) -> ApplyStrategy(c, [RulesMergedStrengthReduce], BUA, opts);


Class(NormalizeLinearExp, RuleSet);
RewriteRules(NormalizeLinearExp, rec(
    # Normalization
    mul_zero := Rule([mul, _0none, @], e -> @(0).val),
    mul_one := Rule([mul, _1, @], e -> @.val),
    mul_assoc := ARule(mul, [ @(1,mul) ], e -> @1.val.args),

    mul_Value_Value := Rule([mul, @(1,Value), @(2,Value)], # UnifyTypes([@(1).val.t, @(2).val.t]).value
    e -> V(@(1).val.v * @(2).val.v)),

    mul_add_rt := Rule([mul, ..., @(2,add)], 
        e -> ApplyFunc(add, List(@(2).val.args, 
                a -> ApplyFunc(mul, Concatenation(Copy(DropLast(e.args, 1)),[a]))))), #Copy prevents aliasing
    mul_add_lft := Rule([mul, @(2,add), ...], 
        e -> ApplyFunc(add, List(@(2).val.args, 
                a -> ApplyFunc(mul, Concatenation([a], Copy(Drop(e.args, 1))))))), #Copy prevents aliasing

    mul_mul := Rule([mul, @(1,Value), [mul, @(2,Value), @(3)]], e -> mul(@(1).val.v * @(2).val.v, @(3).val)),

    sub_to_add := Rule([sub, @(1), @(2)], e -> add(@(1).val, stickyNeg(@(2).val))),

    add_assoc := Rule([add, ..., @(1,add), ... ], e -> let(
	  args := ConcatList(e.args, x->Cond(ObjId(x)=add, x.args, [x])),
	  #args)),
          _groupSummandsSimple(args))), 

    add_zero := ARule(add, [_0none],  e -> [ ]),
    add_single := Rule([add, @(1)], e -> @(1).val),
    add_none := Rule([add], e -> V(0)),

    # Constant folding
    cfold := RulesStrengthReduce.rules.cfold,

    mul_cfold := RulesExpensiveStrengthReduce.rules.mul_cfold,
    add_cfold := RulesExpensiveStrengthReduce.rules.add_cfold,
    # put constants first
    mul_middleValue := RulesExpensiveStrengthReduce.rules.mul_middleValue,

    # negation
    neg        := Rule(neg, e -> stickyNeg(e.args[1])),
    mul_negone := Rule([mul, _neg1, @2], e -> stickyNeg(@2.val)),
    neg_neg := Rule([stickyNeg, [stickyNeg, @1]], e -> @1.val), # -(-1) -> 1

    mul_neg := Rule([mul, ..., @(1, [stickyNeg, Value], e->When(ObjId(e)=Value, isNegV(e), true)), ...],
        e -> let(cnt := Counter(0),
                 count := List(e.args, x -> When(ObjId(x)=stickyNeg or (ObjId(x)=Value and isNegV(x)), cnt.next(), 0)),
                 newargs := List(e.args, x -> Cond(
                         ObjId(x)=stickyNeg, x.args[1],
                         ObjId(x)=Value and isNegV(x), Value.new(x.t, -x.v),
                         x)),
                 res := ApplyFunc(mul, newargs),
                 When(IsOddInt(cnt.n), stickyNeg(res), res))),
    sink_stickyNeg_add := Rule([stickyNeg, @(1, add)], e -> ApplyFunc(add, List(@(1).val.args, stickyNeg))),

    # This rule is in conflict with mul_div rule from ESReduce but it simplifies index computations.
    # div implies divisibility but loop index takes different values from .range field -> other
    # arguments must be divisible by @(3)
    div_pull_out_idx := Rule([div, [@(1, mul), ..., @.cond(IsLoopIndex), ...], @(3, Value)],
        e -> let( m := SplitBy(@(1).val.args, IsLoopIndex),
                  ApplyFunc(mul, [div(ApplyFunc(mul, m[2]), @(3).val)] :: m[1]) )),
));

GroupSummandsExp := s -> NormalizeLinearExp(SubstBottomUp(NormalizeLinearExp(s), add, e->_groupSummands(e.args)));
