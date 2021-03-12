
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


same_params := (pat1, targ, pat2) -> pat1.target(targ).cond(e->e.params = pat2.val.params);
one_by_one := x -> x.range() = 1 and x.domain() = 1;

# does not work with .cond (!!!) NOTE
Enum(100, x->@(x), @z, @n, @m, @k, @l, @j, @g, @phi, @r, @s, @ss, @b, @bb);
Enum(200, x->@(x), @M, @N, @R, @S, @W, @F, @G, @D);

IsIndexMapping := x -> IsBound(x._perm) and x._perm;

# L^mn_n
ltensor_flip := function(match, mn, n)
    local m, i, pos, dom, ldom, lran, ll, rr, res, ldom;
    dom := Product(match, x->x.domain());
    ldom := 1; lran := 1; pos := 1;
    while lran<>n do
        lran:=lran*range(match[pos]);
        ldom:=ldom*domain(match[pos]);
	pos:=pos+1;
    od;
    ll := match{[1..pos-1]};
    rr := match{[pos..Length(match)]};
    res := fTensor(Concatenation(rr, ll));

    if ldom=1 or ldom=dom then
    return res;
    else
    return fCompose(res, L(dom, ldom));
    fi;
end;

# for given diagDirsum looking for child which relative domain start <= offset and relative domain end >= (offset + domain)
# returns [child index, offset relative to child]
_findDirsumChild := function( dirsum, offset, domain)
    local i, ch, d;
    ch := dirsum.children();
    d := 0;
    for i in [1..Length(ch)] do
        if offset >= d then
            d := d + ch[i].domain();
            if offset + domain <= d then
                return [i, offset - d + ch[i].domain()];
            fi;
        else
            return [0, 0];
        fi;
    od;
    return [0, 0];
end;

# *******************************************************************
Class(RulesFuncSimp, RuleSet);
RewriteRules(RulesFuncSimp, rec(
 # ===================================================================
 # Operator flattening and identity
 # ===================================================================
 FlattenTensor  := ARule(fTensor,  [@(1, fTensor) ], e -> @(1).val.children()),
 FlattenCompose := ARule(fCompose, [@(1, fCompose)], e -> @(1).val.children()),

 ComposeId1 := ARule(fCompose, [@(1), fId ], e -> [@(1).val]),
 ComposeId2 := ARule(fCompose, [fId,  @(1)], e -> [@(1).val]),

 fBase_1x1 := Rule(@(1, fBase, one_by_one), e->fId(1)),

 fId_drop_Value := Rule([@(1, fId), @(2, Value)], e->fId(@(2).val.v)),

 Const_fbase := ARule(fCompose, [@(1), @(2, fBase, e -> IsInt(e.params[2]))],
     e -> [ When(IsInt(@(1).val.range()),
                 fBase(@(1).val.range(), @(1).val.at(@(2).val.params[2])),
                 fConst(@(1).val.range(), 1, @(1).val.at(@(2).val.params[2]))) ]),

 Compose1x1_1 := ARule(fCompose, [@(1).cond(one_by_one), @(2)],  e -> [@(2).val]),
 Compose1x1_2 := ARule(fCompose, [@(2), @(1).cond(one_by_one) ], e -> [@(2).val]),

 TensorId1 := ARule(fTensor, [@(1), [@(2,[fId,J]), 1]], e -> [@(1).val]),
 TensorId2 := ARule(fTensor, [[@(2,[fId,J]), 1], @(1)], e -> [@(1).val]),

 TensorId1V := ARule(fTensor, [@(1), [@(2,[fId,J]), @(3, Value, e->e.v=1)]], e -> [@(1).val]),
 TensorId2V := ARule(fTensor, [[@(2,[fId,J]), @(3, Value, e->e.v=1)], @(1)], e -> [@(1).val]),

 diagTensorId1  := ARule(diagTensor, [@(1), [fConst, @(2), 1, 1]], e -> [@(1).val]),
 diagTensorIdV1 := ARule(diagTensor, [@(1), [fConst, @(2), 1, _1]], e -> [@(1).val]),
 diagTensorId2  := ARule(diagTensor, [[fConst, @(2), 1, 1], @(1)], e -> [@(1).val]),
 diagTensorIdV2 := ARule(diagTensor, [[fConst, @(2), 1, _1], @(1)], e -> [@(1).val]),

 diagTensorZero1  := ARule(diagTensor, [@(1), [fConst, @(2), 1, 0]], e -> [fConst(@(1).val.domain(), 0)]),
 diagTensorZeroV1 := ARule(diagTensor, [@(1), [fConst, @(2), 1, _0]], e -> [fConst(@(1).val.domain(), 0)]),
 diagTensorZero2  := ARule(diagTensor, [[fConst, @(2), 1, 0], @(1)], e -> [fConst(@(1).val.domain(), 0)]),
 diagTensorZeroV2 := ARule(diagTensor, [[fConst, @(2), 1, _0], @(1)], e -> [fConst(@(1).val.domain(), 0)]),

 DropL := Rule(@(1, [L,OS], e -> e.params[2] = 1 or e.params[2] = e.params[1]), e -> fId(@1.val.params[1])),
 DropTr := Rule(@(1, Tr, e -> e.params[1] = 1 or e.params[2] = 1), e -> fId(@1.val.domain())),

 IP_toI  := Rule([IP, @(2), fId], e -> fId(@(2).val)),
 J_1   := Rule([J, 1], e->fId(1)),
 OS_2  := Rule([OS, 2, @], e->fId(2)),

 OS_fAdd := ARule(fCompose, [ [OS, @(1), @(2).cond(e -> e in [@(1).val-1, -1])],
                              [fAdd, @(3), @(4).cond(e -> e = @(3).val - 1), 1] ],
     e -> [ fAdd(@(3).val, @(4).val, 1), J(@(4).val) ]),

 Lfold := ARule(fCompose, [@(1,L), @(2,L),e->let(
             prod := @(1).val.params[2] * @(2).val.params[2],
             a:=@(1).val.params[1],
             When(prod < a, prod in FactorsInt(a), (prod / a) in FactorsInt(a)))],
     e -> let(
         prod := @(1).val.params[2] * @(2).val.params[2], a:=@(1).val.params[1],
         [ L(a, When(prod < a, prod, prod / a)) ])),

 Hfold :=  ARule(fCompose, [@(1,H), @(2,H)],
     e-> [H(@(1).val.params[1],
             @(2).val.params[2],
             @(1).val.params[3]+@(2).val.params[3]*@(1).val.params[4],
             @(1).val.params[4]*@(2).val.params[4])]),

 # ===================================================================
 # Diagonal Functions
 # ===================================================================
 ComposePrecompute := Rule([fCompose, ..., [fPrecompute, @(1)], ...],
     e -> fPrecompute(ApplyFunc(fCompose, List(e.children(),
         x->When(ObjId(x)=fPrecompute, x.child(1), x))))),

 PrecomputePrecompute := Rule([fPrecompute, [fPrecompute, @(1)]], e->fPrecompute(@(1).val)),

 # drop diagDirsum when H refers to only one its child
 drop_diagDirsum_H := ARule( fCompose, [ @(1, diagDirsum), @(2, H, x -> x.params[4] = 1 and _findDirsumChild(@(1).val, x.params[3], x.params[2])[1]>0)],
     e -> let( f := _findDirsumChild(@(1).val, @(2).val.params[3], @(2).val.params[2]),
               obj := @(1).val.children()[f[1]],
               [ obj, H(obj.domain(), @(2).val.params[2], f[2], 1) ] )),

 RCData_fCompose_TReal := Rule([RCData, [@(1,fCompose), @(2).cond(e->e.range = TReal), ...]], 
     e -> fCompose(RCData(@(2).val), diagTensor(fCompose(Drop(@(1).val.children(),1)), fId(2)))),

 RCData_fCompose_TInt := Rule([RCData, [@(1,fCompose), @(2).cond(e->e.range = TInt), ...]], 
     e -> fCompose(RCData(@(2).val), fTensor(fCompose(Drop(@(1).val.children(),1)), fId(2)))),

# RCData_fCompose := Rule([RCData, [@(1,fCompose), @(2), ...]], 
#     e -> fCompose(RCData(@(2).val), fTensor(fCompose(Drop(@(1).val.children(),1)), fId(2)))),


 RCData_Precompute := Rule([RCData, [fPrecompute, @(1)]], e->fPrecompute(RCData(@(1).val))),
 RCData_CRData     := Rule([RCData, [CRData, @(1)]], e -> @(1).val),
 CRData_RCData     := Rule([CRData, [RCData, @(1)]], e -> @(1).val),
 
 FConj_Precompute := Rule([@(0,[FConj,FRConj]), [fPrecompute, @(1)]], e->fPrecompute(ObjId(@(0).val)(@(1).val))),

 ComposeConstX := ARule(fCompose, [@(1,fConst), @(2)],
     e -> [fConst(@(1).val.params[1], @(2).val.domain(), @(1).val.params[3])]),

 fAdd0 := Rule([fAdd, @(1), @(2).cond(e->e=@(1).val), 0], e->fId(@(1).val)),

 fAddPair := ARule(fCompose, [@(1, fAdd), @(2, fAdd)],
     e -> [ fAdd(@(1).val.range(), @(2).val.domain(),
             @(1).val.params[3] + @(2).val.params[3]) ]),

 fTensor_fAdd_H := ARule(fTensor, [ [@(1, fAdd),@, _1, @], @(2, H)],
     e -> [ H(@(1).val.params[1] * @(2).val.params[1], 
              @(2).val.params[2], 
              @(2).val.params[1] * @(1).val.params[3] + @(2).val.params[3],
              @(2).val.params[4]) ]),

 # ===================================================================
 # fTensor
 # ===================================================================
 # fId(m) (X) fId(n) -> fId(m*n)
  TensorIdId := ARule(fTensor, [@(1,fId), @(2,fId)],
    e -> [ fId(@(1).val.params[1] * @(2).val.params[1]) ]),

  TensorId1 := ARule(fTensor, [@(1, fId, x->x.params[1]=1)],
    e -> []),

 # L(mn,m) o (fBase(m,j) (X) f) -> f (X) fBase(j,m)
 LTensorFlip := ARule(fCompose,
      [ @(1,L), [ @(3,fTensor), @(2).cond(e->range(e) = @(1).val.params[2] and domain(e)=1), ...] ],
 	e -> [ fTensor(Copy(Drop(@(3).val.children(), 1)), Copy(@(2).val)) ] ),

 # L(mn,n) o f (X) fBase(j,m) -> (fBase(m,j) (X) f)
 LTensorFlip1 := ARule(fCompose,
      [ @(1,L), [ @(3,fTensor), ...,
             @(2, fBase, e->range(e) = @(1).val.params[1]/@(1).val.params[2]) ]],
    e -> [fTensor(Copy(Last(@(3).val.children())), Copy(DropLast(@(3).val.children(), 1)))] ),
    
 L_H := ARule(fCompose, [ @(1, L), [ @(2, fTensor), @(3, fBase), @(4, fId, e->IsInt(@(1).val.params[1]/(@(1).val.params[2] * e.params[1])))] ],
    e-> [ H(@(1).val.params[1], @(4).val.params[1], 
            @(4).val.params[1] * @(1).val.params[2] * imod(@(3).val.params[2], @(1).val.params[1] / (@(4).val.params[1] * @(1).val.params[2])) +
            idiv(@(3).val.params[2], @(1).val.params[1] / (@(4).val.params[1] * @(1).val.params[2])), 
            @(1).val.params[2]) ]),   

  Refl0_u_H0 := ARule(fCompose, [@(1, Refl0_u), @(2, H, e -> e.params[3]=0 and e.params[4]=1)], e -> [ let(
     k := @(1).val.params[1],
     H(@(1).val.range(), @(2).val.domain(), 0, k)) ]),

  Refl0_u_Hrest := ARule(fCompose, [
          @(1, Refl0_u),
          @(2, H, e -> e.params[3]=@(1).val.params[2] and e.params[4]=1),
          [fTensor, @(3,fBase), @(4, fId, e->e.params[1]=@(1).val.params[2])]],
      e -> [ let(k := @(1).val.params[1],
              BH(@(1).val.range(), 2*@(1).val.range(), @(4).val.domain(), 1+@(3).val.params[2], 2*k)) ]),

  Refl0_u_Hrest_1it := ARule(fCompose, [
          @(1, Refl0_u),
          @(2, H, e -> e.params[3]=@(1).val.params[2] and e.params[4]=1 and e.params[3]=e.params[2]) ],
      e -> [ let(k := @(1).val.params[1],
              BH(@(1).val.range(), 2*@(1).val.range(), @(2).val.domain(), 1, 2*k)) ]),

  Refl1_H := ARule(fCompose, [@(1, Refl1), [fTensor, @(2,fBase), @(3, fId, e->e.params[1]=@(1).val.params[2])]],
      e -> let(
          k := @(1).val.params[1],
          [ BH(@(1).val.range(), 2*@(1).val.range() - 1, @(3).val.domain(), @(2).val.params[2], 2*k) ])),

  fTensor_X_fId_H := ARule(fCompose, [[@(1, fTensor), ..., [fId, @(2)]], # NOTE: assumes d|base
        [@(3, H), @(4).cond(e->_divides(@(2).val,e)), @(5).cond(e->_divides(@(2).val,e)),
                  @(6).cond(e->_divides(@(2).val,e)), _1]], e ->
       let(p := @(3).val.params, d := @(2).val, base := p[3],
           [ @(1).val, fTensor(H(p[1]/d, p[2]/d, base/d, 1), fId(d)) ])),

 TrTensor := ARule(fCompose, [@(1, Tr), [ fTensor, @(2), @(3).cond(e->e.range()=@(1).val.params[1]) ]],
     e -> [ fCompose(fTensor(@(3).val, @(2).val), Tr(@(3).val.domain(), @(2).val.domain())) ]
 ),

 LTensorGeneral := ARule(
fCompose, 
[@(1,[L,Tr]),
 @(2,fTensor,
 e -> let(
    p := @(1).val.params,
    a := When(ObjId(@(1).val)=Tr, p[1], p[1]/p[2]), 
    b := p[2],
    ForAll(e.children(), x->AnySyms(x.domain(), x.range()) or x.domain()>1 and x.range()>1) and # workaround old fId(1) bug 
    full_merge_tensor_chains(@(3), [b, a], e.children(), (x,y)->y,
 Product, fTensor,
                             fId, x->x, x->x, range) <> false))],
     e -> [ ltensor_flip(@(3).val, @(1).val.params[1], @(1).val.params[2]) ]
 ),

 # fTensor o fTensor -> fTensor( ..o.., ..o.. , ...)
 TensorMerge := ARule(fCompose,
      [ @(1,fTensor), @(2,fTensor,e->full_merge_tensor_chains(
          @(3), @(1).val.children(), e.children(),
                fCompose, fTensor, fTensor, x->x, x->x, domain, range) <> false) ],
 e -> [ fTensor(@(3).val) ]
 ),

 TensorMerge_fIdInt := ARule(fCompose,
      [ @(1,fTensor,e->let(l1:=Last(e.children()), ObjId(l1)=fId and IsInt(l1.domain()))),
        @(2,fTensor, e->let(l1:=Last(@(1).val.children()), l2 := Last(e.children()), IsInt(l2.domain()) and ObjId(l2)=fId and Gcd(l1.domain(), l2.domain())>1))],
        e -> let(l1 := Last(@(1).val.children()).domain(), l2 := Last(@(2).val.children()).domain(), gcd := Gcd(l1, l2),
            [ fTensor(fCompose(fTensor(DropLast(@(1).val.children(), 1), fId(l1/gcd)), fTensor(DropLast(@(2).val.children(), 1), fId(l2/gcd))), fId(gcd)) ])
    ),

 # diagTensor o fTensor -> diagTensor( ..o.., ..o.. , ...)
 diagTensorMerge := ARule(fCompose,
      [ @(1,diagTensor), @(2,fTensor,e->full_merge_tensor_chains(
          @(3), @(1).val.children(), e.children(),
                fCompose, diagTensor, fTensor, x->x, x->fConst(range(x), 1), domain, range) <> false) ],
 e -> [ diagTensor(@(3).val) ]
 ),

 # fTensor(..., Y, ...) o X -> fTensor(..., Y o X, ...)
 TensorComposeMergeRight := ARule(fCompose,
      [ @(1,fTensor), @(2).cond(e -> compat_tensor_chains(
                                @(1).val.children(), [e], domain, range)) ],
 e -> [ fTensor(merge_tensor_chains(
         @(1).val.children(), [@(2).val], fCompose, x->x, x->x, domain, range)) ]
 ),

 # X o fTensor(..., Y, ...) -> fTensor(..., X o Y, ...)
 TensorComposeMergeLeft := ARule(fCompose,
      [ @(1), @(2,fTensor,e-> IsInt(@(1).val.range()) and compat_tensor_chains(  # NOTE: BAD HACK!!! Cannot do that with diag functions...
                             [@(1).val], e.children(), domain, range)) ],
 e -> [ fTensor(merge_tensor_chains(
         [@(1).val], @(2).val.children(), fCompose, x->x, x->x, domain, range)) ]
 ),

 # [fBase,fId] X (fB o fAny) -> fB2 o (fI X fAny)
 TensorB := ARule(fTensor, [
        @(1, [fBase, fId], e -> Is2Power(e.range())),
        @(2, fCompose, e -> ObjId(e.child(1)) = fB)
    ],
    e -> [fCompose(
        let(i := @(1).val,
            b := @(2).val.child(1),
            fB(i.range() * b.range(), List(b.params[3], e -> Log2Int(i.range()) + e))
        ),
        fTensor(
            @(1).val,
            @(2).val.child(2)
        )
    )]
 ),

 # ===================================================================
 # Other
 # ===================================================================

 # primitive constant folding for functions
 Compose_FList_const := ARule(fCompose, [@(1, FList), @(2).cond(x->x.free()=[])], 
     e -> let(
	 lst := @(1).val,  idx := List(@(2).val.tolist(), EvalScalar),
	 [ FList(lst.t, lst.list{1+idx}) ])),

 # fCond o f  (note: fCond o fCond should only be handled by Compose_fCond_lft)
 Compose_fCond_rt := ARule(fCompose, [[fCond, @(1), @(2), @(3)], @(4).cond(e->ObjId(e)<>fCond)],
     e -> [ fCond(fCompose(@(1).val, @(4).val), fCompose(@(2).val, @(4).val), fCompose(@(3).val, @(4).val)) ]),

 Compose_fCond_lft := ARule(fCompose, [@(4), [fCond, @(1), @(2), @(3)]],
     e -> [ fCond(@(1).val, fCompose(@(4).val, @(2).val), fCompose(@(4).val, @(3).val)) ]),

 fCond_fold0  := Rule([fCond, [fConst, @, @,  0], @(2), @(3)], e -> @(3).val),
 fCond_foldV0 := Rule([fCond, [fConst, @, @, _0], @(2), @(3)], e -> @(3).val),

 fCond_fold1  := Rule([fCond, [fConst, @, @,  1], @(2), @(3)], e -> @(2).val),
 fCond_foldV1 := Rule([fCond, [fConst, @, @, _1], @(2), @(3)], e -> @(2).val),


 # B2 o fTensor

    fCompose_B2_fTensor := ARule(fCompose, [@(1,B2), @(2,fTensor)],
        e -> [fB2(@(1).val.getES(), @(2).val.children())]
    ),

 # L(abc,b) o L(abc,c) -> L(abc,bc)
 fCompose_Tr := ARule( fCompose, [ [@(1, [Tr, L]), @, @(2)], [@(3, [Tr, L]), @, @(4).cond( x -> let( abc := @(1).val.range(), @(3).val.domain() = abc and (abc mod (@(2).val*x) = 0)))]],
     e -> let( abc := @(1).val.range(), bc := @(2).val*@(4).val, 
         When( ObjId(@(1).val) = L, [L(abc, bc)], [Tr(abc/bc, bc)] ))),

 fDirsum_fIds := ARule( fDirsum, [@(1, fId), @(2, fId)], e -> [fId(@(1).val.range()+@(2).val.range())]),

));
