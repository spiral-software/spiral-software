
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# !!!! what's missing: CRT of 2 primes perm pullout

RewriteRules(RulesFuncSimp, rec(
 # ===================================================================
 # OddStride
 # ===================================================================
 OS_Id := Rule([OS, @N, 1], e->fId(@N.val)),

 OS_OS := ARule(fCompose, [[OS, @N, @r], [OS, @, @s]],
     e -> [ OS(@N.val, @r.val * @s.val mod @N.val) ]),

 OS_RR_PhaseShift := ARule( fCompose, [[OS, @N, @r], [RR, @, @phi, @g]],
     e -> [ RR(@N.val, @phi.val * @r.val, @g.val) ]),

 OS_RM_PhaseShift := ARule( fCompose, [[OS, @N, @r], [RM, @, @phi, @g]],
     e -> [ RM(@N.val, @phi.val * @r.val, @g.val) ]),

 # ===================================================================
 # Cyclic shift (Z) propagation
 # Z o (fBase X f)  ->  (Z o f) X (Z o fBase)
 Z_Propagate := ARule(fCompose, [[Z, @, @z], [fTensor, @F, [fBase, @m, @j]]],
     (e, cx) -> [ fTensor(
       fCompose(Z(range(@F.val), memo(cx, "g", idiv(@j.val+@z.val, @m.val))), @F.val),
       fCompose(Z(@m.val, @z.val), fBase(@m.val, @j.val))) ]),

 # ===================================================================
 # CRT / gammaTensor
 # ===================================================================
#   NOTE: This rule triggers problems when fired too early, as gammatensor cannot be transposed -- what to do??
# CRT_toGammaTensor := Rule(CRT, e->e.toGammaTensor()),

 GammaTensorAssoc := ARule(gammaTensor, [@(1,gammaTensor)], e->@(1).val.children()),

 # gammaTensor o fTensor -> gammaTensor( ... o ..., ... o ... , ...)
 GammaTensorMerge := ARule(fCompose,
      [ @(1,gammaTensor), @(2,fTensor,e->compat_tensor_chains(
                         @(1).val.children(), e.children(), domain, range)) ],
 e -> [ gammaTensor(merge_tensor_chains(
       @(1).val.children(), @(2).val.children(), fCompose, x->x, x->x, domain, range))]),

 ### fTensor(..., Y, ...) o X -> fTensor(..., Y o X, ...)
 GammaTensorComposeMergeRight := ARule(fCompose,
      [ @(1,gammaTensor), @(2).cond(e->compat_tensor_chains(
                         @(1).val.children(), [e], domain, range)) ],
 e -> [ gammaTensor(merge_tensor_chains(
       @(1).val.children(), [@(2).val], fCompose, x->x, x->x, domain, range)) ] ),

 ### X o fTensor(..., Y, ...) -> fTensor(..., X o Y, ...)
 GammaTensorComposeMergeLeft := ARule(fCompose,
      [ @(1), @(2,gammaTensor,e->compat_tensor_chains(
                         [@(1).val], e.children(), domain, range)) ],
 e -> [ gammaTensor(merge_tensor_chains(
       [@(1).val], @(2).val.children(), fCompose, x->x, x->x, domain, range)) ] ),

 # ===================================================================
 # Rader permutations
 # ===================================================================
 # NB: RR = fStack(1,RM), RR plugged into gammaTensor, eg. F_15 -> F_3, F_5

 # RR o fAdd(N,1,0) -> (0)_N
 RR_1stPoint_fAdd := ARule(fCompose, [[RR, @N, @phi, @g], [fAdd, @, 1, 0]],
     e -> [ H(@N.val, 1, 0, 1) ]),
 RR_1stPoint_H := ARule(fCompose, [[RR, @N, @phi, @g], [H, @, 1, 0, @]],
     e -> [ H(@N.val, 1, 0, 1) ]),

 # RR o fAdd(N,N-1,1) -> RM
 RR_toRM_fAdd := ARule(fCompose,  [[RR, @N, @phi, @g], [fAdd, @, @(1).cond(e->e=@N.val-1), 1]],
     e -> [ RM(@N.val, @N.val-1, @phi.val, @g.val) ]),
 RR_toRM_H := ARule(fCompose,  [[RR, @N, @phi, @g], [H, @, @(1).cond(e->e=@N.val-1), 1, 1]],
     e -> [ RM(@N.val, @N.val-1, @phi.val, @g.val) ])
));

Class(RulesRaderTensor, RuleSet);

RewriteRules(RulesRaderTensor, rec(
 # RM o fTensor -> RM
 RM_Tensor1 := ARule(fCompose, [ [RM,@N,@n,@phi,@g], [fTensor, @(1,fId), [fBase,@m,@j]] ],
     (e, cx) -> [ RM(@N.val, domain(@1.val),
           memo(cx, "g", powmod(@phi.val,@g.val,@j.val,@N.val)),
           powmod(1,@g.val,@m.val,@N.val).eval()) ] ),

 RM_Tensor2 := ARule(fCompose, [ [RM,@N,@n,@phi,@g], [fTensor, [fBase,@m,@j], @(1,fId)] ],
     (e, cx) -> [ RM(@N.val, domain(@1.val),
           memo(cx, "g", powmod(@phi.val, powmod(1,@g.val,@m.val,@N.val).eval(), @j.val, @N.val)),
           @g.val) ])
));
