
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


canReorder := e -> not IsBound(e.doNotReorder);
Class(RulesSums, RuleSet);

RewriteRules(RulesSums, rec(
# Drop_DPWrapper := Rule([DPWrapper, @(1), ...], e->@(1).val),
 Drop_GrpISum := Rule([Grp, @(1, ISum)], e -> ISum(@(1).val.var, @(1).val.domain, @(1).val.child(1))),
 Drop_GrpInplaceISum := Rule([Grp, [Inplace, @(1, ISum)]], e -> Inplace(ISum(@(1).val.var, @(1).val.domain, @(1).val.child(1)))),

 # ================================================================
 # Associativity of Compose and SUM
 # ================================================================
 ComposeAssoc := ARule( Compose, [ @(1,Compose) ],  e -> @(1).val.children() ),
 SUMAssoc     := ARule( SUM,     [ @(1,SUM)     ],  e -> @(1).val.children() ),
 SUMAccAssoc  := ARule( SUMAcc,  [ @(1,SUMAcc)  ],  e -> @(1).val.children() ),
 TensorAssoc  := ARule( Tensor,  [ @(1,Tensor)  ],  e -> @(1).val.children() ),
 RedundandSUMAcc := Rule([SUMAcc, @(1)], e -> @(1).val),

 # ================================================================
 # Identity Matrix (I)
 # ================================================================
 Compose_IR := ARule(Compose, [@(1), I], e->[@(1).val]),
 Compose_IL := ARule(Compose, [I, @(1)], e->[@(1).val]),
 # rule Prm(fId()) -> I may break vectorization
 fId_toI := Rule([@(1, [Gath, Scat]), [fId, @(2)]], e -> I(@(2).val)),

 # ================================================================
 # Zero Matrix (O)
 # ================================================================
 ZeroSUMAcc1 := ARule(SUMAcc, [O, @(1)], e -> [@(1).val]),
 ZeroSUMAcc2 := ARule(SUMAcc, [@(1), O], e -> [@(1).val]),
 ZeroSUM1    := ARule(SUM, [O, @(1)], e -> [@(1).val]),
 ZeroSUM2    := ARule(SUM, [@(1), O], e -> [@(1).val]),
 ZeroTensor := Rule([Tensor, ..., @(1,O), ...], e -> O(Rows(e), Cols(e))),

 # ================================================================
 # Combining Gath, Scat, and Prm
 # ================================================================
 ComposeGathGath := ARule(Compose, [ @(1, Gath), @(2, [Gath, Prm]) ], # o 1-> 2->
     e -> [ Gath(fCompose(@(2).val.func, @(1).val.func)) ]),

 ComposePrmPrm := ARule(Compose, [ @(1, Prm), @(2, Prm) ], # o 1-> 2->
     e -> [ Prm(fCompose(@(2).val.func, @(1).val.func)) ]),

 ComposeScatScat := ARule(Compose, [ @(1, [Scat, ScatAcc]), @(2, [Scat, ScatAcc]) ], # <-1 <-2 o
     e -> [ Cond(ObjId(@(1).val)=ScatAcc or ObjId(@(2).val)=ScatAcc,
                 ScatAcc(fCompose(@(1).val.func, @(2).val.func)),
                 Scat   (fCompose(@(1).val.func, @(2).val.func))) ]),

 ComposePrmScat  := ARule(Compose, [ @(1, Prm), @(2, [Scat, ScatAcc]) ], # 1-> <-2 o
     e -> [ ObjId(@(2).val)(fCompose(@(1).val.func.transpose(), @(2).val.func)) ]),

 ComposeDPrmDPrm   := ARule(Compose, [ @(1, DelayedPrm), @(2, DelayedPrm) ], # direct:  o 1-> 2->
     e -> [ DelayedPrm(fCompose(@(2).val.func, @(1).val.func)) ]),

 DropGathScat := ARule( Compose, [ @(1, Gath), @(2, Scat, x->x.transpose()=@(1).val) ],
     e -> []),

 # ================================================================                   
 # ScatGath
 # NOTE: YSV ScatGath is ugly and should be removed
 # ================================================================
 ComposeScat_ScatGath := ARule(Compose, [ @(1, Scat), @(2, ScatGath) ], # <-1 <-2 o
     e -> [ ScatGath(fCompose(@(1).val.func, @(2).val.sfunc), @(2).val.gfunc) ]),

 ComposeScatGath_Gath := ARule(Compose, [ @(1, ScatGath), @(2, Gath) ], # o 1-> 2->
     e -> [ ScatGath(@(1).val.sfunc, fCompose(@(2).val.func, @(1).val.gfunc)) ]),

 # ================================================================
 # Using distributivity to pull Prm, Gath, Scat into ISum, SUM, etc.
 # ================================================================
 PullInRight := ARule( Compose,
       [ @(1, [Prm, Scat, ScatAcc, TCast, PushR, PushLR, Conj, ConjL, ConjR, ConjLR, FormatPrm]),
         @(2, [RecursStep, Grp, BB, SUM, Buf, ISum, Data, COND, NoDiagPullin, NoDiagPullinLeft, NoDiagPullinRight, NeedInterleavedComplex]) ],
  e -> [ CopyFields(@(2).val, rec(
             _children :=  List(@(2).val._children, c -> @(1).val * c),
             dimensions := [Rows(@(1).val), Cols(@(2).val)] )) ]),

 PullInLeft := ARule( Compose,
       [ @(1, [RecursStep, Grp, BB, SUM, SUMAcc, Buf, ISum, ISumAcc, Data, COND, NoDiagPullin, NoDiagPullinLeft, NoDiagPullinRight, NeedInterleavedComplex]),
         @(2, [Prm, Gath, TCast, PushL, PushLR, Conj, ConjL, ConjR, ConjLR, FormatPrm]) ],
     e -> [ CopyFields(@(1).val, rec(
                _children := List(@(1).val._children, c -> c * @(2).val),
                dimensions := [Rows(@(1).val), Cols(@(2).val)] )) ]),

 ComposePushRPrmScat := ARule(Compose, [ @(1, PushR, e -> ObjId(e.child(1)) = Prm), @(2, Scat) ],
    e -> [ Scat(fCompose(@(1).val.child(1).func.transpose(), @(2).val.func)) ]),

 # Match Gath * PushL(Prm) structure. The normal Gath * PushL rule occasionally
 # does not apply properly -- not clear why! (MRT)
 ComposeGathPushLPrm := ARule(Compose, [ @(1, Gath), @(2, PushL, e -> ObjId(e.child(1)) = Prm) ],
    e -> [ Gath(fCompose(@(2).val.child(1).func, @(1).val.func)) ]),

 # ================================================================
 # Low Priority
 # ================================================================
 ComposeScatPrm  := ARule(Compose, [ @(1, [Scat, ScatAcc]), @(2, Prm) ],  # <-1 2-> o
     e -> [ ObjId(@(1).val)(fCompose(@(1).val.func, @(2).val.func.transpose())) ]),

 ComposePrmGath  := ARule(Compose, [ @(1, Prm), @(2, Gath) ], # o 1-> 2->
     e -> [ Gath(fCompose(@(2).val.func, @(1).val.func)) ]),


 # ================================================================
 # TCast
 # ================================================================
 ComposeConvScat  := ARule(Compose, [ @(1, TCast), @(2, [Scat, ScatAcc]) ],  # <-1 2-> o
     e -> [ @(2).val, TCast(Cols(@(2).val), @(1).val.params[2], @(1).val.params[3]) ]),

 ComposeGathConv  := ARule(Compose, [ @(1, Gath), @(2, TCast) ], # o 1-> 2->
     e -> [ TCast(Rows(@(1).val), @(2).val.params[2], @(2).val.params[3]), @(1).val ]),



 # ================================================================
 # ConjTranspose
 # ================================================================
 InertConjTranspose_@ := Rule(
     [InertConjTranspose, @.cond(e->not (IsBound(e.isInertConjTranspose) and e.isInertConjTranspose()))], 
     e -> e.child(1).conjTranspose()),
));

# ================================================================
# COND Rules
# ================================================================

# NOTE: this replication should not be necessary
# Rule([@(0,SUMAcc), ..., @(1,COND), ...],
#     e -> let( ch := e.children(),
#         left := Left(...),
#         right := Right(...),
#         leftch := ch{[1..left]},
#         rightch := ch{[right..Length(ch)]},
#         OP := ObjId(@(0).val),
#         COND(@(1).val.cond,
#         List(@(1).val.children(),
#              c -> OP(Concatenation(Copy(leftch), [c], Copy(rightch)))))));

# ================================================================
# OL Rules
# ================================================================

Class(OLRules, RuleSet);
RewriteRules(OLRules, rec(
  Gath_fCross := Rule([Gath, @(1, fCross)], e -> Cross(List(@(1).val.rChildren(),x->Gath(x)))),

  fCross_fCross := ARule(fCompose, [@(1,fCross), @(2,fCross)],
      e-> [fCross(List([1..Length(@(1).val.rChildren())],i->fCompose(@(1).val.rChildren()[i], @(2).val.rChildren()[i])))]),

  f2DTensor_f2DTensor := ARule(fCompose, [@(1,f2DTensor), @(2,f2DTensor)],
      e-> [f2DTensor(List([1..Length(@(1).val.rChildren())],i->fCompose(@(1).val.rChildren()[i], @(2).val.rChildren()[i])))]),

  f4DTensor_f4DTensor := ARule(fCompose, [@(1,f4DTensor), @(2,f4DTensor)],
      e-> [f4DTensor(List([1..Length(@(1).val.rChildren())],i->fCompose(@(1).val.rChildren()[i], @(2).val.rChildren()[i])))]),

  f2DTr_f2DTensor := ARule(fCompose, [@(1,f2DTr), @(2,f2DTensor)],
      e-> let(f:=f2DTensor(Reversed(@(2).val.rChildren())),
              [f, ApplyFunc(ObjId(@(1).val), Reversed(f.advdomain()[1]))])),

  f2DTr_f2DTr := ARule(fCompose, [@(1,f2DTr), @(2,f2DTr)],
      e-> []),

  f2DTensor_f2DTrExplicit := ARule(fCompose, [@(1,f2DTensor), @(2, [f2DTrExplicitL, f2DTrExplicitR])],
      e-> let(f:=f2DTensor(Reversed(@(1).val.rChildren())),
              [ApplyFunc(ObjId(@(2).val), Reversed(f.advrange()[1])), f])),

  f2DTr_f2DTrExplicit := ARule(fCompose, [@(1,f2DTr), @(2,[f2DTrExplicitL, f2DTrExplicitR])],
      e-> []),

  f2DTrExplicitRL := ARule(fCompose, [@(1,f2DTrExplicitR), @(2,f2DTrExplicitL)],
      e-> []),

  ParSeq_ParSeq := Rule([@(1, ParSeq), ..., @(2, ParSeq, x -> x.fb_cnt=@(1).val.fb_cnt), ...], 
      e-> ParSeq(e.fb_cnt, Flat(List(e.children(), x -> Cond( x _is ParSeq and x.fb_cnt=e.fb_cnt, x.children(), x))))),
));
