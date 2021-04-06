
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# ================================================================
# Diagonals
# ================================================================
Class(RulesDiag, RuleSet);

LeftPull  := [Diag, DiagCpxSplit, RCDiag];
RightPull := [Diag, DiagCpxSplit, RCDiag];
PullCont  := [SUM, BB, Buf, Inplace, Grp];
PullContL  := [SUM, BB, Buf, Inplace, Grp, NoDiagPullinRight];
PullContR  := [SUM, BB, Buf, Inplace, Grp, NoDiagPullinLeft];

RewriteRules(RulesDiag, rec(
 DiagPullInRight := ARule( Compose,
       [ @(1, LeftPull), @(2, PullContL) ],
  e -> [ ApplyFunc(ObjId(@2.val), List(@2.val.children(), c -> @1.val * c)) ]),

 DiagPullInLeft  := ARule( Compose,
       [ @(1, PullContR), @(2, RightPull) ],
  e -> [ ApplyFunc(ObjId(@1.val), List(@1.val.children(), c -> c * @2.val)) ]),

 DiagISumRight := ARule( Compose,
       [ @(1, LeftPull), @(2, ISum, canReorder) ],
  e -> [ ISum(@2.val.var, @2.val.domain, @1.val * @2.val.child(1)).attrs(@(2).val) ]),

 DiagISumLeft := ARule( Compose,
       [ @(1, ISum, canReorder), @(2, RightPull) ],
  e -> [ ISum(@1.val.var, @1.val.domain, @1.val.child(1) * @2.val).attrs(@(1).val) ]),

 DiagIComposeRight := ARule( Compose,
       [ @(1, LeftPull), @(2, ICompose) ],
  e -> [ ICompose(@2.val.var, @2.val.domain, @1.val * @2.val.child(1)).attrs(@(2).val) ]),

 DiagIComposeLeft := ARule( Compose,
       [ @(1, ICompose), @(2, RightPull) ],
  e -> [ ICompose(@1.val.var, @1.val.domain, @1.val.child(1) * @2.val).attrs(@(1).val) ]),


 DiagDataRight := ARule( Compose,
       [ @(1, LeftPull), @(2, Data, canReorder) ],
  e -> [ Data(@2.val.var, @2.val.value, @1.val * @2.val.child(1)).attrs(@(2).val) ]),

 DiagDataLeft := ARule( Compose,
       [ @(1, Data, canReorder), @(2, RightPull) ],
  e -> [ Data(@1.val.var, @1.val.value, @1.val.child(1) * @2.val).attrs(@(1).val) ]),

 DiagRecursStepRight := ARule( Compose,
       [ @(1, LeftPull), @(2, RecursStep) ],
  e -> [ RecursStep(@2.val.yofs, @2.val.xofs, @1.val * @2.val.child(1)) ]),

 DiagRecursStepLeft := ARule( Compose,
       [ @(1, RecursStep), @(2, RightPull) ],
  e -> [ RecursStep(@1.val.yofs, @1.val.xofs, @1.val.child(1) * @2.val) ])
));

Class(RulesDiagStandalone, RuleSet);
RewriteRules(RulesDiagStandalone, rec(
 # Gath * Diag
 CommuteGathDiag := ARule( Compose,
       [ @(1, Gath), @(2, Diag) ], # o 1-> 2->
  e -> [ Diag(fCompose(@2.val.element, @1.val.func)).attrs(@(2).val), @1.val ]),

 # Diag * Scat
 CommuteDiagScat := ARule( Compose,
       [ @(1, Diag), @(2, Scat) ], # <-1 <-2 o
  e -> [ @2.val, Diag(fCompose(@1.val.element, @2.val.func)).attrs(@(1).val) ]),

 # Gath * DiagCpxSplit
 CommuteGathDiagCpxSplit := ARule( Compose,
       [ @(1, Gath), @(2, DiagCpxSplit) ], # o 1-> 2->
  e -> [ DiagCpxSplit(fCompose(@2.val.element, fTensor(@1.val.func, fId(2)))).attrs(@(2).val), @1.val ]),

 # DiagCpxSplit * Scat
 CommuteDiagCpxSplitScat := ARule( Compose,
       [ @(1, DiagCpxSplit), @(2, Scat) ], # <-1 <-2 o
  e -> [ @2.val, DiagCpxSplit(fCompose(@1.val.element, fTensor(@2.val.func, fId(2)))).attrs(@(1).val) ]),

 # Gath * RCDiag
 CommuteGathRCDiag := ARule( Compose,
       [ [@(1, Gath), [@(0,fTensor), ..., [fId,@(2).cond(IsEvenInt)]]],
      @(4, RCDiag) ],
  e -> [ RCDiag(fCompose(@(4).val.element, @(0).val), @(4).val.post),
         @(1).val ]),

 # RCDiag * Scat
 CommuteRCDiagScat := ARule( Compose,
       [ @(4, RCDiag),
     [@(1, Scat), [@(0,fTensor), ..., [fId,@(2).cond(IsEvenInt)]]] ],
  e -> [ @(1).val,
         RCDiag(fCompose(@(4).val.element, @(0).val), @(4).val.post) ]),

 # Gath * IDirSum
 CommuteGathIDirSum := ARule( Compose,
       [ [@(1, Gath), [@(3,fTensor), ..., [fId,@(2)]]],
     [@(0, [PushL, PushLR]), @(4, IDirSum, e -> (@(2).val mod Rows(e.child(1))) = 0)] ],
  e -> let(ch := DropLast(@(3).val.children(), 1),
           ratio := @(2).val / Rows(@(4).val.child(1)),
           f := When(ratio=1, fTensor(ch), fTensor(Concatenation(ch, [fId(ratio)]))),
           j := @(4).val.var,
           jj := Ind(f.domain()),

           [ ObjId(@(0).val)(IDirSum(jj,      #SubstVars(@(4).val.child(1), rec((j.id) := f.at(jj)))),
                   Data(j, f.at(jj), SubstTopDown(@(4).val.child(1), @(5,fBase,e->e.params[2]=j),
                                       e -> fCompose(f, fBase(jj)))))),
             @(1).val ])
 ),

 # IDirSum * Scat
 CommuteIDirSumScat := ARule( Compose,
       [ [@(0,[PushR, PushLR]), @(4, IDirSum)],
         [@(1, Scat), [@(3,fTensor), ..., [fId,@(2).cond(e -> (e mod Cols(@(4).val.child(1))) = 0)]]]],

  e -> let(ch := DropLast(@(3).val.children(), 1),
           ratio := @(2).val / Cols(@(4).val.child(1)),
           f := When(ratio=1, fTensor(ch), fTensor(Concatenation(ch, [fId(ratio)]))),
           j := @(4).val.var,
           jj := Ind(f.domain()),

           [ @(1).val,
             ObjId(@(0).val)(IDirSum(jj,      #SubstVars(@(4).val.child(1), rec((j.id) := f.at(jj)))),
                   Data(j, f.at(jj), SubstTopDown(@(4).val.child(1), @(5,fBase,e->e.params[2]=j),
                                       e -> fCompose(f, fBase(jj)))))) ])
  ),

));

#--------------------------------------------------
#   new stuff for II

Class(RulesDiagII, RuleSet);
RewriteRules(RulesDiagII, rec(
    fCompose_diagXX := Rule([fCompose, @(1, [diagAdd, diagMul]), @(2)],
        e -> ApplyFunc(ObjId(@(1).val), List(@(1).val.children(), c->(fCompose(c, @(2).val ))))),

    simp1 := Rule([fCompose, @(1, II, e->e.params[2]=0), [fTensor, @(2, fId), @(3, fBase, e->@(1).val.params[3] = e.range() )] ],
        e -> II(@(2).val.domain(), 0, 1)),

    fBase_cond := Rule([fCompose, @(1, II), @(2, fBase)],
        e -> let(ii := @(1).val.params,
            diagCond(ivElOf(ivII(ii[1], ii[2], ii[3]), @(2).val.params[2]), fConst(1,1), fConst(1,0)))),

    tensor_cond := ARule(diagTensor, [@(1), @(2, diagCond)],
        e -> [diagCond(@(2).val.child(1), diagTensor(@(1).val, @(2).val.child(2)), diagTensor(@(1).val, @(2).val.child(3)))]),

    ivElOf_fold_val := Rule([ivElOf, @(1, ivII), @(2, Value)],
        e-> When(@2.val.v >= @1.val.params[2] and @2.val.v < @1.val.params[3], V(1), V(0))),

    diagCond_fold := Rule([diagCond, @(1, Value), @(2), @(3)],
        e-> When(@1.val.v = 1, @(2).val, @(3).val)),

    diagAdd_combine1 := ARule(diagAdd, [@(1, II), @(2, II, e->@(1).val.params[3]=e.params[2])],
        e -> [II(@(1).val.params[1], @(1).val.params[2], @(2).val.params[3])]),

    diagAdd_combine2 := ARule(diagAdd, [@(1, II), @(2, II, e->@(1).val.params[2]=e.params[3])],
        e -> [II(@(1).val.params[1], @(2).val.params[2], @(1).val.params[3])]),

    II_to_fConst := Rule(@(1, II, e->e.params[1]=e.params[3] and e.params[2]=0), e->fConst(@(1).val.params[1], 1)),

    diagMul0r := ARule(diagMul, [@(1), @(2, fConst, e->e.params[2] = 0)], e -> [@(2).val]),
    diagMul0l := ARule(diagMul, [@(1, fConst, e->e.params[2] = 0), @(2)], e -> [@(1).val]),

    diagAdd0r := ARule(diagAdd, [@(1), @(2, fConst, e->e.params[2] = 0)], e -> [@(1).val]),
    diagAdd0l := ARule(diagAdd, [@(1, fConst, e->e.params[2] = 0), @(2)], e -> [@(2).val]),

    Diag_fConst_to_I := Rule([@(1, Diag), @(2, fConst, e->e.params[2] = 1)],
        e->Prm(fId(@(2).val.params[1]))),

    adjust_ivII := Rule ([ivElOf, @(1, ivII), [@(2, add), @(3, var), @(4, Value)]],
        e -> let(iv := @(1).val.params, shift := @(4).val.v,
            ivElOf(ivII(iv[1], Maximum(0, iv[2]-shift), Maximum(0, iv[3]-shift)), @(3).val))),

    ivElOf_fold_var1 := Rule([ivElOf, @(1, ivII, e->e.params[2]=0),
        @(2, var, e->IsBound(e.range) and e.range <= @(1).val.params[3])],
        e-> V(1)),

    ivElOf_fold_var2 := Rule([ivElOf, @(1, ivII, e->e.params[3]=0), @(2, var)],
        e-> V(0)),
));


RulesDiagAll := MergedRuleSet(RulesDiag, RulesDiagStandalone);
