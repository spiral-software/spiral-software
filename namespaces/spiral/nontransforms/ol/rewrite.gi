
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


FunctionEquality:=(a,b)-> (a=b) or (a.n=b.n and a.N=b.N and let(func1:=a.lambda(), func2:=b.lambda(), 
	ForAll([1..a.N],e->func1.at(e).ev()=func2.at(e).ev())));

#######################################################
########  Redesigning the ruleset 
#######################################################

Class(OLScatFuseRules, RuleSet);
RewriteRules(OLScatFuseRules, rec(
        Scat_ScatAcc := ARule(Compose,[@(1,[Scat, ScatAcc]),@(2,ScatAcc)],
            e->[ScatAcc(fCompose(@(1).val.func,@(2).val.func))]),

        ScatAcc_Scat := ARule(Compose,[@(1,ScatAcc),@(2,[Scat, ScatAcc])],
            e->[ScatAcc(fCompose(@(1).val.func,@(2).val.func))]),

        
        ICScatAcc_Scat := ARule(Compose,[@(1,ICScatAcc),@(2,[Scat, ICScatAcc])],
            e->[ICScatAcc(fCompose(@(1).val.func,@(2).val.func))]),

        Prm_ScatQuestionMark  := ARule(Compose, [ @(1, Prm), @(2, ScatQuestionMark) ],
            e -> [ ScatQuestionMark(fCompose(@(1).val.inverse, @(2).val.func)) ]),


        Prm_ICScatAcc := ARule(Compose, [ @(1, Prm), @(2, ICScatAcc) ],
            e -> [ ICScatAcc(fCompose(@(1).val.inverse, @(2).val.func)) ]),


        ScatQuestionMark_ScatQuestionMark := 
        ARule(Compose, [ @(1, ScatQuestionMark), @(2, ScatQuestionMark) ], 
            e -> [ ScatQuestionMark(fCompose(@(1).val.func, @(2).val.func)) ]),

        Kill_NoPullId :=
        ARule(Compose,[[NoPull,[Prm,@,fId,fId]]],e->[]),

        Kill_ComposeIdL :=
        ARule(Compose,[@(1),[Prm,@,fId,fId]],e->[@(1).val]),

        Kill_ComposeIdR :=
        ARule(Compose,[[Prm,@,fId,fId],@(1).val],e->[@(1).val]),

#        Rule([@(1,Compose,e->Length(e.rChildren())>1),...,[Prm,@,fId,fId],...],e->Error("kill id")),

        

));

Class(OLVectorScatFuseRules, RuleSet);
RewriteRules(OLVectorScatFuseRules, rec(
        VScat_VConstruct := 
        ARule(Compose, [ @(1, VScat), @(2, [VScatAcc,VScatQuestionMark], e->@(1).val.v=e.v) ], 
            e -> [ ObjId(@(2).val)(fCompose(@(1).val.func, @(2).val.func), @(1).val.v) ]),

        VScatQuestionMark_VConstruct := 
        ARule(Compose, [ @(1, VScatQuestionMark), @(2, [VScat,VScatAcc,VScatQuestionMark], e->@(1).val.v=e.v) ], 
            e -> [ ObjId(@(1).val)(fCompose(@(1).val.func, @(2).val.func), @(1).val.v) ]),
        
        VScatAcc_VConstruct := 
        ARule(Compose, [ @(1, VScatAcc), @(2, [VScat,VScatAcc], e->@(1).val.v=e.v) ], 
            e -> [ VScatAcc(fCompose(@(1).val.func, @(2).val.func), @(1).val.v) ]),
        
        ##The following rules are copies of the ones in the subvector system
        ScatQuestionMark_VConstruct := ARule(Compose, [ @(1, ScatQuestionMark), @(2, [VScat,VScatQuestionMark]) ],
            e -> [ VScat_svQuestionMark(@(1).val.func, getV(@(2).val), 1), @(2).val ]),

        ScatAcc_VConstruct := ARule(Compose, [ @(1, ScatAcc), @(2, [VScat,VScatAcc, VScatQuestionMark]) ],
            e -> [ VScat_svAcc(@(1).val.func, getV(@(2).val), 1), @(2).val ]),

    VScat_svQuestionMark__VScat := ARule(Compose, [@(1,[VScat_svQuestionMark,VScat_svAcc]), @(2,VScat)],
    e -> let(func := @(1).val.func,  vfunc := @(2).val.func,
             v    := getV(@(1).val), sv    := @(1).val.sv,
        [ ObjId(@(1).val)(fCompose(func, fTensor(vfunc, fId(v/sv))), v, sv) ])),

    GathScat_svQuestionMark_fTensor := Rule([@(1, [VGath_sv, VScat_svQuestionMark,VScat_svAcc]),
                            [@(2,fTensor), ..., [fId, @(3).cond(e -> Gcd(@(1).val.v/
@(1).val.sv, e) = @(1).val.v)]]],
     e -> let(v := @(1).val.v, sv := @(1).val.sv,
          n := @(3).val,   gcd := Gcd(v / sv, n),
          ObjId(@(1).val)(fTensor(DropLast(@(2).val.children(), 1), fId(n/gcd)), v, sv*gcd))),

    GathScat_sv_H := Rule([@(1, [VGath_sv, VScat_sv, VScat_svAcc]), [H, @(2),@(3).cond(e -> Gcd(@(1).val.v/
@(1).val.sv, e) = @(1).val.v),@(4),@(5)]],
            e->let(v := @(1).val.v, sv := @(1).val.sv,
          n := @(3).val,   gcd := Gcd(v / sv, n),
          mydiv:=When(IsSymbolic(@(4).val),idiv(@(4).val,gcd),@(4).val/gcd),
          ObjId(@(1).val)(H(@(2).val/gcd,n/gcd,mydiv,@(5).val), v, sv*gcd))),
    
        VScat_svQuestionMark__VScatQuestionMark := ARule(Compose, [@(1,[VScat_svQuestionMark,VScat_svAcc]), @(2,VScatQuestionMark)],
            e -> let(func := @(1).val.func,  vfunc := @(2).val.func,
                v    := getV(@(1).val), sv    := @(1).val.sv,
                [ ObjId(@(1).val)(fCompose(func, fTensor(vfunc, fId(v/sv))), v, sv) ])),
   
        VScat_svAcc__VScatAcc := ARule(Compose, [@(1,[VScat_svQuestionMark,VScat_svAcc]), @(2,VScatAcc)],
            e -> let(func := @(1).val.func,  vfunc := @(2).val.func,
                v    := getV(@(1).val), sv    := @(1).val.sv,
                [ VScat_svAcc(fCompose(func, fTensor(vfunc, fId(v/sv))), v, sv) ])),


        VScat_svQuestionMark_to_VScatQuestionMark := 
        Rule(@(1,VScat_svQuestionMark,e->getV(e)=e.sv), e->VScatQuestionMark(@(1).val.func, @(1).val.v)),   

        VScat_svAcc_to_VScatAcc := 
        Rule(@(1,VScat_svAcc,e->getV(e)=e.sv), e->VScatAcc(@(1).val.func, @(1).val.v)),   

));

Class(OLVectorPropagateRules, RuleSet);
RewriteRules(OLVectorPropagateRules, rec(
        VTensorOL_ScatAcc := 
        Rule([@(1,VTensor_OL),[@(2,Compose),@(3,ScatAcc),...]],
            e->VScatAcc(@(3).val.func,@(1).val.vlen)*
            VTensor_OL(Compose(Drop(@(2).val._children,1)),@(1).val.vlen)),

        VTensorOL_ScatQuestionMark := 
        Rule([@(1,VTensor_OL),[@(2,Compose),@(3,ScatQuestionMark),...]],
            e->VScatQuestionMark(@(3).val.func,@(1).val.vlen)*
            VTensor_OL(Compose(Drop(@(2).val._children,1)),@(1).val.vlen)),
        
        VTensorOL_Cross := 
        Rule([@(1,VTensor_OL),[@(2,Compose),...,@(3,Cross)]],
            e->VTensor_OL(Compose(DropLast(@2.val._children,1)),@(1).val.vlen)*
            Cross(List(@(3).val._children,t->VTensor(t,@(1).val.vlen)))),
        
        VTensorOL_ISum := Rule([@(1, VTensor_OL), @(2, [ISum])],
            e -> let(s := @(2).val, CopyFields(s,
                    rec(_children := List(s.children(), c->VTensor_OL(c, @(1).val.vlen)))))),

        VTensor_I :=
        Rule([@(1,VTensor),@(2,I)],
            e->VTensor(Prm(fId(@(2).val.dimensions[1])),@(1).val.vlen)),

));

OLVectorPropagateRuleset := MergedRuleSet(OLVectorPropagateRules, StandardSumsRules, OLScatFuseRules, 
    OLVectorScatFuseRules, RulesVec, RulesPropagate);

Class(OLPushScatQuestionMarkInRules, RuleSet);
RewriteRules(OLPushScatQuestionMarkInRules, rec(
        ScatQuestionMark_NoPullRight := ARule(Compose, [@(1, [ScatQuestionMark,VScatQuestionMark]),@(2,NoPullRight)],
            e -> [ NoPullRight(@(1).val * @(2).val.rChildren()[1])]),

        ScatQuestionMark_Isum := ARule(Compose, [@(1, [ScatQuestionMark,VScatQuestionMark]), @(2,ISum)],
            e-> [ ISum(@(2).val.var,@(2).val.domain,Compose(@(1).val,@(2).val._children))]),
));

OLPushScatQuestionMarkInRuleset := MergedRuleSet(OLPushScatQuestionMarkInRules, StandardSumsRules, OLScatFuseRules, 
    OLVectorScatFuseRules, RulesVec, RulesPropagate);

Class(OLAlreadyInitializedScatQuestionMarkRules, RuleSet);
RewriteRules(OLAlreadyInitializedScatQuestionMarkRules, rec(
        AlreadyInitializedScatQuestionMark := Rule(@(1, [ScatQuestionMark]),
            e-> ScatAcc(@(1).val.func)),

        AlreadyInitializedVScatQuestionMark := Rule(@(1, [VScatQuestionMark]),
            e-> VScatAcc(@(1).val.func,@(1).val.v)),
));

Class(OLScatQuestionMarkToScat, RuleSet);
RewriteRules(OLScatQuestionMarkToScat, rec(
        ScatQuestionMarkToScat := Rule(@(1, [ScatQuestionMark]),
            e-> Scat(@(1).val.func)),

        VScatQuestionMarkToScat := Rule(@(1, [VScatQuestionMark]),
            e-> VScat(@(1).val.func,@(1).val.v)),
));




Class(OLQuickAndDirtyHackForCodelet, RuleSet);
RewriteRules(OLQuickAndDirtyHackForCodelet, rec(
        ScatAcc_NoPullRight := ARule(Compose, [@(1, [ScatAcc,VScatAcc]),@(2,NoPullRight)],
            e -> [ NoPullRight(@(1).val * @(2).val.rChildren()[1])]),

        ScatAcc_Isum := ARule(Compose, [@(1, [ScatAcc, VScatAcc]), @(2,ISum)
],
            e-> [ ISum(@(2).val.var,@(2).val.domain,Compose(@(1).val,@(2).val._children))]),
));

OLAlreadyInitializedScatQuestionMarkRuleset := MergedRuleSet(OLAlreadyInitializedScatQuestionMarkRules,
    StandardSumsRules, OLScatFuseRules, OLVectorScatFuseRules, RulesVec, RulesPropagate);

Class(OLSplitScatQuestionMarkRules, RuleSet);
RewriteRules(OLSplitScatQuestionMarkRules, rec(

        SplitScatQuestionMark := Rule(@(1, [ScatQuestionMark]),
            e-> ScatInit(@(1).val.func , [],ScatAcc(@(1).val.func))),

        PullScatInitOutOfCompose := Rule([@(1,Compose),@(2,ScatInit),...],
            e-> ScatInit(@(2).val.func,@(2).val.cond,
                Compose(@(2).val._children,Drop(@(1).val._children,1)))),

        PullScatInitOutOfISum := Rule([@(1,ISum),@(2,ScatInit)],
            e->When(@(1).val.var in @(2).val.func.free(), 
                ScatInitProbe(@(2).val.func,@(2).val.cond,
                    ISum(@(1).val.var,@(1).val.domain,
                        ScatInitFixed(@(2).val.func,@(2).val.cond,@(2).val._children))), 
                ScatInit(@(2).val.func,@(2).val.cond,
                    ISum(@(1).val.var,@(1).val.domain,@(2).val._children)))),

        PullScatInitProbeOutOfCompose := Rule([@(1,Compose),@(2,ScatInitProbe),...],
            e-> ScatInitProbe(@(2).val.func,@(2).val.cond,
                Compose(@(2).val._children,Drop(@(1).val._children,1)))),

        PullScatInitProbeOutOfISum := Rule([@(1,ISum),@(2,ScatInitProbe)],
            e->When(@(1).val.var in @(2).val.func.free(),
                ScatInitProbe(@(2).val.func,@(2).val.cond,
                    ISum(@(1).val.var,@(1).val.domain,@(2).val._children)),
                ScatInitProbe(@(2).val.func,Concatenation([@(1).val.var],@(2).val.cond),
                    ISum(@(1).val.var,@(1).val.domain,@(2).val._children)))),


));
OLSplitScatQuestionMarkRuleset := MergedRuleSet(OLSplitScatQuestionMarkRules, StandardSumsRules, OLScatFuseRules, 
    OLVectorScatFuseRules, RulesVec, RulesPropagate);

Class(OLScatProbeMergeRules, RuleSet);
RewriteRules(OLScatProbeMergeRules, rec(

        PushScatInitProbeInsideSum := Rule([@(1,ScatInitProbe),@(2,ISum),@],
            e->ISum(@(2).val.var,@(2).val.domain,
                ScatInitProbe(@(1).val.func,@(1).val.cond,Compose(@(2).val._children)))),

        PushScatInitProbeInsideCompose := Rule([@(1,ScatInitProbe),@(2,Compose),@],
            e-> Compose([ScatInitProbe(@(1).val.func,@(1).val.cond,@(2).val._children[1]),
                    Drop(@(2).val._children,1)])),

        ScatInit_Cross := ARule(Compose, [@(1,[ScatInit]), @(2, [Cross])],
            e-> [ ScatInit(@(1).val.func,@(1).val.cond,Compose(@(1).val._children,@(2).val))]),

        ScatInitProbe_ScatInitFixed := Rule([@(1,ScatInitProbe),@(2,ScatInitFixed),@],
            e->When(@(1).val.func=@(2).val.func, 
                ScatInit(@(1).val.func,@(1).val.cond,@(2).val._children),
                Error("ScatInitProbe doesn t remerge with ScatInitFixed, how did that happen?"))),

        PushScatInitInsideCompose := Rule([@(1,ScatInit),@(2,Compose),@],
            e-> Compose([ScatInit(@(1).val.func,@(1).val.cond,@(2).val._children[1]),
                    Drop(@(2).val._children,1)])),

        ScatInit_ScatAcc := 
        Rule([@(1,ScatInit,e->ObjId(e._children)=ScatAcc and e.func=e._children.func and e.cond=[]),@(2,ScatAcc),@],
            e-> Scat(@(1).val.func)),
));

OLScatProbeMergeRuleset := MergedRuleSet(OLScatProbeMergeRules, StandardSumsRules, OLScatFuseRules, 
    OLVectorScatFuseRules, RulesVec, RulesPropagate);

Class(OLScatAccPeelRules, RuleSet);
RewriteRules(OLScatAccPeelRules, rec(
    ScatInitPeel := Rule(
        [@(1,ScatInit, e->e.cond=[]),
            [@(2,ISum), [@(3,Compose), 
                    @(4,ScatAcc,e->FunctionEquality(e.func,@(1).val.func)), ... ]],@],

	e-> let(v := @(2).val.var, newv := Ind(v.range-1), 
#x            Buf(Scat(fId(@(1).val.dims()[1]))) * 
            SUM(SubstVars(
                    Scat(@(4).val.func) * Compose(Drop(Copy(@(3).val.children()),1)),
                    rec((v.id) := V(0))),
                Cond(@(2).val.domain=2,
                    SubstVars(@(2).val.child(1),rec((v.id) := V(1))),
                    SubstVars(
                        ISum(newv, @(2).val.domain-1, @(2).val.child(1)),
                        rec((v.id) := newv+1)))
            )))
));

OLScatAccPeelRuleset := MergedRuleSet(OLScatAccPeelRules,StandardSumsRules, OLScatFuseRules, 
    OLVectorScatFuseRules, RulesVec, RulesPropagate);


Class(OLCrossPullInRules, RuleSet);
RewriteRules(OLCrossPullInRules, rec(
        PullInSMPSumRight := ARule( Compose, [ @(1, [Prm, Scat]), @(2, [SMPSum]) ],
            e -> [ CopyFields(@(2).val, rec(
             _children :=  List(@(2).val._children, c -> @(1).val * c),
             dimensions := [Rows(@(1).val), Cols(@(2).val)] )) ]),

        SMPSum_Cross  := ARule(Compose, [ @(1, [SMPSum]), @(2, [Cross]) ],
            e -> [ ObjId(@(1).val)(@(1).val.p,@(1).val.var, @(1).val.domain, @(1).val.child(1) * @(2).val) ]),

        ISum_Cross  := ARule(Compose, [ @(1, [ISum, ISumAcc]), @(2, [Cross]) ],
            e -> [ ObjId(@(1).val)(@(1).val.var, @(1).val.domain, @(1).val.child(1) * @(2).val) ]),

        SUM_Cross  := ARule(Compose, [ @(1, [SUM]), @(2, [Cross]) ],
            e -> [ ObjId(@(1).val)(List(@(1).val._children,x->x*@(2).val)) ]),

        ScatAcc_RecursStep := ARule(Compose, [ @(1, [ScatAcc]), @(2, [RecursStep]) ],
            e -> [ RecursStep(@2.val.yofs, @2.val.xofs, 
                     @(1).val* @2.val.child(1)).attrs(@(1).val) ]),

        RecursStep_Cross := ARule(Compose, [ @(1, [RecursStep]), @(2, [Cross]) ],
            e -> [ RecursStep(@1.val.yofs, @1.val.xofs, 
                    @1.val.child(1) * @(2).val).attrs(@(1).val) ]),

        Cross_Cross := ARule(Compose, [ @(1, [Cross, CrossBlockTop]), @(2, [Cross]) ],
            e -> [ ObjId(@(1).val)(@(1).val.child(1) * @(2).val.child(1),
                    @(1).val.child(2) * @(2).val.child(2)).attrs(@(1).val) ]),

        CrossBlockTopPass := Rule(@(1,CrossBlockTop,e->not((ObjId(e.rChildren()[2])=Prm) and 
                ObjId(e.rChildren()[2].rChildren()[2])=fId and 
                ObjId(e.rChildren()[2].rChildren()[3])=fId)),
        e->Cross(Prm(fId(@(1).val.rChildren()[1].dims()[1])),@(1).val.rChildren()[2])*
            CrossBlockTop(@(1).val.rChildren()[1],
                Prm(fId(@(1).val.rChildren()[2].dims()[2])))
        )
));

OLCrossPullInRuleset := MergedRuleSet(OLCrossPullInRules, StandardSumsRules, OLScatFuseRules, 
    OLVectorScatFuseRules, RulesVec, RulesPropagate);


Class(OLAfterCrossPullInRuleset, RuleSet);
RewriteRules(OLAfterCrossPullInRuleset, rec(
        Cross_CompatibleISum  := Rule([Cross, @(1, ISum), @(2, ISum,e->e.domain=@(1).val.domain) ],
            function(e)
              local varmap, var;
              varmap:=tab();
              var:=@(2).val.var;
              varmap.(var.id):=@(1).val.var;
              return ISum(@(1).val.var, @(1).val.domain, 
                  Cross(@(1).val.rChildren()[1],SubstVars(@(2).val.rChildren()[1],varmap)));
            end)
));


Class(OLTearCrossRules, RuleSet);
RewriteRules(OLTearCrossRules, rec(
#        FullTearVGath_dup:=ARule(Cross,[@(1,VGath_dup)],e->[VGath(@(1).val.func,@(1).val.v)*VGath_dup(fId(@(1).val.func.range()),@(1).val.v)]),

#HACK, of course
        HalfTearVGath_dup:=Rule([@(1,VGath_dup),[@(2,fTensor),@(3,fBase),fBase,...]],e->
            let(func:=fTensor(Drop(Copy(@(2).val.rChildren()),1)),
            VGath(func,@(1).val.v)*
            VGath_dup(fTensor(@(3).val,fId(func.range())),@(1).val.v))),


        TearCross:=ARule(Compose,[@(1,Cross,e->e.hasRightScat() and e.hasLeftGath())],
            e->[Copy(@(1).val.splitCross()[1]),Copy(@(1).val.splitCross()[2])]),

        PushCrossOutOfIsum := 
        Rule([@(1,ISum), @(2,Compose,e->ObjId(Last(e.rChildren()))=Cross and
                    Last(e.rChildren()).hasRightScat() and not Last(e.rChildren()).hasLeftGath())],
            e->Compose(ISum(@(1).val.var, @(1).val.domain, 
                    Compose(DropLast(@(2).val.rChildren(),1))),Last(@(2).val.rChildren()))),

#        PushCrossOutOfSMPSum := 
#        Rule([@(1,SMPSum), @(2), @(3,Compose,e->ObjId(Last(e.rChildren()))=Cross and
#                    Last(e.rChildren()).hasRightScat() and not Last(e.rChildren()).hasLeftGath())],
#
#            e->Compose(SMPSum(@(2).val,@(1).val.var, @(1).val.domain, 
#                    Compose(DropLast(@(3).val.rChildren(),1))),Last(@(3).val.rChildren()))),

        CrossConditionalMerge := 
        ARule(Compose, [ @(1, Cross), @(2, Cross, e->e.hasRightScat() and 
                    @(1).val.hasRightScat() and not(@(1).val.hasLeftGath()) and not(e.hasLeftGath())) ],
            e -> [ ObjId(@(1).val)(@(1).val.child(1) * @(2).val.child(1),@(1).val.child(2) * 
                    @(2).val.child(2)).attrs(@(1).val) ]),
));

OLTearCrossRuleset := MergedRuleSet(OLTearCrossRules, StandardSumsRules, OLScatFuseRules, 
    OLVectorScatFuseRules, RulesVec);

Class(OLSingleComposeRuleset, RuleSet);
RewriteRules(OLSingleComposeRuleset, rec(
        SingleCompose := Rule([@(1,Compose),@(2)],e->@(2).val),
));

Class(OLDropVGath_dupRuleset, RuleSet);
RewriteRules(OLDropVGath_dupRuleset, rec(
        DropVGath_dup := Rule(@(1,VGath_dup),e->VGath(@(1).val.func,@(1).val.v))
));

Class(OLDropSMPSumRuleset, RuleSet);
RewriteRules(OLDropSMPSumRuleset, rec(
        DropSMPSum := Rule(@(1,SMPSum),e->ISum(@(1).val.var,@(1).val.domain,@1.val.rChildren()[2]))
));


#########  HACCCK !!!
Class(OLMagicVUnrollDupRuleset, RuleSet);
RewriteRules(OLMagicVUnrollDupRuleset, rec(
#        MagicVGath_dup := Rule(@(1,VGath_dup,e->e.func.domain()>128 and e.func.domain() mod (32*4)=0),
#            e->let(itid:=Ind(4),idx:=Ind(@(1).val.func.domain()/(32*4)),
#                SMPSum(4,itid,4,
#                    ISum(idx,@(1).val.func.domain()/(32*4),
#                        VScat(fTensor(fBase(itid),fBase(idx),fId(32)), 2)*
#                        VGath_dup(fTensor(fBase(itid),fBase(idx),fId(32)), 2))),

        MagicVGath_dup := Rule(@(1,VGath_dup,e->e.func.domain()>32),
            e->@(1).val.toloop(32))
));

   

#Class(OLLiftBarrierRuleset, RuleSet);
#RewriteRules(OLLiftBarrierRuleset, rec(
#        EliminateBarrier := ARule(Compose, [@(1,RewriteBarrier),@(2)],
#            e->[@(2).val]),
#));

OLDefaultStrategy := [TerminateSymSPL,RulesTerm];

#################################################################
Class(OLRulesCode, RuleSet);
RewriteRules(OLRulesCode, rec(

        #Cross must not be BBized without neighbours if it has some identities 
        CrossBB_in := Rule([BB,[@(1,Cross),...,[Prm,@,fId,fId],...]],
            e->Cross(List(@(1).val.rChildren(),x->BB(x)))
        ),

        DoubleBB := Rule([BB,@(1,BB)],e->@(1).val),

        #BB should go out if the inner are not Ids (simplified to perms in this case)
        Cross_BB_out := Rule([Cross, @(1, BB,e->not(ObjId(e.rChildren()[1])=Prm)), @(2, BB,e->not(ObjId(e.rChildren()[1])=Prm)) ], 
            e -> BB(Cross(@(1).val.child(1),@(2).val.child(1)))),

        Drop_NoPull := Rule(@(1,NoPull),e->@(1).val.rChildren()[1]),

));



Class(OLRulesBufferFinalize, RuleSet);
RewriteRules(OLRulesBufferFinalize, rec(
    FinalizeNoPull := Rule(@(1, [NoPull, NoPullLeft, NoPullRight]), i->@1.val.child(1)),
    DropCrossBlockTop := Rule(@(1,CrossBlockTop), e->Cross(@1.val.rChildren()))
));


#RewriteRules(RulesStrengthReduce, rec(
# vdupmultsR:= Rule([mul, @(1), @(2).cond(e->ObjId(e.t)=TVect and ObjId(@(1).val.t)<>TVect)], e->mul(vdup(@(1).val, @(2).val.t.size), @(2).val)),
#
#vdupmultsL:= Rule([mul, @(1), @(2).cond(e->ObjId(@(1).val.t)=TVect and ObjId(e.t)<>TVect)], e->mul(@(1).val,vdup(@(2).val, @(1).val.t.size))),
#
#));



Class(OLPushScatAccRules,RuleSet);

#ScatAcc that are pushed inside sums MUST merge or else things get added twice or more...
RewriteRules(OLPushScatAccRules, rec(
        PushScatAccInsideISum := ARule(Compose, [@(1, ScatAcc), @(2,ISum)],
            e-> [ ISum(@(2).val.var,@(2).val.domain,Compose(@(1).val,@(2).val._children))]),

        PushScatAccInsideSUM := ARule(Compose, [@(1, ScatAcc), @(2,SUM)],
            e-> let(childs:=List(@(2).val._children,e->@(1).val*e), [ SUM(childs)])),

        PushICScatAccInsideISum := ARule(Compose, [@(1, ICScatAcc), @(2,ISum)],
            e-> [ ISum(@(2).val.var,@(2).val.domain,Compose(@(1).val,@(2).val._children))]),

        PushICScatAccInsideSUM := ARule(Compose, [@(1, ICScatAcc), @(2,SUM)],
            e-> let(childs:=List(@(2).val._children,e->@(1).val*e), [ SUM(childs)])),


));
OLPushScatAccRuleset := MergedRuleSet(OLPushScatAccRules, StandardSumsRules, OLScatFuseRules, 
    OLVectorScatFuseRules, RulesVec, RulesPropagate);   














