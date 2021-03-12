
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


ImportAll(paradigms.vector);
ImportAll(paradigms.smp);

Class(RulesFuncSimpScratch,RulesFuncSimp);
RewriteRules(RulesFuncSimpScratch, rec(
    #L(mn,m) o (fBase(m,j) (X) f) -> f (X) fBase(j,m)
    LTensorFlip := ARule(fCompose,
       [ @(1,L), [ @(3,fTensor), @(2).cond(e->range(e) = @(1).val.params[2]), ...] ], e -> [ fTensor(Copy(Drop(@(3).val.children(), 1)), Copy(@(2).val)) ] ),
));

Class(RulesSumsScratch, RulesSums);
RewriteRules(RulesSumsScratch, rec(
 
    merge_LSKernel := ARule(Compose, [@(1, LSKernel), @(2, LSKernel)],
        e -> [LSKernel(Compose([@(1).val.child(1), @(2).val.child(1)]), @(1).val.mergeInfo(@(1).val.info, @(2).val.info))]),
    
    mark_DMAGS := Rule([Compose, @(1, Scat), @(2, LSKernel), @(3, Gath)], e -> DMAScat(@(1).val.func) * @(2).val * DMAGath(@(3).val.func) ),
    
    mark_SWPSum := Rule([DMAFence, [@(1, ISum), @(2)]], e -> let(
            var := @(1).val.var,
            dom := @(1).val.domain,
            comp := Compose(@(2).val._children),
            DMAFence(SWPSum(var, dom, comp)))),

    Scat_DMAFence := ARule( Compose,
       [ @(1, [Scat, Prm, Diag]),
         @(2, [DMAFence, ISum]) ],
        e -> [ CopyFields(@(2).val, rec(
             _children :=  List(@(2).val._children, c -> @(1).val * c),
             dimensions := [Rows(@(1).val), Cols(@(2).val)] )) ]),

    Gath_DMAFence := ARule( Compose,
       [ @(1, [DMAFence, ISum]),
         @(2, [Gath, Prm, Diag]) ],
        e -> [ CopyFields(@(1).val, rec(
                _children := List(@(1).val._children, c -> c * @(2).val),
                dimensions := [Rows(@(1).val), Cols(@(2).val)] )) ]),

    DMAFence_fold := Rule([@(1, DMAFence), @(2, DMAFence)],
        e -> @(2).val),
    
    Diag_fPrecompute := Rule([Diag, @(2, fPrecompute)], e -> Diag(@(2).val._children[1])),

    Diag_LSKernel := ARule( Compose,
       [ @(1, Diag), @(2, LSKernel) ],
        e -> [LSKernel(@(1).val * @(2).val.child(1), CopyFields(@(2).val.info, rec(opcount := @(2).val.info.opcount + 6 * Rows(@(1).val))))]),

    LSKernel_Diag := ARule( Compose,
        [ @(1, LSKernel), @(2, Diag) ],
        e -> [
            LSKernel(@(1).val.child(1) * @(2).val, 
                CopyFields(@(1).val.info, 
                    rec(opcount := @(1).val.info.opcount + 6 * Rows(@(2).val))
                )
            )
        ]
    )
));


Class(RulesRCScratch, RulesRC);
RewriteRules(RulesRCScratch, rec(
   RC_ISum := Rule([RC, @(1, ISum)], 
        e -> let(s := @(1).val, ISum(s.var, s.domain, RC(s.child(1))))),
    
   RC_SWPSum := Rule([RC, @(1, SWPSum)],
        e -> let(s := @(1).val, SWPSum(s.var, s.domain, RC(s.child(1))))),

   RC_DMAFence := Rule([RC, @(1, DMAFence)],
        e -> let(s:=@(1).val, DMAFence(RC(s.child(1))))),

   RC_LSKernel := Rule([RC, @(1, LSKernel)],
        e -> let(s:=@(1).val, LSKernel(RC(s.child(1)), s.info))),

   RC_DMAGath := Rule([RC, @(1, DMAGath)], e -> DMAGath(fTensor(@(1).val.func, fId(2)))),

   RC_DMAScat := Rule([RC, @(1, DMAScat)], e -> DMAScat(fTensor(@(1).val.func, fId(2)))),
));

DMAGath.needInterleavedRight := False;
DMAGath.needInterleavedLeft := False;
DMAGath.cannotChangeDataFormat := True;
DMAGath.totallyCannotChangeDataFormat := True;

DMAScat.needInterleavedRight := False;
DMAScat.needInterleavedLeft := False;
DMAScat.cannotChangeDataFormat := True;
DMAScat.totallyCannotChangeDataFormat := True;
