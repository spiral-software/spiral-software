
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# VGath_sv/VScat_sv may have bigger dimensions than their function domain so we need padding  
# to have Diag with correct dimensions when commuting diag through VGath_sv and VScat_sv
_fComposeVGathScatSv_Diag := (m, dim, d) -> let( 
    g := fTensor(m.func, fId(m.sv)),
    mdim := dim(m),
    rem := Cond(m.rem _is Unk, g.domain() mod mdim, m.rem=0, 0, m.v - m.rem * m.sv),
    f := fCompose(d.element, g),
    When(rem = 0, f, diagDirsum(f, fConst(TReal, mdim-rem, 0.0)))
);

Class(RulesVDiag, RuleSet);
RulesVDiag.__transparent__ := [ [Compose, [vRC]] ];  # creates extra rules, A*B->C is made into vRC(A)*vRC(B) -> vRC(C)

RewriteRules(RulesVDiag, rec(
    DiagConst_AnyScat := ARule(Compose, [[Diag, [fConst, @(1), @(2), @(3)]], @(4).cond(IsVScat)],
	e -> [ @(4).val, VDiag(fConst(@(1).val, Cols(@(4).val), @(3).val), getV(@(4).val))]),

    AnyGath_DiagConst := ARule(Compose, [@(4).cond(IsVGath), [Diag, [fConst, @(1), @(2), @(3)]]],
	e -> [ VDiag(fConst(@(1).val, Rows(@(4).val), @(3).val), getV(@(4).val)), @(4).val]),

    DiagConst_NoDiagPullin := ARule(Compose, [[Diag, [fConst, @(1), @(2), @(3)]], @(4, NoDiagPullin)],
	e -> [ @(4).val, Diag(fConst(@(1).val, Cols(@(4).val), @(3).val))]),

    # VGath * IDirSum
     CommuteVGathIDirSum := ARule( Compose,
	 [ [@(1, VGath), @(3)],
	   [@(0, [PushL, PushLR]), @(4, IDirSum, e -> (@(1).val.v mod Rows(e.child(1))) = 0)] ],
       e -> let(
             ratio := @(1).val.v / Rows(@(4).val.child(1)),
             f := When(ratio=1, @(3).val, fTensor(@(3).val, fId(ratio))),
             j := @(4).val.var,
             jj := Ind(f.domain()),

             [ ObjId(@(0).val)(IDirSum(jj,
                   Data(j, f.at(jj), SubstTopDown(@(4).val.child(1), @(5,fBase,e->e.params[2]=j),
                                       e -> fCompose(f, fBase(jj)))))),
               @(1).val ])
    ),

    # IDirSum * VScat
    CommuteIDirSumVScat := ARule( Compose,
       [ [@(0,[PushR, PushLR]), @(4, IDirSum)],
         [@(1, VScat, e -> (e.v mod Cols(@(4).val.child(1)))=0), @(3)] ],

       e -> let(
             ratio := @(1).val.v / Cols(@(4).val.child(1)),
             f := When(ratio=1, @(3).val, fTensor(@(3).val, fId(ratio))),
             j := @(4).val.var,
             jj := Ind(f.domain()),

             [ @(1).val,
               ObjId(@(0).val)(IDirSum(jj,
                   Data(j, f.at(jj), SubstTopDown(@(4).val.child(1), @(5,fBase,e->e.params[2]=j),
                                       e -> fCompose(f, fBase(jj)))))) ])
    ),

    # Gath * RCDiag
    CommuteVGathVRCDiag_fTensor := ARule( Compose,
       [ [@(1, VGath), [@(0,fTensor), ..., [fId, @(2).cond(_isEven)]]], @(4, VRCDiag) ],
       e -> [ VRCDiag(fCompose(@(4).val.element, @(0).val), @(4).val.v), @(1).val ]),

    # Gath * RCDiag
    CommuteVGathVRCDiag_H := ARule( Compose,
       [ [@(1, VGath), [@(2, H), @, @.cond(_isEven), @.cond(_isEven), _1]], @(4, VRCDiag) ],
       e -> [ VRCDiag(fCompose(@(4).val.element, @(2).val), @(4).val.v), @(1).val ]),

    # RCVDiag * VScat
    CommuteRCVDiagVScat := ARule( Compose,
       [ @(1, RCVDiag), @(2, VScat) ], # <-1 <-2 o
        e -> [ @2.val, RCVDiag(fCompose(@1.val.element, @2.val.func), @2.val.v).attrs(@(1).val) ]),

    # RCVDiagSplit * VScat
    CommuteRCVDiagSplitVScat := ARule( Compose,
       [ @(1, RCVDiagSplit), @(2, VScat) ], # <-1 <-2 o
        e -> [ @2.val, RCVDiagSplit(fCompose(@1.val.element, fTensor(@2.val.func, fId(2))), @2.val.v).attrs(@(1).val) ]),


    # VRCDiag * VScat
    CommuteVRCDiagVScat_fTensor := ARule( Compose, 
        [ @(4, VRCDiag), [@(1, VScat), [@(0,fTensor), ..., [fId, @(2).cond(_isEven)]]] ],
        e -> [ @(1).val, VRCDiag(fCompose(@(4).val.element, @(0).val), @(4).val.v) ]),

    # VRCDiag * VScat
    CommuteVRCDiagVScat_H := ARule( Compose, 
        [ @(4, VRCDiag), [@(1, VScat), [@(2, H), @, @.cond(_isEven), @.cond(_isEven), _1]] ],
        e -> [ @(1).val, VRCDiag(fCompose(@(4).val.element, @(2).val), @(4).val.v) ]),

    # VGath * Diag
    CommuteVGathDiag := ARule( Compose,
       [ @(1, VGath), @(2, Diag) ], # o 1-> 2->
        e -> [ VDiag(fCompose(@2.val.element, fTensor(@1.val.func, fId(@1.val.v))), @1.val.v).attrs(@(2).val), @1.val ]),

    # VGath * VDiag
    CommuteVGathVDiag := ARule( Compose,
       [ @(1, VGath), @(2, VDiag) ], # o 1-> 2->
        e -> [ VDiag(fCompose(@2.val.element, fTensor(@1.val.func, fId(@1.val.v))), @1.val.v).attrs(@(2).val), @1.val ]),

    # Diag * VScat
    CommuteDiagVScat := ARule( Compose,
       [ @(1, Diag), @(2, VScat) ], # <-1 <-2 o
        e -> [ @2.val, VDiag(fCompose(@1.val.element, fTensor(@2.val.func, fId(@2.val.v))), @2.val.v).attrs(@(1).val) ]),

    # VDiag * VScat
    CommuteVDiagVScat := ARule( Compose,
       [ @(1, VDiag), @(2, VScat) ], # <-1 <-2 o
        e -> [ @2.val, VDiag(fCompose(@1.val.element, fTensor(@2.val.func, fId(@2.val.v))), @2.val.v).attrs(@(1).val) ]),

    # RCVDiag * VScat
    CommuteRCVDiagVScat := ARule( Compose,
       [ @(1, RCVDiag), @(2, VScat) ], # <-1 <-2 o
        e -> [ @2.val, RCVDiag(fCompose(@1.val.element, @2.val.func), @2.val.v).attrs(@(1).val) ]),

    # RCVDiagSplit * VScat
    CommuteRCVDiagSplitVScat := ARule( Compose,
       [ @(1, RCVDiagSplit), @(2, VScat) ], # <-1 <-2 o
        e -> [ @2.val, RCVDiagSplit(fCompose(@1.val.element, fTensor(@2.val.func, fId(2))), @2.val.v).attrs(@(1).val) ]),

    # VGath * VDiag_x_I
    CommuteVGathVDiag_x_I := ARule( Compose,
        [ @(1, VGath), @(2, VDiag_x_I) ], # o 1-> 2->
    e -> [ VDiag_x_I(fCompose(@2.val.element, @1.val.func), @2.val.v).attrs(@(2).val), @1.val ]),

    # VDiag_x_I * VScat
    CommuteVDiag_x_IVScat := ARule( Compose,
        [ @(1, VDiag_x_I), @(2, VScat) ], # <-1 <-2 o
    e -> [ @2.val, VDiag_x_I(fCompose(@1.val.element, @2.val.func), @1.val.v).attrs(@(1).val) ]),

    #------------------------
    # VGath_sv * Diag
    CommuteVGathsvDiag := ARule( Compose,
       [ @(1, VGath_sv), @(2, Diag) ], # o 1-> 2->
       e -> [ VDiag(_fComposeVGathScatSv_Diag(@1.val, Rows, @2.val), @1.val.v).attrs(@2.val), @1.val ]
    ),

    # VGath_sv * VDiag
    CommuteVGathsvVDiag := ARule( Compose,
       [ @(1, VGath_sv), @(2, VDiag) ], # o 1-> 2->
       e -> [ VDiag(_fComposeVGathScatSv_Diag(@1.val, Rows, @2.val), @1.val.v).attrs(@2.val), @1.val ]
    ),

    # Diag * VScat_sv
    CommuteDiagVScatsv := ARule( Compose,
       [ @(1, Diag), @(2, VScat_sv) ], # <-1 <-2 o
       e -> [ @2.val, VDiag(_fComposeVGathScatSv_Diag(@2.val, Cols, @1.val), @2.val.v).attrs(@1.val) ]
    ),

    # VDiag * VScat_sv
    CommuteVDiagVScatsv := ARule( Compose,
       [ @(1, VDiag), @(2, VScat_sv) ], # <-1 <-2 o
       e -> [ @2.val, VDiag(_fComposeVGathScatSv_Diag(@2.val, Cols, @1.val), @2.val.v).attrs(@1.val) ]
    ),

    #-----------------------------------------------------------
    # VStretchGath * Diag
    CommuteVStretchGathDiag := ARule( Compose,
       [ @(1, VStretchGath), @(2, Diag) ], # o 1-> 2->
       e -> let( vg := @1.val, den := vg.func.domain()/vg.part, num := _roundup(den, vg.v),
            [ VDiag(fStretch(fCompose(@2.val.element, @1.val.func), num, den), @1.val.v).attrs(@2.val), @1.val ])
    ),
    
    # Diag * VStretchScat
    CommuteVStretchScatDiag := ARule( Compose,
       [ @(1, Diag), @(2, VStretchGath)], 
       e -> let( vg := @2.val, den := vg.func.domain()/vg.part, num := _roundup(den, vg.v),
            [ @2.val, VDiag(fStretch(fCompose(@1.val.element, @2.val.func), num, den), @2.val.v).attrs(@1.val) ])
    ),

    #-----------------------------------------------------------
    # Diag * IxVScat_pc
    CommuteDiagIxVScat_pc := ARule( Compose, 
	[ @(1, [Diag, VDiag]), @(2, IxVScat_pc) ], # <-1 <-2 o
        e -> let(s:=@(2).val, [ s, VDiag(fStretch(@(1).val.element, _roundup(s.n, s.v), s.n), s.v) ])),

    # IxVGath_pc * Diag
    CommuteIxVGath_pcDiag := ARule( Compose,
	[ @(1, IxVGath_pc), @(2, [Diag, VDiag]) ], # o 1-> 2->
        e -> let(g:=@(1).val, [ VDiag(fStretch(@(2).val.element, _roundup(g.n, g.v), g.n), g.v), g ])),

    # Diag * VScat_pc
    CommuteDiagVScat_pc := ARule(Compose,
       [ @(1, [Diag, VDiag]), @(2, VScat_pc) ], # <-1 <-2 o
       e -> let(s := @(2).val, f := @(1).val.element,
	   [ s, VDiag(fStretch(fCompose(f, fAdd(f.domain(), s.n, s.ofs)), 
			       _roundup(s.n, s.v), s.n), s.v) ])),

    # VGath_pc * Diag 
    CommuteVGath_pcDiag := ARule( Compose, # o 1-> 2->
       [ @(1, VGath_pc), @(2, [Diag, VDiag]) ], 
       e -> let(g := @(1).val, f := @(2).val.element,
	   [ VDiag(fStretch(fCompose(f, fAdd(f.domain(), g.n, g.ofs)), 
		            _roundup(g.n, g.v), g.n), g.v), g ])),


    #-----------------------------------------------------------
    # YSV: Below 2 rules have limitations which can be removed, 
    #      But it seems that fStretch has built-in assumptions which makes it difficult
    #      to remove these limitations.. 
    
    # NOTE: Remove limitations, the rule as is, is invalid for g.ofs <> 0, or if g.N > g.v
    #        Hence the precondition 
    ## Diag * IxRCVScat_pc
    CommuteDiagIxRCVScat_pc := ARule(Compose,
       [ @(1, [Diag, VDiag]), @(2, IxRCVScat_pc, e->e.ofs = 0 and not IsSymbolic(e.n) and e.n <= e.v) ], # <-1 <-2 o
       e -> let(s := @(2).val, 
	   [ s, VDiag(fStretch(@(1).val.element, 
			       _roundup(s.n, 2*s.v), 2*s.n), s.v) ])),

    # NOTE: Remove limitations, the rule as is, is invalid for g.ofs <> 0, or if g.N > g.v
    #        Hence the precondition 
    # IxRCVGath_pc * Diag 
    CommuteIxRCVGath_pcDiag := ARule( Compose, # o 1-> 2->
       [ @(1, IxRCVGath_pc, e->e.ofs = 0 and not IsSymbolic(e.n) and e.n <= e.v), @(2, [Diag, VDiag]) ], 
       e -> let(g := @(1).val, 
	   [ VDiag(fStretch(@(2).val.element, 
		            _roundup(g.n, 2*g.v), 2*g.n), g.v), g ])),
    #-----------------------------------------------------------
    

    #-----------------------------------------------------------
# NOTE: unclear how to do. Rewriting rule missing: f^{n->N} o \pi^{n} -> \rho^{N} o h^{n->N}
#    # BlockVPerm * VDiag
#    CommuteBlockVPermVDiag := ARule( Compose,
#       [ @(1, BlockVPerm), @(2, VDiag) ], # o 1-> 2->
#         e -> let(vdiag := Copy(@2.val), bperm := Copy(@1.val),
#         [ VDiag(fCompose(vdiag.element, fTensor(fId(bperm.n), bperm.perm)), vdiag.v).attrs(vdiag), BlockVPerm2(bperm.n, bperm._children[1], bperm.perm) ])),
#
#    # VDiag * BlockVPerm
#    CommuteVDiagBlockVPerm := ARule( Compose,
#       [ @(1, VDiag), @(2, BlockVPerm) ], # <-1 <-2 o
#         e -> let(vdiag := Copy(@1.val), bperm := Copy(@2.val),
#        [ BlockVPerm2(bperm.n, bperm._children[1], bperm.perm), VDiag(fCompose(vdiag.element, fTensor(fId(bperm.n), bperm.perm)), vdiag.v).attrs(vdiag) ])),

# this is a hack that *DOES NOT* produce correct code
#    # BlockVPerm * VDiag: HACK that *DOES NOT* produce correct code
#    CommuteBlockVPermVDiag := ARule( Compose,
#       [ @(1, BlockVPerm), @(2, VDiag) ], # o 1-> 2->
#         e -> let(vdiag := Copy(@2.val), bperm := Copy(@1.val),
#         [ vdiag, BlockVPerm2(bperm.n, bperm._children[1], bperm.perm) ])),
#
#    # VDiag * BlockVPerm: HACK that *DOES NOT* produce correct code
#    CommuteVDiagBlockVPerm := ARule( Compose,
#       [ @(1, VDiag), @(2, BlockVPerm) ], # <-1 <-2 o
#         e -> let(vdiag := Copy(@1.val), bperm := Copy(@2.val),
#        [ BlockVPerm2(bperm.n, bperm._children[1], bperm.perm), vdiag ])),


    #--------------------------------------------------
    #   vectorize constructs
    ParDiagVPerm := ARule(Compose, [@(1, Diag), @(2, VPerm)],
        e -> [VDiag(@1.val.element, @(2).val.vlen), @(2).val]
    ),
    ParVPermDiag := ARule(Compose, [@(1, VPerm), @(2, Diag)],
        e -> [@(1).val, VDiag(@2.val.element, @(1).val.vlen)]
    ),
    ParScatVDiag := ARule(Compose,
    [[Scat, [@(0,fTensor), ..., @(2,fId)]], @(1,VDiag,e->@(2).val.size=Rows(e))],
    e -> let(ch := @(0).val.children(), X := @(1).val, v := X.v,
               [VTensor(Scat(fTensor(DropLast(ch,1), fId(Rows(X)/v))), v), X])
    ),
    ParVTensorDiag := ARule(Compose, [@(1, VTensor), @(2, Diag)],
        e -> [ @(1).val, VDiag(@2.val.element, @(1).val.vlen)]
    ),
    ParDiagVTensor := ARule(Compose, [@(1, Diag), @(2, VTensor)],
        e -> [ VDiag(@1.val.element, @(2).val.vlen), @(2).val ]
    ),

    ParVSDiag := ARule(Compose, [@(1, VS), @(2, Diag)],
        e -> [ @(1).val, VDiag(@2.val.element, @(1).val.v)]
    ),

    ParDiagVS := ARule(Compose, [@(1, Diag), @(2, VS)],
        e -> [ VDiag(@1.val.element, @(2).val.v), @(2).val ]
    )
));


RewriteRules(RulesFuncSimp, rec(

    RCData_VDupOnline := Rule( [RCData, VDupOnline], 
        e -> VDupOnline(RCData(e.func.func), e.func.v)),

    RCVData_VDupOnline := Rule( [RCVData, VDupOnline],
        e -> VDupOnline(RCVData(VDup(e.func.func, 1)), e.func.v)),

    RCVDataSplitVec_VDupOnline := Rule( [RCVDataSplitVec, VDupOnline],
        e -> VDupOnline(RCVDataSplitVec(VDup(e.func.func, 1)), e.func.v)),

    fCompose_VDupOnline_FList := ARule(fCompose, [@(1, VDupOnline), @(2, FList)], 
        e -> [VDupOnline(fCompose(@(1).val.func, @(2).val), @(1).val.v)]),

    fPrecompute_VDupOnline := Rule( [fPrecompute, VDupOnline],
        e -> VDupOnline(fPrecompute(e.child(1).func), e.child(1).v)),
    
    RCVData_VDup_CRData := Rule([RCVData, [VDup, [CRData, @(1)], @(2)]],
        e -> VDup( VData(@(1).val, 2), @(2).val) ),
));

