
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


################################################
#   propagate VTensor into products and sums
_VConsRightBlk := [BlockVPerm, VPerm, VDiag_x_I, VDiag, VRCDiag, RCVDiagSplit, RCVDiag, VGath, VGath_u, VGath_sv, VGath_pc, IxVGath_pc, VGath_dup, VPrm_x_I, Conj, ConjL, ConjR, ConjLR, RCVGath_sv, FormatPrm];
_VConsLeftBlk  := [BlockVPerm, VPerm, VDiag_x_I, VDiag, VRCDiag, RCVDiagSplit, RCVDiag, VScat, VScatAcc, VScatAcc_u, VScat_u, VScat_sv, VScat_svAcc, VScat_pc, IxVScat_pc, VScat_pcAcc, VPrm_x_I, Conj, ConjL, ConjR, ConjLR, RCVScat_sv, FormatPrm];

# !!!! do not put VPerm in here - VPerms CANNOT be pulled into ISums!!!!
_VConsDiag := [ VDiag_x_I, VDiag, VRCDiag, RCVDiag, RCVDiagSplit ];

_VConsRightNoDiag := [VGath, VGath_u, VGath_sv, VPrm_x_I];
_VConsLeftNoDiag  := [VScat, VScat_u, VScat_sv, VPrm_x_I, VScatAcc, VScatAcc_u, VScat_svAcc, VScat_pcAcc, VDiag_x_I];

_VConsRight := _VConsRightNoDiag :: _VConsDiag :: [IxVGath_pc ];
_VConsLeft  := _VConsLeftNoDiag :: _VConsDiag :: [IxVScat_pc ];

# things to not pull into Vcontainer:
_VContDontPullIn := [VContainer, Cross, Cvt, TCvt, RC, BB, ISum, SUM, RecursStep];

_PullInLeft  := [RecursStep, Grp, BB, SUM, Buf, ISum, Data, COND, NeedInterleavedComplex];
_PullInRight := _PullInLeft :: [SUMAcc, ISumAcc];

Class(RulesPropagate, RuleSet);
RewriteRules(RulesPropagate, rec(

 VecPullInLeft := ARule( Compose, [ @(1, _VConsLeft), @(2, _PullInLeft :: [NoDiagPullinRight]) ],
  e -> [ CopyFields(@(2).val, rec(
             _children :=  List(@(2).val._children, c -> @(1).val * c),
             dimensions := [Rows(@(1).val), Cols(@(2).val)] )) ]),

 VecPullInRight := ARule( Compose, [ @(1, _PullInRight :: [NoDiagPullinLeft]), @(2, _VConsRight) ],
     e -> [ CopyFields(@(1).val, rec(
                _children := List(@(1).val._children, c -> c * @(2).val),
                dimensions := [Rows(@(1).val), Cols(@(2).val)] )) ]),

 VecPullInLeftNoDiag := ARule( Compose, [ @(1, _VConsLeftNoDiag), @(2, [NoDiagPullinLeft, NoDiagPullin]) ],
  e -> [ CopyFields(@(2).val, rec(
             _children :=  List(@(2).val._children, c -> @(1).val * c),
             dimensions := [Rows(@(1).val), Cols(@(2).val)] )) ]),

 VecPullInRightNoDiag := ARule( Compose, [ @(1, [NoDiagPullinRight, NoDiagPullin]), @(2, _VConsRightNoDiag) ],
     e -> [ CopyFields(@(1).val, rec(
                _children := List(@(1).val._children, c -> c * @(2).val),
                dimensions := [Rows(@(1).val), Cols(@(2).val)] )) ]),

#######################################
    VTensorXXX := Rule([@(1, VTensor), @(2, [SUM, SUMAcc, Compose, BB, Buf, Inplace, RecursStep, ISum, ISumAcc, ISumLS, Data, NeedInterleavedComplex, SMPBarrier, SMPSum])],
    e -> let(s := @(2).val, CopyFields(s,
         rec(_children := List(s.children(), c->VTensor(c, @(1).val.vlen)),
             dimensions := @(1).val.dimensions)))),

    VTensorIndXXX := Rule([@(1, VTensorInd), @(2, [SUM, SUMAcc, Compose, BB, Buf, Inplace, RecursStep, ISum, ISumAcc, ISumLS, Data, NeedInterleavedComplex, SMPBarrier, SMPSum]), @(3)],
    e -> let(s := @(2).val, CopyFields(s,
         rec(_children := List(s.children(), c->VTensorInd(c, @(3).val)),
             dimensions := @(1).val.dimensions)))),

    # Free variables in breakdown rules seems to be broken...
    VTensorInd_VTensor := Rule([@(1, VTensorInd), @(2).cond(e->ObjId(e)<>RTWrap), @(3).cond(e->not(e in @(2).val.free()))],
        e -> VTensor(@(2).val, @(3).val.range)),

#(s) -> SubstTopDownNR(RulesRC(s), @@(1, RCDiag, (x, cx) -> IsBound(cx.VRCLR) and cx.VRCLR <> [  ] or x.element.range() = TComplex), (x) -> x.toloop().sums())

    # NOTE (handle xofs and yofs correctly)
    MergeRS := ARule(Compose, [@(1,RecursStep), @(2,RecursStep)],
    e -> [ RecursStep(0,0,@(1).val.child(1)*@(2).val.child(1)) ]),

    # NOTE (handle xofs and yofs correctly)
    # V * OP(a) -> OP(V*a), including the case V=BlockVPerm
    XXX_VConsBlk := ARule(Compose, [@(1,[RecursStep, Buf, Inplace, BB, NeedInterleavedComplex]), @(2,_VConsRightBlk)],
    e -> [ ObjId(@(1).val)(@(1).val.child(1)*@(2).val) ]),
    XXX_VConsBlk_VCont := ARule(Compose, [@(1,[RecursStep, Buf, Inplace, BB, NeedInterleavedComplex]), [@(2, VContainer), @(3,_VConsRightBlk)]],
    e -> [ ObjId(@(1).val)(@(1).val.child(1)*@(2).val) ]),

    VConsBlk_XXX := ARule(Compose, [@(1,_VConsLeftBlk), @(2, [RecursStep, Buf, Inplace, BB, NeedInterleavedComplex])],
    e -> [ ObjId(@(2).val)(@(1).val*@(2).val.child(1)) ]),
    VConsBlk_VCont_XXX := ARule(Compose, [[@(1, VContainer), @(3,_VConsLeftBlk)], @(2, [RecursStep, Buf, Inplace, BB, NeedInterleavedComplex])],
    e -> [ ObjId(@(2).val)(@(1).val*@(2).val.child(1)) ]),


    VConsBlk_XXX := ARule(Compose, [@(1,_VConsLeftBlk), @(2, [RecursStep, Buf, Inplace, BB, NeedInterleavedComplex])],
    e -> [ ObjId(@(2).val)(@(1).val*@(2).val.child(1)) ]),

    VContainer_XXX := ARule(Compose, [@(1,VContainer), @(2).cond(x -> not (ObjId(x) in _VContDontPullIn))],
        e -> [ ObjId(@(1).val)(@(1).val.child(1)*@(2).val, @(1).val.isa) ]),

    XXX_VContainer := ARule(Compose, [@(1).cond(x -> not (ObjId(x) in _VContDontPullIn)), @(2, VContainer)],
        e -> [ ObjId(@(2).val)(@(1).val*@(2).val.child(1), @(2).val.isa) ]),

    Compose_VContainers := ARule(Compose, [ @(1, VContainer), @(2, VContainer, x -> x.isa=@(1).val.isa and x.isa.isCplx()=@(1).val.isa.isCplx())],
        e -> [VContainer(@(1).val.child(1) * @(2).val.child(1), @(1).val.isa)]),

    ISum_VContainer := Rule( [ISum, [@(1, VContainer), @(2)]],
        e -> VContainer(ISum(e.var, e.domain, @(2).val), @(1).val.isa)),
    IParSeq_VContainer := Rule( [IParSeq, [@(1, VContainer), @(2)]],
        e -> VContainer(IParSeq(e.var, e.domain, e.fb_cnt, @(2).val), @(1).val.isa)),
    SUM_VContainer := Rule( [@(1, SUM), @(2, VContainer, x -> ForAll(@(1).val.children(), a -> ObjId(a)=VContainer)), ...],
        e -> VContainer( ApplyFunc(SUM, List(e.children(), l -> l.child(1))), @(2).val.isa)),
    VContainer_PullFrom := Rule( [@(1, [NoDiagPullin, NoDiagPullinLeft, NoDiagPullinRight, CR, SymSPL]), @(2, VContainer)],
        e -> VContainer( ObjId(@(1).val)(@(2).val.child(1)), @(2).val.isa )),
    BlockVPerm_VContainer := Rule( [BlockVPerm, @(1, VContainer)],
        e -> VContainer( BlockVPerm(e.n, e.vlen, @(2).val.child(1), e.perm), @(1).val.isa )),
    # WrappedVCons_XXX and XXX_WrappedVCons rules duplicate VCons_XXX and XXX_VCons to suck
    # Vconstructs wrapped into VContainer.
    # V * SUM(a,b,...) -> SUM(V*a, V*b, ...), do not suck in BlockVPerm
    WrappedVCons_XXX := ARule(Compose, [ [@(1, VContainer), @(0, _VConsLeft)], @(2, [SUM, SMPSum, SMPBarrier, ISum, Data]) ], e -> let(s:=@(2).val,
    [ CopyFields(s, rec(_children := List(s.children(), c -> @(1).val * c ),
                    dimensions := [@(1).val.dimensions[1], @(2).val.dimensions[2]])) ])),

    XXX_WrappedVCons := ARule(Compose, [ @(1, [SUM, SUMAcc, SMPSum, SMPBarrier, ISum, Data]), [@(2, VContainer), @(0, _VConsRight)]], e -> let(s:=@(1).val,
    [ CopyFields(s, rec(_children := List(s.children(), c -> c * @(2).val ),
                    dimensions := [@(1).val.dimensions[1], @(2).val.dimensions[2]])) ])),

    # V * SUM(a,b,...) -> SUM(V*a, V*b, ...), do not suck in BlockVPerm
    VCons_XXX := ARule(Compose, [ @(1, _VConsLeft), @(2, [SUM, SMPSum, SMPBarrier, ISum, Data]) ], e -> let(s:=@(2).val,
    [ CopyFields(s, rec(_children := List(s.children(), c -> @(1).val * c ),
                    dimensions := [@(1).val.dimensions[1], @(2).val.dimensions[2]])) ])),

    XXX_VCons := ARule(Compose, [ @(1, [SUM, SUMAcc, SMPSum, SMPBarrier, ISum, Data]), @(2, _VConsRight)], e -> let(s:=@(1).val,
    [ CopyFields(s, rec(_children := List(s.children(), c -> c * @(2).val ),
                    dimensions := [@(1).val.dimensions[1], @(2).val.dimensions[2]])) ])),

    COND_VCons := ARule(Compose, [@(1, COND), @(2, _VConsRight)],
        e -> [COND(@(1).val.cond, @(1).val.child(1)*@(2).val, @(1).val.child(2)*@(2).val)]),

    VCons_COND := ARule(Compose, [@(1, _VConsLeft), @(2, COND)],
        e -> [COND(@(2).val.cond, @(1).val*@(2).val.child(1), @(1).val*@(2).val.child(2))]),

    # BlockVPerm * ISumLS
    BlockVPermISum := ARule(Compose,  [ @(1, BlockVPerm), @(2, ISumLS) ],
         e -> [ ISum(@(2).val.var, @(2).val.domain, @(1).val * @(2).val.child(1)) ]),

    ISumBlockVPerm  := ARule(Compose, [ @(1, ISumLS), @(2, BlockVPerm) ],
        e -> [ ISum(@(1).val.var, @(1).val.domain, @(1).val.child(1) * @(2).val) ]),

    # Replace by vector constructs
    VectGath := Rule([@(1, VTensor), @(2, Gath), ...], e->VGath(@(2).val.func, @(1).val.vlen)),
    VectScat := Rule([@(1, VTensor), @(2, Scat), ...], e->VScat(@(2).val.func, @(1).val.vlen)),
    VectScatAcc := Rule([@(1, VTensor), @(2, ScatAcc), ...], e->VScatAcc(@(2).val.func, @(1).val.vlen)),

    # ----------------------
    # Combine Gath/Scat
    #
    ComposeVGathVGath := ARule(Compose, [ @(1, VGath), @(2, [VGath, VGath_u, VGath_dup], e->@(1).val.v=e.v) ], # o 1-> 2->
        e -> [ ObjId(@(2).val)(fCompose(@(2).val.func, @(1).val.func), @(1).val.v) ]),
    # there is a copy of the rule below in autolib, why?
    ComposeVGath_dup_Gath := ARule(Compose, [@(1, VGath_dup), @(2, [Gath, Prm])],
    e -> [ VGath_dup(fCompose(Cond(ObjId(@2.val)=Prm,@(2).val.func,@(2).val.func), @(1).val.func), @(1).val.v) ]),

    ComposeVScatVScat := ARule(Compose, [ @(1, [VScat_u, VScat, VScatAcc, VScatAcc_u]), @(2, VScat, e->@(1).val.v=e.v) ], # <-1 <-2 o
        e -> [ ObjId(@(1).val)(fCompose(@(1).val.func, @(2).val.func), @(1).val.v) ]),

    ComposeVScatVScatAcc := ARule(Compose, [ @(1, [VScat_u, VScat, VScatAcc, VScatAcc_u]), @(2, VScatAcc, e->@(1).val.v=e.v) ], # <-1 <-2 o
        e -> let(scat := Cond(@(1).val _is VScat,   VScatAcc,
		              @(1).val _is VScat_u, VScatAcc_u,
			      @(1).val),
	    [ scat(fCompose(@(1).val.func, @(2).val.func), @(1).val.v) ])),

    ComposeVScatPrm  := ARule(Compose, [@(1, [VScat, VScat_u, VScatAcc, VScatAcc_u]),   @(2, VPrm_x_I)], # 1-> <-2 o
        e -> [ ObjId(@(1).val)(fCompose(@(1).val.func, @(2).val.func.transpose()), @(1).val.v) ]),
    ComposeVGathPrm  := ARule(Compose, [ @(1, VGath), @(2, VPrm_x_I) ], # o 1-> 2->
        e -> [ VGath(fCompose(@(2).val.func, @(1).val.func), @(1).val.v) ]),
    ComposePrmVScat  := ARule(Compose, [ @(1, VPrm_x_I), @(2, [VScat, VScatAcc]) ], # 1-> <-2 o
        e -> [ ObjId(@(2).val)(fCompose(@(1).val.func.transpose(), @(2).val.func), @(2).val.v) ]),
    ComposePrmVGath  := ARule(Compose, [ @(1, VPrm_x_I), @(2, [VGath, VGath_u])], # o 1-> 2->
        e -> [ ObjId(@(2).val)(fCompose(@(2).val.func, @(1).val.func), @(2).val.v) ]),

    ComposeVTensorPrm  := ARule(Compose, [@(1, VTensor),   @(2, Prm)],
        e -> [ @(1).val, VGath_sv(@(2).val.func, @(1).val.vlen, 1) ]),
    ComposePrmVTensor  := ARule(Compose, [ @(1, Prm), @(2, VTensor) ],
        e -> [  VScat_sv(@(1).val.func.transpose(), @(2).val.vlen, 1), @(2).val ]),

## NOTE: These rules seem broken. There seems to be an assumption on the size of the Scatter. look at Compose_IxVGath_pc__Gath which was fixed for BG/Q 3D FFT
    ComposeIxVGath_pc__Prm  := ARule(Compose, [ @(1, IxVGath_pc), @(2, [Prm, DelayedPrm])], # o 1-> 2->
        e -> let(g:=@(1).val, [ VStretchGath(@(2).val.func, g.k, g.v) ])),
    Compose_IxVGath_pc__Gath  := ARule(Compose, [ @(1, IxVGath_pc), @(2, Gath)], # o 1-> 2->
        e -> let(g:=@(1).val, [ VStretchGath(fCompose(@(2).val.func, fTensor(fId(g.k), fAdd(g.N, g.n, g.ofs))), g.k, g.v) ])),
    Compose_IxVGath_pc__VGath_sv  := ARule(Compose, [ @(1, IxVGath_pc), @(2, VGath_sv)], # o 1-> 2->
        e -> let(g:=@(1).val, [ VStretchGath(fTensor(@(2).val.func, fId(@(2).val.sv)), g.k, g.v) ])),
    Compose_IxVGath_pc__VGath  := ARule(Compose, [ @(1, IxVGath_pc), @(2, VGath)], # o 1-> 2->
        e -> let(g:=@(1).val, [ VStretchGath(fTensor(@(2).val.func, fId(g.v)), g.k, g.v) ])),
    Compose_IxRCVGath_pc__RCGath  := ARule(Compose, [ @(1, IxRCVGath_pc), @(2, VGath_sv, e->e.sv=2)], # o 1-> 2->
        e -> let(g:=@(1).val, [ RCVStretchGath(@(2).val.func, g.k, g.v) ])),

    Compose_VGath__IxVGath_pc := ARule(Compose, [ @(1, VGath, e->e.func.domain()<=2), @(2, IxVGath_pc, e->e.n <= e.v)],
        e -> let(g := @(1).val, gpc := @(2).val,
            Cond(g.func.domain()=1,
                [ VGath_pc(gpc.k*gpc.N, gpc.n, fCompose(fTensor(fId(gpc.k), fAdd(gpc.N, gpc.n, gpc.ofs)), fTensor(g.func, fId(gpc.n))).at(0), gpc.v) ],
                [ SUM(
                    VScat(fBase(V(2),V(0)), g.v) * VGath_pc(gpc.k*gpc.N, gpc.n, fCompose(fTensor(fId(gpc.k), fAdd(gpc.N, gpc.n, gpc.ofs)), fTensor(fCompose(g.func, fBase(V(2),V(0))), fId(gpc.n))).at(0), gpc.v),
                    VScat(fBase(V(2),V(1)), g.v) * VGath_pc(gpc.k*gpc.N, gpc.n, fCompose(fTensor(fId(gpc.k), fAdd(gpc.N, gpc.n, gpc.ofs)), fTensor(fCompose(g.func, fBase(V(2),V(1))), fId(gpc.n))).at(0), gpc.v))
                ]
        ))),

## NOTE: These rules seem broken. There seems to be an assumption on the size of the Scatter. look at Compose_IxVGath_pc__Gath which was fixed for BG/Q 3D FFT
    ComposePrmIxVScat_pc  := ARule(Compose, [ @(1, [Prm, DelayedPrm]), @(2, IxVScat_pc) ], # 1-> <-2 o
        e -> let(s:=@(2).val, [ VStretchScat(@(1).val.func.transpose(), s.k, s.v) ])),
    Compose_Scat__IxVScat_pc  := ARule(Compose, [ @(1, Scat), @(2, IxVScat_pc) ], # 1-> <-2 o
        e -> let(s:=@(2).val, [ VStretchScat(fCompose(@(1).val.func, fTensor(fId(s.k), fAdd(s.N, s.n, s.ofs))), s.k, s.v) ])),
    Compose_VScat_sv__IxVScat_pc  := ARule(Compose, [ @(1, VScat_sv), @(2, IxVScat_pc) ], # 1-> <-2 o
        e -> let(s:=@(2).val, [ VStretchScat(fTensor(@(1).val.func, fId(@(1).val.sv)), s.k, s.v) ])),
    Compose_VScat__IxVScat_pc  := ARule(Compose, [ @(1, VScat), @(2, IxVScat_pc) ], # 1-> <-2 o
        e -> let(s:=@(2).val, [ VStretchScat(fTensor(@(1).val.func, fId(s.v)), s.k, s.v) ])),
    Compose_RCScat__IxRCVScat_pc  := ARule(Compose, [ @(1, VScat_sv, e->e.sv=2), @(2, IxRCVScat_pc) ], # 1-> <-2 o
        e -> let(s:=@(2).val, [ RCVStretchScat(@(1).val.func, s.k, s.v) ])),

    Compose_IxVScat_pc__VScat := ARule(Compose, [ @(1, IxVScat_pc, e->e.n <= e.v), @(2, VScat, e->e.func.domain()<=2)],
        e -> let(s := @(2).val, spc := @(1).val,
            Cond(s.func.domain()=1,
                [ VScat_pc(spc.k*spc.N, spc.n, fCompose(fTensor(fId(spc.k), fAdd(spc.N, spc.n, spc.ofs)), fTensor(s.func, fId(spc.n))).at(0), spc.v) ],
                [ SUM(
                    VScat_pc(spc.k*spc.N, spc.n, fCompose(fTensor(fId(spc.k), fAdd(spc.N, spc.n, spc.ofs)), fTensor(fCompose(s.func, fBase(V(2),V(0))), fId(spc.n))).at(0), spc.v) * VGath(fBase(V(2),V(0)), s.v),
                    VScat_pc(spc.k*spc.N, spc.n, fCompose(fTensor(fId(spc.k), fAdd(spc.N, spc.n, spc.ofs)), fTensor(fCompose(s.func, fBase(V(2),V(1))), fId(spc.n))).at(0), spc.v) * VGath(fBase(V(2),V(1)), s.v))
                ]
        ))),

    ComposeVStretchGath__Prm  := ARule(Compose, [ @(1, VStretchGath), @(2, [Prm,DelayedPrm])], # o 1-> 2->
        e -> let(g:=@(1).val, [ VStretchGath(fCompose(@(2).val.func, g.func), g.part, g.v) ])),
    Compose_VStretchGath__Gath  := ARule(Compose, [ @(1, VStretchGath), @(2, Gath)], # o 1-> 2->
        e -> let(g:=@(1).val, [ VStretchGath(fCompose(@(2).val.func, g.func), g.part, g.v) ])),
    Compose_VStretchGath__VGath  := ARule(Compose, [ @(1, VStretchGath), @(2, VGath)], # o 1-> 2->
        e -> let(g:=@(1).val, [ VStretchGath(fCompose(fTensor(@(2).val.func, fId(g.v)), g.func), g.part, g.v) ])),

    ComposePrmVStretchScat  := ARule(Compose, [ @(1, [Prm,DelayedPrm]), @(2, VStretchScat) ], # 1-> <-2 o
        e -> let(s:=@(2).val, [ VStretchScat(fCompose(@(1).val.func.transpose(), s.func), s.part, s.v) ])),
    Compose_Scat__VStretchScat  := ARule(Compose, [ @(1, Scat), @(2, VStretchScat) ], # 1-> <-2 o
        e -> let(s:=@(2).val, [ VStretchScat(fCompose(@(1).val.func, s.func), s.part, s.v) ])),
    Compose_VScat__VStretchScat  := ARule(Compose, [ @(1, VScat), @(2, VStretchScat) ], # 1-> <-2 o
        e -> let(s:=@(2).val, [ VStretchScat(fCompose(fTensor(@(1).val.func, fId(s.v)), s.func), s.part, s.v) ])),

    Drop_GathScat := ARule(Compose, [ @(1), @(2, VGath),  @(3, VScat, x -> x.transpose()=@(2).val), @(4)],
        e -> [@(1).val, @(4).val]),
# uncombinable gather/scatter CANNOT be pulled in!!!
# NOTE: only pull till BB(), then stop -> new rewriting stage _after_ MarkBB
#    XXX_ISum := ARule(Compose,  [ @(1, [IxVScat_pc, IxRCVScat_pc, VStretchScat, RCVStretchScat]), @(2, ISum) ],
#         e -> [ ObjId(@(2).val)(@(2).val.var, @(2).val.domain, @(1).val * @(2).val.child(1)).attrs(@(2).val) ]),
#
#    ISum_XXX  := ARule(Compose, [ @(1, [ISum, ISumAcc]), @(2, [IxVGath_pc, IxRCVGath_pc, VStretchGath, RCVStretchGath]) ],
#        e -> [ ObjId(@(1).val)(@(1).val.var, @(1).val.domain, @(1).val.child(1) * @(2).val).attrs(@(1).val) ]),


# uncombinable gather/scatter CANNOT be pulled in!!!
# NOTE: only pull till BB(), then stop -> new rewriting stage _after_ MarkBB
#    VScatSUM := ARule(Compose, [ @(1, [IxVScat_pc, IxRCVScat_pc, VStretchScat, RCVStretchScat]), @(2, SUM) ],
#     e -> [ ApplyFunc(ObjId(@(2).val),
#                  List(@(2).val.children(), c -> @(1).val * c)) ]),
#
#    VGathSUM  := ARule(Compose, [ @(1, SUM), @(2, [IxVGath_pc, IxRCVGath_pc, VStretchGath, RCVStretchGath])],
#     e -> [ ApplyFunc(ObjId(@(1).val),
#                  List(@(1).val.children(), c -> c * @(2).val)) ]),

################################################
# NOTE!!!!!!!!!!!!!!!!!!
# GUARD MISSING: gather/scatter must be f x fId(v^2)
#    ComposeBlockVPermVScat := ARule(Compose,[ @(1, BlockVPerm), @(2, VScat)],
#        e->[@2.val, BlockVPerm(@2.val.dimensions[2]/@1.val.child(1).dims()[1], @1.val.vlen, @1.val.child(1), @1.val.perm)]
#    ),
#    ComposeVGathBlockVPerm := ARule(Compose,[ @(1, VGath), @(2, BlockVPerm)],
#        e->[BlockVPerm(@1.val.dimensions[1]/@2.val.child(1).dims()[2], @2.val.vlen, @2.val.child(1), @2.val.perm), @1.val ]
#    ),

    ScatH_VScat_pc := ARule(Compose, [ [ @(1, [Scat,ScatAcc]), fId ], @(2, [VScat_pc, VScat_pcAcc]) ],
	e -> let(s   := @(2).val,
	         oid := Cond((@(1).val _is Scat) and (@(2).val _is VScat_pc), VScat_pc, VScat_pcAcc),
	         [ oid(s.N, s.n, s.ofs, s.v) ])),

    ScatId_VScat_pc := ARule(Compose, [ [ @(1, [Scat,ScatAcc]), [ @(2,H), @, @, @, 1 ] ], @(3, [VScat_pc, VScat_pcAcc]) ],
	e -> let(h   := @(2).val,
	         s   := @(3).val,
	         oid := Cond((@(1).val _is Scat) and (@(3).val _is VScat_pc), VScat_pc, VScat_pcAcc),
	         [ oid(h.params[1], s.n, s.ofs + h.params[3], s.v) ])),

    VGath_pc__toVGath := Rule(@(1, VGath_pc, e->let(v := e.v, (e.N mod v = 0) and (e.n mod v = 0) and (e.ofs mod v = 0))),
        e->let(g := @(1).val, v := g.v, VGath(fAdd(g.N/v, g.n/v, g.ofs/v), v))),

    VScat_pc__toVScat := Rule(@(1, VScat_pc, e->let(v := e.v, (e.N mod v = 0) and (e.n mod v = 0) and (e.ofs mod v = 0))),
        e->let(s := @(1).val, v := s.v, VScat(fAdd(s.N/v, s.n/v, s.ofs/v), v))),

    VScat_pcAcc__toVScatAcc := Rule(@(1, VScat_pcAcc, e->let(v := e.v, (e.N mod v = 0) and (e.n mod v = 0) and (e.ofs mod v = 0))),
        e->let(s := @(1).val, v := s.v, VScatAcc(fAdd(s.N/v, s.n/v, s.ofs/v), v)))

));


Class(RulesTerm, RuleSet);
RewriteRules(RulesTerm, rec(
    ISumLS_Term := Rule(@(1,ISumLS), e->ISum(@(1).val.var, @(1).val.domain, @(1).val.child(1)).attrs(@(1).val)),
    BlockVPerm_Term := Rule(@(1,[BlockVPerm, BlockVPerm2]), e-> Tensor(I(@1.val.n), @1.val._children[1]).sums()),
#   BROKEN RULES; to be redone for loop code
#    IxVGath_pc_term := Rule(@(1,IxVGath_pc, e->e.k>1),
#        e->let(i := Ind(), k:= @1.val.k, v:=@1.val.v,
#            ISum(i, k,
#                VScat(fTensor(fBase(k,i), fId(@1.val.nv/v)), v) *
#                IxVGath_pc(1, k*@1.val.N, @1.val.n, add(mul(i,k), @1.val.ofs), v)
#            ))),
#    IxVScat_pc_term := Rule(@(1,IxVScat_pc, e->e.k>1),
#        e->let(i := Ind(), k:= @1.val.k, v:=@1.val.v,
#            ISum(i, k,
#                IxVScat_pc(1, k*@1.val.N, @1.val.n, add(mul(i,k), @1.val.ofs), v) *
#                VGath(fTensor(fBase(k,i), fId(@1.val.nv/v)), v)
#            ))),
#
    term_PushLR := Rule(@(1, PushLR), e->e.child(1).sums()),


));


Class(RulesTermGrp, RuleSet);
RewriteRules(RulesTermGrp, rec(
    Grp_Term := Rule(@(1,Grp), e-> @1.val._children[1])
));


Class(RulesKickout, RuleSet);
RewriteRules(RulesKickout, rec(
    IxVGath_pc__IxVScat_pc_kickout := ARule(Compose, [@(1, IxVGath_pc), @(2, IxVScat_pc, g->let(s:=@(1).val, s.k=g.k and s.n=g.n and s.N=g.N and s.ofs=g.ofs and s.v=g.v))],
            e->let(g:=@(1).val, [VGath(fId(Rows(g)/g.v), g.v), VScat(fId(Rows(g)/g.v), g.v)])),

    IxVScat_VScat_kickout := Rule(@(1,IxVScat_pc, s-> IsInt(s.n/s.v) and IsInt(s.N/s.v) and IsInt(s.ofs/s.v)),
            e->let(s:=@(1).val, v:=s.v, VScat(fTensor(fId(s.k), fAdd(s.N/v, s.n/v, s.ofs/v)), v))),

    IxVGath_VGath_kickout := Rule(@(1,IxVGath_pc, s-> IsInt(s.n/s.v) and IsInt(s.N/s.v) and IsInt(s.ofs/s.v)),
            e->let(s:=@(1).val, v:=s.v, VGath(fTensor(fId(s.k), fAdd(s.N/v, s.n/v, s.ofs/v)), v))),
));


Class(TerminateSymSPL, RuleSet);
RewriteRules(TerminateSymSPL, rec(
    terminateSymSPL := Rule(@(1, SymSPL), e->@(1).val.child(1))
));

Class(TerminateDPrm, RuleSet);
RewriteRules(TerminateSymSPL, rec(
    terminateDPrm := Rule(@(1, DelayedPrm), e->Prm(@(1).val.func))
));
