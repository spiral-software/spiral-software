
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(RulesCxRC_Op, RuleSet);
RewriteRules(RulesCxRC_Op, rec(

    vRC_Compose := Rule([vRC, @(1, Compose)], e -> Compose(List(@(1).val.children(), vRC))),
    vRC_SUM := Rule([vRC, @(1, SUM)], e -> SUM(List(@(1).val.children(), vRC))),
    vRC_SUMAcc := Rule([vRC, @(1, SUMAcc)], e -> SUMAcc(List(@(1).val.children(), vRC))),

    vRC_Container := Rule([vRC, @(1, [BB,Buf,Inplace,Grp,NoPull,NoPullLeft,NoPullRight, NoDiagPullin, NoDiagPullinLeft, NoDiagPullinRight ])],
        e -> ObjId(@(1).val)(vRC(@(1).val.child(1)))),

    vRC_Data := Rule([vRC, @(1, Data)], e -> Data(@(1).val.var, @(1).val.value, vRC(@(1).val.child(1)))),

    vRC_RStep := Rule([vRC, @(1, RecursStep)], e -> RecursStep(2*@(1).val.yofs, 2*@(1).val.xofs,
                                                      vRC(@(1).val.child(1)))),

    vRC_ISum := Rule([vRC, @(1, ISum)], e -> ISum(@(1).val.var, @(1).val.domain, vRC(@(1).val.child(1)))),
    vRC_ICompose := Rule([vRC, @(1, ICompose)], e -> ICompose(@(1).val.var, @(1).val.domain, vRC(@(1).val.child(1)))),

    vRC_Grp := Rule([vRC, @(1, Grp)], e -> Grp(vRC(@(1).val.child(1)))),


    vRC_Scale := Rule([vRC, @(1, Scale)], e ->
         vRC(Diag(fConst(Rows(@(1).val), @(1).val.scalar))) * vRC(@(1).val.child(1))),

    vRC_CR := Rule([vRC, @(1, CR)], e -> @(1).val.child(1)),

    VTensor_VScale := Rule([@(1, VTensor), [VScale, @(2), @(3), @(4)]], e -> VScale(VTensor(@(2).val, @(1).val.vlen), @(3).val, @(4).val*@(1).val.vlen)),
    VTensor_VTensor := Rule([@(1, VTensor), @(2, VTensor)], e -> VTensor(@(2).val.child(1), @(1).val.vlen*@(2).val.vlen)),


    vRC_SMP := Rule([@(1, vRC), @(2, [SMPSum, SMPBarrier])],
        e -> let(s := @(2).val, CopyFields(s, rec(_children := List(s.children(), c->ObjId(@(1).val)(c)), dimensions := @(1).val.dimensions)))),

    # should RC and VRC be in some other place?
    vRC_TCvt := Rule( [@(0, [vRC, RC, VRC]), @(1, TCvt)],
        e -> let( t := @(1).val, TCvt( 2*t.n(), t.isa_to(), t.isa_from(), t.props()).withTags(t.getTags()).takeAobj(t) )),
));

Class(RulesCxRC_Term, RuleSet);
RewriteRules(RulesCxRC_Term, rec(
    vRC_VContainer := Rule([vRC, @(1, VContainer)], e -> VContainer(vRC(@(1).val.child(1)), @(1).val.isa.uncplx())),

#    RC_VTensor := Rule([ RC, @(1, VTensor) ], e -> RCVTensor(@(1).val.child(1), @(1).val.vlen)),
#
#    RCVTensor_VTensor := Rule(@(1, RCVTensor, e -> e.isReal()), e -> VTensor(@(1).val.child(1), 2*@(1).val.vlen)),

#
#    RC_VGathScat_sv := Rule([@(1, RC), @(2, [VGath_sv, VScat_sv])], e->
#        @(2).val.rcVariant(@(2).val.func, 2*@(2).val.v, @(2).val.sv)),
#
#    RC_VGathScat := Rule([@(1, RC), @(2, [VGath, VScat])], e->
#        ObjId(@(2).val)(@(2).val.func, 2*@(2).val.v)),
#
#    RC_VPrm_x_I := Rule([@(1, RC), @(2, VPrm_x_I)], e ->  VPrm_x_I(@(2).val.func, @(2).val.inverse, 2*@(2).val.v)),
#
#    RC_VDiag_x_I := Rule([@(1, RC), @(2, VDiag_x_I)], e ->  VDiag_x_I(@(2).val.element, 2*@(2).val.v)),
#
#    VContainer_RCDiag := Rule(@@(1, RCDiag, (e, cx) -> IsBound(cx.VContainer) and Length(cx.VContainer)=1),
#        (e, cx) -> RCVDiag(@@(1).val.element, Last(cx.VContainer).isa.getV())),
#
#    RCVTensor_Blk := Rule([@(1, RCVtensor), @(2, Blk)], e -> RCVDiag(RCData(@(2).val.element), 2*@(2).val.v)),

##--- RC rules moved to vRC

## Shared Franz/Yevgen Diag rules
    vRC_Diag      := Rule([@(1, vRC), @(2, Diag) ], e -> vRC(VDiag(@(2).val.element, 1))),

#############################
## Original Franz's rules:

    vRC_VDiag     := Rule([@(1, vRC), @(2, VDiag)], e -> let(v:=2*@(2).val.v, 
        RCVDiag(VData(RCData(@(2).val.element), v), v))),

    vRC_VDiag_x_I := Rule([@(1, vRC), @(2, VDiag_x_I)], e -> let(
        v   := @(2).val.v, 
        elt := @(2).val.element, 
        dup := Cond(IsVecT(elt.range()), elt, VDup(elt, v)), 
        RCVDiag(RCVData(dup), 2*v)
    )),


############################################
## Yevgen's searchable rules for autolib
    vRC_VDiag     := Rule([@(1, vRC), @(2, VDiag)], (e,cx) -> let(
	v   := 2*@(2).val.v, 
	alt := Global.autolib.RCOND("diag", Global.autolib.rcondModeFreedom, maybe(), 
		     RCVDiag(VData(RCData(@(2).val.element), v), v),
		     RCVDiagSplit(RCVDataSplit(@(2).val.element, v), v)),
	Cond( not IsBound(cx.opts.hack_vRC_VDiag), alt.child(2),
	      cx.opts.hack_vRC_VDiag = "search",   alt, 
	      cx.opts.hack_vRC_VDiag = "compact",  alt.child(1),
	      cx.opts.hack_vRC_VDiag = "split",    alt.child(2) ))), # strip alternator outside of autolib

    vRC_VDiag_x_I := Rule([@(1, vRC), @(2, VDiag_x_I)], (e,cx) -> let(
        v   := @(2).val.v, 
        elt := @(2).val.element, 
	t   := elt.range(), 
        # Cond(IsVecT(t), VDup(elt, 2), VDup(elt, 2*v)), v),  # does not work, dimension error
	Cond(IsRealT(t),
	         VDiag_x_I(elt, 2*v),

	     IsVecT(t) and IsRealT(t.t), 
	         VDiag_x_I(VDup(elt, 2), 2*v), # NOTE: make sure sub-vector vdup is properly
		                               #        handled by the compiler rewrite rules
	     # else, complex
	     let(
		 dup := Cond(IsVecT(t), elt, VDup(elt, v)), 
		 alt := Global.autolib.RCOND("diag", Global.autolib.rcondModeFreedom, maybe(), 
                     RCVDiag(RCVData(dup), 2*v),
		     RCVDiagSplit(RCVDataSplitVec(dup), 2*v)), 
		 Cond( not IsBound(cx.opts.hack_vRC_VDiag_x_I),  alt.child(2),  # strip alternator outside of autolib
		     cx.opts.hack_vRC_VDiag_x_I = "search",    alt, 
		     cx.opts.hack_vRC_VDiag_x_I = "compact",   alt.child(1),
		     cx.opts.hack_vRC_VDiag_x_I = "split",     alt.child(2) )
	      )
	))),

    # we need to check the divisibility of dimensions, because all VGathXXX constructs output full vectors
    # This means that gather of 3 points won't be handled here.
    vRC_Gath := Rule([vRC, @@(1, Gath, (e,cx)->IsBound(cx.VContainer) and cx.VContainer<>[] and imod(2*e.func.domain(), Last(cx.VContainer).isa.getV())=0)],
        (e,cx) -> VGath_sv(@@(1).val.func, Last(cx.VContainer).isa.getV(), 2)),

    vRC_Scat := Rule([vRC, @@(1, [Scat,ScatAcc], (e,cx)->IsBound(cx.VContainer) and cx.VContainer<>[] and imod(2*e.func.domain(), Last(cx.VContainer).isa.getV())=0)],
        (e,cx) -> Cond(@@(1).val _is Scat, 
	    VScat_sv(@@(1).val.func, Last(cx.VContainer).isa.getV(), 2),
	    VScat_svAcc(@@(1).val.func, Last(cx.VContainer).isa.getV(), 2))),

#    NOTE: why is this commented out?
#    VA: btw - in fixed size code this happens anyway with help of vRC_REALMAT and RulesVec through VTensor
#    vRC_Prm := Rule([vRC, @(1, Prm)], e -> VPrm_x_I(@(1).val.func, 2)),


    # NOTE: This rule looks evil, as explained below.
    #        Rules for VTensor( VRCDiag | VPRM | ... ) do not exist
    #        and if it fires prematurely (maybe due to some composite operator)
    #        the rewrite system will get stuck later. 
    #
    #        This is exactly what happens in autolib ASP.DFT(N) using _CT1Tr.
    #
    #        Thus I blocked it. I think it should be possible to get rid of
    #        it in a graceful way.
    #
    vRC_REALMAT := Rule([vRC, @@(1).cond(
        (e,cx) -> not IsBound(cx.opts.autolib) and e.isReal() and not ObjId(e) in
            [VRCDiag, VPrm_x_I, VIxJ2, VJxI, VBlk,
             VGath_zero, VScat_zero, VGath_pc, VScat_pc, VScat_pcAcc, IxVGath_pc, IxVScat_pc, 
             VGath_sv, VScat_sv, VScat_svAcc, VGath, VScat, VScatAcc, VGath_u, VScat_u, VScatAcc_u, VStretchGath, VStretchScat, 
             Gath, Scat, ScatAcc])],
        e -> VTensor(@@(1).val, 2)),

    vRC_Blk := Rule([vRC, @(1, Blk, e -> not e.isReal())], e -> RCVBlk(MapMat(@(1).val.element, i->[i]), 2)),
    vRC_Blk1 := Rule([vRC, @(1, Blk1, e -> not e.isReal())], e -> vRC(Blk([[@(1).val.element]]))),

    
    VTensor_XXX := Rule([@(1,VTensor,e->e.vlen=2), @(2,[VGath, VScat, VScatAcc, VGath_sv, VScat_sv, VScat_svAcc, VGath_u, VScat_u, VScatAcc_u, VPrm_x_I])], e->vRC(e.child(1))),

    VTensor_I := Rule([@(1, VTensor), [I, @(2)]], e -> VPrm_x_I(fId(@(2).val), e.vlen)), 

    vRC_VGath := Rule([@(1, vRC), @(2, VGath)], e -> VGath(@(2).val.func, 2*@(2).val.v)),
    vRC_VScat := Rule([@(1, vRC), @(2, [VScat, VScatAcc])], e -> ObjId(@(2).val)(@(2).val.func, 2*@(2).val.v)),
    
    vRC_VGath_u := Rule([@(1, vRC), @(2, VGath_u)], e -> VGath_u(fTensor(@(2).val.func, fBase(2, 0)), 2*@(2).val.v)),
    vRC_VScat_u := Rule([@(1, vRC), @(2, [VScat_u, VScatAcc_u])], e -> ObjId(@(2).val)(fTensor(@(2).val.func, fBase(2, 0)), 2*@(2).val.v)),
    
    vRC_VGath_sv := Rule([@(1, vRC), @(2, VGath_sv)], e -> VGath_sv(@(2).val.func, 2*@(2).val.v, 2*@(2).val.sv, Cond(@(2).val.rem _is Unk, Unk(TInt), 2*@(2).val.rem))),
    vRC_VScat_sv := Rule([@(1, vRC), @(2, [VScat_sv, VScat_svAcc])], e -> ObjId(@(2).val)(@(2).val.func, 2*@(2).val.v, 2*@(2).val.sv, Cond(@(2).val.rem _is Unk, Unk(TInt), 2*@(2).val.rem))),

    vRC_VStretchGath := Rule([@(1, vRC), @(2, VStretchGath)], e -> vRCStretchGath(@(2).val.func, @(2).val.part, 2*@(2).val.v)),
    vRC_VStretchScat := Rule([@(1, vRC), @(2, VStretchScat)], e -> vRCStretchScat(@(2).val.func, @(2).val.part, 2*@(2).val.v)),

    fix_VGathScat := Rule([@(1, [VGath, VScat, VScatAcc]), [@(2,fTensor), ...,
            [fId, @@(3).cond((e, cx)->IsBound(cx.VContainer) and cx.VContainer <> [] and @(1).val.v < Last(cx.VContainer).isa.getV() and IsInt(@(1).val.v*e / Last(cx.VContainer).isa.getV())
            and (not IsBound(cx.VTensor) or (IsBound(cx.VTensor) and cx.VTensor = []))
            and (not IsBound(cx.vRC) or (IsBound(cx.vRC) and cx.vRC = []))
	    and not (IsBound(cx.opts.assumeOuter_vRC) and cx.opts.assumeOuter_vRC)
             ) ]]],
        (e, cx) -> ObjId(@(1).val)(fTensor(DropLast(@(2).val.children(), 1), fId(@(1).val.v*@@(3).val / Last(cx.VContainer).isa.getV())), Last(cx.VContainer).isa.getV())),

    diagTensor_fPrecompute := Rule([diagTensor, [fPrecompute, @(1)], @(2)], x->fPrecompute(diagTensor(@(1).val, @(2).val))),
    # shouldn't this be in some other place:
    fStretch_fPrecompute := Rule([fStretch, @(1, fPrecompute), @(2), @(3)], e -> fPrecompute(fStretch(@(1).val.child(1), @(2).val, @(3).val))), 

    RCVData_fPrecompute := Rule([@(0,[RCVData,RCVDataSplitVec,VDataRDup]), [fPrecompute, @(1)]], 
	x->fPrecompute(ObjId(@(0).val)(@(1).val))), 
    RCVData2_fPrecompute := Rule([@(0,[RCVDataSplit]), [fPrecompute, @(1)], @(2)], 
	x->fPrecompute(ObjId(@(0).val)(@(1).val, @(2).val))), 

    RCVData_fCompose    := Rule([RCVData, @(1, fCompose, x -> not IsType(Last(x.children()).range()))], 
        e -> let(ch := @(1).val.children(),
	         chsplit := SplitBy(ch, x->IsType(x.range())),
		 fCompose(RCVData(fCompose(chsplit[1])), fCompose(chsplit[2])))), 

    RCVDataSplitVec_fCompose    := Rule([RCVDataSplitVec, @(1, fCompose, x -> not IsType(Last(x.children()).range()))], 
        e -> let(ch := @(1).val.children(),
	         chsplit := SplitBy(ch, x->IsType(x.range())),	         
		 fCompose(RCVDataSplitVec(fCompose(chsplit[1])), fTensor(fCompose(chsplit[2]), fId(2))))), 

    # NOTE: potential problem -- RCVDataSplit_fCompose rule is missing

    RCVData_diagDirsum  := Rule([@(0,[RCVData,RCVDataSplitVec]), @(1,diagDirsum)], 
        e -> ApplyFunc(diagDirsum, List(@(1).val.children(), ObjId(@(0).val)))),
    
    RCVDataSplit_diagDirsum  := Rule([ @(0, RCVDataSplit, x -> x.v mod 2 = 0), 
                                   @(1, diagDirsum, x -> x.subdomainsDivisibleBy(@(0).val.v/2)), 
                                   @(2)], 
        e -> ApplyFunc(diagDirsum, List(@(1).val.children(), x->ObjId(@(0).val)(x, @(2).val)))),
    # pull out fPrecompute because it's stuck in there, RCVDataSplit cannot go through diagDirsum
    RCVDataSplit_diagDirsum_fPrecompute  := Rule( [@(1, diagDirsum), ..., @@(2, fPrecompute, (x, cx) -> cx.isInside(RCVDataSplit) and not @(1).val.subdomainsDivisibleBy(Last(cx.RCVDataSplit).v/2) ), ...],
        e -> fPrecompute(SubstTopDown(e, fPrecompute, ee -> ee.child(1)))),

    # When vector length is 2 we can go through compose
    RCVData2_fCompose_diagDirsum := Rule([RCVDataSplit, [@(1, fCompose), @(2, diagDirsum), ...], @(3).cond( x -> x = 2)], 
        e -> fCompose(RCVDataSplit(@(2).val, 2), fTensor(fCompose(Drop(@(1).val.children(),1)), fId(2)))),

    VDup_fCompose := Rule([VDup, [@(1,fCompose), @(2), ...], @(3)],
        e -> fCompose(VDup(@(2).val, @(3).val), fCompose(Drop(@(1).val.children(),1)))),

    VDup_diagDirsum := Rule([VDup, @(1,diagDirsum), @(2)], 
        e -> ApplyFunc(diagDirsum, List(@(1).val.children(), x->VDup(x, @(2).val)))),

    # YSV: Previously RCVBlk(...) was created with complex vector length,
    #      now we use real vlen for consistency
    vRC_VTensor_Blk := Rule([@(0, vRC), [@(1, VTensor), @(2, Blk)]],
        e -> let(v := @(1).val.vlen, 
	         RCVBlk(MapMat(@(2).val.element, e -> Replicate(v, e)), 2*v))),

    vRC_VGath_pc := Rule([@(1, vRC), @(2, VGath_pc)],
        e -> let(g := @(2).val, VGath_pc(g.N*2, g.n*2, g.ofs*2, g.v*2))),

    vRC_VScat_pc := Rule([@(1, vRC), @(2, [VScat_pc, VScat_pcAcc])],
        e -> let(s := @(2).val, ObjId(@(2).val)(s.N*2, s.n*2, s.ofs*2, s.v*2))),

    vRC_IxVGath_pc := Rule([@(1, vRC), @(2, IxVGath_pc)],
        e -> let(g := @(2).val, IxVGath_pc(g.k, g.N*2, g.n*2, g.ofs*2, g.v*2))),

    vRC_IxVScat_pc := Rule([@(1, vRC), @(2, IxVScat_pc)],
        e -> let(s := @(2).val, IxVScat_pc(s.k, s.N*2, s.n*2, s.ofs*2, s.v*2))),

    vRC_VIxJ2 := Rule([vRC, @(1, VIxJ2)], e -> RCVIxJ2(@(1).val.v*2)),
    vRC_VJxI  := Rule([vRC, @(1, VJxI)], e -> VJxI(@(1).val.m, @(1).val.v*2)),

    vRC_VBlk_real := Rule([vRC, @(1, VBlk, e -> e.isReal())], 
	e -> VBlk(MapMat(@(1).val.element, i -> 
		TVect(i.t.t.realType(), 2*@(1).val.v).value(
		    ConcatList(i.v, j -> [j, j]))), 2*@(1).val.v)),

    vRC_VBlk_complex := Rule([vRC, @(1, VBlk, e -> not e.isReal())], 
	e -> RCVBlk(MapMat(@(1).val.element, _unwrap), 2*@(1).val.v)),

    vRC_VGath_zero := Rule([vRC, @(1, VGath_zero)], e -> VGath_zero(@(1).val.N, @(1).val.n, @(1).val.v*2)),

    vRC_VScat_zero := Rule([vRC, @(1, VScat_zero)], e -> VScat_zero(@(1).val.N, @(1).val.n, @(1).val.v*2)),

    vRC_VPrm_x_I := Rule([vRC, @(1, VPrm_x_I)], e -> VPrm_x_I(@(1).val.func, @(1).val.v*2)),

    sv2__Scat_sv_fTensor := Rule([@(1, [VScat_sv, VScat_svAcc]), [@(2,fTensor), ..., [fId, @(3).cond(e -> IsInt(@(1).val.sv*EvalScalar(e)/@(1).val.v))]]],
        e -> let(v := @(1).val.v, sv := @(1).val.sv, n := EvalScalar(@(3).val), newn := sv*n/v,
	         scat := Cond(@(1).val _is VScat_sv, VScat, VScatAcc),
             scat(fTensor(DropLast(@(2).val.children(), 1), fId(newn)), v))),

    sv2__Gath_sv_fTensor := Rule([@(1, VGath_sv), [@(2,fTensor), ..., [fId, @(3).cond(e -> IsInt(@(1).val.sv*EvalScalar(e)/@(1).val.v))]]],
        e -> let(v := @(1).val.v, sv := @(1).val.sv, n := EvalScalar(@(3).val), newn := sv*n/v,
             VGath(fTensor(DropLast(@(2).val.children(), 1), fId(newn)), v))),

#    vRC_VRCDiag := Rule([vRC, [@(1, VRCDiag), @(2, VDup)]],
#        e -> let(v := @(1).val.v*@(2).val.v, VRCDiag(VDup(@(2).val.func, v), v))),
#
#    vRC_VRCDiag_fPrecompute := Rule([vRC, [@(1, VRCDiag), [fPrecompute, @(2, VDup)]]],
#        e -> let(v := @(1).val.v*@(2).val.v, VRCDiag(fPrecompute(VDup(@(2).val.func, v)), v)))
#

    # NOTE: this is a recipe for disaster
#    vRC_VRCDiag_VDup := Rule([vRC, @(1, VRCDiag, e->Length(Collect(e.element, VDup))=1)],
#        e -> let(vdup := Collect(@(1).val.element, VDup)[1], v := 2*vdup.v, VRCDiag(SubstTopDown(@(1).val.element, @(2, VDup), i->VDup(@(2).val.func, v)), v))),

    vRC_VRCDiag := Rule([vRC, @(1, VRCDiag)], e -> VRCDiag(VDataRDup(@(1).val.element), 2*@(1).val.v)),

    vRC_I := Rule( [@(0, vRC), @(1, I)],
        e -> I(2*@(1).val.params[1])),

    vRC_VOLMultiplication := Rule([vRC, @(1, VOLMultiplication)],
        e -> let(p := @(1).val.rChildren(), RCVOLMultiplication(p[1], p[2], 2*p[3]))),
    vRC_VOLConjMultiplication := Rule([vRC, @(1, VOLConjMultiplication)],
        e -> let(p := @(1).val.rChildren(), RCVOLConjMultiplication(p[1], p[2], 2*p[3]))),

));

RulesCxRC := MergedRuleSet(RulesCxRC_Op, RulesCxRC_Term);
