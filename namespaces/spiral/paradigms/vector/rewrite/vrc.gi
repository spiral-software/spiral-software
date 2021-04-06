
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# helps translate VTensorInd(Scat(f), i) -> VScat_sv(f \tensor_i fId(i.range))
_fTensorInd := function(fnc, idx)
    local ds, irng, fdom, nidx, lambda;
    irng := idx.range;

    if not (idx in fnc.free()) then
        return fTensor(fnc, fId(irng));
    fi;

    fdom := fnc.domain();
    nidx := Ind(irng*fdom);

    lambda := Lambda(nidx, imod(nidx, irng)+V(irng)*Lambda(idx, fnc.at(idiv(nidx, irng))).at(imod(nidx, irng))).setRange(V(fnc.range())*V(irng));
    return lambda;
end;

#prevent VRC Rules from introducing VRCL/VRCR in the wrong spot
DontNeedSpecialVRC := e->not ForAny(e.children(), f->IsBound(f.needSpecialVRC) and f.needSpecialVRC());

# Test for VRCs that, for any construct, has a needSeparateVRC set
anyNeedSeparateVRC := e->ForAny(e.children(), f->IsBound(f.needSeparateVRC) and f.needSeparateVRC());

# Test for VRCs that for all constructs, cannot change data formats
TotallyNonChangeable := function(e)
    if ObjId(e)=Compose or ObjId(e)=ComposeDists then
      return(ForAll(e.children(), f->IsBound(f.totallyCannotChangeDataFormat) and f.totallyCannotChangeDataFormat()));
    else
      return(IsBound(e.totallyCannotChangeDataFormat) and e.totallyCannotChangeDataFormat());
    fi;
end;

# Test for VRCs that have at least one construct that cannot change data formats
HasNonchangeable := function(e)
    if ObjId(e)=Compose or ObjId(e)=ComposeDists then
      return(ForAny(e.children(), f->IsBound(f.cannotChangeDataFormat) and f.cannotChangeDataFormat()));
    else
      return(IsBound(e.cannotChangeDataFormat) and e.cannotChangeDataFormat());
    fi;
end;

NoSpecialOrNonchangeableVRC := e->(DontNeedSpecialVRC(e) and not HasNonchangeable(e));
NoSpecialAndChangeableVRC   := e->(DontNeedSpecialVRC(e) and     HasNonchangeable(e));

HandleCannotChangeDataFormatVRC := function(vrc, ch, v)
    local remch, ele, left, right, i, vrc1, vrc2;

    remch := Compose(Drop(ch, 1));
    #Error("BP: cannotChangeDataFormatVRC");

    if vrc=VRC then
        # NOTE: How do we handle this?
        # If ALL children cannot change data format, simply distribute the VRC amongst them

        if TotallyNonChangeable(Compose(ch)) then
            return( Compose(VRC(ch[1], v), VRC(remch,v)) );
        fi;

        # if any of the children NOTE: this is inefficient: what we really
        # should to is to bunch together the children that needSeparateVRC, and
        # the children that are regular, and handle them separately. For now,
        # this hack works because we should have no cases where such children
        # will be mixed together.

        if anyNeedSeparateVRC(Compose(ch)) then
            return( Compose(VRC(ch[1], v), VRC(remch,v)) );
        fi;


        # If not, we must do a VRCR/VRCL split
        # Find the rightmost child that can change dataformats. That and
        # everything right of that becomes a VRCL. Everything left of that is a VRCR.

        # Bad (but good:)) HACK: for things of the form S*A*G, where S and G
        # cannot change data formats, stuff the VRC into A, and hope that A is
        # big enough to handle it.

        if Length(ch)=3 and TotallyNonChangeable(ch[1]) and TotallyNonChangeable(ch[3]) then
          return(Compose( VRC(ch[1], v), VRC(ch[2], v), VRC(ch[3], v) ));
        fi;

        for i in Reversed([1..Length(ch)]) do
          if not TotallyNonChangeable(ch[i]) then
            # We found the rightmost child
            #Error("VRC: BP");
            left  := Compose(List([1..(i-1)], e->ch[e]));
            right := Compose(List([i..Length(ch)], e->ch[e]));
            #Error("VRC: BP");
            return( Compose( VRCR(left, v), VRCL(right, v) ) );
          fi;
        od;

        #Error("BP: cannotChangeDataformatVRC: VRC needs to be split into VRCL/VRCR");
    fi;

    if vrc=VRCLR then
        # This should really be handled as if the cannotChangeDataformat doesn't exist.
        #Error("VRCLR: BP");
        vrc1 := [[VRCL, VRCR], [VRCLR, VRCLR]];
        vrc2 := When(NeedInterleavedRight(ch[1]) or NeedInterleavedLeft(ch[2]), vrc1[1], vrc1[2]);
        #Error("VRCLR: BP");
        return( Compose(vrc2[1](ch[1], v), vrc2[2](Compose(Drop(ch, 1)), v)) );
    fi;

    if vrc=VRCL then
        # Find the first child from the right that is capable of making a
        # format change. We then have 2 cases.

        # Note: we really have 3 cases to be general: VRCL -> (VRCLR, VRCL, VRC)
        # Sometimes, the LR won't exist, and sometimes, the VRC won't exist
        # But it's easier to code if we only always break down to 2 cases

        for i in Reversed([1..Length(ch)]) do
          if not TotallyNonChangeable(ch[i]) then
            # We found the child that will do the dataformat change (VRCL)
            # We now have 2 cases.
            #   Case 1: VRCL -> (VRCL,  VRC)   (found child is not rightmost)
            #   Case 2: VRCL -> (VRCLR, VRCL)  (found child is rightmost)

            if i <> Length(ch) then # Case 1
              left  := Compose(List([1..i],              e->ch[e]));
              right := Compose(List([(i+1)..Length(ch)], e->ch[e]));
              #Error("BP:VRCL1\n, ---------VRCL(left)---------", left, "\n---------VRC(right)--------", right);
              return( Compose( VRCL(left, v), VRC(right, v) ) );
            else                    # Case 2
              left  := Compose(List([1..(i-1)], e->ch[e]));
              right := ch[i];
              #Error("BP:VRCL2\n, ---------VRCLR(left)---------", left, "\n---------VRCL(right)--------", right);
              return( Compose( VRCLR(left, v), VRCL(right, v) ) );
            fi;
          fi;
        od;
        Error("cannotChangeDataFormatVRC/L: Didn't find an appropriate child");
    fi;

    if vrc=VRCR then
        # Find the first child from the left that is capable of making a
        # format change. We then have 2 cases.

        for i in [1..Length(ch)] do
          if not TotallyNonChangeable(ch[i]) then
            # We found the child that will do the dataformat change (VRCR)
            # We now have 2 cases.
            #   Case 1: VRCR -> (VRC,  VRCR)   (found child is not leftmost)
            #   Case 2: VRCR -> (VRCR, VRCLR)  (found child is leftmost)

            if i <> 1 then # Case 1
              left  := Compose(List([1..(i-1)],      e->ch[e]));
              right := Compose(List([i..Length(ch)], e->ch[e]));
              #Error("BP:VRCR1\n, ---------VRC(left)---------", left, "\n---------VRCR(right)--------", right);
              return( Compose( VRC(left, v), VRCR(right, v) ) );
            else                    # Case 2
              left  := ch[i];
              right := Compose(List([(i+1)..Length(ch)], e->ch[e]));
              #Error("BP:VRCR2\n, ---------VRCR(left)---------", left, "\n---------VRCLR(right)--------", right);
              return( Compose( VRCR(left, v), VRCLR(right, v) ) );
            fi;
          fi;
        od;
        Error("cannotChangeDataFormatVRC/R: Didn't find an appropriate child");

    fi;

    Error("cannotChangeDataFormat: I don't know what to do with this one. None of the cases matched.");

#            vrc=VRCL and     HasNonchangeable(ch1) and not HasNonchangeable(ch2), Compose(VRCLR(ch1, v), VRCL(remch, v) ),
#            vrc=VRCL and not HasNonchangeable(ch1) and     HasNonchangeable(ch2), Compose(VRCL(ch1, v),  VRC(remch,v)   ),
#            vrc=VRCL and     HasNonchangeable(ch1) and     HasNonchangeable(ch2),
#                When(NeedInterleavedRight(ch1) or NeedInterleavedLeft(ch2), Compose(VRCL(I(ch1.dims()[1]),v),  VRC(ch,v)),
#                                                                            ComposeVRCLR(ch,v), VRCL(I(ch1.dims()[1]),v)),
#
#            vrc=VRCR and     HasNonchangeable(ch1) and not HasNonchangeable(ch2), Compose(VRC(ch1, v),   VRCR(remch, v) ),
#            vrc=VRCR and not HasNonchangeable(ch1) and     HasNonchangeable(ch2), Compose(VRCR(ch1, v),  VRCLR(remch,v) ),
#            vrc=VRCR and     HasNonchangeable(ch1) and     HasNonchangeable(ch2),
#                When(NeedInterleavedRight(ch1) or NeedInterleavedLeft(ch2), Compose(VRCL(I(ch1.dims()[1]),v),  VRC(ch,v)),
#                                                                            ComposeVRCLR(ch,v), VRCL(I(ch1.dims()[1]),v))
#          )
#        )
#    ),

end;



Scat.needInterleavedLeft:=False;
Scat.needInterleavedRight:=False;

# NOTE: Why were these set to False???
Gath.needInterleavedLeft:=True;
Gath.needInterleavedRight:=True;

VBlk.needInterleavedLeft:=False;
VBlk.needInterleavedRight:=False;

# NOTE: remove Diag here
Diag.needInterleavedLeft:=False;
Diag.needInterleavedRight:=False;

Inplace.needInterleavedRight := self >> self.child(1).needInterleavedRight();
Inplace.needInterleavedLeft := self >> self.child(1).needInterleavedLeft();
Inplace.cannotChangeDataFormat := self >> self.child(1).cannotChangeDataFormat();
Inplace.totallyCannotChangeDataFormat := self >> self.child(1).totallyCannotChangeDataFormat();

BB.needInterleavedRight := self >> self.child(1).needInterleavedRight();
BB.needInterleavedLeft := self >> self.child(1).needInterleavedLeft();
BB.cannotChangeDataFormat := self >> self.child(1).cannotChangeDataFormat();
BB.totallyCannotChangeDataFormat := self >> self.child(1).totallyCannotChangeDataFormat();

VContainer.needInterleavedRight := self >> self.child(1).needInterleavedRight();
VContainer.needInterleavedLeft := self >> self.child(1).needInterleavedLeft();
VContainer.cannotChangeDataFormat := self >> self.child(1).cannotChangeDataFormat();
VContainer.totallyCannotChangeDataFormat := self >> self.child(1).totallyCannotChangeDataFormat();

NoDiagPullin.needInterleavedRight := self >> self.child(1).needInterleavedRight();
NoDiagPullin.needInterleavedLeft := self >> self.child(1).needInterleavedLeft();
NoDiagPullin.cannotChangeDataFormat := self >> self.child(1).cannotChangeDataFormat();
NoDiagPullin.totallyCannotChangeDataFormat := self >> self.child(1).totallyCannotChangeDataFormat();

NoDiagPullinLeft.needInterleavedRight := self >> self.child(1).needInterleavedRight();
NoDiagPullinLeft.needInterleavedLeft := self >> self.child(1).needInterleavedLeft();
NoDiagPullinLeft.cannotChangeDataFormat := self >> self.child(1).cannotChangeDataFormat();
NoDiagPullinLeft.totallyCannotChangeDataFormat := self >> self.child(1).totallyCannotChangeDataFormat();

NoDiagPullinRight.needInterleavedRight := self >> self.child(1).needInterleavedRight();
NoDiagPullinRight.needInterleavedLeft := self >> self.child(1).needInterleavedLeft();
NoDiagPullinRight.cannotChangeDataFormat := self >> self.child(1).cannotChangeDataFormat();
NoDiagPullinRight.totallyCannotChangeDataFormat := self >> self.child(1).totallyCannotChangeDataFormat();

SymSPL.needInterleavedRight := self >> self.child(1).needInterleavedRight();
SymSPL.needInterleavedLeft := self >> self.child(1).needInterleavedLeft();
SymSPL.cannotChangeDataFormat := self >> self.child(1).cannotChangeDataFormat();
SymSPL.totallyCannotChangeDataFormat := self >> self.child(1).totallyCannotChangeDataFormat();

VGath_sv.rcVariant := RCVGath_sv;
VScat_sv.rcVariant := RCVScat_sv;

RCVGath_sv.vecVariant := (f,v)->VGath(f,v);
RCVScat_sv.vecVariant := (f,v)->VScat(f,v);

ScatGath.needInterleavedLeft:=True;
ScatGath.needInterleavedRight:=True;

SMPBarrier.needInterleavedLeft := (self) >> self.child(1).needInterleavedLeft();
SMPBarrier.needInterleavedRight := (self) >> self.child(1).needInterleavedRight();

_VRCFamily := [VRC, VRCL, VRCR, VRCLR];

Class(RulesVRC, RuleSet);
RewriteRules(RulesVRC, rec(
    VRC_VContainer := Rule([@(1, _VRCFamily), @(2, VContainer)], 
	e -> let(cont := @(2).val, 
	    VContainer(ObjId(@(1).val)(cont.child(1), @(1).val.v), cont.isa))),

    VRC_ISum := Rule([@(1, _VRCFamily), @(2, [ISum, SMPSum, SMPBarrier, SUM])], e->let(s := @(2).val,
    CopyFields(s, rec(_children := List(s.children(), c->ObjId(@(1).val)(c, @(1).val.v)),
                  dimensions := @(1).val.dimensions)))),

    VRC_Container := Rule([@(1, _VRCFamily),  @(2, [BB,Buf,Inplace,RecursStep,NoDiagPullin, NoDiagPullinLeft, NoDiagPullinRight])],
        e -> ObjId(@(2).val)(ObjId(@(1).val)(@(2).val.child(1), @(1).val.v))),

    VRC_Data := Rule([@(1, _VRCFamily), @(2, Data)], 
	e -> Data(@(2).val.var, @(2).val.value, ObjId(e)(@(2).val.child(1)))),

#   VRC_SymSPL := Rule([@(1, _VRCFamily), @(2, SymSPL)],
#        e -> ObjId(@(1).val)(@(2).val.child(1), @(1).val.v)),

   VRC_Compose := Rule([@(1, _VRCFamily), @(2, [Compose,ComposeStreams]).cond(NoSpecialOrNonchangeableVRC)], # Use with VRC_ComposeNonchangeable below
        e->let(v := @(1).val.v, ch := @(2).val.children(), vrc := ObjId(@(1).val),
        vrc1 := Cond(
                 vrc=VRC,   [[VRC,  VRC ], [VRCR,  VRCL ]],
                 vrc=VRCLR, [[VRCL, VRCR], [VRCLR, VRCLR]],
                 vrc=VRCL,  [[VRCL, VRC ], [VRCLR, VRCL ]],
                 vrc=VRCR,  [[VRC,  VRCR], [VRCR,  VRCLR]]),
        vrc2 := When(NeedInterleavedRight(ch[1]) or NeedInterleavedLeft(ch[2]), vrc1[1], vrc1[2]),
            Compose(vrc2[1](ch[1], v), vrc2[2](Compose(Drop(ch, 1)), v)))),

    # NOTE: When building the format-change I() below, ch1.dims()[1] is
    #        always used. This is probably incorrect in some cases.
    VRC_ComposeNonchangeable := Rule([@(1, _VRCFamily), @(2, Compose).cond(NoSpecialAndChangeableVRC)],
        e->let(v     := @(1).val.v,
               ch    := @(2).val.children(),
               vrc   := ObjId(@(1).val),
               HandleCannotChangeDataFormatVRC(vrc, ch, v)
        )
    ),

    #VRC_ComposeStreams := Rule([@(1, VRC), @(2, ComposeStreams)],
    #    e->let(v := @(1).val.v, ch := @(2).val.children(),
    #        ComposeStreams( VRC(ch[1], v), VRC(ComposeStreams(Drop(ch,1)), v) )
    #        )
    #),

    VRCLR_VRCDiag:= Rule([@(1, _VRCFamily), @(2, VRCDiag)], e -> ObjId(@(1).val)(@(2).val.toloop().sums(), @(1).val.v)),

    VRCLR_VBlk:= Rule([@(1, VRCLR),@(2, VBlk)], e -> VBlk(RealVMatComplexVMat(@(2).val.element), @(2).val.v)),

    VRCLR_VGath_zero:= Rule([@(1, VRCLR),@(2, VGath_zero)], e -> let(g:=@(2).val,
        VRCLR(VGath(fAdd(g.N, g.n, 0), getV(g)), getV(g)))),

    VRCLR_VGathScat := Rule([@(1, VRCLR), @(2, [VGath, VScat, VScatAcc])],
        e -> ObjId(@(2).val)(fTensor(@(2).val.func, fId(2)), @(2).val.v)),

    VRCLR_GathScat := Rule([@(1, VRCLR), @(2, [Gath, Scat, ScatAcc])],
        e -> ObjId(@(2).val)(fTensor(@(2).val.func, fId(2)))),

    VRC_VGathVScat := Rule([@(1, VRC), @(2, [VGath, VScat, VScatAcc])],
        e -> ObjId(@(2).val)(fTensor(@(2).val.func, fId(2)), @(2).val.v)),

    VRCLR_VGathScat_sv := Rule([@(1, VRCLR), @(2, [VGath_sv, VScat_sv])], e->  # NOTE: acc variant missing
        @(2).val.rcVariant(@(2).val.func, @(2).val.v, @(2).val.sv, @(2).val.rem)),

    VRCLR_VTensor := Rule([@(1, VRCLR), @(2, VTensor)],
        e -> VTensor(RC(@(2).val.child(1)), @(2).val.vlen)),

    VRCLR_Diag := Rule([@(1, VRCLR), @(2, [Diag, VDiag])],
        e -> let(d := @(2).val.element, n := d.domain(), v := @(1).val.v,
            VRCDiag(VData(fCompose(RCData(d), fTensor(fId(n / v), L(2*v, 2))), v), v))),

    VRC_BlockVPerm := Rule([@(1, _VRCFamily), @(2, [BlockVPerm, BlockVPerm2])],
        e->let(bd := @(2).val.child(1), vrc := ObjId(@(1).val),
            BlockVPerm(@2.val.n, @2.val.vlen, vrc(bd, @(1).val.v),
            MatSPL(vrc(@2.val.perm, @1.val.v))))),

    fPrecompute_VData := Rule([VData, [fPrecompute, @(1)], @(2)], e -> fPrecompute(VData(@(1).val, @(2).val))),
    fPrecompute_VDup := Rule([VDup, [fPrecompute, @(1)], @(2)], e -> fPrecompute(VDup(@(1).val, @(2).val))),

    RCVScat_sv__fId := Rule(@(1,RCVScat_sv,e->IsInt(2*e.func.domain()/getV(e)) and ObjId(e.func) = fId),
        e->let(v:=getV(@(1).val), When(Cols(@(1).val)<>2*@(1).val.func.domain(),
            VGath_zero(Cols(@(1).val)/v, 2*@(1).val.func.domain()/v, v), VGath(fId(2*@(1).val.func.domain()/v), v))))
));

Class(RulesVRCTermDiag, RuleSet);
RewriteRules(RulesVRCTermDiag, rec(
    VRCLR_VDiag_x_I := Rule([@(1, VRCLR), @(2, VDiag_x_I)], e->VTensor(RC(Diag(@(2).val.element)), @(2).val.v))
));

Class(RulesVRCTerm, RulesVRCTermDiag);
RewriteRules(RulesVRCTerm, rec(
    VIxL := Rule(VIxL, (e, cx) -> e.implement(cx.opts.vector.isa)),
    VL := Rule(VL, (e, cx) -> e.implement(cx.opts.vector.isa)),
#--
### Rules for SAR -- to be moved and fixed...

    Term_VTensorInd_ScatGath := Rule([@(1, VTensorInd), @(2, ScatGath), @(3)],
        e->ScatGath(_fTensorInd(@(2).val.sfunc, @(3).val), _fTensorInd(@(2).val.gfunc, @(3).val))),

    RC_ScatGath := Rule([@@(1, RC), @(2, ScatGath)], (e,cx)->@(2).val.toloopRCVec(@(2).val.maxBkSize(), Last(cx.VContainer).isa.getV())),
##    VRCLR_ScatGath := Rule([@(1, VRCLR), @(2, ScatGath)], e->@(2).val.toloopRCVec(@(2).val.maxBkSize(), @(1).val.v)),

    RCVGathRCSVcat_sv_fTensor := Rule([@(1, [RCVGath_sv, RCVScat_sv]),
                            [@(2,fTensor), ..., [fId, @(3).cond(e -> Gcd(@(1).val.v/@(1).val.sv, EvalScalar(e)) = @(1).val.v)]]],
     e -> let(v := @(1).val.v, sv := @(1).val.sv,
          n := EvalScalar(@(3).val),   gcd := Gcd(v / sv, n),
          @(1).val.vecVariant(fTensor(DropLast(@(2).val.children(), 1), fId(2*n/gcd)), v))),

#    Id_RCV1 := Rule([@(1,[RCVScat_sv,RCVGath_sv]), @(2,fId,e->(IsValue(e.n) or IsInt(e.n)) and IsInt(EvalScalar(e.n/@(1).val.v)))], e->I(EvalScalar(@(2).val.n))),
#
#    Term__VScat_ScatGath := ARule(Compose, [@(1, VScat), @@(2, ScatGath, (e,cx)->not ForAny(_VRCFamily, i->IsBound(cx.(i.name)) and cx.(i.name) <> []))],
#        e->[@(1).val * @@(2).val.toloopVec(@@(2).val.maxBkSize(), @(1).val.v)]),
#
#    # in the precond of this rule there was a guard on VRCXX, but that was never matching due to an error. so I dont know if it was needed at all...
#    Term__ScatGath := Rule(@@(1, ScatGath, (e,cx)->not ForAny([VTensor, VTensorInd], i->IsBound(cx.(i.name)) and cx.(i.name) <> []) and
#        (IsBound(cx.VContainer) and cx.VContainer <> [])),
#        (e,cx)->@@(1).val.toloopVec(@@(1).val.maxBkSize(), Last(cx.VContainer).isa.getV())),
#
    Term_NeedInterleavedComplex_VRC := Rule([@(1, VRC), @(2, NeedInterleavedComplex)], e->RC(@(2).val.child(1))),
#    Term_NeedInterleavedComplex_RC := Rule([@(1, RC), @(2, NeedInterleavedComplex)], e->DRC(@(2).val.child(1))),
#
#    RC_GathScat_sv_fTensor := Rule([@(1, [VGath_sv, VScat_sv]), [@(2,fTensor), ..., [fId, 2]]],
#        e -> let(v := @(1).val.v, sv := @(1).val.sv,
#            @(1).val.rcVariant(fTensor(DropLast(@(2).val.children(), 1)), v, sv))),
#
    RC_XXX := Rule([@(1, RC), @(2, [VGath, VScat, VScatAcc])], e->ObjId(@(2).val)(fTensor(@(2).val.func, fId(2)), @(2).val.v)),
#    RC_XXX_sv := Rule([@(1, RC), @(2, [VGath_sv, VScat_sv])], e->@(2).val.rcVariant(@(2).val.func, @(2).val.v, @(2).val.sv)),
### end SAR
#--
#   these rules are buggy, as Scat_sv does zero padding.
#    RCVScat_sv_toVScat := Rule(@(1, RCVScat_sv, e->e.sv=1 and e.v=2), e -> VScat(e.func, e.v)),
#    RCVGath_sv_toVGath := Rule(@(1, RCVGath_sv, e->e.sv=1 and e.v=2), e -> VGath(e.func, e.v)),

#-- VPrm_x_I --

    #NOTE: Check if this rule is correct!
    #VRC_VPrm_x_I := Rule([@(1, VRC), @(2, VPrm_x_I)], e->let(p := @(2).val, v := p.v, b := p.dims()[1]/v,
    #    Compose(
    #        Tensor(I(2), VTensor(Prm(p.func), v)).sums().unroll()
    #    ))),


    # YSV: we use NoDiagPullin to prevent diagonals from going into the loops resulting from Tensor(I(2),...)
    #      because in some cases they can't be sucked in completely and one ends up with multiple Scat * Diag * Scat
    #      sequences inside a SUM, which overlap, implying that it is a SUMAcc. However, Spiral does not know
    #      that, and generates invalid code. I don't know of a better way to handle this at the moment.
    VRCLR_VPrm_x_I := Rule([@(1, [VRCLR, VRC]), @(2, VPrm_x_I)], e->let(p := @(2).val, v := p.v, b := p.dims()[1]/v,
        Compose(
            VTensor(Prm(L(2*b, b)), v),
            NoDiagPullin(Tensor(I(2), VTensor(Prm(p.func), v)).sums().unroll()),
            VTensor(Prm(L(2*b, 2)), v)
        ))),

    VRCL_VPrm_x_I := Rule([@(1, VRCL), @(2, VPrm_x_I)], e->let(p := @(2).val, v := p.v, b := p.dims()[1]/v,
        Compose(
            VIxL(b, 2, v),
            VTensor(Prm(L(2*b, b)), v),
            NoDiagPullin(Tensor(I(2), VTensor(Prm(p.func), v)).sums().unroll()),
            VTensor(Prm(L(2*b, 2)), v)
        ))),

    VRCR_VPrm_x_I := Rule([@(1, VRCR), @(2, VPrm_x_I)], e->let(p := @(2).val, v := p.v, b := p.dims()[1]/v,
        Compose(
            VTensor(Prm(L(2*b, b)), v),
            NoDiagPullin(Tensor(I(2), VTensor(Prm(p.func), v)).sums().unroll()),
            VTensor(Prm(L(2*b, 2)), v),
            VIxL(b, v, v)
        ))),

#-- VGath ------------------
    VRCL_VGath := Rule([@(1, VRCL), @(2, VGath)], e->let(v := @(2).val.v,
        Compose(
            VIxL(@(2).val.func.domain(), 2, v),
            VGath(fTensor(@(2).val.func, fId(2)), v)
        ))),

    VRCL_VScat := Rule([@(1, VRCL), @(2, [VScat, VScatAcc])], e->let(v := @(2).val.v,
        Compose(
            ObjId(@(2).val)(fTensor(@(2).val.func, fId(2)), v),
            VIxL(@(2).val.func.domain(), 2, v)
        ))),

    VRCL_VGath_sv := Rule([@(1, VRCL), @(2, VGath_sv)], e->let(v := @(2).val.v, sv := @(2).val.sv,
        Compose(
            VIxL(Rows(@(2).val)/v, 2, v),
            RCVGath_sv(@(2).val.func, v, sv, @(2).val.rem)
        ))),

    VRCL_IxVGath_pc := Rule([@(1, VRCL), @(2, IxVGath_pc)], e -> let(
	g := @(2).val, v := g.v,
        Compose(
            VIxL(g.k * _roundup(g.n, g.v) / v, 2, v),
            IxRCVGath_pc(g.k, g.N, g.n, g.ofs, v)
        ))),

    VRCLR_IxVGath_pc := Rule([@(1, VRCLR), @(2, IxVGath_pc, x->x.N*x.k mod x.v=0)], e -> let(
	g := @(2).val, v := g.v,
        Compose(
            VIxL(g.k * _roundup(g.n, v) / v, 2, v),
            IxRCVGath_pc(g.k, g.N, g.n, g.ofs, v),
            VIxL(g.k * g.N / v, v, v)
        ))),

    VRCL_VStretchGath := Rule([@(1, VRCL), @(2, VStretchGath)], e -> let(
	g := @(2).val, 
	v := g.v,
	#XXX  Cond(???,
            Compose(
		VIxL(_roundup(Rows(g), v) / v, 2, v),
		RCVStretchGath(g.func, g.part, v))

	#XXX    VStretchGath( 
	#XXX	fCompose(fTensor(fId(g.func.domain()/v), L(2*v, v)), fTensor(g.func, fId(2))),
	#XXX	...,
	#XXX	...)

        )),

    VRCR_VGath_zero:= Rule([@(1, VRCR),@(2, VGath_zero)], e -> let(g:=@(2).val,
        VGath_zero(2*g.N, 2*g.n, g.v) * VIxL(g.N, g.v, g.v))),

#-- VScat ------------------
    VRCR_VScat := Rule([@(1, VRCR), @(2, [VScat, VScatAcc])], e->let(v := @(2).val.v,
        Compose(
            ObjId(@(2).val)(fTensor(@(2).val.func, fId(2)), v),
            VIxL(@(2).val.func.domain(), v, v)
        ))),

    VRCR_VScat_sv := Rule([@(1, VRCR), @(2, VScat_sv)], e->let(v := @(2).val.v, sv := @(2).val.sv, # NOTE: acc variant missing
        Compose(
            RCVScat_sv(@(2).val.func, v, sv, @(2).val.rem),
            VIxL(Cols(@(2).val)/v, v, v)
        ))),

    VRCR_IxVScat_pc := Rule([@(1, VRCR), @(2, IxVScat_pc)], e->let(s:=@(2).val, v := s.v, # NOTE: acc variant missing
        Compose(
            IxRCVScat_pc(s.k, s.N, s.n, s.ofs, v),
            VIxL(s.k*_roundup(s.n, v)/v, v, v)
        ))),

    VRCLR_IxVScat_pc := Rule([@(1, VRCLR), @(2, IxVScat_pc, x->x.N*x.k mod x.v=0)], e -> let(
        s := @(2).val, v := s.v,
        Compose(
            VIxL(s.k * s.N / v, 2, v),
            IxRCVScat_pc(s.k, s.N, s.n, s.ofs, v),
            VIxL(s.k*_roundup(s.n, v)/v, v, v)
        ))),

    VRCR_VStretchScat := Rule([@(1, VRCR), @(2, VStretchScat)], e->let(s:=@(2).val, v := s.v,
        Compose(
            RCVStretchScat(s.func, s.part, v),
            VIxL(_roundup(Cols(s), v)/v, v, v)
        ))),

#----------------------------
    VRCLR_VPerm := Rule([@(1, VRCLR), @(2, VPerm)], e->let(p := @(2).val, v := p.vlen, b := p.dims()[1]/v,
        Compose(
            VTensor(Prm(L(2*b, b)), v),
            Tensor(I(2), p).sums().unroll(),
            VTensor(Prm(L(2*b, 2)), v)
        ))),

    VRCL_VPerm := Rule([@(1, VRCL), @(2, VPerm)], e->let(p := @(2).val, v := p.vlen, b := p.dims()[1]/v,
        Compose(
            VTensor(Prm(L(2*b, b)), v),
            Tensor(I(2), p).sums().unroll(),
            VTensor(Prm(L(2*b, 2)), v),
            VIxL(b, 2, v)
        ))),

    VRCR_VPerm := Rule([@(1, VRCR), @(2, VPerm)], e->let(p := @(2).val, v := p.vlen, b := p.dims()[1]/v,
        Compose(
            VIxL(b, v, v),
            VTensor(Prm(L(2*b, b)), v),
            Tensor(I(2), p).sums().unroll(),
            VTensor(Prm(L(2*b, 2)), v)
        ))),

#----------------------------
    VRCLR_Perm := Rule([@(1, VRCLR), @(2, Prm)], e->let(p := @(2).val, v := getV(@1.val), b := p.dims()[1]/v, n := @(2).val.dims()[2]/v,
        Compose(
            VIxL(n, 2, v),
            RCVScat_sv(fId(p.func.domain()), v, 1),
            RCVGath_sv(p.func, v, 1),
            VIxL(n, v, v)
        ))),

    VRC_Perm := Rule([@(1, VRC), @(2, Prm)], e->let(p := @(2).val, v := getV(@1.val), b := p.dims()[1]/v, n := @(2).val.dims()[2]/v,
            Compose (RCVScat_sv(fId(p.func.domain()), v, 1),
            RCVGath_sv(p.func, v, 1))
        )),

    VRCL_Split := Rule([@(1, VRCL), @(2, [VDiag, VTensor])], e->let(v := getV(@(2).val),
        Compose(
            VRCLR(@(2).val, v),
            VIxL(@(2).val.dims()[2]/v, 2, v)
        ))),

    VRCR_Split := Rule([@(1, VRCR), @(2, [VDiag, VTensor])], e->let(v := getV(@(2).val),
        Compose(
            VIxL(@(2).val.dims()[1]/v, v, v),
            VRCLR(@(2).val, v)
        ))),

    VRC_VTensor := Rule([@(1, VRC), @(2, [VTensor])], e->let(v := getV(@(2).val),
        Compose(
            VIxL(@(2).val.dims()[1]/v, v, v),
            VRCLR(@(2).val, v),
            VIxL(@(2).val.dims()[2]/v, 2, v)
        ))),

    VRC_VDiag := Rule([@(1, VRC), @(2, [VDiag])], e->let(
       v := getV(@(2).val), d := @(2).val.dims()[1],
       Cond(@(2).val.element.isReal(),
	   VDiag(diagTensor(@(2).val.element, fConst(TReal, 2, 1)), @(2).val.v),
	   VIxL(d/v, v, v) * VRCLR(@(2).val, v) * VIxL(d/v, 2, v)))),

    VRC_Blk := Rule([@(1, VRC), @(2, Blk)], e->RC(@(2).val)),

    VRCLR_VScat_zero := Rule([@(1, VRCLR), @(2, VScat_zero)], e->let(s:=@(2).val, VScat_zero(2*s.N, 2*s.n, s.v))),

    VRC_Gath := Rule([@(1, VRC), @(2, Gath)], e->let(g := @(2).val, v := getV(@1.val), b := g.dims()[1]/v, n := @(2).val.dims()[2]/v,
            Compose (VRCLR(VScat_sv(fId(g.func.domain()), v, 1), v),
            VRCLR(VGath_sv(g.func, v, 1), v))
        )),

    FormatPrm_Term := Rule([FormatPrm, @(1)], e -> Prm(@(1).val)),
));

Class(RulesSplitComplex, RuleSet);
RewriteRules(RulesSplitComplex, rec(
    VRCR_IxVScat_pc := ARule(Compose, [[@(1,[Prm, FormatPrm]), @(4,L,e->e.params[2]=2)], [@(2,VRCR), @(3,IxVScat_pc)]],
                e -> let(v:=@(2).val.v, c:=Cols(@(3).val), s:=@(3).val,
                    [IxVScat_pc(2*s.k, s.N, s.n, s.ofs, v), VPrm_x_I(L(2*c/v, 2), v)])),

    VRCL_IxVGath_pc := ARule(Compose, [[@(2,VRCL), @(3,IxVGath_pc)], [@(1,[Prm, FormatPrm]), @(4,L,e->e.params[2]=e.params[1]/2)]],
                e -> let(v:=@(2).val.v, r:=Rows(@(3).val), g:=@(3).val,
                    [VPrm_x_I(L(2*r/v, r/v), v), IxVGath_pc(2*g.k, g.N, g.n, g.ofs, v)])),

    VRCR_VStretchScat := ARule(Compose, [[@(1,[FormatPrm,Prm]), @(4,L,e->e.params[2]=2)], [@(2,VRCR), @(3,VStretchScat)]],
                e -> let(v:=@(2).val.v, c:=Cols(@(3).val), s:=@(3).val,
                    [VStretchScat(fTensor(fId(2), s.func), 2*s.part, v), VPrm_x_I(L(2*c/v, 2), v)])),

    VRCL_VStretchGath := ARule(Compose, [[@(2,VRCL), @(3,VStretchGath)],[@(1,[FormatPrm,Prm]), @(4,L,e->e.params[2]=e.params[1]/2)]],
                e -> let(v:=@(2).val.v, r:=Rows(@(3).val), g:=@(3).val,
                    [VPrm_x_I(L(2*r/v, r/v), v), VStretchGath(fTensor(fId(2), g.func), 2*g.part, v)])),

    VRCR_VScat_sv := ARule(Compose, [[@(1,[FormatPrm,Prm]), @(4,L,e->e.params[2]=2)], [@(2,VRCR), @(3,VScat_sv)]],
                e -> let(v:=@(2).val.v, c:=Cols(@(3).val), s:=@(3).val,
                    [VStretchScat(fTensor(fId(2), s.func), 2, v), VPrm_x_I(L(2*c/v, 2), v)])),

    VRCL_VGath_sv := ARule(Compose, [[@(2,VRCL), @(3,VGath_sv)],[@(1,[FormatPrm,Prm]), @(4,L,e->e.params[2]=e.params[1]/2)]],
                e -> let(v:=@(2).val.v, r:=Rows(@(3).val), g:=@(3).val,
                    [VPrm_x_I(L(2*r/v, r/v), v), VStretchGath(fTensor(fId(2), g.func), 2, v)])),

    VRCR_VScat := ARule(Compose, [[@(1,[Prm, FormatPrm]), @(4,L,e->e.params[2]=2)], [@(2,VRCR), @(3,[VScat, VScatAcc])]],
                e -> let(v:=@(2).val.v, c:=Cols(@(3).val), s:=@(3).val,
                    [ObjId(@(3).val)(fTensor(fId(2), s.func), v), VPrm_x_I(L(2*c/v, 2), v)])),

    VRCL_VGath := ARule(Compose, [[@(2,VRCL), @(3,VGath)],[@(1,[Prm, FormatPrm]), @(4,L,e->e.params[2]=e.params[1]/2)]],
                e -> let(v:=@(2).val.v, r:=Rows(@(3).val), g:=@(3).val,
                    [VPrm_x_I(L(2*r/v, r/v), v), VGath(fTensor(fId(2), g.func), v)])),

    VRCL_VScat := ARule(Compose, [[@(2,VRCL), @(3,[VScat, VScatAcc])],[@(1,Prm), @(4,L,e->e.params[2]=e.params[1]/2)]],
                e -> let(v:=@(2).val.v, r:=Rows(@(3).val), g:=@(3).val,
                    [VPrm_x_I(L(2*r/v, r/v), v), ObjId(@(3).val)(fTensor(fId(2), g.func), v)])),

    VRCR_VTensor1 := ARule(Compose, [[@(1, [FormatPrm, Prm]), @(4,L,e->e.params[2]=2)], [@(2,VRCR), @(3,VTensor)]],
                e -> let(v:=@(2).val.v, c:=Cols(@(3).val), s:=@(3).val,
                    [VPrm_x_I(L(2*c/v, 2), v), VRCLR(s, v)])),

    VRCR_VTensor1a := ARule(Compose, [[@(1, [FormatPrm, Prm]), @(4,L,e->e.params[2]=2)], [@(2,VRC), @(3,VTensor)]],
                e -> let(v:=@(2).val.v, c:=Cols(@(3).val), s:=@(3).val,
                    [VPrm_x_I(L(2*c/v, 2), v), VRCL(s, v)])),

    VRCL_VTensor2 := ARule(Compose, [[@(2,VRCL), @(3,VTensor)],[@(1, [FormatPrm, Prm]), @(4,L,e->e.params[2]=e.params[1]/2)]],
                e -> let(v:=@(2).val.v, c:=Cols(@(3).val), g:=@(3).val,
                    [VRCLR(g,v), VPrm_x_I(L(2*c/v, c/v), v)])),

    VRCL_VTensor2a := ARule(Compose, [[@(2,VRC), @(3,VTensor)],[@(1, [FormatPrm, Prm]), @(4,L,e->e.params[2]=e.params[1]/2)]],
                e -> let(v:=@(2).val.v, r:=Rows(@(3).val), g:=@(3).val,
                    [VRCR(g,v), VPrm_x_I(L(2*r/v, r/v), v)])),

#    VRCLR_VGath_split := Rule([@(1, VRCLR), @(2, VGath_sv)],
#                e -> let(g := @(2).val, v:= g.v, func := g.func, n := func.domain(), N:= func.range(),
#                        VGath_sv(fCompose(fTensor(L(2*N/v,2), fId(v)), fTensor(fId(2), func), fTensor(L(2*n/v,n/v), fId(v))), v, g.sv))),
#
#    VRCLR_VScat_split := Rule([@(1, VRCLR), @(2, VScat_sv)],
#                e -> let(g := @(2).val, v:= g.v, func := g.func, n := func.domain(), N:= func.range(),
#                        VScat_sv(fCompose(fTensor(L(2*N/v,2), fId(v)), fTensor(fId(2), func), fTensor(L(2*n/v,n/v), fId(v))), v, g.sv)))
#
    VGath_FormatPrm := ARule(Compose, [[@(1, VGath), [fTensor, fBase, @(2, fId)]],
            [@(3, FormatPrm), [fTensor, fId, @(4, L, e->IsInt(@(2).val.domain()/e.domain()))]]],
        e->[FormatPrm(fTensor(fId(Rows(@(1).val)/@(4).val.domain()), @(4).val)), @(1).val]),

    FormatPrm_VScat := ARule(Compose, [[@(1, FormatPrm), [fTensor, fId, @(2, L)]],
            [@(3, [VScat, VScatAcc]), [fTensor, fBase, @(4, fId, e->IsInt(e.domain()/@(2).val.domain()))]]],
        e->[@(3).val, FormatPrm(fTensor(fId(Cols(@(3).val)/@(2).val.domain()), @(2).val))]),

    VRCL_VGath_fIdxL := ARule(Compose, [[@(2,VRCL), @(3,VGath)], [@(1,FormatPrm), [@(4,fTensor), fId, @(5, L, e->e.params[1]=2*e.params[2])]]],
        e -> let(v:=@(2).val.v, r:=Rows(@(3).val), g:=@(3).val, [VGath(fTensor(g.func, fId(2)), v)])),

    VRCR_VScat_fIdxL := ARule(Compose, [[@(1, FormatPrm), [@(4,fTensor), fId, @(5, L, e->e.params[2]=2)]], [@(2,VRCR), @(3,[VScat, VScatAcc])]],
        e -> let(v:=@(2).val.v, c:=Cols(@(3).val), s:=@(3).val, [ObjId(@(3).val)(fTensor(s.func, fId(2)), v)]))

));

Class(RulesVBlkInt, RuleSet);
RewriteRules(RulesVBlkInt, rec(
    Merge_RulesVBlkInt_VRC_rt := ARule(Compose, [@(1, VBlkInt), @(2, VRC)],
                e -> let(v1 := @(1).val, v2 := @(2).val, v := v1.v, [ VIxL(Rows(v1)/(2*v), v, v), v1.child(1), VRCL(v2.child(1), v) ])),

    Merge_RulesVBlkInt_VRC_lft := ARule(Compose, [@(2, VRC), @(1, VBlkInt)],
                e -> let(v1 := @(1).val, v2 := @(2).val, v := v1.v, [ VRCR(v2.child(1), v), v1.child(1), VIxL(Rows(v1)/(2*v), 2, v) ])),

    Merge_RulesVBlkInt_vRC_rt := ARule(Compose, [@(1, VBlkInt), @(2, vRC)],
                e -> let(v1 := @(1).val, v := v1.v, [ VIxL(Rows(v1)/(2*v), v, v), v1.child(1), VIxL(Cols(v1)/(2*v), 2, v), @(2).val ])),

    Merge_RulesVBlkInt_vRC_lft := ARule(Compose, [@(2, vRC), @(1, VBlkInt)],
                e -> let(v1 := @(1).val, v := v1.v, [ @(2).val, VIxL(Rows(v1)/(2*v), v, v), v1.child(1), VIxL(Rows(v1)/(2*v), 2, v) ])),
));
