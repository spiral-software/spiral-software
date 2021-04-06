
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_VConstructR := [VBase, VRCDiag, BlockVPerm, VPerm, VTensor, VDiag_x_I, VDiag, VGath, VScat, VScatAcc, VPrm_x_I, VGath_sv, VBlk, VContainer, RCVDiag, VO1dsJ, VScale, VGath_zero, VReplicate, VIxJ2, VJxI];
_VConstructL := [VBase, VRCDiag, BlockVPerm, VPerm, VTensor, VDiag_x_I, VDiag, VGath, VScat, VScatAcc, VPrm_x_I, VScat_sv, VScat_svAcc, VScat_pc, VBlk, VContainer, RCVDiag, VReplicate, VO1dsJ, VScale, VGath_zero, VIxJ2, VJxI];

_VTags := [AVecReg, AVecRegCx];

_v_divides_cols := x -> _divides(getV(x), Cols(x));
_v_divides_rows := x -> _divides(getV(x), Rows(x));
_v_tagged_ntR := x->IsNonTerminal(x) and x.hasAnyTag(_VTags) and _divides(x.getAnyTag(_VTags).v, Rows(x));
_v_tagged_ntL := x->IsNonTerminal(x) and x.hasAnyTag(_VTags) and _divides(x.getAnyTag(_VTags).v, Cols(x));

Scat.svVariant := VScat_sv;
ScatAcc.svVariant := VScat_svAcc;
Gath.svVariant := VGath_sv;

Class(RulesVec, RuleSet);
RewriteRules(RulesVec, rec(
    ############################################################
    # Below 2 rules are required by SAR and autolib
    # NOTE: They will cause problems when mixing scalar and vector code,
    # because the entire code (including the scalar portion) will reside inside
    # VContainer, and gath/scat in scalar code will be converted to vgath/vscat
    #
    VContainer__Gath_Scat := Rule(@@(1, [Gath, Scat, ScatAcc], (e,cx)->
        cx.isInside(VContainer) and IsSIMD_ISA(Last(cx.VContainer).isa)
        and let(sg := e, v := Last(cx.VContainer).isa.getV(), vsg := sg.svVariant(sg.func, v, 1), sg.dims() = vsg.dims())
        and (not IsBound(cx.VTensor) or cx.VTensor = [])
        and (not IsBound(cx.VJamL) or cx.VJamL = [])
        and (not IsBound(cx.VJamR) or cx.VJamR = [])
        and (not IsBound(cx.VJam1) or cx.VJam1 = [])
        and (not IsBound(cx.VJam ) or cx.VJam  = [])
        # FF, NOTE: this is a terrible hack to solve a contradiction: How to prevent the rader perm 
	#            from turning into VGath. Need to find a better solution.
        and (not ObjId(e.func) in [RM, RR])),
     (e,cx) -> let(
	 #inside_vrc := (IsBound(cx.vRC) and cx.vRC <> []) or 
	 #              (IsBound(cx.opts.assumeOuter_vRC) and cx.opts.assumeOuter_vRC),
	 v := Last(cx.VContainer).isa.getV(), # / Cond(inside_vrc, 2, 1),
	 # YSV: isa.getV() now takes care of 'inside_vrc', which is thus commented out, 
	 #      and can be removed. 
         e.svVariant(e.func, v, 1))),

    VContainer__Diag := Rule(@@(1, [Diag], (e,cx) ->
        cx.isInside(VContainer) and IsSIMD_ISA(Last(cx.VContainer).isa)
        and not cx.isInside(VTensor)
        and not cx.isInside("VJamL")
        and not cx.isInside("VJamR")
        and not cx.isInside("VJam1")
        and not cx.isInside("VJam") 
	and let(
	    #inside_vrc := cx.isInside(vRC) or (IsBound(cx.opts.assumeOuter_vRC) and cx.opts.assumeOuter_vRC),
	    v := Last(cx.VContainer).isa.getV(), # / Cond(inside_vrc, 2, 1),
	    Rows(e) mod v = 0)),
     (e,cx) -> let(
	 #inside_vrc := (IsBound(cx.vRC) and cx.vRC <> []) or 
	 #              (IsBound(cx.opts.assumeOuter_vRC) and cx.opts.assumeOuter_vRC),
	 v := Last(cx.VContainer).isa.getV(), # / Cond(inside_vrc, 2, 1),
	 # YSV: isa.getV() now takes care of 'inside_vrc', which is thus commented out, 
	 #      and can be removed. 
         VDiag(e.element, v))),

# YSV: if VContainer is used, then I need the following (in particular the VDiag)
    VContainer_XXX := ARule(Compose, [@(1, VContainer), @(2, [VDiag, Diag, Gath, Scat, ScatAcc])],
        e -> [ CopyFields(@(1).val, rec(_children := [@(1).val._children[1] * @(2).val])) ]),

    XXX_VContainer := ARule(Compose, [@(1, [VDiag, Diag, Gath, Scat, ScatAcc]), @(2, VContainer)],
        e -> [ CopyFields(@(2).val, rec(_children := [@(1).val * @(2).val._children[1]])) ]),

    VContainer_VContainer := Rule( [@(1, VContainer), @(2, VContainer, x -> x.isa = @(1).val.isa and x.isa.isCplx()=@(1).val.isa.isCplx())],
        e -> e.child(1)),

#    Gath_Scat_ScatGath_XXX := ARule(Compose, [ @(1, Scat), @(2, ScatGath) ],
##    Gath_Scat_ScatGath_XXX := ARule(Compose, [@@(1, [Scat, Gath], (e,cx)->IsBound(cx.VContainer) and cx.VContainer <> []), @(2, ScatGath)],
#        (e, cx)->let(sg := @@(1).val, v := Last(cx.VContainer).v,
#            Concat(Error(Caught), [ sg.svVariant(sg.func, vrc.v, 1)]) )),

#############################
    # Composition of subvector and full vector accesses
    # The result is VScat_sv that has rem=0 guaranteed, i.e., does not use partially filled vectors
    VScat_sv__VScat := ARule(Compose, [@(1,[VScat_sv, VScat_svAcc]), @(2, [VScat, VScatAcc])],
    e -> let(func := @(1).val.func,  vfunc := @(2).val.func,
             v    := getV(@(1).val), sv    := @(1).val.sv,
	     oid  := Cond(ObjId(@(1).val)=VScat_svAcc or ObjId(@(2).val)=VScatAcc, VScat_svAcc, VScat_sv),
        [ oid(fCompose(func, fTensor(vfunc, fId(v/sv))), v, sv, 0) ])),

    # The result is VGath_sv that has rem=0 guaranteed, i.e., does not use partially filled vectors
    VGath__VGath_sv := ARule(Compose, [@(1,VGath), @(2,VGath_sv)],
    e -> let(func := @(2).val.func,  vfunc := @(1).val.func,
             v    := getV(@(2).val), sv    := @(2).val.sv,
         [ VGath_sv(fCompose(func, fTensor(vfunc, fId(v/sv))), v, sv, 0) ])),

    # Composition of subvector and scalar accesses
    Scat__VScat_sv := ARule(Compose, [@(1, Scat), @(2, VScat_sv)], e -> let(s := @(2).val,
        [ VScat_sv(fCompose(@(1).val.func, fTensor(s.func, fId(s.sv))), s.v, 1, s.rem) ])),

    ScatAcc__VScat_sv := ARule(Compose, [@(1, ScatAcc), @(2, VScat_sv)], e -> let(s := @(2).val,
        [ VScat_svAcc(fCompose(@(1).val.func, fTensor(s.func, fId(s.sv))), s.v, 1, s.rem) ])),

    VGath_sv__Gath := ARule(Compose, [@(1, VGath_sv), @(2, Gath)], e -> let(s := @(1).val,
        [ VGath_sv(fCompose(@(2).val.func, fTensor(s.func, fId(s.sv))), s.v, 1, s.rem) ])),

    # Subvector access --> full vector access
    VScat_sv_to_VScat := Rule(@(1,VScat_sv,e->getV(e)=e.sv), e->VScat(@(1).val.func, @(1).val.v)),
    VGath_sv_to_VGath := Rule(@(1,VGath_sv,e->getV(e)=e.sv), e->VGath(@(1).val.func, @(1).val.v)),
    VScat_svAcc_to_VScatAcc := Rule(@(1,VScat_svAcc,e->getV(e)=e.sv), e->VScatAcc(@(1).val.func, @(1).val.v)),

    # Increase granularity if possible (when function is fId(n) or fTensor(X, fId(n)))
    GathScat_sv_fTensor := Rule([@(1, [VGath_sv, VScat_sv, VScat_svAcc]),
         [@(2,fTensor), ..., [fId, @(3).cond(e -> Gcd(@(1).val.v/@(1).val.sv, EvalScalar(e)) > 1 )]]],
     e -> let(v := @(1).val.v, sv := @(1).val.sv, 
          n := EvalScalar(@(3).val),   gcd := Gcd(v / sv, n),
	  rem := Cond(@(1).val.rem _is Unk, Unk(TInt), @(1).val.rem/gcd),
          ObjId(@(1).val)(fTensor(DropLast(@(2).val.children(), 1), fId(n/gcd)), v, sv*gcd, rem))),

    GathScat_sv_HofTensor := Rule([@(1, [VGath_sv, VScat_sv, VScat_svAcc]),
                        [fCompose, @(9, H, e->_divides(@(1).val.sv, e.params[3]) and e.params[4]=1), 
			    [@(2,fTensor), ..., [fId, @(3).cond(e -> Gcd(@(1).val.v/@(1).val.sv, EvalScalar(e)) > 1)]]]],
     e -> let(v := @(1).val.v, sv := @(1).val.sv, 
          n := EvalScalar(@(3).val),   gcd := Gcd(v / sv, n), hp := @(9).val.params,
	  rem := Cond(@(1).val.rem _is Unk, Unk(TInt), @(1).val.rem/gcd),
          ObjId(@(1).val)(fCompose(H(hp[1]/gcd, hp[2]/gcd, hp[3]/gcd, 1),  fTensor(DropLast(@(2).val.children(), 1), fId(n/gcd))), v, sv*gcd, rem))),

    GathScat_sv_fId := Rule([@(1, [VGath_sv, VScat_sv, VScat_svAcc]), 
	                         [fId, @(3).cond(e -> Gcd(@(1).val.v/@(1).val.sv, EvalScalar(e)) = @(1).val.v)]],
     e -> let(v := @(1).val.v, sv := @(1).val.sv, 
          n := EvalScalar(@(3).val),   gcd := Gcd(v / sv, n),
	  rem := Cond(@(1).val.rem _is Unk, Unk(TInt), @(1).val.rem/gcd),
          ObjId(@(1).val)(fId(n/gcd), v, sv*gcd))),

    # Convert objects that are adjacent to vector constructs to vector objects
    #

    VConstruct_Gath := ARule(Compose, [ @(1, _VConstructL), @(2, Gath, e -> _v_divides_cols(@(1).val)) ],
        e -> [ @(1).val, VGath_sv(@(2).val.func, getV(@(1).val), 1) ]),
    VConstruct_Scat := ARule(Compose, [ @(1, _VConstructL), @(2, Scat, e -> _v_divides_cols(@(1).val)) ],
        e -> [ @(1).val, VScat_sv(@(2).val.func, getV(@(1).val), 1) ]),
    VConstruct_ScatAcc := ARule(Compose, [ @(1, _VConstructL), @(2, ScatAcc, e -> _v_divides_cols(@(1).val)) ],
        e -> [ @(1).val, VScat_svAcc(@(2).val.func, getV(@(1).val), 1) ]),
    VConstruct_Diag := ARule(Compose, [ @(1, _VConstructL), [Diag, @(2).cond(e -> _v_divides_cols(@(1).val))] ],
        e -> [ @(1).val, VDiag(@(2).val, getV(@(1).val)) ]),

    Scat_VConstruct := ARule(Compose, [ @(1, Scat), @(2, _VConstructR, _v_divides_rows) ],
        e -> [ VScat_sv(@(1).val.func, getV(@(2).val), 1), @(2).val ]),
    ScatAcc_VConstruct := ARule(Compose, [ @(1, ScatAcc), @(2, _VConstructR, _v_divides_rows) ],
        e -> [ VScat_svAcc(@(1).val.func, getV(@(2).val), 1), @(2).val ]),
    Gath_VConstruct := ARule(Compose, [ @(1, Gath), @(2, _VConstructR, _v_divides_rows) ],
        e -> [ VGath_sv(@(1).val.func, getV(@(2).val), 1), @(2).val ]),
    Diag_VConstruct := ARule(Compose, [ [Diag, @(1)], @(2, _VConstructR, _v_divides_rows) ],
        e -> [ VDiag(@(1).val, getV(@(2).val)), @(2).val ]),

    VGath_Prm := ARule(Compose, [ @(1, VGath), @(2, Prm) ],
        e -> [ @(1).val, VGath_sv(@(2).val.func, getV(@(1).val), 1) ]),
    Prm_VScat := ARule(Compose, [ @(1, Prm), @(2, VScat) ],
        e -> [ VScat_sv(@(1).val.func.transpose(), getV(@(2).val), 1), @(2).val ]),
    Prm_VScatAcc := ARule(Compose, [ @(1, Prm), @(2, VScatAcc) ],
        e -> [ VScat_svAcc(@(1).val.func.transpose(), getV(@(2).val), 1), @(2).val ]),

    # NOTE: RowVec must be of constant size, fix this limitation
    VRowVec := Rule([@(1, VTensor), @(2, RowVec, e -> not IsSymbolic(e.element.domain()))], 
        e -> VTensor(@(2).val.toDiagBlk(), @(1).val.vlen)),

    VColVec := Rule([@(1, VTensor), @(2, ColVec, e -> not IsSymbolic(e.element.domain()))], 
        e -> VTensor(@(2).val.toDiagBlk(), @(1).val.vlen)),
 
    VPrm_x_I := Rule([@(1, VTensor), @(2, Prm)], e->VPrm_x_I(@(2).val.func, @(1).val.vlen)),
    VDiag_x_I := Rule([@(1, VTensor), @(2, Diag)], e->VDiag_x_I(@(2).val.element, @(1).val.vlen)),
    RCDiag_x_I := Rule([@(1, VTensor), @(2, RCDiag)], e->VRCDiag(VDup(@(2).val.element, @(1).val.vlen), @(1).val.vlen)),

    VPrm_x_I_Id := ARule(Compose,[@(1,VPrm_x_I), [Prm, fId]],e->[@(1).val]),
    Id_VPrm_x_I := ARule(Compose, [[Prm, fId], @(1,VPrm_x_I)], e->[@(1).val]),

    # Remove identity gathers / scatters
    RemGSP := Rule([@(1, [Gath, Scat, Prm, VGath, VScat]), @(2, fId)], e->I(@(1).val.dims()[1])),
    RemIL  := ARule(Compose, [ @(1), @(2, I) ], e -> [ @(1).val ]),
    RemIR  := ARule(Compose, [ @(1, I), @(2) ], e -> [ @(2).val ]),

    #   H rules
    VGath_sv_H := Rule( [@(1,VGath_sv), [H,
            @(2).cond(e->not IsSymbolic(e)),
            @(3).cond(e->_divides(getV(@(1).val), e)),
            @(4).cond(e->_divides(getV(@(1).val), e)), _1]], 
	e -> let(
            d := e.v / e.sv,
	    rmod := EvalScalar(@(2).val) mod getV(@(1).val),
	    Cond(rmod=0,
		 VGath(H(@(2).val/d, @(3).val/d, @(4).val/d, 1), e.v),
		 VGath_pc(@(2).val*e.sv, @(3).val*e.sv, @(4).val*e.sv, e.v))
	)),

    VScat_sv_H := Rule( [@(1, [VScat_sv, VScat_svAcc]), [H,
            @(2).cond(e->not IsSymbolic(e)),
            @(3).cond(e->_divides(getV(@(1).val), e)),
            @(4).cond(e->_divides(getV(@(1).val), e)), _1]], 
	e -> let(
            d := e.v / e.sv,
	    oid   := Cond(ObjId(@(1).val)=VScat_sv, VScat, VScatAcc),
	    oidpc := Cond(ObjId(@(1).val)=VScat_sv, VScat_pc, VScat_pcAcc),
	    rmod := EvalScalar(@(2).val) mod getV(@(1).val),
	    Cond(rmod=0,
		 oid(H(@(2).val/d, @(3).val/d, @(4).val/d, 1), e.v), 
		 oidpc(@(2).val*e.sv, @(3).val*e.sv, @(4).val*e.sv, e.v))
	)),

   Scat_VScat_pc_to_VScat_sv := ARule(Compose, [@(1, Scat), @(2, VScat_pc, x -> x.N=x.n and x.ofs = 0)],
       e -> [VScat_sv(@(1).val.func, @(2).val.v, 1)]),

   VGath_pc_Gath_H := ARule(Compose, [@(1, VGath_pc), [Gath, [@(2, H), @, @, @, _1]]],
       e -> [VGath_pc(@(2).val.params[1], @(1).val.n, @(2).val.params[3], @(1).val.v)] ),

   Scat_H_VScat_pc := ARule(Compose, [[Scat, [@(1, H), @, @, @, _1]], @(2, [VScat_pc, VScat_pcAcc])],
       e -> [ObjId(@(2).val)(@(1).val.params[1], @(2).val.n, @(1).val.params[3], @(2).val.v)] ),
        
   ScatGath := Rule([@(1, VTensor), @(2, ScatGath)], 
       e -> ScatGath(fTensor(@(2).val.sfunc, fId(@(1).val.vlen)), fTensor(@(2).val.gfunc, fId(@(1).val.vlen)))),

   Scat_Cvt := ARule(Compose, [@(1, Scat), @(2, Cvt, x -> IsSIMD_ISA(x.params[1].isa_to) and _divides(x.params[1].isa_to.v, Cols(@(1).val)) )],
      e -> let(isa := @(2).val.params[1].isa_to, [ VContainer(VScat_sv(@(1).val.func, isa.v, 1), isa), @(2).val ])),
   Cvt_Gath := ARule(Compose, [@(1, Cvt, x -> IsSIMD_ISA(x.params[1].isa_from)), @(2, Gath, x -> _divides(@(1).val.params[1].isa_from.v, Rows(x)))],
      e -> let(isa := @(1).val.params[1].isa_from, [ @(1).val, VContainer(VGath_sv(@(2).val.func, isa.v, 1), isa) ])),
));
