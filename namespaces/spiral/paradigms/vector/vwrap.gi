
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(VWrap, VWrapBase, rec(
    __call__ := (self,isa) >> Checked(IsSIMD_ISA(isa),
    WithBases(self, rec(operations:=PrintOps, isa:=isa))),

    wrap := (self,r,t,opts) >> let(
        isa := self.isa, 
        v := isa.v,
        tt := When(t.isReal(),
            #paradigms.vector.breakdown.AxI_vec(TTensorI(t, v, AVec, AVec).withTags([AVecReg(isa)]), r),
            @_Base(paradigms.vector.sigmaspl.VTensor(r.node, v), r),
            paradigms.vector.breakdown.AxI_vec(
                TTensorI(TRC(t).withTags([AVecReg(isa)]), v, AVec, AVec).withTags([AVecReg(isa)]),
                paradigms.vector.breakdown.TRC_VRCLR(TRC(t).withTags([AVecReg(isa)]), r)
            )
        ),

#        Chain(Print("---> VWrap.wrap: ", r, "\n* ", t, "\n* ", tt, "\n"), DbgBreak("a"), tt)
        tt
    ),

    twrap := (self, t,opts) >> let(
        isa := self.isa, 
        v := isa.v,
        tt := When(t.isReal(),
            paradigms.vector.sigmaspl.VTensor(t, v),
        #TTensorI(t, v, AVec, AVec).withTags([AVecReg(isa)]),
            TTensorI(TRC(t).withTags([AVecReg(isa)]), v, AVec, AVec).withTags([AVecReg(isa)])
        ),
        
#        Chain(Print("---> VWrap.twrap: ", t, "\n* ", tt, "\n"), DbgBreak("a"), tt)
        tt
    ),

    print := self >> Print(self.name, "(", self.isa, ")"),

   
));

Class( VWrapCpx_Mixin, rec(
    # as VWrap may change data type it must update data types
    complexToRealT := (self, t, wt) >> Cond(t.hasA("t_in") and t.hasA("t_out"),
	wt.setA(t_in => List(t.getA("t_in"), t -> t.realType()),
		t_out => List(t.getA("t_out"), t -> t.realType())),
	wt),

    opts := (self, t, opts) >> CopyFields(opts, rec(
        XType := StripList(List(Flat([opts.XType]), t -> t.realType())),
        YType := StripList(List(Flat([opts.YType]), t -> t.realType())),
    )),
    
));

Class(VWrapTRC, VWrapCpx_Mixin, VWrapBase, rec(
    __call__ := (self, isa) >> Checked(IsSIMD_ISA(isa),
    WithBases(self, rec(isa:=isa, operations:=PrintOps))),
    print := self >> Print(self.name, "(", self.isa, ")"),
    wrap := (self, r, t, opts) >> let(
        tt := paradigms.vector.breakdown.TRC_vect(self.complexToRealT(t, TRC(t).withTags([AVecReg(self.isa)])), r),
#        Chain(Print("---> VWrapTRC.wrap: ", r, "\n* ", t, "\n* ", tt, "\n"), tt)
        tt
    ),
    twrap := (self, t, opts) >> let(
        tt := self.complexToRealT(t, TRC(t).withTags([AVecReg(self.isa)])),
#        Chain(Print("---> VWrapTRC.twrap: ", t, "\n* ", tt, "\n"), tt)
        tt
    ),
));

Class(VWrapTRCcplx, VWrapCpx_Mixin, VWrapBase, rec(
    __call__ := (self, isa) >> Checked(IsSIMD_ISA(isa),
    WithBases(self, rec(isa:=isa, operations:=PrintOps))),
    print := self >> Print(self.name, "(", self.isa, ")"),
    wrap := (self, r, t, opts) >> paradigms.vector.breakdown.TRC_cplx(self.complexToRealT(t, TRC(t).withTags([AVecRegCx(self.isa)])), r),
    twrap := (self, t, opts) >> self.complexToRealT(t, TGrp(TRC(t).withTags([AVecRegCx(self.isa)]))),

#    wrap := (self, r, t) >> @_Base(self.twrap(t), r),
#    twrap := (self, t) >> paradigms.vector.sigmaspl.vRC(t.withTags([AVecRegCx(self.isa)]) )
));
