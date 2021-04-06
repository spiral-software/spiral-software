
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(NEONUnparser, CMacroUnparserProg, rec(

    T_Real := (self, t, vars, i, is) >> Cond(
	t.params[1] = 32, Print("float32_t ", self.infix(vars, ", ", i+is)),
	Inherited(t, vars, i, is)),

    # composite instructions, normally they should go away before unparser
    vrev_neon  := (self, o, i, is) >> self.printf("vrev64q_f32($1)", o.args),
    vrev_half  := (self, o, i, is) >> self.printf("vrev64_f32($1)", o.args),

#    vunpacklo2_neon    := (self, o, i, is) >> self.printf("vzipq_f32(vtrnq_f32($1, $2).val[0], vtrnq_f32($1, $2).val[1]).val[0]", [o.args[1], o.args[2]]),
#    vunpackhi2_neon    := (self, o, i, is) >> self.printf("vzipq_f32(vtrnq_f32($1, $2).val[0], vtrnq_f32($1, $2).val[1]).val[1]", [o.args[1], o.args[2]]),

    vunpacklo_half    := (self, o, i, is) >> self.printf("vtrn_f32($1,$2).val[0]", o.args),
    vunpackhi_half    := (self, o, i, is) >> self.printf("vtrn_f32($1,$2).val[1]", o.args),

    vunpacklolo2_neon := (self, o, i, is) >> self.printf(
	"vcombine_f32(vget_low_f32($1),vget_low_f32($2))", o.args),
    vunpacklohi2_neon := (self, o, i, is) >> self.printf(
	"vcombine_f32(vget_low_f32($1),vget_high_f32($2))", o.args),
    vunpackhilo2_neon := (self, o, i, is) >> self.printf(
	"vcombine_f32(vget_high_f32($1),vget_low_f32($2))", o.args),
    vunpackhihi2_neon := (self, o, i, is) >> self.printf(
	"vcombine_f32(vget_high_f32($1),vget_high_f32($2))", o.args),

    # native instructions
    vuzpq_32f  := (self, o, i, is) >> self.printf("vuzpq_f32($1, $2)", o.args),
    vzipq_32f  := (self, o, i, is) >> self.printf("vzipq_f32($1, $2)", o.args),
    vtrnq_32f  := (self, o, i, is) >> self.printf("vtrnq_f32($1, $2)", o.args),
    vtrnq_half := (self, o, i, is) >> self.printf("vtrn_f32($1, $2)", o.args),
    
    vextract_neon_4x32f  := (self, o, i, is) >> self.printf(
	"$1.val[$2]", [o.args[1], o.args[2].p[1]]),
    vextract_half        := (self, o, i, is) >> self.printf(
	"$1.val[$2]", [o.args[1], o.args[2].p[1]]),

    vext_half  := (self, o, i, is) >> self.prefix("vext_f32", o.args),
    vext_neon  := (self, o, i, is) >> self.prefix("vextq_f32", o.args),



    fdiv := (self, o, i, is) >> self.printf("(((float)$1) / $2)", o.args),
    div  := (self, o, i, is) >> self.printf("($1 / $2)", o.args),
    idiv := (self, o, i, is) >> self.printf("($1 / $2)", o.args),
    imod := (self, o, i, is) >> self.printf("($1 % $2)", o.args),

    Value := (self, o, i, is) >> let(zero := "0", Cond(
	    (o.t = TVect(T_Real(32), 4) or o.t = TVect(TReal, 4))  and Length(Set(o.v)) = 1,
	        Print("vdupq_n_f32(", self(o.v[1], i, is), ")"),
	    (o.t = TVect(T_Real(32), 2) or o.t = TVect(TReal, 2))  and Length(Set(o.v)) = 1,
	        Print("vdup_n_f32(", self(o.v[1], i, is), ")"),
	    o.t = T_Real(32) or o.t = TReal,
	        When(o.v < 0, Print("(", o.v, ")"), Print(o.v)),
	    IsIntT(o.t) or IsUIntT(o.t),
	        When(o.v < 0, Print("(", o.v, ")"), Print(o.v)),
	    o.t = TVect(T_Real(32), 4) or o.t = TVect(TReal, 4),
	        Print("(float32x4_t){",o.v[1].v,", ",o.v[2].v,", ",o.v[3].v,", ",o.v[4].v,"}"),
	    o.t = TVect(T_Real(32), 2) or o.t = TVect(TReal, 2),
	        Print("(float32x2_t){",o.v[1].v,", ",o.v[2].v,"}"),
	    o.t = TString,
	        Print(o.v),
	    Inherited(o, i, is))),

    nth := (self, o, i, is) >> self.printf("$1[$2]", [o.loc, o.idx]),

    TReal := (self, t, vars, i, is) >> Print("float ", self.infix(vars, ", ")),

    TVect := (self, t, vars, i, is) >> Print(
        Cond( t = NEON.t,           "float32x4_t ",
              t = NEON_HALF.t,      "float32x2_t ",
              t = TVect(TReal, 4), "float32x4_t ",
              t = TVect(T_Real(32), 2), "float32x2_t ",
              t = TVect(TReal, 2), "float32x2_t ",
              t = TVect(NEON.t, 2), "float32x4x2_t ",
              t = TVect(NEON_HALF.t, 2), "float32x2x2_t ",
              Error("unexpected vector type")), 
        self.infix(vars, ", ")),

    vcombine_neon   := (self, o, i, is) >> self.prefix("vcombine_f32", o.args),
    vload_half_neon := (self, o, i, is) >> self.printf("vld1_f32((float32_t*)$1)", o.args),

    vload1_neon  := (self, o, i, is) >> self.prefix("vld1q_lane_f32", 
	[o.args[1], o.args[2], o.args[3].p-1]),
    vload1_half  := (self, o, i, is) >> self.prefix("vld1_lane_f32", 
	[o.args[1], o.args[2], o.args[3].p-1]),
    vstore1_neon  := (self, o, i, is) >> Print(Blanks(i), 
	self.prefix("vst1q_lane_f32", [o.args[1], o.args[2], o.args[3]-1]), ";\n"),
    vstore1_half  := (self, o, i, is) >> Print(Blanks(i), 
	self.prefix("vst1_lane_f32", [o.args[1], o.args[2], o.args[3]-1]), ";\n"),

    vstore2lo_neon  := (self, o, i, is) >> Print(Blanks(i), self.printf(
	    "vst1_f32($1,vget_low_f32($2));\n", o.args)),
    vstore2hi_neon  := (self, o, i, is) >> Print(Blanks(i), self.printf(
	    "vst1_f32($1,vget_high_f32($2));\n", o.args)),

    data := (self, o, i, is) >> Cond(
	o.var.t=TVect(T_Real(32), 4) or o.var.t=TVect(TReal, 4), 
	    Print(Blanks(i), self.printf(
              "const float32x4_t $1 = {(float32_t)$2,(float32_t)$3,(float32_t)$4,(float32_t)$5}",
	      [o.var] :: List(o.value.v, x -> x.v)), ";\n", self(o.cmd, i, is)),

	o.var.t=TVect(T_Real(32), 2) or o.var.t=TVect(TReal, 2), 
	    Print(Blanks(i), self.printf(
	      "const float32x2_t $1 = {(float32_t)$2, (float32_t)$3}", 
	      [o.var] :: List(o.value.v, x->x.v)), ";\n", self(o.cmd, i, is)),

	o.var.t = T_Real(32) or o.var.t = TReal,
	    Print(Blanks(i), self.printf(
	      "float32_t $1 = (float32_t)$2", [o.var, o.value.v]), ";\n", self(o.cmd, i, is)),

        Inherited(o, i, is)),

    vpack  := (self, o, i, is) >> Cond(
	o.t=TVect(T_Real(32), 4) or o.t=TVect(TReal, 4), 
	self.printf("(float32x4_t){(float32_t)$1, (float32_t)$2, (float32_t)$3, (float32_t)$4}", o.args),
	o.t=TVect(T_Real(32), 2) or o.t=TVect(TReal, 2), 
	self.printf("(float32x2_t){(float32_t)$1, (float32_t)$2}", o.args),
        
	Inherited(o, i, is)),

    vdup := (self, o, i, is) >> Cond(
	o.t=TVect(T_Real(32), 4) or o.t=TVect(TReal, 4), 
	    self.printf("vdupq_n_f32($1)", o.args),
	o.t=TVect(T_Real(32), 2) or o.t=TVect(TReal, 2), 
	    self.printf("vdup_n_f32($1)", o.args),
	Inherited(o, i, is)),

    vdup_lane_neon := (self, o, i, is) >> Cond(
	o.args[2] = 1 or o.args[2] = 2,
	    self.printf("vdupq_lane_f32(vget_low_f32($1), $2)", [o.args[1], o.args[2]-1]),
	o.args[2] = 3 or o.args[2] = 4,
	    self.printf("vdupq_lane_f32(vget_high_f32($1), $2)", [o.args[1], o.args[2]-3])
    ),

    vdup_lane_half := (self, o, i, is) >> self.prefix("vdup_lane_f32", 
	[o.args[1], o.args[2]-1]),

    vswapcx_neon := (self, o, i, is) >> self.printf("vrev64q_f32($1)", o.args),

    vswapcx_half := (self, o, i, is) >> self.printf("vrev64_f32($1)", o.args),

    mul := (self, o, i, is) >> Cond(ObjId(o.t) = TVect and o.t.size=4,
	CondPat( o,
	    [mul, @TVect, @TVect],
	      self.printf("vmulq_f32($1, $2)", o.args),
	    [mul, @TReal, @TVect],
	      self.printf("vmulq_f32(vdupq_n_f32($1), $2)", o.args),
	    [mul, @TReal, @TReal],
	      self.printf("($1 * $2)", o.args),
	    Inherited(o, i, is)),

	CondPat( o,
	    [mul, @TVect, @TVect],
	      self.printf("vmul_f32($1, $2)", o.args),
	    [mul, @TReal, @TVect],
	      self.printf("vmul_f32(vdup_n_f32($1), $2)", o.args),
	    [mul, @TReal, @TReal],
	      self.printf("($1 * $2)", o.args),
	    Inherited(o, i, is))),

    mul  := (self, o, i, is) >> CondPat( o,
	[mul, @(1).cond(x -> x=TVect(T_Real(32), 4) or x=TVect(TReal,4)), 
	      @(2).cond(x -> x=TVect(T_Real(32), 4) or x=TVect(TReal,4))],
	    self.printf("vmulq_f32($1, $2)", o.args),
	[mul, @(1).cond(x -> x=TVect(T_Real(32), 2) or x=TVect(TReal,2)), 
	      @(2).cond(x -> x=TVect(T_Real(32), 2) or x=TVect(TReal, 2))],
	    self.printf("vmul_f32($1, $2)", o.args),
	[mul, @TReal, @(1).cond(x -> x=TVect(T_Real(32), 4) or x=TVect(TReal,4))],
	    self.printf("vmulq_f32(vdupq_n_f32($1), $2)", o.args),
	[mul, @TReal, @(1).cond(x -> x=TVect(T_Real(32), 2) or x=TVect(TReal,2))],
	    self.printf("vmul_f32(vdup_n_f32($1), $2)", o.args),
	[mul, @TReal, @TReal],
	    self.printf("($1 * $2)", o.args),

        Inherited(o.ooo, i, is)),

    add  := (self, o, i, is) >> Cond(
	not IsVecT(o.t),
	  self.pinfix(o.args, " + "),
	Length(o.args)=2 and o.t.size=4,
	  self.printf("vaddq_f32($1, $2)", o.args),
	Length(o.args)=2 and o.t.size=2, self.printf("vadd_f32($1, $2)", o.args),
	let(rem := ApplyFunc(add, Drop(o.args, 1)),
	    self.printf(Cond(o.t.size=4,"vaddq_f32($1,$2)","vadd_f32($1,$2)"), [o.args[1], rem]))),

    sub  := (self, o, i, is) >> Cond(
	not IsVecT(o.t),                 self.pinfix(o.args, " - "),
	Length(o.args)=2 and o.t.size=4, self.printf("vsubq_f32($1, $2)", o.args),
	Length(o.args)=2 and o.t.size=2, self.printf("vsub_f32($1, $2)", o.args),
	let(rem := ApplyFunc(add, Drop(o.args, 1)),
	    self.printf(Cond(o.t.size=4,"vsubq_f32($1,$2)","vsub_f32($1,$2)"), 
		[o.args[1], rem]))),

		neg  := (self, o, i, is) >> Cond(
		    IsVecT(o.t),
		        self.printf(Cond(o.t.size=4,"vnegq_f32($1)","vneg_f32($1)"), o.args),
		    o.t=TReal,
		        self.printf("(-$1)", o.args),
		    Inherited(o, i, is)),

		fma  := (self, o, i, is) >> CondPat(o,
		    [fma, @TVect, @TVect, @TVect],
		       self.printf(Cond(o.t.size=4,"vmlaq_f32($1, $2, $3)","vmla_f32($1, $2, $3)"), o.args),
		    [fma, @TVect, @TReal, @TVect],
		       self.printf(Cond(o.t.size=4,"vmlaq_f32($1, vdupq_n_f32($2), $3)","vmla_f32($1, vdup_n_f32($2), $3)"), o.args),
		    [fma, @TReal, @TReal, @TReal],
		       self.printf("($1 + ($2 * $3))", o.args),
		    Inherited(o, i, is)),

		fms  := (self, o, i, is) >> CondPat(o,
		    [fms, @TVect, @TVect, @TVect],
		       self.printf(Cond(o.t.size=4,"vmlsq_f32($1, $2, $3)","vmls_f32($1, $2, $3)"), o.args),
		    [fms, @TVect, @TReal, @TVect],
		       self.printf(Cond(o.t.size=4,"vmlsq_f32($1, vdupq_n_f32($2), $3)","vmls_f32($1, vdup_n_f32($2), $3)"), o.args),
		    [fms, @TReal, @TReal, @TReal],
		       self.printf("($1 - ($2 * $3)", o.args),
		    Inherited(o, i, is)),

		nfma  := (self, o, i, is) >> CondPat(o,
		    [nfma, @TVect, @TVect, @TVect],
		       self.printf(Cond(o.t.size=4,"vmlaq_f32(vnegq_f32($1), $2, $3)","vmla_f32(vneg_f32($1), $2, $3)"), o.args),
		    [nfma, @TVect, @TReal, @TVect],
		       self.printf(Cond(o.t.size=4,"vmlaq_f32(vnegq_f32($1), vdupq_n_f32($2), $3)","vmla_f32(vneg_f32($1), vdup_n_f32($2), $3)"), o.args),
		    [nfma, @TReal, @TReal, @TReal],
		       self.printf("(($2 * $3) - $1)", o.args),
		    Inherited(o, i, is)),

		_const := (self,o) >> Cond(
		    o.t = T_Real(32) or o.t = TReal, 
		        ["float32_t", [o.v]],
		    IsVecT(o.t), Cond(
			o.t.t = T_Real(32) and o.t.size=4,
			    ["float32x4_t", List(o.v, x->x.v)],
			o.t.t = T_Real(32) and o.t.size=2,
			    ["float32x2_t", List(o.v, x->x.v)],
			Inherited(o)
		    ),
#				o.t = TString,                     ["", [o.v]],
		    Inherited(o)
		)
));
