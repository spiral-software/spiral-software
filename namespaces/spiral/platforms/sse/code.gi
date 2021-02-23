
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(SSE_LDST, rec(
    list := [],
    __call__ := meth(arg)
        local self;
	self := arg[1];
	Append(self.list, Drop(arg, 1));
    end
));


#######################################################################################
# SSE2 2-way 64-bit floating-point instructions
#######################################################################################

vzero_2x64f_const := TVect(TReal, 2).zero();
vzero_2x64f := () -> vzero_2x64f_const;

# Load
#
SSE_LDST(
    Class(vload1sd_2x64f, VecExp_2.unary()),
    Class(vload_1l_2x64f, VecExp_2.binary()),
    Class(vload_1h_2x64f, VecExp_2.binary()),
    Class(vloadu_2x64f,   VecExp_2.unary())
);

# Store
#
SSE_LDST(
    Class(vstore_1l_2x64f, VecStoreCommand.binary()),
    Class(vstore_1h_2x64f, VecStoreCommand.binary()),
    Class(vstoreu_2x64f,   VecStoreCommand.binary())
);

# Binary shuffle
#
Class(vunpacklo_2x64f, VecExp_2.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpacklo(in1, in2, 2, 1)));

Class(vunpackhi_2x64f, VecExp_2.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpackhi(in1, in2, 2, 1)));

Class(vshuffle_2x64f, VecExp_2.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, p, 2, 1),
    params := self >> sparams(2,2),
    ev := self >> self.t.value(self.semantic(self._vval(1),self. _vval(2), self.args[3].p)),
));

# Unary shuffle, sign change
#
Class(vushuffle_2x64f, VecExp_2.unaryFromBinop(vshuffle_2x64f));

Class(chslo_2x64f,     VecExp_2.unary(), rec(
    ev := self >> let(v := self.args[1], v * self.t.value([-1.0, 1.0])),
));

Class(chshi_2x64f,     VecExp_2.unary(), rec(
    ev := self >> let(v := self.args[1], v * self.t.value([1.0, -1.0])),
));

# SSE3 new addsub and hadd instructions
#
Class(addsub_2x64f, VecExp_2.binary(), rec(
    ev := self >> add(self.args[1], chslo_2x64f(self.args[2]))
));
Class(hadd_2x64f,   VecExp_2.binary());


#######################################################################################
# SSE2 2-way 64-bit integer instructions
# NOTE: use TFixedPt instead of TReal in as the type in below instructions
#######################################################################################

vzero_2x64i_const := TVect(TReal, 2).zero();
vzero_2x64i := () -> vzero_2x64i_const;

# Binary shuffle
#
Class(vunpacklo_2x64i, VecExp_2.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpacklo(in1, in2, 2, 1)
));

Class(vunpackhi_2x64i, VecExp_2.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpackhi(in1, in2, 2, 1)
));

Class(vshuffle_2x64i, VecExp_2.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, p, 2, 1),
    params := self >> sparams(2,2)
));

# Unary shuffle, sign change
#
Class(vushuffle_2x64i, VecExp_2.unaryFromBinop(vshuffle_2x64i));
Class(chslo_2x64i,     VecExp_2.unary());
Class(chshi_2x64i,     VecExp_2.unary());

# cmpge_2x64f
Class(cmpge_2x64f,  VecExp_2.binary());
Class(cmple_2x64f,  VecExp_2.binary());
Class(cmpeq_2x64f,  VecExp_2.binary());


#######################################################################################
# SSE2 2-way 32-bit floating-point instructions
# this architecture simulates 2-way inside 4-way
#######################################################################################

vzero_2x32f_const := TVect(TReal, 2).zero();
vzero_2x32f := () -> vzero_2x32f_const;

# Load
#
SSE_LDST(
    Class(vload1sd_2x32f, VecExp_2.unary()),
    Class(vload_1l_2x32f, VecExp_2.binary()),
    Class(vload_1h_2x32f, VecExp_2.binary()),
    Class(vload_2x32f,    VecExp_2.binary()),
    Class(vloadu_2x32f,   VecExp_2.unary())
);

# Store
#
SSE_LDST(
    Class(vstore_1l_2x32f, VecExpCommand),
    Class(vstore_1h_2x32f, VecExpCommand),
    Class(vstore_2x32f,    VecExpCommand),
    Class(vstoreu_2x32f,   VecExpCommand)
);

# Binary shuffle
#
Class(vunpacklo_2x32f, VecExp_2.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpacklo(in1, in2, 2, 1)
));

Class(vunpackhi_2x32f, VecExp_2.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpackhi(in1, in2, 2, 1)
));

Class(vshuffle_2x32f, VecExp_2.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, p, 2, 1),
    params := self >> sparams(2,2)
));

# Unary shuffle, sign change
#
Class(vushuffle_2x32f, VecExp_2.unaryFromBinop(vshuffle_2x32f));
Class(chslo_2x32f, VecExp_2.unary());
Class(chshi_2x32f, VecExp_2.unary());

# SSE3 new addsub and hadd instructions
#
Class(addsub_2x32f, VecExp_2.binary());
Class(hadd_2x32f,   VecExp_2.binary());


#######################################################################################
#   SSE 4-way 32-bit float instructions
#######################################################################################

vzero_4x32f_const := TVect(T_Real(32), 4).zero();
vzero_4x32f := () -> vzero_4x32f_const;

# Load
#
SSE_LDST(
    Class(vload1_4x32f,   VecExp_4.unary()),
    Class(vload_2l_4x32f, VecExp_4.binary()),
    Class(vload_2h_4x32f, VecExp_4.binary()),
    Class(vloadu_4x32f,   VecExp_4.unary()),
    Class(vloadu2_4x32f,  VecExp_4.unary()),
    Class(vinsert_4x32f,  VecExp_4.binary())
);

# Store
#
SSE_LDST(
    Class(vstore1_4x32f,   VecStoreCommand.binary()),
    Class(vstore_2l_4x32f, VecStoreCommand.binary()),
    Class(vstore_2h_4x32f, VecStoreCommand.binary()),
    Class(vstoreu_4x32f,   VecStoreCommand.binary()),
    Class(vstoreu2_4x32f,  VecStoreCommand.binary()), # cprint: self.loc.svcprint(16)
    Class(vstoremsk_4x32f, VecStoreCommand.binary()),   # cprint: self.loc.svcprint(8)
    Class(vextract_4x32f,  VecStoreCommand.binary())
);

# Binary shuffle
#
Class(vunpacklo_4x32f, VecExp_4.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpacklo(in1, in2, 4, 1),
));

Class(vunpackhi_4x32f, VecExp_4.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpackhi(in1, in2, 4, 1),
));

Class(vshuffle_4x32f,  VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, p, 4, 1),
    params := self >> sparams(4,4),
    permparams := iperm4,
    ev := self >> self.t.value(self.semantic(self._vval(1),self._vval(2), self.args[3].p)),
));

# Unary shuffle
#
Class(vushuffle_4x32f, VecExp_4.unaryFromBinop(vshuffle_4x32f), rec(
    ev := self >> self.t.value(self.semantic(self._vval(1), self.args[2].p)),
));

Class(chshi_4x32f,  VecExp_4.unary(), rec(
    ev := self >> let( v := self.args[1], v * self.t.value([1.0, -1.0, 1.0, -1.0]) )
));

Class(chslo_4x32f,  VecExp_4.unary(), rec(
    ev := self >> let( v := self.args[1], v * self.t.value([-1.0, 1.0, -1.0, 1.0]) )
));

Class(vldup_4x32f,  VecExp_4.unary(), rec(
    ev := self >> let( v := self._vval(1), self.t.value([v[1], v[1], v[3], v[3]]) )
));
Class(vhdup_4x32f,  VecExp_4.unary(), rec(
    ev := self >> let( v := self._vval(1), self.t.value([v[2], v[2], v[4], v[4]]) )
));

# horizontal add
Class(alignr_4x32f, VecExp_4.binary());


#######################################################################################
#   SSE 4-way 32-bit integer instructions
#######################################################################################

vzero_4x32i_const := TVect(TReal, 4).zero();
vzero_4x32i := () -> vzero_4x32i_const;

Class(vloadu_4x32i, VecExp_4.unary());
Class(vstoreu_4x32i, VecExpCommand);

# Binary shuffle
#
Class(vunpacklo_4x32i, VecExp_4.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpacklo(in1, in2, 4, 1)));

Class(vunpackhi_4x32i, VecExp_4.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpackhi(in1, in2, 4, 1)));

Class(vpacks_4x32i, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> pack_semantic(in1, in2, 4),
    computeType := self >> TVect(T_Int(16), 8)));

Class(vshuffle_4x32i, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, p, 4, 1),
    params := self >> sparams(4,4),
    permparams := iperm4
));

# Unary shuffle
#
Class(vushuffle_4x32i, VecExp_4.unaryFromBinop(vshuffle_4x32i));

# for Viterbi
Class(interleavedmask_4x32i, VecExp_4.binary(), rec(
    computeType := self >> TUChar
));

#_mm_testz_si128
Class(testz_4x32i, VecExp_4.binary(), rec(
    computeType := self >> TInt
));

#_mm_testc_si128
Class(testc_4x32i, VecExp_4.binary(), rec(
    computeType := self >> TInt
));

#_mm_testnzc_si128
Class(testnzc_4x32i, VecExp_4.binary(), rec(
    computeType := self >> TInt
));



# SSE3 new addsub and hadd instructions
#
Class(addsub_4x32f, VecExp_4.binary(), rec(
    ev := self >> add(self.args[1], chslo_4x32f(self.args[2]))
));

Class(hadd_4x32f,   VecExp_4.binary());

# type conversion of lower two elements from 4x32f to 2x64f
Class(vcvt_64f32f, VecExp_4.unary(), rec(
    computeType := (self) >> TVect(T_Real(64), 2)
));

# type conversion of 2x64f to lower two of 4x32f
Class(vcvt_32f64f, VecExp_2.unary(), rec(
    computeType := (self) >> TVect(T_Real(32), 4)
));

# type conversion 4x32i to 4x32f
Class(vcvt_4x32_i2f, VecExp_4.unary(), rec(
    computeType := (self) >> TVect(T_Real(32), 4)
));

# type conversion 4x32f to 4x32i, rounding acording MXCSR register rules
Class(vcvt_4x32_f2i, VecExp_4.unary(), rec(
    computeType := (self) >> TVect(T_Int(32), 4)
));

# type conversion 4x32f to 4x32i, truncation toward zero
Class(vcvtt_4x32_f2i, VecExp_4.unary(), rec(
    computeType := (self) >> TVect(T_Int(32), 4)
));

#######################################################################################
#   SSE2 8-way 16-bit integer instructions
#######################################################################################

vzero_8x16i_const := TVect(TReal, 8).zero();
vzero_8x16i := () -> vzero_8x16i_const;

# SSSE3 sign change instructions
Class(chs_8x16i, VecExp_8.binary());

# Load
#
SSE_LDST(
    Class(vload1_8x16i, VecExp_8.ternary()),
    Class(vload2_8x16i, VecExp_8.unary()),
    Class(vload4_8x16i, VecExp_8.unary()),
    Class(vloadu_8x16i, VecExp_8.unary())
);

# Store
#
SSE_LDST(
    Class(vextract1_8x16i, VecExpCommand, rec(numargs := 2)),
    Class(vextract2_8x16i, VecExpCommand, rec(numargs := 1)),
    Class(vstoreu_8x16i,   VecExpCommand),
    Class(vstore4_8x16i,   VecExpCommand),
    Class(vstoremsk_8x16i, VecExpCommand)
);

# Binary shuffle
#
Class(vunpacklo_8x16i, VecExp_8.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpacklo(in1, in2, 8, 1)));

Class(vunpackhi_8x16i, VecExp_8.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpackhi(in1, in2, 8, 1)));

Class(vunpacklo2_8x16i, VecExp_8.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpacklo(in1, in2, 8, 2)));

Class(vunpackhi2_8x16i, VecExp_8.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpackhi(in1, in2, 8, 2)));

Class(vunpacklo4_8x16i, VecExp_8.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpacklo(in1, in2, 8, 4)));

Class(vunpackhi4_8x16i, VecExp_8.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpackhi(in1, in2, 8, 4)));

Class(vpacks_8x16i, VecExp_8.binary(), rec(
    semantic := (in1, in2, p) -> pack_semantic(in1, in2, 16),
    computeType := self >> TVect(T_Int(8), 16)));

Class(vpackus_8x16i, VecExp_8.binary(), rec(
    semantic := (in1, in2, p) -> pack_semantic(in1, in2, 16),
    computeType := self >> TVect(T_UInt(8), 16)));

Class(vshuffle2_8x16i, VecExp_8.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, p, 8, 2),
    params := self >> sparams(4,4),
    permparams := iperm4
));

Class(vshuffle4_8x16i, VecExp_8.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, p, 8, 4),
    params := self >> sparams(2,2),
));

# Unary shuffle
#
Class(vushuffle2_8x16i, VecExp_8.unaryFromBinop(vshuffle2_8x16i), rec(
    params := self >> sparams(4,4),
    permparams := iperm4,
));

Class(vushufflelo_8x16i, VecExp_8.unary(), rec(
    params := self >> sparams(4,4),
    permparams := iperm4,
    semantic := (in1, p) -> shufflelo(in1, p, 8, 1)));  #Complicated, buggy?

Class(vushufflehi_8x16i, VecExp_8.unary(), rec(
    params := self >> sparams(4,4),
    permparams := iperm4,
    semantic := (in1, p) -> shufflehi(in1, p, 8, 1)));  #Complicated, buggy?

# SSSE3 unary shuffle
Class(vushuffle_8x16i, VecExp_8.binary());

# for Viterbi
Class(interleavedmask_8x16i, VecExp_8.binary(), rec(
        computeType := self >> TSym("short int")
));

Class(alignr_8x16i, VecExp_8.binary());

# cmplt_8x16i
Class(cmplt_8x16i,  VecExp_8.binary());


#_mm_testz_si128
Class(testz_8x16i, VecExp_8.binary(), rec(
    computeType := self >> TInt
));
#_mm_testc_si128
Class(testc_8x16i, VecExp_8.binary(), rec(
    computeType := self >> TInt
));

#_mm_testnzc_si128
Class(testnzc_8x16i, VecExp_8.binary(), rec(
    computeType := self >> TInt
));


#######################################################################################
#   SSE2 16-way 8-bit integer instructions
#######################################################################################

Class(vushuffle_16x8i, VecExp_16.unary(), rec(
    params := self >> sparams(4,4),
));

# Load/Store
#
SSE_LDST(
    Class(vloadu_16x8i,    VecExp_16.unary()),
    Class(vstoreu_16x8i,   VecExpCommand),
    Class(vstoremsk_16x8i, VecExpCommand)
);

# Binary shuffle
#
Class(vunpacklo_16x8i, VecExp_16.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpacklo(in1, in2, 16, 1)));

Class(vunpackhi_16x8i, VecExp_16.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpackhi(in1, in2, 16, 1)));

Class(vunpacklo2_16x8i, VecExp_16.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpacklo(in1, in2, 16, 2)));

Class(vunpackhi2_16x8i, VecExp_16.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpackhi(in1, in2, 16, 2)));

Class(vunpacklo4_16x8i, VecExp_16.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpacklo(in1, in2, 16, 4)));

Class(vunpackhi4_16x8i, VecExp_16.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpackhi(in1, in2, 16, 4)));

Class(vunpacklo8_16x8i, VecExp_16.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpacklo(in1, in2, 16, 8)));

Class(vunpackhi8_16x8i, VecExp_16.binary(), _ev_binop_semantic_mixin, rec(
    semantic := (in1, in2, p) -> unpackhi(in1, in2, 16, 8)));

Class(vshuffle4_16x8i, VecExp_16.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, p, 16, 4),
    params := self >> sparams(4,4),
    permparams := iperm4
));

Class(vshuffle8_16x8i, VecExp_16.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, p, 16, 8),
    params := self >> sparams(2,2),
));

# Unary shuffles
#
Class(vushuffle4_16x8i, vushuffle_16x8i, rec(
    semantic := (in1, p) -> shuffle(in1, in1, p, 16, 4)));

Class(vushufflelo2_16x8i, vushuffle_16x8i, rec(
    semantic := (in1, p) -> shufflelo(in1, p, 16, 2)));

Class(vushufflehi2_16x8i, vushuffle_16x8i, rec(
    semantic := (in1, p) -> shufflehi(in1, p, 16, 2)));

# Viterbi
#
Class(interleavedmaskhi_16x8i, VecExp_8.binary(), rec(
    computeType := self >> TSym("short int")));

Class(interleavedmasklo_16x8i, VecExp_8.binary(), rec(
    computeType := self >> TSym("short int")));

Class(average_16x8i, VecExp_16.binary());

Class(vmovemask_16x8i, VecExp_16.unary(), rec(
    computeType := self >> T_UInt(16)));

Class(vcastizxlo, castizx, rec(
    ev := self >> vtakelo(self.args[1].t.toUnsigned().value(self.args[1].ev()))
));

Class(vcastizxhi, castizx, rec(
    ev := self >> vtakehi(self.args[1].t.toUnsigned().value(self.args[1].ev()))
));

Class(vcastuzxlo, castuzx, rec(
    ev := self >> vtakelo(self.args[1].t.toUnsigned().value(self.args[1].ev()))
));

Class(vcastuzxhi, castuzx, rec(
    ev := self >> vtakehi(self.args[1].t.toUnsigned().value(self.args[1].ev()))
));
