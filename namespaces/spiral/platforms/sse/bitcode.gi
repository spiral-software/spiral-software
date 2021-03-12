
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# -------------------------------------------------------------------------
# SSE bit-level instructions
# 128x1i: 128-way bit register
# -------------------------------------------------------------------------

#Class(exp_128x1,  rec(v := 128, computeType := self >> TVect(T_UInt(1), 128)));
#Class(cmd_128x1,  rec(v := 128, computeType := self >> TVect(T_UInt(1), 128)));

Class(sseb_shl, VecExp_128.binary());
Class(sseb_shr, VecExp_128.binary());
Class(sseb_xor, VecExp_128.binary());
Class(sseb_and, VecExp_128.binary());
Class(sseb_or , VecExp_128.binary());
Class(sseb_not, VecExp_128.binary());

Class(sseb_bcast, VecExp_128.binary());

Class(sseb_uload,      VecExp_128.unary());
Class(sseb_load_sd_64, VecExp_128.unary());
Class(sseb_load_sd_32, VecExp_128.unary());
Class(sseb_load_sd_16, VecExp_128.unary());
Class(sseb_load_lo_64, VecExp_128.binary());
Class(sseb_load_hi_64, VecExp_128.binary());

Class(sseb_ustore,      VecExpCommand.binary());
Class(sseb_store_sd_64, VecExpCommand.binary());
Class(sseb_store_sd_32, VecExpCommand.binary());
Class(sseb_store_sd_16, VecExpCommand.binary());

Class(sseb_store_lo_64, VecExpCommand.ternary());
Class(sseb_store_hi_64, VecExpCommand.ternary());

#Class(vunpacklo_16x32f,  vbinop_new, exp_16x32f, rec(semantic := (in1, in2, p) -> unpacklo(in1, in2, 16, 1)));
#Class(vunpackhi_16x32f,  vbinop_new, exp_16x32f, rec(semantic := (in1, in2, p) -> unpackhi(in1, in2, 16, 1)));
#Class(vunpacklo2_16x32f, vbinop_new, exp_16x32f, rec(semantic := (in1, in2, p) -> unpacklo(in1, in2, 16, 2)));
#Class(vunpackhi2_16x32f, vbinop_new, exp_16x32f, rec(semantic := (in1, in2, p) -> unpackhi(in1, in2, 16, 2)));
#Class(vunpacklo4_16x32f, vbinop_new, exp_16x32f, rec(semantic := (in1, in2, p) -> unpacklo(in1, in2, 16, 4)));
#Class(vunpackhi4_16x32f, vbinop_new, exp_16x32f, rec(semantic := (in1, in2, p) -> unpackhi(in1, in2, 16, 4)));

# 128-bit mask split up in 8 bit chunks
 _vstore_16x8_mask := nmany -> List([0..15], i -> "0x" :: HexStringInt(Cond(
    nmany >= (i+1)*8, 255,
    nmany < i*8, 0,
    (nmany mod 8) <> 0, let(b := nmany mod 8, 
	(2^b - 1) * 2^(8-b)),  # upper b bits (of 8) must be 1s
    0
)));


Class(SSEB_128, SIMD_Intel, rec(
    countrec := rec(
        ops := [
            [add, sub], 
	    [mul],
            [sseb_bcast], # shuffles 
            [sseb_load_sd_64, sseb_load_sd_32, sseb_load_sd_16],
            [deref],
            Value      # Value without [] is a keyword in countOps !!
        ],
        printstrings := ["[adds]", "[mults]", "[vperms]", "[svldst]", "[vldst]", "[vval]"],
        type := "TVect",
        arithcost := (self, opcount) >> opcount[1]+opcount[2]
    ),

    includes := () -> ["<include/bitsse.h>"] :: _MM_MALLOC() :: _PMMINTRIN() :: _EMMINTRIN() :: _XMMINTRIN(), 
    active   := true,
    info     := "SSE2 128 x 1-bit",
    ctype    := "__bit",
    instr    := [],
    v        := 128,
    t        := BitVector(128),
    bits     := 1,
    isFloat  := false,
    isFixedPoint := false,
    splopts := rec(),

    v_ones  := BitVector(128).one(), 
    v_zeros := BitVector(128).zero(),
    val     := bits -> BitVector(128).value(bits),

    dupload  := (self, y, x) >> Checked(ObjId(x)=nth, # NOTE: is there a better way?
	let(base := x.loc,
	    ofs  := x.idx,
	    v    := self.v,
	    xvec := Cond(IsUnalignedPtrT(base.t), sseb_uload(base + idiv(ofs,v)*v),
		         vtref(self.t, base, idiv(ofs, v))),
	    assign(y, sseb_bcast(xvec, imod(ofs, v))))),

    svload := [ [ ], # load using subvecs of len 1
                [ ], # load using subvecs of len 2
                [ ], # load using subvecs of len 4
    ],

    svstore := [ [ ], # store using subvecs of len 1
                 [ ], # store using subvecs of len 2
                 [ ], # store using subvecs of len 4
    ],

    # keep the n lower scalars and zero the other ones
    mask := (self, c, n) >> Cond(
	n = 128, c,
        bin_and(c, self.val(Replicate(n, 1) :: Replicate(self.v - n, 0)))),


    loadCont := (self, n, y, yofs, x, xofs, xofs_align, opts) >> let(
#	_t := Checked(xofs=0, 0), # NOTE: remove this limitation, this is because xofs is in bits
	a  := _unwrap(xofs_align),
	nn := _unwrap(n), 
	yy := vtref(self.t, y, yofs),
	m  := x -> self.mask(x, nn),
	# NOTE: BELOW CODE IS NOT VERIFIED, in particular shift directions might be invalid
	# NOTE: sseb_shl/shr need to be implemented in the backend, 
        # NOTE : SSE requires shift values to be immediates (or does it not?)
	Cond(a = 0 and not IsUnalignedPtrT(x.t), 
	         assign(yy, m(vtref(self.t, x, div(xofs, self.v)))),

	     # known alignment, sv is small, so that we only need 1 aligned load + 1 shift + mask
	     IsInt(a) and not IsUnalignedPtrT(x.t) and (nn <= self.v - a), 
		 let(v1 := vtref(self.t, x, div(xofs - a, self.v)), 
		     assign(yy, m(sseb_shl(v1, a)))),

	     # known alignment, sv covers 2 vectors, 2 aligned loads + 2 shifts + mask
	     # NB: no masking is needed because shifts will do the job
	     IsInt(a) and not IsUnalignedPtrT(x.t),
		 let(v1 := vtref(self.t, x, div(xofs - a, self.v)), 
		     v2 := vtref(self.t, x, div(xofs - a, self.v) + 1), 
		     assign(yy, m(sseb_or(sseb_shl(v1, a), sseb_shr(v2, self.v - a))))),
		 
             # else, unknown alignment, use unaligned load
	     assign(yy, m(sseb_uload(x + xofs)))
        )
    ),

    # NOTE: The store mask has 8-bit granularity, how do we express this??
    storeCont := (self, n, y, yofs, yofs_align, x, xofs, opts) >> let(
#	_t := Checked(yofs=0, 0), # NOTE: remove this limitation, this is because yofs is in bits
	a  := _unwrap(yofs_align),
	nn := _unwrap(n), 
	xx := vtref(self.t, x, xofs),
	Cond(nn = self.v and a = 0 and not IsUnalignedPtrT(y.t), 
	        assign(vtref(self.t, y, div(yofs, self.v)), xx), 
	     nn = self.v, 
	        sseb_ustore(y + yofs, xx), 
	     # else
	     #NOTE: yofs is in bits:
	        vstoremsk_16x8i(nth(y, yofs).toPtr(TReal), xx, _vstore_16x8_mask(nn)))),

));

_bitsBytes := (bits) -> let(bytelsts := List([0..Length(bits)/8-1], i -> bits{ [1..8] + 8*i }),
	                 List(bytelsts, l -> Sum([ 1 .. 8 ], i -> l[i] * 2 ^ (i - 1))));

Class(BitSSEUnparser, SSEUnparser, rec(
    infixbreak := 20,

    BitVector := (self, t, vars, i, is) >> Print("__m128i ", self.infix(vars, ", ", i + is)),

    Value := (self, o, i, is) >> Cond(ObjId(o.t) = BitVector, 
	let(bytenums := _bitsBytes(o.v),
            Cond(self.cx.isInside(Value) and Length(self.cx.Value) >= 2, # nested in an array
		 Print(            "{", self.infix(bytenums, ", "), "}"),
		 Print("_mm_set_epi8(", self.infix(Reversed(bytenums), ", "), ")"))),

	Inherited(o, i, is)),

    mul := (self, o, i, is) >> Cond(ObjId(o.t) = BitVector and Length(o.args)=2, 
	self.prefix("_mm_and_si128", o.args),
	Inherited(o, i, is)),

    add := (self, o, i, is) >> Cond(ObjId(o.t) = BitVector and Length(o.args)=2, 
	self.prefix("_mm_xor_si128", o.args),
	Inherited(o, i, is)),

    sseb_uload := (self, o, i, is) >> self(vloadu_16x8i(o.args[1]), i, is),

    sseb_ustore := (self, o, i, is) >> self(vstoreu_16x8i(o.args[1], o.args[2]), i, is)
));

