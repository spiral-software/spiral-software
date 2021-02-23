
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(AVX_LDST, rec(
    list := [],
    __call__ := meth(arg) 
        local self;
	self := arg[1];
	Append(self.list, Drop(arg, 1)); 
    end,

    getList := self >> self.list :: SSE_LDST.list,

));

_evshuf1 := self >> let(
    a := _unwrap(self.args[1].ev()), 
    p := self.args[2].p,
    self.semantic(a, p));

_evshuf2 := self >> let(
    a := _unwrap(self.args[1].ev()), 
    b := _unwrap(self.args[2].ev()), 
    p := self.args[3].p,
    self.semantic(a, b, p));

_evpack := self >> let(
    a := _unwrap(self.args[1].ev()), 
    b := _unwrap(self.args[2].ev()), 
    self.semantic(a, b, []));


#######################################################################################
#   AVX 4-way 64-bit floating-point instructions

vzero_4x64f_const := TVect(T_Real(64), 4).zero();
vzero_4x64f := () -> vzero_4x64f_const;

# Load
#
AVX_LDST( 
    # __m256d _mm256_maskload_pd(double *a, __m256i mask);
    Class(vloadmask_4x64f, VecExp_4.unary(), rec(
        mask := n -> Reversed(Flat(List([1..4], i->["0x00000000", When(i <= n, "0x80000000", "0x00000000")])))
    )),

    # __m256d _mm256_broadcast_sd(const double *a)
    Class(vbroadcast_4x64f, VecExp_4.unary()),
    # __m256d _mm256_loadu_pd(double const *a)
    Class(vloadu_4x64f, VecExp_4.unary()) 
);

# Store
#
AVX_LDST(
    # void _mm256_maskstore_pd(double *a, __m256i mask, __m256d b);
    Class(vstoremask_4x64f, VecStoreCommand.ternary(), rec(
        mask := n -> Reversed(Flat(List([1..4], i->["0x00000000", When(i <= n, "0x80000000", "0x00000000")])))
    )),

    # void _mm256_storeu_pd(double *a, __m256d b)
    Class(vstoreu_4x64f, VecStoreCommand.binary())
);

Class(addsub_4x64f,   VecExp_4.binary(), rec(
    ev := self >> add(self.args[1], mul(self.t.value([-1.0, 1.0, -1.0, 1.0]), self.args[2])).ev(),
));

Class(fmaddsub_4x64f, VecExp_4.ternary()); 

# __m256d _mm256_insertf128_pd(__m256d a, __m128d b, int offset);
Class(vinsert_2l_4x64f, VecExp_4.binary(), rec(
    ev := self >> let( 
        a := _unwrap(self.args[1].ev()),
        b := _unwrap(self.args[2].ev()),
        When( self.args[3].p[1] = 0, 
            b :: a{[3 .. 4]},
            a{[1 .. 2]} :: b )),
    computeType := self >> self.args[1].t, 
));

# __m128d _mm256_extractf128_pd(__m256d a, int offset);
Class(vextract_2l_4x64f, VecExp_4.unary(), rec(
    semantic := (in1, p) -> in1{[1+2*p[1] .. 2 + 2*p[1]]},
    ev       := _evshuf1,

    computeType := self >> Checked(IsVecT(self.args[1].t), 
	TVect(self.args[1].t.t, 2)), #self.args[1].t.size/2, 
));


# Binary shuffle
#

# __m256d _mm256_unpacklo_pd(__m256d a, __m256d b);
Class(vunpacklo_4x64f, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> [in1[1], in2[1], in1[3], in2[3]],
    ev := _evpack
));

# __m256d _mm256_unpackhi_pd(__m256d a, __m256d b);
Class(vunpackhi_4x64f, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> [in1[2], in2[2], in1[4], in2[4]],
    ev := _evpack
));

# __m256d _mm256_shuffle_pd(__m256d a, __m256d b, const int select);
Class(vshuffle_4x64f,  VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> [in1[p[1]], in2[p[2]], in1[p[3]+2], in2[p[4]+2]],
    params := self >> [[1,2], [1,2], [1,2], [1,2]],
    ev := _evshuf2
));

# __m256d _mm256_permute2_pd(__m256d a, __m256d b, __m256i control, int imm);
Class(vperm2_4x64f,    VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> let(lane1 := Concat(in1{[1..2]}, in2{[1..2]}), lane2 := Concat(in1{[3..4]}, in2{[3..4]}),
                                    [lane1[p[1]], lane1[p[2]], lane2[p[3]], lane2[p[4]]]),
    params := self >> [[1..4], [1..4], [1..4], [1..4]],
    ev := _evshuf2
));

#__m256d _mm256_blend_pd(__m256d m1, __m256d m2, const int mask);
Class(vblend_4x64f,  VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> List( Zip2(TransposedMat([in1, in2]), p), e -> e[1][e[2]]),
    params := self >> Replicate(4, [1,2]),
    ev := _evshuf2
));

# __m256d _mm256_permute2f128_pd(__m256d a, __m256d b, int control);
Class(vpermf128_4x64f, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> let(v128 := [[in1[1], in1[2]], [in1[3], in1[4]], [in2[1], in2[2]], [in2[3], in2[4]]],
                                    Concat(v128[p[1]], v128[p[2]])),
    params := self >> [[1..4], [1..4]],
    ev := _evshuf2
));


# Unary shuffle
#

# __m256d _mm256_permute_pd(__m256d a, int control);
Class(vperm_4x64f, VecExp_4.unary(), rec(
    semantic := (in1, p) -> [in1[p[1]], in1[p[2]], in1[p[3]+2], in1[p[4]+2]],
    params := self >> [[1,2], [1,2], [1,2], [1,2]],
    ev := _evshuf1
));

Class(vuunpacklo_4x64f, VecExp_4.unaryFromBinop(vunpacklo_4x64f));
Class(vuunpackhi_4x64f, VecExp_4.unaryFromBinop(vunpackhi_4x64f));
Class(vushuffle_4x64f,  VecExp_4.unaryFromBinop(vshuffle_4x64f));
Class(vuperm2_4x64f,    VecExp_4.unaryFromBinop(vperm2_4x64f));
Class(vupermf128_4x64f, VecExp_4.unaryFromBinop(vpermf128_4x64f));

#######################################################################################
#   AVX 8-way 32-bit float instructions

vzero_8x32f_const := TVect(T_Real(64), 8).zero();
vzero_8x32f := () -> vzero_8x32f_const;

Class(addsub_8x32f,   VecExp_8.binary(), rec(
    ev := self >> add(self.args[1], mul(self.t.value([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]), self.args[2])).ev(),
)); 
Class(fmaddsub_8x32f, VecExp_8.ternary()); 

# Load
#
AVX_LDST( 
    # __m256 _mm256_maskload_ps(float *a, __m256i mask);
    Class(vloadmask_8x32f, VecExp_8.unary(), rec(
        mask := n -> Reversed(List([1..8], i -> When(i <= n, "0x80000000", "0x00000000")))
    )),

    # __m256 _mm256_loadu_ps(float const *a)
    Class(vloadu_8x32f, VecExp_8.unary()) 
);

# Store
#
AVX_LDST(
    # void _mm256_maskstore_ps(float *a, __m256i mask, __m256 b);
    Class(vstoremask_8x32f, VecStoreCommand.ternary(), rec(
        mask := n -> Reversed(List([1..8], i->When(i <= n, "0x80000000", "0x00000000")))
    )),

    # void _mm256_storeu_ps(float *a, __m256 b)
    Class(vstoreu_8x32f, VecStoreCommand.binary())
);


# __m256 _mm256_insertf128_ps(__m256 a, __m128d , int offset);
Class(vinsert_4l_8x32f, VecExp_4.binary(), rec(
    ev := self >> let( 
        a := _unwrap(self.args[1].ev()),
        b := _unwrap(self.args[2].ev()),
        When( self.args[3].p[1] = 0, 
            b :: a{[5 .. 8]},
            a{[1 .. 4]} :: b )),
    computeType := self >> self.args[1].t, 
));

# __m128 _mm256_extractf128_pd(__m256 a, int offset);
Class(vextract_4l_8x32f, VecExp_8.unary(), rec(
    semantic := (in1, p) -> in1{[1+4*p[1] .. 4 + 4*p[1]]},
    ev       := _evshuf1,

    computeType := self >> Checked(IsVecT(self.args[1].t), 
	TVect(self.args[1].t.t, 4)), #self.args[1].t.size/2, 
));

# Binary shuffle
#

# __m256 _mm256_unpacklo_ps(__m256 a, __m256 b);
Class(vunpacklo_8x32f, VecExp_8.binary(), rec(
    semantic := (in1, in2, p) -> [in1[1], in2[1], in1[2], in2[2], in1[5], in2[5], in1[6], in2[6]],
    ev := _evpack,
));

# __m256 _mm256_unpackhi_ps(__m256 a, __m256 b);
Class(vunpackhi_8x32f, VecExp_8.binary(), rec(
    semantic := (in1, in2, p) -> [in1[3], in2[3], in1[4], in2[4], in1[7], in2[7], in1[8], in2[8]],
    ev := _evpack,
));

# __m256 _mm256_permute2f128_ps(__m256 a, __m256 b, int control);
Class(vpermf128_8x32f, VecExp_8.binary(), rec(
    semantic := (in1, in2, p) -> let(v128 := [in1{[1..4]}, in1{[5..8]}, in2{[1..4]}, in2{[5..8]}],
                                    Concat(v128[p[1]], v128[p[2]])),
    params := self >> [[1..4], [1..4]],
    ev := _evshuf2,
));

# __m256 _mm256_permute2_ps(__m256 a, __m256 b, __m256i control, int imm);
Class(vperm2_8x32f, VecExp_8.binary(), rec(
    semantic := (in1, in2, p) -> let(lane1 := Concat(in1{[1..4]}, in2{[1..4]}), lane2 := Concat(in1{[5..8]}, in2{[5..8]}),
                                    [lane1[p[1]], lane1[p[2]], lane1[p[3]], lane1[p[4]], lane2[p[1]], lane2[p[2]], lane2[p[3]], lane2[p[4]]]),
    params := self >> [[1..8], [1..8], [1..8], [1..8]],
    ev := _evshuf2
));

# __m256 _mm256_shuffle_ps(__m256 a, __m256 b, const int select);
Class(vshuffle_8x32f, VecExp_8.binary(), rec(
    semantic := (in1, in2, p) -> [in1[p[1]], in1[p[2]], in2[p[3]], in2[p[4]], in1[p[1]+4], in1[p[2]+4], in2[p[3]+4], in2[p[4]+4]],
    params := self >> [[1..4], [1..4], [1..4], [1..4]],
    ev := _evshuf2,
));

#__m256 _mm256_blend_ps(__m256 m1, __m256 m2, const int mask);
Class(vblend_8x32f,  VecExp_8.binary(), rec(
    semantic := (in1, in2, p) -> List( Zip2(TransposedMat([in1, in2]), p), e -> e[1][e[2]]),
    params := self >> Replicate(8, [1,2]),
    ev := _evshuf2
));


# Unary shuffle
#

# __m256 _mm256_permute_ps(__m256 a, int control);
Class(vperm_8x32f, VecExp_8.unary(), rec(
    semantic := (in1, p) -> [in1[p[1]], in1[p[2]], in1[p[3]], in1[p[4]], in1[p[1]+4], in1[p[2]+4], in1[p[3]+4], in1[p[4]+4]],
    params := self >> [[1..4], [1..4], [1..4], [1..4]],
    ev := _evshuf1
));

Class(vuunpacklo_8x32f, VecExp_8.unaryFromBinop(vunpacklo_8x32f));
Class(vuunpackhi_8x32f, VecExp_8.unaryFromBinop(vunpackhi_8x32f));
Class(vushuffle_8x32f,  VecExp_8.unaryFromBinop(vshuffle_8x32f));
Class(vupermf128_8x32f, VecExp_8.unaryFromBinop(vpermf128_8x32f));



# __m256 _mm256_permutevar_ps(__m256 a, __m256i control);
Class(vpermv_8x32f, VecExp_8.unary(), rec(
    semantic := (in1, p) -> [in1[p[1]], in1[p[2]], in1[p[3]], in1[p[4]], in1[p[5]+4], in1[p[6]+4], in1[p[7]+4], in1[p[8]+4]],
    params := self >> Replicate(8, [1,2]),
    ev := _evshuf1,
));

# __m256 _mm256_movehdup_ps (__m256 a);
Class(vhdup_8x32f, VecExp_8.unary(), rec(
    semantic := (in1, p) -> [in1[2], in1[2], in1[4], in1[4], in1[6], in1[6], in1[8], in1[8]],
    ev := self >> let( v := self._vval(1), self.t.value(self.semantic(v, []))),
));

# __m256 _mm256_moveldup_ps (__m256 a);
Class(vldup_8x32f, VecExp_8.unary(), rec(
    semantic := (in1, p) -> [in1[1], in1[1], in1[3], in1[3], in1[5], in1[5], in1[7], in1[7]],
    ev := self >> let( v := self._vval(1), self.t.value(self.semantic(v, []))),
));

# vcxtr_8x32f is a virtual instruction: in-register Tensor(L(4,2),I(2))
#    later rewritten to vblend_8x32f(x, vperm_8x32f(vupermf128_8x32f(x, [ 2, 1 ]), [3,4,1,2]), [1,1,2,2,2,2,1,1])
#    this helps SIMD_ISA_DB to find this permutation
Class(vcxtr_8x32f, VecExp_8.unary(), rec(
    semantic := (in1, p) -> [in1[1], in1[2], in1[5], in1[6], in1[3], in1[4], in1[7], in1[8]],
    ev := self >> let( v := self._vval(1), self.t.value(self.semantic(v, []))),
    vcost := 5, # 3 instructions + dependency penalty
));

# Type Conversion

# __m256 _mm256_cvtpd_ps (__m256d a)
Class(vcvt_8x32f_4x64f, VecExp_4.unary(), rec(
    computeType := (self) >> TVect(T_Real(32), 8)
));

# __m256d _mm256_cvtps_pd (__m128 a)
Class(vcvt_4x64f_4x32f, VecExp_4.unary(), rec(
    computeType := (self) >> TVect(T_Real(64), 4)
));

# __m256d _mm256_cvtepi32_pd (__m128i src)
Class(vcvt_4x64f_4x32i, VecExp_4.unary(), rec(
    computeType := (self) >> TVect(T_Real(64), 4)
));

# __m128i _mm256_cvtpd_epi32 (__m256d src), rounding
Class(vcvt_4x32i_4x64f, VecExp_4.unary(), rec(
    computeType := (self) >> TVect(T_Int(32), 4)
));

# __m128i _mm256_cvttpd_epi32 (__m256d src), truncation  
Class(vcvtt_4x32i_4x64f, VecExp_4.unary(), rec(
    computeType := (self) >> TVect(T_Int(32), 4)
));

# __m256 _mm256_cvtepi32_ps (__m256i src)
Class(vcvt_8x32f_8x32i, VecExp_8.unary(), rec(
    computeType := (self) >> TVect(T_Real(32), 8)
));

# type conversion 8x32f to 8x32i, rounding
# __m256i _mm256_cvtps_epi32 (__m256 a)
Class(vcvt_8x32i_8x32f, VecExp_8.unary(), rec(
    computeType := (self) >> TVect(T_Int(32), 8)
));

# type conversion 8x32f to 8x32i, truncation toward zero
# __m256i _mm256_cvttps_epi32 (__m256 a)
Class(vcvtt_8x32i_8x32f, VecExp_8.unary(), rec(
    computeType := (self) >> TVect(T_Int(32), 8)
));


