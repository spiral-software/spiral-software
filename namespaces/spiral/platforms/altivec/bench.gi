
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Don't forget to Import(platforms.cellSPU)

_svctSizes := (N, p, v) -> Filtered([1..N], i->ForAll(Factors(i), j->j<=p) and IsInt(i/v^2));
_defaultSizes := (func, default) -> ((arg) -> When(Length(arg) = 1, func(When(IsList(arg[1]), arg[1], [arg[1]])), func(default)));

Class(altivecOpts, rec(
            4x32f := rec(
                dft_sc := rec(
                    small := rec(verify:=true, interleavedComplex := false, stdTTensor := false, globalUnrolling:=10000),
                    medium := rec(tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, oddSizes:=false, interleavedComplex := false),
                ),
                dft_ic := rec(
                    small := rec(verify:=true, interleavedComplex := true, stdTTensor := false, globalUnrolling:=10000),
                    medium := rec(tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, oddSizes:=false, interleavedComplex := true),
                )
            )
));

benchAltiVec := function()
    if LocalConfig.cpuinfo.SIMD().hasAltiVec then
        return rec(
            4x32f := rec(
                1d := rec(
                    dft_sc := rec(
                        small := _defaultSizes(s->doSimdDft(s, altivec_4x32f, altivecOpts.4x32f.dft_sc.small), [ 2..64 ]),
                        medium := _defaultSizes(s->doSimdDft(s, altivec_4x32f, altivecOpts.4x32f.dft_sc.medium), _svctSizes(1024, 16, 4)),
#                        medium := spiral.libgen.doParSimdDft(altivec_4x32f, 1, _svctSizes(1024, 16, 4), false, true, false, false),
#                        large := spiral.libgen.doParSimdDft(altivec_4x32f, 1, List([4..20], i->2^i), false, true, false, false)

                    ),
                    dft_ic := rec(
                        small := _defaultSizes(s->doSimdDft(s, altivec_4x32f, altivecOpts.4x32f.dft_ic.small), [ 2..64 ]),
                        medium := _defaultSizes(s->doSimdDft(s, altivec_4x32f, altivecOpts.4x32f.dft_ic.medium), _svctSizes(1024, 16, 4))
#                        medium := spiral.libgen.doParSimdDft(altivec_4x32f, 1, _svctSizes(1024, 16, 4), false, true, false, true),
#                        large := spiral.libgen.doParSimdDft(altivec_4x32f, 1, List([4..20], i->2^i), false, true, false, true)
                    ),
                    rdft := _defaultSizes(s->doSimdSymDFT(TRDFT, s, altivec_4x32f, rec(verify :=true)), 32*[1..12]),
                    dht  := _defaultSizes(s->doSimdSymDFT(TDHT,  s, altivec_4x32f, rec(verify :=true)), 32*[1..12]),
                    dct2 := _defaultSizes(s->doSimdSymDFT(TDCT2, s, altivec_4x32f, rec(verify :=true)), 32*[1..12]),
                    dct3 := _defaultSizes(s->doSimdSymDFT(TDCT3, s, altivec_4x32f, rec(verify :=true)), 32*[1..12]),
                    dct4 := _defaultSizes(s->doSimdSymDFT(TDCT4, s, altivec_4x32f, rec(verify :=true)), 32*[1..12]),
                    dst2 := _defaultSizes(s->doSimdSymDFT(TDST2, s, altivec_4x32f, rec(verify :=true)), 32*[1..12]),
                    dst3 := _defaultSizes(s->doSimdSymDFT(TDST3, s, altivec_4x32f, rec(verify :=true)), 32*[1..12]),
                    dst4 := _defaultSizes(s->doSimdSymDFT(TDST4, s, altivec_4x32f, rec(verify :=true)), 32*[1..12])
                ),

                2d := rec(
                    dft_ic := rec(
                        medium := _defaultSizes(s->doSimdMddft(s, altivec_4x32f, rec(interleavedComplex := true,
                                    oddSizes := false, svct := true, splitL := false, pushTag := true, flipIxA := false, stdTTensor := true, tsplPFA := false)),
                                    16*List([1..5], i->[i,i])),
                        small := _defaultSizes(s->doSimdMddft(s, altivec_4x32f, rec(verify:=true, interleavedComplex := true, globalUnrolling:=10000,
                                    tsplPFA := false, pushTag:= false, oddSizes := true, svct := true, splitL := false)),
                                    List([2..16], i->[i,i]))
                    ),
                    dft_sc := rec(
                        medium := _defaultSizes(s->doSimdMddft(s, altivec_4x32f, rec(interleavedComplex := false,
                                    oddSizes := false, svct := true, splitL := false, pushTag := true, flipIxA := false, stdTTensor := true, tsplPFA := false)),
                                    16*List([1..5], i->[i,i])),
                        small := _defaultSizes(s->doSimdMddft(s, altivec_4x32f, rec(verify:=true, interleavedComplex := false, globalUnrolling:=10000,
                                    tsplPFA := false, pushTag:= false, oddSizes := true, svct := true, splitL := false)),
                                    List([2..16], i->[i,i]))
                    ),
                    dct2 := _defaultSizes(s->doSimdSymMDDFT(DCT2, s, altivec_4x32f, rec(verify := true)), [4, 8, 12, 16]),
                    dct3 := _defaultSizes(s->doSimdSymMDDFT(DCT3, s, altivec_4x32f, rec(verify := true)), [4, 8, 12, 16]),
                    dct4 := _defaultSizes(s->doSimdSymMDDFT(DCT4, s, altivec_4x32f, rec(verify := true)), [4, 8, 12, 16]),
                    dst2 := _defaultSizes(s->doSimdSymMDDFT(DST2, s, altivec_4x32f, rec(verify := true)), [4, 8, 12, 16]),
                    dst3 := _defaultSizes(s->doSimdSymMDDFT(DST3, s, altivec_4x32f, rec(verify := true)), [4, 8, 12, 16]),
                    dst4 := _defaultSizes(s->doSimdSymMDDFT(DST4, s, altivec_4x32f, rec(verify := true)), [4, 8, 12, 16])
                )
            ),
        );
    else
        return false;
    fi;
end;
