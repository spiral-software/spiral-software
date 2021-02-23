
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Don't forget to Import(platforms.cellSPU)

#F _svctSizes(N, p, v). Uptil N, multiples of p, vectorsize=v
#F Eg: spu_4x32f_medium uses _svctSizes(1024, 16, 4);
_svctSizes := (N, p, v) -> Filtered([1..N], i->ForAll(Factors(i), j->j<=p) and IsInt(i/v^2));

#F _svctSizes(N, p, v, from). Uptil N, multiples of p, vectorsize=v, beginning with from
#F Eg: spu_4x32f_medium uses _svctSizes(1024, 16, 4, 96);
_svctSizesFrom := (N, p, v, low) -> Filtered([low..N], i->ForAll(Factors(i), j->j<=p) and IsInt(i/v^2));

_defaultSizes := (func, default) -> ((arg) -> When(Length(arg) = 1, func(When(IsList(arg[1]), arg[1], [arg[1]])), func(default)));

Class(cellopts, rec(
#    ic := rec(oddSizes := false, svct := true, tsplRader:=false, tsplBluestein:=false,
#              splitL := false, pushTag := true, flipIxA := false,
#              stdTTensor := true, tsplPFA := false, interleavedComplex := true),
#
#    sc := rec(oddSizes := false, svct := true, tsplRader:=false, tsplBluestein:=false,
#              splitL := false, pushTag := true, flipIxA := false,
#              stdTTensor := true, tsplPFA := false, interleavedComplex := false)
sc := rec(tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, oddSizes:=false, interleavedComplex := false),
ic := rec(tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, oddSizes:=false, interleavedComplex := true),
));



Class(cellOpts, rec(
            2x64f := rec(
                dft_sc := rec(
                    small :=  rec(verify:=true, tsplBluestein:=false, interleavedComplex := false, PRDFT:=true, URDFT:= true, cplxVect := true, stdTTensor := false, globalUnrolling:=10000),
                    medium := rec(tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, oddSizes:=false, interleavedComplex := false),
                ),
                dft_ic := rec(
                    small := rec(verify:=true, tsplBluestein:=false, interleavedComplex := true, PRDFT:=true, URDFT:= true, cplxVect := true, stdTTensor := false, globalUnrolling:=10000),
                    medium := rec(tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, oddSizes:=false, interleavedComplex := true),
                )
            ),
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

benchSPU := function()
    if LocalConfig.cpuinfo.SIMD().hasSPU() then
        return rec(
            2x64f := rec(
                1d := rec(
                    dft_sc := rec(
                        small := _defaultSizes(s->doSimdDft(s, spu_2x64f, cellOpts.2x64f.split.small), [ 2..32 ]),
                        medium := _defaultSizes(s->doSimdDft(s, spu_2x64f, cellOpts.2x64f.split.medium), _svctSizes(1024, 16, 2)),
#                        medium := spiral.libgen.doParSimdDft(spu_2x64f, 1, _svctSizes(1024, 16, 4), false, true, false, false),
#                        large := spiral.libgen.doParSimdDft(spu_2x64f, 1, List([4..20], i->2^i), false, true, false, false)

                    ),
                    dft_ic := rec(
                        small := _defaultSizes(s->doSimdDft(s, spu_2x64f,  cellOpts.2x64f.dft_ic.small), [2..32]),
                        medium := _defaultSizes(s->doSimdDft(s, spu_2x64f, cellOpts.2x64f.dft_ic.medium), _svctSizes(1024, 16, 2)),
#                        medium := spiral.libgen.doParSimdDft(spu_2x64f, 1, _svctSizes(1024, 16, 4), false, true, false, true),
#                        large := spiral.libgen.doParSimdDft(spu_2x64f, 1, List([4..20], i->2^i), false, true, false, true)
                    ),
                    rdft := _defaultSizes(s->doSimdSymDFT(TRDFT, s, spu_2x64f, rec(verify :=true)), 32*[1..12]),
                    dht  := _defaultSizes(s->doSimdSymDFT(TDHT,  s, spu_2x64f, rec(verify :=true)), 8*[1..12]),
                    dct2 := _defaultSizes(s->doSimdSymDFT(TDCT2, s, spu_2x64f, rec(verify :=true)), 32*[1..12]),
                    dct3 := _defaultSizes(s->doSimdSymDFT(TDCT3, s, spu_2x64f, rec(verify :=true)), 32*[1..12]),
                    dct4 := _defaultSizes(s->doSimdSymDFT(TDCT4, s, spu_2x64f, rec(verify :=true)), 32*[1..12]),
                    dst2 := _defaultSizes(s->doSimdSymDFT(TDST2, s, spu_2x64f, rec(verify :=true)), 8*[1..12]),
                    dst3 := _defaultSizes(s->doSimdSymDFT(TDST3, s, spu_2x64f, rec(verify :=true)), 8*[1..12]),
                    dst4 := _defaultSizes(s->doSimdSymDFT(TDST4, s, spu_2x64f, rec(verify :=true)), 8*[1..12])
                ),
                2d := rec(
                    dft_ic := rec(
                        medium := _defaultSizes(s->doSimdMddft(s, spu_2x64f, rec(interleavedComplex := true,
                                    oddSizes := false, svct := true, splitL := false, pushTag := true, flipIxA := false, stdTTensor := true, tsplPFA := false)),
                                    4*List([1..16], i->[i,i])),
                        small := _defaultSizes(s->doSimdMddft(s, spu_2x64f, rec(verify:=true, interleavedComplex := true, globalUnrolling:=10000,
                                    tsplPFA := false, pushTag:= false, oddSizes := true, svct := true, splitL := false)),
                                    List([2..16], i->[i,i]))
                    ),
                    dft_sc := rec(
                        medium := _defaultSizes(s->doSimdMddft(s, spu_2x64f, rec(interleavedComplex := false,
                                    oddSizes := false, svct := true, splitL := false, pushTag := true, flipIxA := false, stdTTensor := true, tsplPFA := false)),
                                    4*List([1..16], i->[i,i])),
                        small := _defaultSizes(s->doSimdMddft(s, spu_2x64f, rec(verify:=true, interleavedComplex := false, globalUnrolling:=10000,
                                    tsplPFA := false, pushTag:= false, oddSizes := true, svct := true, splitL := false)),
                                    List([2..16], i->[i,i]))
                    ),
                    dct2 := _defaultSizes(s->doSimdSymMDDFT(DCT2, s, spu_2x64f, rec(verify := true)), [4, 8, 12, 16]),
                    dct3 := _defaultSizes(s->doSimdSymMDDFT(DCT3, s, spu_2x64f, rec(verify := true)), [4, 8, 12, 16]),
                    dct4 := _defaultSizes(s->doSimdSymMDDFT(DCT4, s, spu_2x64f, rec(verify := true)), [4, 8, 12, 16]),
                    dst2 := _defaultSizes(s->doSimdSymMDDFT(DST2, s, spu_2x64f, rec(verify := true)), [4, 8, 12, 16]),
                    dst3 := _defaultSizes(s->doSimdSymMDDFT(DST3, s, spu_2x64f, rec(verify := true)), [4, 8, 12, 16]),
                    dst4 := _defaultSizes(s->doSimdSymMDDFT(DST4, s, spu_2x64f, rec(verify := true)), [4, 8, 12, 16])
                ),
            ),
            4x32f := rec(
                1d := rec(
                    dft_sc := rec(
                        small := _defaultSizes(s->doSimdDft(s, spu_4x32f, cellOpts.4x32f.dft_sc.small), [ 2..64 ]),
                        medium := _defaultSizes(s->doSimdDft(s, spu_4x32f, cellOpts.4x32f.dft_sc.medium), _svctSizes(1024, 16, 4)),
#                        medium := spiral.libgen.doParSimdDft(spu_4x32f, 1, _svctSizes(1024, 16, 4), false, true, false, false),
#                        large := spiral.libgen.doParSimdDft(spu_4x32f, 1, List([4..20], i->2^i), false, true, false, false)

                    ),
                    dft_ic := rec(
                        small := _defaultSizes(s->doSimdDft(s, spu_4x32f, cellOpts.4x32f.dft_ic.small), [ 2..64 ]),
                        medium := _defaultSizes(s->doSimdDft(s, spu_4x32f, cellOpts.4x32f.dft_ic.medium), _svctSizes(1024, 16, 4))
#                        medium := spiral.libgen.doParSimdDft(spu_4x32f, 1, _svctSizes(1024, 16, 4), false, true, false, true),
#                        large := spiral.libgen.doParSimdDft(spu_4x32f, 1, List([4..20], i->2^i), false, true, false, true)
                    ),
                    rdft := _defaultSizes(s->doSimdSymDFT(TRDFT, s, spu_4x32f, rec(verify :=true)), 32*[1..12]),
                    dht  := _defaultSizes(s->doSimdSymDFT(TDHT,  s, spu_4x32f, rec(verify :=true)), 32*[1..12]),
                    dct2 := _defaultSizes(s->doSimdSymDFT(TDCT2, s, spu_4x32f, rec(verify :=true)), 32*[1..12]),
                    dct3 := _defaultSizes(s->doSimdSymDFT(TDCT3, s, spu_4x32f, rec(verify :=true)), 32*[1..12]),
                    dct4 := _defaultSizes(s->doSimdSymDFT(TDCT4, s, spu_4x32f, rec(verify :=true)), 32*[1..12]),
                    dst2 := _defaultSizes(s->doSimdSymDFT(TDST2, s, spu_4x32f, rec(verify :=true)), 32*[1..12]),
                    dst3 := _defaultSizes(s->doSimdSymDFT(TDST3, s, spu_4x32f, rec(verify :=true)), 32*[1..12]),
                    dst4 := _defaultSizes(s->doSimdSymDFT(TDST4, s, spu_4x32f, rec(verify :=true)), 32*[1..12])
                ),

                2d := rec(
                    dft_ic := rec(
                        medium := _defaultSizes(s->doSimdMddft(s, spu_4x32f, rec(interleavedComplex := true,
                                    oddSizes := false, svct := true, splitL := false, pushTag := true, flipIxA := false, stdTTensor := true, tsplPFA := false)),
                                    16*List([1..5], i->[i,i])),
                        small := _defaultSizes(s->doSimdMddft(s, spu_4x32f, rec(verify:=true, interleavedComplex := true, globalUnrolling:=10000,
                                    tsplPFA := false, pushTag:= false, oddSizes := true, svct := true, splitL := false)),
                                    List([2..16], i->[i,i]))
                    ),
                    dft_sc := rec(
                        medium := _defaultSizes(s->doSimdMddft(s, spu_4x32f, rec(interleavedComplex := false,
                                    oddSizes := false, svct := true, splitL := false, pushTag := true, flipIxA := false, stdTTensor := true, tsplPFA := false)),
                                    16*List([1..5], i->[i,i])),
                        medium128 := _defaultSizes(s->doSimdMddft(s, spu_4x32f, rec(interleavedComplex := false,
                                    oddSizes := false, svct := true, splitL := false, pushTag := true, flipIxA := false, stdTTensor := true, tsplPFA := false)),
                                    16*List([4], i->[8,i])),
                        small := _defaultSizes(s->doSimdMddft(s, spu_4x32f, rec(verify:=true, interleavedComplex := false, globalUnrolling:=10000,
                                    tsplPFA := false, pushTag:= false, oddSizes := true, svct := true, splitL := false)),
                                    List([2..16], i->[i,i]))
                    ),
                    dct2 := _defaultSizes(s->doSimdSymMDDFT(DCT2, s, spu_4x32f, rec(verify := true)), [4, 8, 12, 16]),
                    dct3 := _defaultSizes(s->doSimdSymMDDFT(DCT3, s, spu_4x32f, rec(verify := true)), [4, 8, 12, 16]),
                    dct4 := _defaultSizes(s->doSimdSymMDDFT(DCT4, s, spu_4x32f, rec(verify := true)), [4, 8, 12, 16]),
                    dst2 := _defaultSizes(s->doSimdSymMDDFT(DST2, s, spu_4x32f, rec(verify := true)), [4, 8, 12, 16]),
                    dst3 := _defaultSizes(s->doSimdSymMDDFT(DST3, s, spu_4x32f, rec(verify := true)), [4, 8, 12, 16]),
                    dst4 := _defaultSizes(s->doSimdSymMDDFT(DST4, s, spu_4x32f, rec(verify := true)), [4, 8, 12, 16])
                )
            ),
        );
    else
        return false;
    fi;
end;
