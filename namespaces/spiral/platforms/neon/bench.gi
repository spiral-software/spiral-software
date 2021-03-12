
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


benchNEON := function()
    return rec(
        2x32f := rec(
            wht := rec(
                small := _defaultSizes(s->doSimdWht(s, NEON_HALF, rec(
		            verify := true, oddSizes := true, svct := true, stdTTensor := true, tsplPFA := false)), [4]),
                medium := _defaultSizes(s->doSimdWht(s, NEON_HALF, rec(
			    oddSizes := false, svct := true, stdTTensor := true, tsplPFA := false)), List([2..10], i->2^i))
                ),
            1d := rec(
                dft_sc := rec(
                    small := _defaultSizes(s->doSimdDft(s, NEON_HALF, rec(
				verify:=true, tsplBluestein:=false, interleavedComplex := false, PRDFT:=true, URDFT:= true, 
				cplxVect := true, stdTTensor := false, globalUnrolling:=10000)),
                        [ 2..32 ]),
                    medium := _defaultSizes(s->doSimdDft(s, NEON_HALF, rec(tsplRader:=false, tsplBluestein:=false, 
				tsplPFA:=false, oddSizes:=false, interleavedComplex := false)),
                        _svctSizes(1024, 16, 2)),

                    ),
                dft_ic := rec(
                    small := _defaultSizes(s->doSimdDft(s, NEON_HALF, rec(
				verify:=true, tsplBluestein:=false, interleavedComplex := true, PRDFT:=true, URDFT:= true,
				cplxVect := true, stdTTensor := false, globalUnrolling:=10000)),
                        [2..32]),
                    medium := _defaultSizes(s->doSimdDft(s, NEON_HALF, rec(
				tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, oddSizes:=false, interleavedComplex := true)),
                        _svctSizes(1024, 16, 2)),
                    medium_cx := _defaultSizes(s->doSimdDft(s, NEON_HALF, rec(
				tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, oddSizes:=false, interleavedComplex := true, 
				cplxVect := true, realVect := false)),
                        _svctSizes(1024, 16, 2)),
                    ),

                trdft := _defaultSizes(s->doSimdSymDFT(TRDFT, s, NEON_HALF, rec( verify:=true, 
                            interleavedComplex := true, PRDFT:=true, URDFT:= true, tsplBluestein := false, cplxVect := true,
                            realVect := true, propagateNth := true, useDeref := true, 
                            globalUnrolling:=10000)), 4*[1..32]),

                dht := _defaultSizes(s->doSimdSymDFT(TDHT, s, NEON_HALF, rec(verify :=true)), 8*[1..12]),
                dct2 := _defaultSizes(s->doSimdSymDFT(TDCT2, s, NEON_HALF, rec(verify :=true)), 8*[1..12]),
                dct3 := _defaultSizes(s->doSimdSymDFT(TDCT3, s, NEON_HALF, rec(verify :=true)), 8*[1..12]),
                dct4 := _defaultSizes(s->doSimdSymDFT(TDCT4, s, NEON_HALF, rec(verify :=true)), 8*[1..12]),
                dst2 := _defaultSizes(s->doSimdSymDFT(TDST2, s, NEON_HALF, rec(verify :=true)), 8*[1..12]),
                dst3 := _defaultSizes(s->doSimdSymDFT(TDST3, s, NEON_HALF, rec(verify :=true)), 8*[1..12]),
                dst4 := _defaultSizes(s->doSimdSymDFT(TDST4, s, NEON_HALF, rec(verify :=true)), 8*[1..12]),
                mdct := _defaultSizes(s->doSimdSymDFT(TMDCT, s, NEON_HALF, rec(verify :=true)), 8*[1..12]),
                imdct := _defaultSizes(s->doSimdSymDFT(TIMDCT, s, NEON_HALF, rec(verify :=true)), 8*[1..12])
                ),

            2d := rec(
                dft_ic := rec(
                    medium := _defaultSizes(s -> doSimdMddft(s, NEON_HALF, rec(
				interleavedComplex := true,
                                oddSizes := false, svct := true, splitL := false, pushTag := true, 
				flipIxA := false, stdTTensor := true, tsplPFA := false)),
                        4 * List([1..16], i->[i,i])),

                    small := _defaultSizes(s->doSimdMddft(s, NEON_HALF, rec(
				verify:=true, interleavedComplex := true, globalUnrolling:=10000,
                                tsplPFA := false, pushTag:= false, oddSizes := true, svct := true, splitL := false)),
                        List([2..16], i->[i,i]))
                    ),
                dft_sc := rec(
                    medium := _defaultSizes(s -> doSimdMddft(s, NEON_HALF, rec(
				interleavedComplex := false,
                                oddSizes := false, svct := true, splitL := false, pushTag := true, flipIxA := false, 
				stdTTensor := true, tsplPFA := false)),
                        4*List([1..16], i->[i,i])),
                    small := _defaultSizes(s->doSimdMddft(s, NEON_HALF, rec(verify:=true, interleavedComplex := false,
				globalUnrolling:=10000, tsplPFA := false, pushTag:= false, oddSizes := true, 
				svct := true, splitL := false)),
                        List([2..16], i->[i,i]))
                    ),
                dct2 := _defaultSizes(s->doSimdSymMDDFT(DCT2, s, NEON_HALF, rec(verify := true)), [4, 8, 12, 16]),
                dct3 := _defaultSizes(s->doSimdSymMDDFT(DCT3, s, NEON_HALF, rec(verify := true)), [4, 8, 12, 16]),
                dct4 := _defaultSizes(s->doSimdSymMDDFT(DCT4, s, NEON_HALF, rec(verify := true)), [4, 8, 12, 16]),
                dst2 := _defaultSizes(s->doSimdSymMDDFT(DST2, s, NEON_HALF, rec(verify := true)), [4, 8, 12, 16]),
                dst3 := _defaultSizes(s->doSimdSymMDDFT(DST3, s, NEON_HALF, rec(verify := true)), [4, 8, 12, 16]),
                dst4 := _defaultSizes(s->doSimdSymMDDFT(DST4, s, NEON_HALF, rec(verify := true)), [4, 8, 12, 16])
            )
            ),

        4x32f := rec(
            wht := rec(
                small := _defaultSizes(s -> doSimdWht(s, NEON, rec(
			    verify := true, oddSizes := true, svct := true, stdTTensor := true, tsplPFA := false)), 
		    List([1..3], i->2^i)),
                medium := _defaultSizes(s -> doSimdWht(s, NEON, rec(
			    oddSizes := false, svct := true, stdTTensor := true, tsplPFA := false
			    )), List([4..10], i->2^i))
                ),
            1d := rec(
                dft_sc := rec(
                    small := _defaultSizes(s->doSimdDft(s, NEON, rec(
				verify:=true, interleavedComplex := false, stdTTensor := false, globalUnrolling:=10000)),
                        [ 2..64 ]),
                    medium := _defaultSizes(s->doSimdDft(s, NEON, rec(
				tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, oddSizes:=false, interleavedComplex := false)),
                        _svctSizes(1024, 16, 4)),

                    ),
                dft_ic := rec(
                    small := _defaultSizes(s->doSimdDft(s, NEON, rec(
				verify:=true, interleavedComplex := true, PRDFT:=true, URDFT:= true, 
				cplxVect := true, stdTTensor := false, globalUnrolling:=10000)),
                        [ 2..64 ]),
                    medium := _defaultSizes(s->doSimdDft(s, NEON, rec(
				tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, 
				oddSizes:=false, interleavedComplex := true)),
                        _svctSizes(1024, 16, 4)),
                    medium_cx := _defaultSizes(s->doSimdDft(s, NEON, rec(
				tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, 
				oddSizes:=false, interleavedComplex := true,
                                cplxVect := true, realVect := false, PRDFT := false, URDFT := true)),
                        _svctSizes(1024, 16, 4))
                    ),

                rdft := _defaultSizes(s->doSimdSymDFT(TRDFT, s, NEON, rec(verify :=true)), 32*[1..12]),

                trdft := _defaultSizes(s->doSimdSymDFT(TRDFT, s, NEON, rec( 
			    verify := true, interleavedComplex := true, PRDFT:=true, URDFT:= true, 
			    tsplBluestein := false, cplxVect := true,   stdTTensor := false, 
			    realVect := true, propagateNth := true, useDeref := true, globalUnrolling:=10000)),
		    8*[1..32]),

                dht := _defaultSizes(s->doSimdSymDFT(TDHT, s, NEON, rec(verify :=true)), 32*[1..12]),
                dct2 := _defaultSizes(s->doSimdSymDFT(TDCT2, s, NEON, rec(verify :=true)), 32*[1..12]),
                dct3 := _defaultSizes(s->doSimdSymDFT(TDCT3, s, NEON, rec(verify :=true)), 32*[1..12]),
                dct4 := _defaultSizes(s->doSimdSymDFT(TDCT4, s, NEON, rec(verify :=true)), 32*[1..12]),
                dst2 := _defaultSizes(s->doSimdSymDFT(TDST2, s, NEON, rec(verify :=true)), 32*[1..12]),
                dst3 := _defaultSizes(s->doSimdSymDFT(TDST3, s, NEON, rec(verify :=true)), 32*[1..12]),
                dst4 := _defaultSizes(s->doSimdSymDFT(TDST4, s, NEON, rec(verify :=true)), 32*[1..12]),
                mdct := _defaultSizes(s->doSimdSymDFT(TMDCT, s, NEON, rec(verify :=true)), 32*[1..12]),
                imdct := _defaultSizes(s->doSimdSymDFT(TIMDCT, s, NEON, rec(verify :=true)), 32*[1..12])
                ),
            2d := rec(
                dft_ic := rec(
                    medium := _defaultSizes(s -> doSimdMddft(s, NEON, rec(
				interleavedComplex := true,
                                oddSizes := false, svct := true, splitL := false, pushTag := true, 
				flipIxA := false, stdTTensor := true, tsplPFA := false)),
                        16*List([1..8], i->[i,i])),
                    small := _defaultSizes(s->doSimdMddft(s, NEON, rec(
				verify:=true, interleavedComplex := true, globalUnrolling:=10000,
                                tsplPFA := false, pushTag:= false, oddSizes := true, svct := true, splitL := false)),
                        List([2..16], i->[i,i]))
                    ),
                dft_sc := rec(
                    medium := _defaultSizes(s -> doSimdMddft(s, NEON, rec(
				interleavedComplex := false,
                                oddSizes := false, svct := true, splitL := false, pushTag := true, 
				flipIxA := false, stdTTensor := true, tsplPFA := false)),
                        16 * List([1..8], i->[i,i])),
                    small := _defaultSizes(s->doSimdMddft(s, NEON, rec(verify:=true, interleavedComplex := false, 
				globalUnrolling:=10000, tsplPFA := false, pushTag:= false, oddSizes := true, 
				svct := true, splitL := false)),
                        List([2..16], i->[i,i]))
                    ),
                dct2 := _defaultSizes(s -> doSimdSymMDDFT(DCT2, s, NEON, rec(verify := true)), [4, 8, 12, 16]),
                dct3 := _defaultSizes(s -> doSimdSymMDDFT(DCT3, s, NEON, rec(verify := true)), [4, 8, 12, 16]),
                dct4 := _defaultSizes(s -> doSimdSymMDDFT(DCT4, s, NEON, rec(verify := true)), [4, 8, 12, 16]),
                dst2 := _defaultSizes(s -> doSimdSymMDDFT(DST2, s, NEON, rec(verify := true)), [4, 8, 12, 16]),
                dst3 := _defaultSizes(s -> doSimdSymMDDFT(DST3, s, NEON, rec(verify := true)), [4, 8, 12, 16]),
                dst4 := _defaultSizes(s -> doSimdSymMDDFT(DST4, s, NEON, rec(verify := true)), [4, 8, 12, 16]),
                conv := _defaultSizes(s -> let(
                        opts := SIMDGlobals.getOpts(NEON, rec(svct := true, oddSizes := false, 
				splitComplexTPrm := true, TRCDiag_VRCLR := true, globalUnrolling := 150,
				measureFinal := false)),
                        t := Flat(List(s, n -> let(
				    tt := TRConv2D(ImageVar([n, n])).withTags(opts.tags), [tt, tt.forwardTransform()]))),
                        spiral.libgen.DPBench.build(t, opts)), 16*[1..32])
            )
        )
    );
end;


