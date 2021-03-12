
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


benchMACRO := function()
    return rec(
        2xf := rec(
            wht := _defaultSizes(s->doSimdWht(s, MACRO_2xf, rec(verify := true, oddSizes := false, svct := true)), List([3..8], i->2^i)),
            1d := rec(
                dft_ic := rec(
                    small := rec(
                        real := _defaultSizes(s->doSimdDft(s, MACRO_2xf, rec(verify := true, globalUnrolling := 10000, tsplBluestein:=true,
                          PRDFT:=true, URDFT:= true, CT := true, minCost := true, tsplPFA := true, tsplCT := true, oddSizes := true, PD := true, PFA := true,
                          RealRader := true, Rader := true, realVect := true, cplxVect := false, stdTTensor := false, interleavedComplex := true)), [2..64]),
                        cmplx := _defaultSizes(s->doSimdDft(s, MACRO_2xf, rec(verify := true, globalUnrolling := 10000, tsplBluestein:=true,
                          PRDFT:=true, URDFT:= true, CT := true, minCost := true, tsplPFA := true, tsplCT := true, oddSizes := true, PD := true, PFA := true,
                          RealRader := true, Rader := true, realVect := false, cplxVect := true, stdTTensor := false, interleavedComplex := true)), [2..64])
                    ),
                    medium := _defaultSizes(s->doSimdDft(s, MACRO_2xf, rec(verify := true, globalUnrolling := 128, tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, oddSizes:=false,
                            interleavedComplex := true, cplxVect := false, realVect := true)),
                          List([2..16], i->2^i)),
                    medium_cx := _defaultSizes(s->doSimdDft(s, MACRO_2xf, rec(globalUnrolling := 64, tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, oddSizes:=false,
                            interleavedComplex := true, cplxVect := true, realVect := false)),
                          List([1..16], i->2^i))
                ),
                dft_sc := _defaultSizes(s->doSimdDft(s, MACRO_2xf, rec(verify := true, globalUnrolling := 10000, tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, oddSizes:=false, interleavedComplex := false)), 8*[1..16])
            )
        ),

        4xf := rec(
            wht := _defaultSizes(s->doSimdWht(s, MACRO_4xf, rec(verify := true, oddSizes := false, svct := true)), List([4..8], i->2^i)),
            1d := rec(
                dft_ic := rec(
                    small := rec(
                        real := _defaultSizes(s->doSimdDft(s, MACRO_4xf, rec(verify := true, globalUnrolling := 10000, tsplBluestein:=true,
                          PRDFT:=true, URDFT:= true, CT := true, minCost := true, tsplPFA := true, tsplCT := true, oddSizes := true, PD := true, PFA := true,
                          RealRader := true, Rader := true, realVect := true, cplxVect := false, stdTTensor := false, interleavedComplex := true)), [2..64]),
                        cmplx := _defaultSizes(s->doSimdDft(s, MACRO_4xf, rec(verify := true, globalUnrolling := 10000, tsplBluestein:=true,
                          PRDFT:=true, URDFT:= true, CT := true, minCost := true, tsplPFA := true, tsplCT := true, oddSizes := true, PD := true, PFA := true,
                          RealRader := true, Rader := true, realVect := false, cplxVect := true, stdTTensor := false, interleavedComplex := true)), [2..64])
                    ),
                    medium := _defaultSizes(s->doSimdDft(s, MACRO_4xf, rec(verify := true, globalUnrolling := 128, tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, oddSizes:=false,
                            interleavedComplex := true, cplxVect := false, realVect := true)),
                          List([4..16], i->2^i)),
                    medium_cx := _defaultSizes(s->doSimdDft(s, MACRO_4xf, rec(globalUnrolling := 64, tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, oddSizes:=false,
                            interleavedComplex := true, cplxVect := true, realVect := false)),
                          List([2..16], i->2^i))
                ),
                dft_sc := _defaultSizes(s->doSimdDft(s, MACRO_4xf, rec(verify := true, globalUnrolling := 10000, tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, oddSizes:=false, interleavedComplex := false)), 16*[1..8])
            )
        ),
        8xf := rec(
            wht := _defaultSizes(s->doSimdWht(s, MACRO_8xf, rec(verify := true, oddSizes := false, svct := true)), List([6..8], i->2^i)),
            1d := rec(
                dft_ic := rec(
                    small := rec(
                        real := _defaultSizes(s->doSimdDft(s, MACRO_8xf, rec(verify := true, globalUnrolling := 10000, tsplBluestein:=true,
                          PRDFT:=true, URDFT:= true, CT := true, minCost := true, tsplPFA := true, tsplCT := true, oddSizes := true, PD := true, PFA := true,
                          RealRader := true, Rader := true, realVect := true, cplxVect := false, stdTTensor := false, interleavedComplex := true)), [2..64]),
                        cmplx := _defaultSizes(s->doSimdDft(s, MACRO_8xf, rec(verify := true, globalUnrolling := 10000, tsplBluestein:=true,
                          PRDFT:=true, URDFT:= true, CT := true, minCost := true, tsplPFA := true, tsplCT := true, oddSizes := true, PD := true, PFA := true,
                          RealRader := true, Rader := true, realVect := false, cplxVect := true, stdTTensor := false, interleavedComplex := true)), [2..64])
                    ),
                    medium := _defaultSizes(s->doSimdDft(s, MACRO_8xf, rec(verify := true, globalUnrolling := 128, tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, oddSizes:=false,
                            interleavedComplex := true, cplxVect := false, realVect := true)),
                          List([6..16], i->2^i)),
                    medium_cx := _defaultSizes(s->doSimdDft(s, MACRO_8xf, rec(globalUnrolling := 128, tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, oddSizes:=false,
                            interleavedComplex := true, cplxVect := true, realVect := false)),
                          List([4..16], i->2^i))
                ),
                dft_sc := _defaultSizes(s->doSimdDft(s, MACRO_8xf, rec(verify := true, globalUnrolling := 10000, tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, oddSizes:=false, interleavedComplex := false)), 64*[1..4])
            )
        )
    );
end;
