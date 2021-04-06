
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


benchAVX := function()
    if LocalConfig.cpuinfo.SIMD().hasAVX() then
        return rec(
            4x64f := rec(
                wht := _defaultSizes(s->doSimdWht(s, AVX_4x64f, rec(verify := true, oddSizes := false, svct := true)), List([4..8], i->2^i)),
                1d := rec(
                    dft_ic := rec(
                        small := rec(
                            real := _defaultSizes(s->doSimdDft(s, AVX_4x64f, rec(verify := true, globalUnrolling := 10000, tsplBluestein:=true,
                              PRDFT:=true, URDFT:= true, CT := true, minCost := true, tsplPFA := true, tsplCT := true, oddSizes := true, PD := true, PFA := true,
                              RealRader := true, Rader := true, realVect := true, cplxVect := false, stdTTensor := false, interleavedComplex := true)), [2..64]),
                            cmplx := _defaultSizes(s->doSimdDft(s, AVX_4x64f, rec(verify := true, globalUnrolling := 10000, tsplBluestein:=true,
                              PRDFT:=false, URDFT:= true, CT := true, minCost := true, tsplPFA := true, tsplCT := true, oddSizes := true, PD := true, PFA := true,
                              RealRader := true, Rader := true, realVect := false, cplxVect := true, stdTTensor := false, interleavedComplex := true)), [2..64])
                        ),
                        medium := _defaultSizes(s->doSimdDft(s, AVX_4x64f, rec(globalUnrolling := 128, tsplRader:=false, tsplBluestein:=false, tsplPFA:=false, oddSizes:=false,
                                interleavedComplex := true, cplxVect := false, realVect := true)),
                              List([4..16], i->2^i)),
                        medium_cx := _defaultSizes(s->doSimdDft(s, AVX_4x64f, rec(
                                globalUnrolling := 128, 
                                tsplRader       := false, 
                                tsplBluestein   := false, 
                                tsplPFA         := false, 
                                URDFT           := true, 
                                PRDFT           := false,
                                oddSizes        := false, 
                                cplxVect        := true,
                                realVect        := false,
                                svct            := false,
                                splitL          := true,
                                interleavedComplex := true)),
                              List([2..16], i->2^i))
                    ),

                    dft_sc := rec(
                        small :=  _defaultSizes(s -> doSimdDft(s, AVX_4x64f, rec(
				       verify     := true, 
				       stdTTensor := false, 
				       svct       := true,
				       flipIxA    := true,
				       realVect   := true, 
				       oddSizes   := true,
				       interleavedComplex := false, 
				       globalUnrolling := 10000
				  )), List([1 .. 8], i -> 2^i)), # tsplRader:=false, tsplBluestein:=false, tsplPFA:=false,

                        medium := _defaultSizes(s->doSimdDft(s, AVX_4x64f, rec(
				       verify        := true, 
				       tsplRader     := false,
				       tsplBluestein := false,
				       tsplPFA       := false,
				       oddSizes      := false,
				       globalUnrolling := 10000, 
				       interleavedComplex := false
				  )), 16 * [1 .. 8]),
                    ),

                    trdft := _defaultSizes(s -> doSimdSymDFT(TRDFT, s, AVX_4x64f, rec( 
				  verify     := true, 
				  PRDFT      := false, 
				  URDFT      := true, 
				  cplxVect   := true,
                                  stdTTensor := false,
				  realVect   := true,
				  svct       := true, # <-- added from dft_sc
				  #flipIxA    := true, #<-- added from dft_sc, do we need this?
				  useDeref   := true,
                                  oddSizes   := true, # need this for TRDFT(16)
				  propagateNth := true, 
                                  globalUnrolling :=10000,
                                  interleavedComplex := true, 
			      )), 8 * [1 .. 32]),
                )
            ),
            8x32f := rec(
                wht := _defaultSizes(s->doSimdWht(s, AVX_8x32f, rec(verify := true, oddSizes := false, svct := true)), List([6..8], i->2^i)),
                1d := rec(
                    dft_ic := rec(
                        small := rec(
                            real := _defaultSizes(s -> doSimdDft(s, AVX_8x32f, rec(
					  verify     := true, 
					  PRDFT      := true, 
					  URDFT      := true,
					  CT         := true,
					  minCost    := true,
					  tsplPFA    := true,
					  tsplCT     := true,
					  oddSizes   := true, 
					  PD         := true, 
					  PFA        := true,
					  RealRader  := true,
					  Rader      := true,
					  realVect   := true,
					  cplxVect   := false,
					  stdTTensor := false,
					  tsplBluestein := true,
					  globalUnrolling := 10000, 
					  interleavedComplex := true)), [2..64]),

                            cmplx := _defaultSizes(s->doSimdDft(s, AVX_8x32f, rec(
					  verify := true, globalUnrolling := 10000, tsplBluestein:=true,
					  PRDFT:=false, URDFT:= true, CT := true, minCost := true, tsplPFA := true,
					  tsplCT := true, oddSizes := true, PD := true, PFA := true,
					  RealRader := true, Rader := true, realVect := false, cplxVect := true, 
					  stdTTensor := false, interleavedComplex := true)), [2..64])
                        ),

                        medium := _defaultSizes(s->doSimdDft(s, AVX_8x32f, rec(
				      tsplRader     := false,
				      tsplBluestein := false,
				      tsplPFA := false,
				      oddSizes := false,
				      cplxVect := false, 
				      realVect := true,
				      globalUnrolling := 128,
                                      interleavedComplex := true, 
				  )), List([6..16], i->2^i)),

                        medium_cx := _defaultSizes(s->doSimdDft(s, AVX_8x32f, rec(
                                    globalUnrolling := 128,
                                    tsplRader       := false,
                                    tsplBluestein   := false,
                                    tsplPFA         := false,
                                    PRDFT           := false, 
                                    URDFT           := true,
                                    CT              := false,
                                    PD              := false,
                                    oddSizes        := false,
                                    cplxVect        := true,
                                    realVect        := false,
                                    svct            := true,
                                    flipIxA         := true,
                                    interleavedComplex := true
				)), List([4..16], i->2^i))
                    ),

                    dft_sc := rec(
                        small :=  _defaultSizes(s -> doSimdDft(s, AVX_8x32f, rec(
				       verify     := true, 
				       stdTTensor := false, 
				       svct       := true,
				       flipIxA    := true,
				       realVect   := true, 
				       oddSizes   := true,
				       interleavedComplex := false, 
				       globalUnrolling := 10000
				  )), List([1 .. 8], i -> 2^i)),

                        medium := _defaultSizes(s->doSimdDft(s, AVX_8x32f, rec(
				       verify        := true, 
				       tsplRader     := false,
				       tsplBluestein := false,
				       tsplPFA       := false,
				       oddSizes      := false,
				       globalUnrolling := 10000, 
				       interleavedComplex := false
				  )), 64*[1..4]),
                    ),

                    trdft := _defaultSizes(s -> doSimdSymDFT(TRDFT, s, AVX_8x32f, rec( 
				  verify     := true, 
				  PRDFT      := false, 
				  URDFT      := true, 
				  cplxVect   := true,
                                  stdTTensor := false,
				  realVect   := true,
				  svct       := true, # <-- added from dft_sc
				  #flipIxA    := true, #<-- added from dft_sc, do we need this?
				  useDeref   := true,
                                  oddSizes   := true, # need this for TRDFT(16)
				  propagateNth := true, 
                                  globalUnrolling :=10000,
                                  interleavedComplex := true, 
			      )), 16 * [1 .. 32]),
                )
            )
        );
    else
        return false;
    fi;
end;
