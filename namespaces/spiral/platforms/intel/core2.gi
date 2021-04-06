
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#Don't forget to Import(platforms.intel);

#_svctSizes := (N, p, v) -> Filtered([1..N], i->ForAll(Factors(i), j->j<=p) and IsInt(i/v^2));
#_defaultSizes := (func, default) -> ((arg) -> When(Length(arg) = 1, func(When(IsList(arg[1]), arg[1], [arg[1]])), func(default)));

_cores := 4;
_vlen := 4;
_maxN := 16;
_factors :=  [1, 2, 3, 5];
_parSizes := (N, p, v) -> List(Set(Filtered(Flat(Map(Cartesian(v^2*p^2 * List([1.._maxN], i->2^i), _factors), Product)), i->i<=2^N and i>v^2*p^2)));
_large := List([10..20], i->2^i);

_smpOpts := (intlCmplx, buf, omp) -> rec(use_functions := false, use_openmp := omp, use_buffering := buf, interleavedComplex := intlCmplx,
    simd_opts:=rec(svct:=true, splitL:=false, oddSizes:=false));

benchCore2 := function()
    if LocalConfig.cpuinfo.SIMD().hasSSE3() and LocalConfig.cpuinfo.cores >= 2 then
        return rec(
            OpenMP_2x64f := rec(
                split := rec(
                    medium := _defaultSizes(s->spiral.libgen.doParSimdDft(SSE_2x64f, LocalConfig.cpuinfo.cores, s, _smpOpts(false, false, true)), _parSizes(_maxN, _cores, _vlen)),
                    large := _defaultSizes(s->spiral.libgen.doParSimdDft(SSE_2x64f, LocalConfig.cpuinfo.cores, s, _smpOpts(false, true, true)), _large)
                ),
                interleaved := rec(
                    medium := _defaultSizes(s->spiral.libgen.doParSimdDft(SSE_2x64f, LocalConfig.cpuinfo.cores, s, _smpOpts(true, false, true)), _parSizes(_maxN, _cores, _vlen)),
                    large := _defaultSizes(s->spiral.libgen.doParSimdDft(SSE_2x64f, 1, s, _smpOpts(true, true, true)), _large)
                )
            ),
            OpenMP_4x32f := rec(
                split := rec(
                    medium := _defaultSizes(s->spiral.libgen.doParSimdDft(SSE_4x32f, LocalConfig.cpuinfo.cores, s, _smpOpts(false, false, true)), _parSizes(_maxN, _cores, _vlen)),
                    large := _defaultSizes(s->spiral.libgen.doParSimdDft(SSE_4x32f, LocalConfig.cpuinfo.cores, s, _smpOpts(false, true, true)), _large),

                ),
                interleaved := rec(
                    medium := _defaultSizes(s->spiral.libgen.doParSimdDft(SSE_4x32f, LocalConfig.cpuinfo.cores, s, _smpOpts(true, false, true)), _parSizes(_maxN, _cores, _vlen)),
                    large := _defaultSizes(s->spiral.libgen.doParSimdDft(SSE_4x32f, LocalConfig.cpuinfo.cores, s, _smpOpts(true, true, true)), _large)
                )
            ),
            threads_2x64f := rec(
                split := rec(
                    medium := _defaultSizes(s->spiral.libgen.doParSimdDft(SSE_2x64f, LocalConfig.cpuinfo.cores, s, _smpOpts(false, false, false)), _parSizes(_maxN, _cores, _vlen)),
                    large := _defaultSizes(s->spiral.libgen.doParSimdDft(SSE_2x64f, LocalConfig.cpuinfo.cores, s, _smpOpts(false, true, false)), _large)

                ),
                interleaved := rec(
                    medium := _defaultSizes(s->spiral.libgen.doParSimdDft(SSE_2x64f, LocalConfig.cpuinfo.cores, s, _smpOpts(true, false, false)), _parSizes(_maxN, _cores, _vlen)),
                    large := _defaultSizes(s->spiral.libgen.doParSimdDft(SSE_2x64f, 1, s, _smpOpts(true, true, false)), _large)
                )
            ),
            threads_4x32f := rec(
                split := rec(
                    medium := _defaultSizes(s->spiral.libgen.doParSimdDft(SSE_4x32f, LocalConfig.cpuinfo.cores, s, _smpOpts(false, false, false)), _parSizes(_maxN, _cores, _vlen)),
                    large := _defaultSizes(s->spiral.libgen.doParSimdDft(SSE_4x32f, LocalConfig.cpuinfo.cores, s, _smpOpts(false, true, false)), _large)

                ),
                interleaved := rec(
                    medium := _defaultSizes(s->spiral.libgen.doParSimdDft(SSE_4x32f, LocalConfig.cpuinfo.cores, s, _smpOpts(true, false, false)), _parSizes(_maxN, _cores, _vlen)),
                    large := _defaultSizes(s->spiral.libgen.doParSimdDft(SSE_4x32f, LocalConfig.cpuinfo.cores, s, _smpOpts(true, true, false)), _large)
                )
            )
        );
    else
        return false;
    fi;
end;
