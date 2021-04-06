
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

ImportAll(transforms);
Import(compiler, code, spl, rewrite, sigma, search, formgen, profiler);
Import(paradigms, paradigms.common);

#   benchmark infrastructure
Include(bench);
Include(vec_ir);

# SIMD vector architectures
Load(spiral.platforms.sse);
Load(spiral.platforms.avx);
Load(spiral.platforms.vmx);
Load(spiral.platforms.scratch_x86);
Load(spiral.platforms.scalar);
Load(spiral.platforms.macro);
Load(spiral.platforms.neon);

# Experimental platforms below
Load(spiral.platforms.cellSPU);
Load(spiral.platforms.altivec);

Class(SIMDArchitectures, rec(
    hasMMX := False,
    hasWirelessMMX := False,
    hasSSE :=False,
    hasSSE2 :=False,
    hasSSE3 := False,
    hasSSSE3 := False,
    hasSSE4_1 := False,
    hasSSE4_2 := False,
    hasSSE5 := False,
    has3DNow := False,
    has3DNowExtended := False,
    has3DNowProfessional := False,
    hasAltiVec := False,
    hasSPU := False,
    hasAVX := True,
    hasAVX2 := False,
    hasVCP  := False,
    hasAVX3 := False
));

### Initialize SIMD framework

# Imports needed for _generated, don't move them up, so that each
# platform explicitly imports other platforms that it uses, i.e.,
# platforms.sse.
Import(paradigms.vector, paradigms.vector.sigmaspl, paradigms.vector.bases, paradigms.vector.breakdown);
Import(platforms.vmx);
Import(platforms.cellSPU);
Import(platforms.altivec);
Import(platforms.sse);
Import(platforms.avx);
Import(platforms.scalar);
Import(platforms.macro);
Import(platforms.neon);

# annoyingly one needs to add new archs here by hand...
Include(_sse_generated0);
Include(_avx_generated0);
Include(_vmx_generated0);
Include(_cell_generated0);
Include(_macro_generated0);
Include(_neon_generated0);
SIMD_ISA_DB.init0();

Include(_sse_generated1);
Include(_avx_generated1);
Include(_vmx_generated1);
Include(_cell_generated1);
Include(_macro_generated1);
Include(_neon_generated1);
SIMD_ISA_DB.init1();

# Intel Core2 Duo, Quad,...
Load(spiral.platforms.intel);
