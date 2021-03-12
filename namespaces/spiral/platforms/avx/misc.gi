
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_avxT := (t, opts) -> IsVecT(t) and IsBound(opts.vector) and let( isa := opts.vector.isa,
    Cond( t.t = TReal and t.size = 4, isa=AVX_4x64f,
          t.t = TReal and t.size = 8, isa=AVX_8x32f,
          t.t = TInt  and t.size = 4, isa=AVX_4x64f,
          t.t = TInt  and t.size = 8, isa=AVX_8x32f,
          t.t in [T_Real(64), T_Int(64), T_UInt(64)] and t.size=4, true,
          t.t in [T_Real(32), T_Int(32), T_UInt(32)] and t.size=8, true,
          false));
