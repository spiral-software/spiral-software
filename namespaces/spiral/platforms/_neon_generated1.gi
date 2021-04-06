
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 2).withTags([ AVecReg(NEON) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 1, 2).withTags([ AVecReg(NEON) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 1, 2).withTags([ AVecReg(NEON) ]) ),
      measured := 2,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 2, 1).withTags([ AVecReg(NEON) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 2, 1).withTags([ AVecReg(NEON) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 2, 1).withTags([ AVecReg(NEON) ]) ),
      measured := 6.2000000000000002,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 4).withTags([ AVecReg(NEON) ]), [ rec(
      ruletree := IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(NEON) ]) ),
      origtree := IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(NEON) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 4, 1).withTags([ AVecReg(NEON) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(NEON) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(NEON) ]) ),
      measured := 12.4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 8, 1, 1).withTags([ AVecReg(NEON) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(16, 8, 1, 1).withTags([ AVecReg(NEON) ]),
          SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(NEON) ]) ),
          IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(NEON) ]) ) ),
      origtree := IxLxI_kmn_km( TL(16, 8, 1, 1).withTags([ AVecReg(NEON) ]),
          SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(NEON) ]) ),
          IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(NEON) ]) ) ),
      measured := 4,
      globalUnrolling := 10000 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 1, 1).withTags([ AVecReg(NEON) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 4, 1, 1).withTags([ AVecReg(NEON) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 4, 1, 1).withTags([ AVecReg(NEON) ]) ),
      measured := 2,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 1, 2).withTags([ AVecReg(NEON) ]), [ rec(
      ruletree := SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(NEON) ]) ),
      origtree := SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(NEON) ]) ),
      measured := 4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 2, 1).withTags([ AVecReg(NEON) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(NEON) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(NEON) ]) ),
      measured := 4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 1, 2).withTags([ AVecReg(NEON) ]), [ rec(
      ruletree := SIMD_ISA_Bases2( TL(8, 2, 1, 2).withTags([ AVecReg(NEON) ]) ),
      origtree := SIMD_ISA_Bases2( TL(8, 2, 1, 2).withTags([ AVecReg(NEON) ]) ),
      measured := 4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 2, 1).withTags([ AVecReg(NEON) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 2, 2, 1).withTags([ AVecReg(NEON) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 2, 2, 1).withTags([ AVecReg(NEON) ]) ),
      measured := 4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 4, 1, 1).withTags([ AVecReg(NEON) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(16, 4, 1, 1).withTags([ AVecReg(NEON) ]),
          SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(NEON) ]) ),
          SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(NEON) ]) ) ),
      origtree := IxLxI_kmn_n( TL(16, 4, 1, 1).withTags([ AVecReg(NEON) ]),
          SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(NEON) ]) ),
          SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(NEON) ]) ) ),
      measured := 8,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 2, 2).withTags([ AVecReg(NEON) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 2, 2).withTags([ AVecReg(NEON) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 2, 2).withTags([ AVecReg(NEON) ]) ),
      measured := 4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 2, 1, 1).withTags([ AVecReg(NEON) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(16, 2, 1, 1).withTags([ AVecReg(NEON) ]),
          IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(NEON) ]) ),
          SIMD_ISA_Bases1( TL(8, 2, 2, 1).withTags([ AVecReg(NEON) ]) ) ),
      origtree := IxLxI_kmn_n( TL(16, 2, 1, 1).withTags([ AVecReg(NEON) ]),
          IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(NEON) ]) ),
          SIMD_ISA_Bases1( TL(8, 2, 2, 1).withTags([ AVecReg(NEON) ]) ) ),
      measured := 4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 1, 1).withTags([ AVecReg(NEON) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 2, 1, 1).withTags([ AVecReg(NEON) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 2, 1, 1).withTags([ AVecReg(NEON) ]) ),
      measured := 2,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 1).withTags([ AVecReg(NEON_HALF) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 1, 1).withTags([ AVecReg(NEON_HALF) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 1, 1).withTags([ AVecReg(NEON_HALF) ]) ),
      measured := 2,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(1, 1, 1, 2).withTags([ AVecReg(NEON_HALF) ]), [ rec(
      ruletree := IxLxI_vtensor( TL(1, 1, 1, 2).withTags([ AVecReg(NEON_HALF) ]) ),
      origtree := IxLxI_vtensor( TL(1, 1, 1, 2).withTags([ AVecReg(NEON_HALF) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
