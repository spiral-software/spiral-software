
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 2).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 2, 1).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 2, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 2, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 4).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(ppu_4x32f) ]) ),
      origtree := IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(ppu_4x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 4, 1).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 8, 1, 1).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(16, 8, 1, 1).withTags([ AVecReg(ppu_4x32f) ]),
          SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
          SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ) ),
      origtree := IxLxI_kmn_km( TL(16, 8, 1, 1).withTags([ AVecReg(ppu_4x32f) ]),
          SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
          SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ) ),
      measured := 0,
      globalUnrolling := 10000 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 1, 1).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 4, 1, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 4, 1, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 1, 2).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
      origtree := SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 2, 1).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 1, 2).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases2( TL(8, 2, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
      origtree := SIMD_ISA_Bases2( TL(8, 2, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 2, 1).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 2, 2, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 2, 2, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 4, 1, 1).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(16, 4, 1, 1).withTags([ AVecReg(ppu_4x32f) ]),
          SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
          SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(ppu_4x32f) ]) ) ),
      origtree := IxLxI_kmn_n( TL(16, 4, 1, 1).withTags([ AVecReg(ppu_4x32f) ]),
          SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
          SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(ppu_4x32f) ]) ) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 2, 2).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 2, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 2, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 2, 1, 1).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(16, 2, 1, 1).withTags([ AVecReg(ppu_4x32f) ]),
          SIMD_ISA_Bases2( TL(8, 2, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
          SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(ppu_4x32f) ]) ) ),
      origtree := IxLxI_kmn_n( TL(16, 2, 1, 1).withTags([ AVecReg(ppu_4x32f) ]),
          SIMD_ISA_Bases2( TL(8, 2, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
          SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(ppu_4x32f) ]) ) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 1, 1).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 2, 1, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 2, 1, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 1).withTags([ AVecReg(spu_2x64f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 1, 1).withTags([ AVecReg(spu_2x64f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 1, 1).withTags([ AVecReg(spu_2x64f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(1, 1, 1, 2).withTags([ AVecReg(spu_2x64f) ]), [ rec(
      ruletree := IxLxI_vtensor( TL(1, 1, 1, 2).withTags([ AVecReg(spu_2x64f) ]) ),
      origtree := IxLxI_vtensor( TL(1, 1, 1, 2).withTags([ AVecReg(spu_2x64f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 2).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 2, 1).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 2, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 2, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 4).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(ppu_4x32f) ]) ),
      origtree := IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(ppu_4x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 4, 1).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 8, 1, 1).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(16, 8, 1, 1).withTags([ AVecReg(ppu_4x32f) ]),
          SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
          SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ) ),
      origtree := IxLxI_kmn_km( TL(16, 8, 1, 1).withTags([ AVecReg(ppu_4x32f) ]),
          SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
          SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ) ),
      measured := 0,
      globalUnrolling := 10000 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 1, 1).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 4, 1, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 4, 1, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 1, 2).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
      origtree := SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 2, 1).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 1, 2).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases2( TL(8, 2, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
      origtree := SIMD_ISA_Bases2( TL(8, 2, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 2, 1).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 2, 2, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 2, 2, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 4, 1, 1).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(16, 4, 1, 1).withTags([ AVecReg(ppu_4x32f) ]),
          SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
          SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(ppu_4x32f) ]) ) ),
      origtree := IxLxI_kmn_n( TL(16, 4, 1, 1).withTags([ AVecReg(ppu_4x32f) ]),
          SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
          SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(ppu_4x32f) ]) ) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 2, 2).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 2, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 2, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 2, 1, 1).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(16, 2, 1, 1).withTags([ AVecReg(ppu_4x32f) ]),
          SIMD_ISA_Bases2( TL(8, 2, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
          SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(ppu_4x32f) ]) ) ),
      origtree := IxLxI_kmn_n( TL(16, 2, 1, 1).withTags([ AVecReg(ppu_4x32f) ]),
          SIMD_ISA_Bases2( TL(8, 2, 1, 2).withTags([ AVecReg(ppu_4x32f) ]) ),
          SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(ppu_4x32f) ]) ) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 1, 1).withTags([ AVecReg(spu_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 2, 1, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 2, 1, 1).withTags([ AVecReg(ppu_4x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
