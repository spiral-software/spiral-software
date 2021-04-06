
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 2).withTags([ AVecReg(AVX_4x64f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 1, 2).withTags([ AVecReg(AVX_4x64f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 1, 2).withTags([ AVecReg(AVX_4x64f) ]) ),
      measured := 2,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 4).withTags([ AVecReg(AVX_4x64f) ]), [ rec(
      ruletree := IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(AVX_4x64f) ]) ),
      origtree := IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(AVX_4x64f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 8, 1, 1).withTags([ AVecReg(AVX_4x64f) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(16, 8, 1, 1).withTags([ AVecReg(AVX_4x64f) ]),
          SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(AVX_4x64f) ]) ),
          IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(AVX_4x64f) ]) ) ),
      origtree := IxLxI_kmn_km( TL(16, 8, 1, 1).withTags([ AVecReg(AVX_4x64f) ]),
          SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(AVX_4x64f) ]) ),
          IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(AVX_4x64f) ]) ) ),
      measured := 12.4,
      globalUnrolling := 10000 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 1, 1).withTags([ AVecReg(AVX_4x64f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 4, 1, 1).withTags([ AVecReg(AVX_4x64f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 4, 1, 1).withTags([ AVecReg(AVX_4x64f) ]) ),
      measured := 6.2000000000000002,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 1, 2).withTags([ AVecReg(AVX_4x64f) ]), [ rec(
      ruletree := SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(AVX_4x64f) ]) ),
      origtree := SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(AVX_4x64f) ]) ),
      measured := 4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 2, 1).withTags([ AVecReg(AVX_4x64f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(AVX_4x64f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(AVX_4x64f) ]) ),
      measured := 12.4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 1, 2).withTags([ AVecReg(AVX_4x64f) ]), [ rec(
      ruletree := SIMD_ISA_Bases2( TL(8, 2, 1, 2).withTags([ AVecReg(AVX_4x64f) ]) ),
      origtree := SIMD_ISA_Bases2( TL(8, 2, 1, 2).withTags([ AVecReg(AVX_4x64f) ]) ),
      measured := 4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 2, 1).withTags([ AVecReg(AVX_4x64f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 2, 2, 1).withTags([ AVecReg(AVX_4x64f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 2, 2, 1).withTags([ AVecReg(AVX_4x64f) ]) ),
      measured := 12.4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 4, 1, 1).withTags([ AVecReg(AVX_4x64f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(16, 4, 1, 1).withTags([ AVecReg(AVX_4x64f) ]),
          SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(AVX_4x64f) ]) ),
          SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(AVX_4x64f) ]) ) ),
      origtree := IxLxI_kmn_n( TL(16, 4, 1, 1).withTags([ AVecReg(AVX_4x64f) ]),
          SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(AVX_4x64f) ]) ),
          SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(AVX_4x64f) ]) ) ),
      measured := 16.399999999999999,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 2, 2).withTags([ AVecReg(AVX_4x64f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 2, 2).withTags([ AVecReg(AVX_4x64f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 2, 2).withTags([ AVecReg(AVX_4x64f) ]) ),
      measured := 4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 2, 1, 1).withTags([ AVecReg(AVX_4x64f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(16, 2, 1, 1).withTags([ AVecReg(AVX_4x64f) ]),
          IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(AVX_4x64f) ]) ),
          SIMD_ISA_Bases1( TL(8, 2, 2, 1).withTags([ AVecReg(AVX_4x64f) ]) ) ),
      origtree := IxLxI_kmn_n( TL(16, 2, 1, 1).withTags([ AVecReg(AVX_4x64f) ]),
          IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(AVX_4x64f) ]) ),
          SIMD_ISA_Bases1( TL(8, 2, 2, 1).withTags([ AVecReg(AVX_4x64f) ]) ) ),
      measured := 12.4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 1, 1).withTags([ AVecReg(AVX_4x64f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 2, 1, 1).withTags([ AVecReg(AVX_4x64f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 2, 1, 1).withTags([ AVecReg(AVX_4x64f) ]) ),
      measured := 6.2000000000000002,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 2, 1, 2).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(16, 2, 1, 2).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_vtensor( TL(4, 2, 1, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(8, 2, 2, 2).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_kmn_n( TL(16, 2, 1, 2).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_vtensor( TL(4, 2, 1, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(8, 2, 2, 2).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 12.4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 8, 2, 2).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(16, 8, 2, 2).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_kmn_km( TL(16, 8, 2, 2).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 24.800000000000001,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(64, 8, 1, 1).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(64, 8, 1, 1).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_kmn_n( TL(32, 8, 1, 2).withTags([ AVecReg(AVX_8x32f) ]),
            SIMD_ISA_Bases2( TL(16, 8, 1, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_kmn_km( TL(16, 8, 2, 2).withTags([ AVecReg(AVX_8x32f) ]),
              SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
              IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ) ),
          SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_kmn_n( TL(64, 8, 1, 1).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_kmn_n( TL(32, 8, 1, 2).withTags([ AVecReg(AVX_8x32f) ]),
            SIMD_ISA_Bases2( TL(16, 8, 1, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_kmn_km( TL(16, 8, 2, 2).withTags([ AVecReg(AVX_8x32f) ]),
              SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
              IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ) ),
          SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 57.599999999999994,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 2).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 1, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 1, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 4.5,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 4, 4).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 4, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 4, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 8,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 2, 4).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(8, 2, 2, 4).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(4, 2, 4, 4).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_kmn_n( TL(8, 2, 2, 4).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(4, 2, 4, 4).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 8,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 8, 2).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 8, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 8, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 36.0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 4, 2).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 24.800000000000001,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 2, 2).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 4, 2, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 4, 2, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 12.4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(64, 32, 1, 1).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(64, 32, 1, 1).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
          IxLxI_kmn_km( TL(8, 4, 1, 8).withTags([ AVecReg(AVX_8x32f) ]),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(AVX_8x32f) ]) ) ) ),
      origtree := IxLxI_kmn_km( TL(64, 32, 1, 1).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
          IxLxI_kmn_km( TL(8, 4, 1, 8).withTags([ AVecReg(AVX_8x32f) ]),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(AVX_8x32f) ]) ) ) ),
      measured := 24.800000000000001,
      globalUnrolling := 10000 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 4).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 1, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 1, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 2,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 4, 1).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 1.8,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 8, 1, 1).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(16, 8, 1, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(16, 8, 1, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 6.2000000000000002,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(64, 4, 1, 1).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(64, 4, 1, 1).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_kmn_km( TL(8, 4, 1, 8).withTags([ AVecReg(AVX_8x32f) ]),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(AVX_8x32f) ]) ) ),
          IxLxI_kmn_km( TL(32, 4, 2, 1).withTags([ AVecReg(AVX_8x32f) ]),
            SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(AVX_8x32f) ]),
              IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
              SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ) ) ) ),
      origtree := IxLxI_kmn_n( TL(64, 4, 1, 1).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_kmn_km( TL(8, 4, 1, 8).withTags([ AVecReg(AVX_8x32f) ]),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(AVX_8x32f) ]) ) ),
          IxLxI_kmn_km( TL(32, 4, 2, 1).withTags([ AVecReg(AVX_8x32f) ]),
            SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(AVX_8x32f) ]),
              IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
              SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ) ) ) ),
      measured := 49.600000000000001,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(32, 8, 1, 2).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(32, 8, 1, 2).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases2( TL(16, 8, 1, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
          IxLxI_kmn_km( TL(16, 8, 2, 2).withTags([ AVecReg(AVX_8x32f) ]),
            SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ) ),
      origtree := IxLxI_kmn_n( TL(32, 8, 1, 2).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases2( TL(16, 8, 1, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
          IxLxI_kmn_km( TL(16, 8, 2, 2).withTags([ AVecReg(AVX_8x32f) ]),
            SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ) ),
      measured := 32.799999999999997,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(32, 8, 2, 1).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(32, 8, 2, 1).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_kmn_km( TL(16, 8, 2, 2).withTags([ AVecReg(AVX_8x32f) ]),
            SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ),
          SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_kmn_n( TL(32, 8, 2, 1).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_kmn_km( TL(16, 8, 2, 2).withTags([ AVecReg(AVX_8x32f) ]),
            SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ),
          SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 49.600000000000001,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(64, 2, 1, 1).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(64, 2, 1, 1).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_kmn_n( TL(8, 2, 1, 8).withTags([ AVecReg(AVX_8x32f) ]),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ),
          SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_kmn_n( TL(64, 2, 1, 1).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_kmn_n( TL(8, 2, 1, 8).withTags([ AVecReg(AVX_8x32f) ]),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ),
          SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 24.800000000000001,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 8, 1, 2).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(16, 8, 1, 2).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(8, 4, 2, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
          IxLxI_vtensor( TL(4, 2, 1, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_kmn_km( TL(16, 8, 1, 2).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(8, 4, 2, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
          IxLxI_vtensor( TL(4, 2, 1, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 12.4,
      globalUnrolling := 10000 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 1, 8).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(8, 4, 1, 8).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
          IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_kmn_km( TL(8, 4, 1, 8).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
          IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 8, 1).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(8, 4, 8, 1).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(4, 2, 16, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(4, 2, 8, 2).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_kmn_km( TL(8, 4, 8, 1).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(4, 2, 16, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(4, 2, 8, 2).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 43.200000000000003,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(32, 2, 1, 2).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(32, 2, 1, 2).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_kmn_n( TL(8, 2, 1, 8).withTags([ AVecReg(AVX_8x32f) ]),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ),
          SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_kmn_n( TL(32, 2, 1, 2).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_kmn_n( TL(8, 2, 1, 8).withTags([ AVecReg(AVX_8x32f) ]),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ),
          SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 24.800000000000001,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(32, 2, 2, 1).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(32, 2, 2, 1).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_kmn_n( TL(32, 2, 2, 1).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 24.800000000000001,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 1, 8).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(8, 2, 1, 8).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(AVX_8x32f) ]) ),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_kmn_n( TL(8, 2, 1, 8).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(AVX_8x32f) ]) ),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 8, 1).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(8, 2, 8, 1).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(4, 2, 8, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(4, 2, 16, 1).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_kmn_n( TL(8, 2, 8, 1).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(4, 2, 8, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(4, 2, 16, 1).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 43.200000000000003,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 1, 2).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 4, 1, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 4, 1, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 6.2000000000000002,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 2, 1).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(8, 4, 2, 1).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(4, 2, 2, 2).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_kmn_km( TL(8, 4, 2, 1).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(4, 2, 2, 2).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 10.800000000000001,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 8).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_vtensor( TL(4, 2, 1, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := IxLxI_vtensor( TL(4, 2, 1, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 1, 4).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases2( TL(8, 2, 1, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases2( TL(8, 2, 1, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 1, 2).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 2, 1, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 2, 1, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 6.2000000000000002,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 2, 1).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(8, 2, 2, 1).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(4, 2, 2, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_kmn_n( TL(8, 2, 2, 1).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(4, 2, 2, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 10.800000000000001,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 4, 1, 4).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases2( TL(16, 4, 1, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases2( TL(16, 4, 1, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 8,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 4, 4, 1).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_IxLxI_up( TL(16, 4, 4, 1).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_IxLxI_up( TL(16, 4, 4, 1).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 49.600000000000001,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(32, 16, 1, 2).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(32, 16, 1, 2).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
          IxLxI_kmn_km( TL(8, 4, 1, 8).withTags([ AVecReg(AVX_8x32f) ]),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(AVX_8x32f) ]) ) ) ),
      origtree := IxLxI_kmn_km( TL(32, 16, 1, 2).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
          IxLxI_kmn_km( TL(8, 4, 1, 8).withTags([ AVecReg(AVX_8x32f) ]),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(AVX_8x32f) ]) ) ) ),
      measured := 24.800000000000001,
      globalUnrolling := 10000 ) ]);
SIMD_ISA_DB.hashAdd(TL(32, 16, 2, 1).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(32, 16, 2, 1).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_kmn_km( TL(32, 16, 2, 1).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 24.800000000000001,
      globalUnrolling := 10000 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 4, 1, 1).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_IxLxI_down( TL(16, 4, 1, 1).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(16, 8, 1, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(16, 8, 1, 1).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_IxLxI_down( TL(16, 4, 1, 1).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(16, 8, 1, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(16, 8, 1, 1).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 12.399999999999999,
      globalUnrolling := 10000 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 2, 4).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(8, 4, 2, 4).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(4, 2, 4, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_kmn_km( TL(8, 4, 2, 4).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(4, 2, 4, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 8,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 4, 2).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 24.800000000000001,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 2, 2, 2).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 24.800000000000001,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 8, 1, 4).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases2( TL(16, 8, 1, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases2( TL(16, 8, 1, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 8,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 8, 4, 1).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 24.800000000000001,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 2, 2).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 2, 2, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 2, 2, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 12.4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 2, 2).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 2, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 2, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 9.0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(32, 4, 1, 2).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(32, 4, 1, 2).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases2( TL(16, 4, 1, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_kmn_n( TL(32, 4, 1, 2).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases2( TL(16, 4, 1, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 32.799999999999997,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(32, 4, 2, 1).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(32, 4, 2, 1).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
          IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(AVX_8x32f) ]),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
            SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ) ) ),
      origtree := IxLxI_kmn_km( TL(32, 4, 2, 1).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
          IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(AVX_8x32f) ]),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
            SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ) ) ),
      measured := 49.600000000000001,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 2, 1, 1).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(16, 2, 1, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(16, 2, 1, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 6.2000000000000002,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 1, 4).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases2( TL(8, 4, 1, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases2( TL(8, 4, 1, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(64, 16, 1, 1).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(64, 16, 1, 1).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_kmn_km( TL(32, 16, 1, 2).withTags([ AVecReg(AVX_8x32f) ]),
            SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_kmn_km( TL(8, 4, 1, 8).withTags([ AVecReg(AVX_8x32f) ]),
              IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
              IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(AVX_8x32f) ]) ) ) ),
          IxLxI_kmn_km( TL(32, 16, 2, 1).withTags([ AVecReg(AVX_8x32f) ]),
            SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ) ),
      origtree := IxLxI_kmn_n( TL(64, 16, 1, 1).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_kmn_km( TL(32, 16, 1, 2).withTags([ AVecReg(AVX_8x32f) ]),
            SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_kmn_km( TL(8, 4, 1, 8).withTags([ AVecReg(AVX_8x32f) ]),
              IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ),
              IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(AVX_8x32f) ]) ) ) ),
          IxLxI_kmn_km( TL(32, 16, 2, 1).withTags([ AVecReg(AVX_8x32f) ]),
            SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ) ),
      measured := 49.600000000000001,
      globalUnrolling := 10000 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 4, 2, 2).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(16, 4, 2, 2).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_kmn_km( TL(8, 4, 2, 4).withTags([ AVecReg(AVX_8x32f) ]),
            SIMD_ISA_Bases1( TL(4, 2, 4, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ),
          SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_kmn_n( TL(16, 4, 2, 2).withTags([ AVecReg(AVX_8x32f) ]),
          IxLxI_kmn_km( TL(8, 4, 2, 4).withTags([ AVecReg(AVX_8x32f) ]),
            SIMD_ISA_Bases1( TL(4, 2, 4, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(AVX_8x32f) ]) ) ),
          SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 32.799999999999997,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 2, 4).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 2, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 2, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 4, 2).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 4, 2).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 18.0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 16).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 2, 1, 4).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases2( TL(16, 2, 1, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases2( TL(16, 2, 1, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 8,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 16, 1).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 16, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 16, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 7.2000000000000002,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 2, 4, 1).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(AVX_8x32f) ]) ),
      measured := 24.800000000000001,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 4, 1, 2).withTags([ AVecReg(AVX_8x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(16, 4, 1, 2).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases2( TL(8, 4, 1, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(8, 4, 2, 2).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      origtree := IxLxI_kmn_n( TL(16, 4, 1, 2).withTags([ AVecReg(AVX_8x32f) ]),
          SIMD_ISA_Bases2( TL(8, 4, 1, 4).withTags([ AVecReg(AVX_8x32f) ]) ),
          SIMD_ISA_Bases1( TL(8, 4, 2, 2).withTags([ AVecReg(AVX_8x32f) ]) ) ),
      measured := 16.399999999999999,
      globalUnrolling := 8 ) ]);
