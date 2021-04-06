
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 1).withTags([ AVecReg(MACRO_2xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 1, 1).withTags([ AVecReg(MACRO_2xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 1, 1).withTags([ AVecReg(MACRO_2xf) ]) ),
      measured := 2,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(1, 1, 1, 2).withTags([ AVecReg(MACRO_2xf) ]), [ rec(
      ruletree := IxLxI_vtensor( TL(1, 1, 1, 2).withTags([ AVecReg(MACRO_2xf) ]) ),
      origtree := IxLxI_vtensor( TL(1, 1, 1, 2).withTags([ AVecReg(MACRO_2xf) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 2).withTags([ AVecReg(MACRO_4xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 1, 2).withTags([ AVecReg(MACRO_4xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 1, 2).withTags([ AVecReg(MACRO_4xf) ]) ),
      measured := 2,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 2, 1).withTags([ AVecReg(MACRO_4xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 2, 1).withTags([ AVecReg(MACRO_4xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 2, 1).withTags([ AVecReg(MACRO_4xf) ]) ),
      measured := 6.2000000000000002,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 4).withTags([ AVecReg(MACRO_4xf) ]), [ rec(
      ruletree := IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(MACRO_4xf) ]) ),
      origtree := IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(MACRO_4xf) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 4, 1).withTags([ AVecReg(MACRO_4xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(MACRO_4xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(MACRO_4xf) ]) ),
      measured := 12.4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 8, 1, 1).withTags([ AVecReg(MACRO_4xf) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(16, 8, 1, 1).withTags([ AVecReg(MACRO_4xf) ]),
          SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(MACRO_4xf) ]) ),
          IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(MACRO_4xf) ]) ) ),
      origtree := IxLxI_kmn_km( TL(16, 8, 1, 1).withTags([ AVecReg(MACRO_4xf) ]),
          SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(MACRO_4xf) ]) ),
          IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(MACRO_4xf) ]) ) ),
      measured := 4,
      globalUnrolling := 10000 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 1, 1).withTags([ AVecReg(MACRO_4xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 4, 1, 1).withTags([ AVecReg(MACRO_4xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 4, 1, 1).withTags([ AVecReg(MACRO_4xf) ]) ),
      measured := 2,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 1, 2).withTags([ AVecReg(MACRO_4xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(MACRO_4xf) ]) ),
      origtree := SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(MACRO_4xf) ]) ),
      measured := 4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 2, 1).withTags([ AVecReg(MACRO_4xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(MACRO_4xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(MACRO_4xf) ]) ),
      measured := 4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 1, 2).withTags([ AVecReg(MACRO_4xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases2( TL(8, 2, 1, 2).withTags([ AVecReg(MACRO_4xf) ]) ),
      origtree := SIMD_ISA_Bases2( TL(8, 2, 1, 2).withTags([ AVecReg(MACRO_4xf) ]) ),
      measured := 4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 2, 1).withTags([ AVecReg(MACRO_4xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 2, 2, 1).withTags([ AVecReg(MACRO_4xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 2, 2, 1).withTags([ AVecReg(MACRO_4xf) ]) ),
      measured := 4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 4, 1, 1).withTags([ AVecReg(MACRO_4xf) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(16, 4, 1, 1).withTags([ AVecReg(MACRO_4xf) ]),
          SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(MACRO_4xf) ]) ),
          SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(MACRO_4xf) ]) ) ),
      origtree := IxLxI_kmn_n( TL(16, 4, 1, 1).withTags([ AVecReg(MACRO_4xf) ]),
          SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(MACRO_4xf) ]) ),
          SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(MACRO_4xf) ]) ) ),
      measured := 8,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 2, 2).withTags([ AVecReg(MACRO_4xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 2, 2).withTags([ AVecReg(MACRO_4xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 2, 2).withTags([ AVecReg(MACRO_4xf) ]) ),
      measured := 4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 2, 1, 1).withTags([ AVecReg(MACRO_4xf) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(16, 2, 1, 1).withTags([ AVecReg(MACRO_4xf) ]),
          IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(MACRO_4xf) ]) ),
          SIMD_ISA_Bases1( TL(8, 2, 2, 1).withTags([ AVecReg(MACRO_4xf) ]) ) ),
      origtree := IxLxI_kmn_n( TL(16, 2, 1, 1).withTags([ AVecReg(MACRO_4xf) ]),
          IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(MACRO_4xf) ]) ),
          SIMD_ISA_Bases1( TL(8, 2, 2, 1).withTags([ AVecReg(MACRO_4xf) ]) ) ),
      measured := 4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 1, 1).withTags([ AVecReg(MACRO_4xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 2, 1, 1).withTags([ AVecReg(MACRO_4xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 2, 1, 1).withTags([ AVecReg(MACRO_4xf) ]) ),
      measured := 2,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 2, 1, 2).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(16, 2, 1, 2).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_vtensor( TL(4, 2, 1, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
          SIMD_ISA_Bases1( TL(8, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      origtree := IxLxI_kmn_n( TL(16, 2, 1, 2).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_vtensor( TL(4, 2, 1, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
          SIMD_ISA_Bases1( TL(8, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      measured := 4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 8, 2, 2).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(16, 8, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
          SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      origtree := IxLxI_kmn_km( TL(16, 8, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
          SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      measured := 8,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(64, 8, 1, 1).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(64, 8, 1, 1).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_kmn_km( TL(32, 8, 1, 2).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_IxLxI_up( TL(16, 4, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
              IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
                IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
                SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ),
              IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
                IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
                SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ) ),
            IxLxI_kmn_n( TL(8, 2, 1, 8).withTags([ AVecReg(MACRO_8xf) ]),
              IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ),
              IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ) ) ),
          SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      origtree := IxLxI_kmn_n( TL(64, 8, 1, 1).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_kmn_km( TL(32, 8, 1, 2).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_IxLxI_up( TL(16, 4, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
              IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
                IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
                SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ),
              IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
                IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
                SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ) ),
            IxLxI_kmn_n( TL(8, 2, 1, 8).withTags([ AVecReg(MACRO_8xf) ]),
              IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ),
              IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ) ) ),
          SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      measured := 24,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 2).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 1, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 1, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
      measured := 0.90000000000000002,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
      origtree := IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 8, 2).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 8, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 8, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
      measured := 7.2000000000000002,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
      measured := 8,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 2, 2).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 4, 2, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 4, 2, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
      measured := 4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(64, 32, 1, 1).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(64, 32, 1, 1).withTags([ AVecReg(MACRO_8xf) ]),
          SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ),
          IxLxI_kmn_km( TL(8, 4, 1, 8).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ) ) ),
      origtree := IxLxI_kmn_km( TL(64, 32, 1, 1).withTags([ AVecReg(MACRO_8xf) ]),
          SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ),
          IxLxI_kmn_km( TL(8, 4, 1, 8).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ) ) ),
      measured := 8,
      globalUnrolling := 10000 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 8, 1, 1).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(16, 8, 1, 1).withTags([ AVecReg(MACRO_8xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(16, 8, 1, 1).withTags([ AVecReg(MACRO_8xf) ]) ),
      measured := 2,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(64, 4, 1, 1).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(64, 4, 1, 1).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_kmn_km( TL(8, 4, 1, 8).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ) ),
          IxLxI_kmn_km( TL(32, 4, 2, 1).withTags([ AVecReg(MACRO_8xf) ]),
            SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ),
            IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
              IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
              SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ) ) ),
      origtree := IxLxI_kmn_n( TL(64, 4, 1, 1).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_kmn_km( TL(8, 4, 1, 8).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ) ),
          IxLxI_kmn_km( TL(32, 4, 2, 1).withTags([ AVecReg(MACRO_8xf) ]),
            SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ),
            IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
              IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
              SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ) ) ),
      measured := 16,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(32, 8, 1, 2).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(32, 8, 1, 2).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_IxLxI_up( TL(16, 4, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
              IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
              SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ),
            IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
              IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
              SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ) ),
          IxLxI_kmn_n( TL(8, 2, 1, 8).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ) ) ),
      origtree := IxLxI_kmn_km( TL(32, 8, 1, 2).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_IxLxI_up( TL(16, 4, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
              IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
              SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ),
            IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
              IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
              SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ) ),
          IxLxI_kmn_n( TL(8, 2, 1, 8).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ) ) ),
      measured := 16,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(32, 8, 2, 1).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(32, 8, 2, 1).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_kmn_km( TL(16, 8, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
            SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ) ),
          SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      origtree := IxLxI_kmn_n( TL(32, 8, 2, 1).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_kmn_km( TL(16, 8, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
            SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ) ),
          SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      measured := 16,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(64, 2, 1, 1).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(64, 2, 1, 1).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_kmn_n( TL(8, 2, 1, 8).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ) ),
          SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      origtree := IxLxI_kmn_n( TL(64, 2, 1, 1).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_kmn_n( TL(8, 2, 1, 8).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ) ),
          SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      measured := 8,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 8, 1, 2).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(16, 8, 1, 2).withTags([ AVecReg(MACRO_8xf) ]),
          SIMD_ISA_Bases1( TL(8, 4, 2, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
          IxLxI_vtensor( TL(4, 2, 1, 8).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      origtree := IxLxI_kmn_km( TL(16, 8, 1, 2).withTags([ AVecReg(MACRO_8xf) ]),
          SIMD_ISA_Bases1( TL(8, 4, 2, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
          IxLxI_vtensor( TL(4, 2, 1, 8).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      measured := 4,
      globalUnrolling := 10000 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 1, 8).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(8, 4, 1, 8).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
          IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      origtree := IxLxI_kmn_km( TL(8, 4, 1, 8).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
          IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(32, 2, 1, 2).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(32, 2, 1, 2).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_kmn_n( TL(8, 2, 1, 8).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ) ),
          SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      origtree := IxLxI_kmn_n( TL(32, 2, 1, 2).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_kmn_n( TL(8, 2, 1, 8).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ) ),
          SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      measured := 8,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(32, 2, 2, 1).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(32, 2, 2, 1).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
          SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      origtree := IxLxI_kmn_n( TL(32, 2, 2, 1).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
          SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      measured := 8,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 1, 8).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(8, 2, 1, 8).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      origtree := IxLxI_kmn_n( TL(8, 2, 1, 8).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 1, 2).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 4, 1, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 4, 1, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
      measured := 2,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 8).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_vtensor( TL(4, 2, 1, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
      origtree := IxLxI_vtensor( TL(4, 2, 1, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 1, 2).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 2, 1, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 2, 1, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
      measured := 2,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 4, 4, 1).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_IxLxI_up( TL(16, 4, 4, 1).withTags([ AVecReg(MACRO_8xf) ]),
          SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ),
          SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      origtree := IxLxI_IxLxI_up( TL(16, 4, 4, 1).withTags([ AVecReg(MACRO_8xf) ]),
          SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ),
          SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      measured := 16,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(32, 16, 1, 2).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(32, 16, 1, 2).withTags([ AVecReg(MACRO_8xf) ]),
          SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
          IxLxI_kmn_km( TL(8, 4, 1, 8).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ) ) ),
      origtree := IxLxI_kmn_km( TL(32, 16, 1, 2).withTags([ AVecReg(MACRO_8xf) ]),
          SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
          IxLxI_kmn_km( TL(8, 4, 1, 8).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ) ) ),
      measured := 8,
      globalUnrolling := 10000 ) ]);
SIMD_ISA_DB.hashAdd(TL(32, 16, 2, 1).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(32, 16, 2, 1).withTags([ AVecReg(MACRO_8xf) ]),
          SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      origtree := IxLxI_kmn_km( TL(32, 16, 2, 1).withTags([ AVecReg(MACRO_8xf) ]),
          SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      measured := 8,
      globalUnrolling := 10000 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 4, 1, 1).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_IxLxI_up( TL(16, 4, 1, 1).withTags([ AVecReg(MACRO_8xf) ]),
          SIMD_ISA_Bases1( TL(16, 2, 1, 1).withTags([ AVecReg(MACRO_8xf) ]) ),
          SIMD_ISA_Bases1( TL(16, 2, 1, 1).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      origtree := IxLxI_IxLxI_up( TL(16, 4, 1, 1).withTags([ AVecReg(MACRO_8xf) ]),
          SIMD_ISA_Bases1( TL(16, 2, 1, 1).withTags([ AVecReg(MACRO_8xf) ]) ),
          SIMD_ISA_Bases1( TL(16, 2, 1, 1).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      measured := 4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 4, 2).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
      measured := 8,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
          SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      origtree := IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
          SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ),
      measured := 8,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 8, 4, 1).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ),
      measured := 8,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
      measured := 4,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
      measured := 1.8,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(32, 4, 1, 2).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(32, 4, 1, 2).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_kmn_km( TL(8, 4, 1, 8).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ) ),
          IxLxI_IxLxI_up( TL(16, 4, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
              IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
              SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ),
            IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
              IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
              SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ) ) ),
      origtree := IxLxI_kmn_n( TL(32, 4, 1, 2).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_kmn_km( TL(8, 4, 1, 8).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
            IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ) ),
          IxLxI_IxLxI_up( TL(16, 4, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
              IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
              SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ),
            IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
              IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
              SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ) ) ),
      measured := 16,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(32, 4, 2, 1).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(32, 4, 2, 1).withTags([ AVecReg(MACRO_8xf) ]),
          SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ),
          IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
            SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ) ),
      origtree := IxLxI_kmn_km( TL(32, 4, 2, 1).withTags([ AVecReg(MACRO_8xf) ]),
          SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ),
          IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
            SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ) ),
      measured := 16,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 2, 1, 1).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(16, 2, 1, 1).withTags([ AVecReg(MACRO_8xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(16, 2, 1, 1).withTags([ AVecReg(MACRO_8xf) ]) ),
      measured := 2,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(64, 16, 1, 1).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(64, 16, 1, 1).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_kmn_km( TL(32, 16, 1, 2).withTags([ AVecReg(MACRO_8xf) ]),
            SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
            IxLxI_kmn_km( TL(8, 4, 1, 8).withTags([ AVecReg(MACRO_8xf) ]),
              IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
              IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ) ) ),
          IxLxI_kmn_km( TL(32, 16, 2, 1).withTags([ AVecReg(MACRO_8xf) ]),
            SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ) ) ),
      origtree := IxLxI_kmn_n( TL(64, 16, 1, 1).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_kmn_km( TL(32, 16, 1, 2).withTags([ AVecReg(MACRO_8xf) ]),
            SIMD_ISA_Bases1( TL(8, 4, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
            IxLxI_kmn_km( TL(8, 4, 1, 8).withTags([ AVecReg(MACRO_8xf) ]),
              IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
              IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ) ) ),
          IxLxI_kmn_km( TL(32, 16, 2, 1).withTags([ AVecReg(MACRO_8xf) ]),
            SIMD_ISA_Bases1( TL(16, 8, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ) ) ),
      measured := 16,
      globalUnrolling := 10000 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 4, 2, 2).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_IxLxI_up( TL(16, 4, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
            SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ),
          IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
            SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ) ),
      origtree := IxLxI_IxLxI_up( TL(16, 4, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
            SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ),
          IxLxI_kmn_n( TL(16, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 2, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
            SIMD_ISA_Bases1( TL(8, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ) ),
      measured := 16,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 4, 2).withTags([ AVecReg(MACRO_8xf) ]) ),
      measured := 3.6000000000000001,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ),
      origtree := IxLxI_vtensor( TL(4, 2, 1, 16).withTags([ AVecReg(MACRO_8xf) ]) ),
      measured := 0,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 2, 4, 1).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ),
      origtree := SIMD_ISA_Bases1( TL(16, 2, 4, 1).withTags([ AVecReg(MACRO_8xf) ]) ),
      measured := 8,
      globalUnrolling := 8 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 4, 1, 2).withTags([ AVecReg(MACRO_8xf) ]), [ rec(
      ruletree := IxLxI_IxLxI_up( TL(16, 4, 1, 2).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_kmn_n( TL(16, 2, 1, 2).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 1, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
            SIMD_ISA_Bases1( TL(8, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ),
          IxLxI_kmn_n( TL(16, 2, 1, 2).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 1, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
            SIMD_ISA_Bases1( TL(8, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ) ),
      origtree := IxLxI_IxLxI_up( TL(16, 4, 1, 2).withTags([ AVecReg(MACRO_8xf) ]),
          IxLxI_kmn_n( TL(16, 2, 1, 2).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 1, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
            SIMD_ISA_Bases1( TL(8, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ),
          IxLxI_kmn_n( TL(16, 2, 1, 2).withTags([ AVecReg(MACRO_8xf) ]),
            IxLxI_vtensor( TL(4, 2, 1, 8).withTags([ AVecReg(MACRO_8xf) ]) ),
            SIMD_ISA_Bases1( TL(8, 2, 2, 2).withTags([ AVecReg(MACRO_8xf) ]) ) ) ),
      measured := 8,
      globalUnrolling := 8 ) ]);
