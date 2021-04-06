
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 1).withTags([ AVecRegCx(AltiVec_4x32f) ]), [ rec(
      ruletree := L_cx_real( TL(4, 2, 1, 1).withTags([ AVecRegCx(AltiVec_4x32f) ]),
          SIMD_ISA_Bases1( TL(4, 2, 1, 2).withTags([ AVecReg(AltiVec_4x32f) ]) ) ),
      origtree := L_cx_real( TL(4, 2, 1, 1).withTags([ AVecRegCx(AltiVec_4x32f) ]),
          SIMD_ISA_Bases1( TL(4, 2, 1, 2).withTags([ AVecReg(AltiVec_4x32f) ]) ) ),
      measured := 0,
      language := "c.icl.opt",
      globalUnrolling := 32 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 2, 1).withTags([ AVecReg(AltiVec_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 2, 1).withTags([ AVecReg(AltiVec_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 2, 1).withTags([ AVecReg(AltiVec_4x32f) ]) ),
      measured := 0,
      language := "c.icl.opt",
      globalUnrolling := 32 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 2).withTags([ AVecReg(AltiVec_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 1, 2).withTags([ AVecReg(AltiVec_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 1, 2).withTags([ AVecReg(AltiVec_4x32f) ]) ),
      measured := 0,
      language := "c.icl.opt",
      globalUnrolling := 32 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 1, 4).withTags([ AVecReg(AltiVec_4x32f) ]), [ rec(
      ruletree := IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(AltiVec_4x32f) ]) ),
      origtree := IxLxI_vtensor( TL(4, 2, 1, 4).withTags([ AVecReg(AltiVec_4x32f) ]) ),
      measured := 0,
      language := "c.icl.opt",
      globalUnrolling := 32 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 4, 1).withTags([ AVecReg(AltiVec_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(AltiVec_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(AltiVec_4x32f) ]) ),
      measured := 0,
      language := "c.icl.opt",
      globalUnrolling := 32 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 8, 1, 1).withTags([ AVecReg(AltiVec_4x32f) ]), [ rec(
      ruletree := IxLxI_kmn_km( TL(16, 8, 1, 1).withTags([ AVecReg(AltiVec_4x32f) ]),
          SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(AltiVec_4x32f) ]) ),
          SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(AltiVec_4x32f) ]) ) ),
      origtree := IxLxI_kmn_km( TL(16, 8, 1, 1).withTags([ AVecReg(AltiVec_4x32f) ]),
          SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(AltiVec_4x32f) ]) ),
          SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(AltiVec_4x32f) ]) ) ),
      measured := 0,
      language := "c.icl.opt",
      globalUnrolling := 32 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 1, 1).withTags([ AVecReg(AltiVec_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 4, 1, 1).withTags([ AVecReg(AltiVec_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 4, 1, 1).withTags([ AVecReg(AltiVec_4x32f) ]) ),
      measured := 0,
      language := "c.icl.opt",
      globalUnrolling := 32 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 1, 2).withTags([ AVecReg(AltiVec_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(AltiVec_4x32f) ]) ),
      origtree := SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(AltiVec_4x32f) ]) ),
      measured := 0,
      language := "c.icl.opt",
      globalUnrolling := 32 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 4, 2, 1).withTags([ AVecReg(AltiVec_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(AltiVec_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(AltiVec_4x32f) ]) ),
      measured := 0,
      language := "c.icl.opt",
      globalUnrolling := 32 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 1, 2).withTags([ AVecReg(AltiVec_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases2( TL(8, 2, 1, 2).withTags([ AVecReg(AltiVec_4x32f) ]) ),
      origtree := SIMD_ISA_Bases2( TL(8, 2, 1, 2).withTags([ AVecReg(AltiVec_4x32f) ]) ),
      measured := 0,
      language := "c.icl.opt",
      globalUnrolling := 32 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 2, 1).withTags([ AVecReg(AltiVec_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 2, 2, 1).withTags([ AVecReg(AltiVec_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 2, 2, 1).withTags([ AVecReg(AltiVec_4x32f) ]) ),
      measured := 0,
      language := "c.icl.opt",
      globalUnrolling := 32 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 4, 1, 1).withTags([ AVecReg(AltiVec_4x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(16, 4, 1, 1).withTags([ AVecReg(AltiVec_4x32f) ]),
          SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(AltiVec_4x32f) ]) ),
          SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(AltiVec_4x32f) ]) ) ),
      origtree := IxLxI_kmn_n( TL(16, 4, 1, 1).withTags([ AVecReg(AltiVec_4x32f) ]),
          SIMD_ISA_Bases2( TL(8, 4, 1, 2).withTags([ AVecReg(AltiVec_4x32f) ]) ),
          SIMD_ISA_Bases1( TL(8, 4, 2, 1).withTags([ AVecReg(AltiVec_4x32f) ]) ) ),
      measured := 0,
      language := "c.icl.opt",
      globalUnrolling := 32 ) ]);
SIMD_ISA_DB.hashAdd(TL(4, 2, 2, 2).withTags([ AVecReg(AltiVec_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(4, 2, 2, 2).withTags([ AVecReg(AltiVec_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(4, 2, 2, 2).withTags([ AVecReg(AltiVec_4x32f) ]) ),
      measured := 0,
      language := "c.icl.opt",
      globalUnrolling := 32 ) ]);
SIMD_ISA_DB.hashAdd(TL(16, 2, 1, 1).withTags([ AVecReg(AltiVec_4x32f) ]), [ rec(
      ruletree := IxLxI_kmn_n( TL(16, 2, 1, 1).withTags([ AVecReg(AltiVec_4x32f) ]),
          SIMD_ISA_Bases2( TL(8, 2, 1, 2).withTags([ AVecReg(AltiVec_4x32f) ]) ),
          SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(AltiVec_4x32f) ]) ) ),
      origtree := IxLxI_kmn_n( TL(16, 2, 1, 1).withTags([ AVecReg(AltiVec_4x32f) ]),
          SIMD_ISA_Bases2( TL(8, 2, 1, 2).withTags([ AVecReg(AltiVec_4x32f) ]) ),
          SIMD_ISA_Bases1( TL(4, 2, 4, 1).withTags([ AVecReg(AltiVec_4x32f) ]) ) ),
      measured := 0,
      language := "c.icl.opt",
      globalUnrolling := 32 ) ]);
SIMD_ISA_DB.hashAdd(TL(8, 2, 1, 1).withTags([ AVecReg(AltiVec_4x32f) ]), [ rec(
      ruletree := SIMD_ISA_Bases1( TL(8, 2, 1, 1).withTags([ AVecReg(AltiVec_4x32f) ]) ),
      origtree := SIMD_ISA_Bases1( TL(8, 2, 1, 1).withTags([ AVecReg(AltiVec_4x32f) ]) ),
      measured := 0,
      language := "c.icl.opt",
      globalUnrolling := 32 ) ]);
