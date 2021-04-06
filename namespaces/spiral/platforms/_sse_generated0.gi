
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

SSE_2x64f.setRules(rec(
  binrules := [ rec(
          perm := rec(
              spl := L(4, 2),
              l := 1,
              N := 4,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vunpacklo_2x64f,
                  p := [  ] ), rec(
                  instr := vunpackhi_2x64f,
                  p := [  ] ) ],
          v := 2,
          vperm := VPerm(L(4, 2), (y, x) -> chain(assign(vref(y, 0, 2), vunpacklo_2x64f(vref(x, 0, 2), vref(x, 2, 2), [  ])), assign(vref(y, 2, 2), vunpackhi_2x64f(vref(x, 0, 2), vref(x, 2, 2), [  ]))), 2, 2) ) ],
  unrules := [  ],
  x_I_vby2 := [ [ rec(
              instr := vunpacklo_2x64f,
              p := [  ] ), rec(
              instr := vshuffle_2x64f,
              p := [ 1, 2 ] ) ], [ rec(
              instr := vshuffle_2x64f,
              p := [ 2, 1 ] ), rec(
              instr := vunpackhi_2x64f,
              p := [  ] ) ] ] ));
SSE_2x64i.setRules(rec(
  binrules := [ rec(
          perm := rec(
              spl := L(4, 2),
              l := 1,
              N := 4,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vunpacklo_2x64i,
                  p := [  ] ), rec(
                  instr := vunpackhi_2x64i,
                  p := [  ] ) ],
          v := 2,
          vperm := VPerm(L(4, 2), (y, x) -> chain(assign(vref(y, 0, 2), vunpacklo_2x64i(vref(x, 0, 2), vref(x, 2, 2), [  ])), assign(vref(y, 2, 2), vunpackhi_2x64i(vref(x, 0, 2), vref(x, 2, 2), [  ]))), 2, 2) ) ],
  unrules := [  ],
  x_I_vby2 := [ [ rec(
              instr := vunpacklo_2x64i,
              p := [  ] ), rec(
              instr := vshuffle_2x64i,
              p := [ 1, 2 ] ) ], [ rec(
              instr := vshuffle_2x64i,
              p := [ 2, 1 ] ), rec(
              instr := vunpackhi_2x64i,
              p := [  ] ) ] ] ));
SSE_2x32f.setRules(rec(
  binrules := [ rec(
          perm := rec(
              spl := L(4, 2),
              l := 1,
              N := 4,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vunpacklo_2x32f,
                  p := [  ] ), rec(
                  instr := vunpackhi_2x32f,
                  p := [  ] ) ],
          v := 2,
          vperm := VPerm(L(4, 2), (y, x) -> chain(assign(vref(y, 0, 2), vunpacklo_2x32f(vref(x, 0, 2), vref(x, 2, 2), [  ])), assign(vref(y, 2, 2), vunpackhi_2x32f(vref(x, 0, 2), vref(x, 2, 2), [  ]))), 2, 2) ) ],
  unrules := [  ],
  x_I_vby2 := [ [ rec(
              instr := vunpacklo_2x32f,
              p := [  ] ), rec(
              instr := vshuffle_2x32f,
              p := [ 1, 2 ] ) ], [ rec(
              instr := vshuffle_2x32f,
              p := [ 2, 1 ] ), rec(
              instr := vunpackhi_2x32f,
              p := [  ] ) ] ] ));
SSE_4x32f.setRules(rec(
  binrules := [ rec(
          perm := rec(
              spl := L(8, 4),
              l := 1,
              N := 8,
              n := 4,
              r := 1 ),
          instr := [ rec(
                  instr := vunpacklo_4x32f,
                  p := [  ] ), rec(
                  instr := vunpackhi_4x32f,
                  p := [  ] ) ],
          v := 4,
          vperm := VPerm(L(8, 4), (y, x) -> chain(assign(vref(y, 0, 4), vunpacklo_4x32f(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vunpackhi_4x32f(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 2) ), rec(
          perm := rec(
              spl := Tensor(
                  L(4, 2), 
                  I(2)
                ),
              l := 1,
              N := 4,
              n := 2,
              r := 2 ),
          instr := [ rec(
                  instr := vshuffle_4x32f,
                  p := [ 1, 2, 1, 2 ] ), rec(
                  instr := vshuffle_4x32f,
                  p := [ 3, 4, 3, 4 ] ) ],
          v := 4,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vshuffle_4x32f(vref(x, 0, 4), vref(x, 4, 4), [ 1, 2, 1, 2 ])), assign(vref(y, 4, 4), vshuffle_4x32f(vref(x, 0, 4), vref(x, 4, 4), [ 3, 4, 3, 4 ]))), 4, 2) ), rec(
          perm := rec(
              spl := L(8, 2),
              l := 1,
              N := 8,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vshuffle_4x32f,
                  p := [ 1, 3, 1, 3 ] ), rec(
                  instr := vshuffle_4x32f,
                  p := [ 2, 4, 2, 4 ] ) ],
          v := 4,
          vperm := VPerm(L(8, 2), (y, x) -> chain(assign(vref(y, 0, 4), vshuffle_4x32f(vref(x, 0, 4), vref(x, 4, 4), [ 1, 3, 1, 3 ])), assign(vref(y, 4, 4), vshuffle_4x32f(vref(x, 0, 4), vref(x, 4, 4), [ 2, 4, 2, 4 ]))), 4, 2) ), rec(
          perm := rec(
              spl := Tensor(
                  I(2), 
                  L(4, 2)
                ),
              l := 2,
              N := 4,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vshuffle_4x32f,
                  p := [ 1, 3, 1, 3 ] ), rec(
                  instr := vshuffle_4x32f,
                  p := [ 2, 4, 2, 4 ] ) ],
          v := 4,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vshuffle_4x32f(vref(x, 0, 4), vref(x, 4, 4), [ 1, 2, 1, 2 ])), assign(vref(y, 4, 4), vshuffle_4x32f(vref(x, 0, 4), vref(x, 4, 4), [ 3, 4, 3, 4 ]))), 4, 2) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ) * 
              Tensor(
                I(2), 
                L(4, 2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vshuffle_4x32f(vref(x, 0, 4), vref(x, 4, 4), [ 1, 3, 1, 3 ])), assign(vref(y, 4, 4), vshuffle_4x32f(vref(x, 0, 4), vref(x, 4, 4), [ 2, 4, 2, 4 ]))), 4, 4.2000000000000002) * 
            VTensor(I(2), 4) ), rec(
          perm := rec(
              spl := Tensor(
                  I(2), 
                  L(4, 2)
                ),
              l := 2,
              N := 4,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vunpacklo_4x32f,
                  p := [  ] ), rec(
                  instr := vunpackhi_4x32f,
                  p := [  ] ) ],
          v := 4,
          vperm := VTensor(I(2), 4) * 
            VPerm(Tensor(
                I(2), 
                L(4, 2)
              ) * 
              Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vunpacklo_4x32f(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vunpackhi_4x32f(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 4.2000000000000002) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vshuffle_4x32f(vref(x, 0, 4), vref(x, 4, 4), [ 1, 2, 1, 2 ])), assign(vref(y, 4, 4), vshuffle_4x32f(vref(x, 0, 4), vref(x, 4, 4), [ 3, 4, 3, 4 ]))), 4, 2) ) ],
  unrules := [ rec(
          perm := rec(
              spl := L(4, 2),
              l := 1,
              N := 4,
              n := 2,
              r := 1 ),
          instr := rec(
              instr := vushuffle_4x32f,
              p := [ 1, 3, 2, 4 ] ),
          v := 4,
          vperm := VPerm(L(4, 2), (y, x) -> assign(vref(y, 0, 4), vushuffle_4x32f(vref(x, 0, 4), [ 1, 3, 2, 4 ])), 4, 0.90000000000000002) ) ],
  x_I_vby2 := [ [ rec(
              instr := vshuffle_4x32f,
              p := [ 1, 2, 1, 2 ] ), rec(
              instr := vshuffle_4x32f,
              p := [ 1, 2, 3, 4 ] ) ], [ rec(
              instr := vshuffle_4x32f,
              p := [ 3, 4, 1, 2 ] ), rec(
              instr := vshuffle_4x32f,
              p := [ 3, 4, 3, 4 ] ) ] ] ));
SSE_4x32i.setRules(rec(
  binrules := [ rec(
          perm := rec(
              spl := L(8, 4),
              l := 1,
              N := 8,
              n := 4,
              r := 1 ),
          instr := [ rec(
                  instr := vunpacklo_4x32i,
                  p := [  ] ), rec(
                  instr := vunpackhi_4x32i,
                  p := [  ] ) ],
          v := 4,
          vperm := VPerm(L(8, 4), (y, x) -> chain(assign(vref(y, 0, 4), vunpacklo_4x32i(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vunpackhi_4x32i(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 2) ), rec(
          perm := rec(
              spl := Tensor(
                  L(4, 2), 
                  I(2)
                ),
              l := 1,
              N := 4,
              n := 2,
              r := 2 ),
          instr := [ rec(
                  instr := vshuffle_4x32i,
                  p := [ 1, 2, 1, 2 ] ), rec(
                  instr := vshuffle_4x32i,
                  p := [ 3, 4, 3, 4 ] ) ],
          v := 4,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vshuffle_4x32i(vref(x, 0, 4), vref(x, 4, 4), [ 1, 2, 1, 2 ])), assign(vref(y, 4, 4), vshuffle_4x32i(vref(x, 0, 4), vref(x, 4, 4), [ 3, 4, 3, 4 ]))), 4, 2) ), rec(
          perm := rec(
              spl := L(8, 2),
              l := 1,
              N := 8,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vshuffle_4x32i,
                  p := [ 1, 3, 1, 3 ] ), rec(
                  instr := vshuffle_4x32i,
                  p := [ 2, 4, 2, 4 ] ) ],
          v := 4,
          vperm := VPerm(L(8, 2), (y, x) -> chain(assign(vref(y, 0, 4), vshuffle_4x32i(vref(x, 0, 4), vref(x, 4, 4), [ 1, 3, 1, 3 ])), assign(vref(y, 4, 4), vshuffle_4x32i(vref(x, 0, 4), vref(x, 4, 4), [ 2, 4, 2, 4 ]))), 4, 2) ), rec(
          perm := rec(
              spl := Tensor(
                  I(2), 
                  L(4, 2)
                ),
              l := 2,
              N := 4,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vshuffle_4x32i,
                  p := [ 1, 3, 1, 3 ] ), rec(
                  instr := vshuffle_4x32i,
                  p := [ 2, 4, 2, 4 ] ) ],
          v := 4,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vshuffle_4x32i(vref(x, 0, 4), vref(x, 4, 4), [ 1, 2, 1, 2 ])), assign(vref(y, 4, 4), vshuffle_4x32i(vref(x, 0, 4), vref(x, 4, 4), [ 3, 4, 3, 4 ]))), 4, 2) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ) * 
              Tensor(
                I(2), 
                L(4, 2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vshuffle_4x32i(vref(x, 0, 4), vref(x, 4, 4), [ 1, 3, 1, 3 ])), assign(vref(y, 4, 4), vshuffle_4x32i(vref(x, 0, 4), vref(x, 4, 4), [ 2, 4, 2, 4 ]))), 4, 4.2000000000000002) * 
            VTensor(I(2), 4) ), rec(
          perm := rec(
              spl := Tensor(
                  I(2), 
                  L(4, 2)
                ),
              l := 2,
              N := 4,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vunpacklo_4x32i,
                  p := [  ] ), rec(
                  instr := vunpackhi_4x32i,
                  p := [  ] ) ],
          v := 4,
          vperm := VTensor(I(2), 4) * 
            VPerm(Tensor(
                I(2), 
                L(4, 2)
              ) * 
              Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vunpacklo_4x32i(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vunpackhi_4x32i(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 4.2000000000000002) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vshuffle_4x32i(vref(x, 0, 4), vref(x, 4, 4), [ 1, 2, 1, 2 ])), assign(vref(y, 4, 4), vshuffle_4x32i(vref(x, 0, 4), vref(x, 4, 4), [ 3, 4, 3, 4 ]))), 4, 2) ) ],
  unrules := [ rec(
          perm := rec(
              spl := L(4, 2),
              l := 1,
              N := 4,
              n := 2,
              r := 1 ),
          instr := rec(
              instr := vushuffle_4x32i,
              p := [ 1, 3, 2, 4 ] ),
          v := 4,
          vperm := VPerm(L(4, 2), (y, x) -> assign(vref(y, 0, 4), vushuffle_4x32i(vref(x, 0, 4), [ 1, 3, 2, 4 ])), 4, 0.90000000000000002) ) ],
  x_I_vby2 := [ [ rec(
              instr := vshuffle_4x32i,
              p := [ 1, 2, 1, 2 ] ), rec(
              instr := vshuffle_4x32i,
              p := [ 1, 2, 3, 4 ] ) ], [ rec(
              instr := vshuffle_4x32i,
              p := [ 3, 4, 1, 2 ] ), rec(
              instr := vshuffle_4x32i,
              p := [ 3, 4, 3, 4 ] ) ] ] ));
SSE_8x16i.setRules(rec(
  binrules := [ rec(
          perm := rec(
              spl := L(16, 8),
              l := 1,
              N := 16,
              n := 8,
              r := 1 ),
          instr := [ rec(
                  instr := vunpacklo_8x16i,
                  p := [  ] ), rec(
                  instr := vunpackhi_8x16i,
                  p := [  ] ) ],
          v := 8,
          vperm := VPerm(L(16, 8), (y, x) -> chain(assign(vref(y, 0, 8), vunpacklo_8x16i(vref(x, 0, 8), vref(x, 8, 8), [  ])), assign(vref(y, 8, 8), vunpackhi_8x16i(vref(x, 0, 8), vref(x, 8, 8), [  ]))), 8, 2) ), rec(
          perm := rec(
              spl := Tensor(
                  L(8, 4), 
                  I(2)
                ),
              l := 1,
              N := 8,
              n := 4,
              r := 2 ),
          instr := [ rec(
                  instr := vunpacklo2_8x16i,
                  p := [  ] ), rec(
                  instr := vunpackhi2_8x16i,
                  p := [  ] ) ],
          v := 8,
          vperm := VPerm(Tensor(
                L(8, 4), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 8), vunpacklo2_8x16i(vref(x, 0, 8), vref(x, 8, 8), [  ])), assign(vref(y, 8, 8), vunpackhi2_8x16i(vref(x, 0, 8), vref(x, 8, 8), [  ]))), 8, 2) ), rec(
          perm := rec(
              spl := Tensor(
                  L(4, 2), 
                  I(4)
                ),
              l := 1,
              N := 4,
              n := 2,
              r := 4 ),
          instr := [ rec(
                  instr := vunpacklo4_8x16i,
                  p := [  ] ), rec(
                  instr := vunpackhi4_8x16i,
                  p := [  ] ) ],
          v := 8,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(4)
              ), (y, x) -> chain(assign(vref(y, 0, 8), vunpacklo4_8x16i(vref(x, 0, 8), vref(x, 8, 8), [  ])), assign(vref(y, 8, 8), vunpackhi4_8x16i(vref(x, 0, 8), vref(x, 8, 8), [  ]))), 8, 2) ), rec(
          perm := rec(
              spl := Tensor(
                  L(8, 2), 
                  I(2)
                ),
              l := 1,
              N := 8,
              n := 2,
              r := 2 ),
          instr := [ rec(
                  instr := vshuffle2_8x16i,
                  p := [ 1, 3, 1, 3 ] ), rec(
                  instr := vshuffle2_8x16i,
                  p := [ 2, 4, 2, 4 ] ) ],
          v := 8,
          vperm := VPerm(Tensor(
                L(8, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 8), vshuffle2_8x16i(vref(x, 0, 8), vref(x, 8, 8), [ 1, 3, 1, 3 ])), assign(vref(y, 8, 8), vshuffle2_8x16i(vref(x, 0, 8), vref(x, 8, 8), [ 2, 4, 2, 4 ]))), 8, 2) ), rec(
          perm := rec(
              spl := Tensor(
                  I(2), 
                  L(4, 2), 
                  I(2)
                ),
              l := 2,
              N := 4,
              n := 2,
              r := 2 ),
          instr := [ rec(
                  instr := vshuffle2_8x16i,
                  p := [ 1, 3, 1, 3 ] ), rec(
                  instr := vshuffle2_8x16i,
                  p := [ 2, 4, 2, 4 ] ) ],
          v := 8,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(4)
              ), (y, x) -> chain(assign(vref(y, 0, 8), vunpacklo4_8x16i(vref(x, 0, 8), vref(x, 8, 8), [  ])), assign(vref(y, 8, 8), vunpackhi4_8x16i(vref(x, 0, 8), vref(x, 8, 8), [  ]))), 8, 2) * 
            VPerm(Tensor(
                L(4, 2), 
                I(4)
              ) * 
              Tensor(
                I(2), 
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 8), vshuffle2_8x16i(vref(x, 0, 8), vref(x, 8, 8), [ 1, 3, 1, 3 ])), assign(vref(y, 8, 8), vshuffle2_8x16i(vref(x, 0, 8), vref(x, 8, 8), [ 2, 4, 2, 4 ]))), 8, 4.2000000000000002) * 
            VTensor(I(2), 8) ), rec(
          perm := rec(
              spl := Tensor(
                  I(2), 
                  L(8, 4)
                ),
              l := 2,
              N := 8,
              n := 4,
              r := 1 ),
          instr := [ rec(
                  instr := vunpacklo_8x16i,
                  p := [  ] ), rec(
                  instr := vunpackhi_8x16i,
                  p := [  ] ) ],
          v := 8,
          vperm := VTensor(I(2), 8) * 
            VPerm(Tensor(
                I(2), 
                L(8, 4)
              ) * 
              Tensor(
                L(4, 2), 
                I(4)
              ), (y, x) -> chain(assign(vref(y, 0, 8), vunpacklo_8x16i(vref(x, 0, 8), vref(x, 8, 8), [  ])), assign(vref(y, 8, 8), vunpackhi_8x16i(vref(x, 0, 8), vref(x, 8, 8), [  ]))), 8, 4.2000000000000002) * 
            VPerm(Tensor(
                L(4, 2), 
                I(4)
              ), (y, x) -> chain(assign(vref(y, 0, 8), vunpacklo4_8x16i(vref(x, 0, 8), vref(x, 8, 8), [  ])), assign(vref(y, 8, 8), vunpackhi4_8x16i(vref(x, 0, 8), vref(x, 8, 8), [  ]))), 8, 2) ), rec(
          perm := rec(
              spl := Tensor(
                  I(2), 
                  L(4, 2), 
                  I(2)
                ),
              l := 2,
              N := 4,
              n := 2,
              r := 2 ),
          instr := [ rec(
                  instr := vunpacklo2_8x16i,
                  p := [  ] ), rec(
                  instr := vunpackhi2_8x16i,
                  p := [  ] ) ],
          v := 8,
          vperm := VTensor(I(2), 8) * 
            VPerm(Tensor(
                I(2), 
                L(4, 2), 
                I(2)
              ) * 
              Tensor(
                L(4, 2), 
                I(4)
              ), (y, x) -> chain(assign(vref(y, 0, 8), vunpacklo2_8x16i(vref(x, 0, 8), vref(x, 8, 8), [  ])), assign(vref(y, 8, 8), vunpackhi2_8x16i(vref(x, 0, 8), vref(x, 8, 8), [  ]))), 8, 4.2000000000000002) * 
            VPerm(Tensor(
                L(4, 2), 
                I(4)
              ), (y, x) -> chain(assign(vref(y, 0, 8), vunpacklo4_8x16i(vref(x, 0, 8), vref(x, 8, 8), [  ])), assign(vref(y, 8, 8), vunpackhi4_8x16i(vref(x, 0, 8), vref(x, 8, 8), [  ]))), 8, 2) ) ],
  unrules := [ rec(
          perm := rec(
              spl := Tensor(
                  L(4, 2), 
                  I(2)
                ),
              l := 1,
              N := 4,
              n := 2,
              r := 2 ),
          instr := rec(
              instr := vushuffle2_8x16i,
              p := [ 1, 3, 2, 4 ] ),
          v := 8,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> assign(vref(y, 0, 8), vushuffle2_8x16i(vref(x, 0, 8), [ 1, 3, 2, 4 ])), 8, 0.90000000000000002) ), rec(
          perm := rec(
              spl := Tensor(
                  I(2), 
                  L(4, 2)
                ),
              l := 2,
              N := 4,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vushufflelo_8x16i,
                  p := [ 1, 3, 2, 4 ] ), rec(
                  instr := vushufflehi_8x16i,
                  p := [ 1, 3, 2, 4 ] ) ],
          v := 8,
          vperm := VPerm(Tensor(
                I(2), 
                L(4, 2)
              ), (y, x) -> Let(t => TempVec(TArray(x.t.t, Eval(v))),
               decl(t, chain(assign(vref(t, 0, Eval(v)), Eval(i1)(vref(x, 0, Eval(v)), Eval(op1.p))), assign(vref(y, 0, Eval(v)), Eval(i2)(vref(t, 0, Eval(v)), Eval(op2.p))))) )
             , 8, 1.8) ) ],
  x_I_vby2 := [ [ rec(
              instr := vunpacklo4_8x16i,
              p := [  ] ), rec(
              instr := vshuffle2_8x16i,
              p := [ 1, 2, 3, 4 ] ) ], [ rec(
              instr := vshuffle2_8x16i,
              p := [ 3, 4, 1, 2 ] ), rec(
              instr := vunpackhi4_8x16i,
              p := [  ] ) ] ] ));
SSE_16x8i.setRules(rec(
  binrules := [ rec(
          perm := rec(
              spl := L(32, 16),
              l := 1,
              N := 32,
              n := 16,
              r := 1 ),
          instr := [ rec(
                  instr := vunpacklo_16x8i,
                  p := [  ] ), rec(
                  instr := vunpackhi_16x8i,
                  p := [  ] ) ],
          v := 16,
          vperm := VPerm(L(32, 16), (y, x) -> chain(assign(vref(y, 0, 16), vunpacklo_16x8i(vref(x, 0, 16), vref(x, 16, 16), [  ])), assign(vref(y, 16, 16), vunpackhi_16x8i(vref(x, 0, 16), vref(x, 16, 16), [  ]))), 16, 2) ), rec(
          perm := rec(
              spl := Tensor(
                  L(16, 8), 
                  I(2)
                ),
              l := 1,
              N := 16,
              n := 8,
              r := 2 ),
          instr := [ rec(
                  instr := vunpacklo2_16x8i,
                  p := [  ] ), rec(
                  instr := vunpackhi2_16x8i,
                  p := [  ] ) ],
          v := 16,
          vperm := VPerm(Tensor(
                L(16, 8), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 16), vunpacklo2_16x8i(vref(x, 0, 16), vref(x, 16, 16), [  ])), assign(vref(y, 16, 16), vunpackhi2_16x8i(vref(x, 0, 16), vref(x, 16, 16), [  ]))), 16, 2) ), rec(
          perm := rec(
              spl := Tensor(
                  L(8, 4), 
                  I(4)
                ),
              l := 1,
              N := 8,
              n := 4,
              r := 4 ),
          instr := [ rec(
                  instr := vunpacklo4_16x8i,
                  p := [  ] ), rec(
                  instr := vunpackhi4_16x8i,
                  p := [  ] ) ],
          v := 16,
          vperm := VPerm(Tensor(
                L(8, 4), 
                I(4)
              ), (y, x) -> chain(assign(vref(y, 0, 16), vunpacklo4_16x8i(vref(x, 0, 16), vref(x, 16, 16), [  ])), assign(vref(y, 16, 16), vunpackhi4_16x8i(vref(x, 0, 16), vref(x, 16, 16), [  ]))), 16, 2) ), rec(
          perm := rec(
              spl := Tensor(
                  L(4, 2), 
                  I(8)
                ),
              l := 1,
              N := 4,
              n := 2,
              r := 8 ),
          instr := [ rec(
                  instr := vunpacklo8_16x8i,
                  p := [  ] ), rec(
                  instr := vunpackhi8_16x8i,
                  p := [  ] ) ],
          v := 16,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(8)
              ), (y, x) -> chain(assign(vref(y, 0, 16), vunpacklo8_16x8i(vref(x, 0, 16), vref(x, 16, 16), [  ])), assign(vref(y, 16, 16), vunpackhi8_16x8i(vref(x, 0, 16), vref(x, 16, 16), [  ]))), 16, 2) ), rec(
          perm := rec(
              spl := Tensor(
                  L(8, 2), 
                  I(4)
                ),
              l := 1,
              N := 8,
              n := 2,
              r := 4 ),
          instr := [ rec(
                  instr := vshuffle4_16x8i,
                  p := [ 1, 3, 1, 3 ] ), rec(
                  instr := vshuffle4_16x8i,
                  p := [ 2, 4, 2, 4 ] ) ],
          v := 16,
          vperm := VPerm(Tensor(
                L(8, 2), 
                I(4)
              ), (y, x) -> chain(assign(vref(y, 0, 16), vshuffle4_16x8i(vref(x, 0, 16), vref(x, 16, 16), [ 1, 3, 1, 3 ])), assign(vref(y, 16, 16), vshuffle4_16x8i(vref(x, 0, 16), vref(x, 16, 16), [ 2, 4, 2, 4 ]))), 16, 2) ), rec(
          perm := rec(
              spl := Tensor(
                  I(2), 
                  L(4, 2), 
                  I(4)
                ),
              l := 2,
              N := 4,
              n := 2,
              r := 4 ),
          instr := [ rec(
                  instr := vshuffle4_16x8i,
                  p := [ 1, 3, 1, 3 ] ), rec(
                  instr := vshuffle4_16x8i,
                  p := [ 2, 4, 2, 4 ] ) ],
          v := 16,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(8)
              ), (y, x) -> chain(assign(vref(y, 0, 16), vunpacklo8_16x8i(vref(x, 0, 16), vref(x, 16, 16), [  ])), assign(vref(y, 16, 16), vunpackhi8_16x8i(vref(x, 0, 16), vref(x, 16, 16), [  ]))), 16, 2) * 
            VPerm(Tensor(
                L(4, 2), 
                I(8)
              ) * 
              Tensor(
                I(2), 
                L(4, 2), 
                I(4)
              ), (y, x) -> chain(assign(vref(y, 0, 16), vshuffle4_16x8i(vref(x, 0, 16), vref(x, 16, 16), [ 1, 3, 1, 3 ])), assign(vref(y, 16, 16), vshuffle4_16x8i(vref(x, 0, 16), vref(x, 16, 16), [ 2, 4, 2, 4 ]))), 16, 4.2000000000000002) * 
            VTensor(I(2), 16) ), rec(
          perm := rec(
              spl := Tensor(
                  I(2), 
                  L(16, 8)
                ),
              l := 2,
              N := 16,
              n := 8,
              r := 1 ),
          instr := [ rec(
                  instr := vunpacklo_16x8i,
                  p := [  ] ), rec(
                  instr := vunpackhi_16x8i,
                  p := [  ] ) ],
          v := 16,
          vperm := VTensor(I(2), 16) * 
            VPerm(Tensor(
                I(2), 
                L(16, 8)
              ) * 
              Tensor(
                L(4, 2), 
                I(8)
              ), (y, x) -> chain(assign(vref(y, 0, 16), vunpacklo_16x8i(vref(x, 0, 16), vref(x, 16, 16), [  ])), assign(vref(y, 16, 16), vunpackhi_16x8i(vref(x, 0, 16), vref(x, 16, 16), [  ]))), 16, 4.2000000000000002) * 
            VPerm(Tensor(
                L(4, 2), 
                I(8)
              ), (y, x) -> chain(assign(vref(y, 0, 16), vunpacklo8_16x8i(vref(x, 0, 16), vref(x, 16, 16), [  ])), assign(vref(y, 16, 16), vunpackhi8_16x8i(vref(x, 0, 16), vref(x, 16, 16), [  ]))), 16, 2) ), rec(
          perm := rec(
              spl := Tensor(
                  I(2), 
                  L(8, 4), 
                  I(2)
                ),
              l := 2,
              N := 8,
              n := 4,
              r := 2 ),
          instr := [ rec(
                  instr := vunpacklo2_16x8i,
                  p := [  ] ), rec(
                  instr := vunpackhi2_16x8i,
                  p := [  ] ) ],
          v := 16,
          vperm := VTensor(I(2), 16) * 
            VPerm(Tensor(
                I(2), 
                L(8, 4), 
                I(2)
              ) * 
              Tensor(
                L(4, 2), 
                I(8)
              ), (y, x) -> chain(assign(vref(y, 0, 16), vunpacklo2_16x8i(vref(x, 0, 16), vref(x, 16, 16), [  ])), assign(vref(y, 16, 16), vunpackhi2_16x8i(vref(x, 0, 16), vref(x, 16, 16), [  ]))), 16, 4.2000000000000002) * 
            VPerm(Tensor(
                L(4, 2), 
                I(8)
              ), (y, x) -> chain(assign(vref(y, 0, 16), vunpacklo8_16x8i(vref(x, 0, 16), vref(x, 16, 16), [  ])), assign(vref(y, 16, 16), vunpackhi8_16x8i(vref(x, 0, 16), vref(x, 16, 16), [  ]))), 16, 2) ), rec(
          perm := rec(
              spl := Tensor(
                  I(2), 
                  L(4, 2), 
                  I(4)
                ),
              l := 2,
              N := 4,
              n := 2,
              r := 4 ),
          instr := [ rec(
                  instr := vunpacklo4_16x8i,
                  p := [  ] ), rec(
                  instr := vunpackhi4_16x8i,
                  p := [  ] ) ],
          v := 16,
          vperm := VTensor(I(2), 16) * 
            VPerm(Tensor(
                I(2), 
                L(4, 2), 
                I(4)
              ) * 
              Tensor(
                L(4, 2), 
                I(8)
              ), (y, x) -> chain(assign(vref(y, 0, 16), vunpacklo4_16x8i(vref(x, 0, 16), vref(x, 16, 16), [  ])), assign(vref(y, 16, 16), vunpackhi4_16x8i(vref(x, 0, 16), vref(x, 16, 16), [  ]))), 16, 4.2000000000000002) * 
            VPerm(Tensor(
                L(4, 2), 
                I(8)
              ), (y, x) -> chain(assign(vref(y, 0, 16), vunpacklo8_16x8i(vref(x, 0, 16), vref(x, 16, 16), [  ])), assign(vref(y, 16, 16), vunpackhi8_16x8i(vref(x, 0, 16), vref(x, 16, 16), [  ]))), 16, 2) ) ],
  unrules := [ rec(
          perm := rec(
              spl := Tensor(
                  L(4, 2), 
                  I(4)
                ),
              l := 1,
              N := 4,
              n := 2,
              r := 4 ),
          instr := rec(
              instr := vushuffle4_16x8i,
              p := [ 1, 3, 2, 4 ] ),
          v := 16,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(4)
              ), (y, x) -> assign(vref(y, 0, 16), vushuffle4_16x8i(vref(x, 0, 16), [ 1, 3, 2, 4 ])), 16, 0.90000000000000002) ), rec(
          perm := rec(
              spl := Tensor(
                  I(2), 
                  L(4, 2), 
                  I(2)
                ),
              l := 2,
              N := 4,
              n := 2,
              r := 2 ),
          instr := [ rec(
                  instr := vushufflelo2_16x8i,
                  p := [ 1, 3, 2, 4 ] ), rec(
                  instr := vushufflehi2_16x8i,
                  p := [ 1, 3, 2, 4 ] ) ],
          v := 16,
          vperm := VPerm(Tensor(
                I(2), 
                L(4, 2), 
                I(2)
              ), (y, x) -> Let(t => TempVec(TArray(x.t.t, Eval(v))),
               decl(t, chain(assign(vref(t, 0, Eval(v)), Eval(i1)(vref(x, 0, Eval(v)), Eval(op1.p))), assign(vref(y, 0, Eval(v)), Eval(i2)(vref(t, 0, Eval(v)), Eval(op2.p))))) )
             , 16, 1.8) ) ],
  x_I_vby2 := [ [ rec(
              instr := vunpacklo8_16x8i,
              p := [  ] ), rec(
              instr := vshuffle4_16x8i,
              p := [ 1, 2, 3, 4 ] ) ], [ rec(
              instr := vshuffle4_16x8i,
              p := [ 3, 4, 1, 2 ] ), rec(
              instr := vunpackhi8_16x8i,
              p := [  ] ) ] ] ));
