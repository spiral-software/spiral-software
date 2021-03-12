
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

AVX_4x64f.setRules(rec(
  binrules := [ rec(
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
                  instr := vpermf128_4x64f,
                  p := [ 1, 3 ] ), rec(
                  instr := vpermf128_4x64f,
                  p := [ 2, 4 ] ) ],
          v := 4,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vpermf128_4x64f(vref(x, 0, 4), vref(x, 4, 4), [ 1, 3 ])), assign(vref(y, 4, 4), vpermf128_4x64f(vref(x, 0, 4), vref(x, 4, 4), [ 2, 4 ]))), 4, 2) ), rec(
          perm := rec(
              spl := L(8, 4),
              l := 1,
              N := 8,
              n := 4,
              r := 1 ),
          instr := [ rec(
                  instr := vunpacklo_4x64f,
                  p := [  ] ), rec(
                  instr := vunpackhi_4x64f,
                  p := [  ] ) ],
          v := 4,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vpermf128_4x64f(vref(x, 0, 4), vref(x, 4, 4), [ 1, 3 ])), assign(vref(y, 4, 4), vpermf128_4x64f(vref(x, 0, 4), vref(x, 4, 4), [ 2, 4 ]))), 4, 2) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ) * 
              L(8, 4), (y, x) -> chain(assign(vref(y, 0, 4), vunpacklo_4x64f(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vunpackhi_4x64f(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 4.2000000000000002) * 
            VTensor(I(2), 4) ), rec(
          perm := rec(
              spl := L(8, 2),
              l := 1,
              N := 8,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vunpacklo_4x64f,
                  p := [  ] ), rec(
                  instr := vunpackhi_4x64f,
                  p := [  ] ) ],
          v := 4,
          vperm := VTensor(I(2), 4) * 
            VPerm(L(8, 2) * 
              Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vunpacklo_4x64f(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vunpackhi_4x64f(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 4.2000000000000002) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vpermf128_4x64f(vref(x, 0, 4), vref(x, 4, 4), [ 1, 3 ])), assign(vref(y, 4, 4), vpermf128_4x64f(vref(x, 0, 4), vref(x, 4, 4), [ 2, 4 ]))), 4, 2) ) ],
  unrules := [  ],
  x_I_vby2 := [ [ rec(
              instr := vpermf128_4x64f,
              p := [ 1, 3 ] ), rec(
              instr := vperm2_4x64f,
              p := [ 1, 2, 3, 4 ] ) ], [ rec(
              instr := vpermf128_4x64f,
              p := [ 2, 3 ] ), rec(
              instr := vpermf128_4x64f,
              p := [ 2, 4 ] ) ] ] ));
AVX_8x32f.setRules(rec(
  binrules := [ rec(
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
                  instr := vpermf128_8x32f,
                  p := [ 1, 3 ] ), rec(
                  instr := vpermf128_8x32f,
                  p := [ 2, 4 ] ) ],
          v := 8,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(4)
              ), (y, x) -> chain(assign(vref(y, 0, 8), vpermf128_8x32f(vref(x, 0, 8), vref(x, 8, 8), [ 1, 3 ])), assign(vref(y, 8, 8), vpermf128_8x32f(vref(x, 0, 8), vref(x, 8, 8), [ 2, 4 ]))), 8, 2) ), rec(
          perm := rec(
              spl := L(16, 8),
              l := 1,
              N := 16,
              n := 8,
              r := 1 ),
          instr := [ rec(
                  instr := vunpacklo_8x32f,
                  p := [  ] ), rec(
                  instr := vunpackhi_8x32f,
                  p := [  ] ) ],
          v := 8,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(4)
              ), (y, x) -> chain(assign(vref(y, 0, 8), vpermf128_8x32f(vref(x, 0, 8), vref(x, 8, 8), [ 1, 3 ])), assign(vref(y, 8, 8), vpermf128_8x32f(vref(x, 0, 8), vref(x, 8, 8), [ 2, 4 ]))), 8, 2) * 
            VPerm(Tensor(
                L(4, 2), 
                I(4)
              ) * 
              L(16, 8), (y, x) -> chain(assign(vref(y, 0, 8), vunpacklo_8x32f(vref(x, 0, 8), vref(x, 8, 8), [  ])), assign(vref(y, 8, 8), vunpackhi_8x32f(vref(x, 0, 8), vref(x, 8, 8), [  ]))), 8, 4.2000000000000002) * 
            VTensor(I(2), 8) ), rec(
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
                  instr := vshuffle_8x32f,
                  p := [ 1, 2, 1, 2 ] ), rec(
                  instr := vshuffle_8x32f,
                  p := [ 3, 4, 3, 4 ] ) ],
          v := 8,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(4)
              ), (y, x) -> chain(assign(vref(y, 0, 8), vpermf128_8x32f(vref(x, 0, 8), vref(x, 8, 8), [ 1, 3 ])), assign(vref(y, 8, 8), vpermf128_8x32f(vref(x, 0, 8), vref(x, 8, 8), [ 2, 4 ]))), 8, 2) * 
            VPerm(Tensor(
                L(4, 2), 
                I(4)
              ) * 
              Tensor(
                L(8, 4), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 8), vshuffle_8x32f(vref(x, 0, 8), vref(x, 8, 8), [ 1, 2, 1, 2 ])), assign(vref(y, 8, 8), vshuffle_8x32f(vref(x, 0, 8), vref(x, 8, 8), [ 3, 4, 3, 4 ]))), 8, 4.2000000000000002) * 
            VTensor(I(2), 8) ), rec(
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
                  instr := vshuffle_8x32f,
                  p := [ 1, 2, 1, 2 ] ), rec(
                  instr := vshuffle_8x32f,
                  p := [ 3, 4, 3, 4 ] ) ],
          v := 8,
          vperm := VTensor(I(2), 8) * 
            VPerm(Tensor(
                L(8, 2), 
                I(2)
              ) * 
              Tensor(
                L(4, 2), 
                I(4)
              ), (y, x) -> chain(assign(vref(y, 0, 8), vshuffle_8x32f(vref(x, 0, 8), vref(x, 8, 8), [ 1, 2, 1, 2 ])), assign(vref(y, 8, 8), vshuffle_8x32f(vref(x, 0, 8), vref(x, 8, 8), [ 3, 4, 3, 4 ]))), 8, 4.2000000000000002) * 
            VPerm(Tensor(
                L(4, 2), 
                I(4)
              ), (y, x) -> chain(assign(vref(y, 0, 8), vpermf128_8x32f(vref(x, 0, 8), vref(x, 8, 8), [ 1, 3 ])), assign(vref(y, 8, 8), vpermf128_8x32f(vref(x, 0, 8), vref(x, 8, 8), [ 2, 4 ]))), 8, 2) ), rec(
          perm := rec(
              spl := L(16, 2),
              l := 1,
              N := 16,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vshuffle_8x32f,
                  p := [ 1, 3, 1, 3 ] ), rec(
                  instr := vshuffle_8x32f,
                  p := [ 2, 4, 2, 4 ] ) ],
          v := 8,
          vperm := VTensor(I(2), 8) * 
            VPerm(L(16, 2) * 
              Tensor(
                L(4, 2), 
                I(4)
              ), (y, x) -> chain(assign(vref(y, 0, 8), vshuffle_8x32f(vref(x, 0, 8), vref(x, 8, 8), [ 1, 3, 1, 3 ])), assign(vref(y, 8, 8), vshuffle_8x32f(vref(x, 0, 8), vref(x, 8, 8), [ 2, 4, 2, 4 ]))), 8, 4.2000000000000002) * 
            VPerm(Tensor(
                L(4, 2), 
                I(4)
              ), (y, x) -> chain(assign(vref(y, 0, 8), vpermf128_8x32f(vref(x, 0, 8), vref(x, 8, 8), [ 1, 3 ])), assign(vref(y, 8, 8), vpermf128_8x32f(vref(x, 0, 8), vref(x, 8, 8), [ 2, 4 ]))), 8, 2) ) ],
  unrules := [ rec(
          perm := rec(
              spl := Tensor(
                  I(2), 
                  L(4, 2)
                ),
              l := 2,
              N := 4,
              n := 2,
              r := 1 ),
          instr := rec(
              instr := vperm_8x32f,
              p := [ 1, 3, 2, 4 ] ),
          v := 8,
          vperm := VPerm(Tensor(
                I(2), 
                L(4, 2)
              ), (y, x) -> assign(vref(y, 0, 8), vperm_8x32f(vref(x, 0, 8), [ 1, 3, 2, 4 ])), 8, 0.90000000000000002) ), rec(
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
              instr := vcxtr_8x32f,
              p := [  ] ),
          v := 8,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> assign(vref(y, 0, 8), vcxtr_8x32f(vref(x, 0, 8), [  ])), 8, 4.5) ) ],
  x_I_vby2 := [ [ rec(
              instr := vpermf128_8x32f,
              p := [ 1, 3 ] ), rec(
              instr := vpermf128_8x32f,
              p := [ 1, 4 ] ) ], [ rec(
              instr := vpermf128_8x32f,
              p := [ 2, 3 ] ), rec(
              instr := vpermf128_8x32f,
              p := [ 2, 4 ] ) ] ] ));
