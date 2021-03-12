
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

NEON.setRules(rec(
  binrules := [ rec(
          perm := rec(
              spl := L(8, 2),
              l := 1,
              N := 8,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vpacklo_neon,
                  p := [  ] ), rec(
                  instr := vpackhi_neon,
                  p := [  ] ) ],
          v := 4,
          vperm := VPerm(L(8, 2), (y, x) -> chain(assign(vref(y, 0, 4), vpacklo_neon(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vpackhi_neon(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 2) ), rec(
          perm := rec(
              spl := L(8, 4),
              l := 1,
              N := 8,
              n := 4,
              r := 1 ),
          instr := [ rec(
                  instr := vunpacklo_neon,
                  p := [  ] ), rec(
                  instr := vunpackhi_neon,
                  p := [  ] ) ],
          v := 4,
          vperm := VPerm(L(8, 4), (y, x) -> chain(assign(vref(y, 0, 4), vunpacklo_neon(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vunpackhi_neon(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 2) ), rec(
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
                  instr := vunpacklolo2_neon,
                  p := [  ] ), rec(
                  instr := vunpackhihi2_neon,
                  p := [  ] ) ],
          v := 4,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vunpacklolo2_neon(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vunpackhihi2_neon(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 2) ), rec(
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
                  instr := vpacklo_neon,
                  p := [  ] ), rec(
                  instr := vpackhi_neon,
                  p := [  ] ) ],
          v := 4,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vunpacklolo2_neon(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vunpackhihi2_neon(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 2) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ) * 
              Tensor(
                I(2), 
                L(4, 2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vpacklo_neon(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vpackhi_neon(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 4.2000000000000002) * 
            VTensor(I(2), 4) ), rec(
          perm := rec(
              spl := L(8, 4),
              l := 1,
              N := 8,
              n := 4,
              r := 1 ),
          instr := [ rec(
                  instr := vtransposelo_neon,
                  p := [  ] ), rec(
                  instr := vtransposehi_neon,
                  p := [  ] ) ],
          v := 4,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vunpacklolo2_neon(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vunpackhihi2_neon(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 2) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ) * 
              L(8, 4), (y, x) -> chain(assign(vref(y, 0, 4), vtransposelo_neon(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vtransposehi_neon(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 4.2000000000000002) * 
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
                  instr := vunpacklo_neon,
                  p := [  ] ), rec(
                  instr := vunpackhi_neon,
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
              ), (y, x) -> chain(assign(vref(y, 0, 4), vunpacklo_neon(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vunpackhi_neon(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 4.2000000000000002) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vunpacklolo2_neon(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vunpackhihi2_neon(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 2) ), rec(
          perm := rec(
              spl := L(8, 2),
              l := 1,
              N := 8,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vtransposelo_neon,
                  p := [  ] ), rec(
                  instr := vtransposehi_neon,
                  p := [  ] ) ],
          v := 4,
          vperm := VTensor(I(2), 4) * 
            VPerm(L(8, 2) * 
              Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vtransposelo_neon(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vtransposehi_neon(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 4.2000000000000002) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vunpacklolo2_neon(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vunpackhihi2_neon(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 2) ) ],
  unrules := [  ],
  x_I_vby2 := [ [ rec(
              instr := vunpacklolo2_neon,
              p := [  ] ), rec(
              instr := vunpacklohi2_neon,
              p := [  ] ) ], [ rec(
              instr := vunpackhilo2_neon,
              p := [  ] ), rec(
              instr := vunpackhihi2_neon,
              p := [  ] ) ] ] ));
NEON_HALF.setRules(rec(
  binrules := [ rec(
          perm := rec(
              spl := L(4, 2),
              l := 1,
              N := 4,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vunpacklo_half,
                  p := [  ] ), rec(
                  instr := vunpackhi_half,
                  p := [  ] ) ],
          v := 2,
          vperm := VPerm(L(4, 2), (y, x) -> chain(assign(vref(y, 0, 2), vunpacklo_half(vref(x, 0, 2), vref(x, 2, 2), [  ])), assign(vref(y, 2, 2), vunpackhi_half(vref(x, 0, 2), vref(x, 2, 2), [  ]))), 2, 2) ) ],
  unrules := [  ],
  x_I_vby2 := [ [ rec(
              instr := vunpacklo_half,
              p := [  ] ), [  ] ], [ [  ], rec(
              instr := vunpackhi_half,
              p := [  ] ) ] ] ));
