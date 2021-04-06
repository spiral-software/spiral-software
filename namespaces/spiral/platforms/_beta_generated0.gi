
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

LRB_16x32f.setRules(rec(
  binrules := [ rec(
          perm := rec(
              spl := L(32, 16),
              l := 1,
              N := 32,
              n := 16,
              r := 1 ),
          instr := [ rec(
                  instr := vunpacklo_16x32f,
                  p := [  ] ), rec(
                  instr := vunpackhi_16x32f,
                  p := [  ] ) ],
          v := 16,
          vperm := VPerm(L(32, 16), (y, x) -> chain(assign(vref(y, 0, 16), vunpacklo_16x32f(vref(x, 0, 16), vref(x, 16, 16), [  ])), assign(vref(y, 16, 16), vunpackhi_16x32f(vref(x, 0, 16), vref(x, 16, 16), [  ]))), 16, 2) ), rec(
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
                  instr := vunpacklo2_16x32f,
                  p := [  ] ), rec(
                  instr := vunpackhi2_16x32f,
                  p := [  ] ) ],
          v := 16,
          vperm := VPerm(Tensor(
                L(16, 8), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 16), vunpacklo2_16x32f(vref(x, 0, 16), vref(x, 16, 16), [  ])), assign(vref(y, 16, 16), vunpackhi2_16x32f(vref(x, 0, 16), vref(x, 16, 16), [  ]))), 16, 2) ) ],
  unrules := [  ],
  x_I_vby2 := [ [ [  ], [  ] ], [ [  ], [  ] ] ] ));
