
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

MACRO_2xf.setRules(rec(
  binrules := [ rec(
          perm := rec(
              spl := L(4, 2),
              l := 1,
              N := 4,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vunpacklo_2xf,
                  p := [  ] ), rec(
                  instr := vunpackhi_2xf,
                  p := [  ] ) ],
          v := 2,
          vperm := VPerm(L(4, 2), (y, x) -> chain(assign(vref(y, 0, 2), vunpacklo_2xf(vref(x, 0, 2), vref(x, 2, 2), [  ])), assign(vref(y, 2, 2), vunpackhi_2xf(vref(x, 0, 2), vref(x, 2, 2), [  ]))), 2, 2) ) ],
  unrules := [  ],
  x_I_vby2 := [ [ rec(
              instr := vunpacklo_2xf,
              p := [  ] ), [  ] ], [ [  ], rec(
              instr := vunpackhi_2xf,
              p := [  ] ) ] ] ));
MACRO_4xf.setRules(rec(
  binrules := [ rec(
          perm := rec(
              spl := L(8, 2),
              l := 1,
              N := 8,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vpacklo_4xf,
                  p := [  ] ), rec(
                  instr := vpackhi_4xf,
                  p := [  ] ) ],
          v := 4,
          vperm := VPerm(L(8, 2), (y, x) -> chain(assign(vref(y, 0, 4), vpacklo_4xf(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vpackhi_4xf(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 2) ), rec(
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
                  instr := vpacklo2_4xf,
                  p := [  ] ), rec(
                  instr := vpackhi2_4xf,
                  p := [  ] ) ],
          v := 4,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vpacklo2_4xf(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vpackhi2_4xf(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 2) ), rec(
          perm := rec(
              spl := L(8, 4),
              l := 1,
              N := 8,
              n := 4,
              r := 1 ),
          instr := [ rec(
                  instr := vunpacklo_4xf,
                  p := [  ] ), rec(
                  instr := vunpackhi_4xf,
                  p := [  ] ) ],
          v := 4,
          vperm := VPerm(L(8, 4), (y, x) -> chain(assign(vref(y, 0, 4), vunpacklo_4xf(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vunpackhi_4xf(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 2) ), rec(
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
                  instr := vpacklo_4xf,
                  p := [  ] ), rec(
                  instr := vpackhi_4xf,
                  p := [  ] ) ],
          v := 4,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vpacklo2_4xf(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vpackhi2_4xf(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 2) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ) * 
              Tensor(
                I(2), 
                L(4, 2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vpacklo_4xf(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vpackhi_4xf(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 4.2000000000000002) * 
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
                  instr := vunpacklo_4xf,
                  p := [  ] ), rec(
                  instr := vunpackhi_4xf,
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
              ), (y, x) -> chain(assign(vref(y, 0, 4), vunpacklo_4xf(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vunpackhi_4xf(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 4.2000000000000002) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vpacklo2_4xf(vref(x, 0, 4), vref(x, 4, 4), [  ])), assign(vref(y, 4, 4), vpackhi2_4xf(vref(x, 0, 4), vref(x, 4, 4), [  ]))), 4, 2) ) ],
  unrules := [  ],
  x_I_vby2 := [ [ rec(
              instr := vpacklo2_4xf,
              p := [  ] ), [  ] ], [ [  ], rec(
              instr := vpackhi2_4xf,
              p := [  ] ) ] ] ));
MACRO_8xf.setRules(rec(
  binrules := [ rec(
          perm := rec(
              spl := L(16, 2),
              l := 1,
              N := 16,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vpacklo_8xf,
                  p := [  ] ), rec(
                  instr := vpackhi_8xf,
                  p := [  ] ) ],
          v := 8,
          vperm := VPerm(L(16, 2), (y, x) -> chain(assign(vref(y, 0, 8), vpacklo_8xf(vref(x, 0, 8), vref(x, 8, 8), [  ])), assign(vref(y, 8, 8), vpackhi_8xf(vref(x, 0, 8), vref(x, 8, 8), [  ]))), 8, 2) ), rec(
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
                  instr := vpacklo2_8xf,
                  p := [  ] ), rec(
                  instr := vpackhi2_8xf,
                  p := [  ] ) ],
          v := 8,
          vperm := VPerm(Tensor(
                L(8, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 8), vpacklo2_8xf(vref(x, 0, 8), vref(x, 8, 8), [  ])), assign(vref(y, 8, 8), vpackhi2_8xf(vref(x, 0, 8), vref(x, 8, 8), [  ]))), 8, 2) ), rec(
          perm := rec(
              spl := L(16, 8),
              l := 1,
              N := 16,
              n := 8,
              r := 1 ),
          instr := [ rec(
                  instr := vunpacklo_8xf,
                  p := [  ] ), rec(
                  instr := vunpackhi_8xf,
                  p := [  ] ) ],
          v := 8,
          vperm := VPerm(L(16, 8), (y, x) -> chain(assign(vref(y, 0, 8), vunpacklo_8xf(vref(x, 0, 8), vref(x, 8, 8), [  ])), assign(vref(y, 8, 8), vunpackhi_8xf(vref(x, 0, 8), vref(x, 8, 8), [  ]))), 8, 2) ), rec(
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
                  instr := vunpacklo2_8xf,
                  p := [  ] ), rec(
                  instr := vunpackhi2_8xf,
                  p := [  ] ) ],
          v := 8,
          vperm := VPerm(Tensor(
                L(8, 4), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 8), vunpacklo2_8xf(vref(x, 0, 8), vref(x, 8, 8), [  ])), assign(vref(y, 8, 8), vunpackhi2_8xf(vref(x, 0, 8), vref(x, 8, 8), [  ]))), 8, 2) ) ],
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
              instr := vtrcx_8xf,
              p := [  ] ),
          v := 8,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> assign(vref(y, 0, 8), vtrcx_8xf(vref(x, 0, 8), [  ])), 8, 0.90000000000000002) ) ],
  x_I_vby2 := [ [ [  ], [  ] ], [ [  ], [  ] ] ] ));
