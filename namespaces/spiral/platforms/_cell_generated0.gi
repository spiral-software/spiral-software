
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

spu_4x32f.setRules(rec(
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
                  instr := vperm_4x32f_spu,
                  p := [ 1, 2, 5, 6 ] ), rec(
                  instr := vperm_4x32f_spu,
                  p := [ 3, 4, 7, 8 ] ) ],
          v := 4,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 2, 5, 6 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 3, 4, 7, 8 ]))), 4, 2) ), rec(
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
                  instr := vperm_4x32f_spu,
                  p := [ 1, 3, 2, 4 ] ), rec(
                  instr := vperm_4x32f_spu,
                  p := [ 5, 7, 6, 8 ] ) ],
          v := 4,
          vperm := VPerm(Tensor(
                I(2), 
                L(4, 2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 3, 2, 4 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 5, 7, 6, 8 ]))), 4, 2) ), rec(
          perm := rec(
              spl := L(8, 2),
              l := 1,
              N := 8,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vperm_4x32f_spu,
                  p := [ 1, 3, 5, 7 ] ), rec(
                  instr := vperm_4x32f_spu,
                  p := [ 2, 4, 6, 8 ] ) ],
          v := 4,
          vperm := VPerm(L(8, 2), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 3, 5, 7 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 2, 4, 6, 8 ]))), 4, 2) ), rec(
          perm := rec(
              spl := L(8, 4),
              l := 1,
              N := 8,
              n := 4,
              r := 1 ),
          instr := [ rec(
                  instr := vperm_4x32f_spu,
                  p := [ 1, 5, 2, 6 ] ), rec(
                  instr := vperm_4x32f_spu,
                  p := [ 3, 7, 4, 8 ] ) ],
          v := 4,
          vperm := VPerm(L(8, 4), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 5, 2, 6 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 3, 7, 4, 8 ]))), 4, 2) ), rec(
          perm := rec(
              spl := L(8, 2),
              l := 1,
              N := 8,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vperm_4x32f_spu,
                  p := [ 1, 3, 2, 4 ] ), rec(
                  instr := vperm_4x32f_spu,
                  p := [ 5, 7, 6, 8 ] ) ],
          v := 4,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 2, 5, 6 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 3, 4, 7, 8 ]))), 4, 2) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ) * 
              L(8, 2), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 3, 2, 4 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 5, 7, 6, 8 ]))), 4, 4.2000000000000002) * 
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
                  instr := vperm_4x32f_spu,
                  p := [ 1, 3, 5, 7 ] ), rec(
                  instr := vperm_4x32f_spu,
                  p := [ 2, 4, 6, 8 ] ) ],
          v := 4,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 2, 5, 6 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 3, 4, 7, 8 ]))), 4, 2) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ) * 
              Tensor(
                I(2), 
                L(4, 2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 3, 5, 7 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 2, 4, 6, 8 ]))), 4, 4.2000000000000002) * 
            VTensor(I(2), 4) ), rec(
          perm := rec(
              spl := L(8, 4),
              l := 1,
              N := 8,
              n := 4,
              r := 1 ),
          instr := [ rec(
                  instr := vperm_4x32f_spu,
                  p := [ 1, 5, 3, 7 ] ), rec(
                  instr := vperm_4x32f_spu,
                  p := [ 2, 6, 4, 8 ] ) ],
          v := 4,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 2, 5, 6 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 3, 4, 7, 8 ]))), 4, 2) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ) * 
              L(8, 4), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 5, 3, 7 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 2, 6, 4, 8 ]))), 4, 4.2000000000000002) * 
            VTensor(I(2), 4) ), rec(
          perm := rec(
              spl := L(8, 4),
              l := 1,
              N := 8,
              n := 4,
              r := 1 ),
          instr := [ rec(
                  instr := vperm_4x32f_spu,
                  p := [ 1, 3, 2, 4 ] ), rec(
                  instr := vperm_4x32f_spu,
                  p := [ 5, 7, 6, 8 ] ) ],
          v := 4,
          vperm := VTensor(I(2), 4) * 
            VPerm(L(8, 4) * 
              Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 3, 2, 4 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 5, 7, 6, 8 ]))), 4, 4.2000000000000002) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 2, 5, 6 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 3, 4, 7, 8 ]))), 4, 2) ), rec(
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
                  instr := vperm_4x32f_spu,
                  p := [ 1, 5, 2, 6 ] ), rec(
                  instr := vperm_4x32f_spu,
                  p := [ 3, 7, 4, 8 ] ) ],
          v := 4,
          vperm := VTensor(I(2), 4) * 
            VPerm(Tensor(
                I(2), 
                L(4, 2)
              ) * 
              Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 5, 2, 6 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 3, 7, 4, 8 ]))), 4, 4.2000000000000002) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 2, 5, 6 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 3, 4, 7, 8 ]))), 4, 2) ), rec(
          perm := rec(
              spl := L(8, 2),
              l := 1,
              N := 8,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vperm_4x32f_spu,
                  p := [ 1, 5, 3, 7 ] ), rec(
                  instr := vperm_4x32f_spu,
                  p := [ 2, 6, 4, 8 ] ) ],
          v := 4,
          vperm := VTensor(I(2), 4) * 
            VPerm(L(8, 2) * 
              Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 5, 3, 7 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 2, 6, 4, 8 ]))), 4, 4.2000000000000002) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 2, 5, 6 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 3, 4, 7, 8 ]))), 4, 2) ) ],
  unrules := [ rec(
          perm := rec(
              spl := L(4, 2),
              l := 1,
              N := 4,
              n := 2,
              r := 1 ),
          instr := rec(
              instr := vuperm_4x32f_spu,
              p := [ 1, 3, 2, 4 ] ),
          v := 4,
          vperm := VPerm(L(4, 2), (y, x) -> assign(vref(y, 0, 4), vuperm_4x32f_spu(vref(x, 0, 4), [ 1, 3, 2, 4 ])), 4, 0.90000000000000002) ) ],
  x_I_vby2 := [ [ rec(
              instr := vperm_4x32f_spu,
              p := [ 1, 2, 5, 6 ] ), rec(
              instr := vperm_4x32f_spu,
              p := [ 1, 2, 7, 8 ] ) ], [ rec(
              instr := vperm_4x32f_spu,
              p := [ 3, 4, 5, 6 ] ), rec(
              instr := vperm_4x32f_spu,
              p := [ 3, 4, 7, 8 ] ) ] ] ));
spu_2x64f.setRules(rec(
  binrules := [ rec(
          perm := rec(
              spl := L(4, 2),
              l := 1,
              N := 4,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vperm_2x64f_spu,
                  p := [ 1, 3 ] ), rec(
                  instr := vperm_2x64f_spu,
                  p := [ 2, 4 ] ) ],
          v := 2,
          vperm := VPerm(L(4, 2), (y, x) -> chain(assign(vref(y, 0, 2), vperm_2x64f_spu(vref(x, 0, 2), vref(x, 2, 2), [ 1, 3 ])), assign(vref(y, 2, 2), vperm_2x64f_spu(vref(x, 0, 2), vref(x, 2, 2), [ 2, 4 ]))), 2, 2) ) ],
  unrules := [  ],
  x_I_vby2 := [ [ rec(
              instr := vperm_2x64f_spu,
              p := [ 1, 3 ] ), rec(
              instr := vperm_2x64f_spu,
              p := [ 1, 4 ] ) ], [ rec(
              instr := vperm_2x64f_spu,
              p := [ 2, 3 ] ), rec(
              instr := vperm_2x64f_spu,
              p := [ 2, 4 ] ) ] ] ));
ppu_4x32f.setRules(rec(
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
                  instr := vperm_4x32f_spu,
                  p := [ 1, 2, 5, 6 ] ), rec(
                  instr := vperm_4x32f_spu,
                  p := [ 3, 4, 7, 8 ] ) ],
          v := 4,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 2, 5, 6 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 3, 4, 7, 8 ]))), 4, 2) ), rec(
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
                  instr := vperm_4x32f_spu,
                  p := [ 1, 3, 2, 4 ] ), rec(
                  instr := vperm_4x32f_spu,
                  p := [ 5, 7, 6, 8 ] ) ],
          v := 4,
          vperm := VPerm(Tensor(
                I(2), 
                L(4, 2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 3, 2, 4 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 5, 7, 6, 8 ]))), 4, 2) ), rec(
          perm := rec(
              spl := L(8, 2),
              l := 1,
              N := 8,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vperm_4x32f_spu,
                  p := [ 1, 3, 5, 7 ] ), rec(
                  instr := vperm_4x32f_spu,
                  p := [ 2, 4, 6, 8 ] ) ],
          v := 4,
          vperm := VPerm(L(8, 2), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 3, 5, 7 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 2, 4, 6, 8 ]))), 4, 2) ), rec(
          perm := rec(
              spl := L(8, 4),
              l := 1,
              N := 8,
              n := 4,
              r := 1 ),
          instr := [ rec(
                  instr := vperm_4x32f_spu,
                  p := [ 1, 5, 2, 6 ] ), rec(
                  instr := vperm_4x32f_spu,
                  p := [ 3, 7, 4, 8 ] ) ],
          v := 4,
          vperm := VPerm(L(8, 4), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 5, 2, 6 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 3, 7, 4, 8 ]))), 4, 2) ), rec(
          perm := rec(
              spl := L(8, 2),
              l := 1,
              N := 8,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vperm_4x32f_spu,
                  p := [ 1, 3, 2, 4 ] ), rec(
                  instr := vperm_4x32f_spu,
                  p := [ 5, 7, 6, 8 ] ) ],
          v := 4,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 2, 5, 6 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 3, 4, 7, 8 ]))), 4, 2) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ) * 
              L(8, 2), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 3, 2, 4 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 5, 7, 6, 8 ]))), 4, 4.2000000000000002) * 
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
                  instr := vperm_4x32f_spu,
                  p := [ 1, 3, 5, 7 ] ), rec(
                  instr := vperm_4x32f_spu,
                  p := [ 2, 4, 6, 8 ] ) ],
          v := 4,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 2, 5, 6 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 3, 4, 7, 8 ]))), 4, 2) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ) * 
              Tensor(
                I(2), 
                L(4, 2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 3, 5, 7 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 2, 4, 6, 8 ]))), 4, 4.2000000000000002) * 
            VTensor(I(2), 4) ), rec(
          perm := rec(
              spl := L(8, 4),
              l := 1,
              N := 8,
              n := 4,
              r := 1 ),
          instr := [ rec(
                  instr := vperm_4x32f_spu,
                  p := [ 1, 5, 3, 7 ] ), rec(
                  instr := vperm_4x32f_spu,
                  p := [ 2, 6, 4, 8 ] ) ],
          v := 4,
          vperm := VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 2, 5, 6 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 3, 4, 7, 8 ]))), 4, 2) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ) * 
              L(8, 4), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 5, 3, 7 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 2, 6, 4, 8 ]))), 4, 4.2000000000000002) * 
            VTensor(I(2), 4) ), rec(
          perm := rec(
              spl := L(8, 4),
              l := 1,
              N := 8,
              n := 4,
              r := 1 ),
          instr := [ rec(
                  instr := vperm_4x32f_spu,
                  p := [ 1, 3, 2, 4 ] ), rec(
                  instr := vperm_4x32f_spu,
                  p := [ 5, 7, 6, 8 ] ) ],
          v := 4,
          vperm := VTensor(I(2), 4) * 
            VPerm(L(8, 4) * 
              Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 3, 2, 4 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 5, 7, 6, 8 ]))), 4, 4.2000000000000002) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 2, 5, 6 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 3, 4, 7, 8 ]))), 4, 2) ), rec(
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
                  instr := vperm_4x32f_spu,
                  p := [ 1, 5, 2, 6 ] ), rec(
                  instr := vperm_4x32f_spu,
                  p := [ 3, 7, 4, 8 ] ) ],
          v := 4,
          vperm := VTensor(I(2), 4) * 
            VPerm(Tensor(
                I(2), 
                L(4, 2)
              ) * 
              Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 5, 2, 6 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 3, 7, 4, 8 ]))), 4, 4.2000000000000002) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 2, 5, 6 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 3, 4, 7, 8 ]))), 4, 2) ), rec(
          perm := rec(
              spl := L(8, 2),
              l := 1,
              N := 8,
              n := 2,
              r := 1 ),
          instr := [ rec(
                  instr := vperm_4x32f_spu,
                  p := [ 1, 5, 3, 7 ] ), rec(
                  instr := vperm_4x32f_spu,
                  p := [ 2, 6, 4, 8 ] ) ],
          v := 4,
          vperm := VTensor(I(2), 4) * 
            VPerm(L(8, 2) * 
              Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 5, 3, 7 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 2, 6, 4, 8 ]))), 4, 4.2000000000000002) * 
            VPerm(Tensor(
                L(4, 2), 
                I(2)
              ), (y, x) -> chain(assign(vref(y, 0, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 1, 2, 5, 6 ])), assign(vref(y, 4, 4), vperm_4x32f_spu(vref(x, 0, 4), vref(x, 4, 4), [ 3, 4, 7, 8 ]))), 4, 2) ) ],
  unrules := [ rec(
          perm := rec(
              spl := L(4, 2),
              l := 1,
              N := 4,
              n := 2,
              r := 1 ),
          instr := rec(
              instr := vuperm_4x32f_spu,
              p := [ 1, 3, 2, 4 ] ),
          v := 4,
          vperm := VPerm(L(4, 2), (y, x) -> assign(vref(y, 0, 4), vuperm_4x32f_spu(vref(x, 0, 4), [ 1, 3, 2, 4 ])), 4, 0.90000000000000002) ) ],
  x_I_vby2 := [ [ rec(
              instr := vperm_4x32f_spu,
              p := [ 1, 2, 5, 6 ] ), rec(
              instr := vperm_4x32f_spu,
              p := [ 1, 2, 7, 8 ] ) ], [ rec(
              instr := vperm_4x32f_spu,
              p := [ 3, 4, 5, 6 ] ), rec(
              instr := vperm_4x32f_spu,
              p := [ 3, 4, 7, 8 ] ) ] ] ));
