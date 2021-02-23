
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Import(platforms.neon);
d4 :=benchNEON().4x32f.1d.dft_ic.medium();
d2:=benchNEON().2x32f.1d.dft_ic.medium();
opts4 := d4.getOpts();
opts4.profile := default_profiles.arm_icount;
opts2 := d2.getOpts();
opts2.profile := default_profiles.arm_icount;

v := 2;
opts := opts2;

tests := [
    VGath_sv(H(100,1,0,7), v, 1),
    VGath_sv(H(100,2,0,7), v, 1),
    VGath_sv(H(100,3,0,7), v, 1),
    VGath_sv(H(100,4,0,7), v, 1),
    VGath_sv(H(100,5,0,7), v, 1),
    VGath_sv(H(100,12,0,7), v, 1),

    VGath_sv(H(100,1,0,7), v, 2),
    VGath_sv(H(100,2,0,7), v, 2),
    VGath_sv(H(100,3,0,7), v, 2),

    VS(v,v),
    VS(2*v,v),

    VGath_pc(5, 1, 0, v),
    VGath_pc(5, 2, 0, v),
    VGath_pc(5, 3, 0, v),
    VGath_pc(5, 4, 0, v),
    VGath_pc(10, 5, 0, v),
    VGath_pc(10, 6, 0, v),
    VGath_pc(10, 7, 0, v),
    VGath_pc(10, 8, 0, v),
    VGath_pc(10, 9, 0, v)
];

tests2 := [
    VGath_dup(H(100,1,0,7), v),
    VGath_dup(H(100,3,0,7), v),
    VGath_dup(H(100,4,0,7), v),
    VReplicate(v),
    VIxJ2(v),
    RCVIxJ2(v),
    VO1dsJ(v, v),
    VO1dsJ(2*v, v),

    RCVBlk([[  [E(3), E(4)], [0,0]], [[0,0], [E(5), E(7)]]], v),
    
];

for s in tests :: List(tests, x->x.transpose()) :: tests2 do
    c := CodeSums(s, opts);
    t := CMeasure(c, opts);
    PrintLine(s, "   ", t);
od;


