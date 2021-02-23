
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


NewRulesFor(TConjEven, rec(
    TConjEven_vec := rec(
        switch := true,
        applicable := (self, t) >> IsEvenInt(t.params[1]) and t.isTag(1, AVecReg) and
	    t.params[1] mod (2*t.getTag(1).v) = 0,

        apply := (self, t, C, Nonterms) >> let(N := t.params[1], v := t.getTag(1).v, rot := t.params[2],
            m := Mat([[1,0],[0,-1]]),
#            d1 := Diag(diagDirsum(fConst(TReal, 2, 1.0), fConst(TReal, N-2, 1/2))),
            d1 := VDiag(diagDirsum(fConst(TReal, 1, 1.0), fConst(TReal, v-1, 0.5), fConst(TReal, 1, 1.0), fConst(TReal, N-v-1, 0.5)), v),
            #m1 := DirectSum(m, SUM(I(N-2), Tensor(J((N-2)/2), m))),
            #dv1 := Diag(diagDirsum(fConst(TReal, v, 1.0), fConst(TReal, 1,-1.0), fConst(TReal, N-v-1, 1.0))),
            dv1 := DirectSum(VTensor(I(1), v), 
		             VBlk([[ [-1.0] :: Replicate(v-1, 1.0) ]], v), 
			     VTensor(I(N/v-2), v)),
            P1 := VPrm_x_I(L(N/v, N/(2*v)), v),
            Q1 := VPrm_x_I(L(N/v, 2), v),
            jv1 := P1*DirectSum(VO1dsJ(N/2, v), VScale(VO1dsJ(N/2, v), -1, v))*Q1,
            mv1 := _VSUM([dv1,jv1], v),
            #m2 := DirectSum(J(2), SUM(I(N-2), Tensor(J((N-2)/2), -m))),
            e0 := [0.0] :: Replicate(v-1, 1.0), 
            e1 := [1.0] :: Replicate(v-1, 0.0), 
            blk := VBlk([[e0,e1],[e1,e0]], v).setSymmetric(),
            dv2 := DirectSum(blk, VTensor(I(N/v-2), v)),
            jv2 := P1 * DirectSum(VScale(VO1dsJ(N/2, v), -1, v), VO1dsJ(N/2, v))*Q1,
            mv2 := _VSUM([dv2,jv2], v),
            #
            i := VTensor(I(N/v), v), 
            d2 := VRCLR(Diag(fPrecompute(diagDirsum(fConst(TReal, 1, 1.0), diagMul(fConst(TComplex, N/2-1, omegapi(-1/2)), fCompose(dOmega(N, rot), fAdd(N/2, N/2-1, 1)))))), v),
            #
            #d1 * 
	    VBlkInt(d1 * _VHStack([mv1, mv2], v) * VStack(i, d2), v))
    ),

    TConjEven_vec_tr := rec(
        switch := true,
        applicable := (self, t) >> IsEvenInt(t.params[1]) and t.isTag(1, AVecReg) and t.params[1] mod (2*t.getTag(AVecReg).v)=0,
	transposed := true,

        apply := (self, t, C, Nonterms) >> let(N := t.params[1], v := t.getTag(1).v, rot := t.params[2],
            m := Mat([[1,0],[0,-1]]),
#            d1 := Diag(diagDirsum(fConst(TReal, 2, 1.0), fConst(TReal, N-2, 1/2))),
            d1 := VDiag(diagDirsum(fConst(TReal, 1, 1.0), fConst(TReal, v-1, 0.5), fConst(TReal, 1, 1.0), fConst(TReal, N-v-1, 0.5)), v),
            #m1 := DirectSum(m, SUM(I(N-2), Tensor(J((N-2)/2), m))),
            #dv1 := Diag(diagDirsum(fConst(TReal, v, 1.0), fConst(TReal, 1,-1.0), fConst(TReal, N-v-1, 1.0))),
            dv1 := DirectSum(VTensor(I(1), v), 
		             VBlk([[ [-1.0] :: Replicate(v-1, 1.0) ]], v), 
			     VTensor(I(N/v-2), v)),
            P1 := VPrm_x_I(L(N/v, N/(2*v)), v),
            Q1 := VPrm_x_I(L(N/v, 2), v),
            jv1 := P1*DirectSum(VO1dsJ(N/2, v), VScale(VO1dsJ(N/2, v), -1, v))*Q1,
            mv1 := _VSUM([dv1,jv1], v),
            #m2 := DirectSum(J(2), SUM(I(N-2), Tensor(J((N-2)/2), -m))),
            e0 := [0.0] :: Replicate(v-1, 1.0), 
            e1 := [1.0] :: Replicate(v-1, 0.0), 
            blk := VBlk([[e0,e1],[e1,e0]], v).setSymmetric(),
            dv2 := DirectSum(blk, VTensor(I(N/v-2), v)),
            jv2 := P1 * DirectSum(VScale(VO1dsJ(N/2, v), -1, v), VO1dsJ(N/2, v))*Q1,
            mv2 := _VSUM([dv2,jv2], v),
            #
            i := VTensor(I(N/v), v), 
            d2 := VRCLR(Diag(fPrecompute(diagDirsum(fConst(TReal, 1, 1.0), diagMul(fConst(TComplex, N/2-1, omegapi(-1/2)), fCompose(dOmega(N, rot), fAdd(N/2, N/2-1, 1)))))), v),
            #
            #d1 * 
	    VBlkInt(_VHStack([i, d2.transpose()], v) * VStack(mv1, mv2) * d1, v))
    )
));


NewRulesFor(TXMatDHT, rec(
    TXMatDHT_vec := rec(
        switch := true,
        applicable := (self, t) >> IsEvenInt(t.params[1]) and t.isTag(1, AVecReg),
        apply := (self, t, C, Nonterms) >> let(N := t.params[1], v := t.getTag(1).v,

            f0 := Replicate(v, 1.0), 
            f1 := [0.0] :: Replicate(v-1, 1.0), 
            f2 := [1.0] :: Replicate(v-1, -1.0), 
            fblk := VBlk([[f0,f1],[f1,f2]], v).setSymmetric(),
            fdiag := VTensor(Tensor(I(N/(2*v)-1), F(2)), v),
            ff := DirectSum(fblk, fdiag),

            ###
            m := Mat([[1,0],[0,-1]]),
            d1 := Diag(diagDirsum(fConst(TReal, 2, 1.0), fConst(TReal, N-2, 1/2))),
            #m1 := DirectSum(m, SUM(I(N-2), Tensor(J((N-2)/2), m))),
            #dv1 := Diag(diagDirsum(fConst(TReal, v,1), fConst(TReal, 1,-1), fConst(TReal, N-v-1, 1))),
            dv1 := DirectSum(VTensor(I(1), v), 
		             VBlk([[ [-1.0] :: Replicate(v-1, 1.0) ]], v), 
			     VTensor(I(N/v-2), v)),
            P1 := VPrm_x_I(L(N/v, N/(2*v)), v),
            Q1 := VPrm_x_I(L(N/v, 2), v),
            jv1 := P1*DirectSum(VO1dsJ(N/2, v), VScale(VO1dsJ(N/2, v), -1, v))*Q1,
            mv1 := _VSUM([dv1,jv1], v),
            #m2 := DirectSum(J(2), SUM(I(N-2), Tensor(J((N-2)/2), -m))),
            e0 := [0.0] :: Replicate(v-1, 1.0), 
            e1 := [1.0] :: Replicate(v-1, 0.0), 
            blk := VBlk([[e0,e1],[e1,e0]], v).setSymmetric(),
            dv2 := DirectSum(blk, VTensor(I(N/v-2), v)),
            jv2 := P1*DirectSum(VScale(VO1dsJ(N/2, v), -1, v), VO1dsJ(N/2, v))*Q1,
            mv2 := _VSUM([dv2,jv2], v),
            #
            i := VTensor(I(N/v), v),
            d2 := VRCLR(Diag(fPrecompute(diagDirsum(fConst(TReal, 1, 1.0), diagMul(fConst(TComplex, N/2-1, Complex(0,-1)), fCompose(dOmega(N, 1), fAdd(N/2, N/2-1, 1)))))), v),
            #
            d1 * VBlkInt(ff * _VHStack([mv1, mv2], v) * VStack(i, d2), v)
    ))
));
