
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


DropMat := function(fmat)
    local mat;
    mat := Copy(fmat);
    if IsBound(mat.mat) then Unbind(mat.mat); fi;
    return mat;
end;

#   All size x size digit swap permutations that could be implemented in register for v
AllDigitSwaps := function(size, v)
    local ll, l, Nl, N, nl, n, r, mats, L1, L2, spl;

    mats := [];
    ll := DropLast(DivisorsInt(v), 1);
    for l in ll do
        Nl := Filtered(DivisorsInt(size/l), i -> i >= 4);
        for N in Nl do
            r:= size/(N*l);
            nl := DivisorsIntDrop(N);
            for n in nl do
                if l=1 then L1 := []; else L1 := [I(l)]; fi;
                if r=1 then L2 := []; else L2 := [I(r)]; fi;
                spl := Tensor(Concat(L1, [L(N,n)], L2));
                Add(mats, rec(spl:=spl, mat:=MatSPL(spl), l:=l, N:=N, n:=n, r:=r));
            od;
        od;
    od;

    return mats;
end;

#   NOTE: Better algorithm: outer loop over AllDigitSwaps


BuildCandidates := function(ops, v)
    local mats, op, p, v;
    mats := [];
    for op in ops do
        for p in op.permparams() do
            Add(mats,
                rec(instr := op,
                    p := p,
#   NOTE - how to get op.v correctly? maybe introduce op.getVLen() method
                    mat := let(ov := When(IsInt(op.v), op.v, op.v()), When(op.isUnop(),
                        TransposedMat(List([0..ov-1], i->op.semantic(BasisVec(ov, i), p))),
                        TransposedMat(Concat(List([0..ov-1], i->op.semantic(BasisVec(ov, i), Replicate(v, 0), p)),
                            List([0..ov-1], i->op.semantic(Replicate(v, 0), BasisVec(ov, i), p))))
                        ))
                ));
        od;
    od;
    return mats;
end;


#   Build rules from binary operations
#
BuildBinRules := function(arch)
    local archrules, binops, p, op1, op2, i1, i2, v, binmats, binperms, dsperms, opmat, fmat;

    v := arch.v;
    archrules := [];
    binops := Filtered(arch.instr, i->i.isBinop());

    #   find al candidate instructions to build permutations
    binmats := BuildCandidates(binops, v);

    # find all pairs of binops to build valid swap perms
    dsperms := AllDigitSwaps(2*v, v);
    for op1 in binmats do
        for op2 in binmats do
            opmat := Concat(op1.mat, op2.mat);
            if IsPermMat(AMatMat(opmat)) then
                fmat := Filtered(dsperms, i->i.mat=opmat);
                if Length(fmat)=1 then
                    i1 := op1.instr; i2 := op2.instr;
                    Add(archrules, rec(perm := DropMat(fmat[1]), instr:=[DropMat(op1), DropMat(op2)], v:=v, # NOTE: use full list!
                                vperm := VPerm(fmat[1].spl,
                                Subst((y,x) -> chain(
                                assign(vref(y, 0,$v), $i1(vref(x,0,$v), vref(x,$v,$v), $(op1.p))),
                                assign(vref(y,$v,$v), $i2(vref(x,0,$v), vref(x,$v,$v), $(op2.p))))),
                            v, op1.instr.vcost + op2.instr.vcost)
                            ));
                    dsperms := Filtered(dsperms, i->opmat<>i.mat);
                fi;
            fi;
        od;
    od;
    return archrules;
end;

#   Build rules from binary operations - first shuffle Inputs by LxI
#
BuildBinRulesLxI := function(arch, binrules, left, right)
    local archrules, binops, p, op1, op2, i1, i2, v, binmats, binperms, dsperms, opmat, fmat, lm, lmleft, lmright,
        lmc, lmvleft, lmvright, opmatconcat, i2v, splleft, splright, lval, rval;

    v := arch.v;
    archrules := [];
    binops := Filtered(arch.instr, i->i.isBinop());

    lm := MatSPL(Tensor(L(4,2), I(v/2)));
    lmleft := When(left, lm, MatSPL(I(2*v)));
    splleft := When(left, Tensor(L(4,2), I(v/2)), I(2*v));
    lmright := When(right, lm, MatSPL(I(2*v)));
    splright := When(right, Tensor(L(4,2), I(v/2)), I(2*v));
    i2v := MatSPL(I(2*v));

    lmc := Filtered(binrules, i->(i.perm.l=1 and i.perm.r=v/2 and i.perm.N=4 and i.perm.n=2));

    if Length(lmc) = 0 then return []; fi;

    lmvleft := When(left, lmc[1].vperm, VTensor(I(2), v));
    lmvright := When(right, lmc[1].vperm, VTensor(I(2), v));

    lval := When(left, lmc[1].vperm._vcost + 0.2, 0);
    rval := When(right, lmc[1].vperm._vcost + 0.2, 0);

    #   find al candidate instructions to build permutations
    binmats := BuildCandidates(binops, v);

    # find all pairs of binops to build valid swap perms
    dsperms := AllDigitSwaps(2*v, v);
    for op1 in binmats do
        for op2 in binmats do
            opmatconcat := Concat(op1.mat, op2.mat);
            if opmatconcat  <> i2v then
                opmat := lmleft * opmatconcat * lmright;
                if IsPermMat(AMatMat(opmat)) then
                    fmat := Filtered(dsperms, i->i.mat=opmat);
                    if Length(fmat)=1 then
                        i1 := op1.instr; i2 := op2.instr;
                        Add(archrules, rec(perm := DropMat(fmat[1]), instr:=[DropMat(op1), DropMat(op2)], v:=v, # NOTE: use full list!
                                    vperm := lmvleft * VPerm(splleft * fmat[1].spl * splright,
                                    Subst((y,x) -> chain(
                                    assign(vref(y, 0,$v), $i1(vref(x,0,$v), vref(x,$v,$v), $(op1.p))),
                                    assign(vref(y,$v,$v), $i2(vref(x,0,$v), vref(x,$v,$v), $(op2.p))))),
                                v, 2 + lval + rval) * lmvright
                                ));
                        dsperms := Filtered(dsperms, i->opmat<>i.mat);
                    fi;
                fi;
            fi;
        od;
    od;
    return archrules;
end;

#   Build rules from unary operations
#
BuildUnRules := function(arch)
    local archrules, unops, unmats, unperms, unperms2, unmats2, p, op, op1, op2, v, dsperms, opmat, fmat, i1, i2;

    v := arch.v;
    archrules := [];

    #   unops -> perms
    unops := Filtered(arch.instr, i->i.isUnop());
    unmats := BuildCandidates(unops, v);

    # find all unops to build valid swap perms
    dsperms := AllDigitSwaps(v, v);
    unmats2 := [];
    for op in unmats do
        opmat := op.mat;
        if IsPermMat(AMatMat(opmat)) then
            fmat := Filtered(dsperms, i->i.mat=opmat);

            if Length(fmat)>=1 then
        i1 := op.instr;
        Add(archrules, rec(perm := DropMat(fmat[1]), instr:=DropMat(op), v:=v, # NOTE: use full list!
                    vperm := VPerm(fmat[1].spl,
                                   Subst((y,x) -> assign(vref(y, 0, $v), $i1(vref(x,0,$v), $(op.p)))),
                   v, 0.9*op.instr.vcost) # NOTE: avoid unops as they may not exist
        ));
                dsperms := Filtered(dsperms, i->opmat<>i.mat);
            else
                Add(unmats2, op);
            fi;
        fi;
    od;

    #   product of unops -> perms
    for op1 in unmats2 do
        for op2 in unmats2 do
            opmat := op2.mat * op1.mat;
            if IsPermMat(AMatMat(opmat)) then
                fmat := Filtered(dsperms, i->opmat=i.mat);

                if Length(fmat)=1 then
                    i1 := op1.instr;
                    i2 := op2.instr;
                    Add(archrules, rec(
                        perm  := DropMat(fmat[1]),
                        instr := [DropMat(op1), DropMat(op2)],
                        v     := v, # NOTE: use full list!
                        vperm := VPerm( fmat[1].spl, Subst((y,x) -> let(t := TempVec(TArray(x.t.t, $v)),
                                        decl(t, chain(assign(vref(t,0,$v), $i1(vref(x,0,$v), $(op1.p))),
                                                      assign(vref(y,0,$v), $i2(vref(t,0,$v), $(op2.p))))))),
                                        v, 0.9*(op1.instr.vcost+op2.instr.vcost))
                    ));
                    dsperms := Filtered(dsperms, i->opmat<>i.mat);
                fi;
            fi;
        od;
    od;

    return archrules;
end;


Find_x_I_vby2 := function(arch, i, j)
    local gmat, v, binops, binmats, fmat, op;

    v := arch.v;
    binops := Filtered(arch.instr, i->i.isBinop());

    #   find al candidate instructions to build permutations
    binmats := BuildCandidates(binops, v);
    gmat := MatSPL(Gath(fTensor(fDirsum(fBase(2,i), fBase(2,j)), fId(v/2))));

    fmat := Filtered(binmats, i->i.mat=gmat);
    return When(Length(fmat) > 0, DropMat(fmat[1]), []);
end;

Build_x_I_vby2 := arch -> List([0,1], i-> List([0,1], j-> Find_x_I_vby2(arch,i,j)));
