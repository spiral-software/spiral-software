
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#Class(RulesConj, RuleSet);
RewriteRules(RulesVDiag, rec(
#    conj_LR_L := ARule(Compose, [@(1,ConjLR), @(2,ConjL)],
#                    e -> let(c1:=@(1).val.children(), c2:=@(2).val.children(),
#                    [ ConjL(c1[1] * Prm(fCompose(c1[3], c2[2])) * c2[1], c1[2]) ])),
#
#    conj_LR_LR := ARule(Compose, [@(1,ConjLR), @(2,ConjLR)],
#                    e -> let(c1:=@(1).val.children(), c2:=@(2).val.children(),
#                    [ ConjLR(c1[1] * Prm(fCompose(c1[3], c2[2])) * c2[1], c1[2], c2[3]) ])),
#
#    conj_R_LR := ARule(Compose, [@(1,ConjR), @(2,ConjLR)],
#                    e -> let(c1:=@(1).val.children(), c2:=@(2).val.children(),
#                    [ ConjR(c1[1] * Prm(fCompose(c1[2], c2[2])) * c2[1], c2[3]) ])),

    #--------------------------------------------------
    # handle RCDiag

    VGath_Conj_RCDiag := ARule(Compose, [@(1,VGath), [@(2,ConjDiag), @(3,RCDiag), @(4), @(5) ]],
       e -> let(f:=@(1).val.func, d:=@(3).val.element, v:=@(1).val.v, prm:=@(4).val,
                [ VRCDiag(VData(fCompose(d, prm, fTensor(f, fId(v))), v), v),@(1).val ] )),

    Conj_RCDiag_VScat := ARule(Compose, [[@(1,ConjDiag), @(2,RCDiag), @(3), @(4) ], @(5,VScat)],
       e -> let(f:=@(5).val.func, d:=@(2).val.element, v:=@(5).val.v, prm:=@(3).val,
                [ @(5).val, VRCDiag(VData(fCompose(d, prm, fTensor(f, fId(v))), v), v) ] )),

    VTensor_Conj_RCDiag := ARule(Compose, [@(1,VTensor), [@(2,ConjDiag), @(3,RCDiag), @(4, L, e->e.params[2] = e.params[1]/@(1).val.vlen), @(5, L,e->e.params[2] = @(1).val.vlen) ]],
        e -> let(d:=@(3).val.element, v:=@(1).val.vlen, prm:=@(4).val,
                [ @(1).val, VRCDiag(VData(fCompose(d, prm), v), v) ] )),

    PullInRightConjDiag := ARule( Compose,
        [ @(1, ConjDiag),
            @(2, [RecursStep, Grp, BB, SUM, Buf, ISum, Data, COND]) ],
        e -> [ CopyFields(@(2).val, rec(
                _children :=  List(@(2).val._children, c -> @(1).val * c),
                dimensions := [Rows(@(1).val), Cols(@(2).val)] )) ]),

    PullInLeftConjDiag := ARule( Compose,
        [ @(1, [RecursStep, Grp, BB, SUM, SUMAcc, Buf, ISum, ISumAcc, Data, COND]),
            @(2, ConjDiag) ],
        e -> [ CopyFields(@(1).val, rec(
                _children := List(@(1).val._children, c -> c * @(2).val),
                dimensions := [Rows(@(1).val), Cols(@(2).val)] )) ])
));


#_leftPerm := (m,n,v) -> Compose(VTensor(Tensor(I(m/(2*v)),L(n, v)), v), Tensor(I(m*n/(2*v^2)), VL(v^2, v)), VTensor(L(m*n/(2*v), m/v), v)).sums();
#_rightPerm := (m,n,v) -> VTensor(Tensor(I(n/2), L(m/v,m/(2*v))), v).sums();
#
#Class(RulesConjTerm, RuleSet);
#RewriteRules(RulesConjTerm, rec(
#    ConjR_VRC_Diag := ARule(Compose, [ [ConjR, @(1), @(2,L)], [@(3,VRC), @(4,Diag) ] ],
#                    e -> let(l := @(2).val, mn_by_2:=l.params[1], m:=l.params[2], n:=2*mn_by_2/m, v:=@(3).val.v,
#                        [ @(1).val * _leftPerm(m,n,v) * VRCL(@(3).val.child(1), v) ])),
#
#    VRCL_VRC := ARule(Compose, [ @(1, VRCL), @(2, VRC) ],
#                    e -> let(v:=@(1).val.v, [ VRCLR(@(1).val.child(1), v), VRCL(@(2).val.child(1), v) ])),
#
#    VRCL_L_ConjL := ARule(Compose, [ [ @(1,VRCL), [Prm, @(5), @(2,L), @(6)] ], [ ConjL, @(3),
#                        @(4,L, e->let(m2:=e.params[2], n2:=@(2).val.params[2], m2*n2 = @(2).val.params[1])) ] ],
#                    e -> let(lbar := @(2).val, n:= 2*lbar.params[2], m:=2*@(4).val.params[2], v:=@(1).val.v,
#                            [ _rightPerm(m,n,v) * @(3).val]))
#));
