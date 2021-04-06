
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(RulesRC, RuleSet);

RCExpandData := function(datavar)
   local rdatavar, r, i;
   Constraint(IsBound(datavar.value));
   rdatavar := Dat1d(TDouble, datavar.t.size*2);
   r := x->When(IsValue(x), ReComplex(Complex(x.v)), re(x));
   i := x->When(IsValue(x), ImComplex(Complex(x.v)), im(x));
   rdatavar.value := V(ConcatList(datavar.value.v, x->[r(x), i(x)]));
   return rdatavar;
end;

RewriteRules(RulesRC, rec(
 RC_Diag := Rule([RC, @(1,Diag)], e -> RCDiag(RCData(@(1).val.element))),

 RC_Compose := Rule([RC, @(1, Compose)], e -> Compose(List(@(1).val.children(), RC))),
 RC_SUM := Rule([RC, @(1, SUM)], e -> SUM(List(@(1).val.children(), RC))),
 RC_SUMAcc := Rule([RC, @(1, SUMAcc)], e -> SUMAcc(List(@(1).val.children(), RC))),

 RC_Container := Rule([RC, @(1, [BB,Buf,Inplace,Grp,NoPull,NoPullLeft,NoPullRight, NoDiagPullin, NoDiagPullinLeft, NoDiagPullinRight ])], e -> ObjId(@(1).val)(RC(@(1).val.child(1)))),

 RC_RStep := Rule([RC, @(1, RecursStep)], e -> RecursStep(2*@(1).val.yofs, 2*@(1).val.xofs,
                                                      RC(@(1).val.child(1)))),

 RC_Data := Rule([RC, @(1, Data)], e -> Data(@(1).val.var, @(1).val.value,
                                        RC(@(1).val.child(1))).attrs(@(1).val)),

 RC_ISum := Rule([RC, @(1, ISum)], e -> ISum(@(1).val.var, @(1).val.domain, RC(@(1).val.child(1)))),

 RC_ICompose := Rule([RC, @(1, ICompose)], e -> ICompose(@(1).val.var, @(1).val.domain, RC(@(1).val.child(1)))),

 RC_Gath := Rule([RC, @(1, Gath)], e -> Gath(fTensor(@(1).val.func, fId(2)))),
 RC_Scat := Rule([RC, @(1, [Scat, ScatAcc])], e -> ObjId(@(1).val)(fTensor(@(1).val.func, fId(2)))),
 RC_Prm := Rule([RC, @(1, Prm)], e -> Prm(fTensor(@(1).val.func, fId(2)))),

 RC_RowColVec := Rule([RC, @(1, [RowVec, ColVec], e->not IsSymbolic(e.element.domain()))], 
     e -> RC(@(1).val.toDiagBlk())), 

# RC_ScatGath := Rule([RC, @(1, ScatGath)], e -> ScatGath(fTensor(@(1).val.sfunc, fId(2)), fTensor(@(1).val.gfunc, fId(2)))),

 RC_O := Rule([RC, @(1, O)], e -> O(2*@(1).val.params[1], 2*@(1).val.params[2])),

 # yes, this does happen
 RC_RCDiag := Rule([RC, @(1, RCDiag)], e -> let(
     func := @(1).val.element,
     D := Dat1d(func.range(), func.domain()),
     j := Ind(func.domain()/2),
     re := nth(D, 2*j),
     im := nth(D, 2*j+1),
     Data(D, func,  # use Data, to preserve fPrecompute that might be present in <func>
          IterDirectSum(j, RC( Blk([[re, -im], [im, re]]) )).sums()))),

 # NB: add a rule for barring real diagonals with replication:
 #    RCData(FData(..real..) o f) -> FData o (f x proj(2))
 RCData_fCompose_FData := Rule([RCData, [fCompose, @(1,FData), @(2)]],
     e -> fCompose(FData(RCExpandData(@(1).val.var)), fTensor(@(2).val, fId(2)))),

 RCData_FData := Rule([RCData, @(1, FData, x -> IsBound(x.var.value) or x.var.t.t=TReal or ObjId(x.var.t.t)=T_Real)],
     e -> let(t := @(1).val.var.t.t,
	 Cond(t = TReal or ObjId(t) = T_Real, 
	     diagTensor(FData(@(1).val.var), fConst(t, 2, t.one())),
	     FData(RCExpandData(@(1).val.var))))),

 # NOTE: works for real constants only
 RCData_fConst := Rule([RCData, [@(1,fConst), 1, @(2)]],
     e -> FList(TReal, [ReComplex(Complex(@(2).val)), ImComplex(Complex(@(2).val))])),

 RCData_fPrecompute := Rule([RCData, [fPrecompute, @(1)]], e -> fPrecompute(RCData(@(1).val))),

# YSV: Feb07, lets see what happens without this rule!
# #  RCData_fCompose := Rule([RCData, @(1, fCompose)],
# #      e -> let(ch := @(1).val.children(),
# #        fCompose(RCData(ch[1]), fTensor(fCompose(Drop(ch, 1)), fId(2))))),

 RCData_diagDirsum := Rule([RCData, @(1,diagDirsum)],
      e -> ApplyFunc(diagDirsum, List(@(1).val.children(), RCData))),

# RCData_fCompose_FDataOfs := Rule([RCData, [FDataOfs, @(1), @(2), @(3)]],
#     e -> FDataOfs(RCExpandData(@(1).val), 2 * @(2).val, 2 * @(3).val)),

 RC_Blk := Rule([RC, @(1, Blk)], e -> Blk(RealMatComplexMat(@(1).val.element))),
 RC_Blk1 := Rule([RC, @(1, Blk1)], e -> Blk(RealMatComplexMat([[@(1).val.element]]))),

 RC_Scale := Rule([RC, @(1, Scale)], e ->
     RC(Diag(fConst(Rows(@(1).val), @(1).val.scalar))) * RC(@(1).val.child(1))),

 RC_I := Rule([RC, @(1, I)], e -> I(2*Cols(@1.val))),

 RC_OLMultiplication := Rule([RC, @(1, OLMultiplication)],
     e -> let( p := @(1).val.rChildren(), RCOLMultiplication(p[1], p[2]))),
 RC_OLConjMultiplication := Rule([RC, @(1, OLConjMultiplication)],
     e -> let( p := @(1).val.rChildren(), RCOLConjMultiplication(p[1], p[2]))),

));
