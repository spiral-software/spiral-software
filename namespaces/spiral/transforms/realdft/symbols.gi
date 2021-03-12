
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# PR(<dom>, <m>, <n>, <j>) - scatter function for our older PRDFT agorithm,
#                            still used in PRDFT rule for Circulant
Class(PR, FuncClass, rec(
    def := (dom,m,n,j) -> rec(N := approx.FloorRat(m*n/2)+1, n := dom),
    lambda := self >> let(
        m := self.params[2], n := self.params[3], j := self.params[4],
        i := Ind(self.params[1]),
        Lambda(i, cond(leq(i, approx.CeilingRat(m/2)-1), i*n + j, m*n - i*n - j)))
));

#####################
# Matrices
#####################

Class(BotHalf, Sym, rec(
    def := (m, conj) -> let(mc := idiv(m+1, 2), mf := idiv(m, 2),
    DirectSum(I(2*mc), Tensor(I(mf), conj)))
));
Class(TopHalf, Sym, rec(
    def := (m, conj) -> let(mc := idiv(m+1, 2), mf := idiv(m, 2),
    DirectSum(Tensor(I(mc), conj), I(2*mf)))
));

Class(BHD, DiagFunc, rec(
    def := (m, e0, e1) -> Checked(IsPosIntSym(m), rec(size := 2*m)),
    domain := self >> 2*self.params[1],
    range := self >> TReal,
    lambda := self >> let(
	m := self.params[1], mc := idiv(m+1, 2), mf := idiv(m, 2),
	e0:=self.params[2], e1:=self.params[3], i := Ind(2*m),
	Lambda(i, cond(leq(i, 2*mc-1), 1.0, cond(neq(imod(i,2),0), e1, e0))))
));

Class(BHN, DiagFunc, rec(
    def := (m) -> Checked(IsPosIntSym(m), rec(size := m)),
    domain := self >> self.params[1],
    range := self >> TReal,
    lambda := self >> let(
    m := self.params[1], mc := idiv(m+1, 2), i := Ind(m),
    Lambda(i, cond(leq(i, mc-1), 1, -1)))
));

Class(HTwid, Sym, rec(
    def := (N, n, k, a, b, i, hconj_mat) -> let(
    na := Numerator(a),   nb := Numerator(b),
    da := Denominator(a), db := Denominator(b),
    w := Global.Conjugate(E(N*da*db)^k),
    DirectSum(List([0..n-1],
        j -> hconj_mat * Mat(RealMatComplexMat([[w ^ ((i*da+na)*(j*db+nb))]])) )))
));

X13 := (n,blk) -> DirectSum(I(1), Tensor(I(n-1), blk)) ^ (Z(2*n-1,-1)*LIJ(2*n-1));
SymSplit1 := n -> DirectSum(I(n+1), J(n-1)) * DirectSum(I(1), X13(n,F(2)));
SymSplit3 := n -> IJ(2*n,n)   * DirectSum(I(1), X13(n,J(2)*F(2)));

X24 := (n,blk) -> Tensor(blk, I(n)) * IJ(2*n,n);
SymSplit2 := n -> X24(n,F(2));
SymSplit4 := n -> X24(n,J(2)*F(2));

rpdiag := p -> Tensor(pdiag(Int(p/2)+1), I(2));
rperm_ev := p -> Tensor(J((p+1)/2), Diag(1,-1));
rperm_od := p -> Tensor(J((p+1)/2), Diag(-1,1));


# YSV: Many rules below turned out to be invalid for odd stride in H, so I added guards
#
RewriteRules(RulesFuncSimp, rec(
 BHD_Stride_fId := ARule(fCompose,
     [ [BHD, @(1),@(2),@(3)], [fTensor, [H,@,@(4),@,@.cond(e->IsEvenInt(e) and e>1)], [fId, 2]] ],
     e -> [BHD(@(4).val, @(2).val, @(3).val)] ),

 BHD_HH_fId := ARule(fCompose,
     [ [BHD, @(1),@(2),@(3)], [fTensor, [HH,@,@(4),_0,[ListClass, @.cond(e->IsEvenInt(e) and e>1), _1]], [fId, 2]] ],
     e -> [BHD(@(4).val, @(2).val, @(3).val)] ),

 BHD_Stride_H := ARule(fCompose,
     [ [BHD, @(1),@(2),@(3)], [fTensor, [H,@,@(4),@,@.cond(e->IsEvenInt(e) and e>1)], [H,2,2,0,1]] ],
     e -> [BHD(@(4).val, @(2).val, @(3).val)] ),

 BHN_H := ARule(fCompose,
     [ BHN, [@(1,H),@,@,@, @.cond(e->IsEvenInt(e) and e>1)] ],
     e -> [BHN(@(1).val.domain())] ),

# ******* TOGGLE THIS RULE
 # NOTE: IsSymbolic in the condition is dangerous! this rule is not valid if stride is odd.
 #        which is precisely what happens with ASPF_CT1Odd_RFT rule, but this rule is still needed for DCTs
# BHN_HH := ARule(fCompose,
#     [ BHN, [@(1,HH),@,@,_0,[ListClass, @.cond(e->IsSymbolic(e) or (IsEvenInt(e) and e>1)), _1]] ], # vector stride = 1 means stride <> 1
#     e -> [BHN(@(1).val.domain())] ),
##############

# NOTE: breaks for odd sizes -> guard
 BHN_fTensor := ARule(fCompose,
     [ BHN, [@(1, fTensor), @(2,fId,e->IsEvenInt(e.range())), @(3,fBase,e->IsEvenInt(e.range()))] ],
     e -> [BHN(@(1).val.domain())] ),

 BHD_fTensor := ARule(fCompose,
     [ @(0,BHD), [@(1, fTensor), @(2,fId,e->IsEvenInt(e.range())), @(3,fBase,e->IsEvenInt(e.range())), [fId, 2]] ],
     e -> [BHD(@(2).val.params[1], @(0).val.params[2], @(0).val.params[3])] ),

 ## BH o H -> BH
 BH_H := ARule(fCompose, [ [@(1,BH), @N, @r, @n, @b, @s], [H, @n, @m, @bb, @(2).cond(e->e>1 and IsEvenInt(e))] ],
     (e, cx) -> [ BH(@N.val, @r.val, @m.val,
                 memo(cx, "g", @b.val + @s.val * @bb.val), # new base
             @s.val * @(2).val) ]),                     # new stride

 ## H o BH -> BH
 H_BH := ARule(fCompose, [ [H, @N, @n, @b, @(2).cond(e->e>1)], [@(1,BH), @n, @r, @m, @bb, @ss] ],
     (e, cx) -> [ When(@b.val<>0, H(@N.val,@N.val,@b.val, 1), fId(@N.val)),
              BH(@N.val, @r.val * @(2).val, @m.val,
                 memo(cx, "g", @(2).val * @bb.val), # new base
             @(2).val * @ss.val) ]),            # new stride

));
