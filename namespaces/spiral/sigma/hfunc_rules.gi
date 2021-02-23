
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(RulesHfunc, RuleSet, rec(__avoid__ := [])); #Diag]));

# IsInt(range(e)) makes sure that we don't apply this to functions with range of TInt, 
# which are special. Otherwise 
# the following breaks eventually
# RC(Diag(fPrecompute(fCompose(dOmega(6, 1), diagTensor(fCompose(dLin(V(2), 1, 0, TInt), 
#                       fBase(i601)), dLin(3, 1, 0, TInt)), H(3, 1, 0, 1))))), 
# because this rule matches
base := pat -> pat.cond(e->domain(e)=1 and IsInt(range(e))); 

unmemo := exp -> When(IsVar(exp) and IsBound(exp.mapping), exp.mapping, exp);
SetRange := (exp,r) >> When(IsVar(exp), exp.setRange(r), exp);

RewriteRules(RulesHfunc, rec(
 # fId -> H
 #Hid := Rule(fId, e -> H(e.params[1], e.params[1], 0, 1)),

 # fTensor -> H o f
 Tensor_toH1 := ARule(fTensor, [ @F, base(@(1)) ], 
     (e,cx) -> [ fCompose(H(range(@(1).val) * range(@F.val), 
		            range(@F.val), 
		            nomemo(cx, "b", @(1).val.lambda().at(0)), 
			    range(@(1).val)),
	                  @F.val) ]),
 # fTensor -> H o f
 Tensor_toH2 := ARule(fTensor, [ base(@(1)), @F ],
     (e,cx) -> [ fCompose(H(range(@(1).val)*range(@F.val), 
		            range(@F.val), 
		            nomemo(cx, "b", range(@F.val) * @(1).val.lambda().at(0)), 
			    1),
	                  @F.val) ]),
 # gammaTensor -> HZ o f
 GammaTensor_toHZ1 := ARule(gammaTensor, [ @F, base(@(1)) ],
     (e, cx) -> let(F := @F.val, b := @(1).val, j := b.lambda().at(0), 
	 [ fCompose(HZ(range(b)*range(F), range(F), memo(cx,"g",range(F)*j), range(b)), F) ])), 

 # gammaTensor -> HZ o f, same RHS as in rule above
 GammaTensor_toHZ2 := ARule(gammaTensor, [ base(@(1)), @F ], #~.GammaTensor_toHZ1[3]),
     (e, cx) -> let(F := @F.val, b := @(1).val, j := b.lambda().at(0), 
	 [ fCompose(HZ(range(b)*range(F), range(F), memo(cx,"g",range(F)*j), range(b)), F) ])), 

 # H o H -> H
 H_H := ARule(fCompose, [ [H, @N, @n, @b, @s], [H, @n, @m, @bb, @ss] ],
     (e, cx) -> [ H(@N.val, @m.val,
	            memo(cx, "b", @b.val + @s.val * @bb.val), # new base
		    @s.val * @ss.val) ]),                     # new stride
 ## HZ o (H | HZ) -> HZ 
 HZ_H := ARule(fCompose, [ [HZ, @N, @n, @b, @s], [@(1,[H,HZ]), @n, @m, @bb, @ss] ],
     (e, cx) -> [ HZ(@N.val, @m.val,
	             memo(cx, "g", @b.val + @s.val * @bb.val), # new base
		     @s.val * @ss.val) ]),                     # new stride
 # RM o (H | HZ) -> RM  ("rho" in the PLDI paper)
 RM_H := ARule(fCompose, [ [RM, @N, @n, @phi, @g], [@(1,[HZ,H]), @n, @m, @b, @s] ],
     (e, cx) -> [ RM(@N.val, @m.val,
	             SetRange(memo(cx, "f", powmod(@phi.val,@g.val,unmemo(@b.val),@N.val)), @N.val),
		     powmod(1, @g.val, @s.val, @N.val).eval()) ]),

# Following rule is for use in expressions like Gath(fBase(i))
 H_fBase := Rule(@(1,fBase),e ->H(@(1).val.params[1],1,@(1).val.params[2],1)),

#Handle (H tensor I) o H
# H_tensorI := ARule(fCompose,[[fTensor,[H,@(1),@(2),@(3),@(4)],[fId,@(5)]],[H,@.cond(e->e=@(2).val*@(5).val),1,@(6),@]],
#e->[H(@(1).val*@(5).val,1,(@(3).val+@(4).val*idiv(@(6).val,@(5).val))*@(5).val+imod(@(6).val,@(5).val),1)]),

));

