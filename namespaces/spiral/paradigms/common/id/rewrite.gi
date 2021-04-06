
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(RulesId, RuleSet);

RewriteRules(RulesId, rec(

 IdISumRight := ARule( Compose,
       [ @(1, Id), @(2, ISum, canReorder) ],
  e -> [ ISum(@2.val.var, @2.val.domain, @1.val * @2.val.child(1)).attrs(@(2).val) ]),

 IdISumLeft := ARule( Compose,
       [ @(1, ISum, canReorder), @(2, Id) ],
  e -> [ ISum(@1.val.var, @1.val.domain, @1.val.child(1) * @2.val).attrs(@(1).val) ]),

 IdIComposeRight := ARule( Compose,
       [ @(1, Id), @(2, ICompose, canReorder) ],
  e -> [ ICompose(@2.val.var, @2.val.domain, @1.val * @2.val.child(1)).attrs(@(2).val) ]),

 IdIComposeLeft := ARule( Compose,
       [ @(1, ICompose, canReorder), @(2, Id) ],
  e -> [ ICompose(@1.val.var, @1.val.domain, @1.val.child(1) * @2.val).attrs(@(1).val) ]),

 # Gath * Id
 CommuteGathId := ARule( Compose,
       [ @(1, Gath), @(2, Id) ], # o 1-> 2->
  e -> [ Id(fCompose(@2.val.element, @1.val.func)).attrs(@(2).val), @1.val ]),

 # Prm * Id
 CommutePrmId := ARule( Compose,
       [ @(1, Prm), @(2, Id) ], # o 1-> 2->
  e -> [ Id(fCompose(@2.val.element, @1.val.func)).attrs(@(2).val), @1.val ]),

 # Id * Scat
 CommuteIdScat := ARule( Compose,
       [ @(1, Id), @(2, Scat) ], # <-1 <-2 o
  e -> [ @2.val, Id(fCompose(@1.val.element, @2.val.func)).attrs(@(1).val) ]),

  #(A X B) o (C X D) -> (A o C) X (B o D)
  CommuteIdTensorfTensor := Rule([fCompose, [idTensor, @(1), @(2)],
      [@(5, fTensor).cond(e->Length(e.children()) = 2), @(3).cond(e->e.range() = @1.val.domain()), @(4).cond(e->e.range() = @2.val.domain())]],
      e -> idTensor(fCompose(@(1).val, @(3).val), fCompose(@(2).val, @(4).val))
  ),

 IdNoDiagPullinRight := ARule( Compose,
       [ @(1, Id), @(2, NoDiagPullinRight) ],
  e -> [ NoDiagPullinRight(@(1).val * @2.val.child(1)) ]),

 IdNoDiagPullinLeft := ARule( Compose,
       [ @(1, NoDiagPullinLeft), @(2, Id) ],
  e -> [ NoDiagPullinLeft(@1.val.child(1) * @(2).val) ]),


#(A X B) o ((C X D) X E) -> (A o (C X D)) X (B o E)
  CommuteIdTensorfTensor2 := Rule([fCompose, [idTensor, @(1), @(2)],
      [@(3, fTensor).cond(e->
           Length(e.children()) = 3 and
       fTensor(e.child(1), e.child(2)).range() = @1.val.domain() and
       e.child(3).range() = @2.val.domain()
       ),
       @(4), @(5), @(6)
      ]], e->

      idTensor(fCompose(@(1).val, fTensor(@(3).val.child(1), @(3).val.child(2))),
               fCompose(@(2).val, @(3).val.child(3)))
      ),


  # i o (j)n = (j)1->N, where i is an idId
  IdComposeConst := Rule([fCompose, @(1, idId), @(2, fBase)],
                    e -> idConst(1, @2.val.params[2])
  ),
  
  idTensorId1 := ARule(idTensor, [@(1), [@(2,[idId]), 1]], e -> [@(1).val]),
  idTensorId2 := ARule(idTensor, [[@(2,[idId,J]), 1], @(1)], e -> [@(1).val]),

  # This is a special-puprose modification of the IdComposeConst rule, above.
  # It's possible that this is not the best way to do this.  I'd be willing
  # to change it later (Peter).
  # i o ((j)n X (k)m ) = (j)1->N X (k)1->N
#  IdComposeTensConst := Rule([fCompose,
#     @(1, idId),
#     @(2, fTensor).cond(e->
#         Length(e.children()) = 2  and
#         ObjId(e.child(1)) = fBase and
#         ObjId(e.child(2)) = fBase)],
#      e -> @(2).val
#  ),


  # id(a x b) = id(c) where c is derived from merging a and b and
  # computing corresponding domains and ranges.
  FoldConstants := ARule(idTensor, [@(1, idConst), @(2, idConst)],
      e -> [idConst(@1.val.domain()*@2.val.domain(), @1.val.params[2]*@2.val.params[2])]
  )

));
