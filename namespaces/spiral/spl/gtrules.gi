
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_symGreater1 := o -> IsSymbolic(o) or o > 1;

Class(RulesRankedIndf, RuleSet);
RewriteRules(RulesRankedIndf, rec(
   fCompose_flatten := ARule(fCompose, [@(1,fCompose)], e->@(1).val.children()),

   # Degenerate HH rules (HH -> fAdd)
   #HH_to_fAdd := Rule([HH, @, @, @(1).cond(e->e[1]=1 and Sum(e)=1)], e -> fAdd(e.N, e.n, 0)),
   
   fAddElim := ARule(fCompose, 
       [@(1), @(2, fAdd, e -> e.params[3] = 0 and range(e) = range(@(1).val) and domain(e)=domain(@(1).val))], 
        e -> [ @(1).val ]),

   HHZ_fAdd := ARule(fCompose, [@(1, HHZ), @(2, [fAdd])], e -> When(@(2).val.params[3]=0, [ @(1).val ],
	[ HHZ(@(1).val.params[1], @(2).val.params[2], @(1).val.params[3] + @(2).val.params[3] * @(1).val.params[4][1], @(1).val.params[4])])),

   HH_fAdd := ARule(fCompose, [@(1, HH), @(2, fAdd)], e -> 
	[ HH(@(1).val.params[1], @(2).val.params[2], @(1).val.params[3] + @(2).val.params[3] * @(1).val.params[4][1],
	                           @(1).val.params[4]) ]),

   fAdd_HH := ARule(fCompose, [@(2, fAdd), @(1, HH)], e -> 
	[ HH(range(@(2).val), domain(@(1).val), @(1).val.params[3] + @(2).val.params[3], @(1).val.params[4]) ]),

   # Maude:   
   #  var b1 b2 m n MM NN : NatExp . var d : Diagf . var i : Indf . 
   #  var w ww : NatExpList . var v vv : NeNatExpList . vars gg gf : Genf .
   #  eq  H (    b1, n w) o H (    b2, m ww)  =  H (    b1 + n * b2, (n * m)  (w + lmul(n, ww))) .
   #  eq  HZ(NN, b1, n w) o H (    b2, m ww)  =  HZ(NN, b1 + n * b2, (n * m)  (w + lmul(n, ww))) .
   #  eq  HZ(NN, b1, n w) o HZ(MM, b2, m ww)  =  HZ(NN, b1 + n * b2, (n * m)  (w + lmul(n, ww))) .
   #
   HHoHH := ARule(fCompose, [@(1, HH), @(2, HH)], e -> [ let(
       s1 := @(1).val.params[4], s2 := @(2).val.params[4],
       b1 := @(1).val.params[3], b2 := @(2).val.params[3], 
       n := s1[1],  m := s2[1],  w := Drop(s1, 1),  ww := List(Drop(s2, 1), x -> n*x), 
       HH(@(1).val.params[1], @(2).val.params[2], b1 + n*b2, Concatenation([n*m], ListAddZP(w, ww)))) ]),

   # This rule is invalid when HH stride is odd, so in general it should not be used
   # NOTE: this rule is needed for DCTs, it seems that in that case stride is always even, so its ok to use it
   #        how to guard this for libraries??
   BHHoHH := ARule(fCompose, [@(1, BHH), @(2, HH, e->_symGreater1(e.params[4][1]))], e -> [ let(
       s1 := @(1).val.params[4], s2 := @(2).val.params[4],
       b1 := @(1).val.params[3], b2 := @(2).val.params[3], 
       n := s1[1],  m := s2[1],  w := Drop(s1, 1),  ww := List(Drop(s2, 1), x -> n*x), 
       BHH(@(1).val.params[1], @(2).val.params[2], b1 + n*b2, Concatenation([n*m], ListAddZP(w, ww)),
           @(1).val.params[5])) ]),

   #BHH_pull_out_base := Rule([BHH, @, @, @.cond(e->e<>0), @, @], e -> 
   #    fCompose(HH(e.params[1], e.params[1], e.params[3], [1]), BHH(e.params[1], e.params[2], 0, e.params[4], e.params[5]-2*e.params[3]))),
   
   HH_1_ftensor_pull_out := Rule([fTensor, [@(1,fCompose), [@(2,HH), @, @, @, [ListClass, _1]], ...], @(3, fId)], e->
       let(h := @(2).val, id := @(3).val.params[1], 
           fCompose(HH(id*h.params[1], id*h.params[2], id*h.params[3], [1]), 
                    fTensor(fCompose(Drop(@(1).val.rChildren(), 1)), @(3).val)))),
   
   # HHxI2_base_pull_out is for closure reduction in autolib RDFT.
   # Commented out to get smaller closure in DFT and it conflicts with HHxfId_base_pull_in rule below
   #HHxI2_base_pull_out := Rule([fTensor, [@(2,HH), @, @, @.cond(e->e<>0), @], @(3, fId)], e->
   #    let(hp := @(2).val.params, id := @(3).val.params[1], 
   #        fCompose(HH(id*hp[1], id*hp[1], id*hp[3], [1]), 
   #                 fTensor(HH(hp[1], hp[2], 0, hp[4]), @(3).val)))),

   # Pull in HH into fTensor. 
   #
   HHxfId_base_pull_in := ARule(fCompose, [[HH, @(1), @(2), @(3), [ListClass, _1]], 
       [fTensor, [@(4,HH), @, @, @, @], @(5, fId, x -> ForAll([@(1).val, @(2).val, @(3).val], e -> _divides(x.params[1], e)))]], 
       e -> let( t := @(5).val.params[1], hh := @(4).val.params,
                 [fTensor( HH( div(@(1).val, t), hh[2], hh[3] + div(@(3).val, t), hh[4] ), @(5).val )])), 

   HHoBHH := ARule(fCompose, [@(1, HH, e->e.params[3]=0 and _symGreater1(e.params[4][1])), @(2, BHH)], e -> [ let(
       s1 := @(1).val.params[4], s2 := @(2).val.params[4],
       b1 := @(1).val.params[3], b2 := @(2).val.params[3], 
       n := s1[1],  m := s2[1],  w := Drop(s1, 1),  ww := List(Drop(s2, 1), x -> n*x), 
       BHH(@(1).val.params[1], @(2).val.params[2], b1 + n*b2, Concatenation([n*m], ListAddZP(w, ww)),
           n*@(2).val.params[5])) ]),

   HHZoHH := ARule(fCompose, [@(1, HHZ), @(2, HH)], e -> [ let(
	       s1 := @(1).val.params[4],
	       s2 := @(2).val.params[4],
	       b := @(1).val.params[3], bb := @(2).val.params[3], 
	       n := s1[1], m := s2[1], w := Drop(s1, 1),  ww := List(Drop(s2, 1), x->n*x), 
	       HHZ(@(1).val.params[1], @(2).val.params[2], b + n*bb, Concatenation([n*m], ListAddZP(w, ww)))) ]),

   HHZoHHZ := ARule(fCompose, [@(1, HHZ), @(2, HHZ)], e -> [ let(
	       s1 := @(1).val.params[4],
	       s2 := @(2).val.params[4],
	       b := @(1).val.params[3], bb := @(2).val.params[3], 
	       n := s1[1], m := s2[1], w := Drop(s1, 1),  ww := List(Drop(s2, 1), x->n*x), 
	       HHZ(@(1).val.params[1], @(2).val.params[2], b + n*bb, Concatenation([n*m], ListAddZP(w, ww)))) ]),

   KHoKH := ARule(fCompose, [@(1, KH), 
                            [@(2, KH), @, @, _0, [ListClass, @, _1], [ListClass, _0, _0]]], 
       e -> [ let(
	       s1 := @(1).val.params[4],
	       s2 := @(2).val.params[4],
               corr := @(1).val.params[5],
	       b := @(1).val.params[3], bb := @(2).val.params[3], 
	       n := s1[1], m := s2[1], w := Drop(s1, 1),  ww := List(Drop(s2, 1), x->n*x), 
	       KH(@(1).val.params[1], @(2).val.params[2], b, 
                  Concatenation([n*m], ListAddZP(w, ww)), corr)) ]),

   # HHxIoHH assumes @(3) cannot be pulled into fTensor
   HHxIoHH := ARule(fCompose, [[fTensor, [@(1, HH), @, @, @, [ListClass, _1]], @(2, fId)], @(3, HH)],
       e -> let( p := @(1).val.params, n := @(2).val.params[1],
                 [ HH(n*p[1], n*p[2], n*p[3], [1]), @(3).val ] )),
   
   # eq tr(n, m) o H(0, 1 n) = H(0, m 1) .
   # H.domain must be less or equal to n
   TroHH := ARule(fCompose, 
       [@(1, Tr), @(2, HH, e -> let(n := @(1).val.params[1], Cond(AnySyms(n, e.params[2]), n=e.params[2], n>=e.params[2]) and e.params[4][1]=1 and ForAll([e.params[3]]::Drop(e.params[4],1), x -> _divides(n, x))))], 
           e -> [ let(
	       n := @(1).val.params[1], m := @(1).val.params[2], b := @(2).val.params[3], str := Drop(@(2).val.params[4], 1),
	       HH(@(2).val.params[1], @(2).val.params[2], b/n, [m] :: List(str, x -> x/n))) ]),

   Refl0_u_HH0 := ARule(fCompose, [@(1, Refl0_u), @(2, HH, e -> e.params[3]=0 and e.params[4]=[1])], e -> [ let(
	       k := @(1).val.params[1],
	       HH(@(1).val.range(), @(2).val.domain(), 0, [k])) ]),

   Refl0_u_HHrest := ARule(fCompose, [@(1, Refl0_u), @(2, HH, e -> e.params[3]=e.params[2] and e.params[4]=[1, @(1).val.params[2]])], e -> [ let(
	       k := @(1).val.params[1],
	       BHH(@(1).val.range(), @(2).val.domain(), 1, [2*k,1], 2*@(1).val.range())) ]),

   Refl0_odd_HH0 := ARule(fCompose, [@(1, Refl0_odd), @(2, HH, e -> e.params[3]=0 and e.params[4]=[1])], e -> [ let(
	       k := @(1).val.params[1],
	       HH(@(1).val.range(), @(2).val.domain(), 0, [2*k+1])) ]),

   Refl0_odd_HHrest := ARule(fCompose, [@(1, Refl0_odd), @(2, HH, e ->  e.params[4]=[1, @(1).val.params[2]])], e -> [ let(
	       k := @(1).val.params[1],
	       BHH(@(1).val.range(), @(2).val.domain(), 1, [2*k+1,1], 2*@(1).val.range())) ]),


   Refl1_HH := ARule(fCompose, [@(1, Refl1), @(2, HH, e -> e.params[3]=0 and e.params[4]=[1, @(1).val.params[2]])], e -> [ let(
	       k := @(1).val.params[1],
	       BHH(@(1).val.range(), @(2).val.domain(), 0, [2*k,1], 2*@(1).val.range() - 1)) ]),

   MMoHH := ARule(fCompose, [@(1, MM), @(2, HH, e -> e.params[3]=0 and e.params[4]=[1, @(1).val.params[1]])], e -> [ let(
	       m := @(1).val.params[2],
	       KH(@(2).val.params[1], @(2).val.params[2], 0, [m, 1], [0,0])) ]),

   fTensor_AxI_HH_stride_1 := ARule(fCompose, [[@(1, fTensor), ..., [fId, @(2)]], # NOTE: assumes d | base
                                        [@(3,HH), @, 
                                                  @.cond(e->_dividesUnsafe(@(2).val,e)), 
                                                  @.cond(e->_dividesUnsafe(@(2).val,e)), 
                                                  [ListClass, _1, ...]]], e ->
       let(p := @(3).val.params, d := @(2).val, vs := Drop(p[4],1), base := p[3],
           [ fTensor( fCompose( fTensor(DropLast(@(1).val.children(), 1)), HH(div(p[1],d), div(p[2],d), div(base,d), 
                                  Concatenation([1], List(vs, v -> div(v,d))))), fId(d))])),

   fTensor_AxI_HH_stride_n := ARule(fCompose, # NOTE: check assumptions
       [[@(1,fTensor), ..., [fId, @(2)]], 
        [@(3,HH), @, @, _0, [ListClass, @.cond(e->e=@(2).val), _0, _1]]], 

       e -> let(
	   d := @(2).val,
	   p := @(3).val.params,
	   ch := @(1).val.children(),
	   A := fTensor(DropLast(ch, 1)),
           [ fCompose(HH(range(A)*d, range(A), 0, [d, 0, 1]), A) ])
   ), 


   fTensor_HH_domain1_fId := ARule(fTensor, [ [@(1, HH), @, _1, @, @], @(2, fId) ], 
       e -> let(
	   p := @(1).val.params,
	   n := @(2).val.params[1],
	   [ HH(n*p[1], n, n*p[3], [1] :: List(Drop(p[4], 1), x->n*x)) ])
   ),

   HH_domain1_s1_to_fId1 := Rule( [HH, _1, _1, _0, [ListClass, _1]], e -> fId(1) ),

   fTensor_fId_X_HH := ARule(fCompose, [[@(4,fTensor), @(1, fId), @(2)], 
                                        [@(3,HH), @, @.cond(e->e=@(2).val.domain()), _0, [ListClass, _1, @.cond(e->e=@(2).val.domain())]]], 
       e ->let(p := @(3).val.params, vs := p[4][2],
           [ HH(@(4).val.range(), @(2).val.domain(), 0, [1, vs]), @(2).val ])),

   # This rule is a hack because it changing advdims of SPL. Range of the outer loop is also
   # assumed to be less than @(5).val (though if it's not then there is overlap in the
   # inner loop. 
   f2DTensor_HH_HH_HH :=ARule( fCompose, [
       [f2DTensor, [HH, @(1), @(2), @(3), [ListClass, _1]],
                   [HH, @(4), @(5), @(6), [ListClass, _1]]],
       [HH, @(7).cond(x->x=@(2).val*@(5).val or x=@(5).val*@(2).val), 
            @(8).cond(x->x=@(2).val), 
            _0, 
            [ListClass, @(9).cond(x->x=@(5).val), _1]]],
       e -> [HH( @(1).val*@(4).val, @(2).val, @(6).val+@(4).val*@(3).val, [@(4).val, 1] )]),
   
   # simplified merge_tensors rule
   fTensor_fTensor := ARule(fCompose, 
       [[fTensor, @(1), @(2)], [fTensor, @(3), @(4).cond(
                   e->e.range()=@(2).val.domain() or @(3).val.range() = @(1).val.domain())]],
       e -> [ fTensor(fCompose(@(1).val, @(3).val), fCompose(@(2).val, @(4).val)) ]),

   # eq crt(n, m) o H(0, 1 m) = HZ(m * n, 0, m n) .
   # eq crt(n, m) o H(0, m 1) = HZ(m * n, 0, n m) .
   CRToHH := ARule(fCompose, [ [CRT, @(1), @(2), 1, 1], 
	                       @(3,HH,e -> e.params[3]=0 and e.params[4] in [[1,@(2).val], [@(2).val, 1]]) ], e -> [ let(
		n := @(1).val,
                m := @(2).val,
		s1 := @(3).val.params[4][1],
		HHZ(@(3).val.params[1], @(3).val.params[2], 0, When(s1=1, [m, n], [n,m]))) ]),

   ## Scat * ISumn,  ISumn * Gath
   Scat_ISumn := ARule(Compose,  [ @(1, Scat), @(2, ISumn) ],
     e -> [ ISumn(@(2).val.domain, 
	          Scat(@(1).val.func.upRank()) * @(2).val.child(1)) ]),

   ISumn_Gath  := ARule(Compose, [ @(1, ISumn), @(2, Gath) ],
     e -> [ ISumn(@(1).val.domain, 
	          @(1).val.child(1) * Gath(@(2).val.func.upRank())) ]),

   ## Prm * ISumn,  ISumn * Prm
   Prm_ISumn := ARule(Compose,  [ @(1, Prm), @(2, ISumn) ],
     e -> [ ISumn(@(2).val.domain, 
	          Prm(@(1).val.func.upRank()) * @(2).val.child(1)) ]),

   ISumn_Prm  := ARule(Compose, [ @(1, ISumn), @(2, Prm) ],
     e -> [ ISumn(@(1).val.domain, 
	          @(1).val.child(1) * Prm(@(2).val.func.upRank())) ]),

   ## Diag * ISumn,  ISumn * Diag
   Diag_ISumn := ARule(Compose,  [ @(1, Diag), @(2, ISumn) ],
     e -> [ ISumn(@(2).val.domain, 
	          Diag(@(1).val.element.upRank()) * @(2).val.child(1)) ]),

   ISumn_Diag  := ARule(Compose, [ @(1, ISumn), @(2, Diag) ],
     e -> [ ISumn(@(1).val.domain, 
	          @(1).val.child(1) * Diag(@(2).val.element.upRank())) ]),

   ##UU 
   #Implementing non-zero base would probably require to have a 2D base
   #Implicit requirements 2 has base=0 and both leading dims are the same.
    UU_UU := ARule(fCompose, [@(1, UU), @(2, UU)],
        e-> let(b1X := @(1).val.params[3], b2X := @(2).val.params[3], b1Y := @(1).val.params[4], b2Y := @(2).val.params[4], s1 := @(1).val.params[5], s2 := @(2).val.params[5],
            c2 := @(2).val.params[6], ld := @(1).val.params[7], ss1 := @(1).val.params[8], 
            ss2 := @(2).val.params[8], ss:=ListAddLP(When(Length(ss2)>0,Transposed([List(Transposed(ss2)[1],x->x*s1)
                    ,Transposed(ss2)[2]]),[]), ss1),
            [UU(@(1).val.params[1], @(2).val.params[2], b1X + s1*b2X, b1Y+b2Y, s1*s2, c2, ld, ss)])),

#    UU_H := ARule(fCompose, [@(1, UU), @(2, H)],
#        e-> let(b1X := @(1).val.params[3], b2X := @(2).val.params[3], b1Y := @(1).val.params[4], s1 := @(1).val.params[5], s2 := @(2).val.params[4],
#            ld := @(1).val.params[7], ss1 := @(1).val.params[8], c1 := @(1).val.params[6], 
#            [UU(@(1).val.params[1], @(2).val.params[2], b1X + s1*b2X, b1Y, s1*s2, c1, ld, ss1)])),



));
