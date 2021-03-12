
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# ================================================================
# Sum of Monomials
# NOTE: fix sum_mons, mon_diags, mon_perms
# ================================================================
sum_mons := pat -> pat.target(SUMAcc).cond(e->ForAll(e.children(), IS(Mon)));
mon_vdiags := pat -> List(pat.val.children(), c -> c.vdiag);
mon_sdiags := pat -> List(pat.val.children(), c -> c.sdiag);
mon_perms := pat -> List(pat.val.children(), c -> c.perm);

compat_tensor_mons := (p,v,s) -> let(
    pp := Filtered(p, c->domain(c) > 1),
    vv := Filtered(v, c->domain(c) > 1),
    ss := Filtered(s, c->domain(c) > 1),
    fully_compat_tensor_chains(pp, vv, domain, domain) and
    fully_compat_tensor_chains(vv, ss, domain, domain));

merge_tensor_mons := function(pp, vv, ss, combine) #, fid, gid, hid)
   local i, j, k, np, nv, ns, p, v, s, res;
   np := Length(pp);    
   nv := Length(vv);
   ns := Length(ss);
   res := []; i:=1; j:=1; k := 1;

   while i <= np or j <= nv or k <= ns do
       if (i <= np and j <= nv and k <= ns) and 
           domain(pp[i]) = domain(vv[j]) and
           domain(vv[j]) = domain(ss[k]) 
       then
           Add(res, combine(pp[i], vv[j], ss[k]));
           i := i+1;
           j := j+1;
           k := k+1;
       else
           p := fId(1);
           v := II(1);
           s := II(1);

           if i<=np and domain(pp[i]) = 1 then
               p := pp[i]; i:=i+1; fi;
           if j<=nv and domain(vv[j]) = 1 then
               v := vv[j]; j:=j+1; fi;
           if k<=ns and domain(ss[k]) = 1 then
               s := ss[k]; k:=k+1; fi;
           
           Add(res, combine(p,v,s));
       fi;
   od;
   return res;
end;

#UseRuleSet(RulesSums);

# pulling in sum of monomials into ISum
ARule( Compose, 
       [ @(1, ISum), sum_mons(@(2)) ],	
  e -> [ ISum(@(1).val.var, @(1).val.domain, @(1).val.child(1) * @(2).val).attrs(@(1).val) ]);

# Gath * SUMAcc(Mon, Mon,...) -> SUMAcc(GMon, GMon, ...)
ARule( Compose, 
       [ [Gath, @(1)], sum_mons(@(2)) ],	
  e -> [ SUMAcc( 
	  let(vdiags := mon_vdiags(@(2)),
	      sdiags := mon_sdiags(@(2)),
	      perms  := mon_perms(@(2)),
	      numdiags := Length(vdiags),

	      List([1..numdiags], 
		  i -> GMon(fCompose(perms[i], @(1).val),   
		            fCompose(vdiags[i], perms[i], @(1).val),
		            fCompose(sdiags[i], perms[i], @(1).val))))) ]
);

# low-priority suck in of Gath into SUMAcc
ARule( Compose, 
       [ @(1, SUMAcc), @(2, Gath) ],
  e -> [ ApplyFunc(ObjId(@(1).val), 
	           List(@(1).val.children(), c -> c * @(2).val)) ]);


# Combining GMon and Gath, SMon and Scat, Mon and Prm
ARule( Compose,
       [ @(1, GMon), @(2, Gath) ], # o 1-> 2->
  e -> [ GMon(fCompose(@(2).val.func, @(1).val.perm), 
	      @(1).val.vdiag, @(1).val.sdiag) ] );

ARule( Compose,
       [ @(1, Scat), @(2, SMon) ], # o 1-> 2->
  e -> [ GMon(fCompose(@(1).val.func, @(2).val.perm), 
	      @(2).val.vdiag, @(2).val.sdiag) ] );
	    
ARule( Compose,
       [ @(1, Mon), @(2, Prm) ], # o 1-> 2->
  e -> [ GMon(fCompose(@(2).val.func, @(1).val.perm), 
	      fCompose(@(1).val.vdiag, @(2).val.func.transpose()),
	      fCompose(@(1).val.sdiag, @(2).val.func.transpose())) ]);

#UseRuleSet(RulesFuncSimp);

# ================================================================
# Simplification of Intervals
# ================================================================

full_interval := (int, size) -> 
    [int.target(II), size, zero, @.cond(e->e=size.val)];

left_interval := (int, size, endd) -> 
    [int.target(II), size, zero, endd];

right_interval := (int, size,start) -> 
    [int.target(II), size, start, @.cond(e->e=size.val)];

# full interval o index mapping func @(3) = interval with domain of @(3)
# II(n, 0, n) o f = II(domain(f), 0, domain(f))
ARule( fCompose,
      [ full_interval(@, @(1)), @(2) ],
 e -> [ II(pdomain(@(2))) ]
);

# partial interval o dirsum of perms
# II(n, 0, k) o fDirsum(perm_k, perm_?) -> II(n,0,k)
ARule( fCompose, 
      [ left_interval(@(0), @(1), @(2)), 
	[fDirsum, @(3).cond(e -> is_perm(e) and range(e)=@(2).val), ...] ],
 e -> [ @(0).val ]
);

# partial interval o dirsum of perms
# II(n, n-k, n) o fDirsum(perm_?, perm_k) -> II(n,n-k,n)
ARule( fCompose, 
      [ right_interval(@(0), @(1), @(2)),
	[fDirsum, ..., @(4).cond(e -> is_perm(e) and range(e) = @(1).val-@(2).val)] ],
 e -> [ @(0).val ]
);


# fTensor(IIn, IIk) -> II(n*k)
ARule( fTensor, [full_interval(@(0), @(1)), full_interval(@(0), @(2))],
 e -> [ II(@(1).val * @(2).val) ]
);


# fDirsum(IIn, IIk) -> II(n+k)
ARule( fDirsum, [full_interval(@(0), @(1)), full_interval(@(0), @(2))],
 e -> [ II(@(1).val + @(2).val) ]
);

# Interval Shifting, we support shift left (shift <= start), and right (shift >= end)
# II o Z = shifted II# 
ARule( fCompose, 
      [ [II, @(1), @(2), @(3)], 
	[Z,  @(4), @(5).cond(e -> (e<=@(2).val) or (e >= @(3).val)) ] 
      ],
 e -> [	II(@(1).val, (@(2).val-@(5).val) mod @(1).val, 
           # this makes 0->size, because size mod size = 0
	   let(last:=(@(3).val-@(5).val) mod @(1).val,  
	       When(last=0, @(1).val, last))) ]
);
    
# full_interval_diag * mat = mat
ARule( Compose, 
      [ [Diag, [II, @(1), zero, @(2).cond(e->e=@(1).val)]], @(3) ],
 e -> [ @(3).val ]
);

# mat * full_interval_diag  = mat
ARule( Compose, 
      [ @(3), [Diag, [II, @(1), zero, @(2).cond(e->e=@(1).val)]] ],
 e -> [ @(3).val ]
);


# ================================================================
# Sum of Monomials
# ================================================================
# *******************************************************************
Class( RulesGMonTensorPullOut, RuleSet );
# *******************************************************************

# GMon splitting
# GMon(fTensor, fTensor, fTensor) -> Tensor(GMon, GMon, ...)
#
tensor_ch := o -> When(IS_A(o, fTensor), o.children(), [o]);

Rule([GMon,  
        @(1),
        @(2), 
        @(3).cond(e -> 
	    let(p := @(1).val, v := @(2).val, s := e, 
	        ForAny([p,v,s], IS(fTensor)) and
	        compat_tensor_mons(tensor_ch(p), tensor_ch(v), tensor_ch(s))))],
 e -> let(
     cp := tensor_ch(@(1).val),
     cv := tensor_ch(@(2).val),
     cs := tensor_ch(@(3).val),
     Tensor( merge_tensor_mons(cp, cv, cs, GMon) ))
);

# ================================================================
# GMon -> SGMon
#  (delta * gath)
# ================================================================
# *******************************************************************
Class( RulesGMonToSGMon, RuleSet );
# *******************************************************************

Rule( [ GMon, @(1), @(2), full_interval(@(3),@(4)) ],
    e -> SGMon(fId(domain(@(1).val)), # scatter
	       @(1).val, # gather
	       @(2).val, # vdiag,
	       @(3).val));

## Diag(II) * Diag(..) * Gath(fId)
Rule( [ GMon, @(1,fId), @(2), @(3,II) ],
 e -> let(left := @(3).val.params[2], 
          right:= @(3).val.params[3],
          size := @(3).val.params[1],
	  fadd := fAdd(size, right-left, left),
    SGMon(fadd, fadd, 
          fCompose(@(2).val, fadd), II(right-left)))
);

## Diag(II) * Diag(..) * Gath(Z)
Rule( [ GMon, 
	[Z, @n, @z],
	@(1), 
	[II, @, @l, @(3).cond(r -> (@z.val <= @n.val-r) or 
		                   (@z.val >= @n.val-@l.val)) ] ],
e -> let(size  := @n.val, 
	 left  := @l.val, 
         right := @(3).val, 
	 shift := @z.val,
	 scat := fAdd(size, right-left, left),
	 gath := fAdd(size, right-left, (left + shift) mod size),
    SGMon(scat, gath, 
	  fCompose(@(1).val, scat), II(domain(gath))))
);


## Diag(II o fBase) * Gath(fBase)
Rule([GMon, 
	@(1,fBase), 
	@(2), 
	[fCompose, @(3,II), same_params(@(4),fBase,@(1)) ] ],

 e -> let(left := @(3).val.params[2], right := @(3).val.params[3], 
          j    := @(1).val.params[2], 
	  gath := @(1).val,
	  COND(leq(left, j, right-1), 
	       SGMon(fId(1), gath, @(2).val, II(1)),
	       O(1, range(gath))))
);

## Diag(II o fBase) * Gath(fId(1))
Rule([skip, 
	@(1,fId,e->range(e)=1),
	@(2), 
	[fCompose, @(3,II), @(4,fBase) ] ],

 e -> let(left := @(3).val.params[2], right := @(3).val.params[3], 
          j    := @(4).val.params[2], 
	  COND(leq(left, j, right-1), 
	       SGMon(fId(1), fId(1), @(2).val, II(1)),
	       O(1, 1)))
);

# NOTE:Make direct sum rules more general (... + p + ...) o fBase
#
## Diag(leftII o fBase) * Gath((p + ...) o fBase) 
Rule([GMon, 
        [fCompose, 
	    [fDirsum, @(1).cond(e -> is_perm(e)), ...],
	    @(2,fBase) ],
	@(3),
	[fCompose, left_interval(@(10), @(20), @(4).cond(e->range(@(1).val)=e)), 
	           same_params(@(5), fBase, @(2))] ],

 e -> let(right := @(4).val,
          size  := @(2).val.params[1],
          j     := @(2).val.params[2], 
	  COND(leq(0, j, right-1), 
	       SGMon(
		   fId(1),
		   fCompose(fAdd(size, right, 0), @(1).val, fBase(right, j)),
		   @(3).val,
		   II(1)),
	       O(1, size)))
);

## Diag(rightII o fBase) * Gath((... + p) o fBase)
Rule([GMon, 
        [fCompose, 
	    [fDirsum, ..., @(1).cond(e -> is_perm(e))],
	    @(2,fBase) ],
	@(3),
	[fCompose, right_interval(@(0), @(4), @(5).cond(e->range(@(1).val) = @(4).val - e)), 
	           same_params(@(6), fBase, @(2)) ] ],

 e -> let(left := @(4).val,
          size  := @(2).val.params[1],
          j     := @(2).val.params[2], 
	  COND(leq(left, j, size-1), 
	       SGMon(
		   fId(1),
		   fCompose(fAdd(size, size-left, left), @(1).val, fBase(size-left, j-left)),
		   @(3).val,
		   II(1)),
	       O(1, size)))
);
    
# ================================================================
# Going back from Tensor(SGMon, ...) into SGMon(fTensor,...)
# ================================================================

Rule([@(0,Tensor), ..., @(1,COND), ...],
    e -> let( ch := e.children(), 
	      left := Left(...), 
	      right := Right(...),
	      leftch := ch{[1..left]},
	      rightch := ch{[right..Length(ch)]},
	      OP := ObjId(@(0).val),
	      COND(@(1).val.cond,
		  List(@(1).val.children(), 
		       c -> OP(Concatenation(Copy(leftch), [c], Copy(rightch)))))));

# *******************************************************************
Class( RulesSGMonTensorPullIn, RuleSet );
# *******************************************************************

Rule( @(1,Tensor,CHILDREN_ARE(SGMon)),
 e -> let( ch:=e.children(),
     SGMon(
	 fTensor(List(ch, c->c.scat)),
	 fTensor(List(ch, c->c.gath)),
	 fTensor(List(ch, c->c.vdiag)),
	 fTensor(List(ch, c->c.sdiag)))));
	     
# ================================================================
# Finalization
# ================================================================
Class( RulesFinalize, RuleSet );

#z:=SUMAcc(Gath(fId(4)), Scat(fAdd(4,3,0)) * Gath(fAdd(4,3,0)));

Elim := (elim_func, s) -> When(elim_func(s), [], s);

ARule(SUMAcc, 
    [ [Gath, @R],
      [Compose, 
	  [@(1,Scat), [@(2,fAdd), @, @, @k]], 
	  [@(3,Gath), @S]] ],

    e -> let(n   := domain(@R.val), N   := range(@R.val), 
	     nmk := domain(@S.val), nm  := nmk + @k.val, 
	     m := n - nm, 
    [ SUM( Elim(e->@k.val=0, Scat(fAdd(n, @k.val, 0)) * Gath(fCompose(@R.val, fAdd(n, @k.val, 0)))),
	   Scat(Copy(@(2).val)) * 
	   SUMAcc(
	       Gath(fCompose(@R.val, fAdd(n, nmk, @k.val))),
	       Gath(@S.val)),
	   Elim(e->m=0, Scat(fAdd(n, m, nm)) * Gath(fCompose(@R.val, fAdd(n, m, nm))))
      ) ])
);

ARule(Compose,
    [ [Scat, @W], 
      @(1, SUMAcc, e->ForAll(e.children(), c->ObjId(c)=Gath and domain(c.func)>1)) ],

  e -> [ let(n := domain(@W.val), 
	     i := Ind(n),
	     fbase := fBase(n, i),
	  ISum(i, n, 
	      Scat(fCompose(@W.val, fbase)) *
	      SUMAcc(List(@(1).val.children(), c -> Gath(fCompose(c.func, fbase)))))) ]
);

   
ARule(Compose,
    [ [Scat, @W], [Diag, @D], [Gath, @R] ],

  e -> [ let(n := domain(@R.val), 
	     i := Ind(n),
	     fbase := fBase(n, i),
	  ISum(i, n, 
	      Scat(fCompose(@W.val, fbase)) *
	      Blk1(@D.val.lambda().at(i)) *
	      Gath(fCompose(@R.val, fbase)))) ]
);

ARule(Compose, [@(1, O), @(2)], e->[ @(1).val ]);

Rule([Compose, [Scat, @(1)], @(2, O)], e->
    let(n := @(1).val.domain(),
	i := Ind(n),
	fbase := fBase(n,i),
	ISum(i, n,
	     Scat(fCompose(@(1).val, fbase)) * 
	     Blk1(0) * 
	     Gath(fCompose(fId(Cols(@(2).val)), fbase))))
);
