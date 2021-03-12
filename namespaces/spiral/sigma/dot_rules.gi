
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


isXchain := (o,dot) -> 
    Same(ObjId(o), fTensor) and Same(ObjId(o.child(1)), fId) and
    ForAll(Drop(o.children(),1), c->Same(ObjId(c), fXbase) and Same(dot, c.params[4]));

mergeXchains := function(v, c1, c2)
    local terms, i, t, other;
    terms := [ Copy(c2.child(1)) ];
    for t in Drop(c2.children(),1) do
        Add(terms, 
	    fXbase(t.params[1], t.params[2], t.params[3], v));
    od;

    other := c2.child(c2.numChildren());
    for i in [2 .. c1.numChildren()] do
       t := c1.child(i);
       Add(terms, 
	   fXbase(t.params[1], t.params[2], t.params[3] + other.params[2] + other.params[3], v));
    od;
    return fTensor(terms);
end;

# ===================================================================
# fXbase and fDot 
# ===================================================================
RewriteRules(RulesFuncSimp, rec(
  # fCompose  [[IP, @(1), J], [fTensor, fId, fBase]],
  IP_toXbase := ARule(fCompose,
     [ [ IP, @(1), @(2,J) ], [ fTensor, @(3,fId, e -> e.params[1] = @(1).val / @(2).val.params[1]), 
	                                @(4,fBase, e -> e.params[1] = @(2).val.params[1]) ] ],
     e -> let(v:=Ind(), 
	 [ fDot(v, fTensor(@(3).val, fXbase(@(2).val.params[1], @(4).val.params[2], 0, v))) ])
  ),

  IP_toYbase := ARule(fCompose,
     [ [ IP, @(1), [@(2,OS), @, -1] ], [ fTensor, @(3,fId, e -> e.params[1] = @(1).val / @(2).val.params[1]), 
	                                          @(4,fBase, e -> e.params[1] = @(2).val.params[1]) ] ],
     e -> let(v:=Ind(), 
	 [ fDot(v, fTensor(@(3).val, fYbase(@(2).val.params[1], @(4).val.params[2], 0, v))) ])
  ),

 # fCompose, [ fDot, [fTensor, fId, fXbase] ]
 #	     [ fDot, [fTensor, @,   fXbase] ]
 MergeXchains1 := ARule(fCompose,
       [ [ fDot, @(1), 
	     [ fTensor, fId, [@(10,fXbase), @, @, zero, @.cond(e->Same(e, @(1).val)) ]]], 
	 [ @(30,fDot), @(3), 
	     [ fTensor, ..., [@(20,fXbase), @, @,    @, @.cond(e->Same(e, @(3).val)) ]]] ],

 e -> let(v:=IndNR(), f:=@(10).val, g:=@(20).val,
	  dots:=DropLast(@(30).val.params[2].children(), 1),
     [ fDot(v, fTensor(dots, 
	               fXbase(g.params[1], g.params[2], g.params[3], v), 
		       fXbase(f.params[1], f.params[2], g.params[2] + g.params[3], v))) ])
 ),

 # fDot X fDot -> fDot
 MergeXchains2 := ARule(fCompose, 
    [ [ fDot, @(1), @(2).cond(e->isXchain(e,@(1).val)) ],
      [ fDot, @(3), @(4).cond(e->isXchain(e,@(3).val)) ] ],
 e -> let(v:=Ind(), 
	 [ fDot(v, fTensor( mergeXchains(v, @(2).val, @(4).val))) ])
 )
));

Class(RulesDot, RuleSet);
RewriteRules(RulesDot, rec(
  IP_toYbase := ARule(fCompose,
     [ [ IP, @(1), [@(2,OS), @, -1] ], [ fTensor, @(3,fId, e -> e.params[1] = @(1).val / @(2).val.params[1]), 
	                                          @(4,fBase, e -> e.params[1] = @(2).val.params[1]) ] ],
     e -> let(v:=Ind(), 
	 [ fDot(v, fTensor(@(3).val, fYbase(@(2).val.params[1], @(4).val.params[2], 0, v))) ])
  )
));
