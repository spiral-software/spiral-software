
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(TPrmMulti,CodeBlock);
IsTwoPower := i >> 2 ^ Log2Int(i) = i;

get_l_power := function(t,exp)
  local p;
  if (exp=0) then
    p := L(2^t,1);
  else
    p:= L(2^t,2^(t-exp));
  fi;
  return p;   
end;

# Declaration of SortBase, a 2x2 sorter.
Class(SortBase, BaseMat, rec(
   abbrevs   := [()-> []],
   new       := (self) >> SPL( WithBases(self, rec()) ).setDims(),
   dims      := self >> [ 2, 2 ],
   isReal    := True,
   sums     := self >> self,
   rChildren := self >> [],
   rSetChild := rSetChildFields(),
   toAMat := self >> Error("not supported"),
   transpose := self >> self,
));

# Declaration of SortBase, a 1x1 sorter.
Class(SortBase_w1, BaseMat, rec(
   abbrevs   := [(a)-> [a]],
   new       := (self, a) >> SPL( WithBases(self, rec(dimensions:=[1,1], a := a))),
   print := (self, i, is) >> Print(self.name, "(", self.a, ")"),
   dims      := self >> [ 1, 1 ],
   isReal    := True,
   sums     := self >> self,
   rChildren := self >> [self.a],
   rSetChild := rSetChildFields("a"),
   toAMat := self >> Error("not supported"),
   transpose := self >> self,
));

Class(SortConfigBase_w1, BaseMat, rec(
   abbrevs   := [(a)-> [a] , (b)-> [b]],
   new       := (self, a, b) >> SPL( WithBases(self, rec(dimensions:=[1,1], a := a, b:=b))),
   print := (self, i, is) >> Print(self.name, "(", self.a, ",", self.b, ")"),
   dims      := self >> [ 1, 1 ],
   isReal    := True,
   sums     := self >> self,
   rChildren := self >> [self.a, self.b],
   rSetChild := rSetChildFields("a","b"),
   toAMat := self >> Error("not supported"),
   transpose := self >> self,
));

# Declaration of SortConfigBase, a 2x2 Configurable sorter.
Class(SortConfigBase, BaseMat, rec(
   abbrevs   := [(a)-> [a]],
   new       := (self, a) >> SPL( WithBases(self, rec(dimensions:=[2,2], a := a))),
   print := (self, i, is) >> Print(self.name, "(", self.a, ")"),
   dims      := self >> [ 2, 2 ],
   isReal    := True,
   sums     := self >> self,
   rChildren := self >> [self.a],
   rSetChild := rSetChildFields("a"),
   toAMat := self >> Error("not supported"),
   transpose := self >> self,
));

# Declaration of SortLinearBase, used for linear sorter with w = 1
Class(LinearSortBase, BaseMat, rec(
   abbrevs   := [()-> []],
   new       := (self) >> SPL( WithBases(self, rec()) ).setDims(),
   dims      := self >> [ 2, 2 ],
   isReal    := True,
   sums     := self >> self,
   rChildren := self >> [],
   rSetChild := rSetChildFields(),
   toAMat := self >> Error("not supported"),
   transpose := self >> self,
));



#taken from: .../spiral/compiler/dag.gi
regassign.op_in  := self >> ConcatList(self.loc.rChildren(), ArgsExp) :: ArgsExp(self.exp);
regassign.op_out := self >> [self.loc];
regassign.op_inout := self >> [];
regassign.getNoScalar := self >> When(IsBound(self.exp.getNoScalar), self.exp.getNoScalar(), []);

# This tells Spiral how to translate 'SortBase' into code.
HDLCodegen.SortBase := (self, o, y, x, opts) >>
    chain(
	assign(nth(y,0), cond(leq(nth(x,0), nth(x,1)), nth(x,0), nth(x,1))), 
	assign(nth(y,1), cond(leq(nth(x,0), nth(x,1)), nth(x,1), nth(x,0)))
    ); 

HDLCodegen.SortBase_w1 := (self, o, y, x, opts) >>
   let(
     t0 := TempVar(x.t.t),
     t1 := TempVar(x.t.t),
     t3 := TempVar(x.t.t),
     t5 := TempVar(x.t.t),
     t6 := TempVar(x.t.t),

     chain(
	assign(t3, imod(o.a, 2)),
	regassign(t0, cond(t3, t0, nth(x,0))),
	assign(t5, cond(leq(t0, nth(x,0)), nth(x,0), t0)),
	assign(t6, cond(leq(t0, nth(x,0)), t0, nth(x,0))),
	regassign(t1, cond(t3, t5, t1)),
	assign(nth(y,0), cond(t3, t6, t1))
     )
    );

HDLCodegen.SortConfigBase_w1 := (self, o, y, x, opts) >>
   let(
     t0 := TempVar(x.t.t),
     t1 := TempVar(x.t.t),
     t3 := TempVar(x.t.t),
     t5 := TempVar(x.t.t),
     t6 := TempVar(x.t.t),
     t7 := TempVar(x.t.t),
     t8 := TempVar(x.t.t),
     t9 := TempVar(x.t.t),
     t10 := TempVar(x.t.t),
     t2 := TempVar(x.t.t),

     chain(
	assign(t3, imod(o.a, 2)),
	regassign(t0, cond(t3, t0, nth(x,0))),
	assign(t7, eq(o.b,0)),
	assign(t8, eq(o.b,1)),
		
	assign(t2, leq(t0, nth(x,0))),
	assign(t9, cond(t2, t0, nth(x,0))),
	assign(t10, cond(t2, nth(x,0), t0)),
	assign(t5, cond(t7, nth(x,0) , t8, t9, t10)),
	assign(t6, cond(t7, t0 , t8, t10, t9)),
	
	regassign(t1, cond(t3, t5, t1)),
	assign(nth(y,0), cond(t3, t6, t1))
     )
    );

HDLCodegen.LinearSortBase := (self, o, y, x, opts) >>
   let(
     #x[1] indicates the start of a new list. When x[1] equals 1, y[0] 
     #is set to the value of the register t0, and t0 is set to the value of x[0]

     t0 := TempVar(x.t.t),
     t1 := TempVar(x.t.t),
     t2 := TempVar(x.t.t),
     t3 := TempVar(x.t.t),
     t4 := TempVar(x.t.t),
     t5 := TempVar(x.t.t),
     r0 := TempVar(x.t.t),
     x0 := TempVar(x.t.t),
     x1 := TempVar(x.t.t),
       
     chain(

	 assign(x0, nth(x,0)),
	 assign(x1, nth(x,1)),
	 assign(t3, eq(x1, 1)),
	 regassign(r0, cond(t3, x0, t1)),
	 #regassign(r0, t0),
	 assign(t4, leq(r0, x0)),
	 assign(t1, cond(t4, x0, r0)),
	 assign(t2, cond(t4, r0, x0)),
	 assign(nth(y, 0), cond(t3, r0, t2)),
	 #regassign(t5, x1),
	 assign(nth(y, 1), x1)
     )
    );

# HDLCodegen.LinearSortBase := (self, o, y, x, opts) >>
#    let(
#      t0 := TempVar(x.t.t),
#      #x[1] indicates the start of a new list. When x[1] equals 1, y[0] 
#      #is set to the value of the register t0, and t0 is set to the value of x[0]
#      chain(
# 	assign(nth(y,0),cond(eq(nth(x,1),1),t0,leq(t0, nth(x,0)),t0,nth(x,0))),
# 	regassign(t0,cond(eq(nth(x,1),1),nth(x,0),leq(t0, nth(x,0)),nth(x,0),t0)),
# 	assign(nth(y,1),nth(x,1))
#      )
#     );

HDLCodegen.SortConfigBase := (self, o, y, x, opts) >>
    let(
	t0 := TempVar(x.t.t),
	t1 := TempVar(x.t.t),
	t2 := TempVar(x.t.t),
	t3 := TempVar(x.t.t),
	chain(
	    assign(t2, nth(x,0)),
	    assign(t3, nth(x,1)),
	    assign(t0, cond(leq(t2, t3), t2, t3)), 
	    assign(t1, cond(leq(t2, t3), t3, t2)),	    
	    assign(nth(y,0), cond(eq(o.a,0), t2, eq(o.a,1), t1, t0)),
	    assign(nth(y,1), cond(eq(o.a,0), t3, eq(o.a,1), t0, t1))
	)
    ); 


# Declaration of a sorter of size n
Class(Sort, TaggedNonTerminal, rec(
    abbrevs := [
    (n)       -> Checked(IsPosIntSym(n), [_unwrap(n)]),
    ],

    hashAs := self >> ObjId(self)(self.params[1]).withTags(self.getTags()),

    dims := self >> [ self.params[1], self.params[1] ],

    terminate := self >> Error("not supported"), # we could probably support this
));


# SortIJPerm(n): DirectSum(I(n/2), J(n/2))
# We do this so we can define a .permBits() function.
# This will let us easily generate large-size hardware implementations.

# (If we do not do this, the perm tool will have to manually compute the 
# bit matrix representation, which will include making an n-times-n matrix.
Class(SortIJPerm, PermClass, rec(
    def := (n) -> Checked(
        IsPosIntSym(n),
        rec(size := n)),

    lambda := self >> let(
        n := self.params[1],
	fDirsum(fId(n/2), J(n/2)).lambda()
    ),

    transpose := self >> self,
    isSymmetric := self >> true,

    permBits := meth(self)
        local n, k, a, b, tmp, i;
	n := self.params[1];
	k := LogInt(n, 2);
	a := [List([1..k], i->0)];
	for i in [1 .. k-1] do	    
	    tmp := Concatenation([1], List([1..k-1], i->0));
	    Append(a, [tmp]);
        od;
	a := a * GF(2).one;
	b := MatSPL(I(k))*GF(2).one;
	
	return (a+b);
    end,   
));




# Rule to breakdown Sort(n), where n is a power of two.
NewRulesFor(Sort, rec(
    Sort_Stream := rec(
        info         := "Streaming sorting network",

        applicable   := nt -> Length(nt.params) = 1 and IsTwoPower(nt.params[1]),

        children := (self, nt) >> let(

	    tag_w_tmp := nt.tags[1],
	    tag_w := tag_w_tmp.bs,
	    t := Log2Int(nt.params[1]),
	    p := Ind(2^t),
	    get_bb := w -> Cond(w=1, TTensorInd(SortBase_w1(p), p, APar, APar),TTensorI(SortBase(), 2^(t-1), APar, APar)),

	    [[ TCompose(
	           [TCompose(List([1..t-1], i ->
	               TCompose([
		           #TTensorI(SortBase(), 2^(t-1), APar, APar),  
		           get_bb(tag_w), 
		           TCompose(List([2..(t-i+1)], j -> 
		               TCompose([
			           TTensorI(TPrm(Tensor(I(2), L(2^(j-1), 2^(j-2))) * L(2^j,2)), 2^(t-j), APar, APar),
			           get_bb(tag_w)
			           #TTensorI(SortBase(), 2^(t-1), APar, APar)
			       ])
		           )),
		        #   TTensorI(TPrm(L(2^(t-i+1), 2^(t-i)) * DirectSum(I(2^(t-i)), J(2^(t-i)))  ), 2^(i-1), APar, APar) 
			   TTensorI(TPrm(L(2^(t-i+1), 2^(t-i)) * SortIJPerm(2^(t-i+1))), 2^(i-1), APar, APar)
		       ])
	            )),		
		    get_bb(tag_w)] 
		    #TTensorI(SortBase(), 2^(t-1), APar, APar)] 
                ).withTags(nt.getTags())
            ]]
	),

        apply        := (nt, c, cnt) -> c[1],

    ),
    
   Sort_Stream6 := rec(
        info         := "Version of Sort_Stream (sortAlg1) removing J permutations and with configurable 2-input sorters",

        applicable   := nt -> Length(nt.params) = 1 and IsTwoPower(nt.params[1]),

        children := (self, nt) >> let(
	    t := Log2Int(nt.params[1]),
            k := Ind(2^(t-1)),
            z := (s) >> t-s,

            c1 := (s) >> logic_and(eq(bit_sel(k, z(s)), 1), neq(s, 1)),
            access_f := (s) >> cond(c1(s), 1, 2),

	    [[ TCompose(
	           [TCompose(List([1..t-1], i ->
	               TCompose([
			   TTensorInd(SortConfigBase(access_f(i)), k, APar, APar),
		           TCompose(List([2..(t-i+1)], j -> 
		               TCompose([
			           TTensorI(TPrm(Tensor(I(2), L(2^(j-1), 2^(j-2))) * L(2^j,2)), 2^(t-j), APar, APar),
			           TTensorInd(SortConfigBase(access_f(i)), k, APar, APar)
			       ])
		           )),
		           TTensorI(TPrm(L(2^(t-i+1), 2^(t-i))), 2^(i-1), APar, APar) 
		       ])
	            )),		
		    TTensorInd(SortConfigBase(access_f(t)), k, APar, APar)]
		    #TTensorI(SortBase(), 2^(t-1), APar, APar)] #what is this line?
                ).withTags(nt.getTags())
            ]]
	),

        apply        := (nt, c, cnt) -> c[1],

    ),

   Sort_Stream_Iter := rec(
        info         := "Stream/Iter sorting network",

	depth_params := [],

        applicable   := (self, nt) >> Length(nt.params) = 1 and IsTwoPower(nt.params[1]) and
	                              Length(self.depth_params) = Log2Int(nt.params[1])-1 and			      
				      let(
					  t := Log2Int(nt.params[1]),
					  d := self.depth_params,
				      ForAll(List([1..t-1], i -> IsInt((t-i+1)/d[i])), j -> j)
				      ),


        children := (self, nt) >> let(
   	    t := Log2Int(nt.params[1]),

	    tag_w_tmp := nt.tags[1],
	    tag_w := tag_w_tmp.bs,
	    p := Ind(2^t),

	    get_bb := w -> Cond(w=1, TTensorInd(SortBase_w1(p), p, APar, APar),TTensorI(SortBase(), 2^(t-1), APar, APar)),

	    stage := i >> TCompose([
			     get_bb(tag_w),
			     #TTensorI(SortBase(), 2^(t-1), APar, APar),
			     TTensorI(TPrm(L(2^(t-i+1), 2^(t-i))) , 2^(i-1), APar, APar)
			  ]),

	    full_stage := i >> TCompose(List([1..self.depth_params[i]], j2 -> stage(i))), 

  	    [[ TCompose(
	           [TCompose(List([1..t-1], i -> let(
		       j := Ind(t-i+1),
		       d := self.depth_params[i],
		       j1 := Ind(d),
		       j2 := Ind((t-i+1)/d),
	               TCompose([
			   Cond(d = ((t-i+1)),
                               full_stage(i), 
				TCompose(List([1..d],j1 -> 
				TICompose(j2, (t-i+1)/d, stage(i))
				)) 
			   ),
		       	   #TTensorI(TPrm(DirectSum(I(2^(t-i)), J(2^(t-i)))), 2^(i-1), APar, APar)
		       	   TTensorI(TPrm(SortIJPerm(2^(t-i+1))), 2^(i-1), APar, APar)
		       ])
	            ))),
		    get_bb(tag_w)] #last iterations left outside of TCompose
		    #TTensorI(SortBase(), 2^(t-1), APar, APar)] #last iterations left outside of TCompose
               ).withTags(nt.getTags())
            ]]
	),

        apply        := (nt, c, cnt) -> c[1],

   ),
   
   #old version -works-
#      Sort_Stream_Iter := rec(
#        info         := "Stream/Iter sorting network",
#
#	depth_params := [],
#
#        applicable   := (self, nt) >> Length(nt.params) = 1 and IsTwoPower(nt.params[1]) and
#	                              Length(self.depth_params) = Log2Int(nt.params[1])-1 and			      
#				      let(
#					  t := Log2Int(nt.params[1]),
#					  d := self.depth_params,
#				      ForAll(List([1..t-1], i -> IsInt((t-i+1)/d[i])), j -> j)
#				      ),
#
#
#        children := (self, nt) >> let(
#   	    t := Log2Int(nt.params[1]),
#
#	    stage := i >> TCompose([
#			     TTensorI(SortBase(), 2^(t-1), APar, APar),
#			     TTensorI(TPrm(L(2^(t-i+1), 2^(t-i))) , 2^(i-1), APar, APar)
#			  ]),
#
#	    full_stage := i >> TCompose(List([1..self.depth_params[i]], j2 -> stage(i))),
#
#
#  	    [[ TCompose(
#	           [TCompose(List([1..t-1], i -> let(
#		       j := Ind(t-i+1),
#		       d := self.depth_params[i],
#		       j1 := Ind((t-i+1)/d),
#	               TCompose([
#			   Cond(j1.range = 1,
#			       full_stage(i),
#			       TICompose(j1, (t-i+1)/d, full_stage(i))
#			   ),
#		       	   TTensorI(TPrm(DirectSum(I(2^(t-i)), J(2^(t-i)))), 2^(i-1), APar, APar)
#		       ])
#	            ))),
#		    TTensorI(SortBase(), 2^(t-1), APar, APar)]
#               ).withTags(nt.getTags())
#            ]]
#	),
#
#        apply        := (nt, c, cnt) -> c[1],
#
#   ),
   
   Sort_Stream5 := rec(
        info         := "Version of Sort_Stream_Iter (sortAlg2) removing J permutations and with configurable 2-input sorters",

	depth_params := [],

        applicable   := (self, nt) >> Length(nt.params) = 1 and IsTwoPower(nt.params[1]) and
	                              Length(self.depth_params) = Log2Int(nt.params[1])-1 and			      
				      let(
					  t := Log2Int(nt.params[1]),
					  d := self.depth_params,
				      ForAll(List([1..t-1], i -> IsInt((t-i+1)/d[i])), j -> j)
				      ),


        children := (self, nt) >> let(
   	    t := Log2Int(nt.params[1]),
	    k := Ind(2^(t-1)),
	    z := (s) >> t-s,

	    c1 := (s) >> logic_and(eq(bit_sel(k, z(s)), 1), neq(s, 1)),
	    access_f := (s) >> cond(c1(s), 1, 2),
	    
	    stage := i >> TCompose([
			     TTensorInd(SortConfigBase(access_f(i)), k, APar, APar),
			     TTensorI(TPrm(L(2^(t-i+1), 2^(t-i))) , 2^(i-1), APar, APar)
			  ]),

	    full_stage := i >> TCompose(List([1..self.depth_params[i]], j2 -> stage(i))), 

  	    [[ TCompose(
	           [TCompose(List([1..t-1], i -> let(
		       j := Ind(t-i+1),
		       d := self.depth_params[i],
		       j1 := Ind(d),
		       j2 := Ind((t-i+1)/d),
		       Cond(d = ((t-i+1)),
                                full_stage(i), 
				TCompose(List([1..d],j1 -> 
				TICompose(j2, (t-i+1)/d, stage(i))
				)) 
			       #unroll this change to TCompose
			)
	            ))),
		    TTensorInd(SortConfigBase(access_f(t)), k, APar, APar)] #last iterations left outside of TCompose
               ).withTags(nt.getTags())
            ]]
	),

        apply        := (nt, c, cnt) -> c[1],

   ),

  Sort_Stream3 := rec(
        info         := "",
	
	depth_out := 1,
        depth_in := [],

        applicable   := (self, nt) >> Length(nt.params) = 1 and IsTwoPower(nt.params[1]) and
			     	      Length(self.depth_in) = Log2Int(nt.params[1])-1  and
				      let(
                                          t := Log2Int(nt.params[1]), d_out := self.depth_out,
                                          d_in := self.depth_in, IsInt(t/d_out) and ((d_out=1) or ((d_out=t) and 
                                      	  (ForAll(List([1..t-1], i -> IsInt((t-i+1)/d_in[i])), j -> j))))
                                      ),

        children := (self, nt) >> let(
	    d_out := self.depth_out, 
	    d_in := self.depth_in,
            t := Log2Int(nt.params[1]),
	    k := Ind(2^(t-1)),
	    iter_d1 := Ind(t),

            z := (lp, jp) >> (t-1)-(lp+jp),
            c2 := (lp, jp) >> logic_and(eq(bit_sel(k, z(lp, jp)), 1), neq(lp, 0)),
            access_f := (lp, jp) >> cond(c2(lp, jp), 1, 2),
            stage := (lp, jp) >> TCompose([
                       TTensorInd(SortConfigBase(access_f(lp, jp)), k, APar, APar),
                       TPrm(L(2^t, 2^(t-1)))
                   ]),
	    full_stage := i >> TCompose(List([0..d_in[i+1]-1], j -> stage(i,j))),
	    [[ Cond(d_out=1,
			TICompose(iter_d1,t,let(
				iter_d2 := Ind(t-iter_d1),
	    			p_list := List([0..t-1], i-> get_l_power(t,i)),
				TCompose([
					TICompose(iter_d2,t-iter_d1, stage(iter_d1,iter_d2)),
		  	 		TPrmMulti(p_list,iter_d1)
				])	
			)).withTags(nt.getTags()),
		    d_out=t,
			TCompose([
				TCompose(List([0..t-2], iter_d3 -> let(
				  d_stage := d_in[iter_d3+1],	
            			  iter_d5 := Ind((t-iter_d3)/d_stage),
				  TCompose([
					Cond(d_stage=(t-iter_d3),
						full_stage(iter_d3),
						TCompose(List([0..d_stage-1],iter_d4 ->
						  TICompose(iter_d5,(t-iter_d3)/d_stage,stage(iter_d3,iter_d4*iter_d5))))
					),
					get_l_power(t,iter_d3)
				  ])
				))),
				stage(t-1,0),
				get_l_power(t,t-1)
			]).withTags(nt.getTags())	
	    )]]
		
	),
        apply        := (nt, c, cnt) -> c[1],
   ),

   Sort_Stream4 := rec(
        info         := "",

	depth := 1,

        applicable   := (self, nt) >> Length(nt.params) = 1 and IsTwoPower(nt.params[1]) and
	                              let (d := self.depth, t := Log2Int(nt.params[1]), IsInt(t*t/self.depth) and
				      (IsInt(d/t) or IsInt(t/d))),

        children := (self, nt) >> let(
   	    t := Log2Int(nt.params[1]),
	    d := self.depth,
	    d1 := cond(leq(d,t), d, t).ev(),
	    d2 := cond(leq(d,t), 1, d/t).ev(), 

	    k := Ind(2^(t-1)),
	    j := Ind(t),
	    l := Ind(t),
	    n := Ind(t/d2),
	    v := Ind(t/d1),
	    
            d_tmp := cond(leq(d,t), t/d, 0).ev(),
            d2_tmp := cond(leq(t,d), d/t, 0).ev(),
	    s_tmp := ((t*t)/d),
            m2 := Ind(d_tmp),
	    s2 := Ind(s_tmp),
	    v2 := Ind(d2_tmp),
	    n2 := Ind(d),
	    n3 := Ind(d),
	    l2 := Ind(t),
	   
	    tag_w_tmp := nt.tags[1],
	    tag_w := tag_w_tmp.bs,
	    p := Ind(2^t),

 
	    c1 := (lp, jp) >> lt((t-1), (lp+jp)),
	    z := (lp, jp) >> (t-1)-(lp+jp),
	    z_w1 := (lp, jp) >> (t-1)-(lp+jp)+1,
	    c2 := (lp, jp) >> logic_and(eq(bit_sel(k, z(lp, jp)), 1), neq(lp, 0)),
	    c2_w1 := (lp, jp) >> logic_and(eq(bit_sel(p, z_w1(lp, jp)), 1), neq(lp, 0)),
	    	    
	    access_f := (lp, jp) >> cond(c1(lp, jp), 0, c2(lp, jp), 1, 2),
	    access_f_w1 := (lp, jp) >> cond(c1(lp, jp), 0, c2_w1(lp, jp), 1, 2),
	    
	    get_bb := (lp,jp) -> Cond(tag_w=1, TTensorInd(SortConfigBase_w1(p,access_f_w1(lp,jp)), p, APar, APar),TTensorInd(SortConfigBase(access_f(lp, jp)), k, APar, APar)),

	    stage := (lp, jp) >> TCompose([
		       get_bb(lp, jp),
		       #TTensorInd(SortConfigBase(access_f(lp, jp)), k, APar, APar),
		       TPrm(L(2^t, 2^(t-1)))
	           ]),

	    full_stage := np_1 >> TCompose(List([0..t-1], m_1 -> TCompose(List([0..t-1], j_1 -> stage(m_1, j_1))))),
	    full_stage1 := np >> TCompose(List([0..d2-1], m -> TCompose(List([0..t-1], j -> stage(d2*np+m, j))))),
	    full_stage2 := vp >> TCompose(List([0..d1-1], s -> stage(l, vp+s))),
            full_stage1b := np2 >> TICompose(m2,d_tmp, TICompose(j,t, stage((t/d)*np2+m2, j))),

             # Old: problem is that it's assuming the l2 above, which is an unassigned iterator.  
	     # There is also a problem with the vp2+s2 parameter: you need to multiply vp2 by the number of iterations.
             # full_stage2b := (vp2) >> TICompose(s2,s_tmp, stage(l2, vp2+s2)),
            full_stage2b := (vp2, l3) >> TICompose(s2,s_tmp, stage(l3, vp2*s_tmp+s2)),
	
  	    [[ Cond(d=t*t, full_stage(0).withTags(nt.getTags()),		
		    d<t, TCompose(List([0..d-1], n2 -> full_stage1b(n2))).withTags(nt.getTags()),
                    d=t, TCompose(List([0..d-1], n3 -> full_stage1b(n3))).withTags(nt.getTags()),
                    d>t, TCompose(List([0..t-1], l3 -> TCompose(List([0..d2_tmp-1], v2 -> full_stage2b(v2, l3))))).withTags(nt.getTags())) #the problem seems to be the outer most TCompose works if it was TICompose		    
		    #d>t, TICompose(n, t/d2, full_stage1(n)).withTags(nt.getTags())) #old one but will leave it as new one does not work yet
	        ]]
	),
	
#	Old version -works-
#	   Sort_Stream4 := rec(
#        info         := "",
#
#	depth := 1,
#
#        applicable   := (self, nt) >> Length(nt.params) = 1 and IsTwoPower(nt.params[1]) and
#	                              let (d := self.depth, t := Log2Int(nt.params[1]), IsInt(t*t/self.depth) and
#				      (IsInt(d/t) or IsInt(t/d))),
#
#
#        children := (self, nt) >> let(
#   	    t := Log2Int(nt.params[1]),
#	    d := self.depth,
#	    d1 := cond(leq(d,t), d, t).ev(),
#	    d2 := cond(leq(d,t), 1, d/t).ev(), 
#
#	    k := Ind(2^(t-1)),
#	    j := Ind(t),
#	    l := Ind(t),
#	    n := Ind(t/d2),
#	    v := Ind(t/d1),
#	    
#	    
#	    c1 := (lp, jp) >> lt((t-1), (lp+jp)),
#	    z := (lp, jp) >> (t-1)-(lp+jp),
#	    c2 := (lp, jp) >> logic_and(eq(bit_sel(k, z(lp, jp)), 1), neq(lp, 0)),
#	    	    
#	    access_f := (lp, jp) >> cond(c1(lp, jp), 0, c2(lp, jp), 1, 2),
#	    
#	    stage := (lp, jp) >> TCompose([
#		       TTensorInd(SortConfigBase(access_f(lp, jp)), k, APar, APar),
#		       TPrm(L(2^t, 2^(t-1)))
#	           ]),
#
#	    full_stage1 := np >> TCompose(List([0..d2-1], m -> TCompose(List([0..t-1], j -> stage(d2*np+m, j))))),
#	    full_stage2 := vp >> TCompose(List([0..d1-1], s -> stage(l, vp+s))),
#
#  	    [[ Cond(d=t*t, full_stage1(0).withTags(nt.getTags()),		
#		    d>t, TICompose(n, t/d2, full_stage1(n)).withTags(nt.getTags()),
#		    d<t, TICompose(l, t, TICompose(v, t/d1, full_stage2(d1*v))).withTags(nt.getTags()),
#		    d=t, TICompose(l, t, full_stage2(0)).withTags(nt.getTags()))
#	        ]]
#	),
#	
#        apply        := (nt, c, cnt) -> c[1],
#   ),

	# this works, but only for d=1.
#         children := (self, nt) >> let(
#    	    t := Log2Int(nt.params[1]),
# 	    k := Ind(2^(t-1)),
# 	    l := Ind(t),
# 	    j := Ind(t),
	    
# 	    c1 := lt((t-1), (l+j)),
# 	    z := (t-1)-(l+j),
# 	    c2 := logic_and(eq(bit_sel(k, z), 1), neq(l, 0)),
	    	    
# 	    access_f := cond(c1, 0, c2, 1, 2),
	    
# 	    d1 := cond(leq(d,t), d, t),
# 	    d2 := cond(leq(d,t), 1, d/t), 
	    
# 	    c1 := lt((t-1), (l+j)),
# 	    z := (t-1)-(l+j),
# 	    c2 := logic_and(eq(bit_sel(k, z), 1), neq(l, 0)),
	    	    
# 	    access_f := cond(c1, 0, c2, 1, 2),
	    
# 	    d1 := cond(leq(d,t), d, t),
# 	    d2 := cond(leq(d,t), 1, d/t), 

#   	    [[ TICompose(l, t, 
# 		   TICompose(j, t, TCompose([
# 		       TTensorInd(SortConfigBase(access_f), k, APar, APar),
# 		       TPrm(TL(2^t, 2^(t-1), 1, 1))
# 	           ]))).withTags(nt.getTags())
#             ]]
# 	),

        apply        := (nt, c, cnt) -> c[1],

   ),

  Linear_Sort := rec(
        info         := "",
	
        applicable   := (self, nt) >> IsTwoPower(nt.params[1]),

        children := (self, nt) >> let(
            t := Log2Int(nt.params[1]),
	    [[ TCompose(List([1..2^t], i -> 
		TTensorI(LinearSortBase(), nt.params[1], APar, APar)
			)).withTags(nt.getTags()),
	    ]]
		
	),
        apply        := (nt, c, cnt) -> c[1],
   ),

));	        



#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
# Old stuff, not used now


#     Sort_Stream_old := rec(
#         info         := "Streaming sorting network",

#         switch       := true,
#         applicable   := nt ->
#             Length(nt.params) = 1 and IsTwoPower(nt.params[1]),

#         children := (self, nt) >> let(
# 	    t := Log2Int(nt.params[1]),
# 	    [[ TCompose(List([1..t], i ->
# 	 	TCompose([
# 		    TTensorI(BitonicSort(2^(t-i+1)), 2^(i-1), APar, APar),
# 		    TTensorI(TPrm(DirectSum(I(2^(t-i)), J(2^(t-i)))), 2^(i-1), APar, APar)
# 		])
#             )).withTags(nt.getTags())]]
#         ),

#         apply        := (nt, c, cnt) -> c[1],

#     ),

# Declaration of a bitonic sorter of size n
# Class(BitonicSort, TaggedNonTerminal, rec(
#     abbrevs := [
#     (n)       -> Checked(IsPosIntSym(n), [_unwrap(n)]),
#     ],

#     hashAs := self >> ObjId(self)(self.params[1]).withTags(self.getTags()),

#     dims := self >> [ self.params[1], self.params[1] ],

#     terminate := self >> Error("not supported"), # we could probably support this
# ));



# Old stuff, not used now.		       
# # Rules to break-down BitonicSort()
# NewRulesFor(BitonicSort, rec(
#     BitonicSort_Stream := rec(
#         info         := "Streaming Bitonic sorting network",

#         switch       := true,
#         applicable   := nt ->
#             Length(nt.params) = 1 and IsTwoPower(nt.params[1]) and nt.params[1] > 2,

#         children := (self, nt) >> let(
# 	    k := Log2Int(nt.params[1]),
# 	    [[ TCompose([ TTensorI(SortBase(), 2^(k-1), APar, APar),
# 		  TCompose(List([2..k], j -> TCompose([
# 		      TTensorI(TPrm(Tensor(I(2), L(2^(j-1), 2^(j-2))) * L(2^j, 2)), 2^(k-j), APar, APar),
# 		      TTensorI(SortBase(), 2^(k-1), APar, APar)
# 		  ]))),
# 		  TPrm(L(2^k, 2^(k-1)))
# 	       ]).withTags(nt.getTags())
# 	    ]]
#         ),

#         apply        := (nt, c, cnt) -> c[1],
#     ),

#     BitonicSort_Base := rec(
# 	info := "Bitonic soring network base rule",
# 	applicable := nt -> nt.params[1] = 2,
# 	chldren := (self, nt) >> [[ ]],
# 	apply := (nt, c, cnt) -> SortBase()
#     )

# ));	        
