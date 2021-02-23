
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

# Declaration of a sorter of one dimension within tuples of size n
#n is the number of inputs in total. So, there are n/2 tuples
#w should be minimum 2
Class(SortVecBase, BaseMat, rec(
   abbrevs   := [()-> []],
   new       := (self) >> SPL( WithBases(self, rec()) ).setDims(),
   dims      := self >> [ 4, 4 ],
   isReal    := True,
   sums     := self >> self,
   rChildren := self >> [],
   rSetChild := rSetChildFields(),
   toAMat := self >> Error("not supported"),
   transpose := self >> self,
));

Class(SortVecConfigBase_w2, BaseMat, rec(
   abbrevs   := [(a)-> [a] , (b)-> [b]],
   new       := (self, a, b) >> SPL( WithBases(self, rec(dimensions:=[2,2], a := a, b:=b))),
   print := (self, i, is) >> Print(self.name, "(", self.a, ",", self.b, ")"),
   dims      := self >> [ 2, 2 ],
   isReal    := True,
   sums     := self >> self,
   rChildren := self >> [self.a, self.b],
   rSetChild := rSetChildFields("a","b"),
   toAMat := self >> Error("not supported"),
   transpose := self >> self,
));

# Declaration of SortConfigBase, a 2x2 Configurable sorter.
Class(SortVecConfigBase, BaseMat, rec(
   abbrevs   := [(a)-> [a]],
   new       := (self, a) >> SPL( WithBases(self, rec(dimensions:=[4,4], a := a))),
   print := (self, i, is) >> Print(self.name, "(", self.a, ")"),
   dims      := self >> [ 4, 4 ],
   isReal    := True,
   sums     := self >> self,
   rChildren := self >> [self.a],
   rSetChild := rSetChildFields("a"),
   toAMat := self >> Error("not supported"),
   transpose := self >> self,
));

Class(SortVecBase_w2, BaseMat, rec(
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

HDLCodegen.SortVecBase_w2 := (self, o, y, x, opts) >>
   let(
     t0 := TempVar(x.t.t),
     t1 := TempVar(x.t.t),
     t3 := TempVar(x.t.t),
     t5 := TempVar(x.t.t),
     t6 := TempVar(x.t.t),

     t20 := TempVar(x.t.t),
     t21 := TempVar(x.t.t),
     t23 := TempVar(x.t.t),
     t25 := TempVar(x.t.t),
     t26 := TempVar(x.t.t),

     chain(
        assign(t3, imod(o.a, 2)),
        regassign(t0, cond(t3, t0, nth(x,0))),
        assign(t5, cond(leq(t0, nth(x,0)), nth(x,0), t0)),
        assign(t6, cond(leq(t0, nth(x,0)), t0, nth(x,0))),
        regassign(t1, cond(t3, t5, t1)),
        assign(nth(y,0), cond(t3, t6, t1)),

        regassign(t20, cond(t3, t20, nth(x,1))),
        assign(t25, cond(leq(t0, nth(x,0)), nth(x,1), t20)),
        assign(t26, cond(leq(t0, nth(x,0)), t20, nth(x,1))),
        regassign(t21, cond(t3, t25, t21)),
        assign(nth(y,1), cond(t3, t26, t21))

     )
  );

HDLCodegen.SortVecConfigBase_w2 := (self, o, y, x, opts) >>
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

     t20 := TempVar(x.t.t),
     t21 := TempVar(x.t.t),
     t23 := TempVar(x.t.t),
     t25 := TempVar(x.t.t),
     t26 := TempVar(x.t.t),


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
	assign(nth(y,0), cond(t3, t6, t1)),

        regassign(t20, cond(t3, t20, nth(x,1))),
        assign(t25, cond(leq(t0, nth(x,0)), nth(x,1), t20)),
        assign(t26, cond(leq(t0, nth(x,0)), t20, nth(x,1))),
        regassign(t21, cond(t3, t25, t21)),
        assign(nth(y,1), cond(t3, t26, t21))

     )
    );

HDLCodegen.SortVecBase := (self, o, y, x, opts) >>
    chain(
        assign(nth(y,0), cond(leq(nth(x,0), nth(x,2)), nth(x,0), nth(x,2))),
        assign(nth(y,2), cond(leq(nth(x,0), nth(x,2)), nth(x,2), nth(x,0))),

        assign(nth(y,1), cond(leq(nth(x,0), nth(x,2)), nth(x,1), nth(x,3))),
        assign(nth(y,3), cond(leq(nth(x,0), nth(x,2)), nth(x,3), nth(x,1)))
    );



HDLCodegen.SortVecConfigBase := (self, o, y, x, opts) >>
    let(
	t0 := TempVar(x.t.t),
	t1 := TempVar(x.t.t),
	t2 := TempVar(x.t.t),
	t3 := TempVar(x.t.t),
	chain(
	    assign(t2, nth(x,0)),
	    assign(t3, nth(x,2)),
	    assign(t0, cond(leq(t2, t3), t2, t3)), 
	    assign(t1, cond(leq(t2, t3), t3, t2)),	    
	    assign(nth(y,0), cond(eq(o.a,0), t2, eq(o.a,1), t1, t0)),
	    assign(nth(y,2), cond(eq(o.a,0), t3, eq(o.a,1), t0, t1)),

            assign(nth(y,1), cond(leq(nth(x,0), nth(x,2)), nth(x,1), nth(x,3))),
            assign(nth(y,3), cond(leq(nth(x,0), nth(x,2)), nth(x,3), nth(x,1)))

	)
    ); 



Class(SortVec, TaggedNonTerminal, rec(
    abbrevs := [
    (n)       -> Checked(IsPosIntSym(n), [_unwrap(n)]),
    ],

    hashAs := self >> ObjId(self)(self.params[1]).withTags(self.getTags()),

    dims := self >> [ self.params[1], self.params[1] ],

    terminate := self >> Error("not supported"), # we could probably support this
));

NewRulesFor(SortVec, rec(

    Sort_Stream_Vec := rec(
        info         := "Streaming sorting network",

        applicable   := nt -> Length(nt.params) = 1 and IsTwoPower(nt.params[1]),

        children := (self, nt) >> let(

	    tag_w_tmp := nt.tags[1],
	    tag_w := tag_w_tmp.bs,
	    t := Log2Int(nt.params[1]),
	    p := Ind(2^t),
	    #get_bb := w -> TTensorI(SortVecBase(), 2^(t-1), APar, APar),
	    get_bb := w -> Cond(w=2, TTensorInd(SortVecBase_w2(p), p, APar, APar),TTensorI(SortVecBase(), 2^(t-1), APar, APar)),

	    [[ TCompose(
	           [TCompose(List([1..t-1], i ->
	               TCompose([
		           #TTensorI(SortBase(), 2^(t-1), APar, APar),  
		           get_bb(tag_w), 
		           TCompose(List([2..(t-i+1)], j -> 
		               TCompose([
			       
			           # The one below *should* work, but it causes some rewriting problems.
			           #TPrm(Tensor(Tensor(I(2^(t-j)),(Tensor(I(2), L(2^(j-1), 2^(j-2))) * L(2^j,2))),I(2))),
				   # So, I'm going to pull the first I() out of the tensor product.  This
				   # changes it from: 
				   #      TPrm(I x (IxL)*L x I) 
				   # to:
				   #      I x (TPrm(IxL)*L x I)
				   TTensorI(TPrm(Tensor(Tensor(I(2), L(2^(j-1), 2^(j-2))) * L(2^j,2), I(2))), 2^(t-j), APar, APar),

				   # This one doesn't work because it has nested TPrms and it has TTensorI in a TPrm.
			           #TPrm(Tensor(TTensorI(TPrm(Tensor(I(2), L(2^(j-1), 2^(j-2))) * L(2^j,2)), 2^(t-j), APar, APar),I(2))),
			           get_bb(tag_w)
			           #TTensorI(SortBase(), 2^(t-1), APar, APar)
			       ])
		           )),
			 
			   # TPrm(Tensor(TTensorI(TPrm(L(2^(t-i+1), 2^(t-i)) * SortIJPerm(2^(t-i+1))), 2^(i-1), APar, APar),I(2)))
			   # Like above, re-ordering these terms
			   # TPrm(Tensor(Tensor(I(2^(i-1)), L(2^(t-i+1), 2^(t-i)) * SortIJPerm(2^(t-i+1) ) ) ,I(2)))
			   TTensorI(TPrm(Tensor(L(2^(t-i+1), 2^(t-i)) * SortIJPerm(2^(t-i+1)), I(2))), 2^(i-1), APar, APar)

		       ])
	            )),		
		    get_bb(tag_w)] 
		    #TTensorI(SortBase(), 2^(t-1), APar, APar)] 
                ).withTags(nt.getTags())
            ]]
	),

        apply        := (nt, c, cnt) -> c[1],

    ),

   Sort_Stream4_Vec := rec(
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
	   
	    get_bb := (lp,jp) -> Cond(tag_w=2, TTensorInd(SortVecConfigBase_w2(p,access_f_w1(lp,jp)), p, APar, APar),TTensorInd(SortVecConfigBase(access_f(lp, jp)), k, APar, APar)),

	    stage := (lp, jp) >> TCompose([
		       get_bb(lp, jp),
		       #TTensorInd(SortConfigBase(access_f(lp, jp)), k, APar, APar),
		       TPrm(Tensor(L(2^t, 2^(t-1)),I(2)))
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
	
#

        apply        := (nt, c, cnt) -> c[1],

    ),



));
