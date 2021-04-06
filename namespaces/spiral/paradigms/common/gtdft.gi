
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


setLeft := function(tags)
   local i, retval;
   retval := Copy(tags);
   for i in retval do
      i.isLeftChild := true;
      i.isRightChild := false;
 od;
 return(retval);
end; 

setRight := function(tags)
   local i, retval;
   retval := Copy(tags);
   for i in retval do
      i.isLeftChild := false;
      i.isRightChild := true;
 od;
 return(retval);
end; 


NewRulesFor(WHT, rec(
    WHT_GT := rec(
    switch := false,
    maxSize := false,
    minSize := false,
    codeletSize := false,
    inplace := false,

    applicable := (self, t) >> let(n := Rows(t),
        n > 2 and
        (self.maxSize=false or n <= self.maxSize) and
        (self.minSize=false or n >= self.minSize) and
        not IsPrime(n)),

        children := (self, t) >> Map2(Filtered(DivisorPairs(Rows(t)), d->When(IsInt(self.codeletSize), d[1]<=self.codeletSize, true)),
        (m,n) -> [
        GT(WHT(LogInt(m, 2)), XChain([0, 1]), XChain([0, 1]), [n]).withTags(t.getTags()),
        GT(WHT(LogInt(n, 2)), XChain([1, 0]), XChain([1, 0]), [m]).withTags(t.getTags()),
        #GT(WHT(LogInt(m, 2)), XChain([0, 1]), XChain([0, 1]), [n]).withTags( let(ts := t.getTags(),  setLeft(ts)) ),
        #GT(WHT(LogInt(n, 2)), XChain([1, 0]), XChain([1, 0]), [m]).withTags( let(ts := t.getTags(), setRight(ts)) ),
        ]),

    apply := (self, t, C, Nonterms) >>  C[1] * C[2]
    )
));

NewRulesFor(DFT, rec(
    DFT_GT_CT := rec(
    switch := false,
    maxSize := false,
    minSize := 2,
    codeletSize := false,
    inplace := false,

    a := rec(
        precompute := true
    ),

    applicable := (self, t) >> let(n := Rows(t),
        n > 2 and
        (self.maxSize=false or n <= self.maxSize) and
        (self.minSize=false or n >= self.minSize) and
        not IsPrime(n)),

        #GT(DFT(m, t.params[2] mod m), XChain([0, 1]), XChain([0, 1]), [n]).withTags(  let(a := t.getTags(),  ),
        #GT(DFT(n, t.params[2] mod n), XChain([0, 1]), XChain([1, 0]), [m]).withTags(  t.getTags())

        #GT(DFT(m, t.params[2] mod m), XChain([0, 1]), XChain([0, 1]), [n]).withTags(t.getTags()),
        #GT(DFT(n, t.params[2] mod n), XChain([0, 1]), XChain([1, 0]), [m]).withTags(t.getTags())

    children := (self, t) >> Map2(Filtered(DivisorPairs(Rows(t)), d->When(IsInt(self.codeletSize), d[1]<=self.codeletSize, true)),
        (m,n) -> [
        GT(DFT(m, t.params[2] mod m), XChain([0, 1]), XChain([0, 1]), [n]).withTags(t.getTags()),
        GT(DFT(n, t.params[2] mod n), XChain([0, 1]), XChain([1, 0]), [m]).withTags(t.getTags()),
        #GT(DFT(m, t.params[2] mod m), XChain([0, 1]), XChain([0, 1]), [n]).withTags( let(ts := t.getTags(),  setLeft(ts)) ),
        #GT(DFT(n, t.params[2] mod n), XChain([0, 1]), XChain([1, 0]), [m]).withTags( let(ts := t.getTags(), setRight(ts)) ),
         ]),

    apply := (self, t, C, Nonterms) >>  let(
            inplace := When(self.inplace, Inplace, Grp),
            n := Rows(Nonterms[2].params[1]),
            rot := t.params[2],
            compute := When(self.a.precompute, fPrecompute, fComputeOnline),

            inplace(Buf(C[1] * Diag(compute(Tw1(Rows(t), n, rot))))) * C[2])
    )
));

# GT(DFT(), ...) rules
#
NewRulesFor(GT, rec(
    GT_DFT_Base2 := rec(
    applicable := (self, t) >> let(rank := Length(t.params[4]),
        rank = 0 and PatternMatch(t, [GT, DFT, @, @, @, @, @], empty_cx()) and Rows(t.params[1])=2),
    apply := (t, C, Nonterms) -> F(2)
    ),

    GT_DFT_CT := rec(
        maxSize       := false,
        minSize       := false,
        minRank       := 0,
        maxRank       := 1,
        codeletSize := 32,
        inplace := true,

        a := rec(
            precompute := true
        ),

        applicable := (self, t) >> let(
            rank := Length(t.params[4]), 
            dft := t.params[1],

            rank >= self.minRank 
            and rank <= self.maxRank 
            and When(rank>0, t.getTags()=[], true) 
            and (self.maxSize=false or Rows(dft) <= self.maxSize) 
            and (self.minSize=false or Rows(dft) >= self.minSize) 
            and PatternMatch(t, [GT, DFT, XChain, XChain, @, @, @], empty_cx()) 
            and DFT_GT_CT.applicable(dft)
        ),
        
        children := (self, t) >> let(
            dft := t.params[1], 
            g := t.params[2], s := t.params[3], 
            rot := dft.params[2],
            loop_dims := t.params[4],
            nloops := Length(loop_dims),
            tags := t.getTags(),
        
            Map2( Filtered(DivisorPairs(Rows(dft)), d->d[1]<=self.codeletSize),
                (m,n) -> [
                    GT(DFT(m, rot mod m),
                        s.composeWith(XChain([0, 1])), s.composeWith(XChain([0, 1])),
                        Concatenation([n], loop_dims)
                    ).withTags(tags),
        
                    GT(DFT(n, rot mod n),
                        g.composeWith(XChain([0, 1])), s.composeWith(XChain([1, 0])),
                        Concatenation([m], loop_dims)
                    ).withTags(tags)
                ]
            )
        ),
        
        apply := (self, t, C, Nonterms) >> let(
            loop_dims := t.params[4], 
            s := t.params[3],
            N := Rows(t.params[1]), 
            k := t.params[1].params[2], 
            n := Rows(Nonterms[2].params[1]),

            inplace := When(self.inplace, Inplace, Grp),
            compute := When(self.a.precompute, fPrecompute, fComputeOnline),
        
            inplace(Buf(C[1] * Diag(compute(s.toDiag(loop_dims, Tw1(N,n,k)))))) * C[2]
        )
    )
));
