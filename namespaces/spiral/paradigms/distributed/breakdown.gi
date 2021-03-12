
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#-----------------------------------------------------------------------------
# Cell SMP Paradigm (preferred)
#-----------------------------------------------------------------------------

#F SymSPL = Symmetric SPL. In this particular case, we know about the symmetry
#F of this SymSPL, so we can remove it.
Class(KillSymSPL, RuleSet);
RewriteRules(KillSymSPL, rec(
     kill_symspl := Rule([@(1,SymSPL), ISum], e -> @(1).val.child(1)),
));


NewRulesFor(GT, rec(
#F For cases where p doesn't divide n nicely. (eg: Parallelize (I6 x A) across 4 processors
    GT_Cell_nonmultiple := rec(
        maxSize       := false,
        minSize       := false,

        requiredFirstTag := ParCell,

        applicable := (self, t) >> let(
                rank   := Length(t.params[4]),
                procs  := t.firstTag().params[1],
                rank = 1
                  and let(its := t.params[4][1],
                    PatternMatch(t, [GT, @(1), @(2,XChain), @(3,XChain), ...], empty_cx())
                    # Only do IxA for now
                    and t.params[2] = GTPar and t.params[3] = GTPar
                    and IsPosInt(idiv(its, procs).v)
                    and IsPosInt(imod(its, procs).v)
                  )
        ),
        children := (self, t) >> let(
            spl     := t.params[1],
            g       := t.params[2],
            s       := t.params[3],
            its     := t.params[4][1],
            procs   := t.firstTag().params[1],
            pksize  := t.firstTag().params[2],
            remits  := imod(its, procs).v,
            parits  := its - remits,
            origtags:= t.getTags(),
            partag  := t.firstTag(),
            tags    := Drop(t.getTags(), 1),
            When(remits > 1,
                [ [ GT(spl, g, s, [parits]).withTags(origtags), GT(spl, g, s, [remits]).setTags(Concatenation(ParCell(remits, pksize), tags)) ] ],
                [ [ GT(spl, g, s, [parits]).withTags(origtags), spl.withTags(tags) ] ]
            )
        ),

        apply := (self, t, C, Nonterms) >> let(
                 DirectSum(C[1], C[2])
        ) #apply
    ), #GT_Cell_nonmultiple


    GT_Cell := rec(
        maxSize       := false,
        minSize       := false,

        requiredFirstTag := ParCell,

        applicable := (self, t) >> let(
                rank   := Length(t.params[4]),
                procs  := t.firstTag().params[1],
                pkSize := t.firstTag().params[2],
                rank = 1
                  and let(its := t.params[4][1],
                    PatternMatch(t, [GT, @(1), @(2,XChain), @(3,XChain), ...], empty_cx())
                    # The (IxA) construct doesn't need the pkSize constraint that
                    # the (AxI) construct (and (IxA)L, L(IxA)) does:
                    and When(t.params[2] = GTPar and t.params[3] = GTPar,
                                IsPosInt(its/(procs)),
                                IsPosInt(its/(procs*pkSize))
                        )
                  )
        ),

        children := (self, t) >> let(
            spl    := t.params[1],
            g      := t.params[2],
            s      := t.params[3],
            its    := t.params[4][1],
            procs  := t.firstTag().params[1],
            remits := its/procs,
            tags   := Drop(t.getTags(), 1), # Eat up tag at topmost level. NOTE: Nested parallelism not considered for now
            When(remits > 1,
            [ [ GT(spl, g, s, [its / procs]).withTags(tags), ] ],
            [ [ spl.withTags(tags) ] ])
        ),

        apply := (self, t, C, Nonterms) >> let(
            spl     := t.params[1],
            g       := t.params[2],
            s       := t.params[3],
            its     := t.params[4][1],
            gg      := When(g=GTVec, XChain([0,1,2]), XChain([1,2,0])),
            ss      := When(s=GTVec, XChain([0,1,2]), XChain([1,2,0])),
            procs   := t.firstTag().params[1],
            pkSize  := t.firstTag().params[2],
            i       := var("spuid", TInt, procs),
            #z := Error("Breakpoint\n"),

            # NOTE: There is a degree of freedom in allocating packet sizes for
            # ScatDist and GathDist. Here's how we handle things:
            
            # When we do on-chip exchanges, we "combine" ScatSends with
            # GathDists (and vice versa). So we want the packet size of the
            # ScatSends and GathDists (and vice versa) to match across multiple
            # factors of the transform (like the DFT). So we issue a ScatDist
            # here with a matching packet size.

            # However, when we do off-chip stuff, we compose ScatDists with
            # ScatMems instead. Here, we want the highest packet size possible
            # for our ScatDists and GathDists.

            # We need to find a universal solution that works neatly for
            # everything. Perhaps issue the largest available pkSize for
            # ScatDist here, and then step it down if needed in the rewrite
            # rules (since we know it's a ScatDist and that this is possible).

            cscatter:= When(s = GTPar,
                        #ScatDist(ss.part(1, i, Rows(spl), [procs, its/(procs*pkSize)]).range(), pkSize, procs, i),
                        ScatDist(procs, its*Rows(spl)/procs, procs, i),
                        ScatSend(ss.part(1, i, Rows(spl), [procs, its/(procs*pkSize)]),         pkSize, procs, i)),
            cgather := When(g = GTPar,
                        #GathDist(gg.part(1, i, Cols(spl), [procs, its/(procs*pkSize)]).range(), pkSize, procs, i),
                        GathDist(procs, its*Rows(spl)/procs, procs, i),
                        GathRecv(gg.part(1, i, Cols(spl), [procs, its/(procs*pkSize)]),         pkSize, procs, i)),

            DistSum(procs, i, procs, cscatter * C[1] * cgather)
        ) #apply
    ), #GT_Cell


    GT_Cell_auto := rec(
        minPkSize := 4, # Hardcoded for real single precision (4 real elements == 16 bytes)
        requiredFirstTag := ParCell_auto,

        applicable := (self, t) >> let(
                rank   := Length(t.params[4]),
                procs  := t.firstTag().params[1],
                rank = 1
                  and let(its := t.params[4][1],
                    PatternMatch(t, [GT, @(1), @(2,XChain), @(3,XChain), ...], empty_cx())
                    # The (IxA) construct doesn't need the pkSize constraint that
                    # the (AxI) construct (and (IxA)L, L(IxA)) does:
                    and When(t.params[2] = GTPar and t.params[3] = GTPar,
                                IsPosInt(its/(procs)),
                                IsPosInt(its/(procs * self.minPkSize)) #Need a packet size of at least 4
                        )
                  )
        ),

        children := (self, t) >> let(
            spl    := t.params[1],
            g      := t.params[2],
            s      := t.params[3],
            n      := t.params[4][1],
            p      := t.firstTag().params[1],
            r      := n/p,
            tags   := Drop(t.getTags(), 1), # Eat up tag at topmost level. NOTE: Nested parallelism not considered for now
            When(r > 1,
            [ [ GT(spl, g, s, [n/p]).withTags(tags) ] ],
            [ [ spl.withTags(tags) ] ])
        ),

        apply := (self, t, C, Nonterms) >> let(
#           z := Error("Breakpoint\n"),
            spl     := t.params[1],
            g       := t.params[2],
            s       := t.params[3],
            n       := t.params[4][1],
            gg      := When(g.params[1]=[0,1], XChain([0,1,2]), XChain([1,2,0])),
            ss      := When(s.params[1]=[0,1], XChain([0,1,2]), XChain([1,2,0])),
            p       := t.firstTag().params[1],
            pkSize  := n/p,       # Automatically use up the remaining stuff for pkSize
                       # NOTE: pkSize is WRONG! Will be different for Par and Vec!
            i       := var("spuid", TInt, p),
            cscatter:= When(s.params[1]=[1,0],
                        ScatDist(ss.part(1, i, Rows(spl), [p, n/(p*pkSize)]).range(), pkSize, p, i),
                        ScatSend(ss.part(1, i, Rows(spl), [p, n/(p*pkSize)]),         pkSize, p, i)),
            cgather := When(g.params[1]=[1,0],
                        GathDist(gg.part(1, i, Cols(spl), [p, n/(p*pkSize)]).range(), pkSize, p, i),
                        GathRecv(gg.part(1, i, Cols(spl), [p, n/(p*pkSize)]),         pkSize, p, i)),

            DistSum(p, i, p, cscatter * C[1] * cgather)
        ) #apply
    ), #GT_Cell
));


#-----------------------------------------------------------------------------
# Cell DMP Paradigm
#-----------------------------------------------------------------------------
NewRulesFor(GT, rec(
    # Converts (In x A) to A DistSum (parallel loop)
    GT_CellDMP_base_old := rec(
        maxSize       := false,
        minSize       := false,

        requiredFirstTag := ParCellDMP_old,

        applicable := (self, t) >> let(
                rank   := Length(t.params[4]),
                procs  := t.firstTag().params[1],
                rank = 1
                  and let(its := t.params[4][1],
                    PatternMatch(t, [GT, @(1), @(2,XChain), @(3,XChain), ...], empty_cx())
                    and IsPosInt(its/(procs)))
                  and (t.params[2]=GTPar and t.params[3]=GTPar)

        ),

        children := (self, t) >> let(
            spl    := t.params[1],
            g      := t.params[2],
            s      := t.params[3],
            its    := t.params[4][1],
            procs  := t.firstTag().params[1],
            tags  := Drop(t.getTags(), 1),  # Children don't need tags since we parallelize only at the topmost level
            [ [ GT(spl, g, s, [its / procs]).withTags(tags), InfoNt(procs) ] ]
        ),

        apply := (self, t, C, Nonterms) >> let(
            spl     := t.params[1],
            g       := t.params[2],
            s       := t.params[3],
            its     := t.params[4][1],
            gg      := When(g.params[1]=[0,1], XChain([0,1,2]), XChain([1,2,0])),
            ss      := When(s.params[1]=[0,1], XChain([0,1,2]), XChain([1,2,0])),
            procs   := t.firstTag().params[1],
            i       := var("spuid", TInt, procs),
            cscatter:= ScatDist(ss.part(1, i, Rows(spl), [procs, its/(procs)]).range(), 1, procs, i),
            cgather := GathDist(gg.part(1, i, Cols(spl), [procs, its/(procs)]).range(), 1, procs, i),

            DistSum(procs, i, procs, cscatter * C[1] * cgather)

        ) #apply
    ), #GT_CellDMP_Base

    # Converts all other TTensor constructs (IxA)L, L(IxA), (AxI) into L's and (IxA)s
    # The L's are factorized to PTensor * Comm_Cell * PTensor
    GT_CellDMP_gen_old := rec(
        maxSize    := false,
        minSize    := false,
        forTransposition := false,

        requiredFirstTag := ParCellDMP_old,

        applicable := (self, t) >> let(
                rank   := Length(t.params[4]),
                procs  := t.firstTag().params[1],
                rank = 1
                  and let(its := t.params[4][1],
                    PatternMatch(t, [GT, @(1), @(2,XChain), @(3,XChain), ...], empty_cx())
                    and IsPosInt(its/(procs))
                    )
                  and not(t.params[2]=GTPar and t.params[3]=GTPar)
             ),

        children := (self, t) >> let(
            spl    := t.params[1],
            g      := t.params[2],
            s      := t.params[3],
            its    := t.params[4][1],
            procs  := t.firstTag().params[1],
            tags   := t.getTags(),
            m      := Rows(spl),
            n      := its,

            Cond(g = GTVec and s = GTPar, # L(IxA) case
                  [ [
                      TCompose([
                         GT(spl, GTPar, GTPar, t.params[4]),
                         TL(m*n, n)
                      ]).withTags(tags),
                  ] ],

                  g = GTPar and s = GTVec, # (IxA)L case
                  [ [
                      TCompose([
                         TL(m*n, m),
                         GT(spl, GTPar, GTPar, t.params[4]),
                      ]).withTags(tags),
                  ] ],

                  g = GTVec and s = GTVec, # AxI case
                  [ [
                      TCompose([
                         TL(m*n, m),
                         GT(spl, GTPar, GTPar, t.params[4]),
                         TL(m*n, n)
                      ]).withTags(tags),
                  ] ]
            )
        ),

        apply := (self, t, C, Nonterms) >> C[1]
    ) #GT_CellDMP_gen

));

NewRulesFor(TL, rec(
   TL_CellDMP_old := rec(
       maxSize       := false,
       minSize       := false,
       forTransposition := false,

       requiredFirstTag := ParCellDMP_old,

       applicable := (self, t) >> let(
           mn := t.params[1],
           m  := t.params[2],
           p  := t.firstTag().params[1],
           n  := mn/m,
           p2 := p*p,
           IsPosInt(m/p) and IsPosInt(mn/p2) and IsPosInt(n) and (n>p)
       ),

       # Note: TLs with a ParCellDMP tag should never have either of the Ix set (Should always be I1 x L x I1)
       children := (self, t) >> let(
           mn := t.params[1],
           m  := t.params[2],
           p  := t.firstTag().params[1],
           n  := mn/m,
           p2 := p*p,

           # Eat up the parallel (ParCellDMP) tag.
           tags  := Drop(t.getTags(), 1),

           [ [ TL(mn/p, m/p).withTags(tags), TL(n, p, 1, m/p).withTags(tags), InfoNt(m,n,p) ] ]
       ),

       apply := (self, t, C, Nonterms) >> let(
           m   := Nonterms[3].params[1],
           n   := Nonterms[3].params[2],
           p   := Nonterms[3].params[3],
           p2  := p*p,
           mn  := m*n,

           PTensor(C[1], p) * Comm_Cell(p, mn/p2) * PTensor(C[2], p)
       )
   ) #TL_CellDMP
));


setStandAlone := function(tags)
   local i, retval;
   retval := Copy(tags);

   for i in retval do 
      i.isEdge := true;
   od;
   return(retval);
end;



#-----------------------------------------------------------------------------#-----------------------------------------------------------------------------#----------------------------------------------------------------------------- #-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Cell DMP Paradigm - new
#-----------------------------------------------------------------------------
NewRulesFor(GT, rec(
    GT_StickyL := rec(
        requiredFirstTag := StickyL,
        applicable := (self, t) >> true,
        children   := (self, t) >> [[ t.withoutFirstTag() ]],
        apply := (self, t, C, Nonterms) >> C[1]
    ),

    # Converts (In x A) to A DistSum (parallel loop)
    GT_CellDMP_base := rec(
        maxSize       := false,
        minSize       := false,

        requiredFirstTag := ParCellDMP,

        applicable := (self, t) >> let(
                rank   := Length(t.params[4]),
                procs  := t.firstTag().params[1],
                rank = 1
                  and let(its := t.params[4][1],
                    PatternMatch(t, [GT, @(1), @(2,XChain), @(3,XChain), ...], empty_cx())
                    and IsPosInt(its/(procs)))
                  and (t.params[2]=GTPar and t.params[3]=GTPar)

        ),

        children := (self, t) >> let(
            spl    := t.params[1],
            g      := t.params[2],
            s      := t.params[3],
            its    := t.params[4][1],
            procs  := t.firstTag().params[1],
            tags  := Drop(t.getTags(), 1),  # No nested parallelism for now
            [ [ GT(spl, g, s, [its / procs]).withTags(tags), InfoNt(procs) ] ]
        ),

        apply := (self, t, C, Nonterms) >> let(
            spl     := t.params[1],
            g       := t.params[2],
            s       := t.params[3],
            its     := t.params[4][1],
            gg      := When(g.params[1]=[0,1], XChain([0,1,2]), XChain([1,2,0])),
            ss      := When(s.params[1]=[0,1], XChain([0,1,2]), XChain([1,2,0])),
            procs   := t.firstTag().params[1],
            i       := var("spuid", TInt, procs),
            cscatter:= ScatDist(ss.part(1, i, Rows(spl), [procs, its/(procs)]).range(), 1, procs, i),
            cgather := GathDist(gg.part(1, i, Cols(spl), [procs, its/(procs)]).range(), 1, procs, i),

            DistSum(procs, i, procs, cscatter * C[1] * cgather)

        ) #apply
    ), #GT_CellDMP_Base

    # Converts all other TTensor constructs (IxA)L, L(IxA), (AxI) into L's and (IxA)s
    # The L's are factorized to PTensor * Comm_Cell * PTensor
    GT_CellDMP_gen := rec(
        maxSize    := false,
        minSize    := false,
        forTransposition := false,

        requiredFirstTag := ParCellDMP,

        # NOTE: Ensure applicability for vectorized distributed algo
        applicable := (self, t) >> let(
                rank   := Length(t.params[4]),
                procs  := t.firstTag().params[1],
                v      := When(Length(t.firstTag().params)>1, t.firstTag().params[2], 1),
                n      := t.params[4][1],
                rank = 1
                and PatternMatch(t, [GT, @(1), @(2,XChain), @(3,XChain), ...], empty_cx())
                and IsPosInt((n/v) / procs)
                and not(t.params[2]=GTPar and t.params[3]=GTPar)
                #and t.params[2]=GTVec and t.params[3]=GTVec #NOTE: remove this later
             ),

        children := (self, t) >> let(
            spl    := t.params[1],
            g      := t.params[2],
            s      := t.params[3],
            n      := t.params[4][1],
            procs  := t.firstTag().params[1],
            v      := When(Length(t.firstTag().params)>1, t.firstTag().params[2], 1),
            tags   := t.getTags(),

            ltags  := When(tags[1].leftChild(),  setStandAlone(tags), tags),
            rtags  := When(tags[1].rightChild(), setStandAlone(tags), tags),

           # Error("BP"),

            m      := Rows(spl),
            #splnew := When(v=1, spl, GT(spl.withTags([StickyL(v)]), g, s, [v])),
            splnew := When(v=1, spl, GT(spl, g, s, [v])),

            Cond(s = GTPar and g = GTVec, # (IxA)L case
                  [ [
                         GT(splnew, GTPar, GTPar, [n/v]).withTags(tags),
                         TL(m*n/v, n/v, 1, v).withTags(rtags)
                  ] ],

                  # TEMP: commenting this out. We shouldn't see this case
                  s = GTVec and g = GTPar, # L(IxA) case
                  Error("WTF?"),
                  #[ [
                  #       TL(m*n/v, m, 1, v).withTags(ltags),
                  #       GT(splnew, GTPar, GTPar, [n/v]).withTags(tags)
                  #] ],

                  s = GTVec and g = GTVec, # AxI case
                  [ [
                         TL(m*n/v, m, 1, v).withTags(ltags),
                         GT(splnew, GTPar, GTPar, [n/v]).withTags(tags),
                         TL(m*n/v, n/v, 1, v).withTags(rtags)
                  ] ]
            )
        ),

        apply := (self, t, C, Nonterms) >> When(Length(C)=2, C[1]*C[2], C[1]*C[2]*C[3])
    ) #GT_CellDMP_gen

));

NewRulesFor(TL, rec(
   TL_CellDMP := rec(
       maxSize       := false,
       minSize       := false,
       forTransposition := false,

       requiredFirstTag := ParCellDMP,

       applicable := (self, t) >> let(
           mn := t.params[1],
           m  := t.params[2],
           p  := t.firstTag().params[1],
           n  := mn/m,
           p2 := p*p,
           IsPosInt(m/p) and IsPosInt(mn/p2) and IsPosInt(n) and (n>p)
       ),

       children := (self, t) >> let(
           mnByv := t.params[1],
           m     := t.params[2],
           v     := t.params[4],
           p     := t.firstTag().params[1],
           nByv  := mnByv/m,
           p2    := p*p,

           # Eat up the parallel (ParCellDMP) tag.
           tags  := Drop(t.getTags(), 1),
           #When(IsBound(tags[1].isEdgeTL), Error("Edge"), Print("")),

           [ [ TL(mnByv/p, m/p, 1, v).withTags(tags), TL(nByv, p, 1, (m*v)/p).withTags(tags), InfoNt(m,nByv,p,v) ] ]
       ),

       apply := (self, t, C, Nonterms) >> let(
           m      := Nonterms[3].params[1],
           nByv   := Nonterms[3].params[2],
           p      := Nonterms[3].params[3],
           v      := Nonterms[3].params[4],
           p2     := p*p,
           tags   := t.getTags(),


           # Hacking RulesTerm inside here was done by ff.
           #fl     := PTensor(paradigms.vector.rewrite.RulesTerm(C[1]), p),
           #fr     := PTensor(paradigms.vector.rewrite.RulesTerm(C[2]), p),

           #fl     := When(tags[1].leftEdge(),   LeftEdge(PTensor(C[1], p)), PTensor(C[1], p)),
           #fr     := When(tags[1].rightEdge(), RightEdge(PTensor(C[2], p)), PTensor(C[2], p)),

           # Works, but need to terminate blockvperm
           #Cond(tags[1].leftEdge(),  (LeftEdge(fl) * Comm_Cell(p, (m*nByv*v)/p2) * fr),
           #     tags[1].rightEdge(), (    fl  * Comm_Cell(p, (m*nByv*v)/p2) * RightEdge(fr)),
           #                          (    fl  * Comm_Cell(p, (m*nByv*v)/p2) *      fr ))

           frules := MergedRuleSet(RulesSums, RulesFuncSimp, KillSymSPL, RulesVec, RulesTerm, RulesPropagate),

           csums := frules(C[2].sums()),
           sg := ScatGath(fTensor(fId(csums.dimensions[1]/csums.v), fId(csums.v)), fTensor(csums.func, fId(csums.v)) ),
           sgnew := frules(sg.toloop(csums.v)),

           csums1 := frules(C[1].sums()),
           sg1 := ScatGath(fTensor(fId(csums1.dimensions[1]/csums1.v), fId(csums1.v)), fTensor(csums1.func, fId(csums1.v)) ),
           sgnew1 := frules(sg1.toloop(csums1.v)),



           fl     := When(tags[1].leftEdge(), 
                     #PTensor(csums1, p),
                     PTensor(sgnew1, p),
                     PTensor(C[1], p)),

           fr     := When(tags[1].rightEdge(),
                     #PTensor(csums, p),
                     PTensor(sgnew, p),
                     PTensor(C[2], p)),

           #Error("BP0"),

           fl * Comm_Cell(p, (m*nByv*v)/p2) * fr

       )
   ) #TL_CellDMP
));
