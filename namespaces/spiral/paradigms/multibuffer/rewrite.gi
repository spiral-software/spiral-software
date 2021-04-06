
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#Class(MultiBufBB, RuleSet);
#RewriteRules(MultiBufBB, rec(
#     remove_bbs := Rule(@(1, Compose, e->ForAll(e.children(), c->ObjId(c)=DistSum)),
#        e -> ComposeDists(@(1).val.children()) )
#));

applyCellInplace := function(sums, opts)
    SubstBottomUp(sums, GathDist, e->Inplace(e));
    SubstBottomUp(sums, GathRecv, e->Inplace(e));
    SubstBottomUp(sums, ScatDist, e->Inplace(e));

    SubstBottomUp(sums, ScatMem, e->Inplace(e));
    SubstBottomUp(sums, GathMem, e->Inplace(e));


    # BB(Inplace()) -> Inplace() (So that a copy operation need not take place)
    SubstBottomUp(sums, [BB, Inplace], e->e.rChildren()[1]);
    return(sums);
end;


#s := SubstBottomUp(s, ScatMem, e->Inplace(e));
#s := SubstBottomUp(s, GathMem, e->Inplace(e));
#s := SubstBottomUp(s, [BB, Inplace], e->e.rChildren()[1]);


# DistSum(Scat__ * MultiBufISum() * Gath__) -> DistSumLoop(MultiBufISum(ScatMem * () * GathMem))

# DistSum has to be converted to a DistSumLoop because DistSum expects its Scat
# and Gath to span the entire expanse of the computation, and therefore assumes
# an appropriate Scat/Gath. DistSumLoop does not.


doMultiBufBC := function(sums, opts)
    Error("doMultiBufBC: BP");
    return(sums);
end;

#F Dist(MultiBuf) is for independent DFTs running on separate SPEs (or sets of
#F SPEs), each multibuffered separately
Class(DistMultiBuf, RuleSet);

RewriteRules(DistMultiBuf, rec(
     # HACK: rule assumes we do only independent DFTs
     join_dist_multibuf_compose := Rule([@(1,DistSum), @, @(2,Compose, e->Length(Collect(e.rChildren()[1], MultiBufISum))=1) ],
       e -> let(
          dsum := @(1).val,
          Error(""),
          msum := Collect(@(2).val, MultiBufISum)[1],
          sp   := Collect(dsum, @(3,[ScatDist, ScatSend]))[1],
          sm   := msum.scatmem,
          sdm  := fCompose(sp.func, fTensor(sm.func, fId(sm.pkSize/sp.pkSize))),
          
          gp   := Collect(dsum, @(3,[GathDist, GathRecv]))[1],
          gm   := msum.gathmem,
          gdm  := fCompose(gp.func, fTensor(gm.func, fId(gm.pkSize/gp.pkSize))),
          
          DistSumLoop(dsum.P, dsum.var, dsum.domain,
            MultiBufISum(msum.var, msum.domain,
              #ScatDirectMem(sdm, sp.pkSize, dsum.P, dsum.var),
              ScatMem(sdm, sp.pkSize),
              msum.rChildren()[2],
              #GathDirectMem(gdm, gp.pkSize, dsum.P, dsum.var)
              GathMem(gdm, gp.pkSize)
            ) 
          )
       )
     ),

     join_dist_multibuf_separate := Rule(@(1, DistSum, e->Length(Collect(e.rChildren()[2], MultiBufISum))=1),
       e -> let(
          dsum := @(1).val,
          msum := Collect(@(1).val, MultiBufISum)[1],

          sp   := Collect(dsum, @(3,[ScatDist, ScatSend]))[1],
          sm   := msum.scatmem,
          spm  := fCompose(fTensor(sp.func, fId(sp.pkSize/sm.pkSize)), sm.func),

          
          gp   := Collect(dsum, @(3,[GathDist, GathRecv]))[1],
          gm   := msum.gathmem,
          gpm  := fCompose(fTensor(gp.func, fId(gp.pkSize/gm.pkSize)), gm.func),

          
          DistSumLoop(dsum.P, dsum.var, dsum.domain,
            MultiBufISum(msum.var, msum.domain,
              #ScatDirectMem(spm, sp.pkSize, dsum.P, dsum.var),
              ScatMem(spm, sm.pkSize),
              msum.rChildren()[2],
              #GathDirectMem(gpm, gp.pkSize, dsum.P, dsum.var)
              GathMem(gpm, gm.pkSize)
            ) 
          )
       )
     )

));

#F MultiBuf(Dist) is for parallel DFTs that are multibuffered
#F We pull out a parallel scat (gath) function into the multibufisum's scat (gath) function
#F Sums becomes invalid (strictly speaking) because we now use loop vars outside of their definition.
Class(MultiBufDist, RuleSet);
RewriteRules(MultiBufDist, rec(
    # This has to be done exactly once, else will loop infinitely. Look at hack below
    #NOTE: The rule below doesn't seem to fire. Probably doesn't match becacuse of the # of @'s at the end in this line:
     join_multibuf_dist_compose := Rule([@(1,[MultiBufISum,MemISum]), @(2,[Compose,ComposeDists], e->Length(Collect(e.rChildren()[2], DistSum))=1), @, @ ],
       e -> let(

          #HACK: clean way of getting the right gath/scat is to ask for the Compose's leftmost Scat/Gath
          msum := @(1).val,
          dsums := Collect(@(2).val, DistSum),
          dsum1 := dsums[1],
          dsum2 := dsums[Length(dsums)],

          sd   := Collect(dsum1, @(3,[ScatDist, ScatSend]))[1],
          sm   := msum.scatmem,
          sdm  := fCompose(fTensor(sm.func, fId(sm.pkSize/sd.pkSize)), sd.func),
          
          gd   := Collect(dsum2, @(3,[GathDist, GathRecv]))[1],
          gm   := msum.gathmem,
          gdm  := fCompose(fTensor(gm.func, fId(gm.pkSize/gd.pkSize)), gd.func),

          isum := When(ObjId(@(1).val)=MultiBufISum, MultiBufISumFinal, MemISumFinal),

          # HACK: Converting to a MultiBufISumFinal so this rule won't match more than once
          isum(msum.var, msum.domain,
              ScatMem(sdm, sd.pkSize),
              msum.rChildren()[2],    #NOTE: is this okay?
              GathMem(gdm, gd.pkSize)
          )
       )
     ),

     join_multibuf_dist_separate := Rule(@(1, MultiBufISum, e->Length(Collect(e.rChildren()[2], DistSum))=1),
       e -> let(

          msum := @(1).val,
          dsum := Collect(@(1).val, DistSum)[1],

          sd   := Collect(dsum, @(3,[ScatDist, ScatSend]))[1],
          sm   := msum.scatmem,
          sdm  := fCompose(fTensor(sm.func, fId(sm.pkSize/sd.pkSize)), sd.func),
          
          gd   := Collect(dsum, @(3,[GathDist, GathRecv]))[1],
          gm   := msum.gathmem,
          gdm  := fCompose(fTensor(gm.func, fId(gm.pkSize/gd.pkSize)), gd.func),

          # HACK: Converting to a MultiBufISumFinal so this rule won't match more than once
          MultiBufISumFinal(msum.var, msum.domain,
              ScatMem(sdm, sd.pkSize),
              msum.rChildren()[2],    #NOTE: is this okay?
              GathMem(gdm, gd.pkSize)
          )
       )
     )
));

#F MultiBufDist_large is for large DFTs that must be multibuffered in parts.
#F Since the multibuf loop is the outer loop, it assumes the "chip" can bring
#F data from memory on to it. In reality, "chips" don't exist -- only cores do. So
#F this rule distributed chip access across the core. It's different from the
#F MultiBufDist rule in that it doesn't interact with the inside DistSum's
#F Scatters or gathers. It does use the parallel loop var outside of its
#F definition when done.


Class(MultiBufDist_large, RuleSet);

RewriteRules(MultiBufDist_large, rec(
    distribute_scatgathmems := Rule(@(1, MultiBufISum, e->Length(Collect(e.rChildren()[2], DistSum))>=1 ),
      e -> let(

        msum  := @(1).val,
        dsum  := Collect(@(1).val, DistSum)[1],
        sm    := msum.scatmem,
        gm    := msum.gathmem,

        sf    := When(sm.func.domain()=1, dsum.P, 1),
        gf    := When(gm.func.domain()=1, dsum.P, 1),

        sfunc := When(sf=1, sm.func, fTensor(sm.func, fId(sf))),
        gfunc := When(gf=1, gm.func, fTensor(gm.func, fId(gf))),


        Ns    := sf*sm.func.range()/msum.domain,
        ns    := sf*sm.func.domain()/dsum.P,
        bs    := (dsum.var * ns),

        Ng    := gf*gm.func.range()/msum.domain,
        ng    := gf*gm.func.domain()/dsum.P,
        bg    := (dsum.var * ng),

        smnew := ScatMem(fCompose(sfunc, H(Ns, ns, bs, 1)), sm.pkSize/sf),
        gmnew := GathMem(fCompose(gfunc, H(Ng, ng, bg, 1)), gm.pkSize/gf),

        #Error("BP"),


        isum := When(ObjId(@(1).val)=MultiBufISum, MultiBufISumFinal, MemISumFinal),

        isum(msum.var, msum.domain,
            smnew,
            msum.rChildren()[2],    #NOTE: is this okay?
            gmnew
        )
      )
    )
));


# HACK!!!
# This is the way the system works: the above rules (RulesVRC) passes VRCL and
# VRCR tags to the scat and gath of a multibufisum. BUT, these have
# cannotchangedataformat set to true, AND, they're not exposed to the VRC
# rules. So the VRC rules don't really touch these. So in some cases, they end
# up with VRCL and VRCR, when in reality, they don't have either. The following
# rule "fixes" this problem by simply assuming they're either VRCs or VRCLRs.
# Hackity hack.
RewriteRules(RulesVRC, rec(
    VRC_MultiBufISum := Rule([@(1, [VRC,VRCL,VRCR,VRCLR]), @(2, [MultiBufISum])],
       e->let(v := @(1).val,
              m := @(2).val,
              MultiBufISum(m.var, m.domain, 
                ObjId(v)(m.scatmem, v.v),
                ObjId(v)(m._children[1], v.v), 
                ObjId(v)(m.gathmem, v.v))
              )
       ),
    VRC_MemISum := Rule([@(1, [VRC,VRCL,VRCR,VRCLR]), @(2, [MemISum])],
       e->let(v := @(1).val,
              m := @(2).val,
              MemISum(m.var, m.domain, 
                ObjId(v)(m.scatmem, v.v),
                ObjId(v)(m._children[1], v.v), 
                ObjId(v)(m.gathmem, v.v))
              )
       )
));


RewriteRules(CellVRCTerm, rec(
    VRC_ScatMem_Term := Rule([@(1, [RC, VRCL, VRCR, VRC,VRCLR]), @(2, [ScatMem])], 
    e->ScatMem(@(2).val.func, @(2).val.pkSize*2)),

    VRC_GathMem_Term := Rule([@(1, [RC, VRCL, VRCR, VRC,VRCLR]), @(2, [GathMem])], 
    e->GathMem(@(2).val.func, @(2).val.pkSize*2)),
));



RewriteRules(RulesRC, rec(
    RC_MultiBufISum := Rule([RC, @(1, MultiBufISum)],
        e -> let(s:=@(1).val, MultiBufISum( s.var, s.domain, s.scatmem, RC(s.child(1)), s.gathmem ))
    ),
));


# To convert Compose(A, B) -> ComposeStreams(A, B) when A,B=MultiBufISum
# ObjId(c)=DistSumLoop below is a hack. Obviously, DistSumLoop does not necessarily imply composing streams.
Class(RulesComposeStreams, RuleSet);
RewriteRules(RulesComposeStreams, rec(
    stream_compose := Rule(@(1, Compose, e->ForAll(e.children(), c->(ObjId(c)=MultiBufISum  or ObjId(c)=MultiBufISumFinal or ObjId(c)=DistSumLoop) )),
        e -> ComposeStreams(@(1).val.children()) )
));



# Pull Diag into MultiBufISum. D*MBufISum(SAG) -> MBufISum(SDAG) and
# MBufISum(SAG)*D -> MBufISum(SADG) Unlike the DistSum rules to do the same
# thing, this combining has to be done in a single step because the Scat and
# Gath of the MultiBufISum are not exposed.

RewriteRules(RulesDiagStandalone, rec(
 #  MBufISum(SAG)*D
 CellPullInCommuteGathDiag := ARule(Compose, [ @(1, MultiBufISum), @(2, [Prm, Gath, Diag, RCDiag]) ],
    e->let(msum := @(1).val,
           diag := @(2).val,
           gath := msum.gathmem,
           newdiag := Diag(fCompose(diag.element, fTensor(gath.func, fId(gath.pkSize)))).attrs(diag),
        [ MultiBufISum(msum.var, msum.domain, msum.scatmem, msum.child(1) * newdiag, msum.gathmem) ]
       )
 ),

 CellPullInCommuteScatDiag := ARule(Compose,  [ @(1, [RCDiag, Diag, Prm, Scat]), @(2, MultiBufISum) ],
    e->let(msum := @(2).val,
           diag := @(1).val,
           scat := msum.scatmem,
           newdiag := Diag(fCompose(diag.element, fTensor(scat.func, fId(scat.pkSize))  )).attrs(diag),
        [ MultiBufISum(msum.var, msum.domain, msum.scatmem, newdiag * msums.child(1), msum.gathmem) ]
       )
 ),

 # NOTE
 ## Gath * RCDiag
 #CellCommuteGathRCDiag := ARule( Compose,
 #      [ [@(1, [ GathDist, GathRecv ]), [@(0,fTensor), ..., [fId,@(2).cond(IsEvenInt)]]],
 #     @(4, RCDiag) ],
 # e -> [ RCDiag(fCompose(@(4).val.element, @(0).val), @(4).val.post),
 #        @(1).val ]),

 ## RCDiag * Scat
 #CellCommuteRCDiagScat := ARule( Compose,
 #      [ @(4, RCDiag),
 #    [@(1, [Scat, ScatDist, ScatSend]), [@(0,fTensor), ..., [fId,@(2).cond(IsEvenInt)]]] ],
 # e -> [ @(1).val,
 #        RCDiag(fCompose(@(4).val.element, @(0).val), @(4).val.post) ]),

));

