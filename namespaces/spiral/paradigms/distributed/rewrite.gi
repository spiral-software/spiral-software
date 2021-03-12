
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(requiredFreeVars);
Declare(buildMap);
Declare(collectLoopVars);
Declare(buildGlobalList);
Declare(compareMatrices);
Declare(doAllPromotes);
Declare(permToData);
Declare(PadLeft);
Declare(PadRight);
Declare(isContainedInComposeDistsG);
Declare(isContainedInComposeDistsS);

################################## CANDIDATES FOR REMOVAL -- BEGIN #########################3
Class(ProducOutputSDGR, RuleSet);
RewriteRules(ProducOutputSDGR, rec(

));

Class(RulesNoPull_Dist, RuleSet);
RewriteRules(RulesNoPull_Dist, rec(

    vtensor_nopulldist := Rule([@(1, VTensor), @(2, NoPull_Dist)],
        e->NoPull_Dist(VTensor(@(2).val.children(), @(1).val.vlen))
    )
));

#F Removes NoPull_Dist
Class(RemoveNoPull_Dist, RuleSet);
RewriteRules(RemoveNoPull_Dist, rec(
    remove_sizedkernel := Rule(@(1, NoPull_Dist), e->@(1).val.child(1))
));
################################## CANDIDATES FOR REMOVAL -- END #########################3



#F To convert DistSum(A)*DistSum(B)*... -> DistContainer(Seq(A,B,..))
Class(RulesComposeDists, RuleSet);
RewriteRules(RulesComposeDists, rec(
    dist_to_seq := Rule(@(1, Compose, e->ForAll(e.children(), c->(ObjId(c)=DistSum or ObjId(c)=Comm_Cell))),
        e -> ComposeDists(@(1).val.children()) )
));

#F DistSum|*|DistSum -> DistSum
Class(RulesDistMerge, RuleSet);
RewriteRules(RulesDistMerge, rec(
    #NOTE: If these DistSums have ScatSend or GathRecv, this rule will do the wrong thing! 
    # Assuming this rule will only match DMP type DFT algos.
    dist_merge := ARule(ComposeDists, [@(1,DistSum), @(2,DistSum)],
      e -> let(a := @(1).val, b := @(2).val,
          [ DistSum(a.P, a.var, a.domain, a.child(1) * b.child(1)) ])
      ),

    # Above rule produces DistSum(S * child1 * G * S * child2 * G).
    # This rule gets rid of the G * S *

    gathdist_scat_dist := ARule(Compose, [GathDist, ScatDist],
      e -> [])
));

#F Removes Buf() so other rules can work!
Class(RemoveBuf, RuleSet);
RewriteRules(RemoveBuf, rec(
    remove_buf := Rule(@(1, Buf), e->@(1).val.child(1))
));



#F Convert initial and final GathRecv/ScatSend to Dist (null)
Class(CellDFTBlockCyclicLayoutHack, RuleSet);
RewriteRules(CellDFTBlockCyclicLayoutHack, rec(

    GathRecvDataLayout := Rule([@(1, GathRecv), @(2, [H, fTensor, fCompose, fId]), ...], 
        e->GathDist(@1.val.func.range(), @1.val.pkSize, @1.val.P, @1.val.i)),

    ScatSendDataLayout := Rule([@(1, ScatSend), @(2, [H, fTensor, fCompose, fId]), ...], 
        e->ScatDist(@1.val.func.range(), @1.val.pkSize, @1.val.P, @1.val.i))

));

#F Blockcyclic layout hack
CellDFTBlockCyclicLayoutHackWrap := function(sums, opts)
    if not (IsBound(opts.doNotUseBlockCyclic) and opts.doNotUseBlockCyclic = true) then
      return(CellDFTBlockCyclicLayoutHack(sums));
    else
      return(sums);
    fi;
end;


# Terminate VRCs (VRC(Scat/Gath) -> Scat/Gath(pkSize*2..)
# (Removed VRCL, VRCR from this mix Tue 09 Sep 2008 09:55:36 PM EDT)
Class(CellVRCTerm, RuleSet);
RewriteRules(CellVRCTerm, rec(
    VRC_ScatDist_Term := Rule([@(1, [RC, VRC,VRCLR]), @(2, [ScatDist])], 
    e->ScatDist(@(2).val.N, @(2).val.pkSize*2, @(2).val.P, @(2).val.i)),

    VRC_GathDist_Term := Rule([@(1, [RC, VRC,VRCLR]), @(2, [GathDist])], 
    e->GathDist(@(2).val.N, @(2).val.pkSize*2, @(2).val.P, @(2).val.i)),

    VRC_ScatSend_Term := Rule([@(1, [RC, VRC,VRCLR]), @(2, [ScatSend])], 
    e->ScatSend(@(2).val.func, @(2).val.pkSize*2, @(2).val.P, @(2).val.i)),

    VRC_GathRecv_Term := Rule([@(1, [RC, VRC,VRCLR]), @(2, [GathRecv])], 
    e->GathRecv(@(2).val.func, @(2).val.pkSize*2, @(2).val.P, @(2).val.i)),

    VRC_Comm_Cell := Rule([@(1,[RC, VRC,VRCLR]), @(2,Comm_Cell)],
    e->Comm_Cell(@(2).val.P, @(2).val.pkSize*2))

));

# VRC(DistSum.. -> DistSum(VRC...
RewriteRules(RulesVRC, rec(
    VRC_DistSum := Rule([@(1, [VRC,VRCL,VRCR,VRCLR]), @(2, [DistSum, DistSumLoop])],
       e->let(s := @(2).val,
       CopyFields(s, rec(_children := List(s.children(), c->ObjId(@(1).val)(c, @(1).val.v)),
                  dimensions := @(1).val.dimensions))
       ))
));

RewriteRules(RulesRC, rec(
    RC_DistSum := Rule([RC, @(1, DistSum)],
        e -> let(s:=@(1).val, DistSum(s.P, s.var, s.domain, RC(s.child(1))))),
));

# Pull Diag into DistSum (but not into Gath/Scat)
RewriteRules(RulesDiag, rec(
    CellPullInDiagRight := ARule(Compose,  [ @(1, [RCDiag, Diag, Prm, Scat]), @(2, DistSum) ],
     e -> let(s:=@(2).val, [ DistSum(s.P, s.var, s.domain, @(1).val * s.child(1)) ])),

    CellPullInDiagLeft := ARule(Compose, [ @(1, DistSum), @(2, [Prm, Gath, Diag, RCDiag]) ],
     e -> let(s:=@(1).val, [ DistSum(s.P, s.var, s.domain, s.child(1) * @(2).val) ])),

));

# Pull Diag into SAG (D*SAG -> SDAG and SAG*D -> SADG)
RewriteRules(RulesDiagStandalone, rec(
 # Gath * Diag
 CellCommuteGathDiag := ARule( Compose,
       [ @(1, [ GathDist, GathRecv ]), @(2, Diag) ], # o 1-> 2->
  e -> [ Diag(fCompose(@2.val.element, fTensor(@1.val.func, fId(@1.val.pkSize)))).attrs(@(2).val), @1.val ]),

 # Diag * Scat
 CellCommuteDiagScat := ARule( Compose,
       [ @(1, Diag), @(2, [ScatDist, ScatSend]) ], # <-1 <-2 o
  e -> [ @2.val, Diag(fCompose(@1.val.element, fTensor(@2.val.func, fId(@2.val.pkSize))  )).attrs(@(1).val) ]),

 # Gath * RCDiag
 CellCommuteGathRCDiag := ARule( Compose,
       [ [@(1, [ GathDist, GathRecv ]), [@(0,fTensor), ..., [fId,@(2).cond(IsEvenInt)]]],
      @(4, RCDiag) ],
  e -> [ RCDiag(fCompose(@(4).val.element, @(0).val), @(4).val.post),
         @(1).val ]),

 # RCDiag * Scat
 CellCommuteRCDiagScat := ARule( Compose,
       [ @(4, RCDiag),
     [@(1, [Scat, ScatDist, ScatSend]), [@(0,fTensor), ..., [fId,@(2).cond(IsEvenInt)]]] ],
  e -> [ @(1).val,
         RCDiag(fCompose(@(4).val.element, @(0).val), @(4).val.post) ]),

));


# Cell DMP: Fuse and remove PTensors that are next to Sigmas
Class(PTensorRules, RuleSet);
RewriteRules(PTensorRules, rec(
PTensorFuseRight := ARule(Compose, [ @(1,DistSum), @(2,PTensor) ], 
    e -> [ DistSum( @(1).val.P, @(1).val.var, @(1).val.domain, Compose( @(1).val.child(1), @(2).val ) ) ] ),

PTensorFuseLeft := ARule(Compose, [  @(2,PTensor), @(1,DistSum) ], 
    e -> [ DistSum( @(1).val.P, @(1).val.var, @(1).val.domain, Compose( @(2).val,  @(1).val.child(1) ) ) ] ),

PTensorFlipRight := ARule(Compose, [ @(1,GathDist), @(2,PTensor) ],
    e -> [ @(2).val.L, @(1).val ] ),

PTensorFlipLeft := ARule(Compose, [ @(2,PTensor), @(1,ScatDist) ],
    e -> [ @(1).val, @(2).val.L ] )
));

#i := Ind(P),
# Cell DMP: Convert standalone PTensors to Sigmas
Class(PTensorConvertRules, RuleSet);
RewriteRules(PTensorConvertRules, rec(
PTensorConvert := Rule( @(1,PTensor),
    e -> let(P := @(1).val.P,
             i := var("spuid", TInt, P),
             N := @(1).val.dims()[1],
            DistSum(P, i, P, ScatDist(N, 1, P, i) * @(1).val.L * GathDist(N, 1, P, i))
          )
    )
));

#F Gets all the DistSums under a single Compose so we can deal with them effectively
#F Mainly there because of VContainer
Class(DistSumChains, RuleSet);
RewriteRules(DistSumChains, rec(
    ChainDistSumsLeft := ARule(Compose,  [ @(1,DistSum), @(2,VContainer, e->ForAny(e.rChildren()[1].rChildren(), i->ObjId(i)=DistSum)) ],
     e -> let(ds:=@(1).val, vc:=@(2).val, [ VContainer(Compose(ds, vc.rChildren()), vc.isa)  ])),

    ChainDistSumsRight := ARule(Compose,  [ @(2,VContainer,  e->ForAny(e.rChildren()[1].rChildren(), i->ObjId(i)=DistSum)), @(1,DistSum) ],
     e -> let(ds:=@(1).val, vc:=@(2).val, [ VContainer(Compose(vc.rChildren(), ds), vc.isa)  ]))
));

#F Fuse output (SDGR*SS...) to just an SS
Class(FixBorder, RuleSet);
RewriteRules(FixBorder, rec(
    FixLeftBorder := Rule( [@(1,ComposeDists), [DistSum, ..., [Compose, ScatDist, GathRecv]], ...],
     e ->  let(c := @(1).val.rChildren(), ComposeDists(ListWithout(c,1))) )
));

#F Mark Cell Scat/Gaths as Inplace (ultimately, no-ops)
#F GDs, SDs, and GRs are no-ops for the Cell (SS becomes explicit DMA - SCATSEND)
applyCellInplace := function(sums, opts)
    SubstBottomUp(sums, GathDist, e->Inplace(e));
    SubstBottomUp(sums, GathRecv, e->Inplace(e));
    SubstBottomUp(sums, ScatDist, e->Inplace(e));


    # BB(Inplace()) -> Inplace() (So that a copy operation need not take place)
    SubstBottomUp(sums, [BB, Inplace], e->e.rChildren()[1]);
    return(sums);
end;

#NOTE: In the following padding functions, we previously produced pads with
#maximum packet size so that they merged nicely with multibuffered scats and
#gaths. But now, we don't do any merging between scats and gaths of the
#distributed and multibuffer paradigms. So it's better to get these padding
#functions to produce packet sizes that match the adjacent sigmaspl expressions.

#F This produces an SDGR on the left
PadLeft := function(e)
    local ss, N, pkSize, P,i, GR, SD;
    ss      := e.leftMostParScat();
    N       := ss.func.range();
    pkSize  := ss.pkSize;
    P       := ss.P;
    i       := var("spuid", TInt, P);
    GR      := GathRecv(fTensor(fBase(P, i), fId(N/P)), pkSize, P, i);
    SD      := ScatDist(N, pkSize, P, i);
    #SD     := ScatDist(P, (N*pkSize)/P, P, i); # Produce ScatDist with maximum packet size
    return(ComposeDists(DistSum(P, i, P, SD*GR),   e.rChildren()));
end;

#F This produces an SDGR on the left
PadLeftSep := function(e)
    local ss, N, pkSize, P,i, GR, SD;
    ss      := e.leftMostParScat();
    N       := ss.func.range();
    pkSize  := ss.pkSize;
    P       := ss.P;
    i       := var("spuid", TInt, P);
    GR      := GathRecv(fTensor(fBase(P, i), fId(N/P)), pkSize, P, i);
    SD      := ScatDist(N, pkSize, P, i);
    #SD     := ScatDist(P, (N*pkSize)/P, P, i); # Produce ScatDist with maximum packet size
    return(ComposeDists(DistSum(P, i, P, SD*GR), e));
end;

#F This produces a SSGD on the right
PadRight := function(e)
   local gr, N, P, i, pkSize, SS, GD, di;
   gr     := e.rightMostParGath();
   N      := gr.func.range();
   pkSize := gr.pkSize;
   P      := gr.P;
   i      := var("spuid", TInt, P);
   SS     := ScatSend(fAdd(N, N/P, i*(N/P)), pkSize, P, i);
   GD     := GathDist(N, pkSize, P, i);
   #GD     := GathDist(P, (N*pkSize)/P, P, i); # Produce GathDist with maximum packet size
   return(ComposeDists(e.rChildren(), DistSum(P, i, P, SS*GD)));
end;

#F This produces a SSGD on the right
PadRightSep := function(e)
   local gr, N, P, i, pkSize, SS, GD, di;
   gr     := e.rightMostParGath();
   N      := gr.func.range();
   pkSize := gr.pkSize;
   P      := gr.P;
   i      := var("spuid", TInt, P);
   SS     := ScatSend(fAdd(N, N/P, i*(N/P)), pkSize, P, i);
   GD     := GathDist(N, pkSize, P, i);
   #GD     := GathDist(P, (N*pkSize)/P, P, i); # Produce GathDist with maximum packet size
   return(ComposeDists(e, DistSum(P, i, P, SS*GD)));
end;


# These are required only for the CellSMP model, but shouldn't affect DMP, since DMP doesn't inject any SSs or GRs.
#F applyCellRules: 
#F Assumes:
#F - There are no Buf()s in code
#F - Composes within a parallel region have already been converted to ComposeDistss
applyCellRules := function(sums, opts)
    local sumsorig, ss, sdgr, gr, ssgd, border;

#applyCellRules should not touch anything within a GT_ParStream. However, a
#sums expression can contain a mixture of ParStream and StreamChip generated
#expressions. We want to leave the former untouched, but want to act on the
#latter. We do this by marking things to be left untouched, and unmarking them at the end. This has 2 PRORLBEMS:
#
# 1) We shouldn't run ComposeDists before applyCellRules, though we also need to. Fix this.
# 2) If there's a GT_StreamPar rule, this whole thing falls apart.
#
# Essentially, we need a way of figuring out which were generated by StreamCore vs. StreamChip.
# Another approach: we could fuse the par and mbuf loops together via StreamCore's breakdown rule.
# In general, we could take care of StreamCore structures before we execute
# applyCellRules -- we could run DistMBuf in sigmaspl earlier, for instance.




    # Promote {Scat,Gath}Dist to ScatSend/GathRecv as needed
    SubstTopDown(sums, [ComposeDists, DistSum, DistSum], doAllPromotes);

    # Right-Pad DistSums that are standing by themselves (NOTE: cleanup this and the functions it calls)
    # Left-padding will be taken care of by statement below
    # NOTE: assumption here is, this is only for DFTxI.
    if IsBound(opts.doNotUseBlockCyclic) and opts.doNotUseBlockCyclic = true then
      SubstBottomUp(sums, @(1, DistSum, e->ObjId(e.rightMostParGath())=GathRecv), isContainedInComposeDistsG);
      SubstBottomUp(sums, @(1, DistSum, e->ObjId(e.leftMostParScat())=ScatSend),  isContainedInComposeDistsS);
    fi;

    # I/O padding: Don't do any input/output padding if these might get cancelled as a part of the data format change
    if IsBound(opts.doNotUseBlockCyclic) and opts.doNotUseBlockCyclic = true then
        SubstTopDown(sums, @(1, ComposeDists, e->ObjId(e.leftMostParScat())=ScatSend),  e->PadLeft(e));
        SubstTopDown(sums, @(1, ComposeDists, e->ObjId(e.rightMostParGath())=GathRecv),  e->PadRight(e));
    fi;

    #Error("BP");

    # NOTE: We shouldn't be doing this if we have a compose of multiple ParStream structures, for instance
    # Create table to map ScatSend/GathRecv pairs, so DMA can be done

    # NOTE: Why is this an else if structure? For StreamParChip, where we have
    # multiple stream stages each of which contain parallelism, all these
    # patterns could exist within the same sums expression.

    SubstTopDown(sums, [ComposeDists, DistSum, DistSum], permToData);
    SubstTopDown(sums, [ComposeDists, DistSum, DistSum, DistSum], permToData);
    SubstTopDown(sums, [ComposeDists, DistSum, DistSum, DistSum, DistSum], permToData);


    #if Length(Collect(sums, [ComposeDists, DistSum, DistSum])) = 1 then
    #     SubstTopDown(sums, [ComposeDists, DistSum, DistSum], permToData);
    #     else if Length(Collect(sums, [ComposeDists, DistSum, DistSum, DistSum])) = 1 then
    #               SubstTopDown(sums, [ComposeDists, DistSum, DistSum, DistSum], permToData);
    #     else if Length(Collect(sums, [ComposeDists, DistSum, DistSum, DistSum, DistSum])) = 1 then
    #               SubstTopDown(sums, [ComposeDists, DistSum, DistSum, DistSum, DistSum], permToData);
    #          fi;
    #          fi;
    #fi;

    # Fuse output (SDGR*SS...) to just an SS
    FixBorder(sums);

    return(sums);
end;


doAllPromotes := function(e, cx)
   local gathrecv, gathdist, scatsend, scatdist, i;

   # Assume: only one ScatSend and one GathRecv per DistSum.
   i := 1;

   # Check for GR/SD pair
   gathrecv := Collect(e._children[i],   GathRecv);
   scatdist := Collect(e._children[i+1], ScatDist);

   #Error("BP-doAllPromotes");

   if (Length(gathrecv) = 1 and Length(scatdist) = 1) then
     SubstTopDown(e, ScatDist, 
        gs-> let(pkf := gs.pkSize / gathrecv[1].pkSize,
            ScatSend(fAdd(gs.N*pkf, gs.N*pkf/gs.P, gs.i*(gs.N*pkf/gs.P)), gs.pkSize/pkf, gs.P, gs.i))
     );
     return(e);
   fi;

   # Check for GD/SS pair
   gathdist := Collect(e._children[i],   GathDist);
   scatsend := Collect(e._children[i+1], ScatSend);

   if (Length(gathdist) = 1 and Length(scatsend) = 1) then
     SubstTopDown(e, GathDist,
        gs-> let(pkf := gs.pkSize / scatsend[1].pkSize,
            GathRecv(fAdd(gs.N, gs.N/gs.P, gs.i*(gs.N*pkf/gs.P)), gs.pkSize, gs.P, gs.i))
            );
     return(e);
   fi;

   return(e);

end;



#F permToData(e, cx) (e=expression, cx=context)
#F converts an expressions's function to an FList/FData
permToData := function(e, cx)
#permToData := function(d1, d2, cx)
   local gathrecv, scatsend, readMap, writeMap, loopnest, i;


   #gathrecv := e.rChildren()[1].rChildren()[2].rChildren()[Length(e.rChildren()[1].rChildren()[2].rChildren())];
   #scatsend := e.rChildren()[2].rChildren()[2].rChildren()[1];


   # Assume: only one ScatSend and one GathRecv per DistSum (no nested parallelism)
   for i in [1..(Length(Collect(e, DistSum))-1)] do
      gathrecv := Collect(e._children[i], GathRecv)[1];
      scatsend := Collect(e._children[i+1], ScatSend)[1];

      #PrintLine("permToData working on: ", gathrecv, " / ", scatsend);

      if gathrecv.func.__name__ = "FData" or scatsend.func.__name__ = "FData" then
         if gathrecv.func.__name__ = "FData" and scatsend.func.__name__ = "FData" then
            Error("Both are already FData. Why?\n");
         else
            Error("One of these funcs is an FData, the other is not.\n");
            #NOTE: Handle things correctly if both are already FDatas/FDataLists
         fi;
      fi;

      #loopnest := Concat(collectLoopVars(e, cx), [e.rChildren()[i].var, e.rChildren()[i+1].var]);
      loopnest := Concat(collectLoopVars(e, cx), [e._children[i].var, e._children[i+1].var]);

      [readMap, writeMap] := buildGlobalList(gathrecv, scatsend, loopnest);

      #NOTE: Use rSetChild or use .func directly here?
      SubstTopDown(e._children[i], GathRecv, e->GathRecv(FData([0..Length(readMap)-1]), e.pkSize, e.P, e.i));
      SubstTopDown(e._children[i+1], ScatSend, e->ScatSend(FData(readMap), e.pkSize, e.P, e.i));

      # Since we're doing a Gather-side normalize for the cell (the scatter
      # knows where to put an element, the gather is dumb), we write the
      # normalized FData instead of the actual readMap here.

   od;

   #PrintLine(gathrecv, "\n", scatsend, "\n", readMap, writeMap, "\n--\n", scatsend, "\n------------------\n\n");
   return e;
end;


#F Goes through all parents of e as given in cx.parents, and creates a list of
#F all parent loop variables in loop nest order.
collectLoopVars := function(e, cx)
   local parent, retList;
   retList := [];

   for parent in cx.parents do
      if (IsBound(parent.isSums) and parent.isSums = true) then
         if IsBound(parent.var) then
            retList := Concat(retList, [parent.var]);
         fi;
      fi;
   od;

   return retList;
end;

#F buildGlobalList(gathrecv, scatsend, loopnest)
#F Returns the read and write maps (as lists) for a given gath/scat pair
#F loopnest is a list of loop vars ordered by the gath/scat nesting
buildGlobalList := function(gathrecv, scatsend, loopnest)
   local i, j, gathMap, scatMap, t, readMap, writeMap;

   #NOTE: Need to add some error checking.

   gathMap := buildMap(gathrecv, loopnest);
   scatMap := buildMap(scatsend, loopnest);

   if Length(gathMap) <> Length(scatMap) then
      Error("buildGlobalList: packet sizes for scatter and gather don't match.");
   fi;

   t       := List([1..Length(gathMap)], i->0);
   readMap := List([1..Length(gathMap)], i->0);
   writeMap:= List([1..Length(gathMap)], i->0);


   for i in [1..Length(gathMap)] do
      t[gathMap[i]] := i;
   od;

   for i in [1..Length(gathMap)] do
      readMap[i] := t[scatMap[i]];
   od;

   for i in [1..Length(gathMap)] do
      for j in [1..Length(gathMap)] do
         if readMap[j] = i then
            writeMap[i] := j;
         fi;
      od;
   od;

   #NOTE: assuming that element values will range from 0..(n-1).
   readMap := List([1..Length(readMap)], i->readMap[i]-1);
   writeMap := List([1..Length(writeMap)], i->writeMap[i]-1);


   return [readMap, writeMap];
end;

#F buildMap(gathscat, loopnest)
#F Fully unrolls a gath/scat's function to a list based on the loop nest ordering given by loopnest
#F If loopnest has vars that the gathscat doesn't depend on, they will be ignored
buildMap := function(gathscat, loopnest)
   local map, reqFreeVars, loopVar;

   reqFreeVars := requiredFreeVars(loopnest, Flat(gathscat.free()));
   map := gathscat.func.lambda().tolist();
   for loopVar in reqFreeVars do
       map := Flat( List([0..(loopVar.range)-1], i->SubstVars(Copy(map), rec((loopVar.id) := V(i)))) );
   od;
   map := List([1..Length(map)], i->(map[i].ev())+1);

   return map;
end;

#F requiredFreeVars(loopnest, freevars)
#F 
requiredFreeVars := function(loopnest, freevars)
   local returnList, i;
   returnList := [];
   for i in loopnest do
      if i in freevars then
         if not i in returnList then
            returnList := Concat(returnList, [i]);
         fi;
      fi;
   od;
   return returnList;
end;


isContainedInComposeDistsG := function(e,cx)
  local parent, foundComposeDist;
  foundComposeDist := false;
  for parent in cx.parents do
     if ObjId(parent)=ComposeDists then foundComposeDist := true; fi;
  od;

  if foundComposeDist=true then
     return(e);
  else
     return(PadRightSep(e));
  fi;
end;

isContainedInComposeDistsS := function(e,cx)
  local parent, foundComposeDist;
  foundComposeDist := false;
  for parent in cx.parents do
     if ObjId(parent)=ComposeDists then foundComposeDist := true; fi;
  od;

  if foundComposeDist=true then
     return(e);
  else
     return(PadLeftSep(e));
  fi;
end;


