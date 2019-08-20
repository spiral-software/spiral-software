
# Copyright (c) 2018-2019, Carnegie Mellon University
# See LICENSE for details

####libgen + par + vector

[opts,nt,flops,verif]:=doParSimdMMM(1,2,1,true,4,false);
 base_opts:=Copy(opts);
 base_opts.globalUnrolling := opts.globalUnrolling;
 base_opts.breakdownRules  := Copy(opts.breakdownRules)  ;
 base_opts.breakdownRules.MMM :=[ MMM_BaseMult, MMM_KernelReached, MMM_TileHorizontally, MMM_TileVerticaly,
  MMM_TileDepth, MMM_Padding, MMM_VerticalIndependance];
 base_opts.hashFile        := let(p:=Conf("path_sep"),
        Concat(Conf("spiral_dir"), p, "spiral", p, "libgen", p, "bases10.MMMhash"));
 #base_opts.benchTransforms := [MMM(1, 1, 1), MMM(2,2,2)];
 base_opts.benchTransforms := [MMM(4,2,4,[ AVecReg(SSE_2x64f) ])];
 
 bench:= CopyFields(DPBench(rec(bases := base_opts), rec(verbosity:=0)),
                       rec(generateCfiles := false, generateSums := true, matrixVerify := false));
 bench.resumeAll();

 opts.libgen:=rec();
 opts.libgen.basesUnrolling := 2^16;
 opts.libgen.baseBench := bench;
 opts.libgen.codeletTab := CreateCodeletHashTable();
 opts.libgen.terminateStrategy := [ HfuncSumsRules ];
 opts.formulaStrategies.postProcess:=DropLast(opts.formulaStrategies.postProcess,1); #drop BBs
 Add(opts.formulaStrategies.postProcess,HfuncSumsRules);
 Add(opts.formulaStrategies.postProcess,LibgenVectorHackRuleset);
 Add(opts.formulaStrategies.postProcess,OLCrossPullInRules);
 Add(opts.formulaStrategies.postProcess,OLVectorPropagateRuleset);
 Add(opts.formulaStrategies.postProcess,RecursStepTerm);
opts.baseHashes := [ CreateRecursBaseHash(bench.exp.bases.hashTable) ];

nt:=MMM(16,16,16,[AParSMP(4),   AVecReg(SSE_2x64f) ]);
rt:=RandomRuleTree(nt, opts);
s:=SumsRuleTree(nt,opts);
c:=CodeRuleTreeOpts(rt,opts);
CMeasure(c,opts);
CVerifyMMM(c,opts,nt);

--------------------------

#libgen + gauss + newMMM +unrolled scalar

 [opts,nt,flops,verif]:=doParSimdMMM(1,2,1,false,1,false);
 base_opts:=Copy(opts);
 base_opts.globalUnrolling := 2^16;
 base_opts.breakdownRules  := Copy(opts.breakdownRules)  ;
 base_opts.hashFile        := let(p:=Conf("path_sep"),
        Concat(Conf("spiral_dir"), p, "spiral", p, "libgen", p, "bases32.MMMhash"));
 base_opts.breakdownRules.KernelMMM :=[ KernelMMM_Base, KernelMMM_TileH, KernelMMM_TileV, KernelMMM_TileD];
 base_opts.benchTransforms := [KernelMMM(1,1,1),KernelMMM(3,3,3)];
 
 bench:= CopyFields(DPBench(rec(bases := base_opts), rec(verbosity:=0)),
                       rec(generateCfiles := false, generateSums := true, matrixVerify := false));
 bench.resumeAll();
 g:=GaussianPredictor;

 opts.breakdownRules.FactorMMM :=[ FactorMMM_Base, FactorMMM_TileH, FactorMMM_TileV, FactorMMM_TileD];
 opts.libgen:=rec();
 opts.libgen.basesUnrolling := 2^16;
 opts.libgen.baseBench := bench;
 opts.libgen.codeletTab := CreateCodeletHashTable();
 opts.libgen.terminateStrategy := [ HfuncSumsRules ];
 opts.formulaStrategies.postProcess:=DropLast(opts.formulaStrategies.postProcess,1); #drop BBs
 Add(opts.formulaStrategies.postProcess,HfuncSumsRules); 
 Add(opts.formulaStrategies.postProcess,LibgenVectorHackRuleset);
 Add(opts.formulaStrategies.postProcess,OLCrossPullInRules);
 Add(opts.formulaStrategies.postProcess,OLVectorPropagateRuleset);
 Add(opts.formulaStrategies.postProcess,RecursStepTerm);

 t:=[7,4,8];
 k:=g.findMax(bench,KernelMMM,CartesianProduct(List(t,x->DivisorsInt(x))),[]);
 opts.baseHashes := [ CreateRecursBaseHash(bench.exp.bases.hashTable) ];
 nt:=ApplyFunc(FactorMMM,Concat(t,[k]));
 res:=DP(nt,rec(),opts);
 r:=res[1].ruletree;
 c:=CodeRuleTreeOpts(r,opts);
 _compute_mflops(nt.normalizedArithCost(),CMeasure(c,opts));
 CVerifyMMM(c,opts,nt);

--------------
#libgen + gauss + newMMM + looped scalar compiler vectorized 
#needs high dimensional stride to do compiler vectorization. vstride H


 [opts,nt,flops,verif]:=doParSimdMMM(1,2,1,true,1,false);
 opts.profile.makeopts.CFLAGS:=Concat(opts.profile.makeopts.CFLAGS," -fno-alias -fno-fnalias -save-temps -fno-inline-functions");
   opts.compileStrategy:=NoCSE;
   opts.useDeref := false;
   opts.propagateNth := true;
   opts.doScalarReplacement := false;

#add some flags here : c99 restrict assume-aligned xN

 base_opts:=Copy(opts);
 base_opts.globalUnrolling := 5;
 base_opts.breakdownRules  := Copy(opts.breakdownRules)  ;
 base_opts.hashFile        := let(p:=Conf("path_sep"),
        Concat(Conf("spiral_dir"), p, "spiral", p, "libgen", p, "bases156.MMMhash"));
 base_opts.breakdownRules.KernelMMM :=[ KernelMMM_Base, KernelMMM_TileH, KernelMMM_TileV, KernelMMM_TileD];
 base_opts.benchTransforms := [KernelMMM(1,1,1),KernelMMM(3,3,3)];
 
 bench:= CopyFields(DPBench(rec(bases := base_opts), rec(verbosity:=0)),
                       rec(generateCfiles := false, generateSums := true, matrixVerify := false));
 bench.resumeAll();
 g:=GaussianPredictor;

 opts.breakdownRules.FactorMMM :=[ FactorMMM_Base, FactorMMM_TileH, FactorMMM_TileV, FactorMMM_TileD];
 opts.libgen:=rec();
 opts.libgen.basesUnrolling := 10;
 opts.libgen.baseBench := bench;
 opts.libgen.codeletTab := CreateCodeletHashTable();
 opts.libgen.terminateStrategy := [ HfuncSumsRules ];
 opts.formulaStrategies.postProcess:=DropLast(opts.formulaStrategies.postProcess,1); #drop BBs
 Add(opts.formulaStrategies.postProcess,HfuncSumsRules); 
 Add(opts.formulaStrategies.postProcess,LibgenVectorHackRuleset);
 Add(opts.formulaStrategies.postProcess,OLCrossPullInRules);
 Add(opts.formulaStrategies.postProcess,OLVectorPropagateRuleset);
 Add(opts.formulaStrategies.postProcess,RecursStepTerm);

 t:=[7,8,2];
 k:=g.findMax(bench,KernelMMM,CartesianProduct(List(t,x->DivisorsInt(x))),[]);
 opts.baseHashes := [ CreateRecursBaseHash(bench.exp.bases.hashTable) ];
 nt:=ApplyFunc(FactorMMM,Concat(t,[k]));
 res:=DP(nt,rec(),opts);
 r:=res[1].ruletree;
 c:=CodeRuleTreeOpts(r,opts);
 _compute_mflops(nt.normalizedArithCost(),CMeasure(c,opts));
 CVerifyMMM(c,opts,nt);


------------------
#libgen + gauss + newMMM + unrolled vector dup-in

 [opts,nt,flops,verif]:=doParSimdMMM(1,2,1,true,1,false);
 opts.profile.makeopts.CFLAGS:=Concat(opts.profile.makeopts.CFLAGS," -fno-alias -fno-fnalias -save-temps -fno-inline-functions");

 base_opts:=Copy(opts);
 base_opts.globalUnrolling := 2^16;
 base_opts.breakdownRules  := Copy(opts.breakdownRules)  ;
 base_opts.hashFile        := let(p:=Conf("path_sep"),
        Concat(Conf("spiral_dir"), p, "spiral", p, "libgen", p, "bases71.MMMhash"));
 base_opts.breakdownRules.KernelMMM :=[ KernelMMM_Base, KernelMMM_TileH, KernelMMM_TileV, KernelMMM_IndV, KernelMMM_TileD];
 base_opts.benchTransforms := [KernelMMM(1,2,1,[ AVecReg(SSE_2x64f) ]),KernelMMM(3,4,3,[ AVecReg(SSE_2x64f) ])];
 
 bench:= CopyFields(DPBench(rec(bases := base_opts), rec(verbosity:=0)),
                       rec(generateCfiles := false, generateSums := true, matrixVerify := false));
 bench.resumeAll();
 g:=GaussianPredictor;

 opts.breakdownRules.FactorMMM :=[ FactorMMM_Base, FactorMMM_TileH, FactorMMM_TileV, FactorMMM_TileD];
  opts.libgen:=rec();
 opts.libgen.basesUnrolling := 2^16;
 opts.libgen.baseBench := bench;
 opts.libgen.codeletTab := CreateCodeletHashTable();
 opts.libgen.terminateStrategy := [ HfuncSumsRules ];
 opts.formulaStrategies.postProcess:=DropLast(opts.formulaStrategies.postProcess,1); #drop BBs
 Add(opts.formulaStrategies.postProcess,HfuncSumsRules); 
 Add(opts.formulaStrategies.postProcess,LibgenVectorHackRuleset);
 Add(opts.formulaStrategies.postProcess,OLCrossPullInRules);
 Add(opts.formulaStrategies.postProcess,OLVectorPropagateRuleset);
 Add(opts.formulaStrategies.postProcess,RecursStepTerm);

 t:=[12,12,2];
 k:=g.findMax(bench,KernelMMM,Filtered(CartesianProduct(List(t,x->DivisorsInt(x))),
     a->Mod(a[2],AVecReg(SSE_2x64f).v)=0),[AVecReg(SSE_2x64f)]);
 opts.baseHashes := [ CreateRecursBaseHash(bench.exp.bases.hashTable) ];
 nt:=ApplyFunc(FactorMMM,Concat(t,[k]));
 res:=DP(nt,rec(),opts);
 r:=res[1].ruletree;
 c:=CodeRuleTreeOpts(r,opts);
 _compute_mflops(nt.normalizedArithCost(),CMeasure(c,opts));
 CVerifyMMM(c,opts,nt);



--------------------------
Flame is tail-recursion : on donne une loi de breakdown -> retourne un algorithme iteratif
FFTW gere les boucles en introduisant des objets boucles. Des breakdowns peuvent etre directement appliques aux objets boucles.
DFT-> IxDFT * DFTxI   mais DFTxI -> IxDFTxI * DFTxIxI donc DFT -> IxDFT * (IxDFTxI * DFTxIxI) depth2 loops side by side or 
DFT -> IxDFT * (IxDFT * DFTxI)xI depth 1loop containing 2 depth1loops 

fU(H,H)
HH Descend

------------------------
#libgen + gauss + PaddedMMM + unrolled vector dup-in

 [opts,nt,flops,verif]:=doParSimdMMM(1,2,1,true,1,false);
 opts.profile.makeopts.CFLAGS:=Concat(opts.profile.makeopts.CFLAGS," -fno-alias -fno-fnalias -save-temps -fno-inline-functions");

 base_opts:=Copy(opts);
 base_opts.globalUnrolling := 2^16;
 base_opts.breakdownRules  := Copy(opts.breakdownRules)  ;
 base_opts.hashFile        := let(p:=Conf("path_sep"),
        Concat(Conf("spiral_dir"), p, "spiral", p, "libgen", p, "bases71.MMMhash"));
 base_opts.benchTransforms := [KernelMMM(1,2,1,[ AVecReg(SSE_2x64f) ]),KernelMMM(3,4,3,[ AVecReg(SSE_2x64f) ])];
 
 bench:= CopyFields(DPBench(rec(bases := base_opts), rec(verbosity:=0)),
                       rec(generateCfiles := false, generateSums := true, matrixVerify := false));
 bench.resumeAll();
 g:=GaussianPredictor;

 opts.libgen:=rec();
 opts.libgen.basesUnrolling := 2^16;
 opts.libgen.baseBench := bench;
 opts.libgen.codeletTab := CreateCodeletHashTable();
 opts.libgen.terminateStrategy := [ HfuncSumsRules ];
 opts.formulaStrategies.postProcess:=DropLast(opts.formulaStrategies.postProcess,1); #drop BBs
 Add(opts.formulaStrategies.postProcess,HfuncSumsRules); 
 Add(opts.formulaStrategies.postProcess,LibgenVectorHackRuleset);
 Add(opts.formulaStrategies.postProcess,OLCrossPullInRules);
 Add(opts.formulaStrategies.postProcess,OLVectorPropagateRuleset);
 Add(opts.formulaStrategies.postProcess,RecursStepTerm);
 Add(opts.formulaStrategies.postProcess,(s, opts) -> compiler.BlockSums(opts.globalUnrolling, s));
 opts.globalUnrolling := 10;


 obj:=[39,48,39];
 t:=[6,6,6];
space:=CartesianProduct(List(t,x->[1..x]));
 k:=g.findMax(bench,KernelMMM,Filtered(space,
     a->Mod(a[2],AVecReg(SSE_2x64f).v)=0),[AVecReg(SSE_2x64f)],ScaleMMMMFLOPS(0.50,obj));
 opts.baseHashes := [ CreateRecursBaseHash(bench.exp.bases.hashTable) ];
 nt:=PaddedMMM(obj[1],obj[2],obj[3],k);
 res:=DP(nt,rec(),opts);
 r:=res[1].ruletree;
 s:=SumsRuleTree(r,opts);
c:=CodeRuleTreeOpts(r,opts);
 _compute_mflops(nt.normalizedArithCost(),CMeasure(c,opts));
 CVerifyMMM(c,opts,nt);

----------------------------------------------------------------
#save the function and not the inlined

 [opts,nt,flops,verif]:=doParSimdMMM(1,2,1,true,1,false);
 opts.profile.makeopts.CFLAGS:=Concat(opts.profile.makeopts.CFLAGS," -fno-alias -fno-fnalias -save-temps -fno-inline-functions");
 base_opts:=Copy(opts);
 base_opts.globalUnrolling := 2^16;
 base_opts.breakdownRules  := Copy(opts.breakdownRules)  ;
 base_opts.hashFile        := let(p:=Conf("path_sep"),
        Concat(Conf("spiral_dir"), p, "spiral", p, "libgen", p, "bases71.MMMhash"));
 base_opts.benchTransforms := [KernelMMM(1,2,1,[ AVecReg(SSE_2x64f) ]),KernelMMM(3,4,3,[ AVecReg(SSE_2x64f) ])]; 
 bench:= CopyFields(DPBench(rec(bases := base_opts), rec(verbosity:=0)),
                       rec(generateCfiles := false, generateSums := true, matrixVerify := false));
 bench.resumeAll();
 g:=GaussianPredictor;

 opts.libgen:=rec();
 opts.libgen.basesUnrolling := 2^16;
 opts.libgen.baseBench := bench;
 opts.libgen.codeletTab := CreateCodeletHashTable();
 opts.libgen.terminateStrategy := [ HfuncSumsRules ];
 opts.formulaStrategies.postProcess:=DropLast(opts.formulaStrategies.postProcess,1); #drop BBs
 Add(opts.formulaStrategies.postProcess,HfuncSumsRules); 
 Add(opts.formulaStrategies.postProcess,LibgenVectorHackRuleset);
 Add(opts.formulaStrategies.postProcess,OLCrossPullInRules);
 Add(opts.formulaStrategies.postProcess,OLVectorPropagateRuleset);
 Add(opts.formulaStrategies.postProcess,RecursStepTerm);
 Add(opts.formulaStrategies.postProcess,(s, opts) -> compiler.BlockSums(opts.globalUnrolling, s));
 opts.globalUnrolling := 10;

 obj:=[39,48,39];
 t:=[6,6,6];
 space:=CartesianProduct(List(t,x->[1..x]));;
 k:=g.findMax(bench,KernelMMM,Filtered(space,
     a->Mod(a[2],AVecReg(SSE_2x64f).v)=0),[AVecReg(SSE_2x64f)],ScaleMMMMFLOPS(0.50,obj));
 opts.baseHashes := [ CreateRecursBaseHash(bench.exp.bases.hashTable) ];
 nt:=PaddedMMM(obj[1],obj[2],obj[3],k);
 res:=DP(nt,rec(),opts);
 r:=res[1].ruletree;
 s:=SumsRuleTree(r,opts);
c:=CodeRuleTreeOpts(r,opts);
 _compute_mflops(nt.normalizedArithCost(),CMeasure(c,opts));
 CVerifyMMM(c,opts,nt);



================================================================================
 opts := SetMMMOptions(true,1,false);
 base_opts:=CopyFields(opts, rec(
                globalUnrolling := 2^16,
                hashFile        := let(p:=Conf("path_sep"),
                    Concat(Conf("spiral_dir"), p, "spiral", p, "libgen", p, "MMM-bases.hash")),
                benchTransforms := [KernelMMM(1,2,1,[ AVecReg(SSE_2x64f) ]),KernelMMM(3,4,3,[ AVecReg(SSE_2x64f) ])])); 
 base_bench:= CopyFields(DPBench(rec(bases := base_opts), rec(verbosity:=0)),
                       rec(generateCfiles := false, generateSums := true, matrixVerify := false));


 libgen_opts:=CopyFields(opts, rec(
                globalUnrolling := 10,
                hashFile        := let(p:=Conf("path_sep"),
                    Concat(Conf("spiral_dir"), p, "spiral", p, "libgen", p, "MMM-libgen.hash")),
                benchTransforms := [],
                formulaStrategies := Copy(opts.formulaStrategies),
                verbosity:=1,
                libgen:=rec(basesUnrolling := 2^16, baseBench := base_bench, 
                        codeletTab := CreateCodeletHashTable(), terminateStrategy := [ HfuncSumsRules ]))); 
 libgen_opts.formulaStrategies.postProcess:=DropLast(opts.formulaStrategies.postProcess,1);      #drop BBs
 Add(libgen_opts.formulaStrategies.postProcess,HfuncSumsRules); 
 Add(libgen_opts.formulaStrategies.postProcess,LibgenVectorHackRuleset);
 Add(libgen_opts.formulaStrategies.postProcess,OLCrossPullInRules);
 Add(libgen_opts.formulaStrategies.postProcess,OLVectorPropagateRuleset);
 Add(libgen_opts.formulaStrategies.postProcess,RecursStepTerm);
 Add(libgen_opts.formulaStrategies.postProcess,(s, opts) -> compiler.BlockSums(opts.globalUnrolling, s));
 libgen_bench:= CopyFields(DPBench(rec(bases := libgen_opts), rec(verbosity:=0)),
                       rec(generateCfiles := false, generateSums := true, matrixVerify := false));
 libgen_bench.exp.bases.benchTransforms:=[];

 Add(base_bench.callbacks,meth(self)
        local a, hash, r;
        if IsBound(libgen_bench.exp.bases.hashTable) then
            hash:=libgen_bench.exp.bases.hashTable;
            for a in Flat(hash.entries) do
        if ObjId(a.key)=FactorMMM and Length(a.data)>0 and IsBound(a.data[1].measured) and 
                    a.key.params{[1..3]}=a.key.params[4].params{[1..3]}*2 then
                        r:=HashLookup(self.exp.bases.hashTable,a.key.params[4]);
                        HashDelete(self.exp.bases.hashTable,a.key.params[4]);
                        r[1].mflopslibgen:=
                                    _compute_mflops(a.key.normalizedArithCost(),a.data[1].measured);
                        HashAdd(self.exp.bases.hashTable,a.key.params[4],r);
                fi;
            od;
        fi;
    end);
 base_bench.resumeAll();
 Add(libgen_bench.callbacks,meth(self) 
        libgen_bench.exp.bases.baseHashes := [ CreateRecursBaseHash(base_bench.exp.bases.hashTable) ];
    end);
 libgen_bench.resumeAll();
 libgen_bench.runExhaust([FactorMMM(2,4,2,KernelMMM(1,2,1,[ AVecReg(SSE_2x64f) ])),FactorMMM(6,8,6,KernelMMM(3,4,3,[ AVecReg(SSE_2x64f) ]))]);
 base_bench.resumeAll();

HashLookup(base_bench.exp.bases.hashTable,KernelMMM(3,4,3,[ AVecReg(SSE_2x64f) ]));

 obj:=[37,42,63];
 space:=CartesianProduct(List([7,7,7],x->[1..x]));;
# space:=CartesianProduct(List(obj,x->DivisorsInt(x)));;
 fspace:=Filtered(space,a->Mod(a[2],AVecReg(SSE_2x64f).v)=0);;
 g:=GaussianPredictor;
 k:=g.findMax(base_bench,libgen_bench,KernelMMM,FactorMMM, fspace,[AVecReg(SSE_2x64f)],ScaleMMMMFLOPS(0.50,obj));
 libgen_bench.runExhaust([PaddedMMM(obj[1],obj[2],obj[3],k)]);


----------------------------------------------------------------------
#double bench for Duped

 opts := SetMMMOptions(true,1,false);
 base_opts:=CopyFields(opts, rec(
                globalUnrolling := 2^16,
                hashFile        := let(p:=Conf("path_sep"),
                    Concat(Conf("spiral_dir"), p, "spiral", p, "libgen", p, "MMM-bases.hash")),
                benchTransforms := [KernelMMMDuped(1,2,1,[ AVecReg(SSE_2x64f) ]),KernelMMMDuped(3,4,3,[ AVecReg(SSE_2x64f) ])])); 
 base_bench:= CopyFields(DPBench(rec(bases := base_opts), rec(verbosity:=0)),
                       rec(generateCfiles := false, generateSums := true, matrixVerify := false));


 libgen_opts:=CopyFields(opts, rec(
                globalUnrolling := 10,
                hashFile        := let(p:=Conf("path_sep"),
                    Concat(Conf("spiral_dir"), p, "spiral", p, "libgen", p, "MMM-libgen.hash")),
                benchTransforms := [],
                formulaStrategies := Copy(opts.formulaStrategies),
                verbosity:=1,
                libgen:=rec(basesUnrolling := 2^16, baseBench := base_bench, 
                        codeletTab := CreateCodeletHashTable(), terminateStrategy := [ HfuncSumsRules ]))); 
 libgen_opts.formulaStrategies.postProcess:=DropLast(opts.formulaStrategies.postProcess,1);      #drop BBs
 Add(libgen_opts.formulaStrategies.postProcess,HfuncSumsRules); 
 Add(libgen_opts.formulaStrategies.postProcess,LibgenVectorHackRuleset);
 Add(libgen_opts.formulaStrategies.postProcess,OLCrossPullInRules);
 Add(libgen_opts.formulaStrategies.postProcess,OLVectorPropagateRuleset);
 Add(libgen_opts.formulaStrategies.postProcess,RecursStepTerm);
 Add(libgen_opts.formulaStrategies.postProcess,(s, opts) -> compiler.BlockSums(opts.globalUnrolling, s));
 libgen_bench:= CopyFields(DPBench(rec(bases := libgen_opts), rec(verbosity:=0)),
                       rec(generateCfiles := false, generateSums := true, matrixVerify := false));
 libgen_bench.exp.bases.benchTransforms:=[];

 Add(base_bench.callbacks,meth(self)
        local a, hash, r;
        if IsBound(libgen_bench.exp.bases.hashTable) then
            hash:=libgen_bench.exp.bases.hashTable;
            for a in Flat(hash.entries) do
        if ObjId(a.key)=FactorMMMDuped and Length(a.data)>0 and IsBound(a.data[1].measured) and 
                    a.key.params{[1..3]}=a.key.params[4].params{[1..3]}*2 then
                        r:=HashLookup(self.exp.bases.hashTable,a.key.params[4]);
                        HashDelete(self.exp.bases.hashTable,a.key.params[4]);
                        r[1].mflopslibgen:=
                                    _compute_mflops(a.key.normalizedArithCost(),a.data[1].measured);
                        HashAdd(self.exp.bases.hashTable,a.key.params[4],r);
                fi;
            od;
        fi;
    end);
 base_bench.resumeAll();
 Add(libgen_bench.callbacks,meth(self) 
        libgen_bench.exp.bases.baseHashes := [ CreateRecursBaseHash(base_bench.exp.bases.hashTable) ];
    end);
 libgen_bench.resumeAll();
 libgen_bench.runExhaust([FactorMMMDuped(2,4,2,KernelMMMDuped(1,2,1,[ AVecReg(SSE_2x64f) ])),FactorMMMDuped(6,8,6,KernelMMMDuped(3,4,3,[ AVecReg(SSE_2x64f) ]))]);
 base_bench.resumeAll();

HashLookup(base_bench.exp.bases.hashTable,KernelMMMDuped(3,4,3,[ AVecReg(SSE_2x64f) ]));

 obj:=[2,4,32];
 space:=CartesianProduct(List([32,32,32],x->[1..x]));;
# space:=CartesianProduct(List(obj,x->DivisorsInt(x)));;
 fspace:=Filtered(space,a->Mod(a[2],AVecReg(SSE_2x64f).v)=0);;
 g:=GaussianPredictor;
 k:=g.findMax(base_bench,libgen_bench,KernelMMMDuped,FactorMMMDuped, fspace,[AVecReg(SSE_2x64f)],ScaleMMMDupedMFLOPS(1.50,obj));
 libgen_bench.runExhaust([PaddedMMMDuped(obj[1],obj[2],obj[3],k)]);



-------------------------------------------

 opts := SetMMMOptions(true,1,false);
 base_opts:=CopyFields(opts, rec(
                globalUnrolling := 2^16,
                hashFile        := let(p:=Conf("path_sep"),
                    Concat(Conf("spiral_dir"), p, "spiral", p, "libgen", p, "MMM-bases.hash")),
                benchTransforms := [])); 
 base_bench:= CopyFields(DPBench(rec(bases := base_opts), rec(verbosity:=0)),
                       rec(generateCfiles := false, generateSums := true, matrixVerify := false));
 base_bench.resumeAll();

 libgen_opts:=CopyFields(opts, rec(
                globalUnrolling := 1,
                formulaStrategies := Copy(opts.formulaStrategies),
                libgen:=rec(basesUnrolling := 2^16, baseBench := base_bench, 
                        codeletTab := CreateCodeletHashTable(), terminateStrategy := [ HfuncSumsRules ]))); 
 libgen_opts.formulaStrategies.postProcess:=DropLast(opts.formulaStrategies.postProcess,1);      #drop BBs
 Add(libgen_opts.formulaStrategies.postProcess,HfuncSumsRules); 
 Add(libgen_opts.formulaStrategies.postProcess,LibgenVectorHackRuleset);
 Add(libgen_opts.formulaStrategies.postProcess,OLCrossPullInRules);
 Add(libgen_opts.formulaStrategies.postProcess,OLVectorPropagateRuleset);
 Add(libgen_opts.formulaStrategies.postProcess,RecursStepTerm);
 Add(libgen_opts.formulaStrategies.postProcess,(s, opts) -> compiler.BlockSums(opts.globalUnrolling, s));

nt:=MMM_Nt(7,9,10,[AVecReg(SSE_2x64f)]);
GP(nt,libgen_opts, base_bench.exp.bases);

gerer les predictions hyperplates
areter de chercher pour Factor mais utiliser DP
utiliser dpbench
faire un prepredict
jeter les predictions selectivement
le unroll dans le main est moyennement gere...
unrolling ROIs like crazy is prolly a stupid idea



-------------------------------------------

 opts := SetMMMOptions(true,1,false);
 opts.profile.makeopts.CFLAGS:=Concat(opts.profile.makeopts.CFLAGS," -fno-alias -fno-fnalias -save-temps -fno-inline-functions");
 base_opts:=CopyFields(opts, rec(
                globalUnrolling := 2^16,
                hashFile        := let(p:=Conf("path_sep"),
                    Concat(Conf("spiral_dir"), p, "spiral", p, "libgen", p, "MMM-bases.hash")),
                benchTransforms := [])); 
 base_bench:= CopyFields(DPBench(rec(bases := base_opts), rec(verbosity:=0)),
                       rec(generateCfiles := false, generateSums := true, matrixVerify := false));
 base_bench.resumeAll();

 libgen_opts:=CopyFields(opts, rec(
                globalUnrolling := 1,
                formulaStrategies := Copy(opts.formulaStrategies),
                libgen:=rec(basesUnrolling := 2^16, baseBench := base_bench, 
                        codeletTab := CreateCodeletHashTable(), terminateStrategy := [ HfuncSumsRules ]))); 
 libgen_opts.formulaStrategies.postProcess:=DropLast(opts.formulaStrategies.postProcess,1);      #drop BBs
 Add(libgen_opts.formulaStrategies.postProcess,HfuncSumsRules); 
 Add(libgen_opts.formulaStrategies.postProcess,LibgenVectorHackRuleset);
 Add(libgen_opts.formulaStrategies.postProcess,OLCrossPullInRules);
 Add(libgen_opts.formulaStrategies.postProcess,OLVectorPropagateRuleset);
 Add(libgen_opts.formulaStrategies.postProcess,RecursStepTerm);
 Add(libgen_opts.formulaStrategies.postProcess,(s, opts) -> compiler.BlockSums(opts.globalUnrolling, s));

nt:=MMM_Nt(7,9,11,[AVecReg(SSE_2x64f)]); #->1437
GP(nt,libgen_opts, base_bench.exp.bases);

utiliser dpbench
le unroll dans le main est moyennement gere...
unrolling ROIs like crazy is prolly a stupid idea
Pour Factor: faire que factor enleve x% au min et ajoute x% au max empecher l'algorithme de terminer.
Lors de la realisation, si toutes les feuilles sont terminees, lancer un dp au niveau d'au dessus.


-------------------------------------------

 opts := SetMMMOptions(true,1,false);
 opts.profile.makeopts.CFLAGS:=Concat(opts.profile.makeopts.CFLAGS," -fno-alias -fno-fnalias -save-temps -fno-inline-functions");
 base_opts:=CopyFields(opts, rec(
                globalUnrolling := 2^16,
                verbosity:=-1,
                hashFile        := let(p:=Conf("path_sep"),
                    Concat(Conf("spiral_dir"), p, "spiral", p, "libgen", p, "MMM-bases.hash")),
                benchTransforms := [])); 
 base_bench:= CopyFields(DPBench(rec(bases := base_opts), rec(verbosity:=0)),
                       rec(generateCfiles := false, generateSums := true, matrixVerify := false));
 base_bench.resumeAll();


 libgen_opts:=CopyFields(opts, rec(
                globalUnrolling := 1,
                hashFile        := let(p:=Conf("path_sep"),
                    Concat(Conf("spiral_dir"), p, "spiral", p, "libgen", p, "MMM-libgen.hash")),
                benchTransforms := [],
                formulaStrategies := Copy(opts.formulaStrategies),
                verbosity:=-1,
                libgen:=rec(basesUnrolling := 2^16, baseBench := base_bench, 
                        codeletTab := CreateCodeletHashTable(), terminateStrategy := [ HfuncSumsRules ]))); 
 libgen_opts.formulaStrategies.postProcess:=DropLast(opts.formulaStrategies.postProcess,1);      #drop BBs
 Add(libgen_opts.formulaStrategies.postProcess,HfuncSumsRules); 
 Add(libgen_opts.formulaStrategies.postProcess,LibgenVectorHackRuleset);
 Add(libgen_opts.formulaStrategies.postProcess,OLCrossPullInRules);
 Add(libgen_opts.formulaStrategies.postProcess,OLVectorPropagateRuleset);
 Add(libgen_opts.formulaStrategies.postProcess,RecursStepTerm);
 Add(libgen_opts.formulaStrategies.postProcess,(s, opts) -> compiler.BlockSums(opts.globalUnrolling, s));
 libgen_bench:= CopyFields(DPBench(rec(bases := libgen_opts), rec(verbosity:=0)),
                       rec(generateCfiles := false, generateSums := true, matrixVerify := false));
libgen_bench.resumeAll();

nt:=MMM_Nt(6,5,5,[AVecReg(SSE_2x64f)]); 
GP(nt,libgen_bench, base_bench);

-----------------
le unroll dans le main est moyennement gere...
cannot go to big sizes unless ROIs are prevented from unrolling
choisir des tailles de kernels qui font du sens...
