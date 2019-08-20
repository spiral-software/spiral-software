
# Copyright (c) 2018-2019, Carnegie Mellon University
# See LICENSE for details


SetLinuxX86Profile:=function(opts, p, use_openmp)
   opts.profile:=Copy(default_profiles.linux_x86_icc);
   SetMakeOptsAffinity(opts);
   SetMakeOptsLibgen(opts);
   SetMakeOptsAssembly(opts);
   opts.includes := Concat(opts.includes, [ "<string.h>"]);
 
   if IsBound(opts.vector) then
       SetMakeOptsSSE(opts);
   fi;

   if p>1 then
       if (use_openmp) then
           SetMakeOptsOpenMP(opts);
       else
           SetMakeOptsPThreads(opts);
       fi;
   fi;
end;

SetOLOptions:=function(isa, p, use_openmp)
   local opts, MixedUnparser;

   if (isa = false) or (IsString(isa)) then
       opts := Copy(SpiralDefaults);
       if (IsString(isa)) then opts := InitDataType(opts,isa); fi;
   else
       if (isa = true) then isa:=SIMD_ISA_DB.active()[1]; fi;
       opts := SIMDGlobals.getOpts(isa);
   fi;

   if not IsBound(opts.profile) then
       SetLinuxX86Profile(opts, p, use_openmp);
   else
#Hack Cell profile
       opts.profile := default_profiles.linux_cellSPU_gcc_MMM;
       opts.profile.stubopts.ROWS := 1;
       opts.profile.stubopts.COLUMNS := 1;
       opts.includes := Concat(opts.includes, [ "<mm_malloc.h>", "<string.h>"]);
       opts.includes := Filtered(opts.includes, x->x<>"<omega32.h>");
   fi;

   if p>1 then
       if (use_openmp) then
           opts.unparser:=Class(MixedUnparser,OpenMP_UnparseMixin,opts.unparser);
       else
           opts.unparser:=Class(MixedUnparser,SMP_UnparseMixin,opts.unparser);
           opts.subParams := [var("num_threads", TInt), var("tid", TInt)];
       fi;
   fi;

   #OL specific
   opts.codegen := OLCodegen;

   opts.formulaStrategies.sigmaSpl := OLDefaultStrategy;

   opts.formulaStrategies.postProcess := [
              OLVectorPropagateRuleset,
              OLPushScatQuestionMarkInRuleset,
              OLAlreadyInitializedScatQuestionMarkRuleset, 
              OLCrossPullInRuleset,
              OLAfterCrossPullInRuleset,
#              OLDropVGath_dupRuleset,  
#              OLTearCrossRuleset,
#              OLMagicVUnrollDupRuleset,
              OLVectorPropagateRuleset,
#              OLRulesBufferFinalize,
              OLSingleComposeRuleset,
              (s, opts) -> compiler.BlockSumsOpts(s, opts)
          ];
#              OLSplitScatQuestionMarkRuleset,OLScatProbeMergeRuleset 
#              OLScatAccPeelRules
#              OLSingleComposeRuleset

   #Unrolling
   opts.markBlock := MarkBlocksOps;
   opts.globalUnrolling := 300;

   #compile options
   opts.compileStrategy :=IndicesCS2;
   opts.useDeref := true;
   opts.propagateNth := false;
   opts.doScalarReplacement := true;

   #Libgen requires full spec of perms
   VPerm.print:=VPerm.printl;

   #ScatAcc requires full zero allocation for now
   opts.zeroallocate := true;

   #final code options
   opts.subName:="multi";
   opts.subInitName:="init_multi";

   return opts;
end;

SetSAROptions:=function(isa, p, use_openmp)
  local tags,opts;
  opts := SetOLOptions(isa, p, use_openmp); 
  opts.codegen := OLCodegen;
  opts.zeroallocate := false;
  opts.InputTypes := [TDouble,TDouble,TDouble,TDouble];
  opts.OutputTypes := [TDouble,TDouble];
  opts.generateComplexCode := true;
  opts.TRealCType := "complex";
  opts.TRealCtype := "complex";
  opts := InitDataType(opts,"f64c");
  opts.formulaStrategies.sigmaSpl := OLDefaultStrategy;
  opts.breakdownRules.TTensorI_OL := [TTensorI_OL_Base, TTensorI_OL_Parrallelize_AParFirst, TTensorI_OL_Vectorize_AVecLast];
  opts.breakdownRules.SAR := [SAR_Base,SAR_MatchFilter,SAR_MatchFilterSMP,SAR_MatchFilterVec,SAR_MatchFilterVecBase];
  opts.breakdownRules.ExpandMult := [ExpandMult_Base, ExpandMult_One];
  opts.breakdownRules.SAR_Interpolation := [SAR_Interpolation_One];
  tags:=[];

  if (p>1) then
      Add(tags,AParSMP(p));
  fi;
  opts.formulaStrategies.postProcess := [
        OLVectorPropagateRuleset,
 	      OLScatQuestionMarkToScat,
              OLPushScatAccRuleset,
              OLCrossPullInRuleset,
              OLAfterCrossPullInRuleset,
#              OLDropVGath_dupRuleset,  
#              OLTearCrossRuleset,
#              OLMagicVUnrollDupRuleset,
              OLVectorPropagateRuleset,
#              OLRulesBufferFinalize,
              
              OLSingleComposeRuleset,
	      (s, opts) -> compiler.BlockSumsOpts(s, opts)];
	#	];
  
      # (s, opts) -> compiler.BlockSumsOpts(s, opts)];

  #Unrolling
  opts.markBlock := MarkBlocksOps;
  opts.globalUnrolling := 300;

  #compile options
  opts.compileStrategy :=IndicesCS2;
  opts.useDeref := true;
  opts.doNotScalarize := false;
  opts.propagateNth := false;
  opts.doScalarReplacement := true;

  #ScatAcc requires full zero allocation for now
  opts.zeroallocate := false;

  #final code options
  opts.subName:="multi";
  opts.subInitName:="init_multi";

 
  return opts;


end;

SetMMMOptions:=function(isa, p, use_openmp)
   local opts;
   opts:=SetOLOptions(isa, p, use_openmp);
   opts.InputTypes := [TReal,TReal];
   opts.OutputTypes := [TReal];

   #dont allow for genuine vectorization unless you add MMM_VerticalIndependance to it
#   opts.breakdownRules.MMM := [MMM_BaseMult, MMM_KernelReached, MMM_TileHorizontally, MMM_TileVerticaly, MMM_TileDepth, MMM_Padding];
   opts.breakdownRules.TTensorI_OL := [TTensorI_OL_Base, TTensorI_OL_Parrallelize_AParFirst, TTensorI_OL_Vectorize_AVecLast];
#   opts.breakdownRules.KernelMMM :=[ KernelMMM_Base, KernelMMM_TileH, KernelMMM_TileV, KernelMMM_IndV, KernelMMM_TileD];
#   opts.breakdownRules.KernelMMMVec :=[ KernelMMMVec_Base, KernelMMMVec_TileH, KernelMMMVec_TileV, KernelMMMVec_TileD];
#   opts.breakdownRules.KernelMMMDuped :=[ KernelMMMDuped_Base, KernelMMMDuped_TileH, KernelMMMDuped_TileV, KernelMMMDuped_IndV, KernelMMMDuped_TileD];
#   opts.breakdownRules.KernelMVMVec := [ KernelMVMVec_Base_VAdd, KernelMVMVec_TileH, KernelMVMVec_TileD];
#   opts.breakdownRules.FactorMMM :=[ FactorMMM_Base, FactorMMM_TileH, FactorMMM_TileV, FactorMMM_TileD];
#   opts.breakdownRules.FactorMMMDuped :=[ FactorMMMDuped_Base, FactorMMMDuped_TileH, FactorMMMDuped_TileV, FactorMMMDuped_TileD];
#   opts.breakdownRules.PaddedMMM :=[ MMM_Padding ];
#   opts.breakdownRules.PaddedMMMDuped :=[ MMMDuped_Padding ];
#   opts.breakdownRules.ROIGathScalar :=[ROIGathScalar_Base];
#   opts.breakdownRules.ROIScatScalar :=[ROIScatScalar_Base];
#   opts.breakdownRules.ROIGathFullVector :=[ROIGathFullVector_Base];
#   opts.breakdownRules.ROIScatFullVector :=[ROIScatFullVector_Base];
#   opts.breakdownRules.ROIDupScatScalar := [ROIDupScatScalar_Base];
#   opts.breakdownRules.ROIDupScatFullVector := [ROIDupScatFullVector_Base,ROIDupScatFullVector_Par];
   opts.breakdownRules.TL := [spiral.paradigms.vector.bases.SIMD_ISA_Bases1, spiral.paradigms.vector.bases.SIMD_ISA_Bases2, IxLxI_kmn_n];
#   opts.breakdownRules.HAdd := [HAdd_with_VHAdd, HAdd_with_ParVHAdd];
### opts.breakdownRules.MMM_Nt := [Padding_DupIn_Scalar, Padding_DupOut_Vector, Padding_DupIn_Vector, Padding_MVM_Vector];
#  opts.breakdownRules.MMM_Nt := [Padding_DupIn_Vector];
#  opts.breakdownRules.FactorMVMDuped := [FactorMVMDuped_Base];


   opts.profile.makeopts.VERIFIER := "../common/verify_MMM.c";
   opts.profile.makeopts.VERIFIER_OPTS := "";

   if (p>1) and (not use_openmp) then
           opts.profile.makeopts.VERIFIER := "../common/verify_MMM_threads.c";
   fi;

   opts.profile.verify:=(a,b,c) -> _VerifyMMM(a,b,c, "verify");

   if (not IsBound(opts.profile.stubopts)) then opts.profile.stubopts:=rec(); fi;
   #opts.profile.stubopts.ZEROOUTPUT:=true;
   #opts.profile.stubopts.DEBUG:=true;
   
   return opts;
end;

<# MRT
doParSimdMMM:=function(m,n,k,isa,p,use_openmp)
   local opts, tags, nt;

   opts:=SetMMMOptions(isa, p, use_openmp);
   opts.baseHashes:=[MMM_DB.getHash()];

   tags:=[];

   if (p>1) then
       Add(tags,AParSMP(p));
   fi;
   
   #NOTE Horrible hack
   if (isa<>false) then
       Add(tags,AVecReg(opts.vector.isa));
       Add(tags,WithKernel(MMM(4,4,32)));        
       opts.globalUnrolling:=5*4*4*32;
   else
       Add(tags,WithKernel(MMM(3,3,10)));
       opts.globalUnrolling:=5*3*3*10;
   fi;
   
   nt:=MMM(m,n,k,tags);
   return [opts,nt,c->_compute_mflops(nt.normalizedArithCost(),CMeasure(c,opts)),
       c->CVerifyMMM(c,opts,nt)];
end;
#>
