
# Copyright (c) 2018-2021, Carnegie Mellon University
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
              OLVectorPropagateRuleset,
              OLSingleComposeRuleset,
              (s, opts) -> compiler.BlockSumsOpts(s, opts)
          ];

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

              OLVectorPropagateRuleset,
              
              OLSingleComposeRuleset,
	      (s, opts) -> compiler.BlockSumsOpts(s, opts)];
  
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

