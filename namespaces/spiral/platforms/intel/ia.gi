
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

ImportAll(paradigms.smp);
ImportAll(paradigms.vector);

Class(IAGlobals, rec(
    getOpts := meth(arg)
        local opts, self, smpopts, isa, sseopts, optrec, tid;
        
        #   set up defaults
        optrec := rec(
            cpu := LocalConfig.cpuinfo,
            useSIMD := true,
            useSMP := true,
            useNewSettingsForICC := false,
            dataType := T_Real(64),
            globalUnrolling := 128,
            useArea := false,
            processIntTables := false,
            use64bit := LocalConfig.osinfo.is64bit()
        );

        smpopts := rec(
                numproc := LocalConfig.cpuinfo.cores,
                api := "OpenMP"
        );
        
        sseopts := rec(
            svct:=true, 
            splitL:=false, 
            oddSizes := false,
            stdTTensor := true, 
            tsplPFA := false
        );

        self := arg[1];
        arg := Flat(Drop(arg, 1));
        
        if IsList(arg[1]) then arg := arg[1]; fi;
        if Length(arg) >= 1 then optrec := CopyFields(optrec, arg[1]); fi;
        if Length(arg) >= 2 then smpopts := CopyFields(smpopts, arg[2]); fi;
        if Length(arg) >= 3 then sseopts := CopyFields(sseopts, arg[3]); fi;
        
        if optrec.useSIMD then 
            #   handle SSE
            #   let SSE work the unrolling magic
            sseopts.globalUnrolling := optrec.globalUnrolling;
            sseopts.useArea := optrec.useArea;
            sseopts.processIntTables := optrec.processIntTables;
            if IsBound(optrec.mode) then sseopts.mode := optrec.mode; fi;
            
            isa := optrec.cpu.getSimdIsa(optrec.dataType);
            opts := CopyFields(SIMDGlobals.getOpts(isa, sseopts), rec(
                IAconf := rec(
                    optrec := optrec,
                    smpopts := smpopts,
                    sseopts := sseopts
                ),
                
                unparser := When(smpopts.api = "OpenMP", 
                    When(isa in [AVX_8x32f, AVX_4x64f], 
                        When(IsBound(smpopts.OmpMode) and smpopts.OmpMode = "for", 
                            spiral.libgen.OpenMP_AVXUnparser_ParFor, 
                            spiral.libgen.OpenMP_AVXUnparser),
                        When(IsBound(smpopts.OmpMode) and smpopts.OmpMode = "for", 
                            spiral.libgen.OpenMP_SSEUnparser_ParFor, 
                            spiral.libgen.OpenMP_SSEUnparser)),
                    spiral.libgen.SMP_SSEUnparser),
                codegen := spiral.libgen.VecRecCodegen
            ));
            
            if optrec.useNewSettingsForICC then
                if IsBound(opts.language) then Unbind(opts.language); fi;
                if LocalConfig.osinfo.isLinux() then 
                    if optrec.use64bit then
                        opts.profile := default_profiles.linux_x64_icc;
                    else
                        opts.profile := default_profiles.linux_x86_icc;
                    fi;
                elif LocalConfig.osinfo.isDarwin() then 
                    Error("No Darwin SMP profiles defined yet");
                elif LocalConfig.osinfo.isWindows() then
                    if optrec.use64bit then
                        opts.profile := default_profiles.win_x64_icc;
                    else
                        opts.profile :=default_profiles.win_x86_icc;
                    fi;
                else
                    Error("Unknow OS");
                fi;
            fi;

        else
            if optrec.useSMP then Error("scalar SMP not implemented"); fi;
            
            opts := Copy(SpiralDefaults);
            
            opts.globalUnrolling := optrec.globalUnrolling;
            opts.useArea := optrec.useArea;
            opts.processIntTables := optrec.processIntTables;
            
            if optrec.dataType = T_Real(32) then 
                opts := InitDataType(opts, "f32re");
            elif optrec.dataType = T_Real(64) then 
                opts := InitDataType(opts, "f64re");
            else
                Error("unknown data type");    
            fi;
            
            if optrec.useNewSettingsForICC then
                if IsBound(opts.language) then Unbind(opts.language); fi;
                if LocalConfig.osinfo.isLinux() then 
                    if optrec.use64bit then
                        opts.profile := default_profiles.linux_x64_icc;
                    else
                        opts.profile := default_profiles.linux_x86_icc;
                    fi;
                elif LocalConfig.osinfo.isDarwin() then 
                    Error("No Darwin SMP profiles defined yet");
                elif LocalConfig.osinfo.isWindows() then
                    if optrec.use64bit then
                        opts.profile := default_profiles.win_x64_icc;
                    else
                        opts.profile :=default_profiles.win_x86_icc;
                    fi;
                else
                    Error("Unknow OS");
                fi;
            fi;
            
            if optrec.useArea then
                opts.globalUnrolling := 2 * opts.globalUnrolling; # * Log2Int(opts.globalUnrolling);
                opts.markBlock := MarkBlocksAreaSums;
            fi;
            
            return opts;            
        fi;
            
        # handle the SMP pthreads/OpenMP case
        if optrec.useSMP then
            opts.breakdownRules.GT := [ GT_Base, GT_NthLoop, 
                CopyFields(GT_Par, rec(parEntireLoop := false, splitLoop := true)), GT_Par_odd,
                GT_Vec_AxI, GT_Vec_IxA, GT_Vec_IxA_Push, GT_Vec_IxA_L, GT_Vec_L_IxA        
            ];
            opts.breakdownRules.TTensorI := Concat([ 
            CopyFields(TTensorI_toGT, rec(
                applicable := (self, t) >> t.hasTags() and ObjId(t.getTags()[1])=AParSMP ))], 
                opts.breakdownRules.TTensorI);
                
            opts.breakdownRules.TTensorInd := 
                Concat([dsA_base_smp, dsA_smp, L_dsA_L_base_smp, L_dsA_L_smp], 
                    opts.breakdownRules.TTensorInd);    
                
            tid := When(smpopts.api = "OpenMP", threadId(), var("tid", TInt));
            opts.tags := Concat([ AParSMP(smpopts.numproc, tid)  ], opts.tags);
            
            if optrec.useNewSettingsForICC then
                if IsBound(opts.language) then Unbind(opts.language); fi;
                if LocalConfig.osinfo.isLinux() then 
                    if optrec.use64bit then
                        opts.profile := When(smpopts.api = "OpenMP", default_profiles.linux_x64_icc_openmp, default_profiles.linux_x64_threads);
                    else
                        opts.profile := When(smpopts.api = "OpenMP", default_profiles.linux_x86_icc_openmp, default_profiles.linux_x86_threads);
                    fi;
                elif LocalConfig.osinfo.isDarwin() then 
                    Error("No Darwin SMP profiles defined yet");
                elif LocalConfig.osinfo.isWindows() then
                    if optrec.use64bit then
                        opts.profile := When(smpopts.api = "OpenMP", default_profiles.win_x64_icc_openmp, default_profiles.win_x64_icc_threads);
                    else
                        opts.profile := When(smpopts.api = "OpenMP", default_profiles.win_x86_icc_openmp, default_profiles.win_x86_icc_threads);
                    fi;
                else
                    Error("Unknow OS");
                fi;
            else
                opts.language := When(smpopts.api = "OpenMP", optrec.cpu.OpenMP_lang, optrec.cpu.smp_lang);
            fi;
    
            if smpopts.api = "threads" then opts.subParams := [var("num_threads", TInt), var("tid", TInt)]; fi;
            opts.smp := smpopts;
        fi;

        Add(opts.includes, "<include/mm_malloc.h>");
        if not IsBound(opts.globalUnrolling) then opts.globalUnrolling := optrec.globalUnrolling; fi;
        opts.operations := rec(Print := (s) -> Print("<IA options>"));
        return opts;
    end
));
