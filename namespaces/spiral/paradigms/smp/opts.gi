
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(SMPGlobals, rec(
    pthreads := self >> "pthreads",
    OSThreads := self >> Cond(LocalConfig.osinfo.isWindows(), "winthreads", "LinuxThreads"),
    threads := self >> "threads",
    OpenMP := self >> "OpenMP",
    maxThreads := self >> LocalConfig.cpuinfo.cores,
    getOpts := meth(arg)
        local self, opts, optrec, tid;

        self := arg[1];
        opts := CopyFields(SpiralDefaults);
        optrec := rec(api := self.OpenMP(), numproc := self.maxThreads(), parOdd := false);
        if Length(arg) >= 2 then optrec := CopyFields(optrec, arg[2]); fi;

        opts.breakdownRules.GT := Concat([ GT_Base, GT_NthLoop, CopyFields(GT_Par, rec(parEntireLoop := false, splitLoop := true))],
            When(optrec.parOdd, [GT_Par_odd ], []));
        opts.breakdownRules.TTensorI := [ CopyFields(TTensorI_toGT, rec(applicable := (self, t) >> t.hasTags() and ObjId(t.getTags()[1])=AParSMP ))];
        opts.breakdownRules.TTensorInd := [ dsA_base_smp, dsA_smp, L_dsA_L_base_smp, L_dsA_L_smp ];

        tid := When(optrec.api = "OpenMP", threadId(), CopyFields(var("tid", TInt), rec(isParallelLoopIndex := true)));
        opts.tags := [ AParSMP(optrec.numproc, tid) ];

#        if optrec.api = "threads" then opts.subParams := [var("num_threads", TInt), var("tid", TInt)]; fi;
        opts.smp := optrec;

        return opts;
    end
));
