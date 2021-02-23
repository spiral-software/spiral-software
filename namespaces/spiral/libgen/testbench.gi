
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(TestBench);

#F TestBench("name", <transforms> <opts>, <searchopts>)
#F
#F d := TestBench("default", [DFT(2), DFT(4)], SpiralDefaults, rec())
#F
#F TestBench Interface:
#F
#F     .build(transforms, opts, bopts, name, searchopts),
#F
#F     .generateCfiles := [true, false],  default = true
#F     .matrixVerify   := [true, false],  default = false
#F     .fftwVerify     := [true, false],  default = false
#F
#F     .fileTransform(t, opts),    .c filenames:   override in a subclass
#F     .funcTransform(t, opts),    function names: override in a subclass
#F     .txtFileName(runMethod),    timing file name
#F
#F     .run()              run DP
#F     .runExhaustive()    run exhaustive search
#F     .runRandom()        run 1 random tree, do not generate hash
#F     .runRandomSave()    run 1 random tree, generate and save hash
#F     .runRandom10()      run 10 random trees, do not generate hash
#F     .runRandomSave10()  run 10 random trees, generate and save hash
#F     .pickRandom()       generate random trees and save hash, do not run
#F
#F     .generateCode()
#F     .generateProductionCode()   same as .generateCode() but will use opts.production()
#F
#F     .entries()
#F     .times()
#F     .mflops()
#F
Class(TestBench, rec(
    ##
    ## Configuration
    ##
    generateCfiles    := true,
    matrixVerify      := false,
    fftwVerify        := false,
    outputExhaustive  := false,
    outputDir         := ".",
    fileTransform     := (self, t, opts) >> self.outputDir :: Conf("path_sep") :: self.name :: "_" 
                                                   :: Drop(CodeletName(CodeletShape(t)), 1) :: ".c",
    funcTransform     := (self, t, opts) >> "sub1",
    prodFileTransform := (self, t, opts) >> self.fileTransform(t, opts), # used in generateProductionCode
    prodFuncTransform := (self, t, opts) >> self.funcTransform(t, opts),
    txtFileName       := (self, runMethod) >> self.outputDir :: Conf("path_sep") :: self.name :: "." 
                                                  :: SubString(runMethod, 5) :: ".txt",
    ##
    ## Public methods
    ##

    # TestBench(<name>, <transforms>, <opts>, <search_opts>)
    __call__ := meth(self, name, transforms, opts, searchopts)
        local o;
	o := CopyFields(opts); 
	if not IsBound(o.hashTable) then o.hashTable := HashTableDP(); fi;
	if not IsBound(o.hashFile)  then o.hashFile := Concat(self.outputDir, Conf("path_sep"), name, ".hash"); fi;
        return WithBases(self,
            rec(searchopts:=searchopts, opts:=o, transforms:=transforms, name:=name, verbosity:=1, callbacks:=[]));
    end,

    # TestBench.build(<transforms>, [<opts>], [<bench_opts>], [<bench_name>], [<search_opts>])
    #    Alternative constructor that supports default argument values if not provided
    #   
    build := function(arg)
        local transforms, opts, bopts, name, dpr;

        transforms := When(IsList(arg[1]), arg[1], [arg[1]]);
        opts := When(Length(arg) >= 2, arg[2], SpiralDefaults);
        bopts := When(Length(arg) >= 3, arg[3], rec());
        name := When(Length(arg) >= 4, arg[4], "spiral");
        dpr := When(Length(arg) >= 5, arg[5], rec(verbosity := 0, timeBaseCases:=true));

        return CopyFields(TestBench(name, transforms, opts, dpr), bopts);
    end,

    generateCode           := self >> self._generateCode(self.transforms, self.opts),

    generateProductionCode := self >> self._generateCode(self.transforms, self.opts.production()),

    entries := self >> List(self.transforms, t -> self.entry(t)), 

    entry := (self, t) >> let(l := self._rawentry(t), 
	When(l=false, false, CopyFields(l, rec(ruletree := ApplyRuleTreeSPL(l.ruletree, t, self.opts))))),

    times   := self >> List(self.entries(), e -> e.measured),

    mflops  := self >> List(self.entries(), e -> self.acost(e) * LocalConfig.cpuinfo.freq / e.measured),

    # Use NonTerminal.normalizedArithCost() if it is there, otherwise return 0
    acost := (self, entry) >> self._rtflops(entry.ruletree),

    run            := arg >> arg[1]._run([],Drop(arg, 1), "_runDP",         true),
    runDP          := ~.run,
    runExhaustive  := arg >> arg[1]._run([], Drop(arg, 1), "_runExhaustive", true),
    runRandom      := arg >> arg[1]._run([arg[2]], Drop(arg, 2), "_runRandom",     false),
    runRandomSave  := arg >> arg[1]._run([arg[2]], Drop(arg, 2), "_runRandomSave", true),
    pickRandomSave := arg >> arg[1]._run([], Drop(arg, 1), "_pickRandomSave",true), 

    ##
    ## Private methods
    ##

    _rtflops := rtree -> let(t:=rtree.node, When(IsBound(t.normalizedArithCost), EvalScalar(t.normalizedArithCost()), 0)),

    _verify := (self, opts, ruletree) >> VerifyMatrixRuleTree(ruletree, opts),

    # NOTE: Slightly hacked in. Check for opts.profile being bound etc. Look at VerifyMatrixRuleTree.
    _verifyfftw := (self, opts, code) >> opts.profile.verifyfftw(code, opts),

    _startHashFile := (self, hfile, d) >> PrintTo(hfile,
        "<# DPBench experiment '", self.name, "'\n",
        " # Started ", d[2], " ", d[3], " ", d[1], "  ", d[4], ":", d[5], "\n",
        " # Transforms: ", self.transforms, "#> \n\n",
        "ImportAll(spiral); Import(paradigms.common, paradigms.smp, platforms.sse, paradigms.vector); \n",
        "ImportAll(paradigms.vector); \n\n",
        "hash := HashTableDP(); \n"
    ),

    _loadHash := meth(self, hfile)
        local ns, result;
        ns := tab();
        result := READ(hfile, ns);
        if result = false or not IsBound(ns.hash) then return false;
        else return ns.hash;
        fi;
    end,

    _saveHash := meth(self, hfile, date, hash)
        local bucket, e;
        var.print := var.printFull;
        self._startHashFile(hfile, date);
        for bucket in hash.entries do
            for e in bucket do
                if e.data<>[] then
                    AppendTo(hfile, "HashAdd(hash, ", e.key, ", [", e.data[1], "]);\n");
                fi;
            od;
        od;
        var.print := var.printShort;
    end,

    reloadHash := meth(self)
       local hash, e;
       hash := self._loadHash(self.opts.hashFile);
       if (self.verbosity>0) then
	   PrintLine(When(hash=false, "Could not load ", "Loaded "), self.name, " (", self.opts.hashFile, ")");
       fi;
       if hash <> false then
	   self.opts.hashTable := hash;
       fi;
    end,

    _generateCode := meth(self, transforms, opts)
         local entries, e, c, t;
         for t in transforms do
             e := self.entry(t); 
             if e = false then Error("Transform ", t, " not found in hashTable"); fi;
             c := CodeRuleTree(e.ruletree, opts);
             PrintLine(t, " -> ", self.prodFileTransform(t, opts));
             PrintTo(self.prodFileTransform(t, opts), PrintCode(self.prodFuncTransform(t, opts), c, opts));
         od;
    end,

    _showStats := meth(self, runMethod, t, ruletree, c, cycles, searchTime)
        local acc;
        # NOTE: slightly hacked in. Clean up to get both matrix and fftw verification to use already generated c.
	if self.matrixVerify or self.fftwVerify then
	    if self.matrixVerify then acc := self._verify(self.opts, ruletree);
	    else                      acc := self._verifyfftw(self.opts, c); fi;
	    _seqPerfStatsGflopsAcc(self.txtFileName(runMethod), t, self._rtflops(ruletree), cycles, searchTime, acc);
	else
	    When(self.opts.verbosity>-1, 
		 _seqPerfStatsGflops(self.txtFileName(runMethod), t, LocalConfig.cpuinfo.freq, self._rtflops(ruletree), cycles, searchTime));
	fi;
    end,

    _rawentry := (self, t) >> let(
	lookup := MultiHashLookup(Concatenation([self.opts.hashTable], self.opts.baseHashes), HashAsSPL(t)),
	When(lookup=false or lookup=[], false, lookup[1])),

    #
    # Run methods
    #

    _runDP         := (self, t, opts) >> TimedAction(DP(t, self.searchopts, opts)),

 
    # Find best using an exhaustive search
    _runExhaustive := meth(self, t, opts)
       local r, searchTime, mincycles, mintree, rt, c, compiletime, cm, measuretime;

       r := AllRuleTrees(t, opts);
       searchTime := 0;
       mincycles := 10^100;

       for rt in r do
          [c,  compiletime] := TimedAction(CodeRuleTreeOpts(rt, opts));
          [cm, measuretime] := TimedAction(CMeasure(c, opts));
          if self.outputExhaustive then _seqPerfStatsGflops(
		  self.txtFileName("_runExhaustive-all"), t, LocalConfig.cpuinfo.freq, self._rtflops(rt), cm, compiletime+measuretime); fi;
          if (cm < mincycles) then
              mincycles := cm; mintree := Copy(rt);
          fi;
          searchTime:=searchTime+compiletime+measuretime;
       od;
       HashDelete(opts.hashTable,t);
       HashAdd(opts.hashTable, t, [rec(ruletree:=mintree, measured:=mincycles)]);

       return([mintree, searchTime]);
    end,

    # Run a <num> random ruletrees. Useful for quick, dirty, non-comprehensive tests.
    _runRandomNum := meth(self, num, t, opts)
       local r, c, start, cycles, i, res;
       start := TimeInSecs();
       res := [];
       for i in [1..num] do
           r := RandomRuleTree(t, opts); 
           c := CodeRuleTree(r, opts); 
           cycles := CMeasure(c, opts); 
           Add(res, [r, cycles]);
       od;
       Sort(res, (a, b) -> a[2] < b[2]);
       return [res[1][1], res[1][2], TimeInSecs()-start];
     end,

    # Run random search and save result in hash.
    _runRandomSave := meth(self, num, t, opts)
       local r, srchTime, cycles;
       [r, cycles, srchTime] := self._runRandomNum(num, t, opts);
       HashDelete(opts.hashTable, t);
       HashAdd(opts.hashTable, t, [rec(ruletree:=r, measured:=cycles)]);
       return [r, srchTime];
    end,

    _runRandom := (self, num, t, opts) >> self._runRandomNum(num, t, opts){[1,3]},

    # Pick random and save result in hash (NOT measured).
    _pickRandomSave := meth(self, e, t, opts)
       local r, searchtime;
       [r, searchtime] := TimedAction(RandomRuleTree(t, opts));
       HashDelete(opts.hashTable,t);
       HashAdd(opts.hashTable,t,[rec(ruletree:=r)]);
       return [r, searchtime];
    end,

    _resume := self >> When(ForAny(self.entries(), e->e=false), self.reloadHash()),

    _run := meth(self, runArgs, transforms, runMethod, useHash)
        local t, outf, res, nopts, opts, c, cycles, hentry,  date, i, searchTime, acc, f, ruletree;
        MakeDir(self.outputDir);
        transforms := Flat(transforms);
        if transforms=[] then transforms := self.transforms; fi; # equivalent of runAll() in DPBench
        Constraint(ForAll(transforms, IsSPL));
        self._resume();
        for f in self.callbacks do f(self); od;
	opts := self.opts;
	date := Date();

	for t in transforms do
            # For run methods that use hash tables
	    if useHash then
		hentry := self._rawentry(t); 
		if hentry = false then
		    res := ApplyFunc(self.(runMethod), runArgs :: [t, opts]);
		    if res[1] = [] then Error(runMethod, " did not find any ruletrees for <t> (", t, ")"); fi;
		    self._saveHash(opts.hashFile, date, opts.hashTable);
		    hentry := self._rawentry(t); 
		    hentry.searchTime := res[2];
		    searchTime := res[2];
		else
		    searchTime := -1;
		fi;
		hentry.spectree := ApplyRuleTreeSPL(hentry.ruletree, t, opts);
                #NOTE: exhaustive search will not update cycles
		cycles := When(IsBound(hentry.measured), hentry.measured, 0);
		ruletree := hentry.spectree;
            # For run methods that don't use hash tables
	    else
		[ruletree, searchTime] := ApplyFunc(self.(runMethod), runArgs :: [t, opts]);
		cycles := 0;
	    fi;

            # HACK: It's a pain to do this in a cleaner way
	    if runMethod = "_pickRandomSave" then return; fi;

	    if self.generateCfiles then
                #NOTE: Shouldn't have to generate code or run this whole thing again.
		c := CodeRuleTree(ruletree, opts);
		cycles := CMeasure(c, opts);
		if useHash then hentry.measured := cycles; fi;

		nopts := CopyFields(opts, rec(fileinfo := rec(
			    cycles  := cycles,
			    flops   := self._rtflops(ruletree),
			    file    := self.fileTransform(t,opts),
			    algorithm := ruletree)));
		PrintTo(self.fileTransform(t,opts), PrintCode(self.funcTransform(t,opts), c, nopts));
            else
                c := skip();
	    fi;
	    
	    self._showStats(runMethod, t, ruletree, c, cycles, searchTime); 
        od;
    end,
));
