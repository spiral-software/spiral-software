
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(DPBench);

#F DPBench(rec(experiment1 := opts1, ...), <dpopts>)
#F
#F d := DPBench(rec(default := SpiralDefaults), rec())
#F
#F DPBench Interface:
#F
#F     .build(transforms, opts, bopts, name, dpopts),
#F
#F     .generateCfiles := [true, false],  default = true
#F     .matrixVerify   := [true, false],  default = false
#F     .fftwVerify     := [true, false],  default = false
#F     .quickVerify     := [true, false],  default = false
#F
#F     .fileTransform(exp, t, opts),    .c filenames:   override in a subclass
#F     .funcTransform(exp, t, opts),    function names: override in a subclass
#F     .txtFileName(exp, runMethod),    timing file name
#F
#F     .runAll()                        run all transforms from every experiment's opts.benchTransforms
#F     .resumeAll()                     same as .runAll() but try to reload hash from disk
#F
#F     .run(transforms)                 run given transforms in all experiments
#F     .resume(transforms)              same as .run() but try to relaod hash from disk
#F
#F      .runRandomAll(),  .runRandom()    run a random ruletree
#F      .runExhaustiveAll(), .runExhaustive()   run all ruletrees
#F
#F     .generateCode(transforms, exp)
#F     .generateProductionCode(transforms, exp)   same as .generateCode() but will use opts.production()
#F
#F     .entries(transforms, exp)
#F     .times(transforms, exp)
#F     .alltimes(transforms)
#F     .speedup(transforms, baselineExp)
#F
#F     .flopcyc(transforms)      # FLoating point Operations Per Cycle
#F     .mflops(transforms, mhz)
#F     .scaledTimes(transforms, scaleFunc)
#F
#F      getResult(i)                    returns "rec(t, opts, rt)" for the <i>th experiment run by runAll()
Class(DPBench, rec(
    ##
    ## Private methods
    ##
    _checkExp := exp ->
        Cond(not IsRec(exp),        Error("Experiments record <exp> must be a record"),
         NumRecFields(exp) < 1, Error("Experiments record is empty"),
         not ForAll(UserRecFields(exp), f -> IsRec(exp.(f))),
                 Error("Each entry must be a valid Spiral options record (ie. SpiralDefaults)"),
         exp),

    _startHashFile := (self, hfile, exp, d) >> PrintTo(hfile,
        "<# DPBench experiment '", exp, "'\n",
        " # Started ", d[2], " ", d[3], " ", d[1], "  ", d[4], ":", d[5], "\n",
        " # Transforms: ", Cond(IsBound(self.transforms), self.transforms, ""), "#> \n\n",
        "ImportAll(spiral); Import(paradigms.common, paradigms.smp, platforms.sse, platforms.avx, paradigms.vector, nontransforms.ol); \n",
        "ImportAll(platforms.scalar); \n",
        "ImportAll(paradigms.vector); \n\n",
        "hash := HashTableDP(); \n"
    ),

    _loadHash := meth(self, hfile, opts)
        local b, bkdowns, ns, result;
	# Create a package that contains mappings from breakdown names to the corresponding
	# objects. This is needed because global names map to global breakdown rule objects
	# which might have different settings from the ones in opts.breakdownRules
	# Note: package is essentially a namespace that is always on top of imports
	#       so imports within a hash file will be superceded by it
	bkdowns := ConcatList(UserRecFields(opts.breakdownRules), f->opts.breakdownRules.(f));
        ns := tab();
	for b in bkdowns do ns.(b.name) := b; od;

        result := READ(hfile, ns);
        if result = false or not IsBound(ns.hash) then return false;
        else return ns.hash;
        fi;
    end,

    _saveHash := meth(self, hfile, exp, date, hash)
        local bucket, e;
        var.print := var.printFull;
        self._startHashFile(hfile, exp, date);
        for bucket in hash.entries do
            for e in bucket do
                if e.data<>[] then
                    AppendTo(hfile, "HashAdd(hash, ", e.key, ", [", e.data[1], "]);\n");
                fi;
            od;
        od;
        var.print := var.printShort;
    end,

    # merging two hash files by taking fastest entries, returns resulting hash
    _mergeHashes := meth(self, src_file1, src_file2, opts)
        local h1, h2;
        h1 := self._loadHash(src_file1, opts);
        h2 := self._loadHash(src_file2, opts);
        Checked(h1<>false and h2<>false,
            HashWalk(h1, function(key, data)
                local d;
                d := HashLookup(h2, key);
                if d=false or (d[1].measured>data[1].measured and data[1].measured>0) then
                    HashAdd(h2, key, data);
                fi;
            end));
        return h2;
    end,

    _reloadAllHashes := meth(self)
       local hash, exp, e;
       for e in UserRecFields(self.exp) do
           exp := self.exp.(e);
           hash := self._loadHash(exp.hashFile, exp);
           if (self.verbosity>0) then
              PrintLine(When(hash=false, "Could not load ", "Loaded "), e, " (", exp.hashFile, ")");
           fi;
           if hash <> false then
               exp.hashTable := hash;
           fi;
       od;
    end,

    _generateCode := meth(self, transforms, exp, opts)
         local entries, e, c, r, t;
         for t in transforms do
             e := self.entries([HashAsSPL(t)], exp)[1];
             if e = false then Error("Transform ", t, " not found in hashTable for experiment '", exp, "'"); fi;
             r := ApplyRuleTreeSPL(e.ruletree, t, opts);
             c := CodeRuleTree(r, opts);
             PrintLine(t, " -> ", self.prodFileTransform(exp, t, opts));
             PrintTo(self.prodFileTransform(exp, t, opts), PrintCode(self.prodFuncTransform(exp, t, opts), c, opts));
         od;
    end,

    ##
    ## Public methods
    ##
    __call__ := meth(self, experiments, dpopts)
        local e, exp;
        self._checkExp(experiments);
        exp:=rec();
        for e in UserRecFields(experiments) do
           exp.(e) := CopyFields(experiments.(e));
           if not IsBound(exp.(e).hashTable) then
               exp.(e).hashTable := HashTableDP(); fi;
           if not IsBound(exp.(e).hashFile) then
               exp.(e).hashFile := Concat(e, ".hash"); fi;
        od;

        return WithBases(self,
            rec(dpopts:=dpopts, ran:=false, exp:=exp, transforms:=[], verbosity:=1, callbacks:=[]));
    end,

    resume := meth(self, transforms)
        if not ForAll(UserRecFields(self.exp), e -> ForAll(self.entries(transforms, e), e->e<>false))
            then self._reloadAllHashes();
        fi;
        self.run(transforms);
    end,

    generateCfiles := true,
    measureFinal := true,
	
	_fileRoot := meth(self, exp, t, opts)
		if IsBound(opts.vector) and IsBound(opts.vector.conf) and IsBound(opts.vector.conf.functionNameRoot) then
			return opts.vector.conf.functionNameRoot;
		else
			return Concat(exp, "_", Drop(CodeletName(CodeletShape(t)), 1));
		fi;
	end,
	
	_freq := meth(self, opts)
		if IsBound(opts.vector) and IsBound(opts.vector.conf) and IsBound(opts.vector.conf.target) and IsBound(opts.vector.conf.target.freq) then
			return opts.vector.conf.target.freq;
		elif IsBound(LocalConfig.cpuinfo.freq) then
			return LocalConfig.cpuinfo.freq;
		else
			return 1000;
		fi;	
	end,

    fileTimer     := (self, exp, t, opts) >> Concat(self._fileRoot(exp, t, opts), ".timer"),
    fileVerifierf := (self, exp, t, opts) >> Concat(self._fileRoot(exp, t, opts), ".verifier"),
    fileStub      := (self, exp, t, opts) >> Concat(self._fileRoot(exp, t, opts), ".h"),
    fileTransform := (self, exp, t, opts) >> Concat(self._fileRoot(exp, t, opts), ".c"),
    funcTransform := (self, exp, t, opts) >> self._fileRoot(exp, t, opts),
    prodFileTransform := (self, exp, t, opts) >> self.fileTransform(exp, t, opts), # used in generateProductionCode
    prodFuncTransform := (self, exp, t, opts) >> self.funcTransform(exp, t, opts),

    txtFileName   := (exp, runMethod) -> Concat(exp, ".", SubString(runMethod, 5), ".txt"),

    verify := meth(self, opts, ruletree, code)
        local mat;
        mat := When(ruletree.node.isReal() or opts.dataType = "complex" or opts.generateComplexCode,
                    MatSPL(ruletree.node),
                    RCMatCyc(MatSPL(ruletree.node)));
        return VerifyMatrixCode(code, mat, opts);
    end,

    #NOTE: Slightly hacked in. Check for opts.profile being bound etc. Look at VerifyMatrixRuleTree.
    verifyfftw := (self, opts, code) >> opts.profile.verifyfftw(code, opts),
    verifyquick := (self, opts, code) >> opts.profile.verifyquick(code, opts),

    resumeAll := meth(self)
        local e;
        for e in UserRecFields(self.exp) do
            if (self.verbosity>0) then
               PrintLine("Resuming ", e);
            fi;
            self.resume(self.exp.(e).benchTransforms);
        od;
    end,

    _runAll := meth(self, runMethod)
        local e;
        for e in UserRecFields(self.exp) do
            PrintLine("Running ", e);
            self.(runMethod)(self.exp.(e).benchTransforms);
        od;
    end,

    allTrees := (self) >> let(exp := self.exp.(UserRecFields(self.exp)[1]),
    List(exp.benchTransforms, t ->
        ApplyRuleTreeSPL( HashLookup(exp.hashTable, HashAsSPL(t))[1].ruletree,
        t, exp))),

    runAll            := (self) >> self._runAll("run"),
    runExhaustiveAll  := (self) >> self._runAll("runExhaustive"),
    runRandomAll      := (self) >> self._runAll("runRandom"),
    runRandomSaveAll  := (self) >> self._runAll("runRandomSave"),
    pickRandomSaveAll := (self) >> self._runAll("pickRandomSave"),

    run            := (self, transforms) >> self._run(transforms, "_runDP", true),
    runExhaustive  := (self, transforms) >> self._run(transforms, "_runExhaustive", true),
    runRandom      := (self, transforms) >> self._run(transforms, "_runRandom", false),
    runRandomSave  := (self, transforms) >> self._run(transforms, "_runRandomSave", true),
    pickRandomSave := meth(self, transforms)
        local generateCfiles;

        #NOTE: once generateCfiles quits measuring things, this can be removed
        generateCfiles := self.generateCfiles;
        self.generateCfiles := false;
        self._run(transforms, "_pickRandomSave", true);
        self.generateCfiles := generateCfiles;
    end,


    # Find best using DP
    _runDP         := (self, e, t, opts) >> TimedAction(DP(t, self.dpopts, opts)),

    # Find best using an exhaustive search
    outputExhaustive := false,
    _runExhaustive := meth(self, e, t, opts)
       local r, searchTime, mincycles, mintree, rt, c, compiletime, cm, measuretime;

       r := AllRuleTrees(t, opts);
       searchTime := 0;
       mincycles := 10^100;

       for rt in r do
          [c,  compiletime] := TimedAction(CodeRuleTreeOpts(rt, opts));
          [cm, measuretime] := TimedAction(CMeasure(c, opts));
          if self.outputExhaustive then _seqPerfStatsGflops(Concat(e, ".Exhaustive-all.txt"), t, self._freq(opts), self.artcost(rt), cm, compiletime+measuretime); fi;
          if (cm < mincycles) then
              mincycles := cm; mintree := Copy(rt);
          fi;
          searchTime:=searchTime+compiletime+measuretime;
       od;
       HashDelete(opts.hashTable,t);
       HashAdd(opts.hashTable, t, [rec(ruletree:=mintree, measured:=mincycles)]);
       return([mintree, searchTime, c, mincycles]);
    end,

    # Run a Random ruletree. Useful for quick, dirty, non-comprehensive tests.
    _runRandom := meth(self, e, t, opts)
       local r, c, rrtime, codetime, runtime, cycles;

       [r, rrtime]      := TimedAction(RandomRuleTree(t, opts));
       [c, codetime]    := TimedAction(CodeRuleTreeOpts(r, opts));
       [cycles, runtime]:= TimedAction(CMeasure(c, opts));

       return([r, (rrtime+codetime+runtime), c, cycles]);
     end,

    # Run random search and save result in hash.
    _runRandomSave := meth(self, e, t, opts)
       local r, c, rrtime, codetime, runtime, cycles;

       [r, rrtime]      := TimedAction(RandomRuleTree(t, opts));
       [c, codetime]    := TimedAction(CodeRuleTreeOpts(r, opts));
       [cycles, runtime]:= TimedAction(CMeasure(c, opts));

       HashDelete(opts.hashTable, t);
       HashAdd(opts.hashTable, t, [rec(ruletree:=r, measured:=runtime)]);

       return([r, (rrtime+codetime+runtime), c, cycles]);
    end,

    # Pick random and save result in hash (NOT measured).
    _pickRandomSave := meth(self, e, t, opts)
       local r, searchtime;
       [r, searchtime] := TimedAction(RandomRuleTree(t, opts));

       HashDelete(opts.hashTable,t);
       HashAdd(opts.hashTable,t,[rec(ruletree:=r)]);

       return([r, searchtime, false, -1]);
    end,

    _run := meth(self, transforms, runMethod, useHash)
        local code, t, e, outf, res, opts, cycles, hentry,  date, i, searchTime, acc, optsForFile, f, ruletree, randomRes;

        Constraint(ForAll(transforms, IsSPL));

        for f in self.callbacks do 
			f(self);
		od;

        for e in UserRecFields(self.exp) do
            opts := self.exp.(e);
            date := Date();

            for t in transforms do
				code := false;
				t := SumsUnification(t, opts);
				if useHash then
					# For run methods that use hash tables
					hentry := HashLookup(opts.hashTable, HashAsSPL(t));
					if hentry = false or hentry = [] then
						res := self.(runMethod)(e, HashAsSPL(t), opts);
						if res[1] = [] then
							Error("DP did not find any ruletrees for <t> (", t, ")");
						fi;
						self._saveHash(opts.hashFile, e, date, opts.hashTable);
						hentry := HashLookup(opts.hashTable, HashAsSPL(t))[1];
						hentry.searchTime := res[2];
						searchTime := res[2];
						if Length(res) >= 3 then
							# _runDP doesn't return code
							code := res[3];
						fi; 
					else
						hentry := hentry[1];
						searchTime := -1;
					fi;
					hentry.spectree := ApplyRuleTreeSPL(hentry.ruletree, t, opts);
                    #NOTE: exhaustive search will not update cycles
					cycles := When(IsBound(hentry.measured), hentry.measured, 0);
					if not t in self.transforms then
						Add(self.transforms, t);
					fi;
					ruletree := hentry.spectree;
				else
					# For run methods that don't use hash tables
					randomRes  := self.(runMethod)(e, HashAsSPL(t), opts);
					ruletree   := ApplyRuleTreeSPL(randomRes[1], t, opts);
					searchTime := randomRes[2];
					code       := randomRes[3];
					cycles     := randomRes[4];
				fi;

				#HACK: It's a pain to do this in a cleaner way
				if runMethod = "_pickRandomSave" then 
					return; 
				fi;

				if self.generateCfiles then
					#NOTE: Should also output stub.h as <filename.h>
					compiler.CMEASURE_CURRENT_TREE := ruletree;
					compiler.CMEASURE_LAST_CODE := false;
					if (code=false or HashAsSPL(t)<>t) then
						code := CodeRuleTree(ruletree, opts);
					fi;
					compiler.CMEASURE_LAST_CODE := code;
					if (self.measureFinal and (runMethod <> "_runRandom"))then 
						cycles := CMeasure(code, opts);
					fi;

					if useHash then
						hentry.measured := cycles;
					fi;

					opts.fileinfo := rec(
						cycles  := cycles,
						flops   := self.artcost(ruletree),
						file    := self.fileTransform(e,t,opts),
						algorithm := ruletree
					);
					PrintTo(self.fileTransform(e,t,opts), PrintCode(self.funcTransform(e,t,opts), code, opts));
					Unbind(opts.fileinfo);
				fi;

				if self.matrixVerify or self.fftwVerify or self.quickVerify then
					if code=false then
						code := CodeRuleTree(ruletree, opts);
					fi;
					if self.matrixVerify then
						acc := self.verify(opts, ruletree, code);
					elif self.fftwVerify then
                        acc := self.verifyfftw(opts, code);
                    else 
                        acc := self.verifyquick(opts, code);
                    fi;
                    if IsBound(opts.outputVecStatistics) and opts.outputVecStatistics then
                        _seqPerfStatsGflopsAccCount(self.txtFileName(e, runMethod), t,
                                [self.artcost(ruletree), code.countedArithCost(opts.vector.isa.countrec)*opts.vector.vlen], cycles, searchTime, acc,
                                    code.countOps(opts.vector.isa.countrec), opts.vector.isa.countrec);
                    else
                        _seqPerfStatsGflopsAcc(self.txtFileName(e, runMethod), t,
                                [self.artcost(ruletree), self.countedArithCost(code, opts)], cycles, searchTime, acc);
                    fi;
                else
					if (opts.verbosity>-1) then
                       if IsBound(opts.outputVecStatistics) and opts.outputVecStatistics then
                            _seqPerfStatsGflopsCount(self.txtFileName(e, runMethod), t,
                                    [self.artcost(ruletree), code.countedArithCost(opts.vector.isa.countrec)*opts.vector.vlen], cycles, searchTime,
                                        code.countOps(opts.vector.isa.countrec), opts.vector.isa.countrec);
                       else
                            _seqPerfStatsGflops(self.txtFileName(e, runMethod), t, self._freq(opts),
                                    [self.artcost(ruletree), self.countedArithCost(code, opts)], cycles, searchTime);
                       fi;
                    fi;
                fi;
            od;
        od;
        self.ran := true;
    end,

    generateCode := (self, transforms, exp) >> self._generateCode(transforms, exp, self.exp.(exp)),

    generateProductionCode := (self, transforms, exp) >> self._generateCode(transforms, exp, self.exp.(exp).production()),

    generateAllCode := self >> DoForAll(UserRecFields(self.exp), exp ->
        self._generateCode(self.exp.(exp).benchTransforms, exp, self.exp.(exp))),

    generateAllProductionCode := self >> DoForAll(UserRecFields(self.exp), exp ->
        self._generateCode(self.exp.(exp).benchTransforms, exp, self.exp.(exp).production())),

    entries := (self, transforms, exp) >>
        When(not IsBound(self.exp.(exp)), Error("No such experiment '",exp, "'"),
         Map(transforms,
         x -> let(lookup := MultiHashLookup(Concatenation([self.exp.(exp).hashTable], self.exp.(exp).baseHashes), x),
                When(lookup = false, false,
                #Error("Transform '", x, "' not found in the '", exp, "' table"),
                lookup[1])))),

    times := (self, transforms, exp) >> Map(self.entries(transforms, exp), x->x.measured),

    alltimes := (self, transforms) >>
        let(tr := When(IsList(transforms), transforms, [transforms]),
            Map(UserRecFields(self.exp), e -> Concatenation([e], self.times(tr, e)))),

    speedup := (self, transforms, baselineExp) >>
        let(b := self.times(transforms, baselineExp),
        Map(self.alltimes(transforms),
            times -> Map([1..Length(times)],
                         i -> When(not IsInt(times[i]), times[i], times[i] / b[i-1])))),

    flopcyc := (self, transforms) >>
        self.scaledTimes(transforms, e -> self.acost(e) / e.measured),

    mflops := (self, transforms, mhz) >>
        self.scaledTimes(transforms, e -> self.acost(e) * mhz / e.measured),

    # FLoating point Operations Per Cycle
    scaledTimes := (self, transforms, scaleFunc) >>
        let(tr := When(IsList(transforms), transforms, [transforms]),
            Map(UserRecFields(self.exp),
                e -> Concatenation([e], Map(self.entries(tr, e), scaleFunc)))),

    # Use NonTerminal.normalizedArithCost() if it is there, otherwise return 0
    acost := entry -> let(
        t := entry.ruletree.node,
        When(IsBound(t.normalizedArithCost), t.normalizedArithCost(), 0)
    ),

    artcost := ruletree -> let(
        t := ruletree.node,
        When(IsBound(t.normalizedArithCost), t.normalizedArithCost(), 0)
    ),

    countedArithCost := (self, c, opts) >>
        When(IsRec(c) and
             IsBound(c.countedArithCost) and
             IsBound(opts.vector) and IsBound(opts.vector.isa) and IsBound(opts.vector.isa.countrec) and IsBound(opts.vector.vlen),
             c.countedArithCost(opts.vector.isa.countrec)*opts.vector.vlen, 0),

    matrixVerify := false,
    setMatrixVerify := self >> CopyFields(self, rec(matrixVerify := true)),

    fftwVerify   := false,
    quickVerify   := false,

    getOpts := self >> self.exp.(UserRecFields(self.exp)[1]),

    _getResult := meth(arg)
        local self, t, exp, lookup, rt;
        self := arg[1];
        exp := UserRecFields(self.exp)[1];
        t := When(Length(arg) >= 2, self.exp.(exp).benchTransforms[arg[2]], self.exp.(exp).benchTransforms[1]);
        lookup := MultiHashLookup(Concatenation([ self.exp.(exp).hashTable ], self.exp.(exp).baseHashes), HashAsSPL(t));
        rt := When(lookup = false, false, lookup[1].ruletree);
        return rec(opts := self.exp.(exp), t := t, rt:= rt);
    end,

    getResult := meth(arg)
        local self, l, rt, opts, c;
        self := arg[1];
        l := When(Length(arg)>1, self._getResult(arg[2]), self._getResult());
        rt := l.rt;
        opts := l.opts;
        c := CodeRuleTree(rt, opts);
        return CopyFields(l, rec(c := c, opcount := c.countOps(opts.vector.isa.countrec)));
    end,

    _runAll := meth(self, runMethod)
        local e;
        for e in UserRecFields(self.exp) do
            PrintLine("Running ", e);
            self.(runMethod)(self.exp.(e).benchTransforms);
        od;
    end,

    build := function(arg)
        local transforms, opts, bopts, name, dpr;

        transforms := When(IsList(arg[1]), arg[1], [arg[1]]);
        opts := When(Length(arg) >= 2, arg[2], SpiralDefaults);
        opts.benchTransforms := transforms;
        bopts := When(Length(arg) >= 3, arg[3], rec());
        name := When(Length(arg) >= 4, arg[4], "spiral");
        dpr := When(Length(arg) >= 5, arg[5], rec(verbosity := 0, timeBaseCases:=true));

        return CopyFields(DPBench(rec((name) := opts), dpr), bopts);

    end

));

# Class(NewDPBench, DPBench, rec(
#     measureNtimes := 10,
#     remeasure := (self, c, opts) >> List([1..self.measureNtimes],
#         i -> CMeasure(c, opts)),

#     runHooks := [
#         meth(self, t, c, hentry, opts)
#             hentry.remeasure := self.remeasure(c, opts);
#             PrintLine("remeasure : ", hentry.remeasure);
#         end
#     ]
# ));

#opts := SpiralDefaults; opts2 := CopyFields(opts, rec(declareConstants := true));
#d := NewDPBench(rec(default:=opts,pullconst:=opts2), rec(timeBaseCases := false, verbosity := 0));
sampleDPBench := DPBench(rec(default := SpiralDefaults), rec(timeBaseCases := false, verbosity := 0));
sampleDPBench.transforms := [DFT(2), DFT(3), DFT(4)];
