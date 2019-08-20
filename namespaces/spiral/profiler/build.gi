
# Copyright (c) 2018-2019, Carnegie Mellon University
# See LICENSE for details


Import(paradigms.distributed);

# local functions and variables are prefixed with an underscore.

_WriteStub := function(code, outdir, opts)
    local outstr, s, stub, outfile, i;
    
    outfile := Concat(outdir, "/testcode.h");
    
    # build the generic stub info
    stub := CopyFields(rec(
        MEM_SIZE := When(IsBound(code.dimensions), EvalScalar(code.dimensions[1]) * 4, 0),
        DATATYPE := Concat("\"", DeriveScalarType(opts), "\""),
        DATATYPE_NO_QUOTES := DeriveScalarType(opts),
        PAGESIZE := 4096,
        INITFUNC := "init_sub",
        FUNC := "sub",
        DATATYPE_SIZEINBYTES := Cond(DeriveScalarType(opts) = "float", 4, Cond(DeriveScalarType(opts) = "double", 8, 0)),
        OUTDIR := Concat("\"", outdir, "\""),
        NUMTHREADS := When(IsBound(opts.smp) and IsInt(opts.smp.numproc), opts.smp.numproc, 1),
        RADIX :=      When(IsBound(opts.smp) and IsInt(opts.smp.numproc) and IsBound(code.dimensions), (code.dimensions[1]/(2 * opts.smp.numproc)), 2),
        QUICKVERIFIER := When(IsBound(opts.quickverifier), opts.quickverifier, "nulltransform")
    ));
	
    if IsBound(code.dimensions) then
        stub.ALLOCATE_MEMORY := "1";
        stub.ROWS := EvalScalar(code.dimensions[1]);
        stub.COLUMNS := EvalScalar(code.dimensions[2]);
    else
        stub.ALLOCATE_MEMORY := "0";
    fi;

    outstr := opts.unparser.generated_by;
    for s in UserRecFields(stub) do
        if s <> "operations" then
            if IsList(stub.(s)) and not IsString(stub.(s)) then
                outstr := Concat(outstr, "#define ", s, " { ");
                for i in [1..Length(stub.(s))-1] do
                   outstr := Concat(outstr, String(stub.(s)[i]), ", ");
                od;
                outstr := Concat(outstr, String(Last(stub.(s))), " }\n");
            else
                outstr := Concat(outstr, "#define ", s, " ", String(stub.(s)), "\n");
            fi;
        fi;
    od;

    PrintTo(outfile, outstr);
end;


_RunMake := function(profile, outdir, makeTarget)
	local fullcmd, tmpfile, errorvalue, target, retval;
	
	#PrintLine("_RunMake(), makeTarget: \"", makeTarget, "\"");
	
	if (makeTarget = "clean") then
		# backend server handles cleans
		return 0;
	elif not makeTarget in [""] then
		PrintLine("Unknown makeTarget for _RunMake: \"", makeTarget, "\"");
		return -1;
	fi;
	
	target := When(IsBound(profile.target), profile.target, rec());
	
	tmpfile := Concat(outdir, "/time.txt");
	fullcmd := Concat("spiralprofiler -d ", outdir);
	
	if IsBound(target.host) then
		fullcmd := Concat(fullcmd, " -H ", String(target.host));
	fi;
	
	if IsBound(target.port) then
		fullcmd := Concat(fullcmd, " -p ", String(target.port));
	fi;
	
	if IsBound(target.name) then
		fullcmd := Concat(fullcmd, " -t ", String(target.name));
	fi;

    if IsBound(target.prefix) then
        fullcmd := Concat(fullcmd, " -P ", String(target.prefix));
	else
		fullcmd := Concat(fullcmd, " -P ", String(GetPid()), "_");
    fi;
	
	# uncomment following line to hide profiler debug and error messages
	#fullcmd := Concat(fullcmd, " 2> NUL");
	
	PrintLine(fullcmd);
	
	errorvalue := IntExec(fullcmd);
	
	if (errorvalue <> 0) then
        Error("Make failed with error value ", errorvalue, ":\n", fullcmd, "\n");
    fi;

    return ReadVal(tmpfile);
end;


_BuildStatusRelatedActions := function(profile, outdir, stub, opts)
  return 0;
end;


_CellSPUMeasureVerify := function(code, opts, makeTarget)
    local c, p;
    return _StandardMeasureVerify(code, opts, makeTarget);
end;


_MakeOutDirString := (opts) -> Concat(
    Cond(IsBound(opts.outdir),
        opts.outdir,
        "/tmp"
    ),
    "/",
    String(GetPid())
);


#
## _MakeOutDir
#
# create the output directory.
#

_MakeOutDir := function(opts)
    local outdir;

    outdir := _MakeOutDirString(opts);

    if '~' in outdir or '$' in outdir then
        Error("No shell vars are allowed at the moment.");
    fi;

    MakeDir(outdir);

    return outdir;
end;

#code_seq_num := 1;
#code_rec_out := "";
#ruletree_out := "";

_CallProfiler := function(request, code, opts)
    local outdir, fullcmd, errorvalue, retval, target, outputFile, operations;
	
	#PrintLine("_CallProfiler(), request: \"", request, "\"");
	#PrintLine("------------------------");
	#PrintLine("Profiler sequence ", code_seq_num);
	
	#code_rec_out := Concat("coderec_", String(GetPid()), "_", String(code_seq_num), ".txt");
	#ruletree_out := Concat("ruletree", String(GetPid()), "_", String(code_seq_num), ".txt");
	#code_seq_num := code_seq_num + 1;
	
	#PrintTo(code_rec_out, PrintRec(code));
	#PrintTo(ruletree_out, code.ruletree);
		
	if (request = "") then
		request := "time";
	fi;

    outdir := _MakeOutDir(opts);
    
    # output the spiral 'code' representation to the output dir.
    PrintTo(Concat(outdir, "/code.g"), code);

    # output the full options and profile info
    # PrintRec would be great if it didn't freeze the second time you call it for some reason
    if IsBound( opts.operations )  then
        operations := opts.operations;
    fi;
    Unbind(opts.operations);
    PrintTo(Concat(outdir, "/opts.g"), Print(opts));
    if IsBound( operations )  then
        opts.operations := operations;
    fi;

    # generate C code to testcode file
    PrintTo(Concat(outdir, "/testcode.c"), opts.unparser.gen("sub", code, opts));
    
    # write testcode.h
    _WriteStub(code, outdir, opts);

    target := When(IsBound(opts.target), opts.target, rec());
    outputFile := Concat(outdir, "/", request, ".txt");
    fullcmd := Concat("spiralprofiler -d ", outdir);
    fullcmd := Concat(fullcmd, " -r ", request);
    
    if IsBound(target.host) then
        fullcmd := Concat(fullcmd, " -H ", String(target.host));
    fi;
    
    if IsBound(target.port) then
        fullcmd := Concat(fullcmd, " -p ", String(target.port));
    fi;
    
    if IsBound(target.name) then
        fullcmd := Concat(fullcmd, " -t ", String(target.name));
    fi;

    if IsBound(target.prefix) then
        fullcmd := Concat(fullcmd, " -P ", String(target.prefix));
	else
		fullcmd := Concat(fullcmd, " -P ", String(GetPid()), "_");
    fi;

    # Exec the profiler
	# uncomment following line to hide profiler debug and error messages
	#fullcmd := Concat(fullcmd, " 2> NUL");
	
	PrintLine(fullcmd);
	
	errorvalue := IntExec(fullcmd);
    
    if (errorvalue <> 0) then
        Print("Profiler failed with error value ", errorvalue, ":\n", fullcmd, "\n");
        return 1e100;
    fi;

	retval := ReadVal(outputFile);
	
	#PrintLine("Profiler returned: ", retval);
	
    return retval;
end;

CMeasure := (code, opts) -> _CallProfiler("time", code, opts);

CMatrix := function(code, opts)
	ret := _CallProfiler("matrix", code, opts);
	return Cond(IsMat(ret), TransposedMat(ret), ret);
end;

#F Find maximum memory requirement of all arrays
#F Used to determine memory arena size (for temp array reuse)
findMaxMemReq := function(c)
   local i, maxOfChildren, thisChild, myMem, arrays, other;

   if not IsRec(c) or not IsBound(c.rChildren) then
      return 0;
   fi;

   # Catch init functions and return 0
   if ObjId(c) = func and c.id = "init" then
      return 0;
   fi;

   # Determine the mem requirements of each of our children recursively, and
   # take the max of this.
   maxOfChildren := 0;
   for i in c.rChildren() do
      thisChild := findMaxMemReq(i);
      if (thisChild > maxOfChildren) then
         maxOfChildren := thisChild;
      fi;
   od;

   # Determine our own memory requirements (recursion leaf also)
   myMem := 0;
   if ObjId(c) = decl then
      [arrays, other] := SplitBy(c.vars, x->IsArray(x.t));
      for i in arrays do
        #HACK to exclude twiddles
        if i.id[1] <> 'D' then
           myMem := myMem + (i.t.size * When(IsBound(i.t.t) and ObjId(i.t.t)=TVect, i.t.t.size, 1));
        fi;
      od;
   fi;

   #Print(ObjId(c), " myMem = ", myMem, ". maxOfChildren = ", maxOfChildren, "\n");
   return(myMem + maxOfChildren);
end;

#F This function is exclusively for the Cell
BuildStubOpts := function (code, opts)
    local arena_size, dist_loops, mbuf_its, memloop_its, parallel_its;
    if IsBound(opts.useMemoryArena) and opts.useMemoryArena then
       arena_size := findMaxMemReq(code);
       if ObjId(arena_size) = Value then arena_size := arena_size.v; fi;
       opts.profile.stubopts.ARENA_SIZE := arena_size;
    else
       opts.profile.stubopts.ARENA_SIZE := 1;
    fi;

    # # For the Cell: add parallelization param to profile.
    # if IsBound(opts.spus) then
    #    opts.profile.stubopts.SPUS := opts.spus;
    # fi;
    # NOTE (clean this up)
    # Override spus with info extracted from code - for DP to work properly.
    dist_loops := Collect(code, dist_loop);
    if Length(dist_loops) >=1 then
       opts.profile.stubopts.SPUS := dist_loops[1].P;
    else
       opts.profile.stubopts.SPUS := 1;
    fi;

    # # For the Cell: add multibuffer param to profile.
    # if IsBound(opts.multibuffer_its) then
    #    opts.profile.stubopts.MULTIBUFFER_ITERATIONS := opts.multibuffer_its;
    # fi;
    # NOTE (clean this up)
    # Override mbuf_its with info extracted from code - for DP to work properly.
    mbuf_its := Collect(code, multibuffer_loop);
    if Length(mbuf_its) >=1 then
       # Write the loop with the smallest mbuf_its range to stub.h since the backend uses this value to divide and declare arrays
       opts.profile.stubopts.MULTIBUFFER_ITERATIONS :=  Minimum(List([1..Length(mbuf_its)], i->Length(mbuf_its[i].range)));
       # Write info on whether last mbuf stage is a ping or a pong (will result of computation end up in X or Y)?
       opts.profile.stubopts.MBUF_ENDS_IN_Y :=  Length(mbuf_its) mod 2;

       # We're probably doing internal streaming, so let the backend know of that.
       opts.profile.stubopts.INTERNAL_MULTIBUFFERING  := 1;
    else
       opts.profile.stubopts.MULTIBUFFER_ITERATIONS := 1;
       opts.profile.stubopts.MBUF_ENDS_IN_Y := 1;
    fi;

    memloop_its := Collect(code, mem_loop);
    if Length(memloop_its) >=1 then
       opts.profile.stubopts.VECTOR_IN_MEM := Length(memloop_its[1].range);
       opts.profile.stubopts.MBUF_ENDS_IN_Y :=  Length(memloop_its) mod 2;
    else
       if IsBound(opts.profile.stubopts.VECTOR_IN_MEM) then
        Unbind(opts.profile.stubopts.VECTOR_IN_MEM);
       fi;
    fi;

    parallel_its := Collect(code, @(1, func, e->ObjId(e.cmd)=dist_loop)); # Only checking for topmost level of parallelism
    if Length(parallel_its) >=1 then
       opts.profile.stubopts.PARALLEL_ITERATIONS := parallel_its[1].cmd.P;
    else
       opts.profile.stubopts.PARALLEL_ITERATIONS := 1;
    fi;
end;

_StandardMeasure := (code, opts) -> _CallProfiler("time", code, opts);

_StandardBuild := (code, opts) -> _CallProfiler("build", code, opts);

_StandardMeasureVerify := function(code, opts, makeTarget)
	#PrintLine("_StandardMeasureVerify(), makeTarget: \"", makeTarget, "\"");
	if (makeTarget = "") then
		makeTarget := "time";
	fi;
	return _CallProfiler(makeTarget, code, opts);
end;

# This is the verify function for the MMM
_VerifyMMM := function (code, opts, mmm, makeTarget)
    local cgenFunc, outdir, profile, origprofile, stub;

    # grab the profile and makefile options
    origprofile := When(IsBound(opts.profile), opts.profile, default_profiles.linux_x86_gcc);
    profile := Copy(origprofile);
    profile.makeopts := CopyFields(_default_makeopts, When(IsBound(profile.makeopts), profile.makeopts, rec()));

    # determine the output directory and make sure it exists
    outdir := _MakeOutDir(opts);

    # build the stub info
    stub := CopyFields(rec(
        tM := mmm.params[1],
        tN := mmm.params[2],
        tK := mmm.params[3],
        DATATYPE := DeriveType(opts),
        PAGESIZE := 4096,
        INITFUNC := "init_sub",
        FUNC := "sub",
        SUBFUNCTION := opts.subName,
        INITSUB := opts.subInitName
        ), When(IsBound(profile.stubopts), profile.stubopts, rec()));

    # prepend the output dir.
    profile.makeopts.STUB := Concat(outdir, "/", profile.makeopts.STUB);
    profile.makeopts.OUTDIR := outdir;

    _BuildStatusRelatedActions(profile, outdir, stub, opts);

    # prepend the output dir, generate C code and run make
    profile.makeopts.GAP := Concat(outdir, "/", profile.makeopts.GAP);
    PrintTo(profile.makeopts.GAP, opts.unparser.gen(stub.FUNC, code, opts));
    return _RunMake(profile, outdir, makeTarget);
end;

_VerifyViterbi := function (code, opts, R, K, F, P, ebn0, trials, makeTarget)
    local cgenFunc, outdir, profile, origprofile, stub;

    # grab the profile and makefile options
    origprofile := When(IsBound(opts.profile), opts.profile, default_profiles.linux_x86_gcc);
    profile := Copy(origprofile);
    profile.makeopts := CopyFields(_default_makeopts, When(IsBound(profile.makeopts), profile.makeopts, rec()));

    # determine the output directory and make sure it exists
    outdir := _MakeOutDir(opts);

    # build the stub info
    stub := CopyFields(rec(
         K := K,
         RATE := R,
         POLYS := P,
         NUMSTATES := 2^(K-1),
         FRAMEBITS := F,
         DECISIONTYPE := opts.(Concatenation(opts.ViterbiDecisionType.name, "Ctype")),
         DECISIONTYPE_BITSIZE := opts.ViterbiDecisionType.bits,
         COMPUTETYPE := When(IsBound(opts.vector),
                 opts.unparser.ctype(opts.ViterbiComputeType, opts.vector.isa),
                 opts.(Concatenation(opts.ViterbiComputeType.name, "Ctype"))),
         EBN0 := ebn0,
         TRIALS := trials,
     __int32 := "int",
         FUNC := "FULL_SPIRAL"
        ), When(IsBound(profile.stubopts), profile.stubopts, rec()));

    # target dependent commands
    if (makeTarget="verify") then
        stub.DONOTMEASURE := true;
        stub.METRICSHIFT := When(IsBound(opts.metricshift),
            opts.metricshift, 0);
        stub.PRECISIONSHIFT := When(IsBound(opts.precisionshift),
            opts.precisionshift, 0);
        stub.RENORMALIZE_THRESHOLD := When(IsBound(opts.metricspread),
            opts.metricspread, "2000000000");
    elif (not(IsString(makeTarget)) and IsList(makeTarget)) then
        stub.DONOTMEASURE := true;
        stub.GENERICONLY := true;
        stub.METRICSHIFT := makeTarget[1];
        stub.PRECISIONSHIFT := makeTarget[2];
        stub.RENORMALIZE_THRESHOLD := makeTarget[3];
        stub.COMPUTETYPE := "int";
    elif (makeTarget="full") then
        stub.METRICSHIFT := When(IsBound(opts.metricshift),
            opts.metricshift, 0);
        stub.PRECISIONSHIFT := When(IsBound(opts.precisionshift),
            opts.precisionshift, 0);
        stub.RENORMALIZE_THRESHOLD := When(IsBound(opts.metricspread),
            opts.metricspread, "2000000000");
    else
        stub.DONOTVERIFY := true;
        stub.METRICSHIFT := When(IsBound(opts.metricshift),
            opts.metricshift, 0);
        stub.PRECISIONSHIFT := When(IsBound(opts.precisionshift),
            opts.precisionshift, 0);
        stub.RENORMALIZE_THRESHOLD := When(IsBound(opts.metricspread),
            opts.metricspread, "2000000000");
    fi;

    if IsBound(opts.viterbi_file) then
    profile.makeopts.STUB := Concatenation(opts.viterbi_file, ".h");
    else
    profile.makeopts.STUB := Concat(outdir, "/", profile.makeopts.STUB);
    profile.makeopts.OUTDIR := outdir;
    fi;

    _BuildStatusRelatedActions(profile, outdir, stub, opts);

    # prepend the output dir, generate C code and run make
    if IsBound(opts.viterbi_file) then
    profile.makeopts.GAP := Concatenation(opts.viterbi_file, ".c");
    else
    profile.makeopts.GAP := Concat(outdir, "/", profile.makeopts.GAP);
    fi;
    PrintTo(profile.makeopts.GAP, opts.unparser.gen(stub.FUNC, code, opts));

    if IsBound(opts.viterbi_file) then
    return;
    fi;
    makeTarget :="verify"; #everything is inside the Verifier

    return _RunMake(profile, outdir, makeTarget);
end;


RTMeasure := function(rt, opts)
    local s, c, r, dir;

    if not IsBound(opts.profile) then
        Print("opts.profile must be set. This routine does not work with the old profile system.");
        return;
    fi;

    s := SumsRuleTree(rt, opts);

    c := CodeSums(s, opts);

    r := CMeasure(c, opts);

    dir := _MakeOutDir(opts);

    # output the ruletree & sigmaspl as files in the destination folder.
    PrintTo(Concat(dir, "/ruletree.g"), rt);
    PrintTo(Concat(dir, "/sigmaspl.g"), s);

    return r;
end;

FPGAPrintHelp := function(t, p, r, x)
    Print("points(", p, ")\n");
    Print("threshold(", t, ")\n");
    Print("radix(", r, ")\n");
    Print(x);
end;

# this is the measure function for Spiral FPGA code
_FPGAMeasureVerify := function (SPL, opts, makeTarget)
    local cgenFunc, outdir, profile, origprofile, stub;


    # grab the profile and makefile options
    origprofile := opts.profile;
    profile := Copy(origprofile);
    profile.makeopts := CopyFields(_default_makeopts, When(IsBound(profile.makeopts), profile.makeopts, rec()));

    # determine the output directory and make sure it exists
    profile.outdir := Concat(When(IsBound(opts.outdir), opts.outdir, profile.outdir),
        "/", String(GetPid()));
    MakeDir(profile.outdir);

    # prepend the output dir.
    profile.makeopts.STUB := Concat(profile.outdir, "/", profile.makeopts.STUB);
    profile.makeopts.OUTDIR := profile.outdir;


    # prepend the output dir, generate  code and run make
    profile.makeopts.GAP := Concat(profile.outdir, "/", profile.makeopts.GAP);
    PrintTo(profile.makeopts.GAP, FPGAPrintHelp(profile.threshold, SPL.dimensions, 2, SPL));
    return _RunMake(profile, profile.outdir, makeTarget);
end;

#
## _GetSimicsData
#
# Return the full Simics/Flexus stat output as a SPIRAL readable array.
#
# Assume the data file exists. Assume 'awk' is installed.
#
_GetSimicsData := function(opts)
    local outdir, fn, fnout, fullcmd;

    outdir := _MakeOutDirString(opts);

    fn := Concat(outdir, "/timer_db");
    fnout := Concat(outdir, "/timer_db.gi");

    # this awk command turns the simics output into something that
    # SPIRAL can read.
    fullcmd := ApplyFunc(Concat, [
        "cat ",
        fn,
        " | awk 'BEGIN {print \" [\";} {print \"[\\\"\" $1 \"\\\", \" $2 \"],\"} END{print \"[\\\"bogus\\\", 0]];\"}' > ",
        fnout
    ]);

    IntExec(fullcmd);

    return ReadVal(fnout);
end;

