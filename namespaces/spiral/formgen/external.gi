
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Verifier := rec(
    MaxError := 6,
    IgnoreRoots := false
);

if not IsBound(GenerateErrorReports) then
    GenerateErrorReports := true;
fi;

#F HandleTestError ( <func-name>, <func-args>, <bool> )
#F    Creates a directory and reports the given function call in
#F    'test.g', backtrace in 'backtrace' and current configuration in
#F    'conf.g'
#F
#F    Boolean parameter, if true, tells HandleTestError to all Error()
#F    thus causing an exception, if false, program will continue
#F    normally.  
#F
HandleTestError := function (FuncName, FuncArgs, doErr)
    local dir,file,arg,n,k;
    if IsBound(GenerateErrorReports) and GenerateErrorReports = false then
	return;
    fi;
    dir := Concat(Conf("tmp_dir"), Conf("path_sep"), "Error-", 
		  String(TimeInSecs()), Conf("path_sep"));
    file := Concat(dir, "test.g");
    SysMkdir(dir);
    # Create a file (truncate if it exists)
    PrintTo(file, "Import(spiral.spl, spiral.nt, spiral.code); \n");
#F    AppendTo(file, "config_update_val(\"remove_temporaries\", SRC_CMDLINE, int_val(0));\n");
#F    AppendTo(file,"config_update_val(\"tmp_dir\", SRC_CMDLINE, str_val(\"./tmp\"));\n");
    AppendTo(file, "GenerateErrorReports := false; \n");
    AppendTo(file, "SPL_DEFAULTS := ", SPL_DEFAULTS, ";\n");
    # write function arguments
    n := 1;
    for arg in FuncArgs do
	AppendTo(file, "arg", String(n), " := ");
	if IsString(arg) then
	    AppendTo(file, "\"", arg, "\"");
	else
	    AppendTo(file, arg);
	fi;
	AppendTo(file, ";\n\n");
	n := n+1;
    od;

    # write function call
    k := 1;
    AppendTo(file, "result := ", FuncName, "( ");
    while k<>n do
	AppendTo(file, "arg", String(k));
	if k <> n-1 then AppendTo(file, ", "); fi;
	k := k + 1;
    od;
    AppendTo(file, " );\n\n");

    # other info
    # NOTE: implement a config dump in sys_conf
#F    AppendTo(Concat(dir, "conf.g"), ConfigProfileList(), ";\n");
    BacktraceTo(Concat(dir, "backtrace.txt"), 100);

    if doErr then
	Error(FuncName, " failed, see ", dir);
    else
	Print(FuncName, " failed, see ", dir, "\n");
    fi;
end;

#F DeriveSPLOptions ( <spl>, <spl-options-record> )
#F    Merges the defaults with <spl-options-record>, derives other 
#F    fields, such as dataType, from <spl>, and returns a complete
#F    options record.
#F
DeriveSPLOptions := function (S, R)
    # set options
    R := MergeSPLOptionsRecord(R);

    # check if MPI req'd
#    if IsDMP(S) then
#	R.language := "c.mpi.mpich";
#    fi;
  

    # if user didn't specify data type determine it from S
    if R.dataType = "no default" then
	if IsRealSPL(S) then R.dataType := "real";
    	else R.dataType := "complex"; 
	fi;
    else
	;# prevent user from doing nonsense
	#if not IsRealSPL(S) and R.dataType = "real" then
	#    Error("invalid combination: complex <S> and real data type");
	#fi;   
    fi;
    return R;
end;

#F DeriveScalarType ( <spl-options-record> )
#F 
DeriveScalarType := function(SPLOpts) 
    local suffix;
    if IsBound(SPLOpts.customDataType) then return SPLOpts.customDataType;
    elif IsBound(SPLOpts.customReal) and SPLOpts.dataType = "real" then return SPLOpts.customReal;
    elif IsBound(SPLOpts.customComplex) and SPLOpts.dataType = "complex" then return SPLOpts.customComplex;
    else
	if SPLOpts.dataType = "real" then suffix := "";
	elif SPLOpts.dataType = "complex" then suffix := "_cplx";
	else Error("SPLOpts.dataType has invalid value '", SPLOpts.dataType, "'");
	fi;
	if SPLOpts.precision = "single" then return Concat("float",suffix);
	elif SPLOpts.precision = "double" then return Concat("double",suffix);
	elif SPLOpts.precision = "extended" then return Concat("long_double",suffix);
	else Error("SPLOpts.precision has invalid value '", SPLOpts.dataType, "'");
	fi;
    fi;
end;

ProgInputType := rec(
    SPLSource := 0,
    TargetSource := 1,
    ObjFile := 2
);

#F DeriveType ( <spl-options-record> )
DeriveType := (opts)->When(IsBound(opts.vector), opts.vector.isa.ctype,
    DeriveScalarType(opts));

#F ProgSPL ( <spl> , <spl-options-record> )
#F    Convert <spl> to a 'Prog' record used by xxxProg functions.
#F    See gap/src/spiral_spl_prog.c for details.
#F
ProgSPL := function (SPL, Opts)
    local prog;
    Opts := DeriveSPLOptions(SPL, Opts);
    prog := rec();
    prog.profile   := Opts.language;
    prog.type      := ProgInputType.SPLSource;
    prog.data_type := DeriveScalarType(Opts);

    if IsBound(Opts.zeroBits) then prog.zero_bits := Opts.zeroBits;
    else prog.zero_bits := 0; fi;

    prog.dim_rows  := EvalScalar(SPL.dimensions[1]);
    prog.dim_cols  := EvalScalar(SPL.dimensions[2]);
    prog.auto_dim  := 0;

    if IsBound(Opts.compflags) then prog.compiler_flags := Opts.compflags; fi;

    if IsBound(Opts.file) then prog.file := Opts.file;
    else prog.file := SysTmpName(); fi;

    if IsBound(Opts.subName) then prog.sub_name := Opts.subName;
    else prog.sub_name := "sub"; fi;

    prog.spl_file := prog.file;
    return prog;
end;

#F   Valid <compare-type>'s are
#F     "random": compare on random vector (default)
#F     "basis" : compare on standard basis
#F     <int>   : compare on <int> random standard base vectors
#F
VerifierOpts := function (CO)
    local opts;
    opts := Concat("-g -e ", String(Verifier.MaxError), " ");
    if CO = "basis" then return Concat(opts, " -b");
    elif CO = "random" then return Concat(opts, " -r");
    elif IsInt(CO) then return Concat(opts, " -s ", String(CO));
    else Error("<CO> must be an integer, \"random\", or \"basis\"");
    fi;
end;
