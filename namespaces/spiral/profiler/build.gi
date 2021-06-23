
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Import(paradigms.distributed);

# local functions and variables are prefixed with an underscore.

_WriteStub := function(code, opts)
    local outstr, s, stub, i, testvec, multiline, bounds, corner;
    
    # build the generic stub info
    stub := CopyFields(rec(
        MEM_SIZE := When(IsBound(code.dimensions), EvalScalar(code.dimensions[1]) * 4, 0),
        DATATYPE := Concat("\"", DeriveScalarType(opts), "\""),
        DATATYPE_NO_QUOTES := DeriveScalarType(opts),
        PAGESIZE := 4096,
        INITFUNC := "init_sub",
        FUNC := "sub",
        DESTROYFUNC := "destroy_sub",
        DATATYPE_SIZEINBYTES := Cond(DeriveScalarType(opts) = "float", 4, Cond(DeriveScalarType(opts) = "double", 8, 0)),
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

    Print(outstr);

    ##  add extern function declarations ... required for cuda
    Print("\nextern void INITFUNC();\n");
    Print("extern void DESTROYFUNC();\n");
    Print("extern void FUNC( ", DeriveScalarType(opts), " *out, ", DeriveScalarType(opts), " *in );\n");
    
	#add testvector if specified in opts
	
	if IsBound(opts.testvector) then
		testvec := opts.testvector;
		multiline := false;
		if not IsVector(testvec) then
			Error("opts.testvector must be a valid vector");
		fi;
		Print("\n\nstatic ", DeriveScalarType(opts), " testvector[] = {");
		if Length(testvec) > 10 then
			Print("\n    ");
		fi;
		for i in [1 .. Length(testvec)] do
			if i > 1 then
				Print(", ");
				if Mod(i, 10) = 1 then
					Print("\n    ");
					multiline := true;
				fi;
			fi;
			Print(testvec[i]);
		od;
		if multiline then
			Print("\n");
		fi;
		Print("};\n");
	fi;
	
	if IsBound(opts.cmatrixBounds) then
		bounds := opts.cmatrixBounds;
		if not (IsMat(bounds) and Length(bounds) = 2 and
				Length(bounds[1]) = 2 and Length(bounds[2]) = 2) then
			Error("opts.cmatrixBounds must be a list [[ur,uc], [lr,lc]]");
		fi;
		Print("\n");
		Print("#define CMATRIX_UPPER_ROW ", bounds[1][1], "\n");
		Print("#define CMATRIX_UPPER_COL ", bounds[1][2], "\n");
		Print("#define CMATRIX_LOWER_ROW ", bounds[2][1], "\n");
		Print("#define CMATRIX_LOWER_COL ", bounds[2][2], "\n");
	fi;
end;


_MakeOutDirString := function(opts)
	local tmp;
	
	tmp := GetEnv("SPIRAL_TEMP_OUT_PATH");
	if (tmp = "") then
        tmp := "/tmp";
    fi;
    return Concat(
		Cond(IsBound(opts.outdir), opts.outdir, tmp),
		"/",
		String(GetPid()));
end;


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


_CallProfiler := function(request, code, opts)
    local outdir, fullcmd, errorvalue, retval, target, outputFile, operations;
	
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
    PrintTo(Concat(outdir, "/testcode.h"), _WriteStub(code, opts));

    target := When(IsBound(opts.target), opts.target, rec());
    outputFile := Concat(outdir, "/", request, ".txt");
    fullcmd := Concat("spiralprofiler -d ", outdir);
    fullcmd := Concat(fullcmd, " -r ", request);
    
    if IsBound(target.forward) then
        fullcmd := Concat(fullcmd, " -f ", String(target.forward));
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
	
    return retval;
end;


## CMeasure(code, opts)
##
## Call profiler to time transform implemented by code

CMeasure := (code, opts) -> _CallProfiler("time", code, opts);

## CMatrix(code, opts, <bounds>)
##
## Call profiler to generate matrix equivalent of transform implemented by code
## Optional bounds specifies sub matrix upper/lower corners [[ur,uc], [lr,lc]]

CMatrix := function(arg)
	local code, opts, retmat, oldbounds;
	code := arg[1];
	opts := arg[2];
	if IsBound(opts.cmatrixBounds) then
		oldbounds := opts.cmatrixBounds;
	fi;
	if Length(arg) = 3 then
		opts.cmatrixBounds := arg[3];
	fi;
	retmat := _CallProfiler("matrix", code, opts);
	if IsBound(oldbounds) then
		opts.cmatrixBounds := oldbounds;
	elif IsBound(opts.cmatrixBounds) then
		Unbind(opts.cmatrixBounds);
	fi;
	return Cond(IsMat(retmat), TransposedMat(retmat), retmat);
end;

## CVector(code, vector, opts)
##
## Call profiler to apply transform implemented by code to vector

CVector := function (code, vector, opts)
	local retvec;
	opts.testvector := vector;
	retvec :=  _CallProfiler("vector", code, opts);
	return retvec;
end;


_StandardMeasure := (code, opts) -> _CallProfiler("time", code, opts);

_StandardMeasureVerify := function(code, opts, makeTarget)
	#PrintLine("_StandardMeasureVerify(), makeTarget: \"", makeTarget, "\"");
	if (makeTarget = "") then
		makeTarget := "time";
	fi;
	return _CallProfiler(makeTarget, code, opts);
end;




