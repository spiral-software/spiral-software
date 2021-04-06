
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


ImportAll(spiral.formgen);
ImportAll(spiral.transforms);
ImportAll(spiral.compiler);

doScalarDft := function(arg)
	local szs, userOpts, isSingle, opts, direction, func, file, t, rt, ir, N, cycles, flops;
	
	if not (Length(arg) in [1..2]) then
		Error("usage: doScalarDft(sizes, [userOptions])");
	fi;
	
	szs      := When(IsList(arg[1]), arg[1], [ arg[1] ]);
	userOpts := When(IsBound(arg[2]), arg[2], rec());
	
	isSingle := IsBound(userOpts.precision) and userOpts.precision = "single";
	
	opts := CopyFields(SpiralDefaults, rec(
				precision  := When(isSingle, "single", "double"),
				TRealCtype := When(isSingle, "float", "double"),
			));
	
	opts.globalUnrolling := 512;
		
	direction := When(IsBound(userOpts.transInverse) and userOpts.transInverse, 1, -1);
	
	for N in szs do
		if IsBound(userOpts.functionNameRoot) then
			func := ReplaceAll(userOpts.functionNameRoot, "%sz1", StringInt(N));
		else
			func := When(direction=1,"i","")::"DFT"::When(isSingle, "_SC", "_DC")::StringInt(N);
		fi;
		
		file := func::".c";
		
		t    := DFT(N, direction);
		rt   := RandomRuleTree(t, opts);
		ir   := CodeRuleTree(rt, opts);
		
		PrintTo(file, PrintCode(func, ir, opts));
		
		cycles := CMeasure(ir, opts);
		flops:= t.normalizedArithCost();
		PrintLine(t, "  ", cycles, " [cyc]  ", _compute_gflops(flops, cycles), " [Gf/s]");
	od;
	
	return true;
end;


doScalar2DDft := function(arg)
	local szs, userOpts, isSingle, opts, direction, func, file, t, rt, ir, dims, cycles, flops;
	
	if not ((Length(arg) in [1..2]) and IsList(arg[1]) and Length(arg[1]) > 0) then
		Error("usage: doScalar2DDft(sizes, [userOptions])");
	fi;
	
	szs      := arg[1];
	if Length(szs) = 2 and IsInt(szs[1]) and IsInt(szs[2]) then
		szs := [ szs ];
	elif not ForAll(szs, dims -> Length(dims) = 2 and IsInt(dims[1]) and IsInt(dims[2])) then
		Error("sizes must be one or more pairs of integers");
	fi;
	
	userOpts := When(IsBound(arg[2]), arg[2], rec());
	
	isSingle := IsBound(userOpts.precision) and userOpts.precision = "single";
	
	opts := CopyFields(SpiralDefaults, rec(
				precision  := When(isSingle, "single", "double"),
				TRealCtype := When(isSingle, "float", "double"),
			));
				
	opts.globalUnrolling := 128;
	
	direction := When(IsBound(userOpts.transInverse) and userOpts.transInverse, 1, -1);
	
	for dims in szs do
		if IsBound(userOpts.functionNameRoot) then
			func := ReplaceAll(userOpts.functionNameRoot, "%sz1", StringInt(dims[1]));
			func := ReplaceAll(func, "%sz2", StringInt(dims[2]));
		else
			func := When(direction=1,"i","")::"DFT"::When(isSingle, "_SC", "_DC")::StringInt(dims[1])::"x"::StringInt(dims[2]);
		fi;
		
		file := func::".c";
		
		t    := MDDFT(dims, direction);
		rt   := RandomRuleTree(t, opts);
		ir   := CodeRuleTree(rt, opts);
		
		PrintTo(file, PrintCode(func, ir, opts));
		
		cycles := CMeasure(ir, opts);
		flops:= t.normalizedArithCost();
		PrintLine(t, "  ", cycles, " [cyc]  ", _compute_gflops(flops, cycles), " [Gf/s]");
	od;
	
	return true;
end;



Class(ScriptGenScalar, ScriptGenBase, rec(

	_arch := () -> "Scalar",


	_init := meth(self)
		self._setTransform(SGKEY_FFT);
		self._setType(SGKEY_SPCX);
		self._setSize(16);
		self._setFilename("");
	end,
	
	
	_validTransforms := meth(arg)
		return [SGKEY_FFT, SGKEY_IFFT, SGKEY_FFT_2D, SGKEY_IFFT_2D];
	end,
	
	
	_validSizes := meth(arg)
		local self, szs, xform, type;
		self  := arg[1];
		szs   := [];
		xform := self.getSettingsValue(SGKEY_TRANSFORM);
		type  := self.getSettingsValue(SGKEY_DATATYPE);
		
		if xform in [SGKEY_FFT, SGKEY_IFFT] then
			szs := Filtered([2..512], i->ForAll(Factors(i), j->j<=19));
			Append(szs, Filtered([513..1024], i->ForAll(Factors(i), j->j<=19) and IsInt(i/16)));
			Append(szs, List([11..16], i->2^i));
		elif xform in [SGKEY_FFT_2D, SGKEY_IFFT_2D] then
			szs := List(Filtered([2..360], i->ForAll(Factors(i), j->j<=19)), n -> [n, n]);
		elif xform = SGKEY_WHT then
			if type = SGKEY_DPCX then
				szs := List([2..10], i->2^i);
			else
				szs := List([4..10], i->2^i);
			fi;
		fi;
				
		return szs;
	end,
	
	
	_validTypes := meth(arg)
		return [SGKEY_SPCX, SGKEY_DPCX];
	end,

	
	getScriptChoices := (self) >> [SGSTR_RUNRANDOMALL], 
	
	
	_genScript := meth(self, runType)
		local scrstr, xform, type, szs, tempopts, optrec, optstr, funcstr;
		
		xform	 := self.getSettingsValue(SGKEY_TRANSFORM);
		type	 := self.getSettingsValue(SGKEY_DATATYPE);
		szs      := self.getSettingsValue(SGKEY_SIZE);
		
		tempopts := "scriptOpts";
		
		optrec	 := rec(
						precision  := When(type=SGKEY_SPCX, "single", "double"),
						searchType := runType,
					);
		if IsBound(self._settings.(SGKEY_FUNCNAME)) then
			optrec.functionNameRoot := self._settings.(SGKEY_FUNCNAME);
		fi;
		if xform in [SGKEY_IFFT, SGKEY_IFFT_2D] then
			optrec.transInverse := true;
		fi;
				
		if xform in [SGKEY_FFT, SGKEY_IFFT] then
			funcstr := "doScalarDft("::String(szs)::", "::tempopts::")";
		elif xform in [SGKEY_FFT_2D, SGKEY_IFFT_2D] then
			funcstr := "doScalar2DDft("::String(szs)::", "::tempopts::")";
		else
			return "";
		fi;
				
		optstr := StringPrint(optrec);
		
		scrstr := "Import(scriptgen);;\n";
		Append(scrstr, tempopts::" := "::optstr::";;\n");
		Append(scrstr, funcstr::";;\n");
			
		return scrstr;
	end,
)); # Class ScriptGenScalar


SetScriptGenConstructor(ScriptGenScalar);

