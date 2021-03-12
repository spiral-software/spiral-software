
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(ScriptGenAVX, ScriptGenBase, rec(
	

	_arch := () -> "AVX",


	_init := meth(self)
		self._setTransform(SGKEY_FFT);
		self._setType(SGKEY_SPCX);
		self._setSize(64);
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
			if type = SGKEY_DPCX then
				szs := Filtered([1..1024], i->ForAll(Factors(i), j->j<=16) and IsInt(i/16));
				Append(szs, List([11..16], i->2^i));
			else
				szs := Filtered([1..1024], i->ForAll(Factors(i), j->j<=16) and IsInt(i/64));
				Append(szs, List([11..16], i->2^i));
			fi;
		elif xform in [SGKEY_FFT_2D, SGKEY_IFFT_2D] then
			if type = SGKEY_DPCX then
				szs := List([4..8], i -> [2^i, 2^i]);
			else
				szs := List([6..9], i -> [2^i, 2^i]);
			fi;
		elif xform = SGKEY_WHT then
			if type = SGKEY_DPCX then
				szs := List([4..8], i->2^i);
			else
				szs := List([6..8], i->2^i);
			fi;
		fi;
				
		return szs;
	end,
	
	
	_validTypes := meth(arg)
		return [SGKEY_SPCX, SGKEY_DPCX];
	end,
	
	
	_genScript := meth(self, runType)
		local scrstr, xform, type, szs, tempvar, tempopts, optrec, optstr, isa, funcstr,
			use_cx, vec2;
		
		tempvar  := "tdp";
		xform	 := self.getSettingsValue(SGKEY_TRANSFORM);
		type	 := self.getSettingsValue(SGKEY_DATATYPE);
		szs      := self.getSettingsValue(SGKEY_SIZE);
		scrstr	 := "";
		
		tempopts := tempvar::"Opts";
		
		if (type = SGKEY_SPCX) then
			isa    := "AVX_8x32f";
			vec2   := 64;
		else
			isa    := "AVX_4x64f";
			vec2   := 16;
		fi;
		
		use_cx := ForAny(szs, s -> not IsInt(s / vec2));
		
		if xform in [SGKEY_FFT, SGKEY_IFFT] then
			optrec := rec(
				globalUnrolling    := 128,
				tsplRader          := false, 
				tsplBluestein      := false, 
				tsplPFA            := false, 
				oddSizes           := false, 
				interleavedComplex := true,
				cplxVect           := use_cx,
				realVect           := not use_cx,
			);
			if use_cx then
				optrec.RDFT  := false;
            	optrec.URDFT := true;
				if (type = SGKEY_SPCX) then
					optrec.CT      := false;
					optrec.PD      := false;
					optrec.svct    := true;
					optrec.flipIxA := true;
				elif (type = SGKEY_DPCX) then
			        optrec.svct   := false;
					optrec.splitL := true;
				fi;
			fi;
			if xform = SGKEY_IFFT then
				optrec.transInverse := true;
			fi;
			funcstr := "doSimdDft("::String(szs)::", "::isa::", "::tempopts::")";
		elif xform in [SGKEY_FFT_2D, SGKEY_IFFT_2D] then
			optrec := rec(
				interleavedComplex := true,
                oddSizes := false,
				svct := true, 
				splitL := false, 
				pushTag := true, 
				flipIxA := false, 
				stdTTensor := true, 
				tsplPFA := false
			);
			if xform = SGKEY_IFFT_2D then
				optrec.transInverse := true;
			fi;
			funcstr := "doSimdMddft("::String(szs)::", "::isa::", "::tempopts::")";
		elif xform = SGKEY_WHT then
			optrec := rec(
				verify   := true, 
				oddSizes := false, 
				svct     := true
			);
			funcstr := "doSimdWht("::String(szs)::", "::isa::", "::tempopts::")";
		else
			return "";
		fi;
		
		optrec.faultTolerant := true;
		
		if IsBound(self._settings.(SGKEY_FUNCNAME)) then
			optrec.functionNameRoot := self._settings.(SGKEY_FUNCNAME);
		fi;
		
		optstr := StringPrint(optrec);
		scrstr := tempopts::" := "::optstr::";;\n";
		Append(scrstr, tempvar::" := "::funcstr::";;\n");
		Append(scrstr, tempvar::"."::runType::"();\n");
		
		return scrstr;
	end,
	
	_localSupport := function()
		return (IsBound(LocalConfig) and IsBound(LocalConfig.cpuinfo) and IsBound(LocalConfig.cpuinfo.SIMD().hasAVX) and
			LocalConfig.cpuinfo.SIMD().hasAVX());
	end,
	
)); # Class ScriptGenAVX


SetScriptGenConstructor(ScriptGenAVX);

