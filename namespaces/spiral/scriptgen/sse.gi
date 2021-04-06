
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details



Class(ScriptGenSSE, ScriptGenBase, rec(

	_arch := () -> "SSE",


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
			szs := Filtered([1..512], i->ForAll(Factors(i), j->j<=16) and IsInt(i/4));
			Append(szs, Filtered([513..1024], i->ForAll(Factors(i), j->j<=16) and IsInt(i/16)));
			Append(szs, List([11..16], i->2^i));
		elif xform in [SGKEY_FFT_2D, SGKEY_IFFT_2D] then
			szs := List(Filtered([1..360], i->ForAll(Factors(i), j->j<=16) and IsInt(i/4)), n -> [n, n]);
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


	_genScript := meth(self, runType)
		local scrstr, xform, type, szs, tempvar, tempopts, optrec, optstr, isa, funcstr,
			use_cx;
		
		tempvar  := "tdp";
		xform	 := self.getSettingsValue(SGKEY_TRANSFORM);
		type	 := self.getSettingsValue(SGKEY_DATATYPE);
		szs      := self.getSettingsValue(SGKEY_SIZE);
		scrstr	 := "";
		
		tempopts := tempvar::"Opts";
		
		if (type = SGKEY_SPCX) then
			isa    := "SSE_4x32f";
			use_cx := ForAny(szs, s -> not IsInt(s / 16));
		else
			isa    := "SSE_2x64f";
			use_cx := false;
		fi;
		
		if xform in [SGKEY_FFT, SGKEY_IFFT] then
			optrec := rec (
				tsplRader:=false, 
				tsplBluestein:=false, 
				tsplPFA:=false, 
				oddSizes:=false, 
				interleavedComplex := true,
			);
			if use_cx then
				if (type = SGKEY_SPCX) then
					optrec.cplxVect := true;
					optrec.realVect := false;
					optrec.PRDFT := false;
					optrec.URDFT := true;
				fi;
			fi;			
			if xform = SGKEY_IFFT then
				optrec.transInverse := true;
			fi;
			funcstr := "doSimdDft("::String(szs)::", "::isa::", "::tempopts::")";
		elif xform in [SGKEY_FFT_2D, SGKEY_IFFT_2D] then
			optrec := rec (
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
			optrec := rec (
				svct := true, 
				stdTTensor := true, 
				tsplPFA := false
			);
			if (type = SGKEY_SPCX) then
				optrec.oddSizes := false;
			else
				optrec.verify := true;
				optrec.oddSizes := true;
			fi;
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

)); # Class ScriptGenSSE


SetScriptGenConstructor(ScriptGenSSE);

