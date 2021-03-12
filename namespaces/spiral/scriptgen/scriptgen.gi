
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details



#F ScriptGenBase()
#F

Class(ScriptGenBase, rec(


	_settings := rec(),

    ##
    ## Public methods
    ##
	

	
	getAllSettings := (self) >> Copy(self._settings),
	
	
	setAllSettings := meth(self, newrec)
		local field;
		
		if not IsRec(newrec) then
			return false; 
		fi;
		
		for field in RecFields(newrec) do
			self._settings.(field) := Copy(newrec.(field));
		od;
		
		#later validate new settings
		
		return true;
	end,
	
	
	getSettingsValue := meth(self, key)
		if not IsBound(self._settings.(key)) then
			return "";
		fi;
		return self._settings.(key);
	end,
	
	
	setSettingsValue := meth(arg)
		local self, key, value;
		if Length(arg) < 3 then return false; fi;
		
		self  := arg[1];
		key   := arg[2];
		value := arg[3];
		
		#quick out if already set
		if IsBound(self._settings.(key)) and (value = self._settings.(key)) then
			return true;
		fi;
		
		if key = SGKEY_TRANSFORM then
			return self._setTransform(value);
		elif key = SGKEY_SIZE then
			return self._setSize(value);
		elif (key = SGKEY_DATATYPE) then
			return self._setType(value);
		elif key = SGKEY_FILENAME then
			return self._setFilename(value);
		elif key = SGKEY_FUNCNAME then
			return self._setFuncname(value);
		fi;
		
		self._settings.(key) := value;
	
		return true;
	end,
	
	
	# TEMP simple set of sequences for early release, just works for FFT
	getSettingsSequence := meth(self)
		local seq;
		
		seq  := [SGKEY_TRANSFORM, SGKEY_DATATYPE, SGKEY_SIZE];
				
		return seq;
	end,
	
	
	getSettingsDetails := meth(self, key)
		local retrec;
		
		if not IsString(key) then
			return rec();
		fi;
			
		retrec := rec(
			(SGKEY_NAME)		:= key,
			(SGKEY_DISPLAYNAME)	:= self.getDisplayName(key),
		);
		
		if key = SGKEY_SIZE then
			if self.getSettingsValue(SGKEY_TRANSFORM) in [SGKEY_FFT_2D, SGKEY_IFFT_2D] then
				retrec.(SGKEY_TYPE) := [SGTYPE_INT, SGTYPE_INT];
			else
				retrec.(SGKEY_TYPE) := SGTYPE_INT;
			fi;
			retrec.(SGKEY_MULTIPLEVALUES) := true;
		elif key in [SGKEY_DATATYPE, SGKEY_FILENAME, SGKEY_TRANSFORM] then
			retrec.(SGKEY_TYPE) := SGTYPE_STRING;
		fi;
		
		return retrec;
	end,
	
	
	
	getValidValues := meth(self, key)
		if (key = SGKEY_TRANSFORM) then
			return Cond(IsBound(self._validTransforms), self._validTransforms(), []);
		elif (key = SGKEY_SIZE) then
			return Cond(IsBound(self._validSizes), self._validSizes(), []);
		elif (key = SGKEY_DATATYPE) then
			return Cond(IsBound(self._validTypes), self._validTypes(), []);
		else
			return [];
		fi;
	end,
	
	
	getDisplayName := key -> GetScriptGenDisplayName(key),
	

	setDisplayName := function(key, string)
		if not ( IsString(key) and IsString(string) ) then
			Error("usage: setDisplayName(key, string)\n  both <key> and <string> must be strings");
		fi;
		
		SetScriptGenDisplayName(key, string);
	end,
	
		
    getDocumentation := key -> GetScriptGenDocumentation(key),
	
		
	setDocumentation := function(key, string)
		if not ( IsString(key) and IsString(string) ) then
			Error("usage: setDocumentation(key, string)\n  both <key> and <string> must be strings");
		fi;
		
		SetScriptGenDocumentation(key, string);
	end,
	
	
	writeScript := meth(arg)
		local self, key, choiceKey, fname, myPrint, allKeys;
		
		self := arg[1];
		if Length(arg) > 1 then
			key := arg[2];
		else
			key := SGSTR_ALL;
		fi;
		
		fname   := self.getSettingsValue(SGKEY_FILENAME);
		
		if fname = SGSTR_STDOUT then
			myPrint := (arg) -> ApplyFunc(Print, arg);
		else
			# create file or overwrite existing, then use AppendTo for actual writing
			PrintTo(fname, "");
			myPrint := (arg) -> ApplyFunc(AppendTo, Concat([fname], arg));
		fi;
		
		if key in self.getScriptChoices() then
			myPrint(self._genScript(key));
		elif key = SGSTR_CONSTRUCTOR then
			myPrint("NewScriptGen(\""::self._arch()::"\", ");
			myPrint(self._settings);
			myPrint(");\n");
		elif key = SGSTR_ALL then
			myPrint("\n##! BEGIN CONSTRUCTOR\n\n");
			myPrint("NewScriptGen(\""::self._arch()::"\", ");
			myPrint(self._settings);
			myPrint(");\n");
			myPrint("\n##! END CONSTRUCTOR\n\n");
			
			for choiceKey in self.getScriptChoices() do
				myPrint("##! BEGIN SCRIPT "::choiceKey::"\n");
				myPrint("# "::self.getDisplayName(choiceKey)::"\n\n");
				myPrint(self._genScript(choiceKey));
				myPrint("\n##! END SCRIPT "::choiceKey::"\n\n");
			od;
		else
			allKeys := Concat([SGSTR_ALL, SGSTR_CONSTRUCTOR], self.getScriptChoices());
			Error("key must be one of: ", allKeys);
		fi;
	end,
	
	
	getScriptChoices := (self) >> [SGSTR_RUNRANDOMALL, SGSTR_RUNALL], 
	
	
	getAllChoicesAsJSON := meth(self)
		local json, indentStr, tree;
		
		indentStr := "    ";
		
		json := "{\n" :: indentStr :: "\"isa\" : \"" :: self._arch() :: "\"";
		
		tree := self._getChoicesDetailsJSON(1, indentStr);
		
		if Length(tree) > 0 then
			json := json :: ",\n" :: tree;
		fi;
		
		json := json :: "\n}\n";
	
		return json;
	end,
	
	
	writeJSONChoices := meth(arg)
		local self, file, json;
		
		self := arg[1];
		if Length(arg) > 1 then
			file := arg[2];
		else
			file := SGSTR_STDOUT;
		fi;
		
		json := self.getAllChoicesAsJSON();
		
		PrintTo(file, json);
	end,


	
    ##
    ## Private methods
	##
	
	
	__call__ := meth(arg)
		local self, me, field;
		self := arg[1];
		
		# create a new instance of the class
		
		me := WithBases(self, rec(_settings := rec()));
		
		me._init();
		
		if IsBound(arg[2]) and IsRec(arg[2]) then
			for field in RecFields(arg[2]) do
				me._settings.(field) := Copy(arg[2].(field));
			od;
		fi;
		
		return me;
	end,
	

	# subclass must implement _init()
	_init := (self) >> Error("Cannot instantiate abstract class\n"),
	
	
	_setTransform := meth(self, tr)
		local szs;
		if not tr in self._validTransforms() then
			return false;
		fi;
		self._settings.(SGKEY_TRANSFORM) := tr;
		
		# make sure size is still valid
		self._validateAndFixSize();
		
		return true;
	end,
	
	
	_setType := meth(self, type)
		local szs;
		if not type in self._validTypes() then
			return false;
		fi;
		self._settings.(SGKEY_DATATYPE) := type;
		
		# make sure size is still valid
		self._validateAndFixSize();
	
		return true;
	end,
	
	
	_setSize := meth(arg)
		local self, sz, szlist, idx, xform;
		
		if Length(arg) < 2 then return false; fi;
		
		self := arg[1];
		sz   := arg[2];
		
		# 1D is a list of one or more valid ints
		# 2D is a list of one or more valid pairs of ints
		
		xform := self.getSettingsValue(SGKEY_TRANSFORM);
		if xform in [SGKEY_FFT_2D, SGKEY_IFFT_2D] then
			if (not IsList(sz)) or (Length(sz) < 1) then
				return false;
			fi;
			if IsInt(sz[1]) then
				szlist := [ sz ];
			else
				szlist := sz;
			fi;
		else
			if IsList(sz) and (Length(sz) > 0) then
				szlist := sz;
			else
				szlist := [ sz ];
			fi;
		fi;
				
		if szlist <> self._validateSize(szlist) then
			return false;
		fi;

		self._settings.(SGKEY_SIZE) := szlist;
		return true;
	end,
	
	
	_setFilename := meth(self, fname)
		if not IsString(fname) then
			return false;
		fi;
		if Length(fname) = 0 then
			self._settings.(SGKEY_FILENAME) := SGSTR_STDOUT;
		else
			self._settings.(SGKEY_FILENAME) := fname;
		fi;
		return true;
	end,
	
	
	_setFuncname := meth(self, fname)
		if not IsString(fname) then
			return false;
		fi;
		
		### NOTE don't allow whitespace in string
		
		if Length(fname) = 0 then
			Unbind(self._settings.(SGKEY_FUNCNAME));
		else
			self._settings.(SGKEY_FUNCNAME) := fname;
		fi;
		return true;
	end,
	
	
	_validateSize := meth(self, szlist)
		local goodsizes;
		goodsizes := self._validSizes();
		if (Length(szlist) > 0) and ForAll(szlist, s -> s in goodsizes) then
			return szlist;
		fi;
	
		return Cond(Length(goodsizes) > 0, [ goodsizes[1] ], []);
	end,
	
	
	_validateAndFixSize := meth(self)
		local oldsz, newsz;
		oldsz := Cond(IsBound(self._settings.(SGKEY_SIZE)), self._settings.(SGKEY_SIZE), [ 0 ]) ;
		newsz := self._validateSize(oldsz);
		if newsz <> oldsz then
			self._settings.(SGKEY_SIZE) := newsz;
		fi;
	end,
		
	
	_getChoicesDetailsJSON := meth(arg)
		local self, level, indentStr, id, descr, json, indent, i, sequence, key, choices, value;
		
		if Length(arg) < 3 then
			return "";
		fi;
		self := arg[1];
		level := arg[2];
		indentStr := arg[3];
		id    := Cond(IsBound(arg[4]) and IsString(arg[4]), arg[4], "");
		descr := self.getDisplayName(id);
		
		indent := "";
		for i in [2..level] do
			indent := indent :: indentStr :: indentStr;
		od;
		
		sequence := self.getSettingsSequence();
		if level > Length(sequence) then return ""; fi;
		key := sequence[level];
		
		if level > 1 then
			json := indent :: indentStr :: "\"ident\" : \"" :: id :: "\",\n";
			json := json :: indent :: indentStr :: "\"descr\" : \"" :: descr :: "\",\n";
		else
			json := "";
		fi;
		
		json := json :: indent :: indentStr :: "\"choice_name\" : \"" :: key :: "\",\n";
		json := json :: indent :: indentStr :: "\"choice_display\" : \"" :: self.getDisplayName(key) :: "\",\n";
		
		choices := self.getValidValues(key);
		
		if level >= Length(sequence) then
			return json :: indent :: indentStr :: "\"choices\" : " :: String(choices);
		fi;
		
		json := json :: indent :: indentStr :: "\"choices\" : [\n";
		
		for i in [1..Length(choices)] do
			json := json :: indent :: indentStr :: indentStr :: "{\n";
			
			value := choices[i];
			self.setSettingsValue(key, value);
			json := json :: self._getChoicesDetailsJSON(level+1, indentStr, value) :: "\n";
		
			json := json :: indent :: indentStr :: indentStr :: Cond(i < Length(choices), "},\n", "}\n");
		od;
			
		json := json :: indent :: indentStr :: "]";
	
		return json;
	end,
	
	
	_genScript := (self, runType) >> "",
	
	
	_localSupport := () -> true,
	
	
)); # Class ScriptGen


_constructors := rec();


SetScriptGenConstructor := function(gen)
	if IsCallable(gen) and IsBound(gen._arch) then
		_constructors.(gen._arch()) := gen;
	else
		Error("Invalid argument\n");
	fi;
end;




