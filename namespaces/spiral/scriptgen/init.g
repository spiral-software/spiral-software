
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

# Package scriptgen
# Generate Spiral scripts

Include(keys);
Include(scriptgen);

Include(avx);
Include(sse);
Include(scalar);

#
# Return list of names of all registered platforms
#
Global.ScriptGenPlatforms := () -> Sort(UserRecFields(scriptgen._constructors));

#
# Return list of names of registered platforms that local machine can support
#
Global.ScriptGenLocalPlatforms := () -> Filtered(ScriptGenPlatforms(), s -> scriptgen._constructors.(s)._localSupport());


Global.NewScriptGen := function(arg)
	local arch, opts;
	
	if IsBound(arg[1]) and IsString(arg[1]) then
		arch := arg[1];
	fi;
	
	# TEMP default
	if not IsBound(arch) then
		arch := "SSE";
	fi;
		
	if IsBound(arg[2]) and IsRec(arg[2]) then
		opts := arg[2];
	else
		opts := rec();
	fi;
	
	if IsBound(scriptgen._constructors.(arch)) then
		return scriptgen._constructors.(arch)(opts);
	else
		Print("\nusage: NewScriptGen(arch, opts = rec())\n    where arch is one of:\n    ", ScriptGenPlatforms(), "\n");
		return false;
	fi;
end;


#F DumpScriptGenScripts(ISA, <run type>)
#F
#F Generate a set of scripts for all valid options for the specified ISA
#F 
#F Get list of valid ISAs from ScriptGenPlatforms()
#F Default runType is first in ISA's ScriptGen.getScriptChoices()

Global.DumpScriptGenScripts := function(arg)
	local runType, isa, sg, transform, type, size, sizestr, listfile, scriptfile;
	
	if not Length(arg) in [1..2] then
		Error("usage: DumpScriptGenScripts(ISA, <runType>)");
	fi;
	
	isa := arg[1];
	if not isa in ScriptGenPlatforms() then
		Error("ISA must be one of: ", ScriptGenPlatforms());
	fi;
	
	sg := NewScriptGen(isa);
	
	runType := Cond(Length(arg) = 2, arg[2], Concat(sg.getScriptChoices(), [""])[1]);
	if not runType in sg.getScriptChoices() then
		Error("run type must be one of: ", sg.getScriptChoices());
	fi;

	
	listfile := "ListOf"::isa::"ScriptGenScripts.txt";
	PrintTo(listfile, "");
		
	for transform in sg.getValidValues(SGKEY_TRANSFORM) do
		sg.setSettingsValue(SGKEY_TRANSFORM, transform);
		for type in sg.getValidValues(SGKEY_DATATYPE) do
			sg.setSettingsValue(SGKEY_DATATYPE, type);
			for size in sg.getValidValues(SGKEY_SIZE) do
				sg.setSettingsValue(SGKEY_SIZE, size);
				if IsList(size) and Length(size) >= 2 then
					sizestr :=	Concat(String(size[1]), "x", String(size[2]));
				else
					sizestr := String(size);
				fi;
				scriptfile := isa::"_"::transform::"_"::type::"_"::sizestr::".g";
				AppendTo(listfile, scriptfile, "\n");
				sg.setSettingsValue(SGKEY_FILENAME, scriptfile);
				sg.writeScript(runType);
			od; # size
		od; # type
	od; # transform
end;


#F DumpAllScriptGenChoicesAsJSON()
#F
#F Writes a JSON file of all ScriptGen choices for each registered ISA

Global.DumpAllScriptGenChoicesAsJSON := function()
	local isa, file, sg;
	
	for isa in ScriptGenPlatforms() do
		file := Concat(isa, ".json");
		sg := NewScriptGen(isa);
		sg.writeJSONChoices(file);
	od;
end;


 