
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details



LocalConfig.bench := rec();

if IsBound(platforms.sse) then         LocalConfig.bench.SSE     := platforms.sse.benchSSE; fi;
if IsBound(platforms.intel) then       LocalConfig.bench.SMP_SSE := platforms.intel.benchCore2; fi;
if IsBound(platforms.benchScalar) then LocalConfig.bench.scalar  := platforms.benchScalar; fi;
if IsBound(platforms.avx) then         LocalConfig.bench.AVX     := platforms.avx.benchAVX; fi;

_dumpBench := function(obj, objPath, masterList, level)
	local newObj, newObjPath, fieldNames, name, size;
	
	if IsClass(obj) then
		if obj.__name__ = "DPBench" then
			if (level > 0) and IsBound(obj.sizes) and EndsWith(objPath, ")") then
				newObjPath := DropLast(Copy(objPath), 1);
				for size in obj.sizes do
					Append(masterList, [newObjPath::"["::String(size)::"])"]);
				od;
			else
				Append(masterList, [Copy(objPath)]);
			fi;
		fi;
		return;
	elif IsRec(obj) then
		fieldNames := UserRecFields(obj);
		for name in fieldNames do
			newObj := obj.(name);
			newObjPath := Copy(objPath)::"."::Copy(name);
			_dumpBench(newObj, newObjPath, masterList, level);
		od;
	elif IsFunc(obj) then
		newObj := obj();
		newObjPath := Copy(objPath)::"()";
		_dumpBench(newObj, newObjPath, masterList, level);
	fi;
end;

#F
#F DumpBenches(platform, level)
#F     platform : string, name of platform, eg., "SSE"
#F     level    : integer, level of detail
#F                0 : build benches that test a list of sizes
#F                1 : build an individual bench for each size
#F
#F Returns a list of all bench constructors for the specified platform
#F

DumpBenches := function(platform, level)
	local benchlist;
	
	benchlist := [];
	
	if IsBound(LocalConfig.bench.(platform)) and IsFunc(LocalConfig.bench.(platform)) then
		_dumpBench(LocalConfig.bench.(platform), "LocalConfig.bench."::platform, benchlist, level);
	fi;

	return Copy(benchlist);
end;


#F
#F DumpAllBenches(level = 0)
#F     level    : integer, optional (see DumpBenches)
#F
#F Returns a list of bench constructors for all supported platforms.
#F

DumpAllBenches := function(arg)
	local level, platform, platforms, benchlist;
	
	level := Cond(IsBound(arg[1]) and IsInt(arg[1]), arg[1], 0);

	##platforms := UserRecFields(LocalConfig.bench);
	platforms := ["SSE", "AVX"];
	
	benchlist := [];
	
	for platform in platforms do
		Append(benchlist, DumpBenches(platform, level));
	od;
	
	return Copy(benchlist);
end;

