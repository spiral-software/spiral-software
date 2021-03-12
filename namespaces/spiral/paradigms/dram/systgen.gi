
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


ImportAll(paradigms.stream);

# gets/replaces the module names, copies generated files into genpath
setModule := function(filename, searchString, moduleName, genpath, bb)	
	local cmdString;
	# replace the module name
	cmdString := ConcatenationString(paradigms.stream._hardwarePath, "dram_scripts/_getModuleName.sh ", filename," ",searchString," ", moduleName);
	Exec(cmdString);
	if(bb = 1) then
		cmdString := ConcatenationString(paradigms.stream._hardwarePath, "dram_scripts/_blackBoxifyMems.sh ", filename);
		Exec(cmdString);
	fi;
	# Copy the file into genpath
	cmdString := ConcatenationString("cp ",filename," ",genpath);
	Exec(cmdString);
	return moduleName;
end;

# gets the latency of permMems, writes them into a tmp file
getLMLatency := function(filename, RD_WR, memInst, tmpFile )
	local cmdString;
	# Puts lm latency define statement into tmpFile
	cmdString := ConcatenationString(paradigms.stream._hardwarePath, "dram_scripts/_permMemLatencies.sh ",filename," ",RD_WR," ",String(memInst)," ",tmpFile);
	Exec(cmdString);
end;

# prints the latencies into config file, determines the max latency etc., then deletes the tmp file
putLMLatency := function(tmpFile, readFile)
	local cmdString;
	cmdString := ConcatenationString(paradigms.stream._hardwarePath, "dram_scripts/_putPermMemLatencies.sh ",tmpFile," ",readFile);
	Exec(cmdString);
	Read(readFile);
	Exec(ConcatenationString("rm ", readFile));
	Exec(ConcatenationString("rm ", tmpFile));
end;

# adds prefix to module names to avoid overwrites
prefixModules := function(filename, prefix)
	local cmdString;
	cmdString := ConcatenationString(paradigms.stream._hardwarePath, "dram_scripts/_prefixModName.sh ",filename," ",prefix);
	Exec(cmdString);
end;

#genConfigFile := function(srt, prec, genpath)
genConfigFile3 := function(srt, opts, genpath)
	local 	stages, fft_size, i,j,diff_prms,a,b,c,d,e,stream,path,format,cmdString,module,locmodname,bb,memfences,prmObjs,prmObj,
			cube_width,
			mem_wr_prms,                    
			mem_rd_prms,                    
			loc_wr_prms,                    
			loc_rd_prms,
			sym_mem_wr,
			sym_mem_rd,
			sym_loc_wr,
			sym_loc_rd;
	
	stages := Length(Collect(srt, MemFence));
	fft_size := Collect(srt, DFT)[1].params[1];
	memfences := Collect(srt, MemFence);
	
	mem_wr_prms := [];
	mem_rd_prms := [];
	loc_wr_prms := [];
	loc_rd_prms := [];
	
	for i in [1..stages] do
		mem_wr_prms[i] := Collect(memfences[i], MemWrPrm);
		mem_rd_prms[i] := Collect(memfences[i], MemRdPrm);
		loc_wr_prms[i] := Collect(memfences[i], LocalWrPrm);
		loc_rd_prms[i] := Collect(memfences[i], LocalRdPrm);
	od;
	
	cube_width := mem_rd_prms[1][1].func._children[Length(mem_rd_prms[1][1].func._children)].params[1];
	stream := opts.dram_datawidth/opts.precision/2;
	format := When (opts.precision = 64, 2, 1);
	bb := opts.bb;
	
	
	PrintLine("//=========================");
	PrintLine("// DO NOT MODIFY THIS FILE!");
	PrintLine("//=========================\n");
	
	# Print the define statements into config
	PrintLine("`define CONFIG_FILE");
	
	# Streamig width
	if(stream >= 4) then
		PrintLine("`define SW_4");
	fi;
	if(stream >= 8) then
		PrintLine("`define SW_8");
	fi;
	if(stream = 16) then
		PrintLine("`define SW_16");
	fi;
	if(stream > 16 or stream < 2) then
		Error("\n***ERROR: Streaming width = ",stream," is not supported for now!\n");
	fi;

	PrintLine("// Asymmetric algorithm...");
	diff_prms := stages;
	PrintLine("`define ASYMMETRIC_ALGO");

	

	PrintLine("`define APPDATA_WIDTH ", opts.dram_datawidth);
	PrintLine("`define DDR_ADDR_WIDTH ", opts.dram_addrwidth);
	PrintLine("`define LOG_FFT_SIZE ", LogInt(fft_size,2));
	PrintLine("`define PACKET_SIZE ", cube_width/stream);
	PrintLine("`define PRECISION ", opts.precision);
	PrintLine("`define NUM_OF_STAGES ", stages);
	
	
	PrintLine("// ODCM parameters");
	PrintLine("//");
	PrintLine("// MemWrPrm:");
	
	for i in [1..stages] do
		for j in [1..Length(mem_wr_prms[i])] do
			if(mem_wr_prms[i][j].func.numChildren() = 3) then # IxLxI
				a := mem_wr_prms[i][j].func._children[1].params[1];
				b := mem_wr_prms[i][j].func._children[2].params[1];
				c := mem_wr_prms[i][j].func._children[2].params[2];
				d := mem_wr_prms[i][j].func._children[3].params[1];
				PrintLine("// IxLxI: I(",a,")xL(",b,",",c,")xI(",d,") --> need to transpose for write address generation --> IxLxI: I(",a,")xL(",b,",",b/c,")xI(",d,")");
				PrintLine("`define MEM_WR_PERM_MODULE_NAME_",i,"_",Length(mem_wr_prms[i])-j+1," permIL #(.loga(",LogInt(a,2),"), .logb(",LogInt(b,2),"), .logc(",LogInt(b/c,2),"))");
			fi;
			if(mem_wr_prms[i][j].func.numChildren() = 2) then # LxI
				b := mem_wr_prms[i][j].func._children[1].params[1];
				c := mem_wr_prms[i][j].func._children[1].params[2];
				d := mem_wr_prms[i][j].func._children[2].params[1];
				PrintLine("// LxI: L(",b,",",c,")xI(",d,") --> need to transpose for write address generation --> LxI: L(",b,",",b/c,")xI(",d,")");
				PrintLine("`define MEM_WR_PERM_MODULE_NAME_",i,"_",Length(mem_wr_prms[i])-j+1," permL #(.logb(",LogInt(b,2),"), .logc(",LogInt(b/c,2),"))");				
			fi;
			if(mem_wr_prms[i][j].func.numChildren() = 4) then # IxLxIxI
				a := mem_wr_prms[i][j].func._children[1].params[1];
				b := mem_wr_prms[i][j].func._children[2].params[1];
				c := mem_wr_prms[i][j].func._children[2].params[2];
				d := mem_wr_prms[i][j].func._children[3].params[1];
				e := mem_wr_prms[i][j].func._children[4].params[1];
				PrintLine("// IxLxIxI: I(",a,")xL(",b,",",c,")xI(",d,")xI(",e,") --> need to transpose for write address generation --> IxLxIxI: I(",a,")xL(",b,",",b/c,")xI(",d,")xI(",e,")");
				PrintLine("`define MEM_WR_PERM_MODULE_NAME_",i,"_",Length(mem_wr_prms[i])-j+1," permILI #(.loga(",LogInt(a,2),"), .logb(",LogInt(b,2),"), .logc(",LogInt(b/c,2),"), .logd(",LogInt(d,2),"))");
			fi;
		od;
	od;
	
	PrintLine("//");
	PrintLine("// MemRdPrm:");

	for i in [1..stages] do
		for j in [1..Length(mem_rd_prms[i])] do
			if(mem_rd_prms[i][j].func.numChildren() = 3) then # IxLxI
				a := mem_rd_prms[i][j].func._children[1].params[1];
				b := mem_rd_prms[i][j].func._children[2].params[1];
				c := mem_rd_prms[i][j].func._children[2].params[2];
				d := mem_rd_prms[i][j].func._children[3].params[1];
				PrintLine("// IxLxI: I(",a,")xL(",b,",",c,")xI(",d,")");
				PrintLine("`define MEM_RD_PERM_MODULE_NAME_",i,"_",j," permIL #(.loga(",LogInt(a,2),"), .logb(",LogInt(b,2),"), .logc(",LogInt(c,2),"))");
			fi;
			if(mem_rd_prms[i][j].func.numChildren() = 2) then # LxI
				b := mem_rd_prms[i][j].func._children[1].params[1];
				c := mem_rd_prms[i][j].func._children[1].params[2];
				d := mem_rd_prms[i][j].func._children[2].params[1];
				PrintLine("// LxI: L(",b,",",c,")xI(",d,")");
				PrintLine("`define MEM_RD_PERM_MODULE_NAME_",i,"_",j," permL #(.logb(",LogInt(b,2),"), .logc(",LogInt(c,2),"))");
			fi;
			if(mem_rd_prms[i][j].func.numChildren() = 4) then # IxLxIxI
				a := mem_rd_prms[i][j].func._children[1].params[1];
				b := mem_rd_prms[i][j].func._children[2].params[1];
				c := mem_rd_prms[i][j].func._children[2].params[2];
				d := mem_rd_prms[i][j].func._children[3].params[1];
				e := mem_rd_prms[i][j].func._children[4].params[1];
				PrintLine("// IxLxIxI: I(",a,")xL(",b,",",c,")xI(",d,")xI(",e,")");
				PrintLine("`define MEM_RD_PERM_MODULE_NAME_",i,"_",j," permILI #(.loga(",LogInt(a,2),"), .logb(",LogInt(b,2),"), .logc(",LogInt(c,2),"), .logd(",LogInt(d,2),"))");
			fi;
		od;
	od;


	PrintLine("// LM parameters");
	PrintLine("//");
	PrintLine("// LocalWrPrm:");
	for i in [1..stages] do
		prmObjs := [];
		locmodname := ConcatenationString("LocWrPrm_",String(i));
		PrintLine("/*** ");
		for j in [1..Length(loc_wr_prms[i])] do
			if(loc_wr_prms[i][j].func.numChildren() = 3) then # IxLxI
				a := loc_wr_prms[i][j].func._children[1].params[1];
				b := loc_wr_prms[i][j].func._children[2].params[1];
				c := loc_wr_prms[i][j].func._children[2].params[2];
				d := loc_wr_prms[i][j].func._children[3].params[1];
				PrintLine("// IxLxI: I(",a,")xL(",b,",",c,")xI(",d,")");
				prmObjs[j] := TL(b,c,a,d);
				#path := genBRAMPermMem(TRC(TPrm(TL(b,c,a,d))), stream*2, format, opts.precision, locmodname); 
			fi;
			if(loc_wr_prms[i][j].func.numChildren() = 2) then
				if(Length(loc_wr_prms[i][j].func._children[1].params) = 2) then # LxI
					b := loc_wr_prms[i][j].func._children[1].params[1];
					c := loc_wr_prms[i][j].func._children[1].params[2];
					d := loc_wr_prms[i][j].func._children[2].params[1];
					PrintLine("// LxI: L(",b,",",c,")xI(",d,")");
					prmObjs[j] := TL(b,c,1,d);
					#path := genBRAMPermMem(TRC(TPrm(TL(b,c,1,d))), stream*2, format, opts.precision, locmodname); 
				fi;
				if(Length(loc_wr_prms[i][j].func._children[1].params) = 1) then # IxL
					a := loc_wr_prms[i][j].func._children[1].params[1];
					b := loc_wr_prms[i][j].func._children[2].params[1];
					c := loc_wr_prms[i][j].func._children[2].params[2];
					PrintLine("// IxL: I(",a,")xL(",b,",",c,")");
					prmObjs[j] := TL(b,c,a,1);
					#path := genBRAMPermMem(TRC(TPrm(TL(b,c,a,1))), stream*2, format, opts.precision, locmodname); 
				fi;
			fi;
			if(loc_wr_prms[i][j].func.numChildren() = 0) then # L
				b := loc_wr_prms[i][j].func.params[1];
				c := loc_wr_prms[i][j].func.params[2];
				PrintLine("// L: L(",b,",",c,")");
				prmObjs[j] := TL(b,c,1,1);
				#path := genBRAMPermMem(TRC(TPrm(TL(b,c,1,1))), stream*2, format, opts.precision, locmodname); 
			fi;
		od;
		
		prmObj := prmObjs[1];
		for j in [2..Length(loc_wr_prms[i])] do
			prmObj := prmObj * prmObjs[j];
		od;
		path := genBRAMPermMem(TRC(TPrm(prmObj)), stream*2, format, opts.precision, locmodname); 
		
		prefixModules(ConcatenationString(path, locmodname,".v"), ConcatenationString("lw",String(i)));
		module := setModule(ConcatenationString(path, locmodname,".v"), "module_name_is", ConcatenationString("perm_mem_locwr_",String(i)), genpath, bb);
		PrintLine("***/ ");
		PrintLine("`define LOC_WR_MODULE_NAME_",i," ",module);
		getLMLatency(ConcatenationString(path, locmodname,".v"), "WR", i, "/tmp/_spiral.tmp");
	od;

	#genBRAMPermMem(perm, w, format, bits, name)
	#genBRAMPermMem(TRC(TPrm(TL(32,2,1,1))), 4, 2, 16, "LocRdPrm");
	PrintLine("//");
	PrintLine("// LocalRdPrm:");
	for i in [1..stages] do
		prmObjs := [];
		locmodname := ConcatenationString("LocRdPrm_",String(i));
		PrintLine("/*** ");
		for j in [1..Length(loc_rd_prms[i])] do
			if(loc_rd_prms[i][j].func.numChildren() = 3) then # IxLxI
				a := loc_rd_prms[i][j].func._children[1].params[1];
				b := loc_rd_prms[i][j].func._children[2].params[1];
				c := loc_rd_prms[i][j].func._children[2].params[2];
				d := loc_rd_prms[i][j].func._children[3].params[1];
				PrintLine("// IxLxI: I(",a,")xL(",b,",",c,")xI(",d,")");
				prmObjs[j] := TL(b,c,a,d);
				#path := genBRAMPermMem(TRC(TPrm(TL(b,c,a,d))), stream*2, format, opts.precision, locmodname); 
			fi;
			if(loc_rd_prms[i][j].func.numChildren() = 2) then
				if(Length(loc_rd_prms[i][j].func._children[1].params) = 2) then # LxI
					b := loc_rd_prms[i][j].func._children[1].params[1];
					c := loc_rd_prms[i][j].func._children[1].params[2];
					d := loc_rd_prms[i][j].func._children[2].params[1];
					PrintLine("// LxI: L(",b,",",c,")xI(",d,")");
					prmObjs[j] := TL(b,c,1,d);
					#path := genBRAMPermMem(TRC(TPrm(TL(b,c,1,d))), stream*2, format, opts.precision, locmodname); 
				fi;
				if(Length(loc_rd_prms[i][j].func._children[1].params) = 1) then # IxL
					a := loc_rd_prms[i][j].func._children[1].params[1];
					b := loc_rd_prms[i][j].func._children[2].params[1];
					c := loc_rd_prms[i][j].func._children[2].params[2];
					PrintLine("// IxL: I(",a,")xL(",b,",",c,")");
					prmObjs[j] := TL(b,c,a,1);
					#path := genBRAMPermMem(TRC(TPrm(TL(b,c,a,1))), stream*2, format, opts.precision, locmodname); 
				fi;
			fi;
			if(loc_rd_prms[i][j].func.numChildren() = 0) then # L
				b := loc_rd_prms[i][j].func.params[1];
				c := loc_rd_prms[i][j].func.params[2];
				PrintLine("// L: L(",b,",",c,")");
				prmObjs[j] := TL(b,c,1,1);
				#path := genBRAMPermMem(TRC(TPrm(TL(b,c,1,1))), stream*2, format, opts.precision, locmodname); 
			fi;
		od;
		
		prmObj := prmObjs[1];
		for j in [2..Length(loc_rd_prms[i])] do
			prmObj := prmObj * prmObjs[j];
		od;
		path := genBRAMPermMem(TRC(TPrm(prmObj)), stream*2, format, opts.precision, locmodname); 
		
		prefixModules(ConcatenationString(path, locmodname,".v"), ConcatenationString("lr",String(i)));
		module := setModule(ConcatenationString(path, locmodname,".v"), "module_name_is", ConcatenationString("perm_mem_locrd_",String(i)), genpath, bb);
		PrintLine("***/ ");
		PrintLine("`define LOC_RD_MODULE_NAME_",i," ",module);
		getLMLatency(ConcatenationString(path, locmodname,".v"), "RD", i, "/tmp/_spiral.tmp");
	od;
	
	PrintLine("// LM Additive latency ");
	putLMLatency("/tmp/_spiral.tmp", "/tmp/_spiral_readFile.tmp");
	
	PrintLine("/*** ");
	path := HDLGen(streamDFTUnroll(fft_size,2,stream*2), 1, format, 0, 0, 0, "dftgen");
	prefixModules(ConcatenationString(path, "dftgen.v"), "f");
	module := setModule(ConcatenationString(path, "dftgen.v"), "module_name_is", "dftcore", genpath, bb);
	PrintLine("***/ ");
	
	PrintLine("//");
	PrintLine("// FFT Core");
	PrintLine("`define FFT_CORE_MODULE_NAME ", module);
	
	if(opts.throttle = 1) then
		PrintLine("`define THROTTLE_EN");
	fi;
	
	PrintLine("// Summary");
	PrintLine("//");
	PrintLine("// Cube Width:",cube_width/stream," accesses");
	PrintLine("// Cube Width:",cube_width," words");
	PrintLine("// Stages:",stages);
	PrintLine("// Streaming Width:",opts.dram_datawidth/opts.precision/2);
	PrintLine("// DFT Size:",fft_size);	
	PrintLine("// Format: ",format," (2:double, 1:single)");	

end;

#genConfigFile := function(srt, prec, genpath)
genConfigFile2 := function(srt, opts, genpath)
	local 	stages, fft_size, i,j,diff_prms,a,b,c,d,stream,path,format,cmdString,module,locmodname,bb,twiddles,twiddle,
			mem_wr_prms,                    
			mem_rd_prms,                    
			loc_wr_prms,                    
			loc_rd_prms,
			tile_width,
			sym_mem_wr,
			sym_mem_rd,
			sym_loc_wr,
			sym_loc_rd;
	
	stages := Length(Collect(srt, MemFence));
	fft_size := Collect(srt, DFT)[1].params[1];
	mem_wr_prms := Collect(srt, MemWrPrm);
	mem_rd_prms := Collect(srt, MemRdPrm);
	loc_wr_prms := Collect(srt, LocalWrPrm);
	loc_rd_prms := Collect(srt, LocalRdPrm);
	twiddles := Collect(srt, TwiddleROM);
	tile_width := mem_rd_prms[1].func._children[Length(mem_rd_prms[1].func._children)].params[1];
	stream := opts.dram_datawidth/opts.precision/2;
	format := When (opts.precision = 64, 2, 1);
	bb := opts.bb;
	
	sym_mem_wr := true;
	sym_mem_rd := true;
	sym_loc_wr := true;
	sym_loc_rd := true;
	
	for i in [1..stages] do
		for j in [1..stages] do
			sym_mem_wr := sym_mem_wr and (mem_wr_prms[i] = mem_wr_prms[j]);
			sym_mem_rd := sym_mem_rd and (mem_rd_prms[i] = mem_rd_prms[j]);
			sym_loc_wr := sym_loc_wr and (loc_wr_prms[i] = loc_wr_prms[j]);
			sym_loc_rd := sym_loc_rd and (loc_rd_prms[i] = loc_rd_prms[j]);
		od;
	od;
	
	PrintLine("//=========================");
	PrintLine("// DO NOT MODIFY THIS FILE!");
	PrintLine("//=========================\n");
	
	# Print the define statements into config
	PrintLine("`define CONFIG_FILE");
	
	# Streamig width
	if(stream >= 4) then
		PrintLine("`define SW_4");
	fi;
	if(stream >= 8) then
		PrintLine("`define SW_8");
	fi;
	if(stream = 16) then
		PrintLine("`define SW_16");
	fi;
	if(stream > 16 or stream < 2) then
		Error("\n***ERROR: Streaming width = ",stream," is not supported for now!\n");
	fi;


	
	# determine symmetry of the algorithm
	if(sym_mem_wr and sym_mem_rd and sym_loc_wr and sym_loc_rd) then
		PrintLine("// All symmetric algorithm...");
		diff_prms := 1;
	else
		PrintLine("// Asymmetric algorithm...");
		diff_prms := stages;
		PrintLine("`define ASYMMETRIC_ALGO");
	fi;
	

	PrintLine("`define APPDATA_WIDTH ", opts.dram_datawidth);
	PrintLine("`define DDR_ADDR_WIDTH ", opts.dram_addrwidth);
	PrintLine("`define LOG_FFT_SIZE ", LogInt(fft_size,2));
	PrintLine("`define PACKET_SIZE ", tile_width/stream);
	PrintLine("`define PRECISION ", opts.precision);
	PrintLine("`define NUM_OF_STAGES ", stages);
	
	
	PrintLine("// ODCM parameters");
	PrintLine("//");
	PrintLine("// MemWrPrm:");
	for i in [1..diff_prms] do
		if(mem_wr_prms[i].func.numChildren() = 3) then # IxLxI
			a := mem_wr_prms[i].func._children[1].params[1];
			b := mem_wr_prms[i].func._children[2].params[1];
			c := mem_wr_prms[i].func._children[2].params[2];
			d := mem_wr_prms[i].func._children[3].params[1];
			PrintLine("// IxLxI: I(",a,")xL(",b,",",c,")xI(",d,") --> need to transpose for write address generation --> IxLxI: I(",a,")xL(",b,",",b/c,")xI(",d,")");
			PrintLine("`define MEM_WR_PERM_MODULE_NAME_",i," permIL #(.loga(",LogInt(a,2),"), .logb(",LogInt(b,2),"), .logc(",LogInt(b/c,2),"))");
			#PrintLine("`define MEM_WR_PERM_MODULE_NAME permIL #(.loga(",LogInt(a/stream,2),"), .logb(",LogInt(b/stream,2),"), .logc(",LogInt(c/stream,2),"))");
		fi;
		if(mem_wr_prms[i].func.numChildren() = 2) then # LxI
			b := mem_wr_prms[i].func._children[1].params[1];
			c := mem_wr_prms[i].func._children[1].params[2];
			d := mem_wr_prms[i].func._children[2].params[1];
			PrintLine("// LxI: L(",b,",",c,")xI(",d,") --> need to transpose for write address generation --> LxI: L(",b,",",b/c,")xI(",d,")");
			PrintLine("`define MEM_WR_PERM_MODULE_NAME_",i," permL #(.logb(",LogInt(b,2),"), .logc(",LogInt(b/c,2),"))");
			#PrintLine("`define MEM_WR_PERM_MODULE_NAME permL #(.logb(",LogInt(b/stream,2),"), .logc(",LogInt(c/stream,2),"))");
		fi;
	od;
	
	PrintLine("//");
	PrintLine("// MemRdPrm:");
	for i in [1..diff_prms] do
		if(mem_rd_prms[i].func.numChildren() = 3) then # IxLxI
			a := mem_rd_prms[i].func._children[1].params[1];
			b := mem_rd_prms[i].func._children[2].params[1];
			c := mem_rd_prms[i].func._children[2].params[2];
			d := mem_rd_prms[i].func._children[3].params[1];
			PrintLine("// IxLxI: I(",a,")xL(",b,",",c,")xI(",d,")");
			PrintLine("`define MEM_RD_PERM_MODULE_NAME_",i," permIL #(.loga(",LogInt(a,2),"), .logb(",LogInt(b,2),"), .logc(",LogInt(c,2),"))");
			#PrintLine("`define MEM_RD_PERM_MODULE_NAME permIL #(.loga(",LogInt(a/stream,2),"), .logb(",LogInt(b/stream,2),"), .logc(",LogInt(c/stream,2),"))");
		fi;
		if(mem_rd_prms[i].func.numChildren() = 2) then # LxI
			b := mem_rd_prms[i].func._children[1].params[1];
			c := mem_rd_prms[i].func._children[1].params[2];
			d := mem_rd_prms[i].func._children[2].params[1];
			PrintLine("// LxI: L(",b,",",c,")xI(",d,")");
			PrintLine("`define MEM_RD_PERM_MODULE_NAME_",i," permL #(.logb(",LogInt(b,2),"), .logc(",LogInt(c,2),"))");
			#PrintLine("`define MEM_RD_PERM_MODULE_NAME permL #(.logb(",LogInt(b/stream,2),"), .logc(",LogInt(c/stream,2),"))");
		fi;
	od;
	
	PrintLine("// LM parameters");
	PrintLine("//");
	PrintLine("// LocalWrPrm:");
	for i in [1..diff_prms] do
		locmodname := ConcatenationString("LocWrPrm_",String(i));
		PrintLine("/*** ");
		if(loc_wr_prms[i].func.numChildren() = 3) then # IxLxI
			a := loc_wr_prms[i].func._children[1].params[1];
			b := loc_wr_prms[i].func._children[2].params[1];
			c := loc_wr_prms[i].func._children[2].params[2];
			d := loc_wr_prms[i].func._children[3].params[1];
			PrintLine("// IxLxI: I(",a,")xL(",b,",",c,")xI(",d,")");
			path := genBRAMPermMem(TRC(TPrm(TL(b,c,a,d))), stream*2, format, opts.precision, locmodname); 
			
		fi;
		if(loc_wr_prms[i].func.numChildren() = 2) then
			if(Length(loc_wr_prms[i].func._children[1].params) = 2) then # LxI
				b := loc_wr_prms[i].func._children[1].params[1];
				c := loc_wr_prms[i].func._children[1].params[2];
				d := loc_wr_prms[i].func._children[2].params[1];
				PrintLine("// LxI: L(",b,",",c,")xI(",d,")");
				path := genBRAMPermMem(TRC(TPrm(TL(b,c,1,d))), stream*2, format, opts.precision, locmodname); 
			fi;
			if(Length(loc_wr_prms[i].func._children[1].params) = 1) then # IxL
				a := loc_wr_prms[i].func._children[1].params[1];
				b := loc_wr_prms[i].func._children[2].params[1];
				c := loc_wr_prms[i].func._children[2].params[2];
				PrintLine("// IxL: I(",a,")xL(",b,",",c,")");
				path := genBRAMPermMem(TRC(TPrm(TL(b,c,a,1))), stream*2, format, opts.precision, locmodname); 
			fi;
		fi;
		if(loc_wr_prms[i].func.numChildren() = 0) then # L
			b := loc_wr_prms[i].func.params[1];
			c := loc_wr_prms[i].func.params[2];
			PrintLine("// L: L(",b,",",c,")");
			path := genBRAMPermMem(TRC(TPrm(TL(b,c,1,1))), stream*2, format, opts.precision, locmodname); 
		fi;
		
		prefixModules(ConcatenationString(path, locmodname,".v"), ConcatenationString("lw",String(i)));
		module := setModule(ConcatenationString(path, locmodname,".v"), "module_name_is", ConcatenationString("perm_mem_locwr_",String(i)), genpath, bb);
		PrintLine("***/ ");
		PrintLine("`define LOC_WR_MODULE_NAME_",i," ",module);
		getLMLatency(ConcatenationString(path, locmodname,".v"), "WR", i, "/tmp/_spiral.tmp");
	od;

		
	#genBRAMPermMem(perm, w, format, bits, name)
	#genBRAMPermMem(TRC(TPrm(TL(32,2,1,1))), 4, 2, 16, "LocRdPrm");
	PrintLine("//");
	PrintLine("// LocalRdPrm:");
	for i in [1..diff_prms] do
		locmodname := ConcatenationString("LocRdPrm_",String(i));
		PrintLine("/*** ");
		if(loc_rd_prms[i].func.numChildren() = 3) then # IxLxI
			a := loc_rd_prms[i].func._children[1].params[1];
			b := loc_rd_prms[i].func._children[2].params[1];
			c := loc_rd_prms[i].func._children[2].params[2];
			d := loc_rd_prms[i].func._children[3].params[1];
			PrintLine("// IxLxI: I(",a,")xL(",b,",",c,")xI(",d,")");
			path := genBRAMPermMem(TRC(TPrm(TL(b,c,a,d))), stream*2, format, opts.precision, locmodname); 
		fi;
		if(loc_rd_prms[i].func.numChildren() = 2) then
			if(Length(loc_rd_prms[i].func._children[1].params) = 2) then # LxI
				b := loc_rd_prms[i].func._children[1].params[1];
				c := loc_rd_prms[i].func._children[1].params[2];
				d := loc_rd_prms[i].func._children[2].params[1];
				PrintLine("// LxI: L(",b,",",c,")xI(",d,")");
				path := genBRAMPermMem(TRC(TPrm(TL(b,c,1,d))), stream*2, format, opts.precision, locmodname); 
			fi;
			if(Length(loc_rd_prms[i].func._children[1].params) = 1) then # IxL
				a := loc_rd_prms[i].func._children[1].params[1];
				b := loc_rd_prms[i].func._children[2].params[1];
				c := loc_rd_prms[i].func._children[2].params[2];
				PrintLine("// IxL: I(",a,")xL(",b,",",c,")");
				path := genBRAMPermMem(TRC(TPrm(TL(b,c,a,1))), stream*2, format, opts.precision, locmodname); 
			fi;
		fi;
		if(loc_rd_prms[i].func.numChildren() = 0) then # L
			b := loc_rd_prms[i].func.params[1];
			c := loc_rd_prms[i].func.params[2];
			PrintLine("// L: L(",b,",",c,")");
			path := genBRAMPermMem(TRC(TPrm(TL(b,c,1,1))), stream*2, format, opts.precision, locmodname); 
		fi;
		
		prefixModules(ConcatenationString(path, locmodname,".v"), ConcatenationString("lr",String(i)));
		module := setModule(ConcatenationString(path, locmodname,".v"), "module_name_is", ConcatenationString("perm_mem_locrd_",String(i)), genpath, bb);
		PrintLine("***/ ");
		PrintLine("`define LOC_RD_MODULE_NAME_",i," ",module);
		getLMLatency(ConcatenationString(path, locmodname,".v"), "RD", i, "/tmp/_spiral.tmp");
	od;
	
	PrintLine("// LM Additive latency ");
	putLMLatency("/tmp/_spiral.tmp", "/tmp/_spiral_readFile.tmp");
	
	PrintLine("/*** ");
	path := HDLGen(streamDFTUnroll(fft_size,2,stream*2), 1, format, 0, 0, 0, "dftgen");
	prefixModules(ConcatenationString(path, "dftgen.v"), "f");
	module := setModule(ConcatenationString(path, "dftgen.v"), "module_name_is", "dftcore", genpath, bb);
	PrintLine("***/ ");
	
	PrintLine("//");
	PrintLine("// FFT Core");
	PrintLine("`define FFT_CORE_MODULE_NAME ", module);
	
	
	if(opts.throttle = 1) then
		PrintLine("`define THROTTLE_EN");
	fi;
	
	
	# Twiddle ROM
	if(Length(twiddles) > 0) then
		if(Length(twiddles) > 1) then
			Error("Unexpected number of twiddles!");
		fi;
		twiddle := twiddles[1];
		a := twiddle.params[1];
		b := twiddle.params[2];
		c := twiddle.params[3];
		
		PrintLine("/*** ");
		path := HDLGen(streamGen(TRC(TDiag(fPrecompute(Tw1(a, b, c)))).withTags([AStream(stream*2)]), InitStreamHw()), 1, format, 0, 0, 0, "twiddleUnit");
		cmdString := ConcatenationString(paradigms.stream._hardwarePath, "dram_scripts/_twidModuleName.sh ", path, "twiddleUnit.v ", "twiddleUnit ", "iter_in");
		Exec(cmdString);
		cmdString := ConcatenationString("cp ", path, "twiddleUnit.v ", genpath);
		Exec(cmdString);
		PrintLine("***/ ");
		PrintLine("//");
		PrintLine("// Twiddle Unit");
		PrintLine("`define TWIDDLE_UNIT_v2");
		PrintLine("`define TWIDDLE_UNIT_NAME tw_twiddleUnit");
		PrintLine("`define CODEBLOCK_IT iter_in");
	fi;
	
	PrintLine(" ");
	PrintLine("// Summary");
	PrintLine("//");
	PrintLine("// Tile Width:",tile_width/stream," accesses");
	PrintLine("// Tile Width:",tile_width," words");
	PrintLine("// Stages:",stages);
	PrintLine("// Streaming Width:",opts.dram_datawidth/opts.precision/2);
	PrintLine("// DFT Size:",fft_size);	
	PrintLine("// Format: ",format," (2:double, 1:single)");	

end;

genVerifFile := function(srt, opts, genpath)
	local mem_rd_prms, tile_width;

	mem_rd_prms := Collect(srt, MemRdPrm);
	tile_width := mem_rd_prms[1].func._children[Length(mem_rd_prms[1].func._children)].params[1];
	
	
	PrintLine("SIZE=",Collect(srt, DFT)[1].params[1]);
	PrintLine("SW=",opts.dram_datawidth/opts.precision/2);
	PrintLine("TILE=",tile_width);
	
	PrintLine("matlab -nodisplay -nosplash -r \"genRefInputOutput($SIZE,$SW,$TILE); exit;\"");
	PrintLine("./run.sh sim");
	PrintLine("matlab -nodisplay -nosplash -r \"compareOutputs($SIZE); exit;\" > matlab.log");
	PrintLine("echo \"\"");
	PrintLine("grep \"SNR (dB)\" matlab.log");

end;


DRAMSystemGen := function(srt, opts)
	local path,conf,cmd,s,verif;
	
	path := Concat("/tmp/spiral/dramSys", String(GetPid()), "/");
    MakeDir(path);
	
	# generate conf file & permMem & dft & put into genpath
	conf := ConcatenationString(path,"_configFile.vh");
	
	# 8/6/2014 - put the verification scripts in the genpath too
	verif := ConcatenationString(path,"verify.sh");
	
	# 2D or 3D?
	s := Length(Collect(srt, MemFence));
	if( s = 2 ) then 
		PrintTo(conf, genConfigFile2(srt, opts, path));
		PrintTo(verif, genVerifFile(srt, opts, path));
		# copy src files into genpath
		cmd := ConcatenationString("cp -rf ", paradigms.stream._hardwarePath,"dram_src/* ", path);
	Exec(cmd);
	else if ( s = 3) then
		PrintTo(conf, genConfigFile3(srt, opts, path));
		# copy src files into genpath
		cmd := ConcatenationString("cp -rf ", paradigms.stream._hardwarePath, "dram_src_3d/* ", path);
	Exec(cmd);
	else
		PrintLine("SPLRuleTree not recognized! s=",s);
	fi;fi;
		
	
	return path;	
end;

