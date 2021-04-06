
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


runDramSysGen2D := function(k,n,sw,prec)
	local opts,t,rt,srt,path,cmd;
	
	opts := DRAMGlobals.getOpts(k*k, k*n, n, prec*2*sw, prec, 27, 0, 0);
	# restrict the TTensor rule to match the verification script assumptions
	opts.breakdownRules.TTensor := [AxI_L__BxI_L]; 
	t := MDDFT([n,n]).withTags(opts.tags);
	rt := RandomRuleTree(t, opts);
	srt := SPLRuleTree(rt);
	path := DRAMSystemGen(srt,opts);
	
	PrintLine(path);
	
	return path;
	#
	# cmd := ConcatenationString("mkdir /afs/ece.cmu.edu/project/trac/vol3/users/bakin/dramSysGen/", 
	# 		"2D-n-",String(n),"-k-",String(k),"-sw-",String(sw),"-prec-",String(prec));
	# 	
	# Exec(cmd);
	# 
	# cmd := ConcatenationString("mv ",path,"*"," /afs/ece.cmu.edu/project/trac/vol3/users/bakin/dramSysGen/", 
	# 		"2D-n-",String(n),"-k-",String(k),"-sw-",String(sw),"-prec-",String(prec),"/");
	# 
	# Exec(cmd);
end;


#runDramSysGen1D := function(k,n,sw,prec)
#	local opts,t,rt,srt,path,cmd;
#	
#	opts := DRAMGlobals.getOpts(k*k, k*n, n, prec*2*sw, prec, 27);
#	t := DFT(n*n).withTags(opts.tags);
#	rt := RandomRuleTree(t, opts);
#	srt := SPLRuleTree(rt);
#	path := DRAMSystemGen(srt,opts);
#
#	# cmd := ConcatenationString("mkdir /afs/ece.cmu.edu/project/trac/vol3/users/bakin/dramSysGen/", 
#	# 		"1D-n-",String(n),"-k-",String(k),"-sw-",String(sw),"-prec-",String(prec));
#	# 	
#	# Exec(cmd);
#	# 
#	# cmd := ConcatenationString("mv ",path,"*"," /afs/ece.cmu.edu/project/trac/vol3/users/bakin/dramSysGen/", 
#	# 		"1D-n-",String(n),"-k-",String(k),"-sw-",String(sw),"-prec-",String(prec),"/");
#	# 		
#	# Exec(cmd);
#end;
#
#runDramSysGen3D := function(k,n,sw,prec)
#	local opts,t,rt,srt,path,cmd;
#	
#	opts := DRAMGlobals.getOpts(k*k*k, k*k*n, n, prec*2*sw, prec, 27);
#	t := MDDFT([n,n,n]).withTags(opts.tags);
#	rt := RandomRuleTree(t, opts);
#	srt := SPLRuleTree(rt);
#	path := DRAMSystemGen(srt,opts);
#
#	# cmd := ConcatenationString("mkdir /afs/ece.cmu.edu/project/trac/vol3/users/bakin/dramSysGen/", 
#	# 		"3D-n-",String(n),"-k-",String(k),"-sw-",String(sw),"-prec-",String(prec));
#	# 	
#	# Exec(cmd);
#	# 
#	# cmd := ConcatenationString("mv ",path,"*"," /afs/ece.cmu.edu/project/trac/vol3/users/bakin/dramSysGen/", 
#	# 		"3D-n-",String(n),"-k-",String(k),"-sw-",String(sw),"-prec-",String(prec),"/");
#	# 		
#	# Exec(cmd);
#end;
#
#
#
#runDramSysGenAll := function()
#	local k,n,sw,prec;
#	
#	k := 8;		n := 128;	sw := 2;	prec := 32;
#	runDramSysGen2D(k,n,sw,prec);
#	runDramSysGen1D(k,n,sw,prec);
#	k := 8;		n := 128;	sw := 4;	prec := 32;
#	runDramSysGen2D(k,n,sw,prec);
#	runDramSysGen1D(k,n,sw,prec);
#	k := 16;		n := 512;	sw := 2;	prec := 32;
#	runDramSysGen2D(k,n,sw,prec);
#	runDramSysGen1D(k,n,sw,prec);
#	k := 16;		n := 512;	sw := 4;	prec := 32;
#	runDramSysGen2D(k,n,sw,prec);
#	runDramSysGen1D(k,n,sw,prec);
#	k := 16;		n := 512;	sw := 8;	prec := 32;
#	runDramSysGen2D(k,n,sw,prec);
#	runDramSysGen1D(k,n,sw,prec);
#
#end;


runDramSysGenAll := function(k,n,sw,prec)
	local path,cmd;
	
	path := runDramSysGen2D(k,n,sw,prec);
	cmd := ConcatenationString("cd ",path,"; source verify.sh; grep SNR matlab.log > /afs/ece.cmu.edu/project/trac/vol3/users/bakin/perfect/snr-summary/snr.n",String(n),
								".k",String(k),".s",String(sw),".p",String(prec));
	Exec(cmd);
	
end;


runDramSysGenTest := function()
#	runDramSysGenAll(4,128,2,32);
#	runDramSysGenAll(8,128,2,32);
#	runDramSysGenAll(4,128,4,32);
	
#	runDramSysGenAll(4,256,2,32);
#	runDramSysGenAll(8,256,4,32);
#	runDramSysGenAll(16,256,4,32);
	
#	runDramSysGenAll(16,512,8,32);
	
	runDramSysGenAll(32,1024,4,32);
	
	runDramSysGenAll(16,2048,4,32);

end;

