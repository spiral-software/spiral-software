
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

ImportAll(paradigms.scratchpad);

Class(ScratchX86CMContext, rec (
	getOpts := meth(arg)
		local opts, elem, lssize, nrsgmts, vlen, size, swp, globalUnrolling, ttype;
		
		lssize := When (Length(arg) >= 2, arg[2], 2);
		nrsgmts := When (Length(arg) >= 3, arg[3], 1);
		vlen := When (Length(arg) >= 4, arg[4], 1);
		size:= When (Length(arg) >= 5, arg[5], 2);
         	ttype := When (Length(arg) >= 6, arg[6], 'R');
		swp := When (Length(arg) >= 7, arg[7], false);
		globalUnrolling := When(Length(arg) >=8, arg[8], 1);
		elem := ScratchpadGlobals.getOpts(lssize,nrsgmts,vlen,size,ttype,swp,globalUnrolling);
        
        	opts := CopyFields(elem);

		opts.unparser := CCMContextScratchUnparserProg;
		opts.codegen := CMContextScratchCodegen;
        
        	opts.profile.makeopts.CFLAGS := "-O2 -Wall -fomit-frame-pointer -msse4.1 -std=gnu99 -static";
        
        	opts.includes := [ "<include/omega64.h>" ];
        	Add(opts.includes, "\"scratchc.h\"");
        
		return opts;
	end,
));

Class(ScratchX86Globals, rec(
	getOpts := meth(arg)
		local opts, elem, lssize, nrsgmts, vlen, size, swp, globalUnrolling, ttype;
		
		lssize := When (Length(arg) >= 2, arg[2], 2);
		nrsgmts := When (Length(arg) >= 3, arg[3], 1);
		vlen := When (Length(arg) >= 4, arg[4], 1);
		size:= When (Length(arg) >= 5, arg[5], 2);
         	ttype := When (Length(arg) >= 6, arg[6], 'R');
		swp := When (Length(arg) >= 7, arg[7], false);
		globalUnrolling := When(Length(arg) >=8, arg[8], 1);
		elem := ScratchpadGlobals.getOpts(When(swp,lssize/2,lssize),nrsgmts,vlen,size,ttype,swp,globalUnrolling);

        opts := CopyFields(elem);
        
        opts.swp_var := swp;
        
        opts.profile.name := "linux-x86-thread";

		opts.unparser := BarrierScratchUnparserProg;
		opts.codegen := SWPBarrierScratchCodegen;

        opts.register := (self, opts) >> "REG_thread";	

        opts.profile.makeopts.CFLAGS := "-O2 -Wall -fomit-frame-pointer -msse4.1 -std=gnu99 -static -lpthread";
        
        opts.barrierCMD := (self, opts) >> "BARRIER";
        opts.initialization := (self, opts) >> "INITIAL";

        opts.includes := [];
        Add(opts.includes, "\"scratch_barrier.h\"");
		Add(opts.includes, "<include/omega64.h>");
        
		return opts;
	end,
));
