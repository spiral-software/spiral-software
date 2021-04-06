
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# First, for Pease, need parameters:
# 1. n
# 2. direction (forward or inverse)
# 3. data width
# 4. twiddle width
# 5. bit reverse in/out
# 6. scaling?
# 7. parallelism (streaming width)
# 8. BRAM budget

# bitrev = 0 --> n/n
# bitrev = 1 --> n/r
# bitrev = 2 --> r/n

# direction = 0 --> DFT(n, -1) (forward)
# direction = 1 --> DFT(n, 1) (inverse)

# test:
# webGen(0, 16, 2, 0, 0, 16, 16, 0, 0, 4, -1, 0, "test");
# output in /tmp/spiral/[PID]/test.v

_splhdl_path := "";  #set in .gaprc



webGen := function(arch, size, radix, direction, d_type, d_width, t_width, bitrev, scaling, width, brams, ip, name)
   local opts, t, fwd, scale_str, bram_str;

   # With radix limited to 16, this will avoid more expensive 2/8 and 8/2 CT splits.
#   DFT_CT.children := (nt) -> Map2(Filtered(DivisorPairs(nt.params[1]), divpair -> (divpair[1] <= 4 and divpair[2] <= 4)), (m,n) -> [ DFT(m, nt.params[2] mod m), DFT(n, nt.params[2] mod n) ]);

 if (ip = 0) then
       paradigms.stream.avoidPermPatent := 1;
   fi;


   if (arch = 0) then
       opts := InitStreamHw();
   else
       opts := InitStreamUnrollHw();
   fi;

   fwd := Cond(direction = 0, -1, 1);
   scale_str := Cond(scaling = 0, " ", " -s ");
   bram_str := Cond(brams = -1, " ", ConcatenationString(" -br ", String(brams)));
   
    if (_splhdl_path = "") then
	Error("Error: path to SPLHDL compiler not set: paradigms.stream._splhdl_path is not bound.");
    fi;


   opts.profile.makeopts.SPLHDL := _splhdl_path; #"/home/pam/web_backends/splhdl/src/splhdl";
   opts.profile.makeopts.OUTNAME := ConcatenationString(String(name), ".v");
   opts.profile.makeopts.GAP := "gap.spl";   
   opts.profile.makeopts.WRAP := ConcatenationString(scale_str, bram_str, " -lic -web");

   if (d_type = 0) then
       opts.profile.makeopts.DATATYPE := ConcatenationString("fix ", StringInt(d_width));
   else
       opts.profile.makeopts.DATATYPE := ConcatenationString("flt ", StringInt(d_width));
   fi;

#       opts.profile.makeopts.TWIDTYPE := ConcatenationString("-fixtw", " ", StringInt(t_width));
#   Print("name: ", name, "\n");
   
   DFTDR_tSPL_Pease.unroll_its := 1;
   DRDFT_tSPL_Pease.unroll_its := 1;
   
   if ((size = 4) and (width=4) and (radix=2)) then 
       radix := 4; 
   fi; 
   
  if (bitrev = 0) then
       t := streamDFT(TRC(DFT(size, fwd)).withTags([AStream(2*width)]), radix, opts);
   else
       if (bitrev = 1) then
           t := streamDFT(TRC(DRDFT(size, fwd, radix)).withTags([AStream(2*width)]), radix, opts);
       else
           t := streamDFT(TRC(DFTDR(size, fwd, radix)).withTags([AStream(2*width)]), radix, opts);
       fi;
   fi;

#   Print("t = ", t, "\n\n\n");
   
   return CMeasure(t, opts);

end;


webGenDCT := function(size, d_type, d_width, brams, width, ip, name)
   local opts, bram_str, t, width;

   if (ip = 0) then
       paradigms.stream.avoidPermPatent := 1;
   fi;


   opts := InitStreamUnrollHw();
   bram_str := Cond(brams = -1, " ", ConcatenationString(" -br ", String(brams)));

    if (_splhdl_path = "") then
	Error("Error: path to SPLHDL compiler not set: paradigms.stream._splhdl_path is not bound.");
    fi;

   opts.profile.makeopts.SPLHDL := _splhdl_path; 
   opts.profile.makeopts.OUTNAME := ConcatenationString(String(name), ".v");
   opts.profile.makeopts.GAP := "gap.spl";   
   opts.profile.makeopts.WRAP := ConcatenationString(bram_str, " -lic -webdct");

   if (d_type = 0) then
       opts.profile.makeopts.DATATYPE := ConcatenationString("fix ", StringInt(d_width));
   else
       opts.profile.makeopts.DATATYPE := ConcatenationString("flt ", StringInt(d_width));
   fi;
   
   t := streamDCTUnroll(size, width);
   return CMeasure(t, opts);
end;
