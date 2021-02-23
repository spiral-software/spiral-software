
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# DebugRewriting(true);
# spiral.rewrite.RuleTrace := PrintLine;
# spiral.rewrite.RuleStatus := Print;

# s := SumsRuleTree(DFT(3), SpiralDefaults);
# RecursiveFindBug(s, SpiralDefaults, (t,opts) -> t.free()=[]);
# CompareCodeMat


# If set to 1, the permutations will not use the patented
# streaming permutation structures.
avoidPermPatent := 0;  


InitStreamHw := () -> CopyFields(
    StreamDefaults, 

    # Other breakdown rules are set in StreamDefaults, in init.g
    rec(            
    breakdownRules := rec(
        DFTDR     := [DFTDR_tSPL_Pease ],
        DRDFT     := [DRDFT_tSPL_Pease_Stream ],
        DFT       := [DFT_Base, DFT_HW_CT, DFT_tSPL_Stream, DFT_Stream1_Fold, DFT_tSPL_Bluestein_Stream_Rolled, DFT_tSPL_Mixed_Radix_It, DFT_PD], #, DFT_tSPL_StreamR2R4],
        MDDFT     := [MDDFT_tSPL_HorReuse_Stream],
        TDiag     := [TDiag_TC_Pease, TDiag_base, TDiag_tag_stream],
        WHT       := [WHT_Base, WHT_tSPL_Pease, WHT_tSPL_STerm],
        TTensorI  := [IxA_base, IxA_L_base, L_IxA_base, AxI_base, IxA_L_split, L_IxA_split, 
                      IxA_stream_base, TTensorI_Streamtag, TTensorI_Stream_Diag_Uneven, TTensorI_Stream_Perm_Uneven],
        TTensorInd := [TTensorInd_stream_base, TTensorInd_stream_prop],
        TCompose  := [TCompose_tag],
        TICompose := [TICompose_tag],
        TDirectSum := [A_dirsum_B],
        TDR       := [DR_StreamPerm],
        TRC       := [TRC_stream],
        TCond     := [TCond_tag],
        realDFTPerm := [realDFTPerm_Stream],
        TConjEven := [TConjEven_stream],
        TConjEvenStreamBB := [TConjEvenStreamBB_base],
        rDFTSkew  := [rDFT_Skew_CT, rDFT_Skew_Base4, rDFT_Skew_Pease],
        PkRDFT1   := [PkRDFT1_rDFTSkew],
        InfoNt    := [Info_Base],

        # Switch these lines to enable Jarvinen streaming perms
        TL      := [L_base, L_StreamBase, L_StreamPerm],
        #TL      := [L_base, L_Stream0, L_Stream1, L_Stream2],

        TPrm    := [TPrm_stream, TPrm_flat],
        TPrmMulti := [TPrmMulti_stream],

	#Sort := [Sort_Stream],
	Sort := [Sort_Stream, Sort_Stream_Iter, Sort_Stream4],
	SortVec := [Sort_Stream_Vec, Sort_Stream4_Vec],
    ),
    profile := spiral.profiler.default_profiles.fpga_splhdl
));

InitStreamHwRDFTtoDFT := function()
   local opts;

   opts := InitStreamHw();
   opts.breakdownRules.PkRDFT1 := [PkRDFT1_DFT];
   opts.breakdownRules.DFT := [DFT_Base, DFT_HW_CT, DFT_tSPL_Stream_Trans, DFT_Stream1_Fold];
   return opts;
end;

InitStreamUnrollHw := () -> CopyFields(
    StreamDefaults, 

    # Other breakdown rules are set in StreamDefaults, in init.g
    rec(
    breakdownRules := rec(
        DFTDR     := [DFTDR_tSPL_Stream ],
        DRDFT     := [DRDFT_tSPL_Stream ],
        DFT       := [DFT_Base, DFT_HW_CT, DFT_tSPL_Stream, DFT_tSPL_Fold_CT, DFT_tSPL_MultRadix_Stream, DFT_tSPL_Bluestein_Stream, DFT_tSPL_Mixed_Radix_Stream, DFT_tSPL_StreamFull, DFT_PD, DFT_Stream1_Fold], #, DFT_tSPL_StreamR2R4 ],
        MDDFT     := [MDDFT_tSPL_Unroll_Stream],
        Sc_DCT2   := [Sc_DCT2_Stream],   
        TDCT2     := [DCT2_Stream],   
        TDiag     := [TDiag_tag_stream, TDiag_TIFold_It, TDiag_base],
        WHT       := [WHT_Base, WHT_tSPL_Pease, WHT_tSPL_STerm],
        TTensorI  := [IxA_base, IxA_L_base, L_IxA_base, AxI_base, IxA_L_split, L_IxA_split, 
                      IxA_stream_base, TTensorI_Streamtag, TTensorI_Stream_Diag_Uneven, TTensorI_Stream_Perm_Uneven],
        TTensorInd := [TTensorInd_stream_base, TTensorInd_stream_prop],
        TCompose  := [TCompose_tag],
        TICompose := [TICompose_tag],
        TDirectSum := [A_dirsum_B],
        TDR       := [DR_StreamPerm],
        TRC       := [TRC_stream],
        TCond     := [TCond_tag],
        realDFTPerm := [realDFTPerm_Stream],
        TConjEven := [TConjEven_stream],
        TConjEvenStreamBB := [TConjEvenStreamBB_base],
        rDFTSkew  := [rDFT_Skew_CT, rDFT_Skew_Base4, rDFT_Skew_Stream, rDFT_Skew_Stream_Mult_Radices],
        PkRDFT1   := [PkRDFT1_rDFTSkew],
        InfoNt    := [Info_Base],

        # Switch these lines to enable Jarvinen streaming perms
        TL      := [L_base, L_StreamBase, L_StreamPerm],
#         TL      := [L_base, L_Stream0, L_Stream1, L_Stream2]
#        TPrm    := [TPrm_stream_bit_perm, TPrm_stream_not_bit_perm],
        TPrm    := [TPrm_stream, TPrm_flat],
        TPrmMulti := [TPrmMulti_stream],
    ),
    profile := spiral.profiler.default_profiles.fpga_splhdl
));


InitStreamUnrollHwRDFTtoDFT := function()
   local opts;

   opts := InitStreamUnrollHw();
   opts.breakdownRules.PkRDFT1 := [PkRDFT1_DFT];
#   opts.breakdownRules.DFT := [DFT_Base, DFT_HW_CT, DFT_tSPL_Stream_Trans, DFT_Stream1_Fold, DFT_tSPL_Fold_CT];
   return opts;
end;


# a = 0 to make outer prod streaming, a=1 to make it HR
# b = 0 to make inner prod streaming, b=1 to make it HR
InitStreamMultiLevel := (a, b) -> CopyFields(
    StreamDefaults, 

    # Other breakdown rules are set in StreamDefaults, in init.g
    rec(
    breakdownRules := rec(
        DFTDR     := Cond(b=0, [DFTDR_tSPL_Stream], [DFTDR_tSPL_Pease]),
        DRDFT     := Cond(b=0, [DRDFT_tSPL_Stream ], [DRDFT_tSPL_Pease_Stream]),
        DFT       := Cond(a=0, 
                        [DFT_Base, DFT_HW_CT, DFT_tSPL_Stream, DFT_tSPL_Fold_CT, DFT_tSPL_Bluestein_Stream], 
                        [DFT_Base, DFT_HW_CT, DFT_tSPL_Stream, DFT_tSPL_Fold_CT, DFT_tSPL_Bluestein_Stream_Rolled]
                     ),
        MDDFT     := Cond(a=0, [MDDFT_tSPL_Unroll_Stream], [MDDFT_tSPL_HorReuse_Stream]),
        TDiag     := [TDiag_tag_stream, TDiag_TIFold_It, TDiag_base, TDiag_TC_Pease],
        WHT       := [WHT_Base, WHT_tSPL_Pease, WHT_tSPL_STerm],
        TTensorI  := [IxA_base, IxA_L_base, L_IxA_base, AxI_base, IxA_L_split, L_IxA_split, 
                      IxA_stream_base, TTensorI_Streamtag],
        TCompose  := [TCompose_tag],
        TICompose := [TICompose_tag],
        TDirectSum := [A_dirsum_B],
        TDR       := [DR_StreamPerm],
        TRC       := [TRC_stream],
        TCond     := [TCond_tag],
    
        # Switch these lines to enable Jarvinen streaming perms
        TL      := [L_base, L_StreamBase, L_StreamPerm],
        # TL      := [L_base, L_Stream0, L_Stream1, L_Stream2]
        TPrm    := [TPrm_stream, TPrm_flat],
        TPrmMulti := [TPrmMulti_stream],
    ),
    profile := spiral.profiler.default_profiles.fpga_splhdl
));


InitStreamAllHw := () -> CopyFields(
    StreamDefaults, 

    # Other breakdown rules are set in StreamDefaults, in init.g
    rec(
    breakdownRules := rec(
        DFTDR     := [DFTDR_tSPL_Stream, DFTDR_tSPL_Pease ],
        DRDFT     := [DRDFT_tSPL_Stream ],
        DFT       := [DFT_Base, DFT_HW_CT, DFT_tSPL_Stream, DFT_tSPL_Fold_CT] ,# DFT_tSPL_StreamR2R4],
        MDDFT     := [MDDFT_tSPL_Unroll_Stream, MDDFT_tSPL_HorReuse_Stream],
        TDiag     := [TDiag_tag_stream, TDiag_TIFold_It, TDiag_base, TDiag_TC_Pease],
        WHT       := [WHT_Base, WHT_tSPL_Pease, WHT_tSPL_STerm],
        TTensorI  := [IxA_base, IxA_L_base, L_IxA_base, AxI_base, IxA_L_split, L_IxA_split, 
                      IxA_stream_base, TTensorI_Streamtag],
        TCompose  := [TCompose_tag],
        TICompose := [TICompose_tag],
        TDirectSum := [A_dirsum_B],
        TDR       := [DR_StreamPerm],
        TRC       := [TRC_stream],
        TCond     := [TCond_tag],
    
        # Switch these lines to enable Jarvinen streaming perms
        TL      := [L_base, L_StreamPerm]
        # TL      := [L_base, L_Stream0, L_Stream1, L_Stream2]
    ),
    profile := spiral.profiler.default_profiles.fpga_splhdl
));



streamRealDFT := function(trn, radix, opts, unroll)
    local rt, srt, srt2, srt3, srt4;

    rDFT_Skew_Pease.unroll_its := unroll;
    rDFT_Skew_Pease.radix := radix;
    rDFT_Skew_Stream.radix := radix;

    # set radix for complex DFT rules in case I am using half-size method
    DFT_tSPL_Stream.radix := radix;
    DFT_tSPL_Stream_Trans.radix := radix;
    DFTDR_tSPL_Pease.unroll_its := unroll;
    DRDFT_tSPL_Pease_Stream.unroll_its := unroll;

    rt   := RandomRuleTree(trn, opts);
    srt  := SumsRuleTreeStrategy(rt, SumStreamStrategy, opts);

    # I don't understand what's going on here.
    # For some reason, this line messes everything up when I try the half-size complex
    # RDFT method, but I can't figure out what it is. 
#!    srt2 := ApplyStrategy(srt, RuleRDFTStrategy, UntilDone, opts);
    
    # So, this is a temporary hack to make half-size compex gen. work
    srt2 := ApplyStrategy(srt, StreamStrategy, UntilDone, opts);


    srt3 := Process_fPrecompute(srt2, opts);
    srt4 := srt3.createCode();
    return srt4;
    
end;
 

streamRDFTSkew := function(size, skew, radix, width, unroll)
   local k, m, logM, opts, t;

   k    := radix/2;
   m    := size/radix;
   logM := LogInt(m,k);

   if ((logM+1)/unroll = 1) then
       opts := InitStreamUnrollHw();
   else
       opts := InitStreamHw();
   fi;

   t := rDFTSkew(size,V(skew)).withTags([AStream(width)]);

   return streamRealDFT(t, radix, opts, unroll);
end;


streamRDFTSkewUnroll := function(size, skew, radix, width)
   return streamRDFTSkew(size, skew, radix, width, LogInt((size/radix),(radix/2))+1);
end;


streamRDFT := function(size, radix, width, unroll, method)
   # method = 1 for native RDFT algorithms, 2 for half-size complex RDFT

   local k, m, logM, opts, t;

   k    := radix/2;
   m    := size/radix;

   if (method = 1) then
       logM := LogInt(m,k);
       if ((logM+1)/unroll = 1) then
           opts := InitStreamUnrollHw();
       else
           opts := InitStreamHw();
       fi;
   fi;

   if (method = 2) then

       # Keep in mind that if you request an "unrolling" 1 basic block with an incompatible radix, it will
       # unroll here.  E.g., (32,8,16,1,2)
       if (LogInt(size/2, radix) = unroll) then
           opts := InitStreamUnrollHwRDFTtoDFT();
       else
           opts := InitStreamHwRDFTtoDFT();
       fi;
   fi;
       

   t := PkRDFT1(size,1).withTags([AStream(width)]);

   return streamRealDFT(t, radix, opts, unroll);
end;


streamRDFTPease := function(size, radix, width, unroll)
   return streamRealDFT(
        PkRDFT1(size,1).withTags([AStream(width)]),
        radix,
        InitStreamHw(), 
        unroll);
end;
         
streamRDFTUnroll := function(size, radix, width, method)
   return streamRDFT(size, radix, width, LogInt((size/radix),(radix/2))+1, method);
end;


streamDFT := function(trn, radix, opts)
   local rt, srt, srt2, srt3, srt4, opts;
   
#   DFT_tSPL_Stream.minRadix := radix;
#   DFT_tSPL_Stream.maxRadix := radix;
   DFT_tSPL_Stream.radix := radix;

   
   rt   := RandomRuleTree(trn, opts);
   srt  := SumsRuleTreeStrategy(rt, SumStreamStrategy, opts);
   srt2 := ApplyStrategy(srt, StreamStrategy, UntilDone, opts);
   srt3 := Process_fPrecompute(srt2, opts);
   srt4 := srt3.createCode();
   
   return srt4;   
end;

# Fully unrolled.  
streamDFTNoFold := function(n,k)
    local t, r, s, s2, s3, s4, opts;
    
    opts := InitStreamHw();
    t    := RC(DFT(n,k));
    r    := RandomRuleTree(t, SpiralDefaults);
    s    := SumsRuleTreeStrategy(r, SumStreamStrategy, opts);
    s2   := ApplyStrategy(s, StreamStrategy, UntilDone, opts);
    s3   := Process_fPrecompute(s2, opts);
    s4   := s3.createCode();
    return s4;
end;

 
TestRandomPerm := function(size, width)
   local p, s;
   p := RandomPerm(size);
   s := StreamPerm([p], 1, width, 0);
   return MatSPL(p) - MatSPL(s.createCode());
end;

StreamRandomPerm := function(size, width)
   local s;
   s := StreamPerm([RandomPerm(size)], 1, width, 0);
   return s.createCode();
end;

StreamRandomPerms := function(size, width, number)
   local it, s, s2;
   it := Ind(number);
   s := StreamPerm(List([1..number], i->RandomPerm(size)), 1, width, it);
   s2 := ICompose(it, number, s);
   return s2.createCode();
end;


MyPermTest := function()
   local t0, t1, w, p, s, s2,h;
   p := RandomPerm(4096);
   Print("finished generating the random perm\n");

   # 0 = random
#   stream._whichMeth := 0;

#   for w in [2,4,8,16,32,64,128,256, 512] do
   
#   for h in [1,2] do
#   paradigms.stream._whichMeth := h;
   for w in [2,4,8,16,32,64] do
      t0 := TimeInSecs();
      s := StreamPerm(p, 1, w);
      s2 := s.createCode();
      t1 := TimeInSecs();
      Print("meth=", stream._whichMeth, ", width=", w, ", time=", t1-t0, "\n");
   od;
#   od;
end;
   
NewPermTest := function()
   local t0, t1, w,p,s,s2;
   
   p := RandomPerm(4096);

   t0 := TimeInSecs();
   s := StreamPerm(p, 1, 256);
   s2 := s.createCode();
   t1 := TimeInSecs();
   Print("time = ", t1-t0, "\n");
end;

streamGen := function(SPL, opts)
    local rt, srt, srt2, srt3, srt4;
    rt   := RandomRuleTree(SPL, opts);
   srt  := SumsRuleTreeStrategy(rt, SumStreamStrategy, opts);
   srt2 := ApplyStrategy(srt, StreamStrategy, UntilDone, opts);
   srt3 := Process_fPrecompute(srt2, opts);
   srt4 := srt3.createCode();
   
   return srt4;   
end;


streamDCTUnroll := function(size, width)
   return streamGen(TDCT2(size).withTags([AStream(width)]), InitStreamUnrollHw());
end;

streamDFTUnroll := function(size, radix, width)
   return streamDFT(TRC(DFT(size, -1)).withTags([AStream(width)]), radix, InitStreamUnrollHw());
end;

#streamIDFTUnroll := function(size, radix, width)
#   return streamDFT(TRC(DFT(size, 1)).withTags([AStream(width)]), radix, InitStreamUnrollHw());
#end;

streamDFTDRUnroll := function(size, radix, width)
   return streamDFT(TRC(DFTDR(size, 1, radix)).withTags([AStream(width)]), radix, InitStreamUnrollHw());
end;

streamDRDFTUnroll := function(size, radix, width)
   return streamDFT(TRC(DRDFT(size, 1, radix)).withTags([AStream(width)]), radix, InitStreamUnrollHw());
end;

streamIDFTUnroll := function(size, radix, width)
   if (width <= 2*size) then
       return streamDFT(TRC(DFT(size, -1)).withTags([AStream(width)]), radix, InitStreamUnrollHw());
   else
       return streamDFT(TRC(TTensorI(DFT(size, -1), width/(2*size), APar, APar)).withTags([AStream(width)]), radix, InitStreamUnrollHw());
   fi;
end;

streamIDFTDRUnroll := function(size, radix, width)
   return streamDFT(TRC(DFTDR(size, -1, radix)).withTags([AStream(width)]), radix, InitStreamUnrollHw());
end;

streamDRIDFTUnroll := function(size, radix, width)
   return streamDFT(TRC(DRDFT(size, -1, radix)).withTags([AStream(width)]), radix, InitStreamUnrollHw());
end;


streamDFTPease := function(size, radix, width, unroll)
   DFTDR_tSPL_Pease.unroll_its := unroll;
   return streamDFT(TRC(DFT(size, 1)).withTags([AStream(width)]), radix, InitStreamHw());
end;

streamDFTDRPease := function(size, radix, width)
   return streamDFT(TRC(DFTDR(size, 1, radix)).withTags([AStream(width)]), radix, InitStreamHw());
end;

streamDRDFTPease := function(size, radix, width)
   return streamDFT(TRC(DRDFT(size, 1, radix)).withTags([AStream(width)]), radix, InitStreamHw());
end;

streamIDFTPease := function(size, radix, width, unroll)
   DFTDR_tSPL_Pease.unroll_its := unroll;
   return streamDFT(TRC(DFT(size, -1)).withTags([AStream(width)]), radix, InitStreamHw());
end;

streamIDFTDRPease := function(size, radix, width)
   return streamDFT(TRC(DFTDR(size, -1, radix)).withTags([AStream(width)]), radix, InitStreamHw());
end;

streamDRIDFTPease := function(size, radix, width)
   return streamDFT(TRC(DRDFT(size, -1, radix)).withTags([AStream(width)]), radix, InitStreamHw());
end;


# countPerms := function(s)
#    local perms, l;
#    perms := Collect(s, BRAMPerm);
#    l := List([1..Length(perms)], i->perms[i].dimensions[1]);
#    Sort(l);
#    return Reversed(l);
# end;



HDLPrint := function(t, p, r, x)
    Print("points(", p, ")\n");
    Print("threshold(", t, ")\n");
    Print("radix(", r, ")\n");
    Print(x);
end;

HDLPrint2 := function(w, w2, t, p, r, x)
    Print("points(", p, ")\n");
    Print("threshold(", t, ")\n");
    Print("radix(", r, ")\n");
    Print(x);
end;

HDLPrint3 := HDLPrint;

#! This is the code that changes how we deal with Complex mults.
# RCDiag.code := meth(self, y, x)
#    local i, elt, re, im, rct;
#    i := Ind();
#    elt := self.element.lambda();
#    re := elt.at(2 * i);
#    im := elt.at(2*i+1);

#    rct := var("rct");
   
#    return loop(i, Rows(self) / 2, 
#        chain( 
#         assign(nth(rct, (2*i)), mul(re, nth(x, (2*i)))),
#         assign(nth(rct, (2*i+1)), mul(im, nth(x, (2*i+1)))),
#         assign(nth(y, 2*i), nth(rct, 2*i) - nth(rct, 2*i+1)),
#         assign(nth(y, 2*i+1), mul((re + im), (nth(x, 2*i) + nth(x, 2*i+1))) - 
#             nth(rct, 2*i) - nth(rct, 2*i+1))
#        )
#     );
# end;

CodeRCDiag42 := RCDiag.code;
CodeRCDiag33 := meth(self, y, x)
    local i, elt, re, im, rct;
    i := Ind();
    elt := self.element.lambda();
    re := elt.at(2 * i);
    im := elt.at(2*i+1);
    rct := var("rct");
   
    return loop(i, Rows(self) / 2, 
        chain( 
         assign(nth(rct, (2*i)), mul(re, nth(x, (2*i)))),
         assign(nth(rct, (2*i+1)), mul(im, nth(x, (2*i+1)))),
         assign(nth(y, 2*i), nth(rct, 2*i) - nth(rct, 2*i+1)),
         assign(nth(y, 2*i+1), mul((re + im), (nth(x, 2*i) + nth(x, 2*i+1))) - 
             nth(rct, 2*i) - nth(rct, 2*i+1))
        )
     );
 end;

# hack
fCompose.dims := self >> [self.n, self.n];
fCompose.sums := self >> self.child(1) * self.child(2);


# n = size
# r = radix
# w = width
# outer is the amount of times to unroll outer loop (1 or 2)
# inner is the amount of times to unroll inner loop 
stream2DDFT := function(n, r, w, outer, inner)
         local srt, strat, strat2, opts;

         opts := InitStreamMultiLevel(Cond(outer=1, 1, 0), Cond(inner=LogInt(n, r), 0, 1));
         MDDFT_tSPL_Unroll_Stream.minRadix := r;
         MDDFT_tSPL_Unroll_Stream.maxRadix := r;
         MDDFT_tSPL_HorReuse_Stream.minRadix := r;
         MDDFT_tSPL_HorReuse_Stream.maxRadix := r;

         DFTDR_tSPL_Pease.unroll_its := inner;

         srt := SumsRuleTreeStrategy(
                   RandomRuleTree(
                      TRC(MDDFT([n,n])).withTags([AStream(w)]), 
                      opts), 
                   SumStreamStrategy, opts);

         strat := Process_fPrecompute(
                    ApplyStrategy(srt, StreamStrategy, 
                       UntilDone, opts), 
                    opts);
         
         strat2 := strat.createCode();
         return strat2;
end;


# n = size
# r = radix
# w = width
# outer is the amount of times to unroll outer loop (1 or 2)
# inner is the amount of times to unroll inner loop 
streamBluesteinDFT := function(n, r, w, outer, inner)
         local opts, t, r, s, s2, s3, s4, k;

         # Only works for non-two power n.         
         k := 2^(Log2Int(2*n) + 1);

         opts := InitStreamMultiLevel(Cond(outer=1, 1, 0), Cond(inner=LogInt(k, r), 0, 1));
         DFT_tSPL_Stream.radix := r;

         DFTDR_tSPL_Pease.unroll_its := inner;
         DRDFT_tSPL_Pease_Stream.unroll_its := inner;

         t  := TRC(DFT(n)).withTags([AStream(w)]);
         r  := RandomRuleTree(t, opts);
         s  := SumsRuleTreeStrategy(r, SumStreamStrategy, opts);
         s2 := ApplyStrategy(s, StreamStrategy, UntilDone, opts);
         s3 := Process_fPrecompute(s2, opts);
         s4 := s3.createCode();

         return s4;
end;


_AbsFloatMat := function(v)
   local l, i, res, j;
   res := [];
   l := Length(v[1]);
   for i in [1..l] do
      res[i] := [];
      for j in [1..l] do
         res[i][j] := AbsFloat(v[i][j]);
      od;
   od;
   return res;
end;

_MaxAbsMat := function(m)
   local res, l, i, j;
   res := 0;
   l := Length(m[1]);
   for i in [1..l] do
      for j in [1..l] do
         if (res < AbsFloat(m[i][j])) then
             res := m[i][j];
         fi;
      od;
   od;
   return res;
end;

_MaxMat := function(m)
   local res, l, i, j;
   res := 0;
   l := Length(m[1]);
   for i in [1..l] do
      for j in [1..l] do
         if (res < m[i][j]) then
             res := m[i][j];
         fi;
      od;
   od;
   return res;
end;

_AvgMat := function(m)
   local res, l, i, j;
   res := 0;
   l := Length(m[1]);
   for i in [1..l] do
      for j in [1..l] do
         res := res + m[i][j];
      od;
   od;
   res := res / (l*l);
   return res;
end;

_MaxAbsVec := function(m)
   local res, l,i;
   res := 0;
   l := Length(m);
   for i in [1..l] do
      if (AbsFloat(m[i]) > res) then
          res := AbsFloat(m[i]);
      fi;
   od;
   return res;
end;

_setHDLDataType := function(t, w)
    spiral.profiler.default_profiles.fpga_splhdl.makeopts.DATATYPE := ConcatenationString(t, " ", StringInt(w));
end;

_setHDLTwidWidth := function(w)
    spiral.profiler.default_profiles.fpga_splhdl.makeopts.TWIDTYPE := ConcatenationString("-fixtw", " ", StringInt(w));
end;

_setFixPointMode := function(w)
   spiral.paradigms.stream._setHDLDataType("fix", w);
   spiral.profiler.default_profiles.fpga_splhdl.makeopts.TWIDTYPE := "";
end;

_setFloatMode := function()
   spiral.paradigms.stream._setHDLDataType("flt", 1);
   spiral.profiler.default_profiles.fpga_splhdl.makeopts.TWIDTYPE := "";
end;



##
## Note regarding E, omega:
##    omega(a,b) = E(a)^b = exp(2*pi*i*b/a)
##
## 


streamTransDFT := (n,r,w) >> streamDFT(TRC(TCompose([TDR(n,r), DRDFT(n,-1,r)])).withTags([AStream(w)]), r, InitStreamHw());

HDLCompile := function(t, name, dir)
   local opts;
   
   opts := InitStreamHw();

   opts.profile.outdir := ConcatenationString("/Users/pam/run/", String(dir));
   opts.profile.makeopts.OUTNAME := ConcatenationString(String(name), ".v");
   opts.profile.makeopts.GAP := ConcatenationString(String(name), ".spl");
   opts.profile.makeopts.WRAP := "-wrap -r -l";   

   Print(name, "\n");

   return CMeasure(t, opts);

end;

HDLCompileASIC := function(t, name, dir, freq)
   local opts;
   opts := InitStreamHw();

   opts.profile.outdir := ConcatenationString("/afs/scotch/usr/pam/run/", String(dir));
   opts.profile.makeopts.OUTNAME := ConcatenationString(String(name), ".v");
   opts.profile.makeopts.GAP := ConcatenationString(String(name), ".spl");
   opts.profile.makeopts.WRAP := ConcatenationString("-r -l -bb -af ", String(freq));   

   Print(name, "\n");

   return CMeasure(t, opts);
end;


HDLCompileFloat := function(t, name, dir)
   local opts;
   
   opts := InitStreamHw();

   opts.profile.outdir := ConcatenationString("/afs/scotch/usr/pam/run/", String(dir));
   opts.profile.makeopts.OUTNAME := ConcatenationString(String(name), ".v");
   opts.profile.makeopts.GAP := ConcatenationString(String(name), ".spl");
   opts.profile.makeopts.DATATYPE := "flt 1";
#   opts.profile.makeopts.WRAP := "-r -l -br 1152";   
   opts.profile.makeopts.WRAP := "-r -l -af 1000 -bb";   

   Print(name, "\n");

   return CMeasure(t, opts);

end;

HDLCompileDouble := function(t, name, dir)
   local opts;
   
   opts := InitStreamHw();

   opts.profile.outdir := ConcatenationString("/afs/scotch/usr/pam/run/", String(dir));
   opts.profile.makeopts.OUTNAME := ConcatenationString(String(name), ".v");
   opts.profile.makeopts.GAP := ConcatenationString(String(name), ".spl");
   opts.profile.makeopts.DATATYPE := "flt 2";
   opts.profile.makeopts.WRAP := "-r -l -br 1152";   

   Print(name, "\n");

   return CMeasure(t, opts);

end;

HDLCompileIntel := function(name, t, bw, freq)
   local opts, res;
   spiral.paradigms.stream._setHDLDataType("fix", bw);

   opts := InitStreamHw();
   
   opts.profile.makeopts.OUTNAME := ConcatenationString(String(name), ".v");
   opts.profile.makeopts.GAP := ConcatenationString(String(name), ".spl");
   opts.profile.makeopts.WRAP := ConcatenationString("-lic -af ", String(freq), " -s -r -v");
   res := CMeasure(t, opts);

   return res;
end;

# For what we are doing now, we need to turn on scaling + license 
# also: get rid of wrapper
runIntel := function()
    local t;
    t := streamDFTNoFold(128,-1); # -1 for forward, 1 for inverse
    HDLCompileIntel("ifft-128-6.spl", t, 6);
    HDLCompileIntel("ifft-128-8.spl", t, 8);
    HDLCompileIntel("ifft-128-10.spl", t, 10);
    HDLCompileIntel("ifft-128-12.spl", t, 12);
    HDLCompileIntel("ifft-128-14.spl", t, 14);
    HDLCompileIntel("ifft-128-16.spl", t, 16);
end;


checkRules := (rls, ob) >> Filtered(rls, i->i.applicable(ob));


peaseSW := meth(n)
   local r, k, t, s, s2, s3, s4, opts;
   opts := InitStreamHw();
   k := Log2Int(n);
   t := RC(Compose(List([0..k-1], i->L(n,2) * Tensor(I(n/2), DFT(2)) * TDiag(fPrecompute(TC(n, 2, i, -1))))) * DR(n, 2));
   r := RandomRuleTree(t, spiral.SpiralDefaults);
   s := SumsRuleTreeStrategy(r, SumStreamStrategy, opts);
   s2 := ApplyStrategy(s, StreamStrategy, UntilDone, opts);
   s3 := Process_fPrecompute(s2, opts);
   s4 := s3.createCode();
   return s4;
end;

# Generate fully unfolded Pease hardware
# peaseSWR(n, r, inv, dir);
#   n: problem size
#   r: radix
#   inv: 0 for DFT, 1 for IDFT
#   dir: 0 for bit reversal on input side, 1 for bit reversal on output side
peaseSWR := meth(n, r, inv, dir)
   local k, t, s, s2, s3, s4, invparam, opts, t;
   opts := InitStreamHw();
   k := LogInt(n,r);

   if (inv = 0) then
       invparam := -1;
   else
       invparam := 1;
   fi;

   if (dir = 0) then
       t := RC(Compose(List([0..k-1], i->L(n,r) * Tensor(I(n/r), DFT(r, invparam)) * TDiag(fPrecompute(TC(n, r, i, invparam))))) * DR(n, r));
   else
       t :=  RC(DR(n, r) * Compose(List([0..k-1], i->
	         TDiag(fPrecompute(TC(n, r, k-1-i, invparam))) *
		 Tensor(I(n/r), DFT(r, invparam)) * 
		 L(n,n/r) 
       )));
   fi;

   r := RandomRuleTree(t, spiral.SpiralDefaults);
   s := SumsRuleTreeStrategy(r, SumStreamStrategy, opts);
   s2 := ApplyStrategy(s, StreamStrategy, UntilDone, opts);
   s3 := Process_fPrecompute(s2, opts);
   s4 := s3.createCode();
   return s4;
end;

ctSW := meth(n)
   local r, k, t, s, s2, s3, s4, opts;
   opts := InitStreamHw();
   k := Log2Int(n);
   t := RC(Compose(List([0..k-1], i-> Tensor(I(2^i), DFT(2), I(2^(k-i-1))) * Tensor(I(2^i), TDiag(fPrecompute(Tw1(2^(k-i), 2^(k-i-1), 1)))))) * DR(n, 2));
   r := RandomRuleTree(t, spiral.SpiralDefaults);
   s := SumsRuleTreeStrategy(r, SumStreamStrategy, opts);
   s2 := ApplyStrategy(s, StreamStrategy, UntilDone, opts);
   s3 := Process_fPrecompute(s2, opts);
   s4 := s3.createCode();
   return s4;
end;

_splhdl_path := "";
_synth_path := "";

# HDLSynthesize(transform, platform, format, precision, frequency, filename)
# platform: 0 for 65nm ASIC, 1 for FPGA
# format: 0 for fixed point, 1 for floating point
# precision: # bits precision, ignored for floating point
# wrapper: 0 for normal mode, 1 to use a wrapper for synthesizing wide designs on FPGA
HDLSynthesize := function(t, platform, format, precision, frequency, wrapper, filename)
    local opts, cmdString, path, brams, wrapString;
    
    path := Concat("/tmp/spiral/", String(GetPid()), "/");
    MakeDir(path);
    brams := 1440;

    opts := InitStreamHw();
    PrintTo(ConcatenationString(path, filename, ".spl"), HDLPrint(1024, t.dims(), -1, t));

    if (_splhdl_path = "") then
	Error("Error: path to SPLHDL compiler not set: paradigms.stream._splhdl_path is not bound.");
    fi;

    if (_synth_path = "") then
	Error("Error: path to synthesis scripts not set: paradigms.stream._synth_path is not bound.");
    fi;

    #cmdString := ConcatenationString(_splhdl_path, " ", path, filename, ".spl -stb -o ", path, filename, ".v -r -l ");
    cmdString := ConcatenationString(_splhdl_path, " ", path, filename, ".spl -o ", path, filename, ".v -r -l ");

    if (wrapper = 1) then
	wrapString := " -wrap";
    else
	wrapString := "";
    fi;

    # ASIC
    if (platform = 0) then
	cmdString := ConcatenationString(cmdString, "-bb -af ", StringDouble("%f", frequency));
    else #FPGA
	cmdString := ConcatenationString(cmdString, "-br ", StringInt(brams), " ", wrapString);
    fi;

    if (format = 0) then
	cmdString := ConcatenationString(cmdString, " -fix ", StringInt(precision));
    else
	cmdString := ConcatenationString(cmdString, " -flt 1 ");
    fi;

    Print(cmdString, "\n");
    Exec(cmdString);

    # Ok, now file is generated.  Time to synthesize.

    if (platform = 0) then
	cmdString := ConcatenationString("cd ", path, " && /afs/scotch/usr/pam/dc_scripts/run-synth.pl ", filename, ".v ", StringDouble("%f", frequency));
    else
	cmdString := ConcatenationString("cd ", path, " && ", _synth_path, " ", filename, ".v ", StringDouble("%f", frequency));
    fi;

    Exec(cmdString);
end;

HDLSynthesize_no_brams := function(t, platform, format, precision, frequency, wrapper, filename)
    local opts, cmdString, path, brams, wrapString;
    
    path := Concat("/tmp/spiral/", String(GetPid()), "/");
    MakeDir(path);
    brams := 0;

    opts := InitStreamHw();
    PrintTo(ConcatenationString(path, filename, ".spl"), HDLPrint(1024, t.dims(), -1, t));

    if (_splhdl_path = "") then
	Error("Error: path to SPLHDL compiler not set: paradigms.stream._splhdl_path is not bound.");
    fi;

    if (_synth_path = "") then
	Error("Error: path to synthesis scripts not set: paradigms.stream._synth_path is not bound.");
    fi;

    cmdString := ConcatenationString(_splhdl_path, " ", path, filename, ".spl -o ", path, filename, ".v -r -l ");

    if (wrapper = 1) then
	wrapString := " -wrap";
    else
	wrapString := "";
    fi;

    # ASIC
    if (platform = 0) then
	cmdString := ConcatenationString(cmdString, "-bb -af ", StringDouble("%f", frequency));
    else #FPGA
	cmdString := ConcatenationString(cmdString, "-br ", StringInt(brams), " ", wrapString);
    fi;

    if (format = 0) then
	cmdString := ConcatenationString(cmdString, " -fix ", StringInt(precision));
    else
	cmdString := ConcatenationString(cmdString, " -flt 1 ");
    fi;

    Print(cmdString, "\n");
    Exec(cmdString);

    # Ok, now file is generated.  Time to synthesize.

    if (platform = 0) then
	cmdString := ConcatenationString("cd ", path, " && /afs/scotch/usr/pam/dc_scripts/run-synth.pl ", filename, ".v ", StringDouble("%f", frequency));
    else
	cmdString := ConcatenationString("cd ", path, " && ", _synth_path, " ", filename, ".v ", StringDouble("%f", frequency));
    fi;

    Exec(cmdString);
end;

HDLSynthesize_x5 := function(t, platform, format, precision, frequency, wrapper, filename)
    local opts, cmdString, path, brams, wrapString;
    
    path := Concat("/tmp/spiral/", String(GetPid()), "/");
    MakeDir(path);
    brams := 1440;

    opts := InitStreamHw();
    PrintTo(ConcatenationString(path, filename, ".spl"), HDLPrint(1024, t.dims(), -1, t));

    if (_splhdl_path = "") then
	Error("Error: path to SPLHDL compiler not set: paradigms.stream._splhdl_path is not bound.");
    fi;

    if (_synth_path = "") then
	Error("Error: path to synthesis scripts not set: paradigms.stream._synth_path is not bound.");
    fi;

    cmdString := ConcatenationString(_splhdl_path, " ", path, filename, ".spl -o ", path, filename, ".v -r -l ");

    if (wrapper = 1) then
	wrapString := " -wrap";
    else
	wrapString := "";
    fi;

    # ASIC
    if (platform = 0) then
	cmdString := ConcatenationString(cmdString, "-bb -af ", StringDouble("%f", frequency));
    else #FPGA
	cmdString := ConcatenationString(cmdString, "-br ", StringInt(brams), " ", wrapString);
    fi;

    if (format = 0) then
	cmdString := ConcatenationString(cmdString, " -fix ", StringInt(precision));
    else
	cmdString := ConcatenationString(cmdString, " -flt 1 ");
    fi;

    Print(cmdString, "\n");
    Exec(cmdString);
    # Ok, now file is generated.  Time to synthesize.

    if (platform = 0) then
	cmdString := ConcatenationString("cd ", path, " && /afs/scotch/usr/pam/dc_scripts/run-synth.pl ", filename, ".v ", StringDouble("%f", frequency));
    else
	cmdString := ConcatenationString("cd ", path, " && ", _synth_path, " ", filename, ".v ", StringDouble("%f", frequency));
    fi;

    Exec(cmdString);
end;

HDLSynthesize_no_brams_x5 := function(t, platform, format, precision, frequency, wrapper, filename)
    local opts, cmdString, path, brams, wrapString;
    
    path := Concat("/tmp/spiral/", String(GetPid()), "/");
    MakeDir(path);
    brams := 0;

    opts := InitStreamHw();
    PrintTo(ConcatenationString(path, filename, ".spl"), HDLPrint(1024, t.dims(), -1, t));

    if (_splhdl_path = "") then
	Error("Error: path to SPLHDL compiler not set: paradigms.stream._splhdl_path is not bound.");
    fi;

    if (_synth_path = "") then
	Error("Error: path to synthesis scripts not set: paradigms.stream._synth_path is not bound.");
    fi;

    #cmdString := ConcatenationString(_splhdl_path, " ", path, filename, ".spl -stb -o ", path, filename, ".v -r -l ");
    cmdString := ConcatenationString(_splhdl_path, " ", path, filename, ".spl -o ", path, filename, ".v -r -l ");

    if (wrapper = 1) then
	wrapString := " -wrap";
    else
	wrapString := "";
    fi;

    # ASIC
    if (platform = 0) then
	cmdString := ConcatenationString(cmdString, "-bb -af ", StringDouble("%f", frequency));
    else #FPGA
	cmdString := ConcatenationString(cmdString, "-br ", StringInt(brams), " ", wrapString);
    fi;

    if (format = 0) then
	cmdString := ConcatenationString(cmdString, " -fix ", StringInt(precision));
    else
	cmdString := ConcatenationString(cmdString, " -flt 1 ");
    fi;

    Print(cmdString, "\n");
    Exec(cmdString);
    # Ok, now file is generated.  Time to synthesize.

    if (platform = 0) then
	cmdString := ConcatenationString("cd ", path, " && /afs/scotch/usr/pam/dc_scripts/run-synth.pl ", filename, ".v ", StringDouble("%f", frequency));
    else
	cmdString := ConcatenationString("cd ", path, " && ", _synth_path, " ", filename, ".v ", StringDouble("%f", frequency));
    fi;

    Exec(cmdString);
end;

HDLSynthesize_stb := function(t, platform, format, precision, frequency, wrapper, filename)
    local opts, cmdString, path, brams, wrapString;
    
    path := Concat("/tmp/spiral/", String(GetPid()), "/");
    MakeDir(path);
    brams := 1440;

    opts := InitStreamHw();
    PrintTo(ConcatenationString(path, filename, ".spl"), HDLPrint(1024, t.dims(), -1, t));

    if (_splhdl_path = "") then
	Error("Error: path to SPLHDL compiler not set: paradigms.stream._splhdl_path is not bound.");
    fi;

    if (_synth_path = "") then
	Error("Error: path to synthesis scripts not set: paradigms.stream._synth_path is not bound.");
    fi;

    cmdString := ConcatenationString(_splhdl_path, " ", path, filename, ".spl -pp -stb -o ", path, filename, ".v -r -l ");

    if (wrapper = 1) then
	wrapString := " -wrap";
    else
	wrapString := "";
    fi;

    # ASIC
    if (platform = 0) then
	cmdString := ConcatenationString(cmdString, "-bb -af ", StringDouble("%f", frequency));
    else #FPGA
	cmdString := ConcatenationString(cmdString, "-br ", StringInt(brams), " ", wrapString);
    fi;

    if (format = 0) then
	cmdString := ConcatenationString(cmdString, " -fix ", StringInt(precision));
    else
	cmdString := ConcatenationString(cmdString, " -flt 1 ");
    fi;

    Print(cmdString, "\n");
    Exec(cmdString);

    # Ok, now file is generated.  Time to synthesize.

    if (platform = 0) then
	cmdString := ConcatenationString("cd ", path, " && /afs/scotch/usr/pam/dc_scripts/run-synth.pl ", filename, ".v ", StringDouble("%f", frequency));
    else
	cmdString := ConcatenationString("cd ", path, " && ", _synth_path, " ", filename, ".v ", StringDouble("%f", frequency));
    fi;

    Exec(cmdString);
end;



# HDLSynthesize2(transform, platform, format, precision, genfrequency, synthfrequency, filename)
# platform: 0 for 65nm ASIC, 1 for FPGA
# format: 0 for fixed point, 1 for floating point
# precision: # bits precision, ignored for floating point
# wrapper: 0 for normal mode, 1 to use a wrapper for synthesizing wide designs on FPGA
HDLSynthesize2 := function(t, platform, format, precision, genfrequency, synthfrequency, wrapper, filename)
    local opts, cmdString, path, brams, wrapString;

    path := "/afs/scotch/usr/pam/res/"; 
    brams := 256;

    opts := InitStreamHw();
    PrintTo(ConcatenationString(path, filename, ".spl"), HDLPrint(1024, t.dims(), -1, t));

    cmdString := ConcatenationString("/afs/scotch/usr/pam/splhdl/src/splhdl ", path, filename, ".spl -o ", path, filename, ".v -r -l ");

    if (wrapper = 1) then
	wrapString := " -wrap";
    else
	wrapString := "";
    fi;

    # ASIC
    if (platform = 0) then
	cmdString := ConcatenationString(cmdString, "-bb -af ", StringDouble("%f", genfrequency));
    else #FPGA
	cmdString := ConcatenationString(cmdString, "-br ", StringInt(brams), " ", wrapString);
    fi;

    if (format = 0) then
	cmdString := ConcatenationString(cmdString, " -fix ", StringInt(precision));
    else
	cmdString := ConcatenationString(cmdString, " -flt 1 ");
    fi;

    Print(cmdString, "\n");
    Exec(cmdString);

    # Ok, now file is generated.  Time to synthesize.

    if (platform = 0) then
	cmdString := ConcatenationString("cd ", path, " && /afs/scotch/usr/pam/dc_scripts/run-synth.pl ", filename, ".v ", StringDouble("%f", synthfrequency));
    else
	cmdString := ConcatenationString("cd ", path, " && /afs/scotch/usr/pam/synth/synthfpga.pl ", filename, ".v ", StringDouble("%f", synthfrequency));
    fi;

    Exec(cmdString);
end;

# HDLGen(transform, platform, format, precision, frequency, filename)
# platform: 0 for 65nm ASIC, 1 for FPGA
# format: 0 for fixed point, 1 for floating point, 2 for double
# precision: # bits precision, ignored for floating point
# wrapper: 0 for normal mode, 1 to use a wrapper for synthesizing wide designs on FPGA
HDLGen := function(t, platform, format, precision, frequency, wrapper, filename)
    local opts, cmdString, path, brams, wrapString;

    path := Concat("/tmp/spiral/", String(GetPid()), "/");
    MakeDir(path);
    brams := -1;

    opts := InitStreamHw();
    PrintTo(ConcatenationString(path, filename, ".spl"), HDLPrint(1024, t.dims(), -1, t));

    if (_splhdl_path = "") then
	Error("Error: path to SPLHDL compiler not set: paradigms.stream._splhdl_path is not bound.");
    fi;

    cmdString := ConcatenationString(_splhdl_path, " ", path, filename, ".spl -o ", path, filename, ".v -r -l ");

    if (wrapper = 1) then
	wrapString := " -wrap";
    else
	wrapString := "";
    fi;

    # ASIC
    if (platform = 0) then
	cmdString := ConcatenationString(cmdString, "-bb -af ", StringDouble("%f", frequency));
    elif (brams = -1) then
	cmdString := ConcatenationString(cmdString, " ", wrapString);
    else
	cmdString := ConcatenationString(cmdString, "-br ", StringInt(brams), " ", wrapString);
    fi;

    if (format = 0) then
	cmdString := ConcatenationString(cmdString, " -fix ", StringInt(precision));
    elif (format = 1) then
	cmdString := ConcatenationString(cmdString, " -flt 1 ");
    else
	cmdString := ConcatenationString(cmdString, " -flt 2 ");
    fi;

    Print(cmdString, "\n");
    Exec(cmdString);
	return path;
end;

sortVecAlg1 := function(size, w)
   local opts, t, r, s, s2, s3;
   opts := InitStreamHw();
   opts.breakdownRules.SortVec := [Sort_Stream_Vec];

   t := SortVec(size).withTags([AStream(w)]);
   r := RandomRuleTree(t, opts);
   s := SumsRuleTreeStrategy(r, SumStreamStrategy, opts);
   s2 :=  ApplyStrategy(s, StreamStrategy, UntilDone, opts);
   s3 := s2.createCode();
   return s3;
end;

sortVecAlg5 := function(size, w, d)
   local opts, t, r, s, s2, s3;
   opts := InitStreamHw();
   opts.breakdownRules.SortVec := [Sort_Stream4_Vec];
   Sort_Stream4_Vec.depth := d;

   t := SortVec(size).withTags([AStream(w)]);
   r := RandomRuleTree(t, opts);
   s := SumsRuleTreeStrategy(r, SumStreamStrategy, opts);
   s2 :=  ApplyStrategy(s, StreamStrategy, UntilDone, opts);
   s3 := s2.createCode();
   return s3;
end;

sortAlg1 := function(size, w)
   local opts, t, r, s, s2, s3;
   opts := InitStreamHw();
   opts.breakdownRules.Sort := [Sort_Stream];

   t := Sort(size).withTags([AStream(w)]);
   r := RandomRuleTree(t, opts);
   s := SumsRuleTreeStrategy(r, SumStreamStrategy, opts);
   s2 :=  ApplyStrategy(s, StreamStrategy, UntilDone, opts);
   s3 := s2.createCode();
   return s3;
end;

sortAlg6 := function(size, w)
   local opts, t, r, s, s2, s3;
   opts := InitStreamHw();
   opts.breakdownRules.Sort := [Sort_Stream6];

   t := Sort(size).withTags([AStream(w)]);
   r := RandomRuleTree(t, opts);
   s := SumsRuleTreeStrategy(r, SumStreamStrategy, opts);
   s2 :=  ApplyStrategy(s, StreamStrategy, UntilDone, opts);
   s3 := s2.createCode();
   return s3;
end;

sortAlg2 := function(size, w, d)
   local opts, t, r, s, s2, s3;
   opts := InitStreamHw();
   opts.breakdownRules.Sort := [Sort_Stream_Iter];
   Sort_Stream_Iter.depth_params := d;

   t := Sort(size).withTags([AStream(w)]);
   r := RandomRuleTree(t, opts);
   s := SumsRuleTreeStrategy(r, SumStreamStrategy, opts);
   s2 :=  ApplyStrategy(s, StreamStrategy, UntilDone, opts);
   s3 := s2.createCode();
   return s3;
end;

sortAlg5 := function(size, w, d)
   local opts, t, r, s, s2, s3;
   opts := InitStreamHw();
   opts.breakdownRules.Sort := [Sort_Stream5];
   Sort_Stream5.depth_params := d;

   t := Sort(size).withTags([AStream(w)]);
   r := RandomRuleTree(t, opts);
   s := SumsRuleTreeStrategy(r, SumStreamStrategy, opts);
   s2 :=  ApplyStrategy(s, StreamStrategy, UntilDone, opts);
   s3 := s2.createCode();
   return s3;
end;

sortAlg3 := function(size, w, d_out, d_in)
   local opts, t, r, s, s2, s3;
   opts := InitStreamHw();
   opts.breakdownRules.Sort := [Sort_Stream3];
   Sort_Stream3.depth_out := d_out;
   Sort_Stream3.depth_in := d_in;

   t := Sort(size).withTags([AStream(w)]);
   r := RandomRuleTree(t, opts);
   s := SumsRuleTreeStrategy(r, SumStreamStrategy, opts);
   s2 :=  ApplyStrategy(s, StreamStrategy, UntilDone, opts);
   s3 := s2.createCode();
   return s3;
end;

sortAlg4 := function(size, w, d)
   local opts, t, r, s, s2, s3;
   opts := InitStreamHw();
   opts.breakdownRules.Sort := [Sort_Stream4];
   Sort_Stream4.depth := d;

   t := Sort(size).withTags([AStream(w)]);
   r := RandomRuleTree(t, opts);
   s := SumsRuleTreeStrategy(r, SumStreamStrategy, opts);
   s2 :=  ApplyStrategy(s, StreamStrategy, UntilDone, opts);
   s3 := s2.createCode();
   return s3;
end;

sortLinear := function(size,w)
   local opts, t, r, s, s2, s3;
   opts := InitStreamHw();
   opts.breakdownRules.Sort := [Linear_Sort];

   t := Sort(size).withTags([AStream(w)]);
   r := RandomRuleTree(t, opts);
   s := SumsRuleTreeStrategy(r, SumStreamStrategy, opts);
   s2 :=  ApplyStrategy(s, StreamStrategy, UntilDone, opts);
   s3 := s2.createCode();
   return s3;
end;



runThisOFDMTx := function(size, radix, w, bits, freqs)
   local t, i, name;

   t := streamIDFTUnroll(size, radix, 2*w);
   for i in [1..Length(bits)] do
       name := ConcatenationString("ifft", String(size), "w", String(w), "b", String(bits[i]), "f", String(freqs[i]));
       HDLGen(t, 0, 0, bits[i], freqs[i], 0, name);
    od;
end;

runThisOFDMRx := function(size, radix, w, bits, freqs)
   local t, i, name;

   t := streamDFTUnroll(size, radix, 2*w);
   for i in [1..Length(bits)] do
       name := ConcatenationString("fft", String(size), "w", String(w), "b", String(bits[i]), "f", String(freqs[i]));
       HDLGen(t, 0, 0, bits[i], freqs[i], 0, name);
    od;
end;
    
runJobsOFDMTx := function()
    runThisOFDMTx(128, 16, 128,[12, 12, 14, 14, 14, 14, 15], [250, 167.2, 125, 100, 83.6, 71.1, 62.5]);
    runThisOFDMTx(128, 16, 64, [12, 12, 14, 14, 14, 14, 15], [500, 334.4, 250, 200, 167.2, 142.2, 125]);
    runThisOFDMTx(128, 16, 32, [12, 12, 14, 14, 14, 14, 15], [1000, 668.8, 500, 400, 334.4, 284.4, 250]);
    runThisOFDMTx(128, 16, 16, [12, 14, 14, 14, 14, 15], [1337.5, 1000, 800, 668.8, 568.8, 500]);
    runThisOFDMTx(128, 8, 8, [14, 14, 15], [1337.5, 1137.5, 1000]);
end;

runJobsOFDMRx := function()
    runThisOFDMRx(128, 16, 128, [14, 14, 15, 15, 16, 16, 16], [250, 167.2, 125, 100, 83.6, 71.1, 62.5]);
    runThisOFDMRx(128, 16, 64, [14, 14, 15, 15, 16, 16, 16], [500, 334.4, 250, 200, 167.2, 142.2, 125]);
    runThisOFDMRx(128, 16, 32, [14, 14, 15, 15, 16, 16, 16], [1000, 668.8, 500, 400, 334.4, 284.4, 250]);
    runThisOFDMRx(128, 16, 16, [14, 15, 15, 16, 16, 16], [1337.5, 1000, 800, 668.8, 568.8, 500]);
    runThisOFDMRx(128, 8, 8, [16, 16, 16], [1337.5, 1137.5, 1000]);
end;

# I needed to increase the pipelining of mults for the fast cores.
reRunOFDMTx := function()
    runThisOFDMTx(128, 16, 32, [12], [1000]);
    runThisOFDMTx(128, 16, 16, [12], [1337.5]);
    runThisOFDMTx(128, 16, 16, [14], [1000]);
    runThisOFDMTx(128, 8, 8, [14], [1337.5]);
    runThisOFDMTx(128, 8, 8, [14], [1137.5]);
    runThisOFDMTx(128, 8, 8, [15], [1000]);
end;

fourStepTest := (n, u) >> let(
    t0 := Tensor(L(n^2/u, n), I(u)),
    t1 := Tensor(I(n/u), L(n*u, n) * Tensor(I(u), DFT(n)) * (Tensor(L(n,u), I(u)))),
    t2 := Tensor(Tensor(I(n/u), L(n, n/u)), I(u)),
    t3 := Tensor(L(n*n/u, n), I(u)),
    p  := Tensor(L(n*n/u, n), I(u)) * Tensor(I(n*n/(u*u)), L(u*u,u)) * Tensor(I(n/u), L(n, n/u), I(u)) * Tensor(L(n*n/u, n), I(u)),
    t4 := TransposedSPL(p) * Diag(Tw1(n*n,n,1)) * p,
    t5 := Tensor(I(n/u), L(n*u, n) * Tensor(I(u), DFT(n)) * L(n*u,u)),
    t6 := Tensor(L(n*n/u, n/u), I(u)),
    t0*t1*t2*t3*t4*t5*t6
);

fourStepTest2 := (n, u) >> let(
    t0 := Tensor(L(n^3/u, n), I(u)),
    t1 := Tensor(I(n*n/u), L(n*u, n) * Tensor(I(u), DFT(n)) * (Tensor(L(n,u), I(u)))),
    t2 := Tensor(Tensor(I(n*n/u), L(n, n/u)), I(u)),
    t3 := Tensor(L(n*n*n/u, n), I(u)),
    p2 := Tensor(L(n*n*n/u, n), I(u)) * Tensor(I(n*n*n/(u*u)), L(u*u,u)) * Tensor(I(n*n/u), L(n, n/u), I(u)) * Tensor(L(n*n*n/u, n), I(u)),
    t4 := TransposedSPL(p2) * Diag(Tw1(n*n*n,n*n,1)) * p2,
    t5 := Tensor(I(n*n/u), L(n*u, n) * Tensor(I(u), DFT(n)) * L(n*u,u)),
    t6 := Tensor(L(n*n*n/u, n*n/u), I(u)),
    t7 := Tensor(L(n*n,n), I(n)),
    t8 := Tensor(L(n*n*n/u, n), I(u)),
    p  := Tensor(L(n*n, n), I(n)) * Tensor(L(n*n*n/u, n), I(u)),
    t9 := TransposedSPL(p) * Tensor(Diag(Tw1(n*n,n,1)), I(n)) * p,
    t10 := Tensor(I(n*n/u), L(n*u, n) * Tensor(I(u), DFT(n)) * L(n*u,u)),
    t11 := Tensor(L(n*n*n/u, n*n/u), I(u)),
    t0 * t1 * t2 * t3 * t4 * t5 * t6 * t7 * t8 * t9 * t10 * t11
);

runRachid2 := function(size, radix, w, bits)
   local t, i, name;

   t := streamDFTUnroll(size, radix, 2*w);
   for i in [1..Length(bits)] do
       name := ConcatenationString("fft", String(size), "-w", String(w), "-b", String(bits[i]));
       HDLGen(t, 1, 0, bits[i], 0, 0, name);
    od;
end;

runRachid3 := function(size, radix, w, bits)
   local t, i, name;

   t := streamIDFTUnroll(size, radix, 2*w);
   for i in [1..Length(bits)] do
       name := ConcatenationString("ifft", String(size), "-w", String(w), "-b", String(bits[i]));
       HDLGen(t, 1, 0, bits[i], 0, 0, name);
    od;
end;

runOFDMExp := function()
HDLGen(streamDFTUnroll(64, 64, 128), 0, 0, 14, 250, 0, "fft64w64b14f250");
HDLGen(streamDFTUnroll(64, 64, 128), 0, 0, 14, 500, 0, "fft64w64b14f500");
HDLGen(streamDFTUnroll(64, 32, 64), 0, 0, 14, 1000, 0, "fft64w32b14f1000");
HDLGen(streamDFTUnroll(128, 128, 256), 0, 0, 14, 250, 0, "fft128w128b14f250");
HDLGen(streamDFTUnroll(128, 64, 128), 0, 0, 14, 500, 0, "fft128w64b14f500");
HDLGen(streamDFTUnroll(128, 32, 64), 0, 0, 14, 1000, 0, "fft128w32b14f1000");
HDLGen(streamDFTUnroll(256, 128, 256), 0, 0, 14, 250, 0, "fft256w128b14f250");
HDLGen(streamDFTUnroll(256, 64, 128), 0, 0, 14, 500, 0, "fft256w64b14f500");
HDLGen(streamDFTUnroll(256, 32, 64), 0, 0, 14, 1000, 0, "fft256w32b14f1000");
HDLGen(streamDFTUnroll(512, 128, 256), 0, 0, 14, 250, 0, "fft512w128b14f250");
HDLGen(streamDFTUnroll(512, 64, 128), 0, 0, 14, 500, 0, "fft512w64b14f500");
HDLGen(streamDFTUnroll(512, 32, 64), 0, 0, 14, 1000, 0, "fft512w32b14f1000");
HDLGen(streamDFTUnroll(1024, 128, 256), 0, 0, 14, 250, 0, "fft1024w128b14f250");
HDLGen(streamDFTUnroll(1024, 64, 128), 0, 0, 14, 500, 0, "fft1024w64b14f500");
HDLGen(streamDFTUnroll(1024, 32, 64), 0, 0, 14, 1000, 0, "fft1024w32b14f1000");
HDLGen(streamDFTUnroll(2048, 128, 256), 0, 0, 14, 250, 0, "fft2048w128b14f250");
HDLGen(streamDFTUnroll(2048, 64, 128), 0, 0, 14, 500, 0, "fft2048w64b14f500");
HDLGen(streamDFTUnroll(2048, 32, 64), 0, 0, 14, 1000, 0, "fft2048w32b14f1000");
end;

#runForRachid := function()

#end;

runRachid := function()
    runRachid2(256, 16, 128, [8, 10, 12]);
    runRachid2(512, 16, 128, [8, 10, 12]);
    runRachid3(256, 16, 128, [12, 13, 14, 15, 16, 17, 18]);
    runRachid3(512, 16, 128, [12, 13, 14, 15, 16, 17, 18]);
end;

runRachidECOC2012 := function()
    runRachid3(32, 32, 32, [10, 11, 12]);
    runRachid3(32, 32, 128, [10, 11, 12]);
    runRachid3(64, 32, 64, [10, 11, 12]);
    runRachid3(64, 32, 128, [10, 11, 12]);
end;

# genBRAMPermMem(perm, w, format, bits, name)
#     Will generate Verilog for a Permutation Memory, performing permutation perm
#     with streaming width w.  The file will reside in /tmp/spiral/[PID] where [PID]
#     is Spiral's process ID.
#
#     format: 0 for fixed point, 1 for single precision floating point, 2 for double
#     bits: number of bits of precision, ignored for floating point
#
#     Note that if you want the permutation to take complex data, you will need to wrap
#     perm in TRC( ), and that your streaming width is given in *real* words.
#
#    So, to generate L(32,2) on complex data with streaming width 2 complex (= 4 real),
#    16 bit fixed point data, storing in file.v:
#        genBRAMPermMem(TRC(TPrm(TL(32,2,1,1))), 4, 0, 16, "file");

genBRAMPermMem := function(perm, w, format, bits, name)
   local opts,path;
   opts := InitStreamUnrollHw();

   # Rather than having separate functions in Spiral for streaming perms and streaming perm memories,
   # just adapt the BRAMPerm to generate a memory by making this change.
   BRAMPerm.print := (self, i, is) >> Print(self.name, "Mem(", BitMatrixToInts(self._children[1]), ", ", BitMatrixToInts(self._children[2]), ", ", self.streamSize, ")");

   
   path := HDLGen(streamGen(perm.withTags([AStream(w)]), opts), 1, format, bits, 0, 0, name);



   # Put BRAMPerm.print back to normal.
   BRAMPerm.print      := (self,i,is) >> Print(self.name, "(", BitMatrixToInts(self._children[1]), ", ", BitMatrixToInts(self._children[2]), ", ", self.streamSize, ")");
	return path;
end;

runBerkin := function()
   local opts;
   opts := InitStreamUnrollHw();
   BRAMPerm.print      := (self,i,is) >> Print(self.name, "Mem(", BitMatrixToInts(self._children[1]), ", ", BitMatrixToInts(self._children[2]), ", ", self.streamSize, ")");

    HDLGen(streamGen(TRC(TPrm(TL(512, 32, 1, 1))).withTags([AStream(4)]), opts), 1, 2, 0, 0, 0, "L512_32-dbl-w2");
    HDLGen(streamGen(TRC(TPrm(TL(1024, 64, 1, 1))).withTags([AStream(4)]), opts), 1, 2, 0, 0, 0, "L1024_64-dbl-w2");
    HDLGen(streamGen(TRC(TPrm(TL(2048, 128, 1, 1))).withTags([AStream(4)]), opts), 1, 2, 0, 0, 0, "L2048_128-dbl-w2");
    HDLGen(streamGen(TRC(TPrm(TL(4096, 256, 1, 1))).withTags([AStream(4)]), opts), 1, 2, 0, 0, 0, "L4096_256-dbl-w2");
    HDLGen(streamGen(TRC(TPrm(TL(8192, 512, 1, 1))).withTags([AStream(4)]), opts), 1, 2, 0, 0, 0, "L8192_512-dbl-w2");
    HDLGen(streamGen(TRC(TPrm(TL(16384, 1024, 1, 1))).withTags([AStream(4)]), opts), 1, 2, 0, 0, 0, "L16384_1024-dbl-w2");
    HDLGen(streamGen(TRC(TPrm(TL(32768, 2048, 1, 1))).withTags([AStream(4)]), opts), 1, 2, 0, 0, 0, "L32768_2048-dbl-w2");
    HDLGen(streamGen(TRC(TPrm(TL(65536, 4096, 1, 1))).withTags([AStream(4)]), opts), 1, 2, 0, 0, 0, "L65536_4096-dbl-w2");
    HDLGen(streamGen(TRC(TPrm(TL(32, 16, 1, 16))).withTags([AStream(4)]), opts), 1, 2, 0, 0, 0, "L32_16_t16-dbl-w2");
    HDLGen(streamGen(TRC(TPrm(TL(64, 16, 1, 16))).withTags([AStream(4)]), opts), 1, 2, 0, 0, 0, "L64_16_t16-dbl-w2");
    HDLGen(streamGen(TRC(TPrm(TL(128, 16, 1, 16))).withTags([AStream(4)]), opts), 1, 2, 0, 0, 0, "L128_16_t16-dbl-w2");
    HDLGen(streamGen(TRC(TPrm(TL(256, 16, 1, 16))).withTags([AStream(4)]), opts), 1, 2, 0, 0, 0, "L256_16_t16-dbl-w2");
    HDLGen(streamGen(TRC(TPrm(TL(512, 16, 1, 16))).withTags([AStream(4)]), opts), 1, 2, 0, 0, 0, "L512_16_t16-dbl-w2");
    HDLGen(streamGen(TRC(TPrm(TL(1024, 16, 1, 16))).withTags([AStream(4)]), opts), 1, 2, 0, 0, 0, "L1024_16_t16-dbl-w2");
    HDLGen(streamGen(TRC(TPrm(TL(2048, 16, 1, 16))).withTags([AStream(4)]), opts), 1, 2, 0, 0, 0, "L2048_16_t16-dbl-w2");
    HDLGen(streamGen(TRC(TPrm(TL(4096, 16, 1, 16))).withTags([AStream(4)]), opts), 1, 2, 0, 0, 0, "L4096_16_t16-dbl-w2");

# NB: TL is much better in the TPrm than Tensor(T .. .because TL has a fast function to turn it to a bit matrix, while
# Tensor has to do it by brute force.

   BRAMPerm.print      := (self,i,is) >> Print(self.name, "(", BitMatrixToInts(self._children[1]), ", ", BitMatrixToInts(self._children[2]), ", ", self.streamSize, ")");
   HDLGen(streamDFTUnroll(32,2,4), 1, 2, 0, 0, 0, "dft32-dbl-w2");
   HDLGen(streamDFTUnroll(64,2,4), 1, 2, 0, 0, 0, "dft64-dbl-w2");
   HDLGen(streamDFTUnroll(128,2,4), 1, 2, 0, 0, 0, "dft128-dbl-w2");
   HDLGen(streamDFTUnroll(256,2,4), 1, 2, 0, 0, 0, "dft256-dbl-w2");
   HDLGen(streamDFTUnroll(512,2,4), 1, 2, 0, 0, 0, "dft512-dbl-w2");
   HDLGen(streamDFTUnroll(1024,2,4), 1, 2, 0, 0, 0, "dft1024-dbl-w2");
   HDLGen(streamDFTUnroll(2048,2,4), 1, 2, 0, 0, 0, "dft2048-dbl-w2");
   HDLGen(streamDFTUnroll(4096,2,4), 1, 2, 0, 0, 0, "dft4096-dbl-w2");
end;

# HDLSimPerm(transform, format, precision, filename)
# format: 0 for fixed point, 1 for floating point, 2 for double
# precision: # bits precision, ignored for floating point
HDLSimPerm := function(t, format, precision, filename)
    local opts, cmdString, path, brams, wrapString;

    path := Concat("/tmp/spiral/", String(GetPid()), "/");
    Exec("rm -f "::path::"log");
    MakeDir(path);
    brams := -1;

    opts := InitStreamHw();
    PrintTo(ConcatenationString(path, filename, ".spl"), HDLPrint(1024, t.dims(), -1, t));

    if (_splhdl_path = "") then
	Error("Error: path to SPLHDL compiler not set: paradigms.stream._splhdl_path is not bound.");
    fi;

    cmdString := ConcatenationString(_splhdl_path, " ", path, filename, ".spl -o ", path, filename, ".v -ptb");

    if (format = 0) then
	cmdString := ConcatenationString(cmdString, " -fix ", StringInt(precision));
    elif (format = 1) then
	cmdString := ConcatenationString(cmdString, " -flt 1 ");
    else
	cmdString := ConcatenationString(cmdString, " -flt 2 ");
    fi;

    Print(cmdString, "\n");
    Exec(cmdString);

    cmdString := "cd "::path::" && iverilog "::filename::".v && vvp a.out > out";
    Exec(cmdString);

    return ReadVal(path::"/log");

end;


# Generate a random full-rank bit matrix.  I.e., a bit matrix that
# represents a permutation.
RandomBitMatrix := function(size)
    local res, i, j, trow, rank;
    rank := 0;
    
    while (rank < size) do
       res := [];
       for i in [1..size] do
          trow := [];
          for j in [1..size] do
             Append(trow, [Random(GF(2))]);
          od;
          Append(res, [trow]);
       od;
       rank := Rank(res);
   od;

    return res;
end;
   
RandomBitPerm := function(size, width)
   # the L is a hack, but should't matter because toAMat in StreamPermBit does not use LinearBits.child(2).
   return StreamPerm([LinearBits(RandomBitMatrix(size), L(4,2))], 1, width, 0);
end;

simRandomPerm := function(n, w)
   local p, t, c, res;
   p := RandomBitPerm(Log2Int(n), w);
   c := p.createCode();    
   res := 1;

   res := HDLSimPerm(c, 0, 16, "test");
   
   if (res <> 0) then
       PrintBitMatrix(p.child(1)[1].child(1));
   fi;

   return res;  
end;

testPerms := function(n, w)
   local res, count, it;
   count := 100;
   res := 0;
   for it in [1..count] do
      Print(it);
      res := simRandomPerm(n, w);
      if (res <> 0) then
	  Error();
      fi;
   od;
end;

simPerm := function(t)
   local res, r, s, opts, s2;
   opts := InitStreamUnrollHw();
   r := RandomRuleTree(t, opts);
   s := SumsRuleTreeStrategy(r, SumStreamStrategy, opts);
   s2 :=  s.createCode();

   res := 1;
   res := HDLSimPerm(s2, 0, 16, "test");
   
   Print("Errors: "::String(res) :: "\n");

   if (res <> 0) then
       Print(t);
   fi;

   return res;  
end;

randomStridePerm := function(maxN, maxW)
   local n, ln, w, lw, ls, s;
   ln := Random([2..Log2Int(maxN)]);
   n := 2^ln;
   ls := Random([1..ln-1]);
   s := 2^ls;
   lw := Random([1..Min2(Log2Int(maxW), ln-1)]);
   w := 2^lw;
   
   Print(" L("::String(n)::", "::String(s)::"), w="::String(w)::"   ");
   return TPrm(TL(n, s)).withTags([AStream(w)]);

end;

randomVPerm := function(maxN, maxW)
   local n, ln, w, lw;
   ln := Random([2..Log2Int(maxN)]);
   n := 2^ln;
   lw := Random([1..Min2(Log2Int(maxW), ln-1)]);
   w := 2^lw;
   
   Print(" V("::String(n)::"), w="::String(w)::"   ");
   return TPrm(SortIJPerm(n)).withTags([AStream(w)]);

end;

testStridePerms := function(maxN, maxW)
   local res, count, it;
   count := 100;
   res := 0;
   for it in [1..count] do
      Print(it);
      res := simPerm(randomStridePerm(maxN, maxW));
      if (res <> 0) then
	  Error();
      fi;
   od;
end;

testVPerms := function(maxN, maxW)
   local res, count, it;
   count := 100;
   res := 0;
   for it in [1..count] do
      Print(it);
      res := simPerm(randomVPerm(maxN, maxW));
      if (res <> 0) then
	  Error();
      fi;
   od;
end;

#runPermTests := function()
#   testPerms(16,2);
#   testPerms(16,4);
#   testPerms(16,8);
#   testPerms(32,2);
#   testPerms(32,4);
#   testPerms(32,8);
#   testPerms(32,16);
#testPerms(256,2);
#testPerms(256,4);
#testPerms(256,8);
#testPerms(256,16);
#testPerms(256,32);
#testPerms(256,64);
#testPerms(256,128);
#end;

#testStridePerms(256,64)
#testVPerm



# To compile normal C code:
# opts := SpiralDefaults;
# t := WHT(8);
# r := RandomRuleTree(t, opts);
# c := CodeRuleTree(r, opts);
# _StandardMeasure(c, opts);
# code ends up in /tmp/spiral/PID
