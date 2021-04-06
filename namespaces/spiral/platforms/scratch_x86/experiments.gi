
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

ImportAll(paradigms.scratchpad);

doNormalWHT := function(arg)
   local N, t, rt, srt, code, a;
   
   N := When(Length(arg) >=1, arg[1], 2);

   t := WHT(N);
   rt := RandomRuleTree(t, SpiralDefaults);
   srt := SumsRuleTree(rt, SpiralDefaults);
   code := CodeSums(srt, SpiralDefaults);
   a := CMatrix(code, SpiralDefaults);

   return a;
end;

doScratchWHT := function(arg)
   local N, ls, opts, t, rt, srt, code, b;
   N := When(Length(arg) >=1, arg[1], 2);
   ls := When(Length(arg) >=2, 2^arg[2],2);

   opts := ScratchX86CMContext.getOpts(ls,1,1,N);
   t := WHT(opts.size).withTags(opts.tags);
   rt := RandomRuleTree(t, opts);
   srt := SumsRuleTree(rt, opts);
   code := CodeSums(srt, opts);
   b := CMatrix(code, opts);

   return b;
end;

doWHT := function(arg)
    local N, N1, i, j, a, b;

    SpiralDefaults.includes := ["\"scratchc.h\""];    
    SpiralDefaults.profile.makeopts.CFLAGS := "-O2 -Wall -fomit-frame-pointer -msse4.1 -std=gnu99 -static";

    N := When(Length(arg) >= 1, arg[1], 2);
    
    for i in [2..N] do
        a := doNormalWHT(i);
        N1 := i - 1;
        for j in [1..N1] do
            b := doScratchWHT(i,j);
            PrintLine("Size of WHT:", 2^i);
            PrintLine("Scratch buffer size:", 2^j);
            PrintLine(a=b);
            PrintLine("------------------------------------------");
        od;
    od;
end;

