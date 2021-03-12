
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

scriptWHT := function(val1, val2, val3, val4, val5, val6)
       local opts, t, rt, srt, cs, namec, nametxt, a;

       ImportAll(paradigms.scratchpad);

       opts := ScratchX86Globals.getOpts(val1, val2, val3, val4, val5, val6);

       t := WHT(opts.size).withTags(opts.tags);

       rt := RandomRuleTree(t,opts);

       srt := SumsRuleTree(rt,opts);

       cs := CodeSums(srt,opts);
       
       namec := ConcatenationString("whtfile",String(val1),String(val4),String(val5),".c");
       nametxt := ConcatenationString("whtfile",String(val1),String(val4),String(val5),".txt");
       PrintTo(namec,PrintCode("sub",cs,opts));

       a := CMatrix(cs,opts);
       
       PrintTo(nametxt, a);
end;

scriptDFT := function(val1, val2, val3, val4, val5, val6)
       local opts, t, rt, srt, cs, namec, nametxt, a;

       ImportAll(paradigms.scratchpad);

       opts := ScratchX86Globals.getOpts(val1, val2, val3, val4, val5, val6);

       t := DFT(opts.size).withTags(opts.tags);

       rt := RandomRuleTree(t,opts);

       srt := SumsRuleTree(rt,opts);

       cs := CodeSums(srt,opts);
       
       namec := ConcatenationString("dftfile",String(val1),String(val4),String(val5),".c");
       nametxt := ConcatenationString("dftfile",String(val1),String(val4),String(val5),".txt");
       PrintTo(namec,PrintCode("sub",cs,opts));

       a := CMatrix(cs,opts);
       
       PrintTo(nametxt, a);
end;

scriptRunWHT := function(wht_size, name)
             local opts, t, rt;
             
             ImportAll(paradigms.scratchpad);

             i := 2;

             while (2^i) <= (2^(wht_size+1)) do
                   opts := ScratchX86Globals.getOpts(2^i,1,1,wht_size,'R',true);
                   AppendTo(name,PrintLine("--------------------"));
                   t := WHT(opts.size).withTags(opts.tags);
                   AppendTo(name,PrintLine(t));
                   rt := AllRuleTrees(t,opts);
                   L := Length(rt);
                   AppendTo(name,PrintLine(ConcatenationString("Has ", String(L), " decomposition trees")));
                   AppendTo(name,PrintLine("--------------------"));

                   i := i + 1;
             od;
end;

scriptRunDFT := function(dft_size, name)
             local opts, t, rt;
             
             ImportAll(paradigms.scratchpad);

             i := 2;

             while (2^i) <= (2*dft_size) do
                   opts := ScratchX86Globals.getOpts(2^i,1,1,dft_size,'C',true);
                   AppendTo(name,PrintLine("--------------------"));
                   t := DFT(opts.size).withTags(opts.tags);
                   AppendTo(name,PrintLine(t));
                   rt := AllRuleTrees(t,opts);
                   L := Length(rt); 
                   AppendTo(name,PrintLine(ConcatenationString("Has ", String(L), " decomposition trees")));
                   AppendTo(name,PrintLine("--------------------"));

                   i := i + 1;
             od;
end;

scriptRun := function()
          local wht_size, dft_size, wht_file_name, dft_file_name;

          wht_file_name := "wht.txt";
          dft_file_name := "dft.txt";

          wht_size := 2;

          while wht_size <= 13 do
                scriptRunWHT(wht_size, wht_file_name);
                wht_size := wht_size + 1;
          od;

          dft_size := 4;

          while dft_size <= 8192 do
                scriptRunDFT(dft_size, dft_file_name);
                dft_size := dft_size * 2;
          od;
end;
