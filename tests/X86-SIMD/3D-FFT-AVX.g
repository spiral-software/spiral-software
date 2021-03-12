
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

comment("");
comment("3D FFT with AVX");
opts := SIMDGlobals.getOpts(AVX_4x64f);
transform := TRC(MDDFT([16,16,16])).withTags(opts.tags);
ruletree := RuleTreeMid(transform, opts);
icode := CodeRuleTree(ruletree, opts);

##  PrintTo("AVX_3DDFT16.c", PrintCode("AVX_3DDFT16", icode, opts));
