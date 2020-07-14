comment("");
comment("Generate an FFT with SIMD Instructions");
comment("");
opts := SIMDGlobals.getOpts(AVX_4x64f);
transform := TRC(DFT(64)).withTags(opts.tags);
ruletree := RandomRuleTree(transform, opts);
icode := CodeRuleTree(ruletree, opts);

##  PrintTo("AVX_DFT64.c", PrintCode("AVX_DFT64", icode, opts));
