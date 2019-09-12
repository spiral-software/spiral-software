comment("");
comment("Generate a Simple FFT ");
comment("");
opts := SpiralDefaults;
transform := DFT(4);
ruletree := RandomRuleTree(transform, opts);
icode := CodeRuleTree(ruletree, opts);
## PrintCode("DFT4", icode, opts);
PrintTo("DFT4.c", PrintCode("DFT4", icode, opts));
