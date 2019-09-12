comment("");
comment("Dynamic Programming Search Test");
opts := SpiralDefaults;
transform := DFT(8);
best := DP(transform, rec(), opts);
ruletree := best[1].ruletree;
icode := CodeRuleTree(ruletree, opts);
PrintTo("DP_DFT8", PrintCode("DFT8", icode, opts));
