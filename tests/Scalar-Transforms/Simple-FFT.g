comment("");
comment("Generate a Simple FFT ");
comment("Check CMeasure and CMatrix");
comment("");
opts := SpiralDefaults;
transform := DFT(4);
ruletree := RandomRuleTree(transform, opts);
icode := CodeRuleTree(ruletree, opts);
## PrintCode("DFT4", icode, opts);
PrintTo("DFT4.c", PrintCode("DFT4", icode, opts));
meas := CMeasure(icode, opts);
if (meas <= 0) or (meas >= 1e+100) then
  Print("CMeasure failed -- returned: ", meas, "\n");
  TestFailExit();
fi;
cmat := CMatrix(icode, opts);
if not IsList(cmat) then
  Print("CMatrix failed -- returned: ", cmat, "\n");
  TestFailExit();
fi;
smat := MatSPL(RC(transform));
diff := 1;
diff := cmat - smat;
if not IsList(diff) then
  Print("CMatrix failed -- matrix size mismatch\n");
  TestFailExit();
fi;
inorm := InfinityNormMat(diff);
if inorm > 1e-5 then
  Print("Transform failed -- max diff: ", diff, "\n");
  TestFailExit();
fi;