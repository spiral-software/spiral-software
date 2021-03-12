
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

;; This buffer is for notes you don't want to save, and for Lisp evaluation.
;; If you want to create a file, visit that file with C-x C-f,
;; then enter the text in that file's own buffer.

n := 20;

opts := CopyFields(PRDFTDefaults, rec(
    globalUnrolling := 1000,
    compileStrategy := NoCSE,
    compflags := "-O1"
));

r  := RandomRuleTree(BRDFT3(4,1/16), opts);
rr := RandomRuleTree(RC(SkewDFT(2,1/16)), opts);
s  := SumsSPL(Tensor(I(n), SumsRuleTree(r, opts)), opts);
ss := SumsSPL(Tensor(I(n), SumsRuleTree(rr, opts)), opts);
c  := CodeSums(1000, s, opts);;
cc := CodeSums(1000, ss, opts);;
CMeasure(c, opts);
CMeasure(cc, opts);


r  := RandomRuleTree(BRDFT1(4), opts);
rr := RandomRuleTree(SRDFT1(4), opts);
s  := SumsSPL(Tensor(I(n), SumsRuleTree(r, opts)), opts);
ss := SumsSPL(Tensor(I(n), SumsRuleTree(rr, opts)), opts);
c  := CodeSums(1000, s, opts);;
cc := CodeSums(1000, ss, opts);;
CMeasure(c, opts);
CMeasure(cc, opts);


copts := CopyFields(opts, rec(
   generateComplexCode := true,
   XType := TComplex,
   YType := TComplex,
   dataType := "complex",
   unparser := CMacroUnparser,
   includes := ["<include/complex_gcc_sse3.h>"]
));

opts := copts;

r  := RandomRuleTree(Tensor(I(n), BRDFT3(4,1/16)), opts);
rr := RandomRuleTree(Tensor(I(n), RC(SkewDFT(2,1/16))), opts);

c  := CodeRuleTree(r, opts);;
cc := CodeRuleTree(rr, opts);;
