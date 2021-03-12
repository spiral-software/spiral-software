
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

#   Debug functions -- where to put?
Import(spl, formgen, code, sigma);
VerifyRuleTreeSPL := rt -> let(t := rt.node, mt := MatSPL(t), mspl := MatSPL(SPLRuleTree(rt)), InfinityNormMat(mt-mspl));
VerifyRuleTreeSigmaSPL := (rt, opts) -> let(t := rt.node, mt := MatSPL(t), ms := MatSPL(SumsRuleTree(rt, opts)), InfinityNormMat(mt-ms));

_THRESHOLD := 1E-5;
_abs := i -> When(IsComplex(i), AbsComplex(i), AbsFloat(i));
ThresholdMat := arg -> MapMat(arg[1], i-> When(_abs(i) > When(Length(arg)=2, arg[2], _THRESHOLD),i , 0));

vm := x -> VisualizeMat(When(IsSPL(x), MatSPL(x), x), "");
vmt := x -> VisualizeMat(ThresholdMat(When(IsSPL(x), MatSPL(x), x)), "");
