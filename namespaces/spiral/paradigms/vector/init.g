
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Import(compiler, paradigms.common, paradigms.smp, spl);

Declare(SIMDGlobals);

CMeasureCountVectOps := (code, opts) -> Sum(code.countOps(opts.vector.isa.countrec));
MeasureCountVectOpsRT := (rt, opts) -> CMeasureCountVectOps(CodeRuleTree(rt, opts), opts);

getV := e-> Cond(IsBound(e.v), e.v,
                 IsBound(e.vlen), e.vlen,
                 IsBound(e.isa), e.isa.v, Error("Can't find vector length in <e>"));

Include(tag);
Include(code);
Include(isa_db);
Include(vwrap);

Load(spiral.paradigms.vector.sigmaspl);
Import(paradigms.vector.sigmaspl);

Include(cvt);

_VHStack := (L, v) -> VTensor(Tensor(Mat([Replicate(Length(L), 1)]), I(Rows(L[1])/v)), v) * 
                      RowDirectSum(0, L); #FF: yet another BIG trouble with DirectSum and these H's !!!

_VVStack := (L, v) -> _VHStack(L, v).transpose();

_VSUM    := (L, v) -> VTensor(Tensor(Mat([Replicate(Length(L), 1)]), I(Rows(L[1])/v)), v) *
                      VStack(L);

Load(spiral.paradigms.vector.bases);
Load(spiral.paradigms.vector.breakdown);
Load(spiral.paradigms.vector.rewrite);

Include(initfuncs);
Include(bench);
Include(experiments);
