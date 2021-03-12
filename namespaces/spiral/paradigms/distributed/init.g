
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Import(compiler, rewrite, paradigms.common);
ImportAll(paradigms.vector);

Include(tags);
Include(sigmaspl);
Include(breakdown);
Include(dcontainer);
Include(codegen);
Include(unparse);
Include(rewrite);
Include(macros);

Class(ParallelDefaults, SpiralDefaults, rec(
  codegen         := DistCodegen,
  unparser        := DistUnparser,
  globalUnrolling := 2,
  spus            := 2,

#NOTE: This should actually be done at runtime with InitDataType();
# From InitDataType
  isFixedPoint := false,
  bits := 32,
  customDataType := "float",
  XType := TReal,
  YType := TReal,
  TRealCtype := "float",
  includes := [],
));
