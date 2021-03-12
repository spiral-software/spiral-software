
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details



Class(SIMD_Macro, SIMD_ISA, rec(
    
    file := "macro",

    commonIncludes := self >> [],
    
    active       := true,
    
    isFixedPoint := false,
    isFloat      := true,
    ctype        := "double",
    bits         := 64,
    useDeref     := false,

    splopts      := rec(precision := "double"),

    duploadn := (y, x, n) -> Error("broadcast is undefined"),
    
    arrayDataModifier := "static ",
    arrayBufModifier  := "static ",

    declareConstants := true,
    threeOps         := true,
    fma		     := true,

    svload  := [],
    svstore := [],

    unparser := MACROUnparser,

    compileStrategy := self >> BaseIndicesCS
                               :: When(self.fma, [DoFMA], [])
                               :: [ MarkDefUse, #
                                    (c, opts) -> CopyPropagate(c, CopyFields(opts, rec(autoinline := true))),
                                    MarkDefUse, #
                                    (c, opts) -> CopyPropagate(c, CopyFields(opts, rec(autoinline := true)))]
                               :: When(self.threeOps, [DoThreeOp], [])
                               :: [ Compile.declareVars,
                                    (c, opts) -> opts.vector.isa.fixProblems(c, opts),
                                    HashConsts ],

));

Class(MACRO_2xf, SIMD_Macro, rec(

    info := "MACRO 2 x floating point",

    countrec := rec(
        ops := [
            [ add, sub ]
             :: [ add_cmd, sub_cmd ], 
	    [ mul, vmulcx_2xf ]
	     :: [ mul_cmd, vmulcx_2xf_cmd ],
            [ fma, fms, nfma ]
             :: [fma_cmd, fms_cmd, nfma_cmd],
            [ vunpacklo_2xf, vunpackhi_2xf, vpacklo_2xf, vpackhi_2xf, vswapcx_2xf ] 
             :: [ vunpacklo_2xf_cmd, vunpackhi_2xf_cmd, vpacklo_2xf_cmd, vpackhi_2xf_cmd, vswapcx_2xf_cmd ],
            [  ],
            [ deref, nth, vload_cmd, vstore_cmd ],
            Value      # Value without [] is a keyword in countOps !!
        ],
        printstrings := ["[adds]", "[mults]", "[fmas]", "[vperms]", "[svldst]", "[vldst]", "[vval]"],
        arithcost := (self, opcount) >> opcount[1]+opcount[2]
    ),
    
    v     := 2,
    t     := TVect(TReal, 2),
    
    instr := [vunpacklo_2xf, vunpackhi_2xf, vpacklo_2xf, vpackhi_2xf],

    includes := self >> self.commonIncludes() :: ["<include/macro_fv2.h>"], 
    
    dupload  := (y, x) -> assign(y, vdup(x, 2)),

    mul_cx  := (self, opts) >> ((y,x,c) -> assign(y, vmulcx_2xf(x,c))),
    swap_cx := (y, x, opts) -> assign(y, vswapcx_2xf(x)),
));


Class(MACRO_4xf, SIMD_Macro, rec(
    
    info := "MACRO 4 x floating point",

    countrec := rec(
        ops := [
            [ add, sub ]
             :: [ add_cmd, sub_cmd ], 
	    [ mul, vmulcx_4xf ]
	     :: [ mul_cmd, vmulcx_4xf_cmd ],
            [ fma, fms, nfma ]
             :: [fma_cmd, fms_cmd, nfma_cmd],
            [   vpacklo_4xf,   vpackhi_4xf,   vpacklo2_4xf,   vpackhi2_4xf, 
              vunpacklo_4xf, vunpackhi_4xf, vunpacklo2_4xf, vunpackhi2_4xf, 
              vswapcx_4xf ]
             :: [   vpacklo_4xf_cmd,   vpackhi_4xf_cmd,   vpacklo2_4xf_cmd,   vpackhi2_4xf_cmd, 
              vunpacklo_4xf_cmd, vunpackhi_4xf_cmd, vunpacklo2_4xf_cmd, vunpackhi2_4xf_cmd, 
              vswapcx_4xf_cmd ],
            [  ],
            [ deref, nth, vload_cmd, vstore_cmd ],
            Value      # Value without [] is a keyword in countOps !!
        ],
        printstrings := ["[adds]", "[mults]", "[fmas]", "[vperms]", "[svldst]", "[vldst]", "[vval]"],
        arithcost := (self, opcount) >> opcount[1]+opcount[2]
    ),
    
    v     := 4,
    t     := TVect(TReal, 4),

    instr    := [vpacklo_4xf,   vpackhi_4xf,   vpacklo2_4xf,   vpackhi2_4xf, 
                 vunpacklo_4xf, vunpackhi_4xf, vunpacklo2_4xf, vunpackhi2_4xf],

    includes := self >> self.commonIncludes() :: ["<include/macro_fv4.h>"], 

    dupload  := (y, x) -> assign(y, vdup(x, 4)),

    mul_cx := (self, opts) >> ((y,x,c) -> assign(y, vmulcx_4xf(x,c))),
    swap_cx := (y, x, opts) -> assign(y, vswapcx_4xf(x)),
));

Class(MACRO_8xf, SIMD_Macro, rec(

    info := "MACRO 8 x floating point",

    countrec := rec(
        ops := [
            [ add, sub ]
             :: [ add_cmd, sub_cmd ], 
	    [ mul, vmulcx_4xf ]
	     :: [ mul_cmd, vmulcx_4xf_cmd ],
            [ fma, fms, nfma ]
             :: [fma_cmd, fms_cmd, nfma_cmd],
            [   vpacklo_8xf,   vpackhi_8xf,   vpacklo2_8xf,   vpackhi2_8xf, 
              vunpacklo_8xf, vunpackhi_8xf, vunpacklo2_8xf, vunpackhi2_8xf,
              vswapcx_8xf, vtrcx_8xf ]
             :: [ vpacklo_8xf_cmd,   vpackhi_8xf_cmd,   vpacklo2_8xf_cmd,   vpackhi2_8xf_cmd, 
              vunpacklo_8xf_cmd, vunpackhi_8xf_cmd, vunpacklo2_8xf_cmd, vunpackhi2_8xf_cmd,
              vswapcx_8xf_cmd, vtrcx_8xf_cmd ],
            [  ],
            [ deref, nth, vload_cmd, vstore_cmd ],
            Value      # Value without [] is a keyword in countOps !!
        ],
        printstrings := ["[adds]", "[mults]", "[fmas]", "[vperms]", "[svldst]", "[vldst]", "[vval]"],
        arithcost := (self, opcount) >> opcount[1]+opcount[2]
    ),
    
    v     := 8,
    t     := TVect(TReal, 8),

    instr := [  vpacklo_8xf,   vpackhi_8xf,   vpacklo2_8xf,   vpackhi2_8xf, 
              vunpacklo_8xf, vunpackhi_8xf, vunpacklo2_8xf, vunpackhi2_8xf,
              vtrcx_8xf ],

    includes := self >> self.commonIncludes() :: ["<include/macro_fv8.h>"], 

    dupload  := (y, x) -> assign(y, vdup(x, 8)),

    mul_cx := (self, opts) >> ((y,x,c) -> assign(y, vmulcx_8xf(x,c))),
    swap_cx := (y, x, opts) -> assign(y, vswapcx_8xf(x)),
));

SIMD_ISA_DB.addISA(MACRO_2xf);
SIMD_ISA_DB.addISA(MACRO_4xf);
SIMD_ISA_DB.addISA(MACRO_8xf);

