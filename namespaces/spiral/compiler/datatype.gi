
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


DataTypes := rec(
    f64re := rec(
    isFixedPoint := false,
    bits := 64,
    customDataType := "double",
    XType := TPtr(TReal),
    YType := TPtr(TReal),
    TRealCtype := "double",
    includes := [ "<include/omega64.h>" ]
    ),

    f64c := rec(
    isFixedPoint := false,
    bits := 128,
    customDataType := "double_cplx",
    XType := TPtr(TComplex),
    YType := TPtr(TComplex),
    TRealCtype := "double",
    TComplexCtype := "_Complex double",
    includes := [ "<include/omega64c.h>" ]
    ),

    f32re := rec(
    isFixedPoint := false,
    bits := 32,
    customDataType := "float",
    XType := TPtr(TReal),
    YType := TPtr(TReal),
    TRealCtype := "float",
    includes := [ "<include/omega32.h>" ],
    valuePostfix := "f"
    ),

    f32c := rec(
    isFixedPoint := false,
    bits := 64,
    customDataType := "float_cplx",
    XType := TPtr(TComplex),
    YType := TPtr(TComplex),
    TRealCtype := "float",
    TComplexCtype := "_Complex float",
    includes := [ "<include/omega32c.h>" ],
    valuePostfix := "f"
    ),

    i32re := rec(
    isFixedPoint := true,
    bits := 32,
    fracbits := 14,
    customDataType := "int_fp14",
    XType := TPtr(TInt),
    YType := TPtr(TInt),
    TIntCtype := "int",
    TRealCtype := "int",
    includes := [ "<include/omega32i.h>" ]
    ),

    i16re := rec(
    isFixedPoint := true,
    bits := 16,
    fracbits := 7,
    customDataType := "short_fp7",
    XType := TPtr(TInt),
    YType := TPtr(TInt),
    TIntCtype := "signed short",
    TRealCtype := "signed short",
    includes := [ "<include/omega16i.h>" ]
    ),

    i8re := rec(
    isFixedPoint := true,
    bits := 8,
    fracbits := 4,
    customDataType := "char_fp4",
    XType := TPtr(TInt),
    YType := TPtr(TInt),
    TIntCtype := "signed char",
    TRealCtype := "signed char",
    includes := [ "<include/omega8i.h>" ]
    )
);

#F InitDataType(<opts>, <data_type>)
#F   Changes the active data type in <opts> and configures the rest of the
#F   system to generate code with that data type.
#F
#F   data_type = "f64re" | "f32re" | "i32re" | "i16re" | "i8re"
#F
#F   Note: currently this only works correctly with CUnparser, and therefore
#F         best config to use is the one produced by InitIter
#F
#F   Example: opts := InitIter(false);
#F            opts := InitDataType(opts, "i16re");
##
InitDataType := function(opts, data_type)
    local dt;
    Constraint(IsString(data_type));
    dt := DataTypes.(data_type);
    opts := CopyFields(opts, dt);
#    if dt.isFixedPoint then
#   opts.compileStrategy := Concatenation(opts.compileStrategy,
#       [ c -> FixedPointCode(c, dt.bits, dt.fracbits) ]);
#    fi;
    return opts;
end;
