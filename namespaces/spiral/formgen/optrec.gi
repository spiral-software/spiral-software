
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Option Records
# ==============

#F SPL Options Records
#F -------------------

#F An spl options record is used to collect options passed to the
#F spl compiler by functions that communicate with the spl compiler,
#F such as measure, verify, and search functions.
#F
#F There is a system wide options record, SpiralDefailts (set in config.g),
#F which sets the default values.
#F
#F An spl options record is a record, which contains a subset of the 
#F following fields. Adding a new fields requires to change the
#F functions PrintSpecSPLOptionsRecord, CheckSPLOptionsRecord, and
#F MakeSPLCallSPLOptionsRecord.
#F
#F rec(
#F   customDataType  = "int_cplx" | "int_fpN" | "Ipp16sc" (XScale) | "Ipp32sc" (XScale)
#F                                     where N - number of fractional bits, i.e. int_fp8
#F   zeroBits        = zero | <positive int>
#F   dataType        = "real" | "complex",
#F   precision       = "single" | "double" | "extended"
#F   subName         = <string>
#F   schedule        = <integer>
#F   globalUnrolling = <positive int> | "none" | "full",
#F   language        = "fortran" | "c"    # see config.g for languages
#F   compiler        = not to be used
#F   compflags       = <flags for compiler as string>
#F   splflags        = <flags for spl compiler as string>
#F )
#F
#F Note: switching language automatically switches compilers
#F and flags.
#F 
#F spl options records should be used as follows:
#F - create your desired spl options record R
#F - merge with defaults, R1 := MergeSPLOptionsRecord(R)
#F - create spl prog for external operations with ProgSPL(SPL, R)
#F


#F CheckSPLOptionsRecord ( <spl-options-record> )
#F   checks whether <spl-options-record> is a valid spl options record
#F   with valid spl options set. If a field name or a field value
#F   is invalid, then an error is signaled, otherwise true is
#F   returned.
#F
CheckSPLOptionsRecord := function ( R )
  local r;
  if not IsRec(R) then Error("<R> must be an spl options record"); fi;

  for r in RecFields(R) do
    Cond(r = "dataType",
        Constraint(R.dataType in ["no default", "real", "complex"]),
         r = "customDataType",
        Constraint(IsString(R.customDataType)),
         r = "customReal",
        Constraint(IsString(R.customReal)),
         r = "customComplex",
        Constraint(IsString(R.customComplex)),
     r = "zeroBits",
        Constraint(IsInt(R.zeroBits) and R.zeroBits >= 0),
         r = "precision",
            Constraint(R.precision in  ["single", "double", "extended"]),
         r = "subName",
        Constraint(IsString(R.subName)),
         r = "file",
        Constraint(IsString(R.file)),
         r = "schedule",
        Constraint(IsInt(R.schedule) and R.schedule > 0),
         r = "globalUnrolling",
        Constraint(
        (IsInt(R.globalUnrolling) and R.globalUnrolling > 0)
        or R.globalUnrolling in ["none", "full"]),
         r = "compiler",
        Constraint(IsString(R.compiler)),
         r = "compflags",
        Constraint(IsString(R.compflags)),
         r = "dmpcompflags",
        Constraint(IsString(R.dmpcompflags)),
       r = "faultTolerant", 
        Constraint(IsBool(R.faultTolerant)),
     r = "cgen", 
        Constraint(IsFunc(R.cgen)),
     r = "x" or IsSystemRecField(r), 
        Ignore(),
     0); # do nothing if field is unrecognized
  od;

  return true;
end;


#F MergeSPLOptionsRecord ( <spl-options-record> )
#F   returns the option record obtained by starting with SpiralDefaults
#F   (config.g, set at installation) and merging or overwriting with the
#F   spl options given by <spl-options-record>
#F
MergeSPLOptionsRecord := R -> Checked(CheckSPLOptionsRecord(R), 
    CopyFields(SpiralDefaults, R)
);
