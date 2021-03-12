
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# LocalConfig is configured in the user's local _spiral.g (Linux/Unix)
# or _spiral_win.g (Windows) in the Spiral root directory
#
# At a minimum, cpuinfo and osinfo need to be set, see one of those
# above mentioned config files for reference

Declare(LocalConfig, SpiralDefaults);
Class(LocalConfig, rec(
    info := meth(self)
                Print("\nPID: ", GetPid(), "\n");
            end,
			
    appendSym := self >> When(IsBound(self.osinfo.isWindows) and self.osinfo.isWindows(), "%*", "$*"),

    compilerinfo := rec(
        compiler:= "",
        defaultMode := "none",
        modes := rec(none := ""),
        alignmentSpecifier := ()->"",
        info := ()->Print("compiler info not set"),
    ),

    cpuinfo := rec(
        cpuname:="",
        vendor:="",
        freq:=0,
        default_lang:="",
        isWindows := False,
        isLinux := False,
        is32bit := False,
        is64bit := False,
        nproc := 1,
        SIMDname := False,
        info := ()->Print("CPU info not set"),
        SIMD := () -> spiral.platforms.SIMDArchitectures
    ),

    osinfo := rec(info := ()->Print("OS info not set"),
        isWindows := False,
        isLinux := False,
        isDarwin := False,
        isCygwin := False),

    svninfo := rec(version := "unknown",
        modified := "unknown",
        mixed := "unknown",
        isInit := false,
        info := self >> When(self.isInit, Print("SVN: ", self.version, When(self.modified, " (modified)", "")), Print("SVN info not set"))
    ),

    getOpts := arg >> arg[1].cpuinfo.getOpts(Drop(arg, 1)),

    setTitle := meth(arg)
                    local self, title;
                    self := arg[1];
                    if not IsBound(self.osinfo.setTitle) then return false; fi;
                    if Length(arg)=1 then title := ""; else title := Concat(" - ", arg[2]); fi;
                    self.osinfo.setTitle(Concat("Spiral 5.0", title));
                    return true;
                end
));

HighPerfMixin := rec(
    useDeref := true,
    compileStrategy := compiler.IndicesCS2,
    propagateNth := false
);

SpiralDefaults := CopyFields(SpiralDefaults, rec(
    includes := ["<include/omega64.h>"],
    precision       := "double",
    generateInitFunc := true,
    XType := code.TPtr(code.TReal),
    YType := code.TPtr(code.TReal),
    unifyStoredDataType := false, # false | "input" | "output" 
                                  # if non-false, then unifies the datatype of 
                                  # precomputed data with the datatype of input (X) 
                                  # or output (Y)
    # we implement complex transforms using real vector of 2x size
    dataType        := "real",
    globalUnrolling := 32,
    faultTolerant   := false,
    printWebMeasure := false,

    # compiler options
    finalBinSplit := false,
    declareConstants := false,
    doScalarReplacement := false,
    propagateNth := true,
    inplace := false,

    doSumsUnification := false,
    arrayDataModifier := "static",
    scalarDataModifier := "",
    arrayBufModifier := "static",
    funcModifier := "", # for example "__decl" or "__fastcall"
    valuePostfix := "",

    # How much information Spiral is printing on the terminal.
    # Currently rather few functions are using this.
    verbosity := 1,

    # list of include files in generated C code, eg. ["<math.h>"]
    includes := [],

    formulaStrategies := rec(
        sigmaSpl := [ sigma.StandardSumsRules ],
        preRC    := [],
        rc       := [ sigma.StandardSumsRules ],
        postProcess := [
        (s, opts) -> compiler.BlockSums(opts.globalUnrolling, s),
        (s, opts) -> sigma.Process_fPrecompute(s, opts)
        ]
    ),

    baseHashes := [],
    subParams := [],

    sumsgen := sigma.DefaultSumsGen,
    # breakdownRules limits the used breakdown rules.
    # It must be a record of the form
    # rec(
    #   nonTerm := [ breakdown_rule1, breakdown_rule2, ...],
    #   DFT := [ DFT_Base, DFT_CT ]  <-- example
    # ).
    #
    # By default we set it to ApplicableTable for backwards compatibility with
    # older svn revisions.
    #
    # Functions SwitchRulesOn/Off will work only with breakdownRules==ApplicableTable.
    breakdownRules := formgen.ApplicableTable,

    compileStrategy := compiler.IndicesCS,
    simpIndicesInside := [code.nth, code.tcast, code.deref],
    useDeref := true,
    generateComplexCode := false,

    unparser := compiler.CUnparserProg,
    codegen := compiler.DefaultCodegen,
    TCharCtype :=  "char",
    TUCharCtype := "unsigned char",
    TUIntCtype := "unsigned int",
    TULongLongCtype := "unsigned long long",
    TRealCtype := "double",

    operations := rec(Print := s -> Print("<Spiral options record>")),

    highPerf := self >> CopyFields(self, HighPerfMixin),

    coldcache := false,
));

CplxSpiralDefaults := CopyFields(SpiralDefaults, rec(
    includes := ["<include/complex_gcc_sse2.h>"],
    unparser := compiler.CMacroUnparserProg,
    XType := code.TPtr(code.TComplex),
    YType := code.TPtr(code.TComplex),
    dataType := "complex",
    generateComplexCode := true,
    c99 := rec(
        I := "__I__",
        re := "creal",
        im := "cimag"
        )
));

IntelC99Mixin := rec(
    includes := ["<include/omega64c.h>"],
    unparser := compiler.CUnparserProg,
    XType := code.TPtr(code.TComplex),
    YType := code.TPtr(code.TComplex),
    dataType := "complex",
    generateComplexCode := true,
    c99 := rec(
        I := "__I__",
        re := "creal",
        im := "cimag"
        ),
    TComplexCtype := "_Complex double",
    TRealCtype := "double",
);

IBMC99Mixin := rec(
    includes := ["<include/omega64c.h>"],
    unparser := compiler.CUnparserProg,
    XType := code.TPtr(code.TComplex),
    YType := code.TPtr(code.TComplex),
    dataType := "complex",
    generateComplexCode := true,
    c99 := rec(
        I := "__I",
        re := "_creal",
        im := "_cimag"
        ),
    TComplexCtype := "_Complex double",
    TRealCtype := "double",
    postalign := n -> Print("    __alignx(16,", n, ");\n")
);


# How do we determine if we're running on Windows or Linux?
#Try(Load(iswindows));
#Try(Load(islinux));

#  NOTE: I'd like to pull that out into a function call, but failed to use load/include/read inside a function... => ask YSV
#if LocalConfig.osinfo.isWindows() then
#    Exec(let(sdir:=Conf("spiral_dir"), Concat("SubWCRev.exe ", sdir, " ", Concat(sdir, "\\spiral\\svn_win.src ", sdir, "\\spiral\\svn_info.g > NUL"))));
#    Load(svn_info);
#fi;
#if LocalConfig.osinfo.isLinux() then
# NOTE: For now, assume that any non-windows system is a linux system.
#else
 #   Exec(let(sdir:=Conf("spiral_dir"), Concat(". ", sdir, "/spiral/svn_linux.src ", sdir, " > ", sdir, "/spiral/svn_info.g")));
#    Load(svn_info);
#fi;

compiler.Unparser.fileinfo := meth(self, opts)
    local info;

#    if IsBound(opts.fileinfo) then
#        info := opts.fileinfo;
#        Print("/*\tCPU: ");
#        LocalConfig.cpuinfo.info();
#        Print("\n\tOS: ");
#        LocalConfig.osinfo.info();
#        Print("\n\t");
#        LocalConfig.svninfo.info();
#        if IsBound(opts.profile) then
#            Print("\n\tprofile: ", opts.profile.name, ", ", opts.profile.makeopts.CFLAGS);
#        else
#            Print("\n\tlanguage: ", opts.language);
#        fi;
#        PrintLine("\n\ttimestamp: ", let(t:=Date(), Concat(t[2]," ",StringInt(t[3]),", ",StringInt(t[1]), "; ",StringInt(t[4]),":",StringInt(t[5]),":",StringInt(t[6]))),
#            "\n\ttransform: ", info.algorithm.node , "\n\t",
#            "source file: \"", info.file, "(.c)\"\n\t",
#            "performance: ", info.cycles, " cycles, ", spiral._compute_mflops(info.flops, info.cycles), " Mflop/s\n",
#            "\nalgorithm: ", info.algorithm, "\n",
#        "*/\n");
#    fi;
	
    if IsBound(opts.fileinfo) then
        info := opts.fileinfo;
        if IsBound(opts.profile) then
            Print("/*\tprofile: ", opts.profile.name, ", ", opts.profile.makeopts.CFLAGS);
        else
            Print("/*\tlanguage: ", opts.language);
        fi;
        PrintLine("\n\ttimestamp: ", let(t:=Date(), Concat(t[2]," ",StringInt(t[3]),", ",StringInt(t[1]), "; ",StringInt(t[4]),":",StringInt(t[5]),":",StringInt(t[6]))),
            "\n",
            "\nalgorithm: ", info.algorithm, "\n",
        "*/\n");
    fi;	
end;
