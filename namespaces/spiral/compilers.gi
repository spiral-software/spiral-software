
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Import(profiler, rewrite, code);

Class(CompilerDefaults, rec(
    alignmentSpecifier := "",
    modes := [""],
    version := "",
    major := 0,
    minor := 0,
    compiler := "",
    package := "",
    build := "",
    info := self >> Print(self.compiler, " ", self.version),
    default := self >> self.(self.modes[1])(),
    SIMD := () -> spiral.platforms.SIMDArchitectures
));

Class(IntelC, CompilerDefaults, rec(
    alignmentSpecifier := meth(arg)
		local bytes;
		
		if Length(arg) > 1 then
			bytes := arg[2];
		else
			bytes := 16;
		fi;
	
		return "__declspec(align("::String(bytes)::"))";
	end,
		
    postalign := (self, a,i,is) >> Print(Blanks(i), "__assume_aligned(", a, ", 16);\n"),
    restrict := self >> "restrict",

    looppragma := (self, o,i,is) >> When(Collect(o.cmd, loop)=[],
        Print(Blanks(i), "#pragma vector always\n", Blanks(i), "#pragma ivdep\n"),
        Print(Blanks(i), "#pragma novector\n")),

    compiler := "Intel C++ Compiler",

    SIMD := self >> CopyFields(platforms.SIMDArchitectures, rec(
        hasMMX    := True,
		hasSSE    := True,
		hasSSE2   := True,
		hasSSE3   := True,
        hasSSSE3  := () -> self.major >= 10,
		hasSSE4_1 := () -> self.major >= 10,
		hasSSE4_2 := () -> self.major >= 10)),

    modes := ["ia32", "em64t"],

    ia32 := self >> CopyFields(default_profiles.win_x86_icc, rec(
        premake := "iclvars.bat\" > nul"
    )),

    em64t := self >> CopyFields(default_profiles.win_x64_icc, rec(
        premake := "iclvars.bat\" intel64 > nul"
    ))
));


Class(GnuC, IntelC, rec(
    compiler := "gcc (GNU Compiler Collection)",
	
	SIMD := self >> CopyFields(platforms.SIMDArchitectures, rec(
        hasMMX    := True,
		hasSSE    := True,
		hasSSE2   := True,
		hasSSE3   := True,
        hasSSSE3  := True,
		hasSSE4_1 := True,
		hasSSE4_2 := True)),
		
	alignmentSpecifier := meth(arg)
		local bytes;
		
		if Length(arg) > 1 then
			bytes := arg[2];
		else
			bytes := 16;
		fi;
	
		return "__attribute__((aligned("::String(bytes)::")))";
	end,
));

 
 Class(Llvm_Clang, IntelC, rec(
    compiler := "clang (LLVM Compiler Collection)",
	
	SIMD := self >> CopyFields(platforms.SIMDArchitectures, rec(
        hasMMX    := True,
		hasSSE    := True,
		hasSSE2   := True,
		hasSSE3   := True,
        hasSSSE3  := True,
		hasSSE4_1 := True,
		hasSSE4_2 := True)),
		
	alignmentSpecifier := meth(arg)
		local bytes;
		
		if Length(arg) > 1 then
			bytes := arg[2];
		else
			bytes := 16;
		fi;
	
		return "__attribute__((aligned("::String(bytes)::")))";
	end,
));


Class(GnuC_ARM, IntelC, rec(
    compiler := "gcc (GNU Compiler Collection)",
	
	SIMD := self >> CopyFields(platforms.SIMDArchitectures, rec(
        hasMMX    := False,
		hasSSE    := False,
		hasSSE2   := False,
		hasSSE3   := False,
        hasSSSE3  := False,
		hasSSE4_1 := False,
		hasSSE4_2 := False)),
		
	alignmentSpecifier := meth(arg)
		local bytes;
		
		if Length(arg) > 1 then
			bytes := arg[2];
		else
			bytes := 16;
		fi;
	
		return "__attribute__((aligned("::String(bytes)::")))";
	end,
));

Class(VisualC, CompilerDefaults, rec(
    compiler := "MS VisualStudio.NET C++ compiler",
    modes := ["ia32"],
    SIMD := () -> CopyFields(platforms.SIMDArchitectures, rec(hasMMX := True, hasSSE:=True, hasSSE2:=True, hasSSE3:=True)),

	alignmentSpecifier := meth(arg)
		local bytes;
		
		if Length(arg) > 1 then
			bytes := arg[2];
		else
			bytes := 16;
		fi;
	
		return "__declspec(align("::String(bytes)::"))";
	end,
	
    ia32 := self >> CopyFields(default_profiles.win_x86_vcc),
));

Class(VisualC_12, VisualC, rec(
    compiler := "MS VisualStudio.NET C++ 12.0 compiler",
    modes    := ["x86", "x64"],
    SIMD := () -> CopyFields(platforms.SIMDArchitectures, rec(hasMMX := True, hasSSE:=True, hasSSE2:=True, hasSSE3:=True, hasSSE4_1 := True)),
    x86 := self >> CopyFields(default_profiles.win_x86_vcc, rec(
        premake := () -> "call \"%VS120COMNTOOLS%..\\..\\VC\\vcvarsall.bat\" x86 > nul"
    )),
    x64 := self >> CopyFields(default_profiles.win_x64_vcc, rec(
        premake := () -> "call \"%VS120COMNTOOLS%..\\..\\VC\\vcvarsall.bat\" x64 > nul"
    ))
));


Class(NvidiaCuda, IntelC, rec(
    compiler := "NVIDIA Cuda compiler",
	
	SIMD := self >> CopyFields(platforms.SIMDArchitectures, rec(
        hasMMX    := True,
		hasSSE    := True,
		hasSSE2   := True,
		hasSSE3   := True,
        hasSSSE3  := True,
		hasSSE4_1 := True,
		hasSSE4_2 := True)),
		
	alignmentSpecifier := meth(arg)
		local bytes;
		
		if Length(arg) > 1 then
			bytes := arg[2];
		else
			bytes := 16;
		fi;
	
		return "__attribute__((aligned("::String(bytes)::")))";
	end,
));


SupportedCompilers := rec(
    IntelC := IntelC,
    VisualC := VisualC,
    VisualC_12 := VisualC_12,
    GnuC := GnuC,
    GnuC_ARM := GnuC_ARM,
    Llvm_Clang := Llvm_Clang,
    NvidiaCuda := NvidiaCuda
);
