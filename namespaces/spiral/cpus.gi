
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Import(platforms, profiler, code);
Import(platforms.sse, platforms.avx, platforms.intel);

Class(OSInfo, rec(osname := "",
    is8Bit := False,
    is16bit := False,
    is32bit := False,
    is64bit := False,
    isWindows := False,
    isLinux := False,
    isCygwin := False,
    isDarwin := False,
    info := self >> Cond(IsBound(self.vendor), Print(self.vendor, " ", self.osname), Print(self.osname)),
    useColor := True
));

Class(OSWindows, OSInfo, rec(
    setTitle := meth(self, title)
        Exec(Concat("title ", title));
    end,
    vendor := "Microsoft",
    isWindows := True,
    useColor := False
));

Class(OSWindows32, OSWindows, rec(is16bit := True, is32bit := True));
Class(OSWindows64, OSWindows, rec(is16bit := True, is32bit := True, is64bit :=True));
Class(OSLinux32, OSInfo, rec(osname := "Linux32", is32bit := True, isLinux := True));
Class(OSLinux64, OSInfo, rec(osname := "Linux64", is32bit := True, is64bit := True, isLinux := True));
Class(OSArmLinux, OSInfo, rec(osname := "GNU/Linux", is16bit := True, is32bit := True, isLinux := True));
Class(OSCygwin32, OSInfo, rec(osname := "Cygwin32", is32bit := True, isCygwin := True));
Class(OSDarwin, OSInfo, rec(osname := "OSX/Darwin", is32bit := True, is64bit := True, isDarwin := True));

SupportedOSs := rec(
    WindowsNT4 := CopyFields(OSWindows32, rec(osname := "Windows NT 4.0")),
    Windows2000 := CopyFields(OSWindows32, rec(osname := "Windows 2000")),
    WindowsXP32 := CopyFields(OSWindows32, rec(osname := "WindowsXP 32-bit")),
    WindowsXP64 := CopyFields(OSWindows64, rec(osname := "WindowsXP 64-bit")),
    WindowsVista := CopyFields(OSWindows64, rec(osname := "Windows Vista")),
    Windows7 := CopyFields(OSWindows64, rec(osname := "Windows 7")),
    Windows8 := CopyFields(OSWindows64, rec(osname := "Windows 8")),
    Linux32 := OSLinux32,
    Linux64 := OSLinux64,
    Cygwin32 := OSCygwin32,
    Darwin := OSDarwin,
    ArmLinux := OSArmLinux,
);

Class(CPUInfo, rec(
    hasDouble := True,
    hasFloat := True,
    intSize := 32,
    cores := 1,
    cpuname := "",
    vendor := "",
    info := self >> Chain(
        Print(Cond(IsBound(self.vendor), Print(self.vendor, " ", self.cpuname), Print(self.osname))),
        When(IsBound(self.freq) and self.freq > 0, Print(" at ", self.freq, " MHz")),
        When(self.cores > 1, Print(", ", self.cores, " cores")),
        When(IsBound(self.SIMDname), Print(", ", self.SIMDname))
    )
));

Class(IntelCPU, CPUInfo, rec(
    vendor := "Intel",
    getSimdIsa := (self, dt) >> When(self.SIMD().hasAVX(), 
        Cond(
            dt = T_Real(32), AVX_8x32f,
            dt = T_Real(64), AVX_4x64f,
            dt),
        Cond(
            dt = T_Real(32), SSE_4x32f,
            dt = T_Real(64), SSE_2x64f,
            dt)
    ),
    getOpts := arg >> IAGlobals.getOpts(Drop(arg, 1))
));

Class(AMDCPU, CPUInfo, rec(
    vendor := "AMD"
));


Class(STICPU, CPUInfo, rec(
    vendor := "STI"
));

Class(PowerPC, CPUInfo, rec(
    vendor := "IBM"
));

Class(DPA, CPUInfo, rec(
       vendor := "CMU"
));

Class(ARM, CPUInfo, rec(
       vendor := "Raspberry Pi Foundation"
));


SupportedCPUs := rec(
    Pentium := rec(),
    PentiumPro := rec(),
    PentiumII := rec(),
    PentiumIII := rec(),
    Pentium4 := Class(Pentium4, IntelCPU, rec(
        cpuname := "Pentium4",
        SIMD := () -> CopyFields(platforms.SIMDArchitectures, rec(hasMMX := True, hasSSE:=True, hasSSE2:=True, hasSSE3:=False)),
        SIMDname := "SSE2",
        cores := 1,
        default_lang := "c.icl.opt.pentium4"
    )),
    Pentium4Extreme := Class(Pentium4Extreme, IntelCPU, rec(
        cpuname := "Pentium4Extreme",
        SIMD := () -> CopyFields(platforms.SIMDArchitectures, rec(hasMMX := True, hasSSE:=True, hasSSE2:=True, hasSSE3:=True)),
        SIMDname := "SSE3",
        cores := 1,
        default_lang := "c.icl.opt.pentium4extreme"
    )),
    PentiumD := rec(),
    Xeon := Class(Xeon, IntelCPU, rec(
        cpuname := "Xeon",
        SIMD := () -> CopyFields(platforms.SIMDArchitectures, rec(hasMMX := True, hasSSE:=True, hasSSE2:=True, hasSSE3:=True, hasSSSE3:=True, hasSSE4_1 := True, hasSSE4_2 := True)),
        SIMDname := "SSE4.2",
        cores := 4,
        profile := rec(
            EM64T := (arg) -> CopyFields(default_profiles.win_x64_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxS")),
            IA32 := (arg) -> CopyFields(default_profiles.win_x86_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxS")),
            threads := (arg) -> CopyFields(default_profiles.win_x86_icc_threads, rec(CFLAGS := arg -> "/O3 /G7 /QxS"))
        ),
        default_lang := "c.icl.opt.corei7",
        smp_lang := "c.icl.smp_corei7",
        OpenMP_lang := "c.icl.openmp_corei7"
    )),
    XeonMP := rec(),
    PentiumM := Class(PentiumM, IntelCPU, rec(
        cpuname := "PentiumM",
        SIMD := () -> CopyFields(platforms.SIMDArchitectures, rec(hasMMX := True, hasSSE:=True, hasSSE2:=True, hasSSE3:=False)),
        SIMDname := "SSE2",
        cores := 1,
        default_lang := "c.icl.opt.pentiumM"
    )),
    CoreDuo := Class(CoreDuo, IntelCPU, rec(
        cpuname := "CoreDuo",
        SIMD := () -> CopyFields(platforms.SIMDArchitectures, rec(hasMMX := True, hasSSE:=True, hasSSE2:=True, hasSSE3:=True)),
        SIMDname := "SSE3",
        cores := 2,
        profile := rec(
            IA32 := (arg) -> CopyFields(default_profiles.win_x86_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxKWP")),
            threads := (arg) -> When(LocalConfig.osinfo.isWindows(),
        CopyFields(default_profiles.win_x86_icc_threads, rec(CFLAGS := arg -> "/O3 /G7 /QxKWP")),
        CopyFields(default_profiles.linux_x86_threads, rec(CFLAGS := arg -> "-msse3"))
)
        ),
        default_lang := "c.icl.opt.core",
        smp_lang := "c.icl.smp_core",
        OpenMP_lang := "c.icl.openmp_core"
    )),
    Core2Duo := Class(Core2Duo, IntelCPU, rec(
        cpuname := "Core2Duo",
        SIMD := () -> CopyFields(platforms.SIMDArchitectures, rec(hasMMX := True, hasSSE:=True, hasSSE2:=True, hasSSE3:=True, hasSSSE3:=True)),
        SIMDname := "SSSE3",
        cores := 2,
        profile := rec(
            EM64T := (arg) -> CopyFields(default_profiles.win_x64_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxSSSE3")),
            IA32 := (arg) -> CopyFields(default_profiles.win_x86_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxKWP")),
            threads := (arg) -> CopyFields(default_profiles.win_x86_icc_threads, rec(CFLAGS := arg -> "/O3 /G7 /QxKWP"))
        ),
        default_lang := "c.icl.opt.core2",
        smp_lang := "c.icl.smp_core2",
        OpenMP_lang := "c.icl.openmp_core2"
    )),
    Core2Quad := Class(Core2Quad, IntelCPU, rec(
        cpuname := "Core2 Extreme",
        SIMD := () -> CopyFields(platforms.SIMDArchitectures, rec(hasMMX := True, hasSSE:=True, hasSSE2:=True, hasSSE3:=True, hasSSSE3:=True)),
        SIMDname := "SSSE3",
        cores := 4,
        profile := rec(
            EM64T := (arg) -> CopyFields(default_profiles.win_x64_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxSSSE3")),
            IA32 := (arg) -> CopyFields(default_profiles.win_x86_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxKWP")),
            threads := (arg) -> CopyFields(default_profiles.win_x86_icc_threads, rec(CFLAGS := arg -> "/O3 /G7 /QxKWP"))
        ),
        default_lang := "c.icl.opt.core2",
        smp_lang := "c.icl.smp_core2",
        OpenMP_lang := "c.icl.openmp_core2"
    )),


    DPABasic := Class(DPABasic, DPA, rec(
	    cpuname := "DPABasic",
	    cores := 1,
	    SIMD := () -> CopyFields(platforms.SIMDArchitectures, rec( numregs := 32)),
	    )),

    Core2Penryn := Class(Core2Penryn, IntelCPU, rec(
        cpuname := "Core2 Penryn",
        SIMD := () -> CopyFields(platforms.SIMDArchitectures, rec(hasMMX := True, hasSSE:=True, hasSSE2:=True, hasSSE3:=True, hasSSSE3:=True, hasSSE4_1 := True)),
        SIMDname := "SSE4.1",
        cores := 4,
        profile := rec(
            EM64T := (arg) -> CopyFields(default_profiles.win_x64_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxS")),
            IA32 := (arg) -> CopyFields(default_profiles.win_x86_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxS")),
            threads := (arg) -> CopyFields(default_profiles.win_x86_icc_threads, rec(CFLAGS := arg -> "/O3 /G7 /QxS"))
        ),
        default_lang := "c.icl.opt.core2p",
        smp_lang := "c.icl.smp_core2p",
        OpenMP_lang := "c.icl.openmp_core2p"
    )),
    Core_i7 := Class(Core_i7, IntelCPU, rec(
        cpuname := "Core i7",
        SIMD := () -> CopyFields(platforms.SIMDArchitectures, rec(hasMMX := True, hasSSE:=True, hasSSE2:=True, hasSSE3:=True, hasSSSE3:=True, hasSSE4_1 := True, hasSSE4_2 := True)),
        SIMDname := "SSE4.2",
        cores := 4,
        profile := rec(
            EM64T := (arg) -> CopyFields(default_profiles.win_x64_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxS")),
            IA32 := (arg) -> CopyFields(default_profiles.win_x86_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxS")),
            threads := (arg) -> CopyFields(default_profiles.win_x86_icc_threads, rec(CFLAGS := arg -> "/O3 /G7 /QxS"))
        ),
        default_lang := "c.icl.opt.corei7",
        smp_lang := "c.icl.smp_corei7",
        OpenMP_lang := "c.icl.openmp_corei7"
    )),
	# specifically for Spiral FFT GPL 1.0
    Core_AVX := Class(Core_AVX, IntelCPU, rec(
        cpuname := "Intel Core with AVX",
        SIMD := () -> CopyFields(platforms.SIMDArchitectures, rec(hasMMX := True, hasSSE:=True, hasSSE2:=True, hasSSE3:=True, hasSSSE3:=True, hasSSE4_1 := True, hasSSE4_2 := True, hasAVX := True)),
        SIMDname := "AVX",
        cores := 4,
        profile := rec(
            EM64T := (arg) -> CopyFields(default_profiles.win_x64_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxS")),
            IA32 := (arg) -> CopyFields(default_profiles.win_x86_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxS")),
            threads := (arg) -> CopyFields(default_profiles.win_x86_icc_threads, rec(CFLAGS := arg -> "/O3 /G7 /QxS"))
        ),
        default_lang := "c.icl.opt.corei7",
        smp_lang := "c.icl.smp_corei7",
        OpenMP_lang := "c.icl.openmp_corei7"
    )),
	# specifically for Spiral FFT GPL 1.0
    Core_no_AVX := Class(Core_no_AVX, IntelCPU, rec(
        cpuname := "Intel Core without AVX",
        SIMD := () -> CopyFields(platforms.SIMDArchitectures, rec(hasMMX := True, hasSSE:=True, hasSSE2:=True, hasSSE3:=True, hasSSSE3:=True, hasSSE4_1 := True, hasSSE4_2 := True, hasAVX := False)),
        SIMDname := "SSE4.2",
        cores := 4,
        profile := rec(
            EM64T := (arg) -> CopyFields(default_profiles.win_x64_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxS")),
            IA32 := (arg) -> CopyFields(default_profiles.win_x86_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxS")),
            threads := (arg) -> CopyFields(default_profiles.win_x86_icc_threads, rec(CFLAGS := arg -> "/O3 /G7 /QxS"))
        ),
        default_lang := "c.icl.opt.corei7",
        smp_lang := "c.icl.smp_corei7",
        OpenMP_lang := "c.icl.openmp_corei7"
    )),
    Core_i7U := Class(Core_i7U, IntelCPU, rec(
        cpuname := "Core i7 Ultrabook",
        SIMD := () -> CopyFields(platforms.SIMDArchitectures, rec(hasMMX := True, hasSSE:=True, hasSSE2:=True, hasSSE3:=True, hasSSSE3:=True, hasSSE4_1 := True, hasSSE4_2 := True, hasAVX := True)),
        SIMDname := "AVX",
        cores := 2,
        profile := rec(
            EM64T := (arg) -> CopyFields(default_profiles.win_x64_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxS")),
            IA32 := (arg) -> CopyFields(default_profiles.win_x86_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxS")),
            threads := (arg) -> CopyFields(default_profiles.win_x86_icc_threads, rec(CFLAGS := arg -> "/O3 /G7 /QxS"))
        ),
        default_lang := "c.icl.opt.corei7",
        smp_lang := "c.icl.smp_corei7",
        OpenMP_lang := "c.icl.openmp_corei7"
    )),
    Core_i5U := Class(Core_i5U, IntelCPU, rec(
        cpuname := "Core i5U",
        SIMD := () -> CopyFields(platforms.SIMDArchitectures, rec(hasMMX := True, hasSSE:=True, hasSSE2:=True, hasSSE3:=True, hasSSSE3:=True, hasSSE4_1 := True, hasSSE4_2 := True)),
        SIMDname := "SSE4.2",
        cores := 2,
        profile := rec(
            EM64T := (arg) -> CopyFields(default_profiles.win_x64_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxS")),
            IA32 := (arg) -> CopyFields(default_profiles.win_x86_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxS")),
            threads := (arg) -> CopyFields(default_profiles.win_x86_icc_threads, rec(CFLAGS := arg -> "/O3 /G7 /QxS"))
        ),
        default_lang := "c.icl.opt.corei7",
        smp_lang := "c.icl.smp_corei7",
        OpenMP_lang := "c.icl.openmp_corei7"
    )),
    Core_i5_SandyBridge := Class(Core_i5_SandyBridge, IntelCPU, rec(
        cpuname := "Core i5 SandyBridge",
        SIMD := () -> CopyFields(platforms.SIMDArchitectures, rec(hasMMX := True, hasSSE:=True, hasSSE2:=True, hasSSE3:=True, hasSSSE3:=True, hasSSE4_1 := True, hasSSE4_2 := True, hasAVX := True)),
        SIMDname := "AVX",
        cores := 4,
        profile := rec(
            EM64T := (arg) -> CopyFields(default_profiles.win_x64_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxS")),
            IA32 := (arg) -> CopyFields(default_profiles.win_x86_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxS")),
            threads := (arg) -> CopyFields(default_profiles.win_x86_icc_threads, rec(CFLAGS := arg -> "/O3 /G7 /QxS"))
        ),
    )),

    Itanium := rec(),
    Itanium2 := rec(),
    Itanium3 := rec(),
    MPC_G4 := rec(),
    PPC905_G5 := rec(),
    XScale := rec(),
    PowerPC405 := rec(),
    Athlon := rec(),
    AthlonXP := rec(),
    Opteron := rec(),
    OpteronDual := rec(),
    OpteronQuad := Class(OpteronQuad, AMDCPU, rec(
        cpuname := "Dual Opteron 2220",
        SIMD := () -> CopyFields(platforms.SIMDArchitectures, rec(hasMMX := True, hasSSE:=True, hasSSE2:=True, hasSSE3:=True)),
        SIMDname := "SSE3",
        cores := 4,
        profile := rec(
            EM64T := (arg) -> CopyFields(default_profiles.win_x64_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxSSSE3")),
            IA32 := (arg) -> CopyFields(default_profiles.win_x86_icc, rec(CFLAGS := arg -> "/O3 /G7 /QxKWP")),
            threads := (arg) -> CopyFields(default_profiles.win_x86_icc_threads, rec(CFLAGS := arg -> "/O3 /G7 /QxKWP"))
        ),
        default_lang := "c.icl.opt.opteron2220"
    )),
    CellBE := Class(CellBE, STICPU, rec(
        cpuname := "Cell BE",
        SIMD := () -> CopyFields(platforms.SIMDArchitectures, rec(hasSPU := True)),
        SIMDname := "SIMD-SPU",
        cores := 9,
        profile := rec(
            EM64T := (arg) -> CopyFields(default_profiles.win_x64_icc, rec(CFLAGS := arg -> "")),
            IA32 := (arg) -> CopyFields(default_profiles.win_x86_icc, rec(CFLAGS := arg -> "")),
            threads := (arg) -> CopyFields(default_profiles.win_x86_icc_threads, rec(CFLAGS := arg -> ""))
        ),
        default_lang := ""
   )),
   CellBEPS3 := Class(CellBEPS3, STICPU, rec(
        cpuname := "Cell BE (PS3)",
        SIMD := () -> CopyFields(platforms.SIMDArchitectures, rec(hasSPU := True)),
        SIMDname := "SIMD-SPU",
        cores := 6,
        profile := rec(
            EM64T := (arg) -> CopyFields(default_profiles.win_x64_icc, rec(CFLAGS := arg -> "")),
            IA32 := (arg) -> CopyFields(default_profiles.win_x86_icc, rec(CFLAGS := arg -> "")),
            threads := (arg) -> CopyFields(default_profiles.win_x86_icc_threads, rec(CFLAGS := arg -> ""))
        ),
        default_lang := "",
        getSimdIsa := dt -> Cond(
            dt = T_Real(32), platforms.cellSPU.spu_4x32f,
            dt = T_Real(64), platforms.cellSPU.spu_2x64f,
            dt),
   )),
   PowerPC970 := Class(PowerPC970, PowerPC, rec(
        cpuname := "PowerPC 970 (G5)",
        SIMD := () -> CopyFields(platforms.SIMDArchitectures, rec(hasAltiVec := true)),
        SIMDname := "AltiVec",
        cores := 1,
        profile := rec(
        ),
        default_lang := ""
   )),
                     
   ARMV7L := Class(ARMV7L, ARM, rec(
       cpuname := "ARMV7L",
       cores := 4,
   )),

   BlueGeneL := rec()
);
