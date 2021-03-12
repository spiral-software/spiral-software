
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(_StandardMeasureVerify);

SetMakeOptsPThreads:=function(opts)
   opts.profile.makeopts.LDFLAGS:=Concat(When(IsBound(opts.profile.makeopts.LDFLAGS),opts.profile.makeopts.LDFLAGS,"")," -lpthread -L/lib/tls -lm");
   opts.profile.makeopts.ADDSRC:=Concat("../../../lib/smp2.c ",When(IsBound(opts.profile.makeopts.ADDSRC),opts.profile.makeopts.ADDSRC,""));
   opts.profile.makeopts.TIMER:="../common/time_threads.c";
   opts.profile.makeopts.TIMER_OPTS:="";
end;

SetMakeOptsSSE:=function(opts)
   opts.profile.makeopts.CFLAGS:=Concat(opts.profile.makeopts.CFLAGS," -msse3 -vec-report=0");
end;

SetMakeOptsOpenMP:=function(opts)
   opts.profile.makeopts.CFLAGS:=Concat(opts.profile.makeopts.CFLAGS," -openmp -openmp-report0");
end;

SetMakeOptsAffinity:=function(opts)
   opts.profile.makeopts.CFLAGS:=Concat(opts.profile.makeopts.CFLAGS," -DUSE_SCHED_AFFINITY");
end;

SetMakeOptsLibgen:=function(opts)
   opts.profile.makeopts.CFLAGS:=Concat(opts.profile.makeopts.CFLAGS," -fno-alias -fno-fnalias -fno-inline-functions");
end;

SetMakeOptsAssembly:=function(opts)
   opts.profile.makeopts.CFLAGS:=Concat(opts.profile.makeopts.CFLAGS," -save-temps");
end;




_default_makeopts := rec(
    GAP := "gap.c",
    STUB := "stub.h",
);

#
## default options for sim-outorder
#
# simple scalar uses a parametrized architecture. these params
# were inferred from details in the following paper:
#
# "Efficient Resource Sharing in Concurrent Error Detecting
# Superscalar Microarchitectures"
#
# Jared C. Smolens, Jangwoo Kim, James C. Hoe, and Babak Falsafi
#
# 37th Annual IEEE/ACM International Symposium on Microarchitecture
#

default_profiles := rec(
    no_compile := rec(
        name := "no-compile",
        meas := (a,b) -> 1000,
        verify := (a,b) -> false,
        makeopts := rec( CFLAGS := "" )
    ),

    # LINUX profiles
    ###############

    linux_power_gcc:=
    rec(
        name := "linux-power-gcc",
        makeopts := rec(
            CFLAGS := "",
            CC := "gcc" ),
        outdir := "/tmp/spiral/power",
        meas := (a, b) -> _StandardMeasureVerify(a, b, ""),
        verify := (a, b) -> _StandardMeasureVerify(a, b, "verify") ),

    arm_icount := rec(
        name := "arm-icount",
        makeopts := rec(
            CFLAGS := "-O2 -std=c99 -fomit-frame-pointer",
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    linux_arm := rec(
        name := "linux-arm",
        makeopts := rec(
            CFLAGS := "-O2 -std=c99 -fomit-frame-pointer",
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    linux_x86_icc := rec(
        name := "linux-x86",
        makeopts := rec(
            CC := "icc",
            CFLAGS := "-O3 -w -std=c99 -fomit-frame-pointer -vec-report0",
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    linux_avx_icc := rec(
        name := "linux-x86",
        makeopts := rec(
            CC := "icc",
            CFLAGS := "-O3 -w -mavx -std=c99 -fomit-frame-pointer -vec-report0",
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    linux_arm_pthread := rec(
        name := "linux-arm",
        makeopts := rec(
            CFLAGS := "-O3 -std=c99 -fomit-frame-pointer",
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, "pthread"),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verifyPThread")
    ),

    linux_x86_icc_openmp := rec(
        name := "linux-x86",
        makeopts := rec(
            CC := "icc",
            CFLAGS := "-O -openmp -w -std=c99 -fomit-frame-pointer",
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    linux_x86_gcc := rec(
        name := "linux-x86",
        makeopts := rec(
            CC := "gcc",
            CFLAGS := "-O2 -Wall -fomit-frame-pointer -march=native -std=c99",
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    linux_x86_cuda := rec(
        name := "linux-cuda",
        makeopts := rec(
            CC := "nvcc",
            CFLAGS := "-O2 ",
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    linux_x86_threads := rec(
        name := "linux-x86",
        makeopts := rec(
            CC := "icc",
            CFLAGS := "-O -Wall -w -std=c99 -msse2",
            LDFLAGS := "-lpthread -L/lib/tls -lm",
            ADDSRC := "../../../lib/smp2.c",
            TIMER := "../common/time_threads.c",
            TIMER_OPTS := ""
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    linux_x86_icc_threads := rec(
        name := "linux-x86",
        makeopts := rec(
            CC := "icc",
            CFLAGS := "-O -Wall -std=c99 -openmp",
            LDFLAGS := "-lguide -lpthread -L/lib/tls -lm",
            ADDSRC := "../../../lib/smp2.c",
            TIMER := "../common/time_threads.c",
            TIMER_OPTS := ""
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    linux_x86_perfmon2 := rec(
        name := "linux-x86-perfmon2",
        makeopts := rec(
            CC := "gcc",
            CFLAGS := "-O3 -fomit-frame-pointer -std=c99",
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    linux_x86_newtimer := rec(
        name := "linux-x86-newtimer",
        makeopts := rec(
            CC := "gcc",
            CFLAGS := "-O3 -fomit-frame-pointer -std=c99",
            HOST := "localhost"
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    linux_lrb_icc := rec(
      name := "linux-lrb-icc",
        makeopts := rec(
            CC := "icc",
            CFLAGS := "-std=c99 -w -vec-report0 -O",
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

     linux_lrb_icc_openmp := rec(
        name := "linux-lrb-icc",
        makeopts := rec(
            CC := "icc",
            CFLAGS := "-O2 -openmp -w -std=c99 -fomit-frame-pointer",
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    linux_lrb_icc_threads := rec(
        name := "linux-lrb-icc",
        makeopts := rec(
            CC := "icc",
            CFLAGS := "-O2 -Wall -std=c99 -openmp",
            LDFLAGS := "-lguide -lpthread -L/lib/tls -lm",
            ADDSRC := "../../../lib/smp2.c",
            TIMER := "../common/time_threads.c",
            TIMER_OPTS := ""
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    linux_x86_vtune := rec(
        name := "linux-x86-vtune",
        makeopts := rec(
            CC := "gcc",
            CFLAGS := "-O2 -g -fomit-frame-pointer -std=c99",
            RUNS := 1000,
            EVENT := "L1D_REPL",
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    linux_arm_gcc := rec(
        name := "linux-arm-gcc",
        makeopts := rec(
            CC := "gcc",
            CFLAGS := "-O2 -std=c99 -fomit-frame-pointer",
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    # LINUX embedded profiles
    #########################

    linux_xscale_gcc := rec(
        name := "linux-xscale-gcc",
        makeopts := rec(
            CC := "arm-xscale-linux-gnu-gcc",
            CFLAGS := "-O2",
            COMPILER_DIR := "",
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    ti_dsk_tms320c6713 := rec(
        name := "ti-dsk-tms320c6713",
        makeopts := rec(
            CFLAGS := "-O3 -std=c99",
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    # MS WINDOWS profiles
    ##############

    win_x86_vcc := rec(
        name := "win-x86-vcc",
        makeopts := rec(
            CC := "cl",
            CFLAGS := "/O2",
        ),
        outdir := "/temp/vcc32",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    win_x64_vcc := rec(
        name := "win-x64-vcc",
        makeopts := rec(
            CC := "cl",
            CFLAGS := "/O3",
        ),
        premake := () -> "vcvarsall.bat amd64 > nul",
        outdir := "/temp/vcc64",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    win_x86_icc := rec(
        name := "win-x86-icc",
        makeopts := rec(
            CC := "icl",
            CFLAGS := "/O3", # /G7 /QxSSSE3",
        ),
        premake := () -> spiral.IntelC.ia32().premake,
        outdir := "/temp/icc32",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    win_x86_icc_openmp := rec(
        name := "win-x86-icc-openmp",
        makeopts := rec(
            CC := "icl",
            CFLAGS := "/O3 /Qopenmp", # /G7 /QxSSSE3",
#            LDFLAGS := "/NODEFAULTLIB:libc /NODEFAULTLIB:libm /NODEFAULTLIB:libirc /DEFAULTLIB:libcmt /DEFAULTLIB:libmmt /DEFAULTLIB:libircmt"
        ),
        premake := () -> spiral.IntelC.ia32().premake,
        outdir := "/temp/icc32omp",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    win_x86_icc_threads := rec(
        name := "win-x86-icc-threads",
        makeopts := rec(
            CC := "icl",
            CFLAGS := "/O3", # /G7 /QxSSSE3",
        ),
        premake := () -> spiral.IntelC.ia32().premake,
        outdir := "/temp/threads32",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    win_x64_icc := rec(
        name := "win-x64-icc",
        makeopts := rec(
            CC := "icl",
            CFLAGS := "/O3 /G7 /QxSSSE3",
        ),
        premake := () -> spiral.IntelC.em64t().premake,
        outdir := "/temp/icc64",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    win_x64_icc_openmp := rec(
        name := "win-x64-icc-openmp",
        makeopts := rec(
            CC := "icl",
            CFLAGS := "/O3 /G7 /QxSSSE3 /Qopenmp",
#            LDFLAGS := "/NODEFAULTLIB:libc /NODEFAULTLIB:libm /NODEFAULTLIB:libirc /DEFAULTLIB:libcmt /DEFAULTLIB:libmmt /DEFAULTLIB:libircmt"
        ),
        premake := () -> spiral.IntelC.em64t().premake,
        outdir := "/temp/icc64omp",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    win_x64_icc_threads := rec(
        name := "win-x64-icc-threads",
        makeopts := rec(
            CC := "icl",
            CFLAGS := "/O3 /G7 /QxSSSE3",
        ),
        premake := () -> spiral.IntelC.em64t().premake,
        outdir := "/temp/threads64",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    win_x86_gcc := rec(
        name := "win-x86-gcc",
        target := rec(name := "win-x86-gcc"),
        makeopts := rec(
            CC := "gcc",
            CFLAGS := "-O3 -march=native -std=c99 -Wno-implicit -Wno-aggressive-loop-optimizations",
        ),
        outdir := "/temp",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    win_x86_llvm := rec(
        name := "win-x86-llvm",
        target := rec(name := "win-x86-llvm"),
        makeopts := rec(
            CC := "clang",
            CFLAGS := "-O2 -march=native -std=c99 -Wall",
        ),
        outdir := "/temp",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    win_x86_nvcc_gpu := rec(
        name := "win-x86-nvcc-gpu",
        makeopts := rec(
            CC := "nvcc",
            CFLAGS := "",
            GAP:= "gap.cu"
        ),
#   premake := () -> spiral.IntelC.ia32().premake,
        outdir := "/temp/nvcc",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
  ),

    win_x64_cuda := rec (
	name := "win-x64-cuda",
	target := rec(name := "win-x64-cuda"),
	makeopts := rec (
	    CC := "nvcc",
	    CFLAGS := "",
	),
        outdir := "/temp",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    # altivec profiles
    # ------------------------------------------------------------------------------
    linux_altivec_gcc := rec(
        name := "linux-altivec-gcc",
        makeopts := rec(
            CFLAGS := ""
        ),
        stubopts := rec(
        ),
        outdir := "/tmp/spiral/altivec",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify"),
        verifyfftw := (a,b) -> _StandardMeasureVerify(a,b, "fftwverify")
    ),

    # CELL processor profiles
    ##############

    linux_cellSPU_gcc := rec(
        name := "linux-cellSPU-gcc",
        makeopts := rec(
            CFLAGS := ""
        ),
        stubopts := rec(
        ),
        outdir := "/tmp/spiral/cellSPU",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    linux_cellSPU_gcc_MMM := rec(
        name := "linux-cellSPU-gcc-MMM",
        makeopts := rec(
            CFLAGS := ""
        ),
        stubopts := rec(
        ),
        outdir := "/tmp/spiral/cellSPU",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    linux_cellmultiSPU_gcc := rec(
        name := "linux-cellmultiSPU-gcc",
        makeopts := rec(
            CFLAGS := "",
            GAP_PPE := "gap_ppe.c",
            GAP_PPE_TWIDDLES_DECL := "twiddles-declare.h",
            GAP_PPE_TWIDDLES_SET  := "twiddles-set.h",
        ),
        stubopts := rec(
            SPUS := 1,
            MULTIBUFFER_ITERATIONS := 1,
        ),
        outdir := "/tmp/spiral/cellmultiSPU",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify"),
        verifyfftw := (a,b) -> _StandardMeasureVerify(a,b, "fftwverify"),
        verifyquick  := (a,b) -> _StandardMeasureVerify(a,b, "quickverify")
    ),

    linux_cellmultiSPU_speadk := rec(
        name := "linux-cellmultiSPU-speadk",
        makeopts := rec(
            CFLAGS := ""
        ),
        stubopts := rec(
            SPUS := 1,
            MULTIBUFFER_ITERATIONS := 1,
        ),
        outdir := "/tmp/spiral/cellmultiSPUspeadk",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify"),
        verifyfftw := (a,b) -> _StandardMeasureVerify(a,b, "fftwverify")
    ),

    linux_cellPPU_gcc := rec(
        name := "linux-cellPPU-gcc",
        makeopts := rec(
            CFLAGS := ""
        ),
        stubopts := rec(
        ),
        outdir := "/tmp/spiral/cellPPU",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    win_remote := rec(
        name := "win-remote",
        host := "host.domain",
        userid := "user",
        passwd := "password",
        outdir := "S:/temp/win-remote",
        hostdir := ".",
        exec := "makecmd.cmd",
        remote_cmd := "remote.cmd",
#        meas := (a,b) -> _RemoteMeasureVerify(a,b, ""),
#        verify := (a,b) -> _RemoteMeasureVerify(a,b, "verify"),
#        runmake := (a,b) -> _RemoteRunMake(a,b),
        remote := true,
        library := "vanilla",
        remote_library := "vanilla",
        appendSymbol := "$*",
#        download := opts -> _RemoteDownload(opts),
#        clean := opts -> _RemoteClean(opts),
        libdir := "profiler",
        dload_cmd := "download.cmd",
        clean_cmd := "clean.cmd"
    ),

    # SIMPLESCALAR 3.0 profiles
    ##############

    ssnix_simple_gcc_cachemiss := rec(
        name := "ssnix-simple-gcc",
        makeopts := rec(
            SSDIR := "~/ss",
            CFLAGS := "-O2 -I~/ss/include",
            SSBIN := "sim-cache",
            SSPARAM := "dl1.misses"
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b,""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    ssnix_simple_gcc_instr := rec(
        name := "ssnix-simple-gcc",
        makeopts := rec(
            SSDIR := "~/ss",
            CFLAGS := "-O2 -I~/ss/include",
            SSBIN := "sim-outorder",
            SSPARAM := "sim_cycle"
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b,""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    ssnix_simple_gcc_cachemissnew := rec(
        name := "ssnix-simple-gcc-new",
        makeopts := rec(
            SSDIR := "~/ss",
            CFLAGS := "-O2 -I~/ss/include",
            SSBIN := "sim-cache",
            SSPARAM := "dl1.misses",
            TIMER_OPTS := "-n 1"
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b,""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    # FPGA Hardware generator profile
    ####################

    fpga_splhdl := rec(
        name := "fpga-splhdl",
        outdir := "/tmp/spiral",
        threshold := 2048,
        makeopts := rec(
            OUTNAME := "gap.v",
            GAP := "gap.spl",
            VLOGLIB := "/Users/pam/eclwork/SPLHDL/support",
            NCV := "ncverilog",
            SPLHDL := "splhdl",
            GETRES := "getRes",
            IVERILOG := "iverilog",
            VVP := "vvp",
            DATATYPE := "fix 16", #other possibility 'float'.
            TWIDTYPE := ""
        ),
        meas := (a,b) -> _StandardMeasureVerify(a,b,""),
    ),

    # INTEL MAC profiles
    #######

    darwin_x86_gcc := rec(
        name := "darwin-x86",
        makeopts := rec(
            CC := "gcc",
            CFLAGS := "-O2 -fomit-frame-pointer -msse2 -std=c99",
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    darwin_x86_icc := rec(
        name := "darwin-x86",
        makeopts := rec(
            CC := "icc",
            CFLAGS := "-O3 -fomit-frame-pointer -std=c99",
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    # PTLSim backend
    ############

    linux_ptlsim_icc := rec(
        name := "linux-ptlsim",
        makeopts := rec(
            CC := "icc",
            CFLAGS := "-O3 -fomit-frame-pointer -std=c99",
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify"),
    ),

    # Basilio's code analyzer.
     linux_x86_anl := rec(
         name := "linux-x86-anl",
         makeopts := rec(
             CC := "gcc",
             CFLAGS := "-O2 -msse2 -w -std=c99 -fomit-frame-pointer",
         ),
         outdir := "/tmp/spiral",
         meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
         verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
     ),

    # DPA simulator profile
    linux_dpa_sim := rec(
        name := "linux-dpa-simulator",
        makeopts := rec(
            DPA_DIR := "${HOME}/DPA",
            DPA_SPEC := "lmvec_memvec_vecint",
            CC := "gcc",
            CFLAGS := "",
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

    # DPA emulator (modelsim) profile
    linux_dpa_emu := rec(
        name := "linux-dpa-modelsim",
        makeopts := rec(
            DPA_DIR := "${HOME}/DPA",
            DPA_SPEC := "lmvec_memvec_vecint",
            XCC_DIR := "/opt/sparc-elf-4.4.2",
            MODELSIM_DIR := "/opt/modelsim",
            CC := "sparc-elf-gcc",
            CFLAGS := "",
        ),
        outdir := "/tmp/spiral",
        meas := (a,b) -> _StandardMeasureVerify(a,b, ""),
        verify := (a,b) -> _StandardMeasureVerify(a,b, "verify")
    ),

);
