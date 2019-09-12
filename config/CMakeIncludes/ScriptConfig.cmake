##
## SPIRAL License
##
## Copyright (c) 2018, Carnegie Mellon University
## All rights reserved.
## 
## See LICENSE file for full information
##

cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

##  Determine various configuration properties to create necessary inputs for launch/batch scripts

string (COMPARE EQUAL "${CMAKE_HOST_SYSTEM_PROCESSOR}" "AMD64" INTEL_WIN)
string (COMPARE EQUAL "${CMAKE_HOST_SYSTEM_PROCESSOR}" "x86_64" INTEL_LINUX)
string (COMPARE EQUAL "${CMAKE_HOST_SYSTEM_PROCESSOR}" "armv7l" ARM_LINUX)


if (${INTEL_WIN} OR ${INTEL_LINUX})
    ##  Intel architecture, SupportedCPU = Core_AVX
    set (CPU_ARCH_TYPE "Core_AVX")
    set (CPU_FREQUENCY 2195)
    set (SPIRAL_MEMORY_POOL "2048m")
elseif (${ARM_LINUX})
    ##  Raspberry Pi, SupportedCPU = armv7l
    set (CPU_ARCH_TYPE "ARMV7L")
    set (CPU_FREQUENCY 1500)
    set (SPIRAL_MEMORY_POOL "1024m")
elseif (${CMAKE_HOST_APPLE})
    ##  Apple ... MacBook
    set (CPU_ARCH_TYPE "Core_AVX")
    set (CPU_FREQUENCY 2195)
    set (SPIRAL_MEMORY_POOL "2048m")
endif ()

if (WIN32)
    ##  Windows ... now determine which C compiler we're using...
    string (COMPARE EQUAL "${CMAKE_C_COMPILER_ID}" "Intel" ICC_COMPILER)
    string (COMPARE EQUAL "${CMAKE_C_COMPILER_ID}" "GNU" GCC_COMPILER)
    set(SPIRAL_OS_NAME "Windows8")
    if (${ICC_COMPILER})
	##  Intel ICC compiler
	set (USE_COMPILER_ICC "true")
	set (USE_COMPILER_GCC "false")
	set (PROFILER_TARGET "win-x64-icc")
    elseif (${GNU_COMPILER})
	##  GNU C compiler
	set (USE_COMPILER_ICC "false")
	set (USE_COMPILER_GCC "true")
	set (PROFILER_TARGET "win-x86-gcc")
    else ()
	##  Default to Visual Studio
	set (USE_COMPILER_ICC "false")
	set (USE_COMPILER_GCC "false")
	set (PROFILER_TARGET "win-x86-vcc")
    endif ()

elseif (${ARM_LINUX})
    ##  Raspberry Pi
    ##  Use GDB as debugger, --args ==> arguments after program name are for debugged process
    set (PROFILER_TARGET "linux-arm-gcc")
    set (PROFILE_TARGET_ID "linux_arm_gcc")
    set (SPIRAL_OS_NAME "ArmLinux")
    set (SPIRAL_COMPILER_NAME "GnuC_ARM")
    set (SPIRAL_DEBUGGER_NAME "gdb --args")

elseif (${CMAKE_HOST_APPLE})
    ##  Apple ... MacBook
    ##  Use LLDB as debugger, requires --arch, and "--" after program name to identify program arguments
    set (PROFILER_TARGET "darwin-x86")
    set (PROFILE_TARGET_ID "linux_x86_gcc")
    set (SPIRAL_OS_NAME "Linux64")
    set (SPIRAL_COMPILER_NAME "GnuC")
    set (SPIRAL_DEBUGGER_NAME "lldb --arch x86_64")
    set (SPIRAL_CMD_ARGS_FLAG "--")

elseif (${INTEL_WIN} OR ${INTEL_LINUX})
    ##  Linux ...
    ##  Use GDB as debugger, --args ==> arguments after program name are for debugged process
    set (PROFILER_TARGET "linux-x86")
    set (PROFILE_TARGET_ID "linux_x86_gcc")
    set (SPIRAL_OS_NAME "Linux64")
    set (SPIRAL_COMPILER_NAME "GnuC")
    set (SPIRAL_DEBUGGER_NAME "gdb --args")

endif ()
