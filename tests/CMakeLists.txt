##
## SPIRAL License
##
## Copyright (c) 2018, Carnegie Mellon University
## All rights reserved.
## 
## See LICENSE file for full information
##

cmake_minimum_required (VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

project (TESTS)

##  Tests source directory
set (TESTS_SOURCE_DIR ${SPIRAL_SOURCE_DIR}/tests)

##  Define flag settings to govern whether certain tests will be configured...
##  By default, all tests are enabled; however, the cmake command line can
##  chooose to optionally disable framework generation for certain tests...
##  Additonally, some tests can only work on certain processor architectures
##  (e.g., AVX/SIMD instruction sets); these will only be generated when the
##  appropriate architectire is found.

string (COMPARE NOTEQUAL "${TESTS_RUN_BASIC}" "OFF" PERFORM_TESTS)
if (${PERFORM_TESTS})
    add_subdirectory (Basic)
endif ()

string (COMPARE NOTEQUAL "${TESTS_RUN_SCALAR_TRANSFORMS}" "OFF" PERFORM_TESTS)
if (${PERFORM_TESTS})
    add_subdirectory (Scalar-Transforms)
endif ()

string (COMPARE EQUAL "${CMAKE_HOST_SYSTEM_PROCESSOR}" "AMD64" INTEL_WIN)
string (COMPARE EQUAL "${CMAKE_HOST_SYSTEM_PROCESSOR}" "x86_64" INTEL_LINUX)
if (${INTEL_WIN} OR ${INTEL_LINUX})
    ##  Intel architecture, include the AVX/SIMD tests
    string (COMPARE NOTEQUAL "${TESTS_RUN_X86_SIMD}" "OFF" PERFORM_TESTS)
    if (${PERFORM_TESTS})
	add_subdirectory (X86-SIMD)
    endif ()
endif ()

find_package (OpenMP)
if (${OpenMP_FOUND})
    message (STATUS "OpenMP found, flag = ${OpenMP_FOUND}")
    ##  OpenMP support found
    string (COMPARE EQUAL "${CMAKE_HOST_SYSTEM_PROCESSOR}" "armv7l" ARM_LINUX)
    if (NOT ${ARM_LINUX})
	##  Don't generate for ARM, no SIMD support
	string (COMPARE NOTEQUAL "${TESTS_RUN_OPENMP}" "OFF" PERFORM_TESTS)
	if (${PERFORM_TESTS})
	    add_subdirectory (OpenMP)
	endif ()
    endif ()
endif ()

string (COMPARE NOTEQUAL "${TESTS_RUN_ADVANCED}" "OFF" PERFORM_TESTS)
if (${PERFORM_TESTS})
    add_subdirectory (Advanced)
endif ()

enable_testing ()

