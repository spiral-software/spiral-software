##
## SPIRAL License
##
## Copyright (c) 2018-2021, Carnegie Mellon University
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

##  Look for python3, and if found verify if major/minor version.  Don't add any
##  tests (except Basic above) if we can't find python3 or it's major/minor
##  version is less than our required minium

set (Python3_MIN_REQUIRED_VERSION "3.6")
set (_p3_good FALSE)

find_package (Python3 COMPONENTS Interpreter)
if (${Python3_FOUND})
    ##  compare version ...
    string (COMPARE LESS ${Python3_VERSION} ${Python3_MIN_REQUIRED_VERSION} _p3_old)
    if (${_p3_old})
	##  Python3 version found is less than minimum required -- too old
	message (STATUS "Found Python3: Version = ${Python3_VERSION}, Less than required version ${Python3_MIN_REQUIRED_VERSION}, do basic tests ONLY")
    else ()
	message (STATUS "Found Python3: Version = ${Python3_VERSION}")
	set (_p3_good TRUE)
    endif ()
else ()
    message (STATUS "Python3 NOT FOUND: do basic tests ONLY")
endif ()

string (COMPARE NOTEQUAL "${TESTS_RUN_SCALAR_TRANSFORMS}" "OFF" PERFORM_TESTS)
if (${PERFORM_TESTS} AND ${_p3_good})
    add_subdirectory (Scalar-Transforms)
endif ()

string (COMPARE EQUAL "${CMAKE_HOST_SYSTEM_PROCESSOR}" "AMD64" INTEL_WIN)
string (COMPARE EQUAL "${CMAKE_HOST_SYSTEM_PROCESSOR}" "x86_64" INTEL_LINUX)
if (${INTEL_WIN} OR ${INTEL_LINUX})
    ##  Intel architecture, include the AVX/SIMD tests
    string (COMPARE NOTEQUAL "${TESTS_RUN_X86_SIMD}" "OFF" PERFORM_TESTS)
    if (${PERFORM_TESTS} AND ${_p3_good})
	add_subdirectory (X86-SIMD)
    endif ()
endif ()

find_package (OpenMP)
if (${OpenMP_FOUND})
    message (STATUS "OpenMP found, flag = ${OpenMP_FOUND}")
    ##  OpenMP support found
##    string (COMPARE EQUAL "${CMAKE_HOST_SYSTEM_PROCESSOR}" "armv7l" ARM_LINUX)
##    if (NOT ${ARM_LINUX})
    if (${INTEL_WIN} OR ${INTEL_LINUX})
	##  Only run OpenMP when on Intel (getOpts limitations)
	string (COMPARE NOTEQUAL "${TESTS_RUN_OPENMP}" "OFF" PERFORM_TESTS)
	if (${PERFORM_TESTS} AND ${_p3_good})
	    add_subdirectory (OpenMP)
	endif ()
    endif ()
endif ()

string (COMPARE NOTEQUAL "${TESTS_RUN_ADVANCED}" "OFF" PERFORM_TESTS)
if (${PERFORM_TESTS} AND ${_p3_good})
    add_subdirectory (Advanced)
endif ()

list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")

enable_testing ()

