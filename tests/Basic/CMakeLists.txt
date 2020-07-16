##
## SPIRAL License
##
## Copyright (c) 2018, Carnegie Mellon University
## All rights reserved.
## 
## See LICENSE file for full information
##

cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

include ("${SPIRAL_SOURCE_DIR}/config/CMakeIncludes/TestDefines.cmake")

##  Define the Basic tests / simple tests, for a smoke-test approach...
set (TESTS_ALL_TESTS
	basic_test
	break-me
	integers_test
	Memmgr-test
	profiler-test
    )

##  call function my_add_test_target() to add a target for each test
foreach (test ${TESTS_ALL_TESTS})
    my_add_test_target(${test} "Basic")
endforeach()

set_tests_properties (${TESTS_ALL_TESTS} PROPERTIES LABELS "Basic")

enable_testing()
