##
## SPIRAL License
##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
## 
## See LICENSE file for full information
##

cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

include ("${SPIRAL_SOURCE_DIR}/config/CMakeIncludes/TestDefines.cmake")

##  Define the Open MP example tests...
set (TESTS_ALL_TESTS
	OpenMP
    )

##  call function my_add_test_target() to add a target for each test
foreach (test ${TESTS_ALL_TESTS})
    my_add_test_target(${test} "OpenMP")
endforeach()

set_tests_properties (${TESTS_ALL_TESTS} PROPERTIES LABELS "OpenMP")

enable_testing()
