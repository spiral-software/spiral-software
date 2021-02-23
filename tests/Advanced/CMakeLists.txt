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

##  Define the Advanced example tests...
set (TESTS_ALL_TESTS
        Hockney
        Planewave-3D-FFT
        WarpX-1D
        DFT_PD_Stage1
    )

##  call function my_add_test_target() to add a target for each test
foreach (test ${TESTS_ALL_TESTS})
    my_add_test_target(${test} "Advanced")
endforeach()

set_tests_properties (${TESTS_ALL_TESTS} PROPERTIES LABELS "Advanced")

enable_testing()
