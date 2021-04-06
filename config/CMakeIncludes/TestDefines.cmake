##
## SPIRAL License
##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
## 
## See LICENSE file for full information
##

cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

##  include(FindUnixCommands)

##  GAP shell script name
if (WIN32)
    file (TO_NATIVE_PATH ${SPIRAL_SOURCE_DIR}/spiral.bat TEST_GAP_EXEC_NAME)
else ()
    include(FindUnixCommands)
    file (TO_NATIVE_PATH ${SPIRAL_SOURCE_DIR}/spiral TEST_GAP_EXEC_NAME )
endif ()

##  Define a function to add a test target, given a test name.  The assumption
##  is that all targets are executed by invoking Spiral, redirecting input
##  from a test file and waiting until its done.  The test script source file
##  is assumed to be in the [current] subdirectory (parameter subdir) and has
##  the same name as the test, appended with ".g".

function (my_add_test_target testname subdir)
    file (TO_NATIVE_PATH ${TESTS_SOURCE_DIR}/${subdir}/${testname}.g _scrpt)
    if (BASH)
        add_test (NAME ${testname}
          COMMAND ${BASH} -c "${TEST_GAP_EXEC_NAME} < ${_scrpt}"
        )
        SET_TESTS_PROPERTIES(${testname} PROPERTIES SKIP_RETURN_CODE 86)
    else ()
        if (WIN32)
            add_test (NAME ${testname} 
              COMMAND ${TEST_GAP_EXEC_NAME} < ${_scrpt}
            )
            SET_TESTS_PROPERTIES(${testname} PROPERTIES FAIL_REGULAR_EXPRESSION "TEST FAILED")
            SET_TESTS_PROPERTIES(${testname} PROPERTIES SKIP_REGULAR_EXPRESSION "Skipping test")
            SET_TESTS_PROPERTIES(${testname} PROPERTIES SKIP_RETURN_CODE 86)
        else ()
            message(FATAL_ERROR "Unknown shell command for ${CMAKE_HOST_SYSTEM_NAME}")
        endif ()
    endif  ()
    
endfunction ()

