##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

##  GAP shell script name
if (WIN32)
    file (TO_NATIVE_PATH ${SPIRAL_SOURCE_DIR}/spiral.bat TEST_GAP_EXEC_NAME )
else ()
    include(FindUnixCommands)
    file (TO_NATIVE_PATH ${SPIRAL_SOURCE_DIR}/spiral TEST_GAP_EXEC_NAME )
endif ()

##  Define a function to add a test target, given a test name, a current
##  directory and a set of one or more input files.  The input files are handled
##  as optional arguments, but there must be at least 1.

function (RunTestTarget _gpu_reqd testname exdir)
    ##  message ("number of arguments sent to function: ${ARGC}")
    ##  message ("all function arguments:               ${ARGV}")
    ##  message ("all arguments beyond defined:         ${ARGN}")

    if ( ${ARGC} GREATER 3 )
	##  received at least one input file
	set ( _cat_fils "" )
	foreach ( _fil ${ARGN} )
	    file (TO_NATIVE_PATH ${exdir}/${_fil} _scrpt)
	    list ( APPEND _cat_fils ${_scrpt} )
	endforeach ()
	##  message ( "Files for SPIRAL: ${_cat_fils}" )
    else ()
	message ( FATAL_ERROR "RunTestTarget requires at least one input script" )
    endif ()
    
    if (BASH)
        add_test (NAME ${testname}
	    ##           COMMAND ${BASH} -c "cat ${_cat_fils} | ${TEST_GAP_EXEC_NAME}"
	    COMMAND ${Python3_EXECUTABLE} ${SPIRAL_SOURCE_DIR}/gap/bin/exectest.py
	            ${_gpu_reqd} ${TEST_GAP_EXEC_NAME} ${_cat_fils}
        )
        SET_TESTS_PROPERTIES(${testname} PROPERTIES SKIP_RETURN_CODE 86)
	SET_TESTS_PROPERTIES(${testname} PROPERTIES SKIP_REGULAR_EXPRESSION "Skipping test")
    else ()
        if (WIN32)
            add_test (NAME ${testname}
		COMMAND ${Python3_EXECUTABLE} ${SPIRAL_SOURCE_DIR}/gap/bin/exectest.py
		        ${_gpu_reqd} ${TEST_GAP_EXEC_NAME} ${_cat_fils}
            )
            SET_TESTS_PROPERTIES(${testname} PROPERTIES FAIL_REGULAR_EXPRESSION "TEST FAILED")
            SET_TESTS_PROPERTIES(${testname} PROPERTIES SKIP_REGULAR_EXPRESSION "Skipping test")
            SET_TESTS_PROPERTIES(${testname} PROPERTIES SKIP_RETURN_CODE 86)
        else ()
            message(FATAL_ERROR "Unknown shell command for ${CMAKE_HOST_SYSTEM_NAME}")
        endif ()
    endif  ()
endfunction ()
