##
## Copyright (c) 2018-2023, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

## specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required ( VERSION 3.14 )

set ( CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel" )

##  PROJECT & SUFFIX are undefined, they must be specified on the command line
##  to cmake and should be one of (time is either rdtsc_time or papi_time
##  depending on system architecture):
##  PROJECT: { matrix | rdtsc_time | papi_time | cvector }
##  SUFFIX:  { c | cu | cpp }

set ( ADDL_COMPILE_FLAGS )
set ( GPU_COMPILE_DEFNS )
set ( LIBS_TO_LINK )

if  ( ${SUFFIX} STREQUAL "cu" )
    project ( ${PROJECT}_CUDA
        VERSION 1.0.1
        DESCRIPTION "Profile SPIRAL/Cuda code, matrix/vector/time test"
        LANGUAGES C CXX CUDA )
elseif ( ${SUFFIX} STREQUAL "cpp" )
    project ( ${PROJECT}_HIP
        VERSION 1.0.1
        DESCRIPTION "Profile SPIRAL code, matrix/vector/time test"
        LANGUAGES C CXX )
    if ( "${CMAKE_CXX_COMPILER}" MATCHES "hipcc$" )
        message ( STATUS "${CMAKE_CXX_COMPILER} matches 'hipcc$' " )
        ##  Setup what we need to build for HIP/ROCm
        list ( APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm )
        find_package ( hip REQUIRED )
        if ( ${hip_FOUND} )
	    ##  HIP/ROCm support found
	    message ( STATUS "Found HIP: Version = ${hip_VERSION}" )
        else ()
	    message ( SEND_ERROR "HIP NOT FOUND: HIP is required to build")
        endif ()

        ##  Adjust include and library directories
        ##  Need to add $ROCM_PATH for includes and libraries
        if ( DEFINED ENV{ROCM_PATH} )
	    message ( STATUS "ROCM_PATH is defined: $ENV{ROCM_PATH}" )
	    include_directories ( $ENV{ROCM_PATH}/include/hipfft $ENV{ROCM_PATH}/include )
        endif ()
        list ( APPEND LIBS_TO_LINK hipfft rocfft )
        list ( APPEND ADDL_COMPILE_FLAGS -DFFTX_HIP )
    endif ()
else ()
    project ( ${PROJECT}
        VERSION 1.0.1
        DESCRIPTION "Profile SPIRAL code, matrix/vector/time test"
        LANGUAGES C CXX )
endif ()				


set ( PROF_TARGETS ${CMAKE_SOURCE_DIR}/../../targets )
set ( PROF_COMMON  ${PROF_TARGETS}/common )
set ( PROF_INCLUDE ${PROF_TARGETS}/include )
set ( SOURCES      ${PROF_COMMON}/${PROJECT}.cu testcode.${SUFFIX} )

set_source_files_properties ( ${PROF_COMMON}/${PROJECT}.cu
                              PROPERTIES COMPILE_DEFINITIONS  "MAINOBJ" )

include_directories ( ${CMAKE_SOURCE_DIR} ${PROF_TARGETS} ${PROF_COMMON} ${PROF_INCLUDE}  )

##  Setup compilation flags

set ( CMAKE_C_STANDARD 11 )
set ( CMAKE_CXX_STANDARD 11 )

if ( WIN32 )
    list ( APPEND ADDL_COMPILE_FLAGS -DWIN64 )
    if ( ${SUFFIX} STREQUAL "cu" )
	list ( APPEND ADDL_COMPILE_FLAGS -rdc=false --pre-include testcode.h )
	list ( APPEND GPU_COMPILE_DEFNS )			##   -Xptxas -v
	list ( APPEND LIBS_TO_LINK cufft )
    else ()
	list ( APPEND ADDL_COMPILE_FLAGS /FItestcode.h /openmp )
    endif ()
else ()
    if ( ${SUFFIX} STREQUAL "cu" )
	list ( APPEND ADDL_COMPILE_FLAGS -m64 -rdc=false --pre-include testcode.h )
	list ( APPEND GPU_COMPILE_DEFNS )		##   -Xptxas -v
	list ( APPEND LIBS_TO_LINK cufft culibos )
    elseif ( ${SUFFIX} STREQUAL "cpp" )
        set_source_files_properties ( ${PROF_COMMON}/${PROJECT}.cu PROPERTIES LANGUAGE CXX )
	list ( APPEND ADDL_COMPILE_FLAGS -include testcode.h )
    else ()
	list ( APPEND ADDL_COMPILE_FLAGS -include testcode.h )
	message ( STATUS "ADDL_COMPILE_FLAGS = ${ADDL_COMPILE_FLAGS}" )
	message ( STATUS "CMAKE_HOST_APPLE = ${CMAKE_HOST_APPLE}" )
	if ( ${CMAKE_HOST_APPLE} )
	    message ( STATUS "Host system is Apple / MAC, no OpenMP" )
	else ()
	    list ( APPEND ADDL_COMPILE_FLAGS -fopenmp )
	    set ( LIBS_TO_LINK gomp pthread )
	    message ( STATUS "ADDL_COMPILE_FLAGS = ${ADDL_COMPILE_FLAGS}" )
	    message ( STATUS "LIBS_TO_LINK = ${LIBS_TO_LINK}" )
	endif ()
	if ( NOT "X${EXFLAGS}" STREQUAL "X" )
	    message ( STATUS "Extra flags on command line: ${EXFLAGS}" )
	    list ( APPEND ADDL_COMPILE_FLAGS ${EXFLAGS} )
	endif ()
	if ( NOT "X${EXLIBS}" STREQUAL "X" )
	    message ( STATUS "Extra libraries on command line: ${EXLIBS}" )
	endif ()
	list ( APPEND LIBS_TO_LINK m ${EXLIBS} )
	message ( STATUS "LIBS_TO_LINK = ${LIBS_TO_LINK}" )
    endif ()
endif ()

if ( ${SUFFIX} STREQUAL "cu" )
    set ( CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80)
endif ()

add_executable		     ( ${PROJECT} ${SOURCES} )
target_compile_options       ( ${PROJECT} PRIVATE ${ADDL_COMPILE_FLAGS} ${GPU_COMPILE_DEFNS} )

if ( ${SUFFIX} STREQUAL "cu" )
    set_property      ( TARGET ${PROJECT} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON )
endif ()
target_link_libraries    ( ${PROJECT} ${LIBS_TO_LINK} )

install ( TARGETS ${PROJECT} DESTINATION ${CMAKE_SOURCE_DIR} )
