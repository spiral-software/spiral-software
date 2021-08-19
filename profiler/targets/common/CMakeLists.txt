##
## Copyright (c) 2018-2021, Carnegie Mellon University
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
##  SUFFIX:  { c | cu }

set ( _add_cuda )
if ( ${SUFFIX} STREQUAL "cu" )
    set ( _add_cuda CUDA )
endif ()

project ( SPIRAL_CUDA
         VERSION 1.0.1
                DESCRIPTION "Profile SPIRAL/Cuda code, matrix/vector/time test"
                LANGUAGES C CXX ${_add_cuda} )

set ( PROF_TARGETS ${CMAKE_SOURCE_DIR}/../../targets )
set ( PROF_COMMON  ${PROF_TARGETS}/common )
set ( PROF_INCLUDE ${PROF_TARGETS}/include )
set ( SOURCES      ${PROF_COMMON}/${PROJECT}.${SUFFIX} testcode.${SUFFIX} )

set_source_files_properties ( ${PROF_COMMON}/${PROJECT}.${SUFFIX}
                              PROPERTIES COMPILE_DEFINITIONS  "MAINOBJ" )

include_directories ( ${CMAKE_SOURCE_DIR} ${PROF_TARGETS} ${PROF_COMMON} ${PROF_INCLUDE}  )

##  Setup compilation flags

set ( ADDL_COMPILE_FLAGS )
set ( GPU_COMPILE_DEFNS )
set ( LIBS_TO_LINK )
set ( CMAKE_C_STANDARD 11 )
set ( CMAKE_CXX_STANDARD 11 )

if ( WIN32 )
    set ( ADDL_COMPILE_FLAGS -DWIN64 )
    if ( ${SUFFIX} STREQUAL "cu" )
	list ( APPEND ADDL_COMPILE_FLAGS -rdc=false --pre-include testcode.h )
	set ( GPU_COMPILE_DEFNS )			##   -Xptxas -v
	set ( LIBS_TO_LINK cufft )
    else ()
	list ( APPEND ADDL_COMPILE_FLAGS /FItestcode.h /openmp )
    endif ()
else ()
    set ( ADDL_COMPILE_FLAGS )
    if ( ${SUFFIX} STREQUAL "cu" )
	list ( APPEND ADDL_COMPILE_FLAGS -m64 -rdc=false --pre-include testcode.h )
	set ( GPU_COMPILE_DEFNS )		##   -Xptxas -v
	set ( LIBS_TO_LINK cufft culibos )
    else ()
	list ( APPEND ADDL_COMPILE_FLAGS -include testcode.h -fopenmp )
	if ( NOT "X${EXFLAGS}" STREQUAL "X" )
	    message ( STATUS "Extra flags on command line: ${EXFLAGS}" )
	    list ( APPEND ADDL_COMPILE_FLAGS ${EXFLAGS} )
	endif ()
	if ( NOT "X${EXLIBS}" STREQUAL "X" )
	    message ( STATUS "Extra libraries on command line: ${EXLIBS}" )
	endif ()
	set ( LIBS_TO_LINK gomp pthread m ${EXLIBS} )
    endif ()
endif ()

if ( ${SUFFIX} STREQUAL "cu" )
    ##  set ( CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 )
    set ( CMAKE_CUDA_ARCHITECTURES 70 )
endif ()

add_executable		     ( ${PROJECT} ${SOURCES} )
target_compile_options       ( ${PROJECT} PRIVATE ${ADDL_COMPILE_FLAGS} ${GPU_COMPILE_DEFNS} )

if ( ${SUFFIX} STREQUAL "cu" )
    set_property      ( TARGET ${PROJECT} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON )
endif ()
target_link_libraries    ( ${PROJECT} ${LIBS_TO_LINK} )

install ( TARGETS ${PROJECT} DESTINATION ${CMAKE_SOURCE_DIR} )
