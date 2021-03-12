##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

## specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required ( VERSION 3.14 )

set ( CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel" )

project ( SPIRAL_CUDA
         VERSION 1.0.1
                DESCRIPTION "Profile SPIRAL/Cuda code, matrix/vector/time test"
                LANGUAGES C CXX CUDA )

##  PROJECT is undefined, it must be specified on the command line to cmake and
##  should be one of: { matrix | time | cvector }
##  set ( PROJECT )

set ( PROF_TARGETS ${CMAKE_SOURCE_DIR}/../../targets )
set ( PROF_COMMON  ${PROF_TARGETS}/common )
set ( PROF_INCLUDE ${PROF_TARGETS}/include )
set ( SOURCES      ${PROF_COMMON}/${PROJECT}.cu testcode.cu )

include_directories ( ${CMAKE_SOURCE_DIR} ${PROF_TARGETS} ${PROF_COMMON} ${PROF_INCLUDE}  )

##  Setup compilation flags

if ( WIN32 )
    set ( CUDA_COMPILE_FLAGS -DWIN64 -rdc=false --pre-include testcode.h )
    set ( GPU_COMPILE_DEFNS )			##   -Xptxas -v
    set ( LIBS_FOR_CUDA cufft )
else ()
    set ( CUDA_COMPILE_FLAGS -m64 -rdc=true --pre-include testcode.h )
    set ( GPU_COMPILE_DEFNS -dc )		##   -Xptxas -v
    set ( LIBS_FOR_CUDA cufft_static culibos )
endif ()

##  set ( CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 )
set ( CMAKE_CUDA_ARCHITECTURES 70 )
set ( GPU_EXTRAS _CUDAGEN )

add_executable		   ( ${PROJECT} ${SOURCES} )
target_compile_definitions ( ${PROJECT} PRIVATE ${GPU_EXTRAS} )
target_compile_options     ( ${PROJECT} PRIVATE ${CUDA_COMPILE_FLAGS} ${GPU_COMPILE_DEFNS} )
set_property        ( TARGET ${PROJECT} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON )
target_link_libraries	   ( ${PROJECT} ${LIBS_FOR_CUDA} )

install ( TARGETS ${PROJECT} DESTINATION ${CMAKE_SOURCE_DIR} )
