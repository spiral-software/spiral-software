##
## Copyright (c) 2018-2020, Carnegie Mellon University
## All rights reserved.
## 
## See LICENSE file for full information
##

cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

project ( SPIRAL_SUPPORT_UTILS
          DESCRIPTION "Utility tools for running SPIRAL in toolschain"
	  LANGUAGES C CXX CUDA )
      
set ( PROJECT checkforGpu )
set ( SOURCES ${PROJECT}.cu )

include_directories ( ${CMAKE_SOURCE_DIR} ${SPIRAL_SOURCE_DIR}/profiler/targets/include )

##  Setup compilation flags

if (WIN32)
    set ( CUDA_COMPILE_FLAGS -rdc=false )
    set ( GPU_COMPILE_DEFNS )
    set ( LIBS_FOR_CUDA cufft )
    set ( ADDL_COMPILE_FLAGS -DWIN64 )
else ()
    set ( CUDA_COMPILE_FLAGS -m64 -rdc=true )
    set ( GPU_COMPILE_DEFNS -dc )
    set ( LIBS_FOR_CUDA cufft_static culibos )
    set ( ADDL_COMPILE_FLAGS )
endif ()

##  set ( CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 )
set ( CMAKE_CUDA_ARCHITECTURES 70 )
set ( GPU_EXTRAS _CUDAGEN )

add_executable		   ( ${PROJECT} ${SOURCES} )
target_compile_definitions ( ${PROJECT} PRIVATE ${GPU_EXTRAS} )
target_compile_definitions ( ${PROJECT} PRIVATE ${ADDL_COMPILE_FLAGS} )
target_compile_options     ( ${PROJECT} PRIVATE ${CUDA_COMPILE_FLAGS} ${GPU_COMPILE_DEFNS} )
set_property        ( TARGET ${PROJECT} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON )
target_link_libraries	   ( ${PROJECT} ${LIBS_FOR_CUDA} )

set     ( INSTALL_DIR_LOCATION ${SPIRAL_SOURCE_DIR}/gap/bin )
install ( TARGETS ${PROJECT} DESTINATION ${INSTALL_DIR_LOCATION} )

##  Python script to run tests -- allow any package to access test from ${SPIRAL_SOURCE_DIR}/gap/bin

set     ( PYTHON_COMMANDS exectest.py catfiles.py )
install ( FILES ${PYTHON_COMMANDS} DESTINATION ${INSTALL_DIR_LOCATION} )
