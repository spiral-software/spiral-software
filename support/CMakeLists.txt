##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
## 
## See LICENSE file for full information
##

cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

##  Check the code generation ... look for CUDA toolkit

set ( _version "${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" )  ## Actual running cmake version
set ( _vers_cuda_toolkit "3.17" )              ##  Version when FindCUDAToolkit is available

string ( COMPARE LESS ${_version} ${_vers_cuda_toolkit} _cuda_old )
if ( ${_cuda_old} )
    ##  FindCUDAToolkit not available -- cmake too old -- use FindCUDA instead
    message ( STATUS "Cmake version = ${_version}, using FindCUDA module" )
    find_package ( CUDA )
    if ( ${CUDA_FOUND} )
	message ( STATUS "CUDA Found : Version = ${CUDA_VERSION}" )
	set ( _codegen GPU )
    else ()
	set ( _codegen CPU )
    endif ()
else ()
    ##  Cmake is >= version 3.17, FindCUDAToolkit module is available
    message ( STATUS "Cmake version = ${_version}, using FindCUDAToolkit module" )
    find_package ( CUDAToolkit )
    if ( ${CUDAToolkit_FOUND} )
	message ( STATUS "CUDA Toolkit Found : Version = ${CUDAToolkit_VERSION}" )
	set ( _codegen GPU )
    else ()
	set ( _codegen CPU )
    endif ()
endif ()

if ( ${_codegen} STREQUAL "GPU" )
    set ( _lang_add CUDA )
    set ( _suffix cu )
else ()
    message ( STATUS "CUDA toolkit NOT FOUND: No GPU code / tests will be performed" )
    set ( _suffix c )
endif ()

project ( SPIRAL_SUPPORT_UTILS
          DESCRIPTION "Utility tools for running SPIRAL in toolschain"
	  LANGUAGES C CXX ${_lang_add} )
      
set ( PROJECT checkforGpu )
set ( SOURCES ${PROJECT}.${_suffix} )

include_directories ( ${CMAKE_SOURCE_DIR} ${SPIRAL_SOURCE_DIR}/profiler/targets/include )

##  Setup compilation flags

if ( ${_codegen} STREQUAL "GPU" )
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
endif ()

add_executable		   ( ${PROJECT} ${SOURCES} )

target_compile_definitions ( ${PROJECT} PRIVATE ${ADDL_COMPILE_FLAGS} )
if ( ${_codegen} STREQUAL "GPU" )
    target_compile_definitions ( ${PROJECT} PRIVATE ${GPU_EXTRAS} )
    target_compile_options     ( ${PROJECT} PRIVATE ${CUDA_COMPILE_FLAGS} ${GPU_COMPILE_DEFNS} )
    set_property        ( TARGET ${PROJECT} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON )
    target_link_libraries	   ( ${PROJECT} ${LIBS_FOR_CUDA} )
endif ()

set     ( INSTALL_DIR_LOCATION ${SPIRAL_SOURCE_DIR}/gap/bin )
install ( TARGETS ${PROJECT} DESTINATION ${INSTALL_DIR_LOCATION} )

##  Python script to run tests -- allow any package to access test from ${SPIRAL_SOURCE_DIR}/gap/bin

set     ( PYTHON_COMMANDS exectest.py catfiles.py )
install ( FILES ${PYTHON_COMMANDS} DESTINATION ${INSTALL_DIR_LOCATION} )
