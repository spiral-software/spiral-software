##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
## 
## See LICENSE file for full information
##

##  Get info for CUDA paths, defines, and compile flags, set:
##    SPIRAL_CUDA_VERSION
##    SPIRAL_CUDA_INCLUDE_DIRS
##    SPIRAL_CUDA_TOOLKIT_ROOT_DIR
##    SPIRAL_CUDA_LIBRARIES

##  Check the cmake version running

function ( setup_cuda_variables_for_spiral )

set ( _version "${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" )  ## Actual running cmake version
set ( _vers_cuda_toolkit "3.17" )              ##  Version when FindCUDAToolkit is available

string ( COMPARE LESS ${_version} ${_vers_cuda_toolkit} _cuda_old )
if ( ${_cuda_old} )
    ##  FindCUDAToolkit not available -- cmake too old -- use FindCUDA instead
    message ( STATUS "Cmake version = ${_version}, using FindCUDA module" )
    find_package ( CUDA )
    if ( ${CUDA_FOUND} )
	message ( STATUS "CUDA Found : Version = ${CUDA_VERSION}" )
	message ( STATUS "CUDA include dirs = ${CUDA_INCLUDE_DIRS}" )
	message ( STATUS "CUDA Toolkit Root Dir = ${CUDA_TOOLKIT_ROOT_DIR}" )
	message ( STATUS "CUDA libraries = ${CUDA_LIBRARIES}" )
	set ( SPIRAL_CUDA_VERSION ${CUDA_VERSION} PARENT_SCOPE )
	set ( SPIRAL_CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} PARENT_SCOPE )
	set ( SPIRAL_CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR} PARENT_SCOPE )
	set ( SPIRAL_CUDA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64 PARENT_SCOPE )
    else ()
	message ( STATUS "CUDA not found -- shouldn't get here" )
    endif ()
else ()
    ##  Cmake is >= version 3.17, FindCUDAToolkit module is available
    message ( STATUS "Cmake version = ${_version}, using FindCUDAToolkit module" )
    find_package ( CUDAToolkit )
    if ( ${CUDAToolkit_FOUND} )
	message ( STATUS "CUDA Toolkit Found : Version = ${CUDAToolkit_VERSION}" )
	message ( STATUS "Enabled CUDA, rt library = ${CUDA::cudart}" )
	message ( STATUS "CUDA include dirs = ${CUDAToolkit_INCLUDE_DIRS}" )
	message ( STATUS "CUDA library dir = ${CUDAToolkit_LIBRARY_DIR}" )

	set ( SPIRAL_CUDA_VERSION ${CUDAToolkit_VERSION} PARENT_SCOPE )
	set ( SPIRAL_CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS} PARENT_SCOPE )
	set ( SPIRAL_CUDA_TOOLKIT_ROOT_DIR ${CUDAToolkit_LIBRARY_DIR}/.. PARENT_SCOPE )
	set ( SPIRAL_CUDA_LIBRARIES ${CUDAToolkit_LIBRARY_DIR} PARENT_SCOPE )
    else ()
	message ( STATUS "CUDA Toolkit NOT found -- shouldn't get here" )
    endif ()
endif ()

endfunction ()
