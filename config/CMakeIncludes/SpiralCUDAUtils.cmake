##
##  SPIRAL License
##
##  Copyright (c) 2026, Carnegie Mellon University
##  All rights reserved.
## 
##  See LICENSE file for full information
##

cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )

##  Unified dynamic CUDA architecture policy

macro ( spiral_determine_cuda_architectures )
    if ( NOT DEFINED CMAKE_CUDA_ARCHITECTURES )
        message ( STATUS "SPIRAL: Initializing dynamic CUDA architecture detection..." )
        
        ##  1. Attempt to query for a native local GPU using nvidia-smi
        find_program( NVIDIA_SMI "nvidia-smi" )
        if ( NVIDIA_SMI )
            execute_process (
                COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader,nounits 
                OUTPUT_VARIABLE GPU_COMPUTE_CAP 
                OUTPUT_STRIP_TRAILING_WHITESPACE 
                ERROR_QUIET 
            )
        endif ()

        ##  2. Evaluate the result of the hardware probe
        if ( GPU_COMPUTE_CAP )
            ##  Strip any decimal point, make 7.5 -> 75
            string ( REGEX REPLACE "\\." "" TARGET_CAP ${GPU_COMPUTE_CAP} )
            message ( STATUS "SPIRAL: Found local GPU (sm_${TARGET_CAP}). Optimizing for native target." )
            set ( CMAKE_CUDA_ARCHITECTURES "${TARGET_CAP}" CACHE STRING "CUDA architectures" )
        else ()
            ##  3. Version-Safe Headless Fallback (HPC Login Node or WSL Environments)
            message ( STATUS "SPIRAL: No local GPU detected. Evaluating version-safe defaults for compiler: ${CMAKE_CUDA_COMPILER_VERSION}" )
            set ( FALLBACK_ARCHS "80" ) # Ampere baseline safe across modern CUDA lines
            ##  Hopper (sm_90) requires CUDA 11.8+
            if ( CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.8" )
                list ( APPEND FALLBACK_ARCHS "90" )
            endif ()
            ##  Blackwell (sm_120) requires CUDA 12.8 / 13.0+
            if ( CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8" )
                list ( APPEND FALLBACK_ARCHS "120" )
            endif ()

            set ( CMAKE_CUDA_ARCHITECTURES "${FALLBACK_ARCHS}" CACHE STRING "CUDA architectures" )
            message ( STATUS "SPIRAL: Selected safe headless target(s): ${CMAKE_CUDA_ARCHITECTURES}" )
        endif ()
    else ()
        message ( STATUS "SPIRAL: Using user-specified CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}" )
    endif ()
endmacro()
