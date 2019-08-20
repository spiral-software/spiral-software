##
## Copyright (c) 2018-2019, Carnegie Mellon University
## All rights reserved.
## 
## See LICENSE file for full information
##

# specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required(VERSION 3.8...3.14)

set(CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel")

project(SPIRAL
        VERSION 8.1.0
		DESCRIPTION "SPIRAL Project"
		LANGUAGES C CXX)

##  Prevent building directly into the source tree
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" SPIRAL_COMPILE_INPLACE)
if (SPIRAL_COMPILE_INPLACE)
    if (WIN32)
	message (FATAL_ERROR, "Building ${PROJECT_NAME} with CMake requires an out-of-source tree.  To proceed:
		 rm -rf CMakeCache.txt CMakeFiles/   # delete files in ${CMAKE_SOURCE_DIR}
		 mkdir <build>
		 cd <build>
		 cmake ..
		 then build the solution with VS")
    else()
	message (FATAL_ERROR, "Building ${PROJECT_NAME} with CMake requires an out-of-source tree.  To proceed:
		 rm -rf CMakeCache.txt CMakeFiles/   # delete files in ${CMAKE_SOURCE_DIR}
		 mkdir <build>
		 cd <build>
		 cmake ..
		 make | make install")
    endif()
endif()
  
##  Some .h in sys_conf folder; reference root dir as include dir for "sys_conf/xxx.h"
##  set(SPIRAL_INC ${SPIRAL_SOURCE_DIR})	# should not need when no sys_conf library
set(SPIRAL_CONFIG_DIR ${SPIRAL_SOURCE_DIR}/config)
set(SPIRAL_GEN_INC ${SPIRAL_BINARY_DIR}/include/spiral)

##  set(SYSCONF_LIB sys_conf)				# remove sys_conf library
##  message(STATUS "SYSCONF_LIB = ${SYSCONF_LIB}")

##  set(CMAKE_C_STANDARD 99)

include_directories(${SPIRAL_GEN_INC})

# Check for the git commit hash, if using a git repo
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    find_package(Git)
    if (Git_FOUND)
        execute_process(COMMAND ${GIT_EXECUTABLE} log --pretty=format:%H -n 1
                        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                        OUTPUT_VARIABLE   SPIRAL_GIT_HASH)
    endif()
endif()
if (NOT SPIRAL_GIT_HASH)
   set(SPIRAL_GIT_HASH "Not found")
endif()

if ((NOT DEFINED CMAKE_BUILD_TYPE) OR (NOT CMAKE_BUILD_TYPE))
    set(CMAKE_BUILD_TYPE Release)
endif()

configure_file (
    ${SPIRAL_CONFIG_DIR}/spiral_build_info.h.in
    ${SPIRAL_GEN_INC}/spiral_build_info.h
)

# Determine the system's endian type

include (TestBigEndian)
TEST_BIG_ENDIAN(ENDIAN_TYPE)
message(STATUS "My endian type = ${ENDIAN_TYPE}")
if (ENDIAN_TYPE)
    set(ENDIANNESS "BIG_ENDIAN")
else()
    set(ENDIANNESS "LITTLE_ENDIAN")
endif()
message(STATUS "Set ENDIANNESS = ${ENDIANNESS}")
configure_file (
    ${SPIRAL_SOURCE_DIR}/gap/src/machine_endian.h.in
    ${SPIRAL_SOURCE_DIR}/gap/src/machine_endian.h
)

add_subdirectory(gap)

##  enable_testing()

