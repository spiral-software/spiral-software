##
## Copyright (c) 2018-2020, Carnegie Mellon University
## All rights reserved.
## 
## See LICENSE file for full information
##

# specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required(VERSION 3.14)

set(CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel")

project(SPIRAL
        VERSION 8.2.0
		DESCRIPTION "SPIRAL Project"
		LANGUAGES C CXX)
		
set(PROJECT_VERSION_TAG "")

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
  
set(SPIRAL_CONFIG_DIR ${SPIRAL_SOURCE_DIR}/config)
set(SPIRAL_GEN_INC ${SPIRAL_BINARY_DIR}/include/spiral)

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

include ("${SPIRAL_SOURCE_DIR}/config/CMakeIncludes/ScriptConfig.cmake")
if (WIN32)
    configure_file (
	${SPIRAL_CONFIG_DIR}/spiral.bat.in
	${SPIRAL_BINARY_DIR}/gap/spiral.bat
    )
    configure_file (
	${SPIRAL_CONFIG_DIR}/spiral_debug.bat.in
	${SPIRAL_BINARY_DIR}/gap/spiral_debug.bat
    )
    configure_file (
	${SPIRAL_CONFIG_DIR}/_spiral_win.g.in
	${SPIRAL_BINARY_DIR}/gap/_spiral_win.g
    )
else ()
    configure_file (
	${SPIRAL_CONFIG_DIR}/spiral.in
	${SPIRAL_BINARY_DIR}/gap/spiral
    )
    configure_file (
	${SPIRAL_CONFIG_DIR}/spirald.in
	${SPIRAL_BINARY_DIR}/gap/spirald
    )
    configure_file (
	${SPIRAL_CONFIG_DIR}/_spiral.g.in
	${SPIRAL_BINARY_DIR}/gap/_spiral.g
    )
endif ()

##  Rule to install the script files when install target is built

if (WIN32)
    set (SPIRAL_STARTUP_FILES
	    ${SPIRAL_BINARY_DIR}/gap/spiral.bat
	    ${SPIRAL_BINARY_DIR}/gap/spiral_debug.bat
	    ${SPIRAL_BINARY_DIR}/gap/_spiral_win.g
	)
else ()
    set (SPIRAL_STARTUP_FILES
	    ${SPIRAL_BINARY_DIR}/gap/spiral
	    ${SPIRAL_BINARY_DIR}/gap/spirald
	    ${SPIRAL_BINARY_DIR}/gap/_spiral.g
	)
endif ()

install (FILES ${SPIRAL_STARTUP_FILES}
	 DESTINATION ${SPIRAL_SOURCE_DIR}
	 PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
		     GROUP_EXECUTE GROUP_READ
		     WORLD_EXECUTE WORLD_READ
	)

add_subdirectory(gap)

add_subdirectory(tests)		## holds test scripts

enable_testing()

